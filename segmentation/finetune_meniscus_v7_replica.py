"""
finetune_meniscus_v7_replica.py
===============================
Replication of the v7_mixed / v8_mixed run (reported Dice 0.781), WITH
diagnostic instrumentation. Nothing about how the model trains has changed.

Loss is byte-identical to commit 64b1020 (the SPIE-submission state of
finetune_meniscus_v5_mixed.py) — plain CrossEntropyLoss(weight=...) + SoftDice
on fake PD, MergedLoss(w_bg=0.1, w_men=1.5) on real PD.
NOTE: the current finetune_meniscus_v5_mixed.py is the *v9* restricted-CE
variant (Dice 0.753). Do not use that script to replicate v7/v8.

What was ADDED (observation only — no effect on gradients or optimizer):
  1. train Dice logged           -- previously computed as `td` and discarded
  2. fake-PD vs real-PD Dice     -- separates domain gap from overfitting
  3. clean per-patient val Dice  -- see note below on why this differs from `vd`
  4. val predictions dumped      -- NIfTI (image / GT / pred) for Slicer
  5. fixed, logged seed          -- so a re-run measures seed noise, not drift
  6. history CSV                 -- plot curves without parsing the log

How to read the output
----------------------
  dice_real tr vs va, and `gap`:
      gap large & growing  -> OVERFITTING the 13 real training patients
      both low, gap ~0     -> UNDERFITTING / at ceiling; regularizers can't help
      both high, gap small -> near the achievable ceiling for this label quality
  dice_fake vs dice_real on the same row:
      large drop -> the RegGAN translation is the weak link, not the segmenter.
      Caveat: fake-PD masks are aligned by construction and noise-free, real-PD
      masks are human tracings. Part of any drop is annotation noise, not
      domain gap. Do not attribute it all to RegGAN.

Why `dice_va_perpatient` != `val_dice(real)`
--------------------------------------------
run_epoch() iterates the FAKE loader and pulls real batches from a *cycling*
iterator. The fake set is far larger, so the real val set is traversed many
times per epoch, and slices are counted an uneven number of times. `vd` (which
is what ckpt_best is selected on, unchanged here for fidelity) is that pooled,
repeat-weighted number. `dice_va_perpatient` is a separate clean pass over each
val slice exactly once, grouped by patient. The clean number is the trustworthy
one; both are logged so the difference is visible.
"""
import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v2     import Meniscus2_5DDataset
from dataset_2_5d_realpd import RealPDDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES            = 3
PRETRAINED_N_CLASSES = 5

# Real PD after preprocessing is a spatial no-op (native 384x384, already
# isotropic in-plane) -- see PIPELINE_VALIDATION_PLAN.md Stage 5c.
PD_SPACING = (3.6, 0.39, 0.39)   # (through-plane, in-plane, in-plane)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    log.info(f"Seed fixed: {seed} (cudnn.deterministic=True)")


def build_model(ckpt_path, device):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=PRETRAINED_N_CLASSES)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    log.info(f"Loaded DESS baseline from {ckpt_path}")

    old_head = model.segmentation_head[0]
    model.segmentation_head[0] = nn.Conv2d(
        old_head.in_channels, N_CLASSES,
        kernel_size=old_head.kernel_size,
        stride=old_head.stride,
        padding=old_head.padding,
    )
    log.info(f"Replaced head: {PRETRAINED_N_CLASSES} → {N_CLASSES} classes")
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Losses — UNCHANGED from 64b1020. Do not edit; this is the replication target.
# ─────────────────────────────────────────────────────────────────────────────

class MergedLoss(nn.Module):
    def __init__(self, w_bg=0.1, w_men=1.5, eps=1e-6):
        super().__init__()
        self.w_bg, self.w_men, self.eps = w_bg, w_men, eps

    def forward(self, logits, binary_masks):
        probs  = torch.softmax(logits, dim=1)
        p_bg   = probs[:, 0]
        p_men  = probs[:, 1] + probs[:, 2]
        gt_men = (binary_masks > 0).float()
        gt_bg  = (binary_masks == 0).float()
        ce   = -(self.w_bg * gt_bg  * torch.log(p_bg.clamp(self.eps)) +
                 self.w_men * gt_men * torch.log(p_men.clamp(self.eps))).mean()
        tp   = (p_men * gt_men).sum()
        dice = 1.0 - (2*tp + self.eps) / (p_men.sum() + gt_men.sum() + self.eps)
        return ce + dice


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=N_CLASSES, eps=1e-6):
        super().__init__(); self.n_classes = n_classes; self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1); loss = 0.0
        for c in range(1, self.n_classes):
            p = probs[:, c]; t = (targets == c).float()
            loss += 1.0 - (2*(p*t).sum() + self.eps) / (p.sum() + t.sum() + self.eps)
        return loss / (self.n_classes - 1)


def dice_binary(preds, targets, eps=1e-6):
    pred_men = (preds > 0).float(); gt_men = (targets > 0).float()
    tp = (pred_men * gt_men).sum()
    return float((2*tp + eps) / (pred_men.sum() + gt_men.sum() + eps))


def run_epoch(model, real_loader, fake_loader, optimizer,
              merged_loss_fn, fake_loss_fn, device, train=True):
    """Identical to the original except that fake-PD Dice is also accumulated.
    model(xf) still runs exactly once per batch; loss and optimizer are untouched."""
    model.train() if train else model.eval()
    total_loss, dice_sum, dice_fake_sum, n = 0.0, 0.0, 0.0, 0
    real_iter = iter(real_loader)

    with torch.set_grad_enabled(train):
        for fake_batch in fake_loader:
            xf, yf = fake_batch["image"].to(device), fake_batch["mask"].to(device)
            logits_f = model(xf)              # hoisted so Dice can reuse it
            loss_f   = fake_loss_fn(logits_f, yf)

            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter  = iter(real_loader)
                real_batch = next(real_iter)
            xr, yr   = real_batch["image"].to(device), real_batch["mask"].to(device)
            logits_r = model(xr)
            loss_r   = merged_loss_fn(logits_r, yr)

            loss = loss_f + loss_r
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_loss    += loss.item()
            dice_sum      += dice_binary(logits_r.argmax(1), yr)
            dice_fake_sum += dice_binary(logits_f.argmax(1), yf)
            n += 1

    return (total_loss    / max(n, 1),
            dice_sum      / max(n, 1),
            dice_fake_sum / max(n, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Instrumentation (read-only)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_real_per_patient(model, dataset, device, batch_size=8):
    """
    Clean pass: every val slice exactly once, grouped by patient.
    Returns {pid: dice}, and {pid: {"tp","fp","fn"}} voxel counts.
    Indexes the dataset directly so each item's stem is known.
    """
    model.eval()
    acc = {}   # pid -> [tp, pred_sum, gt_sum]

    for start in range(0, len(dataset), batch_size):
        idxs   = range(start, min(start + batch_size, len(dataset)))
        items  = [dataset[i] for i in idxs]
        stems  = [dataset.slices[i] for i in idxs]
        x = torch.stack([it["image"] for it in items]).to(device)
        y = torch.stack([it["mask"]  for it in items]).to(device)
        pred = model(x).argmax(1)

        pm, gm = (pred > 0).float(), (y > 0).float()
        for b, stem in enumerate(stems):
            pid = stem.rsplit("_", 1)[0]
            a = acc.setdefault(pid, [0.0, 0.0, 0.0])
            a[0] += float((pm[b] * gm[b]).sum())
            a[1] += float(pm[b].sum())
            a[2] += float(gm[b].sum())

    dices, counts, eps = {}, {}, 1e-6
    for pid, (tp, ps, gs) in acc.items():
        dices[pid]  = (2 * tp + eps) / (ps + gs + eps)
        counts[pid] = {"tp": tp, "fp": ps - tp, "fn": gs - tp}
    return dices, counts


@torch.no_grad()
def dump_val_predictions(model, dataset, device, out_dir, batch_size=8):
    """
    Save image / GT / prediction per val patient as NIfTI, plus a per-slice
    Dice CSV so the worst slices can be found numerically.

    Affine: a plain diagonal affine with real-PD spacing. All three volumes
    share it, so they are exactly co-registered WITH EACH OTHER — which is what
    you need to see where the prediction departs from GT. For overlay on the
    original scan in world coordinates, use infer_real_pd_v3.py, which carries
    the verified LPS->RAS affine logic.
    """
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel not available — skipping val prediction dump")
        return

    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    vols = {}   # pid -> {sidx: (img, gt, pred)}

    for start in range(0, len(dataset), batch_size):
        idxs  = range(start, min(start + batch_size, len(dataset)))
        items = [dataset[i] for i in idxs]
        stems = [dataset.slices[i] for i in idxs]
        x = torch.stack([it["image"] for it in items]).to(device)
        y = torch.stack([it["mask"]  for it in items])
        pred = model(x).argmax(1).cpu()

        for b, stem in enumerate(stems):
            pid, sidx = stem.rsplit("_", 1)
            # channel 1 of the 2.5D stack is the centre slice
            vols.setdefault(pid, {})[int(sidx)] = (
                x[b, 1].cpu().numpy().astype(np.float32),
                y[b].numpy().astype(np.int16),
                pred[b].numpy().astype(np.int16),
            )

    affine = np.diag([PD_SPACING[0], PD_SPACING[1], PD_SPACING[2], 1.0]).astype(np.float32)
    csv_path = os.path.join(out_dir, "per_slice_dice.csv")
    eps = 1e-6

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "slice_idx", "dice", "gt_voxels",
                    "pred_voxels", "false_neg", "false_pos"])

        for pid, slices in sorted(vols.items()):
            order = sorted(slices.keys())
            img  = np.stack([slices[s][0] for s in order], axis=0)
            gt   = np.stack([slices[s][1] for s in order], axis=0)
            pr   = np.stack([slices[s][2] for s in order], axis=0)

            nib.save(nib.Nifti1Image(img,                  affine), os.path.join(out_dir, f"{pid}_image.nii.gz"))
            nib.save(nib.Nifti1Image((gt > 0).astype(np.int16), affine), os.path.join(out_dir, f"{pid}_gt.nii.gz"))
            nib.save(nib.Nifti1Image((pr > 0).astype(np.int16), affine), os.path.join(out_dir, f"{pid}_pred.nii.gz"))

            for k, s in enumerate(order):
                g = (gt[k] > 0); p = (pr[k] > 0)
                tp = float((g & p).sum()); gs = float(g.sum()); ps = float(p.sum())
                w.writerow([pid, s, f"{(2*tp + eps) / (ps + gs + eps):.4f}",
                            int(gs), int(ps), int(gs - tp), int(ps - tp)])

            log.info(f"  dumped {pid}: {len(order)} slices -> {pid}_{{image,gt,pred}}.nii.gz")

    log.info(f"  per-slice Dice CSV -> {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt",  required=True,
                    help="baseline_best_model.pth (5-class DESS model)")
    ap.add_argument("--fake_data_root",   required=True)
    ap.add_argument("--real_data_root",   required=True)
    ap.add_argument("--out_dir",          required=True)
    ap.add_argument("--epochs",           type=int,   default=100)
    ap.add_argument("--batch_size",       type=int,   default=8)
    ap.add_argument("--lr",               type=float, default=1e-5)
    ap.add_argument("--patience",         type=int,   default=15)
    ap.add_argument("--num_workers",      type=int,   default=4)
    ap.add_argument("--class_weights",    type=float, nargs=3,
                    default=[0.1, 1.5, 1.5])
    ap.add_argument("--seed",             type=int,   default=42,
                    help="change this between runs to measure seed noise")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  LR: {args.lr} → 1e-8  |  Data: mixed real+fake PD (DESS base)")
    log.info("Loss: ORIGINAL v7/v8 (plain weighted CE + SoftDice | MergedLoss w_bg=0.1)")

    model = build_model(args.pretrained_ckpt, device)

    fake_train = Meniscus2_5DDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), augment=True)
    fake_val   = Meniscus2_5DDataset(
        os.path.join(args.fake_data_root, "val",   "images"),
        os.path.join(args.fake_data_root, "val",   "masks"), augment=False)
    real_train = RealPDDataset(
        os.path.join(args.real_data_root, "train", "images"),
        os.path.join(args.real_data_root, "train", "masks"), augment=True)
    real_val   = RealPDDataset(
        os.path.join(args.real_data_root, "val",   "images"),
        os.path.join(args.real_data_root, "val",   "masks"), augment=False)

    fake_train_scans = len({pid for pid, _ in fake_train.items})
    fake_val_scans   = len({pid for pid, _ in fake_val.items})
    real_train_scans = len({s.rsplit("_", 1)[0] for s in real_train.slices})
    real_val_scans   = len({s.rsplit("_", 1)[0] for s in real_val.slices})
    log.info(f"Fake PD — train: {fake_train_scans} scans, {len(fake_train)} slices  |  "
             f"val: {fake_val_scans} scans, {len(fake_val)} slices")
    log.info(f"Real PD — train: {real_train_scans} scans, {len(real_train)} slices  |  "
             f"val: {real_val_scans} scans, {len(real_val)} slices")
    if real_val_scans <= 2:
        log.warning(f"Only {real_val_scans} real val patients — ckpt_best is selected on "
                    f"this. Watch the per-patient columns: if they move in opposite "
                    f"directions, checkpoint selection is being driven by noise.")

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    fake_val_loader   = DataLoader(fake_val,   args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)
    real_train_loader = DataLoader(real_train, args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=False)
    real_val_loader   = DataLoader(real_val,   args.batch_size, shuffle=False,
                                   num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-8)

    weights      = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = lambda logits, y: (nn.CrossEntropyLoss(weight=weights)(logits, y) +
                                      SoftDiceLoss()(logits, y))
    merged_loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    val_pids     = sorted({s.rsplit("_", 1)[0] for s in real_val.slices})
    hist_path    = os.path.join(args.out_dir, "history.csv")
    hist_fh      = open(hist_path, "w", newline="")
    hist         = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va",
                   "dice_real_tr", "dice_real_va", "gap",
                   "dice_fake_tr", "dice_fake_va", "dice_va_perpatient_mean"]
                  + [f"dice_va_{p}" for p in val_pids])

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td, tdf = run_epoch(model, real_train_loader, fake_train_loader,
                                optimizer, merged_loss_fn, fake_loss_fn, device, train=True)
        vl, vd, vdf = run_epoch(model, real_val_loader,   fake_val_loader,
                                optimizer, merged_loss_fn, fake_loss_fn, device, train=False)
        scheduler.step()

        # clean per-patient pass (each val slice exactly once)
        pp, _ = eval_real_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        pp_str  = "  ".join(f"{p}={pp.get(p, float('nan')):.4f}" for p in val_pids)

        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl:.4f}  |  "
                 f"dice_real tr={td:.4f} va={vd:.4f} gap={td - vd:+.4f}  |  "
                 f"dice_fake tr={tdf:.4f} va={vdf:.4f}")
        log.info(f"           per-patient val (clean pass): mean={pp_mean:.4f}   {pp_str}")

        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}",
                       f"{td:.5f}", f"{vd:.5f}", f"{td - vd:+.5f}",
                       f"{tdf:.5f}", f"{vdf:.5f}", f"{pp_mean:.5f}"]
                      + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids])
        hist_fh.flush()

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        # selection criterion UNCHANGED (still `vd`) so this replicates v7/v8
        if vd > best_val_dice:
            best_val_dice, no_improve = vd, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best: {best_val_dice:.4f} (epoch {epoch}) — dumping val predictions")
            dump_val_predictions(model, real_val, device,
                                 os.path.join(args.out_dir, "val_predictions"),
                                 args.batch_size)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}. Best: {best_val_dice:.4f}")
                break

    hist_fh.close()
    log.info(f"Done. Best val Dice: {best_val_dice:.4f}")
    log.info(f"History CSV        -> {hist_path}")
    log.info(f"Val predictions    -> {os.path.join(args.out_dir, 'val_predictions')}")
    log.info("Open <pid>_image.nii.gz in Slicer, load <pid>_gt.nii.gz and "
             "<pid>_pred.nii.gz as segmentations, and check per_slice_dice.csv "
             "for the worst slices.")


if __name__ == "__main__":
    main()
