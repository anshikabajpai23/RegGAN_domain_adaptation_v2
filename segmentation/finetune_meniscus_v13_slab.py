"""
finetune_meniscus_v13_slab.py
==============================
N1 (docs/reggan_paper_planning.md, Phase 6): train on slab-averaged fake-PD
data (segmentation_data_v4_slab/, from prepare_slab_fake_pd.py) instead of
thin-slice fake-PD data. One change from finetune_meniscus_v7_replica.py,
the fixed reference recipe:

  FAKE branch:  SlabMeniscus2_5DDataset (soft one-hot targets) + soft-label
                cross-entropy + soft Dice, same class weights (0.1, 1.5, 1.5)
                as the original hard-label losses.
  REAL branch:  UNCHANGED -- RealPDDataset, MergedLoss(w_bg=0.1, w_men=1.5),
                hard binary masks, aug_mode="equalized" (per spec step 7,
                "equalized (real matches fake, as in P4)").

Epoch definition (spec step 6, decision made -- not a free choice): the fake
slab set is ~4.5x smaller than the thin-slice set (fewer slabs than thin
slices). To keep real-PD exposure per epoch unchanged from the replica run
(P1 showed real exposure is signal), this script samples the fake TRAIN set
WITH REPLACEMENT for a FIXED number of batches per epoch
(--fake_batches_per_epoch, default 1189 -- the replica run's epoch length),
rather than letting epoch length shrink with the dataset. fake VAL uses a
plain deterministic full pass (no replacement) -- val is diagnostic, not
tuned to a fixed exposure count.

Read the output the same way as v7_replica's docstring: dice_real tr/va/gap
for over/underfitting, dice_fake for translation-vs-segmenter attribution
(note: dice_fake here is measured on slab targets thresholded at 0.5, so it
is not directly comparable to v7_replica's dice_fake on thin-slice hard
labels -- compare Dice on the REAL val set and the standard per-slice/
edge-core-empty dumps, not fake-val Dice, across the two runs).
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
from torch.utils.data import DataLoader, RandomSampler
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v4_slab import SlabMeniscus2_5DDataset
from dataset_2_5d_realpd import RealPDDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3
PRETRAINED_N_CLASSES = 5
PD_SPACING = (3.6, 0.39, 0.39)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
# Real-branch loss -- BYTE-IDENTICAL to finetune_meniscus_v7_replica.py.
# Do not edit; the real branch is not part of what N1 changes.
# ─────────────────────────────────────────────────────────────────────────────

class MergedLoss(nn.Module):
    def __init__(self, w_bg=0.1, w_men=1.5, eps=1e-6):
        super().__init__()
        self.w_bg, self.w_men, self.eps = w_bg, w_men, eps

    def forward(self, logits, binary_masks):
        probs = torch.softmax(logits, dim=1)
        p_bg = probs[:, 0]
        p_men = probs[:, 1] + probs[:, 2]
        gt_men = (binary_masks > 0).float()
        gt_bg = (binary_masks == 0).float()
        ce = -(self.w_bg * gt_bg * torch.log(p_bg.clamp(self.eps)) +
               self.w_men * gt_men * torch.log(p_men.clamp(self.eps))).mean()
        tp = (p_men * gt_men).sum()
        dice = 1.0 - (2 * tp + self.eps) / (p_men.sum() + gt_men.sum() + self.eps)
        return ce + dice


# ─────────────────────────────────────────────────────────────────────────────
# Fake-branch loss -- NEW for N1. Soft-label analogues of the original
# CrossEntropyLoss(weight=...) + SoftDiceLoss, taking fractional (3,H,W)
# targets instead of a (H,W) int64 hard label. Weights and structure match
# the original exactly; only "target is one-hot" -> "target is a fraction"
# changes.
# ─────────────────────────────────────────────────────────────────────────────

class SoftLabelCrossEntropy(nn.Module):
    def __init__(self, class_weights, eps=1e-6):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(class_weights, dtype=torch.float32))
        self.eps = eps

    def forward(self, logits, soft_targets):
        # logits: (B, 3, H, W); soft_targets: (B, 3, H, W), channels sum to 1
        log_probs = torch.log_softmax(logits, dim=1)
        w = self.w.to(logits.device).view(1, -1, 1, 1)
        per_pixel = -(w * soft_targets * log_probs).sum(dim=1)
        return per_pixel.mean()


class SoftLabelDiceLoss(nn.Module):
    def __init__(self, n_classes=N_CLASSES, eps=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, logits, soft_targets):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(1, self.n_classes):
            p = probs[:, c]
            t = soft_targets[:, c]
            loss += 1.0 - (2 * (p * t).sum() + self.eps) / (p.sum() + t.sum() + self.eps)
        return loss / (self.n_classes - 1)


def dice_binary(preds, targets, eps=1e-6):
    """preds: argmax labels; targets: either int64 hard mask or float soft
    mask (3,H,W) -- if soft, threshold channels 1+2 at 0.5 to get a pseudo-hard
    GT for logging only (never used in the loss)."""
    pred_men = (preds > 0).float()
    if targets.dim() == preds.dim() + 1:  # (B,3,H,W) soft
        gt_men = ((targets[:, 1] + targets[:, 2]) > 0.5).float()
    else:
        gt_men = (targets > 0).float()
    tp = (pred_men * gt_men).sum()
    return float((2 * tp + eps) / (pred_men.sum() + gt_men.sum() + eps))


def run_epoch(model, real_loader, fake_loader, optimizer,
              merged_loss_fn, fake_loss_fns, device, train=True):
    model.train() if train else model.eval()
    total_loss, dice_sum, dice_fake_sum, n = 0.0, 0.0, 0.0, 0
    real_iter = iter(real_loader)
    soft_ce_fn, soft_dice_fn = fake_loss_fns

    with torch.set_grad_enabled(train):
        for fake_batch in fake_loader:
            xf = fake_batch["image"].to(device)
            yf = fake_batch["mask"].to(device)  # (B,3,H,W) soft
            logits_f = model(xf)
            loss_f = soft_ce_fn(logits_f, yf) + soft_dice_fn(logits_f, yf)

            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_batch = next(real_iter)
            xr, yr = real_batch["image"].to(device), real_batch["mask"].to(device)
            logits_r = model(xr)
            loss_r = merged_loss_fn(logits_r, yr)

            loss = loss_f + loss_r
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_loss += loss.item()
            dice_sum += dice_binary(logits_r.argmax(1), yr)
            dice_fake_sum += dice_binary(logits_f.argmax(1), yf)
            n += 1

    return (total_loss / max(n, 1),
            dice_sum / max(n, 1),
            dice_fake_sum / max(n, 1))


@torch.no_grad()
def eval_real_per_patient(model, dataset, device, batch_size=8):
    model.eval()
    acc = {}
    for start in range(0, len(dataset), batch_size):
        idxs = range(start, min(start + batch_size, len(dataset)))
        items = [dataset[i] for i in idxs]
        stems = [dataset.slices[i] for i in idxs]
        x = torch.stack([it["image"] for it in items]).to(device)
        y = torch.stack([it["mask"] for it in items]).to(device)
        pred = model(x).argmax(1)

        pm, gm = (pred > 0).float(), (y > 0).float()
        for b, stem in enumerate(stems):
            pid = stem.rsplit("_", 1)[0]
            a = acc.setdefault(pid, [0.0, 0.0, 0.0])
            a[0] += float((pm[b] * gm[b]).sum())
            a[1] += float(pm[b].sum())
            a[2] += float(gm[b].sum())

    dices, eps = {}, 1e-6
    for pid, (tp, ps, gs) in acc.items():
        dices[pid] = (2 * tp + eps) / (ps + gs + eps)
    return dices


@torch.no_grad()
def dump_val_predictions(model, dataset, device, out_dir, batch_size=8):
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel not available — skipping val prediction dump")
        return

    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    vols = {}

    for start in range(0, len(dataset), batch_size):
        idxs = range(start, min(start + batch_size, len(dataset)))
        items = [dataset[i] for i in idxs]
        stems = [dataset.slices[i] for i in idxs]
        x = torch.stack([it["image"] for it in items]).to(device)
        y = torch.stack([it["mask"] for it in items])
        pred = model(x).argmax(1).cpu()

        for b, stem in enumerate(stems):
            pid, sidx = stem.rsplit("_", 1)
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
            img = np.stack([slices[s][0] for s in order], axis=0)
            gt = np.stack([slices[s][1] for s in order], axis=0)
            pr = np.stack([slices[s][2] for s in order], axis=0)

            nib.save(nib.Nifti1Image(img, affine), os.path.join(out_dir, f"{pid}_image.nii.gz"))
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
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root", required=True,
                    help="segmentation_data_v4_slab/ (from prepare_slab_fake_pd.py)")
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--fake_batches_per_epoch", type=int, default=1189,
                    help="Fixed fake-batch count per epoch (spec step 6) -- "
                         "keeps real-PD exposure equal to the replica run "
                         "despite the ~4.5x smaller slab dataset. Sampled "
                         "WITH REPLACEMENT.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  LR: {args.lr} → 1e-8  |  N1 slab data + soft labels")
    log.info("Fake loss: soft CE + soft Dice (weights 0.1/1.5/1.5)  |  "
             "Real loss: MergedLoss (UNCHANGED from replica)")

    model = build_model(args.pretrained_ckpt, device)

    fake_train = SlabMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), augment=True)
    fake_val = SlabMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "val", "images"),
        os.path.join(args.fake_data_root, "val", "masks"), augment=False)
    real_train = RealPDDataset(
        os.path.join(args.real_data_root, "train", "images"),
        os.path.join(args.real_data_root, "train", "masks"),
        augment=True, aug_mode="equalized")
    real_val = RealPDDataset(
        os.path.join(args.real_data_root, "val", "images"),
        os.path.join(args.real_data_root, "val", "masks"),
        augment=False, aug_mode="equalized")

    fake_train_scans = len(fake_train.n_slabs)
    fake_val_scans = len(fake_val.n_slabs)
    real_train_scans = len({s.rsplit("_", 1)[0] for s in real_train.slices})
    real_val_scans = len({s.rsplit("_", 1)[0] for s in real_val.slices})
    log.info(f"Fake PD (slab) — train: {fake_train_scans} scans, {len(fake_train)} slabs  |  "
             f"val: {fake_val_scans} scans, {len(fake_val)} slabs")
    log.info(f"Real PD — train: {real_train_scans} scans, {len(real_train)} slices  |  "
             f"val: {real_val_scans} scans, {len(real_val)} slices")
    if real_val_scans <= 2:
        log.warning(f"Only {real_val_scans} real val patients — same caveat as replica: "
                    f"watch per-patient columns for noise-driven checkpoint selection.")

    fake_train_sampler = RandomSampler(
        fake_train, replacement=True,
        num_samples=args.fake_batches_per_epoch * args.batch_size)
    fake_train_loader = DataLoader(fake_train, args.batch_size,
                                   sampler=fake_train_sampler,
                                   num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
    real_train_loader = DataLoader(real_train, args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=False)
    real_val_loader = DataLoader(real_val, args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-8)

    weights = args.class_weights
    fake_loss_fns = (SoftLabelCrossEntropy(weights).to(device),
                      SoftLabelDiceLoss().to(device))
    merged_loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices})
    hist_path = os.path.join(args.out_dir, "history.csv")
    hist_fh = open(hist_path, "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va",
                   "dice_real_tr", "dice_real_va", "gap",
                   "dice_fake_tr", "dice_fake_va", "dice_va_perpatient_mean"]
                  + [f"dice_va_{p}" for p in val_pids])

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td, tdf = run_epoch(model, real_train_loader, fake_train_loader,
                                optimizer, merged_loss_fn, fake_loss_fns, device, train=True)
        vl, vd, vdf = run_epoch(model, real_val_loader, fake_val_loader,
                                optimizer, merged_loss_fn, fake_loss_fns, device, train=False)
        scheduler.step()

        pp = eval_real_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        pp_str = "  ".join(f"{p}={pp.get(p, float('nan')):.4f}" for p in val_pids)

        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl:.4f}  |  "
                 f"dice_real tr={td:.4f} va={vd:.4f} gap={td - vd:+.4f}  |  "
                 f"dice_fake(slab,thr0.5) tr={tdf:.4f} va={vdf:.4f}")
        log.info(f"           per-patient val (clean pass): mean={pp_mean:.4f}   {pp_str}")

        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}",
                       f"{td:.5f}", f"{vd:.5f}", f"{td - vd:+.5f}",
                       f"{tdf:.5f}", f"{vdf:.5f}", f"{pp_mean:.5f}"]
                      + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids])
        hist_fh.flush()

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
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
    log.info("Compare vs run_v7_replica_seed42 -- read against the 0.0226 "
             "per-patient noise floor (phase4_5_seed_noise.py). Then run the "
             "edge/core/empty split (phase4_3_edge_core_split.py) on the "
             "dumped predictions to check whether the gain (if any) is "
             "concentrated at span ends, as N1's mechanism predicts.")


if __name__ == "__main__":
    main()
