"""
finetune_meniscus_fakeonly_base.py
===================================
Run 3 BASELINE — fake PD ONLY. No real PD in training, no real PD in
checkpoint selection.

WHY THIS EXISTS
---------------
Runs 1 and 2 trained on fake PD + 13 real PD patients. In that mixed regime the
model has already seen the real distribution, so the fake->real gap is ~0.02
(docs/DEBUG_HANDOFF.md 17.7: fake-val 0.760 vs real 0.781) and every Phase 5
idea was competing for ~0.01 against a 0.0226 noise floor -- unwinnable by
construction.

Remove real PD from training and the gap reopens to 0.09-0.15:

    fake-only  (run_002_v2, best) -> real PD Dice 0.695
    mixed      (v7_mixed)         -> real PD Dice 0.781

That is 4-6x the noise floor, so an idea that closes even a third of the domain
gap is now measurable. This script is the reference those ideas are measured
against.

SELECTION: FAKE VAL ONLY
------------------------
ckpt_best is chosen on dice_fake_va. Real PD is NEVER used for gradients and
NEVER used for selection, so a result from this arm is a genuine zero-shot
claim ("no target-domain labels at any point"), which is the claim a reviewer
will check first.

Real val Dice IS still computed and logged when --real_data_root is given, as a
DIAGNOSTIC ONLY -- it is written to history.csv so the domain gap is visible
per epoch, and it never touches selection. Omit --real_data_root entirely for a
run that never reads a real-PD label at all.

SMOOTHED SELECTION (--select_smooth, default 3)
-----------------------------------------------
Selection uses a trailing mean of dice_fake_va over the last N epochs rather
than the single best epoch. Measured on the Run 1/2 histories, 73-89% of epochs
sat within the noise floor of their run's best epoch, so picking the single
argmax was picking one arbitrary winner out of ~20 statistically tied
candidates. A trailing mean picks a checkpoint sitting on a stable plateau
instead of one that spiked. Set --select_smooth 1 to recover the old
single-epoch argmax behaviour.

NOTE ON THE NOISE FLOOR
-----------------------
The 0.0226 floor was measured in the MIXED regime (seed42 vs seed7 of
finetune_meniscus_v7_replica.py) and does NOT transfer here. Run this script at
two seeds (SEED=42 and SEED=7) before judging any Run 3 variant against it.
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
from dataset_2_5d_v2 import Meniscus2_5DDataset
from dataset_2_5d_realpd import RealPDDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3
PRETRAINED_N_CLASSES = 5
PD_SPACING = (3.6, 0.39, 0.39)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"Seed fixed: {seed} (cudnn.deterministic=True)")


def build_model(ckpt_path, device):
    """Verbatim from finetune_meniscus_v7_replica.py -- same DESS baseline
    checkpoint and same re-headed plain smp.Unet, so the only difference
    between this arm and Run 2 is the training data, not the architecture."""
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=PRETRAINED_N_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    log.info(f"Loaded DESS baseline from {ckpt_path}")
    old_head = model.segmentation_head[0]
    model.segmentation_head[0] = nn.Conv2d(
        old_head.in_channels, N_CLASSES,
        kernel_size=old_head.kernel_size, stride=old_head.stride, padding=old_head.padding)
    log.info(f"Replaced head: {PRETRAINED_N_CLASSES} -> {N_CLASSES} classes")
    return model.to(device)


class SoftDiceLoss(nn.Module):
    """Verbatim from finetune_meniscus_v7_replica.py."""
    def __init__(self, n_classes=N_CLASSES, eps=1e-6):
        super().__init__(); self.n_classes = n_classes; self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1); loss = 0.0
        for c in range(1, self.n_classes):
            p = probs[:, c]; t = (targets == c).float()
            loss += 1.0 - (2 * (p * t).sum() + self.eps) / (p.sum() + t.sum() + self.eps)
        return loss / (self.n_classes - 1)


def dice_binary(preds, targets, eps=1e-6):
    pm = (preds > 0).float(); gm = (targets > 0).float()
    tp = (pm * gm).sum()
    return float((2 * tp + eps) / (pm.sum() + gm.sum() + eps))


def run_epoch(model, fake_loader, optimizer, fake_loss_fn, device, train=True):
    """Fake-PD branch only. The real-PD branch and MergedLoss of the mixed
    recipe are gone entirely -- not weighted to zero, removed."""
    model.train() if train else model.eval()
    total_loss, dice_fake_sum, n = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for batch in fake_loader:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            logits = model(x)
            loss = fake_loss_fn(logits, y)

            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_loss += loss.item()
            dice_fake_sum += dice_binary(logits.argmax(1), y)
            n += 1

    n = max(n, 1)
    return total_loss / n, dice_fake_sum / n


@torch.no_grad()
def eval_real_per_patient(model, dataset, device, batch_size=8):
    """DIAGNOSTIC ONLY. Never feeds selection -- see module docstring."""
    model.eval(); acc = {}
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
            a[0] += float((pm[b] * gm[b]).sum()); a[1] += float(pm[b].sum()); a[2] += float(gm[b].sum())
    eps = 1e-6
    return {pid: (2 * tp + eps) / (ps + gs + eps) for pid, (tp, ps, gs) in acc.items()}


@torch.no_grad()
def dump_val_predictions(model, dataset, device, out_dir, batch_size=8):
    """Dumps the REAL val patients (diagnostic) so the zero-shot output can be
    inspected in Slicer. Skipped when --real_data_root is not given."""
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel not available -- skipping val prediction dump"); return
    model.eval(); os.makedirs(out_dir, exist_ok=True); vols = {}
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
                y[b].numpy().astype(np.int16), pred[b].numpy().astype(np.int16))
    affine = np.diag([PD_SPACING[0], PD_SPACING[1], PD_SPACING[2], 1.0]).astype(np.float32)
    csv_path = os.path.join(out_dir, "per_slice_dice.csv"); eps = 1e-6
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "slice_idx", "dice", "gt_voxels", "pred_voxels", "false_neg", "false_pos"])
        for pid, slices in sorted(vols.items()):
            order = sorted(slices.keys())
            img = np.stack([slices[s][0] for s in order], 0)
            gt = np.stack([slices[s][1] for s in order], 0)
            pr = np.stack([slices[s][2] for s in order], 0)
            nib.save(nib.Nifti1Image(img, affine), os.path.join(out_dir, f"{pid}_image.nii.gz"))
            nib.save(nib.Nifti1Image((gt > 0).astype(np.int16), affine), os.path.join(out_dir, f"{pid}_gt.nii.gz"))
            nib.save(nib.Nifti1Image((pr > 0).astype(np.int16), affine), os.path.join(out_dir, f"{pid}_pred.nii.gz"))
            for k, s in enumerate(order):
                g = (gt[k] > 0); p_ = (pr[k] > 0)
                tp = float((g & p_).sum()); gs = float(g.sum()); ps = float(p_.sum())
                w.writerow([pid, s, f"{(2*tp + eps) / (ps + gs + eps):.4f}",
                            int(gs), int(ps), int(gs - tp), int(ps - tp)])
            log.info(f"  dumped {pid}: {len(order)} slices")
    log.info(f"  per-slice Dice CSV -> {csv_path}")


def add_common_args(ap):
    """Shared by every Run 3 variant so the arms stay comparable."""
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", default=None,
                    help="OPTIONAL. If given, real val Dice is logged as a DIAGNOSTIC. "
                         "It never affects gradients or checkpoint selection. Omit for a "
                         "run that never reads a real-PD label at all.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select_smooth", type=int, default=3,
                    help="Trailing-mean window over dice_fake_va used for checkpoint "
                         "selection. 1 = old single-epoch argmax. See module docstring.")
    return ap


def build_loaders(args):
    """Fake train/val always. Real val only as a diagnostic, never train."""
    fake_train = Meniscus2_5DDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), augment=True)
    fake_val = Meniscus2_5DDataset(
        os.path.join(args.fake_data_root, "val", "images"),
        os.path.join(args.fake_data_root, "val", "masks"), augment=False)

    real_val = None
    if args.real_data_root:
        real_val = RealPDDataset(
            os.path.join(args.real_data_root, "val", "images"),
            os.path.join(args.real_data_root, "val", "masks"), augment=False)

    n_tr = len({pid for pid, _ in fake_train.items})
    n_va = len({pid for pid, _ in fake_val.items})
    log.info(f"Fake PD - train: {n_tr} scans, {len(fake_train)} slices  |  "
             f"val: {n_va} scans, {len(fake_val)} slices")
    if real_val is not None:
        n_rv = len({s.rsplit('_', 1)[0] for s in real_val.slices})
        log.info(f"Real PD - val ONLY (diagnostic, never trained on): {n_rv} scans, "
                 f"{len(real_val)} slices")
    else:
        log.info("Real PD - NOT LOADED. This run never reads a real-PD label.")
    return fake_train, fake_val, real_val


def history_header(real_val, val_pids, extra_cols=()):
    cols = ["epoch", "lr", "loss_tr", "loss_va",
            "dice_fake_tr", "dice_fake_va", "gap_fake", "dice_fake_va_smooth"]
    if real_val is not None:
        cols += ["dice_real_va_diag"] + [f"dice_va_{p}" for p in val_pids]
    return cols + list(extra_cols)


def main():
    ap = add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  LR: {args.lr} -> 1e-8  |  RUN 3 BASELINE: FAKE PD ONLY")
    log.info(f"Selection: dice_fake_va, trailing mean over {args.select_smooth} epoch(s). "
             f"Real PD never used for selection.")

    model = build_model(args.pretrained_ckpt, device)
    fake_train, fake_val, real_val = build_loaders(args)

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = lambda logits, y: (nn.CrossEntropyLoss(weight=weights)(logits, y) + SoftDiceLoss()(logits, y))

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices}) if real_val is not None else []
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(history_header(real_val, val_pids))

    best_smooth, no_improve, va_hist = -1.0, 0, []
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, tdf = run_epoch(model, fake_train_loader, optimizer, fake_loss_fn, device, train=True)
        vl, vdf = run_epoch(model, fake_val_loader, optimizer, fake_loss_fn, device, train=False)
        scheduler.step()

        va_hist.append(vdf)
        smooth = float(np.mean(va_hist[-args.select_smooth:]))

        row = [epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}",
               f"{tdf:.5f}", f"{vdf:.5f}", f"{tdf - vdf:+.5f}", f"{smooth:.5f}"]
        msg = (f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl:.4f}  |  "
               f"dice_fake tr={tdf:.4f} va={vdf:.4f} gap={tdf - vdf:+.4f} smooth={smooth:.4f}")

        if real_val is not None:
            pp = eval_real_per_patient(model, real_val, device, args.batch_size)
            pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
            row += [f"{pp_mean:.5f}"] + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids]
            msg += f"  |  [diag] real_va={pp_mean:.4f}"
        log.info(msg)

        hist.writerow(row); hist_fh.flush()

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        if smooth > best_smooth:
            best_smooth, no_improve = smooth, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best (smoothed fake val): {best_smooth:.4f} (epoch {epoch})")
            if real_val is not None:
                dump_val_predictions(model, real_val, device,
                                     os.path.join(args.out_dir, "val_predictions"), args.batch_size)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}. Best smoothed: {best_smooth:.4f}")
                break

    hist_fh.close()
    log.info(f"Done. Best smoothed fake val Dice: {best_smooth:.4f}")
    log.info("Run this at SEED=42 and SEED=7 to measure the Run 3 noise floor "
             "before judging any variant -- the mixed-regime 0.0226 does not apply here.")


if __name__ == "__main__":
    main()
