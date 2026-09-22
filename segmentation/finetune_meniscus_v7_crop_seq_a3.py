"""
finetune_meniscus_v7_crop_seq_a3.py
====================================
Phase 5, A3 — slice-position weighted loss, on top of the crop+sequence base.

BASE (unchanged from finetune_meniscus_v7_sequence.py run on cropped data):
  - starts from the SAME pretrained/baseline_best_model.pth as every Phase 4 run
  - SequenceUNet (5 slices, shared encoder, ConvLSTM at the bottleneck)
  - cropped data roots (segmentation_data_v2_crop / real_pd_seg_data_v7_crop)
  - same optimizer, LR schedule, class weights, seed, epochs, patience
  - same history.csv schema, so it drops straight into the notebook's Section 2/3

THE ONE CHANGE: each sample's loss is scaled by a per-slice weight from
scripts/build_slice_weights.py, based on distance to the nearest meniscus-span
end (in mm, so the 0.8 mm fake and 3.6 mm real cohorts stay comparable).

    --variant up    end slices count up to 2x    (does the model just need pressure there?)
    --variant down  end slices count as 0.3x     (check 4A says the ends are label-poisoned)
    --variant none  all weights 1.0              (CONTROL, see below)

⚠ WHY --variant none EXISTS. Weighting requires reducing each loss to a
per-sample value first. The baseline's SoftDice and MergedLoss reduce over the
WHOLE BATCH, and batch-level Dice != mean of per-sample Dice. So "up"/"down"
differ from the crop+sequence baseline in two ways at once: the weighting AND
the reduction. `--variant none` keeps the per-sample reduction but sets every
weight to 1.0, isolating the reduction change on its own. Run it only if up/down
produce something interesting -- otherwise it is a wasted 6 h.

Normalising by w.sum() rather than batch size keeps the loss scale comparable to
the baseline on average (doc §3.3).

Usage: same args as finetune_meniscus_v7_sequence.py, plus
  --variant {up,down,none} --weights_fake <json> --weights_real <json>
"""
import argparse
import csv
import json
import logging
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_sequence import SequenceMeniscusDataset, SequenceRealPDDataset
from sequence_model import build_sequence_model

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


# ─────────────────────────────────────────────────────────────────────────────
# Datasets: identical to the base, plus a per-sample weight. Subclassed HERE
# rather than editing dataset_sequence.py, which 13 other files import.
# ─────────────────────────────────────────────────────────────────────────────

class WeightedSequenceMeniscusDataset(SequenceMeniscusDataset):
    def __init__(self, img_root, mask_root, weights, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.weights = weights            # {pid: {slice_idx: w}} or None

    def __getitem__(self, i):
        item = super().__getitem__(i)
        pid, idx = self.items[i]
        w = 1.0
        if self.weights is not None:
            w = self.weights.get(pid, {}).get(str(idx), self.weights.get(pid, {}).get(idx, 1.0))
        item["weight"] = torch.tensor(float(w), dtype=torch.float32)
        return item


class WeightedSequenceRealPDDataset(SequenceRealPDDataset):
    def __init__(self, img_root, mask_root, weights, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.weights = weights

    def __getitem__(self, i):
        item = super().__getitem__(i)
        stem = self.slices[i]
        pid, sidx = stem.rsplit("_", 1)
        w = 1.0
        if self.weights is not None:
            pw = self.weights.get(pid, {})
            w = pw.get(str(int(sidx)), pw.get(int(sidx), 1.0))
        item["weight"] = torch.tensor(float(w), dtype=torch.float32)
        # base class does not return slice_idx; needed for the val-dump grouping
        item["slice_idx"] = int(sidx)
        return item


# ─────────────────────────────────────────────────────────────────────────────
# Losses — same maths as the base, reduced PER SAMPLE so they can be weighted.
# With w == 1 everywhere these differ from the base only by that reduction.
# ─────────────────────────────────────────────────────────────────────────────

class MergedLossPerSample(nn.Module):
    """Base MergedLoss, reduced to (B,) instead of a scalar."""
    def __init__(self, w_bg=0.1, w_men=1.5, eps=1e-6):
        super().__init__()
        self.w_bg, self.w_men, self.eps = w_bg, w_men, eps

    def forward(self, logits, binary_masks):
        probs = torch.softmax(logits, dim=1)
        p_bg = probs[:, 0]
        p_men = probs[:, 1] + probs[:, 2]
        gt_men = (binary_masks > 0).float()
        gt_bg = (binary_masks == 0).float()
        ce = -(self.w_bg * gt_bg * torch.log(p_bg.clamp(min=self.eps)) +
               self.w_men * gt_men * torch.log(p_men.clamp(min=self.eps))).mean(dim=(1, 2))
        tp = (p_men * gt_men).sum(dim=(1, 2))
        dice = 1.0 - (2 * tp + self.eps) / (p_men.sum(dim=(1, 2)) + gt_men.sum(dim=(1, 2)) + self.eps)
        return ce + dice                                   # (B,)


class SoftDiceLossPerSample(nn.Module):
    def __init__(self, n_classes=N_CLASSES, eps=1e-6):
        super().__init__(); self.n_classes = n_classes; self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(1, self.n_classes):
            p = probs[:, c]; t = (targets == c).float()
            inter = (p * t).sum(dim=(1, 2))
            den = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            loss = loss + (1.0 - (2 * inter + self.eps) / (den + self.eps))
        return loss / (self.n_classes - 1)                 # (B,)


def fake_loss_per_sample(logits, y, class_weights):
    ce = F.cross_entropy(logits, y, weight=class_weights, reduction="none").mean(dim=(1, 2))
    return ce + SoftDiceLossPerSample()(logits, y)


def weighted_mean(per_sample, w):
    """Doc §3.3: normalise by w.sum(), not B, so the loss scale matches the
    baseline's on average regardless of which variant is running."""
    return (per_sample * w).sum() / w.sum().clamp(min=1e-6)


def dice_binary(preds, targets, eps=1e-6):
    pred_men = (preds > 0).float(); gt_men = (targets > 0).float()
    tp = (pred_men * gt_men).sum()
    return float((2 * tp + eps) / (pred_men.sum() + gt_men.sum() + eps))


def run_epoch(model, real_loader, fake_loader, optimizer,
              merged_loss_fn, class_weights, device, train=True):
    model.train() if train else model.eval()
    total_loss, dice_sum, dice_fake_sum, n = 0.0, 0.0, 0.0, 0
    real_iter = iter(real_loader)

    with torch.set_grad_enabled(train):
        for fake_batch in fake_loader:
            xf, yf = fake_batch["image"].to(device), fake_batch["mask"].to(device)
            wf = fake_batch["weight"].to(device)
            logits_f = model(xf)
            loss_f = weighted_mean(fake_loss_per_sample(logits_f, yf, class_weights), wf)

            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_batch = next(real_iter)
            xr, yr = real_batch["image"].to(device), real_batch["mask"].to(device)
            wr = real_batch["weight"].to(device)
            logits_r = model(xr)
            loss_r = weighted_mean(merged_loss_fn(logits_r, yr), wr)

            loss = loss_f + loss_r
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_loss += loss.item()
            dice_sum += dice_binary(logits_r.argmax(1), yr)
            dice_fake_sum += dice_binary(logits_f.argmax(1), yf)
            n += 1

    return (total_loss / max(n, 1), dice_sum / max(n, 1), dice_fake_sum / max(n, 1))


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
    eps = 1e-6
    return {pid: (2 * tp + eps) / (ps + gs + eps) for pid, (tp, ps, gs) in acc.items()}


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
                x[b, 2].cpu().numpy().astype(np.float32),
                y[b].numpy().astype(np.int16),
                pred[b].numpy().astype(np.int16),
            )
    affine = np.diag([PD_SPACING[0], PD_SPACING[1], PD_SPACING[2], 1.0]).astype(np.float32)
    csv_path = os.path.join(out_dir, "per_slice_dice.csv")
    eps = 1e-6
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "slice_idx", "dice", "gt_voxels", "pred_voxels", "false_neg", "false_pos"])
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
            log.info(f"  dumped {pid}: {len(order)} slices")
    log.info(f"  per-slice Dice CSV -> {csv_path}")


def load_weights(path, variant):
    if variant == "none" or not path:
        return None
    with open(path) as fh:
        blob = json.load(fh)
    if variant not in blob:
        raise SystemExit(f"{path} has no '{variant}' block (keys: {list(blob)})")
    meta = blob.get("meta", {})
    log.info(f"  weights {os.path.basename(path)} [{variant}]: "
             f"{meta.get('n_patients')} patients, {meta.get('n_slices')} slices, "
             f"spacing {meta.get('spacing_mm')} mm, tau {meta.get('tau_mm')} mm")
    return blob[variant]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", required=True, choices=["up", "down", "none"])
    ap.add_argument("--weights_fake", default=None)
    ap.add_argument("--weights_real", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.variant != "none" and not (args.weights_fake and args.weights_real):
        ap.error("--variant up/down requires --weights_fake and --weights_real")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  LR: {args.lr} -> 1e-8  |  5-slice sequence + ConvLSTM, cropped data")
    log.info(f"A3 variant: {args.variant}"
             + ("  (CONTROL: uniform weights, isolates the per-sample reduction)"
                if args.variant == "none" else ""))

    w_fake = load_weights(args.weights_fake, args.variant)
    w_real = load_weights(args.weights_real, args.variant)

    model = build_sequence_model(args.pretrained_ckpt, device,
                                 pretrained_n_classes=PRETRAINED_N_CLASSES, n_classes=N_CLASSES)
    log.info(f"Loaded DESS baseline from {args.pretrained_ckpt}, wrapped with SequenceUNet")

    fake_train = WeightedSequenceMeniscusDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), w_fake, augment=True)
    fake_val = WeightedSequenceMeniscusDataset(
        os.path.join(args.fake_data_root, "val", "images"),
        os.path.join(args.fake_data_root, "val", "masks"), w_fake, augment=False)
    real_train = WeightedSequenceRealPDDataset(
        os.path.join(args.real_data_root, "train", "images"),
        os.path.join(args.real_data_root, "train", "masks"), w_real, augment=True)
    real_val = WeightedSequenceRealPDDataset(
        os.path.join(args.real_data_root, "val", "images"),
        os.path.join(args.real_data_root, "val", "masks"), w_real, augment=False)

    log.info(f"Fake PD — train: {len(fake_train)} slices  |  val: {len(fake_val)} slices")
    log.info(f"Real PD — train: {len(real_train)} slices  |  val: {len(real_val)} slices")

    # sanity: weights actually attached, and spread as expected for the variant
    sample_w = [float(fake_train[i]["weight"]) for i in range(0, min(len(fake_train), 400), 7)]
    log.info(f"  fake-train weight sample: min {min(sample_w):.3f}  "
             f"mean {sum(sample_w)/len(sample_w):.3f}  max {max(sample_w):.3f}")

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False, num_workers=args.num_workers)
    real_train_loader = DataLoader(real_train, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    real_val_loader = DataLoader(real_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    merged_loss_fn = MergedLossPerSample(w_bg=0.1, w_men=1.5)

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices})
    hist_path = os.path.join(args.out_dir, "history.csv")
    hist_fh = open(hist_path, "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va", "dice_real_tr", "dice_real_va", "gap",
                   "dice_fake_tr", "dice_fake_va", "dice_va_perpatient_mean"] + [f"dice_va_{p}" for p in val_pids])

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td, tdf = run_epoch(model, real_train_loader, fake_train_loader, optimizer,
                                merged_loss_fn, class_weights, device, train=True)
        vl, vd, vdf = run_epoch(model, real_val_loader, fake_val_loader, optimizer,
                                merged_loss_fn, class_weights, device, train=False)
        scheduler.step()

        pp = eval_real_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        pp_str = "  ".join(f"{p}={pp.get(p, float('nan')):.4f}" for p in val_pids)

        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl:.4f}  |  "
                 f"dice_real tr={td:.4f} va={vd:.4f} gap={td - vd:+.4f}  |  dice_fake tr={tdf:.4f} va={vdf:.4f}")
        log.info(f"           per-patient val (clean pass): mean={pp_mean:.4f}   {pp_str}")

        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}", f"{td:.5f}", f"{vd:.5f}", f"{td - vd:+.5f}",
                       f"{tdf:.5f}", f"{vdf:.5f}", f"{pp_mean:.5f}"] + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids])
        hist_fh.flush()

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        if vd > best_val_dice:
            best_val_dice, no_improve = vd, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best: {best_val_dice:.4f} (epoch {epoch}) — dumping val predictions")
            dump_val_predictions(model, real_val, device, os.path.join(args.out_dir, "val_predictions"), args.batch_size)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}. Best: {best_val_dice:.4f}")
                break

    hist_fh.close()
    log.info(f"Done. Best val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
