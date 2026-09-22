"""
finetune_meniscus_v7_crop_seq_d.py
===================================
Phase 5, D — position-conditioned shape prior (area), on the crop+sequence base.

BASE (unchanged from run_v7_crop_sequence): same pretrained/baseline_best_model.pth,
same SequenceUNet + ConvLSTM, same cropped data, same losses/optimizer/LR/seed/
epochs/patience, same history.csv schema.

THE ONE CHANGE: + lam * L_area, where L_area penalises the predicted soft area
for deviating from the expected area at that slice's WITHIN-SPAN position:

    L_area = mean over valid samples of ((area_pred - A_mean[bin]) / (A_std[bin] + 1))^2

A_mean/A_std come from scripts/build_shape_prior.py, built on the CROPPED masks
(the crop magnifies area ~2.5x, so an uncropped template would push the model to
under-segment by that factor).

TRAINING-TIME ONLY, and that is a hard constraint, not a choice (doc §4.2): the
bin is derived from GT spans, which do not exist at inference. So this is a
regulariser that shapes training; the trained model is used normally afterwards.

SEPARATE TEMPLATES PER COHORT. Fake (0.8 mm DESS-derived) and real (3.6 mm)
capture different tissue thickness per slice, so their apparent areas differ.
Each branch is regularised against its own template rather than a pooled one.

SAMPLES WITHOUT A BIN CONTRIBUTE NOTHING: empty-GT slices, and patients whose
notch was labelled through (single span, so the notch end is unidentifiable and
was excluded from the template). They are dropped from the mean rather than
defaulted, so the term never speaks where it has no reference.

WATCH area_tr / area_frac in history.csv. Doc §4.4 requires the term to stay
under ~10% of the base loss; if area_frac climbs well past that, lam is too high.
"""
import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_sequence import SequenceMeniscusDataset, SequenceRealPDDataset
from sequence_model import build_sequence_model
from rim_loss import meniscus_prob          # reused: identical lat+med probability

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3
PRETRAINED_N_CLASSES = 5
PD_SPACING = (3.6, 0.39, 0.39)
NO_BIN = -1


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"Seed fixed: {seed} (cudnn.deterministic=True)")


class BinnedSequenceMeniscusDataset(SequenceMeniscusDataset):
    """Base behaviour + the within-span bin for this slice (NO_BIN if it has none)."""
    def __init__(self, img_root, mask_root, bin_map, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.bin_map = bin_map or {}

    def __getitem__(self, i):
        item = super().__getitem__(i)
        pid, idx = self.items[i]
        item["bin"] = int(self.bin_map.get(pid, {}).get(str(idx), NO_BIN))
        return item


class BinnedSequenceRealPDDataset(SequenceRealPDDataset):
    def __init__(self, img_root, mask_root, bin_map, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.bin_map = bin_map or {}

    def __getitem__(self, i):
        item = super().__getitem__(i)
        pid, sidx = self.slices[i].rsplit("_", 1)
        item["bin"] = int(self.bin_map.get(pid, {}).get(str(int(sidx)), NO_BIN))
        return item


def area_loss(logits, bins, a_mean, a_std):
    """bins (B,) long, NO_BIN where the slice has no reference.
    a_mean/a_std: (K,) tensors on the same device."""
    valid = bins >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0          # keeps the graph, contributes nothing

    area_pred = meniscus_prob(logits).sum(dim=(1, 2))          # soft area, (B,)
    b = bins.clamp(min=0)
    tgt = a_mean[b]
    sd = a_std[b] + 1.0                                        # doc §4.4: +1 guards std=0
    per_sample = ((area_pred - tgt) / sd).pow(2) * valid.float()
    return per_sample.sum() / valid.float().sum().clamp(min=1.0)


class MergedLoss(nn.Module):
    def __init__(self, w_bg=0.1, w_men=1.5, eps=1e-6):
        super().__init__(); self.w_bg, self.w_men, self.eps = w_bg, w_men, eps

    def forward(self, logits, binary_masks):
        probs = torch.softmax(logits, dim=1)
        p_bg = probs[:, 0]; p_men = probs[:, 1] + probs[:, 2]
        gt_men = (binary_masks > 0).float(); gt_bg = (binary_masks == 0).float()
        ce = -(self.w_bg * gt_bg * torch.log(p_bg.clamp(min=self.eps)) +
               self.w_men * gt_men * torch.log(p_men.clamp(min=self.eps))).mean()
        tp = (p_men * gt_men).sum()
        dice = 1.0 - (2 * tp + self.eps) / (p_men.sum() + gt_men.sum() + self.eps)
        return ce + dice


class SoftDiceLoss(nn.Module):
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


def run_epoch(model, real_loader, fake_loader, optimizer, merged_loss_fn, fake_loss_fn,
              device, lam, prior_f, prior_r, train=True):
    model.train() if train else model.eval()
    total, dice_sum, dice_fake_sum, area_sum, base_sum, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    real_iter = iter(real_loader)

    with torch.set_grad_enabled(train):
        for fake_batch in fake_loader:
            xf, yf = fake_batch["image"].to(device), fake_batch["mask"].to(device)
            bf = fake_batch["bin"].to(device)
            logits_f = model(xf)
            base_f = fake_loss_fn(logits_f, yf)
            area_f = area_loss(logits_f, bf, *prior_f)

            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader); real_batch = next(real_iter)
            xr, yr = real_batch["image"].to(device), real_batch["mask"].to(device)
            br = real_batch["bin"].to(device)
            logits_r = model(xr)
            base_r = merged_loss_fn(logits_r, yr)
            area_r = area_loss(logits_r, br, *prior_r)

            base = base_f + base_r
            area = area_f + area_r
            loss = base + lam * area

            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            total += loss.item(); base_sum += base.item(); area_sum += float(area)
            dice_sum += dice_binary(logits_r.argmax(1), yr)
            dice_fake_sum += dice_binary(logits_f.argmax(1), yf)
            n += 1

    n = max(n, 1)
    return total / n, dice_sum / n, dice_fake_sum / n, area_sum / n, base_sum / n


@torch.no_grad()
def eval_real_per_patient(model, dataset, device, batch_size=8):
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
    try:
        import nibabel as nib
    except ImportError:
        log.warning("nibabel not available — skipping val prediction dump"); return
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
                x[b, 2].cpu().numpy().astype(np.float32),
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


def load_prior(path, device, name):
    with open(path) as fh:
        blob = json.load(fh)
    m = blob["meta"]
    a_mean = torch.tensor(blob["area_mean"], dtype=torch.float32, device=device)
    a_std = torch.tensor(blob["area_std"], dtype=torch.float32, device=device)
    log.info(f"  {name} prior: {m['bins']} bins, {m['n_patients_used']} patients, "
             f"{m['n_slices']} slices, {m['n_skipped_single_span']} skipped (single span)")
    log.info(f"    area mean by bin (outer->notch): "
             f"{[round(float(v)) for v in a_mean.tolist()]}")
    return (a_mean, a_std), blob["bin_map"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--shape_prior_fake", required=True)
    ap.add_argument("--shape_prior_real", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lam", type=float, default=0.01, help="doc §4.4 starts at 0.01")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  LR: {args.lr} -> 1e-8  |  D: position-conditioned area prior, lam={args.lam}")

    prior_f, binmap_f = load_prior(args.shape_prior_fake, device, "fake")
    prior_r, binmap_r = load_prior(args.shape_prior_real, device, "real")

    model = build_sequence_model(args.pretrained_ckpt, device,
                                 pretrained_n_classes=PRETRAINED_N_CLASSES, n_classes=N_CLASSES)
    log.info(f"Loaded DESS baseline from {args.pretrained_ckpt}, wrapped with SequenceUNet")

    fake_train = BinnedSequenceMeniscusDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), binmap_f, augment=True)
    fake_val = BinnedSequenceMeniscusDataset(
        os.path.join(args.fake_data_root, "val", "images"),
        os.path.join(args.fake_data_root, "val", "masks"), binmap_f, augment=False)
    real_train = BinnedSequenceRealPDDataset(
        os.path.join(args.real_data_root, "train", "images"),
        os.path.join(args.real_data_root, "train", "masks"), binmap_r, augment=True)
    real_val = BinnedSequenceRealPDDataset(
        os.path.join(args.real_data_root, "val", "images"),
        os.path.join(args.real_data_root, "val", "masks"), binmap_r, augment=False)

    log.info(f"Fake PD — train: {len(fake_train)} slices  |  val: {len(fake_val)} slices")
    log.info(f"Real PD — train: {len(real_train)} slices  |  val: {len(real_val)} slices")
    cov_f = np.mean([fake_train[i]["bin"] >= 0 for i in range(0, min(len(fake_train), 400), 7)])
    cov_r = np.mean([real_train[i]["bin"] >= 0 for i in range(0, min(len(real_train), 400), 7)])
    log.info(f"  samples WITH a bin (the prior only speaks on these): "
             f"fake {cov_f:.1%}, real {cov_r:.1%}")

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False, num_workers=args.num_workers)
    real_train_loader = DataLoader(real_train, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    real_val_loader = DataLoader(real_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = lambda logits, y: (nn.CrossEntropyLoss(weight=weights)(logits, y) + SoftDiceLoss()(logits, y))
    merged_loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices})
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va", "dice_real_tr", "dice_real_va", "gap",
                   "dice_fake_tr", "dice_fake_va", "dice_va_perpatient_mean"]
                  + [f"dice_va_{p}" for p in val_pids] + ["area_tr", "area_va", "area_frac"])

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td, tdf, area_tr, base_tr = run_epoch(model, real_train_loader, fake_train_loader, optimizer,
                                                  merged_loss_fn, fake_loss_fn, device, args.lam,
                                                  prior_f, prior_r, train=True)
        vl, vd, vdf, area_va, _ = run_epoch(model, real_val_loader, fake_val_loader, optimizer,
                                            merged_loss_fn, fake_loss_fn, device, args.lam,
                                            prior_f, prior_r, train=False)
        scheduler.step()

        pp = eval_real_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        pp_str = "  ".join(f"{p}={pp.get(p, float('nan')):.4f}" for p in val_pids)
        area_frac = (args.lam * area_tr) / max(base_tr, 1e-9)

        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl:.4f}  |  "
                 f"dice_real tr={td:.4f} va={vd:.4f} gap={td - vd:+.4f}  |  "
                 f"dice_fake tr={tdf:.4f} va={vdf:.4f}  |  area tr={area_tr:.3f} frac={area_frac:.3f}")
        log.info(f"           per-patient val (clean pass): mean={pp_mean:.4f}   {pp_str}")
        if epoch == 0 and area_frac > 0.10:
            log.warning(f"area term is {area_frac:.1%} of base loss at epoch 0; doc §4.4 wants "
                        f"< 10%. Consider lowering --lam.")

        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}", f"{td:.5f}", f"{vd:.5f}", f"{td - vd:+.5f}",
                       f"{tdf:.5f}", f"{vdf:.5f}", f"{pp_mean:.5f}"]
                      + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids]
                      + [f"{area_tr:.6f}", f"{area_va:.6f}", f"{area_frac:.4f}"])
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
