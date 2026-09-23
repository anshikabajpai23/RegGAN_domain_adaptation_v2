"""
finetune_meniscus_v7_replica_d.py
====================================
Phase 5, D — position-conditioned shape (area) prior, on the PLAIN 2.5D
replica base (independent of crop+sequence). Diffs directly against
run_v7_replica_seed42, exactly like P1/P2/P4 did.

BASE: everything from finetune_meniscus_v7_replica.py is copied verbatim
(build_model(), the plain smp.Unet architecture, eval/dump helpers) except the
loss, which gets + lam * L_area, and the two dataset classes, which are thin
subclasses adding a `bin` field (not a rewrite).

REUSES area_loss() and calibrate_lambda() from finetune_meniscus_v7_crop_seq_d.py
by import rather than duplicating them: both are already architecture-agnostic
(they call model(x) with a single positional argument and operate on logits
only), so there is no reason to fork them for the plain-Unet case. Only this
file's model construction, dataset wiring, and run_epoch differ.

THE SHAPE PRIOR MUST BE REBUILT ON UNCROPPED MASKS for this experiment
(scripts/build_shape_prior.py --data_root segmentation_data_v2 / real_pd_seg_data_v7,
NOT the _crop directories) -- the crop+sequence D run used a prior built from
CROPPED masks specifically because that model predicts in cropped (magnified)
space. This model predicts in the ORIGINAL uncropped space, so it needs its own,
separately-built prior or every area target would be off by the crop's ~2.5x
magnification factor.

lam is CALIBRATED IN-JOB (never hardcoded) -- this is the exact bug that
collapsed the first crop+sequence D attempt (see docs/phase5_implementation_ideas.md
and the fix history in finetune_meniscus_v7_crop_seq_d.py). Do not reintroduce
a hardcoded default here.
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
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v2 import Meniscus2_5DDataset
from dataset_2_5d_realpd import RealPDDataset
from finetune_meniscus_v7_crop_seq_d import area_loss, calibrate_lambda  # reused, not reimplemented

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


def build_model(ckpt_path, device):
    """Verbatim from finetune_meniscus_v7_replica.py."""
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=PRETRAINED_N_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    log.info(f"Loaded DESS baseline from {ckpt_path}")
    old_head = model.segmentation_head[0]
    model.segmentation_head[0] = nn.Conv2d(
        old_head.in_channels, N_CLASSES,
        kernel_size=old_head.kernel_size, stride=old_head.stride, padding=old_head.padding)
    return model.to(device)


class BinnedMeniscus2_5DDataset(Meniscus2_5DDataset):
    def __init__(self, img_root, mask_root, bin_map, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.bin_map = bin_map or {}

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["bin"] = int(self.bin_map.get(item["patient_id"], {}).get(str(item["slice_idx"]), NO_BIN))
        return item


class BinnedRealPDDataset(RealPDDataset):
    def __init__(self, img_root, mask_root, bin_map, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.bin_map = bin_map or {}

    def __getitem__(self, i):
        item = super().__getitem__(i)
        pid, sidx = self.slices[i].rsplit("_", 1)
        item["bin"] = int(self.bin_map.get(pid, {}).get(str(int(sidx)), NO_BIN))
        return item


class MergedLoss(nn.Module):
    def __init__(self, w_bg=0.1, w_men=1.5, eps=1e-6):
        super().__init__(); self.w_bg, self.w_men, self.eps = w_bg, w_men, eps

    def forward(self, logits, binary_masks):
        probs = torch.softmax(logits, dim=1)
        p_bg = probs[:, 0]; p_men = probs[:, 1] + probs[:, 2]
        gt_men = (binary_masks > 0).float(); gt_bg = (binary_masks == 0).float()
        ce = -(self.w_bg * gt_bg * torch.log(p_bg.clamp(self.eps)) +
               self.w_men * gt_men * torch.log(p_men.clamp(self.eps))).mean()
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


def fake_loss_fn_factory(class_weights):
    return lambda logits, y: (nn.CrossEntropyLoss(weight=class_weights)(logits, y) + SoftDiceLoss()(logits, y))


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


def load_prior(path, device, name):
    with open(path) as fh:
        blob = json.load(fh)
    m = blob["meta"]
    a_mean = torch.tensor(blob["area_mean"], dtype=torch.float32, device=device)
    a_std = torch.tensor(blob["area_std"], dtype=torch.float32, device=device)
    log.info(f"  {name} prior: {m['bins']} bins, {m['n_patients_used']} patients, "
             f"{m['n_slices']} slices, {m['n_skipped_single_span']} skipped (single span), "
             f"built from {m['data_root']}")
    return (a_mean, a_std), blob["bin_map"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--shape_prior_fake", required=True,
                    help="MUST be built from UNCROPPED masks -- see module docstring")
    ap.add_argument("--shape_prior_real", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lam", type=float, default=None, help="explicit; otherwise calibrated")
    ap.add_argument("--lam_target_frac", type=float, default=0.07)
    ap.add_argument("--calib_batches", type=int, default=20)
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
    log.info(f"Device: {device}  |  LR: {args.lr} -> 1e-8  |  D: area prior, PLAIN 2.5D, uncropped")

    prior_f, binmap_f = load_prior(args.shape_prior_fake, device, "fake")
    prior_r, binmap_r = load_prior(args.shape_prior_real, device, "real")

    model = build_model(args.pretrained_ckpt, device)

    fake_train = BinnedMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), binmap_f, augment=True)
    fake_val = BinnedMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "val", "images"),
        os.path.join(args.fake_data_root, "val", "masks"), binmap_f, augment=False)
    real_train = BinnedRealPDDataset(
        os.path.join(args.real_data_root, "train", "images"),
        os.path.join(args.real_data_root, "train", "masks"), binmap_r, augment=True)
    real_val = BinnedRealPDDataset(
        os.path.join(args.real_data_root, "val", "images"),
        os.path.join(args.real_data_root, "val", "masks"), binmap_r, augment=False)

    log.info(f"Fake PD — train: {len(fake_train)} slices  |  val: {len(fake_val)} slices")
    log.info(f"Real PD — train: {len(real_train)} slices  |  val: {len(real_val)} slices")
    cov_f = np.mean([fake_train[i]["bin"] >= 0 for i in range(0, min(len(fake_train), 400), 7)])
    cov_r = np.mean([real_train[i]["bin"] >= 0 for i in range(0, min(len(real_train), 400), 7)])
    log.info(f"  samples WITH a bin: fake {cov_f:.1%}, real {cov_r:.1%}")

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False, num_workers=args.num_workers)
    real_train_loader = DataLoader(real_train, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    real_val_loader = DataLoader(real_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = fake_loss_fn_factory(class_weights)
    merged_loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    if args.lam is None:
        lam, base_m, area_m = calibrate_lambda(model, real_train_loader, fake_train_loader,
                                               merged_loss_fn, fake_loss_fn, prior_f, prior_r,
                                               device, args.lam_target_frac, args.calib_batches)
        log.info(f"Calibrated lam={lam:.6f} over {args.calib_batches} batches "
                 f"(base={base_m:.4f}, area={area_m:.4f}, target {args.lam_target_frac:.0%} of base)")
    else:
        lam = args.lam
        log.info(f"Using explicit lam={lam:.6f}")

    with open(os.path.join(args.out_dir, "d_config.json"), "w") as fh:
        json.dump({"lam": lam, "lam_target_frac": args.lam_target_frac, "seed": args.seed}, fh, indent=2)

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
                                                  merged_loss_fn, fake_loss_fn, device, lam,
                                                  prior_f, prior_r, train=True)
        vl, vd, vdf, area_va, _ = run_epoch(model, real_val_loader, fake_val_loader, optimizer,
                                            merged_loss_fn, fake_loss_fn, device, lam,
                                            prior_f, prior_r, train=False)
        scheduler.step()

        pp = eval_real_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        pp_str = "  ".join(f"{p}={pp.get(p, float('nan')):.4f}" for p in val_pids)
        area_frac = (lam * area_tr) / max(base_tr, 1e-9)

        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl:.4f}  |  "
                 f"dice_real tr={td:.4f} va={vd:.4f} gap={td - vd:+.4f}  |  "
                 f"dice_fake tr={tdf:.4f} va={vdf:.4f}  |  area tr={area_tr:.3f} frac={area_frac:.3f}")
        log.info(f"           per-patient val (clean pass): mean={pp_mean:.4f}   {pp_str}")
        if epoch == 0 and area_frac > 0.10:
            log.warning(f"area term is {area_frac:.1%} of base loss at epoch 0; consider lowering --lam.")

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
