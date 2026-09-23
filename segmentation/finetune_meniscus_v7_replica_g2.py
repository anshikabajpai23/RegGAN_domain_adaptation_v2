"""
finetune_meniscus_v7_replica_g2.py
====================================
Phase 5, G2 — region-darker-than-rim loss, on the PLAIN 2.5D replica base
(independent of crop+sequence). Diffs directly against run_v7_replica_seed42,
exactly like P1/P2/P4 did.

BASE: everything from finetune_meniscus_v7_replica.py is copied verbatim
(build_model(), the plain smp.Unet architecture, datasets, eval/dump helpers)
except the loss, which gets + lam * rim_loss.

REUSES rim_loss() and meniscus_prob() from rim_loss.py by import -- both are
already architecture-agnostic (rim_loss needs only a probability map and a
centre-slice intensity map, no model-specific args), and the pre-check that
gates this experiment (precheck_rim_darkness.py) was ALREADY run against the
UNCROPPED data (real_pd_seg_data_v7 / segmentation_data_v2 -- see that script's
own --real_root/--fake_root defaults), so it applies here without re-running:
real PD held on 1.000 of slices, fake PD on 0.964, both far above the 0.80 gate.

lam is CALIBRATED IN-JOB, same pattern as the crop+sequence G2 script and for
the same reason D's hardcoded default collapsed a run: a hardcoded value would
be wrong at this loss's actual scale, which is only known by measuring it.
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
from rim_loss import rim_loss, meniscus_prob   # reused, not reimplemented

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3
PRETRAINED_N_CLASSES = 5
PD_SPACING = (3.6, 0.39, 0.39)
CENTRE = 1   # channel 1 of the [prev, curr, next] 2.5D stack is the centre slice


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
              device, lam, margin, rim_k, train=True):
    model.train() if train else model.eval()
    total, dice_sum, dice_fake_sum, rim_sum, base_sum, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    real_iter = iter(real_loader)

    with torch.set_grad_enabled(train):
        for fake_batch in fake_loader:
            xf, yf = fake_batch["image"].to(device), fake_batch["mask"].to(device)
            logits_f = model(xf)
            base_f = fake_loss_fn(logits_f, yf)
            rim_f = rim_loss(meniscus_prob(logits_f), xf[:, CENTRE], k=rim_k, margin=margin)

            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader); real_batch = next(real_iter)
            xr, yr = real_batch["image"].to(device), real_batch["mask"].to(device)
            logits_r = model(xr)
            base_r = merged_loss_fn(logits_r, yr)
            rim_r = rim_loss(meniscus_prob(logits_r), xr[:, CENTRE], k=rim_k, margin=margin)

            base = base_f + base_r
            rim = rim_f + rim_r
            loss = base + lam * rim

            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            total += loss.item(); base_sum += base.item(); rim_sum += rim.item()
            dice_sum += dice_binary(logits_r.argmax(1), yr)
            dice_fake_sum += dice_binary(logits_f.argmax(1), yf)
            n += 1

    n = max(n, 1)
    return (total / n, dice_sum / n, dice_fake_sum / n, rim_sum / n, base_sum / n)


@torch.no_grad()
def calibrate_lambda(model, real_loader, fake_loader, merged_loss_fn, fake_loss_fn,
                     device, margin, rim_k, target_frac, n_batches):
    model.eval()
    real_iter = iter(real_loader)
    base_tot, rim_tot, n = 0.0, 0.0, 0
    for fake_batch in fake_loader:
        if n >= n_batches:
            break
        xf, yf = fake_batch["image"].to(device), fake_batch["mask"].to(device)
        logits_f = model(xf)
        try:
            real_batch = next(real_iter)
        except StopIteration:
            real_iter = iter(real_loader); real_batch = next(real_iter)
        xr, yr = real_batch["image"].to(device), real_batch["mask"].to(device)
        logits_r = model(xr)

        base_tot += (fake_loss_fn(logits_f, yf) + merged_loss_fn(logits_r, yr)).item()
        rim_tot += (rim_loss(meniscus_prob(logits_f), xf[:, CENTRE], k=rim_k, margin=margin)
                    + rim_loss(meniscus_prob(logits_r), xr[:, CENTRE], k=rim_k, margin=margin)).item()
        n += 1

    if n == 0 or rim_tot <= 1e-9:
        log.warning("Rim loss is ~0 at init: falling back to lam=1.0.")
        return 1.0, base_tot / max(n, 1), rim_tot / max(n, 1)
    base_mean, rim_mean = base_tot / n, rim_tot / n
    return target_frac * base_mean / rim_mean, base_mean, rim_mean


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
                x[b, CENTRE].cpu().numpy().astype(np.float32),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--margin", type=float, default=0.0293,
                    help="normalised intensity; 25%% of the measured fake-PD gap")
    ap.add_argument("--rim_k", type=int, default=5)
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
    log.info(f"Device: {device}  |  LR: {args.lr} -> 1e-8  |  G2: rim loss, PLAIN 2.5D, uncropped")
    log.info(f"  margin={args.margin}  rim_k={args.rim_k} px")

    model = build_model(args.pretrained_ckpt, device)

    fake_train = Meniscus2_5DDataset(os.path.join(args.fake_data_root, "train", "images"),
                                     os.path.join(args.fake_data_root, "train", "masks"), augment=True)
    fake_val = Meniscus2_5DDataset(os.path.join(args.fake_data_root, "val", "images"),
                                   os.path.join(args.fake_data_root, "val", "masks"), augment=False)
    real_train = RealPDDataset(os.path.join(args.real_data_root, "train", "images"),
                               os.path.join(args.real_data_root, "train", "masks"), augment=True)
    real_val = RealPDDataset(os.path.join(args.real_data_root, "val", "images"),
                             os.path.join(args.real_data_root, "val", "masks"), augment=False)
    log.info(f"Fake PD — train: {len(fake_train)} slices  |  val: {len(fake_val)} slices")
    log.info(f"Real PD — train: {len(real_train)} slices  |  val: {len(real_val)} slices")

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False, num_workers=args.num_workers)
    real_train_loader = DataLoader(real_train, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    real_val_loader = DataLoader(real_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = lambda logits, y: (nn.CrossEntropyLoss(weight=weights)(logits, y) + SoftDiceLoss()(logits, y))
    merged_loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    if args.lam is None:
        lam, base_m, rim_m = calibrate_lambda(model, real_train_loader, fake_train_loader,
                                              merged_loss_fn, fake_loss_fn, device,
                                              args.margin, args.rim_k,
                                              args.lam_target_frac, args.calib_batches)
        log.info(f"Calibrated lam={lam:.4f} over {args.calib_batches} batches "
                 f"(base={base_m:.4f}, rim={rim_m:.4f}, target {args.lam_target_frac:.0%} of base)")
    else:
        lam = args.lam
        log.info(f"Using explicit lam={lam:.4f}")

    with open(os.path.join(args.out_dir, "g2_config.json"), "w") as fh:
        json.dump({"lam": lam, "margin": args.margin, "rim_k": args.rim_k,
                   "lam_target_frac": args.lam_target_frac, "seed": args.seed}, fh, indent=2)

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices})
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va", "dice_real_tr", "dice_real_va", "gap",
                   "dice_fake_tr", "dice_fake_va", "dice_va_perpatient_mean"]
                  + [f"dice_va_{p}" for p in val_pids] + ["rim_tr", "rim_va", "rim_frac"])

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td, tdf, rim_tr, base_tr = run_epoch(model, real_train_loader, fake_train_loader, optimizer,
                                                 merged_loss_fn, fake_loss_fn, device, lam,
                                                 args.margin, args.rim_k, train=True)
        vl, vd, vdf, rim_va, _ = run_epoch(model, real_val_loader, fake_val_loader, optimizer,
                                           merged_loss_fn, fake_loss_fn, device, lam,
                                           args.margin, args.rim_k, train=False)
        scheduler.step()

        pp = eval_real_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        pp_str = "  ".join(f"{p}={pp.get(p, float('nan')):.4f}" for p in val_pids)
        rim_frac = (lam * rim_tr) / max(base_tr, 1e-9)

        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl:.4f}  |  "
                 f"dice_real tr={td:.4f} va={vd:.4f} gap={td - vd:+.4f}  |  "
                 f"dice_fake tr={tdf:.4f} va={vdf:.4f}  |  rim tr={rim_tr:.4f} va={rim_va:.4f} "
                 f"frac={rim_frac:.3f}")
        log.info(f"           per-patient val (clean pass): mean={pp_mean:.4f}   {pp_str}")

        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}", f"{td:.5f}", f"{vd:.5f}", f"{td - vd:+.5f}",
                       f"{tdf:.5f}", f"{vdf:.5f}", f"{pp_mean:.5f}"]
                      + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids]
                      + [f"{rim_tr:.6f}", f"{rim_va:.6f}", f"{rim_frac:.4f}"])
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
