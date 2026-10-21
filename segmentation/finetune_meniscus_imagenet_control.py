"""
finetune_meniscus_imagenet_control.py
=======================================
Baseline: ImageNet-init control (docs/reggan_paper_planning.md, "no
ImageNet-init control" gap). Tests whether the DESS-pretraining + fake-PD
translation pipeline is earning its complexity at all, by training the
SAME architecture on ONLY the 15 real PD labeled patients -- no fake-PD
stream, no DESS baseline checkpoint, encoder initialized from ImageNet
weights instead.

If this control matches or beats the replica (0.781 Dice), the entire
DESS-pretrain + RegGAN-translate + mixed-training pipeline is not earning
its complexity for this label budget. If it's clearly worse, that's the
positive case for the pipeline.

Loss is byte-identical to the real-PD branch of finetune_meniscus_v7_replica.py
(MergedLoss, w_bg=0.1, w_men=1.5) -- only the fake-PD stream and the DESS
pretrained checkpoint are removed. Same seed, augmentation (aug_mode="original",
matching the replica's real branch), 50/10 schedule, and reporting format.

Two deliberate differences, both to avoid an unfair (undertrained) control:
  - epoch = 1189 real batches sampled with replacement, the same number of
    real-PD gradient steps per epoch the replica takes
  - LR is swept (1e-4 / 3e-4) instead of the replica's 1e-5, because the
    decoder and head are randomly initialized here

Usage:
    python segmentation/finetune_meniscus_imagenet_control.py \
        --real_data_root /path/to/real_pd_seg_data_v7 \
        --out_dir        /path/to/segmentation_runs/run_imagenet_control
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
from dataset_2_5d_realpd import RealPDDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3
PD_SPACING = (3.6, 0.39, 0.39)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"Seed fixed: {seed} (cudnn.deterministic=True)")


def build_model(device):
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=N_CLASSES)
    log.info("Built resnet34-UNet with ImageNet encoder weights "
             "(NO DESS pretraining, NO fake-PD stream)")
    return model.to(device)


class MergedLoss(nn.Module):
    """Byte-identical to finetune_meniscus_v7_replica.py's real-branch loss."""
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


def dice_binary(preds, targets, eps=1e-6):
    pred_men = (preds > 0).float(); gt_men = (targets > 0).float()
    tp = (pred_men * gt_men).sum()
    return float((2 * tp + eps) / (pred_men.sum() + gt_men.sum() + eps))


def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss, dice_sum, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            dice_sum += dice_binary(logits.argmax(1), y)
            n += 1
    return total_loss / max(n, 1), dice_sum / max(n, 1)


@torch.no_grad()
def eval_per_patient(model, dataset, device, batch_size=8):
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
    log.info(f"  per-slice Dice CSV -> {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_data_root", required=True,
                    help="real_pd_seg_data_v7/ (15 labeled patients) -- SAME dir "
                         "the replica uses, no fake-PD stream at all here")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="Decoder and head start from random weights, so the "
                         "replica's fine-tuning LR (1e-5) undertrains; sweep 1e-4 / 3e-4")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--real_batches_per_epoch", type=int, default=1189,
                    help="Replica sees one real batch per fake batch, 1189 per epoch; "
                         "match that so real-PD gradient steps are equal")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  LR: {args.lr}  |  ImageNet-init control, real-PD-only")

    model = build_model(device)

    real_train = RealPDDataset(
        os.path.join(args.real_data_root, "train", "images"),
        os.path.join(args.real_data_root, "train", "masks"), augment=True)
    real_val = RealPDDataset(
        os.path.join(args.real_data_root, "val", "images"),
        os.path.join(args.real_data_root, "val", "masks"), augment=False)

    real_train_scans = len({s.rsplit("_", 1)[0] for s in real_train.slices})
    real_val_scans = len({s.rsplit("_", 1)[0] for s in real_val.slices})
    log.info(f"Real PD — train: {real_train_scans} scans, {len(real_train)} slices  |  "
             f"val: {real_val_scans} scans, {len(real_val)} slices")

    train_sampler = RandomSampler(real_train, replacement=True,
                                  num_samples=args.real_batches_per_epoch * args.batch_size)
    train_loader = DataLoader(real_train, args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, drop_last=False)
    log.info(f"Epoch = {args.real_batches_per_epoch} real batches (sampled with replacement), "
             f"matching the replica's real-PD steps per epoch")
    val_loader = DataLoader(real_val, args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-8)
    loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices})
    hist_path = os.path.join(args.out_dir, "history.csv")
    hist_fh = open(hist_path, "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va", "dice_tr", "dice_va", "gap",
                   "dice_va_perpatient_mean"] + [f"dice_va_{p}" for p in val_pids])

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        vl, vd = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        scheduler.step()

        pp = eval_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        pp_str = "  ".join(f"{p}={pp.get(p, float('nan')):.4f}" for p in val_pids)

        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl:.4f}  |  "
                 f"dice tr={td:.4f} va={vd:.4f} gap={td - vd:+.4f}")
        log.info(f"           per-patient val (clean pass): mean={pp_mean:.4f}   {pp_str}")

        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}",
                       f"{td:.5f}", f"{vd:.5f}", f"{td - vd:+.5f}", f"{pp_mean:.5f}"]
                      + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids])
        hist_fh.flush()

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        if vd > best_val_dice:
            best_val_dice, no_improve = vd, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best: {best_val_dice:.4f} (epoch {epoch}) — dumping val predictions")
            dump_val_predictions(model, real_val, device,
                                 os.path.join(args.out_dir, "val_predictions"), args.batch_size)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}. Best: {best_val_dice:.4f}")
                break

    hist_fh.close()
    log.info(f"Done. Best val Dice: {best_val_dice:.4f}")
    log.info(f"Compare vs run_v7_replica_seed42 (0.781) -- if this control matches or "
             f"beats it, the DESS-pretrain + fake-PD pipeline isn't earning its "
             f"complexity at this label budget.")


if __name__ == "__main__":
    main()
