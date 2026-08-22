"""
finetune_meniscus_v5_realPD.py
================================
Fine-tune from 2.5D DESS baseline (baseline_best_model.pth, 5-class) on 10 real PD
patients (D2, binary masks). Head replaced 5→3 class, encoder kept.

Difference vs v4_realPD: starts from DESS baseline, NOT run_002_v2.
Baseline to beat: run_002_v2 Dice=0.6854
"""
import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_realpd import RealPDDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES           = 3
PRETRAINED_N_CLASSES = 5


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
        ce = -(self.w_bg * gt_bg  * torch.log(p_bg.clamp(self.eps)) +
               self.w_men * gt_men * torch.log(p_men.clamp(self.eps))).mean()
        tp   = (p_men * gt_men).sum()
        dice = 1.0 - (2*tp + self.eps) / (p_men.sum() + gt_men.sum() + self.eps)
        return ce + dice


def dice_binary(preds, targets, eps=1e-6):
    pred_men = (preds > 0).float()
    gt_men   = (targets > 0).float()
    tp = (pred_men * gt_men).sum()
    return float((2*tp + eps) / (pred_men.sum() + gt_men.sum() + eps))


def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss, dice_sum, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            logits = model(x)
            loss   = loss_fn(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            dice_sum   += dice_binary(logits.argmax(1), y)
            n += 1
    return total_loss / max(n, 1), dice_sum / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True,
                    help="baseline_best_model.pth (5-class DESS model)")
    ap.add_argument("--data_root",  required=True,
                    help="real_pd_seg_data/ directory")
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--epochs",     type=int,   default=100)
    ap.add_argument("--batch_size", type=int,   default=4)
    ap.add_argument("--lr",         type=float, default=1e-5)
    ap.add_argument("--patience",   type=int,   default=15)
    ap.add_argument("--num_workers",type=int,   default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  LR: {args.lr} → 1e-8  |  Data: real PD only (DESS base)")

    model   = build_model(args.pretrained_ckpt, device)
    loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    train_ds = RealPDDataset(os.path.join(args.data_root, "train", "images"),
                              os.path.join(args.data_root, "train", "masks"), augment=True)
    val_ds   = RealPDDataset(os.path.join(args.data_root, "val", "images"),
                              os.path.join(args.data_root, "val", "masks"), augment=False)
    train_scans = len({s.rsplit("_", 1)[0] for s in train_ds.slices})
    val_scans   = len({s.rsplit("_", 1)[0] for s in val_ds.slices})
    log.info(f"Train — {train_scans} scans, {len(train_ds)} slices  |  "
             f"Val — {val_scans} scans, {len(val_ds)} slices")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-8)

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        vl, vd = run_epoch(model, val_loader,   optimizer, loss_fn, device, train=False)
        scheduler.step()
        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  train={tl:.4f}  val={vl:.4f}  "
                 f"val_dice={vd:.4f}")
        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        if vd > best_val_dice:
            best_val_dice, no_improve = vd, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best: {best_val_dice:.4f}")
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}. Best: {best_val_dice:.4f}")
                break
    log.info(f"Done. Best val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
