"""
finetune_meniscus_v12_tversky.py
==================================
v12_tversky: Same as v7_mixed (DESS baseline, 15 real PD + fake PD)
             but replaces SoftDice with TverskyLoss(alpha=0.3, beta=0.7).

Tversky: T = TP / (TP + alpha*FP + beta*FN)
  alpha=0.3, beta=0.7 -> FN penalised 2.3x more than FP -> over-segment bias.

Key difference from v3_tversky (Dice=0.656):
  v3_tversky used fake PD only (no real PD labels).
  v12_tversky uses the full v7_mixed training recipe (15pt real PD + fake PD),
  which is what actually works. Fair ablation of Tversky on the best setup.

Baseline to beat: v7_mixed Dice=0.781 on 17pt cohort.
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
from dataset_2_5d_v2     import Meniscus2_5DDataset
from dataset_2_5d_realpd import RealPDDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES            = 3
PRETRAINED_N_CLASSES = 5


def build_model(ckpt_path, device):
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


class TverskyLoss(nn.Module):
    """Tversky loss for class-imbalanced segmentation.
    alpha=0.3, beta=0.7 penalises FN more than FP -> better recall on small structures.
    """
    def __init__(self, n_classes=N_CLASSES, alpha=0.3, beta=0.7, eps=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss  = 0.0
        for c in range(1, self.n_classes):
            p  = probs[:, c]
            t  = (targets == c).float()
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
            loss += 1.0 - tversky
        return loss / (self.n_classes - 1)


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


def dice_binary(preds, targets, eps=1e-6):
    pred_men = (preds > 0).float(); gt_men = (targets > 0).float()
    tp = (pred_men * gt_men).sum()
    return float((2*tp + eps) / (pred_men.sum() + gt_men.sum() + eps))


def run_epoch(model, real_loader, fake_loader, optimizer,
              merged_loss_fn, fake_loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss, dice_sum, n = 0.0, 0.0, 0
    real_iter = iter(real_loader)

    with torch.set_grad_enabled(train):
        for fake_batch in fake_loader:
            xf, yf = fake_batch["image"].to(device), fake_batch["mask"].to(device)
            loss_f  = fake_loss_fn(model(xf), yf)

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

            total_loss += loss.item()
            dice_sum   += dice_binary(logits_r.argmax(1), yr)
            n += 1

    return total_loss / max(n, 1), dice_sum / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root",  required=True)
    ap.add_argument("--real_data_root",  required=True)
    ap.add_argument("--out_dir",         required=True)
    ap.add_argument("--epochs",          type=int,   default=50)
    ap.add_argument("--batch_size",      type=int,   default=8)
    ap.add_argument("--lr",              type=float, default=1e-5)
    ap.add_argument("--patience",        type=int,   default=10)
    ap.add_argument("--num_workers",     type=int,   default=4)
    ap.add_argument("--class_weights",   type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--tversky_alpha",   type=float, default=0.3)
    ap.add_argument("--tversky_beta",    type=float, default=0.7)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  LR: {args.lr}  |  "
             f"Tversky alpha={args.tversky_alpha} beta={args.tversky_beta}  |  "
             f"Data: mixed real+fake PD (DESS base)")

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

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    fake_val_loader   = DataLoader(fake_val,   args.batch_size, shuffle=False, num_workers=args.num_workers)
    real_train_loader = DataLoader(real_train, args.batch_size, shuffle=True,  num_workers=args.num_workers, drop_last=False)
    real_val_loader   = DataLoader(real_val,   args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    weights      = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    tversky_loss = TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta)
    fake_loss_fn = lambda logits, y: (nn.CrossEntropyLoss(weight=weights)(logits, y) +
                                      tversky_loss(logits, y))
    merged_loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td = run_epoch(model, real_train_loader, fake_train_loader,
                           optimizer, merged_loss_fn, fake_loss_fn, device, train=True)
        vl, vd = run_epoch(model, real_val_loader,   fake_val_loader,
                           optimizer, merged_loss_fn, fake_loss_fn, device, train=False)
        scheduler.step()
        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  train={tl:.4f}  val={vl:.4f}  val_dice={vd:.4f}")
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
