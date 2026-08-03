"""
finetune_meniscus_v3_rotation.py
=================================
Same as finetune_meniscus_v2.py but uses dataset_2_5d_v3 which adds
random rotation ±15° on top of v2 augmentation (hflip, vflip, brightness, noise).

Change vs v2:
  + dataset_2_5d_v3 (rotation aug)
  Loss, LR schedule, class weights: identical to v2.

Purpose: test whether rotation augmentation alone improves real PD Dice.
"""
import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v3 import Meniscus2_5DDataset  # v3: adds rotation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3
PRETRAINED_N_CLASSES = 5


def build_model(pretrained_ckpt=None, device="cpu"):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained_ckpt is None else None,
        in_channels=3,
        classes=PRETRAINED_N_CLASSES,
    )

    if pretrained_ckpt:
        state = torch.load(pretrained_ckpt, map_location=device)
        model.load_state_dict(state)
        log.info(f"Loaded pretrained weights from {pretrained_ckpt}")

    old_head = model.segmentation_head[0]
    new_head = nn.Conv2d(
        old_head.in_channels, N_CLASSES,
        kernel_size=old_head.kernel_size,
        stride=old_head.stride,
        padding=old_head.padding,
    )
    model.segmentation_head[0] = new_head
    log.info(f"Replaced segmentation head: {PRETRAINED_N_CLASSES} -> {N_CLASSES} classes")

    return model.to(device)


def dice_per_class(preds, targets, n_classes=N_CLASSES, eps=1e-6):
    dices = []
    for c in range(n_classes):
        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dices.append(((2 * inter + eps) / (union + eps)).item())
    return dices


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=N_CLASSES, eps=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(1, self.n_classes):
            p = probs[:, c]
            t = (targets == c).float()
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            loss += 1.0 - (2.0 * inter + self.eps) / (union + self.eps)
        return loss / (self.n_classes - 1)


def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    dice_sums = [0.0] * N_CLASSES
    n_batches = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)

            preds = model(x)
            loss = loss_fn(preds, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pred_cls = preds.argmax(dim=1)
            dices = dice_per_class(pred_cls, y)
            for c in range(N_CLASSES):
                dice_sums[c] += dices[c]
            n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    mean_dices = [d / max(n_batches, 1) for d in dice_sums]
    return mean_loss, mean_dices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs",      type=int,   default=50)
    ap.add_argument("--batch_size",  type=int,   default=8)
    ap.add_argument("--lr",          type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int,   default=4)
    ap.add_argument("--patience",    type=int,   default=10)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info("Augmentation: hflip + vflip + rotation±15° + brightness + noise  [v3 dataset]")

    model = build_model(args.pretrained_ckpt, device)

    train_ds = Meniscus2_5DDataset(
        os.path.join(args.data_root, "train", "images"),
        os.path.join(args.data_root, "train", "masks"),
        augment=True,
    )
    val_ds = Meniscus2_5DDataset(
        os.path.join(args.data_root, "val", "images"),
        os.path.join(args.data_root, "val", "masks"),
        augment=False,
    )
    log.info(f"Train slices: {len(train_ds)}  Val slices: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    ce_loss   = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = SoftDiceLoss(n_classes=N_CLASSES)

    def loss_fn(logits, targets):
        return ce_loss(logits, targets) + dice_loss(logits, targets)

    log.info(f"Loss: weighted CE + SoftDice  (class_weights={args.class_weights})")
    log.info(f"LR: CosineAnnealing {args.lr} → 1e-7 over {args.epochs} epochs, patience={args.patience}")

    best_val_dice = -1.0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_dices = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss,   val_dices   = run_epoch(model, val_loader,   optimizer, loss_fn, device, train=False)

        scheduler.step()

        mean_train_dice    = (train_dices[1] + train_dices[2]) / 2
        mean_meniscus_dice = (val_dices[1]   + val_dices[2])   / 2

        log.info(
            f"Epoch {epoch:03d}  lr={current_lr:.2e}  "
            f"train_loss={train_loss:.4f}  "
            f"train_dice[lat={train_dices[1]:.3f} med={train_dices[2]:.3f}] mean={mean_train_dice:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_dice[lat={val_dices[1]:.3f} med={val_dices[2]:.3f}] mean={mean_meniscus_dice:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        if mean_meniscus_dice > best_val_dice:
            best_val_dice = mean_meniscus_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best val meniscus Dice: {best_val_dice:.4f} -- saved ckpt_best.pth")
        else:
            epochs_no_improve += 1
            log.info(f"  No improvement for {epochs_no_improve}/{args.patience} epochs")
            if args.patience > 0 and epochs_no_improve >= args.patience:
                log.info(f"  Early stopping at epoch {epoch}. Best val Dice: {best_val_dice:.4f}")
                break

    log.info(f"Fine-tuning complete. Best mean meniscus Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
