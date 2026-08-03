"""
finetune_meniscus_v3_5slice.py
================================
Same as finetune_meniscus_v2.py but uses 5-slice 2.5D context (±2 slices)
instead of 3-slice (±1 slice).

Change vs v2:
  + dataset_2_5d_5slice: 5 input channels instead of 3
  + model in_channels=5, encoder_weights=None (ImageNet pretrained = 3ch only)
  Loss, LR: identical to v2.

Purpose: test whether more through-plane context improves real PD Dice.
Baseline to beat: run_002_v2 Dice=0.6854.
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
from dataset_2_5d_5slice import Meniscus2_5DDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3
PRETRAINED_N_CLASSES = 5
IN_CHANNELS = 5


def build_model(pretrained_ckpt=None, device="cpu"):
    # 5 input channels — cannot use 3-channel ImageNet pretrained weights directly
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # no imagenet pretrain for 5-ch input
        in_channels=IN_CHANNELS,
        classes=PRETRAINED_N_CLASSES,
    )
    if pretrained_ckpt:
        # Load what we can from 3-channel pretrained model (skip first conv mismatch)
        state_pretrained = torch.load(pretrained_ckpt, map_location=device)
        state_model = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in state_pretrained.items():
            if k in state_model and state_model[k].shape == v.shape:
                state_model[k] = v
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(state_model)
        log.info(f"Partial load from {pretrained_ckpt}: {loaded} layers loaded, {skipped} skipped (shape mismatch)")

    old_head = model.segmentation_head[0]
    model.segmentation_head[0] = nn.Conv2d(
        old_head.in_channels, N_CLASSES,
        kernel_size=old_head.kernel_size, stride=old_head.stride, padding=old_head.padding,
    )
    log.info(f"Model: ResNet34 UNet  |  in_channels={IN_CHANNELS}  |  classes: 5→3")
    return model.to(device)


def dice_per_class(preds, targets, n_classes=N_CLASSES, eps=1e-6):
    return [((2*(preds==c).float()*(targets==c).float()).sum() + eps) /
            ((preds==c).float().sum() + (targets==c).float().sum() + eps)
            for c in range(n_classes)]


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=N_CLASSES, eps=1e-6):
        super().__init__()
        self.n_classes = n_classes; self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(1, self.n_classes):
            p = probs[:, c]; t = (targets == c).float()
            loss += 1.0 - (2.0*(p*t).sum() + self.eps)/(p.sum()+t.sum()+self.eps)
        return loss / (self.n_classes - 1)


def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    model.train() if train else model.eval()
    total_loss, dice_sums, n = 0.0, [0.0]*N_CLASSES, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            for c, d in enumerate(dice_per_class(preds.argmax(1), y)):
                dice_sums[c] += d
            n += 1
    return total_loss/max(n,1), [d/max(n,1) for d in dice_sums]


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
    log.info(f"Device: {device}  |  5-slice context  |  Baseline to beat: 0.6854")

    model = build_model(args.pretrained_ckpt, device)

    train_ds = Meniscus2_5DDataset(os.path.join(args.data_root,"train","images"),
                                    os.path.join(args.data_root,"train","masks"), augment=True)
    val_ds   = Meniscus2_5DDataset(os.path.join(args.data_root,"val","images"),
                                    os.path.join(args.data_root,"val","masks"),   augment=False)
    log.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    weights   = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    ce_loss   = nn.CrossEntropyLoss(weight=weights)
    dice_loss = SoftDiceLoss()
    loss_fn   = lambda logits, y: ce_loss(logits, y) + dice_loss(logits, y)

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        vl, vd = run_epoch(model, val_loader,   optimizer, loss_fn, device, train=False)
        scheduler.step()
        mean_val = (vd[1]+vd[2])/2
        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  train={tl:.4f}  val={vl:.4f}  "
                 f"val_dice[lat={vd[1]:.3f} med={vd[2]:.3f}] mean={mean_val:.4f}")
        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        if mean_val > best_val_dice:
            best_val_dice, no_improve = mean_val, 0
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
