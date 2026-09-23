"""
finetune_meniscus_fakeonly_a3.py
=================================
Run 3, A3 — slice-position weighted loss, FAKE PD ONLY.

BASE: finetune_meniscus_fakeonly_base.py, imported directly (build_model,
SoftDiceLoss, dice_binary, the diagnostic real-val eval, the arg parser and the
loaders) so this arm cannot silently drift from the baseline it is measured
against. The ONLY change is that each fake-PD sample's loss is scaled by its
distance-to-span-end weight.

    --variant up    end slices count up to 2x
    --variant down  end slices count as 0.3x
    --variant none  all weights 1.0 (control for the per-sample reduction)

CARRIES THE v2 CE FIX. Weighting requires a per-sample loss, and the Run 2 v1
scripts got the per-sample cross-entropy denominator wrong -- they divided by
pixel count where nn.CrossEntropyLoss(weight=w) divides by the SUM OF CLASS
WEIGHTS, making the CE term ~8.3x too small (replica 0.000731 vs v1 0.000088).
weighted_ce_per_sample below uses the correct denominator and the startup check
refuses to train if it ever stops matching. Do not reintroduce the v1 form.

The per-sample-vs-batch Dice difference CANNOT be removed (per-sample Dice
charges every empty-GT slice a flat 1.0 regardless), which is why --variant none
exists: read up/down against none, and none against the Run 3 baseline.

Only the FAKE slice-weights file is needed. The real-PD weights that the mixed
recipe used are irrelevant here -- nothing real is trained on.
"""
import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v2 import Meniscus2_5DDataset
from finetune_meniscus_fakeonly_base import (
    N_CLASSES, set_seed, build_model, dice_binary,
    eval_real_per_patient, dump_val_predictions,
    add_common_args, build_loaders, history_header,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class WeightedMeniscus2_5DDataset(Meniscus2_5DDataset):
    def __init__(self, img_root, mask_root, weights, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.weights = weights  # {pid: {slice_idx: w}} or None

    def __getitem__(self, i):
        item = super().__getitem__(i)
        pid, idx = self.items[i]
        w = 1.0
        if self.weights is not None:
            pw = self.weights.get(pid, {})
            w = pw.get(str(idx), pw.get(idx, 1.0))
        item["weight"] = torch.tensor(float(w), dtype=torch.float32)
        return item


class SoftDiceLossPerSample(nn.Module):
    def __init__(self, n_classes=N_CLASSES, eps=1e-6):
        super().__init__(); self.n_classes = n_classes; self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1); loss = 0.0
        for c in range(1, self.n_classes):
            p = probs[:, c]; t = (targets == c).float()
            inter = (p * t).sum(dim=(1, 2)); den = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            loss = loss + (1.0 - (2 * inter + self.eps) / (den + self.eps))
        return loss / (self.n_classes - 1)


def weighted_ce_per_sample(logits, y, class_weights, eps=1e-6):
    """Per-sample equivalent of nn.CrossEntropyLoss(weight=class_weights).
    reduction="none" already multiplies by the class weight, so the correct
    denominator is the SUM OF THOSE WEIGHTS, not the pixel count (the v1 bug)."""
    ce_map = F.cross_entropy(logits, y, weight=class_weights, reduction="none")
    w_map = class_weights[y]
    return ce_map.sum(dim=(1, 2)) / w_map.sum(dim=(1, 2)).clamp(min=eps)


def fake_loss_per_sample(logits, y, class_weights):
    return weighted_ce_per_sample(logits, y, class_weights) + SoftDiceLossPerSample()(logits, y)


def weighted_mean(per_sample, w):
    return (per_sample * w).sum() / w.sum().clamp(min=1e-6)


@torch.no_grad()
def verify_ce_matches_reference(class_weights, device, atol=1e-6):
    """Refuses to train if the per-sample CE stops matching nn.CrossEntropyLoss."""
    torch.manual_seed(0)
    B, H, W = 4, 64, 64
    logits = torch.randn(B, N_CLASSES, H, W, device=device)
    y = torch.zeros(B, H, W, dtype=torch.long, device=device)
    y[:, 20:26, 20:30] = 1
    y[:, 40:45, 35:42] = 2
    ref = nn.CrossEntropyLoss(weight=class_weights)(logits, y)
    mine = weighted_ce_per_sample(logits, y, class_weights).mean()
    d = float((ref - mine).abs())
    return d < atol, float(ref), float(mine), d


def run_epoch(model, fake_loader, optimizer, class_weights, device, train=True):
    model.train() if train else model.eval()
    total_loss, dice_fake_sum, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in fake_loader:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            w = batch["weight"].to(device)
            logits = model(x)
            loss = weighted_mean(fake_loss_per_sample(logits, y, class_weights), w)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            dice_fake_sum += dice_binary(logits.argmax(1), y)
            n += 1
    n = max(n, 1)
    return total_loss / n, dice_fake_sum / n


def load_weights(path, variant):
    if variant == "none" or not path:
        return None
    with open(path) as fh:
        blob = json.load(fh)
    if variant not in blob:
        raise SystemExit(f"{path} has no '{variant}' block (keys: {list(blob)})")
    m = blob.get("meta", {})
    log.info(f"  weights [{variant}]: {m.get('n_patients')} patients, {m.get('n_slices')} slices")
    return blob[variant]


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--variant", required=True, choices=["up", "down", "none"])
    ap.add_argument("--weights_fake", default=None)
    args = ap.parse_args()
    if args.variant != "none" and not args.weights_fake:
        ap.error("--variant up/down requires --weights_fake")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  RUN 3 A3 variant={args.variant}  |  FAKE PD ONLY")

    cw = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    ok, ref, mine, d = verify_ce_matches_reference(cw, device)
    log.info(f"CE-equivalence check: reference={ref:.8f}  mine={mine:.8f}  diff={d:.2e} "
             f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(f"per-sample CE no longer matches (diff {d:.3e}). Refusing to train.")

    w_fake = load_weights(args.weights_fake, args.variant)

    model = build_model(args.pretrained_ckpt, device)
    _, _, real_val = build_loaders(args)

    fake_train = WeightedMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), w_fake, augment=True)
    fake_val = WeightedMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "val", "images"),
        os.path.join(args.fake_data_root, "val", "masks"), w_fake, augment=False)

    sample_w = [float(fake_train[i]["weight"]) for i in range(0, min(len(fake_train), 400), 7)]
    log.info(f"  fake-train weight sample: min {min(sample_w):.3f}  "
             f"mean {sum(sample_w)/len(sample_w):.3f}  max {max(sample_w):.3f}")

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices}) if real_val is not None else []
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(history_header(real_val, val_pids))

    best_smooth, no_improve, va_hist = -1.0, 0, []
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, tdf = run_epoch(model, fake_train_loader, optimizer, cw, device, train=True)
        vl, vdf = run_epoch(model, fake_val_loader, optimizer, cw, device, train=False)
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


if __name__ == "__main__":
    main()
