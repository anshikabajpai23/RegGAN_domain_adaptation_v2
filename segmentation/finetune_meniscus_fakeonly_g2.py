"""
finetune_meniscus_fakeonly_g2.py
=================================
Run 3, G2 — region-darker-than-rim loss, FAKE PD ONLY.

BASE: finetune_meniscus_fakeonly_base.py, imported directly. The ONLY change is
+ lam * rim_loss on the fake-PD branch.

WHY THIS IS THE STRONGEST ZERO-SHOT CANDIDATE
----------------------------------------------
rim_loss encodes a physical fact, not a dataset statistic: meniscus tissue is
darker than the rim of tissue immediately around it. That is true in DESS, in
fake PD and in real PD alike, so unlike an appearance prior learned from the
training distribution it should survive the domain shift by construction.

In the mixed regime this was largely redundant -- the model had 13 real
patients teaching it real-PD appearance directly. With those removed, a
domain-invariant physical constraint is exactly the kind of supervision that
has somewhere to bite.

The pre-check that gates this experiment (scripts/precheck_rim_darkness.py) was
already run against these exact uncropped roots: the prior holds on 1.000 of
real-PD slices and 0.964 of fake-PD slices, both far above the 0.80 gate. No
re-run needed.

lam is CALIBRATED IN-JOB against the measured base-loss magnitude, never
hardcoded -- a hardcoded lam collapsed an earlier D run to all-background.
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rim_loss import rim_loss, meniscus_prob   # reused, not reimplemented
from finetune_meniscus_fakeonly_base import (
    set_seed, build_model, SoftDiceLoss, dice_binary,
    eval_real_per_patient, dump_val_predictions,
    add_common_args, build_loaders, history_header,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CENTRE = 1   # channel 1 of the [prev, curr, next] 2.5D stack


def run_epoch(model, fake_loader, optimizer, fake_loss_fn, device, lam, margin, rim_k, train=True):
    model.train() if train else model.eval()
    total, dice_fake_sum, rim_sum, base_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in fake_loader:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            logits = model(x)
            base = fake_loss_fn(logits, y)
            rim = rim_loss(meniscus_prob(logits), x[:, CENTRE], k=rim_k, margin=margin)
            loss = base + lam * rim
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item(); base_sum += base.item(); rim_sum += rim.item()
            dice_fake_sum += dice_binary(logits.argmax(1), y)
            n += 1
    n = max(n, 1)
    return total / n, dice_fake_sum / n, rim_sum / n, base_sum / n


@torch.no_grad()
def calibrate_lambda(model, fake_loader, fake_loss_fn, device, margin, rim_k, target_frac, n_batches):
    """Measured, not guessed. Fake branch only -- there is no real branch here."""
    model.eval()
    base_tot, rim_tot, n = 0.0, 0.0, 0
    for batch in fake_loader:
        if n >= n_batches:
            break
        x, y = batch["image"].to(device), batch["mask"].to(device)
        logits = model(x)
        base_tot += fake_loss_fn(logits, y).item()
        rim_tot += rim_loss(meniscus_prob(logits), x[:, CENTRE], k=rim_k, margin=margin).item()
        n += 1
    if n == 0 or rim_tot <= 1e-9:
        log.warning("Rim loss is ~0 at init: falling back to lam=1.0.")
        return 1.0, base_tot / max(n, 1), rim_tot / max(n, 1)
    base_mean, rim_mean = base_tot / n, rim_tot / n
    return target_frac * base_mean / rim_mean, base_mean, rim_mean


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--margin", type=float, default=0.0293,
                    help="normalised intensity; 25%% of the measured fake-PD gap")
    ap.add_argument("--rim_k", type=int, default=5)
    ap.add_argument("--lam", type=float, default=None, help="explicit; otherwise calibrated")
    ap.add_argument("--lam_target_frac", type=float, default=0.07)
    ap.add_argument("--calib_batches", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  RUN 3 G2: rim-darkness loss  |  FAKE PD ONLY")
    log.info(f"  margin={args.margin}  rim_k={args.rim_k} px")

    model = build_model(args.pretrained_ckpt, device)
    fake_train, fake_val, real_val = build_loaders(args)

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = lambda logits, y: (nn.CrossEntropyLoss(weight=weights)(logits, y) + SoftDiceLoss()(logits, y))

    if args.lam is None:
        lam, base_m, rim_m = calibrate_lambda(model, fake_train_loader, fake_loss_fn, device,
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

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices}) if real_val is not None else []
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(history_header(real_val, val_pids, extra_cols=["rim_tr", "rim_va", "rim_frac"]))

    best_smooth, no_improve, va_hist = -1.0, 0, []
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, tdf, rim_tr, base_tr = run_epoch(model, fake_train_loader, optimizer, fake_loss_fn,
                                             device, lam, args.margin, args.rim_k, train=True)
        vl, vdf, rim_va, _ = run_epoch(model, fake_val_loader, optimizer, fake_loss_fn,
                                       device, lam, args.margin, args.rim_k, train=False)
        scheduler.step()

        va_hist.append(vdf)
        smooth = float(np.mean(va_hist[-args.select_smooth:]))
        rim_frac = (lam * rim_tr) / max(base_tr, 1e-9)
        row = [epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}",
               f"{tdf:.5f}", f"{vdf:.5f}", f"{tdf - vdf:+.5f}", f"{smooth:.5f}"]
        msg = (f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl:.4f}  |  "
               f"dice_fake tr={tdf:.4f} va={vdf:.4f} smooth={smooth:.4f}  |  "
               f"rim tr={rim_tr:.4f} frac={rim_frac:.3f}")

        if real_val is not None:
            pp = eval_real_per_patient(model, real_val, device, args.batch_size)
            pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
            row += [f"{pp_mean:.5f}"] + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids]
            msg += f"  |  [diag] real_va={pp_mean:.4f}"
        row += [f"{rim_tr:.6f}", f"{rim_va:.6f}", f"{rim_frac:.4f}"]
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
