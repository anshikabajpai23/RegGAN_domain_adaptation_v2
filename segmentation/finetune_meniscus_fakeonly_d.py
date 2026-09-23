"""
finetune_meniscus_fakeonly_d.py
================================
Run 3, D — position-conditioned shape (area) prior, FAKE PD ONLY.

BASE: finetune_meniscus_fakeonly_base.py, imported directly. The ONLY change is
+ lam * area_loss on the fake-PD branch.

WHY THIS IS A GOOD ZERO-SHOT CANDIDATE
---------------------------------------
The prior is a SHAPE statistic (expected meniscus area at each position along
the scan), and shape is preserved across the RegGAN translation by
construction: the deformation eval measured sub-pixel displacement (0.029 px
global, 0.055 px at the meniscus) with 0% folding. So area statistics learned
from fake PD should transfer to real PD essentially unchanged, while appearance
does not.

In the mixed regime this was redundant -- the model saw real meniscus sizes
directly, and area_frac decayed 0.0305 -> 0.0004 within 3 epochs, i.e. the
constraint was satisfied immediately and stopped contributing. With real PD
removed, the model no longer has that information from anywhere else, so the
prior has something to supply. Watch area_frac in history.csv: if it again
collapses to ~0 within a few epochs, the model found the constraint trivially
satisfiable and D is once more a no-op.

Reuses area_loss() from finetune_meniscus_v7_crop_seq_d.py (architecture- and
regime-agnostic: it only needs logits + bin indices). calibrate_lambda is
reimplemented here because that one takes a real-PD branch this arm doesn't have.

ONLY the FAKE shape prior is used (shape_prior_fake_2_5d.json, built from
UNCROPPED masks). The real prior the mixed recipe needed is irrelevant here.
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
from dataset_2_5d_v2 import Meniscus2_5DDataset
from finetune_meniscus_v7_crop_seq_d import area_loss   # reused, not reimplemented
from finetune_meniscus_fakeonly_base import (
    set_seed, build_model, SoftDiceLoss, dice_binary,
    eval_real_per_patient, dump_val_predictions,
    add_common_args, build_loaders, history_header,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NO_BIN = -1


class BinnedMeniscus2_5DDataset(Meniscus2_5DDataset):
    def __init__(self, img_root, mask_root, bin_map, augment=False):
        super().__init__(img_root, mask_root, augment=augment)
        self.bin_map = bin_map or {}

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["bin"] = int(self.bin_map.get(item["patient_id"], {}).get(str(item["slice_idx"]), NO_BIN))
        return item


def run_epoch(model, fake_loader, optimizer, fake_loss_fn, device, lam, prior, train=True):
    model.train() if train else model.eval()
    total, dice_fake_sum, area_sum, base_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in fake_loader:
            x, y = batch["image"].to(device), batch["mask"].to(device)
            b = batch["bin"].to(device)
            logits = model(x)
            base = fake_loss_fn(logits, y)
            area = area_loss(logits, b, *prior)
            loss = base + lam * area
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item(); base_sum += base.item(); area_sum += float(area)
            dice_fake_sum += dice_binary(logits.argmax(1), y)
            n += 1
    n = max(n, 1)
    return total / n, dice_fake_sum / n, area_sum / n, base_sum / n


@torch.no_grad()
def calibrate_lambda(model, fake_loader, fake_loss_fn, prior, device, target_frac, n_batches):
    """Measured, not guessed. Fake branch only.
    At step 0 the freshly re-headed classifier is near-uniform, so predicted area
    is enormous relative to the prior's targets -- a hardcoded lam=0.01 collapsed
    an earlier run to all-background. Measure before training starts."""
    model.eval()
    base_tot, area_tot, n = 0.0, 0.0, 0
    for batch in fake_loader:
        if n >= n_batches:
            break
        x, y = batch["image"].to(device), batch["mask"].to(device)
        b = batch["bin"].to(device)
        logits = model(x)
        base_tot += fake_loss_fn(logits, y).item()
        area_tot += float(area_loss(logits, b, *prior))
        n += 1
    if n == 0 or area_tot <= 1e-9:
        log.warning("Area loss is ~0 at init: unusual, falling back to lam=0.01.")
        return 0.01, base_tot / max(n, 1), area_tot / max(n, 1)
    base_mean, area_mean = base_tot / n, area_tot / n
    lam = target_frac * base_mean / area_mean
    log.info(f"  calibration: base={base_mean:.4f}  area={area_mean:.4f}  "
             f"(at lam=0.01 the area term would have been {0.01 * area_mean / base_mean:.1%} of base)")
    return lam, base_mean, area_mean


def load_prior(path, device):
    with open(path) as fh:
        blob = json.load(fh)
    m = blob["meta"]
    a_mean = torch.tensor(blob["area_mean"], dtype=torch.float32, device=device)
    a_std = torch.tensor(blob["area_std"], dtype=torch.float32, device=device)
    log.info(f"  fake prior: {m['bins']} bins, {m['n_patients_used']} patients, "
             f"{m['n_slices']} slices, built from {m['data_root']}")
    log.info(f"    area mean by bin (outer->notch): {[round(float(v)) for v in a_mean.tolist()]}")
    return (a_mean, a_std), blob["bin_map"]


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--shape_prior_fake", required=True,
                    help="MUST be built from the UNCROPPED fake masks")
    ap.add_argument("--lam", type=float, default=None, help="explicit; otherwise calibrated")
    ap.add_argument("--lam_target_frac", type=float, default=0.07)
    ap.add_argument("--calib_batches", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  RUN 3 D: shape/area prior  |  FAKE PD ONLY")

    prior, binmap = load_prior(args.shape_prior_fake, device)

    model = build_model(args.pretrained_ckpt, device)
    _, _, real_val = build_loaders(args)

    fake_train = BinnedMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), binmap, augment=True)
    fake_val = BinnedMeniscus2_5DDataset(
        os.path.join(args.fake_data_root, "val", "images"),
        os.path.join(args.fake_data_root, "val", "masks"), binmap, augment=False)
    cov = np.mean([fake_train[i]["bin"] >= 0 for i in range(0, min(len(fake_train), 400), 7)])
    log.info(f"Fake PD - train: {len(fake_train)} slices ({cov:.1%} with a bin)  |  val: {len(fake_val)} slices")

    fake_train_loader = DataLoader(fake_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    fake_val_loader = DataLoader(fake_val, args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = lambda logits, y: (nn.CrossEntropyLoss(weight=weights)(logits, y) + SoftDiceLoss()(logits, y))

    if args.lam is None:
        lam, base_m, area_m = calibrate_lambda(model, fake_train_loader, fake_loss_fn, prior,
                                               device, args.lam_target_frac, args.calib_batches)
        log.info(f"Calibrated lam={lam:.6f} (base={base_m:.4f}, area={area_m:.4f}, "
                 f"target {args.lam_target_frac:.0%} of base)")
    else:
        lam = args.lam
        log.info(f"Using explicit lam={lam:.6f}")

    with open(os.path.join(args.out_dir, "d_config.json"), "w") as fh:
        json.dump({"lam": lam, "lam_target_frac": args.lam_target_frac, "seed": args.seed}, fh, indent=2)

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices}) if real_val is not None else []
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(history_header(real_val, val_pids, extra_cols=["area_tr", "area_va", "area_frac"]))

    best_smooth, no_improve, va_hist = -1.0, 0, []
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, tdf, area_tr, base_tr = run_epoch(model, fake_train_loader, optimizer, fake_loss_fn,
                                              device, lam, prior, train=True)
        vl, vdf, area_va, _ = run_epoch(model, fake_val_loader, optimizer, fake_loss_fn,
                                        device, lam, prior, train=False)
        scheduler.step()

        va_hist.append(vdf)
        smooth = float(np.mean(va_hist[-args.select_smooth:]))
        area_frac = (lam * area_tr) / max(base_tr, 1e-9)
        row = [epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}",
               f"{tdf:.5f}", f"{vdf:.5f}", f"{tdf - vdf:+.5f}", f"{smooth:.5f}"]
        msg = (f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl:.4f}  |  "
               f"dice_fake tr={tdf:.4f} va={vdf:.4f} smooth={smooth:.4f}  |  "
               f"area tr={area_tr:.3f} frac={area_frac:.4f}")

        if real_val is not None:
            pp = eval_real_per_patient(model, real_val, device, args.batch_size)
            pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
            row += [f"{pp_mean:.5f}"] + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids]
            msg += f"  |  [diag] real_va={pp_mean:.4f}"
        row += [f"{area_tr:.6f}", f"{area_va:.6f}", f"{area_frac:.4f}"]
        log.info(msg)
        if epoch == 2 and area_frac < 0.005:
            log.warning("area_frac has already decayed below 0.5% of base loss -- the prior is "
                        "trivially satisfied and D is likely a no-op again, as in the mixed regime.")

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
