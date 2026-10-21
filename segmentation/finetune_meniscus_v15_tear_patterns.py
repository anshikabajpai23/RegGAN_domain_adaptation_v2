"""
finetune_meniscus_v15_tear_patterns.py
======================================
v15: the replica recipe trained with a library of synthetic TEAR PATTERNS
(tear_patterns.py: inplace, ragged, lobe, truncate, absent, flap, bucket) whose labels follow
"all meniscus tissue where it actually is" (removed tissue -> background, displaced tissue ->
meniscus at its new place).

Same as the replica: start checkpoint (baseline_best_model.pth), architecture (plain smp.Unet,
ResNet34, 3-slice input, 3-class head; no extra heads), data, fake/real two-stream loop,
Adam + cosine schedule, losses, seed, checkpoint selection on CLEAN real val.
Only change: training slices pass through TearPatternDataset (and train on its updated mask).

No real torn labels are used for training. The 5 torn patients are held out for inference
only; the run aborts if any of them is found in train/val.

Diagnostics written to history.csv (never used for checkpoint selection): whole-meniscus Dice
on the real val patients with a FIXED synthetic pattern of each mode painted in
("dice_synth_<mode>_va"), so you can see which patterns the model learns.

Synthetic-data log (what training ACTUALLY sees, augmentation included), for the epochs in
--log_epochs (default: epoch 0): one statistics row for every synthetic training sample, plus the
full sample for the first --log_samples_per_mode samples of each pattern and source, then a
per-pattern summary and NIfTI stacks for review -> <out_dir>/synthetic_log/epochNNN/
(see synthetic_log_utils.py). Logging does not change training.
"""
import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v2 import Meniscus2_5DDataset
from dataset_2_5d_realpd import RealPDDataset
from dataset_2_5d_tear_patterns import TearPatternDataset
from tear_patterns import MODES, MODE_P
from synthetic_log_utils import process_epoch
from model_tear_amodal import build_from_dess_baseline      # replica model, wrapped (no extra heads)
from finetune_meniscus_v7_replica import MergedLoss, SoftDiceLoss, set_seed
from finetune_meniscus_v14_tear_amodal import (run_epoch, eval_real_per_patient, dump_val_predictions,
                                               check_heldout, TORN_HELDOUT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_batches", type=int, default=0, help="debug only: cap fake batches per epoch")
    ap.add_argument("--pattern_p", type=float, default=0.5, help="probability a meniscus slice gets a pattern")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--heldout_ids", nargs="*", default=TORN_HELDOUT)
    ap.add_argument("--log_epochs", type=int, nargs="*", default=[0],
                    help="epochs whose synthetic training samples are logged (empty = no logging)")
    ap.add_argument("--log_samples_per_mode", type=int, default=100,
                    help="full samples SAVED per pattern per source per logged epoch (stats are kept for all)")
    args = ap.parse_args()
    # run_epoch (from v14) checks these; v15 has no extra heads, so all aux terms are off
    args.inclusion, args.w_visible, args.w_contour, args.w_inclusion, args.contour_pos_weight = False, 0, 0, 0, 1
    mode_p = [MODE_P[MODES.index(m)] for m in args.modes]

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | v15 tear patterns: {dict(zip(args.modes, np.round(np.array(mode_p)/sum(mode_p), 3)))} "
             f"| pattern_p={args.pattern_p}")

    model = build_from_dess_baseline(args.pretrained_ckpt, device)
    fake_train_b = Meniscus2_5DDataset(os.path.join(args.fake_data_root, "train", "images"),
                                       os.path.join(args.fake_data_root, "train", "masks"), augment=True)
    fake_val_b = Meniscus2_5DDataset(os.path.join(args.fake_data_root, "val", "images"),
                                     os.path.join(args.fake_data_root, "val", "masks"), augment=False)
    real_train_b = RealPDDataset(os.path.join(args.real_data_root, "train", "images"),
                                 os.path.join(args.real_data_root, "train", "masks"), augment=True)
    real_val_b = RealPDDataset(os.path.join(args.real_data_root, "val", "images"),
                               os.path.join(args.real_data_root, "val", "masks"), augment=False)
    check_heldout(args.heldout_ids, fake_train_b, fake_val_b, real_train_b, real_val_b)

    log_dir = os.path.join(args.out_dir, "synthetic_log") if args.log_epochs else None
    wrap = lambda b, synth, **kw: TearPatternDataset(b, synth=synth, p=args.pattern_p, modes=args.modes, mode_p=mode_p, **kw)
    fake_train = wrap(fake_train_b, True, source="fake", log_dir=log_dir, save_per_mode=args.log_samples_per_mode)
    real_train = wrap(real_train_b, True, source="real", log_dir=log_dir, save_per_mode=args.log_samples_per_mode)
    if log_dir:
        log.info(f"Synthetic-data log: epochs {args.log_epochs}, ~{args.log_samples_per_mode} saved samples per "
                 f"pattern per source (stats for all) -> {log_dir}")
    fake_val, real_val = wrap(fake_val_b, False), wrap(real_val_b, False)          # clean val: selection = replica's
    synth_val = {m: TearPatternDataset(real_val_b, synth=True, p=1.0, modes=(m,), mode_p=(1.0,), fixed_seed=12345)
                 for m in args.modes}
    log.info(f"Fake PD train {len(fake_train)} / val {len(fake_val)} | Real PD train {len(real_train)} / val {len(real_val)}")

    dl = lambda ds, sh: DataLoader(ds, args.batch_size, shuffle=sh, num_workers=args.num_workers)
    fake_train_loader, fake_val_loader = dl(fake_train, True), dl(fake_val, False)
    real_train_loader, real_val_loader = dl(real_train, True), dl(real_val, False)
    if args.max_batches:
        from itertools import islice

        class _Cap:
            def __init__(self, loader, n): self.loader, self.n = loader, n
            def __iter__(self): return islice(iter(self.loader), self.n)
        fake_train_loader, fake_val_loader = _Cap(fake_train_loader, args.max_batches), _Cap(fake_val_loader, args.max_batches)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
    weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
    fake_loss_fn = lambda logits, y: nn.CrossEntropyLoss(weight=weights)(logits, y) + SoftDiceLoss()(logits, y)
    merged_loss_fn = MergedLoss(w_bg=0.1, w_men=1.5)

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices})
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va", "dice_real_tr", "dice_real_va", "gap",
                   "dice_fake_tr", "dice_fake_va", "dice_va_perpatient_mean"]
                  + [f"dice_va_{p}" for p in val_pids] + [f"dice_synth_{m}_va" for m in args.modes])

    best, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tag = epoch if epoch in args.log_epochs else None
        fake_train.log_tag = real_train.log_tag = tag        # picked up by the (re-created) workers
        tl, td, tdf, _ = run_epoch(model, real_train_loader, fake_train_loader, optimizer,
                                   merged_loss_fn, fake_loss_fn, args, device, train=True)
        vl, vd, vdf, _ = run_epoch(model, real_val_loader, fake_val_loader, optimizer,
                                   merged_loss_fn, fake_loss_fn, args, device, train=False)
        scheduler.step()
        fake_train.log_tag = real_train.log_tag = None
        if tag is not None:
            try:
                process_epoch(os.path.join(log_dir, f"epoch{tag:03d}"))
            except Exception as e:                       # review material must never stop training
                log.warning(f"synthetic-log summary failed for epoch {tag}: {e}")
        pp = eval_real_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        per_mode = {}
        for m, ds in synth_val.items():
            d = eval_real_per_patient(model, ds, device, args.batch_size)
            per_mode[m] = float(np.mean(list(d.values()))) if d else float("nan")
        log.info(f"Epoch {epoch:03d} lr={lr:.2e} loss tr={tl:.4f} va={vl:.4f} | dice_real tr={td:.4f} va={vd:.4f} | "
                 f"clean val={pp_mean:.4f} | synth val: " + " ".join(f"{m}={v:.3f}" for m, v in per_mode.items()))
        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl:.5f}", f"{td:.5f}", f"{vd:.5f}", f"{td - vd:+.5f}",
                       f"{tdf:.5f}", f"{vdf:.5f}", f"{pp_mean:.5f}"]
                      + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids] + [f"{per_mode[m]:.5f}" for m in args.modes])
        hist_fh.flush()

        torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_latest.pth"))
        if vd > best:
            best, no_improve = vd, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ckpt_best.pth"))
            log.info(f"  New best: {best:.4f} (epoch {epoch})")
            dump_val_predictions(model, real_val, device, os.path.join(args.out_dir, "val_predictions"), args.batch_size)
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}. Best: {best:.4f}")
                break
    hist_fh.close()
    log.info(f"Done. Best val Dice: {best:.4f} -> {args.out_dir}")


if __name__ == "__main__":
    main()
