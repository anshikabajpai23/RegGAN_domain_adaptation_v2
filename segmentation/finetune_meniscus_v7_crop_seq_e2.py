"""
finetune_meniscus_v7_crop_seq_e2.py
====================================
Phase 5, E2 — slice position into GATING (FiLM at the bottleneck), on the
crop+sequence base.

BASE (unchanged from run_v7_crop_sequence): same pretrained/baseline_best_model.pth,
same SequenceUNet + ConvLSTM, same cropped data, same losses/optimizer/LR/seed/
epochs/patience, same history.csv schema.

THE ONE CHANGE vs the baseline: a zero-initialised FiLM gate modulates the
ConvLSTM hidden state by that timestep's position (sequence_model_film.py).

THE ONE CHANGE vs E1: where the position goes in. E1 appends it as a 4th input
channel and makes the encoder carry it; E2 injects it where the network already
has semantic features. Doc §2.4: if E2 beats E1 by more than the noise floor,
position is best used at the semantic level and that becomes the method; if
they tie, keep E1 because it is simpler.

The position values themselves are computed by the SAME dataset classes E1 uses
(imported, not reimplemented) so the two runs differ ONLY in where position is
injected -- otherwise the comparison would be confounded.

ZERO-INIT is re-verified in-job and training is REFUSED if it fails: without it
a difference could come from re-initialised weights rather than from position.

WHAT TO READ: pos_wnorm in history.csv is the FiLM gate's output-layer norm. It
starts at 0. Still ~0 at the end means the model found no use for position even
at the bottleneck -- which, combined with an E1 null, would close out the whole
position line of attack (and make D pointless).
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# shared with E1 on purpose: identical position computation, identical losses
from finetune_meniscus_v7_crop_seq_e1 import (
    PosSequenceMeniscusDataset, PosSequenceRealPDDataset,
    MergedLoss, SoftDiceLoss, dice_binary, set_seed,
    run_epoch, eval_real_per_patient, dump_val_predictions,
)
from sequence_model_film import build_sequence_film_model, verify_zero_init_film

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3
PRETRAINED_N_CLASSES = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--volume_lengths", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--film_hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_zero_init_check", action="store_true",
                    help="NOT RECOMMENDED -- see module docstring")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  LR: {args.lr} -> 1e-8  |  E2: FiLM gating at the bottleneck, cropped data")

    if not args.skip_zero_init_check:
        ok, diff = verify_zero_init_film(device="cpu")
        if not ok:
            raise SystemExit(f"ZERO-INIT CHECK FAILED (max diff {diff:.3e}). Refusing to train: "
                             f"E2's result would be uninterpretable.")

    with open(args.volume_lengths) as fh:
        vl = json.load(fh)
    meta = vl.get("meta", {})
    log.info(f"volume_lengths: {meta.get('n_fake')} fake, {meta.get('n_real')} real  "
             f"(span_coverage_ok={meta.get('span_coverage_ok')})")

    model = build_sequence_film_model(args.pretrained_ckpt, device,
                                      pretrained_n_classes=PRETRAINED_N_CLASSES,
                                      n_classes=N_CLASSES, hidden=args.film_hidden)
    log.info(f"Loaded DESS baseline from {args.pretrained_ckpt}, wrapped with SequenceUNetFiLM "
             f"(hidden={args.film_hidden})")
    log.info(f"  FiLM gate weight norm at init: {model.position_weight_norm():.3e} (must be 0)")

    fake_train = PosSequenceMeniscusDataset(
        os.path.join(args.fake_data_root, "train", "images"),
        os.path.join(args.fake_data_root, "train", "masks"), vl["fake"], augment=True)
    fake_val = PosSequenceMeniscusDataset(
        os.path.join(args.fake_data_root, "val", "images"),
        os.path.join(args.fake_data_root, "val", "masks"), vl["fake"], augment=False)
    real_train = PosSequenceRealPDDataset(
        os.path.join(args.real_data_root, "train", "images"),
        os.path.join(args.real_data_root, "train", "masks"), vl["real"], augment=True)
    real_val = PosSequenceRealPDDataset(
        os.path.join(args.real_data_root, "val", "images"),
        os.path.join(args.real_data_root, "val", "masks"), vl["real"], augment=False)

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

    val_pids = sorted({s.rsplit("_", 1)[0] for s in real_val.slices})
    hist_fh = open(os.path.join(args.out_dir, "history.csv"), "w", newline="")
    hist = csv.writer(hist_fh)
    hist.writerow(["epoch", "lr", "loss_tr", "loss_va", "dice_real_tr", "dice_real_va", "gap",
                   "dice_fake_tr", "dice_fake_va", "dice_va_perpatient_mean"]
                  + [f"dice_va_{p}" for p in val_pids] + ["pos_wnorm"])

    best_val_dice, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tl, td, tdf = run_epoch(model, real_train_loader, fake_train_loader, optimizer,
                                merged_loss_fn, fake_loss_fn, device, train=True)
        vl_, vd, vdf = run_epoch(model, real_val_loader, fake_val_loader, optimizer,
                                 merged_loss_fn, fake_loss_fn, device, train=False)
        scheduler.step()

        pp = eval_real_per_patient(model, real_val, device, args.batch_size)
        pp_mean = float(np.mean(list(pp.values()))) if pp else float("nan")
        pp_str = "  ".join(f"{p}={pp.get(p, float('nan')):.4f}" for p in val_pids)
        wnorm = model.position_weight_norm()

        log.info(f"Epoch {epoch:03d}  lr={lr:.2e}  loss_tr={tl:.4f} loss_va={vl_:.4f}  |  "
                 f"dice_real tr={td:.4f} va={vd:.4f} gap={td - vd:+.4f}  |  "
                 f"dice_fake tr={tdf:.4f} va={vdf:.4f}  |  film_wnorm={wnorm:.4e}")
        log.info(f"           per-patient val (clean pass): mean={pp_mean:.4f}   {pp_str}")

        hist.writerow([epoch, f"{lr:.3e}", f"{tl:.5f}", f"{vl_:.5f}", f"{td:.5f}", f"{vd:.5f}", f"{td - vd:+.5f}",
                       f"{tdf:.5f}", f"{vdf:.5f}", f"{pp_mean:.5f}"]
                      + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids] + [f"{wnorm:.6e}"])
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
    final = model.position_weight_norm()
    log.info(f"Done. Best val Dice: {best_val_dice:.4f}")
    log.info(f"Final FiLM gate weight norm: {final:.4e} (started at 0)")
    if final < 1e-4:
        log.info("  -> Still ~zero: no use found for position even at the bottleneck.")
        log.info("     Combined with an E1 null, that closes out the position line")
        log.info("     of attack and makes D (position-conditioned prior) pointless.")


if __name__ == "__main__":
    main()
