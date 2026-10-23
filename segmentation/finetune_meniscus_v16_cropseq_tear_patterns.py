"""
finetune_meniscus_v16_cropseq_tear_patterns.py
==============================================
v16: the v15 tear-pattern idea on the CROP + SEQUENCE model (run_v7_crop_sequence).

Same as run_v7_crop_sequence: start checkpoint (baseline_best_model.pth), model (SequenceUNet,
5-slice bottleneck ConvLSTM), cropped data (segmentation_data_v2_crop, real_pd_seg_data_v7_crop),
two-stream loop, losses, Adam + cosine schedule, seed, checkpoint selection on CLEAN real val
(training loop / eval / dump functions imported from finetune_meniscus_v7_sequence.py).
Only change: training slices pass through SequenceTearPatternDataset (v15 patterns on the middle
3 slices, applied in native-aspect space; see tear_patterns_seq.py) and train on its updated mask.
Pattern set = v17's (v15 + Fig. 10a "complex_hv" in-place variant, tear_synth.PATTERN_P_V17), with
v17's per-variant logging (samples/<variant>/, tear maps in the npz, per-variant synthetic val Dice).

As in v15: no real torn labels are used; the run aborts if a held-out torn patient is in
train/val; per-pattern synthetic val Dice goes to history.csv (diagnostic only); the
synthetic-data log for --log_epochs goes to <out_dir>/synthetic_log/epochNNN/.
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
from dataset_sequence import SequenceMeniscusDataset, SequenceRealPDDataset
from dataset_sequence_tear_patterns import SequenceTearPatternDataset
from sequence_model import build_sequence_model
from tear_patterns import MODES, MODE_P
from tear_synth import PATTERNS_V17, PATTERN_P_V17
from synthetic_log_utils import process_epoch
from finetune_meniscus_v7_sequence import (MergedLoss, SoftDiceLoss, set_seed, run_epoch,
                                           eval_real_per_patient, dump_val_predictions,
                                           N_CLASSES, PRETRAINED_N_CLASSES)
from finetune_meniscus_v14_tear_amodal import check_heldout, TORN_HELDOUT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--fake_data_root", required=True, help="segmentation_data_v2_crop")
    ap.add_argument("--real_data_root", required=True, help="real_pd_seg_data_v7_crop")
    ap.add_argument("--box_json", required=True, help="crop_box.json the cropped data was made with")
    ap.add_argument("--no_native_fix", action="store_true",
                    help="apply patterns directly in the stretched crop space (ablation only)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", type=float, nargs=3, default=[0.1, 1.5, 1.5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_batches", type=int, default=0, help="debug only: cap fake batches per epoch")
    ap.add_argument("--pattern_p", type=float, default=0.5)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--heldout_ids", nargs="*", default=TORN_HELDOUT)
    ap.add_argument("--log_epochs", type=int, nargs="*", default=[0])
    ap.add_argument("--log_samples_per_mode", type=int, default=100)
    args = ap.parse_args()
    mode_p = [MODE_P[MODES.index(m)] for m in args.modes]

    with open(args.box_json) as fh:
        box = json.load(fh)
    native_hw = None if args.no_native_fix else (box["y1"] - box["y0"], box["x1"] - box["x0"])

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | v16 crop+seq tear patterns: "
             f"{dict(zip(args.modes, np.round(np.array(mode_p) / sum(mode_p), 3)))} | pattern_p={args.pattern_p} "
             f"| native-aspect geometry: {native_hw if native_hw else 'OFF (stretched crop space)'}")

    model = build_sequence_model(args.pretrained_ckpt, device,
                                 pretrained_n_classes=PRETRAINED_N_CLASSES, n_classes=N_CLASSES)
    j = lambda root, split, kind: os.path.join(root, split, kind)
    fake_train_b = SequenceMeniscusDataset(j(args.fake_data_root, "train", "images"), j(args.fake_data_root, "train", "masks"), augment=True)
    fake_val_b = SequenceMeniscusDataset(j(args.fake_data_root, "val", "images"), j(args.fake_data_root, "val", "masks"), augment=False)
    real_train_b = SequenceRealPDDataset(j(args.real_data_root, "train", "images"), j(args.real_data_root, "train", "masks"), augment=True)
    real_val_b = SequenceRealPDDataset(j(args.real_data_root, "val", "images"), j(args.real_data_root, "val", "masks"), augment=False)
    check_heldout(args.heldout_ids, fake_train_b, fake_val_b, real_train_b, real_val_b)

    log_dir = os.path.join(args.out_dir, "synthetic_log") if args.log_epochs else None
    v17 = dict(inplace_variants=PATTERNS_V17, inplace_p=PATTERN_P_V17, log_variants=True)
    wrap = lambda b, synth, **kw: SequenceTearPatternDataset(b, native_hw=native_hw, synth=synth, p=args.pattern_p,
                                                             modes=args.modes, mode_p=mode_p, **v17, **kw)
    fake_train = wrap(fake_train_b, True, source="fake", log_dir=log_dir, save_per_mode=args.log_samples_per_mode)
    real_train = wrap(real_train_b, True, source="real", log_dir=log_dir, save_per_mode=args.log_samples_per_mode)
    fake_val, real_val = wrap(fake_val_b, False), wrap(real_val_b, False)          # clean val = selection
    synth_val = {m: SequenceTearPatternDataset(real_val_b, native_hw=native_hw, synth=True, p=1.0, modes=(m,),
                                               mode_p=(1.0,), fixed_seed=12345, **v17) for m in args.modes}
    if "inplace" in args.modes:                           # every in-place variant on its own (diagnostic only)
        for v in PATTERNS_V17:
            synth_val[f"inplace_{v}"] = SequenceTearPatternDataset(
                real_val_b, native_hw=native_hw, synth=True, p=1.0, modes=("inplace",), mode_p=(1.0,),
                fixed_seed=12345, inplace_variants=(v,), inplace_p=(1.0,))
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
                  + [f"dice_va_{p}" for p in val_pids] + [f"dice_synth_{m}_va" for m in synth_val])

    best, no_improve = -1.0, 0
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        tag = epoch if epoch in args.log_epochs else None
        fake_train.log_tag = real_train.log_tag = tag
        tl, td, tdf = run_epoch(model, real_train_loader, fake_train_loader, optimizer,
                                merged_loss_fn, fake_loss_fn, device, train=True)
        vl, vd, vdf = run_epoch(model, real_val_loader, fake_val_loader, optimizer,
                                merged_loss_fn, fake_loss_fn, device, train=False)
        scheduler.step()
        fake_train.log_tag = real_train.log_tag = None
        if tag is not None:
            try:
                process_epoch(os.path.join(log_dir, f"epoch{tag:03d}"))
            except Exception as e:
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
                      + [f"{pp.get(p, float('nan')):.5f}" for p in val_pids] + [f"{per_mode[m]:.5f}" for m in synth_val])
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
