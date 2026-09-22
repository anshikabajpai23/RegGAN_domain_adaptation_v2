"""
dump_per_slice_fake_val_crop_seq.py
====================================
Crop + sequence (ConvLSTM) variant of dump_per_slice_fake_val.py — runs
run_v7_crop_sequence/ckpt_best.pth over the fake-PD val cohort and writes a
per-slice CSV in the SAME column order, so build_hallucination_table.py and
the notebook's Section 2b consume it unchanged.

Two things the plain script cannot do, both handled here:

  1. ARCHITECTURE. The plain script hardcodes smp.Unet + Meniscus2_5DDataset
     (3-channel [i-1,i,i+1] stack). This checkpoint is a SequenceUNet taking
     (B, 5, H, W) — 5 slices encoded separately, aggregated by a ConvLSTM at
     the bottleneck. Uses load_trained_sequence_model() + the same OFFSETS
     (-2..+2) as dataset_sequence.py.

  2. CROP SPACE. run_v7_crop_sequence was trained on segmentation_data_v2_crop,
     i.e. every slice cropped to crop_box.json's box and resized back up to
     384x384. This script applies that SAME transform on the fly (identical
     crop_resize: order=3 images / order=0 masks, verbatim from
     compute_and_apply_fixed_crop.py) rather than requiring a pre-built cropped
     copy of the valneg split.

WHICH DICE IS REPORTED, AND WHY IT MATTERS
------------------------------------------
The model predicts in CROPPED space, where the meniscus is magnified ~3x. Dice
measured there is NOT comparable to the replica's CSV — magnifying a thin
structure shrinks the boundary-error share and inflates Dice. So the primary
`dice` column here is measured in FULL-FRAME space: the cropped prediction is
mapped back down and placed at the box location on a 384x384 canvas (zeros
outside, exactly as infer_sequence_v3_crop.py does for the 17pt test set), then
scored against the ORIGINAL uncropped GT. That is apples-to-apples with
run_v7_replica_seed42/fake_val_per_slice_dice.csv.

Cropped-space numbers are kept in extra trailing columns (dice_cropspace,
gt_voxels_cropspace, pred_voxels_cropspace) — not to be mixed into the
comparison table, but useful on their own: the gap between the two Dice values
quantifies how much of the crop approach's apparent gain is magnification
rather than better segmentation. Extra trailing columns are safe for
csv.DictReader consumers, which read by name.

Anything the box excludes is a guaranteed miss in the full-frame score. That is
the real cost of the fixed-box approach and it SHOULD show up here — same open
caveat as scripts/check_crop_box_coverage.py.

Read-only: loads a checkpoint, trains nothing, saves no weights.

Usage:
  python segmentation/dump_per_slice_fake_val_crop_seq.py \
      --ckpt      segmentation_runs/run_v7_crop_sequence/ckpt_best.pth \
      --box_json  crop_box.json \
      --img_root  segmentation_data_v3_valneg/val/images \
      --mask_root segmentation_data_v3_valneg/val/masks \
      --out_csv   fake_val_per_slice_dice.csv \
      --dump_volumes --vol_out_dir fake_val_predictions
"""
import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_sequence import SequenceMeniscusDataset, OFFSETS
from sequence_model import load_trained_sequence_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3
TARGET = (384, 384)


def crop_resize(arr, box, order):
    """Verbatim from compute_and_apply_fixed_crop.py — must stay identical, or
    eval-time and train-time cropping silently diverge."""
    y0, y1, x0, x1 = box
    crop = arr[y0:y1, x0:x1]
    scale = (TARGET[0] / crop.shape[0], TARGET[1] / crop.shape[1])
    return zoom(crop, scale, order=order)


def uncrop(pred_crop, box):
    """Inverse of crop_resize for a label map — verbatim convention from
    infer_sequence_v3_crop.py: resize DOWN with order=0 (keeps integer labels
    exact), then place at the box location on a full-frame canvas."""
    y0, y1, x0, x1 = box
    crop_h, crop_w = y1 - y0, x1 - x0
    out = np.zeros(TARGET, dtype=np.int64)
    small = zoom(pred_crop.astype(np.float32), (crop_h / TARGET[0], crop_w / TARGET[1]),
                 order=0).astype(np.int64)
    out[y0:y1, x0:x1] = small
    return out


class CroppedSequenceFakeValDataset(SequenceMeniscusDataset):
    """Same slice indexing as SequenceMeniscusDataset, but returns the 5-slice
    stack CROPPED (what the model was trained on) alongside the ORIGINAL
    uncropped mask (what the full-frame score is computed against)."""

    def __init__(self, img_root, mask_root, box):
        super().__init__(img_root, mask_root, augment=False)
        self.box = box

    def __getitem__(self, i):
        pid, idx = self.items[i]
        slices = [self._load(pid, idx + o) for o in OFFSETS]
        slices = [s if s is not None else self._load(pid, idx) for s in slices]
        image_crop = np.stack([crop_resize(s, self.box, order=3) for s in slices], axis=0)

        mask_full = np.load(os.path.join(self.mask_root, pid,
                                         f"{pid}_slice_{idx:03d}.npy")).astype(np.int64)
        mask_crop = crop_resize(mask_full.astype(np.float32), self.box, order=0).astype(np.int64)

        return {
            "image_crop": torch.from_numpy(np.ascontiguousarray(image_crop)).float(),
            "mask_full":  torch.from_numpy(mask_full),
            "mask_crop":  torch.from_numpy(mask_crop),
            "patient_id": pid,
            "slice_idx":  idx,
        }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        required=True)
    ap.add_argument("--box_json",    required=True,
                    help="crop_box.json the checkpoint was TRAINED on — a different "
                         "box here silently invalidates every number produced")
    ap.add_argument("--img_root",    required=True)
    ap.add_argument("--mask_root",   required=True)
    ap.add_argument("--out_csv",     required=True)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dump_volumes", action="store_true",
                    help="also save <pid>_image/gt/pred.nii.gz per patient, full-frame")
    ap.add_argument("--vol_out_dir", default=None)
    args = ap.parse_args()
    if args.dump_volumes and not args.vol_out_dir:
        ap.error("--dump_volumes requires --vol_out_dir")

    with open(args.box_json) as fh:
        bd = json.load(fh)
    box = (bd["y0"], bd["y1"], bd["x0"], bd["x1"])
    log.info(f"Crop box {box}  size {box[1]-box[0]}x{box[3]-box[2]}  padding={bd.get('padding')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_sequence_model(args.ckpt, device, n_classes=N_CLASSES)
    model.eval()
    log.info(f"Loaded {args.ckpt}  |  device={device}  |  offsets={OFFSETS}  |  eval(), no_grad")

    ds = CroppedSequenceFakeValDataset(args.img_root, args.mask_root, box)
    pids = sorted({pid for pid, _ in ds.items})
    log.info(f"Dataset: {len(ds)} slices across {len(pids)} patients")

    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    eps = 1e-6
    rows = []
    n_empty_gt = 0
    n_outside_box = 0      # GT px the box cannot reach — guaranteed misses
    total_gt_px = 0
    vols = {}

    for batch in loader:
        x = batch["image_crop"].to(device)
        pred_crop = model(x).argmax(1).cpu().numpy()          # (B, 384, 384) crop space
        mask_full = batch["mask_full"].numpy()
        mask_crop = batch["mask_crop"].numpy()

        for b in range(x.shape[0]):
            pid  = batch["patient_id"][b]
            sidx = int(batch["slice_idx"][b])

            pred_full = uncrop(pred_crop[b], box)

            p = pred_full > 0
            g = mask_full[b] > 0
            tp, ps, gs = float((p & g).sum()), float(p.sum()), float(g.sum())
            if gs == 0:
                n_empty_gt += 1

            # how much GT the fixed box physically cannot cover on this slice
            y0, y1, x0, x1 = box
            inside = np.zeros_like(g)
            inside[y0:y1, x0:x1] = True
            n_outside_box += int((g & ~inside).sum())
            total_gt_px   += int(gs)

            pc = pred_crop[b] > 0
            gc = mask_crop[b] > 0
            tpc, psc, gsc = float((pc & gc).sum()), float(pc.sum()), float(gc.sum())

            rows.append({
                "patient_id":  pid,
                "slice_idx":   sidx,
                "dice":        f"{(2*tp + eps) / (ps + gs + eps):.4f}",
                "gt_voxels":   int(gs),
                "pred_voxels": int(ps),
                "false_neg":   int(gs - tp),
                "false_pos":   int(ps - tp),
                "dice_cropspace":        f"{(2*tpc + eps) / (psc + gsc + eps):.4f}",
                "gt_voxels_cropspace":   int(gsc),
                "pred_voxels_cropspace": int(psc),
            })

            if args.dump_volumes:
                # centre of the 5-slice window (index 2), mapped back to full frame
                # so image/gt/pred all open in the SAME space as the replica's dump
                centre_crop = x[b, len(OFFSETS) // 2].cpu().numpy()
                vols.setdefault(pid, {})[sidx] = (
                    centre_crop.astype(np.float32),
                    mask_full[b].astype(np.int16),
                    pred_full.astype(np.int16),
                )

    rows.sort(key=lambda r: (r["patient_id"], r["slice_idx"]))
    fieldnames = ["patient_id", "slice_idx", "dice", "gt_voxels", "pred_voxels",
                  "false_neg", "false_pos",
                  "dice_cropspace", "gt_voxels_cropspace", "pred_voxels_cropspace"]
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    log.info(f"Wrote {len(rows)} slice rows -> {args.out_csv}")
    log.info(f"Empty-GT slices present: {n_empty_gt} / {len(rows)}")
    if n_empty_gt == 0:
        log.warning("ZERO empty-GT slices — this split was prepared with "
                    "min_meniscus_pixels > 0. Re-prep with --min_meniscus_pixels 0.")

    pct = 100.0 * n_outside_box / max(total_gt_px, 1)
    log.info(f"GT px OUTSIDE the crop box: {n_outside_box:,} / {total_gt_px:,} ({pct:.2f}%) "
             f"— guaranteed misses, no model can recover these")
    if pct > 1.0:
        log.warning(f"{pct:.2f}% of fake-val GT is unreachable by this box. The full-frame "
                    f"Dice above is capped by that, independent of model quality.")

    if args.dump_volumes:
        try:
            import nibabel as nib
        except ImportError:
            log.warning("nibabel not available — skipping volume dump")
            return
        os.makedirs(args.vol_out_dir, exist_ok=True)
        # Identity spacing, same reasoning as dump_per_slice_fake_val.py: fake PD
        # has no matching real acquisition, so there is no physical mm to recover.
        # NOTE the image channel here is the CROPPED centre slice while gt/pred are
        # full-frame — they are NOT co-registered with each other in this dump.
        affine = np.eye(4, dtype=np.float32)
        for pid, slices in sorted(vols.items()):
            order = sorted(slices.keys())
            img = np.stack([slices[s][0] for s in order], axis=0)
            gt  = np.stack([slices[s][1] for s in order], axis=0)
            pr  = np.stack([slices[s][2] for s in order], axis=0)
            nib.save(nib.Nifti1Image(img, affine), os.path.join(args.vol_out_dir, f"{pid}_image_cropspace.nii.gz"))
            nib.save(nib.Nifti1Image((gt > 0).astype(np.int16), affine), os.path.join(args.vol_out_dir, f"{pid}_gt.nii.gz"))
            nib.save(nib.Nifti1Image((pr > 0).astype(np.int16), affine), os.path.join(args.vol_out_dir, f"{pid}_pred.nii.gz"))
            log.info(f"  dumped {pid}: {len(order)} slices")
        log.info(f"Volumes -> {args.vol_out_dir}")


if __name__ == "__main__":
    main()
