"""
build_v3_index_files.py
========================
Phase 4 step 4.7/4.8 prep — builds the two index files described in
docs/reggan_dice_debug.md §4.7:

  train_index_posonly.txt  : mask-bearing slices only (for step 7 / P4)
  train_index_withneg.txt  : mask-bearing + selected negatives (for step 8 / P2)

Input is a segmentation_data_v3/ directory produced by prepare_meniscus_masks.py
run with --min_meniscus_pixels 0 (every slice saved, positive and empty —
NOT the segmentation_data_v2/ convention, which already drops all empty
slices and therefore cannot supply negatives at all).

Negative selection (train_index_withneg.txt), per patient:
  - every empty slice within ±5 slices of any span boundary ("near-span")
  - every empty slice strictly between the two spans, when exactly 2 spans
    exist ("notch")
  - a random 10% sample of all remaining empty slices ("far"), seeded for
    reproducibility

Span detection reuses find_spans() from analyze_gt_tapering.py — contiguous
runs of slices with collapsed-mask pixel count > 0.

Usage:
  python scripts/build_v3_index_files.py \
      --data_root segmentation_data_v3 \
      --split train \
      --out_dir segmentation_data_v3 \
      --seed 42
"""
import argparse
import glob
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_gt_tapering import find_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="dir holding <split>/masks/<pid>/<pid>_slice_NNN.npy")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--near_span_radius", type=int, default=5)
    ap.add_argument("--far_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mask_root = os.path.join(args.data_root, args.split, "masks")

    per_patient = defaultdict(dict)   # pid -> {idx: px_count}
    for f in sorted(glob.glob(os.path.join(mask_root, "*", "*.npy"))):
        pid = os.path.basename(os.path.dirname(f))
        idx = int(f.rsplit("_slice_", 1)[1].replace(".npy", ""))
        per_patient[pid][idx] = int((np.load(f) > 0).sum())

    if not per_patient:
        print(f"No mask files found under {mask_root} — did you run "
              f"prepare_meniscus_masks.py with --min_meniscus_pixels 0 "
              f"and --out_root {args.data_root}?")
        return

    pos_lines, withneg_lines = [], []
    total_pos = total_near = total_notch = total_far_kept = total_far_all = 0

    for pid in sorted(per_patient):
        idxs = sorted(per_patient[pid])
        n = idxs[-1] + 1   # assumes 0..n-1 saved (min_meniscus_pixels=0 guarantees this)
        px = np.zeros(n, dtype=np.int64)
        for i in idxs:
            px[i] = per_patient[pid][i]

        pos_idx = [i for i in idxs if px[i] > 0]
        for i in pos_idx:
            pos_lines.append(f"{pid},{i}")
        total_pos += len(pos_idx)

        spans = sorted(find_spans(px), key=lambda s: s[0])
        near_set, notch_set = set(), set()
        for first, last in spans:
            for i in range(max(0, first - args.near_span_radius), first):
                near_set.add(i)
            for i in range(last + 1, min(n, last + 1 + args.near_span_radius)):
                near_set.add(i)
        if len(spans) == 2:
            (f1, l1), (f2, l2) = spans
            for i in range(l1 + 1, f2):
                notch_set.add(i)

        empty_idx = [i for i in idxs if px[i] == 0]
        near_only = sorted(i for i in empty_idx if i in near_set)
        notch_only = sorted(i for i in empty_idx if i in notch_set and i not in near_set)
        far_idx = [i for i in empty_idx if i not in near_set and i not in notch_set]

        far_kept = sorted(rng.sample(far_idx, k=int(round(len(far_idx) * args.far_frac))))

        withneg_lines += [f"{pid},{i}" for i in pos_idx + near_only + notch_only + far_kept]
        total_near += len(near_only)
        total_notch += len(notch_only)
        total_far_kept += len(far_kept)
        total_far_all += len(far_idx)

        print(f"  {pid}: n_slices={n} pos={len(pos_idx)} spans={spans} "
              f"near={len(near_only)} notch={len(notch_only)} "
              f"far_kept={len(far_kept)}/{len(far_idx)}")

    os.makedirs(args.out_dir, exist_ok=True)
    pos_path = os.path.join(args.out_dir, "train_index_posonly.txt")
    withneg_path = os.path.join(args.out_dir, "train_index_withneg.txt")

    with open(pos_path, "w") as fh:
        fh.write("\n".join(pos_lines) + "\n")
    with open(withneg_path, "w") as fh:
        fh.write("\n".join(withneg_lines) + "\n")

    total_neg = total_near + total_notch + total_far_kept
    print(f"\n=== summary ({len(per_patient)} patients) ===")
    print(f"  posonly : {len(pos_lines):,} slices  -> {pos_path}")
    print(f"  withneg : {len(withneg_lines):,} slices "
          f"({total_pos:,} pos + {total_near:,} near-span + "
          f"{total_notch:,} notch + {total_far_kept:,}/{total_far_all:,} far)"
          f"  -> {withneg_path}")
    print(f"  positive:negative ratio in withneg = {total_pos}:{total_neg} "
          f"= 1:{total_neg/total_pos:.2f}" if total_pos else "  (no positives found)")


if __name__ == "__main__":
    main()
