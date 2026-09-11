"""
per_slice_from_predictions.py
==============================
Emits a per-slice CSV from a directory of prediction NIfTIs + the 17-patient GT,
in EXACTLY the format build_hallucination_table.py consumes:

    patient_id, slice_idx, dice, gt_voxels, pred_voxels, false_neg, false_pos

so any variant's predictions (debug test 2's A/B/C, or any infer_real_pd_v3.py
output) can be pushed through the same empty-slice-FP / on-slice-FP / FN split
and hallucination-distance analysis used in debug test 1.

GT loading is the same convention as compare_stack_variants.py / the abstract
notebook (SimpleITK -> DICOMOrient RAS -> isotropic in-plane -> 384x384 order=0
-> threshold). Shape mismatches raise rather than silently resampling.

NOTE on comparability: distances are in SLICES. These 17 patients are real PD
(~3.6mm slice spacing), so slice counts here are directly comparable to the
2-patient real-val cohort, but NOT to the fake-PD val cohort (~0.7mm spacing,
~5.25x finer). Do not compare slice-distance thresholds across those cohorts.

Usage:
  python scripts/per_slice_from_predictions.py \
      --pred_dir results/debug2_stack_variants/B \
      --out_csv  results/debug2_stack_variants/per_slice_B.csv
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_stack_variants import (load_nrrd_binary, merge_meniscus, get_pid,
                                    patients_in_dir, load_vol,
                                    ALL_PATIENTS, DEFAULT_GT_D1, DEFAULT_GT_NEW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--gt_dir_d1", default=DEFAULT_GT_D1)
    ap.add_argument("--gt_dir_new", default=DEFAULT_GT_NEW)
    args = ap.parse_args()

    preds = patients_in_dir(args.pred_dir)
    print(f"predictions found: {len(preds)}")

    eps = 1e-6
    rows = []
    for pid in ALL_PATIENTS:
        hits = (glob.glob(os.path.join(args.gt_dir_d1, f"{pid}*.seg.nrrd")) or
                glob.glob(os.path.join(args.gt_dir_new, f"{pid}*.seg.nrrd")))
        if not hits:
            print(f"  WARNING: no GT for {pid}, skipping"); continue
        if pid not in preds:
            print(f"  WARNING: no prediction for {pid}, skipping"); continue

        gt = load_nrrd_binary(hits[0])
        pr = merge_meniscus(load_vol(preds[pid]))
        if gt.shape != pr.shape:
            raise ValueError(f"shape mismatch {pid}: gt {gt.shape} vs pred {pr.shape} "
                             f"— refusing to silently resample")

        for s in range(gt.shape[0]):
            g, p = gt[s], pr[s]
            tp = float((g & p).sum()); gs = float(g.sum()); ps = float(p.sum())
            rows.append({
                "patient_id":  pid,
                "slice_idx":   s,
                "dice":        f"{(2*tp + eps) / (ps + gs + eps):.4f}",
                "gt_voxels":   int(gs),
                "pred_voxels": int(ps),
                "false_neg":   int(gs - tp),
                "false_pos":   int(ps - tp),
            })

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient_id", "slice_idx", "dice",
                                           "gt_voxels", "pred_voxels",
                                           "false_neg", "false_pos"])
        w.writeheader(); w.writerows(rows)

    n_pat = len({r["patient_id"] for r in rows})
    n_empty = sum(1 for r in rows if r["gt_voxels"] == 0)
    print(f"wrote {len(rows)} slice rows across {n_pat} patients -> {args.out_csv}")
    print(f"  empty-GT slices: {n_empty} / {len(rows)}")


if __name__ == "__main__":
    main()
