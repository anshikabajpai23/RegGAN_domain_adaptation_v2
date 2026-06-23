"""
boundary_distance_eval.py
===========================
Independent, classical-CV anatomy-preservation check -- does NOT use the
trained registration network R at all (sidesteps Stage 3's unresolved
training-target issue entirely).

Method: for each mask label (1-6) in a DESS mask, find the label region's
boundary pixels. Independently run Canny edge detection on the corresponding
fake_B (pseudo-PD) slice -- NOT informed by the mask at all. Measure how far
each mask-boundary pixel is from the nearest INDEPENDENTLY-detected edge in
fake_B, using a distance transform (efficient: O(n log n), not pairwise).

If G_AB preserved anatomy (translated intensity/contrast only, did not move
structure), fake_B's own detected edges should closely hug the DESS mask's
boundary, even though PD and DESS have inverted/different contrast for the
same tissue. This is a real, quantitative, content-based check -- it answers
"did the generator hallucinate or distort structure," independent of R.

Covers:
  - Meniscus-specific deformation (labels 5, 6 reported separately)
  - Whole-image / all-label anatomy preservation (labels 1-6 aggregated)
  - The "DESS anatomy -> pseudo-PD" structural alignment headline claim

Usage:
    python inference/boundary_distance_eval.py \
        --dess_slice_dir preprocessed_v2/slices/dess \
        --fake_pd_dir    results/stage4_fake_pd \
        --mask_dir       preprocessed_v2/masks \
        --splits         preprocessed_v2/splits.json \
        --split          val \
        --out_json       runs/run_004/boundary_distance_val.json \
        --n_slices       100
"""
import argparse
import glob
import json
import logging
import os
import re
import sys

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from skimage.feature import canny
from skimage.segmentation import find_boundaries

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import extract_patient_id, norm, load_slices_from_nifti  # reuse, don't duplicate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ALL_LABELS = [1, 2, 3, 4, 5, 6]
MENISCUS_LABELS = [5, 6]


def boundary_distances_for_label(mask_slice, edge_dist_transform, label):
    """Returns array of distances (one per boundary pixel of this label's
    region) to the nearest independently-detected edge in fake_B."""
    region = (mask_slice == label)
    if not region.any():
        return None
    boundary = find_boundaries(region, mode="inner")
    if not boundary.any():
        return None
    return edge_dist_transform[boundary]


def load_patient_aligned_triplets(dess_slice_dir, fake_pd_dir, mask_dir, splits_path, split, max_patients=None):
    """Same patient+index alignment logic as evaluate.py's Stage 4a fix --
    keyed by patient ID, zipped within-patient by matching slice index."""
    with open(splits_path) as f:
        splits = json.load(f)
    split_pids = set(extract_patient_id(p) for p in splits["dess"][split])

    dess_by_patient = {}
    for f in sorted(glob.glob(os.path.join(dess_slice_dir, "*.npy"))):
        m = re.search(r"_sl(\d{4})\.npy$", os.path.basename(f))
        if not m:
            continue
        idx = int(m.group(1))
        pid = extract_patient_id(f)
        if pid not in split_pids:
            continue
        dess_by_patient.setdefault(pid, {})[idx] = f

    fake_pd_path_by_patient = {}
    for f in sorted(glob.glob(os.path.join(fake_pd_dir, "*.nii.gz"))):
        pid = extract_patient_id(f)
        if pid in split_pids:
            fake_pd_path_by_patient[pid] = f

    mask_path_by_patient = {}
    for f in sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz"))):
        pid = extract_patient_id(f)
        if pid in split_pids:
            mask_path_by_patient[pid] = f

    common_pids = sorted(set(dess_by_patient) & set(fake_pd_path_by_patient) & set(mask_path_by_patient))
    if max_patients:
        common_pids = common_pids[:max_patients]
    log.info(f"Patients in split='{split}' with DESS+fake-PD+mask: {len(common_pids)}")

    for pid in common_pids:
        fake_vol = load_slices_from_nifti(fake_pd_path_by_patient[pid])
        mask_vol = load_slices_from_nifti(mask_path_by_patient[pid])
        for idx in sorted(dess_by_patient[pid].keys()):
            if idx >= len(fake_vol) or idx >= len(mask_vol):
                continue
            yield pid, idx, np.load(dess_by_patient[pid][idx]), fake_vol[idx], mask_vol[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dess_slice_dir", required=True)
    ap.add_argument("--fake_pd_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--n_slices", type=int, default=200,
                     help="Max slices to evaluate (for speed)")
    ap.add_argument("--canny_sigma", type=float, default=1.5)
    args = ap.parse_args()

    per_label_distances = {lbl: [] for lbl in ALL_LABELS}
    n_processed = 0

    for pid, idx, dess_sl, fake_sl, mask_sl in load_patient_aligned_triplets(
        args.dess_slice_dir, args.fake_pd_dir, args.mask_dir, args.splits, args.split
    ):
        if n_processed >= args.n_slices:
            break

        mask_sl = mask_sl.astype(np.int64)
        if not np.isin(mask_sl, ALL_LABELS).any():
            continue  # skip slices with no labeled tissue at all

        fake_norm = norm(fake_sl)
        edges = canny(fake_norm, sigma=args.canny_sigma)
        # distance from every pixel to the nearest detected edge
        edge_dist = distance_transform_edt(~edges)

        any_label_present = False
        for lbl in ALL_LABELS:
            d = boundary_distances_for_label(mask_sl, edge_dist, lbl)
            if d is not None:
                per_label_distances[lbl].extend(d.tolist())
                any_label_present = True

        if any_label_present:
            n_processed += 1

    log.info(f"Processed {n_processed} slices")

    results = {"n_slices_evaluated": n_processed, "per_label": {}}
    for lbl in ALL_LABELS:
        d = np.array(per_label_distances[lbl])
        if len(d) == 0:
            results["per_label"][str(lbl)] = None
            continue
        results["per_label"][str(lbl)] = {
            "n_boundary_pixels": len(d),
            "mean_boundary_distance_px": float(d.mean()),
            "median_boundary_distance_px": float(np.median(d)),
            "p95_boundary_distance_px": float(np.percentile(d, 95)),
            "max_boundary_distance_px": float(d.max()),  # one-directional Hausdorff
        }
        log.info(f"  Label {lbl}: mean={d.mean():.3f}px  p95={np.percentile(d,95):.3f}px  "
                 f"max={d.max():.3f}px  (n={len(d)} boundary pixels)")

    # Meniscus-specific headline (labels 5, 6)
    meniscus_d = np.concatenate([np.array(per_label_distances[l]) for l in MENISCUS_LABELS
                                  if len(per_label_distances[l]) > 0])
    if len(meniscus_d) > 0:
        results["meniscus_mean_boundary_distance_px"] = float(meniscus_d.mean())
        results["meniscus_max_boundary_distance_px"] = float(meniscus_d.max())
        log.info(f"\n  MENISCUS (labels 5,6) mean boundary distance: {meniscus_d.mean():.3f}px")

    # All-label aggregate ("anatomy change" headline)
    all_d = np.concatenate([np.array(per_label_distances[l]) for l in ALL_LABELS
                             if len(per_label_distances[l]) > 0])
    if len(all_d) > 0:
        results["all_labels_mean_boundary_distance_px"] = float(all_d.mean())
        results["all_labels_max_boundary_distance_px"] = float(all_d.max())
        log.info(f"  ALL LABELS mean boundary distance: {all_d.mean():.3f}px")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
