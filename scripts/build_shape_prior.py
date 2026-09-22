"""
build_shape_prior.py
=====================
Phase 5, D prep — a position-conditioned shape template (doc §4.3).

WHAT IT BUILDS
For each bin of WITHIN-SPAN position q (0 = outer end, 1 = notch end), the mean
and std of the meniscus area in pixels. That is all: a table of "how big should
the cross-section be at this point along its own span". Used in training as a
weak area regulariser (doc §4.4).

Deliberately registration-free (doc §4.3): no attempt to align slices across
patients, because area needs no alignment. The component-count statistic the doc
mentions is NOT collected -- doc §4.4 says not to use it in the first run (it
needs a differentiable proxy, and clDice already measured ~0 on this data).

ORIENTATION, SOLVED WITHOUT KNOWING LATERALITY
The doc flags laterality as unknown (§0 ⚠) and §4.3 says to flip q for the medial
span. That is not actually needed: with two spans per patient, the NOTCH is by
definition the gap BETWEEN them, so the notch end of each span is the end facing
the other span. No left/right knowledge required:
    lower span : q = (s - first) / (last - first)     -> 0 at first (outer)
    upper span : q = (last - s) / (last - first)      -> 0 at last  (outer)
Patients with a single span (notch labelled through) are SKIPPED: their notch end
cannot be identified, and guessing would poison the template. Both cohorts came
out at ~2.00 spans/patient in prep_slice_weights, so this drops very little.

⚠ BUILD THIS FROM THE CROPPED MASKS. The crop magnifies the meniscus ~2.5x in
area, and the model D regularises predicts in cropped space. A template built on
uncropped data would be systematically ~2.5x too small and would push the model
to under-segment.

Usage:
  python scripts/build_shape_prior.py \
      --data_root real_pd_seg_data_v7_crop --layout flat \
      --out_json  shape_prior_real.json --bins 10
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def spans_from_indices(idx_sorted):
    if len(idx_sorted) == 0:
        return []
    arr = np.asarray(sorted(idx_sorted))
    runs = np.split(arr, np.where(np.diff(arr) > 1)[0] + 1)
    return [(int(r[0]), int(r[-1])) for r in runs]


def within_span_q(s, span, is_lower_span):
    """0 = outer end, 1 = notch end. Orientation from span order, not laterality."""
    first, last = span
    if last == first:
        return 0.0
    if is_lower_span:
        return (s - first) / (last - first)
    return (last - s) / (last - first)


def collect(data_root, layout, split):
    """Returns {pid: {slice_idx: area_px}} over GT-nonempty slices."""
    per_pid = defaultdict(dict)
    if layout == "nested":
        for f in glob.glob(os.path.join(data_root, split, "masks", "*", "*.npy")):
            pid = os.path.basename(os.path.dirname(f))
            m = re.search(r"_slice_(\d{3})\.npy$", f)
            if not m:
                continue
            a = int((np.load(f) > 0).sum())
            if a > 0:
                per_pid[pid][int(m.group(1))] = a
    else:
        for f in glob.glob(os.path.join(data_root, split, "masks", "*.npy")):
            stem = os.path.splitext(os.path.basename(f))[0]
            pid, sidx = stem.rsplit("_", 1)
            a = int((np.load(f) > 0).sum())
            if a > 0:
                per_pid[pid][int(sidx)] = a
    return per_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="CROPPED root -- see module docstring")
    ap.add_argument("--layout", required=True, choices=["nested", "flat"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    per_pid = collect(args.data_root, args.layout, args.split)
    if not per_pid:
        raise SystemExit(f"No GT-nonempty masks under {args.data_root}/{args.split}/masks")

    K = args.bins
    areas = [[] for _ in range(K)]
    n_skipped_single, n_used = 0, 0
    # slice_idx -> bin, needed by training (spans are known from GT at train time
    # but NOT at inference -- doc §4.2, which is why D is training-time only)
    bin_map = defaultdict(dict)

    for pid, sl in sorted(per_pid.items()):
        spans = spans_from_indices(sorted(sl))
        if len(spans) < 2:
            n_skipped_single += 1
            continue
        lower, upper = spans[0], spans[-1]
        for span, is_lower in ((lower, True), (upper, False)):
            for s in range(span[0], span[1] + 1):
                if s not in sl:
                    continue
                q = within_span_q(s, span, is_lower)
                b = min(int(q * K), K - 1)
                areas[b].append(sl[s])
                bin_map[pid][str(s)] = b
                n_used += 1

    print(f"Patients: {len(per_pid)}   used: {len(per_pid) - n_skipped_single}   "
          f"skipped (single span): {n_skipped_single}")
    print(f"Slices binned: {n_used}   bins: {K}\n")
    print(f"  {'bin':>4}{'q range':>14}{'n':>7}{'area mean':>12}{'area std':>11}")

    a_mean, a_std = [], []
    for b in range(K):
        v = np.array(areas[b], dtype=np.float64)
        if v.size == 0:
            # keep the bin but make it inert: a zero-weight bin would otherwise
            # divide by ~1 and dominate the loss
            a_mean.append(0.0); a_std.append(0.0)
            print(f"  {b:>4}{f'{b/K:.1f}-{(b+1)/K:.1f}':>14}{0:>7}{'EMPTY':>12}{'-':>11}")
            continue
        a_mean.append(float(v.mean())); a_std.append(float(v.std()))
        print(f"  {b:>4}{f'{b/K:.1f}-{(b+1)/K:.1f}':>14}{v.size:>7}"
              f"{v.mean():>12.1f}{v.std():>11.1f}")

    nonempty = [m for m in a_mean if m > 0]
    if nonempty:
        print(f"\n  outer end (bin 0)  mean area: {a_mean[0]:.1f}")
        print(f"  notch end (bin {K-1}) mean area: {a_mean[K-1]:.1f}")
        print("  EXPECTED: outer end LARGER than notch end. Confirmed by this")
        print("  project's own Check 4 (docs/phase4_results.md): outer ends sit at")
        print("  median 0.93 of peak area, inner/notch ends at 0.39 -- same span-end")
        print("  definition used here. If instead bin 0 were SMALLER than bin 9, the")
        print("  orientation convention would be inverted.")

    out = {
        "meta": {"data_root": args.data_root, "layout": args.layout, "split": args.split,
                 "bins": K, "n_patients_used": len(per_pid) - n_skipped_single,
                 "n_skipped_single_span": n_skipped_single, "n_slices": n_used},
        "area_mean": a_mean,
        "area_std": a_std,
        "bin_map": {pid: d for pid, d in bin_map.items()},
    }
    with open(args.out_json, "w") as fh:
        json.dump(out, fh)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
