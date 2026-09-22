"""
build_slice_weights.py
=======================
Phase 5, A3 prep (§3.2 of docs/phase5_implementation_ideas.md).

Computes a per-slice loss weight based on how close that slice is to the end of
a meniscus span, and writes both variants in ONE pass so the prep runs once and
serves both A3 experiments:

    variant "up"   : w = 1 + alpha*k            -> end slices count up to 2x
    variant "down" : w = clip(1 - beta*k, 0.3, 1) -> end slices count as little as 0.3x
    where k = exp(-d_mm / tau_mm), d_mm = distance to the nearest span end.

Distance is in MILLIMETRES, not slice count. That is the whole point: DESS-derived
fake PD is 0.8 mm/slice and real PD is 3.6 mm/slice, so "2 slices away" means
completely different things in the two cohorts (this is Correction C1 in
reggan_dice_debug.md, the exact mistake that invalidated an earlier analysis).

HOW SPANS ARE FOUND, and why it differs per cohort
---------------------------------------------------
A "span" is a contiguous run of slices that contain meniscus. Its first and last
slice are the "ends" that A3 up/down-weights.

  fake PD (nested layout, <pid>/<pid>_slice_<idx>.npy):
      segmentation_data_v2* was prepped with min_meniscus_pixels=10, so every
      slice STORED already contains meniscus and meniscus-free slices are absent
      from disk. Spans are therefore reconstructed from which slice indices are
      PRESENT -- a gap in the indices is a genuine gap in the meniscus. Loading
      24k masks to re-derive this would read ~29 GB to learn what the filenames
      already say. --verify_fake samples a few masks to confirm the assumption.

  real PD (flat layout, <pid>_<sidx>.npy):
      may retain meniscus-free slices, so presence on disk does NOT imply
      nonempty. Masks are loaded to determine emptiness (15 patients x ~36
      slices, cheap).

Empty-GT slices adjacent to a span keep a weight from the same formula, which is
intended: an empty slice next to a span end is exactly where hallucination
happens, so "up" pushes on it and "down" backs off it.

Output JSON:
    {"meta": {...}, "up": {pid: {slice_idx: w}}, "down": {pid: {slice_idx: w}}}

Usage:
  python scripts/build_slice_weights.py \
      --data_root  segmentation_data_v2_crop \
      --layout     nested --spacing_mm 0.8 \
      --out_json   slice_weights_fake.json
  python scripts/build_slice_weights.py \
      --data_root  real_pd_seg_data_v7_crop \
      --layout     flat --spacing_mm 3.6 \
      --out_json   slice_weights_real.json
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def spans_from_indices(idx_sorted):
    """Contiguous runs of present indices -> list of (first, last)."""
    if len(idx_sorted) == 0:
        return []
    arr = np.asarray(sorted(idx_sorted))
    runs = np.split(arr, np.where(np.diff(arr) > 1)[0] + 1)
    return [(int(r[0]), int(r[-1])) for r in runs]


def weights_for_volume(all_indices, nonempty_indices, spacing_mm,
                       alpha=1.0, beta=0.7, tau_mm=3.6):
    """all_indices: every slice index the dataset will serve for this patient.
    nonempty_indices: those containing meniscus (defines the spans).
    Returns {"up": {idx: w}, "down": {idx: w}}."""
    spans = spans_from_indices(nonempty_indices)
    ends = [e for s in spans for e in s]

    z = np.asarray(sorted(all_indices))
    if ends:
        d_slices = np.abs(z[:, None] - np.asarray(ends)[None, :]).min(axis=1)
        d_mm = d_slices.astype(np.float64) * spacing_mm
    else:
        d_mm = np.full(len(z), 1e9)

    k = np.exp(-d_mm / tau_mm)
    w_up = 1.0 + alpha * k
    w_down = np.clip(1.0 - beta * k, 0.3, 1.0)
    return (
        {int(i): float(w) for i, w in zip(z, w_up)},
        {int(i): float(w) for i, w in zip(z, w_down)},
        spans,
    )


def collect_nested(data_root, split):
    """fake PD. Returns {pid: (all_idx, nonempty_idx)} -- present implies nonempty."""
    mask_dir = os.path.join(data_root, split, "masks")
    per_pid = defaultdict(list)
    for f in glob.glob(os.path.join(mask_dir, "*", "*.npy")):
        pid = os.path.basename(os.path.dirname(f))
        m = re.search(r"_slice_(\d{3})\.npy$", f)
        if m:
            per_pid[pid].append(int(m.group(1)))
    return {pid: (sorted(v), sorted(v)) for pid, v in per_pid.items()}


def collect_flat(data_root, split):
    """real PD. Loads masks -- presence on disk does NOT imply nonempty."""
    mask_dir = os.path.join(data_root, split, "masks")
    per_pid_all = defaultdict(list)
    per_pid_nonempty = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(mask_dir, "*.npy"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        pid, sidx = stem.rsplit("_", 1)
        sidx = int(sidx)
        per_pid_all[pid].append(sidx)
        if np.load(f).any():
            per_pid_nonempty[pid].append(sidx)
    return {pid: (sorted(v), sorted(per_pid_nonempty.get(pid, [])))
            for pid, v in per_pid_all.items()}


def verify_fake_assumption(data_root, split, n=20):
    """Sample stored fake-PD masks and confirm they really are all nonempty."""
    files = glob.glob(os.path.join(data_root, split, "masks", "*", "*.npy"))
    if not files:
        return None
    rng = np.random.default_rng(42)
    sample = rng.choice(files, size=min(n, len(files)), replace=False)
    empty = sum(1 for f in sample if not np.load(f).any())
    return len(sample), empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--layout", required=True, choices=["nested", "flat"],
                    help="nested = fake PD (<pid>/<pid>_slice_<idx>.npy); flat = real PD")
    ap.add_argument("--split", default="train")
    ap.add_argument("--spacing_mm", type=float, required=True,
                    help="0.8 for DESS-derived fake PD, 3.6 for real PD")
    ap.add_argument("--tau_mm", type=float, default=3.6)
    ap.add_argument("--alpha", type=float, default=1.0, help="up-variant strength")
    ap.add_argument("--beta", type=float, default=0.7, help="down-variant strength")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--verify_fake", action="store_true",
                    help="sample stored fake-PD masks to confirm present==nonempty")
    args = ap.parse_args()

    if args.layout == "nested":
        per_pid = collect_nested(args.data_root, args.split)
        if args.verify_fake:
            res = verify_fake_assumption(args.data_root, args.split)
            if res:
                n, empty = res
                print(f"Assumption check: sampled {n} stored fake-PD masks, {empty} were empty")
                if empty:
                    print("  WARNING: stored fake-PD masks are NOT all nonempty. Spans derived")
                    print("           from filenames will be wrong. Switch to loading masks.")
    else:
        per_pid = collect_flat(args.data_root, args.split)

    if not per_pid:
        raise SystemExit(f"No masks found under {args.data_root}/{args.split}/masks")

    up_all, down_all = {}, {}
    n_spans_total, n_slices_total = 0, 0
    up_vals, down_vals = [], []

    for pid, (all_idx, nonempty_idx) in sorted(per_pid.items()):
        up, down, spans = weights_for_volume(
            all_idx, nonempty_idx, args.spacing_mm,
            alpha=args.alpha, beta=args.beta, tau_mm=args.tau_mm)
        up_all[pid], down_all[pid] = up, down
        n_spans_total += len(spans)
        n_slices_total += len(all_idx)
        up_vals += list(up.values())
        down_vals += list(down.values())

    up_vals = np.array(up_vals); down_vals = np.array(down_vals)
    print(f"\nPatients: {len(per_pid)}   slices: {n_slices_total}   spans: {n_spans_total} "
          f"({n_spans_total / max(len(per_pid), 1):.2f} per patient)")
    print(f"spacing {args.spacing_mm} mm/slice   tau {args.tau_mm} mm   "
          f"alpha {args.alpha}   beta {args.beta}")
    print(f"  up  : min {up_vals.min():.3f}  median {np.median(up_vals):.3f}  max {up_vals.max():.3f}")
    print(f"  down: min {down_vals.min():.3f}  median {np.median(down_vals):.3f}  max {down_vals.max():.3f}")

    # a couple of patients printed in full, so the shape is eyeballable
    for pid in list(sorted(per_pid))[:2]:
        all_idx, nonempty_idx = per_pid[pid]
        spans = spans_from_indices(nonempty_idx)
        print(f"\n  {pid}: {len(all_idx)} slices, spans {spans}")
        show = sorted(up_all[pid])[:14]
        print(f"    idx : {[i for i in show]}")
        print(f"    up  : {[round(up_all[pid][i], 2) for i in show]}")
        print(f"    down: {[round(down_all[pid][i], 2) for i in show]}")

    out = {
        "meta": {
            "data_root": args.data_root, "layout": args.layout, "split": args.split,
            "spacing_mm": args.spacing_mm, "tau_mm": args.tau_mm,
            "alpha": args.alpha, "beta": args.beta,
            "n_patients": len(per_pid), "n_slices": n_slices_total,
        },
        "up": up_all,
        "down": down_all,
    }
    with open(args.out_json, "w") as fh:
        json.dump(out, fh)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
