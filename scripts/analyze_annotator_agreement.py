"""
analyze_annotator_agreement.py
================================
Debug test 4, Part B — human-vs-human agreement on the SAME volume, labeled
twice (blind second pass — see the "must be blind" caveat in the debug plan).
Gives the actual ceiling: dice_full is the number the model can't beat
reliably; dice_edge tells us how much of the model's error is really at the
same spots two humans disagree about.

Requires the SAME preprocessed volume for both annotation passes (same slice
thickness, no resampling difference) — this script asserts shapes match and
refuses to silently resample if they don't, same rule as the other debug
tests in this project.

Usage:
  python scripts/analyze_annotator_agreement.py \
      --old_mask old_annotator/AC0BA0AE159EF9.seg.nrrd \
      --new_mask new_annotator/AC0BA0AE159EF9.seg.nrrd \
      --patient AC0BA0AE159EF9
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from analyze_gt_tapering import load_native_binary, find_spans


def dice_on_slices(old, new, slice_mask):
    """Dice restricted to the Z-slices where slice_mask is True."""
    if not slice_mask.any():
        return float("nan"), 0
    o = old[slice_mask]
    n = new[slice_mask]
    inter = (o & n).sum()
    denom = o.sum() + n.sum()
    dice = float(2 * inter / denom) if denom > 0 else 1.0
    return dice, int(slice_mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_mask", required=True)
    ap.add_argument("--new_mask", required=True)
    ap.add_argument("--patient", required=True)
    args = ap.parse_args()

    old = load_native_binary(args.old_mask)
    new = load_native_binary(args.new_mask)

    if old.shape != new.shape:
        raise ValueError(
            f"Shape mismatch: old {old.shape} vs new {new.shape} for {args.patient}. "
            f"Refusing to silently resample — re-export both from the SAME "
            f"preprocessed volume (same slice thickness), per the debug plan's "
            f"requirement.")

    n_slices = old.shape[0]

    # dice_full — whole volume, no restriction
    inter_full = (old & new).sum()
    denom_full = old.sum() + new.sum()
    dice_full = float(2 * inter_full / denom_full) if denom_full > 0 else 1.0

    # spans per mask, independently
    old_px = old.sum(axis=(1, 2))
    new_px = new.sum(axis=(1, 2))
    old_spans = sorted(find_spans(old_px), key=lambda s: s[0])
    new_spans = sorted(find_spans(new_px), key=lambda s: s[0])

    print(f"=== {args.patient} ===")
    print(f"  shape: {old.shape}")
    print(f"  old spans: {old_spans}")
    print(f"  new spans: {new_spans}")
    if len(old_spans) != 2 or len(new_spans) != 2:
        print(f"  ⚠️  expected 2 spans each — got {len(old_spans)} old, {len(new_spans)} new. "
             f"span_offset below may be unreliable.")

    # edge slice set: within ±1 of ANY span endpoint, from EITHER mask
    edge_idx = set()
    for spans in (old_spans, new_spans):
        for first, last in spans:
            for i in (first - 1, first, first + 1, last - 1, last, last + 1):
                if 0 <= i < n_slices:
                    edge_idx.add(i)
    edge_mask = np.zeros(n_slices, dtype=bool)
    edge_mask[list(edge_idx)] = True

    # core: non-empty in BOTH, and NOT in the edge set
    nonempty_both = (old_px > 0) & (new_px > 0)
    core_mask = nonempty_both & ~edge_mask

    dice_core, n_core   = dice_on_slices(old, new, core_mask)
    dice_edge, n_edge   = dice_on_slices(old, new, edge_mask)

    print(f"\n  dice_full : {dice_full:.4f}  ({n_slices} slices)")
    print(f"  dice_core : {dice_core:.4f}  ({n_core} slices, non-empty in both, "
         f"excluding ±1 of any span end)")
    print(f"  dice_edge : {dice_edge:.4f}  ({n_edge} slices, within ±1 of any span end)")

    # span_offset — pair spans by position (span 1 <-> span 1, span 2 <-> span 2)
    print(f"\n  span_offset (slices apart, old vs new):")
    for i in range(min(len(old_spans), len(new_spans))):
        o_first, o_last = old_spans[i]
        n_first, n_last = new_spans[i]
        print(f"    span {i+1}: first_offset={abs(o_first - n_first)}  "
             f"last_offset={abs(o_last - n_last)}   "
             f"(old=[{o_first},{o_last}]  new=[{n_first},{n_last}])")

    print(f"\n=== how to read it ===")
    if 0.80 <= dice_full <= 0.84:
        print(f"  dice_full={dice_full:.3f} in 0.80-0.84 -> model at ~0.78 is within "
             f"~0.04 of the human-vs-human ceiling. Report the ceiling; remaining "
             f"gain is small.")
    elif dice_full >= 0.90:
        print(f"  dice_full={dice_full:.3f} >= 0.90 -> labels are consistent between "
             f"annotators. Model's edge error is genuinely the model's, not label noise.")
        if dice_edge <= 0.6:
            print(f"  AND dice_edge={dice_edge:.3f} <= 0.6 -> core is fine, ends are where "
                 f"disagreement concentrates anyway. Same conclusion as the row above: "
                 f"P2 (negatives) is the lever, consider excluding ±1 edge slices from loss.")
    else:
        print(f"  dice_full={dice_full:.3f} — outside the two clean buckets above. "
             f"Read dice_core vs dice_edge directly rather than forcing a verdict:")
        print(f"    core={dice_core:.3f}, edge={dice_edge:.3f} — "
             f"{'core fine, edges noisy' if dice_core > 0.85 and dice_edge < 0.7 else 'inspect manually'}")


if __name__ == "__main__":
    main()
