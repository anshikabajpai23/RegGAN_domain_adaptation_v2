"""
phase4_4_notch_gap.py
======================
Phase 4 step 4.4 — is the single span on AC13300201B926 / AC149BC218E75C a
stray-voxel bridge across the notch, or genuinely labeled through?

GT loaded exactly as analyze_gt_tapering.py (load_native_binary: SimpleITK ->
DICOMOrient RAS -> transpose(2,1,0) -> binarize, native resolution, no resize).

Prints per-slice GT pixel count across the whole volume, and 3D connected
components at 26-connectivity with their sizes.

Reads as (per spec): any slice inside the expected notch with < 10 px -> stray
voxel bridge; otherwise the notch was genuinely labeled through.

Usage:
  python scripts/phase4_4_notch_gap.py \
      --mask_dir labelled-pd-segmentations \
      --patients AC13300201B926 AC149BC218E75C
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_gt_tapering import load_native_binary

CONN26 = np.ones((3, 3, 3), dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--patients", nargs="+", required=True)
    ap.add_argument("--min_px", type=int, default=10,
                    help="threshold for calling a slice a stray-voxel bridge")
    args = ap.parse_args()

    for pid in args.patients:
        hits = glob.glob(os.path.join(args.mask_dir, f"{pid}*.nrrd"))
        if not hits:
            print(f"{pid}: NO MASK FOUND in {args.mask_dir}"); continue
        vol = load_native_binary(hits[0])
        per = vol.sum(axis=(1, 2))

        print(f"\n################ {pid} ################")
        print(f"file: {os.path.basename(hits[0])}")
        print(f"shape: {vol.shape}   total GT px: {int(vol.sum()):,}")

        print(f"\n--- per-slice GT pixel count (whole volume) ---")
        print(f"| {'slice':>5} | {'px':>6} | {'<min_px':>7} |")
        print(f"|{'-'*7}|{'-'*8}|{'-'*9}|")
        for i, p in enumerate(per):
            flag = "YES" if 0 < p < args.min_px else ""
            print(f"| {i:>5} | {int(p):>6} | {flag:>7} |")

        nz = [i for i, p in enumerate(per) if p > 0]
        print(f"\nnon-empty slice range: {nz[0]}..{nz[-1]}  ({len(nz)} slices)")
        thin = [(i, int(per[i])) for i in nz if per[i] < args.min_px]
        print(f"slices with 0 < px < {args.min_px}: {thin if thin else 'NONE'}")

        lab, n = label(vol, structure=CONN26)
        sizes = sorted(((int((lab == c).sum()), c) for c in range(1, n + 1)), reverse=True)
        print(f"\n--- 3D connected components (26-connectivity) ---")
        print(f"count: {n}")
        for size, c in sizes:
            zs = np.where((lab == c).any(axis=(1, 2)))[0]
            print(f"  component {c}: {size:,} px, slices {zs.min()}..{zs.max()}")

        # what would happen with a min-pixel threshold applied per slice
        gated = per.copy()
        gated[gated < args.min_px] = 0
        gz = [i for i, p in enumerate(gated) if p > 0]
        runs, cur = [], [gz[0]]
        for a, b in zip(gz, gz[1:]):
            if b == a + 1: cur.append(b)
            else: runs.append((cur[0], cur[-1])); cur = [b]
        runs.append((cur[0], cur[-1]))
        print(f"\n--- spans if --min_span_pixels {args.min_px} were applied ---")
        print(f"  spans: {runs}   (count: {len(runs)})")


if __name__ == "__main__":
    main()
