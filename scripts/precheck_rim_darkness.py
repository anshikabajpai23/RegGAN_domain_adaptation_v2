"""
precheck_rim_darkness.py
=========================
Phase 5, G2 mandatory pre-check (§5.2 of docs/phase5_implementation_ideas.md).

G2 proposes a loss that penalises predictions whose interior is not darker than
the tissue immediately around it. That is only a valid prior if the ordering
actually HOLDS on this data. This script measures it. No training, no model,
no checkpoint -- it reads images and GT masks only.

For every GT-nonempty slice:
    mean_inside = mean intensity where GT == meniscus
    mean_ring   = mean intensity in a k-px ring just OUTSIDE the mask
and reports the fraction of slices where mean_inside < mean_ring.

DECISION RULE (from the doc, do not soften it):
  - real PD holds on < ~0.80 of slices  -> DO NOT implement G2. The ordering
    does not hold on this sequence and the loss would be pushing the model
    toward something false.
  - real PD holds but fake PD does not  -> that is a GAN intensity error. The
    fix is G1 (tissue-conditioned calibration of the existing fake PD), NOT a
    segmentation loss.
  - both hold                           -> G2 is safe to build; use the printed
    median gap to set the loss margin (~25% of it).

Run on the ORIGINAL (uncropped) data. Crop+resize resamples spatially but does
not change intensity values, so the ordering property is identical either way,
and uncropped is the simpler thing to reason about.

Layouts (differ between cohorts -- see dataset_sequence.py):
  real PD : <root>/train/{images,masks}/<pid>_<sidx:04d>.npy        (flat)
  fake PD : <root>/train/{images,masks}/<pid>/<pid>_slice_<idx:03d>.npy

Usage:
  python scripts/precheck_rim_darkness.py \
      --real_root real_pd_seg_data_v7 \
      --fake_root segmentation_data_v2 \
      --n_fake    10 \
      --ring_px   5
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation


def rim_stats(img, gt, k=5):
    """Returns (mean_inside, mean_ring) or None if either region is empty.
    Ring = k-px dilation of the mask, minus the mask itself."""
    g = gt > 0
    if not g.any():
        return None
    ring = binary_dilation(g, iterations=k) & ~g
    if not ring.any():
        return None
    return float(img[g].mean()), float(img[ring].mean())


def collect_real(root, split="train"):
    """Flat layout: <pid>_<sidx>.npy"""
    img_dir = os.path.join(root, split, "images")
    msk_dir = os.path.join(root, split, "masks")
    out = []
    for f in sorted(glob.glob(os.path.join(img_dir, "*.npy"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        m = os.path.join(msk_dir, f"{stem}.npy")
        if os.path.exists(m):
            out.append((stem.rsplit("_", 1)[0], f, m))
    return out


def collect_fake(root, split="train", n_patients=None):
    """Nested layout: <pid>/<pid>_slice_<idx>.npy"""
    img_dir = os.path.join(root, split, "images")
    msk_dir = os.path.join(root, split, "masks")
    pids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(img_dir, "*"))
                  if os.path.isdir(p))
    if n_patients:
        pids = pids[:n_patients]
    out = []
    for pid in pids:
        for f in sorted(glob.glob(os.path.join(img_dir, pid, "*.npy"))):
            m = os.path.join(msk_dir, pid, os.path.basename(f))
            if os.path.exists(m):
                out.append((pid, f, m))
    return out


def analyze(items, cohort, ring_px):
    per_patient = defaultdict(lambda: {"held": 0, "total": 0, "gaps": []})
    gaps = []
    held = total = skipped = 0

    for pid, img_path, msk_path in items:
        img = np.load(img_path).astype(np.float32)
        gt = np.load(msk_path)
        # fake-PD masks carry DESS labels (meniscus 5/6 remapped to 1/2 at prep);
        # either way anything > 0 is meniscus for this purpose
        r = rim_stats(img, gt, k=ring_px)
        if r is None:
            skipped += 1
            continue
        mu_in, mu_ring = r
        gap = mu_ring - mu_in          # positive == interior darker (the prior holds)
        ok = gap > 0
        held += ok
        total += 1
        gaps.append(gap)
        p = per_patient[pid]
        p["held"] += ok
        p["total"] += 1
        p["gaps"].append(gap)

    print(f"\n{'=' * 72}")
    print(f"{cohort}: {total} GT-nonempty slices across {len(per_patient)} patients "
          f"({skipped} slices skipped: empty GT or empty ring)")
    print(f"{'=' * 72}")
    if total == 0:
        print("  NO USABLE SLICES -- check the paths.")
        return None

    frac = held / total
    gaps = np.array(gaps)
    print(f"  Slices where interior is DARKER than rim : {held}/{total} = {frac:.3f}")
    print(f"  Gap (mean_ring - mean_inside):")
    print(f"      median {np.median(gaps):+.4f}   mean {gaps.mean():+.4f}   "
          f"p25 {np.percentile(gaps, 25):+.4f}   p75 {np.percentile(gaps, 75):+.4f}")
    print(f"  Suggested loss margin (~25% of median gap): {0.25 * np.median(gaps):.4f}")

    print(f"\n  Per patient:")
    print(f"    {'patient':<18}{'held/total':>14}{'frac':>9}{'median gap':>13}")
    for pid in sorted(per_patient):
        p = per_patient[pid]
        f = p["held"] / p["total"] if p["total"] else float("nan")
        flag = "   <-- below 0.8" if f < 0.8 else ""
        print(f"    {pid:<18}{p['held']:>6}/{p['total']:<7}{f:>9.3f}"
              f"{np.median(p['gaps']):>13.4f}{flag}")
    return frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", required=True)
    ap.add_argument("--fake_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_fake", type=int, default=10,
                    help="how many fake-PD patients to sample (doc says 10)")
    ap.add_argument("--ring_px", type=int, default=5)
    args = ap.parse_args()

    print(f"Ring width: {args.ring_px} px   |   split: {args.split}")

    real_items = collect_real(args.real_root, args.split)
    fake_items = collect_fake(args.fake_root, args.split, args.n_fake)
    print(f"Found {len(real_items)} real-PD slices, {len(fake_items)} fake-PD slices")

    frac_real = analyze(real_items, "REAL PD (the target domain -- this is the gate)", args.ring_px)
    frac_fake = analyze(fake_items, f"FAKE PD ({args.n_fake} patients)", args.ring_px)

    print(f"\n{'=' * 72}")
    print("VERDICT")
    print(f"{'=' * 72}")
    if frac_real is None:
        print("  Could not evaluate real PD -- fix paths and re-run.")
        return
    if frac_real < 0.80:
        print(f"  real PD holds on only {frac_real:.3f} of slices (< 0.80).")
        print("  -> DO NOT IMPLEMENT G2. The darker-than-rim prior does not hold on")
        print("     this sequence; a loss enforcing it would push the model toward")
        print("     something false. Check whether this PD is fat-saturated.")
    elif frac_fake is not None and frac_fake < 0.80:
        print(f"  real PD holds ({frac_real:.3f}) but fake PD does not ({frac_fake:.3f}).")
        print("  -> This is a GAN intensity error, not a segmentation-loss problem.")
        print("     The fix is G1 (tissue-conditioned calibration of the fake PD),")
        print("     NOT G2. Building G2 would fight the translation stage.")
    else:
        print(f"  real PD {frac_real:.3f}, fake PD {frac_fake:.3f} -- both >= 0.80.")
        print("  -> G2 is safe to build. Use the suggested margin printed above.")


if __name__ == "__main__":
    main()
