"""Visual check of v14 synthetic tears before any training.

Paints synthetic tears (segmentation/tear_synth.py) on real PD slices that have a
meniscus mask, and saves a grid: original | with synthetic tear | tear outline,
zoomed on the meniscus. Run it on a healthy patient (scan + its GT) and compare
by eye with your real torn scans.

    venv/bin/python scripts/preview_tear_synth.py \
        --nifti data/iu-dataset/pd-files/AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz \
        --seg   /path/to/AC0D5A4D78B628.seg.nrrd \
        --out   figures/tear_synth_preview.png --n 8
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "segmentation"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tear_synth import synth_tear                         # noqa: E402
from torn_compare_utils import process_volume, process_manual_seg   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="number of examples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad", type=int, default=20)
    a = ap.parse_args()

    vol = process_volume(a.nifti, "PD")
    seg = process_manual_seg(a.seg).astype(np.int64)
    rng = np.random.default_rng(a.seed)
    cands = [s for s in range(1, vol.shape[0] - 1) if seg[s].sum() > 50]
    picks = [cands[i] for i in rng.choice(len(cands), size=min(a.n, len(cands)), replace=False)]

    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)), squeeze=False)
    for r, s in enumerate(sorted(picks)):
        img = vol[s - 1:s + 2].astype(np.float32)
        out, tear = img, np.zeros(seg[s].shape, bool)
        for _ in range(20):                      # retry until a tear is actually painted
            out, tear, pats = synth_tear(img, seg[s], rng, p=1.0)
            if tear.any():
                break
        ys, xs = np.where(seg[s] > 0)
        y0, y1 = max(ys.min() - a.pad, 0), ys.max() + a.pad
        x0, x1 = max(xs.min() - a.pad, 0), xs.max() + a.pad
        crop = (slice(y0, y1), slice(x0, x1))
        for c, (im, title) in enumerate(((img[1], "original"), (out[1], f"synthetic: {', '.join(pats)}"),
                                         (out[1], "tear (red) / meniscus (yellow)"))):
            ax = axes[r, c]; ax.imshow(im[crop], cmap="gray", vmin=0, vmax=1); ax.axis("off")
            ax.set_title(f"slice {s}: {title}" if c == 0 else title, fontsize=8)
            if c == 2:
                ax.contour((seg[s] > 0)[crop].astype(float), [0.5], colors=["#ffe600"], linewidths=1)
                if tear.any():
                    ax.contour(tear[crop].astype(float), [0.5], colors=["#ff2a2a"], linewidths=1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("saved", a.out)


if __name__ == "__main__":
    main()
