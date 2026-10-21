"""Step-by-step picture of how v14 paints a synthetic tear, on one real slice.

Row 1: the painting formula  I' = I + alpha * g * (F - I)  one ingredient at a time.
Row 2: the "two parts with a tear in the middle" case: what the three training targets
       (whole wedge, visible part, contour ring) look like when a cleft splits a horn.

    venv/bin/python scripts/show_tear_steps.py --out figures/tear_synth_steps.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "segmentation"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tear_synth import _horns, _cleft, contour_band, FLUID_PERCENTILE      # noqa: E402
from torn_compare_utils import process_volume, process_manual_seg           # noqa: E402

NIFTI = "data/iu-dataset/pd-files/AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz"
SEG = "/Users/anshikabajpai/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final/AC0D5A4D78B628_SAG_PD_TSE_6_1_1.seg.nrrd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/tear_synth_steps.png")
    ap.add_argument("--slice", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    vol = process_volume(os.path.join(ROOT, NIFTI), "PD")
    seg = process_manual_seg(SEG)
    img, mask = vol[a.slice].astype(np.float32), seg[a.slice]
    comps = sorted(_horns(mask), key=lambda c: -c.sum())
    comp = comps[0]
    rng = np.random.default_rng(a.seed)
    tear = np.zeros_like(mask)
    for _ in range(50):                               # a cleft running ACROSS the horn, reaching its surface
        tear = _cleft(comp, 90 + rng.uniform(-8, 8), True, rng) & mask
        if tear.sum() >= 8:
            break

    F = float(np.percentile(img, FLUID_PERCENTILE))   # "fluid brightness" of this slice
    g = ndi.gaussian_filter(tear.astype(np.float32), 0.8)
    g = np.clip(g / max(g.max(), 1e-6), 0, 1)         # soft tear mask, 0..1
    w = a.alpha * g
    out = np.clip(img + w * (F - img), 0, 1)          # THE FORMULA  I' = I + alpha*g*(F - I)
    whole = mask
    visible = whole & ~tear
    ring = contour_band(whole)

    ys, xs = np.where(mask)
    pad = 14
    cs = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
    cr = lambda m: m[cs]

    # three example pixels: tear centre, tear edge, far from the tear
    cy, cx = np.unravel_index(np.argmax(g), g.shape)
    edge = np.argwhere((g > 0.25) & (g < 0.6))
    ey, ex = edge[len(edge) // 2]
    oy, ox = np.argwhere(whole & ~ndi.binary_dilation(tear, iterations=6))[0]
    print(f"alpha = {a.alpha}, F (fluid brightness, {FLUID_PERCENTILE}th percentile of slice) = {F:.3f}")
    print(f"{'pixel':<16}{'I':>7}{'g':>7}{'w=alpha*g':>11}{'I_new':>8}   check: I + w*(F-I)")
    for name, (y, x) in (("tear centre", (cy, cx)), ("tear edge", (ey, ex)), ("meniscus, no tear", (oy, ox))):
        print(f"{name:<16}{img[y, x]:>7.3f}{g[y, x]:>7.2f}{w[y, x]:>11.2f}{out[y, x]:>8.3f}   "
              f"{img[y, x]:.3f} + {w[y, x]:.2f}*({F:.3f}-{img[y, x]:.3f}) = {img[y, x] + w[y, x] * (F - img[y, x]):.3f}")

    def show(ax, im, title, cmap="gray", vmin=0, vmax=1):
        ax.imshow(cr(im), cmap=cmap, vmin=vmin, vmax=vmax); ax.set_title(title, fontsize=9); ax.axis("off")

    def outline(ax, m, col, lw=1.4):
        if cr(m).any():
            ax.contour(cr(m).astype(float), [0.5], colors=[col], linewidths=lw)

    fig, axes = plt.subplots(2, 6, figsize=(17, 8.2))
    ax = axes[0]
    show(ax[0], img, "1. original slice I\n(yellow = meniscus mask)"); outline(ax[0], whole, "#ffe600")
    show(ax[1], tear.astype(float), "2. tear mask (hard): a cleft\ndrawn INSIDE the meniscus", "gray")
    show(ax[2], g, "3. soft tear mask g\n(blurred, 0 to 1)", "magma")
    show(ax[3], w, f"4. w = alpha * g\n(alpha = {a.alpha}, how strong)", "magma")
    show(ax[4], out, f"5. result I' = I + w*(F - I)\nF = {F:.2f} (fluid brightness)"); outline(ax[4], whole, "#ffe600", 0.8)
    show(ax[5], out - img, "6. what changed: I' - I", "magma", 0, 0.6)
    ax = axes[1]
    show(ax[0], img, "A. original slice (2 horns; the tear\nwill go on the lower one)"); outline(ax[0], whole, "#ffe600")
    show(ax[1], out, "B. model INPUT after painting\n(bright cleft across the lower horn)")
    show(ax[2], out, "C. training label 'whole wedge'\n(yellow, unchanged) + tear (red)"); outline(ax[2], whole, "#ffe600"); outline(ax[2], tear, "#ff2a2a")
    show(ax[3], out, "D. 'visible part' target (R2+)\n= whole minus tear: 2 pieces"); outline(ax[3], visible, "#00e5ff")
    show(ax[4], out, "E. 'contour ring' target\n(R1c): ONE unbroken outline"); ax[4].imshow(np.ma.masked_less(cr(ring).astype(float), 0.5), cmap="autumn", alpha=0.9)
    ax[5].axis("off")
    ax[5].text(0, 0.95, "Same slice, three targets:\n\n"
               "whole wedge  (C): continuous,\n  the crack is INSIDE it\n\n"
               "visible part  (D): broken in two\n  by the crack\n\n"
               "contour ring (E): continuous,\n  runs straight over the crack",
               va="top", fontsize=10, family="monospace")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
