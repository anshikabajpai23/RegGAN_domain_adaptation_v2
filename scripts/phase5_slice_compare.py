"""Per-slice, zoomed, outline-only comparison of several runs vs GT (used by the Phase 5 notebook).

Layout per patient: rows = slices where GT has meniscus; columns =
    Image | baseline | approach 1 | ... | approach N | GT
Each mask column is its OWN panel (no overlay): the image with just that mask's boundary outline.
The zoom is one square crop per patient around the union of GT and every shown prediction.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "preprocess"))   # local layout; BigRed has preprocess.py at the root
from preprocess import process_volume            # noqa: E402

TARGET = 384
GT_COLOR = "#ffe600"
# bright, mutually distinct on grayscale; one per prediction column (baseline first)
PALETTE = ["#ff2a2a", "#00e5ff", "#ff9f1a", "#39ff14", "#ff4df0", "#4d7cff", "#b26bff", "#00ffb3", "#ff8fb1"]


def load_raw(pd_root, pid):
    hits = sorted(glob.glob(os.path.join(str(pd_root), f"{pid}*.nii.gz")))
    if not hits:
        return None
    return process_volume(hits[0], "PD")


def _dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else 2.0 * (a & b).sum() / s


def _square_box(masks, pad):
    m = np.zeros_like(masks[0], dtype=bool)
    for x in masks:
        m |= x
    a, s = np.where(m.any(0))
    if len(a) == 0:
        return (0, TARGET, 0, TARGET)
    a0, a1, s0, s1 = a.min() - pad, a.max() + pad + 1, s.min() - pad, s.max() + pad + 1
    side = int(min(max(a1 - a0, s1 - s0), TARGET))
    ca, cs = (a0 + a1) // 2, (s0 + s1) // 2
    a0 = int(np.clip(ca - side // 2, 0, TARGET - side)); s0 = int(np.clip(cs - side // 2, 0, TARGET - side))
    return (a0, a0 + side, s0, s0 + side)


def plot_patient_comparison(pid, vol, gt, preds, patient_dice=None, pad=25, panel=2.6):
    """vol: (n,384,384) float; gt: bool (n,384,384); preds: {label: bool (n,384,384)} in column order
    (baseline first). patient_dice: optional {label: float} shown in the column titles.
    Returns the figure, or None if GT has no meniscus slices."""
    rows = np.where(gt.any((1, 2)))[0]
    if len(rows) == 0:
        print(f"[{pid}] GT has no meniscus slices -- nothing to plot")
        return None
    labels = list(preds)
    # zoom box covers GT + every shown prediction on ALL slices, so stray false positives stay in frame
    a0, a1, s0, s1 = _square_box([gt] + [preds[k] for k in labels], pad)
    ncol = len(labels) + 2
    fig, axes = plt.subplots(len(rows), ncol, figsize=(panel * ncol, panel * len(rows)), squeeze=False)

    def outline(ax, mask, color):
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.5)

    for r, s in enumerate(rows):
        img = vol[s, a0:a1, s0:s1]
        g = gt[s, a0:a1, s0:s1]
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"slice {s}", fontsize=12)
                if r == 0: ax.set_title("Image", fontsize=12)
            elif c == ncol - 1:
                outline(ax, g, GT_COLOR)
                if r == 0: ax.set_title("GT", fontsize=12)
            else:
                k = labels[c - 1]
                outline(ax, preds[k][s, a0:a1, s0:s1], PALETTE[(c - 1) % len(PALETTE)])
                ax.set_xlabel(f"dice {_dice(preds[k][s], gt[s]):.2f}" if gt[s].any() else "", fontsize=10, labelpad=1)
                if r == 0:
                    t = k if not patient_dice or np.isnan(patient_dice.get(k, np.nan)) else f"{k}\n{patient_dice[k]:.3f}"
                    ax.set_title(t, fontsize=12, color=PALETTE[(c - 1) % len(PALETTE)], fontweight="bold")
    fig.suptitle(f"{pid}  (zoomed; outlines only; column title = patient 3D Dice, panel label = per-slice Dice)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    return fig
