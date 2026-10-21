"""Explainer figure: how the v15 label changes for each synthetic tear pattern.

One row per pattern:
  original slice + label | synthetic image (what the model sees) | label change, colour-coded:
      cyan  = meniscus in both (kept)
      red   = was meniscus, now background (removed / moved away)
      green = new meniscus (added / the moved piece at its new place)
  The title says whether the new label is one whole piece or several pieces.

    venv/bin/python scripts/explain_tear_labels.py --out figures/tear_labels_explained.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "segmentation"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tear_patterns import synth_pattern, MODES                      # noqa: E402
from tear_synth import _horns                                        # noqa: E402
from torn_compare_utils import process_volume, process_manual_seg    # noqa: E402

NIFTI = "data/iu-dataset/pd-files/AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz"
SEG = "/Users/anshikabajpai/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final/AC0D5A4D78B628_SAG_PD_TSE_6_1_1.seg.nrrd"
MEANING = {
    "inplace":  "crack painted inside; tissue not moved",
    "ragged":   "frayed, irregular edge",
    "lobe":     "rounded torn lobe attached",
    "truncate": "small radial tear: tip cut off",
    "absent":   "large radial / root tear: horn missing here",
    "flap":     "flap: piece split off and moved sideways",
    "bucket":   "bucket-handle: inner part moved toward centre",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures/tear_labels_explained.png"))
    ap.add_argument("--slice", type=int, default=11)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--pad", type=int, default=12)
    a = ap.parse_args()

    vol = process_volume(os.path.join(ROOT, NIFTI), "PD")
    seg = process_manual_seg(SEG).astype(np.int64)
    s = a.slice
    if len(_horns(seg[s] > 0)) < 2:
        s = next(k for k in range(1, vol.shape[0] - 1) if len(_horns(seg[k] > 0)) >= 2)
    img = vol[s - 1:s + 2].astype(np.float32)
    rng = np.random.default_rng(a.seed)

    fig, axes = plt.subplots(len(MODES), 3, figsize=(9.5, 3.3 * len(MODES)))
    for r, mode in enumerate(MODES):
        out, new = img, seg[s]
        for _ in range(40):
            out, new, got = synth_pattern(img, seg[s], rng, p=1.0, modes=(mode,), mode_p=(1.0,))
            if got != "none" and (mode == "inplace" or not np.array_equal(new, seg[s])):
                break
        old_m, new_m = seg[s] > 0, new > 0
        both, removed, added = old_m & new_m, old_m & ~new_m, new_m & ~old_m
        ys, xs = np.where(old_m | new_m)
        cs = (slice(max(ys.min() - a.pad, 0), ys.max() + a.pad), slice(max(xs.min() - a.pad, 0), xs.max() + a.pad))
        n_old, n_new = ndi.label(old_m)[1], ndi.label(new_m)[1]

        ax = axes[r, 0]
        ax.imshow(img[1][cs], cmap="gray", vmin=0, vmax=1)
        ax.contour(old_m[cs].astype(float), [0.5], colors=["#ffe600"], linewidths=1.3)
        ax.set_title(f"{mode}: original\n(yellow = healthy label, {n_old} pieces)", fontsize=9)
        ax = axes[r, 1]
        ax.imshow(out[1][cs], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"what the model sees\n{MEANING[mode]}", fontsize=9)
        ax = axes[r, 2]
        ax.imshow(out[1][cs], cmap="gray", vmin=0, vmax=1)
        ov = np.zeros(img[1][cs].shape + (4,))
        ov[both[cs]] = (0.0, 0.9, 1.0, 0.65)
        ov[removed[cs]] = (1.0, 0.15, 0.15, 0.8)
        ov[added[cs]] = (0.2, 1.0, 0.2, 0.85)
        ax.imshow(ov)
        if mode == "inplace":
            verdict = "WHOLE WEDGE (unchanged)"
        elif not new_m.any() or n_new < n_old:
            verdict = f"{n_new} piece(s): tissue REMOVED"
        elif n_new > n_old:
            verdict = f"{n_new} pieces: horn BROKEN into pieces"
        else:
            verdict = f"{n_new} pieces: same count, outline changed"
        ax.set_title(f"NEW label the model learns\n{verdict}", fontsize=9, fontweight="bold")
        for c in range(3):
            axes[r, c].axis("off")
    fig.legend(handles=[Patch(color=(0, 0.9, 1), label="meniscus, kept"),
                        Patch(color=(1, 0.15, 0.15), label="was meniscus -> now background (removed / moved away)"),
                        Patch(color=(0.2, 1, 0.2), label="new meniscus (added / moved piece at its new place)")],
               loc="lower center", ncol=1, fontsize=9, frameon=False)
    fig.suptitle(f"How the label changes per synthetic tear pattern (healthy slice {s}, rule: all meniscus tissue where it is)",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.99))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
