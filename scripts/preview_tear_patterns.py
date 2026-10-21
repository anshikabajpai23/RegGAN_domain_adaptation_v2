"""Preview every v15 synthetic tear mode (segmentation/tear_patterns.py) on a healthy scan.

One row per mode, several examples per mode:
  original | synthetic image | original label (yellow) | NEW label (cyan) the model trains on
Show this to the radiologist before training: the label rule is "all meniscus tissue where it
actually is" (removed tissue -> background, displaced tissue -> meniscus at its new place).

    venv/bin/python scripts/preview_tear_patterns.py --out figures/tear_patterns_preview.png
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
from tear_patterns import synth_pattern, MODES                      # noqa: E402
from tear_synth import _horns                                        # noqa: E402
from torn_compare_utils import process_volume, process_manual_seg    # noqa: E402

NIFTI = "data/iu-dataset/pd-files/AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz"
SEG = "/Users/anshikabajpai/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final/AC0D5A4D78B628_SAG_PD_TSE_6_1_1.seg.nrrd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti", default=os.path.join(ROOT, NIFTI))
    ap.add_argument("--seg", default=SEG)
    ap.add_argument("--out", default=os.path.join(ROOT, "figures/tear_patterns_preview.png"))
    ap.add_argument("--per_mode", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad", type=int, default=22)
    a = ap.parse_args()

    vol = process_volume(a.nifti, "PD")
    seg = process_manual_seg(a.seg).astype(np.int64)
    # prefer slices that show two horns (bow-tie), so tip/bucket modes have a direction
    two = [s for s in range(1, vol.shape[0] - 1) if len(_horns(seg[s] > 0)) >= 2]
    one = [s for s in range(1, vol.shape[0] - 1) if seg[s].sum() > 50]
    pool = two or one
    rng = np.random.default_rng(a.seed)

    rows = [(m, k) for m in MODES for k in range(a.per_mode)]
    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.9 * len(rows)), squeeze=False)
    for r, (mode, k) in enumerate(rows):
        s = pool[(r * 3 + k) % len(pool)]
        img = vol[s - 1:s + 2].astype(np.float32)
        out, new = img, seg[s]
        for _ in range(30):
            out, new, got = synth_pattern(img, seg[s], rng, p=1.0, modes=(mode,), mode_p=(1.0,))
            if got != "none" and (not np.array_equal(new, seg[s]) or mode == "inplace"):
                break
        m = (seg[s] > 0) | (new > 0)
        ys, xs = np.where(m)
        crop = (slice(max(ys.min() - a.pad, 0), ys.max() + a.pad), slice(max(xs.min() - a.pad, 0), xs.max() + a.pad))
        titles = [f"{mode} #{k+1} (slice {s}): original", "synthetic image", "original label", "NEW label (trained on)"]
        for c in range(4):
            ax = axes[r, c]
            ax.imshow((img[1] if c in (0, 2) else out[1])[crop], cmap="gray", vmin=0, vmax=1)
            ax.set_title(titles[c], fontsize=8); ax.axis("off")
        axes[r, 2].contour((seg[s] > 0)[crop].astype(float), [0.5], colors=["#ffe600"], linewidths=1.2)
        axes[r, 3].contour((new > 0)[crop].astype(float), [0.5], colors=["#00e5ff"], linewidths=1.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
