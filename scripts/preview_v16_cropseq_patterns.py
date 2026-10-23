"""Preview v16 (crop+seq) tear patterns: the same pattern in the stretched crop space, with and
without the native-aspect correction (tear_patterns_seq.py).

Columns: crop input (centre slice) | with correction (used by v16) | without correction |
label after (with correction). Uses a real PD scan cropped exactly like the crop data / inference.

    venv/bin/python scripts/preview_v16_cropseq_patterns.py --out figures/v16_cropseq_patterns_preview.png
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "segmentation"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tear_patterns import MODES                                       # noqa: E402
from tear_patterns_seq import synth_pattern_seq                       # noqa: E402
from torn_compare_utils import process_volume, process_manual_seg     # noqa: E402

NIFTI = "data/iu-dataset/pd-files/AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz"
SEG = "/Users/anshikabajpai/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final/AC0D5A4D78B628_SAG_PD_TSE_6_1_1.seg.nrrd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures/v16_cropseq_patterns_preview.png"))
    ap.add_argument("--box_json", default=os.path.join(ROOT, "crop_box.json"))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--pad", type=int, default=25)
    a = ap.parse_args()
    box = json.load(open(a.box_json))
    nat = (box["y1"] - box["y0"], box["x1"] - box["x0"])
    vol = process_volume(os.path.join(ROOT, NIFTI), "PD")
    seg = process_manual_seg(SEG).astype(np.int64)

    def cr(x, order):
        c = x[box["y0"]:box["y1"], box["x0"]:box["x1"]]
        return zoom(c, (384 / c.shape[0], 384 / c.shape[1]), order=order)

    pool = [s for s in range(2, vol.shape[0] - 2) if (seg[s] > 0).sum() > 300]
    fig, axes = plt.subplots(len(MODES), 4, figsize=(11, 2.7 * len(MODES)), squeeze=False)
    for r, mode in enumerate(MODES):
        s = pool[r % len(pool)]
        img = np.stack([np.clip(cr(vol[s + o], 3), 0, 1) for o in (-2, -1, 0, 1, 2)]).astype(np.float32)
        msk = cr(seg[s].astype(np.float32), 0).astype(np.int64)
        kw = dict(p=1.0, modes=(mode,), mode_p=(1.0,))
        fixed, new, _, _, _ = synth_pattern_seq(img, msk, np.random.default_rng(a.seed + r), native_hw=nat, **kw)
        raw, _, _, _, _ = synth_pattern_seq(img, msk, np.random.default_rng(a.seed + r), native_hw=None, **kw)
        ys, xs = np.where((msk > 0) | (new > 0))
        cs = (slice(max(ys.min() - a.pad, 0), ys.max() + a.pad), slice(max(xs.min() - a.pad, 0), xs.max() + a.pad))
        for c, (im, t) in enumerate([(img[2], f"{mode} (slice {s}): crop input"), (fixed[2], "with correction (v16)"),
                                     (raw[2], "without correction"), (fixed[2], "label after (v16)")]):
            ax = axes[r, c]; ax.imshow(im[cs], cmap="gray", vmin=0, vmax=1); ax.axis("off"); ax.set_title(t, fontsize=8)
        axes[r, 0].contour((msk[cs] > 0).astype(float), [0.5], colors=["#ffe600"], linewidths=1.0)
        axes[r, 3].contour((new[cs] > 0).astype(float), [0.5], colors=["#00e5ff"], linewidths=1.0)
    fig.suptitle("v16 crop+seq: v15 patterns in crop space (crop is stretched ~1.3x vertically, ~1.9x horizontally)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=100)
    print("saved", a.out)


if __name__ == "__main__":
    main()
