"""Explainer: the two possible label rules for torn menisci, on the same synthetic tears.

Rows: an in-place crack, a cut-off tip, a displaced (bucket-handle) piece.
Columns: what the model sees | label under rule (a) "tissue where it actually is" (what v15 uses)
         | label under rule (b) "original wedge" | what differs.

    venv/bin/python scripts/explain_label_rules.py --out figures/tear_label_rules.png
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
from tear_patterns import synth_pattern                              # noqa: E402
from torn_compare_utils import process_volume, process_manual_seg    # noqa: E402

NIFTI = "data/iu-dataset/pd-files/AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz"
SEG = "/Users/anshikabajpai/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final/AC0D5A4D78B628_SAG_PD_TSE_6_1_1.seg.nrrd"
CASES = [
    ("inplace", "Crack inside, nothing moved",
     "SAME under both rules:\nthe whole wedge, crack included"),
    ("truncate", "Tip of the horn cut off (radial tear)",
     "(a) leaves the tip out (it is gone)\n(b) still draws the missing tip"),
    ("bucket", "Inner piece broke off and moved\n(bucket-handle)",
     "(a) outlines the moved piece where it is,\n     NOT the empty spot it left\n"
     "(b) outlines the empty spot as if intact,\n     and ignores the moved piece"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures/tear_label_rules.png"))
    ap.add_argument("--slice", type=int, default=11)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--pad", type=int, default=12)
    a = ap.parse_args()
    vol = process_volume(os.path.join(ROOT, NIFTI), "PD")
    seg = process_manual_seg(SEG).astype(np.int64)
    s = a.slice
    img = vol[s - 1:s + 2].astype(np.float32)
    rng = np.random.default_rng(a.seed)

    fig, axes = plt.subplots(len(CASES), 4, figsize=(13, 3.6 * len(CASES)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for r, (mode, what, diff) in enumerate(CASES):
        out, new = img, seg[s]
        for _ in range(40):
            out, new, got = synth_pattern(img, seg[s], rng, p=1.0, modes=(mode,), mode_p=(1.0,))
            if mode == "inplace" or not np.array_equal(new, seg[s]):
                break
        rule_a, rule_b = new > 0, seg[s] > 0          # (b) = the original healthy wedge, ignoring what moved
        ys, xs = np.where(rule_a | rule_b)
        cs = (slice(max(ys.min() - a.pad, 0), ys.max() + a.pad), slice(max(xs.min() - a.pad, 0), xs.max() + a.pad))
        base = out[1][cs]
        for c, (m, col, title) in enumerate([(None, None, f"What the model sees\n{what}"),
                                             (rule_a, (0, 0.9, 1), "Rule (a): tissue where it is\n(what v15 trains on)"),
                                             (rule_b, (1, 0.6, 0), "Rule (b): the original wedge\n(not used)")]):
            ax = axes[r, c]
            ax.imshow(base, cmap="gray", vmin=0, vmax=1); ax.axis("off")
            ax.set_title(title, fontsize=10, fontweight="bold" if c == 1 else None)
            if m is not None:
                ov = np.zeros(base.shape + (4,)); ov[m[cs]] = (*col, 0.6); ax.imshow(ov)
                ax.contour(m[cs].astype(float), [0.5], colors=[col], linewidths=1.2)
        axes[r, 3].axis("off")
        axes[r, 3].text(0.02, 0.5, diff, va="center", fontsize=10, family="monospace")
    fig.suptitle("Same tears, two possible label rules (cyan = rule a, used by v15; orange = rule b)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
