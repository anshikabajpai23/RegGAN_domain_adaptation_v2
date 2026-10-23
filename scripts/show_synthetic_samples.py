"""Picture grids of the v15 synthetic tears that training actually saw (from synthetic_log samples).

One PNG per pattern: before | after (what the model saw) | label before | label after, zoomed on the
meniscus. Samples are picked at random (fixed seed), or the most fragmented first (--most_fragmented),
using stats_all.csv to rank by the change in number of pieces.

    venv/bin/python scripts/show_synthetic_samples.py --log_dir results/v15_synthetic_log --n 10
    venv/bin/python scripts/show_synthetic_samples.py --log_dir results/v15_synthetic_log --modes flap ragged --most_fragmented
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "segmentation"))
from synthetic_log_utils import load_sample   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="results/v15_synthetic_log")
    ap.add_argument("--out_dir", default="figures/v15_synthetic_grids")
    ap.add_argument("--modes", nargs="*", default=None, help="default: every pattern folder present")
    ap.add_argument("--source", default="both", choices=["fake", "real", "both"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--most_fragmented", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pad", type=int, default=18)
    a = ap.parse_args()

    modes = a.modes or sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.log_dir, "samples", "*")))
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for mode in modes:
        files = sorted(glob.glob(os.path.join(a.log_dir, "samples", mode, "*.npz")))
        if a.source != "both":
            files = [f for f in files if os.path.basename(f).startswith(a.source + "_")]
        good = []
        for f in files:                      # skip files cut off by an interrupted download
            try:
                with np.load(f) as d:
                    d["mask_after"]
                good.append(f)
            except Exception:
                print(f"  skipping unreadable {f}")
        if not good:
            print(f"{mode}: no samples"); continue
        if a.most_fragmented:
            def frag(f):
                with np.load(f) as d:
                    return ndi.label(d["mask_after"] > 0)[1] - ndi.label(d["mask_before"] > 0)[1]
            pick = sorted(good, key=frag, reverse=True)[:a.n]
        else:
            pick = [good[i] for i in rng.choice(len(good), size=min(a.n, len(good)), replace=False)]

        has_tear = "tear" in load_sample(pick[0])     # v17+ logs store the tear map
        ncol = 5 if has_tear else 4
        fig, axes = plt.subplots(len(pick), ncol, figsize=(2.5 * ncol, 2.6 * len(pick)), squeeze=False)
        for r, f in enumerate(pick):
            d = load_sample(f)
            ib, ia = d["image_before"][1].astype(np.float32), d["image_after"][1].astype(np.float32)
            mb, ma = d["mask_before"] > 0, d["mask_after"] > 0
            src, pid, s = str(d["source"]), str(d["pid"]), int(d["slice"])
            tear = d["tear"] > 0 if has_tear else None
            var = str(d["variant_all"]) if "variant_all" in d else mode
            ys, xs = np.where(mb | ma)
            cs = (slice(max(ys.min() - a.pad, 0), ys.max() + a.pad), slice(max(xs.min() - a.pad, 0), xs.max() + a.pad))
            nb, na = ndi.label(mb)[1], ndi.label(ma)[1]
            for c, (im, t) in enumerate([(ib, f"{src} {pid} s{s}: before"), (ia, "after (model saw this)"),
                                         (ib, f"label before ({nb} pieces)"), (ia, f"label after ({na} pieces)")]):
                ax = axes[r, c]; ax.imshow(im[cs], cmap="gray", vmin=0, vmax=1); ax.axis("off"); ax.set_title(t, fontsize=8)
            axes[r, 2].contour(mb[cs].astype(float), [0.5], colors=["#ffe600"], linewidths=1.1)
            axes[r, 3].contour(ma[cs].astype(float), [0.5], colors=["#00e5ff"], linewidths=1.1)
            if has_tear:
                ax = axes[r, 4]; ax.imshow(ia[cs], cmap="gray", vmin=0, vmax=1); ax.axis("off")
                ax.set_title(f"tear map ({var})", fontsize=7)
                if tear.any():
                    ax.contour(tear[cs].astype(float), [0.5], colors=["#ff2a2a"], linewidths=1.0)
        tag = "most fragmented" if a.most_fragmented else "random"
        fig.suptitle(f"synthetic pattern: {mode} [{os.path.basename(os.path.dirname(os.path.normpath(a.log_dir))) or a.log_dir}] ({tag}, {len(pick)} of {len(good)} saved)", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / len(pick)))
        out = os.path.join(a.out_dir, f"{mode}{'_most_fragmented' if a.most_fragmented else ''}.png")
        fig.savefig(out, dpi=100); plt.close(fig)
        print(f"{mode}: {out}")


if __name__ == "__main__":
    main()
