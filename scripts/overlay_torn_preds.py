"""Montage of predicted meniscus overlaid on the preprocessed PD volume, for visual QA.

    python scripts/overlay_torn_preds.py --pd raw_pd.nii.gz --pred pred.nii.gz --out out.png [--pred2 other.nii.gz]

Uses preprocess.process_volume so the image grid matches the prediction grid
(n_slices, 384, 384). Only slices with any predicted meniscus (in either model) are shown.
Labels: 1 = lateral (cyan), 2 = medial (yellow).
"""
import argparse, os, sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "preprocess"))   # local layout; BigRed has preprocess.py at root
from preprocess import process_volume


def load_pred(path, shape):
    p = np.asarray(nib.load(path).dataobj)
    # predictions are saved as (slices, H, W) -- same array order as process_volume
    assert p.shape == shape, f"pred {p.shape} != vol {shape}"
    return p


def paint(ax, img, pred, title):
    ax.imshow(img, cmap="gray"); ax.axis("off")
    rgba = np.zeros(pred.shape + (4,))
    rgba[pred == 1] = (0, 1, 1, 0.65)
    rgba[pred == 2] = (1, 1, 0, 0.65)
    ax.imshow(rgba); ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--pred2"); ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs=2, default=["model A", "model B"])
    ap.add_argument("--max_slices", type=int, default=8)
    a = ap.parse_args()

    vol = process_volume(a.pd, "PD")
    preds = [load_pred(a.pred, vol.shape)] + ([load_pred(a.pred2, vol.shape)] if a.pred2 else [])
    has = np.zeros(vol.shape[0], bool)
    for p in preds:
        has |= (p > 0).any(axis=(1, 2))
    idx = np.where(has)[0]
    if len(idx) > a.max_slices:
        idx = idx[np.linspace(0, len(idx) - 1, a.max_slices).astype(int)]
    print(f"{has.sum()} / {vol.shape[0]} slices with meniscus predicted; showing {len(idx)}")
    if len(idx) == 0:
        print("No meniscus predicted anywhere -- nothing to plot"); return

    fig, axes = plt.subplots(len(preds), len(idx), figsize=(3.2 * len(idx), 3.4 * len(preds)), squeeze=False)
    for r, p in enumerate(preds):
        for c, s in enumerate(idx):
            paint(axes[r, c], vol[s], p[s], f"{a.names[r]} s{s}")
    fig.tight_layout(); fig.savefig(a.out, dpi=150); print("saved", a.out)


if __name__ == "__main__":
    main()
