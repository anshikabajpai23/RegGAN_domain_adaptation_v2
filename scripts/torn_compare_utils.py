"""Helpers for notebooks/torn_patient_comparison.ipynb.

Everything is binary ("meniscus" vs "no meniscus"): model predictions are collapsed with (pred > 0)
and the hand segmentation (Slicer .seg.nrrd, single label) with (seg > 0).

Grid: all arrays are (n_slices, 384, 384) in the same order as preprocess.process_volume, i.e.
axis 0 = R (through-plane), axis 1 = A, axis 2 = S. The hand segmentation is pushed through the same
RAS reorient -> isotropic in-plane resample -> resize-to-384 steps as the image, with order=0.
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "preprocess"))   # local layout; BigRed has preprocess.py at the root
from preprocess import process_volume            # noqa: E402

logging.getLogger().setLevel(logging.WARNING)    # process_volume logs a lot at INFO

TARGET = 384
COLORS = {"replica": "#ff2a2a", "crop_seq": "#00e5ff", "manual": "#ffe600"}   # red / cyan / yellow
LABELS = {"replica": "Replica", "crop_seq": "Crop+Seq", "manual": "Your seg"}


def process_manual_seg(seg_path):
    """Hand segmentation (.seg.nrrd) -> bool array (n, 384, 384), same grid as process_volume."""
    img = sitk.DICOMOrient(sitk.ReadImage(str(seg_path)), "RAS")
    arr = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0)).astype(np.uint8)   # (R, A, S)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D seg, got {arr.shape}")
    sp_A, sp_S = img.GetSpacing()[1], img.GetSpacing()[2]
    ip = min(sp_A, sp_S)
    fa, fs = sp_A / ip, sp_S / ip
    if abs(fa - 1) > 0.02 or abs(fs - 1) > 0.02:
        arr = zoom(arr, (1.0, fa, fs), order=0, prefilter=False)
    n_R, n_A, n_S = arr.shape
    if n_A != TARGET or n_S != TARGET:
        arr = zoom(arr, (1.0, TARGET / n_A, TARGET / n_S), order=0, prefilter=False)
    return arr[:, :TARGET, :TARGET] > 0


def load_pred(path):
    return np.asarray(nib.load(str(path)).dataobj) > 0


def find_one(root, pattern):
    hits = sorted(Path(root).rglob(pattern))
    return hits[0] if hits else None


def load_patient(pid, replica_dir, cropseq_dir, seg_root, raw_dirs):
    """Returns dict(vol, replica, crop_seq, manual, stem) or None (with a printed reason)."""
    rep = find_one(replica_dir, f"{pid}*_meniscus_pred.nii.gz")
    crp = find_one(cropseq_dir, f"{pid}*_meniscus_pred.nii.gz")
    if rep is None or crp is None:
        print(f"[{pid}] SKIP: missing prediction (replica={rep is not None}, crop_seq={crp is not None})")
        return None
    stem = rep.name.replace("_meniscus_pred.nii.gz", "")
    raw = None
    for d in raw_dirs:
        raw = find_one(d, f"{stem}.nii.gz")
        if raw is not None:
            break
    if raw is None:
        print(f"[{pid}] SKIP: raw scan {stem}.nii.gz not found under {[str(d) for d in raw_dirs]}")
        return None
    seg = find_one(Path(seg_root) / pid, "*.seg.nrrd") if (Path(seg_root) / pid).exists() else None
    if seg is None:
        print(f"[{pid}] SKIP: no .seg.nrrd under {Path(seg_root) / pid}")
        return None

    vol = process_volume(str(raw), "PD")
    d = {"stem": stem, "vol": vol, "replica": load_pred(rep), "crop_seq": load_pred(crp),
         "manual": process_manual_seg(seg)}
    for k in ("replica", "crop_seq", "manual"):
        assert d[k].shape == vol.shape, f"[{pid}] {k} {d[k].shape} != image {vol.shape}"
    return d


def dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else 2.0 * (a & b).sum() / s


def slices_with_any(d):
    return np.where(d["replica"].any((1, 2)) | d["crop_seq"].any((1, 2)) | d["manual"].any((1, 2)))[0]


def summary_table(d):
    import pandas as pd
    rows = []
    for s in slices_with_any(d):
        rows.append({"slice": int(s),
                     "replica_px": int(d["replica"][s].sum()), "crop_seq_px": int(d["crop_seq"][s].sum()),
                     "manual_px": int(d["manual"][s].sum()),
                     "dice_replica": dice(d["replica"][s], d["manual"][s]),
                     "dice_crop_seq": dice(d["crop_seq"][s], d["manual"][s])})
    df = pd.DataFrame(rows).set_index("slice")
    tot = {"replica_px": int(d["replica"].sum()), "crop_seq_px": int(d["crop_seq"].sum()),
           "manual_px": int(d["manual"].sum()),
           "dice_replica": dice(d["replica"], d["manual"]), "dice_crop_seq": dice(d["crop_seq"], d["manual"])}
    df.loc["ALL (3D)"] = tot
    return df.round(3)


def _outline(ax, mask, key, lw=1.6):
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[COLORS[key]], linewidths=lw)


def crop_box(d, pad=30):
    """One SQUARE box per patient from the union of both MODEL masks over every slice, padded, clipped.
    The hand seg is deliberately not used: a stray Slicer stroke would blow the box up to the whole image."""
    m = d["replica"] | d["crop_seq"]
    a, s = np.where(m.any(0))
    if len(a) == 0:
        return (0, TARGET, 0, TARGET)
    a0, a1, s0, s1 = a.min() - pad, a.max() + pad + 1, s.min() - pad, s.max() + pad + 1
    side = min(max(a1 - a0, s1 - s0), TARGET)
    ca, cs = (a0 + a1) // 2, (s0 + s1) // 2
    a0 = int(np.clip(ca - side // 2, 0, TARGET - side)); s0 = int(np.clip(cs - side // 2, 0, TARGET - side))
    return (a0, a0 + side, s0, s0 + side)


def plot_patient(pid, d, zoom_in=False, pad=30, panel=3.0):
    """Rows = every slice with any mask; cols = original | replica | crop+seq | overlay | your seg.
    Outlines only (no fill)."""
    sl = slices_with_any(d)
    a0, a1, s0, s1 = crop_box(d, pad) if zoom_in else (0, TARGET, 0, TARGET)
    cols = ["Original", LABELS["replica"], LABELS["crop_seq"], "Overlay", LABELS["manual"]]
    fig, axes = plt.subplots(len(sl), 5, figsize=(panel * 5, panel * len(sl)), squeeze=False)
    for r, s in enumerate(sl):
        img = d["vol"][s, a0:a1, s0:s1]
        m = {k: d[k][s, a0:a1, s0:s1] for k in ("replica", "crop_seq", "manual")}
        for c in range(5):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 1: _outline(ax, m["replica"], "replica")
            if c == 2: _outline(ax, m["crop_seq"], "crop_seq")
            if c == 4: _outline(ax, m["manual"], "manual")
            if c == 3:
                for k in ("replica", "crop_seq", "manual"):
                    _outline(ax, m[k], k, lw=1.3)
            if r == 0:
                ax.set_title(cols[c], fontsize=12)
        axes[r, 0].set_ylabel(f"slice {s}", fontsize=11)
    outside = int(d["manual"].sum() - d["manual"][:, a0:a1, s0:s1].sum()) if zoom_in else 0
    note = f"   [{outside} px of your seg fall outside this zoom]" if outside else ""
    fig.suptitle(f"{pid}  ({'zoomed' if zoom_in else 'full'}){note}   outlines: "
                 + "  ".join(f"{LABELS[k]}={COLORS[k]}" for k in COLORS), fontsize=12, y=1.0)
    fig.tight_layout()
    return fig
