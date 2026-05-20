"""
preprocess.py
=============
Preprocessing pipeline for DESS (SKM-TEA) -> PD (IU) domain adaptation.

Both DESS and PD are sagittal (P,S,R orientation):
  DESS: 512x512x160  spacing 0.31x0.31x0.80 mm
  PD:   384x384x36   spacing 0.39x0.39x3.60 mm

Both are sliced along axis 0 (R = through-plane for sagittal) after RAS reorientation.
Each slice is (A x S) = in-plane anatomy.

Fix for PD shrink: resample in-plane dims to isotropic before resizing to 384x384.
DO NOT resample through-plane (keeps all sagittal slices).
"""

import os
import json
import glob
import logging
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TARGET_SIZE   = (384, 384)
INTENSITY_PCT = (1, 99)
SLICE_AXIS    = 0    # sagittal: through-plane = R axis = axis 0 after RAS reorientation


def load_nifti(path):
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32), img.header, img.affine


def get_voxel_spacing(header):
    return tuple(float(z) for z in header.get_zooms()[:3])


def reorient_to_ras(nifti_path):
    """Reorient to RAS+. Returns (array, spacing) where array is (R,A,S)."""
    img = sitk.ReadImage(nifti_path)
    img = sitk.DICOMOrient(img, "RAS")
    arr = sitk.GetArrayFromImage(img)   # (S, A, R)
    arr = np.transpose(arr, (2, 1, 0)) # -> (R, A, S)
    sp  = img.GetSpacing()             # (sp_R, sp_A, sp_S)
    return arr.astype(np.float32), sp


def normalise(data):
    lo = np.percentile(data, INTENSITY_PCT[0])
    hi = np.percentile(data, INTENSITY_PCT[1])
    data = np.clip(data, lo, hi)
    return ((data - lo) / (hi - lo + 1e-8)).astype(np.float32)


def process_volume(nifti_path, modality):
    """
    Returns float32 array shape (n_sag_slices, 384, 384) normalised [0,1].

    Pipeline:
      1. Reorient to RAS  -> (R, A, S)
      2. Resample A and S dims to isotropic (finer of the two spacings)
         -- fixes PD shrink: 0.39mm A + 3.6mm S -> both at 0.39mm
      3. Resize (A, S) to 384x384
      4. Normalise
    """
    log.info(f"Processing [{modality}]: {nifti_path}")

    data, sp = reorient_to_ras(nifti_path)
    log.info(f"  After RAS: shape={data.shape}  spacing={tuple(round(s,3) for s in sp)}")

    # sp = (sp_R, sp_A, sp_S)
    sp_A, sp_S = sp[1], sp[2]
    target_ip  = min(sp_A, sp_S)   # resample to finer in-plane spacing
    fa = sp_A / target_ip
    fs = sp_S / target_ip

    if abs(fa - 1.0) > 0.02 or abs(fs - 1.0) > 0.02:
        log.info(f"  In-plane resample: A x{fa:.3f}  S x{fs:.3f}")
        data = zoom(data, (1.0, fa, fs), order=3, prefilter=True)
        log.info(f"  After resample: shape={data.shape}")

    # Resize (A, S) to TARGET_SIZE — always force exact size
    n_R, n_A, n_S = data.shape
    th, tw = TARGET_SIZE
    if n_A != th or n_S != tw:
        log.info(f"  Resize in-plane {n_A}x{n_S} -> {th}x{tw}")
        data = zoom(data, (1.0, th/n_A, tw/n_S), order=3)
    # hard clamp in case of rounding from zoom
    data = data[:, :th, :tw]
    if data.shape[1] < th or data.shape[2] < tw:
        pad = np.zeros((data.shape[0], th, tw), dtype=np.float32)
        pad[:, :data.shape[1], :data.shape[2]] = data
        data = pad

    data = normalise(data)
    log.info(f"  [{modality}] final shape={data.shape} min={data.min():.3f} max={data.max():.3f}")
    return data   # (n_R, 384, 384)


def extract_slices(volume, out_dir, prefix, skip_empty_pct=0.02):
    """volume shape: (n_slices, 384, 384). Saves each as (384,384) .npy."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(volume.shape[0]):
        sl = volume[i]
        if sl.mean() < skip_empty_pct:
            continue
        fname = os.path.join(out_dir, f"{prefix}_sl{i:04d}.npy")
        np.save(fname, sl)
        paths.append(fname)
    log.info(f"  Saved {len(paths)}/{volume.shape[0]} slices -> {out_dir}")
    return paths


def find_niftis(root, extensions=("*.nii.gz", "*.nii")):
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(files)


def run_preprocessing(dess_root, pd_root, out_root, val_ratio=0.1, test_ratio=0.1, max_volumes=None):
    dess_files = find_niftis(dess_root)
    pd_files   = find_niftis(pd_root)
    log.info(f"Found {len(dess_files)} DESS and {len(pd_files)} PD volumes.")
    assert len(dess_files) > 0, f"No NIfTI files in {dess_root}"
    assert len(pd_files)   > 0, f"No NIfTI files in {pd_root}"

    if max_volumes is not None:
        dess_files = dess_files[:max_volumes]
        pd_files   = pd_files[:max_volumes]
        log.info(f"[SAMPLE MODE] Using {len(dess_files)} DESS and {len(pd_files)} PD volumes.")

    dess_slice_paths, pd_slice_paths = [], []

    for vpath in dess_files:
        pid      = Path(vpath).stem.replace(".nii", "")
        sdir     = os.path.join(out_root, "slices", "dess")
        existing = sorted(glob.glob(os.path.join(sdir, f"{pid}_sl*.npy")))
        if existing:
            log.info(f"  DESS [{pid}] already done ({len(existing)} slices) — skipping.")
            dess_slice_paths.extend(existing)
        else:
            dess_slice_paths.extend(extract_slices(process_volume(vpath, "DESS"), sdir, pid))

    for vpath in pd_files:
        pid      = Path(vpath).stem.replace(".nii", "")
        sdir     = os.path.join(out_root, "slices", "pd")
        existing = sorted(glob.glob(os.path.join(sdir, f"{pid}_sl*.npy")))
        if existing:
            log.info(f"  PD [{pid}] already done ({len(existing)} slices) — skipping.")
            pd_slice_paths.extend(existing)
        else:
            pd_slice_paths.extend(extract_slices(process_volume(vpath, "PD"), sdir, pid))

    def split(paths, val_r, test_r):
        tr, tmp = train_test_split(paths, test_size=val_r+test_r, random_state=42)
        va, te  = train_test_split(tmp,   test_size=test_r/(val_r+test_r), random_state=42)
        return tr, va, te

    d_tr, d_val, d_te = split(dess_slice_paths, val_ratio, test_ratio)
    p_tr, p_val, p_te = split(pd_slice_paths,   val_ratio, test_ratio)

    splits = {"dess": {"train": d_tr, "val": d_val, "test": d_te},
              "pd":   {"train": p_tr, "val": p_val, "test": p_te}}

    splits_path = os.path.join(out_root, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    log.info(f"DESS  train={len(d_tr)}  val={len(d_val)}  test={len(d_te)}")
    log.info(f"PD    train={len(p_tr)}  val={len(p_val)}  test={len(p_te)}")
    log.info(f"Splits -> {splits_path}")
    return splits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dess_root",  default="data/skm-tea-dataset/dess-files")
    ap.add_argument("--pd_root",    default="data/iu-dataset/pd-files")
    ap.add_argument("--out_root",   default="data/preprocessed")
    ap.add_argument("--val_ratio",  type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--max_volumes", type=int, default=None,
                    help="Limit number of volumes per modality (for quick sanity check)")
    args = ap.parse_args()
    run_preprocessing(args.dess_root, args.pd_root, args.out_root,
                      args.val_ratio, args.test_ratio, args.max_volumes)