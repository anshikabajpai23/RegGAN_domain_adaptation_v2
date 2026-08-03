"""
prepare_real_pd_labeled.py
==========================
Preprocesses labeled real PD images + .seg.nrrd masks into .npy slices.
Applies same spatial pipeline as preprocess.py (RAS reorient, isotropic
in-plane resample, resize 384x384). Mask uses order=0.

Output:
    real_pd_seg_data/
        train/images/  {pid}_{slice:04d}.npy  float32 [0,1]
        train/masks/   {pid}_{slice:04d}.npy  int8 {0,1}
        val/images/
        val/masks/

Usage:
    python segmentation/prepare_real_pd_labeled.py \
        --img_dir   /N/.../labelled-pd \
        --mask_dir  /N/.../labelled-pd-segmentations \
        --out_dir   /N/.../real_pd_seg_data \
        --val_patients AC2B0AA9AE767D AC2E254F52E467
"""
import argparse
import logging
import os
import re

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_nrrd_binary(path):
    """Read .seg.nrrd, return SimpleITK binary image (0/1)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)

    if arr.ndim == 4:
        # Slicer multi-segment: (segments, Z, Y, X) → merge all
        arr = (arr.max(axis=0) > 0).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        # Copy metadata from first segment
        try:
            comp = sitk.VectorIndexSelectionCast(img, 0)
            out.CopyInformation(comp)
        except Exception:
            out.SetSpacing(img.GetSpacing()[:3])
            out.SetOrigin(img.GetOrigin()[:3])
            out.SetDirection(img.GetDirection()[:9])
    else:
        arr = (arr > 0).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)

    return out


def process_image(path):
    """Same as preprocess.py process_volume("PD") but returns (vol, spacing)."""
    try:
        img = sitk.ReadImage(path)
        img = sitk.DICOMOrient(img, "RAS")
    except Exception:
        import nibabel as nib
        nib_img = nib.load(path)
        nib_img = nib.as_closest_canonical(nib_img)
        arr = nib_img.get_fdata().astype(np.float32)
        sp = nib_img.header.get_zooms()[:3]
        return _process_array(arr, sp, order=3, normalise=True)

    sp  = img.GetSpacing()
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    # sitk array is (S, A, R) → transpose to (R, A, S)
    arr = arr.transpose(2, 1, 0)
    return _process_array(arr, (float(sp[0]), float(sp[1]), float(sp[2])),
                          order=3, normalise=True)


def process_mask(sitk_img):
    """Apply same spatial transforms as process_image but order=0, no normalise."""
    try:
        sitk_img = sitk.DICOMOrient(sitk_img, "RAS")
    except Exception:
        pass

    sp  = sitk_img.GetSpacing()
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    arr = arr.transpose(2, 1, 0)
    return _process_array(arr, (float(sp[0]), float(sp[1]), float(sp[2])),
                          order=0, normalise=False)


def _process_array(arr, spacing, order, normalise):
    """Shared spatial pipeline: isotropic in-plane resample → resize 384×384."""
    sp_R, sp_A, sp_S = spacing

    # Step 1: resample A,S to isotropic
    target_ip = min(sp_A, sp_S)
    fa = sp_A / target_ip
    fs = sp_S / target_ip
    if abs(fa - 1.0) > 0.02 or abs(fs - 1.0) > 0.02:
        arr = zoom(arr, (1.0, fa, fs), order=order, prefilter=(order > 0))

    # Step 2: resize to 384×384
    _, n_A, n_S = arr.shape
    if n_A != 384 or n_S != 384:
        arr = zoom(arr, (1.0, 384 / n_A, 384 / n_S), order=order,
                   prefilter=(order > 0))

    # Step 3: normalise
    if normalise:
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-8)

    return arr.astype(np.float32)


def get_patient_id(fname):
    m = re.match(r"(AC[A-F0-9]+)", fname, re.IGNORECASE)
    return m.group(1) if m else None


def find_image(img_dir, pid):
    for f in os.listdir(img_dir):
        if f.startswith(pid) and (f.endswith(".nii.gz") or f.endswith(".nii")):
            return os.path.join(img_dir, f)
    return None


def find_mask(mask_dir, pid):
    for f in os.listdir(mask_dir):
        if f.startswith(pid) and f.endswith(".nrrd"):
            return os.path.join(mask_dir, f)
    return None


def save_slices(vol, mask_vol, pid, split_dir, skip_bg=True):
    img_dir  = os.path.join(split_dir, "images")
    mask_dir = os.path.join(split_dir, "masks")
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    n_slices = vol.shape[0]
    saved = 0
    for s in range(n_slices):
        img_sl  = vol[s]
        mask_sl = mask_vol[s]
        if skip_bg and img_sl.mean() < 0.02:
            continue
        fname = f"{pid}_{s:04d}.npy"
        np.save(os.path.join(img_dir,  fname), img_sl.astype(np.float32))
        np.save(os.path.join(mask_dir, fname), mask_sl.astype(np.int8))
        saved += 1

    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir",       required=True)
    ap.add_argument("--mask_dir",      required=True)
    ap.add_argument("--out_dir",       required=True)
    ap.add_argument("--val_patients",  nargs="+",
                    default=["AC2B0AA9AE767D", "AC2E254F52E467"])
    args = ap.parse_args()

    val_set = set(args.val_patients)
    train_dir = os.path.join(args.out_dir, "train")
    val_dir   = os.path.join(args.out_dir, "val")

    mask_files = [f for f in os.listdir(args.mask_dir) if f.endswith(".nrrd")]
    log.info(f"Found {len(mask_files)} mask files in {args.mask_dir}")

    total_train, total_val = 0, 0

    for mf in sorted(mask_files):
        pid = get_patient_id(mf)
        if not pid:
            log.warning(f"Cannot extract patient ID from {mf}, skipping")
            continue

        img_path  = find_image(args.img_dir, pid)
        mask_path = os.path.join(args.mask_dir, mf)

        if not img_path:
            log.warning(f"No image found for {pid}, skipping")
            continue

        log.info(f"Processing {pid} ...")

        try:
            vol  = process_image(img_path)
            mask_sitk = load_nrrd_binary(mask_path)
            mask_vol  = process_mask(mask_sitk)
        except Exception as e:
            log.error(f"  FAILED: {e}")
            continue

        mask_vol = (mask_vol > 0.5).astype(np.int8)

        if vol.shape != mask_vol.shape:
            log.warning(f"  Shape mismatch: image={vol.shape} mask={mask_vol.shape} — skipping")
            continue

        men_slices = int((mask_vol > 0).any(axis=(1, 2)).sum())
        log.info(f"  vol={vol.shape}  meniscus slices={men_slices}")

        split  = "val" if pid in val_set else "train"
        outdir = val_dir if pid in val_set else train_dir
        n = save_slices(vol, mask_vol, pid, outdir)

        log.info(f"  → {split}: saved {n} slices")
        if split == "train":
            total_train += n
        else:
            total_val += n

    log.info(f"\nDone. Train slices: {total_train}  Val slices: {total_val}")
    log.info(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()
