"""
preprocess_masks.py
===================
Preprocess segmentation masks to match preprocessed DESS volumes.

Same transforms as preprocess.py but with order=0 (nearest neighbour).
Saves with same affine as translated PD from infer.py.
Skips masks that fail (e.g. non-orthonormal direction cosines) and logs them.

Masks: MTR_XXX.nii.gz
DESS:  MTR_XXX_Anonymized_XXXXXXXX_e1.nii.gz
Matched by MTR_XXX prefix.

Usage:
    python preprocess_masks.py \
        --mask_root /N/project/prostate_cancer_ai/anshika/skm-tea-dataset/segmentation_masks \
        --dess_root /N/project/prostate_cancer_ai/anshika/regGAN/data/skm-tea-dataset/dess-files \
        --out_root  /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed/masks \
        --pd_dir    /N/project/prostate_cancer_ai/anshika/regGAN/results2/translated_pd
"""

import os
import glob
import argparse
import logging
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from scipy.ndimage import zoom

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TARGET_SIZE = (384, 384)


def reorient_to_ras_mask(nifti_path):
    """
    Reorient mask to RAS+.
    Tries SimpleITK first; falls back to nibabel for non-orthonormal cases.
    Returns (int16 array (R,A,S), spacing).
    """
    try:
        img = sitk.ReadImage(nifti_path)
        img = sitk.DICOMOrient(img, "RAS")
        arr = sitk.GetArrayFromImage(img)    # (S, A, R)
        arr = np.transpose(arr, (2, 1, 0))  # -> (R, A, S)
        sp  = img.GetSpacing()              # (sp_R, sp_A, sp_S)
        return arr.astype(np.int16), sp
    except Exception:
        # Fallback: nibabel handles non-orthonormal direction cosines
        img     = nib.load(nifti_path)
        ras_img = nib.as_closest_canonical(img)
        arr     = ras_img.get_fdata(dtype=np.float32).astype(np.int16)
        zooms   = ras_img.header.get_zooms()[:3]
        # nibabel RAS: (x=R, y=A, z=S) already (R,A,S) order
        sp = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
        log.info(f"  Used nibabel fallback for {nifti_path}")
        return arr, sp


def process_mask(mask_path):
    """
    Returns int16 array (n_R, 384, 384).
    Mirrors process_volume() from preprocess.py exactly,
    but uses order=0 interpolation everywhere.
    Raises exception if mask cannot be read (caller handles skip).
    """
    log.info(f"Processing: {mask_path}")
    mask, sp = reorient_to_ras_mask(mask_path)
    log.info(f"  After RAS: shape={mask.shape}  spacing={tuple(round(s,3) for s in sp)}")

    # in-plane resample to isotropic
    sp_A, sp_S = sp[1], sp[2]
    target_ip  = min(sp_A, sp_S)
    fa = sp_A / target_ip
    fs = sp_S / target_ip
    if abs(fa - 1.0) > 0.02 or abs(fs - 1.0) > 0.02:
        log.info(f"  Resample: A x{fa:.3f}  S x{fs:.3f}")
        mask = zoom(mask, (1.0, fa, fs), order=0, prefilter=False)

    # resize to 384x384
    n_R, n_A, n_S = mask.shape
    th, tw = TARGET_SIZE
    if n_A != th or n_S != tw:
        log.info(f"  Resize {n_A}x{n_S} -> {th}x{tw}")
        mask = zoom(mask, (1.0, th/n_A, tw/n_S), order=0, prefilter=False)

    # hard clamp
    mask = mask[:, :th, :tw]
    if mask.shape[1] < th or mask.shape[2] < tw:
        pad = np.zeros((mask.shape[0], th, tw), dtype=np.int16)
        pad[:, :mask.shape[1], :mask.shape[2]] = mask
        mask = pad

    log.info(f"  Final shape={mask.shape}  labels={np.unique(mask)}")
    return mask.astype(np.int16)


def get_pd_affine(pd_dir, stem):
    """Get affine directly from translated PD — guarantees alignment."""
    matches = glob.glob(os.path.join(pd_dir, f"{stem}*.nii.gz"))
    if not matches:
        return None, None
    pd_img = nib.load(matches[0])
    return pd_img.affine, pd_img.header.get_zooms()[:3]


def find_matching_dess(mask_path, dess_root):
    stem    = Path(mask_path).stem.replace(".nii", "")
    matches = glob.glob(os.path.join(dess_root, f"{stem}_*.nii.gz"))
    if not matches:
        matches = glob.glob(os.path.join(dess_root, f"**/{stem}_*.nii.gz"), recursive=True)
    return matches[0] if matches else None


def run(mask_root, dess_root, out_root, pd_dir=None):
    mask_files = sorted(glob.glob(os.path.join(mask_root, "*.nii.gz")))
    log.info(f"Found {len(mask_files)} masks.")
    os.makedirs(out_root, exist_ok=True)

    matched, skipped, errored = 0, 0, []

    for mpath in mask_files:
        stem     = Path(mpath).stem.replace(".nii", "")
        out_path = os.path.join(out_root, f"{stem}_mask.nii.gz")

        if os.path.exists(out_path):
            log.info(f"  {stem} already done — skipping.")
            matched += 1
            continue

        dess_path = find_matching_dess(mpath, dess_root)
        if dess_path is None:
            log.warning(f"  No matching DESS for {stem} — skipping.")
            skipped += 1
            continue

        log.info(f"  {stem} -> {Path(dess_path).name}")

        # ── Try processing — skip on any error ───────────────────────────
        try:
            mask = process_mask(mpath)
        except Exception as e:
            log.warning(f"  SKIPPED {stem}: {e}")
            errored.append({"file": stem, "error": str(e)})
            continue

        # Get affine from translated PD if available
        affine, zooms = None, None
        if pd_dir:
            affine, zooms = get_pd_affine(pd_dir, stem)

        if affine is None:
            # Fallback: compute from DESS spacing
            try:
                img      = sitk.ReadImage(dess_path)
                img      = sitk.DICOMOrient(img, "RAS")
                sp       = img.GetSpacing()
                arr      = sitk.GetArrayFromImage(img)  # (n_S, n_A, n_R)
                sp_R     = float(sp[0])
                sp_A     = float(sp[1])
                sp_S     = float(sp[2])
                target_ip = min(sp_A, sp_S)
                n_A_orig  = arr.shape[1]
                n_S_orig  = arr.shape[0]
                n_A_rs    = round(n_A_orig * sp_A / target_ip)
                n_S_rs    = round(n_S_orig * sp_S / target_ip)
                eff_sp_A  = target_ip * n_A_rs / 384
                eff_sp_S  = target_ip * n_S_rs / 384
                affine    = np.diag([sp_R, eff_sp_A, eff_sp_S, 1.0]).astype(np.float32)
                zooms     = (sp_R, eff_sp_A, eff_sp_S)
            except Exception as e:
                log.warning(f"  SKIPPED {stem} (affine fallback failed): {e}")
                errored.append({"file": stem, "error": f"affine: {e}"})
                continue

        new_img = nib.Nifti1Image(mask, affine)
        new_img.header.set_zooms(zooms)
        new_img.header.set_data_dtype(np.int16)
        nib.save(new_img, out_path)
        log.info(f"  Saved -> {out_path}  affine diag={affine.diagonal()[:3]}")
        matched += 1

    # ── Summary ──────────────────────────────────────────────────────────
    log.info(f"\n{'='*50}")
    log.info(f"Done. Processed={matched}  Skipped(no DESS)={skipped}  Errored={len(errored)}")
    if errored:
        log.warning(f"\nFailed masks ({len(errored)}):")
        for e in errored:
            log.warning(f"  {e['file']}: {e['error']}")
    log.info(f"{'='*50}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_root", required=True)
    ap.add_argument("--dess_root", required=True)
    ap.add_argument("--out_root",  required=True)
    ap.add_argument("--pd_dir",    default=None,
                    help="Path to translated PD NIfTIs — copies affine directly for guaranteed alignment")
    args = ap.parse_args()
    run(args.mask_root, args.dess_root, args.out_root, args.pd_dir)