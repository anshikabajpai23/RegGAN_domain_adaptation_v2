"""
Reconstruct a 3D volume (image or predicted mask) from per-slice .npy files,
correctly handling slices that were skipped during extraction (background,
mean < skip_empty_pct in preprocess.py's extract_slices()).

Filenames are expected as "{patient_stem}_sl{index:04d}.npy" where `index`
is the ORIGINAL slice position in the source volume (preprocess.py preserves
this even when some indices are skipped) -- this script relies on that
invariant to place each slice at the correct depth and pad any missing
(skipped) slices with a fill value, rather than silently compacting the
volume and shifting everything to the wrong depth.

Usage:
    venv/bin/python scripts/reconstruct_volume_from_slices.py \
        --slice_dir preprocessed_v2/slices/dess \
        --patient_stem MTR_005_Anonymized_2378615199_e1 \
        --n_slices 160 \
        --reference_nifti /path/to/original_dess_or_pd.nii.gz \
        --out_path reconstructed_MTR_005.nii.gz \
        --fill_value 0.0
"""
import argparse
import glob
import os
import re
import numpy as np
import nibabel as nib


def reconstruct(slice_dir, patient_stem, n_slices, fill_value=0.0, dtype=np.float32):
    """
    dtype: use np.int16 for segmentation masks (preserves integer labels
    exactly, matches preprocess_masks.py's mask dtype). Default np.float32
    for images.
    """
    files = sorted(glob.glob(os.path.join(slice_dir, f"{patient_stem}_sl*.npy")))
    if not files:
        raise FileNotFoundError(f"No slices found for {patient_stem} in {slice_dir}")

    sample = np.load(files[0])
    H, W = sample.shape
    volume = np.full((n_slices, H, W), fill_value, dtype=dtype)

    pattern = re.compile(r"_sl(\d{4})\.npy$")
    placed, skipped_indices = [], []
    for f in files:
        m = pattern.search(f)
        if not m:
            raise ValueError(f"Filename doesn't match expected _sl#### pattern: {f}")
        idx = int(m.group(1))
        if idx >= n_slices:
            raise ValueError(f"Slice index {idx} >= n_slices={n_slices} for {f}")
        volume[idx] = np.load(f)
        placed.append(idx)

    all_indices = set(range(n_slices))
    skipped_indices = sorted(all_indices - set(placed))

    return volume, placed, skipped_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice_dir", required=True)
    ap.add_argument("--patient_stem", required=True)
    ap.add_argument("--n_slices", type=int, required=True,
                     help="Total slices in the ORIGINAL source volume (R axis size)")
    ap.add_argument("--reference_nifti", default=None,
                     help="Original NIfTI to copy affine/spacing from (recommended)")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--fill_value", type=float, default=0.0,
                     help="Value for skipped/background slice positions")
    ap.add_argument("--mask_mode", action="store_true",
                     help="Use int16 dtype to preserve integer segmentation labels exactly")
    args = ap.parse_args()

    dtype = np.int16 if args.mask_mode else np.float32
    volume, placed, skipped = reconstruct(
        args.slice_dir, args.patient_stem, args.n_slices, args.fill_value, dtype
    )

    print(f"Reconstructed volume shape: {volume.shape}")
    print(f"Placed {len(placed)} slices, {len(skipped)} positions filled with {args.fill_value}")
    if skipped:
        print(f"Skipped (background) indices: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    if args.reference_nifti:
        ref = nib.load(args.reference_nifti)
        affine = ref.affine
        zooms = ref.header.get_zooms()[:3]
        print(f"Using reference affine/spacing from {args.reference_nifti}: spacing={zooms}")
    else:
        affine = np.eye(4)
        print("WARNING: no reference NIfTI given, using identity affine — "
              "spacing/orientation will be WRONG. Pass --reference_nifti.")

    img = nib.Nifti1Image(volume, affine)
    if args.reference_nifti:
        img.header.set_zooms(zooms)
    nib.save(img, args.out_path)
    print(f"Saved -> {args.out_path}")


if __name__ == "__main__":
    main()
