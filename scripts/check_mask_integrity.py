"""
Stage 1b: Mask integrity sweep.

Tries to load every mask .nii.gz file and flags any that fail —
catches truncated/corrupted files (e.g. incomplete scp/rsync transfers)
before they crash downstream steps (Stage 4 evaluation, Stage 8 segmentation).

Usage:
    venv/bin/python scripts/check_mask_integrity.py --mask_dir preprocessed_masks/masks
"""
import argparse
import glob
import os
import numpy as np
import nibabel as nib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", default="preprocessed_masks/masks")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.mask_dir, "*.nii.gz")))
    print(f"Checking {len(files)} mask files in {args.mask_dir}\n")

    good, bad = [], []
    for f in files:
        name = os.path.basename(f)
        try:
            img = nib.load(f)
            data = img.get_fdata()
            labels = np.unique(data)
            file_size = os.path.getsize(f)
            print(f"  OK   {name:45s} shape={data.shape} labels={labels} size={file_size}B")
            good.append(f)
        except Exception as e:
            file_size = os.path.getsize(f)
            print(f"  FAIL {name:45s} size_on_disk={file_size}B  error={e}")
            bad.append((f, str(e)))

    print(f"\n{len(good)} OK, {len(bad)} CORRUPTED")
    if bad:
        print("\nCorrupted files:")
        for f, err in bad:
            print(f"  - {f}")
            print(f"    {err}")
        print("\nThese files cannot be used until re-generated or re-copied.")
        print("If this repo was transferred via scp/rsync, the most likely cause")
        print("is an incomplete transfer of just this file — re-copy it, or")
        print("re-run preprocess_masks.py for this patient from the raw DESS mask NIfTI.")


if __name__ == "__main__":
    main()
