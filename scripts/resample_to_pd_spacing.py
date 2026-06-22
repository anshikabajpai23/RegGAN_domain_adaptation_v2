"""
Fix A: Resample fake PD NIfTIs to match real PD voxel spacing.

The translated fake PD volumes inherit DESS spacing (0.80mm through-plane,
~0.42mm in-plane). Real PD is 3.60mm through-plane, ~0.52mm in-plane.
For a downstream segmentation model trained on fake PD to generalize
to real PD, the spacing must match.

This script reads the target spacing from real PD files (or uses a
provided target), resamples fake PD to match, and saves alongside the
originals as *_spacing_corrected.nii.gz.

It also resamples corresponding DESS segmentation masks (order=0).

Usage:
    python scripts/resample_to_pd_spacing.py \
        --fake_pd_dir inference_from_bigred2 \
        --out_dir inference_from_bigred2_resampled \
        --target_spacing 0.80 0.5208 0.5208 \
        --mask_dir preprocessed_masks/masks

    # Or derive target spacing from actual real PD files:
    python scripts/resample_to_pd_spacing.py \
        --fake_pd_dir inference_from_bigred2 \
        --out_dir inference_from_bigred2_resampled \
        --real_pd_dir /path/to/pd-files
"""

import argparse
import glob
import os
import sys
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom


def get_effective_pd_spacing(directory, n=15):
    """
    AUTHENTIC FIX (not a patch): real PD has heterogeneous native
    acquisition resolutions across scans -- some are 384x384 @ ~0.39mm,
    others are 768x768 @ ~0.18-0.21mm (different protocols/sites). Using
    raw native spacing directly (even after RAS-fixing axis order) produces
    inconsistent, incomparable numbers across files, which is what caused
    the earlier bad resample (mean in-plane spacing was pulled down to
    ~0.24mm by averaging in unusually fine-resolution outlier scans).

    The correct quantity is the EFFECTIVE spacing each file would have
    after going through the SAME standard pipeline every other volume in
    this project goes through (preprocess.py's process_volume(),
    infer2.py's get_effective_spacing()): reorient to RAS, compute the
    isotropic in-plane resample factor, then divide by the final 384x384
    resize target. This is the exact same formula already proven correct
    and used consistently for DESS/fake-PD/masks -- reused here for real
    PD instead of inventing a separate approach. Confirmed empirically:
    native (0.39, 0.18, 0.21mm) -> effective (0.39, 0.36, 0.42mm), which
    cluster tightly together instead of looking inconsistent.

    Through-plane is NEVER resampled anywhere in this pipeline, so it's
    just read directly after RAS reorientation (matches every other usage).

    Uses median (not mean) across the sample for robustness to any
    remaining outlier scans.
    """
    files = sorted(glob.glob(os.path.join(directory, "*.nii.gz")))[:n]
    eff_spacings = []
    for f in files:
        img = sitk.ReadImage(f)
        img = sitk.DICOMOrient(img, "RAS")
        sp  = img.GetSpacing()                  # (sp_R, sp_A, sp_S)
        arr = sitk.GetArrayFromImage(img)        # (n_S, n_A, n_R)

        sp_R, sp_A, sp_S = float(sp[0]), float(sp[1]), float(sp[2])
        n_A, n_S = arr.shape[1], arr.shape[0]

        target_ip = min(sp_A, sp_S)              # isotropic in-plane factor
        n_A_rs = round(n_A * sp_A / target_ip)
        n_S_rs = round(n_S * sp_S / target_ip)

        eff_sp_A = target_ip * n_A_rs / 384      # divide by final resize target
        eff_sp_S = target_ip * n_S_rs / 384

        eff_spacings.append(np.array([sp_R, eff_sp_A, eff_sp_S]))

    eff_spacings = np.array(eff_spacings)
    median = np.median(eff_spacings, axis=0)
    print(f"  Effective spacing per file (after 384-resize, RAS-ordered):")
    for f, sp in zip(files, eff_spacings):
        print(f"    {os.path.basename(f)}: R={sp[0]:.4f} A={sp[1]:.4f} S={sp[2]:.4f} mm")
    print(f"  Median across {len(files)} files: "
          f"R={median[0]:.4f} A={median[1]:.4f} S={median[2]:.4f} mm")
    return median


def resample_volume(data, src_spacing, tgt_spacing, order):
    factors = np.array(src_spacing) / np.array(tgt_spacing)
    resampled = zoom(data, factors, order=order, prefilter=(order > 0))
    return resampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_pd_dir",     default="inference_from_bigred2")
    ap.add_argument("--out_dir",         default="inference_from_bigred2_resampled")
    ap.add_argument("--real_pd_dir",     default=None,
                    help="If set, derive target spacing from real PD files")
    ap.add_argument("--target_spacing",  type=float, nargs=3, default=None,
                    help="Explicit target spacing: R A S in mm. "
                         "Typical real PD effective: 3.60 0.5208 0.5208")
    ap.add_argument("--mask_dir",        default=None,
                    help="Dir of *_mask.nii.gz files to resample alongside")
    ap.add_argument("--mask_out_dir",    default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def absp(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    fake_dir  = absp(args.fake_pd_dir)
    out_dir   = absp(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Determine target spacing
    if args.target_spacing is not None:
        tgt = np.array(args.target_spacing, dtype=np.float64)
        print(f"  Using provided target spacing: R={tgt[0]:.4f}  A={tgt[1]:.4f}  S={tgt[2]:.4f} mm")
    elif args.real_pd_dir:
        pd_dir = absp(args.real_pd_dir)
        tgt = get_effective_pd_spacing(pd_dir)
        print(f"  Derived target spacing from {absp(args.real_pd_dir)}: "
              f"R={tgt[0]:.4f}  A={tgt[1]:.4f}  S={tgt[2]:.4f} mm")
    else:
        # Default: typical real PD effective spacing
        tgt = np.array([3.60, 0.5208, 0.5208])
        print(f"  Using default PD spacing: R={tgt[0]:.4f}  A={tgt[1]:.4f}  S={tgt[2]:.4f} mm")
        print("  NOTE: pass --real_pd_dir or --target_spacing for accurate target")

    # Resample fake PD volumes
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.nii.gz")))
    print(f"\n  Resampling {len(fake_files)} fake PD volumes...")

    for path in fake_files:
        img   = nib.load(path)
        data  = img.get_fdata(dtype=np.float32)
        src   = np.array(img.header.get_zooms()[:3])

        resampled = resample_volume(data, src, tgt, order=3)

        # STAGE 5b SYNC: inherit the source's actual direction+origin
        # (already fixed in infer2.py's get_effective_affine()) instead of
        # a zero-origin identity-direction diagonal -- just rescale the
        # spacing part to the new target spacing.
        direction  = img.affine[:3, :3] / src[None, :]
        new_affine = np.eye(4, dtype=np.float64)
        new_affine[:3, :3] = direction * tgt[None, :]
        new_affine[:3, 3]  = img.affine[:3, 3]
        new_affine = new_affine.astype(np.float32)

        new_img = nib.Nifti1Image(resampled, new_affine)
        new_img.header.set_zooms(tuple(tgt))

        stem    = os.path.basename(path).replace(".nii.gz", "")
        out_path = os.path.join(out_dir, stem + "_spacing_corrected.nii.gz")
        nib.save(new_img, out_path)

        orig_shape = data.shape
        new_shape  = resampled.shape
        print(f"    {stem}: {orig_shape} @({src[0]:.2f},{src[1]:.3f},{src[2]:.3f})mm "
              f"→ {new_shape} @({tgt[0]:.2f},{tgt[1]:.4f},{tgt[2]:.4f})mm")

    # Resample masks (order=0)
    if args.mask_dir:
        mask_dir = absp(args.mask_dir)
        mask_out = absp(args.mask_out_dir) if args.mask_out_dir else os.path.join(out_dir, "masks")
        os.makedirs(mask_out, exist_ok=True)
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
        print(f"\n  Resampling {len(mask_files)} masks (order=0)...")

        for path in mask_files:
            img   = nib.load(path)
            data  = img.get_fdata(dtype=np.float32).astype(np.int16)
            src   = np.array(img.header.get_zooms()[:3])

            resampled = resample_volume(data.astype(np.float32), src, tgt, order=0).astype(np.int16)

            # STAGE 5b SYNC: same fix as the fake-PD branch above.
            direction  = img.affine[:3, :3] / src[None, :]
            new_affine = np.eye(4, dtype=np.float64)
            new_affine[:3, :3] = direction * tgt[None, :]
            new_affine[:3, 3]  = img.affine[:3, 3]
            new_affine = new_affine.astype(np.float32)

            new_img = nib.Nifti1Image(resampled, new_affine)
            new_img.header.set_zooms(tuple(tgt))

            stem    = os.path.basename(path)
            out_path = os.path.join(mask_out, stem)
            nib.save(new_img, out_path)
            print(f"    {stem}: {data.shape} → {resampled.shape}")

            # Verify labels preserved
            orig_labels = set(np.unique(data))
            new_labels  = set(np.unique(resampled))
            if orig_labels != new_labels:
                print(f"    WARNING: label set changed {orig_labels} → {new_labels}")

    print(f"\n  Done. Resampled fake PD saved to: {out_dir}")
    print()


if __name__ == "__main__":
    main()
