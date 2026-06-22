"""
Check 1: Voxel spacing audit.

Loads fake PD outputs (local) and any available real DESS/PD NIfTIs,
prints shape, spacing, and physical FOV for each so you can immediately
see whether fake PD has been saved with the right metadata.

Usage:
    python scripts/check_spacing.py \
        --fake_pd_dir inference_from_bigred2 \
        --dess_dir /path/to/dess-files \
        --pd_dir /path/to/pd-files \
        --n 3

    # local-only (no dataset available):
    python scripts/check_spacing.py --fake_pd_dir inference_from_bigred2 --n 5
"""

import argparse
import glob
import os
import sys
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def report(label, path):
    """
    FIX: previously read raw nib.header.get_zooms() with no RAS
    reorientation. Real PD / DESS raw files store axes in their own native
    order, NOT the (R, A, S) = (through-plane, in-plane, in-plane) order
    everything else in this pipeline uses after RAS reorientation. This
    caused the through-plane value to silently land in the wrong printed
    position for real PD, producing a misleading RATIO summary. Now always
    RAS-reorients first, matching preprocess.py/infer2.py/
    preprocess_masks.py/resample_to_pd_spacing.py's convention.
    """
    img_sitk = sitk.ReadImage(path)
    img_sitk = sitk.DICOMOrient(img_sitk, "RAS")
    zooms    = np.array(img_sitk.GetSpacing())          # (sp_R, sp_A, sp_S)
    arr      = sitk.GetArrayFromImage(img_sitk)          # (n_S, n_A, n_R)
    shape    = np.array([arr.shape[2], arr.shape[1], arr.shape[0]])  # (R, A, S)
    fov      = shape * zooms  # physical extent in mm

    nib_img  = nib.load(path)  # only for affine diag display below

    print(f"\n  [{label}]")
    print(f"    File   : {os.path.basename(path)}")
    print(f"    Shape  : {tuple(shape)}  (R, A, S order, after RAS reorientation)")
    print(f"    Spacing: R={zooms[0]:.4f}  A={zooms[1]:.4f}  S={zooms[2]:.4f}  mm")
    print(f"    FOV    : {fov[0]:.1f} x {fov[1]:.1f} x {fov[2]:.1f}  mm")
    print(f"    Affine diag (raw file, pre-reorientation): {np.diag(nib_img.affine)[:3].round(4)}")
    return dict(shape=tuple(shape), spacing=tuple(zooms.round(4)), fov=tuple(fov.round(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_pd_dir", default="inference_from_bigred2")
    ap.add_argument("--dess_dir",    default=None)
    ap.add_argument("--pd_dir",      default=None)
    ap.add_argument("--n",           type=int, default=3, help="files to sample per group")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def pick(directory, n):
        files = sorted(glob.glob(os.path.join(directory, "*.nii.gz")))
        if not files:
            files = sorted(glob.glob(os.path.join(directory, "*.nii")))
        return files[:n]

    print("\n" + "="*60)
    print(" VOXEL SPACING AUDIT")
    print("="*60)

    results = {}

    # Fake PD
    fake_dir = args.fake_pd_dir if os.path.isabs(args.fake_pd_dir) else os.path.join(base, args.fake_pd_dir)
    fake_files = pick(fake_dir, args.n)
    if not fake_files:
        print(f"  ERROR: no NIfTI files found in {fake_dir}")
        sys.exit(1)
    print(f"\n--- Fake PD (translated DESS→PD) [{fake_dir}] ---")
    fake_reports = [report("Fake PD", f) for f in fake_files]
    results["fake_pd"] = fake_reports

    # Real DESS
    if args.dess_dir:
        dess_files = pick(args.dess_dir, args.n)
        print(f"\n--- Real DESS [{args.dess_dir}] ---")
        results["dess"] = [report("DESS", f) for f in dess_files]

    # Real PD
    if args.pd_dir:
        pd_files = pick(args.pd_dir, args.n)
        print(f"\n--- Real PD [{args.pd_dir}] ---")
        results["real_pd"] = [report("Real PD", f) for f in pd_files]

    # Summary: compare spacings
    print("\n" + "="*60)
    print(" SPACING COMPARISON SUMMARY")
    print("="*60)

    fake_sp = np.array([r["spacing"] for r in fake_reports])
    print(f"\n  Fake PD spacing (mean over {len(fake_reports)} files):")
    print(f"    R (through-plane): {fake_sp[:,0].mean():.4f} mm")
    print(f"    A (in-plane):      {fake_sp[:,1].mean():.4f} mm")
    print(f"    S (in-plane):      {fake_sp[:,2].mean():.4f} mm")

    if "real_pd" in results:
        real_sp_raw = np.array([r["spacing"] for r in results["real_pd"]])
        print(f"\n  Real PD RAW native spacing (mean over {len(results['real_pd'])} files):")
        print(f"    R (through-plane): {real_sp_raw[:,0].mean():.4f} mm")
        print(f"    A (in-plane):      {real_sp_raw[:,1].mean():.4f} mm")
        print(f"    S (in-plane):      {real_sp_raw[:,2].mean():.4f} mm")
        print(f"  (NOTE: raw native spacing varies across real PD scans by acquisition")
        print(f"   protocol -- this raw mean is NOT the right quantity to compare")
        print(f"   against fake PD, which is resampled to an EFFECTIVE spacing. See below.)")

        # FIX: compare fake PD's effective spacing against real PD's effective
        # spacing (same formula resample_to_pd_spacing.py's
        # get_effective_pd_spacing() uses: simulate the 384-resize every
        # volume in this pipeline goes through), not raw native spacing --
        # otherwise this comparison is apples (effective) vs oranges (raw),
        # which produced a false "MISMATCH" even on a correctly-resampled run.
        real_sp_eff = []
        for r in results["real_pd"]:
            n_R, n_A, n_S = r["shape"]
            sp_R, sp_A, sp_S = r["spacing"]
            target_ip = min(sp_A, sp_S)
            eff_A = target_ip * n_A / 384
            eff_S = target_ip * n_S / 384
            real_sp_eff.append([sp_R, eff_A, eff_S])
        real_sp = np.median(np.array(real_sp_eff), axis=0, keepdims=True)
        print(f"\n  Real PD EFFECTIVE spacing (median over {len(results['real_pd'])} files, "
              f"after simulated 384-resize):")
        print(f"    R (through-plane): {real_sp[0,0]:.4f} mm")
        print(f"    A (in-plane):      {real_sp[0,1]:.4f} mm")
        print(f"    S (in-plane):      {real_sp[0,2]:.4f} mm")

        ratio = real_sp.mean(0) / fake_sp.mean(0)
        print(f"\n  RATIO (real_PD / fake_PD):")
        print(f"    R: {ratio[0]:.2f}x  A: {ratio[1]:.2f}x  S: {ratio[2]:.2f}x")
        if abs(ratio[0] - 1.0) > 0.05:
            print("  *** MISMATCH in through-plane spacing — fake PD will appear too thick/thin for segmentation ***")
        if abs(ratio[1] - 1.0) > 0.05 or abs(ratio[2] - 1.0) > 0.05:
            print("  *** MISMATCH in in-plane spacing — resampling needed before segmentation training ***")

    if "dess" in results:
        dess_sp = np.array([r["spacing"] for r in results["dess"]])
        print(f"\n  DESS spacing (mean):")
        print(f"    R (through-plane): {dess_sp[:,0].mean():.4f} mm")
        print(f"    A (in-plane):      {dess_sp[:,1].mean():.4f} mm")
        print(f"    S (in-plane):      {dess_sp[:,2].mean():.4f} mm")

    print()


if __name__ == "__main__":
    main()
