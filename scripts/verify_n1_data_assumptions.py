"""
verify_n1_data_assumptions.py
==============================
Run this on BigRed FIRST, before segmentation/prepare_slab_fake_pd.py touches
anything. N1's hand-off spec (docs/reggan_paper_planning.md, Phase 6 section)
requires confirming three assumptions before writing the slab-averaging code:

  (a) each fake-PD volume has the full 160-slice through-plane stack, not
      only the slices that happen to contain meniscus
  (b) slice index 0..N-1 along axis 0 is through-plane order at ~0.8mm
      (this script reports shape + spacing; you judge (b) against what you
      know about how the volumes were saved -- it can't be fully confirmed
      from the array alone)
  (c) masks exist for every slice (some may be empty/background-only), not
      only for mask-bearing slices

Prints one line per checked patient. Does not write anything.

Usage:
    python scripts/verify_n1_data_assumptions.py \
        --fake_pd_dir /N/project/prostate_cancer_ai/anshika/regGAN/results/fake_pd_all155 \
        --mask_dir    /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed_v3/masks \
        --n_patients  5
"""
import argparse
import glob
import os
import re

import nibabel as nib
import numpy as np

LATERAL_LABEL = 5
MEDIAL_LABEL = 6


def extract_patient_id(filename):
    base = os.path.basename(filename)
    m = re.match(r"(MTR_\d+)", base)
    if m:
        return m.group(1)
    for suf in ["_sl", "_pd_translated", "_mask"]:
        if suf in base:
            return base.split(suf)[0]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_pd_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--n_patients", type=int, default=5,
                     help="How many patients to spot-check (0 = all)")
    args = ap.parse_args()

    fake_files = sorted(glob.glob(os.path.join(args.fake_pd_dir, "*.nii.gz")))
    mask_files = sorted(glob.glob(os.path.join(args.mask_dir, "*.nii.gz")))
    print(f"fake_pd_dir: {len(fake_files)} .nii.gz files")
    print(f"mask_dir:    {len(mask_files)} .nii.gz files")

    fake_by_pid = {extract_patient_id(f): f for f in fake_files}
    mask_by_pid = {extract_patient_id(f): f for f in mask_files}
    common = sorted(set(fake_by_pid) & set(mask_by_pid))
    print(f"patients with both: {len(common)}")

    check_list = common if args.n_patients == 0 else common[:args.n_patients]

    print(f"\n{'patient':<10} {'fake_shape':<18} {'fake_spacing':<20} "
          f"{'mask_shape':<18} {'mask_nonempty_slices':<22} {'mask_total_slices'}")
    for pid in check_list:
        fake_img = nib.load(fake_by_pid[pid])
        mask_img = nib.load(mask_by_pid[pid])
        fake_arr = fake_img.get_fdata(dtype=np.float32)
        mask_arr = mask_img.get_fdata().astype(np.int64)

        fake_spacing = tuple(round(float(z), 3) for z in fake_img.header.get_zooms()[:3])
        nz = mask_arr.shape[0]
        nonempty = sum(
            1 for i in range(nz)
            if np.isin(mask_arr[i], [LATERAL_LABEL, MEDIAL_LABEL]).any()
        )
        print(f"{pid:<10} {str(fake_arr.shape):<18} {str(fake_spacing):<20} "
              f"{str(mask_arr.shape):<18} {nonempty:<22} {nz}")

    print("\nRead this against the three assumptions:")
    print("  (a) fake_shape[0] should be ~160 for every patient, not just the "
          "meniscus-bearing count -- if it's already small (e.g. ~30-40), "
          "the fake-PD volumes were pre-filtered upstream and N1's slab math "
          "needs to change.")
    print("  (b) fake_spacing[0] should be ~0.8mm (through-plane DESS spacing). "
          "If it's ~0.31mm or something else, axis 0 may not be the through-plane axis.")
    print("  (c) mask_total_slices should equal fake_shape[0] (mask covers every "
          "slice, empty ones included as all-zero) for the box-average to be "
          "well-defined across the full stack. If mask files only exist for "
          "mask-bearing slices, report that back before N1 proceeds.")


if __name__ == "__main__":
    main()
