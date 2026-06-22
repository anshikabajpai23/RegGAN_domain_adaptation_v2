"""
prepare_meniscus_masks.py
==========================
The ONE additional preprocessing step needed before fine-tuning:
the reference repo (pitthexai/Knee_MRI_Segmentation_2.5D) expects masks with
a small, fixed set of classes. Our DESS masks have 7 labels (0-6: background +
6 tissue types). We only care about meniscus (5=lateral, 6=medial) -- this
script collapses everything else to background, producing a clean 3-class
mask: 0=background/other-tissue, 1=lateral meniscus, 2=medial meniscus.

Does NOT modify any existing pipeline files -- reads from preprocessed_v2/
(or whatever --mask_dir/--pd_dir is given) and writes new output only.

Output format matches what the reference repo's Dataset class expects:
per-slice .npy files (int64), one per meniscus-containing slice, named
"{patient_stem}_slice_{idx:03d}.npy" in a per-patient subdirectory --
mirrors their "{study_id}_slice_{n}.jpg" + ".../{study_id}/{file}.npy" layout.

Usage:
    python segmentation/prepare_meniscus_masks.py \
        --mask_dir   /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed_v2/masks \
        --fake_pd_dir /N/project/prostate_cancer_ai/anshika/regGAN/results/stage4_fake_pd \
        --out_root   /N/project/prostate_cancer_ai/anshika/regGAN/segmentation_data \
        --splits     /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed_v2/splits.json \
        --split      train
"""
import argparse
import glob
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import nibabel as nib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

LATERAL_LABEL = 5
MEDIAL_LABEL  = 6


def extract_patient_id(filename):
    """Same convention used in evaluate.py's Stage 4a fix -- canonical
    short patient ID (e.g. 'MTR_005') from any of the filename schemes
    used across this pipeline."""
    base = os.path.basename(filename)
    m = re.match(r"(MTR_\d+)", base)
    if m:
        return m.group(1)
    for suf in ["_sl", "_pd_translated", "_mask"]:
        if suf in base:
            return base.split(suf)[0]
    return base


def collapse_to_meniscus_classes(mask_slice):
    """0=background/other tissue, 1=lateral meniscus, 2=medial meniscus."""
    out = np.zeros_like(mask_slice, dtype=np.int64)
    out[mask_slice == LATERAL_LABEL] = 1
    out[mask_slice == MEDIAL_LABEL]  = 2
    return out


def load_split_patients(splits_path, split):
    with open(splits_path) as f:
        splits = json.load(f)
    slice_paths = splits["dess"][split]
    stems = sorted(set(Path(p).stem.rsplit("_sl", 1)[0] for p in slice_paths))
    return stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True,
                     help="Regenerated masks dir (preprocessed_v2/masks)")
    ap.add_argument("--fake_pd_dir", required=True,
                     help="Translated pseudo-PD volumes dir")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--min_meniscus_pixels", type=int, default=10,
                     help="Skip slices with fewer than this many meniscus pixels "
                          "(avoids training on near-empty slices)")
    args = ap.parse_args()

    split_stems = set(load_split_patients(args.splits, args.split))
    log.info(f"Split '{args.split}': {len(split_stems)} patient stems")

    # Key fake-PD volumes by canonical patient ID, restricted to this split
    fake_pd_by_pid = {}
    for f in sorted(glob.glob(os.path.join(args.fake_pd_dir, "*.nii.gz"))):
        pid = extract_patient_id(f)
        stem_matches = [s for s in split_stems if extract_patient_id(s) == pid]
        if stem_matches:
            fake_pd_by_pid[pid] = f

    # Key masks by canonical patient ID
    mask_by_pid = {}
    for f in sorted(glob.glob(os.path.join(args.mask_dir, "*.nii.gz"))):
        pid = extract_patient_id(f)
        mask_by_pid[pid] = f

    common_pids = sorted(set(fake_pd_by_pid) & set(mask_by_pid))
    log.info(f"Patients with both fake-PD and mask, in split '{args.split}': "
             f"{len(common_pids)} / {len(fake_pd_by_pid)} fake-PD available")

    img_out_root  = os.path.join(args.out_root, args.split, "images")
    mask_out_root = os.path.join(args.out_root, args.split, "masks")

    total_slices = 0
    total_skipped_empty = 0

    for pid in common_pids:
        fake_vol = nib.load(fake_pd_by_pid[pid]).get_fdata(dtype=np.float32)
        mask_vol = nib.load(mask_by_pid[pid]).get_fdata().astype(np.int64)

        n = min(fake_vol.shape[0], mask_vol.shape[0])
        img_dir  = os.path.join(img_out_root, pid)
        mask_dir = os.path.join(mask_out_root, pid)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        n_saved = 0
        for idx in range(n):
            collapsed = collapse_to_meniscus_classes(mask_vol[idx])
            if (collapsed > 0).sum() < args.min_meniscus_pixels:
                total_skipped_empty += 1
                continue

            np.save(os.path.join(img_dir, f"{pid}_slice_{idx:03d}.npy"),
                    fake_vol[idx].astype(np.float32))
            np.save(os.path.join(mask_dir, f"{pid}_slice_{idx:03d}.npy"),
                    collapsed)
            n_saved += 1

        log.info(f"  {pid}: {n_saved}/{n} slices kept (meniscus present)")
        total_slices += n_saved

    log.info(f"\nDone. {args.split}: {total_slices} meniscus-positive slices saved, "
             f"{total_skipped_empty} empty slices skipped.")
    log.info(f"Images -> {img_out_root}")
    log.info(f"Masks  -> {mask_out_root}")


if __name__ == "__main__":
    main()
