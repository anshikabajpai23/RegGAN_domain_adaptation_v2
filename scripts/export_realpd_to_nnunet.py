"""
export_realpd_to_nnunet.py
============================
Baseline (nnU-Net, docs/reggan_paper_planning.md's "no nnU-Net" gap): convert
real_pd_seg_data_v7/ (flat per-slice .npy, binary meniscus masks) into
nnU-Net's raw dataset format, reconstructing contiguous 3D volumes per
patient (slices at their true index) so both the 2d and 3d_fullres configs
are valid.

Writes splits_final_fold0.json using the SAME train/val patient split
already fixed by real_pd_seg_data_v7/{train,val}/ -- so nnU-Net's fold 0 is
evaluated on the SAME 2 val patients every other model in this project uses,
not nnU-Net's own random 5-fold split.

Does not modify real_pd_seg_data_v7/ -- reads only, writes to --nnunet_raw.

Usage:
    python scripts/export_realpd_to_nnunet.py \
        --real_data_root /path/to/real_pd_seg_data_v7 \
        --nnunet_raw     /path/to/nnUNet_raw \
        --dataset_id 501 --dataset_name MeniscusPD
"""
import argparse
import glob
import json
import logging
import os

import numpy as np
import nibabel as nib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PD_SPACING = (3.6, 0.39, 0.39)  # through-plane, in-plane, in-plane -- matches
                                 # PD_SPACING used elsewhere in this project


def collect_patients(split_dir):
    """Group flat '{pid}_{sidx:04d}.npy' files by patient, sorted by slice idx."""
    img_dir = os.path.join(split_dir, "images")
    mask_dir = os.path.join(split_dir, "masks")
    by_pid = {}
    for f in sorted(glob.glob(os.path.join(img_dir, "*.npy"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        pid, sidx = stem.rsplit("_", 1)
        by_pid.setdefault(pid, []).append((int(sidx), f, os.path.join(mask_dir, stem + ".npy")))
    for pid in by_pid:
        by_pid[pid].sort(key=lambda t: t[0])
    return by_pid


def build_volume(entries):
    """Place each slice at its true index. prepare_real_pd_labeled.py skipped
    near-empty background slices (image mean < 0.02), so stacking the saved
    files directly would close those gaps and make the volume non-contiguous
    -- wrong for nnU-Net's 3d_fullres config. Skipped slices are zero-filled
    (background image, empty mask)."""
    first = np.load(entries[0][1])
    n_z = entries[-1][0] + 1
    img_vol = np.zeros((n_z,) + first.shape, dtype=np.float32)
    mask_vol = np.zeros((n_z,) + first.shape, dtype=np.uint8)
    for sidx, f, m in entries:
        img_vol[sidx] = np.load(f).astype(np.float32)
        mask_vol[sidx] = np.load(m).astype(np.uint8)
    return img_vol, mask_vol, n_z - len(entries)  # (Nz, H, W), n zero-filled


def save_case(pid, img_vol, mask_vol, images_dir, labels_dir):
    # nibabel axis order (x,y,z) -- our array is (z,y,x)=(slice,H,W); transpose
    # so spacing/orientation are internally consistent (this is a fresh,
    # self-contained coordinate frame for nnU-Net, not aligned to any other
    # NIfTI in this project -- nnU-Net only needs image/label voxel grids to
    # match each other, which they do by construction here).
    img_t = np.transpose(img_vol, (2, 1, 0))
    mask_t = np.transpose(mask_vol, (2, 1, 0))
    affine = np.diag([PD_SPACING[2], PD_SPACING[1], PD_SPACING[0], 1.0]).astype(np.float32)

    nib.save(nib.Nifti1Image(img_t, affine), os.path.join(images_dir, f"{pid}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(mask_t.astype(np.uint8), affine), os.path.join(labels_dir, f"{pid}.nii.gz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_data_root", required=True,
                    help="real_pd_seg_data_v7/ (has train/ and val/ subdirs)")
    ap.add_argument("--nnunet_raw", required=True, help="nnUNet_raw base dir")
    ap.add_argument("--dataset_id", type=int, default=501)
    ap.add_argument("--dataset_name", default="MeniscusPD")
    args = ap.parse_args()

    ds_folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    raw_dir = os.path.join(args.nnunet_raw, ds_folder_name)
    images_dir = os.path.join(raw_dir, "imagesTr")
    labels_dir = os.path.join(raw_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    train_patients = collect_patients(os.path.join(args.real_data_root, "train"))
    val_patients = collect_patients(os.path.join(args.real_data_root, "val"))
    log.info(f"Train patients: {sorted(train_patients)} ({len(train_patients)})")
    log.info(f"Val patients:   {sorted(val_patients)} ({len(val_patients)})")

    all_patients = {**train_patients, **val_patients}
    for pid, entries in sorted(all_patients.items()):
        img_vol, mask_vol, n_filled = build_volume(entries)
        save_case(pid, img_vol, mask_vol, images_dir, labels_dir)
        log.info(f"  {pid}: {img_vol.shape[0]} slices ({n_filled} zero-filled gaps) "
                 f"-> {pid}_0000.nii.gz / {pid}.nii.gz")

    dataset_json = {
        "channel_names": {"0": "PD_MRI"},
        "labels": {"background": 0, "meniscus": 1},
        "numTraining": len(all_patients),
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(raw_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)
    log.info(f"dataset.json -> {os.path.join(raw_dir, 'dataset.json')}")

    # Custom splits_final.json: fold 0 = our SAME 13 train / 2 val patients.
    # nnU-Net uses this instead of its own random 5-fold split if the file is
    # in the preprocessed dataset folder when training starts. Written to the
    # raw folder here; setup_nnunet_baseline.sh copies it into the
    # preprocessed folder AFTER plan_and_preprocess, so preprocessing can't
    # clobber it.
    splits = [{"train": sorted(train_patients), "val": sorted(val_patients)}]
    splits_path = os.path.join(raw_dir, "splits_final_fold0.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    log.info(f"splits (fold 0 = project's own 13/2 split) -> {splits_path}")


if __name__ == "__main__":
    main()
