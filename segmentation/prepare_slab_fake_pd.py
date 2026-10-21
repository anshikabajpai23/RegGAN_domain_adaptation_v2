"""
prepare_slab_fake_pd.py
========================
N1 (docs/reggan_paper_planning.md, Phase 6): make fake-PD training data match
real PD's through-plane physics. Existing fake-PD prep
(prepare_meniscus_masks.py) keeps DESS's thin 0.8mm slices as-is -- sharp
through-plane detail everywhere, unlike a real PD slice which is actually a
3.6mm-thick slab (the scanner averages tissue over that thickness). This
script box-averages groups of ~4.5 consecutive fake-PD slices into one slab,
producing ~35-36 slabs per volume (matching real PD's ~36), and averages the
one-hot mask the same way so slabs that only partially contain meniscus get
a FRACTIONAL label instead of a hard 0/1/2.

Does NOT modify any existing pipeline files or output directories -- reads
fake-PD volumes + DESS-derived masks, writes to a brand-new directory
(--out_root, e.g. segmentation_data_v4_slab/). v2/v3 outputs are untouched.

Output format:
  {out_root}/{split}/images/{pid}/{pid}_slice_{k:03d}.npy   float32 (384,384)
  {out_root}/{split}/masks/{pid}/{pid}_slice_{k:03d}.npy    float32 (3,384,384)
    channel 0 = background fraction, 1 = lateral meniscus fraction,
    2 = medial meniscus fraction. Channels sum to 1.0 at every pixel.

Usage:
    python segmentation/prepare_slab_fake_pd.py \
        --mask_dir    /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed_v3/masks \
        --fake_pd_dir /N/project/prostate_cancer_ai/anshika/regGAN/results/fake_pd_all155 \
        --out_root    /N/project/prostate_cancer_ai/anshika/regGAN/segmentation_data_v4_slab \
        --splits      /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed_v3/splits.json \
        --split       train
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
MEDIAL_LABEL = 6
N_CLASSES = 3  # background, lateral, medial


def extract_patient_id(filename):
    """Same convention as prepare_meniscus_masks.py -- keep patient-ID
    extraction identical across prep scripts so splits.json matching agrees."""
    base = os.path.basename(filename)
    m = re.match(r"(MTR_\d+)", base)
    if m:
        return m.group(1)
    for suf in ["_sl", "_pd_translated", "_mask"]:
        if suf in base:
            return base.split(suf)[0]
    return base


def load_split_patients(splits_path, split):
    with open(splits_path) as f:
        splits = json.load(f)
    slice_paths = splits["dess"][split]
    stems = sorted(set(Path(p).stem.rsplit("_sl", 1)[0] for p in slice_paths))
    return stems


def one_hot_collapse(mask_slice):
    """(H,W) DESS labels -> (3,H,W) one-hot {bg, lateral, medial}."""
    h, w = mask_slice.shape
    out = np.zeros((N_CLASSES, h, w), dtype=np.float32)
    lat = mask_slice == LATERAL_LABEL
    med = mask_slice == MEDIAL_LABEL
    out[1] = lat.astype(np.float32)
    out[2] = med.astype(np.float32)
    out[0] = 1.0 - out[1] - out[2]
    return out


def slab_average(vol, window):
    """
    Box-average a (Nz, ...) array along axis 0 into slabs of width `window`
    slices (window need not be an integer -- boundary slices get fractional
    weight). Trailing slices that don't fill a full window are dropped
    (documented decision, not in the original spec -- state this when
    reporting N1 results).

    Returns (n_slabs, ...) array, same dtype as input (cast to float32).
    """
    nz = vol.shape[0]
    n_slabs = int(np.floor(nz / window))
    out_shape = (n_slabs,) + vol.shape[1:]
    out = np.zeros(out_shape, dtype=np.float32)

    for k in range(n_slabs):
        z0, z1 = k * window, (k + 1) * window
        i_lo = int(np.floor(z0))
        i_hi = int(np.ceil(z1))  # exclusive upper bound on slice index range
        acc = np.zeros(vol.shape[1:], dtype=np.float64)
        wsum = 0.0
        for i in range(i_lo, min(i_hi, nz)):
            weight = max(0.0, min(z1, i + 1) - max(z0, i))
            if weight <= 0:
                continue
            acc += weight * vol[i].astype(np.float64)
            wsum += weight
        out[k] = (acc / wsum).astype(np.float32) if wsum > 0 else 0.0

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True,
                     help="DESS-derived masks dir (7-label, values 0-6)")
    ap.add_argument("--fake_pd_dir", required=True,
                     help="Translated pseudo-PD volumes dir (full-stack .nii.gz)")
    ap.add_argument("--out_root", required=True,
                     help="NEW output dir, e.g. segmentation_data_v4_slab -- "
                          "must not already hold data from another run")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--window", type=float, default=4.5,
                     help="Slab width in source slices (3.6mm / 0.8mm = 4.5)")
    ap.add_argument("--min_meniscus_fraction_sum", type=float, default=10.0,
                     help="Skip slabs where sum over pixels of "
                          "(frac_lateral + frac_medial) is below this -- "
                          "fractional analogue of prepare_meniscus_masks.py's "
                          "--min_meniscus_pixels=10")
    args = ap.parse_args()

    split_stems = set(load_split_patients(args.splits, args.split))
    log.info(f"Split '{args.split}': {len(split_stems)} patient stems")

    fake_pd_by_pid = {}
    for f in sorted(glob.glob(os.path.join(args.fake_pd_dir, "*.nii.gz"))):
        pid = extract_patient_id(f)
        if any(extract_patient_id(s) == pid for s in split_stems):
            fake_pd_by_pid[pid] = f

    mask_by_pid = {}
    for f in sorted(glob.glob(os.path.join(args.mask_dir, "*.nii.gz"))):
        mask_by_pid[extract_patient_id(f)] = f

    common_pids = sorted(set(fake_pd_by_pid) & set(mask_by_pid))
    log.info(f"Patients with both fake-PD and mask, split '{args.split}': "
             f"{len(common_pids)} / {len(fake_pd_by_pid)} fake-PD available")

    if os.path.exists(args.out_root) and os.listdir(args.out_root):
        log.warning(f"{args.out_root} already exists and is non-empty -- "
                    f"per-patient subdirs will be overwritten if names collide.")

    img_out_root = os.path.join(args.out_root, args.split, "images")
    mask_out_root = os.path.join(args.out_root, args.split, "masks")

    total_slabs, total_skipped = 0, 0

    for pid in common_pids:
        fake_vol = nib.load(fake_pd_by_pid[pid]).get_fdata(dtype=np.float32)
        mask_vol = nib.load(mask_by_pid[pid]).get_fdata().astype(np.int64)

        n = min(fake_vol.shape[0], mask_vol.shape[0])
        if fake_vol.shape[0] != mask_vol.shape[0]:
            log.warning(f"  {pid}: fake has {fake_vol.shape[0]} slices, "
                        f"mask has {mask_vol.shape[0]} -- using min={n}")
        fake_vol = fake_vol[:n]

        onehot = np.stack([one_hot_collapse(mask_vol[i]) for i in range(n)], axis=0)
        # onehot: (n, 3, H, W)

        img_slabs = slab_average(fake_vol, args.window)              # (n_slabs, H, W)
        mask_slabs = slab_average(
            onehot.reshape(n, -1), args.window
        ).reshape((-1,) + onehot.shape[1:])                          # (n_slabs, 3, H, W)

        img_dir = os.path.join(img_out_root, pid)
        mdir = os.path.join(mask_out_root, pid)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mdir, exist_ok=True)

        n_saved = 0
        for k in range(img_slabs.shape[0]):
            men_fraction_sum = float(mask_slabs[k, 1].sum() + mask_slabs[k, 2].sum())
            if men_fraction_sum < args.min_meniscus_fraction_sum:
                total_skipped += 1
                continue
            np.save(os.path.join(img_dir, f"{pid}_slice_{k:03d}.npy"),
                    img_slabs[k].astype(np.float32))
            np.save(os.path.join(mdir, f"{pid}_slice_{k:03d}.npy"),
                    mask_slabs[k].astype(np.float32))
            n_saved += 1

        log.info(f"  {pid}: {n_saved}/{img_slabs.shape[0]} slabs kept "
                 f"(from {n} source slices, window={args.window})")
        total_slabs += n_saved

    log.info(f"\nDone. {args.split}: {total_slabs} slabs saved, "
             f"{total_skipped} below-threshold slabs skipped.")
    log.info(f"Images -> {img_out_root}")
    log.info(f"Masks  -> {mask_out_root}")


if __name__ == "__main__":
    main()
