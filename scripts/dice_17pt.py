"""
dice_17pt.py
=============
Generalizes dice_two_val_patients.py's GT-loading and Dice logic to the full
17-patient test set (labeled_patients_v8.txt), for any prediction directory
produced by infer_real_pd_v3.py's "{stem}_meniscus_pred.nii.gz" convention.

GT loading is IDENTICAL to dice_two_val_patients.py (SimpleITK read ->
DICOMOrient RAS -> resample in-plane to isotropic -> resize to 384x384
order=0 -> threshold > 0.5) -- not reimplemented differently, to avoid a
second, silently-different GT convention entering the project.

Usage:
    python scripts/dice_17pt.py --pred_dir results/real_pd_predictions_run_v13_slab_e50_17pt
    python scripts/dice_17pt.py --pred_dir results/real_pd_predictions_run_imagenet_control_lr1e-4_17pt
"""
import argparse
import glob
import os

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom

GT_DIR = "labelled-pd-segmentations"
PATIENT_FILE = "labeled_patients_v8.txt"


def load_nrrd_binary(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    arr = (arr > 0).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    try:
        out = sitk.DICOMOrient(out, "RAS")
    except Exception:
        pass
    sp = out.GetSpacing()
    arr = sitk.GetArrayFromImage(out).astype(np.float32).transpose(2, 1, 0)
    fa = float(sp[1]) / min(sp[1], sp[2])
    fs = float(sp[2]) / min(sp[1], sp[2])
    if abs(fa - 1.0) > 0.02 or abs(fs - 1.0) > 0.02:
        arr = zoom(arr, (1.0, fa, fs), order=0)
    _, n_A, n_S = arr.shape
    if n_A != 384 or n_S != 384:
        arr = zoom(arr, (1.0, 384 / n_A, 384 / n_S), order=0)
    return (arr > 0.5).astype(bool)


def dice(pred, gt, eps=1e-6):
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape} — "
                         f"refusing to silently resample")
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def load_pred(pred_dir, pid):
    hits = glob.glob(os.path.join(pred_dir, f"{pid}*_meniscus_pred.nii.gz"))
    if not hits:
        return None
    return np.asarray(nib.load(hits[0]).get_fdata()) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--patient_file", default=PATIENT_FILE)
    ap.add_argument("--gt_dir", default=GT_DIR)
    args = ap.parse_args()

    with open(args.patient_file) as f:
        patients = [l.strip() for l in f if l.strip()]

    if not os.path.isdir(args.pred_dir):
        print(f"ERROR: {args.pred_dir} does not exist"); return

    dices = {}
    for pid in patients:
        gt_hits = glob.glob(os.path.join(args.gt_dir, f"{pid}*.nrrd"))
        if not gt_hits:
            print(f"  {pid}: NO GT FOUND in {args.gt_dir} — skipping")
            continue
        gt = load_nrrd_binary(gt_hits[0])

        pred = load_pred(args.pred_dir, pid)
        if pred is None:
            print(f"  {pid}: NO PREDICTION FOUND in {args.pred_dir} — skipping")
            continue

        try:
            d = dice(pred, gt)
        except ValueError as e:
            print(f"  {pid}: {e}")
            continue
        dices[pid] = d
        print(f"  {pid}: Dice = {d:.4f}")

    if dices:
        vals = list(dices.values())
        print(f"\nn = {len(vals)} / {len(patients)} patients")
        print(f"Mean Dice: {np.mean(vals):.4f}")
        print(f"Std:       {np.std(vals):.4f}")
        print(f"Per-patient values: {[round(v,4) for v in vals]}")
    else:
        print("No patients scored.")


if __name__ == "__main__":
    main()
