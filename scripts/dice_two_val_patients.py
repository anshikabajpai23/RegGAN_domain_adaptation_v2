"""
dice_two_val_patients.py
==========================
Phase 4 step 10 — Dice against GT for the baseline/avg/tta/avgtta variants
on the 2 real-val patients (AC0CE315D5758B, AC0CEE9C24F2B7).

GT loading reuses the exact convention already validated elsewhere in this
project (compare_stack_variants.py's load_nrrd_binary): SimpleITK read ->
DICOMOrient RAS -> resample in-plane to isotropic -> resize to 384x384
(order=0) -> threshold > 0.5.

"baseline" reads the already-existing run_v7_replica_seed42/val_predictions/
NIfTIs (no re-inference needed, already have them); avg/tta/avgtta read
their own results/real_pd_predictions_*_2pt/ directories produced by
eval_avg_tta_2pt.sh.

Usage:
  python scripts/dice_two_val_patients.py
"""
import glob
import os

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom

PATIENTS = ["AC0CE315D5758B", "AC0CEE9C24F2B7"]
GT_DIR = "labelled-pd-segmentations"

VARIANTS = {
    "baseline": ("segmentation_runs/run_v7_replica_seed42/val_predictions", "dump"),
    "avg":      ("results/real_pd_predictions_avg_2pt", "infer"),
    "tta":      ("results/real_pd_predictions_tta_2pt", "infer"),
    "avgtta":   ("results/real_pd_predictions_avgtta_2pt", "infer"),
}


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
    sp  = out.GetSpacing()
    arr = sitk.GetArrayFromImage(out).astype(np.float32).transpose(2, 1, 0)
    fa  = float(sp[1]) / min(sp[1], sp[2])
    fs  = float(sp[2]) / min(sp[1], sp[2])
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


def load_pred(variant_dir, kind, pid):
    if kind == "dump":
        path = os.path.join(variant_dir, f"{pid}_pred.nii.gz")
        if not os.path.exists(path):
            return None
        return np.asarray(nib.load(path).get_fdata()) > 0
    else:
        hits = glob.glob(os.path.join(variant_dir, f"{pid}*_meniscus_pred.nii.gz"))
        if not hits:
            return None
        return np.asarray(nib.load(hits[0]).get_fdata()) > 0


def main():
    gt_masks = {}
    for pid in PATIENTS:
        hits = glob.glob(os.path.join(GT_DIR, f"{pid}*.nrrd"))
        if not hits:
            print(f"WARNING: no GT found for {pid} in {GT_DIR}")
            continue
        gt_masks[pid] = load_nrrd_binary(hits[0])

    print(f"{'variant':<10} " + " ".join(f"{p:<18}" for p in PATIENTS) + " mean")
    for name, (vdir, kind) in VARIANTS.items():
        if not os.path.isdir(vdir):
            print(f"{name:<10} (missing: {vdir} not pulled down yet)")
            continue
        dices = []
        row = f"{name:<10} "
        for pid in PATIENTS:
            pred = load_pred(vdir, kind, pid)
            gt = gt_masks.get(pid)
            if pred is None or gt is None:
                row += f"{'?':<18} "
                continue
            d = dice(pred, gt)
            dices.append(d)
            row += f"{d:<18.4f} "
        if dices:
            row += f"{np.mean(dices):.4f}"
        print(row)


if __name__ == "__main__":
    main()
