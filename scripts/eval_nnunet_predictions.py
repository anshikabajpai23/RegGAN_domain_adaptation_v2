"""
eval_nnunet_predictions.py
============================
Baseline (nnU-Net): per-patient Dice for nnU-Net's predicted labels vs our
own GT, in the same "clean per-patient pass" format every other run in this
project reports (per-patient Dice + mean), so this is directly comparable to
run_v7_replica_seed42 etc.

Usage:
    python scripts/eval_nnunet_predictions.py \
        --pred_dir  /path/to/nnunet_predictions_fold0 \
        --gt_dir    /path/to/nnUNet_raw/Dataset501_MeniscusPD/labelsTr \
        --val_patients MTR_001 MTR_005   # from splits_final.json's "val" list
"""
import argparse
import os

import numpy as np
import nibabel as nib


def dice(pred, gt, eps=1e-6):
    p = (pred > 0).astype(np.float64)
    g = (gt > 0).astype(np.float64)
    tp = (p * g).sum()
    return (2 * tp + eps) / (p.sum() + g.sum() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="nnUNetv2_predict output dir")
    ap.add_argument("--gt_dir", required=True, help="labelsTr dir with {pid}.nii.gz")
    ap.add_argument("--val_patients", nargs="+", required=True)
    args = ap.parse_args()

    dices = {}
    for pid in args.val_patients:
        pred_path = os.path.join(args.pred_dir, f"{pid}.nii.gz")
        gt_path = os.path.join(args.gt_dir, f"{pid}.nii.gz")
        if not os.path.exists(pred_path):
            print(f"WARNING: missing prediction for {pid}: {pred_path}")
            continue
        pred = nib.load(pred_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()
        d = dice(pred, gt)
        dices[pid] = d
        print(f"  {pid}: Dice = {d:.4f}")

    if dices:
        mean_dice = float(np.mean(list(dices.values())))
        print(f"\nMean Dice (n={len(dices)} val patients): {mean_dice:.4f}")
        print("Compare vs run_v7_replica_seed42 (0.781) and run_imagenet_control.")
    else:
        print("No predictions found -- nothing to report.")


if __name__ == "__main__":
    main()
