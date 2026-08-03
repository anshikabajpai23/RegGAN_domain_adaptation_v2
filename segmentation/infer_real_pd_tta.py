"""
infer_real_pd_tta.py
======================
Test-Time Augmentation (TTA) version of infer_real_pd.py.
Averages softmax probabilities over 3 passes:
  1. Original
  2. Horizontal flip
  3. Vertical flip
Then takes argmax of the averaged probabilities.

Expected improvement: +1-3% Dice on real PD by reducing prediction variance
at meniscus boundaries (ensemble effect without retraining).

Can be run on ANY existing fine-tuned checkpoint — drop-in replacement for
infer_real_pd.py. Just change --ckpt to whichever run you want to evaluate.

Usage:
    python segmentation/infer_real_pd_tta.py \
        --pd_root /N/project/prostate_cancer_ai/anshika/regGAN/data/iu-dataset/pd-files \
        --filenames AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz ... \
        --ckpt /N/project/prostate_cancer_ai/anshika/regGAN/segmentation_runs/run_006/ckpt_best.pth \
        --out_dir /N/project/prostate_cancer_ai/anshika/regGAN/results/real_pd_predictions_tta
"""
import argparse
import logging
import os
import sys

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import process_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3


def build_model(ckpt_path, device):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                      in_channels=3, classes=N_CLASSES)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def get_effective_affine_for_pd(nifti_path):
    img = sitk.ReadImage(nifti_path)
    img = sitk.DICOMOrient(img, "RAS")
    sp  = img.GetSpacing()
    arr = sitk.GetArrayFromImage(img)

    sp_R = float(sp[0])
    sp_A_orig, sp_S_orig = float(sp[1]), float(sp[2])
    n_A_orig, n_S_orig = arr.shape[1], arr.shape[0]

    target_ip = min(sp_A_orig, sp_S_orig)
    n_A_rs = round(n_A_orig * sp_A_orig / target_ip)
    n_S_rs = round(n_S_orig * sp_S_orig / target_ip)
    eff_sp_A = target_ip * n_A_rs / 384
    eff_sp_S = target_ip * n_S_rs / 384

    direction = np.array(img.GetDirection()).reshape(3, 3)
    origin    = np.array(img.GetOrigin())

    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    direction  = lps_to_ras @ direction
    origin     = lps_to_ras @ origin

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction @ np.diag([sp_R, eff_sp_A, eff_sp_S])
    affine[:3, 3]  = origin

    return affine.astype(np.float32), (sp_R, eff_sp_A, eff_sp_S)


def predict_batch_probs(model, batch_stacks, device):
    """Returns softmax probabilities: (B, N_CLASSES, H, W)."""
    x = torch.from_numpy(np.stack(batch_stacks, axis=0)).float().to(device)
    with torch.no_grad():
        logits = model(x)
        return torch.softmax(logits, dim=1).cpu().numpy()


def predict_volume_tta(vol, model, device, batch_size=8):
    """
    vol: (n_slices, H, W) float32 in [0,1]
    Returns: (n_slices, H, W) int64 class map using TTA.
    TTA passes: original | hflip | vflip → average softmax → argmax
    """
    n, H, W = vol.shape
    prob_sum = np.zeros((n, N_CLASSES, H, W), dtype=np.float32)

    for flip_mode in ("none", "hflip", "vflip"):
        if flip_mode == "hflip":
            vol_aug = vol[:, :, ::-1].copy()
        elif flip_mode == "vflip":
            vol_aug = vol[:, ::-1, :].copy()
        else:
            vol_aug = vol

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_stacks = []
            for idx in range(start, end):
                stack = []
                for offset in (-1, 0, 1):
                    j = max(0, min(n - 1, idx + offset))
                    stack.append(vol_aug[j])
                batch_stacks.append(np.stack(stack, axis=0))

            probs = predict_batch_probs(model, batch_stacks, device)  # (B, C, H, W)

            # un-flip predictions back to original orientation
            if flip_mode == "hflip":
                probs = probs[:, :, :, ::-1].copy()
            elif flip_mode == "vflip":
                probs = probs[:, :, ::-1, :].copy()

            prob_sum[start:end] += probs

    # average over 3 TTA passes then argmax
    preds = prob_sum.argmax(axis=1).astype(np.int64)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd_root", required=True)
    ap.add_argument("--filenames", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    log.info("TTA passes: original + hflip + vflip (softmax average)")

    model = build_model(args.ckpt, device)
    log.info(f"Loaded model from {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in args.filenames:
        path = os.path.join(args.pd_root, fname)
        if not os.path.exists(path):
            log.warning(f"  SKIPPED (not found): {path}")
            continue

        log.info(f"Processing {fname} ...")
        vol   = process_volume(path, "PD")  # (n_slices, 384, 384), [0,1]
        preds = predict_volume_tta(vol, model, device, args.batch_size)

        n_meniscus_slices = int((preds > 0).any(axis=(1, 2)).sum())
        log.info(f"  {fname}: {vol.shape[0]} slices, "
                 f"{n_meniscus_slices} with predicted meniscus, "
                 f"labels found: {sorted(np.unique(preds).tolist())}")

        affine, (sp_R, sp_A, sp_S) = get_effective_affine_for_pd(path)
        out_img = nib.Nifti1Image(preds.astype(np.int16), affine)
        out_img.header.set_zooms((sp_R, sp_A, sp_S))
        out_img.header.set_data_dtype(np.int16)

        stem = fname.replace(".nii.gz", "").replace(".nii", "")
        out_path = os.path.join(args.out_dir, f"{stem}_meniscus_pred.nii.gz")
        nib.save(out_img, out_path)
        log.info(f"  Saved -> {out_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
