"""
infer_real_pd_v3.py
====================
Inference script for all v3 ablation experiments.
Supports --encoder (resnet34/resnet50) and --in_channels (3 or 5).

Usage:
    python segmentation/infer_real_pd_v3.py \
        --pd_root   /N/.../pd-files \
        --filenames AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz ... \
        --ckpt      /N/.../segmentation_runs/run_v3_rotation/ckpt_best.pth \
        --out_dir   /N/.../results/real_pd_predictions_v3_rotation \
        --encoder   resnet34 \
        --in_channels 3
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
from preprocess import process_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3


def build_model(ckpt_path, encoder, in_channels, device):
    model = smp.Unet(encoder_name=encoder, encoder_weights=None,
                     in_channels=in_channels, classes=N_CLASSES)
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


def predict_volume(vol, model, device, in_channels, batch_size=8):
    """vol: (n_slices, 384, 384) float32 in [0,1]. Returns (n_slices, 384, 384) int64."""
    n = vol.shape[0]
    preds = np.zeros((n, vol.shape[1], vol.shape[2]), dtype=np.int64)

    if in_channels == 5:
        offsets = (-2, -1, 0, 1, 2)
    else:
        offsets = (-1, 0, 1)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_stacks = []
            for idx in range(start, end):
                stack = [vol[max(0, min(n - 1, idx + o))] for o in offsets]
                batch_stacks.append(np.stack(stack, axis=0))
            x = torch.from_numpy(np.stack(batch_stacks, axis=0)).float().to(device)
            out = model(x)
            preds[start:end] = out.argmax(dim=1).cpu().numpy()

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd_root",    required=True)
    ap.add_argument("--filenames",  nargs="+", required=True)
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--encoder",    default="resnet34", choices=["resnet34", "resnet50"])
    ap.add_argument("--in_channels", type=int, default=3, choices=[3, 5])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--clip_percentile", type=float, default=None,
                    help="If set, clip each slice to this percentile before inference (e.g. 95). "
                         "Suppresses hyperintense tear fluid signal. Default: no clipping.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  encoder: {args.encoder}  |  in_channels: {args.in_channels}  |  "
             f"clip_percentile: {args.clip_percentile}")

    model = build_model(args.ckpt, args.encoder, args.in_channels, device)
    log.info(f"Loaded checkpoint from {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in args.filenames:
        path = os.path.join(args.pd_root, fname)
        if not os.path.exists(path):
            log.warning(f"  SKIPPED (not found): {path}")
            continue

        log.info(f"Processing {fname} ...")
        vol = process_volume(path, "PD")
        if args.clip_percentile is not None:
            threshold = np.percentile(vol, args.clip_percentile)
            vol = np.clip(vol, 0, threshold)
            vol = vol / (threshold + 1e-8)  # renormalise to [0,1]
        preds = predict_volume(vol, model, device, args.in_channels, args.batch_size)

        n_meniscus = int((preds > 0).any(axis=(1, 2)).sum())
        log.info(f"  {fname}: {vol.shape[0]} slices, {n_meniscus} with meniscus, "
                 f"labels: {sorted(np.unique(preds).tolist())}")

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
