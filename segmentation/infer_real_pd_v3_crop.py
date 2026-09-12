"""
infer_real_pd_v3_crop.py
=========================
Crop-aware inference for the fixed-crop-region approach (candidate approach
"Localize/crop", P5). Loads the SAME box that compute_and_apply_fixed_crop.py
saved to --box_json, so train-time and inference-time cropping are
guaranteed identical -- a mismatch here would silently break the whole
approach (verified geometrically: see the crop-then-upsample-predict-
downsample-place round-trip test run before this script was written).

Per-slice pipeline (everything else — preprocessing, edge clamping, affine —
identical to infer_real_pd_v3.py):
  1. crop each of the 3 stack slices to the box
  2. resize each crop UP to 384x384 (order=3, matching training's image resize)
  3. run the model on the cropped+upsampled stack
  4. resize the prediction DOWN to the crop's original size (order=0, nearest,
     to keep integer class labels exact)
  5. place it into a full 384x384 canvas at the box location; everywhere
     outside the box is background (0) -- this is the fixed-crop approach's
     known limitation: meniscus tissue outside the training-derived box
     cannot be predicted at all. Worth checking n_meniscus counts against
     the non-cropped baseline for any patient where this might bite.

Usage:
    python segmentation/infer_real_pd_v3_crop.py \
        --pd_root   /N/.../pd-files \
        --filenames AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz ... \
        --ckpt      /N/.../segmentation_runs/run_v7_crop/ckpt_best.pth \
        --box_json  /N/.../crop_box.json \
        --out_dir   /N/.../results/real_pd_predictions_crop_17pt
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from scipy.ndimage import zoom

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


def crop_up(img2d, box, target_size=(384, 384)):
    y0, y1, x0, x1 = box
    crop = img2d[y0:y1, x0:x1]
    scale = (target_size[0] / crop.shape[0], target_size[1] / crop.shape[1])
    return zoom(crop, scale, order=3)


def predict_volume_crop(vol, model, device, box, batch_size=8):
    """vol: (n_slices, 384, 384) float32 in [0,1]. Returns (n_slices, 384, 384)
    int64, predictions placed back at the box location, zero elsewhere."""
    n = vol.shape[0]
    H, W = vol.shape[1], vol.shape[2]
    preds = np.zeros((n, H, W), dtype=np.int64)
    offsets = (-1, 0, 1)   # real-PD branch never used a stride override in any Phase 4 run
    y0, y1, x0, x1 = box
    crop_h, crop_w = y1 - y0, x1 - x0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_stacks = []
            for idx in range(start, end):
                stack = [crop_up(vol[max(0, min(n - 1, idx + o))], box) for o in offsets]
                batch_stacks.append(np.stack(stack, axis=0))
            x = torch.from_numpy(np.stack(batch_stacks, axis=0)).float().to(device)
            out = model(x).argmax(dim=1).cpu().numpy()   # (batch, 384, 384) in crop space

            for b, idx in enumerate(range(start, end)):
                pred_crop_size = zoom(out[b].astype(np.float32), (crop_h / 384, crop_w / 384),
                                      order=0).astype(np.int64)
                preds[idx, y0:y1, x0:x1] = pred_crop_size

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd_root",    required=True)
    ap.add_argument("--filenames",  nargs="+", required=True)
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--box_json",   required=True,
                    help="JSON saved by compute_and_apply_fixed_crop.py — MUST be the "
                         "exact box the checkpoint was trained on")
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--encoder",    default="resnet34", choices=["resnet34", "resnet50"])
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    with open(args.box_json) as fh:
        box_data = json.load(fh)
    box = (box_data["y0"], box_data["y1"], box_data["x0"], box_data["x1"])
    log.info(f"Loaded crop box from {args.box_json}: {box}  "
             f"(size {box[1]-box[0]}x{box[3]-box[2]}, padding={box_data.get('padding')})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  encoder: {args.encoder}  |  in_channels: 3 (fixed for crop)")

    model = build_model(args.ckpt, args.encoder, 3, device)
    log.info(f"Loaded checkpoint from {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in args.filenames:
        path = os.path.join(args.pd_root, fname)
        if not os.path.exists(path):
            log.warning(f"  SKIPPED (not found): {path}")
            continue

        log.info(f"Processing {fname} ...")
        vol = process_volume(path, "PD")
        preds = predict_volume_crop(vol, model, device, box, args.batch_size)

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
