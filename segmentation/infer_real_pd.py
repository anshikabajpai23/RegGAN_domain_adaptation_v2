"""
infer_real_pd.py
==================
Runs the fine-tuned meniscus segmentation model directly on REAL PD volumes
(no RegGAN translation involved -- these are already real PD, that step is
only for DESS->pseudo-PD). For each input volume:
  1. Preprocess the same way training data was (process_volume("PD") --
     reorient RAS, isotropic in-plane resample, resize to 384x384, normalize)
  2. Run the 2.5D [i-1,i,i+1] stacked fine-tuned model on every slice
  3. Reconstruct the predicted per-slice class maps into a full volume
     (no skip-gaps here -- process_volume never skips real-PD slices)
  4. Save as NIfTI with a correct (LPS->RAS-fixed) affine, so it can be
     loaded in Slicer directly alongside the original real PD volume and
     your own manual labels for visual comparison.

Does not touch any existing pipeline file.

Usage:
    python segmentation/infer_real_pd.py \
        --pd_root /N/project/prostate_cancer_ai/anshika/regGAN/data/iu-dataset/pd-files \
        --filenames AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz AC0D7BF72F7712_SAG_PD_TSE_6.nii.gz ... \
        --ckpt /N/project/prostate_cancer_ai/anshika/regGAN/segmentation_runs/run_001/ckpt_best.pth \
        --out_dir /N/project/prostate_cancer_ai/anshika/regGAN/results/real_pd_predictions
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
from preprocess import process_volume  # reuses the SAME preprocessing as training

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3  # background, lateral meniscus, medial meniscus


def build_model(ckpt_path, device):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                      in_channels=3, classes=N_CLASSES)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def get_effective_affine_for_pd(nifti_path):
    """
    Same logic as infer2.py's get_effective_affine(), including the
    LPS->RAS fix, applied here to a REAL PD source file instead of DESS.
    Duplicated rather than imported to avoid coupling this new script to
    infer2.py's CLI-oriented module structure.
    """
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


def predict_volume(vol, model, device, batch_size=8):
    """vol: (n_slices, 384, 384) float32 in [0,1]. Returns (n_slices, 384, 384) int64 class map."""
    n = vol.shape[0]
    preds = np.zeros((n, vol.shape[1], vol.shape[2]), dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_stacks = []
            for idx in range(start, end):
                stack = []
                for offset in (-1, 0, 1):
                    j = max(0, min(n - 1, idx + offset))  # clamp at volume boundary
                    stack.append(vol[j])
                batch_stacks.append(np.stack(stack, axis=0))
            x = torch.from_numpy(np.stack(batch_stacks, axis=0)).float().to(device)
            out = model(x)
            preds[start:end] = out.argmax(dim=1).cpu().numpy()

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd_root", required=True)
    ap.add_argument("--filenames", nargs="+", required=True,
                     help="Real PD .nii.gz filenames (relative to --pd_root)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    model = build_model(args.ckpt, device)
    log.info(f"Loaded fine-tuned model from {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in args.filenames:
        path = os.path.join(args.pd_root, fname)
        if not os.path.exists(path):
            log.warning(f"  SKIPPED (not found): {path}")
            continue

        log.info(f"Processing {fname} ...")
        vol = process_volume(path, "PD")  # (n_slices, 384, 384), [0,1]
        preds = predict_volume(vol, model, device, args.batch_size)

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
