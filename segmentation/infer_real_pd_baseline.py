"""
infer_real_pd_baseline.py
============================
Runs the ORIGINAL pitthexai/Knee_MRI_Segmentation_2.5D checkpoint
(baseline_best_model.pth, 5 classes, NO fine-tuning) on real PD volumes --
for a direct before/after comparison against infer_real_pd.py's fine-tuned
3-class (background/lateral/medial meniscus) output.

NOTE: the original 5-class scheme's exact label semantics aren't documented
in what's available from the repo (just "5 segmentation categories" -- their
own cartilage/meniscus subdivision). This script outputs the full 5-class
argmax prediction; visually compare against the fine-tuned 3-class output
and your manual labels in Slicer to see which original class(es), if any,
correspond to meniscus, and whether fine-tuning improved on it.

Does not touch any existing pipeline file, including infer_real_pd.py.

Usage:
    python segmentation/infer_real_pd_baseline.py \
        --pd_root /N/project/prostate_cancer_ai/anshika/regGAN/data/iu-dataset/pd-files \
        --filenames AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz ... \
        --pretrained_ckpt /N/project/prostate_cancer_ai/anshika/regGAN/pretrained/baseline_best_model.pth \
        --out_dir /N/project/prostate_cancer_ai/anshika/regGAN/results/real_pd_predictions_baseline
"""
import argparse
import logging
import os
import sys

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch

import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import process_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ORIGINAL_N_CLASSES = 5  # the reference repo's original, un-fine-tuned scheme


def build_baseline_model(ckpt_path, device):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                      in_channels=3, classes=ORIGINAL_N_CLASSES)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)  # loads cleanly -- no head swap, matches checkpoint exactly
    model.eval()
    return model.to(device)


def get_effective_affine_for_pd(nifti_path):
    """Identical to infer_real_pd.py's version (including the LPS->RAS fix)
    -- duplicated rather than imported to keep these two comparison scripts
    independent of each other."""
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
    n = vol.shape[0]
    preds = np.zeros((n, vol.shape[1], vol.shape[2]), dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_stacks = []
            for idx in range(start, end):
                stack = []
                for offset in (-1, 0, 1):
                    j = max(0, min(n - 1, idx + offset))
                    stack.append(vol[j])
                batch_stacks.append(np.stack(stack, axis=0))
            x = torch.from_numpy(np.stack(batch_stacks, axis=0)).float().to(device)
            out = model(x)
            preds[start:end] = out.argmax(dim=1).cpu().numpy()

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd_root", required=True)
    ap.add_argument("--filenames", nargs="+", required=True)
    ap.add_argument("--pretrained_ckpt", required=True,
                     help="Original baseline_best_model.pth, NOT fine-tuned")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    model = build_baseline_model(args.pretrained_ckpt, device)
    log.info(f"Loaded ORIGINAL (non-fine-tuned) checkpoint from {args.pretrained_ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in args.filenames:
        path = os.path.join(args.pd_root, fname)
        if not os.path.exists(path):
            log.warning(f"  SKIPPED (not found): {path}")
            continue

        log.info(f"Processing {fname} ...")
        vol = process_volume(path, "PD")
        preds = predict_volume(vol, model, device, args.batch_size)

        log.info(f"  {fname}: {vol.shape[0]} slices, "
                 f"labels found: {sorted(np.unique(preds).tolist())}")
        for lbl in sorted(np.unique(preds).tolist()):
            if lbl == 0:
                continue
            count = int((preds == lbl).sum())
            n_slices_with_lbl = int((preds == lbl).any(axis=(1, 2)).sum())
            log.info(f"    class {lbl}: {count} px total, present in {n_slices_with_lbl} slices")

        affine, (sp_R, sp_A, sp_S) = get_effective_affine_for_pd(path)
        out_img = nib.Nifti1Image(preds.astype(np.int16), affine)
        out_img.header.set_zooms((sp_R, sp_A, sp_S))
        out_img.header.set_data_dtype(np.int16)

        stem = fname.replace(".nii.gz", "").replace(".nii", "")
        out_path = os.path.join(args.out_dir, f"{stem}_baseline_pred.nii.gz")
        nib.save(out_img, out_path)
        log.info(f"  Saved -> {out_path}")

    log.info("Done. Compare these 5-class outputs against infer_real_pd.py's "
             "3-class fine-tuned output and your manual labels in Slicer.")


if __name__ == "__main__":
    main()
