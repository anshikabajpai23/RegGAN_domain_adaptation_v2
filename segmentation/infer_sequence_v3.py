"""
infer_sequence_v3.py
=====================
Inference for the slice-sequence architecture (sequence_model.py). Cannot
reuse infer_real_pd_v3.py -- that builds a standard smp.Unet expecting a
fixed 3 or 5-channel 2.5D stack; the sequence model is a different class
(SequenceUNet) taking (B, T, H, W) and internally running T separate encoder
passes + a ConvLSTM aggregation. Same volume preprocessing, affine logic,
and --tta flip-averaging as infer_real_pd_v3.py -- reused directly, not
reimplemented, so both scripts share the SAME verified LPS->RAS affine and
percentile-normalisation path. Only the windowing (5 slices, not 3) and the
model call differ.

Usage:
    python segmentation/infer_sequence_v3.py \
        --pd_root   /N/.../pd-files \
        --filenames AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz ... \
        --ckpt      /N/.../segmentation_runs/run_v7_sequence/ckpt_best.pth \
        --out_dir   /N/.../results/real_pd_predictions_sequence_17pt
"""
import argparse
import logging
import os
import sys

import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import process_volume
from sequence_model import build_sequence_model
from infer_real_pd_v3 import get_effective_affine_for_pd   # reused, not reimplemented

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OFFSETS = (-2, -1, 0, 1, 2)   # matches dataset_sequence.py's OFFSETS exactly


def predict_volume(vol, model, device, batch_size=8, tta=False):
    """vol: (n_slices, 384, 384) float32 in [0,1]. Returns (n_slices, 384, 384) int64.
    Same edge-clamp convention as infer_real_pd_v3.py: max(0, min(n-1, idx+o))."""
    n = vol.shape[0]
    preds = np.zeros((n, vol.shape[1], vol.shape[2]), dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_stacks = []
            for idx in range(start, end):
                stack = [vol[max(0, min(n - 1, idx + o))] for o in OFFSETS]
                batch_stacks.append(np.stack(stack, axis=0))   # (5, H, W)
            x = torch.from_numpy(np.stack(batch_stacks, axis=0)).float().to(device)  # (B,5,H,W)

            if not tta:
                out = model(x)
                preds[start:end] = out.argmax(dim=1).cpu().numpy()
            else:
                variants = [(), (-1,), (-2,), (-2, -1)]
                prob_sum = None
                for dims in variants:
                    xv = torch.flip(x, dims) if dims else x
                    p = torch.softmax(model(xv), dim=1)
                    p = torch.flip(p, dims) if dims else p
                    prob_sum = p if prob_sum is None else prob_sum + p
                preds[start:end] = (prob_sum / len(variants)).argmax(dim=1).cpu().numpy()

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd_root",    required=True)
    ap.add_argument("--filenames",  nargs="+", required=True)
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  offsets: {OFFSETS}  |  tta: {args.tta}")

    model = build_sequence_model(args.ckpt, device)
    model.eval()
    log.info(f"Loaded sequence checkpoint from {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in args.filenames:
        path = os.path.join(args.pd_root, fname)
        if not os.path.exists(path):
            log.warning(f"  SKIPPED (not found): {path}")
            continue

        log.info(f"Processing {fname} ...")
        vol = process_volume(path, "PD")
        preds = predict_volume(vol, model, device, args.batch_size, tta=args.tta)

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
