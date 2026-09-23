"""
infer_real_pd_v3_film.py
==========================
Inference for the Phase 5 E2 checkpoint on the PLAIN 2.5D architecture
(run_2_5d_e2_seed42) -- FiLM gating, NO crop, NO sequence. Identical
predict/save round trip to infer_real_pd_v3.py, plus the single-scalar
position value each sample needs (model_2_5d_film.Unet2_5DFiLM.forward(x, pos)
takes ONE pos per sample -- the centre slice's own position -- unlike the
crop+sequence E2 model, which FiLM-gates the ConvLSTM hidden state once per timestep).

Position, same formula as training (finetune_meniscus_v7_replica_e2.py):
    pos = 2 * |idx/(n-1) - 0.5|,  0 at the volume centre (notch), 1 at outer edges

NO LOOKUP TABLE NEEDED, same reasoning as the crop+sequence E2 inference
script: volume_lengths.json existed only because the fake-PD prep step
dropped meniscus-free slices, understating true volume length. At inference
we load the FULL real PD volume directly via process_volume(), so
n = vol.shape[0] IS the true length already.

Usage:
    python segmentation/infer_real_pd_v3_film.py \
        --pd_root   /N/.../pd-files \
        --filenames AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz ... \
        --ckpt      /N/.../segmentation_runs/run_2_5d_e2_seed42/ckpt_best.pth \
        --out_dir   /N/.../results/real_pd_predictions_2_5d_e2_17pt
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
from model_2_5d_film import load_trained_2_5d_film_model
from infer_real_pd_v3 import get_effective_affine_for_pd   # reused, not reimplemented

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OFFSETS = (-1, 0, 1)


def pos_of(idx, n):
    return 2.0 * abs(idx / max(n - 1, 1) - 0.5)


def predict_volume_film(vol, model, device, batch_size=8, tta=False):
    """vol: (n_slices, 384, 384) float32 in [0,1]. Returns (n_slices, 384, 384) int64."""
    n = vol.shape[0]
    preds = np.zeros((n, vol.shape[1], vol.shape[2]), dtype=np.int64)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_stacks, batch_pos = [], []
            for idx in range(start, end):
                stack = [vol[max(0, min(n - 1, idx + o))] for o in OFFSETS]
                batch_stacks.append(np.stack(stack, axis=0))
                batch_pos.append(pos_of(idx, n))   # ONE pos per sample -- the centre slice
            x = torch.from_numpy(np.stack(batch_stacks, axis=0)).float().to(device)
            pos = torch.tensor(batch_pos, dtype=torch.float32, device=device)

            if not tta:
                out = model(x, pos)
                preds[start:end] = out.argmax(dim=1).cpu().numpy()
            else:
                variants = [(), (-1,), (-2,), (-2, -1)]
                prob_sum = None
                for dims in variants:
                    xv = torch.flip(x, dims) if dims else x
                    p = torch.softmax(model(xv, pos), dim=1)
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

    model = load_trained_2_5d_film_model(args.ckpt, device)
    model.eval()
    log.info(f"Loaded E2 (FiLM gating, plain 2.5D) checkpoint from {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in args.filenames:
        path = os.path.join(args.pd_root, fname)
        if not os.path.exists(path):
            log.warning(f"  SKIPPED (not found): {path}")
            continue

        log.info(f"Processing {fname} ...")
        vol = process_volume(path, "PD")
        preds = predict_volume_film(vol, model, device, args.batch_size, tta=args.tta)

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
