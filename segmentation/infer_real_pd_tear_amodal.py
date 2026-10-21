"""
infer_real_pd_tear_amodal.py
============================
Inference for v14 tear-amodal checkpoints (finetune_meniscus_v14_tear_amodal.py).

Same preprocessing (process_volume), same 2.5D stack (-1, 0, 1) with edge
clamping, and same output affine (get_effective_affine_for_pd, imported from
infer_real_pd_v3.py) as the replica's inference, so outputs line up with every
earlier prediction folder and with the torn notebook.

Writes per scan:
  <stem>_meniscus_pred.nii.gz   whole-wedge head, argmax (0 bg / 1 lat / 2 med)
                                -> same file name and format as all earlier runs
  <stem>_visible.nii.gz         visible-meniscus head > 0.5 (only if the ckpt has it)
  <stem>_tearmap.nii.gz         float32, clip(p_whole_meniscus - p_visible, 0, 1)
                                (only if the ckpt has a visible head)
Which heads exist is read from the checkpoint; no training flags are needed.

Usage:
    python segmentation/infer_real_pd_tear_amodal.py \
        --pd_root   /N/.../data/iu-torn/pd-files \
        --filenames AC13D01F99D88C_SAG_PD_TSE_5.nii.gz ... \
        --ckpt      /N/.../segmentation_runs/run_v14_tear_R1_seed42/ckpt_best.pth \
        --out_dir   /N/.../results/torn_predictions_v14_R1
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
from infer_real_pd_v3 import get_effective_affine_for_pd      # reused, not reimplemented
from model_tear_amodal import load_trained_tear_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OFFSETS = (-1, 0, 1)   # replica's 2.5D stack


@torch.no_grad()
def predict_volume(vol, model, device, batch_size=8):
    """vol: (n, 384, 384) float32 in [0, 1].
    Returns whole (n,384,384) int64, p_men (n,384,384) float32, p_vis or None."""
    n = vol.shape[0]
    whole = np.zeros(vol.shape, dtype=np.int64)
    p_men = np.zeros(vol.shape, dtype=np.float32)
    p_vis = None
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = np.stack([np.stack([vol[max(0, min(n - 1, i + o))] for o in OFFSETS], 0)
                      for i in range(start, end)], 0)
        out = model(torch.from_numpy(x).float().to(device))
        prob = torch.softmax(out["whole"], dim=1)
        whole[start:end] = prob.argmax(1).cpu().numpy()
        p_men[start:end] = (prob[:, 1] + prob[:, 2]).cpu().numpy()
        if "visible" in out:
            if p_vis is None:
                p_vis = np.zeros(vol.shape, dtype=np.float32)
            p_vis[start:end] = torch.sigmoid(out["visible"][:, 0]).cpu().numpy()
    return whole, p_men, p_vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd_root", required=True)
    ap.add_argument("--filenames", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_tear_model(args.ckpt, device)
    heads = ["whole"] + [h for h in ("visible", "contour") if getattr(model, f"{h}_head") is not None]
    log.info(f"Device: {device} | loaded {args.ckpt} | heads: {heads}")
    os.makedirs(args.out_dir, exist_ok=True)

    for fname in args.filenames:
        path = os.path.join(args.pd_root, fname)
        if not os.path.exists(path):
            log.warning(f"  SKIPPED (not found): {path}")
            continue
        vol = process_volume(path, "PD")
        whole, p_men, p_vis = predict_volume(vol, model, device, args.batch_size)
        affine, zooms = get_effective_affine_for_pd(path)
        stem = fname.replace(".nii.gz", "").replace(".nii", "")

        def save(arr, suffix, dtype):
            img = nib.Nifti1Image(arr.astype(dtype), affine)
            img.header.set_zooms(zooms)
            img.header.set_data_dtype(dtype)
            nib.save(img, os.path.join(args.out_dir, f"{stem}_{suffix}.nii.gz"))

        save(whole, "meniscus_pred", np.int16)
        msg = f"  {fname}: {vol.shape[0]} slices, {int((whole > 0).any(axis=(1, 2)).sum())} with meniscus"
        if p_vis is not None:
            save(p_vis > 0.5, "visible", np.int16)
            tear = np.clip(p_men - p_vis, 0.0, 1.0)
            save(tear, "tearmap", np.float32)
            msg += f", tearmap>0.5 voxels={int((tear > 0.5).sum())}"
        log.info(msg)
    log.info("Done.")


if __name__ == "__main__":
    main()
