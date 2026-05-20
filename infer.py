"""
infer.py
========
Apply trained G_AB (DESS -> PD) to a folder of DESS NIfTI files.
Outputs translated PD-like NIfTI files.

Usage:
    python infer.py \
        --ckpt runs/reggan_001/ckpt_best.pt \
        --dess_root data/skm-tea-dataset/nifti \
        --out_dir results/translated_pd
"""

import os
import glob
import argparse
import logging
import numpy as np
import nibabel as nib
import torch
from pathlib import Path

from models import Generator
from preprocess import process_volume, TARGET_SIZE, SLICE_AXIS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def translate_volume(vol_dess: np.ndarray,
                     G_AB: torch.nn.Module,
                     device: torch.device,
                     batch_size: int = 8) -> np.ndarray:
    """
    Translate all slices of a preprocessed DESS volume to PD-like.
    vol_dess: float32 (H, W, D) normalised [0,1]
    Returns: float32 (H, W, D) normalised [0,1]
    """
    H, W, D = vol_dess.shape
    translated = np.zeros_like(vol_dess)

    G_AB.eval()
    with torch.no_grad():
        for start in range(0, D, batch_size):
            end   = min(start + batch_size, D)
            slices = []
            for i in range(start, end):
                sl = np.take(vol_dess, i, axis=SLICE_AXIS)   # (H, W)
                slices.append(sl)

            batch = np.stack(slices, 0)[:, None, :, :]       # (B, 1, H, W)
            t = torch.from_numpy(batch).to(device)
            t = t * 2.0 - 1.0                                # [0,1] -> [-1,1]
            out = G_AB(t)
            out = (out + 1.0) / 2.0                          # -> [0,1]
            out = out.squeeze(1).cpu().numpy()                # (B, H, W)

            for j, i in enumerate(range(start, end)):
                sl_out = out[j]
                idx = [slice(None)] * 3
                idx[SLICE_AXIS] = i
                translated[tuple(idx)] = sl_out

    return translated


def run_inference(ckpt_path: str,
                  dess_root: str,
                  out_dir: str,
                  ngf: int = 64,
                  n_res: int = 9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Load generator
    G_AB = Generator(in_ch=1, out_ch=1, ngf=ngf, n_res=n_res).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G_AB.load_state_dict(ckpt["G_AB"])
    G_AB.eval()
    log.info(f"Loaded G_AB from {ckpt_path}")

    nifti_files = sorted(glob.glob(os.path.join(dess_root, "**", "*.nii.gz"), recursive=True) +
                         glob.glob(os.path.join(dess_root, "**", "*.nii"),    recursive=True))
    log.info(f"Found {len(nifti_files)} DESS volumes.")
    os.makedirs(out_dir, exist_ok=True)

    for vpath in nifti_files:
        stem = Path(vpath).stem.replace(".nii", "")
        log.info(f"Translating {stem} ...")

        # Preprocess
        vol = process_volume(vpath, "DESS")           # (H, W, D) in [0,1]

        # Translate
        vol_pd = translate_volume(vol, G_AB, device)  # (H, W, D) in [0,1]

        # Save as NIfTI (preserve original header / affine for reference)
        orig_img = nib.load(vpath)
        # Note: vol_pd is at TARGET_SPACING, so update header zooms
        new_img = nib.Nifti1Image(vol_pd, orig_img.affine, orig_img.header)
        new_img.header.set_zooms(
            [float(z) for z in orig_img.header.get_zooms()[:3]]
        )
        out_path = os.path.join(out_dir, f"{stem}_pd_translated.nii.gz")
        nib.save(new_img, out_path)
        log.info(f"  Saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      required=True)
    ap.add_argument("--dess_root", default="data/skm-tea-dataset/nifti")
    ap.add_argument("--out_dir",   default="results/translated_pd")
    ap.add_argument("--ngf",       type=int, default=64)
    ap.add_argument("--n_res",     type=int, default=9)
    args = ap.parse_args()

    run_inference(args.ckpt, args.dess_root, args.out_dir, args.ngf, args.n_res)
