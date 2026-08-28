"""
postprocess_predictions.py
==========================
Experiment B: fill_holes + connected-component filter on existing predictions.

Loads each _meniscus_pred.nii.gz from --pred_dir, applies per-slice:
  1. remove_small_objects (min_size) — removes tiny fragment islands
  2. binary_fill_holes — fills gaps inside meniscus boundary (tear gaps)

Saves cleaned predictions to --out_dir. Preserves affine/header exactly.

Usage (local):
    python segmentation/postprocess_predictions.py \
        --pred_dir  results/final_results/real_pd_predictions_v8_mixed \
        --out_dir   results/real_pd_predictions_v8_filled \
        --min_size  50
"""
import argparse
import logging
import os

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_fill_holes
from skimage import morphology

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def clean_volume(pred_vol, min_size=50):
    """
    pred_vol: (n_slices, H, W) int array with labels 0, 1, 2
    Returns cleaned volume same shape/dtype.
    """
    cleaned = np.zeros_like(pred_vol)
    for label in [1, 2]:
        for s in range(pred_vol.shape[0]):
            binary = pred_vol[s] == label
            if not binary.any():
                continue
            # Step 1: remove small fragments
            binary = morphology.remove_small_objects(binary, min_size=min_size)
            # Step 2: fill holes (tear gaps read as holes inside meniscus boundary)
            binary = binary_fill_holes(binary)
            cleaned[s][binary] = label
    return cleaned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True,
                    help="Dir containing *_meniscus_pred.nii.gz files")
    ap.add_argument("--out_dir",  required=True,
                    help="Output dir for cleaned predictions")
    ap.add_argument("--min_size", type=int, default=50,
                    help="Min connected-component size to keep (pixels). Default: 50")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    fnames = sorted(f for f in os.listdir(args.pred_dir) if f.endswith(".nii.gz"))
    log.info(f"Found {len(fnames)} prediction files in {args.pred_dir}")
    log.info(f"min_size={args.min_size} | fill_holes=True")

    for fname in fnames:
        path = os.path.join(args.pred_dir, fname)
        img  = nib.load(path)
        vol  = img.get_fdata().astype(np.int16)

        # Handle (H, W, n_slices) layout if needed
        if vol.ndim == 3 and vol.shape[2] < vol.shape[0]:
            vol_in = vol.transpose(2, 0, 1)  # → (n_slices, H, W)
            transposed = True
        else:
            vol_in = vol
            transposed = False

        cleaned = clean_volume(vol_in, min_size=args.min_size)

        if transposed:
            cleaned = cleaned.transpose(1, 2, 0)

        n_before = int((vol > 0).sum())
        n_after  = int((cleaned > 0).sum())
        log.info(f"  {fname}: meniscus voxels {n_before} -> {n_after} "
                 f"(+{n_after - n_before} filled / -{max(0, n_before - n_after)} removed)")

        out_img = nib.Nifti1Image(cleaned, img.affine, img.header)
        out_path = os.path.join(args.out_dir, fname)
        nib.save(out_img, out_path)

    log.info(f"Done. Cleaned predictions saved to {args.out_dir}")


if __name__ == "__main__":
    main()
