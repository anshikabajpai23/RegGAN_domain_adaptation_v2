"""
infer_sequence_v3_crop_pos.py
===============================
Inference for the Phase 5 E1 checkpoint (position channel + crop + sequence,
run_e1_pos_seed42). Identical crop-up/predict/downsize-place round trip to
infer_sequence_v3_crop.py, plus the position channel each of the 5 timesteps
needs.

Position, same formula as training (finetune_meniscus_v7_crop_seq_e1.py):
    pos = 2 * |idx/(n-1) - 0.5|,  0 at the volume centre (notch), 1 at outer edges
Neighbour indices are clamped to [0, n-1] before computing pos, same convention
as PosSequenceRealPDDataset (real PD pads out-of-range neighbours with the
clamped slice, not a zero fill).

NO LOOKUP TABLE NEEDED HERE, unlike training. volume_lengths.json existed only
because segmentation_data_v2's prep dropped meniscus-free slices, so file counts
understated fake-PD volume length. At inference we load the FULL real PD volume
directly via process_volume() -- n = vol.shape[0] IS the true length, already.

Usage:
    python segmentation/infer_sequence_v3_crop_pos.py \
        --pd_root   /N/.../pd-files \
        --filenames AC0D5A4D78B628_SAG_PD_TSE_6.nii.gz ... \
        --ckpt      /N/.../segmentation_runs/run_e1_pos_seed42/ckpt_best.pth \
        --box_json  /N/.../crop_box.json \
        --out_dir   /N/.../results/real_pd_predictions_e1_pos_17pt
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import process_volume
from sequence_model_pos import load_trained_sequence_pos_model
from infer_real_pd_v3 import get_effective_affine_for_pd   # reused, not reimplemented

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OFFSETS = (-2, -1, 0, 1, 2)


def crop_up(img2d, box, target_size=(384, 384)):
    y0, y1, x0, x1 = box
    crop = img2d[y0:y1, x0:x1]
    scale = (target_size[0] / crop.shape[0], target_size[1] / crop.shape[1])
    return zoom(crop, scale, order=3)


def pos_of(idx, n):
    """doc §1.2, same formula training used."""
    return 2.0 * abs(idx / max(n - 1, 1) - 0.5)


def predict_volume_crop_seq_pos(vol, model, device, box, batch_size=8, tta=False):
    """vol: (n_slices, 384, 384) float32 in [0,1]. Returns (n_slices, 384, 384)
    int64, predictions placed back at the box location, zero elsewhere."""
    n = vol.shape[0]
    H, W = vol.shape[1], vol.shape[2]
    preds = np.zeros((n, H, W), dtype=np.int64)
    y0, y1, x0, x1 = box
    crop_h, crop_w = y1 - y0, x1 - x0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_stacks, batch_pos = [], []
            for idx in range(start, end):
                eff = [max(0, min(n - 1, idx + o)) for o in OFFSETS]
                stack = [crop_up(vol[e], box) for e in eff]
                batch_stacks.append(np.stack(stack, axis=0))
                batch_pos.append([pos_of(e, n) for e in eff])
            x = torch.from_numpy(np.stack(batch_stacks, axis=0)).float().to(device)
            pos = torch.tensor(batch_pos, dtype=torch.float32, device=device)

            if not tta:
                out = model(x, pos).argmax(dim=1).cpu().numpy()
            else:
                variants = [(), (-1,), (-2,), (-2, -1)]
                prob_sum = None
                for dims in variants:
                    xv = torch.flip(x, dims) if dims else x
                    p = torch.softmax(model(xv, pos), dim=1)
                    p = torch.flip(p, dims) if dims else p
                    prob_sum = p if prob_sum is None else prob_sum + p
                out = (prob_sum / len(variants)).argmax(dim=1).cpu().numpy()

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
    ap.add_argument("--box_json",   required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    with open(args.box_json) as fh:
        box_data = json.load(fh)
    box = (box_data["y0"], box_data["y1"], box_data["x0"], box_data["x1"])
    log.info(f"Loaded crop box from {args.box_json}: {box}  "
             f"(size {box[1]-box[0]}x{box[3]-box[2]}, padding={box_data.get('padding')})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  |  offsets: {OFFSETS}  |  tta: {args.tta}")

    model = load_trained_sequence_pos_model(args.ckpt, device)
    model.eval()
    log.info(f"Loaded E1 (position-channel) checkpoint from {args.ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in args.filenames:
        path = os.path.join(args.pd_root, fname)
        if not os.path.exists(path):
            log.warning(f"  SKIPPED (not found): {path}")
            continue

        log.info(f"Processing {fname} ...")
        vol = process_volume(path, "PD")
        preds = predict_volume_crop_seq_pos(vol, model, device, box, args.batch_size, tta=args.tta)

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
