"""
infer_sequence_v18_tear.py
==========================
Inference for v18 (crop + sequence + tear map + pattern class). Same preprocessing, crop box,
5-slice stacks and affine as infer_sequence_v3_crop.py. Writes, per volume:
  <stem>_meniscus_pred.nii.gz     meniscus labels (same format as every earlier run)
  <stem>_tear_prob.nii.gz         tear-map probability 0..1 (float32), zero outside the crop box
  <stem>_slice_patterns.csv       per slice: meniscus_px, p_torn (= 1 - p(none)), top class, and
                                  the probability of every mode (inplace = sum of its variants)
and for the whole run:
  knee_summary.csv                per volume: max / mean p_torn over meniscus slices, number of
                                  slices called torn, the most frequent mode among them, the top slices

    python segmentation/infer_sequence_v18_tear.py --pd_root ... --filenames ... \
        --ckpt .../run_v18_.../ckpt_best.pth --box_json .../crop_box.json --out_dir ...
"""
import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter

import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "preprocess"))      # local layout
sys.path.insert(0, _ROOT)                                    # BigRed layout (same as infer_sequence_v3_crop.py)
from preprocess import process_volume
from infer_real_pd_v3 import get_effective_affine_for_pd
from infer_sequence_v3_crop import crop_up, OFFSETS
from sequence_model_tear import load_trained_sequence_tear_model
from dataset_2_5d_tear_patterns import PATTERN_CLASSES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
MODES_OUT = ["none", "inplace", "ragged", "lobe", "truncate", "absent", "flap", "bucket"]


@torch.no_grad()
def predict(vol, model, device, box, bs=8):
    n, H, W = vol.shape
    y0, y1, x0, x1 = box
    seg = np.zeros((n, H, W), np.int16)
    tear = np.zeros((n, H, W), np.float32)
    probs = np.zeros((n, len(PATTERN_CLASSES)), np.float32)
    for start in range(0, n, bs):
        idx = list(range(start, min(start + bs, n)))
        x = np.stack([np.stack([crop_up(vol[max(0, min(n - 1, i + o))], box) for o in OFFSETS]) for i in idx])
        out = model(torch.from_numpy(x).float().to(device))
        s = out["seg"].argmax(1).cpu().numpy()
        t = torch.sigmoid(out["tear"][:, 0]).cpu().numpy()
        probs[idx] = torch.softmax(out["cls"], 1).cpu().numpy()
        for b, i in enumerate(idx):
            seg[i, y0:y1, x0:x1] = zoom(s[b].astype(np.float32), ((y1 - y0) / 384, (x1 - x0) / 384), order=0)
            tear[i, y0:y1, x0:x1] = np.clip(zoom(t[b], ((y1 - y0) / 384, (x1 - x0) / 384), order=1), 0, 1)
    return seg, tear, probs


def mode_probs(p):
    out = {m: 0.0 for m in MODES_OUT}
    for c, v in zip(PATTERN_CLASSES, p):
        out["inplace" if c.startswith("inplace_") else c] += float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pd_root", required=True)
    ap.add_argument("--filenames", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--box_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--torn_threshold", type=float, default=0.5)
    a = ap.parse_args()
    b = json.load(open(a.box_json))
    box = (b["y0"], b["y1"], b["x0"], b["x1"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_sequence_tear_model(a.ckpt, device, len(PATTERN_CLASSES)).eval()
    os.makedirs(a.out_dir, exist_ok=True)
    summary = []
    for fname in a.filenames:
        path = os.path.join(a.pd_root, fname)
        if not os.path.exists(path):
            log.warning(f"SKIPPED (not found): {path}"); continue
        vol = process_volume(path, "PD")
        seg, tear, probs = predict(vol, model, device, box, a.batch_size)
        affine, zooms = get_effective_affine_for_pd(path)
        stem = fname.replace(".nii.gz", "").replace(".nii", "")
        for arr, name, dt in ((seg, "meniscus_pred", np.int16), (tear, "tear_prob", np.float32)):
            img = nib.Nifti1Image(arr.astype(dt), affine); img.header.set_zooms(zooms); img.header.set_data_dtype(dt)
            nib.save(img, os.path.join(a.out_dir, f"{stem}_{name}.nii.gz"))
        rows = []
        for s in range(vol.shape[0]):
            mp = mode_probs(probs[s])
            rows.append({"slice": s, "meniscus_px": int((seg[s] > 0).sum()), "p_torn": round(1 - mp["none"], 4),
                         "top_class": PATTERN_CLASSES[int(probs[s].argmax())], "top_prob": round(float(probs[s].max()), 4),
                         "tear_px": int((tear[s] > 0.5).sum()), **{f"p_{m}": round(v, 4) for m, v in mp.items()}})
        with open(os.path.join(a.out_dir, f"{stem}_slice_patterns.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        men = [r for r in rows if r["meniscus_px"] > 0]
        torn = [r for r in men if r["p_torn"] > a.torn_threshold]
        top_mode = Counter(max(MODES_OUT[1:], key=lambda m: r[f"p_{m}"]) for r in torn).most_common(1)
        summary.append({"file": fname, "meniscus_slices": len(men),
                        "max_p_torn": max((r["p_torn"] for r in men), default=0.0),
                        "mean_p_torn": round(float(np.mean([r["p_torn"] for r in men])), 4) if men else 0.0,
                        "slices_called_torn": len(torn), "most_frequent_mode": top_mode[0][0] if top_mode else "",
                        "top_slices": " ".join(str(r["slice"]) for r in sorted(men, key=lambda r: -r["p_torn"])[:3])})
        log.info(f"{fname}: {summary[-1]}")
    if summary:
        with open(os.path.join(a.out_dir, "knee_summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    log.info("Done.")


if __name__ == "__main__":
    main()
