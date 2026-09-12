"""
compute_and_apply_fixed_crop.py
================================
Candidate approach "Localize/crop" (P5) — first concrete implementation.
Fixed-crop-region variant (not two-stage): one bounding box, computed once
from training GT, applied identically to every patient/slice at train time.
The SAME box is reused at inference time by infer_real_pd_v3_crop.py.

Mechanism: crop tightly around where the meniscus ever appears in training
data, then RESIZE that crop back up to 384x384. The network's input size is
unchanged, but the meniscus now occupies far more pixels within it —
verified on synthetic data: a 6px-thick structure in a 140x140 crop becomes
17px after upsampling to 384x384, ~3x more pixels-per-mm on the exact
structure that's been getting fragmented (Failure 2, "in-plane split").

Step 1 — compute the box: union of meniscus-pixel extent (rows/cols where
ANY meniscus label appears) across ALL fake-PD train slices AND ALL real-PD
train slices, plus padding. TRAIN split only, never val, to avoid any
leakage into the box itself (this is an anatomical prior, not a learned
parameter, so the leakage risk is small, but there's no reason to take it).

Step 2 — apply the box: crop + resize (order=3/cubic for images, order=0/
nearest for masks, matching preprocess.py's own convention for images vs
label maps) every slice in BOTH train and val splits, both fake and real,
into new directories. segmentation_data_v2/ and real_pd_seg_data_v7/ are
never touched — same "new directory, nothing overwritten" pattern as
segmentation_data_v3/.

Output layout mirrors the originals exactly (same filenames, same directory
structure) so finetune_meniscus_v7_replica.py can be pointed at the new
directories with ZERO code changes — the crop is entirely a data-prep
transformation.

The box itself is saved as JSON so infer_real_pd_v3_crop.py can load and
reuse the EXACT same box (mismatch here would silently break the roundtrip).

Usage:
  python scripts/compute_and_apply_fixed_crop.py \
      --fake_data_root  segmentation_data_v2 \
      --real_data_root  real_pd_seg_data_v7 \
      --out_fake        segmentation_data_v2_crop \
      --out_real        real_pd_seg_data_v7_crop \
      --box_json        crop_box.json \
      --padding         15
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.ndimage import zoom


def extent_from_mask(mask):
    """Rows/cols where mask > 0. Returns None if mask is entirely empty."""
    rows = np.where((mask > 0).any(axis=1))[0]
    cols = np.where((mask > 0).any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return rows.min(), rows.max(), cols.min(), cols.max()


def compute_box(fake_mask_dir, real_mask_dir, padding, img_shape=(384, 384)):
    H, W = img_shape
    y0, y1, x0, x1 = H, 0, W, 0
    n_scanned = 0

    fake_files = glob.glob(os.path.join(fake_mask_dir, "*", "*.npy"))
    for f in fake_files:
        m = np.load(f)
        ext = extent_from_mask(m)
        if ext:
            r0, r1, c0, c1 = ext
            y0, y1 = min(y0, r0), max(y1, r1)
            x0, x1 = min(x0, c0), max(x1, c1)
        n_scanned += 1

    real_files = glob.glob(os.path.join(real_mask_dir, "*.npy"))
    for f in real_files:
        m = np.load(f)
        ext = extent_from_mask(m)
        if ext:
            r0, r1, c0, c1 = ext
            y0, y1 = min(y0, r0), max(y1, r1)
            x0, x1 = min(x0, c0), max(x1, c1)
        n_scanned += 1

    print(f"Scanned {len(fake_files)} fake-PD train masks + {len(real_files)} "
         f"real-PD train masks = {n_scanned} files")
    print(f"Raw extent (no padding): rows [{y0},{y1}]  cols [{x0},{x1}]")

    y0 = max(0, y0 - padding); y1 = min(H, y1 + padding + 1)
    x0 = max(0, x0 - padding); x1 = min(W, x1 + padding + 1)
    print(f"Padded box (+{padding}px): rows [{y0},{y1})  cols [{x0},{x1})  "
         f"size {y1-y0}x{x1-x0}")
    return int(y0), int(y1), int(x0), int(x1)


def crop_resize(arr, box, target_size, order):
    y0, y1, x0, x1 = box
    crop = arr[y0:y1, x0:x1]
    scale = (target_size[0] / crop.shape[0], target_size[1] / crop.shape[1])
    return zoom(crop, scale, order=order)


def apply_to_dir(src_images, src_masks, dst_images, dst_masks, box,
                 img_dtype, mask_dtype, target_size=(384, 384), fake_layout=True):
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_masks, exist_ok=True)
    n = 0

    if fake_layout:
        img_files = glob.glob(os.path.join(src_images, "*", "*.npy"))
        for f in img_files:
            pid = os.path.basename(os.path.dirname(f))
            fname = os.path.basename(f)
            mpath = os.path.join(src_masks, pid, fname)
            if not os.path.exists(mpath):
                continue
            img = np.load(f).astype(np.float32)
            msk = np.load(mpath)

            img_c = crop_resize(img, box, target_size, order=3).astype(img_dtype)
            msk_c = crop_resize(msk.astype(np.float32), box, target_size, order=0).astype(mask_dtype)

            os.makedirs(os.path.join(dst_images, pid), exist_ok=True)
            os.makedirs(os.path.join(dst_masks, pid), exist_ok=True)
            np.save(os.path.join(dst_images, pid, fname), img_c)
            np.save(os.path.join(dst_masks, pid, fname), msk_c)
            n += 1
    else:
        img_files = glob.glob(os.path.join(src_images, "*.npy"))
        for f in img_files:
            fname = os.path.basename(f)
            mpath = os.path.join(src_masks, fname)
            if not os.path.exists(mpath):
                continue
            img = np.load(f).astype(np.float32)
            msk = np.load(mpath)

            img_c = crop_resize(img, box, target_size, order=3).astype(img_dtype)
            msk_c = crop_resize(msk.astype(np.float32), box, target_size, order=0).astype(mask_dtype)

            np.save(os.path.join(dst_images, fname), img_c)
            np.save(os.path.join(dst_masks, fname), msk_c)
            n += 1

    print(f"  {n} slices: {src_images} -> {dst_images}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--out_fake", required=True)
    ap.add_argument("--out_real", required=True)
    ap.add_argument("--box_json", required=True)
    ap.add_argument("--padding", type=int, default=15)
    args = ap.parse_args()

    print("=== Step 1: computing fixed crop box from TRAIN masks only ===")
    box = compute_box(
        os.path.join(args.fake_data_root, "train", "masks"),
        os.path.join(args.real_data_root, "train", "masks"),
        args.padding)

    with open(args.box_json, "w") as fh:
        json.dump({"y0": box[0], "y1": box[1], "x0": box[2], "x1": box[3],
                  "padding": args.padding, "target_size": [384, 384]}, fh, indent=2)
    print(f"Box saved -> {args.box_json}  (infer_real_pd_v3_crop.py must load this exact file)")

    print("\n=== Step 2: applying box to every split, both domains ===")
    total = 0
    for split in ("train", "val"):
        print(f"-- fake PD, {split} --")
        total += apply_to_dir(
            os.path.join(args.fake_data_root, split, "images"),
            os.path.join(args.fake_data_root, split, "masks"),
            os.path.join(args.out_fake, split, "images"),
            os.path.join(args.out_fake, split, "masks"),
            box, np.float32, np.int64, fake_layout=True)

        print(f"-- real PD, {split} --")
        total += apply_to_dir(
            os.path.join(args.real_data_root, split, "images"),
            os.path.join(args.real_data_root, split, "masks"),
            os.path.join(args.out_real, split, "images"),
            os.path.join(args.out_real, split, "masks"),
            box, np.float32, np.int64, fake_layout=False)

    print(f"\nDone. {total} slices total written.")
    print(f"Fake -> {args.out_fake}")
    print(f"Real -> {args.out_real}")
    print("finetune_meniscus_v7_replica.py can be pointed at these directories "
         "with NO code changes -- same filenames, same layout, same dtypes.")


if __name__ == "__main__":
    main()
