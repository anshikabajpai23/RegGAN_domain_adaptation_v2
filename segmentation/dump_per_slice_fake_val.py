"""
dump_per_slice_fake_val.py
==========================
Runs a fine-tuned checkpoint over a Meniscus2_5DDataset directory and writes a
per-slice CSV in EXACTLY the same format as dump_val_predictions() in
finetune_meniscus_v7_replica.py:

    patient_id, slice_idx, dice, gt_voxels, pred_voxels, false_neg, false_pos

so scripts/build_hallucination_table.py consumes it unchanged.

Read-only: loads a checkpoint, trains nothing, saves no weights.

⚠️ Point this at a val split prepared with --min_meniscus_pixels 0.
   The standard segmentation_data_v2/val/ has every meniscus-free slice filtered
   out at prep time, so it contains ZERO empty-GT slices and cannot answer the
   empty-slice-FP question at all.

Pass --dump_volumes to also save, per patient, <pid>_image/gt/pred.nii.gz —
same 3-file convention as dump_val_predictions() in finetune_meniscus_v7_replica.py,
so both cohorts open in Slicer the same way.

Usage:
  python segmentation/dump_per_slice_fake_val.py \
      --ckpt      segmentation_runs/run_v7_replica_seed42/ckpt_best.pth \
      --img_root  segmentation_data_v3_valneg/val/images \
      --mask_root segmentation_data_v3_valneg/val/masks \
      --out_csv   fake_val_per_slice_dice.csv \
      --dump_volumes --vol_out_dir fake_val_predictions
"""
import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_2_5d_v2 import Meniscus2_5DDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_CLASSES = 3


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        required=True)
    ap.add_argument("--img_root",    required=True)
    ap.add_argument("--mask_root",   required=True)
    ap.add_argument("--out_csv",     required=True)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dump_volumes", action="store_true",
                    help="also save <pid>_image/gt/pred.nii.gz per patient")
    ap.add_argument("--vol_out_dir", default=None,
                    help="required if --dump_volumes is set")
    args = ap.parse_args()
    if args.dump_volumes and not args.vol_out_dir:
        ap.error("--dump_volumes requires --vol_out_dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=N_CLASSES)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device).eval()
    log.info(f"Loaded {args.ckpt}  |  device={device}  |  eval(), no_grad, nothing trained")

    ds = Meniscus2_5DDataset(args.img_root, args.mask_root, augment=False)
    pids = sorted({pid for pid, _ in ds.items})
    log.info(f"Dataset: {len(ds)} slices across {len(pids)} patients")

    loader = DataLoader(ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    eps = 1e-6
    rows = []
    n_empty_gt = 0
    vols = {}   # pid -> {slice_idx: (img_centre_channel, gt, pred)} — only if --dump_volumes

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"]
        pred = model(x).argmax(1).cpu()

        pm = (pred > 0)
        gm = (y > 0)

        for b in range(x.shape[0]):
            p, g = pm[b], gm[b]
            tp = float((p & g).sum())
            ps = float(p.sum())
            gs = float(g.sum())
            if gs == 0:
                n_empty_gt += 1
            pid  = batch["patient_id"][b]
            sidx = int(batch["slice_idx"][b])
            rows.append({
                "patient_id":  pid,
                "slice_idx":   sidx,
                "dice":        f"{(2*tp + eps) / (ps + gs + eps):.4f}",
                "gt_voxels":   int(gs),
                "pred_voxels": int(ps),
                "false_neg":   int(gs - tp),
                "false_pos":   int(ps - tp),
            })
            if args.dump_volumes:
                # channel 1 of the 2.5D stack [i-1, i, i+1] is the centre slice
                vols.setdefault(pid, {})[sidx] = (
                    x[b, 1].cpu().numpy().astype(np.float32),
                    y[b].numpy().astype(np.int16),
                    pred[b].numpy().astype(np.int16),
                )

    rows.sort(key=lambda r: (r["patient_id"], r["slice_idx"]))
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient_id", "slice_idx", "dice",
                                           "gt_voxels", "pred_voxels",
                                           "false_neg", "false_pos"])
        w.writeheader()
        w.writerows(rows)

    log.info(f"Wrote {len(rows)} slice rows -> {args.out_csv}")
    log.info(f"Empty-GT slices present: {n_empty_gt} / {len(rows)}")
    if n_empty_gt == 0:
        log.warning("ZERO empty-GT slices — this split was prepared with "
                    "min_meniscus_pixels > 0, so the empty-slice-FP question "
                    "CANNOT be answered from it. Re-prep with "
                    "--min_meniscus_pixels 0.")

    if args.dump_volumes:
        try:
            import nibabel as nib
        except ImportError:
            log.warning("nibabel not available — skipping volume dump")
            return

        os.makedirs(args.vol_out_dir, exist_ok=True)
        # NOTE: fake PD has no matching real acquisition (that's the whole point
        # of the translation) so there is no physical spacing to recover here.
        # Identity spacing keeps image/gt/pred exactly co-registered WITH EACH
        # OTHER (what you need to see where pred departs from GT) but is NOT
        # real-world mm — unlike the real-val dump, which uses verified PD spacing.
        affine = np.eye(4, dtype=np.float32)

        for pid, slices in sorted(vols.items()):
            order = sorted(slices.keys())
            img = np.stack([slices[s][0] for s in order], axis=0)
            gt  = np.stack([slices[s][1] for s in order], axis=0)
            pr  = np.stack([slices[s][2] for s in order], axis=0)

            nib.save(nib.Nifti1Image(img, affine), os.path.join(args.vol_out_dir, f"{pid}_image.nii.gz"))
            nib.save(nib.Nifti1Image((gt > 0).astype(np.int16), affine), os.path.join(args.vol_out_dir, f"{pid}_gt.nii.gz"))
            nib.save(nib.Nifti1Image((pr > 0).astype(np.int16), affine), os.path.join(args.vol_out_dir, f"{pid}_pred.nii.gz"))
            log.info(f"  dumped {pid}: {len(order)} slices -> {pid}_{{image,gt,pred}}.nii.gz")

        log.info(f"Volumes -> {args.vol_out_dir}")


if __name__ == "__main__":
    main()
