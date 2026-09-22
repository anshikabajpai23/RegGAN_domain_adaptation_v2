"""
check_crop_box_coverage.py
============================
Methodological check on the fixed-crop-region approach (P5, crop/localize):
crop_box.json was computed from the 13 real-PD TRAIN patients' GT extent
only (+ fake-PD train). It was never verified against the 17 held-out TEST
patients' actual GT meniscus location. Anything outside the box is a
guaranteed miss at inference time (infer_real_pd_v3_crop.py /
infer_sequence_v3_crop.py place zeros everywhere outside the box) --
this measures exactly how much that costs, per patient, before trusting
the crop/crop+sequence 17pt Dice numbers as a fair read.

Same GT-loading convention as analysis_phase4_17patients.ipynb
(load_nrrd_binary, verbatim) so the box's coordinate space matches exactly
what the notebook's Dice computation uses.

Usage:
    python scripts/check_crop_box_coverage.py
"""
import glob
import json
import os
import re

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

BASE = "/Users/anshikabajpai/Desktop/github/RegGAN_domain_adaptation_v2"
GT_DIR_D1  = "/Users/anshikabajpai/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final"
GT_DIR_NEW = "/Users/anshikabajpai/Downloads/final-segmentations"
BOX_JSON   = f"{BASE}/crop_box.json"

D1_PATIENTS = [
    "AC0D5A4D78B628", "AC0D7BF72F7712", "AC0D3459553205", "AC14D3737C0482",
    "AC19E7C19827FF", "AC149BC218E75C", "AC111633B463BB", "AC13300201B926",
    "AC12026D14291F", "AC13637DA25399",
]
NEW_PATIENTS = [
    "AC000550763509", "AC005135D3495B", "AC04433E37DB66", "AC056ADCE8BE28",
    "AC07607B9E5295", "AC0D1A9818A6FC", "AC0F11041F5180",
]
ALL_PATIENTS = D1_PATIENTS + NEW_PATIENTS


def load_nrrd_binary(path):
    """Verbatim convention from analysis_phase4_17patients.ipynb."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 4:
        arr = (arr.max(axis=0) > 0).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        try:
            comp = sitk.VectorIndexSelectionCast(img, 0)
            out.CopyInformation(comp)
        except Exception:
            out.SetSpacing(img.GetSpacing()[:3]); out.SetOrigin(img.GetOrigin()[:3])
            out.SetDirection(img.GetDirection()[:9])
    else:
        arr = (arr > 0).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
    try:
        out = sitk.DICOMOrient(out, "RAS")
    except Exception:
        pass
    sp  = out.GetSpacing()
    arr = sitk.GetArrayFromImage(out).astype(np.float32).transpose(2, 1, 0)
    fa  = float(sp[1]) / min(sp[1], sp[2])
    fs  = float(sp[2]) / min(sp[1], sp[2])
    if abs(fa - 1.0) > 0.02 or abs(fs - 1.0) > 0.02:
        arr = zoom(arr, (1.0, fa, fs), order=0)
    _, n_A, n_S = arr.shape
    if n_A != 384 or n_S != 384:
        arr = zoom(arr, (1.0, 384 / n_A, 384 / n_S), order=0)
    return (arr > 0.5).astype(bool)


def get_pid(fname):
    m = re.match(r"(AC[A-F0-9]+)", os.path.basename(fname), re.IGNORECASE)
    return m.group(1) if m else None


def main():
    if not os.path.exists(BOX_JSON):
        print(f"ERROR: {BOX_JSON} not found -- scp it down from BigRed first "
              f"(/N/project/prostate_cancer_ai/anshika/regGAN/crop_box.json)")
        return

    with open(BOX_JSON) as fh:
        box_data = json.load(fh)
    y0, y1, x0, x1 = box_data["y0"], box_data["y1"], box_data["x0"], box_data["x1"]
    print(f"Crop box: rows [{y0},{y1})  cols [{x0},{x1})  "
          f"size {y1-y0}x{x1-x0}  padding={box_data.get('padding')}")
    print(f"Box computed from TRAIN patients only (13 real-PD + fake-PD train), "
          f"never verified against these 17 TEST patients.\n")

    gt_masks = {}
    for pid in ALL_PATIENTS:
        hits = (glob.glob(os.path.join(GT_DIR_D1,  f"{pid}*.seg.nrrd")) or
                glob.glob(os.path.join(GT_DIR_NEW, f"{pid}*.seg.nrrd")))
        if hits:
            gt_masks[pid] = hits[0]
        else:
            print(f"WARNING: No GT for {pid}")

    rows = []
    for pid in ALL_PATIENTS:
        if pid not in gt_masks:
            continue
        gt = load_nrrd_binary(gt_masks[pid])   # (n_slices, 384, 384) bool

        total_vox = int(gt.sum())
        if total_vox == 0:
            print(f"{pid}: no GT meniscus voxels found (unexpected) -- skipping")
            continue

        in_box_mask = np.zeros_like(gt)
        in_box_mask[:, y0:y1, x0:x1] = True
        inside_vox  = int((gt & in_box_mask).sum())
        outside_vox = total_vox - inside_vox
        pct_inside  = 100.0 * inside_vox / total_vox

        # per-slice: how many slices have ANY meniscus pixel entirely missed
        slices_with_gt = np.where(gt.any(axis=(1, 2)))[0]
        slices_fully_outside = 0
        for s in slices_with_gt:
            if not (gt[s] & in_box_mask[s]).any():
                slices_fully_outside += 1

        cohort = "D1" if pid in D1_PATIENTS else "new"
        rows.append((pid, cohort, total_vox, inside_vox, outside_vox, pct_inside,
                     len(slices_with_gt), slices_fully_outside))

    print(f"{'patient':<16}{'cohort':<6}{'gt_vox':>8}{'in_vox':>8}{'out_vox':>9}"
          f"{'%inside':>9}{'gt_slices':>11}{'fully_missed':>14}")
    for r in rows:
        pid, cohort, total_vox, inside_vox, outside_vox, pct_inside, n_gt_slices, fully_missed = r
        flag = "  <-- CHECK" if pct_inside < 95.0 or fully_missed > 0 else ""
        print(f"{pid:<16}{cohort:<6}{total_vox:>8}{inside_vox:>8}{outside_vox:>9}"
              f"{pct_inside:>8.2f}%{n_gt_slices:>11}{fully_missed:>14}{flag}")

    pct_all = [r[5] for r in rows]
    print(f"\nMean %inside across 17 patients: {np.mean(pct_all):.2f}%")
    print(f"Min  %inside (worst patient):    {np.min(pct_all):.2f}%  ({rows[int(np.argmin(pct_all))][0]})")
    n_any_miss = sum(1 for r in rows if r[4] > 0)
    n_fully_missed_slices = sum(r[7] for r in rows)
    print(f"Patients with ANY GT voxel outside the box: {n_any_miss}/{len(rows)}")
    print(f"Total GT slices with meniscus that fall FULLY outside the box "
          f"(guaranteed-zero prediction): {n_fully_missed_slices}")


if __name__ == "__main__":
    main()
