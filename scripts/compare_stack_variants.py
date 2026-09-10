"""
compare_stack_variants.py
==========================
Debug test 2 — does the 2.5D input stride matter on real PD?

Computes per-patient volumetric Dice for three prediction sets (A/B/C — same
checkpoint, same 17 patients, only the 2.5D stack's neighbour offsets differ)
against the same GT used for the 17-patient abstract numbers.

GT loading matches notebooks/analysis_abstract_17patients.ipynb's
load_nrrd_binary() exactly: SimpleITK read -> DICOMOrient RAS -> resample
in-plane to isotropic -> resize to 384x384 (order=0, nearest-neighbour) ->
threshold > 0.5. Same landmine as before: the 7 new patients' GT must come
from final-segmentations/, not segmentations/.

Usage:
  python scripts/compare_stack_variants.py \
      --dir_a results/debug2_stack_variants/A \
      --dir_b results/debug2_stack_variants/B \
      --dir_c results/debug2_stack_variants/C
"""
import argparse
import glob
import os
import re

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import zoom

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

DEFAULT_GT_D1  = os.path.expanduser("~/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final")
DEFAULT_GT_NEW = os.path.expanduser("~/Downloads/final-segmentations")   # NOT "segmentations" — landmine


def get_pid(fname):
    m = re.match(r"(AC[A-F0-9]+)", os.path.basename(fname), re.IGNORECASE)
    return m.group(1) if m else None


def patients_in_dir(directory):
    files = glob.glob(os.path.join(directory, "*.nii.gz"))
    return {get_pid(f): f for f in files if get_pid(f)}


def load_vol(path):
    return np.asarray(nib.load(path).get_fdata())


def load_nrrd_binary(path):
    """Verbatim convention from analysis_abstract_17patients.ipynb."""
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


def merge_meniscus(pred):
    return (pred > 0).astype(bool)


def dice_score(pred_bin, gt_bin):
    if pred_bin.shape != gt_bin.shape:
        raise ValueError(f"shape mismatch: pred {pred_bin.shape} vs gt {gt_bin.shape} "
                         f"— refusing to silently resample, fix the source")
    inter = (pred_bin & gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_a", required=True, help="baseline [i-1,i,i+1]")
    ap.add_argument("--dir_b", required=True, help="center-only [i,i,i]")
    ap.add_argument("--dir_c", required=True, help="wider [i-2,i,i+2]")
    ap.add_argument("--gt_dir_d1",  default=DEFAULT_GT_D1)
    ap.add_argument("--gt_dir_new", default=DEFAULT_GT_NEW)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    gt_masks = {}
    for pid in ALL_PATIENTS:
        hits = (glob.glob(os.path.join(args.gt_dir_d1,  f"{pid}*.seg.nrrd")) or
                glob.glob(os.path.join(args.gt_dir_new, f"{pid}*.seg.nrrd")))
        if hits:
            gt_masks[pid] = hits[0]
        else:
            print(f"WARNING: No GT for {pid}")
    print(f"GT masks found: {len(gt_masks)}/{len(ALL_PATIENTS)}")

    preds_a = patients_in_dir(args.dir_a)
    preds_b = patients_in_dir(args.dir_b)
    preds_c = patients_in_dir(args.dir_c)
    print(f"Variant A: {len(preds_a)} patients | B: {len(preds_b)} | C: {len(preds_c)}")

    rows = []
    for pid in ALL_PATIENTS:
        if pid not in gt_masks:
            print(f"SKIP {pid}: no GT"); continue
        missing = [v for v, d in [("A", preds_a), ("B", preds_b), ("C", preds_c)] if pid not in d]
        if missing:
            print(f"SKIP {pid}: missing predictions for variant(s) {missing}"); continue

        gt_bin = load_nrrd_binary(gt_masks[pid])

        dices = {}
        for v, d in [("A", preds_a), ("B", preds_b), ("C", preds_c)]:
            pred = load_vol(d[pid])
            if pred.shape != gt_bin.shape:
                print(f"  SHAPE MISMATCH {pid} variant {v}: pred {pred.shape} vs gt {gt_bin.shape}"
                     f" — SKIPPING this patient/variant rather than silently resampling")
                dices[v] = None
                continue
            dices[v] = dice_score(merge_meniscus(pred), gt_bin)

        if any(v is None for v in dices.values()):
            continue

        rows.append({
            "patient":     pid,
            "cohort":      "D1" if pid in D1_PATIENTS else "new",
            "dice_A":      round(dices["A"], 4),
            "dice_B":      round(dices["B"], 4),
            "dice_C":      round(dices["C"], 4),
            "B_minus_A":   round(dices["B"] - dices["A"], 4),
            "C_minus_A":   round(dices["C"] - dices["A"], 4),
        })

    if not rows:
        print("No patients scored — nothing to report.")
        return

    cols = ["patient", "cohort", "dice_A", "dice_B", "dice_C", "B_minus_A", "C_minus_A"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}

    print()
    print("| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |")
    print("|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|")
    for r in sorted(rows, key=lambda r: r["patient"]):
        print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |")

    mean_a = sum(r["dice_A"] for r in rows) / len(rows)
    mean_b = sum(r["dice_B"] for r in rows) / len(rows)
    mean_c = sum(r["dice_C"] for r in rows) / len(rows)
    mrow = {"patient": "MEAN", "cohort": f"{len(rows)}pt",
            "dice_A": round(mean_a, 4), "dice_B": round(mean_b, 4), "dice_C": round(mean_c, 4),
            "B_minus_A": round(mean_b - mean_a, 4), "C_minus_A": round(mean_c - mean_a, 4)}
    print("| " + " | ".join(str(mrow[c]).ljust(widths[c]) for c in cols) + " |")

    print(f"\n=== control check ===")
    print(f"  mean dice_A = {mean_a:.4f}  (should be ~0.781 — the reported v8_mixed number)")
    if abs(mean_a - 0.781) > 0.01:
        print(f"  ⚠️  A does NOT reproduce 0.781 (off by {mean_a - 0.781:+.4f}) — "
             f"something changed besides the stack. Treat B/C results as suspect until resolved.")

    print(f"\n=== how to read B vs A (n={len(rows)}) ===")
    d_ba = mean_b - mean_a
    if abs(d_ba) < 0.01:
        print(f"  B ≈ A ({d_ba:+.4f}): model ignores the side channels on real PD.")
        print(f"  -> 2.5D is not the bottleneck here. Not worth chasing the stride mismatch.")
    elif d_ba >= 0.02:
        print(f"  B > A by {d_ba:+.4f}: side channels are ACTIVELY HURTING on real PD.")
        print(f"  -> Stride mismatch is real. Rebuild fake-PD stacks with wider neighbour spacing, retrain.")
    elif d_ba <= -0.02:
        print(f"  B < A by {d_ba:+.4f}: model genuinely uses 3D context and it helps.")
        print(f"  -> Mismatch isn't the problem here. Move on.")
    else:
        print(f"  B - A = {d_ba:+.4f}: in between the ±0.01/±0.02 thresholds — ambiguous, "
             f"treat as inconclusive rather than forcing a verdict.")

    d_ca = mean_c - mean_a
    print(f"\n  C - A = {d_ca:+.4f}" +
         ("  -> supports the mismatch story (sensitive to neighbour distance)"
          if d_ca <= -0.02 else "  -> no strong sensitivity to neighbour distance"))

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
            w.writerow(mrow)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
