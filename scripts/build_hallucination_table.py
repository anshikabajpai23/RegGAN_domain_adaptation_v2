"""
build_hallucination_table.py
============================
Splits segmentation error into three kinds, per patient:

  1. empty-slice FP : model painted meniscus on a slice with ZERO meniscus in GT
  2. on-slice FP    : model painted too much on a slice that DOES have meniscus
  3. FN             : model missed meniscus the GT has

and — for every kind-1 slice — how far it is (in slices) from the nearest
GT-nonempty slice. That distance is the whole point:

    dist 1        -> model extended the meniscus one slice past where the human
                     stopped drawing. Annotation ambiguity, NOT hallucination.
    dist >= 3     -> model invented meniscus in empty space. Real hallucination.

Input is a per-slice CSV with columns:
    patient_id, slice_idx, dice, gt_voxels, pred_voxels, false_neg, false_pos

Why that CSV is sufficient: on a slice with gt_voxels == 0 the true positives are
necessarily 0, so false_pos == pred_voxels == the hallucinated pixels on that
slice. On a slice with gt_voxels > 0, false_pos is the on-slice FP. No image data
needed.

Usage:
  python scripts/build_hallucination_table.py \
      --csv    segmentation_runs/run_v7_replica_seed42/val_predictions/per_slice_dice.csv \
      --cohort real_val \
      --pd_root data/iu-dataset/pd-files        # optional, for native_res
"""
import argparse
import csv
import glob
import os
from collections import defaultdict


def native_inplane(pd_root, pid):
    """Original in-plane size before any resampling, from the raw PD NIfTI."""
    if not pd_root:
        return "?"
    hits = glob.glob(os.path.join(pd_root, f"{pid}*.nii.gz"))
    if not hits:
        return "?"
    try:
        import nibabel as nib
        shp = nib.load(hits[0]).shape
        # sagittal PD: the two largest dims are in-plane
        inplane = sorted(shp, reverse=True)[:2]
        return str(inplane[0]) if inplane[0] == inplane[1] else f"{inplane[0]}x{inplane[1]}"
    except Exception as e:
        return f"err:{e.__class__.__name__}"


def analyze(rows, cohort, pd_root=None):
    by_pid = defaultdict(list)
    for r in rows:
        by_pid[r["patient_id"]].append({
            "idx":  int(r["slice_idx"]),
            "gt":   int(float(r["gt_voxels"])),
            "pred": int(float(r["pred_voxels"])),
            "fn":   int(float(r["false_neg"])),
            "fp":   int(float(r["false_pos"])),
        })

    out = []
    for pid in sorted(by_pid):
        sl = sorted(by_pid[pid], key=lambda s: s["idx"])
        gt_nonempty_idx = [s["idx"] for s in sl if s["gt"] > 0]

        n_gt_empty = n_halluc = halluc_px = 0
        onslice_fp = fn_px = gt_px = pred_px = tp_px = 0
        on_pred_px = on_tp_px = 0                   # restricted to GT-nonempty slices
        dists = []

        for s in sl:
            gt_px   += s["gt"]
            pred_px += s["pred"]
            tp_px   += s["pred"] - s["fp"]          # TP = pred - FP
            fn_px   += s["fn"]

            if s["gt"] == 0:
                n_gt_empty += 1
                if s["pred"] > 0:
                    n_halluc  += 1
                    halluc_px += s["pred"]          # == false_pos when gt==0
                    if gt_nonempty_idx:
                        dists.append(min(abs(s["idx"] - g) for g in gt_nonempty_idx))
                    else:
                        dists.append(-1)            # patient has no GT at all
            else:
                onslice_fp  += s["fp"]
                on_pred_px  += s["pred"]
                on_tp_px    += s["pred"] - s["fp"]

        # dice          : whole volume — the number normally reported
        # dice_on_slices: same, but only over slices that actually contain meniscus.
        #                 The gap between them is exactly what the empty-slice FP costs.
        dice = (2 * tp_px / (pred_px + gt_px)) if (pred_px + gt_px) > 0 else 1.0
        dice_on = ((2 * on_tp_px / (on_pred_px + gt_px))
                   if (on_pred_px + gt_px) > 0 else 1.0)

        out.append({
            "patient":       pid,
            "cohort":        cohort,
            "native_res":    native_inplane(pd_root, pid),
            "n_slices":      len(sl),
            "n_gt_empty":    n_gt_empty,
            "n_halluc":      n_halluc,
            "halluc_px":     halluc_px,
            "halluc_dist":   ",".join(str(d) for d in sorted(dists)) if dists else "-",
            "onslice_fp_px": onslice_fp,
            "fn_px":         fn_px,
            "gt_px":         gt_px,
            "dice":          round(dice, 4),
            "dice_on_sl":    round(dice_on, 4),
            "d_dice":        round(dice_on - dice, 4),
            "_dists":        dists,
        })
    return out


COLS = ["patient", "cohort", "native_res", "n_slices", "n_gt_empty", "n_halluc",
        "halluc_px", "halluc_dist", "onslice_fp_px", "fn_px", "gt_px",
        "dice", "dice_on_sl", "d_dice"]


def _totals(rows):
    tot = {c: 0 for c in ["n_slices", "n_gt_empty", "n_halluc", "halluc_px",
                          "onslice_fp_px", "fn_px", "gt_px"]}
    all_d = []
    for r in rows:
        for c in tot:
            tot[c] += r[c]
        all_d += r["_dists"]
    return tot, all_d


def render(rows):
    cohorts = sorted({r["cohort"] for r in rows})
    tot, all_d = _totals(rows)

    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in COLS}
    widths["halluc_dist"] = max(widths["halluc_dist"], 11)

    print("| " + " | ".join(c.ljust(widths[c]) for c in COLS) + " |")
    print("|" + "|".join("-" * (widths[c] + 2) for c in COLS) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in COLS) + " |")

    total_fp = tot["halluc_px"] + tot["onslice_fp_px"]
    mean_dice    = sum(r["dice"] for r in rows) / len(rows)
    mean_dice_on = sum(r["dice_on_sl"] for r in rows) / len(rows)

    # per-cohort subtotal row(s) if more than one cohort is present
    if len(cohorts) > 1:
        for coh in cohorts:
            crows = [r for r in rows if r["cohort"] == coh]
            ctot, _ = _totals(crows)
            cmean    = sum(r["dice"] for r in crows) / len(crows)
            cmean_on = sum(r["dice_on_sl"] for r in crows) / len(crows)
            srow = {c: "" for c in COLS}
            srow.update({"patient": f"  subtotal", "cohort": f"{coh} ({len(crows)}pt)",
                        "native_res": "-", "halluc_dist": "-",
                        "dice": f"μ={cmean:.4f}", "dice_on_sl": f"μ={cmean_on:.4f}",
                        "d_dice": f"μ={cmean_on - cmean:+.4f}",
                        **{k: v for k, v in ctot.items()}})
            print("| " + " | ".join(str(srow[c]).ljust(widths[c]) for c in COLS) + " |")

    trow = {c: "" for c in COLS}
    trow.update({"patient": "TOTAL", "cohort": f"{len(rows)}pt", "native_res": "-",
                 "halluc_dist": "-",
                 "dice":       f"μ={mean_dice:.4f}",
                 "dice_on_sl": f"μ={mean_dice_on:.4f}",
                 "d_dice":     f"μ={mean_dice_on - mean_dice:+.4f}",
                 **{k: v for k, v in tot.items()}})
    print("| " + " | ".join(str(trow[c]).ljust(widths[c]) for c in COLS) + " |")

    print("\n=== per-patient val Dice ===")
    for r in sorted(rows, key=lambda r: r["dice"]):
        print(f"  {r['patient']:<18} {r['dice']:.4f}   "
              f"meniscus-slices-only {r['dice_on_sl']:.4f}   "
              f"(empty slices cost {r['d_dice']:+.4f})")
    print(f"  {'MEAN':<18} {mean_dice:.4f}   "
          f"meniscus-slices-only {mean_dice_on:.4f}   "
          f"(empty slices cost {mean_dice_on - mean_dice:+.4f})")

    print("\n=== error split ===")
    denom = total_fp + tot["fn_px"]
    for label, v in [("empty-slice FP (kind 1)", tot["halluc_px"]),
                     ("on-slice FP   (kind 2)", tot["onslice_fp_px"]),
                     ("FN            (kind 3)", tot["fn_px"])]:
        print(f"  {label}: {v:>8,}  = {v/denom*100:5.1f}% of all wrong pixels"
              f"  ({v/total_fp*100:5.1f}% of FP)" if "FP" in label
              else f"  {label}: {v:>8,}  = {v/denom*100:5.1f}% of all wrong pixels")
    print(f"  total wrong pixels     : {denom:,}   (GT total {tot['gt_px']:,})")
    print(f"  FP/FN ratio            : {total_fp/max(tot['fn_px'],1):.2f}x")

    if all_d:
        print("\n=== halluc_dist — is it ambiguity or invention? ===")
        hist = defaultdict(int)
        for d in all_d:
            hist[d] += 1
        for d in sorted(hist):
            lab = "no GT in volume" if d < 0 else f"{d} slice(s) away"
            print(f"  {lab:<18}: {hist[d]:>3} slices")
        near = sum(v for k, v in hist.items() if 0 <= k <= 2)
        far  = sum(v for k, v in hist.items() if k >= 3)
        print(f"\n  dist 1-2 (annotation ambiguity) : {near:>3} / {len(all_d)}"
              f"  = {near/len(all_d)*100:.0f}%")
        print(f"  dist >=3 (real hallucination)   : {far:>3} / {len(all_d)}"
              f"  = {far/len(all_d)*100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="single-cohort mode (use with --cohort)")
    ap.add_argument("--cohort", default="val")
    ap.add_argument("--input", action="append", default=[],
                    help="multi-cohort mode: PATH:COHORT, repeatable, e.g. "
                         "--input a.csv:real_val --input b.csv:fake_val")
    ap.add_argument("--pd_root", default=None)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    if not args.csv and not args.input:
        ap.error("pass --csv/--cohort, or one or more --input PATH:COHORT")

    rows = []
    if args.csv:
        rows += analyze(list(csv.DictReader(open(args.csv))), args.cohort, args.pd_root)
    for spec in args.input:
        path, _, cohort = spec.rpartition(":")
        if not path:
            ap.error(f"--input must be PATH:COHORT, got: {spec}")
        rows += analyze(list(csv.DictReader(open(path))), cohort, args.pd_root)

    render(rows)

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r[c] for c in COLS})
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
