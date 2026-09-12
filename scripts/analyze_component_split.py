"""
analyze_component_split.py
===========================
Per-slice connected-component decomposition of prediction vs GT.

For each slice:
  n_cc_gt      : connected components in GT
  n_cc_pred    : connected components in prediction
  orphan_cc    : prediction components with ZERO overlap against any GT pixel
  orphan_px    : pixels living in those orphan components
  missed_cc    : GT components with ZERO overlap against any prediction pixel
  missed_px    : pixels in those missed GT components
  tp/fp/fn     : standard pixel counts

Two distinct things get conflated by the phrase "split", so both are reported:
  (a) ORPHAN components  — a predicted blob sitting somewhere GT has nothing.
      This is what `orphan_px` measures.
  (b) FRAGMENTATION      — the meniscus broken into more pieces than GT has
      (n_cc_pred > n_cc_gt) where the pieces still overlap GT. Reported as
      `excess_cc`, and is the failure mode documented for torn menisci
      ("model splits the meniscus into two disconnected halves at the tear").

"Core" slices = GT non-empty. On GT-empty slices every predicted pixel is
trivially orphaned (there is no GT component to overlap), which would swamp the
statistic — so the core-slice fraction is computed on GT-non-empty slices only.

Connectivity: 8-connected (3x3 structuring element). 4-connectivity would
artificially split thin diagonal structures, which a meniscus cross-section
frequently is.

Usage:
  python scripts/analyze_component_split.py \
      --val_dir segmentation_runs/run_v7_replica_seed42/val_predictions \
      --out_csv results/debug_component_split_replica.csv
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import nibabel as nib
from scipy.ndimage import label

CONN8 = np.ones((3, 3), dtype=int)


def get_pid(fname):
    m = re.match(r"(AC[A-F0-9]+|MTR_\d+)", os.path.basename(fname), re.IGNORECASE)
    return m.group(1) if m else None


def slice_indices_from_csv(csv_path, pid, n_slices):
    """Recover true volume slice indices if the per-slice CSV is available;
    otherwise fall back to array position (and say so)."""
    if not os.path.exists(csv_path):
        return list(range(n_slices)), False
    idx = [int(r["slice_idx"]) for r in csv.DictReader(open(csv_path))
           if r["patient_id"] == pid]
    idx.sort()
    if len(idx) != n_slices:
        return list(range(n_slices)), False
    return idx, True


def analyze_slice(gt2d, pr2d):
    gt_lab, n_gt = label(gt2d, structure=CONN8)
    pr_lab, n_pr = label(pr2d, structure=CONN8)

    tp = int((gt2d & pr2d).sum())
    fp = int((pr2d & ~gt2d).sum())
    fn = int((gt2d & ~pr2d).sum())

    orphan_cc = orphan_px = 0
    for c in range(1, n_pr + 1):
        comp = pr_lab == c
        if not (comp & gt2d).any():          # no overlap with ANY gt pixel
            orphan_cc += 1
            orphan_px += int(comp.sum())

    missed_cc = missed_px = 0
    for c in range(1, n_gt + 1):
        comp = gt_lab == c
        if not (comp & pr2d).any():
            missed_cc += 1
            missed_px += int(comp.sum())

    return {
        "n_cc_gt": n_gt, "n_cc_pred": n_pr,
        "excess_cc": max(0, n_pr - n_gt),
        "orphan_cc": orphan_cc, "orphan_px": orphan_px,
        "missed_cc": missed_cc, "missed_px": missed_px,
        "tp": tp, "fp": fp, "fn": fn,
        "gt_px": int(gt2d.sum()), "pred_px": int(pr2d.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True,
                    help="dir holding <pid>_gt.nii.gz and <pid>_pred.nii.gz")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    gts = {get_pid(f): f for f in glob.glob(os.path.join(args.val_dir, "*_gt.nii.gz"))}
    prs = {get_pid(f): f for f in glob.glob(os.path.join(args.val_dir, "*_pred.nii.gz"))}
    pids = sorted(set(gts) & set(prs))
    print(f"patients with both gt and pred: {len(pids)} -> {pids}")

    per_slice_csv = os.path.join(args.val_dir, "per_slice_dice.csv")
    rows = []
    for pid in pids:
        gt = np.asarray(nib.load(gts[pid]).get_fdata()) > 0
        pr = np.asarray(nib.load(prs[pid]).get_fdata()) > 0
        if gt.shape != pr.shape:
            raise ValueError(f"shape mismatch {pid}: {gt.shape} vs {pr.shape}")

        idxs, exact = slice_indices_from_csv(per_slice_csv, pid, gt.shape[0])
        if not exact:
            print(f"  note: {pid} slice_idx is array position, not volume index")

        for k in range(gt.shape[0]):
            r = analyze_slice(gt[k], pr[k])
            r.update({"patient_id": pid, "slice_idx": idxs[k]})
            rows.append(r)

    cols = ["patient_id", "slice_idx", "n_cc_gt", "n_cc_pred", "excess_cc",
            "orphan_cc", "orphan_px", "missed_cc", "missed_px",
            "tp", "fp", "fn", "gt_px", "pred_px"]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"wrote {len(rows)} slice rows -> {args.out_csv}\n")

    core = [r for r in rows if r["gt_px"] > 0]
    empty = [r for r in rows if r["gt_px"] == 0]

    print("=== per-patient, CORE slices only (GT non-empty) ===")
    hdr = ["patient", "core_sl", "err_px", "orphan_px", "frac_orphan",
           "orphan_cc", "excess_cc", "missed_cc"]
    print("| " + " | ".join(h.ljust(10) for h in hdr) + " |")
    print("|" + "|".join("-" * 12 for _ in hdr) + "|")
    for pid in pids:
        c = [r for r in core if r["patient_id"] == pid]
        err = sum(r["fp"] + r["fn"] for r in c)
        orp = sum(r["orphan_px"] for r in c)
        vals = [pid, len(c), f"{err:,}", f"{orp:,}",
                f"{orp/err:.4f}" if err else "-",
                sum(r["orphan_cc"] for r in c),
                sum(r["excess_cc"] for r in c),
                sum(r["missed_cc"] for r in c)]
        print("| " + " | ".join(str(v).ljust(10) for v in vals) + " |")

    err_all = sum(r["fp"] + r["fn"] for r in core)
    orp_all = sum(r["orphan_px"] for r in core)
    print(f"\n=== CORE-slice totals ({len(core)} slices) ===")
    print(f"  total error px (fp+fn)        : {err_all:,}")
    print(f"  orphan-component px           : {orp_all:,}")
    print(f"  FRACTION of core error from")
    print(f"    orphan (no-GT-overlap) chunks: {orp_all/err_all:.4f}" if err_all else "")
    print(f"  fp on core slices             : {sum(r['fp'] for r in core):,}")
    print(f"  fn on core slices             : {sum(r['fn'] for r in core):,}")
    print(f"  missed-GT-component px        : {sum(r['missed_px'] for r in core):,}")

    print(f"\n=== component counts on core slices ===")
    same = sum(1 for r in core if r["n_cc_pred"] == r["n_cc_gt"])
    more = sum(1 for r in core if r["n_cc_pred"] > r["n_cc_gt"])
    fewer = sum(1 for r in core if r["n_cc_pred"] < r["n_cc_gt"])
    print(f"  pred has SAME #cc as GT  : {same}/{len(core)}")
    print(f"  pred has MORE  (fragmented): {more}/{len(core)}")
    print(f"  pred has FEWER (merged/missed): {fewer}/{len(core)}")
    from collections import Counter
    print(f"  n_cc_gt distribution   : {dict(sorted(Counter(r['n_cc_gt'] for r in core).items()))}")
    print(f"  n_cc_pred distribution : {dict(sorted(Counter(r['n_cc_pred'] for r in core).items()))}")

    if empty:
        print(f"\n=== GT-EMPTY slices ({len(empty)}), reported separately ===")
        print(f"  (every predicted px here is trivially 'orphan' — no GT to overlap)")
        print(f"  slices with any prediction : {sum(1 for r in empty if r['pred_px']>0)}/{len(empty)}")
        print(f"  predicted px               : {sum(r['pred_px'] for r in empty):,}")
        print(f"  components                 : {sum(r['n_cc_pred'] for r in empty)}")


if __name__ == "__main__":
    main()
