"""
aggregate_kfold_results.py
===========================
Phase 4 step 6 — after all 5 folds finish, pulls each fold's held-out
patients' Dice (at that fold's OWN best epoch, same dice_real_va criterion
as every other Phase 4 run) into one 15-patient table, plus mean +/- SD and
the best-epoch spread across folds.

Reuses the exact fold/patient assignment from build_kfold_real_pd.py so
there's a single source of truth — this script cannot silently drift out of
sync with which patients were actually held out in which fold.

Usage:
  python scripts/aggregate_kfold_results.py \
      --runs_root segmentation_runs \
      --out_csv results/phase4_6_kfold.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kfold_real_pd import FOLDS, PATIENTS, N_FOLDS


def load_history(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in list(r.items()):
            if k == "epoch":
                r[k] = int(v)
            else:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="segmentation_runs")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    per_patient = {}
    best_epochs = []
    missing_folds = []

    for k in range(N_FOLDS):
        hist_path = os.path.join(args.runs_root, f"run_v7_kfold{k}", "history.csv")
        rows = load_history(hist_path)
        if rows is None:
            missing_folds.append(k)
            print(f"fold {k}: MISSING ({hist_path} not found)")
            continue

        best = max(rows, key=lambda r: r["dice_real_va"])
        best_epochs.append(best["epoch"])
        print(f"fold {k}: best epoch {best['epoch']}  dice_real_va={best['dice_real_va']:.4f}  "
             f"val patients={FOLDS[k]}")

        for pid in FOLDS[k]:
            col = f"dice_va_{pid}"
            if col not in best:
                print(f"  WARNING: {col} not found in {hist_path} columns — "
                     f"check the fold's real_data_root matched FOLDS[{k}]")
                continue
            per_patient[pid] = best[col]

    if missing_folds:
        print(f"\n{len(missing_folds)} fold(s) not yet done: {missing_folds}. "
             f"Aggregating what's available; re-run once the rest finish.")

    print(f"\n=== per-patient Dice, each from its OWN fold's best epoch ({len(per_patient)}/15) ===")
    for pid in PATIENTS:
        if pid in per_patient:
            print(f"  {pid:<18} {per_patient[pid]:.4f}")
        else:
            print(f"  {pid:<18} (missing)")

    vals = list(per_patient.values())
    if vals:
        print(f"\nmean = {np.mean(vals):.4f}   SD = {np.std(vals):.4f}   "
             f"min = {min(vals):.4f}   max = {max(vals):.4f}   n = {len(vals)}")
    if best_epochs:
        print(f"best-epoch spread across folds: {sorted(best_epochs)}  "
             f"(range {max(best_epochs) - min(best_epochs)})")

    if args.out_csv and vals:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["patient", "fold", "dice_va_at_fold_best_epoch"])
            for k in range(N_FOLDS):
                for pid in FOLDS[k]:
                    if pid in per_patient:
                        w.writerow([pid, k, f"{per_patient[pid]:.4f}"])
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
