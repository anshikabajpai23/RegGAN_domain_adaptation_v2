"""
phase4_5_seed_noise.py
=======================
Phase 4 step 4.5 — noise floor between two identical runs (seed 42 vs seed 7).

Identical data, loss, hyperparameters. Only torch/numpy/random seed differs.
Whatever |delta| this produces is the floor any later fix (4.7, 4.8, 4.9) must
beat before it counts as a real effect.

Reports, per run:
  best epoch (argmax of dice_real_va — the criterion ckpt_best actually uses)
  dice_real_va at best epoch, and per-val-patient Dice at that epoch
  clean per-patient mean at that epoch
  train Dice / gap / fake-val Dice at that epoch
  final-epoch values (so plateau drift is visible, not just the argmax pick)

Results only — no verdict printed.

Usage:
  python scripts/phase4_5_seed_noise.py \
      --run_a segmentation_runs/run_v7_replica_seed42 \
      --run_b segmentation_runs/run_v7_replica_seed7 \
      --out_csv results/phase4_5_seed_noise.csv
"""
import argparse
import csv
import os


def load_history(run_dir):
    path = os.path.join(run_dir, "history.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no history.csv under {run_dir}")
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


def patient_cols(rows):
    return [k for k in rows[0]
            if k.startswith("dice_va_") and k != "dice_va_perpatient_mean"]


def summarize(rows, label):
    best = max(rows, key=lambda r: r["dice_real_va"])
    last = rows[-1]
    return {
        "run": label,
        "n_epochs": len(rows),
        "best_epoch": best["epoch"],
        "dice_real_va_best": best["dice_real_va"],
        "dice_real_tr_best": best["dice_real_tr"],
        "gap_best": best["gap"],
        "dice_fake_va_best": best["dice_fake_va"],
        "perpatient_mean_best": best.get("dice_va_perpatient_mean"),
        "dice_real_va_last": last["dice_real_va"],
        "last_epoch": last["epoch"],
        "_best_row": best,
        "_rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True)
    ap.add_argument("--run_b", required=True)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    ra, rb = load_history(args.run_a), load_history(args.run_b)
    la = os.path.basename(args.run_a.rstrip("/"))
    lb = os.path.basename(args.run_b.rstrip("/"))
    A, B = summarize(ra, la), summarize(rb, lb)

    pa, pb = patient_cols(ra), patient_cols(rb)
    if set(pa) != set(pb):
        print(f"WARNING: val patient columns differ\n  {la}: {pa}\n  {lb}: {pb}")
    pats = [p for p in pa if p in pb]

    print("=== run summary, each at its OWN best epoch (ckpt_best criterion = dice_real_va) ===\n")
    fields = [
        ("epochs run", "n_epochs", "{}"),
        ("best epoch", "best_epoch", "{}"),
        ("dice_real_va @best", "dice_real_va_best", "{:.4f}"),
        ("dice_real_tr @best", "dice_real_tr_best", "{:.4f}"),
        ("gap @best", "gap_best", "{:+.4f}"),
        ("dice_fake_va @best", "dice_fake_va_best", "{:.4f}"),
        ("perpatient mean @best", "perpatient_mean_best", "{:.4f}"),
        ("dice_real_va @last", "dice_real_va_last", "{:.4f}"),
    ]
    w = 24
    print(f"| {'metric'.ljust(w)} | {la.ljust(14)} | {lb.ljust(14)} | {'delta'.ljust(10)} |")
    print(f"|{'-'*(w+2)}|{'-'*16}|{'-'*16}|{'-'*12}|")
    for name, key, fmt in fields:
        va, vb = A[key], B[key]
        if va is None or vb is None:
            continue
        d = vb - va
        ds = f"{d:+.4f}" if isinstance(d, float) else f"{d:+d}"
        print(f"| {name.ljust(w)} | {fmt.format(va).ljust(14)} | {fmt.format(vb).ljust(14)} | {ds.ljust(10)} |")

    print(f"\n=== per-val-patient Dice, each run at its own best epoch ===\n")
    print(f"| {'patient'.ljust(18)} | {la.ljust(14)} | {lb.ljust(14)} | {'delta'.ljust(10)} |")
    print(f"|{'-'*20}|{'-'*16}|{'-'*16}|{'-'*12}|")
    out_rows = []
    for p in pats:
        va, vb = A["_best_row"][p], B["_best_row"][p]
        pid = p.replace("dice_va_", "")
        print(f"| {pid.ljust(18)} | {va:<14.4f} | {vb:<14.4f} | {vb-va:<+10.4f} |")
        out_rows.append({"patient": pid, f"{la}": round(va, 4),
                         f"{lb}": round(vb, 4), "delta": round(vb - va, 4)})

    deltas = [abs(B["_best_row"][p] - A["_best_row"][p]) for p in pats]
    print(f"\n=== noise floor ===")
    print(f"  |delta| dice_real_va at best epoch : {abs(B['dice_real_va_best']-A['dice_real_va_best']):.4f}")
    print(f"  |delta| per-patient, max           : {max(deltas):.4f}")
    print(f"  |delta| per-patient, mean          : {sum(deltas)/len(deltas):.4f}")
    print(f"  best-epoch spread                  : {A['best_epoch']} vs {B['best_epoch']}"
          f"  (|delta| = {abs(A['best_epoch']-B['best_epoch'])} epochs)")

    # plateau spread within each run, for context on how flat the selection surface is
    print(f"\n=== plateau spread within each run (epoch >= 4) ===")
    for S, lab in ((A, la), (B, lb)):
        vals = [r["dice_real_va"] for r in S["_rows"] if r["epoch"] >= 4]
        if vals:
            print(f"  {lab}: min={min(vals):.4f} max={max(vals):.4f} range={max(vals)-min(vals):.4f}")

    if args.out_csv and out_rows:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            wr.writeheader()
            wr.writerows(out_rows)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
