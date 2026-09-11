"""
phase4_3_edge_core_split.py
============================
Phase 4 step 4.3 — edge-vs-core error split on the 17-patient per-slice CSVs.

Buckets (per spec):
  empty : GT == 0
  edge  : GT > 0 and within +/-1 slice of ANY span end (span first or last)
  core  : all other GT > 0 slices

Spans = contiguous runs of GT-nonempty slices, found per patient on slice_idx.
FP and FN summed per bucket, per variant. All patients are real PD (~3.6mm),
so slice counts are internally comparable (Corrections C1 does not bite here).

Usage:
  python scripts/phase4_3_edge_core_split.py \
      --csv results/debug2_stack_variants/per_slice_A.csv:A \
      --csv results/debug2_stack_variants/per_slice_B.csv:B \
      --out_csv results/phase4_3_edge_core.csv
"""
import argparse
import csv
import os
from collections import defaultdict


def load(path):
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        by[r["patient_id"]].append({
            "idx": int(r["slice_idx"]),
            "gt":  int(float(r["gt_voxels"])),
            "fp":  int(float(r["false_pos"])),
            "fn":  int(float(r["false_neg"])),
        })
    for pid in by:
        by[pid].sort(key=lambda s: s["idx"])
    return by


def spans_of(slices):
    idx = [s["idx"] for s in slices if s["gt"] > 0]
    if not idx:
        return []
    runs, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append((cur[0], cur[-1])); cur = [b]
    runs.append((cur[0], cur[-1]))
    return runs


def bucket_patient(slices):
    spans = spans_of(slices)
    ends = set()
    for f, l in spans:
        ends.add(f); ends.add(l)
    out = {}
    for s in slices:
        if s["gt"] == 0:
            out[s["idx"]] = "empty"
        elif any(abs(s["idx"] - e) <= 1 for e in ends):
            out[s["idx"]] = "edge"
        else:
            out[s["idx"]] = "core"
    return out, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="PATH:LABEL, repeatable")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    variants = {}
    for spec in args.csv:
        path, _, label = spec.rpartition(":")
        variants[label] = load(path)

    BUCKETS = ["empty", "edge", "core"]
    per_variant = {}
    per_patient = defaultdict(dict)

    for label, data in variants.items():
        agg = {b: {"fp": 0, "fn": 0, "n": 0} for b in BUCKETS}
        for pid, slices in data.items():
            bmap, _ = bucket_patient(slices)
            pp = {b: {"fp": 0, "fn": 0} for b in BUCKETS}
            for s in slices:
                b = bmap[s["idx"]]
                agg[b]["fp"] += s["fp"]; agg[b]["fn"] += s["fn"]; agg[b]["n"] += 1
                pp[b]["fp"] += s["fp"]; pp[b]["fn"] += s["fn"]
            per_patient[pid][label] = pp
        per_variant[label] = agg

    rows_out = []
    for label in variants:
        agg = per_variant[label]
        total = sum(agg[b]["fp"] + agg[b]["fn"] for b in BUCKETS)
        print(f"\n=== variant {label} ===")
        print(f"| {'bucket':<7} | {'slices':>7} | {'FP px':>9} | {'FN px':>9} | {'wrong px':>9} | {'share':>7} |")
        print(f"|{'-'*9}|{'-'*9}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*9}|")
        for b in BUCKETS:
            w = agg[b]["fp"] + agg[b]["fn"]
            print(f"| {b:<7} | {agg[b]['n']:>7} | {agg[b]['fp']:>9,} | {agg[b]['fn']:>9,} | "
                  f"{w:>9,} | {w/total:>7.4f} |")
            rows_out.append({"variant": label, "bucket": b, "slices": agg[b]["n"],
                             "fp_px": agg[b]["fp"], "fn_px": agg[b]["fn"],
                             "wrong_px": w, "share": round(w/total, 4)})
        print(f"| {'TOTAL':<7} | {sum(agg[b]['n'] for b in BUCKETS):>7} | "
              f"{sum(agg[b]['fp'] for b in BUCKETS):>9,} | "
              f"{sum(agg[b]['fn'] for b in BUCKETS):>9,} | {total:>9,} | {1.0:>7.4f} |")

    labels = list(variants)
    if len(labels) == 2:
        a, b = labels
        print(f"\n=== {b} - {a}, by bucket (wrong px; negative = {b} better) ===")
        print(f"| {'bucket':<7} | {'d_FP':>9} | {'d_FN':>9} | {'d_wrong':>9} |")
        print(f"|{'-'*9}|{'-'*11}|{'-'*11}|{'-'*11}|")
        for bk in BUCKETS:
            dfp = per_variant[b][bk]["fp"] - per_variant[a][bk]["fp"]
            dfn = per_variant[b][bk]["fn"] - per_variant[a][bk]["fn"]
            print(f"| {bk:<7} | {dfp:>+9,} | {dfn:>+9,} | {dfp+dfn:>+9,} |")

        print(f"\n=== per-patient {b} - {a} (wrong px by bucket) ===")
        print(f"| {'patient':<16} | {'d_empty':>8} | {'d_edge':>8} | {'d_core':>8} |")
        print(f"|{'-'*18}|{'-'*10}|{'-'*10}|{'-'*10}|")
        for pid in sorted(per_patient):
            d = []
            for bk in BUCKETS:
                pa, pb = per_patient[pid][a][bk], per_patient[pid][b][bk]
                d.append((pb["fp"] + pb["fn"]) - (pa["fp"] + pa["fn"]))
            print(f"| {pid:<16} | {d[0]:>+8,} | {d[1]:>+8,} | {d[2]:>+8,} |")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["variant", "bucket", "slices",
                                               "fp_px", "fn_px", "wrong_px", "share"])
            w.writeheader(); w.writerows(rows_out)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
