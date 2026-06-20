"""
Stage 0 (BigRed side) — confirm train/val patient-overlap bug numerically.

No GPU, no torch, no venv needed — stdlib only. Run directly on a BigRed
login node or any node with python3:

    python3 scripts/bigred_check_split.py \
        --splits /N/project/prostate_cancer_ai/anshika/regGAN/preprocessed/splits.json \
        --out_json scripts/stage0_split_report.json

Copy the resulting --out_json back to your laptop and run
stage0_visualize_split.py locally to plot it.
"""
import argparse
import json
import os
from pathlib import Path


def patient_stem(slice_path: str) -> str:
    # "MTR_005_Anonymized_..._sl0042.npy" -> "MTR_005_Anonymized_..."
    return Path(slice_path).stem.rsplit("_sl", 1)[0]


def report(name, splits, modality):
    tr = splits[modality]["train"]
    va = splits[modality]["val"]
    te = splits[modality].get("test", [])

    tr_p = set(patient_stem(p) for p in tr)
    va_p = set(patient_stem(p) for p in va)
    te_p = set(patient_stem(p) for p in te)

    overlap_tr_va = tr_p & va_p
    overlap_tr_te = tr_p & te_p
    overlap_va_te = va_p & te_p

    print(f"\n  [{name}]")
    print(f"    train: {len(tr):5d} slices  from {len(tr_p):3d} patients")
    print(f"    val:   {len(va):5d} slices  from {len(va_p):3d} patients")
    print(f"    test:  {len(te):5d} slices  from {len(te_p):3d} patients")
    print(f"    train∩val patients:  {len(overlap_tr_va)}  {sorted(overlap_tr_va)[:5]}")
    print(f"    train∩test patients: {len(overlap_tr_te)}  {sorted(overlap_tr_te)[:5]}")
    print(f"    val∩test patients:   {len(overlap_va_te)}  {sorted(overlap_va_te)[:5]}")

    return {
        "modality": modality,
        "n_train_slices": len(tr), "n_val_slices": len(va), "n_test_slices": len(te),
        "n_train_patients": len(tr_p), "n_val_patients": len(va_p), "n_test_patients": len(te_p),
        "train_val_overlap_patients": sorted(overlap_tr_va),
        "train_test_overlap_patients": sorted(overlap_tr_te),
        "val_test_overlap_patients": sorted(overlap_va_te),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out_json", default="stage0_split_report.json")
    args = ap.parse_args()

    with open(args.splits) as f:
        splits = json.load(f)

    print("="*60)
    print(" SPLIT OVERLAP REPORT")
    print("="*60)

    results = {}
    results["dess"] = report("DESS", splits, "dess")
    results["pd"]   = report("PD",   splits, "pd")

    any_bug = any(
        len(results[m]["train_val_overlap_patients"]) > 0
        or len(results[m]["train_test_overlap_patients"]) > 0
        or len(results[m]["val_test_overlap_patients"]) > 0
        for m in results
    )

    print("\n" + "="*60)
    if any_bug:
        print("  ❌ CONFIRMED: patient leakage between splits")
    else:
        print("  ✅ No patient leakage detected (lucky random split, still fix going forward)")
    print("="*60)

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved to: {args.out_json}")
    print("  Copy this file back to your laptop and run:")
    print("    python scripts/stage0_visualize_split.py --report <path>")


if __name__ == "__main__":
    main()
