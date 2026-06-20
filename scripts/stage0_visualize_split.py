"""
Stage 0 visualization — plot train/val/test patient counts and overlap
from the report produced by bigred_check_split.py.

Run LOCALLY after copying stage0_split_report.json back from BigRed:

    venv/bin/python scripts/stage0_visualize_split.py \
        --report scripts/stage0_split_report.json
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="scripts/stage0_split_report.json")
    ap.add_argument("--out", default="scripts/stage0_split_overlap.png")
    args = ap.parse_args()

    with open(args.report) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, modality in zip(axes, ["dess", "pd"]):
        r = results[modality]
        splits = ["train", "val", "test"]
        patient_counts = [r["n_train_patients"], r["n_val_patients"], r["n_test_patients"]]
        colors = ["#4C72B0", "#DD8452", "#55A868"]
        ax.bar(splits, patient_counts, color=colors)
        ax.set_title(f"{modality.upper()} — patients per split")
        ax.set_ylabel("# unique patients")
        for i, v in enumerate(patient_counts):
            ax.text(i, v + 0.3, str(v), ha="center")

        n_overlap = len(r["train_val_overlap_patients"])
        if n_overlap > 0:
            ax.text(0.5, max(patient_counts) * 0.9,
                    f"⚠ {n_overlap} patients\nleak train↔val",
                    ha="center", color="red", fontsize=10, fontweight="bold",
                    transform=ax.transData)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
