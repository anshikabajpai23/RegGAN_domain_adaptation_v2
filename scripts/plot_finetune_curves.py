"""
plot_finetune_curves.py
========================
Parses finetune_meniscus.py stdout log and plots:
  - Train loss vs Val loss over epochs
  - Val Dice per class (background, lateral, medial) over epochs
  - Mean meniscus Dice over epochs

Usage:
    python scripts/plot_finetune_curves.py \
        --log_file /N/project/prostate_cancer_ai/anshika/regGAN/logs/finetune_all155_JOBID.out \
        --out_png  /N/project/prostate_cancer_ai/anshika/regGAN/segmentation_runs/run_002/finetune_curves.png
"""
import argparse
import re
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(log_file):
    epochs, train_loss, val_loss = [], [], []
    bg_dice, lat_dice, med_dice, mean_dice = [], [], [], []

    pattern = re.compile(
        r"Epoch (\d+)\s+"
        r"train_loss=([\d.]+)\s+"
        r"val_loss=([\d.]+)\s+"
        r"val_dice\[bg=([\d.]+) lat=([\d.]+) med=([\d.]+)\]\s+"
        r"mean_meniscus_dice=([\d.]+)"
    )

    with open(log_file) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                bg_dice.append(float(m.group(4)))
                lat_dice.append(float(m.group(5)))
                med_dice.append(float(m.group(6)))
                mean_dice.append(float(m.group(7)))

    return epochs, train_loss, val_loss, bg_dice, lat_dice, med_dice, mean_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_file", required=True)
    ap.add_argument("--out_png",  required=True)
    args = ap.parse_args()

    epochs, train_loss, val_loss, bg_dice, lat_dice, med_dice, mean_dice = parse_log(args.log_file)

    if not epochs:
        print("No epoch data found in log. Check the log file path.")
        return

    print(f"Parsed {len(epochs)} epochs  (epoch {epochs[0]} -> {epochs[-1]})")
    print(f"Best mean meniscus Dice: {max(mean_dice):.4f} at epoch {epochs[mean_dice.index(max(mean_dice))]}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fine-tuning Curves — run_002 (155 patients)", fontsize=13, fontweight="bold")

    # --- Panel 1: Loss ---
    axes[0].plot(epochs, train_loss, label="Train Loss", color="tab:blue")
    axes[0].plot(epochs, val_loss,   label="Val Loss",   color="tab:orange")
    axes[0].set_title("Train vs Val Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Panel 2: Dice ---
    axes[1].plot(epochs, mean_dice, label="Mean Meniscus", color="tab:red",    linewidth=2)
    axes[1].plot(epochs, lat_dice,  label="Lateral",       color="tab:green",  linestyle="--")
    axes[1].plot(epochs, med_dice,  label="Medial",        color="tab:purple", linestyle="--")
    axes[1].plot(epochs, bg_dice,   label="Background",    color="tab:gray",   linestyle=":")
    axes[1].axhline(y=max(mean_dice), color="tab:red", linestyle=":", alpha=0.5,
                    label=f"Best={max(mean_dice):.4f}")
    axes[1].set_title("Val Dice per Class", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    plt.savefig(args.out_png, dpi=150)
    print(f"Saved -> {args.out_png}")


if __name__ == "__main__":
    main()
