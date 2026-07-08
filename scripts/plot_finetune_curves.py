"""
plot_finetune_curves.py
========================
Parses finetune_meniscus.py stdout log and plots:
  Panel 1: Train vs Val Loss
  Panel 2: Train Dice per class (bg, lateral, medial, mean meniscus)
  Panel 3: Val Dice per class (bg, lateral, medial, mean meniscus)

Works with both old log format (no train_dice) and new format (with train_dice).

Usage:
    python scripts/plot_finetune_curves.py \
        --log_file /N/project/prostate_cancer_ai/anshika/regGAN/logs/finetune_all155_JOBID.out \
        --out_png  /N/project/prostate_cancer_ai/anshika/regGAN/segmentation_runs/run_003/finetune_curves.png
"""
import argparse
import re
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(log_file):
    epochs = []
    train_loss, val_loss = [], []
    tr_bg, tr_lat, tr_med, tr_mean = [], [], [], []
    val_bg, val_lat, val_med, val_mean = [], [], [], []

    # New format: includes train_dice
    pattern_new = re.compile(
        r"Epoch (\d+)\s+"
        r"train_loss=([\d.]+)\s+"
        r"train_dice\[bg=([\d.]+) lat=([\d.]+) med=([\d.]+)\]\s+"
        r"mean_train_meniscus_dice=([\d.]+)\s+"
        r"val_loss=([\d.]+)\s+"
        r"val_dice\[bg=([\d.]+) lat=([\d.]+) med=([\d.]+)\]\s+"
        r"mean_meniscus_dice=([\d.]+)"
    )
    # Old format: no train_dice
    pattern_old = re.compile(
        r"Epoch (\d+)\s+"
        r"train_loss=([\d.]+)\s+"
        r"val_loss=([\d.]+)\s+"
        r"val_dice\[bg=([\d.]+) lat=([\d.]+) med=([\d.]+)\]\s+"
        r"mean_meniscus_dice=([\d.]+)"
    )

    with open(log_file) as f:
        for line in f:
            m = pattern_new.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                tr_bg.append(float(m.group(3)))
                tr_lat.append(float(m.group(4)))
                tr_med.append(float(m.group(5)))
                tr_mean.append(float(m.group(6)))
                val_loss.append(float(m.group(7)))
                val_bg.append(float(m.group(8)))
                val_lat.append(float(m.group(9)))
                val_med.append(float(m.group(10)))
                val_mean.append(float(m.group(11)))
                continue

            m = pattern_old.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                val_bg.append(float(m.group(4)))
                val_lat.append(float(m.group(5)))
                val_med.append(float(m.group(6)))
                val_mean.append(float(m.group(7)))

    has_train_dice = len(tr_mean) > 0
    return (epochs, train_loss, val_loss,
            tr_bg, tr_lat, tr_med, tr_mean,
            val_bg, val_lat, val_med, val_mean,
            has_train_dice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_file", required=True)
    ap.add_argument("--out_png",  required=True)
    args = ap.parse_args()

    (epochs, train_loss, val_loss,
     tr_bg, tr_lat, tr_med, tr_mean,
     val_bg, val_lat, val_med, val_mean,
     has_train_dice) = parse_log(args.log_file)

    if not epochs:
        print("No epoch data found in log. Check the log file path.")
        print("Expected line format:")
        print("  Epoch 000  train_loss=X.XXXX  val_loss=X.XXXX  "
              "val_dice[bg=X.XXX lat=X.XXX med=X.XXX]  mean_meniscus_dice=X.XXXX")
        return

    print(f"Parsed {len(epochs)} epochs  (epoch {epochs[0]} -> {epochs[-1]})")
    print(f"Best val mean meniscus Dice: {max(val_mean):.4f} "
          f"at epoch {epochs[val_mean.index(max(val_mean))]}")
    if has_train_dice:
        print(f"Best train mean meniscus Dice: {max(tr_mean):.4f} "
              f"at epoch {epochs[tr_mean.index(max(tr_mean))]}")

    ncols = 3 if has_train_dice else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    run_name = os.path.basename(os.path.dirname(args.out_png))
    fig.suptitle(f"Fine-tuning Curves — {run_name}", fontsize=13, fontweight="bold")

    # ── Panel 1: Loss ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", color="tab:blue")
    ax.plot(epochs, val_loss,   label="Val Loss",   color="tab:orange")
    ax.set_title("Train vs Val Loss", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Panel 2: Train Dice (only if new log format) ──────────────────────
    if has_train_dice:
        ax = axes[1]
        ax.plot(epochs, tr_mean, label="Mean Meniscus (train)", color="tab:red",    linewidth=2)
        ax.plot(epochs, tr_lat,  label="Lateral (train)",       color="tab:green",  linestyle="--")
        ax.plot(epochs, tr_med,  label="Medial (train)",        color="tab:purple", linestyle="--")
        ax.plot(epochs, tr_bg,   label="Background (train)",    color="tab:gray",   linestyle=":")
        ax.axhline(max(tr_mean), color="tab:red", linestyle=":", alpha=0.5,
                   label=f"Best={max(tr_mean):.4f}")
        ax.set_title("Train Dice per Class", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Dice Score")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 3: Val Dice ─────────────────────────────────────────────────
    ax = axes[ncols - 1]
    ax.plot(epochs, val_mean, label="Mean Meniscus (val)", color="tab:red",    linewidth=2)
    ax.plot(epochs, val_lat,  label="Lateral (val)",       color="tab:green",  linestyle="--")
    ax.plot(epochs, val_med,  label="Medial (val)",        color="tab:purple", linestyle="--")
    ax.plot(epochs, val_bg,   label="Background (val)",    color="tab:gray",   linestyle=":")
    ax.axhline(max(val_mean), color="tab:red", linestyle=":", alpha=0.5,
               label=f"Best={max(val_mean):.4f}")
    ax.set_title("Val Dice per Class", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    plt.savefig(args.out_png, dpi=150)
    print(f"Saved -> {args.out_png}")


if __name__ == "__main__":
    main()
