"""
plot_finetune_curves.py
========================
Parses finetune_meniscus.py stdout log and plots 3 panels:
  1. Train vs Val Loss
  2. Train vs Val Mean Meniscus Dice (overlaid — shows overfitting gap directly)
  3. Val Dice per class (lateral, medial, background)

Marks best epoch and early stopping point on all panels.
Works with both old log format (no train_dice) and new format (with train_dice).

Usage:
    python plot_finetune_curves.py \
        --log_file /N/.../logs/finetune_run003_JOBID.out \
        --out_png  /N/.../segmentation_runs/run_003/finetune_curves.png
"""
import argparse
import re
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_log(log_file):
    epochs = []
    train_loss, val_loss = [], []
    tr_bg, tr_lat, tr_med, tr_mean = [], [], [], []
    val_bg, val_lat, val_med, val_mean = [], [], [], []

    pattern_new = re.compile(
        r"Epoch (\d+)\s+"
        r"train_loss=([\d.]+)\s+"
        r"train_dice\[bg=([\d.]+) lat=([\d.]+) med=([\d.]+)\]\s+"
        r"mean_train_meniscus_dice=([\d.]+)\s+"
        r"val_loss=([\d.]+)\s+"
        r"val_dice\[bg=([\d.]+) lat=([\d.]+) med=([\d.]+)\]\s+"
        r"mean_meniscus_dice=([\d.]+)"
    )
    pattern_old = re.compile(
        r"Epoch (\d+)\s+"
        r"train_loss=([\d.]+)\s+"
        r"val_loss=([\d.]+)\s+"
        r"val_dice\[bg=([\d.]+) lat=([\d.]+) med=([\d.]+)\]\s+"
        r"mean_meniscus_dice=([\d.]+)"
    )

    early_stopped = False
    best_epoch = None

    with open(log_file) as f:
        for line in f:
            if "Early stopping triggered" in line:
                early_stopped = True
            if "New best mean meniscus dice" in line:
                if epochs:
                    best_epoch = epochs[-1]

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

    # Derive best epoch from val_mean if log parse didn't catch it
    if val_mean and best_epoch is None:
        best_epoch = epochs[val_mean.index(max(val_mean))]

    return (epochs, train_loss, val_loss,
            tr_bg, tr_lat, tr_med, tr_mean,
            val_bg, val_lat, val_med, val_mean,
            has_train_dice, early_stopped, best_epoch)


def add_epoch_markers(ax, best_epoch, last_epoch, early_stopped):
    """Draw vertical lines for best epoch and early stop epoch."""
    ax.axvline(best_epoch, color="gold", linewidth=1.5, linestyle="--",
               label=f"Best epoch ({best_epoch})")
    if early_stopped and last_epoch != best_epoch:
        ax.axvline(last_epoch, color="tomato", linewidth=1.5, linestyle=":",
                   label=f"Early stop ({last_epoch})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_file", required=True)
    ap.add_argument("--out_png",  required=True)
    args = ap.parse_args()

    (epochs, train_loss, val_loss,
     tr_bg, tr_lat, tr_med, tr_mean,
     val_bg, val_lat, val_med, val_mean,
     has_train_dice, early_stopped, best_epoch) = parse_log(args.log_file)

    if not epochs:
        print("No epoch data found in log. Check the log file path and format.")
        print("Run: grep 'Epoch' <logfile> | head -3  to verify format.")
        return

    last_epoch = epochs[-1]
    best_val   = max(val_mean)

    print(f"Parsed {len(epochs)} epochs  (epoch {epochs[0]} → {last_epoch})")
    print(f"Best val mean meniscus Dice: {best_val:.4f} at epoch {best_epoch}")
    print(f"Early stopping triggered: {early_stopped}")
    if has_train_dice:
        print(f"Best train mean meniscus Dice: {max(tr_mean):.4f} at epoch {epochs[tr_mean.index(max(tr_mean))]}")
        gap = max(tr_mean) - best_val
        print(f"Train-val Dice gap at best val epoch: {gap:+.4f} "
              f"({'overfitting' if gap > 0.05 else 'OK'})")

    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.out_png)))
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"Fine-tuning Curves — {run_name}   "
                 f"(best val Dice={best_val:.4f} @ epoch {best_epoch}"
                 f"{', early stopped @ ' + str(last_epoch) if early_stopped else ''})",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Train vs Val Loss ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, train_loss, color="tab:blue",   linewidth=2, label="Train Loss")
    ax.plot(epochs, val_loss,   color="tab:orange", linewidth=2, label="Val Loss")
    add_epoch_markers(ax, best_epoch, last_epoch, early_stopped)
    ax.set_title("Train vs Val Loss", fontweight="bold", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Panel 2: Train vs Val Mean Meniscus Dice (overfitting view) ──────────
    ax = axes[1]
    if has_train_dice:
        ax.plot(epochs, tr_mean, color="tab:blue",   linewidth=2,
                label="Train Mean Meniscus")
        ax.fill_between(epochs, tr_mean, val_mean,
                        where=[t > v for t, v in zip(tr_mean, val_mean)],
                        alpha=0.15, color="red", label="Overfit gap")
    ax.plot(epochs, val_mean, color="tab:orange", linewidth=2,
            label="Val Mean Meniscus")
    ax.axhline(best_val, color="gold", linewidth=1, linestyle=":",
               label=f"Best val={best_val:.4f}")
    add_epoch_markers(ax, best_epoch, last_epoch, early_stopped)
    ax.set_title("Train vs Val Meniscus Dice\n(gap = overfitting indicator)",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Panel 3: Val Dice per class ──────────────────────────────────────────
    ax = axes[2]
    ax.plot(epochs, val_mean, color="tab:red",    linewidth=2.5, label="Mean Meniscus (val)")
    ax.plot(epochs, val_lat,  color="tab:green",  linewidth=1.5, linestyle="--", label="Lateral (val)")
    ax.plot(epochs, val_med,  color="tab:purple", linewidth=1.5, linestyle="--", label="Medial (val)")
    ax.plot(epochs, val_bg,   color="tab:gray",   linewidth=1,   linestyle=":",  label="Background (val)")
    if has_train_dice:
        ax.plot(epochs, tr_lat, color="tab:green",  linewidth=1, linestyle="-.",
                alpha=0.5, label="Lateral (train)")
        ax.plot(epochs, tr_med, color="tab:purple", linewidth=1, linestyle="-.",
                alpha=0.5, label="Medial (train)")
    ax.axhline(best_val, color="gold", linewidth=1, linestyle=":",
               label=f"Best={best_val:.4f}")
    add_epoch_markers(ax, best_epoch, last_epoch, early_stopped)
    ax.set_title("Dice per Class (val + train dashed)",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.out_png}")


if __name__ == "__main__":
    main()
