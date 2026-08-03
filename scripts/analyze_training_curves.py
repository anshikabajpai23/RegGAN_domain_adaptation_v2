"""
analyze_training_curves.py
=============================
Reads the TensorBoard event file from a training run and answers:
"has training plateaued, or is it still improving -- do we need more epochs?"

Extracts the key scalars logged by train.py:
  - G/total              (generator loss, per training step)
  - D/total               (discriminator loss, per training step)
  - val/L1_unpaired_proxy (validation proxy, per epoch)
  - R/mean_magnitude       (registration flow magnitude, per training step)

Plots all four over training progress, and gives a simple heuristic verdict:
compares the trend over the LAST 20% of logged steps/epochs against the
trend over the PRECEDING 20% -- if the recent slope is still meaningfully
negative (for losses) the run likely benefits from more epochs; if it's
flattened out (near-zero slope), it's plateaued.

Usage:
    python scripts/analyze_training_curves.py \
        --tb_dir runs/run_004/tb \
        --out_png runs/run_004/training_curves.png
"""
import argparse
import glob
import os

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise SystemExit("Needs the 'tensorboard' package (already a dependency of "
                      "torch.utils.tensorboard, used elsewhere in this project).")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_scalar(tb_dir, tag):
    event_files = sorted(glob.glob(os.path.join(tb_dir, "events.out.tfevents.*")))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {tb_dir}")

    steps, values = [], []
    for ef in event_files:
        ea = EventAccumulator(ef, size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for e in ea.Scalars(tag):
            steps.append(e.step)
            values.append(e.value)

    if not steps:
        return None, None

    order = np.argsort(steps)
    return np.array(steps)[order], np.array(values)[order]


def trend_verdict(values, label, lower_is_better=True):
    """Compare slope over the last 20% of points vs the preceding 20%."""
    n = len(values)
    if n < 10:
        return f"{label}: not enough data points ({n}) for a trend verdict"

    chunk = max(n // 5, 2)
    recent = values[-chunk:]
    prior = values[-2 * chunk:-chunk]

    recent_mean, prior_mean = recent.mean(), prior.mean()
    pct_change = (recent_mean - prior_mean) / (abs(prior_mean) + 1e-8) * 100

    improving = (pct_change < -1) if lower_is_better else (pct_change > 1)
    plateaued = abs(pct_change) < 1

    if plateaued:
        verdict = "PLATEAUED -- more epochs unlikely to help much without other changes"
    elif improving:
        verdict = "STILL IMPROVING -- more epochs likely to help"
    else:
        verdict = "GETTING WORSE -- check for instability/overfitting"

    return f"{label}: last-20%-vs-prior-20% change = {pct_change:+.2f}%  ->  {verdict}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb_dir", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--zoom_after", type=int, default=2000,
                    help="Zoom panel shows only steps >= this value (default: 2000)")
    args = ap.parse_args()

    tags = {
        "G/total": "Generator loss",
        "D/total": "Discriminator loss",
        "val/L1_unpaired_proxy": "Validation L1 (unpaired proxy)",
        "R/mean_magnitude": "Registration flow magnitude",
    }

    # 2 rows × 4 cols: top = full run, bottom = zoomed after --zoom_after
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.out_png)))
    fig.suptitle(f"RegGAN Training Curves — {run_name}  "
                 f"(bottom row: zoomed after step {args.zoom_after})",
                 fontsize=13, fontweight="bold")

    print("=" * 70)
    print(" TRAINING CURVE ANALYSIS")
    print("=" * 70)

    for col, (tag, label) in enumerate(tags.items()):
        steps, values = load_scalar(args.tb_dir, tag)

        is_epoch_metric = "val/" in tag  # val L1 logged by epoch, not global step

        for row in range(2):
            ax = axes[row][col]
            xlabel = "epoch" if is_epoch_metric else "step"

            if steps is None:
                ax.set_title(f"{label}\n(no data for '{tag}')")
                continue

            s, v = steps, values
            if row == 1 and not is_epoch_metric:  # zoom only applies to step-based metrics
                mask = s >= args.zoom_after
                s, v = s[mask], v[mask]
                if len(s) == 0:
                    ax.set_title(f"{label}\n(no data after step {args.zoom_after})")
                    continue

            # smooth with rolling mean for readability
            window = max(1, len(v) // 100)
            v_smooth = np.convolve(v, np.ones(window) / window, mode="valid")
            s_smooth = s[:len(v_smooth)]

            ax.plot(s, v, alpha=0.2, linewidth=0.5, color="steelblue")
            ax.plot(s_smooth, v_smooth, linewidth=1.8, color="steelblue")
            if row == 0:
                title = label
            elif is_epoch_metric:
                title = f"{label}\n(all epochs)"
            else:
                title = f"{label}\n(zoom ≥ {args.zoom_after})"
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("value")
            ax.grid(alpha=0.3)

        if steps is not None:
            print(f"\n{label} ({len(values)} points, last={values[-1]:.5f}):")
            print(f"  {trend_verdict(values, label)}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png, dpi=150)
    print(f"\nSaved plot -> {args.out_png}")
    print("\nRule of thumb: if G/total and val/L1 are still trending down (not")
    print("plateaued), more epochs should keep helping. If they've flattened,")
    print("additional epochs at the current LR likely won't improve much --")
    print("would need the LR-decay phase to kick in (epoch > total_epochs/2)")
    print("or other changes (Stage 3 R-network fix, more data, etc.)")


if __name__ == "__main__":
    main()
