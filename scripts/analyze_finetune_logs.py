"""
analyze_finetune_logs.py
========================
Parses all v3 finetune logs and prints a summary table:
  - Total epochs run
  - Early stopping epoch
  - Best val Dice (on fake PD val set)
  - Train Dice at best epoch
  - Train/Val gap (overfitting indicator)
  - Diagnosis

Usage (on BigRed):
    python scripts/analyze_finetune_logs.py \
        --log_dir /N/project/prostate_cancer_ai/anshika/regGAN/logs \
        --prefix ft_run002v2_v3
"""
import argparse
import glob
import os
import re

EPOCH_RE  = re.compile(r"Epoch\s+(\d+).*train=([\d.]+).*val=([\d.]+).*val_dice\[lat=([\d.]+)\s+med=([\d.]+)\]")
BEST_RE   = re.compile(r"New best:\s*([\d.]+)")
STOP_RE   = re.compile(r"Early stopping at epoch\s+(\d+)\.\s+Best:\s*([\d.]+)")
DONE_RE   = re.compile(r"Done\.\s*Best val Dice:\s*([\d.]+)")


def parse_log(path):
    epochs = []
    best_val = None
    stop_epoch = None
    final_best = None

    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                ep   = int(m.group(1))
                tl   = float(m.group(2))
                vl   = float(m.group(3))
                lat  = float(m.group(4))
                med  = float(m.group(5))
                mean_dice = (lat + med) / 2
                epochs.append({"ep": ep, "train_loss": tl, "val_loss": vl,
                                "val_dice": mean_dice, "lat": lat, "med": med})

            m = BEST_RE.search(line)
            if m:
                best_val = float(m.group(1))

            m = STOP_RE.search(line)
            if m:
                stop_epoch = int(m.group(1))
                final_best = float(m.group(2))

            m = DONE_RE.search(line)
            if m:
                final_best = float(m.group(1))

    return epochs, best_val, stop_epoch, final_best


def diagnose(epochs, stop_epoch, final_best, total_epochs=50):
    if not epochs:
        return "NO LOG DATA"

    last = epochs[-1]
    best_ep = stop_epoch if stop_epoch else len(epochs) - 1

    # train dice at best epoch (approximate — use last epoch with best val)
    best_epoch_data = None
    if stop_epoch is not None:
        candidates = [e for e in epochs if e["ep"] <= stop_epoch]
        if candidates:
            best_epoch_data = max(candidates, key=lambda x: x["val_dice"])
    else:
        best_epoch_data = max(epochs, key=lambda x: x["val_dice"])

    notes = []

    # Stopped early?
    if stop_epoch and stop_epoch < total_epochs - 5:
        notes.append(f"early stop ep{stop_epoch}/{total_epochs}")
    else:
        notes.append(f"ran full {len(epochs)} epochs")

    # Overfitting: train loss much lower than val loss
    if best_epoch_data:
        train_loss = best_epoch_data["train_loss"]
        val_loss   = best_epoch_data["val_loss"]
        if val_loss > train_loss * 1.5:
            notes.append("OVERFITTING (val>1.5×train loss)")
        elif val_loss < train_loss * 0.9:
            notes.append("underfitting (val<train loss)")

    # Still improving at end?
    if not stop_epoch and len(epochs) >= total_epochs:
        last5_dice = [e["val_dice"] for e in epochs[-5:]]
        if last5_dice[-1] > last5_dice[0]:
            notes.append("STILL IMPROVING → needs more epochs")
        else:
            notes.append("plateaued")

    return " | ".join(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--prefix", default="ft_run002v2_v3")
    args = ap.parse_args()

    # Python logging goes to stderr → .err files; bash echo goes to .out
    pattern = os.path.join(args.log_dir, f"{args.prefix}_*.err")
    log_files = sorted(glob.glob(pattern))

    if not log_files:
        print(f"No logs found matching: {pattern}")
        return

    print(f"\n{'Experiment':<25} {'Epochs':>7} {'StopEp':>7} {'BestValDice':>12} {'Diagnosis'}")
    print("-" * 100)

    for path in log_files:
        fname = os.path.basename(path)
        # Extract experiment name from filename
        exp = fname.replace(args.prefix + "_", "").rsplit("_", 1)[0]
        # Clean up: remove job id suffix
        exp = re.sub(r"_\d+$", "", exp)

        epochs, best_val, stop_epoch, final_best = parse_log(path)
        diag = diagnose(epochs, stop_epoch, final_best)

        best_str = f"{final_best:.4f}" if final_best else ("N/A" if not best_val else f"{best_val:.4f}")
        stop_str = str(stop_epoch) if stop_epoch else f"{len(epochs)}"

        print(f"{exp:<25} {len(epochs):>7} {stop_str:>7} {best_str:>12}   {diag}")

    print()
    print("── Per-experiment epoch-level detail ──")
    for path in log_files:
        fname = os.path.basename(path)
        exp = re.sub(r"_\d+\.out$", "", fname.replace(args.prefix + "_", ""))
        epochs, best_val, stop_epoch, final_best = parse_log(path)

        if not epochs:
            print(f"\n{exp}: NO DATA")
            continue

        print(f"\n{exp}:")
        print(f"  {'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValDice(mean)':>14}  {'lat':>6}  {'med':>6}")
        for e in epochs:
            marker = " ← best" if e["val_dice"] == max(x["val_dice"] for x in epochs) else ""
            print(f"  {e['ep']:>6}  {e['train_loss']:>10.4f}  {e['val_loss']:>10.4f}  "
                  f"{e['val_dice']:>14.4f}  {e['lat']:>6.3f}  {e['med']:>6.3f}{marker}")

        if stop_epoch:
            print(f"  → Early stopping at epoch {stop_epoch}, best val Dice = {final_best:.4f}")


if __name__ == "__main__":
    main()
