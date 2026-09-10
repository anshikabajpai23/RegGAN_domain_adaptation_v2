"""
plot_replica_diagnostics.py
===========================
Turns the instrumented replica run's CSVs into diagnostic plots.

Reads:
  <run_dir>/history.csv                      (per-epoch curves)
  <run_dir>/val_predictions/per_slice_dice.csv  (where errors sit, per slice)

Writes:
  <out_dir>/diagnostics_curves.png     4 panels: overfit, domain gap, selection
  <out_dir>/diagnostics_per_slice.png  3 panels: where the Dice is actually lost

Usage (local, after scp'ing the run dir down):
  python scripts/plot_replica_diagnostics.py \
      --run_dir segmentation_runs/run_v7_replica_seed42
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows or None


def col(rows, name, cast=float):
    out = []
    for r in rows:
        v = r.get(name, "")
        try:
            out.append(cast(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.array(out, dtype=float)


def plot_curves(hist, out_path):
    ep      = col(hist, "epoch", int)
    loss_tr = col(hist, "loss_tr")
    loss_va = col(hist, "loss_va")
    d_rtr   = col(hist, "dice_real_tr")
    d_rva   = col(hist, "dice_real_va")
    d_ftr   = col(hist, "dice_fake_tr")
    d_fva   = col(hist, "dice_fake_va")
    pp_mean = col(hist, "dice_va_perpatient_mean")

    pat_cols = [k for k in hist[0].keys()
                if k.startswith("dice_va_") and k != "dice_va_perpatient_mean"]

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # A — loss. The classic overfitting signature.
    a = ax[0, 0]
    a.plot(ep, loss_tr, "o-", label="train loss", color="#2563eb")
    a.plot(ep, loss_va, "s-", label="val loss",   color="#dc2626")
    a.set_title("A. Loss — train falling while val flat/rising = overfitting")
    a.set_xlabel("epoch"); a.set_ylabel("loss"); a.legend(); a.grid(alpha=.3)

    # B — real-PD Dice, train vs val, gap shaded.
    b = ax[0, 1]
    b.plot(ep, d_rtr, "o-", label="train Dice (real PD)", color="#2563eb")
    b.plot(ep, d_rva, "s-", label="val Dice (real PD)",   color="#dc2626")
    b.fill_between(ep, d_rva, d_rtr, alpha=.15, color="#dc2626",
                   label="generalization gap")
    if len(ep):
        b.annotate(f"gap = {d_rtr[-1] - d_rva[-1]:+.3f}",
                   xy=(ep[-1], (d_rtr[-1] + d_rva[-1]) / 2),
                   xytext=(-110, 0), textcoords="offset points",
                   fontsize=11, fontweight="bold",
                   bbox=dict(boxstyle="round", fc="#fef3c7", ec="#d97706"))
    b.set_title("B. Real-PD Dice — the gap you could never see before")
    b.set_xlabel("epoch"); b.set_ylabel("Dice"); b.legend(); b.grid(alpha=.3)

    # C — domain gap: held-out fake vs held-out real.
    c = ax[1, 0]
    c.plot(ep, d_fva, "^-", label="val Dice (FAKE PD)", color="#7c3aed")
    c.plot(ep, d_rva, "s-", label="val Dice (REAL PD)", color="#dc2626")
    c.set_title("C. Domain gap — if these overlap, RegGAN realism is NOT\n"
                "the binding constraint on held-out data")
    c.set_xlabel("epoch"); c.set_ylabel("Dice"); c.legend(); c.grid(alpha=.3)

    # D — what ckpt_best is actually selected on.
    d = ax[1, 1]
    for pc in pat_cols:
        d.plot(ep, col(hist, pc), "-", alpha=.75, lw=1.4,
               label=pc.replace("dice_va_", ""))
    d.plot(ep, pp_mean, "k-", lw=2.4, label="clean per-patient mean")
    d.plot(ep, d_rva, "--", lw=2.2, color="#dc2626",
           label="pooled `vd`  <- ckpt_best uses THIS")
    if len(ep):
        best_i = int(np.nanargmax(d_rva))
        d.axvline(ep[best_i], color="#059669", ls=":", lw=2)
        d.annotate(f"selected epoch {ep[best_i]}", xy=(ep[best_i], np.nanmin(pp_mean)),
                   rotation=90, fontsize=9, color="#059669", va="bottom")
    d.set_title("D. Checkpoint selection — pooled vs clean per-patient")
    d.set_xlabel("epoch"); d.set_ylabel("Dice")
    d.legend(fontsize=8); d.grid(alpha=.3)

    fig.suptitle("Replica run diagnostics — v7/v8 loss, instrumented",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_per_slice(rows, out_path):
    pids  = sorted({r["patient_id"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # A — Dice along the slice axis. Reveals horn-tip / tapering failure.
    for pid in pids:
        sub = sorted([r for r in rows if r["patient_id"] == pid],
                     key=lambda r: int(r["slice_idx"]))
        ax[0].plot([int(r["slice_idx"]) for r in sub],
                   [float(r["dice"]) for r in sub], "o-", ms=3, lw=1.2, label=pid)
    ax[0].set_title("A. Dice per slice\n(dips at the ends = horn-tip failure)")
    ax[0].set_xlabel("slice index"); ax[0].set_ylabel("Dice")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    # B — distribution. Bimodal => whole-slice misses, not boundary jitter.
    dices = np.array([float(r["dice"]) for r in rows])
    ax[1].hist(dices, bins=28, color="#2563eb", alpha=.8, edgecolor="white")
    ax[1].axvline(dices.mean(), color="#dc2626", ls="--", lw=2,
                  label=f"mean {dices.mean():.3f}")
    ax[1].set_title("B. Per-slice Dice distribution\n(long left tail = a few bad slices)")
    ax[1].set_xlabel("Dice"); ax[1].set_ylabel("# slices")
    ax[1].legend(); ax[1].grid(alpha=.3)

    # C — is the error missing meniscus, or inventing it?
    fn = np.array([float(r["false_neg"]) for r in rows])
    fp = np.array([float(r["false_pos"]) for r in rows])
    gt = np.array([float(r["gt_voxels"]) for r in rows])
    ax[2].scatter(gt, fn, s=14, alpha=.6, label="false negatives (missed)",
                  color="#dc2626")
    ax[2].scatter(gt, fp, s=14, alpha=.6, label="false positives (invented)",
                  color="#7c3aed")
    lim = max(gt.max(), 1)
    ax[2].plot([0, lim], [0, lim], "k--", lw=1, alpha=.5, label="= all of GT")
    ax[2].set_title(f"C. Error type vs meniscus size\n"
                    f"total FN={fn.sum():,.0f}  FP={fp.sum():,.0f}")
    ax[2].set_xlabel("GT meniscus voxels in slice"); ax[2].set_ylabel("error voxels")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.suptitle("Where the Dice is actually lost (best checkpoint, val patients)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    hist = read_csv(os.path.join(args.run_dir, "history.csv"))
    if hist:
        plot_curves(hist, os.path.join(out_dir, "diagnostics_curves.png"))
        ep = col(hist, "epoch", int)
        d_rtr, d_rva = col(hist, "dice_real_tr"), col(hist, "dice_real_va")
        d_fva, pp    = col(hist, "dice_fake_va"), col(hist, "dice_va_perpatient_mean")
        i = int(np.nanargmax(d_rva))
        print("\n--- summary ---")
        print(f"  best pooled val Dice   : {d_rva[i]:.4f}  (epoch {ep[i]})")
        print(f"  train Dice at that ep  : {d_rtr[i]:.4f}")
        print(f"  generalization gap     : {d_rtr[i] - d_rva[i]:+.4f}")
        print(f"  clean per-patient mean : {pp[i]:.4f}   "
              f"(pooled overstates by {d_rva[i] - pp[i]:+.4f})")
        print(f"  fake-PD val Dice       : {d_fva[i]:.4f}   "
              f"(vs real-PD val {d_rva[i]:.4f})")
    else:
        print(f"no history.csv under {args.run_dir}")

    ps = read_csv(os.path.join(args.run_dir, "val_predictions", "per_slice_dice.csv"))
    if ps:
        plot_per_slice(ps, os.path.join(out_dir, "diagnostics_per_slice.png"))
    else:
        print("no per_slice_dice.csv found")


if __name__ == "__main__":
    main()
