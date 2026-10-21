"""
synthetic_log_utils.py
======================
Turns the v15 training logs (written by TearPatternDataset while logging is on) into review
material. Called automatically by finetune_meniscus_v15_tear_patterns.py after each logged
epoch; can also be run by hand:

    python segmentation/synthetic_log_utils.py <run_dir>/synthetic_log/epoch000

Writes into that epoch folder:
  stats_all.csv               every synthetic sample of the epoch (merged from all workers)
  summary_per_mode.csv        per pattern: count, share, label/piece/intensity statistics
  summary_histograms.png      distributions per pattern (label change, pieces, intensities)
  nifti/<mode>_<source>_{image_before,image_after,label_before,label_after}.nii.gz
                              the saved samples stacked along axis 0 (CENTRE slice of each
                              sample, one sample per "slice"), for scrolling in Slicer/ITK-SNAP;
                              images are int16 = intensity x 1000, labels are 0/1
  nifti/<mode>_<source>_index.csv   which sample is at which position of the stack
"""
import csv
import glob
import os
import sys

import numpy as np


def merge_stats(epoch_dir):
    import pandas as pd
    files = sorted(glob.glob(os.path.join(epoch_dir, "stats_*_w*_p*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(epoch_dir, "stats_all.csv"), index=False)
    return df


def summarize(epoch_dir):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = merge_stats(epoch_dir)
    if df is None or df.empty:
        print(f"no stats in {epoch_dir}")
        return
    df["label_change_px"] = df["meniscus_px_after"] - df["meniscus_px_before"]
    df["pieces_change"] = df["pieces_after"] - df["pieces_before"]
    df["int_change_meniscus"] = df["mean_int_meniscus_after"] - df["mean_int_meniscus_before"]
    g = df.groupby(["mode", "source"])
    summ = g.agg(n=("mode", "size"),
                 label_change_px_mean=("label_change_px", "mean"),
                 removed_px_mean=("removed_px", "mean"), added_px_mean=("added_px", "mean"),
                 pieces_before_mean=("pieces_before", "mean"), pieces_after_mean=("pieces_after", "mean"),
                 share_more_pieces=("pieces_change", lambda x: float((x > 0).mean())),
                 share_fewer_pieces=("pieces_change", lambda x: float((x < 0).mean())),
                 image_changed_px_mean=("image_changed_px", "mean"),
                 int_meniscus_before=("mean_int_meniscus_before", "mean"),
                 int_meniscus_after=("mean_int_meniscus_after", "mean"),
                 int_changed_after=("mean_int_changed_after", "mean")).reset_index()
    summ["share_of_all"] = summ["n"] / summ["n"].sum()
    summ.round(4).to_csv(os.path.join(epoch_dir, "summary_per_mode.csv"), index=False)

    modes = sorted(df["mode"].unique())
    cols = [("label_change_px", "label size change (px)"), ("pieces_change", "change in number of pieces"),
            ("image_changed_px", "image pixels changed"), ("mean_int_changed_after", "intensity of changed pixels (after)")]
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 3.2 * len(cols)))
    for ax, (c, title) in zip(axes, cols):
        data = [df.loc[df["mode"] == m, c].dropna().values for m in modes]
        ax.boxplot(data, labels=modes, showfliers=False)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{os.path.basename(epoch_dir)}: {len(df)} synthetic samples, all patterns", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(epoch_dir, "summary_histograms.png"), dpi=100); plt.close(fig)
    print(f"{epoch_dir}: {len(df)} samples summarised")


def build_nifti_stacks(epoch_dir):
    import nibabel as nib
    out = os.path.join(epoch_dir, "nifti")
    os.makedirs(out, exist_ok=True)
    for mode_dir in sorted(glob.glob(os.path.join(epoch_dir, "samples", "*"))):
        mode = os.path.basename(mode_dir)
        for source in ("fake", "real"):
            files = sorted(glob.glob(os.path.join(mode_dir, f"{source}_*.npz")))
            if not files:
                continue
            stacks = {"image_before": [], "image_after": [], "label_before": [], "label_after": []}
            with open(os.path.join(out, f"{mode}_{source}_index.csv"), "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["stack_position", "file", "pid", "slice"])
                for k, f in enumerate(files):
                    d = np.load(f)
                    # images stored as int16 = intensity x 1000 (0..1000) to keep downloads small
                    stacks["image_before"].append(np.round(d["image_before"][1].astype(np.float32) * 1000).astype(np.int16))
                    stacks["image_after"].append(np.round(d["image_after"][1].astype(np.float32) * 1000).astype(np.int16))
                    stacks["label_before"].append((d["mask_before"] > 0).astype(np.int16))
                    stacks["label_after"].append((d["mask_after"] > 0).astype(np.int16))
                    w.writerow([k, os.path.relpath(f, epoch_dir), str(d["pid"]), int(d["slice"])])
            aff = np.diag([1.0, 0.39, 0.39, 1.0])
            for name, arr in stacks.items():
                nib.save(nib.Nifti1Image(np.stack(arr, 0), aff), os.path.join(out, f"{mode}_{source}_{name}.nii.gz"))
    print(f"NIfTI stacks -> {out}")


def process_epoch(epoch_dir):
    summarize(epoch_dir)
    build_nifti_stacks(epoch_dir)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        process_epoch(d)
