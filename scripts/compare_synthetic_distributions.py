"""Compare synthetic-tear distributions across runs (v15 / v17 / v16 / v18) and against real torn GT.

Each --logs entry is NAME=PATH, where PATH is a synthetic_log epoch folder (or its stats_all.csv).
Crop-space runs (v16, v18) are stretched (crop box -> 384 x 384), so their pixel counts are divided by
the stretch area (--crop_runs NAME ...; factor read from crop_box.json) to be comparable with native
pixels. Piece counts and intensities need no correction. Real-PD source only by default.

Outputs (--out_dir):
  per_group.csv       per run: healthy (before) / synthetic (after), size, pieces, share 3+ pieces, intensity
  per_variant.csv     per run x mode (and x variant when logged, v17+): n, share, size, pieces, intensity
  ks_vs_real_torn.csv KS distance of each run's synthetic slices to the real torn GT (lower = closer)
  distributions.png   boxplots: size / pieces / intensity, healthy vs each run vs real torn
  mix.png             pattern mix per run (share of synthetic samples), variant level when available

    venv/bin/python scripts/compare_synthetic_distributions.py \
        --logs v15=results/v15_synthetic_log v17=results/v17_synthetic_log v16=results/v16_synthetic_log \
        --crop_runs v16 --out_dir figures/synthetic_distribution_compare
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.stats import ks_2samp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
TORN = ["AC13D01F99D88C", "AC116435D25299", "AC1CEC4C14F850", "AC152F0CF07CF3", "AC1A200F61EDC1"]
SEG_ROOT = Path("/Users/anshikabajpai/Desktop/delete/nikhil-segmentations 2")
RAW_DIRS = [SEG_ROOT, Path("/Users/anshikabajpai/Desktop/delete/aug 22")]


def load_stats(path):
    p = Path(path)
    f = p if p.suffix == ".csv" else p / "stats_all.csv"
    if not f.exists():
        import glob
        parts = sorted(glob.glob(str(p / "stats_*_w*_p*.csv")))
        if not parts:
            raise FileNotFoundError(f"no stats in {p}")
        return pd.concat([pd.read_csv(x) for x in parts], ignore_index=True)
    return pd.read_csv(f)


def real_torn():
    from torn_compare_utils import load_patient
    rows = []
    for p in TORN:
        d = load_patient(p, "results/torn_predictions_replica_seed42", "results/torn_predictions_crop_sequence", SEG_ROOT, RAW_DIRS)
        if d is None:
            continue
        for s in range(d["manual"].shape[0]):
            m = d["manual"][s]
            if m.sum() >= 20:
                rows.append({"px": int(m.sum()), "pieces": int(ndi.label(m)[1]), "intensity": float(d["vol"][s][m].mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="NAME=PATH")
    ap.add_argument("--crop_runs", nargs="*", default=[], help="names whose logs are in crop space (v16, v18)")
    ap.add_argument("--box_json", default=os.path.join(ROOT, "crop_box.json"))
    ap.add_argument("--source", default="real", choices=["real", "fake", "both"])
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "figures/synthetic_distribution_compare"))
    ap.add_argument("--no_torn", action="store_true", help="skip the real torn GT (e.g. GT not available)")
    a = ap.parse_args()
    os.chdir(ROOT)
    b = json.load(open(a.box_json))
    area = (384 / (b["y1"] - b["y0"])) * (384 / (b["x1"] - b["x0"]))
    os.makedirs(a.out_dir, exist_ok=True)

    groups, runs = [], {}
    for item in a.logs:
        name, path = item.split("=", 1)
        st = load_stats(path)
        if a.source != "both":
            st = st[st.source == a.source]
        f = area if name in a.crop_runs else 1.0
        st = st.assign(px_b=st.meniscus_px_before / f, px_a=st.meniscus_px_after / f)
        if "variant" not in st.columns:
            st["variant"] = st["mode"]
        runs[name] = st
        groups.append(pd.DataFrame({"group": f"{name} healthy (before)", "px": st.px_b, "pieces": st.pieces_before,
                                    "intensity": st.mean_int_meniscus_before}))
        after = st[st.px_a >= 20]
        groups.append(pd.DataFrame({"group": f"{name} synthetic (after)", "px": after.px_a, "pieces": after.pieces_after,
                                    "intensity": after.mean_int_meniscus_after}))
    torn = None if a.no_torn else real_torn()
    if torn is not None and len(torn):
        groups.append(torn.assign(group="real torn (GT)"))
    allg = pd.concat(groups, ignore_index=True)

    per_group = allg.groupby("group", sort=False).agg(
        n=("px", "size"), px_median=("px", "median"), pieces_mean=("pieces", "mean"),
        share_3plus_pieces=("pieces", lambda x: float((x >= 3).mean())), intensity_median=("intensity", "median")).round(3)
    per_group.to_csv(os.path.join(a.out_dir, "per_group.csv"))
    print(per_group.to_string())

    pv = []
    for name, st in runs.items():
        g = st.groupby(["mode", "variant"]).agg(n=("mode", "size"), px_after_median=("px_a", "median"),
                                                pieces_after_mean=("pieces_after", "mean"),
                                                int_after_median=("mean_int_meniscus_after", "median")).reset_index()
        g.insert(0, "run", name); g["share"] = g["n"] / g["n"].sum()
        pv.append(g)
    pv = pd.concat(pv, ignore_index=True).round(3)
    pv.to_csv(os.path.join(a.out_dir, "per_variant.csv"), index=False)

    if torn is not None and len(torn):
        ks = []
        for name, st in runs.items():
            after = st[st.px_a >= 20]
            ks.append({"group": f"{name} synthetic", **{c: ks_2samp(v, torn[c]).statistic for c, v in
                       (("px", after.px_a), ("pieces", after.pieces_after), ("intensity", after.mean_int_meniscus_after))}})
            ks.append({"group": f"{name} healthy", **{c: ks_2samp(v, torn[c]).statistic for c, v in
                       (("px", st.px_b), ("pieces", st.pieces_before), ("intensity", st.mean_int_meniscus_before))}})
        ks = pd.DataFrame(ks).round(3)
        ks.to_csv(os.path.join(a.out_dir, "ks_vs_real_torn.csv"), index=False)
        print("\nKS distance to real torn GT (lower = closer):\n" + ks.to_string(index=False))

    order = list(dict.fromkeys(allg.group))
    fig, axes = plt.subplots(1, 3, figsize=(max(12, 2.2 * len(order) * 1.0), 4.8))
    for ax, (c, t) in zip(axes, [("px", "meniscus pixels per slice (native px)"), ("pieces", "pieces per slice"),
                                 ("intensity", "mean meniscus intensity")]):
        ax.boxplot([allg.loc[allg.group == g, c].dropna() for g in order], showfliers=False)
        ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([g.replace(" (", "\n(") for g in order], fontsize=7, rotation=30)
        ax.set_title(t, fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Synthetic tear distributions vs real torn GT ({a.source} PD source)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "distributions.png"), dpi=110); plt.close(fig)

    piv = pv.pivot_table(index="variant", columns="run", values="share", fill_value=0)
    piv = piv.drop(index="none", errors="ignore")
    ax = piv.plot.barh(figsize=(9, 0.35 * len(piv) + 2))
    ax.set_xlabel("share of synthetic samples"); ax.set_title("Pattern / variant mix per run"); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "mix.png"), dpi=110); plt.close()
    print(f"\nsaved -> {a.out_dir}")


if __name__ == "__main__":
    main()
