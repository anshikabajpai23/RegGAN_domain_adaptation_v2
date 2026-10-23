"""UMAP + quantitative check: do v15 synthetic tears look like real torn menisci to the network?

Each slice becomes one feature vector: the replica encoder's features (stages 3, 4, 5) averaged over
the meniscus region (mask dilated by a few pixels, so the immediate surroundings count too).
Groups:
  healthy        = synthetic-log samples BEFORE the tear (real-PD training slices by default)
  synth:<mode>   = the same samples AFTER the tear, per pattern
  real torn      = your torn patients' GT slices (a 3-slice stack around each GT slice)
Outputs (in --out_dir):
  umap_by_group.png      UMAP coloured by group
  umap_by_source.png     UMAP coloured by patient source (shows patient/scanner clustering)
  closeness.csv          per group: mean distance to the nearest real-torn slice; two-sample AUC vs real torn
                         (logistic regression, 5-fold CV, balanced; 0.5 = indistinguishable, 1.0 = trivially separable)

CAVEAT: real torn slices come from 5 patients, synthetic ones from the training patients, so patient and
scanner differences also separate them. Compare groups RELATIVE to each other (does synth move toward real
torn compared with healthy?), not the absolute AUC.

    venv/bin/python scripts/umap_synthetic_vs_torn.py --log_dir results/v15_synthetic_log --out_dir figures/v15_umap
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "segmentation"))
from torn_compare_utils import load_patient    # noqa: E402
from synthetic_log_utils import load_sample    # noqa: E402

TORN = ["AC13D01F99D88C", "AC116435D25299", "AC1CEC4C14F850", "AC152F0CF07CF3", "AC1A200F61EDC1"]
SEG_ROOT = Path("/Users/anshikabajpai/Desktop/delete/nikhil-segmentations 2")
RAW_DIRS = [SEG_ROOT, Path("/Users/anshikabajpai/Desktop/delete/aug 22")]
STAGES = (3, 4, 5)


def load_encoder(ckpt, device):
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
    state = torch.load(ckpt, map_location=device)
    state = {k: v for k, v in state.items() if not k.startswith(("visible_head", "contour_head"))}
    m.load_state_dict(state)
    return m.encoder.to(device).eval()


@torch.no_grad()
def embed(encoder, img3, mask, device, dilate=4):
    """img3: (3, H, W) float in [0,1]; mask: (H, W) bool -> 1-D feature vector."""
    feats = encoder(torch.from_numpy(img3[None].astype(np.float32)).to(device))
    region = ndi.binary_dilation(mask, iterations=dilate) if dilate else mask
    r = torch.from_numpy(region.astype(np.float32))[None, None].to(device)
    out = []
    for k in STAGES:
        f = feats[k]
        rr = torch.nn.functional.adaptive_max_pool2d(r, f.shape[-2:])     # keep tiny regions alive
        w = rr.sum().clamp(min=1.0)
        out.append(((f * rr).sum((-2, -1)) / w)[0].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="results/v15_synthetic_log")
    ap.add_argument("--ckpt", default="segmentation_runs/run_v7_replica_seed42/ckpt_best.pth",
                    help="feature extractor: the replica (default) or e.g. a v15 checkpoint")
    ap.add_argument("--source", default="real", choices=["real", "fake", "both"])
    ap.add_argument("--out_dir", default="figures/v15_umap")
    ap.add_argument("--max_per_group", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.chdir(ROOT)
    device = torch.device("cpu")
    enc = load_encoder(a.ckpt, device)
    rng = np.random.default_rng(a.seed)
    rows, X = [], []

    # synthetic samples (before = healthy, after = per pattern)
    for mdir in sorted(glob.glob(os.path.join(a.log_dir, "samples", "*"))):
        mode = os.path.basename(mdir)
        files = sorted(glob.glob(os.path.join(mdir, "*.npz")))
        if a.source != "both":
            files = [f for f in files if os.path.basename(f).startswith(a.source + "_")]
        files = [files[i] for i in rng.permutation(len(files))[:a.max_per_group]]
        for f in files:
            try:
                d = load_sample(f)
                ib, ia = d["image_before"].astype(np.float32), d["image_after"].astype(np.float32)
                mb, ma = d["mask_before"] > 0, d["mask_after"] > 0
                pid = str(d["pid"])
            except Exception:
                continue
            if mb.sum() >= 20:
                X.append(embed(enc, ib, mb, device)); rows.append({"group": "healthy", "source": "train " + pid})
            if ma.sum() >= 20:
                X.append(embed(enc, ia, ma, device)); rows.append({"group": f"synth:{mode}", "source": "train " + pid})

    # real torn, GT slices
    for p in TORN:
        dd = load_patient(p, "results/torn_predictions_replica_seed42", "results/torn_predictions_crop_sequence", SEG_ROOT, RAW_DIRS)
        if dd is None:
            continue
        vol, gt = dd["vol"], dd["manual"]
        n = vol.shape[0]
        for s in range(n):
            if gt[s].sum() < 20:
                continue
            img3 = np.stack([vol[max(0, min(n - 1, s + o))] for o in (-1, 0, 1)])
            X.append(embed(enc, img3, gt[s], device)); rows.append({"group": "real torn", "source": "torn " + p})

    X = np.asarray(X)
    df = pd.DataFrame(rows)
    print(df.group.value_counts().to_string())
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X)

    # closeness to real torn (nearest-neighbour distance) and two-sample AUC
    from sklearn.neighbors import NearestNeighbors
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    torn_idx = np.where(df.group == "real torn")[0]
    nn = NearestNeighbors(n_neighbors=1).fit(Z[torn_idx])
    res = []
    for g in sorted(df.group.unique()):
        if g == "real torn":
            continue
        gi = np.where(df.group == g)[0]
        dist = nn.kneighbors(Z[gi])[0][:, 0]
        k = min(len(gi), len(torn_idx))
        a_i = rng.choice(gi, k, replace=False); b_i = rng.choice(torn_idx, k, replace=False)
        Xi = np.vstack([Z[a_i], Z[b_i]]); yi = np.r_[np.zeros(k), np.ones(k)]
        auc = cross_val_score(LogisticRegression(max_iter=2000, C=0.1), Xi, yi, cv=5, scoring="roc_auc").mean() if k >= 10 else np.nan
        res.append({"group": g, "n": len(gi), "mean_dist_to_nearest_real_torn": dist.mean(), "auc_vs_real_torn": auc})
    res = pd.DataFrame(res).sort_values("mean_dist_to_nearest_real_torn")
    os.makedirs(a.out_dir, exist_ok=True)
    res.round(3).to_csv(os.path.join(a.out_dir, "closeness.csv"), index=False)
    print("\n" + res.round(3).to_string(index=False))

    import umap
    emb = umap.UMAP(n_neighbors=15, min_dist=0.15, random_state=a.seed).fit_transform(Z)
    groups = ["healthy"] + sorted(g for g in df.group.unique() if g.startswith("synth:")) + ["real torn"]
    colors = {"healthy": "#9aa0a6", "real torn": "#d62728"}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for i, g in enumerate(groups[1:-1]):
        colors[g] = palette[i % len(palette)]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for g in groups:
        m = (df.group == g).values
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=40 if g == "real torn" else 12, c=colors[g], label=f"{g} ({m.sum()})",
                       alpha=0.95 if g == "real torn" else 0.6, edgecolors="k" if g == "real torn" else "none", linewidths=0.4)
    ax.legend(fontsize=8, markerscale=1.3, loc="best"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"UMAP of replica-encoder features in the meniscus region ({a.source} PD)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "umap_by_group.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    srcs = sorted(df.source.unique()); cmap = plt.get_cmap("tab20")
    for i, s in enumerate(srcs):
        m = (df.source == s).values
        ax.scatter(emb[m, 0], emb[m, 1], s=30 if s.startswith("torn") else 8, color=cmap(i % 20),
                   label=s if s.startswith("torn") else None, alpha=0.8, marker="*" if s.startswith("torn") else "o")
    ax.legend(fontsize=7, loc="best", title="torn patients (stars); dots = training patients"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Same UMAP coloured by patient: patient / scanner clustering check", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "umap_by_source.png"), dpi=120); plt.close(fig)
    print(f"\nsaved -> {a.out_dir}")


if __name__ == "__main__":
    main()
