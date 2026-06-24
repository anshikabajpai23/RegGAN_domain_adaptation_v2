"""
domain_gap_tsne.py
====================
t-SNE visualization comparing domain separation BEFORE vs AFTER translation --
the same idea as multi-site MRI harmonization figures (e.g. "TSNE: Raw vs
ComBat"), adapted here as "TSNE: DESS-vs-RealPD vs FakePD-vs-RealPD".

Panel A (before): DESS vs Real PD features -- expect well-SEPARATED clusters,
confirming the original domain gap.
Panel B (after):  Fake PD vs Real PD features -- expect MORE MIXED/overlapping
clusters if translation closed the domain gap, consistent with the FID
improvement already measured (191 -> 118, p<0.001).

Reports the same 3 clustering-separation metrics as the reference figure:
  - Silhouette score    (higher = more separated; ~0 = overlapping; want LOWER after)
  - Davies-Bouldin      (lower = more separated; want HIGHER after, i.e. less separated)
  - Calinski-Harabasz   (higher = more separated; want LOWER after)
All three computed using the domain label (not a true unsupervised cluster) as
the grouping -- i.e. "how separable are these two domains in feature space."

Reuses evaluate.py's compute_simple_features() (same 64x64 pixel features used
for FID elsewhere in this project) -- not duplicated. Does not modify any
existing file.

Usage:
    python inference/domain_gap_tsne.py \
        --dess_slice_dir preprocessed_v2/slices/dess \
        --fake_pd_dir    results/stage4_fake_pd \
        --real_pd_dir    data/iu-dataset/pd-files \
        --out_png        runs/run_004/domain_gap_tsne.png \
        --max_slices     400
"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import compute_simple_features, slices_to_array, load_slices_from_nifti
from preprocess import process_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_slices(dess_dir, fake_dir, real_dir, max_slices):
    log.info("Loading DESS slices...")
    dess_npys = sorted(glob.glob(os.path.join(dess_dir, "*.npy")))[:max_slices]
    dess_slices = [np.load(f) for f in dess_npys]

    log.info("Loading fake PD slices...")
    fake_slices = []
    for f in sorted(glob.glob(os.path.join(fake_dir, "*.nii.gz"))):
        fake_slices.extend(load_slices_from_nifti(f))
        if len(fake_slices) >= max_slices:
            break
    fake_slices = fake_slices[:max_slices]

    log.info("Loading real PD slices (preprocessed)...")
    real_slices = []
    for f in sorted(glob.glob(os.path.join(real_dir, "**", "*.nii.gz"), recursive=True)):
        try:
            vol = process_volume(f, "PD")
            real_slices.extend([vol[i] for i in range(vol.shape[0])])
        except Exception as e:
            log.warning(f"  Skipping {f}: {e}")
        if len(real_slices) >= max_slices:
            break
    real_slices = real_slices[:max_slices]

    log.info(f"DESS={len(dess_slices)}  Fake PD={len(fake_slices)}  Real PD={len(real_slices)}")
    return dess_slices, fake_slices, real_slices


def compute_separation_metrics(embedding, labels):
    return {
        "silhouette": float(silhouette_score(embedding, labels)),
        "davies_bouldin": float(davies_bouldin_score(embedding, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(embedding, labels)),
    }


def run_tsne_panel(ax, feats_a, feats_b, label_a, label_b, color_a, color_b, title, seed=42):
    feats = np.concatenate([feats_a, feats_b], axis=0)
    labels = np.array([0] * len(feats_a) + [1] * len(feats_b))

    tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30)
    emb = tsne.fit_transform(feats)

    ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1], s=8, c=color_a, label=label_a, alpha=0.6)
    ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1], s=8, c=color_b, label=label_b, alpha=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="upper right", fontsize=9)

    metrics = compute_separation_metrics(emb, labels)
    text = (f"Silhouette: {metrics['silhouette']:.3f}\n"
            f"Davies-Bouldin: {metrics['davies_bouldin']:.3f}\n"
            f"Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}")
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dess_slice_dir", required=True)
    ap.add_argument("--fake_pd_dir", required=True)
    ap.add_argument("--real_pd_dir", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--max_slices", type=int, default=400)
    args = ap.parse_args()

    dess_slices, fake_slices, real_slices = load_slices(
        args.dess_slice_dir, args.fake_pd_dir, args.real_pd_dir, args.max_slices)

    log.info("Computing features (same 64x64 pixel features used for FID elsewhere)...")
    dess_feats = compute_simple_features(slices_to_array(dess_slices)).astype(np.float64)
    fake_feats = compute_simple_features(slices_to_array(fake_slices)).astype(np.float64)
    real_feats = compute_simple_features(slices_to_array(real_slices)).astype(np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("t-SNE: Domain Gap Before vs After Translation", fontsize=14, fontweight="bold")

    log.info("Running t-SNE: DESS vs Real PD (BEFORE translation)...")
    metrics_before = run_tsne_panel(
        axes[0], dess_feats, real_feats, "DESS", "Real PD", "tab:red", "tab:green",
        "BEFORE: DESS vs Real PD\n(expect separated clusters)")
    log.info(f"  {metrics_before}")

    log.info("Running t-SNE: Fake PD vs Real PD (AFTER translation)...")
    metrics_after = run_tsne_panel(
        axes[1], fake_feats, real_feats, "Fake PD", "Real PD", "tab:blue", "tab:green",
        "AFTER: Fake PD vs Real PD\n(expect more overlap if domain gap closed)")
    log.info(f"  {metrics_after}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png, dpi=150)
    log.info(f"\nSaved -> {args.out_png}")

    log.info("\nInterpretation: lower Silhouette/Calinski-Harabasz and higher "
             "Davies-Bouldin in the AFTER panel = domains are less separable, "
             "i.e. translation moved fake PD's distribution closer to real PD.")
    if metrics_after["silhouette"] < metrics_before["silhouette"]:
        log.info("  Silhouette DECREASED after translation -- consistent with a closed domain gap.")
    else:
        log.info("  Silhouette did NOT decrease -- worth a closer look.")


if __name__ == "__main__":
    main()
