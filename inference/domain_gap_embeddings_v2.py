"""
domain_gap_embeddings_v2.py
==============================
SEPARATE from domain_gap_tsne.py (kept as-is, not modified) -- this is the
fixed version addressing professor feedback on the first plot:

  1. RANDOM PER-PATIENT SAMPLING instead of consecutive slices from a few
     patients. domain_gap_tsne.py loaded slices in directory/alphabetical
     order up to --max_slices, which for DESS (~160 slices/patient) and
     fake PD (~36-44 slices/volume) meant only a handful of patients were
     ever represented, each contributing a long run of highly-similar
     adjacent slices -- adjacent MRI slices are visually near-identical, so
     t-SNE correctly traced this out as a smooth continuous curve rather
     than scattered independent points (a sampling artifact, not a t-SNE
     bug). Now: --slices_per_patient random slices from EVERY available
     patient/volume, so points represent many independent patients/
     positions instead of a few long sequential trajectories.
  2. UMAP added alongside t-SNE (--methods tsne umap, default both) -- via
     the `umap-learn` package.
  3. EXPLICITLY CONFIRMED: features are raw 64x64-downsampled pixel
     intensities (compute_simple_features() from evaluate.py, same features
     used for FID elsewhere in this project) -- no texture/radiomic/learned
     features.

Same 3 clustering-separation metrics as before (Silhouette, Davies-Bouldin,
Calinski-Harabasz). Does not modify domain_gap_tsne.py or any other file.

Usage:
    python inference/domain_gap_embeddings_v2.py \
        --dess_slice_dir preprocessed_v2/slices/dess \
        --fake_pd_dir    results/stage4_fake_pd \
        --real_pd_dir    data/iu-dataset/pd-files \
        --out_png        runs/run_004/domain_gap_embeddings_v2.png \
        --slices_per_patient 3 \
        --methods tsne umap
"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import compute_simple_features, extract_patient_id
from preprocess import process_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def sample_dess_per_patient(dess_dir, n_per_patient, seed=42):
    rng = np.random.RandomState(seed)
    by_patient = {}
    for f in sorted(glob.glob(os.path.join(dess_dir, "*.npy"))):
        pid = extract_patient_id(f)
        by_patient.setdefault(pid, []).append(f)

    slices = []
    for pid, files in by_patient.items():
        chosen = rng.choice(files, size=min(n_per_patient, len(files)), replace=False)
        for f in chosen:
            slices.append(np.load(f))
    log.info(f"  DESS: {len(by_patient)} patients, {len(slices)} slices sampled "
             f"({n_per_patient}/patient)")
    return slices


def sample_volume_per_patient(nifti_files, n_per_patient, modality, seed=42):
    rng = np.random.RandomState(seed)
    slices = []
    n_patients = 0
    for f in nifti_files:
        try:
            if modality == "fake_pd":
                import nibabel as nib
                vol = nib.load(f).get_fdata(dtype=np.float32)
                if vol.ndim == 3 and vol.shape[2] < vol.shape[0] and vol.shape[2] < vol.shape[1]:
                    vol = np.transpose(vol, (2, 0, 1))
            else:  # real_pd raw volumes -> same preprocessing used elsewhere
                vol = process_volume(f, "PD")
        except Exception as e:
            log.warning(f"  Skipping {f}: {e}")
            continue

        n = vol.shape[0]
        idxs = rng.choice(n, size=min(n_per_patient, n), replace=False)
        for i in idxs:
            slices.append(vol[i])
        n_patients += 1
    log.info(f"  {modality}: {n_patients} volumes, {len(slices)} slices sampled "
             f"({n_per_patient}/volume)")
    return slices


def load_slices_consecutive(dess_dir, fake_dir, real_dir, max_slices):
    """
    Reproduces domain_gap_tsne.py's ORIGINAL sampling exactly (consecutive
    slices in directory order, up to max_slices) -- kept here so the same
    script can produce the "before" comparison plot (consecutive + UMAP)
    alongside the new random-sampling mode, without touching
    domain_gap_tsne.py itself.
    """
    from evaluate import load_slices_from_nifti

    log.info("Loading DESS slices (consecutive, directory order)...")
    dess_npys = sorted(glob.glob(os.path.join(dess_dir, "*.npy")))[:max_slices]
    dess_slices = [np.load(f) for f in dess_npys]

    log.info("Loading fake PD slices (consecutive, directory order)...")
    fake_slices = []
    for f in sorted(glob.glob(os.path.join(fake_dir, "*.nii.gz"))):
        fake_slices.extend(load_slices_from_nifti(f))
        if len(fake_slices) >= max_slices:
            break
    fake_slices = fake_slices[:max_slices]

    log.info("Loading real PD slices (consecutive, directory order, preprocessed)...")
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

    log.info(f"Total: DESS={len(dess_slices)}  Fake PD={len(fake_slices)}  Real PD={len(real_slices)}")
    return dess_slices, fake_slices, real_slices


def load_slices(dess_dir, fake_dir, real_dir, slices_per_patient, seed=42):
    log.info("Sampling DESS slices (random, per-patient)...")
    dess_slices = sample_dess_per_patient(dess_dir, slices_per_patient, seed)

    log.info("Sampling fake PD slices (random, per-volume)...")
    fake_files = sorted(glob.glob(os.path.join(fake_dir, "*.nii.gz")))
    fake_slices = sample_volume_per_patient(fake_files, slices_per_patient, "fake_pd", seed)

    log.info("Sampling real PD slices (random, per-volume, preprocessed)...")
    real_files = sorted(glob.glob(os.path.join(real_dir, "**", "*.nii.gz"), recursive=True))
    real_slices = sample_volume_per_patient(real_files, slices_per_patient, "real_pd", seed)

    log.info(f"Total: DESS={len(dess_slices)}  Fake PD={len(fake_slices)}  Real PD={len(real_slices)}")
    return dess_slices, fake_slices, real_slices


def compute_separation_metrics(embedding, labels):
    return {
        "silhouette": float(silhouette_score(embedding, labels)),
        "davies_bouldin": float(davies_bouldin_score(embedding, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(embedding, labels)),
    }


def run_embedding_panel(ax, feats_a, feats_b, label_a, label_b, color_a, color_b, title, method, seed=42):
    feats = np.concatenate([feats_a, feats_b], axis=0)
    labels = np.array([0] * len(feats_a) + [1] * len(feats_b))

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=seed, init="pca",
                        perplexity=min(30, max(5, len(feats) // 4)))
    elif method == "umap":
        if not UMAP_AVAILABLE:
            ax.set_title(f"{title}\n(umap-learn not installed)")
            ax.axis("off")
            return None
        reducer = umap.UMAP(n_components=2, random_state=seed,
                             n_neighbors=min(15, max(2, len(feats) // 4)))
    else:
        raise ValueError(f"Unknown method: {method}")

    emb = reducer.fit_transform(feats)

    ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1], s=10, c=color_a, label=label_a, alpha=0.6)
    ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1], s=10, c=color_b, label=label_b, alpha=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="upper right", fontsize=8)

    metrics = compute_separation_metrics(emb, labels)
    text = (f"Silhouette: {metrics['silhouette']:.3f}\n"
            f"Davies-Bouldin: {metrics['davies_bouldin']:.3f}\n"
            f"Calinski-Harabasz: {metrics['calinski_harabasz']:.1f}")
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=7,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dess_slice_dir", required=True)
    ap.add_argument("--fake_pd_dir", required=True)
    ap.add_argument("--real_pd_dir", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--slices_per_patient", type=int, default=3,
                     help="Random slices sampled per patient/volume (only used if --sampling random)")
    ap.add_argument("--max_slices", type=int, default=400,
                     help="Only used if --sampling consecutive (matches domain_gap_tsne.py)")
    ap.add_argument("--sampling", default="random", choices=["random", "consecutive"],
                     help="'random' = new fixed per-patient sampling (default); "
                          "'consecutive' = reproduces domain_gap_tsne.py's original "
                          "directory-order sampling, for an apples-to-apples UMAP "
                          "comparison against that same plot")
    ap.add_argument("--methods", nargs="+", default=["tsne", "umap"], choices=["tsne", "umap"])
    args = ap.parse_args()

    if args.sampling == "random":
        dess_slices, fake_slices, real_slices = load_slices(
            args.dess_slice_dir, args.fake_pd_dir, args.real_pd_dir, args.slices_per_patient)
    else:
        dess_slices, fake_slices, real_slices = load_slices_consecutive(
            args.dess_slice_dir, args.fake_pd_dir, args.real_pd_dir, args.max_slices)

    log.info("Computing features: RAW 64x64-downsampled pixel intensities "
             "(compute_simple_features() -- same as FID elsewhere in this project, "
             "NOT texture/radiomic/learned features)")
    dess_feats = compute_simple_features(np.stack(dess_slices, axis=0)).astype(np.float64)
    fake_feats = compute_simple_features(np.stack(fake_slices, axis=0)).astype(np.float64)
    real_feats = compute_simple_features(np.stack(real_slices, axis=0)).astype(np.float64)

    n_methods = len(args.methods)
    fig, axes = plt.subplots(n_methods, 2, figsize=(14, 6 * n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, 2)
    sampling_desc = ("random per-patient sampling" if args.sampling == "random"
                      else "consecutive/directory-order sampling (original method)")
    fig.suptitle(f"Domain Gap Before vs After Translation\n"
                 f"({sampling_desc}, raw pixel-intensity features)",
                 fontsize=13, fontweight="bold")

    for row, method in enumerate(args.methods):
        log.info(f"\nRunning {method.upper()}: DESS vs Real PD (BEFORE)...")
        m_before = run_embedding_panel(
            axes[row, 0], dess_feats, real_feats, "DESS", "Real PD", "tab:red", "tab:green",
            f"{method.upper()} BEFORE: DESS vs Real PD", method)
        if m_before:
            log.info(f"  {m_before}")

        log.info(f"Running {method.upper()}: Fake PD vs Real PD (AFTER)...")
        m_after = run_embedding_panel(
            axes[row, 1], fake_feats, real_feats, "Fake PD", "Real PD", "tab:red", "tab:green",
            f"{method.upper()} AFTER: Fake PD vs Real PD", method)
        if m_after:
            log.info(f"  {m_after}")
            if m_before and m_after["silhouette"] < m_before["silhouette"]:
                log.info(f"  [{method}] Silhouette DECREASED after translation -- "
                         f"consistent with a closed domain gap.")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png, dpi=150)
    log.info(f"\nSaved -> {args.out_png}")


if __name__ == "__main__":
    main()
