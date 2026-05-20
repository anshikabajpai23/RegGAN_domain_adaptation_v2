"""
evaluate.py
===========
Comprehensive evaluation for RegGAN DESS->PD domain adaptation.

Computes:
  1. FID  — fake PD vs real PD distribution distance
  2. KID  — kernel inception distance (more reliable for small datasets)
  3. SSIM — structural similarity DESS vs fake PD (structure preservation)
  4. Intensity histogram comparison
  5. Jacobian determinant of deformation field (topology check)
  6. Deformation magnitude (global + meniscus-region specific)
  7. Summary report

Usage:
    python evaluate.py \
        --fake_pd_dir  /N/.../results2/translated_pd \
        --real_pd_dir  /N/.../data/iu-dataset/pd-files \
        --dess_dir     /N/.../data/skm-tea-dataset/dess-files \
        --mask_dir     /N/.../preprocessed/masks \
        --ckpt         /N/.../runs/run_002/ckpt_latest.pt \
        --out_dir      /N/.../evaluation \
        --ngf 48 --n_res 9
"""

import os
import glob
import json
import logging
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import linalg
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import Generator, RegistrationNet
from preprocess import process_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_slices_from_nifti(nifti_path, n_slices=None):
    """Load preprocessed NIfTI (n_slices, 384, 384) and return slices list."""
    vol = nib.load(nifti_path).get_fdata(dtype=np.float32)
    if vol.ndim == 3:
        slices = [vol[i] for i in range(vol.shape[0])]
    else:
        slices = [vol[:, :, i] for i in range(vol.shape[2])]
    if n_slices:
        idx = np.linspace(0, len(slices)-1, n_slices, dtype=int)
        slices = [slices[i] for i in idx]
    return slices


def load_npy_slices(npy_dir, prefix=None, max_slices=500):
    """Load .npy slices from a directory."""
    pattern = os.path.join(npy_dir, f"{prefix}*.npy" if prefix else "*.npy")
    files   = sorted(glob.glob(pattern))[:max_slices]
    return [np.load(f) for f in files]


def slices_to_array(slices):
    """Stack list of (H,W) slices -> (N, H, W)."""
    return np.stack(slices, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FID
# ─────────────────────────────────────────────────────────────────────────────

def compute_simple_features(slices_array):
    """
    Simple feature extraction without InceptionV3 (avoids internet dependency).
    Uses flattened pixel patches as features — approximation of FID.
    For proper FID use torchmetrics or pytorch-fid.
    """
    # Downsample to 64x64 for speed
    feats = []
    for sl in slices_array:
        small = zoom(sl, 64/sl.shape[0], order=1)[:64, :64]
        feats.append(small.flatten())
    return np.stack(feats)   # (N, 4096)


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff  = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(fake_slices, real_slices):
    log.info("Computing FID...")
    fake_arr = slices_to_array(fake_slices)
    real_arr = slices_to_array(real_slices)

    fake_feats = compute_simple_features(fake_arr).astype(np.float64)
    real_feats = compute_simple_features(real_arr).astype(np.float64)

    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)

    fid = frechet_distance(mu_f, sigma_f, mu_r, sigma_r)
    log.info(f"  FID = {fid:.4f}")
    return fid


# ─────────────────────────────────────────────────────────────────────────────
# 2. KID (Maximum Mean Discrepancy approximation)
# ─────────────────────────────────────────────────────────────────────────────

def rbf_kernel(X, Y, sigma=1.0):
    XX = (X**2).sum(1, keepdims=True)
    YY = (Y**2).sum(1, keepdims=True)
    XY = X @ Y.T
    sq  = XX + YY.T - 2*XY
    return np.exp(-sq / (2 * sigma**2))


def compute_kid(fake_slices, real_slices, n_subsets=10, subset_size=100):
    log.info("Computing KID (MMD)...")
    fake_arr  = slices_to_array(fake_slices)
    real_arr  = slices_to_array(real_slices)
    fake_feat = compute_simple_features(fake_arr).astype(np.float64)
    real_feat = compute_simple_features(real_arr).astype(np.float64)

    # Normalise
    mu  = fake_feat.mean(0)
    std = fake_feat.std(0) + 1e-8
    fake_feat = (fake_feat - mu) / std
    real_feat = (real_feat - mu) / std

    mmds = []
    for _ in range(n_subsets):
        fi = np.random.choice(len(fake_feat), min(subset_size, len(fake_feat)), replace=False)
        ri = np.random.choice(len(real_feat), min(subset_size, len(real_feat)), replace=False)
        f  = fake_feat[fi]
        r  = real_feat[ri]
        sigma = 1.0
        mmd = (rbf_kernel(f, f, sigma).mean()
               + rbf_kernel(r, r, sigma).mean()
               - 2 * rbf_kernel(f, r, sigma).mean())
        mmds.append(mmd)

    kid_mean = float(np.mean(mmds))
    kid_std  = float(np.std(mmds))
    log.info(f"  KID = {kid_mean:.6f} ± {kid_std:.6f}")
    return kid_mean, kid_std


# ─────────────────────────────────────────────────────────────────────────────
# 3. SSIM — structural preservation DESS -> fake PD
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(dess_slices, fake_pd_slices, n=200):
    log.info("Computing SSIM (DESS vs fake PD)...")
    n    = min(n, len(dess_slices), len(fake_pd_slices))
    vals = []
    for i in range(n):
        d = dess_slices[i].astype(np.float32)
        f = fake_pd_slices[i].astype(np.float32)
        # normalise both to [0,1]
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        vals.append(ssim(d, f, data_range=1.0))
    mean_ssim = float(np.mean(vals))
    log.info(f"  SSIM = {mean_ssim:.4f}")
    return mean_ssim


# ─────────────────────────────────────────────────────────────────────────────
# 4. Intensity histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_histograms(dess_slices, fake_pd_slices, real_pd_slices, out_path):
    log.info("Plotting intensity histograms...")
    def flat(slices, n=2000):
        arr = slices_to_array(slices[:n]).flatten()
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    d = flat(dess_slices)
    f = flat(fake_pd_slices)
    r = flat(real_pd_slices)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 100)
    ax.hist(d, bins=bins, alpha=0.5, label="DESS",    density=True, color="blue")
    ax.hist(f, bins=bins, alpha=0.5, label="Fake PD", density=True, color="orange")
    ax.hist(r, bins=bins, alpha=0.5, label="Real PD", density=True, color="green")
    ax.set_xlabel("Normalised intensity")
    ax.set_ylabel("Density")
    ax.set_title("Intensity Distribution: DESS vs Fake PD vs Real PD")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"  Histogram saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Jacobian determinant of deformation field
# ─────────────────────────────────────────────────────────────────────────────

def compute_jacobian_det(flow):
    """
    flow: (2, H, W) deformation field
    Returns jacobian determinant map (H, W).
    det(J) > 0 = valid, det(J) < 0 = folding (bad)
    det(J) ~ 1 = near-rigid (minimum deformation)
    """
    dy_dx = np.gradient(flow[0], axis=1)  # d(dx)/dx
    dy_dy = np.gradient(flow[0], axis=0)  # d(dx)/dy
    dx_dx = np.gradient(flow[1], axis=1)  # d(dy)/dx
    dx_dy = np.gradient(flow[1], axis=0)  # d(dy)/dy

    # J = [[1+dy_dx, dy_dy], [dx_dx, 1+dx_dy]]
    det = (1 + dy_dx) * (1 + dx_dy) - dy_dy * dx_dx
    return det


def evaluate_deformation(dess_slices, fake_pd_slices, R_net, device,
                          mask_slices=None, meniscus_label=None, n=50):
    """
    Run registration network on (fake_pd, dess) pairs and analyse deformation.
    """
    log.info("Evaluating deformation field...")
    n = min(n, len(dess_slices), len(fake_pd_slices))

    mags, jac_dets = [], []
    meniscus_mags  = []

    R_net.eval()
    with torch.no_grad():
        for i in range(n):
            d = torch.from_numpy(dess_slices[i][None, None]).to(device) * 2 - 1
            f = torch.from_numpy(fake_pd_slices[i][None, None]).to(device) * 2 - 1

            flow = R_net(f, d)   # (1, 2, H, W)
            flow_np = flow[0].cpu().numpy()   # (2, H, W)

            # magnitude
            mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
            mags.append(mag.mean())

            # jacobian
            jac = compute_jacobian_det(flow_np)
            jac_dets.append(jac)

            # meniscus-specific deformation
            if mask_slices is not None and i < len(mask_slices) and meniscus_label is not None:
                m = mask_slices[i]
                labels = meniscus_label if isinstance(meniscus_label, list) else [meniscus_label]
                meniscus_region = np.isin(m, labels)
                if meniscus_region.any():
                    meniscus_mags.append(mag[meniscus_region].mean())

    jac_arr = np.stack(jac_dets)

    results = {
        "mean_deformation_magnitude":   float(np.mean(mags)),
        "max_deformation_magnitude":    float(np.max(mags)),
        "jacobian_det_mean":            float(jac_arr.mean()),
        "jacobian_det_min":             float(jac_arr.min()),
        "jacobian_det_max":             float(jac_arr.max()),
        "jacobian_folding_pct":         float((jac_arr < 0).mean() * 100),
    }

    if meniscus_mags:
        results["meniscus_mean_deformation"] = float(np.mean(meniscus_mags))
        results["meniscus_max_deformation"]  = float(np.max(meniscus_mags))

    for k, v in results.items():
        log.info(f"  {k}: {v:.6f}")

    return results, jac_arr


def plot_jacobian(jac_arr, out_path):
    """Plot jacobian determinant heatmap for a sample slice."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, ax in enumerate(axes):
        idx = i * len(jac_arr) // 3
        im  = ax.imshow(jac_arr[idx], cmap="RdBu", vmin=0.5, vmax=1.5)
        ax.set_title(f"Jacobian det slice {idx}")
        ax.axis("off")
        plt.colorbar(im, ax=ax)
    plt.suptitle("Jacobian Determinant (1=rigid, <0=folding)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"  Jacobian plot saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visual sample grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_grid(dess_slices, fake_pd_slices, real_pd_slices, out_path, n=4):
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    indices = np.linspace(0, min(len(dess_slices), len(fake_pd_slices))-1, n, dtype=int)

    for j, idx in enumerate(indices):
        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        axes[0][j].imshow(norm(dess_slices[idx]),     cmap="gray")
        axes[0][j].set_title(f"DESS {idx}")
        axes[0][j].axis("off")

        axes[1][j].imshow(norm(fake_pd_slices[idx]),  cmap="gray")
        axes[1][j].set_title(f"Fake PD {idx}")
        axes[1][j].axis("off")

        ri = min(idx, len(real_pd_slices)-1)
        axes[2][j].imshow(norm(real_pd_slices[ri]),   cmap="gray")
        axes[2][j].set_title(f"Real PD {ri}")
        axes[2][j].axis("off")

    axes[0][0].set_ylabel("DESS", fontsize=12)
    axes[1][0].set_ylabel("Fake PD", fontsize=12)
    axes[2][0].set_ylabel("Real PD", fontsize=12)
    plt.suptitle("DESS → Fake PD vs Real PD", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"  Sample grid saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Meniscus overlay visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_meniscus_overlays(fake_pd_slices, mask_slices, meniscus_labels, out_dir, n_scans=None):
    """
    For each scan, find 2 representative meniscus slices:
      - Lateral meniscus  (first 1/3 of meniscus slices)
      - Medial meniscus   (last 1/3 of meniscus slices)
    Saves one PNG per scan with DESS | Fake PD | Mask overlay.
    All wrapped in try/except per scan.
    """
    os.makedirs(os.path.join(out_dir, "meniscus_overlays"), exist_ok=True)
    labels = meniscus_labels if isinstance(meniscus_labels, list) else [meniscus_labels]

    # Group slices into scan-sized chunks
    # We don't have scan boundaries so use fixed chunk of 160 (DESS depth)
    chunk = 160
    total = min(len(fake_pd_slices), len(mask_slices))
    scan_idx = 0

    for start in range(0, total, chunk):
        end        = min(start + chunk, total)
        fp_chunk   = fake_pd_slices[start:end]
        mask_chunk = mask_slices[start:end]
        scan_idx  += 1
        if n_scans and scan_idx > n_scans:
            break

        try:
            # Find slices containing meniscus
            men_slices = [i for i in range(len(mask_chunk))
                          if np.isin(mask_chunk[i], labels).any()]

            if len(men_slices) < 2:
                log.warning(f"  Scan {scan_idx}: not enough meniscus slices ({len(men_slices)}) — skipping overlay")
                continue

            # Lateral = first 1/3, Medial = last 1/3
            lateral_idx = men_slices[len(men_slices) // 6]
            medial_idx  = men_slices[5 * len(men_slices) // 6]

            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            titles = ["Fake PD", "Meniscus mask", "Overlay"]

            for row, (sl_idx, horn) in enumerate([(lateral_idx, "Lateral"), (medial_idx, "Medial")]):
                fp   = fake_pd_slices[start + sl_idx]
                mask = mask_slices[start + sl_idx]
                men  = np.isin(mask, labels)

                fp_norm = (fp - fp.min()) / (fp.max() - fp.min() + 1e-8)

                axes[row][0].imshow(fp_norm, cmap="gray")
                axes[row][0].set_title(f"{horn} — Fake PD (slice {sl_idx})")
                axes[row][0].axis("off")

                axes[row][1].imshow(men, cmap="Reds", vmin=0, vmax=1)
                axes[row][1].set_title(f"{horn} — Meniscus mask")
                axes[row][1].axis("off")

                axes[row][2].imshow(fp_norm, cmap="gray")
                axes[row][2].imshow(men, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
                axes[row][2].set_title(f"{horn} — Overlay")
                axes[row][2].axis("off")

            plt.suptitle(f"Scan {scan_idx} — Meniscus on Fake PD (labels {labels})", fontsize=12)
            plt.tight_layout()
            out_path = os.path.join(out_dir, "meniscus_overlays", f"scan_{scan_idx:03d}_meniscus.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            log.info(f"  Meniscus overlay saved -> {out_path}")

        except Exception as e:
            log.warning(f"  Scan {scan_idx} meniscus overlay failed: {e}")
            plt.close()
            continue


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Load slices ──────────────────────────────────────────────────────────
    log.info("Loading DESS slices from preprocessed npy...")
    dess_npys    = sorted(glob.glob(os.path.join(args.dess_slice_dir, "*.npy")))
    dess_slices  = [np.load(f) for f in dess_npys[:args.max_slices]]
    log.info(f"  DESS: {len(dess_slices)} slices")

    log.info("Loading fake PD slices from translated NIfTIs...")
    fake_pd_files  = sorted(glob.glob(os.path.join(args.fake_pd_dir, "*.nii.gz")))
    fake_pd_slices = []
    for f in fake_pd_files:
        fake_pd_slices.extend(load_slices_from_nifti(f))
        if len(fake_pd_slices) >= args.max_slices:
            break
    fake_pd_slices = fake_pd_slices[:args.max_slices]
    log.info(f"  Fake PD: {len(fake_pd_slices)} slices")

    log.info("Loading real PD slices from raw NIfTIs...")
    real_pd_files  = sorted(glob.glob(os.path.join(args.real_pd_dir, "**", "*.nii.gz"),
                                      recursive=True))
    real_pd_slices = []
    for f in real_pd_files:
        vol = process_volume(f, "PD")   # preprocess same way
        real_pd_slices.extend([vol[i] for i in range(vol.shape[0])])
        if len(real_pd_slices) >= args.max_slices:
            break
    real_pd_slices = real_pd_slices[:args.max_slices]
    log.info(f"  Real PD: {len(real_pd_slices)} slices")

    # ── Load masks (optional) ────────────────────────────────────────────────
    mask_slices = None
    if args.mask_dir and os.path.exists(args.mask_dir):
        log.info("Loading mask slices...")
        mask_files  = sorted(glob.glob(os.path.join(args.mask_dir, "*.nii.gz")))
        mask_slices = []
        for f in mask_files:
            try:
                vol = nib.load(f).get_fdata(dtype=np.float32)
                mask_slices.extend([vol[i] for i in range(vol.shape[0])])
            except Exception as e:
                log.warning(f"  Skipping corrupted mask {f}: {e}")
                continue
            if len(mask_slices) >= args.max_slices:
                break
        mask_slices = mask_slices[:len(dess_slices)]
        log.info(f"  Masks: {len(mask_slices)} slices")

    # ── Load registration net ────────────────────────────────────────────────
    R_net = None
    if args.ckpt and os.path.exists(args.ckpt):
        log.info("Loading registration network...")
        R_net = RegistrationNet(nf=16).to(device)
        ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
        R_net.load_state_dict(ckpt["R"])
        R_net.eval()

    # ── Compute metrics ──────────────────────────────────────────────────────
    results = {}

    # 1. FID
    results["FID"] = compute_fid(fake_pd_slices, real_pd_slices)

    # 2. KID
    kid_mean, kid_std = compute_kid(fake_pd_slices, real_pd_slices)
    results["KID_mean"] = kid_mean
    results["KID_std"]  = kid_std

    # 3. SSIM
    n_ssim = min(len(dess_slices), len(fake_pd_slices))
    results["SSIM_DESS_FakePD"] = compute_ssim(
        dess_slices[:n_ssim], fake_pd_slices[:n_ssim]
    )

    # 4. Histogram
    plot_histograms(
        dess_slices, fake_pd_slices, real_pd_slices,
        os.path.join(args.out_dir, "intensity_histogram.png")
    )

    # 5 & 6. Deformation
    if R_net is not None:
        n_def = min(len(dess_slices), len(fake_pd_slices), 100)
        deform_results, jac_arr = evaluate_deformation(
            dess_slices[:n_def], fake_pd_slices[:n_def],
            R_net, device,
            mask_slices=mask_slices[:n_def] if mask_slices else None,
            meniscus_label=args.meniscus_label,
            n=n_def
        )
        results.update(deform_results)
        plot_jacobian(jac_arr, os.path.join(args.out_dir, "jacobian_det.png"))

    # 7. Visual sample grid
    plot_sample_grid(
        dess_slices, fake_pd_slices, real_pd_slices,
        os.path.join(args.out_dir, "sample_grid.png")
    )

    # 8. Meniscus overlay (lateral + medial per scan)
    if mask_slices is not None:
        try:
            plot_meniscus_overlays(
                fake_pd_slices, mask_slices,
                meniscus_labels=args.meniscus_label,
                out_dir=args.out_dir,
            )
        except Exception as e:
            log.warning(f"Meniscus overlay skipped: {e}")

    # ── Save report ──────────────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, "metrics.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n{'='*50}")
    log.info("EVALUATION SUMMARY")
    log.info(f"{'='*50}")
    for k, v in results.items():
        log.info(f"  {k:40s}: {v:.6f}")
    log.info(f"{'='*50}")
    log.info(f"Full report -> {report_path}")
    log.info(f"Plots       -> {args.out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_pd_dir",    required=True,
                    help="Dir with translated PD NIfTIs from infer.py")
    ap.add_argument("--real_pd_dir",    required=True,
                    help="Dir with real PD NIfTIs")
    ap.add_argument("--dess_slice_dir", required=True,
                    help="Dir with preprocessed DESS .npy slices")
    ap.add_argument("--mask_dir",       default=None,
                    help="Dir with preprocessed mask NIfTIs (optional)")
    ap.add_argument("--ckpt",           default=None,
                    help="Checkpoint path for deformation evaluation")
    ap.add_argument("--out_dir",        default="evaluation")
    ap.add_argument("--ngf",            type=int, default=48)
    ap.add_argument("--n_res",          type=int, default=9)
    ap.add_argument("--max_slices",     type=int, default=500,
                    help="Max slices per modality to use")
    ap.add_argument("--meniscus_label", type=int, nargs="+", default=[2],
                    help="Integer label(s) for meniscus in segmentation mask (e.g. 5 6)")
    args = ap.parse_args()
    main(args)