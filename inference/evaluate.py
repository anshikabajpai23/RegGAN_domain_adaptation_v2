"""
evaluate.py
===========
Comprehensive evaluation for RegGAN DESS->PD domain adaptation.

Changes from v1:
  - sample_grid now shows slices WITH meniscus
  - meniscus overlay uses bright boundary contours (red=label5, green=label6)
  - full knee boundary overlay showing deformation
  - difference map (anomaly-style heatmap like diffusion papers)
  - histogram fix: preprocess real PD same way
  - jacobian plot improvements
"""

import os
import re
import glob
import json
import logging
import argparse
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from scipy import linalg
from scipy.ndimage import zoom, binary_dilation
from skimage.metrics import structural_similarity as ssim
from skimage import measure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from models import Generator, RegistrationNet
from preprocess import process_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def norm(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def load_slices_from_nifti(nifti_path):
    vol = nib.load(nifti_path).get_fdata(dtype=np.float32)
    # Our translated PD NIfTIs are saved as (n_slices, H, W)
    # where n_slices=160, H=W=384
    # Do NOT transpose — axis 0 is always the slice axis
    # Only transpose if clearly (H, W, n) i.e. last dim is smallest
    if vol.ndim == 3 and vol.shape[2] < vol.shape[0] and vol.shape[2] < vol.shape[1]:
        vol = np.transpose(vol, (2, 0, 1))
    return [vol[i] for i in range(vol.shape[0])]


def draw_boundary(mask_bin, color, ax, linewidth=2):
    """Draw boundary contour of a binary mask on ax."""
    try:
        contours = measure.find_contours(mask_bin.astype(float), 0.5)
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color=color, linewidth=linewidth)
    except Exception:
        pass


def slices_to_array(slices):
    return np.stack(slices, axis=0)


def extract_patient_id(filename):
    """
    STAGE 4a FIX: canonical patient ID extraction, used to key dess_slices,
    fake_pd_slices, and mask_slices by patient before zipping them together
    (previously they were concatenated independently and zipped purely by
    list position, which silently misaligned different patients' slices).

    Mask filenames use a SHORT stem ("MTR_005_mask.nii.gz") while DESS/fake-PD
    filenames use the FULL stem ("MTR_005_Anonymized_2378615199_e1_..."), so
    we extract the common leading patient code (e.g. "MTR_005") from both.
    Falls back to stripping known suffixes for other naming schemes.
    """
    base = os.path.basename(filename)
    m = re.match(r"(MTR_\d+)", base)
    if m:
        return m.group(1)
    for suf in ["_sl", "_pd_translated", "_mask"]:
        if suf in base:
            return base.split(suf)[0]
    return os.path.splitext(os.path.splitext(base)[0])[0]


# ─────────────────────────────────────────────────────────────────────────────
# 1. FID
# ─────────────────────────────────────────────────────────────────────────────

def compute_simple_features(slices_array):
    feats = []
    for sl in slices_array:
        n = norm(sl)
        small = zoom(n, 64 / max(n.shape), order=1)[:64, :64]
        if small.shape != (64, 64):
            pad = np.zeros((64, 64), dtype=np.float32)
            pad[:small.shape[0], :small.shape[1]] = small
            small = pad
        feats.append(small.flatten())
    return np.stack(feats)


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    # regularise to avoid singular matrix
    eps = 1e-6
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(fake_slices, real_slices):
    # NOTE ON FID VALIDITY: this implementation uses 64x64 raw pixel features,
    # NOT InceptionV3 features (no internet on HPC). The absolute FID value is
    # non-standard and NOT comparable to published scores. The relative comparison
    # (fake_PD vs real_PD) < (DESS vs real_PD baseline) is valid and shows the
    # translation improves distributional similarity. State this limitation clearly
    # in any writeup.
    log.info("Computing FID (64x64 pixel features — non-standard, relative comparison only)...")
    fake_feats = compute_simple_features(slices_to_array(fake_slices)).astype(np.float64)
    real_feats = compute_simple_features(slices_to_array(real_slices)).astype(np.float64)
    mu_f, s_f  = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    mu_r, s_r  = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    fid_val    = frechet_distance(mu_f, s_f, mu_r, s_r)
    log.info(f"  FID = {fid_val:.4f}")
    return fid_val


def compute_fid_baseline(dess_slices, real_slices):
    log.info("Computing baseline FID (DESS vs Real PD)...")
    dess_feats = compute_simple_features(slices_to_array(dess_slices)).astype(np.float64)
    real_feats = compute_simple_features(slices_to_array(real_slices)).astype(np.float64)
    mu_d, s_d  = dess_feats.mean(0), np.cov(dess_feats, rowvar=False)
    mu_r, s_r  = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    fid_val    = frechet_distance(mu_d, s_d, mu_r, s_r)
    log.info(f"  Baseline FID (DESS vs Real PD) = {fid_val:.4f}")
    return fid_val


# ─────────────────────────────────────────────────────────────────────────────
# 2. KID
# ─────────────────────────────────────────────────────────────────────────────

def rbf_kernel(X, Y, sigma=1.0):
    XX = (X**2).sum(1, keepdims=True)
    YY = (Y**2).sum(1, keepdims=True)
    sq = XX + YY.T - 2 * X @ Y.T
    sq = np.clip(sq, 0, 500)   # avoid overflow in exp
    return np.exp(-sq / (2 * sigma**2))


def compute_kid(fake_slices, real_slices, n_subsets=10, subset_size=100):
    log.info("Computing KID...")
    fake_feat = compute_simple_features(slices_to_array(fake_slices)).astype(np.float64)
    real_feat = compute_simple_features(slices_to_array(real_slices)).astype(np.float64)
    mu = fake_feat.mean(0); std = fake_feat.std(0) + 1e-8
    fake_feat = (fake_feat - mu) / std
    real_feat = (real_feat - mu) / std
    mmds = []
    for _ in range(n_subsets):
        fi = np.random.choice(len(fake_feat), min(subset_size, len(fake_feat)), replace=False)
        ri = np.random.choice(len(real_feat), min(subset_size, len(real_feat)), replace=False)
        f, r = fake_feat[fi], real_feat[ri]
        mmds.append(rbf_kernel(f,f).mean() + rbf_kernel(r,r).mean() - 2*rbf_kernel(f,r).mean())
    kid_mean, kid_std = float(np.mean(mmds)), float(np.std(mmds))
    log.info(f"  KID = {kid_mean:.6f} ± {kid_std:.6f}")
    return kid_mean, kid_std


# ─────────────────────────────────────────────────────────────────────────────
# 3. SSIM
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssim(dess_slices, fake_pd_slices, n=200):
    log.info("Computing SSIM...")
    n = min(n, len(dess_slices), len(fake_pd_slices))
    vals = []
    for i in range(n):
        d = norm(dess_slices[i])
        f = norm(fake_pd_slices[i])
        # resize to match if shapes differ
        if d.shape != f.shape:
            f = zoom(f, (d.shape[0]/f.shape[0], d.shape[1]/f.shape[1]), order=1)
        try:
            vals.append(ssim(d, f, data_range=1.0))
        except Exception as e:
            log.warning(f"  SSIM slice {i} failed: {e}")
    if not vals:
        return 0.0
    mean_ssim = float(np.mean(vals))
    log.info(f"  SSIM = {mean_ssim:.4f}")
    return mean_ssim


# ─────────────────────────────────────────────────────────────────────────────
# 4. Histogram — fixed: preprocess real PD same way
# ─────────────────────────────────────────────────────────────────────────────

def plot_histograms(dess_slices, fake_pd_slices, real_pd_slices, out_path):
    log.info("Plotting intensity histograms...")

    def flat_norm(slices, n=2000):
        arr = slices_to_array(slices[:n]).flatten()
        return norm(arr)

    d = flat_norm(dess_slices)
    f = flat_norm(fake_pd_slices)
    r = flat_norm(real_pd_slices)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(0, 1, 80)
    axes[0].hist(d, bins=bins, alpha=0.55, label="DESS (input)",   density=True, color="steelblue")
    axes[0].hist(f, bins=bins, alpha=0.55, label="Fake PD (ours)", density=True, color="darkorange")
    axes[0].hist(r, bins=bins, alpha=0.45, label="Real PD (target)",density=True, color="green")
    axes[0].set_xlabel("Normalised intensity")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Intensity Distribution")
    axes[0].legend()

    # CDF
    for arr, lbl, col in [(d,"DESS","steelblue"),(f,"Fake PD","darkorange"),(r,"Real PD","green")]:
        s = np.sort(arr)
        cdf = np.arange(1, len(s)+1) / len(s)
        axes[1].plot(s[::max(1,len(s)//500)], cdf[::max(1,len(s)//500)], label=lbl, color=col)
    axes[1].set_xlabel("Normalised intensity")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Cumulative Distribution\n(Fake PD should overlap Real PD)")
    axes[1].legend()

    plt.suptitle("Intensity Distribution Shift: DESS → Fake PD vs Real PD", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"  Histogram saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Jacobian
# ─────────────────────────────────────────────────────────────────────────────

def compute_jacobian_det(flow):
    dy_dx = np.gradient(flow[0], axis=1)
    dy_dy = np.gradient(flow[0], axis=0)
    dx_dx = np.gradient(flow[1], axis=1)
    dx_dy = np.gradient(flow[1], axis=0)
    return (1 + dy_dx) * (1 + dx_dy) - dy_dy * dx_dx


def evaluate_deformation(dess_slices, fake_pd_slices, R_net, device,
                          mask_slices=None, meniscus_label=None, n=50):
    log.info("Evaluating deformation...")
    n = min(n, len(dess_slices), len(fake_pd_slices))
    mags, jac_dets, flows_list = [], [], []
    meniscus_mags = []

    R_net.eval()
    with torch.no_grad():
        for i in range(n):
            d_np = norm(dess_slices[i])
            f_np = norm(fake_pd_slices[i])
            # ensure same shape before passing to R_net
            if d_np.shape != f_np.shape:
                f_np = zoom(f_np, (d_np.shape[0]/f_np.shape[0], d_np.shape[1]/f_np.shape[1]), order=1)
            d = torch.from_numpy(d_np[None, None]).to(device) * 2 - 1
            f = torch.from_numpy(f_np[None, None]).to(device) * 2 - 1
            flow    = R_net(f, d)
            flow_np = flow[0].cpu().numpy()
            mag     = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
            jac     = compute_jacobian_det(flow_np)
            mags.append(mag.mean())
            jac_dets.append(jac)
            flows_list.append(flow_np)
            if mask_slices is not None and i < len(mask_slices) and meniscus_label is not None:
                m      = mask_slices[i]
                labels = meniscus_label if isinstance(meniscus_label, list) else [meniscus_label]
                region = np.isin(m, labels)
                if region.any():
                    meniscus_mags.append(mag[region].mean())

    jac_arr = np.stack(jac_dets)
    results = {
        "mean_deformation_magnitude": float(np.mean(mags)),
        "max_deformation_magnitude":  float(np.max(mags)),
        "jacobian_det_mean":          float(jac_arr.mean()),
        "jacobian_det_min":           float(jac_arr.min()),
        "jacobian_det_max":           float(jac_arr.max()),
        "jacobian_folding_pct":       float((jac_arr < 0).mean() * 100),
    }
    if meniscus_mags:
        results["meniscus_mean_deformation"] = float(np.mean(meniscus_mags))
        results["meniscus_max_deformation"]  = float(np.max(meniscus_mags))
    for k, v in results.items():
        log.info(f"  {k}: {v:.6f}")
    return results, jac_arr, flows_list


def plot_jacobian(jac_arr, fake_pd_slices, out_path):
    """Improved jacobian plot: image + jacobian side by side."""
    n = len(jac_arr)
    idxs = [n//6, n//3, n//2]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for j, idx in enumerate(idxs):
        # top row: fake PD
        axes[0][j].imshow(norm(fake_pd_slices[idx]), cmap="gray")
        axes[0][j].set_title(f"Fake PD slice {idx}", fontsize=9)
        axes[0][j].axis("off")

        # bottom row: jacobian
        jac = jac_arr[idx]
        im = axes[1][j].imshow(jac, cmap="RdBu", vmin=0.85, vmax=1.15)
        axes[1][j].set_title(
            f"Jacobian det\nmean={jac.mean():.4f}  fold%={(jac<0).mean()*100:.2f}%", fontsize=8
        )
        axes[1][j].axis("off")
        plt.colorbar(im, ax=axes[1][j], fraction=0.04, label="det(J)")

    axes[0][0].set_ylabel("Fake PD", fontsize=11, fontweight="bold")
    axes[1][0].set_ylabel("Jacobian det\n(white=1=rigid)", fontsize=10, fontweight="bold")
    plt.suptitle("Jacobian Determinant — Deformation Topology Check\n"
                 "White/neutral = near-rigid (good). Red/blue = local expansion/compression.",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"  Jacobian plot saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sample grid — show slices WITH meniscus
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_grid(dess_slices, fake_pd_slices, real_pd_slices, out_path,
                     mask_slices=None, meniscus_label=None, n=4):
    log.info("Plotting sample grid (meniscus slices)...")

    # Find indices that have meniscus if masks provided
    if mask_slices is not None and meniscus_label is not None:
        labels = meniscus_label if isinstance(meniscus_label, list) else [meniscus_label]
        men_idxs = [i for i in range(min(len(mask_slices), len(dess_slices)))
                    if np.isin(mask_slices[i], labels).any()]
        if len(men_idxs) >= n:
            step    = len(men_idxs) // n
            indices = [men_idxs[i * step] for i in range(n)]
            log.info(f"  Using meniscus slice indices: {indices}")
        else:
            indices = np.linspace(0, min(len(dess_slices), len(fake_pd_slices))-1, n, dtype=int)
    else:
        indices = np.linspace(0, min(len(dess_slices), len(fake_pd_slices))-1, n, dtype=int)

    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))

    for j, idx in enumerate(indices):
        d = norm(dess_slices[idx])
        f = norm(fake_pd_slices[idx])
        ri = min(idx, len(real_pd_slices)-1)
        r = norm(real_pd_slices[ri])

        axes[0][j].imshow(d, cmap="gray")
        axes[0][j].set_title(f"DESS {idx}", fontsize=9)
        axes[0][j].axis("off")

        axes[1][j].imshow(f, cmap="gray")
        axes[1][j].set_title(f"Fake PD {idx}", fontsize=9)
        axes[1][j].axis("off")

        # Overlay meniscus boundary on fake PD
        if mask_slices is not None and meniscus_label is not None and idx < len(mask_slices):
            labels = meniscus_label if isinstance(meniscus_label, list) else [meniscus_label]
            for lbl, col in zip(labels, ["red", "lime"]):
                bin_mask = (mask_slices[idx] == lbl)
                if bin_mask.any():
                    draw_boundary(bin_mask, col, axes[1][j], linewidth=1.5)

        axes[2][j].imshow(r, cmap="gray")
        axes[2][j].set_title(f"Real PD {ri}", fontsize=9)
        axes[2][j].axis("off")

    axes[0][0].set_ylabel("DESS\n(input)", fontsize=11, fontweight="bold")
    axes[1][0].set_ylabel("Fake PD\n(translated)\n+ meniscus boundary", fontsize=10, fontweight="bold")
    axes[2][0].set_ylabel("Real PD\n(target domain)", fontsize=11, fontweight="bold")
    plt.suptitle("DESS → Fake PD Translation at Meniscus Slices\n"
                 "Red=lateral meniscus boundary, Green=medial meniscus boundary",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"  Sample grid saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Meniscus overlay — bright boundary contours
# ─────────────────────────────────────────────────────────────────────────────

def plot_meniscus_overlays(dess_slices, fake_pd_slices, mask_slices,
                            meniscus_labels, out_dir):
    log.info("Plotting meniscus overlays with boundary contours...")
    os.makedirs(os.path.join(out_dir, "meniscus_overlays"), exist_ok=True)
    labels = meniscus_labels if isinstance(meniscus_labels, list) else [meniscus_labels]
    colors = ["red", "lime"]   # label 1 = red, label 2 = green

    chunk    = min(160, len(fake_pd_slices))
    total    = min(len(fake_pd_slices), len(mask_slices), len(dess_slices))
    scan_idx = 0

    for start in range(0, total, chunk):
        end        = min(start + chunk, total)
        scan_idx  += 1

        try:
            # Find best slice per label (most pixels)
            def best_for_label(lbl):
                counts = [(mask_slices[start+i] == lbl).sum() for i in range(end-start)]
                best   = int(np.argmax(counts))
                return best if counts[best] > 0 else None

            lateral_i = best_for_label(labels[0])
            medial_i  = best_for_label(labels[1] if len(labels) > 1 else labels[0])

            if lateral_i is None and medial_i is None:
                log.warning(f"  Scan {scan_idx}: no meniscus found — skipping")
                continue

            horns = []
            if lateral_i is not None: horns.append((lateral_i, "Lateral", labels[0], "red"))
            if medial_i  is not None: horns.append((medial_i,  "Medial",  labels[1] if len(labels)>1 else labels[0], "lime"))

            fig, axes = plt.subplots(len(horns), 4, figsize=(16, 4*len(horns)))
            if len(horns) == 1:
                axes = axes[np.newaxis, :]

            for row, (sl_i, horn, lbl, col) in enumerate(horns):
                abs_i   = start + sl_i
                d_sl    = norm(dess_slices[abs_i])
                fp_sl   = norm(fake_pd_slices[abs_i])
                bin_mask = (mask_slices[abs_i] == lbl)

                # Col 0: DESS + boundary
                axes[row][0].imshow(d_sl, cmap="gray")
                draw_boundary(bin_mask, col, axes[row][0], linewidth=2)
                axes[row][0].set_title(f"DESS (slice {sl_i})")
                axes[row][0].axis("off")

                # Col 1: Fake PD + boundary
                axes[row][1].imshow(fp_sl, cmap="gray")
                draw_boundary(bin_mask, col, axes[row][1], linewidth=2)
                axes[row][1].set_title(f"Fake PD (slice {sl_i})")
                axes[row][1].axis("off")

                # Col 2: filled overlay (semi-transparent)
                axes[row][2].imshow(fp_sl, cmap="gray")
                overlay = np.zeros((*fp_sl.shape, 4))
                if col == "red":
                    overlay[bin_mask] = [1, 0, 0, 0.4]
                else:
                    overlay[bin_mask] = [0, 1, 0, 0.4]
                axes[row][2].imshow(overlay)
                draw_boundary(bin_mask, col, axes[row][2], linewidth=2)
                axes[row][2].set_title(f"Filled overlay + boundary")
                axes[row][2].axis("off")

                # Col 3: zoom on meniscus region
                rows_nz, cols_nz = np.where(bin_mask)
                if len(rows_nz):
                    pad = 30
                    r0,r1 = max(0,rows_nz.min()-pad), min(fp_sl.shape[0],rows_nz.max()+pad)
                    c0,c1 = max(0,cols_nz.min()-pad), min(fp_sl.shape[1],cols_nz.max()+pad)
                    axes[row][3].imshow(fp_sl[r0:r1,c0:c1], cmap="gray")
                    draw_boundary(bin_mask[r0:r1,c0:c1], col, axes[row][3], linewidth=2)
                    axes[row][3].set_title(f"Zoom — {horn} meniscus")
                    axes[row][3].axis("off")

                axes[row][0].set_ylabel(f"{horn}\n(label {lbl})", fontsize=11, fontweight="bold",
                                         color=col if col != "lime" else "darkgreen")

            plt.suptitle(f"Scan {scan_idx} — Meniscus Overlay\n"
                         "Red=lateral, Green=medial | Boundary contours for precise alignment check",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            out_path = os.path.join(out_dir, "meniscus_overlays", f"scan_{scan_idx:03d}_meniscus.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            log.info(f"  Meniscus overlay saved -> {out_path}")

        except Exception as e:
            log.warning(f"  Scan {scan_idx} meniscus overlay failed: {e}")
            plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full knee boundary overlay — DESS vs Fake PD
# ─────────────────────────────────────────────────────────────────────────────

def plot_knee_boundary_overlay(dess_slices, fake_pd_slices, mask_slices,
                                out_dir, n=4):
    log.info("Plotting full knee boundary overlay...")
    os.makedirs(out_dir, exist_ok=True)

    # Find good slices (not empty)
    total   = min(len(dess_slices), len(fake_pd_slices), len(mask_slices))
    indices = np.linspace(total//5, 4*total//5, n, dtype=int)

    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))

    for j, idx in enumerate(indices):
        d  = norm(dess_slices[idx])
        f  = norm(fake_pd_slices[idx])
        m  = mask_slices[idx]

        # All tissue = any non-zero label
        tissue = (m > 0)

        for row, (img, title) in enumerate([(d, f"DESS {idx}"), (f, f"Fake PD {idx}")]):
            axes[row][j].imshow(img, cmap="gray")
            # All tissue boundary in cyan
            draw_boundary(tissue, "cyan", axes[row][j], linewidth=1.5)
            # Per-label boundaries
            for lbl, col in zip(np.unique(m)[1:], ["red","lime","blue","yellow","magenta"]):
                draw_boundary((m == lbl), col, axes[row][j], linewidth=1)
            axes[row][j].set_title(title, fontsize=9)
            axes[row][j].axis("off")

    axes[0][0].set_ylabel("DESS + boundaries", fontsize=10, fontweight="bold")
    axes[1][0].set_ylabel("Fake PD + boundaries", fontsize=10, fontweight="bold")
    plt.suptitle("Full Knee Boundary Overlay\n"
                 "Cyan=all tissue | Red/Green/Blue=individual labels\n"
                 "Boundaries should align between DESS and Fake PD",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "knee_boundary_overlay.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"  Knee boundary overlay saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Difference / anomaly map (inspired by diffusion anomaly detection)
# ─────────────────────────────────────────────────────────────────────────────

def plot_difference_map(dess_slices, fake_pd_slices, out_dir,
                         mask_slices=None, meniscus_label=None, n=4):
    log.info("Plotting difference / anomaly maps...")
    total   = min(len(dess_slices), len(fake_pd_slices))

    # Prefer meniscus slices
    if mask_slices is not None and meniscus_label is not None:
        labels   = meniscus_label if isinstance(meniscus_label, list) else [meniscus_label]
        men_idxs = [i for i in range(min(len(mask_slices), total))
                    if np.isin(mask_slices[i], labels).any()]
        step    = max(1, len(men_idxs) // n)
        indices = [men_idxs[i*step] for i in range(min(n, len(men_idxs)))]
        if len(indices) < n:
            indices = np.linspace(total//5, 4*total//5, n, dtype=int).tolist()
    else:
        indices = np.linspace(total//5, 4*total//5, n, dtype=int).tolist()

    fig, axes = plt.subplots(3, n, figsize=(4*n, 11))

    for j, idx in enumerate(indices):
        d    = norm(dess_slices[idx])
        f    = norm(fake_pd_slices[idx])
        diff = np.abs(d - f)   # pixel-wise absolute difference

        axes[0][j].imshow(d, cmap="gray")
        axes[0][j].set_title(f"DESS (slice {idx})", fontsize=9)
        axes[0][j].axis("off")

        axes[1][j].imshow(f, cmap="gray")
        axes[1][j].set_title(f"Fake PD (slice {idx})", fontsize=9)
        axes[1][j].axis("off")

        # Difference heatmap (hot colormap like anomaly detection papers)
        im = axes[2][j].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[2][j].set_title(f"Diff map\nmax={diff.max():.3f}  mean={diff.mean():.3f}", fontsize=8)
        axes[2][j].axis("off")
        plt.colorbar(im, ax=axes[2][j], fraction=0.04, label="|DESS-FakePD|")

        # Overlay meniscus boundary on difference map
        if mask_slices is not None and meniscus_label is not None and idx < len(mask_slices):
            labels = meniscus_label if isinstance(meniscus_label, list) else [meniscus_label]
            for lbl, col in zip(labels, ["cyan", "lime"]):
                draw_boundary((mask_slices[idx] == lbl), col, axes[2][j], linewidth=1.5)

    axes[0][0].set_ylabel("DESS", fontsize=11, fontweight="bold")
    axes[1][0].set_ylabel("Fake PD", fontsize=11, fontweight="bold")
    axes[2][0].set_ylabel("Difference map\n(bright=large change)", fontsize=10, fontweight="bold")
    plt.suptitle("Difference Map: DESS vs Fake PD\n"
                 "Bright = regions of large change (contrast adaptation)\n"
                 "Cyan/green borders = meniscus boundary",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "difference_map.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"  Difference map saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Load DESS / fake-PD / masks — PATIENT-ALIGNED (Stage 4a fix) ──────────
    # Previously these three were each concatenated independently (different
    # glob patterns, different file-naming schemes) and zipped together by
    # raw list position. That silently misaligned different patients' slices
    # since patient counts/per-patient slice counts differ across sources.
    # Now: key everything by patient ID first, only use patients present in
    # all required sources, and pair slices within each patient by the
    # DESS slice's own index (fake-PD/mask volumes are full un-skipped
    # volumes, so DESS slice index i maps directly to position i in them).
    log.info("Loading DESS/fake-PD/mask slices (patient-aligned)...")

    dess_by_patient = {}  # pid -> {slice_idx: npy_path}
    for f in sorted(glob.glob(os.path.join(args.dess_slice_dir, "*.npy"))):
        m = re.search(r"_sl(\d{4})\.npy$", os.path.basename(f))
        if not m:
            continue
        idx = int(m.group(1))
        pid = extract_patient_id(f)
        dess_by_patient.setdefault(pid, {})[idx] = f

    fake_pd_path_by_patient = {}  # pid -> nifti path
    for f in sorted(glob.glob(os.path.join(args.fake_pd_dir, "*.nii.gz"))):
        fake_pd_path_by_patient[extract_patient_id(f)] = f

    mask_path_by_patient = {}  # pid -> nifti path
    if args.mask_dir and os.path.exists(args.mask_dir):
        for f in sorted(glob.glob(os.path.join(args.mask_dir, "*.nii.gz"))):
            mask_path_by_patient[extract_patient_id(f)] = f

    common_patients = sorted(set(dess_by_patient) & set(fake_pd_path_by_patient))
    if args.mask_dir:
        common_patients = sorted(set(common_patients) & set(mask_path_by_patient))

    if args.splits and args.split:
        with open(args.splits) as f:
            splits_json = json.load(f)
        split_pids = set(extract_patient_id(p) for p in splits_json["dess"][args.split])
        before = len(common_patients)
        common_patients = sorted(set(common_patients) & split_pids)
        log.info(f"  Filtered to --split={args.split}: {before} -> {len(common_patients)} patients "
                 f"({len(split_pids)} patient IDs in this split)")
    else:
        log.warning("  No --splits/--split given — evaluating on ALL available patients "
                    "(may include train-set patients). Pass --splits/--split to restrict "
                    "to a held-out split.")

    log.info(f"  Patients: {len(dess_by_patient)} DESS, {len(fake_pd_path_by_patient)} fake PD, "
             f"{len(mask_path_by_patient)} masks -> {len(common_patients)} usable in common")

    dess_slices, fake_pd_slices = [], []
    mask_slices = [] if args.mask_dir else None
    for pid in common_patients:
        fake_vol = load_slices_from_nifti(fake_pd_path_by_patient[pid])
        mask_vol = load_slices_from_nifti(mask_path_by_patient[pid]) if args.mask_dir else None
        for idx in sorted(dess_by_patient[pid].keys()):
            if idx >= len(fake_vol):
                continue  # DESS slice index out of range for this patient's fake-PD volume depth
            if mask_vol is not None and idx >= len(mask_vol):
                continue  # same guard for mask volume depth
            dess_slices.append(np.load(dess_by_patient[pid][idx]))
            fake_pd_slices.append(fake_vol[idx])
            if mask_vol is not None:
                mask_slices.append(mask_vol[idx])
            if len(dess_slices) >= args.max_slices:
                break
        if len(dess_slices) >= args.max_slices:
            break

    log.info(f"  DESS: {len(dess_slices)} slices")
    log.info(f"  Fake PD: {len(fake_pd_slices)} slices")
    if mask_slices is not None:
        log.info(f"  Masks: {len(mask_slices)} slices (patient+index aligned with DESS/fake PD above)")

    # ── Load real PD — preprocess same way as training ────────────────────────
    # NOTE: real_pd_slices is intentionally NOT patient-aligned with the
    # above — it's pooled for distribution-level metrics only (FID/KID),
    # which don't require per-slice anatomical correspondence.
    log.info("Loading real PD slices (preprocessed)...")
    real_pd_slices = []
    for f in sorted(glob.glob(os.path.join(args.real_pd_dir, "**", "*.nii.gz"), recursive=True)):
        try:
            vol = process_volume(f, "PD")
            real_pd_slices.extend([vol[i] for i in range(vol.shape[0])])
        except Exception as e:
            log.warning(f"  Skipping real PD {f}: {e}")
        if len(real_pd_slices) >= args.max_slices: break
    real_pd_slices = real_pd_slices[:args.max_slices]
    log.info(f"  Real PD: {len(real_pd_slices)} slices")

    # ── Load registration net ─────────────────────────────────────────────────
    R_net = None
    if args.ckpt and os.path.exists(args.ckpt):
        log.info("Loading registration network...")
        R_net = RegistrationNet(nf=16).to(device)
        ckpt  = torch.load(args.ckpt, map_location=device, weights_only=False)
        R_net.load_state_dict(ckpt["R"])
        R_net.eval()

    results = {}

    # 1. FID (fake vs real) + baseline (dess vs real)
    results["FID_fake_vs_real"] = compute_fid(fake_pd_slices, real_pd_slices)
    results["FID_dess_vs_real"] = compute_fid_baseline(dess_slices[:len(real_pd_slices)], real_pd_slices)
    results["FID_improvement"]  = results["FID_dess_vs_real"] - results["FID_fake_vs_real"]

    # 2. KID
    kid_mean, kid_std = compute_kid(fake_pd_slices, real_pd_slices)
    results["KID_mean"] = kid_mean
    results["KID_std"]  = kid_std

    # 3. SSIM
    n_ssim = min(len(dess_slices), len(fake_pd_slices))
    results["SSIM_DESS_FakePD"] = compute_ssim(dess_slices[:n_ssim], fake_pd_slices[:n_ssim])

    # 4. Histogram
    plot_histograms(dess_slices, fake_pd_slices, real_pd_slices,
                    os.path.join(args.out_dir, "intensity_histogram.png"))

    # 5. Deformation + Jacobian
    flows_list = None
    if R_net is not None:
        n_def = min(len(dess_slices), len(fake_pd_slices), 100)
        deform_results, jac_arr, flows_list = evaluate_deformation(
            dess_slices[:n_def], fake_pd_slices[:n_def], R_net, device,
            mask_slices=mask_slices[:n_def] if mask_slices else None,
            meniscus_label=args.meniscus_label, n=n_def
        )
        results.update(deform_results)
        plot_jacobian(jac_arr, fake_pd_slices[:n_def],
                      os.path.join(args.out_dir, "jacobian_det.png"))

    # 6. Sample grid — meniscus slices with boundary
    plot_sample_grid(dess_slices, fake_pd_slices, real_pd_slices,
                     os.path.join(args.out_dir, "sample_grid.png"),
                     mask_slices=mask_slices,
                     meniscus_label=args.meniscus_label)

    # 7. Meniscus overlay — bright boundary contours
    if mask_slices is not None:
        try:
            plot_meniscus_overlays(dess_slices, fake_pd_slices, mask_slices,
                                    args.meniscus_label, args.out_dir)
        except Exception as e:
            log.warning(f"Meniscus overlay failed: {e}")

    # 8. Full knee boundary overlay
    if mask_slices is not None:
        try:
            plot_knee_boundary_overlay(dess_slices, fake_pd_slices, mask_slices,
                                        args.out_dir)
        except Exception as e:
            log.warning(f"Knee boundary overlay failed: {e}")

    # 9. Difference / anomaly map
    try:
        plot_difference_map(dess_slices, fake_pd_slices, args.out_dir,
                             mask_slices=mask_slices,
                             meniscus_label=args.meniscus_label)
    except Exception as e:
        log.warning(f"Difference map failed: {e}")

    # ── Save report ──────────────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, "metrics.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n{'='*55}")
    log.info("EVALUATION SUMMARY")
    log.info(f"{'='*55}")
    for k, v in results.items():
        log.info(f"  {k:45s}: {v:.6f}")
    log.info(f"{'='*55}")
    log.info(f"Report -> {report_path}")
    log.info(f"Plots  -> {args.out_dir}/")
    log.info("Output files:")
    log.info("  intensity_histogram.png  — distribution shift")
    log.info("  jacobian_det.png         — deformation topology")
    log.info("  sample_grid.png          — DESS/FakePD/RealPD at meniscus slices")
    log.info("  difference_map.png       — anomaly-style diff heatmap")
    log.info("  knee_boundary_overlay.png— full tissue boundaries")
    log.info("  meniscus_overlays/       — per-scan lateral+medial")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_pd_dir",    required=True)
    ap.add_argument("--real_pd_dir",    required=True)
    ap.add_argument("--dess_slice_dir", required=True)
    ap.add_argument("--mask_dir",       default=None)
    ap.add_argument("--ckpt",           default=None)
    ap.add_argument("--out_dir",        default="evaluation")
    ap.add_argument("--ngf",            type=int,   default=48)
    ap.add_argument("--n_res",          type=int,   default=9)
    ap.add_argument("--max_slices",     type=int,   default=500)
    ap.add_argument("--meniscus_label", type=int,   nargs="+", default=[5, 6])
    # STAGE 4b: restrict evaluation to a held-out split. Without these, all
    # available patients (which may include train-set patients) get evaluated,
    # inflating apparent performance.
    ap.add_argument("--splits", default=None,
                     help="Path to splits.json (required with --split)")
    ap.add_argument("--split",  default=None, choices=["train", "val", "test"],
                     help="Only evaluate DESS patients in this split. Omit "
                          "both --splits/--split for old behavior (no filtering).")
    args = ap.parse_args()
    if bool(args.splits) != bool(args.split):
        ap.error("--splits and --split must be given together")
    main(args)