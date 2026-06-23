"""
Check 2: Registration network non-triviality check.

Loads a checkpoint, runs R on paired slice batches, and inspects
the actual flow magnitude to confirm R is NOT trivially outputting
near-zero displacements (i.e., verifying the "0.029 px" claim is real).

Also plots a flow magnitude histogram saved to scripts/check_reg_histogram.png

Usage (run on BigRed where checkpoint lives):
    python scripts/check_registration.py \
        --ckpt /path/to/checkpoint_best.pth \
        --fake_pd_dir /path/to/fake_pd_npy_slices \
        --real_pd_dir /path/to/real_pd_npy_slices \
        --n 50

Must be run where the checkpoint is available (HPC).
"""

import argparse
import glob
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import RegistrationNet


def load_slices(directory, n):
    files = sorted(glob.glob(os.path.join(directory, "*.npy")))[:n]
    if not files:
        raise FileNotFoundError(f"No .npy files in {directory}")
    slices = [np.load(f).astype(np.float32) for f in files]
    return slices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",         required=True)
    ap.add_argument("--fake_pd_dir",  required=True, help="npy slices of fake PD")
    ap.add_argument("--real_pd_dir",  required=True, help="npy slices of real PD")
    ap.add_argument("--n",            type=int, default=50)
    ap.add_argument("--nf",           type=int, default=16, help="RegistrationNet nf")
    ap.add_argument("--img_size",     type=int, default=384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load model
    R = RegistrationNet(nf=args.nf).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    R.load_state_dict(ckpt["R"])
    R.eval()
    print(f"  Loaded checkpoint: {args.ckpt}")

    # Load slices
    fake_slices = load_slices(args.fake_pd_dir, args.n)
    real_slices = load_slices(args.real_pd_dir, args.n)
    n = min(len(fake_slices), len(real_slices))
    print(f"  Evaluating {n} slice pairs...")

    mags = []
    flow_all = []

    with torch.no_grad():
        for i in range(n):
            # Normalize [0,1] slices to [-1,1]
            f = torch.from_numpy(fake_slices[i]).unsqueeze(0).unsqueeze(0).to(device)
            r = torch.from_numpy(real_slices[i]).unsqueeze(0).unsqueeze(0).to(device)
            f = f * 2.0 - 1.0
            r = r * 2.0 - 1.0

            flow = R(f, r)  # (1, 2, H, W) in pixel units
            flow_np = flow[0].cpu().numpy()  # (2, H, W)
            mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)  # (H, W)
            mags.append(mag.mean())
            flow_all.append(flow_np)

    mags = np.array(mags)
    all_mags = np.concatenate([
        np.sqrt(f[0]**2 + f[1]**2).flatten() for f in flow_all
    ])

    print("\n" + "="*60)
    print(" REGISTRATION NETWORK FLOW MAGNITUDE REPORT")
    print("="*60)
    print(f"\n  Per-slice mean magnitude:")
    print(f"    mean:   {mags.mean():.6f} px")
    print(f"    std:    {mags.std():.6f} px")
    print(f"    min:    {mags.min():.6f} px")
    print(f"    max:    {mags.max():.6f} px")
    print(f"\n  Per-pixel magnitude (all {len(all_mags):,} pixels):")
    print(f"    p0    (min):  {np.percentile(all_mags, 0):.8f} px")
    print(f"    p25:          {np.percentile(all_mags, 25):.6f} px")
    print(f"    p50 (median): {np.percentile(all_mags, 50):.6f} px")
    print(f"    p75:          {np.percentile(all_mags, 75):.6f} px")
    print(f"    p99:          {np.percentile(all_mags, 99):.6f} px")
    print(f"    p100 (max):   {np.percentile(all_mags, 100):.6f} px")

    eps = 1e-7
    trivial_frac = (all_mags < eps).mean()
    print(f"\n  Fraction of pixels with |flow| < {eps:.0e}: {trivial_frac*100:.2f}%")
    if trivial_frac > 0.99:
        print("  *** WARNING: >99% of pixels have near-zero flow — R may have collapsed ***")
    elif trivial_frac > 0.50:
        print("  *** CAUTION: >50% of pixels have near-zero flow — check if R is learning ***")
    else:
        print("  OK: flow has meaningful non-zero values — R is producing real deformations")

    # Histogram
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_reg_histogram.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(all_mags[all_mags < np.percentile(all_mags, 99)], bins=100, color="steelblue", edgecolor="none")
    axes[0].set_title("Pixel flow magnitude distribution (p0–p99)")
    axes[0].set_xlabel("Displacement magnitude (px)")
    axes[0].set_ylabel("Count")
    axes[1].hist(mags, bins=30, color="darkorange", edgecolor="none")
    axes[1].set_title("Per-slice mean flow magnitude")
    axes[1].set_xlabel("Mean displacement magnitude (px)")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\n  Histogram saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
