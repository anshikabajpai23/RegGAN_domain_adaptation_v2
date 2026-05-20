"""
smoke_test.py
=============
End-to-end smoke test for the RegGAN pipeline.
Runs every stage on TINY synthetic data and prints images at each checkpoint.

Usage:
    python smoke_test.py                        # synthetic data (no real files needed)
    python smoke_test.py --real                 # use a few real NIfTI files
    python smoke_test.py --real --n_real 2      # limit to 2 volumes per modality
    python smoke_test.py --save_figs            # also save .png files to smoke_out/

The script will PRINT PASS / FAIL at every stage.
"""

import os, sys, json, shutil, argparse, traceback
import numpy as np
import torch

# ── optional matplotlib (graceful fallback to ASCII) ─────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # headless – saves PNGs instead of showing GUI
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

OUT_DIR = "smoke_out"
os.makedirs(OUT_DIR, exist_ok=True)

PASS = "\033[92m✔ PASS\033[0m"
FAIL = "\033[91m✘ FAIL\033[0m"


# ─────────────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def ok(msg): print(f"  {PASS}  {msg}")
def fail(msg): print(f"  {FAIL}  {msg}")


def show_slice(arr2d: np.ndarray,
               title: str,
               save: bool = True,
               ascii_fallback: bool = True):
    """Print a 2-D slice as ASCII art + optionally save PNG."""
    arr = arr2d.copy().astype(np.float32)
    # normalise to [0,1] for display
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    print(f"\n  ── {title}  shape={arr2d.shape}  "
          f"min={arr2d.min():.4f}  max={arr2d.max():.4f}  "
          f"mean={arr2d.mean():.4f}")

    # ASCII art (always printed to terminal)
    if ascii_fallback:
        _ascii_image(arr)

    # PNG save
    if save and HAS_MPL:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        fname = os.path.join(OUT_DIR, title.replace(" ", "_").replace("/", "-") + ".png")
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  PNG saved → {fname}")


def show_flow(flow2d: np.ndarray, title: str, save: bool = True):
    """Visualise a (2, H, W) deformation field as quiver plot."""
    dx = flow2d[0]
    dy = flow2d[1]
    mag = np.sqrt(dx**2 + dy**2)
    print(f"\n  ── {title}")
    print(f"     flow magnitude: mean={mag.mean():.6f}  max={mag.max():.6f}")

    if save and HAS_MPL:
        step = max(1, mag.shape[0] // 20)
        ys = np.arange(0, mag.shape[0], step)
        xs = np.arange(0, mag.shape[1], step)
        Y, X = np.meshgrid(ys, xs, indexing="ij")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(mag, cmap="hot")
        axes[0].set_title("Deformation magnitude")
        axes[0].axis("off")
        axes[1].quiver(X, Y, dx[::step, ::step], dy[::step, ::step],
                       scale=1, scale_units="xy", color="cyan")
        axes[1].invert_yaxis()
        axes[1].set_title("Deformation vectors")
        axes[1].set_facecolor("black")
        axes[1].axis("off")
        plt.suptitle(title, fontsize=8)
        fname = os.path.join(OUT_DIR, title.replace(" ", "_") + ".png")
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Flow PNG saved → {fname}")


def _ascii_image(arr: np.ndarray, width: int = 48, height: int = 20):
    """Render a 2-D float32 [0,1] array as ASCII art."""
    chars = " .:-=+*#%@"
    # downsample
    from scipy.ndimage import zoom as sz
    fy = height / arr.shape[0]
    fx = width  / arr.shape[1]
    small = sz(arr, (fy, fx), order=1)
    small = np.clip(small, 0, 1)
    rows = []
    for row in small:
        line = "".join(chars[int(v * (len(chars) - 1))] for v in row)
        rows.append("    |" + line + "|")
    print("\n".join(rows))


def show_multi(images: dict, title: str, save: bool = True):
    """Print / save a grid of named 2-D slices."""
    section(f"Image grid: {title}")
    for name, arr in images.items():
        show_slice(arr, name, save=save)

    if save and HAS_MPL and len(images) > 1:
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        for ax, (name, arr) in zip(axes, images.items()):
            a = arr.astype(np.float32)
            a = (a - a.min()) / (a.max() - a.min() + 1e-8)
            ax.imshow(a, cmap="gray", vmin=0, vmax=1)
            ax.set_title(name, fontsize=7)
            ax.axis("off")
        plt.suptitle(title, fontsize=9)
        fname = os.path.join(OUT_DIR, title.replace(" ", "_") + "_grid.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n  Grid PNG saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic NIfTI data generator
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_nifti(path: str,
                         shape=(64, 64, 20),
                         spacing=(0.5, 0.5, 1.0),
                         noise_scale: float = 0.3,
                         modality: str = "dess"):
    """
    Create a fake NIfTI volume that looks vaguely like a knee.
    DESS: darker background, bright cartilage ring
    PD:   brighter overall, softer contrast
    """
    import nibabel as nib
    H, W, D = shape
    vol = np.zeros((H, W, D), dtype=np.float32)
    cy, cx = H // 2, W // 2
    for z in range(D):
        Y, X = np.ogrid[:H, :W]
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        if modality == "dess":
            # bright ring, dark centre
            vol[:, :, z] = np.exp(-((r - H * 0.3)**2) / (2 * (H * 0.06)**2))
        else:
            # smoother, brighter PD-like
            vol[:, :, z] = 0.6 * np.exp(-(r**2) / (2 * (H * 0.35)**2)) + 0.2

    vol += noise_scale * np.random.randn(*vol.shape).astype(np.float32)
    vol = np.clip(vol, 0, None)

    affine = np.diag([*spacing, 1.0])
    img = nib.Nifti1Image(vol, affine)
    img.header.set_zooms(spacing)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(img, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Stage tests
# ─────────────────────────────────────────────────────────────────────────────

def test_imports():
    section("STAGE 0 — Import check")
    required = ["nibabel", "SimpleITK", "numpy", "scipy",
                "torch", "torchvision", "sklearn"]
    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            ok(pkg)
        except ImportError as e:
            fail(f"{pkg}  →  {e}")
            all_ok = False

    if HAS_MPL:
        ok("matplotlib (PNG output enabled)")
    else:
        print("  ℹ  matplotlib not found – ASCII art only, no PNGs")

    return all_ok


def test_synthetic_data(save_figs: bool = True):
    section("STAGE 1 — Synthetic NIfTI creation")
    dess_path = os.path.join(OUT_DIR, "synthetic/dess/vol_001.nii.gz")
    pd_path   = os.path.join(OUT_DIR, "synthetic/pd/vol_001.nii.gz")

    try:
        make_synthetic_nifti(dess_path, modality="dess")
        make_synthetic_nifti(pd_path,   modality="pd")
        ok(f"DESS NIfTI → {dess_path}")
        ok(f"PD   NIfTI → {pd_path}")

        import nibabel as nib
        d = nib.load(dess_path).get_fdata(dtype=np.float32)
        p = nib.load(pd_path).get_fdata(dtype=np.float32)

        mid = d.shape[2] // 2
        show_multi({
            f"DESS mid-slice (z={mid})":   d[:, :, mid],
            f"PD   mid-slice (z={mid})":   p[:, :, mid],
        }, "Stage1_synthetic_volumes", save=save_figs)

        return True, dess_path, pd_path
    except Exception:
        fail(traceback.format_exc())
        return False, None, None


def test_preprocessing(dess_path: str, pd_path: str, save_figs: bool = True):
    section("STAGE 2 — Preprocessing (reorient → resample → resize → normalise)")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from preprocess import process_volume, TARGET_SPACING, TARGET_SIZE, SLICE_AXIS

        print(f"  Config: TARGET_SPACING={TARGET_SPACING}  "
              f"TARGET_SIZE={TARGET_SIZE}  SLICE_AXIS={SLICE_AXIS}")

        dess_vol = process_volume(dess_path, "DESS")
        pd_vol   = process_volume(pd_path,   "PD")

        assert dess_vol.dtype == np.float32, "DESS dtype not float32"
        assert pd_vol.dtype   == np.float32, "PD dtype not float32"
        assert dess_vol.min() >= -1e-5 and dess_vol.max() <= 1+1e-5, "DESS out of [0,1]"
        assert pd_vol.min()   >= -1e-5 and pd_vol.max()   <= 1+1e-5, "PD out of [0,1]"

        ok(f"DESS processed: shape={dess_vol.shape}")
        ok(f"PD   processed: shape={pd_vol.shape}")

        mid_d = dess_vol.shape[SLICE_AXIS] // 2
        mid_p = pd_vol.shape[SLICE_AXIS]   // 2

        show_multi({
            f"DESS processed (z={mid_d})": np.take(dess_vol, mid_d, axis=SLICE_AXIS),
            f"PD   processed (z={mid_p})": np.take(pd_vol,   mid_p, axis=SLICE_AXIS),
        }, "Stage2_preprocessed_slices", save=save_figs)

        return True, dess_vol, pd_vol
    except Exception:
        fail(traceback.format_exc())
        return False, None, None


def test_slice_extraction(dess_vol, pd_vol, save_figs: bool = True):
    section("STAGE 3 — Slice extraction")
    try:
        from preprocess import extract_slices, SLICE_AXIS

        dess_slice_dir = os.path.join(OUT_DIR, "slices/dess")
        pd_slice_dir   = os.path.join(OUT_DIR, "slices/pd")

        d_paths = extract_slices(dess_vol, dess_slice_dir, prefix="dess_001")
        p_paths = extract_slices(pd_vol,   pd_slice_dir,   prefix="pd_001")

        assert len(d_paths) > 0, "No DESS slices extracted"
        assert len(p_paths) > 0, "No PD slices extracted"
        ok(f"DESS: {len(d_paths)} slices")
        ok(f"PD:   {len(p_paths)} slices")

        # Load and inspect a few slices
        sample_d = np.load(d_paths[len(d_paths) // 2])
        sample_p = np.load(p_paths[len(p_paths) // 2])

        ok(f"DESS slice shape={sample_d.shape} min={sample_d.min():.4f} max={sample_d.max():.4f}")
        ok(f"PD   slice shape={sample_p.shape} min={sample_p.min():.4f} max={sample_p.max():.4f}")

        show_multi({
            "DESS extracted slice": sample_d,
            "PD   extracted slice": sample_p,
        }, "Stage3_extracted_slices", save=save_figs)

        return True, d_paths, p_paths
    except Exception:
        fail(traceback.format_exc())
        return False, None, None


def test_dataset(d_paths, p_paths, save_figs: bool = True):
    section("STAGE 4 — PyTorch Dataset & DataLoader")
    try:
        # Write a tiny splits.json
        def split(paths):
            n = len(paths)
            tr = paths[:max(1, n-2)]
            va = paths[max(1, n-2):max(1, n-1)]
            te = paths[max(1, n-1):]
            return {"train": tr, "val": va or [tr[0]], "test": te or [tr[0]]}

        splits = {"dess": split(d_paths), "pd": split(p_paths)}
        splits_path = os.path.join(OUT_DIR, "splits.json")
        with open(splits_path, "w") as f:
            json.dump(splits, f)
        ok(f"splits.json written → {splits_path}")

        from dataset import UnpairedSliceDataset
        from torch.utils.data import DataLoader

        ds = UnpairedSliceDataset(splits_path, split="train", aug=True)
        dl = DataLoader(ds, batch_size=2, shuffle=True)
        ok(f"Dataset length = {len(ds)}")

        batch = next(iter(dl))
        dess_t = batch["dess"]
        pd_t   = batch["pd"]

        ok(f"Batch DESS: shape={tuple(dess_t.shape)} range=[{dess_t.min():.2f}, {dess_t.max():.2f}]")
        ok(f"Batch PD:   shape={tuple(pd_t.shape)}   range=[{pd_t.min():.2f}, {pd_t.max():.2f}]")

        # Denorm from [-1,1] to [0,1] for display
        d_show = ((dess_t[0, 0].numpy() + 1) / 2)
        p_show = ((pd_t  [0, 0].numpy() + 1) / 2)
        show_multi({
            "DataLoader DESS (augmented)": d_show,
            "DataLoader PD   (augmented)": p_show,
        }, "Stage4_dataloader_batch", save=save_figs)

        return True, splits_path, (dess_t, pd_t)
    except Exception:
        fail(traceback.format_exc())
        return False, None, None


def test_models(batch_tensors, save_figs: bool = True):
    section("STAGE 5 — Model forward passes")
    dess_t, pd_t = batch_tensors
    device = torch.device("cpu")   # smoke test always on CPU

    try:
        from models import (Generator, PatchDiscriminator, RegistrationNet,
                            GANLoss, gradient_smoothness_loss, deformation_magnitude_loss)

        G_AB = Generator(in_ch=1, out_ch=1, ngf=32, n_res=3)   # tiny for speed
        G_BA = Generator(in_ch=1, out_ch=1, ngf=32, n_res=3)
        D_B  = PatchDiscriminator(in_ch=1, ndf=32, n_layers=2)
        R    = RegistrationNet(nf=8)

        real_A = dess_t[:1].to(device)   # (1, 1, H, W)
        real_B = pd_t[:1].to(device)

        # Generator
        fake_B = G_AB(real_A)
        ok(f"G_AB forward: {tuple(real_A.shape)} -> {tuple(fake_B.shape)}")
        assert fake_B.shape == real_A.shape

        # Discriminator
        patch = D_B(fake_B)
        ok(f"D_B  forward: {tuple(fake_B.shape)} -> {tuple(patch.shape)}")

        # Registration
        flow   = R(fake_B, real_B)
        warped = RegistrationNet.warp(fake_B, flow)
        ok(f"R    forward: flow={tuple(flow.shape)}  warped={tuple(warped.shape)}")

        # Losses
        crit = GANLoss()
        l_g = crit(patch, True)
        l_s = gradient_smoothness_loss(flow)
        l_m = deformation_magnitude_loss(flow)
        ok(f"GAN loss = {l_g.item():.4f}")
        ok(f"Smoothness loss = {l_s.item():.6f}")
        ok(f"Magnitude loss  = {l_m.item():.6f}")

        # Visualise model outputs
        def t2np(t): return ((t[0, 0].detach().numpy() + 1) / 2).clip(0, 1)
        show_multi({
            "Input DESS (real_A)": t2np(real_A),
            "G_AB output (fake PD)": t2np(fake_B),
            "Real PD (real_B)": t2np(real_B),
            "Warped fake_B": t2np(warped),
        }, "Stage5_model_outputs", save=save_figs)

        show_flow(flow[0].detach().numpy(), "Stage5_deformation_field", save=save_figs)

        return True
    except Exception:
        fail(traceback.format_exc())
        return False


def test_one_train_step(splits_path: str, save_figs: bool = True):
    section("STAGE 6 — One training step (no GPU needed)")
    import tempfile
    tmp_run = tempfile.mkdtemp(prefix="smoke_run_", dir=OUT_DIR)

    try:
        from train import RegGANTrainer

        class Args:
            splits        = splits_path
            out_dir       = tmp_run
            resume        = None
            ngf           = 32
            ndf           = 32
            n_res         = 3
            nf_reg        = 8
            epochs        = 2
            batch_size    = 2
            lr            = 2e-4
            lr_reg        = 1e-4
            num_workers   = 0      # 0 workers for smoke test
            log_interval  = 1
            lambda_cycle      = 10.0
            lambda_reg_sim    = 5.0
            lambda_reg_smooth = 10.0
            lambda_reg_mag    = 5.0

        # Force CPU for smoke test — MPS lacks grid_sampler_2d_backward
        import unittest.mock as mock
        with mock.patch('torch.cuda.is_available', return_value=False),              mock.patch('torch.backends.mps.is_available', return_value=False):
            trainer = RegGANTrainer(Args())
        ok("RegGANTrainer initialised (forced CPU for smoke test)")

        # Run one batch manually
        batch = next(iter(trainer.train_loader))
        metrics = trainer._step(batch, global_step=0)
        ok(f"Training step completed: G_total={metrics['G/total']:.4f}")
        ok(f"  cycle_A={metrics['G/cycle_A']:.4f}  "
           f"reg_sim={metrics['G/reg_sim']:.4f}  "
           f"smooth={metrics['G/smooth']:.6f}  "
           f"mag={metrics['G/mag']:.6f}")

        # Confirm minimum-deformation losses are finite and positive
        assert metrics["G/smooth"] >= 0, "Smoothness loss negative!"
        assert metrics["G/mag"]    >= 0, "Magnitude loss negative!"
        ok("Minimum-deformation constraints are active and finite")

        trainer.writer.close()

        # Check checkpoint can be saved/loaded
        trainer._save_checkpoint(0, "smoke")
        ckpt_path = os.path.join(tmp_run, "ckpt_smoke.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ok(f"Checkpoint saved & loaded: keys={list(ckpt.keys())}")

        # Visualise one sample from validation
        val_batch = next(iter(trainer.val_loader))
        with torch.no_grad():
            real_A = val_batch["dess"][:1].to(trainer.device)
            real_B = val_batch["pd"][:1].to(trainer.device)
            fake_B = trainer.G_AB(real_A)
            flow   = trainer.R(fake_B, real_B)
            warped = trainer.R.warp(fake_B, flow)

        def t2np(t): return ((t[0, 0].detach().cpu().numpy() + 1) / 2).clip(0, 1)
        show_multi({
            "After 1 step — DESS input": t2np(real_A),
            "After 1 step — fake PD":   t2np(fake_B),
            "After 1 step — real PD":   t2np(real_B),
            "After 1 step — warped":    t2np(warped),
        }, "Stage6_after_one_train_step", save=save_figs)

        show_flow(flow[0].detach().cpu().numpy(),
                  "Stage6_deformation_after_1step", save=save_figs)

        return True
    except Exception:
        fail(traceback.format_exc())
        return False


def test_real_data(dess_root: str, pd_root: str,
                   n_real: int = 1, save_figs: bool = True):
    section(f"STAGE 7 — Real data check ({n_real} volume(s) each)")
    import glob
    from preprocess import process_volume, SLICE_AXIS

    def find(root):
        f = sorted(
            glob.glob(os.path.join(root, "**", "*.nii.gz"), recursive=True) +
            glob.glob(os.path.join(root, "**", "*.nii"),    recursive=True)
        )
        return f[:n_real]

    dess_files = find(dess_root)
    pd_files   = find(pd_root)

    if not dess_files:
        fail(f"No NIfTI files found in {dess_root}")
        return False
    if not pd_files:
        fail(f"No NIfTI files found in {pd_root}")
        return False

    ok(f"DESS files: {[os.path.basename(f) for f in dess_files]}")
    ok(f"PD   files: {[os.path.basename(f) for f in pd_files]}")

    for tag, paths, mod in [("DESS", dess_files, "DESS"), ("PD", pd_files, "PD")]:
        for vpath in paths:
            try:
                import nibabel as nib
                raw_img = nib.load(vpath)
                raw = raw_img.get_fdata(dtype=np.float32)
                sp  = raw_img.header.get_zooms()[:3]
                print(f"\n  [{tag}] {os.path.basename(vpath)}")
                print(f"        raw shape={raw.shape}  spacing={tuple(round(float(s),2) for s in sp)}")

                mid = raw.shape[SLICE_AXIS] // 2
                show_slice(np.take(raw, mid, axis=SLICE_AXIS),
                           f"Real {tag} raw mid-slice", save=save_figs)

                vol = process_volume(vpath, mod)
                mid2 = vol.shape[SLICE_AXIS] // 2
                show_slice(np.take(vol, mid2, axis=SLICE_AXIS),
                           f"Real {tag} preprocessed mid-slice", save=save_figs)

                ok(f"{tag} preprocessed: {raw.shape} -> {vol.shape}")
            except Exception:
                fail(traceback.format_exc())
                return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real",      action="store_true",
                    help="Also test with real NIfTI files")
    ap.add_argument("--dess_root", default="data/skm-tea-dataset/nifti")
    ap.add_argument("--pd_root",   default="data/iu-dataset/pd-files")
    ap.add_argument("--n_real",    type=int, default=1,
                    help="Number of real volumes to use for the real-data check")
    ap.add_argument("--save_figs", action="store_true", default=True,
                    help="Save PNG figures to smoke_out/ (default: on)")
    ap.add_argument("--no_figs",   action="store_true",
                    help="Disable PNG saving (ASCII art only)")
    args = ap.parse_args()
    save = args.save_figs and not args.no_figs

    results = {}

    # Stage 0: imports
    results["imports"] = test_imports()

    # Stage 1: synthetic data
    ok1, dess_path, pd_path = test_synthetic_data(save_figs=save)
    results["synthetic_data"] = ok1
    if not ok1:
        print("\nCannot continue without NIfTI creation. Fix the error above.")
        return

    # Stage 2: preprocessing
    ok2, dess_vol, pd_vol = test_preprocessing(dess_path, pd_path, save_figs=save)
    results["preprocessing"] = ok2
    if not ok2:
        print("\nPreprocessing failed. Fix before continuing.")
        return

    # Stage 3: slice extraction
    ok3, d_paths, p_paths = test_slice_extraction(dess_vol, pd_vol, save_figs=save)
    results["slice_extraction"] = ok3
    if not ok3:
        return

    # Stage 4: dataset
    ok4, splits_path, batch_tensors = test_dataset(d_paths, p_paths, save_figs=save)
    results["dataset"] = ok4
    if not ok4:
        return

    # Stage 5: models
    results["models"] = test_models(batch_tensors, save_figs=save)

    # Stage 6: one training step
    results["train_step"] = test_one_train_step(splits_path, save_figs=save)

    # Stage 7: real data (optional)
    if args.real:
        results["real_data"] = test_real_data(
            args.dess_root, args.pd_root, args.n_real, save_figs=save
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    section("SMOKE TEST SUMMARY")
    all_passed = True
    for stage, passed in results.items():
        if passed:
            ok(stage)
        else:
            fail(stage)
            all_passed = False

    print()
    if all_passed:
        print("  \033[92m🎉 All stages passed! Ready for full training.\033[0m")
        if save and HAS_MPL:
            print(f"  Images saved in: {os.path.abspath(OUT_DIR)}/")
        print()
        print("  Next steps:")
        print("  1.  python preprocess.py --dess_root data/skm-tea-dataset/nifti "
              "--pd_root data/iu-dataset/pd-files")
        print("  2.  python train.py --splits data/preprocessed/splits.json "
              "--out_dir runs/reggan_001")
        print("  3.  tensorboard --logdir runs/reggan_001/tb")
    else:
        print("  \033[91m⚠  Some stages failed. Fix the errors above before training.\033[0m")


if __name__ == "__main__":
    main()
