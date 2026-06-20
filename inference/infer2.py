"""
infer.py — DESS -> PD translation
Volume shape after process_volume: (n_slices, 384, 384)
Slices along axis 0 (sagittal through-plane).
"""

import os, glob, argparse, logging
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
from pathlib import Path
from models import Generator
from preprocess import process_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def translate_volume(vol, G_AB, device, batch_size=8):
    n_slices   = vol.shape[0]
    translated = np.zeros_like(vol)
    G_AB.eval()
    with torch.no_grad():
        for start in range(0, n_slices, batch_size):
            end   = min(start + batch_size, n_slices)
            batch = vol[start:end, None, :, :]
            t     = torch.from_numpy(batch).to(device)
            t     = t * 2.0 - 1.0
            out   = G_AB(t)
            out   = (out + 1.0) / 2.0
            translated[start:end] = out.squeeze(1).cpu().numpy()
    return translated


def get_effective_spacing(nifti_path):
    """
    Compute the effective voxel spacing after preprocessing.
    After RAS reorientation:
      sitk.GetSpacing() -> (sp_x, sp_y, sp_z) = (sp_R, sp_A, sp_S)
      sitk.GetArrayFromImage() -> (z, y, x) = (S, A, R)
      so arr.shape = (n_S, n_A, n_R)
    
    Through-plane = R = axis 0 in our array after transpose in preprocess.py
    In-plane = A (dim1) and S (dim0) of sitk array
    """
    img      = sitk.ReadImage(nifti_path)
    img      = sitk.DICOMOrient(img, "RAS")
    sp       = img.GetSpacing()            # (sp_R, sp_A, sp_S)
    arr      = sitk.GetArrayFromImage(img) # (n_S, n_A, n_R)

    sp_R     = float(sp[0])   # through-plane spacing: unchanged
    sp_A_orig = float(sp[1])
    sp_S_orig = float(sp[2])
    n_A_orig  = arr.shape[1]  # A dim (sitk array dim 1)
    n_S_orig  = arr.shape[0]  # S dim (sitk array dim 0)

    # in-plane resample to isotropic (min spacing)
    target_ip = min(sp_A_orig, sp_S_orig)
    n_A_rs    = round(n_A_orig * sp_A_orig / target_ip)
    n_S_rs    = round(n_S_orig * sp_S_orig / target_ip)

    # after resize to 384x384
    eff_sp_A  = target_ip * n_A_rs / 384
    eff_sp_S  = target_ip * n_S_rs / 384

    # our array dim order after preprocess: (n_R, n_A, n_S)
    # diagonal affine maps dim0->sp_R, dim1->sp_A, dim2->sp_S
    return sp_R, eff_sp_A, eff_sp_S


def run_inference(ckpt_path, dess_root, out_dir, ngf=64, n_res=9, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    G_AB = Generator(in_ch=1, out_ch=1, ngf=ngf, n_res=n_res).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    G_AB.load_state_dict(ckpt["G_AB"])
    G_AB.eval()
    log.info(f"Loaded G_AB from {ckpt_path}")

    nifti_files = sorted(
        glob.glob(os.path.join(dess_root, "**", "*.nii.gz"), recursive=True) +
        glob.glob(os.path.join(dess_root, "**", "*.nii"),    recursive=True)
    )
    log.info(f"Found {len(nifti_files)} DESS volumes.")
    os.makedirs(out_dir, exist_ok=True)

    for vpath in nifti_files:
        stem = Path(vpath).stem.replace(".nii", "")
        log.info(f"Translating {stem} ...")

        vol    = process_volume(vpath, "DESS")     # (n_slices, 384, 384)
        vol_pd = translate_volume(vol, G_AB, device, batch_size)

        # Correct affine: simple diagonal with effective spacing
        sp_R, sp_A, sp_S = get_effective_spacing(vpath)
        log.info(f"  Effective spacing: R={sp_R:.3f} A={sp_A:.3f} S={sp_S:.3f} mm")

        # Simple diagonal affine (R,A,S) — ITK-SNAP will display correctly
        affine = np.diag([sp_R, sp_A, sp_S, 1.0]).astype(np.float32)

        new_img = nib.Nifti1Image(vol_pd.astype(np.float32), affine)
        new_img.header.set_zooms((sp_R, sp_A, sp_S))
        new_img.header.set_data_dtype(np.float32)

        out_path = os.path.join(out_dir, f"{stem}_pd_translated.nii.gz")
        nib.save(new_img, out_path)
        log.info(f"  Saved -> {out_path}  shape={vol_pd.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",       required=True)
    ap.add_argument("--dess_root",  default="data/skm-tea-dataset/dess-files")
    ap.add_argument("--out_dir",    default="results/translated_pd")
    # Defaults MUST match train.py defaults exactly (load_state_dict requires
    # identical architecture). If you trained with non-default --ngf/--n_res,
    # pass the same values here explicitly.
    ap.add_argument("--ngf",        type=int, default=48)
    ap.add_argument("--n_res",      type=int, default=9)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    run_inference(args.ckpt, args.dess_root, args.out_dir,
                  args.ngf, args.n_res, args.batch_size)