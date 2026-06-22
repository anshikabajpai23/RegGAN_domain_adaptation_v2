"""
infer.py — DESS -> PD translation
Volume shape after process_volume: (n_slices, 384, 384)
Slices along axis 0 (sagittal through-plane).
"""

import os, glob, json, argparse, logging
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


def get_effective_affine(nifti_path):
    """
    STAGE 5b FIX: build the output affine from the source image's actual
    direction matrix and origin (after RAS reorientation), instead of a
    plain np.diag([sp_R, sp_A, sp_S, 1.0]) which silently assumes:
      (a) direction is exactly identity (true only up to RAS canonicalization,
          not guaranteed bit-exact for all source files), and
      (b) origin is (0, 0, 0) -- this was always wrong; it discarded the
          volume's actual position in scanner/world space, so the saved
          fake-PD NIfTI could not be spatially overlaid with the original
          DESS volume (or a real PD volume) in a viewer that respects
          world coordinates.

    affine = direction_matrix @ diag(effective_spacing) , with the
    RAS-reoriented source image's origin as the translation column.
    """
    sp_R, sp_A, sp_S = get_effective_spacing(nifti_path)

    img       = sitk.ReadImage(nifti_path)
    img       = sitk.DICOMOrient(img, "RAS")
    direction = np.array(img.GetDirection()).reshape(3, 3)
    origin    = np.array(img.GetOrigin())

    # LPS -> RAS FIX: SimpleITK/ITK reports GetDirection()/GetOrigin() in
    # LPS coordinates (DICOM convention) regardless of DICOMOrient("RAS")
    # array-labeling -- that call only reorders the array, it doesn't change
    # the coordinate system values are reported in. nibabel's Nifti1Image
    # affine is defined in RAS+ world coordinates per the NIfTI standard.
    # Copying sitk's direction/origin directly into a nibabel affine without
    # this conversion silently flips left-right and anterior-posterior --
    # this is the standard, well-known LPS->RAS conversion (flip sign of the
    # first two axes, third axis unchanged).
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    direction  = lps_to_ras @ direction
    origin     = lps_to_ras @ origin

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction @ np.diag([sp_R, sp_A, sp_S])
    affine[:3, 3]  = origin

    return affine.astype(np.float32), (sp_R, sp_A, sp_S)


def get_split_patient_stems(splits_path, split):
    """
    STAGE 4b FIX: splits.json's dess/pd lists contain individual SLICE
    paths (e.g. ".../MTR_005_Anonymized_2378615199_e1_sl0086.npy"), not
    whole-volume paths. Extract the patient stem from each so we can filter
    whole-volume DESS NIfTIs down to only those belonging to `split`.
    """
    with open(splits_path) as f:
        splits = json.load(f)
    slice_paths = splits["dess"][split]
    stems = set(Path(p).stem.rsplit("_sl", 1)[0] for p in slice_paths)
    return stems


def run_inference(ckpt_path, dess_root, out_dir, ngf=64, n_res=9, batch_size=8,
                   splits_path=None, split=None):
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
    log.info(f"Found {len(nifti_files)} DESS volumes total.")

    if splits_path and split:
        stems = get_split_patient_stems(splits_path, split)
        before = len(nifti_files)
        nifti_files = [f for f in nifti_files
                       if Path(f).stem.replace(".nii", "") in stems]
        log.info(f"  Filtered to --split={split}: {before} -> {len(nifti_files)} volumes "
                  f"({len(stems)} patient stems in this split)")
    else:
        log.warning("  No --splits/--split given — translating ALL volumes "
                    "(train+val+test mixed). Pass --splits/--split to restrict "
                    "to held-out patients for evaluation.")

    os.makedirs(out_dir, exist_ok=True)

    for vpath in nifti_files:
        stem = Path(vpath).stem.replace(".nii", "")
        log.info(f"Translating {stem} ...")

        vol    = process_volume(vpath, "DESS")     # (n_slices, 384, 384)
        vol_pd = translate_volume(vol, G_AB, device, batch_size)

        # STAGE 5b FIX: affine now carries the source image's actual
        # direction matrix + origin (not just a zero-origin diagonal of
        # spacing values) — see get_effective_affine() docstring.
        affine, (sp_R, sp_A, sp_S) = get_effective_affine(vpath)
        log.info(f"  Effective spacing: R={sp_R:.3f} A={sp_A:.3f} S={sp_S:.3f} mm")

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
    # STAGE 4b: restrict translation to a held-out split for evaluation.
    # Without these, ALL volumes (train+val+test) get translated and any
    # downstream evaluation may include patients the GAN was trained on.
    ap.add_argument("--splits", default=None,
                     help="Path to splits.json (required with --split)")
    ap.add_argument("--split",  default=None, choices=["train", "val", "test"],
                     help="Only translate DESS volumes in this split. "
                          "Omit both --splits/--split to translate everything "
                          "(old behavior, train+val+test mixed).")
    args = ap.parse_args()
    if bool(args.splits) != bool(args.split):
        ap.error("--splits and --split must be given together")
    run_inference(args.ckpt, args.dess_root, args.out_dir,
                  args.ngf, args.n_res, args.batch_size,
                  args.splits, args.split)