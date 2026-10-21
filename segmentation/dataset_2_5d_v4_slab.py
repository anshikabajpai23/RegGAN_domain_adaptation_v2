"""
dataset_2_5d_v4_slab.py
=========================
2.5D dataset for N1's slab-averaged fake-PD data (segmentation_data_v4_slab/,
written by prepare_slab_fake_pd.py). Differs from dataset_2_5d_v2.py in two
ways, both required by slab data:

  1. Masks are FLOAT soft one-hot (3,H,W) fractions, not int64 hard labels
     (H,W) -- a slab can partially contain meniscus.
  2. Neighbour offsets are ±1 in SLAB space (spec: "stride 1 in slab space"),
     with edge clamping identical to infer_real_pd_v3.py's convention
     (max(0, min(n_slabs-1, idx+offset)) -- repeat the nearest valid slab at
     volume ends) rather than dataset_2_5d_v2.py's "fall back to center slice"
     behavior. Every slab file that survives prepare_slab_fake_pd.py's
     meniscus-fraction threshold exists on disk, but neighbours of a kept
     slab may have been thresholded out -- so this still needs a fallback,
     and it uses the same nearest-existing-file walk as v2's "nearest" mode,
     bounded by the clamped index.

Augmentation (hflip, vflip, brightness, noise) matches dataset_2_5d_v2.py
exactly so the fake branch's augmentation distribution is unchanged by N1 --
mask channels are flipped spatially alongside the image, and channel identity
(bg/lateral/medial) is untouched (no label remapping happens on flip, matching
the existing fake and real pipelines' convention of not swapping lateral/
medial on flip).
"""
import glob
import os
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class SlabMeniscus2_5DDataset(Dataset):
    def __init__(self, img_root, mask_root, augment=False):
        self.img_root = img_root
        self.mask_root = mask_root
        self.augment = augment
        self.items = []          # (patient_id, idx)
        self.n_slabs = {}        # patient_id -> slab count (max existing idx + 1)

        for img_path in sorted(glob.glob(os.path.join(img_root, "*", "*.npy"))):
            patient_id = os.path.basename(os.path.dirname(img_path))
            m = re.search(r"_slice_(\d{3})\.npy$", img_path)
            if not m:
                continue
            idx = int(m.group(1))
            self.items.append((patient_id, idx))
            self.n_slabs[patient_id] = max(self.n_slabs.get(patient_id, 0), idx + 1)

        assert len(self.items) > 0, f"No slabs found under {img_root}"

    def __len__(self):
        return len(self.items)

    def _load_slice(self, patient_id, idx):
        path = os.path.join(self.img_root, patient_id, f"{patient_id}_slice_{idx:03d}.npy")
        return np.load(path).astype(np.float32) if os.path.exists(path) else None

    def _load_mask(self, patient_id, idx):
        path = os.path.join(self.mask_root, patient_id, f"{patient_id}_slice_{idx:03d}.npy")
        return np.load(path).astype(np.float32) if os.path.exists(path) else None

    def _clamped_idx(self, patient_id, idx):
        n = self.n_slabs[patient_id]
        return max(0, min(n - 1, idx))

    def _load_nearest_existing(self, patient_id, target_idx, center_idx):
        """Walk from the clamped target back toward center; return the first
        slab that actually has a saved file (neighbours can be missing if
        they fell below the meniscus-fraction threshold)."""
        step = 1 if target_idx >= center_idx else -1
        n = target_idx
        while n != center_idx:
            sl = self._load_slice(patient_id, n)
            if sl is not None:
                return sl
            n -= step
        sl = self._load_slice(patient_id, center_idx)
        if sl is None:
            raise FileNotFoundError(
                f"Center slab {patient_id}_slice_{center_idx:03d} missing -- "
                f"should be impossible, it's in self.items")
        return sl

    def __getitem__(self, i):
        patient_id, idx = self.items[i]

        stack = []
        for offset in (-1, 0, 1):
            n = self._clamped_idx(patient_id, idx + offset)
            sl = self._load_slice(patient_id, n)
            if sl is None:
                sl = self._load_nearest_existing(patient_id, n, idx)
            stack.append(sl)
        image = np.stack(stack, axis=0)  # (3, H, W)

        mask = self._load_mask(patient_id, idx)  # (3, H, W) soft fractions

        if self.augment:
            if random.random() > 0.5:
                image = image[:, :, ::-1].copy()
                mask = mask[:, :, ::-1].copy()
            if random.random() > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1, :].copy()

            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0.0, 1.0)

            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "patient_id": patient_id,
            "slice_idx": idx,
        }
