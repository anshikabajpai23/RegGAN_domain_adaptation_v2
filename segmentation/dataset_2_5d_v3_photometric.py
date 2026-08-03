"""
dataset_2_5d_v3_photometric.py
================================
Same as dataset_2_5d_v2.py but adds heavy photometric augmentation:
  + Random gamma correction (simulates PD vs fake PD contrast difference)
  + Random Gaussian blur (simulates different scanner sharpness)
  + Random contrast stretch

Purpose: fake PD ≠ real PD in texture/brightness. Heavy photometric aug
forces the model to be invariant to these differences → better generalisation.
All existing v2 augs (hflip, vflip, brightness, noise) unchanged.
"""
import glob
import os
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter


class Meniscus2_5DDataset(Dataset):
    def __init__(self, img_root, mask_root, augment=False):
        self.img_root  = img_root
        self.mask_root = mask_root
        self.augment   = augment
        self.items = []

        for img_path in sorted(glob.glob(os.path.join(img_root, "*", "*.npy"))):
            patient_id = os.path.basename(os.path.dirname(img_path))
            m = re.search(r"_slice_(\d{3})\.npy$", img_path)
            if not m:
                continue
            self.items.append((patient_id, int(m.group(1))))

        assert len(self.items) > 0, f"No slices found under {img_root}"

    def __len__(self):
        return len(self.items)

    def _load_slice(self, patient_id, idx):
        path = os.path.join(self.img_root, patient_id, f"{patient_id}_slice_{idx:03d}.npy")
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
        return None

    def __getitem__(self, i):
        patient_id, idx = self.items[i]

        stack = []
        for offset in (-1, 0, 1):
            sl = self._load_slice(patient_id, idx + offset)
            if sl is None:
                sl = self._load_slice(patient_id, idx)
            stack.append(sl)
        image = np.stack(stack, axis=0)  # (3, H, W)

        mask_path = os.path.join(self.mask_root, patient_id, f"{patient_id}_slice_{idx:03d}.npy")
        mask = np.load(mask_path).astype(np.int64)

        if self.augment:
            # Spatial: hflip, vflip (same as v2)
            if random.random() > 0.5:
                image = image[:, :, ::-1].copy()
                mask  = mask[:, ::-1].copy()
            if random.random() > 0.5:
                image = image[:, ::-1, :].copy()
                mask  = mask[::-1, :].copy()

            # v2 brightness
            factor = random.uniform(0.8, 1.2)
            image  = np.clip(image * factor, 0.0, 1.0)

            # NEW: Gamma correction — simulates PD vs fake PD contrast
            if random.random() > 0.5:
                gamma = random.uniform(0.7, 1.5)
                image = np.power(np.clip(image, 1e-8, 1.0), gamma).astype(np.float32)

            # NEW: Gaussian blur — simulates scanner PSF differences
            if random.random() > 0.5:
                sigma = random.uniform(0.3, 1.2)
                for c in range(image.shape[0]):
                    image[c] = gaussian_filter(image[c], sigma=sigma)
                image = np.clip(image, 0.0, 1.0)

            # NEW: Contrast stretch — randomly shift black/white points
            if random.random() > 0.5:
                lo = random.uniform(0.0, 0.1)
                hi = random.uniform(0.9, 1.0)
                image = np.clip((image - lo) / (hi - lo + 1e-8), 0.0, 1.0)

            # v2 noise
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return {
            "image":      torch.from_numpy(image),
            "mask":       torch.from_numpy(mask),
            "patient_id": patient_id,
            "slice_idx":  idx,
        }
