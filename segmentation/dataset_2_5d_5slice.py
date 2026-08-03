"""
dataset_2_5d_5slice.py
=======================
Same as dataset_2_5d_v2.py but uses 5-slice context (±2 slices = 5 channels)
instead of 3-slice (±1 slice).

Change vs v2:
  + offsets = (-2, -1, 0, +1, +2) → 5-channel input
  Model must use in_channels=5.
  Augmentation: same as v2 (hflip, vflip, brightness, noise).
"""
import glob
import os
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset


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

        # 5-slice context: offsets ±2
        stack = []
        for offset in (-2, -1, 0, 1, 2):
            sl = self._load_slice(patient_id, idx + offset)
            if sl is None:
                sl = self._load_slice(patient_id, idx)
            stack.append(sl)
        image = np.stack(stack, axis=0)  # (5, H, W)

        mask_path = os.path.join(self.mask_root, patient_id, f"{patient_id}_slice_{idx:03d}.npy")
        mask = np.load(mask_path).astype(np.int64)

        if self.augment:
            if random.random() > 0.5:
                image = image[:, :, ::-1].copy()
                mask  = mask[:, ::-1].copy()
            if random.random() > 0.5:
                image = image[:, ::-1, :].copy()
                mask  = mask[::-1, :].copy()
            factor = random.uniform(0.8, 1.2)
            image  = np.clip(image * factor, 0.0, 1.0)
            noise  = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image  = np.clip(image + noise, 0.0, 1.0)

        return {
            "image":      torch.from_numpy(image),
            "mask":       torch.from_numpy(mask),
            "patient_id": patient_id,
            "slice_idx":  idx,
        }
