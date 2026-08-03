"""
dataset_2_5d_v3.py
===================
Same as dataset_2_5d_v2.py but adds random rotation ±15° to augmentation.
Changes vs v2:
  + Random rotation ±15° applied to image (bilinear) and mask (nearest-neighbour)
  Everything else (hflip, vflip, brightness, noise) unchanged.
"""
import glob
import os
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate as scipy_rotate


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
            idx = int(m.group(1))
            self.items.append((patient_id, idx))

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
            n = idx + offset
            sl = self._load_slice(patient_id, n)
            if sl is None:
                sl = self._load_slice(patient_id, idx)
            stack.append(sl)
        image = np.stack(stack, axis=0)  # (3, H, W), float32 in [0,1]

        mask_path = os.path.join(self.mask_root, patient_id, f"{patient_id}_slice_{idx:03d}.npy")
        mask = np.load(mask_path).astype(np.int64)  # (H, W)

        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                image = image[:, :, ::-1].copy()
                mask  = mask[:, ::-1].copy()

            # Vertical flip
            if random.random() > 0.5:
                image = image[:, ::-1, :].copy()
                mask  = mask[::-1, :].copy()

            # Random rotation ±15° — bilinear for image, nearest-neighbour for mask
            if random.random() > 0.5:
                angle = random.uniform(-15.0, 15.0)
                rotated = np.zeros_like(image)
                for c in range(image.shape[0]):
                    rotated[c] = scipy_rotate(
                        image[c], angle, reshape=False, order=1, mode="reflect"
                    )
                image = np.clip(rotated, 0.0, 1.0)
                mask = scipy_rotate(
                    mask.astype(np.float32), angle, reshape=False, order=0, mode="reflect"
                ).astype(np.int64)

            # Brightness/contrast jitter — image only
            factor = random.uniform(0.8, 1.2)
            image  = np.clip(image * factor, 0.0, 1.0)

            # Additive noise
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return {
            "image":      torch.from_numpy(image),
            "mask":       torch.from_numpy(mask),
            "patient_id": patient_id,
            "slice_idx":  idx,
        }
