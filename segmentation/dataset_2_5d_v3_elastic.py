"""
dataset_2_5d_v3_elastic.py
============================
Same as dataset_2_5d_v2.py but adds random elastic deformation to augmentation.

Change vs v2:
  + Elastic deformation (alpha=30px magnitude, sigma=4 smoothing)
    Applied to image (bilinear) and mask (nearest-neighbour) consistently.
  Everything else (hflip, vflip, brightness, noise) unchanged.

Elastic deformation simulates natural shape variability of the meniscus,
helping the model generalise to slightly different anatomical shapes in real PD.
"""
import glob
import os
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, map_coordinates


def elastic_transform(image, mask, alpha=30, sigma=4):
    """
    image: (C, H, W) float32
    mask:  (H, W)   int64
    alpha: max displacement in pixels
    sigma: gaussian smoothing (larger = smoother, more realistic deformation)
    """
    H, W = image.shape[1], image.shape[2]
    dx = gaussian_filter(np.random.randn(H, W), sigma) * alpha
    dy = gaussian_filter(np.random.randn(H, W), sigma) * alpha

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    indices_y = np.clip(y_coords + dy, 0, H - 1).ravel()
    indices_x = np.clip(x_coords + dx, 0, W - 1).ravel()
    coords = [indices_y, indices_x]

    out_image = np.zeros_like(image)
    for c in range(image.shape[0]):
        out_image[c] = map_coordinates(image[c], coords, order=1, mode="reflect").reshape(H, W)
    out_mask = map_coordinates(mask.astype(np.float32), coords, order=0, mode="reflect").reshape(H, W).astype(np.int64)

    return np.clip(out_image, 0.0, 1.0), out_mask


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
            if random.random() > 0.5:
                image = image[:, :, ::-1].copy()
                mask  = mask[:, ::-1].copy()
            if random.random() > 0.5:
                image = image[:, ::-1, :].copy()
                mask  = mask[::-1, :].copy()

            # Elastic deformation (p=0.5)
            if random.random() > 0.5:
                image, mask = elastic_transform(image, mask, alpha=30, sigma=4)

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
