"""
dataset_sequence.py
====================
Dataset classes for the slice-sequence architecture (sequence_model.py).
Returns 5 individual single-channel slices [i-2,i-1,i,i+1,i+2] as a
(5, H, W) tensor, NOT a 3-channel stack — a fundamentally different
__getitem__ contract than Meniscus2_5DDataset/RealPDDataset, hence a
separate file rather than another opt-in parameter on those classes.

Edge handling: missing neighbor (only possible in the fake-PD case, where
segmentation_data_v2 was prepared with min_meniscus_pixels=10 and skips
interior slices too) falls back to the center slice, same convention as
the ORIGINAL (pre-P4) Meniscus2_5DDataset default -- deliberately the
simpler "center" rule, not P4's "nearest", since this experiment is testing
learned aggregation, not neighbor spacing; conflating the two would make a
negative result ambiguous about which change caused it.
"""
import glob
import os
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset

OFFSETS = (-2, -1, 0, 1, 2)


class SequenceMeniscusDataset(Dataset):
    """Fake-PD (DESS-derived) side. Same directory layout as Meniscus2_5DDataset."""
    def __init__(self, img_root, mask_root, augment=False):
        self.img_root, self.mask_root, self.augment = img_root, mask_root, augment
        self.items = []
        for img_path in sorted(glob.glob(os.path.join(img_root, "*", "*.npy"))):
            patient_id = os.path.basename(os.path.dirname(img_path))
            m = re.search(r"_slice_(\d{3})\.npy$", img_path)
            if m:
                self.items.append((patient_id, int(m.group(1))))
        assert len(self.items) > 0, f"No slices found under {img_root}"

    def __len__(self):
        return len(self.items)

    def _load(self, pid, idx):
        path = os.path.join(self.img_root, pid, f"{pid}_slice_{idx:03d}.npy")
        return np.load(path).astype(np.float32) if os.path.exists(path) else None

    def __getitem__(self, i):
        pid, idx = self.items[i]
        slices = [self._load(pid, idx + o) for o in OFFSETS]
        slices = [s if s is not None else self._load(pid, idx) for s in slices]
        image = np.stack(slices, axis=0)   # (5, H, W)

        mask = np.load(os.path.join(self.mask_root, pid, f"{pid}_slice_{idx:03d}.npy")).astype(np.int64)

        if self.augment:
            if random.random() > 0.5:
                image = image[:, :, ::-1].copy(); mask = mask[:, ::-1].copy()
            if random.random() > 0.5:
                image = image[:, ::-1, :].copy(); mask = mask[::-1, :].copy()
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0.0, 1.0)
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask),
                "patient_id": pid, "slice_idx": idx}


class SequenceRealPDDataset(Dataset):
    """Real-PD side. Same flat '{pid}_{sidx:04d}.npy' layout as RealPDDataset."""
    def __init__(self, img_root, mask_root, augment=False):
        self.img_root, self.mask_root, self.augment = img_root, mask_root, augment
        self.slices = sorted(os.path.splitext(os.path.basename(f))[0]
                             for f in glob.glob(os.path.join(img_root, "*.npy")))

    def __len__(self):
        return len(self.slices)

    def _load(self, pid, s):
        path = os.path.join(self.img_root, f"{pid}_{s:04d}.npy")
        return np.load(path).astype(np.float32) if os.path.exists(path) else np.zeros((384, 384), np.float32)

    def __getitem__(self, i):
        stem = self.slices[i]
        pid, sidx = stem.rsplit("_", 1)
        sidx = int(sidx)
        slices = [self._load(pid, sidx + o) for o in OFFSETS]
        image = np.stack(slices, axis=0)   # (5, H, W)

        mask = np.load(os.path.join(self.mask_root, f"{stem}.npy")).astype(np.int64)

        if self.augment:
            if np.random.rand() < 0.5:
                image = image[:, :, ::-1].copy(); mask = mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                image = image[:, ::-1, :].copy(); mask = mask[::-1, :].copy()
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0.0, 1.0)
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)
        image = np.clip(image, 0.0, 1.0)

        return {"image": torch.from_numpy(image).float(), "mask": torch.from_numpy(mask).long(),
                "patient_id": pid}
