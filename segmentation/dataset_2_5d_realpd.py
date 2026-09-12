"""
dataset_2_5d_realpd.py
=======================
2.5D dataset for real PD labeled data with binary masks (0=bg, 1=meniscus).
Same augmentation as v2. Stacks ±1 slice context.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class RealPDDataset(Dataset):
    def __init__(self, img_root, mask_root, augment=False, aug_mode="original"):
        """
        aug_mode: "original" (default) keeps this class's own augmentation
            exactly as before — additive brightness U(-0.1,0.1) at p=0.5,
            additive noise N(0,0.02) at p=0.3. Every existing caller
            (including finetune_meniscus_v7_replica.py, the fixed reference
            recipe) is unaffected unless it opts in.
            "equalized" matches dataset_2_5d_v2.py's fake-branch augmentation
            exactly: multiplicative brightness U(0.8,1.2) at p=1.0, additive
            noise N(0,0.02) at p=1.0, clip after each op. Requested by P4 so
            real and fake branches see the same augmentation distribution.
        """
        self.img_root  = img_root
        self.mask_root = mask_root
        self.augment   = augment
        self.aug_mode  = aug_mode
        self.slices    = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(img_root, "*.npy"))
        ])

    def __len__(self):
        return len(self.slices)

    def _load_vol_slices(self, stem):
        """Return (prev, curr, next) slices from the same patient volume."""
        pid, sidx = stem.rsplit("_", 1)
        sidx = int(sidx)

        def load_or_zeros(s):
            path = os.path.join(self.img_root, f"{pid}_{s:04d}.npy")
            if os.path.exists(path):
                return np.load(path).astype(np.float32)
            return np.zeros((384, 384), dtype=np.float32)

        return load_or_zeros(sidx - 1), load_or_zeros(sidx), load_or_zeros(sidx + 1)

    def __getitem__(self, idx):
        stem = self.slices[idx]
        prev, curr, nxt = self._load_vol_slices(stem)
        mask = np.load(os.path.join(self.mask_root, f"{stem}.npy")).astype(np.int64)

        if self.augment:
            # Horizontal flip
            if np.random.rand() < 0.5:
                prev, curr, nxt = prev[:, ::-1].copy(), curr[:, ::-1].copy(), nxt[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            # Vertical flip
            if np.random.rand() < 0.5:
                prev, curr, nxt = prev[::-1].copy(), curr[::-1].copy(), nxt[::-1].copy()
                mask = mask[::-1].copy()
            if self.aug_mode == "equalized":
                # Matches dataset_2_5d_v2.py's fake-branch augmentation exactly.
                factor = np.random.uniform(0.8, 1.2)
                prev, curr, nxt = (np.clip(a * factor, 0.0, 1.0) for a in (prev, curr, nxt))
                prev, curr, nxt = (np.clip(a + np.random.normal(0, 0.02, a.shape).astype(np.float32), 0.0, 1.0)
                                   for a in (prev, curr, nxt))
            else:
                # Brightness + noise (original)
                if np.random.rand() < 0.5:
                    for arr in [prev, curr, nxt]:
                        arr += np.random.uniform(-0.1, 0.1)
                if np.random.rand() < 0.3:
                    for arr in [prev, curr, nxt]:
                        arr += np.random.randn(*arr.shape).astype(np.float32) * 0.02

        image = np.stack([prev, curr, nxt], axis=0)
        image = np.clip(image, 0.0, 1.0)

        # patient_id lets per-patient (not just pooled) Dice be computed at
        # eval time. Training reads only "image"/"mask" — this key is inert there.
        pid, _ = stem.rsplit("_", 1)

        return {
            "image":      torch.from_numpy(image).float(),
            "mask":       torch.from_numpy(mask).long(),
            "patient_id": pid,
        }
