"""
dataset_2_5d_v2.py
===================
Same as dataset_2_5d.py but adds optional augmentation for training:
  - Random horizontal flip
  - Random vertical flip
  - Random brightness/contrast jitter
Applied consistently to image+mask (spatial) or image-only (photometric).
"""
import glob
import os
import re
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class Meniscus2_5DDataset(Dataset):
    def __init__(self, img_root, mask_root, augment=False,
                 stride_choices=(1,), edge_fill="center", index_file=None):
        """
        stride_choices: neighbor offset k, drawn per __getitem__ call from this
            tuple; stack is [i-k, i, i+k]. Default (1,) reproduces the original
            fixed (-1,0,1) stack exactly — every existing caller is unaffected
            unless it opts in (P4 passes (4,5)).
        edge_fill: "center" (default, ORIGINAL behavior) falls back to the
            center slice whenever a neighbor file is missing (used both for
            true volume edges and for prep pipelines that skip empty slices).
            "nearest" walks from the missing neighbor back toward the center
            and uses the first slice that exists — matches the boundary-clamp
            convention in infer_real_pd_v3.py. Only meaningful (and only
            requested, by P4) when the data dir has no skipped interior
            slices (i.e. prepared with --min_meniscus_pixels 0), otherwise
            "nearest" would silently substitute a skipped-empty neighbor for
            a real gap rather than a true edge.
        index_file: optional path to a text file of "patient_id,idx" lines
            (one per row) restricting which slices this dataset serves,
            instead of every slice under img_root. Used to select a subset
            (e.g. mask-bearing-only, or mask-bearing + chosen negatives) out
            of a directory that was prepared with --min_meniscus_pixels 0
            and therefore contains every slice, positive and empty.
        """
        self.img_root  = img_root
        self.mask_root = mask_root
        self.augment   = augment
        self.stride_choices = stride_choices
        self.edge_fill = edge_fill
        self.items = []  # (patient_id, idx)

        if index_file is not None:
            with open(index_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    pid, idx_s = line.split(",")
                    self.items.append((pid, int(idx_s)))
        else:
            for img_path in sorted(glob.glob(os.path.join(img_root, "*", "*.npy"))):
                patient_id = os.path.basename(os.path.dirname(img_path))
                m = re.search(r"_slice_(\d{3})\.npy$", img_path)
                if not m:
                    continue
                idx = int(m.group(1))
                self.items.append((patient_id, idx))

        assert len(self.items) > 0, f"No slices found under {img_root}" \
            + (f" matching {index_file}" if index_file else "")

    def __len__(self):
        return len(self.items)

    def _load_slice(self, patient_id, idx):
        path = os.path.join(self.img_root, patient_id, f"{patient_id}_slice_{idx:03d}.npy")
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
        return None

    def _load_nearest(self, patient_id, target_idx, center_idx):
        """Walk from target_idx back toward center_idx, return the first slice
        that exists. Falls back to the center slice if none is found."""
        step = 1 if target_idx >= center_idx else -1
        n = target_idx
        while n != center_idx:
            sl = self._load_slice(patient_id, n)
            if sl is not None:
                return sl
            n -= step
        return self._load_slice(patient_id, center_idx)

    def __getitem__(self, i):
        patient_id, idx = self.items[i]
        k = random.choice(self.stride_choices)

        stack = []
        for offset in (-k, 0, k):
            n = idx + offset
            sl = self._load_slice(patient_id, n)
            if sl is None:
                sl = (self._load_nearest(patient_id, n, idx) if self.edge_fill == "nearest"
                      else self._load_slice(patient_id, idx))
            stack.append(sl)
        image = np.stack(stack, axis=0)  # (3, H, W), float32 in [0,1]

        mask_path = os.path.join(self.mask_root, patient_id, f"{patient_id}_slice_{idx:03d}.npy")
        mask = np.load(mask_path).astype(np.int64)  # (H, W), values in {0,1,2}

        if self.augment:
            # Horizontal flip — applied to all 3 channels + mask
            if random.random() > 0.5:
                image = image[:, :, ::-1].copy()
                mask  = mask[:, ::-1].copy()

            # Vertical flip
            if random.random() > 0.5:
                image = image[:, ::-1, :].copy()
                mask  = mask[::-1, :].copy()

            # Brightness/contrast jitter — image only, not mask
            factor = random.uniform(0.8, 1.2)
            image  = np.clip(image * factor, 0.0, 1.0)

            # Additive noise (subtle)
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return {
            "image":      torch.from_numpy(image),
            "mask":       torch.from_numpy(mask),
            "patient_id": patient_id,
            "slice_idx":  idx,
        }
