"""
dataset_2_5d.py
================
2.5D dataset matching pitthexai/Knee_MRI_Segmentation_2.5D's exact stacking
convention: for slice i, stack [i-1, i, i+1] as 3 channels (repeat current
slice if a neighbor is missing). Adapted to load OUR pseudo-PD .npy slices
(float32, [0,1]) + meniscus-filtered .npy masks (int64, 0/1/2) produced by
prepare_meniscus_masks.py, instead of their .jpg/.npy layout.
"""
import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset


class Meniscus2_5DDataset(Dataset):
    def __init__(self, img_root, mask_root):
        """
        img_root / mask_root: .../images/<patient_id>/*.npy , .../masks/<patient_id>/*.npy
        File naming: "{patient_id}_slice_{idx:03d}.npy" (idx = ORIGINAL slice
        index in the source volume -- preserved by prepare_meniscus_masks.py
        the same way preprocess.py preserves it for DESS slices).
        """
        self.img_root  = img_root
        self.mask_root = mask_root
        self.items = []  # (patient_id, idx)

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
                # Match reference repo: fall back to the CENTER slice if a
                # neighbor doesn't exist (volume boundary or gap from
                # meniscus-positive-only filtering).
                sl = self._load_slice(patient_id, idx)
            stack.append(sl)
        image = np.stack(stack, axis=0)  # (3, H, W), float32 in [0,1]

        mask_path = os.path.join(self.mask_root, patient_id, f"{patient_id}_slice_{idx:03d}.npy")
        mask = np.load(mask_path).astype(np.int64)  # (H, W), values in {0,1,2}

        return {
            "image": torch.from_numpy(image),
            "mask":  torch.from_numpy(mask),
            "patient_id": patient_id,
            "slice_idx": idx,
        }
