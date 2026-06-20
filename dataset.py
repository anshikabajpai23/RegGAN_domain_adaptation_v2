"""
dataset.py
==========
PyTorch Dataset for unpaired DESS / PD slices.
"""

import os, json, random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class UnpairedSliceDataset(Dataset):
    """
    Returns random pairs (dess_slice, pd_slice) – unpaired, CycleGAN-style.
    Each .npy file is float32 [0,1] with shape (H, W).
    """

    def __init__(self,
                 splits_json: str,
                 split: str = "train",      # "train" | "val" | "test"
                 aug: bool = True):
        with open(splits_json) as f:
            splits = json.load(f)

        self.dess_paths = splits["dess"][split]
        self.pd_paths   = splits["pd"][split]
        self.split = split
        self.aug = aug and (split == "train")

        assert len(self.dess_paths) > 0, f"No DESS slices for split={split}"
        assert len(self.pd_paths)   > 0, f"No PD slices for split={split}"

    def __len__(self):
        return max(len(self.dess_paths), len(self.pd_paths))

    def _load(self, path: str) -> torch.Tensor:
        sl = np.load(path).astype(np.float32)       # (H, W)
        t  = torch.from_numpy(sl).unsqueeze(0)       # (1, H, W)  in [0,1]
        t  = t * 2.0 - 1.0                           # -> [-1, 1] for tanh
        return t

    def _augment(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            t = TF.hflip(t)
        if random.random() > 0.5:
            t = TF.vflip(t)
        angle = random.uniform(-10, 10)
        # fill=-1.0: background in [-1,1]-normalized space is -1, not the
        # torchvision default of 0 (which is mid-gray here and would bake
        # gray corner artifacts into rotated training samples)
        t = TF.rotate(t, angle, fill=-1.0)
        return t

    def __getitem__(self, idx):
        d_path = self.dess_paths[idx % len(self.dess_paths)]
        if self.split == "train":
            # train split: random cross-patient PD pairing is intentional
            # (unpaired CycleGAN-style training)
            p_path = self.pd_paths[random.randint(0, len(self.pd_paths) - 1)]
        else:
            # val/test split: deterministic pairing so the same metric value
            # is reproducible across calls/epochs, not random noise
            p_path = self.pd_paths[idx % len(self.pd_paths)]

        dess = self._load(d_path)
        pd   = self._load(p_path)

        if self.aug:
            dess = self._augment(dess)
            pd   = self._augment(pd)

        return {"dess": dess, "pd": pd,
                "dess_path": d_path, "pd_path": p_path}
