"""
dataset_2_5d_tear.py
====================
Wraps the existing replica datasets (Meniscus2_5DDataset for fake PD,
RealPDDataset for real PD) without modifying them, and adds v14 targets:

  image    possibly with a synthetic tear painted in (training only)
  mask     UNCHANGED original label = the whole meniscus wedge (amodal target)
  visible  bool, mask > 0 minus the synthetic tear (what is actually dark)
  contour  bool, thin band on the outer edge of the whole wedge
  tear     bool, the painted tear (all zeros when nothing was painted)

The base dataset's own flips / brightness / noise run first; the tear is
painted afterwards, so its position always matches the flipped mask.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from tear_synth import synth_tear, contour_band


# [EXPLAIN] WRAPPER around the replica's two datasets (fake PD and real PD). It does not edit them.
# [EXPLAIN] REPLICA (before): each item = {image (3,384,384), mask (384,384) in {0,1,2}}.
# [EXPLAIN] NOW (after): same item PLUS 'visible', 'contour', 'tear' masks, and the image may contain a painted tear.
class TearAmodalDataset(Dataset):
    def __init__(self, base, synth=False, p=0.5, neighbor_p=0.6, fixed_seed=None):
        """
        base: a Meniscus2_5DDataset or RealPDDataset instance.
        synth: paint synthetic tears (use True only for training sets, or for the
            fixed-seed "synthetic-tear val" diagnostic).
        fixed_seed: if set, item i always gets the same tear (deterministic
            diagnostic set). If None, tears vary per epoch and per worker but
            stay reproducible for a given training seed.
        """
        self.base, self.synth, self.p, self.neighbor_p = base, synth, p, neighbor_p
        self.fixed_seed = fixed_seed

    def __len__(self):
        return len(self.base)

    # pass-through so the replica's eval helpers (which read .slices / .items) still work
    def __getattr__(self, name):
        if name in ("base", "synth", "p", "neighbor_p", "fixed_seed"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def _rng(self, i):
        if self.fixed_seed is not None:
            return np.random.default_rng(self.fixed_seed * 1_000_003 + i)
        return np.random.default_rng((torch.initial_seed() + i) % (2 ** 32))

    # [EXPLAIN] Order matters: (1) the base dataset applies the replica's own flip/brightness/noise, (2) THEN we paint the
    # [EXPLAIN] tear, so the tear always sits exactly where the (already flipped) mask is.
    def __getitem__(self, i):
        # [EXPLAIN] This is exactly what the replica would have used for this sample.
        item = self.base[i]
        image = item["image"].numpy().astype(np.float32)
        mask = item["mask"].numpy()
        tear = np.zeros(mask.shape, dtype=bool)
        # [EXPLAIN] Only training sets have synth=True (R1+). Validation stays clean so checkpoint selection matches the replica.
        # [EXPLAIN] The mask is NOT modified by painting: the label stays the WHOLE meniscus wedge (this is the 'amodal' idea).
        if self.synth:
            image, tear, _ = synth_tear(image, mask, self._rng(i), p=self.p, neighbor_p=self.neighbor_p)
        # [EXPLAIN] whole wedge = lateral + medial merged into one binary mask.
        whole = mask > 0
        item["image"] = torch.from_numpy(np.ascontiguousarray(image)).float()
        # [EXPLAIN] visible = whole wedge minus the painted tear = what is still dark. Used by the visible head (R2/R3/R4).
        item["visible"] = torch.from_numpy(whole & ~tear)
        # [EXPLAIN] contour = 1-pixel band on the wedge's outer edge. Used by the contour head (R3/R4/R1c).
        item["contour"] = torch.from_numpy(contour_band(whole))
        item["tear"] = torch.from_numpy(tear)
        return item
