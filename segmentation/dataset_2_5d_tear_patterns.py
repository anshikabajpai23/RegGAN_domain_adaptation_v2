"""
dataset_2_5d_tear_patterns.py
=============================
Wraps the replica datasets (Meniscus2_5DDataset for fake PD, RealPDDataset for real PD)
without modifying them, and applies one v15 synthetic tear pattern (tear_patterns.py) per
sample. Unlike v14, the returned "mask" is the UPDATED label (removed tissue -> background,
displaced tissue -> meniscus at its new place), so the replica's own losses train on it as-is.

The base dataset's flips / brightness / noise run first; the pattern is applied afterwards.

Optional logging of what training ACTUALLY sees (set log_dir and, per epoch, log_tag):
  * one statistics row for EVERY synthetic sample generated while log_tag is set
    -> <log_dir>/epochNNN/stats_<source>_w<worker>_p<pid>.csv
  * the full sample (3-slice image + label, before and after) for the first `save_per_mode`
    samples of each mode in each worker -> <log_dir>/epochNNN/samples/<mode>/*.npz
Each DataLoader worker writes its own stats file (no locking needed). The save limit is checked
against the files already on disk, because the real-PD loader is restarted ~21 times per epoch
(new worker processes each time); a few extra samples can slip through when workers race.
Stats rows include those repeats on purpose: they reflect how often training actually saw each case.
"""
import glob
import csv
import os

import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from tear_patterns import synth_pattern, MODES, MODE_P
from tear_synth import PATTERNS_V17

_QUOTA_FULL = set()     # per process: (log_tag, source, mode) whose save quota is already reached

# v18 pattern classes (fine level): none, each in-place variant, each other mode. Mode-level
# probabilities (e.g. "inplace") are sums over the matching fine classes.
PATTERN_CLASSES = (("none",) + tuple(f"inplace_{v}" for v in PATTERNS_V17)
                   + tuple(m for m in MODES if m != "inplace"))


class TearPatternDataset(Dataset):
    def __init__(self, base, synth=False, p=0.5, modes=MODES, mode_p=MODE_P, fixed_seed=None,
                 source="", log_dir=None, save_per_mode=0,
                 inplace_variants=None, inplace_p=None, log_variants=False, return_tear=False):
        """
        base: a Meniscus2_5DDataset or RealPDDataset.
        synth: apply patterns (training sets, or the fixed-seed per-mode diagnostic sets).
        p: probability that a meniscus slice gets a pattern.
        modes / mode_p: which patterns and how often (default: all, see tear_patterns.MODE_P).
        fixed_seed: if set, item i always gets the same pattern (deterministic diagnostic set).
        source: "fake" or "real", recorded in the logs.
        log_dir / save_per_mode: logging, active only while self.log_tag is an int (set per epoch).
        v17+ options (defaults = v15):
        inplace_variants / inplace_p: in-horn variants for "inplace" (tear_synth.PATTERNS_V17 adds Fig. 10a).
        log_variants: save samples per VARIANT (samples/inplace_horizontal/, ..., samples/flap/) and add
            variant / variant_all / tear_px columns to the stats; the npz also stores the tear map.
        return_tear: add item["tear"] (H, W) bool and item["pattern_id"] (index into PATTERN_CLASSES) for v18.
        """
        self.base, self.synth, self.p = base, synth, p
        self.modes, self.mode_p, self.fixed_seed = tuple(modes), tuple(mode_p), fixed_seed
        self.source, self.log_dir, self.save_per_mode, self.log_tag = source, log_dir, save_per_mode, None
        self.inplace_variants, self.inplace_p = inplace_variants, inplace_p
        self.log_variants, self.return_tear = log_variants, return_tear

    def __len__(self):
        return len(self.base)

    # pass-through so the replica/v14 eval helpers (which read .slices / .items) still work
    def __getattr__(self, name):
        if name in ("base", "synth", "p", "modes", "mode_p", "fixed_seed",
                    "source", "log_dir", "save_per_mode", "log_tag",
                    "inplace_variants", "inplace_p", "log_variants", "return_tear"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def _rng(self, i):
        if self.fixed_seed is not None:
            return np.random.default_rng(self.fixed_seed * 1_000_003 + i)
        return np.random.default_rng((torch.initial_seed() + i) % (2 ** 32))

    def _pid_slice(self, i):
        if hasattr(self.base, "items"):              # Meniscus2_5DDataset (fake PD)
            pid, idx = self.base.items[i]
            return str(pid), int(idx)
        pid, s = self.base.slices[i].rsplit("_", 1)  # RealPDDataset
        return pid, int(s)

    def _log(self, i, img0, msk0, img1, msk1, mode, info=None):
        tag = int(self.log_tag)
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi is not None else 0
        d = os.path.join(self.log_dir, f"epoch{tag:03d}")
        pid, sidx = self._pid_slice(i)
        m0, m1 = msk0 > 0, msk1 > 0
        diff = np.abs(img1[1] - img0[1]) > 0.02
        folder = info["variant"] if info is not None else mode       # v17+: one folder per variant
        key = (tag, self.source, folder)
        save = False
        if self.save_per_mode > 0 and key not in _QUOTA_FULL:
            n_on_disk = len(glob.glob(os.path.join(d, "samples", folder, f"{self.source}_*.npz")))
            save = n_on_disk < self.save_per_mode
            if not save:
                _QUOTA_FULL.add(key)
        row = {
            "epoch": tag, "source": self.source, "mode": mode, "pid": pid, "slice": sidx, "index": int(i),
            "worker": wid, "process": os.getpid(),
            "meniscus_px_before": int(m0.sum()), "meniscus_px_after": int(m1.sum()),
            "pieces_before": int(ndi.label(m0)[1]), "pieces_after": int(ndi.label(m1)[1]),
            "removed_px": int((m0 & ~m1).sum()), "added_px": int((m1 & ~m0).sum()),
            "image_changed_px": int(diff.sum()),
            "mean_int_meniscus_before": float(img0[1][m0].mean()) if m0.any() else float("nan"),
            "mean_int_meniscus_after": float(img1[1][m1].mean()) if m1.any() else float("nan"),
            "mean_int_changed_before": float(img0[1][diff].mean()) if diff.any() else float("nan"),
            "mean_int_changed_after": float(img1[1][diff].mean()) if diff.any() else float("nan"),
            "saved_file": f"samples/{folder}/{self.source}_{pid}_{sidx:04d}_i{int(i)}.npz" if save else "",
        }
        extra, imgs = {}, {"image_before": img0.astype(np.float16), "image_after": img1.astype(np.float16)}
        if info is not None:
            row.update({"variant": info["variant"], "variant_all": info["variant_all"],
                        "tear_px": int(info["tear"].sum())})
            extra = {"tear": info["tear"].astype(np.int8), "variant": info["variant"], "variant_all": info["variant_all"],
                     "image_scale": 255}
            # v17+: images as uint8 (value x 255), 2.4x smaller; read with synthetic_log_utils.load_sample
            imgs = {k: np.round(np.clip(v.astype(np.float32), 0, 1) * 255).astype(np.uint8) for k, v in imgs.items()}
        if save:
            os.makedirs(os.path.join(d, "samples", folder), exist_ok=True)
            np.savez_compressed(os.path.join(d, row["saved_file"]), **imgs,
                                mask_before=msk0.astype(np.int8), mask_after=msk1.astype(np.int8),
                                mode=mode, source=self.source, pid=pid, slice=sidx, **extra)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"stats_{self.source}_w{wid}_p{os.getpid()}.csv")
        new_file = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new_file:
                w.writeheader()
            w.writerow(row)

    def __getitem__(self, i):
        item = self.base[i]
        mode = "none"
        info = None
        if self.synth:
            image = item["image"].numpy().astype(np.float32)
            mask = item["mask"].numpy()
            if self.log_variants or self.return_tear or self.inplace_variants is not None:
                out, new, mode, info = synth_pattern(image, mask, self._rng(i), p=self.p, modes=self.modes,
                                                     mode_p=self.mode_p, inplace_variants=self.inplace_variants,
                                                     inplace_p=self.inplace_p, return_info=True)
            else:                                                    # v15 path, unchanged
                out, new, mode = synth_pattern(image, mask, self._rng(i), p=self.p,
                                               modes=self.modes, mode_p=self.mode_p)
            if self.log_dir and self.log_tag is not None and mode != "none":
                self._log(i, image, mask, out, new, mode, info if self.log_variants else None)
            item["image"] = torch.from_numpy(np.ascontiguousarray(out)).float()
            item["mask"] = torch.from_numpy(np.ascontiguousarray(new)).long()
        item["synth_mode"] = mode
        if self.return_tear:
            add_tear_targets(item, info)
        return item


def add_tear_targets(item, info):
    """v18 targets: tear map (H, W) and fine pattern class (index into PATTERN_CLASSES)."""
    hw = tuple(item["mask"].shape[-2:])
    if info is None or item["synth_mode"] == "none":
        item["tear"] = torch.zeros(hw, dtype=torch.bool)
        item["pattern_id"] = 0
    else:
        item["tear"] = torch.from_numpy(np.ascontiguousarray(info["tear"]).astype(bool))
        v = info["variant"]
        item["pattern_id"] = PATTERN_CLASSES.index(v) if v in PATTERN_CLASSES else 0
