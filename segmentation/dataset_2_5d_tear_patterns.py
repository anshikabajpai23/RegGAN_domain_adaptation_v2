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

_QUOTA_FULL = set()     # per process: (log_tag, source, mode) whose save quota is already reached


class TearPatternDataset(Dataset):
    def __init__(self, base, synth=False, p=0.5, modes=MODES, mode_p=MODE_P, fixed_seed=None,
                 source="", log_dir=None, save_per_mode=0):
        """
        base: a Meniscus2_5DDataset or RealPDDataset.
        synth: apply patterns (training sets, or the fixed-seed per-mode diagnostic sets).
        p: probability that a meniscus slice gets a pattern.
        modes / mode_p: which patterns and how often (default: all, see tear_patterns.MODE_P).
        fixed_seed: if set, item i always gets the same pattern (deterministic diagnostic set).
        source: "fake" or "real", recorded in the logs.
        log_dir / save_per_mode: logging, active only while self.log_tag is an int (set per epoch).
        """
        self.base, self.synth, self.p = base, synth, p
        self.modes, self.mode_p, self.fixed_seed = tuple(modes), tuple(mode_p), fixed_seed
        self.source, self.log_dir, self.save_per_mode, self.log_tag = source, log_dir, save_per_mode, None

    def __len__(self):
        return len(self.base)

    # pass-through so the replica/v14 eval helpers (which read .slices / .items) still work
    def __getattr__(self, name):
        if name in ("base", "synth", "p", "modes", "mode_p", "fixed_seed",
                    "source", "log_dir", "save_per_mode", "log_tag"):
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

    def _log(self, i, img0, msk0, img1, msk1, mode):
        tag = int(self.log_tag)
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi is not None else 0
        d = os.path.join(self.log_dir, f"epoch{tag:03d}")
        pid, sidx = self._pid_slice(i)
        m0, m1 = msk0 > 0, msk1 > 0
        diff = np.abs(img1[1] - img0[1]) > 0.02
        key = (tag, self.source, mode)
        save = False
        if self.save_per_mode > 0 and key not in _QUOTA_FULL:
            n_on_disk = len(glob.glob(os.path.join(d, "samples", mode, f"{self.source}_*.npz")))
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
            "saved_file": f"samples/{mode}/{self.source}_{pid}_{sidx:04d}_i{int(i)}.npz" if save else "",
        }
        if save:
            os.makedirs(os.path.join(d, "samples", mode), exist_ok=True)
            np.savez_compressed(os.path.join(d, row["saved_file"]),
                                image_before=img0.astype(np.float16), image_after=img1.astype(np.float16),
                                mask_before=msk0.astype(np.int8), mask_after=msk1.astype(np.int8),
                                mode=mode, source=self.source, pid=pid, slice=sidx)
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
        if self.synth:
            image = item["image"].numpy().astype(np.float32)
            mask = item["mask"].numpy()
            out, new, mode = synth_pattern(image, mask, self._rng(i), p=self.p,
                                           modes=self.modes, mode_p=self.mode_p)
            if self.log_dir and self.log_tag is not None and mode != "none":
                self._log(i, image, mask, out, new, mode)
            item["image"] = torch.from_numpy(np.ascontiguousarray(out)).float()
            item["mask"] = torch.from_numpy(np.ascontiguousarray(new)).long()
        item["synth_mode"] = mode
        return item
