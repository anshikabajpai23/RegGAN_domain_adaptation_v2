"""
dataset_sequence_tear_patterns.py
=================================
v16: wraps the crop+sequence datasets (SequenceMeniscusDataset / SequenceRealPDDataset, 5-slice
input) without modifying them, and applies one v15 tear pattern per sample through
tear_patterns_seq.synth_pattern_seq (middle 3 slices, native-aspect geometry).

Everything else is inherited from v15's TearPatternDataset: probability p, mode mix, fixed-seed
diagnostic sets, and the synthetic-data log. The log stores the middle 3 slices, so the saved
samples have exactly v15's format and the v15 review scripts (show_synthetic_samples.py,
synthetic_log_utils.py, umap_synthetic_vs_torn.py) work on them unchanged.
"""
import numpy as np
import torch

from dataset_2_5d_tear_patterns import TearPatternDataset, add_tear_targets
from tear_patterns_seq import synth_pattern_seq


class SequenceTearPatternDataset(TearPatternDataset):
    def __init__(self, base, native_hw=None, **kw):
        """base: a SequenceMeniscusDataset or SequenceRealPDDataset.
        native_hw: crop box size (h, w) before the resize to 384 x 384 (None = no correction)."""
        super().__init__(base, **kw)
        self.native_hw = native_hw

    def __getattr__(self, name):
        if name == "native_hw":
            raise AttributeError(name)
        return super().__getattr__(name)

    def __getitem__(self, i):
        item = self.base[i]
        mode, info = "none", None
        if self.synth:
            image = item["image"].numpy().astype(np.float32)
            mask = item["mask"].numpy()
            out, new, mode, sub0, sub1, info = synth_pattern_seq(
                image, mask, self._rng(i), p=self.p, modes=self.modes, mode_p=self.mode_p,
                native_hw=self.native_hw, inplace_variants=self.inplace_variants, inplace_p=self.inplace_p)
            if self.log_dir and self.log_tag is not None and mode != "none":
                self._log(i, sub0, mask, sub1, new, mode, info if self.log_variants else None)
            item["image"] = torch.from_numpy(np.ascontiguousarray(out)).float()
            item["mask"] = torch.from_numpy(np.ascontiguousarray(new)).long()
        item["synth_mode"] = mode
        if self.return_tear:
            add_tear_targets(item, info)
        return item
