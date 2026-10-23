"""Generate a synthetic-tear log LOCALLY (no training, no GPU) with the exact generator a run uses,
so distributions and every variant can be inspected before / without the BigRed run.

Runs the training wrappers (TearPatternDataset / SequenceTearPatternDataset) and their logging code
unchanged on labelled real PD scans, writing the same folder layout as a training run:
  <out>/epoch000/stats_all.csv, summary_per_mode.csv, summary_per_variant.csv, summary_histograms.png,
  samples/<variant>/*.npz, nifti/<variant>_real_{image_before,image_after,label_before,label_after,tear}.nii.gz
Differences from training: real-PD scans only (no fake PD), no flip / brightness / noise augmentation,
`--passes` draws per slice (training draws once per slice per epoch).

  --run v15   v15 generator (3 slices, original in-place mix)
  --run v17   v17 generator (3 slices, + Fig. 10a complex_hv)
  --run v16   v16 / v18 generator (5 slices, crop space, native-aspect geometry, v17 mix)

    venv/bin/python scripts/simulate_synthetic_log.py --run v17 --out results/sim_synthetic_log/v17
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "segmentation"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from dataset_2_5d_tear_patterns import TearPatternDataset             # noqa: E402
from dataset_sequence_tear_patterns import SequenceTearPatternDataset  # noqa: E402
from tear_synth import PATTERNS_V17, PATTERN_P_V17                    # noqa: E402
from synthetic_log_utils import process_epoch                         # noqa: E402
from torn_compare_utils import process_volume, process_manual_seg     # noqa: E402

SEG_DIR = "/Users/anshikabajpai/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final"
TORN = ("AC13D01F99D88C", "AC116435D25299", "AC1CEC4C14F850", "AC152F0CF07CF3", "AC1A200F61EDC1")


class VolumeSlices(torch.utils.data.Dataset):
    """Labelled slices of whole volumes, in the layout the wrappers expect (.slices = '<pid>_<ssss>')."""
    def __init__(self, vols, segs, offsets, box=None):
        self.vols, self.segs, self.offsets, self.box = vols, segs, offsets, box
        self.slices = [f"{pid}_{s:04d}" for pid, seg in segs.items() for s in range(seg.shape[0]) if (seg[s] > 0).sum() >= 20]

    def _crop(self, a, order):
        if self.box is None:
            return a
        from scipy.ndimage import zoom
        y0, y1, x0, x1 = self.box
        c = a[y0:y1, x0:x1]
        return zoom(c, (384 / c.shape[0], 384 / c.shape[1]), order=order)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        pid, s = self.slices[i].rsplit("_", 1)
        s, v = int(s), self.vols[pid]
        n = v.shape[0]
        img = np.stack([np.clip(self._crop(v[max(0, min(n - 1, s + o))], 3), 0, 1) for o in self.offsets]).astype(np.float32)
        msk = self._crop(self.segs[pid][s].astype(np.float32), 0).astype(np.int64)
        return {"image": torch.from_numpy(img), "mask": torch.from_numpy(msk)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["v15", "v17", "v16"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg_dir", default=SEG_DIR)
    ap.add_argument("--passes", type=int, default=6, help="independent draws per slice")
    ap.add_argument("--save_per_variant", type=int, default=60)
    ap.add_argument("--pattern_p", type=float, default=0.5)
    a = ap.parse_args()
    vols, segs = {}, {}
    for seg_f in sorted(glob.glob(os.path.join(a.seg_dir, "*.seg.nrrd"))):
        pid = os.path.basename(seg_f).split("_")[0]
        img_f = glob.glob(os.path.join(a.seg_dir, f"{pid}*.nii.gz"))
        if pid in TORN or not img_f:
            continue
        vols[pid] = process_volume(img_f[0], "PD")
        segs[pid] = process_manual_seg(seg_f).astype(np.int64)
    print(f"{len(vols)} labelled healthy scans")
    v17 = dict(inplace_variants=PATTERNS_V17, inplace_p=PATTERN_P_V17)
    if a.run == "v16":
        import json
        b = json.load(open(os.path.join(ROOT, "crop_box.json")))
        base = VolumeSlices(vols, segs, (-2, -1, 0, 1, 2), (b["y0"], b["y1"], b["x0"], b["x1"]))
        mk = lambda seed: SequenceTearPatternDataset(base, native_hw=(b["y1"] - b["y0"], b["x1"] - b["x0"]), synth=True,
                                                     p=a.pattern_p, fixed_seed=seed, source="real", log_dir=a.out,
                                                     save_per_mode=a.save_per_variant, log_variants=True, **v17)
    else:
        base = VolumeSlices(vols, segs, (-1, 0, 1))
        extra = v17 if a.run == "v17" else {}
        mk = lambda seed: TearPatternDataset(base, synth=True, p=a.pattern_p, fixed_seed=seed, source="real", log_dir=a.out,
                                             save_per_mode=a.save_per_variant, log_variants=True, **extra)
    print(f"{len(base)} meniscus slices x {a.passes} passes")
    for k in range(a.passes):
        ds = mk(1000 + k)
        ds.log_tag = 0
        for i in range(len(ds)):
            ds[i]
    process_epoch(os.path.join(a.out, "epoch000"))


if __name__ == "__main__":
    main()
