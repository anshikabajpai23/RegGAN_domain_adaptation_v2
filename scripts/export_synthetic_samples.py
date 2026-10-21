"""Save a fixed, reproducible sample of v15 synthetic tears for later review.

Training generates tears on the fly (a new random tear every epoch), so nothing is kept. This
script draws N examples per pattern from the SAME training data with the SAME code
(segmentation/tear_patterns.py) and a fixed seed, and saves them so you can check later that
each pattern matches its description, show them to a radiologist, or run realism tests.

Output (in --out_dir):
  samples/<mode>/<source>_<pid>_<slice>.npz   image_before, image_after (3,384,384 float16),
                                              mask_before, mask_after (int8), mode, source, pid, slice
  sheets/<mode>.png                           contact sheet: before | after | label before | label after
  index.csv                                   one row per sample, plus pixel counts and piece counts

Usage (run where the training data is, e.g. BigRed):
  python scripts/export_synthetic_samples.py \
      --fake_data_root /N/.../segmentation_data_v2 --real_data_root /N/.../real_pd_seg_data_v7 \
      --out_dir /N/.../results/v15_synthetic_samples --per_mode 10
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "segmentation"))
from dataset_2_5d_v2 import Meniscus2_5DDataset        # noqa: E402
from dataset_2_5d_realpd import RealPDDataset          # noqa: E402
from tear_patterns import synth_pattern, MODES         # noqa: E402


def pid_slice(ds, i, source):
    if source == "fake":
        pid, idx = ds.items[i]
        return pid, int(idx)
    stem = ds.slices[i]
    pid, s = stem.rsplit("_", 1)
    return pid, int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_data_root", required=True)
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_mode", type=int, default=10, help="examples per pattern per source (fake, real)")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    sources = {
        "fake": Meniscus2_5DDataset(os.path.join(a.fake_data_root, "train", "images"),
                                    os.path.join(a.fake_data_root, "train", "masks"), augment=False),
        "real": RealPDDataset(os.path.join(a.real_data_root, "train", "images"),
                              os.path.join(a.real_data_root, "train", "masks"), augment=False),
    }
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out_dir, "sheets"), exist_ok=True)
    rows = []
    for mode in a.modes:
        os.makedirs(os.path.join(a.out_dir, "samples", mode), exist_ok=True)
        sheet = []
        for source, ds in sources.items():
            order = rng.permutation(len(ds))
            got = 0
            for i in order:
                if got >= a.per_mode:
                    break
                item = ds[int(i)]
                img = item["image"].numpy().astype(np.float32)
                msk = item["mask"].numpy().astype(np.int64)
                if (msk > 0).sum() < 50:
                    continue
                out, new, m = synth_pattern(img, msk, rng, p=1.0, modes=(mode,), mode_p=(1.0,))
                if m == "none":
                    continue
                pid, s = pid_slice(ds, int(i), source)
                name = f"{source}_{pid}_{s:04d}"
                np.savez_compressed(os.path.join(a.out_dir, "samples", mode, name + ".npz"),
                                    image_before=img.astype(np.float16), image_after=out.astype(np.float16),
                                    mask_before=msk.astype(np.int8), mask_after=new.astype(np.int8),
                                    mode=mode, source=source, pid=pid, slice=s)
                rows.append({"mode": mode, "source": source, "pid": pid, "slice": s, "file": f"samples/{mode}/{name}.npz",
                             "meniscus_px_before": int((msk > 0).sum()), "meniscus_px_after": int((new > 0).sum()),
                             "pieces_before": int(ndi.label(msk > 0)[1]), "pieces_after": int(ndi.label(new > 0)[1]),
                             "image_changed_px": int((np.abs(out[1] - img[1]) > 0.02).sum())})
                sheet.append((img, out, msk, new, name))
                got += 1
            if got < a.per_mode:
                print(f"WARNING: {mode}/{source}: only {got} usable examples")
        # contact sheet for this mode
        if sheet:
            fig, axes = plt.subplots(len(sheet), 4, figsize=(9, 2.4 * len(sheet)), squeeze=False)
            for r, (img, out, msk, new, name) in enumerate(sheet):
                ys, xs = np.where((msk > 0) | (new > 0)); p = 20
                cs = (slice(max(ys.min() - p, 0), ys.max() + p), slice(max(xs.min() - p, 0), xs.max() + p))
                for c, (im, t) in enumerate([(img[1], f"{name}: before"), (out[1], "after (model sees)"),
                                             (img[1], "label before"), (out[1], "label after")]):
                    axes[r, c].imshow(im[cs], cmap="gray", vmin=0, vmax=1); axes[r, c].axis("off")
                    axes[r, c].set_title(t, fontsize=7)
                axes[r, 2].contour((msk > 0)[cs].astype(float), [0.5], colors=["#ffe600"], linewidths=1)
                axes[r, 3].contour((new > 0)[cs].astype(float), [0.5], colors=["#00e5ff"], linewidths=1)
            fig.suptitle(f"v15 synthetic pattern: {mode}", fontsize=10)
            fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "sheets", f"{mode}.png"), dpi=100); plt.close(fig)
        print(f"{mode}: {sum(r['mode'] == mode for r in rows)} samples saved")
    with open(os.path.join(a.out_dir, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"done: {len(rows)} samples -> {a.out_dir}")


if __name__ == "__main__":
    main()
