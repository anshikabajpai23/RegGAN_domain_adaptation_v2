"""View fastMRI+ knee bounding-box annotations (meniscus by default) on fastMRI h5 volumes.

Annotations: data_external/fastmri_plus/fastMRI_csv/singlecoil_bb_{train,val}.csv
Image data:  fastMRI knee singlecoil_{train,val} *.h5 (needs NYU registration).
Coords in the CSV are [x0, x1, y0, y1] and `slice` is 1-indexed (per the repo's main.py).

    python scripts/view_fastmri_plus.py --list --split val
    python scripts/view_fastmri_plus.py --h5_root /path/to/fastMRI --split val --out_dir figures/fastmri_plus
"""
import argparse, ast, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CSV_DIR = "data_external/fastmri_plus/fastMRI_csv"
COLORS = {"Med_Men": "lime", "Lat_Men": "red"}


def load_csv(split, keep="Men"):
    df = pd.read_csv(os.path.join(CSV_DIR, f"singlecoil_bb_{split}.csv"))
    df["labels"] = df["labels"].apply(lambda s: ast.literal_eval(s.replace("' '", "', '")))
    df["coords"] = df["coords"].apply(
        lambda s: np.array(re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", s), dtype=float).reshape(-1, 4))
    if keep:
        df = df[df["labels"].apply(lambda l: any(keep in x for x in l))]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--h5_root", help="dir with singlecoil_{split}/ (or the h5 files directly)")
    ap.add_argument("--out_dir", default="figures/fastmri_plus")
    ap.add_argument("--all_labels", action="store_true", help="include non-meniscus boxes")
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    df = load_csv(a.split, None if a.all_labels else "Men")
    print(f"{len(df)} annotated slices in {df.filename.nunique()} exams")
    print(pd.Series([l for ls in df.labels for l in ls]).value_counts())
    if a.list or not a.h5_root:
        print(df.head(20)); return

    import h5py
    os.makedirs(a.out_dir, exist_ok=True)
    sub = os.path.join(a.h5_root, f"singlecoil_{a.split}")
    root = sub if os.path.isdir(sub) else a.h5_root
    n = 0
    for _, r in df.iterrows():
        f = os.path.join(root, r.filename + ".h5")
        if not os.path.exists(f):
            continue
        with h5py.File(f, "r") as h:
            img = h["reconstruction_esc"][r.slice - 1]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap="gray"); ax.axis("off")
        for lab, (x0, x1, y0, y1) in zip(r.labels, r.coords):
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2,
                                           ec=COLORS.get(lab, "yellow")))
            ax.text(x0, y0 - 3, lab, color=COLORS.get(lab, "yellow"), fontsize=9)
        ax.set_title(f"{r.filename} slice {r.slice}")
        fig.savefig(os.path.join(a.out_dir, f"{r.filename}_{r.slice}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig); n += 1
        if n >= a.max: break
    print(f"saved {n} figures to {a.out_dir}")


if __name__ == "__main__":
    main()
