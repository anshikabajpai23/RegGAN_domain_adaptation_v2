"""
build_kfold_real_pd.py
=======================
Phase 4 step 6: k-fold over the 15 real patients.

Pools real_pd_seg_data_v7's existing train/ + val/ splits back into one set
of 15 patients, then re-splits into 5 folds of 3 val patients each (12
train). Fake-PD stream is UNCHANGED across all folds — this script only
touches the real-PD side. real_pd_seg_data_v7/ itself is never modified,
only read from — new directories per fold.

Fold assignment is fixed and deterministic (sorted patient IDs, chunked in
order into 5 groups of 3) so it's reproducible without needing its own seed:

  fold 0: AC0407F05FAF53, AC045F8F6ACBA7, AC0925B79D4836
  fold 1: AC0BA0AE159EF9, AC0CE315D5758B, AC0CEE9C24F2B7
  fold 2: AC1638B3D9A430, AC1B16C5EBA405, AC2089D56058F0
  fold 3: AC242B1DC40419, AC25B946B88F6A, AC29C3C6B4F385
  fold 4: AC2A47E3CF3E07, AC2B0AA9AE767D, AC2E254F52E467

Each patient appears as val exactly once across the 5 folds (verified by
this script's own consistency check, printed at the end).

Usage:
  python scripts/build_kfold_real_pd.py \
      --real_data_root real_pd_seg_data_v7 \
      --out_prefix     real_pd_seg_data_v7_fold
"""
import argparse
import glob
import os
import re
import shutil


PATIENTS = sorted([
    "AC0407F05FAF53", "AC045F8F6ACBA7", "AC0925B79D4836",
    "AC0BA0AE159EF9", "AC0CE315D5758B", "AC0CEE9C24F2B7",
    "AC1638B3D9A430", "AC1B16C5EBA405", "AC2089D56058F0",
    "AC242B1DC40419", "AC25B946B88F6A", "AC29C3C6B4F385",
    "AC2A47E3CF3E07", "AC2B0AA9AE767D", "AC2E254F52E467",
])

N_FOLDS = 5
FOLDS = [PATIENTS[i * 3:(i + 1) * 3] for i in range(N_FOLDS)]


def pid_from_stem(fname):
    """'{pid}_{sidx:04d}.npy' -> pid. Same convention as RealPDDataset."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    pid, _ = stem.rsplit("_", 1)
    return pid


def pool_files(real_data_root):
    """Every image/mask file across BOTH existing splits, keyed by patient."""
    by_pid = {}
    for split in ("train", "val"):
        for img_path in glob.glob(os.path.join(real_data_root, split, "images", "*.npy")):
            pid = pid_from_stem(img_path)
            mask_path = os.path.join(real_data_root, split, "masks", os.path.basename(img_path))
            if not os.path.exists(mask_path):
                print(f"  WARNING: no mask for {img_path}, skipping")
                continue
            by_pid.setdefault(pid, []).append((img_path, mask_path))
    return by_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_data_root", required=True)
    ap.add_argument("--out_prefix", required=True,
                    help="fold k written to <out_prefix>{k}/")
    args = ap.parse_args()

    pooled = pool_files(args.real_data_root)
    found_pids = set(pooled.keys())
    expected_pids = set(PATIENTS)

    print(f"Pooled {sum(len(v) for v in pooled.values())} slice pairs across "
         f"{len(found_pids)} patients")
    missing = expected_pids - found_pids
    extra = found_pids - expected_pids
    if missing:
        print(f"  WARNING: expected patients not found in pool: {sorted(missing)}")
    if extra:
        print(f"  WARNING: unexpected patients found (not in the known-15 list): {sorted(extra)}")

    val_membership = {}   # pid -> which fold it's val in
    for k, fold_pids in enumerate(FOLDS):
        out_dir = f"{args.out_prefix}{k}"
        for split in ("train", "val"):
            os.makedirs(os.path.join(out_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, split, "masks"), exist_ok=True)

        n_train_slices = n_val_slices = 0
        for pid, pairs in pooled.items():
            split = "val" if pid in fold_pids else "train"
            for img_path, mask_path in pairs:
                shutil.copy2(img_path, os.path.join(out_dir, split, "images", os.path.basename(img_path)))
                shutil.copy2(mask_path, os.path.join(out_dir, split, "masks", os.path.basename(mask_path)))
            if split == "val":
                n_val_slices += len(pairs)
                val_membership.setdefault(pid, []).append(k)
            else:
                n_train_slices += len(pairs)

        print(f"fold {k}: val={fold_pids}  "
             f"train_slices={n_train_slices}  val_slices={n_val_slices}  -> {out_dir}")

    print("\n=== consistency check ===")
    ok = True
    for pid in PATIENTS:
        folds_as_val = val_membership.get(pid, [])
        status = "OK" if len(folds_as_val) == 1 else "PROBLEM"
        if status != "OK":
            ok = False
        print(f"  {pid}: val in fold(s) {folds_as_val}  [{status}]")
    print(f"\nAll 15 patients val exactly once: {ok}")


if __name__ == "__main__":
    main()
