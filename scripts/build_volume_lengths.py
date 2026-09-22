"""
build_volume_lengths.py
========================
Phase 5, E1 prep — records the TRUE number of slices per volume, per cohort.

WHY THIS IS NEEDED (and why the obvious shortcut is wrong)
E1's position feature is
    z_frac = slice_idx / (n_slices - 1);  pos = 2*|z_frac - 0.5|
which requires the full volume length.

For fake PD you CANNOT get that by counting stored files. segmentation_data_v2
was built by prepare_meniscus_masks.py with min_meniscus_pixels=10, which SKIPS
meniscus-free slices -- about 76 of ~160 slices per patient survive. Counting
files would make the meniscus span cover ~0..1 of the range, while in real PD
(unfiltered) it covers only the middle band. The feature would then mean
something DIFFERENT in each domain, which is precisely what doc §1.2's symmetric
formulation exists to avoid.

What the filenames DO give us correctly is slice_idx itself: prepare_meniscus_masks.py
saves `{pid}_slice_{idx:03d}.npy` using the original volume index, skipping
rather than renumbering. So only n_slices has to be recovered here.

Source of truth, matching prepare_meniscus_masks.py exactly:
    n = min(fake_vol.shape[0], mask_vol.shape[0])
Read from NIfTI HEADERS only (nibabel is lazy; .shape touches no pixel data),
so this is fast even across 155 volumes.

Real PD is not filtered, so max(stored slice index) + 1 is its true length; it is
validated against the expected ~36 rather than trusted blindly.

Usage:
  python scripts/build_volume_lengths.py \
      --fake_pd_dir  results/fake_pd_all155 \
      --fake_mask_dir preprocessed_v3/masks \
      --real_root    real_pd_seg_data_v7 \
      --out_json     volume_lengths.json
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import nibabel as nib

EXPECT_FAKE = 160     # DESS, ~0.8 mm slices
EXPECT_REAL = 36      # real PD, ~3.6 mm slices


def extract_patient_id(filename):
    """Verbatim convention from prepare_meniscus_masks.py -- must match, or the
    lengths key onto the wrong patients."""
    base = os.path.basename(filename)
    m = re.match(r"(MTR_\d+)", base)
    if m:
        return m.group(1)
    for suf in ["_sl", "_pd_translated", "_mask"]:
        if suf in base:
            return base.split(suf)[0]
    return base


def fake_lengths(fake_pd_dir, fake_mask_dir):
    """n = min(fake.shape[0], mask.shape[0]) -- exactly what prep used."""
    fake_by_pid, mask_by_pid = {}, {}
    for f in sorted(glob.glob(os.path.join(fake_pd_dir, "*.nii.gz"))):
        fake_by_pid[extract_patient_id(f)] = f
    for f in sorted(glob.glob(os.path.join(fake_mask_dir, "*.nii.gz"))):
        mask_by_pid[extract_patient_id(f)] = f

    out, mismatches = {}, []
    for pid in sorted(set(fake_by_pid) & set(mask_by_pid)):
        n_fake = nib.load(fake_by_pid[pid]).shape[0]      # header only
        n_mask = nib.load(mask_by_pid[pid]).shape[0]
        if n_fake != n_mask:
            mismatches.append((pid, n_fake, n_mask))
        out[pid] = int(min(n_fake, n_mask))
    return out, mismatches, len(fake_by_pid), len(mask_by_pid)


def real_lengths(real_root, splits=("train", "val")):
    """Real PD keeps every slice, so max index + 1 is the true length.
    Validated against EXPECT_REAL rather than assumed.

    MUST cover train AND val. The trainer builds a real_val dataset too, and a
    patient missing from this table makes its position undefined -- the E1/E2
    datasets (correctly) refuse to run rather than divide by the wrong n. Scanning
    train only is what broke the first E1 submission.
    Fake PD needs no equivalent fix: its lengths come from globbing all 155 source
    volumes, which is already a superset of every split."""
    per_pid = defaultdict(list)
    found = {}
    for split in splits:
        n_before = len(per_pid)
        for f in glob.glob(os.path.join(real_root, split, "images", "*.npy")):
            stem = os.path.splitext(os.path.basename(f))[0]
            pid, sidx = stem.rsplit("_", 1)
            per_pid[pid].append(int(sidx))
        found[split] = len(per_pid) - n_before
    print(f"  real-PD patients per split: " +
          ", ".join(f"{s}={n}" for s, n in found.items()))
    return {pid: int(max(v) + 1) for pid, v in per_pid.items()}


def report(name, lengths, expected):
    if not lengths:
        print(f"\n{name}: NOTHING FOUND -- check the paths.")
        return False
    vals = np.array(sorted(lengths.values()))
    print(f"\n{name}: {len(lengths)} volumes")
    print(f"  slices/volume: min {vals.min()}  median {int(np.median(vals))}  max {vals.max()}")
    off = [(p, n) for p, n in sorted(lengths.items()) if abs(n - expected) > 0.5 * expected]
    if off:
        print(f"  WARNING: {len(off)} volume(s) more than 50% away from the expected ~{expected}:")
        for p, n in off[:10]:
            print(f"    {p}: {n}")
        print("  If most volumes look wrong, the source dir is probably not the")
        print("  unfiltered one and the position feature would be meaningless.")
        return False
    print(f"  all within 50% of the expected ~{expected} -- looks like the unfiltered source")
    return True


def pos_of(idx, n):
    """doc §1.2: 0 at the volume centre (notch), 1 at the outer edges."""
    return 2.0 * abs(idx / max(n - 1, 1) - 0.5)


def span_pos_coverage(fake_prepped, real_prepped, fake_len, real_len, split="train"):
    """doc §1.2 ⚠: 'check that the meniscus spans occupy roughly the same pos
    range in both cohorts'. If they do not, a given pos value means a different
    anatomical thing in each domain and E1's feature is not domain-shared.

    fake: every STORED slice contains meniscus (prep filtered), so filenames suffice.
    real: unfiltered, so masks are loaded to find the nonempty ones."""
    out = {}

    fake_pos = []
    for f in glob.glob(os.path.join(fake_prepped, split, "masks", "*", "*.npy")):
        pid = os.path.basename(os.path.dirname(f))
        m = re.search(r"_slice_(\d{3})\.npy$", f)
        if m and pid in fake_len:
            fake_pos.append(pos_of(int(m.group(1)), fake_len[pid]))
    out["fake"] = np.array(fake_pos)

    real_pos = []
    for f in glob.glob(os.path.join(real_prepped, split, "masks", "*.npy")):
        stem = os.path.splitext(os.path.basename(f))[0]
        pid, sidx = stem.rsplit("_", 1)
        if pid in real_len and np.load(f).any():
            real_pos.append(pos_of(int(sidx), real_len[pid]))
    out["real"] = np.array(real_pos)
    return out


def report_span_coverage(cov):
    print(f"\n{'=' * 72}")
    print("POSITION COVERAGE OF MENISCUS SLICES (doc §1.2 check)")
    print(f"{'=' * 72}")
    stats = {}
    for name in ("fake", "real"):
        a = cov[name]
        if a.size == 0:
            print(f"  {name}: no slices found -- cannot check")
            return False
        stats[name] = (np.percentile(a, 5), np.percentile(a, 95), np.median(a))
        print(f"  {name:<5}: n={a.size:<7} pos p5={stats[name][0]:.3f}  "
              f"median={stats[name][2]:.3f}  p95={stats[name][1]:.3f}")

    # do the two 5-95% bands overlap substantially?
    lo = max(stats["fake"][0], stats["real"][0])
    hi = min(stats["fake"][1], stats["real"][1])
    width_f = stats["fake"][1] - stats["fake"][0]
    width_r = stats["real"][1] - stats["real"][0]
    overlap = max(0.0, hi - lo) / max(min(width_f, width_r), 1e-6)
    print(f"\n  5-95% band overlap: {overlap:.2f} of the narrower band")
    if overlap < 0.6:
        print("  WARNING: the cohorts' meniscus spans sit at DIFFERENT pos ranges.")
        print("  A given pos value would mean a different anatomical location in each")
        print("  domain, so the position channel is not a shared feature. E1's result")
        print("  would be hard to interpret -- consider raw z_frac with known laterality,")
        print("  or restricting E1 to the real branch, before spending 10 h of GPU.")
        return False
    print("  Bands overlap substantially -- pos means roughly the same thing in both.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake_pd_dir", required=True, help="flat dir of translated *.nii.gz volumes")
    ap.add_argument("--fake_mask_dir", required=True, help="flat dir of DESS mask *.nii.gz volumes")
    ap.add_argument("--real_root", required=True)
    ap.add_argument("--real_splits", nargs="+", default=["train", "val"],
                    help="MUST include every split the trainer will build a dataset for; "
                         "train-only leaves the val patients undefined")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--fake_prepped_root", default=None,
                    help="prepped fake root (segmentation_data_v2) for the §1.2 coverage check")
    ap.add_argument("--real_prepped_root", default=None,
                    help="prepped real root (real_pd_seg_data_v7) for the §1.2 coverage check")
    args = ap.parse_args()

    fake, mismatches, n_fake_files, n_mask_files = fake_lengths(args.fake_pd_dir, args.fake_mask_dir)
    print(f"fake-PD volumes found: {n_fake_files}   masks found: {n_mask_files}   paired: {len(fake)}")
    if mismatches:
        print(f"  NOTE: {len(mismatches)} patient(s) where fake and mask lengths differ; "
              f"min() used, same as prep. First few:")
        for pid, a, b in mismatches[:5]:
            print(f"    {pid}: fake {a} vs mask {b} -> {min(a, b)}")

    real = real_lengths(args.real_root, args.real_splits)

    ok_fake = report("FAKE PD", fake, EXPECT_FAKE)
    ok_real = report("REAL PD", real, EXPECT_REAL)

    ok_cov = None
    if args.fake_prepped_root and args.real_prepped_root:
        cov = span_pos_coverage(args.fake_prepped_root, args.real_prepped_root,
                                fake, real, "train")
        ok_cov = report_span_coverage(cov)
    else:
        print("\n(skipping the §1.2 position-coverage check -- pass "
              "--fake_prepped_root and --real_prepped_root to run it)")

    out = {
        "meta": {
            "fake_pd_dir": args.fake_pd_dir, "fake_mask_dir": args.fake_mask_dir,
            "real_root": args.real_root, "real_splits": args.real_splits,
            "n_fake": len(fake), "n_real": len(real),
            "fake_validated": ok_fake, "real_validated": ok_real,
            "span_coverage_ok": ok_cov,
        },
        "fake": fake,
        "real": real,
    }
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.out_json}")

    if not (ok_fake and ok_real):
        print("\nVALIDATION FAILED -- do not train E1 on this table. The position")
        print("feature would not mean the same thing across the two cohorts.")
        raise SystemExit(1)
    if ok_cov is False:
        print("\nLENGTHS OK, BUT THE §1.2 COVERAGE CHECK FAILED. The table is usable,")
        print("but read the warning above before committing 10 h of GPU to E1.")
        raise SystemExit(2)
    print("\nValidated. Safe to train E1 against this table.")


if __name__ == "__main__":
    main()
