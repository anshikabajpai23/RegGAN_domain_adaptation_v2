"""
fid_statistical_test.py
=========================
Statistical significance for the FID improvement claim (fake-PD vs real-PD
scoring lower than DESS vs real-PD). Answers: is this difference real, or
within noise -- addresses the "statistical constraint" feedback directly.

Two tests, both reusing evaluate.py's existing compute_simple_features()/
frechet_distance() (not duplicated):

1. Bootstrap confidence intervals: resample (with replacement) from each
   slice set many times, recompute FID each time, get a distribution ->
   95% CI for FID_fake_vs_real and FID_dess_vs_real independently.

2. Permutation test on the IMPROVEMENT (FID_dess_vs_real - FID_fake_vs_real):
   pool all fake+dess+real slices together, repeatedly reassign into
   randomly-shuffled groups of the original sizes, recompute the same
   "improvement" statistic under this null (no real difference) hypothesis,
   and report the p-value: fraction of random reshuffles that produced an
   improvement >= the actually observed one.

Usage:
    python inference/fid_statistical_test.py \
        --dess_slice_dir preprocessed_v2/slices/dess \
        --fake_pd_dir    results/stage4_fake_pd \
        --real_pd_dir    data/iu-dataset/pd-files \
        --out_json       runs/run_004/fid_statistical_test.json \
        --n_bootstrap    500 \
        --n_permutations 500
"""
import argparse
import glob
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import (compute_simple_features, frechet_distance,
                       slices_to_array, load_slices_from_nifti)
from preprocess import process_volume

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def fid_from_features(feats_a, feats_b):
    mu_a, s_a = feats_a.mean(0), np.cov(feats_a, rowvar=False)
    mu_b, s_b = feats_b.mean(0), np.cov(feats_b, rowvar=False)
    return frechet_distance(mu_a, s_a, mu_b, s_b)


def bootstrap_fid_ci(feats_a, feats_b, n_bootstrap=500, sample_size=None, seed=42):
    rng = np.random.RandomState(seed)
    n_a, n_b = len(feats_a), len(feats_b)
    sample_size_a = sample_size or n_a
    sample_size_b = sample_size or n_b

    fids = []
    for _ in range(n_bootstrap):
        idx_a = rng.randint(0, n_a, size=sample_size_a)
        idx_b = rng.randint(0, n_b, size=sample_size_b)
        fids.append(fid_from_features(feats_a[idx_a], feats_b[idx_b]))
    fids = np.array(fids)
    return {
        "mean": float(fids.mean()),
        "std": float(fids.std()),
        "ci_2.5": float(np.percentile(fids, 2.5)),
        "ci_97.5": float(np.percentile(fids, 97.5)),
    }


def permutation_test_improvement(dess_feats, fake_feats, real_feats, n_permutations=500, seed=42):
    """
    Null hypothesis: there's no real difference between DESS's and fake-PD's
    similarity to real PD -- any apparent "improvement" is due to random
    sampling. Pool dess_feats + fake_feats, randomly resplit into two groups
    of the original sizes, recompute the improvement statistic each time.
    """
    rng = np.random.RandomState(seed)

    observed_fid_fake = fid_from_features(fake_feats, real_feats)
    observed_fid_dess = fid_from_features(dess_feats, real_feats)
    observed_improvement = observed_fid_dess - observed_fid_fake

    pooled = np.concatenate([dess_feats, fake_feats], axis=0)
    n_dess, n_fake = len(dess_feats), len(fake_feats)
    n_total = n_dess + n_fake

    null_improvements = []
    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        group_dess = pooled[perm[:n_dess]]
        group_fake = pooled[perm[n_dess:]]
        fid_d = fid_from_features(group_dess, real_feats)
        fid_f = fid_from_features(group_fake, real_feats)
        null_improvements.append(fid_d - fid_f)

    null_improvements = np.array(null_improvements)
    p_value = float((null_improvements >= observed_improvement).mean())

    return {
        "observed_fid_fake_vs_real": float(observed_fid_fake),
        "observed_fid_dess_vs_real": float(observed_fid_dess),
        "observed_improvement": float(observed_improvement),
        "null_improvement_mean": float(null_improvements.mean()),
        "null_improvement_std": float(null_improvements.std()),
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dess_slice_dir", required=True)
    ap.add_argument("--fake_pd_dir", required=True)
    ap.add_argument("--real_pd_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_slices", type=int, default=500)
    ap.add_argument("--n_bootstrap", type=int, default=500)
    ap.add_argument("--n_permutations", type=int, default=500)
    args = ap.parse_args()

    log.info("Loading DESS slices...")
    dess_npys = sorted(glob.glob(os.path.join(args.dess_slice_dir, "*.npy")))[:args.max_slices]
    dess_slices = [np.load(f) for f in dess_npys]

    log.info("Loading fake PD slices...")
    fake_slices = []
    for f in sorted(glob.glob(os.path.join(args.fake_pd_dir, "*.nii.gz"))):
        fake_slices.extend(load_slices_from_nifti(f))
        if len(fake_slices) >= args.max_slices:
            break
    fake_slices = fake_slices[:args.max_slices]

    log.info("Loading real PD slices (preprocessed)...")
    real_slices = []
    for f in sorted(glob.glob(os.path.join(args.real_pd_dir, "**", "*.nii.gz"), recursive=True)):
        try:
            vol = process_volume(f, "PD")
            real_slices.extend([vol[i] for i in range(vol.shape[0])])
        except Exception as e:
            log.warning(f"  Skipping {f}: {e}")
        if len(real_slices) >= args.max_slices:
            break
    real_slices = real_slices[:args.max_slices]

    log.info(f"DESS={len(dess_slices)}  Fake PD={len(fake_slices)}  Real PD={len(real_slices)}")

    dess_feats = compute_simple_features(slices_to_array(dess_slices)).astype(np.float64)
    fake_feats = compute_simple_features(slices_to_array(fake_slices)).astype(np.float64)
    real_feats = compute_simple_features(slices_to_array(real_slices)).astype(np.float64)

    log.info(f"Running bootstrap CI ({args.n_bootstrap} resamples)...")
    ci_fake = bootstrap_fid_ci(fake_feats, real_feats, args.n_bootstrap)
    ci_dess = bootstrap_fid_ci(dess_feats, real_feats, args.n_bootstrap)
    log.info(f"  FID_fake_vs_real: {ci_fake['mean']:.2f} (95% CI: {ci_fake['ci_2.5']:.2f}-{ci_fake['ci_97.5']:.2f})")
    log.info(f"  FID_dess_vs_real: {ci_dess['mean']:.2f} (95% CI: {ci_dess['ci_2.5']:.2f}-{ci_dess['ci_97.5']:.2f})")

    log.info(f"Running permutation test ({args.n_permutations} permutations)...")
    perm_result = permutation_test_improvement(dess_feats, fake_feats, real_feats, args.n_permutations)
    log.info(f"  Observed improvement: {perm_result['observed_improvement']:.2f}")
    log.info(f"  p-value: {perm_result['p_value']:.4f}  "
             f"(significant at 0.05: {perm_result['significant_at_0.05']})")

    results = {
        "bootstrap_ci_fid_fake_vs_real": ci_fake,
        "bootstrap_ci_fid_dess_vs_real": ci_dess,
        "permutation_test": perm_result,
        "n_slices": {"dess": len(dess_slices), "fake_pd": len(fake_slices), "real_pd": len(real_slices)},
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
