"""
Stage 0 Pre-flight Check
========================
Prints a full picture of the local data state and a full code-logic audit:
  1. Directory inventory (what exists locally vs on BigRed only)
  2. Split bug analysis (from preprocess.py source code)
  3. Fake PD NIfTI spacing audit
  4. Existing evaluation metrics summary
  5. Mask/slice/fake-PD alignment audit (evaluate.py)            [CRITICAL]
  6. Checkpoint resume audit (train.py)                          [scheduler bug]
  7. Validation reproducibility audit (dataset.py)                [seeding bug]
  8. Full consolidated summary — what is broken vs confirmed correct

Run from repo root:
    venv/bin/python scripts/stage0_preflight.py
"""

import os, glob, json, sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def section(title):
    print("\n" + "="*65)
    print(f"  {title}")
    print("="*65)

def ok(msg):   print(f"  ✅  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def bug(msg):  print(f"  ❌  BUG: {msg}")
def info(msg): print(f"      {msg}")

# ─────────────────────────────────────────────────────────────────────
# 1. Directory inventory
# ─────────────────────────────────────────────────────────────────────
section("1. DIRECTORY INVENTORY")

dirs_to_check = {
    "inference_from_bigred2 (fake PD NIfTIs)": "inference_from_bigred2",
    "inference_from_bigred/translated_pd":     "inference_from_bigred/translated_pd",
    "preprocessed_masks/masks":                "preprocessed_masks/masks",
    "inference/eval2 (metrics + visuals)":     "inference/eval2",
    "inference/evaluation":                    "inference/evaluation",
    "runs/reggan_001 (local TB run)":          "runs/reggan_001",
    "scripts":                                 "scripts",
}

for label, rel in dirs_to_check.items():
    d = os.path.join(BASE, rel)
    if os.path.isdir(d):
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        print(f"  [{label}]")
        info(f"{len(files)} files")
        if len(files) <= 5:
            for f in sorted(files):
                info(f"  - {f}")
    else:
        warn(f"[{label}] — NOT FOUND locally (lives on BigRed)")

# Count fake PD volumes
fake_dir = os.path.join(BASE, "inference_from_bigred2")
fake_niftis = sorted(glob.glob(os.path.join(fake_dir, "*.nii.gz")))
print(f"\n  Fake PD volumes available locally: {len(fake_niftis)}")
if len(fake_niftis) < 69:
    info(f"  (expected ~69 total — only {len(fake_niftis)} synced to this machine;")
    info(f"   confirmed all 69 exist on BigRed, this is a local copy gap, not a pipeline bug)")

# ─────────────────────────────────────────────────────────────────────
# 2. Split bug analysis (from reading preprocess.py source)
# ─────────────────────────────────────────────────────────────────────
section("2. TRAIN/VAL SPLIT ANALYSIS  [STAGE 1a — FIXED & VERIFIED]")

print("""
  STATUS: ✅ FIXED on BigRed (job stage1_preprocess_7431588)

  preprocess/preprocess.py now uses split_by_patient(): groups slice
  paths by patient stem FIRST, then splits PATIENT IDs (not individual
  slices) into train/val/test, with a built-in leakage check that
  raises RuntimeError if any patient overlap is detected.

  Verified result (BigRed run, 2026-06-20):
    DESS  train=8800 slices/55 patients   val=1120/7   test=1120/7   -> 0 overlap
    PD    train=1804 slices/55 patients   val=243/7    test=224/7    -> 0 overlap

  Old splits.json (pre-fix, slice-level split) showed:
    DESS  train/val/test ALL contained ~69/69 patients (near-total leakage)
    PD    similar leakage (69/68/67)

  This section originally described the bug before the fix. Kept here
  as a historical record — see PIPELINE_VALIDATION_PLAN.md Stage 1 for
  the authoritative up-to-date status.
""")

ok("Stage 1a resolved — patient-level split confirmed, zero overlap")
info("Use preprocessed_v2/splits.json for any future training run")
info("(NOT the old preprocessed/splits.json, which has the leakage bug)")
info("")

# Try to read splits.json locally if it exists (for re-verification only)
local_splits = os.path.join(BASE, "data/preprocessed/splits.json")

if os.path.exists(local_splits):
    with open(local_splits) as f:
        splits = json.load(f)
    d_tr = splits["dess"]["train"]
    d_val = splits["dess"]["val"]
    d_te  = splits["dess"].get("test", [])

    def stem(path): return os.path.basename(path).rsplit("_sl", 1)[0]
    tr_patients  = set(stem(p) for p in d_tr)
    val_patients = set(stem(p) for p in d_val)
    overlap      = tr_patients & val_patients

    print(f"\n  Local splits.json found — re-verifying:")
    print(f"  DESS train: {len(d_tr)} slices from {len(tr_patients)} patients")
    print(f"  DESS val:   {len(d_val)} slices from {len(val_patients)} patients")
    print(f"  DESS test:  {len(d_te)} slices")
    if overlap:
        bug(f"{len(overlap)} patients appear in BOTH train AND val: {sorted(overlap)[:5]}...")
        info("(this would be the OLD pre-fix splits.json — check the path)")
    else:
        ok("No patient overlap between train and val")
else:
    info("No local splits.json to re-verify against (expected — it lives on BigRed")
    info("at preprocessed_v2/splits.json). Verification already done via")
    info("scripts/bigred_check_split.py on BigRed — see result above.")

# ─────────────────────────────────────────────────────────────────────
# 3. Fake PD spacing audit
# ─────────────────────────────────────────────────────────────────────
section("3. FAKE PD VOXEL SPACING AUDIT")

try:
    import nibabel as nib

    sample = fake_niftis[:5]
    spacings, shapes = [], []
    for path in sample:
        img   = nib.load(path)
        zooms = np.array(img.header.get_zooms()[:3])
        shape = np.array(img.get_fdata().shape[:3])
        fov   = shape * zooms
        spacings.append(zooms)
        shapes.append(shape)
        print(f"\n  {os.path.basename(path)}")
        info(f"shape  : {tuple(shape)}")
        info(f"spacing: R={zooms[0]:.4f}  A={zooms[1]:.4f}  S={zooms[2]:.4f}  mm")
        info(f"FOV    : {fov[0]:.1f} x {fov[1]:.1f} x {fov[2]:.1f}  mm")

    spacings = np.array(spacings)
    print(f"\n  Mean spacing over {len(sample)} files:")
    info(f"  R (through-plane): {spacings[:,0].mean():.4f} mm")
    info(f"  A (in-plane):      {spacings[:,1].mean():.4f} mm")
    info(f"  S (in-plane):      {spacings[:,2].mean():.4f} mm")

    expected_pd_R = 3.60
    actual_R      = spacings[:,0].mean()
    print()
    if abs(actual_R - expected_pd_R) > 0.1:
        bug(f"Through-plane spacing is {actual_R:.3f} mm but real PD is {expected_pd_R} mm")
        info(f"Fake PD inherited DESS through-plane spacing ({actual_R:.2f}mm)")
        info(f"This is {expected_pd_R/actual_R:.1f}x too fine vs real PD")
        info("Fix: resample_to_pd_spacing.py (Stage 2)")
    else:
        ok(f"Through-plane spacing {actual_R:.3f} mm matches real PD")

except ImportError:
    warn("nibabel not available in current Python — run with:")
    info("  ./venv/bin/python scripts/stage0_preflight.py")

# ─────────────────────────────────────────────────────────────────────
# 4. Existing evaluation metrics
# ─────────────────────────────────────────────────────────────────────
section("4. EXISTING EVALUATION METRICS (from BigRed)")

metrics_path = os.path.join(BASE, "inference/eval2/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
    print()
    print(f"  FID (fake PD vs real PD):   {m['FID_fake_vs_real']:.2f}")
    print(f"  FID (DESS vs real PD):       {m['FID_dess_vs_real']:.2f}  ← baseline")
    print(f"  FID improvement:             {m['FID_improvement']:.2f}  ({m['FID_improvement']/m['FID_dess_vs_real']*100:.1f}%)")
    print(f"  KID:                         {m['KID_mean']:.4f} ± {m['KID_std']:.2e}")
    print(f"  SSIM (DESS vs fake PD):      {m['SSIM_DESS_FakePD']:.4f}  (expected low for cross-modal)")
    print(f"  Mean deformation:            {m['mean_deformation_magnitude']:.6f} px")
    print(f"  Max deformation:             {m['max_deformation_magnitude']:.6f} px")
    print(f"  Jacobian det mean:           {m['jacobian_det_mean']:.6f}  (1.0 = no deformation)")
    print(f"  Jacobian det min:            {m['jacobian_det_min']:.4f}")
    print(f"  Folding %:                   {m['jacobian_folding_pct']:.1f}%")
    print(f"  Meniscus mean deformation:   {m['meniscus_mean_deformation']:.6f} px")
    print(f"  Meniscus max deformation:    {m['meniscus_max_deformation']:.6f} px")

    print()
    # Flag concerns
    kid_std = m['KID_std']
    if kid_std < 1e-10:
        warn(f"KID std = {kid_std:.2e} — suspiciously close to machine zero")
        info("This can happen if all KID subset estimates are identical")
        info("(may indicate evaluate.py KID subset sampling is not truly random)")
    else:
        ok(f"KID std looks reasonable")

    if m['mean_deformation_magnitude'] < 1e-4:
        bug(f"Mean deformation {m['mean_deformation_magnitude']:.8f} px is suspiciously near-zero")
        info("Could indicate R collapsed to trivial zero output")
        info("Run check_registration.py on BigRed to confirm (Stage 3)")
    else:
        ok(f"Deformation {m['mean_deformation_magnitude']:.4f} px is non-trivially small")

else:
    warn("metrics.json not found at inference/eval2/metrics.json")

# ─────────────────────────────────────────────────────────────────────
# 5. Mask / slice / fake-PD alignment audit (evaluate.py)
# ─────────────────────────────────────────────────────────────────────
section("5. MASK/SLICE/FAKE-PD ALIGNMENT AUDIT (evaluate.py)  [CRITICAL]")

print("""
  Source: inference/evaluate.py

  dess_slices    <- glob(dess_slice_dir/*.npy)        sorted alphabetically
  fake_pd_slices <- glob(fake_pd_dir/*.nii.gz)        per-patient volumes,
                                                       concatenated in sorted order
  mask_slices    <- glob(mask_dir/*.nii.gz)           per-patient volumes,
                                                       concatenated in sorted order,
                                                       then hard-truncated to
                                                       len(dess_slices)
""")

bug("mask_slices[i] / fake_pd_slices[i] / dess_slices[i] are zipped POSITIONALLY")
info("None of the three loaders match by patient ID/stem before zipping.")
info("Patient counts and slice counts differ between DESS, fake PD, and masks —")
info("so index i can silently pair patient A's mask with patient B's fake PD slice.")
info("")
info("Affected functions in evaluate.py:")
info("  - evaluate_deformation()      <- meniscus deformation metric (0.055px)")
info("  - plot_meniscus_overlays()")
info("  - plot_knee_boundary_overlay()")
info("  - plot_difference_map()")
info("")
bug("The headline meniscus deformation result is NOT provably computed on matched anatomy")
info("Fix: build dicts keyed by patient stem for all three slice sources,")
info("     iterate over the INTERSECTION of available patient IDs,")
info("     zip slices within each patient by matching slice index, not global position.")

# ─────────────────────────────────────────────────────────────────────
# 6. Checkpoint resume audit (train.py)
# ─────────────────────────────────────────────────────────────────────
section("6. CHECKPOINT RESUME AUDIT (train.py)")

print("""
  Source: train.py  _save_checkpoint() / _load_checkpoint()

  Saved:  epoch, global_step, G_AB, G_BA, D_A, D_B, R, opt_G, opt_D, opt_R
  NOT saved: sched_G, sched_D, sched_R  (LambdaLR schedulers)
""")

bug("LR schedulers are not saved/restored on resume")
info("LambdaLR keeps an internal step counter (last_epoch) separate from")
info("the optimizer state. On resume, this counter resets to 0, so the")
info("linear LR decay schedule restarts from the beginning instead of")
info("continuing from the epoch you resumed at.")
info("Impact: every BigRed job resubmission (common given the ~4hr SLURM")
info("time limit noted in CLAUDE.md) silently re-triggers the LR warmup/decay.")
info("Fix: save sched_G.state_dict() etc., or call sched.step() start_epoch times on resume.")

# ─────────────────────────────────────────────────────────────────────
# 7. Validation reproducibility audit (dataset.py)
# ─────────────────────────────────────────────────────────────────────
section("7. VALIDATION REPRODUCIBILITY AUDIT (dataset.py)")

print("""
  Source: dataset.py  UnpairedSliceDataset.__getitem__()

  d_path = self.dess_paths[idx % len(self.dess_paths)]      # deterministic
  p_path = self.pd_paths[random.randint(0, len(self.pd_paths)-1)]  # NOT deterministic
""")

bug("PD pairing uses random.randint with no fixed seed, even when aug=False")
info("`aug=False` only disables the augmentation transform — it does NOT")
info("freeze which PD slice gets paired with each DESS slice.")
info("Every call to the val_loader re-randomizes DESS-PD pairing.")
info("Impact: val/L1_unpaired_proxy is not just an unpaired metric (Stage 4),")
info("it is also NON-REPRODUCIBLE across epochs — comparing epoch N vs N+1")
info("partly compares random pairing noise, not real model improvement.")
info("This metric was used for 'best checkpoint' selection.")
info("Fix: seed the val dataset's PD index deterministically per-item")
info("(e.g. idx %% len(pd_paths) instead of random.randint, for val split only).")

# ─────────────────────────────────────────────────────────────────────
# 8. Full consolidated summary
# ─────────────────────────────────────────────────────────────────────
section("8. STAGE 0 SUMMARY — FULL LOGIC AUDIT")

print("""
  NOTE: this summary uses the CURRENT stage numbering from
  PIPELINE_VALIDATION_PLAN.md (Stage 1-11). That file is the
  authoritative source of truth going forward, not this script.

  STAGE 1 — Preprocessing fixes
  ──────────────────────────────
  ✅ [1a] Train/val split by SLICE not PATIENT — FIXED & VERIFIED on BigRed
          (55/7/7 patients, zero overlap — see section 2 above)
  ❌ [1b] MTR_016_mask.nii.gz corrupted (byte-count mismatch) — NOT FIXED
          fix in progress: scripts/check_mask_integrity.py

  STAGE 2 — Dataset/training-loop mechanical fixes — NOT FIXED
  ───────────────────────────────────────────────────────────────
  ❌ [2a] Validation PD pairing not seeded (dataset.py:54, random.randint)
  ❌ [2b] Checkpoint resume resets global_step/best_val; schedulers not saved
          (train.py:316-343, train())
  ❌ [2c] Augmentation rotate fill=0 in [-1,1] space (dataset.py _augment())
  ❌ [2d] Train/infer architecture defaults mismatch (ngf/n_res)

  STAGE 3 — R network redesign (root cause) — NOT FIXED, needs design decision
  ──────────────────────────────────────────────────────────────────────────────
  ❌ [3a] R trained on (fake_B, random unrelated patient's real_B)
  ❌ [3b] Eval calls R_net(fake_PD, DESS) — domain pairing R never trained on
          → deformation/Jacobian/meniscus metrics below are not reliable evidence

  STAGE 4 — Evaluation alignment fix — NOT FIXED [CRITICAL, see section 5 above]
  ─────────────────────────────────────────────────────────────────────────────────
  ❌ [4a] mask/fake-PD/DESS slices zipped positionally, no patient-ID match

  STAGE 5 — Inference spacing correctness — NOT FIXED [see section 3 above]
  ───────────────────────────────────────────────────────────────────────────
  ❌ [5a] Fake PD inherits DESS spacing (0.80mm) not real PD (3.60mm), 4.5x mismatch
  ❌ [5b] Diagonal affine doesn't preserve world-orientation matrix (lower impact)

  STAGE 6 — Optional / documented limitations, not blocking
  ─────────────────────────────────────────────────────────────
  ⚠️  FID uses 64x64 pixel features not InceptionV3 — relative comparison still valid
  ⚠️  KID std ≈ 0 (section 4 above) — check evaluate.py KID sampling logic

  WHAT IS VERIFIED CORRECT (no action needed):
  ────────────────────────────────────────────────
  ✅  Training step order: G → R → D (G.backward before R.step)
  ✅  Warp function: broadcast not expand, no inplace ops
  ✅  Discriminator image pool: 50-image history, correct
  ✅  Data normalisation: [0,1] → [-1,1] matches tanh, consistent train vs infer
  ✅  Inference (infer2.py) reuses preprocess.py's process_volume/normalise directly
  ✅  Mask preprocessing: order=0 nearest-neighbour (labels preserved)
  ✅  FID direction: fake_PD vs real_PD (correct for domain adaptation)
  ✅  SSIM direction: DESS vs fake_PD (expected low = translation happened)
  ✅  Identity/GAN/cycle loss directions all correct (CycleGAN convention)
  ✅  Gradient detach: no cross-network gradient leakage found
  ✅  Generator is fully-convolutional, no spatial-transformer — H x W exactly
      preserved input-to-output, confirmed by direct image inspection
      (so anatomy alignment doesn't actually depend on R at all)
""")

print("  STAGE 7+ (retrain, baseline segmentation, novel work) all depend on")
print("  Stages 1-6 above being resolved first. See PIPELINE_VALIDATION_PLAN.md")
print("  for the full Stage 1-11 roadmap and current progress.\n")
