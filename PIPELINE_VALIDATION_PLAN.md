# RegGAN Pipeline Validation & Extension Plan

> **Rule for every stage:** no stage begins until the previous one is reviewed, fixed, and verified.
> Every stage ends with a printed report or visualization so results can be inspected before moving on.
> Current checkpoints (`run_002`, `run_003`) and `inference/eval2/metrics.json` predate Stage 1's fix and
> should be treated as **provisional, not trustworthy** until Stage 7 (retrain) is complete.

---

## Stage 1 — Preprocessing fixes

| # | Bug | File | Status |
|---|---|---|---|
| 1a | Train/val/test split was by **slice**, not **patient** — same patient's slices could land in both train and val (data leakage) | `preprocess/preprocess.py:159` (old) | ✅ **FIXED & verified on BigRed.** New `split_by_patient()` groups slices by patient stem before splitting; built-in leakage check runs automatically and raises `RuntimeError` if any overlap is found. Verified result: DESS 55/7/7 patients, PD 55/7/7 patients, **zero overlap** in both. |
| 1b | `MTR_016_mask.nii.gz` failed to load locally — byte-count mismatch | `preprocessed_masks/masks/MTR_016_mask.nii.gz` (local copy only) | ✅ **RESOLVED — local-only transfer artifact, not a real data bug.** `scripts/check_mask_integrity.py` run on BigRed against the full mask directory (`/N/project/prostate_cancer_ai/anshika/regGAN/preprocessed/masks`) returned **0 corrupted** out of all files present. Fix: re-`scp` the file down from BigRed to replace the bad local copy. |
| 1c | Apparent mask count mismatch (64 visible vs 69 expected) | `preprocessed/masks/` (BigRed) | ✅ **RESOLVED — false alarm.** `ls` output was column-wrapped in the terminal, undercounting at a glance. Confirmed all 69 masks exist on BigRed. |

**Validation for this stage:**
- `scripts/bigred_check_split.py` — already run, confirmed 0 overlap (Stage 1a)
- `scripts/check_mask_integrity.py` — written and run both locally (1 failure, local-only) and on BigRed (0 failures) — confirms 1b resolved
- 1c confirmed resolved by direct count on BigRed (69/69)

**Stage 1 is now fully complete (1a, 1b, 1c all resolved).**

---

## Stage 2 — Dataset / training-loop mechanical fixes

Contained fixes, no design decisions required. All should land before any retrain.

| # | Bug | File | Status |
|---|---|---|---|
| 2a | Validation PD pairing uses `random.randint()` with no seed — non-reproducible val metric even when `aug=False` | `dataset.py:54` | ❌ Not fixed |
| 2b | Checkpoint resume resets `global_step=0`, `best_val=inf` regardless of `self.start_epoch`; LR schedulers (`sched_G/D/R`) are not saved/restored at all | `train.py` `_save_checkpoint()`/`_load_checkpoint()`/`train()` | ❌ Not fixed |
| 2c | Augmentation `TF.rotate(t, angle)` has no `fill` arg — torchvision defaults to `fill=0`, which is **mid-gray** in the `[-1,1]` normalization, not background. Rotated corners get gray artifacts baked into training data | `dataset.py` `_augment()` | ❌ Not fixed |
| 2d | Train/inference architecture defaults mismatch: `train.py` defaults `ngf=48, n_res=6`; `infer2.py` defaults `ngf=64, n_res=9`. Not yet triggered because actual SLURM jobs pass explicit matching args, but a future run without explicit args will crash on `load_state_dict` | `train.py` vs `inference/infer2.py` argparse defaults | ❌ Not fixed |

**Validation for this stage:**
- Instantiate `UnpairedSliceDataset(split="val", aug=False)`, call `__getitem__` on the same index twice, confirm identical PD pairing both times
- Save a dummy checkpoint mid-training, reload it, confirm `epoch`/`global_step`/scheduler `last_epoch` all continue correctly (not reset to 0)
- Render one augmented training sample, visually confirm no gray-fill artifact at rotated corners
- Print `train.py` and `infer2.py` argparse defaults side by side, confirm they match

---

## Stage 3 — Core registration (R) network redesign — root cause, needs a design decision

| # | Bug | File | Status |
|---|---|---|---|
| 3a | R is trained on `(fake_B, real_B)` where `real_B` is a **random unrelated patient's** PD slice — no consistent anatomical relationship across batches. R's smoothness+magnitude regularization likely wins by default, meaning R may have learned to output near-zero flow not because deformation is genuinely minimal, but because there's no consistent target to learn from | `dataset.py:54`, `train.py` `_forward()` | ❌ Not fixed — needs decision on what R's training target should be |
| 3b | Evaluation calls `R_net(fake_PD, DESS)` — a domain pairing R was **never trained on** (training used `(fake_PD, real_PD)`). The reported deformation/Jacobian/meniscus metrics (0.029px, 0.055px, 0% folding) are computed from R's behavior on an out-of-distribution input it has no learned basis for | `inference/evaluate.py:259` | ❌ Not fixed — depends on how 3a is resolved |

**Key supporting finding (verified by direct image inspection):** `Generator` (`models.py`) is a pure fully-convolutional encoder-decoder with no spatial-transformer layer. Stride-2 downsampling ×2 is matched by bilinear-upsample ×2 in the decoder, so **output H×W exactly matches input H×W with zero learned spatial offset by construction.** A direct mask overlay test (DESS mask → fake PD, zero warp) on a real volume showed the meniscus boundary landing exactly on the correct dark wedge-shaped structure in the fake PD image — confirming G_AB alone preserves anatomy correctly, independent of R. This means R's role should likely be re-scoped: a training-time regularizer only, not the tool used to *measure* anatomy preservation.

**Validation for this stage:**
- `scripts/check_registration.py` (already written) — run after retraining, confirm flow magnitude histogram is non-trivially non-zero
- Confirm train-time and eval-time R calls use the *same* domain pairing once redesigned
- Re-run the no-warp mask overlay check (already done once manually) as the primary anatomy-preservation evidence, independent of R

---

## Stage 4 — Evaluation pipeline alignment fix

| # | Bug | File | Status |
|---|---|---|---|
| 4a | `dess_slices`, `fake_pd_slices`, `mask_slices` are loaded independently (different glob patterns, different file-naming schemes) and zipped **positionally** by index — no patient-ID cross-referencing. `mask_slices[i]` can belong to a different patient than `fake_pd_slices[i]` | `inference/evaluate.py` (slice loaders, `evaluate_deformation()`, `plot_meniscus_overlays()`, `plot_knee_boundary_overlay()`, `plot_difference_map()`) | ❌ Not fixed |

**Validation for this stage:** rebuild slice loading to key all three sources by patient stem, iterate only over the intersection of available patient IDs, zip within-patient by matching slice index. Re-run `evaluate.py`, spot-check that the meniscus mask used at index *i* belongs to the same patient as the fake PD slice at index *i*.

---

## Stage 5 — Inference spacing correctness

| # | Bug | File | Status |
|---|---|---|---|
| 5a | Fake PD output inherits **DESS** through-plane spacing (0.80mm) instead of matching **real PD** (3.60mm) — confirmed numerically via `check_spacing.py` on actual local NIfTI files | `inference/infer2.py` `get_effective_spacing()` | ❌ Not fixed — diagnostic confirmed, fix script written, not yet applied |
| 5b | Diagonal affine (`np.diag([sp_R, sp_A, sp_S, 1.0])`) preserves spacing magnitude only, not the full world-orientation matrix | `inference/infer2.py` | ❌ Not fixed — lower impact, only matters for scanner-space visualization |

**Validation for this stage:**
- `scripts/check_spacing.py` (written, already run once — confirmed mismatch)
- `scripts/resample_to_pd_spacing.py` (written, not yet run) — apply, then re-run `check_spacing.py` on the resampled output to confirm match to real PD spacing

---

## Stage 6 — Optional / good-to-have (not blocking any goal)

| # | Item | Status |
|---|---|---|
| 6a | FID uses 64×64 raw pixel features, not InceptionV3 — absolute value non-standard, relative comparison (fake_PD vs real_PD < DESS vs real_PD) still valid | Documented in `evaluate.py` comment. No fix needed unless absolute FID comparability to published numbers is required |
| 6b | Diagonal affine world-orientation nuance (if not addressed in 5b) | Optional |

---

## Stage 7 — Retrain clean baseline + re-evaluate

Once Stages 1–6 are resolved:
1. Retrain from scratch using the Stage 1 patient-level `splits.json`, with all Stage 2 mechanical fixes and the Stage 3 R-network redesign in place
2. Re-run inference with Stage 5 spacing fix applied
3. Re-run evaluation with Stage 4 alignment fix applied
4. Compare new metrics against the old (invalid) `inference/eval2/metrics.json` — expect different (likely more honest) numbers, especially for deformation/meniscus metrics

This produces the first **trustworthy** checkpoint and metric set in the project.

---

## Stage 8 — Baseline segmentation (GATE before novel work)

**Goal:** train a segmentation model on fake PD (Stage 7 output) + DESS masks, test on real PD. This is the proof that domain adaptation actually helps — and the reference point all novel contributions (Stage 9+) must beat.

**Depends on:**
- Stage 1 (no leaked patients between the fake PD used for training segmentation and any held-out test patients)
- Stage 4 (masks must actually correspond to the fake PD slices used)
- Stage 5 (spacing must match real PD for the segmentation model to generalize)
- Stage 7 (must use the retrained, valid checkpoint — not `run_002`/`run_003`)

**This stage cannot be run meaningfully before Stage 7 completes.** Running it earlier would train on mismatched masks, wrong spacing, and an invalid generator checkpoint — any result would be uninterpretable.

**Validation:** Dice/boundary distance on any labeled real PD scans (if available) or visual inspection of predicted masks on real PD; compare segmentation trained on fake-PD vs DESS-direct as a sanity baseline.

---

## Stage 9 — Novel work: Rigid/non-rigid decomposition

**Idea:** decompose the learned deformation field into rigid (harmless repositioning) and non-rigid (harmful shape distortion) components, penalizing only the residual:
```
flow = flow_rigid + flow_nonrigid
flow_rigid = least-squares best-fit translation + rotation over the meniscus region
L_nonrigid = E[‖flow_nonrigid(x,y)‖²]   (replaces uniform L_mag penalty)
```

**Why novel:** existing methods use rigid alignment as *preprocessing* or apply uniform Jacobian/smoothness/incompressibility constraints — none explicitly decompose a *learned* field into rigid vs non-rigid components inside the training loop and penalize only the harmful part.

**Gated behind Stage 8** — no reference baseline to know if this actually improves segmentation transfer without it.

---

## Stage 10 — Novel work: Statistical SDM (Signed Distance Map) constraint

**Idea:** instead of penalizing raw deformation magnitude, penalize change in the anatomical boundary's signed distance field before/after warping:
```
SDF_A = compute_sdf(mask_A)
SDF_warped = warp(SDF_A, flow)
L_sdm = mean((SDF_warped - SDF_A)²)
```
More principled than magnitude/smoothness losses — directly constrains shape change rather than all displacement equally.

**Gated behind Stage 8**, same reasoning as Stage 9.

---

## Stage 11 — Ablation study + final evaluation

Compare across: CycleGAN baseline → RegGAN (current, fixed) → +rigid/nonrigid (Stage 9) → +SDM (Stage 10) → +both. Metrics: meniscus deformation, FID, segmentation Dice on real PD. Final visualizations: boundary overlays, flow fields, Jacobian maps, segmentation predictions — all models side by side.

---

## Progress Summary (as of this writing)

| Stage | Status |
|---|---|
| 1 | 1a done + verified. 1b found, not fixed. |
| 2 | All 4 identified, none fixed. |
| 3 | Root cause fully diagnosed; fix needs a design decision from you. |
| 4 | Identified via dedicated audit, not fixed. |
| 5 | Diagnostic confirmed numerically; fix script written, not applied. |
| 6 | Documented only (optional, by design). |
| 7 | Not started — depends on 1–6. |
| 8 | Not started — depends on 7. |
| 9–11 | Not started — depends on 8. |

**Net status:** 1 of 13 identified bugs fixed and verified. All others documented with file:line evidence. No retraining has happened since Stage 1's fix — current checkpoints predate it entirely.
