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

**Re-validated against freshly regenerated masks (`preprocessed_v2/masks`):** running `preprocess_masks.py` from scratch surfaced a real **regression** — 22/69 masks failed with `"ITK only supports orthonormal direction cosines"`. This exactly matches a previously-documented, previously-fixed issue (CLAUDE.md "Key Bugs Fixed #5"), whose nibabel fallback had gone missing from the current `reorient_to_ras_mask()`. Restored the fallback (`nib.as_closest_canonical()` when SimpleITK rejects non-orthonormal affines), re-ran, and re-validated 1b/1c against the new output. **Confirmed 1b and 1c complete on the regenerated masks.**

**Stage 1 is now fully complete (1a, 1b, 1c all resolved and re-validated on the current, regenerated masks — not stale/old data).**

---

## Stage 2 — Dataset / training-loop mechanical fixes  ✅ COMPLETE

Contained fixes, no design decisions required.

| # | Bug | File | Status |
|---|---|---|---|
| 2a | Validation PD pairing uses `random.randint()` with no seed — non-reproducible val metric even when `aug=False` | `dataset.py` | ✅ **FIXED.** Val/test split now uses deterministic `idx % len(pd_paths)` pairing; train split keeps random cross-patient pairing (intentional). Verified synthetically: same index returns identical PD path across repeated calls. |
| 2b | Checkpoint resume resets `global_step=0`, `best_val=inf` regardless of `self.start_epoch`; LR schedulers (`sched_G/D/R`) are not saved/restored at all | `train.py` `_save_checkpoint()`/`_load_checkpoint()`/`train()` | ✅ **FIXED & verified on real BigRed training run** (job `stage2_verify_7432661`). Schedulers now saved/restored via `state_dict()`, with a fast-forward fallback for older checkpoints. Evidence: `global_step` continued 4400→6600 across resume (not reset to 0), `best_val=0.4736` carried over (not `inf`), log explicitly printed `"Resumed from epoch 1 step 4400 best_val=0.47361599813614574"` with no `"No scheduler state in checkpoint"` fallback warning — confirms the scheduler `load_state_dict()` path was taken successfully. |
| 2c | Augmentation `TF.rotate(t, angle)` has no `fill` arg — torchvision defaults to `fill=0`, which is **mid-gray** in the `[-1,1]` normalization, not background. Rotated corners get gray artifacts baked into training data | `dataset.py` `_augment()` | ✅ **FIXED.** Added `fill=-1.0`. Verified synthetically: rotated sample's corner fill value is `-0.97` (near background `-1`), not `0` (mid-gray). |
| 2d | Train/inference architecture defaults mismatch: `train.py` defaults `ngf=48, n_res=6`; `infer2.py` defaults `ngf=64, n_res=9`. Not yet triggered because actual SLURM jobs pass explicit matching args, but a future run without explicit args will crash on `load_state_dict` | `train.py` vs `inference/infer2.py` argparse defaults | ✅ **FIXED.** Both now default to `ngf=48, n_res=9` (matching actual production config). Verified: both Phase 1 and Phase 2 of the BigRed verification run used these defaults successfully with no shape mismatch. |

**Validation performed:**
- 2a, 2c: synthetic local unit tests (dummy `.npy` files, no real dataset needed)
- 2b, 2d: real BigRed training run — short training (2 epochs) → checkpoint → resume (1 more epoch) → confirmed via `.err` log grep and direct checkpoint inspection
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

### Attempt 1 (reverted) — retarget R to `(fake_B, real_A)`

Tried retargeting R's fixed image from random `real_B` to the true source `real_A` (in an isolated `train_stage3.py`, `train.py` never touched). Also dropped the raw-pixel `l_reg_sim` loss since `warped` (PD-contrast) vs `real_A` (DESS-contrast) pixel L1 isn't meaningful across the contrast gap.

**Result (verified on a real BigRed training run, job `stage3_verify_7434020`):** R's flow collapsed to **exactly zero** (`mean=0.000000px`, `smooth=0.0000`, `mag=0.0000` on literally every logged step across 2 epochs) — worse than the original near-zero collapse. Root cause: with `l_reg_sim` removed, the only remaining losses (`l_smooth`, `l_mag`) are both uniquely minimized at `flow≡0` — there was nothing left in the loss function rewarding R for predicting anything else. This is a pure consequence of the loss design, not noisy data.

**Status:** reverted. `train_stage3.py` restored to be byte-identical to `train.py` again — none of this attempt's changes are active anywhere. Stage 3 is back to "not started, root cause diagnosed."

**Lesson for the next attempt:** any retargeting of R must keep *some* similarity-driving loss (rewarding R for finding/correcting real misalignment), not just the smoothness/magnitude regularizers. Likely direction: a contrast-invariant similarity loss (e.g. edge/gradient-magnitude L1 between `warped` and `real_A`, instead of raw pixel L1) — proposed but not yet implemented or tested.

### Decision: deferred, proceeding with original R-network logic

**2026-06-20 — user decision:** leave Stage 3 unresolved for now. Proceed to Stage 7 (retrain) using `train.py` exactly as it stands today (Stage 1+2 fixes applied, original `R(fake_B, real_B)` + `l_reg_sim` logic unchanged — this is the same logic that has shipped since the project's original `run_002`/`run_003` checkpoints).

**What this means for interpreting Stage 7's results:** the theoretical concern from 3a/3b (R trained on inconsistent cross-patient targets, evaluated on a pairing it never saw) still applies to whatever Stage 7 produces. However, the empirical check from the reverted Attempt 1 (`stage3_verify_7434734`, with the *original* logic) showed R does **not** collapse to literal zero — it produced a small, consistent, non-degenerate flow (`mean=0.0226px`) even on an out-of-distribution pairing. So the original logic is not catastrophically broken, just not rigorously validated as measuring genuine registration. Treat Stage 7's deformation/meniscus numbers as informative but not fully trustworthy until Stage 3 is revisited.

**Stage 3 remains open for future work** — revisit with the edge-magnitude similarity loss approach if/when there's time to do it properly.

---

## Stage 4 — Evaluation pipeline alignment fix

| # | Bug | File | Status |
|---|---|---|---|
| 4a | `dess_slices`, `fake_pd_slices`, `mask_slices` are loaded independently (different glob patterns, different file-naming schemes) and zipped **positionally** by index — no patient-ID cross-referencing. `mask_slices[i]` can belong to a different patient than `fake_pd_slices[i]` | `inference/evaluate.py` (slice loaders, `evaluate_deformation()`, `plot_meniscus_overlays()`, `plot_knee_boundary_overlay()`, `plot_difference_map()`) | ✅ **Code fix applied.** Added `extract_patient_id()` helper + rewrote the loading section in `main()` to key all three sources by patient ID, intersect available patients, and zip within-patient by matching slice index. Syntax-checked locally. **Real-data validation deferred** — see note below. |
| 4b | Neither `infer2.py` nor `evaluate.py` reference `splits.json` at all. `infer2.py` recursively globs and translates *every* DESS volume in `dess_root` (train+val+test all mixed, no filtering). `evaluate.py` takes raw `--fake_pd_dir`/`--dess_slice_dir`/`--mask_dir` paths with no split filtering either | `inference/infer2.py` (`run_inference`), `inference/evaluate.py` (argparse) | ✅ **Code fix applied.** Added `--splits`/`--split` args to both. `infer2.py`: `get_split_patient_stems()` reads `splits.json`, filters whole-volume DESS NIfTIs by full patient stem before translating. `evaluate.py`: filters `common_patients` (from the Stage 4a fix) against the chosen split's patient IDs (via `extract_patient_id()`, reused from 4a — confirmed consistent short-ID matching with a standalone test). Both args optional — omitting them preserves old behavior (no filtering) with an explicit warning logged. Syntax-checked locally. |

**Why 4b matters:** all reported metrics (FID, deformation, meniscus) are very likely computed on a mix that includes patients the GAN was *trained* on, not held-out val/test patients. This is evaluating on training data and inflates apparent performance — directly relevant to "does everything align with train/val/eval," not just an evaluation-correctness nuance.

**Validation for this stage:**
- 4a: rebuild slice loading to key all three sources by patient stem, iterate only over the intersection of available patient IDs, zip within-patient by matching slice index. **Code fix applied and syntax-checked locally** (`inference/evaluate.py`).
- 4b: add `--split` filtering to both `infer2.py` and `evaluate.py`. **Code fix applied, syntax-checked, patient-ID extraction consistency verified locally with a standalone test** (`MTR_005_Anonymized_2378615199_e1_sl0086.npy` → `MTR_005`, matching the short ID scheme masks use).

**✅ REAL-DATA VALIDATION COMPLETE (job `stage4_evaluate_7441595`, run against the interim Stage 7 checkpoint + freshly regenerated `preprocessed_v2/masks`):**
- 4b confirmed in both `infer2.py` (`"Filtered to --split=val: 69 -> 7 volumes"`) and `evaluate.py` (`"Filtered to --split=val: 7 -> 7 patients"`) logs — split filtering active and consistent in both places
- 4a confirmed structurally: `"Patients: 69 DESS, 7 fake PD, 69 masks -> 7 usable in common"` — intersection logic correctly narrows to only patients present in all three sources, instead of naively zipping mismatched lists
- 4a confirmed visually: `meniscus_overlays/` output inspected (and cross-checked in 3D Slicer with the regenerated `MTR_005` mask + translated PD volume) — boundaries land on correct anatomy, user-confirmed "looks fine"

**Stage 4 (4a + 4b) is now fully validated on real data, not just syntax-checked.**

---

## Stage 5 — Inference spacing correctness

| # | Bug | File | Status |
|---|---|---|---|
| 5a | Fake PD output inherits **DESS** through-plane spacing (0.80mm) instead of matching **real PD** (3.60mm) — confirmed numerically via `check_spacing.py` on actual local NIfTI files | `inference/infer2.py` `get_effective_spacing()`, `scripts/resample_to_pd_spacing.py` | ✅ **FIXED & verified on real BigRed data (job `stage5a_resample`).** Ran against the interim Stage 7 checkpoint's fake PD output per user decision (not waiting for final retrain). Found and fixed two real bugs along the way: (1) `get_mean_spacing()` read raw native spacing without RAS reorientation, comparing mismatched physical axes; (2) real PD has heterogeneous native acquisition resolutions (384×384@0.39mm vs 768×768@0.18-0.21mm) — averaging raw spacing across these produced a meaningless target. Fixed by reusing the same effective-spacing formula already proven correct elsewhere (`preprocess.py`/`infer2.py`'s reorient→isotropic-resample→divide-by-384), applied to real PD, using median over 15 files for robustness. Also fixed the same axis-order bug in `check_spacing.py`'s real-PD reading and summary comparison (was averaging raw native spacing against fake PD's effective spacing — apples vs oranges). **Final verified result: `RATIO (real_PD / fake_PD): R: 1.00x A: 1.00x S: 1.00x` — exact match on all axes, no mismatch.** |
| 5b | Diagonal affine (`np.diag([sp_R, sp_A, sp_S, 1.0])`) preserves spacing magnitude only, zeroes out origin, assumes pure-identity direction | `inference/infer2.py`, `scripts/resample_to_pd_spacing.py` | ✅ **Code fix applied in both files.** `infer2.py`'s `get_effective_affine()` and `resample_to_pd_spacing.py`'s resample loops now build the affine from the source's actual `direction` matrix and `origin` instead of a zero-origin diagonal. |
| 5b-2 | **NEW — LPS/RAS coordinate convention bug.** SimpleITK reports `GetDirection()`/`GetOrigin()` in LPS (DICOM convention) regardless of `DICOMOrient(img,"RAS")` array-labeling. Copying these directly into a `nibabel` affine (which expects RAS+ per the NIfTI standard) silently flipped left-right/anterior-posterior — caught via visual inspection in 3D Slicer (anatomy appeared mirrored). | `inference/infer2.py` `get_effective_affine()`, `preprocess/preprocess_masks.py` fallback affine | ✅ **Fixed and verified on real data.** Standard LPS→RAS conversion applied (`diag([-1,-1,1]) @ direction`, same flip on origin) in both files. **Verified**: affine diagonal changed from `[-3.60, -0.42, +0.42]` (two negative — the bug signature) to `[+3.60, +0.42, +0.42]` (all positive — correct RAS+ pattern) on the re-generated `MTR_005` output. Origin's first two components correctly flipped sign as expected from the conversion. Spacing ratio remains `1.00x` on all axes (unaffected, confirming this was purely an orientation fix, not a magnitude regression). |

**Validation for this stage — complete, on real data:**
- `scripts/check_spacing.py` — run multiple times, bugs found and fixed along the way, final run confirms 1.00x match on all axes
- `scripts/resample_to_pd_spacing.py` — run against real BigRed data, two real bugs found and fixed (axis-order + heterogeneous-resolution averaging), final output verified correct
- Visual/Slicer cross-check performed on `MTR_005` (fake PD + mask) — confirmed no distortion after the fix (was visibly blocky/warped before)

### Stage 5c — Reverse-preprocessing / round-trip verification (for long-term inference goal)

Checked whether converting pseudo-PD slices back to NIfTI, and converting predicted segmentation masks on real PD back to NIfTI, loses information or misaligns with the original volumes.

| Finding | Detail | Status |
|---|---|---|
| Real PD preprocessing is a spatial no-op | Verified numerically: native PD shape (384×384) already equals `TARGET_SIZE`, and native in-plane spacing (0.39mm both dims) is already isotropic — so `process_volume()` triggers **neither** the resample nor the resize step for PD. Only intensity normalization happens. | ✅ **Good news, no fix needed** — converting a predicted mask back to NIfTI for a real PD scan just needs the original PD affine reattached, no inverse-resize math required |
| DESS preprocessing is NOT a spatial no-op | Verified numerically: native DESS shape (512×512) does trigger a real resize to 384×384 (scale factor 0.75, cubic interpolation) — effective in-plane spacing becomes `0.31/0.75 = 0.413mm`, vs real PD's native `0.39mm` (~6% mismatch) | ⚠️ Reinforces Stage 5a — pseudo-PD must be resampled to real PD's *exact* native spacing (not just through-plane) before segmentation training, or the model trains at a different effective resolution than it'll see at real-PD inference time |
| No script existed to reconstruct a volume from saved `.npy` slices, handling background slices that `extract_slices()` skips (`mean < 0.02`) | `extract_slices()` preserves the *original* slice index in the filename (`sl{i:04d}`, not a renumbered count) even when slices are skipped — confirmed reconstruction is mechanically possible, just not implemented anywhere | ✅ **Written and verified**: `scripts/reconstruct_volume_from_slices.py`. Synthetic test (20-slice volume, 4 intentionally-skipped background slices) confirmed: skipped indices detected exactly, non-skipped slices reconstructed exactly, gaps correctly filled rather than silently compacted |

**Why this matters for the 3-phase goal:** the long-term goal is fine-tune on pseudo-PD → infer on real unlabeled PD → get usable segmentation masks back in NIfTI form. This confirms the *real PD* side of that round-trip is clean (no resampling to invert), and surfaces that the *pseudo-PD* side needs Stage 5a's spacing fix to target real PD's *exact* spacing (both in-plane and through-plane) for the round-trip to be consistent end-to-end. The reconstruction script is a new prerequisite tool for Stage 8 (turning per-slice segmentation predictions back into a usable NIfTI volume).

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

**New prerequisite found:** `preprocess_masks.py` only produces whole-volume mask NIfTIs (for visualization/evaluation) — it has **no slice-extraction step** analogous to `preprocess.py`'s `extract_slices()` for images. There is currently no mechanism producing paired `(image_slice.npy, mask_slice.npy)` training data at the slice level. This needs to be built as part of Stage 8, not assumed to already exist.

**Reconstruction tooling ready:** `scripts/reconstruct_volume_from_slices.py` (built during Stage 5c) supports `--mask_mode` for `int16` label-preserving reconstruction — verified synthetically that integer labels survive exactly (no float casting/interpolation). This will be used to convert Stage 8's per-slice segmentation predictions on real PD back into a usable NIfTI volume.

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
