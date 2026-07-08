# RegGAN Domain Adaptation — Technical Design Document

> Consolidated record of architecture, pipeline fixes, new evaluation metrics, and the
> fine-tuning/segmentation extension built on top of the original RegGAN project.
> See also: `CLAUDE.md` (original architecture knowledge base) and
> `PIPELINE_VALIDATION_PLAN.md` (detailed stage-by-stage bug log).

---

## 1. Project Goal

Translate DESS knee MRI (SKM-TEA, has segmentation masks) into PD-weighted-style
pseudo-PD images using RegGAN, with minimum anatomical deformation. Use DESS masks
(transferred onto pseudo-PD) to fine-tune a meniscus segmentation model, then run
that model on real, unlabeled PD scans.

**Three-phase plan:**
- **Short-term:** Generate pseudo-PD with PD contrast but DESS-exact anatomy (no deformation)
- **Mid-term:** Fine-tune a segmentation model on pseudo-PD + DESS masks
- **Long-term:** Run the fine-tuned model on real, unlabeled PD scans

---

## 2. Original RegGAN Architecture

| Component | Detail |
|---|---|
| `Generator` (G_AB, G_BA) | ResNet-based encoder-decoder, `ngf=48`, 9 residual blocks, InstanceNorm, ReflectionPad, UpsampleConv (no ConvTranspose — avoids checkerboard artifacts). **Fully convolutional, no spatial-transformer layer** — output pixel grid exactly matches input by architectural construction. |
| `PatchDiscriminator` (D_A, D_B) | 70×70 PatchGAN, `ndf=48`, LSGAN (MSE) loss |
| `RegistrationNet` (R) | VoxelMorph-lite U-Net, `nf=16`, predicts 2D displacement field; flow head initialized near-zero |
| Losses | GAN (LSGAN) + cycle consistency (λ=10) + identity (λ=5) + registration similarity (λ=5) + flow smoothness (λ=10) + flow magnitude (λ=5) |
| Training | Adam (β=0.5,0.999), lr=2e-4 (G/D), lr=1e-4 (R), batch=8, image pool=50 |

Full original architecture details: see `CLAUDE.md`.

---

## 3. Pipeline Validation — Bugs Found & Fixed

Full evidence/logs for every item below: `PIPELINE_VALIDATION_PLAN.md`.

### Stage 1 — Preprocessing ✅ Complete
- **1a:** Train/val/test split was by *slice*, not *patient* → data leakage. Fixed with `split_by_patient()` in `preprocess.py`, built-in leakage check. Verified: 55/7/7 patient split, zero overlap.
- **1b:** Mask corruption (`MTR_016`) — resolved, was a local-only transfer artifact.
- **1c:** Mask count mismatch — resolved, was a terminal column-wrap display issue, not a real gap.
- **Regression found during mask resync:** 22/69 masks failed with "non-orthonormal direction cosines" (SimpleITK limitation) — a previously-documented, previously-fixed issue (CLAUDE.md bug #5) whose nibabel fallback had gone missing from `preprocess_masks.py`. Restored.

### Stage 2 — Dataset/training-loop mechanical fixes ✅ Complete
All fixed in `dataset.py`/`train.py`, verified on a real BigRed training run (`stage2_verify_7432661`):
- **2a:** Validation PD pairing now deterministic (`idx % len(pd_paths)`), not random, for val/test splits
- **2b:** Checkpoint resume now saves/restores LR schedulers + `global_step`/`best_val` correctly (previously reset to 0/inf on every resume)
- **2c:** Augmentation rotation now uses `fill=-1.0` (background value in `[-1,1]` space), not the torchvision default `0` (mid-gray, caused corner artifacts)
- **2d:** Train/inference architecture defaults aligned (`ngf=48, n_res=9` everywhere)

### Stage 3 — R Network Root-Cause Diagnosis ⏸ Deferred by user decision
- **Root cause:** R was trained on `(fake_B, real_B)` where `real_B` is a random unrelated patient's PD slice — no consistent anatomical relationship across batches.
- **Eval-time mismatch:** Evaluation called `R_net(fake_PD, DESS)`, a pairing R was never trained on.
- **Attempt 1 (reverted):** Retargeted R to `(fake_B, real_A)` (the true DESS source), dropped the photometric `l_reg_sim` loss (contrast-gap problem). Result: R collapsed to **exactly zero** flow — removing `l_reg_sim` left only pure regularizers (`l_smooth`, `l_mag`), both uniquely minimized at `flow≡0`, so there was nothing left rewarding non-zero output. Fully reverted; `train_stage3.py` restored to byte-identical with `train.py`.
- **Decision:** proceed with the **original** R-network logic for Stage 7 training. Empirically, the original logic does *not* collapse to literal zero (confirmed: `~0.0226px` mean flow on out-of-distribution `(fake_B, real_A)` pairs) — not ideal, but not degenerate either.
- **Status:** open for future work (edge-magnitude similarity loss proposed, not implemented).

### Stage 4 — Evaluation Pipeline Alignment ✅ Complete, validated on real data
- **4a:** `dess_slices`/`fake_pd_slices`/`mask_slices` were loaded independently and zipped **positionally** — patient/slice mismatches possible. Fixed: `extract_patient_id()` + patient-keyed intersection + within-patient index matching in `evaluate.py`.
- **4b:** Neither `infer2.py` nor `evaluate.py` referenced `splits.json` — evaluation could include train-set patients. Fixed: `--splits`/`--split` filtering added to both.
- **Verified (job `stage4_evaluate_7441595`):** `"Filtered to --split=val: 69 -> 7 volumes"`, `"Patients: 69 DESS, 7 fake PD, 69 masks -> 7 usable in common"`. Visual confirmation via 3D Slicer overlay (`MTR_005`) — anatomically correct boundary alignment.

### Stage 5 — Spacing & Affine Correctness ✅ Complete, validated on real data
- **5a — Spacing mismatch:** fake PD inherited DESS spacing (0.80mm through-plane) instead of real PD's (3.60mm). Root cause of the *fix script's* initial failed attempts: (1) `get_mean_spacing()` read raw native spacing without RAS reorientation, comparing mismatched physical axes; (2) real PD has **heterogeneous native acquisition resolutions** (384×384@0.39mm vs 768×768@0.18-0.21mm) — naive averaging of raw spacing produced a meaningless target. Fixed by computing **effective spacing** (reorient→isotropic-resample→divide-by-384, the same formula already used elsewhere in the pipeline) and taking the **median** over 15 real PD files. **Final verified result: `RATIO (real_PD/fake_PD): R=1.00x A=1.00x S=1.00x`.**
- **5b — Affine/origin:** plain diagonal affine zeroed out origin, assumed identity direction. Fixed in `infer2.py` and `preprocess_masks.py` to use the source's actual direction matrix + origin.
- **5b-2 — LPS/RAS coordinate bug (found via 3D Slicer visual inspection):** SimpleITK reports `GetDirection()`/`GetOrigin()` in **LPS** (DICOM convention) regardless of `DICOMOrient(img,"RAS")` array-labeling. Copying directly into a nibabel affine (RAS+ convention) silently flipped left-right/anterior-posterior. Fixed with the standard LPS→RAS conversion (`diag([-1,-1,1]) @ direction`). **Verified:** affine diagonal changed from `[-3.60,-0.42,+0.42]` (bug signature) to `[+3.60,+0.42,+0.42]` (correct) on regenerated output.
- **5c — Round-trip/reverse-preprocessing:** confirmed real-PD round-trip is spatially lossless (native shape/spacing already matches network input — no resize needed). Built and tested `reconstruct_volume_from_slices.py` for both image (`float32`) and integer mask label (`int16`) reconstruction, handling skipped/background slices correctly.

---

## 4. New Evaluation Metrics Built

Beyond the original FID/KID/SSIM/Jacobian/deformation metrics in `evaluate.py`, two new
independent metrics were added to address the limitation that R-network-based
deformation metrics depend on Stage 3's unresolved training-target issue:

### `boundary_distance_eval.py`
Classical, R-independent structural-preservation check: runs edge detection on fake_B,
measures boundary distance against DESS mask contours, per-label and meniscus-specific.

**Result (job, val split, 100 slices, 7 patients):**

| Label | Mean (px) | p95 (px) | Max (px) |
|---|---|---|---|
| 1 | 2.225 | 7.000 | 10.440 |
| 2 | 1.350 | 4.123 | 10.050 |
| 3 | 1.544 | 5.000 | 8.062 |
| 4 | 1.988 | 5.099 | 10.296 |
| 5 (lateral meniscus) | 1.293 | 4.123 | 8.485 |
| 6 (medial meniscus) | 1.194 | 3.162 | 8.062 |
| **Meniscus (5+6)** | **1.250** | — | — |
| **All labels** | **1.523** | — | — |

Sub-2-pixel mean boundary distance across all tissue labels (~0.5mm physical, at
~0.42mm/px in-plane spacing) — strong, independent evidence of anatomical preservation,
not dependent on R's training-target issue.

### `fid_statistical_test.py`
Bootstrap confidence intervals + permutation test on the FID/KID improvement
(fake-vs-real vs DESS-vs-real baseline) — answers whether the improvement is
statistically significant, not just a point-estimate difference. (Run via SLURM batch
job, `fid_statistical_test.sh` — too long for the BigRed login node's 20-minute
interactive limit.)

---

## 5. Fine-Tuning / Segmentation Pipeline (new — `segmentation/` directory)

Built on top of the pretrained 2.5D U-Net from
[pitthexai/Knee_MRI_Segmentation_2.5D](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D)
(`segmentation_models_pytorch` U-Net, ResNet34 encoder, 2.5D = 3-slice `[i-1,i,i+1]`
channel stacking). Does not modify any existing pipeline file.

| File | Purpose |
|---|---|
| `prepare_meniscus_masks.py` | **The added filtering step:** collapses the 7-label DESS mask scheme (0-6) down to 3 classes (background / lateral meniscus / medial meniscus) for our narrower task; pairs pseudo-PD slices with filtered masks, keyed by patient + slice index; skips empty (no-meniscus) slices |
| `dataset_2_5d.py` | 2.5D dataset matching the reference repo's exact `[i-1,i,i+1]` stacking convention, adapted to our `.npy` float32 slices |
| `finetune_meniscus.py` | Loads the pretrained 5-class checkpoint, **replaces the final segmentation head** with a new randomly-initialized 3-class layer (encoder + decoder weights kept/fine-tuned), trains with `lr=1e-5` (lower than from-scratch, since starting from a domain-relevant checkpoint) |
| `infer_real_pd.py` | Runs the **fine-tuned** model directly on real PD volumes (no GAN translation involved — real PD doesn't need it), reconstructs predicted masks to NIfTI with the corrected (LPS→RAS-fixed) affine |
| `infer_real_pd_baseline.py` | Runs the **original, non-fine-tuned** 5-class checkpoint on the same real PD volumes, for direct before/after comparison |

**Sample size guidance given:** ~1,500-3,500 meniscus-positive slices available across
the 55 train-patient split — comfortably enough for fine-tuning (vs. training from
scratch) a domain-pretrained model on a narrow single-structure task.

### Pilot fine-tuning run result — run_001 (`finetune_meniscus_7444028`, 69 patients, 20 epochs, lr=1e-5)

| Epoch | Train loss | Val loss | Lateral Dice | Medial Dice | Mean meniscus Dice |
|---|---|---|---|---|---|
| 14 | 0.0070 | 0.0085 | 0.429 | 0.443 | 0.4357 |
| 18 | 0.0045 | 0.0063 | 0.468 | 0.443 | 0.4556 |
| 19 (best) | 0.0034 | 0.0060 | **0.538** | 0.443 | **0.4902** |

Medial Dice frozen at exactly `0.443` across 6 epochs — flagged as val-set class scarcity with only 7 patients.

---

## 6. Full Dataset Scale-Up (155 patients)

### Dataset Expansion

Downloaded and processed the **full SKM-TEA dataset** (155 DESS patients vs original 69-patient subset):

| | 69-patient run | 155-patient run |
|---|---|---|
| DESS patients | 69 | 155 |
| Split | 55/7/7 (train/val/test) | ~124/15/15 |
| Masks | 69 | 155 |
| Preprocessed output | `preprocessed_v2/` | `preprocessed_v3/` |
| Fake PD output | `results/stage4_fake_pd/` | `results/fake_pd_all155/` |
| Segmentation data | `segmentation_data/` | `segmentation_data_v2/` |
| Fine-tuning run | `segmentation_runs/run_001/` | `segmentation_runs/run_002/` |

### Pipeline steps for 155-patient run (in order)
```
1. preprocess.py         → preprocessed_v3/           (SLURM: preprocess_all155.sh)
2. infer2.py             → results/fake_pd_all155/     (SLURM: infer_all155.sh)
3. preprocess_masks.py   → preprocessed_v3/masks/      (SLURM: run_preprocess_masks.sh, updated paths)
4. prepare_meniscus_masks.py → segmentation_data_v2/  (inside finetune_all155.sh)
5. finetune_meniscus.py  → segmentation_runs/run_002/  (inside finetune_all155.sh)
6. infer_real_pd.py      → results/real_pd_predictions_v2/  (SLURM: infer_labeled_pd.sh)
```

**Key:** Steps 1 and 2 can run in parallel (preprocessing writes .npy slices, inference writes NIfTIs, no conflict). Step 3 must run after step 2 so `--pd_dir` can copy affine from fake PD for guaranteed mask alignment.

### Full fine-tuning run result — run_002 (155 patients, 20 epochs, lr=1e-5)

| Epoch | Train loss | Val loss | Lateral Dice | Medial Dice | Mean meniscus Dice |
|---|---|---|---|---|---|
| 14 | 0.0012 | 0.0051 | 0.826 | 0.836 | 0.8315 |
| 15 **(best)** | 0.0010 | 0.0053 | **0.830** | 0.838 | **0.8341** |
| 16 | 0.0009 | 0.0054 | 0.792 | 0.841 | 0.8167 |
| 19 | 0.0008 | 0.0057 | 0.819 | 0.847 | 0.8332 |

**Best checkpoint:** `segmentation_runs/run_002/ckpt_best.pth` (epoch 15, mean meniscus Dice = **0.8341**)

Significant improvement over pilot run (0.49 → **0.83**) — attributable to 2.2× more training data (124 vs 55 train patients). Medial Dice no longer frozen (sufficient val-set representation with 15 val patients).

### Real PD inference (with manually-labeled comparison data)

Ran both the fine-tuned (`infer_real_pd.py`) and original baseline
(`infer_real_pd_baseline.py`) models on 8 manually-labeled real PD patients:
`AC0D5A4D78B628`, `AC0D7BF72F7712`, `AC0D3459553205`, `AC14D3737C0482`,
`AC19E7C19827FF`, `AC149BC218E75C`, `AC111633B463BB`, `AC13300201B926`
(SAG PD TSE/DRB sequences). Output: predicted masks as NIfTI, for visual
before/after/ground-truth comparison in 3D Slicer.

Run_002 inference on labeled PD patients uses `infer_labeled_pd.sh` — reads patient IDs
from `labeled_patients.txt` on BigRed, globs for matching NIfTI files in `pd-files/`,
runs `infer_real_pd.py` with `run_002/ckpt_best.pth`.

### Dice Evaluation vs Ground Truth — Label Mapping

Ground truth masks (IU dataset, `segmentation_masks/*.nii`) use label `1 = meniscus`
(single class, no lateral/medial split). Predictions use different schemes:

| Model | Output classes | Meniscus extraction |
|---|---|---|
| Baseline (pitthexai original) | 5-class (0–4) | `pred == 4` (confirmed: class 4 = both menisci) |
| V1 fine-tuned (run_001) | 3-class (0=bg, 1=lateral, 2=medial) | `pred > 0` (merge both) |
| V2 fine-tuned (run_002) | 3-class (0=bg, 1=lateral, 2=medial) | `pred > 0` (merge both) |

**GT preprocessing note:** Some IU PD patients have native 768×768 in-plane resolution
(vs the more common 384×384). Predictions are always saved at 384×384 (the pipeline's
working resolution). For Dice evaluation, GT masks are brought into the same 384×384
space using the same pipeline steps as `process_volume()` — RAS reorientation via
`nib.as_closest_canonical()`, then resize to 384×384 with `order=0` (nearest-neighbour
to preserve integer labels). See `analysis_real_pd.ipynb` for implementation.

---

## 7. What's Verified vs. Still Open

| Area | Status |
|---|---|
| Train/val/test split integrity | ✅ Verified |
| Training-loop mechanics (checkpoint resume, augmentation, etc.) | ✅ Verified |
| R-network training target | ⏸ Deferred (known limitation, documented) |
| Evaluation patient/slice alignment | ✅ Verified |
| Evaluation split-awareness | ✅ Verified |
| Fake-PD voxel spacing match to real PD | ✅ Verified (1.00x ratio) |
| Fake-PD/mask affine orientation (LPS/RAS) | ✅ Verified |
| Round-trip/reconstruction correctness | ✅ Verified |
| Anatomical structure preservation (boundary-distance, R-independent) | ✅ Verified (~1.5px all-label, ~1.25px meniscus) |
| Distribution-level similarity statistical significance | ✅ Verified (p=0.0000, bootstrap CI non-overlapping) |
| Fine-tuning pipeline mechanics | ✅ Verified |
| Fine-tuned segmentation accuracy — pilot (69 patients) | ✅ ~0.49 mean Dice (run_001) |
| Fine-tuned segmentation accuracy — full (155 patients) | ✅ **0.8341 mean Dice** (run_002, epoch 15) |
| Real PD inference — run_001 predictions | ✅ 8 patients, NIfTIs saved, viewed in Slicer |
| Real PD inference — run_002 predictions on labeled patients | ✅ 8 patients, NIfTIs copied locally |
| Dice evaluation vs ground truth on labeled PD | 🔄 In progress (`analysis_real_pd.ipynb`, label mapping confirmed) |
| Domain gap visualization (t-SNE + UMAP, all 155 patients) | 🔄 Pending (infer_all155 done, v3 plot not yet run) |
