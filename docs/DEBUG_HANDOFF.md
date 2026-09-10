# RegGAN Meniscus Segmentation — Complete Debug Handoff

> **Purpose:** Full self-contained context for debugging why this project is stuck at ~0.78 Dice.
> Paste this into a fresh conversation. Assumes zero prior knowledge.
> **Author:** Anshika Bajpai (anbajpai@iu.edu) · **Advisor:** Rakesh Shiradkar
> **Last updated:** 2026-09-09

---

# 0. TL;DR — What I Need Debugged

I built a two-stage pipeline (GAN translation → segmentation fine-tuning) for meniscus segmentation on knee MRI. It works, but **it has been stuck at 0.78 Dice for 12+ experiments across two phases**. Every loss function, augmentation, architecture, and post-processing variant I tried landed within ±0.03 of 0.78.

In Phase 3 I finally instrumented the training run properly and found what I believe is the actual root cause. **I need help validating that diagnosis and deciding what to do about it.**

**The suspected root cause, in one sentence:** my synthetic training data contains *only* slices where the meniscus exists — zero negative examples — so the model was never taught to predict "nothing here," and it hallucinates meniscus on empty slices, which accounts for 46% of all false positives.

**Key numbers:**
- Train Dice 0.9456 vs val Dice 0.7793 → **generalization gap +0.166** (overfitting confirmed)
- Fake-PD val Dice 0.760 ≈ real-PD val Dice 0.781 → **domain gap is NOT the bottleneck**
- On slices that contain meniscus: mean Dice 0.79, median 0.844 → **near geometric ceiling**
- On slices with no meniscus: hallucinated on 9 of 32 → **2,001 voxels = 46% of all FP**
- FP is **2.6× FN**
- Oracle upside from fixing only the hallucination: **0.752 → 0.819 (+0.067)**

**Main caveat:** the diagnosis comes from only 2 validation patients / 62 slices. It needs confirming on the 17-patient cohort before I commit to a retrain.

---

# 1. Project Goal & Clinical Motivation

## The problem

Automated meniscus segmentation on **PD-weighted (proton density) knee MRI** is blocked by label scarcity. Expert voxel-level annotation takes hours of radiologist time per volume.

## The asymmetry I'm exploiting

| | DESS MRI | PD MRI |
|---|---|---|
| Public labels available? | **Yes** (SKM-TEA dataset) | No |
| Clinical standard? | No | **Yes** |
| Fluid appearance | Dark (suppressed) | Bright |

DESS has free public segmentation labels. PD is what clinicians actually acquire. If I can translate DESS→PD, I can use DESS masks to supervise a model that works on real PD.

## Why direct transfer fails

A DESS-trained segmentation model applied directly to real PD gives **Dice 0.00** — complete failure. The contrast is reversed (fluid dark on DESS, bright on PD), so the model's learned features are actively wrong.

## Why RegGAN and not CycleGAN

Standard CycleGAN transfers style but **distorts anatomy** while doing so:

| | CycleGAN | RegGAN |
|---|---|---|
| Deformation | >2px, uncontrolled | **0.055px at meniscus** |
| Topology violations (folding) | Yes | **0%** |
| Anatomy preservation | Poor | Excellent |
| Extra component | None | Registration network R |

**40× difference in deformation.** For medical imaging this is disqualifying — if the anatomy moves during translation, the transferred masks no longer align with the image, and the whole supervision signal is corrupted.

---

# 2. Datasets — Complete Inventory

## 2.1 Source and target imaging data

| | DESS (SKM-TEA) | PD (IU institutional) |
|---|---|---|
| Local path | `data/skm-tea-dataset/dess-files` | `~/Desktop/AImed-lab/IU-Dess-dataset/iu-dataset/pd-files` |
| BigRed path | `/N/project/prostate_cancer_ai/anshika/regGAN/data/dess-files` | `/N/.../data/iu-dataset/pd-files` |
| Volumes | 69 (also have 155-patient version) | 69 |
| Raw shape | (512, 512, 160) | (384, 384, 36) |
| Voxel spacing | 0.31 × 0.31 × 0.80 mm | 0.39 × 0.39 × 3.60 mm |
| Orientation | P,S,R (sagittal) | P,S,R (sagittal) |
| Slices extracted | ~11,040 | ~2,271 |
| Masks | `.seg.nrrd`, 6 tissue classes | **None** |
| Meniscus labels | 5 = lateral, 6 = medial | N/A |

**Total slices processed: 13,311**

⚠️ Some IU PD patients have native 768×768 in-plane resolution instead of 384×384. Handled by resampling, but worth knowing.

## 2.2 Patient cohorts

### D1 — Original hold-out eval (10 patients) — NEVER used in training
File: `labeled_patients.txt`
```
AC0D5A4D78B628  AC0D7BF72F7712  AC0D3459553205  AC14D3737C0482  AC19E7C19827FF
AC149BC218E75C  AC111633B463BB  AC13300201B926  AC12026D14291F  AC13637DA25399
```
GT masks: `~/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final/` (`.seg.nrrd`)

### D1+New — Extended hold-out (17 patients) — current eval cohort
File: `labeled_patients_v8.txt` — D1 plus 7 new:
```
AC000550763509  AC005135D3495B  AC04433E37DB66  AC056ADCE8BE28
AC07607B9E5295  AC0D1A9818A6FC  AC0F11041F5180
```
GT masks for the new 7: `~/Downloads/final-segmentations/`

⚠️ **Known landmine:** the path is `final-segmentations`, NOT `segmentations`. Using the wrong one silently skips 7 patients and inflates results. This bug bit me once — I caught it, fixed it, and re-ran everything with the lower correct numbers.

### D2 — Real PD fine-tuning cohort

| Version | Patients | Train | Val |
|---|---|---|---|
| `real_pd_seg_data` (v4, v5) | 10 | 8 | AC2B0AA9AE767D, AC2E254F52E467 |
| `real_pd_seg_data_v7` (v7+) | 15 | 13 | AC0CE315D5758B, AC0CEE9C24F2B7 |

5 masks added for v7: AC0BA0AE159EF9, AC0CE315D5758B, AC0CEE9C24F2B7, AC045F8F6ACBA7, AC0407F05FAF53

### Torn meniscus cohort (20 patients) — no GT masks
BigRed: `/N/.../data/iu-control/pd-files`

⚠️ **The folder is named `iu-control` but these are actual TORN meniscus patients** annotated by doctors (s83_2 dataset). The name is misleading. No local GT masks, so no quantitative Dice available for this cohort yet.

## 2.3 Approximately 54 unlabeled real PD patients

69 total PD volumes minus 15 used for fine-tuning = ~54 unlabeled. These are currently unused and are the basis for the self-training idea (see §9).

---

# 3. Phase 0 — RegGAN Translation Model

## 3.1 Architecture

### Generators G_AB, G_BA (ResNet-based)
```
Input: (B, 1, 384, 384) grayscale MRI slice

Encoder:
  ReflectionPad2d(3)
  Conv2d(1, ngf, 7) → InstanceNorm2d → ReLU
  Conv2d(ngf, ngf*2, 3, stride=2) → InstanceNorm2d → ReLU
  Conv2d(ngf*2, ngf*4, 3, stride=2) → InstanceNorm2d → ReLU

Residual: 9 × ResBlock(ngf*4)
  ReflectionPad → Conv → InstanceNorm → ReLU → Conv → InstanceNorm + skip

Decoder:
  UpsampleConv(ngf*4, ngf*2)   # bilinear upsample + Conv
  UpsampleConv(ngf*2, ngf)
  ReflectionPad2d(3)
  Conv2d(ngf, 1, 7)
  Tanh()

Output: (B, 1, 384, 384), range [-1, 1]
ngf = 48, n_res = 9
```

### Discriminators D_A, D_B (PatchGAN 70×70)
```
ndf = 48, 4 conv layers stride-2
InstanceNorm2d + LeakyReLU(0.2)
Final: Conv2d(ndf*8, 1), no sigmoid (LSGAN)
```

### Registration Network R (VoxelMorph-lite 2D)
```
Input:  (B, 2, 384, 384) = concat(fake_B, real_B)
Output: (B, 2, 384, 384) = displacement field (Δx, Δy) in pixels

U-Net: 3 conv blocks with AvgPool2d(2) encoder
       bilinear upsample + skip connections decoder
       Flow head: Conv2d(nf, 2, 3), near-zero weight init
nf = 16   # intentionally small — less capacity = less aggressive warping

Warping: differentiable bilinear grid_sample
  grid = meshgrid(-1,1) + normalized_flow
  output = F.grid_sample(img, grid)
```

## 3.2 Design decisions and why

| Decision | Rationale |
|---|---|
| **InstanceNorm over BatchNorm** | BatchNorm normalizes across the batch — unstable for unpaired training with small batches and high image variability. InstanceNorm normalizes per-image per-channel. |
| **ReflectionPad over ZeroPad** | Avoids border artifacts at image edges. |
| **UpsampleConv over ConvTranspose2d** | Eliminates checkerboard artifacts. |
| **PatchGAN over full-image discriminator** | Classifies overlapping local patches — better for texture/style transfer, encourages sharp local texture matching PD contrast rather than global statistics. |
| **LSGAN (MSE) over BCE** | More stable gradients, avoids vanishing-gradient problem. |
| **nf=16 for R (deliberately small)** | Limited capacity acts as an extra regularizer against aggressive warping. |
| **Near-zero flow head init** | Biases the model toward small deformations from step 1. |

## 3.3 Loss functions

```
L_G = L_GAN_AB + L_GAN_BA                        # fool discriminators
    + λ_cycle × (L_cycle_A + L_cycle_B)          # cycle consistency
    + λ_cycle × 0.5 × (L_idt_A + L_idt_B)        # identity
    + λ_reg_sim × L_reg_sim                      # warped fake_B ≈ real_B
    + λ_smooth × L_smooth                        # smooth deformation field
    + λ_mag × L_mag                              # small deformation magnitude

L_D = 0.5 × (MSE(D(real), 1) + MSE(D(fake), 0))  # LSGAN
```

**The minimum-deformation losses are the core contribution:**
```python
L_smooth = mean(dx² + dy²)   # dx, dy = spatial gradients of flow
L_mag    = mean(flow²)       # direct displacement penalty
```

**Loss weights:**
```
λ_cycle      = 10.0
λ_reg_sim    = 5.0
λ_reg_smooth = 10.0
λ_reg_mag    = 5.0
```

**⚠️ Note — every RegGAN loss is computed over the WHOLE IMAGE.** There is no
anatomy-aware or mask-restricted term anywhere in the translation stage:
`L_cycle`, `L_idt`, `L_reg_sim`, `L_smooth` and `L_mag` are all uniform pixel-wise
or flow-wise penalties. `dataset.py` never loads a segmentation mask at all — masks
enter the project only in the segmentation stage.

This means the "meniscus deformation 0.055px" number is a *measurement* taken after
the fact, not something the loss ever explicitly optimized. Deformation at the
meniscus is small because deformation is small **everywhere**, not because the
meniscus was protected. Two unimplemented ideas that would change this are in §9
(rows 13–14).

## 3.4 Training config

| Parameter | Value | Reason |
|---|---|---|
| Optimizer | Adam β=(0.5, 0.999) | Standard for GANs |
| LR (G, D) | 2e-4 | Standard CycleGAN LR |
| LR (R) | 1e-4 | Lower — R should learn slowly |
| Batch size | 8 | A100 40GB limit |
| Epochs | 200 planned, ~4 run | HPC time limit |
| LR schedule | Constant first half, linear decay second | Standard GAN schedule |
| Image pool | 50 | Discriminator stability |
| Gradient clip | 5.0 (generators only) | Prevent exploding gradients |
| Hardware | NVIDIA A100 40GB, IU BigRed200 | |

**⚠️ Critical training step order:**
```
1. Forward pass → all outputs
2. Update G (freeze D) → backward G loss
3. Update R (after G) → backward R loss
4. Update D (unfreeze) → backward D loss
```
`R.step()` mutates R weights. If R runs before G's backward, the computation graph version mismatches → PyTorch inplace operation error. **G must backward before R.step().**

## 3.5 RegGAN results

Canonical source: `inference/eval2/metrics.json`

| Metric | Value | Interpretation |
|---|---|---|
| FID (fake PD vs real PD) | **164.4** | |
| FID (DESS vs real PD) | **260.4** | baseline |
| FID improvement | **37%** | ✅ |
| KID | 0.020 ± 0.000 | Low = distributions similar ✅ |
| SSIM (DESS vs fake PD) | 0.357 | Expected — see below |
| Jacobian det mean | **1.000056** | Near-perfectly rigid ✅ |
| Jacobian det min | 0.74 | Boundary artifact, not real |
| Jacobian folding % | **0.0%** | Zero topology violations ✅ |
| Global deformation | 0.029 px | Sub-pixel ✅ |
| Meniscus deformation | **0.055 px** | Sub-pixel at key anatomy ✅ |

**Why low SSIM (0.357) is correct, not a failure:** DESS fluid is dark, PD fluid is bright. High SSIM would mean the generator did nothing. 0.357 means contrast changed substantially (good) while structure is preserved (confirmed independently by the Jacobian analysis).

**FID caveat:** using 64×64 pixel features, not InceptionV3, because BigRed has no internet. Relative comparison is valid; absolute values differ from standard FID.

**Independent structural check** (`boundary_distance_eval.py`, R-independent):

| Label | Mean (px) |
|---|---|
| 5 (lateral meniscus) | 1.293 |
| 6 (medial meniscus) | 1.194 |
| **Meniscus (5+6)** | **1.250** |
| All labels | 1.523 |

Sub-2-pixel mean boundary distance across all tissue labels (~0.5mm physical). Strong independent evidence of anatomical preservation.

## 3.6 RegGAN model versions

| Name | BigRed path | RegGAN trained on | Patients **translated** | Epochs | FID | Used for segmentation? |
|---|---|---|---|---|---|---|
| **Old RegGAN** | `segmentation_data_v2/` | ~69 | **155** | ? | 164.4 | **Yes — all seg runs** |
| run006 ep14 | `fake_pd_run006_ep14/` | 155 | 155 | 14 | 174.4 | Tested, worse |
| run006 ep52 | `fake_pd_run006_ep52/` | 155 | 155 | 52 | **147.6** | Tested, worse downstream |

> ⚠️ **`segmentation_data_v2/` contains 155 patients, NOT 69.** The source docs conflict on this and the "~69" figure is a conflation — it is how many DESS patients the *RegGAN generator was trained on*, not how many were translated into the segmentation training set.
>
> **Verified from the SLURM scripts** (see §17.1):
> ```
> FAKE_PD_DIR = results/fake_pd_all155      ← 155-patient inference output
> MASK_DIR    = preprocessed_v3/masks       ← preprocessed_v3 = the 155-patient run
> SPLITS      = preprocessed_v3/splits.json
> ```
> `experiment_version_tracking.md` and `PROJECT_HANDOFF.md` say "~69" and are **wrong** in this respect. `TECHNICAL_DESIGN.md` §6 says 155 and is **correct**.
>
> **This matters for Experiment 1** (§9): regenerating the fake-PD set with negative slices means regenerating a **155-patient** dataset from `fake_pd_all155` + `preprocessed_v3/masks`, not a 69-patient one.

**⚠️ Counterintuitive and important:** the *best FID* model (ep52, FID 147.6) produced **worse** downstream segmentation Dice (0.640) than the old RegGAN (FID 164.4 → Dice 0.695). Better image realism did not translate to better segmentation. This is early evidence that translation quality is not the bottleneck — later confirmed in Phase 3.

## 3.7 Known unresolved issue with R

**Root cause identified but deferred:** R is trained on `(fake_B, real_B)` where `real_B` is a *random unrelated patient's* PD slice — there is no consistent anatomical relationship across batches. At eval time, the code calls `R_net(fake_PD, DESS)`, a pairing R was never trained on.

**Attempted fix (reverted):** retargeted R to `(fake_B, real_A)` (the true DESS source) and dropped the photometric `l_reg_sim` loss due to the contrast gap. Result: **R collapsed to exactly zero flow** — removing `l_reg_sim` left only `l_smooth` and `l_mag`, both uniquely minimized at `flow ≡ 0`, so nothing rewarded non-zero output. Fully reverted.

**Current state:** original R logic retained. It does not collapse to literal zero (~0.0226px mean flow on out-of-distribution pairs) — not ideal, but not degenerate. Proposed but unimplemented fix: edge-magnitude similarity loss.

---

# 4. Phase 1 — Segmentation Model

## 4.1 Base model

- `pretrained/baseline_best_model.pth`
- Source: [pitthexai/Knee_MRI_Segmentation_2.5D](https://github.com/pitthexai/Knee_MRI_Segmentation_2.5D)
- `segmentation_models_pytorch` U-Net, **ResNet34 encoder**, pretrained on DESS, 5-class output
- **Applied directly to real PD → Dice 0.00** (complete failure, as expected)

## 4.2 Architecture for fine-tuning

```python
smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=5)
# load pretrained 5-class weights
# then replace head: 5 → 3 classes (background, lateral meniscus, medial meniscus)

Input: 3-slice 2.5D stack [i-1, i, i+1], shape (3, 384, 384)
```

**2.5D, not 3D.** Three consecutive slices are stacked as *channels* through a 2D U-Net. This matters for the professor's shape-prior suggestion (see §10) — there is no volumetric convolution, so a 3D shape prior is not directly applicable.

## 4.3 Loss functions

**Fake PD branch:**
```python
CrossEntropyLoss(weight=[0.1, 1.5, 1.5]) + SoftDiceLoss()
```

**Real PD branch (`MergedLoss`):**
```python
# Real PD GT is binary (meniscus/not), model outputs 3 classes
# So lateral + medial softmax probs are merged before comparison
probs  = softmax(logits)
p_men  = probs[:, 1] + probs[:, 2]
gt_men = (masks > 0)
ce   = -(0.1 * gt_bg * log(p_bg) + 1.5 * gt_men * log(p_men)).mean()
dice = 1 - (2*TP + eps) / (p_men.sum() + gt_men.sum() + eps)
loss = ce + dice
```

## 4.4 Hyperparameters (the "v7_mixed recipe")

| Parameter | Value |
|---|---|
| LR | 1e-5 → 1e-8, CosineAnnealingLR, T_max=50 |
| Epochs | 50, early stopping patience=10 |
| Batch size | 8 |
| Class weights | bg=0.1, lateral=1.5, medial=1.5 |
| Augmentation | h/v flip (p=0.5), brightness ±20%, Gaussian noise σ=0.02 |
| Starting ckpt | `pretrained/baseline_best_model.pth` (always) |

## 4.5 Training loop structure

```python
for fake_batch in fake_loader:          # fake PD is the DRIVING loop
    loss_f = fake_loss_fn(model(x_fake), y_fake)
    real_batch = next(real_iter)        # real PD cycles, refilled on StopIteration
    loss_r = merged_loss_fn(model(x_real), y_real)
    loss = loss_f + loss_r
    ...
```

**⚠️ This structure matters for the diagnosis.** Fake PD drives the loop. Real PD is the minority stream, cycled repeatedly. Whatever bias exists in the fake PD dataset dominates training.

---

# 5. Complete Experiment Log

## 5.1 Fake PD only (no real PD labels)

| Run | Fake PD source | Start ckpt | Eval | Dice |
|---|---|---|---|---|
| run_001 | 55pt DESS | baseline | D1 8pt | 0.429 |
| run_002 | segmentation_data_v2 | baseline | D1 10pt | 0.632 |
| run_003 | segmentation_data_v2 | baseline | D1 10pt | 0.650 |
| **run_002_v2** | segmentation_data_v2 | baseline | D1 10pt | **0.695** ← best fake-only |
| run_005 | fake_pd_run006_ep52 | baseline | D1 10pt | 0.640 |
| run_ep14_v2 | fake_pd_run006_ep14 | baseline | D1 10pt | 0.603 |

**What made run_002_v2 better than run_002 (+5.4%):**

| Change | Why it helped |
|---|---|
| Weighted CE (bg=0.1, lat/med=1.5) | Background dominates pixels — model was ignoring the tiny meniscus |
| Augmentation (hflip, vflip, brightness, noise) | Fake PD ≠ real PD in appearance; more aug = better cross-domain generalization |
| Cosine LR (1e-5 → 1e-7) | Fixed LR was jumping near convergence |
| 50 epochs vs 20 | More time to converge, early stopping prevents overfit |

## 5.2 v3 ablations — all fake PD only, all from DESS baseline

Reference to beat: run_002_v2 = 0.685 (on the 8pt subset)

| Run | Change | Dice (8pt) | Δ |
|---|---|---|---|
| run_v3_photometric | + gamma/blur/contrast aug | 0.681 | −0.004 |
| run_v3_rotation | + rotation ±15° | 0.676 | −0.009 |
| run_v3_tversky | Tversky loss α=0.3, β=0.7 | 0.656 | −0.029 |
| run_v3_elastic | + elastic deformation | 0.652 | −0.033 |
| run_v3_resnet50 | ResNet50 encoder | 0.639 | −0.046 |
| run_v3_5slice | 5-slice 2.5D context | 0.486 | −0.199 |

**Every single one hurt.** Note especially `resnet50` at 0.639 — a *bigger* encoder made things *worse*. Phase 3 retroactively explains this: the model is overfitting, so more capacity memorizes faster.

**⚠️ Caveat on all v3 results:** these were run on fake PD only, which is the weaker setup. They are not a fair test of whether these ideas help in the mixed (real+fake) regime. That's why I re-ran Tversky and photometric as v12 — see §5.6.

## 5.3 v4 / v5 — introducing real PD

| Run | Start ckpt | Real PD | Fake PD | Dice (8pt) |
|---|---|---|---|---|
| run_v4_realPD | run_002_v2 | 10pt | — | 0.775 |
| run_v4_mixed | run_002_v2 | 10pt | ✓ | 0.759 |
| run_v5_realPD | **baseline** | 10pt | — | **0.135** ← collapsed |
| run_v5_mixed | **baseline** | 10pt | ✓ | 0.765 |

**The single most informative result in the whole project:** v4_realPD (0.775) vs v5_realPD (0.135). *Identical training data*, only the starting checkpoint differs. Starting from a fake-PD-pretrained model works; starting from the DESS baseline with only 10 real patients collapses entirely.

**Conclusion:** fake PD is essential as a warm-start signal. Real PD alone is not enough data to adapt from DESS.

## 5.4 v7 — more real PD patients

| Run | Real PD | Fake PD | Dice (17pt) |
|---|---|---|---|
| run_v7_realPD | 15pt | — | 0.774 |
| **run_v7_mixed** | 15pt | ✓ | **0.781** ← BEST OVERALL |

10 → 15 real patients gave +0.016. `run_v7_mixed/ckpt_best.pth` is the checkpoint used for everything downstream.

## 5.5 Phase 2a — inference-only experiments (no retraining)

All use `run_v7_mixed/ckpt_best.pth`. Eval on 17pt.

| Label | Change | Dice | Δ |
|---|---|---|---|
| v8_mixed | baseline inference | 0.781 | — |
| **v8_clipped (Exp A)** | Input clipped to 95th percentile before inference — suppresses bright outlier pixels/coil artifacts | **0.784** | +0.003 |
| v8_filled (Exp B) | Post-proc: `binary_fill_holes` + `remove_small_objects` | 0.778 | −0.003 |

## 5.6 Phase 2b — retraining with modified losses

All from DESS baseline, `real_pd_seg_data_v7` + `segmentation_data_v2`, same hyperparams as v7_mixed.

| Run | Change | Dice | Δ | Win count |
|---|---|---|---|---|
| run_v9_mixed | Segmentation-restricted CE (`ignore_index=0`, background excluded) | 0.753 | −0.028 | 0/17 |
| run_v10_boundary | + boundary-aware loss (λ=1.0), CE penalty at 3×3 erosion ring | 0.753 | −0.028 | — |
| run_v11_cldice | + clDice topology loss (λ=1.0, iters=10), differentiable soft skeleton | 0.777 | −0.004 | **7/17** |

**v9 and v10 landing on *exactly* 0.753 is not a coincidence** — see §7.

**v11_cldice is interesting:** it wins the most individual patients (7 of 17) despite a marginally lower mean. The aggregate is hiding per-case improvement.

### Per-patient breakdown (17pt cohort)

| Patient | Cohort | v8_mixed | v8_filled | v8_clipped | v9_mixed | v10_boundary | v11_cldice |
|---|---|---|---|---|---|---|---|
| AC0D5A4D78B628 | D1 | 0.709 | 0.703 | 0.713 | 0.692 | 0.698 | **0.730** |
| AC0D7BF72F7712 | D1 | 0.777 | 0.767 | **0.781** | 0.721 | 0.735 | 0.779 |
| AC0D3459553205 | D1 | 0.725 | 0.723 | 0.737 | 0.733 | 0.705 | **0.756** |
| AC14D3737C0482 | D1 | **0.791** | 0.771 | 0.786 | 0.745 | 0.740 | 0.785 |
| AC19E7C19827FF | D1 | 0.802 | 0.798 | 0.806 | 0.788 | **0.809** | 0.779 |
| AC149BC218E75C | D1 | 0.801 | 0.791 | **0.803** | 0.788 | 0.777 | 0.797 |
| AC111633B463BB | D1 | 0.809 | 0.805 | 0.811 | 0.771 | 0.787 | **0.822** |
| AC13300201B926 | D1 | 0.763 | 0.755 | 0.764 | 0.702 | 0.733 | **0.770** |
| AC12026D14291F | D1 | **0.786** | 0.784 | 0.783 | 0.728 | 0.722 | 0.767 |
| AC13637DA25399 | D1 | 0.808 | 0.808 | 0.813 | 0.761 | 0.797 | **0.815** |
| AC000550763509 | new | 0.809 | 0.811 | 0.811 | 0.791 | 0.782 | **0.838** |
| AC005135D3495B | new | 0.822 | 0.824 | 0.826 | 0.825 | **0.851** | 0.849 |
| AC04433E37DB66 | new | 0.728 | **0.733** | 0.728 | 0.688 | 0.729 | 0.730 |
| AC056ADCE8BE28 | new | 0.685 | 0.690 | **0.693** | 0.665 | 0.633 | 0.655 |
| AC07607B9E5295 | new | 0.843 | **0.845** | 0.840 | 0.828 | 0.680 ⚠️ | 0.773 |
| AC0D1A9818A6FC | new | 0.817 | 0.814 | **0.824** | 0.802 | 0.810 | 0.782 |
| AC0F11041F5180 | new | 0.806 | 0.800 | 0.804 | 0.768 | **0.812** | 0.779 |
| **MEAN** | | **0.7812** | 0.7776 | **0.7836** | 0.7527 | 0.7529 | 0.7769 |

⚠️ **AC07607B9E5295 with v10_boundary: 0.843 → 0.680 (−0.163).** The boundary loss catastrophically hurt this one patient. Worth investigating what's anatomically unusual about it.

**Spread across patients is ~0.69 to 0.84** — a range of 0.15, much larger than any experimental effect I've measured.

## 5.7 Phase 2c — v12 fair re-ablations (submitted, results pending)

Re-running the two most promising v3 ablations on the *correct* v7_mixed recipe:

| Run | Change | Status | v3 equivalent |
|---|---|---|---|
| run_v12_tversky | Tversky α=0.3, β=0.7 on fake PD branch | Inference done, eval pending | v3_tversky 0.656 (unfair setup) |
| run_v12_photometric | Photometric aug (gamma 0.7–1.5, blur σ 0.3–1.2, contrast stretch) on fake PD | Inference done, eval pending | v3_photometric 0.681 (unfair setup) |

**Rationale:** v3 tested these on fake-PD-only, which caps out at 0.695. Testing them on the mixed setup (caps at 0.781) is the fair comparison.

---

# 6. Phase 3 — The Actual Diagnosis

This is the most important section. **After 11 experiments all landing within ±0.03, I stopped guessing and instrumented the training run.**

Source: `segmentation_runs/run_v7_replica_seed42/`, seed 42, original v7/v8 loss, with train Dice logged **for the first time**.

⚠️ Train Dice had been computed as `td` in `run_epoch` and then **silently discarded inside an f-string** in every previous run. That's why the overfitting was invisible for the entire project.

## 6.1 Overfitting — CONFIRMED

| Signal | Value |
|---|---|
| Train Dice (real PD) | **0.9456** |
| Val Dice (real PD) | **0.7793** |
| **Generalization gap** | **+0.166** |
| Train loss | 0.263 → 0.209 (falling) |
| Val loss | 1.016 → 1.017 (flat from epoch ~4) |

**Not underfitting. Not at ceiling.** The model has ample capacity — it hits 0.946 on data it has seen. Val loss goes flat at epoch 4 while train loss keeps falling.

This retroactively explains `v3_resnet50` (0.639): a bigger encoder memorizes faster.

## 6.2 Domain gap is SMALL — RegGAN is not the bottleneck

```
fake-PD val Dice = 0.760
real-PD val Dice = 0.781
```

The model generalizes about **equally poorly** to unseen synthetic and unseen real data. If translation realism were the binding constraint, held-out fake PD would score much higher than held-out real PD. It doesn't.

**Consistent with the earlier anomaly:** run_005 (best FID, 147.6) gave *worse* Dice (0.640) than old RegGAN (FID 164.4 → 0.695).

**Implication: improving the GAN will not help.** This argues strongly against the diffusion-model direction.

## 6.3 It is a DETECTION problem, not a SEGMENTATION problem

From `val_predictions/per_slice_dice.csv` — 2 val patients, 62 slices:

```
Slices with NO meniscus in GT : 32
  model correctly silent      : 23
  model HALLUCINATED          :  9  → 2,001 voxels = 46% of ALL false positives
  worst single slice          : 538 voxels invented where there is nothing

Slices that DO contain meniscus: 30
  mean Dice                   : 0.7919
  median Dice                 : 0.844
  slices below 0.5            : 2 of 30

FN total 1,682   FP total 4,373   →  FP is 2.6× FN
```

**On slices where the meniscus exists, the model is already near the geometric ceiling.** For a ~6px-thick structure, a 1px boundary offset caps Dice at roughly 0.83. Median is 0.844.

**The Dice is being lost on slices that should be empty.**

Oracle calculation — removing only the empty-slice hallucinations:
```
TP = 9199   GT_total = 10881   PRED_total = 13572   halluc = 2001

current : 2×9199 / (13572 + 10881) = 0.7524
oracle  : 2×9199 / (11571 + 10881) = 0.8194     (+0.0671)
```

**+0.067 from one fix — larger than every experiment in Phase 2 combined.**

> ⚠️ **This 0.7524 is a different aggregation from the 0.7793 in §6.1 and the 0.781 in §5.** See §17.2 — three different Dice aggregations are in use across this document. All numbers in this section were recomputed from the source CSV on 2026-09-09 and verified; see §17.1.

> ⚠️ **Trap when recomputing:** in `per_slice_dice.csv`, a slice with `gt_voxels=0` and `pred_voxels=0` is recorded as `dice=1.0000`. A naive mean over all 62 rows is therefore inflated by the 23 correctly-silent slices. The 0.7919 figure above correctly averages over meniscus-bearing slices only.

## 6.4 Checkpoint selection is arbitrary / measurement noise floor

- Only **2 validation patients**; the plateau spans just 0.755–0.781 across 17 epochs
- Two metrics correlating **+0.967** still selected epochs **10 and 19**
- Pooled `vd` runs ~0.028 above the clean per-patient mean — cycling-iterator artifact in `run_epoch`, where the small real-val set is traversed repeatedly and weights slices unevenly
- Seed noise: replica peaked 0.7809 @ ep10; original v7_mixed peaked 0.7644 @ ep26

**Implication: every Phase-2 difference smaller than ~0.02 was unmeasurable.** That includes v8_clipped (0.784), v8_filled (0.778), and v11_cldice (0.777) versus 0.781. **I have been reading noise as signal.**

### Would selecting on fake-PD val fix this? No — tested.

The fake-PD val set has many more patients than the real-PD val set's 2, so it is
the obvious candidate for a lower-variance selection signal. Measured on the
replica run's plateau (epochs ≥ 4):

```
corr(fake_va, real_va)          : +0.577      moderate only
corr(real_va, clean_perpatient) : +0.967      (both measure real PD)

Which epoch does each criterion pick, and what real-PD Dice do you GET?
  real-PD val (current)  -> epoch 10   real-PD Dice = 0.7809
  fake-PD val            -> epoch 13   real-PD Dice = 0.7657    <- costs 0.015
  clean per-patient      -> epoch 19   real-PD Dice = 0.7800

Stability on the plateau (std):
  real-PD val 0.0071  |  fake-PD val 0.0048  |  clean per-patient 0.0082
```

**Verdict:** fake-PD val is genuinely steadier (~30% lower std — the larger-sample
benefit is real), but it correlates only **+0.577** with real-PD performance.
Switching to it would have selected epoch 13 and handed back a model **0.015 worse**
on real PD. More data, wrong domain — here the domain error exceeds the variance win.

**The deeper point:** `real-PD val` and `clean per-patient` correlate **+0.967** —
they measure the same 2 patients and agree almost perfectly on ranking — yet they
pick epochs **10 and 19**. When the plateau spans only 0.026 across 17 epochs,
*every* criterion is choosing close to arbitrarily. Switching signals is not the
fix; the fix is k-fold (§9 row 4) or weight averaging (§9 row 5).

---

# 7. Root Cause of the Hallucination

`segmentation/prepare_meniscus_masks.py:129`

```python
if (collapsed > 0).sum() < args.min_meniscus_pixels:   # default 10
    total_skipped_empty += 1
    continue
```

**The fake-PD training set contains only meniscus-bearing slices — ZERO negatives.**

Fake PD is the stream driving `run_epoch`'s loop (see §4.5), so for the bulk of training the model is **never shown a slice whose correct answer is "nothing here."**

Real PD prep (`prepare_real_pd_labeled.py:151`) filters on *image* emptiness (`img_sl.mean() < 0.02`), not mask content — so it *does* supply negatives, but only from 13 patients in the minority stream.

## Why this also explains v9 and v10

Both landed on **exactly 0.753**. Both further reduced background supervision on a training set that already had almost none:

- **v9** — `ignore_index=0` removed background from CE entirely
- **v10** — boundary loss concentrated learning on the meniscus rim

`future_experiments.md` predicted this risk for v9 *before it ran*:
> *"No CE pressure to suppress false positives in background — only Dice catches over-prediction."*

That prediction was correct and I ran it anyway.

---

# 8. Current Open Problems

| # | Problem | Evidence | Status |
|---|---|---|---|
| 1 | **Model hallucinates on empty slices** | 46% of all FP, +0.067 oracle upside | Root cause found, fix not implemented |
| 2 | **Overfitting, gap +0.166** | train 0.946 vs val 0.779 | Confirmed, not addressed |
| 3 | **Val set too small (2 patients)** | Noise floor ~0.02, checkpoint selection arbitrary | k-fold proposed, not run |
| 4 | **Pooled val Dice biased ~+0.028 high** | Cycling-iterator artifact in `run_epoch` | Identified, not fixed |
| 5 | **All Phase 2 conclusions may be noise** | Effects < 0.02 are unmeasurable | Needs re-evaluation with CI |
| 6 | **No GT for torn cohort** | 20 patients, no masks | Blocks the clinically interesting evaluation |
| 7 | **R network trained on mismatched pairs** | See §3.7 | Deferred, documented |
| 8 | **Diagnosis based on 2 patients only** | 62 slices | Must confirm on 17pt before retraining |

## Observed failure modes on the torn cohort (qualitative, no GT)

**Failure Mode 1 — Split prediction:** the model splits the meniscus into two disconnected halves at the tear site. Bright fluid in the tear is read as a tissue boundary.

**Failure Mode 2 — Missed detection:** on some slices the model misses the meniscus entirely, or catches only a small fragment. Happens where tear signal is most severe.

Both are expected — the model trained exclusively on healthy knees.

---

# 9. Candidate Experiments, Ranked

Sources tagged: **[Claude]** = surfaced during Phase 3 diagnostics, **[prof]** = advisor suggestion.

| # | Experiment | Source | Targets | Effort | Evidence |
|---|---|---|---|---|---|
| 1 | **Include meniscus-free slices in fake-PD prep** | **[Claude]** | §6.3 hallucination | Minimal (a flag) | **Strong — +0.067 oracle** |
| 2 | **Image crop → train on cropped only** | [prof] | §6.1 overfit, in-plane FP | Medium | Good |
| 3 | **Self-training on ~54 unlabeled real PD** | [Claude] | §6.1 overfit | Medium | Good |
| 4 | **k-fold over the 15 real PD patients** | [Claude] | §6.4 selection noise | Low | Strong |
| 5 | **Weight averaging over last N epochs** | [Claude] | §6.4 selection noise | Minimal | Moderate |
| 6 | Knowledge distillation | [prof] | — | Medium | Weak — not capacity-limited |
| 7 | Contrast change / basic image processing | [prof] | inter-patient variation | Low | Weak — v8_clipped gave +0.003 |
| 8 | Meniscus shape-based training | [prof] | implausible blobs | Medium | Weak — v11_cldice ≈ no change |
| 9 | Image contour approach | [prof] | boundary precision | Medium | **Against** — already near ceiling |
| 10 | Diffusion instead of fine-tuning | [prof] | image realism | **Very high** | **Against** — see §6.2 |
| 11 | FMed paper | [prof] | — | — | Unassessed |
| 12 | Knee cartilage segmentation review | [prof] | — | Low | Orientation reading |
| 13 | **Rigid / non-rigid flow decomposition in RegGAN** | **[own — Stage 9]** | anatomy preservation | High | Unimplemented — **novel** |
| 14 | **Statistical SDM (signed distance map) constraint** | **[own — Stage 10]** | shape change | High | Unimplemented — **novel** |

## Rows 13–14 — my own unimplemented novel ideas

Both are already specified in `docs/PIPELINE_VALIDATION_PLAN.md` (Stages 9 and 10),
written during pipeline validation and never run. They are the answer to "I can't
think of anything novel" — **I already did, and then forgot about them.**

Both address the gap flagged in §3.3: RegGAN's deformation penalties are uniform
over the whole image and know nothing about anatomy.

**Stage 9 — rigid/non-rigid decomposition**
```
flow          = flow_rigid + flow_nonrigid
flow_rigid    = least-squares best-fit translation + rotation over the meniscus region
L_nonrigid    = E[||flow_nonrigid(x,y)||²]        # replaces the uniform L_mag
```
*Why novel:* existing methods use rigid alignment as *preprocessing*, or apply
uniform Jacobian/smoothness/incompressibility constraints. None decompose a
*learned* field into rigid vs non-rigid inside the training loop and penalize only
the harmful part. Rigid repositioning is harmless; shape distortion is not.

**Stage 10 — signed distance map constraint**
```
SDF_A      = compute_sdf(mask_A)
SDF_warped = warp(SDF_A, flow)
L_sdm      = mean((SDF_warped - SDF_A)²)
```
*Why novel:* directly constrains anatomical *boundary* change rather than
penalizing all displacement equally. More principled than raw magnitude/smoothness.

**⚠️ Honest caveat given the Phase 3 diagnosis:** §6.2 shows the domain gap is
small and translation quality is *not* the current bottleneck. So these two are
**scientifically the most novel things on this list, but not the highest expected
Dice gain.** They are the right direction for a methods paper; rows 1–5 are the
right direction for the number. Do not confuse the two.

## Experiment 1 — full detail

**Change:** re-run `prepare_meniscus_masks.py` with `--min_meniscus_pixels 0`, or better, keep a *controlled fraction* of negative slices (all meniscus-bearing slices + N empty slices per volume) to avoid swamping the positives.

**Why:** the model currently has no training signal for "predict nothing here."

**Risk:** too many negatives re-introduces the class-imbalance problem that the weighted CE (bg=0.1) was built to solve. **The negative fraction is a hyperparameter, not a free win** — worth a sweep over 0%, 25%, 50%, 100% of available empty slices.

**⚠️ Implementation note:** this changes `segmentation_data_v2/`, which every previous run used. It needs a **new data directory** (e.g. `segmentation_data_v3_withneg/`), not an overwrite.

---

# 10. Professor's Feedback and My Responses

## Feedback received

> *"What is the input to your model — entire field of view or a cropped view? Would a 2-step approach help — first you localize the meniscus (bounding box) and then you segment."*

> *"You can try to relax your objective function so that it starts to over-segment rather than under-segment."*

> *"Is this a 3D model or 2D? If it is 3D, we could include a shape prior."*

## My responses and the tension with the diagnosis

**On input/localization:** Currently the full 384×384 FOV, no crop. Two-step localize-then-segment is ranked #2 in §9 and targets the overfitting directly. **Still worth doing.**

**On relaxing the objective to over-segment:** This is what Tversky (α=0.3, β=0.7) does — it penalizes FN 2.3× more than FP. **But the diagnosis says the opposite of what this assumes.** FP is already 2.6× FN. The model is *over*-segmenting, not under-segmenting. Pushing it further toward recall may make the hallucination worse. v12_tversky will tell us — results pending.

⚠️ **This is a genuine disagreement between the advisor's intuition and the measured data, and it needs resolving.**

**On 3D and shape priors:** It's 2.5D (3 slices as channels through a 2D U-Net), not volumetric, so a learned 3D shape prior isn't directly applicable. A cheap approximation is 3D connected-component post-processing to enforce inter-slice consistency — no retraining required. Not yet implemented.

## Papers suggested by professor
1. **FMed paper** — not yet assessed
2. **"Advancing DL knee cartilage segmentation in MRI"** (review) — orientation reading, not yet read

---

# 11. Code Map

## Key training scripts

| File | Purpose |
|---|---|
| `segmentation/finetune_meniscus_v7_replica.py` | **v7/v8 loss + full instrumentation** (train Dice, per-patient val Dice, val prediction dump, seed) — use this as the base |
| `segmentation/finetune_meniscus_v5_mixed.py` | ⚠️ **This is now the v9 restricted-CE variant, NOT v7/v8** |
| `segmentation/finetune_meniscus_v10_boundary.py` | Boundary-aware loss |
| `segmentation/finetune_meniscus_v11_cldice.py` | clDice topology loss |
| `segmentation/finetune_meniscus_v12_tversky.py` | Tversky on v7_mixed recipe |
| `segmentation/finetune_meniscus_v12_photometric.py` | Photometric aug on v7_mixed recipe |
| `segmentation/diagnose_checkpoint.py` | Read-only diagnostic on existing checkpoint, no retraining |

## Data prep

| File | Purpose | Note |
|---|---|---|
| `segmentation/prepare_meniscus_masks.py` | Fake PD slices + 3-class masks | **⚠️ line 129 is the root cause — skips empty slices** |
| `segmentation/prepare_real_pd_labeled.py` | Real PD slices + masks | Line 151 filters on image emptiness, not mask |
| `segmentation/dataset_2_5d_v2.py` | Fake PD dataset with augmentation | |
| `segmentation/dataset_2_5d_realpd.py` | Real PD dataset | |
| `segmentation/infer_real_pd_v3.py` | Inference on real PD volumes | |

## Analysis

| File | Purpose |
|---|---|
| `notebooks/analysis_phase2_experiments.ipynb` | Phase 2 comparison, per-patient Dice, bar charts |
| `scripts/plot_replica_diagnostics.py` | `diagnostics_curves.png`, `diagnostics_per_slice.png` |
| `docs/experiment_version_tracking.md` | Full run log |
| `docs/phase3_experiments.md` | Diagnosis document |
| `docs/future_experiments.md` | Earlier idea list |

## BigRed200 paths

Base: `/N/project/prostate_cancer_ai/anshika/regGAN/`
```
regGAN/                                  # cd here before running anything
├── segmentation_runs/
│   ├── run_v7_mixed/ckpt_best.pth       # ← BEST MODEL, Dice 0.781
│   └── run_v7_replica_seed42/           # ← instrumented diagnostic run
├── pretrained/baseline_best_model.pth   # DESS 5-class source
├── segmentation_data_v2/                # Fake PD slices (Old RegGAN)
├── real_pd_seg_data_v7/                 # Real PD, 15 patients
├── data/
│   ├── iu-dataset/pd-files/             # 69 IU PD volumes
│   └── iu-control/pd-files/             # ⚠️ TORN patients, misleading name
└── results/real_pd_predictions_*/       # inference outputs
```

Module: `python/gpu/3.11.5 cudatoolkit/12.2`

⚠️ **On BigRed, scripts live at the regGAN root, not in `slurm_files/`.**

---

# 12. Historical Bugs Fixed

These are resolved but explain design choices in the code.

| # | Bug | Fix |
|---|---|---|
| 1 | **Inplace autograd error** — `R.step()` mutated weights before `G.backward()` | Reorder: G backward → R step → D step |
| 2 | **Grid `.expand()` bug** — inplace modification in warp function | Use broadcast addition instead |
| 3 | **PD preprocessing 9× stretch** — wrong axis conflated in-plane and through-plane | Resample to isotropic in-plane first, then resize |
| 4 | **Affine mismatch** — output inherited DESS spacing | Compute effective spacing, build correct diagonal affine |
| 5 | **Non-orthonormal direction cosines** — 22/69 masks crashed SimpleITK | Try SimpleITK, fall back to `nibabel.as_closest_canonical()` |
| 6 | **SSIM shape mismatch** — wrong transpose gave (384,160) | Only transpose when last dim is smallest |
| 7 | **LPS/RAS coordinate flip** — SimpleITK reports LPS regardless of `DICOMOrient("RAS")` | Apply `diag([-1,-1,1]) @ direction` conversion |
| 8 | **Data leakage** — train/val/test split by slice, not patient | `split_by_patient()` with built-in leakage check |
| 9 | **GT mask path wrong** — `~/Downloads/segmentations/` doesn't exist | Corrected to `final-segmentations/`; re-ran all affected evals |
| 10 | **Train Dice discarded** — computed as `td`, dropped in an f-string | Fixed in `finetune_meniscus_v7_replica.py`; **this is what hid the overfitting** |

---

# 13. Standing Caveats — Read Before Trusting Any Number

1. **Findings in §6 come from 2 validation patients / 62 slices.** Confirm the hallucination pattern against the 17-patient predictions in `results/final_results/real_pd_predictions_v8_mixed/` before committing to a retrain.

2. **`finetune_meniscus_v5_mixed.py` is now the v9 restricted-CE variant, NOT v7/v8.** The original v7/v8 loss lives in commit `64b1020` and in `finetune_meniscus_v7_replica.py`. **Do not use v5_mixed to replicate v7/v8.**

3. **Report per-patient mean ± CI, not a bare Dice.** With 17 patients and a per-patient spread of ~0.05, SEM is ~0.012. Any difference under ~0.02 is not measurable.

4. **The pooled `vd` metric in `run_epoch` runs ~0.028 high** due to the cycling-iterator artifact. Use the clean per-patient mean.

5. **`iu-control` contains torn patients, not controls.**

6. **Old RegGAN (FID 164.4) beats new RegGAN (FID 147.6) downstream.** Better FID ≠ better segmentation.

---

# 14. Status Board

| Item | Status |
|---|---|
| v7 replica, seed 42 | ✅ Done — 0.7809 @ ep10 |
| v7 replica, seed 7 (noise floor) | ❌ Not submitted |
| `diagnose_checkpoint.py` on run_v7_mixed | ❌ **Built and copied to BigRed, never run** — 30-min read-only job (`slurm_files/diagnose_v7_mixed.sh`), scores the *actual* 0.781 checkpoint on all four splits + bootstraps the 2-patient val set |
| Confirm hallucination on 17pt cohort | ❌ Not started |
| Inspect the 9 hallucination slices in Slicer | ❌ Not started |
| Experiment 1 (negatives in fake-PD prep) | ❌ Not started |
| v12_tversky eval | 🔄 Inference done, Dice pending |
| v12_photometric eval | 🔄 Inference done, Dice pending |
| k-fold over 15 real PD patients | ❌ Not started |
| Localization + crop | ❌ Not started |
| 3D connected-component post-proc | ❌ Not started |
| GT masks for torn cohort | ❌ Blocked on annotation |

---

# 15. Specific Questions I Want Help With

1. **Is the hallucination diagnosis sound?** 2 patients / 62 slices is thin. What's the minimum evidence I should gather before committing to a retrain on new data?

2. **What negative-slice fraction should I use?** 0%, 25%, 50%, 100% of available empty slices? Is there a principled way to pick this rather than sweeping blindly?

3. **How do I reconcile the advisor's over-segmentation suggestion with FP being 2.6× FN?** Am I misreading the data, or is the suggestion based on an assumption that no longer holds?

4. **Given the +0.166 generalization gap, is the crop approach or self-training the better use of effort?** Both target overfitting. Which has the better expected value?

5. **Should I re-run Phase 2 experiments with proper CI reporting**, or accept that they were all noise and move on?

6. **Is there anything structurally wrong with the two-stage design** (translate → fine-tune) that I'm not seeing, given that the domain gap turned out to be small?

7. **The v10_boundary catastrophic failure on AC07607B9E5295** (0.843 → 0.680). Worth investigating, or a distraction?

---

# 16. Publication Status

**SPIE abstract — submitted.**

**Title:** Cross-Modality Transfer Learning for Knee Meniscus Segmentation via Registration-Constrained Unpaired Image Translation

**Authors:** Anshika Bajpai^a, Abhay Sista^b, Madilyn Feik^c, Mounica Chidurala^c, Ashley Ellenberger^c, Bryan Saltzman^c, Chia-Ying James Lin^d, Rakesh Shiradkar^d

a: Indiana University Bloomington · b: University of Virginia · c: IU School of Medicine · d: BME & Informatics, IU Indianapolis

**Headline claims in the abstract:**

| Model | Training data | Mean Dice (17pt) |
|---|---|---|
| Pretrained DESS baseline | DESS only | **0.00** |
| Synthetic PD fine-tune | Fake PD (RegGAN) | **0.69** |
| Synthetic + real PD | Fake PD + 15 real PD | **0.78** |

Relative gain synthetic→mixed: **13%**. Per-patient range: **0.69 to 0.84**.

Journal manuscript is the next step.

## A publication angle I already have data for: the label-efficiency curve

The abstract reports three models. The same numbers also trace a **label-efficiency
curve** — how Dice scales with the number of expert-annotated real PD volumes, given
anatomy-preserving synthetic pretraining:

| Real PD annotations | Dice | Source run |
|---|---|---|
| 0 | 0.695 | run_002_v2 |
| 10 | ~0.77 | run_v5_mixed |
| 15 | 0.781 | run_v7_mixed |

Three points, already measured, no new method required. Reframing the paper around
*"how few expert annotations do you need if your synthetic data preserves anatomy?"*
turns the existing results into the contribution, rather than treating annotation
count as an incidental detail.

**Why this is worth considering:** it is the question a clinical audience actually
has (annotation cost is the barrier, §1), it needs no new architecture, and the
curve is already flattening (10 → 15 patients bought only +0.016), which is itself
a reportable finding. Adding one or two more points — e.g. 5 and 20 annotations —
would make it a proper curve.

**Relation to §9:** this is a *framing* change, not an experiment. It is compatible
with everything in §9 and competes with none of it.

---

# 17. Verification Audit

> Every claim in this document was audited against source code, SLURM scripts, git history,
> and the raw diagnostic CSV on **2026-09-09**. This section records what was confirmed,
> what was corrected, and what remains unresolved. Read it before trusting any number above.

## 17.1 Claims verified against source

| # | Claim | Method | Result |
|---|---|---|---|
| 1 | Fake-PD prep skips empty slices | Read `prepare_meniscus_masks.py:129` | ✅ **CONFIRMED** |
| 2 | Default `min_meniscus_pixels = 10` | Read `prepare_meniscus_masks.py:84` | ✅ **CONFIRMED** |
| 3 | Real-PD prep filters on *image*, not mask | Read `prepare_real_pd_labeled.py:151` | ✅ **CONFIRMED** |
| 4 | `v5_mixed.py` is now the v9 restricted-CE variant | Found `ignore_index=0` at line 85 | ✅ **CONFIRMED** |
| 5 | `v7_replica.py` holds the original v7/v8 loss | Found plain `CrossEntropyLoss(weight=...)` at line 348 | ✅ **CONFIRMED** |
| 6 | Commit `64b1020` exists | `git log` | ✅ **CONFIRMED** |
| 7 | All §6.3 diagnostic numbers | Recomputed from `per_slice_dice.csv` | ✅ **CONFIRMED** (see 17.3) |
| 8 | `segmentation_data_v2` patient count | Traced SLURM `FAKE_PD_DIR` / `MASK_DIR` | ⚠️ **CORRECTED — 155, not 69** |
| 9 | Augmentation identical across branches | Diffed both dataset files | ❌ **FALSE — they differ, see 17.4** |
| 10 | Seeding might be broken by mixed RNG | Read `set_seed()` in v7_replica | ✅ **NOT A PROBLEM, see 17.5** |

### Exact source lines

```python
# segmentation/prepare_meniscus_masks.py:129   ← THE ROOT CAUSE
if (collapsed > 0).sum() < args.min_meniscus_pixels:   # line 84: default=10
    total_skipped_empty += 1
    continue

# segmentation/prepare_real_pd_labeled.py:151  ← filters on IMAGE, not mask
if skip_bg and img_sl.mean() < 0.02:
    continue
```

```python
# segmentation/finetune_meniscus_v5_mixed.py:85   ← IS the v9 variant
F.cross_entropy(..., ignore_index=0, reduction="none")

# segmentation/finetune_meniscus_v7_replica.py:348   ← IS the original v7/v8 loss
fake_loss_fn = lambda logits, y: (nn.CrossEntropyLoss(weight=weights)(logits, y) + ...)
```

## 17.2 Dice — definition and the three aggregations in use

**The document previously never defined Dice.** It is, from `analysis_phase2_experiments.ipynb`:

- **Binary** — lateral and medial merged via `merge_meniscus(pred) = (pred > 0)`
- **Volumetric** — computed over the whole 3D volume, not per-slice then averaged
- **Per-patient, then averaged** across the cohort

```python
def dice_score(pred_bin, gt_bin):
    inter = (pred_bin & gt_bin).sum(); denom = pred_bin.sum() + gt_bin.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0
```

**Real PD ground truth is BINARY** (label 1 = meniscus, no lateral/medial split), while the model
outputs 3 classes. That mismatch is the entire reason `MergedLoss` exists — it sums the lateral
and medial softmax probabilities before comparing against the binary GT.

### The three aggregations — do not confuse them

| Value | What it actually is | Where it appears |
|---|---|---|
| **0.781** | Per-patient mean over the **17-patient** eval cohort | §5.4–5.6, headline number |
| **0.7793** | Per-patient mean over the **2 real-PD val patients** | §6.1 |
| **0.7524** | **Pooled voxel-level** Dice across those same 2 val patients | §6.3 oracle calc |

These are three different quantities. 0.7524 is lower than 0.7793 because pooling weights large
slices more heavily. **Comparing them directly is a mistake**, and the oracle's `+0.067` applies to
the pooled figure only — it is not a promise of `0.781 → 0.848` on the 17-patient cohort.

### Other undefined terms now pinned down

- **"Win count" (§5.6)** = number of patients out of 17 where that run beat the `v8_mixed` baseline.
- **Baseline 0.00 vs 0.43** — these are two different things:
  - **0.00** = the pretrained DESS 5-class model applied directly to real PD, no fine-tuning. This is the true zero point.
  - **0.429** = `run_001`, the first fine-tuned model (55 DESS patients). Not a baseline.
  - The "43% → 78%" framing used in resume material refers to `run_001`, **not** the untrained baseline.

## 17.3 Phase 3 numbers — recomputed from raw data

Recomputed directly from `segmentation_runs/run_v7_replica_seed42/val_predictions/per_slice_dice.csv`
(62 rows, patients `AC0CE315D5758B` and `AC0CEE9C24F2B7`):

| Quantity | Doc claimed | Actual | Status |
|---|---|---|---|
| Slices total | 62 | 62 | ✅ |
| GT-empty slices | 32 | 32 | ✅ |
| Correctly silent | 23 | 23 | ✅ |
| Hallucinated slices | 9 | 9 | ✅ |
| Hallucinated voxels | 2,001 | 2,001 | ✅ |
| Worst single slice | 538 | 538 | ✅ |
| GT-nonempty slices | 30 | 30 | ✅ |
| Mean Dice (nonempty) | 0.7919 | 0.7919 | ✅ |
| Median Dice (nonempty) | 0.844 | 0.8443 | ✅ |
| Slices below 0.5 | 2 | 2 | ✅ |
| FN total | 1,682 | 1,682 | ✅ |
| FP total | 4,373 | 4,373 | ✅ |
| FP / FN ratio | 2.6× | 2.60× | ✅ |
| Halluc as % of FP | 46% | 45.8% | ✅ |
| Pooled Dice | 0.752 | 0.7524 | ✅ |
| Oracle Dice | 0.819 | 0.8194 | ✅ |
| Oracle gain | +0.067 | +0.0671 | ✅ |
| **TP** | **9,170** | **9,199** | ⚠️ corrected |
| **GT total** | **10,852** | **10,881** | ⚠️ corrected |
| **PRED total** | **13,543** | **13,572** | ⚠️ corrected |

**The diagnosis holds.** `phase3_experiments.md` has a transcription slip of exactly −29 in each of
TP / GT / PRED. Every ratio and conclusion is unaffected; §6.3 above has been corrected to the true values.

## 17.4 ⚠️ NEW FINDING — augmentation differs between the two training branches

This was **not** previously documented anywhere and is a live confound in every mixed run.

| Aspect | Fake PD (`dataset_2_5d_v2.py`) | Real PD (`dataset_2_5d_realpd.py`) |
|---|---|---|
| H-flip | `random.random() > 0.5` | `np.random.rand() < 0.5` |
| V-flip | `random.random() > 0.5` | `np.random.rand() < 0.5` |
| **Brightness form** | **Multiplicative** `image * U(0.8, 1.2)` | **Additive** `arr += U(-0.1, 0.1)` |
| **Brightness rate** | **100% — always applied** | **50%** |
| **Noise form** | `normal(0, 0.02)` | `randn * 0.02` |
| **Noise rate** | **100% — always applied** | **30%** |
| Clipping | After every op | Once at the end |

**The fake-PD branch is augmented substantially harder than the real-PD branch** — brightness and
noise are applied unconditionally on fake PD but gated at 50% / 30% on real PD, and the brightness
transform is a different *kind* of operation on each side.

**Why this matters for the diagnosis:** the fake stream drives the training loop *and* gets heavier
augmentation. Both point the same direction — fake PD dominates what the model learns. It is also a
plausible secondary contributor to the +0.166 train/val gap on the real-PD side, which receives
comparatively weak regularization.

**This is unlikely to be deliberate.** It should be either equalized or justified before any further
loss-function experiments, since it confounds every fake-vs-real comparison in §6.2.

## 17.5 Claims from my own critique that turned out to be WRONG

Recorded so nobody re-raises them:

| Concern | Verdict |
|---|---|
| Mixed RNG (`random` vs `np.random`) breaks reproducibility | ❌ **False.** `set_seed()` in v7_replica seeds `random`, `np.random`, `torch`, `torch.cuda`, and sets `cudnn.deterministic=True`. Seeding is correct. |
| Uncommitted `dataset_2_5d_realpd.py` changes affect results | ❌ **Benign.** The diff only adds a `patient_id` key to the returned dict, explicitly inert for training. |
| Uncommitted `v5_mixed.py` changes alter the loss | ❌ **No.** The uncommitted diff only adds instrumentation (fake-PD Dice logging, train/val gap in the log line). The v9 loss change is *already committed* — so the file is divergent from v5/v7 by the committed loss change, not by the working-tree diff. |

## 17.6 Content that was missing and is still missing

| Gap | Impact | Status |
|---|---|---|
| **155-patient scale-up phase absent** | run_001→run_002 gave 0.49 → 0.83 fake-PD val Dice from 2.2× more data. The single largest gain in project history, and it's a data-scale lever, consistent with the Phase 3 diagnosis. | ⚠️ Still missing — see 17.7 |
| **Earlier analysis reached the OPPOSITE conclusion** | `TECHNICAL_DESIGN.md` §6b concluded "no overfitting… the gap is domain gap, not memorisation." Phase 3 concludes the reverse. | ⚠️ Reconciled in 17.7 |
| **No training curves referenced** | `diagnostics_curves.png` and `diagnostics_per_slice.png` exist and are never pointed at | ⚠️ Noted here |
| **v3_5slice at 0.486 not flagged as suspicious** | A 0.676 → 0.486 drop from 3→5 slice context is too large to be a real effect; likely a channel/preprocessing bug in `finetune_meniscus_v3_5slice.py` | ⚠️ Should be re-examined or excluded |
| **Fake-val and real-val are different patient sets** | §6.2 compares "fake-PD val 0.760" to "real-PD val 0.781" as if one cohort. They are not — fake val is DESS-derived patients, real val is `AC0CE315D5758B` / `AC0CEE9C24F2B7`. Weakens but does not overturn the comparison. | ⚠️ Caveat noted |
| **v5_mixed: 0.765 or 0.769?** | `experiment_version_tracking.md` §4d says 0.765; `PROJECT_HANDOFF.md` §6 says 0.769. Unresolved. | ⚠️ Sources conflict |
| **`run_v7_realPD` Dice** | `PROJECT_HANDOFF.md` says "PENDING"; `experiment_version_tracking.md` says 0.774. Tracking doc is newer; 0.774 used. | ℹ️ Resolved in favour of newer doc |

## 17.7 Reconciling the contradiction with the earlier analysis

`TECHNICAL_DESIGN.md` §6b, written when analyzing `run_002` (fake-PD only, 155 patients):

> *"Train and val loss track closely → **no overfitting** in the classical sense.
> The 0.83 val Dice → ~0.63 real PD Dice gap is **domain gap**, not memorisation."*

Phase 3 concludes the opposite: overfitting confirmed (+0.166), domain gap small (0.760 vs 0.781).

**Both can be true, because they describe different training regimes:**

| | run_002 (fake-only) | v7_replica (mixed) |
|---|---|---|
| Training data | Fake PD only | Fake PD **+ 15 real PD patients** |
| Val Dice on fake PD | 0.8341 | 0.760 |
| Dice on real PD | 0.632 | 0.781 |
| Apparent gap | **0.20 — large** | **0.02 — small** |

Under fake-only training the model has never seen real PD, so the fake→real gap is genuinely large.
Once 15 real PD patients enter training, that gap closes **by construction** — the model has now
seen the real distribution. So §6.2's "domain gap is small" is a statement about the *mixed* regime
only, and does not contradict the earlier fake-only observation.

**Two important caveats on this reconciliation:**

1. It weakens §6.2's argument against the diffusion idea. The gap being small *after* training on
   real data does not prove better synthetic images wouldn't help — it partly reflects that real
   data was added. The stronger evidence against diffusion remains the run_005 anomaly
   (best FID 147.6 → worst Dice 0.640).
2. The "no overfitting" claim in the earlier doc was **never actually supported**, because train
   Dice was computed and discarded in an f-string until `v7_replica` (bug #10 in §12). It was
   inferred from loss curves alone.

## 17.8 Runtimes (previously missing)

From the SLURM headers:

| Job | Wall-clock limit | Partition |
|---|---|---|
| `finetune_v7_mixed.sh` | 4 h | gpu |
| `finetune_v11_cldice.sh` | 8 h | gpu |
| `infer_v11_cldice.sh` | 4 h | gpu |
| `diagnose_v7_mixed.sh` | **30 min** | gpu |

The read-only diagnostic is 30 minutes and requires no retraining — it is by far the cheapest way
to test a hypothesis against an existing checkpoint.

## 17.9 Recommended reading order for a fresh debugger

1. §0 (TL;DR) → §6 (diagnosis) → §7 (root cause) — the core argument
2. **§17.2** — before touching any number, so the three Dice aggregations don't get conflated
3. **§17.4** — the augmentation asymmetry, because it confounds §6.2
4. **§17.7** — so the earlier opposite conclusion doesn't undermine confidence
5. §9 (ranked experiments) → §15 (open questions)
6. §13 + §17.6 — everything that is still uncertain

## 17.10 What still cannot be verified from documentation

These need a live check on BigRed or a new run:

- Actual patient/slice counts in `segmentation_data_v2/` on BigRed (`ls | wc -l`) — confirms the 155 figure empirically
- How many empty slices `--min_meniscus_pixels 0` would actually add (the log line
  `total_skipped_empty` reports it, but the value was not retained)
- Whether the hallucination pattern reproduces on the 17-patient cohort (the single most
  important open item — see §14)
- Seed-noise floor: only one seed (42) has been run; seed 7 was never submitted
