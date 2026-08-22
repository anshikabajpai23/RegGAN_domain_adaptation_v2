# RegGAN Domain Adaptation — Project Handoff

> **Purpose:** Complete context document for continuing this project in a new conversation.  
> **Last updated:** 2026-08-22  
> **Author:** Anshika Bajpai (anbajpai@iu.edu)  
> **Advisor:** Rakesh Shiradkar

---

## 1. Project Goal

Translate **DESS knee MRI** (SKM-TEA dataset, publicly annotated) → **PD-weighted knee MRI** (IU institutional dataset, no labels) using unpaired image-to-image translation with **minimum anatomical deformation**, so DESS segmentation masks can supervise a meniscus segmentation model that works on real PD MRI.

**Why this matters:** PD MRI is the clinical standard for knee diagnosis. Labeling PD MRI requires hours of expert radiologist time per volume. DESS has free public labels (SKM-TEA). Direct model transfer fails because fluid is dark on DESS and bright on PD — reversed contrast. CycleGAN produces 40× higher deformation than RegGAN — unacceptable for medical imaging.

---

## 2. Datasets

| | DESS (SKM-TEA) | PD (IU dataset) |
|---|---|---|
| Local path | `data/skm-tea-dataset/dess-files` | `~/Desktop/AImed-lab/IU-Dess-dataset/iu-dataset/pd-files` |
| BigRed path | `/N/.../data/dess-files` | `/N/.../data/iu-dataset/pd-files` |
| # volumes | 69 | 69 |
| Raw shape | (512, 512, 160) | (384, 384, 36) |
| Voxel spacing | 0.31×0.31×0.80 mm | 0.39×0.39×3.60 mm |
| Slices extracted | ~11,040 | ~2,271 |
| Masks | SKM-TEA `.seg.nrrd` (6 tissue classes) | None |
| Meniscus labels | 5=lateral, 6=medial | N/A |

**Torn meniscus patients cohort (new, not in training/eval):**
- BigRed: `/N/project/prostate_cancer_ai/anshika/regGAN/data/iu-control/pd-files` ⚠️ folder name `iu-control` is misleading — these are actual torn meniscus patients annotated by doctors (s83_2 dataset on BigRed)
- 20 patients, first 20 files alphabetically, no GT segmentation masks locally
- Local PD files (5 of 20): `data/iu-control/pd-files/`

---

## 3. RegGAN Architecture

### Generator G_AB and G_BA (ResNet-based)
- Input: (B, 1, 384, 384)
- Encoder: 3 conv blocks with stride-2 downsampling
- 9 residual blocks (`n_res=9`, `ngf=48`)
- Decoder: bilinear upsample + conv (no checkerboard)
- Output: (B, 1, 384, 384), Tanh, range [-1,1]
- InstanceNorm2d, ReflectionPad2d

### Discriminator D_A and D_B (PatchGAN 70×70)
- `ndf=48`, 4 conv layers stride-2
- InstanceNorm + LeakyReLU(0.2)
- No sigmoid (LSGAN / MSE loss)

### Registration Network R (VoxelMorph-lite 2D)
- Input: (B, 2, 384, 384) — concat(fake_B, real_B)
- Output: (B, 2, 384, 384) — displacement field (Δx, Δy)
- U-Net with AvgPool2d encoder, bilinear upsample decoder
- `nf=16` (small = less aggressive warping)
- Flow head: Conv2d(nf, 2, 3), near-zero weight init
- Differentiable bilinear grid_sample for warping

### Loss Functions
```
L_G = L_GAN_AB + L_GAN_BA
    + λ_cycle=10.0 × (L_cycle_A + L_cycle_B)
    + λ_cycle × 0.5 × (L_idt_A + L_idt_B)
    + λ_reg_sim=5.0 × L_reg_sim        # warped fake_B ≈ real_B
    + λ_smooth=10.0 × L_smooth         # spatial gradients of field
    + λ_mag=5.0 × L_mag                # direct deformation magnitude

L_D = 0.5 × (MSE(D(real),1) + MSE(D(fake),0))   # LSGAN
```

### Training Details
| Param | Value |
|---|---|
| Optimizer | Adam β=(0.5, 0.999) |
| LR generators | 2e-4 |
| LR registration | 1e-4 |
| Batch size | 8 |
| Epochs run | ~4 of 200 |
| Image pool | 50 |
| Gradient clip | 5.0 (generators only) |
| Hardware | NVIDIA A100 40GB, BigRed200 |

**Training step order (critical):** G backward → R step → D step. (R.step() before G.backward() causes inplace autograd error.)

---

## 4. RegGAN Evaluation Results

Source file: `inference/eval2/metrics.json` (this is the canonical metrics file for the abstract)

| Metric | Value |
|---|---|
| FID (fake PD vs real PD) | **164.4** |
| FID (DESS vs real PD) | **260.4** |
| FID improvement | **37%** |
| KID | **0.020 ± 0.000** |
| SSIM (DESS vs fake PD) | **0.357** |
| Jacobian det mean | **1.000056** |
| Jacobian det min | 0.74 |
| Jacobian folding (det<0) | **0.0%** |
| Global mean deformation | **0.029 px** |
| Meniscus mean deformation | **0.055 px** |

**Note on SSIM:** 0.357 is expected and correct. DESS fluid=dark, PD fluid=bright — high SSIM would mean no translation. Low SSIM = translation worked.

**Note on FID:** Using 64×64 pixel features (no InceptionV3 — no internet on HPC). Relative comparison valid; absolute may differ from standard FID.

---

## 5. Segmentation Pipeline

### Baseline Model
- `pretrained/baseline_best_model.pth`
- Source: pitthexai 2.5D U-Net, ResNet34 encoder, pre-trained on DESS, 5-class output
- Applied directly to real PD → Dice **0.00** (complete failure, expected)

### Architecture Used for Fine-tuning
```
smp.Unet(encoder_name="resnet34", in_channels=3, classes=5)
→ head replaced: 5 → 3 classes (background, lateral meniscus, medial meniscus)
Input: 3-slice 2.5D stack, (3, 384, 384)
```

### Fine-tuning Loss
- **Fake PD data:** `CrossEntropyLoss(weight=[0.1, 1.5, 1.5]) + SoftDiceLoss()`
- **Real PD data:** `MergedLoss(w_bg=0.1, w_men=1.5)` — combines lateral+medial softmax probs against binary GT

### Fine-tuning Hyperparameters
- LR: 1e-5 → 1e-8 (CosineAnnealingLR, T_max=50, eta_min=1e-8)
- Epochs: 50, early stopping patience=10
- Batch size: 8
- Augmentation: h/v flip (p=0.5), brightness ±20%, Gaussian noise σ=0.02
- Class weights: bg=0.1, lateral=1.5, medial=1.5

### Key Scripts
| Script | Purpose |
|---|---|
| `segmentation/finetune_meniscus_v5_mixed.py` | Fine-tuning script (used for both v5 AND v7 runs) |
| `segmentation/dataset_2_5d_v2.py` | Fake PD dataset with augmentation |
| `segmentation/dataset_2_5d_realpd.py` | Real PD dataset |
| `segmentation/infer_real_pd_v3.py` | Inference on real PD volumes |

---

## 6. All Experiment Versions

### RegGAN Translation Models
| Name | FID | Notes |
|---|---|---|
| Old RegGAN | 164.4 | Used in all segmentation runs (segmentation_data_v2/) |
| run006 ep14 | 174.4 | Worse than old |
| run006 ep52 | 147.6 | Best FID, but not used for seg (venv compatibility) |

### Segmentation Runs — Full Table

| Run | Starting Ckpt | Real PD | Fake PD | Eval | Dice |
|---|---|---|---|---|---|
| run_002 | baseline | No | Old RegGAN | D1 (10pt) | 0.632 |
| run_002_v2 | baseline | No | Old RegGAN | D1 (10pt) | **0.695** |
| run_003 | baseline | No | Old RegGAN | D1 (10pt) | 0.650 |
| run_v5_realPD | baseline | 10pt only | No | D1 (10pt) | 0.135 (collapsed) |
| run_v5_mixed | baseline | 10pt (8 train, 2 val) | Old RegGAN | D1 (10pt) | **0.769** |
| run_v7_realPD | baseline | 15pt (13 train, 2 val) | No | TBD | PENDING |
| run_v7_mixed | baseline | 15pt (13 train, 2 val) | Old RegGAN | D1+New (17pt) | **0.781** |

### v8 = Inference Only (no new training)
v8 = running `run_v7_mixed/ckpt_best.pth` on **17 patients** (D1 10pt + 7 new)

### Final Results for SPIE Abstract (v8, 17 patients)
| Model | Training Data | Mean Dice |
|---|---|---|
| Pre-trained DESS baseline | DESS only | **0.00** |
| Synthetic PD FT (run_002_v2) | Fake PD (RegGAN) | **0.69** |
| Synthetic + Real PD FT (v7_mixed) | Fake PD + 15 annotated real PD | **0.78** |

- Relative gain synthetic→mixed: **13%**
- Per-patient range (v7_mixed): **0.69 to 0.84** across 17 patients
- Per-patient showcase (used in paper figure):
  - Good case 1 — AC07607B9E5295: mixed=0.843, syn=0.871
  - Good case 2 — AC005135D3495B: mixed=0.822, syn=0.728
  - Challenging — AC056ADCE8BE28: mixed=0.685, syn=0.558

---

## 7. Patient Cohorts

### D1 — Hold-out Eval (10 patients, NEVER in training)
File: `labeled_patients.txt`
```
AC0D5A4D78B628  AC0D7BF72F7712  AC0D3459553205  AC14D3737C0482  AC19E7C19827FF
AC149BC218E75C  AC111633B463BB  AC13300201B926  AC12026D14291F  AC13637DA25399
```
GT masks (local): `~/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final/` (`.seg.nrrd`)

### D1+New — Extended Eval (17 patients, v8 only)
File: `labeled_patients_v8.txt`
D1 (10 above) + 7 new: `AC000550763509, AC005135D3495B, AC04433E37DB66, AC056ADCE8BE28, AC07607B9E5295, AC0D1A9818A6FC, AC0F11041F5180`
GT masks (new 7, local): `segmentation_pd/segmentations/` (in repo)

### D2 — Fine-tuning Real PD
- **v5:** 10 patients (8 train, 2 val: AC2B0AA9AE767D, AC2E254F52E467)
- **v7:** 15 patients (13 train, 2 val: AC0CE315D5758B, AC0CEE9C24F2B7)
GT masks (local): `segmentation_pd/` root (mixed with D2 + others)

### Torn Meniscus Patients — No GT masks (20 patients)
BigRed: `/N/.../data/iu-control/pd-files` (folder named `iu-control` but actually torn patients, s83_2 dataset annotated by doctors), first 20 alphabetically

---

## 8. Local File Structure

```
RegGAN_domain_adaptation_v2/
├── models.py                    # Generator, Discriminator, RegistrationNet, losses
├── train.py                     # RegGAN training loop
├── dataset.py                   # Unpaired slice dataset
├── preprocess/                  # NIfTI preprocessing scripts
├── segmentation/
│   ├── finetune_meniscus_v5_mixed.py   # Fine-tuning (used for v5 AND v7)
│   ├── dataset_2_5d_v2.py              # Fake PD dataset with augmentation
│   ├── dataset_2_5d_realpd.py          # Real PD dataset
│   └── infer_real_pd_v3.py             # Inference script
├── slurm_files/
│   ├── finetune_v7_mixed.sh            # v7_mixed training job
│   ├── infer_v8_mixed.sh               # Inference on 17 patients
│   ├── infer_control_20pt.sh           # Inference on 20 control patients (NEW)
│   └── ...
├── results/
│   ├── real_pd_predictions_v5_mixed/   # D1 predictions (v5_mixed)
│   ├── real_pd_predictions_v8_mixed/   # 17pt predictions (v7_mixed checkpoint)
│   ├── real_pd_predictions_v8_pseudoPD/ # 17pt predictions (run_002_v2)
│   ├── real_pd_predictions_v8_baseline/ # 17pt predictions (baseline)
│   └── eval_run006/metrics.json        # RegGAN eval for run006 (NOT the abstract source)
├── inference/
│   └── eval2/metrics.json              # ← CANONICAL metrics for abstract (old RegGAN)
├── segmentation_pd/
│   ├── *.seg.nrrd                      # D2 + some new eval patients (mixed, 16 files)
│   └── segmentations/                  # New 7 eval patients only (7 files)
├── labeled_patients.txt                # D1: 10 hold-out eval patients
├── labeled_patients_v8.txt             # D1+New: 17 hold-out eval patients
├── experiment_version_tracking.md      # Full run log (all versions)
├── analysis_abstract_3models.ipynb     # D1 10pt analysis (v5_mixed, run_002_v2, baseline)
├── analysis_abstract_17patients.ipynb  # 17pt abstract analysis (v8 models) ← USE THIS
├── analysis_v8_17patients.ipynb        # v8 analysis notebook
├── analysis_all_runs.ipynb             # All 9 prediction dirs compared
├── spie_abstract/
│   ├── spie_abstract.tex               # LaTeX abstract (updated to 17pt numbers)
│   └── spie_abstract.docx
└── supplementary_material_final.pdf    # Submitted supplementary (5 pages)
```

---

## 9. BigRed200 Paths

Base: `/N/project/prostate_cancer_ai/anshika/regGAN/`

```
regGAN/                           # Working directory (cd here before any script)
├── segmentation_runs/
│   ├── run_v5_mixed/ckpt_best.pth     # Dice 0.769 on D1 10pt
│   └── run_v7_mixed/ckpt_best.pth     # Dice 0.781 on 17pt ← BEST MODEL
├── pretrained/baseline_best_model.pth  # DESS 5-class source baseline
├── segmentation_data_v2/              # Fake PD slices (Old RegGAN, 69pt)
├── real_pd_seg_data/                  # Real PD v5 (10 patients)
├── real_pd_seg_data_v7/               # Real PD v7 (15 patients)
├── data/
│   ├── iu-dataset/pd-files/           # 69 IU PD volumes
│   └── iu-control/pd-files/           # Control cohort (20 patients for inference)
└── results/
    ├── real_pd_predictions_v8_mixed/  # 17pt v7_mixed predictions
    └── real_pd_predictions_control_20pt/  # Control 20pt (pending)
```

**Module to load on BigRed:** `python/gpu/3.11.5 cudatoolkit/12.2`  
(Previously `python/3.11` — that module no longer exists)

---

## 10. SPIE Abstract — Submitted Version

**Title:** Cross-Modality Transfer Learning for Knee Meniscus Segmentation via Registration-Constrained Unpaired Image Translation

**Authors:** Anshika Bajpai^a, Abhay Sista^b, Madilyn Feik^c, Mounica Chidurala^c, Ashley Ellenberger^c, Bryan Saltzman^c, Chia-Ying James Lin^d, Rakesh Shiradkar^d

**Affiliations:**
- a: Indiana University Bloomington
- b: University of Virginia
- c: Indiana University School of Medicine
- d: BME & Informatics, Indiana University Indianapolis

**Submitted file:** `supplementary_material_final.pdf` (5 pages)

**Program abstract (124 words):**
> Automated meniscus segmentation in proton density (PD)-weighted knee MRI is hindered by the scarcity of labeled training data, as expert voxel-level annotation demands hours of radiologist time per volume. We propose RegGAN, a registration-constrained generative adversarial network that translates DESS knee MRI—which carries rich public segmentation labels—to synthetic PD images while explicitly penalizing anatomical deformation, preserving meniscus anatomy (mean displacement 0.055 pixels, 0% topology violations). A DESS-trained segmentation model applied directly to real PD achieves near-zero meniscus Dice (0.00). Fine-tuning on synthetic PD raises Dice to 0.69 on 17 held-out expert-annotated real PD patients; adding 15 annotated real PD cases further improves Dice to 0.78—a 13% relative gain—demonstrating that anatomy-preserving synthetic data substantially reduces annotation burden for cross-modality MRI segmentation.

**Key verified numbers:**
| Claim | Verified Source |
|---|---|
| FID 164.4 / 260.4 / 37% | `inference/eval2/metrics.json` ✅ |
| SSIM 0.357 | `inference/eval2/metrics.json` ✅ |
| Jacobian mean 1.000056, 0% folding | `inference/eval2/metrics.json` ✅ |
| Meniscus 0.055 px | `inference/eval2/metrics.json` ✅ |
| Baseline 0.00, Syn 0.69, Mixed 0.78 | `analysis_abstract_17patients.ipynb` ✅ |
| 13% relative gain, range 0.69–0.84 | `analysis_abstract_17patients.ipynb` ✅ |
| LR 1e-5 → 1e-8 | `finetune_meniscus_v5_mixed.py:172` ✅ |
| Loss weights, epochs, augmentation | Code-verified ✅ |

---

## 11. Known Bugs Fixed (Historical)

1. **Inplace autograd error** — R.step() before G.backward() caused version mismatch. Fix: G backward → R step → D step.
2. **Grid expand bug** — `.expand()` on grid tensor caused inplace modification. Fix: use broadcast addition instead.
3. **PD preprocessing shrink bug** — Wrong axis caused 9× stretch. Fix: resample in-plane first, then resize.
4. **Affine mismatch in NIfTIs** — Saved with wrong affine. Fix: compute effective spacing, build diagonal affine.
5. **Non-orthonormal direction cosines** — 22/69 masks crashed SimpleITK. Fix: try SimpleITK, fallback to nibabel.as_closest_canonical().
6. **SSIM shape mismatch** — Wrong transpose loaded (384×160) instead of (384×384). Fix: only transpose when last dim is smallest.

---

## 12. Open Issues / Pending

- `analysis_all_runs.ipynb`: `v5_realPD` row incorrectly points to v5_mixed dir — no separate v5_realPD dir exists locally. Needs correction.
- `run_v7_realPD` Dice: PENDING (not yet evaluated)
- Control 20pt inference: in progress (`infer_control_20pt.sh` submitted, output at `real_pd_predictions_control_20pt/`)
- `supplementary_material_final.pdf` formatting: Section 2 header ("METHODDatasets") merged — needs line break. Two em-dash punctuation issues in new sentences.

---

## 13. High-Impact Future Directions

Ranked by research novelty + clinical value:

| Direction | Novelty | Clinical Value | Notes |
|---|---|---|---|
| **Tear detection from segmentation masks** | Medium | Very High | Pipeline exists; add classification head on masks |
| **Active learning for annotation selection** | High | High | Which PD cases to annotate next — reduces 15→5 labels |
| **Diffusion model for DESS→PD** (DDPM + CycleGAN) | Very High | Medium | Better image quality → higher Dice ceiling |
| **Separate lateral/medial Dice reporting** | Low | High | Model already has 3 classes — just report separately |
| **Uncertainty quantification** | Medium | High | MC dropout / ensemble — flags low-confidence cases |
| **Quantitative biomarkers** | Medium | High | Volume, extrusion from masks → OA grading |
| **Multi-site generalization** | Medium | Medium | Control cohort is first step toward this |

**Most actionable next step:** Tear detection — the segmentation masks are already generated, annotating tear presence is cheaper than full voxel-level annotation, and it directly answers the clinical question radiologists face.

---

## 14. How to Run Inference (Quick Reference)

```bash
# On BigRed200
module load python/gpu/3.11.5 cudatoolkit/12.2
source /N/project/prostate_cancer_ai/anshika/regGAN/regGAN/venv/bin/activate
cd /N/project/prostate_cancer_ai/anshika/regGAN/regGAN

# Submit job
sbatch infer_control_20pt.sh

# SCP results to local
scp -r anbajpai@bigred200.uits.iu.edu:/N/project/prostate_cancer_ai/anshika/regGAN/results/real_pd_predictions_control_20pt \
    /Users/anshikabajpai/Desktop/github/RegGAN_domain_adaptation_v2/results/
```

---

*All numbers in this document have been verified against source code and metrics files. Do not use CLAUDE.md alone — some values there (e.g., Jacobian det min 0.74 for a different run) differ from the canonical abstract source (`inference/eval2/metrics.json`).*
