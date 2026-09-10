# Experiment Version Tracking — Full Log
> BigRed200 base: `/N/project/prostate_cancer_ai/anshika/regGAN/`
> Local base: `~/Desktop/github/RegGAN_domain_adaptation_v2/`
> Last updated: 2026-08-29

---

## 1. RegGAN Translation Models (DESS → Fake PD)

| Name | BigRed Path | # DESS Patients | # PD Patients | Epochs Run | FID | Notes |
|---|---|---|---|---|---|---|
| Old RegGAN | `segmentation_data_v2/` | ~69 | ~69 | Unknown | 164.4 | Used in most segmentation runs |
| New RegGAN run006 ep14 | `fake_pd_run006_ep14/` | 155 | 155 | 14 | 174.4 | Worse than old |
| New RegGAN run006 ep52 | `fake_pd_run006_ep52/` | 155 | 155 | 52 | 147.6 | Best FID overall |

> **Note:** `segmentation_data_v2/` is the fake PD used in all fine-tuning runs unless stated otherwise.

---

## 2. Evaluation Cohorts

### D1 — Original Hold-out (10 patients) — NEVER used in training
**File:** `labeled_patients.txt` (local + BigRed root)
```
AC0D5A4D78B628   AC0D7BF72F7712   AC0D3459553205   AC14D3737C0482   AC19E7C19827FF
AC149BC218E75C   AC111633B463BB   AC13300201B926   AC12026D14291F   AC13637DA25399
```
- GT masks (local): `Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final/` (`.seg.nrrd`)
- GT masks (BigRed): not needed — eval runs locally after SCP

### D1+New — Extended Hold-out (17 patients) — v8 only
**File:** `labeled_patients_v8.txt` (local + BigRed root)
- D1 (10 above) + 7 new: `AC000550763509, AC005135D3495B, AC04433E37DB66, AC056ADCE8BE28, AC07607B9E5295, AC0D1A9818A6FC, AC0F11041F5180`
- GT masks (new 7, local): `~/Downloads/final-segmentations/` (`.seg.nrrd`)
- PD images (local): `Desktop/AImed-lab/IU-Dess-dataset/iu-dataset/pd-files/`

---

## 3. Real PD Training Cohorts (D2)

### real_pd_seg_data — v4, v5 (10 patients total)
| Split | # Patients | Patient IDs |
|---|---|---|
| Train | 8 | On BigRed in `labelled-pd/` (IDs not listed locally) |
| Val | 2 | AC2B0AA9AE767D, AC2E254F52E467 |

- Masks: `data/iu-dataset/labelled-pd-segmentations/` (original 10 `.seg.nrrd`)
- Prepare script: `slurm_files/prepare_real_pd.sh`

### real_pd_seg_data_v7 — v7 (15 patients total)
| Split | # Patients | Patient IDs |
|---|---|---|
| Train | 13 | All 10 original D2 + AC0BA0AE159EF9, AC045F8F6ACBA7, AC0407F05FAF53 |
| Val | 2 | AC0CE315D5758B, AC0CEE9C24F2B7 (from aug 3rd batch) |

- 5 new masks added: `AC0BA0AE159EF9, AC0CE315D5758B, AC0CEE9C24F2B7, AC045F8F6ACBA7, AC0407F05FAF53`
- Prepare script: `slurm_files/prepare_real_pd_v7.sh`

---

## 4. Segmentation Runs — Full Table

> Starting checkpoint: **baseline** = `pretrained/baseline_best_model.pth` (pitthexai 2.5D U-Net, 5-class DESS)
> All checkpoints: `segmentation_runs/<run_name>/ckpt_best.pth`

### 4a. Fake PD only (no real PD labels)

| Run | Fake PD Data | RegGAN | Starting Ckpt | Eval Set | D1 Dice (10pt) | Notes |
|---|---|---|---|---|---|---|
| run_002 | segmentation_data_v2 | Old | baseline | D1 | 0.632 | Early run |
| run_002_v2 | segmentation_data_v2 | Old | baseline | D1 | **0.695** | Best fake-PD-only model |
| run_003 | segmentation_data_v2 | Old | baseline | D1 | 0.650 | + early stopping |
| run_005 | fake_pd_run006_ep52 | New (ep52) | baseline | D1 | 0.640 | New RegGAN ep52 |
| run_ep14_v2 | fake_pd_run006_ep14 | New (ep14) | baseline | D1 | 0.603 | New RegGAN ep14 |

### 4b. v3 Ablations — fake PD only, all use Old RegGAN segmentation_data_v2, from baseline

| Run | What changed | D1 Dice (8pt) |
|---|---|---|
| run_v3_rotation | +rotation augmentation | 0.676 |
| run_v3_tversky | Tversky loss (α=0.3, β=0.7) | 0.656 |
| run_v3_elastic | +elastic deform aug | 0.652 |
| run_v3_photometric | +photometric aug | 0.681 |
| run_v3_resnet50 | ResNet50 encoder | 0.639 |
| run_v3_5slice | 5-slice 2.5D context | 0.486 |

> All v3 ablations worse than or equal to run_002_v2 baseline (0.685). Tversky loss specifically -2.9%.

### 4c. v4 — Real PD fine-tuning starting from run_002_v2, D2 (10 patients)

| Run | Real PD Data | Fake PD Data | Starting Ckpt | Eval Set | D1 Dice (8pt) |
|---|---|---|---|---|---|
| run_v4_realPD | real_pd_seg_data (8 train, 2 val) | — | run_002_v2 | D1 | 0.775 |
| run_v4_mixed | real_pd_seg_data (8 train, 2 val) | segmentation_data_v2 | run_002_v2 | D1 | 0.759 |

### 4d. v5 — Real PD fine-tuning starting from DESS baseline, D2 (10 patients)

| Run | Real PD Data | Fake PD Data | Starting Ckpt | Eval Set | D1 Dice (8pt) | Notes |
|---|---|---|---|---|---|---|
| run_v5_realPD | real_pd_seg_data (8 train, 2 val) | — | baseline | D1 | 0.135 | Collapsed — real PD only not enough from scratch |
| run_v5_mixed | real_pd_seg_data (8 train, 2 val) | segmentation_data_v2 | baseline | D1 | 0.765 | Best on 8pt at the time |

### 4e. v7 — Real PD fine-tuning starting from DESS baseline, D2+aug3rd (15 patients)

| Run | Real PD Data | Fake PD Data | Starting Ckpt | Key change | D1+New Dice (17pt) |
|---|---|---|---|---|---|
| run_v7_realPD | real_pd_seg_data_v7 (13 train, 2 val) | — | baseline | 15pt real PD, no fake | 0.774 |
| run_v7_mixed | real_pd_seg_data_v7 (13 train, 2 val) | segmentation_data_v2 | baseline | 15pt real PD + fake | **0.781** |

> run_v7_mixed = **best training run overall**. Checkpoint used for all v8+ inference.

### 4f. Phase 2 — Inference experiments on v7_mixed checkpoint (no retraining)

> All use run_v7_mixed/ckpt_best.pth. Evaluated on D1+New (17pt). "v8" = naming convention for 17pt inference.

| Label | What changed | Script | D1+New Dice (17pt) | vs baseline |
|---|---|---|---|---|
| v8_mixed | Baseline inference, no changes | infer_v8_mixed.sh | 0.781 | — |
| v8_filled (Exp B) | Post-processing: fill_holes + remove_small_objects | postprocess_predictions.py | 0.778 | -0.003 |
| v8_clipped (Exp A) | Input clipped to 95th percentile before inference | infer_v8_clipped.sh | 0.784 | +0.003 |

### 4g. Phase 2 — Retraining experiments (new loss functions, same data as v7_mixed)

> All start from DESS baseline, use real_pd_seg_data_v7 (15pt) + segmentation_data_v2 (fake PD).
> Same hyperparams as v7_mixed (LR=1e-5, class weights 0.1/1.5/1.5, 50 epochs, patience=10) unless noted.

| Run | Key difference | Script | D1+New Dice (17pt) | vs baseline | Win count |
|---|---|---|---|---|---|
| run_v9_mixed | Segmentation-restricted CE (background excluded from loss) | finetune_meniscus_v9_mixed.py | 0.753 | -0.028 | 0/17 |
| run_v10_boundary | Standard CE + Dice + boundary-aware loss (λ=1.0) | finetune_meniscus_v10_boundary.py | — | — | — |
| run_v11_cldice | Standard CE + Dice + clDice topology loss (λ=1.0, iters=10) | finetune_meniscus_v11_cldice.py | — | — | — |

> v9 confirmed: restricting background CE hurt performance. Background context is needed for boundary discrimination.
> v10 and v11: inference complete, fill in Dice when eval runs.

**Key lessons from Phase 2:**
- Loss function changes (Tversky, restricted CE) consistently hurt — do not retry without new insight
- Post-processing (fill_holes) neutral on healthy patients — designed for torn cohort
- Intensity clipping marginally helps (+0.003) — minimal effect on healthy patients
- clDice (v11): wins most individual patients (7/17) — most promising topology loss so far

---

## 5. Inference & Evaluation Versions

> All infer scripts read from a patient list file, run `infer_real_pd_v3.py` (or `infer_real_pd_baseline.py` for baseline), output `.nii.gz` predictions.

### On D1 (10 patients) — labeled_patients.txt

| Infer Script | Model Checkpoint | Output Dir | Analysis Notebook |
|---|---|---|---|
| infer_baseline_pd.sh | baseline_best_model.pth | real_pd_predictions_baseline | analysis_abstract_3models.ipynb |
| infer_labeled_pd_run002_v2.sh | run_002_v2/ckpt_best.pth | real_pd_predictions_run002_v2 | analysis_abstract_3models.ipynb |
| infer_v5_mixed.sh | run_v5_mixed/ckpt_best.pth | real_pd_predictions_v5_mixed | analysis_abstract_3models.ipynb |
| infer_v7_realPD.sh | run_v7_realPD/ckpt_best.pth | real_pd_predictions_v7_realPD | analysis_v7_models.ipynb |
| infer_v7_mixed.sh | run_v7_mixed/ckpt_best.pth | real_pd_predictions_v7_mixed | analysis_v7_models.ipynb |

### On D1+New (17 patients) — labeled_patients_v8.txt

| Infer Script | Model Checkpoint | Output Dir | Analysis Notebook |
|---|---|---|---|
| infer_v8_baseline.sh | baseline_best_model.pth | real_pd_predictions_v8_baseline | analysis_v8_17patients.ipynb |
| infer_v8_realPD.sh | run_v7_realPD/ckpt_best.pth | real_pd_predictions_v8_realPD | analysis_v8_17patients.ipynb |
| infer_v8_pseudoPD.sh | run_002_v2/ckpt_best.pth | real_pd_predictions_v8_pseudoPD | analysis_v8_17patients.ipynb |
| infer_v8_mixed.sh | run_v7_mixed/ckpt_best.pth | real_pd_predictions_v8_mixed | analysis_v8_17patients.ipynb |
| infer_v8_clipped.sh | run_v7_mixed/ckpt_best.pth | real_pd_predictions_v8_clipped | analysis_phase2_experiments.ipynb |
| infer_v9_mixed.sh | run_v9_mixed/ckpt_best.pth | real_pd_predictions_v9_mixed | analysis_phase2_experiments.ipynb |
| infer_v10_boundary.sh | run_v10_boundary/ckpt_best.pth | real_pd_predictions_v10_boundary | analysis_phase2_experiments.ipynb |
| infer_v11_cldice.sh | run_v11_cldice/ckpt_best.pth | real_pd_predictions_v11_cldice | analysis_phase2_experiments.ipynb |

### On Control cohort (20 patients) — iu-control/pd-files, first 20 alphabetically

| Infer Script | Model Checkpoint | Output Dir | Analysis Notebook |
|---|---|---|---|
| infer_control_20pt.sh | run_v7_mixed/ckpt_best.pth | real_pd_predictions_control_20pt | TBD |

> **Note:** No patient list file — script picks first 20 `.nii.gz` from `iu-control/pd-files` sorted alphabetically. Control cohort = separate from IU dataset used in training/eval; no GT masks available.

---

## 6. Key Differences Summary

| Dimension | v5 | v7 | Phase 2 (v8–v11) |
|---|---|---|---|
| Real PD training patients | 10 (8 train, 2 val) | 15 (13 train, 2 val) | 15 (same as v7) |
| Val patients | AC2B0AA9AE767D, AC2E254F52E467 | AC0CE315D5758B, AC0CEE9C24F2B7 | AC0CE315D5758B, AC0CEE9C24F2B7 |
| Fake PD (RegGAN) | Old RegGAN, 69pt | Old RegGAN, 69pt | Old RegGAN, 69pt |
| Starting checkpoint | baseline | baseline | baseline (v9–v11) / v7_mixed ckpt (v8 infer) |
| Eval cohort | D1 (10pt) | D1 (10pt) | D1+New (17pt) |
| Best Dice | 0.769 (v5_mixed) | 0.781 (v7_mixed) | 0.784 (v8_clipped); v11_cldice wins 7/17 patients |
| Analysis notebook | analysis_abstract_3models.ipynb | analysis_v7_models.ipynb | analysis_phase2_experiments.ipynb |

---

## 7. Local GT Mask Paths

| Cohort | Local Path | Format | # Files |
|---|---|---|---|
| D1 (10 eval patients) | `~/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final/` | `.seg.nrrd` | 10 |
| New 7 eval patients | `~/Downloads/final-segmentations/` | `.seg.nrrd` | 7+ |
| D2 fine-tuning + others | `segmentation_pd/` root (in repo) | `.seg.nrrd` | 16 |

> D1 masks are NOT in the repo — they are at the AImed-lab path above
> New 7 eval masks are in `~/Downloads/final-segmentations/` — this is the correct path (NOT `~/Downloads/segmentations/` which does not exist)

---

## 8. Key Rules

- **D1 (labeled_patients.txt)** = hold-out eval only. NEVER in training.
- **D2** = real PD patients with manual meniscus labels used for fine-tuning.
- **v8 eval cohort** adds 7 new labeled patients on top of D1 — still all hold-out.
- `segmentation_runs/` = all model checkpoints, each in own subdirectory.
- `results/` = BigRed inference outputs. `results/final_results/` = local copies after SCP.
- v7 dirs (`run_v7_*`, `real_pd_seg_data_v7`) and v8 infer dirs are all new — nothing existing overwritten.
