# Experiment Version Tracking — Full Log
> BigRed200 base: `/N/project/prostate_cancer_ai/anshika/regGAN/`
> Local base: `~/Desktop/github/RegGAN_domain_adaptation_v2/`
> Last updated: 2026-08-05

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
- GT masks (new 7, local): `~/Downloads/segmentations/` (`.seg.nrrd`)
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

| Run | What changed | D1 Dice (10pt) |
|---|---|---|
| run_v3_rotation | +rotation augmentation | TBD |
| run_v3_tversky | Tversky loss | TBD |
| run_v3_elastic | +elastic deform aug | TBD |
| run_v3_photometric | +photometric aug | TBD |
| run_v3_resnet50 | ResNet50 encoder | TBD |
| run_v3_5slice | 5-slice 2.5D context | TBD |

### 4c. v4 — Real PD fine-tuning starting from run_002_v2, D2 (10 patients)

| Run | Real PD Data | Fake PD Data | Starting Ckpt | Eval Set | D1 Dice (10pt) |
|---|---|---|---|---|---|
| run_v4_realPD | real_pd_seg_data (8 train, 2 val) | — | run_002_v2 | D1 | TBD |
| run_v4_mixed | real_pd_seg_data (8 train, 2 val) | segmentation_data_v2 | run_002_v2 | D1 | TBD |

### 4d. v5 — Real PD fine-tuning starting from DESS baseline, D2 (10 patients)

| Run | Real PD Data | Fake PD Data | Starting Ckpt | Eval Set | D1 Dice (10pt) | Notes |
|---|---|---|---|---|---|---|
| run_v5_realPD | real_pd_seg_data (8 train, 2 val) | — | baseline | D1 | 0.135 | Collapsed — real PD only not enough from scratch |
| run_v5_mixed | real_pd_seg_data (8 train, 2 val) | segmentation_data_v2 | baseline | D1 | **0.769** | **Best overall on D1** |

### 4e. v7 — Real PD fine-tuning starting from DESS baseline, D2+aug3rd (15 patients)

| Run | Real PD Data | Fake PD Data | Starting Ckpt | Val Dice (2pt) | D1 Dice (10pt) | D1+New Dice (17pt) |
|---|---|---|---|---|---|---|
| run_v7_realPD | real_pd_seg_data_v7 (13 train, 2 val) | — | baseline | PENDING | PENDING | PENDING |
| run_v7_mixed | real_pd_seg_data_v7 (13 train, 2 val) | segmentation_data_v2 | baseline | **0.7644** (ep26) | PENDING | PENDING |

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

### On Control cohort (20 patients) — iu-control/pd-files, first 20 alphabetically

| Infer Script | Model Checkpoint | Output Dir | Analysis Notebook |
|---|---|---|---|
| infer_control_20pt.sh | run_v7_mixed/ckpt_best.pth | real_pd_predictions_control_20pt | TBD |

> **Note:** No patient list file — script picks first 20 `.nii.gz` from `iu-control/pd-files` sorted alphabetically. Control cohort = separate from IU dataset used in training/eval; no GT masks available.

---

## 6. Key Differences Summary

| Dimension | v5 | v7 | v8 |
|---|---|---|---|
| Real PD training patients | 10 (8 train, 2 val) | 15 (13 train, 2 val) | N/A (inference only) |
| Val patients | AC2B0AA9AE767D, AC2E254F52E467 | AC0CE315D5758B, AC0CEE9C24F2B7 | N/A |
| Fake PD (RegGAN) | Old RegGAN, 69pt | Old RegGAN, 69pt | Old RegGAN, 69pt |
| Starting checkpoint | baseline | baseline | N/A |
| Eval cohort | D1 (10pt) | D1 (10pt) | D1+New (17pt) |
| # models compared | 3 | 3 | 4 |
| Models compared | baseline, run_002_v2, v5_mixed | baseline, v7_realPD, v7_mixed | baseline, v7_realPD, run_002_v2, v7_mixed |
| Analysis notebook | analysis_abstract_3models.ipynb | analysis_v7_models.ipynb | analysis_v8_17patients.ipynb |

---

## 7. Local GT Mask Paths

| Cohort | Local Path | Format | # Files |
|---|---|---|---|
| D1 (10 eval patients) | `~/Desktop/AImed-lab/SEGMENTATIONS/PD-segmentations-final/` | `.seg.nrrd` | 10 |
| New 7 eval patients | `segmentation_pd/segmentations/` (in repo) | `.seg.nrrd` | 7 |
| D2 fine-tuning + others | `segmentation_pd/` root (in repo) | `.seg.nrrd` | 16 |

> `segmentation_pd/segmentations/` = exactly the 7 new eval patients from labeled_patients_v8.txt
> `segmentation_pd/` root = mix of D2 fine-tuning patients + some new eval patients (NOT all 17 eval)
> D1 masks are NOT in the repo — they are at the AImed-lab path above

---

## 8. Key Rules

- **D1 (labeled_patients.txt)** = hold-out eval only. NEVER in training.
- **D2** = real PD patients with manual meniscus labels used for fine-tuning.
- **v8 eval cohort** adds 7 new labeled patients on top of D1 — still all hold-out.
- `segmentation_runs/` = all model checkpoints, each in own subdirectory.
- `results/` = BigRed inference outputs. `results/final_results/` = local copies after SCP.
- v7 dirs (`run_v7_*`, `real_pd_seg_data_v7`) and v8 infer dirs are all new — nothing existing overwritten.
