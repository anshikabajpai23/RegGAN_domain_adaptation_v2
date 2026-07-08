# RegGAN File Map
> Quick reference for every file in this repo — what it does and when to use it.

---

## Root Level

| File | What it does |
|---|---|
| `analysis_real_pd.ipynb` | Local analysis notebook — compares Baseline / V1 (55pt) / V2 (155pt) predictions against GT masks on 8 labeled real PD patients. Applies same preprocessing as pipeline (RAS reorient + 384×384 resize). Label mapping: baseline class 4 = meniscus, V1/V2 merge classes 1+2. Outputs `dice_comparison.png` and `visual_comparison.png` |
| `models.py` | RegGAN architecture — Generator, PatchDiscriminator, RegistrationNet, and all loss functions (LSGAN, smoothness, magnitude) |
| `dataset.py` | PyTorch Dataset for unpaired DESS/PD slice loading with augmentation (rotation, flip) — used by train.py |
| `train.py` | Main training loop — G/D/R optimization, image pool, TensorBoard logging, checkpoint save/resume |
| `train_stage3.py` | Byte-identical to train.py — kept separate for stage 3 experiments without touching the main training file |
| `requirements.txt` | Python package dependencies |

---

## preprocess/

| File | What it does |
|---|---|
| `preprocess.py` | Full preprocessing pipeline — reorient to RAS, isotropic in-plane resample, resize to 384×384, percentile normalize, extract slices as .npy. Does patient-level train/val/test split with leakage check. Used for both 69-patient (preprocessed_v2/) and 155-patient (preprocessed_v3/) runs |
| `preprocess_masks.py` | Same pipeline as preprocess.py but for segmentation masks — uses nearest-neighbor interpolation (order=0) to preserve integer labels. Has nibabel fallback for non-orthonormal files. Pass `--pd_dir` to copy affine from fake PD for guaranteed alignment |
| `preprocess_marks_failed.py` | Logs which mask files failed preprocessing — diagnostic only |

---

## inference/

| File | What it does |
|---|---|
| `infer2.py` | Translates DESS volumes → fake PD using trained G_AB. Handles per-slice batching, saves NIfTIs with LPS→RAS-corrected affine. Pass `--splits/--split` to restrict to one split, or omit to run on all patients |
| `evaluate.py` | Main evaluation script — FID, KID, SSIM, Jacobian determinant, meniscus deformation, overlay visualizations, intensity histograms |
| `boundary_distance_eval.py` | R-independent anatomy preservation check — Canny edges on fake PD vs DESS mask boundaries via distance transform. Does NOT rely on registration network R |
| `fid_statistical_test.py` | Bootstrap CI + permutation test on FID improvement (fake PD vs real PD vs DESS vs real PD). Uses PCA to 50 dims before resampling for speed |
| `domain_gap_embeddings_v2.py` | t-SNE + UMAP domain gap plots with random per-patient sampling. Supports `--sampling consecutive` (original artifact mode) or `--sampling random` (fixed). Train split patients only |
| `domain_gap_embeddings_v3.py` | Same as v2 but uses ALL available patients (69 or 155). Requires fake PD for all patients (run infer2.py without --splits). Red plotted on top of green (fixes v2 hidden-points issue) |

---

## segmentation/

| File | What it does |
|---|---|
| `prepare_meniscus_masks.py` | Converts 7-class DESS masks → 3-class (0=background, 1=lateral meniscus label5, 2=medial meniscus label6). Saves matched pseudo-PD slice + mask .npy pairs. Skips slices with <10 meniscus pixels |
| `dataset_2_5d.py` | 2.5D Dataset — stacks [slice i-1, slice i, slice i+1] as 3 channels, matching the pretrained model's input convention |
| `finetune_meniscus.py` | Fine-tunes pretrained SMP UNet (ResNet34 encoder, pitthexai/Knee_MRI_Segmentation_2.5D checkpoint) on 3-class meniscus task. Swaps 5-class head → 3-class. LR=1e-5, CrossEntropy loss |
| `infer_real_pd.py` | Runs the FINE-TUNED meniscus model on real PD volumes. Saves per-volume segmentation masks as NIfTIs with correct LPS→RAS affine |
| `infer_real_pd_baseline.py` | Same as infer_real_pd.py but uses the ORIGINAL pretrained 5-class checkpoint (no head swap). For before/after comparison |

---

## scripts/

| File | What it does |
|---|---|
| `analyze_training_curves.py` | Reads TensorBoard event files, plots G/D/R losses + val L1, and gives a plateau verdict (last 20% vs prior 20% trend) |
| `bigred_check_split.py` | Audits splits.json for patient overlap between train/val/test — raises error if leakage detected |
| `check_mask_integrity.py` | Loads every mask NIfTI and checks for corruption/truncation from incomplete scp transfers |
| `check_registration.py` | Confirms R produces non-zero deformation flow (validates R is not collapsed to output zeros) |
| `check_spacing.py` | Audits voxel spacing of fake PD, DESS, and real PD NIfTIs — checks effective spacing after preprocessing matches expected 1.00× ratio |
| `reconstruct_volume_from_slices.py` | Re-assembles per-slice .npy files back into a 3D NIfTI volume, handling gaps from skipped background slices |
| `resample_to_pd_spacing.py` | Resamples fake PD outputs to match real PD voxel spacing — needed for downstream segmentation model consistency |
| `stage0_preflight.py` | Pre-flight audit of the full pipeline — checks data state, split correctness, spacing metadata, and code integrity before a training run |
| `stage0_visualize_split.py` | Plots patient counts and overlap stats from bigred_check_split.py output — quick visual split sanity check |
| `plot_finetune_curves.py` | Parses finetune_meniscus.py stdout log and plots train/val loss + per-class Dice curves over epochs. Used to assess if more epochs/data are needed |

---

## slurm_files/

| File | What it does |
|---|---|
| `bigred_submit.sh` | Submit main RegGAN training job to BigRed200 GPU queue |
| `bigred_submit_int.sh` | Same but for interactive GPU partition |
| `stage1_preprocess.sh` | Run preprocess.py on BigRed — outputs to preprocessed_v2/ |
| `stage2_verify_resume.sh` | Verify checkpoint resume works correctly (scheduler + step restoration) |
| `stage3_verify_train_stage3.sh` | Run train_stage3.py for R-network stage 3 experiments (CPU) |
| `stage3_verify_train_stage3_gpu.sh` | Same as above but on GPU partition |
| `stage4_evaluate.sh` | Run evaluate.py on BigRed for FID/SSIM/Jacobian metrics |
| `stage5a_resample.sh` | Run resample_to_pd_spacing.py on BigRed |
| `stage7_retrain.sh` | Full retrain from scratch (stage 7) |
| `stage7_resume.sh` | Resume stage 7 training from latest checkpoint |
| `stage7_interim_validation.sh` | Run validation metrics mid-training during stage 7 |
| `evaluate.sh` | General evaluate.py submission (CPU) |
| `evaluate_gpu.sh` | General evaluate.py submission (GPU) |
| `reggan_infer.sh` | Run original infer.py (v1 inference, kept for reference) |
| `reggan_infer2.sh` | Run infer2.py (current inference with LPS→RAS fix) |
| `run_preprocess_masks.sh` | Run preprocess_masks.py on BigRed |
| `fid_baseline.sh` | Compute baseline FID (DESS vs real PD, before translation) |
| `fid_statistical_test.sh` | Run fid_statistical_test.py — bootstrap CI + permutation test |
| `finetune_meniscus.sh` | Fine-tuning pipeline for 69-patient run — prepare_meniscus_masks.py → finetune_meniscus.py → segmentation_data/ + segmentation_runs/run_001/ |
| `finetune_all155.sh` | Fine-tuning pipeline for 155-patient run — prepare_meniscus_masks.py → finetune_meniscus.py → segmentation_data_v2/ + segmentation_runs/run_002/ |
| `infer_all69.sh` | Run infer2.py on all 69 DESS patients (no split filter) → stage4_fake_pd_all69/ |
| `infer_all155.sh` | Run infer2.py on all 155 DESS patients (full SKM-TEA) → fake_pd_all155/ |
| `preprocess_all155.sh` | Run preprocess.py on full 155-patient SKM-TEA dataset → preprocessed_v3/ |
| `infer_labeled_pd.sh` | Run infer_real_pd.py on specific PD patients listed in labeled_patients.txt → real_pd_predictions_v2/ |

---

## Key Output Directories (on BigRed)

### 69-patient run (original subset)
| Path | Contents |
|---|---|
| `preprocessed_v2/slices/dess/` | Per-slice .npy files — 69 DESS patients |
| `preprocessed_v2/slices/pd/` | Per-slice .npy files — 69 PD patients |
| `preprocessed_v2/masks/` | Preprocessed segmentation mask NIfTIs |
| `preprocessed_v2/splits.json` | Patient-level train/val/test split (55/7/7) |
| `results/stage4_fake_pd/` | Fake PD NIfTIs — train split (55 patients) |
| `results/stage4_fake_pd_all69/` | Fake PD NIfTIs — all 69 DESS patients |
| `runs/run_004/` | RegGAN training — checkpoints (ckpt_best.pt), TensorBoard, eval outputs |
| `segmentation_data/` | Pseudo-PD + 3-class mask .npy pairs — 69-patient fine-tuning |
| `segmentation_runs/run_001/` | Fine-tuning checkpoints — pilot run (~0.49 Dice) |

### 155-patient run (full SKM-TEA)
| Path | Contents |
|---|---|
| `preprocessed_v3/slices/dess/` | Per-slice .npy files — 155 DESS patients |
| `preprocessed_v3/slices/pd/` | Per-slice .npy files — 69 PD patients (re-preprocessed) |
| `preprocessed_v3/masks/` | Preprocessed segmentation mask NIfTIs — 155 patients |
| `preprocessed_v3/splits.json` | Patient-level train/val/test split (~124/15/15) |
| `results/fake_pd_all155/` | Fake PD NIfTIs — all 155 DESS patients |
| `segmentation_data_v2/` | Pseudo-PD + 3-class mask .npy pairs — 155-patient fine-tuning |
| `segmentation_runs/run_002/` | Fine-tuning checkpoints — full run (**best Dice: 0.8341**, lateral=0.830, medial=0.838) |

### Real PD inference
| Path | Contents |
|---|---|
| `results/real_pd_predictions/` | Fine-tuned model predictions on 8 labeled PD scans (run_001) |
| `results/real_pd_predictions_baseline/` | Baseline (non-fine-tuned) predictions on same 8 scans |
| `results/real_pd_predictions_v2/` | Fine-tuned model predictions using run_002 checkpoint |

---

## Pipeline Order (end-to-end)

```
1. preprocess/preprocess.py           → slices + splits.json
2. train.py                           → ckpt_best.pt  (RegGAN domain adaptation)
3. inference/infer2.py                → fake PD NIfTIs  (omit --splits for all patients)
4. preprocess/preprocess_masks.py     → mask NIfTIs  (pass --pd_dir for affine alignment)
5. inference/evaluate.py              → FID, SSIM, Jacobian metrics
6. inference/boundary_distance_eval.py → anatomy preservation check (R-independent)
7. inference/fid_statistical_test.py  → statistical significance of FID improvement
8. inference/domain_gap_embeddings_v3.py → t-SNE + UMAP domain gap visualization
9. segmentation/prepare_meniscus_masks.py → 3-class mask .npy pairs
10. segmentation/finetune_meniscus.py → fine-tuned segmentation model
11. segmentation/infer_real_pd.py     → predictions on real PD
12. [eval_dice script — pending]       → Dice vs ground truth on labeled PD
```

## Results Summary

| Run | Patients | Best Dice (mean meniscus) | Checkpoint |
|---|---|---|---|
| run_001 (pilot) | 69 | 0.49 | `segmentation_runs/run_001/ckpt_best.pth` |
| run_002 (full) | 155 | **0.8341** | `segmentation_runs/run_002/ckpt_best.pth` |
