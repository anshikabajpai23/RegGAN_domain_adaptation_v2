# Phase 3 — Diagnosis-Driven Experiments

> **Baseline to beat:** `run_v7_mixed` — Dice 0.781 on 17pt (`labeled_patients_v8.txt`)
> **Base script:** `segmentation/finetune_meniscus_v7_replica.py` (original v7/v8 loss + instrumentation)
> **Eval:** same 17-patient cohort, for direct comparison with everything in `experiment_version_tracking.md`
> **Last updated:** 2026-09-09

**What changed vs Phase 2:** Phase 2 tried eleven loss/architecture/post-processing
variants and every one landed within ±0.03 of 0.78. Phase 3 starts from an actual
diagnosis instead of a hypothesis list.

---

## 1. Diagnosis (from the instrumented replica run)

Source: `segmentation_runs/run_v7_replica_seed42/` — seed 42, original v7/v8 loss,
train Dice logged for the first time (it had been computed as `td` and discarded
in an f-string in every previous run).

### 1a. Overfitting — confirmed

| Signal | Value |
|---|---|
| train Dice (real PD) | 0.9456 |
| val Dice (real PD) | 0.7793 |
| **generalization gap** | **+0.166** |
| train loss | 0.263 → 0.209 (falling) |
| val loss | 1.016 → 1.017 (flat from epoch ~4) |

Not underfitting, not at ceiling. The model has ample capacity — it reaches 0.946
on data it has seen. This retroactively explains `v3_resnet50` (0.639): a bigger
encoder memorizes faster.

### 1b. Domain gap is small — RegGAN is not the bottleneck

```
fake-PD val Dice = 0.760
real-PD val Dice = 0.781
```

The model generalizes about equally poorly to unseen *synthetic* and unseen *real*
data. If translation realism were the binding constraint, held-out fake PD would
score far higher than held-out real PD. It doesn't.

Consistent with the older observation that `run_005` (best FID, 147.6) gave *worse*
downstream Dice (0.640) than the old RegGAN (FID 164.4 → 0.695).

### 1c. It is a DETECTION problem, not a segmentation problem

From `val_predictions/per_slice_dice.csv` (2 val patients, 62 slices):

```
slices with NO meniscus in GT : 32
  model correctly silent      : 23
  model HALLUCINATED          :  9   -> 2,001 voxels = 46% of ALL false positives
  worst single slice          : 538 voxels invented where there is nothing

slices that DO contain meniscus: 30
  mean Dice                   : 0.7919
  median Dice                 : 0.844
  slices below 0.5            : 2 of 30

FN total 1,682   FP total 4,373   ->  FP is 2.6x FN
```

**On slices where the meniscus exists, the model is already near the geometric
ceiling** (~0.83 for a 1px boundary offset on a ~6px-thick structure). The Dice is
being lost on slices that should be empty.

Oracle upside from removing only the empty-slice hallucinations:

```
current : 2*9170 / (13543 + 10852) = 0.752
oracle  : 2*9170 / (11542 + 10852) = 0.819     (+0.067)
```

### 1d. Checkpoint selection is arbitrary

- 2 validation patients; the plateau spans only 0.755–0.781 across 17 epochs
- Two metrics correlating **+0.967** still selected epochs **10 and 19**
- Pooled `vd` runs ~0.028 above the clean per-patient mean (cycling-iterator artifact
  in `run_epoch` — the small real-val set is traversed repeatedly, weighting slices unevenly)
- Seed noise: this replica peaked 0.7809 @ ep10; original v7_mixed peaked 0.7644 @ ep26

**Implication:** every Phase-2 difference smaller than ~0.02 was unmeasurable.
That covers v8_clipped (0.784), v8_filled (0.778) and v11_cldice (0.777) vs 0.781.

---

## 2. Root cause of the hallucination

`segmentation/prepare_meniscus_masks.py:129`

```python
if (collapsed > 0).sum() < args.min_meniscus_pixels:   # default 10
    total_skipped_empty += 1
    continue
```

**The fake-PD training set contains only meniscus-bearing slices — zero negatives.**
Fake PD is the stream driving `run_epoch`'s loop, so for the bulk of training the
model is never shown a slice whose correct answer is "nothing here."

Real PD prep (`prepare_real_pd_labeled.py:151`) filters on *image* emptiness
(`img_sl.mean() < 0.02`), not mask content — so it does supply negatives, but only
from 13 patients in the minority stream.

**This also explains v9 and v10 both landing on exactly 0.753.** Both further reduced
background supervision on a training set that already had almost none:
- v9 — `ignore_index=0` removed background from CE entirely
- v10 — boundary loss concentrated learning on the meniscus rim

`future_experiments.md` called this risk correctly for v9 before it ran:
*"No CE pressure to suppress false positives in background — only Dice catches
over-prediction."*

---

## 3. Candidate experiments, ranked

Sources tagged: **[Claude]** = surfaced during Phase 3 diagnostics, **[prof]** = advisor suggestion.

| # | Experiment | Source | Targets | Effort | Evidence |
|---|---|---|---|---|---|
| 1 | **Include meniscus-free slices in fake-PD prep** | **[Claude]** | 1c hallucination | Minimal (a flag) | Strong — +0.067 oracle |
| 2 | **Image crop → train on cropped only** | [prof] | 1a overfit, in-plane FP | Medium | Good |
| 3 | **Self-training on ~54 unlabeled real PD** | [Claude] | 1a overfit | Medium | Good |
| 4 | **k-fold over the 15 real PD patients** | [Claude] | 1d selection noise | Low | Strong |
| 5 | **Weight averaging over last N epochs** | [Claude] | 1d selection noise | Minimal | Moderate |
| 6 | Knowledge distillation | [prof] | — | Medium | Weak — not capacity-limited |
| 7 | Contrast change / basic image processing | [prof] | inter-patient variation | Low | Weak — v8_clipped gave +0.003 |
| 8 | Meniscus shape-based training | [prof] | implausible blobs | Medium | Weak — v11_cldice ≈ no change |
| 9 | Image contour approach | [prof] | boundary precision | Medium | Against — already near ceiling |
| 10 | Diffusion instead of fine-tuning | [prof] | image realism | **Very high** | Against — see 1b |
| 11 | FMed paper | [prof] | — | — | Unassessed |
| 12 | Knee cartilage segmentation review | [prof] | — | Low | Orientation reading |

### Experiment 1 — detail

**Change:** re-run `prepare_meniscus_masks.py` with `--min_meniscus_pixels 0`, or
better, keep a controlled fraction of negative slices (e.g. all meniscus-bearing
slices + N empty slices per volume) to avoid swamping the positives.

**Why:** the model currently has no training signal for "predict nothing here."

**Risk:** too many negatives re-introduces the class-imbalance problem that the
weighted CE (bg=0.1) was built to solve. The negative fraction is a hyperparameter,
not a free win — worth a small sweep (0%, 25%, 50%, 100% of available empty slices).

**Note:** this changes `segmentation_data_v2/`, so it needs a new data dir
(e.g. `segmentation_data_v3_withneg/`) rather than overwriting the one every
previous run used.

---

## 4. Standing caveats

- Findings 1a–1d come from **2 validation patients / 62 slices**. Confirm the
  hallucination pattern against the 17-patient predictions already local in
  `results/final_results/real_pd_predictions_v8_mixed/` before committing to a retrain.
- `finetune_meniscus_v5_mixed.py` is now the **v9** restricted-CE variant, NOT v7/v8.
  The original v7/v8 loss lives in commit `64b1020` and in
  `finetune_meniscus_v7_replica.py`. Do not use v5_mixed to replicate v7/v8.
- Report per-patient mean ± CI, not a bare Dice. With 17 patients and a per-patient
  spread of ~0.05, SEM is ~0.012.

---

## 5. Status

| Item | Status |
|---|---|
| v7 replica, seed 42 | Done — 0.7809 @ ep10 |
| v7 replica, seed 7 (noise floor) | Not submitted |
| Confirm hallucination on 17pt cohort | Not started |
| Inspect the 9 hallucination slices in Slicer | Not started |
| Experiment 1 (negatives in fake-PD prep) | Not started |

### Tooling built in Phase 3

| File | Purpose |
|---|---|
| `segmentation/finetune_meniscus_v7_replica.py` | v7/v8 loss + train Dice, per-patient val Dice, val prediction dump, seed |
| `segmentation/diagnose_checkpoint.py` | Read-only diagnostic on an existing checkpoint (no retraining) |
| `scripts/plot_replica_diagnostics.py` | `diagnostics_curves.png` + `diagnostics_per_slice.png` |
| `slurm_files/finetune_v7_replica.sh` | Seed-suffixed output dirs + overwrite guard |
| `slurm_files/diagnose_v7_mixed.sh` | 30-min read-only diagnostic job |
