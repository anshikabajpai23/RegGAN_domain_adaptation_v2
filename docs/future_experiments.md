# Future Experiment Ideas
> All experiments build on top of `run_v7_mixed` (Dice 0.781 on 17pt) as the baseline to beat.
> Base script: `segmentation/finetune_meniscus_v5_mixed.py`
> Eval: same 17-patient cohort (`labeled_patients_v8.txt`) for direct Dice comparison.

---

## Queued Experiments

### v9_mixed — Segmentation-Restricted CE
**Script:** `slurm_files/finetune_v9_mixed.sh` (ready to submit)
**Change:** Background pixels fully excluded from CE loss (`ignore_index=0`) on both fake PD and real PD sides. Previously background was included but downweighted (weight=0.1).
**Why:** Background (~90% of pixels) dilutes gradient signal from meniscus pixels even at low weight. Removing it forces 100% of CE gradient to come from meniscus pixels.
**Risk:** No CE pressure to suppress false positives in background — only Dice catches over-prediction.
**Status:** Not yet submitted.

---

### v10_boundary — Boundary-Aware Loss (on top of v7_mixed)
**Script:** TBD (`finetune_meniscus_v10_boundary.py` + SLURM)
**Change:** Keep v7_mixed loss unchanged. Add an extra boundary penalty term:
```python
def boundary_loss(logits, targets, eps=1e-6):
    # Extract GT boundary pixels via morphological erosion
    kernel = torch.ones(1, 1, 3, 3).to(targets.device)
    gt_men = (targets > 0).float().unsqueeze(1)
    eroded = F.conv2d(gt_men, kernel, padding=1) == 9
    gt_boundary = (gt_men.squeeze(1) - eroded.float().squeeze(1)).clamp(0, 1)

    # CE only at boundary pixels
    ce_per_px = F.cross_entropy(logits, targets, reduction="none")
    return (ce_per_px * gt_boundary).sum() / gt_boundary.sum().clamp(min=1.0)

# In training loop:
loss = existing_loss + λ_boundary * boundary_loss(logits, targets)
# Suggested λ_boundary = 0.5–1.0
```
**Why:** Torn meniscus predictions fragment at the tear gap — a boundary error. This directly penalizes wrong predictions at meniscus boundaries, which is exactly where tears cause failures.
**Novelty:** More novel than v9 — boundary-aware loss specific to meniscus tear context. Publishable as a contribution.
**Risk:** Additive — no existing signal removed. Low risk.
**Status:** Not yet implemented.

---

## Longer-Term Strategies (ranked by impact)

| Priority | Strategy | Targets | Novel? | Effort |
|---|---|---|---|---|
| 1 | **Fine-tune on a few labeled torn patients** | Distribution mismatch | Not novel | Low (data collection) |
| 2 | **Boundary-aware loss** (v10, above) | Boundary fragmentation at tears | Novel | Low |
| 3 | **Topology-aware loss (clDice)** | Fragmented predictions | Not novel | Medium |
| 4 | **Connected-component post-processing** | Fragmentation | Not novel | Minimal |
| 5 | **binary_fill_holes post-processing** | Tear gap = hole in prediction | Not novel | Minimal |
| 6 | **Synthetic tear augmentation** | Distribution mismatch | Not novel | Medium |
| 7 | **Two-stage coarse-to-fine** | Shape prior | Not novel | High |
| 8 | **RegGAN → pathology domain adaptation** | Torn MRI normalized before segmenting | **Novel** | High |
| 9 | **Digital twin (pre-tear counterfactual)** | Generate healthy knee, segment that | **Novel** | Very High |
| 10 | **Uncertainty-aware segmentation** | Flag low-confidence regions as potential tears | Novel in context | High |

---

## Quick Wins (No Retraining)

### Post-processing: fill_holes + component filter
```python
from scipy.ndimage import binary_fill_holes
from skimage import morphology

def clean_prediction(pred_mask, label=1, min_size=50):
    binary = pred_mask == label
    cleaned = morphology.remove_small_objects(binary, min_size=min_size)
    filled = binary_fill_holes(cleaned)
    return filled
```
Apply after inference in `infer_real_pd_v3.py`. Tear gaps appear as holes inside the meniscus boundary — `binary_fill_holes` fills them directly.

### Pre-processing: intensity clipping
```python
# Suppress hyperintense tear fluid signal before feeding to model
slice = np.clip(slice, 0, np.percentile(slice, 95))
```

---

## Notes
- All queued experiments use `real_pd_seg_data_v7` (15 patients, 13 train / 2 val).
- Starting checkpoint always: `pretrained/baseline_best_model.pth` (DESS 2.5D U-Net).
- Eval always: 17pt cohort (`labeled_patients_v8.txt`) for apples-to-apples Dice.
- v9 and v10 are A/B tests — submit one at a time to isolate the effect of each change.
