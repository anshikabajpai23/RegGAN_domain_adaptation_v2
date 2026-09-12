# RegGAN Meniscus Segmentation — Dice 0.78 Debug Log

**Started:** 2026-09-09
**Source doc:** `~/Downloads/DEBUG_HANDOFF.md`
**Goal:** Find the bottleneck (data / processing / modeling / domain adaptation / fine-tuning) before touching any fix.
**Constraint:** Debugging must be fast. No retraining until the bottleneck is confirmed.
**Local state:** Nothing from the project is on this Mac. All checks run on BigRed (`/N/project/prostate_cancer_ai/anshika/regGAN/`).

---

## Working hypothesis

The bottleneck is the real-PD fine-tuning data, not the model and not the GAN.

Evidence from the existing log:
- Adding fake PD to real-PD training changes nothing (v4: 0.775 vs 0.759, v7: 0.774 vs 0.781). Synthetic data only matters as a warm start.
- Per-patient spread 0.69 to 0.84, five times larger than any experiment effect.
- Train Dice 0.946 on the 13 real patients vs 0.779 val. Memorized.

## Doc claims NOT trusted until verified

| Claim | Where | Why not trusted |
|---|---|---|
| Domain gap is not the bottleneck | §6.2 | Fake-val and real-val are different patients, and model trained on real PD. Confounded (§17.7 admits). |
| Hallucination is the root cause | §7 | 9 slices from 2 patients. Could be annotation-edge ambiguity, not hallucination. |
| v3_5slice 0.486 is a bug | §17.6 | May be a real signal of 2.5D neighbor-spacing mismatch. |
| Near geometric ceiling on meniscus slices | §6.3 | Mean 0.79 not 0.844, only 30 slices. |

## Finding not in the doc

**2.5D neighbor spacing differs 4.5× between training streams.** Fake-PD stacks use DESS neighbors 0.8mm apart (channels nearly identical to center). Real-PD neighbors are 3.6mm apart. Model learns from the dominant fake stream that context channels are copies of the center. Would also explain the 5-slice collapse (outer channels 7.2mm away on real PD).

---

## Debug plan

| # | Check | What it decides | Why it matters | How (where, what to run) | Time | Status |
|---|---|---|---|---|---|---|
| 1 | **Per-patient error decomposition, 17 patients** | Is the empty-slice hallucination real, or annotation-edge ambiguity? Does resolution (384 vs 768) or cohort (D1 vs new) explain the 0.69–0.84 spread? | Doc's #1 open item. Retrain decision depends on it. 2 patients is not enough evidence. | BigRed. Paste `ls` + array shapes from `results/final_results/real_pd_predictions_v8_mixed/` and `real_pd_seg_data_v7/`. Claude writes script. Run, paste table. | 2 hrs | 🔶 Partially done on fake-val (15 pts) + real-val (2 pts) CSVs. Reduced ask for 17 pts: full Dice, core-only Dice, halluc count + distance. |
| 2 | **Neighbor-spacing test** | Do the 2.5D context channels help, hurt, or get ignored on real PD? | Fake-PD neighbors 0.8mm, real-PD 3.6mm. Never tested. May explain 5-slice collapse. | BigRed. Inference only with `infer_real_pd_v3.py`, stacks (i,i,i) instead of (i-1,i,i+1). Claude writes patch. Compare Dice to 0.781. | 30 min GPU | ⬜ Not started |
| 3 | **Real-PD cycle count** | How many times each real slice is seen per epoch | If ≥15×, memorization is structural. Fix is sampling ratio, not loss. | BigRed, one command (below). | 5 min | ⬜ Not started |
| 4 | **Label ceiling** | Is 0.78 near what two humans get against each other? | If inter-annotator Dice ~0.80, at ceiling, target changes. Doc never mentions it. | Ask annotators: any PD volume labeled twice? If yes, Dice between the two masks. | Depends | 🔶 Moved up. Slicer showed GT stops early / starts late at the ends on SKM-TEA too. Reduced ask: annotator + stopping rule for real-PD GT; human call on 8 specific edge slices (see Findings). |
| 5 | Seed-7 replica | Noise floor between runs | Needed before believing any fix, not for finding the bottleneck | BigRed, 4 hr job | 4 hrs | ⏸ Skip for now |
| 6 | **Intensity statistics per patient** | Does the Dice spread track image contrast or noise? Does real PD sit outside the fake-PD intensity range? | v8_clipped +0.003 is weak evidence on one axis only. Contrast never tested directly. If it drives the spread, image processing helps and no retrain is needed. | Notebook. Per patient on preprocessed images: mean, std, p1/p50/p99, meniscus mean, 5px ring mean, CNR = (ring − men) / std, noise. Same for 10 fake-PD volumes. Correlate Dice vs CNR. Cheap version: side-by-side of 3 worst vs 3 best patients, same window. | 1 hr | ⏸ Deprioritized. Nothing in Slicer or the CSVs pointed at contrast. Error pattern is geometric (through-plane ends), not intensity. |

**Order (updated 2026-09-10):** real-PD edge slices in Slicer → check 3 numbers → 17-pt core-only Dice → check 2.

## Conclusion and results (2026-09-10)

All gains are **expected, not verified**. Seed noise 0.017; anything under 0.02 is unmeasurable until check 5.

### Checks

| # | Check | Status | Finding | Fixable? | Expected gain | Approach |
|---|---|---|---|---|---|---|
| 1 | Error location | ✅ | Core at ceiling (0.84). Error at z-ends. Cohort-wide FN 47%, halluc 17%. | Partly (half labels) | +0.02–0.05 | P2 + P4 |
| 2 | Neighbor slices | ✅ | Model has no z-context; wider neighbors hurt 17/17. | Yes | +0.02–0.04, largest single lever | **P4** |
| 3 | Real-PD repeats | ✅ cause unproven | 21.6×/epoch, 562 by best epoch. | Yes | 0–0.01 (v4_realPD evidence) | P1 |
| 4A | Label stopping | ✅ | Cliff outer (0.93), ramp inner (0.39). No rule. | Partly | 0–0.03 pending 4A2 | P3 |
| 4A2 | Cliff real? | ⬜ radiologist | | decides P3 | | |
| 4B | Ceiling | ⬜ Nikhil | | decides target | | |
| 5 | Seed noise | ⏸ | 0.017 (one point) | not a fix | | needed to judge fixes |
| 6 | Contrast | dropped | geometric error | no | 0 | |
| Split | Chunks | ✅ | 0.6% of real-PD core error | not needed | 0 | |

### Candidate approaches

| Approach | Useful? | Why |
|---|---|---|
| Negatives in fake-PD prep | **Yes** | Adjacent + notch first. +0.02–0.03, not +0.067. |
| Localize/crop | Later | Targets in-plane detail = 0.6% of real-PD error. |
| Self-training on 54 unlabeled | Later | Pseudo-labels copy end errors. After ends fixed. |
| k-fold | **Yes, measurement** | Real CI and checkpoint choice. |
| Weight averaging | **Yes, free** | +0.005–0.01. |
| Knowledge distillation | No | Not capacity-limited. |
| Contrast / image processing | No | Error is geometric. |
| Shape prior | Only in 3D | 2D can't use it. |
| Contour | No | Core at ceiling. |
| Diffusion | No | Translation not limiting. |
| FMed 3D nnU-Net + adversarial | **Yes, big step** | Only arch with z-context. After P4. |
| Review paper | reading | |
| RegGAN rigid/SDM | No for Dice | Methods novelty only. |
| Tversky / over-segment | No | Model under-segments cohort-wide (FP/FN 1.1). |
| 3D CC post-proc | No | Overshoot is connected to body. |
| (0,0,0) inference | No | Coin flip per patient. |
| Flip TTA | **Yes, free** | +0.005–0.01. |
| Equalize aug branches | Hygiene | ~0, removes confound. Do with P4. |

### Conclusions

| # | Conclusion | How |
|---|---|---|
| 1 | Error at z-ends | per-slice split, 2 real + 15 fake, Slicer |
| 2 | Core at ceiling | core Dice 0.84 vs ~0.83 geometric |
| 3 | No z-context | variant C −0.022 on 17/17; B ≈ 0 |
| 4 | No label stopping rule | outer 0.93 / inner 0.39, all 15; Slicer calls; team-annotated |
| 5 | No cohort-wide over-segmentation | FP/FN 1.1 on 17 |
| 6 | Hallucination 17%, not 46% | 17-pt vs 2-pt decomposition |
| 7 | GAN not bottleneck | same failure fake/real; better FID → worse Dice; real-only ≈ mixed |
| 8 | Memorization structural, maybe not limiting | 21.6 cycles; v4_realPD 0.775 |
| 9 | Splits minor on real PD | 0.6%; cc match 26/30 |
| 10 | Phase 2 was noise | seed 0.017 vs effects ≤0.03 |
| 11 | Hallucination is near GT in both domains | 89% ≤3.6mm real, 68% fake (C1 corrected) |

### Next steps

| # | Step | Cost | Gives |
|---|---|---|---|
| 1 | 4A2 radiologist, 30 slices | 20 min | cliff anatomy vs habit → P3 go/no-go |
| 2 | 4B Nikhil, 2 volumes | 2–3 h | ceiling → realistic target |
| 3 | Edge/core split on 17-pt A and B CSVs | 10 min | cohort-wide end share; where B's gain lives |
| 4 | Notch-gap px counts, 2 single-span pts | 5 min | closes C3 |
| 5 | Seed-7 replica | 4 h GPU bg | noise floor |
| 6 | k-fold (≥3 folds) | 3×4 h | CI + checkpoint |
| 7 | **P4** stride 4–5 + equalize aug, one retrain | prep + 4 h | does z-context cut end error; case for 3D |
| 8 | **P2** negatives adjacent+notch, new dir, one retrain | prep + 4 h | halluc drop, +0.02–0.03 |
| 9 | **P1** sampler cap, one retrain | 4 h | closes exposure question |
| 10 | Weight avg + flip TTA | 30 min | +0.01–0.02 free |
| 11 | 3D nnU-Net (FMed) if P4 helps | days | arch that sees the ends |
| 12 | **P3** stopping rule + GT end cleanup on 13 training pts, one retrain — **only if 4A2 says the cliff is habit** | annotation + 4 h | label half of the end error, 0–0.03 |

Realistic: P4 + P2 + P1 + free wins → **0.78 → 0.81–0.83**, capped by the ceiling from step 2.

### Next-step dependencies

| # | Step | Depends on | Start now? | Why |
|---|---|---|---|---|
| 1 | Radiologist 30 slices | — | yes | |
| 2 | Nikhil re-annotation | — | yes | |
| 3 | Edge/core split, 17-pt A+B | — | yes | |
| 4 | Notch-gap counts | — | yes | |
| 5 | Seed-7 replica | — | yes | unattended 4 h |
| 10 | Weight avg + flip TTA | — | yes | inference only |
| 6 | k-fold | 5 | after 5 | seed needed to read k-fold |
| 7 | P4 stride + equalize aug | 5 | after 5 | need noise floor to judge |
| 9 | P1 sampler cap | 5 | after 5, parallel with 7 | different code path |
| 8 | P2 negatives | 7 | after 7 | same prep pipeline; train as separate one-change runs |
| 12 | P3 GT ends | 1 | only if cliff = habit | annotation wasted otherwise |
| 11 | 3D nnU-Net | 7 | only if P4 cuts end error | days; P4 is the cheap test |

**Batches:** now → 1, 2, 3, 4, 5, 10 in parallel. After 5 → 7 ∥ 9, then 6. After 7 → 8, and 11 if 7 helped. After 1 → 12 if habit.

## Phase 4 steps (hand-off spec, 2026-09-10)

Rules that apply to every step:
- **Never train on, select checkpoints on, or tune against the 17-patient test set.** Inference-only evaluation on it is Anshika's call per step.
- **One change per retrain.** Base recipe = `segmentation/finetune_meniscus_v7_replica.py`, seed 42, so every run is comparable to `run_v7_replica_seed42` (0.7809 @ ep10 on the 2 val patients).
- Report per-patient Dice on the 2 real-val patients **and** the fake-val set, plus train Dice, plus the per-slice CSV dumps (same format as `val_predictions/per_slice_dice.csv`).
- Report **results only**, no interpretation, in a `docs/phase4_<step>.md` with commands run and file paths. Numbers must come from script stdout, not typed (see Corrections C2).
- Any comparison across cohorts with different slice thickness must be in **mm**, never slice counts (Corrections C1).

### Batches

| Batch | Steps | Condition to start |
|---|---|---|
| **A — now, all in parallel** | 4.1, 4.2, 4.3, 4.4, 4.5, 4.10 | none |
| **B — after 4.5 (seed-7) finishes** | 4.7 ∥ 4.9, then 4.6 | noise floor known |
| **C — after 4.7 (P4) finishes** | 4.8; and 4.11 only if 4.7 reduced end error | P4 result known |
| **D — after 4.1 (radiologist) answers** | 4.12 only if the cliff is annotation habit | 4A2 answered |

### 4.1 Radiologist yes/no on the 30 cliff+1 slices

- **Goal:** decide whether the abrupt outer end of the GT is anatomy or annotators stopping early.
- **Input:** the 15 training patients' real PD volumes (not masks) and the slice list in the Step 4 scope table (span1.first − 1, span2.last + 1). Indices are in the `analyze_gt_tapering.py` orientation; confirm alignment by checking that span1.first shows the first labeled slice.
- **Method:** show each slice **without any overlay**. Ask: "Is there meniscus tissue on this slice?" Record yes / no / can't tell. Do not show the GT or the model prediction.
- **Output:** 30-row table: patient, slice, end (outer-before / outer-after), answer. Summary: % yes, % no, % can't tell. Who answered, date.
- **Reads as:** mostly "no" → cliff is anatomy, model overshoots it, P2/P4 are the fix, skip 4.12. Many "yes" → training GT stops early, run 4.12.

### 4.2 Blind re-annotation, 2 volumes (Nikhil)

- **Goal:** the label ceiling: Dice between two humans on the same volume.
- **Input:** preprocessed PD volumes for AC0CE315D5758B and AC0CEE9C24F2B7. **Do not give the mask files.**
- **Method:** Nikhil labels from scratch with the same instructions the original annotator had (none, if none). Then run `scripts/analyze_annotator_agreement.py` old vs new.
- **Output:** dice_full, dice_core, dice_edge, span_offset per span, for both volumes. Also the model's per-slice errors on those same slices (already in `val_predictions/per_slice_dice.csv`) side by side with the human disagreement per slice.
- **Reads as:** dice_full is the ceiling. dice_edge vs dice_core says whether the ends are where humans disagree.

### 4.3 Edge-vs-core split on the 17-pt A and B per-slice CSVs

- **Goal:** cohort-wide share of error at the meniscus ends; whether variant B's gain lives at the ends.
- **Input:** per-slice CSVs for variants A and B from `scripts/per_slice_from_predictions.py` (already built, check 2 follow-up).
- **Method:** per patient, spans = contiguous GT-nonempty runs. Buckets: **empty** (GT = 0), **edge** (GT > 0 and within ±1 slice of a span end), **core** (other GT > 0). Sum FP and FN per bucket. Same for A and B.
- **Output:** per-variant table: bucket, FP px, FN px, share of total wrong px. Per-patient table for B − A by bucket.
- **Reads as:** replaces the 2-patient "75% at ends" with the 17-patient number. If B's FP reduction is in the edge bucket and its FN increase is in core, the neighbors were causing end overshoot.

### 4.4 Notch-gap pixel counts, 2 single-span patients

- **Goal:** close correction C3: is the single span a stray voxel bridge?
- **Input:** GT masks for AC13300201B926 and AC149BC218E75C, loaded exactly as in `analyze_gt_tapering.py`.
- **Method:** print GT pixel count per slice across the whole volume. Also count 3D connected components (26-connectivity) and their sizes.
- **Output:** per-slice count table; component count and sizes.
- **Reads as:** any slice inside the expected notch with < 10 px → stray voxel bridge; add `--min_span_pixels 10` to `analyze_gt_tapering.py` and re-run Table 2. Otherwise the notch was genuinely labeled through.

### 4.5 Seed-7 replica

- **Goal:** noise floor between two identical runs.
- **Method:** `finetune_meniscus_v7_replica.py` with seed 7, nothing else changed. Existing SLURM script for the replica with the seed flag changed. 4 h.
- **Output:** per-epoch train Dice, fake-val Dice, real-val per-patient Dice; best epoch; per-slice dumps. Table: seed 42 vs seed 7 per val patient at each run's best epoch.
- **Reads as:** |Δ| between seeds is the floor. Any later fix must beat it.

### 4.6 k-fold over the 15 real patients

- **Goal:** checkpoint selection and a confidence interval that isn't driven by 2 patients.
- **Method:** 5 folds × 3 val patients, same recipe as replica (fake-PD stream unchanged). Each fold trains once. Record per-patient val Dice at the fold's best epoch.
- **Output:** 15 per-patient Dice values (each patient held out once), mean ± SD, and the best-epoch spread across folds.
- **Reads as:** SD across 15 is the real per-patient noise. Best-epoch spread says how arbitrary selection is.
- **Cost:** 5 × 4 h. Start after 4.5.

### 4.7 P4: fake-PD 2.5D stride 4–5 + equalize augmentation (one retrain)

- **Goal:** give the model neighbors that look like real 3.6mm neighbors.
- **Prep change** (`segmentation/prepare_meniscus_masks.py`, new dir `segmentation_data_v3/`, never overwrite v2): save **every** slice image of every volume from `results/fake_pd_all155` (not only mask-bearing ones), plus masks for all slices (empty mask where no meniscus). Write two index files: `train_index_posonly.txt` = mask-bearing slices only (for 4.7), `train_index_withneg.txt` = mask-bearing + selected negatives (for 4.8). Keep `preprocessed_v3/splits.json` patient split.
- **Loader change** (`dataset_2_5d_v2.py`): neighbor offset k drawn per sample from {4, 5} (3.2–4.0 mm, brackets 3.6). Stack = [i−k, i, i+k]. If a neighbor index is outside the volume, repeat the nearest valid slice (same rule `infer_real_pd_v3.py` uses at volume ends, check and match it).
- **Augmentation equalization** (`dataset_2_5d_realpd.py`): make the real branch identical to the fake branch: multiplicative brightness U(0.8, 1.2) at p = 1.0, Gaussian noise σ = 0.02 at p = 1.0, clip after each op. **Decision: real matches fake, not the reverse**, because fake is the recipe that produced 0.78. Say the word to flip it.
- **Train:** replica recipe, seed 42, `train_index_posonly.txt`.
- **Output:** standard report + per-slice dumps + 4.3-style edge/core split on the 2 val patients.
- **Reads as:** end-bucket error drops and total Dice ≥ replica + noise floor → z-context matters, go to 4.11. No change → neighbors are not the lever, skip 4.11.

### 4.8 P2: negative slices in fake-PD training (one retrain)

- **Goal:** teach the model "nothing here."
- **Prep:** uses `segmentation_data_v3/` from 4.7. Build `train_index_withneg.txt`: all mask-bearing slices + negatives chosen as: every empty slice within ±5 DESS slices (±4 mm) of each span end, every empty slice in the notch between the two spans, plus a random 10% of the remaining far-empty slices. Log the resulting positive:negative ratio.
- **Train:** replica recipe, seed 42, **stride 1 (original), original augmentation** so that only the negatives change vs replica. (If 4.7 already won, run a second variant with stride + negatives, labeled separately.)
- **Output:** standard report + empty-slice FP count on the 2 val patients vs replica's 9 slices / 2,001 px.
- **Reads as:** empty-slice FP down and core Dice not down → success. Core Dice down → class imbalance regression; try 5% far-negatives or raise the meniscus CE weight.

### 4.9 P1: sampler cap on real PD (one retrain)

- **Goal:** test whether 21.6× per-epoch exposure of real PD limits val Dice.
- **Change** in `run_epoch` of the replica script: instead of refilling `real_iter` on StopIteration, draw one real batch every `ceil(n_fake_batches / n_real_batches)` = every 22 fake batches, so each real slice is seen once per epoch, spread across the epoch. All else identical.
- **Train:** seed 42. Watch: early stopping may trigger later since real signal per epoch is 22× smaller; keep patience 10 but allow 100 epochs.
- **Output:** train Dice, val Dice, epoch of val plateau, vs replica.
- **Reads as:** gap shrinks and val rises → exposure was limiting. Gap shrinks, val flat → symptom only, drop P1.

### 4.10 Free wins: weight averaging + flip TTA (inference only)

- **Weight averaging:** if per-epoch checkpoints exist for the replica run, average the state_dicts of the last 5 epochs before the best epoch ± 2 (report which); if only ckpt_best/ckpt_last exist, average those two and note it. Evaluate on the 2 val patients + fake val.
- **Flip TTA:** in `infer_real_pd_v3.py`, predict on x, hflip(x), vflip(x), hvflip(x); un-flip the softmax maps; average; argmax. Evaluate on the 2 val patients.
- **Output:** table: baseline, weight-avg, TTA, both. Per patient.
- **Reads as:** anything ≥ +0.01 keeps. Below noise floor, still keep TTA (no downside) but don't count it.

### 4.11 3D nnU-Net (FMed route) — only if 4.7 helped

- **Goal:** the architecture that actually sees the ends.
- **Outline:** nnU-Net v2, `3d_fullres`. Resample all fake-PD volumes to 3.6 mm through-plane (every 4.5 DESS slices → use sitk resample, not slice picking) so fake and real share one spacing. Stage 1: train on 155 fake volumes. Stage 2: fine-tune on 13 real, val on 2 (mirror the v7 recipe as far as nnU-Net allows). Adversarial shape prior from the paper: **not in the first pass**; add only if notch fill persists.
- **Output:** same reports. Also per-slice dumps so 4.3-style split is comparable.
- **Cost:** days. Decision point, not a default.

### 4.12 P3: stopping rule + GT end cleanup — only if 4.1 says habit

- **Goal:** fix the label half of the end error on the 13 training patients.
- **Method:** write a one-sentence stopping rule (proposed: "label a slice if any meniscus tissue is visible, including partial-volume"). Re-check only slices within ±2 of each span end on the 13 training patients under that rule. Save as `real_pd_seg_data_v8/`, never overwrite v7. Retrain replica recipe, seed 42, on v8.
- **Output:** how many end slices changed and by how many px; standard report vs replica.
- **Reads as:** edge-bucket Dice rises on the 2 val patients (whose GT is unchanged) → labels were the lever.

### Filtered views (fixable checks → useful approaches → next-step row)

| Source | Item | Next-step row |
|---|---|---|
| Check 1 | P2 negatives | 8 |
| Check 2 | P4 stride | 7 |
| Check 3 | P1 sampler cap | 9 |
| Check 4A | P3 GT ends | 12 (gated on row 1) |
| Approach | k-fold | 6 |
| Approach | weight averaging | 10 |
| Approach | flip TTA | 10 |
| Approach | 3D nnU-Net | 11 |
| Approach | equalize aug | 7 (bundled with P4) |

## Corrections (2026-09-10, from cross-check by the BigRed-side chat)

Trust rule that fell out of this: **numbers computed by scripts and read from stdout are reliable; prose written around them is where the errors were.**

| # | What was wrong | Where | Who | Corrected value | Does the conclusion change? |
|---|---|---|---|---|---|
| C1 | **Hallucination distance compared in slice counts across cohorts with different slice thickness.** "≥3 slices = real hallucination" was calibrated on 3.6mm real PD, then applied to 0.8mm fake PD where 3 slices = 2.4mm. Produced "57% real hallucination" (this chat, Fig 4) and "66% vs 0%, cohorts disagree" (other chat). | Results log row 1, evidence chain step 1, Fig 4 caption, `debug4_results.md` Table 3 | **Both chats.** This chat drew the 2.5-slice line on 0.8mm data first. | In mm, px-weighted: **fake-val ≤3.6mm 68%, 3.6–7.2mm 24%, >7.2mm 8%. Real-val ≤3.6mm 89%, 3.6–7.2mm 11%, >7.2mm 0%.** Median 2.4mm fake, 3.6mm real. **The cohorts agree.** | **Partly.** "Most hallucination is far from GT" is retracted. "Hallucination is real, not annotation edge" survives only because Slicer confirmed non-meniscus tissue at the medial start of MTR_230 and at slice 25 in both real-val patients; it does not follow from the distance data. Fig 4 regenerated with mm axis. |
| C2 | Three summary numbers in `debug4_results.md` Part A (15-pt) were typed from impression, not computed. | Check 4A | Other chat | median ratio_last **0.70** (was 0.85), mean **0.67** (was 0.73), sum px_first+px_last **90,855** (was 47,839), fraction of GT px on span ends **0.249**. Outer/inner split (0.93 vs 0.39) and boundary check (0/30) were computed and stand. | No. Cliff/ramp finding unchanged. "21% of GT on end slices" → 25% for the 15 train pts. |
| C3 | Span detection uses any >0 pixel with no minimum size. One stray voxel can bridge or split a span. | Check 4A | Other chat | The two single-span eval patients (AC13300201B926, AC149BC218E75C) **may be a stray-voxel bridge, not "notch labeled through."** | The claim "radiologist accepted notch labeled through" is **withdrawn until checked** (see request). |
| C4 | 433 = 54×8 + 1 → 55 real batches per cycle, not 54.1. | Check 3 | Other chat | **21.6 cycles/epoch, 562 passes** at epoch 26 (was 22 / 572). | No. |

Suggested by the other chat, not yet done: add slice spacing to `build_hallucination_table.py` output and refuse to pool cohorts whose spacing differs. Recommended: yes, it makes C1 structurally impossible.

## Caveats on the evidence so far (2026-09-10)

**Which model each check ran on.**

| Check | Model | Why |
|---|---|---|
| 1a fake-val CSV | `run_v7_replica_seed42` | Only the replica dumps per-slice predictions |
| 1b real-val CSV | `run_v7_replica_seed42` | Same |
| 1c Slicer | replica predictions | Same |
| 3 cycle count | `run_v7_mixed` (best epoch 26) | Read from the original run. Replica best epoch was 10 → ~220 exposures, still high. |
| 1d item C | `run_v7_mixed/ckpt_best.pth` via `real_pd_predictions_v8_mixed` | **On the actual best model.** Bridges replica findings to the headline checkpoint. |

The replica was built because the original run did not save val-patient predictions. Same recipe, different seed. Replica peaked 0.7809 @ ep10; original 0.7644 @ ep26 on the same 2 val patients. Findings from 1a–1c describe **what the recipe does**, not that one checkpoint. Item C confirms transfer to the best model.

**Check 3 is a mechanism, not yet a proven cause.**
It explains train Dice 0.946. It does not show val Dice would rise with less exposure. Existing counter-evidence: `run_v4_realPD` (real only, 1 pass/epoch, ~50 exposures) scored 0.775 vs 0.781 mixed. A 10× drop in exposure moved val Dice ~0. So memorization may not be what limits val Dice.
Note on v4_realPD (Anshika, 2026-09-10): it started from `run_002_v2`, the best fake-PD-only fine-tune (0.695), and then trained on real PD only. So the fake-PD signal was already baked in via the warm start. It is a fair comparison for exposure count, not for "fake PD vs no fake PD."
**Decisive test (parked, see Parked fix candidates):** retrain v7_mixed with real PD capped at 1 pass/epoch, nothing else changed.

## Step 4 scope: label ceiling (issued 2026-09-10)

Runs on the 15 patients in `real_pd_seg_data_v7` only. Test set untouched.

**Part A, no annotation, 30 min.** From the 15 GT masks, one row per meniscus span (~30 rows): patient, span, first_slice, last_slice, px_first, px_last, px_peak, ratio_first = px_first/px_peak, ratio_last = px_last/px_peak, gap_to_other_span. Summaries: fraction of GT px on first/last slices; median ratio_last. Tests whether the annotator stops abruptly (ratio_last > 0.5 = stopped early) or tapers (≈ 0.2).

**Part B, blind re-annotation, 2–3 h.** 2 training volumes, re-labeled from scratch without seeing the old mask. Compute old-vs-new: dice_full, dice_core (non-empty in both, not within ±1 of either span end), dice_edge (within ±1 of either span end in either mask), and span_offset per span. dice_full is the ceiling.

| Result | Meaning | Next |
|---|---|---|
| median ratio_last > 0.5, span_offset ≥ 1 common | Inconsistent stopping. Edge error is label noise. | P3 stopping rule + GT end cleanup |
| human-vs-human dice_full ≈ 0.80–0.84 | Model within ~0.04 of ceiling | Report ceiling; remaining gain mostly P2 |
| human-vs-human ≥ 0.90, dice_edge high | Labels consistent, edge error is the model's | P2 is the main lever |
| dice_core ≥ 0.90, dice_edge ≤ 0.6 | Core fine, ends are noise | P3; consider excluding ±1 edge slices from loss |

Watch: Part B must be blind. Same preprocessed volume for both passes.

### Part A results (2026-09-10, `~/Downloads/debug4_results.md`, source `data/iu-dataset/labelled-pd-segmentations/`, 15 train patients, native resolution)

| End | Median ratio | Mean | Exactly 1.0 |
|---|---|---|---|
| Outer ends (span1.first, span2.last) | **0.93** | 0.87 | 11/30 |
| Inner ends (span1.last, span2.first) | **0.39** | 0.40 | 0/30 |

- All 15 patients have exactly 2 spans. Notch gap 1–7 slices.
- 0/30 outer ends within 1 slice of the volume boundary → **not a FOV cutoff**.
- 21% of all GT pixels sit on a span's first or last slice.
- Full 32-patient pull: median ratio_last 0.72; two eval patients (AC13300201B926, AC149BC218E75C) show only 1 span (notch labeled through).

**Reading:**
1. Outer end is a **cliff**: last labeled slice is usually the peak slice, next slice empty. Inner end is a **ramp**: tapers to ~40%. Systematic across all 15, not random.
2. Model errors land on the cliff: slice 25 in both real-val patients is one past an outer end with ratio 1.0 / 0.99; 517 and 267 px painted; Anshika confirmed not meniscus. Cliff slice is peak-sized, so one slice of overshoot costs a core slice of FP.
3. Inner ends are where GT misses tissue (Anshika's calls on 14, 18).
4. Model learned a ramp on both sides from DESS (0.8mm, transition over 3–4 slices). On 3.6mm PD the outer side is a 1-slice drop. Links to check 2.

**Cannot conclude from Part A:** whether the cliff is anatomy (periphery ends against capsule, plausible at 3.6mm) or annotators stopping one slice early. Slicer calls split: slice 25 "not meniscus," slice 5 "could be." Needs a second human. Ceiling number still needs Part B.

**Part A2 — radiologist yes/no on the 30 cliff+1 slices (recommended before Part B).** ~20 min for a radiologist. Mostly "no meniscus" → model wrong at cliff, trainable. Many "yes" → training GT stops early, P3 is the lever.

Slice list (from Table 1: span1.first − 1 and span2.last + 1). Indices are in the script's orientation (`DICOMOrient RAS` → `transpose(2,1,0)`); confirm alignment in Slicer by checking that span1.first shows the first labeled slice.

| patient | before span 1 | after span 2 |
|---|---|---|
| AC0407F05FAF53 | 4 | 26 |
| AC045F8F6ACBA7 | 9 | 33 |
| AC0925B79D4836 | 3 | 25 |
| AC0BA0AE159EF9 | 5 | 28 |
| AC0CE315D5758B | 5 | 25 |
| AC0CEE9C24F2B7 | 5 | 25 |
| AC1638B3D9A430 | 3 | 22 |
| AC1B16C5EBA405 | 4 | 25 |
| AC2089D56058F0 | 10 | 30 |
| AC242B1DC40419 | 6 | 26 |
| AC25B946B88F6A | 5 | 28 |
| AC29C3C6B4F385 | 5 | 29 |
| AC2A47E3CF3E07 | 5 | 26 |
| AC2B0AA9AE767D | 6 | 30 |
| AC2E254F52E467 | 3 | 35 |

Answer format per slice: **yes / no / can't tell**. Record who was asked and when. Status: ⬜ to be asked. Asked: — . Answers: — .

**Loose end closed, Table 3:** combined 17-pt hallucination table (2 real + 15 fake) on replica ckpt. Real-val halluc dist 100% at 1–2; fake-val 66% at ≥3. Combined error split: empty-slice FP 17%, on-slice FP 41%, FN 42%. dice_on_sl − dice: real +0.070, fake +0.034. Confirms earlier numbers. Tooling for Part B ready (`scripts/analyze_annotator_agreement.py`, self-test passes).

**Part B dependency (2026-09-10):** requires a human annotator; no script substitute. Candidate: **Nikhil**. Give him the preprocessed PD volumes only, not the mask files. Same instructions as the original annotator had (or none, if none). If Nikhil was an original annotator, the result is self-consistency rather than inter-rater; slightly optimistic but still the right magnitude. Status: ⬜ to be asked.

## Architecture: verdict against the evidence (2026-09-10)

An architecture change helps only if it adds information the model lacks.

| Idea | Adds | Fits evidence? |
|---|---|---|
| Through-plane context at correct spacing (3D, or 2.5D matched to 3.6mm, or whole-stack) | Where the meniscus starts/stops along z | **Yes.** 75% of error. Wait for check 2. |
| Localize → crop → segment at full res (advisor #2) | Thin body visibility | **Yes, secondary.** 25% of error. |
| Bigger encoder / attention / transformer | Capacity | **No.** ResNet50 lost 0.05. Core already at ceiling. |
| Shape prior (advisor #3) | 3D template | **Only in 3D.** Useless in 2D for a z-extent problem. |

### Advisor's papers, assessed 2026-09-10

**Paper 2 = the "FMed paper": Chen et al., Frontiers in Medicine 2022, "Knee Bone and Cartilage Segmentation Based on a 3D Deep Neural Network Using Adversarial Loss for Prior Shape Constraint." [PMC9163741](https://pmc.ncbi.nlm.nih.gov/articles/PMC9163741/)**

| Component | Maps to |
|---|---|
| 3D nnU-Net full-res for cartilage, patch 160×192×64 | **Failure 1 (z-ends).** One patch covers a 36-slice PD volume. Direct architectural answer to the 75%. |
| Adversarial shape prior (discriminator on image+mask patches) | **Failure 2 (split) + notch fill.** But they used it only for bone at half-res; untested on thin structures. |
| Downsample-then-restore for bone | Not relevant; they kept cartilage full-res for the same reason meniscus must be. |
| Dilated bone mask filters cartilage | Weak version of crop/ROI. |

Cautions: their slices are 1mm, ours 3.6mm; 3D on anisotropic data works (nnU-Net) but z-resolution is still 3.6mm. Going 3D forces one spacing → fake PD resampled to 3.6mm = P4 by another name. Results are bone/cartilage only; meniscus named as future work. **Verdict: strongest architecture candidate, targets the right failure, days of work, after checks 2 and 4A2.**

**Paper 1: "Advancing deep learning based knee cartilage segmentation in MRI: innovations, challenges and applications," review, 2025, Osteoarthritis Imaging. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2665913125001384)**
Framing, not method. Names inter-reader variability as a core limitation (= check 4, cite when reporting the ceiling). Points at semi-supervised learning / domain adaptation (= self-training on the 54 unlabeled PD volumes, priority unchanged: after the ends). Nothing on thin-structure splits or z-extent. **Verdict: read for the paper's related-work section; no new lever.**

Flagged, unread: "A memory based model for cartilage and meniscus segmentation in 3D knee MRI," Sci Reports 2025 — meniscus-specific and 3D. [link](https://www.nature.com/articles/s41598-025-31503-2)

Minor unruled-out items: LR 1e-5 (train Dice 0.946 says it fits, not limiting); no test-time augmentation (flip-averaging is free, ~+0.005–0.01, not a bottleneck).

**Check 5 explained:** same recipe, different seed. Replica seed 42 peaked 0.781, original 0.764 on the same 2 val pts → **0.017 from seed alone.** Any fix gaining less than that is indistinguishable from chance. Required before claiming any fix worked.

## Next items (to discuss after checks complete)

| # | Topic | What we have so far | Question to settle |
|---|---|---|---|
| N1 | **Contrast** | Check 6 was scoped (per-patient intensity stats, CNR, real-vs-fake percentile ranges; cheap version = 3 worst vs 3 best slices side by side) but **never run**. Deprioritized because every error found so far is geometric: z-ends and thin-body splits, same on fake and real. v8_clipped gave +0.003. | Do we run the cheap 10-minute version anyway to close it formally, or accept it as ruled out by the geometry finding? |
| N2 | **Architecture** | Bigger encoders: no (ResNet50 lost 0.05, core already at ceiling). Shape prior in 2D: no (can't fix z-extent). Two ideas fit the evidence: (a) 3D or correctly-spaced 2.5D context, targets Failure 1 (75%); (b) localize → crop → segment at full res, targets Failure 2 thin body (part of 25%). Both are rebuilds, not tweaks. Check 2 result decides whether (a) is needed. | Is a 3D rebuild worth it before trying P2 (negatives) and P3 (GT ends) which are cheaper and target the same 75%? |
| N3 | **Paper 2 (FMed, Chen 2022)** | 3D nnU-Net full-res + adversarial shape prior. Maps to both failures. Their data 1mm, ours 3.6mm. Going 3D forces fake-PD resampling to 3.6mm (= P4). Meniscus not in the paper. Strongest architecture candidate; days of work. | If N2 says yes to 3D, is nnU-Net 3D the implementation, and do we add the adversarial prior in the first pass or after? |
| N4 | **Paper 1 (2025 review)** | Framing only. Supports check 4 (inter-reader variability) and the self-training idea (semi-supervised on 54 unlabeled PD). No method to adopt. Flagged unread: Sci Reports 2025 memory-based meniscus model, 3D, meniscus-specific. | Read the Sci Reports paper before deciding N2/N3? |

## Parked fix candidates (do not run during debug; revisit when choosing the first fix)

| # | Fix | Tests which finding | One change only | Cost | What result means |
|---|---|---|---|---|---|
| P1 | **Sampler cap**: real PD limited to 1 pass per epoch inside the mixed loop, everything else identical to v7_mixed | Check 3 (exposure) | Yes | 4 h GPU | Gap shrinks and val rises → exposure was limiting. Gap shrinks, val flat → exposure was a symptom, drop it. |
| P2 | **Negative slices in fake-PD prep**, adjacent-to-span and notch slices first, new data dir `segmentation_data_v3_withneg/` | Check 1 (no negatives) | Yes | prep + 4 h GPU | Hallucination px drop and edge-slice Dice rises → confirmed. Watch for class-imbalance regression in core Dice. |
| P3 | **Edge GT cleanup** on the 13 training patients with a written stopping rule | Check 4 (labels) | Yes | annotation time + 4 h GPU | Edge-slice Dice rises without model change → label half confirmed. |
| P4 | **Fake-PD stride 4–5 for 2.5D stacks** | Check 2 (**done: confirmed**) | Yes | prep + 4 h GPU | **Now well-motivated.** Check 2 showed neighbors are noise, not context. Stride fix gives the model its first look at real 3.6mm neighbors. Test: does z-end error drop? |
| P5 | Crop / localize-then-segment (advisor #2) | Thin-body miss, 25% core error | No (two-stage) | days | Secondary. After P1–P4. |

## Status board (2026-09-10)

| # | Check | Status | Short finding |
|---|---|---|---|
| 1a | Error decomposition, fake-val 15 pts | ✅ Done | Core 0.80, thin ends 0.61. Hallucination 17% of error. Worst patient fills the notch and overshoots ends. Spread 0.57–0.82 on synthetic data. |
| 1b | Error decomposition, real-val 2 pts | ✅ Done | Core 0.84. **75% of error at the through-plane ends.** Every hallucinated slice within 2 slices of GT. |
| 1c | Slicer inspection, fake + real | ✅ Done | Hallucination is real at the medial start and slice 25. GT under-labels at lateral end and notch. Thin body dropped on peripheral slices (the "split"). ~40% model / ~25% GT / ~25% ambiguous. |
| 1d | Same on 17-pt cohort (item C) | ⏸ Parked (test set, not to touch) | 10 of 17 radiologist-confirmed. Which 10 not recorded. |
| 2 | Neighbor-spacing test (i,i,i) vs (i-1,i,i+1) vs (i-2,i,i+2) | ✅ Done (on 17-pt test set, inference only, Anshika's call) | **Model gets no usable z-context from the stack.** (i,i,i): +0.007, 12/17 up, within noise. (i-2,i,i+2): **−0.022, 17/17 worse**, p≈1e-5. Neighbors act as noise. v3_5slice collapse was this effect, not a bug. P4 now well-motivated. |
| 3 | Real-PD cycle count | ✅ Done, cause unproven | **22 cycles/epoch, 572 exposures per real slice by best epoch.** Explains train Dice 0.946. Whether it limits val Dice is untested; v4_realPD (~50 exposures, 0.775) suggests not. Causal test = sampler-capped retrain, parked. |
| 4 | Label ceiling | 🔶 Part A done, A2 + B pending | **Outer ends are a cliff (median 0.93, 11/30 at peak), inner ends a ramp (0.39).** Systematic across all 15. Not FOV. Model overshoots the cliff by 1 slice. Inner ends under-labeled. **Radiologist-confirmed = D1 exactly; confirmed pts score 0.01 lower, so validation status does not explain the spread.** Both single-span pts are confirmed → cliff/ramp is the team's habit, not a radiologist rule. Ceiling number needs Part B. |
| 5 | Seed-7 replica | ⏸ Parked | Noise floor, needed before trusting any fix. |
| 6 | Intensity / contrast | ⏸ Dropped | Error pattern is geometric, not intensity. |

## Commands

Check 1 (formats, so the script is right first time):
```bash
cd /N/project/prostate_cancer_ai/anshika/regGAN && ls results/final_results/real_pd_predictions_v8_mixed/ | head -20 && ls real_pd_seg_data_v7/ | head && find real_pd_seg_data_v7 -type f | head -5 && python -c "import numpy as np,glob; f=sorted(glob.glob('real_pd_seg_data_v7/**/*.np*',recursive=True))[:2]; [print(p, np.load(p).shape, np.load(p).dtype) for p in f]"
```

Check 3:
```bash
cd /N/project/prostate_cancer_ai/anshika/regGAN && find segmentation_data_v2 -name "*.np*" | wc -l && find real_pd_seg_data_v7 -name "*.np*" | wc -l
```

## How to read the results

| Result pattern | Bottleneck is | Next move |
|---|---|---|
| Hallucinated slices far from labeled span, high FP on empty slices across most patients | Data (missing negatives) | Retrain with negative slices (Exp 1) |
| Hallucinated slices adjacent to labeled span, spread tracks cohort or annotator | Labels | Fix GT consistency, re-annotate edges |
| Spread tracks 768 vs 384 resolution | Processing (resampling) | Fix resample pipeline, re-evaluate |
| (i,i,i) Dice ≥ 0.78 | Processing (2.5D mismatch) | Rebuild fake-PD stacks with stride 4–5 |
| Cycle count ≥ 15× | Fine-tuning setup | Cap real-PD repeats per epoch |
| Inter-annotator Dice ≈ 0.80 | Label ceiling | Stop chasing Dice, report ceiling |
| Dice vs CNR correlation > 0.5 | Contrast (image appearance) | Per-patient contrast normalization before inference, test on existing checkpoint |
| Real PD percentiles outside fake-PD range | Processing (normalization mismatch) | Match real-PD intensities to fake-PD at preprocessing, inference-only test |
| Low-Dice patients have high noise, normal CNR | Noise | Denoising or heavier noise augmentation |

---

## Findings so far (updated 2026-09-10)

### Conclusion

**The bottleneck is the ends of the meniscus along the slice axis. It is part model, part labels.** (Revised 2026-09-10 after the 17-pt decomposition and corrections C1–C4.)
On the 2 val patients, 75% of wrong pixels sit on the first/last slice of each meniscus or one beyond; the 17-pt share is pending. Cohort-wide, missed meniscus (FN, 47%) is the largest bucket, hallucination is 17%, and the model does not over-segment overall (FP/FN 1.1). Core slices score 0.84, the ceiling for a ~6px-thick structure. GT is unreliable at the ends: cliff at the outer edge, ramp at the notch, no stopping rule. The model has no usable through-plane context (check 2), so end decisions are made in-plane only. Not domain adaptation, not contrast, not model capacity, not loss function, not in-plane splits.

### Two distinct failure modes

**Failure 1 — through-plane ends (75% of real-PD error).** Model over-extends past the outer cliff and mis-handles the inner ramp. Covered in detail below. **Mechanism (check 2, 2026-09-10):** the model gets no usable z-context from the 2.5D stack on real PD. Trained on 0.8mm DESS neighbors that look like copies of the center, it treats 3.6mm real-PD neighbors as noise. Dropping them costs nothing; spacing them further hurts 17/17 patients. So the z-end decisions are made with in-plane information only.

**Failure 2 — in-plane split of the meniscus. DOWNGRADED 2026-09-10: quantified at 0.64% of core error on the 2 real-val patients (26 px). Frequent on fake PD in Slicer, rare on real PD. Minor for the target domain.** On peripheral slices where GT is one continuous C-shaped band, the prediction breaks into separate chunks: the posterior wedge and one or two anterior pieces, with the **thin body between the horns dropped**. Seen by Anshika in Slicer on:
- MTR_230: slices 34, 38, 44, 45, 46, 74, 76, 77, 87, 90, 91, 92, 98, 101
- MTR_038: slices 90, 91, 98
- Screenshot of MTR_230 slice 34 confirms: GT continuous, pred = 3 chunks, anterior chunks partly outside GT.

This is a **real miss of thin structure**, not annotation ambiguity: the tissue is visibly there and continuous. It is the same shape as the "split prediction" failure the doc reports on the torn cohort (§8), so that failure is **not tear-specific**. It happens on healthy knees. Candidate causes: in-plane resolution too low for a ~2–3 px body (targets: crop / localize-then-segment, P5), or the 2.5D neighbors adding noise rather than context (check 2). Not yet quantified as a share of core error; would need a per-slice connected-component count on pred vs GT.

### Evidence chain

| Step | Source | What it showed |
|---|---|---|
| 1 | fake-val CSV, 15 pts × 160 slices | Core slices 0.80; thin-end slices 0.61 with most FP; hallucination 21% of empty slices, **68% of it within 3.6mm of GT, 8% beyond 7.2mm (corrected C1)**; hallucination is only 17% of total error; per-patient spread 0.57–0.82 on synthetic data |
| 2 | Figure `fakeval_diagnosis.png`, Fig 1 vs 2 | Worst patient fills the notch between the two menisci and overshoots the ends; best patient hugs GT. Good vs bad is entirely the notch and the ends. |
| 3 | Slicer, MTR_230 (Anshika) | Pred starts at slice 20 on non-meniscus tissue, GT correctly at 27 → **real hallucination**. Slices 132–135 look like meniscus but GT stopped at 132 → **GT under-labeled**. 135+ ambiguous. |
| 4 | Slicer, MTR_038 (Anshika) | Pred/GT disagreements where neither is clearly right → **SKM-TEA GT has ambiguity too**. Splits at 90, 91, 98. |
| 5 | Slicer screenshot, MTR_230 slice 85/86 | Dark wedge on posterior tibial plateau, position consistent with posterior horn/root; GT starts 4 slices later. **Not decidable from image.** |
| 6 | Slicer screenshot, MTR_230 slice 34 | GT one continuous band; pred = posterior wedge + two anterior chunks, **thin body dropped**. The "split" is a real in-plane miss of thin tissue on peripheral slices. Splits also at 101, 98, 92, 91, 90, 87, 77, 76, 74, 46, 45, 44, 38. |
| 7 | real-val CSV, 2 pts × 30/32 slices, 3.6mm | **Every hallucinated slice is 1–2 slices from GT, none further.** Error: hallucination 33%, edge slices 42%, core 25%. Core Dice 0.84 both patients. |

### Real-PD error budget

Two cohorts, two checkpoints. The 17-pt numbers supersede the 2-pt ones for magnitude; the 2-pt ones still carry the edge-vs-core split, which has not yet been computed on 17.

| Where | 2 val pts, replica | 17 test pts, v7_mixed (variant A) |
|---|---|---|
| Hallucination on GT-empty slices | 33% | **17%** |
| Over-paint on meniscus slices (on-slice FP) | 22% | **36%** |
| Missed meniscus (FN) | 15% | **47%** |
| of which: on first/last slice ±1 | 42% of total | not yet computed |
| of which: core slices | 25% of total | not yet computed |
| FP / FN | 2.6 | **1.1** |

**What changed with 17 patients:** FN is the largest bucket, not FP. The doc's "model over-segments, FP 2.6× FN" was a 2-patient artifact. The hallucination oracle of +0.067 was also 2-patient; cohort-wide it is worth ~+0.02–0.03. The splits are not the FN (0.64% of core error), so the FN is boundary rim plus under-segmentation at the inner ramp.

**Still true:** hallucination sits within one to two real-PD slices of the GT span (89% within 3.6mm, none beyond 7.2mm). Core-slice Dice is at ceiling. The z-ends remain the largest single error location, but "75%" was 2 patients; the 17-pt share is pending one more computation.

### What this means for fixes

| Fix | Targets | Expected | Note |
|---|---|---|---|
| Add negative slices, **especially adjacent to spans and in the notch** | 33% + part of 42% | +0.03 to +0.05 | Adjacent empties matter more than far ones on real PD |
| Clean GT at the ends of training patients | label half of the 42% | unknown, possibly the bigger lever | No retrain helps if the target is wrong |
| Crop / higher effective resolution (advisor #2) | 25% core, thin body | +0.01 to +0.02 | Real, secondary |
| GAN, diffusion, contrast, loss functions | nothing above | ~0 | Drop |

### Doc claims, verdict after evidence

| Claim | Verdict |
|---|---|
| Hallucination is the root cause, +0.067 | Partly. Real but ~1/3 of error. Oracle inflated by 2 high-hallucination patients. |
| FP is 2.6× FN, model over-segments | 2 patients only. 15-patient fake-val gives 1.4×. Over-extension along z is real; in-plane it's balanced. |
| Domain gap not the bottleneck | Agree, now on evidence: same failure on fake and real, error pattern is geometric. |
| Near geometric ceiling on meniscus slices | True for **core** slices (0.84). False for edge slices (0.72). |
| v3_5slice 0.486 is a bug | **Not a bug.** Check 2: neighbors at 2× distance cost −0.022 on 17/17 patients. 5 slices at 3.6mm = 7.2mm outer neighbors, same effect, larger. |

### Open questions (decide the next lever)

1. How much of the edge error is the model vs the GT? Needs a human call on 8 real-PD edge slices.
2. Who annotated real PD, and what stopping rule at the ends?
3. Does core-only Dice ≈ 0.84 hold across all 17 patients?
4. Real-PD cycle count (check 3).
5. Does the 2.5D stride mismatch matter (check 2)?

### Items requested from Anshika (all independent)

| # | Item | Status |
|---|---|---|
| A | Slicer, real PD val, per-slice call (meniscus present GT missed / no meniscus model wrong / can't tell): AC0CE315D5758B slices **14, 18, 25**; AC0CEE9C24F2B7 slices **5, 6, 17, 18, 25** | ✅ Done, see Results log 2026-09-10 |
| B | Check 3: fake train slices, real train slices, batch size, best epoch | ⬜ How: count train-split files (command above) or `print(len(fake_dataset), len(real_dataset))` after dataset construction in the training script. Batch size from script, best epoch from run_v7_mixed log. |
| C | 17-pt cohort, per patient: full Dice, core-only Dice (drop GT-empty and ±1 edge slices), halluc slice count + distance | ⏸ **Parked by Anshika (2026-09-10): the 17 patients are the test set, not to be touched during debugging.** Would have confirmed the 2-patient finding cohort-wide. Diagnosis stands on replica val (2 real + 15 fake) and Slicer. |
| D | Annotator of real-PD GT and their stopping rule at the ends | ✅ Answered: annotated by Anshika's team, some validated by radiologists, unknown which of the test set were validated. No stated stopping rule. |
| E | Which epoch produced the fake-val CSV | ⬜ minor |
| F | **Step 2 results**: 17-pt table with dice_A [i-1,i,i+1], dice_B [i,i,i], dice_C [i-2,i,i+2], same checkpoint, same preprocessing | ⬜ Next |

### Step 1 status: **done enough to conclude.** Item C is confirmation, not discovery.

### Annotation provenance (logged 2026-09-10)

- Real-PD GT masks were drawn by Anshika's team, not by radiologists.
- **10 of the 17 test patients are radiologist-confirmed, and they are exactly the D1 cohort** (2026-09-10): AC0D3459553205, AC0D5A4D78B628, AC0D7BF72F7712, AC111633B463BB, AC12026D14291F, AC13300201B926, AC13637DA25399, AC149BC218E75C, AC14D3737C0482, AC19E7C19827FF. The 7 new patients are not confirmed.
- **Confirmed vs unconfirmed Dice (v8_mixed):** D1 mean 0.777, new-7 mean 0.787. Difference 0.01, within noise. **Validation status does not explain the per-patient spread.**
- Both single-span patients (AC13300201B926, AC149BC218E75C) are radiologist-confirmed. **Withdrawn pending check (C3):** the single span may be a stray voxel bridging the notch, not the notch labeled through. Until the notch-gap pixel counts are checked, no claim about radiologist stopping conventions rests on these two.
- No explicit stopping rule at the meniscus ends was used. Combined with the slice calls above, the end slices are annotated inconsistently.
- **To carry forward:** recover which of the 17 test patients were radiologist-validated. Add as a column in item C. If the unvalidated patients are the low-Dice ones, that measures the label half of the error directly, with no new annotation.

---

## Results log

| Date | Check | Result | Conclusion |
|---|---|---|---|
| 2026-09-10 | 1 (partial, on fake-PD val, replica run, 15 pts × 160 slices, `~/Downloads/fake_val_per_slice_dice.csv`) | Pooled Dice 0.749. Hallucination on 254/1219 empty slices (21%), 29% of FP. 57% of hallucinated px are ≥3 slices from any GT slice, tail to 19 slices (15mm) → real hallucination, not annotation edge. Oracle from removing hallucination only: +0.033. On-slice FP 104k + FN 106k = 210k vs halluc 43k → hallucination is ~17% of total error voxels. FP/FN 1.39 (not 2.6). **Smallest-GT quartile of slices: Dice 0.61, largest FP of any quartile (34k on 59k GT). Quartiles 2–4: ~0.80.** Per-patient 0.57–0.82, std 0.062; Dice correlates −0.70 with halluc rate, −0.57 with on-slice FP, −0.40 with FN. | Model is at ceiling in the meniscus core and fails at its **ends along the slice axis**: hallucinates past the extent and over-segments the thin end slices. Same failure on synthetic data with consistent labels → not a real-PD annotator issue, not domain gap. Points to no-negatives + weak through-plane context (checks 2, 3). Hallucination fix alone is worth ~+0.03 here, not +0.067. Per-patient spread exists on fake data too. |
| 2026-09-10 | 1 (figure `fakeval_diagnosis.png`) | Fig 1, worst patient MTR_230: model paints 100–700 px on every slice of the **gap between the two menisci** (slices 60–90, the intercondylar notch) and overshoots the far end by 8 slices with a 1100-px spike. Fig 2, best patient MTR_038: prediction hugs GT, gap stays empty, ends overshoot by 1–2 slices. | Good vs bad patient is entirely about the notch and the ends. Model confuses something in the notch (ligament / fat / transverse ligament) for meniscus. Next: look at the notch slice as an image to see what tissue it paints. |
| 2026-09-10 | Slicer inspection, MTR_230 + MTR_038 (fake PD, GT, pred, DESS loaded; CSV indices match Slicer directly) | Medial start: pred begins at 20, GT correctly at 27, slices 20–27 painted on **non-meniscus tissue** → real hallucination. Lateral end 132–135: tissue looks like meniscus, **GT stopped early** → annotation edge. 135+: ambiguous. MTR_038: pred/GT disagreements where neither is clearly right. Meniscus **split into two** in places (location not yet determined). | Error is three things mixed: real hallucination (trainable), GT under-labeling at ends (ceiling), unadjudicable disagreements (ceiling). SKM-TEA GT is not clean either. Open: notch tissue identity (stop 3), core FN chunk vs rim (stop 5), whether split is mid-compartment (normal two-horn anatomy) or at the body (real miss). |
| 2026-09-10 | Slicer screenshots MTR_230 slices 85/86 and 34 | 85/86 (notch): pred is a dark wedge on the posterior tibial plateau, shape and position consistent with posterior horn/root; GT starts 4 slices later. Ambiguous, not decidable from image. 34 (core): GT is one continuous band; pred = posterior wedge + two anterior chunks, **thin body between horns missing**, anterior chunks partly outside GT. | The "split" is the model dropping the thin body on peripheral slices where GT is continuous → real in-plane miss of thin structure. Notch hallucination may be GT starting late. |
| 2026-09-10 | Real-PD val CSV (`per_slice_dice.csv`, 2 pts, 62 slices, 3.6mm) | **Every hallucinated slice is 1–2 slices (3.6–7.2mm) from the GT span. None far.** Error budget: hallucination 33%, edge slices (±1 of span end) 42%, core 25%. Core-slice Dice 0.84 both patients; edge-slice Dice 0.78 / 0.72. Oracle remove halluc 0.752 → 0.819. | **75% of real-PD error is at the through-plane ends.** Core is at the thin-structure ceiling (0.84). Combined with Slicer: the ends are where GT itself is unreliable (stops early, ambiguous at notch). So the remaining error is part model over-extension (trainable) and part label ambiguity (ceiling). Not domain gap, not contrast. |
| 2026-09-10 | Slicer, real-PD val edge slices (Anshika's calls) | **AC0CE315D5758B**: 14 (99px) meniscus present, GT missed. 18 (366px) meniscus present, GT missed. 25 (517px) not meniscus, model wrong. **AC0CEE9C24F2B7**: 5 (538px) could be meniscus. 6 (fp 493) could be meniscus. 17, 18 (fn 167, 121) model under-segmented real meniscus. 25 (267px) not meniscus, model wrong. Of 2,001 halluc px: ~39% confirmed model error, ~23% GT missed meniscus, ~27% ambiguous, ~10% unchecked. | "Half model, half labels" confirmed on real PD. **Slice 25 wrong in both patients**: same lateral-end position, a consistent anatomical confuser at the outer edge of the knee. **Notch side (17–18) unreliable in both directions**: GT missed it in one patient, model missed it in the other. |
| 2026-09-10 | Slicer, slice 25, both real-val patients (Anshika) | Model paints meniscus-like tissue where the meniscus has not started yet. Same position in both patients. Specific tissue not identified. | Model expects meniscus and has no learned reason to say "not yet." Consistent with no negatives in training. |
| 2026-09-10 | **Check 2, neighbor-spacing, 17-pt test set, `run_v7_mixed/ckpt_best.pth`** | Control: dice_A 0.7812 = reported 0.781. **B (i,i,i): 0.7886, +0.0074, 12/17 up**, range −0.037 to +0.078. **C (i-2,i,i+2): 0.7588, −0.0225, 17/17 down.** Biggest B gain: AC0D5A4D78B628 0.709→0.788 (worst D1 pt). Biggest B loss: AC07607B9E5295 0.843→0.805 (best pt). | Neighbors are treated as corrupted copies of the center, not as context. Dropping them costs nothing; making them more different hurts uniformly. The 75% z-end error is made with in-plane info only. Mechanism for v3_5slice collapse. (0,0,0) at inference is not a fix: mean within seed noise, sign flips per patient. **Caveat: ran on the parked test set; inference only.** |
| 2026-09-10 | **Check 2 follow-up: per-slice decomposition of A/B/C, 17-pt test set, v7_mixed ckpt** | **A**: empty-slice FP 9,824 (17.3%), on-slice FP 20,130 (35.5%), FN 26,700 (47.1%), FP/FN 1.12, total 56,654. **B (i,i,i)**: 6,949 (12.9%), 17,752 (33.1%), 28,970 (54.0%), FP/FN 0.85, total 53,671. **C**: 9,980, 18,516, 32,332, FP/FN 0.88, total 60,828. Halluc distance (3.6mm slices): A 83% at 1–2, max 5; B 89% at 1–2, max 3; C 87%, max 5. | **Cohort-wide error budget is different from the 2-patient one.** FN is the largest bucket (47%), hallucination is 17%, FP/FN is 1.1 not 2.6. Doc's +0.067 oracle is a 2-patient artifact; cohort-wide hallucination fix is worth ~+0.02–0.03. **B cuts hallucination 29% and on-slice FP 12% but adds 8.5% FN**: without neighbors the model is tighter. Neighbors push it toward painting more (union-like effect of three near-copies). C adds both FN and hallucination: worse everywhere. |
| 2026-09-10 | **Split quantification, 2 real-val pts, replica ckpt, core slices (n=30), 8-connectivity** | Orphan px (pred component overlapping no GT component): **26 of 4,054 core-error px = 0.64%.** Component count pred = GT on 26/30 slices; pred fragmented on 3/30 (4 excess components); pred fewer on 1/30. GT-empty slices: 9 slices / 2,001 px, matches check 1b exactly. | **Failure 2 (split) is rare on real PD**, at least on 2 patients. The frequent splits seen in Slicer were on fake PD (MTR_230, 0.8mm). So the 47% FN on the test set is not splits; it must be boundary rim + inner-ramp under-segmentation (e.g. F2B7 slices 17–18). Failure 2 downgraded to minor for real PD. |
| 2026-09-10 | **Check 4 Part A, GT tapering, 15 train pts** | Outer ends median ratio 0.93 (11/30 exactly 1.0), inner ends 0.39 (0/30). 0/30 at volume boundary. 21% of GT px on first/last slices. Two eval patients have a single span (notch labeled through). | Labels have a consistent shape: cliff at the outer edge, ramp at the notch. Model overshoots the cliff by one slice, at peak-slice cost. Whether the cliff is anatomy or early stopping is undecidable without a second human. |
| 2026-09-10 | **Check 3, real-PD cycle count** | fake train: 124 scans, 9,512 slices. real train: 13 scans, 433 slices. batch 8 both. best epoch 26. → fake batches/epoch 1189, real batches/cycle 54, **cycles/epoch 22**. By epoch 26: each real slice seen **~572×**, each fake slice 26×. | **Memorization is structural.** The loop wraps the real set 22× per epoch. Train Dice 0.946 is what 570 exposures of 433 slices produces. Fix is the sampler (cap real passes per epoch or fixed fake:real ratio), not the loss. Reframes the +0.166 gap: not capacity, exposure. |

---

## Terms

- **2.5D stack** = three neighboring slices fed as the three input channels of a 2D network
- **FP / FN** = pixels predicted as meniscus but GT says no / GT says yes but model missed
- **warm start** = starting training from an already-trained checkpoint
- **cycle count** = how many times the small real-PD set repeats inside one pass over the fake-PD set
- **label ceiling** = Dice two human annotators get against each other; a model can't reliably beat it
- **annotation-edge** = slices at the meniscus tip where the human stopped labeling but tissue is faintly present
- **stride** = spacing between slices picked as neighbors, in slice counts
