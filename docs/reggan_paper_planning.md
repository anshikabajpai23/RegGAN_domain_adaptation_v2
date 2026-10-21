# RegGAN Meniscus — Paper Planning

**Created:** 2026-09-12. Split out of `reggan_dice_debug.md` (debugging stays there).
**Contents:** current-state scoring for ISBI / MICCAI / CVPR, novelty / depth / rigor detail, venue and deadline facts, O-1 and job impact, novelty ideas with hand-off specs (Phase 6), literature table.

---

## Paper readiness — current state (scored 2026-09-12, out of 10)

Scores reflect the work **as it stands today**. No projections; re-score after Phase 5 and the levers.

| Metric | ISBI | MICCAI | CVPR | Why this score | What would raise it | Effort |
|---|---|---|---|---|---|---|
| Novelty | 5 | 4 | 3 | RegGAN is an existing method applied to a new pair; Stages 9–10 unrun; label-efficiency is framing, not method | lever 2 (spacing-aware aggregation), extent head, lever 3 (Stage 9) | 2–3 wk |
| Experimental rigor | 5 | 4 | 3 | seed repeat, k-fold, per-slice decomposition done; but test set read ~12×, only 2 val patients, no ceiling, and the headline root cause was tested and rejected | clean held-out protocol, per-patient sign tests, disclose test reuse | 1 wk |
| Baselines | 3 | 2 | 2 | "DESS model on PD = 0.00" is a straw man; no CycleGAN downstream number, no ImageNet-init control, no nnU-Net | lever 1 (CycleGAN→same recipe), ImageNet-init on 15 real PD only | 1–2 wk |
| Results strength | 5 | 4 | 3 | 0.78–0.80 on 17 patients, no CI, no ceiling to compare against; DESS literature sits at 0.85–0.90 | seq + crop + TTA to ~0.82 with CI, reported as % of human ceiling | 2 wk + Nikhil 3 h |
| Dataset scale | 6 | 4 | 2 | 155 DESS, 69 PD, 15 labeled, 17 test, one site, one anatomy | torn-cohort GT as a second untouched cohort | not in hand |
| Clinical relevance | 8 | 7 | 5 | real problem: PD is clinical standard, labels scarce, DESS labels public | label-efficiency curve at 5 and 20 annotations (lever 4) | 1 wk |
| Analysis / ablations | 7 | 6 | 5 | strongest section: diagnostic chain, corrected root causes, negative results with mechanisms | end-tolerant Dice (lever 5), deformation-vs-Dice plot | 3 d |
| Reproducibility | 8 | 7 | 7 | verification audit, corrections log, numbers traced to stdout | release per-slice scripts + this log | 2 d |
| Story | 4 | 3 | 3 | not written; the doc's original narrative (hallucination root cause) is now known to be wrong | rewrite around label efficiency + "fixed 2.5D can't use z-context under spacing mismatch" | 2–3 wk |
| **Overall** | **5.5** | **4.5** | **3** | | | **~10 wk, mostly parallel** |

Runs overlap (all 4 h GPU jobs); the 10 weeks is dominated by writing and by waiting on other people. Long poles: writing, Nikhil (ceiling), torn-cohort GT.

**CVPR is capped by dataset scale and single-site/single-anatomy** — not fixable inside this project. CV-side fit is a CVPR workshop after MICCAI, not the main track. NeurIPS main very low, ICML near zero.

### Novelty / technical depth / rigor — detail (2026-09-12)

| Dimension | ISBI | MICCAI | CVPR |
|---|---|---|---|
| Novelty | 5 | 4 | 3 |
| Technical depth | 6 | 5 | 3.5 |
| Experimental rigor | 5.5 | 4 | 3 |

**Novelty.** Every component is off the shelf: RegGAN, standard smoothness/magnitude regularizers, ConvLSTM, smp.Unet+ResNet34. New = the DESS→PD meniscus pairing and one empirical finding. The finding is the best asset and is currently unframed: *a 2D net with fixed slice channels cannot use z-context when slice spacing differs between train and test domains; a model that learns to aggregate across slices can.* Raise via: lever 2 spacing-aware aggregation (+1.5, largest single move), extent head (+0.5), Stage 9 (+1 at MICCAI).

**Technical depth.** Strong: Jacobian determinant, folding %, per-label boundary distance. Most application papers skip this; foreground it. **Two weaknesses reviewers will find:**
1. **R trained and evaluated on different pairings** (own doc §3.7): trains on (fake_B, random unrelated real PD), evaluated as R(fake_PD, DESS). Central component used out of distribution. Attempted fix reverted (field collapsed to zero). **Fix properly or state as an explicit limitation — do not leave it to be found.**
2. **"FID" is not FID**: computed on 64×64 pixel features, not InceptionV3 activations, because BigRed has no internet. Reporting 164.4 alongside published FID is wrong. Rename the metric, or transfer cached Inception weights (one-time file copy) and compute it properly.

Raise via: fix/limit R pairing (+1, removes rejection risk), honest FID (+0.5), Stage 9 on boundary distance (+1).

**Experimental rigor.** Strong: measured seed noise, 5-fold CV, per-slice decomposition, corrections log C1–C5, deterministic inference verified. **One problem dominates: the 17-pt test set has been read ~12× to compare variants** (sequence, crop, P4, P2, P1, 5 k-folds, replica, replica_v2). That is model selection on test. Reporting the best of those as a clean held-out number is not honest, and a reviewer counting the ablation rows will see it. Fix: freeze one config chosen on val + k-fold only, then a single test read; report mean ± CI over 17; disclose exploratory reads as exploratory; add Nikhil's ceiling; per-patient sign tests.

**Effort priority, one per dimension:** (1) rigor — freeze config, one clean test read (near-zero cost, biggest rejection risk removed); (2) novelty — lever 2 (one retrain); (3) depth — address R pairing even if only as a stated limitation with evidence.

### Novelty raisers, re-examined (2026-09-12)

| Idea | Verdict | Why |
|---|---|---|
| Lever 2 spacing-aware aggregation | Optional | Sequence result + check 2 already make the point; keep only if N1 doesn't close the gap |
| Extent head | **Drop** | Same 17% bucket P2 targeted; P2 gave +0.0015; model already balanced (FP/FN 1.1) |
| Stage 9 rigid/non-rigid | MICCAI only | Requires R pairing fix first; evaluate on boundary distance not Dice; no room at ISBI |

**Missed insight:** P4 used thin 0.8mm slices spaced 3.6mm apart. A real PD slice is a **3.6mm-thick slab** (average over thickness), not a thin sample. Fake PD has sharp 0.8mm through-plane content everywhere because the GAN is 2D. P4 fixed spacing, not slab blur → its borderline result does not rule out the through-plane gap. "Domain gap is small" was measured in-plane only; the through-plane gap was never closed by anything in the pipeline, and error sits where through-plane blur matters (ends).

### New ideas (2026-09-12)

| # | Idea | Pipeline part changed | Effort / time | Dice upside | Addresses | Novelty |
|---|---|---|---|---|---|---|
| **N1** | **Slab simulation + partial-volume soft labels**: average 4–5 consecutive DESS-derived fake-PD slices into one 3.6mm slab; average masks the same way → end slices get fractional labels. | **Processing** (fake-PD prep) + **fine-tuning** (soft-label loss). RegGAN untouched. | ~3 days: 1 d prep, 4 h retrain, 1 d eval | **+0.01 to +0.03** at ends; first change to the physics of the training data | spacing, slab blur, end ambiguity, P4 null | real |
| **N2** | **Through-plane gap as the story**: 2D translation closes in-plane appearance but not through-plane resolution; error concentrates at structure ends. | **No changes.** Writing. | 2–3 d drafting | 0 | unifies checks 1, 2, 4, P4, sequence, crop | the memorable claim |
| N3 | Z-consistency loss on 54 unlabeled PD | **Fine-tuning** (loss + unlabeled loader) | 2 d code + 4 h | 0 to +0.01 | real-PD signal, ends | moderate |
| N4 | Soft end labels from human disagreement | **Fine-tuning** (targets). Needs 4B. | blocked on Nikhil 3 h + 1 d + 4 h | 0 to +0.01 | end ambiguity | moderate |
| Lever 2 | Spacing-aware aggregation | **Fine-tuning** (sequence input) | 1 wk | 0 to +0.01 over sequence | z-context | moderate, optional |
| Extent head | Per-slice "any meniscus" gate | **Fine-tuning** (extra head) | 3 d | ~0 (P2 predicts null) | hallucination | dropped |
| Stage 9 | Rigid/non-rigid flow split | **RegGAN** (loss) → re-translate 155 vols → re-run seg. Requires R pairing fix first. | 2–3 wk | **0** on Dice (own §6.2); gain on boundary distance | anatomy preservation | real, MICCAI only |

Only N1 and Stage 9 touch anything upstream of fine-tuning; everything else is a loss or head on the seg model, where Phase 2 already landed within noise. Stage 9 is the most expensive with zero Dice upside.

**Decision:** N1 replaces lever 2 and the extent head. Stage 9 stays MICCAI-only.

## Literature: meniscus segmentation (compiled 2026-09-12)

Tags: ✅ read from paper text, ◐ search snippet, ○ memory, verify before citing.

| Paper | Approach | Arch | Data | N | Meniscus Dice | Tag | Differs from ours |
|---|---|---|---|---|---|---|---|
| SKM-TEA, Desai 2021 NeurIPS D&B | per-slice 2D seg on qDESS | 2D V-Net / U-Net, soft Dice | qDESS 0.38×0.31 mm, 0.8 mm slices, 1 annotator/scan | 155: 86/33/36 | **0.85** V-Net, 0.84 U-Net | ✅ | our DESS source; thin slices, native labels, no modality gap |
| IWOAI challenge, Desai 2021 Rad AI | 6 teams, U-Net variants | 2D/2.5D/3D | OAI DESS, 88 pts × 2 tp | 88 | winner ~0.88; top 4 tied | ◐ | architecture stops mattering when data + labels match |
| Gaj 2020 MRM | adversarial mask prior | cGAN seg | OAI DESS | ~176 | lat **0.89**, med **0.87** | ◐ | highest reported; high-res, ~10× our labels; close to FMed idea |
| Tack 2018 OAC | CNN + statistical shape model | 2D CNN + SSM | OAI DESS | 88 segs | med ~0.84, lat ~0.89 | ○ | 3D shape prior needs thin slices |
| Norman 2018 Radiology | first large 2D U-Net on OAI | 2D U-Net | OAI DESS + T1ρ | ~600 | ~0.75–0.81 | ◐ | large N, thin slices |
| Jeon 2021 | adversarial + object-aware map | 2D U-Net + D | knee MR | ? | med **0.85**, lat **0.84** | ◐ | localize-then-segment ≈ our crop |
| Semi-supervised DESS, OAC 2019 | unlabeled DESS consistency | 2D U-Net | OAI DESS | ? | ~0.86 | ○ | same idea as our 54 unlabeled PD, no modality gap |
| **SAMRI-2, Sci Rep 2025** (PMC12804825) | SAM2-style memory across slices, mask propagation | transformer + memory vs 3D nnU-Net / V-Net | multi-protocol; **external test, 2 radiologists** | 575 vols / 270 pts; 57 ext | avg tissues **0.81–0.84**; nnU-Net 0.80–0.82 | ◐ | closest to our sequence model; has the ceiling measurement we lack |
| Clinical 2D U-Net preliminary, 2024/25 (PubMed 39974461) | plain U-Net, surgeon labels | 2D U-Net | clinical, seq not stated | 104 / 50 | **0.76** | ✅ | only clinical small-N row; lands where we are |
| ATTU-Net pilot 2021 | attention U-Net, small clinical | Att-U-Net | clinical | small | unverified | ○ | same regime |

**Summary.** Nearly all work is on thin-slice 3D DESS (OAI, SKM-TEA) with dense native labels: 2D U-Net 0.84–0.85, shape/adversarial prior 0.87–0.89, top architectures statistically tied (IWOAI). Recent direction: memory across slices beats 3D convs (SAMRI-2) = our sequence finding. Clinical 2D small-N: ~0.76.

**Our regime vs literature:** 2D PD TSE (clinical) vs 3D DESS; **3.6 mm** vs 0.8 mm slices; ~36 vs ~160 slices; **15** vs 88–600 labeled volumes; labels from another modality via translation vs native; 0.78–0.82 vs 0.84–0.89; ceiling unmeasured vs SAMRI-2 measured.

**Consequences:** (1) 0.78–0.82 is not low for this regime; the clinical small-N row is 0.76 with native labels. (2) Nobody in the table faces the through-plane problem; N2 has no competitor. (3) SAMRI-2 is the closest comparison and best citation.

**Verify before citing:** Tack, Norman, semi-supervised numbers; SAMRI-2 meniscus-specific Dice from PMC full text.

## Phase 6 hand-off specs: novelty ideas (2026-09-12)

Global rules from Phase 4 apply: one change per retrain, base = `finetune_meniscus_v7_replica.py` seed 42, report on 2 real-val + fake-val with per-slice dumps, results-only markdown with commands and stdout-sourced numbers, mm not slice counts across cohorts, **no test-set reads except one final frozen candidate**. Before any step: verify the file/format assumptions marked ⚠ and report them back; do not guess.

### N1 — Slab simulation + partial-volume soft labels

**Goal.** Make fake-PD training data physically match real PD through-plane: 3.6mm-thick slabs (averaged), not 0.8mm thin slices; and give end slices fractional labels equal to the true meniscus fraction in the slab.

**Inputs.** `results/fake_pd_all155/` (fake-PD volumes), `preprocessed_v3/masks/` (DESS-derived 3-class masks), `preprocessed_v3/splits.json`.
⚠ Verify first: (a) each volume has the full 160-slice stack, not only mask-bearing slices; (b) slice index = through-plane order at 0.8mm; (c) masks exist for every slice (empty where no meniscus) or only for mask-bearing ones. Report `ls | wc -l` per volume for both dirs.

**Method.**
1. Load each fake-PD volume as (160, 384, 384) float and its mask as one-hot (160, 3, 384, 384).
2. Box-average along z with window 4.5 slices: for output slab k covering [4.5k, 4.5(k+1)), weight boundary slices fractionally. Output ~35 slabs per volume (matches real PD's ~36). Same operation on the one-hot mask → per-class fraction in [0, 1] per pixel. Store images as float16 and masks as float16 (35, 3, 384, 384) in a **new dir `segmentation_data_v4_slab/`**. Never overwrite v2/v3.
   - Alternative if simpler: SimpleITK resample along z to 3.6mm with a box/Gaussian prefilter of σ ≈ 1.5 slices. State which was used.
3. 2.5D stacks: neighbors at slab index ±1 (= 3.6mm), i.e. **stride 1 in slab space**. Volume-end rule: repeat nearest slab, same as `infer_real_pd_v3.py`.
4. Slab inclusion: keep slabs where the summed meniscus fraction ≥ the equivalent of `min_meniscus_pixels=10` (i.e. sum over pixels of (frac_lat + frac_med) ≥ 10). Positives only in run 1; negatives optional in a later run.
5. Loss on fake branch: soft cross-entropy `−Σ_c w_c · t_c · log q_c` with the same weights (0.1, 1.5, 1.5) and soft targets t; soft Dice with fractional targets. Real-PD branch `MergedLoss` unchanged.
6. **Epoch definition (decision):** the fake set shrinks ~4.5× (9,512 → ~2,100 slabs). Keep **fixed 1,189 fake batches per epoch, sampling fake with replacement**, so real-PD exposure per epoch stays at ~21.6 cycles. P1 showed real exposure is signal; do not let it drop as a side effect. Say the word to use a natural epoch instead.
7. Augmentation: equalized (real matches fake, as in P4).
8. Train replica recipe, seed 42. Also run **N1 + sequence model** as a second run if the sequence spec is available.

**Output.** Standard report vs replica and vs P4; per-slice dumps; edge/core/empty split on the 2 val patients; fake-val Dice on slab-val (note: fake-val changes definition, report both slab-val and thin-val if possible).

**Reads as.** End-bucket error down and 2-val Dice up by > 0.008 → through-plane physics matters; P4's null was the wrong fix, not the wrong hypothesis. No change → spacing/blur is not the lever; keep sequence + crop.

### N2 — Through-plane gap as the story (writing)

**Claim sentence.** "A 2D translation network closes the in-plane appearance gap but cannot alter through-plane resolution; under an anisotropy mismatch between source (0.8mm) and target (3.6mm), segmentation error concentrates at structure ends where through-plane resolution dominates, and only models that aggregate across slices, or training data that reproduces the target's slab physics, reduce it."

**Evidence map (one row per clause).**
| Clause | Evidence |
|---|---|
| in-plane gap is closed | fake-val ≈ real-val Dice (0.749 vs 0.752); real-only ≈ mixed (v4, v7); better-FID model gave worse Dice |
| error at ends | 2-val: 75% of error at ±1 of span ends; 17-pt: FN 47%, halluc 17%, all halluc ≤ 7.2mm from GT |
| core at ceiling | core-slice Dice 0.84 vs ~0.83 geometric |
| fixed 2.5D cannot use z | check 2: (i,i,i) ≈ baseline, (i−2,i,i+2) −0.022 on 17/17; P4 stride +0.008 |
| learned aggregation can | sequence +0.018; crop +0.013 |
| slab physics | N1 result (pending) |
| labels ambiguous at ends | cliff 0.93 / ramp 0.39; Slicer calls; ceiling (4B pending) |

**Deliverables.** Abstract (150 words) with this claim; Fig 1 = the per-slice profile (Fig 1/2 of `fakeval_diagnosis.png`) redrawn for one real patient; Fig 2 = error budget by bucket, 2-val and 17-pt; Table 1 = ablation with a column "reads on test set: exploratory / final".

### N3 — Z-consistency on unlabeled real PD

**Goal.** Use the ~54 unlabeled PD volumes (`data/iu-dataset/pd-files` minus the 15 labeled) as a consistency signal.
**Method.** New unlabeled loader yielding adjacent slice pairs (i, i+1). Loss: symmetric KL between softmax(i) and softmax(i+1), **masked to pixels where both predictions have p_men > 0.5** so the loss never pushes an empty slice toward non-empty. λ = 0.1, linear ramp over first 5 epochs. Warm start from the best sequence or replica checkpoint. One change.
**⚠ Risk.** Unmasked consistency smooths across the outer cliff and would increase the exact error we have. If the masked version still raises end-bucket FP, stop.
**Output.** Standard report; edge/core/empty split.

### N4 — Soft end labels calibrated from human disagreement

**Prereq.** 4B. Note 4B re-annotates the 2 **val** patients, which cannot be trained on. So N4 does not use those masks as targets; it uses them to **calibrate a rule**: measure, per span end, how many slices the two annotators differ by (span_offset) and what fraction of the end-slice area they agree on. Derive a rule such as "first and last labeled slice of each span → target 0.5" or "±1 slice → linear ramp," and apply it to the 13 training patients' masks. One change, one retrain.
**Output.** Standard report; edge-bucket Dice on the 2 val patients (GT unchanged there).

### Lever 2 — Spacing-aware aggregation (optional; needs sequence spec)

**Method.** In the sequence model, condition the cross-slice aggregator on physical spacing: add a scalar channel (spacing in mm, normalized) to the ConvLSTM input, or FiLM-modulate its gates. Train on mixed data with the true spacing per sample (0.8 for thin fake, 3.6 for slab fake and real). Compare to the unconditioned sequence model. Only worth running if N1 does not already close the end gap.

### Stage 9 — Rigid / non-rigid flow decomposition (MICCAI only)

**Prereq: fix the R pairing (own doc §3.7).** Current R trains on (fake_B, random real_B) and is evaluated as R(fake_PD, DESS). Options: (a) train R on (fake_B, real_A) with an edge-magnitude similarity (the doc's proposal); (b) **MIND descriptors** (modality-independent neighbourhood descriptor) as the similarity term, the standard for cross-modality registration. Recommend (b). Confirm the field does not collapse to zero (mean |flow| > 0.01 px) before proceeding.
**Method.** `dataset.py` must load DESS masks (currently never does). Per batch: fit a least-squares similarity transform (translation + rotation, optionally scale) to the predicted flow over the meniscus mask region → `flow_rigid`. `flow_nonrigid = flow − flow_rigid`. Replace `L_mag = mean(flow²)` with `L_nonrigid = mean(flow_nonrigid²)` over the mask region (or whole image; state which). Keep `L_smooth`. Same weights as before.
**Then.** Re-translate the 155 volumes; run `boundary_distance_eval.py` and the Jacobian analysis; then one seg run on the new fake PD.
**Output.** Boundary distance per label (5, 6, all) vs old RegGAN; Jacobian folding %; downstream 2-val Dice (expected unchanged; say so in the paper).
**Cost.** 2–3 weeks. Not before ISBI.

### Extent head — dropped

Recorded for completeness: per-slice "any meniscus" head gating the mask. Not run. P2's +0.0015 on the same error bucket predicts a null.

## Deadlines (verified 2026-09-12)

| Venue | Abstract/registration step | Full paper | Notification | Conference |
|---|---|---|---|---|
| **ISBI 2027** | none for the paper track | **26 Oct 2026** | 12 Jan 2027 | 25–28 May 2027, Lausanne |
| **CVPR 2027** | paper registration **10 Nov 2026** | 16 Nov 2026 (suppl. 23 Nov) | reviews 25 Jan, decisions 25 Feb 2027 | 20–25 Jun 2027, Seattle |
| **MICCAI 2027** | abstract registration **~11 Feb 2027** (TBA; 2026 was Feb 12) | ~25 Feb 2027 | ~mid Jun 2027 | 26 Sep–1 Oct 2027, Auckland |

Registration steps for CVPR and MICCAI are title + abstract + authors only, not reviewed. ISBI also has a separate **1-page abstract track (1 Feb 2027)** — a poster-level submission, not an indexed 4-page paper; much weaker for CV and O-1.

**Plan:** ISBI 4-page 26 Oct 2026 → MICCAI 8-page ~25 Feb 2027. ISBI notifies 12 Jan, six weeks before the MICCAI deadline, so there is no simultaneous-submission problem. MICCAI version must be substantially extended (baselines, ceiling, spacing-aware method, label curve) and cite the ISBI paper. Skip CVPR 2027.

**Confirm:** ISBI 2027 page limit and whether it is double-blind, when the CFP posts.

## O-1 and applied-ML-job impact (2026-09-12)

**O-1A:** decided on criteria, not venues. Scholarly articles (any peer-reviewed; lawyer cites acceptance rate), original contributions (**citations**, venue irrelevant), judging (**reviewing** for MICCAI/TMI/MedIA is a strong line). Citation clock matters more than tier → publish early. Start reviewing this year via advisor.

**Applied ML jobs:** conferences barely matter for ML engineer roles; recruiters recognize NeurIPS/ICML/CVPR by name, MICCAI/ISBI both read as "a conference." Medical imaging AI companies (Siemens, GE, Philips, Aidoc, Lunit, Qure.ai, Tempus): MICCAI + journals are the currency, ISBI respectable. The resume line that sells is the project description, not the venue.

**Combined plan:** MICCAI primary; SPIE already out starts the citation clock; ISBI only as fallback (must be substantially new material, disclosed). Review for MICCAI/ISBI now.

## Venue comparison (2026-09-12)

| Venue | CORE | CCF | Acceptance | Note |
|---|---|---|---|---|
| CVPR | A* | A | ~22–25% | not a fit for this work |
| MICCAI | A | not listed | ~27% | flagship; target |
| ISBI | unranked now (B in ERA2010) | not listed | ~50% | 4 pages; fallback |
| MIDL | unranked | not listed | ~35–40% | good for careful empirical work |
| SPIE MI | unranked | not listed | high | clinical audience |

Recommendation: MICCAI, using the extra months for baselines + ceiling. ISBI as fallback cycle.

## Novelty and depth: five levers (2026-09-12)

| # | Lever | Adds | Cost |
|---|---|---|---|
| 1 | CycleGAN → same seg recipe; plot per-patient deformation vs Dice | Makes "anatomy preservation matters" causal; doubles as the missing baseline | 1 translation + 1 seg run |
| 2 | Spacing-aware slice aggregation: feed physical slice spacing to the sequence model | Names the check-2 finding as a method; anisotropy mismatch in cross-modality transfer | modify sequence model, 1 retrain |
| 3 | Stage 9 rigid/non-rigid flow split in RegGAN | The real methods contribution; eval on boundary distance, not Dice | loss change, 1 translation run |
| 4 | Label-efficiency curve with 5 and 20 annotations, reported as % of human ceiling | Turns 3 points into a clinical claim | 2 seg runs + 4B |
| 5 | End-tolerant Dice (ignore ±1 slice at each span end) reported alongside standard | Quantifies label-ambiguity cost; makes core-at-ceiling a headline | metric only |

Priority if only two: 1 and 2. Not for novelty: diffusion, transformers, bigger encoder.

## Expected Dice after everything (2026-09-12, estimates)

| Step | Adds | Running |
|---|---|---|
| Sequence | +0.018 measured | 0.796 |
| + crop | +0.005–0.013 | 0.80–0.81 |
| + flip TTA | +0.01–0.017 | 0.81–0.82 |
| + spacing-aware agg / real upweight | 0–0.01 each | 0.81–0.83 |
| 3D nnU-Net (replaces seq+crop) | maybe +0.01 over them | 0.82–0.83 |
| Levers 1, 3, 4, 5 | 0 Dice | rigor/novelty only |

**Realistic 0.81–0.82, optimistic 0.83.** Wall above that = end-slice labels. Ceiling guess 0.82–0.88; needs 4B.

## Ideas that touch the findings (vs surface)

| Finding | Idea |
|---|---|
| No usable z-context; learned aggregation works | Spacing-aware aggregation (lever 2) |
| Ends unreliable in GT | **Soft end labels**: 0.5 target or drop ±1 end slices from the loss |
| Outer cliff overshoot, 17% halluc | **Extent head**: per-slice "any meniscus?" output, gates the mask at inference. Cleaner than negatives-in-data, which gave +0.0015. |
| Real-PD exposure is signal | Real-loss upweight; then self-training on 54 unlabeled |
| Crop helped | Keep native 768 at the crop, don't downsample to 384 |
| Core at ceiling | Stop optimizing in-plane |

Surface, not findings: diffusion (realism not limiting), transformer/bigger encoder (in-plane solved, ResNet50 lost 0.05).

