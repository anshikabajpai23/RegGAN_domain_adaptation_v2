# Replica — Manual Visual Evaluation (Slicer)

Manual, per-patient Slicer review of `run_v7_replica_seed42`'s predictions vs GT, on the fake-PD and real-PD validation sets. This is separate from the automated Dice numbers (`analysis_phase4_17patients.ipynb` Section 2 / Section 2b) — the point is to catch failure patterns that a single aggregate Dice number can hide, and to cross-check whether the manual read agrees with what the automated checks already found (Check 1: error concentrated at edges; Check 4: outer-end cliff / inner-end ramp in the labels).

Dice values below are pulled from the notebook's clean per-patient computation, not typed from memory.

## Fake-PD validation set (15 patients)

| Patient | Dice | Reviewed | Observation |
|---|---|---|---|
| MTR_026 | 0.7918 | ✅ | Undersegmented on most slices; one edge slice oversegmented. |
| MTR_033 | 0.8027 | ✅ | Some oversegmentation, mixed quality — some of it looks anatomically reasonable, some of it doesn't. |
| MTR_038 | 0.8218 | ✅ | Mid-slice wrongly segmented, and fragmented into chunks there. |
| MTR_053 | 0.7747 | ✅ | Both edges completely wrong; middle also completely wrong. |
| MTR_057 | 0.7776 | ✅ | Hard to call most slices right/under/wrong; one edge is clearly oversegmented. |
| MTR_083 | 0.7442 | ⬜ | Not yet reviewed. |
| MTR_156 | 0.7915 | ⬜ | Not yet reviewed. |
| MTR_172 | 0.7069 | ✅ | GT itself may be wrong in places; some genuine model errors too (small mis-segmented patches — tissue/bone confusion). |
| MTR_174 | 0.7706 | ⬜ | Not yet reviewed. |
| MTR_203 | 0.6781 | ✅ | Many places wrong or oversegmented in a way that doesn't look like meniscus; also undersegmented in many places. |
| MTR_212 | 0.7870 | ⬜ | Not yet reviewed. |
| MTR_218 | 0.7413 | ⬜ | Not yet reviewed. |
| MTR_224 | 0.7273 | ✅ | Core slices undersegmented; edge slices (start/end of both medial and lateral meniscus) oversegmented — the now-familiar edge pattern. |
| MTR_230 | 0.5737 | ⬜ | Not yet reviewed (already flagged separately as the dataset's clear outlier — see `docs/reggan_dice_debug.md`). |
| MTR_237 | 0.7428 | ⬜ | Not yet reviewed. |

## Real-PD validation set (2 patients)

| Patient | Dice | Reviewed | Observation |
|---|---|---|---|
| AC0CEE9C24F2B7 ("F2B7") | 0.7506 | ✅ | Prediction goes outside the body in several places — oversegmenting light-grey boundary-area tissue. Candidate fix: contrast / image-processing changes targeting that boundary region. |
| AC0CE315D5758B ("758B") | 0.7545 | ✅ | Mostly correct; errors look like **GT missing** meniscus rather than the model being wrong. Edge slices may be segmenting real grey-area tissue GT doesn't label. Both patients: a second annotator's read is needed to settle how much of the "error" is actually missing GT. |

## Cross-check against the automated checks

- **MTR_224's edge-oversegmented / core-undersegmented pattern matches Check 1 exactly** (edge slices carry a disproportionate share of the error) and **Check 4** (labels cliff at the outer end) — independent manual confirmation of an automated finding, on a fake-PD patient this time rather than just the 2 real-val patients Check 1 was computed on.
- **758B's "GT missing, not model wrong" read matches the real-val edge-slice calls already logged in `docs/reggan_dice_debug.md`** (slices 14, 18 for this same patient: "meniscus present, GT missed it").
- **F2B7's "goes outside the body" observation is new** — not something the automated per-pixel checks (edge/core split, component split) specifically isolated. Worth a dedicated look: is this the same failure as the "notch hallucination" already documented, or a distinct boundary-contrast issue?

## New candidate idea surfaced here

- **Shape-based training (MTR_026)** — noted as a possible direction while reviewing consistent undersegmentation. Not yet scoped against `docs/phase4_results.md`'s existing "Shape prior — only in 3D" candidate-approach row (2D can't use a 3D shape prior per that doc); needs a decision on whether this is the same idea or a different, 2D-compatible version before it's added there as its own row.

## Log (raw notes, lightly cleaned for spelling only)

**Fake-PD:**
- MTR_026: undersegmented most of the slices; also shape-based training might work; 1 edge slice oversegmented.
- MTR_033: some places oversegmented which is good [anatomically justified], some oversegmentation is wrong.
- MTR_038: mid slice wrongly segmented, and in chunks too.
- MTR_053: both edges completely wrong, mid completely wrong.
- MTR_057: not sure about majority of slices whether correct/under/wrong, but one side edge is oversegmented.
- MTR_172: GT may be wrong, and some places it is wrongly predicted (segmented tissue/bone or something else, but a very small area).
- MTR_203: many places wrong or oversegmented in a way that doesn't look like meniscus; undersegmented in many places.
- MTR_224: core slices undersegmented, but as usual edge slices (start/end of medial and lateral meniscus) are oversegmented (prediction).

**Real-PD:**
- F2B7: going outside the body in many places; need contrast/image-processing changes since it's oversegmenting boundary areas (light grey) somewhere.
- 758B: mostly correct, GT missing in places; edges might be segmenting real grey-area tissue. For both patients, an inter-annotator result is definitely needed — many places feel like GT might be missing meniscus, not the model being wrong.
