# Phase 4 Results

> Pulled from `docs/reggan_dice_debug.md`. Tables only, as specified.

## Checks

| # | Check | Status | Finding | Fixable? | Expected gain | Approach |
|---|---|---|---|---|---|---|
| 1 | Error location | ✅ | Core at ceiling (0.84). Error at z-ends. Cohort-wide FN 47%, halluc 17%. | Partly (half labels) | +0.02–0.05 | P2 + P4 |
| 2 | Neighbor slices | ✅ | Model has no z-context; wider neighbors hurt 17/17. | Yes | +0.02–0.04, largest single lever | **P4** |
| 3 | Real-PD repeats | ✅ cause tested, step 9 | 21.6×/epoch, 562 by best epoch. **Capping it (step 9/P1) dropped val Dice −0.0380, above noise — repeats are useful signal, not just memorization.** | No — tested, hurts | 0 (was 0–0.01, now answered negative) | ~~P1~~ dropped |
| 4A | Label stopping | ✅ | Cliff outer (0.93), ramp inner (0.39). No rule. | Partly | 0–0.03 pending 4A2 | P3 |
| 4A2 | Cliff real? | ⬜ radiologist | | decides P3 | | |
| 4B | Ceiling | ⬜ Nikhil | | decides target | | |
| 5 | Seed noise | ✅ | 0.017 (doc estimate, replica vs original) | not a fix | | needed to judge fixes |
| 6 | Contrast | dropped | geometric error | no | 0 | |
| Split | Chunks | ✅ | 0.6% of real-PD core error | not needed | 0 | |

## Candidate approaches

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
| Slice-sequence architecture (ConvLSTM / cross-slice attention) instead of fixed 2.5D stack (Anshika) | Worth trying | Check 2 showed the model ignores neighbors at FIXED channel positions (B≈A, C hurts 17/17). A sequence model that learns to weight/aggregate neighboring slices adaptively, rather than baking z-context into 3 fixed input channels, targets the same failure (no usable z-context) via a different mechanism than P4's stride fix or 3D nnU-Net's full-volume patch. Untested; would need its own retrain, not a drop-in change. |
| Contrast-based training + shape training for torn meniscus (Anshika) | Different target, not yet scoped | None of checks 1-6 were run on the 20-patient torn cohort (no GT there). Contrast/intensity training was already checked and dropped for the HEALTHY-knee error (check 6: geometric, not intensity-driven) — but tear signal (bright fluid at the tear site) is a genuinely different appearance problem, so that verdict may not transfer. A shape prior specific to torn anatomy (vs. the general 3D shape prior already covered by FMed's paper, which is healthy-shape only) targets the doc's own torn-cohort failure modes (§8: split prediction at the tear, missed detection) that no current check or approach addresses. Needs GT on the torn cohort before it can be evaluated at all. |
| Review paper | reading | |
| RegGAN rigid/SDM | No for Dice | Methods novelty only. |
| Tversky / over-segment | No | Model under-segments cohort-wide (FP/FN 1.1). |
| 3D CC post-proc | No | Overshoot is connected to body. |
| (0,0,0) inference | No | Coin flip per patient. |
| Flip TTA | **Yes, free** | +0.005–0.01. |
| Equalize aug branches | Hygiene | ~0, removes confound. Do with P4. |

## Conclusions

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

## Next steps

| # | Step | Cost | Gives | Result |
|---|---|---|---|---|
| 1 | 4A2 radiologist, 30 slices | 20 min | cliff anatomy vs habit → P3 go/no-go | **Parked** |
| 2 | 4B Nikhil, 2 volumes | 2–3 h | ceiling → realistic target | **Parked** |
| 3 | Edge/core split on 17-pt A and B CSVs | 10 min | cohort-wide end share; where B's gain lives | Done — see `reggan_dice_debug.md` §4.3 |
| 4 | Notch-gap px counts, 2 single-span pts | 5 min | closes C3 | Done — see `reggan_dice_debug.md` §4.4 |
| 5 | Seed-7 replica | 4 h GPU bg | noise floor | **Done.** Clean same-script, two-seed comparison (seed 42 vs seed 7): best epoch 10 vs 12 (+2). dice_real_va @best 0.7809 vs 0.7860 (+0.0051). Per-patient: AC0CE315D5758B −0.0032, AC0CEE9C24F2B7 +0.0226. **Per-patient max \|Δ\| = 0.0226, mean \|Δ\| = 0.0129** — larger than the doc's earlier 0.017 estimate (which compared replica vs the original run, not two seeds of the same script). Plateau range: seed42 0.0258, seed7 0.0212. |
| 6 | k-fold (≥3 folds) | 3×4 h | CI + checkpoint | Pending |
| 7 | **P4** stride 4–5 + equalize aug, one retrain | prep + 4 h | does z-context cut end error; case for 3D | **Done.** dice_real_va @best 0.7809→0.7899 (+0.0090), per-patient max \|Δ\|=0.0177, mean=0.0133 — **below the 0.0226 noise floor.** Edge/core split vs replica: edge wrong-px −210, core −123, empty +4. Cross-checked against the seed42-vs-seed7 noise draw on the same buckets (edge Δ=−208, core Δ=+82, empty Δ=−331): **edge's −210 is indistinguishable from the −208 noise draw** — same direction P4 targeted, but seed randomness alone produces nearly the same shift. Core's −123 (opposite sign from the noise draw) is the only piece with any signal, on n=1 noise comparison. dice_fake_va dropped −0.0536 (0.7598→0.7062), best epoch shifted 10→15. **Also newly established: seed noise alone swings the empty-slice-FP pixel count by 331 px on a base of ~2,000 — larger noise band on that specific metric than previously known, relevant to reading step 8's result.** |
| 8 | **P2** negatives adjacent+notch, new dir, one retrain | prep + 4 h | halluc drop, +0.02–0.03 | **Running.** Same prep as step 7 (withneg=13,840 = 9,516 pos + 4,324 neg). Training submitted to BigRed, `run_v7_p2`, in parallel with step 7. Read its empty-slice-FP result against the ±331px noise band established in step 7's row, not just directionally. |
| 9 | **P1** sampler cap, one retrain | 4 h | closes exposure question | **Done.** dice_real_va @best 0.7809→0.7429 (**−0.0380**), per-patient −0.0371/−0.0383 (near-identical across both patients, unlike a noise draw) — **above the 0.0226 floor, a real effect.** Train Dice dropped 0.9247→0.8430 (−0.0817), gap shrank +0.1438→+0.1000 as intended, but val Dice got worse rather than flat/better. Per the doc's own decision rule ("gap shrinks, val flat → symptom only, drop P1") this is that outcome in a stronger form — val didn't just fail to improve, it declined. **Updates check 3's "cause unproven" to answered: the 21.6×/epoch repetition looks like useful signal, not just harmful memorization** (consistent with existing `run_v4_realPD` counter-evidence). Sampler cap as implemented does not help. |
| 10 | Weight avg + flip TTA | 30 min | +0.01–0.02 free | Pending |
| 11 | 3D nnU-Net (FMed) if P4 helps | days | arch that sees the ends | Pending |
| 12 | **P3** stopping rule + GT end cleanup on 13 training pts, one retrain — **only if 4A2 says the cliff is habit** | annotation + 4 h | label half of the end error, 0–0.03 | **Blocked** (step 1 parked) |

Realistic: P4 + P2 + P1 + free wins → **0.78 → 0.81–0.83**, capped by the ceiling from step 2.

## Next-step dependencies

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
