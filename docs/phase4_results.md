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
| 7 | **P4** stride 4–5 + equalize aug, one retrain | prep + 4 h | does z-context cut end error; case for 3D | **Done, confirmed on 17pt test set.** 2-val-patient read: dice_real_va @best 0.7809→0.7899 (+0.0090), below the 0.0226 noise floor; edge/core split vs seed42-vs-seed7 noise draw showed the edge-bucket "win" (−210) indistinguishable from noise (−208), core (−123) the only piece with any signal. **17-patient test set (`analysis_phase4_17patients.ipynb`): 0.7780→0.7856, Δ=+0.0076 — SMALLER than the 2-patient read, not larger, despite 8.5× the patients.** More data made the effect shrink, not confirm it. **No real improvement.** |
| 8 | **P2** negatives adjacent+notch, new dir, one retrain | prep + 4 h | halluc drop, +0.02–0.03 | **Done, confirmed on 17pt test set.** 2-val-patient history.csv not yet re-pulled locally. **17-patient test set: 0.7780→0.7795, Δ=+0.0015 — essentially nothing**, on the exact metric (test Dice) the negatives were meant to move. **No real improvement.** |
| 9 | **P1** sampler cap, one retrain | 4 h | closes exposure question | **Done, confirmed on 17pt test set — stronger than the 2-patient read.** 2-val-patient: 0.7809→0.7429 (−0.0380), above the 0.0226 floor. **17-patient test set: 0.7780→0.7489, Δ=−0.0292** — same direction, comparable magnitude, now backed by 17 patients instead of 2. Train Dice dropped 0.9247→0.8430, gap shrank +0.1438→+0.1000 as intended, but val AND test both got worse, not flat/better. **Updates check 3 to answered: the 21.6×/epoch repetition is useful signal, not just memorization.** Sampler cap as implemented actively hurts — confirmed, not just suggested. |
| 10 | Weight avg + flip TTA | 30 min | +0.01–0.02 free | **Done**, on `run_v7_replica_seed42`, 2 val patients. `avg` alone: +0.0034 mean, negligible (patient deltas +0.0003/+0.0065) — checkpoint averaging isn't doing anything. **`tta`: +0.0175 mean (+0.0195/+0.0155), `avgtta`: +0.0181 mean (+0.0184/+0.0179)** — both below the strict 0.0226 per-patient noise ceiling individually, but the pattern differs from the seed42-vs-seed7 noise draw (which moved the 2 patients in *opposite* directions, −0.0032/+0.0226): TTA moved both patients the *same* direction by *similar* magnitude — suggestive of a small real effect, not proven with only 1 noise-floor reference. `avgtta` ≈ `tta` alone, so averaging adds nothing on top of TTA. **Generalizes: flip-TTA is inference-only, zero-cost, no-downside — apply it to any future checkpoint (P4, P2, crop, k-fold, sequence) for a likely small free gain, not just this one.** Tooling: `scripts/average_checkpoints.py`, `--tta` flag on `infer_real_pd_v3.py`, `scripts/dice_two_val_patients.py` (verified against known baseline Dice 0.7545/0.7506 before trusting the new variants). |
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
