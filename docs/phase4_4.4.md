# Phase 4 — Step 4.4: notch-gap pixel counts, 2 single-span patients

> Results only. Numbers from script stdout.
> Date: 2026-09-10

## Inputs

| | |
|---|---|
| Patients | AC13300201B926, AC149BC218E75C (the 2 flagged single-span cases from Check 4A Table 2) |
| GT source | `labelled-pd-segmentations/` (pulled from `data/iu-dataset/labelled-pd-segmentations/` on BigRed) |
| Loader | `load_native_binary()` from `analyze_gt_tapering.py` — SimpleITK → DICOMOrient RAS → transpose(2,1,0) → binarize. Native resolution, no resize |
| Connectivity | 26 (3×3×3) for 3D components |

## Command

```
./venv/bin/python scripts/phase4_4_notch_gap.py \
  --mask_dir labelled-pd-segmentations \
  --patients AC13300201B926 AC149BC218E75C
```

---

## AC13300201B926

File: `AC13300201B926_SAG_PD_TSE_14_1_1.seg.nrrd` · shape `(32, 768, 768)` · total GT px 24,808

Per-slice GT pixel count (whole volume):

```
0:0     1:0     2:0     3:0     4:0     5:0     6:0     7:1452  8:1842  9:1473
10:1310 11:1217 12:1124 13:973  14:501  15:378  16:773  17:1252 18:868  19:1055
20:1408 21:1944 22:1940 23:2178 24:1548 25:1572 26:0    27:0    28:0    29:0
30:0    31:0
```

| | |
|---|---|
| non-empty slice range | 7..25 (19 slices) |
| slices with 0 < px < 10 | **NONE** |
| 3D connected components (26-conn) | **2** |
| component 1 | 9,892 px, slices 7..14 |
| component 2 | 14,916 px, slices 15..25 |
| spans if `--min_span_pixels 10` applied | `[(7, 25)]` — count 1 (unchanged) |

---

## AC149BC218E75C

File: `AC149BC218E75C_SAG_PD_TSE_5_1_1.seg.nrrd` · shape `(28, 384, 384)` · total GT px 6,819

Per-slice GT pixel count (whole volume):

```
0:0    1:0    2:0    3:219  4:555  5:480  6:370  7:370  8:383  9:456
10:215 11:126 12:82  13:136 14:96  15:260 16:280 17:347 18:378 19:448
20:537 21:628 22:453 23:0   24:0   25:0   26:0   27:0
```

| | |
|---|---|
| non-empty slice range | 3..22 (20 slices) |
| slices with 0 < px < 10 | **NONE** |
| 3D connected components (26-conn) | **2** |
| component 1 | 3,174 px, slices 3..11 |
| component 2 | 3,645 px, slices 12..22 |
| spans if `--min_span_pixels 10` applied | `[(3, 22)]` — count 1 (unchanged) |

---

## Summary across both patients

| | AC13300201B926 | AC149BC218E75C |
|---|---|---|
| slices with 0 < px < 10 | NONE | NONE |
| minimum non-zero slice px | 378 (slice 15) | 82 (slice 12) |
| 3D components | 2 | 2 |
| component slice ranges | 7..14 / 15..25 | 3..11 / 12..22 |
| empty slices between components | **0** | **0** |
| `--min_span_pixels 10` changes span count | No (1 → 1) | No (1 → 1) |

Both patients: the two 3D components occupy **adjacent** slice ranges with zero empty slices between them (component 1 ends at slice 14 / 11, component 2 begins at slice 15 / 12).

## Files

| File | Contents |
|---|---|
| `scripts/phase4_4_notch_gap.py` | Analysis script |
| `labelled-pd-segmentations/` | GT source (32 masks, gitignored as `*.nrrd`) |
