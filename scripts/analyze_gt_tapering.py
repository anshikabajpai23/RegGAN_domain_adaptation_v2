"""
analyze_gt_tapering.py
=======================
Debug test 4, Part A — how abruptly does the GT stop at each meniscus span's
ends? No model involved. Native-resolution GT only, no resize/resample (that
would interpolate label pixels and distort the exact tapering shape we're
trying to measure) — only a RAS reorientation to get a consistent slice axis,
using the same DICOMOrient + transpose(2,1,0) convention already validated
elsewhere in this project (compare_stack_variants.py's load_nrrd_binary()).

A meniscus, viewed slice-by-slice in a sagittal PD scan, shows up as TWO
separate contiguous runs of non-empty slices along the slice axis (lateral
and medial are spatially separate — there's an anatomical gap between them,
even though the GT label itself is a single binary "meniscus" class with no
lateral/medial distinction). Each such run is one "span" here.

Span numbering is BY POSITION along the slice axis (1 = lower slice indices,
2 = higher), not by anatomical side — the binary GT has no side information to
recover that from. If you need true medial/lateral identity, that requires
knowing scan laterality per patient, which isn't in these files.

If a patient's mask doesn't show exactly 2 contiguous runs, it's flagged
rather than silently forced into a "span 1 / span 2" shape — that's a data
quality signal worth looking at, not an error to paper over.

Usage:
  python scripts/analyze_gt_tapering.py \
      --mask_dir labelled-pd-segmentations \
      --out_csv results/debug4_gt_tapering.csv
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import SimpleITK as sitk


def get_pid(fname):
    m = re.match(r"(AC[A-F0-9]+)", os.path.basename(fname), re.IGNORECASE)
    return m.group(1) if m else os.path.splitext(os.path.basename(fname))[0]


def load_native_binary(path):
    """Native resolution, RAS-reoriented, slice axis = array axis 0.
    Same transpose convention as load_nrrd_binary() elsewhere in this repo,
    minus the isotropic-resample/384-resize steps (deliberately — we want the
    raw annotated pixel counts, not interpolated ones)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 4:   # multi-component seg.nrrd — same handling as load_nrrd_binary
        arr = (arr.max(axis=0) > 0).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        try:
            comp = sitk.VectorIndexSelectionCast(img, 0)
            out.CopyInformation(comp)
        except Exception:
            out.SetSpacing(img.GetSpacing()[:3]); out.SetOrigin(img.GetOrigin()[:3])
            out.SetDirection(img.GetDirection()[:9])
    else:
        arr = (arr > 0).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
    try:
        out = sitk.DICOMOrient(out, "RAS")
    except Exception:
        pass
    arr = sitk.GetArrayFromImage(out).transpose(2, 1, 0)
    return (arr > 0).astype(bool)


def find_spans(per_slice_px):
    """Contiguous runs of nonzero per_slice_px. Returns list of (first, last, peak_idx)."""
    nonzero = per_slice_px > 0
    spans = []
    i = 0
    n = len(nonzero)
    while i < n:
        if nonzero[i]:
            j = i
            while j < n and nonzero[j]:
                j += 1
            spans.append((i, j - 1))   # inclusive first, last
            i = j
        else:
            i += 1
    return spans


def analyze_patient(pid, path):
    vol = load_native_binary(path)
    per_slice_px = vol.sum(axis=(1, 2))
    raw_spans = find_spans(per_slice_px)

    rows = []
    warning = None
    if len(raw_spans) == 0:
        warning = "NO non-empty slices at all — empty mask"
        return rows, warning
    if len(raw_spans) == 1:
        warning = f"only 1 contiguous span found (expected 2) — spans: {raw_spans}"
    elif len(raw_spans) > 2:
        # keep the two largest by total pixel mass, flag the rest
        by_mass = sorted(raw_spans, key=lambda s: per_slice_px[s[0]:s[1]+1].sum(), reverse=True)
        dropped = by_mass[2:]
        warning = (f"{len(raw_spans)} contiguous spans found (expected 2) — "
                  f"using the 2 largest, dropped {len(dropped)} small run(s): {dropped}")
        raw_spans = sorted(by_mass[:2], key=lambda s: s[0])   # back to position order

    spans = sorted(raw_spans, key=lambda s: s[0])   # span 1 = lower slice indices

    for span_num, (first, last) in enumerate(spans, start=1):
        seg = per_slice_px[first:last + 1]
        px_first, px_last, px_peak = int(seg[0]), int(seg[-1]), int(seg.max())

        if len(spans) == 2:
            other = spans[1] if span_num == 1 else spans[0]
            if span_num == 1:
                gap = other[0] - last - 1          # slices strictly between this span's end and the other's start
            else:
                gap = first - other[1] - 1
        else:
            gap = None   # can't define a notch without exactly 2 spans

        rows.append({
            "patient":            pid,
            "span":               span_num,
            "first_slice":        first,
            "last_slice":         last,
            "px_first":           px_first,
            "px_last":            px_last,
            "px_peak":            px_peak,
            "ratio_first":        round(px_first / px_peak, 4) if px_peak else None,
            "ratio_last":         round(px_last / px_peak, 4) if px_peak else None,
            "gap_to_other_span":  gap,
        })
    return rows, warning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    # "*.nrrd" alone already matches "*.seg.nrrd" files too (the wildcard covers
    # the ".seg" part) — globbing both patterns would double-count every file.
    files = sorted(glob.glob(os.path.join(args.mask_dir, "*.nrrd")))
    if not files:
        print(f"No .nrrd files found under {args.mask_dir}")
        return
    print(f"Found {len(files)} mask files")

    all_rows = []
    warnings = []
    for f in files:
        pid = get_pid(f)
        rows, warning = analyze_patient(pid, f)
        all_rows.extend(rows)
        if warning:
            warnings.append((pid, warning))

    cols = ["patient", "span", "first_slice", "last_slice", "px_first", "px_last",
            "px_peak", "ratio_first", "ratio_last", "gap_to_other_span"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in all_rows)) for c in cols}

    print()
    print("| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |")
    print("|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|")
    for r in all_rows:
        print("| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |")

    if warnings:
        print(f"\n=== {len(warnings)} patient(s) flagged (not exactly 2 clean spans) ===")
        for pid, w in warnings:
            print(f"  {pid}: {w}")

    edge_px = sum(r["px_first"] + r["px_last"] for r in all_rows)
    ratios_last = [r["ratio_last"] for r in all_rows if r["ratio_last"] is not None]

    print(f"\n=== summary ({len(all_rows)} spans across {len(files)} patients) ===")
    print(f"  sum(px_first + px_last) across all spans : {edge_px:,}")
    print(f"  median ratio_last                         : {np.median(ratios_last):.4f}")
    print(f"  mean   ratio_last                          : {np.mean(ratios_last):.4f}")
    print(f"  ratio_last distribution: min={min(ratios_last):.3f}  "
         f"25%={np.percentile(ratios_last,25):.3f}  "
         f"75%={np.percentile(ratios_last,75):.3f}  max={max(ratios_last):.3f}")

    # NOTE on "fraction of all GT pixels" — needs total GT pixel mass per patient,
    # not just the two edge slices. Computed properly here as a second pass.
    total_all_px = 0
    for f in files:
        vol = load_native_binary(f)
        total_all_px += int(vol.sum())
    frac = edge_px / total_all_px if total_all_px else float("nan")
    print(f"  total GT pixels (all patients, all slices) : {total_all_px:,}")
    print(f"  fraction of all GT pixels on a span's first/last slice : {frac:.4f}")

    print(f"\n=== how to read it (per your thresholds) ===")
    med = np.median(ratios_last)
    if med > 0.5:
        print(f"  median ratio_last = {med:.3f} > 0.5 -> abrupt stops.")
        print(f"  Annotators are stopping inconsistently, not following a clean taper.")
        print(f"  -> Edge error is largely label noise, not model failure.")
    else:
        print(f"  median ratio_last = {med:.3f} <= 0.5 -> reasonably clean tapering.")
        print(f"  -> Stopping point looks anatomically consistent; edge error less likely to be pure label noise.")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
