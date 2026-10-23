"""
tear_patterns_seq.py
====================
v16: the v15 tear-pattern library (tear_patterns.py, UNCHANGED) applied to the crop+sequence
model's input: a 5-slice stack [i-2, i-1, i, i+1, i+2] in the stretched crop space.

Two adaptations, nothing else:
  1. Slices. The pattern is applied to the middle three slices (i-1, i, i+1) exactly as in v15;
     the outer slices (i-2, i+2, 7.2 mm away) are left untouched.
  2. Crop stretch. The crop box (e.g. 294 x 201 px) is resized to 384 x 384, so pixels are
     stretched ~1.3x vertically and ~1.9x horizontally. v15's sizes are in native pixels (lobe
     radius, flap / bucket distance, cleft width...), so applying them directly in crop space
     would make tears physically smaller and the wrong shape. With native_hw set, the stack is
     resized to the native box size, the pattern is applied there, and only the CHANGED pixels
     are mapped back (image: changed regions blended in; label: only where the label changed).
     Pixels the pattern did not touch keep their original crop-space values exactly.

The label rule is v15's: tissue where it actually is (removed -> background, displaced ->
meniscus at its new place, signal inside the meniscus -> still meniscus).
"""
import numpy as np
from scipy import ndimage as ndi

from tear_patterns import synth_pattern, MODES, MODE_P


def _resize(arr, shape, order):
    """Resize the last two axes of arr to `shape` (exact output size)."""
    hw = arr.shape[-2:]
    f = (1.0,) * (arr.ndim - 2) + (shape[0] / hw[0], shape[1] / hw[1])
    out = ndi.zoom(arr, f, order=order, mode="nearest", grid_mode=True)
    sl = (slice(None),) * (arr.ndim - 2) + (slice(0, shape[0]), slice(0, shape[1]))
    out = out[sl]
    pad = [(0, 0)] * (arr.ndim - 2) + [(0, shape[0] - out.shape[-2]), (0, shape[1] - out.shape[-1])]
    return np.pad(out, pad, mode="edge") if any(p[1] for p in pad) else out


def _resize_labels(lab, shape):
    """Smooth label resize: per-class indicator, linear interpolation, then argmax."""
    classes = np.unique(lab)
    if len(classes) == 1:
        return np.full(shape, classes[0], dtype=lab.dtype)
    stack = np.stack([_resize((lab == c).astype(np.float32), shape, 1) for c in classes])
    return classes[stack.argmax(0)].astype(lab.dtype)


def synth_pattern_seq(image, mask, rng, p=0.5, modes=MODES, mode_p=MODE_P, native_hw=None,
                      inplace_variants=None, inplace_p=None):
    """
    image: (T, H, W) float32 in [0, 1], T odd, centre slice = T // 2 (T = 5 for the sequence model).
    mask:  (H, W) int labels of the centre slice.
    native_hw: (h, w) size of the crop box before it was resized to (H, W); None = apply directly.
    inplace_variants / inplace_p: passed to tear_patterns.synth_pattern (PATTERNS_V17 adds Fig. 10a).
    Returns (image_out, mask_out, mode, sub_before, sub_after, info): sub_* are the 3 middle slices
    before / after, in v15's log format; info = {"variant", "variant_all", "tear"} with the tear map
    (H, W) in crop space (see tear_patterns.synth_pattern).
    """
    T = image.shape[0]
    c = T // 2
    sub = image[c - 1:c + 2].astype(np.float32)
    H, W = mask.shape
    kw = dict(p=p, modes=modes, mode_p=mode_p, inplace_variants=inplace_variants, inplace_p=inplace_p,
              return_info=True)

    if native_hw is None or tuple(native_hw) == (H, W):
        out3, new, mode, info = synth_pattern(sub, mask, rng, **kw)
    else:
        h, w = native_hw
        sub_n = np.clip(_resize(sub, (h, w), 1), 0.0, 1.0)
        mask_n = _resize_labels(mask, (h, w))
        out_n, new_n, mode, info = synth_pattern(sub_n, mask_n, rng, **kw)
        if mode == "none":
            info["tear"] = np.zeros((H, W), bool)
            return image, mask, mode, sub, sub, info
        # image: blend the native result back only where the pattern changed something
        changed = np.abs(out_n - sub_n) > 1e-3                              # (3, h, w)
        changed = np.stack([ndi.binary_dilation(ch, iterations=1) for ch in changed])
        wgt = np.clip(_resize(changed.astype(np.float32), (H, W), 1), 0.0, 1.0)
        out_up = np.clip(_resize(out_n, (H, W), 1), 0.0, 1.0)
        out3 = (sub * (1 - wgt) + out_up * wgt).astype(np.float32)
        # label: replace only where the native label changed (inplace -> label untouched)
        new = mask.copy()
        lab_changed = new_n != mask_n
        if lab_changed.any():
            zone = ndi.binary_dilation(_resize(lab_changed.astype(np.float32), (H, W), 1) > 0.05, iterations=1)
            new_up = _resize_labels(new_n, (H, W))
            new[zone] = new_up[zone]
        info["tear"] = _resize(info["tear"].astype(np.float32), (H, W), 1) >= 0.5

    if mode == "none":
        return image, mask, mode, sub, sub, info
    img = image.astype(np.float32).copy()
    img[c - 1:c + 2] = out3
    return img, new, mode, sub, out3.astype(np.float32), info
