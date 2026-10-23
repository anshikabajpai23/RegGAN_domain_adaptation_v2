"""
tear_synth.py
=============
Synthetic meniscal tears for "tear-amodal" training (v14).

Paints a tear-like bright region INSIDE the meniscus of a healthy 2.5D training
sample and returns the painted image plus the tear mask. The segmentation label
is NOT changed by the caller: the model is asked to segment the whole meniscus
wedge through the tear ("amodal"), and optionally to also predict the visible
(dark) part and the outer contour.

Patterns (per horn = connected component of the mask), in the sagittal plane:
  horizontal  thin cleft roughly along the horn's long axis
  vertical    thin cleft roughly perpendicular to it
  oblique     thin cleft at about 45 degrees
  radial      vertical cleft that always reaches the meniscus surface
  complex     two clefts (horizontal + oblique)
  maceration  a large, irregular, patchy brighter area covering 30-80% of the horn
Intensity: I' = I + w * (fluid - I), w = alpha * soft_tear, with fluid = 90th
percentile of the centre slice. Clefts: alpha ~ U(0.25, 0.8) (faint grey line to
near-fluid). Maceration: alpha ~ U(0.2, 0.6) on a blurred, patchy field, so the
horn turns partly grey rather than flat white. Tuned by eye with
scripts/preview_tear_synth.py; change ALPHA_* if the preview looks off.
Neighbouring channels (slices i-1, i+1) get the same tear, shifted by up to
2 px, with probability neighbor_p, since real tears span 3.6 mm slices.

Only numpy + scipy, so it runs in DataLoader workers on BigRed unchanged.
"""
import numpy as np
from scipy import ndimage as ndi

PATTERNS = ("horizontal", "vertical", "oblique", "radial", "complex", "maceration")
PATTERN_P = (0.25, 0.20, 0.20, 0.10, 0.10, 0.15)
# v17 / v16 / v18 (opt-in; v15 keeps the two lines above): adds "complex_hv" = horizontal + vertical
# (longitudinal) cleft in the same horn, radsource.us/meniscal-tear-patterns Fig. 10a. The complex share
# rises from 0.10 to 0.18 (split 0.08 / 0.10), taken evenly from horizontal / vertical / oblique.
PATTERNS_V17 = ("horizontal", "vertical", "oblique", "radial", "complex", "complex_hv", "maceration")
PATTERN_P_V17 = (0.22, 0.18, 0.18, 0.10, 0.08, 0.10, 0.14)
MIN_HORN_PX = 20
ALPHA_CLEFT = (0.25, 0.8)
ALPHA_MACERATION = (0.2, 0.6)
FLUID_PERCENTILE = 90


# [EXPLAIN] Edge band of a mask = (mask grown by `width`) minus (mask shrunk by `width`).
def contour_band(whole, width=1):
    """Thin band on the outer edge of a binary mask (inside and outside by `width` px)."""
    if not whole.any():
        return np.zeros_like(whole, dtype=bool)
    outer = ndi.binary_dilation(whole, iterations=width)
    inner = ndi.binary_erosion(whole, iterations=width)
    return outer & ~inner


def _horns(mask_bin):
    lab, n = ndi.label(mask_bin)
    return [lab == k for k in range(1, n + 1) if (lab == k).sum() >= MIN_HORN_PX]


def _long_axis(comp):
    """Unit vector of the horn's long axis (PCA of its pixel coordinates), as (dy, dx)."""
    pts = np.argwhere(comp).astype(np.float64)
    pts -= pts.mean(0)
    if len(pts) < 3:
        return np.array([0.0, 1.0])
    w, v = np.linalg.eigh(np.cov(pts.T))
    return v[:, np.argmax(w)]


def _rotate(vec, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def _cleft(comp, angle_deg, reach_edge, rng):
    """One thin, slightly wavy cleft through the horn, rasterised as a bool mask."""
    H, W = comp.shape
    region = comp if reach_edge else ndi.binary_erosion(comp, iterations=2)
    if region.sum() < 5:
        region = comp
    seeds = np.argwhere(region)
    p0 = seeds[rng.integers(len(seeds))].astype(np.float64)
    d = _rotate(_long_axis(comp), angle_deg)
    perp = np.array([-d[1], d[0]])
    span = float(np.ptp(np.argwhere(comp) @ d)) + 4.0
    amp, freq, phase = rng.uniform(0, 1.5), rng.uniform(0.05, 0.25), rng.uniform(0, 2 * np.pi)
    out = np.zeros((H, W), dtype=bool)
    for t in np.arange(-span, span, 0.5):
        q = p0 + t * d + amp * np.sin(freq * t + phase) * perp
        y, x = int(round(q[0])), int(round(q[1]))
        if 0 <= y < H and 0 <= x < W:
            out[y, x] = True
    width = int(rng.integers(1, 4))           # 1-3 px
    if width > 1:
        out = ndi.binary_dilation(out, iterations=width - 1)
    out &= region
    if reach_edge:                             # make sure it actually opens to the surface
        out |= ndi.binary_dilation(out, iterations=1) & comp & ~ndi.binary_erosion(comp, iterations=1)
    return out


def _maceration(comp, rng):
    """Large irregular bright area covering 40-90% of the horn."""
    inner = ndi.binary_erosion(comp, iterations=1)
    if inner.sum() < 5:
        inner = comp
    field = ndi.gaussian_filter(rng.standard_normal(comp.shape), sigma=rng.uniform(2.0, 5.0))
    vals = field[inner]
    thr = np.quantile(vals, 1.0 - rng.uniform(0.3, 0.8))
    return inner & (field >= thr)


# [EXPLAIN] Pick a tear pattern for ONE horn (anterior or posterior piece of the meniscus) with the probabilities in
# [EXPLAIN] PATTERN_P, then build it as a boolean mask that always lies inside the horn.
def _horn_tear(comp, rng, patterns=None, pattern_p=None):
    # None = the module-level PATTERNS / PATTERN_P (v14 / v15 behaviour, read at call time)
    patterns = PATTERNS if patterns is None else patterns
    pattern_p = PATTERN_P if pattern_p is None else pattern_p
    pat = patterns[rng.choice(len(patterns), p=pattern_p)]
    if pat == "horizontal":
        return _cleft(comp, rng.uniform(-15, 15), rng.random() < 0.3, rng), pat
    if pat == "vertical":
        return _cleft(comp, 90 + rng.uniform(-15, 15), rng.random() < 0.3, rng), pat
    if pat == "oblique":
        return _cleft(comp, rng.choice([-1, 1]) * rng.uniform(30, 60), rng.random() < 0.3, rng), pat
    if pat == "radial":
        return _cleft(comp, 90 + rng.uniform(-10, 10), True, rng), pat
    if pat == "complex":
        a = _cleft(comp, rng.uniform(-15, 15), rng.random() < 0.3, rng)
        b = _cleft(comp, rng.choice([-1, 1]) * rng.uniform(30, 60), rng.random() < 0.3, rng)
        return a | b, pat
    if pat == "complex_hv":                    # Fig. 10a: horizontal + vertical (longitudinal) components
        a = _cleft(comp, rng.uniform(-15, 15), rng.random() < 0.3, rng)
        b = _cleft(comp, 90 + rng.uniform(-15, 15), rng.random() < 0.3, rng)
        return a | b, pat
    return _maceration(comp, rng), pat


# [EXPLAIN] MAIN ENTRY. Input: the 3-slice image and the mask. Output: image with a bright tear painted inside the
# [EXPLAIN] meniscus, the tear mask, and the pattern names. With probability 1-p (or if there is no meniscus) nothing
# [EXPLAIN] changes, so some training slices stay healthy. The caller never changes the label.
def synth_tear(image, mask, rng, p=0.5, neighbor_p=0.6, variants=None, variant_p=None):
    """
    image: (3, H, W) float32 in [0, 1], channel 1 = centre slice.
    mask:  (H, W) int, 0 = background, >0 = meniscus.
    Returns (image_out, tear, patterns): tear is a bool (H, W) mask, always a
    subset of mask > 0; patterns lists the pattern used per painted horn.
    With probability 1 - p, or if there is no meniscus, the image is returned unchanged.
    variants / variant_p: in-horn variants to draw from (None = PATTERNS / PATTERN_P; v17 passes the _V17 ones).
    """
    mask_bin = mask > 0
    tear = np.zeros(mask_bin.shape, dtype=bool)
    if rng.random() >= p or mask_bin.sum() < MIN_HORN_PX:
        return image, tear, []

    horns = _horns(mask_bin)
    if not horns:
        return image, tear, []
    # paint one horn, sometimes both (sagittal slices often show anterior + posterior horn)
    order = rng.permutation(len(horns))
    n_paint = 1 if (len(horns) == 1 or rng.random() < 0.6) else len(horns)
    patterns = []
    for k in order[:n_paint]:
        t, pat = _horn_tear(horns[k], rng, variants, variant_p)
        tear |= t
        patterns.append(pat)
    # [EXPLAIN] Safety: a tear can never leave the meniscus, so the outer boundary is never altered.
    tear &= mask_bin
    if not tear.any():
        return image, tear, []

    img = image.copy()
    fluid = float(np.percentile(img[1], FLUID_PERCENTILE))
    macerated = "maceration" in patterns
    alpha = rng.uniform(*(ALPHA_MACERATION if macerated else ALPHA_CLEFT))
    soft = ndi.gaussian_filter(tear.astype(np.float32), 1.5 if macerated else 0.8)
    soft = np.clip(soft / max(soft.max(), 1e-6), 0.0, 1.0)
    w = (alpha * soft).astype(np.float32)

    def paint(ch, weight):
        target = fluid + rng.normal(0, 0.02, ch.shape).astype(np.float32)   # fluid is not perfectly flat
        return np.clip(ch + weight * (target - ch), 0.0, 1.0)

    # [EXPLAIN] Brighten the CENTRE slice toward 'fluid' brightness: I' = I + w*(fluid - I). Then do the same, shifted a few
    # [EXPLAIN] pixels and weaker, on the neighbouring slices (real tears span several 3.6 mm slices).
    img[1] = paint(img[1], w)
    for c in (0, 2):
        if rng.random() < neighbor_p:
            dy, dx = rng.integers(-2, 3, size=2)
            w_shift = ndi.shift(w, (dy, dx), order=0, mode="constant", cval=0.0)
            img[c] = paint(img[c], w_shift * rng.uniform(0.5, 1.0))
    return img.astype(np.float32), tear, patterns
