"""
tear_patterns.py
================
v15 synthetic tear-pattern library, built on healthy training slices.

Label rule ("all meniscus tissue where it actually is"):
  * tear signal inside the meniscus  -> still meniscus (label unchanged there)
  * tissue that is removed            -> background
  * tissue that is displaced          -> meniscus at its NEW position, background at the old one
So, unlike v14, several modes CHANGE the mask, and the caller must train on the returned mask.

Modes (one horn = one connected piece of the meniscus mask on the centre slice), mapped to the
patterns on radsource.us/meniscal-tear-patterns:
  inplace   horizontal / longitudinal / oblique / radial-interior / complex / maceration signal
            inside the meniscus (re-uses tear_synth.synth_tear); label unchanged
  ragged    irregular, lumpy outline (torn, frayed edge); label = the new outline
  lobe      a rounded dark lobe attached to the horn (irregular torn tissue, like the dark
            pocket the replica rejected); label includes the lobe
  truncate  blunted / truncated inner tip (small radial tear at the free edge); tip removed
  absent    horn missing on the centre slice only (large radial / root tear on 1-2 slices)
  flap      part of the horn split off along its long axis and displaced into the gutter
  bucket    inner part of the horn split off and displaced toward the centre (bucket-handle)

Geometry is computed on the centre slice and applied to all three input slices (a tear spans
several 3.6 mm slices), except `absent`, which only affects the centre slice.

These are approximations of real tear appearance. They must be checked by a radiologist
(scripts/preview_tear_patterns.py) before being trusted; the v14 painted tears did not
resemble real tears on the held-out torn patients.
"""
import numpy as np
from scipy import ndimage as ndi

from tear_synth import (_horns, _long_axis, synth_tear, FLUID_PERCENTILE, MIN_HORN_PX)

MODES = ("inplace", "ragged", "lobe", "truncate", "absent", "flap", "bucket")
MODE_P = (0.30, 0.15, 0.15, 0.10, 0.05, 0.12, 0.13)


# ───────────────────────────── intensity helpers ─────────────────────────────

def _soft(region, sigma):
    w = ndi.gaussian_filter(region.astype(np.float32), sigma)
    return np.clip(w / max(float(w.max()), 1e-6), 0.0, 1.0)


def _fill(ch, region, mu, sd, rng, sigma=0.8):
    """Paint `region` of one channel with texture of mean mu / std sd, soft-edged."""
    if not region.any():
        return ch
    tex = ndi.gaussian_filter(mu + rng.normal(0, max(sd, 0.01), ch.shape), 0.7).astype(np.float32)
    w = _soft(region, sigma)
    return np.clip(ch * (1 - w) + tex * w, 0.0, 1.0)


def _stats(ch, region, fallback):
    v = ch[region]
    return (float(v.mean()), float(v.std())) if v.size >= 5 else (fallback, 0.02)


def _outside_ring(comp, inner=2, outer=5):
    return ndi.binary_dilation(comp, iterations=outer) & ~ndi.binary_dilation(comp, iterations=inner)


def _removal_target(ch, comp, rng):
    """What fills the space where tissue was removed: a blend between the surrounding tissue and
    joint fluid, and ALWAYS clearly brighter than the meniscus itself, so the image never shows
    meniscus-dark tissue where the label says background (that would teach the wrong thing)."""
    ring_mu, ring_sd = _stats(ch, _outside_ring(comp), float(ch.mean()))
    fluid = float(np.percentile(ch, FLUID_PERCENTILE))
    men_mu, _ = _stats(ch, comp, 0.1)
    t = rng.uniform(0.35, 0.75)
    mu = max((1 - t) * ring_mu + t * fluid, men_mu + 0.12)
    return min(mu, fluid), max(0.7 * ring_sd, 0.02)


# ───────────────────────────── geometry helpers ─────────────────────────────

def _sdf(comp):
    """Signed distance, > 0 inside the horn."""
    return ndi.distance_transform_edt(comp) - ndi.distance_transform_edt(~comp)


def _tip_direction(comp, others, rng):
    """Unit vector (dy, dx) pointing from this horn toward its free-edge tip. On sagittal
    slices the two horns' tips point toward each other, so use the other horn if present."""
    c = np.argwhere(comp).mean(0)
    if others:
        o = np.argwhere(others[0]).mean(0)
        d = o - c
        n = np.linalg.norm(d)
        if n > 1e-6:
            return d / n
    d = _long_axis(comp)
    return d if rng.random() < 0.5 else -d


def _keep_touching(new, comp):
    """Keep only the connected pieces of `new` that overlap the original horn."""
    lab, n = ndi.label(new)
    keep = np.zeros_like(new)
    for k in range(1, n + 1):
        piece = lab == k
        if (piece & comp).any():
            keep |= piece
    return keep


def _shift_mask(m, dy, dx):
    return ndi.shift(m.astype(np.float32), (dy, dx), order=0, mode="constant", cval=0.0) > 0.5


# ───────────────────────────── modes ─────────────────────────────

def _mode_ragged(img, mask, comp, cls, rng):
    noise = ndi.gaussian_filter(rng.standard_normal(comp.shape), rng.uniform(1.5, 3.0))
    noise /= noise.std() + 1e-6
    new = _keep_touching((_sdf(comp) + rng.uniform(1.5, 3.5) * noise) > 0, comp)
    added, removed = new & ~comp, comp & ~new
    for c in range(3):
        mu, sd = _stats(img[c], comp & ~removed, 0.1)
        img[c] = _fill(img[c], added, mu, sd, rng)
        rmu, rsd = _stats(img[c], _outside_ring(comp), float(img[c].mean()))
        img[c] = _fill(img[c], removed, rmu, rsd, rng)
    mask[removed] = 0
    mask[added] = cls
    return img, mask


def _mode_lobe(img, mask, comp, cls, rng):
    edge = np.argwhere(comp & ~ndi.binary_erosion(comp))
    if len(edge) == 0:
        return img, mask
    py, px = edge[rng.integers(len(edge))]
    gy, gx = np.gradient(_sdf(comp))
    out = -np.array([gy[py, px], gx[py, px]])               # outward normal
    out = out / (np.linalg.norm(out) + 1e-6)
    r1, r2 = rng.uniform(3, 8), rng.uniform(3, 8)
    cy, cx = np.array([py, px]) + out * 0.6 * min(r1, r2)
    yy, xx = np.mgrid[:comp.shape[0], :comp.shape[1]]
    ang = rng.uniform(0, np.pi)
    ry, rx = (yy - cy) * np.cos(ang) + (xx - cx) * np.sin(ang), -(yy - cy) * np.sin(ang) + (xx - cx) * np.cos(ang)
    lobe = ((ry / r1) ** 2 + (rx / r2) ** 2 <= 1.0) & ~comp
    for c in range(3):
        mu, sd = _stats(img[c], comp, 0.1)
        img[c] = _fill(img[c], lobe, mu * rng.uniform(0.6, 1.0), sd, rng)   # as dark or darker than the horn
    mask[lobe] = cls
    return img, mask


def _mode_truncate(img, mask, comp, others, rng):
    d = _tip_direction(comp, others, rng)
    pts = np.argwhere(comp)
    proj = pts @ d
    thr = np.quantile(proj, 1.0 - rng.uniform(0.15, 0.45))
    removed = np.zeros_like(comp)
    sel = pts[proj > thr]
    removed[sel[:, 0], sel[:, 1]] = True
    for c in range(3):
        mu, sd = _removal_target(img[c], comp, rng)
        img[c] = _fill(img[c], removed, mu, sd, rng, sigma=1.1)
    mask[removed] = 0
    return img, mask


def _mode_absent(img, mask, comp, rng):
    mu, sd = _removal_target(img[1], comp, rng)
    img[1] = _fill(img[1], ndi.binary_dilation(comp), mu, sd, rng, sigma=1.8)
    mask[comp] = 0
    return img, mask


def _split_and_move(img, mask, comp, frag, cls, dy, dx, rng):
    """Remove `frag` from its place (fill), paste its pixels at (dy, dx). Returns img, mask."""
    moved = _shift_mask(frag, dy, dx)
    if not moved.any():
        return img, mask
    for c in range(3):
        src = np.where(frag, img[c], 0.0)
        moved_vals = ndi.shift(src, (dy, dx), order=0, mode="constant", cval=0.0)
        mu, sd = _removal_target(img[c], comp, rng)
        img[c] = _fill(img[c], frag, mu, sd, rng, sigma=1.2)
        w = _soft(moved, 0.6) * moved
        img[c] = np.clip(img[c] * (1 - w) + moved_vals * w, 0.0, 1.0)
    mask[frag] = 0
    mask[moved] = cls
    return img, mask


def _mode_flap(img, mask, comp, cls, rng):
    """Horizontal split along the long axis; one side is the flap, displaced across the axis."""
    pts = np.argwhere(comp).astype(float)
    c0 = pts.mean(0)
    a = _long_axis(comp)
    perp = np.array([-a[1], a[0]])
    side = (pts - c0) @ perp
    cut = np.quantile(side, rng.uniform(0.3, 0.7))
    sgn = 1 if rng.random() < 0.5 else -1
    sel = pts[(side - cut) * sgn > 0].astype(int)
    frag = np.zeros_like(comp)
    frag[sel[:, 0], sel[:, 1]] = True
    if frag.sum() < 5:
        return img, mask
    dist = rng.uniform(4, 12)
    dy, dx = sgn * perp * dist
    return _split_and_move(img, mask, comp, frag, cls, dy, dx, rng)


def _mode_bucket(img, mask, comp, others, cls, rng):
    """Inner (tip-side) part split off and displaced toward the centre, between the horns."""
    d = _tip_direction(comp, others, rng)
    pts = np.argwhere(comp)
    proj = pts @ d
    thr = np.quantile(proj, 1.0 - rng.uniform(0.3, 0.5))
    sel = pts[proj > thr]
    frag = np.zeros_like(comp)
    frag[sel[:, 0], sel[:, 1]] = True
    if frag.sum() < 5:
        return img, mask
    dy, dx = d * rng.uniform(8, 25)
    return _split_and_move(img, mask, comp, frag, cls, dy, dx, rng)


# ───────────────────────────── entry point ─────────────────────────────

def synth_pattern(image, mask, rng, p=0.5, modes=MODES, mode_p=MODE_P, neighbor_p=0.6):
    """
    image: (3, H, W) float32 in [0, 1] (slices i-1, i, i+1); mask: (H, W) int (0 bg, >0 meniscus).
    Returns (image_out, mask_out, mode). With probability 1 - p, or if there is no meniscus,
    returns the inputs unchanged and mode "none". mask_out follows the label rule above.
    """
    mode_p = np.asarray(mode_p, float) / np.sum(mode_p)
    if rng.random() >= p or (mask > 0).sum() < MIN_HORN_PX:
        return image, mask, "none"
    horns = _horns(mask > 0)
    if not horns:
        return image, mask, "none"
    mode = str(rng.choice(modes, p=mode_p))
    if mode == "inplace":
        img, _, pats = synth_tear(image, mask, rng, p=1.0, neighbor_p=neighbor_p)
        return img, mask, "inplace"
    k = int(rng.integers(len(horns)))
    comp = horns[k]
    others = [h for j, h in enumerate(horns) if j != k]
    cls = int(np.bincount(mask[comp].astype(np.int64)).argmax())
    img, msk = image.copy().astype(np.float32), mask.copy()
    if mode == "ragged":
        img, msk = _mode_ragged(img, msk, comp, cls, rng)
    elif mode == "lobe":
        img, msk = _mode_lobe(img, msk, comp, cls, rng)
    elif mode == "truncate":
        img, msk = _mode_truncate(img, msk, comp, others, rng)
    elif mode == "absent":
        img, msk = _mode_absent(img, msk, comp, rng)
    elif mode == "flap":
        img, msk = _mode_flap(img, msk, comp, cls, rng)
    elif mode == "bucket":
        img, msk = _mode_bucket(img, msk, comp, others, cls, rng)
    return img.astype(np.float32), msk, mode
