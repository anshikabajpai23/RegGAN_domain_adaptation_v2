"""
rim_loss.py
============
Phase 5, G2 — "the meniscus is darker than what surrounds it" as a loss.

Penalises predictions whose predicted interior is NOT darker than a thin ring
just outside it. Needs no labels, so it applies to both the fake and real branches.

THE PRIOR WAS MEASURED BEFORE THIS WAS WRITTEN (doc §5.2, mandatory pre-check).
scripts/precheck_rim_darkness.py on the 13 real training volumes:
    real PD: interior darker on 1.000 of GT-nonempty slices, median gap 0.15-ish
    fake PD: 0.964, median gap 0.1171  -> suggested margin ~0.0293
Both far above the 0.80 gate, so the prior holds on this data and the loss is
not pushing the model toward something false.

Form v1 (doc §5.3), mean ordering with a margin:
    rim   = softDilate(p_men, k) - p_men          # a ring outside the prediction
    mu_in = <img weighted by p_men>               # mean intensity inside
    mu_rim= <img weighted by rim>                 # mean intensity in the ring
    loss  = relu(mu_in - mu_rim + margin)
Zero whenever the interior is darker than the rim by at least `margin`; positive
and proportional otherwise. Everything is differentiable w.r.t. the prediction.

`img` must be the centre slice in the SAME normalisation the model sees.
Brightness augmentation is multiplicative, so the ordering survives it.

MARGIN UNITS. `margin` is in normalised intensity, and the doc sets it to ~25%
of the measured gap. Too large and the model is pushed to over-darken, which
shows up as rising false negatives (doc §5.5) -- the conservative choice is the
smaller of the two cohorts' suggested margins.

NOT IMPLEMENTED: the doc's v2 (boundary-sign form) is only for the case where v1
is confounded by genuinely dark rim tissue (e.g. cortical bone or capsule). With
real PD holding at 1.000 there is no evidence of that, so building v2 now would
be speculative. It stays in the doc if v1's result suggests it.
"""
import torch
import torch.nn.functional as F


def soft_dilate(m, k):
    """Soft morphological dilation of a probability map via max-pooling.
    m: (B,H,W) in [0,1]. Returns (B,H,W)."""
    return F.max_pool2d(m.unsqueeze(1), kernel_size=2 * k + 1, stride=1, padding=k).squeeze(1)


def rim_loss(p_men, img, k=5, margin=0.0293, eps=1e-6, min_area=10.0, reduce=True):
    """p_men: (B,H,W) soft meniscus probability (lateral + medial).
    img:   (B,H,W) centre-slice intensity, model's input scale.
    Returns scalar (reduce=True) or (B,) per-sample loss.

    Zero when the predicted interior is already darker than its rim by at
    least `margin`; positive otherwise.

    min_area GUARDS TWO DEGENERATE CASES, both of which occur in this data and
    both of which were caught by the self-test before this was ever trained:

      1. EMPTY SLICES. If the model correctly predicts nothing (p_men ~ 0),
         then mu_in = mu_rim = 0 and the loss returns exactly `margin` -- a
         constant penalty for being right. Real PD is unfiltered and contains
         many meniscus-free slices, so this would apply on a large fraction of
         the real branch.
      2. NEAR-UNIFORM PREDICTIONS. A flat p_men has an EMPTY rim (dilation
         changes nothing), so mu_rim becomes 0/eps and gradients explode --
         measured at 2e6 before this guard.

    Samples failing either test contribute nothing and are excluded from the
    mean, so the loss only speaks where it has something to say."""
    rim = (soft_dilate(p_men, k) - p_men).clamp(min=0)

    area_in = p_men.sum(dim=(1, 2))
    area_rim = rim.sum(dim=(1, 2))

    mu_in = (p_men * img).sum(dim=(1, 2)) / (area_in + eps)
    mu_rim = (rim * img).sum(dim=(1, 2)) / (area_rim + eps)

    valid = ((area_in > min_area) & (area_rim > min_area)).float()
    per_sample = F.relu(mu_in - mu_rim + margin) * valid

    if not reduce:
        return per_sample
    return per_sample.sum() / valid.sum().clamp(min=1.0)


def meniscus_prob(logits):
    """lateral + medial probability from 3-class logits."""
    p = torch.softmax(logits, dim=1)
    return p[:, 1] + p[:, 2]


@torch.no_grad()
def _selftest():
    """Behavioural checks. The loss must be ~0 when the prior already holds and
    clearly positive when it is violated, or it is not measuring what it claims."""
    torch.manual_seed(0)
    B, H, W = 4, 64, 64

    # a square "meniscus" prediction
    p = torch.zeros(B, H, W)
    p[:, 24:40, 20:44] = 1.0

    # case 1: interior DARK, surroundings BRIGHT -> prior holds -> loss ~0
    img_ok = torch.full((B, H, W), 0.60)
    img_ok[:, 24:40, 20:44] = 0.20
    l_ok = rim_loss(p, img_ok, k=5, margin=0.0293)

    # case 2: interior BRIGHT, surroundings DARK -> prior violated -> loss large
    img_bad = torch.full((B, H, W), 0.20)
    img_bad[:, 24:40, 20:44] = 0.70
    l_bad = rim_loss(p, img_bad, k=5, margin=0.0293)

    # case 3: flat image, no contrast -> loss == margin (nothing to reward)
    img_flat = torch.full((B, H, W), 0.4)
    l_flat = rim_loss(p, img_flat, k=5, margin=0.0293)

    # case 4: EMPTY prediction -> must contribute NOTHING, not `margin`
    p_empty = torch.zeros(B, H, W)
    l_empty = rim_loss(p_empty, img_ok, k=5, margin=0.0293)

    # case 5: UNIFORM prediction (empty rim) -> must contribute nothing, no blowup
    p_uniform = torch.full((B, H, W), 2.0 / 3.0)
    l_uniform = rim_loss(p_uniform, img_ok, k=5, margin=0.0293)

    print(f"  prior holds (dark interior)   : {l_ok.item():.6f}   (want ~0)")
    print(f"  prior violated (bright inside): {l_bad.item():.6f}   (want >> 0)")
    print(f"  flat image, no contrast       : {l_flat.item():.6f}   (want == margin 0.0293)")
    print(f"  EMPTY prediction              : {l_empty.item():.6f}   (want 0, not margin)")
    print(f"  UNIFORM prediction (no rim)   : {l_uniform.item():.6f}   (want 0, no blowup)")

    ok = (l_ok.item() < 1e-6 and l_bad.item() > 0.4
          and abs(l_flat.item() - 0.0293) < 1e-5
          and l_empty.item() < 1e-9 and l_uniform.item() < 1e-9)
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def _gradtest():
    """The loss must push the prediction, and must NOT explode on the degenerate
    all-zeros-logits case that occurs at initialisation."""
    torch.manual_seed(0)
    img = torch.full((1, 32, 32), 0.6)
    img[:, 10:20, 10:20] = 0.15                     # a dark blob to find

    # degenerate: uniform logits -> uniform p_men -> empty rim. Must stay finite.
    flat = torch.zeros(1, 3, 32, 32, requires_grad=True)
    rim_loss(meniscus_prob(flat), img, k=3, margin=0.0293).backward()
    g_flat = float(flat.grad.abs().max())

    def localised_logits(box):
        """Background class dominant everywhere, meniscus class dominant only in
        `box`. Naively setting every class to -4 outside gives a uniform softmax
        (p_men = 2/3 everywhere), which is not a localised prediction at all."""
        lg = torch.full((1, 3, 32, 32), -4.0)
        lg[:, 0] = 4.0                                   # background wins outside
        y0, y1, x0, x1 = box
        lg[:, 0, y0:y1, x0:x1] = -4.0
        lg[:, 1, y0:y1, x0:x1] = 4.0                     # meniscus wins inside
        return lg.clone().requires_grad_(True)

    # correct prediction, sitting ON the dark blob: prior already satisfied,
    # so loss == 0 and zero gradient is the RIGHT answer, not a bug.
    good = localised_logits((10, 20, 10, 20))
    l_good = rim_loss(meniscus_prob(good), img, k=3, margin=0.0293)
    l_good.backward()
    g_good = float(good.grad.abs().max())

    # WRONG prediction, on a BRIGHT patch surrounded by darker tissue: the prior
    # is violated, so the loss must be active with usable finite gradients.
    img_bad = torch.full((1, 32, 32), 0.3)
    img_bad[:, 22:30, 22:30] = 0.85                      # bright where we predict
    bad = localised_logits((22, 30, 22, 30))
    l_bad = rim_loss(meniscus_prob(bad), img_bad, k=3, margin=0.0293)
    l_bad.backward()
    g_bad = float(bad.grad.abs().max())

    print(f"  uniform logits   max|grad| = {g_flat:.3e}   (want 0 -- guarded, was 2e+06)")
    print(f"  correct pred     loss={l_good.item():.4f} max|grad| = {g_good:.3e}   (want 0 -- already satisfied)")
    print(f"  violating pred   loss={l_bad.item():.4f} max|grad| = {g_bad:.3e}   (want finite and > 0)")
    ok = g_flat == 0.0 and g_good == 0.0 and 0.0 < g_bad < 1e3
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("rim_loss self-test:")
    a = _selftest()
    print("rim_loss gradient test:")
    b = _gradtest()
    raise SystemExit(0 if (a and b) else 1)
