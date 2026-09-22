# Phase 5 — Implementation Specs for Novel Ideas

**Created:** 2026-09-22. Companion to `ISBI_ideas.md` (what and why) and `reggan_dice_debug.md` (evidence).
**Purpose:** enough detail that another chat can implement each idea correctly, as a one-change experiment, without re-deriving the reasoning.

Code below is written against the interfaces described in `DEBUG_HANDOFF.md`, not against the actual files. Every ⚠ line is an assumption to verify by reading the code before running.

---

## 0. Protocol shared by every idea

| Rule | Detail |
|---|---|
| Baseline | The current frozen best config. As of 2026-09-22 that is **ImageNet-init ResNet34 U-Net, 15 real PD, replica hyperparameters** (0.81). If the "fourth cell" (ImageNet + fake + real) turns out better, it becomes the baseline. Every idea = baseline + exactly one change. |
| Seed | 42. Repeat with seed 7 before claiming any gain. Noise floor per patient ≈ 0.023. |
| Eval | 2 real-val patients, per-slice dumps (`per_slice_dice.csv` format), edge/core/empty split. **No test-set reads.** |
| Zero-init rule | Any new module or channel must be initialized so the model is **numerically identical to the baseline at step 0**. Then the only thing that can differ is training. Each spec has a check for this. |
| Report | Results-only markdown, commands, stdout-sourced numbers. |

⚠ Verify first, once, for all ideas:
- Stack shape returned by the datasets: expected `(3, 384, 384)` float, center slice at channel 1.
- Model construction call: expected `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=3)`.
- Loss for real branch: `MergedLoss` (CE on merged lateral+medial prob + soft Dice against binary GT). Loss for fake branch: `CrossEntropyLoss(weight=[0.1,1.5,1.5]) + SoftDice`.
- Where the volume index and slice index are available inside `__getitem__` (needed for E1, E2, A3, D). If the dataset only holds a flat list of slice files, the filename must encode `patient` and `slice_idx` and the volume length must be looked up from a table built in `__init__`.
- Left vs right knee: **whether laterality is known per volume.** Sagittal slice order runs medial→lateral for one side and lateral→medial for the other. This affects E1/E2/A3/D; see §1.2.

---

## 1. E1 — Slice position as an input channel

### 1.1 Goal
Give the network the one piece of through-plane information that survived check 2: where along the slice axis this slice sits. The model currently decides "is the meniscus still here" from in-plane appearance only.

### 1.2 The position feature (decision)
Use the **symmetric** position, not the raw fraction:

```
z_frac = slice_idx / (n_slices - 1)          # 0..1
pos    = 2 * abs(z_frac - 0.5)               # 0 at the volume centre (notch), 1 at the outer edges
```

Reason: raw `z_frac` flips meaning between left and right knees (medial meniscus is at low z on one side, high z on the other). `pos` is laterality-invariant and still encodes the two places where the errors are: the outer cliffs (pos→1) and the notch (pos→0). If laterality is known and consistent, a second channel with raw `z_frac` can be added as a follow-up run.

Both DESS (160 slices) and real PD (36) map to the same 0..1 range, so the feature means the same thing in both domains. ⚠ This assumes the field of view covers the knee similarly in both; check that the meniscus spans occupy roughly the same `pos` range in both cohorts (Part A tables in `debug4_results.md` give real-PD spans; compute the same for 10 DESS volumes).

### 1.3 Dataset change
In both `dataset_2_5d_v2.py` (fake) and `dataset_2_5d_realpd.py` (real), after the stack is built:

```python
# __init__: build {patient_id: n_slices} once
self.n_slices = {}   # ⚠ fill from the file listing or a saved index

# __getitem__:
stack = ...                                      # (3, H, W) float32, already augmented
z_frac = slice_idx / max(self.n_slices[pid] - 1, 1)
pos    = 2.0 * abs(z_frac - 0.5)
pos_ch = torch.full((1, stack.shape[1], stack.shape[2]), pos, dtype=stack.dtype)
stack  = torch.cat([stack, pos_ch], dim=0)       # (4, H, W)
```

Augmentation note: flips do not change `pos`. Do **not** apply brightness/noise augmentation to the position channel; append it after augmentation, as above.

### 1.4 Model change with zero-init
Build the 3-channel pretrained model, then widen the first conv by hand so the 4th input channel starts at zero weight:

```python
import torch, torch.nn as nn, segmentation_models_pytorch as smp

def build_model_with_pos(classes=3):
    m = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=classes)
    conv = m.encoder.conv1                            # ⚠ first conv of resnet34 in smp is encoder.conv1
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3] = conv.weight               # pretrained RGB weights
        # channel 3 (position) stays zero -> identical output to baseline at step 0
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    m.encoder.conv1 = new
    return m
```

Do **not** use `smp.Unet(in_channels=4)` directly: smp re-initialises the first conv by tiling and rescaling the pretrained weights, which changes the baseline.

**Zero-init check** (run once before training):
```python
m3 = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=3).eval()
m4 = build_model_with_pos().eval()
x  = torch.randn(2, 3, 384, 384); pos = torch.rand(2, 1, 384, 384)
assert torch.allclose(m3(x), m4(torch.cat([x, pos], 1)), atol=1e-5)
```

### 1.5 Sequence model variant
If the baseline is the ConvLSTM model, append the position channel to **each timestep's** slice input the same way. The zero-init rule applies to whatever the first conv is in that model.

### 1.6 Inference
`infer_real_pd_v3.py` must build the same 4th channel from the real volume's `n_slices`. Same formula. Same for TTA flips (position channel is not flipped).

### 1.7 What to report
Standard report plus: edge-bucket FP and FN vs baseline. Also the learned weight norm of `conv1.weight[:, 3]` at the end of training: if it stays ≈0, the model found no use for position.

### 1.8 Reads as
Edge-bucket error down, Dice up ≥ noise floor → position is usable through-plane information; proceed to E2. Weight norm ≈ 0 and no change → the model cannot exploit position at the input; try E2 (gating) before giving up.

---

## 2. E2 — Slice position into gating (FiLM at the bottleneck)

### 2.1 Goal
Same information as E1, injected where the network has semantic features rather than raw pixels. Run only after E1 shows a signal or a null with zero weights.

### 2.2 Module

```python
class PosFiLM(nn.Module):
    """gamma, beta = MLP(pos); features * (1 + gamma) + beta. Zero-init so it is the identity at step 0."""
    def __init__(self, channels, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, feat, pos):                     # feat (B,C,h,w), pos (B,1)
        gamma, beta = self.mlp(pos).chunk(2, dim=1)
        return feat * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

class UnetPosFiLM(nn.Module):
    def __init__(self, base: smp.Unet):
        super().__init__()
        self.base = base
        c_bottleneck = base.encoder.out_channels[-1]  # ⚠ 512 for resnet34
        self.film = PosFiLM(c_bottleneck)
    def forward(self, x, pos):                        # pos (B,1) float in [0,1]
        feats = self.base.encoder(x)                  # list of feature maps
        feats[-1] = self.film(feats[-1], pos)
        dec = self.base.decoder(*feats)               # ⚠ smp.Unet.forward: encoder -> decoder(*feats) -> segmentation_head
        return self.base.segmentation_head(dec)
```

Dataset returns `pos` as a scalar tensor alongside the stack (no extra channel). Training loop passes it: `logits = model(x, pos)`.

**Zero-init check:** with FiLM zero-initialised, `UnetPosFiLM(m)(x, pos)` must equal `m(x)` to 1e-5.

### 2.3 Sequence-model variant
Apply FiLM to the ConvLSTM hidden state at each timestep, or concatenate `pos` to the aggregator's input. Prefer FiLM on the hidden state: it lets position change *how* neighbours are combined, which is the point.

### 2.4 Reads as
Compare to E1. If E2 > E1 by more than noise, position is best used at the semantic level; this becomes the method. If equal, keep E1 (simpler).

---

## 3. A3 — Slice-position weighted loss (two opposite variants)

### 3.1 Goal
Change how much each slice contributes to the loss, based on its distance to the nearest span end. Variant (a) upweights the ends; variant (b) downweights them. Check 4A (labels at the ends are inconsistent) predicts (b) is the right one, but both are one retrain, so run both.

### 3.2 Per-slice weight, computed once at prep
For each training volume with a GT mask (fake PD via DESS masks; real PD via the 13 training masks):

```python
import numpy as np

def slice_weights(mask_vol, variant, alpha=1.0, beta=0.7, tau_mm=3.6, spacing_mm=0.8):
    """mask_vol: (Z,H,W) binary meniscus. Returns w (Z,) float."""
    nonempty = mask_vol.reshape(mask_vol.shape[0], -1).any(1)
    idx = np.where(nonempty)[0]
    # span ends = first/last index of each contiguous run
    ends = []
    if len(idx):
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for r in runs: ends += [r[0], r[-1]]
    z = np.arange(mask_vol.shape[0])
    d_mm = (np.abs(z[:, None] - np.array(ends)[None, :]).min(1) * spacing_mm) if ends else np.full(len(z), 1e9)
    k = np.exp(-d_mm / tau_mm)                      # 1 at a span end, ~0.37 one real-PD slice away
    if variant == "up":   return 1.0 + alpha * k    # ends count up to 2x
    if variant == "down": return np.clip(1.0 - beta * k, 0.3, 1.0)   # ends count as little as 0.3x
```

`spacing_mm` = 0.8 for DESS-derived fake PD, 3.6 for real PD. Using mm keeps the two cohorts comparable (Corrections C1). Save `w` per volume to a small `.npy` or a column in the slice index; `__getitem__` returns `w[slice_idx]` with the sample.

Slices with **empty** GT get `w` from the same formula (distance to nearest end), which is what we want: an empty slice adjacent to a span is exactly where hallucination happens, and variant (a) upweights it, (b) downweights it.

### 3.3 Weighted loss
Both losses must produce a per-sample value before weighting.

```python
def weighted_ce_dice(logits, target, w, ce_weight, eps=1e-6):
    # logits (B,3,H,W), target (B,H,W) long, w (B,)
    ce = F.cross_entropy(logits, target, weight=ce_weight, reduction="none").mean(dim=(1, 2))  # (B,)
    p = logits.softmax(1)
    dice = []
    for c in (1, 2):
        pc, tc = p[:, c], (target == c).float()
        inter = (pc * tc).sum((1, 2)); den = pc.sum((1, 2)) + tc.sum((1, 2))
        dice.append(1 - (2 * inter + eps) / (den + eps))
    dice = torch.stack(dice, 1).mean(1)                                                        # (B,)
    return ((ce + dice) * w).sum() / w.sum()
```

⚠ `MergedLoss` for the real branch needs the same treatment: reduce to per-sample, multiply by `w`, normalise by `w.sum()`. Normalising by `w.sum()` (not `B`) keeps the loss scale equal to the baseline's on average.

### 3.4 Soft-target alternative for variant (b)
Instead of downweighting, keep `w=1` and soften the target on end slices: `t = 0.5 * onehot(gt)` for slices with `k > 0.5`, using a soft cross-entropy `-(t * log_softmax).sum(1)`. This says "half-credit here" rather than "care less here." Run as a third variant only if (b) shows signal.

### 3.5 Reads as
(a) up: edge-bucket Dice up **and** val Dice up → the model could learn the ends and just needed pressure. Edge FP up → it learned the label noise; discard.
(b) down: val Dice up with core unchanged → the ends were poisoning training; adopt. Both null → the ends are label-limited and only relabelling (P3) moves them.

---

## 4. D — Position-conditioned shape prior

### 4.1 Goal
A template of what the meniscus cross-section looks like at each position along its own span, used as a weak regulariser. **Run only if E1 or E2 shows the model can use position.** Otherwise skip.

### 4.2 Hard constraint to know before starting
The template is indexed by position **within the span** (0 = outer end, 1 = notch end), which is known from GT at training time but **not at inference**. So D can only be a training-time regulariser, or a two-pass method at inference (predict spans, then apply). Keep it training-time only in the first run.

### 4.3 Build the template (registration-free version)
Do not try to align full slices across patients. Use two shape statistics that need no alignment:

```python
K = 10   # bins of within-span position
# per training volume, per span (lateral, medial separately, from DESS labels 5/6)
for each slice s in span:
    q = (s - span_start) / (span_end - span_start)      # 0..1 within the span
    if this span is the medial one and z runs lateral->medial, q = 1 - q   # ⚠ orient so 0 = outer, 1 = notch
    bin = min(int(q * K), K - 1)
    area[bin].append(mask[s].sum())
    n_cc[bin].append(number_of_connected_components(mask[s]))   # 8-connectivity
A_mean[bin], A_std[bin] = stats(area[bin]); C_mean[bin] = mean(n_cc[bin])
```

That yields, per bin: expected area (in pixels) and expected number of pieces (≈1 at the periphery, ≈2 mid-compartment where the two horns separate). Save as a small JSON.

### 4.4 Loss
For a training slice with known `bin`:

```python
area_pred = p_men.sum((1, 2))                                     # soft area, (B,)
L_area = ((area_pred - A_mean[bin]) / (A_std[bin] + 1)).pow(2).mean()
loss = base_loss + lam * L_area        # lam = 0.01 to start; the term must stay < 10% of base_loss
```

Do not add the component-count term in the first run; it needs a differentiable proxy (soft skeleton or persistent homology) and clDice already showed ~0 on this data.

### 4.5 Reads as
Any gain ≥ noise floor with `lam=0.01` → the position-indexed area prior is informative; try `lam=0.05`. Null → drop D; E1/E2 already carry the position information.

---

## 5. G2 — Region-darker-than-rim loss

### 5.1 Goal
Encode the one intensity fact about PD that the model appears to ignore: the meniscus is darker than the tissues it touches (cartilage, fluid). Penalise predictions whose interior is not darker than their immediate surroundings.

### 5.2 Mandatory pre-check (no training, 30 min)
The prior must hold on the data before it is used as a loss. On the 13 real training volumes with GT:

```python
def rim_stats(img, gt, k=5):
    """img (H,W) float, gt (H,W) {0,1}. Returns mean inside, mean in a k-px ring outside."""
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(gt, iterations=k) & ~gt.astype(bool)
    return img[gt > 0].mean(), img[ring].mean()
```

Report: fraction of GT-nonempty slices where `mean_inside < mean_ring`. **If below ~0.8, do not implement G2**: the ordering does not hold on this sequence (possible if PD is fat-saturated and the peripheral rim is dark capsule or cortical bone). Also report the same for fake PD on 10 volumes; if fake PD violates it while real PD holds, that is a GAN intensity error and the fix is G1 (tissue-conditioned calibration), not a loss.

⚠ Also record whether the real PD is fat-saturated (`SAG_PD_TSE` vs `SAG_PD_FS`?); it changes which tissues are bright.

### 5.3 Differentiable loss, form v1 (mean ordering with margin)

```python
def rim_loss(p_men, img, k=5, margin=0.05, eps=1e-6):
    """p_men (B,H,W) soft meniscus prob (lat+med); img (B,H,W) centre-slice intensity in the model's input scale."""
    m   = p_men
    dil = F.max_pool2d(m[:, None], kernel_size=2 * k + 1, stride=1, padding=k)[:, 0]   # soft dilation
    rim = (dil - m).clamp(min=0)
    mu_in  = (m * img).sum((1, 2)) / (m.sum((1, 2)) + eps)
    mu_rim = (rim * img).sum((1, 2)) / (rim.sum((1, 2)) + eps)
    return F.relu(mu_in - mu_rim + margin).mean()
```

`img` is channel 1 of the input stack (the centre slice), **after** the same normalisation the model sees; brightness augmentation is multiplicative so ordering is preserved. `margin` is in normalised intensity units; set it to ~25% of the typical `(mu_rim − mu_in)` gap measured in §5.2.

Total loss: `base_loss + lam * rim_loss`, `lam` chosen so the rim term is 5–10% of `base_loss` at epoch 1. Apply on **both** fake and real branches (needs no labels).

### 5.4 Form v2, if v1 is confounded by dark rim tissue
Penalise boundary segments where the image does **not** brighten going outward:

```python
def boundary_sign_loss(p_men, img, eps=1e-6):
    gm = torch.stack(torch.gradient(p_men, dim=(1, 2)), 0)    # (2,B,H,W) grad of soft mask (points inward)
    gi = torch.stack(torch.gradient(img,   dim=(1, 2)), 0)    # grad of intensity (points toward brighter)
    dot = (gm * gi).sum(0)                                     # <0 where brighter outward (good)
    w   = gm.norm(dim=0)                                       # boundary weight
    return (w * F.relu(dot)).sum((1, 2)).div(w.sum((1, 2)) + eps).mean()
```

This only fires on the boundary and only where intensity fails to rise outward, so a correct boundary against dark capsule contributes little (flat or small gradient). Still validate with §5.2-style statistics on GT boundaries first.

### 5.5 Reads as
Over-segmentation FP on meniscus slices down, FN unchanged → the model was ignoring intensity and the prior corrected it. FN up → margin too large or prior wrong at the periphery; halve `margin`, or switch to v2. Null → the model already uses intensity; the leak was a boundary/shape problem or a label problem.

---

## 6. G6 — Tissue-ordering loss inside RegGAN (parked; approach only, no code)

**Why parked.** Requires changing the translation stage, retraining the GAN (days on A100), re-translating 155 volumes, then re-running segmentation. Two to three weeks. MICCAI-scale, not ISBI-scale. Do G1 (tissue-conditioned calibration of the existing fake PD, in `ISBI_ideas.md`) first; it captures most of the benefit with no GAN retrain and tells you whether the GAN's tissue ordering is actually wrong.

**Approach.**
1. `dataset.py` in the RegGAN stage currently loads no masks. Add the DESS 6-class mask for each source slice (they exist in `preprocessed_v3/masks` or the `.seg.nrrd` files).
2. Measure once, on real PD with the 15 masks and rough thresholds for other tissues, the expected **ordering** of mean intensities: e.g. fluid > cartilage > muscle > meniscus > cortical bone (verify on this sequence; fat-sat changes marrow and fat).
3. In the generator loss, after producing `fake_B` from a DESS slice with mask `M`, compute per-tissue mean intensity `mu_t = mean(fake_B[M == t])` for each tissue present in the slice.
4. Add pairwise hinge losses for each ordered pair `(t_bright, t_dark)` in the expected ordering: `L_order = Σ relu(mu_dark − mu_bright + margin_pair)`, with margins taken from the real-PD gaps measured in step 2, scaled down (e.g. 50%) so the GAN is not forced to exact values.
5. Weight `λ_order` small (start at 1.0 relative to `λ_cycle = 10`) and apply only to `G_AB` (DESS→PD direction). Leave the discriminator untouched: it still judges texture; this term judges only tissue-level brightness ordering.
6. Optional stronger variant: replace ordering hinges with a per-tissue histogram-matching term (differentiable soft histogram or 1-D sliced Wasserstein between `fake_B[M==t]` and a stored real-PD reference histogram for tissue `t`). More precise, more brittle.
7. Evaluate: per-tissue mean intensities of the new fake PD vs real PD (the G1 diagnostic table), boundary distance and Jacobian folding (must not regress), then one segmentation run.

**Nearest published work:** shape-consistent CycleGAN (Zhang et al. 2018) and SynSeg-Net (Huo et al. 2018) keep *labels* consistent through translation. Keeping per-tissue *intensity ordering* consistent is not found.

**Prerequisite:** the R-network pairing issue (§3.7 of the handoff) should be fixed or explicitly limited before adding any new loss to the translation stage, otherwise the new term sits on a component used out of distribution.
