# v14 tear-amodal: what changed vs the replica, and why

Replica = `segmentation/finetune_meniscus_v7_replica.py` (reference, 17pt test Dice 0.7780).
v14 = `segmentation/finetune_meniscus_v14_tear_amodal.py` plus `tear_synth.py`, `dataset_2_5d_tear.py`, `model_tear_amodal.py`.
Nothing in the replica files was edited. Whole-wedge losses are imported from the replica script.

## The problem we were trying to solve
On torn menisci the tear shows up as brighter signal inside or at the tip of the dark meniscus. A model trained
only on healthy menisci learned "meniscus = dark wedge", so it can stop at the bright part. Goal: still output the
WHOLE wedge through the tear ("amodal": predict the full object even where part of it looks different), and as a
bonus say which part is tear. There are no tear labels and no public pixel-level torn-meniscus data, so the tears
are synthesized on healthy slices where the full mask is already known.

## 1. Synthetic tears (R1, and the base of all variants)
Replica: the image goes straight from the dataset to the model.
```python
item = self.base[i]                   # image (3,384,384), mask (384,384)   <- all the replica has
```
v14: paint a tear inside the meniscus AFTER the replica's own flips/brightness/noise, keep the mask unchanged.
```python
item = self.base[i]
if self.synth:
    image, tear, _ = synth_tear(image, mask, rng, p=0.5, neighbor_p=0.6)   # tear: bool, subset of mask
whole = mask > 0                      # label is STILL the whole wedge (the amodal target)
```
Inside `synth_tear` (tear_synth.py):
```python
tear &= mask_bin                      # a tear never leaves the meniscus -> outer edge never altered
fluid = percentile(centre slice, 90)  # "fluid brightness" for this slice
img[1] = ch + weight * (fluid - ch)   # blend toward fluid; weight = alpha * soft tear, alpha 0.25-0.8
# neighbours (slices i-1, i+1): same tear, shifted up to 2 px, weaker, with probability 0.6
```
Patterns (per horn): horizontal, vertical, oblique, radial (always reaches the surface), complex (two clefts),
maceration (large patchy bright area). Why each design choice:
- Painted inside the mask only: the model must keep the outer boundary, which is the thing you said it must still find.
- Label unchanged: this is what teaches "bright inside the wedge is still meniscus".
- Applied to the neighbouring slices too: the model sees 3 slices; real tears span 3.6 mm slices, so a tear on
  only the centre slice would look unnatural.
- Soft edges, faint-to-white intensity, patchy maceration: the first version was too bright and blocky (the
  preview showed flat white blobs), so alpha and the blur were reduced.
- Only on training slices: validation stays clean so checkpoint selection matches the replica.

## 2. Contour head (R3, R4, R1c)
Replica: one output, the 3-class logits.
```python
logits = model(x)                                   # (B,3,384,384)
```
v14: same trunk, an extra 16->1 conv reading the same decoder output.
```python
d = self.decoder(self.encoder(x))
out = {"whole": self.segmentation_head(d)}          # the replica's own head, same weights
if self.contour_head is not None:
    out["contour"] = self.contour_head(d)           # nn.Conv2d(16, 1, 3, padding=1)
```
Target and loss:
```python
def contour_band(whole, width=1):
    outer = ndi.binary_dilation(whole, iterations=width)
    inner = ndi.binary_erosion(whole, iterations=width)
    return outer & ~inner                           # thin ring on the wedge's outer edge

terms["contour"] = w_contour(0.5) * BCE_with_logits(out["contour"], band, pos_weight=5)
```
Why: your idea was to separate "the line between the meniscus and the tissue above and below" from whatever is
inside. The contour target is the same for a healthy and a painted slice (painting never changes the outer edge),
so this head is forced to find the boundary independent of tear brightness. pos_weight = 5 because the ring is a
tiny fraction of pixels and an unweighted loss would just predict zeros.

## 3. Visible head (R2, R3, R4)
v14: another 16->1 conv.
```python
self.visible_head = nn.Conv2d(ch, 1, 3, padding=1)
item["visible"] = whole & ~tear                     # wedge minus the painted tear = what is still dark
terms["visible"] = 0.5 * (BCE(logit, visible) + soft_dice_binary(sigmoid(logit), visible))
```
Why: it gives the model two views of the same slice: "whole wedge" (through the tear) and "what is visibly
dark". Their difference is a tear map, with no real tear labels needed:
`tearmap = clip(p_whole_meniscus - p_visible, 0, 1)` (written by `infer_real_pd_tear_amodal.py`).
This is the part that could turn the work into more than a Dice improvement.

## 4. Inclusion loss (R4 only)
```python
p_whole = softmax(out["whole"], dim=1)
p_men   = p_whole[:, 1] + p_whole[:, 2]             # probability of meniscus (lateral + medial)
p_vis   = sigmoid(out["visible"][:, 0])
terms["inclusion"] = 0.1 * relu(p_vis - p_men).mean()
```
Why: the two heads are trained independently, so nothing stops "visible" from firing where "whole" says
background. A tear map computed as whole minus visible would then be nonsense there. relu(p_vis - p_men) is zero
whenever visible <= whole and positive only when visible exceeds whole, so it enforces visible inside whole
without pushing the two together when they already agree. Weight 0.1 so it is a constraint, not the main signal.

## 5. How they combine in the loop
Replica:
```python
logits_f = model(xf);  loss_f = fake_loss_fn(logits_f, yf)
logits_r = model(xr);  loss_r = merged_loss_fn(logits_r, yr)
loss = loss_f + loss_r
```
v14:
```python
out_f = model(xf);  loss_f = fake_loss_fn(out_f["whole"], yf)        # unchanged replica loss
aux_f, _ = aux_losses(out_f, fake_batch, args, device)               # NEW (0 for R1)
out_r = model(xr);  loss_r = merged_loss_fn(out_r["whole"], yr)      # unchanged replica loss
aux_r, _ = aux_losses(out_r, real_batch, args, device)               # NEW (0 for R1)
loss = loss_f + loss_r + aux_f + aux_r
```
So R1's loss is identical to the replica's; only the input images differ. Checkpoint selection, optimizer,
schedule, seed and data are the replica's.

## Which run switches on what
| Run | tear synth | visible head | contour head | inclusion loss |
|---|---|---|---|---|
| R1 | yes | | | |
| R1c | yes | | yes | |
| R2 | yes | yes | | |
| R3 | yes | yes | yes | |
| R4 | yes | yes | yes | yes |

## What is NOT shown yet
None of these have shown a benefit. Whether synthetic tears help on real torn menisci, whether the contour and
inclusion terms add anything beyond R1, and whether the tear map lands on real tears are all open until the
runs are evaluated on the held-out torn patients.
