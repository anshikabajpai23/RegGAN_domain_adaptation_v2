# RegGAN Domain Adaptation — Project Knowledge Base
> Use this file to generate interview questions, prep answers, and recall technical details.

---

## Project Goal
Adapt **DESS knee MRI** (SKM-TEA dataset) to look like **PD-weighted knee MRI** (IU dataset) using unpaired image-to-image translation, with minimum deformation — specifically preserving meniscus anatomy for downstream segmentation tasks.

**Why:** IU PD dataset has no segmentation masks. SKM-TEA DESS dataset has masks. By translating DESS→PD, we can use DESS masks to train a segmentation model that works on PD images (transfer learning across modalities).

---

## Datasets

| | DESS (SKM-TEA) | PD (IU) |
|---|---|---|
| Path | `skm-tea-dataset/dess-files` | `iu-dataset/pd-files` |
| Volumes | ~69 | ~69 |
| Raw shape | (512, 512, 160) | (384, 384, 36) |
| Voxel spacing | 0.31 × 0.31 × 0.80 mm | 0.39 × 0.39 × 3.60 mm |
| Orientation | P,S,R (sagittal) | P,S,R (sagittal) |
| Slices extracted | ~11,040 | ~2,271 |
| Masks | SKM-TEA segmentation masks | None |
| Mask labels | 1-6 (cartilage, meniscus etc.) | N/A |
| Meniscus labels | 5 (lateral), 6 (medial) | N/A |

---

## Why RegGAN over CycleGAN

| | CycleGAN | RegGAN |
|---|---|---|
| Deformation | High (>2px, uncontrolled) | Sub-pixel (0.055px at meniscus) |
| Topology violations | Yes (folding artifacts) | 0% |
| Anatomy preservation | Poor | Excellent |
| Extra component | None | Registration network R |
| Key insight | Cycle consistency only | Cycle + registration constraint |

**CycleGAN failure:** Showed 40× higher deformation than RegGAN — unacceptable for medical imaging where anatomy must be preserved exactly.

---

## Architecture

### Generator G_AB and G_BA (ResNet-based)
```
Input: (B, 1, 384, 384)  — grayscale MRI slice

Encoder:
  ReflectionPad2d(3)
  Conv2d(1, ngf, 7)  → InstanceNorm2d → ReLU
  Conv2d(ngf, ngf*2, 3, stride=2)  → InstanceNorm2d → ReLU
  Conv2d(ngf*2, ngf*4, 3, stride=2)  → InstanceNorm2d → ReLU

Residual Blocks: 9 × ResBlock(ngf*4)
  Each ResBlock: ReflectionPad → Conv → InstanceNorm → ReLU → Conv → InstanceNorm + skip

Decoder:
  UpsampleConv(ngf*4, ngf*2)  — bilinear upsample + Conv (no checkerboard)
  UpsampleConv(ngf*2, ngf)
  ReflectionPad2d(3)
  Conv2d(ngf, 1, 7)
  Tanh()

Output: (B, 1, 384, 384)  range [-1, 1]
```

**Key params:**
- `ngf = 48` (base filters)
- `n_res = 9` (residual blocks)
- InstanceNorm2d (not BatchNorm — better for unpaired translation)
- ReflectionPad (not ZeroPad — avoids border artifacts)
- UpsampleConv instead of ConvTranspose2d (avoids checkerboard artifacts)

### Discriminator D_A and D_B (PatchGAN)
```
PatchGAN 70×70 — classifies overlapping patches as real/fake
Input: (B, 1, 384, 384)
Output: (B, 1, H', W')  — patch-level real/fake map

4 layers of Conv2d with stride 2
ndf = 48
InstanceNorm2d + LeakyReLU(0.2)
Final: Conv2d(ndf*8, 1)  — no sigmoid (LSGAN)
```

**Why PatchGAN:** Focuses on local texture/style rather than global structure — better for texture transfer tasks like modality translation.

### Registration Network R (VoxelMorph-lite, 2D)
```
Input: (B, 2, 384, 384)  — concat(fake_B, real_B)
Output: (B, 2, 384, 384)  — displacement field (Δx, Δy) in pixels

U-Net architecture:
  Encoder: 3 conv blocks with AvgPool2d(2)
  Decoder: bilinear upsample + skip connections
  Flow head: Conv2d(nf, 2, 3) — init weights near-zero (small deformations)

nf = 16  (intentionally small — less capacity = less aggressive warping)

Warping: differentiable bilinear grid_sample
  grid = meshgrid(-1,1) + normalized_flow
  output = F.grid_sample(img, grid)
```

**Key design:** Near-zero weight init on flow head ensures small deformations from the start.

---

## Loss Functions

### Generator Loss
```
L_G = L_GAN_AB + L_GAN_BA          # fool discriminators
    + λ_cycle × (L_cycle_A + L_cycle_B)   # cycle consistency
    + λ_cycle × 0.5 × (L_idt_A + L_idt_B) # identity
    + λ_reg_sim × L_reg_sim         # warped fake_B ≈ real_B
    + λ_smooth × L_smooth           # smooth deformation field
    + λ_mag × L_mag                 # small deformation magnitude
```

### Minimum Deformation Losses (KEY CONTRIBUTION)
```python
# Smoothness: penalise spatial gradients of deformation field
L_smooth = mean(dx² + dy²)   where dx,dy = spatial gradients of flow

# Magnitude: directly penalise displacement size
L_mag = mean(flow²)
```

### Discriminator Loss
```
LSGAN (MSE): L_real = MSE(D(real), 1)
             L_fake = MSE(D(fake), 0)
             L_D = (L_real + L_fake) × 0.5
```

**Why LSGAN over BCE:** More stable training gradients, avoids vanishing gradient problem.

### Loss Weights (Hyperparameters)
```
λ_cycle      = 10.0   # cycle consistency strength
λ_reg_sim    = 5.0    # registration similarity
λ_reg_smooth = 10.0   # smoothness (higher = smoother field)
λ_reg_mag    = 5.0    # magnitude (higher = smaller deformation)
```

---

## Training Details

| Parameter | Value | Reason |
|---|---|---|
| Optimizer | Adam β=(0.5, 0.999) | Standard for GANs |
| LR (G, D) | 2e-4 | Standard CycleGAN LR |
| LR (R) | 1e-4 | Lower — R should learn slowly |
| Batch size | 8 | A100 40GB |
| Epochs | 200 (ran ~4) | Time-limited by HPC |
| LR schedule | Constant first half, linear decay second half | Standard GAN schedule |
| Image pool | 50 images | Discriminator stability |
| Gradient clip | 5.0 (generators only) | Prevent exploding gradients |
| Input range | [-1, 1] | Tanh output |
| Save every | 500 steps | Mid-epoch checkpointing |

### Training Step Order (CRITICAL BUG FIX)
```
1. Forward pass → compute all outputs
2. Update G (freeze D) → backward G loss
3. Update R (after G) → backward R loss
4. Update D (unfreeze) → backward D loss
```
**Why this order matters:** R.step() mutates R weights. If R runs before G backward, the computation graph version mismatches → inplace operation error. G must backward before R.step().

### Hardware
- **NVIDIA A100 40GB** on IU BigRed200 HPC
- CUDA 12.2, PyTorch cu121
- Training time: ~15 hours for 4 epochs (11,040 DESS + 2,271 PD slices)

---

## Preprocessing Pipeline

### Problem
| Issue | DESS | PD |
|---|---|---|
| Orientation | P,S,R sagittal | P,S,R sagittal |
| Shape | (512,512,160) | (384,384,36) |
| In-plane spacing | 0.31mm (isotropic) | 0.39mm (isotropic) |
| Through-plane | 0.80mm | 3.60mm |
| Non-orthonormal | Some files | Some files (22/69) |

### Solution
```
1. Reorient to RAS+ canonical orientation
   - SimpleITK.DICOMOrient(img, "RAS")
   - Falls back to nibabel.as_closest_canonical() for non-orthonormal files
   - Result: (R, A, S) array — R=through-plane for sagittal

2. Resample in-plane to isotropic
   - target_ip = min(sp_A, sp_S)
   - scipy.ndimage.zoom with order=3 (cubic)
   - Only A,S dims — through-plane R kept as-is

3. Resize to 384×384
   - scipy.ndimage.zoom (1.0, 384/n_A, 384/n_S)
   - Must be divisible by 4 (U-Net in R)

4. Intensity normalisation
   - Percentile clip (1st, 99th)
   - Scale to [0,1]
   - Then to [-1,1] for GAN training

5. Slice extraction along axis 0 (R = through-plane)
   - Skip slices with mean < 0.02 (background)
   - Save as (384,384) .npy files
```

### Mask Preprocessing (different from images)
```
Same pipeline BUT:
- order=0 (nearest neighbour) instead of order=3
  → preserves integer labels (0,1,2,3,5,6)
- No intensity normalisation
- Same affine as translated PD for alignment
- nibabel fallback for 22/69 non-orthonormal masks
```

---

## Evaluation Metrics

### Distribution-Level (unpaired)
| Metric | Value | Baseline (DESS) | Interpretation |
|---|---|---|---|
| FID (fake vs real PD) | 164.4 | 260.4 | 37% improvement ✅ |
| KID | 0.020 ± 0 | - | Low = distributions similar ✅ |

**FID limitation:** Using 64×64 pixel features (no InceptionV3 — no internet on HPC). Relative comparison is valid, absolute value may differ from standard FID.

### Structural Preservation
| Metric | Value | Interpretation |
|---|---|---|
| SSIM (DESS vs Fake PD) | 0.357 | Expected for cross-modality ✅ |

**Why low SSIM is correct:** DESS fluid = dark, PD fluid = bright. High SSIM would mean no translation happened. 0.35 means significant contrast change = translation working.

### Deformation (Core RegGAN Contribution)
| Metric | Value | Interpretation |
|---|---|---|
| Jacobian det mean | 1.000056 | Near-perfectly rigid ✅ |
| Jacobian det min | 0.74 | Boundary artifact, not real ✅ |
| Jacobian folding % | 0.0% | Zero topology violations ✅ |
| Global deformation | 0.029 px | Sub-pixel ✅ |
| Meniscus deformation | 0.055 px | Sub-pixel at key anatomy ✅ |

**Jacobian determinant explained:**
- det(J) = 1.0 → perfectly rigid, no deformation
- det(J) > 1 → local expansion
- det(J) < 1 → local compression  
- det(J) < 0 → folding (topology violation, anatomy destroyed)
- Mean of 1.000056 ≈ 1.0 → minimum deformation constraint working

---

## Key Bugs Fixed (Interview Gold)

### 1. Inplace Operation Error
**Problem:** `R.step()` mutated R weights mid-computation graph → version mismatch in PyTorch autograd
**Fix:** Reorder training steps — G backward before R.step()

### 2. Warp Function Expand Bug
**Problem:** `.expand()` on grid tensor caused inplace modification → backward error
**Fix:** Use broadcast (`grid + norm_flow`) instead of `.expand()`

### 3. PD Preprocessing Shrink Bug
**Problem:** PD has 36 through-plane slices but 384 in-plane — wrong axis caused (384×36) slices force-resized to 384×384 = 9× stretch
**Fix:** Resample in-plane to isotropic first, then resize

### 4. Affine Mismatch in Saved NIfTIs
**Problem:** Saved translated PD with original DESS affine → wrong voxel sizes → ITK-SNAP renders distorted
**Fix:** Compute effective spacing after preprocessing, build diagonal affine

### 5. Non-orthonormal Direction Cosines
**Problem:** 22/69 segmentation masks had non-orthonormal affines → SimpleITK crash
**Fix:** Try SimpleITK first, fallback to nibabel.as_closest_canonical()

### 6. SSIM Shape Mismatch
**Problem:** Fake PD NIfTIs loaded with wrong transpose → (384,160) instead of (384,384)
**Fix:** Only transpose when last dim is smallest (slice axis heuristic)

---

## Visualizations Generated

| File | What it shows |
|---|---|
| `sample_grid.png` | DESS / Fake PD / Real PD at meniscus slices with boundary contours |
| `intensity_histogram.png` | Distribution shift + CDF comparison |
| `jacobian_det.png` | Deformation topology heatmaps |
| `difference_map.png` | Pixel-wise abs diff (hot colormap) with meniscus boundary |
| `knee_boundary_overlay.png` | All tissue boundaries on DESS vs Fake PD |
| `meniscus_overlays/` | Per-scan lateral+medial with zoom, filled overlay, boundary |

---

## What the Visualizations Show

**Meniscus overlay:** Boundary contours (red=lateral label 5, green=medial label 6) sit on exact same anatomical location in DESS and Fake PD → alignment confirmed.

**Grey area inside meniscus boundary:** Correct — DESS meniscus is dark (fluid suppressed), PD meniscus is intermediate grey. Generator learned correct tissue appearance.

**Difference map:** Bright areas = fluid/cartilage (large contrast change = correct). Dark areas = bone/background (unchanged = correct). Meniscus boundary shows moderate diff (contrast changed, shape preserved).

**Jacobian:** Overwhelmingly white (≈1.0), faint orange/blue only at tissue boundaries = expected registration noise, not real deformation.

---

## Future Work (On Resume)

Extending to **diffusion models (DDPM) + CycleGAN** to generate synthetic healthy knee MRI — creating a "digital twin" of the knee before injury for patient-specific treatment planning and longitudinal modeling.

---

## Interview Q&A Prep

**Q: Why not just use CycleGAN?**
A: CycleGAN showed 40× higher deformation (>2px) — unacceptable for medical imaging. Anatomy must be preserved for downstream segmentation to work. RegGAN adds a registration network as a constraint.

**Q: What is the registration network doing?**
A: Takes (fake_B, real_B) as input, predicts a deformation field. The smoothness and magnitude losses on this field penalise large deformations, forcing the generator to achieve translation through contrast change rather than spatial warping.

**Q: Why is SSIM 0.35 — isn't that bad?**
A: No — it's expected for cross-modality translation. DESS and PD have fundamentally different tissue contrast. High SSIM would mean the generator did nothing. 0.35 means contrast changed significantly (good) while structure is preserved (confirmed by Jacobian).

**Q: How did you handle the unpaired nature of the data?**
A: CycleGAN-style — separate DESS and PD datasets, no paired correspondences needed. G_AB translates DESS→PD, G_BA translates PD→DESS, cycle consistency enforces that G_BA(G_AB(x)) ≈ x.

**Q: What is FID and why use it?**
A: Fréchet Inception Distance — measures distance between feature distributions of generated vs real images. Lower = distributions more similar. Used because we have no paired ground truth — can only measure distribution-level similarity.

**Q: What does Jacobian determinant tell you?**
A: It measures local volume change of the deformation field. det=1 means no deformation, det<0 means topology violation (folding — anatomy destroyed). Our mean of 1.000056 with 0% folding confirms minimum deformation is working.

**Q: What was the hardest bug?**
A: The inplace operation error in PyTorch autograd — R.step() was mutating weights after the computation graph was built but before G.backward() ran, causing a version mismatch. Fixed by reordering training steps: G update → R update → D update.

**Q: Why InstanceNorm instead of BatchNorm?**
A: BatchNorm normalises across the batch — problematic for unpaired training with small batches and high image variability. InstanceNorm normalises per-image per-channel, more stable for image translation tasks.

**Q: Why PatchGAN discriminator?**
A: PatchGAN classifies overlapping local patches rather than the whole image. Better for texture/style transfer — encourages sharp local texture matching PD contrast rather than global image statistics.

---

## File Structure
```
regGAN/
  preprocess.py        # NIfTI loading, resampling, slice extraction
  preprocess_masks.py  # Same pipeline for segmentation masks (order=0)
  dataset.py           # PyTorch Dataset — unpaired slice loading
  models.py            # Generator, Discriminator, RegistrationNet, losses
  train.py             # Training loop with TensorBoard logging
  infer.py             # Translate new DESS volumes → PD-like NIfTIs
  evaluate.py          # FID, KID, SSIM, Jacobian, meniscus deformation
  bigred_submit.sh     # SLURM training job
  bigred_infer.sh      # SLURM inference job
  bigred_evaluate.sh   # SLURM evaluation job
  demo_evaluation.ipynb# Step-by-step notebook for prof demo
```
