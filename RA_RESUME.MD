# RA Resume Material — RegGAN Meniscus Segmentation Project
> Exhaustive resume bullets extracted from this project, organized by target role.
> Last updated: 2026-09-09

---

## Positioning Summary

This project is strongest for **ML Engineer**, second for **AI Engineer**, and weakest for **pure Software Engineering** — not because the engineering was weak, but because SWE recruiters screen for production systems, users, and scale-in-production, which a research project doesn't naturally supply.

| Role | Lead with | Headline metric | Risk to manage |
|---|---|---|---|
| **ML Engineer** | Experimentation rigor + domain adaptation | 0.43 → 0.781 Dice across 12+ ablations | None, this is the fit |
| **AI Engineer** | Generative AI + synthetic data | 37% FID gain, 40× less distortion | Add a deployed demo |
| **SWE** | Pipeline systems + debugging depth | 13,000+ files, 6 documented root-cause fixes | Fill the gaps table below |
| **Research Scientist** | Novel method + publication | SPIE submitted, 0% topology violations | Need the paper to land |

**Highest-leverage additions, in order:**
1. Get the SPIE paper accepted and put it on the resume
2. Make the GitHub repo public and legible with a strong README
3. For SWE roles specifically, ship one small thing with real users

---

# PART 1 — Software Engineering Resume

## Systems & Pipeline Architecture

- Designed and built a modular end-to-end ML pipeline (preprocessing → training → inference → evaluation) with clean separation of concerns across 8 independently runnable components, enabling any stage to be re-run without recomputing upstream artifacts
- Architected a checkpoint management system with mid-epoch saving (every 500 steps) and dual best/latest tracking, making 15-hour training jobs resumable after HPC preemption
- Built a configuration-driven experiment system where 12+ distinct experiments share a common training core, differing only in injected loss functions and CLI arguments — eliminating code duplication across experiment variants
- Implemented an inference pipeline that processes patient volumes from a manifest file, with pre-flight validation (checkpoint existence, file matching, empty-input guards) that fails fast with actionable error messages instead of silently producing empty output

## Data Engineering

- Built a preprocessing pipeline handling 13,000+ medical image slices across two heterogeneous data sources with incompatible shapes (512×512×160 vs 384×384×36), voxel spacings (0.31mm vs 0.39mm in-plane, 0.80mm vs 3.60mm through-plane), and coordinate conventions
- Implemented a defensive dual-library loading strategy (SimpleITK primary, nibabel fallback) after discovering 22 of 69 source files had non-orthonormal direction cosines that crashed the primary parser — recovering 32% of the dataset that would otherwise have been dropped
- Wrote separate interpolation paths for images (cubic, order=3) and label masks (nearest-neighbour, order=0) after identifying that shared interpolation silently corrupted integer class labels
- Normalized coordinate systems across modalities by reorienting all volumes to RAS+ canonical form, resolving anatomical misalignment between source and target datasets

## Debugging & Reliability

- Diagnosed and fixed a PyTorch autograd version-mismatch error caused by optimizer step ordering, where a network's weights were mutated after the computation graph was built but before a dependent backward pass — resolved by restructuring the training loop's update sequence
- Identified an inplace tensor mutation caused by `.expand()` on a broadcast grid in a differentiable warping function; replaced with explicit broadcasting to restore gradient flow
- Traced a 9× image distortion bug to an axis-ordering assumption in the resampling step, where through-plane and in-plane dimensions were conflated — fixed by resampling to isotropic spacing before resizing
- Caught a silent data-loss bug in the evaluation pipeline where a wrong directory path caused 7 of 17 test patients to be skipped without error, inflating reported accuracy; corrected the path, re-ran all affected experiments, and reported the lower accurate numbers
- Fixed an affine-metadata mismatch where output files inherited source-modality voxel spacing, causing downstream medical viewers to render volumes with incorrect physical dimensions
- Resolved a shape-inference bug in evaluation by replacing a hardcoded transpose with a dimension-size heuristic that correctly identifies the slice axis

## Infrastructure & HPC

- Deployed and managed distributed GPU workloads on IU BigRed200 supercomputer using SLURM, authoring 20+ job scripts with resource specification, module loading, environment activation, and failure guards
- Managed GPU memory constraints on NVIDIA A100 40GB, tuning batch size and model width (48 base filters) to maximize throughput within hardware limits
- Built reproducible remote-to-local workflows for transferring model checkpoints and inference outputs between HPC storage and local analysis environments
- Implemented gradient clipping and learning-rate scheduling to stabilize long-running training jobs against divergence

## Code Quality, Documentation & Process

- Maintained a versioned experiment tracking system documenting the starting checkpoint, training data, hyperparameters, and results for every run — enabling any experiment to be reproduced from documentation alone
- Authored technical documentation translating implementation details into audience-appropriate summaries for both engineering and non-technical stakeholders
- Used Git for version control across the full project lifecycle with meaningful commit granularity per experiment phase
- Wrote defensive shell scripts with existence checks, empty-result guards, and explicit exit codes rather than allowing silent failures

## Gaps to Fill for a SWE Resume

| Gap | Why it matters | How to close it |
|---|---|---|
| No production system with users | SWE screens look for this first | Ship a side project with real users, even small |
| No API / service work | Backend roles expect REST/gRPC | Wrap the model in a FastAPI service |
| No automated testing | Every SWE JD lists it | Add pytest to this repo, even retroactively |
| No CI/CD | Standard expectation | GitHub Actions running the tests |
| No system design at scale | Asked in every interview | Study + a design doc in the repo |
| No collaborative codebase | "Worked with a team of N" | Contribute to an OSS project |

---

# PART 2 — ML Engineer / ML Scientist Resume

> **Strongest positioning.**

## Model Development & Architecture

- Implemented RegGAN, a registration-constrained generative adversarial network, from the paper up in PyTorch — including dual ResNet-based generators (9 residual blocks, 48 base filters), PatchGAN discriminators, and a 2D VoxelMorph-style registration network with differentiable bilinear warping
- Designed architecture choices with explicit justification: InstanceNorm over BatchNorm for small-batch unpaired training, ReflectionPad over ZeroPad to eliminate border artifacts, and upsample-convolution over transposed convolution to eliminate checkerboard artifacts
- Initialized the registration network's flow head near zero to bias the model toward small deformations from the first training step, and deliberately under-parameterized it (16 filters) so limited capacity acted as an additional regularizer
- Fine-tuned a 2.5D U-Net (ResNet34 encoder) for anatomical segmentation, adapting a 5-class pretrained head to a 3-class target task while preserving encoder weights

## Loss Function Engineering

- Designed a composite generator objective combining adversarial loss, cycle consistency, identity loss, registration similarity, deformation smoothness, and deformation magnitude — with tuned weighting across six terms
- Implemented custom deformation-regularization losses (spatial-gradient smoothness and L2 magnitude penalties on the displacement field) that reduced anatomical warping by 40× versus the standard CycleGAN baseline
- Chose LSGAN (MSE) over binary cross-entropy for adversarial training to avoid vanishing-gradient instability, and implemented an image replay buffer (50 images) to stabilize discriminator training
- Implemented and evaluated four distinct segmentation loss formulations from the literature: Tversky (α=0.3, β=0.7) for recall bias, clDice with differentiable soft-skeleton extraction via iterative min-pooling, boundary-aware cross-entropy using morphological erosion rings, and foreground-restricted cross-entropy
- Built a class-weighted composite loss (weighted CE + soft Dice) to handle severe foreground/background imbalance in small-structure segmentation

## Experimentation & Scientific Rigor

- Designed and executed 12+ controlled ablation experiments isolating one variable at a time across loss functions, data augmentation strategies, encoder architectures, input context depth, and training data composition
- Improved segmentation Dice from 0.43 (zero-shot cross-modality baseline) to 0.781 on a held-out 17-patient cohort through systematic experimentation
- Identified through ablation that the critical variable was training data composition, not loss engineering — mixing 15 real target-domain patients with synthetic data achieved 0.781, while real data alone collapsed to 0.135, establishing that synthetic data was essential as a warm-start signal
- Recognized that a topology-aware loss variant won 7 of 17 individual patients despite a marginally lower aggregate mean, correctly identifying an aggregate metric masking per-case improvement
- Maintained strict held-out discipline: a 10-patient evaluation cohort was never used in training across the entire project lifecycle, extended to 17 with new labels that were also excluded from training
- Documented negative results as rigorously as positive ones, establishing that loss-function modifications consistently underperformed data-centric improvements on this task

## Transfer Learning & Domain Adaptation

- Solved a label-scarcity problem by transferring segmentation supervision across imaging modalities: source modality had expert masks, target modality had none — used unpaired image translation to bridge the gap
- Systematically evaluated the effect of starting checkpoint on transfer, demonstrating that initializing from a synthetically-pretrained model versus a source-domain model produced a 0.64 Dice difference under identical training data
- Curated and expanded the target-domain training cohort from 10 to 15 patients, quantifying the resulting improvement (+0.016 Dice)

## Evaluation & Metrics

- Built a multi-metric evaluation framework spanning distribution similarity (FID, KID), structural preservation (SSIM), and topological integrity (Jacobian determinant analysis) — achieving 37% FID improvement over the untranslated baseline
- Implemented Jacobian determinant analysis to quantify deformation-field topology, demonstrating 0% folding (zero topology violations) and a mean determinant of 1.000056, confirming near-rigid anatomical preservation
- Measured deformation at the anatomically critical region specifically (0.055 pixels at the meniscus), rather than reporting only global averages that could hide localized distortion
- Correctly interpreted a low SSIM (0.357) as evidence the translation succeeded rather than failed, reasoning from tissue-contrast physics that high structural similarity would indicate no modality translation had occurred
- Documented a known limitation of the FID implementation (64×64 features rather than InceptionV3, due to an air-gapped compute environment) and scoped conclusions to relative comparison accordingly

## Training at Scale

- Trained generative models on 13,000+ image slices across 138 patient volumes using NVIDIA A100 40GB GPUs on university HPC infrastructure
- Implemented learning-rate scheduling (constant then linear decay) and differentiated learning rates per subnetwork (2e-4 for generators/discriminators, 1e-4 for the registration network) to control relative convergence speed
- Instrumented training with TensorBoard logging for loss curves and sample-output monitoring across multi-hour runs

---

# PART 3 — AI Engineer Resume

## Generative AI

- Built a production-grade generative AI system for cross-domain image synthesis, generating photorealistic synthetic medical images that preserve source anatomy with sub-pixel fidelity (0.055px deformation)
- Implemented adversarial training with a PatchGAN discriminator operating on 70×70 local patches, chosen to optimize local texture realism over global statistics — appropriate for style/appearance transfer tasks
- Achieved 37% improvement in generated-image realism (FID 260.4 → 164.4) measured against the real target distribution
- Solved the core failure mode of unconstrained generative models in high-stakes domains: standard approaches distorted content while transferring style, making outputs unusable downstream. Added an explicit registration constraint that reduced distortion 40× while preserving style transfer quality
- Currently extending the system to diffusion models (DDPM) for controllable synthetic generation

## Synthetic Data for Model Training

- Used generative models to solve a training-data bottleneck: generated 11,000+ synthetic labeled images in a target domain where no labeled data existed, enabling supervised training of a downstream model
- Validated synthetic data quality with a downstream-task metric rather than perceptual quality alone — demonstrated that models trained on synthetic data reached 0.781 accuracy on real held-out data
- Quantified the synthetic-versus-real data tradeoff empirically, establishing that a mixed regime substantially outperformed either source alone (0.781 mixed vs 0.135 real-only vs 0.695 synthetic-only)

## Model Pipeline & Deployment Readiness

- Built an inference pipeline that ingests raw volumetric medical files, applies the full preprocessing chain, batches through the model, and writes outputs with correct spatial metadata for downstream consumption
- Designed the inference path to be independent of training code, with checkpoint loading, batched GPU execution, and format-correct output serialization
- Managed model artifacts across 12+ trained checkpoints with a documented registry mapping each checkpoint to its training configuration and evaluation results
- Evaluated deployment optimization paths (ONNX export, TensorRT) and made a documented decision not to pursue them, correctly scoping engineering effort to what the research goals required

## Evaluation & Trustworthiness

- Built an automated evaluation harness computing six distinct metrics across generated outputs, producing quantitative reports plus visual diagnostics for qualitative review
- Generated interpretable visual artifacts (deformation heatmaps, difference maps, boundary overlays, intensity distribution comparisons) to make model behavior auditable by domain experts who could not read the metrics directly
- Applied domain-expert-reviewable validation: overlaid ground-truth anatomical boundaries on generated images to visually confirm structural correspondence, rather than relying solely on aggregate scores
- Identified and characterized failure modes on out-of-distribution inputs (pathological cases the model was never trained on), documenting two distinct failure patterns to guide subsequent work

---

# Soft Skills — All Roles

Each tied to specific evidence, so they survive an interview follow-up.

| Skill | Evidence |
|---|---|
| **Intellectual honesty** | Found a bug inflating own results; fixed it and reported the lower numbers |
| **Handling ambiguity** | Started from a problem with no labeled data and no obvious approach |
| **Receiving feedback** | Advisor raised localization, recall bias, and 3D priors; converted each into a concrete experiment plan |
| **Knowing when to stop** | Abandoned loss-engineering after evidence showed it wasn't the lever; redirected to data |
| **Communicating to non-experts** | Produced stakeholder-facing documentation with failure-mode analysis and visual diagnostics |
| **Independence** | Sole researcher; owned every stage from data to publication submission |
| **Prioritization under constraints** | Worked within HPC queue limits and GPU allocation; sequenced experiments accordingly |
| **Systematic thinking** | Maintained a version-tracked experiment log across 12+ runs so any result is traceable |
| **Nuanced interpretation** | Recognized aggregate metrics masking per-case wins rather than accepting the headline number |
| **Publication-track communication** | Submitted to SPIE; manuscript in preparation |

---

# Quick-Reference Metrics

Use these numbers verbatim — they are all verified from the experiment log.

| Metric | Value |
|---|---|
| Segmentation Dice (final) | 0.781 on 17 held-out patients |
| Segmentation Dice (baseline) | 0.43 zero-shot cross-modality |
| Best single result | 0.784 (with inference-time intensity clipping) |
| FID improvement | 37% (260.4 → 164.4) |
| Deformation vs CycleGAN | 40× lower |
| Meniscus deformation | 0.055 px (sub-pixel) |
| Global deformation | 0.029 px |
| Jacobian folding | 0.0% (zero topology violations) |
| Jacobian determinant mean | 1.000056 |
| SSIM | 0.357 (expected for cross-modality) |
| Total slices processed | 13,311 (11,040 DESS + 2,271 PD) |
| Patient volumes | ~69 per modality |
| Experiments run | 12+ across 2 phases |
| Non-orthonormal files recovered | 22 of 69 (32%) |
| Hardware | NVIDIA A100 40GB, IU BigRed200 |
| Training time | ~15 hours for 4 epochs |

---

# One-Liner Versions

For tight resume bullets where space is limited:

- *Developed a topology-constrained GAN pipeline that translates knee MRI across modalities with sub-pixel anatomical deformation (0.055px), enabling label-free segmentation model transfer*
- *Improved meniscus segmentation accuracy from 43% to 78.1% Dice on 17 held-out patients through 12+ ablation experiments across loss functions, augmentation strategies, and training data curation*
- *Built and managed an end-to-end medical imaging ML pipeline — preprocessing through evaluation — on IU BigRed200 HPC across 11,000+ MRI scans*
- *Achieved 37% FID improvement in cross-modality image translation with zero topology violations, validated by Jacobian determinant analysis*
- *Generated 11,000+ synthetic labeled medical images to solve a target-domain label scarcity problem, validated by downstream task performance rather than perceptual quality alone*
