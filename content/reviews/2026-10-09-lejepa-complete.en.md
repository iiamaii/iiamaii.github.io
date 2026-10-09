---
title: "LeJEPA in full: from target distribution to theory, algorithm and experiments"
description: "A connected review of the 50-page paper: problem, related work, target distribution, SIGReg, training, experiments and limitations."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-09T23:54:43+09:00"
topics: ["self-supervised learning", "representation learning", "joint embedding predictive architectures", "distribution matching", "computer vision"]
visibility: "public"
lang: "en"
translationKey: "lejepa-complete"
paperTitle: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
paperPublishedDate: "2025-11-11"
authors: ["Randall Balestriero", "Yann LeCun"]
year: "2025"
paperUrl: "https://arxiv.org/abs/2511.08544v3"
thumbnail: "/assets/reviews/lejepa/lejepa-core.svg"
thumbnailAlt: "LeJEPA combines within-image alignment to the global-view mean with across-image Gaussian matching through random projections in each view."
---

LeJEPA connects two questions in self-supervised representation learning: <strong>what embedding distribution is useful before downstream tasks are known, and how can that distribution be matched efficiently in large models?</strong> The authors motivate an isotropic Gaussian through statistical estimation, then combine a projected distribution loss, SIGReg, with agreement between image views. The connection between theory, algorithm and varied image experiments is its main strength. This article follows the complete argument of the 50-page paper while preserving proof assumptions and experimental conditions. [§§1–6](https://arxiv.org/pdf/2511.08544v3#page=1)

This is the complete version. The [introductory article](/en/reviews/lejepa-overview/) explains the two constraints intuitively; the [technical review](/en/reviews/lejepa-technical/) develops proofs, costs and equation checks.

<strong>The hero diagram's distinction.</strong> All views pass through the same encoder and projector, but the two losses collect different samples. The left branch groups views of one image; the right groups different images within each view. Agreement preserves shared content, and distribution matching discourages point or directional collapse. The Gaussian constraint acts after the projector. [Enlarge](/assets/reviews/lejepa/lejepa-core.svg)

## A map of the argument

| Paper section | Question | Main connection |
| --- | --- | --- |
| §§1–2, pp. 1–4 | Is bringing related views together sufficient? | Agreement needs non-degeneracy |
| §3, pp. 5–6 | Which distribution helps subsequent prediction? | Linear isotropy and nonlinear Fisher information |
| §4, pp. 6–11 | How can high-dimensional Gaussianity be encouraged? | Random 1D projections, characteristic functions, SIGReg |
| §5, pp. 11–12 | What does the encoder actually optimize? | Global-view-centered agreement plus per-view distribution matching |
| §6, pp. 13–18 | Does it work across architectures, scales and domains? | Sensitivity, backbone diversity, transfer, Galaxy10 |
| Appendices A–F, pp. 24–46; Figures 16–21, pp. 47–50 | What assumptions and additional results matter? | Local-estimation conditions, numerical approximations, supplementary evidence |

The paper expands LeJEPA as <strong>Latent-Euclidean JEPA</strong>; the repository README also uses Lean Joint-Embedding Predictive Architecture. Here the paper's formulation is the reference. JEPA learns relationships in representation space instead of reconstructing pixels directly. This implementation primarily studies symmetric image multi-crop agreement. It does not evaluate action-conditioned future prediction or control. [§2.1, p. 4](https://arxiv.org/pdf/2511.08544v3#page=4), [official README](https://github.com/galilai-group/lejepa)

## The problem: invariance alone can erase information

Large and small crops from one photograph should retain common content. Minimizing their feature distance encourages this. Yet a constant encoder also makes every distance zero. We need both agreement within an image and an informative distribution across images.

- <strong>Within an image:</strong> do features preserve shared information when the view changes?
- <strong>Across images:</strong> does the distribution retain directions and variation useful for distinguishing inputs?

Previous methods use negatives, teachers, stop-gradient, centering, variance or covariance constraints to construct this balance. LeJEPA specifies a distributional target and asks why it should help subsequent predictors. Its core combines symmetric agreement with distribution regularization; a predictor and EMA teacher are not required. This does not establish that a predictor is unnecessary in every asymmetric prediction task. [§§2, 5](https://arxiv.org/pdf/2511.08544v3#page=3)

## Related work: locating the difference

Five primary sources were read for this comparison. Coverage is limited to the method/definition sections listed below, rather than full reviews of every source.

| Work | Collapse-prevention or distribution mechanism | Connection and distinction |
| --- | --- | --- |
| [VICReg](https://arxiv.org/pdf/2105.04906v3#page=3), §§3.1–3.3, pp. 3–4 | View agreement, a lower bound on coordinate standard deviation, suppression of off-diagonal covariance | Already symmetric without stop-gradient, a teacher or negatives. LeJEPA targets projected distributions beyond low-order statistics. VICReg's original penalty is not identical to exact zero mean and identity covariance. |
| [DINO](https://arxiv.org/pdf/2104.14294v2#page=2), §3, pp. 2–3 | EMA teacher, centering/sharpening, teacher–student probability cross-entropy | LeJEPA replaces teacher-output rules with an explicit Gaussian target. Original DINO also studies convolutional networks; architectural variety was not first introduced here. |
| [I-JEPA](https://arxiv.org/pdf/2301.08243v3#page=3), §3, pp. 3–4 | Predict masked target-block features from a context with a predictor and EMA target encoder | Asymmetric block prediction and symmetric crop agreement are different tasks. Table 2 must be read together with model and training conditions. |
| [Gaussian Embeddings](https://arxiv.org/pdf/2510.05949v1#page=1), pp. 1, 3–4 | Gaussian density through representation transformations and Jacobian volume | Shares interest in Gaussian embeddings, but not the same downstream bias/variance argument or sliced characteristic-function objective. |
| [Gretton et al., MMD](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf), PDF pp. 1, 3–4 | Compare kernel mean embeddings of distributions | Background for Gaussian-weighted CF/kernel discrepancies. Standard pairwise estimators are quadratic, but linear-time MMD estimators already existed. LeJEPA is not the first linear-time distribution comparison. |

The contribution is best located in the connection <strong>downstream estimation theory → target distribution → tractable loss → simple training structure</strong>, rather than claiming every ingredient is new.

## Why isotropy, and why a Gaussian?

Consider feature matrix `X` and downstream labels `y=Xβ+ε`. The coefficient `β` specifies a task; `ε` is noise. Small feature variation along a task-relevant direction makes estimation difficult.

The paper examines ridge shrinkage bias and OLS coefficient variance. Under fixed-design, full-rank and homoscedastic-noise conditions, the variance trace is `σ²Σ(1/λk)`, where `λk` are eigenvalues of `XᵀX`. For fixed total energy, equal eigenvalues minimize this quantity. As a review-authored example, `(1,1)` gives reciprocal sum 2, whereas `(0.2,1.8)` gives approximately 5.56 despite the same trace. [§3.1; Appendix B.1–B.2, pp. 26–27](https://arxiv.org/pdf/2511.08544v3#page=26)

This establishes a reason for <strong>isotropic covariance</strong>, not Gaussian uniqueness. The authors next study neighborhood averaging and Nadaraya–Watson kernel regression, which predict a query from nearby labeled features.

Their local bias contains both target-function variation and density variation. The density score is `∇log p`; its squared expectation `J(p)=E‖∇log p‖²` is Fisher information. With assumptions including an isotropic task-gradient prior, a bias component relates to `J(p)`. The kernel analysis controls a worst-case bias upper bound containing it. [§3.2; Appendix A, B.3–B.7](https://arxiv.org/pdf/2511.08544v3#page=24)

For regular densities with fixed covariance, a Gaussian minimizes Fisher information. Fixing the covariance trace further selects isotropy. These arguments use local approximations, smoothness, task-prior assumptions and relations between query and training distributions. They do not establish unconditional minimization of the exact total risk of every nonlinear predictor. The [technical review](/en/reviews/lejepa-technical/) details the conditions.

## What variance and covariance can miss

![LeJEPA Figure 5: an X-shaped non-Gaussian joint density can have Gaussian-like coordinate marginals but different directional projections.](/assets/reviews/lejepa/paper-figure-5.webp)

[Enlarge figure](/assets/reviews/lejepa/paper-figure-5.webp)

<strong>Figure 5 explained.</strong> The coordinate histograms on the left have similar bell shapes. The second panel's joint density is X-shaped rather than multivariate Gaussian. The third shows densities projected along the colored directions. The fourth compares responses of regularizers/test statistics across those directions. These responses are not all probabilities on a common scale, nor downstream accuracy. The point is that <strong>apparently satisfactory marginal statistics need not identify the joint distribution</strong>. Attribution: Balestriero and LeCun, [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Complete panels cropped and encoded as lossless WebP.

Matching finitely many moments also leaves ambiguity: distinct distributions can share a finite set of moments. Increasing moment order can make tails dominate values and gradients. LeJEPA instead compares characteristic functions. [§4.2, Theorem 3, pp. 7–9](https://arxiv.org/pdf/2511.08544v3#page=7)

## SIGReg: matching distributional fingerprints

Project `z` onto a unit direction `a`, obtaining `s=aᵀz`. The empirical characteristic function is a sample average at frequency `t`:

```text
φ̂a(t) = (1/N) Σn exp(i t aᵀzn)
Gaussian target: φG(t) = exp(−t²/2)
```

The imaginary unit `i` can be implemented through cosine and sine averages. The Gaussian target is analytic, so target samples are unnecessary. SIGReg integrates the weighted squared difference across frequencies and averages over directions. The paper's Epps–Pulley statistic and Algorithm 1 use a scale multiplied by batch size `N`. [§4.3, Definition 2; Algorithm 1, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10)

![LeJEPA Figure 2: projected embedding densities compared with Gaussian targets.](/assets/reviews/lejepa/paper-figure-2.webp)

[Enlarge figure](/assets/reviews/lejepa/paper-figure-2.webp)

<strong>Figure 2 explained.</strong> The input distribution is mapped into embedding space, then projected along colored directions. On the right, the horizontal axis is the projection coordinate, black curves are targets and shading illustrates discrepancies. These are explanatory densities, not ImageNet measurements. Attribution: Balestriero and LeCun, [Figure 2, p. 3](https://arxiv.org/pdf/2511.08544v3#page=3), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Complete panels cropped and encoded as lossless WebP.

Cramér–Wold links agreement along all directions with agreement of the original vector law. Training uses finite directions and frequency grids, so a small SIGReg loss is not certification of exact multivariate Gaussianity. Resampling directions discourages fitting only a fixed set of projections. [§4.1, §4.4; Appendix B.8–B.10](https://arxiv.org/pdf/2511.08544v3#page=6)

Dense projection and CF evaluation cost approximately `O(NKM+NMT)` for `K` feature dimensions, `M` directions and `T` frequencies. With `K,M,T` fixed, this is linear in sample count and avoids all sample pairs. Increasing directions with dimension, or including encoder computation, changes the interpretation of “linear.” [§4.3, p. 9; Algorithm 1](https://arxiv.org/pdf/2511.08544v3#page=9)

## Integrating the two losses

Let `zn,v` denote the projected embedding of image `n`, view `v`. There are `Vg` global views and `V` total views. Algorithm 2 centers agreement on the mean of each image's global views:

```text
μn = (1/Vg) Σg zn,g
Lpred = (1/NVK) Σn,v ||zn,v − μn||²
Lsigreg = (1/V) Σv SIGReg({zn,v : n=1…N})
L = (1−λ)Lpred + λLsigreg
```

Here `K` is embedding dimension. The formula averages feature coordinates as Algorithm 2's `.square().mean()` does. Gradients also flow through `μn`; it is not a frozen teacher target. Agreement groups views at fixed `n`, while SIGReg groups images at fixed `v`. A projector remains, and evaluated backbone features differ from regularized projected embeddings. Algorithm 2 sets global and all views equal for its non-ViT configuration, making the center an all-used-view mean in that case. [§5, Algorithm 2, pp. 11–12](https://arxiv.org/pdf/2511.08544v3#page=11)

Equations 5–7 describe converting average pairwise distances into center distances as equivalent. Direct expansion leaves an additional <strong>variance of global views around their mean</strong>. This review follows the center objective explicitly specified by Algorithm 2. The [technical review](/en/reviews/lejepa-technical/) shows the missing term and a numerical example. This algebraic discrepancy alone does not establish that the experimental results are invalid. [§5, p. 12; Appendix B.6, p. 34](https://arxiv.org/pdf/2511.08544v3#page=34)

## Simple objectives still have training choices

Section 6.1 specifies `λ=0.05`, two global and eight local views, batches of at least 128, 1,024 directions and 17 integration points on `[-5,5]`. Experiment Details 1 on p. 14 instead uses eight total views: two global and six local. Settings must be attached to each experiment. [§6.1; Experiment Details 1, pp. 13–14](https://arxiv.org/pdf/2511.08544v3#page=13)

The experiments use AdamW, learning-rate warmup and cosine scheduling. Some architecture experiments cross-validate learning rates `5×10⁻³/5×10⁻⁴` and weight decays `0.1/0.01/10⁻⁵`. Optional SWA results appear in Table 4. “Without the heuristics” therefore describes simplifying the core anti-collapse structure, not eliminating every schedule or optimizer choice. [p. 14; Table 4, p. 42](https://arxiv.org/pdf/2511.08544v3#page=14)

The [official repository snapshot](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052) inspected here has a MINIMAL example with 17 points on `[0,3]`, doubled positive-side weights, 256 directions and an all-view mean for invariance. This is not numerically identical to the paper's grid or global-view center. The example was read, not executed. Reproduction should pin both version and configuration.

## Experiment 1: sensitivity and backbone diversity

Table 1 studies ImageNet-1K, ViT-L/14, 100 epochs and frozen linear top-1 accuracy (%). Selected rows from separate ablations are shown below. [Table 1, p. 13](https://arxiv.org/pdf/2511.08544v3#page=13)

| Ablation | Values | Top-1 accuracy |
| --- | --- | --- |
| Batch size | 128 / 256 / 512 / 1024 | 72.20 / 74.15 / 74.72 / 74.07 |
| Global-view count, four total views | 1 / 2 | 53.06 / 72.26 |

Smaller batches can work, but accuracy is not identical. One global view causes a substantial drop. Integration-range rows also differ by roughly two percentage points. This supports useful robustness over several settings rather than complete insensitivity.

A separate ImageNet-100/ResNet-50 experiment varies views and `λ` together in Figure 8/Table 7: two views/λ=0.01 gives 83.49%, four/λ=0.02 gives 84.68%, and eight/λ=0.05 gives 84.32%. The best measured mixture is not identical across view counts. The default is a useful starting point rather than a unique optimum for every setting. [Figure 8, p. 13; Table 7, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

Figure 9 evaluates approximately 50 models below 20M parameters from eight families on a ten-class ImageNet subset, with accuracy around 91.5–95%. The paper's aggregate 60+ architectures and Figure 1's 1.8B-model training curve do not mean every model was evaluated on full ImageNet and all transfer tasks. These results chiefly expand the evidence for architectural compatibility. [Figures 1, 9; §6.1, p. 14](https://arxiv.org/pdf/2511.08544v3#page=14)

## Experiment 2: can loss indicate feature quality?

![LeJEPA Figure 10: accuracy colors in the plane of SIGReg and prediction losses.](/assets/reviews/lejepa/paper-figure-10.webp)

[Enlarge figure](/assets/reviews/lejepa/paper-figure-10.webp)

<strong>Figure 10 explained.</strong> The panels show ResNet-50/Galaxy10, ResNet-50/ImageNet-10 and ViT-B/14/ImageNet-1K. Horizontal and vertical axes are SIGReg and prediction losses, both logarithmic. Color indicates accuracy: red is higher, blue lower. Better accuracy generally concentrates toward the lower-left region where both losses are small. A single loss component is insufficient, and the measured relationship is not a universal law. Attribution: Balestriero and LeCun, [Figure 10, p. 15](https://arxiv.org/pdf/2511.08544v3#page=15), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Complete panels cropped and encoded as lossless WebP.

Figure 11 reports Spearman magnitudes around 0.60–0.90, improving to roughly 0.93–0.99 after `λ⁰·⁴` scaling. Loss decreases while accuracy increases, yet the displayed values are positive; the signed convention is not clarified here. The defensible description is a strong inverse monotonic association under the measured conditions, rather than an unqualified positive correlation. [§6.2, Figure 11, p. 15](https://arxiv.org/pdf/2511.08544v3#page=15)

This is not proof of universally label-free model selection on new domains. The scaling exponent is evaluated against labeled observations. A prospective test would fix it on previous domains and select checkpoints before seeing labels in a new domain.

## Experiment 3: transfer averages and task-specific differences

Table 2 evaluates DTD, Aircraft, Cars, CIFAR-10/100, Flowers102, Food101 and Pets. These averages (%) belong to its ImageNet-1K-pretrained block. [Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16)

| Pretrained model | Parameters | Epochs | 1 label/class | 10 labels/class | All labels |
| --- | --- | --- | --- | --- | --- |
| I-JEPA | 632M | 300 | 30.20 | 60.51 | 78.50 |
| I-JEPA + STOP | 632M | 300 | 32.05 | 62.92 | 80.70 |
| LeJEPA ViT-L | 304M | 100 | 29.55 | 60.95 | 79.48 |

With a smaller model and fewer epochs, LeJEPA is competitive with plain I-JEPA and exceeds some averages. Its one-shot average is lower, and +STOP is stronger at all three label budgets. It does not beat every I-JEPA variant.

At ten shots, LeJEPA versus plain I-JEPA gives 64.72 versus 57.68% on DTD, 92.53 versus 88.24% on Flowers102 and 50.90 versus 43.97% on Food101, but 77.00 versus 83.23% on Pets. With all labels, CIFAR-10/100 are also lower: 96.50/83.71 versus 97.54/86.42%. A macro average alone hides these strengths and weaknesses.

100 versus 300 epochs compares passes over images. Different model sizes, crops, resolutions and operations prevent reading this as exactly one-third wall time or FLOPs. The ImageNet-22K 900-epoch block is a separate pretraining condition.

## Experiment 4: learning directly in a small specialized domain

![LeJEPA Figure 12: Galaxy10 accuracy across downstream label budgets with full finetuning and frozen encoders.](/assets/reviews/lejepa/paper-figure-12.webp)

[Enlarge figure](/assets/reviews/lejepa/paper-figure-12.webp)

<strong>Figure 12 explained.</strong> The horizontal axis is downstream labels per class; the vertical axis is accuracy (%). Left: full finetuning updates the backbone. Right: frozen evaluation keeps it fixed. Orange/brown curves are LeJEPA models pretrained on Galaxy10 for 400 epochs. Blue curves are generally pretrained DINOv2/v3 models. This shows the potential of small in-domain learning, not a loss-only comparison with architecture, data and compute matched. Three seeds are reported; the exact error-bar statistic is unspecified. Attribution: Balestriero and LeCun, [Figure 12, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Complete panels cropped and encoded as lossless WebP.

Galaxy10 has ten galaxy classes and 11,008 training images according to Table 5. Full-label frozen accuracy in Table 3 is 78.17% for LeJEPA ResNet-34, 76.52% for ConvNeXt Nano, 67.62% for DINOv2 Small and 71.38% for DINOv3 Small. Full finetuning gives 83.28, 82.72, 78.34 and 81.60%, respectively. Separating fixed-feature quality from adaptation of the full model clarifies the result. [Table 3, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42)

At one-shot frozen evaluation, LeJEPA LeViT reaches 25.85% versus DINOv3 Small's 30.17%, so not all LeJEPA architectures lead. Table 5 also gives Flowers102 accuracy of 82.19% for LeJEPA ResNeXt versus I-JEPA's 85.76%. Galaxy10 values differ between Tables 3 and 5 and are not pooled as one protocol here. Table 5's I-JEPA row name conflicts with its pretraining-data cell; that detail remains unverified. [Tables 3, 5, pp. 42–43](https://arxiv.org/pdf/2511.08544v3#page=43)

## Additional evidence: visual structure and cost

Section 6.4 reports ImageNet-1K online evaluation of 77.1% for ViT-L and 78.5% for ConvNeXt-H. These differ from Table 1's post-training frozen evaluation and Table 2's transfer macro averages. The abstract's ViT-H 79% is not treated as the same detailed configuration. [§6.4, p. 17](https://arxiv.org/pdf/2511.08544v3#page=17)

Figure 13's thresholded attention regions and Figure 14's feature-PCA colors are qualitative observations of visual structure. They are not segmentation IoU benchmarks. Separately fitted per-image PCA colors do not form universal semantic coordinates across images. [§6.5, pp. 17–18](https://arxiv.org/pdf/2511.08544v3#page=17)

Supplementary Figures 16–21 add distribution-test comparisons, regression/regularization illustrations, loss–accuracy relationships and quadrature results. Appendix D discusses spherical directions; E discusses Shapiro–Wilk order statistics; F supplies multivariate normality statistics. These are not all visualizations of learned image features. Figure 17's caption says anisotropy has lower variance, conflicting with the main isotropy discussion. Its conditions/labeling remain unresolved, so it is not used here as independent performance evidence. [pp. 43–50](https://arxiv.org/pdf/2511.08544v3#page=43)

Table 6 measures SIGReg on a V100 SXM2 16GB over ten runs: `N=512,M=512,T=16` gives `0.465236±0.011642 ms`, and `N=8192` gives `6.188304±0.007226 ms`. Section 4.4 describes forward–backward timing. It is loss computation, not full encoder training. Embedding dimension `K` and separate forward/backward timings are absent from the table, limiting precise system comparisons. [§4.4; Table 6, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

## Connecting proofs and experiments without merging their guarantees

The theory motivates the Gaussian target. SIGReg analysis explains properties of computing that target, including characteristic-function existence, sample-gradient control and finite-sample bias. Experiments show that finite directions and grids can train useful representations with real backbones.

These are different guarantees. Smooth-density population statements differ from finite-batch training. A bounded derivative with respect to a sample is not a bound on all encoder gradients or global SGD convergence. Distribution matching does not imply highest accuracy on every downstream task. [§§3–4, Theorems 4–6; Appendix B](https://arxiv.org/pdf/2511.08544v3#page=24)

My assessment is that LeJEPA is a compelling simple image-SSL baseline to reproduce, particularly for specialized domains where general pretrained representations transfer poorly. The next questions concern semantics preserved by augmentation, comparisons at matched compute, and whether unseen projections/frequencies confirm distribution matching beyond the training loss.

## Designing the next checks

First, compare against VICReg or I-JEPA-family alternatives with the same backbone, crops, resolution, optimizer and total compute. Record label budgets, seed variation, actual time and memory alongside top-1.

Second, evaluate distribution discrepancy using held-out directions and a finer frequency grid. This tests whether low training SIGReg generalizes to unobserved distributional measurements.

Third, fix loss scaling and checkpoint selection using previous domains, then evaluate selection before revealing labels in a new domain. This converts retrospective correlation into a prospective test. These are proposed follow-up experiments, not results obtained in this review.

## Sources, reading coverage and validation

- Paper: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, Randall Balestriero and Yann LeCun, [arXiv 2511.08544](https://arxiv.org/abs/2511.08544). First submission 2025-11-11; reviewed v3, revised 2025-11-14, 50 pages.
- First pass: pp. 1–3, 17–18 and the section structure to establish the problem and argument. Second pass: main text pp. 1–18, Appendices A–F pp. 24–46, supplementary Figures 16–21 pp. 47–50. The bibliography on pp. 18–23 was checked selectively.
- Related sources: VICReg v3 pp. 3–4, DINO v2 pp. 2–3, I-JEPA v3 pp. 3–4, Gaussian Embeddings v1 pp. 1, 3–4, Gretton et al. 2012 PDF pp. 1, 3–4 (printed pp. 723, 725–726). Coverage is the cited methods/definitions, not every source in full.
- Code: README and MINIMAL at [commit c293d291](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052), read but not executed.
- Hero: one original conceptual diagram based on §§4–5 and Algorithms 1–2, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), font weight 300 with emphasis 600. Paper-figure captions record attribution and CC BY-SA 4.0. Complete panels were cropped, rasterized and losslessly encoded without editing plotted data.
- <strong>Executed:</strong> review-authored center-distance identity, fixed 1D grid alias, OLS reciprocal-eigenvalue sum and ECF sampling-bias calculations. These are not model reproductions; details are in the [technical review](/en/reviews/lejepa-technical/).
- <strong>Not run:</strong> official code, GPU training, frozen/finetuning reproduction, repeated-seed training, DDP, paper runtime/memory remeasurement and full formal proof verification. Conflicting settings and the unspecified error-bar definition remain explicitly recorded above.
