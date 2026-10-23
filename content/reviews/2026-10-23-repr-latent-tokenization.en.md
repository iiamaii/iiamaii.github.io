---
title: "[1/2] RePR: learning representations that make generation easier"
description: "Follow Sections 1–7 through predictive information differences, patch features, watermark tests, and combined training costs."
date: "2026-10-23"
publishedAt: "2026-10-23T15:12:16+09:00"
updatedAt: "2026-10-23T15:12:16+09:00"
topics: ["representation learning", "latent tokenization", "diffusion models", "generative modeling", "scaling laws"]
visibility: "public"
lang: "en"
translationKey: "repr-latent-tokenization"
paperTitle: "Generative Representation Learning for Scalable Latent Tokenization"
paperPublishedDate: "2026-10-14"
authors: ["Alan Baade", "Kyle Sargent", "Vincent Sitzmann", "Eric Ryan Chan", "Justin Johnson", "Li Fei-Fei", "Gordon Wetzstein"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.20533v1"
thumbnail: "/assets/reviews/repr/figure1.webp"
thumbnailAlt: "Original Figure 1: RePR matches conditional predictive information differences with increments of patch-representation inner products."
---

Does a representation that classifies images well also make them easy to generate? RePR (Relative Prediction Representations) learns from **how much an additional view improves prediction of a new patch**. Its question includes the cost of obtaining that representation: after counting encoder pretraining, is the combined system better than a model trained directly on pixels?

This article follows arXiv v1 in **§1→§2→§3→§4→§5→§6→§7** order. [\[2/2\] Appendix, implementation, and theory](/en/reviews/repr-latent-tokenization-technical/) follows A.1–A.7 through pseudocode, probability derivations, reconstruction conditions, and raw scaling results. Here a tokenizer is an encoder producing continuous patch features. The paper does not introduce a new discrete codebook or vector quantization method.

## 1. Introduction: what representations does generation need?

Self-supervised representation learning often brings augmented views of an image together. Discarding color, background, or layout can help classification, but generation must produce those variables as well. The authors identify **feature suppression**, a mismatch between the representations explicitly supervised and the patch features used for generation, and the cost of scaling encoders.

Feature suppression is distinct from dimensional collapse. Many embedding dimensions can be active while carrying only an identifying watermark. Non-degeneracy of the feature distribution does not establish preservation of every generative factor. This connects the distributional regularization discussed in the previous [LeJEPA review](/en/reviews/lejepa-complete/) to the information-preservation question asked here.

A second concern is accounting. Fast training on an existing DINOv2 encoder does not by itself make the encoder's pretraining efficient. The paper therefore separates encoder scaling from **encoder training plus generative-model training**. [§1, pp.1–2](https://arxiv.org/pdf/2610.20533v1#page=1)

## 2. Related Work: what RePR changes

The original discussion connects representation learning, PMI and representations, and pixel/latent generation. Five central comparisons help distinguish the later experiments.

| Approach | Basic design | Connection and difference |
| --- | --- | --- |
| [LeJEPA](https://arxiv.org/abs/2511.08544v3) | Jointly learn view predictability and a Gaussian representation distribution | The present watermark experiment tests what information remains when identification has an easy shortcut. It does not refute every LeJEPA theorem. |
| [DINOv2](https://arxiv.org/abs/2304.07193v2) | Teacher–student learning with image and patch objectives | An encoder used for generation and a scaling reference. The watermark experiment instead uses **DINOv3-B/16**. |
| [RAE](https://arxiv.org/abs/2510.11690v1) | Train a decoder on a frozen representation encoder, then generate in its latent space | RePR studies how to learn the representation and account for its cost. RAE decoding is an additional transfer setting. |
| [JiT](https://arxiv.org/abs/2511.13720v2) | Pixel diffusion Transformers predicting clean images | The main tokenizer-free baseline for combined-compute comparisons. |
| [Latent Forcing](https://arxiv.org/abs/2602.11401v1) | Jointly generate latents and pixels with separate time variables | Generates features first, then pixels conditioned on them, separating encoder diffusability from a lossy decoder bottleneck. |

DINOv2 already has an iBOT **patch-level objective**. The distinction should not be reduced to an absence of patch supervision in DINO. RePR directly supervises the geometry of the feature inner products against an information target. Also, LeJEPA already avoids explicit negative samples; the information-difference target matters beyond the absence of negatives. [DINOv2 §4, pp.5–6](https://arxiv.org/pdf/2304.07193v2#page=5)

The theoretical background includes density-ratio estimation in [CPC §2.3](https://arxiv.org/pdf/1807.03748v2#page=3) and diffusion bounds in [Variational Diffusion Models, E.1–E.3](https://arxiv.org/pdf/2107.00630v6#page=17). These were selected-section readings rather than full reviews of every cited work. [Original §2, pp.2–3](https://arxiv.org/pdf/2610.20533v1#page=2)

## 3. Background: PMI measures a relationship between observations

Pointwise mutual information (PMI) measures how much more often two particular observations co-occur than independence predicts. The paper's bold symbols $\mathbf{x},\mathbf{y}$ denote observed views; $R_\theta$ is the representation model.

$$
\operatorname{PMI}(\mathbf{x},\mathbf{y})
=\log\frac{p(\mathbf{y}\mid\mathbf{x})}{p(\mathbf{y})}
=\log\frac{p(\mathbf{x},\mathbf{y})}{p(\mathbf{x})p(\mathbf{y})}.
$$

This is original Eq. (1). As a review example, an event with probability 0.01 that becomes probability 0.1 after observing another view has PMI $\log 10$. Positive, zero, and negative values indicate more frequent, independence-level, and less frequent co-occurrence for that pair. PMI is not a probability or a necessarily positive similarity. Its expectation, mutual information, is a different quantity.

An optimal contrastive score is proportional to a conditional-to-marginal density ratio. With a log-inner-product score, original Eq. (2) gives

$$
\langle R_\theta(\mathbf{x}),R_\theta(\mathbf{y})\rangle
\approx\operatorname{PMI}(\mathbf{x},\mathbf{y})+a(\mathbf{x})
$$

The offset $a(\mathbf{x})$ cannot be distinguished by a softmax over candidates. This describes an ideal optimal score under the required capacity and negative-sampling assumptions, not exact PMI recovery by every finite embedding network.

RePR estimates PMI with a generative model and regresses the representation score against it. Original Eq. (3) uses $\log P_{G_\theta}(\mathbf{y}\mid\mathbf{x})-\log P_{G_\theta}(\mathbf{y})$. Whether those model probabilities equal the data probabilities is a separate approximation question. [§3, p.3](https://arxiv.org/pdf/2610.20533v1#page=3)

## 4. Method: transferring relative predictive advantage into features

The initial proposal minimizes squared error between an embedding inner product and estimated PMI. Original Eqs. (4–5) have the following core form:

$$
\mathcal{L}_{\mathrm{PMI}}
=\left(\langle R_\theta(\mathbf{x}),R_\theta(\mathbf{y})\rangle
-\operatorname{stopgrad}\!\left[
\log P_{G_\theta}(\mathbf{y}\mid\mathbf{x})-\log P_{G_\theta}(\mathbf{y})
\right]\right)^2.
$$

Stop-gradient prevents the regression term from reducing error by directly moving its generative target. The predictor receives a separate generative training loss. Full-view likelihood estimation remains expensive, and ordinary PMI can be dominated by fine-scale information. Sections 4.1–4.4 address these issues in order.

### 4.1 PMI Estimation from Generative Models

A discrete-probability model exposes log probabilities directly. For diffusion, the paper uses a variational lower bound (VLB). Original Eq. (6) decomposes negative log likelihood into prior, reconstruction, and diffusion terms. The noise path is

$$
\mathbf{z}_t=(1-t)\mathbf{x}+t\boldsymbol{\epsilon},
\qquad\boldsymbol{\epsilon}\sim\mathcal{N}(0,I).
$$

In this paper's equations, **$t=0$ means data and $t=1$ means pure noise**. Importing the time convention of the JiT or Latent Forcing papers without conversion reverses the interpretation of weighting.

Under an optimal denoiser and vanishing endpoint terms, Eqs. (7–8) express PMI by integrating the reduction in denoising error obtained from the extra view:

$$
\operatorname{PMI}(\mathbf{x},\mathbf{y})
=\mathbb{E}_{t,\boldsymbol{\epsilon}}\!\left[
\frac{1-t}{t^3}
\left(
\|\mathbf{y}-\mathbb{E}(\mathbf{y}\mid\mathbf{y}_t,t)\|_2^2
-\|\mathbf{y}-\mathbb{E}(\mathbf{y}\mid\mathbf{x},\mathbf{y}_t,t)\|_2^2
\right)\right].
$$

Both predictions use the same noisy target. If the context improves prediction, it produces positive supervision. The paper treats finite-precision pixels and features as discrete data. The equality depends on ideal estimation and endpoint conditions; a finite network supplies a **difference of likelihood approximations**. Subtracting two lower bounds also does not establish a lower bound on PMI. Part 2 expands the weighting and approximation gaps. [§4.1, p.4](https://arxiv.org/pdf/2610.20533v1#page=4)

### 4.2 Timestep Weighting

Treating every shared bit equally can let fine texture dominate the objective. The authors emphasize dependencies surviving additional target corruption. They define $\operatorname{PMI}^{*}$ by weighting corrupted-target PMI with $q(s)$, then implement timestep sampling $f(t)$ and weighting $g(t)$:

$$
\operatorname{PMI}^{*}(\mathbf{x},\mathbf{y})
=\mathbb{E}_{s,\boldsymbol{\epsilon}}
\left[q(s)\operatorname{PMI}(\mathbf{x},\mathbf{y}_s)\right].
$$

This supplies an inductive bias about which information determines similarity. **Weighted PMI and ordinary PMI must be distinguished.** The practical setting uses $g(t)=(1-t)/\max(0.33,t)^2$ and changes the sampling distribution. The original $(1-t)/t^3$ setting produces NaN in Table 1(f). Exact-PMI intuition alone therefore does not describe the working recipe. [§4.2, pp.4–5](https://arxiv.org/pdf/2610.20533v1#page=4)

### 4.3 Bootstrapped PMI Estimation

Instead of predicting an entire large view, estimate the information increment from adding one small token to a prefix. The chain-rule identity in Eq. (13) is

$$
\operatorname{PMI}(\mathbf{x},y_{\le k})
-\operatorname{PMI}(\mathbf{x},y_{<k})
=\log\frac{p(y_k\mid y_{<k},\mathbf{x})}{p(y_k\mid y_{<k})}.
$$

The right side measures the advantage of knowing $\mathbf{x}$ when predicting the new token given the existing prefix. Summing the increments telescopes to the full PMI. This motivates small-target supervision of relationships between large contexts. It does not guarantee exact satisfaction of every PMI relation by the finite-sample, weighted objective. [§4.3, p.5](https://arxiv.org/pdf/2610.20533v1#page=5)

### 4.4 Representation Objective

For contexts $\mathbf{x},\mathbf{y}$ and added information $\mathbf{z}$, Eq. (15) takes the form

$$
\mathcal{L}_{\mathrm{RePR}}
=\left(
\left\langle R_\theta(\mathbf{x}),
R_\theta(\mathbf{y},\mathbf{z})-R_\theta(\mathbf{y})\right\rangle
-\widehat{\operatorname{PMI}}^{*}(\mathbf{x},\mathbf{z}\mid\mathbf{y})
\right)^2.
$$

The hat is **review-added notation** for the learned, Monte Carlo estimate; the paper writes $\operatorname{PMI}^{*}$. Its target is the weighted mean difference between denoising $\mathbf{z}$ given $\mathbf{y}$ and given both contexts, entirely under stop-gradient.

The loss specifies how much the representation should change after observing the additional patch and how that change relates to another view. Weighted PMI is asymmetric, however, and PMI matrices are not generally positive semidefinite (PSD). Appendix A.3 adds an embedding identifying the incremental patch role, enabling asymmetric factorization. A simple same-embedding Gram-matrix interpretation misses this important implementation detail.

## 5. Architecture: two paths explain the same prediction advantage

[![Figure 1. The representation-path inner-product increment matches an information difference from the prediction path; relative-position queries produce spatial outputs.](/assets/reviews/repr/figure1.webp)](/assets/reviews/repr/figure1.webp)

*Original Figure 1, p.6. The top path performs feature regression, the bottom compares two conditional predictions of one patch, and the right shows relative-position queries. Quoted “log P” labels denote weighted estimates. The same figure is the thumbnail. Click body figures to enlarge. Original CC BY 4.0 assets retain their fonts, arrows, and data; whitespace trims follow the author LaTeX, followed by rasterization and WebP encoding.*

### 5.1 Prediction Model

The predictor uses MAR: a masked autoencoder (MAE) with a small diffusion multilayer perceptron (MLP) replacing patch regression. $E_\theta$ encodes visible patches, $D_\theta$ builds context with mask tokens, and $M_\theta$ denoises a patch. In Eq. (16), $z_p$ is the **decoder context at a patch location**, not the incremental information $\mathbf{z}$ of Section 4:

$$
\mathcal{L}_{\mathrm{MAR}}
=w(t)\|M_\theta(z_p,p_t,t)-p\|_2^2.
$$

The expensive encoder/decoder computation is shared; the small diffusion head runs repeatedly at different noise levels. Having no negatives does not mean having no additional computation.

### 5.2 Representing Patch Features

Features form an $H\times W\times D$ spatial grid rather than a single classification token. $H,W$ index the output patch grid and $D$ the feature dimension. Inner products compare corresponding spatial positions. Absolute-position cues are withheld from the encoder; queries indexed by offsets from the visible input's center retrieve output features through cross-attention. The intention is spatial equivariance: translating the image should translate its features.

Appendix A.3 explains RoPE, role embeddings, and queries padded beyond image boundaries. Appendix A.5 visualizes the feature similarities. This should not be read as a proof of exact equivariance under every mask and boundary condition.

### 5.3 Masking Strategy

An image supplies potentially overlapping contexts $A,B$ and a target patch $p$ absent from both. $AB$ is their combined context; $Bp$ adds the target to $B$. The predictor compares denoising $p$ from $B$ versus $AB$. The representation path matches $\langle R_\theta(A),R_\theta(Bp)-R_\theta(B)\rangle$.

Mask ratios are sampled from $[M_A,100\%]$ and $[M_B,100\%]$, not fixed. Default minimum ratios are 75% and 93.75%: at $256\times256$ resolution and $16\times16$ patches, these leave at most 64 and 16 visible patches. $A_{\mathrm{Repeat}}=8$ and $p_{\mathrm{Repeat}}=16$ reuse intermediate computation.

### 5.4 Iterative Feature Prediction

The first iteration predicts pixels. Later iterations obtain full-image features from $R_{\theta_{\mathrm{old}}}$ and predict the corresponding patch feature. This changes the signal previously determined by pixel prediction. The identity $\operatorname{PMI}(\mathbf{x},R(\mathbf{y}))=\operatorname{PMI}(\mathbf{x},\mathbf{y})$ requires ideal information-preserving invertibility. Finite-dimensional, weighted, masked features do not automatically satisfy it. Part 2 examines Appendix A.4's argument.

### 5.5 Joint Optimization

Prediction and representation losses train jointly, but the representation loss is applied once every four steps. They share $E_\theta$, so the paths are not entirely independent networks. The regression target is stopped while the MAR loss trains prediction. This distinction is essential for understanding or implementing Algorithm 1. [§5, pp.5–7; A.1/A.3](https://arxiv.org/pdf/2610.20533v1#page=6)

## 6. Experiments: testing information retention and cost

### 6.1 Feature Suppression

[![Figure 2. Clean and watermarked ImageNet validation top-1 versus effective training epochs accounting for views.](/assets/reviews/repr/figure2.webp)](/assets/reviews/repr/figure2.webp)

*Original Figure 2, p.7. The x-axis counts effective epochs accounting for images and views; the y-axis is ImageNet validation top-1 accuracy (%). Dashed means clean, solid means watermarked. Blue is DINOv3-B/16, yellow LeJEPA-S/8, and black RePR-78M.*

Each patch receives a watermark identifying the image. This creates an easy shortcut for identifying whether two views belong to one instance. DINOv3 and LeJEPA's watermarked curves fall to low accuracy, whereas RePR retains substantial class-relevant information.

The result supports robustness **under this constructed watermark condition**. RePR also loses performance relative to its clean curve, and DINOv3's clean classification curve is higher. It establishes neither universal classification superiority nor retention of every latent factor. Probing is also distinct from generation quality. The paper provides no multi-seed confidence intervals for this plot. Part 2 A.6 shows the actual watermark and its $16\times16$ construction.

### 6.2 Scaling Laws

[![Figure 3. Encoder-training FLOPs versus ImageNet FID-10K and GPIC FD-DINOv2-10K, with iterations and extrapolated trends.](/assets/reviews/repr/figure3.webp)](/assets/reviews/repr/figure3.webp)

*Original Figure 3, p.8. Encoder-training FLOPs use a logarithmic x-axis. Left: FID-10K after 80 epochs of ImageNet JiT-B training. Right: FD-DINOv2-10K after one GPIC epoch of MiniT2I. Lower is better. Circles denote pixel prediction; diamonds predict previous features; colors indicate encoder size. Distinguish observed points from extensions of fitted lines and the reference stars.*

Diffusability means how easily a diffusion model can learn the representation distribution. Encoder comparisons hold generative-model size and training duration constant. Latent Forcing generates features before pixels, avoiding a separate lossy decoder as the determining reconstruction bottleneck. A lossless generation route here means **the final target remains original pixels**, not that the encoder is perfectly invertible.

Widths vary from 256 to 1024, giving 20M–306M inference encoders at fixed depth. Model size, training compute, and a second feature-prediction iteration generally improve the metric. There are limits: the width-1024 ImageNet model begins overfitting after about 200 epochs, while GPIC continues improving over the larger budgets examined.

The headline **7× and 8× are extrapolations**. Appendix A.7 reports the following predicted encoder costs for reaching reference quality:

| Evaluation | Reference encoder cost and quality | Extrapolated RePR cost | Compute ratio |
| --- | --- | --- | --- |
| ImageNet, FID-10K | DINOv2: $1.1685129\times10^{22}$ FLOPs, 17.95 | $1.504\times10^{21}$ FLOPs | about 1/7.77 |
| GPIC, FD-DINOv2-10K | WebSSL-DINO-1B: $9.78\times10^{21}$ FLOPs, 277.1 | $1.229\times10^{21}$ FLOPs | about 1/7.96 |

Measured RePR-173M iteration 2 reaches ImageNet FID 24.73; the best listed GPIC iteration-2 value is 325. They do not yet reach 17.95 and 277.1. The paper therefore does not report already attaining those reference qualities at the predicted cheaper costs. The raw tables do not provide all per-point FLOPs or fit uncertainty, and this review has not independently reproduced the regression coefficients or confidence intervals.

FID (Fréchet Inception Distance) compares feature means and covariances in Inception space; FD-DINOv2 uses DINOv2 features. Their numerical scales are not interchangeable. Neither are 10K and 50K sample evaluations, which have different finite-sample biases. These are proxies for visual distributional quality, not guarantees of semantic fidelity, all forms of diversity, or per-image reconstruction accuracy.

### 6.3 Compute Optimal Generation

[![Figure 4. Conditional and unconditional unguided FID-10K against combined encoder and generative-model training FLOPs.](/assets/reviews/repr/figure4.webp)](/assets/reviews/repr/figure4.webp)

*Original Figure 4, p.9. Conditional generation is above, unconditional below. The x-axis includes encoder pretraining. Black is pixel JiT-L; blue/yellow add RePR-44M/78M. Both y-axes show unguided FID-10K but use different ranges.*

At $1.19\times10^{20}$ FLOPs, the authors report unguided FID-10K improvements of 2.61 for conditional and 4.18 for unconditional generation. This comparison uses no perceptual loss, GAN, or external image decoder. Table 3 provides a separate guided 50K evaluation:

| Setting | Total training FLOPs ↓ | FID-50K ↓ | IS ↑ |
| --- | --- | --- | --- |
| Pixel, JiT-L 600ep | $3.57\times10^{20}$ | 2.36 | 298.5 |
| RePR-78M + Pixel, JiT-L 440ep | $3.49\times10^{20}$ | **2.18** | **338.4** |

The budgets differ by about 2.2%; they are close rather than numerically identical. RePR encoder training is included, and Appendix A.5 says the auxiliary training used for guidance is also counted. Inception Score (IS) combines classifier confidence with class diversity, with higher values preferred. It does not directly compare generated and real distributions or establish text fidelity. See the definition in [the IS original, §4](https://arxiv.org/pdf/1606.03498v1#page=4).

The result supports assigning some compute to representation learning **at the tested model sizes and budgets**. It is not a global optimal-allocation theorem. FLOPs use a $6ND$ approximation excluding self-attention: $N$ denotes parameter count and $D$ processed data, distinct from the integration count $N$ and feature dimension $D$ elsewhere. The same ratios need not hold for GPU time, memory, or inference. GPIC also uses a FLAN-T5-large text encoder, so the ImageNet comparison's lack of external models must not be generalized to the entire GPIC pipeline. [§6.3; A.3/A.5](https://arxiv.org/pdf/2610.20533v1#page=8)

### 6.4 Ablation Studies

Table 1 contains six groups answering different questions. Several use classification as a proxy because generative evaluations are expensive; the numbers do not form one common ranking.

| Original group | Settings and results | Interpretation |
| --- | --- | --- |
| (a), 78M, 100ep | $M_A$: 62.5%→58.62, 75%→58.75, 87.5%→56.21 top-1 | Too little context hurts. Accuracy is in %. |
| (b), 78M, 100ep | $M_B$: 75%→56.78, 93.75%→58.75, 98.5%→58.55 top-1 | Short $B$ context helps, but more masking is not monotonically better. |
| (c), 78M, 100ep | uniform 57.63, block 58.75 top-1 | Block masking helps this probing setting. |
| (d), 78M, 200ep | representation weight 0.005→63.1, 0.01→63.6, 0.4→43.9 top-1 | Default 0.005 differs from best listed 0.01. Large weights hurt. |
| (e), 173M, 600ep allocation | pixels only 30.68 → iteration 2 24.73 → iteration 3 23.81 FID-10K | Redistributing total epochs to feature prediction helps; this does not establish exact equality of all operations. |
| (f), 78M, 200ep | original PMI weight: NaN; clipped denominator: 42.15; changed distribution/weight: 35.09/35.72 FID-10K | Stability and timestep bias matter. The default $\mu=1.2$ gives 35.72; $\mu=0.6$ gives the lower 35.09. |

Distinguish Table 1(e)'s nominal 600ep allocation from Appendix A.3's checkpoint accounting with additional 10ep learning-rate decay. Reproduction requires actual executed budgets including decay. Classification improvements in (a–d) do not independently establish generation improvements. [§6.4; Table 1, p.9](https://arxiv.org/pdf/2610.20533v1#page=9)

### 6.5 GAN+LPIPS Decoding

Table 2 tests RePR in the RAE route. With ImageNet-only encoders and 20 epochs of DiT-XL, MoCo-v3-B scores 13.84, I-JEPA-H 12.46, and RePR-173M iteration 3 **11.13 FID-50K**. This is neither a claim to beat every RAE record nor a directly comparable evaluation to Table 3's 2.18.

The decoder uses Learned Perceptual Image Patch Similarity (LPIPS) and generative adversarial network (GAN) losses. The authors explicitly note the external pretrained models: DINOv1/8-based GAN pretraining alone costs $4.7\times10^{20}$ FLOPs. This auxiliary experiment therefore has a different accounting scope from Section 6.3's controlled comparison.

## 7. Conclusion: what can we take from the paper?

The proposal is to **measure useful relationships through generative prediction and directly supervise the patch representation with them**. Watermark robustness, iterative feature prediction, encoder scaling, and pixel-generation comparisons including encoder cost provide complementary evidence.

The useful review-level shift is from selecting an existing classifier-friendly encoder to deciding how much training compute to assign to a generation-friendly encoder. Ideal PMI properties, the practical weighted/bootstrap objective, and finite-budget observations have different conditions. They do not yet establish universal information preservation or unlimited scaling. [§7, p.10](https://arxiv.org/pdf/2610.20533v1#page=10)

Continue to [Part 2](/en/reviews/repr-latent-tokenization-technical/) for A.1–A.7, including the algorithm, accounting details, PMI chain rule, diffusion weighting, asymmetric factorization, and reconstruction conditions.

## Sources and reading scope

- Original: Alan Baade, Kyle Sargent, Vincent Sitzmann, Eric Ryan Chan, Justin Johnson, Li Fei-Fei, Gordon Wetzstein, [Generative Representation Learning for Scalable Latent Tokenization, arXiv:2610.20533v1](https://arxiv.org/abs/2610.20533v1). First submitted 2026-10-14; 22 pages; preprint. A conference-style filename was not treated as acceptance evidence.
- First reading: 2026-10-23, official abstract and introduction pp.1–2, active LaTeX section map, figure/table captions, and §7. An independent orientation was saved before detailed reading.
- Second reading: all active author LaTeX for §1–7 and A.1–A.7, checked against key PDF formulas, tables, and figures. The bibliography's approximately pp.10–15 does not imply full readings of every cited source. This article covers the body; Part 2 completes the appendices.
- Selected originals: Latent Forcing v1 pp.1–3; JiT v2 pp.1–3; RAE v1 pp.1–4; DINOv2 v2 pp.4–6; CPC v2 pp.2–4; VDM v6 pp.17–18. Existing detailed LeJEPA v3 reading records were reused. Absolute FID rankings across mismatched protocols were not combined.
- Verification: source numbers, notation, equality conditions, and supplementary mathematical calculations were checked. GPU training reproduction, raw logs, multiple seeds, and independent scaling fits were **not executed**. Equation, link, and rendered-page checks are recorded in the handoff.
- Figures 1–4: original [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) assets, preserving content and fonts with author whitespace trims, rasterization, and WebP conversion. No additional custom diagram was created.
