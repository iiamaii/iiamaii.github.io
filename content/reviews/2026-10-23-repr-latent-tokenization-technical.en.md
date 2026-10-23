---
title: "[2/2] RePR in depth: PMI derivations, implementation, and scaling evidence"
description: "Follow Appendices A.1–A.7 through the algorithm, PMI conditions, raw results, and limits of the extrapolations."
date: "2026-10-23"
publishedAt: "2026-10-23T15:12:15+09:00"
updatedAt: "2026-10-23T15:12:15+09:00"
topics: ["representation learning", "latent tokenization", "diffusion models", "generative modeling", "scaling laws"]
visibility: "public"
lang: "en"
translationKey: "repr-latent-tokenization-technical"
paperTitle: "Generative Representation Learning for Scalable Latent Tokenization"
paperPublishedDate: "2026-10-14"
authors: ["Alan Baade", "Kyle Sargent", "Vincent Sitzmann", "Eric Ryan Chan", "Justin Johnson", "Li Fei-Fei", "Gordon Wetzstein"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.20533v1"
thumbnail: "/assets/reviews/repr/figure1.webp"
thumbnailAlt: "Original Figure 1: RePR matches conditional predictive information differences with increments of patch-representation inner products."
---

RePR (Relative Prediction Representations) transfers a generative model's **relative predictive advantage** into an increment of representation inner products. This article follows that explanation into its algorithm, assumptions, and measurements. Read [\[1/2\] Main-paper review](/en/reviews/repr-latent-tokenization/) first; here we retain **A.1→A.2→A.3→A.4→A.5→A.6→A.7**.

Keep three levels separate. The probability chain rule is an exact identity. Ideal pointwise mutual information (PMI) factorization has capacity and information-preservation conditions. Practical RePR uses learned predictors, timestep weights, masks, and finite Monte Carlo estimates. The paper has no numbered Theorem/Lemma sequence or long formal proof; its short Appendix A.4 reconstruction argument and the main probability derivations motivate the **review-added derivations** below.

## A.1 Algorithm

### Targets and gradient paths

Algorithm 1 samples contexts $A,B$, their combined context $AB$, and a target patch $p$ absent from both from image $X$. The feature path requires $A,B,Bp$; the prediction path requires $B,AB$.

| Path | Computation | Purpose |
| --- | --- | --- |
| MAR training | $E_\theta,D_\theta,M_\theta$ | Generative reconstruction loss for the selected masked targets |
| Information estimate | Denoise $p$ with both contexts using the same $p_t,t$ | Weighted predictive advantage of adding context $A$ |
| Feature training | $\langle R_\theta(A),R_\theta(Bp)-R_\theta(B)\rangle$ | Feature change that should match the information estimate |

The following pseudocode preserves the core order of original Algorithm 1. It is not an executed result or a ready-to-run library API. Its $R$ branch operates on shared encoder outputs, while the main text's $R_\theta(\mathbf{x})$ denotes the full representation path.

```python
a, b, ab, bp = E(A), E(B), E(AB), E(B, p)
d_ab, d_b = D(ab), D(b)
# MAR: selected targets x, independently sampled t and eps
mar_loss = mean(w(t) * (M(x_t, t, d_ab[m_AB]) - x)**2)
with no_grad():
    # N noise/time samples; the two predictions share them
    p_ab = M(p_t, t, d_ab[p_i])
    p_b = M(p_t, t, d_b[p_i])
    pmi = mean(g(t) * ((p_b - p)**2 - (p_ab - p)**2))
pred = dot(R(a), R(bp) - R(b))
repr_loss = mean((pred - pmi)**2)
loss = mar_loss + loss_weight * repr_loss
```

As in the source, target estimation sits inside `no_grad()`. The representation loss cannot directly improve itself by changing its numerical target. However, the paths share $E_\theta$, so the feature loss is not unrelated to every predictor parameter. The MAR loss continues training prediction. One joint representation/MAR step is followed by three MAR-only steps.

The main Eq. (16) and pseudocode present $M$'s arguments in different orders. Track the meanings—noisy patch, time, context—and inspect the actual implementation's function signature before coding. The public code was not executed for this review. Averaging axes in `mean` versus summed vector norms can also change numerical scaling and the interpretation of a loss coefficient. [A.1; Algorithm 1, p.16](https://arxiv.org/pdf/2610.20533v1#page=16)

## A.2 Additional Experiments

### Integration samples and shared noise

[![Figure 5. Error against a 4096-step PMI estimate versus Monte Carlo sample count, comparing shared and independent noise.](/assets/reviews/repr/figure5.webp)](/assets/reviews/repr/figure5.webp)

*Original Figure 5, p.16. The x-axis is integration sample count $N$; the y-axis is mean squared error (MSE) against the 4096-sample estimate. Purple solid uses shared noise; orange dashed uses independent noise. The reference is a higher-sample approximation, not exact true PMI. Click body figures to enlarge.*

Comparing the same patch under the same noisy realization can help. With review-added symbols $U,V$ for the two error estimates,

$$
\operatorname{Var}(U-V)
=\operatorname{Var}(U)+\operatorname{Var}(V)
-2\operatorname{Cov}(U,V).
$$

If common randomness makes the estimates fluctuate together with positive covariance, the difference has lower variance. This is not a universal theorem favoring shared noise; Figure 5 demonstrates the effect for this setup. The authors choose $N=32$ and report diffusion integration as 1.3% of total training FLOPs. That percentage describes the **integration component**, not every encoder/decoder path or all representation-learning overhead.

### Reconstruction and generation answer different questions

Original Table 4 evaluates reconstruction through an RAE decoder:

| Encoder | Reconstruction FID (rFID) ↓ | PSNR ↑ |
| --- | --- | --- |
| DINOv2-B | 0.49 | 18.09 dB |
| RePR-B iteration 3, labeled 600ep | 0.47 | 21.72 dB |

Peak signal-to-noise ratio (PSNR) measures pixel error. The following standard definition explains the metric; it is not a new theorem from this paper. Let $L$ denote the reference maximum signal value and $n$ the number of compared elements:

$$
\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2,
\qquad
\operatorname{PSNR}=10\log_{10}\frac{L^2}{\operatorname{MSE}}.
$$

Higher PSNR is better under identical ranges, preprocessing, and aggregation; exact reconstruction gives infinity. Perceptually natural generation is a different construct. rFID compares the feature distribution of reconstructed images with real images: a low rFID does not establish exact per-input reconstruction. Nor should 0.47 be compared with generation FID-50K 2.18. Table 4's encoder label and 600ep description are distinct from Appendix A.3's detailed RAE comparison budget; they should not silently be treated as the same checkpoint accounting. [A.2; Table 4, pp.16–17](https://arxiv.org/pdf/2610.20533v1#page=17)

## A.3 Implementation Details

### Architecture and actual budgets

Inference uses 18 $E_\theta$ layers and six representation-branch layers: three additional self-attention and three cross-attention layers. The six training decoder $D_\theta$ layers are separate. At width 1024 the inference architecture matches ViT-L. Sizes are 20M/44M/78M/173M/306M at widths 256/384/512/768/1024. The output dimension remains 1024.

| Item | Key settings in original Tables 5–6 |
| --- | --- |
| Images and patches | $256\times256$, $16\times16$ patches |
| Encoder training | batch 2048, AdamW, learning rate $2.4\times10^{-4}$, width-specific $\mu$P multipliers |
| Augmentation | crop scale $[0.8,1.0]$, horizontal flip |
| MAR time sampling | logit-normal $\mu=1.2,\sigma=1.2$, 10% uniform mixture |
| RePR time sampling | pixels $\mu=1.2$, features $\mu=1.6$, shared $\sigma=1.2$ |
| Weighting and reuse | $g(t)=(1-t)/\max(0.33,t)^2$, loss weight 0.005, $A_{\mathrm{Repeat}}=8$, $p_{\mathrm{Repeat}}=16$, $N=32$ |
| Generative scaling evaluation | ImageNet JiT-B, GPIC MiniT2I; 80 ImageNet-equivalent epochs |
| GPIC conditioning | FLAN-T5-large text encoder, at most 128 text tokens |

An epoch does not always mean one pass over the named dataset. The authors divide total processed image views by the ImageNet sample count. One GPIC epoch is approximately 80 ImageNet-equivalent epochs. Checkpoints also receive ten additional learning-rate-decay epochs. Thus iteration 2 at $210+210$ includes 200+10 pixel epochs and 200+10 feature epochs.

Figure 4 uses iteration-2 models trained for 420 total epochs. Training RePR-44M/78M represents roughly 20%/35% of the overall budget corresponding to JiT-L 200ep. Table 3 uses RePR-78M iteration 3 with 830 total epochs, split 210+210+410; the authors describe encoder training as about 25% of the corresponding JiT-L 600ep budget. The RAE comparison separately uses RePR-B for 1030 total epochs and $2.19\times10^{20}$ FLOPs, versus I-JEPA-H's $2.45\times10^{20}$.

FLOPs use $6ND$ without self-attention, not measured wall-clock cost. The authors report about 31 GPU-hours for an ImageNet generative evaluation and 56 for GPIC, and **estimate** 250 hours on one eight-GPU B200 node to reproduce Figure 3. This review did not execute or measure those runtimes. [A.3; Tables 5–6, pp.17–19](https://arxiv.org/pdf/2610.20533v1#page=17)

### Why asymmetric role information matters

Appendix A.3 explicitly notes that weighted PMI is asymmetric and PMI is not generally PSD. A same-embedding Gram matrix is always PSD. For any coefficient vector $c$, a review derivation gives

$$
c^\top Kc
=\left\|\sum_i c_iR(\mathbf{x}_i)\right\|_2^2\ge0,
\qquad K_{ij}=\langle R(\mathbf{x}_i),R(\mathbf{x}_j)\rangle.
$$

Yet a two-state joint distribution with diagonal probabilities 0.45 and off-diagonal probabilities 0.05 has uniform marginals and PMI matrix

$$
\begin{pmatrix}\log1.8&\log0.2\\\log0.2&\log1.8\end{pmatrix}
$$

One eigenvalue is $\log(0.36)<0$. Exact reconstruction by a same-embedding inner product is therefore impossible for this example. This is a review-authored illustration.

Role-dependent encodings permit singular value decomposition (SVD). For a target matrix $M=U\Sigma V^\top$, choose

$$
F=U\Sigma^{1/2},\qquad G=V\Sigma^{1/2},
\qquad FG^\top=M
$$

This requires dimension at least equal to rank and exact fitting of all required targets. Arbitrary finite-dimensional, weighted sampled regressions do not automatically reach that exact solution. RePR marks the incremental patch $\mathbf{z}$ with a learned location embedding, providing the role distinction that supplements the apparently symmetric main-text notation.

## A.4 Properties of PMI

The original section contains a short reconstruction argument. We also expand the relevant probability derivations from main Sections 3–4 here because they are prerequisites. No new Theorem numbers are attributed to the paper, and these calculations do not establish a complete guarantee for every practical loss.

### Why a contrastive score estimates a density ratio — review derivation

[CPC Eq. (5)](https://arxiv.org/pdf/1807.03748v2#page=4) assumes one candidate is drawn from $p(\mathbf{y}\mid\mathbf{x})$ and the others from $p(\mathbf{y})$. In the Bayes posterior for the positive index, the common product of marginal probabilities cancels:

$$
p(d=i\mid\mathbf{y}_{1:N},\mathbf{x})
=\frac{p(\mathbf{y}_i\mid\mathbf{x})/p(\mathbf{y}_i)}
{\sum_j p(\mathbf{y}_j\mid\mathbf{x})/p(\mathbf{y}_j)}.
$$

Categorical cross-entropy is minimized by that posterior, so its log score equals PMI up to an offset depending on $\mathbf{x}$. A different negative proposal changes the denominator of the ratio. Arbitrary augmentation and sampling choices cannot all be interpreted as estimating the same PMI.

### Deriving the diffusion weight $(1-t)/t^3$

The diffusion loss in the cited [VDM Eqs. (53–54)](https://arxiv.org/pdf/2107.00630v6#page=18) contains the derivative of the signal-to-noise ratio (SNR). Under RePR's path,

$$
\operatorname{SNR}(t)=\frac{(1-t)^2}{t^2},
\qquad
-\frac12\frac{\mathrm{d}}{\mathrm{d}t}\operatorname{SNR}(t)
=\frac{1-t}{t^3}.
$$

The weight follows from the variational objective for this noise path. It is singular as $t\to0$, however. Ideal discrete-data and optimal-denoiser limits are distinct from finite-precision training stability, making Table 1(f)'s failure consequential.

Vanishing prior/reconstruction endpoints and a tight denoising bound connect the expression to true log probability. Finite predictors leave errors in the likelihood difference. Let the two nonnegative gaps between a fixed model G's exact log probabilities and its VLBs be $\Delta_{\mathrm{cond}},\Delta_{\mathrm{base}}$, **review-added notation**:

$$
\log\frac{P_G(\mathbf{y}\mid\mathbf{x})}{P_G(\mathbf{y})}
=\operatorname{VLB}_G(\mathbf{y}\mid\mathbf{x})-\operatorname{VLB}_G(\mathbf{y})
+\Delta_{\mathrm{cond}}-\Delta_{\mathrm{base}}.
$$

The gap difference has no fixed sign. Both gaps approaching zero recovers that model's likelihood ratio. Matching data PMI additionally requires matching the true conditional and marginal distributions; subtracting lower bounds does not yield another lower bound. The actual changed $f,g$ objective should not silently be identified with the original probability quantity.

### Intermediate steps of the bootstrap chain rule

Main Eqs. (13–14) expand as

$$
\begin{aligned}
\operatorname{PMI}(\mathbf{x},y_{\le K})
&=\log\frac{p(y_1,\ldots,y_K\mid\mathbf{x})}{p(y_1,\ldots,y_K)}\\
&=\sum_{k=1}^{K}\log\frac{p(y_k\mid y_{<k},\mathbf{x})}{p(y_k\mid y_{<k})}.
\end{aligned}
$$

Subtracting the expression for the shorter prefix cancels every term except the newest token's conditional PMI. Summing over $k$ cancels all intermediate prefixes; empty-context PMI is zero. Required conditional probabilities must be defined on the relevant support, with zero-probability handling addressed separately.

Training on finitely many sampled contexts does not make a neural network satisfy every prefix identity exactly. Treating weighted patch increments as the original unweighted global PMI also needs consistency conditions. This is the transition from an exact identity to a practical regression target.

### The scope of reconstruction from PMI

The source considers an uncorrupted image $\mathbf{x}$ and a target pixel $\mathbf{y}$, with nonzero support over candidate pixel values, and proposes recovering the pixel by maximizing PMI. At a fixed pixel position, the argument is clear if **the image determines the true pixel, the alphabet is finite, marginals are positive, and factorization is exact**. The true conditional probability is one and all others zero; the true candidate has finite log ratio and the others have $-\infty$.

Positive support alone at nonzero noise does not establish exact argmax reconstruction in every case. As a review example, if $p(y=1)=0.0001$ and $p(y=1\mid\mathbf{x})=0.01$, candidate 1 has ratio 100. Candidate 0 has ratio about 0.9901. PMI argmax chooses 1 although the conditional mode is 0. A positive joint distribution can realize these marginals and conditionals. **Maximizing PMI and maximizing conditional probability are different operations in general.**

A zero-noise limit can be argued separately for a finitely separated alphabet with fixed positive priors. It should not be promoted to unconditional reconstruction at finite noise, under masking, finite rank, and weighted regression. The source itself qualifies practical reconstruction by mask sampling and convergence.

This also matters for iterative feature prediction. For an invertible deterministic $R$, the corresponding probability ratios preserve

$$
\operatorname{PMI}(\mathbf{x},R(\mathbf{y}))
=\operatorname{PMI}(\mathbf{x},\mathbf{y})
$$

A noninvertible representation that merges outcomes need not preserve it. The data processing inequality bounds average mutual information and allows information loss. The paper combines an ideal explanation with empirical benefits from iterative prediction; it does not prove lossless information retention at every practical iteration. [§5.4; A.4, pp.7,17–18](https://arxiv.org/pdf/2610.20533v1#page=17)

## A.5 Qualitative Results

[![Figure 6. Uncurated class-conditional ImageNet examples from the compute-comparison JiT-L system.](/assets/reviews/repr/figure6.webp)](/assets/reviews/repr/figure6.webp)

*Original Figure 6, p.20. Labeled uncurated by the authors. These examples show objects, texture, and backgrounds; they do not establish duplication rates, failure frequency, or full diversity.*

Sampling applies classifier-free guidance (CFG) 3.0 to latent diffusion on $[0.06,1.0]$ and AutoGuidance weight 1.5. Appendix A.5 counts AutoGuidance training as 1.6% of total training FLOPs and includes it in accounting. This protocol differs from unguided FID-10K scaling evaluations.

[![Figure 7. Cosine similarity to a red center patch, comparing features with and without spatial equivariance.](/assets/reviews/repr/figure7.webp)](/assets/reviews/repr/figure7.webp)

*Original Figure 7, p.20. Cosine similarity to the red center patch visualizes spatial structure. It does not directly measure generation-FID improvement or prove mathematical equivariance for every translation.*

## A.6 Watermarking

[![Figure 8. Clean images and watermarked versions with repeated identifying information at patch scale.](/assets/reviews/repr/figure8.webp)](/assets/reviews/repr/figure8.webp)

*Original Figure 8, p.21. Compare clean and watermarked rows. The authors reduce the prior setup's watermark from $32\times32$ to $16\times16$, making individual patches sufficient for identification. This constructed condition does not establish the same failure frequency across all natural images.*

The setup distinguishes information sufficient for identifying a view from information useful for predicting another patch's content. Comparing two conditional predictions gives RePR a way to learn the latter structure. New data where the watermark also shortcuts the generative objective would need another test. Figure 2's clean performance and RePR's remaining watermark degradation should also be retained.

## A.7 Raw Numbers for Scaling Plots

### ImageNet: measured results

Original Table 7 uses JiT-B trained for 80 epochs and FID-10K. Encoder epoch sums include earlier feature-learning iterations.

| Encoder | Pixel-prediction epochs → FID | Feature-prediction epochs → FID |
| --- | --- | --- |
| 20M | 110→41.61, 210→40.19, 410→37.01, 610→37.21 | 210+110→36.24, 210+210→34.89 |
| 44M | 110→40.93, 210→38.20, 410→36.07, 610→34.82 | 210+110→31.64, 210+210→29.48, 210+410→29.22 |
| 78M | 110→38.26, 210→35.72, 410→34.31, 610→34.00 | 210+110→29.81, 210+210→27.80, 210+410→27.36 |
| 173M | 110→37.22, 210→33.73, 410→31.27, 610→30.68 | 210+210→25.82, 210+410→24.73 |
| 306M | 110→36.14, 211→31.23 | not listed |

The 20M model worsens slightly from 410 to 610 epochs. The 306M entry **211 is preserved from the source**, not silently changed to 210. Missing entries are neither zero nor demonstrated failures. Every point is not a monotonic improvement with width.

### GPIC: a different feature space

Table 8 reports FD-DINOv2-10K, not numerically interchangeable FID. MiniT2I trains for one GPIC epoch; the encoder epochs below are ImageNet equivalents.

| Encoder | Pixel-prediction epochs → FD | Feature-prediction epochs → FD |
| --- | --- | --- |
| 44M | 110→514, 210→479, 410→456 | 210+110→416, 210+210→393 |
| 78M | 110→498, 210→463, 410→443 | 210+110→384, 210+210→365 |
| 173M | 110→450, 210→423, 410→409 | 210+110→359, 210+210→354 |
| 306M | 110→451, 210→404, 410→381 | 210+110→347, 210+210→325 |

The MiniT2I reference is 546.3 and the WebSSL-DINO-1B reference 277.1. Retain both the measured improvement and the fact that the WebSSL target has not yet been reached.

### What produces the 7× and 8× claims?

Dividing the paper's power-law extrapolations gives the following. This is **arithmetic checking of reported values**, not an independent fit:

$$
\frac{1.1685129\times10^{22}}{1.504\times10^{21}}
\approx7.77,
\qquad
\frac{9.78\times10^{21}}{1.229\times10^{21}}
\approx7.96.
$$

The source fits trends across approximately two orders of magnitude and describes extrapolating at most one measured log-scale range beyond observations. The conclusion depends on fit form, data scale, overfitting, and downstream training budget. Full per-point compute details, regression residuals, and multi-seed uncertainty are not available in the tables; this review did not invent a new fitted curve or confidence interval. [§6.2; A.7; Tables 7–8, pp.8,22](https://arxiv.org/pdf/2610.20533v1#page=22)

### Interpreting FID and FD

Using **review notation**, let $h$ be a feature extractor, $\mu_r,\mu_g$ real/generated feature means, and $C_r,C_g$ covariances. The standard Fréchet expression is

$$
D_F(h)=\|\mu_r-\mu_g\|_2^2+
\operatorname{Tr}\!\left(C_r+C_g-
2(C_r^{1/2}C_gC_r^{1/2})^{1/2}\right).
$$

The standard FID definition comes from [Heusel et al., Eq. (6), p.6](https://arxiv.org/pdf/1706.08500v6#page=6). FID uses Inception features; GPIC FD here uses DINOv2. Ideally the distance is nonnegative and lower is better, but zero only matches those feature means and covariances. It does not establish equality of full image distributions or text fidelity. Finite-sample estimation bias prevents mixing 10K and 50K protocols, and reconstruction rFID differs from newly generated-image FID. Feature-extraction details, reference subsets, and error bars not established by the source remain unverified.

## Review interpretation and next verification

The work's strength is connecting an information-based feature objective to practical patch training and combined-cost evaluation. The first recommended independent check is **a small-budget reproduction with complete tokenizer-cost accounting**. This tests accounting boundaries before attempting the extrapolated reference-quality targets.

Alternatives are testing weighted-PMI bias and variance on a small discrete distribution, or comparing LeJEPA's distributional non-degeneracy and generation-relevant information under matched conditions. The next research decision is which of compute reproduction, estimator validation, or representation comparison to prioritize. These are proposed investigations, not completed results.

## Sources and reading scope

- Original: [arXiv:2610.20533v1](https://arxiv.org/abs/2610.20533v1), by Alan Baade and all seven authors listed in the expandable metadata. First submitted 2026-10-14; preprint; 22 pages.
- Two readings: an independent orientation was saved on 2026-10-23 before reading all active author LaTeX for §1–7 and A.1–A.7 and comparing key PDF equations, experiments, and visuals. This article follows A.1–A.7; main-text derivations are relocated into A.4 to explain its prerequisites. Not every cited original was read in full.
- Selected originals: CPC v2 pp.2–4; VDM v6 pp.17–18; Latent Forcing v1 pp.1–3; JiT v2 pp.1–3; RAE v1 pp.1–4; DINOv2 v2 pp.4–6. Prior detailed LeJEPA v3 records were reused.
- Verification: review calculations examine the chain rule, SNR derivative, non-PSD PMI and asymmetric SVD, support/argmax distinction, invertible-feature ratios, shared-noise variance, and extrapolation ratios. These are partial mathematical checks, not formal verification of the entire paper.
- **Not executed:** GPU training; public code execution or complete audit; dataset, raw-log, or multi-seed reproduction; independent extrapolation fitting and validation at future budgets.
- Figures 5–8 and thumbnail Figure 1 are original [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) assets. Content, layout, and fonts are retained with author whitespace trims, rasterization/resizing, and WebP conversion. Photographic panels use quality 92. No new model outputs were synthesized.
