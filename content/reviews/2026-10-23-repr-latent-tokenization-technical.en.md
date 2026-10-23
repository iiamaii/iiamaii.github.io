---
title: "[2/2] RePR in depth: PMI derivations, implementation, and scaling evidence"
description: "Follow Appendices A.1–A.7 through the algorithm, PMI conditions, raw results, and the conditions behind the reconstruction argument."
date: "2026-10-23"
publishedAt: "2026-10-23T15:12:15+09:00"
updatedAt: "2026-10-23T15:26:26+00:00"
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

RePR’s appendix contains the pseudocode, implementation settings, properties of pointwise mutual information (PMI), and raw scaling numbers behind Relative Prediction Representations. In A.7 Table 7, the 78M encoder reaches a Fréchet Inception Distance (FID)-10K of 34.31 after 410 epochs of pixel prediction, but 27.80 at iteration 2, which adds 210 epochs of feature prediction to 210 pixel epochs (420 in total; the generator is JiT-B trained for 80 epochs). [Table 7](https://arxiv.org/pdf/2610.20533v1#page=22) **Review interpretation.** Of the paper's theory, the diffusion weight (§4.1) and chain rule (§4.3) are exact derivations, whereas reconstruction (A.4) and PMI preservation under feature prediction (§5.4) are arguments that assume an optimal factorization, so the evidence for trained encoders is Table 4's reconstruction peak signal-to-noise ratio (PSNR) of 21.72 dB (DINOv2: 18.09 dB) and the iteration-2 gains in Tables 7–8.

[\[1/2\] The main part](/en/reviews/repr-latent-tokenization/) covers Sections 1–7. The paper has no numbered Theorem/Lemma or long formal proof; its theory appendix is the short reconstruction argument in A.4. Intermediate steps for the probability formulas of main Sections 3–4 are collected in the **review-added derivations** at the end of A.4.

## A.1 Algorithm

### Targets and gradient paths

Algorithm 1 samples contexts $A,B$, their combined context $AB$, and a target patch $p$ absent from both from image $X$. The feature path requires $A,B,Bp$; the prediction path requires $B,AB$.

| Path | Computation | Purpose |
| --- | --- | --- |
| MAR training | $E_\theta,D_\theta,M_\theta$ | Generative reconstruction loss for the selected masked targets |
| Information estimate | Denoise $p$ with both contexts using the same $p_t,t$ | Weighted predictive advantage of adding context $A$ |
| Feature training | $\langle R_\theta(A),R_\theta(Bp)-R_\theta(B)\rangle$ | Feature change that should match the information estimate |

The following pseudocode preserves the core order of original Algorithm 1. Its $R$ branch operates on shared encoder outputs, while the main text's $R_\theta(\mathbf{x})$ denotes the full representation path. The efficiency parameters $A_{\mathrm{Repeat}}$ and $p_{\mathrm{Repeat}}$ are omitted, as in the original.

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

As in the source, target estimation sits inside `no_grad()`, so the representation loss cannot shrink by moving its own target. Its gradient does flow into the shared encoder $E_\theta$, and the MAR loss keeps training the predictor. According to original Table 5, joint MAR+representation steps and MAR-only steps are interleaved 1:3. In feature-prediction iterations, the MAR target becomes $R_{\theta_{\mathrm{old}}}(E_{\theta_{\mathrm{old}}}(x))$ and $p$ becomes that feature at position $p_i$. Eq. (16) writes $M_\theta(z_p,p_t,t)$, while the pseudocode orders the arguments as `M(p_t, t, d)`. [A.1; Algorithm 1, p.16](https://arxiv.org/pdf/2610.20533v1#page=16)

## A.2 Additional Experiments

### Integration samples and shared noise

[![Figure 5. Error against a 4096-step PMI estimate versus Monte Carlo sample count, comparing shared and independent noise.](/assets/reviews/repr/figure5.webp)](/assets/reviews/repr/figure5.webp)

*Original Figure 5, p.16. The x-axis is integration sample count $N$; the y-axis is mean squared error (MSE) against the 4096-sample estimate. Purple solid uses shared noise; orange dashed uses independent noise. The reference is the 4096-sample approximation, not the true PMI.*

Comparing the same patch under the same noisy realization can help. With review-added symbols $U,V$ for the two error estimates,

$$
\operatorname{Var}(U-V)
=\operatorname{Var}(U)+\operatorname{Var}(V)
-2\operatorname{Cov}(U,V).
$$

If common randomness makes the estimates fluctuate together (positive covariance), the difference has lower variance, and the caption of original Figure 5 likewise says that sharing noise strongly reduces variance. The authors choose $N=32$, at which diffusion integration (running the small head $N$ times) takes 1.3% of total training FLOPs.

### Reconstruction metrics

Original Table 4 reports reconstruction with the RAE decoder configuration:

| Encoder | Reconstruction FID (rFID) ↓ | PSNR ↑ |
| --- | --- | --- |
| DINOv2-B | 0.49 | 18.09 dB |
| RePR-B iteration 3, labeled 600ep | 0.47 | 21.72 dB |

PSNR measures pixel error. With $L$ the maximum pixel value and $n$ the number of compared elements, the standard definition is

$$
\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2,
\qquad
\operatorname{PSNR}=10\log_{10}\frac{L^2}{\operatorname{MSE}},
$$

and higher is better under the same pixel range and preprocessing. rFID is a distributional metric comparing the features of reconstructed and real image sets, while PSNR measures per-input reconstruction accuracy. Appendix A.4 cites this table's PSNR gap as empirical support for the reconstruction claim. [A.2; Table 4, pp.16–17](https://arxiv.org/pdf/2610.20533v1#page=17)

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

The paper's epoch is the number of processed image views divided by the ImageNet sample count, which keeps it proportional to FLOPs regardless of $A_{\mathrm{Repeat}}$. One GPIC epoch is approximately 80 ImageNet-equivalent epochs. The authors train one model at a stable learning rate, save a checkpoint every 100 epochs, and linearly decay the learning rate for 10 more epochs from each checkpoint. Thus iteration 2 at $210+210$ includes 200+10 pixel epochs and 200+10 feature epochs. [Section 6.4 of Part 1](/en/reviews/repr-latent-tokenization/) maps these counts to the epoch labels of main Table 1.

Figure 4 uses iteration-2 models trained for 420 total epochs. Training RePR-44M/78M represents roughly 20%/35% of the overall budget corresponding to JiT-L 200ep. Table 3 uses RePR-78M iteration 3 with 830 total epochs, split 210+210+410; the authors describe encoder training as about 25% of the corresponding JiT-L 600ep budget. The RAE comparison separately uses RePR-B for 1030 total epochs and $2.19\times10^{20}$ FLOPs, versus I-JEPA-H's $2.45\times10^{20}$.

FLOPs use $6ND$ without self-attention. The authors report about 31 GPU-hours for an ImageNet generative evaluation and 56 for GPIC, and **estimate** 250 hours on one eight-GPU B200 node to reproduce Figure 3. [A.3; Tables 5–6, pp.17–19](https://arxiv.org/pdf/2610.20533v1#page=17)

### Why asymmetric role information matters

Appendix A.3 states that because weighted PMI is asymmetric and PMI is not positive semidefinite (PSD), inputs corresponding to the incremental information $\mathbf{z}$ are marked separately from $\mathbf{x}$ and $\mathbf{y}$. A same-embedding Gram matrix is always PSD. For any coefficient vector $c$, a review derivation gives

$$
c^\top Kc
=\left\|\sum_i c_iR(\mathbf{x}_i)\right\|_2^2\ge0,
\qquad K_{ij}=\langle R(\mathbf{x}_i),R(\mathbf{x}_j)\rangle.
$$

As a review example, a two-state joint distribution with diagonal probabilities 0.45 and off-diagonal probabilities 0.05 has uniform marginals and PMI matrix

$$
\begin{pmatrix}\log1.8&\log0.2\\\log0.2&\log1.8\end{pmatrix}
$$

with one eigenvalue $\log(0.36)<0$, so no same-embedding inner product can match it exactly.

The paper states that the role marking makes the inner product asymmetric, allowing a perfect singular value decomposition (SVD) reconstruction of the PMI at optimality. For a target matrix $\Pi=U\Sigma V^\top$,

$$
F=U\Sigma^{1/2},\qquad G=V\Sigma^{1/2},
\qquad FG^\top=\Pi,
$$

which requires a representation dimension at least equal to the rank. RePR marks this role by adding a learned embedding to the patch features at the position of $\mathbf{z}$.

## A.4 Properties of PMI

### The reconstruction claim

The paper takes an uncorrupted image view $\mathbf{x}$ and a pixel $\mathbf{y}$ to reconstruct, and gives every pixel value nonzero support by adding Gaussian noise of arbitrarily small variance $\delta>0$ to $\mathbf{y}$ during training. It then states that $\mathbf{y}$ can be recovered by taking, over all candidate pixel values $v$, the one with the largest $\langle R(\mathbf{x}),R(v)\rangle$, that is, the largest PMI. It qualifies that practical reconstruction depends on the sampled masks and on convergence, and cites Table 4's PSNR of 21.72 dB, 3.63 dB above DINOv2 with the RAE ViT-XL decoder, as empirical support.

**Review-added derivation.** Under this assumption, let $y$ be the true pixel value, $k(d)=e^{-d^2/(2\delta)}$ the kernel, and $\pi$ the marginal distribution of pixel values. Candidate $v$ then has PMI $\log k(v-y)-\log\sum_u\pi(u)k(v-u)$. As $\delta\to0$, the PMI of the true value $y$ tends to $-\log\pi(y)$ while every other candidate goes to $-\infty$, so for fixed marginals the argmax is $y$ once $\delta$ is small enough. At a fixed $\delta$ this can fail. If a pixel takes only the values 0, 1, 2 (spacing 1) and the true value is $y=1$, then

$$
\operatorname{PMI}(\mathbf{x},0)>\operatorname{PMI}(\mathbf{x},1)
\iff \pi(0)<\pi(2)\,e^{-1/\delta}.
$$

For example, with $\delta=0.5$ and $\pi=(0.01,0.49,0.50)$, candidate 0 has PMI about 0.61 nat and the true value 1 about 0.39 nat, so the rare neighbor 0 is selected. **Review interpretation.** The paper's argument holds in the limit $\delta\to0$ at fixed marginals, and how small $\delta$ must be depends on how rare the neighboring pixel values are (here $\delta\le1/\log(\pi(2)/\pi(0))$).

### Feature prediction and PMI preservation

Main Section 5.4 builds on this reconstruction property. If an optimal factorization reconstructs the input, the deterministic representation $R$ is injective, and an injective $R$ carries probabilities over unchanged, so for unweighted PMI

$$
\operatorname{PMI}(\mathbf{x},R(\mathbf{y}))
=\operatorname{PMI}(\mathbf{x},\mathbf{y}).
$$

**Review interpretation.** The identity is therefore a property of the optimum at which the reconstruction above holds; it breaks once a trained $R_{\theta_{\mathrm{old}}}$ maps different inputs to the same feature, and average mutual information can then decrease by the data processing inequality. [§5.4; A.4, pp.7,17–18](https://arxiv.org/pdf/2610.20533v1#page=17)

### Review-added derivations: intermediate steps for main Sections 3–4

**Why a contrastive score estimates a density ratio (Eq. (2)).** [Eq. (5) of Contrastive Predictive Coding (CPC)](https://arxiv.org/pdf/1807.03748v2#page=4) assumes one candidate is drawn from $p(\mathbf{y}\mid\mathbf{x})$ and the others from $p(\mathbf{y})$. In the Bayes posterior for the positive index, the common product of marginal probabilities cancels:

$$
p(d=i\mid\mathbf{y}_{1:N},\mathbf{x})
=\frac{p(\mathbf{y}_i\mid\mathbf{x})/p(\mathbf{y}_i)}
{\sum_j p(\mathbf{y}_j\mid\mathbf{x})/p(\mathbf{y}_j)}.
$$

Categorical cross-entropy is minimized by that posterior, so its log score equals PMI up to an offset depending on $\mathbf{x}$, which is Eq. (2). Drawing negatives from a different distribution replaces the denominator of the ratio with that distribution.

**The diffusion weight $(1-t)/t^3$ (Eq. (7)).** The diffusion loss in the cited [Eqs. (53–54) of Variational Diffusion Models (VDM)](https://arxiv.org/pdf/2107.00630v6#page=18) contains the derivative of the signal-to-noise ratio (SNR). Under RePR's path,

$$
\operatorname{SNR}(t)=\frac{(1-t)^2}{t^2},
\qquad
-\frac12\frac{\mathrm{d}}{\mathrm{d}t}\operatorname{SNR}(t)
=\frac{1-t}{t^3},
$$

so the weight follows from the variational objective for this noise path. It diverges as $t\to0$, and in original Table 1(f) the setting that uses it unchanged produced NaN.

**VLB gaps.** Main Section 4.1 states that the VLB equals $\log p$ when the endpoint terms vanish and the denoiser is optimal, and that the approximation approaches the true PMI as prediction improves. Writing the two nonnegative gaps between a fixed model $G$'s exact log probabilities and its VLBs as $\Delta_{\mathrm{cond}},\Delta_{\mathrm{base}}$ (review notation),

$$
\log\frac{P_G(\mathbf{y}\mid\mathbf{x})}{P_G(\mathbf{y})}
=\operatorname{VLB}_G(\mathbf{y}\mid\mathbf{x})-\operatorname{VLB}_G(\mathbf{y})
+\Delta_{\mathrm{cond}}-\Delta_{\mathrm{base}}.
$$

Because the gap difference has no fixed sign, the estimate during training is neither a lower nor an upper bound on PMI, and even with both gaps at zero it gives model $G$'s likelihood ratio. Matching the data PMI further requires $G$ to match the true conditional and marginal distributions.

**The bootstrap chain rule (Eqs. (13–14)).** The main-text identity expands as

$$
\begin{aligned}
\operatorname{PMI}(\mathbf{x},y_{\le K})
&=\log\frac{p(y_1,\ldots,y_K\mid\mathbf{x})}{p(y_1,\ldots,y_K)}\\
&=\sum_{k=1}^{K}\log\frac{p(y_k\mid y_{<k},\mathbf{x})}{p(y_k\mid y_{<k})}.
\end{aligned}
$$

Subtracting the expression for the shorter prefix leaves only the newest token's conditional PMI, and summing over $k$ cancels all intermediate prefixes; empty-context PMI is zero. The paper's telescoping sum concerns unweighted PMI and does not address how the sum of timestep-weighted increments relates to the weighted PMI of the full view ($\operatorname{PMI}^{*}$, main Eq. (11)).

## A.5 Qualitative Results

[![Figure 6. Uncurated class-conditional ImageNet examples from the compute-comparison JiT-L system.](/assets/reviews/repr/figure6.webp)](/assets/reviews/repr/figure6.webp)

*Original Figure 6, p.20. Class-conditional samples from the JiT-L system used in the compute comparison, labeled uncurated by the authors.*

Sampling follows Latent Forcing's CFG-Interval+AutoGuidance setup exactly, applying classifier-free guidance (CFG) 3.0 to latent diffusion on $[0.06,1.0]$ and AutoGuidance weight 1.5. Appendix A.5 counts AutoGuidance training as 1.6% of total training FLOPs and includes it in the accounting. This protocol differs from the unguided FID-10K scaling evaluations in the main text.

[![Figure 7. Cosine similarity to a red center patch, comparing features with and without spatial equivariance.](/assets/reviews/repr/figure7.webp)](/assets/reviews/repr/figure7.webp)

*Original Figure 7, p.20. Cosine similarity between the red center patch and features at other positions, with and without the spatial-equivariance constraint.*

## A.6 Watermarking

[![Figure 8. Clean images and watermarked versions with repeated identifying information at patch scale.](/assets/reviews/repr/figure8.webp)](/assets/reviews/repr/figure8.webp)

*Original Figure 8, p.21. Compare clean and watermarked rows. The authors follow an earlier watermarking strategy exactly but shrink the watermark from $32\times32$ to $16\times16$ pixels, so that every individual patch provides enough signal for feature suppression.*

**Review interpretation.** A contrastive objective saturates as soon as the watermark distinguishes two views, but RePR's regression target is the predictive gain from all shared information, watermark included, so representing only the watermark leaves residual error. The gap in main Figure 2 is consistent with this structure.

## A.7 Raw Numbers for Scaling Plots

### ImageNet

Original Table 7 reports FID-10K of a JiT-B generator trained for 80 epochs. Iteration-2 epoch sums include 210 epochs of iteration 1 (200 plus 10 decay).

| Encoder | Pixel-prediction epochs → FID | Feature-prediction epochs → FID |
| --- | --- | --- |
| 20M | 110→41.61, 210→40.19, 410→37.01, 610→37.21 | 210+110→36.24, 210+210→34.89 |
| 44M | 110→40.93, 210→38.20, 410→36.07, 610→34.82 | 210+110→31.64, 210+210→29.48, 210+410→29.22 |
| 78M | 110→38.26, 210→35.72, 410→34.31, 610→34.00 | 210+110→29.81, 210+210→27.80, 210+410→27.36 |
| 173M | 110→37.22, 210→33.73, 410→31.27, 610→30.68 | 210+210→25.82, 210+410→24.73 |
| 306M | 110→36.14, 211→31.23 | not listed |

The 20M model worsens slightly from 410 to 610 epochs (37.01→37.21). The second 306M row is printed as 211, unlike the 210 of the other rows. The DINOv2 reference is 17.95 at $1.1685129\times10^{22}$ FLOPs.

### GPIC

Table 8 reports FD-DINOv2-10K of a MiniT2I generator trained for one GPIC epoch; the encoder epochs are ImageNet equivalents.

| Encoder | Pixel-prediction epochs → FD | Feature-prediction epochs → FD |
| --- | --- | --- |
| 44M | 110→514, 210→479, 410→456 | 210+110→416, 210+210→393 |
| 78M | 110→498, 210→463, 410→443 | 210+110→384, 210+210→365 |
| 173M | 110→450, 210→423, 410→409 | 210+110→359, 210+210→354 |
| 306M | 110→451, 210→404, 410→381 | 210+110→347, 210+210→325 |

The MiniT2I reference is 546.3, and the WebSSL-DINO-1B reference is 277.1 at $9.78\times10^{21}$ FLOPs. [Section 6.2 of the main part](/en/reviews/repr-latent-tokenization/) covers the 7.77× and 7.96× predictions from power-law fits to these two tables and how far the fits are extended. [A.7](https://arxiv.org/pdf/2610.20533v1#page=22)

### Review supplement: defining FID and FD

Using review notation, let $h$ be a feature extractor, $\mu_r,\mu_g$ real/generated feature means, and $C_r,C_g$ covariances. The standard Fréchet expression ([Heusel et al., Eq. (6), p.6](https://arxiv.org/pdf/1706.08500v6#page=6)) is

$$
D_F(h)=\|\mu_r-\mu_g\|_2^2+
\operatorname{Tr}\!\left(C_r+C_g-
2(C_r^{1/2}C_gC_r^{1/2})^{1/2}\right).
$$

FID uses Inception features and the paper's GPIC FD uses DINOv2 features, so the two scales differ. A value of zero means matching feature means and covariances, and finite-sample estimation bias means 10K and 50K values are read separately.

<span id="review-interpretation-and-next-verification" aria-hidden="true"></span>

## Review interpretation and open questions

**Review interpretation.** The appendix's strongest empirical evidence is the iteration-2 rows of Tables 7–8. For every encoder size with an iteration-2 entry, 420 total epochs of feature prediction give lower FID or FD than 410 epochs of pixel prediction (an epoch-based comparison; main Figure 3 makes the FLOP-based one). On the implementation side, noise sharing keeps integration at 1.3% of training FLOPs, and A.3 separately lists the checkpoints and FLOPs behind Figure 4, Table 3, and Table 2, which makes the boundaries of the combined-cost comparison clear.

The limitation is the distance between theory and measurement. The theory appendix is a single reconstruction argument that assumes an optimal factorization, and its empirical counterpart, Table 4's PSNR, comes from an RAE decoder trained separately for each encoder, so it reflects decoder performance as well as the encoder's information. Tables 7–8 also give a single value per setting.

Open questions are how closely the practical objective, which regresses timestep-weighted increments, approximates $\operatorname{PMI}^{*}$ of the full view, and whether gains continue from iteration 3 onward. Table 7 has no iteration-3 entries, and the only iteration-3 value under the same FID-10K evaluation is Table 1(e)'s 23.81.

## Sources and reading scope

- Paper: *Generative Representation Learning for Scalable Latent Tokenization* · Alan Baade et al. · first public date 2026-10-14 · [reviewed version](https://arxiv.org/abs/2610.20533v1) · reviewed 2026-10-23.
- Scope read: Version 1, all of A.1–A.7, plus intermediate probability derivations connected to sections 3–4. Complete original texts of the entire bibliography are outside scope.
- External sources consulted: [CPC Eq. (5)](https://arxiv.org/pdf/1807.03748v2#page=4), [VDM Eqs. (53–54)](https://arxiv.org/pdf/2107.00630v6#page=18), [Heusel et al. Eq. (6)](https://arxiv.org/pdf/1706.08500v6#page=6).
- Figure sources: The thumbnail (Figure 1) and body Figures 5–8 are author assets from v1 (CC BY 4.0), with author-margin trimming, resolution adjustment, and WebP conversion; photographic examples use quality 92. Content, layout, and original fonts are retained. [Paper](https://arxiv.org/abs/2610.20533v1).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
