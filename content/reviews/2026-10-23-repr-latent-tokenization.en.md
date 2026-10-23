---
title: "[1/2] RePR: learning representations that make generation easier"
description: "Follow Sections 1–7 through predictive information differences, patch features, watermark tests, and combined training costs."
date: "2026-10-23"
publishedAt: "2026-10-23T15:12:16+09:00"
updatedAt: "2026-10-23T15:26:27+00:00"
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

Relative Prediction Representations (RePR) learns patch features whose inner-product differences match the predictive gain from added context. At a total training cost of $3.49\times10^{20}$ FLOPs including encoder training, RePR-78M+JiT-L reaches a guided ImageNet 50K Fréchet Inception Distance (FID) of 2.18, below the 2.36 of pixel JiT-L at $3.57\times10^{20}$ FLOPs. [Table 3](https://arxiv.org/pdf/2610.20533v1#page=9) **Review interpretation.** The combined-compute advantage is established for two encoder sizes (RePR-44M/78M) over the roughly $0.5$–$3.6\times10^{20}$ FLOPs covered by Figure 4 and Table 3; the abstract’s seven- to eight-fold savings extend a trend whose best measured iteration-2 point (ImageNet FID-10K 24.73) has not yet reached the DINOv2 reference of 17.95.

[\[2/2\] The in-depth part](/en/reviews/repr-latent-tokenization-technical/) covers the pseudocode, probability derivations, and raw numbers of Appendices A.1–A.7. Here a tokenizer is an encoder producing continuous patch features, without a discrete codebook.

## 1. Introduction: what representations does generation need?

The authors start from the observation that representation learning decides which information to keep. Designs such as augmentation invariance preserve semantic content and discard other latent variables, yet encoders built this way, such as DINOv2, now serve as diffusion tokenizers in methods like RAE. These encoders were not designed to make generation easier, and they are trained with orders of magnitude more data and compute than the generative models built on them. The paper therefore asks what makes a representation good for generation, and whether learning one is more compute-efficient than modeling pixels directly.

The goal is **tokenizer encoder scaling laws**: giving the encoder more compute and data should make generative models trained on its features converge faster, and the compute-optimal split between encoder and generator should give the encoder a nonzero share. The authors note that earlier tokenizer scaling studies couple decoders, which improve with scale, with encoders that have been reported to get worse as they grow.

The authors give three reasons current recipes fail to yield scalable encoders. First, **feature suppression**: contrastive learning and LeJEPA objectives can be saturated by a few bits. ImageNet probing needs only a few semantic bits, but generation must sample every latent variable in the data. Second, missing **causal theoretical grounding**: contrastive methods optimize a CLS token, whereas RAE tokenizes with emergent internal patch features that become unstable at scale. Third, **architectural scalability**: classic tokenizers such as VAEs and VQ-VAEs bottleneck gradient flow through the reparameterization trick or straight-through estimation and fail to learn semantics, which drove the shift to representation-learning tokenizers such as DINOv2.

The remedy moves representation learning from discriminative toward generative objectives. Starting from the observation that word embeddings and contrastive learning implicitly factorize the data's pointwise mutual information (PMI) matrix, RePR estimates a weighted PMI with a generative model and has the representation factorize it explicitly. This needs no negative samples, resists feature suppression, and supervises patch features directly. Experiments on ImageNet and the text-to-image dataset GPIC train generators with Latent Forcing to isolate gains to encoder diffusability, meaning how easily a diffusion model learns the feature distribution, and the authors state that they use no decoder, perceptual loss, adversarial loss, or external model. [§1, pp.1–2](https://arxiv.org/pdf/2610.20533v1#page=1)

## 2. Related Work: what RePR changes

**Representation Learning.** Contrastive learning pulls augmented views of a scene together and pushes different scenes apart, leading to the DINO v1–v3 series. The authors argue that strongly conditioned views make hard negatives, which keep the objective from becoming trivial, difficult to find; the result is feature suppression, where representations probe well for object class but lose scene structure. They also review hand-crafted augmentations designed against this, and predictive objectives: pixel prediction in masked autoencoders (MAE) and latent prediction in BYOL, Data2Vec, JEPA, and MAGE.

**Pointwise Mutual Information and Representations.** Levy and Goldberg first connected PMI factorization to representation learning for Word2Vec, and the link was later extended to contrastive learning, including Contrastive Predictive Coding (CPC). The Platonic Representation Hypothesis proposes that large networks converge to similarly structured representations, with PMI factorization as a candidate, but recent work finds that large-scale representations fail to match that structure. RePR therefore supervises representations to match PMI directly rather than expecting the structure to emerge at scale.

**Pixel and Latent Space Generative Modeling.** Diffusion Transformers split into pixel diffusion such as JiT and latent diffusion such as RAE and REPA. RAE and REPA reach state-of-the-art ImageNet FID but do not control for representation pretraining: the DINOv2-B they use is distilled from a 1.1-billion-parameter model trained on over 100 times more data than ImageNet. The authors conclude that the FLOP efficiency of representation encoding is not yet understood. [Original §2, pp.2–3](https://arxiv.org/pdf/2610.20533v1#page=2)

### Review supplement: roles of the compared models

| Approach | Basic design | Role in this paper |
| --- | --- | --- |
| [LeJEPA](https://arxiv.org/abs/2511.08544v3) | Jointly learn view predictability and a Gaussian representation distribution | Comparison method in the watermark experiment (§6.1) |
| [DINOv2](https://arxiv.org/abs/2304.07193v2) | Teacher–student learning with image and patch objectives | Representative encoder reused for generation and scaling reference (§6.2). The watermark experiment uses DINOv3-B/16. |
| [RAE](https://arxiv.org/abs/2510.11690v1) | Train a decoder on a frozen representation encoder, then generate in its latent space | Route for the GAN+LPIPS transfer experiment (§6.5) |
| [JiT](https://arxiv.org/abs/2511.13720v2) | Pixel diffusion Transformers predicting clean images | Main tokenizer-free baseline (§6.3) |
| [Latent Forcing](https://arxiv.org/abs/2602.11401v1) | Jointly generate latents and pixels with separate time variables | Evaluation device that generates features, then pixels, to compare encoders without a lossy decoder (§§6.2–6.3) |

## 3. Background: PMI measures a relationship between observations

PMI measures how much more or less often two particular observations co-occur than independence predicts. The paper's bold symbols $\mathbf{x},\mathbf{y}$ denote observed views; $R_\theta$ is the representation model.

$$
\operatorname{PMI}(\mathbf{x},\mathbf{y})
=\log\frac{p(\mathbf{y}\mid\mathbf{x})}{p(\mathbf{y})}
=\log\frac{p(\mathbf{x},\mathbf{y})}{p(\mathbf{x})p(\mathbf{y})}.
$$

This is original Eq. (1). As a review example, an event with probability 0.01 that becomes probability 0.1 after observing another view has PMI $\log 10$. Positive, zero, and negative values indicate more frequent, independence-level, and less frequent co-occurrence; averaging PMI gives mutual information.

The paper cites the CPC authors' result that contrastive embedding inner products converge to a PMI estimate up to an unidentifiable offset (Eq. (2)):

$$
\langle R_\theta(\mathbf{x}),R_\theta(\mathbf{y})\rangle
\approx\operatorname{PMI}(\mathbf{x},\mathbf{y})+a(\mathbf{x})
$$

The offset $a(\mathbf{x})$ cannot be distinguished by a softmax over candidates. Levy and Goldberg likewise showed that word–context embedding inner products form a low-rank weighted factorization of the PMI matrix. The supplementary derivations under [A.4 in Part 2](/en/reviews/repr-latent-tokenization-technical/) derive this optimal score.

RePR starts from this connection, **estimating PMI with a generative model and regressing the representation inner product onto it**. If a generative model $G_\theta$ provides log probabilities, Eq. (3) uses $\log P_{G_\theta}(\mathbf{y}\mid\mathbf{x})-\log P_{G_\theta}(\mathbf{y})$ as the estimate. How closely those model probabilities match data probabilities depends on the conditions in the next section. [§3, p.3](https://arxiv.org/pdf/2610.20533v1#page=3)

## 4. Method: transferring relative predictive advantage into features

The initial proposal minimizes squared error between an embedding inner product and estimated PMI. Original Eqs. (4–5) have the following core form:

$$
\mathcal{L}_{\mathrm{PMI}}
=\left(\langle R_\theta(\mathbf{x}),R_\theta(\mathbf{y})\rangle
-\operatorname{stopgrad}\!\left[
\log P_{G_\theta}(\mathbf{y}\mid\mathbf{x})-\log P_{G_\theta}(\mathbf{y})
\right]\right)^2.
$$

Stop-gradient prevents the regression term from reducing error by moving its generative target, and the predictor trains on a separate generative loss. The authors choose this regression over contrastive InfoNCE because InfoNCE depends on negative sampling, which does not scale to strongly conditioned settings. Two obstacles remain: image PMI is dominated by high-frequency bits rather than semantics (§§4.1–4.2), and likelihood estimation on large inputs is expensive or intractable (§4.3). Section 4.4 assembles the full objective.

### 4.1 PMI Estimation from Generative Models

A discrete-probability model exposes log probabilities directly. The paper assumes all data are discrete (24-bit pixels and finite-precision features). For diffusion it uses a variational lower bound (VLB), which gives exact probabilities on discrete data; original Eq. (6) decomposes negative log likelihood into prior, reconstruction, and diffusion terms. The noise path is

$$
\mathbf{z}_t=(1-t)\mathbf{x}+t\boldsymbol{\epsilon},
\qquad\boldsymbol{\epsilon}\sim\mathcal{N}(0,I).
$$

In this paper's equations, **$t=0$ means data and $t=1$ means pure noise**. The prior term vanishes at $t=1$ and, because the data are discrete, the reconstruction term vanishes at $t=0$, leaving only the diffusion loss. With an optimal denoiser the bound is tight, and Eqs. (7–8) obtain PMI by integrating the reduction in denoising error from the extra view:

$$
\operatorname{PMI}(\mathbf{x},\mathbf{y})
=\mathbb{E}_{t,\boldsymbol{\epsilon}}\!\left[
\frac{1-t}{t^3}
\left(
\|\mathbf{y}-\mathbb{E}(\mathbf{y}\mid\mathbf{y}_t,t)\|_2^2
-\|\mathbf{y}-\mathbb{E}(\mathbf{y}\mid\mathbf{x},\mathbf{y}_t,t)\|_2^2
\right)\right].
$$

Both predictions use the same noisy target; if the context makes prediction easier, it produces positive supervision. The outer expectation is a Monte Carlo average and the inner expectations come from a learned denoiser. The authors note that the VLB equals $\log p$ only for an optimal denoiser, but that the approximation approaches the true PMI as prediction improves. [Part 2](/en/reviews/repr-latent-tokenization-technical/) derives the $(1-t)/t^3$ weight and discusses the approximation gaps. [§4.1, p.4](https://arxiv.org/pdf/2610.20533v1#page=4)

### 4.2 Timestep Weighting

Just as practical diffusion models optimize a timestep-weighted objective rather than the VLB itself, the authors weight PMI. Likelihood counts every bit equally, whereas information at lower signal-to-noise ratios (SNRs) often matters more semantically and perceptually. The PMI of a target corrupted to noise level $s$, $\mathbf{y}_s$, measures only dependencies that survive stronger corruption as $s$ grows; weighting it by $q(s)$ defines $\operatorname{PMI}^{*}$:

$$
\operatorname{PMI}^{*}(\mathbf{x},\mathbf{y})
=\mathbb{E}_{s,\boldsymbol{\epsilon}}
\left[q(s)\operatorname{PMI}(\mathbf{x},\mathbf{y}_s)\right].
$$

Because the corruption at $s$ and the diffusion time $t$ are both Gaussian, they combine into one effective timestep weighting, implemented through timestep sampling $f(t)$ and error weighting $g(t)$. The authors describe this as an inductive bias about which SNRs determine similarity. The practical setting uses $g(t)=(1-t)/\max(0.33,t)^2$; using the original $(1-t)/t^3$ directly produces NaN in Table 1(f). [§4.2, pp.4–5](https://arxiv.org/pdf/2610.20533v1#page=4)

### 4.3 Bootstrapped PMI Estimation

Estimating the likelihood of an entire large view would take too many Monte Carlo steps, so RePR instead supervises the information gained by adding one small token to an existing prefix. The chain-rule identity in Eq. (13) is

$$
\operatorname{PMI}(\mathbf{x},y_{\le k})
-\operatorname{PMI}(\mathbf{x},y_{<k})
=\log\frac{p(y_k\mid y_{<k},\mathbf{x})}{p(y_k\mid y_{<k})}.
$$

The right side measures the advantage of knowing $\mathbf{x}$ when predicting the new token given the existing prefix. Summing the increments telescopes to the full PMI, which is why training on small targets can supervise relationships between large contexts. [§4.3, p.5](https://arxiv.org/pdf/2610.20533v1#page=5)

### 4.4 Representation Objective

For contexts $\mathbf{x},\mathbf{y}$ and added information $\mathbf{z}$, Eq. (15) takes the form

$$
\mathcal{L}_{\mathrm{RePR}}
=\left(
\left\langle R_\theta(\mathbf{x}),
R_\theta(\mathbf{y},\mathbf{z})-R_\theta(\mathbf{y})\right\rangle
-\operatorname{PMI}^{*}(\mathbf{x},\mathbf{z}\mid\mathbf{y})
\right)^2.
$$

The target $\operatorname{PMI}^{*}$ is the $g(t)$-weighted Monte Carlo average of the difference between denoising $\mathbf{z}$ given $\mathbf{y}$ and given both contexts, entirely under stop-gradient.

The loss specifies how much the representation should change after observing the additional patch and how that change relates to another view. Because weighted PMI is asymmetric and PMI is not generally positive semidefinite (PSD), the implementation adds a learned embedding at the position of $\mathbf{z}$ to make the inner product asymmetric (Appendix A.3, [Part 2](/en/reviews/repr-latent-tokenization-technical/)).

## 5. Architecture: two paths explain the same prediction advantage

[![Figure 1. The representation-path inner-product increment matches an information difference from the prediction path; relative-position queries produce spatial outputs.](/assets/reviews/repr/figure1.webp)](/assets/reviews/repr/figure1.webp)

*Original Figure 1, p.6. The top path performs feature regression, the bottom compares two conditional predictions of one target patch, and the right shows relative-position queries. Quoted “log P” labels denote weighted estimates.*

### 5.1 Prediction Model

The predictor uses MAR: an MAE whose patch-regression head is replaced by a small diffusion multilayer perceptron (MLP). $E_\theta$ encodes visible patches, $D_\theta$ builds context with mask tokens, and $M_\theta$ denoises a patch. In Eq. (16), $z_p$ is the **decoder context at a patch location**, not the incremental information $\mathbf{z}$ of Section 4:

$$
\mathcal{L}_{\mathrm{MAR}}
=w(t)\|M_\theta(z_p,p_t,t)-p\|_2^2.
$$

MAR is chosen because the expensive encoder/decoder computation is shared once, after which the small diffusion head can run many times at different noise levels and be averaged.

### 5.2 Representing Patch Features

The authors distinguish features directly supervised for structure, such as the CLS token of contrastive learners, DINO, or LeJEPA, from unregularized intermediate latents, such as DINO's non-CLS tokens or the latents of an unregularized autoencoder. Without explicit supervision the latter can diverge or collapse. DINOv2 showed outlier patch features that motivated register tokens, and in DINOv3 feature quality degraded again, requiring an extra distillation stage toward a checkpoint taken before the degradation. Arguing that such instability may become harder to control at scale, the authors supervise patch features directly. **Review interpretation.** DINOv2 includes iBOT's masked patch-token objective, so its non-CLS tokens do receive a training signal ([DINOv2 §4, pp.5–6](https://arxiv.org/pdf/2304.07193v2#page=5)); RePR's distinction lies in fitting the geometry of patch-feature inner products directly to an information target.

Features form an $H\times W\times D$ spatial grid rather than a single classification token. $H,W$ index the output patch grid and $D$ the feature dimension. Inner products compare corresponding spatial positions. Absolute-position cues are withheld from the encoder; queries indexed by offsets from the visible input's center retrieve output features through cross-attention. This induces spatial equivariance: translating the image should translate its features. Appendix A.3 covers RoPE and queries padded beyond the image, and Figure 7 in A.5 compares feature similarities with and without this constraint.

### 5.3 Masking Strategy

An image supplies potentially overlapping contexts $A,B$ and a target patch $p$ absent from both. $AB$ is their combined context; $Bp$ adds the target to $B$. The predictor compares denoising $p$ from $B$ versus $AB$. The representation path matches $\langle R_\theta(A),R_\theta(Bp)-R_\theta(B)\rangle$.

Mask ratios are sampled from $U[M_A,100\%]$ and $U[M_B,100\%]$ rather than fixed, so that training covers the range of context sizes needed by the telescoping sum in §4.3. Default minimum ratios are 75% and 93.75%, leaving at most 64 and 16 visible patches at $256\times256$ resolution with $16\times16$ patches. For efficiency, $E(Bp)$ is reused across several views $A_i$, and one decoder pass predicts several target patches $p_j$ at only extra integration cost; $A_{\mathrm{Repeat}}=8$ and $p_{\mathrm{Repeat}}=16$ set these reuse counts.

### 5.4 Iterative Feature Prediction

Repeating the whole training procedure as a further iteration, with the predicted incremental patches replaced by features from the previous iteration, yields additional gains. The authors relate this to how feature-space prediction (HuBERT, iBOT, data2vec) improved on pixel prediction such as MAE. Iteration 1 predicts pixels; later iterations embed the whole image with the previous model $R_{\theta_{\mathrm{old}}}$ and use the patch feature at the same location as the target. The paper argues that because a PMI factorization can reconstruct the input at optimality (A.4), unweighted PMI satisfies $\operatorname{PMI}(\mathbf{x},R_\theta(\mathbf{y}))=\operatorname{PMI}(\mathbf{x},\mathbf{y})$, and that timestep weighting and bootstrapping then let feature-space prediction control the learning signal further. [A.4 in Part 2](/en/reviews/repr-latent-tokenization-technical/) examines the conditions for this identity. Sharing the $R_{\theta_{\mathrm{old}}}$ computation across $A_{\mathrm{Repeat}}$ views keeps it cheap.

### 5.5 Joint Optimization

Prediction and representation losses train jointly. Finding that the representation model converges relatively easily, the authors speed up training by applying the representation loss once every four steps. The paths share $E_\theta$, so they are not entirely independent networks; the regression target is under stop-gradient and the MAR path trains on the prediction loss. [§5, pp.5–7; A.1/A.3](https://arxiv.org/pdf/2610.20533v1#page=6)

## 6. Experiments: testing information retention and cost

### 6.1 Feature Suppression

[![Figure 2. Clean and watermarked ImageNet validation top-1 versus effective training epochs accounting for views.](/assets/reviews/repr/figure2.webp)](/assets/reviews/repr/figure2.webp)

*Original Figure 2, p.7. The x-axis counts effective epochs accounting for images and views; the y-axis is ImageNet validation top-1 accuracy (%). Dashed means clean, solid means watermarked. Blue is DINOv3-B/16, yellow LeJEPA-S/8, and black RePR-78M.*

The watermark experiment uses ImageNet with a watermark in every patch that uniquely identifies the image. On such data, recognizing the watermark alone attains a perfect contrastive loss. The authors train DINOv3, LeJEPA, and RePR on both standard and watermarked ImageNet; on watermarked data DINOv3 and LeJEPA collapse to representing only the watermark, whereas RePR retains substantial class-relevant information. They summarize that RePR is robust while sharing the same PMI-factorization guarantees as contrastive learners. **Review interpretation.** In the plot RePR also falls below its clean curve, and DINOv3 is higher under clean training, so what the experiment shows is robustness to the watermark shortcut rather than superior classification. [A.6 in Part 2](/en/reviews/repr-latent-tokenization-technical/) shows the actual watermark.

### 6.2 Scaling Laws

[![Figure 3. Encoder-training FLOPs versus ImageNet FID-10K and GPIC FD-DINOv2-10K, with iterations and extrapolated trends.](/assets/reviews/repr/figure3.webp)](/assets/reviews/repr/figure3.webp)

*Original Figure 3, p.8. Encoder-training FLOPs use a logarithmic x-axis. Left: FID-10K after 80 epochs of ImageNet JiT-B training. Right: FD-DINOv2-10K after one GPIC epoch of MiniT2I. Lower is better. Circles denote pixel prediction (iteration 1), diamonds feature prediction (iteration 2), and colors encoder size. Solid lines fit the observed points and the dashed or dotted lines beyond them are extensions; stars mark the reference encoders (DINOv2, WebSSL-DINO-1B), and the red dashed line on the right is the MiniT2I baseline.*

Comparing encoder diffusability requires holding generator size and training duration fixed. The authors use Latent Forcing to generate features first and then pixels conditioned on them. Because the final output is generated in pixel space, the paper calls this generative process lossless; it avoids the reconstruction–generation trade-off of lossy latent tokenizers and attributes gains to diffusability rather than compression. The same setup permits FLOP- and parameter-matched comparisons with pixel diffusion such as JiT.

Widths vary from 256 to 1024, giving 20M–306M inference encoders at fixed depth. The authors report power-law scaling over roughly two orders of magnitude of training compute on both datasets, and iteration 2, which predicts the features of a same-size pixel model trained for 200 epochs, improves substantially on iteration 1. The width-1024 ImageNet model begins overfitting after 200 epochs, while GPIC continues improving over the larger budgets examined.

The abstract's 7× and 8× come from extending these fits by at most the width of the measured log-FLOP range. Appendix A.7 reports:

| Evaluation | Reference encoder cost and quality | Extrapolated RePR cost | Compute ratio |
| --- | --- | --- | --- |
| ImageNet, FID-10K | DINOv2: $1.1685129\times10^{22}$ FLOPs, 17.95 | $1.504\times10^{21}$ FLOPs | about 1/7.77 |
| GPIC, FD-DINOv2-10K | WebSSL-DINO-1B: $9.78\times10^{21}$ FLOPs, 277.1 | $1.229\times10^{21}$ FLOPs | about 1/7.96 |

The best measured iteration-2 points are FID-10K 24.73 for RePR-173M on ImageNet and FD 325 for RePR-306M on GPIC, still short of the references 17.95 and 277.1. The paper does not report the fit coefficients or their uncertainty. FD-DINOv2 is the same Fréchet distance computed in DINOv2 feature space, so its scale differs from FID ([A.7 in Part 2](/en/reviews/repr-latent-tokenization-technical/)).

### 6.3 Compute Optimal Generation

[![Figure 4. Conditional and unconditional unguided FID-10K against combined encoder and generative-model training FLOPs.](/assets/reviews/repr/figure4.webp)](/assets/reviews/repr/figure4.webp)

*Original Figure 4, p.9. Conditional generation is above, unconditional below. The x-axis includes encoder pretraining. Black is pixel JiT-L; blue/yellow add RePR-44M/78M. Both y-axes show unguided FID-10K but use different ranges.*

At $1.19\times10^{20}$ FLOPs, equal to 200 epochs of JiT-L training, the authors report unguided FID-10K improvements of 2.61 for conditional and 4.18 for unconditional generation. This comparison uses no perceptual loss, GAN, or external image decoder. Table 3 is a separate guided 50K evaluation:

| Setting | Total training FLOPs ↓ | FID-50K ↓ | IS ↑ |
| --- | --- | --- | --- |
| Pixel, JiT-L 600ep | $3.57\times10^{20}$ | 2.36 | 298.5 |
| RePR-78M + Pixel, JiT-L 440ep | $3.49\times10^{20}$ | **2.18** | **338.4** |

The RePR budget is about 2.2% smaller, and it includes RePR encoder training and the AutoGuidance training used for guidance (A.3, A.5). Inception Score (IS) combines classifier confidence with class diversity, with higher values preferred ([the IS original, §4](https://arxiv.org/pdf/1606.03498v1#page=4)). FLOPs use a $6ND$ approximation (6 × parameters × processed data) excluding self-attention. Combining this result with the scaling laws, the authors conclude that representation learning is an additional scaling axis that improves generation more efficiently than scaling the generative model alone. [§6.3; A.3/A.5](https://arxiv.org/pdf/2610.20533v1#page=8)

### 6.4 Ablation Studies

Table 1 contains six groups testing different design choices. The caption says ImageNet top-1 serves as a proxy because generative training is expensive, but only (a)–(d) report top-1 accuracy (%); (e) and (f) report FID-10K.

| Original group | Settings and results | Interpretation |
| --- | --- | --- |
| (a), 78M, 100ep | $M_A$: 62.5%→58.62, 75%→58.75, 87.5%→56.21 top-1 | Too little context in $A$ hurts. |
| (b), 78M, 100ep | $M_B$: 75%→56.78, 93.75%→58.75, 98.5%→58.55 top-1 | Short $B$ context helps, but 98.5% is slightly lower. |
| (c), 78M, 100ep | uniform 57.63, block 58.75 top-1 | Block masking is better. |
| (d), 78M, 200ep | representation loss weight 0.005→63.1, 0.01→63.6, 0.025→63.0, 0.1→49.5, 0.4→43.9 top-1 | 0.01 is slightly above the default 0.005; weights of 0.1 and above hurt badly. |
| (e), 173M, nominal 600ep allocation | pixels only $\{600\}$→30.68, iteration 2 $\{200,400\}$→24.73, iteration 3 $\{200,200,200\}$→23.81 FID-10K | Reallocating total epochs to feature prediction lowers FID. |
| (f), 78M, 200ep | original weight $(1-t)/t^3$: NaN; denominator floor 0.05: 42.15; logit-normal $\mu=0.6$/1.2 with floor 0.33: 35.09/35.72 FID-10K | Stability and timestep bias matter. $\mu=0.6$ beats the default $\mu=1.2$. |

The authors read the masking settings as useful but not critical, and the optimum at a very small representation-loss weight as a sign that the difficulty lies in generation rather than in learning the factorization. Additional iterations in (e) improve results substantially, which they liken to the gains from pixel-predictive MAE to latent-predictive Data2Vec.

**Review interpretation.** The epoch labels disagree with the appendix. The (e) values 30.68 and 24.73 equal Table 7's 173M entries at 610 and $210+410$ epochs, and (f)'s default 35.72 equals Table 7's 78M entry at 210 epochs. So (e) and (f) use Table 7's generative evaluation (JiT-B, 80 epochs, FID-10K), and Table 1's $\{600\}$, $\{200,400\}$, and 200ep are nominal counts that exclude the 10-epoch learning-rate decay attached to each checkpoint (A.3). Read this way, (e)'s actual totals are 610 and 620 epochs, and 630 for iteration 3 under the same rule, so the caption's "holding total epochs constant" holds nominally. The gap is within about 3% and does not change (e)'s conclusion. [§6.4; Table 1, p.9](https://arxiv.org/pdf/2610.20533v1#page=9)

### 6.5 GAN+LPIPS Decoding

Table 2 tests RePR in the RAE route. With ImageNet-only encoders and 20 epochs of DiT-XL, MoCo-v3-B scores 13.84, I-JEPA-H 12.46, and RePR-173M iteration 3 **11.13 FID-50K**. The MoCo-v3 and I-JEPA values are quoted from RAEv2 (caption), and RePR follows RAEv2's ablation setup. The authors cite this as evidence that RePR is a competitive tokenizer even at small scale.

The decoder uses Learned Perceptual Image Patch Similarity (LPIPS) and generative adversarial network (GAN) losses. The authors note that this route relies on two external models, and that the DINOv1/8-based GAN pretraining alone costs $4.7\times10^{20}$ FLOPs, more than any model they train.

## 7. Conclusion: what can we take from the paper?

The authors summarize that RePR shares the theoretical guarantees of contrastive learning without negative samples while remaining robust to feature suppression where prior methods collapse. Citing empirical scaling laws for diffusability and the finding that RePR-based generators beat direct diffusion on raw data at matched training FLOPs, they conclude that representation learning with strong theoretical foundations can coexist with, and improve, the scaling of generative modeling. [§7, p.10](https://arxiv.org/pdf/2610.20533v1#page=10)

## Review interpretation and open questions

**Review interpretation.** The paper's most convincing design choice is to include encoder-training FLOPs in the generation comparison and to use Latent Forcing to remove a separate decoder, perceptual loss, and GAN, attributing the difference to encoder diffusability. This shifts the question from which off-the-shelf encoder to use to how much compute to assign to encoder training, and in Figure 4 both RePR curves lie below pixel JiT-L at every plotted budget.

The precision of that evidence is limited, however. Table 3's 0.18 FID-50K gap and Figure 4's curves are reported without error bars, and because FLOPs are an approximation excluding self-attention, GPU time need not fall in the same proportion. GPIC generation also feeds a FLAN-T5-large text encoder as conditioning, so the introduction's "no external models" applies exactly to the ImageNet experiments.

The abstract says RePR avoids both dimensional collapse and feature suppression where DINO and LeJEPA fail, but the evidence is the class-probing curve on watermarked ImageNet (Figure 2). Feature suppression includes cases where many dimensions are active but carry only partial information such as a watermark, so an experiment that does not measure effective dimensionality supports the feature-suppression claim, while dimensional collapse is not directly measured. Compared with the distributional non-degeneracy discussed in the [LeJEPA review](/en/reviews/lejepa-complete/), the question here is which information survives rather than how many dimensions are used.

Open questions are whether the power law continues to reference quality even though the width-1024 ImageNet model overfits after 200 epochs, and whether Table 3's advantage holds at budgets above $3.6\times10^{20}$ FLOPs and with larger generators.

## Sources and reading scope

- Paper: *Generative Representation Learning for Scalable Latent Tokenization* · Alan Baade et al. · first public date 2026-10-14 · [reviewed version](https://arxiv.org/abs/2610.20533v1) · reviewed 2026-10-23.
- Scope read: Version 1, sections 1–7. Algorithms, detailed derivations, and raw numbers in A.1–A.7 are covered in part 2.
- External sources consulted: [CPC §2.3](https://arxiv.org/pdf/1807.03748v2#page=3), [Variational Diffusion Models E.1–E.3](https://arxiv.org/pdf/2107.00630v6#page=17), [DINOv2 §4](https://arxiv.org/pdf/2304.07193v2#page=5), [the IS original §4](https://arxiv.org/pdf/1606.03498v1#page=4).
- Figure sources: The thumbnail and body Figures 1–4 are author assets from v1 (CC BY 4.0), with author-margin trimming, resolution adjustment, and WebP conversion. Content, layout, and original fonts are retained. [Paper](https://arxiv.org/abs/2610.20533v1).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
