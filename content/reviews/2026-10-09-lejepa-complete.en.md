---
title: "[2/3] LeJEPA in full: from target distribution to theory, algorithm and experiments"
description: "A section-aligned account of the theory, SIGReg design, related work, and central experiments, including conditions and contrary results."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-23T15:26:17+00:00"
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

LeJEPA derives an isotropic-Gaussian target from downstream estimation analysis and implements it with Sketched Isotropic Gaussian Regularization (SIGReg) and view agreement. After 400 epochs of in-domain pretraining on Galaxy10, a ResNet-34 reaches 78.17% in full-label frozen linear evaluation, above the 71.38% of DINOv3 ViT-S/16 pretrained on natural images. [Section 6.3 and Table 3](https://arxiv.org/pdf/2511.08544v3#page=42) **Review interpretation.** The claim that small in-domain models beat large transfer models holds from 1-shot to full labels for the best LeJEPA architecture, but the frozen 1-shot gap is only 0.91 points (31.08 vs 30.17), and LeViT-128 stays below DINOv3 even with full labels, so the result depends on architecture.

This **[2/3] complete commentary** follows the paper’s section order; it continues from [\[1/3\] introduction](/en/reviews/lejepa-overview/), and intermediate proof calculations are in [\[3/3\] theorem and proof commentary](/en/reviews/lejepa-technical/).

The thumbnail’s objectives gather samples along different axes: within-image agreement on the lower left and across-image regularization within each view on the lower right. The Gaussian constraint acts after the projector head. [Enlarge](/assets/reviews/lejepa/lejepa-core.svg)

<span id="1-introduction-the-problem-and-the-authors-route" aria-hidden="true"></span>

## 1. Introduction

Self-supervised learning uses relationships between inputs instead of target labels. For images, this connects crops and augmentations of one source image. Minimizing embedding distances alone also permits constant outputs. Avoiding collapse while supporting different datasets and architectures calls for an explicit objective.

The authors argue that accumulating empirical mechanisms increases configuration costs, while anti-collapse theory alone does not select a favorable feature geometry. Section 2 defines the requirements of a Joint-Embedding Predictive Architecture (JEPA); Section 3 derives a distribution target through downstream prediction. Section 4 makes high-dimensional matching tractable; Section 5 assembles the objective. Section 6 evaluates stability, loss informativeness, domains, scale, and semantic structure. [Section 1, pp. 1–3](https://arxiv.org/pdf/2511.08544v3#page=1)

<span id="2-background-and-notations-definitions-and-design-requirements" aria-hidden="true"></span>

## 2. Background and Notations

### 2.1 Notations and Definitions

The input tensor has shape $(N,V,D)$: $N$ independent samples, $V$ views each, and input dimension $D$. The encoder $f_{\boldsymbol\theta}:\mathbb R^D\to\mathbb R^K$ produces $K$-dimensional embeddings. $\mathbf Z\in\mathbb R^{N\times K}$ collects embeddings for one view.

JEPA requires predictability between related embeddings and retention of information that distinguishes inputs. The paper regards a separate predictor as justified only when views carry asymmetric information, for example when predictions are conditioned on actions observed in robotics data. Its experiments are image representation learning with symmetric agreement between views of the same image. [Section 2.1, p. 4](https://arxiv.org/pdf/2511.08544v3#page=4)

### 2.2 The Need for Reliable Pretraining

A useful pretraining method should support small datasets and convolutional or hybrid architectures as well as large natural-image datasets and ViTs. Retuning interacting teacher updates, temperatures, or prototypes makes specialized-domain use difficult. This motivates Sections 6.1 and 6.3. [Section 2.2](https://arxiv.org/pdf/2511.08544v3#page=4)

### 2.3 The Need for Actionable Theory

Non-collapse is a necessary starting point but does not identify which representation is favorable downstream. The authors seek theory that selects an embedding law before specific labels are known, and state that Sections 3–5 derive the design from proved necessary conditions for optimality. [Section 2.3](https://arxiv.org/pdf/2511.08544v3#page=4)

## 3. Latent Euclidean — Embeddings Should Be Isotropic Gaussian

### 3.1 Linear Probing

Start with fixed embeddings and linear prediction of labels $\mathbf y$. The paper’s OLS notation includes ridge regularization:

$$
\hat\beta=\underset{\beta\in\mathbb R^K}{\arg\min}
\|\mathbf y-\mathbf Z\beta\|_2^2+\lambda\|\beta\|_2^2.
$$

$\mathbf Z_{\rm iso}$ and $\mathbf Z_{\rm aniso}$ have the same column span and total energy, with different covariance shapes. Appendices B.1–B.2 use fixed design, full rank, and centered homoscedastic noise.

**Lemma 1 — Anisotropy amplifies bias.** For $\lambda>0$, **there exists** a downstream task with greater coefficient-estimation bias under anisotropy. Choosing a true task coefficient along a small-eigenvalue direction increases ridge shrinkage.

**Lemma 2 — Anisotropy amplifies variance.** With $\lambda=0$, isotropy minimizes total coefficient variance. Writing $\lambda_k$ for the eigenvalues of the Gram matrix $\mathbf Z^\top\mathbf Z$,

$$
\operatorname{tr}\operatorname{Var}(\hat{\boldsymbol\beta}\mid\mathbf Z)
=\sigma^2\sum_{k=1}^K\frac1{\lambda_k}.
$$

At fixed energy, the reciprocal sum is minimized by equal eigenvalues. A review example gives 2 for $(1,1)$ versus about 5.56 for $(0.2,1.8)$. The ridge expansion, the Jensen step, and the difference between coefficient variance and prediction risk appear in the [technical version, Section 3.1](/en/reviews/lejepa-technical/#31-linear-probing-lemmas-12-and-appendices-b1b2). [Section 3.1; Appendices B.1–B.2, pp. 26–27](https://arxiv.org/pdf/2511.08544v3#page=26)

### 3.2 Nonlinear Probing

Many distributions have isotropic covariance. To motivate the Gaussian shape, the authors analyze radius-based kNN and Nadaraya–Watson kernel regression:

$$
\widehat{\mathbf y}(\mathbf q)=\frac1{|\mathcal N_{r_0}(\mathbf q)|}
\sum_{n\in\mathcal N_{r_0}(\mathbf q)}\mathbf y_n,
\qquad
\mathcal N_{r_0}(\mathbf q)=\{n:\|\mathbf z_n-\mathbf q\|\le r_0\}.
$$

$$
\widehat{\mathbf y}(\mathbf q)=
\frac{\sum_{n=1}^N K_h(\mathbf q-\mathbf z_n)\mathbf y_n}
{\sum_{n=1}^N K_h(\mathbf q-\mathbf z_n)}.
$$

$\mathbf q$ is the query embedding, $r_0$ the neighborhood radius, and $h$ the bandwidth. The paper reuses $K$ for both the kernel function and embedding dimension. Radius-based kNN averages the neighbors within radius $r_0$ rather than a fixed number of them.

**Theorem 1 — Isotropic Gaussian Optimality** expresses integrated squared bias (ISB) through Fisher information $J(p)$ and states that, under a scalar covariance constraint, the isotropic Gaussian is the unique minimizer of ISB:

$$
\begin{aligned}
\mathrm{ISB}_{k\text{-NN}}&=\frac{r_0^4}{(K+2)^2}\tau_g^2J(p)+O(r_0^4),\\
\mathrm{ISB}_{\rm kernel}&\le
\left(\frac{h^2\mu_2(K)}2\right)^2
\left(2B^2+8L^2J(p)\right)+o(h^4),\\
J(p)&=\int\|\nabla\log p(\mathbf z)\|_2^2p(\mathbf z)\,d\mathbf z.
\end{aligned}
$$

Here $p$ is the embedding density and $\eta$ the target function mapping embeddings to labels. $\tau_g^2$ is the scale of the isotropic task-gradient prior $\mathbb E[\nabla\eta\nabla\eta^\top]=\tau_g^2I$, and $\mu_2(K)$ is the kernel’s second moment, $\int\mathbf u\mathbf u^\top K(\mathbf u)\,d\mathbf u=\mu_2(K)I$. The density must be sufficiently smooth with finite Fisher information and covariance, and queries follow the training density. The kNN analysis assumes a small radius; the kernel analysis assumes small $h$, sufficient local samples, a symmetric normalized kernel, and bounds $\|\nabla\eta\|\le L$, $|\Delta\eta|\le B$. Here $B$ is a curvature bound, distinct from the batch notation in Section 5. [Appendix A, Lemmas 4–5, Theorems 7–8](https://arxiv.org/pdf/2511.08544v3#page=24)

At fixed covariance $\Sigma$, $J(p)\ge\operatorname{tr}(\Sigma^{-1})$, with equality for a Gaussian. A fixed total-variance budget then selects isotropic $\Sigma$. This is the mathematical core of the target choice. [Section 3.2; Appendix B.4](https://arxiv.org/pdf/2511.08544v3#page=28) **Review interpretation.** Because the kNN remainder $O(r_0^4)$ has the same order as the leading term and the kernel expression is an upper bound, the “unique minimizer” conclusion concerns the $J(p)$ term rather than exact ISB. The Taylor expansions and the Fisher-information bound are worked out in the [technical version, Section 3.2](/en/reviews/lejepa-technical/#32-nonlinear-probing-theorem-1-and-appendices-a-b3b7).

### 3.3 Geometric and Practical Insights

The authors illustrate estimated coefficients and decision boundaries under fixed total variance. Figure 3 repeats logistic regression on 100 samples for a two-class task, comparing isotropic and anisotropic decision boundaries. Appendix Figure 18 visualizes regularization and bias. [Section 3.3, Figures 3 and 18](https://arxiv.org/pdf/2511.08544v3#page=6)

## 4. SIGReg — Reliable Isotropic Gaussian Regularization in High-Dimension

### 4.1 Hypothesis Testing as a Judge

Instead of estimating the full high-dimensional density, compare one-dimensional laws. Let $P_{\boldsymbol\theta}$ be the embedding law and $Q=\mathcal N(0,I)$ the target. **Lemma 3 — Hyperspherical Cramér–Wold** equates equality of all unit-direction projection laws with equality of vector laws. Appendix B.8 proves this through characteristic functions and $\mathbf t=r\mathbf a$.

**Theorem 2 — Sufficiency of directional tests** concerns consistency of combined projection tests. Appendix B.9 requires vanishing directional mesh, a threshold calibrated for the global null, and power in a neighborhood of a separating direction. The test in this theorem takes the maximum over directional statistics, whereas SIGReg used for training is the directional average:

$$
\operatorname{SIGReg}(\{\mathbf z_n\}_{n=1}^N)
=\frac1{|\mathcal A|}\sum_{\mathbf a\in\mathcal A}
T(\{\mathbf a^\top\mathbf z_n\}_{n=1}^N).
$$

This is the directional average in Definition 2. Section 4.2 chooses the statistic $T$. [Section 4.1, Lemma 3, Theorem 2, Definition 2; Appendices B.8–B.9](https://arxiv.org/pdf/2511.08544v3#page=6)

![Figure 2: Input distribution, embedding space, and directional density comparisons.](/assets/reviews/lejepa/paper-figure-2.webp)

**Reading Figure 2.** Each colored arrow in the middle yields a projected density on the right. The horizontal coordinate is the projection; black is the Gaussian target. The gray hatching marks the error between each projected density and the target. [Enlarge](/assets/reviews/lejepa/paper-figure-2.webp) · [Figure 2, p. 3](https://arxiv.org/pdf/2511.08544v3#page=3). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

### 4.2 SIGReg: Sketching the Epps–Pulley Test Is Stable and Scalable

#### 4.2.1 Moments Are Unstable and Insufficient

**Theorem 3** constructs different distributions matching moments through any fixed finite order. Mean, variance, and covariance alone cannot establish a joint Gaussian law. Increasing the order also creates polynomial gradients that can grow on tail samples. The linear-algebra construction in Appendix B.11 is explained in the [technical version, Section 4.2](/en/reviews/lejepa-technical/#42-sigreg-sketching-the-eppspulley-test). [Section 4.2.1; Appendix B.11](https://arxiv.org/pdf/2511.08544v3#page=7)

![Figure 5: Gaussian coordinate marginals and identity covariance can coexist with an X-shaped joint law.](/assets/reviews/lejepa/paper-figure-5.webp)

**Reading Figure 5.** Coordinate densities are Gaussian and covariance is identity, while joint samples form an X. Projecting onto ten directions $\mathbf a_i$ reveals non-Gaussian densities along oblique directions, and the test statistics in the rightmost panel also respond strongly there (i:2, i:7). [Enlarge](/assets/reviews/lejepa/paper-figure-5.webp) · [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

#### 4.2.2 Cumulative Density Functions Are Impractical

A cumulative distribution function (CDF) gives the probability below a threshold. The paper argues that sorting breaks the parallelism of SGD because multi-GPU training needs synchronization, that sorting and order statistics are non-differentiable operations unsuitable for gradient learning without relaxations, and that sketching solutions add hyperparameters. [Section 4.2.2, p. 9](https://arxiv.org/pdf/2511.08544v3#page=9) **Review interpretation.** Sorted values are piecewise-linear functions of the inputs away from ties, so gradients exist; the practical burden lies more in synchronization and extra hyperparameters than in differentiability.

#### 4.2.3 Characteristic Functions Are Stable, Scalable and Identifiable

A characteristic function averages a complex exponential. The directional subscripts below map the main text’s $\hat\phi_X$, $\phi$ to the projection and Gaussian target. Its empirical version uses sample means; the Gaussian target is analytic:

$$
\hat\varphi_{\mathbf a}(t)=\frac1N\sum_{n=1}^N e^{it\mathbf a^\top\mathbf z_n},
\qquad \varphi_{\mathcal N}(t)=e^{-t^2/2}.
$$

$$
EP(\mathbf a)=N\int_{\mathbb R}
|\hat\varphi_{\mathbf a}(t)-\varphi_{\mathcal N}(t)|^2w(t)\,dt.
$$

The paper’s Epps–Pulley equation uses a Gaussian weight $w(t)$. **Theorem 4 — Stability of Epps–Pulley Test** uses $|e^{itz}|=1$, which bounds the CF and its derivatives independently of input magnitude, to state that the first and second derivatives of $EP$ with respect to projected samples are bounded at $O(1/N)$. [Section 4.2.3, Epps–Pulley equation, Theorem 4; Appendix B.12](https://arxiv.org/pdf/2511.08544v3#page=9) **Review interpretation.** Appendix B.12 derives this for the unscaled distance $EP/N$, so the gradient bound for $EP$ itself is $N$ times larger and only boundedness survives. The calculation is in the [technical version, Section 4.2.3](/en/reviews/lejepa-technical/#423-cf-deriving-theorem-4-and-checking-normalization).

**Reading Algorithm 1.** Normalize columns of a $K\times M$ Gaussian matrix to obtain unit directions. Multiply $\mathbf Z\mathbf A$ by frequencies and average complex exponentials. Synchronize the step seed across GPUs and all-reduce the CF mean. On 17 points in $[-5,5]$, weight the squared Gaussian-CF error with a Gaussian window, apply trapezoidal integration, then multiply by the global sample count $N$. SIGReg averages the directional statistics. [Algorithm 1, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10)

With $M$ directions and $n_t$ frequency points (a review shorthand), dense cost is $\mathcal O(NKM+NMn_t)$, linear in $N$ for fixed $K,M,n_t$. Table 6 times the SIGReg loss computation over 10 runs (mean ± std) on a V100-SXM2-16GB: $(N,M,n_t)=(512,512,16)$ gives $0.465236\pm0.011642$ ms; $(8192,512,16)$ gives $6.188304\pm0.007226$ ms. Embedding dimension is absent from the table. [Table 6, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

### 4.3 How SIGReg Beats the Curse of Dimensionality

The paper proceeds through **smoothness**, **SGD resampling**, and **synthetic validation**. Theorem 5’s bound decays as $M^{-2\alpha/(K-1)}$ in the number of directions $M$, where $\alpha$ is Sobolev regularity. The authors conclude that $O(K)$ directions suffice when $\alpha$ is large; a check that includes the dimension dependence of the constant $C(K,\alpha)$ is in the [technical version, Section 4.3](/en/reviews/lejepa-technical/#43-how-sigreg-beats-the-curse-of-dimensionality-theorem-5). They also report that resampling directions at every step makes the cumulative number of directions grow with training time, so that $M=16$ outperforms thousands of fixed directions. [Section 4.3, Theorem 5; Appendix B.10](https://arxiv.org/pdf/2511.08544v3#page=10)

![Figure 7: Final directional statistic by number of directions, fixed versus resampled.](/assets/reviews/lejepa/paper-figure-7.webp)

**Reading Figure 7.** Horizontal: directions $M$ per step. Vertical: expected directional statistic after the synthetic optimization, where lower means closer matching. Green resampled directions reach low values with fewer directions than blue fixed directions. [Enlarge](/assets/reviews/lejepa/paper-figure-7.webp) · [Figure 7, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

The synthetic experiment then **directly optimizes samples**, without an encoder, whose coordinates are mostly Gaussian but whose first two dimensions have an X structure, isolating the effect of the loss. Section 4.3 introduces this experiment (Figure 6) as a $D=512$, $M=16$ case, whereas Figure 6’s caption specifies $N=100$, $D=1024$, $M=10$, and the appendix’s additional figures contain no $D=512$ setting either. [Section 4.3, p. 11](https://arxiv.org/pdf/2511.08544v3#page=11) · [Figure 6, p. 9](https://arxiv.org/pdf/2511.08544v3#page=9)

## 5. LeJEPA — Stable and Scalable Implementation

### 5.1 LeJEPA: SIGReg + Prediction Loss

Create $V_g$ global and $V_l$ local views, with $V=V_g+V_l$, then process them through the encoder and projector. $\boldsymbol\mu_n$ averages global-view embeddings. The following describes Algorithm 2’s center-based objective:

$$
\begin{aligned}
\boldsymbol\mu_n&=\frac1{V_g}\sum_{v=1}^{V_g}\mathbf z_{n,v},\\
\mathcal L_{\rm pred}^{\rm code}&=\frac1{NVK}\sum_{n=1}^N\sum_{v=1}^V
\|\mathbf z_{n,v}-\boldsymbol\mu_n\|_2^2,\\
\mathcal L_{\rm SIGReg}&=\frac1V\sum_{v=1}^V\operatorname{SIGReg}(\{\mathbf z_{n,v}\}_{n=1}^N),\\
\mathcal L_{\rm LeJEPA}&=(1-\lambda)\mathcal L_{\rm pred}^{\rm code}+\lambda\mathcal L_{\rm SIGReg}.
\end{aligned}
$$

$\mathcal L_{\rm pred}^{\rm code}$ and $\mathcal L_{\rm SIGReg}$ are review abbreviations; $1/K$ exposes the code’s feature-coordinate mean. The total loss uses batch notation $B$ in the paper, corresponding to $N$ here. Algorithm 2’s caption tells non-ViT architectures such as ResNets to treat all views as global ($V_g=V$), in which case $\boldsymbol\mu_n$ averages all views. [Section 5.1, Algorithm 2](https://arxiv.org/pdf/2511.08544v3#page=11)

The paper states that the pairwise average loss in Eq. (5) equals the center loss in Eq. (6) (Appendix B.6). **Review interpretation.** That equality omits the variance among global views, so what the experiments optimize is the center loss of Eq. (6). The expansion and a counterexample are in the [technical version, Section 5.1](/en/reviews/lejepa-technical/#51-sigreg-prediction-loss-theorem-6-appendices-b6-and-b13).

The authors write that, with no prototypes, stop-gradients, or teacher–student networks, the loss has a single hyperparameter, $\lambda$. A projector head is still used, and evaluation uses the backbone features before it. Apart from model definitions, optimizers, and data loaders, the implementation takes a few dozen lines.

### 5.2 Relation to Prior Work

The paper notes that no earlier JEPA used slicing and distribution matching, while similar pipelines exist for generative models and optimal transport, citing sliced score matching and the sliced Wasserstein distance. Computing the Epps–Pulley integral exactly instead of by quadrature turns each slice loss into a kernel MMD, at quadratic cost. Using the statistic $T=\operatorname{mean}^2+(\operatorname{std}-1)^2$ recovers VICReg in the limit of many slices (Appendix B.14), and the authors warn, citing Theorem 3, that such moment statistics admit shortcut solutions. [Section 5.2; Appendix B.14, pp. 39–40](https://arxiv.org/pdf/2511.08544v3#page=12) The exact correspondence with the VICReg loss is discussed in the [technical version, Section 5.2](/en/reviews/lejepa-technical/#52-relation-to-prior-work-appendix-b14).

### Review supplement: comparing the original methods

| Primary source and reading scope | Central method | Relation to LeJEPA |
| --- | --- | --- |
| [VICReg, Sections 3.1–3.3, pp. 3–4](https://arxiv.org/pdf/2105.04906v3#page=3) | Agreement, standard-deviation floor, off-diagonal covariance penalty | Symmetric anti-collapse predates LeJEPA; SIGReg targets projected laws beyond finitely many moments. |
| [DINO, Section 3, pp. 2–3](https://arxiv.org/pdf/2104.14294v2#page=2) | EMA teacher, centering/sharpening, output-probability agreement | LeJEPA uses an explicit Gaussian target and symmetric agreement without a teacher. |
| [I-JEPA, Section 3, pp. 3–4](https://arxiv.org/pdf/2301.08243v3#page=3) | A predictor maps context to target-block representations | Information-asymmetric prediction differs from symmetric crop agreement. |
| [Gaussian Embeddings, pp. 1, 3–4](https://arxiv.org/pdf/2510.05949v1#page=1) | Gaussian density, representation transformations, Jacobian volume | LeJEPA’s Gaussian target comes from downstream-estimation analysis and sliced CF computation. |
| [Gretton et al., MMD, PDF pp. 1, 3–4](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) | Kernel mean embeddings for comparing distributions | Background for exact integration. Gretton et al. also give a linear-time MMD estimator. |

## 6. LeJEPA: Empirical Validation

Follow Sections 6.1→6.5. Numbers are **paper-reported** accuracy percentages, with higher being better. Frozen evaluation keeps the encoder fixed and trains only a linear classifier on labels; finetuning also updates the backbone.

### 6.1 LeJEPA’s Stability Across Hyper-Parameters and Architectures

The first question is whether useful training survives configuration and architecture changes. Table 1 evaluates **ImageNet-1K, ViT-Large/14, 100 epochs, frozen linear top-1**.

| Ablation | Representative reported results | What to read |
| --- | --- | --- |
| Integration range, 17 grid points, $M=512$ | $[-1,1]$ 72.13 → $[-5,5]$ 74.21 | Changing the number of quadrature points (5, 17, 41) moves accuracy by at most 0.6 points. |
| Batch size | 128: 72.20; 256: 74.15; 512: 74.72; 1024: 74.07 | Training works at 128; 512 is best. |
| Global views, 4 total | $V_g=1$: 53.06; $V_g=2$: 72.26 | The largest change in the table (19.20 points). |
| Global views, 10 total | $V_g=2$: 74.06; $V_g=4$: 75.08 | $V_g=4$ is 1.02 points above the recommended $V_g=2$, $V_l=8$. |
| Registers, $M=4096$ | 0: 75.61; 8: 75.84 | A 0.23-point difference. |

[Table 1, p. 13](https://arxiv.org/pdf/2511.08544v3#page=13). The authors write that the integration domain and number of quadrature points have “negligible impact” on performance, because the characteristic function is accurate near zero and so characterizes the moments even with a modest range. **Review interpretation.** That holds for the number of quadrature points, but widening the range from $[-1,1]$ to $[-5,5]$ adds 2.1–2.5 points at both 512 and 2048 directions, comparable to raising the batch size from 128 to 512 (2.52 points).

The paper’s recommended starting point is $\lambda=0.05$, two global and eight local views, batch size $\ge128$, 1024 directions, and 17 points on $[-5,5]$. However, the common setting applied to all experiments (Experiment Details 1) uses **eight views in total** including two global views (six local), AdamW, a warmup/cosine learning-rate schedule, and learning-rate/weight-decay candidates. [Section 6.1, Experiment Details 1](https://arxiv.org/pdf/2511.08544v3#page=14)

Figure 9 evaluates about 50 timm models below 20M parameters, across eight architecture families, with frozen evaluation on IN-10; the authors report that all reach 91.5–95% top-1. In Figure 8 and Table 7 (IN-100, ResNet50, 400 epochs), the authors regard performance as stable in $\lambda$ and the number of views. With eight views, $\lambda=0.010$–$0.150$ gives 83.07–84.32, while the ends of the range give 76.67 at $\lambda=0.001$ and 77.72 at $0.500$. [Section 6.1, Figures 8 and 9; Table 7, p. 43](https://arxiv.org/pdf/2511.08544v3#page=14)

Table 4, on IN-100 for 400 epochs, compares projector depth, a predictor, and stochastic weight averaging (SWA). In the paper, SWA averages the weights of the encoder that produces the center $\boldsymbol\mu_n$, that is, the teacher side. Without a predictor and with a three-layer projector, SWA raises ViT-S from 81.07 to 84.12, while ResNet50 moves from 83.93 to 83.50. The authors conclude that a teacher–student configuration gives ViTs a small boost but is not needed to prevent collapse, and recommend training without a predictor or registers, optionally applying SWA with ViTs. [Section 6.1; Table 4, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42)

### 6.2 LeJEPA’s Training Loss Is Informative of Downstream Performance

The second question is whether training loss signals representation quality before label-based evaluation. Minimizing one term alone is insufficient: agreement alone permits collapse; Gaussian matching alone does not ensure shared image content.

![Figure 10: Downstream accuracy in the prediction-loss and SIGReg-loss plane.](/assets/reviews/lejepa/paper-figure-10.webp)

**Reading Figure 10.** The horizontal axis is SIGReg loss and the vertical axis prediction loss, both on log scales; color encodes downstream accuracy, increasing from blue to red. The three panels are ResNet50 on Galaxy10, ResNet50 on IN-10, and ViT-B/14 on IN-1K. Points turn red toward the lower left, where both losses are low. The caption describes points of similar accuracy as forming linear trade-off fronts that correspond to different $\lambda$ values and point toward the lower left. [Enlarge](/assets/reviews/lejepa/paper-figure-10.webp) · [Figure 10, p. 15](https://arxiv.org/pdf/2511.08544v3#page=15). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

Figure 11 plots, for each $\alpha$, the Spearman rank correlation $C^{(\alpha)}$ between the training loss divided by $\lambda^\alpha$ and downstream accuracy. The authors report about 85% for the unadjusted loss ($\alpha=0$) and about 99% at $\alpha\approx0.4$, and present this as grounds for label-free model selection. [Section 6.2, Figures 10–11; supplementary Figures 19 and 21](https://arxiv.org/pdf/2511.08544v3#page=15)

**How to read the correlation.** The Spearman coefficient is the Pearson correlation of the ranks of two observation lists $A,B$ (review auxiliary symbols).

$$
\rho_{\mathrm S}=\operatorname{corr}\!\left(\operatorname{rank}(A),\operatorname{rank}(B)\right).
$$

It ranges from $-1$ to $1$; a larger magnitude indicates a stronger monotonic relation. **Review interpretation.** If lower loss goes with higher accuracy, the coefficient should be negative, so the paper’s 85% and 99% appear to be sign-flipped or absolute values expressing the strength of the monotonic relation.

### 6.3 In-Domain LeJEPA Outperforms Frontier Model Transfer Learning

The third question is whether a small model pretrained directly in a specialized domain can be preferable to a general foundation model. Galaxy10 contains about 11,000 training images in ten morphology classes. LeJEPA uses its default pipeline for **400 epochs** of in-domain pretraining and is compared with DINOv2/v3 ViT-S pretrained on the natural-image datasets LVD-142M and LVD-1.7B.

![Figure 12: Galaxy10 by downstream label count, with full finetuning and frozen evaluation.](/assets/reviews/lejepa/paper-figure-12.webp)

**Reading Figure 12.** Horizontal: downstream labels per class. Vertical: accuracy. Left: full finetuning. Right: frozen evaluation. Orange shades are LeJEPA; blue shades are DINOv2/v3. In finetuning, the gap between LeJEPA ConvNeXtV2-Nano and DINOv3 is largest at 5–10 labels per class (13–15 points) and shrinks to 1.12 points with all labels. In frozen evaluation, the curves overlap at 1–2 labels, and from 5 labels on ResNet-34 stays 6.5–9.1 points above DINOv3. The error bars come from three seeds, but the paper does not say whether they are standard deviations or confidence intervals. [Enlarge](/assets/reviews/lejepa/paper-figure-12.webp) · [Figure 12, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

| Table 3 model | All-label frozen | All-label finetuning | 1-shot frozen |
| --- | --- | --- | --- |
| LeJEPA ConvNeXtV2-Nano | 76.52 | 82.72 | 28.74 |
| LeJEPA ResNet-34 | 78.17 | 83.28 | 31.08 |
| LeJEPA LeViT-128 | 69.00 | 79.41 | 25.85 |
| DINOv2 Small | 67.62 | 78.34 | 27.68 |
| DINOv3 ViT-S/16 | 71.38 | 81.60 | 30.17 |

[Table 3, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42). The authors write that in-domain pretraining “substantially” outperforms DINOv2 and DINOv3 in both frozen evaluation and finetuning. In frozen evaluation, ResNet-34 is above DINOv3 from 1-shot (31.08 vs 30.17) to full labels, while LeViT-128 stays below DINOv3 even with full labels (69.00 vs 71.38).

Table 5 extends frozen evaluation to small datasets, including 1,020 training samples in flowers102. On flowers102, LeJEPA ResNet18 (74.57) and ResNeXt26ts (82.19) are below I-JEPA ViT-H/14 (85.76); on Galaxy10, ResNet34’s 77.29 is above I-JEPA’s 62.93. Table 5’s I-JEPA row name says IN-22K while its pretraining cell says IN-1K, and the Galaxy10 ResNet-34 frozen value differs between Table 3 (78.17) and Table 5 (77.29). [Table 5, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

### 6.4 LeJEPA Scales Across Data and Models

For larger models trained on IN-1K, online linear probing gives 77.1% for ViT-L (0.3B) and 78.5% for ConvNeXtV2-H (0.6B). Transfer evaluation keeps the backbone fixed and trains classifiers on eight datasets; the baselines are I-JEPA ViT-H (0.6B), the closest setup, and I-JEPA + STOP, an improved version that adds stochastic prediction tasks. [Section 6.4, p. 17](https://arxiv.org/pdf/2511.08544v3#page=17) **Review interpretation.** The abstract says a ViT-H/14 reaches 79% in frozen linear evaluation, but neither the main text nor the appendix contains a table or figure with a LeJEPA ViT-H result; the largest reported IN-1K linear-evaluation value is ConvNeXtV2-H’s 78.5%.

| Table 2 model and budget | 1-shot average | 10-shot average | All-label average |
| --- | --- | --- | --- |
| LeJEPA ViT-L, 304M, IN-1K, 100 epochs | 29.55 | 60.95 | 79.48 |
| LeJEPA ConvNeXtV2-H, 660M, IN-1K, 100 epochs | 31.58 | 60.70 | 77.56 |
| I-JEPA ViT-H, 632M, IN-1K, 300 epochs | 30.20 | 60.51 | 78.50 |
| I-JEPA ViT-H + STOP, same size/data/epochs | 32.05 | 62.92 | 80.70 |
| I-JEPA ViT-H, IN-22K, 900 epochs | 35.15 | 63.28 | 80.82 |

[Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16). The authors write that LeJEPA “consistently” outperforms I-JEPA with a smaller model and a shorter schedule. The smaller-model condition holds only for ViT-L (304M); ConvNeXtV2-H (660M) is larger than I-JEPA (632M). On averages, both LeJEPA models are higher at 10-shot, but ViT-L’s 1-shot (29.55 vs 30.20) and ConvNeXtV2-H’s full-label result (77.56 vs 78.50) are lower. Per task, full-label results are higher on DTD, flowers102, and food101 and lower on CIFAR-10, CIFAR-100, and pets. I-JEPA + STOP exceeds both LeJEPA models on all three averages, and the IN-22K, 900-epoch row belongs to a different budget group.

Table 2’s caption describes 100 versus 300 epochs as a “3× reduction” in training time and computational resources, and the authors cite the smooth training-loss curve of a ViT-gigantic (1.8B) model (Figure 1) as an example of optimization stability. [Section 6.4; Figure 1](https://arxiv.org/pdf/2511.08544v3#page=1) **Review interpretation.** The two models differ in size (304M vs 632M) and architecture, so the epoch ratio is not a compute ratio, and the paper reports neither wall-clock time nor FLOPs.

### 6.5 Emergent Semantic Structure in LeJEPA Representations

Figure 13 thresholds [CLS] attention to reveal foregrounds in video frames. Figure 14 maps patch features to RGB using the first three principal components (PCA) for ViT-L trained on IN-1K for 100 epochs. The authors describe warm colors as consistently capturing foreground objects and cool colors the background, and cite attention masks that track objects across frames as evidence of temporal structure. The full panels and videos are in [Figures 13–14](https://arxiv.org/pdf/2511.08544v3#page=17) and on the [official project page](https://rbalestr-lab.github.io/lejepa/).

**Review interpretation.** Because PCA is computed **independently per image**, matching colors across images do not indicate a shared feature direction, and no segmentation metric such as IoU is reported, so these results remain qualitative evidence.

<span id="7-conclusion-the-authors-conclusion-and-its-scope" aria-hidden="true"></span>

## 7. Conclusion

The authors conclude that they proved isotropic Gaussian embeddings uniquely minimize worst-case downstream risk, that SIGReg enforces this distribution tractably, and that the design removes collapse by construction rather than through ad hoc combinations of teacher–student networks, stop-gradients, or asymmetric architectures. They summarize that LeJEPA was validated on more than 60 architectures, up to 1.8B parameters, and matches state-of-the-art performance with fewer than 50 lines of core implementation. [Section 7, p. 18](https://arxiv.org/pdf/2511.08544v3#page=18)

## Review interpretation and open questions

**Review interpretation.** The paper’s strength is tying the choice of a target distribution (Section 3) and a linear-time loss that enforces it (Section 4) into one design, then applying the same default setting to many ImageNet architectures and to a non-natural-image domain such as Galaxy10. The limits differ by kind of evidence. In the theory, “unique optimality” concerns an upper bound or leading term of local-estimator bias, and Eqs. (5)→(6) and Theorems 4 and 6 have omitted-term and normalization issues ([\[3/3\] technical version](/en/reviews/lejepa-technical/)). In the experiments, the transfer comparison has no equal-budget I-JEPA, the full-label finetuning advantage on Galaxy10 is 1–2 points, and the $\alpha$ in the loss/accuracy correlation is chosen by looking at evaluation accuracy. Open questions are whether the correlation holds in a new domain with $\alpha$ fixed without evaluation labels, whether a gap to I-JEPA remains at equal size and schedule, and whether the predictor-free conclusion carries over to asymmetric JEPAs such as action-conditioned prediction.

## Sources and reading scope

- Paper: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics* · Randall Balestriero et al. · first public date 2025-11-11 · [reviewed version](https://arxiv.org/abs/2511.08544v3) · reviewed 2026-10-23.
- Scope read: Version 3, sections 1–7 and experiments 6.1–6.5, key appendix-C tables, and alternatives in D–F. Long proofs from A–B are linked to part 3.
- Figure sources: The concept thumbnail is review-authored (CC BY 4.0). Paper figures are Balestriero and LeCun’s v3 assets (CC BY-SA 4.0), with full-panel cropping and lossless WebP conversion. [Paper](https://arxiv.org/abs/2511.08544v3).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
