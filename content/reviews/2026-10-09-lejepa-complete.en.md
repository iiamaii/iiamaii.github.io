---
title: "[2/3] LeJEPA in full: from target distribution to theory, algorithm and experiments"
description: "A section-aligned account of the theory, SIGReg design, related work, and central experiments, including conditions and contrary results."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-23T08:56:16+00:00"
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

LeJEPA derives an isotropic-Gaussian target from downstream estimation analysis and implements it with SIGReg and view agreement. In full-label frozen evaluation on Galaxy10, its ResNet-34 reaches 78.17%, compared with 71.38% for DINOv3 ViT-S. [Section 6.3 and Table 3](https://arxiv.org/pdf/2511.08544v3#page=42) **Review interpretation.** Training directly on a small domain demonstrates practical value, while the local and distributional assumptions of the theory remain essential.

This is **[2/3] the complete, section-aligned commentary**. Read [\[1/3\] introduction](/en/reviews/lejepa-overview/) → this article → [\[3/3\] theorem and proof commentary](/en/reviews/lejepa-technical/). Headings and subsection numbers follow the paper. Main statements and assumptions remain where they occur; long intermediate calculations move to the technical version in the same Section 3→4→5 order.

The thumbnail’s objectives gather samples along different axes: within-image agreement on the lower left and across-image regularization within each view on the lower right. The Gaussian constraint acts after the projector. [Enlarge](/assets/reviews/lejepa/lejepa-core.svg)

<span id="1-introduction-the-problem-and-the-authors-route" aria-hidden="true"></span>

## 1. Introduction

Self-supervised learning uses relationships between inputs instead of target labels. For images, this connects crops and augmentations of one source image. Minimizing embedding distances alone also permits constant outputs. Avoiding collapse while supporting different datasets and architectures calls for an explicit objective.

The authors argue that accumulating empirical mechanisms increases configuration costs, while anti-collapse theory alone does not select a favorable feature geometry. Section 2 defines JEPA’s requirements; Section 3 derives a distribution target through downstream prediction. Section 4 makes high-dimensional matching tractable; Section 5 assembles the objective. Section 6 evaluates stability, loss informativeness, domains, scale, and semantic structure. [Section 1, pp. 1–3](https://arxiv.org/pdf/2511.08544v3#page=1)

<span id="2-background-and-notations-definitions-and-design-requirements" aria-hidden="true"></span>

## 2. Background and Notations

### 2.1 Notations and Definitions

The input tensor has shape $(N,V,D)$: $N$ independent samples, $V$ views each, and input dimension $D$. The encoder $f_{\boldsymbol\theta}:\mathbb R^D\to\mathbb R^K$ produces $K$-dimensional embeddings. $\mathbf Z\in\mathbb R^{N\times K}$ collects embeddings for one view.

JEPA requires predictability between related embeddings and retention of information that distinguishes inputs. A separate predictor can be omitted for symmetric image-view agreement. The same conclusion does not automatically apply to information-asymmetric context/target or time/action-conditioned prediction. This paper principally evaluates image representation learning. [Section 2.1, p. 4](https://arxiv.org/pdf/2511.08544v3#page=4)

### 2.2 The Need for Reliable Pretraining

A useful pretraining method should support small datasets and convolutional or hybrid architectures as well as large natural-image datasets and ViTs. Retuning interacting teacher updates, temperatures, or prototypes makes specialized-domain use difficult. This motivates Sections 6.1 and 6.3. [Section 2.2](https://arxiv.org/pdf/2511.08544v3#page=4)

### 2.3 The Need for Actionable Theory

Non-collapse is a necessary starting point but does not identify which representation is favorable downstream. The authors seek theory that selects an embedding law before specific labels are known. The following results motivate that choice; they do not directly prove the performance of an entire neural network. [Section 2.3](https://arxiv.org/pdf/2511.08544v3#page=4)

## 3. Latent Euclidean — Embeddings Should Be Isotropic Gaussian

### 3.1 Linear Probing

Start with fixed embeddings and linear prediction of labels $\mathbf y$. The paper’s OLS notation includes ridge regularization:

$$
\hat\beta=\underset{\beta\in\mathbb R^K}{\arg\min}
\|\mathbf y-\mathbf Z\beta\|_2^2+\lambda\|\beta\|_2^2.
$$

$\mathbf Z_{\rm iso}$ and $\mathbf Z_{\rm aniso}$ have the same column span and total energy, with different covariance shapes. Appendices B.1–B.2 use fixed design, full rank, and centered homoscedastic noise.

**Lemma 1 — Anisotropy amplifies bias.** For $\lambda>0$, **there exists** a downstream task with greater coefficient-estimation bias under anisotropy. Choosing a true task coefficient along a small-eigenvalue direction increases ridge shrinkage. This is not a greater-bias statement for every label vector.

**Lemma 2 — Anisotropy amplifies variance.** With $\lambda=0$, isotropy minimizes total coefficient variance. Writing $\lambda_k$ for the eigenvalues of the Gram matrix $\mathbf Z^\top\mathbf Z$,

$$
\operatorname{tr}\operatorname{Var}(\hat{\boldsymbol\beta}\mid\mathbf Z)
=\sigma^2\sum_{k=1}^K\frac1{\lambda_k}.
$$

At fixed energy, the reciprocal sum is minimized by equal eigenvalues. A review example gives 2 for $(1,1)$ versus about 5.56 for $(0.2,1.8)$. A conclusion about **coefficient variance** should not be equated with prediction risk at every test point. The ridge expansion and Jensen step appear in the [technical version, Section 3.1](/en/reviews/lejepa-technical/#31-linear-probing-lemmas-12-and-appendices-b1b2). [Section 3.1; Appendices B.1–B.2, pp. 26–27](https://arxiv.org/pdf/2511.08544v3#page=26)

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

$\mathbf q$ is the query embedding, $r_0$ the neighborhood radius, and $h$ the bandwidth. The paper reuses $K$ for both the kernel function and embedding dimension. Radius-based kNN differs from selecting a fixed number of neighbors.

**Theorem 1 — Isotropic Gaussian Optimality** connects integrated squared bias to Fisher information:

$$
\begin{aligned}
\mathrm{ISB}_{k\text{-NN}}&=\frac{r_0^4}{(K+2)^2}\tau_g^2J(p)+O(r_0^4),\\
\mathrm{ISB}_{\rm kernel}&\le
\left(\frac{h^2\mu_2(K)}2\right)^2
\left(2B^2+8L^2J(p)\right)+o(h^4),\\
J(p)&=\int\|\nabla\log p(\mathbf z)\|_2^2p(\mathbf z)\,d\mathbf z.
\end{aligned}
$$

The density must be sufficiently smooth with finite Fisher information and covariance, and queries follow the training density. The kNN analysis uses a small radius and an isotropic task-gradient prior. The kernel analysis uses small $h$, sufficient local sample density, a symmetric normalized kernel, and bounds $\|\nabla\eta\|\le L$, $|\Delta\eta|\le B$. Here $B$ is a curvature bound, distinct from the batch notation in Section 5. [Appendix A, Lemmas 4–5, Theorems 7–8](https://arxiv.org/pdf/2511.08544v3#page=24)

At fixed covariance $\Sigma$, $J(p)\ge\operatorname{tr}(\Sigma^{-1})$, with equality for a Gaussian. A fixed total-variance budget then selects isotropic $\Sigma$. This is the mathematical core of the target choice. However, the kNN remainder $O(r_0^4)$ has the same order as the displayed term; eliminating its cross term requires extra orthogonality. The kernel expression is a worst-case bias **upper bound**. These calculations alone do not establish a unique minimum of exact ISB or total MSE for every task. Read the authors’ optimality claim with its conditions and the scope of the argument. Taylor expansions and the Fisher lower bound are developed in the [technical version, Section 3.2](/en/reviews/lejepa-technical/#32-nonlinear-probing-theorem-1-and-appendices-a-b3b7). [Section 3.2; Appendices B.3–B.7](https://arxiv.org/pdf/2511.08544v3#page=28)

### 3.3 Geometric and Practical Insights

The authors illustrate estimated coefficients and decision boundaries under fixed total variance. Figure 3 repeats logistic regression on 100 samples for a two-class task, comparing isotropic and anisotropic decision boundaries. This is an empirical connection to OLS theory, rather than an identical instantiation of its assumptions. Appendix Figure 18 visualizes regularization and bias. [Section 3.3, Figures 3 and 18](https://arxiv.org/pdf/2511.08544v3#page=6)

## 4. SIGReg — Reliable Isotropic Gaussian Regularization in High-Dimension

### 4.1 Hypothesis Testing as a Judge

Instead of estimating the full high-dimensional density, compare one-dimensional laws. Let $P_{\boldsymbol\theta}$ be the embedding law and $Q=\mathcal N(0,I)$ the target. **Lemma 3 — Hyperspherical Cramér–Wold** equates equality of all unit-direction projection laws with equality of vector laws. Appendix B.8 proves this through characteristic functions and $\mathbf t=r\mathbf a$.

**Theorem 2 — Sufficiency of directional tests** concerns consistency of combined projection tests. Appendix B.9 requires vanishing directional mesh, a threshold calibrated for the global null, and power in a neighborhood of a separating direction. It does not certify arbitrary distributions from a few finite directions. The maximum statistic for testing also differs from the averaged training objective:

$$
\operatorname{SIGReg}(\{\mathbf z_n\}_{n=1}^N)
=\frac1{|\mathcal A|}\sum_{\mathbf a\in\mathcal A}
T(\{\mathbf a^\top\mathbf z_n\}_{n=1}^N).
$$

This is the directional average in Definition 2. Section 4.2 chooses the statistic $T$. [Section 4.1, Lemma 3, Theorem 2, Definition 2; Appendices B.8–B.9](https://arxiv.org/pdf/2511.08544v3#page=6)

![Figure 2: Input distribution, embedding space, and directional density comparisons.](/assets/reviews/lejepa/paper-figure-2.webp)

**Reading Figure 2.** Each colored arrow in the middle yields a projected density on the right. The horizontal coordinate is the projection; black is the Gaussian target. Gray discrepancy is conceptual, not downstream accuracy. [Enlarge](/assets/reviews/lejepa/paper-figure-2.webp) · [Figure 2, p. 3](https://arxiv.org/pdf/2511.08544v3#page=3). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

### 4.2 SIGReg: Sketching the Epps–Pulley Test Is Stable and Scalable

#### 4.2.1 Moments Are Unstable and Insufficient

**Theorem 3** constructs different distributions matching moments through any fixed finite order. Mean, variance, and covariance alone cannot establish a joint Gaussian law. Increasing the order also creates polynomial gradients that can grow on tail samples. The linear-algebra construction in Appendix B.11 is explained in the [technical version, Section 4.2](/en/reviews/lejepa-technical/#42-sigreg-sketching-the-eppspulley-test). [Section 4.2.1; Appendix B.11](https://arxiv.org/pdf/2511.08544v3#page=7)

![Figure 5: Gaussian coordinate marginals and identity covariance can coexist with an X-shaped joint law.](/assets/reviews/lejepa/paper-figure-5.webp)

**Reading Figure 5.** Coordinate densities are Gaussian and covariance is identity, while joint samples form an X. An oblique projection reveals a non-Gaussian density. The right panel plots test responses by direction; the statistics need not share a common scale. The lesson is the difference between plausible coordinate summaries and matching a joint law. [Enlarge](/assets/reviews/lejepa/paper-figure-5.webp) · [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

#### 4.2.2 Cumulative Density Functions Are Impractical

A CDF gives the probability below a threshold and can represent a distribution. The paper emphasizes sorting/order-statistic costs, gathering samples across GPUs, and approximations suited to gradient learning. This should not be exaggerated into a claim that sorting is nowhere differentiable. The practical judgment concerns this training and distributed-computation setting. [Section 4.2.2, p. 9](https://arxiv.org/pdf/2511.08544v3#page=9)

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

The paper’s Epps–Pulley equation uses a Gaussian weight $w(t)$. **Theorem 4 — Stability of Epps–Pulley Test** relies on $|e^{itz}|=1$, giving bounded CFs independently of input magnitude. However, Appendix B.12 derives the $O(1/N)$ sample-gradient bound for the **unscaled distance**, $EP/N$. Multiplying by $N$ to obtain $EP=N\widehat D$ multiplies the gradient too. The [technical version, Section 4.2.3](/en/reviews/lejepa-technical/#423-cf-deriving-theorem-4-and-checking-normalization) retains this distinction. Parameter gradients also contain the encoder Jacobian, so bounded CF factors alone cannot establish an unconditional gradient bound for the whole network. [Section 4.2.3, Epps–Pulley equation, Theorem 4; Appendix B.12](https://arxiv.org/pdf/2511.08544v3#page=9)

**Reading Algorithm 1.** Normalize columns of a $K\times M$ Gaussian matrix to obtain unit directions. Multiply $\mathbf Z\mathbf A$ by frequencies and average complex exponentials. Synchronize the step seed across GPUs and all-reduce the CF mean. On 17 points in $[-5,5]$, weight the squared Gaussian-CF error with a Gaussian window, apply trapezoidal integration, then multiply by the global sample count $N$. SIGReg averages the directional statistics. This describes the paper’s Algorithm 1, not a fresh execution of current official code. [Algorithm 1, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10)

Dense cost is $\mathcal O(NKM+NMT)$ using review shorthands $M,T$. It is linear in $N$ with $K,M,T$ fixed, not a claim about every cost including the encoder. Table 6 reports 10 loss-forward/backward runs (mean ± std) on a V100-SXM2-16GB: $(N,M,T)=(512,512,16)$ gives $0.465236\pm0.011642$ ms; $(8192,512,16)$ gives $6.188304\pm0.007226$ ms. Embedding dimension is absent from the table. These are not end-to-end training comparisons. [Table 6, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

### 4.3 How SIGReg Beats the Curse of Dimensionality

The paper proceeds through **smoothness**, **SGD resampling**, and **synthetic validation**. Theorem 5 gives a direction-count rate $M^{-2\alpha/(K-1)}$, where $\alpha$ is Sobolev regularity. Appendix B.10 references spherical-harmonic approximation and interpolation with quasi-uniform points. Large smoothness alone does not guarantee small practical error for arbitrary $O(K)$ directions: dimension/smoothness-dependent constants and norms remain. [Section 4.3, Theorem 5; Appendix B.10](https://arxiv.org/pdf/2511.08544v3#page=10)

![Figure 7: Final directional statistic by number of directions, fixed versus resampled.](/assets/reviews/lejepa/paper-figure-7.webp)

**Reading Figure 7.** Horizontal: directions $M$ per step. Vertical: expected directional statistic after training, where lower means closer matching. Green resampled directions achieve low discrepancy with fewer directions than blue fixed directions. This is synthetic optimization, not ImageNet accuracy or a formal convergence rate. [Enlarge](/assets/reviews/lejepa/paper-figure-7.webp) · [Figure 7, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

The synthetic experiment then **directly optimizes samples** whose coordinates are mostly Gaussian but whose first two dimensions have an X structure. Section 4.3 mentions a $D=512$, $M=16$ case, whereas Figure 6’s caption specifies $N=100$, $D=1024$, $M=10$. These settings are not equated. Removing encoder constraints isolates the loss; semantic learning from images is assessed separately in Section 6. [Section 4.3, p. 11](https://arxiv.org/pdf/2511.08544v3#page=11) · [Figure 6, p. 9](https://arxiv.org/pdf/2511.08544v3#page=9)

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

$\mathcal L_{\rm pred}^{\rm code}$ and $\mathcal L_{\rm SIGReg}$ are review abbreviations; $1/K$ exposes the code’s feature-coordinate mean. The total loss uses batch notation $B$ in the paper, corresponding to $N$ here. Do not confuse the within-image center axis with the across-image SIGReg axis.

Sample expectations, normalization, and the exact relation between the two loss forms are developed in [technical part section 5](/en/reviews/lejepa-technical/).

The core architecture requires no prototypes, stop-gradient, EMA target teacher, or separate predictor. It still uses a projector; evaluation uses backbone features. Removing a predictor for symmetric image agreement does not establish that every JEPA task needs none. Optimizers, loaders, and learning-rate schedules remain ordinary configuration choices.

### 5.2 Relation to Prior Work

The paper discusses sliced score matching, sliced Wasserstein, exact integration and MMD, then the moment-limited VICReg connection, in that order. Projection and distribution discrepancy already exist in other fields; the contribution locates them within a JEPA objective. Appendix B.14 shows that zero mean and unit variance in every direction force zero vector mean and identity covariance. It does not establish numerical equality with VICReg’s original standard-deviation hinge and covariance penalties. [Section 5.2; Appendix B.14, pp. 39–40](https://arxiv.org/pdf/2511.08544v3#page=12)

### Review supplement: comparing the original methods

| Primary source and reading scope | Central method | Relation to LeJEPA |
| --- | --- | --- |
| [VICReg, Sections 3.1–3.3, pp. 3–4](https://arxiv.org/pdf/2105.04906v3#page=3) | Agreement, standard-deviation floor, off-diagonal covariance penalty | Symmetric anti-collapse predates LeJEPA; SIGReg targets projected laws beyond finitely many moments. |
| [DINO, Section 3, pp. 2–3](https://arxiv.org/pdf/2104.14294v2#page=2) | EMA teacher, centering/sharpening, output-probability agreement | LeJEPA uses an explicit Gaussian target and symmetric agreement. Architectural diversity itself is not new here. |
| [I-JEPA, Section 3, pp. 3–4](https://arxiv.org/pdf/2301.08243v3#page=3) | A predictor maps context to target-block representations | Information-asymmetric prediction differs from symmetric crop agreement. |
| [Gaussian Embeddings, pp. 1, 3–4](https://arxiv.org/pdf/2510.05949v1#page=1) | Gaussian density, representation transformations, Jacobian volume | Shared Gaussian terminology differs from the downstream-estimation/sliced-CF argument. |
| [Gretton et al., MMD, PDF pp. 1, 3–4](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) | Kernel mean embeddings for comparing distributions | Background for exact integration. MMD also has linear-time estimators, so LeJEPA is not the first linear-time distribution comparison. |

## 6. LeJEPA: Empirical Validation

Follow Sections 6.1→6.5. Numbers are **paper-reported** accuracy percentages, with higher being better. Frozen probing trains a labeled classifier while keeping the encoder fixed; finetuning also updates the backbone. Unlabeled pretraining does not imply label-free evaluation.

### 6.1 LeJEPA’s Stability Across Hyper-Parameters and Architectures

The first question is whether useful training survives configuration and architecture changes. Table 1 evaluates **ImageNet-1K, ViT-Large/14, 100 epochs, frozen linear top-1**.

| Ablation | Representative reported results | Interpretation |
| --- | --- | --- |
| Integration range, 17 grid points | $M=512$: $[-1,1]$ 72.13 → $[-5,5]$ 74.21 | The 2.08-point gap means the range is not irrelevant. |
| Batch size | 128: 72.20; 256: 74.15; 512: 74.72; 1024: 74.07 | Small batches can train; accuracy is neither equal nor monotonic. |
| Global views, 4 total | $V_g=1$: 53.06; $V_g=2$: 72.26 | View configuration can have a large effect. |
| Global views, 10 total | $V_g=2$: 74.06; $V_g=4$: 75.08 | A recommended default is not the optimum of every ablation. |
| Registers, $M=4096$ | 0: 75.61; 8: 75.84 | The improvement is small in this comparison. |

[Table 1, p. 13](https://arxiv.org/pdf/2511.08544v3#page=13). Do not combine separate ablation rows into one purported configuration.

The recommended recipe uses $\lambda=0.05$, two global/eight local views, $N\ge128$, $M=1024$, and 17 points on $[-5,5]$. However, Experiment Details 1 uses two global/six local views, **eight total**, with AdamW, warmup/cosine scheduling, and learning-rate/weight-decay candidates. These are different settings and are recorded separately.

Figure 9 evaluates about 50 timm models below 20M parameters, across eight architecture families, on IN-10. It is not an IN-1K result for every model. Figure 8 and Table 7 examine $\lambda$ and views; their low-sensitivity interpretation coexists with degraded extreme configurations. For ResNet50/eight views, $\lambda=0.020$ gives 83.49 versus 77.72 at $0.500$. [Section 6.1, Figures 8 and 9; Table 7, p. 43](https://arxiv.org/pdf/2511.08544v3#page=14)

Table 4, on IN-100 for 400 epochs, compares projector depth, predictor, and SWA. SWA averages model weights and is optional here. Without a predictor, ViT-S with a three-layer projector improves from 81.07 to 84.12 with SWA. The corresponding ResNet50 comparison is 83.93→83.50, without improvement. This does not show an EMA teacher is necessary to prevent collapse. [Table 4, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42)

### 6.2 LeJEPA’s Training Loss Is Informative of Downstream Performance

The second question is whether training loss signals representation quality before label-based evaluation. Minimizing one term alone is insufficient: agreement alone permits collapse; Gaussian matching alone does not ensure shared image content.

![Figure 10: Downstream accuracy in the prediction-loss and SIGReg-loss plane.](/assets/reviews/lejepa/paper-figure-10.webp)

**Reading Figure 10.** Coordinates are prediction and SIGReg losses; color encodes downstream accuracy, increasing from blue to red. Inspect the lower-left frontier where both objectives are low. One low coordinate is insufficient. Different $\lambda$ values also complicate direct comparison of final total losses. [Enlarge](/assets/reviews/lejepa/paper-figure-10.webp) · [Figure 10, p. 15](https://arxiv.org/pdf/2511.08544v3#page=15). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

Figure 11 investigates an adjustment such as $\mathcal L/\lambda^\alpha$ and Spearman association. The text reports 85% rising to 99% near $\alpha\approx0.4$. The sign convention for lower loss versus higher accuracy is unclear, so this review describes **strong monotonic association**. Selecting $\alpha$ uses observed downstream results; this is not a prospective guarantee of the same label-free selection rule in a new domain. [Section 6.2, Figures 10–11; supplementary Figures 19 and 21](https://arxiv.org/pdf/2511.08544v3#page=15)

**How to read the correlation.** Spearman correlation applies Pearson correlation to the two lists of ranks. Here $A,B$ are review auxiliary symbols for the compared observation lists.

$$
\rho_{\mathrm S}=\operatorname{corr}\!\left(\operatorname{rank}(A),\operatorname{rank}(B)\right).
$$

It ranges from $-1$ to $1$; a larger magnitude indicates a stronger monotonic relation. Lower loss paired with higher accuracy can produce a negative sign. The reported “99%” is not 99% classification accuracy, a success probability, or an error guarantee for new models. Without verified tie and sign conventions, retain only the preceding interpretation of association strength. [§6.2, Figure 11](https://arxiv.org/pdf/2511.08544v3#page=15)

### 6.3 In-Domain LeJEPA Outperforms Frontier Model Transfer Learning

The third question is whether a small model pretrained directly in a specialized domain can be preferable to a general foundation model. Galaxy10 contains about 11,000 training images in ten morphology classes. LeJEPA uses its default pipeline for **400 epochs** of in-domain pretraining. DINOv2/v3 bring general-image pretraining to the comparison.

![Figure 12: Galaxy10 by downstream label count, with full finetuning and frozen evaluation.](/assets/reviews/lejepa/paper-figure-12.webp)

**Reading Figure 12.** Horizontal: downstream labels per class. Vertical: accuracy. Left: full finetuning. Right: frozen evaluation. LeJEPA curves correspond to small domain-trained architectures, with clear strengths at larger label counts. Three seeds are reported; the error-bar summary is not specified as standard deviation or confidence intervals, so no such interpretation is inferred. [Enlarge](/assets/reviews/lejepa/paper-figure-12.webp) · [Figure 12, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

| Table 3 model | All-label frozen | All-label finetuning | 1-shot frozen |
| --- | --- | --- | --- |
| LeJEPA ConvNeXtV2-Nano | 76.52 | 82.72 | 28.74 |
| LeJEPA ResNet-34 | 78.17 | 83.28 | 31.08 |
| LeJEPA LeViT-128 | 69.00 | 79.41 | 25.85 |
| DINOv2 Small | 67.62 | 78.34 | 27.68 |
| DINOv3 ViT-S/16 | 71.38 | 81.60 | 30.17 |

[Table 3, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42). Both all-label strengths and 1-shot exceptions remain visible. Different domains, architectures, and budgets make this broader than a controlled loss-only substitution.

Table 5 extends to small datasets, including 1,020 training samples in flowers102. Frozen flowers102 gives 74.57 for ResNet18, 82.19 for ResNeXt26ts, and 85.76 for the I-JEPA comparison. Galaxy10 gives 77.29 for ResNet34 versus 62.93 for I-JEPA. Practical small-data learning does not mean first place everywhere. The I-JEPA row name says IN-22K while its pretraining cell says IN-1K; the exact source configuration remains unconfirmed. Galaxy10 results in Tables 3 and 5 are not pooled. [Table 5, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

### 6.4 LeJEPA Scales Across Data and Models

For larger data/models, IN-1K online linear probing gives 77.1 for ViT-L (0.3B) and 78.5 for ConvNeXtV2-H (0.6B). These differ from the **eight-task transfer average** below. The abstract’s roughly 79% for ViT-H is not merged into the same row either. [Section 6.4, p. 17](https://arxiv.org/pdf/2511.08544v3#page=17)

| Table 2 model and budget | 1-shot average | 10-shot average | All-label average |
| --- | --- | --- | --- |
| LeJEPA ViT-L, 304M, IN-1K, 100 epochs | 29.55 | 60.95 | 79.48 |
| LeJEPA ConvNeXtV2-H, 660M, IN-1K, 100 epochs | 31.58 | 60.70 | 77.56 |
| I-JEPA ViT-H, 632M, IN-1K, 300 epochs | 30.20 | 60.51 | 78.50 |
| I-JEPA ViT-H + STOP, same size/data/epochs | 32.05 | 62.92 | 80.70 |
| I-JEPA ViT-H, IN-22K, 900 epochs | 35.15 | 63.28 | 80.82 |

[Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16). Strengths on DTD, flowers102, and food101 coexist with weaknesses on CIFAR and pets. ViT-L exceeds base I-JEPA in all-label average but is below it in 1-shot average. +STOP exceeds ViT-L on all three averages. The IN-22K row belongs to a different budget group.

100 versus 300 epochs demonstrates a shorter schedule, not a measured threefold reduction in wall-clock or FLOPs across different models, resolutions, crops, and hardware. Figure 1’s ViT-gigantic (1.8B) curve illustrates large-model optimization stability; it is not a table establishing downstream superiority at every scale. [Section 6.4; Figure 1](https://arxiv.org/pdf/2511.08544v3#page=1)

### 6.5 Emergent Semantic Structure in LeJEPA Representations

Figure 13 thresholds [CLS] attention to reveal foregrounds in video frames. Figure 14 maps patch features to RGB using the first three PCA components for ViT-L trained on IN-1K for 100 epochs. These are qualitative evidence of spatial and semantic organization after unlabeled pretraining.

PCA is computed **independently per image**, so equal colors do not imply a shared semantic direction across images. Temporal consistency is illustrated, without an IoU or video-segmentation benchmark score. The figures should not be generalized to action/future-prediction performance. See the complete panels and videos in [Figures 13–14](https://arxiv.org/pdf/2511.08544v3#page=17) and the [official project page](https://rbalestr-lab.github.io/lejepa/).

<span id="7-conclusion-the-authors-conclusion-and-its-scope" aria-hidden="true"></span>

## 7. Conclusion

The authors connect an isotropic Gaussian motivation, stable and scalable SIGReg, a simple LeJEPA objective, and experiments across architectures and domains. Selecting a representation distribution first provides an alternative starting point for designing anti-collapse training mechanisms. Their conclusion emphasizes theory guiding a practical design. [Section 7, p. 18](https://arxiv.org/pdf/2511.08544v3#page=18)

## Review interpretation and open questions

**Review interpretation.** This review locates the contribution in **combining a rationale for distribution choice with a tractable constraint**. Exact total-risk optimality, identification through finite grids, and domain-specific loss/accuracy relationships need further verification. The omitted variance in Eqs. (5)–(6) and normalization differences in Theorems 4 and 6 also deserve explicit treatment before implementation or reproduction.

Long proofs are expanded rather than removed in the [technical version: ridge/local bias/Fisher bounds in Section 3, identification/stability/directional approximation in Section 4, and sampling expectations/omitted terms in Section 5](/en/reviews/lejepa-technical/), following the paper’s order.

## Sources and reading scope

- Paper: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics* · Randall Balestriero et al. · first public date 2025-11-11 · [reviewed version](https://arxiv.org/abs/2511.08544v3) · reviewed 2026-10-23.
- Scope read: Version 3, sections 1–7 and experiments 6.1–6.5, key appendix-C tables, and alternatives in D–F. Long proofs from A–B are linked to part 3.
- Figure sources: The concept thumbnail is review-authored (CC BY 4.0). Paper figures are Balestriero and LeCun’s v3 assets (CC BY-SA 4.0), with full-panel cropping and lossless WebP conversion. [Paper](https://arxiv.org/abs/2511.08544v3).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
