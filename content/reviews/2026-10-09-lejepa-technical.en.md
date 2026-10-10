---
title: "[3/3] LeJEPA technical review: proof conditions, SIGReg and checked equations"
description: "Examines Gaussian minimality, projected identification, ECF gradients and bias, computational cost and a missing term in the distance-loss derivation."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-10T18:16:18+09:00"
topics: ["self-supervised learning", "representation learning", "joint embedding predictive architectures", "distribution matching", "computer vision"]
visibility: "public"
lang: "en"
translationKey: "lejepa-technical"
paperTitle: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
paperPublishedDate: "2025-11-11"
authors: ["Randall Balestriero", "Yann LeCun"]
year: "2025"
paperUrl: "https://arxiv.org/abs/2511.08544v3"
thumbnail: "/assets/reviews/lejepa/lejepa-core.svg"
thumbnailAlt: "LeJEPA combines within-image alignment to the global-view mean with across-image Gaussian matching through random projections in each view."
---

Assessing LeJEPA's “provable” claim requires separating three questions: <strong>when a Gaussian helps downstream estimation, when projected distribution equality identifies the original distribution, and how finite batches and numerical grids approximate that goal.</strong> This review develops those connections, records an equation discrepancy checked against the PDF, and reports small calculations executed specifically for the review. The aim is to understand both the theoretical contribution and the additional validation needed for training. [§§3–5; Appendices A–B](https://arxiv.org/pdf/2511.08544v3#page=24)

**Notation.** Embedding vectors $\mathbf z_{n,v}$, global means $\boldsymbol\mu_n$, global-view count $V_g$ and loss $\mathcal L$ follow the paper. In complexity discussions, $M=|\mathcal A|$ counts directions and $T$ counts quadrature frequencies as auxiliary review symbols. $\mathcal L_{\mathrm{SIGReg}}$ abbreviates the per-view average; $1/K$ in the centered loss records Algorithm 2’s feature averaging. Linear design $\mathbf Z$ and ridge coefficient $\lambda$ follow §3.1. $G=\mathbf Z^\top\mathbf Z$, unscaled discrepancy $D$ and distance decomposition $A,B,C$ are review definitions. $z_n=\mathbf a^\top\mathbf z_n$ is the scalar projection in §4; $s$ is the inverse window bandwidth used in Appendix B.12.

This is the technical version. The [introductory review](/en/reviews/lejepa-overview/) explains the intuition; the [complete explanation](/en/reviews/lejepa-complete/) maps related work and experiments.

<strong>The hero's sample axes.</strong> Views of one image align with their global-view mean. Distribution regularization operates across different images in each view, after the projector. It does not constrain every backbone layer or the crop set of one photograph to be Gaussian. Both branches receive gradients; core agreement has no stop-gradient. [Enlarge](/assets/reviews/lejepa/lejepa-core.svg)

## Notation and levels of analysis

$N$: different images per batch. $V$: views per image. $V_g$: global views. $K$: projected embedding dimension. $M$: projection directions. $T$: frequency-grid points. $\mathbf z_{n,v}$ is an encoder-plus-projector embedding.

| Level | Object of the claim | Additional requirement for training |
| --- | --- | --- |
| Downstream estimation | Fixed-design linear regression; local/kernel prediction under smooth density | Features, tasks and distributions must satisfy assumptions |
| Distribution identification | All directions/frequencies, or specified growing-set/test conditions | Finite directions and grids leave unobserved differences |
| Training loss | Finite-batch ECF discrepancy averaged across directions | Optimizer, encoder Jacobian, numerical error, generalization |
| Experiments | Particular data, models and evaluation protocols | Matched-budget reproduction, new domains, seed variation |

Original notation is retained where possible; auxiliary symbols introduced for the derivations are defined above. Review-authored examples are distinguished from reported paper results.

## Linear analysis: why equal directional variance helps

Assume a fixed full-column-rank feature matrix $\mathbf Z$, labels $\mathbf y=\mathbf Z\boldsymbol\beta+\boldsymbol\varepsilon$, conditional zero-mean noise and noise covariance $\sigma ^{2}I$. Write $G=\mathbf Z^\top\mathbf Z$ and ridge coefficient $\lambda>0$. [Lemma 1; Appendix B.1, p. 26](https://arxiv.org/pdf/2511.08544v3#page=26)

$$
\begin{aligned}
\hat{\boldsymbol\beta}_{\mathrm{ridge}}&=(G+\lambda I)^{-1}\mathbf Z^\top\mathbf y,\\
\mathbb E[\hat{\boldsymbol\beta}_{\mathrm{ridge}}\mid\mathbf Z]-\boldsymbol\beta
&=-\lambda(G+\lambda I)^{-1}\boldsymbol\beta.
\end{aligned}
$$

Along an eigenvector of $G$, shrinkage is $\lambda/(\lambda_k+\lambda)$. A weak direction therefore creates an unfavorable task relative to an isotropic design with the same total energy. This is an existence argument, not a claim that isotropy improves every specific task.

For OLS ($\lambda=0$), coefficient variance gives the next argument. [Lemma 2; Appendix B.2, p. 27](https://arxiv.org/pdf/2511.08544v3#page=27)

$$
\begin{aligned}
\operatorname{Cov}(\hat{\boldsymbol\beta}_{\mathrm{OLS}}\mid\mathbf Z)&=\sigma^2G^{-1},\\
\operatorname{tr}\operatorname{Cov}(\hat{\boldsymbol\beta}_{\mathrm{OLS}}\mid\mathbf Z)&=\sigma^2\sum_{k=1}^K\frac1{\lambda_k},\\
\sum_{k=1}^K\lambda_k=c>0&\quad\Longrightarrow\quad\sum_{k=1}^K\frac1{\lambda_k}\ge\frac{K^2}{c},\\
\text{equality: }&\lambda_k=c/K\quad\text{for all }k.
\end{aligned}
$$

Convexity of the reciprocal function, or Cauchy–Schwarz, proves the last inequality. In the review's two-dimensional calculation, eigenvalues $(1,1)$ give reciprocal sum 2; $(0.2,1.8)$ give approximately 5.5556 despite the same trace of 2.

This motivates <strong>isotropic covariance</strong>. It does not uniquely determine a Gaussian joint density, guarantee prediction risk under arbitrary test distributions, or establish learned-encoder performance.

## Nonlinear analysis: which term does Fisher information control?

Neighborhood averaging predicts from nearby labels; Nadaraya–Watson regression uses kernel weights. Small-radius/bandwidth expansions combine target curvature with density variation. The key kinds of terms are: [§3.2; Appendix A, B.3–B.7, pp. 24–35](https://arxiv.org/pdf/2511.08544v3#page=24)

$$
\begin{aligned}
\nabla m(\mathbf x)\cdot\nabla\log p(\mathbf x),\qquad&\frac12\Delta m(\mathbf x),\\
J(p)&=\int\|\nabla\log p(\mathbf x)\|_2^2p(\mathbf x)\,\mathrm d\mathbf x.
\end{aligned}
$$

Here $m$ is the target function and $\nabla\log p$ is the density score. If a task-gradient prior has zero mean and second moment $\tau_g^2I$, the squared score contribution becomes $\tau_g^2J(p)$.

Appendix B.4 additionally discusses decorrelation between gradient and curvature terms. Without it, a cross term remains $\mathcal O(r^{4})$, the same order as other squared-bias terms. It cannot simply be treated as lower order. Interpreting curvature as independent of $p$ also needs task-prior conditions. A controlled bias component is consequently not unconditional unique minimization of total bias. [Appendix B.4, p. 29](https://arxiv.org/pdf/2511.08544v3#page=29)

The kernel argument bounds worst-case bias using a quantity containing $2B^{2}+8L^{2}J(p)$, with $L,B$ target smoothness bounds. It minimizes an upper bound, not the exact risk of every target. Pointwise variance depends on query density; cancellation in an integrated variance requires integrability and compatible train/query distributions. [Appendix B.7, pp. 34–35](https://arxiv.org/pdf/2511.08544v3#page=34)

## The essential Gaussian-minimality proof

Assume zero mean, covariance $\Sigma\succ0$, a smooth density and sufficient boundary decay. With score $u(\mathbf x)=\nabla\log p(\mathbf x)$, integration by parts gives $\mathbb E[u(X)X^\top]=-I$. Non-negativity of a squared norm yields: [Appendix A; B.5, pp. 32–33](https://arxiv.org/pdf/2511.08544v3#page=32)

$$
\begin{aligned}
0&\le\mathbb E\|u(X)+\Sigma^{-1}X\|_2^2\\
&=J(p)-\operatorname{tr}(\Sigma^{-1}),\\
J(p)&\ge\operatorname{tr}(\Sigma^{-1}).
\end{aligned}
$$

Equality requires $u(\mathbf x)=-\Sigma^{-1}\mathbf x$ almost everywhere. Integrating gives $\log p(\mathbf x)=\text{const}-\frac12\mathbf x^\top\Sigma^{-1}\mathbf x$: a Gaussian density. With $\operatorname{tr}\Sigma=c$ fixed, $\operatorname{tr}\Sigma^{-1}\ge K^2/c$, with equality at $\Sigma=(c/K)I$. Normalizing the target covariance to $I$ gives the standard isotropic Gaussian.

This proves minimality of the Fisher objective under the conditions. Whether deterministic embeddings of a finite image dataset form a smooth full-dimensional density, and whether actual downstream tasks follow the prior, remain separate questions.

## Why all Gaussian projections identify the vector law

The vector characteristic function is $\varphi_Z(\mathbf t)=\mathbb E\exp(i\mathbf t^\top Z)$. Every nonzero $t$ can be written $\mathbf t=s\mathbf a$ for a unit direction $\mathbf a$. Agreement for every scalar frequency in every direction therefore implies agreement of vector characteristic functions at every $t$. Uniqueness of characteristic functions identifies the distribution. [§4.1; Appendix B.8, p. 35](https://arxiv.org/pdf/2511.08544v3#page=35)

![LeJEPA Figure 5: Gaussian-looking coordinate marginals do not establish Gaussianity of an X-shaped joint density.](/assets/reviews/lejepa/paper-figure-5.webp)

[Enlarge figure](/assets/reviews/lejepa/paper-figure-5.webp)

<strong>Connecting Figure 5 to the proof.</strong> The left histograms look Gaussian, but the X-shaped joint density gives different projections along diagonal directions. Third-panel rows correspond to colored directions; the final panel compares statistical responses. “All directions” cannot be replaced by a few coordinate moments. Attribution: Balestriero and LeCun, [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Complete panels cropped, rasterized and encoded as lossless WebP.

The maximum test statistic in Equation 4 also differs from the averaged training objective. Appendix B.9 uses growing dense direction sets, calibrated global null thresholds and power over a separating neighborhood. A finite directional average is not automatically a level-α test. Failure to reject a null is not proof of Gaussianity.

Candidates also include CDF distances and order-statistic tests such as Shapiro–Wilk. They require sorting/order relationships and different distributed communication. Sorting is not nowhere differentiable: it can be differentiated piecewise while order remains fixed. Order-changing boundaries and its computational structure differ from ECF averaging. Appendices E–F supply definitions of comparison statistics. [§4.2, pp. 7–9; Appendices E–F, pp. 43–46](https://arxiv.org/pdf/2511.08544v3#page=7)

## ECF discrepancy: computation and gradient control

For projected samples $z_n=\mathbf a^\top\mathbf z_n$, define an unscaled discrepancy $D$:

$$
\begin{aligned}
\hat\varphi_{\mathbf a}(t)&=\frac1N\sum_{n=1}^N[\cos(tz_n)+i\sin(tz_n)],\\
\varphi_{\mathcal N}(t)&=\exp(-t^2/2),\\
D(\mathbf a)&=\int|\hat\varphi_{\mathbf a}(t)-\varphi_{\mathcal N}(t)|^2w(t)\,\mathrm dt,\\
\operatorname{SIGReg}&=\frac1M\sum_{m=1}^M N D(\mathbf a_m).
\end{aligned}
$$

The squared complex difference is $\left(\frac1N\sum_n\cos(tz_n)-\varphi_{\mathcal N}(t)\right)^2+\left(\frac1N\sum_n\sin(tz_n)\right)^2$. Characteristic functions exist even without high-order moments. Sample means need no sorting and can be combined across workers using shared directions and grids. Unequal local batches require sample-count weighting; collectives must pass gradients. These are implementation requirements identified in the review, not executed DDP results. [§4.3; Algorithm 1](https://arxiv.org/pdf/2511.08544v3#page=10)

Theorem 4's appendix bounds derivatives of unscaled $D$ for $w(t)=\exp(-s^2t^2)$, using $|\hat\varphi|\le1$, $|\varphi_{\mathcal N}|\le1$ and $\int|t|w(t)\,\mathrm dt=1/s^2$:

$$
\begin{aligned}
\left|\frac{\partial D}{\partial z_n}\right|&\le\frac4{Ns^2},\\
\left|\frac{\partial(ND)}{\partial z_n}\right|&\le\frac4{s^2}.
\end{aligned}
$$

The scale matters when interpreting dependence on $N$. Differentiating network parameters adds encoder/projector Jacobians through the chain rule. <strong>A sample-loss derivative bound is not a bound on all parameter gradients or a global SGD convergence proof.</strong> [Theorem 4; Appendix B.12, pp. 37–38](https://arxiv.org/pdf/2511.08544v3#page=37)

## Sampling bias and finite frequency grids

For independent samples, $\exp(it z_n)$ has unit magnitude. Separating matching and distinct sample indices gives: [Theorem 6; Appendix B.13, pp. 38–40](https://arxiv.org/pdf/2511.08544v3#page=38)

$$
\mathbb E|\hat\varphi(t)-\varphi_{\mathcal N}(t)|^2
=|\varphi(t)-\varphi_{\mathcal N}(t)|^2+\frac{1-|\varphi(t)|^2}{N}.
$$

The extra term is $\mathcal O(1/N)$ for the unscaled discrepancy. Under conditions allowing differentiation under expectation, related gradient bias is analyzed at that scale. Multiplication by $N$, as in Algorithm 1, makes the <strong>absolute extra term O(1)</strong>. Claims of vanishing absolute bias must therefore specify normalization. A finite Gaussian batch also does not have exactly zero loss.

A review-authored Monte Carlo check used $N=32$, standard-Gaussian samples, $t=1$ and 20,000 replicates. Mean unscaled squared discrepancy was $0.0198677$; theory predicts $(1-e^{-1})/32=0.0197538$, with Monte Carlo standard error $0.0001505$. This checks a small formula, not neural-network training.

Quadrature creates another approximation. Algorithm 1 uses 17 equally spaced frequencies on $[-5,5]$, spacing $\Delta =0.625$. In a review-authored 1D example, translating a Gaussian by $2\pi/\Delta\approx10.0531$ leaves its CF unchanged at every grid point: frequencies are integer multiples of $\Delta$, so the phase factor is one.

$$
\begin{aligned}
\max_{t\in\mathrm{grid}}|\varphi(t)-\varphi_{\mathcal N}(t)|&\approx2.24\times10^{-16},\\
|\varphi(0.3)-\varphi_{\mathcal N}(0.3)|&\approx1.9082.
\end{aligned}
$$

This shows that a <strong>fixed 1D frequency grid cannot certify full distributional equality</strong>. It does not refute population CF uniqueness and does not demonstrate failure of high-dimensional training with resampled directions. Increasing sample count alone does not remove gaps in a fixed quadrature grid.

## Projection count and the meaning of linear complexity

Dense projection multiplies $N\times K$ by $K\times M$; CF evaluation uses $T$ frequencies for each projected value:

$$
\begin{aligned}
\text{time}&:\;\mathcal O(NKM+NMT),\\
\text{intermediate CF tensor}&:\;\mathcal O(NMT).
\end{aligned}
$$

With $K,M,T$ fixed, cost is linear in $N$. It is linear in $K$ only with $M$ fixed; if $M\propto K$, projection becomes $\mathcal O(NK^{2})$. Encoder computation and optimizer storage are outside these expressions. Gaussian-weighted CF/kernel connections are useful, but linear-time MMD estimators predate this work. [§4.3, p. 9; Gretton et al., abstract](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf#page=1)

Theorem 5 includes a rate of the form $M^{-2\alpha/(K-1)}$ under smoothness $\alpha$, quasi-uniform directional coverage and exact projection constraints. Constants and density norms also depend on dimension and smoothness. At fixed $\alpha$, the exponent worsens with dimension. This is not unconditional removal of the curse of dimensionality. Random finite-slice training curves provide a different kind of evidence. [Appendix B.10, pp. 36–37](https://arxiv.org/pdf/2511.08544v3#page=36)

![LeJEPA Figure 7: discrepancy versus projection count for fixed and resampled directions.](/assets/reviews/lejepa/paper-figure-7.webp)

[Enlarge figure](/assets/reviews/lejepa/paper-figure-7.webp)

<strong>Figure 7 explained.</strong> The horizontal axis is projection count $M$ on a logarithmic scale; the vertical axis is expected projected discrepancy. Green uses resampled directions, blue fixed directions. Resampling reaches lower discrepancy at smaller $M$ in this experiment. The fitted $\boldsymbol\beta$ and $R^{2}$ summarize this experiment rather than guarantee convergence on arbitrary data. The graph is not downstream accuracy. Attribution: Balestriero and LeCun, [Figure 7, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Complete graph/legend cropped and encoded as lossless WebP.

Table 6 reports V100 loss timings of $0.465236\pm 0.011642 \,\mathrm{ms}$ at $N=512,M=512,T=16$ and $6.188304\pm 0.007226 \,\mathrm{ms}$ at $N=8192$. Section 4.4 describes forward–backward timing. These are SIGReg measurements, not full pretraining time. The table omits embedding dimension $K$ and separate forward/backward times, so precise matched-system comparisons need more detail. [§4.4, p. 10; Table 6, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

## A checked equation discrepancy: pairwise and center distances

Equations 5–7 and Appendix B.6 describe replacing average distances between all views and global views with distances to their global mean. For one image, let $\boldsymbol\mu=\frac1{V_g}\sum_{g=1}^{V_g}\mathbf z_g$. The exact identity is:

$$
\begin{aligned}
A&=\frac1{VV_g}\sum_{v=1}^V\sum_{g=1}^{V_g}\|\mathbf z_v-\mathbf z_g\|_2^2,\\
B&=\frac1V\sum_{v=1}^V\|\mathbf z_v-\boldsymbol\mu\|_2^2,\\
C&=\frac1{V_g}\sum_{g=1}^{V_g}\|\mathbf z_g-\boldsymbol\mu\|_2^2,\\
A&=B+C.
\end{aligned}
$$

Expanding $\mathbf z_v-\mathbf z_g=(\mathbf z_v-\boldsymbol\mu)-(\mathbf z_g-\boldsymbol\mu)$ removes the cross term after summing over global views, but leaves $C$. This term depends on model parameters and cannot generally be dropped as a constant.

In the review's scalar example, both the all-view and global-view sets are $[0,2]$: $\mu =1$, $A=2$, $B=1$, $C=1$. The rendered PDF was checked directly, including Appendix Equations 23→24; the observation is not attributed to OCR. [§5, p. 12; Appendix B.6, p. 34](https://arxiv.org/pdf/2511.08544v3#page=34)

The identity above uses vector squared distances. Algorithm 2's `.square().mean()` also averages feature dimension $K$, so its agreement loss is $B/K$; the missing term is correspondingly $C/K$. This remains a clearly defined center objective. The review explains it and flags the claimed exact equivalence with $A$. <strong>The discrepancy does not by itself invalidate reported accuracies or reveal the exact code used for every experiment.</strong> Comparing both objectives under matched training conditions is a separate research question.

## Distinguishing the PDF from the current official example

The official README and MINIMAL at [commit c293d291](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052) were read but not run.

| Choice | Paper | Inspected MINIMAL example |
| --- | --- | --- |
| Frequency grid | Algorithm 1: 17 points on $[-5,5]$ | 17 points on $[0,3]$, doubled weights using symmetry |
| Directions | Algorithm 1 default 256; §6.1 recommends 1,024 | 256 |
| Agreement center | Algorithm 2: global-view mean | All-view mean |
| Teacher/detach | Absent from core design | Also absent from core agreement |

Positive-half integration is a reasonable use of symmetry, but changed endpoints and spacing make it a different quadrature. The example also postdates PDF v3. The ten-view default and eight-view Experiment Details 1 configuration must likewise be recorded separately.

## What the experiments test about the theory

Table 1 assesses trainability and sensitivity; Figure 9 expands architecture coverage; Table 2 shows use across downstream tasks; Figure 12/Table 3 examine small specialized domains. None certifies all density smoothness or task-prior assumptions for learned representations.

The ImageNet-1K transfer averages at one shot/ten shots/all labels are 29.55/60.95/79.48% for LeJEPA ViT-L, 30.20/60.51/78.50% for plain I-JEPA and 32.05/62.92/80.70% for +STOP. Model/training conditions are 304M/100 epochs versus 632M/300 epochs. These support a promising method with conditional comparisons, not unconditional superiority or exactly threefold total compute savings. [Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16)

Proposed follow-ups are to evaluate held-out directions and denser frequency grids, compare center and exact pairwise losses under the same settings, and compare alternative objectives with matched backbone/augmentations/compute. Keep frozen and finetuned results and seed variation separate. Potential counterexamples include low training loss with poor unseen projections, or task-relevant anisotropy benefiting a particular target. These experiments were not run here.

## Sources, coverage and validation status

- Paper: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, Randall Balestriero and Yann LeCun, first released 2025-11-11; reviewed v3, revised 2025-11-14. [Official metadata](https://arxiv.org/abs/2511.08544).
- First pass: pp. 1–3, 17–18 and the section map. Second pass: main text pp. 1–18, Appendices A–F pp. 24–46, supplementary Figures 16–21 pp. 47–50. Bibliography pp. 18–23 checked selectively. Key formulas, tables and axes were also compared with rendered PDF pages.
- Related-source coverage: VICReg v3 pp. 3–4; DINO v2 pp. 2–3; I-JEPA v3 pp. 3–4; Gaussian Embeddings v1 pp. 1, 3–4; Gretton et al. 2012 PDF pp. 1, 3–4. Comparisons appear in the [complete explanation](/en/reviews/lejepa-complete/).
- Hero: original conceptual diagram, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), Light (300) default and Semi Bold (600) emphasis. Paper captions record author attribution, location, CC BY-SA 4.0 and crop/encoding changes.
- <strong>Executed:</strong> four review-authored checks—center-distance identity, fixed 1D grid alias, OLS reciprocal-eigenvalue sum, and ECF sampling-bias Monte Carlo. Their results are limited formula illustrations, not model reproduction.
- <strong>Not run:</strong> official code, GPU pretraining, downstream probing/finetuning, DDP, repeated-seed training, paper runtime/memory remeasurement and full formal proof verification.
