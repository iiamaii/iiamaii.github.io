---
title: "[3/3] LeJEPA technical review: proof conditions, SIGReg and checked equations"
description: "Examines Gaussian minimality, projected identification, ECF gradients and bias, computational cost and a missing term in the distance-loss derivation."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-09T23:59:47+09:00"
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

This is the technical version. The [introductory review](/en/reviews/lejepa-overview/) explains the intuition; the [complete explanation](/en/reviews/lejepa-complete/) maps related work and experiments.

<strong>The hero's sample axes.</strong> Views of one image align with their global-view mean. Distribution regularization operates across different images in each view, after the projector. It does not constrain every backbone layer or the crop set of one photograph to be Gaussian. Both branches receive gradients; core agreement has no stop-gradient. [Enlarge](/assets/reviews/lejepa/lejepa-core.svg)

## Notation and levels of analysis

`N`: different images per batch. `V`: views per image. `Vg`: global views. `K`: projected embedding dimension. `M`: projection directions. `T`: frequency-grid points. `zn,v` is an encoder-plus-projector embedding.

| Level | Object of the claim | Additional requirement for training |
| --- | --- | --- |
| Downstream estimation | Fixed-design linear regression; local/kernel prediction under smooth density | Features, tasks and distributions must satisfy assumptions |
| Distribution identification | All directions/frequencies, or specified growing-set/test conditions | Finite directions and grids leave unobserved differences |
| Training loss | Finite-batch ECF discrepancy averaged across directions | Optimizer, encoder Jacobian, numerical error, generalization |
| Experiments | Particular data, models and evaluation protocols | Matched-budget reproduction, new domains, seed variation |

Equations are restated with some notation changes for readability. Review-authored examples are distinguished from reported paper results.

## Linear analysis: why equal directional variance helps

Assume a fixed full-column-rank feature matrix `X`, labels `y=Xβ+ε`, conditional zero-mean noise and noise covariance `σ²I`. Write `G=XᵀX` and ridge coefficient `η>0`. [Lemma 1; Appendix B.1, p. 26](https://arxiv.org/pdf/2511.08544v3#page=26)

```text
β̂ridge = (G + ηI)⁻¹Xᵀy
E[β̂ridge | X] − β = −η(G + ηI)⁻¹β
```

Along an eigenvector of `G`, shrinkage is `η/(λk+η)`. A weak direction therefore creates an unfavorable task relative to an isotropic design with the same total energy. This is an existence argument, not a claim that isotropy improves every specific task.

For OLS (`η=0`), coefficient variance gives the next argument. [Lemma 2; Appendix B.2, p. 27](https://arxiv.org/pdf/2511.08544v3#page=27)

```text
Cov(β̂OLS | X) = σ²G⁻¹
tr Cov = σ² Σk 1/λk
With Σk λk = c > 0: Σk 1/λk ≥ K²/c
Equality: every λk = c/K
```

Convexity of the reciprocal function, or Cauchy–Schwarz, proves the last inequality. In the review's two-dimensional calculation, eigenvalues `(1,1)` give reciprocal sum 2; `(0.2,1.8)` give approximately 5.5556 despite the same trace of 2.

This motivates <strong>isotropic covariance</strong>. It does not uniquely determine a Gaussian joint density, guarantee prediction risk under arbitrary test distributions, or establish learned-encoder performance.

## Nonlinear analysis: which term does Fisher information control?

Neighborhood averaging predicts from nearby labels; Nadaraya–Watson regression uses kernel weights. Small-radius/bandwidth expansions combine target curvature with density variation. The key kinds of terms are: [§3.2; Appendix A, B.3–B.7, pp. 24–35](https://arxiv.org/pdf/2511.08544v3#page=24)

```text
Density–task interaction: ∇m(x) · ∇log p(x)
Target curvature: Δm(x) / 2
J(p) = ∫ ||∇log p(x)||² p(x) dx
```

Here `m` is the target function and `∇log p` is the density score. If a task-gradient prior has zero mean and second moment `τg²I`, the squared score contribution becomes `τg²J(p)`.

Appendix B.4 additionally discusses decorrelation between gradient and curvature terms. Without it, a cross term remains `O(r⁴)`, the same order as other squared-bias terms. It cannot simply be treated as lower order. Interpreting curvature as independent of `p` also needs task-prior conditions. A controlled bias component is consequently not unconditional unique minimization of total bias. [Appendix B.4, p. 29](https://arxiv.org/pdf/2511.08544v3#page=29)

The kernel argument bounds worst-case bias using a quantity containing `2B²+8L²J(p)`, with `L,B` target smoothness bounds. It minimizes an upper bound, not the exact risk of every target. Pointwise variance depends on query density; cancellation in an integrated variance requires integrability and compatible train/query distributions. [Appendix B.7, pp. 34–35](https://arxiv.org/pdf/2511.08544v3#page=34)

## The essential Gaussian-minimality proof

Assume zero mean, covariance `Σ≻0`, a smooth density and sufficient boundary decay. With score `u(x)=∇log p(x)`, integration by parts gives `E[u(X)Xᵀ]=−I`. Non-negativity of a squared norm yields: [Appendix A; B.5, pp. 32–33](https://arxiv.org/pdf/2511.08544v3#page=32)

```text
0 ≤ E ||u(X) + Σ⁻¹X||²
  = J(p) − tr(Σ⁻¹)
Therefore J(p) ≥ tr(Σ⁻¹)
```

Equality requires `u(x)=−Σ⁻¹x` almost everywhere. Integrating gives `log p(x)=constant−½xᵀΣ⁻¹x`: a Gaussian density. With `trΣ=c` fixed, `trΣ⁻¹≥K²/c`, with equality at `Σ=(c/K)I`. Normalizing the target covariance to `I` gives the standard isotropic Gaussian.

This proves minimality of the Fisher objective under the conditions. Whether deterministic embeddings of a finite image dataset form a smooth full-dimensional density, and whether actual downstream tasks follow the prior, remain separate questions.

## Why all Gaussian projections identify the vector law

The vector characteristic function is `φZ(t)=E exp(i tᵀZ)`. Every nonzero `t` can be written `t=s a` for a unit direction `a`. Agreement for every scalar frequency in every direction therefore implies agreement of vector characteristic functions at every `t`. Uniqueness of characteristic functions identifies the distribution. [§4.1; Appendix B.8, p. 35](https://arxiv.org/pdf/2511.08544v3#page=35)

![LeJEPA Figure 5: Gaussian-looking coordinate marginals do not establish Gaussianity of an X-shaped joint density.](/assets/reviews/lejepa/paper-figure-5.webp)

[Enlarge figure](/assets/reviews/lejepa/paper-figure-5.webp)

<strong>Connecting Figure 5 to the proof.</strong> The left histograms look Gaussian, but the X-shaped joint density gives different projections along diagonal directions. Third-panel rows correspond to colored directions; the final panel compares statistical responses. “All directions” cannot be replaced by a few coordinate moments. Attribution: Balestriero and LeCun, [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Complete panels cropped, rasterized and encoded as lossless WebP.

The maximum test statistic in Equation 4 also differs from the averaged training objective. Appendix B.9 uses growing dense direction sets, calibrated global null thresholds and power over a separating neighborhood. A finite directional average is not automatically a level-α test. Failure to reject a null is not proof of Gaussianity.

Candidates also include CDF distances and order-statistic tests such as Shapiro–Wilk. They require sorting/order relationships and different distributed communication. Sorting is not nowhere differentiable: it can be differentiated piecewise while order remains fixed. Order-changing boundaries and its computational structure differ from ECF averaging. Appendices E–F supply definitions of comparison statistics. [§4.2, pp. 7–9; Appendices E–F, pp. 43–46](https://arxiv.org/pdf/2511.08544v3#page=7)

## ECF discrepancy: computation and gradient control

For projected samples `sn=aᵀzn`, define an unscaled discrepancy `D`:

```text
φ̂a(t) = (1/N) Σn [cos(t sn) + i sin(t sn)]
ψ(t) = exp(−t²/2)
D(a) = ∫ |φ̂a(t) − ψ(t)|² w(t) dt
SIGReg = (1/M) Σm N D(am)    # Algorithm 1's scale
```

The squared complex difference is `(mean cosine−ψ)²+(mean sine)²`. Characteristic functions exist even without high-order moments. Sample means need no sorting and can be combined across workers using shared directions and grids. Unequal local batches require sample-count weighting; collectives must pass gradients. These are implementation requirements identified in the review, not executed DDP results. [§4.3; Algorithm 1](https://arxiv.org/pdf/2511.08544v3#page=10)

Theorem 4's appendix bounds derivatives of unscaled `D` for `w(t)=exp(−s²t²)`, using `|φ̂|≤1`, `|ψ|≤1` and `∫|t|w(t)dt=1/s²`:

```text
|∂D/∂sn| ≤ 4/(N s²)
For ND: |∂(ND)/∂sn| ≤ 4/s²
```

The scale matters when interpreting dependence on `N`. Differentiating network parameters adds encoder/projector Jacobians through the chain rule. <strong>A sample-loss derivative bound is not a bound on all parameter gradients or a global SGD convergence proof.</strong> [Theorem 4; Appendix B.12, pp. 37–38](https://arxiv.org/pdf/2511.08544v3#page=37)

## Sampling bias and finite frequency grids

For independent samples, `exp(it sn)` has unit magnitude. Separating matching and distinct sample indices gives: [Theorem 6; Appendix B.13, pp. 38–40](https://arxiv.org/pdf/2511.08544v3#page=38)

```text
E |φ̂(t) − ψ(t)|²
 = |φ(t) − ψ(t)|² + (1 − |φ(t)|²)/N
```

The extra term is `O(1/N)` for the unscaled discrepancy. Under conditions allowing differentiation under expectation, related gradient bias is analyzed at that scale. Multiplication by `N`, as in Algorithm 1, makes the <strong>absolute extra term O(1)</strong>. Claims of vanishing absolute bias must therefore specify normalization. A finite Gaussian batch also does not have exactly zero loss.

A review-authored Monte Carlo check used `N=32`, standard-Gaussian samples, `t=1` and 20,000 replicates. Mean unscaled squared discrepancy was `0.0198677`; theory predicts `(1−e⁻¹)/32=0.0197538`, with Monte Carlo standard error `0.0001505`. This checks a small formula, not neural-network training.

Quadrature creates another approximation. Algorithm 1 uses 17 equally spaced frequencies on `[-5,5]`, spacing `Δ=0.625`. In a review-authored 1D example, translating a Gaussian by `2π/Δ≈10.0531` leaves its CF unchanged at every grid point: frequencies are integer multiples of `Δ`, so the phase factor is one.

```text
Maximum CF difference on the grid: approximately 2.24×10⁻¹⁶
CF difference at off-grid t=0.3: approximately 1.9082
```

This shows that a <strong>fixed 1D frequency grid cannot certify full distributional equality</strong>. It does not refute population CF uniqueness and does not demonstrate failure of high-dimensional training with resampled directions. Increasing sample count alone does not remove gaps in a fixed quadrature grid.

## Projection count and the meaning of linear complexity

Dense projection multiplies `N×K` by `K×M`; CF evaluation uses `T` frequencies for each projected value:

```text
Time: O(NKM + NMT)
Naive intermediate CF tensor: O(NMT)
```

With `K,M,T` fixed, cost is linear in `N`. It is linear in `K` only with `M` fixed; if `M∝K`, projection becomes `O(NK²)`. Encoder computation and optimizer storage are outside these expressions. Gaussian-weighted CF/kernel connections are useful, but linear-time MMD estimators predate this work. [§4.3, p. 9; Gretton et al., abstract](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf#page=1)

Theorem 5 includes a rate of the form `M^(−2α/(K−1))` under smoothness `α`, quasi-uniform directional coverage and exact projection constraints. Constants and density norms also depend on dimension and smoothness. At fixed `α`, the exponent worsens with dimension. This is not unconditional removal of the curse of dimensionality. Random finite-slice training curves provide a different kind of evidence. [Appendix B.10, pp. 36–37](https://arxiv.org/pdf/2511.08544v3#page=36)

![LeJEPA Figure 7: discrepancy versus projection count for fixed and resampled directions.](/assets/reviews/lejepa/paper-figure-7.webp)

[Enlarge figure](/assets/reviews/lejepa/paper-figure-7.webp)

<strong>Figure 7 explained.</strong> The horizontal axis is projection count `M` on a logarithmic scale; the vertical axis is expected projected discrepancy. Green uses resampled directions, blue fixed directions. Resampling reaches lower discrepancy at smaller `M` in this experiment. The fitted `β` and `R²` summarize this experiment rather than guarantee convergence on arbitrary data. The graph is not downstream accuracy. Attribution: Balestriero and LeCun, [Figure 7, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Complete graph/legend cropped and encoded as lossless WebP.

Table 6 reports V100 loss timings of `0.465236±0.011642 ms` at `N=512,M=512,T=16` and `6.188304±0.007226 ms` at `N=8192`. Section 4.4 describes forward–backward timing. These are SIGReg measurements, not full pretraining time. The table omits embedding dimension `K` and separate forward/backward times, so precise matched-system comparisons need more detail. [§4.4, p. 10; Table 6, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

## A checked equation discrepancy: pairwise and center distances

Equations 5–7 and Appendix B.6 describe replacing average distances between all views and global views with distances to their global mean. For one image, let `μ=(1/Vg)Σg zg`. The exact identity is:

```text
A = (1/VVg) Σv,g ||zv − zg||²
B = (1/V) Σv ||zv − μ||²
C = (1/Vg) Σg ||zg − μ||²
A = B + C
```

Expanding `zv−zg=(zv−μ)−(zg−μ)` removes the cross term after summing over global views, but leaves `C`. This term depends on model parameters and cannot generally be dropped as a constant.

In the review's scalar example, both the all-view and global-view sets are `[0,2]`: `μ=1`, `A=2`, `B=1`, `C=1`. The rendered PDF was checked directly, including Appendix Equations 23→24; the observation is not attributed to OCR. [§5, p. 12; Appendix B.6, p. 34](https://arxiv.org/pdf/2511.08544v3#page=34)

The identity above uses vector squared distances. Algorithm 2's `.square().mean()` also averages feature dimension `K`, so its agreement loss is `B/K`; the missing term is correspondingly `C/K`. This remains a clearly defined center objective. The review explains it and flags the claimed exact equivalence with `A`. <strong>The discrepancy does not by itself invalidate reported accuracies or reveal the exact code used for every experiment.</strong> Comparing both objectives under matched training conditions is a separate research question.

## Distinguishing the PDF from the current official example

The official README and MINIMAL at [commit c293d291](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052) were read but not run.

| Choice | Paper | Inspected MINIMAL example |
| --- | --- | --- |
| Frequency grid | Algorithm 1: 17 points on `[-5,5]` | 17 points on `[0,3]`, doubled weights using symmetry |
| Directions | Algorithm 1 default 256; §6.1 recommends 1,024 | 256 |
| Agreement center | Algorithm 2: global-view mean | All-view mean |
| Teacher/detach | Absent from core design | Also absent from core agreement |

Positive-half integration is a reasonable use of symmetry, but changed endpoints and spacing make it a different quadrature. The example also postdates PDF v3. The ten-view default and eight-view Experiment Details 1 configuration must likewise be recorded separately.

## What the experiments test about the theory

Table 1 assesses trainability and sensitivity; Figure 9 expands architecture coverage; Table 2 shows use across downstream tasks; Figure 12/Table 3 examine small specialized domains. None certifies all density smoothness or task-prior assumptions for learned representations.

The ImageNet-1K transfer averages at one shot/ten shots/all labels are `29.55/60.95/79.48%` for LeJEPA ViT-L, `30.20/60.51/78.50%` for plain I-JEPA and `32.05/62.92/80.70%` for +STOP. Model/training conditions are 304M/100 epochs versus 632M/300 epochs. These support a promising method with conditional comparisons, not unconditional superiority or exactly threefold total compute savings. [Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16)

Proposed follow-ups are to evaluate held-out directions and denser frequency grids, compare center and exact pairwise losses under the same settings, and compare alternative objectives with matched backbone/augmentations/compute. Keep frozen and finetuned results and seed variation separate. Potential counterexamples include low training loss with poor unseen projections, or task-relevant anisotropy benefiting a particular target. These experiments were not run here.

## Sources, coverage and validation status

- Paper: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, Randall Balestriero and Yann LeCun, first released 2025-11-11; reviewed v3, revised 2025-11-14. [Official metadata](https://arxiv.org/abs/2511.08544).
- First pass: pp. 1–3, 17–18 and the section map. Second pass: main text pp. 1–18, Appendices A–F pp. 24–46, supplementary Figures 16–21 pp. 47–50. Bibliography pp. 18–23 checked selectively. Key formulas, tables and axes were also compared with rendered PDF pages.
- Related-source coverage: VICReg v3 pp. 3–4; DINO v2 pp. 2–3; I-JEPA v3 pp. 3–4; Gaussian Embeddings v1 pp. 1, 3–4; Gretton et al. 2012 PDF pp. 1, 3–4. Comparisons appear in the [complete explanation](/en/reviews/lejepa-complete/).
- Hero: original conceptual diagram, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), Light (300) default and Semi Bold (600) emphasis. Paper captions record author attribution, location, CC BY-SA 4.0 and crop/encoding changes.
- <strong>Executed:</strong> four review-authored checks—center-distance identity, fixed 1D grid alias, OLS reciprocal-eigenvalue sum, and ECF sampling-bias Monte Carlo. Their results are limited formula illustrations, not model reproduction.
- <strong>Not run:</strong> official code, GPU pretraining, downstream probing/finetuning, DDP, repeated-seed training, paper runtime/memory remeasurement and full formal proof verification.
