---
title: "[2/2] HNC in Depth: Why Function Preservation Works, and Where It Fails"
description: "Follow appendices A–J through Gauss–Newton derivations, soft projection, compute costs, ablations, and off-probe validation."
date: "2026-10-17"
publishedAt: "2026-10-17T22:11:59+09:00"
updatedAt: "2026-10-23T08:56:22+00:00"
topics: ["representation learning", "loss landscape", "mechanistic interpretability", "computational neuroscience", "reinforcement learning"]
visibility: "public"
lang: "en"
translationKey: "hnc-mechanics"
paperTitle: "Traversing the solution space of neural networks with Hessian Null Space Continuation"
paperPublishedDate: "2026-09-29"
authors: ["Ann Huang", "Mitchell Ostrow", "Zhouyang Lu", "William T. Redman", "Leo Kozachkov", "Kanaka Rajan"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2609.38081v1"
thumbnail: "/assets/reviews/hnc/overview.webp"
thumbnailAlt: "Original Figure 1: HNC alternates low-curvature exploration and output restoration, with optional steering."
---

HNC starts from the anchor’s output Jacobian and Gauss–Newton operator, then combines small steps, restoration, and relinearization. For a 4,611-parameter RNN, the matrix-free route lowers storage from 85 to 1.2 MB but increases computation from 10 to 39 seconds. [Appendix A, Table 1](https://arxiv.org/pdf/2609.38081v1#page=19) **Review interpretation.** Computing local preservation directions at scale is the central benefit, with memory, time, and off-probe drift assessed together.

Read [\[1/2\] Methods and experiments](/en/reviews/hnc-solutions/) first. This article follows **appendices A→B→C→D→E→F→G→H→I→J**. The paper has no numbered theorem establishing global function equivalence or convergence of the entire HNC procedure. Its central mathematical argument is the derivative calculation in B. Expanded calculations and examples below are identified as **review supplements**.

## A. Hessian Null-Space Continuation Pseudocode

### Algorithms 1–3: What Is Actually Accepted?

Inputs include anchor $\boldsymbol\theta_0$, probes $\mathbf X$, exploration size $\eta$, restoration rate $\alpha$, restoration count $m$, ceiling $\tau$, and optional objective $\varphi$. Algorithm 1:

1. Computes a direction at the current weights and normalizes it.
2. Proposes a small step.
3. Applies $m$ gradient steps reducing function-matching loss.
4. Rejects a proposal still exceeding $\tau$, keeps the old weights, shrinks the step, and refreshes directions.
5. Records only accepted proposals as new checkpoints.

**Not every proposed point is a preserved solution.** An accepted checkpoint passes a specified finite-probe loss check. The procedure also does not certify every point on a continuous interpolation between checkpoints.

Algorithm 2 uses products $\mathbf G\mathbf v=\mathbf J^\top(\mathbf J\mathbf v)$ to obtain $k$ low-eigenvalue directions. Algorithm 3 samples an undirected direction or computes a soft-projected steering direction. The RNN undirected configuration keeps one random heading and reprojects it into refreshed bases; it is not a fresh independent uniform direction at every step. [A, p.18; F.3](https://arxiv.org/pdf/2609.38081v1#page=18)

### A.1 Computational Cost

| Original Table 1 | Dense eigendecomposition | Matrix-free route |
| --- | --- | --- |
| Time | $O(P^3)$ | $O(kP)$ for fixed block size and iteration count |
| Memory | $O(P^2)$ | $O(kP)$ |
| 3BFF RNN, $P=4611$ | 85 MB; 10 seconds | 1.2 MB; 39 seconds; $k=64$ |
| ViT-S, $P\approx22$M | Approximately 1.9 PB to store the Hessian; impractical factorization | Approximately 5.6 GB for a $k=64$ block; about 1 second per product and 1 minute per continuation step |

RNN times use an eight-thread CPU node; ViT times use an NVIDIA H200 and 512 probes. This is not a hardware-matched speed comparison. $O(kP)$ hides the cost of probes, network computation, and solver iterations. Steering also requires CG and restoration. The block storage estimate should not be interpreted as total peak GPU memory for the complete execution. Dense computation is faster for the small RNN. [A.1, Table 1, pp.18–19](https://arxiv.org/pdf/2609.38081v1#page=19)

### A.2 Finding the Flattest Directions by LOBPCG

Equation (4) gives the Rayleigh quotient:

$$
\rho(\mathbf v)=
\frac{\mathbf v^\top\mathbf H\mathbf v}
{\mathbf v^\top\mathbf v}.
$$

In an orthonormal eigenbasis of a symmetric matrix, write $\mathbf v=\sum_i a_i\mathbf v_i$. The numerator is $\sum_i\lambda_i a_i^2$ and the denominator $\sum_i a_i^2$: a weighted average of eigenvalues, minimized at the smallest eigenvalue. This **supplementary calculation** expands the paper's minimization explanation.

LOBPCG updates a block of mutually orthogonal directions using current directions, residuals, and previous directions. It works through products rather than the full matrix. Finding $k$ vectors does not establish that only $k$ flat directions exist. The original LOBPCG paper and its complete convergence proof were not independently reviewed here; this account checks HNC's A.2 and the symmetric-matrix identity.

### A.3 Solving the Soft Projection by Conjugate Gradients

The implemented direction solves:

$$
(\mathbf I+\mathbf G/\mu)\mathbf d
=\nabla\varphi,\qquad \mu>0.
$$

Because $\mathbf G$ is positive semidefinite (PSD), for nonzero $\mathbf v$:

$$
\mathbf v^\top(\mathbf I+\mathbf G/\mu)\mathbf v
=\|\mathbf v\|^2+\|\mathbf J\mathbf v\|^2/\mu>0.
$$

The system matrix is therefore positive definite. CG approximately solves it without storing an inverse. The paper uses an iteration budget with early exit at relative residual $10^{-8}$, generally warm-starting from the previous direction. The 3BFF steering experiments instead start from zero with a budget of 200 iterations; ViT uses 40 preconditioned-CG iterations.

**Supplementary interpretation:** with $\mathbf g=\nabla\varphi$, the direction uniquely minimizes:

$$
\begin{aligned}
Q(\mathbf d)&=\frac12\|\mathbf d-\mathbf g\|^2
+\frac1{2\mu}\mathbf d^\top\mathbf G\mathbf d,\\
\underset{\mathbf d}{\arg\min}\,Q(\mathbf d)
&=(\mathbf I+\mathbf G/\mu)^{-1}\mathbf g.
\end{aligned}
$$

$Q$ is a review-added objective name. Differentiating gives the preceding linear system. It stays close to the desired gradient while penalizing first-order output sensitivity. This does not prove monotonic improvement of the steering objective after direction normalization, finite movement, and restoration. [A.2–A.3, p.19](https://arxiv.org/pdf/2609.38081v1#page=19)

## B. Gauss–Newton approximation of the Hessian at loss minima

The paper stacks probe outputs into a residual vector:

$$
\begin{aligned}
\mathbf r(\boldsymbol\theta)
&=f_{\boldsymbol\theta}(\mathbf X)
-f_{\boldsymbol\theta_0}(\mathbf X),\\
\mathcal L(\boldsymbol\theta)&=\frac12\|\mathbf r(\boldsymbol\theta)\|^2,\\
\mathbf J&=\frac{\partial\mathbf r}{\partial\boldsymbol\theta}.
\end{aligned}
$$

This retains B's sum convention. A sample mean, as in main equation (1), contributes the same $1/N$ factor to gradient and Hessian. Absolute cutoffs cannot be transferred between normalizations without adjustment. With $\mathbf r\in\mathbb R^q$ and $\boldsymbol\theta\in\mathbb R^P$, $\mathbf J\in\mathbb R^{q\times P}$ and $\mathbf G\in\mathbb R^{P\times P}$; $q$ is a review-added count of output components.

**Expand the source's derivative sequence.** For parameter component $a$:

$$
\frac{\partial\mathcal L}{\partial\theta_a}
=\sum_k r_k\frac{\partial r_k}{\partial\theta_a}.
$$

Differentiate again with respect to $b$, retaining both product-rule terms:

$$
\begin{aligned}
\frac{\partial^2\mathcal L}{\partial\theta_b\partial\theta_a}
&=\sum_k
\frac{\partial r_k}{\partial\theta_b}
\frac{\partial r_k}{\partial\theta_a}\\
&\quad+\sum_k r_k
\frac{\partial^2r_k}{\partial\theta_b\partial\theta_a}.
\end{aligned}
$$

Collecting entries produces equation (5):

$$
\nabla^2\mathcal L(\boldsymbol\theta)
=\underbrace{\mathbf J^\top\mathbf J}_{\mathbf G}
+\sum_k r_k(\boldsymbol\theta)\nabla^2r_k(\boldsymbol\theta).
$$

At the anchor, $\mathbf r(\boldsymbol\theta_0)=\mathbf0$, so the second term vanishes, giving equation (6):

$$
\nabla^2\mathcal L(\boldsymbol\theta_0)
=\mathbf J^\top\mathbf J=\mathbf G.
$$

This identity assumes the required derivatives exist at the point considered. At another point, residual-weighted curvature can remain, so $\mathbf H=\mathbf G$ is not an identity along the entire walk. $\mathbf G$ remains PSD and measures first-order output sensitivity without negative-curvature directions. [B, equations (5)–(6), pp.19–20](https://arxiv.org/pdf/2609.38081v1#page=19)

### Why Does a Null Direction Not Guarantee Finite-Step Preservation?

**Review supplement:**

$$
\mathbf v^\top\mathbf G\mathbf v=\|\mathbf J\mathbf v\|^2,
\qquad\ker\mathbf G=\ker\mathbf J.
$$

A null direction has zero *first-order* output change. Consider the explanatory example $f(a,b)=(a+b,b^2)$ at anchor $(0,0)$ with unit direction $\mathbf v=(1,-1)/\sqrt2$:

$$
\begin{aligned}
\mathbf J(0,0)\mathbf v&=\mathbf0,\\
f(t\mathbf v)-f(0,0)&=(0,t^2/2),\\
\mathcal L(t\mathbf v)&=t^4/8.
\end{aligned}
$$

Even an exact Hessian null direction has nonzero loss for finite $t$. Indeed, this example's exact zero-loss set contains only the origin. A large local linear null space alone does not establish an equally dimensional global solution manifold. This motivates small steps, restoration, and acceptance checks. It is a review example, not an experiment in the paper.

<span id="c-ablation-study-on-the-hnc-algorithm" aria-hidden="true"></span>

## C. Ablation study on HNC algorithm

The source tests four components on 3BFF in C.1–C.4:

| Section | Change | Result and interpretation |
| --- | --- | --- |
| C.1 | 0,3,6,9,12 restoration steps at three step sizes | At large step size $\eta=0.3$, 12 corrections reduce endpoint loss from 6.6× to 3.6× anchor loss, not to zero drift. Small exploration steps matter. |
| C.2 | Soft-projection damping | $10^{-9}$ barely moves. Relaxation increases displacement and drift; removing projection breaks function within dozens of steps despite restoration. |
| C.3 | Refresh never or every 50,20,5 steps | More frequent relinearization lowers loss, especially for large steps. |
| C.4 | HNC versus similarity-penalized training | Over 600 steps and five seeds per setting, HNC reports greater CKA distance at a given loss level. Penalized training initially crosses high loss before returning. |

Matching step count and step length does not establish universal superiority at equal wall-clock cost including curvature and CG. Restoration limits drift rather than projecting exactly onto a constraint surface. [C, Figures 6–9, pp.20–22](https://arxiv.org/pdf/2609.38081v1#page=20)

## D. Preserving function versus task loss in HNC

First, a trained model's task gradient need not be exactly zero. A flat-direction step retains the first-order term $\eta\mathbf g^\top\mathbf v$. In the source's 3BFF comparison, it exceeds the second-order term by two to three orders of magnitude and task loss rises roughly tenfold. Function matching starts with exactly zero residual and gradient because it targets the anchor's own outputs.

Second, a full output vector imposes stronger constraints than one scalar loss. Original Table 2 compares **matched representation distances** on 512 ImageNet probes and 1,000 classes. Values are fractions retaining the anchor's top-1 prediction:

| Preserved quantity | Component constraints | $\Delta\mathrm{CKA}=0.2$ | 0.4 | 0.6 |
| --- | --- | --- | --- | --- |
| Full logits | 512,000 | 0.996 | 0.994 | 0.993 |
| Per-image cross-entropy | 512 | 0.916 | 0.894 | 0.875 |
| Mean cross-entropy | 1 | 0.831 | 0.811 | 0.793 |

Component counts are not claims of independent Jacobian rank. Multiple logit vectors can share a cross-entropy value and disagree on predictions. Per-image loss matching permits redistribution within each image's output distribution; matching only the mean additionally permits compensating loss changes across images.

![Original Figure 11: Comparing full-logit, per-image-loss, and mean-loss preservation](/assets/reviews/hnc/constraint-strength.webp)

[Enlarge Figure 11](/assets/reviews/hnc/constraint-strength.webp)

**Read B alongside D.** Solid lines in B are probe results, dashed lines held-out results. Full-logit matching, purple, preserves more agreement. But representation changes at different speeds in A, so matching steps is insufficient. D instead uses $\Delta\mathrm{CKA}$ horizontally. Accuracy in C and anchor agreement in B are distinct. Bands show the range across probe seeds, not universal confidence intervals. [D, Figure 11, Table 2, pp.22–25](https://arxiv.org/pdf/2609.38081v1#page=24)

To explore broader task-compatible solutions, one can constrain the squared difference from the anchor's task loss. That gives zero initial gradient without resolving the weaker output-information constraint. Finite-probe function matching needs no labels, but off-probe preservation still requires evaluation.

## E. Combining HNC with global search for alternative solutions

The authors alternate DSA-steered local walks with fresh training regularized to differ dynamically from earlier solutions. The eight-network chain reports bit accuracy above $0.999$ using fixed-point memories, continuing drift, and extra attractors. Local walks explore the current low-loss region; fresh training can jump far in weight space.

The retraining transitions are not certified as one continuous function-preserving path. The experiment does not find or count every basin. Looser ceilings to cross ridges and random-kick restarts are future proposals in this section. [E, Figures 12–14, pp.26–28](https://arxiv.org/pdf/2609.38081v1#page=26)

## F. Details for the 3BFF RNN experiments

### F.1–F.3 Settings and the Size of the Flat Subspace

The 64-unit RNN has 4,611 parameters, permitting a full spectrum. Across five anchors its Hessian participation ratio is $23.2\pm1.6$. This effective curvature dimension is not a count to subtract from parameter dimension to obtain nullity.

**What participation ratio measures.** Let $D_{\mathrm{PR}}$ be a review abbreviation for effective dimension. For a nonzero PSD spectrum,

$$
D_{\mathrm{PR}}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
$$

It equals $r$ for $r$ equal positive eigenvalues and approaches one when curvature concentrates in one. Tiny positive eigenvalues count toward rank but contribute little here. Thus 23.2 does not mean only 23 directions exist or every other direction is exactly null. An all-zero spectrum makes the ratio undefined. [Appendix F.2](https://arxiv.org/pdf/2609.38081v1#page=29)

Original Table 3 reports median flat dimensions at different relative thresholds:

| $\mu_{\mathrm{rel}}$ | Median dimension | Fraction of parameters |
| --- | --- | --- |
| $10^{-7}$ | 1701 | 36.9% |
| $10^{-6}$ | 2896 | 62.8% |
| $10^{-5}$ | 3684 | 79.9% |
| $10^{-4}$ | 4145 | 89.9% |
| $10^{-3}$ | 4418 | 95.8% |
| $10^{-2}$ | 4565 | 99.0% |

There is no single spectral gap defining one unambiguous dimension. Allowing more output drift labels more directions flat. CKA steering runs 600 steps and DSA steering 2,000, with 12 restoration steps each. The 32-trial steering probe differs from the 128-trial acceptance evaluation. [F.1–F.3, pp.29–30](https://arxiv.org/pdf/2609.38081v1#page=29)

### F.4–F.8 Diversity, Pruning, Resumed SGD, and OOD Inputs

Galleries and ten-anchor MDS show multiple alternative directions rather than one unusual endpoint; repeated snapshots are not reproduced here. Pruning targets recurrent/output outgoing weight groups and reduces one 64-unit network to 11 units with 100% evaluated bit accuracy, a small MSE increase, and approximately 5% of the original parameters. Counting the specified architecture gives 4,611→212 parameters.

After 8,000 resumed SGD steps, lower task loss coexists with retained differences in CKA, DSA, or sparsity. Ordinary training does not necessarily pull alternatives back to the anchor. OOD pulse amplitudes and rates, long input-free intervals, and state perturbations expose earlier failure in some endpoints. Diversity is not synonymous with increased robustness. [F.4–F.8, pp.30–35](https://arxiv.org/pdf/2609.38081v1#page=31)

## G. Additional experiments on ViT

### G.1–G.4 Steering Objectives and Settings

**What CKA compares.** Here $X,Y$ are review notation for column-centered feature matrices with matching sample order, distinct from probe inputs $\mathbf X$.

$$
\operatorname{CKA}_{\mathrm{linear}}(X,Y)
=\frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F\,\|Y^\top Y\|_F}.
$$

This linear CKA ranges from zero to one, with larger values indicating more similar sample geometry. It is invariant to orthogonal coordinate changes and isotropic scaling, not arbitrary invertible transforms. Constant features make the denominator zero. Lower CKA does not directly establish changed outputs or inequivalence of every computation. [CKA source §3–5, Eq. (14)](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf#page=5)

With $c_{ij}(\boldsymbol\theta)$ denoting linear CKA between anchor layer $i$ and current layer $j$, equation (12) is:

$$
\begin{aligned}
\varphi_{\mathrm{CKA}}(\boldsymbol\theta)
&=1-\frac1\beta\log\sum_{i,j=1}^{L}
\exp\!\bigl(\beta c_{ij}(\boldsymbol\theta)\bigr),\\
L&=13,\qquad\beta=30.
\end{aligned}
$$

Increasing it applies stronger downward pressure to the most similar remaining pairs. Log-sum-exp differs from hard max; reported evaluations use hard max. G.2's semantic, color, and layout scores are respectively DINOv2 cosine, HSV $8^3$ histogram cosine, and $32\times32$ grayscale correlation, not equivalent human judgments.

G.3's mutual-kNN metric measures overlap of $k=10$ neighbor sets and uses smooth neighborhood weights for differentiation. It tests whether a complementary metric also permits divergence. However, G.3 specifies 256 probes whereas G.4 Table 8 lists 512, so the exact $N$ of that kNN configuration remains unresolved. Differences in preservation cannot be attributed intrinsically to the metric alone.

### G.5–G.6 Held-Out Preservation and the Damage Control

CKA steering uses five probe draws and the kNN off-probe evaluation uses three. The diff-seed reference has two fine-tuning seeds; single-anchor and independently trained ViT-S/in1k bands instead use image bootstraps. Error bars therefore have different sampling units.

CKA endpoints have small accuracy losses and error consistency near 0.8 across ImageNet/V2/R/Sketch. The kNN configuration has error consistency near 0.75 and lower exact prediction agreement. Gaussian weight noise matched for accuracy loss produces smaller representational displacement. This weakens a generic-damage explanation without proving that a changed metric captures every computational change. [G.5–G.6, pp.39–42](https://arxiv.org/pdf/2609.38081v1#page=39)

Error consistency is chance-corrected agreement on whether both models are correct or both are wrong on each image. Unlike prediction match, agreement when wrong does not require the same incorrect class. A value near 0.8 therefore means neither 80% accuracy nor exactly 80% identical predictions. [Appendix G.5](https://arxiv.org/pdf/2609.38081v1#page=39)

### G.7–G.10 Layers, Directions, and Probe Size

An early block can be steered, but it costs more function drift per representation change than the pre-logit feature. Steering ViT-S toward ViT-B raises mean depthwise CKA from 0.78 to 0.96. It retains **96% of probe predictions**, approximately 90% on 5,000 held-out ImageNet images, and loses a little over one percentage point of top-1 accuracy. It does not become ViT-B with exactly identical function.

Larger probes can support better preservation and longer accepted walks, allowing more displacement in the tested configuration. Compute grows approximately linearly with probe count. This is not a set-theoretic claim that adding constraints always enlarges the feasible region. The final MDS shows alternatives depend on metric, probe draw, and steering away from previous endpoints. [G.7–G.10, pp.43–46](https://arxiv.org/pdf/2609.38081v1#page=43)

## H. Steering Brain-Score at fixed function

The paper's MajajHong2015 IT benchmark uses 3,200 images and 168 macaque IT recording sites. The reported official score uses ten-fold PLS-25 prediction, median correlation across sites, averaging across folds, and normalization by noise ceiling 0.82.

Steering uses a different differentiable surrogate: ridge regression on a frozen anchor-derived 64-dimensional PCA basis, maximizing or minimizing mean neural-response correlation on the other image half. The official PLS score is not differentiated directly.

An 86M relative-position ViT spans 0.618–0.719, approximately 0.10, between down/up endpoints; the range on unseen sites and images is 0.039. Plain ViT-S spans 0.546–0.627, or 0.081, on held-out images. The seven-model comparison spans approximately 0.074–0.075. Top-1 decisions are preserved on at least 94% of images, with logit KL in the $10^{-5}$–$10^{-3}$ range. **Approximate preservation is not exact equality.** Steering IT predictivity leaves V1/V2/V4 scores largely unchanged. Neural predictivity can depend on which solution is selected, not only architecture and original training. [H, pp.46–48](https://arxiv.org/pdf/2609.38081v1#page=46)

## I. Extending HNC to reinforcement learning

### I.1 Advantages and the Surrogate

Plume and Ant use GAE. With rewards $r_t$ and critic $V$:

$$
\begin{aligned}
\delta_t&=r_t+\gamma V(s_{t+1})-V(s_t),\\
A_t&=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},\\
\gamma&=0.99,\qquad\lambda=0.95.
\end{aligned}
$$

The next-state value is zero at episode termination. Boat race instead uses critic-free discounted proxy returns with $\gamma=0.9$. Standardizing advantages makes the reference surrogate average zero when all importance ratios are one. **A zero value does not imply a zero gradient.** The PSD reasoning for a supervised squared-residual minimum cannot simply be transferred to the RL surrogate. Buffer refresh and actual episode evaluation remain necessary.

### I.2 From Gaussian KL to Mean-Action MSE

For equal Gaussian covariances, trace and log-determinant terms in the source's general KL cancel:

$$
\begin{aligned}
D_{\mathrm{KL}}(\pi_{\boldsymbol\theta_0}\Vert\pi_{\boldsymbol\theta})
&=\frac12(\boldsymbol\mu_{\boldsymbol\theta}-\boldsymbol\mu_{\boldsymbol\theta_0})^\top
\boldsymbol\Sigma^{-1}
(\boldsymbol\mu_{\boldsymbol\theta}-\boldsymbol\mu_{\boldsymbol\theta_0})\\
&=\sum_{j=1}^d
\frac{(\mu_{\boldsymbol\theta,j}-\mu_{\boldsymbol\theta_0,j})^2}{2\sigma_j^2}.
\end{aligned}
$$

State arguments are omitted only in this block to shorten it. Covariance stays fixed throughout the walk. Unweighted mean-action MSE is proportional to KL only when all $\sigma_j$ agree; otherwise it omits fixed diagonal weights. This expands I.2's cancellation rather than introducing a new original theorem.

### I.3–I.7 Conditions and Failures in Three Environments

| Environment | Buffer/evaluation | Observation and limitation |
| --- | --- | --- |
| Plume | Eight current-policy episodes; refresh every five steps; 60 standard initial conditions | Less casting and retained return. OOD evaluation uses 240 initial conditions per condition, with SE across 8×30 batches. Performance collapses when the walk leaves the level set at step 20. |
| Boat race | 32 episodes; refresh every three steps; $\epsilon=0.25$ exploration; 100 evaluation episodes | Ten true-return-trained anchors. The three most divergent endpoints collect proxy reward with zero net progress. |
| Ant-v3 | 4,000 transitions; refresh every five steps; 30 fixed evaluation states | Action divergence reaches 1.0 in 19 steps. Matched-action-KL random directions lose over half the return. HNC preserves mean return while changing failures. |

Ant gait analysis uses walking episodes only: 25/30 for the anchor and 27/30 for the alternative. The latter stalls less but sometimes terminates early when torso height becomes unhealthy. Returns across shared initial states are reported as uncorrelated. Equal average performance is neither episodewise equivalence nor a safety guarantee. Each plume OOD condition uses its own initial-state grid, so paired comparisons within a condition differ from comparisons of absolute rates across conditions. [I, pp.49–53](https://arxiv.org/pdf/2609.38081v1#page=49)

## J. Probing loss-landscape geometry with HNC

Five widths, five class counts, and three seeds make 75 CNN anchors. GroupNorm avoids batch-composition-dependent outputs; 512 training inputs constrain each model. J.2 varies absolute drift tolerance and J.3 relative spectral cutoffs. J.4 finds feature-covariance participation ratio grows about fivefold with class count but changes little with width. **Feature effective dimension and parameter flat fraction are different measurements.**

J.1's identification of 100% training accuracy with zero cross-entropy is not adopted here. A correct top-1 prediction with target probability 0.6 still has positive cross-entropy. Matching one's own outputs gives zero anchor residual independently of this distinction.

**Review supplement:** with $q$ finite-probe output components, $\operatorname{rank}(\mathbf J)\le q$ and $\dim\ker\mathbf J\ge P-q$. Many parameters and relatively few probe constraints can themselves yield a large linear null space. It does not follow that the true input distribution permits the same freedom. Off-probe evaluation is essential. [J, pp.53–55](https://arxiv.org/pdf/2609.38081v1#page=53)

## Review interpretation and open questions

**Review interpretation.** Computing local preservation directions at scale is the central benefit, with memory, time, and off-probe drift assessed together.

## Sources and reading scope

- Paper: *Traversing the solution space of neural networks with Hessian Null Space Continuation* · Ann Huang et al. · first public date 2026-09-29 · [reviewed version](https://arxiv.org/abs/2609.38081v1) · reviewed 2026-10-23.
- Scope read: Version 1, appendices A–J: algorithms, derivative arguments, ablations, and validation conditions. Repeated galleries and exhaustive hyperparameter rows are compressed.
- Figure sources: The Figure-1 thumbnail and body figures in this part are author assets from v1 (CC BY 4.0), PDF-rasterized and WebP-encoded with content, layout, text, and arrows retained. [Paper](https://arxiv.org/abs/2609.38081v1).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
