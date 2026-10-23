---
title: "[2/2] HNC in Depth: Why Function Preservation Works, and Where It Fails"
description: "Follow appendices A–J through Gauss–Newton derivations, soft projection, compute costs, ablations, and off-probe validation."
date: "2026-10-17"
publishedAt: "2026-10-17T22:11:59+09:00"
updatedAt: "2026-10-23T15:26:22+00:00"
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

Hessian Null Space Continuation (HNC) relies on appendix B's derivation that the Hessian of the function-matching loss equals the Gauss–Newton operator $\mathbf G=\mathbf J^\top\mathbf J$ at the anchor, the trained reference model, where $\mathbf J$ is the Jacobian of the probe outputs. Using this operator only through products, HNC finds flat directions in ViT-S, a 22M-parameter vision Transformer (ViT), with a $k=64$ block of about 5.6 GB instead of the roughly 1.9 PB a dense Hessian would need, and one continuation step takes about a minute on an H200. [Appendices A–B, Table 1](https://arxiv.org/pdf/2609.38081v1#page=19) **Review interpretation.** The derivation guarantees only $\mathbf H=\mathbf G$ and positive semidefiniteness at the anchor itself; preservation along the path rests on empirical safeguards (small steps, restoration, and a loss ceiling), and in the ablation of C.1 a large step still leaves endpoint task loss at 3.6 times the anchor's after 12 restoration steps.

Read [\[1/2\] Methods and experiments](/en/reviews/hnc-solutions/) first; this part follows appendices A→J.

## A. Hessian Null-Space Continuation Pseudocode

### Algorithms 1–3: What Is Actually Accepted?

Inputs include anchor $\boldsymbol\theta_0$, probes $\mathbf X$, exploration size $\eta$, restoration rate $\alpha$, restoration count $m$, ceiling $\tau$, and optional objective $\varphi$. Algorithm 1:

1. Computes a direction at the current weights and normalizes it.
2. Proposes a small step.
3. Applies $m$ gradient steps reducing function-matching loss.
4. Rejects a proposal still exceeding $\tau$, keeps the old weights, shrinks the step, and refreshes directions.
5. Records only accepted proposals, those passing the loss-ceiling check on the finite probe, as new checkpoints.

Algorithm 2 uses products $\mathbf G\mathbf v=\mathbf J^\top(\mathbf J\mathbf v)$ to obtain $k$ low-eigenvalue directions. Algorithm 3 takes a direction inside that subspace for undirected exploration or a soft-projected direction for steering. In the recurrent neural network (RNN) experiments on the 3-bit flip-flop (3BFF) memory task, undirected exploration holds one heading sampled uniformly from the null space and reprojects it onto the refreshed basis at every relinearization. [A, p.18; F.3](https://arxiv.org/pdf/2609.38081v1#page=18)

### A.1 Computational Cost

| Original Table 1 | Dense eigendecomposition | Matrix-free route |
| --- | --- | --- |
| Time | $O(P^3)$ | $O(kP)$ for fixed block size and iteration count |
| Memory | $O(P^2)$ | $O(kP)$ |
| 3BFF RNN, $P=4611$ | 85 MB; 10 seconds | 1.2 MB; 39 seconds; $k=64$ |
| ViT-S, $P\approx22$M | Approximately 1.9 PB to store the Hessian; $2\times10^7$ products (months of GPU time) just to assemble it; factorization infeasible | Approximately 5.6 GB for a $k=64$ block; about 1 second per product and 1 minute per continuation step |

RNN times use an eight-thread CPU node; ViT times use an NVIDIA H200 and 512 probes. Each product costs about one gradient evaluation, so $O(kP)$ carries the probe count, network computation, and iteration count as constants. Steering adds conjugate-gradient (CG) iterations and restoration steps. For the 4,611-parameter 3BFF RNN, the dense route is faster thanks to lower overhead, and the authors use it to validate the iterative solver. [A.1, Table 1, pp.18–19](https://arxiv.org/pdf/2609.38081v1#page=19)

### A.2 Finding the Flattest Directions by LOBPCG

Equation (4) gives the Rayleigh quotient:

$$
\rho(\mathbf v)=
\frac{\mathbf v^\top\mathbf H\mathbf v}
{\mathbf v^\top\mathbf v}.
$$

In an orthonormal eigenbasis of a symmetric matrix, write $\mathbf v=\sum_i a_i\mathbf v_i$. The numerator is $\sum_i\lambda_i a_i^2$ and the denominator $\sum_i a_i^2$: a weighted average of eigenvalues, minimized at the smallest eigenvalue. This **supplementary calculation** expands the paper's minimization explanation.

LOBPCG (locally optimal block preconditioned conjugate gradient) updates a block of mutually orthogonal directions within the small space spanned by current directions, residuals, and previous directions (the previous-direction term plays the role of momentum in CG). It works through products rather than the full matrix.

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

The system matrix is therefore positive definite. CG approximately solves it without storing an inverse. The paper uses an iteration budget with early exit at relative residual $10^{-8}$, generally warm-starting from the previous direction. The 3BFF steering experiments instead start from zero with a budget of 200 iterations; ViT uses 40 preconditioned-CG iterations. Because steered walks re-form the operator at the current weights every step, they need no separate relinearization.

**Supplementary interpretation:** with $\mathbf g=\nabla\varphi$, the direction uniquely minimizes:

$$
\begin{aligned}
Q(\mathbf d)&=\frac12\|\mathbf d-\mathbf g\|^2
+\frac1{2\mu}\mathbf d^\top\mathbf G\mathbf d,\\
\underset{\mathbf d}{\arg\min}\,Q(\mathbf d)
&=(\mathbf I+\mathbf G/\mu)^{-1}\mathbf g.
\end{aligned}
$$

$Q$ is a review-added objective name. Differentiating gives the preceding linear system. The direction stays close to the desired gradient while paying for the size of the first-order output change, $\mathbf d^\top\mathbf G\mathbf d=\|\mathbf J\mathbf d\|^2$. [A.2–A.3, p.19](https://arxiv.org/pdf/2609.38081v1#page=19)

## B. Gauss–Newton approximation of the Hessian at loss minima

The paper has no numbered theorem on the convergence of HNC or function equivalence along the whole path; its central mathematics is the derivative calculation of this section. The paper stacks probe outputs into a residual vector:

$$
\begin{aligned}
\mathbf r(\boldsymbol\theta)
&=f_{\boldsymbol\theta}(\mathbf X)
-f_{\boldsymbol\theta_0}(\mathbf X),\\
\mathcal L(\boldsymbol\theta)&=\frac12\|\mathbf r(\boldsymbol\theta)\|^2,\\
\mathbf J&=\frac{\partial\mathbf r}{\partial\boldsymbol\theta}.
\end{aligned}
$$

This retains B's sum convention. A sample mean, as in main equation (1), only contributes the same $1/N$ factor to gradient and Hessian, so a null space chosen with the relative cutoff $\mu_{\mathrm{rel}}\lambda_1$ is unchanged. With $\mathbf r\in\mathbb R^q$ and $\boldsymbol\theta\in\mathbb R^P$, $\mathbf J\in\mathbb R^{q\times P}$ and $\mathbf G\in\mathbb R^{P\times P}$; $q$ is a review-added count of output components.

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

Away from the anchor, the residual-weighted curvature term remains, so $\mathbf G$ approximates the Hessian during the walk. $\mathbf G$ is nevertheless always PSD: its eigenvalues are nonnegative, the flat subspace is unambiguously the bottom of its spectrum, and there are no negative-curvature directions to disentangle. Each product $\mathbf G\mathbf v$ costs one Jacobian–vector product and one vector–Jacobian product. [B, equations (5)–(6), pp.19–20](https://arxiv.org/pdf/2609.38081v1#page=19)

### Why Does a Null Direction Not Guarantee Finite-Step Preservation?

**Review supplement:**

$$
\mathbf v^\top\mathbf G\mathbf v=\|\mathbf J\mathbf v\|^2,
\qquad\ker\mathbf G=\ker\mathbf J.
$$

A null direction has zero *first-order* output change. As an illustration, take $f(a,b)=(a+b,b^2)$ at anchor $(0,0)$ with unit direction $\mathbf v=(1,-1)/\sqrt2$:

$$
\begin{aligned}
\mathbf J(0,0)\mathbf v&=\mathbf0,\\
f(t\mathbf v)-f(0,0)&=(0,t^2/2),\\
\mathcal L(t\mathbf v)&=t^4/8.
\end{aligned}
$$

Even an exact Hessian null direction has nonzero loss for finite $t$, and this example's exact zero-loss set contains only the origin. This is the higher-order effect for which main-text §2.2 introduces restoration, small steps, and acceptance checks.

<span id="c-ablation-study-on-the-hnc-algorithm" aria-hidden="true"></span>

## C. Ablation study on HNC algorithm

Appendix C asks on 3BFF whether restoration (C.1), soft projection (C.2), and relinearization (C.3) are each needed, and compares HNC with similarity-penalized training (C.4). Unless stated otherwise, walks are steered to lower centered kernel alignment (CKA) similarity.

| Section | Change | Result and interpretation |
| --- | --- | --- |
| C.1 | 0,3,6,9,12 restoration steps at three step sizes | At large step size $\eta=0.3$, 12 corrections reduce endpoint loss from 6.6× to 3.6× anchor loss, not to zero drift. Small exploration steps matter. |
| C.2 | Soft-projection damping | $10^{-9}$ barely moves. Relaxation increases displacement and drift; removing projection breaks function within dozens of steps despite restoration. |
| C.3 | Refresh never or every 50,20,5 steps | More frequent relinearization lowers loss, especially for large steps. |
| C.4 | HNC versus similarity-penalized training | Over 600 steps and five seeds per setting, HNC reports greater CKA distance at a given loss level. Penalized training initially crosses high loss before returning. |

The authors describe restoration as a safeguard that limits drift rather than a projection that removes it, which is why HNC walks combine it with small steps, an adaptive step schedule, and a loss ceiling. [C, Figures 6–9, pp.20–22](https://arxiv.org/pdf/2609.38081v1#page=20)

## D. Preserving function versus task loss in HNC

First, training stops when the task gradient $\mathbf g$ is small, not when it vanishes. A step of size $\eta$ along a flat direction $\mathbf v$ removes the second-order term but keeps the first-order term $\eta\mathbf g^\top\mathbf v$, and at the small step sizes HNC uses, this term, linear in the step, is the larger of the two. Function matching has exactly zero gradient at the anchor, so its flat directions are flat to both first and second order.

Second, the two Hessians differ in sign. The function-matching Hessian is PSD at the anchor, so its algebraically smallest eigenvalues are also those closest to zero and identify the flattest directions. Writing the task loss $\mathcal L_{\mathrm{task}}$ through a loss $\ell$ on the outputs $f$, its Hessian carries an additional term of output Hessians weighted by $\partial\ell/\partial f_k$ (equation (9)):

$$
\nabla^2\mathcal L_{\mathrm{task}}
=\mathbf J^\top\bigl(\nabla_f^2\ell\bigr)\mathbf J
+\sum_k\frac{\partial\ell}{\partial f_k}\nabla^2 f_k.
$$

The output Hessians $\nabla^2 f_k$ have no sign constraint, so where the second term outweighs the first, the task-loss Hessian becomes indefinite. Selecting the smallest eigenvalues then picks negative-curvature directions instead of nearly flat ones.

The authors confirm both problems on four 3BFF anchors using dense double-precision eigendecompositions. The task-loss Hessian has several hundred negative eigenvalues out of 4,611, and the function-matching Hessian has none. Walking 300 steps without a loss ceiling along the flat direction in which task loss rises fastest (the worst case for a task-loss walk), the task-loss walk's first-order drift exceeds its second-order term by two to three orders of magnitude and task loss grows roughly tenfold, while the function-matching walk keeps task loss within 10% of its anchor value. The authors add that on 3BFF the two losses are close in practice because the task target is the full output trajectory.

Third, a full output vector imposes stronger constraints than one scalar loss. Original Table 2 compares **matched representation distances** on 512 ImageNet probes and 1,000 classes. Values are fractions of probes retaining the anchor's top-1 prediction (mean over probe seeds):

| Preserved quantity | Component constraints | $\Delta\mathrm{CKA}=0.2$ | 0.4 | 0.6 |
| --- | --- | --- | --- | --- |
| Full logits | 512,000 | 0.996 | 0.994 | 0.993 |
| Per-image cross-entropy | 512 | 0.916 | 0.894 | 0.875 |
| Mean cross-entropy | 1 | 0.831 | 0.811 | 0.793 |

Many logit vectors share a cross-entropy value, so a pinned loss can still change predictions. Pinning per-image loss leaves each image's output distribution free to shift; pinning only the mean also lets losses on some probe images be traded for gains on others.

![Original Figure 11: Comparing full-logit, per-image-loss, and mean-loss preservation](/assets/reviews/hnc/constraint-strength.webp)

[Enlarge Figure 11](/assets/reviews/hnc/constraint-strength.webp)

**Read B alongside D.** Solid lines in B are probe results and dashed lines 5,000 held-out images; full-logit matching, in purple, preserves more agreement. Because representation changes at different speeds under each method (A), a given step means something different for each, so D compares them at matched $\Delta\mathrm{CKA}$ instead. Lines are medians over probe seeds and bands the min–max range. [D, Figure 11, Table 2, pp.22–25](https://arxiv.org/pdf/2609.38081v1#page=24)

The authors note that function matching needs neither labels nor a task, so it applies to unlabeled probes, shifted inputs, policies whose return is flat or non-differentiable, and language models matched on next-token distributions; and because every task loss is a function of the outputs, the function-preserving set is a subset of the task-compatible set. To explore the broader task-compatible set, one can hold the task loss at its anchor value through the squared residual $\tfrac12(\mathcal L_{\mathrm{task}}(\boldsymbol\theta)-\mathcal L_{\mathrm{task}}(\boldsymbol\theta_0))^2$, which has zero gradient at the anchor, or project steps orthogonal to the task gradient; the constraint can be tightened by adding probes or by constraining per-input losses or decisions. Each tightening moves toward the full output and, in the limit, recovers the function-matching loss.

## E. Combining HNC with global search for alternative solutions

HNC explores only the local region around an anchor, while a freshly trained model can land anywhere in weight space but not where one chooses. The authors alternate local walks steered by dynamical similarity analysis (DSA) distance with fresh training that carries a repulsive regularizer pushing its dynamics away from every network found so far. All eight networks in the chain exceed $0.999$ bit accuracy while implementing memory through fixed points, ongoing drift without fixed points, and more attractors than memory states. Each retraining moves about as far as an unrelated seed, each walk moves a further fraction of the anchor norm, and every new network lies farther in DSA distance from all its predecessors than two independent seeds lie from each other.

The authors present this as a proof of principle that global and local search can be combined. Loosening the loss ceiling to cross shallow ridges and filtering by task loss afterwards, restarting from an endpoint after a small random kick, and steering toward another network to bridge two solutions through function-preserving intermediates are left to future work. [E, Figures 12–14, pp.26–28](https://arxiv.org/pdf/2609.38081v1#page=26)

## F. Details for the 3BFF RNN experiments

### F.1–F.3 Settings and the Size of the Flat Subspace

With 4,611 parameters, the 64-unit RNN permits a full Hessian spectrum. Across five anchors, curvature concentrates in a few directions, with participation ratio (effective dimension) $23.2\pm1.6$, and the remaining eigenvalues decay smoothly over nine orders of magnitude with no gap between sharp and flat directions. The size of the flat subspace therefore depends on where the threshold is placed.

**What participation ratio measures.** Let $D_{\mathrm{PR}}$ be a review abbreviation for effective dimension. For a nonzero PSD spectrum,

$$
D_{\mathrm{PR}}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
$$

It equals $r$ for $r$ equal positive eigenvalues and approaches one when curvature concentrates in one. Tiny positive eigenvalues count toward rank but contribute little here. [Appendix F.2](https://arxiv.org/pdf/2609.38081v1#page=29)

Original Table 3 reports median flat dimensions at different relative thresholds:

| $\mu_{\mathrm{rel}}$ | Median dimension | Fraction of parameters |
| --- | --- | --- |
| $10^{-7}$ | 1701 | 36.9% |
| $10^{-6}$ | 2896 | 62.8% |
| $10^{-5}$ | 3684 | 79.9% |
| $10^{-4}$ | 4145 | 89.9% |
| $10^{-3}$ | 4418 | 95.8% |
| $10^{-2}$ | 4565 | 99.0% |

Allowing more output drift classifies more directions as flat. In the walk settings, CKA steering runs 600 steps and DSA steering 2,000, with 12 restoration steps after every step. The steering objective is computed on a 32-trial probe, while acceptance uses task MSE on a fixed batch of 128 trials rather than Algorithm 1's function-matching loss (F.3). [F.1–F.3, pp.29–30](https://arxiv.org/pdf/2609.38081v1#page=29)

### F.4–F.8 Diversity, Pruning, Resumed SGD, and OOD Inputs

Galleries and a ten-anchor multidimensional scaling (MDS) show alternatives in many directions rather than one unusual endpoint. In weight distance the networks cluster by anchor, whereas in representation distance all anchors collapse together and the steered walks spread in many directions. Pruning-steered HNC, which drives each unit's outgoing recurrent and output weight group toward zero, shrinks one 64-unit RNN to 11 units with 100% bit accuracy: about 17% of the hidden size and 5% of the parameters (review count: 4,611→212), with a small rise in MSE.

After 8,000 resumed SGD steps, the networks settle at lower task loss while the steered differences in CKA, DSA, or sparsity stay where the walks left them; retraining moves the weights only slightly and does not pull endpoints back toward the anchor. Under out-of-distribution (OOD) pulses, weak pulses degrade the DSA-steered and undirected endpoints before the anchor, strong pulses degrade only the undirected endpoint, whose recurrent spectral radius has grown, and rare pulses hit the DSA-steered endpoint, whose memory retention is weaker, first. [F.4–F.8, pp.30–35](https://arxiv.org/pdf/2609.38081v1#page=31)

## G. Additional experiments on ViT

### G.1–G.4 Steering Objectives and Settings

**What CKA compares.** Here $X,Y$ are review notation for column-centered feature matrices with matching sample order, distinct from probe inputs $\mathbf X$.

$$
\operatorname{CKA}_{\mathrm{linear}}(X,Y)
=\frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F\,\|Y^\top Y\|_F}.
$$

This linear CKA ranges from zero to one, with larger values indicating more similar sample geometry. It is invariant to orthogonal coordinate changes and isotropic scaling, but not to arbitrary invertible transforms. [CKA source §3–5, Eq. (14)](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf#page=5)

With $c_{ij}(\boldsymbol\theta)$ denoting linear CKA between anchor layer $i$ and current layer $j$, equation (12) is:

$$
\begin{aligned}
\varphi_{\mathrm{CKA}}(\boldsymbol\theta)
&=1-\frac1\beta\log\sum_{i,j=1}^{L}
\exp\!\bigl(\beta c_{ij}(\boldsymbol\theta)\bigr),\\
L&=13,\qquad\beta=30.
\end{aligned}
$$

Increasing it applies the strongest downward pressure to whichever layer pairs are currently most similar; reported values use the hard max. G.2's semantic, color, and layout scores are respectively DINOv2 ViT-S/14 cosine, HSV $8^3$ histogram cosine, and $32\times32$ grayscale correlation.

G.3 repeats the experiment with the other similarity score used for the Platonic Representation Hypothesis (PRH), mutual $k$-nearest-neighbor (kNN) alignment. It measures, per image, the fraction of $k=10$ nearest neighbors shared by the two representations, maximized over layer pairs, and uses smooth neighborhood weights for differentiation. The endpoint's alignment falls below every independently trained model in the comparison but stays above an untrained initialization. G.3's text holds the function fixed on 256 Places-365 images ($N=256$), whereas G.4 Table 8 lists 512 probes for the same kNN walk.

### G.5–G.6 Held-Out Preservation and the Damage Control

The headline results for CKA-steered endpoints (error consistency near 0.8, agreement at least that of the different-seed reference, and accuracy-matched noise leaving CKA above 0.98) are in [\[1/2\] §4](/en/reviews/hnc-solutions/). The comparisons use different replicate units: CKA steering has five probe draws and kNN steering three; the diff-seed reference, which repeats the anchor's final ImageNet-1k fine-tuning from the same ImageNet-21k checkpoint with only the seed changed, has two seeds; and the single-anchor and independently trained ViT-S/in1k bands are image bootstraps.

kNN-steered endpoints keep the shape of the anchor's top-$k$ accuracy curves but lose more accuracy. Exact prediction agreement falls from about 0.85 on clean ImageNet to about 0.6 on Sketch, and error consistency is about 0.75. They agree with the anchor more than ViT-S/in1k does but fall below the diff-seed reference on all four datasets. Noise matched to the walk's accuracy loss moves the weights about five times farther than the walk yet stays more similar to the anchor than any independently trained model. The authors attribute the weaker preservation to the preservation–displacement trade-off of the configuration tested rather than an intrinsic limit of the metric, citing G.9's finding that larger probes improve off-probe preservation. **Review interpretation.** That explanation rests on probe size, and because the kNN configuration's probe size is stated differently in G.3's text and in Table 8, the paper does not establish whether the two steering objectives were compared at the same probe size. [G.5–G.6, pp.39–42](https://arxiv.org/pdf/2609.38081v1#page=39)

Error consistency is chance-corrected agreement on whether both models are correct or both are wrong on each image; unlike prediction match, it does not require the same incorrect class when both are wrong. This is why prediction match can fall on the strongly shifted Rendition and Sketch sets while error consistency holds. [Appendix G.5](https://arxiv.org/pdf/2609.38081v1#page=39)

### G.7–G.10 Layers, Directions, and Probe Size

An early block can also be steered, but it costs more function per unit of similarity lost than the pre-logit feature; the authors conclude that cross-model similarity of early layers is not forced by the function either. Steering ViT-S toward a frozen ViT-B, with the function held on 256 Places-365 images, raises depth-averaged CKA from 0.78 to 0.96 while retaining **96% of the predictions on this probe**. On 5,000 held-out ImageNet images it keeps about nine in ten predictions and loses a little over one percentage point of top-1 accuracy. The authors conclude that the similarity two independently trained models happen to share is not a ceiling set by the task, so the degree of convergence needs an explanation beyond what the function requires.

Enlarging the probe improves off-probe preservation while letting the representation move farther, at compute that grows roughly linearly with probe count. The authors explain that a larger probe pins the function more firmly, so the walk stays under its loss ceiling longer and spends more steps steering. The final MDS shows CKA- and kNN-steered walks leaving the anchor in different directions, fanning out by probe set, and appearing to move different distances depending on the distance used. [G.7–G.10, pp.43–46](https://arxiv.org/pdf/2609.38081v1#page=43)

## H. Steering Brain-Score at fixed function

Brain-Score measures how well a model's internal features predict recorded neural responses and is usually treated as a fixed property of architecture, objective, and data. The authors ask how much of the score instead depends on which function-equivalent solution training reached. The MajajHong2015 benchmark uses 3,200 images and 168 recording sites in macaque inferior temporal cortex (IT). The official score fits ten-fold partial least squares (PLS) regression with 25 components, correlates predictions site by site, and divides the fold-averaged median over sites by the noise ceiling of 0.82.

Because PLS is fit iteratively and is not differentiated, steering uses a separate surrogate: ridge regression on a 64-dimensional PCA basis fitted once to the anchor and frozen, trained on one half of the steering images, with mean site-wise correlation on the other half raised or lowered. The authors note that the surrogate and the official score use different regressions and, in held-out evaluations, different images and sites, so a change in the official score is not the direct target of the optimization.

Results are reported for two models (top-1 decisions preserved on at least 94% of images, logit KL $10^{-5}$–$10^{-3}$). For the top model of the Brain-Score neural leaderboard, an 86M-parameter ViT with relative-position embeddings, IT predictivity spans 0.618–0.719 (about 0.10) between down and up endpoints; this range is measured on the images and sites used for steering, and on sites and images that neither the walk nor the surrogate saw it is 0.039. The plain ViT-S/16, the lowest scorer in the comparison, spans 0.546–0.627 (0.081) on held-out images. The spread across seven architectures run through the same pipeline is 0.074 in the text and 0.075 in the figure caption, and the authors conclude that the range opened within one network "matches or exceeds" the entire gap across architectures. Steering IT leaves V1, V2, and V4 predictivity essentially unchanged. [H, pp.46–48](https://arxiv.org/pdf/2609.38081v1#page=46)

**Review interpretation.** "Matches or exceeds" holds for ViT-S on held-out images (0.081 versus 0.074). For the 86M model it holds only for the 0.10 range measured on the steering images and sites; the held-out range of 0.039 is about half the cross-model spread. The conclusion that neural predictivity also depends on which solution is reached therefore stands for both models, but the evidence that the effect is as large as architectural differences comes from ViT-S alone.

## I. Extending HNC to reinforcement learning

### I.1 Advantages and the Surrogate

Plume and Ant use generalized advantage estimation (GAE). With rewards $r_t$ and critic $V$:

$$
\begin{aligned}
\delta_t&=r_t+\gamma V(s_{t+1})-V(s_t),\\
A_t&=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},\\
\gamma&=0.99,\qquad\lambda=0.95.
\end{aligned}
$$

The next-state value is zero at episode termination. Boat race instead uses critic-free discounted proxy returns with $\gamma=0.9$ as advantages. All three tasks standardize advantages over the buffer, so at the reference policy, where every importance ratio is one, the surrogate is zero.

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

State arguments are omitted only in this block. Covariance stays frozen at the anchor's value throughout the walk, so the weights $1/(2\sigma_j^2)$ are constants, and the authors maximize the unweighted mean-action MSE. It is a constant multiple of KL when all $\sigma_j$ agree and otherwise differs only by the fixed diagonal weights.

### I.3–I.7 Conditions and Failures in Three Environments

| Environment | Buffer/evaluation | Observation and limitation |
| --- | --- | --- |
| Plume | Eight current-policy episodes; refresh every five steps; 60 standard initial conditions | Less casting and retained return. OOD evaluation uses 240 initial conditions per condition, with SE across 8×30 batches. Performance collapses when the walk leaves the level set at step 20. |
| Boat race | 32 episodes; refresh every three steps; $\epsilon=0.25$ exploration; 100 evaluation episodes | Ten true-return-trained anchors. The three most divergent endpoints collect proxy reward with zero net progress. |
| Ant-v3 | 4,000 transitions; refresh every five steps; 30 fixed evaluation states | Action divergence reaches 1.0 in 19 steps. Matched-action-KL random directions lose over half the return. HNC preserves mean return while changing failures. |

Ant gait analysis uses walking episodes only: 25/30 for the anchor and 27/30 for the alternative. The latter stalls less but sometimes terminates early when torso height becomes unhealthy. Per-episode returns are uncorrelated across shared initial states, which the authors read as two different competence profiles hidden behind equal mean return. Each plume OOD condition has its own initial-condition grid, and the paper states that rates are comparable across steps within a condition but not across conditions. [I, pp.49–53](https://arxiv.org/pdf/2609.38081v1#page=49)

## J. Probing loss-landscape geometry with HNC

Five widths, five class counts, and three seeds make 75 CNN anchors. GroupNorm makes each image's output independent of batch composition, and 512 training inputs pin each model's function. J.2 and J.3 show that the trends hold across drift tolerances and relative spectral cutoffs. In J.4, the participation ratio of pre-logit feature covariance on held-out images grows about fivefold with class count but barely changes with width; the authors use this to confirm class count as a proxy for task complexity. **Feature effective dimension and parameter flat fraction are different measurements.**

**Review interpretation.** J.1 states that because training stops at 100% training accuracy, "every anchor [is] an exact zero-loss solution." Under cross-entropy, a correct top-1 prediction with target probability 0.6 still has positive loss, so the accurate statement is that every anchor classifies its training data correctly. The §6 measurements rely only on the function-matching loss, which targets the anchor's own outputs and is zero at the anchor by definition, so the conclusions are unaffected.

**Review supplement:** with $q$ probe output components, $\operatorname{rank}(\mathbf J)\le q$ and $\dim\ker\mathbf J\ge P-q$. In §6, $q=512C$, so for width-16 models ($P=75{,}703$–$81{,}748$ in the paper's Table 12) the linear null fraction is at least about 95% at $C=7$ and about 37% at $C=100$. Since $P$ grows with width and $q$ with class count, both trends in main-text Figure 5A ([\[1/2\] §6](/en/reviews/hnc-solutions/)) point in the same direction as this counting bound. [J, pp.53–55](https://arxiv.org/pdf/2609.38081v1#page=53)

## Review interpretation and open questions

**Review interpretation.** The appendices' strength is that they show, through failures, why each component is needed. Without projection the function breaks within a few dozen steps despite restoration (C.2); with large steps and infrequent relinearization the loss blows up (C.3); and preserving task loss instead lets first-order drift and negative-curvature directions pull the walk off the task (D). However, the ablations (C) use only the 3BFF RNN and the constraint-strength comparison (D) only one ViT-S anchor, so it remains unshown whether the same settings transfer to other architectures. C.4 matches step count and step length, so it leaves out the compute gap between HNC, which needs curvature products and CG iterations, and penalized training, which needs no curvature information.

Three questions remain. Is the function preserved off the probe along the path between accepted checkpoints? Does kNN steering preserve function as well as CKA steering at the same probe size? And can the point where preserving surrogate return turns into a collapse of actual return (step 20 for Plume) be detected during the walk?

## Sources and reading scope

- Paper: *Traversing the solution space of neural networks with Hessian Null Space Continuation* · Ann Huang et al. · first public date 2026-09-29 · [reviewed version](https://arxiv.org/abs/2609.38081v1) · reviewed 2026-10-23.
- Scope read: Version 1, appendices A–J: algorithms, derivative arguments, ablations, and validation conditions. Repeated galleries and exhaustive hyperparameter rows are compressed.
- External source consulted: Kornblith et al. (2019) for the CKA definition (§3–5).
- Figure sources: The Figure-1 thumbnail and body figures in this part are author assets from v1 (CC BY 4.0), PDF-rasterized and WebP-encoded with content, layout, text, and arrows retained. [Paper](https://arxiv.org/abs/2609.38081v1).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
