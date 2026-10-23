---
title: "[2/2] Single Global Merging in depth: appendix experiments and proof conditions"
description: "Follow Appendices A–D through setup, ablations, average dynamics, Taylor expansions, and unresolved proof conditions."
date: "2026-10-23"
publishedAt: "2026-10-23T12:14:58+09:00"
updatedAt: "2026-10-23T08:56:24+00:00"
topics: ["decentralized learning", "model merging", "distributed optimization", "federated learning", "communication efficiency"]
visibility: "public"
paperTitle: "On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning"
paperPublishedDate: "2025-07-09"
authors: ["Tongtian Zhu", "Tianyu Zhang", "Mingze Wang", "Zhanpeng Zhou", "Can Wang"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2507.06542v4"
thumbnail: "/assets/reviews/single-global-merging/thumbnail.webp"
lang: "en"
translationKey: "single-global-merging-theory"
thumbnailAlt: "Original Figure 1(b): global test accuracy recovers sharply after the final weight average in sparse decentralized ResNet-18 training."
---

The global-merging theory connects communication-controlled model disagreement to curvature corrections in the average update. The main setup uses 300 rounds of 100 local updates each, with neighbor exchanges at probability 0.2 followed by a final average. [Appendices C.1 and D](https://arxiv.org/pdf/2507.06542v4#page=22) **Review interpretation.** Its explanatory scope depends on retaining non-IID residuals and RMS bounds, and on distinguishing observed recovery from a general convergence guarantee.

Read [[1/2] Main paper and core experiments](/en/reviews/single-global-merging/) first. We retain the original **A→B→C→D** appendix order and follow Lemma D.1 through Proposition D.9 within D. Intermediate derivations and counterexamples are labeled **review additions**; they are not presented as replacement theorems proved by the authors.

## A. Limitations and Potential Questions

The seven questions in Appendix A clarify the intended contribution.

| Original question | Authors' response | Qualification to retain |
| --- | --- | --- |
| Why AdamW when the theory is SGD? | AdamW performs well under non-IID data; the main phenomenon also occurs with SGD. | The same convergence proof is not provided for AdamW. |
| How does convergence establish mergeability? | The merged model can match parallel-SGD convergence, suggesting an advantage over local models. | A stationarity upper bound does not directly establish terminal test accuracy or dominance over every local model. |
| How were hyperparameters selected? | Grid search based on global test performance. | This differs from independent validation selection; selection bias was not independently tested. |
| How is initialization different from Model soups? | Gains occur for distinct random initializations and shared pretrained initialization. | The common-initialization theory does not cover every initialization experiment. |
| How was the landscape visualized? | Sixteen vertices of a regular polygon with Wachspress interpolation. | The display coordinates do not preserve high-dimensional parameter distances. |
| Is the evidence vision-only? | Yes; broader tasks remain future work. | No corresponding LLM, speech, or RL result is demonstrated. |
| Is this ordinary gossip convergence to identical models? | Heterogeneous models can remain different while being mergeable. | A good average and full consensus are distinct properties. |

The central question is **what relationship sparse communication leaves among models**, rather than merely whether an averaging operator is used. Comparing upper bounds does not transitively prove actual model dominance or the loss inequality in Definition 2. [A](https://arxiv.org/pdf/2507.06542v4#page=19)

## B. Additional Background and Related Work

B.1 surveys graph design, asynchronous updates, heterogeneity, privacy, and large-model systems. B.2 discusses implicit bias; B.3 connects mode connectivity and model merging. The application lists are compressed here while retaining the argument.

**B.1:** Local optimization with gossip is an established framework. Changing topologies, local steps, and non-IID objectives already have convergence analyses. The present experiments are not new measurements of throughput on a deployed geographically distributed system.

**B.2:** Earlier work motivates treating disagreement as more than harmful noise. The direct dependency, [Zhu et al. 2023, Theorem 1 and C.2](https://proceedings.mlr.press/v202/zhu23e.html), analyzes curvature corrections when workers sample from a common distribution. D.3 below explains why that common-distribution assumption matters.

**B.3:** Shared pretrained checkpoints, homogeneous training conditions, or permutation alignment can make averaging easier. The present observation combines different initializations, heterogeneous data, and no post-hoc permutation alignment with **lightweight communication during training**. It does not invalidate successful independent averaging in other regimes. [B](https://arxiv.org/pdf/2507.06542v4#page=21)

## C. Additional Experiments

Open the body images to inspect axes and legends at full resolution.

### C.1 Experimental Setups

| Item | Reported setup |
| --- | --- |
| Data | CIFAR-10/100 and Tiny ImageNet; central figures use CIFAR-100 and Tiny ImageNet |
| Agents | 32 in main Figure 1; 16 in sliding-window/landscape experiments; appendix compares 16/32 |
| Heterogeneity | Primarily Dirichlet $\alpha=0.1$; C.5 also uses $\alpha=1.0$ |
| Training | 300 communication rounds × 100 local mini-batch updates per round |
| Models | Scratch ResNet-18 and pretrained CLIP ViT-B/32 |
| Batch size and weight decay | Default 128 and $5\times10^{-4}$ |
| SGD learning rates | ResNet $10^{-2}$; CLIP $10^{-3}$ |
| AdamW learning rates | ResNet $5\times10^{-4}$; CLIP $10^{-5}$, as written in C.1 |
| Augmentation | RandomCrop with padding 4, RandomHorizontalFlip, RandAugment with 2 operations and magnitude 9 |
| Sparse communication | $R=0.2$: one random peer with probability 0.2; comparisons include $R=1$, ring, and exponential graphs |
| Compute | 80GB A100 facility; example 16-model simulations: about 15GB/8 hours for ResNet and 30GB/12 hours for CLIP |

Tiny ImageNet inputs are $64\times64$ with patch size 32, giving four image patches plus CLS, or five CLIP tokens. This is not conventional $224\times224$ CLIP evaluation or full ImageNet training. The reported runtimes describe simulations, not measured WAN communication bottlenecks.

The paper states 4096 samples per agent. Treating these as fixed, disjoint ownership would conflict with total dataset sizes for 32 agents. The current [fixed code snapshot](https://github.com/Raiden-Zhu/ICLR-2026-Grokking-in-Decentralized-Learning/tree/6a5c139c2cdf6d6d7e0b6ba3c00f123e494e6660) documents a paper-facing resample mode: class proportions remain fixed while concrete samples are redrawn as iterators are recreated, using sampling with replacement. This differs from fixed mutually exclusive on-device datasets. Equivalence between the current code and the historical figure-generating revision was not confirmed.

Appendix A's CLIP AdamW search grid also excludes the $10^{-5}$ stated in C.1. This discrepancy is retained rather than silently corrected. Current code documentation describes averaging learnable parameters and recalibrating BatchNorm statistics afterward. That is a reproduction detail to check, not evidence that the published results were independently reproduced. [C.1](https://arxiv.org/pdf/2507.06542v4#page=24)

### C.2 Practical Evaluation Metrics

The three metrics evaluate local models on local distributions, local models on the global distribution, and the averaged model on the global distribution. Under strong label skew, a model specialized in its own classes can look strong on the first metric. Evaluating a final merge requires separating the second and third.

**Weight averaging is also different from output ensembling.** Ensembling retains multiple models at inference; this paper evaluates a single model with averaged parameters. [C.2](https://arxiv.org/pdf/2507.06542v4#page=24)

### C.3 Additional comparisons and ablations

| Original figure | Varied condition | Evidence and limit |
| --- | --- | --- |
| C.1–C.2 | 16/32 agents with AdamW | The merge gap also appears beyond SGD; this is not an all-optimizer theorem. |
| C.3 | $R=0.2/1$, ring/exponential | Tests whether the observation is restricted to one random-peer setup; topology differences remain. |
| C.4(a,b) | Batch 64/128; learning rates $5\times10^{-4}/10^{-4}$ | Limited hyperparameter variations, not a map of the entire stability region. |
| C.4(c) | Shared versus different initializations | Gains occur with distinct initializations in the tested scratch setting; common-initialization theory remains separate. |
| C.5 | CIFAR-100/Tiny ImageNet, $\alpha=0.1/1$, two graphs | Merge gaps depend on heterogeneity and topology. |
| C.6 | One/five final exponential-gossip rounds versus a perfect global merge | Constrained gossip can approximate merging, but one gossip operation is not a perfect average. |

#### C.3.1 Agent count and optimizer

C.1 with 16 agents and C.2 with 32 show pre/post-merge gaps for CLIP and ResNet using AdamW. Importantly, a gap below the federated baseline does not always remain. In C.2, the final orange point is approximately 0.61 versus blue 0.59 for CLIP and 0.49 versus 0.45 for ResNet. These are **approximate readings from PDF p.26**, without verified repeated-run means or significance. This evidence would be lost by summarizing all results as recovery that remains below FL. The AdamW observations are not direct verification of the SGD convergence theorem.

#### C.3.2 Peer count and topology

C.3(a) compares $R=0.2$ and $R=1$ with pretrained ResNet weights. More frequent communication gives better pre-merge accuracy, while $R=0.2$ has a larger final merge jump. **A larger merging benefit is different from higher final accuracy after merging.** In C.3(b), exponential graphs generally outperform rings for both individual and averaged models, while both benefit from averaging. Topology therefore still matters. [C.3.1–C.3.2, Figures C.1–C.3, pp.25–26](https://arxiv.org/pdf/2507.06542v4#page=26)

#### C.3.3 Hyperparameters, initialization, and heterogeneity

C.4 asks whether average-versus-individual gaps persist under changed batch sizes and learning rates. Shared and different initialization curves in C.4(c) also recover similarly in the tested setup. This rules against treating distinct initializations as an automatic failure, without guaranteeing arbitrary neuron permutations or initialization combinations. C.5 then examines how datasets and label skew change the gap.

[![Figure C.5. Individual and averaged models across datasets, Dirichlet heterogeneity, and ring or exponential graphs.](/assets/reviews/single-global-merging/figure-c5.webp)](/assets/reviews/single-global-merging/figure-c5.webp)

*Original Figure C.5(a–d), p.27. Horizontal axes are rounds; vertical axes are global test accuracy. Within each color, dark denotes local models and light denotes their average. Panels (a,c) use $\alpha=0.1$; (b,d) use $\alpha=1$. Larger gaps occur under stronger heterogeneity, while weaker heterogeneity leaves smaller gaps. Vertical-axis ranges also differ across panels. CC BY 4.0; PDF extraction and lossless WebP conversion.*

This figure limits an interpretation of uniformly large gains. For $\alpha=1$, individual and averaged models can perform similarly. Local specialization and the benefit of averaging must be assessed together.

#### C.3.4 Final topology-constrained merging

This experiment asks whether recovery survives when a perfect final average is unavailable. After training, agents perform one or five mixing rounds on an exponential graph rather than further local optimization. **A communication round means different things under different topologies**; this does not establish equivalence between one neighbor exchange and global AllReduce.

[![Figure C.6. One or five final exponential-gossip rounds compared with one perfect global merge.](/assets/reviews/single-global-merging/figure-c6.webp)](/assets/reviews/single-global-merging/figure-c6.webp)

*Original Figure C.6(a–c), p.27. Axes show communication rounds and average global test accuracy. Panels (a,b) constrain final merging to an exponential graph; (c) uses a perfect final average. The orange terminal jumps show recovery with limited gossip, without identical final results. The printed caption says AdamW whereas linked figure-source filenames contain SGD; this review follows the caption without claiming independent confirmation of the exact optimizer used. CC BY 4.0; PDF extraction and lossless WebP conversion.*

Repetition counts, seeds, and error bars are not sufficiently specified in the paper to treat small curve differences as statistically established advantages. [C.3, pp.25–27](https://arxiv.org/pdf/2507.06542v4#page=25)

## D. Theory

We follow the original result order. Review abbreviations are $\delta_k^{(t)}=\theta_k^{(t)}-\bar\theta^{(t)}$, $g_t=\nabla\mathcal L(\bar\theta^{(t)})$, $H_t=\nabla^2\mathcal L(\bar\theta^{(t)})$, and $q_t=\nabla\operatorname{Tr}(H_t\Gamma^{(t)})$. Covariance is $d\times d$; $\Theta$ is $d\times m$. Covariance is held fixed when differentiating $q_t$.

### Review supplement: nontriviality of mergeability

Definition 2 asks for nonnegative weights satisfying a loss inequality, while the experiments evaluate uniform averaging. [Section 3, Definition 2](https://arxiv.org/pdf/2507.06542v4#page=4)

**Review interpretation.** The printed definition omits weights summing to one; even adding that condition allows a single unit weight, which satisfies equality. A nontrivial definition needs a fixed uniform average or a condition requiring multiple positive weights. This affects the definition’s discriminating power, not the observed recovery of the uniform average.

### Lemma D.1: the consensus-distance recursion

Combining mixing contraction with gradient bounds, the paper obtains the following. Here $\mathbb E_t$ conditions on the past state and averages fresh gradient and communication randomness:

$$
\mathbb E_t\Xi_{t+1}^2
\leq\left(1-\frac p2\right)\Xi_t^2
+\frac{12(1-p)\eta^2}{p}(\phi_t^2+\sigma^2),
\quad
\phi_t^2=\frac1m\sum_k\|\nabla\mathcal L_k(\theta_k^{(t)})\|^2.
$$

The proof separates disagreement introduced by local updates from contraction through mixing. Young's inequality splits the cross term in $\|a+b\|^2$, relaxing contraction from $1-p$ to $1-p/2$. Local gradients and stochastic noise control the second term. Conditional expectations must then be passed to full expectations. [Lemma D.1, p.28](https://arxiv.org/pdf/2507.06542v4#page=28)

**Review derivation of the recursion.** Assume the paper's Assumptions 1–3 and use the appendix's right-multiplication convention. Let $G_t\in\mathbb R^{d\times m}$ stack agents' stochastic gradients as columns. Averaging is the projection

$$
J=\frac{\mathbf1\mathbf1^\top}{m},\qquad \Theta(I-J)=\Theta-\bar\Theta.
$$

and $WJ=JW=J$ gives

$$
\Theta^{(t+1)}=(\Theta^{(t)}-\eta G_t)W,
\qquad
\Theta^{(t+1)}-\bar\Theta^{(t+1)}
=(\Theta^{(t)}-\eta G_t)(W-J).
$$

When mixing contraction can be applied conditionally to the fresh gradients and communication randomness, Young's inequality $\|A+B\|_F^2\leq(1+a)\|A\|_F^2+(1+1/a)\|B\|_F^2$ with $a=p/2$ yields

$$
\begin{aligned}
m\mathbb E_t\Xi_{t+1}^2
&\leq(1-p)\mathbb E_t
\|(\Theta^{(t)}-\bar\Theta^{(t)})-\eta G_t(I-J)\|_F^2\\
&\leq(1-p)(1+p/2)m\Xi_t^2
+(1-p)(1+2/p)\eta^2\mathbb E_t\|G_t\|_F^2.
\end{aligned}
$$

Here $\|G_t(I-J)\|_F\leq\|G_t\|_F$ because $I-J$ is an orthogonal projection. For $0<p\leq1$, use the coefficient bounds and gradient/noise decomposition

$$
(1-p)(1+p/2)\leq1-p/2,\qquad 1+2/p\leq6/p,
\qquad
\mathbb E_t\|G_t\|_F^2\leq2m(\phi_t^2+\sigma^2).
$$

to bound the right side by $(1-p/2)m\Xi_t^2+12m(1-p)\eta^2(\phi_t^2+\sigma^2)/p$. Dividing by $m$ gives the preceding conditional recursion. Taking full expectation also places expectations on $\Xi_t^2$ and $\phi_t^2$. The factor 12 comes from these loose bounds, rather than being an optimal constant.

### Corollary D.2: unrolling the recursion

**Review derivation.** Set $\rho=1-p/2$ and suppose $\phi_t^2\leq\phi^2$. Repeated substitution gives

$$
\begin{aligned}
\mathbb E\Xi_t^2
&\leq\rho^t\Xi_0^2+
\frac{12(1-p)\eta^2}{p}(\phi^2+\sigma^2)
\sum_{s=0}^{t-1}\rho^s\\
&\leq\rho^t\Xi_0^2+
\frac{24(1-p)\eta^2}{p^2}(\phi^2+\sigma^2)
\end{aligned}
$$

The last step uses $\sum_{s\geq0}\rho^s=1/(1-\rho)=2/p$. Common initialization sets $\Xi_0=0$; distinct initializations leave the decaying initial term.

The paper also supplies a more local bound involving $\phi_t$ when gradients change sufficiently slowly. That extra condition rules out abrupt gradient decreases and should not be dropped when substituting the local bound for the uniform one. [Corollary D.2, p.29](https://arxiv.org/pdf/2507.06542v4#page=29)

Writing the slow-change condition makes its role explicit. With common initialization, the paper uses

$$
\phi_s^2\leq(1+p/4)\phi_{s+1}^2,\qquad
(1-p/2)(1+p/4)=1-p/4-p^2/8\leq1-p/4.
$$

to compare past gradients with a recent value. The product still contracts by at most $1-p/4$, so the geometric-series bound becomes $4/p$ rather than $2/p$, producing the stated local bound

$$
\mathbb E\Xi_t^2
\leq\frac{48(1-p)\eta^2}{p^2}
(\mathbb E\phi_{t-1}^2+\sigma^2).
$$

The display uses full-expectation notation, including the recent gradient term. It uses recent $\phi_{t-1}$ rather than a uniform $\phi$, but only within the assumed slow-change regime.

### Proposition D.3: average dynamics and Gaussian smoothing

With doubly stochastic mixing and SGD, the actual conditional mean update is

$$
\mathbb E_t\bar\theta^{(t+1)}
=\bar\theta^{(t)}-\frac\eta m
\sum_k\nabla\mathcal L_k(\theta_k^{(t)})
$$

The paper invokes [Zhu et al. 2023](https://proceedings.mlr.press/v202/zhu23e.html) to replace this with a globally smoothed gradient under $\epsilon\sim\mathcal N(0,\Gamma^{(t)})$, plus cubic residuals. The Gaussian is an analytical covariance-matching device, not noise explicitly injected by the implementation.

The relation claimed in Proposition D.3 is worth retaining explicitly. Its $\delta^{(t)}$ is an update-unit residual, distinct from individual deviations $\delta_k^{(t)}$.

$$
\begin{aligned}
\mathbb E_t\bar\theta^{(t+1)}
&=\bar\theta^{(t)}
-\eta\mathbb E_{\epsilon^{(t)}\sim\mathcal N(0,\Gamma^{(t)})}
\nabla\mathcal L(\bar\theta^{(t)}+\epsilon^{(t)})
+\delta^{(t)},\\
\delta^{(t)}
&=\mathcal O\!\left(\frac{\eta}{m}\sum_k
\|\theta_k^{(t)}-\bar\theta^{(t)}\|^3\right).
\end{aligned}
$$

We first explain the cancellation intended by this relation, then identify the term that survives under non-IID losses. This does not certify the identity under only the assumptions printed in the paper.

**Why IID matters — review addition.** If all local objectives equal $\mathcal L$,

$$
\frac1m\sum_k\nabla\mathcal L(\bar\theta+\delta_k)
=g+\frac12\nabla\operatorname{Tr}(H\Gamma)+r,
\quad
\|r\|\leq\frac{L_4}{6m}\sum_k\|\delta_k\|^3.
$$

The first-order term cancels because $H(m^{-1}\sum_k\delta_k)=0$. The quadratic term of the gradient, involving third derivatives of the loss, contracts with covariance. A centered Gaussian with the same covariance shares this second-order term, but its third moment need not match the empirical deviations.

Under heterogeneous objectives, local Hessians differ and

$$
\frac1m\sum_k H_k(\bar\theta)\delta_k
$$

need not vanish. Appendix C.2 of Zhu 2023 actually uses the common expected Hessian to obtain cancellation. The global $C^4$ condition stated in the target paper's D.3 does not supply it.

**Explicit counterexample.** Let two scalar local losses and models be

$$
\mathcal L_1(x)=1-\cos x,\quad
\mathcal L_2(x)=2(1-\cos x),\quad
\theta_1=b+a,\quad\theta_2=b-a
$$

The global objective is $\mathcal L(x)=\tfrac32(1-\cos x)$. Derivatives of orders 1–4 and gradient heterogeneity are bounded. Nevertheless,

$$
\begin{aligned}
\frac12\bigl[\mathcal L_1'(b+a)+\mathcal L_2'(b-a)\bigr]
&=\frac32\sin b\cos a-\frac12\cos b\sin a,\\
\frac12\bigl[\mathcal L'(b+a)+\mathcal L'(b-a)\bigr]
&=\frac32\sin b\cos a.
\end{aligned}
$$

Their difference is $-\tfrac12\cos b\sin a=\mathcal O(a)$. At $b=0.4$ and $a=0.1,0.01,0.001$, the difference divided by $a$ approaches approximately $-0.4605$. It cannot be absorbed into a cubic residual. Gaussian global-gradient smoothing supplies a quadratic correction, not a cancellation of this first-order term. This identifies **an extra term needed for D.3 under non-IID objectives**, not a refutation of the experimental curves. [D.3, p.30](https://arxiv.org/pdf/2507.06542v4#page=30)

### Corollary D.4: stochastic variance

If local gradient noises are conditionally independent, centered, and have average variance bounded by $\sigma^2$, their average has variance at most $\sigma^2/m$. Writing the actual mean gradient as $\bar g_t=m^{-1}\sum_k\nabla\mathcal L_k(\theta_k^{(t)})$ gives

$$
\mathbb E_t\|\bar\theta^{(t+1)}-\bar\theta^{(t)}\|^2
\leq\eta^2\|\bar g_t\|^2+\frac{\eta^2\sigma^2}{m}.
$$

This is separate from replacing $\bar g_t$ by $g_t+q_t/2$. Variance reduction does not remove heterogeneous bias. [Corollary D.4, p.30](https://arxiv.org/pdf/2507.06542v4#page=30)

**Review derivation: why $1/m$?** Let $\nu_k$ be agent $k$'s centered gradient noise. Then

$$
\mathbb E_t\left\|\frac1m\sum_k\nu_k\right\|^2
=\frac1{m^2}\sum_k\mathbb E_t\|\nu_k\|^2
+\frac1{m^2}\sum_{k\ne l}\mathbb E_t[\nu_k^\top\nu_l]
\leq\frac{\sigma^2}{m}.
$$

Conditional independence and zero means remove the $k\ne l$ cross terms; Assumption 3 controls the remaining sum. Standard deviation scales as $1/\sqrt m$ and variance as $1/m$. This also identifies where independence is needed: correlated agent noise does not automatically give the same reduction.

### Corollary D.5: checking the Taylor remainder

The paper expands the gradient to second order:

$$
\nabla\mathcal L(\theta+\epsilon)
=g+H\epsilon+\frac12\nabla^3\mathcal L(\theta)[\epsilon,\epsilon]+R_3(\epsilon).
$$

Zero mean removes the linear term and covariance turns the quadratic term into $q/2$. This structure is central to the proposed explanation.

The covariance-to-trace step can be checked componentwise using $\mathbb E[\epsilon_j\epsilon_l]=\Gamma_{jl}$:

$$
\left[\mathbb E\,\nabla^3\mathcal L(\theta)
[\epsilon,\epsilon]\right]_i
=\sum_{j,l}\partial_{ijl}\mathcal L(\theta)\,\Gamma_{jl}
=\partial_i\operatorname{Tr}(\nabla^2\mathcal L(\theta)\Gamma).
$$

The last equality uses interchangeable mixed derivatives and symmetric $\Gamma$, held fixed under differentiation. This second-moment identity also holds for other centered distributions; their expected remainders must still be controlled separately.

**Review check:** the appropriate integral remainder for this expansion is

$$
R_3(\epsilon)=\frac12\int_0^1(1-s)^2
\nabla^4\mathcal L(\theta+s\epsilon)
[\epsilon,\epsilon,\epsilon]\,ds,
\qquad
\|R_3(\epsilon)\|\leq\frac{L_4}{6}\|\epsilon\|^3.
$$

The paper instead uses kernel $(1-s)^3/6$ and bound $L_4/24$, which do not match this gradient expansion. For $\mathcal L(x)=1-\cos x$ at $\theta=0$, the remainder is $\sin x-x\sim-x^3/6$ and exceeds the claimed $|x|^3/24$ bound. Cubic order survives, but explicit constants in subsequent sufficient conditions require reconsideration. [Corollary D.5, p.31](https://arxiv.org/pdf/2507.06542v4#page=31)

### Theorem D.6: the earlier rate used for comparison

The paper restates an earlier DSGD bound with $\eta\lesssim p/L$:

$$
T=\mathcal O\!\left(
\frac{\sigma^2}{m\varepsilon^2}
+\frac{\sqrt p\,\sigma+\zeta}{p\varepsilon^{3/2}}
+\frac1{p\varepsilon}\right)
\cdot\bigl(\mathcal L(\theta^{(0)})-\mathcal L^\star\bigr)
$$

As in the paper's comparison, smoothness constants and other fixed factors are suppressed here. The point is to expose communication and heterogeneity terms. Koloskova 2020's original Theorem 2 also includes a local-update window and more general assumptions; this simplified form is connected to Kong 2021's restatement. Comparing upper bounds under different assumptions does not establish actual model superiority. [D.6, p.31](https://arxiv.org/pdf/2507.06542v4#page=31)

### Theorem D.7: expanding Steps A→B→C

D.7 and main Theorem 1 state, with common initialization, Assumptions 2–3, and $\eta<2/L_2$,

$$
T=\mathcal O\!\left(
\frac{\sigma^2}{m\varepsilon^2}+\frac1\varepsilon+
\sum_{t=0}^{T-1}U^{(t)}\right)
\cdot\bigl(\mathcal L(\theta^{(0)})-\mathcal L^\star\bigr)
$$

Table 1 and the proof's final expression instead insert $1/\varepsilon$ before the last sum. The cross-term coefficient in $U$ is also inconsistent across locations. The following calculation separates a directly checkable conditional derivation from those unresolved statements.

**Step A — decompose descent.** As a review addition, suppose the homogeneous expansion in D.3 is valid, or the extra heterogeneity term has been separately controlled. Define a gradient-unit residual by $\bar g_t=g_t+q_t/2+r_t$. Smoothness and independent noise give

$$
\begin{aligned}
\mathbb E_t\mathcal L(\bar\theta^{(t+1)})
&\leq\mathcal L(\bar\theta^{(t)})
-\left(\eta-\frac{\eta^2L_2}{2}\right)\|g_t\|^2\\
&\quad+\frac{\eta^2L_2-\eta}{2}g_t^\top q_t
+\frac{\eta^2L_2}{8}\|q_t\|^2
+\frac{\eta^2L_2\sigma^2}{2m}+\mathcal R_t,
\end{aligned}
$$

$$
\mathcal R_t=-\eta g_t^\top r_t
+\eta^2L_2(g_t+q_t/2)^\top r_t
+\frac{\eta^2L_2}{2}\|r_t\|^2.
$$

The cross-term factor $1/2$ follows from $\|g+q/2\|^2=\|g\|^2+g^\top q+\|q\|^2/4$. It appears in original D.8 but is missing in main Equation (9) and D.17. Gradient-unit $r_t$ must also be distinguished from a parameter-update-unit $\delta^{(t)}$, or factors of $\eta$ change.

**Step B — control disagreement terms.** Covariance trace and the third-derivative bound give

$$
\|q_t\|\leq L_3\Xi_t^2,\qquad
\frac1m\sum_k\|\delta_k^{(t)}\|^3\leq\sqrt m\,\Xi_t^3
$$

For the second inequality, set $a_k=\|\delta_k\|$ and use $\sum a_k^3\leq(\sum a_k^2)^{3/2}$. The paper's reference to concavity of $x^{3/2}$ is incorrect, although this norm inequality is valid. With $m,\eta$ fixed, residual terms start at cubic order and $\|q_t\|^2$ at fourth order. The omitted non-IID first-order term does not satisfy that argument.

Mapping the five groups in original Equation (D.8) to the gradient-unit $r_t$ above exposes their orders. This is a **review correspondence with explicitly separated coefficients**.

| Group | Term in the preceding descent bound | Order in $\Xi_t$ |
| --- | --- | --- |
| $T_1$ | $\eta^2L_2\|q_t\|^2/8$ | Fourth |
| $T_2$ | $(\eta^2L_2-\eta)g_t^\top q_t/2$ | Second; the sign-sensitive central term |
| $T_3$ | $-\eta g_t^\top r_t$ | Third and higher |
| $T_4$ | $\eta^2L_2(g_t+q_t/2)^\top r_t$ | Third and fifth |
| $T_5$ | $\eta^2L_2\|r_t\|^2/2$ | Sixth |

This uses the homogeneous-expansion bound $\|r_t\|\leq L_4\sqrt m\,\Xi_t^3/6$. For $0\leq\Xi_t\leq1$, fourth-, fifth-, and sixth-order terms can each be upper-bounded by a constant times a cubic term. This explains why the authors retain the favorable quadratic contribution while collecting the rest into a cubic residual. It does not establish those same orders after including the missing non-IID term.

**Step C — sum through time.** Collect correctly scaled corrections as $\eta\widetilde U_t$. The tilde distinguishes this review quantity from inconsistent definitions of $U^{(t)}$ in the paper. Set $c=1-\eta L_2/2>0$ and $\Delta=\mathcal L(\bar\theta^{(0)})-\mathcal L^\star$. Telescoping yields

$$
\frac1T\sum_{t=0}^{T-1}\mathbb E\|g_t\|^2
\leq\frac{\Delta}{\eta cT}
+\frac1{cT}\sum_{t=0}^{T-1}\mathbb E\widetilde U_t
+\frac{\eta L_2\sigma^2}{2mc}
$$

The last term is the constant-step noise floor. Original D.12 includes an extra $1/\eta$ on the correction term, whereas dividing its preceding telescoping inequality gives the coefficient above.

This explains the intended **sign-sensitive descent analysis** without certifying the paper's displayed explicit rate or constants. Average stationarity still does not directly guarantee terminal test accuracy or Definition 2's loss inequality. [D.7, pp.31–34](https://arxiv.org/pdf/2507.06542v4#page=32)

### Proposition D.8: why small disagreement might help

First consider why the gradient margin $\mu_t$ matters. A negative sign alone is insufficient if the leading term's magnitude vanishes. The main text motivates the margin using the Polyak–Łojasiewicz condition. Distinguish its constant as $\mu_{\mathrm{PL}}$:

$$
\frac12\|g_t\|^2\geq\mu_{\mathrm{PL}}
(\mathcal L(\bar\theta^{(t)})-\mathcal L^\star)
\quad\Longrightarrow\quad
\|g_t\|\geq
\sqrt{2\mu_{\mathrm{PL}}
(\mathcal L(\bar\theta^{(t)})-\mathcal L^\star)}.
$$

If this condition actually holds, a positive gradient margin follows while the objective remains above optimum. The paper does not prove PL for the experimental networks, and the resulting margin can shrink near an optimum. This is different from a fixed positive lower bound throughout training.

Assumption 4 literally requires every PSD $\Sigma$ and every $\theta$ to satisfy

$$
\nabla\mathcal L(\theta)^\top
\nabla\operatorname{Tr}(\nabla^2\mathcal L(\theta)\Sigma)<0
$$

This fails for $\Sigma=0$ or $\nabla\mathcal L(\theta)=0$. Scaling $\Sigma$ toward zero also precludes an unnormalized uniform positive alignment margin.

To understand the intended proof, add a **local assumption at a fixed nonstationary point, with nonzero directions normalized to unit length**. Then

$$
F_t(v)=\frac{\nabla^3\mathcal L(\bar\theta^{(t)})[v,v,g_t]}{\|g_t\|},
\qquad \|v\|=1
$$

is continuous on the compact unit sphere. If it is strictly negative in every direction, its attained maximum is negative too. Hence some $\gamma_t>0$ gives

$$
g_t^\top q_t\leq-\gamma_t\|g_t\|\Xi_t^2
\leq-\gamma_t\mu_t\Xi_t^2
$$

A uniform lower margin $\gamma^*>0$ across all times requires a further argument.

For $\eta>1/L_2$, the leading term is negative and quadratic, while a residual upper bound is cubic. The elementary inequality $-a\Xi_t^2+C\Xi_t^3<0$ holds for $0<\Xi_t<a/C$. This is the core dominance argument. It does not say that more disagreement is always better or that acceleration occurs for arbitrarily small learning rates.

Combining with stability requires $1/L_2<\eta<2/L_2$, and noise remains. Remark D.2 discusses increasing agents with desired accuracy, for example $m>3\sigma^2/((2-\ell)\varepsilon)$. A fixed network, positive noise, and a large constant step do not jointly imply arbitrary-precision convergence. [D.8, Remarks D.1–D.2, pp.34–36](https://arxiv.org/pdf/2507.06542v4#page=35)

### Proposition D.9: Critical Consensus Edge

The last step translates small disagreement into a communication condition. Retaining the paper's constants,

$$
\begin{aligned}
K&=(\eta L_2-1)\gamma^*\mu_t,\\
C_{\mathrm{lin}}&=(\eta L_2+L_4/24)\sqrt m L_1,\\
\Sigma_{\mathrm{high}}&=\frac{\eta L_2L_3^2}{8}
+\frac{\eta\sqrt mL_2L_3}{2}
+\frac{\eta mL_2L_4^2}{1152}
\end{aligned}
$$

For $\Xi_t\leq1$, a polynomial bound reduces the residual budget to a linear and a quadratic component. Allocating $K/2$ to each gives the proposed sufficient condition

$$
\Xi_t<\min\left\{\frac{K}{2C_{\mathrm{lin}}},
\sqrt{\frac{K}{2\Sigma_{\mathrm{high}}}}\right\}
$$

These constants inherit the coefficient issues above and are not certified here as corrected sufficient constants.

The paper substitutes the following for the left-hand disagreement:

$$
\sqrt{\frac{24(1-p)\eta^2}{p^2}(\phi^2+\sigma^2)}
$$

Corollary D.2, however, bounds **$\mathbb E\Xi_t^2$**. Its square root is an RMS bound, not a bound for every realized $\Xi_t$. For example, let $\Xi_t=0.8$ with probability 0.01 and zero otherwise. Its RMS is 0.08, below 0.1, but the realized value still exceeds 0.1 with positive probability. All realizations even satisfy $\Xi_t\leq1$.

A probability guarantee would require an argument such as Markov's inequality, $\Pr(\Xi_t\geq a)\leq\mathbb E\Xi_t^2/a^2$, or stronger concentration. Alternatively, the expected descent proof must consistently control all relevant moments. This connection is not supplied. D.9 also omits sharpening from its listed assumptions while using it in the proof.

Thus the **design intuition that smaller late-stage gradient margins require stronger mixing** should be separated from a rigorous algorithm guaranteeing mergeability in every run. [D.9, pp.36–37; main Proposition 3](https://arxiv.org/pdf/2507.06542v4#page=36)

<span id="review-interpretation-and-next-checks" aria-hidden="true"></span>

## Review interpretation and open questions

**Review interpretation.** The paper's strength is an empirical question that separates individual models' global performance from the global performance of their weight average. Its theory proposes a higher-order geometric mechanism, but the current statements do not establish a universal non-IID merging guarantee.

Three concrete next checks are useful: compare early, late, and terminal merging at matched volume using independent validation and multiple seeds; separate fixed disjoint local data from resampled label-skew streams; and preserve heterogeneous-Hessian terms while proving probabilistic consensus control. These are **proposed follow-ups, not experiments run for this review**.

## Sources and reading scope

- Paper: *On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning* · Tongtian Zhu et al. · first public date 2025-07-09 · [reviewed version](https://arxiv.org/abs/2507.06542v4) · reviewed 2026-10-23.
- Scope read: Version 4, appendices A–D and results D.1–D.9 in source order. Sections 1–6 are linked to part 1.
- Figure sources: The Figure-1(b) thumbnail and this part’s Figures 1–2 or C.5–C.6 are author assets from v4 (CC BY 4.0), PDF-extracted/rasterized to lossless WebP with data and labels retained. [Paper](https://arxiv.org/abs/2507.06542v4).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
