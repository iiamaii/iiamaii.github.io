---
title: "[1] HNC: Do Networks with the Same Outputs Compute the Same Way?"
description: "Follow the paper's sections 1–7 to see how HNC changes RNN memory, ViT representations, and RL behavior while preserving a specified observable."
date: "2026-10-17"
publishedAt: "2026-10-17T22:12:00+09:00"
updatedAt: "2026-10-17T22:12:00+09:00"
topics: ["representation learning", "loss landscape", "mechanistic interpretability", "computational neuroscience", "reinforcement learning"]
visibility: "public"
lang: "en"
translationKey: "hnc-solutions"
paperTitle: "Traversing the solution space of neural networks with Hessian Null Space Continuation"
paperPublishedDate: "2026-09-29"
authors: ["Ann Huang", "Mitchell Ostrow", "Zhouyang Lu", "William T. Redman", "Leo Kozachkov", "Kanaka Rajan"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2609.38081v1"
thumbnail: "/assets/reviews/hnc/overview.webp"
thumbnailAlt: "Original Figure 1: HNC alternates flat parameter steps and output restoration to reach networks with different internal representations."
---

If two networks give the same answers, must they compute those answers the same way? **Hessian Null Space Continuation (HNC) starts from one trained network and finds alternatives with substantially different representations and dynamics while approximately retaining observed outputs.** An RNN can replace fixed-point memory with moving internal states; a ViT can reorganize image similarity after a small weight change. Preservation, however, is enforced approximately on a finite set of inputs. It is not equality of functions on every possible input.

This two-part review covers the 55-page [v1](https://arxiv.org/pdf/2609.38081v1). This article follows **main sections 1–7**, explaining the method and four experimental families. [[2] Mathematics, algorithms, and validation](/en/reviews/hnc-mechanics/) follows appendices A–J, including derivations, restoration failures, off-probe behavior, and additional experiments. Read 1→2.

## 1. Introduction: What Computations Exist Inside a Connected Solution Region?

Overparameterized networks admit many weights with similar training loss. **Mode connectivity** research shows that independently trained solutions can be connected by low-loss paths. **Solution degeneracy** research shows that one task can support different internal structures. Degeneracy here means multiple compatible solutions, not performance deterioration or representation collapse.

HNC joins these questions. Instead of first training two models and connecting them, **how different a computation can we find by exploring the connected low-loss region around one model?** The trained reference is the anchor. Four sources help locate the contribution.

| Prior work | Relevant contribution | Relationship to HNC |
| --- | --- | --- |
| [Garipov et al., 2018](https://arxiv.org/pdf/1802.10026v4), §3–4 | Low-loss paths between trained endpoints, including meaningful representation differences | HNC explores from one anchor without a prespecified destination. Representational diversity itself was not first observed here. |
| [Kornblith et al., 2019](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf), §2–3 | Centered kernel alignment (CKA) compares representational similarity structures | CKA evaluates and steers the search. Low CKA alone does not establish every kind of computational difference. |
| [Ostrow et al., 2023](https://arxiv.org/pdf/2306.10168v2), §2 | Dynamical similarity analysis (DSA) compares temporal computational structure | It separates recurrent dynamics from the spatial appearance of state trajectories. |
| [Huh et al., 2024](https://proceedings.mlr.press/v235/huh24a.html), §2 | The Platonic Representation Hypothesis (PRH) and observations of alignment with competence and scale | HNC distinguishes diversity permitted by a task from diversity sampled by ordinary optimization. It does not refute every empirical observation in PRH. |

Selected sections of these sources were checked; this is not an exhaustive survey of subsequent work. HNC contributes constructive curvature-based exploration and examines representations, dynamics, behavior, and geometry along the resulting paths. [Original §1, pp.1–3](https://arxiv.org/pdf/2609.38081v1#page=1)

## 2. Method

### 2.1 Loss Landscape Curvature Reveals Function-Preserving Directions

Let $\boldsymbol\theta$ be the weight vector, $\boldsymbol\theta_0$ the anchor, and $f_{\boldsymbol\theta}(x)$ the output. Equation (1) defines a **function-matching loss**, rather than loss against task labels.

$$
\mathcal L(\boldsymbol\theta)
=\frac12\mathbb E_{x\sim\mathcal D}
\left[\left\|f_{\boldsymbol\theta}(x)
-f_{\boldsymbol\theta_0}(x)\right\|^2\right].
$$

$\mathcal D$ is the input distribution. In practice, a fixed sample $\mathbf X$, the **probe set**, estimates this expectation. For classification, the target is the entire anchor logit vector, a stronger constraint than preserving one score.

The anchor matches itself exactly, so this loss and its gradient vanish there. Equation (2) approximates the effect of a small parameter displacement $\boldsymbol\delta$ by curvature:

$$
\begin{aligned}
\mathcal L(\boldsymbol\theta_0+\boldsymbol\delta)
&=\frac12\boldsymbol\delta^\top\mathbf H\boldsymbol\delta
+O(\|\boldsymbol\delta\|^3),\\
\mathbf H&=\nabla^2_{\boldsymbol\theta}\mathcal L(\boldsymbol\theta_0).
\end{aligned}
$$

Large eigenvalues of the Hessian $\mathbf H$ identify **sharp directions** sensitive to parameter changes; small ones identify **flat directions**. With eigenvectors $\mathbf v_i$ and eigenvalues $\lambda_i$, the paper defines an approximate null space:

$$
\mathcal V_0=\operatorname{span}
\{\mathbf v_i:\lambda_i\le\epsilon\}.
$$

This is not restricted to exactly zero eigenvalues. Its size depends on the threshold $\epsilon$, and the approximation is local. A large finite step in a locally flat direction need not preserve outputs. [§2.1, p.3](https://arxiv.org/pdf/2609.38081v1#page=3)

### 2.2 Hessian Null Space Continuation

![Original Figure 1: HNC alternates flat parameter steps and output restoration](/assets/reviews/hnc/overview.webp)

[Enlarge Figure 1](/assets/reviews/hnc/overview.webp)

**Read Figure 1 as A→D→B.** Height in A is output-matching loss; the horizontal axes are conceptual parameter directions. Purple arrows in D take flat steps; teal arrows restore outputs. Repetition can change the internal geometry illustrated in B. Its PC axes describe activations, not the parameter space of A. C shows the eigenvalue cutoff and E steering toward another objective. The output equality in B is schematic: experimental preservation has a tolerance and a probe scope. [Figure 1, p.2](https://arxiv.org/pdf/2609.38081v1#page=2)

Equation (3) illustrates two steps:

$$
\begin{aligned}
\tilde{\boldsymbol\theta}_t
&=\boldsymbol\theta_t+\eta\mathbf d_t,\\
\boldsymbol\theta_{t+1}
&=\tilde{\boldsymbol\theta}_t
-\alpha\nabla\mathcal L(\tilde{\boldsymbol\theta}_t).
\end{aligned}
$$

$\eta$ is the exploration step size and $\alpha$ the restoration learning rate. Actual Algorithm 1 normalizes $\mathbf d_t$ and performs $m$ restoration steps. A proposal still exceeding loss ceiling $\tau$ is **rejected**, with a reduced step size. Undirected exploration refreshes flat directions periodically and after rejection. This predict-correct procedure follows a low-loss region; it does not uniformly sample all solutions.

#### 2.2.1 Forming the Null Space

With $P$ parameters, the Hessian is $P\times P$. HNC accesses curvature through matrix–vector products. LOBPCG finds some low-eigenvalue directions for undirected exploration; steering instead solves a linear system. **Appendices A–B specify the implemented Gauss–Newton operator $\mathbf G=\mathbf J^\top\mathbf J$.** It equals the Hessian at the anchor but generally differs at points with nonzero output residual. [Part 2](/en/reviews/hnc-mechanics/) develops this distinction and derivation.

The relative cutoff is $\epsilon=\mu_{\mathrm{rel}}\lambda_1$, with $\lambda_1$ the largest eigenvalue. Main-text settings span $10^{-7}$–$10^{-3}$; appendix sweeps cover a wider range. The $k$ directions returned by LOBPCG are not a count of the full null-space dimension. [§2.2.1, A–B, pp.4,18–20](https://arxiv.org/pdf/2609.38081v1#page=18)

#### 2.2.2 Steering

To increase a differentiable objective $\varphi$, attenuate gradient components that strongly change outputs:

$$
\begin{aligned}
\mathbf d
&=\sum_i
\frac{\mathbf v_i^\top\nabla\varphi}
{1+\lambda_i/\mu}\mathbf v_i,\\
&=(\mathbf I+\mathbf H/\mu)^{-1}\nabla\varphi,\qquad
\mu=\mu_{\mathrm{rel}}\lambda_1.
\end{aligned}
$$

Flat components mostly pass through; sharp components shrink. This **soft projection** is not an exact orthogonal projector with binary spectral weights. Implementation substitutes $\mathbf G$ and solves with conjugate gradients (CG) instead of forming an inverse. Objectives can target CKA distance, dynamics, or weight sparsity. [§2.2.2, A.3](https://arxiv.org/pdf/2609.38081v1#page=4)

## 3. HNC Drives RNNs to Learn Drastically Different Dynamics in a Memory Task

The **3-bit flip-flop (3BFF)** task retains the latest nonzero input on each of three channels, yielding eight binary memory states. The authors train 64-unit tanh RNNs with BPTT and Adam. Standard solutions store the states at eight stable fixed points arranged approximately as a cube in activation space.

HNC compares undirected exploration, steering to increase CKA distance $1-\mathrm{CKA}$, and steering to increase DSA distance. CKA concerns similarities among activation samples; DSA concerns temporal state-transition structure. Movement under one metric need not match movement under the other.

![Original Figure 2: Changes in fixed points, output, representation, and dynamics on 3BFF](/assets/reviews/hnc/rnn-dynamics.webp)

[Enlarge Figure 2](/assets/reviews/hnc/rnn-dynamics.webp)

**Reading Figure 2.** A shows the original eight fixed points. B plots task loss and distances against relative weight movement $\|\Delta\boldsymbol\theta\|/\|\boldsymbol\theta_0\|$. Steering produces representation or dynamics distances larger than independently trained seed pairs. C reports 0, 7, and 15 stable fixed points in representative alternatives that still solve the task. These are numerically identified counts, not a mathematical certificate that no other fixed point exists anywhere.

D provides the key mechanism. States barely move in the **readout subspace**, but keep moving in the **readout-null subspace** invisible to the output layer. Stable output memory need not require stationary hidden activity. This activation-space null space is different from the parameter-space null space in §2.

E is an important counter-result: the CKA-steered example is less robust to hidden-state perturbation, and the DSA-steered example drifts during extended input-free memory. F embeds pairwise distances among 131 networks using MDS. Representation and weight distances separate, but a two-dimensional embedding is not a complete topological map of the solution set. [§3, Figure 2, pp.4–6](https://arxiv.org/pdf/2609.38081v1#page=4)

The main experiment uses five anchors and three replicates per arm and anchor. Representative walks maintain task MSE below $0.01$; the Figure 2F networks exceed $0.999$ accuracy. Trials last 100 steps, pulse probability is 0.3, and the model has 4,611 parameters. F.3 describes acceptance using task MSE on 128 fixed trials, which must be distinguished from the function-matching criterion of general Algorithm 1.

Appendix F tests weak or rare pulses and finds some alternatives deteriorate earlier than the anchor. It also reports one pruning-steered 64→11-unit example and preservation of endpoint characteristics after 8,000 SGD steps. These support additional uses without guaranteeing that compression ratio or stability for every seed and task. [F.1–F.8, pp.29–35](https://arxiv.org/pdf/2609.38081v1#page=29)

## 4. HNC Reveals Underexplored Degrees of Freedom in ViTs

PRH discusses increasing alignment across larger and more capable models. HNC asks a different question: **how much room to change representations remains inside one trained model when its outputs are approximately fixed?**

The anchor is an ImageNet-pretrained ViT-S/16 with approximately 22M parameters. Its logits are matched on 512 Places-365 validation images. Steering lowers the **maximum CKA over every pair of anchor and current-model layers**, using CLS/pre-logits features at 13 depths. This discourages changing only the final feature while leaving an original representation available in another layer. A smooth log-sum-exp drives optimization; evaluation reports the hard maximum.

![Original Figure 3: Separate evaluation of ViT representation similarity and prediction preservation](/assets/reviews/hnc/vit-representations.webp)

[Enlarge Figure 3](/assets/reviews/hnc/vit-representations.webp)

**Read A, B, and C together.** A uses relative weight displacement, not walk step, on its horizontal axis. CKA and the unsteered mutual-kNN alignment decline on probe and held-out images. B distinguishes agreement with anchor predictions from ImageNet accuracy against labels. The paper reports approximately 1.3% relative weight movement, less than 1% top-1 accuracy loss, and nearly unchanged top-5 accuracy. Preservation is approximate and has a small performance cost.

In C, endpoint CKA is below **the seven trained models in this comparison and a randomly initialized ViT**: ViT-S/in1k, ViT-L/in21k, ViT-B/in21k, DeiT3-S, ViT-S/DINO, ResNet-50, and ConvNeXt-S. This is not a universal ranking against every architecture and training objective. A–C show means and standard deviations across five random probe sets.

D's cosine-similarity matrices and E's MDS of 20 images show reorganized neighborhoods. F selects the five most similar image pairs per model and compares semantic similarity (DINOv2), color composition, and spatial layout as percentiles among all pairs. The random-pair reference is 50. The endpoint retains substantial semantic and color similarity while becoming less sensitive to layout. This is an observation about five selected pairs, not an exhaustive account of visual perception. [§4, G.1–G.2, pp.6–7,36](https://arxiv.org/pdf/2609.38081v1#page=6)

**Off-probe evaluation determines the strength of the claim.** G.5's CKA-steered endpoints have small accuracy deficits and imperfect prediction agreement on ImageNet, V2, Rendition, and Sketch. Error consistency is around 0.8; agreement is comparable to, or generally above, repeating the anchor's last fine-tuning stage with another seed. Gaussian weight noise matched to the accuracy deficit leaves CKA above 0.98, so generic damage alone does not explain the representation change.

G.6's mutual-kNN-steered configuration preserves behavior less faithfully: prediction match is approximately 0.85 on clean ImageNet and approximately 0.6 on Sketch. One cannot merge these configurations into a claim of unrestricted representation change with identical function. Probe size, steering, and step size matter. G.3 specifies 256 kNN probe images whereas Table 8 lists 512; that setting remains an internal source inconsistency. [G.3–G.10, pp.36–46](https://arxiv.org/pdf/2609.38081v1#page=39)

## 5. HNC Finds Behaviorally Distinct, High-Reward Policies in Reinforcement Learning

**The preserved quantity changes here.** Preserving the same action distribution would prevent substantial behavioral divergence. Given a state-action buffer from reference policy $\pi_{\boldsymbol\theta_0}$ and advantages $A_i^0$, the authors preserve a surrogate return:

$$
\phi_R(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
\frac{\pi_{\boldsymbol\theta}(a_i\mid s_i)}
{\pi_{\boldsymbol\theta_0}(a_i\mid s_i)}A_i^0.
$$

Behavioral divergence is increased with:

$$
\phi_B(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
D_{\mathrm{KL}}\!\left(
\pi_{\boldsymbol\theta_0}(\cdot\mid s_i)
\,\Vert\,
\pi_{\boldsymbol\theta}(\cdot\mid s_i)\right).
$$

$s_i,a_i$ are observed states and actions, and $N$ is the buffer sample count. This surrogate is not exact future return under every changed state distribution. Buffers are recollected periodically and actual episode return is evaluated separately. Continuous-action experiments steer mean-action MSE with frozen Gaussian covariance. It is an overall constant multiple of KL only when variances agree across action dimensions; otherwise fixed diagonal weights differ. [§5, I.1–I.3, pp.7–8,49–50](https://arxiv.org/pdf/2609.38081v1#page=7)

![Original Figure 4: Alternative plume-tracking strategies and boat-race reward hacking](/assets/reviews/hnc/rl-strategies.webp)

[Enlarge Figure 4](/assets/reviews/hnc/rl-strategies.webp)

**Left, A–C: another way to follow odor.** The PPO-trained 64-unit recurrent anchor casts across the wind after losing odor. The alternative follows the plume edge more smoothly. B plots increasing action KL alongside actual return. C pairs cast counts and distance from the centerline over 240 shared initial conditions: points below the diagonal in the cast plot indicate fewer reversals. Success is not inferred from three illustrated trajectories alone.

The 30-step walk refreshes its buffer every five steps. Standard evaluation uses common random numbers and 60 fixed initial conditions; additional OOD evaluation uses 240 per condition. The selected alternative improves under wind switches, sparse odor, and noisy wind, but I.5 also reports a collapse across conditions at step 20 when the walk leaves the preserved level set. The most distant checkpoint is not necessarily the best policy.

**Right, D–E: reward can survive while intent fails.** Boat race awards proxy reward +3 for entering arrow tiles clockwise; intended performance measures actual net clockwise progress. HNC finds **reward-hacking** policies that repeatedly enter and leave one tile. E separates proxy return from true return. The three most divergent endpoints lose net progress entirely.

An essential condition is that **all ten anchors were PPO-trained on true return**. Proxy-only training discovers the exploit itself, so the experiment first creates well-behaved policies and then searches while preserving proxy return. It does not show that HNC newly creates every risk in normal proxy-only training. Additional Ant-v3 results preserve mean return while changing leg timing and failure modes. Equal mean reward does not imply equal competence on each initial state. [§5, I.4–I.7, pp.8,51–53](https://arxiv.org/pdf/2609.38081v1#page=51)

## 6. HNC Measures Loss Landscape Geometry

The final main experiment varies CNN width $w\in\{16,24,32,48,96\}$ and CIFAR-100 class count $C\in\{7,10,20,50,100\}$, with three seeds per combination: **75 anchors**. Each class contributes 300 training images without augmentation; 512 of each anchor's training images constrain its outputs. Class count is the chosen proxy for task complexity.

![Original Figure 5: Effective null fraction and normalized curvature against width and class count](/assets/reviews/hnc/geometry.webp)

[Enlarge Figure 5](/assets/reviews/hnc/geometry.webp)

**A and B measure different things.** A's effective null fraction counts directions whose small step, 1% of weight norm, raises function-matching loss by at most $0.05$. It grows with width and decreases with class count. This is not a condition-independent measurement of exact nullity.

B uses the common relative threshold $\mu=10^{-6}\lambda_{\max}$ and measures normalized directional curvature. Main-text notation is $\mathbf d^\top\mathbf H\mathbf d/(nC)$; implementation and J.3 use $\mathbf G$. Here $\mathbf d$ has unit norm and $n$ is probe size. Curvature generally rises with class count and falls with width. Bands are standard errors over three seeds; threshold sweeps preserve the trends.

More classes also impose more output constraints. The results are not a universal law for every notion of task difficulty or a proof of disconnected basins. J.1's statement that 100% training accuracy makes an “exact zero-loss solution” is not generally valid for cross-entropy. The separate function-matching loss is nevertheless zero at its own anchor by definition. [§6, J.1–J.4, pp.8–9,53–55](https://arxiv.org/pdf/2609.38081v1#page=8)

## 7. Discussion

The distinction is between **diversity a task permits** and **diversity ordinary optimization actually discovers**. Larger trained models can resemble each other while any one large model still permits many local alternatives. Function alone need not uniquely determine representation.

Limitations are part of this conclusion. Preservation is approximate on finite probes and may fail under distribution shift, long memory intervals, or perturbation. Search is local and does not count all basins. Steering metrics and tolerances influence the alternatives found. Matrix-free computation still requires curvature products, restoration, CG, and additional RL rollouts. ViT-S continuation steps are reported at roughly one minute on an H200; for the small RNN, dense computation is faster.

Appendix H further changes Brain-Score by steering internal features while approximately retaining outputs. It changes a model's predictivity of neural recordings, not the brain itself; its ridge steering surrogate differs from the reported official PLS-based score. Scaling to LLMs and general improvements in merging, editing, and fine-tuning remain prospects in Discussion, not established benchmark improvements here. [§7, A.1, H, pp.9–10,19,46–48](https://arxiv.org/pdf/2609.38081v1#page=9)

## Review Interpretation and Open Questions

A practical reading is that HNC tests **which observations fail to identify an internal mechanism uniquely**, rather than finding the one true structure behind a function. Further comparisons should match probe size and distribution, allowed drift, and actual compute, then report representation changes alongside accuracy, prediction agreement, OOD behavior, and failures. Moving one optimized metric is insufficient. These are proposed follow-ups, not experiments performed for this review.

## Sources and Reading Scope

- Original: Huang, Ostrow, Lu, Redman, Kozachkov, Rajan, [Traversing the solution space of neural networks with Hessian Null Space Continuation, v1](https://arxiv.org/abs/2609.38081v1). First submitted September 29, 2026; reviewed October 17, 2026. It is a preprint, not labeled an accepted conference paper.
- First reading: abstract, introduction pp.1–3, Figure 1 caption, Discussion pp.9–10, appendix outline p.17, and section map. An independent orientation was saved before detailed reading.
- Second reading: main §1–7 and appendices A–J, including text, equations, tables, and captions, checked against PDF and active author LaTeX. Key method/algorithm/table pages 3,4,18–20,24,38,54 and original figures used here were visually inspected. Relevant bibliography entries were checked; not every cited paper was independently read.
- Additional originals: Garipov v4 pp.2–4; CKA pp.2–3; DSA v2 pp.2–4; PRH conference version pp.2–4. These selected sections establish path construction, metric definitions and invariances, and convergence observations, not complete reviews of all four papers.
- Organization: main order is retained, with critical experimental settings and counter-results added nearby. [Part 2](/en/reviews/hnc-mechanics/) follows A–J for mathematical details and validation. Repetitive galleries, every hyperparameter row, and the complete bibliography are compressed.
- Images: Figures 1–5 from the [authors' v1 source](https://arxiv.org/src/2609.38081v1), credited to the six authors above under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original PDF graphics were rasterized and WebP-encoded. Layout, colors, text, arrows, and data were retained; resolution and encoding changed.
- Verification: source numbers, comparison conditions, and local algebraic/numerical checks. **Not executed:** author code, GPU/RNN/ViT/RL training or walk reproduction, significance tests from raw data, complete formal verification, hardware timing/memory replication, or author confirmation of inconsistent settings.
