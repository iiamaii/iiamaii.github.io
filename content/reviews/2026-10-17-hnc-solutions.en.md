---
title: "[1/2] HNC: Do Networks with the Same Outputs Compute the Same Way?"
description: "Follow the paper's sections 1–7 to see how HNC changes RNN memory, ViT representations, and RL behavior while preserving a specified observable."
date: "2026-10-17"
publishedAt: "2026-10-17T22:12:00+09:00"
updatedAt: "2026-10-23T15:26:23+00:00"
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

Hessian Null Space Continuation (HNC) keeps one trained network's outputs (the anchor's) approximately fixed on a finite probe set while taking small steps along flat directions of the loss curvature, searching for alternative networks with different internal computations. On an ImageNet-pretrained vision Transformer (ViT-S/16), moving the weights by only 1.3% in norm, with less than 1% loss of ImageNet top-1 accuracy, drives the maximum cross-layer centered kernel alignment (CKA) below that of seven trained comparison models and even a randomly initialized ViT. [Original §4, pp.6–7](https://arxiv.org/pdf/2609.38081v1#page=6) **Review interpretation.** The abstract's claim of representations that differ more than any independently trained model of other architectures and objectives holds for these seven models under CKA, backed by off-probe evaluation and a random-weight-noise control; but what the ViT experiments show is a change in representational similarity, and mechanism-level evidence that the way memory is implemented has changed comes from the 64-unit recurrent neural network (RNN) experiments.

This part follows main sections 1–7 through the method and four experimental families; [\[2/2\] Mathematics, algorithms, and validation](/en/reviews/hnc-mechanics/) covers the derivations, ablations, and off-probe validation in appendices A–J.

## 1. Introduction: What Computations Exist Inside a Connected Solution Region?

Overparameterized networks have many local minima with similarly low loss, and which one training reaches depends on optimizer, data, architecture, and task. Studies of loss geometry find many nearly flat directions around trained solutions, and **mode connectivity** research shows that independently trained solutions are often joined by nonlinear or even linear low-loss paths. According to the authors, this literature characterizes the connected region mainly through connectivity and loss; Lubana et al. (2023), who did look at internal computation, compared pairs of trained networks. Whether the low-loss region around a single solution contains genuinely different representations and mechanisms remains open.

Meanwhile, **solution degeneracy** research shows that one task can admit a family of solutions with different out-of-distribution generalization and different geometry and dynamics of their internal representations. Degeneracy here means multiple compatible solutions to one task. Some work deliberately finds alternatives by retraining with similarity penalties or hard constraints, or by following Hessian eigenvectors from a saddle, but how these solutions relate in weight space is poorly understood. The authors' question sits where the two lines meet: **how much representational diversity exists in the connected, function-preserving region around one trained solution, and how can that region be traversed constructively?**

There are two motivations. Trained solutions are shaped by optimizer biases as well as by the task (for example, SGD's preference for low-rank, low-norm solutions), so characterizing everything a model admits separates task demands from optimizer effects. Practically, representationally distinct solutions reachable through local, function-preserving updates would expose degrees of freedom for fine-tuning, model merging, and post-hoc editing. HNC starts from the anchor, explores function-preserving alternatives using local curvature, and can be steered toward specified properties by any differentiable objective. [Original §1, pp.1–3](https://arxiv.org/pdf/2609.38081v1#page=1)

## 2. Method

### 2.1 Loss Landscape Curvature Reveals Function-Preserving Directions

Let $\boldsymbol\theta$ be the weight vector, $\boldsymbol\theta_0$ the anchor, and $f_{\boldsymbol\theta}(x)$ the output. Equation (1) defines a **function-matching loss**, rather than loss against task labels.

$$
\mathcal L(\boldsymbol\theta)
=\frac12\mathbb E_{x\sim\mathcal D}
\left[\left\|f_{\boldsymbol\theta}(x)
-f_{\boldsymbol\theta_0}(x)\right\|^2\right].
$$

$\mathcal D$ is the input distribution. In practice, a fixed sample $\mathbf X$, the **probe set**, estimates this expectation. For classification, the target is the entire anchor logit vector rather than the label.

The anchor reproduces its own outputs exactly, so it is a global minimizer with zero loss and zero gradient. A small parameter displacement $\boldsymbol\delta$ therefore changes the loss only at second order, as in equation (2):

$$
\begin{aligned}
\mathcal L(\boldsymbol\theta_0+\boldsymbol\delta)
&=\frac12\boldsymbol\delta^\top\mathbf H\boldsymbol\delta
+O(\|\boldsymbol\delta\|^3),\\
\mathbf H&=\nabla^2_{\boldsymbol\theta}\mathcal L(\boldsymbol\theta_0).
\end{aligned}
$$

Large eigenvalues of the Hessian $\mathbf H$ identify **sharp directions** along which outputs change rapidly; near-zero ones identify **flat directions** with little output drift. With eigenvectors $\mathbf v_i$ and eigenvalues $\lambda_i$, the paper defines an approximate null space:

$$
\mathcal V_0=\operatorname{span}
\{\mathbf v_i:\lambda_i\le\epsilon\}.
$$

The authors note that trained networks typically have a few large and many near-zero Hessian eigenvalues, so this approximate null space often spans a substantial fraction of the parameter dimensions. [§2.1, p.3](https://arxiv.org/pdf/2609.38081v1#page=3)

### 2.2 Hessian Null Space Continuation

![Original Figure 1: HNC alternates flat parameter steps and output restoration](/assets/reviews/hnc/overview.webp)

[Enlarge Figure 1](/assets/reviews/hnc/overview.webp)

**Read Figure 1 as A→D→B.** Height in A is function-matching loss; the horizontal axes are conceptual parameter directions. Purple arrows in D take flat steps; teal arrows restore outputs. Repeating them reaches a network like the one in B, with the same input-output mapping but different internal representations of the same inputs. B's PC axes describe activations, not the parameter space of A. C selects flat directions with an eigenvalue cutoff $\epsilon$, and E steers the step toward an objective. [Figure 1, p.2](https://arxiv.org/pdf/2609.38081v1#page=2)

A single step within the null space explores only the immediate neighborhood of the anchor, so going further takes many steps. The authors identify two problems. The Hessian is a second-order approximation, so even a step along a zero-eigenvalue direction can change the loss through higher-order terms; and away from the anchor its null space may stop aligning with the truly flat directions. HNC handles both by alternating null-space steps with function restoration (equation (3)):

$$
\begin{aligned}
\tilde{\boldsymbol\theta}_t
&=\boldsymbol\theta_t+\eta\mathbf d_t,\\
\boldsymbol\theta_{t+1}
&=\tilde{\boldsymbol\theta}_t
-\alpha\nabla\mathcal L(\tilde{\boldsymbol\theta}_t).
\end{aligned}
$$

$\eta$ is the exploration step size and $\alpha$ the restoration learning rate. The equation shows one restoration step; in practice there are $m$, and Algorithm 1 normalizes $\mathbf d_t$. If the loss still exceeds the ceiling $\tau$ after restoration, the authors take this as a sign that the Hessian's linear approximation has broken down: the step is **rejected**, the step size shrinks, and the null space is recomputed at the current weights (relinearization). The authors liken this to the classical predict-correct structure of numerical continuation, which takes a small step along a curve's tangent and then corrects back onto the curve.

#### 2.2.1 Forming the Null Space

With $P$ parameters, the Hessian is $P\times P$, too large to store or decompose. HNC computes the $k$ smallest eigenpairs with LOBPCG (locally optimal block preconditioned conjugate gradient), which uses only matrix–vector products from automatic differentiation, reducing time and memory from $O(P^3)$ and $O(P^2)$ to $O(kP)$. Because the Hessian is positive semidefinite at the anchor, the null space is spanned by the eigenvectors with the numerically smallest eigenvalues. **Appendices A–B specify the implemented Gauss–Newton operator $\mathbf G=\mathbf J^\top\mathbf J$, with $\mathbf J$ the Jacobian of the probe outputs.** It equals the Hessian at the anchor and stays positive semidefinite along the walk; [\[2/2\] B](/en/reviews/hnc-mechanics/) gives the derivation.

The relative cutoff is $\epsilon=\mu_{\mathrm{rel}}\lambda_1$, with $\lambda_1$ the largest eigenvalue at the current linearization point. Because the cutoff is relative, the null space is recomputed at every relinearization; $\mu_{\mathrm{rel}}$ is chosen per experiment in $10^{-7}$–$10^{-3}$, and appendix sweeps cover a wider range. [§2.2.1, A–B, pp.4,18–20](https://arxiv.org/pdf/2609.38081v1#page=18)

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

Flat components mostly pass through and sharp components shrink, hence the name **soft projection**. Implementation substitutes $\mathbf G$ and solves with conjugate gradients (CG) instead of forming an inverse. Objectives can target CKA distance, dynamics, or weight sparsity. [§2.2.2, A.3](https://arxiv.org/pdf/2609.38081v1#page=4)

## 3. HNC drives RNNs to learn drastically different dynamics in a memory task

In the **3-bit flip-flop (3BFF)** task, a network must output the most recent nonzero input on each of three channels, giving $2^3=8$ binary memory states. The authors train 64-unit tanh RNNs with backpropagation through time (BPTT) and Adam. The trained anchor stores the eight states at eight stable fixed points, hidden states that stay unchanged when input is turned off, which form the vertices of a cube in activation space.

**Review supplement: the memory task and fixed points.** Zero input means “no new instruction,” not “set memory to zero.” For one channel, $+1,0,0,-1,0$ requires outputs $+1,+1,+1,-1,-1$, excluding any interval before the first pulse. Appendix F.1's RNN update and the zero-input fixed-point condition are:

$$
\begin{aligned}
\mathbf h_t&=\tanh(\mathbf W_{\mathrm{ih}}\mathbf u_t+\mathbf b_{\mathrm{ih}}
+\mathbf W_{\mathrm{hh}}\mathbf h_{t-1}+\mathbf b_{\mathrm{hh}}),\\
\mathbf y_t&=\mathbf W_{\mathrm{out}}\mathbf h_t+\mathbf b_{\mathrm{out}},\\
\mathbf h^\star&=\tanh(\mathbf b_{\mathrm{ih}}+\mathbf W_{\mathrm{hh}}\mathbf h^\star+\mathbf b_{\mathrm{hh}})
\quad(\mathbf u_t=\mathbf0).
\end{aligned}
$$

A fixed point is stable when nearby state perturbations decay back toward it. Constant output $\mathbf W_{\mathrm{out}}\mathbf h_t$ can coexist with movement in the readout null space, which is exactly what Figure 2D shows. [§3, Appendix F.1](https://arxiv.org/pdf/2609.38081v1#page=29)

HNC compares undirected exploration with steering that increases one of two distances. CKA distance $1-\mathrm{CKA}$ measures how much the representational geometry of hidden activations has changed; dynamical similarity analysis (DSA) distance measures how much the recurrent dynamics have changed up to invertible transformations. The walks use an adaptive step size and periodic relinearization.

![Original Figure 2: Changes in fixed points, output, representation, and dynamics on 3BFF](/assets/reviews/hnc/rnn-dynamics.webp)

[Enlarge Figure 2](/assets/reviews/hnc/rnn-dynamics.webp)

**Reading Figure 2.** A shows the task and the anchor's eight fixed points. In B, task loss stays below $0.01$ for all HNC walks (left), and the middle and right panels plot representational and dynamical distance against relative weight movement $\|\Delta\boldsymbol\theta\|/\|\boldsymbol\theta_0\|$. Undirected walks reach distances comparable to independently trained seed pairs; steered walks go further with less weight movement. In C, the undirected endpoint loses its eight fixed points, which are replaced by a continuous manifold with one region per memory state, while the CKA- and DSA-steered endpoints change the number of fixed points (0, 7, and 15 stable fixed points as labeled in the figure). All three endpoints solve the task.

D is the paper's most intuitive scene. During input-free memory periods, the undirected endpoint's hidden states collapse to eight compact points in the **readout subspace** seen by the output layer, but keep moving within eight clusters in the **readout-null subspace** invisible to it (a subspace of activation space, distinct from the parameter-space null space of §2). The authors interpret this network as holding memory in distinct regions of state space rather than at fixed points, keeping outputs stable despite ongoing internal dynamics.

In E, the CKA-steered endpoint recovers less after hidden-state perturbations, and the DSA-steered endpoint's memory drifts over extended input-free periods. The authors conclude that networks with matching in-distribution input-output mappings can still behave differently under extended or perturbed conditions. F uses multidimensional scaling (MDS) to embed pairwise distances among 131 networks: the anchor, ten independently trained seeds, and every checkpoint of three walks of each type. In representation distance, independent seeds and undirected walks stay near the anchor while steered walks extend far beyond them; in weight distance, steered walks stay closer to the anchor than both undirected walks and independent seeds. The authors read this as a decoupling of representational from weight-space distance. [§3, Figure 2, pp.4–6](https://arxiv.org/pdf/2609.38081v1#page=4)

The main experiment uses five anchors and three replicates per arm and anchor, and all 131 networks in Figure 2F exceed $0.999$ accuracy. Trials last 100 steps, pulse probability is 0.3, and the model has 4,611 parameters. Appendix F adds out-of-distribution (OOD) inputs such as weak or rare pulses, on which some alternatives lose accuracy before the anchor; a pruning-steered walk that shrinks the network from 64 to 11 units; and resumed 8,000-step SGD after which the steered properties remain ([\[2/2\] F](/en/reviews/hnc-mechanics/)). [F.1–F.8, pp.29–35](https://arxiv.org/pdf/2609.38081v1#page=29)

## 4. HNC reveals underexplored degrees of freedom in ViTs

The Platonic Representation Hypothesis (PRH) holds that representations become increasingly similar across models as models scale and tasks get harder, and attributes this to convergence toward a shared representation of reality. HNC asks a different question: **how much freedom to change representations remains inside one trained model when its function is held approximately fixed?**

The anchor is an ImageNet-pretrained ViT-S/16 with approximately 22M parameters. Its logits are matched on 512 Places-365 validation images, the dataset on which PRH was evaluated. Using the CLS feature after every block plus the final pre-logits feature (13 depths), steering lowers the **maximum CKA over every pair of anchor and current-model layers**. Because the hard maximum is not differentiable, a smooth log-sum-exp drives the steering and the hard maximum is reported.

![Original Figure 3: Separate evaluation of ViT representation similarity and prediction preservation](/assets/reviews/hnc/vit-representations.webp)

[Enlarge Figure 3](/assets/reviews/hnc/vit-representations.webp)

**Read A, B, and C together.** A uses relative weight displacement, not walk step, on its horizontal axis. CKA declines both on the probe and on held-out images never seen during HNC, and the mutual k-nearest-neighbor (kNN) alignment, which is never steered, falls comparably. The authors take this as evidence that HNC reorganizes representations in general rather than exploiting one metric on one image set. B's left panel shows agreement with the anchor's predictions and its right panel ImageNet accuracy against labels. The walk keeps the anchor's top-1 predictions on the probe throughout; on ImageNet, never used as a constraint, top-1 accuracy drops by less than 1% and top-5 is nearly unchanged.

In C, the endpoint's weights have moved only 1.3% in norm, yet its CKA falls below **the seven trained models in this comparison and a randomly initialized ViT**. The comparison models, ViT-S/in1k, ViT-L/in21k, ViT-B/in21k, DeiT3-S, ViT-S/DINO, ResNet-50, and ConvNeXt-S, differ in size, architecture, objective, and data. A–C show means and standard deviations across five random probe sets.

D's cosine-similarity matrices show that pairwise similarities among probe images change drastically from anchor to endpoint, and E's MDS of 20 images from four classes shows neighborhoods reorganized within and across classes. F takes the five most similar image pairs per model and scores semantic similarity (DINOv2), color composition, and spatial layout as percentiles among all pairs; 50 is the random-pair reference. The endpoint's closest pairs remain close in semantics and color, but their layout similarity approaches the random baseline. From this the authors argue that cross-model convergence need not reflect a unique representation fully specified by the task; it may come from an optimization bias that samples a narrow subset of solutions. [§4, G.1–G.2, pp.6–7,36](https://arxiv.org/pdf/2609.38081v1#page=6)

**Off-probe preservation (appendices G.5–G.6).** CKA-steered endpoints have small accuracy deficits on ImageNet, V2, Rendition, and Sketch, with error consistency around 0.8. Their image-level agreement with the anchor is comparable to, and generally higher than, a model that repeats the anchor's last fine-tuning stage with only the seed changed. Gaussian weight noise matched to the accuracy deficit leaves CKA above 0.98, so the authors conclude that generic damage cannot explain the representation change. Mutual-kNN-steered endpoints preserve behavior less well than the different-seed reference, and the authors conclude from the two experiments together that the degree of functional preservation depends on the walk configuration and the steering objective. Detailed values and replicate units are in [\[2/2\] G](/en/reviews/hnc-mechanics/). [G.5–G.6, pp.39–42](https://arxiv.org/pdf/2609.38081v1#page=39)

## 5. HNC finds behaviorally distinct, high-reward policies in reinforcement learning

Unlike supervised targets, a scalar reward in reinforcement learning rarely specifies the optimal sequence of actions over an episode. The authors argue that this underspecification admits behaviorally distinct policies at similar reward, including **reward-hacking** strategies that exploit the reward function instead of accomplishing the task, so **the preserved quantity here is return rather than the action distribution.** From a state-action buffer collected by the anchor policy $\pi_{\boldsymbol\theta_0}$, with advantages $A_i^0$, the importance-sampling surrogate return is:

$$
\phi_R(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
\frac{\pi_{\boldsymbol\theta}(a_i\mid s_i)}
{\pi_{\boldsymbol\theta_0}(a_i\mid s_i)}A_i^0.
$$

The steering objective is the KL divergence between the two policies' action distributions on buffer states:

$$
\phi_B(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
D_{\mathrm{KL}}\!\left(
\pi_{\boldsymbol\theta_0}(\cdot\mid s_i)
\,\Vert\,
\pi_{\boldsymbol\theta}(\cdot\mid s_i)\right).
$$

$s_i,a_i$ are observed states and actions, and $N$ is the buffer sample count. Because the visited state distribution changes with the policy, the buffer is periodically recollected from the current policy and the surrogate and its null space are recomputed; actual episode return is evaluated separately. Continuous-action experiments steer mean-action MSE under a frozen Gaussian covariance ([\[2/2\] I.2](/en/reviews/hnc-mechanics/)). [§5, I.1–I.3, pp.7–8,49–50](https://arxiv.org/pdf/2609.38081v1#page=7)

![Original Figure 4: Alternative plume-tracking strategies and boat-race reward hacking](/assets/reviews/hnc/rl-strategies.webp)

[Enlarge Figure 4](/assets/reviews/hnc/rl-strategies.webp)

**Left, A–C: another way to follow odor.** Like real flies, the 64-unit recurrent anchor trained with proximal policy optimization (PPO) surges upwind on detecting odor and casts across the wind after losing it. The alternative instead slides to the plume's edge and tracks it smoothly. In B, action KL keeps rising while episode return and success rate are maintained, and even increase early in the walk. C pairs cast counts and distance from the centerline over 240 shared initial conditions: the alternative casts less and on average stays farther from the centerline.

The 30-step walk refreshes its buffer every five steps; standard evaluation uses common random numbers and 60 fixed initial conditions. In the OOD evaluation of I.5, with 240 initial conditions per condition, the anchor fails every sparse-odor and noisy-wind episode while the analyzed alternative succeeds on some, and under a wind switch the alternative more than doubles the anchor's homing rate. At step 20 all conditions collapse together, which the authors identify as the point where the walk leaves the reward level set.

**Right, D–E: reward can survive while intent fails.** Boat race awards proxy reward +3 each time the agent enters an arrow tile clockwise, while the true return, hidden from the agent, measures net clockwise progress around the track. Stepping on and off one tile therefore collects proxy reward without progress. E plots proxy and true return separately: the endpoints split into two clusters, and the three most divergent adopt this reward hack and lose net progress entirely.

An essential condition is that **all ten anchors were PPO-trained on true return** and all follow the track. An anchor trained on the proxy alone discovers the exploit by itself, leaving nothing for HNC to find, so the experiment first builds well-behaved policies and then searches while preserving proxy return. In MuJoCo Ant-v3 (appendix I.7), HNC likewise preserves mean return while changing leg timing and failure modes ([\[2/2\] I](/en/reviews/hnc-mechanics/)). [§5, I.4–I.7, pp.8,51–53](https://arxiv.org/pdf/2609.38081v1#page=51)

## 6. HNC measures loss landscape geometry

The final main experiment uses HNC to measure the local geometry of the solution set. It addresses a question earlier work approached through loss barriers between independently trained networks or stylized theoretical models, namely how model size and task complexity shape the solution set, by measuring directly from single trained networks. It varies CNN width $w\in\{16,24,32,48,96\}$ and CIFAR-100 class count $C\in\{7,10,20,50,100\}$, with three seeds per combination: **75 anchors**. Each class contributes 300 training images without augmentation; 512 of each anchor's training images constrain its outputs. Class count is the chosen proxy for task complexity.

![Original Figure 5: Effective null fraction and normalized curvature against width and class count](/assets/reviews/hnc/geometry.webp)

[Enlarge Figure 5](/assets/reviews/hnc/geometry.webp)

**A and B measure different things.** A's effective null fraction is the fraction of directions along which a step of 1% of the weight norm raises function-matching loss by at most $\epsilon=0.05$. Tying the cutoff to an output-drift tolerance allows comparison across conditions with different overall Hessian scales. The fraction grows with width and shrinks with class count, yet remains large even for the smallest network on the hardest task.

B measures the cost of moving along these directions. To avoid constraining output sensitivity itself, it replaces the drift cutoff with a common relative threshold $\mu=10^{-6}\lambda_{\max}$ and measures the normalized curvature $\mathbf d^\top\mathbf H\mathbf d/(nC)$ of each step, where $\mathbf d$ has unit norm and $n$ is the probe size; implementation and J.3 use $\mathbf G$ in place of $\mathbf H$. Curvature generally rises with class count and falls with width. Bands are standard errors over three seeds, and threshold sweeps in the appendix preserve the trends.

The authors conclude that flat directions remain abundant, but on harder tasks the directions HNC follows become more sensitive to weight changes. They present this stiffening of the landscape around a local minimum as a potential mechanism for prior findings that the loss landscape fragments into separate basins as tasks get harder. [§6, J.1–J.4, pp.8–9,53–55](https://arxiv.org/pdf/2609.38081v1#page=8)

## 7. Discussion

The authors read §6 alongside Huang et al. (2025), who found that harder tasks and larger models reduce the variability of dynamics across independently trained RNNs, whereas HNC shows larger null spaces for larger models on the same task. They interpret this as larger models admitting more alternatives within a local region even as training converges to a narrower subset of them, consistent with the stronger simplicity bias that PRH invokes to explain representational convergence. The gap lies between **the diversity a task permits** and **the diversity ordinary optimization actually selects**.

An expected objection is that parameter symmetries and reparameterizations can also reorganize representations while preserving function (Theiss et al., 2026). The authors answer that HNC's solutions go beyond reparameterization: they change the dynamical regime qualitatively, alter out-of-distribution generalization, and produce qualitatively distinct behavioral strategies. Moreover, the symmetries in that work are discrete operations such as neuron duplication that may change network size, whereas HNC makes small continuous changes to the weights of a trained network.

As limitations, the authors note that function preservation is enforced only approximately on a finite probe set, so output agreement on unseen inputs, especially under distribution shift, is not guaranteed, and the reachable diversity depends on probe coverage and allowed drift. HNC is a local search and cannot tell whether several basins exist, but alternating it with global retraining broadens the search (appendix E).

The preserved quantity can be any differentiable function of the network. Preserving task loss instead would explore the broader task-compatible set (appendix D), and the steering objective could encode performance on a second task or alignment with neural data (the Brain-Score steering of appendix H). When the goal is to change the function, sharp directions are the most effective. The authors list as future applications empirical studies of the optimizer's implicit bias, scaling to large language models, model merging, fine-tuning, and editing, and auditing reward specifications by uncovering behaviorally distinct or reward-hacking policies. [§7, pp.9–10](https://arxiv.org/pdf/2609.38081v1#page=9)

## Review interpretation and open questions

**Review interpretation.** The strongest evidence is the 3BFF RNN. While task loss stays below the ceiling, the number of fixed points changes, and the undirected endpoint holds memory through continuing movement in the readout-null subspace. Of the arguments the authors give in §7, the change in out-of-distribution generalization is the most direct answer to the reparameterization objection: a symmetry that preserves function on every input would also behave identically on inputs such as long input-free periods or OOD pulses, yet the 3BFF alternatives fail differently from the anchor on exactly these inputs (Figure 2E, appendix F). The same result also shows that off-probe preservation is only approximate.

In ViTs, the controls are the strength. CKA and kNN alignment fall together on images never used as constraints, and random noise matched for accuracy loss barely moves CKA. However, the kNN-steered endpoint keeps more alignment than a randomly initialized ViT (G.3) and preserves function less well, so the size of the effect is tied to both the steering metric and the preservation level. The trends in §6's effective null fraction point in the same direction as a lower bound on the linear null space set purely by output-constraint and parameter counts ([\[2/2\] J](/en/reviews/hnc-mechanics/)); the effect of task difficulty is supported more directly by the normalized curvature and by the growth in representation dimensionality (J.4).

Three questions remain. Do the mechanism-level differences seen in RNNs also appear in ViT internal circuits? How does the preservation–displacement trade-off of CKA versus kNN steering change once probe size and allowed drift are matched? And do the §6 trends persist when the class count is fixed and only the number of probe output constraints varies?

## Sources and reading scope

- Paper: *Traversing the solution space of neural networks with Hessian Null Space Continuation* · Ann Huang et al. · first public date 2026-09-29 · [reviewed version](https://arxiv.org/abs/2609.38081v1) · reviewed 2026-10-23.
- Scope read: Version 1, sections 1–7, with appendix settings, metrics, and contrary results linked where needed; detailed A–J coverage is in part 2.
- Figure sources: The Figure-1 thumbnail and body figures in this part are author assets from v1 (CC BY 4.0), PDF-rasterized and WebP-encoded with content, layout, text, and arrows retained. [Paper](https://arxiv.org/abs/2609.38081v1).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
