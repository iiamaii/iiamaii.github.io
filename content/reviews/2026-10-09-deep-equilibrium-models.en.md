---
title: "Deep Equilibrium Models: learning an endpoint instead of stacking layers"
description: "An accessible guide to fixed points, Broyden solves and implicit differentiation, with the main proofs and language-model experiments explaining memory savings, runtime costs and convergence conditions."
date: "2026-10-09"
publishedAt: "2026-10-09T20:40:07+09:00"
updatedAt: "2026-10-23T15:26:15+00:00"
topics: ["language models", "fixed points", "implicit learning", "memory efficiency", "numerical optimization"]
visibility: "public"
lang: "en"
translationKey: "deep-equilibrium-models"
paperTitle: "Deep Equilibrium Models"
paperPublishedDate: "2019-09-03"
authors: ["Shaojie Bai", "J. Zico Kolter", "Vladlen Koltun"]
year: "2019"
paperUrl: "https://arxiv.org/abs/1909.01377v2"
thumbnail: "/assets/reviews/deep-equilibrium-models/deq-core.svg"
thumbnailAlt: "A comparison between unrolled shared layers and a DEQ: solve numerically for an equilibrium, then solve a linear system at that endpoint to obtain parameter gradients."
---

Deep Equilibrium Models (DEQs) solve for the equilibrium of a weight-tied block by root finding and train it through implicit differentiation, making activation storage constant in depth. On WikiText-103, comparing medium models with adaptive embeddings, the DEQ lowers test perplexity (PPL) from 23.6 to 23.2 relative to an 18-layer Transformer-XL and cuts memory, measured at sequence length 150 and batch size 15 without word embeddings, from 9.0 to 3.7 GB, but takes about 2.82 times as long to train. [Table 3](https://arxiv.org/pdf/1909.01377v2#page=8) · [Table 4](https://arxiv.org/pdf/1909.01377v2#page=9) **Review interpretation.** The claim of large memory savings at unchanged PPL and equal model size holds on the length- and batch-controlled measurements of Tables 2–3, but the abstract's “similar computational requirements to existing models” does not fit Table 4, where training is 2.4–2.8 times slower.

<strong>Reading the overview figure.</strong> The top row unrolls a shared block into successive layers. The middle row numerically solves $f_\theta(z^\star_{1:T};x_{1:T})=z^\star_{1:T}$. The bottom row solves a linear system at that endpoint for gradients. Arrows indicate computation order within each row. Iterative work remains; the key is avoiding backpropagation through the entire forward solver trajectory. This is an original illustration made for this review. [Enlarge](/assets/reviews/deep-equilibrium-models/deq-core.svg)

The article follows original §1–6. Appendix A’s proof of Theorem 1 accompanies §3.1.2; Appendix B’s Theorem 2 and Appendix C’s representation result accompany §3.2; Appendix D’s convergence observations open §3. Appendix E’s further remarks and the review interpretation follow the authors’ conclusion.

## 1. Introduction

The authors start from the observation that weight-tied deep sequence models compete with the state of the art, and ask: if the same transformation is applied at every layer, what is the limit of this process, and how can it be modeled? Their answer, the DEQ, solves for the transformation's fixed point with any root-finding method and backpropagates through it by implicit differentiation without storing intermediate activations, training with constant memory equivalent to a single layer. Applying it to TrellisNet and universal-transformer-style models, they claim accuracy on par with, or slightly better than, the corresponding layer-based networks despite constant memory. [Original §1](https://arxiv.org/pdf/1909.01377v2#page=1)

## 2. Background

Original §2 reviews four strands of prior work. Deep sequence models are functions satisfying causality, meaning that output at time $t$ depends only on $x_{1:t}$; the section introduces three families, RNNs, temporal convolutional networks (TCNs), and self-attention, along with TrellisNet and the universal transformer that connect them. Among memory-efficient networks, gradient checkpointing reduces an $L$-layer network's memory to $O(\sqrt L)$ at the cost of extra forward passes, while reversible networks reconstruct each layer's activations from the next. Continuous-view ODE methods backpropagate through the entire latent trajectory with an adjoint; the authors distinguish DEQ as solving directly for the equilibrium with a quasi-Newton method and backpropagating through the fixed point regardless of the path that reached it. [§2, pp. 2–3](https://arxiv.org/pdf/1909.01377v2#page=2)

The last strand is implicit layers. The authors note that a DEQ can also be viewed as a single layer whose output is defined as the solution of a nonlinear equation, and claim that their models are the largest-scale practical application of implicit layers they are aware of. They distinguish DEQ from recurrent backpropagation (RBP) by (1) using equilibrium as a replacement for depth in general networks, with a proof of universality; (2) applying it beyond fixed-input RNNs to state-of-the-art architectures; and (3) scaling to practical tasks, whereas RBP has typically been applied at small scale. [§2, p. 3](https://arxiv.org/pdf/1909.01377v2#page=3)

### Review supplement: comparing background sources

The table below is a review supplement comparing the indicated sections of five background sources with DEQ §2–3.

| Research | Why read it alongside DEQ? | Difference from DEQ |
| --- | --- | --- |
| [Trellis Networks for Sequence Modeling](https://arxiv.org/pdf/1810.06682v2#page=3), §3–4 | Input injection, depth-wise weight sharing, and a bridge between convolution and recurrent networks | Unrolls finite depth; DEQ defines the output through the block's equilibrium. |
| [Universal Transformers](https://arxiv.org/pdf/1807.03819v3#page=2), §2 | Repeats shared attention and transition operations with adjustable computation | Uses finite repetitions or position-wise halting, and depth-time representations. DEQ solves a stationary mapping with a residual-based stopping rule. |
| [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366v4#page=1), §1–2 | Solver-defined computation and memory savings through an adjoint | Solves a trajectory from an initial state to a finite time; DEQ solves a fixed-point equation and a local linear system. |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/pdf/1604.06174v2#page=4), §4 | Checkpointing trades recomputation for activation storage | Preserves the original finite network's computation; DEQ changes the output definition to an equilibrium. |
| [Reviving and Improving Recurrent Back-Propagation](https://proceedings.mlr.press/v80/liao18c/liao18c.pdf#page=3), §3–4 | Equilibrium gradients, invertibility requirements, and Neumann/CG computation | The same implicit-differentiation lineage; its differences from DEQ are the three listed by original §2 in the paragraph above. |

## 3. The Deep Equilibrium Sequence Model

An ordinary deep network has different parameters at different layers. Sharing weights across depth prevents parameter count from growing with the number of repetitions, but does not remove the need to store intermediate activations for backpropagation. DEQ starts from a structure with both <strong>weight sharing</strong> and <strong>input injection at every update</strong>. With input $x$ fixed, it repeatedly updates hidden state $z$. [§3, Equation 3](https://arxiv.org/pdf/1909.01377v2#page=3)

$$
\begin{aligned}
z^{[0]}_{1:T}&=0,\\
z^{[i+1]}_{1:T}&=f_\theta(z^{[i]}_{1:T};x_{1:T}).
\end{aligned}
$$

Here $i$ indexes <strong>computational depth</strong>, not the order of tokens in a sentence. Both $x$ and $z$ can represent an entire sequence. For autoregressive language modeling, causal convolution or an attention mask prevents output at position $t$ from depending on future inputs. [§2, §4](https://arxiv.org/pdf/1909.01377v2#page=3)

Instead of asking how many layers should define the output, DEQ asks whether a state unchanged by another application of the block can define it. The authors hypothesize, and support with observations, that for certain classes of $f_\theta$ weight-tied models tend to converge to a fixed point as depth increases. Appendix D shows differences between successive layer activations shrinking in a trained TrellisNet and a weight-tied transformer on test sequences of length 100–800, and notes that the transformer often oscillates around the actual fixed point. [§3; Appendix D, Figure 4](https://arxiv.org/pdf/1909.01377v2#page=14)

### 3.1 The Deep Equilibrium Approach

A fixed point, or equilibrium $z^\star$, satisfies:

$$
\begin{aligned}
z^\star_{1:T}&=f_\theta(z^\star_{1:T};x_{1:T}),\\
g_\theta(z_{1:T};x_{1:T})&=f_\theta(z_{1:T};x_{1:T})-z_{1:T},\\
g_\theta(z^\star_{1:T};x_{1:T})&=0.
\end{aligned}
$$

Repeatedly applying $f_\theta$ and solving the equation $g=0$ can target the same answer, but their computation and convergence properties differ. DEQ adopts the root-finding view. [§3, Equation 4; §3.1.1, Equation 5](https://arxiv.org/pdf/1909.01377v2#page=4)

Consider the scalar function $f(z; x)=0.5z+0.75x$ with $x=2$. This is an explanatory example for this review, not an experiment from the paper.

$$
\begin{gathered}
0\longrightarrow1.5\longrightarrow2.25\longrightarrow2.625\longrightarrow\cdots\longrightarrow3,\\
z^\star=0.5z^\star+1.5\quad\Longrightarrow\quad z^\star=3.
\end{gathered}
$$

A finite number of iterations gives a value close to $3$; the equilibrium equation defines the output as $3$. Real DEQs do not generally have such a closed-form answer. They use numerical solvers with tolerances and iteration limits. “Infinite depth” describes the output definition. It does not mean executing infinitely many operations or always obtaining an exact answer at finite cost.

#### 3.1.1 Forward Pass

Newton's method updates a candidate state using the residual $g_\theta(z)$ and its Jacobian. A Jacobian records how each output changes when each state component changes. Building and inverting that matrix is expensive for large sequences. The paper uses <strong>Broyden's method</strong> to update an approximation of the inverse Jacobian. [§3.1.1, Equation 6; §3.1.3, Equation 10](https://arxiv.org/pdf/1909.01377v2#page=4)

$$
\begin{aligned}
z^{[i+1]}_{1:T}&=z^{[i]}_{1:T}-\alpha B^{[i]}_{g_\theta}\,g_\theta(z^{[i]}_{1:T};x_{1:T}),\\
B^{[i]}_{g_\theta}&\approx\left(J_{g_\theta}\big|_{z^{[i]}_{1:T}}\right)^{-1}.
\end{aligned}
$$

$\alpha$ is a step size. The inverse approximation uses the paper’s $B^{[i]}_{g_\theta}$. It is maintained through low-rank updates instead of constructing a new large inverse each time; the paper initializes the approximation at $-I$. Computation stops when the residual becomes small enough or the iteration limit is reached. Solver settings therefore form part of the model's execution procedure.

Crucially, <strong>a root can exist even when naive iteration is unstable</strong>. The function $f(z)=1.2z+1$ has a fixed point at $-5$, but iteration from zero produces $1, 2.2, 3.64, \ldots$. Existence of a root, convergence of repeated application, and successful root finding are distinct questions. Original §3.2 likewise states that finding the equilibrium reliably and efficiently generally requires $f_\theta$ to be stable and constrained. [§3.2, p. 5](https://arxiv.org/pdf/1909.01377v2#page=5)

#### 3.1.2 Backward Pass

After obtaining an endpoint, training requires its sensitivity to parameter changes. <strong>Implicit differentiation</strong> supplies that sensitivity. Differentiate both sides of $z^\star_{1:T}=f_\theta(z^\star_{1:T};x_{1:T})$ with respect to $\theta$. [Theorem 1, Equation 8; Appendix A, Equations 13–14](https://arxiv.org/pdf/1909.01377v2#page=13)

$$
\begin{aligned}
J_{f_\theta}&=\left.\frac{\partial f_\theta}{\partial z_{1:T}}\right|_{z^\star_{1:T}},\\
\frac{\mathrm dz^\star_{1:T}}{\mathrm d\theta}
&=J_{f_\theta}\frac{\mathrm dz^\star_{1:T}}{\mathrm d\theta}
+\frac{\partial f_\theta(z^\star_{1:T};x_{1:T})}{\partial\theta},\\
(I-J_{f_\theta})\frac{\mathrm dz^\star_{1:T}}{\mathrm d\theta}
&=\frac{\partial f_\theta(z^\star_{1:T};x_{1:T})}{\partial\theta},\\
\frac{\mathrm dz^\star_{1:T}}{\mathrm d\theta}
&=(I-J_{f_\theta})^{-1}\frac{\partial f_\theta(z^\star_{1:T};x_{1:T})}{\partial\theta}.
\end{aligned}
$$

Theorem 1 writes the following row-gradient formula. Since $J_{g_\theta}=J_{f_\theta}-I$, it is equivalent to the derivation above. The adjoint below uses transposed column-vector gradients.

$$
\frac{\partial\ell}{\partial\theta}
=-\frac{\partial\ell}{\partial z^\star_{1:T}}
\left(J_{g_\theta}\big|_{z^\star_{1:T}}\right)^{-1}
\frac{\partial f_\theta(z^\star_{1:T};x_{1:T})}{\partial\theta}.
$$

Let $\ell$ be the loss and $q=\nabla_{z^\star_{1:T}}\ell$ its endpoint gradient. Using column-vector notation, solve the following linear system instead of explicitly constructing the inverse:

$$
\begin{aligned}
q&=\nabla_{z^\star_{1:T}}\ell,\\
(I-J_{f_\theta})^\top v&=q,\\
\nabla_\theta\ell&=\left(\frac{\partial f_\theta}{\partial\theta}\right)^\top v.
\end{aligned}
$$

This is the core proof: local derivatives at the endpoint determine its parameter sensitivity. Computation uses <strong>vector–Jacobian products (VJPs)</strong>, which differentiate against a vector without building the full Jacobian. The entire series of forward solver updates need not remain in an automatic-differentiation graph. Direct parameter dependencies through the output head or loss must be differentiated separately and added. [§3.1.2; §3.1.3, Equation 11](https://arxiv.org/pdf/1909.01377v2#page=5)

For the scalar example, choose target $1$ and loss $\frac12(z^\star-1)^2$. Then $q=2$, $J_{f_\theta}=0.5$, and $v=4$. Writing $f(z;x)=az+bx$ with $a=0.5$, $b=0.75$ gives $\partial\ell/\partial a=12$ and $\partial\ell/\partial b=8$.

The derivation has conditions. The function must be differentiable around the selected solution, and <strong>$I-J_{f_\theta}$ must be invertible</strong>. The paper's inverse notation presupposes this requirement. A nearly singular system can amplify state or gradient errors despite a small residual. Moreover, the formula describes an exact equilibrium, while implementation uses an approximate state and approximate linear solve. The paper itself notes that the tolerance $\varepsilon$ should ideally be as small as possible for Theorem 1's analytical gradients to be accurate. [§5, p. 9](https://arxiv.org/pdf/1909.01377v2#page=9)

#### 3.1.3 Accelerating DEQ by Approximating the Inverse Jacobian

Equation (10) adds a rank-one update to an inverse approximation instead of recomputing an inverse. The difference vectors come from consecutive search states. The displayed denominator must be nonzero.

$$
\begin{aligned}
\Delta z^{[i+1]}&=z^{[i+1]}_{1:T}-z^{[i]}_{1:T},\\
\Delta g_\theta^{[i+1]}&=g_\theta(z^{[i+1]}_{1:T};x_{1:T})-g_\theta(z^{[i]}_{1:T};x_{1:T}),\\
B_{g_\theta}^{[i+1]}&=B_{g_\theta}^{[i]}
+\frac{\bigl(\Delta z^{[i+1]}-B_{g_\theta}^{[i]}\Delta g_\theta^{[i+1]}\bigr)
\Delta z^{[i+1]\top}B_{g_\theta}^{[i]}}
{\Delta z^{[i+1]\top}B_{g_\theta}^{[i]}\Delta g_\theta^{[i+1]}}.
\end{aligned}
$$

**Review-supplied derivation.** Multiplying the updated matrix by $\Delta g_\theta^{[i+1]}$ cancels the scalar numerator and denominator, giving $B_{g_\theta}^{[i+1]}\Delta g_\theta^{[i+1]}=\Delta z^{[i+1]}$. This secant condition matches only the change between the two newest points, and a near-zero denominator makes the update numerically unstable. The backward linear problem also uses VJPs and indirect solvers to avoid storing an explicit Jacobian. [§3.1.3, Eqs. (10)–(11)](https://arxiv.org/pdf/1909.01377v2#page=4)

### 3.2 Properties of Deep Equilibrium Models

Original §3.2 starts from the fact that a DEQ's output is agnostic to the choice of forward RootFind algorithm, then discusses memory cost, the choice of $f_\theta$, and stacking DEQs.

#### Memory and the choice of transformation

DEQ's memory claim concerns <strong>activation storage as depth increases</strong>. The backward pass needs only the equilibrium sequence $z^\star_{1:T}$, input-related variables $x_{1:T}$, and $f_\theta$; because the backward linear system needs only a VJP of size $N\times Td$, the $N\times(Td)^2$ Jacobian is never constructed ($N$ is the minibatch size). The authors say this constant memory lets models that previously needed multiple GPUs, half precision, or checkpointing fit on a single GPU. The memory benefit does not depend on the type of $f_\theta$, and the gated activation and layer normalization of the two instantiations are presented as examples of stable transformations that constrain output ranges. [§3.2, p. 5](https://arxiv.org/pdf/1909.01377v2#page=5)

Costs that scale with sequence length, hidden width, and batch size remain. The authors themselves attribute the largest Transformer-XL's memory requirements to very wide hidden features, batch sizes, and training-time sequence lengths, which a DEQ would not decrease. [§5, p. 8](https://arxiv.org/pdf/1909.01377v2#page=8)

#### Theorem 2 and Appendix C’s Theorem 3: representation and width

<strong>Theorem 2: two stacked DEQs can be represented by one larger DEQ.</strong> Let the first equilibrium be $w^{(1)}_{1:T}$ and the second $w^{(2)}_{1:T}$. Place both updates in a combined state $[w^{(1)}_{1:T};w^{(2)}_{1:T}]$. [Appendix B, Equations 15–16, p. 13](https://arxiv.org/pdf/1909.01377v2#page=13)

$$
\Gamma_\Theta\!\left(\begin{bmatrix}w^{(1)}_{1:T}\\w^{(2)}_{1:T}\end{bmatrix};x_{1:T}\right)
=\begin{bmatrix}
f_{\theta^{[1]}}(w^{(1)}_{1:T};x_{1:T})\\
v_{\theta^{[2]}}(w^{(2)}_{1:T};w^{(1)}_{1:T})
\end{bmatrix}.
$$

At a fixed point of $\Gamma_\Theta$, both original equations hold. Reading the final part of the state reproduces the stacked output. However, the combined hidden dimension is the <strong>sum</strong> of the two state dimensions. The main-text statement of Theorem 2 prints $\mathsf{RootFind}(g^f_{\theta^{[2]}};\mathsf{RootFind}(g^v_{\theta^{[1]}};x_{1:T}))$, pairing $f$ and $v$ with the wrong parameter sets relative to their definitions, whereas the restatement in Appendix B (Equation 15) uses $g^v_{\theta^{[2]}}$ and $g^f_{\theta^{[1]}}$ consistently. This review follows Appendix B; the theorem's conclusion is unaffected.

<strong>Theorem 3: a finite untied network can be embedded in a wider tied network.</strong> Put each layer's state in a separate part of a larger vector and construct a block-shift update that passes one part's result to the next. Repetition computes the original layer outputs in sequence. [Appendix C, p. 14](https://arxiv.org/pdf/1909.01377v2#page=14)

Again, this is a width-expanding construction, and the paper notes that actual DEQs use dense matrices $W_z$ and $W_x$ rather than this sparse structure. The authors cite the theorem as evidence that weight tying causes no notable loss of representational capacity.

**Review-supplied derivation: Theorem 3’s induction.** Equations (17)–(21) place $W^{[1]},\ldots,W^{[L-1]}$ on the first block subdiagonal of $W_z$ and $W^{[0]}$ in the first block of $W_x$. The blockwise nonlinearity $\sigma$ and $\tilde b$ retain each original layer’s activation and bias. Subscript $k$ below is a **review auxiliary index** for a block.

$$
\begin{aligned}
\tilde z^{[i+1]}_1&=\sigma^{[0]}(W^{[0]}x+b^{[0]}),\\
\tilde z^{[i+1]}_k&=\sigma^{[k-1]}(W^{[k-1]}\tilde z^{[i]}_{k-1}+b^{[k-1]}),
\quad 2\leq k\leq L,\\
\tilde z^{[i]}_k&=z^{[k]}\quad\text{for }i\geq k.
\end{aligned}
$$

The first block equals the original first-layer output after one update and stays fixed. If block $k-1$ is correct for $i\geq k-1$, the next update makes block $k$ equal the original layer-$k$ output. After $L$ updates, all blocks therefore contain the original layer outputs. Hidden width becomes the sum of original layer widths, with original parameters stored inside the larger shared matrices. [Appendix C, Theorem 3](https://arxiv.org/pdf/1909.01377v2#page=14)

## 4. Instantiations of DEQ

Original §4 relies on the forward and backward analyses being independent of $f_\theta$'s internal structure, presents two instantiations, and argues that they make DEQ compatible with all three major families: transformers, RNNs, and TCNs. [§4, Figure 1, p. 6](https://arxiv.org/pdf/1909.01377v2#page=6)

| Block | State update | Input and context handling |
| --- | --- | --- |
| DEQ-TrellisNet | Causal 1D convolution and LSTM-style gating | Input projection injected at every shared convolution update; earlier segment states and padding supply context |
| DEQ-Transformer | Self-attention, feed-forward transformations, residuals and layer normalization | Input injected into Q/K/V computation; causal masking, relative positions and earlier segment equilibria |

The Transformer implementation uses Transformer-XL's context handling and relative positions. Its depth update must remain the same function, so its update differs from recurrent transformers that use depth-dependent time embeddings.

## 5. Experiments

The experiments cover a copy task, Penn Treebank (PTB), and WikiText-103 (WT103). Both instantiations use Broyden's method; during training the forward and backward tolerances are $\varepsilon=\sqrt{T}\cdot10^{-5}$ and $\sqrt{T}\cdot10^{-8}$, and at inference the tolerance is relaxed to $\sqrt{T}\cdot10^{-2}$. On WT103, the DEQ-Transformer uses relative positional embeddings and sequences of length 150 for both training and inference. PPL exponentiates the average negative log probability of target tokens; lower is better. [§5, p. 7; Appendix F](https://arxiv.org/pdf/1909.01377v2#page=7)

<strong>The copy task is a small stress test of retaining information across a delay.</strong> The model reproduces ten initial symbols after a waiting interval. For $T=400$, the total sequence length is $T+20=420$. Models have approximately 14–16K parameters. Reported losses are $3.5\times 10^{-6}$ for DEQ-Transformer, $2.7\times 10^{-5}$ for TCN, $0.0501$ for LSTM, and $0.0491$ for GRU; the authors read this as substantially better memory retention than recurrent architectures such as LSTM and GRU. [Table 1, p. 7; Appendix F, p. 16](https://arxiv.org/pdf/1909.01377v2#page=16)

The following table transcribes selected language-model rows. <strong>All memory values use sequence length 150, batch size 15, and exclude word embeddings</strong>; as the paper's note says, this does not reflect the hyperparameters actually used for training. Transformer-XL models in Table 3 are not weight-tied unless marked otherwise. [Table 2 note; Table 3, pp. 7–8](https://arxiv.org/pdf/1909.01377v2#page=7)

| Data | Model | Total / non-embedding parameters | Test PPL ↓ | Memory GB ↓ |
| --- | --- | --- | --- | --- |
| PTB | 60-layer TrellisNet, auxiliary loss, without MoS | 24M / 20M | 57.0 | 8.5 |
| PTB | DEQ-TrellisNet | 24M / 20M | 57.1 | 1.2 |
| WT103 | 70-layer TrellisNet, auxiliary loss | 180M / 45M | 29.2 | 24.7 |
| WT103 | Same TrellisNet with checkpointing | 180M / 45M | 29.2 | 5.2 |
| WT103 | DEQ-TrellisNet | 180M / 45M | 29.0 | 3.3 |
| WT103 | 18-layer medium Transformer-XL, adaptive embedding | 110M / 72M | 23.6 | 9.0 |
| WT103 | Medium DEQ-Transformer, adaptive embedding | 110M / 70M | 23.2 | 3.7 |
| WT103 | Small Transformer-XL, weight-tied, 16 layers | 138M / 4.5M | 34.9 | 6.8 |
| WT103 | Small DEQ-Transformer | 138M / 4.5M | 32.4 | 1.1 |

On PTB, PPL is nearly unchanged while memory falls from $8.5$ to $1.2\,\mathrm{GB}$, approximately 85.9%. WT103 Trellis memory falls from $24.7$ to $3.3\,\mathrm{GB}$, approximately <strong>86.6%</strong>, but against the checkpointed $5.2\,\mathrm{GB}$ baseline the reduction is approximately <strong>36.5%</strong>. For transformers, the only same-size weight-tied baseline is the small pair: PPL falls from $34.9$ to $32.4$ and memory from $6.8$ to $1.1\,\mathrm{GB}$, approximately <strong>83.8%</strong>. The three pairs against non-weight-tied Transformer-XL models are medium 16-layer $8.5\to2.7\,\mathrm{GB}$ (68.2%), small 4-layer $4.8\to1.1\,\mathrm{GB}$ (77.1%), and the adaptive-embedding pair $9.0\to 3.7\,\mathrm{GB}$ (<strong>58.9%</strong>). All percentages use rounded table values. [Table 3](https://arxiv.org/pdf/1909.01377v2#page=8)

Original §5 states that on both instantiations DEQ yields “an over 80% (up to 88%) reduction in memory consumption,” and that even against baselines that are not weight-tied it achieves “a reduction of over 60%.” **Review interpretation.** Over 80% holds for the three weight-tied pairs (85.9%, 86.6%, 83.8%), but the largest reduction computed from printed values is 86.6%, and about 86.9% even at the extremes of rounding, so the 88% in the abstract and §5 cannot be reproduced from the tables. Over 60% also fails for the adaptive-embedding pair at 58.9%. I found no other row or measurement in the paper that yields 88%; neither discrepancy changes the conclusion of large memory savings, only the stated maximum and lower bound.

PTB Table 2 includes DARTS at PPL $55.7$, and WT103 Table 3 includes a much larger Transformer-XL at $18.7$, both below DEQ. As the Table 3 caption states, the authors' unit of comparison is state-of-the-art models of the same size. [Tables 2–3](https://arxiv.org/pdf/1909.01377v2#page=8)

### Figure 2: solver computation during training

On initialization, the authors say that reliably approximating the equilibrium with quasi-Newton methods requires starting with small operator norms in the weight matrices. Neither instantiation was sensitive to a specific scheme, which they attribute partly to nonlinearities such as $\sigma$, $\tanh$, and LayerNorm helping make $f_\theta$ contractive; in practice they initialize from $\mathcal N(0,0.05)$. [§5, p. 8](https://arxiv.org/pdf/1909.01377v2#page=8)

<figure class="review-figure" id="paper-figure-2">
<a href="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" width="1732" height="509" alt="Original Figure 2: Broyden iterations per time step across training epochs on the left; log equilibrium residual versus function evaluations on the right." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 2 · Equilibrium solving can become harder during training</span>The two panels were isolated from surrounding prose on v2 p. 8. Axes, legends and curves are preserved. <span class="figure-links"><a href="https://arxiv.org/pdf/1909.01377v2#page=8">Shaojie Bai, J. Zico Kolter, Vladlen Koltun, 2019, Figure 2</a> · <a href="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" target="_blank" rel="noopener noreferrer">Enlarge</a> · Figure quotation for critical review; rights remain with the original authors</span></figcaption>
</figure>

<strong>The left panel describes computational work.</strong> Its horizontal axis is training epoch; by the paper's definition, the vertical axis is total Broyden iterations divided by sequence length, which is 150 in this figure (see its title). Blue is forward computation and red is backward computation; both increase as training progresses. The authors suggest that training may push the operator norm of $J_{f_\theta}$ to larger values, making the fixed point harder to solve, and attribute the much smaller backward count to the simplicity of the linear system in Equation (11). [Figure 2, §5, p. 8](https://arxiv.org/pdf/1909.01377v2#page=8)

<strong>The right panel describes a residual.</strong> The horizontal axis counts function evaluations; the vertical axis plots $\|f(z)-z\|_2$ logarithmically. Reading from the figure, at epoch 1 both naive iteration of the weight-tied transformer and DEQ reach about $10^{-4}$, but DEQ needs roughly half as many evaluations. At epoch 12, naive iteration oscillates around $2\times10^{1}$, while DEQ reaches about $10^{-4}$ within roughly 200 evaluations. The authors cite this as evidence that DEQs almost always converge to the sequence-level fixed point, much more efficiently than the original weight-tied transformers.

### Runtime (Table 4)

DEQ runtime depends strongly on the number of Broyden steps needed to reach equilibrium, and the authors state that DEQs are typically slower than layer-based networks. Table 4 reports `DEQ time / baseline time`, so values above one indicate slower execution. [Table 4, p. 9](https://arxiv.org/pdf/1909.01377v2#page=9)

| WT103 baseline | Training time ratio | Inference time ratio |
| --- | --- | --- |
| 18-layer Transformer | 2.82× | 1.76× |
| 70-layer TrellisNet | 2.40× | 1.64× |

### Figure 3: precision and quality

<figure class="review-figure" id="paper-figure-3">
<a href="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" width="1737" height="469" alt="Original Figure 3: forward residual tolerance versus validation PPL on the left, and forward iteration limit versus validation PPL on the right." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 3 · Stopping too early reduces quality</span>The panels were isolated from v2 p. 9, preserving data and original labels. <span class="figure-links"><a href="https://arxiv.org/pdf/1909.01377v2#page=9">Shaojie Bai, J. Zico Kolter, Vladlen Koltun, 2019, Figure 3</a> · <a href="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" target="_blank" rel="noopener noreferrer">Enlarge</a> · Figure quotation for critical review; rights remain with the original authors</span></figcaption>
</figure>

The left panel shows where relaxing the forward residual tolerance damages quality. Its horizontal axis is logarithmic; smaller values are stricter. Validation PPL stays similar across small tolerances, then worsens sharply when tolerance becomes too loose. The right panel shows improving PPL as the iteration limit grows, with diminishing gains from additional computation. The original includes shaded regions but does not specify their aggregation. [Figure 3, §5, p. 9](https://arxiv.org/pdf/1909.01377v2#page=9)

These experiments use the medium DEQ-Transformer without adaptive embeddings and report validation PPL. The text states that $\varepsilon <0.1$ or an iteration limit of 30 (<strong>on sequence length 75</strong>) is generally sufficient for competitive performance, and cites the rapid divergence at large tolerances as evidence that a poor estimate of the equilibrium hurts DEQ performance.

## 6. Conclusion

The authors conclude that DEQ, which directly solves for and optimizes the sequence-level fixed point, needs only $O(1)$ memory at training time, is agnostic to the choice of forward root solver, and is versatile enough to subsume very different architectures. They summarize the experiments as showing good temporal memory retention, scaling to realistic large-scale sequence tasks, and performance competitive with, or slightly better than, state-of-the-art methods. [Original §6](https://arxiv.org/pdf/1909.01377v2#page=9)

## Appendix E: More Remarks on DEQ

Appendix E covers practical conditions that the mathematical formulation does not reveal. [Appendix E, p. 15](https://arxiv.org/pdf/1909.01377v2#page=15)

- <strong>Applicability of other techniques.</strong> Layer normalization and weight normalization work as is, but without layers it is unclear how to apply intermediate auxiliary losses. Following TrellisNet, dropout uses the same mask at all levels, and adding dropout slows quasi-Newton convergence. Weight normalization finds more stable parameters and slows the growth in Broyden iterations seen in Figure 2.
- <strong>Imbalance within minibatches.</strong> Sequences need different numbers of iterations to reach equilibrium, so those that converge quickly wait for slower ones. The authors find the imbalance relatively small but note that it can sometimes make GPU utilization inefficient.
- <strong>Warm-up with shallow nets.</strong> Pretraining a shallow weight-tied stack, for example two layers, and using its parameters to initialize the DEQ accelerates training.
- <strong>Training with subsequences.</strong> On sequences longer than 1,000, forward fixed points can be hard to solve accurately, especially early in training, so sequences can be split into a few subsequences, with the same split applied in the backward pass. The memory cost equals that of applying $f_\theta$ once to the whole sequence.

<span id="review-interpretation-and-follow-up-questions" aria-hidden="true"></span>

## Review interpretation and open questions

**Review interpretation.** DEQ changes depth from a list of layers to retain into a state equation to solve. The most direct evidence for the memory claim is the comparison with a same-size weight-tied small Transformer-XL, where PPL falls from 34.9 to 32.4 while memory falls from 6.8 to 1.1 GB. DEQ also uses 36.5% less memory than the 70-layer TrellisNet with checkpointing, so the comparison with an existing technique that trades storage for recomputation is measured as well. The cost is time: the text concedes that DEQs are typically slower than layer-based models, and in Figure 2 Broyden iterations grow as training proceeds. Equilibrium convergence holds almost always in the settings the authors observed, but its stability depends on practical conditions such as small-operator-norm initialization, normalization, and weight normalization. The abstract's claim that DEQs “often improve performance” at similar parameter counts points in the right direction on WT103, but apart from the small pair the PPL differences are 0.1–0.4, and no run-to-run variance is reported.

The remaining question is <strong>which mappings admit usable equilibria under which conditions, and at what solve precision the quality–memory–time tradeoff becomes favorable</strong>. Subsequent work includes [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656), which extends the approach to multiresolution vision, and [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342), which addresses stability and efficiency directly. For these two follow-ups, the review relies on their official abstracts and metadata.

## Sources and reading scope

- Paper: *Deep Equilibrium Models* · Shaojie Bai et al. · first public date 2019-09-03 · [reviewed version](https://arxiv.org/abs/1909.01377v2) · reviewed 2026-10-23.
- Scope read: Version 2, sections 1–6 and appendices A–F; theorems and proofs from A–C accompany their corresponding explanations in section 3.
- Figure sources: The thumbnail is review-authored (CC BY 4.0). Excerpts of Figures 2 and 3 are quoted for commentary with author rights retained; arXiv’s nonexclusive distribution license is not a general reuse license. [Paper](https://arxiv.org/abs/1909.01377v2).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
