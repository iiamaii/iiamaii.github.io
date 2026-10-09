---
title: "Deep Equilibrium Models: learning an endpoint instead of stacking layers"
description: "An accessible guide to fixed points, Broyden solves and implicit differentiation, with the main proofs and language-model experiments explaining memory savings, runtime costs and convergence conditions."
date: "2026-10-09"
publishedAt: "2026-10-09T20:40:07+09:00"
updatedAt: "2026-10-09T20:40:07+09:00"
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

A deep neural network changes its representation through successive layers. Training usually requires retaining intermediate states for differentiation. <strong>If repeated application of the same block reaches a nearly unchanged state, could that state itself define the model's output?</strong> The 2019 paper *Deep Equilibrium Models* turns this question into an architecture and training algorithm. The forward pass finds an equilibrium; the backward pass computes how that equilibrium responds to parameters. Activation storage across depth shrinks, while numerical cost and stability become central concerns. [§1–3, pp. 1–5](https://arxiv.org/pdf/1909.01377v2#page=1)

<strong>Reading the overview figure.</strong> The top row unrolls a shared block into successive layers. The middle row numerically solves `f(z*, x) = z*`. The bottom row solves a linear system at that endpoint for gradients. Arrows indicate computation order within each row. Iterative work remains; the key is avoiding backpropagation through the entire forward solver trajectory. This is an original illustration made for this review. [Enlarge](/assets/reviews/deep-equilibrium-models/deq-core.svg)

## Two costs of increasing depth

An ordinary deep network has different parameters at different layers. Sharing weights across depth prevents parameter count from growing with the number of repetitions, but does not remove the need to store intermediate activations for backpropagation. DEQ starts from a structure with both <strong>weight sharing</strong> and <strong>input injection at every update</strong>. With input `x` fixed, it repeatedly updates hidden state `z`. [§2–3, Equations 3–4](https://arxiv.org/pdf/1909.01377v2#page=3)

```text
z⁰ = 0
zⁱ⁺¹ = fθ(zⁱ; x)       # shared θ, the same input x injected each time
```

Here `i` indexes <strong>computational depth</strong>, not the order of tokens in a sentence. Both `x` and `z` can represent an entire sequence. For autoregressive language modeling, causal convolution or an attention mask prevents output at position `t` from depending on future inputs. Computing sequence states together does not permit access to future target tokens. [§2.2, §4](https://arxiv.org/pdf/1909.01377v2#page=3)

Instead of asking how many layers should define the output, DEQ asks whether a state unchanged by another application of the block can define it.

## Defining an output by a fixed point

A fixed point, or equilibrium `z*`, satisfies:

```text
z* = fθ(z*; x)
gθ(z; x) = fθ(z; x) − z
Therefore gθ(z*; x) = 0
```

Repeatedly applying `f` and solving the equation `g=0` can target the same answer, but their computation and convergence properties differ. DEQ adopts the root-finding view. [§3.1, Equations 5–7](https://arxiv.org/pdf/1909.01377v2#page=4)

Consider the scalar function `f(z; x)=0.5z+0.75x` with `x=2`. This is an explanatory example for this review, not an experiment from the paper.

```text
0 → 1.5 → 2.25 → 2.625 → ... → 3
z* = 0.5z* + 1.5  ⇒  z* = 3
```

A finite number of iterations gives a value close to `3`; the equilibrium equation defines the output as `3`. Real DEQs do not generally have such a closed-form answer. They use numerical solvers with tolerances and iteration limits. “Infinite depth” describes the output definition. It does not mean executing infinitely many operations or always obtaining an exact answer at finite cost.

## Forward computation: finding an equilibrium with Broyden's method

Newton's method updates a candidate state using the residual `g(z)` and its Jacobian. A Jacobian records how each output changes when each state component changes. Building and inverting that matrix is expensive for large sequences. The paper uses <strong>Broyden's method</strong> to update an approximation of the inverse Jacobian. [§3.1.1, Equations 6–7 and 10, p. 4](https://arxiv.org/pdf/1909.01377v2#page=4)

```text
z_next = z − α H gθ(z; x)
H ≈ (∂gθ/∂z)⁻¹
```

`α` is a step size. I denote the inverse approximation by `H` here. It is maintained through low-rank updates instead of constructing a new large inverse each time; the paper initializes the approximation at `−I`. Computation stops when the residual becomes small enough or the iteration limit is reached. Solver settings therefore form part of the model's execution procedure.

Crucially, <strong>a root can exist even when naive iteration is unstable</strong>. The function `f(z)=1.2z+1` has a fixed point at `−5`, but iteration from zero produces `1, 2.2, 3.64, …`. Existence of a root, convergence of repeated application, and successful root finding are distinct questions. This paper does not provide a guarantee that a DEQ solver succeeds for every input. [§3.2, Appendix D, pp. 5, 14–15](https://arxiv.org/pdf/1909.01377v2#page=14)

## Backward computation: why the whole search path need not be saved

After obtaining an endpoint, training requires its sensitivity to parameter changes. <strong>Implicit differentiation</strong> supplies that sensitivity. Differentiate both sides of `z*=fθ(z*; x)` with respect to `θ`. [Theorem 1, Equation 8; Appendix A, Equations 13–14](https://arxiv.org/pdf/1909.01377v2#page=13)

```text
J = ∂fθ/∂z      # state derivative at the endpoint
B = ∂fθ/∂θ      # parameter derivative with the state held fixed

dz*/dθ = J(dz*/dθ) + B
(I − J)(dz*/dθ) = B
dz*/dθ = (I − J)⁻¹ B
```

Let `ℓ` be the loss and `q=∇zℓ` its endpoint gradient. Using column-vector notation, solve the following linear system instead of explicitly constructing the inverse:

```text
(I − J)ᵀ v = q
∇θℓ = Bᵀv
```

This is the core proof: local derivatives at the endpoint determine its parameter sensitivity. Computation uses <strong>vector–Jacobian products (VJPs)</strong>, which differentiate against a vector without building the full Jacobian. The entire series of forward solver updates need not remain in an automatic-differentiation graph. Direct parameter dependencies through the output head or loss must be differentiated separately and added. [§3.1.2, Equation 11, p. 5](https://arxiv.org/pdf/1909.01377v2#page=5)

For the scalar example, choose target `1` and loss `½(z*−1)²`. Then `q=2`, `J=0.5`, and `v=4`. Writing `f(z;x)=az+bx` with `a=0.5`, `b=0.75` gives `∂ℓ/∂a=12` and `∂ℓ/∂b=8`. I checked this example against central finite differences; the maximum absolute error was approximately `3.3×10⁻¹⁰`. This validates a small explanatory calculation, not training of the paper's models.

The derivation has conditions. The function must be differentiable around the selected solution, and <strong><code>I−J</code> must be invertible</strong>. The paper's inverse notation presupposes this requirement. A nearly singular system can amplify state or gradient errors despite a small residual. Moreover, the formula describes an exact equilibrium, while implementation uses an approximate state and approximate linear solve. A small residual alone does not guarantee an accurate gradient. [Appendix A; RBP §3.2](https://proceedings.mlr.press/v80/liao18c/liao18c.pdf#page=3)

## What “constant memory” covers

DEQ's memory claim concerns <strong>activation storage as depth increases</strong>. Instead of retaining the full trajectory, it retains the endpoint, input, a local computation graph, and related quantities. Sequence length, hidden width, batch size, parameters, optimizer states, attention matrices, and solver workspace still consume memory. Storage for the Broyden approximation's history depends on implementation and configuration. [§3.2, p. 5](https://arxiv.org/pdf/1909.01377v2#page=5)

Depth-independent storage does not imply fixed total GPU usage for every model. Long inputs and wide attention blocks retain their costs. The empirical savings below also require the paper's specific measurement conditions, including exclusion of word embeddings.

## The actual blocks: TrellisNet and Transformer

DEQ does not introduce a single new attention operation. It applies an equilibrium definition and implicit training to two sequence blocks. [§4, Figure 1, p. 6](https://arxiv.org/pdf/1909.01377v2#page=6)

| Block | State update | Input and context handling |
| --- | --- | --- |
| DEQ-TrellisNet | Causal 1D convolution and LSTM-style gating | Input projection injected at every shared convolution update; earlier segment states and padding supply context |
| DEQ-Transformer | Self-attention, feed-forward transformations, residuals and layer normalization | Input injected into Q/K/V computation; causal masking, relative positions and earlier segment equilibria |

The Transformer implementation uses Transformer-XL's context handling and relative positions. Its depth update must remain the same function, so it is not identical to every recurrent Transformer, including those with depth-dependent time embeddings. Layer normalization and gating help empirical stability; their presence alone does not prove that the full mapping is contractive.

## What the two representation proofs establish

<strong>Theorem 2: two stacked DEQs can be represented by one larger DEQ.</strong> Let the first equilibrium be `u` and the second `w`. Place both updates in a combined state `[u; w]`. [Appendix B, Equations 15–16, p. 13](https://arxiv.org/pdf/1909.01377v2#page=13)

```text
u = f(u; x)
w = h(w; u)
Γ([u; w]; x) = [f(u; x); h(w; u)]
```

At a fixed point of `Γ`, both original equations hold. Reading the final part of the state reproduces the stacked output. However, the combined hidden dimension is the <strong>sum</strong> of the two state dimensions. This does not prove equal expressivity for a single block at unchanged width or faster solution of the combined system. I follow the construction in Appendix B, whose function subscripts are clearer than those in the main-text equation.

<strong>Theorem 3: a finite untied network can be embedded in a wider tied network.</strong> Put each layer's state in a separate part of a larger vector and construct a block-shift update that passes one part's result to the next. Repetition computes the original layer outputs in sequence. [Appendix C, p. 14](https://arxiv.org/pdf/1909.01377v2#page=14)

Again, this is a width-expanding construction. It does not establish that arbitrary networks retain their expressivity under weight sharing at unchanged width or that every cost decreases. These theorems explain representation possibilities; practical efficiency requires separate evidence from experiments and solver behavior.

## DEQ in relation to earlier work

I read five primary antecedents for their distinct roles. The comparison below uses the indicated sections and DEQ §2–3.

| Research | Why read it alongside DEQ? | Difference from DEQ |
| --- | --- | --- |
| [Trellis Networks for Sequence Modeling](https://arxiv.org/pdf/1810.06682v2#page=3), §3–4 | Input injection, depth-wise weight sharing, and a bridge between convolution and recurrent networks | Unrolls finite depth; DEQ defines the output through the block's equilibrium. |
| [Universal Transformers](https://arxiv.org/pdf/1807.03819v3#page=2), §2 | Repeats shared attention and transition operations with adjustable computation | Uses finite repetitions or position-wise halting, and depth-time representations. DEQ solves a stationary mapping with a residual-based stopping rule. |
| [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366v4#page=1), §1–2 | Solver-defined computation and memory savings through an adjoint | Solves a trajectory from an initial state to a finite time; DEQ solves a fixed-point equation and a local linear system. |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/pdf/1604.06174v2#page=4), §4 | Checkpointing trades recomputation for activation storage | Preserves the original finite network's computation; DEQ changes the output definition to an equilibrium. |
| [Reviving and Improving Recurrent Back-Propagation](https://proceedings.mlr.press/v80/liao18c/liao18c.pdf#page=3), §3–4 | Equilibrium gradients, invertibility requirements, and Neumann/CG computation | Implicit differentiation has this and an older RBP lineage. DEQ turns it into a replacement for modern deep sequence stacks and evaluates it at language-model scale. |

Describing DEQ as the first invention of fixed-point differentiation would miss these antecedents. My interpretation is that its contribution combines <strong>direct learning of a deep shared model's endpoint, practical root finding with implicit backward computation, and evaluation at language-model scale</strong>.

## Experiments: compare quality and memory together

The experiments cover a copy task, Penn Treebank (PTB), and WikiText-103 (WT103). <strong>Perplexity (PPL)</strong> exponentiates average negative log probability of target tokens; lower is better. Comparisons require compatible data, tokenization, vocabulary and evaluation conditions. A percentage change in PPL is not a percentage change in accuracy, and PTB and WT103 values should not be compared directly. [§5, Tables 1–3; Appendix F](https://arxiv.org/pdf/1909.01377v2#page=7)

<strong>The copy task is a small stress test of retaining information across a delay.</strong> The model reproduces ten initial symbols after a waiting interval. For `T=400`, the total sequence length is `T+20=420`. Models have approximately 14–16K parameters. Reported losses are `3.5×10⁻⁶` for DEQ-Transformer, `2.7×10⁻⁵` for TCN, `0.0501` for LSTM, and `0.0491` for GRU. Table 1 and the appendix do not explicitly define this loss's exact aggregation, so I retain the paper's label “loss.” This does not establish general language understanding or reasoning ability. [Table 1, p. 7; Appendix F, p. 16](https://arxiv.org/pdf/1909.01377v2#page=16)

The following table transcribes selected language-model rows. <strong>All memory values use sequence length 150, batch size 15, and exclude word embeddings.</strong> This common measurement setup does not mean every training setting is identical. Auxiliary losses, parameter totals and architectural details differ. [Table 2 note; Table 3, pp. 7–8](https://arxiv.org/pdf/1909.01377v2#page=7)

| Data | Model | Total / non-embedding parameters | Test PPL ↓ | Memory GB ↓ |
| --- | --- | --- | --- | --- |
| PTB | 60-layer TrellisNet, auxiliary loss, without MoS | 24M / 20M | 57.0 | 8.5 |
| PTB | DEQ-TrellisNet | 24M / 20M | 57.1 | 1.2 |
| WT103 | 70-layer TrellisNet, auxiliary loss | 180M / 45M | 29.2 | 24.7 |
| WT103 | Same TrellisNet with checkpointing | 180M / 45M | 29.2 | 5.2 |
| WT103 | DEQ-TrellisNet | 180M / 45M | 29.0 | 3.3 |
| WT103 | 18-layer medium Transformer-XL, adaptive embedding | 110M / 72M | 23.6 | 9.0 |
| WT103 | Medium DEQ-Transformer, adaptive embedding | 110M / 70M | 23.2 | 3.7 |

On PTB, PPL is nearly unchanged while memory decreases substantially. WT103 Trellis memory falls from `24.7` to `3.3GB`, approximately <strong>86.6%</strong>. Against the checkpointed `5.2GB` baseline, the reduction is approximately <strong>36.5%</strong>. In the adaptive-embedding Transformer rows, `9.0→3.7GB` corresponds to approximately <strong>58.9%</strong>. These calculations use rounded table values. The abstract's “up to 88%” is the authors' aggregate claim, not a percentage that applies to every block and baseline.

DEQ does not beat every row on quality. PTB Table 2 includes DARTS at PPL `55.7`; WT103 Table 3 includes a much larger Transformer-XL at `18.7`. These rows differ in scale and training conditions. The persuasive evidence is therefore less about the absolute best PPL and more about <strong>reducing depth-wise activation storage at similar model scale and quality</strong>. [Tables 2–3](https://arxiv.org/pdf/1909.01377v2#page=8)

Runtime comes at a cost. Table 4 reports `DEQ time / baseline time`, so values above one indicate slower execution. [Table 4, p. 9](https://arxiv.org/pdf/1909.01377v2#page=9)

| WT103 baseline | Training time ratio | Inference time ratio |
| --- | --- | --- |
| 18-layer Transformer | 2.82× | 1.76× |
| 70-layer TrellisNet | 2.40× | 1.64× |

In these experiments, DEQ saves memory but runs more slowly. These are not universal latency ratios across hardware and implementations. Iteration count, the cost of each block evaluation, batching and solver implementation all matter.

## Original Figure 2: how computation changes

<figure class="review-figure" id="paper-figure-2">
<a href="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" width="1732" height="509" alt="Original Figure 2: Broyden iterations per time step across training epochs on the left; log equilibrium residual versus function evaluations on the right." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 2 · Equilibrium solving can become harder during training</span>The two panels were isolated from surrounding prose on v2 p. 8. Axes, legends and curves are preserved. <span class="figure-links"><a href="https://arxiv.org/pdf/1909.01377v2#page=8">Shaojie Bai, J. Zico Kolter, Vladlen Koltun, 2019, Figure 2</a> · <a href="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" target="_blank" rel="noopener noreferrer">Enlarge</a> · Figure quotation for critical review; rights remain with the original authors</span></figcaption>
</figure>

<strong>The left panel describes computational work.</strong> Its horizontal axis is training epoch; the vertical axis divides Broyden iterations by the number of sequence time steps. A value of `0.9` does not mean the solver performed less than one iteration: this plot normalizes by sequence length 150. Blue is forward computation, and red is backward computation. Both increase as training progresses in this setting. [Figure 2, §5, p. 8](https://arxiv.org/pdf/1909.01377v2#page=8)

<strong>The right panel describes a residual.</strong> The horizontal axis counts function evaluations; the vertical axis plots `‖f(z)−z‖` logarithmically. It is neither PPL nor distance to a known correct state. Naive iteration of the weight-tied Transformer decreases the residual at epoch 1 but oscillates at a large residual at epoch 12. DEQ root finding reaches much smaller residuals at both epochs. This supports a distinction between naive iteration and numerical root finding, not a claim of faster wall-clock execution than a finite-depth baseline.

## Original Figure 3: choosing precision and quality

<figure class="review-figure" id="paper-figure-3">
<a href="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" width="1737" height="469" alt="Original Figure 3: forward residual tolerance versus validation PPL on the left, and forward iteration limit versus validation PPL on the right." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 3 · Stopping too early reduces quality</span>The panels were isolated from v2 p. 9, preserving data and original labels. <span class="figure-links"><a href="https://arxiv.org/pdf/1909.01377v2#page=9">Shaojie Bai, J. Zico Kolter, Vladlen Koltun, 2019, Figure 3</a> · <a href="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" target="_blank" rel="noopener noreferrer">Enlarge</a> · Figure quotation for critical review; rights remain with the original authors</span></figcaption>
</figure>

The left panel shows where relaxing the forward residual tolerance damages quality. Its horizontal axis is logarithmic; smaller values are stricter. Validation PPL stays similar across small tolerances, then worsens sharply when tolerance becomes too loose. The right panel shows improving PPL as the iteration limit grows, with diminishing gains from additional computation. The original includes shaded regions but does not specify their aggregation, so I do not interpret them as confidence intervals. [Figure 3, §5, p. 9](https://arxiv.org/pdf/1909.01377v2#page=9)

These experiments use the medium DEQ-Transformer without adaptive embeddings. The accompanying text describes competitive results with `ε<0.1` or an iteration limit of 30 <strong>for sequences of length 75</strong>. This should not be generalized to the length-150 memory benchmark or treated as an optimal setting for every model. The figure reports validation PPL, distinct from the test PPL in the earlier table.

## Practical conditions and remaining limits

The appendices explain implementation details that are easy to miss in the mathematical formulation. [Appendix D–F, pp. 14–16](https://arxiv.org/pdf/1909.01377v2#page=14)

- <strong>Share dropout masks across solver iterations.</strong> Randomly changing the mapping at every iteration disrupts the premise of solving one fixed equation.
- <strong>Stabilization matters.</strong> Small initial weights, gating, normalization and optional shallow-model warm-up help. Successful examples do not constitute a global convergence guarantee throughout training.
- <strong>Sequences in a batch need different iteration counts.</strong> Waiting for the slowest sequence affects utilization, so function evaluations alone do not measure GPU efficiency.
- <strong>Very long sequences can be harder.</strong> The appendix discusses difficult initial solves beyond length 1,000 and segmenting sequences. Removing depth-wise storage does not remove context-length costs.
- <strong>Intermediate-layer auxiliary losses do not transfer directly.</strong> DEQ attaches the loss to an endpoint, so its training setup is not identical to finite-depth deep supervision.

This review checked reported numbers, the logic of the equations and constructive proofs, and a scalar explanatory calculation. Large-model retraining, GPU peak-memory and runtime measurement, multi-seed uncertainty evaluation, and formal verification of the complete implementation were <strong>not run</strong>. I did not infer unreported error bars or convergence rates for every input.

## What to retain from the paper

DEQ changes depth from a list of layers to retain into a state equation to solve. Its output is an equilibrium; its learning signal is that equilibrium's sensitivity. This can substantially reduce activation storage across depth, while solver convergence and numerical gradient stability become central problems.

My main reading question is therefore <strong>which mappings admit usable equilibria, and at what solve precision does the quality–memory–time tradeoff become favorable?</strong> Subsequent work includes [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656), which extends the approach to multiresolution vision, and [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342), which addresses stability and efficiency. I checked only official abstracts and metadata for these two follow-ups, leaving detailed result comparisons for a later reading.

## Sources and reading scope

- Main paper: [*Deep Equilibrium Models*, arXiv:1909.01377v2](https://arxiv.org/abs/1909.01377v2), Shaojie Bai, J. Zico Kolter, Vladlen Koltun. First public submission: 2019-09-03; reviewed v2 revision: 2019-10-28. [Official NeurIPS 2019 page](https://proceedings.neurips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html).
- On 2026-10-09, the first pass mapped the overall argument, problem and concept; the second examined §1–6, tables, figures, references and Appendices A–F across the 16-page paper. Key equation, table and proof pages were also checked against rendered PDF pages.
- Related-paper reading scope: TrellisNet pp. 2–4; Universal Transformers v3 pp. 2–4; Neural ODE v4 pp. 1–3; sublinear-memory v2 pp. 1 and 3–4; ICML 2018 RBP pp. 2–4. Transformer-XL's implementation role was described from DEQ §4 and official metadata. The two later papers were read at abstract/metadata scope only.
- The [authors' code repository](https://github.com/locuslab/deq) was inspected but not executed. Its current contents include later work and implementation changes; I did not treat it as an unchanged environment for reproducing the 2019 experiments.
- The overview diagram is original to this review ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Figures 2 and 3 are limited quotations from the paper for criticism and explanation, with authors' rights retained. The paper's [arXiv distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html) is not described as a general reuse license.
