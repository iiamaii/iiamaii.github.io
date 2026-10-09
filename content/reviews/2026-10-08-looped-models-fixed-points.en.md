---
title: "Can an endpoint replace the recurrent path? Efficiency at fixed points in looped models"
description: "A full-paper review of TBPTT, terminal KV sharing, RL state reuse, and distilled prefill, including the assumptions, appendix controls, quality losses, and end-to-end costs."
date: "2026-10-08"
updatedAt: "2026-10-09T05:04:32Z"
publishedAt: "2026-10-08T16:55:04+09:00"
topics: ["language models", "looped models", "fixed points", "efficient inference", "reinforcement learning"]
translationKey: "looped-models-fixed-points"
paperTitle: "Towards Looped Models Done Right, Part II: Rethinking at Fixed Points"
paperPublishedDate: "2026-10-05"
authors: ["Benhao Huang", "Chufan Shi", "Junlin Chen", "Shicheng Wen", "Zhengzhong Liu", "Eric Xing", "Xuezhe Ma"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.06833v1"
visibility: "public"
lang: "en"
thumbnail: "/assets/reviews/looped-models-fixed-points/paper-figure-1.svg"
thumbnailAlt: "Paper Figure 1 connecting the depth prior and orthogonal input injection to fixed-point endpoints and savings in training activations, KV cache, prefill, and RL scoring plus backward."
---

If a looped language model repeatedly applies the same computation until its state barely changes, must it keep and replay **the entire path to that endpoint**? This paper treats fixed points as states that can replace parts of a computational trajectory. It connects this view to backpropagation, inference KV caches, reinforcement-learning replay, and prompt processing. Cache savings, faster updates, and faster prefill are separate experiments, with separate costs in quality and total runtime. [§1–3, §5, PDF pp. 1–13](https://arxiv.org/pdf/2610.06833v1#page=1)

**Figure 1 · How to read the representative figure.** The left panel presents two training improvements, the learned depth prior and OrthoInj. The center contrasts the recurrent path with an endpoint near a fixed point. The right panel summarizes four computational shortcuts. Its bars report separate experiments, not cumulative speedups from one combined system. In particular, the RL 2× refers to scoring and backward, not total training time. [Benhao Huang et al., 2026, Figure 1](https://arxiv.org/html/2610.06833v1#S0.F1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · Original retained · [View full size](/assets/reviews/looped-models-fixed-points/paper-figure-1.svg).

## The question

A conventional Transformer passes activations through layers with distinct weights. A **looped language model** repeatedly runs a shared block, increasing computation per token without adding weights. However, retaining every recurrence's intermediate states makes training memory and generation caches grow with recurrence depth. Processing the initial prompt, called **prefill**, also becomes expensive. **Decoding** then generates subsequent tokens one at a time. [§1–2](https://arxiv.org/html/2610.06833v1#S2)

The authors connect four questions. Why can training backpropagate through only the last few recurrences? Can decoding reuse only previous tokens' final KV? Must RL recompute states already produced during generation? Can a smaller model predict a prompt's endpoint? They also modify the training-depth distribution and input injection to encourage states that support these shortcuts. [§3–4](https://arxiv.org/html/2610.06833v1#S3)

## Key idea: replacing a path with its endpoint

The experimental backbone is Huginn. A prelude embeds the input; a recurrent core containing two Transformer blocks repeatedly receives that input representation; a coda converts the endpoint into next-token probabilities. The complete recurrent state `z` contains both hidden states `H` and attention key/value banks `C`. [§2](https://arxiv.org/html/2610.06833v1#S2)

```text
e = Pθ(x)                         # input representation from the prelude
zʳ = (Hʳ, Cʳ)
zʳ⁺¹ = Fθ(zʳ; x)                  # shared recurrent core
z* = Fθ(z*; x)                    # fixed point
```

At a mathematical fixed point, one more application leaves the state unchanged. The paper also uses the term for approximately stationary finite-depth states. These meanings must remain distinct: small state changes neither imply correct predictions nor establish convergence at the same rate for every input. [§2, Appendix A.2](https://arxiv.org/pdf/2610.06833v1#page=22)

| Application | What is removed | What remains or is required |
| --- | --- | --- |
| Pretraining TBPTT | Storage and backward computation for earlier recurrences | Forward computation to the sampled depth |
| Terminal KV sharing | Previous tokens' visit-specific KV banks | Current-token recurrence and visit-local current-token KV |
| RL rollout-state reuse | Recomputing sampled tokens' endpoints | One core forward graph, several backward VJPs, and rollout generation |
| Distilled prefill | Multiple teacher recurrences over the prompt | Student, one teacher recurrence and coda, then teacher decoding |

A VJP, or vector–Jacobian product, applies a backward derivative without constructing the full Jacobian. The paper does not measure a cumulative speedup from combining all four techniques in one deployed system. In particular, RL rollout and evaluation retain visit-specific KV. [§3, Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=40)

<figure class="review-figure" id="paper-figure-3">
<a href="/assets/reviews/looped-models-fixed-points/paper-figure-3.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/looped-models-fixed-points/paper-figure-3.svg" width="508" height="160" alt="Original schematic of suffix backpropagation, terminal-prefix KV sharing, and student prefill with recurrent teacher decoding." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 3 · Three different computational paths</span>(a) Detach below the dashed boundary and backpropagate through the suffix. (b) Reuse previous tokens’ terminal KV across the current token’s recurrences. (c) Predict the prefix endpoint with a non-recurrent student and retain teacher decoding. Read the text for the intervening teacher recurrence and coda used to build the banks. <span class="figure-links"><a href="/assets/reviews/looped-models-fixed-points/paper-figure-3.svg" target="_blank" rel="noopener noreferrer">View full size</a> · <a href="https://arxiv.org/html/2610.06833v1#S3.F3">Benhao Huang et al., 2026, Figure 3</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original retained; no edits or cropping</span></figcaption>
</figure>

## Tokens settle at different depths and in different orders

Figure 2 plots relative changes between adjacent recurrent hidden states, with recurrence on the vertical axis and token position horizontally. A later token can settle before an earlier one, undermining a simple left-to-right convergence story. Across qualifying tokens from eight observed sequences, the Spearman correlation between position and convergence depth is 0.14. This is a measurement on those sequences, not a universal law about language. [Figure 2, Appendix C.4](https://arxiv.org/pdf/2610.06833v1#page=35)

```text
ρ(r,t) = ||hₜʳ − hₜʳ⁻¹||₂ / (||hₜʳ⁻¹||₂ + ε)
```

The figure identifies the first depth after which changes remain below 2% through the observation horizon, requiring at least four stable updates. With a horizon of 64, Appendix Table 19 reports that 99.74% of tokens in the Small PLN-5 model meet a 1% threshold. This describes **state stability** on a finite observation window and specified data. Under much stricter convergence checks, some inputs fail and others require hundreds of recurrences. [Appendix C.4, Table 19; A.1](https://arxiv.org/pdf/2610.06833v1#page=36)

<figure class="review-figure" id="paper-figure-2">
<a href="/assets/reviews/looped-models-fixed-points/paper-figure-2.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/looped-models-fixed-points/paper-figure-2.png" width="836" height="232" alt="Token-position versus recurrence hidden-state-change heatmap, local convergence-depth reversal, and position/depth scatterplot." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 2 · Tokens do not settle in causal order</span>Darker areas indicate smaller relative state changes. The zoomed staircase shows neighboring tokens exchanging their convergence-depth order. The weak position/depth correlation is specific to the observed sequences. <span class="figure-links"><a href="/assets/reviews/looped-models-fixed-points/paper-figure-2.png" target="_blank" rel="noopener noreferrer">View full size</a> · <a href="https://arxiv.org/html/2610.06833v1#S2.F2">Benhao Huang et al., 2026, Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original retained; no edits or cropping</span></figcaption>
</figure>

## Why backpropagating through a suffix can work

TBPTT, truncated backpropagation through time, differentiates the last `b` recurrences and detaches earlier states. At an exact fixed point, let `J*` be the core's state Jacobian and `B*` its parameter derivative with state held fixed. With `I` denoting the identity matrix, the following sensitivity connects equilibrium differentiation to a truncated Neumann series. [§3.1, Eq. 1; Appendix A.2, Eqs. 14–17](https://arxiv.org/pdf/2610.06833v1#page=22)

```text
D* = dz*/dθ = (I − J*)⁻¹ B*
   = B* + J*B* + J*²B* + ...       # spectral radius(J*) < 1

D_b = Σ[k=0..b−1] J*ᵏ B*
D* − D_b = J*ᵇ D*
```

If `||J*||₂ ≤ κ < 1`, the omitted loss-gradient tail is bounded by `||B*||₂ ||g||₂ κᵇ/(1−κ)`, where `g` is the endpoint loss gradient. Direct parameter paths through input injection, the prelude, and the output head must remain differentiable. [Appendix A.2, Lemma 2](https://arxiv.org/pdf/2610.06833v1#page=23)

However, **finite-depth and equilibrium objectives differ**. Equal endpoint values need not have equal parameter derivatives. Actual TBPTT uses changing Jacobians along the trajectory; endpoint Neumann estimation repeatedly uses the endpoint Jacobian. A small endpoint residual alone does not guarantee that these estimates agree. [Appendix A.2](https://arxiv.org/pdf/2610.06833v1#page=22)

The experiments also show that excessive truncation hurts quality. For Medium PLN-5, windows `b≥4` keep WikiText PPL within 1% of full BPTT, while `b=2` is approximately 5.5% worse and `b=1` approximately 15% worse. On Small fixed-depth models, short windows improve robustness to extra depth and cache sharing while reducing base quality. An endpoint-Neumann-from-initialization control collapses at all five tested learning rates. A fixed-point interpretation does not make every gradient approximation safe. [Appendix C.3, Tables 17–18](https://arxiv.org/pdf/2610.06833v1#page=34)

Separate memory measurements report peak allocation of 48/86/124GB for windows 5/10/15 at local batch eight. Full BPTT at depth 30 runs out of memory; per-layer activation recomputation lowers memory to 28GiB but takes approximately 1.3 times the step time of window ten. These are configuration-specific measurements, not universal memory-reduction ratios. [§3.1, Appendix C.3](https://arxiv.org/pdf/2610.06833v1#page=33)

## When retaining only terminal KV is justified

The model has four physical attention layers: one prelude, two core blocks, and one coda. At `R=5`, the logical depth is `1+2×5+1=12`. A visit-specific cache stores twelve banks; terminal sharing retains four, one per physical layer. At `R=16/32`, it retains four instead of 34/66. These are **KV-storage savings**, not corresponding reductions in total GPU memory or FLOPs. [§3.2, Table 20](https://arxiv.org/pdf/2610.06833v1#page=36)

Lemma 1 shows that joint iteration and iteration with a converged prefix frozen reach the same limit if the prefix context converges, the current-token update is uniformly contractive on the specified invariant region, and it is Lipschitz in context. For finite prefix error `η`, two error terms remain. [§3.2, Lemma 1; Appendix A.1, Eq. 12](https://arxiv.org/pdf/2610.06833v1#page=20)

```text
current-token error ≤ κᴿ × initial error + β(1−κᴿ)/(1−κ) × η
```

The first term measures unfinished current-token refinement. The second reflects an imperfect frozen context and does not vanish merely by adding recurrences. When `κ` approaches one, context error can be strongly amplified.

Of 304 prefixes under strict empirical checks, 206 qualify for the fixed-depth model and 191 for PLN-5. Their median prefix convergence depths are 152.5 and 114, far beyond the operational depth of five. Freezing at depth five instead gives current-token endpoint gaps five to six orders of magnitude larger than freezing a strictly converged prefix. Practical sharing at `R=5` therefore rests on **finite-error analysis and measured quality**, rather than an assertion that those prefixes have reached exact fixed points. [Appendix A.1, Figure 6](https://arxiv.org/pdf/2610.06833v1#page=21)

Appendix C.6 independently varies prefill and decoding depths over eight values between 1 and 32, giving an 8×8 comparison. Once decoding is sufficiently deep, prefill beyond five brings little gain. With shallow decoding, shallow prefill can work better. More prompt computation is not universally beneficial, and the two depths need not be equal. [Appendix C.6, Figure 11](https://arxiv.org/pdf/2610.06833v1#page=37)

## Learning the distribution of training depths

The Huginn baseline samples from a shifted Poisson-lognormal, or PLN, distribution with mean depth five. The paper initializes a learned categorical distribution over depths 1–64 from that prior. This is **one global training distribution, not an input-dependent stopping policy**. One depth is sampled per microbatch before running the recurrence. [§4.1, Algorithm 2; Appendix B.3](https://arxiv.org/pdf/2610.06833v1#page=25)

```text
J_prior = −E[stop_gradient(A) log pφ(R)]
          − λ_H H(pφ) + λ_m(E[R] − 5)²
```

The advantage `A` comes from `exp(−CE)`, an inverse-perplexity reward, adjusted using moving averages and variance. The language model still trains with cross-entropy, or CE. Entropy `H` discourages collapse to one depth; the final term keeps mean depth near the compute budget. The controller reuses the already-computed model loss instead of adding another model forward pass. Defaults are `λ_H=0.01`, `λ_m=1`, and a maximum backward window of ten. [§4.1, Appendix B.3](https://arxiv.org/pdf/2610.06833v1#page=25)

In Figure 4, the orange fixed-`R=5` model becomes unstable outside its training depth. PLN and learned-depth training remain more stable, with the learned prior lowering PPL relative to PLN. This does not mean every task improves monotonically with more recurrences. [Figure 4; Appendix C.5](https://arxiv.org/pdf/2610.06833v1#page=8)

<figure class="review-figure" id="paper-figure-4">
<a href="/assets/reviews/looped-models-fixed-points/paper-figure-4.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/looped-models-fixed-points/paper-figure-4.svg" width="496" height="168" alt="WikiText PPL across test depths 1–64 for S and M models, comparing fixed depth, PLN, and learned priors." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 4 · Perplexity outside the training depth</span>The horizontal axis is test recurrence depth; lower WikiText PPL is better. Fixed-depth training performs well near five but becomes unstable elsewhere. Sampled and learned priors are more stable. Insets expose small differences near five; this is not a generation-accuracy plot. <span class="figure-links"><a href="/assets/reviews/looped-models-fixed-points/paper-figure-4.svg" target="_blank" rel="noopener noreferrer">View full size</a> · <a href="https://arxiv.org/html/2610.06833v1#S4.F4">Benhao Huang et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original retained; no edits or cropping</span></figcaption>
</figure>

A consequential control appears in the appendix. At Small scale, sampling from the final learned distribution **held fixed from the first update** gives validation PPL 5.63 versus 5.67 for the adapting prior. Finding a better distribution shape may explain the benefit without requiring continual adaptation. Higher entropy reduces the Medium CacheEval sharing penalty, but does not monotonically improve WikiText or task accuracy. Equal mean depth also does not imply identical realized FLOPs. [Appendix C.1, Tables 10, 12](https://arxiv.org/pdf/2610.06833v1#page=29)

## OrthoInj: preserving the injected direction

Parcae-style injection carries the previous state into the core as `Λh+q`, where `q` is the injected input and `Λ` is a learned diagonal decay. The carryover can already contain a component along `q`, changing the effective magnitude of that direction. OrthoInj removes that component before adding `q`. [§4.2, Eqs. 7–8](https://arxiv.org/pdf/2610.06833v1#page=8)

```text
q = Δ ⊙ W e
Q_q = I − qqᵀ / (||q||₂² + ε)
core input = Q_q Λh + q
```

Here `⊙` denotes elementwise multiplication. For nonzero `q` and `ε=0`, the component along `q` is exactly `q`; the implementation uses `ε=10⁻⁶`. The projection does not increase the norm of the decayed carryover, but this is **not a proof that the entire Transformer update is contractive**. The recipe also removes prelude RMSNorm. A 2×2 appendix comparison finds contributions from both changes to validation PPL, with a separately selected learning rate for each configuration. [§4.2; Appendix B.4, C.2, Table 14](https://arxiv.org/pdf/2610.06833v1#page=32)

In Table 16, the observed Parcae carryover amplifies rather than cancels the input direction. Median effective gains are approximately 2.18/2.75/3.43 at S/M/L. The measured intervention is therefore better described as stabilizing the input-direction magnitude than rescuing an input observed to disappear. [Appendix C.2, Table 16](https://arxiv.org/pdf/2610.06833v1#page=33)

## Before reading the results: size and metrics

The paper's S/M/L names round the **non-embedding parameter counts**. Total looped-model counts are larger. [§5.1; Appendix B.1, Table 6](https://arxiv.org/pdf/2610.06833v1#page=24)

| Scale | Looped non-embedding parameters | Total including embeddings | Pretraining tokens |
| --- | --- | --- | --- |
| S, “100M” | 0.106B | 0.304B | 21.47B |
| M, “400M” | 0.425B | 0.819B | 85.90B |
| L, “1.6B” | 1.699B | 2.488B | 343.60B |

All use context length 8,192, global batch 512, BF16, a mixture including web, QA, code, and mathematics, and learning-rate screening per configuration. There is one training seed per configuration. The official implementation README identifies NVIDIA H200 as the experimental hardware. Latencies below apply to specified batches and prompts, not arbitrary hardware. [Appendix B.1](https://arxiv.org/pdf/2610.06833v1#page=24), [official implementation README](https://github.com/ifm-ai/xllm-loop#installation)

PPL exponentiates token-averaged CE and is better when lower. **AVG equally weights accuracy on LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, OpenBookQA, and SciQ**; it excludes GSM8K, DROP, and MBPP+. Some entropy-sweep appendix results replace SciQ with WinoGrande, so their AVG is not interchangeable. Main GSM8K results use all 1,319 test questions, eight-shot prompting, and greedy accuracy. DROP uses a fixed 500-question subset, three-shot prompting, and F1. MBPP+ uses 100 fixed tasks, three-shot prompting, and pass@1 estimated from eight samples at temperature 0.8. [Appendix D.2, D.5, Table 27](https://arxiv.org/pdf/2610.06833v1#page=44)

There is an especially important PPL trap. **Terminal-KV PPL in Tables 2–3 and native PPL in Tables 8/13 score different tokens.** The former scores document continuations after a 4,096-token prefix; the latter scores full sequences. Dividing these numbers does not measure the cache-sharing penalty. Use Table 20/21 or CacheEval comparisons that score identical tokens with both cache layouts. [Appendix D.3–4](https://arxiv.org/pdf/2610.06833v1#page=45)

## Results and evidence: what does the learned prior improve?

The following Table 2 comparisons replace fixed PLN-5 with the learned prior at `λ_H=0.01`. All use test depth five, four terminal KV banks, and Parcae without prelude normalization. They do not combine the learned prior with OrthoInj. [Table 2](https://arxiv.org/pdf/2610.06833v1#page=10)

| Scale | Val. PPL ↓ | WikiText PPL ↓ | Seven-task AVG, % ↑ | GSM8K, % ↑ |
| --- | --- | --- | --- | --- |
| S | 5.73 → 5.67 | 20.19 → 19.91 | 41.32 → 41.65 | 1.52 → 1.59 |
| M | 4.00 → 3.93 | 12.55 → 12.28 | 49.06 → 49.07 | 13.72 → 15.62 |
| L | 3.10 → 3.07 | 8.76 → 8.68 | 59.30 → 60.00 | 47.61 → 47.92 |

PPL improves at all scales, but Medium AVG improves by just 0.01 percentage point (pp). Large training FLOPs increase from `1253` to `1273 ×10¹⁹`, approximately 1.6%. A matched mean-depth budget is not an exact FLOP match.

At Large scale, learned-prior terminal-sharing AVG 60.00 is close to fixed-depth full-cache AVG 59.90. It still trails the distinct-weight twelve-layer Untied 12's 61.47. The roughly threefold reduction relative to Untied 12 concerns non-embedding parameters and KV-bank count. Quality exceeds same-physical-depth Untied 4, but training FLOPs are `1273/465≈2.74` times larger. **Parameter efficiency and compute efficiency are different claims.** [Tables 2, 6, 8](https://arxiv.org/pdf/2610.06833v1#page=28)

In a matched-token comparison for the Large learned prior at `R=5`, terminal sharing raises validation PPL from 3.047 to 3.073, approximately 0.85%, and reduces GSM8K from 50.42% to 47.92%. At `R=16/32`, the PPL gap is at most 0.11%. “Nearly lossless sharing” must be read within those settings and metrics. [Table 20](https://arxiv.org/pdf/2610.06833v1#page=36)

## OrthoInj gains depend on the metric

Table 3 compares injection recipes under fixed PLN-5. Relative to Parcae-Decay, OrthoInj improves validation PPL and AVG, but not every WikiText or generation score. [Table 3](https://arxiv.org/pdf/2610.06833v1#page=11)

| Scale | Val. PPL ↓ | WikiText PPL ↓ | AVG, % ↑ | GSM8K, % ↑ | MBPP+, % ↑ |
| --- | --- | --- | --- | --- | --- |
| S | 5.75 → 5.67 | 19.90 → 20.02 | 41.28 → 41.65 | 1.90 → 1.90 | 0.25 → 1.62 |
| M | 3.96 → 3.94 | 12.45 → 12.32 | 49.03 → 49.16 | 15.01 → 13.95 | 8.25 → 7.88 |
| L | 3.09 → 3.08 | 8.72 → 8.72 | 59.22 → 59.26 | 52.92 → 51.71 | 30.88 → 27.75 |

Large AVG gains only 0.04pp, while code performance declines. The prose claims lowest WikiText PPL at every scale, but Table 3 shows Parcae ahead at Small and a tie at Large. This review follows the table. Another Medium learning-rate comparison in Appendix Table 15 likewise shows that lower PPL need not yield higher task AVG. [§5.3; Appendix C.2, Table 15](https://arxiv.org/pdf/2610.06833v1#page=33)

## RL: avoiding recomputation of already-generated states

RL generates responses, then scores the same tokens to update the policy. The authors save rollout endpoint `zᴿ` and, before any parameter update, apply the core once there to construct a differentiation graph. The forward value remains the saved state, while repeated endpoint-Jacobian VJPs approximate the gradient. **Neumann-4 means four state VJPs through the same graph plus the identity term: five series terms, not four forward passes.** [Algorithm 1; Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=40)

The comparison uses a Large learned-prior checkpoint, depth six, Dr. GRPO, and sixteen responses per training question. GSM8K training covers 400 questions; evaluation uses 500 fixed test questions and eight samples each. Here pass@1 averages the eight outcomes, unlike the main greedy accuracy; pass@8 counts questions with at least one successful response. [§5.4; Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=41)

| Method | GSM8K pass@1 / @8, % | MATH500 pass@1, % | Update, seconds ↓ | GSM total training, minutes ↓ |
| --- | --- | --- | --- | --- |
| Before RL | 40.63 / 77.60 | 13.40 | — | — |
| Full BPTT | 63.20 / 88.00 | 18.27 | 2.77 | 65.45 |
| Recompute + Neumann-4 | 60.05 / 89.60 | 17.32 | 2.09 | 61.71 |
| Reuse + Neumann-4 | 61.65 / 90.20 | 17.82 | 1.39 | 57.30 |
| Reuse + Neumann-3 | 61.30 / 89.80 | 18.77 | 1.30 | 58.98 |

Update time is the **median scoring-plus-backward time, excluding rollout and the optimizer step**. The 2.77→1.39-second change is approximately 1.99 times faster. Total training changes from 65.45 to 57.30 minutes, approximately 1.14 times faster, with different response lengths and realized update counts. The total-time difference is not an isolated causal effect of the estimator. [Tables 4, 25](https://arxiv.org/pdf/2610.06833v1#page=42)

For MBPP+ code RL, updates improve from 1.03 to 0.51 seconds, but total time **increases from 20.0 to 21.8 minutes** because rollout and code verification dominate. Pass@1 changes from 40.88 to 39.38 and pass@8 from 69 to 64. [Table 26](https://arxiv.org/pdf/2610.06833v1#page=43)

Reuse minus full-BPTT GSM8K pass@1 is −1.55pp, with a paired question-bootstrap 95% interval of `[−3.13, 0.05]`. This conditions on one training seed; it establishes neither equivalence nor stability across seeds. Reuse/recompute gradient cosine is 0.997–0.999, versus 0.76–0.87 for reuse/full BPTT. The mean relative endpoint residual is 0.034, not zero. Reusing stale states across optimizer updates is untested. [Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=42)

## Distilled prefill: predicting only the prompt endpoint

A non-recurrent two-block student predicts the teacher's pre-coda endpoint. Initialized from the teacher core, it trains with normalized hidden-state error and KL, or Kullback–Leibler, divergence to the teacher output distribution. The teacher is frozen, and the student consumes an additional quarter of the teacher's pretraining token budget. **One teacher recurrence and the coda turn the student prediction into terminal KV; subsequent decoding still runs the teacher at depth five.** This does not replace the entire language model with the student. [§3.2, Eq. 4; §5.5, Appendix C.7](https://arxiv.org/pdf/2610.06833v1#page=38)

| Scale | Teacher → distilled prefill, ms ↓ | Prefill speedup | AVG, % ↑ | GSM8K, % ↑ | MBPP+, % ↑ |
| --- | --- | --- | --- | --- | --- |
| S | 195.0 → 125.9 | 1.55× | 41.65 → 40.63 | 1.59 → 1.74 | 0.62 → 0.62 |
| M | 495.8 → 299.3 | 1.66× | 49.07 → 46.77 | 15.62 → 8.95 | 10.38 → 6.12 |
| L | 1482.1 → 830.3 | 1.79× | 60.00 → 55.53 | 47.92 → 32.90 | 27.25 → 16.50 |

These are means over six warmed prefill repetitions, with 8K prompts and batch eight. The student path is approximately 1–4% faster than stopping teacher prefill after two recurrences and gains 0.5–0.9pp AVG. Against full teacher prefill, however, Large AVG loses 4.47pp and GSM8K loses 15.02pp. These measurements do not establish the same speedup for time to first token, total generation, or low-batch interactive serving. [Table 5, Appendix C.7.1](https://arxiv.org/pdf/2610.06833v1#page=13)

Further controls expose handoff sensitivity. Feeding the student's own KV directly to the teacher costs 2–7pp AVG relative to the one-teacher-recurrence path. Standalone students do not beat Untied 4. Distilling a deeper `R=25` teacher gives 4.3–5.4× prefill speedups but larger quality losses; its student is slower and less accurate than the `R=5` student. This comparison uses **hidden loss only**, unlike the main hidden-plus-KL recipe. Lower hidden MSE can coexist with worse downstream quality, suggesting that a single state-distance metric does not capture decoder sensitivity. [Appendix C.7, Tables 22–23](https://arxiv.org/pdf/2610.06833v1#page=39)

## Position among related work

The paper does not introduce fixed points or looped Transformers. It connects existing components through an efficiency argument and evaluates two endpoint-reuse paths and training recipes. Official abstracts and bibliographic metadata were checked for the following background papers; detailed comparisons rely on the focal paper's §6 and Appendix E. This is not a full rereading of all related papers.

| Background work | Relevant question | This paper's distinction |
| --- | --- | --- |
| [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377v2), 2019 | Solve equilibria directly and differentiate implicitly | Connects TBPTT, KV sharing, and RL reuse in finite-loop models |
| [Huginn](https://arxiv.org/abs/2502.05171v2), 2025 | Scale latent test-time compute with a shared recurrent block | Uses it as the backbone and analyzes endpoint reuse |
| [Parcae](https://arxiv.org/abs/2604.12946v1), 2026 | Stable input injection and loop scaling | Evaluates OrthoInj's projection of the carryover's input-direction component |
| [RL-Halting](https://arxiv.org/abs/2606.29983v1), 2026 | Learned stochastic stopping for stable length extrapolation | Learns an input-independent training-depth prior with a mean budget and deep supervision |

Appendix E discusses how KV-sharing results for Ouro, MELT, and continuous depth batching depend on the model and evaluation protocol. These results do not justify applying one shared cache to every looped model. Unlike earlier distillation that retains a recurrent student, this work uses a non-recurrent student only for prefill. Additional primary-source verification of those detailed comparisons remains open. [§6, Appendix E.1–5](https://arxiv.org/pdf/2610.06833v1#page=46)

## My reflections: validate the endpoint and the shortcut separately

The paper's strength is distinguishing problems easily hidden inside the phrase “reduce recurrence”: the backward path, previous-token context, RL recomputation on identical tokens, and approximation of a prompt endpoint. Its appendix provides controls separating exact limits from depth five, similar states from similar gradients, and component speedups from total runtime. This is my interpretation of its contribution.

The most useful design principle is to **define what replaces the omitted path and which error remains**. Terminal KV leaves context error; RL reuse leaves gradient approximation and state-storage cost; distilled prefill leaves handoff error and extra training cost. Lower PPL alone does not guarantee generation quality, optimization stability, or serving speed.

The evidence remains limited. Each configuration has one training seed. Larger scales, other architectures, long reasoning completions, and production serving are untested; pretraining decontamination is unverified. These tables cannot establish additional gains from combining the learned prior with OrthoInj, or cumulative gains from adding terminal sharing to the RL experiments. [§7; Appendix C.8, D, E.3](https://arxiv.org/pdf/2610.06833v1#page=14)

## A map of the whole paper and open questions

The appendix answers questions that the main results alone leave unresolved. This review covers every section's argument while selecting consequential comparisons and counterexamples instead of reproducing all 27 tables or the bibliography.

| Source range | What to check |
| --- | --- |
| §1–2, pp. 1–3 | Why fixed points, and how do token trajectories differ? |
| §3, pp. 4–6 | How do TBPTT, RL reuse, KV sharing, and prefill paths differ? |
| §4, pp. 6–8 | How do depth priors and injection shape the state? |
| §5, pp. 9–13 | Which quality/cost trade-offs appear at three scales? |
| §6–7, pp. 13–14 | What is the relationship to prior work and the stated scope? |
| A.1–2, pp. 20–23 | What assumptions support cache sharing and gradient approximation? |
| B.1–4, pp. 23–27 | What are the model, LR, TBPTT, controller, and projection recipes? |
| C.1–2, pp. 27–33 | What do fixed-shape, entropy, normalization, projection, and LR controls show? |
| C.3–6, pp. 33–37 | How do windows, stability, deeper caches, and separate prefill/decode budgets behave? |
| C.7–8, pp. 38–43 | Where do distillation and end-to-end RL savings fail? |
| D.1–5, pp. 43–46 | Do identically named metrics score the same data and tokens? |
| E.1–5, pp. 46–47 | How do learned exits, KV sharing, and prior distillation differ? |

A useful next experiment would vary `R`, the TBPTT window, and prefix residual on the same checkpoint and tokens while measuring quality, memory, and total runtime. The question is how well state distance or PPL predicts generation success. Testing whether online prior adaptation itself matters requires repeating frozen-final-prior comparisons at Medium/Large scale with multiple seeds. Neither proposal was executed for this review.

## Sources and reading scope

- Focal paper: Benhao Huang, Chufan Shi, Junlin Chen, Shicheng Wen, Zhengzhong Liu, Eric Xing, and Xuezhe Ma, **Towards Looped Models Done Right, Part II: Rethinking at Fixed Points** (2026), [arXiv:2610.06833v1](https://arxiv.org/abs/2610.06833v1), submitted October 5, 2026; reviewed October 8, 2026.
- Reading scope: all explanatory text, equations, algorithms, experimental protocols, and tables in main §1–7 and Appendices A–E of the 47-page PDF, cross-checked with the authors' submitted TeX. Core figures and cited table/equation pages were visually inspected; this does not claim visual QA of every page. Appendices missing after C.1 in the [HTML](https://arxiv.org/html/2610.06833v1) were recovered from the PDF and submitted source.
- Additional sources: official abstracts/metadata of the four background papers above and the [official implementation README](https://github.com/ifm-ai/xllm-loop). Code and checkpoints were not executed. Detailed related-work claims without a primary-paper rereading are attributed to the focal paper.
- Figure sources: original Figures 1, 2, 3, and 4, with source links and explanations at their respective locations; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Tables select and reorganize reported values; speedup ratios and differences were calculated from those values.
- **Not executed:** GPU training/inference reproduction, code/checkpoint execution checks, numerical gradient checks, independent reconstruction and assumption audit of the proofs, additional-seed experiments, and contamination checks. Numerical results are the authors' reports, not independently reproduced observations.
