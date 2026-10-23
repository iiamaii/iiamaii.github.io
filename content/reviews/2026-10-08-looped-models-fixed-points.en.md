---
title: "Can an endpoint replace the recurrent path? Efficiency at fixed points in looped models"
description: "Explain TBPTT, KV sharing, RL state reuse, and distilled prefill through fixed points, separating update-segment speedups from total time and accuracy tradeoffs."
date: "2026-10-08"
updatedAt: "2026-10-23T15:26:13+00:00"
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

This paper argues that when a looped language model's states are near fixed points, the endpoint can replace the recurrent path and cut the cost of training activations, the KV cache, prefill, and reinforcement-learning (RL) scoring; it also proposes a learned depth prior and orthogonal input injection to shape such fixed points. When the 1.6B (L) model is trained with RL on GSM8K at recurrence depth $R=6$, reusing states saved during rollout halves the per-update median scoring-plus-backward time relative to full backpropagation through time (full BPTT), from 2.77 to 1.39 seconds, but total training time, dominated by rollout, falls only about 12%, from 65.45 to 57.30 minutes, and GSM8K pass@1 drops from 63.20% to 61.65%. [Table 4](https://arxiv.org/html/2610.06833v1#S5.T4) · [Table 25](https://arxiv.org/pdf/2610.06833v1#page=42) **Review interpretation.** The measurements support the claim that each shortcut reduces the cost of its own stage. The introduction's claim that savings grow with every added recurrence holds for terminal KV sharing ($R=16$ and 32) and distilled prefill ($R=25$), but RL scoring is measured at the single depth $R=6$ and gives no evidence for it.

**Figure 1 · How to read the representative figure.** The left panel presents two training improvements, the learned depth prior and OrthoInj. The center contrasts the recurrent path with an endpoint near a fixed point. The right panel summarizes four computational shortcuts. Its bars report separate experiments, not cumulative speedups from one combined system. In particular, the RL 2× refers to scoring and backward, not total training time. [Benhao Huang et al., 2026, Figure 1](https://arxiv.org/html/2610.06833v1#S0.F1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · Original retained · [View full size](/assets/reviews/looped-models-fixed-points/paper-figure-1.svg).

This article follows original §1–7. Appendix A.1’s Lemma 1 is explained alongside §3.2; Appendix A.2’s Lemma 2 accompanies §3.1. For RL and prefill, §3 explains mechanisms and §5 their quality and runtime tradeoffs.

## 1. Introduction

A conventional Transformer passes activations through layers with distinct weights. A **looped language model** repeatedly runs a shared block, increasing computation per token without adding weights. However, retaining every recurrence's intermediate states makes training memory and generation caches grow with recurrence depth. Processing the initial prompt, called **prefill**, also becomes expensive. **Decoding** then generates subsequent tokens one at a time. In RL, every loop is replayed to score each rollout. [§1](https://arxiv.org/html/2610.06833v1#S1)

Whether looped models should learn fixed points is contested. One camp forgoes fixed points and keeps the loop count small and fixed; the other shapes fixed points and loops much deeper. The authors observe that the fixed-point equation $z^\star=F_\theta(z^\star;x)$ involves only the endpoint, not the path, so once states are near a fixed point the endpoint can stand in for the whole trajectory. Their evidence is terminal KV sharing, which keeps only each earlier token's last-recurrence KV for every recurrence of later tokens to read. In cited prior measurements, this sharing barely changes Huginn's zero-shot GSM8K accuracy but collapses Ouro's at four loops, a split that tracks convergence: Huginn's states approach fixed points and Ouro's do not. [§1](https://arxiv.org/html/2610.06833v1#S1)

The authors argue that the same fixed points give each remaining cost a shortcut. Truncated backpropagation through time (TBPTT), which backpropagates through only the last few recurrences, and terminal KV sharing are shortcuts looped models already take; fixed points explain why they work and yield two new ones, RL state reuse and distilled prefill. The representative number for the first contribution is that, for a 1.6B model trained at fixed depth with full BPTT, terminal KV sharing drops GSM8K accuracy from 50.6% to 21.2%. To shape such fixed points, they also change the training-depth distribution and input injection. [§1, §3–4](https://arxiv.org/html/2610.06833v1#S3)

## 2. Fixed Points in Looped Language Models

Recurrent states in looped language models tend to approach fixed points as recurrence depth grows (prior observations). The experimental backbone is Huginn. A prelude embeds the input; a recurrent core containing two Transformer blocks repeatedly receives that input representation; a coda converts the endpoint into next-token probabilities. The complete recurrent state $z$ contains both hidden states $\mathbf H$ and attention key/value banks $C$. [§2](https://arxiv.org/html/2610.06833v1#S2)

$$
\begin{aligned}
e&=P_\theta(x),\\
z^r&=(\mathbf H^r,C^r),\\
z^{r+1}&=F_\theta(z^r;x),\\
z^\star&=F_\theta(z^\star;x).
\end{aligned}
$$

At a mathematical fixed point, one more application leaves the state unchanged. The paper also calls approximately stationary finite-depth states fixed points. [§2](https://arxiv.org/html/2610.06833v1#S2)

Figure 2 plots relative changes between adjacent recurrent hidden states, with recurrence on the vertical axis and token position horizontally. A later token can settle before an earlier one, undermining a simple left-to-right convergence story. Across qualifying tokens from eight observed sequences, the Spearman correlation between position and convergence depth is a weak 0.14. [Figure 2, Appendix C.4](https://arxiv.org/pdf/2610.06833v1#page=35)

$$
\rho_{r,t}=\frac{\|h_t^r-h_t^{r-1}\|_2}{\|h_t^{r-1}\|_2+\epsilon}.
$$

The figure identifies the first depth after which changes remain below 2% through the observation horizon, requiring at least four stable updates. With a horizon of 64, Appendix Table 19 reports that 99.74% of tokens in the Small PLN-5 model meet a 1% threshold. This describes **state stability** on a finite observation window and specified data. Under much stricter convergence checks, some inputs fail and others require hundreds of recurrences. [Appendix C.4, Table 19; A.1](https://arxiv.org/pdf/2610.06833v1#page=36)

<figure class="review-figure" id="paper-figure-2">
<a href="/assets/reviews/looped-models-fixed-points/paper-figure-2.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/looped-models-fixed-points/paper-figure-2.png" width="836" height="232" alt="Token-position versus recurrence hidden-state-change heatmap, local convergence-depth reversal, and position/depth scatterplot." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 2 · Tokens do not settle in causal order</span>Darker areas indicate smaller relative state changes. The zoomed staircase shows neighboring tokens exchanging their convergence-depth order. The weak position/depth correlation is specific to the observed sequences. <span class="figure-links"><a href="/assets/reviews/looped-models-fixed-points/paper-figure-2.png" target="_blank" rel="noopener noreferrer">View full size</a> · <a href="https://arxiv.org/html/2610.06833v1#S2.F2">Benhao Huang et al., 2026, Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original retained; no edits or cropping</span></figcaption>
</figure>

## 3. What Fixed-Point Shaping Enables

Original §3 argues that fixed-point shaping, training with supervision near the fixed points of the recurrence, lets the endpoint stand in for the trajectory and so decouples per-recurrence training and inference costs from depth. The four shortcuts remove and retain the following. [§3](https://arxiv.org/html/2610.06833v1#S3)

| Application | What is removed | What remains or is required |
| --- | --- | --- |
| Pretraining TBPTT | Storage and backward computation for earlier recurrences | Forward computation to the sampled depth |
| Terminal KV sharing | Previous tokens' visit-specific KV banks | Current-token recurrence and visit-local current-token KV |
| RL rollout-state reuse | Recomputing sampled tokens' endpoints | One core forward graph, several backward vector–Jacobian products (VJPs), and rollout generation |
| Distilled prefill | Multiple teacher recurrences over the prompt | Student, one teacher recurrence and coda, then teacher decoding |

A VJP applies a backward derivative without constructing the full Jacobian, the matrix of partial derivatives. RL rollout and evaluation keep visit-specific KV. [§3, Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=40)

<figure class="review-figure" id="paper-figure-3">
<a href="/assets/reviews/looped-models-fixed-points/paper-figure-3.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/looped-models-fixed-points/paper-figure-3.svg" width="508" height="160" alt="Original schematic of suffix backpropagation, terminal-prefix KV sharing, and student prefill with recurrent teacher decoding." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 3 · Three different computational paths</span>(a) Detach below the dashed boundary and backpropagate through the suffix. (b) Reuse previous tokens’ terminal KV across the current token’s recurrences. (c) Predict the prefix endpoint with a non-recurrent student and retain teacher decoding. Read the text for the intervening teacher recurrence and coda used to build the banks. <span class="figure-links"><a href="/assets/reviews/looped-models-fixed-points/paper-figure-3.svg" target="_blank" rel="noopener noreferrer">View full size</a> · <a href="https://arxiv.org/html/2610.06833v1#S3.F3">Benhao Huang et al., 2026, Figure 3</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original retained; no edits or cropping</span></figcaption>
</figure>

### 3.1 Training: Endpoint Gradients

Full BPTT differentiates through all $R$ recurrences, so its cost grows with depth. The authors argue that near a fixed point the gradient can instead be computed from the endpoint. [§3.1](https://arxiv.org/html/2610.06833v1#S3.SS1)

#### Pretraining: truncated backpropagation and Lemma 2

TBPTT differentiates the last $b$ recurrences and detaches earlier states. At a fixed point $z^\star$, let $J_\star=\partial_zF_\theta(z^\star;x)$ and $B_\star=\partial_\theta F_\theta(z^\star;x)$. If $F_\theta$ is continuously differentiable and the spectral radius $\rho(J_\star)$ is below one, original Equation (1) gives three sensitivities. Here $I$ is the identity matrix, and $J_r$ and $B_r$ are the same derivatives evaluated at the visited state $z^r$. [§3.1, Eq. 1](https://arxiv.org/html/2610.06833v1#S3.E1)

$$
\begin{aligned}
D_\star&=\frac{\mathrm dz^\star}{\mathrm d\theta}=(I-J_\star)^{-1}B_\star
=\sum_{k=0}^{\infty}J_\star^kB_\star,\\
D_b&=\sum_{k=0}^{b-1}J_\star^kB_\star,\qquad
\tilde D_b=\sum_{k=0}^{b-1}\Bigl(\prod_{j=1}^{k}J_{R-j}\Bigr)B_{R-1-k}.
\end{aligned}
$$

$D_\star$ is the equilibrium sensitivity, determined by the endpoint alone, and equals the $R\to\infty$ limit of full BPTT when the trajectory converges. The truncated Neumann estimate $D_b$ has bias $D_\star-D_b=J_\star^bD_\star$, which decays geometrically in $b$, but both are meaningful only once a stable fixed point exists. Endpoint gradients applied from initialization therefore destabilize training, which is why implicit models first warm up with unrolled backpropagation. TBPTT's $\tilde D_b$ instead uses Jacobians at visited states, so from any state it is the exact gradient of the loss through the last $b$ recurrences with earlier states held fixed, and as visited states converge to $z^\star$ it approaches $D_b$ regardless of $R$. The authors conclude that TBPTT is as stable as full BPTT early in training and recovers the fixed-point gradient once fixed points form, without a warm-up phase or a switch of estimator. [§3.1](https://arxiv.org/html/2610.06833v1#S3.SS1.SSS0.Px1)

The authors add that **truncation alone shapes fixed points**. A fixed point has two testable properties: extra recurrences leave the state unchanged, and terminal KV sharing costs little. Full BPTT at fixed depth $R=5$ fails both, with perplexity (PPL) rising by over 25% when extrapolated to $R=64$ and by 21% under terminal KV sharing. TBPTT through the last two recurrences ($b=2$) raises it by at most 2% and 3%, respectively, at a cost of about 0.07–0.10 cross-entropy (CE) at the training depth. An endpoint-Neumann control trained from initialization collapses at all five tested learning rates. With sampled depths at Medium PLN-5, windows $b\ge 4$ keep WikiText PPL within 1% of full BPTT, while $b=2$ is approximately 5.5% worse and $b=1$ approximately 15% worse. [§3.1; Appendix C.3, Tables 17–18](https://arxiv.org/pdf/2610.06833v1#page=34)

Separate memory measurements report peak allocation of 48/86/124GB for windows 5/10/15 at local batch eight. Full BPTT at depth 30 runs out of memory; per-layer activation recomputation lowers memory to 28GiB but takes approximately 1.3 times the step time of window ten. [§3.1, Appendix C.3](https://arxiv.org/pdf/2610.06833v1#page=33)

**Lemma 2, step by step.** Lemma 2 in Appendix A.2 states that the gradient $G_b$ obtained by detaching an exact equilibrium and differentiating $b$ applications of the same recurrence equals the truncated Neumann series, and that under $\|J_\star\|_2\le\kappa<1$ the omitted tail obeys the bound below. Here $J_\star\in\mathbb R^{d_z\times d_z}$ and $B_\star\in\mathbb R^{d_z\times d_\theta}$. Differentiating the fixed-point equation gives $D_\star=J_\star D_\star+B_\star$, so the equilibrium sensitivity requires $I-J_\star$ to be invertible. Continuous differentiability and this invertibility permit the implicit function theorem. The stronger condition $\rho(J_\star)<1$ ensures convergence of the Neumann series.

Let $c$ be the direct parameter gradient holding state fixed, and $g$ the column gradient with respect to endpoint state. Detaching an exact equilibrium and applying the same recurrence $b$ times gives $D_0=0$ and $D_{k+1}=J_\star D_k+B_\star$, so $G_b=c+D_b^\top g$. The following **review-supplied intermediate derivation** expands Appendix A.2, Eqs. (14)–(17).

$$
\begin{aligned}
D_\star&=J_\star D_\star+B_\star,\\
(I-J_\star)\sum_{k=0}^{b-1}J_\star^k&=I-J_\star^b,\\
G_{\mathrm{imp}}&=c+D_\star^\top g,\qquad G_b=c+D_b^\top g,\\
\|G_{\mathrm{imp}}-G_b\|_2
&\leq\|B_\star\|_2\|g\|_2\sum_{k=b}^{\infty}\kappa^k
=\|B_\star\|_2\|g\|_2\frac{\kappa^b}{1-\kappa}.
\end{aligned}
$$

The constant $1/(1-\kappa)$ needs a uniform Euclidean operator-norm bound. If only the spectral radius is below one, the paper notes that for any $\gamma$ between it and one the error is $O(\|B_\star\|_2\|g\|_2\gamma^b)$. Input/prelude paths remain in $B_\star$ and the output-head path in $c$. [Appendix A.2, Lemma 2](https://arxiv.org/pdf/2610.06833v1#page=23)

Appendix A.2 also qualifies the relation between the two estimators. Finite-depth and equilibrium losses optimize different predictions, so equal endpoint values need not have equal parameter derivatives: for $F_{\theta,a}(z;x)=az+(1-a)e$, the equilibrium $z^\star=e$ is independent of $a$, whereas $z^R=(1-a^R)e$ depends on it. The finite-trajectory $\tilde D_b$ (appendix Eq. (18)) reduces to $D_b$ when all visited Jacobians equal $J_\star$ and $B_\star$. Because early derivatives come from transient states and truncation adds its own error, the authors state that neither estimator is uniformly more accurate and that a small endpoint residual alone is not enough to control the approximation error. [Appendix A.2](https://arxiv.org/pdf/2610.06833v1#page=22)

#### Post-training: reusing rollout states for RL

RL generates responses, then scores the same tokens to update the policy. The authors save rollout endpoint $z^R$ and, before any parameter update, apply the core once there to construct a differentiation graph. The forward value remains the saved state, while repeated endpoint-Jacobian VJPs approximate the gradient. **Neumann-4 means four state VJPs through the same graph plus the identity term: five series terms, not four forward passes.** [Algorithm 1; Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=40)

Algorithm 1 treats a state’s **value** and **derivative path** separately. In the expression below, $\operatorname{sg}$ is stop-gradient. Since $f-\operatorname{sg}(f)$ has value zero, the forward state remains saved $z^R$, while derivatives use the graph of one recorded $F_\theta$ evaluation.

$$
\bar z=\operatorname{sg}(z^R),\quad f=F_\theta(\bar z;x),\quad
\widetilde z=\bar z+\bigl(f-\operatorname{sg}(f)\bigr),\qquad
v\leftarrow g+J_R^\top v.
$$

Starting from readout gradient $g$, initialize $v=g$, repeat the VJP $b-1$ times, then propagate $B_R^\top v$ to parameters. Endpoints belong to the same on-policy rollout before a parameter update and are discarded afterward. The authors note that the update cost and memory of reuse are therefore independent of $R$, though rollout generation still takes most of the training time. [§3.1, Algorithm 1](https://arxiv.org/pdf/2610.06833v1#page=4)

### 3.2 Inference: Endpoint Caching and Distillation

At inference, past-token context and prompt endpoints are reused.

#### Decoding: terminal KV sharing and Lemma 1

The model has four physical attention layers: one prelude, two core blocks, and one coda. At $R=5$, the logical depth is $1+2\times 5+1=12$. A visit-specific cache stores twelve banks; terminal sharing retains four, one per physical layer. At $R=16/32$, it retains four instead of 34/66. These are **KV-storage savings**. [§3.2, Table 20](https://arxiv.org/pdf/2610.06833v1#page=36)

Teacher-forced training, however, updates all tokens of a sequence in parallel, so recurrence $r$ of the current token reads prefix banks from depth $r$, whereas sharing supplies terminal banks from the first recurrence. The authors explain that this gap is small when prefix banks change little across depth and the current token's update contracts, and that sampled training depths push the model toward this regime. [§3.2](https://arxiv.org/html/2610.06833v1#S3.SS2.SSS0.Px1)

Lemma 1 shows that joint iteration and iteration with a converged prefix frozen reach the same limit if the prefix context converges, the current-token update is uniformly contractive on the specified invariant region, and it is Lipschitz in context. [§3.2, Lemma 1](https://arxiv.org/html/2610.06833v1#S3.SS2.SSS0.Px1)

**Lemma 1: assumptions and proof.** The original $\mathcal H$ is a nonempty closed Euclidean subset, invariant under updates for all relevant contexts. Completeness and uniform $\kappa<1$ contraction permit the Banach contraction theorem, yielding a unique fixed point for frozen context $C^\star_{<t}$. Extending this to joint iteration additionally requires $C^r_{<t}\to C^\star_{<t}$ and $\beta$-Lipschitz dependence on context.

Appendix A.1 sets $\delta_r=\|h^r-h^\star\|$ and obtains the first line below by the triangle inequality. Context convergence makes $\eta_{\max}$ finite. Induction bounds $\delta_r$; taking limsup gives finite $\bar\delta\leq\kappa\bar\delta$, hence $\bar\delta=0$.

$$
\begin{aligned}
\delta_{r+1}&\leq\kappa\delta_r+\beta\|C^r_{<t}-C^\star_{<t}\|,\\
\delta_r&\leq\max\left\{\delta_0,\frac{\beta\eta_{\max}}{1-\kappa}\right\},\qquad
\eta_{\max}=\sup_r\|C^r_{<t}-C^\star_{<t}\|,\\
\bar\delta&=\limsup_{r\to\infty}\delta_r\leq\kappa\bar\delta,\qquad \bar\delta=0.
\end{aligned}
$$

Continuity of the state-and-context readout then gives the same limiting next-token distribution. The token-by-token extension inducts over position: preceding tokens’ limits determine the next context, independently of the order in which tokens settle. The appendix notes that if the prefix input contains state beyond KV, sharing only KV falls under Lemma 1 only when that state also coincides. Freezing at finite depth $R$ a context $\bar C_{<t}$ with error $\eta=\|\bar C_{<t}-C^\star_{<t}\|$ leaves the two error terms of Eq. (12). [Appendix A.1, Lemma 1, Eqs. (9)–(12)](https://arxiv.org/pdf/2610.06833v1#page=20)

$$
\|\bar h^R-h^\star\|
\le\kappa^R\|\bar h^0-h^\star\|
+\frac{\beta(1-\kappa^R)}{1-\kappa}\,\eta.
$$

The first term measures unfinished current-token refinement. The second reflects an imperfect frozen context and does not vanish merely by adding recurrences. When $\kappa$ approaches one, context error can be strongly amplified.

Of 304 prefixes under strict empirical checks, 206 qualify for the fixed-depth model and 191 for PLN-5. Their median prefix convergence depths are 152.5 and 114, far beyond the operational depth of five. Freezing at depth five instead gives current-token endpoint gaps five to six orders of magnitude larger than freezing a strictly converged prefix. The authors themselves state that practical sharing at $R=5$ rests on the finite-depth bound of Eq. (12) and measured cached perplexity, **not on exact convergence**. [Appendix A.1, Figure 6](https://arxiv.org/pdf/2610.06833v1#page=21)

Appendix C.6 independently varies prefill and decoding depths over eight values between 1 and 32, giving an 8×8 comparison. Once decoding is sufficiently deep, prefill beyond five brings little gain. With shallow decoding, shallow prefill can work better. The authors summarize that because decoding reads only terminal banks, the two depths can be set separately, and that prefill gains saturate at moderate depth while decoding gains persist deeper. [§3.2 footnote; Appendix C.6, Figure 11](https://arxiv.org/pdf/2610.06833v1#page=37)

#### Prefill: distill the recurrent endpoint

The authors explain that because decoding reads only terminal banks, prefill can be distilled into a student that predicts only the endpoint; without terminal sharing, the student would have to reproduce the KV banks of every recurrence. A non-recurrent two-block student predicts the teacher's pre-coda endpoint and trains with normalized hidden-state error and KL, or Kullback–Leibler, divergence to the teacher output distribution. The teacher is frozen, and the student consumes an additional quarter of the teacher's pretraining token budget. **One teacher recurrence and the coda turn the student prediction into terminal KV; subsequent decoding still runs the teacher at depth five.** [§3.2, Eq. 4; §5.5, Appendix C.7](https://arxiv.org/pdf/2610.06833v1#page=38)

Equation (4) has two distinct training targets. Here $N$ is prompt length, $d$ hidden width, and $\sigma_{\mathrm T}^2$ teacher-state energy calibrated on training data. The first term is per-coordinate state error on the teacher’s scale; the second compares teacher distribution $p_t$ with $\widehat p_t$ read from the student state.

$$
\mathcal L_{\mathrm{distill}}(\psi)=\mathbb E_x\!\left[
\frac{\|\widehat z-z_\theta^R(x)\|_F^2}{Nd\,\sigma_{\mathrm T}^2}
+\frac1N\sum_{t=1}^{N}\mathrm{KL}(p_t\Vert\widehat p_t)
\right].
$$

$\widehat p_t$ applies the frozen coda and output head directly to the student state, unlike the inference handoff, which passes through one teacher recurrence after the student. Initialized from the teacher core, the student approximately outputs the teacher's state after one recurrence before training, and Equation (4) moves this output toward the depth-$R$ endpoint. [§3.2, Eq. (4)](https://arxiv.org/pdf/2610.06833v1#page=6)

## 4. Improving Fixed-Point Shaping

Original §4 holds that the benefits of §3 depend on how training shapes fixed points, and improves the two techniques that do the shaping: depth sampling, which sets where supervision lands, and input injection, which sets what a fixed point retains about the input. [§4](https://arxiv.org/html/2610.06833v1#S4)

### 4.1 Learning a Depth Prior for Fixed-Point Training

The Huginn baseline samples from a shifted Poisson-lognormal, or PLN, distribution with mean depth five. The authors argue that under a fixed training budget, the depth prior trades prediction quality at the target depth against fixed-point shaping: fixed-depth training concentrates supervision at one depth and predicts better there, while PLN spreads supervision across depths and yields the fixed-point behavior that supports terminal KV sharing. The paper initializes a learned categorical distribution over depths 1–64 from that prior. This is **one global training distribution, not an input-dependent stopping policy**. One depth is sampled per microbatch before running the recurrence. [§4.1, Algorithm 2; Appendix B.3](https://arxiv.org/pdf/2610.06833v1#page=25)

$$
\begin{aligned}
\mathcal J_{\mathrm{prior}}(\phi)
&=-\widehat{\mathbb E}_{n}\!\left[\operatorname{sg}(A_n)\log p_\phi(r)\right]\\
&\quad-\lambda_H H(p_\phi)
+\lambda_m\!\left(\mathbb E_{p_\phi}[R]-\bar R\right)^2,\quad\bar R=5.
\end{aligned}
$$

The advantage $A_n$ comes from $\exp(-CE)$, an inverse-perplexity reward, adjusted using moving averages and variance. The language model still trains with CE. Entropy $H(p_\phi)$ discourages collapse to one depth; the final term keeps mean depth near the compute budget. The controller reuses the already-computed model loss instead of adding another model forward pass. Defaults are $\lambda _H=0.01$, $\lambda _m=1$, and a maximum backward window of ten. [§4.1, Appendix B.3](https://arxiv.org/pdf/2610.06833v1#page=25)

In Figure 4, the orange fixed-$R=5$ model becomes unstable outside its training depth. PLN and learned-depth training remain more stable, with the learned prior lowering PPL relative to PLN. [Figure 4; Appendix C.5](https://arxiv.org/pdf/2610.06833v1#page=8)

<figure class="review-figure" id="paper-figure-4">
<a href="/assets/reviews/looped-models-fixed-points/paper-figure-4.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/looped-models-fixed-points/paper-figure-4.svg" width="496" height="168" alt="WikiText PPL across test depths 1–64 for S and M models, comparing fixed depth, PLN, and learned priors." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 4 · Perplexity outside the training depth</span>The horizontal axis is test recurrence depth; lower WikiText PPL is better. Fixed-depth training performs well near five but becomes unstable elsewhere. Sampled and learned priors are more stable. Insets expose small differences near five. <span class="figure-links"><a href="/assets/reviews/looped-models-fixed-points/paper-figure-4.svg" target="_blank" rel="noopener noreferrer">View full size</a> · <a href="https://arxiv.org/html/2610.06833v1#S4.F4">Benhao Huang et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original retained; no edits or cropping</span></figcaption>
</figure>

### 4.2 Orthogonal Input Injection

Parcae-style injection carries the previous state into the core as $\Lambda h_t^r+q_t$, where $q_t$ is the injected input and $\Lambda$ is a learned diagonal decay. The carryover can already contain a component along $q_t$, changing the effective magnitude of that direction. OrthoInj removes that component before adding $q_t$. [§4.2, Eqs. 7–8](https://arxiv.org/pdf/2610.06833v1#page=8)

$$
\begin{aligned}
q_t&=\Delta\odot We_t,\\
Q_{q_t}&=I-\frac{q_tq_t^\top}{\|q_t\|_2^2+\epsilon},\\
\zeta_t^r&=Q_{q_t}\Lambda h_t^r+q_t.
\end{aligned}
$$

Here $\odot$ denotes elementwise multiplication. For nonzero $q_t$ and $\epsilon =0$, the component along $q_t$ is exactly $q_t$; the implementation uses $\epsilon =10^{-6}$. Since $\|Q_{q_t}\|_2\le1$, the projected carryover stays contractive: $\|Q_{q_t}\Lambda\|_2\le\|\Lambda\|_2<1$. The recipe also removes prelude RMSNorm. A 2×2 appendix comparison finds contributions from both changes to validation PPL, with a separately selected learning rate for each configuration. [§4.2; Appendix B.4, C.2, Table 14](https://arxiv.org/pdf/2610.06833v1#page=32)

## 5. Experiments

Original §5 first evaluates the two training changes from §4, the learned depth prior and OrthoInj, and then tests RL rollout-state reuse and distilled prefill.

### 5.1 Setup

The paper's S/M/L names round the **non-embedding parameter counts**. Total looped-model counts are larger. [§5.1; Appendix B.1, Table 6](https://arxiv.org/pdf/2610.06833v1#page=24)

| Scale | Looped non-embedding parameters | Total including embeddings | Pretraining tokens |
| --- | --- | --- | --- |
| S, “100M” | 0.106B | 0.304B | 21.47B |
| M, “400M” | 0.425B | 0.819B | 85.90B |
| L, “1.6B” | 1.699B | 2.488B | 343.60B |

All use context length 8,192, global batch 512, BF16, a mixture including web, QA, code, and mathematics, and learning-rate screening per configuration. There is one training seed per configuration. The official implementation README identifies NVIDIA H200 as the experimental hardware. [Appendix B.1](https://arxiv.org/pdf/2610.06833v1#page=24), [official implementation README](https://github.com/ifm-ai/xllm-loop#installation)

PPL exponentiates token-averaged CE and is better when lower. **AVG equally weights accuracy on LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, OpenBookQA, and SciQ**; it excludes GSM8K, DROP, and MBPP+. Some entropy-sweep appendix tables use an AVG with WinoGrande in place of SciQ. Main GSM8K results use all 1,319 test questions, eight-shot prompting, and greedy accuracy. DROP uses a fixed 500-question subset, three-shot prompting, and F1. MBPP+ uses 100 fixed tasks, three-shot prompting, and pass@1 estimated from eight samples at temperature 0.8. [Appendix D.2, D.5, Table 27](https://arxiv.org/pdf/2610.06833v1#page=44)

PPL also differs in scoring scope across tables. **Terminal-KV PPL in Tables 2–3 and native PPL in Tables 8/13 score different tokens.** The former scores document continuations after a 4,096-token prefix; the latter scores full sequences. The paper itself notes that only the task columns compare directly across the two tables, and measures the sharing penalty in Tables 20/21 and CacheEval, which score identical tokens with both cache layouts. [Appendix D.3–4](https://arxiv.org/pdf/2610.06833v1#page=45)

### 5.2 Learning the Depth Prior

The following Table 2 comparisons replace fixed PLN-5 with the learned prior at $\lambda _H=0.01$. All are evaluated at test depth five with four terminal KV banks, and use Parcae without prelude normalization rather than OrthoInj. [Table 2](https://arxiv.org/pdf/2610.06833v1#page=10)

| Scale | Val. PPL ↓ | WikiText PPL ↓ | Seven-task AVG, % ↑ | GSM8K, % ↑ |
| --- | --- | --- | --- | --- |
| S | 5.73 → 5.67 | 20.19 → 19.91 | 41.32 → 41.65 | 1.52 → 1.59 |
| M | 4.00 → 3.93 | 12.55 → 12.28 | 49.06 → 49.07 | 13.72 → 15.62 |
| L | 3.10 → 3.07 | 8.76 → 8.68 | 59.30 → 60.00 | 47.61 → 47.92 |

PPL improves at all scales, but Medium AVG improves by just 0.01 percentage point (pp). Training FLOPs rise by at most 1.6% (Large: $1253\to1273\times10^{19}$). The entropy term is essential at Large: without it ($\lambda_H=0$), GSM8K drops from 47.92% to 36.47%.

In the same table, fixed-depth training has the highest PPL under terminal KV sharing, and sharing cuts its GSM8K accuracy relative to the full cache by more than half: from 15.54% to 3.34% at Medium and from 50.64% to 21.23% at Large. [Tables 2, 8](https://arxiv.org/pdf/2610.06833v1#page=28)

At Large scale, learned-prior terminal-sharing AVG 60.00 is close to fixed-depth full-cache AVG 59.90. It still trails the distinct-weight twelve-layer Untied 12's 61.47. The roughly threefold reduction relative to Untied 12 concerns non-embedding parameters and KV-bank count. Quality exceeds same-physical-depth Untied 4, but training FLOPs are $1273/465\approx2.74$ times larger. **Review interpretation.** The advantage over Untied 4 is therefore an advantage at equal parameters and cache, not at equal compute. [Tables 2, 6, 8](https://arxiv.org/pdf/2610.06833v1#page=28)

In matched-token comparisons at $R=5$, sharing raises validation PPL by at most 0.85% (Large learned prior, 3.047 to 3.073), and that model's GSM8K falls from 50.42% to 47.92%. At $R=16$ and 32, the same model's PPL gap is at most 0.11%. [Table 20](https://arxiv.org/pdf/2610.06833v1#page=36)

A consequential control appears in the appendix. At Small scale, sampling from the final learned distribution **held fixed from the first update** gives validation PPL 5.63 versus 5.67 for the adapting prior. The authors read this as evidence that the learned prior finds a better depth distribution without a search over distributions. At Medium, higher entropy lowers the CacheEval sharing penalty monotonically from +0.96% to −0.22%, but WikiText PPL at $R=5$ rises from 12.89 to 13.20, and the sharing penalty on WikiText and LAMBADA does not fall monotonically. [Appendix C.1, Tables 10, 12](https://arxiv.org/pdf/2610.06833v1#page=29)

### 5.3 Input Injection

Table 3 compares injection recipes under fixed PLN-5. Relative to Parcae-Decay, OrthoInj improves validation PPL and AVG, but not every WikiText or generation score. [Table 3](https://arxiv.org/pdf/2610.06833v1#page=11)

| Scale | Val. PPL ↓ | WikiText PPL ↓ | AVG, % ↑ | GSM8K, % ↑ | MBPP+, % ↑ |
| --- | --- | --- | --- | --- | --- |
| S | 5.75 → 5.67 | 19.90 → 20.02 | 41.28 → 41.65 | 1.90 → 1.90 | 0.25 → 1.62 |
| M | 3.96 → 3.94 | 12.45 → 12.32 | 49.03 → 49.16 | 15.01 → 13.95 | 8.25 → 7.88 |
| L | 3.09 → 3.08 | 8.72 → 8.72 | 59.22 → 59.26 | 52.92 → 51.71 | 30.88 → 27.75 |

Large AVG gains only 0.04pp, while code performance declines. The text's “lowest training loss, validation and WikiText PPL at every scale” describes the end of training, and native PPL over full sequences (Table 13) agrees: OrthoInj has the lowest WikiText PPL at every scale (Small 21.16 vs. Parcae 21.24, Large 9.21 vs. 9.22). Under terminal-KV scoring in Table 3, Parcae is lower at Small (19.90 vs. 20.02) and the two tie at Large. In Table 15, which also trains Medium at $1.5\times10^{-3}$, OrthoInj has lower PPL at both learning rates, but its AVG leads by 0.13 points at $10^{-3}$ and trails by 0.51 at $1.5\times10^{-3}$. [§5.3; Appendix C.2, Tables 13, 15](https://arxiv.org/pdf/2610.06833v1#page=33)

In Table 16, the observed Parcae carryover amplifies rather than cancels the input direction. Median effective gains are approximately 2.18/2.75/3.43 at S/M/L, and the authors report that no token's carryover cancels half of the input. OrthoInj removes this component so that every recurrence receives the input at the same strength. [Appendix C.2, Table 16](https://arxiv.org/pdf/2610.06833v1#page=33)

### 5.4 Rollout-State Reuse in RL

The comparison uses the Large learned-prior model of Table 2 at depth six, where its greedy GSM8K accuracy is highest, with Dr. GRPO and sixteen responses per training question. A recomputation control computes the same Neumann gradient as reuse but rebuilds the endpoint by replaying the recurrences without gradient tracking. GSM8K training covers 400 questions; evaluation uses 500 fixed test questions and eight samples each. Here pass@1 averages the eight outcomes, unlike the main greedy accuracy; pass@8 counts questions with at least one successful response. [§5.4; Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=41)

| Method | GSM8K pass@1 / @8, % | MATH500 pass@1, % | Update, seconds ↓ | GSM total training, minutes ↓ |
| --- | --- | --- | --- | --- |
| Before RL | 40.63 / 77.60 | 13.40 | — | — |
| Full BPTT | 63.20 / 88.00 | 18.27 | 2.77 | 65.45 |
| Recompute + Neumann-4 | 60.05 / 89.60 | 17.32 | 2.09 | 61.71 |
| Reuse + Neumann-4 | 61.65 / 90.20 | 17.82 | 1.39 | 57.30 |
| Reuse + Neumann-3 | 61.30 / 89.80 | 18.77 | 1.30 | 58.98 |

Update time is the **per-update median scoring-plus-backward time, excluding rollout and the optimizer step**. The 2.77→1.39-second change is approximately 1.99 times faster. Total training time (Train in Table 25, including rollout, answer checking, and updates) changes from 65.45 to 57.30 minutes, approximately 1.14 times faster. Of these 8.15 minutes, 4.45 come from shorter rollouts (57.42 to 52.97 minutes, with sampled tokens falling from 1,009,546 to 886,381), while summed update time falls from 463.5 to 244.1 seconds, about 3.7 minutes. [Tables 4, 25](https://arxiv.org/pdf/2610.06833v1#page=42)

For MBPP+ code RL, updates improve from 1.03 to 0.51 seconds, but total time **increases from 20.0 to 21.8 minutes**: the reuse arms' rollouts take 17.0–17.8 minutes against 14.4 for full BPTT, outweighing their 1.5–1.6-minute saving outside rollout. Pass@1 changes from 40.88 to 39.38 and pass@8 from 69 to 64. [Table 26](https://arxiv.org/pdf/2610.06833v1#page=43)

The authors argue that because reuse and the recomputation control compute nearly identical gradients (cosine 0.997–0.999), the gap between their pass@1 (1.6 points on GSM8K) measures run-to-run variation, and conclude that reuse trails full BPTT by no more than this variation on any benchmark. Reuse minus full-BPTT GSM8K pass@1 is −1.55pp, with a paired question-bootstrap 95% interval of $[-3.13,0.05]$ that conditions on one training seed. The reuse/full-BPTT gradient cosine is 0.76–0.87, and the mean relative endpoint residual is 0.034. The paper notes that at $R=6$ the four state VJPs of Neumann-4 are close to the six recurrent visits of full BPTT, so this experiment does not test how the saving scales with depth. Multiple seeds, the full test split, and reuse across updates are also untested. [Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=42)

### 5.5 Distilled Prefill

| Scale | Teacher → distilled prefill, ms ↓ | Prefill speedup | AVG, % ↑ | GSM8K, % ↑ | MBPP+, % ↑ |
| --- | --- | --- | --- | --- | --- |
| S | 195.0 → 125.9 | 1.55× | 41.65 → 40.63 | 1.59 → 1.74 | 0.62 → 0.62 |
| M | 495.8 → 299.3 | 1.66× | 49.07 → 46.77 | 15.62 → 8.95 | 10.38 → 6.12 |
| L | 1482.1 → 830.3 | 1.79× | 60.00 → 55.53 | 47.92 → 32.90 | 27.25 → 16.50 |

These are means over six warmed prefill repetitions, with 8K prompts and batch eight. Every prefill path is followed by the same teacher decoding at $R=5$. Distilled prefill is approximately 1–4% faster than stopping teacher prefill after two recurrences and gains 0.5–0.9pp AVG. Stopping after three recurrences stays within 1.6 points of the teacher's AVG but saves only 21–29% of its latency, against 35–44% for distilled prefill. A single recurrence is 12–23% faster than distilled prefill but loses 2.7–3.5 times as much AVG. Against full teacher prefill, Large AVG loses 4.47pp and GSM8K loses 15.02pp. [Table 5, Appendix C.7.1](https://arxiv.org/pdf/2610.06833v1#page=13)

Further controls expose handoff sensitivity. Feeding the student's own KV directly to the teacher costs 2–7pp AVG relative to the one-teacher-recurrence path. Standalone students do not beat Untied 4. Distilling a deeper $R=25$ teacher gives 4.3–5.4× prefill speedups but larger quality losses; its student is slower and less accurate than the $R=5$ student. This comparison uses **hidden loss only**, unlike the main hidden-plus-KL recipe. Medium students fit the deeper endpoint better (relative MSE 0.011–0.017 vs. 0.036) yet lose more quality, which the authors attribute to the deeper teacher's decoder being more sensitive to small handoff errors rather than to state error. [Appendix C.7, Tables 22–23](https://arxiv.org/pdf/2610.06833v1#page=39)

## 6. Related Work

The original section has three strands. Looped models increase test-time computation through shared blocks, while DEQs solve directly for limiting fixed points. Earlier work stabilizes those points, trains supervised states as attractors, or halts at convergence; this paper examines efficiency across training, inference, and post-training. [Section 6](https://arxiv.org/pdf/2610.06833v1#page=13)

Second, the efficiency strand explains existing TBPTT and recurrence-level KV sharing through fixed points, then connects new RL state reuse and non-recurrent student prefill. Unlike recurrent-student distillation or predictors of each recurrence’s prompt KV, the student predicts only the endpoint and one teacher recurrence builds all KV banks. [Section 6 and appendix E](https://arxiv.org/pdf/2610.06833v1#page=46)

Third, fixed-point shaping concerns supervised depth distributions and input injection. Ouro and RL-Halting learn state-dependent entropy-regularized depth distributions while unrolling to maximum depth. This paper learns an input-independent prior; in contrast to Parcae’s contractive injection, OrthoInj removes the carryover component along the input direction. [Section 6](https://arxiv.org/pdf/2610.06833v1#page=13)

## 7. Discussion and Conclusion

The authors conclude that near a fixed point the endpoint can replace the recurrent trajectory to cut costs in training, inference, and post-training, and that whether training produces such fixed points depends on the depth prior and input injection. The learned prior lowers PPL over a fixed PLN prior at the same mean depth while keeping terminal KV sharing intact, and OrthoInj gives lower PPL than the injection schemes of Huginn and Parcae. They summarize terminal KV sharing and rollout-state reuse as cutting memory or time at a small cost in quality, and distilled prefill as trading downstream accuracy for prefill latency, with a gap that grows with scale.

As limitations, the authors list models of up to 1.6B parameters, one seed per configuration, and ablations that do not exhaust the design space, adding that larger scales, other configurations, and real-world feasibility need further study. The appendix states that pretraining decontamination was not verified for the RL experiments. Combining the learned prior with OrthoInj, or adding terminal KV sharing to RL, is not tested. [§7; Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=14)

<span id="review-interpretation-and-follow-up-checks" aria-hidden="true"></span>

## Review interpretation and open questions

**Review interpretation.** The strongest evidence is the matched-token terminal KV sharing measurement: the PPL increase is at most 0.85% at depth five and at most 0.11% at $R=16$ and 32, so storage savings grow with depth while the loss does not. The contrast with fixed-depth full-BPTT models, which lose more than half their GSM8K accuracy under the same sharing, directly shows the effect of fixed-point shaping. A strength of the paper is that it measures separately, for each shortcut, what replaces the omitted path and which error remains: context error, gradient approximation, or handoff error.

The limitation is that the cost differs by shortcut. RL reuse halves the update stage, but about half of the total-time gain comes from shorter rollouts, and the claim of equal quality rests on using the gap between two conditions, reuse and recomputation, in a single seed as the measure of variation. Distilled prefill cuts latency by 35–44%, but its AVG loss grows with scale from 1.0 point at Small to 4.5 at Large. OrthoInj's gain at Large is 0.04 points of AVG and 0.01 of validation PPL.

Three questions remain. Do RL-reuse savings actually grow in much deeper models? Is online adaptation of the depth prior itself necessary? At Small, the control that freezes the final distribution from the start reached lower PPL, and no Medium/Large or multi-seed comparison exists. How well do state distance or PPL predict generation success? As in the $R=25$ distillation, a better-fitted state can still yield worse generation.

## Sources and reading scope

- Paper: *Towards Looped Models Done Right, Part II: Rethinking at Fixed Points* · Benhao Huang et al. · first public date 2026-10-05 · [reviewed version](https://arxiv.org/abs/2610.06833v1) · reviewed 2026-10-23.
- Scope read: Sections 1–7 and appendices A–E, including equations, algorithms, and experiments; proofs from A.1–A.2 accompany the corresponding methods.
- Additional sources: the [official implementation README](https://github.com/ifm-ai/xllm-loop) (experimental hardware).
- Figure sources: Figures 1 (thumbnail), 2, 3, and 4 are unchanged author assets from v1, CC BY 4.0. [Paper](https://arxiv.org/abs/2610.06833v1).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
