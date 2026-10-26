---
title: "EGGROLL: expanding trainable systems through low-rank search"
description: "GPU-efficient perturbations, high-dimensional theory, integer RNN pretraining, and transformer fine-tuning, with research connections to Dust."
date: "2026-10-26"
publishedAt: "2026-10-26T19:34:40+09:00"
updatedAt: "2026-10-26T19:34:40+09:00"
topics: ["zeroth-order optimization", "evolution strategies", "credit assignment", "language model pretraining", "efficient training"]
visibility: "public"
lang: "en"
translationKey: "eggroll-low-rank-evolution"
paperTitle: "Evolution Strategies at the Hyperscale"
paperPublishedDate: "2025-11-20"
authors: ["Bidipta Sarkar", "Mattie Fellows", "Juan Agustin Duque", "Alistair Letcher", "Antonio León Villares", "Anya Sims", "Clarisse Wibault", "Dmitry Samsonov", "Dylan Cope", "Jarek Liesen", "Kang Li", "Lukas Seier", "Theo Wolf", "Uljad Berdica", "Valentin Mohl", "Alexander David Goldie", "Aaron Courville", "Karin Sevegnani", "Shimon Whiteson", "Jakob Nicolaus Foerster"]
year: 2025
paperUrl: "https://arxiv.org/abs/2511.16652v2"
thumbnail: "/assets/reviews/eggroll/eggroll-fig1.webp"
thumbnailAlt: "EGGROLL Figure 1: fitness-weighted low-rank candidates update dense weights."
---

EGGROLL makes weight perturbations low-rank so a GPU can efficiently evaluate a large evolution-strategy (ES) population. Its integer RNN pretraining reaches 3.40 bits/byte against a transformer baseline's 3.58, but the largest population uses about 180 times the GPU-hours. [§6.1, Figure 2](https://arxiv.org/html/2511.16652v2#S6.SS1) **Review interpretation.** This is a strong systems contribution that expands training to inference-friendly, non-differentiable models; compute efficiency against backprop remains a separate question requiring matched tasks and architectures.

An approach that increases learning signals through token-wise activation perturbations is explained in the independent [Dust review](/en/reviews/dust-activation-search/).

## 1. Introduction

ES evaluates the fitness of many perturbed candidates and combines successful directions to update parameters. Neither the model nor its objective needs differentiation, and candidates are easy to evaluate in parallel. At billions of parameters, however, replicating and moving all weights for each candidate becomes expensive. GPUs efficiently apply a large shared matrix to many inputs; different large matrices per input instead create a memory-traffic bottleneck.

Evolution Guided GeneRal Optimisation via Low-rank Learning (EGGROLL) represents perturbations as products of thin matrices. Individual candidates are low-rank, but their fitness-weighted sum is applied to the original dense weights and can have high rank. The paper supports this with high-dimensional analysis, integer language-model pretraining, control tasks, and language-model fine-tuning. [§1](https://arxiv.org/html/2511.16652v2#S1)

[![EGGROLL Figure 1: rank-one candidates are evaluated and combined into a dense weight update.](/assets/reviews/eggroll/eggroll-fig1.webp)](/assets/reviews/eggroll/eggroll-fig1.webp)

Original Figure 1. Different low-rank changes share initial weights. The aggregate update on the right can have higher rank than any candidate perturbation.

## 2. Preliminaries

### 2.1 Low-Rank Matrix Approximations

The paper's LoRA comparison is $M_i\approx M_i^0+A_iB_i^\top$, with frozen base weights $M_i^0$ and thin factors $A_i\in\mathbb R^{m\times r}$, $B_i\in\mathbb R^{n\times r}$. Whereas LoRA learns those factors, EGGROLL resamples perturbations at each step and folds the result into the dense weights. [§2.1](https://arxiv.org/html/2511.16652v2#S2.SS1)

### 2.2 Evolution Strategies

Let fitness $f:\mathbb R^d\to\mathbb R$ be maximized. With a Gaussian population of mean $\mu$ and covariance $\sigma^2I_d$, the smoothed objective and its update direction are

$$
J(\theta)=\mathbb E_{v\sim\mathcal N(0,I_d)}[f(\mu+\sigma v)],
\qquad
\nabla_\mu J(\theta)=\frac1\sigma\mathbb E[v f(\mu+\sigma v)].
$$

Here $\theta$ parameterizes the population distribution; the method updates $\mu$ and tunes $\sigma$. It averages $N$ candidates and takes an ascent step of size $\alpha_t$. Differentiation acts on the distribution, not through the fitness evaluation. [§2.2](https://arxiv.org/html/2511.16652v2#S2.SS2)

### 2.3 Evolution Strategies for Matrix Parameters

For $W,M,E\in\mathbb R^{m\times n}$ with a standard Gaussian matrix $E$, the estimate is

$$
\widehat\nabla_M J(\theta)=\frac{1}{\sigma N}\sum_{i=1}^N E_i f(W=M+\sigma E_i)
$$

Column-stacking with $\operatorname{vec}$ gives dimension $d=mn$. The Frobenius norm equals the Euclidean norm of the vectorized matrix, connecting the later vector-space proofs to matrix updates. [§2.3 and Appendix B](https://arxiv.org/html/2511.16652v2#S2.SS3)

## 3. Related Work

The authors first discuss distributed evolutionary algorithms and applications to policies, meta-learning, and hyperparameter search. Earlier low-rank methods have different goals: finding promising subspaces for small populations, or optimizing a factorized model whose weights remain low-rank. EGGROLL instead updates dense weights.

For language models, the comparisons include MeZO-style two-point estimates, low-rank and LoRA fine-tuning, and ES for reasoning. Earlier systems sometimes reuse each perturbation for many generated responses to utilize the GPU. EGGROLL aims to assign a different perturbation to each generation, making population size approach inference batch size. Its pretraining experiments connect the need for large populations to this distinction. [§3](https://arxiv.org/html/2511.16652v2#S3)

## 4. EGGROLL

### 4.1 Low-Rank Evolution Strategies

Assumption 1 samples all factor entries independently from the same zero-mean, unit-variance, symmetric, absolutely continuous distribution with finite fourth moments. The perturbation is

$$
E=\frac1{\sqrt r}AB^\top,
\qquad A\in\mathbb R^{m\times r},\quad B\in\mathbb R^{n\times r}
$$

Scaling by $1/\sqrt r$ keeps entry variance at one as rank grows. At low rank, this is not Gaussian; it lives on a low-rank manifold, so the ordinary Euclidean density score is not directly available. The authors use the Gaussian score $\hat S(E)=-E$ as an **approximation**.

Workers share scalar fitnesses, reconstruct perturbations from common random seeds, and update the original weights:

$$
M_{t+1}=M_t+\frac{\alpha_t}{N_{\mathrm{workers}}}
\sum_{i=1}^{N_{\mathrm{workers}}}E_{i,t}f(W=M_t+\sigma E_{i,t}).
$$

The constant $1/\sigma$ is absorbed into the learning rate here. This is Algorithm 1's update. Under generic non-degeneracy, the sum can have rank $\min(Nr,m,n)$. Restricting each candidate's perturbation differs from restricting the final weights to a low-rank subspace. [§4.1](https://arxiv.org/html/2511.16652v2#S4.SS1)

### 4.2 Hardware-Efficient Implementation

For row input $u_i$, the large perturbed matrix need not be materialized:

$$
u_i(M+\sigma E_i)^\top
=u_iM^\top+\frac{\sigma}{\sqrt r}(u_iB_i)A_i^\top.
$$

This preserves the large shared matrix multiplication and adds cheap candidate-specific work. At rank one, the extra work is a dot product followed by a scalar-vector product. The update likewise aggregates $\sum_iE_if_i$ as $(\operatorname{diag}(f)A)^\top B$. [§4.2](https://arxiv.org/html/2511.16652v2#S4.SS2)

Arithmetic intensity measures operations per transferred byte. In Appendix F's square-matrix bf16 model, full Gaussian matrix ES has intensity $m/(m+2)<1$, while EGGROLL reuses base weights as batch size grows. At width $m=8,192$, rank one, and a target of 300 operations/byte, ordinary inference requires batch size 324 and EGGROLL 352. [Appendix F](https://arxiv.org/html/2511.16652v2#A6)

[![EGGROLL Figure 2: normalized throughput and integer EGG pretraining loss.](/assets/reviews/eggroll/eggroll-fig2.webp)](/assets/reviews/eggroll/eggroll-fig2.webp)

Original Figure 2. On the left, pure batch inference is normalized to 100; EGGROLL reaches 91 and OpenES 0.41. On the right, colors represent population sizes in integer RNN training. Throughput and training quality are separate experiments.

The 91% result uses one GH200 GPU, a bf16 linear model of width 8,192, maximum batch size 1,024, and **pregenerated noise**. Figure 7 in Appendix E separately includes JAX noise regeneration. **Review interpretation.** This supports efficient candidate evaluation; it is not 91% of the total cost of reaching a target loss, or an end-to-end throughput guarantee for every network. [Appendix E](https://arxiv.org/html/2511.16652v2#A5)

## 5. Analysis

### 5.1 High-Dimensional Gaussian ES

A standard Gaussian vector typically has norm around $\sqrt d$. Small entry-wise noise can therefore produce a total perturbation of size $\sigma_d\sqrt d$. The paper studies $\sigma_d=o(d^{-1/2})$, where this vanishes and ES approaches a local derivative.

Theorem 1 assumes a fixed-radius $C^1$ neighborhood around initialization $\mu$, an $\alpha$-Hölder gradient there, globally polynomial-bounded fitness and derivatives with dimension-independent constants, and $O(1)$ bounds on $\lVert\mu\rVert$ and $\lVert\nabla f(\mu)\rVert$. It prints the error rate as $\Theta((\sigma_d\sqrt d)^\alpha)$. [§5.1, Theorem 1](https://arxiv.org/html/2511.16652v2#S5.SS1)

Appendix C.1's proof proceeds in this order. A cutoff equals one inside a local ball and zero outside a larger ball, creating an absolutely continuous function with integrable derivatives. A polynomial-tail lemma bounds the difference exponentially. Gaussian integration by parts, or Stein's lemma, converts reward-weighted noise for the cutoff function into an expected derivative. Local Hölder continuity then gives

$$
\mathbb E\lVert\nabla f(\mu+\sigma_dv)-\nabla f(\mu)\rVert
\le L\sigma_d^\alpha\mathbb E\lVert v\rVert^\alpha
=O((\sigma_d\sqrt d)^\alpha).
$$

This is the local bound, with a separate tail error. The cutoff matters because Stein's lemma cannot simply be applied to the original discontinuous function. The authors use $f(x)=\frac L2\sum_jx_j\lvert x_j\rvert+a^\top x$ at $\mu=0$ to attain error $L\sigma_d\sqrt{2d/\pi}$ when $\alpha=1$. [Appendix C.1](https://arxiv.org/html/2511.16652v2#A3.SS1)

Theorem 2 computes an exact remainder for cubic objectives

$$
f(x)=a^\top x+\frac12x^\top Bx+\frac16 C(x,x,x)
$$

where $B$ is symmetric, $C$ is a symmetric trilinear form, and their operator norms and that of $a$ are $O(1)$. Odd Gaussian moments vanish, leaving

$$
\nabla_\mu J-\nabla f(\mu)
=\frac{\sigma_d^2}{2}\mathbb E[C(v,v,\cdot)]
$$

The paper distinguishes localization, a critical regime $\sigma_d\asymp d^{-1/2}$, and slower decay that permits divergence for some objectives. The critical scale is a worst-case boundary for the function class. [Theorem 2 and Appendix C.2](https://arxiv.org/html/2511.16652v2#A3.SS2)

### 5.2 High-Dimensional EGGROLL

With rank $r$ fixed and dimension $d=mn$ growing, factor-product perturbations have heavier tails than Gaussian noise. Theorem 3 adds local $C^2$ regularity, a Lipschitz Hessian with constant $L_d$, and sub-Gaussian factor tails to the earlier polynomial-growth and bounded-derivative assumptions. It requires $\sigma_d=o(d^{-1/2})$ and $L_d(\sigma_dd)^2=o(1)$. The displayed bound is

$$
\lVert\hat g_{\mathrm{LR}}-\nabla_W f(M)\rVert_F
=O\!\left(L_d(\sigma_dd)^2\right)
+O\!\left(\frac{\sqrt d}{\sigma_d^2}
\exp\!\left[-K\frac{\rho}{\sqrt d\sigma_d}\right]\right).
$$

Here $K>0$ is a tail constant and $\rho$ the neighborhood radius; this $K$ is unrelated to Dust's population notation. The paper states that the full bound is $o(1)$ and that the difference from Gaussian ES also vanishes. [§5.2, Theorem 3](https://arxiv.org/html/2511.16652v2#S5.SS2)

The proof uses $v=\operatorname{vec}(E)$ with zero mean and identity covariance. The constant Taylor term vanishes by zero mean, the quadratic term by symmetry and zero third moments, and the linear term produces the desired derivative. Bounding the Hessian remainder gives

$$
\lVert I_{\mathrm{loc}}\rVert
\le\frac{L_d\sigma_d^2}{6}\mathbb E\lVert v\rVert^4
=O(L_d\sigma_d^2d^2)
$$

Sub-Gaussian factors give sub-exponential product tails and finite moments, controlling the integral outside the local ball. The final comparison to Gaussian ES uses the triangle inequality through their respective local-derivative errors. [Appendix C.3](https://arxiv.org/html/2511.16652v2#A3.SS3)

### 5.3 Rank Analysis

This limit keeps dimension fixed and increases rank $r$. Theorem 4 states an $O(r^{-1})$ difference from the Gaussian ES update under Assumption 1 and **globally bounded fitness**. [§5.3](https://arxiv.org/html/2511.16652v2#S5.SS3)

The vectorized perturbation is $\operatorname{vec}(E^r)=r^{-1/2}\sum_{j=1}^r\operatorname{vec}(a_jb_j^\top)$, with identity covariance for each summand. The multivariate central limit theorem gives the Gaussian limit. Appendix D uses an Edgeworth expansion to analyze the rate: symmetry removes third cumulants and the usual $r^{-1/2}$ term, leaving a fourth-cumulant $r^{-1}$ term. Bounds between the cited Edgeworth approximation and each distribution are combined by the triangle inequality. This is a fixed-dimension, fixed-$\sigma$ result, not a simultaneous dimension/rank/noise limit. Appendix D.1 also explores mean-field score approximators, but experiments use the Gaussian approximation. [Appendix D](https://arxiv.org/html/2511.16652v2#A4)

## 6. Experiments

### 6.1 Pure Integer Language Model Pretraining

EGG, the Evolved Generative GRU, is a six-layer nonlinear RNN of width 256. It uses int8 weights, integer activations, int32 accumulation, bit shifts, and lookup tables. Saturation and clipping when converting back to int8 provide nonlinearity without separate ReLU or tanh functions. Normalization uses mean absolute value instead of square roots; next-token log-likelihood uses exponential and logarithm tables. [§6.1 and Appendix G](https://arxiv.org/html/2511.16652v2#S6.SS1)

The integer variant shapes the fitness difference between positive and negative perturbations into $\{-1,0,1\}$. When the weighted matrix update exceeds a threshold, a weight moves by one integer bin. This differs from directly applying the real-valued ES formula on integers. [Appendix H](https://arxiv.org/html/2511.16652v2#A8)

Training uses character/byte-level MiniPile prediction with a fixed batch of 16 sequences. Each candidate updates after 100 tokens, retaining hidden state until document boundaries in truncated ES. Population ranges from 2 to $2^{20}=1,048,576$. Bits/byte is average negative base-two log probability of the correct byte; lower is better. The largest population reaches 3.40 against 3.58 for a same-depth, same-width fp32 transformer with backprop SGD and the same data batch, using about 180 times the GPU-hours. Architecture and numerical representation both change. [§6.1, Figure 2b](https://arxiv.org/html/2511.16652v2#S6.SS1)

[![EGGROLL Figure 8: EGG test loss against data batch size and population.](/assets/reviews/eggroll/eggroll-fig8.webp)](/assets/reviews/eggroll/eggroll-fig8.webp)

Original Figure 8. The x-axis is data batch size; colors indicate population. Larger batches can clean up candidate fitness estimates, which differs from adding perturbation directions. Appendix I also accounts for extra computation when $b>N/2$, since parallel evaluations become $\max(2b,N)$. The largest two populations perform similarly at data batch four and separate with more data. [Appendix I](https://arxiv.org/html/2511.16652v2#A9)

### 6.2 Reinforcement Learning Tasks

Sixteen tabula-rasa control environments use three-layer, width-256 MLP policies. Each method/environment pair receives hyperparameter optimization; selected configurations are evaluated over ten seeds. Returns are normalized by PPO, with standard errors. Compared with OpenES, EGGROLL is competitive on seven environments, better on seven, and worse on two. Episode-length changes can make total wall time differ from per-step throughput. [§6.2, Figure 4a](https://arxiv.org/html/2511.16652v2#S6.SS2), [Appendix N.4, Figures 14–15](https://arxiv.org/html/2511.16652v2#A14.SS4)

Three cooperative multi-agent environments use three-layer, width-64 policies, rank one, and A100 40GB GPUs. The paper reports up to 2.4× faster training than OpenES at large batches while remaining competitive. [Appendix N.1, Figure 12](https://arxiv.org/html/2511.16652v2#A14.SS1)

### 6.3 Foundation Model Fine-tuning

RWKV-7's recurrent state permits more parallel candidates than a transformer's growing KV cache. Countdown fine-tuning of a 1.5B model uses one H200 for eight hours and three seeds, reporting about 35% validation accuracy for EGGROLL versus 23% for Group Relative Policy Optimization (GRPO), a backprop-based policy-learning baseline. Fitness centers correctness per question, normalizes by a global standard deviation, and averages across questions. [§6.3](https://arxiv.org/html/2511.16652v2#S6.SS3), [Appendix N.2](https://arxiv.org/html/2511.16652v2#A14.SS2)

[![EGGROLL Figure 5: wall-clock GSM8K fine-tuning curves and hard reasoning results for RWKV-7 7B.](/assets/reviews/eggroll/eggroll-fig5.webp)](/assets/reviews/eggroll/eggroll-fig5.webp)

Original Figure 5. The left panel shows seed-level GSM8K curves on eight GPUs. Total parallel generations are 8,192 for EGGROLL and 256 for GRPO; generations per prompt, temperatures, and optimizers differ. The right panel is a separate 7B DeepScaleR experiment. Its caption states 128 GPUs for 12 hours, while the main text states 24 hours, an unresolved budget discrepancy. [Figure 5 and §6.3](https://arxiv.org/html/2511.16652v2#S6.F5), [Appendix N.3, Tables 12–13](https://arxiv.org/html/2511.16652v2#A14.SS3)

The 14B model uses 32 GPUs for 12 hours and a 5,000-token thinking budget in training and evaluation. Reported accuracies improve from 13% to 30% on AIME24, 7% to 33% on AIME25, and 11% to 13% on HMMT25. No equivalent GRPO run was feasible at this size; AIME24 selects the best checkpoint. A separate 7B Countdown result of 72.9% is compared with another paper's Qwen results, with different models and unavailable matching hardware details; the authors acknowledge the limits of fair time comparisons. [Figure 13 and Appendix N.2](https://arxiv.org/html/2511.16652v2#A14.F13)

### 6.4 Fine-tuning Integer Quantised LLMs

A per-channel absmax-int8 RWKV-7 7B model is distilled from its non-quantized teacher. GSM8K solutions are teacher-forced and summed token-wise Kullback–Leibler (KL) divergence is reduced. Figure 6 shows training perplexity and held-out GSM8K accuracy recovering across three seeds. [§6.4, Figure 6](https://arxiv.org/html/2511.16652v2#S6.SS4)

This path differs from pure-integer EGG. Large matrices are int8, while scales, some small parameters, and Adam moments use bf16. Adam proposals are normalized and thresholded into $-1,0,+1$ integer-bin changes. The printed KL sum is a distillation cost to minimize; a maximizing ES fitness must use its negative. [Appendix K](https://arxiv.org/html/2511.16652v2#A11)

## 7. Conclusion

The authors conclude that low-rank search enables large-model black-box optimization and dense updates even with small candidate ranks. Future directions include neurosymbolic systems with memory or calculation modules, models aware of inference harnesses, and end-to-end optimization of interacting agents. [§7](https://arxiv.org/html/2511.16652v2#S7)

## L. Additional transformer fine-tuning (Fine-tuning Pretrained Transformer LLMs with Verifiable Rewards)

Appendix L in v2 also fine-tunes transformers. Qwen3-4B-Base is trained on DeepScaleR's 40k math questions with rank one and population 2,048. The complete Table 1 is below. RL values are taken from the GEM paper rather than independently rerun in this study; the authors match relevant shared settings such as response length and prompts. [Appendix L.1](https://arxiv.org/html/2511.16652v2#A12.SS1)

| Method | MATH500 | OlympiadBench | AIME24 | AMC | MinervaMath | Reported mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B-Base | 50.2 | 24.4 | 10.0 | 33.7 | 21.7 | 28.0 |
| +EGGROLL | 75.8 | 37.3 | 13.3 | 49.4 | 31.3 | 41.4 |
| +RL (cited) | 67.4 | 33.5 | 16.7 | 49.4 | 40.1 | 41.4 |

Values are accuracy percentages. Equal means of 41.4 hide higher EGGROLL results on MATH500 and OlympiadBench and lower results on AIME24 and MinervaMath. The mean of rounded cells is 41.42 for both EGGROLL and RL. [Table 1](https://arxiv.org/html/2511.16652v2#A12.T1)

Qwen3-1.7B uses rank one and population 256 to optimize pass@1 or pass@4. Here pass@4 is the fraction of questions with at least one correct answer among four responses. This set reward is directly used as candidate fitness. Figure 10 shows increasing distinct final answers within four responses under pass@4 optimization and decreasing diversity under pass@1.

[![EGGROLL Figure 10: fitness and four-response answer diversity under pass@1 and pass@4 optimization.](/assets/reviews/eggroll/eggroll-fig10.webp)](/assets/reviews/eggroll/eggroll-fig10.webp)

Original Figure 10. Purple optimizes pass@1, pink pass@4. The left curves use different fitness metrics, so their heights are not a comparison on one common metric. The right counts distinct final answers in four responses. [Appendix L.1](https://arxiv.org/html/2511.16652v2#A12.F10)

This path uses vLLM multi-LoRA inference, tensor-parallel slice-aware updates, layer-wise temporary memory management, and GPU-to-GPU weight synchronization. [Appendix L.2, Tables 2–3](https://arxiv.org/html/2511.16652v2#A12.SS2)

## M. Additional time-series example (Fine-tuning Time Series Foundation Model: High-Frequency Trading)

Appendix M adapts an S5 order-flow model into an execution agent. Pretraining uses about 25B tokens of GOOG data from 2022; fine-tuning uses a fixed January 2023 window and the Jax-LOB simulator. The task sells 30 shares within ten steps, submitting a final market order for unexecuted quantity. Fitness is a rank transformation of proceeds relative to the initial mid-price, or PnL. Rank-four LoRA targets projection matrices while SSM and normalization parameters remain frozen. Figure 11's training curves are a supplementary demonstration of directly optimizing this objective. [Appendix M, Figure 11 and Table 4](https://arxiv.org/html/2511.16652v2#A13)

## Review interpretation and open questions

**Review interpretation.** Separating systems efficiency from statistical efficiency makes the contribution clearer. Evaluating many candidates without replicating large weights opens paths through integer computation, outcome-only rewards, and inference engines used for training. Integer EGG's data reuse and transformer fine-tuning illustrate different applications. The next question is how many cheap evaluations produce a useful update direction.

### Two qualifications for using the theory

**Review interpretation.** The printed $\Theta$ statements in Theorems 1–2 are too strong if read as two-sided error rates for every admissible function. Appendix C gives general upper bounds and examples attaining them. Linear $f(x)=a^\top x$ satisfies the assumptions with exactly zero Gaussian ES error. For cubic $f(x)=x_1^3$ at $\mu=0$, the error is $3\sigma_d^2$ regardless of dimension, rather than $\Theta(\sigma_d^2d)$. The general statement is better separated into an $O$ upper bound and worst-case sharpness. The original theorem statements and examples are discussed in §5.1 above.

**Supplementary derivation.** For the authors' worst-case example $f(x)=(u^\top x)\lVert x\rVert^2$ with $\lVert u\rVert=1$, direct Gaussian averaging yields

$$
\mathbb E[\nabla f(\sigma_dv)]
=\sigma_d^2\{u\mathbb E\lVert v\rVert^2+2\mathbb E[vv^\top]u\}
=\sigma_d^2(d+2)u
$$

Thus the necessity of the critical scale for some objectives survives.

**Review interpretation.** The step declaring Theorem 3's displayed tail bound automatically $o(1)$ also needs additional control. With $\sigma_d=d^{-1/2}/\log\log d$, one has $\sigma_d\sqrt d\to0$, but that displayed term becomes $d^{3/2}(\log\log d)^2(\log d)^{-K\rho}$ and does not vanish. This identifies an insufficient bound-to-conclusion step, not a counterexample proving divergence of the actual update. For instance, $\sigma_d=d^{-1/2-\varepsilon}$ with $\varepsilon>0$ makes the displayed tail vanish; the local remainder condition must also hold. Appendix C.3's two-term bound is the strongest relevant argument. This qualification does not overturn empirical results or the separate rank-limit Theorem 4.

**Review interpretation.** Finite-population variance, fitness shaping, and integer-grid dynamics are separate from limits of expected updates. A theorem about local derivatives alone cannot explain successful training of a quantized model whose ordinary local derivative is zero. Understanding the search performed by useful finite noise remains valuable.

### Connecting to Dust: from cheap evaluations to useful credit

**Review interpretation.** [Dust's method](/en/reviews/dust-activation-search/#2-method) scores token-wise activation changes instead of giving a global score to each candidate. Its weak EGGROLL-Transformer baseline does not contradict this paper's integer RNN or RL fine-tuning successes. Dense-loss pretraining from scratch, outcome-reward adaptation of a pretrained model, and integer recurrence are different conditions.

**Review interpretation.** The priority is a cost curve measuring both axes. At matched models, data, and GPU time, compare cost to target loss and account for population, token-level signals, communication, caches, and tuning. A first experiment should test layer-wise EGGROLL/Dust hybrids on dense-loss pretraining, measuring estimator variance and test loss. Replacing dense loss with a final outcome reward tests whether Dust's local-signal advantage persists. The hybrid assumptions and falsification criterion are collected in [Dust's connection section](/en/reviews/dust-activation-search/#connecting-to-eggroll-allocating-budget-where-information-is-weak).

## Sources and reading scope

- Bidipta Sarkar et al., *Evolution Strategies at the Hyperscale*, [arXiv:2511.16652v2](https://arxiv.org/abs/2511.16652v2), first published 2025-11-20, revised 2026-02-16; 76 pages. Reviewed 2026-10-26.
- Scope: §§1–7; central theorem and proof paths in Appendices A–D; speed, integer-model, ablation, and quantization details in E–K; additional experiments in L–M; principal settings and learning curves in N. Repetitive configuration rows and peripheral mean-field derivations are condensed. Additional sources are the [official project](https://eshyperscale.github.io/) and [Dust](https://qlabs.sh/research/dust).
- Figures: v2 Figures 1, 2, 5, 8, and 10 under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Figure regions are separated from page prose and captions, rasterized, and encoded as WebP. Panels, axes, data, and original fonts are preserved.
- Unverified: independent GPU reproduction of the authors' experiments and formal verification of the complete proofs.
