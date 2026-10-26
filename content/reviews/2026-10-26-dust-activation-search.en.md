---
title: "Dust: token-wise search approaches backpropagation"
description: "Virtual populations and credit assignment, with measured losses separated from population extrapolation and connected to EGGROLL."
date: "2026-10-26"
publishedAt: "2026-10-26T19:34:40+09:00"
updatedAt: "2026-10-26T19:34:40+09:00"
topics: ["zeroth-order optimization", "evolution strategies", "credit assignment", "language model pretraining", "efficient training"]
visibility: "public"
lang: "en"
translationKey: "dust-activation-search"
paperTitle: "Dust: Pretraining Transformers Without Backpropagation"
paperPublishedDate: "2026-10-05"
authors: ["Samip Dahal", "Bishwas Mandal", "Serdar Gülbahar", "Akshay Vegesna"]
year: 2026
paperUrl: "https://qlabs.sh/research/dust"
thumbnail: "/assets/reviews/dust/dust-mechanism.webp"
thumbnailAlt: "Dust: token-wise activation perturbations, loss evaluation, credit assignment, and weight updates."
---

Dust replaces backpropagation with token-wise activation perturbations and estimates learning signals from the resulting loss changes. In FineWeb pretraining on 1M tokens, its test cross-entropy at a nominal population of 1,024 is 5.934, below backprop's 5.959; at 20M tokens, even its largest population gives 4.802, above backprop's 4.633. [Table 1](https://qlabs.sh/research/dust/appendix.html#tab-1) **Review interpretation.** The convincing result is that additional computation can produce learning close to backpropagation; the evidence for reducing today's training costs remains limited.

The weight-space alternative is explained in the independent [EGGROLL review](/en/reviews/eggroll-low-rank-evolution/).

## 1. Introduction

Language models must assign prediction errors to many weights. Backpropagation performs this credit assignment through the chain rule. Dust asks whether search—trying many small changes and combining those that reduce loss—can perform this job. It is a zeroth-order method: it queries function values rather than differentiating through the network.

The authors argue that abundant computation may permit learning algorithms and architectures less constrained by differentiability. Their contributions concern virtual populations in activation space, more efficient pretraining than weight-space evolution strategies (ES), population efficiency in larger models, and alignment with backprop gradients. Making the method compute-efficient enough to replace backprop, and applying it to external programs or deeply looped models, remain future work. [§1](https://qlabs.sh/research/dust#introduction)

## 2. Method

### 2.1 Activation-Space Perturbation

In weight-space ES, a candidate is a perturbed model whose weight changes act across an entire sequence. Dust instead adds independent Gaussian noise to linear-layer outputs at each token. A forward evaluation containing thousands of tokens therefore contains many separate search directions. All weights use activation perturbations except the $2L$ residual mixing scalars, which use ordinary weight-space ES; $L$ is the number of blocks. [§2.1](https://qlabs.sh/research/dust#activation-space-perturbation)

[![Dust's token-wise perturbations, loss evaluations, credit assignment, and weight updates.](/assets/reviews/dust/dust-mechanism.webp)](/assets/reviews/dust/dust-mechanism.webp)

Reviewer illustration. Tokens within a draw receive separate scores. The reported population $K$ counts draws allocated across layer groups, and does not always count full-network forward passes.

### Supplementary derivation: the one-token connection to EGGROLL

For an input $x_t\ne0$, one weight change producing an output change $\sigma a_t$ is

$$
\Delta W_t=\sigma\frac{a_t x_t^\top}{\lVert x_t\rVert^2},
\qquad \Delta W_t x_t=\sigma a_t.
$$

This explanatory equation is added by the review. It is rank one for one token, but independent changes chosen for different tokens need not correspond to one shared $\Delta W$ across the sequence. **Review interpretation.** Dust is better understood as a different credit-assignment design allowing token-wise search and rewards, rather than simply implementing EGGROLL's low-rank perturbations elsewhere. The shared weight perturbation is explained in [EGGROLL §4](/en/reviews/eggroll-low-rank-evolution/#4-eggroll).

### 2.2 Credit Assignment

For the paper's linear layer $y_t=W x_t$, draw $a_t\sim\mathcal N(0,I)$ and evaluate $y_t+\sigma a_t$, where $\sigma$ is the noise scale. Compare the perturbed token loss $\ell_s$ to the mean loss $\tilde\ell_s$ across draws evaluated together: $c_s=\tilde\ell_s-\ell_s$. A positive score means lower loss.

Since an earlier token's perturbation can affect later tokens, Equation 1 introduces credit decay $\gamma$:

$$
r_t=\sum_{s\ge t}\gamma^{s-t}c_s.
$$

Equations 2–3 average reward-weighted noise across $K$ draws to estimate output error and form the outer product with the cached input:

$$
\hat g_t=-\frac{1}{K\sigma}\sum_{i=1}^{K}r_t^{(i)}a_t^{(i)},
\qquad
\widehat G_W=\sum_t\hat g_t x_t^\top.
$$

The dimensions of $\hat g_t$ match the layer output, and those of $\widehat G_W$ match $W$. Backprop forms the same local outer product, but obtains output error by differentiation through the network. For embeddings, the one-hot input turns the update into a scatter-add. The implementation divides by the valid-token count for a mean-loss update. [§2.2, Equations 1–3](https://qlabs.sh/research/dust#credit-assignment)

### 2.3 Interference and Tuning

Simultaneous perturbations interfere: the score for one direction includes effects from others. The authors use separate evaluations for different blocks and layer families, caching the clean forward and rerunning only the relevant suffix. Queries, keys, values, gates, and value embeddings receive separate perturbations. The language-model head perturbs cached logits in vocabulary slabs and recomputes cross-entropy, permitting many cheap head draws.

Attention-internal perturbations barely affect token losses. They instead use an estimated attention-output error $\hat g_s$ as a target and score the output change $\Delta o_s$ with Equation 4:

$$
c_s=-\langle\hat g_s,\Delta o_s\rangle.
$$

Equation 1 then assigns credit per head. Direct-loss estimates and queries use $\gamma=0$; keys, values, gates, and value embeddings generally use 0.98. The 16k-population cells at 10M and 20M tokens use 0.99. [§2.3](https://qlabs.sh/research/dust#interference-and-tuning), [Appendix E](https://qlabs.sh/research/dust/appendix.html#population-allocation)

Noise scales, allocations, and decay can be chosen by short training searches or by screening for cosine similarity to a backprop gradient on one batch, followed by training confirmation. The final recipe uses the latter diagnostic. The public minimal implementation assigns estimated gradients to parameter gradient fields and calls SGD without autograd. Thus backprop-free updates and the use of backprop while choosing settings are distinct. [§2.3 and Appendix F](https://qlabs.sh/research/dust/appendix.html#tuning), [official implementation](https://github.com/qlabs-eng/dust/blob/b20f7c03eac630b6441ba6f254128c1761dc45ad/dust.py#L353)

## 3. Pretraining Without a Backward Pass

### 3.1 Setup

The base decoder-only transformer has eight blocks, width 512, and about 37.7M parameters. It uses RMS normalization, rotary positions, squared-ReLU MLPs, and value embeddings in alternating blocks. FineWeb is tokenized with a 4,096-token BPE vocabulary. Training uses eight sequences of 2,048 tokens per batch, one epoch, constant-rate momentum SGD, and three seeds per cell. Validation and test each contain 544 sequences. The reported test loss is taken at the best-validation checkpoint. [§3.1](https://qlabs.sh/research/dust#setup), [Appendix A](https://qlabs.sh/research/dust/appendix.html#architecture)

The metric is average negative natural-log probability of the correct token, in nats/token; lower is better. EGGROLL-Transformer applies weight-space ES to the same transformer. It differs from the original EGGROLL paper's integer RNN, EGG, in architecture, tokenizer, and data.

Dust's $K$ counts draws scored using token losses. Attention internals, head draws, and residual-scalar evaluations are additional; suffix reruns are cheaper than full forwards. At 1M tokens with SGD and $K=16,384$, writers and MLP hidden layers each receive 7,040 draws, attention outputs 1,664, and embeddings 640. The head gets 65,536 additional draws. Adam uses $11K/8$ direct-loss draws at the same nominal $K$. The 20M/16k cell shifts work to attention internals and uses 14,336 direct-loss draws. The authors report less compute for Dust than EGGROLL at equal nominal populations from 256 upward. [Appendix E, Tables 3–4](https://qlabs.sh/research/dust/appendix.html#population-allocation)

### 3.2 Main Results

This selection from Table 1 retains the smallest population, a population near the crossover with backprop, the largest population, and stronger backprop results. Values are three-seed mean test losses at validation-selected checkpoints.

| Method / nominal population | 100k tokens | 1M tokens | 10M tokens | 20M tokens |
| --- | ---: | ---: | ---: | ---: |
| Dust / 64 | 7.240 | 6.290 | 5.397 | 5.265 |
| Dust / 1,024 | 7.151 | 5.934 | 5.111 | 4.960 |
| Dust / 16,384 | 7.170 | 5.916 | 5.049 | 4.802 |
| EGGROLL-Transformer / 16,384 | 7.263 | 6.708 | 6.033 | 5.865 |
| Backprop | 7.216 | 5.959 | 4.989 | 4.633 |

[Table 1](https://qlabs.sh/research/dust/appendix.html#tab-1)

[![Measured test losses for Dust, EGGROLL-Transformer, and backprop. The x-axis is nominal population; the 100k panel has its own y range.](/assets/reviews/dust/dust-measured-loss.webp)](/assets/reviews/dust/dust-measured-loss.webp)

A replot of all finite-population rows in Table 1, not new experiments. Each point is a three-seed mean. Dust can beat backprop with little data, but larger token budgets require larger populations. The 100k ladder is not monotonically improving. Figure 1's training curves also show large-population Dust tracking backprop, while EGGROLL falls behind earlier. [Figures 1–2](https://qlabs.sh/research/dust#main-results)

The “1,000–10,000×” claim concerns an **extrapolated population ratio needed for EGGROLL to reach Dust's loss at 64 draws**, rather than a measured wall-clock ratio. Within the observed range, EGGROLL at 16k has 256 times the population yet still falls short of Dust at 64. Appendix D extrapolates using a power law and straight lines in log population. At 1M tokens the full power-law fit predicts 23,000×; a line through the top three populations predicts 8,200×. Some leave-one-out fits never cross the target at a finite population. [Appendix D, Table 2](https://qlabs.sh/research/dust/appendix.html#eggrolls-extrapolated-population)

At 20M tokens, Dust's fitted infinite-population loss is 4.431, with a seed-bootstrap 95% interval of 3.89–4.58. The authors emphasize that the observed ladder is still falling, so this is not a measured limit. **Review interpretation.** It motivates larger-population experiments; it is neither an achieved 4.431 loss nor established superiority over backprop.

### 3.3 Dust Under Adam

Retuning with Adam at 1M tokens improves Dust and backprop. EGGROLL stays within 0.01 of its SGD result at every population. Figure 3 reports backprop at 5.361 and Dust's fitted infinite-population limit at 5.248, with a 95% interval of 5.17–5.30. Dust and backprop use five seeds, EGGROLL three. The fit remains distinct from observed cells. Additional embedding draws and learning-rate groups appear in Tables 3 and 7. [§3.3, Figure 3](https://qlabs.sh/research/dust#dust-under-adam)

## 4. Search in High-Dimensional Space

### 4.1 Overparameterization

At a fixed 10M tokens, the study compares 2.0M, 7.3M, 38M, and 243M parameters. At population 1,024, test losses are 5.265, 5.161, 5.158, and 5.214. Expressive capacity can offset the increased difficulty of noisy estimation. The largest model is not always best: at 16k, the 38M and 243M models yield 5.036 and 5.086, while their backprop baselines yield 5.015 and 5.048. [§4.1, Figure 4](https://qlabs.sh/research/dust#overparameterization)

This study predates the main comparison's recipe. Embedding and key/value learning rates differ, so its rows should not be combined with §3 merely because model size matches. Both methods select settings per size and population. [Appendix F, Table 6](https://qlabs.sh/research/dust/appendix.html#tuning)

### 4.2 Emergence of Backprop-Like Gradients

The authors measure per-layer cosine similarity between Dust estimates and backprop gradients on the same batch. A value of 1 indicates the same direction, 0 orthogonality, and −1 the opposite direction; magnitude and training performance are separate. The checkpoints were **trained by backprop for 10M, 100M, and 1B tokens**. This is not Dust pretraining to 1B tokens. Populations range from 64 to 131,072. Figure 6 in Appendix B also shows Dust at initialization and EGGROLL at initialization and 100M tokens. Equation 5 fits

$$
\cos(K)=\frac{c_{\max}}{\sqrt{1+c/K}}.
$$

Here $c_{\max}$ is the fitted alignment ceiling and $c$ the population budget reaching $c_{\max}/\sqrt2$. [§4.2, Figure 5](https://qlabs.sh/research/dust#emergence-of-backprop-like-gradients)

[![Selected layer-type cosine fits at a backprop-trained 100M-token checkpoint. The head and key projections require very different populations.](/assets/reviews/dust/dust-cosine-fits.webp)](/assets/reviews/dust/dust-cosine-fits.webp)

Six layer-type fits from Table 8, plotted only over the measured population range. The head has $c=759$, while keys have $c=616,595$. Fitted ceilings are about 0.666 for value gates and 0.732 for token embeddings. **Review interpretation.** A useful next step separates population-reducible variance from directional bias due to noise, decay, and local surrogate targets. This diagnostic does not imply that increasing $K$ alone makes every layer converge exactly to backprop.

## 5. Conclusion

The authors envision credit assignment by computation-rich search beyond the architectures, optimizers, and hardware that have co-evolved with backprop. They interpret overparameterization as useful search space and geometry. Improving compute efficiency, testing new architectures, and co-designing optimizers remain future directions. [§5](https://qlabs.sh/research/dust#conclusion)

## 6. Related Work

The paper distinguishes three lines. Weight perturbation methods include MeZO for fine-tuning and EGGROLL for cheaper low-rank candidate evaluation. Earlier zeroth-order language-model pretraining is distinguished by scale and comparisons with backprop, while searching around pretrained models differs from learning representations from scratch.

The activation-perturbation lineage includes node perturbation and GEMINI. Zoop combines output perturbations with local derivatives for fine-tuning. Forward-gradient methods use forward-mode automatic differentiation, while predictive-coding methods propagate credit through local derivatives and iterative dynamics. Dust combines forward perturbations, token rewards, and local attention-output targets. These distinctions define the comparison scope of its “first competitive zeroth-order transformer pretraining” claim. [§6](https://qlabs.sh/research/dust#related-work)

## Review interpretation and open questions

**Review interpretation.** The important shift is from evaluating more candidates to deciding where a forward evaluation produces useful credit. Dense token losses make this effective. The method also uses structural knowledge: linear outer products, causal attention, per-head scoring, and selected credit decay. It demonstrates both the value of abundant computation and useful inductive structure.

**Review interpretation.** The evidence supports the authors' priority claim within their stated task and comparison setting, but does not establish priority over every alternative to backprop. Keeping both the small-data wins and larger-data deficits best describes its current position. Equal-GPU-time competitiveness including tuning costs needs separate evaluation.

### Connecting to EGGROLL: allocating budget where information is weak

**Review interpretation.** [EGGROLL's hardware path](/en/reviews/eggroll-low-rank-evolution/#42-hardware-efficient-implementation) reduces the cost of evaluating a candidate; Dust increases the credit information obtained from evaluation. A first combination experiment should compare weight ES, Dust, and a layer-wise hybrid on the same transformer and FineWeb batches at **equal GPU time**. Record test loss, per-layer cosine and estimator variance, noise scale, cache memory, and tuning time to test whether allocating more budget to weak keys or embeddings helps.

**Review interpretation.** A hybrid using global EGGROLL perturbations for discontinuous integer or external-program modules and Dust for linear blocks with dense losses remains a hypothesis. A minimal falsification test replaces dense token losses with a sparse final-token reward in the same model. If the local-reward advantage disappears, the hybrid's role becomes clearer. The original outcome- and set-level objectives are explained in [EGGROLL Appendix L](/en/reviews/eggroll-low-rank-evolution/#l-additional-transformer-fine-tuning-fine-tuning-pretrained-transformer-llms-with-verifiable-rewards).

## Sources and reading scope

- Samip Dahal, Bishwas Mandal, Serdar Gülbahar, and Akshay Vegesna, *Dust: Pretraining Transformers Without Backpropagation*. [Official article](https://qlabs.sh/research/dust) and [Appendices A–G](https://qlabs.sh/research/dust/appendix.html), accessed 2026-10-26. The page states October 2026; the public date follows the homepage-linked [author announcement of 2026-10-05](https://x.com/industriaalist/status/2107194534501433804).
- Scope: §§1–6 and Appendices A–G. Implementation details refer to the [pinned official minimal implementation](https://github.com/qlabs-eng/dust/tree/b20f7c03eac630b6441ba6f254128c1761dc45ad). EGGROLL methods and experiments are covered in the [companion review](/en/reviews/eggroll-low-rank-evolution/).
- Figures: original explanatory diagram; test-loss plot reconstructed from Table 1; cosine curves reconstructed from Table 8. They are not new measurements. Encoded as WebP.
- Unverified: independent reproduction of large-scale pretraining and the full hyperparameter search.
