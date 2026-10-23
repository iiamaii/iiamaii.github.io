---
title: "[1/2] Diffusion Forcing: What Changes When Time and Noise Can Be Controlled Separately?"
description: "Follow independent per-token noise from the training objective to long video rollouts, Monte Carlo Guidance, maze planning, and robotic memory in the authors' section order."
date: "2026-10-14"
publishedAt: "2026-10-14T13:58:28+09:00"
updatedAt: "2026-10-22T23:22:52+09:00"
topics: ["diffusion models", "sequence modeling", "world action model", "robot learning", "probabilistic modeling"]
visibility: "public"
lang: "en"
translationKey: "diffusion-forcing-methods"
paperTitle: "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion"
paperPublishedDate: "2024-07-01"
authors: ["Boyuan Chen", "Diego Marti Monso", "Yilun Du", "Max Simchowitz", "Russ Tedrake", "Vincent Sitzmann"]
year: "2024"
paperUrl: "https://arxiv.org/abs/2407.01392v4"
thumbnail: "/assets/reviews/diffusion-forcing/overview.webp"
thumbnailAlt: "Original Figure 2: sequence time and diffusion noise axes, with independent per-token training and sampling compared against teacher forcing and full-sequence diffusion."
---

A next-frame model can easily generate sequences of different lengths. Small prediction errors, however, can accumulate until its video collapses. Diffusing several frames together makes it easier to guide a sequence toward a future objective, but resolving every frame at the same noise level restricts flexibility in length and sequential control. **Diffusion Forcing trains on independent noise levels for different tokens, allowing sampling to move between these regimes.** It uses that freedom for rollout stabilization and planning with future rewards. The experiments support these capabilities; they do not establish superiority in every environment or accurate generation at infinite length. [§1, pp. 1–2](https://arxiv.org/pdf/2407.01392v4#page=1)

This review divides the 35-page v4 into two articles. This part follows **main sections 1–5**, incorporating essential experimental and implementation details from appendices B–F. [\[2/2\] Theory and proof walkthrough](/en/reviews/diffusion-forcing-theory/) covers appendix A's ELBO and subsequence claims. Read this article first. The theory article distinguishes the printed statements from supplementary derivations because the source contains mathematical inconsistencies.

## 1. Introduction

A **token** can be a video frame, a multivariate observation, or a robot action–observation bundle. It need not be a word.

Teacher forcing predicts the next token from ground-truth history during training. At inference, the model instead consumes its own outputs. On continuous data, this mismatch can cause compounding error.

Full-sequence diffusion adds and removes noise from a collection of tokens. In the conventional setup considered here, all tokens share the same noise level at a diffusion step. This supports guidance toward desirable future outcomes.

The authors relax two restrictions: previous tokens need not be completely resolved before predicting future ones, and different tokens need not share the same noise level. The near future can be resolved sooner while the distant future retains alternatives. Independent per-token training makes this a choice of **sampling schedule**, rather than a separate training run for each pattern. [§1, Figures 1–2](https://arxiv.org/pdf/2407.01392v4#page=2)

## 2. Related Work and Preliminaries

There are two axes. Subscript $t$ denotes sequence time; superscript $k$ denotes diffusion noise. Thus $\mathbf x_t^{k_t}$ is the token at time $t$ corrupted to noise level $k_t$. The latent state $\mathbf z_t$ summarizes history.

### Bayesian filtering meets diffusion

Bayesian filtering updates a belief using a new observation. A learned latent need not estimate a physical coordinate; it can summarize information useful for predicting the future.

The forward diffusion process, preserving equation (2.1), is

$$
q(\mathbf x^k\mid\mathbf x^{k-1})
=\mathcal N\!\left(
\mathbf x^k;\sqrt{1-\beta_k}\mathbf x^{k-1},
\beta_k\mathbf I\right).
$$

With $\alpha_k=1-\beta_k$ and $\bar\alpha_k=\prod_{j=1}^k\alpha_j$, a noisy token can be sampled directly:

$$
\mathbf x_t^{k_t}
=\sqrt{\bar\alpha_{k_t}}\mathbf x_t^0
+\sqrt{1-\bar\alpha_{k_t}}\boldsymbol\epsilon_t,
\qquad
\boldsymbol\epsilon_t\sim\mathcal N(0,\mathbf I).
$$

The second expression uses normalized noise following [DDPM equation (4)](https://arxiv.org/pdf/2006.11239v2#page=2). Level zero is clean; sufficiently large $K$ makes the token approximately pure noise. At finite $K$, this need not mean exactly zero information. Mixed $t/k$ subscripts in the source's background exposition are normalized to the noise index $k$ here.

### Nearby approaches

| Approach | What it learns | Difference from Diffusion Forcing |
| --- | --- | --- |
| [DDPM](https://arxiv.org/pdf/2006.11239v2), §2–3 | Forward/reverse diffusion and a denoising objective | DF adds sequential memory and independent per-token noise. |
| [TimeGrad](https://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf), §3 | Conditional next-vector diffusion with an RNN summarizing clean history | DF trains with noisy history and supports guidance across several future tokens. |
| [Diffuser](https://proceedings.mlr.press/v162/janner22a/janner22a.pdf), §2–3.1 | Joint state–action trajectory generation with objective guidance | DF controls horizon and future uncertainty through a causal architecture and token-specific schedules. |
| [AR-Diffusion](https://arxiv.org/pdf/2305.09515v1), §3.1–3.2 | Token diffusion levels determined by sentence step and position | Different token noise levels precede DF; DF's distinction is independent training rather than a position-linked pattern. |

This is a conceptual comparison. TimeGrad appears in the forecasting table and Diffuser in the maze experiments; AR-Diffusion is not a text-generation performance baseline here. Appendix C also discusses Rolling Diffusion, whose complete paper was not independently reviewed for this article. Diffuser §3.1 also explicitly allows planning horizon changes through its fully convolutional input dimensionality. DF is therefore not presented as the first variable-horizon planner: its distinction concerns flexible noise/conditioning patterns and causal memory within one training framework.

## 3. Method

### 3.1 Noising as partial masking

Partial masking interpolates between preserving and hiding information. A lightly corrupted frame retains substantial structure; stronger corruption makes its original content harder to recover.

DF controls this degree separately from token position. Noise can be interpreted as a handle on how much an observation should be trusted. It is not automatically a calibrated uncertainty estimate. [§3.1, p. 4](https://arxiv.org/pdf/2407.01392v4#page=4)

![Original Figure 2: time/noise axes and the three training and sampling configurations](/assets/reviews/diffusion-forcing/overview.webp)

[Open Figure 2 at full size](/assets/reviews/diffusion-forcing/overview.webp)

**Reading Figure 2.** On the left, time is horizontal and noise is vertical. In DF's training row, differently colored tokens have independently chosen corruption levels. Teacher forcing uses clean history for a next-token prediction; full-sequence diffusion treats tokens at a common noise level. The lower sampling row shows DF maintaining temporal connections while denoising different columns at different rates. The broad horizontal bars represent joint full-sequence processing. Colors indicate noise levels, not success rates or confidence. [Figure 2, p. 3](https://arxiv.org/pdf/2407.01392v4#page=3)

### 3.2 Different noise levels for different tokens

The evaluated implementation uses a causal recurrent neural network. During training, $k_1,\ldots,k_T$ are independently sampled and each token is corrupted accordingly. Latents evolve as

$$
\mathbf z_t\sim p_\theta\!\left(
\mathbf z_t\mid\mathbf z_{t-1},
\mathbf x_t^{k_t},k_t\right).
$$

In the implementation, this update is deterministic; the distributional notation does not imply that a latent sample is drawn at every step. Level zero corresponds to a posterior update using an observation, while level $K$ approximates prior prediction without informative observation content.

The essential objective in equation (3.1) is

$$
\mathbb E_{\substack{k_{1:T},\,\mathbf x_{1:T},\,\boldsymbol\epsilon_{1:T}\\
\mathbf z_t\sim p_\theta(\mathbf z_t\mid
\mathbf z_{t-1},\mathbf x_t^{k_t},k_t)}}
\left[
\sum_{t=1}^{T}
\left\|
\boldsymbol\epsilon_t-
\boldsymbol\epsilon_\theta(
\mathbf z_{t-1},\mathbf x_t^{k_t},k_t)
\right\|^2
\right].
$$

The consequential change is **noisy conditioning history**, rather than MSE itself. Latents summarize that history, and gradients also flow through their recurrence. Algorithm 1 samples a sequence, samples independent levels, corrupts tokens, computes latent updates and predictions, and backpropagates the sequence loss. The source alternates between $[K]$ and $\{0,\ldots,K\}$. Its normalized-noise division cannot simply be evaluated at a clean endpoint; parameterization and endpoint handling need implementation-specific treatment. [§3.2, Algorithm 1, pp. 4–5](https://arxiv.org/pdf/2407.01392v4#page=4)

The experiments do not all use this illustrative $\boldsymbol\epsilon$ parameterization. Appendix D.3 uses $\mathbf v$ prediction for video and forecasting, and $\mathbf x_0$ prediction for planning and imitation. Video also uses fused SNR reweighting that incorporates information from history. These training details should be separated from the contribution of independent noise alone.

**Sampling selects a two-dimensional schedule.** Entry $\mathcal K_{m,t}$ specifies the desired noise level for token $t$ at sampling row $m$. Algorithm 2 traverses rows and updates latents from past to future within each row. The final row is clean.

For three tokens and even $K$, an illustrative pattern is

$$
\begin{aligned}
(K,K,K)
&\longrightarrow(0,K/2,K)\\
&\longrightarrow(0,0,K/2)\\
&\longrightarrow(0,0,0).
\end{aligned}
$$

These are **lists of noise levels**, not token values. They illustrate resolving the near future before the far future, rather than every intermediate step of an implemented DDIM sampler. Appendix D.7 specifies pyramid scheduling. [§3.2–3.3, D.7](https://arxiv.org/pdf/2407.01392v4#page=5)

Informal Theorem 3.1 connects training to a reweighted evidence lower bound and claims simultaneous modeling of noise patterns under appropriate conditions. The expressivity assumptions and appendix inconsistencies are addressed in the [theory article](/en/reviews/diffusion-forcing-theory/). The claim should not be read as exact subsequence prediction by a finite RNN.

### 3.3 New Capabilities in Sequence Generation

**Rollout stabilization.** Generated frames are not treated as flawless ground-truth history. The next generation uses a latent associated with slightly noisy history. In appendix B.4's RNN construction, a latent at $k_{\mathrm{small}}>0$ is carried forward. This puts imperfect inference history closer to the noisy histories seen during training, without resetting the recurrent state or sliding a fixed window.

**Keeping the future uncertain.** A zig-zag or pyramid schedule resolves nearby tokens earlier and leaves alternatives for distant tokens. It avoids committing the entire future too soon.

**Long-horizon guidance.** Future tokens depend on earlier ones in a causal model. Gradients of future objectives can therefore propagate through those dependencies to guide earlier tokens that are still being resolved. Backpropagation through time is different from supplying future observations as generation inputs. [§3.3, B.3–B.5](https://arxiv.org/pdf/2407.01392v4#page=6)

### 3.4 Flexible Sequential Decision Making

An action, reward, and next observation form a token:

$$
\mathbf x_t=
[\mathbf a_t,\mathbf r_t,\mathbf o_{t+1}]^\top.
$$

From $\mathbf z_{t-1}$, generate a plan $\hat{\mathbf x}_{t:t+H}$ and execute only $\hat{\mathbf a}_t$. Update the latent using **the reward and observation actually received from the environment**, rather than the predicted ones, then replan as in model predictive control.

A short look-ahead behaves more like a policy; a longer look-ahead supports planning with future objectives. The source denotes look-ahead by $H$ and writes the inclusive interval $t:t+H$. It does not report a measured latency reduction curve.

**Monte Carlo Guidance (MCG)** holds the current token as a shared condition, samples multiple futures, and averages their reward gradients. The following is **supplementary review notation**: $N$ is the future sample count and $\widehat G^{(i)}$ the estimated reward of sample $i$.

$$
\widehat{\mathbf g}
=\frac1N\sum_{i=1}^{N}
\nabla_{\mathbf x_t^k}\widehat G^{(i)}.
$$

Estimating expected reward over futures is different from selecting one fortunate future. Keeping distant tokens noisy supports conditional future variation while guiding the immediate choice. Interchanging differentiation and expectation requires regularity; model and sampling error are still present. Appendix B.5 explains why this conditional rollout operation is not straightforward in conventional simultaneous full-sequence denoising. It is not a universal impossibility theorem for every full-sequence architecture. [§3.4, B.3, B.5](https://arxiv.org/pdf/2407.01392v4#page=6)

## 4. Experiments

### 4.1 Video Prediction: Consistent, Stable Sequence Generation and Infinite Rollout

The datasets contain first-person Minecraft and DMLab navigation videos. DF, teacher-forced next-frame diffusion, and causal full-sequence diffusion use **the same RNN architecture**. This is not a comparison against every large noncausal video diffusion model.

![Original Figure 3: long video rollouts under three training paradigms in DMLab and Minecraft](/assets/reviews/diffusion-forcing/video-rollouts.webp)

[Open Figure 3 at full size](/assets/reviews/diffusion-forcing/video-rollouts.webp)

**Reading Figure 3.** DMLab is on the left and Minecraft on the right. Red borders identify real conditioning inputs; the remaining frames are generated. Labels 496, 500, 996, and 1000 are frame positions. Upper DF rows show examples retaining scene structure through long rollouts. Middle causal full-sequence rows show abrupt scene changes; lower teacher-forcing rows deteriorate into patterns or noise. The two sequence rows are examples, not means or error bars. Static frames cannot establish all aspects of temporal consistency. [Figure 3, §4.1, pp. 7–8](https://arxiv.org/pdf/2407.01392v4#page=7)

Appendix E.3 presents additional 180-frame examples and reports attempts reaching 2000 frames without blow-up. These are **observed finite rollouts**, not guarantees of infinite-length quality or accurate future states. Appendix F.1 specifies 128×128 Minecraft and 64×64 DMLab frames and approximately 10% of the data for training. Minecraft training clips have 72 frames. DMLab is described as 36 frames in E.3 captions and 48 in F.1, leaving an internal length discrepancy.

The Minecraft and DMLab models have 36M and 24M parameters. They use $K=1000$, 100 DDIM sampling steps, and frame stacks of 8 and 4. Video training uses eight A100s, batch 128, and 50K steps, with convergence around 40K steps after roughly 12 hours. A quantitative FVD comparison is not the principal evidence supplied here. [D.2, D.6, D.10, E.3, F.1](https://arxiv.org/pdf/2407.01392v4#page=24)

### 4.2 Diffusion Planning: MCG, Causal Uncertainty, Flexible Horizon Control

On D4RL Maze2D U-Maze, Medium, and Large, one model is trained per maze using random-walk logs. Observations contain 2D position and velocity; actions are 2D acceleration. Maze2D uses fixed goals, while Multi2D randomizes goals.

![Upper diagram of original Table 1: denoising maze plans with DF and Diffuser](/assets/reviews/diffusion-forcing/maze-planning.webp)

[Open the planning diagram at full size](/assets/reviews/diffusion-forcing/maze-planning.webp)

**Reading the planning diagram.** Each maze shows DF above Diffuser. Horizontal progress is **denoising**, not environmental execution time. The start-to-end color scale marks position within a plan. DF resolves nearby portions while retaining distant alternatives. Scattered points are unresolved sampling states, not all the positions visited by an executing robot. Read this visualization with the quantitative table.

All comparison methods in Table 1 are retained below, split into two tables. These are reported benchmark scores, **not success percentages**, and may exceed 100. The original $\pm$ values are preserved; the table does not clearly specify its run count or the meaning of the error term.

| Environment | MPPI | CQL | IQL | Diffuser\* | Diffuser generated actions | DF without MCG | DF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U-Maze | 33.2 | 5.7 | 47.4 | $113.9\pm3.1$ | $6.3\pm2.1$ | $110.1\pm3.9$ | $116.7\pm2.0$ |
| Medium | 10.2 | 5.0 | 34.9 | $121.5\pm2.7$ | $13.5\pm2.3$ | $136.1\pm10.2$ | $149.4\pm7.5$ |
| Large | 5.1 | 12.5 | 58.6 | $123.0\pm6.4$ | $6.3\pm2.1$ | $142.8\pm5.6$ | $159.0\pm2.7$ |
| Reported average | 16.2 | 7.7 | 47.0 | 119.5 | 8.7 | 129.67 | 141.7 |

| Multi2D environment | MPPI | CQL | IQL | Diffuser\* | Diffuser generated actions | DF without MCG | DF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U-Maze | 41.2 | — | 24.8 | $128.9\pm1.8$ | $32.8\pm1.7$ | $107.7\pm4.9$ | $119.1\pm4.0$ |
| Medium | 15.4 | — | 12.1 | $127.2\pm3.4$ | $22.0\pm2.7$ | $145.6\pm6.5$ | $152.3\pm9.9$ |
| Large | 8.0 | — | 13.9 | $132.1\pm5.8$ | $6.9\pm1.7$ | $129.8\pm1.5$ | $167.1\pm2.7$ |
| Reported average | 21.5 | — | 16.9 | 129.4 | 20.6 | 127.7 | 146.2 |

The asterisk matters. <strong>Diffuser*</strong> uses a handcrafted PD controller to turn predicted state paths into actions. Directly executing generated actions performs much worse. DF executes its generated actions. This supports attention to consistency between plans and actions, while retaining the fact that controller conditions differ. [Table 1, §4.2, p. 8](https://arxiv.org/pdf/2407.01392v4#page=8)

**Why a score can exceed 100.** D4RL normalization measures return relative to environment-specific reference values. This review notation shows the 100-point scale:

$$
S(R)=100\,\frac{R-R_{\min}}{R_{\max}-R_{\min}}.
$$

If $R>R_{\max}$, the score exceeds 100. The reference maximum is not a theoretical ceiling. This definition does not recover the experiment’s raw returns or success counts. [Official D4RL get_normalized_score implementation](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py)

MCG improves the **mean score** in all six settings, without establishing statistical significance separately for each. Single-task averages are 141.7 for DF and 119.5 for Diffuser\*; multi-task averages are 146.2 and 129.4. **On Multi2D U-Maze, DF scores 119.1 versus Diffuser\*'s 128.9.** The prose claim of superiority in all six environments therefore does not match the complete table.

Appendix D.8 warns that reward is an imperfect planning metric: a slow path spending longer near the goal may score highly. First hitting time would be useful alongside reward, but a complete numerical comparison is not supplied.

The maze model has 4.33M parameters, uses a frame stack of 10 and 50 DDIM sampling steps, and can be trained on an 11GB 2080Ti. The MCG future sample count and precise inference-budget matching are insufficiently reported. The scores should not be interpreted as algorithmic gains at demonstrated equal compute cost. [D.2, D.6, D.8, D.10, F.2](https://arxiv.org/pdf/2407.01392v4#page=25)

### 4.3 Controllable Sequential Compositional Generation

Two families of diagonal trajectories connect opposite corners of a 2D square. Full memory reproduces the training distribution's X shape. Shorter plans with discarded history can join portions into a V shape.

The experiment illustrates changing **memory and sampling scope** without retraining. It does not establish that arbitrary real-world sub-trajectories can be safely stitched together. Discarding history can remove both restrictions and useful long-context constraints. [§4.3, E.2, Figure 7](https://arxiv.org/pdf/2407.01392v4#page=27)

### 4.4 Robotics: Long Horizon Imitation Learning and Robust Visuomotor Control

A Franka robot swaps an apple and orange using a third empty slot, with two randomized initial configurations. The same intermediate image can require a different next move depending on the starting arrangement, making memory necessary.

![Original Figure 4: identical intermediate arrangements with different goals, and generated videos from two cameras](/assets/reviews/diffusion-forcing/robot-memory.webp)

[Open Figure 4 at full size](/assets/reviews/diffusion-forcing/robot-memory.webp)

**Reading Figure 4.** Cases A and B on the left have the same intermediate arrangement but different final goals below. The next arrow cannot be selected from the current image alone. On the right are conditioning images and generated wrist/external-camera frames; 120–480 label frame positions. Plausible generated video and successful executed control are different evidence. [Figure 4, §4.4, p. 9](https://arxiv.org/pdf/2407.01392v4#page=9)

Training uses 150 expert demonstrations, two cameras, and 6-DoF actions. Each demonstration has 500–600 frames/actions, reduced to 40 video tokens with 15 actions bundled per token.

| Condition or comparison | Reported success | Scope |
| --- | --- | --- |
| DF, ordinary observations | 80% | One memory-dependent fruit-swap task |
| DF, corrupted observations | 76% | Distractors/camera occlusion, with corruption designated as noisy input |
| Next-frame diffusion baseline, corrupted observations | 48% | Treats corrupted observations as ground truth |
| Memoryless Diffusion Policy, ordinary task | Reported to fail | No exact success percentage in the main text |

The change from 80 to 76 is **four percentage points**. Evaluation counts and confidence intervals are unclear; 150 training demonstrations must not be substituted for test trials. The system allows $k>0$ to designate corruption, rather than demonstrating an automatically calibrated corruption detector. Appendix E.5 describes failure when initial object placement information is missed.

Generating video with the same model motivates unlabeled-video pretraining. It is not an experiment showing gains from large-scale unlabeled pretraining. [§4.4, E.5, F.3](https://arxiv.org/pdf/2407.01392v4#page=33)

### 4.5 Time Series Forecasting: A General-purpose Sequence Model

Appendices E.1 and F.4 evaluate six GluonTS datasets. Context length equals prediction length: 30 for Exchange and Wikipedia, 24 otherwise. Dimensions are 8, 137, 370, 963, 1214, and 2000.

The metric is summed continuous ranked probability score, $\operatorname{CRPS}_{\mathrm{sum}}$, with **lower values better**. It compares the predicted distribution of the feature sum with the observed sum, rather than completely evaluating each feature or the full dependency structure. Estimation uses 100 sampled trajectories and 19 quantiles. DF reports means and standard deviations over five seeds.

For predictive CDF $F$ and observation $y$, basic CRPS is

$$
\operatorname{CRPS}(F,y)=\int_{\mathbb R}\left(F(s)-\mathbf1\{y\leq s\}\right)^2\,ds.
$$

Here $F,y,s$ are explanatory metric notation. For $\mathrm{CRPS}_{\mathrm{sum}}$, first sum dimensions within each sampled trajectory and the observations, then score and average over prediction times. This differs from summing per-dimension CRPS: different feature errors can have the same aggregate. Basic CRPS has the observation’s units, so comparisons retain preprocessing and normalization. [TimeGrad §4.1, pp. 4–5](https://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf#page=5)

This table preserves Table 2's principal modern comparisons and the NKF row needed to assess Electricity and Wikipedia. Other statistical and copula rows are available in the original and were included in the review's full-table ranking checks.

| Method | Exchange | Solar | Electricity | Traffic | Taxi | Wikipedia |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer-MAF | $0.005\pm0.003$ | $0.301\pm0.014$ | $0.021\pm0.000$ | $0.056\pm0.001$ | $0.179\pm0.002$ | $0.063\pm0.003$ |
| TimeGrad | $0.006\pm0.001$ | $0.287\pm0.020$ | $0.021\pm0.001$ | $0.044\pm0.006$ | $0.114\pm0.020$ | $0.049\pm0.002$ |
| ScoreGrad sub-VP SDE | $0.006\pm0.001$ | $0.256\pm0.015$ | $0.019\pm0.001$ | $0.041\pm0.004$ | $0.101\pm0.004$ | $0.043\pm0.002$ |
| NKF | — | $0.320\pm0.020$ | $0.016\pm0.001$ | $0.100\pm0.002$ | — | $0.071\pm0.002$ |
| DF | $0.003\pm0.001$ | $0.289\pm0.002$ | $0.023\pm0.001$ | $0.040\pm0.004$ | $0.075\pm0.002$ | $0.085\pm0.007$ |

**DF has the lowest mean on Exchange, Traffic, and Taxi, but not Solar, Electricity, or Wikipedia.** On Electricity, NKF's 0.016 is lower even than the source's emphasized ScoreGrad value of 0.019. Four models have lower Wikipedia means than DF, placing it fifth across the full table rather than the fourth claimed in the prose. Overlapping error ranges and different comparison-model conditions prevent mean rankings from replacing significance tests. [Table 2, E.1, p. 27](https://arxiv.org/pdf/2407.01392v4#page=27)

Validation subsequences are drawn randomly from the training series. Early stopping follows six epochs without improvement, with 100 batches per epoch and batch 32. The results support applicability across forecasting domains, rather than superiority over every existing forecasting model.

## 5. Discussion

Independent per-token noise expands the **conditioning, memory, horizon, and guidance choices available at inference**. Video stabilization, MCG planning, and memory-based robot control exploit different aspects of this freedom.

The authors identify the current RNN implementation, higher-resolution and more complex distributions, and untested internet-scale behavior as limitations. Appendix B.1's transformer extension is a discussion, not a replacement for the evaluated RNNs by a demonstrated large-scale system. [§5, p. 10](https://arxiv.org/pdf/2407.01392v4#page=10)

## Review interpretation and open questions

The interesting contribution is the freedom to change sampling behavior without building a new model for every objective. Noise-as-uncertainty intuition, confidence calibration, and accurate long-horizon state prediction remain distinct.

Useful follow-up checks would separate independent noise, stabilization, and fused SNR under matched budgets; measure MCG sample-count versus success/latency; evaluate reward together with first hitting time; and separate explicitly designated observation corruption from automatically detected corruption. These are review proposals, not executed experiments.

## Sources and reading scope

- Original: Chen, Marti Monso, Du, Simchowitz, Tedrake, Sitzmann, *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion*, [arXiv:2407.01392v4](https://arxiv.org/abs/2407.01392v4). First submitted July 1, 2024; v4 revised December 10, 2024; reviewed October 14, 2026.
- First reading: abstract, introduction, discussion, section map, Figures 1–4/Table 1 captions, and appendix A's overview/statements; an independent orientation was saved before detailed reading.
- Second reading: text, equations, tables, and algorithms in sections 1–5 and appendices A–F, checked against author LaTeX and PDF; key method/theory/planning pages were rendered. References were inspected and selected original sections read.
- Additional original sources: DDPM v2 §2–3.4, TimeGrad §2–3.1, Diffuser §2–3.1, AR-Diffusion v1 §3.1–3.2, and Chan tutorial v1 §2.6–2.7. This was not a full independent review of those papers or a current literature survey. HINT's theorem and application assumptions cited in B.4 were not independently checked and are not used here to certify stability.
- Mapping: sections 1–5 and experiments 4.1–4.5 remain in source order; essential B/D/E/F material is explained alongside its corresponding main section. Appendix A is fully addressed in [part 2](/en/reviews/diffusion-forcing-theory/). Peripheral related work, repeated additional videos, and the entire bibliography are compressed; key comparisons, contrary results, and compute costs are retained.
- Images: Figures 2–4 and Table 1's upper diagram come from the [authors' v4 source](https://arxiv.org/src/2407.01392v4). Original PDF graphics were rasterized and WebP-encoded, preserving content, colors, arrows, and notation while changing resolution/encoding. Credit belongs to the paper's listed authors; reuse follows [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with source and change attribution.
- Verification: transcription, means, rankings, units, and local ELBO calculations were checked. **Not executed:** model/GPU/robot reproduction, public research-code execution, full quantitative video evaluation, significance tests from raw results, actual latency/memory measurement, and formal verification of the complete theorems.

- Recheck on October 22, 2026: this revision reuses the recorded two readings and source mapping, adding the metric, assumption, and example definitions above. It is not counted as two new full-paper readings. New model, GPU, robot reproduction, and complete formal verification were not executed.

- Additional source scope: the return expression in official D4RL get_normalized_score (read only, not executed), and TimeGrad §4.1 pp. 4–5 for CRPS and summation. Experiment-specific raw returns and preprocessing were not newly verified.
