---
title: "[1/2] Diffusion Forcing: What Changes When Time and Noise Can Be Controlled Separately?"
description: "Follow independent per-token noise from the training objective to long video rollouts, Monte Carlo Guidance, maze planning, and robotic memory in the authors' section order."
date: "2026-10-14"
publishedAt: "2026-10-14T13:58:28+09:00"
updatedAt: "2026-10-23T15:26:21+00:00"
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

Diffusion Forcing (DF) trains on independently sampled per-token noise levels, so generation horizon, memory, and future guidance can be chosen at sampling time without retraining. In Table 1's Maze2D Large (fixed goal), DF's episode reward is 159.0±2.7, versus 142.8±5.6 for the same model without Monte Carlo Guidance (MCG). [Table 1](https://arxiv.org/pdf/2407.01392v4#page=8) **Review interpretation.** The MCG ablation changes only sampling and raises the score in all six maze environments, making it the most direct support for the planning claims; long-horizon video stability is shown only through examples without a quantitative metric, and the robot memory and corrupted-observation results only as success rates without trial counts, so the remaining evidence for "combining the strengths of both paradigms" is demonstration-level.

This article covers v4's main sections 1–5 and the experimental and implementation appendices B–F in source order; appendix A's ELBO argument and the printed errors in its KL signs, transition directions, and weights are covered in [\[2/2\] Theory and proof walkthrough](/en/reviews/diffusion-forcing-theory/).

## 1. Introduction

A **token** can be a video frame, a multivariate observation, or a robot action–observation bundle. It need not be a word.

**Teacher forcing** trains a model to predict the next token from ground-truth history. The authors name two limitations of next-token models trained this way: there is no mechanism for guiding sampling toward an objective, and on continuous data the model consumes its own outputs at inference, so small errors compound and the model diverges past the training horizon.

**Full-sequence diffusion** concatenates a fixed number of tokens and noises and denoises them all at a shared noise level. It supports **guidance** toward desirable outcomes, but its non-causal architecture makes variable-length and subsequence generation difficult. The authors also report that naively training a next-token model with full-sequence diffusion generates poorly.

DF relaxes two restrictions at once: earlier tokens need not be fully resolved before predicting later ones, and all tokens need not share one noise level. The near future can be resolved sooner while the distant future keeps alternatives. Because per-token noise is sampled independently during training, this becomes a choice of **sampling schedule** rather than a retraining decision. [§1, Figures 1–2](https://arxiv.org/pdf/2407.01392v4#page=2)

## 2. Related Work and Preliminaries

There are two axes. Subscript $t$ denotes sequence time; superscript $k$ denotes diffusion noise. Thus $\mathbf x_t^{k_t}$ is the token at time $t$ corrupted to noise level $k_t$. The latent state $\mathbf z_t$ summarizes history.

### Bayesian filtering meets diffusion

Bayesian filtering updates latent states recursively with a prior model $p(\mathbf z_{t+1}\mid\mathbf z_t)$, an observation model $p(\mathbf x_t\mid\mathbf z_t)$, and a posterior model $p(\mathbf z_{t+1}\mid\mathbf z_t,\mathbf x_{t+1})$ that incorporates a new observation. When trained end to end with neural networks, the latent is not an estimate of a physical quantity but a summary of the past that is useful for predicting future observations.

The forward diffusion process is equation (2.1):

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

The second expression is the closed form in [DDPM equation (4)](https://arxiv.org/pdf/2006.11239v2#page=2). Level zero is clean, and $\mathbf x_t^K$ is approximately pure noise. The reverse process is a Gaussian Markov chain whose mean is predicted by a network, trained in the standard way with a least-squares noise-prediction objective $\boldsymbol\epsilon_\theta$. For guidance, the source takes classifier guidance as its reference: the gradient of the log of a classifier $c(y\mid\mathbf x^k)$ for a desired attribute $y$ adjusts the sampling direction.

Section 2 notes that next-token models cannot accept multi-step guidance, because earlier frames must be fully determined before future frames are sampled. Among diffusion sequence models it lists Diffusion-LM for guided text control, video diffusion extended with a sliding window, Diffuser as a planner for offline reinforcement learning, and TimeGrad trained with teacher forcing; the closest work is AR-Diffusion, whose noise level depends linearly on position along the time axis. Appendix C adds that Rolling Diffusion shares the same limitation, since its position-linked noise levels must be reused unchanged at sampling time.

### Review supplement: nearby approaches compared

| Approach | What it learns | Difference from Diffusion Forcing |
| --- | --- | --- |
| [DDPM](https://arxiv.org/pdf/2006.11239v2), §2–3 | Forward/reverse diffusion and a denoising objective | DF adds sequential memory and independent per-token noise. |
| [TimeGrad](https://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf), §3 | Next-vector diffusion conditioned on a recurrent neural network (RNN) state summarizing clean history | DF trains with noisy history and supports guidance across several future tokens. |
| [Diffuser](https://proceedings.mlr.press/v162/janner22a/janner22a.pdf), §2–3.1 | Joint state–action trajectory generation with objective guidance | DF controls horizon and future uncertainty through a causal architecture and token-specific schedules. |
| [AR-Diffusion](https://arxiv.org/pdf/2305.09515v1), §3.1–3.2 | Token diffusion levels determined by sentence step and position | Different token noise levels precede DF; DF's distinction is independent training rather than a position-linked pattern. |

TimeGrad appears in the forecasting table and Diffuser in the maze experiments.

## 3. Method

### 3.1 Noising as partial masking

Partial masking interpolates between preserving and hiding information. The authors view teacher forcing as masking along the time axis and forward diffusion as partial masking along the noise axis: zero noise leaves a token unmasked, and level $K$ masks it completely.

DF unifies the two axes. Each token in a sequence can receive its own noise level $k_t$, which is the degree of partial masking applied to that token. [§3.1, p. 4](https://arxiv.org/pdf/2407.01392v4#page=4)

![Original Figure 2: time/noise axes and the three training and sampling configurations](/assets/reviews/diffusion-forcing/overview.webp)

[Open Figure 2 at full size](/assets/reviews/diffusion-forcing/overview.webp)

**Reading Figure 2.** On the left, time is horizontal and noise is vertical. In DF's training row, differently colored tokens have independently chosen corruption levels. Teacher forcing uses clean history for a next-token prediction; full-sequence diffusion treats tokens at a common noise level. The lower sampling row shows DF maintaining temporal connections while denoising different columns at different rates. The broad horizontal bars represent joint full-sequence processing, and colors indicate noise levels. [Figure 2, p. 3](https://arxiv.org/pdf/2407.01392v4#page=3)

### 3.2 Different noise levels for different tokens

The evaluated implementation is a causal RNN. During training, $k_1,\ldots,k_T$ are independently sampled and each token is corrupted accordingly. Latents evolve as

$$
\mathbf z_t\sim p_\theta\!\left(
\mathbf z_t\mid\mathbf z_{t-1},
\mathbf x_t^{k_t},k_t\right).
$$

In the implementation this update is deterministic; the authors describe $\mathbf z_t$ as representing a distribution over beliefs rather than a sample from it. Level zero corresponds to the Bayes-filter posterior update with an observation, and level $K$ to prior prediction without observation content.

The training objective is equation (3.1):

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

The loss itself is standard diffusion MSE; what changes is that **the conditioning history is noisy too**. Latents summarize that history, and gradients also flow through their recurrence. Algorithm 1 samples a sequence, samples independent noise levels, corrupts tokens, computes latent updates and predictions, and backpropagates the sequence loss. The source alternates between $[K]$ and $\{0,\ldots,K\}$ for the set of noise levels. [§3.2, Algorithm 1, pp. 4–5](https://arxiv.org/pdf/2407.01392v4#page=4)

Not every experiment uses this $\boldsymbol\epsilon$ prediction. According to appendix D.3, video and forecasting use $\mathbf v$ prediction, and planning and imitation use $\mathbf x_0$ prediction. Video also uses **fused SNR reweighting**, a signal-to-noise-ratio weighting re-derived to account for the signal carried by the noisy history. The authors report that it greatly accelerated convergence in video prediction, and that they did not use it in non-image domains because they saw no benefit there (D.1).

**Sampling selects a two-dimensional schedule.** Entry $\mathcal K_{m,t}$ is the noise level that token $t$ should reach at sampling row $m$. Algorithm 2 moves down the rows and, within each row, updates latents from past to future while denoising. The final row is clean. Because training covers all combinations of noise levels, the authors' point is that changing $\mathcal K$ alone yields different behaviors without retraining.

Informal Theorem 3.1 states that training optimizes a reweighted evidence lower bound (ELBO) and, under appropriate conditions, simultaneously maximizes a likelihood lower bound for every sequence of noise levels. The authors add that the special case $k_t\in\{0,K\}$ means DF learns conditionals with arbitrary earlier tokens masked out, that is, the distribution of every subsequence of the training set. The conditions behind this claim and the printed errors in the appendix proof are covered in the [theory article](/en/reviews/diffusion-forcing-theory/).

### 3.3 New Capabilities in Sequence Generation

**Stabilizing autoregressive generation.** For high-dimensional continuous sequences such as video, autoregressive architectures tend to diverge past the training horizon. DF does not treat generated frames as flawless ground-truth history; it continues from the latent associated with a token at a small noise level $0<k_{\mathrm{small}}\ll K$ (appendix B.4). This slightly corrupted history is in distribution for training, so inference errors are absorbed by noise the model has already seen. The recurrent state is never reset, and no sliding window is used.

**Keeping the future uncertain.** The source explains this with three tokens. For even $K$, the noise levels evolve as

$$
\begin{aligned}
(K,K,K)
&\longrightarrow(0,K/2,K)\\
&\longrightarrow(0,0,K/2)\\
&\longrightarrow(0,0,0).
\end{aligned}
$$

These are **lists of noise levels**, not token values. Interpreting noise level as uncertainty, the authors describe this "zig-zag" sampling as keeping the near future more certain than the far future. Appendix D.7 generalizes the idea into a scheduling matrix for planning, called pyramid scheduling.

**Long-horizon guidance.** In a causal model, future tokens depend on earlier ones, so gradients of future objectives propagate backward along those dependencies. The distinction the authors emphasize is that future tokens can be denoised without fully resolving the past, so those gradients guide the sampling of past tokens that are still being resolved. [§3.3, B.3–B.5, D.7](https://arxiv.org/pdf/2407.01392v4#page=5)

### 3.4 Flexible Sequential Decision Making

An action, reward, and next observation form a token:

$$
\mathbf x_t=
[\mathbf a_t,\mathbf r_t,\mathbf o_{t+1}]^\top.
$$

From $\mathbf z_{t-1}$, generate a plan $\hat{\mathbf x}_{t:t+H}$ and execute only $\hat{\mathbf a}_t$. Update the latent using **the reward and observation actually received from the environment**, rather than the predicted ones, then replan as in model predictive control (MPC). The source calls $H$ the look-ahead window and writes the inclusive interval $t:t+H$.

**Flexible planning horizon.** The authors argue that because DF selects actions one at a time, it can serve tasks of variable horizon, and that without retraining $H$ can be shortened for a low-latency policy or lengthened for a long-horizon planner. Their contrast is that Diffuser, which generates at full-trajectory length, cannot do the former, while diffusion policies with short fixed look-aheads cannot do the latter. The guidance reward can likewise be a dense reward over the whole trajectory, a reward within the look-ahead, or a sparse goal reward $-\|\mathbf o_T-\mathbf g\|^2$. [§3.4, p. 6](https://arxiv.org/pdf/2407.01392v4#page=6) **Review interpretation.** Diffuser §3.1 itself states that its fully convolutional architecture can change planning horizon by changing input length, so DF's distinction lies less in horizon control than in selecting actions sequentially and updating memory with real observations. The source does not measure how much latency is reduced.

**MCG** holds the current token $\mathbf x_t^k$ fixed, samples multiple futures, and averages their guidance gradients. The authors describe this, in the spirit of shooting methods such as model predictive path integral (MPPI) control, as guiding the current token by the expected reward over the distribution of futures rather than by one particular future. The following is **supplementary review notation**: $N$ is the number of future samples and $\widehat G^{(i)}$ the estimated reward of sample $i$.

$$
\widehat{\mathbf g}
=\frac1N\sum_{i=1}^{N}
\nabla_{\mathbf x_t^k}\widehat G^{(i)}.
$$

The effect is said to grow when MCG is combined with the zig-zag schedule, which keeps distant tokens at higher noise levels. Appendix B.5 explains that full-sequence diffusion denoises all tokens in tandem, leaving no obvious source of randomness for a Monte Carlo estimate at a fixed noise level, and §4.2 says this was "formally derive[d]". [§3.4, B.3, B.5](https://arxiv.org/pdf/2407.01392v4#page=6) **Review interpretation.** B.5 is an argument about architecture rather than a formal derivation, and the authors themselves leave open simulating future rollouts with the replacement trick of B.6.

## 4. Experiments

### 4.1 Video Prediction: Consistent, Stable Sequence Generation and Infinite Rollout

A convolutional RNN implementation is trained on first-person Minecraft and DMLab navigation videos and rolled out autoregressively with the stabilization above. The baselines, teacher-forced next-frame diffusion and causal full-sequence diffusion, use **the same RNN architecture**. The authors report that DF rolls out stably far beyond its training horizon (for example, 1000 frames) while both baselines diverge quickly, and that within the training horizon full-sequence diffusion shows abrupt jumps between frames.

![Original Figure 3: long video rollouts under three training paradigms in DMLab and Minecraft](/assets/reviews/diffusion-forcing/video-rollouts.webp)

[Open Figure 3 at full size](/assets/reviews/diffusion-forcing/video-rollouts.webp)

**Reading Figure 3.** DMLab is on the left and Minecraft on the right. Red borders identify real conditioning inputs; the remaining frames are generated. Labels 496, 500, 996, and 1000 are frame positions. Upper DF rows retain scene structure through long rollouts; middle causal full-sequence rows show abrupt scene changes; lower teacher-forcing rows deteriorate into patterns or noise. The two sequence rows are separate examples. [Figure 3, §4.1, pp. 7–8](https://arxiv.org/pdf/2407.01392v4#page=7)

Appendix E.3 presents non-cherry-picked 180-frame examples and reports rolling out to 2000 frames without the model blowing up. The Minecraft agent occasionally gets stuck with the whole screen showing a dirt block, which the authors attribute to the dataset. Appendix F.1 specifies 128×128 Minecraft and 64×64 DMLab frames, with about 10% of the data used for training. Minecraft training clips have 72 frames; for DMLab, E.3 states 36 frames and F.1 states 48.

The Minecraft and DMLab models have 36M and 24M parameters. They use $K=1000$, 100 sampling steps of DDIM (denoising diffusion implicit models), and frame stacks of 8 and 4. Video training uses eight A100s, batch 128, and 50K steps, converging around 40K steps after roughly 12 hours. The video experiments report no quantitative metric; the evidence is the examples in Figure 3 and E.3. [D.2, D.6, D.10, E.3, F.1](https://arxiv.org/pdf/2407.01392v4#page=24)

### 4.2 Diffusion Planning: MCG, Causal Uncertainty, Flexible Horizon Control

On Maze2D U-Maze, Medium, and Large from the offline reinforcement learning benchmark D4RL, one model is trained per maze from random-walk data. Observations are 4D vectors of 2D position and velocity, actions are 2D acceleration, and the reward is sparse: 1 within a radius of 0.5 around the goal and 0 otherwise. Maze2D uses fixed goals, while Multi2D randomizes goals as well.

![Upper diagram of original Table 1: denoising maze plans with DF and Diffuser](/assets/reviews/diffusion-forcing/maze-planning.webp)

[Open the planning diagram at full size](/assets/reviews/diffusion-forcing/maze-planning.webp)

**Reading the planning diagram.** Each maze shows DF above Diffuser. Horizontal progress is **denoising**, not environmental execution time. The start-to-end color scale marks position within a plan. DF resolves nearby portions first and keeps distant portions more uncertain, whereas Diffuser treats every portion at the same noise level (source caption). Scattered points are sampling states that are not yet resolved.

All comparison methods in Table 1 are reproduced below, split into two tables. The baselines are MPPI, the offline reinforcement learning methods conservative Q-learning (CQL) and implicit Q-learning (IQL), and Diffuser. The values are what the source calls episode reward; the number of runs and the meaning of $\pm$ are not reported.

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

**Why a score can exceed 100.** The source calls these values "episode reward" (D.8) and "average reward" (Table 1 caption) without stating whether they are normalized. **Review interpretation.** The scale of the values above 100 matches the normalized score defined by D4RL. The D4RL implementation ([get_normalized_score](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py)) computes the following from environment-specific reference returns $R_{\min},R_{\max}$; because the reference maximum is not a theoretical ceiling, a return above it scores above 100.

$$
S(R)=100\,\frac{R-R_{\min}}{R_{\max}-R_{\min}}.
$$

The caption's statement that DF achieves the "highest average reward" matches the table: single-task averages are 141.7 for DF and 119.5 for Diffuser\*, and multi-task averages are 146.2 and 129.4. Section 4.2's claim that DF "outperforms Diffuser and all baselines across all 6 environments" does not match it, however: on Multi2D U-Maze, **DF scores 119.1 to Diffuser\*'s 128.9**. Adding MCG raises DF's score in all six environments (for example, Multi2D Large 129.8→167.1), and the authors note that DF remains competitive without it.

<strong>Diffuser*</strong> discards its generated actions and follows the generated state path with a handcrafted PD (proportional–derivative) controller. Executing the generated actions directly drops the averages to 8.7 and 20.6. The authors take this as evidence that Diffuser's generated states and actions are not causally consistent, and stress that DF, executing its own generated actions, beats even Diffuser\* with its PD controller. The two methods also use different guidance objectives. Both discard the dataset reward and plan toward the goal, but DF guides with the goal distance summed over all time steps so that no arrival time has to be specified. The authors tried this reward with Diffuser, but it did not work even with substantial tuning (D.5). [Table 1, §4.2, p. 8](https://arxiv.org/pdf/2407.01392v4#page=8)

Appendix D.8 warns that episode reward is not a good metric. The dataset contains no trajectories that stay at the goal, so agents tend to leave after reaching it (D.5), and Diffuser may have gained an unfair advantage by generating slow plans that linger near the goal. The authors recommend first hitting time as an alternative and write that DF excels at it, but give no numbers.

**Flexible horizon.** In fixed-horizon tasks the remaining planning horizon must shrink as the task progresses. The authors argue that DF supports this by design, whereas full-sequence models such as Diffuser perform poorly even with tweaks. Appendix B.6 gives the reasons: "conditioning by replacement", which fixes already generated tokens and updates only the rest, still denoises the whole sequence even with one step left, and is mathematically unprincipled, so the generated sequence can become inconsistent. Table 1 has no separate numbers for this claim.

The maze model has 4.33M parameters, a frame stack of 10, and 50 DDIM steps, and can be trained on one 11GB 2080Ti. The number of MCG future samples and the inference cost are not reported. [D.2, D.6, D.8, D.10, F.2](https://arxiv.org/pdf/2407.01392v4#page=25)

### 4.3 Controllable Sequential Compositional Generation

The training data are trajectories across a 2D square from one corner to the opposite corner, forming an X. Keeping full memory reproduces the training distribution's X shape; generating short plans with MPC and no memory stitches parts of different trajectories into a V shape.

The authors' point is that **only the sampling scheme changes**, without retraining, to control how compositional the generation is. The figure in appendix E.2 describes discarding previous states as recovering Markovian dynamics. [§4.3, E.2, Figure 7](https://arxiv.org/pdf/2407.01392v4#page=27) **Review interpretation.** Discarding long context removes the long-range constraints the data obeyed along with enabling new combinations, so this control is useful where such constraints can be dropped.

### 4.4 Robotics: Long Horizon Imitation Learning and Robust Visuomotor Control

A Franka robot swaps an apple and an orange using a third empty slot. The initial arrangement is one of two, chosen at random. In an intermediate state with one fruit in the third slot, the next move cannot be determined from the current image alone, so the robot must remember the initial arrangement.

![Original Figure 4: identical intermediate arrangements with different goals, and generated videos from two cameras](/assets/reviews/diffusion-forcing/robot-memory.webp)

[Open Figure 4 at full size](/assets/reviews/diffusion-forcing/robot-memory.webp)

**Reading Figure 4.** Cases A and B on the left have the same intermediate arrangement but different final goals below. The next arrow cannot be selected from the current image alone. On the right are conditioning images and generated wrist/external-camera frames; 120–480 label frame positions. Plausible generated video and successful executed control are different evidence. [Figure 4, §4.4, p. 9](https://arxiv.org/pdf/2407.01392v4#page=9)

Training uses 150 expert demonstrations collected by VR teleoperation (half for each initial arrangement), two camera views, and 6-DoF actions. Each demonstration has 500–600 frames and actions, downsampled to 40 video tokens with 15 actions bundled per token.

| Condition or comparison | Reported success | Condition |
| --- | --- | --- |
| DF, ordinary observations | 80% | Memory-dependent fruit-swap task |
| DF, corrupted observations | 76% | Visual distractors and full camera occlusion; the distractor bag was absent from the training data (zero-shot), and corrupted observations are designated with $k>0$ |
| Next-frame diffusion baseline, corrupted observations | 48% | Treats corrupted observations as ground truth |
| Memoryless Diffusion Policy, ordinary task | Reported to fail | No success rate given |

The change from 80 to 76 is **four percentage points**, and the source reports neither trial counts nor confidence intervals. Corruption is designated by a person with $k>0$, after which DF relies heavily on its prior model to predict actions. According to appendix E.5, the data were collected without the bag, and the corrupted-observation test threw the bag onto the table zero-shot. The typical failure is the robot no longer reacting to visual cues about the randomized object locations; the authors did not observe the robot acting wildly because of the distractor.

The authors write that the same model can also generate a video of the task from the first frame alone, unifying diffusion policy/imitation learning with video generative modeling and paving the way to pre-training on unlabeled video. [§4.4, E.5, F.3](https://arxiv.org/pdf/2407.01392v4#page=33)

### 4.5 Time Series Forecasting: A General-purpose Sequence Model

Appendices E.1 and F.4 use six GluonTS datasets. Context length equals prediction length: 30 for Exchange and Wikipedia, 24 otherwise. Dimensions are 8, 137, 370, 963, 1214, and 2000, and all splits are normalized by the per-feature mean and standard deviation of the training split. Validation sequences are random subsequences of the training series, and training stops early after six epochs without improvement. Each epoch has 100 batches of 32, and all datasets share one architecture and set of hyperparameters.

The metric is the summed continuous ranked probability score, $\operatorname{CRPS}_{\mathrm{sum}}$, with **lower values better**. For a predictive cumulative distribution function (CDF) $F$ and an observation $x$, CRPS is

$$
\operatorname{CRPS}(F,x)=\int_{\mathbb R}\left(F(z)-\mathbb I\{x\leq z\}\right)^2\,dz.
$$

For $\operatorname{CRPS}_{\mathrm{sum}}$, the 100 sampled trajectories and the ground truth are each summed over the feature dimension, the CDF of the sum is estimated from 19 quantiles, and the score is averaged over the prediction window. Unlike summing per-dimension CRPS, this cannot distinguish different per-feature errors with the same sum. Scores are computed with the GluonTS implementation, and DF reports means and standard deviations over five seeds.

The table below keeps, from the 12 comparison methods in Table 2, the three flow- and diffusion-based methods and NKF, which is lower than DF on Electricity and Wikipedia. Rankings refer to the full table, including the classical statistical and copula models.

| Method | Exchange | Solar | Electricity | Traffic | Taxi | Wikipedia |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer-MAF | $0.005\pm0.003$ | $0.301\pm0.014$ | $0.021\pm0.000$ | $0.056\pm0.001$ | $0.179\pm0.002$ | $0.063\pm0.003$ |
| TimeGrad | $0.006\pm0.001$ | $0.287\pm0.020$ | $0.021\pm0.001$ | $0.044\pm0.006$ | $0.114\pm0.020$ | $0.049\pm0.002$ |
| ScoreGrad sub-VP SDE | $0.006\pm0.001$ | $0.256\pm0.015$ | $0.019\pm0.001$ | $0.041\pm0.004$ | $0.101\pm0.004$ | $0.043\pm0.002$ |
| NKF | — | $0.320\pm0.020$ | $0.016\pm0.001$ | $0.100\pm0.002$ | — | $0.071\pm0.002$ |
| DF | $0.003\pm0.001$ | $0.289\pm0.002$ | $0.023\pm0.001$ | $0.040\pm0.004$ | $0.075\pm0.002$ | $0.085\pm0.007$ |

Appendix E.1 states that DF outperforms all prior methods except ScoreGrad, with which it is overall tied, and takes fourth place only on Wikipedia. In the table, DF is lowest on Exchange, Traffic, and Taxi. Besides ScoreGrad, TimeGrad (0.287) is below DF (0.289) on Solar, and NKF (0.016), Transformer-MAF, and TimeGrad (0.021) are below DF (0.023) on Electricity. On Wikipedia four methods are below DF, placing it fifth. [Table 2, E.1, p. 27](https://arxiv.org/pdf/2407.01392v4#page=27) **Review interpretation.** The table supports claims at the level of §4.5's "competitive" and E.1's "applicable to diverse domains with no apparent trade-off in performance", but not E.1's sentence that DF outperforms prior methods across the board.

## 5. Discussion

In the conclusion, the authors summarize that a next-token model trained with DF combines the benefits of next-token models and full-sequence diffusion, and that the new sampling and guidance schemes yield large gains on sequential decision-making tasks. As limitations they list the current RNN-based causal implementation, the likely need for large transformers following appendix B.1 for higher-resolution video or more complex distributions, and the absence of any study of scaling to internet-scale datasets and tasks. [§5, p. 10](https://arxiv.org/pdf/2407.01392v4#page=10)

## Review interpretation and open questions

**Review interpretation.** The strongest evidence comes from comparisons within one model: the three video methods share one RNN architecture, and the maze MCG ablation changes only sampling. The paper's substantive contribution is that a single training run yields stabilized rollouts, zig-zag planning, adjustable memory length, and designated observation corruption, all as sampling choices. Comparisons with external baselines rest on differing conditions: Diffuser\* uses a PD controller and a different guidance objective, the authors themselves note that episode reward favors slow plans, and in forecasting DF is lowest on only three of six datasets. The intuition of reading noise level as uncertainty is also never measured as calibrated confidence.

Open questions remain. How large is each effect when independent noise, stabilization, and fused SNR weighting are separated under a matched budget? How do score and latency change with the number of MCG samples? Does the ordering against Diffuser\* hold when the mazes are evaluated by first hitting time? With trial counts reported, and with human-designated corruption separated from automatically detected corruption, does the 76% hold?

## Sources and reading scope

- Paper: *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion* · Boyuan Chen et al. · first public date 2024-07-01 · [reviewed version](https://arxiv.org/abs/2407.01392v4) · reviewed 2026-10-23.
- Scope read: Version 4, sections 1–5 and experimental and implementation appendices B–F. Appendix-A ELBO and subsequence arguments are covered in part 2.
- External sources consulted: [DDPM v2](https://arxiv.org/abs/2006.11239v2) §2–3, [TimeGrad](https://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf) §2–3.1, [Diffuser](https://proceedings.mlr.press/v162/janner22a/janner22a.pdf) §2–3.1, [AR-Diffusion v1](https://arxiv.org/abs/2305.09515v1) §3.1–3.2, and `get_normalized_score` in the [official D4RL implementation](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py).
- Figure sources: The Figure-2 thumbnail and body Figures 2–4/Table-1 diagram are author assets from v4 (CC BY 4.0), rasterized and WebP-encoded with content, colors, arrows, and notation retained. [Paper](https://arxiv.org/abs/2407.01392v4).
- Not verified: code execution and experimental reproduction, whether the Table 1 scores are normalized (not stated in the source), and the Rolling Diffusion paper itself.
