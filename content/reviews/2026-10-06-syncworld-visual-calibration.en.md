---
title: "Can a World Model Read Actions After the Camera Moves? SyncWorld and Visual Calibration"
description: "How visual calibration and history distillation adapt SyncWorld to new robot setups, with conditioned comparisons of video quality, policy success, and runtime."
date: "2026-10-06"
updatedAt: "2026-10-23T15:26:11+00:00"
topics: ["world action model", "robot learning", "world simulation", "in-context learning", "visual calibration"]
visibility: "public"
lang: "en"
translationKey: "syncworld-visual-calibration"
paperTitle: "SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators"
paperPublishedDate: "2026-09-08"
authors: ["Yuncong Yang", "Zhengtao Han", "Furkan Ozyurt", "Zeyuan Yang", "Han Yang", "Junyi Cao", "Haoyu Zhen", "Yilun Du", "Chuang Gan"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2609.09155v1"
thumbnail: "/assets/reviews/syncworld/paper-figure-2.webp"
thumbnailAlt: "The model architecture from SyncWorld Figure 2"
---

SyncWorld is a zero-shot world model that uses a short visual-calibration context to infer action–visual mappings for unseen cameras and single-arm robots. Evaluated without inference-time calibration, a model trained without calibration worsens LIBERO LPIPS (Learned Perceptual Image Patch Similarity) from 0.049 to 0.134 (Table 4), and ranking policy candidates with SyncWorld raises Orange Juice successes, the largest gain of the three tasks, from 28 of 50 under direct π0 execution to 36 (Table 3). [Table 4](https://arxiv.org/html/2609.09155v1#S4.T4), [Table 3](https://arxiv.org/html/2609.09155v1#S4.T3) **Review interpretation.** The same-backbone ablation (§4.5) shows the effect of calibration training most directly, but that evidence is limited to video metrics, and the policy gain added by inference-time calibration, 34→36 of 50 even on Orange Juice, is weak evidence about policy performance.

This article follows original §1–5; Appendix C.2's runtime and Appendix E's failure cases come after the conclusion.

## 1. Introduction

An action-conditioned world model predicts future observations from the current observation and actions to be executed. The same movement command produces different visual motion when camera placement, robot base placement or embodiment changes. SyncWorld treats this setup-dependent relation as an **action–visual mapping**. The authors argue that this dependence creates two problems: mixing setups in training forces one model to fit incompatible mappings for the same action values (conflicting supervision), and in a new setup the mapping shifts, so models break without additional training. [SyncWorld §1](https://arxiv.org/html/2609.09155v1#S1)

<figure class="review-figure" id="figure-syncworld-core">
<a href="/assets/reviews/syncworld/syncworld-core.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/syncworld-core.svg" width="1200" height="1590" alt="Calibration action-video pairs C_s, interaction history H_t and future actions A_t condition a SyncWorld model whose parameters theta stay fixed at deployment, yielding predicted video for the new setup." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Core idea · New context, fixed model</span>Calibration supplies setup-specific action–visual correspondence as input context. Parameters θ stay fixed at deployment. C_s contains twelve directional calibration segments, H_t records past observations and actions, and A_t contains future commands to predict. The strips distinguish calibration pairs, an executed approach-grasp-lift history and planned commands, with blue ghost targets showing intent. The samples on the three input cards are AI-generated teaching examples, not paper data. <span class="figure-links"><a href="/assets/reviews/syncworld/syncworld-core.svg" target="_blank" rel="noopener noreferrer">Open full size</a></span></figcaption>
</figure>

## 2. Related Work

### 2.1 Controllable Action-Conditioned World Models

The authors distinguish video-based trajectory synthesis and action inference, joint video/action WAMs, and models predicting future observations from actions. Ray maps, visual action prompts, and embodiment masks already ground actions visually. SyncWorld instead focuses on setup-dependent numerical action semantics, specified through calibration context under changes in camera, controller convention, or embodiment. [Section 2.1](https://arxiv.org/html/2609.09155v1#S2.SS1)

### 2.2 Latent Action World Modeling

Latent-action methods learn controls explaining changes in unlabeled video. These controls can guide rollouts or be distilled into policies, but require grounding in executable low-level commands. The authors contrast this with directly conditioning on real commands and inferring their setup-specific visual effects from context. [Section 2.2](https://arxiv.org/html/2609.09155v1#S2.SS2)

### 2.3 In-Context Adaptation of World Models

Earlier adaptation routes include geometry or extrinsic estimation, privileged deployment supervision, test-time optimization/fine-tuning, and closed-loop refinement with real rollouts. SyncWorld instead supplies a short action–video prefix and infers the current mapping without parameter updates. [Section 2.3](https://arxiv.org/html/2609.09155v1#S2.SS3)

### Review supplement: establishing the comparison scope

IRASim studies frame-level action grounding, WorldGym evaluates policies in generated environments, and Ctrl-World uses multiple views/history and reports zero-shot results for new cameras. [IRASim section 3](https://arxiv.org/html/2406.14540v2), [WorldGym sections 3–4](https://arxiv.org/html/2506.00613v3), [Ctrl-World section 5.4](https://arxiv.org/html/2510.10125v3)

**Review interpretation.** Novelty is best assessed through explicit setup-specific calibration combined with history distillation, rather than through a claim to be the first model to transfer across setups.

## 3. Method

The four subsections retain the authors’ formulation, training, decision-making, and data order.

### 3.1 Calibration-based World Modeling

Visual calibration records the robot moving in basic directions under a fixed camera. Positive and negative segments for six motion degrees of freedom—translation and rotation—form 12 segments, placed in a canonical order at the beginning of the context. The control space has three translation dimensions, three rotation dimensions, and one gripper dimension. Gripper control is excluded from directional calibration because its visual meaning is considered directly interpretable. [SyncWorld §3.1, Eq. 4, Appendix B.2.2–B.2.3](https://arxiv.org/html/2609.09155v1#S3.SS1)

The conditional distribution in Eq. 3 uses the following original notation.

$$
I_{t+1:t+H}\sim W_{\theta}\!\left(\cdot\mid\mathcal{C}^{s},H_t,A_t\right).
$$

$\mathcal C^s$ is calibration context for setup $s$; $H_t$ is the observation–action history; and $A_t$ contains future actions. $W_\theta$ denotes a conditional distribution over future video, with model parameters $\theta$ fixed at deployment. Zero-shot here means no parameter retraining for the new setup. [SyncWorld §3.1, Eq. 3](https://arxiv.org/html/2609.09155v1#S3.SS1)

<figure class="review-figure" id="figure-syncworld-architecture">
<a href="/assets/reviews/syncworld/paper-figure-2.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-2.webp" width="1600" height="704" alt="SyncWorld architecture: calibration, interaction history and future actions become pose embeddings and video latents that condition the DiT generating future video." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Paper Figure 2 · From calibration to future video</span>Read the three left-hand streams as inputs, the central DiT as the generator, and the right-hand frames as predictions. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-2.webp" target="_blank" rel="noopener noreferrer">Open full size</a> · <a href="https://arxiv.org/html/2609.09155v1#S3.F2">Yuncong Yang et al., 2026, Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Content and layout preserved; resized and WebP-compressed, without cropping</span></figcaption>
</figure>

The three inputs have different roles. **Calibration** reveals the action–visual correspondence in this setup; **history** records interactions that have already occurred; and **future actions** specify commands to be executed. Poses become vector embeddings and video becomes a compressed latent representation. A diffusion transformer (DiT) uses these conditions to refine noisy future-video representations. The future noise in the diagram is the starting point for frames to be generated. [Paper Figure 2, §3.1, Appendix B.1.1](https://arxiv.org/html/2609.09155v1#S3.F2)

### 3.2 Learning World Model with Visual Calibration

The authors note that with calibration merely placed in context, a model may take the shortcut of memorizing a fixed coordinate convention instead of reading the calibration, so they add two training designs. First, the same coordinate transformation (sign flips of x, y and z, axis permutation, and translation scaling) is applied to calibration, history, and future actions, encouraging the model to read their contextual correspondence rather than rely on the numbers alone. Second, **calibration-to-history distillation** aligns predictions from inputs with calibration and history-only inputs with calibration removed. "No inference calibration" results therefore still come from a calibration-trained model; a model trained without calibration is compared separately in §4.5. [SyncWorld §3.2, Appendix B.1.2–B.1.3](https://arxiv.org/html/2609.09155v1#S3.SS2)

### 3.3 Zero-shot Policy Improvement via World-Model Ranking

Policy improvement is a separate question. The system samples eight action candidates from a frozen vision-language-action policy (VLA), π0, and generates two-view future videos for each candidate. A GPT-5 vision-language model (VLM) judges task progress and physical plausibility to select a candidate. This uses GPC-Rank, the ranking variant of Generative Predictive Control (GPC). [SyncWorld §3.3, Appendix C.4.1–C.4.2](https://arxiv.org/html/2609.09155v1#S3.SS3), [GPC §III–IV](https://arxiv.org/html/2502.00622v4)

<figure class="review-figure" id="figure-syncworld-policy">
<a href="/assets/reviews/syncworld/paper-figure-4.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-4.webp" width="1800" height="710" alt="Paper Figure 4. A VLA proposes action candidates, SyncWorld predicts two-view outcomes, and a VLM scores them against the instruction before the highest-scored candidate is executed." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Paper Figure 4 · From imagined outcomes to action selection</span>Read candidate generation, future video and scoring from left to right. Rows with the same color represent the same action candidate. The numbers on the right are ranking scores. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-4.webp" target="_blank" rel="noopener noreferrer">Open full size</a> · <a href="https://arxiv.org/html/2609.09155v1#S3.F4">Yuncong Yang et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Content and layout preserved; resized and WebP-compressed, without cropping</span></figcaption>
</figure>

The diagram illustrates three candidate rows, while the evaluation uses eight candidates, sixteen actions per candidate, and two views. The base policy is an unofficial LeRobot π0 LIBERO checkpoint used without task-specific fine-tuning, and the selected sixteen actions run open-loop before the next call. GPT-5 receives three representative frames sampled from generated video. The authors themselves note that the outcome depends on two factors, rollout-simulator quality and ranking-procedure quality. [Paper Figure 4, §3.3, Appendix C.4.1–C.4.3 and Table 7](https://arxiv.org/html/2609.09155v1#S3.F4)

### 3.4 Training Data

Training data come mainly from RLBench, RoboCasa, and RoboMimic simulations, with varied cameras and replayed actions. They include outcomes of actions perturbed away from successful trajectories, alongside real DROID data without calibration. The corpus totals about 166 hours (about 54 hours real), each trajectory recorded from 3.6 viewpoints on average, and calibration supervision comes mainly from the simulated data. [SyncWorld §3.4, Appendix B.2](https://arxiv.org/html/2609.09155v1#S3.SS4)

## 4. Experiments

The authors run three experiments: future prediction in unseen environments, viewpoints and embodiments; cross-view 3D consistency; and training-free policy improvement.

### 4.1 Experiment Setups

All evaluation environments are unseen during training. LIBERO and ManiSkill use the Franka Panda embodiment present in the training data, while the real-world evaluation uses an xArm unseen in training, testing embodiment generalization. LIBERO and ManiSkill use 50 trajectories each and the real setting 25; two views per trajectory yield 100, 100, and 50 evaluation videos respectively. The model predicts a future chunk corresponding to 16 actions (about 1 s) at 512×512 resolution. [SyncWorld §4.1](https://arxiv.org/html/2609.09155v1#S4.SS1)

SyncWorld adds pose-conditioning modules to every block of the Wan2.2-TI2V-5B checkpoint and fine-tunes the whole denoising model. Inference uses a 20-step flow-matching schedule, action classifier-free guidance (CFG) with strength 5.0, and 4-GPU Unified Sequence Parallelism (USP). The baselines IRASim, WorldGym, and Ctrl-World are fine-tuned on the same downstream data and evaluated on the same held-out trajectories. [Appendix B.1.1, C.1–C.2](https://arxiv.org/html/2609.09155v1#A3.SS1)

### 4.2 World Model Quality Analysis

The video metrics answer different questions. PSNR measures reconstruction through pixel mean-squared error (dB, higher is better); SSIM compares local luminance, contrast, and structure (higher is better). LPIPS measures distance in learned visual features (lower is better). FID compares the means and covariances of real and generated image features (lower is better). The first three compare each predicted frame with the real frame at the same time; FID compares sets. [§4.1–4.2, Table 1](https://arxiv.org/html/2609.09155v1#S4.T1)

**LIBERO columns excerpted from Table 1.** The authors report that SyncWorld beats every baseline on all 12 columns across the three domains even without inference calibration, and is best on all 12 with calibration. ManiSkill and real (xArm) LPIPS are 0.071 and 0.042 without calibration and 0.049 and 0.039 with it, against Ctrl-World's 0.178 and 0.142.

| Model | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | FID ↓ |
| --- | --- | --- | --- | --- |
| IRASim | 23.5 | 0.876 | 0.149 | 19.9 |
| WorldGym | 24.1 | 0.882 | 0.142 | 17.4 |
| Ctrl-World | 24.8 | 0.892 | 0.137 | 16.5 |
| SyncWorld, no inference calibration | 27.9 | 0.933 | 0.049 | 8.7 |
| SyncWorld, with inference calibration | 28.3 | 0.935 | 0.035 | 7.0 |

The two SyncWorld rows are two inference conditions of the same model. **Review interpretation.** The gap to the baselines may also reflect differences in architecture and pretraining, but because SyncWorld trained without calibration reaches LIBERO LPIPS 0.134 in §4.5, close to Ctrl-World's 0.137, much of the large gap appears together with calibration training.

<figure class="review-figure" id="figure-syncworld-qualitative">
<a href="/assets/reviews/syncworld/paper-figure-5.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-5.webp" width="2400" height="893" alt="Two camera views show real robot scenes over time, comparing ground-truth observations with Ctrl-World and SyncWorld future predictions in the original paper figure." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Paper Figure 5 · Real-robot prediction comparison</span>The left and right groups show different camera views; within each group, rows are ground truth, Ctrl-World, and SyncWorld from top to bottom. Read time from left to right and compare arm and gripper positions. These are author-selected qualitative examples on the xArm embodiment, unseen in training, and new viewpoints. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-5.webp" target="_blank" rel="noopener noreferrer">Open full size</a> · <a href="https://arxiv.org/html/2609.09155v1#S4.F5">Yuncong Yang et al., 2026, Figure 5</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Content and layout preserved; resized and WebP-compressed, without cropping</span></figcaption>
</figure>

### 4.3 Multi-view Spatial Consistency Analysis

Cross-view consistency compares a rollout generated from one view with the real video from another view of the same trajectory. Met3R uses DUSt3R to predict 3D pointmaps for both videos, lifts DINO features into 3D, and scores the cosine similarity of features in overlapping regions. Scores are unitless, and lower is better. The oracle is the Met3R between the two real views, which the authors treat as an empirical upper bound on cross-view agreement. The average covers LIBERO, ManiSkill, and real data. [§4.3, Appendix C.3](https://arxiv.org/html/2609.09155v1#A3.SS3)

**All rows of Paper Table 2 — cross-view consistency (Met3R, lower is better).**

| Model | LIBERO ↓ | ManiSkill ↓ | Real ↓ | Average ↓ |
| --- | --- | --- | --- | --- |
| IRASim | 0.579 | 0.633 | 0.468 | 0.560 |
| WorldGym | 0.583 | 0.632 | 0.515 | 0.577 |
| Ctrl-World | 0.559 | 0.642 | 0.494 | 0.565 |
| SyncWorld, no inference calibration | 0.545 | 0.607 | 0.474 | 0.542 |
| SyncWorld, with inference calibration | 0.540 | 0.602 | 0.473 | 0.538 |
| Oracle | 0.515 | 0.594 | 0.459 | 0.523 |

Adding calibration at inference lowers the average from 0.542 to 0.538. The authors write that SyncWorld achieves lower Met3R than all baselines in all three domains and approaches the oracle (§4.3, Table 2 caption), yet the same table's real column bolds IRASim's 0.468, against SyncWorld's 0.474 and 0.473. [SyncWorld §4.3, Table 2](https://arxiv.org/html/2609.09155v1#S4.T2) **Review interpretation.** By the printed values IRASim is lowest in the real column, so "all three domains" holds only for LIBERO and ManiSkill, while the conclusion of a better average (0.538 versus 0.560) stands.

### 4.4 Zero-shot Policy Improvement with SyncWorld

**Paper Table 3 — Three tasks with oracle headroom.** Success fractions are converted to percentages, including the row without inference calibration.

| Method | BBQ Sauce | Orange Juice | Black Bowl |
| --- | --- | --- | --- |
| Direct π0 | 52% | 56% | 48% |
| SyncWorld, no inference calibration | 54% | 68% | 58% |
| SyncWorld, with inference calibration | 58% | 72% | 60% |
| Ranking with ground-truth simulator | 60% | 80% | 66% |

Success rate is the fraction of successful runs over 50 episodes per task. In counts, direct execution → no calibration → with calibration is 26→27→29 for BBQ Sauce, 28→34→36 for Orange Juice, and 24→29→30 for Black Bowl. The total ranking gain is largest on Orange Juice (8 successes), and inference calibration adds 2, 2 and 1 successes on the three tasks. The authors conclude that calibration consistently improves policy improvement and that SyncWorld approaches the oracle upper bound. The three tasks were chosen because ranking with the ground-truth simulator improves on the direct policy, and the authors state that the selection is not based on SyncWorld's performance. [SyncWorld §4.4, Table 3, Appendix C.4.3](https://arxiv.org/html/2609.09155v1#A3.SS4.SSS3)

**Paper Table 8 — Additional tasks with little or no oracle headroom.** The authors include these in the appendix for transparency; fractions are again converted to percentages.

| Method | Alphabet Soup | Ketchup | Put Cream Cheese |
| --- | --- | --- | --- |
| Direct π0 | 58% | 80% | 34% |
| SyncWorld | 58% | 72% | 34% |
| Ranking with ground-truth simulator | 62% | 74% | 30% |

Even ground-truth simulator ranking trails direct execution by six percentage points on Ketchup and four on Put Cream Cheese. Accurate future observations can still lead the evaluator to choose a worse candidate. The authors interpret the VLM/GPC-Rank ranking signal as the bottleneck on these tasks, making them less informative for comparing simulators. [Paper Appendix C.4.3, Table 8](https://arxiv.org/html/2609.09155v1#A3.T8)

### 4.5 Analysis: removing training calibration and distillation

Tables 4–5 remove **calibration or distillation during training**, unlike the earlier inference-calibration comparison. Table 4 reports video metrics for the three domains and Table 5 Met3R; below are two LIBERO metrics and the three-domain Met3R average.

| Training configuration | LIBERO LPIPS ↓ | LIBERO FID ↓ | Mean Met3R across three domains ↓ |
| --- | --- | --- | --- |
| Full model | 0.049 | 8.7 | 0.542 |
| Without calibration | 0.134 | 15.6 | 0.566 |
| Without distillation | 0.148 | 17.8 | 0.574 |

The full-model row matches Table 1’s **no inference calibration** row. The authors read the results as follows: removing calibration in training causes clear generalization degradation, while removing distillation and keeping calibration "mainly hurts performance when test-time calibration is unavailable." In their summary, calibration improves cross-domain action alignment and distillation carries that ability to calibration-free deployment. [§4.5, Tables 4–5](https://arxiv.org/html/2609.09155v1#S4.SS5)

The authors also report that SyncWorld stays close to the calibrated variant without inference calibration in all three experiments, attributing this to coordinate augmentation and distillation making interaction history act as an implicit calibration signal. Appendix F.5 adds, however, that on embodiments unseen in training, qualitative controllability degrades without inference calibration, as the model maps action dimensions to visual motion directions less consistently. In Appendix D's augmentation-probability comparison (with inference calibration), the default 0.6 beats 0, 0.3 and 0.9 (LIBERO LPIPS 0.035 versus 0.055, 0.041 and 0.053). [Appendix F.5](https://arxiv.org/html/2609.09155v1#A6.SS5), [Appendix D, Table 9](https://arxiv.org/html/2609.09155v1#A4)

## 5. Conclusion

The authors conclude that conditioning on short visual calibration and history can produce controllable predictions under new cameras and single-arm embodiments, rather than forcing one global interpretation of numerical actions. Distillation connects this ability to inference without explicit calibration; candidate ranking demonstrates potential decision-time utility. [Original §5](https://arxiv.org/html/2609.09155v1#S5)

## Appendix C.2 and E: runtime and failure cases

**Original Table 6 — generation time and conditions.** All baseline and SyncWorld configurations are retained.

| Model | GPU setup | Denoising steps | Time |
| --- | --- | --- | --- |
| WorldGym | 1 A100 | 50 | 37.2 s |
| Ctrl-World | 1 H100 | 50 | 4.0 s |
| SyncWorld | One H100 | 50 | 39.4 s |
| SyncWorld | One H100 | 20 | 15.6 s |
| SyncWorld | Four H100s | 20 | 3.2 s |

Table 6 gives the time for one video generation. The inference setting used in all experiments, four H100s with 20 steps, takes 3.2 seconds; one H100 with 20 steps takes 15.6 seconds. At one H100 and 50 steps, Ctrl-World takes 4.0 seconds versus SyncWorld's 39.4 seconds. The policy-improvement experiment generates two-view videos for eight candidates, sixteen videos in all, at every decision step and scores them with GPT-5. [SyncWorld Appendix C.2, Table 6](https://arxiv.org/html/2609.09155v1#A3.SS2)

The authors show unstable motion at an extreme camera angle, blur and position errors for unseen objects, hallucination of some objects in multi-object scenes, and blur during precise rotation with the xArm unseen in training. [SyncWorld Appendix E.1–E.4](https://arxiv.org/html/2609.09155v1#A5)

## Review interpretation and open questions

**Review interpretation.** The strongest evidence is the §4.5 ablation that removes calibration training and distillation within the same backbone. Without calibration training, all 12 video metrics across the three domains worsen, the Met3R average rises from 0.542 to 0.566, and LIBERO LPIPS climbs to roughly the level of the Ctrl-World baseline. The effect of providing calibration at inference (LIBERO LPIPS 0.049→0.035) is also clear on video metrics, but smaller than the change from removing calibration training (0.049→0.134).

This evidence has limits. First, the ablations report only video metrics without run-to-run variance, and there is no evaluation of the no-distillation model with inference calibration, so the authors' reading that distillation "mainly" matters when calibration is unavailable is not directly checked by the table. Second, a new embodiment is tested only on 25 real xArm trajectories, where, as Appendix F.5 notes, controllability degrades without inference calibration. Third, the main text reports policy improvement only on three tasks with oracle headroom, and inference calibration adds 1–2 successes per task, which is hard to separate from run-to-run variation as a "consistent" improvement. As Table 8 shows, ground-truth simulator ranking gains nothing where the judge signal is weak, so policy gains also depend on the quality of the VLM that ranks candidates. That two of the four failure types in Appendix E involve object interaction (unseen objects, multiple objects) is another reason to assess arm controllability and object-interaction accuracy separately.

Recent work separates the setup changes SyncWorld treats together. XEWorld evaluates transfer across bimanual embodiments in controlled scenes, separating appearance, kinematics, and object state, and the action-format study investigates how equivalent absolute and relative commands can change model outcomes. Both differ from SyncWorld in embodiment and action space or in evaluation (fixed camera, offline proxy evaluation), so neither is a direct comparison. On compute, Think Like a World Model, Act Like a VLA (distilling Cosmos3-Nano understanding representations into a compact policy) and RoboActualizer (training future and action experts on frozen V-JEPA 2.1 and discarding future outputs during execution) avoid generating video rollouts at deployment, but both require platform-specific training. ReWAM's feature calibration has a similar name but is a different concept from SyncWorld's action–video calibration episode. [XEWorld §3 and §5–6](https://arxiv.org/html/2608.05799v1), [Robot World Models Are Not Invariant to How the Actions Are Written §3–6](https://arxiv.org/html/2609.23252v1), [Representation distillation §III-B–III-C and §IV](https://arxiv.org/html/2609.24682v2), [RoboActualizer §3–4](https://arxiv.org/html/2609.36413v3), [ReWAM §3.2–3.4 and §4.3](https://arxiv.org/html/2609.38163v1)

Open questions remain.

1. Does calibration remain effective when changing only the camera, only an information-preserving action format, or only the embodiment?
2. Can changes to calibration directions or ordering, together with separate object-interaction error measurements, identify which context the model uses?
3. With matched candidates, judges, and total compute budgets, how do video simulation and representation distillation into policies differ in practical benefit?

## Sources and reading scope

- Paper: *SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators* · Yuncong Yang et al. · first public date 2026-09-08 · [reviewed version](https://arxiv.org/abs/2609.09155v1) · reviewed 2026-10-23.
- Scope read: Sections 1–5 and appendices A–F; supplementary videos are excluded.

| Additional source | Scope read |
| --- | --- |
| [IRASim v2](https://arxiv.org/html/2406.14540v2) | Relevant passages and experiment descriptions in §3.1–3.3, §4.1–4.3, §5 |
| [WorldGym v3](https://arxiv.org/html/2506.00613v3) | Relevant passages in §3.1.1–3.1.3, §4.1–4.2, §6 |
| [Ctrl-World v3](https://arxiv.org/html/2510.10125v3) | Relevant passages in §4.1–4.2, §5.3–5.4, §6 |
| [GPC v4](https://arxiv.org/html/2502.00622v4) | Method, experiment, and limitation passages in §III–IV, §V-A–V-C, §VII |
| [Context and Diversity Matter v2](https://arxiv.org/html/2509.22353v2) | Relevant passages in §3.1–3.4, §4.1–4.2, §5 |
| [XEWorld v1](https://arxiv.org/html/2608.05799v1) | Relevant passages and tables in §3.1–3.2, §5.1–5.6, §6 |
| [Action-format invariance study v1](https://arxiv.org/html/2609.23252v1) | Relevant passages in §3–4, §5.1, §6 Limitations |
| [Think Like a World Model, Act Like a VLA v2](https://arxiv.org/html/2609.24682v2) | Relevant passages and tables in §III-B–III-C, §IV-B–IV-E, §V |
| [RoboActualizer v3](https://arxiv.org/html/2609.36413v3) | Relevant passages and tables in §3.1–3.4, §4.1–4.5.2, §5, Appendix C.1–C.2 |
| [ReWAM v1](https://arxiv.org/html/2609.38163v1) | Relevant passages and tables in §3.2–3.4, §4.2–4.3, §4.5, §5 |

- Figure sources: Figures 2 (thumbnail), 4, and 5 are author assets from v1, CC BY 4.0; resolution reduction and WebP compression preserve content and layout. The self-made concept diagram includes illustrative AI-generated samples and optimized embedded rasters. [Paper](https://arxiv.org/abs/2609.09155v1).
- Not verified: research-code execution, model training/reproduction, and supplementary videos.
