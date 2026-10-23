---
title: "Can a World Model Read Actions After the Camera Moves? SyncWorld and Visual Calibration"
description: "SyncWorld uses a short action–video context to adapt to a new robot setup. This review separates video prediction from policy success and connects ten related papers to questions about transfer and inference cost."
date: "2026-10-06"
updatedAt: "2026-10-22T23:22:42+09:00"
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

Does knowing a robot's numerical action tell us how it will move on screen? SyncWorld first observes a short action–video correspondence in a new setup, then uses that context to predict future video without additional training. Its reported results support the promise of this approach, but do not establish physical accuracy for every embodiment or policy improvement on every task. [SyncWorld §3–4, Appendix C.4.3 and E](https://arxiv.org/html/2609.09155v1#S3)

**Notation.** The text uses the paper’s $\mathcal C^s$. C_s in the existing conceptual illustration is a simplified label for the same calibration context.

This article follows original §1–5. The ranking **method** is in §3.3 and success rates are in §4.4. Runtime and failure cases are grouped at the end under Appendix C.2 and E. Additional literature comparisons follow the authors’ conclusion as review interpretation.

## 1. Introduction

An action-conditioned world model predicts future observations from the current observation and actions to be executed. The same movement command can appear to move in a different direction when the camera changes. Moving the camera to the opposite side is an illustrative example, not a separate experimental result. SyncWorld treats this setup-dependent correspondence as an **action–visual mapping**. [SyncWorld §3.1](https://arxiv.org/html/2609.09155v1#S3.SS1)

<figure class="review-figure" id="figure-syncworld-core">
<a href="/assets/reviews/syncworld/syncworld-core.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/syncworld-core.svg" width="1200" height="1590" alt="Calibration action-video pairs C_s, interaction history H_t and future actions A_t condition a SyncWorld model whose parameters theta stay fixed at deployment, yielding predicted video for the new setup." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Core idea · New context, fixed model</span>Calibration supplies setup-specific action–visual correspondence as input context. Parameters θ stay fixed at deployment. C_s contains twelve directional calibration segments, H_t records past observations and actions, and A_t contains future commands to predict. This reviewer schematic explains the conditioning relationship; it is not generated video or an experimental result. The sample strips are AI-generated teaching examples of calibration pairs, an executed approach-grasp-lift history, and planned commands. Blue ghost targets show intent. They are not paper data, calibrated trajectories, or model-predicted video. <span class="figure-links"><a href="/assets/reviews/syncworld/syncworld-core.svg" target="_blank" rel="noopener noreferrer">Open full size</a></span></figcaption>
</figure>

## 2. Related Work

Section 2 distinguishes action-conditioned world models, latent-action world models, and in-context adaptation. Latent controls inferred from video still need grounding into executable robot commands. SyncWorld directly conditions on low-level actions and infers their setup-specific meaning from context. This distinction motivates the authors’ calibration episode. [Original §2.1–2.3](https://arxiv.org/html/2609.09155v1#S2)

Earlier work had already addressed parts of this problem. IRASim models the alignment between actions and video frames; WorldGym evaluates policies in generated environments. Ctrl-World uses multiple views and history, and also reports zero-shot results in new DROID camera setups. My reading is therefore that SyncWorld explicitly designs a calibration context to reveal the correspondence, rather than being the first model to make setup transfer possible. That comparison is a review interpretation. [IRASim §3](https://arxiv.org/html/2406.14540v2), [WorldGym §3–4](https://arxiv.org/html/2506.00613v3), [Ctrl-World §4 and §5.4](https://arxiv.org/html/2510.10125v3)

Context and Diversity Matter provides useful theoretical background: it distinguishes recognizing a previously encountered environment from learning transitions through context. Its theoretical assumptions about states and spaces, and its experimental environments, differ from robot video. I therefore do not treat its results as a proof of SyncWorld's performance. [Context and Diversity Matter §3–4](https://arxiv.org/html/2509.22353v2)

## 3. Method

The four subsections retain the authors’ formulation, training, decision-making, and data order.

### 3.1 Calibration-based World Modeling

Visual calibration records the robot moving in basic directions under a fixed camera. Positive and negative segments for six motion degrees of freedom—translation and rotation—form 12 segments, placed in a canonical order at the beginning of the context. The control space has three translation dimensions, three rotation dimensions, and one gripper dimension. This does not mean the robot has seven mechanical joints. Gripper control is excluded from directional calibration because its visual meaning is considered directly interpretable. [SyncWorld §3.1, Eq. 4, Appendix B.2.2–B.2.3](https://arxiv.org/html/2609.09155v1#S3.SS1)

The conditional distribution in Eq. 3 uses the following original notation.

$$
I_{t+1:t+H}\sim W_{\theta}\!\left(\cdot\mid\mathcal{C}^{s},H_t,A_t\right).
$$

$\mathcal C^s$ is calibration context for setup $s$; $H_t$ is the observation–action history; and $A_t$ contains future actions. $W_\theta$ denotes a conditional distribution over future video, with model parameters $\theta$ fixed at deployment. Zero-shot here means no parameter retraining for the new setup. It does not mean that calibration observations or computation are unnecessary. [SyncWorld §3.1, Eq. 3](https://arxiv.org/html/2609.09155v1#S3.SS1)

<figure class="review-figure" id="figure-syncworld-architecture">
<a href="/assets/reviews/syncworld/paper-figure-2.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-2.webp" width="1600" height="704" alt="SyncWorld architecture: calibration, interaction history and future actions become pose embeddings and video latents that condition the DiT generating future video." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Paper Figure 2 · From calibration to future video</span>Read the three left-hand streams as inputs, the central DiT as the generator, and the right-hand frames as predictions. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-2.webp" target="_blank" rel="noopener noreferrer">Open full size</a> · <a href="https://arxiv.org/html/2609.09155v1#S3.F2">Yuncong Yang et al., 2026, Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Content and layout preserved; resized and WebP-compressed, without cropping</span></figcaption>
</figure>

The three inputs have different roles. **Calibration** reveals the action–visual correspondence in this setup; **history** records interactions that have already occurred; and **future actions** specify commands to be executed. Poses become vector embeddings and video becomes a compressed latent representation. A diffusion transformer (DiT) uses these conditions to refine noisy future-video representations. The future noise in the diagram is the starting point for frames to be generated. [Paper Figure 2, §3.1, Appendix B.1.1](https://arxiv.org/html/2609.09155v1#S3.F2)

### 3.2 Learning World Model with Visual Calibration

The method requires more than appending calibration video to an existing generator. During training, the same coordinate transformation is applied to calibration, history, and future actions, encouraging the model to read their contextual correspondence rather than rely on the numbers alone. **Calibration-to-history distillation** also aligns predictions from inputs with calibration and history-only inputs with calibration removed. Results without explicit calibration at inference still come from a model trained this way; they are distinct from training a model without calibration altogether. [SyncWorld §3.2, Appendix B.1.2–B.1.3](https://arxiv.org/html/2609.09155v1#S3.SS2)

### 3.3 Zero-shot Policy Improvement via World-Model Ranking

Policy improvement is a separate question. The system samples eight action candidates from a frozen vision-language-action policy (VLA), π0, and generates two-view future videos for each candidate. A GPT-5 vision-language model (VLM) judges task progress and physical plausibility to select a candidate. This uses GPC-Rank, the ranking variant of Generative Predictive Control (GPC). [SyncWorld §3.3, Appendix C.4.1–C.4.2](https://arxiv.org/html/2609.09155v1#S3.SS3), [GPC §III–IV](https://arxiv.org/html/2502.00622v4)

<figure class="review-figure" id="figure-syncworld-policy">
<a href="/assets/reviews/syncworld/paper-figure-4.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-4.webp" width="1800" height="710" alt="Paper Figure 4. A VLA proposes action candidates, SyncWorld predicts two-view outcomes, and a VLM scores them against the instruction before the highest-scored candidate is executed." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Paper Figure 4 · From imagined outcomes to action selection</span>Read candidate generation, future video and scoring from left to right. Rows with the same color represent the same action candidate. The numbers on the right are ranking scores, not success rates in percent. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-4.webp" target="_blank" rel="noopener noreferrer">Open full size</a> · <a href="https://arxiv.org/html/2609.09155v1#S3.F4">Yuncong Yang et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Content and layout preserved; resized and WebP-compressed, without cropping</span></figcaption>
</figure>

The diagram illustrates three candidate rows, while the evaluation uses eight candidates, sixteen actions per candidate, and two views. GPT-5 receives three representative frames sampled from generated video. SyncWorld predicts outcomes; GPT-5 assigns scores. Errors in both stages contribute to the decision, so better video does not automatically imply better action selection. [Paper Figure 4, §3.3, Appendix C.4.2 and Table 7](https://arxiv.org/html/2609.09155v1#S3.F4)

### 3.4 Training Data

Training data come mainly from RLBench, RoboCasa, and RoboMimic simulations, with varied cameras and replayed actions. They include outcomes of actions perturbed away from successful trajectories, alongside real DROID data without calibration. The absence of additional training in a new setup should be read together with this prior training on diverse data. [SyncWorld §3.4, Appendix B.2](https://arxiv.org/html/2609.09155v1#S3.SS4)

## 4. Experiments

Read quality, cross-view consistency, and policy improvement as distinct evidence.

### 4.1 Experiment Setups

Video evaluation predicts a future chunk corresponding to 16 actions at 512×512 resolution. It uses 50 trajectories each for LIBERO and ManiSkill, and 25 real trajectories. Two views per trajectory yield 100, 100, and 50 evaluation videos respectively. Table 1 reports broadly better video-quality metrics than IRASim, WorldGym, and Ctrl-World. The baselines are fine-tuned on the same downstream data, but their backbones and training procedures are not identical, so differences between complete models do not isolate calibration alone. [SyncWorld §4.1–4.2, Table 1, Appendix C.1](https://arxiv.org/html/2609.09155v1#S4.SS1)

### 4.2 World Model Quality Analysis

The video metrics answer different questions. PSNR measures reconstruction through pixel mean-squared error (dB, higher is better); SSIM compares local luminance, contrast, and structure (higher is better). LPIPS measures distance in learned visual features (lower is better). FID compares the means and covariances of real and generated image features (lower is better). The first three compare corresponding frames; FID compares sets. Small evaluation samples and different aggregation procedures can change FID comparisons. None directly measures contact physics or task success. [§4.1–4.2, Table 1](https://arxiv.org/html/2609.09155v1#S4.T1)

**LIBERO columns excerpted from Table 1.** The same metrics for the other two domains remain in the original table.

| Model | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | FID ↓ |
| --- | --- | --- | --- | --- |
| Ctrl-World | 24.8 | 0.892 | 0.137 | 16.5 |
| SyncWorld, no inference calibration | 27.9 | 0.933 | 0.049 | 8.7 |
| SyncWorld, with inference calibration | 28.3 | 0.935 | 0.035 | 7.0 |

The two SyncWorld rows compare inference conditions after the same training. The comparison with Ctrl-World concerns entire models and does not isolate a causal effect of calibration.

<figure class="review-figure" id="figure-syncworld-qualitative">
<a href="/assets/reviews/syncworld/paper-figure-5.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-5.webp" width="2400" height="893" alt="Two camera views show real robot scenes over time, comparing ground-truth observations with Ctrl-World and SyncWorld future predictions in the original paper figure." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Paper Figure 5 · Real-robot prediction comparison</span>The left and right groups show different camera views; within each group, rows are ground truth, Ctrl-World, and SyncWorld from top to bottom. Read time from left to right and compare arm and gripper positions. These are author-selected qualitative examples, not proof of overall success rates or physical accuracy. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-5.webp" target="_blank" rel="noopener noreferrer">Open full size</a> · <a href="https://arxiv.org/html/2609.09155v1#S4.F5">Yuncong Yang et al., 2026, Figure 5</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Content and layout preserved; resized and WebP-compressed, without cropping</span></figcaption>
</figure>

### 4.3 Multi-view Spatial Consistency Analysis

Met3R measures cross-view consistency through feature correspondence. Its scores are unitless, and lower is better. The reported average covers LIBERO, ManiSkill, and real data. Read the individual domain columns before the average.

**Paper Table 2 — Cross-view consistency.** All rows are transcribed below. Lower is better; Oracle is a reference score from actual observations.

| Model | LIBERO ↓ | ManiSkill ↓ | Real ↓ | Average ↓ |
| --- | --- | --- | --- | --- |
| IRASim | 0.579 | 0.633 | 0.468 | 0.560 |
| WorldGym | 0.583 | 0.632 | 0.515 | 0.577 |
| Ctrl-World | 0.559 | 0.642 | 0.494 | 0.565 |
| SyncWorld, no inference calibration | 0.545 | 0.607 | 0.474 | 0.542 |
| SyncWorld, with inference calibration | 0.540 | 0.602 | 0.473 | 0.538 |
| Oracle | 0.515 | 0.594 | 0.459 | 0.523 |

Adding calibration at inference lowers the average from 0.542 to 0.538. Both rows use models trained with calibration and distillation, so this differs from removing calibration during training. SyncWorld has the better average, but IRASim has the lower score in the real-data column. The statement that SyncWorld is most consistent in every domain therefore does not exactly match the table. The statistical significance of this difference has not been verified. The metric evaluates cross-view feature consistency, not the accuracy of forces, contact, or physical laws as a whole. [SyncWorld §4.3, Table 2, Appendix C.3](https://arxiv.org/html/2609.09155v1#S4.SS3)

### 4.4 Zero-shot Policy Improvement with SyncWorld

**Paper Table 3 — Three tasks with oracle headroom.** Success fractions are converted to percentages, including the row without inference calibration.

| Method | BBQ Sauce | Orange Juice | Black Bowl |
| --- | --- | --- | --- |
| Direct π0 | 52% | 56% | 48% |
| SyncWorld, no inference calibration | 54% | 68% | 58% |
| SyncWorld, with inference calibration | 58% | 72% | 60% |
| Ranking with ground-truth simulator | 60% | 80% | 66% |

Success rate is the fraction of successful runs over 50 episodes per task. Reading Orange Juice as 56% direct execution → 68% without calibration → 72% with calibration distinguishes the benefit of the candidate-ranking system from the additional inference-calibration gain. The changes are sixteen percentage points over direct execution and four over the no-calibration model. These three tasks were selected because ranking with the ground-truth simulator improved on the direct policy. They were not selected based on SyncWorld's performance, but they are not a full LIBERO average either. [SyncWorld §4.4, Table 3, Appendix C.4.3, Table 8](https://arxiv.org/html/2609.09155v1#A3.SS4.SSS3)

**Paper Table 8 — Additional tasks where gains are not assured.** Fractions are again converted to percentages. Considering only the three tasks above would hide these counterexamples.

| Method | Alphabet Soup | Ketchup | Put Cream Cheese |
| --- | --- | --- | --- |
| Direct π0 | 58% | 80% | 34% |
| SyncWorld | 58% | 72% | 34% |
| Ranking with ground-truth simulator | 62% | 74% | 30% |

Even ground-truth simulator ranking trails direct execution by six percentage points on Ketchup and four on Put Cream Cheese. Accurate future observations can still lead the evaluator to choose a worse candidate. The authors interpret the VLM/GPC-Rank signal as the bottleneck on these tasks. The table alone does not separate the contributions of SyncWorld prediction errors and ranking errors. [Paper Appendix C.4.3, Table 8](https://arxiv.org/html/2609.09155v1#A3.T8)

### 4.5 Analysis: removing training calibration and distillation

Tables 4–5 remove **calibration or distillation during training**, unlike the earlier inference-calibration comparison. The LIBERO columns of Table 4 and the three-domain average of Table 5 help distinguish these roles.

| Training configuration | LIBERO LPIPS ↓ | LIBERO FID ↓ | Mean Met3R across three domains ↓ |
| --- | --- | --- | --- |
| Full model | 0.049 | 8.7 | 0.542 |
| Without calibration | 0.134 | 15.6 | 0.566 |
| Without distillation | 0.148 | 17.8 | 0.574 |

The full-model row matches Table 1’s **no inference calibration** row. Removing calibration removes setup-specific correspondence supervision; removing distillation removes the training route for using that knowledge through history alone. Degradation in both variants supports the designs’ roles, but does not establish independent additive effects. Statistical significance from repeated runs or error bars cannot be verified from this table. Appendix D also matters: augmentation probabilities 0, 0.3, and 0.9 underperform the default 0.6, so stronger augmentation is not uniformly better. [§4.5, Tables 4–5; Appendix D, Table 9](https://arxiv.org/html/2609.09155v1#S4.SS5)

## 5. Conclusion

The authors conclude that conditioning on short visual calibration and history can produce controllable predictions under new cameras and single-arm embodiments, rather than forcing one global interpretation of numerical actions. Distillation connects this ability to inference without explicit calibration; candidate ranking demonstrates potential decision-time utility. The paper does not prove improvement on every task or accuracy for every physical interaction. [Original §5](https://arxiv.org/html/2609.09155v1#S5)

## Appendix C.2 and E: runtime and failure cases

**Paper Table 6 — Generation time and compute conditions.** Only the SyncWorld rows are transcribed here.

| GPU configuration | Denoising steps | Generation time |
| --- | --- | --- |
| One H100 | 50 | 39.4 s |
| One H100 | 20 | 15.6 s |
| Four H100s | 20 | 3.2 s |

Runtime also needs its conditions attached. The 3.2 seconds in Table 6 uses four H100 GPUs and 20 denoising steps. One H100 with 20 steps takes 15.6 seconds in the same table. Neither value has been verified as full decision latency including policy inference, evaluation of eight candidates and two views, and the GPT-5 judgment. [SyncWorld Appendix C.2, Table 6](https://arxiv.org/html/2609.09155v1#A3.SS2)

The authors show unstable motion at an extreme camera angle, blur and position errors for unseen objects, hallucination of some objects in multi-object scenes, and blur during precise rotation with a new xArm. Producing video in which the arm follows the command should be distinguished from accurately predicting object interactions. The latter distinction is my criterion for assessing the model. [SyncWorld Appendix E.1–E.4](https://arxiv.org/html/2609.09155v1#A5)

## Review interpretation and open questions

Recent work makes this distinction more concrete. XEWorld evaluates transfer across bimanual embodiments in controlled scenes, separating appearance, kinematics, and object state. The action-format study investigates how equivalent absolute and relative commands can change model outcomes. XEWorld uses different embodiments and action spaces from SyncWorld, while the action-format study uses a fixed camera and offline proxy evaluation. Neither is a matched experiment directly refuting SyncWorld. My proposal from reading them together is to test camera, action-format, and embodiment transfer separately. [XEWorld §3 and §5–6](https://arxiv.org/html/2608.05799v1), [Robot World Models Are Not Invariant to How the Actions Are Written §3–6](https://arxiv.org/html/2609.23252v1)

There are also alternative ways to reduce inference cost. Think Like a World Model, Act Like a VLA distills a teacher's representations into a compact policy and removes the teacher at deployment. Its default teacher supplies understanding representations from Cosmos3-Nano rather than acting as a video-rollout simulator. RoboActualizer trains future and action experts on frozen V-JEPA 2.1 representations, then discards future outputs during execution. Both involve platform-specific training, which differs from SyncWorld's contextual adaptation to a new setup. [Representation distillation §III-B–III-C and §IV](https://arxiv.org/html/2609.24682v2), [RoboActualizer §3–4](https://arxiv.org/html/2609.36413v3)

ReWAM studies representation design in relation to action learning. Its feature calibration differs from SyncWorld's action–video calibration episode. I find it useful to organize this literature around three questions: does the video look plausible, does it change according to the action, and does it help select a better action? This is a synthesis of the literature, not the result of a combined experiment. [ReWAM §3.2–3.4 and §4.3](https://arxiv.org/html/2609.38163v1)

The following are proposed checks and have not been executed.

1. Does calibration remain effective when changing only the camera, only an information-preserving action format, or only the embodiment?
2. Can changes to calibration directions or ordering, together with separate object-interaction error measurements, identify which context the model uses?
3. With matched candidates, judges, and total compute budgets, how do video simulation and representation distillation into policies differ in practical benefit?

## Sources and reading scope

The focal paper is *SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators* by Yuncong Yang and eight coauthors (2026), arXiv:2609.09155v1, first submitted on September 8, 2026. The source text and metadata below were checked on October 6, 2026. The requested PDF could not be read because of the tool's size limit; the official HTML of the same v1 was used instead. Reading covered the focal paper's main text, §1–5, and the appendix portions listed below, not every appendix or supplementary video. [Official metadata](https://arxiv.org/abs/2609.09155v1), [Official HTML](https://arxiv.org/html/2609.09155v1)

The related literature comprises five core papers and five recent representative papers. The recent group prioritizes relevance among sources first submitted between August 1 and October 6, 2026. It is neither a chronological list of the five newest papers nor a list of direct SyncWorld follow-ups. Versions below are the editions actually analyzed.

| Source and version | Actual reading scope |
| --- | --- |
| [SyncWorld v1](https://arxiv.org/html/2609.09155v1) | Main text §1–5; relevant settings, tables, and failure descriptions in Appendix B.1–B.2, C.1–C.4, D–E; original Figures 2, 4 and 5 and their captions; rows of Tables 2, 3, 6 and 8 rechecked |
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

The official GPC v4 title is *Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling*. RoboActualizer v3's RoboTwin evaluation coverage remains unverified because §5 and Appendix C.2 give different descriptions; this article draws no conclusion from that coverage.

Verification covered source text, tables, metadata, and the figures specified below. Complete reading of the related papers, full proof checking, code and weight execution, supplementary-video inspection, simulation and real-robot reproduction, and statistical-significance testing were **not executed (미실행)**. For this revision, the original images and captions of Figures 2, 4 and 5 and the rows of Tables 2, 3, 6 and 8 were inspected directly. The three WebP images are adapted from Yuncong Yang and coauthors (2026), [SyncWorld v1](https://arxiv.org/abs/2609.09155v1), and are reproduced with attribution under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with resizing and WebP compression only; content and layout are preserved without cropping. One reviewer-created schematic explains calibration context and fixed model parameters; it is not an experimental result. Tables transcribe source values into Markdown, converting only success fractions to percentages. The cover uses paper Figure 2. The full-size links open the optimized images; original-resolution figures remain available in the source paper. The schematic SVG retains its vector text and shapes; only its embedded raster images were resized and compressed.

**Structure and evidence recheck, October 22, 2026.** This revision reuses the recorded orientation and detailed reading of October 6. It rechecks v1 HTML §1–5, Tables 1, 4, 5, and 9, and relevant runtime and failure descriptions in Appendix C.2 and E. It is not counted as two newly completed full-paper readings. Section 2’s citation lists are condensed and Appendix F’s supplementary videos remain unviewed; core quality, policy, ablation, and failure evidence is preserved above. Experimental reproduction and statistical testing were not executed.
