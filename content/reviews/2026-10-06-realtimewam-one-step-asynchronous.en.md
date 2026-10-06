---
title: "Fewer Steps, Less Waiting: Understanding RealtimeWAM"
description: "How teacher endpoints compress action generation into one step, and KV-ready events reduce expert waiting. Read aggregate accuracy, perturbation losses, device-specific speedups and training cost together."
date: "2026-10-06"
topics: ["world action model", "robot learning", "policy distillation", "real-time inference", "gpu optimization"]
visibility: "public"
lang: "en"
translationKey: "realtimewam-one-step-asynchronous"
paperTitle: "RealtimeWAM: One-Step Asynchronous World Action Models"
authors: "Chengtao Lv, Jinyang Du, Shuyi Feng, Yang Yong, Shiqiao Gu, Shunzi Yang, Ruihao Gong, Shen Ren, Tianwei Zhang, Wenya Wang"
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.06617v1"
thumbnail: "/assets/reviews/realtimewam/paper-figure-2.png"
thumbnailAlt: "Original Figure 2 showing the TACD teacher, EMA target, student and shared video KV"
---

A world action model can predict useful actions and still deliver them too late for control. RealtimeWAM separately reduces **repeated action denoising** and **waiting between video and action experts**. It reports large H100 speedups, but these combine several execution optimizations and do not establish real-time control on every device or under every perturbation. [§4–5, Figure 5, Table 2](https://arxiv.org/html/2610.06617v1#S4)

## What question does the paper ask?

World action models connect learning about environmental change with action policies. In the Fast-WAM and Faster-WAM backbones studied here, a video expert supplies visual conditioning, and an action expert reads it to generate an **action chunk**, a sequence of future actions. The connected transformer experts form a mixture of transformers, or MoT. [§3](https://arxiv.org/html/2610.06617v1#S3)

Even when the video expert runs once, the action expert usually removes noise repeatedly. After reducing those iterations, waiting for the entire video expert still leaves another bottleneck. One-step generation and starting computation as soon as the required information is available address different problems. [§1, Figure 1, §4](https://arxiv.org/html/2610.06617v1#S4)

<figure class="review-figure" id="figure-realtimewam-core">
<a href="/assets/reviews/realtimewam/realtimewam-core.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/realtimewam-core.svg" width="1200" height="1010" alt="Top: the frozen ten-step teacher supplies an endpoint target to a one-step action student sharing the same noisy action and frozen video KV. Bottom: one conditioning block records KV readiness after projection and preprocessing, with a wait immediately before action attention." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Concept guide · Learn fewer steps, schedule less waiting</span>The teacher is used during training only, and the local consistency loss is retained. CEWP can overlap projections, but action attention must wait for its required KV. Line lengths are not measured durations or an execution trace. Sparse conditioning also requires KV fusion to finish. Original explanatory diagram made for this review. <span class="figure-links"><a href="/assets/reviews/realtimewam/realtimewam-core.svg" target="_blank" rel="noopener noreferrer">Full size</a></span></figcaption>
</figure>

The preceding [SyncWorld review](https://iiamaii.github.io/en/reviews/syncworld-visual-calibration/) concerned a simulator that predicts future video from supplied actions. This paper concerns efficient inference for a policy that directly generates actions using world information. Both belong under `world action model`; their architectures and evaluation roles remain different.

## Core idea: local agreement and endpoint accuracy

Teacher-Anchored Consistency Distillation, or TACD, trains an action student to approach a multi-step teacher endpoint in one evaluation. **Consistency distillation** aligns endpoint predictions from nearby noisy states on the same trajectory. In finite training, however, agreement between two predictions need not mean that either approaches the teacher's correct endpoint. RealtimeWAM explains this through local and global residuals. This observation does not refute consistency-model theory in general. [§4.1, Eq.6, Figure 3, Appendix C.1](https://arxiv.org/html/2610.06617v1#S4.SS1), [Consistency Models §3–4](https://arxiv.org/html/2303.01469v2#S3)

<figure class="review-figure" id="figure-realtimewam-training">
<a href="/assets/reviews/realtimewam/paper-figure-2.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-2.png" width="1283" height="428" alt="Original TACD diagram: a frozen video expert supplies shared KV to teacher rollout, EMA target and action student, with local-consistency and teacher-anchor losses." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Original Figure 2 · Two targets to align</span>Read from the noisy action on the right toward the clean action on the left. The green teacher rollout supplies the endpoint reference. Local consistency aligns the blue EMA and red student predictions; a separate teacher-anchor loss constrains the remaining endpoint deviation. Snowflakes mark frozen modules and the flame marks training. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-2.png" target="_blank" rel="noopener noreferrer">Full size</a> · <a href="https://arxiv.org/html/2610.06617v1#S4.F2">Chengtao Lv et al., 2026, Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original bytes; no editing or cropping</span></figcaption>
</figure>

The teacher starts from the **same noisy action and frozen video KV** as the student and takes ten steps. KV denotes the key/value representations read by attention. Its total displacement to the endpoint, divided by the current noise time, supplies an average velocity target. This differs from directly copying the teacher's instantaneous velocity at the current time. The local consistency term remains in the objective. [§4.1, Eq.7–8](https://arxiv.org/html/2610.06617v1#S4.SS1)

```text
f_S(a_t,t) = a_t - t v_S(a_t,t)
a0_T = Solver(a_t,t,0; teacher)    # K=10 steps
u_T = (a_t - a0_T)/t              # t>0
L_TA = E ||v_S - stop_gradient(u_T)||²
L = L_CD + 0.2 L_TA
```

`a_t` is a noisy action; time `t` is clean at zero and noise at one. `v_S` is the student velocity prediction, `f_S` its clean endpoint prediction, `a0_T` the teacher's numerical rollout endpoint, and `u_T` the corresponding interval-average velocity. `L_CD` is the local consistency loss and `L_TA` the teacher-anchor loss. The video expert is frozen and only action-expert LoRA parameters are trained. The teacher and EMA target are not called at deployment. [§4.1, §5.1](https://arxiv.org/html/2610.06617v1#S5.SS1)

The target is a teacher action, rather than a verified optimal action in the environment. Appendix C's endpoint-error bound also includes numerical integration error; it is not a task-success guarantee. Teacher budgets of 5, 10 and 20 give RoboTwin overall success rates of 90.39, 90.84 and 90.77%, respectively. More teacher computation does not invariably improve success. [Eq.9, Appendix C.2–C.3, Table 4](https://arxiv.org/html/2610.06617v1#A3)

## Method and assumptions: wait immediately before KV use

Cross-Expert Wavefront Pipelining, or CEWP, overlaps video and action computation on two nonblocking CUDA streams on one GPU. A video block records a readiness event after projection and required KV preprocessing. The action projection can proceed once its own hidden state is available, then attention waits for the event immediately before reading the video KV. Meanwhile, the video stream continues its remaining work. [§4.2, Figure 4, Appendix D.1](https://arxiv.org/html/2610.06617v1#S4.SS2)

<figure class="review-figure" id="figure-realtimewam-pipeline">
<a href="/assets/reviews/realtimewam/paper-figure-4.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-4.svg" width="1137" height="581" alt="Original comparison of sequential expert execution, two-stream CEWP with KV-ready events and waits before action attention, and the dependency graph within a block." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Original Figure 4 · From a full-cache barrier to block events</span>The top schedule starts the action expert after the whole video expert finishes. The lower schedule publishes each ready KV block and waits when action attention needs it. Read the projection→attention→remainder dependencies on the right first: the blocks are not independent. The time axis illustrates execution order. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-4.svg" target="_blank" rel="noopener noreferrer">Full size</a> · <a href="https://arxiv.org/html/2610.06617v1#S4.F4">Chengtao Lv et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original bytes; no editing or cropping</span></figcaption>
</figure>

This preserves the original operators and block dependencies. It does not make all blocks independent. Sparse interaction in Faster-WAM also requires synchronization at its selected conditioning stages, after KV fusion is complete. Appendix D describes equivalence under exact arithmetic when dependencies are preserved; this review did not test GPU bitwise identity. [§4.2, Appendix D.1–D.2](https://arxiv.org/html/2610.06617v1#A4)

The **Faster-WAM here is arXiv:2608.04404**, the future-conditioning model. The identically named 2608.02365 studies Depth-of-Thought and is a separate paper. `*` denotes Fast-WAM-based RealtimeWAM and `†` this future-conditioning Faster-WAM-based version. [RealtimeWAM §3, Table 1](https://arxiv.org/html/2610.06617v1#S3), [Faster-WAM Method](https://arxiv.org/html/2608.04404v1#Sx3)

## Results: inspect averages and subsets together

**Eight relevant rows from original Table 1.** Success rates are percentages; higher is better. NFE counts denoising function evaluations separately for video and action. RoboTwin overall averages clean/random conditions, while LIBERO averages its four suites. The naive one-step row illustrates why simply reducing the teacher's iteration count is insufficient. [Table 1, Appendix F.1](https://arxiv.org/html/2610.06617v1#S5.T1)

| Model | Video NFE | Action NFE | RoboTwin Clean ↑ | Random ↑ | Overall ↑ | LIBERO ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Fast-WAM | 1 | 10 | 91.82 | 91.19 | 91.51 | 97.0 |
| Fast-WAM, naive one-step | 1 | 1 | 70.42 | 68.12 | 69.27 | 95.0 |
| CD* | 1 | 1 | 89.88 | 89.38 | 89.63 | 96.9 |
| MeanFlow* | 1 | 1 | 85.74 | 85.72 | 85.73 | 96.9 |
| DMD* | 1 | 1 | 88.64 | 87.80 | 88.22 | 96.1 |
| RealtimeWAM* | 1 | 1 | 91.96 | 89.72 | 90.84 | 97.0 |
| Faster-WAM | 1 | 10 | 93.20 | 92.66 | 92.93 | 98.9 |
| RealtimeWAM† | 1 | 1 | 92.98 | 92.30 | 92.64 | 99.0 |

RealtimeWAM* loses **0.67 percentage points(pp)** of RoboTwin overall relative to Fast-WAM; † loses **0.29 pp** relative to Faster-WAM. LIBERO is unchanged for * and increases by 0.1 pp for †. These small average differences do not establish statistical equivalence or preservation on every task. Some baseline scores are cited from preceding papers rather than freshly reproduced in one common run. [§5.3, Table 1](https://arxiv.org/html/2610.06617v1#S5.SS3)

The trained experts must also match when isolating the anchor's effect. Unlike CD* in Table 1, Table 3 compares **frozen-video, action-only training**: CD 89.85%→TACD 90.84%, an increase of 0.99 pp. Its random subset nevertheless falls from 89.94 to 89.72%. This is not improvement under every condition. [Table 3](https://arxiv.org/html/2610.06617v1#S5.T4)

**All perturbation columns for Faster and RealtimeWAM† from original Table 2.** LIBERO-Plus evaluates camera, robot, language, lighting and other shifts. The final pp column is subtraction performed for this review. Overall is weighted by evaluation trial counts; it is not the arithmetic mean of the seven printed category rates. [Table 2, Appendix F.1](https://arxiv.org/html/2610.06617v1#S5.T2)

| Perturbation | Faster-WAM | RealtimeWAM† | Change (pp) |
| --- | --- | --- | --- |
| Camera | 53.8 | 54.2 | +0.4 |
| Robot | 71.6 | 69.0 | −2.6 |
| Language | 94.7 | 92.5 | −2.2 |
| Lighting | 96.3 | 94.7 | −1.6 |
| Background | 61.3 | 59.4 | −1.9 |
| Noise | 63.6 | 64.6 | +1.0 |
| Layout | 79.1 | 79.3 | +0.2 |
| Overall | 73.6 | 73.0 | −0.6 |

The overall loss is 0.6 pp, but the robot category loses 2.6 pp and language loses 2.2 pp. A below-one-percent aggregate summary therefore cannot be applied to every perturbation. The exact focal evaluation rollout counts, repeated seeds and confidence intervals could not be established from the checked material. The predecessor papers' trial counts were not automatically assigned to RealtimeWAM's own evaluations.

## Speed and training cost: where does 25× come from?

<figure class="review-figure" id="figure-realtimewam-speed">
<a href="/assets/reviews/realtimewam/paper-figure-5.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-5.svg" width="1152" height="263" alt="H100 latency bars for Fast-WAM and Faster-WAM after adding TACD, CUDA Graph, CEWP and efficient kernels, with cumulative and incremental speedups." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Original Figure 5 · Read the cumulative speedup</span>Read the Fast and Faster panels separately, adding one optimization at a time from top to bottom. The horizontal axis is latency in milliseconds; lower is faster. Blue speedup labels are cumulative relative to the initial baseline; connector labels are gains over the preceding stage. The final roughly 25× is not the effect of TACD or CEWP alone. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-5.svg" target="_blank" rel="noopener noreferrer">Full size</a> · <a href="https://arxiv.org/html/2610.06617v1#S5.F5">Chengtao Lv et al., 2026, Figure 5</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original bytes; no editing or cropping</span></figcaption>
</figure>

On H100, Fast-WAM goes from 299.7 ms to 65.8 ms with TACD, 23.3 ms after CUDA Graph, 17.4 ms after CEWP, and 12.2 ms after efficient kernels. The Faster sequence is 218.9→57.1→26.2→22.4→16.1 ms. The reported cumulative speedups are 24.55× and 13.56×. CEWP adds about 1.34× and 1.17× after CUDA Graph. [§5.4, Figure 5](https://arxiv.org/html/2610.06617v1#S5.F5)

**Baseline→final latency in milliseconds, from original Figure 5(H100) and Table F.3(additional GPUs).** The measurement includes VAE encoding and excludes text encoding cached once per episode. It is not sensor-to-motor end-to-end latency. [Figure 5](https://arxiv.org/html/2610.06617v1#S5.F5), [Appendix F.3](https://arxiv.org/html/2610.06617v1#A6.SS3)

| Backbone | H100 | RTX 5090 | RTX 4090D |
| --- | --- | --- | --- |
| Fast → RealtimeWAM* | 299.7→12.2 | 251.5→17.8 | 480.8→27.2 |
| Faster → RealtimeWAM† | 218.9→16.1 | 214.1→30.9 | 362.4→44.7 |

The 12.2 ms H100 call fits within a 30 Hz per-call budget of roughly 33.3 ms. That alone does not establish a 30 Hz closed loop. The† version's 44.7 ms on RTX 4090D exceeds even that per-call budget. The need to inspect device-specific gains, p 95/p 99 latency and observation age is this review's interpretation.

**Training cost from original Table A.1**, using 16 H100 GPUs and 30,000 iterations. Distinguish the memory effect of freezing video from the extra time spent computing teacher anchors. [Appendix A](https://arxiv.org/html/2610.06617v1#A1)

| Training setup | Time | Peak memory |
| --- | --- | --- |
| CD, video + action tuned | 8 h 1 m | 65.98 GiB |
| CD, action only | 5 h 21 m | 24.73 GiB |
| TACD, action only | 8 h 3 m | 24.73 GiB |

Action-only CD and TACD have the same reported peak memory, but training time rises from 5 h 21 m to 8 h 3 m, approximately 50.5%. Teacher rollout computation is paid during training to reduce deployment cost. A real-robot garment-folding rollout uses dual-arm AgileX PiPER with RTX 4090. This is qualitative evidence rather than a common-trial success-rate or closed-loop frequency assessment. [Table A.1, Appendix F.6, Figure F.1](https://arxiv.org/html/2610.06617v1#A6.SS6)

## Questions raised by related work

The five core papers are the direct **Fast-WAM and Faster-WAM** backbones, **Consistency Models and MeanFlow** for one-step learning, and **RTC** for execution continuity. First distinguish future information use in the two backbones, then interval-average velocity from endpoint supervision, and chunk execution from waiting inside a forward pass. Their sources and reading scope appear below.

The five recent representative papers address other cost axes. **DIDO** compresses interaction-focused video tokens and video denoising: its one-step title does not imply one action evaluation. **AnyStep-WAM** selects observation-dependent budgets instead of fixing one step, but its risk proxy is not calibrated physical failure probability. **OpenWAM** studies world/action noise-state synchronization, a different layer from CEWP's CUDA scheduling; bidirectional architectures require rechecking independent KV precomputation. **ReWAM** changes the action-relevant representation, while **WAM-OPD** supervises histories visited by the student. Different hardware and task subsets were not combined into a common ranking. [DIDO §3](https://arxiv.org/html/2609.15570v2#S3), [AnyStep §4](https://arxiv.org/html/2609.33748v2#S4), [OpenWAM §3–4](https://arxiv.org/pdf/2609.07398v1), [ReWAM §3](https://arxiv.org/html/2609.38163v1#S3), [WAM-OPD §3–4](https://arxiv.org/html/2608.22364v2#S3)

## My interpretation and remaining questions

The useful perspective is to separate iteration count from execution dependencies instead of explaining efficiency solely through model size. Figure 5's staged speedups and Table 3's controlled training comparison help guide implementation. However, approaching a teacher endpoint does not establish that the teacher is right in an unfamiliar state. Questions about student visitation and difficult contacts from WAM-OPD and AnyStep remain relevant. This is review interpretation, not a tested result for their combination.

I recommend first measuring sequential/CEWP output differences, event traces and p 95/p 99 latency on the same GPU. Subsequent work could compare fixed-one and adaptive budgets at matched average compute, and test whether combining RTC improves observation freshness. These validations, code/GPU/real-robot reproduction, statistical significance checks and supplementary-video review are **not run**.

## Sources and reading scope

- Focal: Chengtao Lv et al., *RealtimeWAM: One-Step Asynchronous World Action Models*,2026. [arXiv:2610.06617 v1](https://arxiv.org/abs/2610.06617v1), first submitted 2026-10-05, checked 2026-10-06. Main §1–6 and selected Appendices A, C.1–C.3, D.1–D.2, E, F.1/F.3/F.6. The 20-page PDF was accessed and text-extracted; Figures 2, 4, 5 and PDF pp.5, 6, 9 were visually inspected. This does not mean every appendix or proof was independently audited.
- Core: [Fast-WAM v2](https://arxiv.org/html/2603.16666v2), §3.1–3.3, §4.1–4.2 setup, §5; [Faster-WAM v1](https://arxiv.org/html/2608.04404v1), relevant Method and Experiment Setup/Main Results passages; [Consistency Models v2](https://arxiv.org/html/2303.01469v2), §3–4; [Mean Flows v1](https://arxiv.org/html/2505.13447v1), §3, §4.1, initial §5 setup; [RTC v2](https://arxiv.org/html/2506.07339v2), relevant §3 method, §4 setup, §6 limitations.
- Recent representatives: [DIDO v2](https://arxiv.org/html/2609.15570v2), §3.1–3.4 and initial §4 setup/Table 1 discussion; [AnyStep-WAM v2](https://arxiv.org/html/2609.33748v2), §4.1–4.3 and §5.1/Table 1; [OpenWAM v1](https://arxiv.org/pdf/2609.07398v1), selected passages on PDF pp.1, 5, 8,13–16,33; [ReWAM v1](https://arxiv.org/html/2609.38163v1), abstract and §3.2–3.4; [WAM-OPD v2](https://arxiv.org/html/2608.22364v2), §3.1–3.3, §4.1–4.2/Table 2, §5. The recent window is first submission 2026-08-01 through 10-06. These are five relevant representatives, not the five latest papers overall.
- Original Figures 2, 4,5: Chengtao Lv et al.,2026, [same-version source](https://arxiv.org/html/2610.06617v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original bytes were retained with no editing or cropping. Only the opening concept diagram was made for this review.
- Numerical results are author-reported; independent reproduction and separate native-English proofreading were not run.
