---
title: "Fewer Steps, Less Waiting: Understanding RealtimeWAM"
description: "How teacher endpoints compress action generation into one step, and KV-ready events reduce expert waiting. Read aggregate accuracy, perturbation losses, device-specific speedups and training cost together."
date: "2026-10-06"
updatedAt: "2026-10-22T23:22:43+09:00"
topics: ["world action model", "robot learning", "policy distillation", "real-time inference", "gpu optimization"]
visibility: "public"
lang: "en"
translationKey: "realtimewam-one-step-asynchronous"
paperTitle: "RealtimeWAM: One-Step Asynchronous World Action Models"
paperPublishedDate: "2026-10-05"
authors: ["Chengtao Lv", "Jinyang Du", "Shuyi Feng", "Yang Yong", "Shiqiao Gu", "Shunzi Yang", "Ruihao Gong", "Shen Ren", "Tianwei Zhang", "Wenya Wang"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.06617v1"
thumbnail: "/assets/reviews/realtimewam/paper-figure-2.png"
thumbnailAlt: "Original Figure 2 showing the TACD teacher, EMA target, student and shared video KV"
---

A world action model can predict useful actions and still deliver them too late for control. RealtimeWAM separately reduces **repeated action denoising** and **waiting between video and action experts**. It reports large H100 speedups, but these combine several execution optimizations and do not establish real-time control on every device or under every perturbation. [§4–5, Figure 5, Table 2](https://arxiv.org/html/2610.06617v1#S4)

**Notation.** The equations use the paper’s $f_{\theta_{\mathrm S}}$, $v_{\theta_{\mathrm S}}$, $a_0^{\mathrm T}$, $u_{\theta_{\mathrm T}}$ and $\mathcal L_{\mathrm{TA}}$. $\operatorname{sg}$ denotes stop-gradient. The exact ODE endpoint $a_0^\star$ is distinct from the numerical teacher endpoint $a_0^{\mathrm T}$.

This article follows original §1–6. Appendix C’s error derivation is expanded immediately after §4.1; training cost and real-robot demonstrations are grouped under their appendices after the conclusion.

## 1. Introduction

World action models connect learning about environmental change with action policies. In the Fast-WAM and Faster-WAM backbones studied here, a video expert supplies visual conditioning, and an action expert reads it to generate an **action chunk**, a sequence of future actions. The connected transformer experts form a mixture of transformers, or MoT. [§3](https://arxiv.org/html/2610.06617v1#S3)

Even when the video expert runs once, the action expert usually removes noise repeatedly. After reducing those iterations, waiting for the entire video expert still leaves another bottleneck. One-step generation and starting computation as soon as the required information is available address different problems. [§1, Figure 1, §4](https://arxiv.org/html/2610.06617v1#S4)

<figure class="review-figure" id="figure-realtimewam-core">
<a href="/assets/reviews/realtimewam/realtimewam-core.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/realtimewam-core.svg" width="1200" height="1010" alt="Top: the frozen ten-step teacher supplies an endpoint target to a one-step action student sharing the same noisy action and frozen video KV. Bottom: one conditioning block records KV readiness after projection and preprocessing, with a wait immediately before action attention." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Concept guide · Learn fewer steps, schedule less waiting</span>The teacher is used during training only, and the local consistency loss is retained. CEWP can overlap projections, but action attention must wait for its required KV. Line lengths are not measured durations or an execution trace. Sparse conditioning also requires KV fusion to finish. Original explanatory diagram made for this review. <span class="figure-links"><a href="/assets/reviews/realtimewam/realtimewam-core.svg" target="_blank" rel="noopener noreferrer">Full size</a></span></figcaption>
</figure>

The preceding [SyncWorld review](https://iiamaii.github.io/en/reviews/syncworld-visual-calibration/) concerned a simulator that predicts future video from supplied actions. This paper concerns efficient inference for a policy that directly generates actions using world information. Both belong under `world action model`; their architectures and evaluation roles remain different.

## 2. Related Work

Section 2 separates WAMs, few-step distillation, and asynchronous robot-policy inference. The first two concern generation models and training; the third concerns timing between generation and execution. CEWP specifically reduces waiting between experts inside one forward pass. [Original §2](https://arxiv.org/html/2610.06617v1#S2)

The five core papers are the direct **Fast-WAM and Faster-WAM** backbones, **Consistency Models and MeanFlow** for one-step learning, and **RTC** for execution continuity. First distinguish future information use in the two backbones, then interval-average velocity from endpoint supervision, and chunk execution from waiting inside a forward pass. Their sources and reading scope appear below.

The five recent representative papers address other cost axes. **DIDO** compresses interaction-focused video tokens and video denoising: its one-step title does not imply one action evaluation. **AnyStep-WAM** selects observation-dependent budgets instead of fixing one step, but its risk proxy is not calibrated physical failure probability. **OpenWAM** studies world/action noise-state synchronization, a different layer from CEWP's CUDA scheduling; bidirectional architectures require rechecking independent KV precomputation. **ReWAM** changes the action-relevant representation, while **WAM-OPD** supervises histories visited by the student. Different hardware and task subsets were not combined into a common ranking. [DIDO §3](https://arxiv.org/html/2609.15570v2#S3), [AnyStep §4](https://arxiv.org/html/2609.33748v2#S4), [OpenWAM §3–4](https://arxiv.org/pdf/2609.07398v1), [ReWAM §3](https://arxiv.org/html/2609.38163v1#S3), [WAM-OPD §3–4](https://arxiv.org/html/2608.22364v2#S3)

## 3. Preliminaries

In Eqs. (1)–(2), observation $o$ and instruction $l$ become blockwise video KV conditioning $z(o,l)$. Here $N$ is the number of video transformer blocks, $a$ an action chunk, and $\psi,\theta$ the video and action parameters. Computing visual conditioning once can still leave many denoising evaluations to sample actions.

### 3.1 Efficient Mixture-of-Transformers (MoT)-based WAMs

$$
\begin{aligned}
z(o,l)&=F_{\psi}^{\mathrm{KV}}(o,l)=\{(K_i^v,V_i^v)\}_{i=1}^{N},\\
p_{\theta}(a\mid o,l)&=p_{\theta}(a\mid z(o,l)).
\end{aligned}
$$

### 3.2 Consistency Distillation for Action Generation

$$
\begin{aligned}
a_t&=(1-t)a_0^{\mathrm{data}}+t\epsilon,\quad\epsilon\sim\mathcal N(0,I),\\
f_\theta(a_t,t)&=a_t-t\,v_\theta(a_t,t),\\
\tilde a_s&=\texttt{Solver}(a_t,t,s;\theta_{\mathrm T}),\quad 0\leq s<t,\\
\mathcal L_{\mathrm{CD}}&=\mathbb E_{a_t,t,s}\!\left[
\left\|f_{\theta_{\mathrm S}}(a_t,t)-\operatorname{sg}\!\left[
f_{\theta_{\mathrm{ema}}}(\tilde a_s,s)\right]\right\|_2^2\right].
\end{aligned}
$$

In Eqs. (3)–(5), clean data actions $a_0^{\mathrm{data}}$ and Gaussian noise $\epsilon$ form $a_t$. The teacher takes one integration step to a neighboring $s<t$, producing $\tilde a_s$. Training aligns student and EMA-target endpoint predictions. Here $v_\theta$ predicts velocity and $f_\theta$ predicts a clean endpoint. The operator $\operatorname{sg}$ blocks gradients; $\theta_{\mathrm{ema}}$ is an exponential moving average of student parameters. EMA tracks the student; it is not an oracle for optimal environmental actions. [§3.1–3.2, Eqs. (1)–(5)](https://arxiv.org/html/2610.06617v1#S3)

## 4. RealtimeWAM

TACD and CEWP address the two bottlenecks.

### 4.1 Teacher-Anchored Consistency Distillation

Teacher-Anchored Consistency Distillation, or TACD, trains an action student to approach a multi-step teacher endpoint in one evaluation. **Consistency distillation** aligns endpoint predictions from nearby noisy states on the same trajectory. In finite training, however, agreement between two predictions need not mean that either approaches the teacher's correct endpoint. RealtimeWAM explains this through local and global residuals. This observation does not refute consistency-model theory in general. [§4.1, Eq.6, Figure 3, Appendix C.1](https://arxiv.org/html/2610.06617v1#S4.SS1), [Consistency Models §3–4](https://arxiv.org/html/2303.01469v2#S3)

<figure class="review-figure" id="figure-realtimewam-training">
<a href="/assets/reviews/realtimewam/paper-figure-2.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-2.png" width="1283" height="428" alt="Original TACD diagram: a frozen video expert supplies shared KV to teacher rollout, EMA target and action student, with local-consistency and teacher-anchor losses." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Original Figure 2 · Two targets to align</span>Read from the noisy action on the right toward the clean action on the left. The green teacher rollout supplies the endpoint reference. Local consistency aligns the blue EMA and red student predictions; a separate teacher-anchor loss constrains the remaining endpoint deviation. Snowflakes mark frozen modules and the flame marks training. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-2.png" target="_blank" rel="noopener noreferrer">Full size</a> · <a href="https://arxiv.org/html/2610.06617v1#S4.F2">Chengtao Lv et al., 2026, Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original bytes; no editing or cropping</span></figcaption>
</figure>

The teacher starts from the **same noisy action and frozen video KV** as the student and takes ten steps. KV denotes the key/value representations read by attention. Its total displacement to the endpoint, divided by the current noise time, supplies an average velocity target. This differs from directly copying the teacher's instantaneous velocity at the current time. The local consistency term remains in the objective. [§4.1, Eq.7–8](https://arxiv.org/html/2610.06617v1#S4.SS1)

$$
\begin{aligned}
f_{\theta_{\mathrm S}}(a_t,t)&=a_t-t\,v_{\theta_{\mathrm S}}(a_t,t),\\
a_0^{\mathrm T}&=\texttt{Solver}(a_t,t,0;\theta_{\mathrm T}),\\
u_{\theta_{\mathrm T}}(a_t,t)&=\frac{a_t-a_0^{\mathrm T}}{t},\quad t>0,\\
\mathcal L_{\mathrm{TA}}&=\mathbb E_{a_t,t}\!\left[\left\|v_{\theta_{\mathrm S}}(a_t,t)-\operatorname{sg}\!\left[u_{\theta_{\mathrm T}}(a_t,t)\right]\right\|_2^2\right],\\
\mathcal L&=\mathcal L_{\mathrm{CD}}+\lambda\mathcal L_{\mathrm{TA}},\quad \lambda=0.2.
\end{aligned}
$$

$a_t$ is a noisy action; time $t$ is clean at zero and noise at one. $v_{\theta_{\mathrm S}}$ is the student velocity prediction, $f_{\theta_{\mathrm S}}$ its clean endpoint prediction, $a_0^{\mathrm T}$ the teacher's numerical rollout endpoint, and $u_{\theta_{\mathrm T}}$ the corresponding interval-average velocity. $\mathcal L_{\mathrm{CD}}$ is the local consistency loss and $\mathcal L_{\mathrm{TA}}$ the teacher-anchor loss. The video expert is frozen and only action-expert LoRA parameters are trained. The teacher and EMA target are not called at deployment. [§4.1, §5.1](https://arxiv.org/html/2610.06617v1#S5.SS1)

The target is a teacher action, rather than a verified optimal action in the environment. Appendix C's endpoint-error bound also includes numerical integration error; it is not a task-success guarantee. Teacher budgets of 5, 10 and 20 give RoboTwin overall success rates of 90.39, 90.84 and 90.77%, respectively. More teacher computation does not invariably improve success. [Eq.9, Appendix C.2–C.3, Table 4](https://arxiv.org/html/2610.06617v1#A3)

#### Appendix C.1–C.3 beside the method: what the endpoint bound guarantees

This explanation relocates Appendix C next to its method. Fix shared conditioning $z(o,l)$ and the training sampling distribution; assume a unique teacher ODE solution and finite second moments of errors and velocity residuals. Adding and subtracting the EMA prediction gives Eq. (6). Expanding the square retains a cross term, so independence must not be assumed.

$$
\begin{aligned}
\mathbf e_{\mathrm{total}}&=\mathbf e_{\mathrm{local}}+\mathbf e_{\mathrm{global}},\\
\mathbb E\|\mathbf e_{\mathrm{total}}\|_2^2
&=\mathbb E\|\mathbf e_{\mathrm{local}}\|_2^2+\mathbb E\|\mathbf e_{\mathrm{global}}\|_2^2
+2\mathbb E\langle\mathbf e_{\mathrm{local}},\mathbf e_{\mathrm{global}}\rangle.
\end{aligned}
$$

Cauchy–Schwarz bounds the cross term by the product of the root mean-squared errors. In particular, zero local error leaves the global error unchanged. This motivates additional supervision; it does not invalidate every theoretical property of CD.

Appendix C.2 then gives an exact identity. The common $a_t$ cancels, leaving the velocity residual multiplied by $-t$. For $t>0$, squaring and dividing by $t^2$ shows that the velocity loss is an **endpoint loss weighted by $1/t^2$**.

$$
\begin{aligned}
f_{\theta_{\mathrm S}}(a_t,t)-a_0^{\mathrm T}
&=-t\left[v_{\theta_{\mathrm S}}(a_t,t)-u_{\theta_{\mathrm T}}(a_t,t)\right],\\
\mathcal L_{\mathrm{TA}}
&=\mathbb E\!\left[\frac{\|f_{\theta_{\mathrm S}}(a_t,t)-a_0^{\mathrm T}\|_2^2}{t^2}\right].
\end{aligned}
$$

The errors follow Eq. (6): $\mathbf e_{\mathrm{local}}=f_{\theta_{\mathrm S}}(a_t,t)-f_{\theta_{\mathrm{ema}}}(\tilde a_s,s)$ and $\mathbf e_{\mathrm{global}}=f_{\theta_{\mathrm{ema}}}(\tilde a_s,s)-a_0^\star$. Velocity arguments $(a_t,t)$ are omitted inside the norm below for compactness.

Appendix C.3 adds $\boldsymbol\delta=a_0^{\mathrm T}-a_0^\star$. The inequality $\|p+q\|^2\leq2\|p\|^2+2\|q\|^2$ follows from Cauchy–Schwarz and $2ab\leq a^2+b^2$. The last line uses $0<t\leq1$.

$$
\begin{aligned}
\mathbf e_{\mathrm{total}}&=-t\left[v_{\theta_{\mathrm S}}(a_t,t)-u_{\theta_{\mathrm T}}(a_t,t)\right]+\boldsymbol\delta,\\
\mathbb E\|\mathbf e_{\mathrm{total}}\|_2^2
&\leq 2\mathbb E\!\left[t^2\|v_{\theta_{\mathrm S}}-u_{\theta_{\mathrm T}}\|_2^2\right]+2\mathbb E\|\boldsymbol\delta\|_2^2\\
&\leq 2\mathcal L_{\mathrm{TA}}+2\mathbb E\|\boldsymbol\delta\|_2^2.
\end{aligned}
$$

Teacher integration error remains separate. If its velocity is $L$-Lipschitz in state and the exact trajectory has second time derivative bounded in norm by $M$, a uniform Euler step $h=t/K$ has local error at most $Mh^2/2$. Here $D_k$ is the distance between numerical and exact states after step $k$. The geometric-series expansion below is a **review-supplied intermediate calculation**, retaining Appendix C.3’s order.

$$
\begin{aligned}
D_{k+1}&\leq(1+Lh)D_k+\frac{Mh^2}{2},\qquad D_0=0,\quad h=\frac{t}{K},\\
\|\boldsymbol\delta\|_2=D_K
&\leq\frac{Mh^2}{2}\sum_{j=0}^{K-1}(1+Lh)^j\\
&=\frac{Mh}{2L}\left[(1+Lh)^K-1\right]
\leq\frac{Me^{Lt}t^2}{2K}\quad(L>0).
\end{aligned}
$$

The final inequality uses $(1+Lh)^K\leq e^{Lt}$ and $e^{Lt}-1\leq Lt e^{Lt}$. For $L=0$, direct summation gives $D_K\leq Mt^2/(2K)$. Uniform $L,M$ yield an $O(K^{-2})$ squared numerical-error term, not monotonic task success for finite $K$. The target is the exact **teacher ODE endpoint**, distinct from an optimal action in the environment. [Appendix C.1–C.3, Eqs. (C.1)–(C.4)](https://arxiv.org/html/2610.06617v1#A3)

### 4.2 Cross-Expert Wavefront Pipelining

Cross-Expert Wavefront Pipelining, or CEWP, overlaps video and action computation on two nonblocking CUDA streams on one GPU. A video block records a readiness event after projection and required KV preprocessing. The action projection can proceed once its own hidden state is available, then attention waits for the event immediately before reading the video KV. Meanwhile, the video stream continues its remaining work. [§4.2, Figure 4, Appendix D.1](https://arxiv.org/html/2610.06617v1#S4.SS2)

<figure class="review-figure" id="figure-realtimewam-pipeline">
<a href="/assets/reviews/realtimewam/paper-figure-4.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-4.svg" width="1137" height="581" alt="Original comparison of sequential expert execution, two-stream CEWP with KV-ready events and waits before action attention, and the dependency graph within a block." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Original Figure 4 · From a full-cache barrier to block events</span>The top schedule starts the action expert after the whole video expert finishes. The lower schedule publishes each ready KV block and waits when action attention needs it. Read the projection→attention→remainder dependencies on the right first: the blocks are not independent. The time axis illustrates execution order. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-4.svg" target="_blank" rel="noopener noreferrer">Full size</a> · <a href="https://arxiv.org/html/2610.06617v1#S4.F4">Chengtao Lv et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original bytes; no editing or cropping</span></figcaption>
</figure>

This preserves the original operators and block dependencies. It does not make all blocks independent. Sparse interaction in Faster-WAM also requires synchronization at its selected conditioning stages, after KV fusion is complete. Appendix D describes equivalence under exact arithmetic when dependencies are preserved; this review did not test GPU bitwise identity. [§4.2, Appendix D.1–D.2](https://arxiv.org/html/2610.06617v1#A4)

The **Faster-WAM here is arXiv:2608.04404**, the future-conditioning model. The identically named 2608.02365 studies Depth-of-Thought and is a separate paper. `*` denotes Fast-WAM-based RealtimeWAM and `†` this future-conditioning Faster-WAM-based version. [RealtimeWAM §3, Table 1](https://arxiv.org/html/2610.06617v1#S3), [Faster-WAM Method](https://arxiv.org/html/2608.04404v1#Sx3)

## 5. Experiments

Success rates and runtime are evaluated separately.

### 5.1 Implementation Details; 5.2 Evaluation Benchmarks

The three evaluation axes are clean/random manipulation in RoboTwin, four LIBERO suites, and seven LIBERO-Plus perturbation categories. Success rate is the fraction of evaluation rollouts judged successful; higher is better. NFE and latency measure computational cost with different units. Training freezes the video expert and adapts action LoRA; TACD uses a ten-step teacher. This does not mean every baseline was retrained at a matched budget. The tables below retain aggregation rules and costs. [§5.1–5.2, Appendix F.1](https://arxiv.org/html/2610.06617v1#S5.SS1)

### 5.3 Main Results

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

### 5.4 Ablation Study and latency analysis

The trained experts must also match when isolating the anchor's effect. Unlike CD* in Table 1, Table 3 compares **frozen-video, action-only training**: CD 89.85%→TACD 90.84%, an increase of 0.99 pp. Its random subset nevertheless falls from 89.94 to 89.72%. This is not improvement under every condition. [Table 3](https://arxiv.org/html/2610.06617v1#S5.T4)



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

## 6. Conclusion

The authors combine one-step action denoising through TACD with dependency-preserving blockwise overlap through CEWP. Experiments show substantially lower latency while largely retaining aggregate quality. They do not prove calibrated real-robot failure probabilities or complete control-loop frequencies for every device. [Original §6](https://arxiv.org/html/2610.06617v1#S6)

## Appendix A and F.6: training cost and real-robot demonstration

**Training cost from original Table A.1**, using 16 H100 GPUs and 30,000 iterations. Distinguish the memory effect of freezing video from the extra time spent computing teacher anchors. [Appendix A](https://arxiv.org/html/2610.06617v1#A1)

| Training setup | Time | Peak memory |
| --- | --- | --- |
| CD, video + action tuned | 8 h 1 m | 65.98 GiB |
| CD, action only | 5 h 21 m | 24.73 GiB |
| TACD, action only | 8 h 3 m | 24.73 GiB |

Action-only CD and TACD have the same reported peak memory, but training time rises from 5 h 21 m to 8 h 3 m, approximately 50.5%. Teacher rollout computation is paid during training to reduce deployment cost. A real-robot garment-folding rollout uses dual-arm AgileX PiPER with RTX 4090. This is qualitative evidence rather than a common-trial success-rate or closed-loop frequency assessment. [Table A.1, Appendix F.6, Figure F.1](https://arxiv.org/html/2610.06617v1#A6.SS6)

## Review interpretation and open questions

The useful perspective is to separate iteration count from execution dependencies instead of explaining efficiency solely through model size. Figure 5's staged speedups and Table 3's controlled training comparison help guide implementation. However, approaching a teacher endpoint does not establish that the teacher is right in an unfamiliar state. Questions about student visitation and difficult contacts from WAM-OPD and AnyStep remain relevant. This is review interpretation, not a tested result for their combination.

I recommend first measuring sequential/CEWP output differences, event traces and p 95/p 99 latency on the same GPU. Subsequent work could compare fixed-one and adaptive budgets at matched average compute, and test whether combining RTC improves observation freshness. These validations, code/GPU/real-robot reproduction, statistical significance checks and supplementary-video review are **not run**.

## Sources and reading scope

- Focal: Chengtao Lv et al., *RealtimeWAM: One-Step Asynchronous World Action Models*,2026. [arXiv:2610.06617 v1](https://arxiv.org/abs/2610.06617v1), first submitted 2026-10-05, checked 2026-10-06. Main §1–6 and selected Appendices A, C.1–C.3, D.1–D.2, E, F.1/F.3/F.6. The 20-page PDF was accessed and text-extracted; Figures 2, 4, 5 and PDF pp.5, 6, 9 were visually inspected. This does not mean every appendix or proof was independently audited.
- Core: [Fast-WAM v2](https://arxiv.org/html/2603.16666v2), §3.1–3.3, §4.1–4.2 setup, §5; [Faster-WAM v1](https://arxiv.org/html/2608.04404v1), relevant Method and Experiment Setup/Main Results passages; [Consistency Models v2](https://arxiv.org/html/2303.01469v2), §3–4; [Mean Flows v1](https://arxiv.org/html/2505.13447v1), §3, §4.1, initial §5 setup; [RTC v2](https://arxiv.org/html/2506.07339v2), relevant §3 method, §4 setup, §6 limitations.
- Recent representatives: [DIDO v2](https://arxiv.org/html/2609.15570v2), §3.1–3.4 and initial §4 setup/Table 1 discussion; [AnyStep-WAM v2](https://arxiv.org/html/2609.33748v2), §4.1–4.3 and §5.1/Table 1; [OpenWAM v1](https://arxiv.org/pdf/2609.07398v1), selected passages on PDF pp.1, 5, 8,13–16,33; [ReWAM v1](https://arxiv.org/html/2609.38163v1), abstract and §3.2–3.4; [WAM-OPD v2](https://arxiv.org/html/2608.22364v2), §3.1–3.3, §4.1–4.2/Table 2, §5. The recent window is first submission 2026-08-01 through 10-06. These are five relevant representatives, not the five latest papers overall.
- Original Figures 2, 4,5: Chengtao Lv et al.,2026, [same-version source](https://arxiv.org/html/2610.06617v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original bytes were retained with no editing or cropping. Only the opening concept diagram was made for this review.
- Numerical results are author-reported; independent reproduction and separate native-English proofreading were not run.

- Recheck on October 22, 2026: this revision reuses the recorded two readings and rechecks v1 §3–6 and Appendix C.1–C.3. It adds the background in Eqs. (1)–(5), the derivation of (C.2)–(C.4), and intermediate summation of Euler error. Original §1–6 order is restored; training cost moves to Appendix A. Repetitive per-task tables and kernel implementation details are condensed while core success rates, adverse perturbation results, latency ablations, and training cost remain. GPU execution, formal proof verification, and reproduction were not executed. Although v2 is available, the review retains the originally requested v1.
