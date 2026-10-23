---
title: "Fewer Steps, Less Waiting: Understanding RealtimeWAM"
description: "How teacher endpoints compress action generation into one step, and KV-ready events reduce expert waiting. Read aggregate accuracy, perturbation losses, device-specific speedups and training cost together."
date: "2026-10-06"
updatedAt: "2026-10-23T15:26:12+00:00"
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

RealtimeWAM reduces action generation to one step with Teacher-Anchored Consistency Distillation (TACD) and overlaps video- and action-expert computation block by block. In the Fast-WAM-based H100 setting, adding CUDA Graph and efficient kernels as well brings latency from 299.7 to 12.2 ms, while RoboTwin average success changes from 91.51% for ten-step Fast-WAM to 90.84%. [Table 1 and Figure 5](https://arxiv.org/html/2610.06617v1#S5.T1) **Review interpretation.** The abstract's "<1% drop" holds for benchmark averages but not for the LIBERO-Plus robot perturbation (−2.6 percentage points), and of the roughly 25× speedup, the two methods themselves account for TACD's 4.56× and pipelining's further 1.34× after CUDA Graph.

This article follows original §1–6. Appendix C's error derivation follows §4.1 directly; Appendix A's limitations and the F.6 real-robot demonstration come after the conclusion.

## 1. Introduction

World action models (WAMs) add future visual prediction as a learning objective and use visual representations from video generation backbones to guide action prediction. In the Fast-WAM and Faster-WAM backbones studied here, a video expert supplies visual conditioning, and an action expert reads it to generate an **action chunk**, a sequence of future actions. Transformer experts connected through attention in this way form a Mixture-of-Transformers (MoT). [§3](https://arxiv.org/html/2610.06617v1#S3)

The authors identify two remaining bottlenecks. The first is intra-expert iteration: the video expert runs once, but the action expert removes noise repeatedly, and Figure 1 shows the action expert taking nearly 90% of end-to-end latency across GPUs. The second is inter-expert waiting: the action expert starts only after the video expert finishes its whole forward pass. [§1, Figure 1, §4](https://arxiv.org/html/2610.06617v1#S4)

<figure class="review-figure" id="figure-realtimewam-core">
<a href="/assets/reviews/realtimewam/realtimewam-core.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/realtimewam-core.svg" width="1200" height="1010" alt="Top: the frozen ten-step teacher supplies an endpoint target to a one-step action student sharing the same noisy action and frozen video KV. Bottom: one conditioning block records KV readiness after projection and preprocessing, with a wait immediately before action attention." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Concept guide · Learn fewer steps, schedule less waiting</span>The teacher is used during training only, and the local consistency loss is retained. CEWP can overlap the two streams' projections, but action attention must wait for its required KV. Line lengths are not measured durations or an execution trace. Sparse conditioning also requires KV fusion to finish. Original explanatory diagram made for this review. <span class="figure-links"><a href="/assets/reviews/realtimewam/realtimewam-core.svg" target="_blank" rel="noopener noreferrer">Full size</a></span></figcaption>
</figure>

The preceding [SyncWorld review](https://iiamaii.github.io/en/reviews/syncworld-visual-calibration/) concerned a simulator that predicts future video from supplied actions. This paper concerns efficient inference for a policy that directly generates actions using world information. Both belong under `world action model`; their architectures and evaluation roles remain different.

## 2. Related Work

### 2.1 World Action Models

The authors distinguish shared-backbone WAMs such as DreamZero and Cosmos from MoT designs such as Motus, whose video and action experts interact through attention. Fast-WAM computes reusable video features once; Light-WAM and Faster-WAM further address efficiency and out-of-distribution robustness. This is RealtimeWAM’s immediate model lineage. [Section 2.1](https://arxiv.org/html/2610.06617v1#S2.SS1)

### 2.2 Few-Step Distillation

Progressive distillation, consistency models, distribution matching, MeanFlow, and adversarial distillation reduce iterative sampling. The closest WAM comparison, Flash-WAM, addresses modality-dependent noise regimes while retaining future-video denoising at inference; its one-step setting still loses performance. This motivates the authors’ focus on one-step action distillation for MoT-based WAMs. [Section 2.2](https://arxiv.org/html/2610.06617v1#S2.SS2)

### 2.3 Asynchronous Inference for Robot Policies

The paper distinguishes execution-level overlap of next-chunk prediction with current-chunk execution, module-level separation of slow world/semantic modules from fast action modules, and sampling-level scheduling across positions, chunks, or modalities. RealtimeWAM goes finer, applying asynchrony at the block level within a single forward pass. [Section 2.3](https://arxiv.org/html/2610.06617v1#S2.SS3)

### Review supplement: other efficiency axes

DIDO compresses video tokens and denoising, AnyStep-WAM varies observation-dependent budgets, OpenWAM coordinates world/action noise states, ReWAM changes action-useful representations, and WAM-OPD supervises student-visited histories. [DIDO section 3](https://arxiv.org/html/2609.15570v2#S3), [AnyStep section 4](https://arxiv.org/html/2609.33748v2#S4), [OpenWAM sections 3–4](https://arxiv.org/pdf/2609.07398v1), [ReWAM section 3](https://arxiv.org/html/2609.38163v1#S3), [WAM-OPD sections 3–4](https://arxiv.org/html/2608.22364v2#S3)

## 3. Preliminaries

In Eqs. (1)–(2), observation $o$ and instruction $l$ become blockwise video KV conditioning $z(o,l)$. KV denotes the key/value representations read by attention; stored, they form the KV cache. Here $N$ is the number of video transformer blocks, $a$ an action chunk, and $\psi,\theta$ the video and action parameters. Visual conditioning is computed once, but actions are still generated through several denoising steps.

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

Consistency distillation (CD) in Eqs. (3)–(5) uses three models: a frozen teacher $\theta_{\mathrm T}$, a trainable student $\theta_{\mathrm S}$, and an exponential moving average (EMA) target $\theta_{\mathrm{ema}}$. Clean data actions $a_0^{\mathrm{data}}$ and Gaussian noise $\epsilon$ form $a_t$, and the teacher takes one integration step to a neighboring $s<t$, producing $\tilde a_s$. Training aligns the student's endpoint prediction with the EMA target's. Here $v_\theta$ predicts velocity, $f_\theta$ predicts a clean endpoint, and $\operatorname{sg}$ is stop-gradient; the EMA target averages student parameters after each update. [§3.1–3.2, Eqs. (1)–(5)](https://arxiv.org/html/2610.06617v1#S3)

## 4. RealtimeWAM

TACD (§4.1) targets intra-expert iteration, and wavefront pipelining (§4.2) targets inter-expert waiting.

### 4.1 Teacher-Anchored Consistency Distillation

TACD trains an action student to approach a multi-step teacher endpoint in one evaluation. CD aligns endpoint predictions from nearby noisy states on the same trajectory, but the authors argue that applying it directly to WAMs creates a local-global error gap. In Figure 3, CD reduces the consistency loss but does not substantially reduce the distance between the student's velocity and the teacher's endpoint-average velocity. The authors explain this residual through local and global errors. [§4.1, Eq.6, Figure 3, Appendix C.1](https://arxiv.org/html/2610.06617v1#S4.SS1), [Consistency Models §3–4](https://arxiv.org/html/2303.01469v2#S3)

<figure class="review-figure" id="figure-realtimewam-training">
<a href="/assets/reviews/realtimewam/paper-figure-2.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-2.png" width="1283" height="428" alt="Original TACD diagram: a frozen video expert supplies shared KV to teacher rollout, EMA target and action student, with local-consistency and teacher-anchor losses." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Original Figure 2 · Two targets to align</span>Read from the noisy action on the right toward the clean action on the left. The green teacher rollout supplies the endpoint reference. Local consistency aligns the blue EMA and red student predictions; a separate teacher-anchored loss constrains the remaining endpoint deviation. Snowflakes mark frozen modules and the flame marks training. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-2.png" target="_blank" rel="noopener noreferrer">Full size</a> · <a href="https://arxiv.org/html/2610.06617v1#S4.F2">Chengtao Lv et al., 2026, Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original bytes; no editing or cropping</span></figcaption>
</figure>

The teacher starts from the **same noisy action and frozen video KV** as the student and takes ten steps. The three action experts, teacher, student and EMA target, share the KV cache of a single frozen video expert. Because the exact endpoint $a_0^\star$ of the teacher ODE started at $a_t$ is unavailable in practice, the teacher's total displacement to its own endpoint, divided by the current noise time, supplies an interval-average velocity target. This differs from directly copying the teacher's instantaneous velocity at the current time. The local consistency term remains in the objective. [§4.1, Eq.7–8](https://arxiv.org/html/2610.06617v1#S4.SS1)

$$
\begin{aligned}
f_{\theta_{\mathrm S}}(a_t,t)&=a_t-t\,v_{\theta_{\mathrm S}}(a_t,t),\\
a_0^{\mathrm T}&=\texttt{Solver}(a_t,t,0;\theta_{\mathrm T}),\\
u_{\theta_{\mathrm T}}(a_t,t)&=\frac{a_t-a_0^{\mathrm T}}{t},\quad t>0,\\
\mathcal L_{\mathrm{TA}}&=\mathbb E_{a_t,t}\!\left[\left\|v_{\theta_{\mathrm S}}(a_t,t)-\operatorname{sg}\!\left[u_{\theta_{\mathrm T}}(a_t,t)\right]\right\|_2^2\right],\\
\mathcal L&=\mathcal L_{\mathrm{CD}}+\lambda\mathcal L_{\mathrm{TA}},\quad \lambda=0.2.
\end{aligned}
$$

$a_t$ is a noisy action; time $t$ is clean at zero and noise at one. $v_{\theta_{\mathrm S}}$ is the student velocity prediction, $f_{\theta_{\mathrm S}}$ its clean endpoint prediction, $a_0^{\mathrm T}$ the teacher's numerical rollout endpoint, and $u_{\theta_{\mathrm T}}$ the corresponding interval-average velocity. $\mathcal L_{\mathrm{CD}}$ is the local consistency loss and $\mathcal L_{\mathrm{TA}}$ the teacher-anchored loss. The video expert is frozen and only action-expert LoRA parameters are trained. The teacher and EMA target are not called at deployment. [§4.1, §5.1](https://arxiv.org/html/2610.06617v1#S5.SS1)

The accuracy of the teacher endpoint depends on the number of teacher steps $K$. With $K$=5, 10 and 20, RoboTwin overall success is 90.39, 90.84 and 90.77%; the authors choose $K=10$ because $K=20$ adds teacher computation without further improvement. [Eq.9, Appendix C.2–C.3, Table 4](https://arxiv.org/html/2610.06617v1#A3)

#### Appendix C.1–C.3 beside the method: what the endpoint bound guarantees

This explanation relocates Appendix C next to its method. Fix shared conditioning $z(o,l)$ and the training sampling distribution; assume a unique teacher ODE solution and finite second moments of errors and velocity residuals. Adding and subtracting the EMA prediction gives the two errors of Eq. (6), and expanding the square leaves a cross term.

The total error $\mathbf e_{\mathrm{total}}=f_{\theta_{\mathrm S}}(a_t,t)-a_0^\star$ compares the student endpoint prediction with $a_0^\star$. The local error $\mathbf e_{\mathrm{local}}=f_{\theta_{\mathrm S}}(a_t,t)-f_{\theta_{\mathrm{ema}}}(\tilde a_s,s)$ compares the student with the EMA prediction; the global error $\mathbf e_{\mathrm{global}}=f_{\theta_{\mathrm{ema}}}(\tilde a_s,s)-a_0^\star$ compares that EMA prediction with $a_0^\star$.

$$
\begin{aligned}
\mathbf e_{\mathrm{total}}&=\mathbf e_{\mathrm{local}}+\mathbf e_{\mathrm{global}},\\
\mathbb E\|\mathbf e_{\mathrm{total}}\|_2^2
&=\mathbb E\|\mathbf e_{\mathrm{local}}\|_2^2+\mathbb E\|\mathbf e_{\mathrm{global}}\|_2^2
+2\mathbb E\langle\mathbf e_{\mathrm{local}},\mathbf e_{\mathrm{global}}\rangle.
\end{aligned}
$$

Cauchy–Schwarz bounds the cross term by the product of the root mean-squared errors. In particular, if the mean-squared local error is zero, the mean-squared total error equals the mean-squared global error. The authors conclude that agreement with the EMA target does not by itself show that the target is accurate.

Appendix C.2 then gives an exact identity. The common $a_t$ cancels, leaving the velocity residual multiplied by $-t$. For $t>0$, squaring and dividing by $t^2$ shows that the velocity loss is an **endpoint loss weighted by $1/t^2$**.

$$
\begin{aligned}
f_{\theta_{\mathrm S}}(a_t,t)-a_0^{\mathrm T}
&=-t\left[v_{\theta_{\mathrm S}}(a_t,t)-u_{\theta_{\mathrm T}}(a_t,t)\right],\\
\mathcal L_{\mathrm{TA}}
&=\mathbb E\!\left[\frac{\|f_{\theta_{\mathrm S}}(a_t,t)-a_0^{\mathrm T}\|_2^2}{t^2}\right].
\end{aligned}
$$

Appendix C.3 adds the teacher's numerical integration error $\boldsymbol\delta=a_0^{\mathrm T}-a_0^\star$. The inequality $\|p+q\|^2\leq2\|p\|^2+2\|q\|^2$ follows from Cauchy–Schwarz and $2ab\leq a^2+b^2$. The last line uses $0<t\leq1$, and velocity arguments $(a_t,t)$ are omitted inside the norm. The bound constrains the sum of local and global errors directly, without requiring each component to shrink separately.

$$
\begin{aligned}
\mathbf e_{\mathrm{total}}&=-t\left[v_{\theta_{\mathrm S}}(a_t,t)-u_{\theta_{\mathrm T}}(a_t,t)\right]+\boldsymbol\delta,\\
\mathbb E\|\mathbf e_{\mathrm{total}}\|_2^2
&\leq 2\mathbb E\!\left[t^2\|v_{\theta_{\mathrm S}}-u_{\theta_{\mathrm T}}\|_2^2\right]+2\mathbb E\|\boldsymbol\delta\|_2^2\\
&\leq 2\mathcal L_{\mathrm{TA}}+2\mathbb E\|\boldsymbol\delta\|_2^2.
\end{aligned}
$$

The size of $\boldsymbol\delta$ is bounded as follows. If the teacher velocity is $L$-Lipschitz in state and the exact trajectory has second time derivative bounded in norm by $M$, a uniform Euler step $h=t/K$ has local error at most $Mh^2/2$. Here $D_k$ is the distance between numerical and exact states after step $k$. The geometric-series expansion below is a **review-supplied intermediate calculation**, retaining Appendix C.3’s order.

$$
\begin{aligned}
D_{k+1}&\leq(1+Lh)D_k+\frac{Mh^2}{2},\qquad D_0=0,\quad h=\frac{t}{K},\\
\|\boldsymbol\delta\|_2=D_K
&\leq\frac{Mh^2}{2}\sum_{j=0}^{K-1}(1+Lh)^j\\
&=\frac{Mh}{2L}\left[(1+Lh)^K-1\right]
\leq\frac{Me^{Lt}t^2}{2K}\quad(L>0).
\end{aligned}
$$

The final inequality uses $(1+Lh)^K\leq e^{Lt}$ and $e^{Lt}-1\leq Lt e^{Lt}$. For $L=0$, direct summation gives $D_K\leq Mt^2/(2K)$. Under uniform $L,M$, the numerical-error term in the bound is $O(K^{-2})$, and the authors conclude that increasing the number of teacher steps $K$ gives explicit control over it. [Appendix C.1–C.3, Eqs. (C.1)–(C.4)](https://arxiv.org/html/2610.06617v1#A3)

### 4.2 Cross-Expert Wavefront Pipelining

Cross-Expert Wavefront Pipelining, or CEWP, overlaps video and action computation on two nonblocking CUDA streams on one GPU. The authors split each diffusion-transformer (DiT) block into Q/K/V projection, attention and remaining computation, and show that action attention is the only operation in an action block that reads the video KV directly. A video block therefore records a readiness event after projection and required KV preprocessing; the action projection proceeds as soon as its own hidden state is available, and waits for the event only immediately before attention reads the video KV. Meanwhile, the video stream continues its remaining work. [§4.2, Figure 4](https://arxiv.org/html/2610.06617v1#S4.SS2)

<figure class="review-figure" id="figure-realtimewam-pipeline">
<a href="/assets/reviews/realtimewam/paper-figure-4.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-4.svg" width="1137" height="581" alt="Original comparison of sequential expert execution, two-stream CEWP with KV-ready events and waits before action attention, and the dependency graph within a block." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Original Figure 4 · From a full-cache barrier to block events</span>The top schedule starts the action expert after the whole video expert finishes. The lower schedule publishes each ready KV block and waits when action attention needs it. The right side shows the projection→attention→remainder dependencies within one block. The time axis illustrates execution order. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-4.svg" target="_blank" rel="noopener noreferrer">Full size</a> · <a href="https://arxiv.org/html/2610.06617v1#S4.F4">Chengtao Lv et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original bytes; no editing or cropping</span></figcaption>
</figure>

CEWP uses the same parameters, inputs, masks and operators, and changes only the execution order. Appendix D argues by induction that identical states entering a block yield identical Q/K/V and the same completed video KV, hence identical next-block states, so that in exact arithmetic the output matches sequential inference. The authors call this another valid topological ordering of the original computation graph, requiring no additional training and independent of TACD. [Appendix D](https://arxiv.org/html/2610.06617v1#A4)

Appendix B's Algorithm 1 gives pseudocode for one paired forward pass. The set $\mathcal C$ of conditioning blocks that receive video KV is every block $\{1,\ldots,N\}$ in Fast-WAM and $\{1,5,\ldots,N\}$ in the authors' Faster-WAM implementation. At each conditioning block, `Fuse` returns the single-block KV (Fast) or a weighted fusion of the accumulated interval KV (Faster) before the event is recorded; action blocks in between need no cross-expert synchronization. [Appendix B, Algorithm 1](https://arxiv.org/html/2610.06617v1#algorithm1)

Appendix E summarizes the latency the two methods remove with a simple model. With single-pass video and action latencies $T_v,T_a$ and $S$ action denoising steps, sequential execution costs $T_v+ST_a$, and TACD sets $S=1$. CEWP latency is written with the overlap saving $T_{\mathrm{overlap}}$ and a synchronization, scheduling and resource-contention cost $\Delta$; under ideal overlap ($T_{\mathrm{overlap}}\approx\min(T_v,T_a)$) it becomes $\max(T_v,T_a)+\Delta$. The authors add that block dependencies and shared GPU resources keep real overlap below this. [Appendix E](https://arxiv.org/html/2610.06617v1#A5)

$$
T_{\mathrm{seq}}(S)=T_v+ST_a,\qquad
T_{\mathrm{pipe}}(1)=T_v+T_a-T_{\mathrm{overlap}}+\Delta\approx\max(T_v,T_a)+\Delta.
$$

The **Faster-WAM here is arXiv:2608.04404**, the future-conditioning model. The identically named 2608.02365 studies Depth-of-Thought and is a separate paper. `*` denotes Fast-WAM-based RealtimeWAM and `†` this future-conditioning Faster-WAM-based version. [RealtimeWAM §3, Table 1](https://arxiv.org/html/2610.06617v1#S3), [Faster-WAM Method](https://arxiv.org/html/2608.04404v1#Sx3)

## 5. Experiments

Success rates and runtime are evaluated separately.

### 5.1 Implementation Details; 5.2 Evaluation Benchmarks

Evaluation covers clean/random conditions in RoboTwin 2.0 (multi-task training on 50 tasks with 2,500 clean and 25,000 randomized demonstrations), four LIBERO suites, and seven LIBERO-Plus perturbation categories. Success rate is the fraction of rollouts judged successful; higher is better. Computational cost is reported separately as the number of function evaluations (NFE) and latency. Training freezes the video expert and adapts only rank-128 action-expert LoRA for 30,000 iterations; TACD uses a ten-step teacher. The authors state that, for a fair comparison, the DMD, CD and MeanFlow distillation baselines use the same training settings. [§5.1–5.2](https://arxiv.org/html/2610.06617v1#S5.SS1)

### 5.3 Main Results

**All 16 rows of original Table 1, grouped by base model.** Success rates are percentages; higher is better. NFE counts denoising function evaluations for video and action. RoboTwin overall averages clean/random conditions, while LIBERO averages its four suites. According to Appendix F.1, the eight "(cited)" rows retain scores reported in prior papers (Light-WAM, Flash-WAM and others), while ten-step Fast/Faster-WAM, the three distillation baselines and RealtimeWAM were evaluated by the authors. The naive one-step row, which runs the teacher with a single step, shows why one-step generation needs dedicated training. [Table 1, Appendix F.1](https://arxiv.org/html/2610.06617v1#S5.T1)

| Model | Video NFE | Action NFE | RoboTwin Clean ↑ | Random ↑ | Overall ↑ | LIBERO ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| $\pi_0$ (cited) | — | — | 65.92 | 58.40 | 62.16 | 94.1 |
| $\pi_{0.5}$ (cited) | — | — | 82.74 | 76.76 | 79.75 | 96.9 |
| X-VLA (cited) | — | — | 72.90 | 72.80 | 72.85 | 98.1 |
| Motus (cited) | 10 | 10 | 88.66 | 87.02 | 87.84 | 97.7 |
| LingBot-VA (cited) | 25 | 50 | 92.90 | 91.50 | 92.20 | 98.5 |
| Flash-WAM (cited) | 1 | 2 | 88.42 | 82.66 | 85.54 | 96.1 |
| Flash-WAM (cited) | 1 | 1 | 82.56 | 80.26 | 81.41 | 95.1 |
| Light-WAM (cited) | 1 | 1 | 76.40 | 76.30 | 76.40 | 97.2 |
| Fast-WAM | 1 | 10 | 91.82 | 91.19 | 91.51 | 97.0 |
| Fast-WAM, naive one-step | 1 | 1 | 70.42 | 68.12 | 69.27 | 95.0 |
| CD* | 1 | 1 | 89.88 | 89.38 | 89.63 | 96.9 |
| MeanFlow* | 1 | 1 | 85.74 | 85.72 | 85.73 | 96.9 |
| DMD* | 1 | 1 | 88.64 | 87.80 | 88.22 | 96.1 |
| RealtimeWAM* | 1 | 1 | 91.96 | 89.72 | 90.84 | 97.0 |
| Faster-WAM | 1 | 10 | 93.20 | 92.66 | 92.93 | 98.9 |
| RealtimeWAM† | 1 | 1 | 92.98 | 92.30 | 92.64 | 99.0 |

RealtimeWAM* loses **0.67 percentage points (pp)** of RoboTwin overall relative to Fast-WAM; † loses **0.29 pp** relative to Faster-WAM. LIBERO is unchanged for * and increases by 0.1 pp for †. The authors read this as near-lossless one-step generation that clearly outperforms one-step Flash-WAM (81.41%) and DMD* (88.22%). [§5.3, Table 1](https://arxiv.org/html/2610.06617v1#S5.SS3)

**All perturbation columns for Faster and RealtimeWAM† from original Table 2.** LIBERO-Plus evaluates camera, robot, language, lighting and other shifts. The final pp column is subtraction performed for this review. Overall is not the arithmetic mean of the seven values but an average weighted by trial counts per category. [Table 2, Appendix F.1](https://arxiv.org/html/2610.06617v1#S5.T2)

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

Overall falls by 0.6 pp. Citing small gains under camera, noise and layout shifts, the authors conclude that step distillation largely preserves out-of-distribution (OOD) robustness. The largest drops are robot (2.6 pp) and language (2.2 pp). [Table 2](https://arxiv.org/html/2610.06617v1#S5.T2)

### 5.4 Ablation Study and latency analysis

Table 3 is a RoboTwin ablation over loss terms and fine-tuned experts. CD with both video and action tuned reaches 89.63%, action-only CD with a frozen video expert 89.85%, and TACD, which adds the teacher-anchored loss, 90.84%, a further 0.99 pp. The random column, however, falls from 89.94 to 89.72%. The authors conclude that tuning the video expert lowers success while adding training cost, and that TACD's gain matches their analysis that the anchor reduces global deviation left by the local objective. [Table 3](https://arxiv.org/html/2610.06617v1#S5.T4)

The first row of Table 3 (89.88/89.38/89.63%) is identical to CD* in Table 1. **Review interpretation.** Table 1's CD* therefore appears to tune the video expert as well, so §5.1's "same training settings" seems not to extend to which experts are fine-tuned; the 1.21 pp gap over CD* in Table 1 includes 0.22 pp from freezing the video expert.

<figure class="review-figure" id="figure-realtimewam-speed">
<a href="/assets/reviews/realtimewam/paper-figure-5.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-5.svg" width="1152" height="263" alt="H100 latency bars for Fast-WAM and Faster-WAM after adding TACD, CUDA Graph, CEWP and efficient kernels, with cumulative and incremental speedups." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Original Figure 5 · Read the cumulative speedup</span>Read the Fast and Faster panels separately, adding one optimization at a time from top to bottom. The horizontal axis is latency in milliseconds; lower is faster. Blue speedup labels are cumulative relative to the initial baseline; connector labels are gains over the preceding stage. The final roughly 25× also includes CUDA Graph and efficient kernels. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-5.svg" target="_blank" rel="noopener noreferrer">Full size</a> · <a href="https://arxiv.org/html/2610.06617v1#S5.F5">Chengtao Lv et al., 2026, Figure 5</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · Original bytes; no editing or cropping</span></figcaption>
</figure>

On H100, Fast-WAM goes from 299.7 ms to 65.8 ms with TACD, 23.3 ms after CUDA Graph, 17.4 ms after CEWP, and 12.2 ms after efficient kernels. The Faster sequence is 218.9→57.1→26.2→22.4→16.1 ms. TACD alone gives 4.56× and 3.83×, CEWP after CUDA Graph adds 1.34× and 1.17×, and the cumulative speedups are 24.55× and 13.56×. The efficient kernels are five LightX2V-based operator-level optimizations (kernel fusion, removal of all-True masks, and packed projection GEMMs; Appendix F.2). [§5.4, Figure 5](https://arxiv.org/html/2610.06617v1#S5.F5)

**Baseline→final latency in milliseconds, from original Figure 5 (H100) and Table F.3 (additional GPUs).** The measurement includes variational autoencoder (VAE) encoding and excludes text encoding, which runs once per episode. Sensor capture, communication and motor actuation are not included. [Figure 5](https://arxiv.org/html/2610.06617v1#S5.F5), [Appendix F.5](https://arxiv.org/html/2610.06617v1#A6.SS5)

| Backbone | H100 | RTX 5090 | RTX 4090D |
| --- | --- | --- | --- |
| Fast → RealtimeWAM* | 299.7→12.2 | 251.5→17.8 | 480.8→27.2 |
| Faster → RealtimeWAM† | 218.9→16.1 | 214.1→30.9 | 362.4→44.7 |

On consumer GPUs the cumulative speedup shrinks to 14.1× and 17.7× for Fast-based and 6.9× and 8.1× for Faster-based models (RTX 5090 and 4090D); the authors suggest this may be because the current kernels are tuned mainly for H-series GPUs. They state that per-call latency is well below the execution time of 30 Hz control, enabling real-time action generation (§1). [Appendix F.5](https://arxiv.org/html/2610.06617v1#A6.SS5) **Review interpretation.** The 12.2 ms H100 call fits within one 30 Hz period (about 33.3 ms), but the † model's 44.7 ms on RTX 4090D exceeds it.

## 6. Conclusion

The authors combine one-step action denoising through TACD with dependency-preserving blockwise overlap through CEWP. They conclude that RealtimeWAM enables real-time action generation (e.g., 12.2 ms) with near-lossless performance on Fast-WAM and Faster-WAM, with end-to-end speedups of about 14–25× on H100. [Original §6](https://arxiv.org/html/2610.06617v1#S6)

## Appendix A and F.6: Limitations; Real-World Evaluation of Long-Horizon Garment Folding

The first limitation in Appendix A is applicability. RealtimeWAM targets MoT-based WAMs and is not directly applicable to shared-backbone architectures such as DreamZero. The authors explain that step distillation in shared-backbone WAMs resembles video-generation step distillation, making action-specific characteristics harder to isolate. Within this scope, it supports both Fast-WAM, which omits future-video prediction at inference, and Faster-WAM, which retains it. [Appendix A](https://arxiv.org/html/2610.06617v1#A1)

The second limitation is training cost. **Original Table A.1** reports training time and peak memory for 30,000 iterations on 16 H100 GPUs. [Table A.1](https://arxiv.org/html/2610.06617v1#A1.T1)

| Training setup | Time | Peak memory |
| --- | --- | --- |
| CD, video + action tuned | 8 h 1 m | 65.98 GiB |
| CD, action only | 5 h 21 m | 24.73 GiB |
| TACD, action only | 8 h 3 m | 24.73 GiB |

Action-only CD and TACD have the same peak memory, but training time rises from 5 h 21 m to 8 h 3 m, about 50%. The authors call this manageable: it is comparable to joint video-and-action fine-tuning (8 h 1 m), and freezing the video expert cuts peak memory by 62.5% relative to joint fine-tuning. Teacher rollout computation is paid during training to reduce deployment cost.

Appendix F.6 presents a qualitative real-robot T-shirt-folding rollout with RealtimeWAM* on dual-arm AgileX PiPER and an RTX 4090. The authors report that consecutive actions proceeded without perceptible pauses even without execution-level asynchrony such as RTC; no success rate or trial count is reported. [Appendix F.6, Figure F.1](https://arxiv.org/html/2610.06617v1#A6.SS6)

## Review interpretation and open questions

**Review interpretation.** The paper's strength is treating iteration count and execution dependency as separate bottlenecks and showing each method's share through the staged latencies in Figure 5. Consistent with Figure 1's analysis that the action expert takes about 90% of latency, TACD gives the largest speedup. In Appendix E's model, CEWP can save at most $\min(T_v,T_a)$, so its speedup cannot exceed 2× even ignoring $\Delta$; 1.34× and 1.17× fall within that range. Of the Fast path's 24.55×, the product of the two methods is about 6.1×, and the remaining factor of about 4 comes from CUDA Graph, which applies even without TACD (2.6–4.5× on its own in Table F.3), and efficient kernels.

The quality evidence is average success without run-to-run variance. Table 3, which isolates the teacher anchor, shows a 0.99 pp gain while the random condition falls, so this table alone cannot show whether differences of about 1 pp exceed run-to-run variation. Appendix C's bound concerns error up to the exact teacher ODE endpoint, leaving errors the teacher itself makes in unfamiliar states outside its scope, and tightening it by raising $K$ from 10 to 20 did not raise success (Table 4). The introduction's claim to apply to "any MoT-based WAM" rests on two cases, Fast-WAM and Faster-WAM.

Open questions remain. How large are the actual GPU output differences and p95/p99 latencies between sequential execution and CEWP? At matched average compute, how does a fixed single step compare with observation-dependent budgets such as AnyStep-WAM? How much does teacher-endpoint supervision reduce errors in states the student actually visits, the question WAM-OPD raises? Does combining CEWP with execution-level asynchrony such as RTC improve observation freshness?

## Sources and reading scope

- Paper: *RealtimeWAM: One-Step Asynchronous World Action Models* · Chengtao Lv et al. · first public date 2026-10-05 · [reviewed version](https://arxiv.org/abs/2610.06617v1) · reviewed 2026-10-23.
- Scope read: Version 1, sections 1–6 and appendices A–F.
- Figure sources: Figures 2, 4, and 5 are unchanged author assets from v1, CC BY 4.0. The thumbnail concept diagram is review-authored. [Paper](https://arxiv.org/abs/2610.06617v1).
- Not verified: research-code execution, model training/reproduction, GPU measurements, and supplementary video.
