---
title: "SWAP: Which VLA Should Control the Robot Next?"
description: "Follow SWAP's policy-routing MDP and IQL critic in the authors' section order, from real-robot success gains to cases where routing loses to the best policy under distribution shift."
date: "2026-10-12"
publishedAt: "2026-10-12T13:18:52+09:00"
updatedAt: "2026-10-23T08:56:18+00:00"
topics: ["robot learning", "vision-language-action", "policy routing", "offline reinforcement learning", "multi-model systems"]
visibility: "public"
lang: "en"
translationKey: "swap-policy-routing"
paperTitle: "SWAP: Stepwise Action Policy Routing for Vision-Language-Action Models"
paperPublishedDate: "2026-10-03"
authors: ["Mousumi Das", "Aditeya Prajapati", "Abrar Anwar", "Jesse Thomason"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.06926v1"
thumbnail: "/assets/reviews/swap-policy-routing/swap-overview.webp"
thumbnailAlt: "SWAP Figure 2: collect successful and failed candidate-policy rollouts, train an offline routing critic, then compare policy-ID Q-values and execute the selected policy's action chunk."
---

SWAP evaluates policy values from the current observation and switches vision-language-action (VLA) models between action chunks. It succeeds in 29 of 30 real-robot trials across three tasks versus 19 for the best single candidate, but reaches 89.0% against a dominant policy’s 92.0% in one LIBERO-Plus setting. [Tables I and III(b)](https://arxiv.org/pdf/2610.06926v1#page=4) **Review interpretation.** Complementary candidates and reliable value estimates under distribution shift determine the benefit of routing.

The article follows the original sections I–VI through the problem definition, learning objective, algorithm, principal experiments, and failure conditions. SWAP contains no separate theorem or proof section. The expectile explanation and small calculation below supplement the review; they are not an optimality proof for SWAP.

## I. Introduction: Must One Policy Control the Entire Episode?

A VLA maps camera observations and language instructions to robot actions. Typically, one model is selected and used throughout an episode. Yet models produce different behaviors and failure modes under the same observations. Free-space motion, precise grasping, and recovery after a drop can favor different controllers. An average task score does not identify the best policy for the next state. [§I, pp. 1–2](https://arxiv.org/pdf/2610.06926v1#page=1)

The proposed change is **policy routing**: above the decision about motor actions, decide which policy should generate the next actions. This selection repeats after a short action chunk rather than occurring only at the episode's start.

The key premise is **complementarity**. If one candidate dominates every state, switching alone cannot create an advantage. SWAP's experiments explore complementary behavior while also showing how a router can become unreliable when candidate rankings change.

## II. Background and Related Work

The paper connects VLAs, LLM routing, and value functions for robotics, in that order. Two useful comparison questions are **what gets selected** and **whether selection changes future observations**. [§II, p. 2](https://arxiv.org/pdf/2610.06926v1#page=2)

The authors first connect the distinct failures of VLAs trained with different data and architectures under distribution shifts. LLM routing trades cost against capability per request, whereas robot routing is sequential because actions change subsequent observations. The value-function lineage includes policy optimization, SayCan skill selection, failure detection, and deployment-time action steering; SWAP uses it to score fixed VLA candidates. [Section II](https://arxiv.org/pdf/2610.06926v1#page=2)

### Review supplement: roles of five approaches

| Approach | What it selects or learns | Connection to SWAP and distinction |
| --- | --- | --- |
| [OpenVLA-OFT](https://arxiv.org/html/2502.19645v1), §IV–V | VLA action generation and fine-tuning | A source of simulation candidates. OFT improves the controller itself; SWAP selects among prepared controllers. |
| [RouteLLM](https://arxiv.org/html/2406.18665v4), §3.1 | Strong/weak LLM selection from preference data | Both select before calling a model. Robot actions also change subsequent observations, requiring sequential decisions about downstream return. |
| [IQL](https://arxiv.org/pdf/2110.06169v1), §4.1–4.3 | Q and state-value learning from logged data | The offline RL component used directly by SWAP. Here the action is a policy ID rather than a motor command. |
| [SayCan](https://arxiv.org/abs/2204.01691v2), abstract | Skills with linguistic relevance and physical feasibility | An antecedent for value-based executable skill selection. SWAP switches among VLAs attempting the same task, at chunk boundaries. |
| [V-GPS](https://arxiv.org/html/2410.13816v2), §5.1–5.2 | Value-based re-ranking of multiple actions sampled from a policy | V-GPS generates action candidates before scoring them. SWAP scores policy IDs first and requests actions only from the selected VLA. |

This is a conceptual comparison, not a direct performance evaluation against RouteLLM, SayCan, or V-GPS. SWAP also differs from a jointly trained mixture of experts: the candidate policies are fixed black boxes, and the router is trained separately. Adding a policy without retraining **experts** does not establish that the **router** requires no further training.

## III. Robot Policy Routing

The authors define a Markov decision process and a candidate policy library:

$$
\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma),
\qquad \mathcal P=\{\pi_1,\pi_2,\ldots,\pi_K\}.
$$

$\mathcal S$ and $\mathcal A$ are state and robot-action spaces, $P$ is the transition model, $r$ the reward, and $\gamma$ the discount factor. The policy library $\mathcal P$ is distinct from transition $P$. The routing policy $\phi$ chooses a policy index $z_t$ from state $s_t$:

$$
\phi(z_t\mid s_t),\qquad z_t\in\{1,\ldots,K\},
\qquad a_t\sim\pi_{z_t}(\cdot\mid s_t).
$$

The objective maximizes the long-horizon performance of actions generated by these choices. [§III, p. 2](https://arxiv.org/pdf/2610.06926v1#page=2)

$$
\underset{\phi}{\arg\max}\;
\mathbb E_{\phi,P}
\left[\sum_{t=1}^{T}\gamma^{t-1}r(s_t,a_t)\right].
$$

The router's action is a **policy index**, not a joint angle or gripper command. Its chosen controller produces the motor actions. Selection concerns downstream success rather than only the apparent quality of the current motion.

The formal definition uses state $s_t$, whereas the implementation uses camera observation $o_t$ and a language instruction. The paper does not prove that these inputs fully represent the MDP state. Occluded or unobserved information can still matter.

## IV. SWAP: Stepwise Action Policy Routing

![SWAP Figure 2: offline dataset construction, IQL critic learning, and online policy selection](/assets/reviews/swap-policy-routing/swap-overview.webp)

[View at full size](/assets/reviews/swap-policy-routing/swap-overview.webp)

**Reading Figure 2.** The left panel collects successful and failed candidate rollouts. The center learns Q-values from an image, instruction, and policy ID. The right scores **every candidate ID**, requests an action chunk from the selected policy, executes it, and observes again. The loop is observation→selection→execution→new observation. It does not generate actions from every VLA before comparing them. The illustrative IDs start at 0 in the drawing, while §III's formal IDs start at 1; these are separate indexing conventions, not different policy counts. [Fig. 2, p. 3](https://arxiv.org/pdf/2610.06926v1#page=3)

### Offline Dataset: Retain Failures Alongside Successes

Each candidate's logged trajectories are divided into chunks of $c$ actions. The routing dataset $\mathcal D$ contains the following transitions. [§IV, pp. 2–3](https://arxiv.org/pdf/2610.06926v1#page=2)

$$
(o_t,z_t,r_t,o_{t+c},d_t).
$$

| Symbol | Meaning |
| --- | --- |
| $o_t$ | Observation at chunk start |
| $z_t$ | ID of the policy executing that chunk |
| $r_t$ | Resulting sparse reward |
| $o_{t+c}$ | Observation after $c$ actions |
| $d_t$ | Episode termination indicator |

The experiments assign **10** to a successful final transition and **0** to failures and all intermediate transitions. The method section's “binary” describes two outcomes, not necessarily numerical rewards of 0 and 1. Failed trajectories provide examples of unproductive states and selections. [§V-A, p. 5](https://arxiv.org/pdf/2610.06926v1#page=5)

### IQL Critic: Connect Today's Choice to Later Success

SWAP learns two action-value functions $Q_1,Q_2$ and a state-value function $V$. Equation (1) is central. Line breaks below preserve its meaning while improving readability on mobile. [§IV, Eq. (1), p. 3](https://arxiv.org/pdf/2610.06926v1#page=3)

$$
\begin{aligned}
\mathcal L_Q={}&
\mathbb E_{(o_t,z_t,r_t,o_{t+c},d_t)\sim\mathcal D}\\
&\Bigg[\sum_{i=1}^{2}\Big(Q_i(o_t,z_t)\\
&\qquad-\big(r_t+\gamma(1-d_t)V(o_{t+c})\big)\Big)^2\Bigg].
\end{aligned}
$$

The temporal-difference target is $r_t+\gamma(1-d_t)V(o_{t+c})$. A completed success supplies direct reward. A nonterminal transition also includes discounted value at the next observation. When the episode ends, $1-d_t=0$, removing further value.

This bootstrapping allows unrewarded movement to acquire value through its connection to later success. It does not mechanically label every intermediate state in a failed rollout as worthless. Alternative supported choices from that state could still lead to success, which state-value learning and transitions can reflect. The dataset must support those states and alternatives.

The equation applies $\gamma$ once per chunk transition. The paper does not explain its correspondence to low-level discounting by $\gamma^c$; this review does not silently change the exponent.

### Review Supplement: Why Use Expectiles Instead of a Mean?

SWAP states that $V$ uses expectile regression but does not specify its loss or coefficient $\tau$. The IQL original helps explain the mechanism. The symbols $s,a,\psi,\hat\theta$ below follow **IQL's notation**; this does not establish additional SWAP implementation settings. [IQL §4.1–4.2, Eq. (5), pp. 4–5](https://arxiv.org/pdf/2110.06169v1#page=4)

$$
L_2^\tau(u)=|\tau-\mathbf 1(u<0)|u^2,
\qquad \tau\in(0,1),
$$

$$
L_V(\psi)=\mathbb E_{(s,a)\sim\mathcal D}
\left[L_2^\tau\big(Q_{\hat\theta}(s,a)-V_\psi(s)\big)\right].
$$

A positive residual means Q exceeds the current V. For $\tau>0.5$, such residuals receive greater weight, pulling V toward better Q-values. This encourages composing good choices supported by the data rather than reproducing average behavior. Expectiles use an asymmetric **squared** loss, unlike quantiles' asymmetric absolute loss.

**Illustrative calculation.** Suppose Q-values 3 and 9 occur equally often at a state. For $3\leq v\leq9$, the first-order condition is below; $v$ is a review-only auxiliary symbol.

$$
(1-\tau)(v-3)+\tau(v-9)=0,
\qquad v=3+6\tau.
$$

At $\tau=0.5$, the estimate is 6; at $\tau=0.8$, it is 7.8. Neither 7.8 nor this coefficient is a reported SWAP result or setting. They illustrate the weighting mechanism. Expectile regression does not guarantee accurate values outside the data distribution.

Original IQL extracts a separate action policy from learned values. SWAP can compare a finite set of policy IDs directly at inference. IQL's theoretical results cannot simply become optimality or OOD guarantees for an image-based SWAP implementation.

### Architecture and Inference: Choose the Policy Before Generating Actions

A **frozen** Qwen2.5-VL-3B encoder turns the image and instruction into a 2,048-dimensional representation. This is concatenated with a learned policy-ID embedding and fed to lightweight MLP double-Q critics. A separate V network is used in training. [§IV, pp. 3–4](https://arxiv.org/pdf/2610.06926v1#page=3)

Selection follows:

$$
z_t=\underset{z}{\arg\max}\;
\min\big(Q_1(o_t,z),Q_2(o_t,z)\big).
$$

Taking the smaller critic estimate reduces overestimation. It is not a certified lower bound on true return, nor is Q a calibrated success probability.

**The unit of Q.** With success reward 10 and zero intermediate rewards, Q reflects both success and discounting until success. In a review example, certain success after two discounts gives value $10\gamma^2$. Dividing a Q difference by 10 therefore does not generally give a success-probability difference. Read it as a relative candidate comparison with reward scale, discounting, and data distribution fixed. The two stages in this example are not a measured SWAP completion time. [§III–IV, §V-A](https://arxiv.org/pdf/2610.06926v1#page=3)

Algorithm 1's execution can be read in four steps:

1. Encode the current inputs as $h_t=f(o_t,\text{instruction})$.
2. Compare $\min(Q_1(h_t,k),Q_2(h_t,k))$ for every candidate $k$.
3. Query only the selected $\pi_{z_t}$ and execute $c$ actions.
4. Update $t\leftarrow t+c$ and repeat from the new observation.

The equation's $Q_i(o_t,z)$ implicitly includes observation encoding; the algorithm makes $h_t$ explicit. Neither the base policies nor router weights are updated during deployment. “Online” means **repeated selection during execution**. Avoiding calls to every VLA does not eliminate the 3B encoder and critic overhead. Actual latency and memory usage are unreported.

## V. Experiments and Results

### A. Experiment Setup

Real-robot evaluation uses DROID's 7-DoF Franka Panda with external and wrist cameras. Tasks are marker into bowl, cube into cup, and cube from cloth into bowl. Candidates are $\pi_0$, $\pi_{0.5}$, and $\pi_0$-FAST. Public DROID checkpoints of the latter two are used zero-shot; $\pi_0$ is fine-tuned on a small amount of data. Keeping candidates fixed during routing learning does not mean their preparation involved no fine-tuning. [§V-A, pp. 4–5](https://arxiv.org/pdf/2610.06926v1#page=4)

Ten successful and ten failed rollouts per policy per task yield **180 training trajectories**. Each method is evaluated on ten held-out trials per task. These are new trials of the same three tasks, not unseen task identities. Real-robot critic training runs for 4,000–8,000 epochs. Q-values are aggregated across camera views, but the aggregation rule—such as a sum or mean—is unspecified.

In simulation, logs contain 50 rollouts per policy per task on eight LIBERO-Spatial tasks; tasks 5 and 9 are held out for router evaluation. A task can be unseen by the router yet familiar to the base VLAs. LIBERO-Plus Spatial supplies OOD variations. Real-robot and simulation critics are trained separately, with 200 epochs in simulation. Candidate sets vary across experiments.

Baselines are **fixed-policy execution** and **uniform random routing**, separating the ability to switch from the learned criterion for switching. There is no direct comparison to a learned router selecting once at episode start. The paper and author sources do not specify chunk size $c$, $\gamma$, expectile coefficient, learning rate, seed-level error bars, or detailed LIBERO-Plus evaluation trial counts.

### B. Results

#### 1. Real-Robot Success Above Fixed Policies and Random Routing

The following preserves **every row of Table I**. Each cell is **success rate / mean action steps among successful trajectories**. Steps are not averaged over successes and failures together. [Table I, p. 5](https://arxiv.org/pdf/2610.06926v1#page=5)

| Task | $\pi_0$ | $\pi_{0.5}$ | $\pi_0$-FAST | Random | SWAP |
| --- | --- | --- | --- | --- | --- |
| Marker → bowl | 30% / 274 | 60% / 249 | 80% / 209 | 60% / 233 | **100% / 200** |
| Cube → cup | 50% / 358 | 20% / 420 | 40% / 372 | 60% / 336 | **90% / 224** |
| Cube from cloth → bowl | 40% / 288 | 50% / 201 | 70% / 227 | 80% / 262 | **100% / 155** |
| Reported average | 40% / 307 | 43.3% / 290 | 63.3% / 269 | 66.7% / 277 | **96.7% / 193** |

SWAP corresponds to 29 successes out of 30 trials, versus 19 for the best fixed policy, $\pi_0$-FAST. Subtracting rounded percentages gives **33.4 percentage points**; computing from counts gives approximately **33.3 points**. This is not a 33.4% relative increase. The difference from Random is 30 points.

Random switching slightly exceeds the best fixed policy. The authors interpret switching away from a stalled controller as crude recovery; SWAP's additional gain supports learning which policy to select. With ten trials per task and no reported confidence intervals or multi-seed analysis, these numbers should not be treated as stable expected performance across arbitrary environments.

#### 2. Shorter Successful Trajectories

The reduction from reported mean 269 to 193 steps is:

$$
\frac{269-193}{269}\times100\approx28.3\%.
$$

This compares **trajectory length conditioned on success**. The reported averages are rounded, unweighted means of task-level step averages. Successful subsets and their sizes also differ between methods, so this is not a matched comparison of identical successful trajectories. Fewer actions support an efficiency claim, but do not measure a 28.3% reduction in wall-clock time including encoding, model calls, and switching.

The authors suggest discounted return can favor faster progress. As a review supplement, with the same success reward of 10 and $0<\gamma<1$, success after fewer routing transitions is discounted less. This is not explicit time-cost optimization or a latency guarantee.

#### 3. Q-Values and Policy Preferences Along the Trajectory

The paper reports low Q-values and rare selection for $\pi_0$, with the other policies alternating. Mean switching frequency is **8.7 per rollout**, and mean predicted Q improvement over the previously active policy is **+0.117** per switch. This is a comparison within the learned critic, not measured reward improvement or 11.7 percentage points of success probability. [§V-B, pp. 5–6](https://arxiv.org/pdf/2610.06926v1#page=5)

**Table II** reports selection frequency over normalized thirds of trajectories:

| Segment | $\pi_0$-FAST | $\pi_{0.5}$ |
| --- | --- | --- |
| First third | 51.0% | 49.0% |
| Middle third | 22.6% | 77.4% |
| Final third | 36.3% | 63.7% |

Selections begin balanced, then favor $\pi_{0.5}$ in the middle. Its share falls from 77.4% to 63.7% at the end, so the trend is not monotonically increasing. Normalized progress segments are not predefined physical skill labels shared by all rollouts. [Table II, p. 5](https://arxiv.org/pdf/2610.06926v1#page=5)

The Q-analysis prose calls $\pi_0$-FAST the weaker standalone policy, but Table I reports 63.3% versus $\pi_{0.5}$'s 43.3%. Its 36%/64% value-ranking statement also does not clearly define mutually exclusive events. This review therefore uses the explicit Table I–II values and Figure 4, rather than that sentence, to establish performance rankings and occupancy.

#### 4. Figure 4: Movement, Grasping, and Recovery

![SWAP Figure 4: candidate Q-values and selected-policy band during a marker drop and recovery](/assets/reviews/swap-policy-routing/swap-q-values.webp)

[View at full size](/assets/reviews/swap-policy-routing/swap-q-values.webp)

**How to read the figure.** Start with the lower color band: teal is $\pi_0$-FAST, red $\pi_{0.5}$, and orange $\pi_0$. The band shows the selected policy; the three curves above show predicted candidate Q-values. The horizontal axis follows execution progress, but the drawing gives no explicit time unit such as seconds. The vertical axis is Q-value, not probability. [Fig. 4, p. 7; §V-B explanation, p. 6](https://arxiv.org/pdf/2610.06926v1#page=7)

At marker 1, FAST approaches the object. At 2, routing favors $\pi_{0.5}$ for precise grasping. At 3, the marker drops and all Q-values fall. At 4, $\pi_{0.5}$ retries the grasp. At 5, FAST moves the secured object toward the goal; 6 marks completion. Blue arrows connect curve events to the corresponding image annotations.

Together, the images and curves illustrate the authors' interpretation of FAST for free-space movement and $\pi_{0.5}$ for manipulation and recovery. This is **one qualitative rollout**, not evidence of training with skill labels or assigning fixed roles, and not a causal intervention establishing that switching caused recovery. The authors observed no visible motion discontinuities, but all candidates used the same DROID action space. That observation cannot establish compatibility across arbitrary robots or action formats.

#### 5. Distribution Shift: Real-Robot Gains and a Simulation Failure

The real-robot OOD evaluation adds unseen distractor objects. Here the authors explicitly retrain all three base policies on in-distribution tasks, while evaluating without retraining the critic on the distractors. This is not exactly the same candidate preparation as the preceding zero-shot setup. **Table IV**, with ten trials per task, reports: [§V-B and Table IV, p. 6](https://arxiv.org/pdf/2610.06926v1#page=6)

| Task with distractors | $\pi_0$ | $\pi_{0.5}$ | $\pi_0$-FAST | SWAP |
| --- | --- | --- | --- | --- |
| Marker → bowl | 10% | 40% | 40% | **90%** |
| Cube from plate → bowl | 10% | 30% | 0% | **80%** |
| Average | 10% | 35% | 20% | **85%** |

SWAP exceeds the best fixed-policy average by 50 points here. OOD real-robot Random results are absent, so no comparison is invented.

But the five-candidate setting in **Table III(a)** supplies a counterexample. Dashes denote unreported values, not zero. [Table III(a), p. 6](https://arxiv.org/pdf/2610.06926v1#page=6)

| Method | LIBERO | LIBERO-Plus |
| --- | --- | --- |
| $\pi_0$ | 81% | 54.2% |
| OpenVLA-Spatial | 62% | 0% |
| OpenVLA-OFT | 50.6% | 69.0% |
| OpenVLA-OFT$_{\text{no-wrist}}$ | 59.8% | 63.5% |
| OpenVLA-OFT+ | 54.8% | **80.8%** |
| Random | — | 2.7% |
| SWAP | — | 62.65% |
| SWAP without $\pi_0$ | — | 70.65% |
| SWAP without OpenVLA-Spatial | — | 63.00% |

SWAP exceeds Random by 59.95 points but trails the best fixed candidate, OpenVLA-OFT+, by **18.15 points**. Removing the worst policy, OpenVLA-Spatial, changes performance by only 0.35 points. Removing $\pi_0$, the strongest in-distribution candidate, improves SWAP by 8.00 points, still below 80.8%.

The authors attribute this failure to an in-distribution preference for $\pi_0$ becoming inappropriate OOD. They interpret random routing's collapse as destructive switching between incompatible behaviors. The evidence supports a requirement that **candidate strengths and critic rankings remain useful after a shift**, not a guarantee of improvement from larger candidate sets. Variant checkpoint and training-budget details are also insufficient to equate these rows with the original OFT paper's strongest published results.

#### 6. When One Policy Dominates

**Table III(b)** evaluates two candidates: strong $\pi_{0.5}$ and weak $\pi_0$. [Table III(b) and §V-B, pp. 6–7](https://arxiv.org/pdf/2610.06926v1#page=6)

| Method | LIBERO | LIBERO-Plus |
| --- | --- | --- |
| $\pi_0$ | 81% | 49.4% |
| $\pi_{0.5}$ | 95% | **92.0%** |
| Random | 90% | 79.0% |
| SWAP | **97.5%** | 89.0% |

On LIBERO, SWAP is 2.5 points above the stronger candidate, on the two tasks held out from router training (tasks 5 and 9). On LIBERO-Plus, it exceeds Random by 10 points but trails the stronger candidate by **3.0 points**. The caption's “matches or slightly outperforms” should not be interpreted as outperforming the fixed candidate in both columns. Limiting the damage relative to random switching is different from guaranteeing preservation of the best policy's performance.

## VI. Conclusion and the Authors' Limitations

SWAP makes existing VLA differences useful through **learned selection during execution**. Instead of assigning a whole task to one model, it chooses the next chunk's controller from the observation and predicted downstream return. Real-robot success, successful trajectory length, and switching examples support its promise. Simulation counterexamples reveal its conditions. [§VI, p. 7](https://arxiv.org/pdf/2610.06926v1#page=7)

The authors identify small candidate pools, short tabletop tasks, weak terminal reward signals on longer tasks, and the systems cost of operating many heterogeneous models as limitations. The objective does not incorporate latency or monetary cost. Longer tasks may need intermediate subtask-completion signals; larger policy libraries require cost-aware objectives and deployment design.

## Review Interpretation and Open Questions

The central insight is that average model performance can differ from usefulness in a particular state. On the robot, $\pi_{0.5}$ has lower standalone average success than FAST but is often selected for grasping and recovery. Yet this complementarity need not survive OOD. **Routing generalization** requires both interpreting a new scene and correctly estimating relative candidate strengths in it.

Three follow-up checks would be useful. A learned episode-start router could isolate the extra benefit of mid-execution switching. Switching costs or hysteresis could retain the previous policy when Q differences are small, with joint measurement of switching frequency, success, and elapsed time. Finally, critic uncertainty and incorrect preferences could be examined when candidate rankings invert, as in Table III(a). These are review proposals, not executed SWAP experiments.

## Review interpretation and open questions

**Review interpretation.** Complementary candidates and reliable value estimates under distribution shift determine the benefit of routing.

## Sources and reading scope

- Paper: *SWAP: Stepwise Action Policy Routing for Vision-Language-Action Models* · Mousumi Das et al. · first public date 2026-10-03 · [reviewed version](https://arxiv.org/abs/2610.06926v1) · reviewed 2026-10-23.
- Scope read: Version 1, sections I–VI, Algorithm 1, equation (1), Tables I–IV, and Figures 2 and 4. The paper has no original formal theorem or proof section.
- Figure sources: Figures 2 (thumbnail) and 4 are quoted for commentary from the authors’ v1 PDF, rasterized to lossless WebP with content retained. Author rights remain; arXiv’s nonexclusive distribution license is not a Creative Commons reuse license. [Paper](https://arxiv.org/abs/2610.06926v1).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
