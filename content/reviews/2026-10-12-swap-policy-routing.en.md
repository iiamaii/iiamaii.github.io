---
title: "SWAP: Which VLA Should Control the Robot Next?"
description: "Follow SWAP's policy-routing MDP and IQL critic in the authors' section order, from real-robot success gains to cases where routing loses to the best policy under distribution shift."
date: "2026-10-12"
publishedAt: "2026-10-12T13:18:52+09:00"
updatedAt: "2026-10-23T15:26:19+00:00"
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

SWAP evaluates policy values from the current observation and switches vision-language-action (VLA) models between action chunks. On three real Franka tasks (ten trials each), it succeeds in 29 of 30 trials versus 19 for the best single candidate, $\pi_0$-FAST, but on LIBERO-Plus in simulation it trails the best single candidate in both the five-candidate setting (62.65% vs. 80.8%) and the two-candidate setting (89.0% vs. 92.0%). [Tables I and III](https://arxiv.org/pdf/2610.06926v1#page=5) **Review interpretation.** The real-robot advantage comes from a small evaluation, yet it is consistent in direction, with SWAP beating each task's best single candidate by two to four successes out of ten; the conclusion's claim that routing also beats fixed-policy execution in LIBERO simulation holds only on the two LIBERO tasks held out from router training (97.5% vs. 95%).

The article follows the paper's sections I–VI; because the paper has no theorems or proofs, it concentrates on the method (§IV) and experiments (§V).

## I. Introduction: Must One Policy Control the Entire Episode?

A VLA maps camera observations and language instructions to robot actions, and typically one model is selected and used until execution ends. The authors argue that this assumption is especially limiting in real manipulation. A policy that performs well in one scene configuration may fail in another, and a policy that is stronger overall may still make poor decisions in specific states. Distribution shift, compounding execution errors, and recovery behavior change the situation during execution, yet choosing one policy at the start of an episode commits the robot to a fixed behavior regardless. [§I, pp. 1–2](https://arxiv.org/pdf/2610.06926v1#page=1)

The proposed change is **policy routing**. Instead of choosing one policy for the whole episode, the robot decides again, after each short action chunk, which candidate policy should generate the next actions. Because a locally reasonable choice can eventually lead to failure while another choice enables recovery, routing must account for long-horizon outcomes. SWAP casts this as an offline reinforcement learning (RL) problem whose discrete actions are policy IDs.

## II. Background and Related Work

The paper connects VLAs, LLM routing, and value functions for robotics, in that order. Work on evaluating VLAs finds them still brittle under distribution shifts such as rephrased instructions, lighting changes, or added clutter, and fine-tuning on a new target task to mitigate these shifts can reduce robustness on the original training distribution. The authors instead aim to exploit the diverse behavior of existing VLAs within a single rollout. [§II, p. 2](https://arxiv.org/pdf/2610.06926v1#page=2)

LLM routing selects a model per request from context, estimated difficulty, or predicted performance, balancing cost factors such as latency against capability; a common theme is that different models have complementary strengths. On a robot, actions change later observations and outcomes, so selection must happen online throughout execution. Unlike a mixture of experts (MoE), which trains a gate jointly with its experts, SWAP treats candidate policies as fixed black boxes and learns only the router, so the authors note that new policies can be added without retraining any expert. **Review interpretation.** Adding a policy requires no expert retraining, but because the critic is learned from per-candidate rollout logs and policy-ID embeddings, a new candidate still requires new logs and critic retraining.

Value functions have served not only policy learning in offline RL but also SayCan's skill selection, failure detection, and deployment-time steering of VLAs. SWAP carries this idea into a **routing critic** that chooses among candidate policies.

### Review supplement: roles of five approaches

| Approach | What it selects or learns | Connection to SWAP and distinction |
| --- | --- | --- |
| [OpenVLA-OFT](https://arxiv.org/html/2502.19645v1), §IV–V | VLA action generation and fine-tuning | A source of simulation candidates. OFT improves the controller itself; SWAP selects among prepared controllers. |
| [RouteLLM](https://arxiv.org/html/2406.18665v4), §3.1 | Strong/weak LLM selection from preference data | Both select before calling a model. Robot actions also change subsequent observations, requiring sequential decisions about downstream return. |
| [IQL](https://arxiv.org/pdf/2110.06169v1), §4.1–4.3 | Q and state-value learning from logged data | The offline RL component used directly by SWAP. Here the action is a policy ID rather than a motor command. |
| [SayCan](https://arxiv.org/abs/2204.01691v2), abstract | Skills with linguistic relevance and physical feasibility | An antecedent for value-based executable skill selection. SWAP switches among VLAs attempting the same task, at chunk boundaries. |
| [V-GPS](https://arxiv.org/html/2410.13816v2), §5.1–5.2 | Value-based re-ranking of multiple actions sampled from a policy | V-GPS generates action candidates before scoring them. SWAP scores policy IDs first and requests actions only from the selected VLA. |

## III. Robot Policy Routing

The authors motivate policy routing with examples: a policy that is good at free-space motion may fail during contact-rich manipulation, while one that excels at precise object interaction may behave conservatively during exploration or recovery. The key assumption is **complementarity**: each policy behaves differently under identical observations, and no candidate dominates across all reachable states. [§III, p. 2](https://arxiv.org/pdf/2610.06926v1#page=2)

They formalize this as a Markov decision process with a candidate policy library:

$$
\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma),
\qquad \mathcal P=\{\pi_1,\pi_2,\ldots,\pi_K\}.
$$

$\mathcal S$ and $\mathcal A$ are state and robot-action spaces, $P$ is the transition model, $r$ the reward, and $\gamma$ the discount factor. The policy library $\mathcal P$ is distinct from transition $P$. The routing policy $\phi$ chooses a policy index $z_t$ from state $s_t$:

$$
\phi(z_t\mid s_t),\qquad z_t\in\{1,\ldots,K\},
\qquad a_t\sim\pi_{z_t}(\cdot\mid s_t).
$$

The objective maximizes the long-horizon performance of actions generated by these choices.

$$
\underset{\phi}{\arg\max}\;
\mathbb E_{\phi,P}
\left[\sum_{t=1}^{T}\gamma^{t-1}r(s_t,a_t)\right].
$$

The router's action is a **policy index**, not a joint angle or gripper command. Its chosen controller produces the motor actions. Selection concerns downstream success rather than only the apparent quality of the current motion.

The formal definition uses state $s_t$, whereas the implementation (§IV) takes camera observation $o_t$ and a language instruction as input.

## IV. SWAP: Stepwise Action Policy Routing

SWAP's routing critic predicts the long-horizon return of each candidate policy from the current observation. Training happens entirely offline on logged rollouts; at deployment, the learned critic switches policies during execution.

![SWAP Figure 2: offline dataset construction, IQL critic learning, and online policy selection](/assets/reviews/swap-policy-routing/swap-overview.webp)

[View at full size](/assets/reviews/swap-policy-routing/swap-overview.webp)

**Reading Figure 2.** The left panel collects successful and failed candidate rollouts. The center learns Q-values from an image, instruction, and policy ID. The right scores **every candidate ID**, requests an action chunk from the selected policy, executes it, and observes again. The loop is observation→selection→execution→new observation. [Fig. 2, p. 3](https://arxiv.org/pdf/2610.06926v1#page=3)

### Offline Dataset: Retain Failures Alongside Successes

Each candidate's logged trajectories are divided into chunks of $c$ actions. The authors explain that routing at chunk boundaries matches the VLAs' own inference cadence, adding no policy queries, and avoids interrupting a policy mid-chunk, where individual actions are not meaningful on their own. The routing dataset $\mathcal D$ contains the following transitions. [§IV, pp. 2–3](https://arxiv.org/pdf/2610.06926v1#page=2)

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

The method section (§IV) calls this reward “sparse and binary,” and the experiments assign **10** to a successful final transition and **0** to failures and all intermediate transitions. The authors consider this success signal particularly important for routing, because the quality of a policy choice may become apparent only several steps later. [§V-A, p. 5](https://arxiv.org/pdf/2610.06926v1#page=5)

### IQL Critic: Connect Today's Choice to Later Success

SWAP learns two action-value functions $Q_1,Q_2$ and a state-value function $V$. Equation (1) is central. [§IV, Eq. (1), p. 3](https://arxiv.org/pdf/2610.06926v1#page=3)

$$
\begin{aligned}
\mathcal L_Q={}&
\mathbb E_{(o_t,z_t,r_t,o_{t+c},d_t)\sim\mathcal D}\\
&\Bigg[\sum_{i=1}^{2}\Big(Q_i(o_t,z_t)\\
&\qquad-\big(r_t+\gamma(1-d_t)V(o_{t+c})\big)\Big)^2\Bigg].
\end{aligned}
$$

The temporal-difference target is $r_t+\gamma(1-d_t)V(o_{t+c})$. A completed success supplies the reward directly, while a nonterminal transition also includes the discounted value of the next observation. When the episode ends, $1-d_t=0$ removes any further value.

The authors explain that this training pushes the critic toward higher values for policy choices that led to success and lower values for choices observed in failed trajectories. Through this bootstrapping, unrewarded movement also acquires value from its connection to later success.

The equation applies $\gamma$ once per chunk transition, and the paper does not explain how this corresponds to low-level discounting by $\gamma^c$ over $c$ actions.

### Review Supplement: Expectile Regression

The paper says only, in one sentence, that $V$ is optimized separately by **expectile regression** over the learned Q-values, without giving the loss or the coefficient $\tau$. The IQL loss is shown below in IQL's notation $s,a,\psi,\hat\theta$. [IQL §4.1–4.2, Eq. (5), pp. 4–5](https://arxiv.org/pdf/2110.06169v1#page=4)

$$
L_V(\psi)=\mathbb E_{(s,a)\sim\mathcal D}
\left[L_2^\tau\big(Q_{\hat\theta}(s,a)-V_\psi(s)\big)\right],
\qquad L_2^\tau(u)=|\tau-\mathbf 1(u<0)|u^2.
$$

With $\tau>0.5$, residuals where Q exceeds V receive more weight, so V tracks the better-than-average Q-values among the choices present in the data. Original IQL then extracts a separate action policy from these values; because SWAP has a finite set of policy IDs, it compares their values directly at inference.

### Architecture and Inference: Choose the Policy Before Generating Actions

A **frozen** Qwen2.5-VL-3B encoder turns the image and instruction into a 2,048-dimensional representation. This is concatenated with a learned policy-ID embedding and fed to lightweight MLP double-Q critics. A separate V network is used in training. [§IV, pp. 3–4](https://arxiv.org/pdf/2610.06926v1#page=3)

Selection follows:

$$
z_t=\underset{z}{\arg\max}\;
\min\big(Q_1(o_t,z),Q_2(o_t,z)\big).
$$

The paper uses the smaller of the two critic estimates during both training and inference to reduce overestimation bias. Because it mixes a success reward of 10 with discounting, Q is not a calibrated success probability but a value for comparing candidates with one another.

Algorithm 1's execution can be read in four steps:

1. Encode the current inputs as $h_t=f(o_t,\text{instruction})$.
2. Compare $\min(Q_1(h_t,k),Q_2(h_t,k))$ for every candidate $k$.
3. Query only the selected $\pi_{z_t}$ and execute $c$ actions.
4. Update $t\leftarrow t+c$ and repeat from the new observation.

The equation's $Q_i(o_t,z)$ implicitly includes observation encoding; the algorithm makes $h_t$ explicit. “Online” here means **repeated selection during execution**, not weight updates during deployment. Selection avoids calling every VLA, but the 3B encoder and critic add cost at every chunk, and latency and memory usage are unreported.

## V. Experiments and Results

### A. Experiment Setup

Real-robot evaluation uses DROID's 7-DoF Franka Panda with external and wrist cameras. Tasks are marker into bowl, cube into cup, and cube from cloth into bowl. Candidates are $\pi_0$, $\pi_{0.5}$, and $\pi_0$-FAST. Public DROID checkpoints of the latter two are used zero-shot; $\pi_0$ showed no zero-shot capability on these tasks and is fine-tuned on a small amount of data. [§V-A, pp. 4–5](https://arxiv.org/pdf/2610.06926v1#page=4)

Ten successful and ten failed rollouts per policy per task yield **180 training trajectories**, and each method is evaluated on ten held-out trials per task. These are new trials of the same three tasks. Real-robot critic training runs for 4,000–8,000 epochs, and at inference Q-values are aggregated across camera views to handle viewpoint ambiguity and occlusion, though the aggregation rule is not given.

In simulation, logs contain 50 rollouts per policy per task on eight LIBERO-Spatial tasks; tasks 5 and 9 are held out for router evaluation. These two tasks are new to the router but were seen in training by the candidate VLAs. To probe generalization further, the out-of-distribution (OOD) variations of LIBERO-Plus Spatial are also evaluated. Real-robot and simulation critics are trained separately, with 200 epochs in simulation. Candidate sets vary across experiments.

Baselines are **fixed-policy execution** and **random routing**, which picks a candidate uniformly at each step. The authors use the latter to separate the benefit of switching itself from the contribution of the learned selection criterion. The paper does not report chunk size $c$, $\gamma$, the expectile coefficient $\tau$, learning rate, seed-level error, or LIBERO-Plus evaluation trial counts.

### B. Results

#### 1. Real-Robot Success Above Fixed Policies and Random Routing

The following preserves **every row of Table I**. Each cell is **success rate / mean action steps among successful trajectories**. [Table I, p. 5](https://arxiv.org/pdf/2610.06926v1#page=5)

| Task | $\pi_0$ | $\pi_{0.5}$ | $\pi_0$-FAST | Random | SWAP |
| --- | --- | --- | --- | --- | --- |
| Marker → bowl | 30% / 274 | 60% / 249 | 80% / 209 | 60% / 233 | **100% / 200** |
| Cube → cup | 50% / 358 | 20% / 420 | 40% / 372 | 60% / 336 | **90% / 224** |
| Cube from cloth → bowl | 40% / 288 | 50% / 201 | 70% / 227 | 80% / 262 | **100% / 155** |
| Reported average | 40% / 307 | 43.3% / 290 | 63.3% / 269 | 66.7% / 277 | **96.7% / 193** |

SWAP corresponds to 29 successes out of 30 trials, versus 19 for the best fixed policy, $\pi_0$-FAST. The authors' **+33.4 percentage points** is the difference of rounded averages; computed from counts it is about 33.3 points. The difference from random routing is 30 points.

Random routing also slightly exceeds the best fixed policy (66.7% vs. 63.3%). The authors attribute this to switching acting as crude recovery, since a stalled or drifting policy is replaced within one chunk, and they cite SWAP's further 30-point gain over random routing as showing that most of the benefit comes from learning which policy to switch to.

#### 2. Shorter Successful Trajectories

The reduction from reported mean 269 to 193 steps is:

$$
\frac{269-193}{269}\times100\approx28.3\%.
$$

These are mean lengths of successful trajectories only (unweighted means of task-level averages), and the set and number of successful trajectories differ between methods. Wall-clock time including encoding and model calls is not reported.

The authors interpret the reduction as SWAP choosing more suitable policies throughout execution, which removes unnecessary behavior and recovery from poor choices, and suggest it may follow from the critic optimizing discounted return, which implicitly favors steady progress. With the same success reward, reward received after fewer routing transitions is discounted less.

#### 3. Q-Values and Policy Preferences Along the Trajectory

The paper reports that $\pi_0$ consistently receives low values and is rarely selected, while the other two policies alternate as the preferred one. SWAP switches policies **8.7 times per rollout** on average, and each switch raises the predicted Q over the previously active policy by **+0.117** on average. [§V-B, pp. 5–6](https://arxiv.org/pdf/2610.06926v1#page=5)

**Table II** reports selection frequency over normalized thirds of trajectories:

| Segment | $\pi_0$-FAST | $\pi_{0.5}$ |
| --- | --- | --- |
| First third | 51.0% | 49.0% |
| Middle third | 22.6% | 77.4% |
| Final third | 36.3% | 63.7% |

Selections begin balanced, then favor $\pi_{0.5}$ in the middle. The authors summarize this as routing that “increasingly shifts toward $\pi_{0.5}$ as trajectories progress.” [Table II, p. 5](https://arxiv.org/pdf/2610.06926v1#page=5) **Review interpretation.** The share of $\pi_{0.5}$ falls from 77.4% to 63.7% in the final third, so the table shows a concentration in the middle rather than a monotonic increase.

The Q-analysis sentence in §V-B reads: “the weaker standalone policy $\pi_{0\text{-FAST}}$ has a lower value 36% of the time, while ($\pi_{0.5}$) is assigned the higher value 64% of the time.” **Review interpretation.** Table I gives $\pi_0$-FAST a standalone average of 63.3%, above $\pi_{0.5}$'s 43.3%, so “weaker” contradicts the table. Comparing only these two policies, “FAST is lower” and “$\pi_{0.5}$ is higher” are the same event, so 36% and 64% cannot both hold. Reading “lower” as “higher” makes the two shares sum to 100% and matches the unweighted mean of Table II's three segments (FAST 36.6%, $\pi_{0.5}$ 63.4%), so that correction seems close to the intent. Since the paper offers no clarification, policy preferences are read here from Table II and Figure 4.

#### 4. Figure 4: Movement, Grasping, and Recovery

![SWAP Figure 4: candidate Q-values and selected-policy band during a marker drop and recovery](/assets/reviews/swap-policy-routing/swap-q-values.webp)

[View at full size](/assets/reviews/swap-policy-routing/swap-q-values.webp)

**How to read the figure.** Start with the lower color band: teal is $\pi_0$-FAST, red $\pi_{0.5}$, and orange $\pi_0$. The band shows the policy selected at each decision step; the three curves above show each candidate's predicted Q-value. The horizontal axis follows execution progress, with no unit shown. [Fig. 4, p. 7; §V-B explanation, p. 6](https://arxiv.org/pdf/2610.06926v1#page=7)

At marker 1, FAST approaches the object. At 2, routing favors $\pi_{0.5}$ for precise grasping. At 3, the marker drops and all Q-values fall. At 4, $\pi_{0.5}$ retries the grasp. At 5, FAST moves the secured object toward the goal; 6 marks completion. Blue arrows connect curve events to the corresponding image annotations.

The authors take this example to show that the critic captures complementary strengths, preferring FAST for free-space motion and $\pi_{0.5}$ for fine-grained manipulation and recovery. They report that the individual policies, after a failure, made no recovery attempt beyond hovering until timeout. They also saw no visible motion discontinuities when switching, explaining that all candidates consume the same observation and output actions in the same DROID action space, so switching amounts to asking a different model for the next chunk from the current state. **Review interpretation.** The direct evidence for this division of roles is **one qualitative rollout**; whether the same division appears in other rollouts is supported only indirectly by Table II's segment frequencies.

#### 5. Distribution Shift: Real-Robot Gains and a Simulation Failure

The real-robot OOD evaluation adds unseen distractor objects to two tasks. Here all three base policies are retrained on the in-distribution tasks, so $\pi_{0.5}$ and $\pi_0$-FAST, used zero-shot earlier, are retrained checkpoints in this experiment. The critic is not retrained for the distractors. The second task is “cube on plate,” not Table I's “cube on cloth,” and the paper does not explain the change. **Table IV**, with ten trials per task, reports: [§V-B and Table IV, p. 6](https://arxiv.org/pdf/2610.06926v1#page=6)

| Task with distractors | $\pi_0$ | $\pi_{0.5}$ | $\pi_0$-FAST | SWAP |
| --- | --- | --- | --- | --- |
| Marker → bowl | 10% | 40% | 40% | **90%** |
| Cube from plate → bowl | 10% | 30% | 0% | **80%** |
| Average | 10% | 35% | 20% | **85%** |

SWAP exceeds the best fixed-policy average ($\pi_{0.5}$, 35%) by 50 points. Table IV has no random-routing row.

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

SWAP exceeds random routing by 59.95 points but trails the best fixed candidate, OpenVLA-OFT+, by **18.15 points**. Removing the worst policy, OpenVLA-Spatial, changes performance by only 0.35 points, whereas removing $\pi_0$, the strongest in-distribution candidate, improves SWAP by 8.00 points, still below 80.8%.

The authors attribute the gap to the preference for $\pi_0$. Because $\pi_0$ is the best policy on the critic's training distribution, that preference was correct for the data SWAP saw; the failure is that in-distribution quality does not predict OOD quality. They attribute random routing's collapse to destructive switching between behaviorally incompatible policies, and conclude that the router itself generalizes to novel scenes, the limiting factor being whether candidate strengths are stable across the shift. **Review interpretation.** Removing $\pi_0$ recovers only 8.00 of the 18.15-point gap, and 70.65% still trails OpenVLA-OFT+ by 10.15 points. The remaining gap is hard to attribute to $\pi_0$ alone, and this table by itself does not establish that the router generalizes.

#### 6. When One Policy Dominates

**Table III(b)** evaluates two candidates: strong $\pi_{0.5}$ and weak $\pi_0$. The LIBERO column covers only tasks 5 and 9, held out from router training but seen by the candidate policies. [Table III(b) and §V-B, pp. 6–7](https://arxiv.org/pdf/2610.06926v1#page=6)

| Method | LIBERO | LIBERO-Plus |
| --- | --- | --- |
| $\pi_0$ | 81% | 49.4% |
| $\pi_{0.5}$ | 95% | **92.0%** |
| Random | 90% | 79.0% |
| SWAP | **97.5%** | 89.0% |

On these two LIBERO tasks, SWAP is 2.5 points above the stronger candidate; on LIBERO-Plus, it exceeds random routing by 10 points but trails the stronger candidate by **3.0 points**. The text summarizes the LIBERO-Plus result as “closely matching the stronger standalone policy,” and the caption, which names the LIBERO-Plus setting, says SWAP “matches or slightly outperforms the performance of the dominant policy.” **Review interpretation.** “Slightly outperforms” holds only in the two-task LIBERO column; the LIBERO-Plus column shows no more than that mixing in a weak candidate costs less than random routing does.

## VI. Conclusion and the Authors' Limitations

The authors conclude that, across real-world DROID evaluations and large-scale LIBERO simulation, learned policy routing improves over fixed-policy execution and non-adaptive routing baselines. In their view, individual robot policies often have complementary strengths, routing explicitly between them yields more reliable manipulation, and policy routing is a simple but effective mechanism for composing heterogeneous robot policies. [§VI, p. 7](https://arxiv.org/pdf/2610.06926v1#page=7)

The authors list four limitations. The candidate sets are small, so scaling to dozens or hundreds of models, especially under stronger OOD conditions, remains open. Evaluation is restricted to short-horizon tabletop tasks; on longer tasks the sparse terminal reward becomes a weaker training signal, and intermediate subtask-completion signals would likely be needed. As the candidate set grows, operating many heterogeneous policies may make on-device deployment intractable. The objective optimizes only task return and ignores inference latency and monetary cost.

## Review interpretation and open questions

**Review interpretation.** The strongest evidence is the real-robot Table I. In every task, SWAP beats both that task's best fixed policy and random routing. There is also evidence of state-dependent usefulness: Figure 4 shows $\pi_{0.5}$, the policy with lower standalone success, being used for grasping and recovery, and Table II shows its use concentrating mid-trajectory. The limits concern sample size and scope. The real-robot evaluation has ten trials per task with no confidence intervals or repeated seeds, the OOD real-robot test covers two tasks without a random-routing comparison, and the simulation trial counts are unreported. A more fundamental limit is that in Table III(a), where distribution shift inverts the candidate ranking, the critic followed the ranking from its training distribution. **Routing generalization** therefore requires not only recognizing a new scene but also re-estimating the candidates' relative strengths in it.

Three questions remain. First, comparing against a learned router that selects only at episode start would isolate the extra benefit of switching mid-execution. Second, adding a switching cost or hysteresis that keeps the previous policy when Q differences are small, while jointly measuring switching frequency, success, and elapsed time, would show whether all 8.7 switches per rollout are needed. Third, one could test whether critic uncertainty estimates detect ranking inversions like the one in Table III(a).

## Sources and reading scope

- Paper: *SWAP: Stepwise Action Policy Routing for Vision-Language-Action Models* · Mousumi Das et al. · first public date 2026-10-03 · [reviewed version](https://arxiv.org/abs/2610.06926v1) · reviewed 2026-10-23.
- Scope read: Version 1, sections I–VI, Algorithm 1, equation (1), Tables I–IV, and Figures 2 and 4. The paper has no original formal theorem or proof section.
- Figure sources: Figures 2 (thumbnail) and 4 are quoted for commentary from the authors’ v1 PDF, rasterized to lossless WebP with content retained. Author rights remain; arXiv’s nonexclusive distribution license is not a Creative Commons reuse license. [Paper](https://arxiv.org/abs/2610.06926v1).
- Not verified: research-code execution and reproduction of training or evaluation.
