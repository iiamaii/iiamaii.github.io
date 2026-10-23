---
title: "[1/2] Single Global Merging: why a final average recovers performance after sparse communication"
description: "Read the experiments, theory, and limits in original section order, separating communication timing and three evaluation targets."
date: "2026-10-23"
publishedAt: "2026-10-23T12:14:59+09:00"
updatedAt: "2026-10-23T08:56:25+00:00"
topics: ["decentralized learning", "model merging", "distributed optimization", "federated learning", "communication efficiency"]
visibility: "public"
paperTitle: "On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning"
paperPublishedDate: "2025-07-09"
authors: ["Tongtian Zhu", "Tianyu Zhang", "Mingze Wang", "Zhanpeng Zhou", "Can Wang"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2507.06542v4"
thumbnail: "/assets/reviews/single-global-merging/thumbnail.webp"
lang: "en"
translationKey: "single-global-merging"
thumbnailAlt: "Original Figure 1(b): global test accuracy recovers sharply after the final weight average in sparse decentralized ResNet-18 training."
---

This paper studies the large recovery obtained by globally merging models trained on heterogeneous data with sparse neighbor communication. In Tiny ImageNet with 32 agents and SGD, Figure 1(b) shows accuracy rising from approximately 20% to 48% after uniform averaging, with about 60 expected neighbor exchanges per agent over 300 rounds. [Section 4 and Figure 1](https://arxiv.org/pdf/2507.06542v4#page=2) **Review interpretation.** It motivates evaluating both individual and averaged models, while graph-read gains and actual communication time remain different quantities.

This review follows **§1→§2→§3→§4→§5→§6** of arXiv v4, the ICLR 2026 version. [[2/2] Appendices and proofs](/en/reviews/single-global-merging-theory/) follows A–D, retaining experimental details and expanding intermediate calculations. The observed recovery in accuracy and the scope of its theoretical explanation need separate assessment.

Open the body images to inspect axes and legends at full resolution.

## 1. Introduction: from whom to communicate with to when

Decentralized learning replaces a central parameter server with agents that update on their own data and exchange parameters with neighbors. Sparse communication lets their models diverge; non-IID data can also make each model specialize in a narrow set of classes.

Much prior work studies the communication graph: **who exchanges information with whom**. The authors ask whether the same communication budget has different effects depending on **when it is allocated**. They activate dense synchronization in a selected window, move that window through training, and eventually reduce it to one final global merge. [§1, pp.1–3](https://arxiv.org/pdf/2507.06542v4#page=1)

[![Figure 1. Final merging for pretrained CLIP and scratch ResNet, an interpolated loss landscape, and the communication structures of three training modes.](/assets/reviews/single-global-merging/figure1.webp)](/assets/reviews/single-global-merging/figure1.webp)

*Original Figure 1(a–d). Panel (a) uses pretrained CLIP ViT-B/32; (b) uses ResNet-18 without pretraining. Blue denotes federated training, orange sparse decentralized training, and green communication-free local training. In (a,b), the horizontal axis is communication rounds and the vertical axis average global test accuracy. The final orange jump is the central observation. Panel (c) shows interpolated cross-entropy loss, not accuracy; (d) illustrates communication structures. CC BY 4.0; extracted from the PDF and converted to lossless WebP.*

Figure 1(b) is the thumbnail because it isolates the important change. Orange and green look similarly weak during training but behave very differently after averaging. **“Single global merging” does not mean one communication event in the entire run.** The orange models have been communicating sparsely throughout training.

## 2. Related Work: what changes here?

The authors compare temporal communication allocation in parallel, federated, and decentralized learning. Federated learning usually maintains a global model at a server. The decentralized setting studied here has no model shared by all agents throughout training. Conclusions about early alignment therefore need to be transferred with care, after matching data distributions and evaluation targets.

The following five primary studies were read in selected sections to check the paper's positioning and theoretical dependencies. This is not an exhaustive survey of recent follow-up work.

Section 2 begins with local-update scheduling and implicit bias in Local SGD, then discusses early alignment in federated learning, which retains a global model. Fully decentralized learning instead evaluates local models on a global distribution. The authors present Kong et al.’s IID evidence in the regime where local and global objectives coincide, and defer implicit-bias and model-merging background to B.2–B.3. [Section 2 and appendix B](https://arxiv.org/pdf/2507.06542v4#page=3)

### Review supplement: scope of the source comparisons

| Primary study | Existing result | Connection and distinction |
| --- | --- | --- |
| [Koloskova et al., 2020](https://proceedings.mlr.press/v119/koloskova20a.html), §5–6 | Unified DSGD analysis with changing graphs, local updates, and non-IID data | The present paper tries to retain a constructive higher-order effect of model disagreement. Earlier theory already covered heterogeneity. |
| [Kong et al., 2021](https://proceedings.mlr.press/v139/kong21a.html), §3–4 | A critical consensus distance under which centralized convergence rates can be recovered; phase-wise communication control | Matching centralized rates without perfect consensus is an existing idea. The present emphasis is final merging under non-IID data and a different higher-order explanation. |
| [Zhu et al., 2023](https://proceedings.mlr.press/v202/zhu23e.html), §3–4 and C.2 | An asymptotic connection between DSGD and average-direction SAM | This is a direct theoretical dependency. Its common-distribution IID sampling matters when transferring the argument to heterogeneous local losses. |
| [Wortsman et al., 2022](https://proceedings.mlr.press/v162/wortsman22a.html), §1–2 | Model soups average models independently fine-tuned from a shared pretrained checkpoint | The present paper investigates sparse communication with different initializations and heterogeneous data. Its scratch-model failure without communication is not a universal impossibility result for independent averaging. |
| [de Vos et al., 2023](https://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html), §3 | Benefits of randomized communication for mixing and convergence | This motivates random graphs, but different random-peer protocols have different stochasticity properties. They do not automatically satisfy the same assumptions. |

Section 4.2 also distinguishes earlier final averaging in D-PSGD, periodic global averaging, and SCSP with gradient sparsification. The intended question is less whether a final averaging operator exists and more **why averaging can remain effective after long local updates and strong heterogeneity**. Those additional originals were not independently read here; their comparison is attributed to the target paper. [§2, §4.2, B](https://arxiv.org/pdf/2507.06542v4#page=3)

## 3. Notations and Preliminaries: distinguish three evaluations

### 3.1 Non-IID Decentralized Learning

There are $m$ agents. Agent $k$ has parameters $\theta_k\in\mathbb R^d$ and distribution $\mathcal D_k$. Equation (1) defines the global population risk:

$$
\mathcal L(\theta)=\frac1m\sum_{k\in\mathcal V}\mathcal L_k(\theta),
\qquad
\mathcal L_k(\theta)=\mathbb E_{\xi_k\sim\mathcal D_k}\mathcal L(\theta;\xi_k).
$$

Agents alternate local optimization and neighbor aggregation using a mixing matrix $W^{(t)}$. The theory assumes doubly stochastic mixing, with both row and column sums equal to one, to preserve the simple average.

**Separate optimization from merging.** With SGD as Algorithm 1's optimizer, the intended post-update mixing is written below. The send-line indexing discrepancy is discussed separately.

$$
\begin{aligned}
\theta_k^{(t+1/2)}
&=\theta_k^{(t)}-\eta\nabla\mathcal L_k(\theta_k^{(t)};\xi_k^{(t)}),\\
\theta_k^{(t+1)}
&=\sum_l W_{k,l}^{(t)}\theta_l^{(t+1/2)}.
\end{aligned}
$$

The first line learns from local data; the second mixes neighbor information. Weight one on the agent itself gives communication-free training, while weights $1/m$ on all agents give a global average. Sparse training lies between these cases and adds one final global average. Experiments perform 100 local mini-batch updates between communication rounds, so one theoretical update and one experimental communication round are different time units.

The global average is

$$
\bar\theta^{(t)}=\frac1m\sum_{k=1}^m\theta_k^{(t)}
$$

Defining it for analysis does not mean it is materialized and broadcast every round. The final global merge turns this average into the actual output model.

The distribution subscripts below are explanatory review notation.

| Evaluation | Expression | Question |
| --- | --- | --- |
| Each local model on its own distribution | $m^{-1}\sum_k\operatorname{Acc}_{\mathcal D_k}(\theta_k)$ | How well does each agent solve its own problem? |
| Each local model on the global distribution | $m^{-1}\sum_k\operatorname{Acc}_{\mathcal D}(\theta_k)$ | How well do individual agents solve the whole problem? This is the primary metric in Definition 1. |
| The averaged model on the global distribution | $\operatorname{Acc}_{\mathcal D}(\bar\theta)$ | How well does combining weights solve the whole problem? |

Averaging parameters is different from averaging accuracies. The jump across the final merge includes a **change in the evaluated parameters**, rather than many extra training updates to the same model. [§3.1, Definition 1, pp.3–4; C.2](https://arxiv.org/pdf/2507.06542v4#page=4)

There are minor notation inconsistencies: Equation (2) does not explicitly normalize each local sample sum by its sample count; Algorithm 1 sends pre-update $\theta^{(t)}$ but mixes post-update $\theta^{(t+1/2)}$. This review uses the population objective and the appendix's average-update analysis without silently treating these statements as identical.

### 3.2 Mergeability

Definition 2 calls models mergeable if there exist $w_k\in[0,1]$ such that

$$
\mathcal L\!\left(\sum_k w_k\theta_k\right)
\leq\sum_k w_k\mathcal L(\theta_k)
$$

The intention is that interpolation need not enter a bad region of a nonconvex loss.

The experiments instantiate the weights as the uniform average $w_k=1/m$. The definition’s nontriviality is examined in [technical part D](/en/reviews/single-global-merging-theory/#d-theory).

## 4. Empirical Observations

### 4.1 Increasing Impact of Communication in the Later Stages of Training

The window experiment uses CIFAR-100, ResNet-18, AdamW, 16 agents, and Dirichlet $\alpha=0.1$. AllReduce synchronization is active only in a chosen window. Outside it, an agent exchanges with one random peer with probability $R=0.2$. Moving equal-length windows tests communication timing under comparable communication budgets.

[![Figure 2. Peak and final accuracy for different dense-communication windows, and counterfactual averaged-model evaluation during sparse or communication-free training.](/assets/reviews/single-global-merging/figure2.webp)](/assets/reviews/single-global-merging/figure2.webp)

*Original Figure 2(a–c), p.5. In (a,b), the horizontal axis identifies the dense-communication window and the vertical axis is global test accuracy. Lighter bars show peak accuracy; darker bars show final accuracy. In (c), the horizontal axis is training rounds. Colors denote different comparisons across panels, so each legend must be read separately. CC BY 4.0; PDF extraction and lossless WebP conversion.*

Later windows particularly improve **final accuracy**. A synchronization event in the middle can briefly help, after which heterogeneous local updates reduce performance again. Peak and terminal accuracy answer different questions.

The caption specifies dense communication during $1/10$ of all rounds in (a) and $1/20$ in (b). However, (b) labels 15-round intervals while §4.2 says ten rounds. This inconsistency remains unresolved rather than being replaced with a guessed configuration. These experiments do not establish that early communication is universally unnecessary.

### 4.2 A Single Global Merging Significantly Improves Global Test Performance

The main Figure 1 uses Tiny ImageNet, 32 agents, $\alpha=0.1$, and SGD. Training runs for 300 communication rounds with 100 local mini-batch updates per round. Random-peer exchange with probability 0.2 yields about 60 exchanges per agent in expectation. One global uniform average is performed at the end.

In Figure 1(b), orange rises from roughly 0.2 before merging to roughly 0.48 afterward, compared with about 0.56 for the blue federated baseline. These are **approximate visual readings**, not exact statistics from raw logs or an independent reproduction. The observation is substantial recovery, not mathematical equivalence to FL.

The final average of communication-free scratch models almost fails in this setup. Pretrained CLIP in (a) also benefits from merging after sparse training. This should not be generalized to every independent average: shared-checkpoint fine-tuning in Model soups is a different successful regime. [§4.2, Remark 2, pp.5–6; C.1](https://arxiv.org/pdf/2507.06542v4#page=6)

**Communication accounting includes the last merge.** For model size $P$, agents $m$, and rounds $T$, the paper compares

$$
C_{\mathrm{dense}}=\mathcal O(2mPT),\qquad
C_{\mathrm{sparse+merge}}=\mathcal O(mRPT+2mP)
$$

Using the displayed volume terms on the same accounting basis gives $R/2+1/T$, about 10.3% for $R=0.2,T=300$. This is idealized transfer volume, not a measured 9.7-fold WAN speedup. One AllReduce itself contains multiple transfers. Implementing it with Ring-AllReduce removes the need for a central parameter server, not the cost of global aggregation.

### 4.3 Mergeability Persists under Limited but Nonzero Communication

Figure 2(c) separates sparse local models, in dark blue, from their **hypothetically averaged weights evaluated at that time**, in light blue. The latter already performs better during training: the opportunity for a good average is not created only at the last instant.

“Counterfactual” means the averaged model is tested but **not substituted back into the agents to continue training**. Dark and light orange give the communication-free comparison, where the averaged model performs poorly. Diagnostic evaluation aggregation has costs separate from the deployed training protocol.

The authors interpret this as mergeability without complete consensus. Figure 1(c) places 16 models at the vertices of a regular hexadecagon and uses Wachspress barycentric coordinates to interpolate parameters inside it, then evaluates cross-entropy. A low-loss interior helps explain why averaging might work. The polygon placement is a visualization choice, however, not a distance-preserving picture of a literal ring in parameter space or proof that every pairwise path has low loss. [§4.3, p.6; A](https://arxiv.org/pdf/2507.06542v4#page=6)

## 5. Theoretical Analysis: retaining a constructive disagreement term

### 5.1 Assumptions

Model disagreement is represented by covariance and consensus distance:

$$
\Gamma^{(t)}=\frac1m\sum_k
(\theta_k^{(t)}-\bar\theta^{(t)})
(\theta_k^{(t)}-\bar\theta^{(t)})^\top\in\mathbb R^{d\times d},
\qquad
\Xi_t^2=\operatorname{Tr}(\Gamma^{(t)}).
$$

The matrix $\Theta$ stacks agents as columns; $\bar\Theta$ replaces every column by their average.

$$
\Theta=[\theta_1,\ldots,\theta_m]\in\mathbb R^{d\times m},
\qquad
\bar\Theta=\Theta\frac{\mathbf1\mathbf1^\top}{m}
=[\bar\theta,\ldots,\bar\theta].
$$

Since $\|\Theta-\bar\Theta\|_F^2=m\Xi_t^2$, matrix contraction directly describes contraction of disagreement.

Zero distance means identical parameters. The mixing parameter $p$ appears in Assumption 1:

$$
\mathbb E_W\|\Theta W-\bar\Theta\|_F^2
\leq(1-p)\|\Theta-\bar\Theta\|_F^2.
$$

It is not the same quantity as peer-activation probability $R$. Assumption 2 requires $\mathcal L\in C^4$ and globally bounded derivatives of orders 1–4, with bounds $L_q$. Assumption 3 bounds gradient noise by $\sigma^2$ and gradient diversity by $\zeta^2$. These are substantial assumptions, not automatic properties of practical neural networks. The basic DSGD analysis alternates one local SGD update with mixing; it does not establish the same guarantee for 100 local updates per communication round or for AdamW. [§5.1, p.7](https://arxiv.org/pdf/2507.06542v4#page=7)

The constants have distinct jobs in the proof. Assumption 2 imposes

$$
\|\nabla^q\mathcal L(\theta)\|\leq L_q,\qquad q=1,2,3,4,\quad\forall\theta\in\mathbb R^d.
$$

$L_1$ controls gradient magnitude, $L_2$ curvature and smoothness, $L_3$ changing curvature, and $L_4$ the gradient Taylor remainder. The two conditions in Assumption 3 distinguish different sources of variation.

$$
\frac1m\sum_k\mathbb E_{\xi_k}
\|\nabla\mathcal L_k(\theta_k;\xi_k)-\nabla\mathcal L_k(\theta_k)\|^2
\leq\sigma^2.
$$

The first bounds within-agent stochastic mini-batch noise.

$$
\frac1m\sum_k
\|\nabla\mathcal L_k(\theta_k)-\nabla\mathcal L(\theta_k)\|^2
\leq\zeta^2.
$$

The second bounds systematic differences caused by local data distributions, even when gradients are evaluated at the same parameter. Parameter disagreement $\Xi_t$, sampling noise $\sigma$, and data heterogeneity $\zeta$ are distinct quantities.

### 5.2 Convergence Analysis

Instead of controlling all disagreement as detrimental error, the authors try to retain a structured effect involving third derivatives. Introduce review abbreviations

$$
g_t=\nabla\mathcal L(\bar\theta^{(t)}),\qquad
q_t=\nabla\operatorname{Tr}
\bigl(\nabla^2\mathcal L(\bar\theta^{(t)})\Gamma^{(t)}\bigr)
$$

with covariance held fixed under differentiation. The intended approximation is an average update direction $g_t+q_t/2$. The sign of $g_t^\top q_t$ can then be useful. The vector $q_t$ describes how curvature changes in the directions in which models are dispersed.

The [supplementary Taylor derivation](/en/reviews/single-global-merging-theory/#d-theory) is developed in the technical part.

[![Table 1. Convergence-rate comparison between parallel SGD, earlier DSGD analysis, and the present disagreement-dependent analysis.](/assets/reviews/single-global-merging/table1.webp)](/assets/reviews/single-global-merging/table1.webp)

*Original Table 1, p.8. The target $\varepsilon$ is an average squared-gradient tolerance and $m$ the number of agents. Earlier communication/heterogeneity terms are compared with the present $U^{(t)}$. The theorem/table coefficient correspondence is compared in [technical part D](/en/reviews/single-global-merging-theory/#d-theory). CC BY 4.0; complete table extraction and lossless WebP conversion.*

**Theorem 1 concerns** equally initialized DSGD and average stationarity:

$$
\frac1T\sum_{t=0}^{T-1}
\mathbb E\|\nabla\mathcal L(\bar\theta^{(t)})\|^2
\leq\varepsilon
$$

The paper claims an implicit bound combining the parallel-SGD terms $\mathcal O(\sigma^2/(m\varepsilon^2)+1/\varepsilon)$ with a disagreement correction $U^{(t)}$. Since that correction depends on the trajectory and on $T$, it is not a plug-in stopping-time formula.

Assumption 4 motivates a progressive-sharpening alignment

$$
g_t^\top q_t<0
$$

Moving along loss descent, $-g_t$, then increases the corresponding average curvature. Proposition 2 argues that, for $\eta>1/L_2$, $\|g_t\|\geq\mu_t>0$, and sufficiently small positive disagreement,

$$
U^{(t)}=\frac12(\eta L_2-1)g_t^\top q_t
+\mathcal O(\Xi_t^3)<0
$$

is possible. The favorable term is quadratic in $\Xi_t$ while the residual is cubic, allowing the former to dominate locally. Combined with stability, this requires $1/L_2<\eta<2/L_2$. Nonzero disagreement is not always beneficial.

**Review interpretation.** These formulas describe the authors’ proposed conditional explanation. Non-IID terms, assumptions, coefficients, and possible corrections are examined in [technical part D](/en/reviews/single-global-merging-theory/#d-theory).

The proof-dependency map is in [technical part D.1–D.9](/en/reviews/single-global-merging-theory/#d-theory).

### 5.3 A Theoretical Explanation for Communication Allocation

Corollary D.2 gives, with common initialization and a uniform gradient bound,

$$
\mathbb E\Xi_t^2
\leq\frac{24(1-p)\eta^2}{p^2}(\phi^2+\sigma^2).
$$

Increasing $p$ decreases this upper bound. Proposition 3 combines mixing with the gradient lower bound $\mu_t$ to propose a safe disagreement threshold. Holding other constants fixed, a smaller $\mu_t$ tightens the threshold and motivates stronger mixing. The authors interpret this as a reason to spend more communication late, when global gradients become small.

This is not a complete adaptive algorithm that estimates $p$ and $\mu_t$ online. Actual gradients need not decrease monotonically, and an expected squared-distance bound is not a per-run bound. The failure of this expression at $p=0$ also does not prove that every communication-free set of models is unmergeable. [§5.3, Proposition 3, p.10; D.9](https://arxiv.org/pdf/2507.06542v4#page=10)

## 6. Implications and Discussions

The first implication is for model merging. Lightweight synchronization might make models trained on different distributions compatible for later averaging, complementing shared pretrained checkpoints and post-hoc permutation alignment. The second is adaptive decentralized learning in which communication responds to training state. [§6, p.10](https://arxiv.org/pdf/2507.06542v4#page=10)

The most actionable evaluation lesson is to measure **both individual models' global accuracy and the final averaged model's global accuracy**. Communication cost, data access, BatchNorm treatment, initialization, and test-set-based tuning should be documented too. The paper provides vision classification observations, not verified LLM-training gains or measured WAN time and cost savings.

The appendices vary heterogeneity, optimizer, topology, initialization, and final topology-constrained gossip. Those comparisons are retained in A–C of [[2/2] Appendices and proofs](/en/reviews/single-global-merging-theory/).

## Review interpretation and open questions

**Review interpretation.** It motivates evaluating both individual and averaged models, while graph-read gains and actual communication time remain different quantities.

## Sources and reading scope

- Paper: *On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning* · Tongtian Zhu et al. · first public date 2025-07-09 · [reviewed version](https://arxiv.org/abs/2507.06542v4) · reviewed 2026-10-23.
- Scope read: Version 4, sections 1–6. Appendix A–D experimental conditions, additional results, and proof details are covered in part 2.
- Figure sources: The Figure-1(b) thumbnail and this part’s Figures 1–2 or C.5–C.6 are author assets from v4 (CC BY 4.0), PDF-extracted/rasterized to lossless WebP with data and labels retained. [Paper](https://arxiv.org/abs/2507.06542v4).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
