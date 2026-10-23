---
title: "[1/2] Single Global Merging: why a final global merge recovers performance after sparse communication"
description: "Read the experiments, theory, and limits in original section order, separating communication timing and three evaluation targets."
date: "2026-10-23"
publishedAt: "2026-10-23T12:14:59+09:00"
updatedAt: "2026-10-23T15:26:25+00:00"
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

This paper reports that a single global merge at the end of training sharply recovers global test performance for models trained on heterogeneous data with sparse neighbor communication, and offers a convergence analysis of the averaged model to explain it. For ResNet-18 trained from scratch with stochastic gradient descent (SGD) on Tiny ImageNet with 32 agents and Dirichlet $\alpha=0.1$ (Figure 1(b)), uniformly averaging models that made only about 60 expected neighbor exchanges per agent raises average global test accuracy from roughly 20% to 48% on the plot, approaching the federated-learning (FL) baseline at about 56%. [Section 4.2 and Figure 1](https://arxiv.org/pdf/2507.06542v4#page=2) **Review interpretation.** The ablation in which the average of communication-free models stays near 0% (Figure 2(c)) directly supports the empirical claim that limited but nonzero communication creates mergeability, but the convergence theory offered as the explanation assumes common initialization and decentralized SGD (DSGD) that mixes after every SGD step, so it does not directly cover experiments with distinct initializations, 100 local updates per round, or AdamW.

This review reads main-text §1–6 of arXiv v4, the ICLR 2026 version, in source order; [[2/2] Appendices and proofs](/en/reviews/single-global-merging-theory/) covers the experimental details and proof checks in Appendices A–D.

Open the body images to inspect axes and legends at full resolution.

## 1. Introduction: from whom to communicate with to when

Decentralized learning replaces a central parameter server with agents that update on their own data and exchange parameters with neighbors. Sparse communication lets their models diverge; non-IID data can also make each model specialize in a narrow set of classes.

Much prior work studies the communication graph: **who exchanges information with whom**. The authors ask whether the same communication budget has different effects depending on **when it is allocated**. They activate dense synchronization in a selected window, move that window through training, and eventually reduce it to a single global merge at the end of training. [§1, pp.1–3](https://arxiv.org/pdf/2507.06542v4#page=1)

[![Figure 1. Final merging for pretrained CLIP and scratch ResNet, an interpolated loss landscape, and the communication structures of three training modes.](/assets/reviews/single-global-merging/figure1.webp)](/assets/reviews/single-global-merging/figure1.webp)

*Original Figure 1(a–d). Panel (a) uses pretrained CLIP ViT-B/32; (b) uses ResNet-18 without pretraining. Blue denotes federated training, orange sparse decentralized training, and green communication-free local training. In (a,b), the horizontal axis is communication rounds and the vertical axis average global test accuracy. The final orange jump is the central observation. Panel (c) shows interpolated cross-entropy loss, not accuracy; (d) illustrates communication structures. CC BY 4.0; extracted from the PDF and converted to lossless WebP.*

Orange and green look similarly weak during training but diverge sharply after the final global merge. “Single global merging” does not mean one communication event in the entire run: the orange models have been communicating sparsely throughout training.

## 2. Related Work: what changes here?

Section 2 contrasts how parallel, federated, and decentralized learning allocate communication over time. In parallel learning, Gu et al. (2024) schedule local steps by analyzing the implicit bias of Local SGD. Federated learning extends the server-based paradigm to non-IID data but retains a global model; the authors argue that this has produced a broad consensus in the FL literature that “frequent, early-stage communication is considered essential” for aligning local models.

Fully decentralized learning has no central server, so the goal is not a single global model but local models that generalize to the global distribution. Work on communication allocation in decentralized learning has mostly addressed spatial allocation such as topology design; temporal allocation is rarely studied. In a pioneering IID study, Kong et al. (2021) showed that stronger early alignment to the global average can “modestly” improve test performance. The authors argue that this does not transfer to non-IID data, where local and global objectives differ, so it characterizes local rather than global generalization. Related work on the implicit bias of decentralized learning and on model merging is deferred to Appendices B.2–B.3. [Section 2 and appendix B](https://arxiv.org/pdf/2507.06542v4#page=3)

### Review supplement: scope of the source comparisons

| Primary study | Existing result | Connection and distinction |
| --- | --- | --- |
| [Koloskova et al., 2020](https://proceedings.mlr.press/v119/koloskova20a.html), §5–6 | Unified DSGD analysis with changing graphs, local updates, and non-IID data | Treats model disagreement as error to control. The present paper tries to keep part of it as a constructive higher-order term; this is the comparison in Table 1 of §5.2. |
| [Kong et al., 2021](https://proceedings.mlr.press/v139/kong21a.html), §3–4 | An analysis showing centralized convergence rates are recovered below a critical consensus distance, and IID experiments in which early consensus control improves performance | Section 2 cites the experimental side; the consensus-distance analysis in Appendices D.1–D.2 is also taken from this source. |
| [Zhu et al., 2023](https://proceedings.mlr.press/v202/zhu23e.html), §3–4 and C.2 | An asymptotic connection between DSGD and average-direction sharpness-aware minimization (SAM) | The direct starting point of the present theory; the source assumes IID sampling from a common distribution. |
| [Wortsman et al., 2022](https://proceedings.mlr.press/v162/wortsman22a.html), §1–2 | Model soups average models independently fine-tuned from a shared pretrained checkpoint | The present paper asks whether a little communication makes models averageable under different initializations and heterogeneous data. |
| [de Vos et al., 2023](https://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html), §3 | Benefits of randomized communication for mixing and convergence | Cited in §5.2 as evidence that random graphs mix quickly. |

## 3. Notations and Preliminaries: three evaluations

### 3.1 Non-IID Decentralized Learning

There are $m$ agents. Agent $k$ has parameters $\theta_k\in\mathbb R^d$ and distribution $\mathcal D_k$. Equation (1) defines the global population risk:

$$
\mathcal L(\theta)=\frac1m\sum_{k\in\mathcal V}\mathcal L_k(\theta),
\qquad
\mathcal L_k(\theta)=\mathbb E_{\xi_k\sim\mathcal D_k}\mathcal L(\theta;\xi_k).
$$

Agents alternate local optimization and neighbor aggregation using a mixing matrix $W^{(t)}$. The theory assumes doubly stochastic mixing, with both row and column sums equal to one, to preserve the simple average.

**Update and mixing.** Algorithm 1 with SGD as the optimizer is the DSGD of the theory; one iteration is

$$
\begin{aligned}
\theta_k^{(t+1/2)}
&=\theta_k^{(t)}-\eta\nabla\mathcal L_k(\theta_k^{(t)};\xi_k^{(t)}),\\
\theta_k^{(t+1)}
&=\sum_l W_{k,l}^{(t)}\theta_l^{(t+1/2)}.
\end{aligned}
$$

The first line learns from local data; the second mixes neighbor information. Weight one on the agent itself gives communication-free training, while weights $1/m$ on all agents give a global average. Sparse training lies between these cases and adds one global average at the end. The experiments insert 100 local mini-batch updates ($H=100$) between communication events, so one experimental communication round differs from one iteration of the display above.

The global average is

$$
\bar\theta^{(t)}=\frac1m\sum_{k=1}^m\theta_k^{(t)}.
$$

It is an analytical quantity and is not distributed to agents during training. The final global merge turns this average into the actual output model.

Combining Definition 1 with Appendix C.2 gives three evaluations. The subscripts $\mathcal D_k$ and $\mathcal D$ are review notation that names the evaluation distribution.

| Evaluation | Expression | Question |
| --- | --- | --- |
| Each local model on its own distribution | $m^{-1}\sum_k\operatorname{Acc}_{\mathcal D_k}(\theta_k)$ | How well does each agent solve its own problem? |
| Each local model on the global distribution | $m^{-1}\sum_k\operatorname{Acc}_{\mathcal D}(\theta_k)$ | How well do individual agents solve the whole problem? This is the primary metric in Definition 1. |
| The averaged model on the global distribution | $\operatorname{Acc}_{\mathcal D}(\bar\theta)$ | How well does combining weights solve the whole problem? |

Averaging parameters is different from averaging accuracies. The jump across the final global merge comes not from extra training but from switching the evaluated object from individual models (second row) to the averaged model (third row). [§3.1, Definition 1, pp.3–4; C.2](https://arxiv.org/pdf/2507.06542v4#page=4)

There are minor notation inconsistencies. Equation (2) does not normalize each local sample sum by its sample count, and Algorithm 1 sends pre-update $\theta^{(t)}$ but mixes post-update $\theta^{(t+1/2)}$. The display above follows the mixing line and the proof in Appendix D.1.

### 3.2 Mergeability

Definition 2 calls models mergeable if there exist $w_k\in[0,1]$ such that

$$
\mathcal L\!\left(\sum_k w_k\theta_k\right)
\leq\sum_k w_k\mathcal L(\theta_k)
$$

The intention is that interpolation need not enter a bad region of a nonconvex loss.

The experiments instantiate the weights as the uniform average $w_k=1/m$. The definition’s nontriviality is examined in [[2/2] D](/en/reviews/single-global-merging-theory/#d-theory).

## 4. Empirical Observations

### 4.1 Increasing Impact of Communication in the Later Stages of Training

The window experiment uses CIFAR-100, ResNet-18, AdamW, 16 agents, and Dirichlet $\alpha=0.1$. AllReduce synchronization is active only in a chosen window. Outside it, an agent exchanges with one random peer with probability $R=0.2$. Moving equal-length windows tests communication timing under comparable communication budgets.

[![Figure 2. Peak and final accuracy for different dense-communication windows, and counterfactual averaged-model evaluation during sparse or communication-free training.](/assets/reviews/single-global-merging/figure2.webp)](/assets/reviews/single-global-merging/figure2.webp)

*Original Figure 2(a–c), p.5. In (a,b), the horizontal axis identifies the dense-communication window and the vertical axis is global test accuracy. Lighter bars show peak accuracy; darker bars show final accuracy. In (c), the horizontal axis is communication rounds; blue is sparse communication, orange is no communication, dark denotes individual models, and light denotes the averaged model. CC BY 4.0; PDF extraction and lossless WebP conversion.*

Panel (a) uses dense communication in $1/10$ of all rounds and (b) in $1/20$. The authors summarize that shifting the budget toward later stages yields substantial improvements, particularly in final accuracy. On the plot, peak accuracy (light bars) rises steadily for later windows, whereas final accuracy (dark bars) jumps only for the last window that includes the end of training, to about 0.55 in (a) and 0.53 in (b); all earlier windows stay around 0.24–0.28. As the inset in (a) shows for synchronization during rounds 150–180, a mid-training window raises accuracy while it lasts, but accuracy drops again once sparse communication resumes. [§4.1 and Figure 2, p.5](https://arxiv.org/pdf/2507.06542v4#page=5) **Review interpretation.** The late-allocation effect on final accuracy therefore comes almost entirely from the window that ends training with dense communication, which is the same phenomenon as the final global merge in §4.2.

The horizontal axis of (b) labels 15-round intervals (e.g., 285–300), whereas §4.2 describes the window length as ten rounds.

### 4.2 A Single Global Merging Significantly Improves Global Test Performance

Because improvements persist when the window is reduced to ten rounds, the authors ask what happens when it shrinks to a single round, and apply dense communication only once, as a global merge at the end of training. The main Figure 1 uses Tiny ImageNet, 32 agents, $\alpha=0.1$, and SGD, with 300 communication rounds of 100 local mini-batch updates each and an exchange with one random peer with probability 0.2 per round.

In Figure 1(b), orange rises from about 0.2 before the final global merge to about 0.48 afterward, while the blue FL baseline is about 0.56. Pretrained CLIP in (a) also rises from about 0.38 to 0.55, approaching FL at about 0.59 (values read from the plot). By contrast, the green curve for the communication-free ResNet trained from scratch falls to nearly zero after the final global merge. The authors report the same gain across datasets, architectures, and topologies in the appendix and suggest that the potential of decentralized learning may be underestimated. [§4.2 and Figure 1, pp.2 and 6](https://arxiv.org/pdf/2507.06542v4#page=6)

**Comparisons.** D-PSGD introduced final global merging under IID settings but did not analyze the performance gap before and after merging. Periodic global averaging by Chen et al. (2021) requires global communication every $H=48$ steps, whereas this paper recovers performance with a single merge. Skew-Compensated Sparse Push (SCSP), which also includes a final global merging step, uses top-$k$ gradient sparsification over a fixed topology and a single local step ($H=1$). This paper instead studies mergeability under topological sparsification (sparse gossip), long local updates ($H=100$), and strong heterogeneity. The new question the authors claim is why local models retain mergeability despite extremely limited communication and high heterogeneity.

**Is the gain trivial? (Remark 2).** Exchanging with one peer with probability $R=0.2$ over $T=300$ rounds gives about 60 exchanges per agent in expectation, which might look like several implicit global aggregations on average. The authors reply that, if this were the explanation, local models would already be aligned and perform close to the merged model; instead, in Figure 1(b) the pre-merge local models stay close to the communication-free case and only the single merge produces the large gain. [§4.2 and Remark 2, p.6](https://arxiv.org/pdf/2507.06542v4#page=6)

**Communication cost.** For model size $P$, agents $m$, rounds $T$, and expected peers per round $R$, the paper compares Ring-AllReduce-based dense synchronization with this setup:

$$
C_{\mathrm{dense}}=\mathcal O(2mPT),\qquad
C_{\mathrm{sparse+merge}}=\mathcal O(mRPT+2mP).
$$

The $2mP$ term is the cost of the final global merge via Ring-AllReduce. Counting only the displayed terms gives a transfer-volume ratio of $R/2+1/T$, about 10.3% for $R=0.2$ and $T=300$. The authors add that where a global merge is impractical, it can be approximated by several gossip rounds (C.3.4 of [[2/2]](/en/reviews/single-global-merging-theory/)).

### 4.3 Mergeability Persists under Limited but Nonzero Communication

Section 4.3 asks whether the effect of global merging is specific to the end of training. Figure 2(c) compares global test accuracy under $R=0.2$ sparse communication (blue) and no communication (orange), for individual models (dark) and for the weights averaged at each round (light). “Counterfactual” means the averaged model is computed and tested but never fed back into training. Because light blue stays above dark blue, the authors conclude that local models remain mergeable at all stages.

In the ablation without communication, the averaged model's accuracy stays close to zero (light orange), which the authors take as evidence that mergeability does not arise from the local models themselves. Before merging, individual models under sparse communication (dark blue, about 0.24 at the end on the plot) and without communication (dark orange, about 0.20) perform similarly, yet their averages split into about 0.48 and nearly zero. From this contrast the authors conclude that extremely limited but nonzero communication enables mergeability. **Review interpretation.** On the plot, light blue lies below dark blue until about round 50, so “throughout training” does not hold for the earliest phase.

The authors also describe this as mergeability without consensus. Earlier gossip work suggested that local models can converge to a similar state even with little communication; under heterogeneous data, the authors find that local models do not reach a single consensus point yet remain mergeable. Figure 1(c) places 16 models at the vertices of a regular hexadecagon, interpolates parameters inside it with Wachspress barycentric coordinates, and plots cross-entropy; the authors read it as local models sitting on a ring-like high-loss region around a central low-loss basin. They stress that this is a harder cross-initialization, cross-distribution merge without a shared pretrained checkpoint. [§4.3, p.6; A](https://arxiv.org/pdf/2507.06542v4#page=6) **Review interpretation.** Placing the vertices on a circle is a choice of the visualization tool, so the ring shape itself comes from that layout; what the figure shows is that the loss inside the convex hull is lower than at the vertices.

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

It is a different quantity from the peer-activation probability $R$. [§5.1, p.7](https://arxiv.org/pdf/2507.06542v4#page=7)

Assumption 2 requires $\mathcal L\in C^4$ and global bounds $L_q$ on derivatives of orders 1–4:

$$
\|\nabla^q\mathcal L(\theta)\|\leq L_q,\qquad q=1,2,3,4,\quad\forall\theta\in\mathbb R^d.
$$

$L_1$ controls gradient magnitude, $L_2$ curvature and smoothness, $L_3$ changing curvature, and $L_4$ the gradient Taylor remainder. Assumption 3 bounds gradient noise by $\sigma^2$ and gradient diversity across agents by $\zeta^2$; the two conditions address different sources of variation.

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

The [supplementary Taylor derivation](/en/reviews/single-global-merging-theory/#d-theory) relating curvature to model covariance is in D.5 of [2/2].

[![Table 1. Convergence-rate comparison between parallel SGD, earlier DSGD analysis, and the present disagreement-dependent analysis.](/assets/reviews/single-global-merging/table1.webp)](/assets/reviews/single-global-merging/table1.webp)

*Original Table 1, p.8. The target $\varepsilon$ is an average squared-gradient tolerance and $m$ the number of agents. Earlier communication/heterogeneity terms are compared with the present $U^{(t)}$. The theorem/table coefficient correspondence is compared in [[2/2] D.7](/en/reviews/single-global-merging-theory/#d-theory). CC BY 4.0; complete table extraction and lossless WebP conversion.*

**Theorem 1 concerns** equally initialized DSGD and average stationarity:

$$
\frac1T\sum_{t=0}^{T-1}
\mathbb E\|\nabla\mathcal L(\bar\theta^{(t)})\|^2
\leq\varepsilon
$$

The paper gives an iteration count that adds a disagreement correction $\sum_tU^{(t)}$ to the parallel-SGD terms $\mathcal O(\sigma^2/(m\varepsilon^2)+1/\varepsilon)$. The authors themselves note that this is an implicit bound depending on $U^{(t)}$, not a closed form, and that it serves as a bridge to the sign conditions on $U^{(t)}$ derived in Propositions 2 and 3. With a single agent, or with full synchronization that keeps all models identical, $U^{(t)}=0$ and the parallel-SGD rate is recovered.

Assumption 4 motivates a progressive-sharpening alignment

$$
g_t^\top q_t<0
$$

Moving along loss descent, $-g_t$, then increases the corresponding average curvature. Proposition 2 argues that, for $\eta>1/L_2$, $\|g_t\|\geq\mu_t>0$, and sufficiently small positive disagreement,

$$
U^{(t)}=\frac12(\eta L_2-1)g_t^\top q_t
+\mathcal O(\Xi_t^3)<0
$$

is possible. The favorable term is quadratic in $\Xi_t$ while the residual is cubic, allowing the former to dominate locally. Combined with stability, this requires $1/L_2<\eta<2/L_2$, an interval the authors note coincides with the regime of oscillatory convergence for quadratic objectives in classical optimization theory. Whether this expansion of the average update holds for non-IID losses, and how the coefficients and proof steps depend on each other, is examined in [[2/2] D](/en/reviews/single-global-merging-theory/#d-theory).

**Why Limited but Nonzero Communication Enables Mergeability.** Citing Corollary D.2, the paper bounds the expected consensus distance by $\mathcal O((1-p)/p^2)$. The explicit form in Appendix D.2, under a uniform bound $\phi^2$ on the average squared local gradient, is

$$
\mathbb E\Xi_t^2
\leq\frac{24(1-p)\eta^2}{p^2}(\phi^2+\sigma^2).
$$

Here $p=1$ is a fully connected topology and $p=0$ is local training without communication. The authors argue that random communication graphs achieve $p=\Theta(1)$ with little communication and rapidly shrink $\Xi_t$ (de Vos et al., 2023), so they can satisfy the condition of Proposition 2 and ensure mergeability even under extremely limited communication; this is why random topologies are the main experimental setup. At $p=0$, by contrast, the right side becomes infinite, $\Xi_t$ may diverge, and the condition of Proposition 2 can no longer be met, which explains why fully local models cannot be reliably merged (green curve in Figure 1(b)). [§5.2, p.9](https://arxiv.org/pdf/2507.06542v4#page=9) **Review interpretation.** The argument relies on the experimental $R=0.2$ protocol satisfying Assumption 1 with doubly stochastic mixing and a constant $p$, but the paper does not report the value of $p$ or the mixing-matrix construction for that protocol.

### 5.3 A Theoretical Explanation for Communication Allocation

Proposition 3 (Critical Consensus Edge) is a sufficient condition under Assumptions 1–4, $\eta>1/L_2$, and $\Xi_t\leq1$: if the square root of the bound above is below a threshold set by the degree of progressive sharpening $\gamma^*$ and the gradient lower bound $\mu_t$, then $U^{(t)}<0$. The authors derive guidance from the fact that the left side decreases in $p$ while the right side increases in $\mu_t$. Early in training, when the averaged model is far from a minimum, $\mu_t$ is large and infrequent communication (small $p$) suffices; late in training, $\mu_t$ shrinks and frequent communication (large $p$) becomes necessary. They see this as consistent with the late-communication observations of §4. [§5.3, Proposition 3, p.10](https://arxiv.org/pdf/2507.06542v4#page=10)

**Review interpretation.** Turning this guidance into an adaptive schedule would require estimating $p$ and $\mu_t$ during training, which the paper leaves to future algorithms in §6. The substitution of an expected squared bound into a per-run condition is discussed in [[2/2] D.9](/en/reviews/single-global-merging-theory/#d-theory).

## 6. Implications and Discussions

The first implication is for model merging. Lightweight synchronization might make models trained on different distributions compatible for later averaging, complementing shared pretrained checkpoints and post-hoc permutation alignment. The second is adaptive decentralized learning in which communication responds to training state. [§6, p.10](https://arxiv.org/pdf/2507.06542v4#page=10)

Appendix experiments that vary heterogeneity, optimizer, topology, initialization, and final gossip are covered in C of [[2/2] Appendices and proofs](/en/reviews/single-global-merging-theory/).

## Review interpretation and open questions

**Review interpretation.** The strongest part of the paper is its measurement design. Because individual and averaged models are evaluated separately on the global distribution, the difference between sparse communication and no communication, invisible in the individual models, becomes visible in the averaged model (§4.3), and Remark 2 rests on the same contrast. The appendices add results in the same direction across optimizers, topologies, initializations, and heterogeneity levels. On the other hand, the “performance close to federated learning” of contribution (1) means, on the plots, about 8 points below FL for ResNet and about 4 points below for CLIP.

The evidence has three limits. First, hyperparameters were selected by global test performance, and the number of runs and error bars are not reported ([2/2] A and C). Second, the experiments are low-resolution vision classification, and communication cost is a transfer-volume count, so training time on a real network is not measured. Third, the theory assumes common initialization, DSGD that mixes after every SGD step, and progressive sharpening for every PSD matrix and every parameter (Assumption 4), whereas the authors themselves state that most experiments use distinct initializations, and Figure 2 uses AdamW with 100 local updates per round.

The abstract's claim to be “the first to establish that the globally merged model of decentralized SGD can match the convergence rate of parallel SGD” is new in its proof route rather than in its conclusion. Kong et al. (2021) already showed that DSGD converges as fast as centralized training when the consensus distance stays below a critical value, and the consensus-distance analysis of Appendices D.1–D.2 is taken from that source. What is new is retaining the sign of the quadratic disagreement term and using it as a beneficial term; and because Theorem 1 is an implicit bound in $U^{(t)}$, the “match” follows only when all conditions of Propositions 2 and 3 hold. Whether they hold for non-IID losses is examined in [[2/2] D](/en/reviews/single-global-merging-theory/#d-theory).

Two questions remain. The communication threshold at which mergeability appears cannot be located from Appendix C.3, which compares only $R=0.2$ and $R=1$. And how does the early phase seen in §4.3, where the averaged model trails the individual models until about round 50, square with Proposition 3's guidance that infrequent communication suffices early on?

## Sources and reading scope

- Paper: *On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning* · Tongtian Zhu et al. · first public date 2025-07-09 · [reviewed version](https://arxiv.org/abs/2507.06542v4) · reviewed 2026-10-23.
- Scope read: Version 4, sections 1–6. Appendix A–D experimental conditions, additional results, and proof details are covered in part 2.
- External sources consulted: the five studies in the §2 review-supplement table (sections listed there).
- Figure sources: The Figure-1(b) thumbnail and this part’s Figures 1–2 and Table 1 are author assets from v4 (CC BY 4.0), PDF-extracted/rasterized to lossless WebP with data and labels retained. [Paper](https://arxiv.org/abs/2507.06542v4).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
