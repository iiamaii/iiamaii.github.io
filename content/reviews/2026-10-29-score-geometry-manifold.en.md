---
title: "[1/4] Score Geometry: Shape and Density"
description: "Why small-noise scores encode shape more strongly than frequency, and what the five main theorems and experiments establish."
date: "2026-10-29"
publishedAt: "2026-10-29T08:55:43+09:00"
updatedAt: "2026-10-29T11:16:15+09:00"
topics: ["생성 모델", "확률과 기하"]
visibility: "public"
lang: "en"
translationKey: "score-geometry-manifold"
paperTitle: "When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis"
paperPublishedDate: "2025-09-29"
authors: ["Xiang Li", "Zebang Shen", "Ya-Ping Hsieh", "Niao He"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2509.24912v2"
thumbnail: "/assets/score-geometry/cover-1.svg"
thumbnailAlt: "Geometry before density in score learning"
topicNames: {"생성 모델": "Generative Models", "확률과 기하": "Probability and Geometry"}
---

At small noise levels, a score contains a strong instruction to return to the data manifold and a weaker instruction about how frequently to visit different points on it. *When Scores Learn Geometry* separates these contributions at scales $\sigma^{-2}$ and $O(1)$, then uses Tempered Score (TS) Langevin dynamics to suppress frequency information while retaining geometric confinement. [Theorems 3.1 and 5.1–5.2, §7](https://arxiv.org/html/2509.24912v2#S3)

<strong>Review interpretation.</strong> The separation provides a concrete reason to target uniform geometric exploration; the image experiments offer preliminary evidence for that direction rather than a quantitative test of uniformity on an image manifold.

This first part covers §§1–8; [[2/4] Score Geometry: Laplace and Error Separation](/en/reviews/score-geometry-laplace/), [[3/4] Score Geometry: Stationarity and Mixing](/en/reviews/score-geometry-wkb/), and [[4/4] Score Geometry: Proving a Uniform Limit](/en/reviews/score-geometry-weak-proof/).

## 1. Introduction

A circle in the plane has two ambient coordinates but only one direction of motion along the circle. Learning its shape and learning where observations occur most frequently are different tasks. The manifold hypothesis generalizes this picture: a data measure $\mu_{\mathrm{data}}$ lives on an $n$-dimensional manifold $\mathcal M\subset\mathbb R^d$, with $n<d$.

Adding Gaussian noise produces an ambient density $p_\sigma$. Its score, $s^*(x,\sigma)=\nabla\log p_\sigma(x)$, combines attraction toward the support with a preference for some locations on that support. Conventional distribution learning seeks both. The authors ask whether the first signal can be retained while the second is deliberately removed. Their proposed operation is simple: multiply the estimated score by $\sigma^\alpha$ while leaving the Langevin noise unchanged.

![The paper's Figure 1 contrasts distribution recovery with geometric learning](/assets/score-geometry/paper-figure-1.svg)

Figure 1 places support recovery and accurate frequency recovery at different error scales. Its right panel changes the target to the intrinsic uniform distribution. [Paper Figure 1](https://arxiv.org/html/2509.24912v2#S1.F1)

### 1.1 Related Work

The paper organizes its comparison around four lines of work.

- <strong>Distribution learning.</strong> Existing convergence analyses connect score accuracy to recovery of a target law, including questions about dimension dependence. The authors' distinction is that a joint distributional error bound does not itself separate shape information from density information.
- <strong>Detection of the data manifold.</strong> The paper distinguishes Stanczuk and colleagues' intrinsic-dimension estimation, Ventura and colleagues' analysis of linear subspaces, Pavlova and colleagues' primarily empirical evidence, and Pidstrigach's regularity conditions for concentration. It positions the present work around noise-dependent approximation rates and uniform sampling, rather than support detection alone.
- <strong>Asymptotic behavior of the score.</strong> The authors describe prior results by Lu, Lyu, Liu, and collaborators as identifying the singular projection-like component. Their claimed advance is to isolate the density-dependent higher-order information instead of placing it inside an undifferentiated bounded remainder.
- <strong>Uniform manifold sampling.</strong> Graph normalizations associated with Coifman, Hein, and collaborators cancel sampling density and recover a Laplace–Beltrami operator. The paper contrasts those geometric-operator constructions with large ambient generative models, and contrasts its inference-time modification with De Santi and colleagues' fine-tuning approach to uniform exploration.

These are the comparisons made in the target paper's related-work section. [Paper §1.1](https://arxiv.org/html/2509.24912v2#S1.SS1)

The background behind the proofs is developed where it becomes necessary: Laplace asymptotics in [Part 2](/en/reviews/score-geometry-laplace/), and the Wentzel–Kramers–Brillouin (WKB) stationary-density expansion in [Part 3](/en/reviews/score-geometry-wkb/).

## 2. Preliminaries and Notation

### Manifolds, coordinates, and two densities

<strong>Assumption 2.1.</strong> The support $\mathcal M$ is a compact, boundaryless, $C^4$ embedded submanifold of dimension $n$. Compactness makes finite coordinate covers and uniform geometric bounds available. Absence of a boundary matters when integrating tangential diffusion, while the fourth derivatives enter the refined Laplace estimates used later for scores.

Let $\Phi:U\subset\mathbb R^n\to\mathcal M$ be a chart. The columns of $D\Phi$ span the tangent space; its orthogonal complement is the normal space. The induced Riemannian metric and volume element are

$$
g(u)=D\Phi(u)^\top D\Phi(u),\qquad
d\mathcal M=\sqrt{\det g(u)}\,du.
$$

The factor $\sqrt{\det g}$ converts coordinate volume $du$ into actual volume along the surface. It is essential even for a one-dimensional ellipse.

<strong>Assumption 2.2.</strong> The paper's $p_{\mathrm{data}}(u)$ is a strictly positive $C^1$ density with respect to $du$. Write $\rho$ for density with respect to intrinsic volume:

$$
d\mu_{\mathrm{data}}=p_{\mathrm{data}}(u)\,du
=\rho(\Phi(u))\,d\mathcal M,\qquad
\rho(\Phi(u))=\frac{p_{\mathrm{data}}(u)}{\sqrt{\det g(u)}}.
$$

Uniform intrinsic volume means constant $\rho$, not generally constant $p_{\mathrm{data}}$. For an ellipse $\Phi(u)=(a\cos u,b\sin u)$,

$$
\sqrt{\det g(u)}=\|\Phi'(u)\|
=\sqrt{a^2\sin^2u+b^2\cos^2u},\qquad
p_{\mathrm{unif}}(u)=
\frac{\sqrt{a^2\sin^2u+b^2\cos^2u}}
{\int_0^{2\pi}\sqrt{a^2\sin^2v+b^2\cos^2v}\,dv}.
$$

Equal angular increments therefore carry different arc lengths. Only a circle makes uniform angle and uniform arc length coincide.

<details class="review-background">
<summary><span>Supplementary explanation: the tube and nearest-point projection</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

Inside a sufficiently narrow tubular neighborhood $T_{\mathcal M}(\epsilon)$, each point has a unique nearest point $P_{\mathcal M}(x)$. Choose a local orthonormal normal frame $N(u)\in\mathbb R^{d\times(d-n)}$. Then

$$
x=\Phi(u,r)=\Phi(u)+N(u)r,\qquad
P_{\mathcal M}(x)=\Phi(u),\qquad
d_{\mathcal M}(x)=\frac12\|r\|^2.
$$

Here $r$ is a vector of normal coordinates and $d_{\mathcal M}$ is half the squared distance. The center of a circle illustrates why the tube restriction matters: it has many nearest points, so the projection fails to be unique there.

Since the derivative of the projection maps into a tangent space and $x-P_{\mathcal M}(x)$ is normal,

$$
\nabla d_{\mathcal M}(x)
=(I-DP_{\mathcal M}(x)^\top)(x-P_{\mathcal M}(x))
=x-P_{\mathcal M}(x).
$$

The large restoring component of the score is therefore directed along the shortest displacement back to the manifold. Projection regularity is discussed in [Leobacher–Steinicke, Theorem 2 and Theorem C, Portable Document Format (PDF) pp.15–16](https://arxiv.org/pdf/1811.10578#page=15).

The metric $g=(D_u\Phi)^\top D_u\Phi$ is $n\times n$. The full tube Jacobian $J=D_{(u,r)}\Phi$ instead produces $G=J^\top J\in\mathbb R^{d\times d}$, with $G(u,0)=\operatorname{diag}(g(u),I)$. These matrices play different roles.

</div>
</details>

### 2.1 The Gaussian Smoothed Measure and Connection to Diffusion Models

Variance-exploding (VE) smoothing is $X_\sigma=Y+\sigma Z$; variance-preserving (VP) smoothing is $X_\sigma=\sqrt{1-\sigma^2}Y+\sigma Z$, where $Y\sim\mu_{\mathrm{data}}$ and $Z\sim\mathcal N(0,I_d)$. VP uses $0<\sigma<1$. Set $\gamma(\sigma)=1$ or $\sqrt{1-\sigma^2}$ respectively. Equation (3) becomes

$$
p_\sigma(x)=\frac1{(2\pi\sigma^2)^{d/2}}
\int_U
\exp\left[-\frac{\|x-\gamma(\sigma)\Phi(u)\|^2}{2\sigma^2}\right]
p_{\mathrm{data}}(u)\,du.
$$

Multiple charts are combined with a partition of unity, so overlapping patches are not counted twice. The smoothed measure has an ambient Lebesgue density, whereas the original lower-dimensional measure does not. The appropriate limiting statement is weak convergence:

$$
\mu_\sigma\Rightarrow\mu
\quad\Longleftrightarrow\quad
\int h\,d\mu_\sigma\to\int h\,d\mu
\quad\text{for every bounded continuous function }h.
$$

The ambient density can diverge like $\sigma^{-(d-n)}$ on the manifold while the measures converge weakly. There is no contradiction: the mass occupies a progressively thinner normal tube. [Paper §2.1](https://arxiv.org/html/2509.24912v2#S2.SS1)

These two families are the marginals of forward noising processes in diffusion models. A reverse process follows a schedule $\sigma(t)$ toward zero. Alternatively, one can run Langevin at a fixed noise level, or use Langevin correction between predictor updates. The paper treats both smoothing conventions with the same notation when possible.

### 2.2 Bayesian Inverse Problems

For an observation $y=A(x)+\xi$, write the likelihood as $p(y\mid x)\propto e^{-v(x;y)}$. Gaussian observation noise of standard deviation $s$ gives $v(x;y)=\|A(x)-y\|^2/(2s^2)$. A prior measure $\mu_{\mathrm{data}}$ leads to a posterior proportional to $e^{-v}d\mu_{\mathrm{data}}$.

Plug-and-Play (PnP) sampling can use a prior score even when the prior's normalizing constant is unknown or the prior is accessible only through samples. With an estimated prior score $\hat s$, the paper's Equation (4) is

$$
dX_t=-\nabla_x v(X_t;y)dt+\hat s(X_t)dt+\sqrt2\,dW_t.
$$

The likelihood contributes $-\nabla v$ and the prior contributes $\hat s$. Section 6 changes the prior to intrinsic uniform volume; it consequently changes the posterior being sought. [Paper §2.2](https://arxiv.org/html/2509.24912v2#S2.SS2)

### 2.3 Stationary Distribution for Non-reversible Dynamics

The stochastic differential equation (SDE)

$$
dX_t=b(X_t)\,dt+\sqrt2\,dW_t
$$

has the Fokker–Planck equation $\partial_t\pi=-\nabla\cdot(b\pi)+\Delta\pi$. Stationarity means that its right-hand side vanishes.

<strong>Supplementary derivation.</strong> Applying Itô's formula to a test function $h$ gives generator $Lh=b\cdot\nabla h+\Delta h$. Brownian increments are of size $\sqrt{dt}$, so their quadratic variation is $dX_i\,dX_j=2\delta_{ij}dt$. The second-order Taylor term contributes $\Delta h\,dt$, explaining the diffusion coefficient and the absence of an extra factor $1/2$. Integration by parts transfers derivatives to the density and gives the Fokker–Planck equation.

If $b=-\nabla f$, the normalizable Gibbs density $e^{-f}$ has zero probability current $j=b\pi-\nabla\pi$. A non-gradient field may instead have nonzero current with $\nabla\cdot j=0$. A stationary law can exist without a scalar-potential formula. This is the reason for the WKB analysis later. [Paper §2.3](https://arxiv.org/html/2509.24912v2#S2.SS3)

## 3. Central Insight: Gaussian Smoothing Recovers Geometry Before Distribution

### Theorem 3.1

Under Assumptions 2.1–2.2, let $u_x=\Phi^{-1}(P_{\mathcal M}(x))$ and $q=d-n$. In a fixed sufficiently narrow tube,

$$
\log p_\sigma(x)=
-\frac{d_{\mathcal M}(x)}{\sigma^2}
+\log p_{\mathrm{data}}(u_x)
-\frac q2\log(2\pi\sigma^2)
+H(x)+o(1).
$$

The remainder is uniform in $x$. The detailed statement, Theorem B.2, uses the tangent-coordinate Hessian

$$
\widehat H_{ij}(u,x)
=\left\langle\partial_{ij}\Phi(u),\Phi(u)-x\right\rangle
+\left\langle\partial_i\Phi(u),\partial_j\Phi(u)\right\rangle
$$

For VE, $H(x)=-\tfrac12\log\det\widehat H(u_x,x)$. VP adds $-\tfrac12\langle P_{\mathcal M}(x),x-P_{\mathcal M}(x)\rangle$. [Theorem 3.1](https://arxiv.org/html/2509.24912v2#S3.Thmtheorem1), [Theorem B.2](https://arxiv.org/html/2509.24912v2#A2.Thmtheorem2)

Multiplication by $\sigma^2$ removes the bounded corrections and also the logarithmic normalization, since $\sigma^2\log\sigma\to0$:

$$
\lim_{\sigma\to0}\sigma^2\log p_\sigma(x)=-d_{\mathcal M}(x)
\quad\text{uniformly in the tube}.
$$

The leading rescaled log-density is a distance function independent of the original frequency along the manifold. [Paper §3, Equation (7)](https://arxiv.org/html/2509.24912v2#S3)

### Reading the scales correctly

The distance penalty has coefficient $\sigma^{-2}$, while the spatially varying density information is finite. The term $-\tfrac q2\log(2\pi\sigma^2)$ also diverges, but is constant in $x$ and disappears on taking a score. This distinction makes it possible to erase frequency without erasing confinement.

<details class="review-background">
<summary><span>Supplementary explanation: how large is the score itself?</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

The stronger score estimate is proved separately in Appendix B.5:

$$
s^*(x,\sigma)
=-\frac{x-P_{\mathcal M}(x)}{\sigma^2}+O(1).
$$

At a fixed positive normal distance $\delta$, the leading magnitude is $\delta/\sigma^2$. At a typical noise displacement of size $O(\sigma)$, it is $O(\sigma^{-1})$. Exactly on the manifold, that leading vector vanishes. The $\sigma^{-2}$ separation concerns the coefficient and the fixed-tube asymptotics, not a universal magnitude at every sample location.

The log-density expansion is a singular small-noise asymptotic obtained by Laplace's method. Calling it a Taylor expansion does not make it an ordinary smooth Taylor series at $\sigma=0$, nor does a uniform $o(1)$ remainder automatically have a small derivative. [The proof is in Part 3](/en/reviews/score-geometry-wkb/).

</div>
</details>

## 4. Scale Separation in Existing Generative Learning: Geometry versus Distribution

### The object whose error is controlled

Here the paper compares the score of the <strong>final generated density</strong> $\pi_\sigma$ with the exact smoothed score:

$$
s_{\pi_\sigma}=\nabla\log\pi_\sigma,\qquad
E_\sigma=
\|s_{\pi_\sigma}-s^*(\cdot,\sigma)\|_{L^\infty(K)}.
$$

Assumption 4.1 requires $-\log\pi_\sigma\in C^1(K)$, concentration on a compact $K$ containing the tube, and a uniform bound $L_K$ on lengths of paths connecting any two points of $K$. The norm is worst-case error on $K$. It is stronger than an average squared training error. The theorem does not propagate each network error through an entire reverse SDE to obtain this final-density score error.

### The three conclusions of Theorem 4.1

1. If $E_\sigma=o(\sigma^{-2})$, then for every fixed $\delta>0$, the mass farther than $\delta$ from $\mathcal M$ tends to zero.
2. For an arbitrary $C^1$ target density on $\mathcal M$, the paper constructs generated laws converging to that target with $E_\sigma=\Omega(1)$.
3. If $E_\sigma=o(1)$, then the generated laws converge weakly to the original data measure.

These are concentration, possible density distortion, and a sufficient condition for distribution recovery respectively. The $\Omega(1)$ statement is a lower bound; it does not itself assert an upper bound by a constant. Concentration alone also permits almost all mass to collect on a small part of the support. [Theorem 4.1](https://arxiv.org/html/2509.24912v2#S4.Thmtheorem1)

The proof works with $\Delta_\sigma=\log\pi_\sigma-\log p_\sigma$. Path integration gives

$$
|\Delta_\sigma(x)-\Delta_\sigma(y)|\le L_KE_\sigma.
$$

This controls oscillation of a log-density ratio. Normalization and concentration on $K$ then fix its otherwise unknown additive constant. [Part 2](/en/reviews/score-geometry-laplace/) provides the complete argument and a smooth ambient counterexample showing why the $o(1)$ sufficient condition is not a necessary condition for every successful sequence.

## 5. New Paradigm of Geometric Learning: Recover Uniform Distributions with $o(\sigma^{-2})$ Score Error

### The modification and its target

The paper's Equation (8) is

$$
dX_t=\sigma^\alpha s(X_t,\sigma)\,dt+\sqrt2\,dW_t.
$$

Only the drift is weakened. A time change would scale both drift and noise and preserve the stationary law. Here, if $s=-\nabla f_\sigma$, stationarity instead gives

$$
\widetilde\pi_\sigma(x)\propto e^{-\sigma^\alpha f_\sigma(x)}.
$$

For an exact score, this is the normalized power $p_\sigma^{\sigma^\alpha}$. Substitution of the density expansion yields

$$
\log\widetilde\pi_\sigma(x)
=-\frac{d_{\mathcal M}(x)}{\sigma^{2-\alpha}}
+\sigma^\alpha\log\rho(P_{\mathcal M}(x))
+\text{vanishing geometric corrections}+\text{normalizing constant}.
$$

For $0<\alpha<2$, the frequency term disappears while the normal penalty diverges. [Paper §5](https://arxiv.org/html/2509.24912v2#S5)

### Theorem 5.1: a gradient score oracle

Assume 2.1–2.2 and 4.1, now applied to $\widetilde\pi_\sigma$, and suppose that the estimated score is a gradient. For some <strong>fixed</strong> $\beta>-2$, require

$$
\|s(\cdot,\sigma)-s^*(\cdot,\sigma)\|_{L^\infty(K)}
=o(\sigma^\beta)
$$

Then any exponent satisfying

$$
\max\{-\beta,0\}<\alpha<2
$$

gives weak convergence of the stationary laws to normalized Riemannian volume. In coordinates, the limit has density proportional to $\sqrt{\det g(u)}$.

The conditions have separate jobs: $\alpha<2$ retains confinement, $\alpha+\beta>0$ removes approximation error, and $\alpha>0$ erases the data-density term. For example, $E_\sigma=O(1)$ allows $\beta=-1/2$ and $\alpha=1$. With a specified bound $E_\sigma=o(\sigma^{-3/2})$, the theorem's exponent interval is $3/2<\alpha<2$. [Theorem 5.1](https://arxiv.org/html/2509.24912v2#S5.Thmtheorem1)

The effective normal variance is $\theta=\sigma^{2-\alpha}$ and its standard deviation is $\sigma^{1-\alpha/2}$. At $\alpha=1$, the width is $\sqrt{\sigma}$, wider than the original smoothing width $\sigma$.

![An explanatory annular model separates normal width from angular preference](/assets/score-geometry/tempering-toy.svg)

<strong>Explanatory model.</strong> The separable density $q_\sigma(u,r)\propto\exp[-r^2/(2\sigma^2)+4\cos u]$ gives the width on the left and angular density proportional to $e^{4\sigma^\alpha\cos u}$ at $\sigma=0.01$ on the right. This illustrates the two scales; it is not the paper's learned-model experiment.

The fixed-power hypothesis is more specific than merely $o(\sigma^{-2})$. A logarithmic boundary example and a variable-tempering extension are developed once, in [Part 3, Appendix B.5](/en/reviews/score-geometry-wkb/).

### Theorem 5.2: a general score oracle

For a non-gradient oracle, the paper adds $p_{\mathrm{data}}\in C^2$, the same score-error and exponent conditions, a unique stationary law, and local WKB Assumption B.2 with $\theta=\sigma^{2-\alpha}$. Under these assumptions it obtains the same intrinsic uniform limit.

The WKB assumption includes concentration in the tube, a positive $C^2$ amplitude converging in $C^2$, and selection of an action solution. These are substantive assumptions about the stationary law. The coefficient argument is detailed in [Part 3](/en/reviews/score-geometry-wkb/). [Theorem 5.2](https://arxiv.org/html/2509.24912v2#S5.Thmtheorem2), [Assumption B.2](https://arxiv.org/html/2509.24912v2#A2.Thmassumption2)

The order of limits matters: first consider a stationary law for each fixed $\sigma$, then send $\sigma$ to zero. Finite corrector budgets along an annealing schedule involve an additional approximation problem.

## 6. Uniform Prior is More Robust For Bayesian Inverse Problems

### A different prior means a different posterior

A data-density prior gives posterior $e^{-v}\rho\,d\mathcal M$. An intrinsic uniform prior gives $e^{-v}d\mathcal M$. The tempered construction targets the second law.

<strong>Supplementary derivation.</strong> Let $V_{\mathcal M}$ be the manifold's volume. Kullback–Leibler (KL) divergence is $D_{\mathrm{KL}}(p\|q)=\int p\log(p/q)$ for densities relative to the same reference measure. Writing $t=p/q$, convexity of $t\log t$ and $\int tq=1$ give nonnegativity, with equality exactly when the laws coincide. Consequently,

$$
-\int_{\mathcal M}\rho\log\rho\,d\mathcal M
=\log V_{\mathcal M}
-D_{\mathrm{KL}}\!\left(\rho\,d\mathcal M\;\middle\|\;
\frac{d\mathcal M}{V_{\mathcal M}}\right)
\le\log V_{\mathcal M}.
$$

Uniform volume maximizes differential entropy relative to that volume reference. The statement depends on the chosen reference measure and does not imply uniform coordinate density.

### Theorem 6.1

Under the stationary-law and WKB conditions needed for the guided dynamics, with $v$ bounded on $\mathbb R^d$ and $C^1$ in the tube, the paper writes the gradient case as

$$
dX_t=-\nabla v(X_t)\,dt-\sigma^\alpha\nabla f_\sigma(X_t)\,dt+\sqrt2\,dW_t
$$

The general prior-score term is $\sigma^\alpha s$. Its limiting coordinate density is

$$
\pi_{\mathrm{post}}(u)
=\frac{e^{-v(\Phi(u))}\sqrt{\det g(u)}}
{\int_U e^{-v(\Phi(w))}\sqrt{\det g(w)}\,dw}.
$$

For a gradient score, the stationary exponent is $-v-\sigma^\alpha f_\sigma$. After the time rescaling $\theta=\sigma^{2-\alpha}$, normal confinement survives while $v$ remains as the finite tangential objective. The general argument uses $f_1=v$ in the WKB calculation. [Theorem 6.1](https://arxiv.org/html/2509.24912v2#S6.Thmtheorem1)

The familiar quadratic Gaussian observation loss is generally unbounded on the entire ambient space. Its bounded restriction to a compact manifold does not establish the theorem's global assumption; a tail argument or an external cutoff is needed for that example.

### Classifier-free guidance

Classifier-free guidance (CFG) decomposes the score into an unconditional term and a conditional increment:

$$
s_{\mathrm{CFG}}=s_{\mathrm{uncond}}
+w(s_{\mathrm{cond}}-s_{\mathrm{uncond}}).
$$

The modified corrector scales only the unconditional part:

$$
\widetilde s_{\mathrm{CFG}}
=\sigma^\alpha s_{\mathrm{uncond}}
+w(s_{\mathrm{cond}}-s_{\mathrm{uncond}}).
$$

The increment representing the likelihood must remain at finite scale. Scaling the entire CFG field would also erase that increment. Whether a practical noise-dependent CFG field meets the theorem's fixed-$v$ assumptions is a further question; the Stable Diffusion experiment is a proof of concept for the connection.

## 7. Experiments

### 7.1 Numerical Simulations on Ellipse

The authors sample angular data from a von Mises law on a circle or ellipse, learn a score, and compare standard Langevin with TS at $\alpha=1$. Figure 2 shows that the standard sampler need not reproduce the original frequencies, while the tempered sampler approaches uniform intrinsic volume. On the ellipse, that target has a nonconstant angular density.

The von Mises density is proportional to $e^{\kappa\cos(u-\mu)}$. Here $\mu$ is the preferred angle and $\kappa\ge0$ controls concentration. Its normalizer is $2\pi I_0(\kappa)$, where $I_0(\kappa)=(2\pi)^{-1}\int_0^{2\pi}e^{\kappa\cos u}du$ is the modified Bessel function of order zero. At $\kappa=0$ the angle is uniform.

Appendix C.3 additionally corrupts an exact score by an error of order $O(1)$ in $\sigma$. Figure 4 contrasts distorted distribution recovery with successful uniform targeting in that controlled example; its setup and figure appear in Part 3.

![Paper Figure 2: standard and tempered sampling on an ellipse and a circle](/assets/score-geometry/paper-figure-2.svg)

Figure 2 orders its panels as standard Langevin and TS on the ellipse, then the same pair on the circle. The horizontal coordinate is angle. The uniform-volume reference is flat only for the circle. [Paper Figure 2](https://arxiv.org/html/2509.24912v2#S7.F2)

### 7.2 Image Generation with Diffusion Models

The prompts concern furniture, cars, and architecture. Each setting generates 512 images, resized to $256\times256$ for Contrastive Language–Image Pre-training (CLIP) features. Guidance is 7.5 and the predictor has 30 steps. Predictor–corrector (PC) and TS use matching numbers of final deterministic projection steps.

- <strong>Prompt similarity (P-sim)</strong> averages image–prompt feature cosine similarity; higher means better prompt alignment.
- <strong>Inter-image similarity (I-sim)</strong> averages pairwise image feature similarity within a prompt; lower indicates greater diversity in that feature space.

Neither metric directly measures uniform Riemannian volume, and prompt alignment covers only one aspect of image quality. The numerical scale below is the paper's. The third baseline uses Denoising Diffusion Probabilistic Models (DDPM).

| Paper Table 1 | Furniture P-sim ↑ | Furniture I-sim ↓ | Car P-sim ↑ | Car I-sim ↓ | Architecture P-sim ↑ | Architecture I-sim ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPM | 29.56 | 80.78 | 26.23 | 87.30 | 27.36 | 81.53 |
| PC | 29.40 | 81.24 | 26.30 | 87.20 | 27.13 | 81.03 |
| TS | 30.20 | 80.76 | 26.62 | 87.14 | 27.32 | 80.76 |

Table 1 reports best metrics after tuning corrector counts and, for TS, $\alpha$. It does not identify a common hyperparameter setting for each pair of best metric values. For furniture PC in Table 2, for example, maximum P-sim occurs with 5 correctors whereas minimum I-sim occurs with 10. A row of separately optimized metrics should not be treated as the performance of one joint operating point.

At fixed $\alpha=1$, Table 2 improves P-sim in all 15 matched prompt/count comparisons and improves I-sim in 14. For cars with 20 correctors, I-sim changes from 88.06 to 88.07. Tables 3–4 also include deteriorations, and increasing $\alpha$ does not improve both metrics monotonically. The full numbers are in Part 3. Repeated-seed uncertainty and significance tests are not reported. [Paper §7 and Appendix C](https://arxiv.org/html/2509.24912v2#S7)

![Paper Figure 3: PC samples above and TS samples below under matched conditions](/assets/score-geometry/paper-figure-3.webp)

Each column matches prompt, corrector count, and random seed. The authors describe the bottom row as more authentic and detailed. These qualitative examples complement the feature metrics. [Paper Figure 3](https://arxiv.org/html/2509.24912v2#S7.F3)

## 8. Conclusion

The paper separates a large geometric penalty from smaller frequency-dependent terms and uses the difference to design another target law. Laplace asymptotics explain both normal concentration and the intrinsic-volume correction. In inverse problems, likelihood information can be retained while the original data-density prior is removed.

The authors identify several limitations: error is not tracked along the full generation trajectory; the worst-case score criterion needs a closer connection to average squared training objectives; score-estimation sample complexity is not derived; uniformity error is not quantified; and the image experiments do not cover current large state-of-the-art models. [Paper §8](https://arxiv.org/html/2509.24912v2#S8)

## Review interpretation and open questions

<strong>Review interpretation.</strong> With strictly positive data coverage of the manifold, retaining normal attraction while suppressing finite tangential bias is a coherent strategy. The synthetic examples directly separate shape from frequency, and the image results suggest that this change of target can improve prompt alignment and feature diversity together.

The remaining bridge is from stationary, continuous-time, small-noise laws to trained networks with finite sampling budgets. Another question is how image-feature diversity relates to intrinsic geometric uniformity. These issues are distinct from the decision to replace a data-density prior with a uniform-volume prior. Parts 2 and 3 examine the exact proof conditions; Part 4 develops a separate model where weaker stationary equations give a direct proof.

## Sources and reading scope

- Target paper: Xiang Li, Zebang Shen, Ya-Ping Hsieh, Niao He, *When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis*. First released 2025-09-29; reviewed version v2, 2026-03-16; reviewed 2026-10-29. [Official record](https://arxiv.org/abs/2509.24912v2).
- Scope: §§1–8, Figures 1–3, Table 1; the complete additional experimental settings and tables are in [Part 3](/en/reviews/score-geometry-wkb/).
- Figures: the cover and separable-model plot are original explanatory vector graphics. Paper Figures 1–3 are limited figure-area quotations for analysis. Vector elements are retained; Figure 1's embedded rasters keep their pixel dimensions and are encoded as WebP at quality 95. Figure 3 is rendered to a 1,500-pixel long edge and encoded as WebP at quality 92. The source carries the arXiv perpetual non-exclusive distribution license.
- Not independently reproduced: neural training, Stable Diffusion image generation, or formal verification of the theorems.
