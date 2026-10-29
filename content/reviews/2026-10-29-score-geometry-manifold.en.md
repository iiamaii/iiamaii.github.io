---
title: "[1/4] Score Geometry: Shape and Density"
description: "Why small-noise scores encode shape more strongly than frequency, and what the five main theorems and experiments establish."
date: "2026-10-29"
publishedAt: "2026-10-29T08:55:43+09:00"
updatedAt: "2026-10-29T16:07:03+09:00"
topics: ["생성 모델", "확률과 기하"]
visibility: "public"
lang: "en"
translationKey: "score-geometry-manifold"
paperTitle: "When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis"
paperPublishedDate: "2025-09-29"
authors: ["Xiang Li", "Zebang Shen", "Ya-Ping Hsieh", "Niao He"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2509.24912v2"
thumbnail: "/assets/score-geometry/paper-figure-1.svg"
thumbnailAlt: "Paper Figure 1: distribution recovery, manifold geometry learning, and uniform exploration"
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

<strong>Assumption 2.1.</strong> The support $\mathcal M$ is a compact, boundaryless, $C^4$ embedded submanifold of dimension $n$. Compactness makes finite coordinate covers and uniform geometric bounds available. An embedded submanifold sits smoothly in the ambient space. Absence of a boundary matters when integrating tangential diffusion. The $C^4$ regularity condition means that derivatives through order four are continuous; those derivatives enter the refined Laplace estimates used later for scores.

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

A tubular neighborhood is a thin tube around the manifold. We choose it narrow enough that every point has a unique nearest point on the manifold; nearest-point projection sends a point to that nearest point.

The central distance function is the **half squared distance** in paper Equation (2). Where the nearest point is unique, its gradient is explicit:

$$
\begin{aligned}
d_{\mathcal M}(x)
&=\frac12\operatorname{dist}(x,\mathcal M)^2
=\min_{y\in\mathcal M}\frac12\|x-y\|^2,\\
\nabla d_{\mathcal M}(x)&=x-P_{\mathcal M}(x).
\end{aligned}
$$

[Paper Equation (2)](https://arxiv.org/html/2509.24912v2#S2.E2). The coordinate derivation follows below.

<details class="review-background">
<summary><span>Supplementary explanation: the tube and nearest-point projection</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

Inside a sufficiently narrow tubular neighborhood $T_{\mathcal M}(\epsilon)$, each point has a unique nearest point $P_{\mathcal M}(x)$. Choose a local orthonormal normal frame $N(u)\in\mathbb R^{d\times(d-n)}$: its normal vectors are mutually perpendicular and have unit length. Then

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

Variance-exploding (VE) and variance-preserving (VP) smoothing construct the following random variables, with $Y\sim\mu_{\mathrm{data}}$ and $Z\sim\mathcal N(0,I_d)$. VP requires $0<\sigma<1$.

$$
X_\sigma^{\mathrm{VE}}=Y+\sigma Z,
\qquad
X_\sigma^{\mathrm{VP}}=\sqrt{1-\sigma^2}\,Y+\sigma Z.
$$

These are the two laws in paper Equation (1). Set $\gamma(\sigma)=1$ or $\sqrt{1-\sigma^2}$ to write their ambient density as Equation (3):

$$
p_\sigma(x)=\frac1{(2\pi\sigma^2)^{d/2}}
\int_U
\exp\left[-\frac{\|x-\gamma(\sigma)\Phi(u)\|^2}{2\sigma^2}\right]
p_{\mathrm{data}}(u)\,du.
$$

Multiple charts are combined with a partition of unity: weights on overlapping patches sum to one, so those patches are not counted twice. The smoothed measure has an ambient Lebesgue density, whereas the original lower-dimensional measure does not. The appropriate limiting statement is weak convergence, tested against bounded continuous functions. Bounded means that the absolute value stays below some finite constant:

$$
\mu_\sigma\Rightarrow\mu
\quad\Longleftrightarrow\quad
\int h\,d\mu_\sigma\to\int h\,d\mu
\quad\text{for every bounded continuous function }h.
$$

Fix a point $y$ on the manifold. Saying that its ambient density diverges at scale $\sigma^{-(d-n)}$ means that **the density height grows approximately as a positive constant times $\sigma^{-(d-n)}$ when the noise width tends to zero**. The codimension $q=d-n$ counts the normal directions. For variance-exploding smoothing, the positive intrinsic-volume density $\rho(y)$ gives the precise limit

$$
\lim_{\sigma\to0}\sigma^q p_\sigma(y)
=\frac{\rho(y)}{(2\pi)^{q/2}}>0,
\qquad q=d-n,\quad y\in\mathcal M.
$$

At sufficiently small noise, halving $\sigma$ therefore multiplies the height at the same point by approximately $2^q$. For a curve in the plane, $d=2$ and $n=1$, so the factor is approximately two. **A density value is not itself a probability:** it measures probability per unit ambient volume and can exceed one. Normal widths shrink with $\sigma$, so the corresponding $q$-dimensional volume scales as $\sigma^q$. This shrinking volume offsets the increasing height, leaving total probability equal to one. Pointwise density growth is therefore compatible with weak convergence of measures. [Density definition in §2.1](https://arxiv.org/html/2509.24912v2#S2.SS1); [Theorem B.2 evaluated on the manifold](https://arxiv.org/html/2509.24912v2#A2.Thmtheorem2).

<details class="review-background">
<summary><span>Supplementary explanation: density height versus probability</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

As an exact explanatory example, add Gaussian noise of standard deviation $\sigma$ to a point located at zero. Its density is

$$
\varphi_\sigma(x)=\frac{1}{\sqrt{2\pi}\sigma}
\exp\!\left(-\frac{x^2}{2\sigma^2}\right),
\qquad \int_{\mathbb R}\varphi_\sigma(x)\,dx=1.
$$

The peak $\varphi_\sigma(0)=1/(\sqrt{2\pi}\sigma)$ is about 1.995 at $\sigma=0.2$ and 3.989 at $\sigma=0.1$. Doubling height while halving width preserves the total area of one. An interval's probability is the **area under its part of the curve**, rather than the height at one point.

As $\sigma\to0$, these Gaussian probability measures converge weakly to the point mass $\delta_0$, which places all probability at zero, even though their density heights at zero grow without bound. This is an exact example of divergent densities accompanying weak convergence of measures. For a general manifold, the thinning occurs in normal directions. The displayed density is the $d=1$, $n=0$ explanatory case of Equation (3), not a learned-model experiment. [Paper Equation (3)](https://arxiv.org/html/2509.24912v2#S2.SS1).

</div>
</details>

These two families are the marginals of forward noising processes in diffusion models. A reverse process follows a schedule $\sigma(t)$ toward zero. Alternatively, one can run Langevin at a fixed noise level, or use Langevin correction between predictor updates. The paper treats both smoothing conventions with the same notation when possible.

### 2.2 Bayesian Inverse Problems

A Bayesian inverse problem combines an observation $y=A(x)+\xi$ with information available before the observation to infer a distribution over the unknown signal. Here $x\in\mathbb R^d$, $y\in\mathbb R^m$, $A:\mathbb R^d\to\mathbb R^m$ is the measurement map, and the observation noise $\xi$ is independent of $x$. The prior distribution describes information before seeing the data; the likelihood $p(y\mid x)$ evaluates how a proposed signal explains the data; the posterior distribution combines both. A likelihood is a density in $y$ for fixed $x$, so it need not integrate to one over candidate signals $x$. [Paper §2.2](https://arxiv.org/html/2509.24912v2#S2.SS2)

Fix $y$ and write $p(y\mid x)\propto e^{-v(x;y)}$. The potential $v$ is the negative log-likelihood after removing terms independent of $x$. For a general prior measure $\mu_{\mathrm{prior}}$, Bayes' formula reads

$$
\begin{aligned}
Z(y)&=\int e^{-v(z;y)}\,d\mu_{\mathrm{prior}}(z),\qquad 0<Z(y)<\infty,\\
d\mu_y(x)&=\frac{e^{-v(x;y)}}{Z(y)}\,d\mu_{\mathrm{prior}}(x).
\end{aligned}
$$

The normalizing constant $Z(y)$ makes total probability one. If the prior has a positive smooth ambient density $p_{\mathrm{prior}}$, the score of the posterior density $\pi_y$ separates into

$$
\nabla_x\log\pi_y(x)
=-\nabla_x v(x;y)+\nabla_x\log p_{\mathrm{prior}}(x).
$$

The following calculation develops this decomposition from a Gaussian observation model. The observation-noise standard deviation $s$ and the data-smoothing width $\sigma$ from the preceding section are different parameters.

<details class="review-background">
<summary><span>Supplementary explanation: from Gaussian likelihood to posterior score</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

<strong>Supplementary derivation.</strong> Hold $y$ and $s>0$ fixed throughout the derivatives with respect to $x$.

**1. Turn the noise density into a likelihood.** With $\xi\sim\mathcal N(0,s^2I_m)$, a candidate $x$ leaves residual noise $y-A(x)$. Thus

$$
\begin{aligned}
p(y\mid x)
&=(2\pi s^2)^{-m/2}
\exp\!\left[-\frac{\|y-A(x)\|^2}{2s^2}\right],\\
-\log p(y\mid x)
&=\frac m2\log(2\pi s^2)+\frac{\|A(x)-y\|^2}{2s^2}.
\end{aligned}
$$

The first term does not depend on $x$ and cancels when the posterior is normalized. This leaves $v(x;y)=\|A(x)-y\|^2/(2s^2)$. Smaller $s$ assigns a larger penalty to the same observation error.

**2. Convert the joint density into a conditional density.** When the prior has a density, $p(x,y)=p(y\mid x)p_{\mathrm{prior}}(x)$. Divide by its integral over $x$, namely $p(y)$:

$$
\begin{aligned}
\pi_y(x)
&=\frac{p(y\mid x)p_{\mathrm{prior}}(x)}
{\int_{\mathbb R^d}p(y\mid z)p_{\mathrm{prior}}(z)\,dz}\\
&=\frac{e^{-v(x;y)}p_{\mathrm{prior}}(x)}
{\int_{\mathbb R^d}e^{-v(z;y)}p_{\mathrm{prior}}(z)\,dz}
=\frac{e^{-v(x;y)}p_{\mathrm{prior}}(x)}{Z(y)}.
\end{aligned}
$$

The Gaussian prefactor is the same in numerator and denominator. Because $Z(y)$ is defined after removing that likelihood prefactor, it is generally different from the marginal observation density $p(y)$. The measure formulation in the main text also covers a prior supported entirely on the manifold.

**3. Take a logarithm, then differentiate.** For a positive differentiable prior density,

$$
\begin{aligned}
\log\pi_y(x)&=-v(x;y)+\log p_{\mathrm{prior}}(x)-\log Z(y),\\
\nabla_x\log Z(y)&=0,\\
\nabla_x\log\pi_y(x)&=-\nabla_x v(x;y)+\nabla_x\log p_{\mathrm{prior}}(x).
\end{aligned}
$$

The integration variable in $Z(y)$ is $z$; $y$ remains fixed, so differentiation with respect to $x$ gives zero. An unknown prior normalizer cancels too: if $p_{\mathrm{prior}}=\widetilde p/C$, then $\nabla\log p_{\mathrm{prior}}=\nabla\log\widetilde p$.

**4. Differentiate the observation loss explicitly.** For differentiable $A$, let $J_A(x)\in\mathbb R^{m\times d}$ be its Jacobian. The componentwise chain rule gives

$$
\begin{aligned}
\partial_{x_j}v(x;y)
&=\frac{1}{s^2}\sum_{k=1}^m
\bigl(A_k(x)-y_k\bigr)\,\partial_{x_j}A_k(x),\\
\nabla_x v(x;y)&=\frac{1}{s^2}J_A(x)^\top\bigl(A(x)-y\bigr).
\end{aligned}
$$

For a linear measurement $A(x)=Mx$, the Jacobian is $M$, and the guidance term becomes $-\nabla_xv=M^\top(y-Mx)/s^2$. It maps the prediction residual back to signal space; the minus sign selects the direction of decreasing observation loss.

These are intermediate calculations for [paper §2.2 and Equation (4)](https://arxiv.org/html/2509.24912v2#S2.SS2). For the density and measure formulations of Bayes' formula, see [Kekkonen's lecture notes, §1.3, Theorem 1.9, printed pp.7–9](https://www.damtp.cam.ac.uk/research/cia/files/teaching/Inverse_Problems_Lent_2018/19y_01m_16d_LectureNotes.pdf#page=8).

</div>
</details>

Plug-and-Play (PnP) methods can insert a learned prior score $\hat s$ when the prior normalizer is unknown or only prior samples are available. Using the posterior score as the Langevin drift and replacing its prior term with $\hat s$ produces the paper's Equation (4):

$$
dX_t=-\nabla_x v(X_t;y)dt+\hat s(X_t)dt+\sqrt2\,dW_t.
$$

At fixed $\sigma>0$, if $p_{\mathrm{prior}}=p_\sigma$ and $\hat s=\nabla\log p_\sigma$ is exact, the target stationary density is $\pi_y\propto e^{-v}p_\sigma$, subject to the required normalization and regularity conditions. The next section verifies this by substituting into the probability current. An approximate learned score does not automatically preserve this identity.

For the original data prior $d\mu_{\mathrm{data}}=\rho\,d\mathcal M$, Bayes' formula instead defines a target proportional to $e^{-v}\rho\,d\mathcal M$. [Section 6 and its derivation below](#theorem-61) replace that prior with uniform manifold volume and target $e^{-v}\,d\mathcal M$. See [Theorem 6.1](https://arxiv.org/html/2509.24912v2#S6.Thmtheorem1), [the proof connection in Appendix B.6](https://arxiv.org/html/2509.24912v2#A2.SS6), and [Part 3's WKB proof discussion](/en/reviews/score-geometry-wkb/#b6-proof-for-theorem-61).

### 2.3 Stationary Distribution for Non-reversible Dynamics

In the stochastic differential equation (SDE)

$$
dX_t=b(X_t)\,dt+\sqrt2\,dW_t
$$

$b$ is the drift and $W_t$ is standard $d$-dimensional Brownian motion. When its time-dependent density $\pi_t$ is sufficiently smooth, it satisfies the Fokker–Planck equation

$$
\partial_t\pi_t=-\nabla\cdot(b\pi_t)+\Delta\pi_t
$$

Here $\nabla\cdot$ is the divergence of a vector field and $\Delta=\sum_{i=1}^d\partial_{x_i}^2$ is the Laplacian. Vector-field divergence is a differential operator measuring local outward flow; it is distinct from a density value diverging to infinity, discussed earlier. A stationary distribution has a time-independent density and hence a zero right-hand side. [Paper §2.3](https://arxiv.org/html/2509.24912v2#S2.SS3)

<details class="review-background" id="ito-fokker-planck">
<summary><span>Supplementary explanation: from Itô's lemma to the Fokker–Planck equation</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

<strong>Supplementary derivation.</strong> Start with the motion of a path $X_t$ and ask how its density $\pi_t$ evolves. The bridge is to **apply Itô's lemma to an observable and then take expectations**. Assume that $b$ is locally smooth and that the process does not explode in finite time. We first compute with a sufficiently smooth density, then retain an integral formulation that does not require classical density derivatives.

**1. Choose an observable of the distribution.** Fix a test function $h\in C_c^\infty(\mathbb R^d)$: a smooth function that vanishes outside a compact set. A nonnegative bump supported near a region of interest provides a smoothly weighted observation of the probability there. The same observation has a random-variable representation and a density representation:

$$
\mathbb E[h(X_t)]=\int_{\mathbb R^d}h(x)\pi_t(x)\,dx.
$$

Computing the time variation on the left using Itô's lemma will therefore reveal the density evolution on the right. Since $h=h(x)$ has no explicit time dependence, there is no $\partial_t h$ term. Write $\partial_i=\partial/\partial x_i$ below.

**2. Track the diffusion coefficient in Itô's lemma.** Brownian quadratic variation obeys $dW_i\,dW_j=\delta_{ij}\,dt$. The Kronecker delta $\delta_{ij}$ is one when $i=j$ and zero otherwise. Because the noise coefficient is $\sqrt2$,

$$
\begin{aligned}
dX_i\,dX_j&=2\delta_{ij}\,dt,\\
\frac12\sum_{i,j=1}^d\partial_{ij}h(X_t)\,dX_i\,dX_j
&=\frac12\sum_{i,j=1}^d\partial_{ij}h(X_t)\,2\delta_{ij}\,dt\\
&=\sum_{i=1}^d\partial_{ii}h(X_t)\,dt
=\Delta h(X_t)\,dt.
\end{aligned}
$$

These are quadratic-variation rules rather than ordinary products of infinitesimals. Over a short interval of length $\delta$, Brownian increments have size $\sqrt\delta$, so their squares accumulate at order $\delta$. Products $dt\,dt$ and $dt\,dW_i$ make no contribution to quadratic variation. The displayed calculation tracks the cancellation between Itô's factor $1/2$ and the noise covariance factor two.

$$
\begin{aligned}
dh(X_t)
&=\sum_i\partial_i h(X_t)\,dX_i
+\frac12\sum_{i,j}\partial_{ij}h(X_t)\,dX_i\,dX_j\\
&=\bigl(b\cdot\nabla h+\Delta h\bigr)(X_t)\,dt
+\sqrt2\,\nabla h(X_t)\cdot dW_t.
\end{aligned}
$$

The differential operator $Lh=b\cdot\nabla h+\Delta h$ is the generator. At a current position $x$, $Lh(x)$ gives the instantaneous mean rate of change of the observable $h(X_t)$.

**3. Integrate in time before taking expectations.** Instead of directly differentiating an expected stochastic differential, integrate the identity from $0$ to $t$:

$$
\begin{aligned}
h(X_t)-h(X_0)
&=\int_0^t(Lh)(X_s)\,ds+M_t,\\
M_t&=\sqrt2\int_0^t\nabla h(X_s)\cdot dW_s.
\end{aligned}
$$

The zero mean of $M_t$ has a specific justification. Its integrand depends on the path observed so far, without anticipating future Brownian increments. Together with the square-integrability condition below, this makes $M_t$ a martingale: conditional on the information available now, its future change has mean zero. Since $M_0=0$, its unconditional mean remains zero as well. Here the boundedness of $\nabla h$ supplies the condition automatically. Write $\|\nabla h\|_\infty=\sup_x\|\nabla h(x)\|$.

$$
\begin{aligned}
\mathbb E\int_0^t2\|\nabla h(X_s)\|^2\,ds
&\le 2t\|\nabla h\|_\infty^2<\infty,\\
\mathbb E[M_t]&=0.
\end{aligned}
$$

For finite sums, each future Brownian increment has conditional mean zero. The condition above preserves this property in the limit defining the stochastic integral. Also, $b$ is bounded on the support of $h$, making $Lh$ bounded. We may therefore exchange the time integral and expectation to obtain

$$
\mathbb E[h(X_t)]-\mathbb E[h(X_0)]
=\int_0^t\mathbb E[(Lh)(X_s)]\,ds.
$$

**4. Replace expectations by density integrals.** Applying the identity from step 1 gives a time-integrated equation for the law, with $\pi_0$ denoting the initial density:

$$
\int h(x)\pi_t(x)\,dx-\int h(x)\pi_0(x)\,dx
=\int_0^t\int (Lh)(x)\pi_s(x)\,dx\,ds.
$$

This is the time-integrated **weak form**. Spatial derivatives act only on the known smooth function $h$. Differentiating in time, whenever justified, gives the following identity; in general it holds for almost every time. All spatial integrals below are over $\mathbb R^d$.

$$
\begin{aligned}
\frac{d}{dt}\int h\pi_t\,dx
&=\int(Lh)\pi_t\,dx\\
&=\sum_{i=1}^d\int b_i\pi_t\,\partial_i h\,dx
+\sum_{i=1}^d\int\pi_t\,\partial_{ii}h\,dx.
\end{aligned}
$$

**5. Transfer spatial derivatives by integration by parts.** With a smooth density, integrate the drift term once and the diffusion term twice. The test function and its derivatives vanish sufficiently far away, so the boundary terms vanish:

$$
\begin{aligned}
\int b_i\pi_t\,\partial_i h\,dx
&=-\int h\,\partial_i(b_i\pi_t)\,dx,\\
\int\pi_t\,\partial_{ii}h\,dx
&=-\int(\partial_i\pi_t)(\partial_i h)\,dx\\
&=\int h\,\partial_{ii}\pi_t\,dx.
\end{aligned}
$$

One transfer gives a minus sign for the drift; two transfers give a plus sign for diffusion. The drift derivative acts on the **entire product $b_i\pi_t$**. Summing the components gives

$$
\int h\,\partial_t\pi_t\,dx
=\int h\bigl[-\nabla\cdot(b\pi_t)+\Delta\pi_t\bigr]\,dx.
$$

**6. Turn the test-function identity into the density equation.** Bring the two sides together:

$$
\int h\,\bigl[\partial_t\pi_t
+\nabla\cdot(b\pi_t)-\Delta\pi_t\bigr]\,dx=0.
$$

If the bracketed function were continuous and positive at some point, a nonnegative test function supported near that point would make the integral positive. The same argument excludes a negative value. The continuous bracket must therefore vanish everywhere, giving the equation stated in the main text:

$$
\partial_t\pi_t=-\nabla\cdot(b\pi_t)+\Delta\pi_t.
$$

With less regular densities, retain the integral identity in step 4 and interpret the equation in the distributional sense. This does not presuppose classical derivatives of the density. Writing $L^*\pi=-\nabla\cdot(b\pi)+\Delta\pi$ gives

$$
\int(Lh)\pi\,dx=\int h(L^*\pi)\,dx,
\qquad \partial_t\pi_t=L^*\pi_t.
$$

This integral relation makes $L^*$ the formal adjoint of the generator. $L$ acts on observables, whereas $L^*$ acts on densities. The route is **pathwise Itô formula → evolution of an expected observable → integration by parts → density equation**.

**What changes with a different noise coefficient?** For the Itô equation $dX_t=b(X_t)dt+\Sigma(X_t)dW_t$, with $m$-dimensional Brownian motion and $\Sigma(x)\in\mathbb R^{d\times m}$, define the diffusion covariance matrix $a(x)=\Sigma(x)\Sigma(x)^\top$. The same calculation gives

$$
\begin{aligned}
Lh&=b\cdot\nabla h+\frac12\sum_{i,j=1}^d a_{ij}\partial_{ij}h,\\
\partial_t\pi_t
&=-\sum_{i=1}^d\partial_i(b_i\pi_t)
+\frac12\sum_{i,j=1}^d\partial_i\partial_j(a_{ij}\pi_t).
\end{aligned}
$$

In the main text, $\Sigma=\sqrt2 I_d$ gives $a=2I_d$, so the diffusion term is exactly $\Delta\pi_t$. When $a$ varies with position, the derivatives in the last equation act on the full product $a_{ij}\pi_t$.

This derives the background needed for the stationary equation in [paper §2.3](https://arxiv.org/html/2509.24912v2#S2.SS3), using the notation of this review. For the generator, adjoint, and integration-by-parts connection, see [Goodman's lecture notes, Lesson 5, §2, pp.4–6](https://math.nyu.edu/~goodman/teaching/StochCalc2018/notes/Lesson5.pdf#page=4).

</div>
</details>

Define the probability current $j_t$ to express the density equation as a conservation law:

$$
j_t=b\pi_t-\nabla\pi_t,\qquad
\partial_t\pi_t=-\nabla\cdot j_t.
$$

In particular, for $b=-\nabla f$ and a normalizable smooth Gibbs density $\pi=Z_f^{-1}e^{-f}$, direct differentiation gives

$$
\begin{aligned}
\nabla\pi&=-\pi\nabla f,\\
j&=(-\nabla f)\pi-(-\pi\nabla f)=0,\\
L^*\pi&=-\nabla\cdot j=0
\end{aligned}
$$

Taking $f=v-\log p_{\mathrm{prior}}$ recovers the posterior from the previous section. This verifies that the target density satisfies the stationary equation; long-time convergence to it additionally requires suitable existence, uniqueness, and convergence conditions.

A learned non-gradient field may have $j\ne0$ while $\nabla\cdot j=0$: probability can circulate without changing the density at any location. A non-reversible stationary law therefore need not have an arbitrary scalar-potential expression $e^{-f}$. This motivates the paper's WKB analysis, developed in [Part 3's Appendix B.4 discussion](/en/reviews/score-geometry-wkb/).

The WKB starting form, for a small parameter $\theta>0$, is

$$
\pi_\theta(x)=\frac1{Z_\theta}
e^{-V(x)/\theta}c_\theta(x),\qquad
c_\theta\longrightarrow c_0>0\quad\text{in }C^2.
$$

Here $Z_\theta$ normalizes the density, $V$ determines where mass concentrates, and $c_\theta$ is the amplitude. This combines the exponential/amplitude separation of Equation (5) with the convergence requirement of Assumption B.2. The stationary equation determines $V$ and $c_0$ once this form is assumed. [Paper Equation (5)](https://arxiv.org/html/2509.24912v2#S2.E5), [Assumption B.2](https://arxiv.org/html/2509.24912v2#A2.Thmassumption2).

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

With the additional $C^2$ data-density regularity used in Appendix B.5, uniformly in a fixed sufficiently narrow tube,

$$
s^*(x,\sigma)
=-\frac{x-P_{\mathcal M}(x)}{\sigma^2}+O(1).
$$

The distance-gradient identity makes the leading term point toward the nearest manifold point. This follows by estimating differentiated Gaussian integrals, rather than differentiating an $o(1)$ log-density remainder. [Paper Equation (20), Appendix B.5](https://arxiv.org/html/2509.24912v2#A2.E20).

<details class="review-background">
<summary><span>Supplementary explanation: how large is the score itself?</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

The estimate above has different magnitudes depending on where the point is evaluated.

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

**1. Concentration near the manifold.** For every fixed $\delta>0$,

$$
E_\sigma=o(\sigma^{-2})
\quad\Longrightarrow\quad
\int_{\{x:\operatorname{dist}(x,\mathcal M)>\delta\}}
\pi_\sigma(x)\,dx\longrightarrow0
\qquad(\delta>0).
$$

**2. The frequency along the manifold may change.** For any $C^1$ target density with $\eta\ge0$ and $\int_{\mathcal M}\eta\,d\mathcal M=1$, one can construct generated densities satisfying the theorem's assumptions and

$$
\pi_\sigma(x)\,dx\Rightarrow\eta(x)\,d\mathcal M(x),
\qquad E_\sigma=\Omega(1).
$$

**3. Recovery of the original data law.** Vanishing absolute score error gives

$$
E_\sigma=o(1)
\quad\Longrightarrow\quad
\pi_\sigma(x)\,dx\Rightarrow\mu_{\mathrm{data}}.
$$

These are concentration, possible density distortion, and a sufficient condition for distribution recovery respectively. The $\Omega(1)$ statement is a lower bound; it does not itself assert an upper bound by a constant. Concentration alone also permits almost all mass to collect on a small part of the support. [Theorem 4.1](https://arxiv.org/html/2509.24912v2#S4.Thmtheorem1)

The proof works with $\Delta_\sigma=\log\pi_\sigma-\log p_\sigma$. Path integration gives

$$
|\Delta_\sigma(x)-\Delta_\sigma(y)|\le L_KE_\sigma.
$$

This controls oscillation of a log-density ratio, the difference between its supremum and infimum. Normalization and concentration on $K$ then fix its otherwise unknown additive constant. [Part 2](/en/reviews/score-geometry-laplace/) provides the complete argument and a smooth ambient counterexample showing why the $o(1)$ sufficient condition is not a necessary condition for every successful sequence.

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

For an exact score, normalization gives

$$
\widetilde\pi_\sigma(x)
=\frac{p_\sigma(x)^{\sigma^\alpha}}
{\int_{\mathbb R^d}p_\sigma(z)^{\sigma^\alpha}\,dz}.
$$

Substitution of the density expansion yields

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

gives the normalized uniform limit

$$
\begin{aligned}
\widetilde\pi_\sigma(x)\,dx
&\Rightarrow\frac{d\mathcal M}{V_{\mathcal M}},
\qquad V_{\mathcal M}=\int_{\mathcal M}d\mathcal M,\\
\widetilde\pi_0(u)
&=\frac{\sqrt{\det g(u)}}
{\int_U\sqrt{\det g(w)}\,dw}.
\end{aligned}
$$

The coordinate expression uses the earlier convention of combining local integrals with a partition of unity.

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

Insert each prior into Bayes' formula. Hold the observation $y$ fixed and abbreviate $v(x;y)$ to $v(x)$. The two normalized posterior measures are

$$
\begin{aligned}
d\mu_y^{\mathrm{data}}(x)
&=\frac{e^{-v(x)}\rho(x)\,d\mathcal M(x)}
{\int_{\mathcal M}e^{-v(z)}\rho(z)\,d\mathcal M(z)},\\
d\mu_y^{\mathrm{unif}}(x)
&=\frac{e^{-v(x)}\,d\mathcal M(x)}
{\int_{\mathcal M}e^{-v(z)}\,d\mathcal M(z)}.
\end{aligned}
$$

The first gives more prior weight to locations frequently observed in the data, through $\rho$. The second assigns equal prior probability to equal manifold volumes. The tempered construction targets the second posterior. [Paper §6 and Theorem 6.1](https://arxiv.org/html/2509.24912v2#S6)

Uniform volume maximizes differential entropy relative to the manifold-volume reference. Use Kullback–Leibler divergence (KL) to calculate this bound. Let $V_{\mathcal M}=\int_{\mathcal M}d\mathcal M$. Then

$$
-\int_{\mathcal M}\rho\log\rho\,d\mathcal M
=\log V_{\mathcal M}
-D_{\mathrm{KL}}\!\left(\rho\,d\mathcal M\;\middle\|\;
\frac{d\mathcal M}{V_{\mathcal M}}\right)
\le\log V_{\mathcal M}.
$$

The calculation below establishes both the equality and the inequality. The maximum-entropy statement uses manifold volume as its reference measure.

<details class="review-background">
<summary><span>Supplementary explanation: calculating the maximum entropy of uniform volume</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

<strong>Supplementary derivation.</strong> Write the uniform volume density as $u_{\mathcal M}=1/V_{\mathcal M}$. Both $\rho$ and $u_{\mathcal M}$ integrate to one against $d\mathcal M$. Consider positive densities for which the following integrals are finite.

**1. Expand the logarithm in the divergence.** For these densities with respect to the same reference measure,

$$
\begin{aligned}
D_{\mathrm{KL}}(\rho\|u_{\mathcal M})
&=\int_{\mathcal M}\rho\log\frac{\rho}{u_{\mathcal M}}\,d\mathcal M\\
&=\int_{\mathcal M}\rho\bigl(\log\rho+\log V_{\mathcal M}\bigr)\,d\mathcal M\\
&=\int_{\mathcal M}\rho\log\rho\,d\mathcal M+\log V_{\mathcal M}.
\end{aligned}
$$

The last line takes the constant $\log V_{\mathcal M}$ outside the integral and uses $\int\rho\,d\mathcal M=1$. Rearranging gives the entropy identity in the main text.

**2. Establish nonnegativity explicitly.** Set $t=\rho/u_{\mathcal M}>0$. The function $G(t)=t\log t-t+1$ satisfies

$$
G'(t)=\log t,\qquad G''(t)=\frac1t>0,\qquad
G(t)\ge G(1)=0
$$

Thus $t\log t\ge t-1$. Multiply by the positive density $u_{\mathcal M}$ and integrate:

$$
\begin{aligned}
D_{\mathrm{KL}}(\rho\|u_{\mathcal M})
&=\int_{\mathcal M}u_{\mathcal M}\,t\log t\,d\mathcal M\\
&\ge\int_{\mathcal M}u_{\mathcal M}(t-1)\,d\mathcal M\\
&=\int_{\mathcal M}\rho\,d\mathcal M
-\int_{\mathcal M}u_{\mathcal M}\,d\mathcal M=0.
\end{aligned}
$$

Equality requires $t=1$ almost everywhere, meaning that the densities coincide. The maximum entropy is therefore $\log V_{\mathcal M}$, attained by uniform volume. This is an elementary supplementary calculation of density ratios, not a new theorem of the paper. Uniform coordinate density is a different object; see the [metric and volume correction earlier](#manifolds-coordinates-and-two-densities).

</div>
</details>

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

The following <strong>supplementary derivation</strong> establishes the gradient case through explicit normalization. The paper's proof for general non-gradient scores is linked separately afterward.

**1. Separate the stationary law without observations.** Define the normalizable law $\nu_\sigma$ below. Under the conditions of Theorem 5.1, it converges weakly to uniform manifold volume:

$$
\nu_\sigma(dx)
=\frac{e^{-\sigma^\alpha f_\sigma(x)}\,dx}
{\int_{\mathbb R^d}e^{-\sigma^\alpha f_\sigma(z)}\,dz}
\ \Rightarrow\ \frac{d\mathcal M}{V_{\mathcal M}}.
$$

Adding observation guidance gives drift $-\nabla(v+\sigma^\alpha f_\sigma)$. Substitute $f=v+\sigma^\alpha f_\sigma$ into the probability-current calculation in §2.3. The guided stationary law $\Pi_\sigma$ is exactly

$$
\Pi_\sigma(dx)
=\frac{e^{-v(x)}\nu_\sigma(dx)}
{\int_{\mathbb R^d}e^{-v(z)}\nu_\sigma(dz)}
$$

The same prior normalizing constant appears in numerator and denominator and cancels.

**2. Evaluate the numerator and denominator separately.** For any bounded continuous function $h$, weak convergence gives

$$
\begin{aligned}
\int h(x)e^{-v(x)}\nu_\sigma(dx)
&\longrightarrow\frac{1}{V_{\mathcal M}}
\int_{\mathcal M}h(x)e^{-v(x)}\,d\mathcal M(x),\\
\int e^{-v(x)}\nu_\sigma(dx)
&\longrightarrow\frac{1}{V_{\mathcal M}}
\int_{\mathcal M}e^{-v(x)}\,d\mathcal M(x)>0.
\end{aligned}
$$

The paper assumes continuity of $v$ only in the tube, which needs attention here. Choose a smooth $\chi\in[0,1]$ equal to one on a smaller neighborhood $N$ of the manifold and zero outside the tube. Then $\widetilde v=\chi v$ extends to a bounded continuous function on the whole space. Since $v=\widetilde v$ on $N$, the discrepancy outside it satisfies

$$
\left|\int h(e^{-v}-e^{-\widetilde v})\,d\nu_\sigma\right|
\le2\|h\|_\infty e^{\|v\|_\infty}\nu_\sigma(N^c)
\longrightarrow0
$$

The limiting law is supported on $\mathcal M\subset N$, and the closed complement $N^c$ has mass tending to zero. We can therefore apply weak convergence to the continuous extension and return to the original $v$. Taking $h=1$ establishes the denominator limit.

**3. Take the ratio and change coordinates.** The limiting denominator is positive, so

$$
\int h\,d\Pi_\sigma
=\frac{\int he^{-v}\,d\nu_\sigma}{\int e^{-v}\,d\nu_\sigma}
\longrightarrow
\frac{\int_{\mathcal M}he^{-v}\,d\mathcal M}
{\int_{\mathcal M}e^{-v}\,d\mathcal M}.
$$

The common constant $1/V_{\mathcal M}$ cancels. This is weak convergence to $d\mu_y^{\mathrm{unif}}$. Substituting $x=\Phi(u)$ and $d\mathcal M=\sqrt{\det g(u)}\,du$ yields the displayed coordinate density $\pi_{\mathrm{post}}$. As earlier, multiple charts are combined using a partition of unity.

**Connection to the general-score proof.** For a non-gradient score, multiplying the unguided stationary law by $e^{-v}$ need not produce the guided stationary law. In [Appendix B.6](https://arxiv.org/html/2509.24912v2#A2.SS6), the paper instead inserts $f_1=v$ into [Theorem B.3 in B.4](https://arxiv.org/html/2509.24912v2#A2.Thmtheorem3) and applies [Theorem B.1 in B.1](https://arxiv.org/html/2509.24912v2#A2.Thmtheorem1). The rescaled guidance $-\theta\nabla v$, amplitude $e^{-v}$, and normal Hessian are connected in [Part 3's Appendix B.6 discussion](/en/reviews/score-geometry-wkb/#b6-proof-for-theorem-61). Its B.4 discussion develops the WKB calculation for the general-score stationary density.

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
- Background sources: [Kekkonen §1.3, printed pp.7–9](https://www.damtp.cam.ac.uk/research/cia/files/teaching/Inverse_Problems_Lent_2018/19y_01m_16d_LectureNotes.pdf#page=8) for Bayes' formula; [Goodman Lesson 5 §2, pp.4–6](https://math.nyu.edu/~goodman/teaching/StochCalc2018/notes/Lesson5.pdf#page=4) for the Itô/forward-equation connection.
- Figures: the series thumbnail reuses paper Figure 1; the separable-model plot is an original explanatory vector graphic. Paper Figures 1–3 are limited figure-area quotations for analysis. Vector elements are retained; Figure 1's embedded rasters keep their pixel dimensions and are encoded as WebP at quality 95. Figure 3 is rendered to a 1,500-pixel long edge and encoded as WebP at quality 92. The source carries the arXiv perpetual non-exclusive distribution license.
- Not independently reproduced: neural training, Stable Diffusion image generation, or formal verification of the theorems.
