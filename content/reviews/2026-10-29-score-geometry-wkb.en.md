---
title: "[3/4] Non-conservative scores, stationary laws, and mixing"
description: "The stationary-density expansion, its assumptions, the complete additional experiments, and the argument connecting tempering to mixing."
date: "2026-10-29"
publishedAt: "2026-10-29T08:55:41+09:00"
updatedAt: "2026-10-29T08:55:41+09:00"
topics: ["생성 모델", "확률과 기하"]
visibility: "public"
lang: "en"
translationKey: "score-geometry-wkb"
paperTitle: "When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis"
paperPublishedDate: "2025-09-29"
authors: ["Xiang Li", "Zebang Shen", "Ya-Ping Hsieh", "Niao He"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2509.24912v2"
thumbnail: "/assets/score-geometry/cover-3.svg"
thumbnailAlt: "Non-conservative scores, stationary laws, and mixing"
topicNames: {"생성 모델": "Generative Models", "확률과 기하": "Probability and Geometry"}
---

A learned score need not be a gradient, so its stationary density cannot generally be read from a Gibbs potential. The paper assumes a Wentzel–Kramers–Brillouin (WKB) form and derives the intrinsic uniform limit of Tempered Score (TS) Langevin when $\beta>-2$ and $\max(-\beta,0)<\alpha<2$. [Theorem 5.2, Appendices B.4–B.5](https://arxiv.org/html/2509.24912v2#A2.SS4)

<strong>Review interpretation.</strong> The important structure is the passage from normal transport to a closed tangential elliptic equation; differentiating the residual and obtaining uniform global mixing constants require their own justification.

This third part covers B.4–B.6, C, and D; [1/4 presents the main results](/en/reviews/score-geometry-manifold/), [2/4 develops the Laplace and density-ratio arguments](/en/reviews/score-geometry-laplace/), and [4/4 gives supplementary weak-stationarity proofs](/en/reviews/score-geometry-weak-proof/).

## B.4. Manifold WKB Analysis of the Stationary Distribution

Let $\mathcal M\subset\mathbb R^d$ be compact, boundaryless, $C^4$, and of dimension $n<d$. Write $d_{\mathcal M}=\tfrac12\operatorname{dist}^2(\cdot,\mathcal M)$, and let $P$ denote nearest-point projection within a narrow tube. The exact score is $s_\sigma^*=\nabla\log p_\sigma$ and its estimate is $s_\sigma$. The general-score theorem requires the positive coordinate data density to be $C^2$.

In tube coordinates $x=\Phi(u)+N(u)r$, use $g=(D_u\Phi)^\top D_u\Phi$, $J=D_{u,r}\Phi$, and $G=J^\top J$. The source locations are Theorem B.3 and Lemmas B.5–B.6, Portable Document Format (PDF) pp.24–28.

### Why a scalar potential is insufficient

For $s=-\nabla f$, the normalizable stationary density is $e^{-f}$. A neural vector field need not have this structure: the integral of $s\cdot dx$ can depend on the path. The paper instead studies the stochastic differential equation (SDE)

$$
dX_t=b_\theta(X_t)\,dt+\sqrt{2\theta}\,dW_t,\qquad
b_\theta=-\nabla f_0-\theta\nabla f_1+\widehat b_\theta,
\quad \widehat b_\theta=o(\theta).
$$

A time change gives drift $b_\theta/\theta$ and noise amplitude $\sqrt2$. Its stationary law is unchanged, although finite-time trajectories and mixing measured in those time units differ.

### What the WKB assumption contains

The stationary density is assumed to split into an exponential action and a slower amplitude:

$$
\pi_\theta(x)\propto
e^{-V(x)/\theta}c_\theta(x),\qquad
c_\theta\longrightarrow c_0>0\quad\text{in }C^2.
$$

The action $V$ locates mass at exponential scale. The limiting amplitude $c_0$ determines relative frequencies between points with the same minimal action. When an entire manifold minimizes $V$, the amplitude is essential to identifying the law along that manifold.

Assumption B.2 includes concentration in the tube, positive $c_0\in C^2$, convergence $c_\theta\to c_0$ in $C^2$, and a $C^3$ action with a uniqueness assumption. These properties are assumed, not established for arbitrary learned fields. [Bouchet–Reygner §§3.1–3.3](https://arxiv.org/abs/1507.02104) supplies the Hamilton–Jacobi/transport background, but its isolated-attractor setting differs from a manifold of minima. [Paper Assumption B.2](https://arxiv.org/html/2509.24912v2#A2.Thmassumption2)

### 1. Expand the stationary Fokker–Planck equation

Start from

$$
0=-\nabla\cdot(b_\theta\pi_\theta)+\theta\Delta\pi_\theta.
$$

Substitute the ansatz and cancel its exponential factor:

$$
\begin{aligned}
0={}&-(\nabla\cdot b_\theta)c_\theta
-b_\theta\cdot\nabla c_\theta
+\frac{b_\theta\cdot\nabla V}{\theta}c_\theta\\
&+\theta\Delta c_\theta
-2\nabla V\cdot\nabla c_\theta
-(\Delta V)c_\theta
+\frac{|\nabla V|^2}{\theta}c_\theta.
\end{aligned}
$$

As a sign check, setting $b_\theta=-\nabla f$, $V=f$, and $c_\theta=1$ cancels every term in the reversible case. The following calculations unpack the source's coefficient matching; the regularity issues associated with the residual are discussed after the derivation.

### 2. Order $\theta^{-1}$: choose the action

The leading coefficient gives the Hamilton–Jacobi equation

$$
|\nabla V|^2-\nabla f_0\cdot\nabla V=0.
$$

The choice $V=f_0$ satisfies it. The equation alone does not prove uniqueness: a constant $V$ also solves it. Selecting the action associated with the actual stationary density requires an appropriate minimum-set condition, matching, or quasipotential selection. The source uses its uniqueness assumption to select the stated branch.

### 3. Order $\theta^0$: transport along normal relaxation

With $V=f_0$, the next equation is

$$
\nabla f_0\cdot(\nabla c_0+c_0\nabla f_1)=0.
$$

Define $\widetilde c_0=e^{f_1}c_0$. Then

$$
\nabla f_0\cdot\nabla\widetilde c_0=0.
$$

Along the deterministic descent $\dot x=-\nabla f_0(x)$,

$$
\frac{d}{dt}\widetilde c_0(x(t))
=-\nabla f_0\cdot\nabla\widetilde c_0=0.
$$

Thus $\widetilde c_0$ is constant along each relaxation trajectory. This connects points on the same normal flow line; it has not yet related distinct points along the minimum manifold.

### 4. Order $\theta^1$: retain the normal derivatives

On $\mathcal M$, $\nabla f_0=0$. The next surviving equation is

$$
(\Delta f_1)c_0+\nabla f_1\cdot\nabla c_0+\Delta c_0=0,
$$

Equivalently,

$$
\Delta\widetilde c_0-\nabla f_1\cdot\nabla\widetilde c_0=0
\qquad\text{on }\mathcal M
$$

The Laplacian here is still the ambient Laplacian. Replacing it immediately by the intrinsic operator would drop normal second derivatives.

### Lemma B.5: resolve the ambient operator in tube coordinates

The Laplace–Beltrami operator is intrinsic divergence of intrinsic gradient:

$$
\Delta_{\mathcal M}h
=\frac1{\sqrt{\det g}}
\partial_{u_i}\left(
\sqrt{\det g}\,[g^{-1}]_{ij}\partial_{u_j}h
\right).
$$

Repeated indices are summed. On the unit circle it is $\partial_u^2$. On a curve parameterized with speed $v(u)=|\Phi'(u)|$, it is $v^{-1}\partial_u(v^{-1}\partial_u)$. The speed factors are why ordinary angular diffusion on an ellipse is not arc-length diffusion.

In full tube coordinates,

$$
\Delta h=\frac1{|\det J|}
\nabla_{u,r}\cdot\left(|\det J|G^{-1}\nabla_{u,r}h\right).
$$

Evaluating at $r=0$ yields the source's decomposition:

$$
\begin{aligned}
0={}&\Delta_{\mathcal M}\widetilde c_0
-(\partial_u\widetilde c_0)^\top g^{-1}\partial_u f_1\\
&+\frac{1}{\sqrt{\det g}}
(\partial_r|\det J|)^\top\partial_r\widetilde c_0
-(\partial_r\widetilde c_0)^\top\partial_r f_1
+\operatorname{Tr}(\partial^2_{rr}\widetilde c_0).
\end{aligned}
$$

Orthonormality of the normal frame cancels a mixed-metric divergence. For its column $N_p$,

$$
\langle N_p,\partial_{u_j}N_p\rangle
=\tfrac12\partial_{u_j}|N_p|^2=0.
$$

In particular, the term obtained from $d_{p,j}=\langle N_p,\partial_{u_j}N_q\rangle r_q$ has $\sum_p\partial_{r_p}d_{p,j}=0$. The normal terms are being calculated and eliminated, rather than assumed absent.

### Lemma B.6: use transport to eliminate those normal derivatives

Write transport as $a(u,r)\cdot\partial_u\widetilde c_0+b(u,r)\cdot\partial_r\widetilde c_0=0$. At $r=0$, $a=b=0$. Since the gradient vanishes all along the minimum set, tangent and mixed Hessian components vanish there. Together with the block metric, this gives $\partial_ra=0$ and $\partial_rb=H_N$, the positive-definite normal Hessian.

One normal differentiation gives

$$
H_N\,\partial_r\widetilde c_0=0.
$$

Hence $\partial_r\widetilde c_0=0$. Two normal differentiations give a Sylvester equation for $C=\partial_{rr}^2\widetilde c_0$:

$$
H_NC+CH_N+\sum_i A_i(u)\,\partial_{u_i}\widetilde c_0=0.
$$

The matrices $A_i$ come from normal second derivatives of $a_i$. Multiply by $H_N^{-1}$ and take a trace:

$$
2\operatorname{Tr}C+
\sum_i\operatorname{Tr}(H_N^{-1}A_i)\,
\partial_{u_i}\widetilde c_0=0.
$$

The normal Hessian trace is therefore a linear combination of first tangential derivatives. Crucially, no term proportional to the function itself is introduced.

### 5. Close the equation and apply the strong maximum principle

Substitution produces an intrinsic equation of the form

$$
\Delta_{\mathcal M}\widetilde c_0+
B_{\mathcal M}\cdot\nabla_{\mathcal M}\widetilde c_0=0
$$

Its principal coefficient is $g^{-1}$, locally uniformly positive definite. Compactness gives a maximum, absence of boundary makes it an interior maximum, and the strong maximum principle forces a solution to be constant on each connected component, under the coefficient regularity conditions of that principle.

The source cites Gilbarg–Trudinger Theorem 3.5. The relevant elliptic form is also stated in [Guerra–Orriols §3.2, Theorem 3.12](https://www.dpmms.cam.ac.uk/~adblg2/EllipticPDELent2026.pdf). The result is

$$
V=f_0,\qquad c_0=C e^{-f_1}
$$

which is Theorem B.3's stated conclusion on a connected domain.

The main text calls the resulting partial differential equation (PDE) “parabolic.” [Paper §5](https://arxiv.org/html/2509.24912v2#S5)

<strong>Review interpretation.</strong> The displayed operator has no evolution variable; the maximum-principle step here concerns an elliptic equation on the manifold.

### Supplementary derivation for the squared-distance well

For the application $f_0=d_{\mathcal M}=\tfrac12|r|^2$, normal relaxation fixes $u$ and sends $r$ to $e^{-t}r$. Transport and continuity at $r=0$ therefore imply

$$
\widetilde c_0(u,r)=\widetilde c_0(u,0).
$$

All normal derivatives vanish, and the remaining equation becomes

$$
\Delta_{\mathcal M}\widetilde c_0
-\nabla_{\mathcal M}f_1\cdot\nabla_{\mathcal M}\widetilde c_0=0.
$$

Multiply by $\widetilde c_0e^{-f_1}$ and integrate over the boundaryless manifold:

$$
0=-\int_{\mathcal M}e^{-f_1}
|\nabla_{\mathcal M}\widetilde c_0|^2\,d\mathcal M.
$$

The integrand is nonnegative, so the tangential gradient vanishes and the function is constant on each connected component. This is a supplementary proof using the special distance structure.

### Conditions needed to justify the coefficient argument

<strong>Review interpretation.</strong> Three issues should remain attached to the derivation.

1. <strong>Connectedness.</strong> A maximum principle gives one constant per connected component. Assumption 2.1 does not explicitly require connectedness, so component weights need an additional argument before there is one global uniform law.
2. <strong>Residual differentiation.</strong> A uniform function-value estimate $\hat b_\theta=o(\theta)$ does not imply $\nabla\cdot\hat b_\theta=o(\theta)$. Pointwise coefficient matching differentiates this residual. It needs derivative control or a weak-equation justification. Likewise, pointwise $\Delta f_1$ requires more than a bare $C^1$ assumption, or a suitable weak interpretation.
3. <strong>The ansatz itself.</strong> Positive amplitudes, their $C^2$ convergence, action selection, and tube concentration remain assumptions about stationary laws.

These are proof-justification questions, not an established counterexample under every stated assumption. The gradient theorem has a separate Gibbs–Laplace argument. [Part 4](/en/reviews/score-geometry-weak-proof/) gives direct weak proofs for a reflecting circle model and a whole-plane extension under additional global conditions.

## B.5. Proof for Theorems 5.1–5.2

The decisive score remainder is Equation (20), on PDF pp.28–31. It must be established from the integrals themselves rather than by formally differentiating the earlier log-density remainder.

### The uniform remainder in Equation (20)

For variance-exploding (VE) smoothing, differentiating the Gaussian kernel gives

$$
s_\sigma^*(x)=
\frac{\int p_{\mathrm{data}}(u)
e^{-|x-\Phi(u)|^2/(2\sigma^2)}
(\Phi(u)-x)\,du}
{\sigma^2\int p_{\mathrm{data}}(u)
e^{-|x-\Phi(u)|^2/(2\sigma^2)}\,du}
$$

This is the conditional-mean identity $s_\sigma^*=(\mathbb E[Y\mid X=x]-x)/\sigma^2$. Subtract the nearest point $P=P_{\mathcal M}(x)$:

$$
s_\sigma^*(x)+\frac{x-P}{\sigma^2}
=\frac{\mathbb E[Y-P\mid X=x]}{\sigma^2}.
$$

Knowing only that conditional displacements are $O(\sigma)$ gives a remainder $O(\sigma^{-1})$, which is too weak. The required fact is that their <strong>centered mean is $O(\sigma^2)$</strong>.

The refined Laplace result of [Majerski, Theorem 2.4 and Corollary 2.5](https://arxiv.org/abs/1511.00302v2) gives relative error $O(\sigma^2)$ when the phase is $C^4$, the amplitude is $C^2$, and non-degeneracy and tail constants are controlled. Under $u=u_*+\sigma z$, the first correction candidates are the cubic phase and linear amplitude. They are odd under the centered Gaussian and integrate to zero. The next even terms provide the improved order.

For a unit vector $v$, choose amplitude

$$
a_v(u)=p_{\mathrm{data}}(u)
\left[1+v^\top(\Phi(u)-P)\right]
$$

It agrees with the denominator amplitude at the minimum. The added 1 avoids applying a relative-error theorem to an amplitude that vanishes at its minimizer. Subtracting the ratio of the two expansions yields

$$
\mathbb E[v^\top(Y-P)\mid X=x]=O(\sigma^2).
$$

Uniform control in $x$ and $v$ gives

$$
\boxed{\quad
\sigma^2s_\sigma^*(x)+\nabla d_{\mathcal M}(x)=O(\sigma^2),
\qquad
s_\sigma^*(x)+\frac{\nabla d_{\mathcal M}(x)}{\sigma^2}=O(1).
\quad}
$$

The correction for variance-preserving (VP) smoothing is also absorbed into this bounded remainder. This estimate supplies the drift structure used in the stationary analysis.

### Derive the interval for $\alpha$

Let $e_\sigma=s_\sigma-s_\sigma^*$ satisfy $\|e_\sigma\|=o(\sigma^\beta)$. After a time change,

$$
b_\theta
=-\nabla d_{\mathcal M}
+o(\sigma^{2+\beta})+O(\sigma^2),
\qquad \theta=\sigma^{2-\alpha}.
$$

The residual must be $o(\theta)$ to use the $f_1=0$ case above:

$$
\frac{o(\sigma^{2+\beta})}{\theta}
=o(\sigma^{\alpha+\beta})\to0
\quad\text{if }\alpha+\beta>0,
\qquad
\frac{O(\sigma^2)}{\theta}=O(\sigma^\alpha)\to0
\quad\text{if }\alpha>0.
$$

The first term disappears if $\alpha+\beta>0$, the second if $\alpha>0$. Normal concentration additionally requires $\theta\to0$, or $\alpha<2$. Together,

$$
\max\{-\beta,0\}<\alpha<2
$$

A nonempty interval exists exactly for $\beta>-2$. The proof chain now has three distinct links: refined smoothing gives the drift, WKB identifies stationary action and amplitude, and the manifold Laplace limit integrates the normal directions. For the squared-distance well, $H_N=I$ then gives intrinsic uniform volume.

### A direct proof in the gradient case

If $s_\sigma=-\nabla f_\sigma$, the stationary density is $e^{-hf_\sigma}$ with $h=\sigma^\alpha$. Integrating the gradient error along bounded paths shows that $f_\sigma+\log p_\sigma$ differs from a constant by $O(E_\sigma)$. The constant disappears after normalization:

$$
-h f_\sigma=
-\frac{h}{\sigma^2}d_{\mathcal M}+o(1)
+\text{a constant absorbed into normalization}
$$

Finite density and curvature terms vanish with $h$, whereas the distance well remains strong when $h/\sigma^2\to\infty$. A rigorous denominator bound comes from integrating over the shrinking tube $|r|\le\sqrt\theta$: after removing the constant gauge, its mass is at least $c\theta^{q/2}$, where $q=d-n$. The numerator at fixed distance is exponentially small, so their ratio is bounded by

$$
C\theta^{-q/2}e^{-c_\delta/\theta}\to0.
$$

The assumed concentration on $K$ controls the rest. This supplies the source's argument with an explicit shrinking-tube normalization bound.

### The logarithmic boundary of the abstract's shorthand

Consider

$$
E_\sigma=\frac{\sigma^{-2}}{\log(1/\sigma)}.
$$

It is $o(\sigma^{-2})$, but is not $o(\sigma^\beta)$ for any fixed $\beta>-2$. Thus no fixed $\alpha<2$ in the theorem directly handles this example. For a gradient score, the same proof instead suggests the sufficient conditions

$$
h\to0,\qquad h/\sigma^2\to\infty,\qquad hE_\sigma\to0
$$

They are met by

$$
h(\sigma)=\sigma^2\sqrt{\log(1/\sigma)}
$$

for the logarithmic error above. This variable-tempering choice is a <strong>supplementary extension</strong>, retaining the same local expansion, normalization, and tightness requirements. It is not the fixed-power theorem printed in the paper.

## B.6. Proof for Theorem 6.1

With likelihood $e^{-v}$, the actual guided drift is

$$
-\nabla v+\sigma^\alpha s_\sigma.
$$

The time-rescaled form is

$$
b_\theta^v=\sigma^2s_\sigma-\theta\nabla v
=-\nabla d_{\mathcal M}-\theta\nabla v+o(\theta).
$$

Thus $f_0=d_{\mathcal M}$ and $f_1=v$. The surviving amplitude is $e^{-v}$, while the normal Hessian is the identity:

$$
\pi_\theta^v\Rightarrow
\frac{e^{-v(x)}\,d\mathcal M(x)}
{\int_{\mathcal M}e^{-v(y)}\,d\mathcal M(y)}.
$$

The gradient case follows directly by integrating $e^{-v-\sigma^\alpha f_\sigma}$. The general-score case retains the stationary-law assumptions already discussed. [Paper Appendix B.6](https://arxiv.org/html/2509.24912v2#A2.SS6)

This is a posterior under a uniform-volume prior. Keeping the original intrinsic density $\rho$ instead would give $\rho e^{-v}d\mathcal M$. Kullback–Leibler (KL) divergence expresses the maximum-entropy characterization of uniform volume:

$$
H(\eta)=\log\operatorname{Vol}(\mathcal M)
-D_{\mathrm{KL}}\!\left(
\eta\,\middle\|\,\frac{d\mathcal M}{\operatorname{Vol}(\mathcal M)}
\right).
$$

Differential entropy depends on its reference measure. Also, the source assumes globally bounded $v$ and local $C^1$ regularity. An unbounded quadratic observation loss needs additional tail analysis. Classifier-free guidance (CFG) is an application only insofar as its conditional increment represents the required fixed likelihood term; finite sampling introduces further approximation.

## C. Experimental Details and Further Experiments

Appendix C occupies PDF pp.31–34. Tables 3–4 float to the top of p.34. The main-text Table 2 is included here to keep matched settings beside their protocol.

### C.1 Numerical Simulations on Ellipse

The data manifold is $(a\cos u,b\sin u)$, with $(a,b)=(1,2)$ for the ellipse and $(1,1)$ for the circle. Training angles follow a von Mises law with concentration $\kappa=1$. Intrinsic uniformity must account for the ellipse's varying coordinate speed.

A network with four Transformer blocks and hidden dimension 128 predicts $\widehat s_\sigma=\sigma^2s_\sigma$. The denoising score matching (DSM) objective is equivalently

$$
\frac12\mathbb E\left[
\sigma^2\left|s_\sigma(X)+\frac{X-\Phi(u)}{\sigma^2}\right|^2
\right]
=\frac12\mathbb E\left[
\frac{|\widehat s_\sigma(X)+X-\Phi(u)|^2}{\sigma^2}
\right].
$$

The noise standard deviation ranges over $[0.01,50]$. Batch size is 1024; AdamW weight decay is $10^{-4}$, gradient clipping is 1, and initial learning rate is $3\times10^{-3}$.

The optimization paragraph prints a cosine decay over $4\times10^{-4}$ steps to 1% of the initial rate, followed by constant rate $4\times10^{-4}$. A fractional step count and a subsequent value different from $3\times10^{-5}$ leave the intended schedule unclear. The printed values are retained rather than replaced by a guessed training duration. [Paper C.1](https://arxiv.org/html/2509.24912v2#A3.SS1)

Sampling uses $\sigma_{\min}=0.01$, Euler–Maruyama step size 0.1, 10,000 steps, and 10,000 runs. Its drift is $\widehat s$; standard noise is $\sqrt{2\sigma^2}$ and tempered noise is $\sqrt{2\sigma^{2-\alpha}}$. This rescaling preserves the stationary comparison but changes physical time units.

### C.2 Image Generation with Diffusion Models

The comparisons use Stable Diffusion 1.5 with Denoising Diffusion Probabilistic Models (DDPM), predictor–corrector (PC), and TS sampling. Contrastive Language–Image Pre-training (CLIP) features provide prompt similarity (P-sim), where higher is better alignment, and inter-image similarity (I-sim), where lower means more feature diversity. Table cells marked P / I use these two metrics without changing the source's numerical scale.

Each setting uses 512 images resized to $256\times256$, 30 predictor steps, and CFG scale 7.5. Corrector counts are $\{5,10,15,20,30\}$; the TS sweep is $\alpha\in\{0.1,0.5,1,1.5\}$. The baseline is tuned over the same corrector counts. Feature similarities are not direct measurements of manifold-volume uniformity.

The network predicts noise $\epsilon(x,t,y)$, with prompt condition $y$ and unconditional input $\emptyset$. Let $\bar\alpha_t$ be cumulative signal power and $\sigma_t=\sqrt{1-\bar\alpha_t}$. The source's standard and tempered scores are

$$
\begin{aligned}
s_t(x,y)&=-\frac1{\sigma_t}\left[
\epsilon(x,t,\emptyset)+w\big(\epsilon(x,t,y)-\epsilon(x,t,\emptyset)\big)\right],\\
\widetilde s_t(x,y)&=-\frac1{\sigma_t}\left[
\sigma_t^\alpha\epsilon(x,t,\emptyset)+w\big(\epsilon(x,t,y)-\epsilon(x,t,\emptyset)\big)\right].
\end{aligned}
$$

Only the unconditional prediction receives the $\sigma_t^\alpha$ factor. After the DDPM predictor moves to level $t_i$, each of $n_{\mathrm{corr.}}$ corrector steps updates

$$
x_{k+1}=x_k+\delta_i\widetilde s_{t_i}(x_k,y)
+\sqrt{2\delta_i}\,\xi_k,\qquad \xi_k\sim\mathcal N(0,I).
$$

Here $\delta_i$ is the corrector step size, chosen as in Song and colleagues' Algorithm 5. After the complete reverse schedule, another $n_{\mathrm{corr.}}$ deterministic projection steps follow

$$
dx_\tau=\nabla\log p_{t_0}(x_\tau)\,d\tau
$$

at final noise level $t_0$. This final operation uses neither guidance nor injected noise. Its step count is matched between PC and TS, so that postprocessing is controlled in the comparison. [Paper C.2, Algorithm details](https://arxiv.org/html/2509.24912v2#A3.SS2)

The complete Table 2 fixes $\alpha=1$. Each cell gives P-sim / I-sim:

| Prompt | Correctors | PC | TS, $\alpha=1$ |
| --- | ---: | ---: | ---: |
| Furniture | 5 | 29.40 / 81.34 | 29.54 / 81.11 |
| Furniture | 10 | 29.30 / 81.24 | 29.58 / 80.95 |
| Furniture | 15 | 29.32 / 81.64 | 29.68 / 81.34 |
| Furniture | 20 | 28.98 / 81.72 | 29.52 / 81.15 |
| Furniture | 30 | 28.67 / 82.33 | 29.43 / 81.87 |
| Car | 5 | 26.20 / 87.20 | 26.23 / 87.14 |
| Car | 10 | 26.30 / 87.57 | 26.37 / 87.42 |
| Car | 15 | 26.24 / 87.98 | 26.32 / 87.88 |
| Car | 20 | 26.26 / 88.06 | 26.28 / 88.07 |
| Car | 30 | 26.17 / 87.94 | 26.20 / 87.87 |
| Architecture | 5 | 27.13 / 81.83 | 27.23 / 81.58 |
| Architecture | 10 | 27.13 / 81.81 | 27.27 / 81.57 |
| Architecture | 15 | 26.92 / 81.64 | 27.14 / 81.54 |
| Architecture | 20 | 26.87 / 81.60 | 27.06 / 80.97 |
| Architecture | 30 | 26.60 / 81.03 | 26.84 / 80.76 |


Across its 15 matched settings, TS improves P-sim in all 15 and reduces I-sim in 14. Cars with 20 correctors are the exception, worsening I-sim by 0.01. Repeated-seed uncertainty and error bars are not supplied. Table 1's best P-sim and best I-sim need not share a hyperparameter setting.

The paper does not separately report hardware-specific runtime or total training/inference cost for these image comparisons. More corrector steps require additional score evaluations, so metric gains and computational cost remain separate evaluation axes.

### C.3 Controlled Experiment with Ground Truth Scores

The unit-circle data have von Mises concentration $\kappa=4$ and preferred angle $\pi$. An exact score is perturbed by

$$
e(x)=-\nabla\left(\frac12|x-(1,0)|^4\right)
$$

This error is $O(1)$ in $\sigma$ on a fixed compact $K$; it is not globally bounded as a function of $x$. The example tests how a fixed error can distort tangential frequencies and how tempering removes that bias.

![Paper Figure 4 compares corrupted-score diffusion with tempered-score sampling](/assets/score-geometry/paper-figure-4.svg)

The horizontal axis is angle and the vertical axis is density. The black curve is the original von Mises frequency, while the uniform target on the right is intentionally different. Both methods use the specified $O(1)$ score error. [Paper Figure 4](https://arxiv.org/html/2509.24912v2#A3.F4)

### C.4 Sensitivity Analysis of Hyperparameter $\alpha$

The following combines every numerical entry of Tables 3–4. The $\alpha=0$ rows are the PC baseline.

| Prompt | $\alpha$ | 10 correctors: P / I | 20 correctors: P / I |
| --- | ---: | ---: | ---: |
| Architecture | 0 | 27.13 / 81.81 | 26.87 / 81.60 |
| Architecture | 0.1 | 27.12 / 81.73 | 26.85 / 81.56 |
| Architecture | 0.5 | 27.14 / 81.67 | 26.97 / 81.49 |
| Architecture | 1 | 27.27 / 81.57 | 27.06 / 80.97 |
| Architecture | 1.5 | 27.32 / 81.52 | 27.10 / 81.13 |
| Furniture | 0 | 29.30 / 81.24 | 28.98 / 81.72 |
| Furniture | 0.1 | 29.32 / 81.37 | 28.99 / 81.65 |
| Furniture | 0.5 | 29.33 / 81.06 | 29.07 / 81.40 |
| Furniture | 1 | 29.58 / 80.95 | 29.52 / 81.15 |
| Furniture | 1.5 | 30.16 / 80.76 | 30.20 / 81.39 |
| Car | 0 | 26.30 / 87.57 | 26.26 / 88.06 |
| Car | 0.1 | 26.30 / 87.58 | 26.26 / 88.09 |
| Car | 0.5 | 26.31 / 87.44 | 26.25 / 87.95 |
| Car | 1 | 26.37 / 87.42 | 26.28 / 88.07 |
| Car | 1.5 | 26.50 / 87.34 | 26.62 / 87.70 |


The two metrics are not monotone in $\alpha$. For architecture with 20 correctors, for instance, I-sim at 1.5 is worse than at 1. The theorem's admissible interval $\alpha<2$ is not an optimization rule for perceptual metrics in a pretrained image model.

## D. Convergence of TS Langevin

Appendix D, PDF pp.34–37, connects the reduction of tangential barriers to functional inequalities and mixing.

### D.1 Convergence analysis of Langevin dynamics using functional inequality

For a law $\pi$, the Poincaré inequality (PI) is

$$
\operatorname{Var}_\pi(h)
\le\frac1{C_{\mathrm{PI}}}
\int|\nabla h|^2\,d\pi
$$

The convention here makes $C_{\mathrm{PI}}$ a spectral gap: <strong>larger means faster</strong>. Some literature calls its reciprocal the Poincaré constant. The test functions belong to the weighted Sobolev space $H^1(\pi)$, with square-integrable function and weak gradient.

For reversible Langevin with drift $-\nabla U$ and noise $\sqrt2$, $\pi\propto e^{-U}$. Set $r_t=p_t/\pi-1$. Integration by parts gives

$$
\chi^2(p_t\|\pi)=\int r_t^2\,d\pi,\qquad
\frac{d}{dt}\chi^2(p_t\|\pi)
=-2\int|\nabla r_t|^2\,d\pi
\le-2C_{\mathrm{PI}}\chi^2(p_t\|\pi).
$$

For finite initial $\chi^2$ divergence,

$$
\chi^2(p_t\|\pi)
\le e^{-2C_{\mathrm{PI}}t}\chi^2(p_0\|\pi).
$$

The source's $e^{-C_{\mathrm{PI}}t}$ bound is weaker but valid. A point-mass initialization does not have finite initial $\chi^2$ and needs a smoothing argument before this estimate applies.

On a uniform unit circle, Fourier modes $e^{iku}$ have intrinsic Laplacian eigenvalues $k^2$, so the first nonzero gap is 1. Very stiff normal confinement does not remove slow transitions between deep tangential wells.

### D.2 Analyzing the effect of drift scaling to the Poincaré constant

The source considers

$$
U_\sigma(x)=\frac{d_{\mathcal M}(x)}{\sigma^2}
+\sigma^\beta\phi(x),\qquad -2<\beta<0
$$

with $\phi(x)=(|x_1|-1)^2$. For $-2<\beta<0$, the factor $\sigma^\beta$ grows and creates a large tangential barrier. Smooth double-well models exhibit the exponentially slow transitions associated with Eyring–Kramers asymptotics. Tempering changes the potential to

$$
U_\sigma^{\mathrm{TS}}=
\frac{d_{\mathcal M}}{\sigma^{2-\alpha}}
+\sigma^{\alpha+\beta}\phi.
$$

When $\alpha+\beta>0$, that tangential barrier vanishes; when $\alpha<2$, normal confinement remains. The two structures can be controlled separately because they occur at different scales.

<strong>Review interpretation.</strong> The printed $\phi$ is not differentiable at $x_1=0$ and has unbounded global oscillation. A smooth Eyring–Kramers theorem or a globally bounded perturbation theorem cannot be applied to that expression literally. A regular supplementary model is

$$
\phi_{\mathrm{smooth}}(x)=\frac{x_2^2}{1+|x|^2}
$$

which is smooth, bounded between 0 and 1, and equals $\tfrac12\sin^2u$ on the unit circle. It retains a double-well angular barrier without claiming to reproduce a paper experiment.

<details class="review-background">
<summary><span>Supplementary explanation: the Holley–Stroock perturbation bound</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

Suppose $\mu\propto e^{-U}$ and $\nu\propto e^{-U-\delta U}$, with finite oscillation $\operatorname{osc}(\delta U)=\sup\delta U-\inf\delta U$. Then

$$
C_{\mathrm{PI}}(\nu)
\ge e^{-\operatorname{osc}(\delta U)}
C_{\mathrm{PI}}(\mu).
$$

For a direct derivation, let $m\le d\nu/d\mu\le M$, so $M/m=e^{\operatorname{osc}(\delta U)}$. The variance minimization formula gives

$$
\operatorname{Var}_\nu h
=\inf_a\int(h-a)^2d\nu
\le M\operatorname{Var}_\mu h
\le\frac{M}{C_{\mathrm{PI}}(\mu)}
\int|\nabla h|^2d\mu
\le\frac{M}{mC_{\mathrm{PI}}(\mu)}
\int|\nabla h|^2d\nu.
$$

Rearranging proves the bound. The stated version is given in [Menz–Schlichting, Theorem 3.2](https://arxiv.org/abs/1202.1510v4).

</div>
</details>

For smooth bounded $\phi$, the oscillation $\sigma^{\alpha+\beta}\operatorname{osc}\phi$ tends to zero, making the TS gap close to the distance-only gap in this lower bound. The growing untempered barrier does not give a noise-independent lower bound by the same argument.

### D.3 A Refined Analysis for the Smoothed Poincaré Constant

Half the squared distance to the unit circle is nonsmooth at the origin. With $R=|x|$, the source smooths it by

$$
V_c(x)=\frac{R^2}2+\frac12-\sqrt{R^2+c^2},
\qquad c=\sigma^{3-\alpha}
$$

Direct differentiation gives

$$
|V_c-d_{\mathcal M}|\le c,\qquad
\nabla V_c=
\left(1-\frac1{\sqrt{R^2+c^2}}\right)x.
$$

At temperature $\theta=\sigma^{2-\alpha}$, the normalized potential difference has oscillation $O(c/\theta)=O(\sigma)$. Holley–Stroock can therefore transfer a gap estimate from the smoothed potential to the distance potential.

For $c<1$, its minimizing circle has radius $\sqrt{1-c^2}$. The origin is a local maximum, and in two dimensions

$$
\Delta V_c=
2-\frac{R^2+2c^2}{(R^2+c^2)^{3/2}}.
$$

Although the absolute Laplacian bound grows as $c\to0$, its sign near the maximum can help a Lyapunov argument. One must distinguish that region from the compact remainder and the far tail.

### What the Polyak–Łojasiewicz condition contributes

The Polyak–Łojasiewicz (PL) inequality near the minimum set is

$$
|\nabla V|^2\ge\nu(V-V_{\min})
$$

It allows nonconvexity and a manifold of minimizers. For $V\approx r^2/2$ in normal coordinates, $|\nabla V|^2\approx r^2=2(V-V_{\min})$ even though tangent directions are flat.

[Gong–He–Shen v2, Theorems 2 and 4](https://arxiv.org/abs/2501.00429v2) connects such local geometry, conditions on other critical points, and tail growth to a global PI through the minimum-set neighborhood. The intrinsic Laplace–Beltrami gap governs the surviving tangential mixing. These results have constants and small-temperature thresholds for their potential. If the potential itself varies through $c=c(\sigma)$, those constants and thresholds must be tracked uniformly.

### From local control to a global gap

For $L_\theta=-\nabla V\cdot\nabla+\theta\Delta$ and $W=e^{V/(2\theta)}$,

$$
\frac{L_\theta W}{\theta W}
=\frac{\Delta V}{2\theta}
-\frac{|\nabla V|^2}{4\theta^2}.
$$

If this is at most $-\lambda+b\mathbf1_U$, and the conditional law inside $U$ has gap $\rho_U$, the source's Proposition D.1 gives a bound of the form

$$
C_{\mathrm{PI}}\ge
\frac{\lambda\rho_U}{b+\rho_U}
$$

The auxiliary symbol $\lambda$ denotes Lyapunov decay, avoiding the source's reuse of $\sigma$. The argument combines return toward a central region with good mixing inside that region. [Menz–Schlichting, Definition 3.7 and Theorem 3.8](https://arxiv.org/abs/1202.1510v4) provides the corresponding local-to-global framework.

The source divides space into a minimum neighborhood, a maximum neighborhood around the origin, far tails, and the remaining compact region. A final asymptotic rate requires uniform control of all corresponding constants as $c(\sigma)$ changes.

### Printed lower bounds and their implications

Appendix D.2 gives the direct bound $C_{\mathrm{PI}}^{\mathrm{smooth}}=\Omega(c)$ and discusses a structural improvement to $\Omega(1)$. The opening of D.3 instead says the direct result is $\Omega(1/\sigma)$ and that it is improved to $\Omega(1)$. [Paper D.2–D.3, PDF pp.36–37](https://arxiv.org/pdf/2509.24912v2#page=36)

<strong>Review interpretation.</strong> With this gap convention, a larger lower bound is stronger. Passing from $\Omega(1/\sigma)$ to $\Omega(1)$ is not an improvement. Confidence is high that these direct-bound statements are inconsistent. A coherent intended direction is to distinguish $\Omega(c)$ from a structurally stronger $\Omega(1)$, but establishing the latter requires uniform constants and temperature thresholds. This affects the quantitative mixing discussion in Appendix D, not the separate chain of stationary weak-limit arguments.

The robust mechanism is that tempering can reduce metastability by lowering tangential barriers while preserving normal confinement. Exact finite-time bounds still depend on the geometry, initialization, and discretization.

## Review interpretation and open questions

<strong>Review interpretation.</strong> The strongest part of the general-score derivation is the removal of normal derivatives: a transport equation leads to a Sylvester identity, which closes a tangential elliptic equation and permits a maximum principle. For the squared-distance well, a weighted energy identity makes the same structure especially transparent.

Three qualifications remain distinct: controlling a differentiated residual, interpreting pointwise second derivatives under the stated regularity, and determining weights across disconnected components. Part 4 shows how a weak-equation approach avoids residual differentiation in explicit circle models with their own boundary or global confinement assumptions.

For mixing, the barrier-removal mechanism is useful even before a universal quantitative rate is established. The source's nonsmooth example, the changing smoothing parameter, and the inconsistent printed gap scales prevent silently upgrading the discussion to a fully verified general constant-time mixing theorem.

## Sources and reading scope

- Target paper: Xiang Li, Zebang Shen, Ya-Ping Hsieh, Niao He, *When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis*. First release 2025-09-29; reviewed v2, 2026-03-16, on 2026-10-29. [Official record](https://arxiv.org/abs/2509.24912v2).
- Scope: B.4–B.6, C.1–C.4, D.1–D.3; Theorem B.3, Lemmas B.5–B.6, Equation (20), Proposition D.1, Tables 2–4, Figure 4.
- Supporting sources: [Bouchet–Reygner §§3.1–3.3](https://arxiv.org/abs/1507.02104); [Majerski Theorem 2.4 and Corollary 2.5](https://arxiv.org/abs/1511.00302v2); [Menz–Schlichting Theorems 3.2 and 3.8](https://arxiv.org/abs/1202.1510v4); [Gong–He–Shen v2, Assumptions 1–4 and Theorems 2 and 4](https://arxiv.org/abs/2501.00429v2); [Guerra–Orriols §3.2, Theorem 3.12](https://www.dpmms.cam.ac.uk/~adblg2/EllipticPDELent2026.pdf). These are the indicated selected sections; the Gilbarg–Trudinger book is not independently read in full.
- Figures: an original vector cover; paper Figure 4 is a figure-area quotation preserved as vectors for the analysis. The paper carries the arXiv perpetual non-exclusive distribution license.
- Not independently reproduced: neural training, Stable Diffusion generation, or formal machine verification of the proofs.
