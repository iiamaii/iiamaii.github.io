---
title: "[2/4] From Laplace asymptotics to score-error separation"
description: "A detailed derivation of the manifold Laplace limit, Gaussian smoothing, and the distinct score-error requirements for support and density."
date: "2026-10-29"
publishedAt: "2026-10-29T08:55:42+09:00"
updatedAt: "2026-10-29T08:55:42+09:00"
topics: ["생성 모델", "확률과 기하"]
visibility: "public"
lang: "en"
translationKey: "score-geometry-laplace"
paperTitle: "When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis"
paperPublishedDate: "2025-09-29"
authors: ["Xiang Li", "Zebang Shen", "Ya-Ping Hsieh", "Niao He"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2509.24912v2"
thumbnail: "/assets/score-geometry/cover-2.svg"
thumbnailAlt: "From Laplace asymptotics to score-error separation"
topicNames: {"생성 모델": "Generative Models", "확률과 기하": "Probability and Geometry"}
---

When an energy is minimized along a manifold rather than at one point, small-temperature mass retains the inverse square root of the normal Hessian determinant. This Laplace limit connects the geometric score scale $\sigma^{-2}$ to the finer $O(1)$ density scale and explains the different sufficient error bounds $o(\sigma^{-2})$ and $o(1)$. [Appendices B.1–B.3](https://arxiv.org/html/2509.24912v2#A2.SS1)

<strong>Review interpretation.</strong> The rate separation is most convincing when the proof tracks both normalization and coordinate volume: neither can be discarded as a harmless constant before the relevant ratio is formed.

This second part covers Appendix A and B.1–B.3; [1/4 states the main results](/en/reviews/score-geometry-manifold/), [3/4 treats non-conservative scores and mixing](/en/reviews/score-geometry-wkb/), and [4/4 develops supplementary weak proofs](/en/reviews/score-geometry-weak-proof/).

## A. Additional Notation and Preliminaries

Let $\mathcal M\subset\mathbb R^d$ be a compact, boundaryless, $C^4$ embedded manifold of dimension $n<d$. Its coordinate data density $p_{\mathrm{data}}$ is strictly positive and $C^1$. The Gaussian smoothing parameter $\sigma$ is a standard deviation, $p_\sigma$ is the resulting ambient density, and $\pi_\sigma$ is a comparison density. The compact set $K$ contains a tube around the manifold and carries asymptotically all comparison mass.

| Symbol | Meaning and dimension |
| --- | --- |
| $d,n,q=d-n$ | Ambient, tangent, and normal dimensions |
| $\Phi(u)$ | Local chart from $\mathbb R^n$ to $\mathcal M$ |
| $\Phi(u,r)=\Phi(u)+N(u)r$ | Tube chart with normal coordinate $r\in\mathbb R^q$ |
| $P_{\mathcal M}(x)$ | Unique nearest point inside the tube |
| $d_{\mathcal M}(x)$ | Half the squared distance to the manifold |
| $J_T=D_u\Phi$, $g=J_T^\top J_T$ | Tangent Jacobian and $n\times n$ metric |
| $J=D_{u,r}\Phi$, $G=J^\top J$ | Full Jacobian and $d\times d$ metric |
| $p_{\mathrm{data}}(u)$ | Density relative to coordinate volume $du$ |
| $\rho=p_{\mathrm{data}}/\sqrt{\det g}$ | Auxiliary notation for intrinsic-volume density |
| $\widehat H(u,x)$ | Tangent Hessian in the smoothing integral, $n\times n$ |
| $H_N(u)=\partial_{rr}^2f_0(u,0)$ | Normal Hessian in the concentration integral, $q\times q$ |
| $\theta$ | Small temperature: $\sigma^2$ for ordinary smoothing, $\sigma^{2-\alpha}$ for the tempered sampler |
| $E_\sigma,\beta,\alpha$ | Worst-case score error, its exponent, and the tempering exponent |

The two Hessians answer different questions. Smoothing integrates over the <strong>tangent coordinates of the source data</strong>. Concentration of the resulting ambient law integrates over <strong>normal coordinates around the support</strong>. Their determinants cannot be substituted for one another.

<details class="review-background">
<summary><span>Supplementary explanation: why finitely many charts give uniform bounds</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

Compactness permits a finite atlas. Shrink its patches so that their closures lie inside the original domains. Continuous derivatives attain bounds on these compact closures, and a finite maximum gives constants valid across all patches. Positive continuous coordinate densities also obtain local positive lower bounds there. This is the role of the shrinking argument in Remark B.1; an arbitrary open chart domain need not itself have compact closure or uniform derivative bounds. [Paper Appendix B.1](https://arxiv.org/html/2509.24912v2#A2.SS1)

</div>
</details>

<details class="review-background">
<summary><span>Supplementary explanation: preserving positive definiteness</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

For symmetric matrices $A,B$ and a unit vector $v$,

$$
v^\top(A+B)v\ge\lambda_{\min}(A)-\|B\|,
\qquad
\lambda_{\min}(A+B)\ge\lambda_{\min}(A)-\|B\|.
$$

Thus a curvature perturbation with norm smaller than half the least eigenvalue of $g$ leaves $\widehat H=g+\text{curvature correction}$ uniformly positive definite. This elementary variational estimate supplies the part of Weyl's eigenvalue inequality used in the smoothing argument. [Paper Appendix B.2](https://arxiv.org/html/2509.24912v2#A2.SS2)

</div>
</details>

## B.1. A General Framework for the Convergence of the Limiting Distribution

### Corollary B.1: a non-degenerate Laplace integral

The paper rewrites Łapiński's Theorem 2 in minimization notation. Suppose $F$ has a unique interior minimum $x^*$, is $C^3$ nearby, has a positive-definite Hessian there, and has a positive energy gap outside that neighborhood. Let the amplitude $a$ be locally $C^1$, with the integrability and tail control required by the theorem. In dimension $m$,

$$
\int a(x)e^{-F(x)/\theta}\,dx
=e^{-F(x^*)/\theta}
\frac{(2\pi\theta)^{m/2}}{\sqrt{\det\nabla^2F(x^*)}}
\left[a(x^*)+O(\sqrt\theta)\right].
$$

Uniformity requires simultaneous control of derivatives and amplitudes, a positive lower bound for Hessian eigenvalues, and the external energy gap. The constants $c_1,\ldots,c_6,m_1,m_2$ in the paper keep track of these different requirements. [Łapiński v7, Theorem 2, Portable Document Format (PDF) p.5](https://arxiv.org/pdf/1502.03266v7#page=5)

<strong>Supplementary derivation.</strong> Set $x=x^*+\sqrt\theta,z$ and $H=\nabla^2F(x^*)$:

$$
F(x)=F(x^*)+\frac\theta2 z^\top Hz+O(\theta^{3/2}\|z\|^3).
$$

The leading integral is $e^{-F(x^*)/\theta}\theta^{m/2}a(x^*)\int e^{-z^\top Hz/2}dz$. Changing variables to $H^{1/2}z$ contributes $(\det H)^{-1/2}$, and the remaining Gaussian integral is $(2\pi)^{m/2}$. This explains both the $\sqrt\theta$ width and the determinant factor. An external contribution bounded by $e^{-c/\theta}$ is smaller than every fixed power of temperature, allowing this local computation to describe the entire integral.

### Assumption B.1 and Lemma B.1: integrate the normal directions first

The general family is

$$
f_\theta=f_0+\theta f_1+\widehat f_\theta,\qquad
\|\widehat f_\theta\|_\infty=o(\theta),
\qquad
\pi_\theta\propto e^{-f_\theta/\theta}.
$$

Here $\mathcal M$ is the minimum set of nonnegative $C^3$ function $f_0$, $f_1$ is $C^1$, and the normal Hessian is uniformly positive definite in a sufficiently narrow tube. The minimum manifold is compact, boundaryless, and $C^4$; there is a positive gap away from its minima. The local assumptions identify the tube contribution. Concentration outside the tube is imposed separately when obtaining the global limit.

An error of size $o(\theta)$ is special because it is divided by temperature in the exponential:

$$
\sup_x\left|e^{-\widehat f_\theta(x)/\theta}-1\right|\to0.
$$

Lemma B.1 fixes $u$ and integrates only $r$. For a suitable bounded $C^1$ test amplitude $h$,

$$
\begin{aligned}
&\int_{\|r\|<\epsilon}e^{-f_\theta(u,r)/\theta}h(u,r)\,dr\\
&\quad=
e^{-f_0(u,0)/\theta-f_1(u,0)}
\frac{(2\pi\theta)^{q/2}}{\sqrt{\det H_N(u)}}
\left[h(u,0)+o(1)\right].
\end{aligned}
$$

To obtain this, first apply Corollary B.1 without $\hat f_\theta$. Then bound the change by the supremum in the previous display times the positive unperturbed integral. The finite atlas and uniform non-degeneracy make the remainder uniform in $u$.

### Lemma B.2: concentration and the Portmanteau step

The exponential's denominator must be controlled, not merely its numerator. Normalize the minimum of $f_0$ to zero and suppose $f_\theta=f_0+o(1)$ uniformly in the tube. For fixed $\eta>0$, eventually $|f_\theta-f_0|<\eta/8$. On $\{f_0>\eta\}$ the perturbed energy exceeds $7\eta/8$. Choose a fixed positive-volume region $U$ near a minimum where $f_0<\eta/8$; there $f_\theta<\eta/4$. The normalized tube mass then satisfies

$$
\frac{\int_{\{f_0>\eta\}}e^{-f_\theta/\theta}\,dx}
{\int_{T_{\mathcal M}(\epsilon)}e^{-f_\theta/\theta}\,dx}
\le
\frac{\operatorname{Vol}(T_{\mathcal M}(\epsilon))}
{\operatorname{Vol}(U)}
e^{-5\eta/(8\theta)}\to0.
$$

Adding the assumed vanishing mass outside the tube gives concentration on the minimum set. If a weak limit exists, the open-set inequality in the Portmanteau theorem identifies its support:

$$
\mu(G)\le\liminf_{\theta\to0}\mu_\theta(G)
\quad(G\text{ is open}).
$$

<strong>Supplementary derivation.</strong> For an open set $G$, the continuous functions $h_k(x)=\min\{1,k,\operatorname{dist}(x,G^c)\}$ increase to its indicator. Weak convergence applies at each fixed $k$, and monotone convergence then gives the inequality. Applying it to regions separated from $\mathcal M$ rules out limiting mass there.

The source lemma states connectedness. Concentration itself only requires a common minimum value and the relevant gaps on the components; connectedness becomes more decisive in the later maximum-principle argument for a single uniform amplitude.

### Theorem B.1: the density left on the minimum manifold

With the stated local assumptions and vanishing mass outside the tube,

$$
\frac{d\mu_0}{d\mathcal M}(u)
\propto
\frac{e^{-f_1(u,0)}}{\sqrt{\det H_N(u)}}.
$$

This is an intrinsic-volume density. Its coordinate version includes another factor $\sqrt{\det g(u)}$.

<strong>Supplementary proof.</strong> Write $j(u,r)=|\det J(u,r)|$ for the ambient-volume Jacobian. For a bounded smooth test function $h$,

$$
\int h\,d\mu_\theta
=
\frac{\int_U\int_R e^{-f_\theta(u,r)/\theta}h(u,r)j(u,r)\,dr\,du}
{\int_U\int_R e^{-f_\theta(u,r)/\theta}j(u,r)\,dr\,du}
+o(1).
$$

Apply the normal integral estimate to numerator and denominator. The common minimum value of $f_0$ and the common factor $(2\pi\theta)^{q/2}$ cancel. At $r=0$, $j(u,0)=\sqrt{\det g(u)}$, giving the claimed law. Uniform errors justify integrating over each chart; continuous tests follow by approximation on the compact region.

A global chart is not assumed. Choose smooth nonnegative weights $\chi_i$ with $\sum_i\chi_i=1$, multiply each chart's amplitude by its weight, and sum finitely many local integrals. This partition of unity avoids counting overlaps twice.

The source cites Hwang's Theorem 3.1 for the limiting measure. The calculation above reconstructs the special case needed here from the displayed Laplace assumptions; Hwang's full original article was not independently accessed. [Paper Theorem B.1](https://arxiv.org/html/2509.24912v2#A2.Thmtheorem1), [Hwang's bibliographic record](https://doi.org/10.1214/aop/1176994579)

### Why the minimum set alone does not imply uniformity

Take $f_0(u,r)=\tfrac12 a(u)r^2$ and $f_1=0$. The minimum set is always $r=0$, yet a smaller normal stiffness $a(u)$ gives a wider Gaussian cross-section and hence more mass. The limiting intrinsic density is proportional to $a(u)^{-1/2}$.

For half the squared distance, $f_0=d_{\mathcal M}$, the normal Hessian is exactly $I_q$. This stronger geometric fact, together with a constant surviving amplitude, is what produces uniform volume.

## B.2. Proof for Theorem 3.1

Theorem B.2 is the precise form of the main-text expansion. Its derivation separates minimization, integration, logarithms, and the variance-preserving (VP) correction.

### 1. Locate the minimum and compute its tangent Hessian

For variance-exploding (VE) smoothing, set $F_x(u)=\tfrac12\|x-\Phi(u)\|^2$. The minimizer is the nearest-point coordinate $u_x$. Differentiation gives

$$
\partial_iF_x(u)=\langle\Phi(u)-x,\partial_i\Phi(u)\rangle.
$$

At the minimum the displacement is normal, so this derivative vanishes. Differentiating again,

$$
\partial_{ij}F_x(u)
=\langle\partial_i\Phi,\partial_j\Phi\rangle
+\langle\Phi-x,\partial_{ij}\Phi\rangle
=\widehat H_{ij}(u,x).
$$

On the manifold, $\widehat H=g$. In a sufficiently narrow tube the curvature correction is small, and the eigenvalue estimate from Appendix A preserves positive definiteness. Only the normal part of $\partial_{ij}\Phi$, the second fundamental form, contributes to its inner product with the normal displacement.

For the unit circle and $x=(R,0)$ near it, $u_x=0$, $g=1$, and $\widehat H=R$. The curvature correction to log-density is therefore $-\tfrac12\log R$. At the center, $R=0$, the Hessian degenerates and the nearest point is no longer unique. This is a geometric failure of the local assumptions, not a removable algebraic inconvenience.

### 2. Apply Laplace's method

Use $\theta=\sigma^2$ and amplitude $p_{\mathrm{data}}$:

$$
p_\sigma^{\mathrm{VE}}(x)
=
e^{-d_{\mathcal M}(x)/\sigma^2}
\frac{(2\pi\sigma^2)^{(n-d)/2}}
{\sqrt{\det\widehat H(u_x,x)}}
\left[p_{\mathrm{data}}(u_x)+O(\sigma)\right].
$$

The Gaussian normalization is $d$-dimensional, while the integration is $n$-dimensional. Their difference gives the exponent $(n-d)/2$.

### 3. Take logarithms with a uniform positive lower bound

Write the amplitude bracket as $p_{\mathrm{data}}(u_x)[1+O(\sigma)/p_{\mathrm{data}}(u_x)]$. Strict positivity on the selected compact chart regions controls the logarithm:

$$
\begin{aligned}
\log p_\sigma^{\mathrm{VE}}(x)
={}&-\frac{d_{\mathcal M}(x)}{\sigma^2}
+\log p_{\mathrm{data}}(u_x)
-\frac{d-n}{2}\log(2\pi\sigma^2)\\
&-\frac12\log\det\widehat H(u_x,x)+o(1).
\end{aligned}
$$

At $x=\Phi(u)$, the combination $\log p_{\mathrm{data}}-\tfrac12\log\det g$ equals $\log\rho$. Thus the coordinate density and geometric determinant combine into an intrinsic quantity.

### 4. Treat variance-preserving smoothing as an amplitude correction

For variance-preserving (VP) smoothing, $\gamma=\sqrt{1-\sigma^2}=1-\sigma^2/2+O(\sigma^4)$. Uniformly on compact regions,

$$
\|x-\gamma\Phi(u)\|^2
=\|x-\Phi(u)\|^2
+\sigma^2\langle\Phi(u),x-\Phi(u)\rangle+O(\sigma^4).
$$

One may therefore retain the VE phase and use the modified amplitude

$$
p_{\mathrm{data}}(u)
\exp\!\left[-\frac12\langle\Phi(u),x-\Phi(u)\rangle\right]
$$

The extra logarithmic term is $-\tfrac12\langle P_{\mathcal M}(x),x-P_{\mathcal M}(x)\rangle$, bounded on the fixed tube and zero on the manifold.

<strong>Review interpretation.</strong> The source initially treats the VE and VP minimizers as the same. At finite noise, the projected point on the scaled manifold is generally $\gamma P_{\mathcal M}(x/\gamma)$. The phase–amplitude argument above recovers the intended asymptotic without equating those finite-noise minimizers.

### 5. A small remainder need not have a small derivative

The function $\sigma\sin(x/\sigma^2)$ tends uniformly to zero, but its derivative is $\sigma^{-1}\cos(x/\sigma^2)$. Hence a log-density remainder $o(1)$ cannot simply be differentiated to obtain an $O(1)$ score remainder. Appendix B.5 uses a more precise ratio of integrals for that step. [Paper Appendix B.2](https://arxiv.org/html/2509.24912v2#A2.SS2)

## B.3. Proofs for Theorem 4.1

### Lemma B.3: exact smoothing already concentrates

For VE, $X_\sigma=Y+\sigma Z$ with $Y\in\mathcal M$, so

$$
\Pr(\operatorname{dist}(X_\sigma,\mathcal M)\ge\epsilon)
\le\Pr(\|Z\|\ge\epsilon/\sigma)\to0.
$$

The Gaussian tail decays exponentially. For VP, the additional displacement $(\sqrt{1-\sigma^2}-1)Y$ is uniformly $O(\sigma^2)$ because the support is compact; the triangle inequality gives the same conclusion. Tonelli's theorem justifies exchanging nonnegative integrals in this argument.

### Fix the unknown normalization constant

Let $m_\sigma=\pi_\sigma(K)$, $n_\sigma=\mu_\sigma(K)$, and $\Delta_\sigma=\log(\pi_\sigma/p_\sigma)$. A path bound gives $\Delta_\sigma(x)=\Delta_\sigma(x_0)+r_\sigma(x)$ with $\|r_\sigma\|_\infty\le L_KE_\sigma$. Normalization then says

$$
m_\sigma=e^{\Delta_\sigma(x_0)}
\int_Kp_\sigma(x)e^{r_\sigma(x)}\,dx.
$$

Bounding the exponential above and below yields

$$
\left|\Delta_\sigma(x_0)-\log\frac{m_\sigma}{n_\sigma}\right|
\le L_KE_\sigma,\qquad
\|\Delta_\sigma\|_\infty
\le2L_KE_\sigma+
\left|\log\frac{m_\sigma}{n_\sigma}\right|.
$$

Both masses tend to one, by Assumption 4.1 and Lemma B.3. Therefore $E_\sigma=o(\sigma^{-2})$ gives $\|\Delta_\sigma\|_\infty=o(\sigma^{-2})$, while $E_\sigma=o(1)$ gives $\|\Delta_\sigma\|_\infty=o(1)$.

<strong>Review interpretation.</strong> The source describes this reduction for every rate $o(\sigma^\eta)$ with $\eta\ge-2$. Merely knowing $m_\sigma,n_\sigma\to1$ does not give a normalization error of rate $o(\sigma^\eta)$ when $\eta>0$. The two cases needed by the main theorem, $\eta=-2$ and $0$, are supported by the displayed bound; they suffice for its conclusions.

### Lemma B.4 and conclusion 1: concentration

At distance at least $\delta$ from the manifold, the smoothed density has a bound of the form $C\sigma^{-d}e^{-\delta^2/(2\sigma^2)}$. Multiply it by $e^{\Delta_\sigma}$ and absorb an $o(\sigma^{-2})$ exponent into the negative distance penalty:

$$
\pi_\sigma(K\cap\{\operatorname{dist}(x,\mathcal M)>\delta\})
\le C_K\sigma^{-d}
\exp\!\left[-\frac{\delta^2}{4\sigma^2}\right]\to0.
$$

For VP, a fixed smaller distance constant handles the small displacement of the source manifold. Assumption 4.1 handles $K^c$. This proves the first conclusion of Theorem 4.1.

### Conclusion 3: recover the original measure

If $E_\sigma=o(1)$, then $e^{\Delta_\sigma}=1+o(1)$ uniformly on $K$. Integrating this density-ratio difference, then adding the two vanishing tails outside $K$, proves that $\pi_\sigma$ and the smooth law $\mu_\sigma$ approach each other in total variation. Since $\mu_\sigma\Rightarrow\mu_{\mathrm{data}}$, the generated law has the same weak limit.

This is a direct supplementary proof of the source's conclusion. Total variation compares two ambient smooth measures here. It is not total-variation convergence from a smooth measure to a singular manifold-supported measure.

### Conclusion 2: change the limiting density at finite score cost

Let $\rho$ be the original intrinsic density and $q(x)$ a different strictly positive $C^1$ target density on the manifold. In this paragraph $q$ is a function, rather than the codimension used earlier. Tilt the smoothed density in the tube by

$$
a(x)=\log\frac{q(P_{\mathcal M}(x))}{\rho(P_{\mathcal M}(x))},
\qquad
\pi_\sigma(x)\propto p_\sigma(x)e^{a(x)}
$$

Projection regularity and positive density lower bounds make $\nabla a$ bounded. The score changes by $O(1)$, yet the limiting intrinsic density is $q$. A smooth cutoff extends the tilt outside the tube. When $q\ne\rho$, the change need not vanish.

<strong>Supplementary completion for targets with zeros.</strong> Set $q_\eta=(q+\eta\rho)/(1+\eta)$ and let $\eta\downarrow0$. The positive amplitude $q_\eta/\rho$ converges uniformly to $q/\rho$ on the compact manifold. Extend it by a fixed smooth tube cutoff, blending it with 1 outside. The resulting bounded continuous amplitudes converge uniformly, so their integrals against $\mu_\sigma\Rightarrow\mu_{\mathrm{data}}$ identify the same target limit after normalization. Their logarithmic derivatives may grow as $\eta$ decreases; the paper's $\Omega(1)$ statement imposes no constant upper bound. For the special target $q=\rho$, the next oscillatory construction supplies a nonvanishing score discrepancy while keeping the original weak limit. [Paper Theorem 4.1 and Appendix B.3](https://arxiv.org/html/2509.24912v2#S4.Thmtheorem1)

### Supplementary derivation: why the sufficient condition has no general converse

For a positive circular density $\rho(u)$ and integer $k\to\infty$, define

$$
\pi_k(u)=Z_k^{-1}\rho(u)\exp\!\left(\frac{\sin ku}{k}\right)
$$

The normalized density ratio tends uniformly to one. Nevertheless,

$$
\frac{d}{du}\log\frac{\pi_k(u)}{\rho(u)}=\cos ku
$$

so the angular score error has supremum 1. Convergence of laws does not force convergence of their logarithmic derivatives.

The example also fits the ambient setting. Choose $\sigma_k=1/k$ and a smooth radial cutoff $\chi(R)$ equal to 1 near the unit circle and 0 near the origin and far away. Put $F_k(x)=\chi(R)\sin(ku)/k$ and $\pi^{\mathrm{amb}}_{\sigma_k}\propto p_{\sigma_k}e^{F_k}$. Since $\|F_k\|_\infty\le1/k$, the density ratio converges uniformly to one. On the circle the tangential score difference is $\cos(ku)$, so the ambient supremum error is at least 1. The cutoff removes the angular singularity at the origin, and concentration and path conditions hold on a fixed sufficiently large ball $K$.

The example concerns the converse of a sufficient condition. It does not invalidate the theorem's worst-case guarantee.

### Supplementary numerical illustrations

For exact Gaussian smoothing of a uniform unit circle, at fixed radius $R=1.2$ and $\sigma=0.01$, subtracting the geometric score gives radial remainders approximately $-0.4167$ for VE and $-0.9167$ for VP. They are close to their limits $-1/(2R)$ and $-1/2-1/(2R)$. At $R=1+\sigma$, by contrast, the leading score is of order $\sigma^{-1}$. Observation location is part of the asymptotic scaling.

With uniform $\rho$ in the oscillatory example, the maximum normalized density-ratio errors for $k=10,100,1000$ are approximately $0.1024,0.0100,0.0010$, while the angular score supremum remains 1. These are explanatory quadrature results, separate from the paper's learned-score experiments.

## Review interpretation and open questions

<strong>Review interpretation.</strong> The proof turns on two careful distinctions: tangent versus normal integration, and oscillation control versus normalization. A minimum manifold alone does not determine its limiting density because the normal cross-section can vary. For the squared-distance well, the identity normal Hessian removes that extra weighting.

Theorem 4.1 compares final generated-density scores. Its vanishing-error condition is sufficient, not a characterization of every convergent sequence. The normalization step supports the two rates actually used in the theorem; claims about faster rates require faster control of the normalizing masses as well.

## Sources and reading scope

- Target paper: Xiang Li, Zebang Shen, Ya-Ping Hsieh, Niao He, *When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis*. First released 2025-09-29; reviewed v2, 2026-03-16, on 2026-10-29. [Official record](https://arxiv.org/abs/2509.24912v2).
- Scope: Appendix A, B.1–B.3, Corollary B.1, Lemmas B.1–B.4, Theorems B.1–B.2, and the connection to Theorem 4.1.
- Supporting sources: [Łapiński v7, Theorem 2, p.5](https://arxiv.org/pdf/1502.03266v7#page=5); [Leobacher–Steinicke, Theorem 2 and Theorem C](https://arxiv.org/abs/1811.10578). Hwang's full text is not independently verified; the needed special case is derived above.
- Figures: the representative cover is an original vector illustration.
- Not independently reproduced: neural training, image generation, or formal machine verification of the proofs.
