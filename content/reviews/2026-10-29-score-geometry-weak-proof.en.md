---
title: "[4/4] Score Geometry: Proving a Uniform Limit"
description: "Complete supplementary proofs on a reflecting annulus and a confining whole-plane model, allowing residual derivatives to grow."
date: "2026-10-29"
publishedAt: "2026-10-29T08:55:40+09:00"
updatedAt: "2026-10-29T13:13:09+09:00"
topics: ["생성 모델", "확률과 기하"]
visibility: "public"
lang: "en"
translationKey: "score-geometry-weak-proof"
paperTitle: "When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis"
paperPublishedDate: "2025-09-29"
authors: ["Xiang Li", "Zebang Shen", "Ya-Ping Hsieh", "Niao He"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2509.24912v2"
thumbnail: "/assets/score-geometry/cover-4.svg"
thumbnailAlt: "A weak-stationarity proof of uniform sampling"
topicNames: {"생성 모델": "Generative Models", "확률과 기하": "Probability and Geometry"}
---

A test function is used to compare distributions through their integrals. Integrating a stationary equation against such functions can identify a uniform limit without differentiating the residual drift, the error remaining after the main drift terms are separated. For temperature $\theta$ and residual bound $\delta_\theta$, this article proves the normal mean-square estimate $M_\theta\le\delta_\theta^2+4\theta$ on a reflecting annulus, the ring between two concentric circles. This is a second-moment bound: it controls mean squared normal distance. The article then removes reflection in an explicit confining whole-plane model. <strong>Review interpretation.</strong> This supplies a supplementary route that permits growing residual derivatives, within the circle models and global assumptions stated below.

This final part contains reviewer-derived results; [[1/4] Score Geometry: Shape and Density](/en/reviews/score-geometry-manifold/), [[2/4] Score Geometry: Laplace and Error Separation](/en/reviews/score-geometry-laplace/), and [[3/4] Score Geometry: Stationarity and Mixing](/en/reviews/score-geometry-wkb/).

## 1. The justification problem

Appendix B.4 considers a stochastic differential equation (SDE) with drift

$$
b_\theta=-\nabla f_0-\theta\nabla f_1+\widehat b_\theta,
\qquad \|\widehat b_\theta\|_\infty=o(\theta)
$$

and substitutes $e^{-V/\theta}c_\theta$ into its stationary Fokker–Planck equation (FPE). This partial differential equation (PDE), when expanded pointwise, contains $\nabla\cdot\widehat b_\theta$.

Small function values do not imply small derivatives. The field amplitude $\theta^{3/2}\sin(u/\theta^2)$ is $o(\theta)$ while its angular derivative has amplitude $\theta^{-1/2}$. On a circle, integer frequencies make the example periodic. This observation questions a coefficient-matching justification; it is not by itself a counterexample to the limiting distribution.

A weak stationary equation moves derivatives onto a fixed test function. The plan is to prove normal concentration, test functions depending only on angle, and identify the stationary law of the remaining circular Laplacian. This requires less information than identifying an entire stationary-density action and amplitude.

## 2. A reflecting annulus: model and assumptions

### 2.1 Geometry and notation

Fix $0<a<1/2$ and define

$$
\mathcal A=\{x\in\mathbb R^2:1-a\le |x|\le1+a\},
\qquad r_-=1-a,\quad r_+=1+a
$$

Write $x=r(\cos u,\sin u)$ with $2\pi$-periodic angle $u$. Here $r$ is radius; the signed normal displacement is $q=r-1$. Set $e_r=(\cos u,\sin u)$ and $e_u=(-\sin u,\cos u)$. Half the squared distance to the unit circle is $q^2/2$, with gradient $qe_r$.

### 2.2 Dynamics and reflection

Start without a likelihood term:

$$
dX_t=\big[-q(X_t)e_r(X_t)+\varepsilon_\theta(X_t)\big]\,dt
+\sqrt{2\theta}\,dW_t
+n_{\mathrm{in}}(X_t)\,d\ell_t.
\tag{S1}
$$

The planar Brownian motion is $W_t$. The regulator $\ell_t$ increases only at the boundary, with inward unit normal $n_{\mathrm{in}}$: reflection pushes inward at the outer circle and away from the origin at the inner circle.

For each positive temperature assume sufficiently smooth coefficients, a well-defined reflected diffusion, and an invariant probability measure $\mu_\theta$. The argument applies to every such stationary law. Smoothness at each temperature is allowed; uniform derivative bounds as temperature vanishes are not imposed. The residual need not be a gradient and satisfies

$$
\delta_\theta=\|\varepsilon_\theta\|_{L^\infty(\mathcal A)},
\qquad \delta_\theta/\theta\longrightarrow0
\tag{S2}
$$

We use total variation distance (TV) for the projected angular laws: it is the largest difference between the probabilities the two laws assign to the same event.

### 2.3 Supplementary Proposition S.1

Let $\nu_\theta=(\operatorname{arg})_\#\mu_\theta$ be the pushforward by angle, and let $\lambda(du)=du/(2\pi)$. Then

$$
M_\theta:=\int(r-1)^2\,d\mu_\theta
\le\delta_\theta^2+4\theta,
\tag{S3}
$$

and the ambient law converges weakly to uniform length measure on the unit circle. More strongly,

$$
\|\nu_\theta-\lambda\|_{\mathrm{TV}}
\le C_a\sqrt{M_\theta}
+\frac{2\pi}{r_-}\frac{\delta_\theta}{\theta},
\qquad C_a=\frac{r_-+1}{r_-^2}
\tag{S4}
$$

The bound can be capped at the trivial TV maximum of 1. Its constants are not claimed optimal.

The added assumptions are a fixed annulus and normal reflection. A Wentzel–Kramers–Brillouin (WKB) ansatz, uniform derivative bounds on the stationary densities, and a small divergence of the residual are not required. Proposition S.1 belongs to this supplementary analysis, not to the paper's numbered theorems.

## 3. Weak stationarity and the boundary sign

### 3.1 The generator identity

The interior generator is

$$
L_\theta f=b_\theta\cdot\nabla f+\theta\Delta f,
\qquad b_\theta=-q e_r+\varepsilon_\theta
\tag{S5}
$$

For a smooth test function with zero normal derivative on the boundary, stationarity gives

$$
\int_{\mathcal A}L_\theta f\,d\mu_\theta=0
\quad\text{if }\partial_{n_{\mathrm{out}}}f=0
\tag{S6}
$$

The residual enters only through $\int\varepsilon_\theta\cdot\nabla f\,d\mu_\theta$. Its derivatives never appear. General generator characterizations of reflected invariant measures are developed in [Kang–Ramanan §§2–3, Equation (6) and Theorem 1, Portable Document Format (PDF) pp.4–7](https://arxiv.org/pdf/1204.4969v1#page=4). Only the identities needed for this smooth annulus are used here.

### 3.2 Check the boundary term explicitly

If the stationary density $p_\theta$ is smooth, define current $J_\theta=b_\theta p_\theta-\theta\nabla p_\theta$. Stationarity is $\nabla\cdot J_\theta=0$, and reflection gives zero outward current $n_{\mathrm{out}}\cdot J_\theta=0$. Integration by parts yields

$$
\begin{aligned}
\int_{\mathcal A}L_\theta f\,p_\theta dx
&=\int_{\mathcal A}J_\theta\cdot\nabla f\,dx
+\theta\int_{\partial\mathcal A}p_\theta
\partial_{n_{\mathrm{out}}}f\,dS\\
&=\theta\int_{\partial\mathcal A}p_\theta
\partial_{n_{\mathrm{out}}}f\,dS.
\end{aligned}
\tag{S7}
$$

Zero outward normal derivative of $f$ gives equality to zero. Nonnegative outward derivative gives a nonnegative generator integral. The test-function Neumann condition is distinct from the density's no-flux condition, which sets the net current through the boundary to zero: reflection generally requires $n\cdot J=0$, not $\partial_n p=0$.

[Pavliotis §4.1, Equations (4.9)–(4.12) and the reflecting-boundary discussion](https://wwwf.imperial.ac.uk/~pavl/PavliotisBook.pdf#page=83) explains this distinction. The author-hosted manuscript is dated 2015-11-11; its physical pages differ from printed page numbers. Itô's formula gives the same sign: if inward reflection decreases $f$, the stationary mean interior generator must compensate that decrease.

## 4. Prove normal concentration

Take $h=q^2/2$. Its outward normal derivative equals $a>0$ at both boundaries, so the applicable identity is an inequality:

$$
\int L_\theta h\,d\mu_\theta\ge0
\tag{S8}
$$

In polar coordinates,

$$
\nabla h=q e_r,\qquad
\Delta h=1+\frac qr=2-\frac1r\le2.
$$

and therefore

$$
L_\theta h=-q^2+q\,\varepsilon_{\theta,r}
+\theta\left(2-\frac1r\right).
$$

Combining the nonnegative generator integral with Cauchy–Schwarz gives

$$
M_\theta\le\delta_\theta\mathbb E_{\mu_\theta}|q|+2\theta
\le\delta_\theta\sqrt{M_\theta}+2\theta.
$$

Young's inequality $xy\le(x^2+y^2)/2$ then gives

$$
M_\theta\le\tfrac12M_\theta+\tfrac12\delta_\theta^2+2\theta
\quad\Longrightarrow\quad
M_\theta\le\delta_\theta^2+4\theta.
$$

Normal mean-square distance is $O(\theta)$ and the typical width is $O(\sqrt\theta)$. For fixed $\eta>0$, Markov's inequality gives

$$
\mu_\theta(|r-1|>\eta)
\le\frac{\delta_\theta^2+4\theta}{\eta^2}\to0
$$

Concentration has been proved within this model. Its angular distribution remains to be determined.

## 5. The angular equation and a total-variation bound

### 5.1 Test functions depending only on angle

Extend a periodic $C^2$ function $\varphi$ by $f(x)=\varphi(u)$. Since $r_->0$, it is smooth on the annulus and has zero normal derivative at both boundaries:

$$
\nabla f=\frac{\varphi'(u)}r e_u,\qquad
\Delta f=\frac{\varphi''(u)}{r^2}.
$$

The restoring drift is orthogonal to its gradient. Divide the stationary identity by temperature:

$$
0=\int\left[
\frac{\varphi''(u)}{r^2}
+\frac{\varepsilon_{\theta,u}(r,u)}{\theta r}
\varphi'(u)\right]d\mu_\theta.
\tag{S9}
$$

No residual derivative has been used.

### 5.2 First identify the limit through Fourier modes

The elementary bound $|1-r^{-2}|\le C_a|r-1|$ yields

$$
\left|\int\varphi''\,d\nu_\theta\right|
\le C_a\sqrt{M_\theta}\|\varphi''\|_\infty
+\frac{\delta_\theta}{\theta r_-}\|\varphi'\|_\infty.
\tag{S10}
$$

For each nonzero integer $k$, use $\varphi(u)=e^{iku}$:

$$
\left|\int e^{iku}\,d\nu_\theta\right|
\le C_a\sqrt{M_\theta}
+\frac{\delta_\theta}{\theta r_-|k|}\to0.
$$

Applying the real identity to real and imaginary parts justifies this complex notation. All nonconstant Fourier coefficients vanish. Uniform approximation of continuous circular functions by trigonometric polynomials then gives weak convergence to $\lambda$.

### 5.3 Solve a Poisson equation to control all continuous tests

For continuous periodic $g$ with $\|g\|_\infty\le1$, solve

$$
\bar g=\frac1{2\pi}\int_0^{2\pi}g(u)\,du,\qquad
\varphi''=g-\bar g
\tag{S11}
$$

The forcing has mean zero. Set $H(u)=\int_0^u(g(s)-\bar g)ds$, choose $\varphi'=H-\bar H$, then integrate once more. Both derivative and function are periodic. This construction gives

$$
\|\varphi''\|_\infty\le2,\qquad
\|\varphi'\|_\infty\le4\pi
$$

independently of the oscillation frequency or derivatives of $g$. Substituting into the angular estimate,

$$
\left|\int g\,d\nu_\theta-\int g\,d\lambda\right|
\le2C_a\sqrt{M_\theta}
+\frac{4\pi}{r_-}\frac{\delta_\theta}{\theta}.
$$

TV is half the supremum of this difference over tests bounded by 1. For finite regular measures on the circle, continuous tests give the same supremum. This proves (S4).

The TV statement concerns the <strong>angular projection</strong>. A smooth ambient law and a law supported exactly on the circle are mutually singular, so their ambient TV distance remains 1.

### 5.4 Recover ambient weak convergence

For a continuous function $F$ on the compact annulus, uniform continuity and normal concentration imply

$$
\int\big[F(r e_r(u))-F(e_r(u))\big]\,d\mu_\theta\to0.
$$

The remaining integral on the circle is determined by angular convergence. This completes the ambient weak-limit claim of Proposition S.1.

## 6. What remains at the critical residual scale?

### 6.1 An exactly solvable boundary example

Let $g(u)=-\kappa\cos u$, $\kappa>0$, and use the gradient potential

$$
U_\theta(r,u)=\frac{(r-1)^2}{2}+t_\theta g(u),
\qquad b_\theta=-\nabla U_\theta
$$

The invariant density relative to $dx=r\,dr\,du$ is proportional to $e^{-U_\theta/\theta}$. The radial integral, including its Jacobian $r$, is angle-independent and cancels. The exact angular marginal is

$$
\rho_\theta(u)=
\frac{\exp\!\big[\kappa(t_\theta/\theta)\cos u\big]}
{\int_0^{2\pi}\exp\!\big[\kappa(t_\theta/\theta)\cos s\big]\,ds}.
\tag{S12}
$$

The residual is $-t_\theta\nabla g$, of supremum $\kappa t_\theta/r_-$. Setting $t_\theta=\theta^p$ gives:

| Exponent | $t_\theta/\theta$ | Angular limit | Support concentration |
| --- | --- | --- | --- |
| $p>1$ | Tends to zero | Uniform | Unit circle |
| $p=1$ | Equals one | von Mises with concentration $\kappa$ | Unit circle |
| $0<p<1$ | Diverges | Concentrates at $u=0$ | Still the unit circle |

Thus $o(\theta)$ is a sufficient worst-case residual criterion. Allowing errors of order $\theta$ permits a nonuniform limit. It is not necessary for every individual field: circulating drifts can preserve uniformity at larger sizes.

### 6.2 Return to the paper's exponents

Appendix B.5 uses

$$
\theta=\sigma^{2-\alpha},\qquad
b_\theta=\sigma^2s_\sigma
=-\nabla d_{\mathcal M}
+o(\sigma^{2+\beta})+O(\sigma^2)
$$

so

$$
\frac{\delta_\theta}{\theta}
\le o(\sigma^{\alpha+\beta})+O(\sigma^\alpha).
$$

The conditions $\alpha>0$, $\alpha+\beta>0$, and $\alpha<2$ respectively suppress both residual contributions and retain concentration. They reproduce $\max\{-\beta,0\}<\alpha<2$. For this reflecting model, the angular estimate has the structure

$$
\|\nu_\theta-\lambda\|_{\mathrm{TV}}
\le C\left[
\sigma^{1-\alpha/2}
+o(\sigma^{\alpha+\beta})+O(\sigma^\alpha)
\right]
$$

This is an error bound for invariant angular laws in the specified model, not a finite-time guarantee for all diffusion samplers.

## 7. Exact non-conservative examples with large derivatives

### 7.1 Uniformity can coexist with circulation

An angularly invariant tangential residual $\varepsilon_\theta=\omega_\theta h(r)e_u$ rotates probability around the circle. For generic $h$ it is not a conservative field. Nevertheless,

$$
p_\theta(x)\propto w_\theta(r),
\qquad w_\theta(r)=e^{-(r-1)^2/(2\theta)}
$$

is stationary: its current is $p_\theta\varepsilon_\theta$, which is divergence-free and tangent to both boundaries. Stationarity requires conserved current, not zero current. The general distinction between reversibility and circulation is described in [Pavliotis §4.6, Proposition 4.13 and Equation (4.101)](https://wwwf.imperial.ac.uk/~pavl/PavliotisBook.pdf#page=109).

### 7.2 Make derivatives grow while preserving exactly the same density

Choose

$$
h(r)=1-\frac{(r-1)^2}{a^2},\qquad
\eta_\theta=\theta^{3/2},\qquad
k_\theta=\lceil\theta^{-2}\rceil.
$$

The integer frequency enforces periodicity. Define a stream function

$$
\psi_\theta(r,u)=
\frac{\eta_\theta}{k_\theta}
h(r)w_\theta(r)\sin(k_\theta u)
$$

and set

$$
\mathcal J_{\theta,r}=\frac1r\partial_u\psi_\theta,\qquad
\mathcal J_{\theta,u}=-\partial_r\psi_\theta,\qquad
\varepsilon_\theta=\frac{\mathcal J_\theta}{w_\theta}
$$

Mixed derivatives cancel, giving $\nabla\cdot\mathcal J_\theta=0$. Since $h(r_\pm)=0$, normal boundary current vanishes. The residual components are

$$
\begin{aligned}
\varepsilon_{\theta,r}
&=\frac{\eta_\theta h(r)}r\cos(k_\theta u),\\
\varepsilon_{\theta,u}
&=\frac{\eta_\theta}{k_\theta}
\left[\frac{(r-1)h(r)}\theta-h'(r)\right]
\sin(k_\theta u).
\end{aligned}
\tag{S13}
$$

On the fixed annulus their supremum is $O(\theta^{3/2})=o(\theta)$. But

$$
\|\partial_u\varepsilon_{\theta,r}\|_\infty
\asymp\eta_\theta k_\theta\asymp\theta^{-1/2}.
$$

diverges. The weighted conservation identity also gives

$$
\nabla\cdot\varepsilon_\theta
=\frac{r-1}{\theta}\varepsilon_{\theta,r}
$$

whose supremum is $O(\theta^{1/2})$. The divergence itself tends to zero; after division by $\theta$, its size generally diverges. These are different statements.

Non-conservativity can be checked directly. At $r=1$, $u=\pi/(2k_\theta)$,

$$
\operatorname{curl}\varepsilon_\theta
=\eta_\theta\left[
k_\theta+\frac{\theta^{-1}+2a^{-2}}{k_\theta}
\right]>0.
$$

Yet the complete stationary current is exactly $\mathcal J_\theta/Z_\theta$. Hence $p_\theta=w_\theta/Z_\theta$ is an exact invariant density at every temperature, with uniform angle. Large residual derivatives do not themselves invalidate the uniform conclusion; conservation structure explains why a weak proof can still succeed.

## 8. Retaining likelihood identifies the posterior

### 8.1 The surviving tangential drift

For $v\in C^2$ on the annulus, use

$$
b_\theta=-q e_r-\theta\nabla v+\varepsilon_\theta,
\qquad \delta_\theta=o(\theta)
$$

The same moment calculation yields

$$
M_\theta\le
\left(\theta\|\nabla v\|_\infty+\delta_\theta\right)^2+4\theta
$$

Compactness bounds the derivatives of $v$ used here. It does not replace the paper's global assumptions for an ambient whole-space model.

Write $v_0(u)=v(1,u)$. Normal concentration and continuity of the derivatives turn the angular stationary equation into

$$
\int\left[\varphi''(u)-v_0'(u)\varphi'(u)\right]d\nu(u)=0.
\tag{S14}
$$

Compactness provides a weakly convergent subsequence along any temperature sequence. Its limit lies on the circle. If every such limit is uniquely determined, the entire family converges.

### 8.2 Prove uniqueness without presupposing a density for the limit

The candidate density is $\rho_0=e^{-v_0}/\int e^{-v_0}$. For arbitrary continuous $g$, let $g_0=g-\int g\rho_0du$ and solve

$$
\varphi''-v_0'\varphi'=g_0
\quad\Longleftrightarrow\quad
(\rho_0\varphi')'=\rho_0g_0
$$

Set $H(u)=\int_0^u\rho_0(s)g_0(s)ds$. Its endpoint values agree, and the following constant makes the derivative have zero mean:

$$
\varphi'(u)=\frac{H(u)+c}{\rho_0(u)},\qquad
c=-\frac{\int_0^{2\pi}H(u)/\rho_0(u)\,du}
{\int_0^{2\pi}1/\rho_0(u)\,du}.
$$

Integrating gives a periodic $C^2$ function. Inserting it into (S14) gives $\int g\,d\nu=\int g\rho_0du$ for every continuous $g$. Thus $\nu=\rho_0du$, without assuming beforehand that the weak limit has a density or a smooth amplitude. Consequently,

$$
\mu_\theta\Rightarrow
\frac{e^{-v}\,d\mathcal M}{\int_{\mathcal M}e^{-v}\,d\mathcal M}
$$

This uses a uniform-volume prior. Retaining the original data density instead would give $\rho e^{-v}d\mathcal M$.

### 8.3 A general tangential drift can carry nonzero current

Replacing $-\nabla v$ by $A(x)$ at order $\theta$ gives limiting generator $\varphi''+a(u)\varphi'$, where $a(u)=A(1,u)\cdot e_u$. The stationary current $j=a\rho-\rho'$ is constant. Let $B(u)=\int_0^u a(s)ds$, $F=B(2\pi)$, and extend $B(u+2\pi)=B(u)+F$. Then

$$
\rho(u)=C e^{B(u)}
\int_u^{u+2\pi}e^{-B(s)}\,ds,\qquad
j=C(1-e^{-F}).
\tag{S15}
$$

Normalization fixes $C$. Differentiation gives $\rho'=a\rho+C(e^{-F}-1)$, verifying the current formula. Constant $a=\omega$ gives $\rho=1/(2\pi)$ and $j=\omega/(2\pi)$; $a=-\kappa\sin u$ gives the von Mises law and zero current.

![Residual-to-temperature scaling and stationary circulation in circle examples](/assets/score-geometry/weak-stationarity-toy.svg)

These are deterministic illustrative calculations. The left panel compares residual scaling regimes; the right shows that uniformity can coexist with circulation. For unit diffusion on the circle, $a(u)=0.7$ gives $j=0.7/(2\pi)\approx0.111408$. For $a(u)=0.7-1.2\sin u$, numerical normalization gives approximately 0.068969.

## 9. Removing reflection: a whole-plane model

Angle is undefined at the origin. On the whole plane, a test function depending on angle must be multiplied by a cutoff equal to 1 near the circle and 0 near the origin and far away. Its extra generator terms include a factor $\theta^{-1}$. Mass $o(\theta)$ on the fixed transition region is enough to remove them. The following supplementary model supplies that control using additional global assumptions.

For $x\in\mathbb R^2$ and $r=|x|$, consider

$$
dX_t=[-\nabla U(X_t)+\varepsilon_\theta(X_t)]dt+\sqrt{2\theta}\,dW_t,
\qquad U(x)=\frac{(|x|^2-1)^2}{8}
\tag{F1}
$$

The potential is globally smooth, grows at infinity, and has exactly the unit circle as its minimum set:

$$
U'(r)=\frac{r(r^2-1)}2,\quad U''(1)=1,\quad
U(1+q)=\frac{q^2}{2}+\frac{q^3}{2}+\frac{q^4}{8}.
\tag{F2}
$$

Its quadratic normal stiffness equals that of the annular distance well, while it also resolves the origin smoothly.

For each temperature assume a sufficiently smooth residual and a smooth, strictly positive, normalized invariant density $p_\theta$. Existence and regularity are assumptions here. Require the <strong>global</strong> bound $\delta_\theta=\sup_{x\in\mathbb R^2}|\varepsilon_\theta(x)|=o(\theta)$, with no temperature-uniform derivative bound.

<strong>Supplementary Proposition F.1.</strong> The angular law converges in TV to $\lambda$, and the ambient law converges weakly to uniform length measure on the unit circle. For fixed positive constants $C,c$ and sufficiently small temperatures,

$$
\mathbb E_{\mu_\theta}(r-1)^2=O(\theta),\qquad
\|\nu_\theta-\lambda\|_{\rm TV}
\le C\left[\sqrt\theta+\frac{\delta_\theta}{\theta}
+\theta^{-3/2}e^{-c/\theta}\right].
\tag{F3}
$$

The TV convention is $\sup_A|\nu(A)-\lambda(A)|=\tfrac12\sup_{\|g\|_\infty\le1}|\int g,d(\nu-\lambda)|$.

## 10. Angular integration gives an exact radial equation

The stationary current is

$$
J=(-\nabla U+\varepsilon_\theta)p_\theta-\theta\nabla p_\theta,
\qquad \nabla\cdot J=0.
\tag{F4}
$$

The current–generator correspondence is reviewed in [Pavliotis §4.1, physical PDF pp.83–84](https://wwwf.imperial.ac.uk/~pavl/PavliotisBook.pdf#page=83). The following radial reduction is a supplementary derivation.

The divergence theorem converts a volume integral of divergence into a boundary flux integral. Apply it to a disk of radius $r$. Its total outward current is zero: the current is smooth at the origin, with no point source or sink. Thus $r\int_0^{2\pi}J_rdu=0$ for every $r>0$.

Let $w_\theta(r)=\int_0^{2\pi}p_\theta(r,u)du$ and define the conditional radial residual

$$
\bar\varepsilon_r(r)=
\frac{\int_0^{2\pi}\varepsilon_r(r,u)p_\theta(r,u)du}{w_\theta(r)},
\qquad |\bar\varepsilon_r(r)|\le\delta_\theta
\tag{F5}
$$

The zero-flux equation becomes $\theta w_\theta'=(-U'+\bar\varepsilon_r)w_\theta$. The radial density is $m_\theta=rw_\theta$, including the polar Jacobian. Therefore

$$
m_\theta(r)=\frac{r}{D_\theta}
\exp\left[-\frac{U(r)}\theta+A_\theta(r)\right],\quad
A_\theta(r)=\frac1\theta\int_1^r\bar\varepsilon_r(s)ds,
\tag{F6}
$$

and

$$
|A_\theta(r)-A_\theta(s)|\le
\frac{\delta_\theta}{\theta}|r-s|.
\tag{F7}
$$

The unknown stationary density still appears inside $\bar\varepsilon_r$; this is not a closed-form solution. The bound on its primitive is nevertheless sufficient for tails and moments, without differentiating the residual.

## 11. The cutoff-region mass is smaller than temperature

Eventually $\delta_\theta/\theta\le1$. On $1\le r\le1+\sqrt\theta$, $U(r)\le C\theta$ and $|A_\theta(r)|\le\sqrt\theta$. Thus

$$
D_\theta\ge c_0\sqrt\theta.
\tag{F8}
$$

For fixed $0<\eta<1$, the set $|r-1|\ge\eta$, $0\le r\le R$ has a strictly positive minimum energy. Its unnormalized mass is exponentially small. The origin has $U(0)=1/8>0$, so it does not create another low-energy component.

The infinite tail needs a separate bound. For $r\ge4$ and $\theta\le1$, $U(r)\ge r^4/32$ and $r-1\le U(r)/(2\theta)$, giving

$$
\int_R^\infty r e^{-U(r)/\theta+A_\theta(r)}dr
\le\int_R^\infty r e^{-r^4/(64\theta)}dr
\le\frac{16\theta}{R^2}e^{-R^4/(64\theta)}\quad(R\ge4).
\tag{F9}
$$

The final inequality follows by substituting $t=r^2$ and bounding the Gaussian tail. Together with the denominator bound,

$$
\mu_\theta\{|r-1|\ge\eta\}
\le C_\eta\theta^{-1/2}e^{-c_\eta/\theta}
=o(\theta^N)\qquad\text{every fixed }N>0.
\tag{F10}
$$

This is stronger than concentration alone: it controls the temperature-divided cutoff terms. Near the circle $U(r)\ge c(r-1)^2$, so the numerator for the second normal moment is $O(\theta^{3/2})$. Polynomial factors in the remaining compact region and the far tail do not destroy exponential decay. Dividing by the denominator gives

$$
M_\theta:=\mathbb E(r-1)^2\le C\theta.
\tag{F11}
$$

## 12. Turn an angular function into a global test function

Choose smooth $\chi(r)$ equal to 1 on $[3/4,5/4]$ and zero outside $[1/2,3/2]$. For periodic $C^2$ function $\varphi$, the product $f(x)=\chi(r)\varphi(u)$ is compactly supported and smooth across the origin. Divide $\int L_\theta f\,d\mu_\theta=0$ by temperature:

$$
\begin{aligned}
0={}&\int\chi\left[r^{-2}\varphi''+
\frac{\varepsilon_u}{\theta r}\varphi'\right]d\mu_\theta\\
&+\int\left[
\frac{-U'+\varepsilon_r}{\theta}\chi'
+\chi''+\frac{\chi'}r\right]\varphi\,d\mu_\theta.
\end{aligned}
\tag{F12}
$$

The radial restoring drift is orthogonal to the angular gradient. The second line is supported where $|r-1|\ge1/4$ and is bounded in absolute value by $C\theta^{-3/2}e^{-c/\theta}\|\varphi\|_\infty$. Replacing $\chi/r^2$ by 1 in the first line costs $C\sqrt\theta\|\varphi''\|_\infty$ by the moment bound and Cauchy–Schwarz. The tangential residual costs $C\delta_\theta\|\varphi'\|_\infty/\theta$. Consequently,

$$
\left|\int\varphi''d\nu_\theta\right|
\le C\sqrt\theta\|\varphi''\|_\infty
+C\frac{\delta_\theta}{\theta}\|\varphi'\|_\infty
+C\theta^{-3/2}e^{-c/\theta}\|\varphi\|_\infty.
\tag{F13}
$$

## 13. Identify the full angular law with a Poisson equation

For continuous $g$ bounded by 1, solve

$$
\varphi''=g-\int g\,d\lambda
\tag{F14}
$$

The mean-zero forcing permits the same periodic integration as before. Fix the additive constant so that $\varphi$ has zero mean. Uniform bounds are

$$
\|\varphi''\|_\infty\le2,\quad
\|\varphi'\|_\infty\le4\pi,\quad
\|\varphi\|_\infty\le8\pi^2.
\tag{F15}
$$

Substituting into (F13) and taking the supremum over $g$ proves (F3). Continuous tests again determine TV for finite regular circular measures.

For an ambient bounded continuous $h$, uniform continuity near the circle and the exponential tail bound give $\int[h(x)-h(e_r)]d\mu_\theta\to0$. The angular limit determines the remaining integral. Ambient weak convergence follows, while ambient TV to the circle-supported law is still 1.

## 14. Retain observational guidance

Replace the drift by $-\nabla U-\theta\nabla v+\varepsilon_\theta$. Add $v\in C^2(\mathbb R^2)$ and globally bounded $\|\nabla v\|_\infty$. The total radial perturbation is bounded by $\theta\|\nabla v\|_\infty+\delta_\theta=O(\theta)$. The tail and moment estimates (F8)–(F11) survive with altered constants: the primitive's Lipschitz bound need only remain bounded for those steps.

Writing $v_0(u)=v(\cos u,\sin u)$, the limiting equation is

$$
\int[\varphi''-v_0'\varphi']d\nu=0.
\tag{F16}
$$

Its unique probability solution is $Z^{-1}e^{-v_0(u)}du$. The weighted Poisson construction in §8.2 proves this without first assuming a limiting density: solve $(\rho_0\varphi')'=\rho_0g_0$ and test the limiting equation. The posterior is likelihood times the uniform-volume prior on the unit circle.

## 15. An exact whole-plane example with growing derivatives

Let $w=e^{-U/\theta}$. Choose a smooth radial $h$ supported in a fixed annulus away from the origin, with $h(1)=1$, $h'(1)=0$. Set $k_\theta=\lceil\theta^{-2}\rceil$ and $\eta_\theta=\theta^{3/2}$, and define

$$
\psi_\theta=\frac{\eta_\theta}{k_\theta}h(r)w(r)\sin(k_\theta u),
\quad J_r=\frac1r\partial_u\psi_\theta,\quad J_u=-\partial_r\psi_\theta
\tag{F17}
$$

The current is divergence-free and vanishes near the origin and infinity. With $\varepsilon=J/w$,

$$
\varepsilon_r=\frac{\eta_\theta h}r\cos(k_\theta u),\qquad
\varepsilon_u=\frac{\eta_\theta}{k_\theta}
\left(\frac{U'h}{\theta}-h'\right)\sin(k_\theta u).
\tag{F18}
$$

Its global supremum is $O(\theta^{3/2})=o(\theta)$, but at $r=1$ the angular derivative amplitude is $\eta_\theta k_\theta\asymp\theta^{-1/2}$. Its curl is generally nonzero. Nevertheless the complete current of $p=w/Z$ is exactly $J/Z$, so this is an exact stationary density with uniform angle at every temperature. The global assumptions are therefore realized by an explicit non-gradient family.

![Cutoff-region mass and normal variance in the quartic whole-plane model](/assets/score-geometry/full-space-toy.svg)

This deterministic illustration uses the exact zero-residual law for $U=(r^2-1)^2/8$. The left panel plots probability of $|r-1|\ge1/4$ divided by temperature; the right plots the second normal moment divided by temperature. Quadrature uses 400,001 uniform points on $0\le r\le4$. At temperature 0.00125 the ratios are approximately $2.45\times10^{-7}$ and 1.00475. These are calculations for the supplementary model, not learned-model results.

## 16. The geometric structure beyond a circle

For a general smooth manifold within a narrow tubular neighborhood, a thin tube with unique nearest-point projection, extend an intrinsic function by $F=\varphi\circ P$. It is constant along each straight normal fiber, the points with the same nearest projection, so

$$
\nabla d_{\mathcal M}\cdot\nabla F=0,\qquad
\left.\Delta F\right|_{\mathcal M}=\Delta_{\mathcal M}\varphi.
$$

For the second identity, the normal Hessian diagonal vanishes because the extension is fiber-constant. Along an intrinsic geodesic, a path locally minimizing length within the manifold, the second derivative has an ambient Hessian contribution and a curvature-acceleration contribution. The latter is normal while $\nabla F$ is tangent, so their inner product is zero. Summing tangent directions leaves the intrinsic Laplacian.

This suggests a weak-equation extension for squared-distance dynamics with normal reflection. Quantitative TV estimates would still need uniform projection derivatives and suitable Poisson estimates. A general potential sharing only the same minimum manifold need not have this exact orthogonality, and its varying normal Hessian can leave a nonuniform weight. Whole-space concentration and relative weights of disconnected components remain separate tasks.

## Review interpretation and open questions

<strong>Review interpretation.</strong> Reflection controls boundary current directly; the whole-plane model replaces it with an exact radial-flux argument and exponentially small cutoff error. Both proofs ultimately solve a Poisson equation to identify every continuous angular average. Neither requires a stationary-density amplitude expansion or uniform convergence of residual derivatives.

The whole-plane extension adds a globally small residual and a confining radial potential. A local score-error bound in a manifold tube does not supply those global assumptions. General manifolds, disconnected-component weights, and finite-time mixing therefore remain open directions here. Propositions S.1 and F.1 are supplementary results with their own models and numbering.

## Sources and reading scope

- Starting paper: Xiang Li, Zebang Shen, Ya-Ping Hsieh, Niao He, *When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis*. First released 2025-09-29; reviewed v2, 2026-03-16, on 2026-10-29. [Official record](https://arxiv.org/abs/2509.24912v2).
- Scope: the connection to B.4–B.6; Propositions S.1 and F.1, their derivations, and their exact examples are supplementary reviewer analysis.
- Supporting sources: [Kang–Ramanan §§2–3 and Theorem 1, physical PDF pp.1–7](https://arxiv.org/pdf/1204.4969v1#page=7); [Pavliotis author manuscript §§4.1 and 4.6, physical pp.83–84 and 109–110](https://wwwf.imperial.ac.uk/~pavl/PavliotisBook.pdf#page=83). The selected passages supply reflected-stationarity signs and probability-current background.
- Figures: an original vector cover and deterministic explanatory plots of the explicitly stated models.
- Not independently reproduced: neural training, Stable Diffusion generation, or formal machine verification of the proofs.
