---
title: "[2/2] LeHami in depth: the proof and the conditions for useful representations"
description: "The shear-map proof of leapfrog, the assumptions behind SG momentum, and ablations of regularization, temporal symmetry, and compute."
date: "2026-10-26"
publishedAt: "2026-10-26T13:56:31+09:00"
updatedAt: "2026-10-26T13:56:31+09:00"
topics: ["world models", "representation learning", "Hamiltonian dynamics", "geometric integration", "model predictive control"]
visibility: "public"
lang: "en"
translationKey: "lehami-symplectic-world-models-technical"
paperTitle: "LeHami: Symplectic Latent Dynamics for World Models"
paperPublishedDate: "2026-10-11"
authors: ["Romina Yalovetzky", "Randall Balestriero", "Yann LeCun"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.15267v1"
thumbnail: "/assets/reviews/lehami/lehami-fig1-concept.webp"
thumbnailAlt: "LeHami Figure 1: position-like and momentum-like blocks connected by symplectic Hamiltonian transitions."
---

LeHami's symplectic guarantee preserves transition volume, while retaining actual motion information also depends on regularization and encoder design. On the actuated pendulum with a per-frame encoder, joint Sketched-Isotropic-Gaussian Regularizer (SIGReg) application gives a within-episode share of latent variance of 0.01%, versus 48.89±16.39% with separate block regularization. [Table 8](https://arxiv.org/html/2610.15267v1#A5.T8) **Review interpretation.** The appendix makes its strongest contribution by showing empirically why volume-preserving dynamics and motion-bearing representations need separate validation: training-design failures occur within the proposed structure itself.

Following [\[1/2\] the method and main experiments](/en/reviews/lehami-symplectic-world-models/), this part follows Appendices B–E, omitting Appendix A's navigation guide.

## B. Background in Hamiltonian mechanics

Hamiltonian mechanics pairs generalized positions $\mathbf q\in\mathbb R^d$ with conjugate momenta $\mathbf p\in\mathbb R^d$. With $\mathbf z=(\mathbf q,\mathbf p)$, the flow is $\dot{\mathbf z}=\mathbf J\nabla H$:

$$
\mathbf J=\begin{pmatrix}0&I_d\\-I_d&0\end{pmatrix},
\qquad
\dot{\mathbf q}=\nabla_{\mathbf p}H,\quad
\dot{\mathbf p}=-\nabla_{\mathbf q}H.
$$

The symplectic form $\Omega=\sum_i dq_i\wedge dp_i$ records oriented area. Preserving it also preserves the volume form $\Omega^d/d!$. The Liouville equation $\partial_t\rho=-\{\rho,H\}$ states that a density $\rho$ stays constant for an observer moving with the flow. The Poisson bracket is $\{\rho,H\}=\nabla_{\mathbf q}\rho\cdot\nabla_{\mathbf p}H-\nabla_{\mathbf p}\rho\cdot\nabla_{\mathbf q}H$. [Appendix B, p.13](https://arxiv.org/pdf/2610.15267v1#page=13)

### Supplementary derivation: volume and energy conservation

For twice continuously differentiable $H$, mixed partial derivatives cancel in the vector field's divergence:

$$
\nabla\cdot\dot{\mathbf z}
=\sum_i\left(
\frac{\partial^2H}{\partial q_i\partial p_i}
-\frac{\partial^2H}{\partial p_i\partial q_i}\right)=0.
$$

Along a trajectory, however, $dH/dt=\partial_tH$. A continuous Hamiltonian without explicit time dependence conserves energy; changing actions can change $H$. This distinction underlies the actuated-pendulum experiment. Discrete integration introduces its own energy error.

## C. Theory

### C.1 Symplecticity of the leapfrog step: formal statement and proof

Appendix **Proposition 2** formalizes main-text Proposition 1. Assume $K_{\phi_1}\in C^2(\mathbb R^d)$ and $V_{\phi_2}\in C^2(\mathbb R^d\times\mathcal U)$, holding action $\mathbf u\in\mathcal U$ fixed within a step. For every integration step $h>0$ and parameter value $\phi$, leapfrog $\Psi_h$ is a $C^1$ diffeomorphism of $\mathbb R^{2d}$ satisfying

$$
D\Psi_h(\mathbf z)^\top\mathbf J D\Psi_h(\mathbf z)=\mathbf J,
\qquad \det D\Psi_h(\mathbf z)=1
$$

at every $\mathbf z$. Here $D$ denotes the state Jacobian. The proof composes two shear maps in this order: [C.1, Proposition 2, Eqs.7–8, pp.14–15](https://arxiv.org/pdf/2610.15267v1#page=14)

$$
\begin{aligned}
A_{h/2}(\mathbf q,\mathbf p)
 &=\left(\mathbf q,\mathbf p-\tfrac h2\nabla_{\mathbf q}V_{\phi_2}(\mathbf q,\mathbf u)\right),\\
B_h(\mathbf q,\mathbf p)
 &=\left(\mathbf q+h\nabla_{\mathbf p}K_{\phi_1}(\mathbf p),\mathbf p\right),\\
\Psi_h&=A_{h/2}\circ B_h\circ A_{h/2}.
\end{aligned}
$$

**1. Each shear is symplectic.** Their Jacobians are block triangular:

$$
DA_{h/2}=\begin{pmatrix}I_d&0\\-\tfrac h2\nabla^2_{\mathbf q\mathbf q}V&I_d\end{pmatrix},
\quad
DB_h=\begin{pmatrix}I_d&h\nabla^2_{\mathbf p\mathbf p}K\\0&I_d\end{pmatrix}.
$$

$C^2$ regularity makes both Hessians symmetric. Expanding a matrix product abbreviated in the paper, for symmetric $C$ and $L=\begin{pmatrix}I&0\\C&I\end{pmatrix}$, one obtains $L^\top\mathbf JL=\begin{pmatrix}C-C^\top&I\\-I&0\end{pmatrix}=\mathbf J$. The upper shear works analogously. This is a **supplementary derivation** and establishes more than unit determinant. Both determinants equal 1, and the global inverses are explicitly $A_{-h/2}$ and $B_{-h}$.

**2. Composition preserves these properties.** The chain rule gives

$$
D(\Psi\circ\Phi)^\top\mathbf J D(\Psi\circ\Phi)
=D\Phi^\top(D\Psi^\top\mathbf JD\Psi)D\Phi
=\mathbf J.
$$

Determinants multiply to 1. Actions may differ across steps, provided each shear holds its action fixed. The implementation's $K_\phi(\mathbf S^{-1}\mathbf p)$ can be absorbed into the proof's function $K(\mathbf p)$. Its Hessian $\mathbf S^{-1}\nabla^2K\mathbf S^{-1}$ remains symmetric for the learned scale matrix $\mathbf S=\operatorname{diag}(e^{\boldsymbol\sigma})$, fixed after training, so the proof applies.

**3. The three consequences follow.** If $\sigma_i$ are singular values of the $K$-step Jacobian, $\prod_{i=1}^{2d}\sigma_i=|\det D\Psi_h^K|=1$. Thus $\sigma_{\max}\ge1\ge\sigma_{\min}$, excluding simultaneous contraction in every direction. Composition is also injective, so different anchors cannot merge at any finite $K$.

An asymptotically stable fixed point requires both **Lyapunov stability** and **attraction of every point in an open neighborhood**. Suppose one exists at $\mathbf z^*$. Choose a positive-volume closed ball $C$ inside its attraction region. Stability provides a neighborhood whose future remains in an arbitrarily small $\varepsilon$-ball. Attraction and continuity give a neighborhood of each point in $C$ entering it at a finite time. Compactness yields a finite cover, and after the largest entry time the entire ball remains inside the $\varepsilon$-ball. Consequently,

$$
0<\operatorname{vol}(C)
=\operatorname{vol}(\Psi_h^K(C))
\le c_{2d}\varepsilon^{2d}
$$

a contradiction as $\varepsilon\to0$. This excludes stable attraction of an open neighborhood, a stronger condition than an individual trajectory approaching a point. [C.1 proof](https://arxiv.org/pdf/2610.15267v1#page=15)

### C.2 Scope and limitations of the guarantee

The three author remarks separate volume from accuracy, regularization, and physical coordinates. First, unlimited shear and stretching can still produce growing rollout error. Second, keeping an encoder's isotropic Gaussian distribution invariant would additionally require $\{\|\mathbf z\|^2,H_\phi\}=0$; volume preservation alone does not fix distributional shape. Third, physical quantities enter as evaluation targets for probes, not training supervision, and the names $\mathbf q$ and $\mathbf p$ do not identify them with observation-space positions and momenta. [C.2, pp.15–16](https://arxiv.org/pdf/2610.15267v1#page=15)

### C.3 Why Hamiltonian and not Lagrangian

The authors choose according to predictor computation. Hamilton's equations are first order in $(\mathbf q,\mathbf p)$, and a separable Hamiltonian admits explicit leapfrog. The Lagrangian formulation on the position–velocity tangent bundle is second order in position and does not directly give this implementation's inexpensive phase-space guarantee. [C.3, p.16](https://arxiv.org/pdf/2610.15267v1#page=16)

### C.4 Design choices in the flow and the encoder

**Actions and energy.** Holding $\mathbf u_t$ fixed within each step separates the Hamiltonian change into an action-switch term $W_t$ and a fixed-action integration residual $r_t$. Equation 10 is an exact add-and-subtract identity:

$$
\begin{aligned}
H_\phi(\mathbf z_{t+1},\mathbf u_{t+1})-H_\phi(\mathbf z_t,\mathbf u_t)
&=W_t+r_t,\\
W_t&=H_\phi(\mathbf z_{t+1},\mathbf u_{t+1})-H_\phi(\mathbf z_{t+1},\mathbf u_t),\\
r_t&=H_\phi(\mathbf z_{t+1},\mathbf u_t)-H_\phi(\mathbf z_t,\mathbf u_t).
\end{aligned}
$$

The paper interprets $W_t$ as work. Its proposed dissipative extension introduces a positive-semidefinite matrix $D_{\phi_3}=L_{\phi_3}L_{\phi_3}^\top$ into the momentum equation; it is not evaluated here. [C.4](https://arxiv.org/pdf/2610.15267v1#page=16)

**Two uses of Savitzky–Golay (SG) differentiation.** A least-squares quadratic fit to encoded positions is differentiated at the current instant. A causal four-point stencil over $t-3,\ldots,t$ initializes rollouts; a centered five-point stencil over $t-2,\ldots,t+2$ supplies training targets. Those targets never enter the predictor. Once prediction begins, no further stencil is applied.

Centered weights are $(-2,-1,0,1,2)/10$, divided by the time interval for derivative units. Their antisymmetry makes reversal of the window exactly negate the estimate. The independent frame-noise variance gain $\sum_mw_m^2$ is 0.10 for the centered stencil versus 2.45 for the causal one, a factor of 24.5. The paper also compares truncation bias. Under time reversal the causal stencil becomes a forward stencil, so it lacks the centered stencil's exact antisymmetry. [C.4, D.2](https://arxiv.org/pdf/2610.15267v1#page=17)

**Momentum and mass.** The authors explain $\mathbf p\approx\dot{\mathbf q}$ as choosing unit mass in learned coordinates. A continuous flow reproducing this relation exactly must satisfy $\nabla_{\mathbf p}H_\phi=\mathbf p$ on the data support, giving kinetic energy $\frac12\|\mathbf p\|^2+C_\phi$. In practice, differentiation and discretization introduce error and do not constrain off-support behavior, so $K_\phi$ remains an MLP. A constant mass matrix can be absorbed by a coordinate transformation; a configuration-dependent matrix, as in a double pendulum, cannot be removed by the same global linear transformation. Treating fixed-stencil velocity as conjugate momentum can then mis-specify the separable model. [C.4, Eqs.11–12, p.17](https://arxiv.org/pdf/2610.15267v1#page=17)

**Scaling and regularization.** The learned $\mathbf S=\operatorname{diag}(e^{\boldsymbol\sigma})$ adjusts relative scale; reported diagonal values range from 0.95 to 1.09. SIGReg may act on $\mathbf q$ alone, the concatenated $[\mathbf q\Vert\mathbf p]$, or separately on each block with the penalties summed. With $A=3$, prediction length $K=8$, and centered half-width 2, a clip needs 14 latent frames. Regularization uses the 10 instants admitting centered momentum. [C.4, pp.17–18](https://arxiv.org/pdf/2610.15267v1#page=18)

## D. Description of the experimental set up

### D.1 Methodology

Models share the data pipeline, optimizer, and learning-rate schedule; validation loss selects checkpoints. The default LeHami regularizes only $\mathbf q$, and planning uses 35-epoch models. Equal latent width still allocates capacity differently: LeHami uses 32+32 coordinates, LeWM one 64-dimensional code. [D.1; D.4.2](https://arxiv.org/pdf/2610.15267v1#page=18)

### D.2 Implementation of LeHami

The temporal ConvNet places one temporal convolution after a per-frame stem, followed by the existing spatial trunk and projector (Fig.6). The main LeHami uses $W=1$ without temporal mixing. Hamiltonian automatic differentiation runs in fp32. The appendix also describes mixed-recording-rate training, but **no mixed-rate training experiment is performed**. Time re-stepping changes inference intervals after single-rate training. [D.2, Fig.6, pp.19–21](https://arxiv.org/pdf/2610.15267v1#page=19)

### D.3 Datasets and their architectures

The environments follow $\ddot\theta=-g\sin\theta/L$ and $\ddot\theta=-g\sin\theta/L+u/(mL^2)$, with $g=9.81$ m/s², $L\in\{0.7,1.0,1.4\}$ m, and actuated masses $m\in\{0.5,1.0,2.0\}$ kg. Rod length is visible geometrically and mass through bob radius (Fig.7). Each torque is held for 0.3 s, or three frames. [D.3, Eqs.13–14](https://arxiv.org/pdf/2610.15267v1#page=21)

Conservative initial angles are sampled from $[-\pi,\pi]$ and angular-velocity bounds scale with $\sqrt{g/L}$, giving $|\omega_0|\le2$ rad/s at 1 m. Both oscillations and rotations occur. Episode splitting uses seed 42 and a 70/15/15 ratio within each length or length×mass cell. Only the 300 episodes at 0.1 s are retained from 1,200 generated conservative episodes. Episode counts and splits are in [part 1, §3.1](/en/reviews/lehami-symplectic-world-models/). [Table 3, D.3](https://arxiv.org/pdf/2610.15267v1#page=23)

### D.4 Evaluation

Rollout probes are fitted at a designated instant and frozen. By contrast, Figure 8's blockwise analysis uses **five-fold cross-validation grouped by episode** on features drawn from test episodes. Its protocol differs from the full-representation probes fitted on training episodes in Table 1. [D.4.1; Fig.8 caption](https://arxiv.org/pdf/2610.15267v1#page=25)

Planning encodes actual start and goal observations; intermediate future images and physical coordinates never rank candidate plans. Replanning reads only frames observed by that time. For centered temporal encoders, the context and goal are shifted by encoder lag and extra rollout steps compensate for it. Goal-time tolerances and the CEM budget are in [part 1, §3.4](/en/reviews/lehami-symplectic-world-models/). [D.4.2, pp.23–24](https://arxiv.org/pdf/2610.15267v1#page=23)

For probe $R^2=1-\mathrm{SSE}/\mathrm{SST}$, SSE sums squared residuals and SST sums squared target deviations from the mean; 1 is optimal and negative values are worse than mean prediction. Angles are read as sine/cosine pairs and errors wrapped on the circle.

## E. Supplementary experimental results

### E.1 LeHami replaces manual design choices with structural priors and lower compute

A single-frame representation has no direct information about motion direction. Reading LeHami's two blocks separately, the authors find angle mainly in $\mathbf q$ and velocity mainly in $\mathbf p$. Using a 25-nearest-neighbor (25-NN) probe, Figure 8's angular $R^2$ evaluates $(\sin\theta,\cos\theta)$ rather than raw angle.

[![Angle and velocity R squared from LeHami blocks and the full LeWM code.](/assets/reviews/lehami/lehami-fig8-blocks.webp)](/assets/reviews/lehami/lehami-fig8-blocks.webp)

Original Figure 8. Solid bars are $\mathbf q$, hatched bars $\mathbf p$; LeWM uses its full $\mathbf z$. Probes use 25-NN and five episode-grouped folds; error bars summarize three training seeds. The right panel locates the velocity signal in the momentum-like block. [Fig.8](https://arxiv.org/html/2610.15267v1#A5.F8)

Table 4 compares checkpoints selected under matched cumulative training budgets on the conservative pendulum. This table uses a $k=20$ nearest-neighbor probe and one seed per model. Each cell gives angle mean absolute error (MAE) [rad] / velocity MAE [rad/s], both lower-is-better; the checkpoint epochs are 9/19/39/79 across budgets.

| Model | 0.75 PFLOP | 1.5 PFLOP | 3 PFLOP | 6 PFLOP |
| --- | --- | --- | --- | --- |
| LeHami $W=1$ | 0.283 / 1.002 | 0.207 / 0.929 | 0.252 / 0.934 | 0.222 / 0.902 |
| LeWM $W=1$ | 0.590 / 1.625 | 0.871 / 2.146 | 0.487 / 1.620 | 0.321 / 1.270 |

LeHami at 0.75 PFLOP beats LeWM at 6 PFLOP on both metrics. This is the basis for the authors' “at most one-eighth” training-compute comparison. A 100-epoch run costs approximately 6.0–6.5 PFLOP and 0.7–1.1 GPU-hours for either model. [Table 4, E.1](https://arxiv.org/html/2610.15267v1#A5.T4)

Table 5 counts parameters in millions, separating the predictor and its projection head:

| Model | Total | Encoder | Predictor | Latent projector | Predictor projector |
| --- | --- | --- | --- | --- | --- |
| LeHami $W=1$ | 3.03 | 0.315 | 2.234 | 0.465 | None |
| LeWM $W=1$ | 4.44 | 0.315 | 3.310 | 0.530 | 0.268 |
| LeWM $W=3$ | 4.45 | 0.318 | 3.310 | 0.530 | 0.268 |

Total parameters fall by approximately 32%. Despite §3.4's 3.31M wording, Table 5 assigns 3.310M to LeWM's **predictor alone**, with another 0.268M for its projection head. [Table 5](https://arxiv.org/html/2610.15267v1#A5.T5)

### E.2 Conservative pendulum

Figures 9–11 visualize latent trajectories and energy strata using Principal Component Analysis (PCA). Figure 9 uses several principal-component pairs; Figure 10 compares LeHami and LeWM trajectories; Figure 11 juxtaposes physical energy with learned $H_\phi$. The authors observe closed curves and energy stratification. [Figs.9–11, pp.27–28](https://arxiv.org/pdf/2610.15267v1#page=27)

Table 6 measures $R^2$ between **predicted latent blocks and encoder outputs from actual observations**, rather than physical coordinates:

| Rollout step | 0 | 1 | 5 | 10 | 20 | 25 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $R^2_q$ | 1.000 | 0.997 | 0.985 | 0.950 | 0.929 | 0.646 | 0.373 |
| $R^2_p$ | 1.000 | 0.935 | 0.867 | 0.613 | 0.586 | −0.324 | −1.165 |

Momentum-like predictions depart from their reference faster than position-like predictions. Table 7 separately reports physical readout. The 25-NN probe is fitted at $K=1$ and frozen; interval-averaged MAEs are in rad for angle and rad/s for velocity. Ratios are LeWM÷LeHami, so values above 1 favor LeHami.

| Interval | Angle LeHami | Angle LeWM | Ratio | Velocity LeHami | Velocity LeWM | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| $K=1$–10 | 0.107 | **0.100** | 0.93 | **0.364** | 1.020 | 2.80 |
| $K=11$–20 | **0.225** | 0.287 | 1.27 | **0.669** | 1.169 | 1.75 |
| $K=21$–70 | **0.596** | 0.981 | 1.65 | **1.534** | 2.331 | 1.52 |

The same table gives anchor velocity $R^2$ of 0.975 for LeHami and −0.667 for LeWM, while LeHami's ridge gives 0.448. Velocity information is not fully exposed linearly. [Tables 6–7](https://arxiv.org/pdf/2610.15267v1#page=28)

Encoder checks use different conditions. On a single 1 m rod, 0.05 s spacing, and 500 episodes of 100 steps, ViT/ConvNet anchor-velocity $R^2$ is 0.929/0.939 for LeHami and −0.031/−0.018 for LeWM. Figure 12 adds learned projectors over frozen DINOv2, also at 0.05 s, using one training seed per model at epoch 60. It reports a velocity advantage for LeHami and rising errors for both beyond $K=60$. These results are not pooled with the three-length main experiment. [E.2, Fig.12](https://arxiv.org/pdf/2610.15267v1#page=29)

**Which Jacobian measures volume?** For a square state transition $F$, write $J(\mathbf z_i)=\partial F/\partial\mathbf z$. Normalized cumulative log-volume is

$$
\frac1n\sum_{i=0}^{K-1}\log|\det J(\mathbf z_i)|
$$

Zero denotes volume preservation, negative values contraction. LeHami has $n=64$. LeWM maps three embeddings to one, so its square full-state transition is $(\mathbf z_{t-2},\mathbf z_{t-1},\mathbf z_t)\mapsto(\mathbf z_{t-1},\mathbf z_t,\mathbf z_{t+1})$, with $n=192$. Up to sign, its determinant equals $\det(\partial\mathbf z_{t+1}/\partial\mathbf z_{t-2})$. The newest block $\partial\mathbf z_{t+1}/\partial\mathbf z_t$ alone does not measure full-context volume. [E.2, pp.29–30](https://arxiv.org/pdf/2610.15267v1#page=29)

[![Full-state and newest-block Jacobians as separate volume-change diagnostics.](/assets/reviews/lehami/lehami-fig13-volume.webp)](/assets/reviews/lehami/lehami-fig13-volume.webp)

Original Figure 13. Left: full maps, 64-dimensional for LeHami and 192-dimensional for LeWM. Right: 64-dimensional newest blocks. Across 45 test anchors and one training seed per model, both LeWM diagnostics are negative and LeHami is numerically near zero. This figure provides the full-context evidence. [Fig.13](https://arxiv.org/html/2610.15267v1#A5.F13)

### E.3 Actuated pendulum

Figure 14 shows trajectories crossing $H_\phi$ level sets as actions add or remove energy. Symplectic constraints still apply to each fixed-action transition. [Fig.14](https://arxiv.org/html/2610.15267v1#A5.F14)

**Regularization-placement ablation.** Table 8 computes within-episode variance $V_{\rm within}$ and variance of episode means $V_{\rm between}$ per coordinate, then averages $V_{\rm within}/(V_{\rm within}+V_{\rm between})$ across coordinates. Near-zero values indicate episode identification rather than changing state. It uses test encoder outputs over $K=0,\ldots,74$ and three training seeds.

| Window | SIGReg placement | Within-episode share [%] | $\mathrm{std}(q)$ | $\mathrm{std}(p)$ | $\mathrm{std}(p)/\mathrm{std}(q)$ |
| --- | --- | --- | --- | --- | --- |
| $W=1$ | joint | 0.01±0.00 | 1.356 | 0.006 | 0.0043 |
| $W=1$ | $q$ only | 15.32±0.29 | 0.980 | 0.101 | 0.1031 |
| $W=1$ | split | **48.89±16.39** | 1.174 | 0.236 | 0.2014 |
| $W=3$ | joint | 4.90±0.30 | 1.405 | 0.051 | 0.0362 |
| $W=3$ | $q$ only | 8.87±0.54 | 0.979 | 0.051 | 0.0519 |
| $W=3$ | split | **47.06±2.56** | 1.124 | 0.201 | 0.1784 |

The authors observe joint regularization compensating for near-zero $p$ by inflating the standard deviation of $q$ toward $\sqrt2$. Split regularization increases within-episode variation without making $p$ variance immediately equal to 1. Planning and main Table 1 use the $q$-only model. The within-episode share measures something different from physical-state decoding or planning success. [Table 8, E.3](https://arxiv.org/html/2610.15267v1#A5.T8)

**Temporal symmetry and velocity sign.** A tied temporal convolution $[a,b,a]$ is unchanged by reversing its three input frames. It cannot distinguish forward and backward windows with the same center pose. In Eq.16's constant-angular-velocity example, the symmetric average of $(\sin\theta,\cos\theta)$ is

$$
\frac{b+2a\cos(\omega\,dt)}{b+2a}(\sin\theta,\cos\theta)
$$

The cosine is even, removing velocity sign. Table 9 reports anchor 25-NN results; both predictors retain history length 3:

| LeWM encoder | $R^2(\omega)$ | $R^2(\lvert\omega\rvert)$ | Gap |
| --- | --- | --- | --- |
| $W=1$ | −0.048 | 0.032 | 0.080 |
| $W=3$, tied | −0.115 | 0.921 | 1.036 |

This concerns the **single goal representation from the encoder**. LeWM's predictor can infer direction from ordered history; the paper also reports velocity $R^2\approx0.90$ after one predictor step for $W=1$. Tied windows attenuate changes between adjacent latents, whereas untied windows learn three independent weights. [Table 9, Eq.16; Fig.15 rollouts are in part 1](https://arxiv.org/pdf/2610.15267v1#page=32)

**Goals and compute.** Of Figure 16's 50 goals, 46 (92%) satisfy $|\omega^*|\ge0.25$ rad/s. Arriving at the same speed with opposite sign gives error $2|\omega^*|$, reaching the 0.5 rad/s tolerance. Goal angles are not concentrated at upright rest. At matched replanning settings, a predictor step per candidate costs 13.6 MFLOP for LeHami versus 21.7 MFLOP for LeWM, a ratio near 0.63; over 99% of planning arithmetic goes to predictor rollout. [Fig.16, E.3 Planning compute](https://arxiv.org/pdf/2610.15267v1#page=33)

Table 10 separately optimizes speed-error and sign-agreement success under the budget. Multiplying these columns does not recover main-table velocity success.

| Model | Speed success | TFLOP, $R$ | Sign success | TFLOP, $R$ |
| --- | --- | --- | --- | --- |
| LeWM $W=1$ | 0.251±0.051 | 8.78, 15 | 0.656±0.017 | 5.85, 30 |
| LeWM $W=3$, tied | 0.638±0.008 | 13.65, 9 | 0.649±0.022 | 13.65, 9 |
| LeWM $W=3$, untied | **0.813±0.052** | 13.65, 9 | **0.962±0.017** | 13.65, 9 |
| LeHami | 0.547±0.022 | 10.98, 6 | 0.800±0.034 | 8.05, 9 |
| Passive coast | 0.300 | — | 0.680 | — |

Only LeHami and untied LeWM beat passive coasting on sign agreement. The authors identify positional reaching as LeHami's joint-success bottleneck at fixed $R=9$: $P(\theta)P(\omega\mid\theta)$ is approximately $0.48\times0.55=0.26$, versus $0.63\times0.61=0.39$ for tied LeWM. This fixed-$R$ comparison differs from the **best-per-model settings** in part 1's Table 2. [Table 10, E.3](https://arxiv.org/pdf/2610.15267v1#page=34)

Finally, Table 11 checks whether inability to identify length or mass explains planning failures. Probes are fitted at anchors of 630 training episodes and frozen over 135 test episodes. All anchor $R^2$ values are at least 0.999. After 76 rollout steps:

| Probe | Model | Length $R^2$ | Mass $R^2$ |
| --- | --- | --- | --- |
| ridge | LeHami | 0.900 [0.871, 0.951] | 0.977 [0.958, 0.993] |
| ridge | LeWM | 0.983 [0.976, 0.990] | 0.993 |
| 25-NN | LeHami | 0.997 [0.992, 1.000] | 1.000 |
| 25-NN | LeWM | 1.000 | 1.000 |

Brackets retain source-reported seed ranges. Fixed linear readout drifts more for LeHami, while local nonlinear readout continues to recover both parameters. Rounding ridge mass estimates to the nearest of the three physical masses gives 99.3% for both models, versus a 34.8% majority-class floor. The authors conclude that missing mass information does not explain the planning difference. [Table 11, E.3](https://arxiv.org/pdf/2610.15267v1#page=35)

## Review interpretation and open questions

**Review interpretation.** The central shear-composition proof is clear. Claims of a “complete physical state” or structurally guaranteed signed velocity are stronger than that theorem. Their strongest evidence is velocity readout in Table 1 and Figure 8, plus the symmetry analysis in Table 9. Appendix C.2 does not assume physical-coordinate identity, and C.4 explicitly identifies configuration-dependent mass as a limitation. The supported distinction is therefore **structurally guaranteed symplectic latent transitions, experimentally demonstrated useful physical-state readout**. Table 8's joint-regularization failure shows weakened motion information within the same symplectic architecture.

**Review interpretation.** Statements deriving exact energy conservation from symplecticity need correction. Figures 9–11 and E.3's final discussion describe conservation, but C.4's Eq.10 already retains an integration residual $r_t$. Generic leapfrog preserves symplectic structure without preserving the original $H$ exactly at each step. An **illustrative counterexample** within Proposition 2's smooth, separable assumptions uses $H=(q^2+p^2)/2$, $h=1$, and initial $(q,p)=(1,0)$. One step yields $(1/2,-3/4)$, changing energy from $1/2$ to $13/32$. The visualizations therefore establish empirical energy structure, while the volume-preservation theorem remains intact.

**Review interpretation.** Figure 4's “cumulative log-volume” also has a measurement-definition distinction. Section 3.2 describes the full-transition determinant, whereas the released plotting notebook accumulates LeWM's `logdet_per_dim_newest`. Appendix E.2 explicitly distinguishes this block from full volume, and Figure 13 separately confirms contraction of the full 192-dimensional context. Figure 13 is consequently the appropriate full-volume evidence. This does not reverse the baseline-contraction observation, but describing Figure 4 alone as measuring full rollout volume overstates its scope.

**Review interpretation.** The action condition matters when connecting the theorem to control. Composition along a prescribed action sequence is symplectic, but a state-dependent feedback policy contributes an extra derivative. For $G(\mathbf z)=\Psi(\mathbf z,\pi(\mathbf z))$, $DG=D_{\mathbf z}\Psi+D_{\mathbf u}\Psi D\pi$. A useful next test would isolate the geometric contribution using equal temporal information and goal representations, additional physical systems, and closed-loop planning. Table 4's compute advantage and Figure 13's diagnostic each use one training seed; the planning tables select the best configuration on test goals, motivating independent configuration selection and uncertainty checks.

## Sources and reading scope

- Paper: Yalovetzky, Balestriero, LeCun, *LeHami: Symplectic Latent Dynamics for World Models*, first public date 2026-10-11, arXiv v1, reviewed 2026-10-26. [Official source](https://arxiv.org/abs/2610.15267v1).
- Scope: Appendices B–E in full and their correspondence to main-text claims and experiments, using the figures, tables, and equations of the PDF and author LaTeX. [Part 1](/en/reviews/lehami-symplectic-world-models/) covers §§1–4.
- Additional sources: [SRNN v2 §§3.1–3.4](https://arxiv.org/html/1909.13334v2#S3) supplies background on separable Hamiltonians, leapfrog, and multi-step learning. The public [LeHami code](https://github.com/rominay/LeHami/tree/f76f65f8b99afb6b7e3063f573645d3e2f290ede), specifically `paper_figures/paper_figures.ipynb` and stored planning CSVs, establishes Jacobian definitions and aggregation settings.
- Figures: LeHami v1 Figures 8 and 13, with Figure 1 as the thumbnail, Yalovetzky et al., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). PNG downsampling or PDF rasterization followed by WebP encoding; content and labels are retained.
- Not verified: formal proof verification, model retraining, and independent simulator-based performance reproduction.
