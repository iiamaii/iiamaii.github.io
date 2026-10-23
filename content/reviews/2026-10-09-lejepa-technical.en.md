---
title: "[3/3] LeJEPA technical review: proof conditions, SIGReg and checked equations"
description: "Detailed theorem and proof commentary following Sections 3–5 and Appendices A–B, with intermediate derivations and unresolved gaps."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-23T15:26:16+00:00"
topics: ["self-supervised learning", "representation learning", "joint embedding predictive architectures", "distribution matching", "computer vision"]
visibility: "public"
lang: "en"
translationKey: "lejepa-technical"
paperTitle: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
paperPublishedDate: "2025-11-11"
authors: ["Randall Balestriero", "Yann LeCun"]
year: "2025"
paperUrl: "https://arxiv.org/abs/2511.08544v3"
thumbnail: "/assets/reviews/lejepa/lejepa-core.svg"
thumbnailAlt: "LeJEPA combines within-image alignment to the global-view mean with across-image Gaussian matching through random projections in each view."
---

LeJEPA’s theory links isotropic covariance in linear estimation, Fisher information in local nonlinear estimation, and Sketched Isotropic Gaussian Regularization (SIGReg) computed through finitely many projections. The central result is a lower bound: for a density $p$ with zero mean and covariance $\Sigma$, the Fisher information $J(p)=\mathbb E\|\nabla\log p\|^2$ satisfies $J(p)\ge\operatorname{tr}(\Sigma^{-1})$, with equality only for a Gaussian, and fixing the total variance leaves the isotropic Gaussian. [Section 3.2 and Appendix B.4](https://arxiv.org/pdf/2511.08544v3#page=28) **Review interpretation.** The bound is correct, but the Gaussian’s “unique optimality” in Theorem 1 concerns the $J(p)$ term rather than exact bias, and on the implementation side the derivations omit the within-global-view variance in Eqs. (5)→(6) and misplace a factor of $N$ in Theorems 4 and 6.

This **[3/3] theorem and proof commentary** follows the theorems of Sections 3→4→5 and the proofs in Appendices A–B; the problem setting and the Section 6 experiments are in [\[1/3\] introduction](/en/reviews/lejepa-overview/) and [\[2/3\] complete commentary](/en/reviews/lejepa-complete/).

**Notation correspondence.** $\mathbf Z\in\mathbb R^{N\times K}$ is the embedding matrix; $\mathbf z_{n,v}$ indexes image/view embeddings. The linear appendix’s $\mathbf X$, $p$, and $\lambda_{\rm wd}$ correspond to main-text $\mathbf Z$, $K$, and the ridge coefficient. Density $p$ elsewhere has a different role from that dimension symbol. Appendix $d,n$ denote dimension/sample count. The direction count $M=|\mathcal A|$ and $\widehat D=EP/N$, the paper’s Epps–Pulley statistic $EP$ divided by $N$, are explanatory abbreviations.

<span id="3-latent-euclidean-theory-behind-the-target-law" aria-hidden="true"></span>

## 3. Latent Euclidean — Embeddings Should Be Isotropic Gaussian

### 3.1 Linear Probing — Lemmas 1–2 and Appendices B.1–B.2

Under fixed design, let $\mathbf y=\mathbf Z\boldsymbol\beta_{\rm true}+\boldsymbol\varepsilon$, with $\mathbb E[\boldsymbol\varepsilon\mid\mathbf Z]=0$ and $\operatorname{Cov}(\boldsymbol\varepsilon\mid\mathbf Z)=\sigma^2I$. Assume full column rank. Compare embeddings with the same column span and total energy. The results concern coefficient estimation.

**Lemma 1.** Under positive ridge regularization, there exists a task with greater bias for anisotropic embeddings. Differentiating the objective and solving the normal equations gives:

$$
\hat{\boldsymbol\beta}
=(\mathbf Z^\top\mathbf Z+\lambda_{\rm wd}I)^{-1}\mathbf Z^\top\mathbf y.
$$

Taking conditional expectation removes the noise. Define the review shorthand $\mathbf G=\mathbf Z^\top\mathbf Z$:

$$
\begin{aligned}
\mathbb E[\hat{\boldsymbol\beta}\mid\mathbf Z]-\boldsymbol\beta_{\rm true}
&=\left[(\mathbf G+\lambda_{\rm wd}I)^{-1}\mathbf G-I\right]\boldsymbol\beta_{\rm true}\\
&=-\lambda_{\rm wd}(\mathbf G+\lambda_{\rm wd}I)^{-1}\boldsymbol\beta_{\rm true}.
\end{aligned}
$$

The intermediate step substitutes $\mathbf G=(\mathbf G+\lambda_{\rm wd}I)-\lambda_{\rm wd}I$. With $\mathbf G=\mathbf Q\boldsymbol\Lambda\mathbf Q^\top$, each eigendirection’s bias scales by $\lambda_{\rm wd}/(\lambda_k+\lambda_{\rm wd})$. Let $\bar\lambda=K^{-1}\sum_k\lambda_k$ and choose the true task along the smallest-eigenvalue direction:

$$
\frac{\lambda_{\rm wd}}{\lambda_{\min}+\lambda_{\rm wd}}
>\frac{\lambda_{\rm wd}}{\bar\lambda+\lambda_{\rm wd}}.
$$

Along the largest-eigenvalue direction the inequality reverses, which is why the lemma is stated existentially. Appendix B.1’s smallest-eigenvalue indexing varies, but its argument uses the minimum direction above. [Lemma 1; Appendix B.1, p. 26](https://arxiv.org/pdf/2511.08544v3#page=26)

**Lemma 2.** With zero ridge, isotropy minimizes the trace of coefficient variance. Since $\hat{\boldsymbol\beta}-\boldsymbol\beta_{\rm true}=\mathbf G^{-1}\mathbf Z^\top\boldsymbol\varepsilon$:

$$
\begin{aligned}
\operatorname{Cov}(\hat{\boldsymbol\beta}\mid\mathbf Z)
&=\mathbf G^{-1}\mathbf Z^\top(\sigma^2I)\mathbf Z\mathbf G^{-1}
=\sigma^2\mathbf G^{-1},\\
\operatorname{tr}\operatorname{Cov}(\hat{\boldsymbol\beta}\mid\mathbf Z)
&=\sigma^2\sum_{k=1}^K\frac1{\lambda_k}
\ge\sigma^2\frac K{\bar\lambda}.
\end{aligned}
$$

The final step uses strict convexity of $1/x$ on positive values and Jensen’s inequality. Equality requires equal eigenvalues. At a fixed test embedding $\mathbf q$, prediction variance is $\sigma^2\mathbf q^\top\mathbf G^{-1}\mathbf q$; minimizing coefficient trace and minimizing risk at every query are different statements. Linear theory selects covariance shape, not a complete Gaussian law. [Lemma 2; Appendix B.2, p. 27](https://arxiv.org/pdf/2511.08544v3#page=27)

### 3.2 Nonlinear Probing — Theorem 1 and Appendices A, B.3–B.7

**Read the assumptions first.** The density $p_z\in C^3$ has bounded required derivatives, finite Fisher information, and finite covariance. The target $\eta\in C^2$ is queried under the training density. The radius/bandwidth is small, with enough local samples to estimate an average. Kernels are normalized, symmetric, isotropic in their second moment, and have finite roughness. Integration by parts needs tails that eliminate boundary terms. [Appendix A, pp. 24–26](https://arxiv.org/pdf/2511.08544v3#page=24)

**Lemma 4 — kNN pointwise bias: why the density score appears.** Appendix B.3 approximates the neighborhood mean by a ratio of density integrals. This is a local surrogate with PPP conditions, setting aside empty-neighborhood and finite-sample ratio errors. Write local displacement as $\mathbf u$:

$$
\mathbb E[\widehat{\mathbf y}(\mathbf q)]\simeq
\frac{\int_{\|\mathbf u\|\le r_0}\eta(\mathbf q+\mathbf u)p(\mathbf q+\mathbf u)\,d\mathbf u}
{\int_{\|\mathbf u\|\le r_0}p(\mathbf q+\mathbf u)\,d\mathbf u}.
$$

**Review-supplied derivation.** Let $v_K$ be the unit-ball volume. Symmetry eliminates the first integral and yields the second, which corresponds to the source’s volume shorthand:

$$
\int_{B(0,r_0)}\mathbf u\,d\mathbf u=0,\qquad
\int_{B(0,r_0)}\mathbf u\mathbf u^\top d\mathbf u
=\frac{v_Kr_0^{K+2}}{K+2}I.
$$

The denominator’s second-order term contains $\Delta p$; the numerator contains $\Delta(\eta p)=p\Delta\eta+\eta\Delta p+2\nabla\eta^\top\nabla p$. In the ratio, $\eta\Delta p$ cancels:

$$
\mathrm{Bias}(\mathbf q)=\frac{r_0^2}{K+2}
\left(\nabla\eta(\mathbf q)^\top\nabla\log p(\mathbf q)
+\frac12\Delta\eta(\mathbf q)\right)+o(r_0^2).
$$

Because $\eta\in C^2$ alone does not bound a third-order Taylor remainder, the remainder is written as $o(r_0^2)$ instead of the $O(\|\mathbf u\|^3)$ in some of the source’s intermediate lines. [Lemma 4; Appendix B.3, p. 27](https://arxiv.org/pdf/2511.08544v3#page=27)

**Theorem 7 connection.** Appendix B.4 abbreviates $A=\nabla\eta^\top\nabla\log p$ and $C=\Delta\eta/2$. Squaring produces all three terms: $\mathbb E[A^2]+2\mathbb E[AC]+\mathbb E[C^2]$. A mean-zero task-gradient prior with pointwise second moment $\tau_g^2I$ gives:

$$
\mathbb E_{\mathbf z,\eta}[A(\mathbf z)^2]
=\mathbb E_{\mathbf z}\left[
\nabla\log p(\mathbf z)^\top(\tau_g^2I)\nabla\log p(\mathbf z)\right]
=\tau_g^2J(p).
$$

The cross term vanishes only under the additional condition that $\nabla\eta$ and $\Delta\eta$ are uncorrelated. The authors acknowledge that without it the cross term remains at $O(r_0^4)$, and argue that the leading $p$-dependent term is still $\tau_g^2J(p)$. The curvature term is independent of density when the pointwise task second moment is constant. [Theorem 7; Appendix B.4, pp. 28–29](https://arxiv.org/pdf/2511.08544v3#page=28) **Review interpretation.** The cross term $\mathbb E[AC]$ also contains $\nabla\log p$, so it is a $p$-dependent term of the same order; without the uncorrelatedness condition, this formula does not yield a unique minimum of ISB.

**Lemma 6 and Theorem 9 — A Fisher-information lower bound.** The paper invokes the matrix Cramér–Rao bound for a location family. With the necessary regularity, zero mean, and covariance $\Sigma\succ0$, the score $\mathbf s(\mathbf z)=\nabla\log p(\mathbf z)$ satisfies $J(p)\ge\operatorname{tr}(\Sigma^{-1})$, with Gaussian equality.

**Review-supplied derivation of this Cramér–Rao case.** Integration by parts gives $\mathbb E[\mathbf s(\mathbf z)\mathbf z^\top]=-I$, provided the boundary terms vanish. Hence:

$$
\begin{aligned}
0&\preceq\mathbb E\left[
(\mathbf s+\Sigma^{-1}\mathbf z)(\mathbf s+\Sigma^{-1}\mathbf z)^\top\right]\\
&=\mathbb E[\mathbf s\mathbf s^\top]-\Sigma^{-1}.
\end{aligned}
$$

Taking traces gives the bound. Equality makes the residual almost surely zero, so $\nabla\log p=-\Sigma^{-1}\mathbf z$. Under regularity, integration gives $p\propto\exp(-\mathbf z^\top\Sigma^{-1}\mathbf z/2)$. Fixing total variance $\operatorname{tr}\Sigma=c$ then gives:

$$
J(p)\ge\sum_{k=1}^K\frac1{\lambda_k}
\ge\frac{K^2}{c},\qquad
\Sigma=\frac cK I\ \text{at equality}.
$$

This is uniqueness for the Fisher information $J(p)$. Appendix B.4 treats determinant, Frobenius, and spectral-radius constraints with the same argument. Although the source boxes are titled “Recovery of VCReg,” the result used in Lemma 6 and Theorem 9 is this Fisher bound. [Lemma 6, Theorem 9; Appendix B.4, pp. 30–32](https://arxiv.org/pdf/2511.08544v3#page=30)

**Lemma 5 — Kernel bias and variance.** Change kernel variables in the numerator/denominator expectations and expand to second order. Symmetry removes first-order terms; the isotropic kernel moment is $\int\mathbf u\mathbf u^\top K(\mathbf u)d\mathbf u=\mu_2(K)I$. The intermediate ratio identity is:

$$
\frac{a_0+h^2a_2+o(h^2)}{b_0+h^2b_2+o(h^2)}
=\frac{a_0}{b_0}+h^2\frac{a_2b_0-a_0b_2}{b_0^2}+o(h^2).
$$

Substitute $a_0=\eta p$, $b_0=p>0$:

$$
\begin{aligned}
\mathrm{Bias}(\mathbf q)&=\frac{h^2\mu_2(K)}2
\left(\Delta\eta(\mathbf q)+2\nabla\eta(\mathbf q)^\top\nabla\log p(\mathbf q)\right)+o(h^2),\\
\mathrm{Var}(\mathbf q)&=\frac{R(K)}{nh^d}\frac{v(\mathbf q)}{p(\mathbf q)}
+o((nh^d)^{-1}).
\end{aligned}
$$

Here $d=K$, $n=N$, $R(K)=\int K^2$, and $v(\mathbf q)=\operatorname{Var}(Y\mid\mathbf q)$. At finite sample size, a ratio of expectations is not the expectation of a ratio. Local-sample/asymptotic conditions must control denominator fluctuations. The source’s intermediate equality $\operatorname{Var}(K_hY)=\mathbb E[K_h^2v]$ is also not generally exact. **Review supplement.** A delta-method expansion of the ratio residual centered by $Y-\eta(\mathbf q)$ yields the leading conditional-noise term; control of all finite-sample remainders remains an open gap. [Lemma 5; Appendix B.5, pp. 32–33](https://arxiv.org/pdf/2511.08544v3#page=32)

**Theorem 8 — What is minimized?** With $\|\nabla\eta\|\le L$ and $|\Delta\eta|\le B$, apply $(a+b)^2\le2a^2+2b^2$, then Cauchy–Schwarz:

$$
\begin{aligned}
\mathrm{ISB}_{\rm kernel}
&\le\left(\frac{h^2\mu_2(K)}2\right)^2
\left(2\int(\Delta\eta)^2p+8\int(\nabla\eta^\top\nabla\log p)^2p\right)+o(h^4)\\
&\le\left(\frac{h^2\mu_2(K)}2\right)^2(2B^2+8L^2J(p))+o(h^4).
\end{aligned}
$$

The Gaussian minimizes this $J(p)$-based upper bound. In integrated variance, the leading density factor cancels against query weighting, but finiteness of $\int v(\mathbf z)d\mathbf z$, the support, and uniform remainder control still matter. The Appendix A Theorem 8 box omits a square in its notation; main-text Theorem 1 and B.7 derive squared bias. [Theorem 8; Appendix B.7, p. 34](https://arxiv.org/pdf/2511.08544v3#page=34) **Review interpretation.** The authors’ conclusion summarizes this as minimizing worst-case downstream risk, but no proof shows that the bound is attained by the worst case in the class $\|\nabla\eta\|\le L$, $|\Delta\eta|\le B$, so what is minimized is the bound.

### 3.3 Geometric and Practical Insights

Figure 3 illustrates isotropy through repeated logistic boundaries. Figure 17 plots the distribution of $\beta$ obtained by resampling training data for binary classification, linear regression, and nonlinear regression targets; citing Lemmas 1–2, its caption states that “the anisotropic version (blue) provides much lower variance compared to the isotropic case (red).” Lemma 2 says that isotropy gives lower total coefficient variance in unregularized OLS with the same column space and total energy. [Section 3.1, Lemma 2](https://arxiv.org/pdf/2511.08544v3#page=5), [Figure 17](https://arxiv.org/pdf/2511.08544v3#page=48)

**Review interpretation.** The caption has most likely swapped the two cases or their colors, but the paper does not state whether the figure’s target design and noise match Lemma 2’s assumptions, so confidence is moderate. Either way, Lemma 2’s algebraic proof is unaffected.

<span id="4-sigreg-identification-stability-and-finite-direction-conditions" aria-hidden="true"></span>

## 4. SIGReg — Reliable Isotropic Gaussian Regularization in High-Dimension

### 4.1 Hypothesis Testing as a Judge — Lemma 3, Theorem 2

**Lemma 3 and Appendix B.8’s Theorem 10, Cramér–Wold.** Assume projection laws agree for every unit direction. Any $\mathbf t\ne0$ can be written as $\mathbf t=r\mathbf a$, giving:

$$
\varphi_X(\mathbf t)=\mathbb E[e^{ir\mathbf a^\top X}]
=\mathbb E[e^{ir\mathbf a^\top Y}]=\varphi_Y(\mathbf t).
$$

At $\mathbf t=0$, both equal 1. Since the characteristic functions (CFs) agree at every vector frequency, CF uniqueness gives $X\overset d=Y$. The weak-convergence argument connects projection convergence with Lévy’s continuity theorem. [Lemma 3; Appendix B.8, p. 35](https://arxiv.org/pdf/2511.08544v3#page=35)

**Theorem 2’s proof path.** The combined test uses a maximum over directional statistics and a threshold for the global null. Reusing each individual significance level does not by itself control the global false-positive rate. For example, Bonferroni assigns $\alpha/|\mathcal A|$ per direction, letting a union bound control the total at $\alpha$. This example is a review supplement.

For power, the argument uses a neighborhood $U$ of a separating direction $\mathbf a^\star$ and a vanishing mesh, ensuring a sampled point eventually lies in $U$. Applying this to moving directions $\mathbf a_n$ needs sufficiently uniform power over $U$. B.9’s $\inf_a\lim_n$ notation alone does not directly imply $\lim_n\inf_a$; rigorous justification of this exchange remains unconfirmed. [Theorem 2; Appendix B.9, p. 36](https://arxiv.org/pdf/2511.08544v3#page=36)

### 4.2 SIGReg: Sketching the Epps–Pulley Test

#### 4.2.1 Moments — Theorem 3 and Appendix B.11

The finite moment order $K$ in Theorem 3 is distinct from embedding dimension. Place masses on distinct real points $x_0,\ldots,x_{K+1}$. The matrix $A_{rj}=x_j^r$ for $r=0,\ldots,K$ has shape $(K+1)\times(K+2)$, so its kernel contains a nonzero $\mathbf v$. The zeroth row implies $\sum_jv_j=0$.

For a positive probability vector $\mathbf p$ and sufficiently small $\varepsilon$, $\mathbf p^\pm=\mathbf p\pm\varepsilon\mathbf v$ are different valid probability vectors, yet $A\mathbf p^+=A\mathbf p^-$. Thus matching moments through order $K$ need not match distributions. Higher moments can also produce sample-gradient polynomials such as $x_i^{k-1}$ that lack global boundedness. [Theorem 3; Appendices B.11–B.12, p. 37](https://arxiv.org/pdf/2511.08544v3#page=37)

![Figure 5: Gaussian marginals can conceal an X-shaped joint law.](/assets/reviews/lejepa/paper-figure-5.webp)

**Figure 5.** The paper’s example projecting an X-shaped law, with standard Gaussian marginals and identity covariance, onto ten directions. A reading guide is in the [complete version, Section 4.2.1](/en/reviews/lejepa-complete/#421-moments-are-unstable-and-insufficient). [Enlarge](/assets/reviews/lejepa/paper-figure-5.webp) · [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

#### 4.2.2 CDF — Computational requirements and alternatives

The selection criteria include gradient stability and distributed (DDP) computation as well as identification. The sorting and sample-gathering burden of cumulative distribution function (CDF) tests is discussed in the [complete version, Section 4.2.2](/en/reviews/lejepa-complete/#422-cumulative-density-functions-are-impractical). Appendix E gives the evidence that the Shapiro–Wilk test was unstable in practice, and Appendix F covers multivariate statistics such as multivariate characteristic-function comparison. [Section 4.2.2; Appendices E–F](https://arxiv.org/pdf/2511.08544v3#page=44)

#### 4.2.3 CF — Deriving Theorem 4 and checking normalization

Retain Appendix B.12’s Gaussian-window notation. The projected scalar is $x_i=\mathbf a^\top\mathbf z_i$:

$$
\widehat\varphi_N(t)=\frac1N\sum_{i=1}^Ne^{itx_i},\quad
\varphi_G(t)=e^{-t^2/2},\quad w_s(t)=e^{-s^2t^2},
\qquad
\widehat D=\int w_s(t)|\widehat\varphi_N(t)-\varphi_G(t)|^2dt.
$$

Unlike the main-text Epps–Pulley equation, $EP=N\widehat D$, this $\widehat D$ has no prefactor $N$. Since $|\widehat\varphi_N|,|\varphi_G|\le1$ and $\partial_{x_i}\widehat\varphi_N=it e^{itx_i}/N$:

$$
\begin{aligned}
|\partial_{x_i}\widehat D|
&\le\frac4N\int w_s(t)|t|dt=\frac4{Ns^2},\\
|\partial_{x_i}^2\widehat D|
&\le\left(\frac2{N^2}+\frac4N\right)\int w_s(t)t^2dt
\le\frac{6\sqrt\pi}{2Ns^3}.
\end{aligned}
$$

**Review-supplied derivation.** The first bound uses error magnitude at most 2. The second derivative contains $2|\partial\widehat\varphi_N|^2$ and $2\operatorname{Re}[(\widehat\varphi_N-\varphi_G)\overline{\partial^2\widehat\varphi_N}]$. Gaussian-weighted integrals satisfy $\int|t|w_s=1/s^2$ and $\int t^2w_s=\sqrt\pi/(2s^3)$, producing input-independent bounds. These integrals also provide domination for differentiation under the integral.

Main-text Theorem 4 is printed as $|\partial EP/\partial z_i|\le4\sigma^2/N$ and $|\partial^2EP/\partial z_i^2|\le C\sqrt\pi\sigma^3/(2N)$, matching the expressions above through $s=1/\sigma$. Parameter gradients multiply these by an encoder Jacobian factor of the form $\|\mathbf a^\top\partial_{\boldsymbol\theta}f_{\boldsymbol\theta}\|$. [Theorem 4; Appendix B.12, pp. 37–38](https://arxiv.org/pdf/2511.08544v3#page=37) **Review interpretation.** The derivation above is for the unscaled $\widehat D$, so the correct bound for $EP=N\widehat D$ is $|\partial EP|\le4/s^2=4\sigma^2$. Boundedness survives and only the $1/N$ decay disappears, so Theorem 4’s stability conclusion is unchanged; confidence is high.

Algorithm 1’s CF-mean all-reduce relies on the fact that, with equally sized device batches, the average of local CF means equals the global CF mean (review supplement). The algorithm’s steps and cost are in the [complete version, Section 4.2.3](/en/reviews/lejepa-complete/#423-characteristic-functions-are-stable-scalable-and-identifiable).

**Review example: finite-frequency identification limits.** The 17-point grid has $\Delta t=0.625$. Translate $X\sim\mathcal N(0,1)$ by $c=2\pi/\Delta t$ to obtain $Y=X+c$. At every grid point, $e^{itc}=1$, so both laws have the same CF values, while off-grid frequencies distinguish them. This 1D example shows that fixed-grid quadrature can miss a difference that the exact integral identifies.

### 4.3 How SIGReg Beats the Curse of Dimensionality — Theorem 5

Retaining the paper’s rate:

$$
\mathbb E_{\mathbf a}\int_{\mathbb R}
|\varphi_{\mathbf a}(t)-\varphi_{\mathcal N}(t)|^2dt
\le C(K,\alpha)M^{-2\alpha/(K-1)}
\int_0^\infty\|\varphi_{\cdot}(r)-\varphi_{\mathcal N}(r)\|_{H^\alpha(\mathcal S^{K-1})}^2dr.
$$

The paper assumes $p_{\boldsymbol\theta}\in H^\alpha(\mathbb R^K)$ and equality of the complete projected laws in the selected directions. **Appendix B.10’s route:** spherical-harmonic approximation through degree $L$ has error roughly $L^{-\alpha}$; $M$ quasi-uniform directions correspond to $L\sim M^{1/(K-1)}$. Squaring yields $M^{-2\alpha/(K-1)}$, followed by integration over frequency. Marcinkiewicz–Zygmund conditions support stable recovery of the harmonic space from sampled values.

**Unconfirmed gap.** The paper leaves to cited sources the route from Euclidean Sobolev regularity to finite angular CF norms and frequency integrals, and the stability conditions for point placement and interpolation; whether those conditions hold in this setting is not established within the paper. From this bound the authors conclude that $|\mathcal A|=O(K)$ directions suffice when $\alpha$ is large. [Theorem 5; Appendix B.10, pp. 36–37](https://arxiv.org/pdf/2511.08544v3#page=36) **Review interpretation.** The paper’s $C(K,\alpha)$ contains a factor $\pi^{(K-1)/2}$ that grows quickly with $K$, and with $M=O(K)$ the factor $M^{-2\alpha/(K-1)}$ approaches 1 for large $K$, so that conclusion needs $\alpha$ to grow with $K$ or an additional assumption on the size of the norm.

![Figure 7: Synthetic discrepancy with fixed versus step-resampled directions.](/assets/reviews/lejepa/paper-figure-7.webp)

**Figure 7.** A synthetic comparison of fixed directions (blue) and directions resampled at every step (green). A reading guide is in the [complete version, Section 4.3](/en/reviews/lejepa-complete/#43-how-sigreg-beats-the-curse-of-dimensionality). [Enlarge](/assets/reviews/lejepa/paper-figure-7.webp) · [Figure 7, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10). Balestriero and LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); full-panel crop and lossless WebP conversion.

<span id="5-lejepa-sampling-and-normalization-in-the-actual-objective" aria-hidden="true"></span>

## 5. LeJEPA — Stable and Scalable Implementation

### 5.1 SIGReg + Prediction Loss — Theorem 6, Appendices B.6 and B.13

**Theorem 6: the origin of sampling bias.** For one projection/frequency, define review shorthand $U_j=e^{itx_j}$. Independent samples give expectation 1 for diagonal pairs and $|\varphi_P|^2$ for off-diagonal pairs:

$$
\begin{aligned}
\mathbb E|\widehat\varphi_N|^2
&=\frac{N+N(N-1)|\varphi_P|^2}{N^2}
=|\varphi_P|^2+\frac{1-|\varphi_P|^2}N,\\
\mathbb E|\widehat\varphi_N-\varphi_G|^2
&=|\varphi_P-\varphi_G|^2+\frac{1-|\varphi_P|^2}N.
\end{aligned}
$$

After weighted integration, $\mathbb E\widehat D=D+N^{-1}\int w_s(1-|\varphi_P|^2)$. With domination permitting differentiation and expectation to commute, parameter-gradient bias is $-N^{-1}\int w_s\nabla_{\boldsymbol\theta}|\varphi_P|^2$. This $O(1/N)$ is the absolute bias of the unscaled $\widehat D$ relative to its population value $D$. The authors report that the bias was small and not a concern even for minibatches of 16, and leave unbiased alternatives such as U-statistics or sample splitting unexplored. [Theorem 6; Appendix B.13, pp. 38–39](https://arxiv.org/pdf/2511.08544v3#page=38)

**Review interpretation.** Theorem 6 states this identity for the expectation of $EP$, but with $EP=N\widehat D$ as in Algorithm 1 the additional term becomes $\int w_s(1-|\varphi_P|^2)$, which is $O(1)$. The baseline loss also scales by $N$, so the relative bias is still $O(1/N)$, but the implemented loss and gradient do not carry an absolute $O(1/N)$ bias. Deterministic quadrature keeps the same sampling algebra and adds a separate integration error.

**Eqs. (5)→(6): expand the omitted term.** The paper replaces a global/all-view pair average by a center average. Let $\boldsymbol\mu_n=V_g^{-1}\sum_{g=1}^{V_g}\mathbf z_{n,g}$ and $\boldsymbol\delta_g=\mathbf z_{n,g}-\boldsymbol\mu_n$, so $\sum_g\boldsymbol\delta_g=0$:

$$
\begin{aligned}
\frac1{V_gV}\sum_{g=1}^{V_g}\sum_{v=1}^V
\|\mathbf z_{n,g}-\mathbf z_{n,v}\|^2
&=\frac1V\sum_{v=1}^V\|\boldsymbol\mu_n-\mathbf z_{n,v}\|^2\\
&\quad+\frac1{V_g}\sum_{g=1}^{V_g}\|\mathbf z_{n,g}-\boldsymbol\mu_n\|^2.
\end{aligned}
$$

**Review-supplied derivation.** Write each difference as $\boldsymbol\delta_g+(\boldsymbol\mu_n-\mathbf z_{n,v})$ and square it. The cross term vanishes because the deviations sum to zero; both squared terms remain. B.6’s final step equates the mean square with the squared mean and omits this variance. The additional term is zero only when all global embeddings agree. For scalar views $[0,2]$ and $V_g=V=2$, pairwise loss is 2, center loss is 1, and the extra variance is 1. It depends on learned parameters and is not generally a discardable constant. [Eqs. (5)–(7); Appendix B.6, pp. 33–34](https://arxiv.org/pdf/2511.08544v3#page=33)

**Review interpretation.** Algorithm 2 implements the center loss of Eq. (6) (see the [complete version, Section 5.1](/en/reviews/lejepa-complete/#51-lejepa-sigreg-prediction-loss)), so the reported experimental results are unaffected by this error. What needs correcting is the motivation that the pairwise loss of Eq. (5) and the center loss are the same objective; the expansion above is a general identity, so confidence is high.

**Four sources of error.** Finite-sample bias in Theorem 6, finite-direction approximation, numerical quadrature on the 17-point grid, and finite-encoder optimization or representation error are distinct terms. Increasing $N$ reduces the first while leaving directional and frequency identifiability unresolved, as the alias example shows. [§4.1–4.3 and 5.1, Algorithms 1–2](https://arxiv.org/pdf/2511.08544v3#page=10)

### 5.2 Relation to Prior Work — Appendix B.14

An exact CF integral connects to kernel discrepancy; quadrature does not construct every sample pair. A moment-limited test instead asks that every unit $\mathbf a$ satisfy $\mathbb E[\mathbf a^\top\mathbf z]=0$ and $\operatorname{Var}(\mathbf a^\top\mathbf z)=1$.

Let the mean be $\boldsymbol\mu$. If it were nonzero, selecting $\mathbf a=\boldsymbol\mu/\|\boldsymbol\mu\|$ would contradict the first condition, so $\boldsymbol\mu=0$. The second gives $\mathbf a^\top\Sigma\mathbf a=1$. Taking $\mathbf a=\mathbf e_i$ sets diagonal entries to 1; taking $\mathbf a=(\mathbf e_i+\mathbf e_j)/\sqrt2$ sets off-diagonal entries to 0. Thus $\Sigma=I$.

The paper states that, with the statistic $T=\operatorname{mean}^2+(\operatorname{std}-1)^2$, LeJEPA recovers VICReg in the limit of many slices. The relation to sliced score matching, sliced Wasserstein, and MMD is in the [complete version, Section 5.2](/en/reviews/lejepa-complete/#52-relation-to-prior-work). [Section 5.2; Appendix B.14, pp. 39–40](https://arxiv.org/pdf/2511.08544v3#page=39) **Review interpretation.** The calculation above recovers moment-level constraints, zero mean and identity covariance. VICReg’s standard-deviation hinge penalizes only standard deviations below its target and has no mean-matching term, so the sets of embeddings on which the two regularizers vanish also differ.

## Review interpretation and open questions

**Review interpretation.** The solid parts of the proofs are those that transcribe standard results correctly: the Fisher-information bound (Lemma 6, Theorem 9), Cramér–Wold (Lemma 3), and the moment counterexample (Theorem 3). The weak parts are the links that turn these results into a training objective; the $p$-dependence of the kNN cross term (Section 3.2), the limit exchange in Theorem 2 (Section 4.1), the constant dependence behind the $O(K)$-directions conclusion of Theorem 5 (Section 4.3), and the omitted term in Eqs. (5)→(6) and the normalization in Theorems 4 and 6 (Sections 4.2.3 and 5.1) each need extra assumptions or corrections. Open questions are whether the lower accuracy at batch 128 (72.20) than at 512 (74.72) in Table 1(c) comes from the effective SIGReg weight changing with batch size under the $N$-scaled statistic, and whether resampling frequencies at every step instead of the fixed 17-point grid changes the identification limit in actual training.

## Sources and reading scope

- Paper: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics* · Randall Balestriero et al. · first public date 2025-11-11 · [reviewed version](https://arxiv.org/abs/2511.08544v3) · reviewed 2026-10-23.
- Scope read: Version 3, sections 3–5 and the relevant theorems and proofs in appendix A and B.1–B.14. Reconstructed intermediate steps are identified as review supplements.
- Figure sources: The concept thumbnail is review-authored (CC BY 4.0). Paper figures are Balestriero and LeCun’s v3 assets (CC BY-SA 4.0), with full-panel cropping and lossless WebP conversion. [Paper](https://arxiv.org/abs/2511.08544v3).
- Not verified: research-code execution, model training/reproduction, raw-data uncertainty estimates, and complete formal certification.
