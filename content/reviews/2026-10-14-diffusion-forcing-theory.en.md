---
title: "[2/2] Diffusion Forcing: What Does Learning Every Subsequence Likelihood Mean?"
description: "Follow appendix A's ELBO, Gaussian loss, and subsequence argument through two Jensen steps and intermediate calculations, separating printed inconsistencies from supplementary derivations."
date: "2026-10-14"
publishedAt: "2026-10-14T13:58:27+09:00"
updatedAt: "2026-10-23T15:26:20+00:00"
topics: ["diffusion models", "sequence modeling", "world action model", "robot learning", "probabilistic modeling"]
visibility: "public"
lang: "en"
translationKey: "diffusion-forcing-theory"
paperTitle: "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion"
paperPublishedDate: "2024-07-01"
authors: ["Boyuan Chen", "Diego Marti Monso", "Yilun Du", "Max Simchowitz", "Russ Tedrake", "Vincent Sitzmann"]
year: "2024"
paperUrl: "https://arxiv.org/abs/2407.01392v4"
thumbnail: "/assets/reviews/diffusion-forcing/overview.webp"
thumbnailAlt: "Original Figure 2: independently noised tokens trained together and sampled under different conditioning patterns."
---

Appendix A of Diffusion Forcing (DF) establishes an evidence lower bound (ELBO) on the expected log-likelihood of sequences corrupted with uniformly sampled per-token noise levels (Theorem A.1), and turns it into a weighted denoising MSE for Gaussian diffusion (Corollary A.2). The only inequalities in the proof are two Jensen steps, over latent states and over reverse-diffusion paths, and the authors state that with deterministic latents, equality holds if and only if the variational approximation is exact. [Appendix A, Theorem A.1](https://arxiv.org/pdf/2407.01392v4#page=14) **Review interpretation.** The printed bound has the KL sign and the reverse-transition direction opposite to their definitions, but correcting them yields the same weighted-MSE objective, so the first claim, that training optimizes a reweighted ELBO, stands; the second, that it optimizes the likelihood of every noise pattern simultaneously, rests entirely on the full-expressivity assumption.

Read [\[1/2\] Methods and experiments](/en/reviews/diffusion-forcing-methods/) first. This article follows **appendix A, pp. 14–19**, in the order setup, A.1 main results, A.1.1 Gaussian specialization, A.1.2 subsequences, and A.2 proof, placing the printed formulas side by side with this review's supplementary derivations.

## A. Theoretical Justification: Objects and Assumptions

The clean token is $\mathbf x_t^0$, its level-$k$ version is $\mathbf x_t^k$, sequence length is $T$, and maximum noise level is $K$.

The fixed forward process is a Markov chain, as in equation (A.1):

$$
q(\mathbf x^{1:K}\mid\mathbf x^0)
=\prod_{k=1}^{K}q(\mathbf x^k\mid\mathbf x^{k-1}).
$$

The model assumes a **Markov property**: the latent $\mathbf z_{t-1}$ is a sufficient statistic for the history.

$$
\begin{aligned}
&p_\theta(\mathbf z_t,\mathbf x_t^{k_t}\mid
\mathbf z_{1:t-1},(\mathbf x_s^{k_s})_{s<t})\\
&\qquad=p_\theta(\mathbf z_t,\mathbf x_t^{k_t}\mid
\mathbf z_{t-1}).
\end{aligned}
$$

**Deterministic latents** mean that $p_\theta(\mathbf z_t\mid\mathbf z_{1:t-1},(\mathbf x_s^{k_s})_{s<t},\mathbf x_t^{k_t})$ is a Dirac delta. The RNN implementation in part 1 has latents of this kind.

Remark 1 notes that for $p_\theta$ to be a valid probability model in this case, the latents must be viewed as a collection of variables $\mathbf z_t(k_{1:t})$ indexed by noise history. Using one variable for two conditions with different noise histories would leave that conditional undefined, so $p_\theta$ would not be a joint probability measure. Later notation suppresses this dependence. [Equations (A.1)–(A.3), Remark 1, p. 14](https://arxiv.org/pdf/2407.01392v4#page=14)

## A.1 Main Results: Theorem A.1

Theorem A.1 fixes clean data $\mathbf x_{1:T}^0$, samples per-token noise levels uniformly, corrupts the sequence with the forward process, and considers its **expected log-likelihood**. The statement samples $k_t$ from $[K]$, but the proof uses $\{0,\ldots,K\}$, so this article reads it with the latter convention. $\mathbb E_{\mathrm{forward}}$ averages over noise levels and forward corruption, and $\mathbb E_{p,\mathbf z_{1:T}}$ over latents given those. The bound printed on p. 14 is

$$
\begin{aligned}
&\mathbb E_{\mathrm{forward}}\big[\ln p_\theta((\mathbf x_t^{k_t})_{1\le t\le T})\big]
\ge C(\mathbf x_{1:T}^0)\\
&\quad+\mathbb E_{\mathrm{forward}}\mathbb E_{p,\mathbf z_{1:T}}\Bigg[\sum_{t=1}^{T}\bigg(\frac{1}{K+1}\ln p_\theta(\mathbf x_t^0\mid\mathbf x_t^1,\mathbf z_{t-1})\\
&\qquad+\sum_{j=2}^{K}\frac{j}{K+1}\,\mathrm D_{\mathrm{KL}}\big(q(\mathbf x_t^{j-1}\mid\mathbf x_t^{j},\mathbf x_t^0)\,\Vert\,p_\theta(\mathbf x_t^{j}\mid\mathbf x_t^{j-1},\mathbf z_{t-1})\big)\bigg)\Bigg].
\end{aligned}
$$

The equality statement says that with deterministic latents, equality holds if and only if $q(\mathbf x_t^{k_t+1:T}\mid\mathbf x_t^{k_t})\equiv p_\theta(\mathbf x_t^{k_t+1:T}\mid\mathbf x_t^{k_t},\mathbf z_{t-1})$, although the source prints "holds with inequality". The superscript bound $T$ also reads as a slip for the noise bound $K$.

**Clarification within the source, and correction.** The final display of the proof (p. 19) writes the KL with $p_\theta(\mathbf x_t^{j}\mid\mathbf x_t^{j+1},\mathbf z_{t-1})$, coefficient $\frac{j+1}{K+1}$, and $j=1,\ldots,K-1$, then states that it reindexes $j\gets j+1$. The transition direction $\mathbf x_t^j\mid\mathbf x_t^{j-1}$ in the theorem statement is therefore a reindexing slip, and the coefficient $\frac{j}{K+1}$ agrees with the proof. The sign is not resolved anywhere in the proof: Claim 3 (p. 18) writes the expected log ratio $\mathbb E_q\log(p_\theta/q)$ as $+\mathrm D_{\mathrm{KL}}(q\Vert p_\theta)$, and that sign carries into the theorem. By the definition $\mathbb E_q\log(p_\theta/q)=-\mathrm D_{\mathrm{KL}}(q\Vert p_\theta)$, replacing the KL term as follows gives the form of the standard diffusion ELBO, with $q=q(\mathbf x_t^{j-1}\mid\mathbf x_t^j,\mathbf x_t^0)$:

$$
+\mathrm D_{\mathrm{KL}}\big(q\,\Vert\,p_\theta(\mathbf x_t^{j}\mid\mathbf x_t^{j-1},\mathbf z_{t-1})\big)
\;\longrightarrow\;
-\mathrm D_{\mathrm{KL}}\big(q\,\Vert\,p_\theta(\mathbf x_t^{j-1}\mid\mathbf x_t^{j},\mathbf z_{t-1})\big)
$$

The derivation of this correction is followed in A.2 below. [Theorem A.1 and proof, pp. 14–19](https://arxiv.org/pdf/2407.01392v4#page=14)

**Consequences and confidence.** The sign and direction are fixed by the definitions of KL and of the reverse transition alone, so confidence is high. Raising the corrected bound means reducing the KL, that is, the weighted denoising error, which matches the main-text objective of minimizing noise-prediction MSE; the conclusion that training optimizes a reweighted ELBO stands. The equality conditions come from the proof's two inequalities regardless of the sign. The same sign error carries into Corollary A.2 and Theorem A.3, and A.3's general-distribution weights contain a separate error (A.1.2).

## A.1.1 Specializing to Gaussian Diffusion

Corollary A.2 applies DDPM's standard likelihood and KL computations to Theorem A.1, turning **Gaussian KL into weighted MSE**. The source states it in the $\mathbf x$-prediction form used in the implementation and notes that $\boldsymbol\epsilon$- and $\mathbf v$-prediction are affinely related and can be treated the same way. The result printed on p. 15 is

$$
\begin{aligned}
&\mathbb E_{\mathrm{forward}}\big[\ln p_\theta((\mathbf x_t^{k_t})_{1\le t\le T})\big]+C(\mathbf x_{1:T}^0)\\
&\quad\ge\mathbb E_{\mathrm{forward}}\mathbb E_{p,\mathbf z_{1:T}}\Bigg[\sum_{t=1}^{T}\frac{j}{K+1}\sum_{j=1}^{K}c_j\big\|\hat{\mathbf x}^0_\theta(\mathbf x_t^j,\mathbf z_{t-1},j)-\mathbf x_t^0\big\|^2\Bigg],\\
&\sigma_j^2:=\frac{(1-\alpha_j)(1-\sqrt{\bar\alpha_{j-1}})}{1-\bar\alpha_j},\qquad
c_j=\frac{(1-\alpha_j)^2\bar\alpha_{j-1}}{2\sigma^2(1-\bar\alpha_j)^2}.
\end{aligned}
$$

Here a positive squared error sits on the lower-bound side of the log-likelihood, carrying over Theorem A.1's sign error. In addition, the coefficient $j/(K+1)$ sits outside the sum over $j$; the variance contains $1-\sqrt{\bar\alpha_{j-1}}$ instead of the standard posterior variance's $1-\bar\alpha_{j-1}$; and the denominator of $c_j$ is $\sigma^2$ rather than $\sigma_j^2$. The model transition is defined as $p_\theta(\mathbf x_t^j\mid\mathbf x_t^{j+1},\mathbf z_{t-1})$, while the mean $\boldsymbol\mu_\theta(\mathbf x_t^j,\mathbf z_{t-1},j)$ is written with the coefficients of step $j\to j-1$, so the indices are also off by one. The following **supplementary review derivation** redoes the calculation with the DDPM posterior. For $j\ge2$, the standard forward posterior is

$$
q(\mathbf x_t^{j-1}\mid\mathbf x_t^j,\mathbf x_t^0)
=\mathcal N(\tilde{\boldsymbol\mu}_j,\tilde\beta_j\mathbf I),
$$

$$
\begin{aligned}
\tilde{\boldsymbol\mu}_j
&=A_j\mathbf x_t^j+B_j\mathbf x_t^0,\\
A_j&=\frac{\sqrt{\alpha_j}(1-\bar\alpha_{j-1})}
{1-\bar\alpha_j},\\
B_j&=\frac{(1-\alpha_j)\sqrt{\bar\alpha_{j-1}}}
{1-\bar\alpha_j},\\
\tilde\beta_j&=
\frac{(1-\alpha_j)(1-\bar\alpha_{j-1})}
{1-\bar\alpha_j}.
\end{aligned}
$$

$A_j,B_j$ are review-added abbreviations for coefficients in DDPM equations (6)–(7). Because the cited Chan tutorial v1 has a related variance notation issue, this derivation takes DDPM's posterior variance $\tilde\beta_j$ as its reference. [DDPM, §2–3.2](https://arxiv.org/pdf/2006.11239v2#page=3)

Parameterize the model mean as

$$
\boldsymbol\mu_\theta
=A_j\mathbf x_t^j+
B_j\hat{\mathbf x}_\theta^0(
\mathbf x_t^j,\mathbf z_{t-1},j)
$$

The difference from the known posterior mean is then

$$
\tilde{\boldsymbol\mu}_j-\boldsymbol\mu_\theta
=B_j(\mathbf x_t^0-\hat{\mathbf x}_\theta^0)
$$

When both covariances equal $\sigma_j^2\mathbf I$, Gaussian KL simplifies to

$$
\mathrm D_{\mathrm{KL}}\!
\left(\mathcal N(\boldsymbol\mu_q,\sigma_j^2\mathbf I)
\Vert\mathcal N(\boldsymbol\mu_p,\sigma_j^2\mathbf I)\right)
=\frac{\|\boldsymbol\mu_q-\boldsymbol\mu_p\|^2}
{2\sigma_j^2}.
$$

Thus the KL is $c_j\|\hat{\mathbf x}_\theta^0-\mathbf x_t^0\|^2$, with

$$
c_j=\frac{B_j^2}{2\sigma_j^2}
=\frac{(1-\alpha_j)^2\bar\alpha_{j-1}}
{2\sigma_j^2(1-\bar\alpha_j)^2}.
$$

Read with $\sigma_j^2$ in the denominator, this $c_j$ equals the printed coefficient. Different **fixed** covariances add other terms while retaining the same weighted MSE in the model mean. Step $j=1$ is the clean-data reconstruction (decoder) term and is handled separately, and the forward posterior's zero endpoint variance is not inserted into a division. Because KL is subtracted in the ELBO, writing $w_j$ for the weights from averaging over noise levels, the corrected bound has the form $\ln p_\theta\ge C-\sum_j w_jc_j\|\hat{\mathbf x}_\theta^0-\mathbf x_t^0\|^2$, and raising it means lowering the MSE. [Corollary A.2, p. 15](https://arxiv.org/pdf/2407.01392v4#page=15)

### When may weights be dropped?

In the remarks after Corollary A.2, the authors introduce an idealized latent made of all past tokens and their noise levels, and write that taking the infimum over function approximators makes each term's minimum the conditional variance $\mathbf{Var}_q[\mathbf x_t^0\mid(\mathbf x_s^{k_s})_{s\le t},k_{1:t}]$. With enough expressivity each time step's term can be optimized separately, so the weights do not affect the optimum and can be dropped or rescaled. This argument does not depend on the sign of the bound. [Corollary A.2 remarks, p. 15](https://arxiv.org/pdf/2407.01392v4#page=15)

Under squared error, the optimal predictor is the **conditional mean**. Writing input information as the review-added $\mathcal I$,

$$
\begin{aligned}
\mathbb E\|\mathbf x-f(\mathcal I)\|^2
&=\mathbb E\|\mathbf x-\mathbb E[\mathbf x\mid\mathcal I]\|^2\\
&\quad+\mathbb E\|\mathbb E[\mathbf x\mid\mathcal I]-f(\mathcal I)\|^2.
\end{aligned}
$$

The cross term vanishes because the residual has zero conditional mean. The first term is irreducible conditional uncertainty; the second is the predictor's excess error. For vectors, the first is the expected trace of the conditional covariance, which is what the source's “conditional variance” refers to.

If every conditional mean is representable and positive weights depend on the condition, reweighting preserves these conditional optima. **A finite shared network can behave differently.** For illustration, predicting targets 0 and 2 with one shared scalar $u$ gives

$$
\underset{u}{\arg\min}\,
\{w_1u^2+w_2(u-2)^2\}
=\frac{2w_2}{w_1+w_2}.
$$

Weights 1:1 give 1; weights 1:3 give 1.5. Without adequate expressivity, weights change the optimum. Appendix D.1's report that fused SNR reweighting greatly accelerated convergence in video prediction is consistent with weights not being neutral for a finite model.

## A.1.2 Capturing All Subsequences: Reading Theorem A.3

Theorem A.3 allows an arbitrary distribution $\mathcal D$ over sequences of noise levels and defines

$$
P_t(j\mid k_{1:t-1})
=\Pr_{\mathcal D}(k_t=j\mid k_{1:t-1})
$$

The extension is meant to cover temporally correlated noise patterns, and a fixed pattern is a distribution concentrated on one sequence of levels. The source says the theorem follows from a straightforward modification of the proof of A.1 and gives no separate proof. The per-time-step term of the bound printed on p. 16 is

$$
\begin{aligned}
\Xi_t:=\;&P_t(1\mid k_{1:t-1})\ln p_\theta(\mathbf x_t^0\mid\mathbf x_t^1,\mathbf z_{t-1})\\
&+\sum_{j=2}^{K}jP_t(j\mid k_{1:t-1})\,\mathrm D_{\mathrm{KL}}\big(q(\mathbf x_t^{j-1}\mid\mathbf x_t^j,\mathbf x_t^0)\,\Vert\,p_\theta(\mathbf x_t^{j}\mid\mathbf x_t^{j-1},\mathbf z_{t-1})\big).
\end{aligned}
$$

The KL sign and direction share Theorem A.1's problem, and the weight $jP_t(j\mid\cdot)$ has a separate one. Under the uniform distribution $jP_t(j)=j/(K+1)$, matching A.1, but a general distribution needs cumulative probabilities.

### Why a general distribution requires cumulative probabilities

The A.2 expansion includes reverse steps above the starting noise level $k_t$. For step $j\to j-1$, the KL is included when $k_t\le j-1$, so its coefficient is

$$
\begin{aligned}
&\mathbb E_{\mathcal D}[
\mathbf1\{k_t\le j-1\}\mid k_{1:t-1}]\\
&\quad=\Pr_{\mathcal D}(k_t\le j-1\mid k_{1:t-1})\\
&\quad=\sum_{h=0}^{j-1}P_t(h\mid k_{1:t-1})
\end{aligned}
$$

This is a **supplementary review calculation** that applies A.2's indicator expansion directly to a general distribution.

For uniform noise, the $j$ included levels $0,\ldots,j-1$ give $j/(K+1)$, agreeing with the printed formula. A general distribution differs. For example, if $K=3$ and all mass lies at $k_t=1$, the cumulative weight for $j=2$ is 1, whereas $2P_t(2)=0$. The reconstruction coefficient should likewise be the probability of level zero, but the printed formula has $P_t(1\mid\cdot)$, so it must be corrected together with the endpoint convention. One counterexample settles this, so confidence is high.

**Consequences.** For the fixed patterns (Dirac distributions) the authors consider most important, $P_t(j\mid\cdot)=\mathbf1\{j=k_t\}$. The printed formula then keeps only the KL term $j=k_t$, while the cumulative correction keeps every term with $j\ge k_t+1$. Either way, the terms of the bound are, up to weighting, a subset of the uniform training objective's terms, so the line of conclusion below is unaffected. [Theorem A.3 and A.2, pp. 16–19](https://arxiv.org/pdf/2407.01392v4#page=16)

### What counts as a subsequence?

The source itself states that for a fixed pattern the training objective need not be a lower bound on that pattern's likelihood. Instead, the terms of the bound are, up to reweighting, a subset of the training objective's terms, and by the remark in A.1.1 a sufficiently expressive network can optimize all of them simultaneously; the authors conclude that the training objective is a valid surrogate for maximizing the likelihood of every noise sequence. A pattern with $k_t\in\{0,K\}$ keeps some tokens clean and masks the rest as nearly pure noise, which is the case behind the "all subsequences" claim in main-text §3.2. **Review interpretation.** This conclusion yields exact conditional generation only together with $\mathbf x_t^K$ being approximately pure noise and full expressivity, and the source does not measure how close a finite RNN comes to that idealization.

## A.2 Proof of Theorem A.1

The source uses three claims: expand latents, apply an ELBO over noise paths, then compute the expected bound. The walkthrough keeps this order while checking signs and conditional distributions.

### Claim 1: Expand Latents and Apply Jensen

For a fixed noise history, integrate the model's joint distribution:

$$
\begin{aligned}
p_\theta((\mathbf x_t^{k_t})_{t=1}^{T})
&=\int\prod_{t=1}^{T}
p_\theta(\mathbf z_t\mid\mathbf z_{t-1},\mathbf x_t^{k_t})\\
&\qquad\qquad\cdot
p_\theta(\mathbf x_t^{k_t}\mid\mathbf z_{t-1})
\,\mathrm d\mathbf z_{1:T}\\
&=\mathbb E_{p,\mathbf z_{1:T}}
\left[\prod_{t=1}^{T}
p_\theta(\mathbf x_t^{k_t}\mid\mathbf z_{t-1})\right].
\end{aligned}
$$

Since the logarithm is concave, $\log\mathbb E[Y]\ge\mathbb E[\log Y]$:

$$
\log p_\theta((\mathbf x_t^{k_t})_{t=1}^{T})
\ge
\mathbb E_{p,\mathbf z_{1:T}}
\left[\sum_{t=1}^{T}
\log p_\theta(\mathbf x_t^{k_t}\mid\mathbf z_{t-1})\right].
$$

As Claim 1 states, deterministic latents leave one path in the latent integral, making this step an equality. With stochastic latents, equality also holds if the product is almost surely constant, so determinism is a sufficient condition for equality in this step. [Claim 1, equation (A.9), pp. 16–17](https://arxiv.org/pdf/2407.01392v4#page=16)

### Claim 2: Introduce a Noise Path and Apply Jensen Again

Fix sequence time $t$ and write $h=k_t$, a review-added shorthand. Obtaining the likelihood of $\mathbf x_t^h$ requires integrating more-noised variables $\mathbf x_t^{h+1:K}$. With $r_h=q(\mathbf x_t^{h+1:K}\mid\mathbf x_t^h)$,

$$
\begin{aligned}
\log p_\theta(\mathbf x_t^h\mid\mathbf z_{t-1})
&=\log\mathbb E_{r_h}
\left[
\frac{p_\theta(\mathbf x_t^{h:K}\mid\mathbf z_{t-1})}
{r_h(\mathbf x_t^{h+1:K})}
\right]\\
&\ge\mathbb E_{r_h}
\left[
\log\frac{p_\theta(\mathbf x_t^{h:K}\mid\mathbf z_{t-1})}
{r_h(\mathbf x_t^{h+1:K})}
\right].
\end{aligned}
$$

The proposal's support must cover the model paths and the integrals must exist. Equality holds when the importance ratio is constant, that is, when $r_h$ equals the model posterior $p_\theta(\mathbf x_t^{h+1:K}\mid\mathbf x_t^h,\mathbf z_{t-1})$; this is the source's equality condition $q\equiv p_\theta$.

**Review supplement: the gap in this inequality.** Name the right-hand expectation $\mathcal B_h$ within this paragraph. Under normalization, support, and integrability conditions,

$$
\log p_\theta(\mathbf x_t^h\mid\mathbf z_{t-1})-\mathcal B_h
=\mathrm D_{\mathrm{KL}}\!\left(r_h\,\Vert\,p_\theta(\mathbf x_t^{h+1:K}\mid\mathbf x_t^h,\mathbf z_{t-1})\right)\geq0.
$$

The gap is a KL between path distributions. Deterministic latents make Claim 1 an equality, and an exact variational approximation decides equality in Claim 2. Theorem A.1's "with deterministic latents, equality iff the variational approximation is exact" combines these two steps.

Averaging again over $\mathbf x_t^h$ corrupted from clean data gives the source's forward expectation. The source replaces $r_h$ in the denominator with $q(\mathbf x_t^{h+1:K}\mid\mathbf x_t^0)$ and collects the difference into a constant $C_1$ that depends only on $\mathbf x_t^0$ and $k_t$. [Claim 2, p. 17](https://arxiv.org/pdf/2407.01392v4#page=17)

### Claim 3, Part 1: Bayes' Identity and Telescoping

Equation (A.10) reverses a forward transition through Bayes' identity:

$$
q(\mathbf x_t^j\mid\mathbf x_t^{j-1},\mathbf x_t^0)
=q(\mathbf x_t^{j-1}\mid\mathbf x_t^j,\mathbf x_t^0)
\frac{q(\mathbf x_t^j\mid\mathbf x_t^0)}
{q(\mathbf x_t^{j-1}\mid\mathbf x_t^0)}.
$$

All these conditionals belong to the same forward process. Products of marginal-density ratios cancel intermediate terms:

$$
\prod_{j=h+2}^{K}
\frac{q(\mathbf x_t^j\mid\mathbf x_t^0)}
{q(\mathbf x_t^{j-1}\mid\mathbf x_t^0)}
=\frac{q(\mathbf x_t^K\mid\mathbf x_t^0)}
{q(\mathbf x_t^{h+1}\mid\mathbf x_t^0)}.
$$

This telescoping cancellation replaces the path ratio with boundary terms and reverse-step comparisons. Clean reconstruction at $h=0$ is handled separately from density KL. Because the source does not parameterize the pure-noise base distribution $p(\mathbf x_t^K\mid\mathbf z_{t-1})$, it treats this boundary term as a constant; the supplementary derivation below likewise takes the base as fixed, independent of both parameters and latents. [Claim 3, pp. 17–18](https://arxiv.org/pdf/2407.01392v4#page=18)

### Claim 3, Part 2: The Expected Log Ratio Is Negative KL

For normalized distributions,

$$
\mathbb E_q\log\frac{p}{q}
=-\mathbb E_q\log\frac{q}{p}
=-\mathrm D_{\mathrm{KL}}(q\Vert p).
$$

Apply this to reverse conditionals at fixed $\mathbf x_t^j$, then average over $\mathbf x_t^j$. A single sampled log ratio is not itself the KL. Page 18 prints this expectation as positive KL, which is where the theorem's sign error seen in A.1 originates.

Define review-added abbreviations:

$$
\begin{aligned}
R_t&=\mathbb E_{q(\mathbf x_t^1\mid\mathbf x_t^0)}
\big[\log p_\theta(\mathbf x_t^0\mid\mathbf x_t^1,\mathbf z_{t-1})\big],\\
D_{t,j}&=\mathbb E_{q(\mathbf x_t^j\mid\mathbf x_t^0)}
\mathrm D_{\mathrm{KL}}\!\left(\begin{gathered}
q(\mathbf x_t^{j-1}\mid\mathbf x_t^j,\mathbf x_t^0)\ \Vert\\
p_\theta(\mathbf x_t^{j-1}\mid\mathbf x_t^j,\mathbf z_{t-1})
\end{gathered}\right).
\end{aligned}
$$

For fixed noise level $h$, the **supplementary derivation has the form**

$$
\begin{aligned}
&\mathbb E_{q(\mathbf x_t^h\mid\mathbf x_t^0)}
\log p_\theta(\mathbf x_t^h\mid\mathbf z_{t-1})\\
&\quad\ge C_h+\mathbf1\{h=0\}R_t\\
&\qquad-\sum_{j=2}^{K}\mathbf1\{h\le j-1\}D_{t,j}.
\end{aligned}
$$

$C_h$ collects the parameter-independent boundary and entropy terms from the fixed forward process and base distribution. At $h=K$ no reverse steps remain and only the base distribution is left. This is the printed Claim 3 corrected to agree with Jensen, Bayes, and the KL definition.

### Finish by Averaging Noise Levels and Combining Time Steps

Uniform $h\in\{0,\ldots,K\}$ gives

$$
\begin{aligned}
\Pr(h=0)&=\frac1{K+1},\\
\Pr(h\le j-1)&=\frac{j}{K+1}.
\end{aligned}
$$

Using the source's $\mathbb E_{<t}$ for the expectation over past noisy history and latents, the supplementary form becomes the following; it is A.1's printed bound with the KL term corrected, written with $R_t$ and $D_{t,j}$.

$$
\begin{aligned}
&\mathbb E_{\mathrm{forward}}
\log p_\theta((\mathbf x_t^{k_t})_{t=1}^{T})\\
&\quad\ge C+
\sum_{t=1}^{T}\mathbb E_{<t}
\left[
\frac{R_t}{K+1}
-\sum_{j=2}^{K}\frac{j}{K+1}D_{t,j}
\right].
\end{aligned}
$$

Because the current noise level is independent of the past, these coefficients can be computed while keeping the expectation over history. The bound rewards reconstruction likelihood and penalizes reverse-step KL, and substituting the Gaussian calculation turns it into **weighted denoising MSE minimization**.

For general $\mathcal D$, the coefficients become $P_t(0\mid k_{1:t-1})$ and $\Pr(k_t\le j-1\mid k_{1:t-1})$, the cumulative probability used in A.1.2.

## Review interpretation and open questions

**Review interpretation.** The proof's backbone extends the standard diffusion ELBO derivation to a different starting noise level for each token. The two Jensen steps, Bayes' identity with telescoping, and the coefficients from uniform noise averaging hold as followed above. The printed errors are concentrated in the KL sign, the reindexing, the Gaussian variance, and the general-distribution weights, and correcting them leaves the training objective unchanged. The weak link is weight invariance: it holds only for a network that can represent every conditional mean, and, as the scalar example in A.1.1 shows, weights change the solution for a finite shared model.

Two questions remain. In a real RNN, how much does the denoising error on fixed patterns that rarely occur during training (for example, the small-noise history used for stabilization in part 1) differ from that on common patterns? If Theorem A.3's cumulative-probability weights were used in training to match a particular sampling schedule, would performance under that schedule change?

## Sources and reading scope

- Paper: *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion* · Boyuan Chen et al. · first public date 2024-07-01 · [reviewed version](https://arxiv.org/abs/2407.01392v4) · reviewed 2026-10-23.
- Scope read: Version 4, appendix A, pages 14–19, in the order setup, A.1, A.1.1, A.1.2, A.2, plus the cited main-text §3.2 and appendix D.1.
- External sources consulted: [DDPM v2](https://arxiv.org/abs/2006.11239v2) §2–3.2, [Chan tutorial v1](https://arxiv.org/abs/2403.18103v1) §2.6–2.7.
- Figure sources: The thumbnail is the authors' v4 Figure 2 (CC BY 4.0), rasterized and WebP-encoded with content, colors, arrows, and notation retained. [Paper](https://arxiv.org/abs/2407.01392v4).
- Not verified: formal verification of the complete theorems, author confirmation, and the proof of Theorem A.3, which the source omits.
