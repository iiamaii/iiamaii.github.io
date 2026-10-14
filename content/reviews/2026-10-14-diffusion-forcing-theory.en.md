---
title: "[2] Diffusion Forcing: What Does Learning Every Subsequence Likelihood Mean?"
description: "Follow appendix A's ELBO, Gaussian loss, and subsequence argument through two Jensen steps and intermediate calculations, separating printed inconsistencies from supplementary derivations."
date: "2026-10-14"
publishedAt: "2026-10-14T13:58:27+09:00"
updatedAt: "2026-10-14T13:58:27+09:00"
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

Diffusion Forcing's theory is more specific than “random noise makes a model robust.” It connects noise-pattern likelihoods to a lower bound, then argues that an expressive model can optimize multiple denoising problems simultaneously. **An average training objective, an ELBO for a particular pattern, and a finite model's actual accuracy are different objects.** That distinction is essential to understanding “all subsequences.”

Read [[1] Methods and experiments](/en/reviews/diffusion-forcing-methods/) first. This article follows **appendix A, pp. 14–19**, in its original order: setup, A.1 main results, A.1.1 Gaussian specialization, A.1.2 subsequences, and A.2 proof. The proof is expanded without accepting printed sign, transition-direction, and weighting inconsistencies as correct statements. Consistent formulas below are **supplementary review derivations under stated conditions**, not an author-issued erratum.

## A. Theoretical Justification: Objects and Assumptions

The clean token is $\mathbf x_t^0$, its level-$k$ version is $\mathbf x_t^k$, sequence length is $T$, and maximum noise level is $K$. Sequence time and diffusion level remain separate.

The fixed forward process is Markov, as in equation (A.1):

$$
q(\mathbf x^{1:K}\mid\mathbf x^0)
=\prod_{k=1}^{K}q(\mathbf x^k\mid\mathbf x^{k-1}).
$$

The model assumes a Markov property in which $\mathbf z_t$ summarizes the relevant history:

$$
\begin{aligned}
&p_\theta(\mathbf z_t,\mathbf x_t^{k_t}\mid
\mathbf z_{1:t-1},(\mathbf x_s^{k_s})_{s<t})\\
&\qquad=p_\theta(\mathbf z_t,\mathbf x_t^{k_t}\mid
\mathbf z_{t-1}).
\end{aligned}
$$

This is a model factorization, not a claim that the observed world is always Markov. Memory summarizes information unavailable from the current observation alone.

Deterministic latents mean that input, previous latent, and noise level determine one next latent. They do not eliminate uncertainty in the data; a deterministic representation can support uncertain predictions.

Remark 1 identifies a subtle requirement: different noise histories can produce different latents at the same sequence position. Variables should therefore be understood as $\mathbf z_t(k_{1:t})$, indexed by noise history. Treating them as one identical value across incompatible conditioning patterns creates problems in defining a joint probabilistic model. That dependence is suppressed in later notation. [Equations (A.1)–(A.3), Remark 1, p. 14](https://arxiv.org/pdf/2407.01392v4#page=14)

## A.1 Main Results: Theorem A.1's Intent and Printed Expression

Theorem A.1 fixes clean data $\mathbf x_{1:T}^0$, samples tokenwise noise uniformly, and considers the resulting sequence's expected log-likelihood. Its intended construction combines reconstruction and reverse-diffusion KL terms into an evidence lower bound.

The proof uses $\{0,\ldots,K\}$; that endpoint convention is used here rather than assuming every occurrence of $[K]$ is unambiguous. The source gives deterministic latents and exact variational approximation as equality conditions. Deterministic latents alone do not satisfy both.

### Inconsistencies to check before using the formula

| Source location | Printed expression | Comparison with the derivation |
| --- | --- | --- |
| Theorem A.1, p. 14 | $+\mathrm D_{\mathrm{KL}}(q\Vert p_\theta)$ inside the lower bound | $\mathbb E_q\log(p_\theta/q)=-\mathrm D_{\mathrm{KL}}(q\Vert p_\theta)$ has the opposite sign. |
| Model distribution in that KL | $p_\theta(\mathbf x_t^j\mid\mathbf x_t^{j-1},\mathbf z_{t-1})$ | A reverse-step comparison requires matching the direction $\mathbf x_t^{j-1}\mid\mathbf x_t^j$. |
| Corollary A.2, p. 15 | Positive squared error on the lower-bound side | Gaussian KL must be reconciled with the negative-KL ELBO direction. |
| Theorem A.3, p. 16 | $jP_t(j\mid k_{1:t-1})$ | The proof's indicator average is a cumulative probability under a general distribution. |

These appear in both PDF and author LaTeX, so they are not dismissed as OCR artifacts. Other issues include using $T$ where the diffusion range calls for $K$ and writing “inequality” while discussing equality. Definitions and intermediate calculations need to be checked alongside the statements. [Theorems A.1–A.3](https://arxiv.org/pdf/2407.01392v4#page=14)

These issues do not automatically erase empirical results. Conversely, empirical success does not certify the printed theorems.

## A.1.1 Specializing to Gaussian Diffusion

Corollary A.2's central mechanism is **Gaussian KL becoming weighted MSE**. For $j\ge2$, the standard DDPM forward posterior is

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

$A_j,B_j$ are review-added abbreviations for coefficients in DDPM equations (6)–(7). DF's printed variance contains $1-\sqrt{\bar\alpha_{j-1}}$; it is not treated here as the standard posterior variance. A related variance issue also appears in the cited Chan tutorial v1, so DDPM and an independent Gaussian calculation are used as cross-checks. [DDPM, §2–3.2](https://arxiv.org/pdf/2006.11239v2#page=3)

Parameterize the model mean as

$$
\boldsymbol\mu_\theta
=A_j\mathbf x_t^j+
B_j\hat{\mathbf x}_\theta^0(
\mathbf x_t^j,\mathbf z_{t-1},j)
$$

The mean difference is then

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

Different **fixed** covariances add other terms while retaining a weighted mean-squared-error term. Step $j=1$ is handled separately as clean-data reconstruction; the forward posterior's zero endpoint variance is not inserted into a division.

This expands the source's abbreviated calculation as a **supplementary review derivation**. Because KL is subtracted in the ELBO, reducing MSE raises the bound. Increasing a positive MSE does not improve a likelihood lower bound. [Corollary A.2, p. 15](https://arxiv.org/pdf/2407.01392v4#page=15)

### When may weights be dropped?

The authors argue that sufficiently expressive latents and predictors can implement each conditional denoising optimum simultaneously. Under squared error, the optimal predictor is the conditional mean.

Writing input information as the review-added $\mathcal I$,

$$
\begin{aligned}
\mathbb E\|\mathbf x-f(\mathcal I)\|^2
&=\mathbb E\|\mathbf x-\mathbb E[\mathbf x\mid\mathcal I]\|^2\\
&\quad+\mathbb E\|\mathbb E[\mathbf x\mid\mathcal I]-f(\mathcal I)\|^2.
\end{aligned}
$$

The cross term vanishes because the residual has zero conditional mean. The first term is irreducible conditional uncertainty; the second is the predictor's excess error. For vectors, the first is the expected trace of the conditional covariance, clarifying the source's abbreviated “conditional variance.”

If every conditional mean is representable and positive weights depend on the condition, reweighting preserves these conditional optima. **A finite shared network can behave differently.** For illustration, predicting targets 0 and 2 with one shared scalar $u$ gives

$$
\underset{u}{\arg\min}\,
\{w_1u^2+w_2(u-2)^2\}
=\frac{2w_2}{w_1+w_2}.
$$

Weights 1:1 give 1; weights 1:3 give 1.5. Without adequate expressivity, weights change the optimum. This is compatible with fused SNR weighting being practically important in appendix D.1.

## A.1.2 Capturing All Subsequences: Reading Theorem A.3

Theorem A.3 allows an arbitrary noise-sequence distribution $\mathcal D$ and defines

$$
P_t(j\mid k_{1:t-1})
=\Pr_{\mathcal D}(k_t=j\mid k_{1:t-1})
$$

The intended extension includes temporally correlated patterns. A fixed pattern is a distribution concentrated on one sequence of levels.

Importantly, **the training loss itself need not be the same lower bound for each fixed pattern**. Independent uniform training includes terms from many conditioning patterns; the simultaneous-optimization argument additionally needs sufficient expressivity.

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

This is a **supplementary calculation following the proof's indicator**, not a transcription of the printed A.3 formula.

For uniform noise, the $j$ included levels $0,\ldots,j-1$ give $j/(K+1)$. Under a general distribution, this is not $jP_t(j\mid k_{1:t-1})$. For example, if $K=3$ and all mass lies at $k_t=1$, the cumulative weight for $j=2$ is 1, whereas $2P_t(2)=0$. Reconstruction similarly corresponds to the probability of level zero, requiring examination of the source's printed $P_t(1)$ and endpoint convention. [Theorem A.3 and A.2, pp. 16–19](https://arxiv.org/pdf/2407.01392v4#page=16)

### What counts as a subsequence?

A pattern with $k_t\in\{0,K\}$ keeps some tokens clean and makes others nearly uninformative. In idealized complete masking, marginalizing masked tokens connects the construction to the distribution of retained tokens.

This requires expressivity, appropriate forward/reverse processes, and consistent endpoint and conditioning definitions. Approximate masking at finite $K$, finite data, and a finite RNN do not automatically imply exact conditional generation. Modeling subsets **within training sequences** also does not guarantee accurate arbitrarily longer sequences or safe composition of arbitrary trajectories.

## A.2 Proof of Theorem A.1

The source uses three claims: expand latents, apply an ELBO over noise paths, then compute the expected bound. The walkthrough retains this order while checking signs and conditional distributions.

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

Deterministic latents leave one path, making this step an equality. More generally, equality can hold if the product is almost surely constant, so determinism is read as a sufficient condition. [Claim 1, equation (A.9), pp. 16–17](https://arxiv.org/pdf/2407.01392v4#page=16)

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

The proposal must have suitable support and the integrals must exist. Equality requires a constant importance ratio, equivalently matching the model's posterior over this path.

Averaging over $\mathbf x_t^h$ corrupted from clean data gives the forward expectation. The source moves differences between $r_h$ and forward paths conditioned on $\mathbf x_t^0$ into a parameter-independent constant. Dropping such constants does not remove a parameter-dependent training term. [Claim 2, p. 17](https://arxiv.org/pdf/2407.01392v4#page=17)

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

This telescoping cancellation replaces the path ratio with boundary terms and reverse-step comparisons. Clean reconstruction at $h=0$ is handled separately from density KL. The supplementary derivation assumes a pure-noise base fixed independently of both parameters and latents, allowing the terminal term to be absorbed into a constant. [Claim 3, pp. 17–18](https://arxiv.org/pdf/2407.01392v4#page=18)

### Claim 3, Part 2: The Expected Log Ratio Is Negative KL

For normalized distributions,

$$
\mathbb E_q\log\frac{p}{q}
=-\mathbb E_q\log\frac{q}{p}
=-\mathrm D_{\mathrm{KL}}(q\Vert p).
$$

Apply this to reverse conditionals at fixed $\mathbf x_t^j$, then average over $\mathbf x_t^j$. A single sampled log ratio is not itself the KL. Page 18 prints the expectation as positive KL, contrary to its definition.

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

$C_h$ collects parameter-independent boundary/entropy terms under the fixed forward process and base distribution. At $h=K$, no reverse steps remain; this is a separate fixed-base case. The expression follows Jensen, Bayes, and the KL definition rather than copying the printed theorem literally.

### Finish by Averaging Noise Levels and Combining Time Steps

Uniform $h\in\{0,\ldots,K\}$ gives

$$
\begin{aligned}
\Pr(h=0)&=\frac1{K+1},\\
\Pr(h\le j-1)&=\frac{j}{K+1}.
\end{aligned}
$$

Using the source's $\mathbb E_{<t}$ for the expectation over past noisy history and latents, the supplementary form becomes

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

Independent current noise allows these coefficients to be computed while retaining the history expectation. Raise reconstruction likelihood and reduce reverse-step KL to improve the bound. Substituting the Gaussian calculation gives **weighted denoising MSE minimization**.

For general $\mathcal D$, replace coefficients by $P_t(0\mid k_{1:t-1})$ and $\Pr(k_t\le j-1\mid k_{1:t-1})$, the cumulative probability discussed earlier. A fixed pattern involves a subset of the uniform objective's terms; this alone does not show simultaneous maximum likelihood for every pattern in a finite model.

## Review Interpretation and Open Questions

The design argument is clear: train denoising over a broad range of noisy histories, allowing one model to handle multiple conditioning patterns. Assessing it requires separating **the relationship between reweighted objectives**, **the expressivity idealization**, and **the accuracy of the printed expressions**.

This review checks the two Jensen steps, Bayes cancellation, KL sign, Gaussian mean calculations, and noise weights locally. It does not formally verify the theorems or provide author confirmation. The printed general-distribution expression in A.3 is not adopted literally, and supplementary calculations are not labeled an official correction.

## Sources and Reading Scope

- Original: Chen and the five coauthors, [Diffusion Forcing v4](https://arxiv.org/abs/2407.01392v4), first submitted July 1, 2024; reviewed October 14, 2026. Shared first-pass and full second-pass scope is recorded in [part 1's sources](/en/reviews/diffusion-forcing-methods/#sources-and-reading-scope).
- Detailed scope here: all of appendix A, pp. 14–19, Theorem A.1, Corollary A.2, Theorem A.3, Claims 1–3, and equations (A.1)–(A.13), checked against author LaTeX and rendered PDF. Original numbers identify source locations; review equations are not assigned original equation numbers.
- Cross-checks: [DDPM v2 §2–3.4](https://arxiv.org/pdf/2006.11239v2), [Chan tutorial v1 §2.6–2.7](https://arxiv.org/pdf/2403.18103v1). Standard posterior variance was compared with DDPM equation (7) and an independent Gaussian calculation.
- Supplementary notation: $A_j,B_j,h,r_h,R_t,D_{t,j},\mathcal I,u,w_1,w_2$. The review expands Gaussian KL, conditional-mean decomposition, indicator averaging, and a finite shared-model example. It is not an official erratum or complete theorem certification.
- Verification: local checks of table claims, KL sign, Gaussian variance, cumulative noise weights, and the shared-model example. **Not executed:** complete mechanical/formal theorem verification, author confirmation, model training/sampling/GPU/robot reproduction, and statistical tests from raw results.
- Both parts share [original Figure 2](https://arxiv.org/pdf/2407.01392v4#page=3). Credit: Chen, Marti Monso, Du, Simchowitz, Tedrake, Sitzmann, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The PDF graphic was rasterized and WebP-encoded with content, arrows, and text preserved.
