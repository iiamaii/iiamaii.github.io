---
title: "[2/2] Diffusion Forcing: 모든 부분 시퀀스의 우도를 학습한다는 것은 무엇을 뜻할까?"
description: "부록 A의 ELBO·Gaussian 손실·부분 시퀀스 논증을 따라 두 Jensen 단계와 중간 계산을 설명하고, 원문 부호·전이·가중치의 불일치를 구분한다."
date: "2026-10-14"
publishedAt: "2026-10-14T13:58:27+09:00"
updatedAt: "2026-10-17T22:20:55+09:00"
topics: ["diffusion models", "sequence modeling", "world action model", "robot learning", "probabilistic modeling"]
visibility: "public"
lang: "ko"
translationKey: "diffusion-forcing-theory"
paperTitle: "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion"
paperPublishedDate: "2024-07-01"
authors: ["Boyuan Chen", "Diego Marti Monso", "Yilun Du", "Max Simchowitz", "Russ Tedrake", "Vincent Sitzmann"]
year: "2024"
paperUrl: "https://arxiv.org/abs/2407.01392v4"
thumbnail: "/assets/reviews/diffusion-forcing/overview.webp"
thumbnailAlt: "Diffusion Forcing 원문 Figure 2: 서로 다른 토큰별 노이즈를 학습하고 여러 conditioning pattern으로 샘플링하는 구조."
---

Diffusion Forcing의 이론은 “노이즈를 무작위로 넣으면 모델이 강해진다”보다 구체적이다. 서로 다른 noise pattern의 우도를 하한으로 연결하고, 충분히 표현력 있는 모델이 여러 denoising 문제를 동시에 최적화할 수 있다는 논증이다. **평균 noise pattern의 학습 손실, 특정 pattern의 ELBO, 유한 모델의 실제 정확도는 서로 다르다.** 이 구분을 유지해야 “모든 부분 시퀀스”라는 표현을 정확하게 읽을 수 있다.

먼저 [[1/2] 방법·실험 편](/reviews/diffusion-forcing-methods/)을 읽는 것을 권한다. 이 글은 v4 **부록 A, pp. 14–19**의 순서인 설정→A.1 주요 결과→A.1.1 Gaussian→A.1.2 부분 시퀀스→A.2 증명을 따른다. A.2의 계산을 자세히 설명하되, 원문에 인쇄된 수식의 부호·방향·가중치 불일치를 그대로 올바른 정리로 인정하지 않는다. 아래의 일관된 형태는 명시한 조건에서의 **리뷰 보충 유도**이며 저자가 발행한 정오표가 아니다.

## A. 이론적 정당화: 먼저 대상과 가정을 정한다 (Theoretical Justification)

$\mathbf x_t^0$는 깨끗한 토큰, $\mathbf x_t^k$는 noise level $k$의 토큰, $T$는 시퀀스 길이, $K$는 최대 노이즈 수준이다. 시간 $t$와 노이즈 $k$를 혼동하지 않는다.

전방 과정은 고정된 Markov chain이다. 식 (A.1)은 다음과 같다.

$$
q(\mathbf x^{1:K}\mid\mathbf x^0)
=\prod_{k=1}^{K}q(\mathbf x^k\mid\mathbf x^{k-1}).
$$

모델 $p_\theta$는 latent $\mathbf z_t$가 과거를 충분히 요약한다는 **마르코프 성질**(Markov property)을 가정한다.

$$
\begin{aligned}
&p_\theta(\mathbf z_t,\mathbf x_t^{k_t}\mid
\mathbf z_{1:t-1},(\mathbf x_s^{k_s})_{s<t})\\
&\qquad=p_\theta(\mathbf z_t,\mathbf x_t^{k_t}\mid
\mathbf z_{t-1}).
\end{aligned}
$$

이것은 모델의 분해에 대한 가정이다. 실제 관측 세계가 항상 Markov라는 주장과 다르다. 모델은 기억을 사용해 현재 관찰만으로 불충분한 과거 정보를 요약한다.

**결정론적 잠재 상태**(deterministic latents)는 입력·이전 latent·noise level이 주어지면 다음 latent가 하나로 정해진다는 뜻이다. 무작위 데이터의 불확실성이 사라진다는 뜻은 아니다. 결정론적 벡터가 불확실한 미래를 예측하는 데 필요한 정보를 요약할 수 있다.

Remark 1은 더 미묘한 조건을 설명한다. 서로 다른 noise history에서는 같은 시간의 latent도 달라질 수 있으므로 $\mathbf z_t(k_{1:t})$처럼 history별 변수를 생각해야 한다. 이를 하나의 동일한 값으로 취급하면 여러 conditioning을 한 확률모형 안에서 정의할 때 문제가 생긴다. 이 dependence는 이후 표기에서 생략된다. [식 (A.1)–(A.3)·Remark 1, p. 14](https://arxiv.org/pdf/2407.01392v4#page=14)

## A.1 주요 결과: Theorem A.1의 의도와 인쇄된 식 (Main Results)

Theorem A.1은 깨끗한 시퀀스 $\mathbf x_{1:T}^0$를 고정하고, 토큰별 noise level을 균등하게 뽑은 뒤 전방 과정으로 오염시킨 시퀀스의 **평균 로그 우도**(expected log-likelihood)를 다룬다. 과거는 latent로 요약하고, reverse diffusion의 reconstruction·KL 항을 합해 ELBO를 만든다는 내용이다.

원문 증명은 $\{0,\ldots,K\}$를 사용한다. 이 글은 그 끝점 규약으로 읽으며 $[K]$의 의미가 문맥마다 같은 것으로 가정하지 않는다. 등호 조건으로는 결정론적 latent와 정확한 변분 근사(exact variational approximation)를 제시한다. 결정론적 latent만으로 두 조건 모두가 충족되지는 않는다.

### 인쇄된 식에서 먼저 확인할 불일치

| 원문 위치 | 인쇄된 내용 | 대조 결과 |
| --- | --- | --- |
| Theorem A.1, p. 14 | 하한 안에 $+\mathrm D_{\mathrm{KL}}(q\Vert p_\theta)$ | $\mathbb E_q\log(p_\theta/q)=-\mathrm D_{\mathrm{KL}}(q\Vert p_\theta)$이므로 ELBO 전개와 부호가 반대다. |
| 같은 KL의 모델 분포 | $p_\theta(\mathbf x_t^j\mid\mathbf x_t^{j-1},\mathbf z_{t-1})$ | 비교 분포는 reverse step이므로 $\mathbf x_t^{j-1}\mid\mathbf x_t^j$와 방향을 맞춰야 한다. |
| Corollary A.2, p. 15 | 로그 우도 하한 쪽에 양의 제곱오차 | Gaussian KL을 빼는 ELBO와 일관된 방향으로 다시 유도해야 한다. |
| Theorem A.3, p. 16 | $jP_t(j\mid k_{1:t-1})$ | 증명에 등장하는 indicator의 평균은 일반 분포에서 누적 확률이다. 균등 분포의 단순식을 그대로 옮길 수 없다. |

PDF와 저자 LaTeX 양쪽에 같은 표기가 있어 단순 OCR 문제로 처리하지 않았다. 또한 $K$여야 할 diffusion 범위에 $T$를 쓰는 부분과 등호를 설명하며 “inequality”라고 적은 문장도 있다. 핵심 논증을 이해하려면 정의와 유도를 대조해야 한다. [Theorem A.1–A.3](https://arxiv.org/pdf/2407.01392v4#page=14)

이 불일치를 발견했다고 경험적 결과가 자동으로 사라지는 것은 아니다. 반대로 실험이 잘 되었다고 인쇄된 정리의 정확성이 인증되는 것도 아니다.

## A.1.1 Gaussian diffusion으로 구체화하기 (Specializing to Gaussian diffusion)

Corollary A.2의 중요한 아이디어는 **Gaussian KL이 가중 MSE로 바뀐다**는 것이다. DDPM의 posterior 계산을 적용한다. $j\ge2$에서 표준 전방 과정의 역조건부 분포는 다음과 같다.

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

$A_j,B_j$는 긴 계수를 쉽게 읽기 위해 이 리뷰가 추가한 기호다. DDPM 식 (6)–(7)에 대응한다. DF의 인쇄된 variance에는 $1-\sqrt{\bar\alpha_{j-1}}$가 들어가는데, 이를 표준 posterior variance와 같은 것으로 사용하지 않았다. 원문이 참조한 Chan tutorial v1에도 관련 variance 표기 문제가 보이므로 DDPM과 독립적인 Gaussian 계산을 기준으로 대조했다. [DDPM, §2–3.2](https://arxiv.org/pdf/2006.11239v2#page=3)

모델 mean을

$$
\boldsymbol\mu_\theta
=A_j\mathbf x_t^j+
B_j\hat{\mathbf x}_\theta^0(
\mathbf x_t^j,\mathbf z_{t-1},j)
$$

로 놓으면 known posterior와의 차이는

$$
\tilde{\boldsymbol\mu}_j-\boldsymbol\mu_\theta
=B_j(\mathbf x_t^0-\hat{\mathbf x}_\theta^0)
$$

다. 공분산을 같은 $\sigma_j^2\mathbf I$로 선택했을 때 Gaussian KL의 일반식이 단순화된다.

$$
\mathrm D_{\mathrm{KL}}\!
\left(\mathcal N(\boldsymbol\mu_q,\sigma_j^2\mathbf I)
\Vert\mathcal N(\boldsymbol\mu_p,\sigma_j^2\mathbf I)\right)
=\frac{\|\boldsymbol\mu_q-\boldsymbol\mu_p\|^2}
{2\sigma_j^2}.
$$

따라서 KL은 $c_j\|\hat{\mathbf x}_\theta^0-\mathbf x_t^0\|^2$이며

$$
c_j=\frac{B_j^2}{2\sigma_j^2}
=\frac{(1-\alpha_j)^2\bar\alpha_{j-1}}
{2\sigma_j^2(1-\bar\alpha_j)^2}.
$$

서로 다른 **고정** 공분산이라면 추가 항이 붙지만 모델 mean에 관한 부분은 같은 가중 MSE다. $j=1$은 깨끗한 데이터를 재구성하는 decoder 항으로 별도 처리해야 한다. 표준 전방 posterior의 끝점 variance 0을 나눗셈에 넣지 않는다.

이 계산은 원문에서 짧게 넘어간 부분의 **리뷰 보충 유도**다. KL은 ELBO에서 빼기 때문에, 하한을 높이는 방향은 MSE를 낮추는 방향이다. 양의 MSE가 커질수록 하한이 커진다는 해석은 아니다. [Corollary A.2, p. 15](https://arxiv.org/pdf/2407.01392v4#page=15)

### 가중치를 없애도 된다는 말의 조건

저자들은 충분한 latent와 함수의 표현력(expressivity)이 있으면 각 조건부 denoising 문제의 최적해를 동시에 구현할 수 있다고 설명한다. 제곱 손실의 최적 예측은 **조건부 평균**(conditional mean)이다.

리뷰 보충으로 입력 정보를 $\mathcal I$라 두면

$$
\begin{aligned}
\mathbb E\|\mathbf x-f(\mathcal I)\|^2
&=\mathbb E\|\mathbf x-\mathbb E[\mathbf x\mid\mathcal I]\|^2\\
&\quad+\mathbb E\|\mathbb E[\mathbf x\mid\mathcal I]-f(\mathcal I)\|^2.
\end{aligned}
$$

교차항은 $\mathbb E[\mathbf x-\mathbb E(\mathbf x\mid\mathcal I)\mid\mathcal I]=0$이라 사라진다. 첫 항은 남는 불확실성, 둘째는 예측 함수의 추가 오류다. 벡터라면 첫 항은 조건부 covariance의 trace를 평균한 값이다. 원문의 “조건부 분산” 표현은 이 구분을 압축한다.

조건별 최적 평균을 모두 표현할 수 있고 양의 가중치가 조건에 따라 정해지면, 가중치를 바꾸어도 같은 조건별 최적해를 얻을 수 있다. **유한한 공유 네트워크에서는 다르다.** 설명용으로 두 조건의 목표 0·2를 하나의 스칼라 $u$로 예측하면

$$
\underset{u}{\arg\min}\,
\{w_1u^2+w_2(u-2)^2\}
=\frac{2w_2}{w_1+w_2}.
$$

가중치가 1:1이면 1, 1:3이면 1.5다. 충분한 표현력이라는 조건을 제거하면 가중치가 최적해를 바꾼다. 원문 D.1의 fused SNR이 실제 성능에 중요하다는 사실과도 모순되지 않는다.

## A.1.2 모든 부분 시퀀스: Theorem A.3을 읽는 법 (Capturing all subsequences)

Theorem A.3은 noise sequence의 임의 분포 $\mathcal D$를 허용하고

$$
P_t(j\mid k_{1:t-1})
=\Pr_{\mathcal D}(k_t=j\mid k_{1:t-1})
$$

를 정의한다. noise가 시간상 상관되어도 각 pattern의 우도를 다룰 수 있다는 것이 의도다. 특정 고정 pattern은 $\mathcal D$가 그 pattern에 집중된 경우다.

중요한 구분은 **학습 목적 자체가 모든 고정 pattern의 동일한 하한인 것은 아니라는 점**이다. 균등한 독립 noise 학습이 많은 조건의 항들을 포함하고, 충분한 표현력 아래 그 항들을 동시에 잘 최적화할 수 있다는 주장이다.

### 왜 일반 분포에서는 누적 확률이 필요한가?

A.2의 전개에서는 중간 noise level $k_t$에서 시작해도 그보다 더 noisy한 reverse steps가 함께 등장한다. 역방향 step을 $j\to j-1$로 쓰면 해당 KL이 포함되는 조건은 $k_t\le j-1$다. 따라서 그 가중치는

$$
\begin{aligned}
&\mathbb E_{\mathcal D}[
\mathbf1\{k_t\le j-1\}\mid k_{1:t-1}]\\
&\quad=\Pr_{\mathcal D}(k_t\le j-1\mid k_{1:t-1})\\
&\quad=\sum_{h=0}^{j-1}P_t(h\mid k_{1:t-1})
\end{aligned}
$$

가 되어야 한다. 이것은 원문의 indicator 전개를 따른 **리뷰 보충 계산**이며 A.3의 인쇄된 식을 그대로 전사한 것이 아니다.

균등 분포라면 $j$개의 끝점 $0,\ldots,j-1$이 포함되어 $j/(K+1)$이 된다. 그러나 일반 분포에서 이를 $jP_t(j\mid k_{1:t-1})$로 대체할 수는 없다. 예를 들어 $K=3$, $k_t=1$에 모든 확률이 있으면 $j=2$의 누적 확률은 1이고 $2P_t(2)$는 0이다. 복원 항의 계수도 noise level 0에 대한 확률과 연결되어야 하므로 인쇄된 $P_t(1)$은 끝점 규약과 함께 확인해야 한다. [Theorem A.3·A.2, pp. 16–19](https://arxiv.org/pdf/2407.01392v4#page=16)

### “부분 시퀀스”는 어떤 경우인가?

$k_t\in\{0,K\}$인 pattern은 어떤 토큰은 깨끗하게 남기고 다른 토큰은 거의 정보가 없도록 오염시킨다. 완전한 masking의 이상화에서는 지운 토큰을 marginalize하여 남은 토큰의 분포와 연결한다.

여기에는 충분한 표현력, 적절한 전방·역방향 과정, noise 끝점과 conditioning의 정합성이 필요하다. 유한한 $K$의 근사 masking과 유한 데이터·RNN의 실제 학습에서는 자동으로 exact conditional generation이 따라오지 않는다. 또한 **학습 시퀀스 안의 부분 분포**를 모델링하는 논증은 학습보다 긴 모든 시퀀스의 정확도나 임의의 안전한 궤적 조합을 보장하지 않는다.

## A.2 Theorem A.1의 증명 따라가기 (Proof of Theorem A.1)

원문은 세 Claim을 사용한다. latent를 펼치고, noise 경로에 ELBO를 적용하고, 그 하한의 기대값을 계산한다. 아래는 이 순서를 유지하며 부호·조건부 분포를 대조한 해설이다.

### Claim 1: latent를 펼치는 첫 번째 Jensen 부등식

고정된 noise history에 대해 joint model을 적분하면

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

로그는 concave하므로 $\log\mathbb E[Y]\ge\mathbb E[\log Y]$다.

$$
\log p_\theta((\mathbf x_t^{k_t})_{t=1}^{T})
\ge
\mathbb E_{p,\mathbf z_{1:T}}
\left[\sum_{t=1}^{T}
\log p_\theta(\mathbf x_t^{k_t}\mid\mathbf z_{t-1})\right].
$$

결정론적 latent에서는 latent 적분에서 한 경로만 남아 이 단계가 등호가 된다. 확률적 latent에서도 product가 거의 확실하게 상수라면 등호일 수 있으므로, 결정론은 충분조건으로 읽는다. [Claim 1·식 (A.9), pp. 16–17](https://arxiv.org/pdf/2407.01392v4#page=16)

### Claim 2: noise 경로를 넣는 두 번째 Jensen 부등식

시간 $t$를 고정하고 $h=k_t$라고 쓰자. $h$는 이 해설의 보조 기호다. 관찰한 $\mathbf x_t^h$의 우도를 얻으려면 더 noisy한 경로 $\mathbf x_t^{h+1:K}$를 적분한다. 전방 조건부 분포를 $r_h=q(\mathbf x_t^{h+1:K}\mid\mathbf x_t^h)$로 두면

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

분모에 넣은 분포의 support가 모델 경로를 다루기에 적절하고 적분이 존재해야 한다. 등호는 importance ratio가 상수인 경우, 즉 이 경로의 변분분포가 모델 posterior와 일치하는 경우다.

다시 깨끗한 데이터로부터 오염시킨 $\mathbf x_t^h$에 대해 평균하면 원문의 forward expectation을 얻는다. 원문은 $r_h$와 $\mathbf x_t^0$에 조건부인 전방 경로 사이의 차이를 모델 parameter와 무관한 상수로 묶는다. 이 상수를 생략했다고 새로운 학습 항이 사라지는 것은 아니다. [Claim 2, p. 17](https://arxiv.org/pdf/2407.01392v4#page=17)

### Claim 3, Part 1: Bayes 항등식과 telescoping

원문 식 (A.10)은 전방 전이를 역조건부 분포로 바꾸는 Bayes 항등식이다.

$$
q(\mathbf x_t^j\mid\mathbf x_t^{j-1},\mathbf x_t^0)
=q(\mathbf x_t^{j-1}\mid\mathbf x_t^j,\mathbf x_t^0)
\frac{q(\mathbf x_t^j\mid\mathbf x_t^0)}
{q(\mathbf x_t^{j-1}\mid\mathbf x_t^0)}.
$$

조건부 분포는 같은 전방 과정에서 나온 것이다. marginal density의 비율을 여러 $j$에 곱하면 중간항이 소거된다.

$$
\prod_{j=h+2}^{K}
\frac{q(\mathbf x_t^j\mid\mathbf x_t^0)}
{q(\mathbf x_t^{j-1}\mid\mathbf x_t^0)}
=\frac{q(\mathbf x_t^K\mid\mathbf x_t^0)}
{q(\mathbf x_t^{h+1}\mid\mathbf x_t^0)}.
$$

이 **망원합 소거**(telescoping cancellation)가 경로 전체의 복잡한 비율을 boundary 항과 reverse-step 비교들로 바꾼다. $h=0$의 깨끗한 decoder는 density KL과 따로 처리한다. 보충 유도에서는 순수 noise의 base distribution이 학습 parameter와 latent 모두에 무관하게 고정되어 있다고 가정한다. 이 조건이 있어야 terminal 항을 상수에 넣을 수 있다. [Claim 3, pp. 17–18](https://arxiv.org/pdf/2407.01392v4#page=18)

### Claim 3, Part 2: 기대값은 KL의 음수다

임의의 정규화된 분포 $q,p$에 대해

$$
\mathbb E_q\log\frac{p}{q}
=-\mathbb E_q\log\frac{q}{p}
=-\mathrm D_{\mathrm{KL}}(q\Vert p).
$$

이를 고정된 $\mathbf x_t^j$의 역조건부 분포에 적용한 뒤 $\mathbf x_t^j$에 대해 평균해야 한다. 단일 샘플의 log ratio 자체가 KL은 아니다. 원문 p. 18에서는 이 기대값을 양의 KL로 적었는데, 정의와 반대다.

계산을 간단히 쓰기 위해 리뷰 보조 기호를 정의한다.

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

고정 noise level $h$에서 얻는 **보충 유도의 형태**는

$$
\begin{aligned}
&\mathbb E_{q(\mathbf x_t^h\mid\mathbf x_t^0)}
\log p_\theta(\mathbf x_t^h\mid\mathbf z_{t-1})\\
&\quad\ge C_h+\mathbf1\{h=0\}R_t\\
&\qquad-\sum_{j=2}^{K}\mathbf1\{h\le j-1\}D_{t,j}.
\end{aligned}
$$

$C_h$는 위의 고정 전방 과정·base distribution에서 parameter와 무관한 boundary/entropy 항을 묶는다. $h=K$에서는 reverse steps가 없어 별도의 fixed-base 경우다. 이 식은 인쇄된 정리의 문자 그대로의 복사가 아니라 Jensen·Bayes·KL의 정의에 맞춘 해설이다.

### 마무리: 균등 noise 평균과 전체 시퀀스

$h$를 $\{0,\ldots,K\}$에서 균등하게 뽑으면

$$
\begin{aligned}
\Pr(h=0)&=\frac1{K+1},\\
\Pr(h\le j-1)&=\frac{j}{K+1}.
\end{aligned}
$$

원문처럼 $\mathbb E_{<t}$를 과거 noisy history와 그 latent에 대한 평균으로 쓰면 보충 유도는 다음 형태를 갖는다.

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

독립적인 현재 noise level 때문에 과거의 평균을 유지하면서 이 계수를 계산할 수 있다. 복원 항은 높이고 reverse-step KL은 낮추는 하한이다. 앞의 Gaussian 계산을 대입하면 **가중 denoising MSE를 최소화**하는 학습 문제로 이어진다.

일반 $\mathcal D$에서는 같은 계산의 계수를 $P_t(0\mid k_{1:t-1})$와 $\Pr(k_t\le j-1\mid k_{1:t-1})$로 바꾼다. 이것이 앞에서 구분한 누적 확률이다. 고정 pattern은 균등 목적에 포함된 항들의 일부에 대응하지만, 그 사실만으로 유한 모델에서 모든 pattern의 우도가 동시에 최대로 된다고 결론내릴 수 없다.

## 리뷰 해석과 남은 질문

부록의 설계 논리는 분명하다. noisy history를 받아 denoise하는 학습 항들을 넓게 만들고, 여러 conditioning pattern을 같은 모델로 처리하려 한다. 다만 이론을 읽을 때는 **재가중 목적의 관계**, **충분한 표현력의 이상화**, **인쇄된 수식의 정확성**을 각각 확인해야 한다.

이 글은 두 Jensen 단계·Bayes 소거·KL 부호·Gaussian mean 계산·noise 가중치를 국소적으로 검토했다. 정리를 형식 검증하거나 저자의 수정·확인을 받은 것은 아니다. 특히 원문 A.3의 인쇄된 일반 분포 식은 그대로 사용하지 않으며, 이 리뷰의 보충 계산을 공식 수정본으로 표시하지 않는다.

## 출처와 읽은 범위

- 원문: Chen 등 여섯 저자, [Diffusion Forcing v4](https://arxiv.org/abs/2407.01392v4), 최초 제출 2024-07-01, 검토 2026-10-14. 첫 읽기와 전문 두 번째 읽기의 공통 기록은 [1편 출처 절](/reviews/diffusion-forcing-methods/#출처와-읽은-범위)에 있다.
- 이 글의 정독 범위: 부록 A 전체 pp. 14–19, Theorem A.1·Corollary A.2·Theorem A.3·Claim 1–3·식 (A.1)–(A.13). 저자 LaTeX와 렌더링한 PDF를 대조했다. 원문 번호는 출처 식을 찾기 위한 번호이며 리뷰의 새 식에 원문 번호를 붙이지 않았다.
- 추가 대조: [DDPM v2 §2–3.4](https://arxiv.org/pdf/2006.11239v2), [Chan tutorial v1 §2.6–2.7](https://arxiv.org/pdf/2403.18103v1). 표준 posterior variance는 DDPM 식 (7)과 독립 Gaussian 계산으로 확인했다.
- 리뷰 보충: $A_j,B_j,h,r_h,R_t,D_{t,j},\mathcal I,u,w_1,w_2$는 설명을 위해 정의한 기호다. Gaussian KL, conditional-mean decomposition, indicator 평균과 유한 공유 모델 예제를 펼쳤다. 원문 인쇄식의 공식 정오표나 전체 정리 인증이 아니다.
- 검증: 표 기반 주장의 국소 검산과 KL 부호·Gaussian variance·누적 noise 가중치·공유 모델 예제를 계산했다. **미실행:** 전체 정리의 기계/형식 검증, 저자 확인, 모델 학습·sampling·GPU·로봇 재현, raw results 기반 통계 검정.
- 대표 그림은 [저자 v4 Figure 2](https://arxiv.org/pdf/2407.01392v4#page=3)를 두 편에 공통 사용한다. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), 원저자 Chen·Marti Monso·Du·Simchowitz·Tedrake·Sitzmann. 원본 PDF를 래스터화·WebP 인코딩했으며 내용·화살표·문자를 유지했다.
