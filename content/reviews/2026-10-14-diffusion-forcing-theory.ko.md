---
title: "[2/2] Diffusion Forcing: 모든 부분 시퀀스의 우도를 학습한다는 것은 무엇을 뜻할까?"
description: "부록 A의 ELBO·Gaussian 손실·부분 시퀀스 논증을 따라 두 Jensen 단계와 중간 계산을 설명하고, 원문 부호·전이·가중치의 불일치를 구분한다."
date: "2026-10-14"
publishedAt: "2026-10-14T13:58:27+09:00"
updatedAt: "2026-10-23T15:26:20+00:00"
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

Diffusion Forcing(DF)의 부록 A는 토큰별 노이즈 수준을 균등하게 뽑아 오염시킨 시퀀스의 평균 로그 우도에 증거 하한(evidence lower bound, ELBO)을 세우고(Theorem A.1), Gaussian 확산에서 이를 가중 노이즈 제거 MSE로 바꾼다(Corollary A.2). 증명에 쓰인 부등식은 잠재 상태와 역확산 경로에 대한 Jensen 단계 두 개뿐이며, 저자들은 잠재 상태가 결정론적이면 등호가 정확한 변분 근사(exact variational approximation)와 동치라고 진술한다. [원문 Appendix A, Theorem A.1](https://arxiv.org/pdf/2407.01392v4#page=14) **리뷰 해석.** 인쇄된 하한은 KL 항의 부호와 역방향 전이의 방향이 정의와 반대이지만 고치면 같은 가중 MSE 목적이 나오므로 “학습이 ELBO의 재가중을 최적화한다”는 첫 주장은 유지되고, “모든 노이즈 패턴의 우도를 동시에 최적화한다”는 두 번째 주장은 완전한 표현력 가정에만 기댄다.

먼저 [\[1/2\] 방법·실험 편](/reviews/diffusion-forcing-methods/)을 읽는 것을 권한다. 이 글은 v4 **부록 A, pp. 14–19**를 설정→A.1 주요 결과→A.1.1 Gaussian→A.1.2 부분 시퀀스→A.2 증명의 순서로 따르며, 인쇄식과 이 리뷰의 보충 유도를 나란히 둔다.

## A. 이론적 정당화: 대상과 가정 (Theoretical Justification)

$\mathbf x_t^0$는 깨끗한 토큰, $\mathbf x_t^k$는 노이즈 수준 $k$의 토큰, $T$는 시퀀스 길이, $K$는 최대 노이즈 수준이다.

전방 과정은 고정된 마르코프 연쇄다. 식 (A.1)은 다음과 같다.

$$
q(\mathbf x^{1:K}\mid\mathbf x^0)
=\prod_{k=1}^{K}q(\mathbf x^k\mid\mathbf x^{k-1}).
$$

모델 $p_\theta$는 잠재 상태 $\mathbf z_{t-1}$이 과거에 대한 충분 통계량(sufficient statistic)이라는 **마르코프 성질**(Markov property)을 가정한다.

$$
\begin{aligned}
&p_\theta(\mathbf z_t,\mathbf x_t^{k_t}\mid
\mathbf z_{1:t-1},(\mathbf x_s^{k_s})_{s<t})\\
&\qquad=p_\theta(\mathbf z_t,\mathbf x_t^{k_t}\mid
\mathbf z_{t-1}).
\end{aligned}
$$

**결정론적 잠재 상태**(deterministic latents)는 $p_\theta(\mathbf z_t\mid\mathbf z_{1:t-1},(\mathbf x_s^{k_s})_{s<t},\mathbf x_t^{k_t})$가 디랙 델타(Dirac delta)라는 뜻이다. 1편에서 본 RNN 구현의 잠재 상태가 이 경우다.

Remark 1은 이때 확률모형이 성립하려면 잠재 상태를 노이즈 이력별 변수 $\mathbf z_t(k_{1:t})$의 모음으로 봐야 한다고 지적한다. 노이즈 이력이 다른 두 조건에 같은 변수를 쓰면 그 조건부 분포가 정의되지 않아 $p_\theta$가 결합 확률측도가 되지 않기 때문이다. 이후 표기에서는 이 의존성을 생략한다. [식 (A.1)–(A.3)·Remark 1, p. 14](https://arxiv.org/pdf/2407.01392v4#page=14)

## A.1 주요 결과: Theorem A.1 (Main Results)

Theorem A.1은 깨끗한 시퀀스 $\mathbf x_{1:T}^0$를 고정하고, 토큰별 노이즈 수준을 균등하게 뽑아 전방 과정으로 오염시킨 시퀀스의 **평균 로그 우도**(expected log-likelihood)를 다룬다. 진술은 $k_t$를 $[K]$에서 뽑는다고 쓰지만 증명은 $\{0,\ldots,K\}$를 쓰므로 이 글은 뒤의 규약으로 읽는다. $\mathbb E_{\mathrm{forward}}$는 노이즈 수준과 전방 오염에 대한 평균, $\mathbb E_{p,\mathbf z_{1:T}}$는 그 조건에서 잠재 상태에 대한 평균이다. p. 14에 인쇄된 하한은 다음과 같다.

$$
\begin{aligned}
&\mathbb E_{\mathrm{forward}}\big[\ln p_\theta((\mathbf x_t^{k_t})_{1\le t\le T})\big]
\ge C(\mathbf x_{1:T}^0)\\
&\quad+\mathbb E_{\mathrm{forward}}\mathbb E_{p,\mathbf z_{1:T}}\Bigg[\sum_{t=1}^{T}\bigg(\frac{1}{K+1}\ln p_\theta(\mathbf x_t^0\mid\mathbf x_t^1,\mathbf z_{t-1})\\
&\qquad+\sum_{j=2}^{K}\frac{j}{K+1}\,\mathrm D_{\mathrm{KL}}\big(q(\mathbf x_t^{j-1}\mid\mathbf x_t^{j},\mathbf x_t^0)\,\Vert\,p_\theta(\mathbf x_t^{j}\mid\mathbf x_t^{j-1},\mathbf z_{t-1})\big)\bigg)\Bigg].
\end{aligned}
$$

등호 진술은 잠재 상태가 결정론적이면 $q(\mathbf x_t^{k_t+1:T}\mid\mathbf x_t^{k_t})\equiv p_\theta(\mathbf x_t^{k_t+1:T}\mid\mathbf x_t^{k_t},\mathbf z_{t-1})$일 때 그리고 그때에만 등호가 성립한다는 내용인데, 원문은 이를 “holds with inequality”로 적었다. 첨자의 상한 $T$도 노이즈 상한 $K$의 오기로 읽힌다.

**원문 안의 해명과 수정.** 증명의 마지막 식(p. 19)은 KL을 $p_\theta(\mathbf x_t^{j}\mid\mathbf x_t^{j+1},\mathbf z_{t-1})$, 계수 $\frac{j+1}{K+1}$, $j=1,\ldots,K-1$로 쓴 뒤 $j\gets j+1$로 다시 색인한다고 밝힌다. 따라서 정리 진술의 전이 방향 $\mathbf x_t^j\mid\mathbf x_t^{j-1}$은 재색인 과정의 오기이고, 계수 $\frac{j}{K+1}$는 증명과 맞는다. 부호는 증명 안에서도 해명되지 않는다. Claim 3(p. 18)이 기대 로그 비 $\mathbb E_q\log(p_\theta/q)$를 $+\mathrm D_{\mathrm{KL}}(q\Vert p_\theta)$로 적었고, 이 부호가 정리까지 이어진다. KL의 정의 $\mathbb E_q\log(p_\theta/q)=-\mathrm D_{\mathrm{KL}}(q\Vert p_\theta)$에 따라 KL 항을 다음처럼 바꾸면 표준 확산 ELBO와 같은 꼴이 된다. 여기서 $q=q(\mathbf x_t^{j-1}\mid\mathbf x_t^j,\mathbf x_t^0)$이다.

$$
+\mathrm D_{\mathrm{KL}}\big(q\,\Vert\,p_\theta(\mathbf x_t^{j}\mid\mathbf x_t^{j-1},\mathbf z_{t-1})\big)
\;\longrightarrow\;
-\mathrm D_{\mathrm{KL}}\big(q\,\Vert\,p_\theta(\mathbf x_t^{j-1}\mid\mathbf x_t^{j},\mathbf z_{t-1})\big)
$$

이 수정의 유도는 아래 A.2에서 따라간다. [Theorem A.1·증명, pp. 14–19](https://arxiv.org/pdf/2407.01392v4#page=14)

**영향과 확신 수준.** 부호와 방향은 KL과 역방향 전이의 정의만으로 정해지므로 확신이 높다. 고친 하한을 높이는 방향은 KL, 곧 가중 노이즈 제거 오차를 줄이는 방향이라 노이즈 예측 MSE를 최소화하는 본문 학습 목적과 맞고, “학습이 ELBO의 재가중을 최적화한다”는 결론은 유지된다. 등호 조건은 부호와 무관하게 증명의 두 부등식에서 나온다. 같은 부호 오류가 Corollary A.2와 Theorem A.3으로 이어지고, A.3의 일반 분포 가중치에는 별도의 오류가 있다(A.1.2).

## A.1.1 Gaussian 확산으로 구체화하기 (Specializing to Gaussian diffusion)

Corollary A.2는 Theorem A.1에 DDPM의 표준 우도·KL 계산을 적용해 **Gaussian KL을 가중 MSE로 바꾼다**. 원문은 구현에 쓰는 $\mathbf x$ 예측 형식으로 진술하며, $\boldsymbol\epsilon$·$\mathbf v$ 예측도 아핀 관계라 같은 방식으로 쓸 수 있다고 한다. p. 15에 인쇄된 결과는 다음과 같다.

$$
\begin{aligned}
&\mathbb E_{\mathrm{forward}}\big[\ln p_\theta((\mathbf x_t^{k_t})_{1\le t\le T})\big]+C(\mathbf x_{1:T}^0)\\
&\quad\ge\mathbb E_{\mathrm{forward}}\mathbb E_{p,\mathbf z_{1:T}}\Bigg[\sum_{t=1}^{T}\frac{j}{K+1}\sum_{j=1}^{K}c_j\big\|\hat{\mathbf x}^0_\theta(\mathbf x_t^j,\mathbf z_{t-1},j)-\mathbf x_t^0\big\|^2\Bigg],\\
&\sigma_j^2:=\frac{(1-\alpha_j)(1-\sqrt{\bar\alpha_{j-1}})}{1-\bar\alpha_j},\qquad
c_j=\frac{(1-\alpha_j)^2\bar\alpha_{j-1}}{2\sigma^2(1-\bar\alpha_j)^2}.
\end{aligned}
$$

인쇄식에서는 양의 제곱오차가 로그 우도의 하한 쪽에 있어 Theorem A.1의 부호 오류가 그대로 이어진다. 그 밖에 계수 $j/(K+1)$가 $j$에 대한 합 바깥에 있고, 분산에는 표준 사후 분산의 $1-\bar\alpha_{j-1}$ 대신 $1-\sqrt{\bar\alpha_{j-1}}$이 있으며, $c_j$의 분모는 $\sigma_j^2$가 아니라 $\sigma^2$다. 모델 전이를 $p_\theta(\mathbf x_t^j\mid\mathbf x_t^{j+1},\mathbf z_{t-1})$로 정의한 뒤 평균 $\boldsymbol\mu_\theta(\mathbf x_t^j,\mathbf z_{t-1},j)$는 단계 $j\to j-1$의 계수로 적어 색인도 한 칸 어긋난다. 아래는 DDPM의 사후분포로 이 계산을 다시 한 **리뷰 보충 유도**다. $j\ge2$에서 표준 전방 과정의 역조건부 분포는 다음과 같다.

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

$A_j,B_j$는 긴 계수를 줄여 쓰려고 이 리뷰가 붙인 기호이며 DDPM 식 (6)–(7)에 대응한다. 원문이 참조한 Chan tutorial v1에도 관련 분산 표기 문제가 있어, 이 유도는 DDPM의 사후 분산 $\tilde\beta_j$를 기준으로 삼는다. [DDPM, §2–3.2](https://arxiv.org/pdf/2006.11239v2#page=3)

모델 평균을

$$
\boldsymbol\mu_\theta
=A_j\mathbf x_t^j+
B_j\hat{\mathbf x}_\theta^0(
\mathbf x_t^j,\mathbf z_{t-1},j)
$$

로 놓으면 알려진 사후분포 평균과의 차이는

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

분모를 $\sigma_j^2$로 읽으면 이 $c_j$는 인쇄된 계수와 같다. 서로 다른 **고정** 공분산이라면 추가 항이 붙지만 모델 평균에 관한 부분은 같은 가중 MSE다. $j=1$은 깨끗한 데이터를 재구성하는 복원(decoder) 항이라 따로 다루며, 표준 전방 사후분포의 끝점 분산 0을 나눗셈에 넣지 않는다. KL은 ELBO에서 빼므로, 노이즈 평균에서 오는 가중치를 $w_j$라 하면 고친 하한은 $\ln p_\theta\ge C-\sum_j w_jc_j\|\hat{\mathbf x}_\theta^0-\mathbf x_t^0\|^2$의 꼴이고, 하한을 높이는 방향은 MSE를 낮추는 방향이다. [Corollary A.2, p. 15](https://arxiv.org/pdf/2407.01392v4#page=15)

### 가중치를 없애도 된다는 말의 조건

Corollary A.2 뒤의 소견(remark)에서 저자들은 과거 토큰과 노이즈 수준 전체로 이루어진 이상적 잠재 상태를 두고, 함수 근사기에 대한 하한(infimum)을 취하면 각 항의 최솟값이 조건부 분산 $\mathbf{Var}_q[\mathbf x_t^0\mid(\mathbf x_s^{k_s})_{s\le t},k_{1:t}]$이 된다고 쓴다. 표현력(expressivity)이 충분하면 각 시점의 항을 따로 최적화할 수 있으므로 가중치는 최적화에 영향이 없어 버리거나 바꿔도 된다는 결론이다. 이 논증은 하한의 부호와 무관하다. [Corollary A.2 소견, p. 15](https://arxiv.org/pdf/2407.01392v4#page=15)

제곱 손실의 최적 예측은 **조건부 평균**(conditional mean)이다. 리뷰 보충으로 입력 정보를 $\mathcal I$라 두면

$$
\begin{aligned}
\mathbb E\|\mathbf x-f(\mathcal I)\|^2
&=\mathbb E\|\mathbf x-\mathbb E[\mathbf x\mid\mathcal I]\|^2\\
&\quad+\mathbb E\|\mathbb E[\mathbf x\mid\mathcal I]-f(\mathcal I)\|^2.
\end{aligned}
$$

교차항은 $\mathbb E[\mathbf x-\mathbb E(\mathbf x\mid\mathcal I)\mid\mathcal I]=0$이라 사라진다. 첫 항은 남는 불확실성, 둘째는 예측 함수의 추가 오류다. 벡터라면 첫 항은 조건부 공분산의 대각합(trace)을 평균한 값이며, 원문의 “조건부 분산”은 이것을 가리킨다.

조건별 최적 평균을 모두 표현할 수 있고 양의 가중치가 조건에 따라 정해지면, 가중치를 바꾸어도 같은 조건별 최적해를 얻을 수 있다. **유한한 공유 네트워크에서는 다르다.** 설명용으로 두 조건의 목표 0·2를 하나의 스칼라 $u$로 예측하면

$$
\underset{u}{\arg\min}\,
\{w_1u^2+w_2(u-2)^2\}
=\frac{2w_2}{w_1+w_2}.
$$

가중치가 1:1이면 1, 1:3이면 1.5다. 표현력이 부족하면 가중치가 최적해를 바꾼다. 부록 D.1이 융합 SNR 가중치(fused SNR reweighting)로 영상 예측의 수렴이 크게 빨라졌다고 보고하는 것도, 유한 모델에서 가중치가 학습에 중립적이지 않다는 점과 맞는다.

## A.1.2 모든 부분 시퀀스: Theorem A.3을 읽는 법 (Capturing all subsequences)

Theorem A.3은 노이즈 수준 시퀀스의 임의 분포 $\mathcal D$를 허용하고

$$
P_t(j\mid k_{1:t-1})
=\Pr_{\mathcal D}(k_t=j\mid k_{1:t-1})
$$

를 정의한다. 시간상 상관된 노이즈 패턴도 다루려는 것이고, 특정 고정 패턴은 $\mathcal D$가 그 패턴에 집중된 경우다. 원문은 이 정리를 A.1 증명의 간단한 수정으로 얻는다고만 하고 증명을 따로 싣지 않는다. p. 16에 인쇄된 하한의 시점별 항은 다음과 같다.

$$
\begin{aligned}
\Xi_t:=\;&P_t(1\mid k_{1:t-1})\ln p_\theta(\mathbf x_t^0\mid\mathbf x_t^1,\mathbf z_{t-1})\\
&+\sum_{j=2}^{K}jP_t(j\mid k_{1:t-1})\,\mathrm D_{\mathrm{KL}}\big(q(\mathbf x_t^{j-1}\mid\mathbf x_t^j,\mathbf x_t^0)\,\Vert\,p_\theta(\mathbf x_t^{j}\mid\mathbf x_t^{j-1},\mathbf z_{t-1})\big).
\end{aligned}
$$

KL의 부호·방향은 Theorem A.1과 같은 문제를 갖고, 가중치 $jP_t(j\mid\cdot)$에는 별도의 문제가 있다. 균등 분포에서는 $jP_t(j)=j/(K+1)$이라 A.1과 맞지만, 일반 분포에서는 누적 확률이 필요하다.

### 왜 일반 분포에서는 누적 확률이 필요한가?

A.2의 전개에서는 중간 노이즈 수준 $k_t$에서 시작해도 그보다 노이즈가 많은 역방향 단계가 함께 등장한다. 역방향 단계를 $j\to j-1$로 쓰면 해당 KL이 포함되는 조건은 $k_t\le j-1$이다. 따라서 그 가중치는

$$
\begin{aligned}
&\mathbb E_{\mathcal D}[
\mathbf1\{k_t\le j-1\}\mid k_{1:t-1}]\\
&\quad=\Pr_{\mathcal D}(k_t\le j-1\mid k_{1:t-1})\\
&\quad=\sum_{h=0}^{j-1}P_t(h\mid k_{1:t-1})
\end{aligned}
$$

가 되어야 한다. 이것은 A.2의 지시함수(indicator) 전개를 일반 분포에 그대로 적용한 **리뷰 보충 계산**이다.

균등 분포라면 끝점 $0,\ldots,j-1$의 $j$개가 포함되어 $j/(K+1)$이 되고 인쇄식과 일치한다. 일반 분포에서는 다르다. 예를 들어 $K=3$이고 $k_t=1$에 모든 확률이 있으면 $j=2$의 누적 확률은 1이지만 $2P_t(2)=0$이다. 복원 항의 계수도 노이즈 수준 0의 확률이어야 하는데 인쇄식은 $P_t(1\mid\cdot)$이므로, 끝점 규약과 함께 고쳐야 한다. 이 판정은 반례 하나로 정해지므로 확신이 높다.

**영향.** 저자들이 가장 중요하게 보는 고정 패턴(디랙 분포)에서는 $P_t(j\mid\cdot)=\mathbf1\{j=k_t\}$이다. 인쇄식대로면 KL 항이 $j=k_t$ 하나만 남고, 누적 확률로 고치면 $j\ge k_t+1$인 항이 모두 남는다. 어느 쪽이든 하한의 항은 가중치를 빼면 균등 학습 목적 항의 부분집합이므로, 아래의 결론 경로는 유지된다. [Theorem A.3·A.2, pp. 16–19](https://arxiv.org/pdf/2407.01392v4#page=16)

### “부분 시퀀스”는 어떤 경우인가?

원문은 고정 패턴에서는 학습 목적이 그 패턴 우도의 하한일 필요가 없다고 스스로 밝힌다. 대신 하한의 항들이 재가중을 빼면 학습 목적 항의 부분집합이고, A.1.1의 소견대로 표현력이 충분한 네트워크는 모든 항을 동시에 최적화할 수 있으므로, 학습 목적이 모든 노이즈 시퀀스의 우도를 최대화하는 타당한 대리 목적(surrogate)이라고 결론짓는다. $k_t\in\{0,K\}$인 패턴은 일부 토큰을 깨끗하게 두고 나머지를 거의 순수 노이즈로 만들어 가리므로, 본문 §3.2의 “모든 부분 시퀀스” 주장이 이 경우다. **리뷰 해석.** $\mathbf x_t^K$가 근사적인 순수 노이즈라는 점과 완전한 표현력 가정이 함께 있어야 이 결론이 정확한 조건부 생성으로 이어지며, 유한한 RNN이 이 이상화에 얼마나 가까운지는 원문이 측정하지 않는다.

## A.2 Theorem A.1의 증명 따라가기 (Proof of Theorem A.1)

원문은 세 Claim을 쓴다. 잠재 상태를 펼치고, 노이즈 경로에 ELBO를 적용하고, 그 하한의 기대값을 계산한다. 아래는 이 순서를 유지하며 부호와 조건부 분포를 대조한 해설이다.

### Claim 1: 잠재 상태를 펼치는 첫 번째 Jensen 부등식

고정된 노이즈 이력에 대해 결합 모델을 적분하면

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

로그는 오목 함수이므로 $\log\mathbb E[Y]\ge\mathbb E[\log Y]$다.

$$
\log p_\theta((\mathbf x_t^{k_t})_{t=1}^{T})
\ge
\mathbb E_{p,\mathbf z_{1:T}}
\left[\sum_{t=1}^{T}
\log p_\theta(\mathbf x_t^{k_t}\mid\mathbf z_{t-1})\right].
$$

원문 Claim 1대로, 결정론적 잠재 상태에서는 잠재 상태 적분에 경로 하나만 남아 이 단계가 등호가 된다. 확률적 잠재 상태에서도 곱이 거의 확실하게 상수면 등호가 성립하므로, 결정론은 이 단계 등호의 충분조건이다. [Claim 1·식 (A.9), pp. 16–17](https://arxiv.org/pdf/2407.01392v4#page=16)

### Claim 2: 노이즈 경로를 넣는 두 번째 Jensen 부등식

시간 $t$를 고정하고 $h=k_t$라고 쓰자. $h$는 이 해설의 보조 기호다. 관찰한 $\mathbf x_t^h$의 우도를 얻으려면 노이즈가 더 많은 경로 $\mathbf x_t^{h+1:K}$를 적분한다. 전방 조건부 분포를 $r_h=q(\mathbf x_t^{h+1:K}\mid\mathbf x_t^h)$로 두면

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

분모 분포의 지지집합(support)이 모델 경로를 덮고 적분이 존재해야 한다. 등호는 중요도 비(importance ratio)가 상수일 때, 곧 $r_h$가 모델의 사후분포 $p_\theta(\mathbf x_t^{h+1:K}\mid\mathbf x_t^h,\mathbf z_{t-1})$와 같을 때이며, 원문의 등호 조건 $q\equiv p_\theta$가 이것이다.

**리뷰 보충: 이 부등식의 간극.** 오른쪽 기대값을 이 문단에서만 $\mathcal B_h$라고 부르면, 정규화·지지집합·적분 조건 아래

$$
\log p_\theta(\mathbf x_t^h\mid\mathbf z_{t-1})-\mathcal B_h
=\mathrm D_{\mathrm{KL}}\!\left(r_h\,\Vert\,p_\theta(\mathbf x_t^{h+1:K}\mid\mathbf x_t^h,\mathbf z_{t-1})\right)\geq0.
$$

즉 간극은 두 경로 분포의 KL이다. 결정론적 잠재 상태는 Claim 1을 등호로 만들고, Claim 2의 등호는 정확한 변분 근사가 정한다. Theorem A.1의 “결정론적이면 등호 ⇔ 정확한 변분 근사”는 이 두 단계를 합친 진술이다.

깨끗한 데이터에서 오염시킨 $\mathbf x_t^h$에 대해 다시 평균하면 원문의 전방 기대값을 얻는다. 원문은 분모의 $r_h$를 $q(\mathbf x_t^{h+1:K}\mid\mathbf x_t^0)$로 바꾸고, 그 차이를 $\mathbf x_t^0$와 $k_t$에만 의존하는 상수 $C_1$로 묶는다. [Claim 2, p. 17](https://arxiv.org/pdf/2407.01392v4#page=17)

### Claim 3, Part 1: Bayes 항등식과 망원합 소거

원문 식 (A.10)은 전방 전이를 역조건부 분포로 바꾸는 Bayes 항등식이다.

$$
q(\mathbf x_t^j\mid\mathbf x_t^{j-1},\mathbf x_t^0)
=q(\mathbf x_t^{j-1}\mid\mathbf x_t^j,\mathbf x_t^0)
\frac{q(\mathbf x_t^j\mid\mathbf x_t^0)}
{q(\mathbf x_t^{j-1}\mid\mathbf x_t^0)}.
$$

조건부 분포는 모두 같은 전방 과정에서 나온 것이다. 주변 밀도(marginal density)의 비율을 여러 $j$에 걸쳐 곱하면 중간항이 소거된다.

$$
\prod_{j=h+2}^{K}
\frac{q(\mathbf x_t^j\mid\mathbf x_t^0)}
{q(\mathbf x_t^{j-1}\mid\mathbf x_t^0)}
=\frac{q(\mathbf x_t^K\mid\mathbf x_t^0)}
{q(\mathbf x_t^{h+1}\mid\mathbf x_t^0)}.
$$

이 **망원합 소거**(telescoping cancellation)가 경로 전체의 비율을 경계 항과 역방향 단계별 비교로 바꾼다. $h=0$의 복원 항은 밀도 KL과 따로 다룬다. 원문은 순수 노이즈의 기저 분포 $p(\mathbf x_t^K\mid\mathbf z_{t-1})$를 매개변수화하지 않으므로 이 경계 항을 상수로 본다. 아래 보충 유도도 이 기저 분포가 매개변수와 잠재 상태 모두에 무관하게 고정되어 있다고 둔다. [Claim 3, pp. 17–18](https://arxiv.org/pdf/2407.01392v4#page=18)

### Claim 3, Part 2: 기대값은 KL의 음수다

임의의 정규화된 분포 $q,p$에 대해

$$
\mathbb E_q\log\frac{p}{q}
=-\mathbb E_q\log\frac{q}{p}
=-\mathrm D_{\mathrm{KL}}(q\Vert p).
$$

이를 고정된 $\mathbf x_t^j$의 역조건부 분포에 적용한 뒤 $\mathbf x_t^j$에 대해 평균한다. 단일 표본의 로그 비 자체는 KL이 아니다. 원문 p. 18은 이 기대값을 양의 KL로 적었고, A.1에서 본 정리의 부호 오류가 여기서 시작된다.

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

고정 노이즈 수준 $h$에서 얻는 **보충 유도의 형태**는

$$
\begin{aligned}
&\mathbb E_{q(\mathbf x_t^h\mid\mathbf x_t^0)}
\log p_\theta(\mathbf x_t^h\mid\mathbf z_{t-1})\\
&\quad\ge C_h+\mathbf1\{h=0\}R_t\\
&\qquad-\sum_{j=2}^{K}\mathbf1\{h\le j-1\}D_{t,j}.
\end{aligned}
$$

$C_h$는 고정된 전방 과정과 기저 분포에서 나오는, 매개변수와 무관한 경계·엔트로피 항을 묶는다. $h=K$에서는 역방향 단계가 없어 기저 분포만 남는다. 이 식은 인쇄된 Claim 3을 Jensen·Bayes·KL 정의에 맞춰 고친 형태다.

### 마무리: 균등 노이즈 평균과 전체 시퀀스

$h$를 $\{0,\ldots,K\}$에서 균등하게 뽑으면

$$
\begin{aligned}
\Pr(h=0)&=\frac1{K+1},\\
\Pr(h\le j-1)&=\frac{j}{K+1}.
\end{aligned}
$$

원문처럼 $\mathbb E_{<t}$를 노이즈가 섞인 과거와 그 잠재 상태에 대한 평균으로 쓰면 보충 유도는 다음 형태를 갖는다. A.1의 인쇄식에서 KL 항을 고친 형태를 $R_t$, $D_{t,j}$로 쓴 것이다.

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

현재 시점의 노이즈 수준이 과거와 독립이므로 과거에 대한 평균을 유지한 채 이 계수를 계산할 수 있다. 복원 항은 높이고 역방향 단계의 KL은 낮추는 하한이며, 앞의 Gaussian 계산을 대입하면 **가중 노이즈 제거 MSE를 최소화**하는 학습 문제가 된다.

일반 $\mathcal D$에서는 계수를 $P_t(0\mid k_{1:t-1})$와 $\Pr(k_t\le j-1\mid k_{1:t-1})$로 바꾸며, 이것이 A.1.2에서 쓴 누적 확률이다.

## 리뷰 해석과 남은 질문

**리뷰 해석.** 증명의 뼈대는 표준 확산 ELBO 유도를 토큰마다 다른 시작 노이즈 수준으로 넓힌 것이다. 두 Jensen 단계, Bayes 항등식과 망원합 소거, 균등 노이즈 평균의 계수 계산은 위에서 따라간 대로 성립한다. 인쇄식의 오류는 KL 부호, 재색인, Gaussian 분산, 일반 분포 가중치에 몰려 있어 고쳐도 학습 목적은 바뀌지 않는다. 약한 고리는 가중치 무관성이다. 이 단계는 모든 조건부 평균을 표현하는 네트워크에서만 성립하며, A.1.1의 스칼라 예처럼 유한한 공유 모델에서는 가중치가 해를 바꾼다.

남은 질문은 두 가지다. 실제 RNN에서 학습 중 드물게 나오는 고정 패턴(예: 1편의 안정화에 쓰는 작은 노이즈 이력)의 노이즈 제거 오차는 흔한 패턴과 얼마나 다른가? Theorem A.3의 누적 확률 가중치를 특정 샘플링 스케줄에 맞춰 학습에 쓰면 그 스케줄의 성능이 달라지는가?

## 출처와 읽은 범위

- 원문: *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion* · Boyuan Chen et al. · 최초 공개 2024-07-01 · [검토 버전](https://arxiv.org/abs/2407.01392v4) · 검토일 2026-10-23.
- 읽은 범위: v4 부록 A, pp. 14–19의 설정→A.1→A.1.1→A.1.2→A.2와, 인용한 본문 §3.2·부록 D.1.
- 참고한 외부 원전: [DDPM v2](https://arxiv.org/abs/2006.11239v2) §2–3.2, [Chan tutorial v1](https://arxiv.org/abs/2403.18103v1) §2.6–2.7.
- 그림 출처: 대표 이미지는 저자 v4의 Figure 2(CC BY 4.0)이며, PDF 래스터화·WebP 변환으로 해상도·인코딩을 바꾸고 내용·색·화살표·기호를 유지한다. [원문](https://arxiv.org/abs/2407.01392v4).
- 확인하지 않은 것: 정리 전체의 형식 검증, 저자 확인, 원문이 생략한 Theorem A.3의 증명.
