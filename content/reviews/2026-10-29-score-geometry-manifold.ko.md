---
title: "[1/4] Score Geometry: 기하와 빈도"
description: "작은 잡음에서 기하와 빈도 정보의 척도가 갈라지는 이유, 다섯 정리와 원·타원 및 이미지 생성 실험을 설명한다."
date: "2026-10-29"
publishedAt: "2026-10-29T08:55:43+09:00"
updatedAt: "2026-10-29T14:52:51+09:00"
topics: ["생성 모델", "확률과 기하"]
visibility: "public"
lang: "ko"
translationKey: "score-geometry-manifold"
paperTitle: "When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis"
paperPublishedDate: "2025-09-29"
authors: ["Xiang Li", "Zebang Shen", "Ya-Ping Hsieh", "Niao He"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2509.24912v2"
thumbnail: "/assets/score-geometry/paper-figure-1.svg"
thumbnailAlt: "논문 Fig.1: 분포 복원과 다양체 기하 학습, 균등 탐색 목표 비교"
---

작은 잡음의 점수 함수(score function)는 데이터 다양체로 돌아오는 기하 신호를 $\sigma^{-2}$ 척도로, 그 위의 빈도 신호를 $O(1)$ 척도로 담는다. Li 등의 *When Scores Learn Geometry*는 이 차이를 이용해 score에 $\sigma^\alpha$를 곱하는 조절된 score Langevin(Tempered Score Langevin, TS Langevin)을 제시한다. [원문 정리 3.1·5.1–5.2, §7](https://arxiv.org/html/2509.24912v2#S3)

<strong>리뷰 해석.</strong> 기하와 빈도를 분리한 정리는 균등 탐색이라는 별도 목표를 설계할 근거가 되며, 대규모 이미지 결과는 그 가능성을 보여 주는 초기 증거다.

이 시리즈의 1편이다. [[2/4] Score Geometry: Laplace와 오차 분리](/reviews/score-geometry-laplace/) · [[3/4] Score Geometry: 정상분포와 혼합](/reviews/score-geometry-wkb/) · [[4/4] Score Geometry: 균등 극한 증명](/reviews/score-geometry-weak-proof/).

## 1. 서론과 관련 연구 (Introduction)

데이터가 $\mathbb R^d$ 전체를 채우지 않고 $n<d$차원 다양체(manifold) $\mathcal M$에 놓인다고 하자. 예를 들어 원 둘레 위의 점들은 평면 좌표 두 개를 가지지만, 원을 따라 움직일 자유도는 하나다. 원의 모양을 맞히는 것과 어느 각도에서 데이터가 많이 관찰되는지 맞히는 것은 다른 일이다.

데이터의 확률을 나타내는 측도(measure)를 $\mu_{\mathrm{data}}$라 하고 Gaussian 잡음을 더하는 평활화(smoothing)를 적용하면 주변 공간에 매끄러운 밀도 $p_\sigma$가 생긴다. 점수 함수(score function)는 $s^*(x,\sigma)=\nabla\log p_\sigma(x)$다. 저자들은 작은 잡음에서 이 함수가 “다양체 쪽으로 돌아오라”는 큰 신호와 “다양체 위의 더 빈번한 위치로 이동하라”는 작은 신호를 함께 담는다고 본다.

기존 분포 복원 이론은 보통 두 신호를 함께 맞추려 한다. 이 논문은 이를 나누어, 먼저 기하를 배우고 원래 밀도의 영향을 약화시키면 균등한 탐색이 가능하다고 제안한다. 구체적인 알고리즘은 잡음 세기 $\sigma$에서 score에 $\sigma^\alpha$를 곱하는 <strong>Tempered Score Langevin(TS Langevin)</strong>이다.

![원문 Fig.1의 기존 분포 학습과 기하 학습 비교](/assets/score-geometry/paper-figure-1.svg)

원문 Fig.1은 분포가 놓이는 범위를 나타내는 지지집합(support)을 복원하는 오차 척도와 세밀한 빈도를 복원하는 척도를 나란히 놓는다. 오른쪽은 원래 빈도를 지우고 균등 부피분포를 표적으로 삼는 제안이다. [원문 Fig.1](https://arxiv.org/html/2509.24912v2#S1.F1)

### 1.1 관련 연구 (Related Work)

저자들은 관련 연구를 네 흐름으로 나누어 이 논문의 위치를 설명한다.

- <strong>분포 복원 이론.</strong> 정확한 score를 전제로 목표 분포에 수렴하는 이론과 차원 의존성을 다룬다. 기존의 오차 상계(upper bound), 즉 오차가 이 값을 넘지 않는다는 보장은 기하와 밀도를 한꺼번에 비교하며, 두 정보의 서로 다른 척도를 분리하지 않는다는 것이 저자들의 대비다.
- <strong>다양체를 검출하는 확산 모델.</strong> 원문은 Stanczuk 등의 내재 차원 추정, Ventura 등의 선형 부분공간 분석, Pavlova 등의 경험적 근거, Pidstrigach의 높은 정확도 아래 집중 조건을 구별한다. 저자들은 이들과 달리 잡음에 따른 오차율을 분리하고 균등 표본까지 목표로 한다.
- <strong>score의 점근적 특성(asymptotic behavior of the score).</strong> Lu·Lyu·Liu 등의 선행 분석이 잡음 세기가 0에 가까워질 때 나타나는 score의 특이성과 최근접점으로 향하는 사영(projection) 방향을 설명한다고 정리한다. 저자들은 크기가 일정한 상수 이내로 제한된 나머지(bounded remainder)에 포함된 데이터 밀도 항을 명시적으로 분리하는 것이 정확도 비교에 필요하다고 주장한다.
- <strong>다양체 균등 표본화.</strong> Coifman·Hein 등의 그래프 정규화는 표본 밀도를 상쇄하여 Laplace–Beltrami 연산자를 얻는다. 저자들은 대규모 주변 공간 생성으로의 확장 문제를 지적하고, De Santi 등의 균등 탐색을 위한 미세 조정과 대비하여 추론 단계에서만 조절하는 접근을 제안한다.

이 비교는 대상 논문의 관련 연구 기술을 따른다. [원문 §1.1의 네 소절](https://arxiv.org/html/2509.24912v2#S1.SS1)

Laplace 근사의 원전과 비가역 Wentzel–Kramers–Brillouin(WKB) 분석은 각각 [2편](/reviews/score-geometry-laplace/)과 [3편](/reviews/score-geometry-wkb/)에서 적용 조건과 함께 설명한다.

## 2. 공간·밀도와 기본 기호 (Preliminaries and Notation)

### 다양체 가설과 좌표

<strong>가정 2.1.</strong> $\mathcal M\subset\mathbb R^d$는 콤팩트(compact)하고 경계가 없고 주변 공간에 매끄럽게 들어 있는 $C^4$ 부분다양체(embedded submanifold)이고 차원은 $n$이다. 데이터 측도는 그 위에 놓인다.

콤팩트성은 유한 개의 좌표로 전체를 덮고, 필요한 미분계수·곡률의 상수를 균일하게 제어하는 데 쓰인다. 경계가 없다는 조건은 나중에 부피 적분의 경계항과 정상 확률 흐름을 없애는 데 중요하다. $C^4$라는 매끄러움 조건(regularity)은 4차까지의 도함수가 연속임을 뜻하며, score 전개의 더 정밀한 Laplace 오차를 얻을 때 사용한다.

곡면의 일부를 평면 좌표로 표현하는 국소 좌표(chart)를 $\Phi:U\subset\mathbb R^n\to\mathcal M$라 하자. 접공간(tangent space)은 $D\Phi(u)$의 열벡터들이 생성하는 공간이고, 법공간(normal space)은 접공간의 모든 벡터에 수직인 벡터들의 공간, 즉 직교 여공간(orthogonal complement)이다. 리만 계량(Riemannian metric)은 다음 $n\times n$ 행렬이다.

$$
g(u)=D\Phi(u)^\top D\Phi(u),\qquad
d\mathcal M=\sqrt{\det g(u)}\,du.
$$

$du$는 좌표에서의 평평한 부피이고 $d\mathcal M$은 실제 곡면의 부피다. 같은 좌표 간격이라도 곡면에서의 길이는 달라질 수 있어 $\sqrt{\det g}$라는 야코비안(Jacobian) 보정이 필요하다.

<strong>가정 2.2.</strong> 원문의 $p_{\mathrm{data}}(u)$는 $du$에 대한 밀도이며 $C^1$이고 모든 점에서 0보다 크다(strictly positive). 이 해설에서는 부피에 대한 밀도를 나타내기 위해 보조 기호 $\rho$를 사용한다.

$$
d\mu_{\mathrm{data}}=p_{\mathrm{data}}(u)\,du
=\rho(\Phi(u))\,d\mathcal M,\qquad
\rho(\Phi(u))=\frac{p_{\mathrm{data}}(u)}{\sqrt{\det g(u)}}.
$$

따라서 “균등분포”의 의미는 $p_{\mathrm{data}}(u)$가 상수가 되는 것이 아니라 $\rho$가 상수가 되는 것이다.

<strong>설명용 타원.</strong> $\Phi(u)=(a\cos u,b\sin u)$이면 다음이 성립한다.

$$
\sqrt{\det g(u)}=\|\Phi'(u)\|
=\sqrt{a^2\sin^2u+b^2\cos^2u},\qquad
p_{\mathrm{unif}}(u)=
\frac{\sqrt{a^2\sin^2u+b^2\cos^2u}}
{\int_0^{2\pi}\sqrt{a^2\sin^2v+b^2\cos^2v}\,dv}.
$$

이 식은 원의 경우에만 각도에 대한 상수가 된다. 논문의 타원 실험을 볼 때도 각도 히스토그램이 평평한지보다 <strong>호 길이(arc length)에 대한 균등성</strong>을 보아야 한다.

다양체를 얇은 관처럼 둘러싼 영역을 <strong>튜브형 이웃(tubular neighborhood)</strong>이라 부른다. 여기서는 각 점에서 가장 가까운 다양체 위의 점이 유일하게 정해지도록 충분히 작은 영역을 사용한다. 그 점으로 보내는 연산이 최근접점 사영(nearest-point projection)이다.

<details class="review-background">
<summary><span>보충 설명: 튜브형 이웃과 최근접점 사영</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

이 영역을 $T_{\mathcal M}(\epsilon)$, 최근접점을 $P_{\mathcal M}(x)$로 쓰자. 서로 수직이고 길이가 1인 법선 벡터들로 이루어진 정규직교 프레임(orthonormal frame) $N(u)\in\mathbb R^{d\times(d-n)}$을 국소적으로 택하면

$$
x=\Phi(u,r)=\Phi(u)+N(u)r,\qquad
P_{\mathcal M}(x)=\Phi(u),\qquad
d_{\mathcal M}(x)=\frac12\|r\|^2.
$$

여기서 $r$는 법선 좌표다. 사영이 유일한 범위를 벗어나면 거리 함수가 매끄럽지 않을 수 있다. 예를 들어 원의 중심에서는 모든 원 위의 점이 같은 거리이므로 최근접점이 하나가 아니다.

최근접 사영의 미분은 접공간으로 향하므로

$$
\nabla d_{\mathcal M}(x)
=(I-DP_{\mathcal M}(x)^\top)(x-P_{\mathcal M}(x))
=x-P_{\mathcal M}(x).
$$

두 번째 등식은 $x-P_{\mathcal M}(x)$가 법선이고 $DP_{\mathcal M}(x)$의 상이 접공간에 놓이기 때문이다. 이것이 “score의 큰 항이 사영 방향을 알려준다”는 말의 정확한 기하적 근거다. 매끄러움 조건은 [Leobacher–Steinicke, Theorem 2 및 Theorem C, Portable Document Format(PDF) pp.15–16](https://arxiv.org/pdf/1811.10578#page=15)에서 확인할 수 있다.

차원 표기에 주의할 부분도 있다. $D_u\Phi$로 만든 $g$는 $n\times n$인 반면, 튜브형 이웃의 전체 야코비안 $J=D_{(u,r)}\Phi$로 만든 $G=J^\top J$는 $d\times d$다. 원문 부록 A의 축약 표기를 이 해설에서는 이 둘로 나누며, 다양체 위에서는 $G(u,0)=\operatorname{diag}(g(u),I)$다.

</div>
</details>

### 2.1 Gaussian 평활화(Gaussian smoothing): 분산 폭증과 분산 보존

분산 폭증(variance exploding, VE)은 $X_\sigma=Y+\sigma Z$, 분산 보존(variance preserving, VP)은 $X_\sigma=\sqrt{1-\sigma^2}Y+\sigma Z$다. $Y\sim\mu_{\mathrm{data}}$, $Z\sim\mathcal N(0,I_d)$이고, VP에서는 $0<\sigma<1$이다. 통합 표기 $\gamma(\sigma)=1$ 또는 $\sqrt{1-\sigma^2}$를 쓰면 원문 Eq.(3)은 다음이다.

$$
p_\sigma(x)=\frac1{(2\pi\sigma^2)^{d/2}}
\int_U
\exp\left[-\frac{\|x-\gamma(\sigma)\Phi(u)\|^2}{2\sigma^2}\right]
p_{\mathrm{data}}(u)\,du.
$$

여러 좌표가 필요한 경우에는 겹치는 영역마다 가중치 함수를 두고 그 합이 1이 되게 하는 단위 분할(partition of unity)로 적분을 합친다. 서로 겹치는 좌표 패치를 단순히 더해 확률을 중복 계산한다는 뜻은 아니다.

$\mu_\sigma$는 주변 공간에서 밀도를 갖지만 $\mu_{\mathrm{data}}$는 $n<d$차원에 놓여 주변 공간의 Lebesgue 밀도를 갖지 않는다. 따라서 $\mu_\sigma\Rightarrow\mu_{\mathrm{data}}$라는 <strong>약수렴(weak convergence)</strong>을 사용한다. 여기서 유계(bounded) 함수란 절댓값이 어떤 유한한 상수를 넘지 않는 함수를 말한다.

$$
\mu_\sigma\Rightarrow\mu
\quad\Longleftrightarrow\quad
\int h\,d\mu_\sigma\to\int h\,d\mu
\quad\text{모든 유계 연속 함수 }h.
$$

다양체 위의 고정된 점 $y$에서 밀도 $p_\sigma(y)$가 $\sigma^{-(d-n)}$ 크기로 발산한다는 것은, **잡음 폭 $\sigma$가 0으로 줄어들수록 그 점의 밀도 높이가 대략 $\sigma^{-(d-n)}$에 비례하여 커진다**는 뜻이다. 여기서 $q=d-n$은 다양체에 수직인 방향의 개수다. 위의 분산 폭증 평활화를 예로 들면, 양의 부피 밀도 $\rho(y)$에 대해 다음과 같이 정확히 쓸 수 있다.

$$
\lim_{\sigma\to0}\sigma^q p_\sigma(y)
=\frac{\rho(y)}{(2\pi)^{q/2}}>0,
\qquad q=d-n,\quad y\in\mathcal M.
$$

따라서 작은 $\sigma$를 절반으로 줄이면 같은 점의 밀도 높이는 대략 $2^q$배가 된다. 평면 위의 곡선처럼 $d=2$, $n=1$이면 대략 두 배다. 이는 **확률 자체가 커진다는 말이 아니다.** 밀도는 단위 부피당 확률을 나타내는 값이며 1보다 클 수 있다. 동시에 법선 방향의 폭은 $\sigma$에 비례해 줄어든다. 그 $q$차원 부피가 $\sigma^q$ 규모로 줄어들기 때문에, 높이가 커져도 밀도를 적분한 전체 확률은 1로 유지된다. 이 점별 밀도 변화와 약수렴은 서로 양립한다. [원문 §2.1의 밀도 정의](https://arxiv.org/html/2509.24912v2#S2.SS1), [Theorem B.2를 다양체 위에 대입한 결과](https://arxiv.org/html/2509.24912v2#A2.Thmtheorem2).

<details class="review-background">
<summary><span>보충 설명: 밀도의 높이와 확률은 왜 다른가</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

가장 간단한 설명 예제로, 0에 놓인 점에 표준편차 $\sigma$의 Gaussian 잡음을 더하자. 밀도를 $\varphi_\sigma$라 쓰면

$$
\varphi_\sigma(x)=\frac{1}{\sqrt{2\pi}\sigma}
\exp\!\left(-\frac{x^2}{2\sigma^2}\right),
\qquad \int_{\mathbb R}\varphi_\sigma(x)\,dx=1.
$$

꼭대기의 높이 $\varphi_\sigma(0)=1/(\sqrt{2\pi}\sigma)$는 $\sigma=0.2$에서 약 1.995, $\sigma=0.1$에서 약 3.989다. 높이는 두 배가 되지만 그래프의 폭은 절반이 되어 전체 면적은 1로 같다. 어떤 구간의 확률은 그래프의 높이 한 개가 아니라 **그 구간 아래의 면적**이다.

$\sigma\to0$이면 이 Gaussian 확률측도는 0에 확률이 모두 놓인 점질량 $\delta_0$로 약수렴한다. 한편 0에서의 밀도 높이는 무한히 커진다. 이것이 “밀도는 발산하지만 확률측도는 약수렴한다”는 상황의 정확한 예다. 일반 다양체에서는 이 얇아지는 방향이 법선 방향에 해당한다. 위 식은 원문 Eq.(3)의 $d=1$, $n=0$인 설명 예제이며 학습 실험의 결과가 아니다. [원문 Eq.(3)](https://arxiv.org/html/2509.24912v2#S2.SS1).

</div>
</details>

두 평활화 밀도는 확산 모델의 전방 잡음 과정(forward noising process)에서 시점별 주변분포에 해당한다. 잡음 일정 $\sigma(t)$를 쓰면 역과정은 $p_{\sigma(t)}$를 거쳐 $\sigma(t)\to0$에서 원래 분포를 복원하려 한다. 고정된 잡음에서 학습 score로 Langevin을 직접 실행하거나, 역과정의 predictor 뒤에 Langevin corrector를 붙일 수도 있다. 원문의 이론은 두 평활화에 공통으로 적용된다. [원문 §2.1](https://arxiv.org/html/2509.24912v2#S2.SS1)

### 2.2 베이지안 역문제(Bayesian Inverse Problems)

베이지안 역문제는 관측 $y=A(x)+\xi$에서 관측 전의 정보와 관측값을 결합해 미지의 신호 $x$의 분포를 구하는 문제다. $x\in\mathbb R^d$, $y\in\mathbb R^m$이고 $A:\mathbb R^d\to\mathbb R^m$는 관측 연산자(measurement map), $\xi$는 $x$와 독립인 관측 잡음이다. 사전분포(prior distribution)는 관측 전의 지식, 우도(likelihood) $p(y\mid x)$는 후보 $x$가 주어진 관측을 얼마나 잘 설명하는지, 사후분포(posterior distribution)는 둘을 결합한 결과다. 우도는 $x$를 고정했을 때 $y$의 밀도이며, $x$에 대해 적분하면 1이어야 하는 함수는 아니다. [원문 §2.2](https://arxiv.org/html/2509.24912v2#S2.SS2)

관측 $y$를 고정하고 $p(y\mid x)\propto e^{-v(x;y)}$로 쓰자. $v$는 $x$와 무관한 상수를 제외한 음의 로그 우도(negative log-likelihood)다. 일반 사전측도 $\mu_{\mathrm{prior}}$에 대한 Bayes 공식은

$$
\begin{aligned}
Z(y)&=\int e^{-v(z;y)}\,d\mu_{\mathrm{prior}}(z),\qquad 0<Z(y)<\infty,\\
d\mu_y(x)&=\frac{e^{-v(x;y)}}{Z(y)}\,d\mu_{\mathrm{prior}}(x).
\end{aligned}
$$

이다. $Z(y)$는 전체 확률을 1로 맞추는 정규화 상수다. 주변 공간에 양의 매끄러운 사전밀도 $p_{\mathrm{prior}}$가 있을 때에는 사후밀도 $\pi_y$의 score가 두 항으로 분리된다.

$$
\nabla_x\log\pi_y(x)
=-\nabla_x v(x;y)+\nabla_x\log p_{\mathrm{prior}}(x).
$$

아래 보충 설명은 Gaussian 우도부터 이 분해까지 계산한다. 여기서 관측 잡음의 표준편차 $s$와 앞 절에서 데이터를 평활화한 잡음 폭 $\sigma$는 서로 다른 매개변수다.

<details class="review-background">
<summary><span>보충 설명: Gaussian 우도에서 사후 score까지</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

<strong>리뷰 보충 유도.</strong> $y$와 $s>0$를 고정하고 $x$에 대해 계산한다.

**1. 관측 잡음의 밀도를 우도로 바꾼다.** $\xi\sim\mathcal N(0,s^2I_m)$이면 $x$를 정했을 때 남는 잡음은 $y-A(x)$다. 따라서

$$
\begin{aligned}
p(y\mid x)
&=(2\pi s^2)^{-m/2}
\exp\!\left[-\frac{\|y-A(x)\|^2}{2s^2}\right],\\
-\log p(y\mid x)
&=\frac m2\log(2\pi s^2)+\frac{\|A(x)-y\|^2}{2s^2}.
\end{aligned}
$$

첫 항은 $x$와 무관하므로 사후분포를 정규화할 때 상쇄된다. 그래서 $v(x;y)=\|A(x)-y\|^2/(2s^2)$만 남긴다. 같은 관측 오차라도 $s$가 작을수록 더 큰 비용을 준다.

**2. 결합밀도를 조건부 밀도로 바꾼다.** 사전밀도가 있는 경우 $p(x,y)=p(y\mid x)p_{\mathrm{prior}}(x)$다. 이 식을 $x$에 대해 적분한 $p(y)$로 나누면

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

Gaussian 앞의 상수는 분자와 분모에서 같다. $Z(y)$는 우도의 그 상수를 제거한 정규화 상수이므로, 일반적으로 관측의 밀도 $p(y)$ 자체와 같지는 않다. 사전분포가 다양체 위에만 놓일 때에도 본문의 측도 형태는 그대로 쓸 수 있다.

**3. 로그를 취한 뒤 미분한다.** $p_{\mathrm{prior}}(x)>0$이고 필요한 미분이 존재하면

$$
\begin{aligned}
\log\pi_y(x)&=-v(x;y)+\log p_{\mathrm{prior}}(x)-\log Z(y),\\
\nabla_x\log Z(y)&=0,\\
\nabla_x\log\pi_y(x)&=-\nabla_x v(x;y)+\nabla_x\log p_{\mathrm{prior}}(x).
\end{aligned}
$$

$Z(y)$ 안의 적분 변수는 $z$다. $y$를 고정한 뒤 $x$로 미분하므로 정규화 상수의 미분이 0이 된다. 사전밀도 자체도 $p_{\mathrm{prior}}=\widetilde p/C$로만 알려져 있다면 $\nabla\log p_{\mathrm{prior}}=\nabla\log\widetilde p$여서 알 수 없는 상수 $C$가 사라진다.

**4. 관측 오차의 미분을 실제로 계산한다.** $A$가 미분 가능하고 $J_A(x)\in\mathbb R^{m\times d}$가 그 야코비안이면, 성분별 연쇄법칙(chain rule)으로

$$
\begin{aligned}
\partial_{x_j}v(x;y)
&=\frac{1}{s^2}\sum_{k=1}^m
\bigl(A_k(x)-y_k\bigr)\,\partial_{x_j}A_k(x),\\
\nabla_x v(x;y)&=\frac{1}{s^2}J_A(x)^\top\bigl(A(x)-y\bigr).
\end{aligned}
$$

$A(x)=Mx$인 선형 관측에서는 $J_A=M$이므로 관측 유도항(guidance)은 $-\nabla_xv=M^\top(y-Mx)/s^2$다. 현재 예측과 관측의 차이를 신호 공간으로 되돌리는 방향이며, 앞의 음수 부호는 관측 오차를 줄이는 쪽을 선택한다.

이 유도는 [원문 §2.2와 Eq.(4)](https://arxiv.org/html/2509.24912v2#S2.SS2)의 중간 계산을 펼친 것이다. Bayes 공식의 밀도·측도 형태는 [Kekkonen 강의노트 §1.3, Theorem 1.9, 인쇄 pp.7–9](https://www.damtp.cam.ac.uk/research/cia/files/teaching/Inverse_Problems_Lent_2018/19y_01m_16d_LectureNotes.pdf#page=8)에서 함께 확인할 수 있다.

</div>
</details>

플러그 앤 플레이(Plug-and-Play, PnP) 접근에서는 사전밀도의 상수를 모르거나 사전 표본만 가지고 있어도 학습한 사전 score $\hat s$를 넣을 수 있다. 위 사후 score를 Langevin 이동항(drift)으로 쓰고 그 사전 항을 $\hat s$로 교체하면 원문 Eq.(4)가 된다.

$$
dX_t=-\nabla_x v(X_t;y)dt+\hat s(X_t)dt+\sqrt2\,dW_t.
$$

고정된 $\sigma>0$에서 $p_{\mathrm{prior}}=p_\sigma$이고 $\hat s=\nabla\log p_\sigma$가 정확하다면, 정규화·매끄러움 등 필요한 조건 아래 목표 정상밀도는 $\pi_y\propto e^{-v}p_\sigma$다. 바로 다음 절에서 확률 흐름을 대입해 이 사실을 확인한다. 학습한 score가 근사값이면 그 동일성은 자동으로 성립하지 않는다.

사전분포가 원래 데이터 측도 $d\mu_{\mathrm{data}}=\rho\,d\mathcal M$이면 Bayes 공식의 목표는 $e^{-v}\rho\,d\mathcal M$에 비례한다. 원문의 [§6과 이 글의 정리·유도](#정리와-유도)는 사전을 부피 균등분포로 바꿔 $e^{-v}\,d\mathcal M$를 목표로 한다. [원문 Theorem 6.1](https://arxiv.org/html/2509.24912v2#S6.Thmtheorem1), [부록 B.6의 증명 연결](https://arxiv.org/html/2509.24912v2#A2.SS6), [3편의 WKB 증명 해설](/reviews/score-geometry-wkb/#b6-다양체-위-베이지안-사후분포-proof-for-theorem-61)을 함께 연결한다.

### 2.3 Langevin과 비가역 정상분포

확률미분방정식(stochastic differential equation, SDE)

$$
dX_t=b(X_t)\,dt+\sqrt2\,dW_t
$$

에서 $b$는 이동항이고 $W_t$는 $d$차원 표준 Brownian 운동(standard Brownian motion)이다. 시간 $t$에서의 밀도 $\pi_t$는 Fokker–Planck 방정식

$$
\partial_t\pi_t=-\nabla\cdot(b\pi_t)+\Delta\pi_t
$$

을 따른다. 여기서 $\nabla\cdot$는 벡터장의 발산(divergence), $\Delta=\sum_{i=1}^d\partial_{x_i}^2$는 Laplacian이다. 이때 발산은 앞서 밀도값이 무한히 커진다는 뜻으로 쓴 발산과 달리, 벡터장이 한 지점에서 퍼져 나가는 양을 나타내는 미분 연산이다. 정상분포(stationary distribution)는 밀도가 시간에 따라 변하지 않아 우변이 0인 분포다. [원문 §2.3](https://arxiv.org/html/2509.24912v2#S2.SS3)

<details class="review-background">
<summary><span>보충 설명: Itô 공식에서 Fokker–Planck 방정식까지</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

<strong>리뷰 보충 유도.</strong> 분포의 변화를 적분으로 관찰하는 시험함수(test function) $h\in C_c^\infty(\mathbb R^d)$를 잡는다. 이는 매끄럽고 충분히 먼 곳에서 0이 되는 함수다. 아래 계산에는 밀도의 미분과 적분·기댓값을 교환할 수 있는 매끄러움 및 적분 가능성을 둔다. 시험함수의 지지집합이 콤팩트하므로 부분적분의 무한대 경계항은 0이다.

**1. Brownian 잡음의 2차 효과를 계산한다.** 독립인 Brownian 성분은 이차변동(quadratic variation) 규칙 $dW_i\,dW_j=\delta_{ij}\,dt$를 따른다. $\delta_{ij}$는 $i=j$이면 1, 다르면 0이다. 이동항과의 곱은 이 규칙에서 0이므로

$$
\begin{aligned}
dX_i\,dX_j
&=(\sqrt2\,dW_i)(\sqrt2\,dW_j)
=2\delta_{ij}\,dt,\\
dh(X_t)
&=\sum_i\partial_i h(X_t)\,dX_i
+\frac12\sum_{i,j}\partial_{ij}h(X_t)\,dX_i\,dX_j\\
&=\bigl(b\cdot\nabla h+\Delta h\bigr)(X_t)\,dt
+\sqrt2\,\nabla h(X_t)\cdot dW_t.
\end{aligned}
$$

Itô 공식의 $1/2$와공분산(covariance)의 2가 곱해져 1이 된다. 그래서 $\Delta h$ 앞에 $1/2$가 남지 않는다. $dX_i\,dX_j$는 보통의 미소량을 곱한 것이 아니라 확률 과정의 이차변동을 적는 표기다.

**2. 기댓값을 취한다.** 필요한 적분 가능성 아래 Itô 확률적분의 기댓값은 0이다. 따라서 시험함수에 작용하는 생성자(generator) $L$과 평균 변화율은

$$
\begin{aligned}
Lh&=b\cdot\nabla h+\Delta h,\\
\frac{d}{dt}\mathbb E[h(X_t)]
&=\mathbb E[(Lh)(X_t)]
=\int_{\mathbb R^d}(Lh)(x)\pi_t(x)\,dx.
\end{aligned}
$$

한편 $\mathbb E[h(X_t)]=\int h\pi_t\,dx$이므로 왼쪽은 $\int h\,\partial_t\pi_t\,dx$다. 이제 오른쪽의 $h$에 붙은 미분을 밀도 쪽으로 옮긴다.

**3. 이동항은 한 번, 확산항은 두 번 부분적분한다.** 공간 적분의 영역은 모두 $\mathbb R^d$다. 각 성분에 대해

$$
\begin{aligned}
\int b_i\pi_t\,\partial_i h\,dx
&=-\int h\,\partial_i(b_i\pi_t)\,dx,\\
\int\pi_t\,\partial_{ii}h\,dx
&=-\int(\partial_i\pi_t)(\partial_i h)\,dx\\
&=\int h\,\partial_{ii}\pi_t\,dx.
\end{aligned}
$$

첫째 줄에는 음수 부호가 하나 생긴다. 둘째·셋째 줄은 두 번 옮기므로 음수 부호가 두 번 생겨 최종적으로 양수다. 모든 성분을 더하면

$$
\int h\,\partial_t\pi_t\,dx
=\int h\bigl[-\nabla\cdot(b\pi_t)+\Delta\pi_t\bigr]\,dx.
$$

이 식이 모든 시험함수 $h$에 대해 성립하므로 본문의 Fokker–Planck 방정식을 얻는다. 매끄러운 경우에는 점별 방정식으로 읽으며, 더 약한 매끄러움에서는 위 적분 등식이 약한 형태(weak form)의 정의가 된다. 밀도 쪽의 연산자 $L^*\pi=-\nabla\cdot(b\pi)+\Delta\pi$를 생성자의 수반 연산자(adjoint operator)라 부르는 이유도 $\int(Lh)\pi=\int h(L^*\pi)$라는 관계에 있다.

이는 원문 §2.3에서 쓰는 정상 방정식의 기초 유도다. Itô 공식과 수반 연산자를 통한 전방 방정식의 연결은 [Goodman 강의노트 Lesson 5, §2, pp.4–6](https://math.nyu.edu/~goodman/teaching/StochCalc2018/notes/Lesson5.pdf#page=4)에서 확인할 수 있다.

</div>
</details>

이 방정식은 확률 흐름(probability current) $j_t$를 사용해 다음처럼 쓸 수 있다.

$$
j_t=b\pi_t-\nabla\pi_t,\qquad
\partial_t\pi_t=-\nabla\cdot j_t.
$$

특히 $b=-\nabla f$이고 $\pi=Z_f^{-1}e^{-f}$가 정규화 가능한 매끄러운 밀도이면, 연쇄법칙을 직접 대입해

$$
\begin{aligned}
\nabla\pi&=-\pi\nabla f,\\
j&=(-\nabla f)\pi-(-\pi\nabla f)=0,\\
L^*\pi&=-\nabla\cdot j=0
\end{aligned}
$$

을 얻는다. 앞 절의 사후분포에는 $f=v-\log p_{\mathrm{prior}}$를 넣으면 된다. 이 계산은 목표 밀도가 정상 방정식을 만족함을 확인하며, 그 밀도로의 장시간 수렴에는 별도의 존재·유일성·수렴 조건이 필요하다.

학습된 벡터장이 gradient가 아니면 $j\ne0$이어도 $\nabla\cdot j=0$일 수 있다. 확률은 계속 순환하면서도 각 위치의 밀도는 일정할 수 있다는 뜻이다. 따라서 비가역(non-reversible) 동역학의 정상분포를 임의로 $e^{-f}$라고 놓을 수 없다. 논문이 WKB를 도입하는 이유이며, 그 전개는 [3편 부록 B.4 해설](/reviews/score-geometry-wkb/)에서 이어진다.

## 3. Gaussian 평활화는 분포보다 기하를 먼저 복원한다 (Central Insight: Gaussian Smoothing Recovers Geometry Before Distribution)

### 정리의 정확한 내용

가정 2.1·2.2 아래 충분히 작은 고정 튜브형 이웃에서 원문 Theorem 3.1은 다음 전개를 준다. $u_x=\Phi^{-1}(P_{\mathcal M}(x))$, $q=d-n$으로 쓴다.

$$
\log p_\sigma(x)=
-\frac{d_{\mathcal M}(x)}{\sigma^2}
+\log p_{\mathrm{data}}(u_x)
-\frac q2\log(2\pi\sigma^2)
+H(x)+o(1).
$$

나머지는 $x$에 대해 균일하다. $H$는 기하 정보를 담고 VE·VP에 따라 달라진다. 엄밀한 형태는 부록 Theorem B.2에 있으며,

$$
\widehat H_{ij}(u,x)
=\left\langle\partial_{ij}\Phi(u),\Phi(u)-x\right\rangle
+\left\langle\partial_i\Phi(u),\partial_j\Phi(u)\right\rangle
$$

에 대해 VE의 $H(x)=-\frac12\log\det\widehat H(u_x,x)$다. VP에는 추가로 $-\frac12\langle P_{\mathcal M}(x),x-P_{\mathcal M}(x)\rangle$가 들어간다. [원문 Theorem 3.1](https://arxiv.org/html/2509.24912v2#S3.Thmtheorem1), [Theorem B.2](https://arxiv.org/html/2509.24912v2#A2.Thmtheorem2)

이 전개의 모든 유한 보정은 $\sigma^2$를 곱하면 사라진다. 정규화 항도 $\sigma^2\log\sigma\to0$이므로

$$
\lim_{\sigma\to0}\sigma^2\log p_\sigma(x)=-d_{\mathcal M}(x)
\quad\text{튜브형 이웃에서 균일하게}.
$$

즉 재조정한 log-density에서 가장 큰 항이 정하는 극한(leading-order limit)은 원래 데이터의 빈도와 무관한 거리 함수다. [원문 §3, Eq.(7)](https://arxiv.org/html/2509.24912v2#S3)

### 어떤 항이 무엇을 학습하는가

$-\sigma^{-2}d_{\mathcal M}$는 다양체 밖의 점에 큰 비용을 매긴다. 원래 데이터가 다양체의 어느 부분에서 더 자주 나오는지는 $\log p_{\mathrm{data}}$에 들어간다. $-\frac q2\log(2\pi\sigma^2)$는 $\sigma$에 따라 발산하지만 <strong>공간 $x$와 무관한 정규화 항</strong>이므로 score를 취하면 사라진다. “밀도 항이 $O(1)$”이라는 표현은 이 공간적으로 일정한 항까지 유계라는 뜻이 아니다.

공간에 따라 달라지는 밀도 항을 지우더라도 큰 거리 항은 남길 수 있다. 바로 이 분리가 §4·5의 서로 다른 정확도 조건으로 이어진다.

<details class="review-background">
<summary><span>보충 설명: score의 실제 크기</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

아래 score 근사는 더 강한 매끄러움 조건으로 부록 B.5에서 따로 증명해야 한다. log-density의 $o(1)$을 단순 미분해서 얻는 식이 아니다.

$$
s^*(x,\sigma)
=-\frac{x-P_{\mathcal M}(x)}{\sigma^2}+O(1).
$$

법선 거리가 고정된 $\delta>0$라면 큰 항은 $\delta/\sigma^2$다. 하지만 거리 자체가 $O(\sigma)$인 전형적인 잡음 샘플에서는 $O(\sigma^{-1})$이고, 정확히 다양체 위에서는 이 항이 0이다. 따라서 $\Theta(\sigma^{-2})$라는 표현은 주로 <strong>거리 항의 계수와 고정된 튜브형 이웃의 점근 전개</strong>를 가리킨다.

또한 $\sigma^2\log p_\sigma\to-d_{\mathcal M}$는 특이 극한이다. 원문의 “Taylor expansion”을 $\sigma=0$에서 매끄러운 보통 Taylor 급수로 이해하면 안 된다. 자세한 유도는 Laplace 근사인 부록 B.2에서 한다.

</div>
</details>

## 4. 기존 생성 학습의 척도 분리 (Scale Separation in Existing Generative Learning: Geometry versus Distribution)

### 오차를 어느 함수에서 재는가

이 절은 학습된 score network 자체가 아니라 <strong>실제로 생성된 최종 밀도</strong> $\pi_\sigma$의 score를 비교한다.

$$
s_{\pi_\sigma}=\nabla\log\pi_\sigma,\qquad
E_\sigma=
\|s_{\pi_\sigma}-s^*(\cdot,\sigma)\|_{L^\infty(K)}.
$$

가정 4.1은 $-\log\pi_\sigma$가 $K$에서 $C^1$이고, 튜브형 이웃을 포함하는 콤팩트 집합 $K$에 확률이 집중하며, $K$ 안의 두 점을 잇는 경로 길이를 공통 상수로 제어할 수 있다고 요구한다.

$L^\infty$ 오차는 $K$에서의 최악 오차다. 학습 샘플 평균의 제곱오차가 작다는 $L^2$ 조건보다 강하다. 또한 reverse SDE의 매 단계 학습 오차에서 위 최종 밀도 오차를 구하는 정리는 여기서 제공하지 않는다. 이 차이는 원문 §8도 한계로 밝힌다.

### Theorem 4.1 세 결론

1. $E_\sigma=o(\sigma^{-2})$이면 모든 $\delta>0$에 대해 $\pi_\sigma(\operatorname{dist}(x,\mathcal M)>\delta)\to0$이다.
2. 다양체 위의 임의의 $C^1$ 목표 밀도로 수렴하면서도 score 오차가 사라지지 않는 구성을 만들 수 있다. 원문의 표기는 $E_\sigma=\Omega(1)$이다.
3. $E_\sigma=o(1)$이면 $\pi_\sigma$는 원래 데이터 측도로 약수렴한다.

[원문 Theorem 4.1](https://arxiv.org/html/2509.24912v2#S4.Thmtheorem1)

이 가운데 1번은 <strong>질량의 집중</strong>이지 다양체의 모든 위치에 양의 질량을 고르게 주는 결론이 아니다. 하나의 작은 부분으로 몰려도 다양체 가까이에 있다는 조건은 충족할 수 있다. 2번의 $\Omega(1)$은 하한 표기이므로 그 자체가 “오차가 상수 이하”라는 뜻도 아니다.

### 증명의 핵심 도구

$\Delta_\sigma=\log\pi_\sigma-\log p_\sigma$라 두면 $\nabla\Delta_\sigma$가 score 오차다. 길이 $L_K$ 이하의 경로를 따라 적분하면

$$
|\Delta_\sigma(x)-\Delta_\sigma(y)|\le L_KE_\sigma.
$$

즉, 미분 오차가 log-density ratio의 <strong>변동 폭(oscillation)</strong>을 제어한다. 여기서 변동 폭은 함수의 상한에서 하한을 뺀 값이다. 여기에 두 밀도의 정규화와 $K$에서의 질량 집중을 사용해야 더해지는 상수(additive constant)의 값이 정해진다. 그 결과 작은 상대 오차이면 원래 밀도를 복원하고, $\sigma^{-2}$보다만 작은 오차이면 거리 장벽을 보존한다. 전체 정규화 계산과 집중 증명은 부록 B.3에 있다.

$o(1)$은 원래 분포 복원의 충분조건이다. 밀도는 수렴해도 그 미분은 수렴하지 않는 예와, 이를 주변 공간의 가정에 맞춘 구성은 [2편의 정리 4.1 증명](/reviews/score-geometry-laplace/)에서 설명한다.

## 5. 기하 학습과 균등분포 복원 (New Paradigm of Geometric Learning: Recover Uniform Distributions with $o(\sigma^{-2})$ Score Error)

### 왜 score만 약하게 만드는가

원문의 Eq.(8)은 다음과 같다.

$$
dX_t=\sigma^\alpha s(X_t,\sigma)\,dt+\sqrt2\,dW_t.
$$

잡음 계수는 고정하고 drift만 줄인다. 이것은 시간을 느리게 바꾸는 것과 다르다. 시간만 바꾸면 drift와 Brownian 항을 함께 바꾸어야 하며 정상분포는 변하지 않는다.

만약 $s=-\nabla f_\sigma$이면 정상분포는

$$
\widetilde\pi_\sigma(x)\propto e^{-\sigma^\alpha f_\sigma(x)}.
$$

정확한 score이면 이는 $p_\sigma(x)^{\sigma^\alpha}$를 정규화한 것이다. log-density 전개를 넣으면

$$
\log\widetilde\pi_\sigma(x)
=-\frac{d_{\mathcal M}(x)}{\sigma^{2-\alpha}}
+\sigma^\alpha\log\rho(P_{\mathcal M}(x))
+\text{작아지는 기하 보정}+\text{정규화 상수}.
$$

$0<\alpha<2$이면 밀도 항은 사라지지만 거리 항은 계속 발산한다. 기하를 남기면서 원래 빈도를 지우는 원리다. [원문 §5](https://arxiv.org/html/2509.24912v2#S5)

### Theorem 5.1 gradient score

가정 2.1·2.2와, $\pi_\sigma$ 대신 $\widetilde\pi_\sigma$에 적용한 가정 4.1을 둔다. score oracle은 gradient이고, 어떤 <strong>고정된</strong> $\beta>-2$에 대해

$$
\|s(\cdot,\sigma)-s^*(\cdot,\sigma)\|_{L^\infty(K)}
=o(\sigma^\beta)
$$

라고 하자. 이때

$$
\max\{-\beta,0\}<\alpha<2
$$

이면 정상분포는 $\mathcal M$의 리만 부피에 대한 균등 확률측도로 약수렴한다. 좌표 밀도로 쓰면 $\widetilde\pi(u)\propto\sqrt{\det g(u)}$다.

두 부등식은 서로 다른 일을 한다. $\alpha<2$는 법선 집중을 남기며, $\alpha+\beta>0$는 학습 오차의 효과를 사라지게 하고, $\alpha>0$는 원래 데이터 밀도의 효과를 지운다. 예컨대 $E_\sigma=O(1)$이면 $\beta=-1/2$를 택할 수 있으므로 $\alpha=1$이 조건을 만족한다. $E_\sigma=o(\sigma^{-3/2})$이면 $3/2<\alpha<2$가 정리가 허용하는 범위다. [Theorem 5.1](https://arxiv.org/html/2509.24912v2#S5.Thmtheorem1)

법선 방향 분산은 대략 $\theta=\sigma^{2-\alpha}$이며 표준편차는 $\sigma^{1-\alpha/2}$다. $\alpha=1$일 때는 $\sqrt{\sigma}$다. 표적 빈도를 지우는 대가로 같은 작은 $\sigma$에서 원래 smoothing보다 더 넓은 튜브형 이웃을 탐색한다.

![별도 annulus toy에서 온도 조절에 따른 법선 폭과 접선 밀도](/assets/score-geometry/tempering-toy.svg)

<strong>설명용 그림.</strong> 분리 가능한 toy 밀도 $q_\sigma(u,r)\propto\exp[-r^2/(2\sigma^2)+4\cos u]$에 tempering을 적용했다. 왼쪽은 leading 법선 표준편차 $\sigma^{1-\alpha/2}$, 오른쪽은 $\sigma=0.01$에서 정규화한 접선 밀도 $\propto e^{4\sigma^\alpha\cos u}$다. 원 논문의 그림이나 학습 실험을 재현한 것이 아니며, 두 척도의 분리를 시각화한 것이다.

이 정리는 어떤 고정 지수 $\beta>-2$에서의 여유를 요구한다. 초록의 더 넓은 $o(\sigma^{-2})$ 표현과의 경계, 가변 조절 계수를 사용하는 보충 유도는 [3편의 부록 B.5 해설](/reviews/score-geometry-wkb/)에 둔다.

### Theorem 5.2 일반 비gradient score

원래 $p_{\mathrm{data}}$에 $C^2$ 매끄러움 조건을 더하고 같은 score 오차·$\alpha,\beta$ 조건을 둔다. 추가로 SDE가 유일한 정상분포를 가지며, $\theta=\sigma^{2-\alpha}$에서 <strong>국소 WKB 가정 B.2</strong>를 만족한다고 가정한다. 그때 Theorem 5.1과 같은 균등 부피 극한을 얻는다는 것이 Theorem 5.2다.

WKB 가정은 단순한 계산 편의가 아니다. 튜브형 이웃으로의 확률 집중, 지수항 앞에 곱해지는 양의 $C^2$ 계수 함수(prefactor)와 그 함수의 $C^2$ 수렴, 선택된 action 해의 유일성을 포함한다. 이 논문은 이런 가정들이 모든 학습 score network에서 성립한다고 증명하지 않는다. 부록 B.4에서 그 가정 아래의 계수 결정 과정을 따라간다. [Theorem 5.2](https://arxiv.org/html/2509.24912v2#S5.Thmtheorem2), [가정 B.2](https://arxiv.org/html/2509.24912v2#A2.Thmassumption2)

또한 각각의 $\sigma$에서 먼저 정상상태를 얻고 나서 $\sigma\to0$를 취한다. 고정된 수의 corrector step으로 $\sigma$를 계속 바꾸는 실용 알고리즘의 분포가 이 극한과 같다는 결론은 별도의 문제다.

## 6. 역문제에서 더 강건한 균등 사전분포 (Uniform Prior is More Robust For Bayesian Inverse Problems)

### 목표가 어떻게 달라지는가

앞 절의 Bayes 공식에 사전분포를 대입하자. 관측 $y$를 고정하고 $v(x;y)$를 $v(x)$로 줄여 쓰면, 두 사후분포는 정확히 다음과 같다.

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

첫째는 데이터가 자주 나타난 위치에 $\rho$만큼 더 큰 사전 가중치를 둔다. 둘째는 다양체의 같은 부피에 같은 사전 확률을 준다. TS는 둘째를 목표로 하므로, 원래 데이터 prior의 posterior와 목표 분포가 달라진다. [원문 §6과 Theorem 6.1](https://arxiv.org/html/2509.24912v2#S6)

부피 균등분포는 부피측도를 기준으로 한 미분 엔트로피(differential entropy)를 최대화한다. 이를 Kullback–Leibler 발산(Kullback–Leibler divergence, KL)으로 계산하자. $V_{\mathcal M}=\int_{\mathcal M}d\mathcal M$라 하면

$$
-\int_{\mathcal M}\rho\log\rho\,d\mathcal M
=\log V_{\mathcal M}
-D_{\mathrm{KL}}\!\left(\rho\,d\mathcal M\;\middle\|\;
\frac{d\mathcal M}{V_{\mathcal M}}\right)
\le\log V_{\mathcal M}.
$$

이 등식과 부등식의 계산은 아래에 둔다. “최대 엔트로피 prior”는 기준측도를 다양체 부피로 택했을 때의 표현이다.

<details class="review-background">
<summary><span>보충 설명: 균등분포의 최대 엔트로피를 직접 계산하기</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

<strong>리뷰 보충 유도.</strong> 부피 기준 균등 밀도를 $u_{\mathcal M}=1/V_{\mathcal M}$라 쓰자. 이미 정의한 $\rho$와 $u_{\mathcal M}$는 둘 다 $d\mathcal M$에 대해 적분하면 1이다. 여기서는 두 밀도가 양수이고 아래 적분이 유한한 경우를 생각한다.

**1. 발산의 로그를 전개한다.** 같은 기준측도에 대한 두 밀도의 KL 발산을 현재 경우에 쓰면

$$
\begin{aligned}
D_{\mathrm{KL}}(\rho\|u_{\mathcal M})
&=\int_{\mathcal M}\rho\log\frac{\rho}{u_{\mathcal M}}\,d\mathcal M\\
&=\int_{\mathcal M}\rho\bigl(\log\rho+\log V_{\mathcal M}\bigr)\,d\mathcal M\\
&=\int_{\mathcal M}\rho\log\rho\,d\mathcal M+\log V_{\mathcal M}.
\end{aligned}
$$

마지막 줄에서는 상수 $\log V_{\mathcal M}$를 적분 밖으로 꺼내고 $\int\rho\,d\mathcal M=1$을 썼다. 항을 옮기면 본문의 엔트로피 등식이다.

**2. 왜 KL 발산이 0 이상인지 보인다.** $t=\rho/u_{\mathcal M}>0$라 두자. 함수 $G(t)=t\log t-t+1$은

$$
G'(t)=\log t,\qquad G''(t)=\frac1t>0,\qquad
G(t)\ge G(1)=0
$$

을 만족한다. 즉 $t\log t\ge t-1$이다. 이 부등식에 양의 밀도 $u_{\mathcal M}$를 곱해 적분하면

$$
\begin{aligned}
D_{\mathrm{KL}}(\rho\|u_{\mathcal M})
&=\int_{\mathcal M}u_{\mathcal M}\,t\log t\,d\mathcal M\\
&\ge\int_{\mathcal M}u_{\mathcal M}(t-1)\,d\mathcal M\\
&=\int_{\mathcal M}\rho\,d\mathcal M
-\int_{\mathcal M}u_{\mathcal M}\,d\mathcal M=0.
\end{aligned}
$$

등호는 $t=1$, 즉 두 밀도가 거의 모든 곳에서 같을 때만 성립한다. 따라서 엔트로피의 최댓값은 $\log V_{\mathcal M}$이고 부피 균등분포가 이를 달성한다. 이는 두 분포의 비율에 관한 기초 계산을 리뷰에서 덧붙인 것이며, 원문의 새로운 정리가 아니다. 좌표 $u$에 대한 균등 밀도와는 다르며, 둘의 차이는 [앞 절의 계량·부피 보정](#다양체-가설과-좌표)에서 설명했다.

</div>
</details>

### 정리와 유도

Theorem 5.2와 같은 조건에 더해 $v$는 $\mathbb R^d$ 전체에서 유계이고 튜브형 이웃에서 $C^1$이라고 하자. 원문 Theorem 6.1의 gradient 표기는

$$
dX_t=-\nabla v(X_t)\,dt-\sigma^\alpha\nabla f_\sigma(X_t)\,dt+\sqrt2\,dW_t
$$

이며, 일반 oracle 형태로는 prior 항을 $\sigma^\alpha s$로 읽는다. 이 <strong>guided 동역학에 필요한 정상분포·WKB 조건 아래</strong> 극한 좌표 밀도는

$$
\pi_{\mathrm{post}}(u)
=\frac{e^{-v(\Phi(u))}\sqrt{\det g(u)}}
{\int_U e^{-v(\Phi(w))}\sqrt{\det g(w)}\,dw}.
$$

이다. 아래는 그 결론을 gradient인 경우에 정규화까지 직접 연결하는 <strong>리뷰 보충 유도</strong>다. 일반적인 비gradient 경우의 원문 증명은 뒤의 링크로 구별한다.

**1. 관측항이 없을 때의 정상분포를 분리한다.** 정규화 가능한 밀도 $\nu_\sigma$를 아래처럼 정의한다. Theorem 5.1의 조건 아래에서는 그 확률측도가 부피 균등분포로 약수렴한다.

$$
\nu_\sigma(dx)
=\frac{e^{-\sigma^\alpha f_\sigma(x)}\,dx}
{\int_{\mathbb R^d}e^{-\sigma^\alpha f_\sigma(z)}\,dz}
\ \Rightarrow\ \frac{d\mathcal M}{V_{\mathcal M}}.
$$

관측 유도항을 더한 drift는 $-\nabla(v+\sigma^\alpha f_\sigma)$다. §2.3의 확률 흐름 계산에 $f=v+\sigma^\alpha f_\sigma$를 대입하면 guided 정상측도 $\Pi_\sigma$는 정확히

$$
\Pi_\sigma(dx)
=\frac{e^{-v(x)}\nu_\sigma(dx)}
{\int_{\mathbb R^d}e^{-v(z)}\nu_\sigma(dz)}
$$

가 된다. 분자와 분모에는 같은 사전 정규화 상수가 들어 있으므로 상쇄된다.

**2. 분자와 분모의 극한을 따로 계산한다.** $h$를 임의의 유계 연속 함수라 하면, 약수렴은 다음 적분의 극한을 준다.

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

원문은 $v$의 연속성을 튜브형 이웃에서만 가정하므로 그 점을 확인해야 한다. 다양체를 포함하는 작은 이웃 $N$에서 1이고 튜브 바깥에서 0인 매끄러운 함수 $\chi\in[0,1]$를 택하면 $\widetilde v=\chi v$는 전역 유계 연속 함수로 확장된다. $v=\widetilde v$인 $N$ 바깥의 차이는

$$
\left|\int h(e^{-v}-e^{-\widetilde v})\,d\nu_\sigma\right|
\le2\|h\|_\infty e^{\|v\|_\infty}\nu_\sigma(N^c)
\longrightarrow0
$$

이다. 극한 측도는 $\mathcal M\subset N$에 놓이고 $N^c$는 닫혀 있으므로 그 질량이 0으로 간다. 따라서 연속 확장에 약수렴을 적용한 뒤 원래 $v$로 돌아올 수 있다. 분모는 같은 계산에서 $h=1$로 두면 된다.

**3. 두 극한의 비를 취하고 좌표를 바꾼다.** 극한 분모가 양수이므로

$$
\int h\,d\Pi_\sigma
=\frac{\int he^{-v}\,d\nu_\sigma}{\int e^{-v}\,d\nu_\sigma}
\longrightarrow
\frac{\int_{\mathcal M}he^{-v}\,d\mathcal M}
{\int_{\mathcal M}e^{-v}\,d\mathcal M}.
$$

공통 상수 $1/V_{\mathcal M}$가 상쇄된다. 이것이 $d\mu_y^{\mathrm{unif}}$로의 약수렴이다. 마지막으로 $x=\Phi(u)$, $d\mathcal M=\sqrt{\det g(u)}\,du$를 대입하면 위의 좌표 밀도 $\pi_{\mathrm{post}}$가 나온다. 하나의 좌표로 덮이지 않는 경우에는 앞 절과 같이 단위 분할을 사용한다.

**일반 score와 원문 증명의 연결.** 비gradient score에서는 guided 정상측도를 단순히 $e^{-v}\nu_\sigma$로 쓸 수 없다. 원문은 [부록 B.6](https://arxiv.org/html/2509.24912v2#A2.SS6)에서 [B.4의 Theorem B.3](https://arxiv.org/html/2509.24912v2#A2.Thmtheorem3)에 $f_1=v$를 넣고 [B.1의 Theorem B.1](https://arxiv.org/html/2509.24912v2#A2.Thmtheorem1)을 적용한다. 재조정된 이동항에서 관측항이 $-\theta\nabla v$가 되는 계산, 진폭 $e^{-v}$와 법선 Hessian의 역할은 [3편의 부록 B.6 해설](/reviews/score-geometry-wkb/#b6-다양체-위-베이지안-사후분포-proof-for-theorem-61)에 연결한다. 일반 score의 정상밀도를 정하는 WKB 계산은 그 글의 B.4에 있다.

가정의 범위도 구별해야 한다. 흔한 Gaussian likelihood의 이차함수 $v$는 보통 전역 유계가 아니다. 콤팩트 다양체 위에서 유계라는 사실만으로 원문의 전역 가정이 그대로 성립하지 않는다. 전역 꼬리·집중에 대한 추가 분석이나 바깥에서의 절단을 따로 확인해야 한다.

### Classifier Free Guidance와의 연결

분류기 없는 유도(classifier-free guidance, CFG)는 unconditional score와 conditional increment를 합친다.

$$
s_{\mathrm{CFG}}=s_{\mathrm{uncond}}
+w(s_{\mathrm{cond}}-s_{\mathrm{uncond}}).
$$

TS corrector에서는 아래처럼 <strong>unconditional 부분만</strong> 약하게 만든다.

$$
\widetilde s_{\mathrm{CFG}}
=\sigma^\alpha s_{\mathrm{uncond}}
+w(s_{\mathrm{cond}}-s_{\mathrm{uncond}}).
$$

likelihood에 해당하는 두 번째 항은 $O(1)$로 남아야 한다. 전체 CFG score를 통째로 $\sigma^\alpha$배하면 유도 항까지 지워져 같은 posterior 결론이 나오지 않는다. 실제 CFG의 벡터장과 그 미분·잡음 의존성이 정리의 고정 $v$ 가정을 만족하는지는 또 확인해야 한다. 원문의 Stable Diffusion 실험은 이 연결을 사용한 개념 검증이다.

## 7. 실험 (Experiments)

### 7.1 원과 타원

원문은 원과 타원의 각도 좌표에서 von Mises 데이터 분포를 만들고, 학습 score로 일반 Langevin과 $\alpha=1$인 TS를 비교한다. Fig.2에서는 일반 Langevin이 원래 빈도를 제대로 재현하지 못하는 반면 TS가 부피 균등 분포에 가까워지는 모습을 보인다. 튜브형 이웃의 법선 집중과 접선 빈도를 함께 보아야 하며, 타원에서는 각도 균등을 정답으로 삼지 않는다.

von Mises 분포는 원 위의 집중 정도를 조절하는 분포로 $p(u)\propto e^{\kappa\cos(u-\mu)}$다. $\mu$는 선호 각도, $\kappa\ge0$는 집중도다. $\kappa=0$이면 각도 균등이고 클수록 $\mu$ 주변에 모인다. 정규화 상수는 $2\pi I_0(\kappa)$이며 $I_0(\kappa)=(2\pi)^{-1}\int_0^{2\pi}e^{\kappa\cos u}\,du$는 0차 수정 Bessel 함수다. 이름을 몰라도 이 적분 정의로 실험의 밀도를 이해할 수 있다.

부록 C.3은 진짜 score에 $\sigma$에 대해 $O(1)$인 고정 오차를 더하는 통제 실험도 제공한다. 이때 원래 분포 복원은 왜곡되어도 TS가 균등 원 분포를 얻는다는 Fig.4를 제시한다. 이는 유용한 특정 예이며 최악 오차 정리의 모든 경우를 실험으로 증명하는 것은 아니다.

![원문 Fig.2의 학습 score 실험: 타원·원 각각에서 일반 Langevin과 조절된 score 비교](/assets/score-geometry/paper-figure-2.svg)

원문 Fig.2. 네 패널은 타원의 일반 Langevin·조절된 score, 원의 일반 Langevin·조절된 score 순서다. 가로축은 각도이고, 타원에서 부피 균등 표적의 각도 밀도는 평평하지 않다. [원문 Fig.2](https://arxiv.org/html/2509.24912v2#S7.F2)

### 7.2 Stable Diffusion 1.5

세 prompts는 가구·자동차·건축이다. 설정당 512개 이미지를 생성하고 $256\times256$으로 줄여 대조적 언어·이미지 사전학습(Contrastive Language–Image Pre-training, CLIP) 특징을 평가한다. CLIP의 이미지·텍스트 embedding을 같은 특징 공간에서 비교한다. 기본 guidance는 7.5, predictor는 30단계다. 예측–보정(predictor–corrector, PC)과 TS 모두 마지막에 같은 수의 결정적 projection step을 추가한다.

- <strong>프롬프트 유사도(prompt similarity, P-sim):</strong> 이미지와 prompt의 평균 특징 cosine similarity. 높을수록 prompt 정렬이 좋다.
- <strong>이미지 간 유사도(image-to-image similarity, I-sim):</strong> 같은 prompt로 생성한 이미지들 사이의 평균 pairwise 특징 similarity. 낮을수록 이 특징 공간에서 이미지가 다양하다.

높은 P-sim은 사실성·시각적 품질 전체를 인증하지 않으며, 낮은 I-sim도 실제 데이터 다양체의 부피 균등성을 증명하지 않는다. 표의 점수는 원문 단위대로 유지한다.

기준 샘플러인 잡음 제거 확산 확률 모델(Denoising Diffusion Probabilistic Models, DDPM)을 비교에 사용한다.

| 원문 Table 1 | 가구 P-sim ↑ | 가구 I-sim ↓ | 자동차 P-sim ↑ | 자동차 I-sim ↓ | 건축 P-sim ↑ | 건축 I-sim ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DDPM | 29.56 | 80.78 | 26.23 | 87.30 | 27.36 | 81.53 |
| PC | 29.40 | 81.24 | 26.30 | 87.20 | 27.13 | 81.03 |
| TS | 30.20 | 80.76 | 26.62 | 87.14 | 27.32 | 80.76 |

Table 1은 corrector 수와 $\alpha$를 조정한 최상 결과다. 같은 행의 두 지표가 <strong>하나의 동일한 설정에서 함께 나온 값인지</strong>는 표에 표시되지 않는다. 예를 들어 Table 2에서 가구 PC의 최고 P-sim은 5 correctors에서, 최저 I-sim은 10 correctors에서 나온다. 따라서 이 표의 지표 쌍만으로 하나의 설정이 두 축을 동시에 지배한다고 결론내리기보다, 고정 설정의 비교를 함께 보아야 한다.

$\alpha=1$을 고정한 Table 2에서는 3 prompts×5 corrector 수의 15개 비교 모두에서 P-sim이 올라가고, I-sim은 15개 중 14개에서 낮아진다. 자동차·20 correctors에서는 88.06에서 88.07로 약간 나빠진다. Table 3–4도 작은 $\alpha$나 특정 지표에서 악화가 있으며, 더 큰 $\alpha$가 두 지표를 항상 함께 개선하지는 않는다. 전체 숫자는 부록 C에 정리했다. 반복 seed별 오차·유의성 검정은 보고되지 않아 작은 차이를 강한 결론으로 바꾸지 않는다. [원문 §7·부록 C](https://arxiv.org/html/2509.24912v2#S7)

![원문 Fig.3의 동일 조건 생성 이미지 비교. 위는 예측–보정, 아래는 조절된 score](/assets/score-geometry/paper-figure-3.webp)

원문 Fig.3. 위 행은 PC, 아래 행은 TS이며 같은 열은 prompt·corrector 횟수·무작위 seed가 같다. 저자들은 아래 행의 사실성과 세부 묘사가 더 좋다고 평가한다. 이 정성 예시는 앞의 특징 유사도 표와 함께 보는 비교 자료다. [원문 Fig.3](https://arxiv.org/html/2509.24912v2#S7.F3)

## 8. 결론 (Conclusion)

이 논문의 중요한 기여는 작은 잡음에서 <strong>모양을 강제하는 큰 항</strong>과 <strong>그 모양 위의 빈도를 담는 작은 항</strong>을 분리하고, 그 차이를 이용해 다른 표적 분포를 설계한 것이다. Laplace 근사에서 나타나는 법선 Hessian과 부피 보정이 이 연결을 정량화한다.

저자들은 이 척도 분리를 이용해 원래의 빈도 복원과 기하에 따른 균등 탐색을 다른 목표로 제시한다. 균등 사전분포를 사용할 때 관측 정보는 남기고 데이터 빈도는 지운다. 원·타원과 이미지 생성 실험은 이 방향의 가능성을 확인한다. [원문 §8](https://arxiv.org/html/2509.24912v2#S8)


저자들이 밝힌 한계는 생성 경로를 따른 누적 오차를 직접 추적하지 않는다는 점, 최악 오차 조건을 평균제곱형 score 학습 목적과 연결할 필요, score 추정의 표본 복잡도를 정리로 옮길 필요, 균등 표본의 정확도를 정량화하지 않았다는 점, 그리고 최신 대규모 확산 모델까지 검증하지 않았다는 점이다. [원문 §8, Limitations and future work](https://arxiv.org/html/2509.24912v2#S8)

## 리뷰 해석과 남은 질문

<strong>리뷰 해석.</strong> 학습 데이터의 양의 밀도가 다양체 전체를 덮는다는 가정 아래, 강한 법선 신호를 남기면서 유한한 접선 편향을 약화시키는 전략은 일관적이다. 원·타원 실험은 기하와 빈도를 직접 비교하고, 이미지 실험은 정렬과 다양성의 동시 개선 가능성을 확인한다.

정리의 극한은 각 잡음에서의 정상분포에 관한 것이다. 학습 오차가 유한 predictor–corrector 경로 전체에 어떻게 전파되는지와 이미지 특징 유사도가 실제 다양체 부피 균등성을 얼마나 반영하는지는 별도 질문이다. 원래 prior의 posterior와 균등 부피 prior의 posterior도 서로 다른 표적이다. 증명 조건과 세부 검토는 [2편](/reviews/score-geometry-laplace/)과 [3편](/reviews/score-geometry-wkb/)에 둔다.

## 출처와 읽은 범위

- 원문: Xiang Li, Zebang Shen, Ya-Ping Hsieh, Niao He, *When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis*. 최초 공개 2025-09-29, 검토 버전 v2(2026-03-16), 검토일 2026-10-29. [공식 원문](https://arxiv.org/abs/2509.24912v2).
- 읽은 범위: 본문 §1–8, Fig.1–3과 Table 1. 실험의 전체 표와 설정은 [3편](/reviews/score-geometry-wkb/)에서 다룬다.
- 보충 설명의 기초 출처: [Kekkonen §1.3, 인쇄 pp.7–9](https://www.damtp.cam.ac.uk/research/cia/files/teaching/Inverse_Problems_Lent_2018/19y_01m_16d_LectureNotes.pdf#page=8), [Goodman Lesson 5 §2, pp.4–6](https://math.nyu.edu/~goodman/teaching/StochCalc2018/notes/Lesson5.pdf#page=4). 해당 부분은 Bayes 공식과 Itô·전방 방정식 연결에 사용한다.
- 그림: 시리즈 썸네일은 원문 Fig.1을 재사용한다. 원문 Fig.1–3은 해당 결과의 해설을 위한 부분 인용이며, 저자·버전은 위 원문과 같다. 그림 영역을 벡터로 추출하며 Fig.1 내부의 래스터만 원래 픽셀 크기를 유지한 WebP(품질 95)로 바꾼다. Fig.3은 긴 변 1,500픽셀의 WebP(품질 92)로 변환한다. 원문에는 arXiv 영구 비독점 공개 라이선스가 표시되어 있다. 별도 예제 그림은 명시한 모형의 자체 계산이다.
- 확인하지 않은 것: 신경망 학습, Stable Diffusion 이미지 생성의 독립 재현과 정리의 형식 검증.
