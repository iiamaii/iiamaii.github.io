---
title: "[1/4] Score Geometry: 기하와 빈도"
description: "작은 잡음에서 기하와 빈도 정보의 척도가 갈라지는 이유, 다섯 정리와 원·타원 및 이미지 생성 실험을 설명한다."
date: "2026-10-29"
publishedAt: "2026-10-29T08:55:43+09:00"
updatedAt: "2026-10-29T09:31:36+09:00"
topics: ["생성 모델", "확률과 기하"]
visibility: "public"
lang: "ko"
translationKey: "score-geometry-manifold"
paperTitle: "When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis"
paperPublishedDate: "2025-09-29"
authors: ["Xiang Li", "Zebang Shen", "Ya-Ping Hsieh", "Niao He"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2509.24912v2"
thumbnail: "/assets/score-geometry/cover-1.svg"
thumbnailAlt: "기하·분포 학습 시리즈 1편: 기하를 배우는 score와 확률을 배우는 score"
---

작은 잡음의 점수 함수(score function)는 데이터 다양체로 돌아오는 기하 신호를 $\sigma^{-2}$ 척도로, 그 위의 빈도 신호를 $O(1)$ 척도로 담는다. Li 등의 *When Scores Learn Geometry*는 이 차이를 이용해 score에 $\sigma^\alpha$를 곱하는 조절된 score Langevin(Tempered Score Langevin, TS Langevin)을 제시한다. [원문 정리 3.1·5.1–5.2, §7](https://arxiv.org/html/2509.24912v2#S3)

<strong>리뷰 해석.</strong> 기하와 빈도를 분리한 정리는 균등 탐색이라는 별도 목표를 설계할 근거가 되며, 대규모 이미지 결과는 그 가능성을 보여 주는 초기 증거다.

이 시리즈의 1편이다. [[2/4] Score Geometry: Laplace와 오차 분리](/reviews/score-geometry-laplace/) · [[3/4] Score Geometry: 정상분포와 혼합](/reviews/score-geometry-wkb/) · [[4/4] Score Geometry: 균등 극한 증명](/reviews/score-geometry-weak-proof/).

## 1. 서론과 관련 연구 (Introduction)

데이터가 $\mathbb R^d$ 전체를 채우지 않고 $n<d$차원 다양체(manifold) $\mathcal M$에 놓인다고 하자. 예를 들어 원 둘레 위의 점들은 평면 좌표 두 개를 가지지만, 원을 따라 움직일 자유도는 하나다. 원의 모양을 맞히는 것과 어느 각도에서 데이터가 많이 관찰되는지 맞히는 것은 다른 일이다.

원래 데이터 측도(measure)를 $\mu_{\mathrm{data}}$라 하고 Gaussian 잡음을 더하면 주변 공간에 매끄러운 밀도 $p_\sigma$가 생긴다. 점수 함수(score function)는 $s^*(x,\sigma)=\nabla\log p_\sigma(x)$다. 저자들은 작은 잡음에서 이 함수가 “다양체 쪽으로 돌아오라”는 큰 신호와 “다양체 위의 더 빈번한 위치로 이동하라”는 작은 신호를 함께 담는다고 본다.

기존 분포 복원 이론은 보통 두 신호를 함께 맞추려 한다. 이 논문은 이를 나누어, 먼저 기하를 배우고 원래 밀도의 영향을 약화시키면 균등한 탐색이 가능하다고 제안한다. 구체적인 알고리즘은 잡음 세기 $\sigma$에서 score에 $\sigma^\alpha$를 곱하는 <strong>Tempered Score Langevin(TS Langevin)</strong>이다.

![원문 Fig.1의 기존 분포 학습과 기하 학습 비교](/assets/score-geometry/paper-figure-1.svg)

원문 Fig.1은 지지집합을 복원하는 오차 척도와 세밀한 빈도를 복원하는 척도를 나란히 놓는다. 오른쪽은 원래 빈도를 지우고 균등 부피분포를 표적으로 삼는 제안이다. [원문 Fig.1](https://arxiv.org/html/2509.24912v2#S1.F1)

### 1.1 관련 연구 (Related Work)

저자들은 관련 연구를 네 흐름으로 나누어 이 논문의 위치를 설명한다.

- <strong>분포 복원 이론.</strong> 정확한 score를 전제로 목표 분포에 수렴하는 이론과 차원 의존성을 다룬다. 기존 오차 상계는 기하와 밀도를 한꺼번에 비교하며, 두 정보의 서로 다른 척도를 분리하지 않는다는 것이 저자들의 대비다.
- <strong>다양체를 검출하는 확산 모델.</strong> 원문은 Stanczuk 등의 내재 차원 추정, Ventura 등의 선형 부분공간 분석, Pavlova 등의 경험적 근거, Pidstrigach의 높은 정확도 아래 집중 조건을 구별한다. 저자들은 이들과 달리 잡음에 따른 오차율을 분리하고 균등 표본까지 목표로 한다.
- <strong>score의 점근 거동.</strong> Lu·Lyu·Liu 등의 선행 분석이 작은 잡음에서의 특이성과 사영 방향을 설명한다고 정리한다. 저자들은 유계 나머지에 포함된 데이터 밀도 항을 명시적으로 분리하는 것이 정확도 비교에 필요하다고 주장한다.
- <strong>다양체 균등 표본화.</strong> Coifman·Hein 등의 그래프 정규화는 표본 밀도를 상쇄하여 Laplace–Beltrami 연산자를 얻는다. 저자들은 대규모 주변 공간 생성으로의 확장 문제를 지적하고, De Santi 등의 균등 탐색을 위한 미세 조정과 대비하여 추론 단계에서만 조절하는 접근을 제안한다.

이 비교는 대상 논문의 관련 연구 기술을 따른다. [원문 §1.1의 네 소절](https://arxiv.org/html/2509.24912v2#S1.SS1)

Laplace 근사의 원전과 비가역 Wentzel–Kramers–Brillouin(WKB) 분석은 각각 [2편](/reviews/score-geometry-laplace/)과 [3편](/reviews/score-geometry-wkb/)에서 적용 조건과 함께 설명한다.

## 2. 공간·밀도와 기본 기호 (Preliminaries and Notation)

### 다양체 가설과 좌표

<strong>가정 2.1.</strong> $\mathcal M\subset\mathbb R^d$는 콤팩트(compact), 경계가 없는 $C^4$ 매장 부분다양체(embedded submanifold)이고 차원은 $n$이다. 데이터 측도는 그 위에 놓인다.

콤팩트성은 유한 개의 좌표로 전체를 덮고, 필요한 미분계수·곡률의 상수를 균일하게 제어하는 데 쓰인다. 경계가 없다는 조건은 나중에 부피 적분의 경계항과 정상 확률 흐름을 없애는 데 중요하다. $C^4$ 정규성은 특히 score 전개의 더 정밀한 Laplace 오차를 얻을 때 사용한다.

국소 좌표사상(chart)을 $\Phi:U\subset\mathbb R^n\to\mathcal M$라 하자. 접공간(tangent space)은 $D\Phi(u)$의 열벡터들이 생성하는 공간이고, 법공간(normal space)은 그 직교 여공간이다. 리만 계량(Riemannian metric)은 다음 $n\times n$ 행렬이다.

$$
g(u)=D\Phi(u)^\top D\Phi(u),\qquad
d\mathcal M=\sqrt{\det g(u)}\,du.
$$

$du$는 좌표에서의 평평한 부피이고 $d\mathcal M$은 실제 곡면의 부피다. 같은 좌표 간격이라도 곡면에서의 길이는 달라질 수 있어 $\sqrt{\det g}$라는 야코비안(Jacobian) 보정이 필요하다.

<strong>가정 2.2.</strong> 원문의 $p_{\mathrm{data}}(u)$는 $du$에 대한 밀도이며 $C^1$이고 엄밀히 양수다. 이 해설에서는 부피에 대한 밀도를 나타내기 위해 보조 기호 $\rho$를 사용한다.

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

<details class="review-background">
<summary><span>보충 설명: 관상근방과 최근접 사영</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

충분히 가까운 점 $x$는 다양체 위의 유일한 최근접점 $P_{\mathcal M}(x)$를 갖는다. 이런 영역이 <strong>관상근방(tubular neighborhood)</strong> $T_{\mathcal M}(\epsilon)$이다. 국소적으로 법선 벡터들의 정규직교 프레임 $N(u)\in\mathbb R^{d\times(d-n)}$을 택하면

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

두 번째 등식은 $x-P_{\mathcal M}(x)$가 법선이고 $DP_{\mathcal M}(x)$의 상이 접공간에 놓이기 때문이다. 이것이 “score의 큰 항이 사영 방향을 알려준다”는 말의 정확한 기하적 근거다. 정규성은 [Leobacher–Steinicke, Theorem 2 및 Theorem C, Portable Document Format(PDF) pp.15–16](https://arxiv.org/pdf/1811.10578#page=15)에서 확인할 수 있다.

차원 표기에 주의할 부분도 있다. $D_u\Phi$로 만든 $g$는 $n\times n$인 반면, 관상근방의 전체 야코비안 $J=D_{(u,r)}\Phi$로 만든 $G=J^\top J$는 $d\times d$다. 원문 부록 A의 축약 표기를 이 해설에서는 이 둘로 나누며, 다양체 위에서는 $G(u,0)=\operatorname{diag}(g(u),I)$다.

</div>
</details>

### 2.1 Gaussian smoothing 분산 폭증(variance exploding, VE)과 분산 보존(variance preserving, VP)

분산 폭증(variance exploding, VE)은 $X_\sigma=Y+\sigma Z$, 분산 보존(variance preserving, VP)은 $X_\sigma=\sqrt{1-\sigma^2}Y+\sigma Z$다. $Y\sim\mu_{\mathrm{data}}$, $Z\sim\mathcal N(0,I_d)$이고, VP에서는 $0<\sigma<1$이다. 통합 표기 $\gamma(\sigma)=1$ 또는 $\sqrt{1-\sigma^2}$를 쓰면 원문 Eq.(3)은 다음이다.

$$
p_\sigma(x)=\frac1{(2\pi\sigma^2)^{d/2}}
\int_U
\exp\left[-\frac{\|x-\gamma(\sigma)\Phi(u)\|^2}{2\sigma^2}\right]
p_{\mathrm{data}}(u)\,du.
$$

여러 좌표가 필요한 경우에는 단위 분할(partition of unity)로 적분을 합친다. 서로 겹치는 좌표 패치를 단순히 더해 확률을 중복 계산한다는 뜻은 아니다.

$\mu_\sigma$는 주변 공간에서 밀도를 갖지만 $\mu_{\mathrm{data}}$는 $n<d$차원에 놓여 주변 공간의 Lebesgue 밀도를 갖지 않는다. 따라서 $\mu_\sigma\Rightarrow\mu_{\mathrm{data}}$라는 <strong>약수렴(weak convergence)</strong>을 사용한다.

$$
\mu_\sigma\Rightarrow\mu
\quad\Longleftrightarrow\quad
\int h\,d\mu_\sigma\to\int h\,d\mu
\quad\text{모든 유계 연속 함수 }h.
$$

낮은 차원 지지집합으로의 약수렴은 주변 공간의 밀도가 점별로 유한한 함수에 수렴한다는 말이 아니다. 실제로 다양체 위의 $p_\sigma$는 $\sigma^{-(d-n)}$ 크기로 발산할 수 있다. [원문 §2.1](https://arxiv.org/html/2509.24912v2#S2.SS1)

두 평활화 밀도는 확산 모델의 전방 잡음 과정(forward noising process)에서 시점별 주변분포에 해당한다. 잡음 일정 $\sigma(t)$를 쓰면 역과정은 $p_{\sigma(t)}$를 거쳐 $\sigma(t)\to0$에서 원래 분포를 복원하려 한다. 고정된 잡음에서 학습 score로 Langevin을 직접 실행하거나, 역과정의 predictor 뒤에 Langevin corrector를 붙일 수도 있다. 원문의 이론은 두 평활화에 공통으로 적용된다. [원문 §2.1](https://arxiv.org/html/2509.24912v2#S2.SS1)

### 2.2 Bayesian Inverse Problems

관측 $y$와 likelihood $p(y\mid x)\propto e^{-v(x)}$가 주어지면, 원래 prior가 $\mu_{\mathrm{data}}$일 때 posterior는 $e^{-v(x)}d\mu_{\mathrm{data}}(x)$에 비례한다. $v$는 관측에 맞지 않는 정도를 측정하는 음의 로그 likelihood다. 미분 가능한 경우 guidance는 $-\nabla v$다. 원문 §2.2는 이를 정의하고, §6에서 prior를 다양체 부피 균등분포로 바꾸었을 때의 posterior를 분석한다. 이 해설의 자세한 유도는 §6과 부록 B.6에 있다.

구체적으로 베이지안 역문제(Bayesian inverse problem)는 $y=A(x)+\xi$에서 $x$를 추정하는 문제다. Gaussian 관측 잡음이면 $v(x;y)=\|A(x)-y\|^2/(2s^2)$이고 $s$는 관측 잡음의 표준편차다. 이상적인 posterior Langevin의 drift는 $-\nabla v+\nabla\log p_{\mathrm{prior}}$다.

원문이 다루는 플러그 앤 플레이(Plug-and-Play, PnP) 접근에서는 사전분포의 정규화 상수를 모르거나 표본만 가진 상황에서도 그 score $\hat s$를 사용할 수 있다. 관측 오차 항과 사전 score를 합친 원문 Eq.(4)는

$$
dX_t=-\nabla_x v(X_t;y)dt+\hat s(X_t)dt+\sqrt2\,dW_t.
$$

이다. $\hat s$가 $p_\sigma$의 score를 근사할 때, 저자들은 기존 분포 복원의 엄격한 정확도 요구와 부피 균등 사전분포의 강건성을 대비한다. [원문 §2.2, Eq.(4)](https://arxiv.org/html/2509.24912v2#S2.SS2)

### 2.3 Langevin과 비가역 정상분포

확률미분방정식(stochastic differential equation, SDE)

$$
dX_t=b(X_t)\,dt+\sqrt2\,dW_t
$$

의 밀도는 Fokker–Planck 방정식 $\partial_t\pi=-\nabla\cdot(b\pi)+\Delta\pi$를 따른다. 정상분포(stationary distribution)는 우변이 0인 분포다.

<strong>리뷰 보충 유도.</strong> 시험함수 $h$에 Itô 공식을 적용하면 평균 변화율은 $\mathbb E[b\cdot\nabla h+\Delta h]$다. 이를 부분적분하여 미분을 밀도 쪽으로 옮기면 위 방정식이 나온다. $\sqrt2$가 확산 계수를 1로 만들기 때문에 Laplacian 앞에 $1/2$가 붙지 않는다.

Itô 공식에서 2차 미분이 사라지지 않는 이유는 Brownian 증가량이 $O(\sqrt{dt})$이기 때문이다. $dX_i\,dX_j=2\delta_{ij}dt$인 이차변동(quadratic variation)을 Taylor의 $\tfrac12\partial_{ij}h\,dX_i\,dX_j$에 넣으면 $\Delta h\,dt$가 남는다. 생성자(generator) $Lh=b\cdot\nabla h+\Delta h$와 그 밀도에 대한 수반 $L^*\pi$가 각각 시험함수·밀도의 진화를 표현한다.

$b=-\nabla f$이면 $\pi\propto e^{-f}$에서 확률 흐름 $j=b\pi-\nabla\pi$가 0이다. 반면 학습된 벡터장이 gradient가 아닐 때는 $j$가 0이 아니어도 $\nabla\cdot j=0$일 수 있다. 따라서 비가역(non-reversible) 동역학의 정상분포를 임의로 $e^{-f}$라고 쓸 수 없다. 논문이 WKB를 도입하는 이유다. [원문 §2.3](https://arxiv.org/html/2509.24912v2#S2.SS3)

## 3. Gaussian 평활화는 분포보다 기하를 먼저 복원한다 (Central Insight: Gaussian Smoothing Recovers Geometry Before Distribution)

### 정리의 정확한 내용

가정 2.1·2.2 아래 충분히 작은 고정 관상근방에서 원문 Theorem 3.1은 다음 전개를 준다. $u_x=\Phi^{-1}(P_{\mathcal M}(x))$, $q=d-n$으로 쓴다.

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
\quad\text{관상근방에서 균일하게}.
$$

즉 재조정한 log-density의 선도 극한은 원래 데이터의 빈도와 무관한 거리 함수다. [원문 §3, Eq.(7)](https://arxiv.org/html/2509.24912v2#S3)

### 어떤 항이 무엇을 학습하는가

$-\sigma^{-2}d_{\mathcal M}$는 다양체 밖의 점에 큰 비용을 매긴다. 원래 데이터가 다양체의 어느 부분에서 더 자주 나오는지는 $\log p_{\mathrm{data}}$에 들어간다. $-\frac q2\log(2\pi\sigma^2)$는 $\sigma$에 따라 발산하지만 <strong>공간 $x$와 무관한 정규화 항</strong>이므로 score를 취하면 사라진다. “밀도 항이 $O(1)$”이라는 표현은 이 공간적으로 일정한 항까지 유계라는 뜻이 아니다.

공간에 따라 달라지는 밀도 항을 지우더라도 큰 거리 항은 남길 수 있다. 바로 이 분리가 §4·5의 서로 다른 정확도 조건으로 이어진다.

<details class="review-background">
<summary><span>보충 설명: score의 실제 크기</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

아래 score 근사는 더 강한 정규성으로 부록 B.5에서 따로 증명해야 한다. log-density의 $o(1)$을 단순 미분해서 얻는 식이 아니다.

$$
s^*(x,\sigma)
=-\frac{x-P_{\mathcal M}(x)}{\sigma^2}+O(1).
$$

법선 거리가 고정된 $\delta>0$라면 큰 항은 $\delta/\sigma^2$다. 하지만 거리 자체가 $O(\sigma)$인 전형적인 잡음 샘플에서는 $O(\sigma^{-1})$이고, 정확히 다양체 위에서는 이 항이 0이다. 따라서 $\Theta(\sigma^{-2})$라는 표현은 주로 <strong>거리 항의 계수와 고정된 관상근방의 점근 전개</strong>를 가리킨다.

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

가정 4.1은 $-\log\pi_\sigma$가 $K$에서 $C^1$이고, 관상근방을 포함하는 콤팩트 집합 $K$에 확률이 집중하며, $K$ 안의 두 점을 잇는 경로 길이를 공통 상수로 제어할 수 있다고 요구한다.

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

즉, 미분 오차가 log-density ratio의 <strong>진동(oscillation)</strong>을 제어한다. 여기에 두 밀도의 정규화와 $K$에서의 질량 집중을 사용해야 알 수 없는 가산 상수가 고정된다. 그 결과 작은 상대 오차이면 원래 밀도를 복원하고, $\sigma^{-2}$보다만 작은 오차이면 거리 장벽을 보존한다. 전체 정규화 계산과 집중 증명은 부록 B.3에 있다.

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

법선 방향 분산은 대략 $\theta=\sigma^{2-\alpha}$이며 표준편차는 $\sigma^{1-\alpha/2}$다. $\alpha=1$일 때는 $\sqrt{\sigma}$다. 표적 빈도를 지우는 대가로 같은 작은 $\sigma$에서 원래 smoothing보다 더 넓은 관상근방을 탐색한다.

![별도 annulus toy에서 온도 조절에 따른 법선 폭과 접선 밀도](/assets/score-geometry/tempering-toy.svg)

<strong>설명용 그림.</strong> 분리 가능한 toy 밀도 $q_\sigma(u,r)\propto\exp[-r^2/(2\sigma^2)+4\cos u]$에 tempering을 적용했다. 왼쪽은 leading 법선 표준편차 $\sigma^{1-\alpha/2}$, 오른쪽은 $\sigma=0.01$에서 정규화한 접선 밀도 $\propto e^{4\sigma^\alpha\cos u}$다. 원 논문의 그림이나 학습 실험을 재현한 것이 아니며, 두 척도의 분리를 시각화한 것이다.

이 정리는 어떤 고정 지수 $\beta>-2$에서의 여유를 요구한다. 초록의 더 넓은 $o(\sigma^{-2})$ 표현과의 경계, 가변 조절 계수를 사용하는 보충 유도는 [3편의 부록 B.5 해설](/reviews/score-geometry-wkb/)에 둔다.

### Theorem 5.2 일반 비gradient score

원래 $p_{\mathrm{data}}$에 $C^2$ 정규성을 더하고 같은 score 오차·$\alpha,\beta$ 조건을 둔다. 추가로 SDE가 유일한 정상분포를 가지며, $\theta=\sigma^{2-\alpha}$에서 <strong>국소 WKB 가정 B.2</strong>를 만족한다고 가정한다. 그때 Theorem 5.1과 같은 균등 부피 극한을 얻는다는 것이 Theorem 5.2다.

WKB 가정은 단순한 계산 편의가 아니다. 관상근방으로의 확률 집중, 양의 $C^2$ prefactor, 그 prefactor의 $C^2$ 수렴, 선택된 action 해의 유일성을 포함한다. 이 논문은 이런 가정들이 모든 학습 score network에서 성립한다고 증명하지 않는다. 부록 B.4에서 그 가정 아래의 계수 결정 과정을 따라간다. [Theorem 5.2](https://arxiv.org/html/2509.24912v2#S5.Thmtheorem2), [가정 B.2](https://arxiv.org/html/2509.24912v2#A2.Thmassumption2)

또한 각각의 $\sigma$에서 먼저 정상상태를 얻고 나서 $\sigma\to0$를 취한다. 고정된 수의 corrector step으로 $\sigma$를 계속 바꾸는 실용 알고리즘의 분포가 이 극한과 같다는 결론은 별도의 문제다.

## 6. 역문제에서 더 강건한 균등 사전분포 (Uniform Prior is More Robust For Bayesian Inverse Problems)

### 목표가 어떻게 달라지는가

데이터 빈도를 prior로 쓰면 posterior는 $e^{-v}\rho\,d\mathcal M$다. 반면 부피에 대한 균등 prior를 쓰면 $e^{-v}\,d\mathcal M$다. TS는 후자를 목표로 한다. 따라서 원래 데이터 prior의 posterior를 더 부정확한 score로 그대로 얻는 알고리즘이 아니다.

<strong>보충 유도 최대 엔트로피.</strong> $\mathcal M$의 부피를 $V_{\mathcal M}$라 하면 균등 밀도는 $1/V_{\mathcal M}$이고, 부피 기준 밀도 $\rho$의 엔트로피를 생각하자.

Kullback–Leibler 발산(Kullback–Leibler divergence, KL)은 같은 기준측도에 대한 두 밀도 $p,q$의 $D_{\mathrm{KL}}(p\|q)=\int p\log(p/q)$다. $q$를 기준으로 $t=p/q$라 두면 $t\log t$의 볼록성과 $\int tq=1$ 때문에 적분이 0 이상이다. 등호는 두 분포가 같은 경우에만 성립한다. 아래 엔트로피 상계와 균등분포의 최적성은 바로 이 비음성에서 나온다.

$$
-\int_{\mathcal M}\rho\log\rho\,d\mathcal M
=\log V_{\mathcal M}
-D_{\mathrm{KL}}\!\left(\rho\,d\mathcal M\;\middle\|\;
\frac{d\mathcal M}{V_{\mathcal M}}\right)
\le\log V_{\mathcal M}.
$$

“최대 엔트로피 prior”는 이 기준 측도를 먼저 택했을 때의 표현이다. 다른 좌표의 Lebesgue 측도에 대해 무조건 균등하다는 뜻이 아니다.

### 정리와 유도

Theorem 5.2와 같은 조건에 더해 $v$는 $\mathbb R^d$ 전체에서 유계이고 관상근방에서 $C^1$이라고 하자. 원문 Theorem 6.1의 gradient 표기는

$$
dX_t=-\nabla v(X_t)\,dt-\sigma^\alpha\nabla f_\sigma(X_t)\,dt+\sqrt2\,dW_t
$$

이며, 일반 oracle 형태로는 prior 항을 $\sigma^\alpha s$로 읽는다. 이 <strong>guided 동역학에 필요한 정상분포·WKB 조건 아래</strong> 극한 좌표 밀도는

$$
\pi_{\mathrm{post}}(u)
=\frac{e^{-v(\Phi(u))}\sqrt{\det g(u)}}
{\int_U e^{-v(\Phi(w))}\sqrt{\det g(w)}\,dw}.
$$

gradient인 경우 정상밀도의 지수에 $-v-\sigma^\alpha f_\sigma$가 들어간다. $\theta=\sigma^{2-\alpha}$로 바꾸면 주위의 큰 거리 항은 $d_{\mathcal M}/\theta$, 부피 위에 남는 유효 목적은 $v$다. 일반 경우에도 부록 B.4의 $f_1=v$를 사용한다. 자세한 연결은 부록 B.6에 있다. [원문 Theorem 6.1](https://arxiv.org/html/2509.24912v2#S6.Thmtheorem1)

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

원문은 원과 타원의 각도 좌표에서 von Mises 데이터 분포를 만들고, 학습 score로 일반 Langevin과 $\alpha=1$인 TS를 비교한다. Fig.2에서는 일반 Langevin이 원래 빈도를 제대로 재현하지 못하는 반면 TS가 부피 균등 분포에 가까워지는 모습을 보인다. 관상근방의 법선 집중과 접선 빈도를 함께 보아야 하며, 타원에서는 각도 균등을 정답으로 삼지 않는다.

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
- 그림: 대표 이미지는 자체 제작 벡터 그림이다. 원문 Fig.1–3은 해당 결과의 해설을 위한 부분 인용이며, 저자·버전은 위 원문과 같다. 그림 영역을 벡터로 추출하며 Fig.1 내부의 래스터만 원래 픽셀 크기를 유지한 WebP(품질 95)로 바꾼다. Fig.3은 긴 변 1,500픽셀의 WebP(품질 92)로 변환한다. 원문에는 arXiv 영구 비독점 공개 라이선스가 표시되어 있다. 별도 예제 그림은 명시한 모형의 자체 계산이다.
- 확인하지 않은 것: 신경망 학습, Stable Diffusion 이미지 생성의 독립 재현과 정리의 형식 검증.
