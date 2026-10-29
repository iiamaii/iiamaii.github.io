---
title: "[2/4] Score Geometry: Laplace와 오차 분리"
description: "좌표 밀도와 부피 밀도, 두 Hessian, Laplace 근사의 중간 계산을 통해 지지집합과 원래 분포의 복원 조건을 증명한다."
date: "2026-10-29"
publishedAt: "2026-10-29T08:55:42+09:00"
updatedAt: "2026-10-29T09:31:35+09:00"
topics: ["생성 모델", "확률과 기하"]
visibility: "public"
lang: "ko"
translationKey: "score-geometry-laplace"
paperTitle: "When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis"
paperPublishedDate: "2025-09-29"
authors: ["Xiang Li", "Zebang Shen", "Ya-Ping Hsieh", "Niao He"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2509.24912v2"
thumbnail: "/assets/score-geometry/cover-2.svg"
thumbnailAlt: "기하·분포 학습 시리즈 2편: Laplace 근사에서 오차 분리 정리까지"
---

작은 온도 $\theta$에서 최소점이 곡면 전체를 이루면, Laplace 근사의 극한 밀도에는 법선 Hessian 행렬식의 역제곱근이 남는다. 이 계산은 기하 복원에 충분한 score 오차 $o(\sigma^{-2})$와 원래 분포 복원에 충분한 $o(1)$의 차이를 연결한다. [원문 부록 B.1–B.3](https://arxiv.org/html/2509.24912v2#A2.SS1)

<strong>리뷰 해석.</strong> 이 논문의 정확도 분리는 정규화 상수와 좌표 야코비안을 함께 추적할 때 가장 분명해진다.

이 시리즈의 2편이다. [[1/4] Score Geometry: 기하와 빈도](/reviews/score-geometry-manifold/) · [[3/4] Score Geometry: 정상분포와 혼합](/reviews/score-geometry-wkb/) · [[4/4] Score Geometry: 균등 극한 증명](/reviews/score-geometry-weak-proof/).

## A. 증명을 위한 기호와 기초 (Additional Notation and Preliminaries)

$\mathcal M\subset\mathbb R^d$는 $n<d$차원의 콤팩트하고 경계 없는 $C^4$ 매장 다양체다. 데이터의 좌표 밀도 $p_{\mathrm{data}}$는 양의 $C^1$ 함수다. $\sigma$는 Gaussian 잡음의 표준편차이며, $p_\sigma$는 그 잡음으로 평활화한 주변 공간 밀도다. $\pi_\sigma$는 비교할 생성 밀도이고 $K$는 다양체를 포함하여 원문이 집중을 가정하는 콤팩트 영역이다. 아래 표는 기호의 차원을 구별한다.

| 기호 | 의미와 차원 |
| --- | --- |
| $d,n,q=d-n$ | 주변 공간·다양체·법공간의 차원 |
| $\Phi(u)$ | 다양체의 국소 좌표, $\mathbb R^n\to\mathbb R^d$ |
| $\Phi(u,r)=\Phi(u)+N(u)r$ | 관상근방 좌표, $r\in\mathbb R^q$ |
| $P_{\mathcal M}(x)$ | 최근접점, 관상근방에서 유일 |
| $d_{\mathcal M}(x)$ | $\frac12\operatorname{dist}^2(x,\mathcal M)$ |
| $J_T=D_u\Phi,\;g=J_T^\top J_T$ | 접선 야코비안과 $n\times n$ 계량 |
| $J=D_{(u,r)}\Phi,\;G=J^\top J$ | 전체 야코비안과 $d\times d$ 계량 |
| $p_{\mathrm{data}}(u)$ | 원문의 좌표 밀도, 기준 측도 $du$ |
| $\rho=p_{\mathrm{data}}/\sqrt{\det g}$ | 해설의 보조 기호, 부피 기준 밀도 |
| $\widehat H(u,x)$ | smoothing 적분의 <strong>접선 변수</strong> Hessian, $n\times n$ |
| $H_N(u)=\partial_{rr}^2f_0(u,0)$ | Laplace 집중의 <strong>법선 변수</strong> Hessian, $q\times q$ |
| $\theta$ | 작은 온도 변수. §4에서는 $\sigma^2$, §5·6에서는 $\sigma^{2-\alpha}$ |
| $E_\sigma,\beta,\alpha$ | score 최악 오차, 그 오차의 지수, score 조절의 지수 |

두 Hessian은 서로 다르다. Gaussian smoothing의 적분에서는 <strong>데이터의 접선 좌표 $u$를 적분</strong>해 $\widehat H$를 얻고, 최종 측도를 다양체로 제한할 때는 <strong>주변 공간의 법선 좌표 $r$를 적분</strong>해 $H_N$을 얻는다. 하나는 $n\times n$, 다른 하나는 $(d-n)\times(d-n)$다.

<details class="review-background">
<summary><span>보충 설명: 유한 좌표와 균일한 상수</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

콤팩트 다양체는 유한한 좌표 패치로 덮을 수 있다. 각 패치를 조금 줄여 그 폐포가 원래 패치 안에 놓이게 하면, 연속인 미분계수들은 이 폐포들에서 최대·최소를 갖는다. 패치 수가 유한하므로 모든 패치에 공통인 상수를 고를 수 있다.

이것이 원문 Remark B.1에서 shrinking lemma를 인용하는 이유다. 콤팩트하지 않은 좌표 영역 전체에서 모든 미분계수가 자동으로 유계라는 뜻은 아니다. 특히 양의 좌표 밀도의 하한은 잘 선택한 유한한 국소 영역에서 확보한다.

</div>
</details>

<details class="review-background">
<summary><span>보충 설명: 행렬 양의 정부호를 유지하는 법</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

대칭행렬 $A,B$에 대해 단위벡터 $v$를 사용하면

$$
v^\top(A+B)v\ge\lambda_{\min}(A)-\|B\|,
\qquad
\lambda_{\min}(A+B)\ge\lambda_{\min}(A)-\|B\|.
$$

따라서 $\widehat H=g+\text{곡률 보정}$에서 곡률 보정의 크기를 $g$의 최소 고유값의 절반보다 작게 만들면 $\widehat H$는 균일하게 양의 정부호다. 원문이 Weyl 부등식으로 사용하는 부분은 이 변분 원리만으로도 확인할 수 있다.

</div>
</details>

## B.1. 최소점 다양체에서의 극한 분포 (A General Framework for the Convergence of the Limiting Distribution)

### 따름정리 B.1 Laplace 근사의 진술과 이유

Łapiński의 Theorem 2를 최소화 표기로 바꾼 것이 원문의 Corollary B.1이다. $F$가 유일한 내부 최소점 $x^*$를 갖고, 그 근처에서 $C^3$, Hessian은 양의 정부호이며, 바깥에는 양의 에너지 간격이 있다고 하자. 진폭 $a$는 근처에서 $C^1$이고 적분 가능하다고 하자. 그러면

$$
\int a(x)e^{-F(x)/\theta}\,dx
=e^{-F(x^*)/\theta}
\frac{(2\pi\theta)^{m/2}}{\sqrt{\det\nabla^2F(x^*)}}
\left[a(x^*)+O(\sqrt\theta)\right].
$$

여기서 균일한 오차를 얻으려면 세 가지 종류의 상수가 함께 제어되어야 한다. 미분계수와 진폭의 상한, Hessian 고유값의 양의 하한, 근처 영역 바깥의 에너지 간격이다. 원문의 여섯 상수 $c_1,\ldots,c_6$와 두 양의 상수 $m_1,m_2$가 이 역할을 한다. [Łapiński v7, Theorem 2, Portable Document Format(PDF) p.5](https://arxiv.org/pdf/1502.03266v7#page=5)

<strong>리뷰 보충 유도.</strong> $x=x^*+\sqrt\theta\,z$로 바꾸고 Hessian을 $H$라 하자.

$$
F(x)=F(x^*)+\frac\theta2 z^\top Hz+O(\theta^{3/2}\|z\|^3).
$$

leading integral은 $e^{-F(x^*)/\theta}\theta^{m/2}a(x^*)\int e^{-z^\top Hz/2}\,dz$다. 선형변환 $w=H^{1/2}z$의 야코비안이 $(\det H)^{-1/2}$이므로 Gaussian 적분이 $(2\pi)^{m/2}/\sqrt{\det H}$가 된다. 원래 적분의 폭이 $\sqrt\theta$라는 사실과 Hessian 행렬식이 나타나는 이유를 동시에 설명한다.

최소점에서 떨어진 부분은 $e^{-c/\theta}$처럼 작다. 이 항은 모든 고정 거듭제곱 $\theta^m$보다 빨리 작아진다. 지역 Gaussian 계산을 전역 적분 근사로 바꾸려면 이 꼬리 제어가 반드시 필요하다.

### 가정 B.1과 보조정리 B.1 법선 방향 적분

원문의 일반 모형은

$$
f_\theta=f_0+\theta f_1+\widehat f_\theta,\qquad
\|\widehat f_\theta\|_\infty=o(\theta),
\qquad
\pi_\theta\propto e^{-f_\theta/\theta}.
$$

$\mathcal M$은 $f_0$의 최소점 집합이다. $f_0\ge0$는 $C^3$, $f_1$은 $C^1$이고, 법선 Hessian은 작은 관상근방에서 균일하게 양의 정부호다. 최소점에서 떨어진 구간에는 양의 에너지 간격이 있으며 $\mathcal M$은 콤팩트하고 경계가 없는 $C^4$ 다양체다. 이 조건들이 가정 B.1의 내용이다.

중요한 크기는 $\widehat f_\theta$ 그 자체가 아니라 $\widehat f_\theta/\theta$다. 지수에 들어가기 때문이다. 균일한 $o(\theta)$이면

$$
\sup_x\left|e^{-\widehat f_\theta(x)/\theta}-1\right|\to0.
$$

따라서 이 작은 섭동은 적분의 leading factor를 바꾸지 않는다.

Lemma B.1은 $u$를 고정하고 법선 $r$만 적분한다. $H_N(u)=\partial^2_{rr}f_0(u,0)$라 하면, 적절한 유계 $C^1$ 함수 $h$에 대해

$$
\begin{aligned}
&\int_{\|r\|<\epsilon}e^{-f_\theta(u,r)/\theta}h(u,r)\,dr\\
&\quad=
e^{-f_0(u,0)/\theta-f_1(u,0)}
\frac{(2\pi\theta)^{q/2}}{\sqrt{\det H_N(u)}}
\left[h(u,0)+o(1)\right].
\end{aligned}
$$

원문의 증명은 섭동을 뺀 적분에 Corollary B.1을 적용하고, 위 지수 섭동의 supremum과 양의 적분을 곱해 차이를 제어한다. 같은 하한 조건·유한 atlas 덕분에 나머지는 $u$에도 균일하다.

### 보조정리 B.2 집중과 Portmanteau

$f_\theta=f_0+o(1)$이고 이미 관상근방에 질량이 집중한다면, 정규화된 $e^{-f_\theta/\theta}$는 $f_0$의 최소점 집합에 집중한다. 정규화 분모도 함께 제어해야 한다. 최소값을 0으로 놓고 $|f_\theta-f_0|<\eta/8$인 충분히 작은 $\theta$를 택하자. $f_0>\eta$인 집합에서는 $f_\theta>7\eta/8$이다. 반면 최소점 근처의 고정된 양의 부피 영역 $U$를 $f_0<\eta/8$이 되도록 택하면 그곳에서 $f_\theta<\eta/4$다. 따라서 관상근방 안에서의 정규화 질량 비는

$$
\frac{\int_{\{f_0>\eta\}}e^{-f_\theta/\theta}\,dx}
{\int_{T_{\mathcal M}(\epsilon)}e^{-f_\theta/\theta}\,dx}
\le
\frac{\operatorname{Vol}(T_{\mathcal M}(\epsilon))}
{\operatorname{Vol}(U)}
e^{-5\eta/(8\theta)}\to0.
$$

관상근방 밖의 질량 소멸을 더하면 전체 집중이 된다. 이 단계는 상대 오차를 확인할 때 분모를 생략하지 않도록 풀어 쓴 것이다.

약극한이 존재할 때 그 지지집합이 $\mathcal M$에 놓인다는 마지막 단계는 <strong>Portmanteau 정리</strong>의 열린 집합 부등식이다.

$$
\mu(G)\le\liminf_{\theta\to0}\mu_\theta(G)
\quad(G\text{가 열린 집합}).
$$

<strong>리뷰 보충 유도.</strong> $h_k(x)=\min\{1,k\,\operatorname{dist}(x,G^c)\}$는 $1_G$로 증가하는 유계 연속 함수다. 약수렴을 먼저 각 $h_k$에 적용하고 $k\to\infty$에서 단조수렴을 쓰면 위 부등식이 나온다. 열린 집합들의 질량이 0으로 가는 것을 이용해 다양체 바깥의 극한 질량이 없다고 결론낸다.

원문은 이 보조정리에 연결성도 명시하지만, 이 단계의 “최소점 집합으로의 집중” 자체는 각 성분에 공통 최소값과 간격이 있으면 된다. 연결성은 이후 최대원리로 전체 부피 균등성을 논할 때 더 직접적으로 중요해진다.

### 정리 B.1 최종적으로 남는 다양체 위의 밀도

위 조건에 더해 관상근방 밖의 질량이 사라진다고 하자. 원문의 Theorem B.1은 다음 부피 기준 밀도를 준다.

$$
\frac{d\mu_0}{d\mathcal M}(u)
\propto
\frac{e^{-f_1(u,0)}}{\sqrt{\det H_N(u)}}.
$$

좌표 $du$ 기준으로는 여기에 $\sqrt{\det g(u)}$를 곱한다.

<strong>보충 증명.</strong> 유계 매끄러운 시험함수 $h$에 대해, 관상근방 좌표의 부피 야코비안을 $j(u,r)=|\det J(u,r)|$라 쓰자.

$$
\int h\,d\mu_\theta
=
\frac{\int_U\int_R e^{-f_\theta(u,r)/\theta}h(u,r)j(u,r)\,dr\,du}
{\int_U\int_R e^{-f_\theta(u,r)/\theta}j(u,r)\,dr\,du}
+o(1).
$$

분자·분모에 Lemma B.1을 적용한다. $f_0(u,0)$는 최소점 집합에서 일정하므로 그 지수와 $(2\pi\theta)^{q/2}$가 약분된다. $j(u,0)=\sqrt{\det g(u)}$이므로 남는 비가 바로 위 식이다. 유한 좌표 합과 균일 오차로 극한·적분을 교환하고, 연속 시험함수는 콤팩트 영역에서 근사해 확장한다.

전역적으로 한 좌표 $U$가 다양체 전체를 덮는다는 가정을 한 것은 아니다. 실제로는 좌표별 부드러운 가중치 $\chi_i\ge0$, $\sum_i\chi_i=1$인 <strong>단위 분할(partition of unity)</strong>을 사용한다. 각 패치에서 진폭에 $\chi_i$를 곱해 위 적분을 계산한 뒤 유한 합을 취한다. 겹친 영역을 중복 계산하지 않으면서 국소 유도를 전역 적분으로 이어주는 장치다.

이 계산은 원문이 Hwang(1980)의 Theorem 3.1을 인용해 짧게 넘긴 필요한 특수형을 직접 재구성한 것이다. Hwang 원전 전문은 여기서 독립 확인하지 않았으며, 증명은 이 문서에 적은 Laplace 조건으로 읽으면 된다. [원문 Theorem B.1](https://arxiv.org/html/2509.24912v2#A2.Thmtheorem1), [Hwang 서지](https://doi.org/10.1214/aop/1176994579)

### 왜 “같은 최소점 집합”만으로 균등성이 나오지 않는가

설명용으로 $f_0(u,r)=\frac12 a(u)r^2$, $f_1=0$을 생각하자. 최소점은 항상 $r=0$이지만 법선 방향이 느슨한 $a(u)$가 작은 곳은 더 넓은 Gaussian 단면을 갖는다. 따라서 극한 부피 밀도는 $a(u)^{-1/2}$에 비례한다.

균등성을 주는 특별한 사실은 이 논문의 $f_0=d_{\mathcal M}$에 대해 $H_N=I_q$라는 점이다. 단지 minimum이 다양체라는 사실보다 훨씬 구체적인 구조를 이용한다.

## B.2. Gaussian 평활화의 점근 전개 (Proof for Theorem 3.1)

이 절의 정밀한 진술은 원문 <strong>Theorem B.2</strong>다. 본문 Theorem 3.1은 이를 비형식적으로 요약한 것이다.

### 1단계 적분의 최소점과 Hessian

분산 폭증(variance exploding, VE)의 위상함수는 $F_x(u)=\frac12\|x-\Phi(u)\|^2$다. 최소점은 최근접점 좌표 $u_x$이며,

$$
\partial_iF_x(u)=\langle\Phi(u)-x,\partial_i\Phi(u)\rangle.
$$

최소점에서는 변위가 접선에 직교하므로 이 값이 0이다. 한 번 더 미분하면

$$
\partial_{ij}F_x(u)
=\langle\partial_i\Phi,\partial_j\Phi\rangle
+\langle\Phi-x,\partial_{ij}\Phi\rangle
=\widehat H_{ij}(u,x).
$$

$x=\Phi(u)$이면 두 번째 항은 0이므로 $\widehat H(u,\Phi(u))=g(u)$다. 작은 관상근방에서는 곡률 보정이 작고, 부록 A의 고유값 부등식으로 $\widehat H$가 양의 정부호임을 보장한다.

곡률을 더 구체적으로 보면 $\partial_{ij}\Phi$의 법선 성분인 <strong>제2기본형(second fundamental form)</strong>만 두 번째 내적에 기여한다. $\Phi-x$가 법선이기 때문에 접선 성분과의 내적은 0이다. 따라서 곡면에서 떨어진 거리와 곡률의 곱이 적분의 폭을 바꾼다.

단위원의 $\Phi(u)=(\cos u,\sin u)$, $x=(R,0)$를 예로 들면 $u_x=0$, $g=1$, $\widehat H=1+(R-1)=R$다. 따라서 VE의 곡률 보정은 $H(x)=-\tfrac12\log R$다. 원점 $R=0$에서는 Hessian이 퇴화하고 모든 원 위 점이 최근접점이므로 이 국소 Laplace 설명을 쓸 수 없다. 관상근방 조건이 단순한 기술적 장식이 아닌 이유다.

### 2단계 Laplace 근사를 적용한다

$\theta=\sigma^2$, 진폭 $a(u)=p_{\mathrm{data}}(u)$로 Corollary B.1을 적용한다.

$$
p_\sigma^{\mathrm{VE}}(x)
=
e^{-d_{\mathcal M}(x)/\sigma^2}
\frac{(2\pi\sigma^2)^{(n-d)/2}}
{\sqrt{\det\widehat H(u_x,x)}}
\left[p_{\mathrm{data}}(u_x)+O(\sigma)\right].
$$

주변 Gaussian 정규화의 $d$차원과 데이터 적분의 $n$차원이 달라 $(n-d)/2$가 생긴다. 이 차원을 빠뜨리면 작은 잡음의 밀도 발산과 법선 폭을 잘못 해석하게 된다.

### 3단계 로그와 균일 remainder

밀도의 양의 하한을 이용해 괄호를 $p_{\mathrm{data}}(u_x)[1+O(\sigma)/p_{\mathrm{data}}(u_x)]$로 쓸 수 있다. 로그를 취하면

$$
\begin{aligned}
\log p_\sigma^{\mathrm{VE}}(x)
={}&-\frac{d_{\mathcal M}(x)}{\sigma^2}
+\log p_{\mathrm{data}}(u_x)
-\frac{d-n}{2}\log(2\pi\sigma^2)\\
&-\frac12\log\det\widehat H(u_x,x)+o(1).
\end{aligned}
$$

특히 $x=\Phi(u)$에서는 $\log p_{\mathrm{data}}-\frac12\log\det g=\log\rho$다. 좌표를 바꿔도 물리적인 밀도 전개가 달라지지 않는 이유를 이 상쇄가 보여준다.

### 4단계 분산 보존(variance preserving, VP)의 추가 항

$\gamma=\sqrt{1-\sigma^2}=1-\sigma^2/2+O(\sigma^4)$를 사용하면, 콤팩트한 영역에서 균일하게

$$
\|x-\gamma\Phi(u)\|^2
=\|x-\Phi(u)\|^2
+\sigma^2\langle\Phi(u),x-\Phi(u)\rangle+O(\sigma^4).
$$

따라서 같은 VE 위상함수에 진폭

$$
p_{\mathrm{data}}(u)
\exp\!\left[-\frac12\langle\Phi(u),x-\Phi(u)\rangle\right]
$$

을 넣어 Laplace 근사를 적용하면 된다. 로그에 $-\frac12\langle P_{\mathcal M}(x),x-P_{\mathcal M}(x)\rangle$가 추가되고, 이는 고정 관상근방에서 $O(1)$이며 다양체 위에서는 0이다.

<strong>리뷰 해석.</strong> 원문 증명 초반은 VE·VP의 최소점을 같은 것으로 적는다. 유한한 $\sigma$에서 VP의 정확한 최근접점은 일반적으로 $\gamma P_{\mathcal M}(x/\gamma)$이므로 문자 그대로 같은 점은 아니다. 위처럼 VE의 위상함수와 VP의 진폭 보정으로 분리하면 원문이 의도한 점근 전개를 얻을 수 있다. 정확한 최소점을 동일시할 필요가 없다.

### 5단계 점근식을 미분할 수 있는가

$r_\sigma=o(1)$이라고 해서 $\nabla r_\sigma=o(1)$인 것은 아니다. 예를 들어 $\sigma\sin(x/\sigma^2)$는 균일하게 작지만 미분은 $\sigma^{-1}\cos(x/\sigma^2)$다. 따라서 score의 $O(1)$ 나머지는 이 절의 log-density 전개만으로 증명되지 않는다. 부록 B.5의 더 정밀한 적분 비 계산이 필요하다.

## B.3. score 오차에서 밀도 오차로 (Proofs for Theorem 4.1)

### 보조정리 B.3 정확한 smoothing은 다양체 가까이에 있다

VE에서 $X_\sigma=Y+\sigma Z$이고 $Y\in\mathcal M$이므로

$$
\Pr(\operatorname{dist}(X_\sigma,\mathcal M)\ge\epsilon)
\le\Pr(\|Z\|\ge\epsilon/\sigma)\to0.
$$

Gaussian 꼬리는 지수적으로 작아진다. VP에서는 추가 변위 $(\sqrt{1-\sigma^2}-1)Y=O(\sigma^2)$를 삼각부등식으로 처리할 수 있다. $Y$의 크기가 유계이므로 같은 결론이 나온다. 원문에서 이 적분 순서를 교환하는 근거는 integrand가 음이 아닌 경우에 적용되는 <strong>Tonelli 정리</strong>다.

### 정규화가 빠지면 안 되는 핵심 부등식

$m_\sigma=\pi_\sigma(K)$, $n_\sigma=\mu_\sigma(K)$라 놓고 $\Delta_\sigma=\log(\pi_\sigma/p_\sigma)$라 하자. 경로 적분으로 $\Delta_\sigma(x)=\Delta_\sigma(x_0)+r_\sigma(x)$이고 $\|r_\sigma\|_\infty\le L_KE_\sigma$다. 정규화를 사용하면

$$
m_\sigma=e^{\Delta_\sigma(x_0)}
\int_Kp_\sigma(x)e^{r_\sigma(x)}\,dx.
$$

따라서

$$
\left|\Delta_\sigma(x_0)-\log\frac{m_\sigma}{n_\sigma}\right|
\le L_KE_\sigma,\qquad
\|\Delta_\sigma\|_\infty
\le2L_KE_\sigma+
\left|\log\frac{m_\sigma}{n_\sigma}\right|.
$$

가정과 Lemma B.3에 의해 $m_\sigma,n_\sigma\to1$이다. 그러므로 $E_\sigma=o(\sigma^{-2})$이면 $\|\Delta_\sigma\|_\infty=o(\sigma^{-2})$이고, $E_\sigma=o(1)$이면 $\|\Delta_\sigma\|_\infty=o(1)$이다.

<strong>리뷰 해석.</strong> 원문은 이 논증을 모든 $\eta\ge-2$의 $o(\sigma^\eta)$로 일반화하지만, $m_\sigma,n_\sigma\to1$만으로 정규화 오차가 양의 $\eta$에 대해 $o(\sigma^\eta)$가 된다고 할 수는 없다. 주정리에서 사용하는 $\eta=-2,0$의 두 경우는 위 부등식으로 충분하다. 따라서 필요한 결론을 그 두 경우로 좁히면 이 문제가 사라진다.

### 보조정리 B.4와 1번 결론 집중

다양체에서 $\delta$ 이상 떨어진 $x$에 대해 Gaussian 적분은 대략 $\sigma^{-d}e^{-\delta^2/(2\sigma^2)}$ 이하로 작아진다. 여기에 $\pi_\sigma=p_\sigma e^{\Delta_\sigma}$와 $\Delta_\sigma=o(\sigma^{-2})$를 넣으면 충분히 작은 $\sigma$에서

$$
\pi_\sigma(K\cap\{\operatorname{dist}(x,\mathcal M)>\delta\})
\le C_K\sigma^{-d}
\exp\!\left[-\frac{\delta^2}{4\sigma^2}\right]\to0.
$$

VP에서는 $\delta$와 $\sigma$에 독립인 충분히 작은 상수 배를 사용하면 된다. $K$ 밖의 질량은 가정으로 사라진다. 이것이 Theorem 4.1의 첫 결론이다.

### 3번 결론 원래 분포 복원

$E_\sigma=o(1)$이면 $K$에서 $e^{\Delta_\sigma}=1+o(1)$가 균일하다. 따라서 $\pi_\sigma$와 $\mu_\sigma$의 차이는 $K$에서의 적분도, $K$ 밖의 질량도 0으로 간다. 즉 두 smoothing 측도 사이에는 전변동(total variation) 오차까지 0으로 간다. $\mu_\sigma\Rightarrow\mu_{\mathrm{data}}$이므로 $\pi_\sigma\Rightarrow\mu_{\mathrm{data}}$다.

이는 원문의 Theorem B.1 사용과 같은 결론을 더 직접적인 밀도 비로 얻는 보충 증명이다. 전변동으로 직접 비교한 대상은 $\pi_\sigma$와 <strong>매끄러운 $\mu_\sigma$</strong>다. 주변 공간에서 매끄러운 측도와 저차원 $\mu_{\mathrm{data}}$ 자체가 전변동으로 수렴한다는 주장은 아니다.

### 2번 결론 다른 접선 밀도도 가능하다

부피 기준 원래 밀도를 $\rho$, 다른 양의 $C^1$ 목표 밀도를 $q$라 하자. 관상근방에서

$$
a(x)=\log\frac{q(P_{\mathcal M}(x))}{\rho(P_{\mathcal M}(x))},
\qquad
\pi_\sigma(x)\propto p_\sigma(x)e^{a(x)}
$$

로 기울이면 최근접 사영의 미분과 두 밀도의 양의 하한 때문에 $\nabla a$는 유계다. score는 $O(1)$만 바뀌지만 약극한은 $q\,d\mathcal M$다. $q\ne\rho$이면 이 변화는 일반적으로 사라지지 않는다. 바깥에서는 매끄러운 cutoff로 확장한다.

이 예는 작은 잡음에서 $O(1)$의 score 차이가 다양체 위의 빈도를 바꿀 수 있음을 직접 보여준다.

<strong>리뷰 보충 유도: 목표 밀도가 0이 되는 경우.</strong> $q_\eta=(q+\eta\rho)/(1+\eta)$로 양의 밀도를 만들고 $\eta\downarrow0$로 보내자. 콤팩트 다양체에서 진폭 $q_\eta/\rho$는 $q/\rho$로 균일 수렴한다. 고정된 매끄러운 관상근방 cutoff로 이 진폭과 바깥의 1을 연결하면, 주변 공간의 유계 연속 진폭도 균일 수렴한다. 따라서 $\mu_\sigma\Rightarrow\mu_{\mathrm{data}}$에 대한 적분과 정규화로 같은 목표 극한을 얻는다. 로그 미분은 $\eta$에 따라 커질 수 있지만, 원문의 $\Omega(1)$은 상수 상계를 요구하는 표기가 아니다. $q=\rho$인 특별한 목표에서도 아래 진동 구성을 쓰면 약극한을 유지하면서 score 오차를 사라지지 않게 할 수 있다.


### 리뷰 보충 유도: 충분조건의 역과 진동 밀도

$E_\sigma=o(1)$은 원래 분포 복원의 <strong>충분조건</strong>이다. 정리는 모든 성공적인 근사열이 반드시 이 조건을 만족해야 한다고 증명하지 않는다. 다음은 이 차이를 보여주는 보충 예다.

원 위의 양의 밀도 $\rho(u)$와 정수 $k\to\infty$에 대해

$$
\pi_k(u)=Z_k^{-1}\rho(u)\exp\!\left(\frac{\sin ku}{k}\right)
$$

라 두면 밀도 비는 균일하게 1로 수렴한다. 그러나

$$
\frac{d}{du}\log\frac{\pi_k(u)}{\rho(u)}=\cos ku
$$

이므로 score 오차의 supremum은 1이다. 밀도는 수렴해도 미분은 수렴하지 않을 수 있다. 논문의 메시지는 <strong>최악의 오차를 허용하는 보장에 필요한 정밀도의 분리</strong>로 이해하는 편이 정확하다.

이 예를 정리의 주변 공간 설정에도 맞출 수 있다. $\sigma_k=1/k$, 단위원 근처에서 1이고 원점 근처와 먼 곳에서는 0인 매끄러운 radial cutoff $\chi(R)$를 택하고 $F_k(x)=\chi(R)\sin(ku)/k$라 하자. $\pi^{\mathrm{amb}}_{\sigma_k}\propto p_{\sigma_k}e^{F_k}$로 정의하면 $\|F_k\|_\infty\le1/k$여서 밀도 비가 1로 수렴하고 원래 데이터 측도로 약수렴한다. 그러나 원 위에서는 접선 score 차이가 $\cos(ku)$여서 sup 오차는 적어도 1이다. cutoff가 원점을 피하므로 전체 밀도는 매끄럽고, 고정된 큰 공 $K$에서 집중·경로 조건도 만족한다. 이는 <strong>충분조건의 역이 일반적으로 성립하지 않음</strong>을 보여주는 해설자의 구성이다.

### 리뷰 보충 예제: score의 관찰 위치와 진동 밀도

단위원을 정확한 Gaussian 적분으로 평활화하면, 고정 반지름 $R=1.2$, $\sigma=0.01$에서 기하 항을 뺀 방사형 score의 나머지는 분산 폭증(variance exploding, VE) 모형에서 약 $-0.4167$, 분산 보존(variance preserving, VP) 모형에서 약 $-0.9167$다. 각각의 극한 $-1/(2R)$와 $-1/2-1/(2R)$에 가깝다. 반면 $R=1+\sigma$에서는 선도 score가 $\sigma^{-1}$ 크기다. 같은 전개라도 관찰 위치의 척도가 다르면 전체 score의 크기가 달라진다.

위 진동 예제에서 $\rho$를 균등밀도로 두면 $k=10,100,1000$에서 정규화된 최대 밀도 비 오차는 약 $0.1024,0.0100,0.0010$으로 줄고, 각도 score의 상한 오차는 항상 1이다. 두 계산은 이 글의 설명용 적분 결과다.

## 리뷰 해석과 남은 질문

<strong>리뷰 해석.</strong> 두 개의 Hessian을 구별하는 것이 이 증명의 중심이다. 평활화에서는 데이터의 접선 좌표를 적분하고, 극한 측도에서는 주변 공간의 법선 좌표를 적분한다. 일반 퍼텐셜은 같은 최소점 다양체를 가져도 법선 우물의 폭이 다르면 균등한 극한을 만들지 않는다.

정리 4.1은 최종 생성 밀도의 score를 비교한다. 원래 분포를 복원하는 $o(1)$은 최악 오차에 대한 충분조건이며, 개별 수열의 필요조건이라는 주장은 진동 밀도 예제로 구별된다. 밀도 비의 정규화 오차는 본 정리에서 필요한 두 차수에서는 사라진다. 그보다 빠른 일반 수렴률을 말하려면 별도 속도 가정이 필요하다.

## 출처와 읽은 범위

- 원문: Xiang Li, Zebang Shen, Ya-Ping Hsieh, Niao He, *When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis*. 최초 공개 2025-09-29, 검토 버전 v2(2026-03-16), 검토일 2026-10-29. [공식 원문](https://arxiv.org/abs/2509.24912v2).
- 읽은 범위: 부록 A, B.1–B.3의 Corollary B.1, Lemmas B.1–B.4, Theorems B.1–B.2와 정리 4.1의 연결.
- 추가 원전: [Łapiński v7, Theorem 2, 파일 p.5](https://arxiv.org/pdf/1502.03266v7#page=5), [Leobacher–Steinicke, Theorems 2 및 C](https://arxiv.org/abs/1811.10578). Hwang(1980)의 전문은 미확인이며 필요한 특수형은 본문에서 유도한다.
- 그림: 대표 이미지는 자체 제작한 벡터 설명 그림이다. 본문의 별도 수치 예제 그림은 명시한 모형의 결정론적 적분 결과다.
- 확인하지 않은 것: 신경망 학습, Stable Diffusion 이미지 생성의 독립 재현과 정리의 형식 검증.
