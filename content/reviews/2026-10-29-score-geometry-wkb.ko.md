---
title: "[3/4] Score Geometry: 정상분포와 혼합"
description: "비보존적 score의 정상밀도 전개와 온도 조건을 유도하고, 추가 실험 전체와 Poincaré 부등식의 혼합 논증을 검토한다."
date: "2026-10-29"
publishedAt: "2026-10-29T08:55:41+09:00"
updatedAt: "2026-10-29T11:06:36+09:00"
topics: ["생성 모델", "확률과 기하"]
visibility: "public"
lang: "ko"
translationKey: "score-geometry-wkb"
paperTitle: "When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis"
paperPublishedDate: "2025-09-29"
authors: ["Xiang Li", "Zebang Shen", "Ya-Ping Hsieh", "Niao He"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2509.24912v2"
thumbnail: "/assets/score-geometry/cover-3.svg"
thumbnailAlt: "기하·분포 학습 시리즈 3편: 비보존적 score의 정상분포와 혼합 시간"
---

기울기가 아닌 score의 정상밀도는 Gibbs 공식만으로 정해지지 않는다. 원문은 Wentzel–Kramers–Brillouin(WKB) 전개를 가정하고, 오차 지수 $\beta>-2$에 대해 $\max(-\beta,0)<\alpha<2$인 조절된 score Langevin(Tempered Score Langevin, TS Langevin)의 균등 극한을 유도한다. [원문 정리 5.2, 부록 B.4–B.5](https://arxiv.org/html/2509.24912v2#A2.SS4)

<strong>리뷰 해석.</strong> 법선 수송에서 접선 타원형 방정식으로 이어지는 구조가 핵심 성과이며, 잔차 미분과 전역 혼합의 정당화는 가정과 별도로 추적해야 한다.

이 시리즈의 3편이다. [[1/4] Score Geometry: 기하와 빈도](/reviews/score-geometry-manifold/) · [[2/4] Score Geometry: Laplace와 오차 분리](/reviews/score-geometry-laplace/) · [[4/4] Score Geometry: 균등 극한 증명](/reviews/score-geometry-weak-proof/).

## B.4. 비보존적 동역학의 정상밀도 (Manifold WKB Analysis of the Stationary Distribution)

$\mathcal M\subset\mathbb R^d$는 $n<d$차원의 콤팩트하고 경계 없는 $C^4$ 다양체다. $\sigma$는 평활화 잡음의 표준편차, $s_\sigma^*=\nabla\log p_\sigma$는 정확한 score, $s_\sigma$는 추정 score다. 일반 score 정리에서는 데이터의 양의 좌표 밀도에 $C^2$ 정칙성을 추가한다. $d_{\mathcal M}=\tfrac12\operatorname{dist}^2(\cdot,\mathcal M)$, $P$는 관상근방의 최근접 사영이다. 국소 좌표 $\Phi(u)$와 법선 프레임 $N(u)$를 써서 $x=\Phi(u)+N(u)r$, $g=(D_u\Phi)^\top D_u\Phi$, $J=D_{u,r}\Phi$, $G=J^\top J$로 둔다.

원문 위치: 부록 B.4, Theorem B.3, Lemma B.5–B.6, Portable Document Format(PDF) pp. 24–28. 이 절은 정상밀도의 지수부와 진폭을 순서대로 정한다. 중간 계산은 리뷰 보충 유도로 풀어 쓴다.

### 왜 Gibbs 밀도만으로 끝낼 수 없는가

벡터장 $s$가 $-\nabla f$이면 정상밀도는 정규화 가능한 경우 $e^{-f}$다. 신경망 score가 항상 어떤 함수의 정확한 기울기라는 보장은 없다. 이때는 $e^{\int s\cdot dx}$를 밀도로 정의할 수 없다. 적분 경로에 따라 값이 달라질 수 있기 때문이다.

일반 벡터장에서는 정상 Fokker–Planck 방정식을 풀어야 한다. 원문은 다음 형태를 다룬다.

$$
dX_t=b_\theta(X_t)\,dt+\sqrt{2\theta}\,dW_t,\qquad
b_\theta=-\nabla f_0-\theta\nabla f_1+\widehat b_\theta,
\quad \widehat b_\theta=o(\theta).
$$

이를 시간 재조정하면 drift $b_\theta/\theta$, 잡음 $\sqrt2$인 확률미분방정식(stochastic differential equation, SDE)이 된다. <strong>정상분포는 같지만, 같은 시간 길이에 같은 궤적 분포를 만든다는 뜻은 아니다.</strong>

### WKB가 무엇을 가정하는가

Wentzel–Kramers–Brillouin(WKB) 근사는 작은 매개변수를 지수부와 느린 진폭으로 나누는 방법이다.

$$
\pi_\theta(x)\propto
e^{-V(x)/\theta}c_\theta(x),\qquad
c_\theta\longrightarrow c_0>0\quad\text{in }C^2.
$$

$V$는 지수 수준에서 질량이 어디에 모이는지 정하는 action 또는 준퍼텐셜(quasipotential)의 역할을 한다. $c_0$는 같은 최소 action을 가진 지점들 사이의 상대 질량을 정한다. 점 하나가 최소점이면 상대 빈도를 비교할 공간이 작지만, 최소점들이 다양체 전체를 이루면 <strong>진폭을 알아야 다양체 위의 분포를 안다.</strong>

원문 Assumption B.2는 이 형태, $V$의 $C^3$ 정칙성과 선택된 해의 유일성, 그리고 관상근방으로의 질량 집중을 가정한다. 따라서 여기서 WKB는 임의의 학습된 score에 대해 새로 증명한 사실이 아니다. Bouchet–Reygner의 [§3.1–3.3](https://arxiv.org/abs/1507.02104)은 정상 Fokker–Planck에서 Hamilton–Jacobi와 transport 식을 얻는 배경을 제공하지만, 그 문헌의 고립된 안정점 설정과 이 논문의 최소점 다양체 설정은 구분해야 한다.

### 1단계: Fokker–Planck에 대입한다

정상 방정식은

$$
0=-\nabla\cdot(b_\theta\pi_\theta)+\theta\Delta\pi_\theta.
$$

지수 인자를 약분하면 다음 항들을 얻는다.

$$
\begin{aligned}
0={}&-(\nabla\cdot b_\theta)c_\theta
-b_\theta\cdot\nabla c_\theta
+\frac{b_\theta\cdot\nabla V}{\theta}c_\theta\\
&+\theta\Delta c_\theta
-2\nabla V\cdot\nabla c_\theta
-(\Delta V)c_\theta
+\frac{|\nabla V|^2}{\theta}c_\theta.
\end{aligned}
$$

부호를 확인하는 좋은 방법은 $b_\theta=-\nabla f$, $V=f$, $c_\theta=1$인 가역적 경우를 넣는 것이다. 모든 항이 정확히 상쇄되어야 한다.

### 2단계: $\theta^{-1}$ 차수 — Hamilton–Jacobi 방정식

가장 큰 항의 계수는

$$
|\nabla V|^2-\nabla f_0\cdot\nabla V=0.
$$

$V=f_0$가 해다. 그러나 <strong>이 식만으로 해가 유일해지는 것은 아니다.</strong> 상수 함수도 식 자체를 만족한다. 실제 정상밀도와 맞는 action의 선택에는 최소점에서의 조건, 외부와의 연결 또는 준퍼텐셜의 선택 원리가 필요하다. 원문은 Assumption B.2의 유일성 가정을 사용해 $V=f_0$를 택한다. 단순히 방정식을 대입해 맞았다는 이유만으로 유일성을 증명했다고 읽으면 안 된다.

### 3단계: $\theta^0$ 차수 — 법선 방향의 transport

$V=f_0$를 넣고 다음 차수를 비교하면

$$
\nabla f_0\cdot(\nabla c_0+c_0\nabla f_1)=0.
$$

$\widetilde c_0=e^{f_1}c_0$로 두면 식이 간단해진다.

$$
\nabla f_0\cdot\nabla\widetilde c_0=0.
$$

하강 흐름 $\dot x=-\nabla f_0(x)$를 따라

$$
\frac{d}{dt}\widetilde c_0(x(t))
=-\nabla f_0\cdot\nabla\widetilde c_0=0.
$$

즉 같은 흐름선을 타고 최소점으로 내려가는 점들은 같은 $\widetilde c_0$ 값을 가진다. 이 단계는 법선 쪽 값을 연결한다. <strong>다양체 위에서 서로 다른 두 점의 값까지 같다고 말하지는 않는다.</strong>

### 4단계: $\theta^1$ 차수를 다양체에서 본다

$x\in\mathcal M$에서는 $\nabla f_0=0$이다. 이 사실을 이용하면 남은 식은

$$
(\Delta f_1)c_0+\nabla f_1\cdot\nabla c_0+\Delta c_0=0,
$$

또는

$$
\Delta\widetilde c_0-\nabla f_1\cdot\nabla\widetilde c_0=0
\qquad\text{on }\mathcal M
$$

가 된다. 여기서 $\Delta$는 아직 <strong>주변 공간의 Laplacian</strong>이다. 이를 바로 $\Delta_{\mathcal M}$로 바꾸면 법선 미분 항을 잃는다.

### Lemma B.5: 주변 공간 미분을 접선·법선으로 분해한다

Laplace–Beltrami 연산자는 다양체 안의 gradient를 다시 divergence한 것이다. 좌표에서는

$$
\Delta_{\mathcal M}h
=\frac1{\sqrt{\det g}}
\partial_{u_i}\left(
\sqrt{\det g}\,[g^{-1}]_{ij}\partial_{u_j}h
\right).
$$

반복된 $i,j$는 합산한다. 단위원에서는 $\partial_u^2h$이며, 속도 $v(u)=|\Phi'(u)|$인 일반 곡선에서는 $v^{-1}\partial_u(v^{-1}\partial_uh)$다. 타원에서 각도 좌표로 단순한 $\partial_u^2$를 쓰면 호길이에 따른 확산을 놓친다.

좌표 $(u,r)$와 $G=J^\top J$를 쓰면

$$
\Delta h=\frac1{|\det J|}
\nabla_{u,r}\cdot\left(|\det J|G^{-1}\nabla_{u,r}h\right).
$$

$r=0$에서 전개하면 원문의 보조정리 B.5는 다음 식을 준다.

$$
\begin{aligned}
0={}&\Delta_{\mathcal M}\widetilde c_0
-(\partial_u\widetilde c_0)^\top g^{-1}\partial_u f_1\\
&+\frac{1}{\sqrt{\det g}}
(\partial_r|\det J|)^\top\partial_r\widetilde c_0
-(\partial_r\widetilde c_0)^\top\partial_r f_1
+\operatorname{Tr}(\partial^2_{rr}\widetilde c_0).
\end{aligned}
$$

혼합 metric 항을 없애는 데는 법선 기저의 직교 정규성이 쓰인다. $N_p$가 법선 기저의 $p$번째 열이면

$$
\langle N_p,\partial_{u_j}N_p\rangle
=\tfrac12\partial_{u_j}|N_p|^2=0.
$$

따라서 $d_{p,j}=\langle N_p,\partial_{u_j}N_q\rangle r_q$에서 나오는 $\sum_p\partial_{r_p}d_{p,j}$가 0이다. 곡률과 좌표 Jacobian을 무시한 결과가 아니라, 남는 항을 실제로 계산해 소거한 것이다.

### Lemma B.6: 법선 미분을 transport로 제거한다

transport 식을 좌표로 $a(u,r)\cdot\partial_u\widetilde c_0+b(u,r)\cdot\partial_r\widetilde c_0=0$라 쓰자. $r=0$에서는 $a=b=0$이다. 최소점 집합을 따라 기울기가 0이므로 Hessian의 접선·혼합 성분도 0이고, metric의 block 구조를 사용하면 $\partial_r a=0$이다. 법선 Hessian을 $H_N$이라 하면 $\partial_r b=H_N$이다.

한 번 법선 미분한 식은

$$
H_N\,\partial_r\widetilde c_0=0.
$$

$H_N$이 양의 정부호이므로 $\partial_r\widetilde c_0=0$이다. 두 번 미분하면 법선 Hessian $C=\partial^2_{rr}\widetilde c_0$에 대해 다음 형태의 Sylvester 방정식을 얻는다.

$$
H_NC+CH_N+\sum_i A_i(u)\,\partial_{u_i}\widetilde c_0=0.
$$

$A_i$는 $a_i$의 법선 2차 미분에서 오는 알려진 행렬이다. 왼쪽에 $H_N^{-1}$을 곱하고 trace를 취하면

$$
2\operatorname{Tr}C+
\sum_i\operatorname{Tr}(H_N^{-1}A_i)\,
\partial_{u_i}\widetilde c_0=0.
$$

결국 $\operatorname{Tr}C$는 접선 1차 미분의 선형결합이다. <strong>$\widetilde c_0$ 자체에 비례하는 0차 항이 생기지 않는 것</strong>이 최대원리를 적용할 수 있게 한다.

### 5단계: 강한 최대원리

법선 항을 대체하면 다양체 위에 닫힌 식

$$
\Delta_{\mathcal M}\widetilde c_0+
B_{\mathcal M}\cdot\nabla_{\mathcal M}\widetilde c_0=0
$$

을 얻는다. 주계수는 $g^{-1}$이므로 국소적으로 균일 타원형이다. 경계 없는 compact 다양체에서는 함수가 최댓값을 달성한다. <strong>강한 최대원리(strong maximum principle)</strong>는 연결된 영역에서 이 타원형 방정식의 해가 내부 최댓값을 달성하면 상수임을 말한다. 계수의 국소 유계성과 정칙성도 전제한다.

직관은 확산이 주변보다 엄격히 높은 정지 봉우리를 유지할 수 없다는 것이다. 보통 최대원리는 상한을 제어하고, 강한 최대원리는 등호가 생겼을 때 영역 전체가 평평함을 결론낸다. 원문은 Gilbarg–Trudinger Theorem 3.5를 인용한다. 적용되는 형태는 [Guerra–Orriols 강의노트 §3.2, Theorem 3.12](https://www.dpmms.cam.ac.uk/~adblg2/EllipticPDELent2026.pdf)에서도 확인했다. 원문 §5는 이 편미분방정식(partial differential equation, PDE)을 “parabolic PDE”라고 표현한다. [원문 §5](https://arxiv.org/html/2509.24912v2#S5)

<strong>리뷰 해석.</strong> 이 증명 단계의 연산자는 시간 변수가 없는 타원형 연산자다.

따라서 연결된 범위에서 $\widetilde c_0=C$, 즉

$$
V=f_0,\qquad c_0=C e^{-f_1}
$$

이다. 이것이 Theorem B.3의 결론이다.

### 보충: $f_0=d_{\mathcal M}$에서는 더 짧게 보인다

이 논문의 핵심 적용에서는 $f_0=\tfrac12|r|^2$다. 법선 흐름은 $u$를 고정하고 $r(t)=e^{-t}r(0)$로 움직인다. transport와 $r=0$에서의 연속성 때문에

$$
\widetilde c_0(u,r)=\widetilde c_0(u,0).
$$

모든 법선 미분이 사라져 다양체 위 식은

$$
\Delta_{\mathcal M}\widetilde c_0
-\nabla_{\mathcal M}f_1\cdot\nabla_{\mathcal M}\widetilde c_0=0.
$$

양변에 $\widetilde c_0e^{-f_1}$을 곱해 적분하고, 경계가 없다는 조건으로 부분적분하면

$$
0=-\int_{\mathcal M}e^{-f_1}
|\nabla_{\mathcal M}\widetilde c_0|^2\,d\mathcal M.
$$

따라서 연결 성분마다 상수다. 이는 일반 보조정리의 모든 기하 항을 계산하는 대신, <strong>거리 제곱이라는 특별한 구조</strong>를 사용한 해설자의 보충 증명이다.

### 리뷰 해석: 최대원리 적용의 조건

<strong>리뷰 해석.</strong> 이 계산의 정당화에는 다음 조건이 쓰인다.

1. <strong>연결성:</strong> 최대원리는 연결 성분마다 상수를 준다. $\mathcal M$의 연결성이 없으면 서로 다른 성분 사이의 상수와 질량 비율을 별도로 결정해야 한다. 본문 Assumption 2.1에는 연결성이 명시되어 있지 않으므로, 위 증명만으로 전 성분에 하나의 상수가 정해졌다고 읽을 수 없다.
2. <strong>잔차 미분:</strong> $\widehat b_\theta=o(\theta)$라는 함수값의 균일 제어만으로 $\nabla\cdot\widehat b_\theta=o(\theta)$가 따라오지 않는다. 원문의 차수 비교는 잔차를 미분하므로, 해당 미분 제어를 추가하거나 약한 Fokker–Planck 형식으로 극한을 정당화해야 한다. 또한 $\Delta f_1$를 점별로 쓰려면 단순 $C^1$보다 강한 정칙성 또는 약한 해 해석이 필요하다.
3. <strong>WKB 자체:</strong> 양의 $C^2$ 진폭, 선택된 action, 관상근방 집중은 가정이다. 모든 비보존적 neural score에 대한 자동 보장은 아니다.

이들은 원문의 증명에서 확인해야 할 조건을 지적한 것이다. 이 해설이 모든 조건 아래의 결론에 대한 반례를 확립했다는 뜻은 아니다. 기울기 score의 Theorem 5.1은 별도의 Gibbs·Laplace 논증이 있으므로 이 WKB 문제와 분리해 읽을 수 있다.

[4편](/reviews/score-geometry-weak-proof/)은 반사 원환과 추가 전역 조건을 둔 전체 평면 모형을 다룬다.

## B.5. score의 나머지와 온도 선택 (Proof for Theorems 5.1–5.2)

원문 위치: 부록 B.5의 식 (20)과 뒤따르는 추정, PDF pp. 28–31. 필요한 정밀도는 log-density의 점별 전개를 형식적으로 미분해서 얻지 않는다.

### 식 (20): Gaussian score의 균일한 나머지

분산 폭증(variance exploding, VE)의 정확한 score는 적분을 미분해

$$
s_\sigma^*(x)=
\frac{\int p_{\mathrm{data}}(u)
e^{-|x-\Phi(u)|^2/(2\sigma^2)}
(\Phi(u)-x)\,du}
{\sigma^2\int p_{\mathrm{data}}(u)
e^{-|x-\Phi(u)|^2/(2\sigma^2)}\,du}
$$

로 쓴다. 즉 Tweedie 형태 $s_\sigma^*=(\mathbb E[Y\mid X=x]-x)/\sigma^2$다. 최근접점 $P=P_{\mathcal M}(x)$를 빼면

$$
s_\sigma^*(x)+\frac{x-P}{\sigma^2}
=\frac{\mathbb E[Y-P\mid X=x]}{\sigma^2}.
$$

조건부로 $Y$가 $P$의 $O(\sigma)$ 근방에 있다는 것만 쓰면 오른쪽이 $O(\sigma^{-1})$다. 논문에 필요한 더 강한 결론은 <strong>중심화된 평균은 $O(\sigma^2)$</strong>라는 것이다.

이를 위해 [Majerski, Theorem 2.4 및 Corollary 2.5](https://arxiv.org/abs/1511.00302v2)의 더 정밀한 Laplace 오차를 사용한다. 최소점 근방의 phase가 $C^4$, amplitude가 $C^2$이고 비퇴화 조건과 적분 꼬리 조건이 갖춰지면, 적분의 상대 오차는 $O(\sigma^2)$다. 단순 $C^3/C^1$의 $O(\sigma)$ 결과보다 한 차수 좋다.

왜 좋아지는가? $u=u_*+\sigma z$로 놓으면 phase의 3차 항과 amplitude의 1차 항이 첫 $O(\sigma)$ 후보인데, 중심 Gaussian에 대해 홀함수이므로 적분이 0이다. 다음 짝수 항이 $O(\sigma^2)$를 만든다. 균일 오차를 얻으려면 최소점·Hessian·고차 미분·꼬리 상수의 균일성까지 확인해야 한다.

임의의 단위벡터 $v$에 대해 amplitude를

$$
a_v(u)=p_{\mathrm{data}}(u)
\left[1+v^\top(\Phi(u)-P)\right]
$$

로 선택하자. 최소점에서는 $a_v(u_*)=p_{\mathrm{data}}(u_*)>0$다. 분모의 amplitude $p_{\mathrm{data}}$와 같은 leading term을 가지므로 두 Laplace 근사의 비를 빼면

$$
\mathbb E[v^\top(Y-P)\mid X=x]=O(\sigma^2).
$$

앞의 $1$은 중심화한 amplitude가 최소점에서 0이 되어 상대 오차 정리를 바로 쓰지 못하는 문제를 피한다. $x$와 $v$에 대해 균일하게 정리하면

$$
\boxed{\quad
\sigma^2s_\sigma^*(x)+\nabla d_{\mathcal M}(x)=O(\sigma^2),
\qquad
s_\sigma^*(x)+\frac{\nabla d_{\mathcal M}(x)}{\sigma^2}=O(1).
\quad}
$$

분산 보존(variance preserving, VP)에서도 해당 보정이 이 $O(1)$ 나머지 안에 포함된다. 이것이 이후 WKB drift의 출발점이다.

### $\alpha$의 두 부등식은 어디서 오는가

실제 score 오차를 $e_\sigma=s_\sigma-s_\sigma^*$, $\|e_\sigma\|=o(\sigma^\beta)$라 하자. 재조정된 drift를 $b_\theta=\sigma^2s_\sigma$로 쓰면

$$
b_\theta
=-\nabla d_{\mathcal M}
+o(\sigma^{2+\beta})+O(\sigma^2),
\qquad \theta=\sigma^{2-\alpha}.
$$

WKB의 $f_1=0$ 경우에 넣으려면 잔차가 $o(\theta)$여야 한다. 두 나머지를 나누면

$$
\frac{o(\sigma^{2+\beta})}{\theta}
=o(\sigma^{\alpha+\beta})\to0
\quad\text{if }\alpha+\beta>0,
\qquad
\frac{O(\sigma^2)}{\theta}=O(\sigma^\alpha)\to0
\quad\text{if }\alpha>0.
$$

동시에 집중하려면 $\theta\to0$, 즉 $\alpha<2$가 필요하다. 세 조건을 모은 것이

$$
\max\{-\beta,0\}<\alpha<2
$$

다. $\beta>-2$는 이 열린 구간이 존재하게 하는 조건이다.

Theorem B.3이 정당화되는 범위에서는 $V=d_{\mathcal M}$, $c_0=C$를 얻는다. 그리고 Theorem B.1에서 법선 Hessian $H_N=I$를 사용하면 다양체 위 부피 균등분포가 나온다. 단계마다 필요한 역할이 다르다.

- score의 정밀한 Laplace 근사: drift의 형태를 보장한다.
- WKB: 비보존적 drift의 정상밀도 action과 진폭을 연결한다.
- 최소점 다양체 Laplace 정리: 그 정상밀도의 약극한을 계산한다.

### 기울기 score의 경우에는 밀도로 직접 증명한다

$s_\sigma=-\nabla f_\sigma$라면 TS 정상밀도는 $e^{-h f_\sigma}$, $h=\sigma^\alpha$다. $f_\sigma$와 $-\log p_\sigma$의 차이는 $K$에서 상수항을 제외하면 $O(E_\sigma)$이며, 기울기 오차를 경로 적분해 얻는다. 상수항은 밀도를 정규화하면 사라진다. 따라서

$$
-h f_\sigma=
-\frac{h}{\sigma^2}d_{\mathcal M}+o(1)
+\text{정규화에 흡수되는 상수}
$$

이다. 원래 밀도와 곡률의 유한 항은 $h\to0$로 지워진다. 법선 우물은 $h/\sigma^2\to\infty$로 유지된다.

집중을 확인할 때는 적분의 최솟값에 대한 부정확한 하한을 쓸 필요가 없다. $|r|\le\sqrt\theta$인 작은 관상근방을 적분하면, 상수 gauge를 제거한 분모는 $c\theta^{q/2}$ 이상이다. 반면 고정 거리 $\delta$ 바깥의 분자는 $Ce^{-c_\delta/\theta}$ 이하다. 따라서 그 비는

$$
C\theta^{-q/2}e^{-c_\delta/\theta}\to0.
$$

$K$ 밖의 질량은 해당 가정으로 처리한다. 이는 원문의 논증을 표준적인 shrinking-tube 하한으로 풀어 쓴 보충이다.

### 초록의 $o(\sigma^{-2})$와 고정 지수 정리는 같지 않다

다음 오차를 생각하자.

$$
E_\sigma=\frac{\sigma^{-2}}{\log(1/\sigma)}.
$$


$E_\sigma=\sigma^{-2}/\log(1/\sigma)$는 $o(\sigma^{-2})$지만 어떤 고정 $\beta>-2$에 대해서도 $o(\sigma^\beta)$가 아니다. 따라서 Theorem 5.1–5.2의 고정 $\alpha<2$ 조건을 그대로 적용할 수 없다.

기울기 score에서 앞의 논증은 더 일반적으로

$$
h\to0,\qquad h/\sigma^2\to\infty,\qquad hE_\sigma\to0
$$

를 요구한다. 예를 들어 위 로그 경계 오차에

$$
h(\sigma)=\sigma^2\sqrt{\log(1/\sigma)}
$$

를 쓰면 세 조건을 만족한다. 이 선택은 <strong>해설에서 제안한 가변 tempering의 충분조건</strong>이며, 논문에 적힌 고정 지수 정리를 그대로 인용한 결과는 아니다. 같은 국소 전개·정규화·tightness 조건을 함께 유지해야 한다.

## B.6. 다양체 위 베이지안 사후분포 (Proof for Theorem 6.1)

원문 위치: 부록 B.6, PDF p. 31.

관측 likelihood를 $e^{-v(x)}$라 쓰고 guidance를 $-\nabla v$로 더하면 실제 drift는

$$
-\nabla v+\sigma^\alpha s_\sigma.
$$

$\theta=\sigma^{2-\alpha}$로 시간 재조정한 drift는

$$
b_\theta^v=\sigma^2s_\sigma-\theta\nabla v
=-\nabla d_{\mathcal M}-\theta\nabla v+o(\theta).
$$

따라서 WKB 일반식에서 $f_0=d_{\mathcal M}$, $f_1=v$다. 진폭이 $e^{-v}$이고 법선 Hessian은 $I$이므로

$$
\pi_\theta^v\Rightarrow
\frac{e^{-v(x)}\,d\mathcal M(x)}
{\int_{\mathcal M}e^{-v(y)}\,d\mathcal M(y)}.
$$

기울기 score에서는 $e^{-v-\sigma^\alpha f_\sigma}$를 직접 적분해 같은 결론을 얻는다. 일반 score까지 확장하는 해석에는 앞 절의 WKB 조건이 따라온다.

<strong>어떤 사전분포인가?</strong> 원래 데이터 밀도 $\rho$를 prior로 유지한 posterior는 $\rho e^{-v}d\mathcal M$다. 여기서는 $\rho$를 지운 부피 균등 prior를 쓴다. 의도적으로 다른 posterior를 만드는 것이다. 균등분포는 유한 부피 다양체에서 부피측도를 기준으로 한 미분 엔트로피를 최대화한다. Kullback–Leibler(KL) 발산을 $D_{\mathrm{KL}}$로 쓰면 다음 관계가 성립한다.

$$
H(\eta)=\log\operatorname{Vol}(\mathcal M)
-D_{\mathrm{KL}}\!\left(
\eta\,\middle\|\,\frac{d\mathcal M}{\operatorname{Vol}(\mathcal M)}
\right).
$$

엔트로피는 기준측도에 의존하므로 좌표 Lebesgue 균등과 혼동하면 안 된다.

원문 정리는 $v$가 전역 유계이고 관상근방에서 $C^1$인 조건을 둔다. Gaussian 관측의 이차 손실은 전역 유계가 아니므로, 그것을 예로 사용할 때는 꼬리 제어나 cutoff에 대한 별도 논증이 필요하다. 분류기 없는 유도(classifier-free guidance, CFG)의 조건부 score 차이가 실제로 고정된 $-\nabla v$를 근사하는지, 유한 predictor–corrector가 정상분포에 얼마나 가까운지는 또 다른 문제다.

## C. 실험 설정과 추가 실험 (Experimental Details and Further Experiments)

원문 위치: 부록 C.1–C.4, PDF pp. 31–34. Tables 3–4는 p. 34 상단에 배치되어 있다. 본문 §7의 Table 2도 설정을 함께 비교할 수 있도록 이곳에 모았다.

### C.1 타원·원 데이터와 학습된 score

타원은 $(a,b)=(1,2)$, 원은 $(1,1)$이고 좌표는 $(a\cos u,b\sin u)$다. 데이터의 각도 $u$는 농도 $\kappa=1$인 von Mises 분포를 따른다. 타원에서 각도 균등과 호길이 균등이 다르므로 Jacobian 보정이 시각적 결론에도 중요하다.

score 네트워크는 hidden dimension 128의 4-block Transformer이며, $\widehat s_\sigma=\sigma^2s_\sigma$를 예측한다. 잡음 제거 score matching(denoising score matching, DSM) 목적함수는

$$
\frac12\mathbb E\left[
\sigma^2\left|s_\sigma(X)+\frac{X-\Phi(u)}{\sigma^2}\right|^2
\right]
=\frac12\mathbb E\left[
\frac{|\widehat s_\sigma(X)+X-\Phi(u)|^2}{\sigma^2}
\right].
$$

잡음 범위는 $[0.01,50]$, batch size 1024, AdamW weight decay $10^{-4}$, gradient clipping 1, 초기 learning rate $3\times10^{-3}$다. 원문은 cosine schedule을 $4\times10^{-4}$ steps에 걸쳐 초기값의 1%까지 낮춘 뒤 $4\times10^{-4}$의 일정한 학습률로 계속한다고 적는다. 단계 수가 분수로 인쇄되고 1%인 $3\times10^{-5}$와 뒤의 값도 달라 의도한 일정이 불분명하다. 인쇄 수치를 임의의 학습 단계 수로 고쳐 쓰지 않는다. [원문 C.1](https://arxiv.org/html/2509.24912v2#A3.SS1)

sampling은 $\sigma_{\min}=0.01$, $\Delta t=0.1$, 10,000 steps와 10,000 runs를 사용한다. 부록의 구현은 $\widehat s$를 drift로 쓰고 vanilla 잡음을 $\sqrt{2\sigma^2}$, TS 잡음을 $\sqrt{2\sigma^{2-\alpha}}$로 쓴다. 이 시간 재조정은 정상분포 비교에는 맞지만 실제 수치 시간과 수렴 속도는 구분해야 한다.

### C.2 Stable Diffusion 1.5

이미지 평가는 대조적 언어·이미지 사전학습(Contrastive Language–Image Pre-training, CLIP) 특징을 쓴다. 프롬프트 유사도(prompt similarity, P-sim)는 높을수록 정렬이 좋고, 이미지 간 유사도(image-to-image similarity, I-sim)는 낮을수록 이 특징 공간에서 다양하다. 표의 P / I는 이 두 지표이며 원문 단위를 유지한다. 비교 방법은 잡음 제거 확산 확률 모델(Denoising Diffusion Probabilistic Models, DDPM), 예측–보정(predictor–corrector, PC), 그리고 TS다.


DDPM predictor 30 steps, CFG 7.5, 설정당 512장, CLIP 계산 시 $256\times256$ 크기를 사용한다. corrector 횟수는 $\{5,10,15,20,30\}$, $\alpha$는 $\{0.1,0.5,1,1.5\}$를 비교한다. 마지막 corrector 단계에는 두 방법 모두 잡음 없는 unconditional projection을 추가한다.

P-sim은 이미지와 prompt embedding의 평균 cosine similarity라서 높을수록 좋다. I-sim은 이미지 쌍 embedding의 평균 cosine similarity라서 낮을수록 다양하다고 해석한다. I-sim 감소가 모든 종류의 의미적 다양성·사실성·선호도 향상을 증명하지는 않는다. 원문 표의 수치 표기를 그대로 유지하며, 명시되지 않은 단위 변환을 추가하지 않는다.

신경망의 실제 출력은 잡음을 예측하는 $\epsilon(x,t,y)$다. $y$는 prompt 조건, $\emptyset$은 무조건 입력, $w$는 guidance 세기다. 전방 과정에서 누적 신호 전력인 $\bar\alpha_t$에 대해 $\sigma_t=\sqrt{1-\bar\alpha_t}$로 두면, 원문의 표준·조절 score는 다음과 같다.

$$
\begin{aligned}
s_t(x,y)&=-\frac1{\sigma_t}\left[
\epsilon(x,t,\emptyset)+w\big(\epsilon(x,t,y)-\epsilon(x,t,\emptyset)\big)\right],\\
\widetilde s_t(x,y)&=-\frac1{\sigma_t}\left[
\sigma_t^\alpha\epsilon(x,t,\emptyset)+w\big(\epsilon(x,t,y)-\epsilon(x,t,\emptyset)\big)\right].
\end{aligned}
$$

DDPM predictor가 잡음 수준 $t_i$로 이동하면 corrector를 $n_{\mathrm{corr.}}$회 실행한다. 그 안의 한 단계는

$$
x_{k+1}=x_k+\delta_i\widetilde s_{t_i}(x_k,y)
+\sqrt{2\delta_i}\,\xi_k,\qquad \xi_k\sim\mathcal N(0,I).
$$

이다. $\delta_i$는 corrector 스텝 크기이며 원문은 Song 등의 Algorithm 5를 따른다고 명시한다. 역과정 전체가 끝난 뒤에는 최종 잡음 수준 $t_0$의 무조건 score만으로

$$
dx_\tau=\nabla\log p_{t_0}(x_\tau)\,d\tau
$$

라는 결정적 사영 흐름을 $n_{\mathrm{corr.}}$단계 적용한다. 이 단계에는 guidance와 잡음이 없다. PC와 TS의 사영 횟수는 같으므로 비교에서 이 후처리의 차이가 제안법의 효과에 섞이지 않는다. [원문 C.2, Algorithm details](https://arxiv.org/html/2509.24912v2#A3.SS2)

아래는 본문 §7, PDF p. 11의 Table 2 전체 값이다. 각 셀은 P-sim / I-sim이다.

| Prompt | Correctors | PC | TS, $\alpha=1$ |
| --- | ---: | ---: | ---: |
| Furniture | 5 | 29.40 / 81.34 | 29.54 / 81.11 |
| Furniture | 10 | 29.30 / 81.24 | 29.58 / 80.95 |
| Furniture | 15 | 29.32 / 81.64 | 29.68 / 81.34 |
| Furniture | 20 | 28.98 / 81.72 | 29.52 / 81.15 |
| Furniture | 30 | 28.67 / 82.33 | 29.43 / 81.87 |
| Car | 5 | 26.20 / 87.20 | 26.23 / 87.14 |
| Car | 10 | 26.30 / 87.57 | 26.37 / 87.42 |
| Car | 15 | 26.24 / 87.98 | 26.32 / 87.88 |
| Car | 20 | 26.26 / 88.06 | 26.28 / 88.07 |
| Car | 30 | 26.17 / 87.94 | 26.20 / 87.87 |
| Architecture | 5 | 27.13 / 81.83 | 27.23 / 81.58 |
| Architecture | 10 | 27.13 / 81.81 | 27.27 / 81.57 |
| Architecture | 15 | 26.92 / 81.64 | 27.14 / 81.54 |
| Architecture | 20 | 26.87 / 81.60 | 27.06 / 80.97 |
| Architecture | 30 | 26.60 / 81.03 | 26.84 / 80.76 |

동일 corrector 횟수끼리 비교하면 P-sim은 15/15에서 높고 I-sim은 14/15에서 낮다. Car의 20회 설정은 I-sim이 0.01 높다. 오차 막대와 여러 seed의 불확실성 보고가 없으므로 이 작은 차이에 통계적 의미를 부여할 수 없다. 본문 Table 1의 best metric 요약은 두 지표의 최적 corrector 횟수가 같다고 보장하지 않는다.

원문은 이 이미지 실험의 하드웨어별 실행 시간이나 총 학습·추론 비용을 별도 수치로 보고하지 않는다. corrector 횟수를 늘리면 추가 score 평가가 필요하므로, 유사도 지표의 변화와 실제 계산 비용의 변화는 별도로 평가해야 한다.

### C.3 통제된 score 오류

단위원 위 von Mises 밀도는 $\kappa=4$, 중심 각도 $\pi$다. 정확한 score에

$$
e(x)=-\nabla\left(\frac12|x-(1,0)|^4\right)
$$

를 더해 오류를 통제한다. 이 오류는 compact $K$에서 $\sigma$에 관해 $O(1)$이지만, $x$에 대해 전역 유계인 벡터장은 아니다. 따라서 “$O(1)$ 오류”의 변수와 norm 범위를 밝히는 것이 필요하다. 결과는 오류가 접선 분포를 바꿀 수 있고 tempering이 그 편향을 줄이는 예를 제공한다. 모든 학습 오류의 형태를 재현한 실험은 아니다.

![원문 Fig.4의 통제된 score 오류 비교: 왼쪽은 확산, 오른쪽은 조절된 score Langevin](/assets/score-geometry/paper-figure-4.svg)

원문 Fig.4. 가로축은 각도, 세로축은 밀도다. 검은 선은 원래 von Mises 데이터 빈도이며, 오른쪽의 균등 표적은 그 빈도를 평평하게 만드는 목표다. 같은 $O(1)$ score 오류를 사용한 비교다. [원문 Fig.4](https://arxiv.org/html/2509.24912v2#A3.F4)

### C.4 하이퍼파라미터 민감도

아래는 원문 Tables 3–4를 합쳐 $\alpha$ sweep을 빠짐없이 보존한 것이다. $\alpha=0$은 PC와 대응한다.

| Prompt | $\alpha$ | 10 correctors: P / I | 20 correctors: P / I |
| --- | ---: | ---: | ---: |
| Architecture | 0 | 27.13 / 81.81 | 26.87 / 81.60 |
| Architecture | 0.1 | 27.12 / 81.73 | 26.85 / 81.56 |
| Architecture | 0.5 | 27.14 / 81.67 | 26.97 / 81.49 |
| Architecture | 1 | 27.27 / 81.57 | 27.06 / 80.97 |
| Architecture | 1.5 | 27.32 / 81.52 | 27.10 / 81.13 |
| Furniture | 0 | 29.30 / 81.24 | 28.98 / 81.72 |
| Furniture | 0.1 | 29.32 / 81.37 | 28.99 / 81.65 |
| Furniture | 0.5 | 29.33 / 81.06 | 29.07 / 81.40 |
| Furniture | 1 | 29.58 / 80.95 | 29.52 / 81.15 |
| Furniture | 1.5 | 30.16 / 80.76 | 30.20 / 81.39 |
| Car | 0 | 26.30 / 87.57 | 26.26 / 88.06 |
| Car | 0.1 | 26.30 / 87.58 | 26.26 / 88.09 |
| Car | 0.5 | 26.31 / 87.44 | 26.25 / 87.95 |
| Car | 1 | 26.37 / 87.42 | 26.28 / 88.07 |
| Car | 1.5 | 26.50 / 87.34 | 26.62 / 87.70 |

고온일수록 모든 지표가 단조롭게 좋아지는 것은 아니다. 예를 들어 Architecture 20회에서 $\alpha=1.5$의 I-sim은 $\alpha=1$보다 높다. 이론의 $\alpha<2$ 조건은 현실 이미지에서 최적의 $\alpha$를 지정하는 식이 아니다.

## D. 조절된 score Langevin의 수렴 (Convergence of TS Langevin)

원문 위치: 부록 D.1–D.3, PDF pp. 34–37. 이 부록은 접선 장벽을 낮추는 효과를 Poincaré 부등식으로 연결한다.

### D.1 Poincaré 상수는 무엇인가

확률측도 $\pi$에 대한 Poincaré 부등식(Poincaré inequality, PI)은

$$
\operatorname{Var}_\pi(h)
\le\frac1{C_{\mathrm{PI}}}
\int|\nabla h|^2\,d\pi
$$

다. 이 글과 논문은 <strong>큰 $C_{\mathrm{PI}}$가 빠른 혼합</strong>을 뜻하는 spectral-gap 관례를 쓴다. 문헌에 따라 그 역수를 Poincaré constant라 부르므로 표기를 먼저 확인해야 한다. $h$는 함수와 약한 미분의 제곱이 적분 가능한 가중 Sobolev 공간 $H^1(\pi)$에서 고려한다.

가역 Langevin $dX=-\nabla U\,dt+\sqrt2\,dW$의 정상밀도는 $\pi\propto e^{-U}$다. $r_t=p_t/\pi-1$이라 놓으면

$$
\chi^2(p_t\|\pi)=\int r_t^2\,d\pi,\qquad
\frac{d}{dt}\chi^2(p_t\|\pi)
=-2\int|\nabla r_t|^2\,d\pi
\le-2C_{\mathrm{PI}}\chi^2(p_t\|\pi).
$$

따라서 초기 $\chi^2$가 유한한 경우

$$
\chi^2(p_t\|\pi)
\le e^{-2C_{\mathrm{PI}}t}\chi^2(p_0\|\pi).
$$

원문의 $e^{-C_{\mathrm{PI}}t}$ 상계는 이보다 느슨하지만 유효하다. 점질량 초기화는 처음부터 유한 $\chi^2$라는 조건을 만족하지 않으므로 smoothing 이후의 논증이 필요하다.

단위원 균등분포에서는 Fourier 모드 $e^{iku}$가 $-\Delta_{\mathcal M}$의 고유함수이고 고유값은 $k^2$다. 첫 비영 고유값은 1이다. 법선 우물이 아무리 강해도 접선 방향에 깊은 두 봉우리 사이의 장벽이 남으면 전역 혼합은 느릴 수 있다.

### D.2 장벽을 제거하는 메커니즘

원문은 단위원 주변 potential

$$
U_\sigma(x)=\frac{d_{\mathcal M}(x)}{\sigma^2}
+\sigma^\beta\phi(x),\qquad -2<\beta<0
$$

를 고려한다. $\phi(x)=(|x_1|-1)^2$는 원 위에 두 선호 영역을 만든다. $\sigma^\beta\to\infty$이므로 접선 장벽이 커지고, 적절히 매끄러운 이중 우물에서는 Eyring–Kramers 유형의 지수적으로 느린 혼합을 예상한다.

TS를 적용하면

$$
U_\sigma^{\mathrm{TS}}=
\frac{d_{\mathcal M}}{\sigma^{2-\alpha}}
+\sigma^{\alpha+\beta}\phi.
$$

$\alpha+\beta>0$이면 접선 장벽은 사라지고 $\alpha<2$이면 법선 집중은 남는다. <strong>지우고 싶은 장벽과 남기고 싶은 장벽의 척도가 다르기 때문에</strong> 가능한 선택이다.

다만 원문의 $\phi$는 $x_1=0$에서 미분 불가능하고 전역 oscillation도 무한하다. 매끄러운 Eyring–Kramers 정리나 전역 유계 perturbation 정리를 이 식에 문자 그대로 적용해서는 안 된다. 예를 들어 해설용으로

$$
\phi_{\mathrm{smooth}}(x)=\frac{x_2^2}{1+|x|^2}
$$

를 쓰면 전역 매끄럽고 $0\le\phi_{\mathrm{smooth}}\le1$이며, 단위원에서 $\tfrac12\sin^2u$라는 이중 우물 장벽을 만든다. 이것은 <strong>정칙성을 갖춘 대체 toy 예제</strong>다. 원문 실험을 다시 실행한 결과로 제시하지 않는다.

<details class="review-background">
<summary><span>보충 설명: Holley–Stroock 섭동 정리</span><span class="review-background-toggle" aria-hidden="true"></span></summary>
<div class="review-background-body">

$\mu\propto e^{-U}$이고 $\nu\propto e^{-U-\delta U}$라 하자. $\operatorname{osc}(\delta U)=\sup\delta U-\inf\delta U<\infty$이면

$$
C_{\mathrm{PI}}(\nu)
\ge e^{-\operatorname{osc}(\delta U)}
C_{\mathrm{PI}}(\mu).
$$

직접 증명도 가능하다. $m\le d\nu/d\mu\le M$이면 $M/m=e^{\operatorname{osc}(\delta U)}$이고

$$
\operatorname{Var}_\nu h
=\inf_a\int(h-a)^2d\nu
\le M\operatorname{Var}_\mu h
\le\frac{M}{C_{\mathrm{PI}}(\mu)}
\int|\nabla h|^2d\mu
\le\frac{M}{mC_{\mathrm{PI}}(\mu)}
\int|\nabla h|^2d\nu.
$$

이것이 위 하한이다. [Menz–Schlichting Theorem 3.2](https://arxiv.org/abs/1202.1510v4)에서 적용 형태를 확인했다. 이 글이 참고한 진술은 Menz–Schlichting의 정리 3.2다.

</div>
</details>

bounded smooth $\phi$에 대해 $\sigma^{\alpha+\beta}\operatorname{osc}\phi\to0$이면 TS와 순수 거리 potential의 PI 상수는 이 하한에서 거의 같다. 반면 $\sigma^\beta\to\infty$인 vanilla에는 이 논리로 잡음에 독립적인 하한을 얻지 못한다.

### D.3 거리 potential의 매끄러운 근사

단위원에서 $R=|x|$라 쓰면 $d_{\mathcal M}=\tfrac12(R-1)^2$는 원점에서 미분 불가능하다. 원문은

$$
V_c(x)=\frac{R^2}2+\frac12-\sqrt{R^2+c^2},
\qquad c=\sigma^{3-\alpha}
$$

로 근사한다. 다음 계산을 직접 확인할 수 있다.

$$
|V_c-d_{\mathcal M}|\le c,\qquad
\nabla V_c=
\left(1-\frac1{\sqrt{R^2+c^2}}\right)x.
$$

따라서 $\theta=\sigma^{2-\alpha}$일 때 potential 차이의 oscillation은 $O(c/\theta)=O(\sigma)$다. Holley–Stroock은 이 근사에서 생긴 작은 변화가 spectral gap을 크게 악화시키지 않음을 연결한다.

최소점 집합은 $c<1$에서 반지름 $\sqrt{1-c^2}$인 원이다. 원점은 지역 최댓값이고, 그 부근의 음의 Laplacian이 Lyapunov 조건에 도움이 된다. 2차원에서

$$
\Delta V_c=
2-\frac{R^2+2c^2}{(R^2+c^2)^{3/2}}.
$$

절댓값의 전역 상계는 $c\to0$에서 커지지만 원점 부근에서는 음수다. 필요한 양의 부분의 상계와 멀리 떨어진 영역의 제어를 분리해야 한다.

### Polyak–Łojasiewicz(PL) 조건이 연결하는 것

Polyak–Łojasiewicz(PL) 조건은 최소점 근처에서

$$
|\nabla V|^2\ge\nu(V-V_{\min})
$$

라는 부등식이다. 함수가 convex일 필요는 없으며, 최소점이 한 점일 필요도 없다. 원 주변의 $V\approx\tfrac12r^2$에서는 $|\nabla V|^2\approx r^2=2(V-V_{\min})$이므로 접선이 평평해도 성립할 수 있다.

[Gong–He–Shen v2, Theorem 2 및 Theorem 4](https://arxiv.org/abs/2501.00429v2)는 이런 국소 구조, 최소점 외 임계점과 꼬리에 대한 조건을 사용해 전역 PI를 최소점 관상근방의 국소 PI와 연결한다. 최소점 다양체 위의 Laplace–Beltrami spectral gap이 법선 집중 이후 남는 혼합을 설명한다. 해당 문헌도 고정 potential과 충분히 작은 온도에 대한 상수·문턱 조건을 가진다. 여기처럼 $c=c(\sigma)$로 potential 자체가 변하면 상수와 허용 온도를 다시 균일하게 추적해야 한다.

### Lyapunov 조건은 국소 결과를 어떻게 전역화하는가

생성자 $L_\theta=-\nabla V\cdot\nabla+\theta\Delta$와 $W=e^{V/(2\theta)}$를 쓰면

$$
\frac{L_\theta W}{\theta W}
=\frac{\Delta V}{2\theta}
-\frac{|\nabla V|^2}{4\theta^2}.
$$

우변이 $-\lambda+b\mathbf1_U$ 이하이고, $U$ 내부의 조건부 측도가 PI 상수 $\rho_U$를 가지면 원문 Proposition D.1 형태의 결과는

$$
C_{\mathrm{PI}}\ge
\frac{\lambda\rho_U}{b+\rho_U}
$$

를 준다. 원문의 중복된 $\sigma$ 표기와 혼동하지 않도록 Lyapunov 감소율은 $\lambda$로 썼다. “대부분의 공간에서는 중심으로 돌아오는 힘이 있고, 남는 중심 영역 안에서는 이미 잘 섞인다”는 두 사실을 결합하는 원리다. [Menz–Schlichting Definition 3.7, Theorem 3.8](https://arxiv.org/abs/1202.1510v4)에서 관련 local-to-global 형식을 확인했다.

원문은 최소점 근처·그 바깥, 원점의 최대점 근처, 먼 꼬리, 그 나머지로 영역을 나누어 이 조건을 확인하는 개요를 제시한다. $c(\sigma)$에 따른 미분 상수와 국소 PI, 허용 온도의 균일성을 모두 기록해야 최종 차수를 엄밀하게 결론낼 수 있다.

### 리뷰 해석: 하한 차수와 전역 상수

원문 D.2는 직접 적용의 하한을 $C_{\mathrm{PI}}^{\mathrm{smooth}}=\Omega(c)$라 적고, 구조를 이용하면 $\Omega(1)$로 개선한다고 설명한다. D.3의 첫 문장은 직접 적용 결과를 $\Omega(1/\sigma)$라 적은 뒤 $\Omega(1)$로 개선한다고 쓴다. [원문 D.2–D.3, 파일 pp.36–37](https://arxiv.org/pdf/2509.24912v2#page=36)

<strong>리뷰 해석.</strong> 이 관례에서는 큰 하한이 강하므로 $\Omega(1/\sigma)$에서 $\Omega(1)$로의 변화는 개선이 아니다. D.2와 D.3의 직접 적용 차수에 표기 불일치가 있다는 판단의 확신은 높다. 일관된 수정 방향은 D.2의 $\Omega(c)$와 구조를 이용한 $\Omega(1)$을 구별하는 것이지만, 최종 하한을 확정하려면 $c(\sigma)$에 따른 상수와 허용 온도를 끝까지 추적해야 한다. 이 문제는 부록 D의 정량 혼합 차수에 영향을 주며, 본문 정리의 정상측도 약극한 논증과는 의존 경로가 다르다.

유효한 연구 메시지는 다음과 같다. <strong>법선 집중과 접선 장벽을 분리할 수 있는 분포에서는, 온도 조절이 기하를 남기면서 metastability를 줄일 수 있다.</strong> 정확한 상수와 유한 시간 오차는 분포·기하·초기화·수치화에 따라 별도로 분석해야 한다.


## 리뷰 해석과 남은 질문

<strong>리뷰 해석.</strong> 원문의 가장 강한 근거는 Assumption B.2 아래 action을 선택한 뒤, 법선 수송식의 미분으로 법선 Hessian을 제거하여 닫힌 접선 타원형 방정식을 얻는 계산이다. 연결된 성분에서 최대원리는 진폭의 상수성을 준다. 거리 제곱 퍼텐셜의 경우에는 가중 에너지 적분으로 같은 결론을 더 직접적으로 확인할 수 있다.

남는 정당화는 세 가지다. 함수값의 $o(\theta)$만으로 잔차의 발산을 같은 차수에서 버리려면 추가 근거가 필요하고, 점별 $\Delta f_1$을 쓰려면 명시한 정칙성과 맞는 해석이 필요하다. 연결성은 성분별 상수를 전체의 한 상수로 만드는 단계에 쓰인다. 이는 원문 결론의 반례가 아니라 제시된 증명의 연결 조건이다. [4편](/reviews/score-geometry-weak-proof/)은 잔차를 미분하지 않는 별도 단위원 모형의 증명을 제시한다.

혼합 분석에서는 장벽 제거의 정성적 효과와 최종 상수의 균일성을 구별한다. 원문의 비매끄럽고 전역 유계가 아닌 접선 퍼텐셜에는 매끄러운 장벽 정리와 유계 섭동 정리를 그대로 적용하기 어렵다. 잡음에 따라 변하는 매끄러운 근사의 상수·허용 온도까지 추적해야 정량 차수가 완성된다.

## 출처와 읽은 범위

- 원문: Xiang Li, Zebang Shen, Ya-Ping Hsieh, Niao He, *When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis*. 최초 공개 2025-09-29, 검토 버전 v2(2026-03-16), 검토일 2026-10-29. [공식 원문](https://arxiv.org/abs/2509.24912v2).
- 읽은 범위: 부록 B.4–B.6, C.1–C.4, D.1–D.3; Theorem B.3, Lemmas B.5–B.6, Eq.(20), Proposition D.1, Tables 2–4와 Fig.4.
- 추가 원전: [Bouchet–Reygner §3.1–3.3](https://arxiv.org/abs/1507.02104), [Majerski, Theorem 2.4·Corollary 2.5](https://arxiv.org/abs/1511.00302v2), [Menz–Schlichting, Theorems 3.2·3.8](https://arxiv.org/abs/1202.1510v4), [Gong–He–Shen v2, Assumptions 1–4·Theorems 2·4](https://arxiv.org/abs/2501.00429v2), [Guerra–Orriols §3.2, Theorem 3.12](https://www.dpmms.cam.ac.uk/~adblg2/EllipticPDELent2026.pdf). Gilbarg–Trudinger 책 전문은 독립 열람하지 않았으며 필요한 최대원리의 형태는 위 강의노트로 대조한다.
- 그림: 대표 이미지는 자체 제작 벡터 그림이다. 원문 Fig.4는 해당 결과의 해설을 위한 부분 인용이며, 저자·버전은 위 원문과 같다. 이 편에서 사용하는 Fig.4의 영역을 벡터로 추출한다. 원문에는 arXiv 영구 비독점 공개 라이선스가 표시되어 있다. 별도 예제 그림은 명시한 모형의 자체 계산이다.
- 확인하지 않은 것: 신경망 학습, Stable Diffusion 이미지 생성의 독립 재현과 정리의 형식 검증.
