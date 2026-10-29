---
title: "[4/4] Score Geometry: 균등 극한 증명"
description: "반사 경계를 둔 고리 영역과 전체 평면의 단위원 모형에서 잔차를 미분하지 않고 균등 극한과 사후분포를 증명한다."
date: "2026-10-29"
publishedAt: "2026-10-29T08:55:40+09:00"
updatedAt: "2026-10-29T15:42:19+09:00"
topics: ["생성 모델", "확률과 기하"]
visibility: "public"
lang: "ko"
translationKey: "score-geometry-weak-proof"
paperTitle: "When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis"
paperPublishedDate: "2025-09-29"
authors: ["Xiang Li", "Zebang Shen", "Ya-Ping Hsieh", "Niao He"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2509.24912v2"
thumbnail: "/assets/score-geometry/paper-figure-1.svg"
thumbnailAlt: "논문 Fig.1: 분포 복원과 다양체 기하 학습, 균등 탐색 목표 비교"
---

정상 방정식을 분포 비교에 쓰는 시험함수(test function)에 적분하면, 근사 뒤 남는 오차항인 잔차 drift(residual drift)를 미분하지 않고 균등 극한을 식별할 수 있다. 이 글은 온도 $\theta$와 잔차 크기 $\delta_\theta$에 대해 단위원을 둘러싼 고리 영역(annulus)에 반사 경계를 두고, 다양체까지 거리의 제곱을 평균한 법선(normal) 방향의 2차 모멘트(second moment) $M_\theta\le\delta_\theta^2+4\theta$를 얻고, 반사 경계가 없는 명시적 전역 모형까지 증명을 이어 간다. <strong>리뷰 해석.</strong> 작은 잔차의 큰 미분을 허용하면서도 결론을 얻는 보충 경로이며, 그 범위는 아래에 명시한 단위원 모형과 전역 조건이다.

이 시리즈의 4편이다. [[1/4] Score Geometry: 기하와 빈도](/reviews/score-geometry-manifold/) · [[2/4] Score Geometry: Laplace와 오차 분리](/reviews/score-geometry-laplace/) · [[3/4] Score Geometry: 정상분포와 혼합](/reviews/score-geometry-wkb/).

## 1 원문에서 무엇을 더 정당화하려는가

원문 부록 B.4는 확률미분방정식(stochastic differential equation, SDE)의 drift를

$$
b_\theta=-\nabla f_0-\theta\nabla f_1+\widehat b_\theta,
\qquad \|\widehat b_\theta\|_\infty=o(\theta)
$$

로 쓰고, 정상밀도에 $e^{-V/\theta}c_\theta$를 대입한다. 정상 Fokker–Planck 방정식(Fokker–Planck equation, FPE)은 밀도를 미분하는 편미분방정식(partial differential equation, PDE)이다. 이를 점별로 전개하면 $\nabla\cdot\widehat b_\theta$가 나타난다.

비교하는 강한 형태도 명시하자. 정상측도가 매끄러운 밀도 $p_\theta$를 갖는 경우에는

$$
\begin{aligned}
0&=-\nabla\cdot(b_\theta p_\theta)+\theta\Delta p_\theta,\\
-\nabla\cdot(\widehat b_\theta p_\theta)
&=-p_\theta\nabla\cdot\widehat b_\theta
-\widehat b_\theta\cdot\nabla p_\theta.
\end{aligned}
$$

다. 두 번째 줄은 곱의 미분 공식이다. 잔차의 함수값과 그 발산이 서로 다른 항으로 들어가는 지점이 여기다. 아래 약한 유도는 미분을 시험함수로 옮겨 이 강한 밀도 전개를 필요로 하지 않는다. [원문 부록 B.4, Eq.(14)](https://arxiv.org/html/2509.24912v2#A2.E14).

문제는 함수값이 작다고 그 미분까지 같은 차수로 작아지는 것은 아니라는 점이다. 예를 들어 $\theta^{3/2}\sin(u/\theta^2)$의 진폭은 $o(\theta)$지만 미분의 진폭은 $\theta^{-1/2}$다. 원 위의 주기성을 맞추려면 주파수를 정수로 고르면 된다.

이 관찰은 원문 결론의 반례가 아니다. <strong>어떤 항을 어느 차수에서 버릴 수 있는지 별도 근거가 필요하다</strong>는 뜻이다. 밀도와 drift를 직접 미분하는 대신, 미분을 고정된 시험함수 쪽으로 옮기면 그 문제를 피할 수 있다.

이 증명의 전략은 세 단계다.

1. 법선 방향에서 질량이 단위원으로 모임을 모멘트로 증명한다.
2. 각도만의 시험함수를 써서 법선 drift와 경계 반사를 제거한다.
3. 원 위에 남는 Laplacian의 정상측도가 균등분포임을 증명한다.

원문의 부록 B.4는 action과 진폭을 함께 식별하려 한다. 여기서 필요한 대상은 각도 분포의 극한이므로, 그보다 적은 정보를 구하는 접근이다.

## 2 반사 경계를 둔 고리 영역 모델과 정확한 가정

### 2.1 공간과 기호

$0<a<1/2$를 고정하고

$$
\mathcal A=\{x\in\mathbb R^2:1-a\le |x|\le1+a\},
\qquad r_-=1-a,\quad r_+=1+a
$$

를 두 동심원 사이의 고리 영역이라 하자. $x=r(\cos u,\sin u)$이고 $u$는 $2\pi$ 주기다. 단위원은 $r=1$이다. 이 글의 $r$는 <strong>반지름</strong>이며, 이전 해설의 법선 변위 $r$와 다르다. 법선 변위는 새 기호 $q=r-1$로 쓴다.

반지름 방향과 원을 따라가는 접선(tangent) 방향의 단위벡터는 $e_r=(\cos u,\sin u)$, $e_u=(-\sin u,\cos u)$다. 단위원까지의 거리 제곱의 절반은 $d_{\mathcal M}(x)=q^2/2$이고 기울기는 $q e_r$다.

### 2.2 동역학과 경계

우선 likelihood가 없는 경우를 다룬다.

$$
dX_t=\big[-q(X_t)e_r(X_t)+\varepsilon_\theta(X_t)\big]\,dt
+\sqrt{2\theta}\,dW_t
+n_{\mathrm{in}}(X_t)\,d\ell_t.
\tag{S1}
$$

$\theta>0$는 온도, $W_t$는 2차원 표준 Brownian motion이다. $\ell_t$는 경계에서만 증가하는 반사 조절 과정이고 $n_{\mathrm{in}}$은 안쪽 단위법선이다. 반사는 바깥 경계에서 안으로, 안쪽 경계에서는 원점에서 멀어지는 방향으로 작용한다.

각 $\theta$에서 계수가 충분히 매끄럽고 반사 확산이 잘 정의되며 정상분포 $\mu_\theta$가 존재한다고 둔다. 아래 핵심 증명은 <strong>그러한 모든 정상분포</strong>에 적용한다. 각 온도에서 필요한 미분 가능성·연속성 조건(regularity)은 허용하되, $\theta\to0$에서 drift 미분의 크기가 온도와 무관한 같은 상수 이내로 제한되는 유계성(boundedness)을 가져야 한다고 요구하지 않는다.

잔차 크기는

$$
\delta_\theta=\|\varepsilon_\theta\|_{L^\infty(\mathcal A)},
\qquad \delta_\theta/\theta\longrightarrow0
\tag{S2}
$$

다. $\varepsilon_\theta$는 기울기 벡터장일 필요가 없다.

각도 분포의 차이는 사건에 배정하는 확률이 최대 얼마나 다른지를 나타내는 전변동 거리(total variation distance, TV)로 측정한다.

### 2.3 보충 명제의 진술

각도만 남기는 사영(projection)을 $\operatorname{arg}:\mathcal A\to\mathbb S^1$라 하고, 각도 주변측도(marginal measure)를 $\nu_\theta=(\operatorname{arg})_\#\mu_\theta$라 하자. $\#$는 측도를 사영으로 옮기는 pushforward를 뜻한다. 원의 균등측도는 $\lambda(du)=du/(2\pi)$다.

<strong>보충 명제 S.1.</strong> 위 가정 아래

$$
M_\theta:=\int(r-1)^2\,d\mu_\theta
\le\delta_\theta^2+4\theta,
\tag{S3}
$$

이고 $\mu_\theta$는 단위원의 부피 균등측도로 약수렴한다. 더 강하게 각도 분포에는

$$
\|\nu_\theta-\lambda\|_{\mathrm{TV}}
\le C_a\sqrt{M_\theta}
+\frac{2\pi}{r_-}\frac{\delta_\theta}{\theta},
\qquad C_a=\frac{r_-+1}{r_-^2}
\tag{S4}
$$

가 성립한다. 상계가 1을 넘으면 TV 거리의 바로 알 수 있는 상한 1을 함께 쓰면 된다. 이 상계는 최적 상수를 주장하지 않는다.

추가한 것은 고정된 고리 영역과 정상 반사다. 반대로 Wentzel–Kramers–Brillouin(WKB) 형태, 정상밀도의 미분에 대한 균일 상계, 잔차 divergence의 작은 차수는 요구하지 않는다.

## 3 약한 정상 방정식과 경계항

### 3.1 왜 drift를 미분하지 않아도 되는가

내부 생성자(generator)는

$$
L_\theta f=b_\theta\cdot\nabla f+\theta\Delta f,
\qquad b_\theta=-q e_r+\varepsilon_\theta
\tag{S5}
$$

다. 경계에서 법선 미분이 0인 매끄러운 시험함수 $f$에 대해서 정상성은

$$
\int_{\mathcal A}L_\theta f\,d\mu_\theta=0
\quad\text{if }\partial_{n_{\mathrm{out}}}f=0
\tag{S6}
$$

를 준다. 잔차가 들어가는 곳은 $\int\varepsilon_\theta\cdot\nabla f\,d\mu_\theta$뿐이다. $f$의 미분은 우리가 직접 정하므로 $\varepsilon_\theta$의 미분은 필요 없다.

반사 확산의 정상측도를 시험함수의 생성자 적분으로 특징짓는 일반 배경은 [Kang–Ramanan, §2–3, 식 (6)과 Theorem 1, Portable Document Format(PDF) pp. 4–7](https://arxiv.org/pdf/1204.4969v1#page=4)에서 확인했다. 여기서는 그 일반 정리를 전부 적용하기보다 매끄러운 고리 영역의 필요한 항등식을 직접 사용한다.

### 3.2 경계항의 부호를 직접 확인한다

각 $\theta$에서 충분히 매끄러운 정상밀도 $p_\theta$를 갖는 경우로 계산을 보자. 확률 흐름(probability current)은 $J_\theta=b_\theta p_\theta-\theta\nabla p_\theta$다. 정상성은 $\nabla\cdot J_\theta=0$, 반사 경계는 $n_{\mathrm{out}}\cdot J_\theta=0$이다.

부분적분하면

$$
\begin{aligned}
\int_{\mathcal A}L_\theta f\,p_\theta dx
&=\int_{\mathcal A}J_\theta\cdot\nabla f\,dx
+\theta\int_{\partial\mathcal A}p_\theta
\partial_{n_{\mathrm{out}}}f\,dS\\
&=\theta\int_{\partial\mathcal A}p_\theta
\partial_{n_{\mathrm{out}}}f\,dS.
\end{aligned}
\tag{S7}
$$

따라서 법선 미분이 0이면 적분이 0이고, 법선 미분이 음이 아니면 적분도 음이 아니다. 시험함수의 경계 조건은 Neumann 조건이지만, <strong>밀도의 반사 조건은 일반적으로 $\partial_n p=0$이 아니라 경계를 통과하는 순흐름이 0인 무유출(no-flux) 조건 $n\cdot J=0$</strong>다. 둘을 구별해야 한다.

[Pavliotis의 저자 공개 강의본 §4.1, 식 (4.9)–(4.12)와 반사 경계 설명](https://wwwf.imperial.ac.uk/~pavl/PavliotisBook.pdf#page=83)은 이 흐름·경계 구분을 설명한다. 이 공개본은 2015-11-11 표기본이며 인쇄판과 페이지가 다를 수 있다.

반사 SDE에 Itô 공식을 적용해도 같은 부호를 얻는다. 반사가 $f$를 감소시키는 방향이면 정상 상태에서 그 감소를 내부 생성자의 양의 평균이 보상해야 한다.

## 4 법선 집중을 모멘트로 증명한다

거리 시험함수 $h(x)=q^2/2$를 택하자. 두 경계 모두에서 바깥 법선 미분은 $a>0$이다. 따라서 $h$에는 식 (S6)의 0 등식이 아니라

$$
\int L_\theta h\,d\mu_\theta\ge0
\tag{S8}
$$

를 써야 한다. 이 부호가 경계를 생략하지 않고 집중을 증명하는 핵심이다.

극좌표에서

$$
\nabla h=q e_r,\qquad
\Delta h=1+\frac qr=2-\frac1r\le2.
$$

그러므로

$$
L_\theta h=-q^2+q\,\varepsilon_{\theta,r}
+\theta\left(2-\frac1r\right).
$$

식 (S8)과 Cauchy–Schwarz 부등식을 쓰면

$$
M_\theta\le\delta_\theta\mathbb E_{\mu_\theta}|q|+2\theta
\le\delta_\theta\sqrt{M_\theta}+2\theta.
$$

$xy\le(x^2+y^2)/2$를 적용해

$$
M_\theta\le\tfrac12M_\theta+\tfrac12\delta_\theta^2+2\theta
\quad\Longrightarrow\quad
M_\theta\le\delta_\theta^2+4\theta.
$$

즉 법선 평균제곱 거리는 $O(\theta)$, 전형적인 법선 폭은 $O(\sqrt\theta)$다. 고정된 $\eta>0$에 대해 Markov 부등식은

$$
\mu_\theta(|r-1|>\eta)
\le\frac{\delta_\theta^2+4\theta}{\eta^2}\to0
$$

를 준다. 이것으로 집중을 <strong>가정하지 않고</strong> 증명했다. 아직 각도별 빈도는 결정하지 않았다.

## 5 각도 방정식과 전변동 오차

### 5.1 각도만의 시험함수

$2\pi$ 주기인 $C^2$ 함수 $\varphi(u)$를 $f(x)=\varphi(u)$로 확장한다. $r_->0$이므로 원점 특이점이 없고, 두 경계에서 법선 미분이 0이다.

$$
\nabla f=\frac{\varphi'(u)}r e_u,\qquad
\Delta f=\frac{\varphi''(u)}{r^2}.
$$

법선 복원력과 $\nabla f$는 직교한다. 식 (S6)을 $\theta$로 나누면

$$
0=\int\left[
\frac{\varphi''(u)}{r^2}
+\frac{\varepsilon_{\theta,u}(r,u)}{\theta r}
\varphi'(u)\right]d\mu_\theta.
\tag{S9}
$$

여기까지 모든 계산에서 잔차의 미분이 등장하지 않는다.

### 5.2 먼저 Fourier 모드로 극한을 이해한다

$|1-r^{-2}|\le C_a|r-1|$이므로

$$
\left|\int\varphi''\,d\nu_\theta\right|
\le C_a\sqrt{M_\theta}\|\varphi''\|_\infty
+\frac{\delta_\theta}{\theta r_-}\|\varphi'\|_\infty.
\tag{S10}
$$

$\varphi(u)=e^{iku}$, 정수 $k\ne0$를 넣으면

$$
\left|\int e^{iku}\,d\nu_\theta\right|
\le C_a\sqrt{M_\theta}
+\frac{\delta_\theta}{\theta r_-|k|}\to0.
$$

균등분포는 0번을 제외한 모든 Fourier 계수가 0이다. 삼각다항식이 원 위 연속함수를 균일 근사한다는 사실을 이용하면 $\nu_\theta\Rightarrow\lambda$다. 복소 시험함수는 실수부·허수부에 각각 정상 식을 적용하면 된다.

### 5.3 Poisson 방정식으로 모든 연속 시험함수를 한꺼번에 제어한다

각 모드별 수렴보다 더 정량적인 결과를 얻자. $\|g\|_\infty\le1$인 연속 주기함수에 대해

$$
\bar g=\frac1{2\pi}\int_0^{2\pi}g(u)\,du,\qquad
\varphi''=g-\bar g
\tag{S11}
$$

를 푼다. 우변의 평균이 0이므로 주기해가 존재한다. 구체적으로 $H(u)=\int_0^u(g(s)-\bar g)\,ds$를 놓고 $\varphi'=H-\bar H$라 택한 뒤 다시 적분하면 된다.

이 구성은

$$
\|\varphi''\|_\infty\le2,\qquad
\|\varphi'\|_\infty\le4\pi
$$

를 준다. $g$의 미분이나 진동수는 이 상수에 들어가지 않는다. 식 (S10)에 넣으면

$$
\left|\int g\,d\nu_\theta-\int g\,d\lambda\right|
\le2C_a\sqrt{M_\theta}
+\frac{4\pi}{r_-}\frac{\delta_\theta}{\theta}.
$$

TV 거리는 두 확률측도의 차이에 대해 위 supremum의 절반이다. 원 위의 유한 측도에서는 연속 시험함수로도 같은 supremum을 근사할 수 있다. 따라서 식 (S4)가 성립한다.

이는 <strong>각도 사영의 TV 수렴</strong>이다. 주변 공간에서 매끄러운 $\mu_\theta$와 원 위에만 놓인 극한측도의 TV 거리는 여전히 1이다. 주변 공간에 대해서는 약수렴을 말해야 한다.

### 5.4 주변 공간의 약수렴까지 연결한다

연속 함수 $F$를 고리 영역에서 택하자. 고리 영역은 닫히고 유계인 콤팩트(compact) 집합이므로, 그 위의 연속 함수는 균일 연속이다. 법선 집중 때문에

$$
\int\big[F(r e_r(u))-F(e_r(u))\big]\,d\mu_\theta\to0.
$$

나머지 원 위 적분은 각도 분포의 수렴으로 처리한다. 그러므로 $\mu_\theta$는 단위원의 부피 균등측도로 약수렴한다. 지지집합 집중과 각도 균등성의 두 단계를 결합한 것이다.

## 6 온도와 같은 차수의 오차에서는 무엇이 남는가

### 6.1 완전히 계산 가능한 경계 예제

$g(u)=-\kappa\cos u$, $\kappa>0$라 놓고

$$
U_\theta(r,u)=\frac{(r-1)^2}{2}+t_\theta g(u),
\qquad b_\theta=-\nabla U_\theta
$$

를 쓰자. 정상밀도는 주변 공간 부피 $dx=r\,dr\,du$에 대해 $e^{-U_\theta/\theta}$에 비례한다. 각도 주변밀도는 정확히

$$
\rho_\theta(u)=
\frac{\exp\!\big[\kappa(t_\theta/\theta)\cos u\big]}
{\int_0^{2\pi}\exp\!\big[\kappa(t_\theta/\theta)\cos s\big]\,ds}.
\tag{S12}
$$

반지름 적분의 Jacobian $r$도 각도와 무관하므로 이 식에서 약분된다. 잔차는 $-t_\theta\nabla g$이고 크기는 $\kappa t_\theta/r_-$다.

$t_\theta=\theta^p$로 두면 세 경우가 분명하다.

| 지수 | $t_\theta/\theta$ | 각도 극한 | 기하 |
| --- | --- | --- | --- |
| $p>1$ | 0으로 감 | 균등 | 단위원 집중 |
| $p=1$ | 1 | 원래 $\kappa$의 von Mises 밀도 | 단위원 집중 |
| $0<p<1$ | 무한대로 감 | $u=0$에 집중 | 여전히 단위원 위 |

즉 <strong>$o(\theta)$는 균등성을 모든 방향의 잔차에 대해 보장하는 충분조건</strong>이다. $O(\theta)$까지만 허용하면 비균등 극한이 남는 예가 있다. 다만 다음 절처럼 더 큰 순환 drift도 균등성을 보존할 수 있으므로 개별 벡터장마다 꼭 필요한 조건이라고 말하지 않는다.

### 6.2 원 논문의 $\sigma,\alpha,\beta$로 돌아간다

원문 부록 B.5의 재조정은

$$
\theta=\sigma^{2-\alpha},\qquad
b_\theta=\sigma^2s_\sigma
=-\nabla d_{\mathcal M}
+o(\sigma^{2+\beta})+O(\sigma^2)
$$

다. 그러므로

$$
\frac{\delta_\theta}{\theta}
\le o(\sigma^{\alpha+\beta})+O(\sigma^\alpha).
$$

$\alpha>0$, $\alpha+\beta>0$, $\alpha<2$가 각각 잔차 소멸과 정상 집중을 보장한다. 이것이 원문의 $\max\{-\beta,0\}<\alpha<2$와 같은 범위다.

반사 경계를 둔 고리 영역 모델에 한해서 위 TV 상계는

$$
\|\nu_\theta-\lambda\|_{\mathrm{TV}}
\le C\left[
\sigma^{1-\alpha/2}
+o(\sigma^{\alpha+\beta})+O(\sigma^\alpha)
\right]
$$

라는 오차 구조를 준다. 이 표현을 원 논문의 모든 생성 알고리즘에 대한 유한 시간 오차로 바꾸어 읽으면 안 된다. 각 온도에서 이미 정상분포를 대상으로 한다.

## 7 기울기장으로 나타낼 수 없는 drift와 큰 미분을 가진 정확한 예제

### 7.1 순환 흐름은 균등성을 망치지 않을 수 있다

$\varepsilon_\theta=\omega_\theta h(r)e_u$처럼 각도에 무관한 접선 drift는 원을 따라 확률을 회전시킨다. $h$가 일반적이면 벡터장의 curl은 0이 아니므로 퍼텐셜의 기울기로 나타낼 수 있는 보존장(conservative field)이 아니다.

그럼에도

$$
p_\theta(x)\propto w_\theta(r),
\qquad w_\theta(r)=e^{-(r-1)^2/(2\theta)}
$$

는 정상밀도다. 전체 확률 흐름은 $p_\theta\varepsilon_\theta$이고, 각도에 무관한 접선장이므로 divergence가 0이며 경계의 법선 흐름도 0이다. 정상성은 흐름이 <strong>없음</strong>이 아니라 그 divergence가 0임을 요구한다.

가역성과 정상 흐름의 차이는 [Pavliotis §4.6, Proposition 4.13과 식 (4.101)](https://wwwf.imperial.ac.uk/~pavl/PavliotisBook.pdf#page=109)과 연결된다. 여기서는 그 원리를 이 고리 영역에 직접 계산한 것이다.

### 7.2 잔차의 미분이 커져도 같은 밀도를 보존하게 만든다

여기서는 잔차가 각도에 빠르게 진동하도록 만들자. 다음은 이 해설의 구체적 구성이다.

$$
h(r)=1-\frac{(r-1)^2}{a^2},\qquad
\eta_\theta=\theta^{3/2},\qquad
k_\theta=\lceil\theta^{-2}\rceil.
$$

정수 주파수 $k_\theta$를 택해 원 위 주기성을 맞췄다. 흐름 함수(stream function)를

$$
\psi_\theta(r,u)=
\frac{\eta_\theta}{k_\theta}
h(r)w_\theta(r)\sin(k_\theta u)
$$

로 두고

$$
\mathcal J_{\theta,r}=\frac1r\partial_u\psi_\theta,\qquad
\mathcal J_{\theta,u}=-\partial_r\psi_\theta,\qquad
\varepsilon_\theta=\frac{\mathcal J_\theta}{w_\theta}
$$

라 정의한다. 혼합 미분이 상쇄되어 $\nabla\cdot\mathcal J_\theta=0$이다. $h(r_\pm)=0$이어서 경계 법선 흐름도 0이다.

실제 drift 성분은

$$
\begin{aligned}
\varepsilon_{\theta,r}
&=\frac{\eta_\theta h(r)}r\cos(k_\theta u),\\
\varepsilon_{\theta,u}
&=\frac{\eta_\theta}{k_\theta}
\left[\frac{(r-1)h(r)}\theta-h'(r)\right]
\sin(k_\theta u).
\end{aligned}
\tag{S13}
$$

고정 영역에서 $\|\varepsilon_\theta\|_\infty=O(\theta^{3/2})=o(\theta)$다. 하지만

$$
\|\partial_u\varepsilon_{\theta,r}\|_\infty
\asymp\eta_\theta k_\theta\asymp\theta^{-1/2}.
$$

미분의 크기는 발산한다. 또한 가중 divergence가 0이라는 식에서

$$
\nabla\cdot\varepsilon_\theta
=\frac{r-1}{\theta}\varepsilon_{\theta,r}
$$

를 얻는다. 이 divergence의 크기는 $O(\theta^{1/2})$이며, $\theta$로 나눈 크기는 일반적으로 발산한다. <strong>divergence 자체가 발산한다는 주장과는 다르다.</strong> 원문의 차수 비교에 필요한 작은 차수가 자동으로 따라오지 않음을 보여준다.

이 벡터장이 실제로 비보존적인지도 확인할 수 있다. $r=1$, $u=\pi/(2k_\theta)$에서

$$
\operatorname{curl}\varepsilon_\theta
=\eta_\theta\left[
k_\theta+\frac{\theta^{-1}+2a^{-2}}{k_\theta}
\right]>0.
$$

그런데 전체 정상 흐름은 $\mathcal J_\theta/Z_\theta$이므로 $p_\theta=w_\theta/Z_\theta$가 <strong>모든 $\theta$에서 정확한 정상밀도</strong>다. 각도 밀도는 정확히 균등하다. 큰 미분이 있다고 균등 결론이 깨지는 것은 아니며, 잔차를 개별 항으로 미분하기보다 보존 구조나 약한 방정식이 중요한 예다.

## 8 likelihood를 더하면 posterior도 식별할 수 있다

### 8.1 남는 접선 drift

$v$가 고리 영역에서 $C^2$라 하고

$$
b_\theta=-q e_r-\theta\nabla v+\varepsilon_\theta,
\qquad \delta_\theta=o(\theta)
$$

로 바꾸자. 법선 모멘트 증명은

$$
M_\theta\le
\left(\theta\|\nabla v\|_\infty+\delta_\theta\right)^2+4\theta
$$

를 준다. $v$는 compact 영역에서 유계다. 이는 원문 전체 공간의 전역 유계 가정을 그대로 대체한다는 뜻은 아니다.

$v_0(u)=v(1,u)$라 쓰면 각도 극한 방정식은

$$
\int\left[\varphi''(u)-v_0'(u)\varphi'(u)\right]d\nu(u)=0.
\tag{S14}
$$

법선 집중과 $v$의 연속 미분 덕분에 $r^{-2}\partial_uv(r,u)$를 $v_0'(u)$로 바꿀 수 있다.

고리 영역이 콤팩트하므로 임의의 온도 수열에서 약수렴 부분수열을 고를 수 있다. 법선 집중 때문에 그 극한은 단위원 위에 놓인다. 아래에서 가능한 각도 극한을 하나로 결정하면, 모든 부분수열에 같은 결론이 적용되어 전체 수렴을 얻는다.

### 8.2 약한 해에 밀도가 있다고 가정하지 않고 유일성을 보인다

후보는 $\rho_0(u)=e^{-v_0(u)}/\int e^{-v_0}$다. 임의의 연속 $g$에 대해 $g_0=g-\int g\rho_0\,du$로 두고

$$
\varphi''-v_0'\varphi'=g_0
\quad\Longleftrightarrow\quad
(\rho_0\varphi')'=\rho_0g_0
$$

를 풀자. $H(u)=\int_0^u\rho_0(s)g_0(s)\,ds$는 $H(2\pi)=H(0)=0$이다. 다음 선택은 $\varphi'$의 평균을 0으로 만든다.

$$
\varphi'(u)=\frac{H(u)+c}{\rho_0(u)},\qquad
c=-\frac{\int_0^{2\pi}H(u)/\rho_0(u)\,du}
{\int_0^{2\pi}1/\rho_0(u)\,du}.
$$

따라서 주기적인 $C^2$ 해 $\varphi$가 존재한다. 식 (S14)에 넣으면 $\int g\,d\nu=\int g\rho_0\,du$다. 모든 연속 함수에 대해 같으므로 $\nu=\rho_0\,du$다. 극한측도의 밀도나 진폭 매끄러움 조건을 먼저 가정하지 않았다.

결론적으로 반사 모델에서는

$$
\mu_\theta\Rightarrow
\frac{e^{-v}\,d\mathcal M}{\int_{\mathcal M}e^{-v}\,d\mathcal M}
$$

를 WKB 없이 얻는다. 원래 데이터 빈도 $\rho$를 prior로 유지한 $\rho e^{-v}d\mathcal M$와는 여전히 다른 목표다.

### 8.3 일반 접선 drift에는 확률 흐름이 남을 수 있다

$-\nabla v$ 대신 $\theta A(x)$를 넣으면 원 위의 극한 생성자는 $\varphi''+a(u)\varphi'$이며 $a(u)=A(1,u)\cdot e_u$다. 정상밀도는 일정한 흐름 $j=a\rho-\rho'$를 만족한다.

$B(u)=\int_0^u a(s)\,ds$, $F=B(2\pi)$라 하고 $B(u+2\pi)=B(u)+F$로 확장하면 양의 주기밀도는

$$
\rho(u)=C e^{B(u)}
\int_u^{u+2\pi}e^{-B(s)}\,ds,\qquad
j=C(1-e^{-F}).
\tag{S15}
$$

$C$는 적분이 1이 되게 하는 상수다. 적분을 미분하면 $\rho'=a\rho+C(e^{-F}-1)$가 되어 공식을 확인할 수 있다.

상수 $a=\omega$이면 $\rho=1/(2\pi)$, $j=\omega/(2\pi)$다. 반면 $a=-\kappa\sin u$이면 $j=0$, $\rho\propto e^{\kappa\cos u}$다. “접선 drift가 있다”와 “비균등하다”는 같은 뜻이 아니다.

![온도 대비 잔차와 원 위의 정상 순환 흐름](/assets/score-geometry/weak-stationarity-toy.svg)

별도 고리 영역 예제의 결정론적 계산. 왼쪽은 같은 원에서도 잔차/온도의 크기에 따라 각도 밀도가 달라지는 모습을, 오른쪽은 균등분포에도 순환 흐름이 있을 수 있음을 보여준다. 확산 계수가 1인 원 위에서 $a(u)=0.7$이면 $j=0.7/(2\pi)\approx0.111408$, $a(u)=0.7-1.2\sin u$이면 수치 정규화한 흐름은 약 0.068969다.

## 9. 전체 공간 모형과 가정

고리 영역에서 반사를 없애면 각도 시험함수가 원점에서 정의되지 않는 문제가 생긴다. 원 근처에서 1이고 원점 근처와 충분히 먼 곳에서 0인 절단함수를 곱해야 한다. 그 미분 항에는 $\theta^{-1}$이 곱해지므로, 전이 영역의 질량이 $o(\theta)$이면 이를 없앨 수 있다. 다음 명제는 전역 조건을 추가하여 그 제어를 완성한다. 명제와 식 (F1) 이후의 유도는 해설자의 보충 결과다.


$x\in\mathbb R^2$, $r=|x|$, $e_r=(\cos u,\sin u)$, $e_u=(-\sin u,\cos u)$로 쓴다. 온도 $\theta\downarrow0$에 대해 확률미분방정식(stochastic differential equation, SDE)을

$$
dX_t=[-\nabla U(X_t)+\varepsilon_\theta(X_t)]dt+\sqrt{2\theta}\,dW_t,
\qquad U(x)=\frac{(|x|^2-1)^2}{8}
\tag{F1}
$$

로 둔다. $W_t$는 평면 Brownian 운동이다. 이 퍼텐셜은 평면 전체에서 매끄럽고 최소점 집합은 단위원이다.

$$
U'(r)=\frac{r(r^2-1)}2,\quad U''(1)=1,\quad
U(1+q)=\frac{q^2}{2}+\frac{q^3}{2}+\frac{q^4}{8}.
\tag{F2}
$$

원 근처의 2차 법선 복원력은 반사 경계를 둔 고리 영역의 $q^2/2$와 같다. 원점에서도 매끄럽고 원점에서 멀어질수록 커진다는 장점이 있다.

각 $\theta$에서 잔차는 충분히 매끄럽고, 정규화된 매끄러운 양의 정상밀도 $p_\theta$가 존재한다고 가정한다. 이 존재·매끄러움 조건은 명제의 가정이다. 전역 상계는 $\delta_\theta=\sup_{x\in\mathbb R^2}|\varepsilon_\theta(x)|=o(\theta)$이며, 잔차의 미분에는 $\theta$에 균일한 상계를 요구하지 않는다.

<strong>보충 명제 F.1.</strong> 위 조건이면 각도 사영의 분포 $\nu_\theta$는 원의 균등 각도 측도 $\lambda(du)=du/(2\pi)$에 전변동 거리(total variation distance, TV)로 수렴한다. 주변 공간의 정상측도 $\mu_\theta(dx)=p_\theta(x)dx$는 단위원의 균등 길이 측도에 약수렴한다. 더 구체적으로 0보다 큰 상수 $C,c$를 택하여

$$
\mathbb E_{\mu_\theta}(r-1)^2=O(\theta),\qquad
\|\nu_\theta-\lambda\|_{\rm TV}
\le C\left[\sqrt\theta+\frac{\delta_\theta}{\theta}
+\theta^{-3/2}e^{-c/\theta}\right].
\tag{F3}
$$

여기서 전변동의 규약은 $\sup_A|\nu(A)-\lambda(A)|=\tfrac12\sup_{\|g\|_\infty\le1}|\int g\,d(\nu-\lambda)|$다.

## 10. 각도 방향으로 적분하면 정확한 방사형 방정식이 남는다

정상 Fokker–Planck 방정식의 확률 흐름(probability current)은

$$
J=(-\nabla U+\varepsilon_\theta)p_\theta-\theta\nabla p_\theta,
\qquad \nabla\cdot J=0.
\tag{F4}
$$

이 식과 생성자의 대응은 [Pavliotis, §4.1, 실제 파일 pp.83–84](https://wwwf.imperial.ac.uk/~pavl/PavliotisBook.pdf#page=83)에 설명되어 있다. 다음 방사형 계산은 이 해설의 유도다.

반지름 $r$의 원판에 내부의 발산 적분을 경계의 흐름 적분으로 바꾸는 발산정리(divergence theorem)를 적용하면 원을 통과하는 총 흐름은 0이다. 원점에 별도 원천이나 흡수원이 없고, 각 $\theta$의 $J$가 그곳에서도 매끄럽기 때문이다. 따라서 $r>0$에서 $r\int_0^{2\pi}J_rdu=0$이다.

$w_\theta(r)=\int_0^{2\pi}p_\theta(r,u)du$와 조건부 방사형 잔차

$$
\bar\varepsilon_r(r)=
\frac{\int_0^{2\pi}\varepsilon_r(r,u)p_\theta(r,u)du}{w_\theta(r)},
\qquad |\bar\varepsilon_r(r)|\le\delta_\theta
\tag{F5}
$$

를 정의하면 정확히 $\theta w_\theta'=(-U'+\bar\varepsilon_r)w_\theta$다. 반지름의 밀도는 야코비안 $r$을 포함한 $m_\theta=rw_\theta$이므로

$$
m_\theta(r)=\frac{r}{D_\theta}
\exp\left[-\frac{U(r)}\theta+A_\theta(r)\right],\quad
A_\theta(r)=\frac1\theta\int_1^r\bar\varepsilon_r(s)ds,
\tag{F6}
$$

$$
|A_\theta(r)-A_\theta(s)|\le
\frac{\delta_\theta}{\theta}|r-s|.
\tag{F7}
$$

$D_\theta$는 정규화 적분이다. 미지의 정상밀도가 $\bar\varepsilon_r$에 들어 있으므로 밀도를 명시적으로 푼 것은 아니다. 하지만 (F7)만으로 꼬리와 모멘트를 제어할 수 있다. 잔차의 미분은 계산에 들어가지 않는다.

## 11. 절단 전이 영역의 질량은 온도보다 훨씬 작다

작은 $\theta$에서 $\delta_\theta/\theta\le1$이다. $1\le r\le1+\sqrt\theta$에서는 $U(r)\le C\theta$, $|A_\theta(r)|\le\sqrt\theta$이므로

$$
D_\theta\ge c_0\sqrt\theta.
\tag{F8}
$$

고정 $0<\eta<1$에 대해 $|r-1|\ge\eta$, $0\le r\le R$인 콤팩트 영역에서 $U$의 최솟값은 양수다. 이 부분의 비정규화 질량은 $C_Re^{-c_\eta/\theta}$ 이하이다. 원점에서도 $U(0)=1/8>0$이다.

무한 꼬리는 따로 처리한다. $r\ge4$, $\theta\le1$이면 $U(r)\ge r^4/32$, $r-1\le U(r)/(2\theta)$이므로

$$
\int_R^\infty r e^{-U(r)/\theta+A_\theta(r)}dr
\le\int_R^\infty r e^{-r^4/(64\theta)}dr
\le\frac{16\theta}{R^2}e^{-R^4/(64\theta)}\quad(R\ge4).
\tag{F9}
$$

마지막 부등식은 $t=r^2$로 치환한 뒤 Gaussian 꼬리를 적분하여 얻는다. 분모가 적어도 얼마인지를 보장하는 하계(lower bound)와 결합하면

$$
\mu_\theta\{|r-1|\ge\eta\}
\le C_\eta\theta^{-1/2}e^{-c_\eta/\theta}
=o(\theta^N)\qquad\text{모든 고정 }N>0.
\tag{F10}
$$

단순한 집중보다 강한 추정이다. 절단함수의 미분이 있는 고정 전이 영역에서 필요한 $o(\theta)$가 포함된다.

원 근처에서 $U(r)\ge c(r-1)^2$이므로 $(r-1)^2$을 곱한 분자 적분은 $O(\theta^{3/2})$다. 다른 콤팩트 영역과 무한 꼬리에 다항식을 곱해도 지수 감소가 유지된다. (F8)로 나누면

$$
M_\theta:=\mathbb E(r-1)^2\le C\theta.
\tag{F11}
$$

## 12. 각도 시험함수를 전역 함수로 만든다

$\chi(r)$를 $[3/4,5/4]$에서 1이고 $[1/2,3/2]$ 밖에서 0인 매끄러운 함수로 택한다. $2\pi$-주기 $C^2$ 함수 $\varphi$에 대해 $f(x)=\chi(r)\varphi(u)$는 원점 근처에서 0인 매끄러운 평면 함수다. 또한 0이 아닌 부분이 일정한 반지름의 원 안에 모두 들어가는 콤팩트 지지(compact support)를 갖는다. 정상성 $\int L_\theta f\,d\mu_\theta=0$을 온도로 나누면

$$
\begin{aligned}
0={}&\int\chi\left[r^{-2}\varphi''+
\frac{\varepsilon_u}{\theta r}\varphi'\right]d\mu_\theta\\
&+\int\left[
\frac{-U'+\varepsilon_r}{\theta}\chi'
+\chi''+\frac{\chi'}r\right]\varphi\,d\mu_\theta.
\end{aligned}
\tag{F12}
$$

큰 방사형 drift는 각도 함수와 직교한다.

두 번째 줄은 $\chi'$ 또는 $\chi''$가 있는 영역에만 놓인다. 그곳은 $|r-1|\ge1/4$이므로 (F10)에 의해 절댓값은 $C\theta^{-3/2}e^{-c/\theta}\|\varphi\|_\infty$ 이하이다. 첫 번째 줄의 $\chi/r^2$를 1로 바꾸는 오차는 (F11)과 Cauchy–Schwarz 부등식으로 $C\sqrt\theta\|\varphi''\|_\infty$로 제어된다. 잔차의 접선 성분은 $C\delta_\theta\|\varphi'\|_\infty/\theta$ 이하이다. 따라서

$$
\left|\int\varphi''d\nu_\theta\right|
\le C\sqrt\theta\|\varphi''\|_\infty
+C\frac{\delta_\theta}{\theta}\|\varphi'\|_\infty
+C\theta^{-3/2}e^{-c/\theta}\|\varphi\|_\infty.
\tag{F13}
$$

## 13. 원의 Poisson 방정식으로 전체 분포를 식별한다

연속 함수 $g$, $\|g\|_\infty\le1$에 대해

$$
\varphi''=g-\int g\,d\lambda
\tag{F14}
$$

를 원에서 푼다. 우변의 평균이 0이므로 한 번 적분한 뒤 도함수의 평균을 빼고 다시 적분하면 주기적인 $\varphi$를 얻는다. 상수를 정해 평균도 0으로 두면

$$
\|\varphi''\|_\infty\le2,\quad
\|\varphi'\|_\infty\le4\pi,\quad
\|\varphi\|_\infty\le8\pi^2.
\tag{F15}
$$

이 상계는 $g$와 무관하다. (F13)에 넣고 모든 $g$에 대해 상한을 취하면 (F3)을 얻는다. 원 위 유한 Borel 측도의 전변동은 연속 시험함수로도 같은 상한을 얻는다.

주변 공간의 약수렴에는 별도 한 단계가 필요하다. 유계 연속 $h(x)$의 원 근처 균등 연속성과 (F10)을 쓰면 $\int[h(x)-h(e_r)]d\mu_\theta\to0$이다. 각도 분포의 수렴이 나머지 적분을 균등 원 측도의 적분으로 보낸다. (F3)의 전변동은 각도 사영에 관한 결과다. 주변 공간에서 매끄러운 측도와 원 위 측도의 전변동 거리는 1이다.

## 14. 관측 guidance를 남겼을 때

drift를 $-\nabla U-\theta\nabla v+\varepsilon_\theta$로 바꾸고 $v\in C^2(\mathbb R^2)$, 전역 $\|\nabla v\|_\infty<\infty$를 추가로 가정하자. 방사형 총 잔차는 $\theta\|\nabla v\|_\infty+\delta_\theta=O(\theta)$다. 따라서 (F8)–(F11)은 상수만 바뀌어 유지된다. (F7)의 계수는 여기서 0으로 갈 필요 없이 유계이면 충분하다.

각도 극한 방정식은 $v_0(u)=v(\cos u,\sin u)$에 대해

$$
\int[\varphi''-v_0'\varphi']d\nu=0.
\tag{F16}
$$

guided 정상측도를 $\mu_\theta^v$라 쓰면 최종 약극한은 정규화까지 포함해

$$
\mu_\theta^v\Rightarrow
\frac{e^{-v(\cos u,\sin u)}}
{\int_0^{2\pi}e^{-v(\cos w,\sin w)}\,dw}\,du.
$$

다. $du$는 단위원의 호 길이와 같고, 관측 $y$는 $v$ 안에 고정되어 있다. 따라서 이 식이 부피 균등 사전에 likelihood를 곱한 사후분포를 직접 보여 준다.

유일한 확률 해는 $\rho_0(u)du=Z^{-1}e^{-v_0(u)}du$다. 밀도 존재를 먼저 가정하지 않는 식별은 [반사 경계를 둔 고리 영역 해설](/reviews/score-geometry-weak-proof/)의 가중 Poisson 방정식과 같다. $\int g_0\rho_0=0$이면 $(\rho_0\varphi')'=\rho_0g_0$를 주기적으로 풀어 (F16)에서 $\int g_0d\nu=0$을 얻는다. 결과는 단위원의 부피 균등 사전분포(prior)에 관측 우도(likelihood)를 곱한 사후분포(posterior)다.

## 15. 미분이 커지는 비보존적(non-conservative) 정확 예제

$w=e^{-U/\theta}$라 하자. 원점에서 떨어진 고정 고리 영역 밖에서는 0이 되고 $h(1)=1$, $h'(1)=0$인 매끄러운 방사형 함수 $h$를 택한다. $k_\theta=\lceil\theta^{-2}\rceil$, $\eta_\theta=\theta^{3/2}$로 두고 흐름 함수(stream function)를

$$
\psi_\theta=\frac{\eta_\theta}{k_\theta}h(r)w(r)\sin(k_\theta u),
\quad J_r=\frac1r\partial_u\psi_\theta,\quad J_u=-\partial_r\psi_\theta
\tag{F17}
$$

로 만든다. $\nabla\cdot J=0$이고 $J$는 원점 근처와 충분히 먼 곳에서 0이다. $\varepsilon=J/w$이면

$$
\varepsilon_r=\frac{\eta_\theta h}r\cos(k_\theta u),\qquad
\varepsilon_u=\frac{\eta_\theta}{k_\theta}
\left(\frac{U'h}{\theta}-h'\right)\sin(k_\theta u).
\tag{F18}
$$

고정 고리 영역에서 $\|\varepsilon\|_\infty=O(\theta^{3/2})=o(\theta)$이지만 $r=1$에서 $\partial_u\varepsilon_r$의 진폭은 $\eta_\theta k_\theta\asymp\theta^{-1/2}$로 발산한다. curl도 일반적으로 0이 아니다. 그런데 $p=w/Z$의 전체 확률 흐름은 정확히 $J/Z$이고 발산은 0이다. 따라서 $p$는 정확한 정상밀도이고 각도는 모든 온도에서 균등하다. 이는 작은 잔차의 큰 미분과 균등 극한이 양립하며, 전역 모형의 가정이 실제로 충족될 수 있음을 보여준다.


![전역 방사형 모형의 절단 영역 질량과 법선 분산](/assets/score-geometry/full-space-toy.svg)

별도 결정론적 적분 예제. $U=(r^2-1)^2/8$, 잔차 0인 정확한 정상밀도를 사용한다. 왼쪽은 $|r-1|\ge1/4$인 질량을 온도로 나눈 값, 오른쪽은 법선 제곱편차를 온도로 나눈 값이다. 적분 범위는 $0\le r\le4$, 균일 격자 400,001점이다. 온도가 0.00125일 때 두 값은 각각 약 $2.45\times10^{-7}$과 1.00475다. 이 값은 학습 실험이 아니라 보충 모형의 계산이다.

## 16. 일반 다양체로 이어지는 구조


원에서는 각도에 대한 Laplacian을 직접 계산했다. 일반 다양체의 거리 제곱 모델에서도, 다양체를 얇은 관처럼 둘러싼 튜브형 이웃(tubular neighborhood) 안에서는 유용한 구조가 있다. 최근접 사영 $P$로 확장한 $F=\varphi\circ P$는 같은 최근접점을 공유하는 각 법선 방향 집합(normal fiber)에서 상수이므로

$$
\nabla d_{\mathcal M}\cdot\nabla F=0,\qquad
\left.\Delta F\right|_{\mathcal M}=\Delta_{\mathcal M}\varphi.
$$

두 번째 등식은 다음처럼 확인한다. 법선 직선을 따라 $F$가 상수이므로 법선 Hessian의 대각 성분은 0이다. 다양체 안에서 국소적으로 거리를 최소화하는 경로인 측지선(geodesic)을 따라 2차 미분하면 주변 Hessian과 곡률 가속도 항이 나오지만, $\nabla F$는 접선이고 그 가속도는 법선이어서 내적이 0이다. 접선 방향들을 합하면 Laplace–Beltrami 연산자가 남는다.

따라서 거리 제곱과 정상 반사가 있는 일반 튜브형 이웃에서도 같은 약한 증명의 확장 경로는 명확하다. 다만 이 문서에서 정량 TV 상계까지 완결한 대상은 단위원이다. 일반 다양체에서는 필요한 사영 미분의 균일 상수와 Poisson 해의 제어를 추적해야 한다. 단순히 최소점만 다양체인 일반 $f_0$에는 이 정확한 직교성이 없을 수 있고 법선 Hessian에 따른 비균등 가중치도 남을 수 있다.


비연결 성분의 상대 질량과 일반 다양체의 전역 집중 조건은 별도로 결정해야 한다.

## 리뷰 해석과 남은 질문

<strong>리뷰 해석.</strong> 반사 경계를 둔 고리 영역은 경계 흐름을 정확하게 제어하고, 전체 평면 모형은 방사형 정상 흐름으로 절단 오차를 제어한다. 두 증명 모두 최종적으로 원의 Poisson 방정식을 풀어 모든 연속 시험함수의 평균을 정한다. 정상밀도 진폭의 극한에서의 모양(asymptotic form)이나 잔차 미분의 균일 수렴이 필요하지 않다는 점이 장점이다.

전체 평면 결과에는 전역적으로 작은 잔차와 멀리 나가지 못하게 되돌리는 힘(confinement)이 추가된다. 튜브형 이웃 내부 오차만으로 이 조건이 따라오지는 않는다. 일반 다양체의 전역 집중, 비연결 성분의 상대 질량, 유한 시간 혼합은 남은 질문이다. 이 글의 명제 S.1과 F.1은 리뷰 보충 결과이며 원문 정리 번호와 구별한다.

## 출처와 읽은 범위

- 원문: Xiang Li, Zebang Shen, Ya-Ping Hsieh, Niao He, *When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis*. 최초 공개 2025-09-29, 검토 버전 v2(2026-03-16), 검토일 2026-10-29. [공식 원문](https://arxiv.org/abs/2509.24912v2).
- 연결 범위: 원문 부록 B.4–B.6을 출발점으로 하며, 명제 S.1과 F.1 및 그 예제는 리뷰 보충 유도다.
- 추가 원전: [Kang–Ramanan, §2–3·Theorem 1, 파일 pp.1–7](https://arxiv.org/pdf/1204.4969v1#page=7), [Pavliotis, 저자 공개 원고 §4.1·4.6, 파일 pp.83–84·109–110](https://wwwf.imperial.ac.uk/~pavl/PavliotisBook.pdf#page=83). 전자는 반사 정상성의 부등식 부호를, 후자는 확률 흐름과 가역성의 배경을 제공한다.
- 그림: 시리즈 썸네일은 [원문 Fig.1](https://arxiv.org/html/2509.24912v2#S1.F1)을 재사용한다. 벡터 도형은 유지하고 내부 래스터만 원래 픽셀 크기의 WebP(품질 95)로 변환한 자산이다. 원문에는 arXiv 영구 비독점 공개 라이선스가 표시되어 있다. 본문의 별도 수치 예제 그림은 명시한 모형의 결정론적 적분 결과다.
- 확인하지 않은 것: 신경망 학습, Stable Diffusion 이미지 생성의 독립 재현과 정리의 형식 검증.
