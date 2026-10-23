---
title: "[3/3] LeJEPA 기술 심층 리뷰: 증명 조건, SIGReg와 수식 검토"
description: "원문 §3→§4→§5의 논증과 부록 A·B를 따라 주요 정리, 보조정리, 중간 유도와 미확인 간극을 자세히 읽는다."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-23T08:56:15+00:00"
topics: ["self-supervised learning", "representation learning", "joint embedding predictive architectures", "distribution matching", "computer vision"]
visibility: "public"
lang: "ko"
translationKey: "lejepa-technical"
paperTitle: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
paperPublishedDate: "2025-11-11"
authors: ["Randall Balestriero", "Yann LeCun"]
year: "2025"
paperUrl: "https://arxiv.org/abs/2511.08544v3"
thumbnail: "/assets/reviews/lejepa/lejepa-core.svg"
thumbnailAlt: "LeJEPA의 두 제약 설명도: 한 이미지의 뷰들은 전역 뷰 평균에 정렬하고, 각 뷰에서 여러 이미지의 표현은 임의 방향으로 투영해 표준 Gaussian과 맞춘다."
---

LeJEPA의 이론은 선형 추정의 등방성 공분산(isotropic covariance), 국소 비선형 추정의 피셔 정보(Fisher information), 유한 투영으로 계산하는 등방성 가우시안 투영 정규화(Sketched Isotropic Gaussian Regularization, SIGReg)를 잇는다. 고정된 공분산에서 Gaussian이 피셔 정보를 최소화한다는 결과와, 유한 표본 손실을 계산하는 근사 구현이 각기 다른 역할을 한다. [원문 §3–5·Appendix A–B](https://arxiv.org/pdf/2511.08544v3#page=5) **리뷰 해석.** 분포 선택의 근거는 유용하지만 등호 조건·정규화·유한 근사를 구분해야 구현의 보장을 정확히 읽을 수 있다.

**[3/3] 정리·증명 심층 편**이다. [입문](/reviews/lejepa-overview/) → [전체 해설](/reviews/lejepa-complete/) → 이 글 순서가 권장된다. §§1–2의 문제·배경과 §6 전체 실험은 전체 해설에서 읽는다. 아래에서는 주요 정리의 가정과 계산에 집중하며, 부록 증명은 그 정리가 나오는 본문 위치로 옮겼다.

**표기 대응.** $\mathbf Z\in\mathbb R^{N\times K}$는 표현 행렬, $\mathbf z_{n,v}$는 이미지·뷰 표현이다. 선형 증명의 $\mathbf X$, $p$, $\lambda_{\rm wd}$는 본문의 $\mathbf Z$, $K$, ridge 계수에 대응한다. 여기서 일반 밀도 $p$는 선형 증명의 차원 기호와 다른 역할이다. Appendix의 $d,n$은 해당 계산의 차원·표본 수다. $M=|\mathcal A|$, 주파수 수 $T$, $\widehat D=EP/N$는 설명을 위한 약칭이다.

<span id="3-latent-euclidean-목표-분포의-이론" aria-hidden="true"></span>

## 3. 잠재 유클리드 공간: 표현은 등방성 가우시안이어야 한다 (Latent Euclidean — Embeddings Should Be Isotropic Gaussian)

### 3.1 Linear Probing — Lemmas 1–2와 Appendix B.1–B.2

고정 설계(fixed design)에서 $\mathbf y=\mathbf Z\boldsymbol\beta_{\rm true}+\boldsymbol\varepsilon$이고 $\mathbb E[\boldsymbol\varepsilon\mid\mathbf Z]=0$, $\operatorname{Cov}(\boldsymbol\varepsilon\mid\mathbf Z)=\sigma^2I$라 하자. $\mathbf Z$는 모든 열이 선형 독립(full column rank)이다. 같은 column span과 총 에너지의 두 표현을 비교한다. 원문 선형 결과는 계수 추정에 관한 것이다.

**Lemma 1의 진술.** 비등방성(anisotropy)에서는 양의 리지 정규화(ridge regularization) 아래 등방성(isotropy)보다 편향(bias)이 큰 과제가 존재한다. 먼저 ridge 목적을 미분하고 정규방정식(normal equation)을 풀면,

$$
\hat{\boldsymbol\beta}
=(\mathbf Z^\top\mathbf Z+\lambda_{\rm wd}I)^{-1}\mathbf Z^\top\mathbf y.
$$

조건부 기대값에서 잡음이 사라진다. $\mathbf G=\mathbf Z^\top\mathbf Z$를 리뷰 약칭으로 두면,

$$
\begin{aligned}
\mathbb E[\hat{\boldsymbol\beta}\mid\mathbf Z]-\boldsymbol\beta_{\rm true}
&=\left[(\mathbf G+\lambda_{\rm wd}I)^{-1}\mathbf G-I\right]\boldsymbol\beta_{\rm true}\\
&=-\lambda_{\rm wd}(\mathbf G+\lambda_{\rm wd}I)^{-1}\boldsymbol\beta_{\rm true}.
\end{aligned}
$$

중간 전환은 $\mathbf G=(\mathbf G+\lambda_{\rm wd}I)-\lambda_{\rm wd}I$를 대입한 것이다. $\mathbf G=\mathbf Q\boldsymbol\Lambda\mathbf Q^\top$에서 편향(bias)의 각 고유방향(eigendirection) 성분은 $\lambda_{\rm wd}/(\lambda_k+\lambda_{\rm wd})$만큼 줄어든다. $\bar\lambda=K^{-1}\sum_k\lambda_k$라 두고 가장 작은 고윳값(eigenvalue)의 방향에 실제 과제를 잡으면,

$$
\frac{\lambda_{\rm wd}}{\lambda_{\min}+\lambda_{\rm wd}}
>\frac{\lambda_{\rm wd}}{\bar\lambda+\lambda_{\rm wd}}.
$$

이것이 존재 양화의 이유다. 가장 큰 고윳값 방향에서는 반대 효과도 가능하므로 모든 과제로 강화하면 안 된다. Appendix B.1에서 작은 고윳값의 index 표기가 흔들리지만 proof가 사용하는 것은 위의 최소 방향이다. [Lemma 1; Appendix B.1, p. 26](https://arxiv.org/pdf/2511.08544v3#page=26)

**Lemma 2의 진술.** ridge를 0으로 두면 계수 분산(variance)의 대각합(trace)은 등방성에서 최소다. $\hat{\boldsymbol\beta}-\boldsymbol\beta_{\rm true}=\mathbf G^{-1}\mathbf Z^\top\boldsymbol\varepsilon$이므로,

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

마지막은 양수에서 $1/x$의 엄격한 볼록성(strict convexity)과 옌센 부등식(Jensen’s inequality)이다. 고윳값이 모두 같아야 등호다. 고정된 test 표현 $\mathbf q$의 예측 분산은 $\sigma^2\mathbf q^\top\mathbf G^{-1}\mathbf q$이므로 계수 trace 최소화와 모든 질의(query) 위험(risk) 최소화는 다른 진술이다. 선형 분석이 선택한 것은 공분산 모양이지 Gaussian 법칙 전체가 아니다. [Lemma 2; Appendix B.2, p. 27](https://arxiv.org/pdf/2511.08544v3#page=27)

### 3.2 Nonlinear Probing — Theorem 1과 Appendix A, B.3–B.7

**가정부터 읽기.** $p_z\in C^3$이고 필요한 도함수가 유계이며 Fisher information·공분산이 유한하다. 목표 함수 $\eta\in C^2$이며 query도 학습 밀도를 따른다. 반경·bandwidth가 작고 국소 평균을 추정할 표본이 충분해야 한다. 커널(kernel)은 정규화·대칭·등방적 2차 모멘트(moment)·유한 거칠기(roughness)를 갖는다. 부분적분에는 경계항이 사라지는 꼬리 조건이 필요하다. [Appendix A, pp. 24–26](https://arxiv.org/pdf/2511.08544v3#page=24)

**Lemma 4 — kNN pointwise bias: 왜 밀도 스코어(density score)가 등장하는가.** Appendix B.3는 반경 내 평균을 밀도 적분의 비율로 근사한다. 비어 있는 이웃 문제와 유한 표본 비율의 오차를 생략한 국소 surrogate이며 포아송 점과정(Poisson point process, PPP) 조건을 사용한다. $\mathbf u$를 중심에서의 변위로 쓰면,

$$
\mathbb E[\widehat{\mathbf y}(\mathbf q)]\simeq
\frac{\int_{\|\mathbf u\|\le r_0}\eta(\mathbf q+\mathbf u)p(\mathbf q+\mathbf u)\,d\mathbf u}
{\int_{\|\mathbf u\|\le r_0}p(\mathbf q+\mathbf u)\,d\mathbf u}.
$$

**리뷰 보충 유도.** 단위 공 부피를 $v_K$라 두면 대칭성으로 1차 적분은 0이고 2차 적분은 다음과 같다. 원문 중간 줄의 부피 약칭을 이 식으로 풀었다.

$$
\int_{B(0,r_0)}\mathbf u\,d\mathbf u=0,\qquad
\int_{B(0,r_0)}\mathbf u\mathbf u^\top d\mathbf u
=\frac{v_Kr_0^{K+2}}{K+2}I.
$$

분모의 2차 항은 $\Delta p$, 분자의 2차 항은 product rule로 $\Delta(\eta p)=p\Delta\eta+\eta\Delta p+2\nabla\eta^\top\nabla p$다. 비율에서 $\eta\Delta p$가 상쇄되어,

$$
\mathrm{Bias}(\mathbf q)=\frac{r_0^2}{K+2}
\left(\nabla\eta(\mathbf q)^\top\nabla\log p(\mathbf q)
+\frac12\Delta\eta(\mathbf q)\right)+o(r_0^2).
$$

위 표기는 본문의 $K$와 query 위치를 유지했다. $C^2$만으로 3차 Taylor 잔차가 유계라고 단정할 수는 없으므로, 원문의 일부 $O(\|\mathbf u\|^3)$ 중간 표기보다 약한 $o(r_0^2)$ 진술을 사용한다. [Lemma 4; Appendix B.3, p. 27](https://arxiv.org/pdf/2511.08544v3#page=27)

**Theorem 7의 연결.** $A=\nabla\eta^\top\nabla\log p$, $C=\Delta\eta/2$는 Appendix B.4의 약칭이다. 이를 제곱하면 $\mathbb E[A^2]+2\mathbb E[AC]+\mathbb E[C^2]$가 모두 나온다. 과제 사전 가정(task prior)이 각 위치에서 평균 0·등방적 2차 moment $\tau_g^2I$를 갖는다면,

$$
\mathbb E_{\mathbf z,\eta}[A(\mathbf z)^2]
=\mathbb E_{\mathbf z}\left[
\nabla\log p(\mathbf z)^\top(\tau_g^2I)\nabla\log p(\mathbf z)\right]
=\tau_g^2J(p).
$$

교차항은 gradient와 Laplacian의 추가 비상관 조건이 있어야 0이 된다. 곡률항도 task의 위치별 2차 moment가 일정해야 밀도에 독립이라고 할 수 있다. 원문의 $\tau_g^2J(p)$ 항과 $O(r_0^4)$ 잔차는 같은 차수다. 잔차의 $p$ 의존성을 통제하지 않으면 이 식만으로 정확한 integrated bias 전체의 유일 최소를 얻지는 못한다. [Theorem 7; Appendix B.4, pp. 28–29](https://arxiv.org/pdf/2511.08544v3#page=28)

**Lemma 6와 Theorem 9 — Fisher information의 하한.** 원문은 location family의 matrix Cramér–Rao bound를 사용한다. 필요한 정규성 아래, 평균 0·공분산 $\Sigma\succ0$인 밀도 스코어(density score) $\mathbf s(\mathbf z)=\nabla\log p(\mathbf z)$에 대해 $J(p)\ge\operatorname{tr}(\Sigma^{-1})$이고 등호는 Gaussian이다.

**리뷰 보충 유도: Cramér–Rao의 이 경우를 직접 확인.** 부분적분으로 $\mathbb E[\mathbf s(\mathbf z)\mathbf z^\top]=-I$다. 경계항이 0이어야 한다. 따라서,

$$
\begin{aligned}
0&\preceq\mathbb E\left[
(\mathbf s+\Sigma^{-1}\mathbf z)(\mathbf s+\Sigma^{-1}\mathbf z)^\top\right]\\
&=\mathbb E[\mathbf s\mathbf s^\top]-\Sigma^{-1}.
\end{aligned}
$$

trace를 취하면 하한이다. 등호에서는 잔차가 거의 확실하게 0이라 $\nabla\log p=-\Sigma^{-1}\mathbf z$다. 정규성 아래 적분하면 $p\propto\exp(-\mathbf z^\top\Sigma^{-1}\mathbf z/2)$가 된다. 이제 총 분산 $\operatorname{tr}\Sigma=c$를 고정하면,

$$
J(p)\ge\sum_{k=1}^K\frac1{\lambda_k}
\ge\frac{K^2}{c},\qquad
\Sigma=\frac cK I\ \text{at equality}.
$$

이는 Fisher functional에 대한 유일 최적성이다. Appendix B.4는 determinant·Frobenius·spectral-radius 제약도 다룬다. 같은 방향의 논리이므로 trace를 자세히 전개하고 나머지 제약의 원문 위치를 남긴다. 제목이 ‘Recovery of VCReg’로 붙어 있지만 Lemma 6·Theorem 9에서 사용하는 결과는 이 Fisher 하한이다. [Lemma 6, Theorem 9; Appendix B.4, pp. 30–32](https://arxiv.org/pdf/2511.08544v3#page=30)

**Lemma 5 — Kernel bias와 variance.** 분자·분모 기대값을 kernel 변수로 치환해 2차 전개한다. 대칭성으로 1차 항이 없어지고 $\int\mathbf u\mathbf u^\top K(\mathbf u)d\mathbf u=\mu_2(K)I$를 사용한다. 비율의 중간 계산은,

$$
\frac{a_0+h^2a_2+o(h^2)}{b_0+h^2b_2+o(h^2)}
=\frac{a_0}{b_0}+h^2\frac{a_2b_0-a_0b_2}{b_0^2}+o(h^2).
$$

$a_0=\eta p$, $b_0=p>0$에 대입하면,

$$
\begin{aligned}
\mathrm{Bias}(\mathbf q)&=\frac{h^2\mu_2(K)}2
\left(\Delta\eta(\mathbf q)+2\nabla\eta(\mathbf q)^\top\nabla\log p(\mathbf q)\right)+o(h^2),\\
\mathrm{Var}(\mathbf q)&=\frac{R(K)}{nh^d}\frac{v(\mathbf q)}{p(\mathbf q)}
+o((nh^d)^{-1}).
\end{aligned}
$$

여기서 $d=K$, $n=N$, $R(K)=\int K^2$, $v(\mathbf q)=\operatorname{Var}(Y\mid\mathbf q)$다. 분자 기대값/분모 기대값은 유한 표본에서 비율의 기대값과 같지 않다. 표본 분모의 변동(fluctuation)을 통제하는 국소 표본·점근 조건이 필요하다. 분산을 유도할 때도 원문 중간 줄의 $\operatorname{Var}(K_hY)=\mathbb E[K_h^2v]$는 일반적 정확 등식이 아니다. **리뷰 보충 설명:** 델타 방법(delta method)에서 $Y-\eta(\mathbf q)$로 중심화한 비율 잔차가 leading conditional-noise 항을 만든다고 읽어야 한다. 원문의 유한 표본 잔차 전체를 따로 증명한 것은 아니다. [Lemma 5; Appendix B.5, pp. 32–33](https://arxiv.org/pdf/2511.08544v3#page=32)

**Theorem 8 — 무엇을 최소화했는가.** target의 $\|\nabla\eta\|\le L$, $|\Delta\eta|\le B$에서 $(a+b)^2\le2a^2+2b^2$와 Cauchy–Schwarz를 차례로 쓰면,

$$
\begin{aligned}
\mathrm{ISB}_{\rm kernel}
&\le\left(\frac{h^2\mu_2(K)}2\right)^2
\left(2\int(\Delta\eta)^2p+8\int(\nabla\eta^\top\nabla\log p)^2p\right)+o(h^4)\\
&\le\left(\frac{h^2\mu_2(K)}2\right)^2(2B^2+8L^2J(p))+o(h^4).
\end{aligned}
$$

Gaussian이 최소화하는 것은 이 $J(p)$ 기반 상한이다. 상한(bound)의 최소화가 정확한 편향(bias)의 최소화나 정확한 최소최대 등식(sharp minimax equality)을 자동으로 주지는 않는다. 적분 분산에서는 leading 항의 $p$가 query 가중 $p$와 상쇄되지만 $\int v(\mathbf z)d\mathbf z$의 유한성·지원 영역·균일 잔차 통제가 필요하다. Appendix A의 Theorem 8 박스에 제곱이 누락된 표기가 있지만 본문 Theorem 1과 B.7의 유도는 squared bias다. [Theorem 8; Appendix B.7, p. 34](https://arxiv.org/pdf/2511.08544v3#page=34)

### 3.3 Geometric and Practical Insights

Figure 3의 반복 logistic 경계는 등방성의 경험적 예시다. Figure 17은 이진·선형·비선형 타깃의 재표본화 결과를 함께 표시하며, 캡션은 비등방성이 더 낮은 분산을 준다고 서술한다. Lemma 2는 같은 열공간·총 에너지, 정규화 없는 OLS에서 등방성의 더 작은 총 계수 분산을 말한다. [원문 §3.1, Lemma 2](https://arxiv.org/pdf/2511.08544v3#page=5), [Figure 17](https://arxiv.org/pdf/2511.08544v3#page=48)

**리뷰 해석.** 두 서술의 분산 방향은 반대지만, 타깃 유형과 재표본화 조건이 모두 같다고 확정할 근거는 부족하다. 캡션의 방향을 뒤집는 정오표로 단정하지 않고, 선형 타깃의 설계·잡음·색 대응을 확인할 문제로 남긴다. Lemma 2의 대수적 증명은 이 그림의 캡션과 독립적으로 평가할 수 있다.

<span id="4-sigreg-식별과-안정성-유한-방향의-조건" aria-hidden="true"></span>

## 4. SIGReg: 고차원의 신뢰할 수 있는 등방성 가우시안 정규화 (SIGReg — Reliable Isotropic Gaussian Regularization in High-Dimension)

### 4.1 Hypothesis Testing as a Judge — Lemma 3, Theorem 2

**Lemma 3와 Appendix B.8의 Theorem 10(Cramér–Wold).** 모든 unit 방향의 투영 법칙이 같다고 하자. 임의의 $\mathbf t\ne0$는 $\mathbf t=r\mathbf a$로 쓸 수 있다. 따라서,

$$
\varphi_X(\mathbf t)=\mathbb E[e^{ir\mathbf a^\top X}]
=\mathbb E[e^{ir\mathbf a^\top Y}]=\varphi_Y(\mathbf t).
$$

$\mathbf t=0$에서는 둘 다 1이다. 모든 벡터 주파수에서 CF가 같고 특성함수(characteristic function, CF)의 유일성 정리를 적용하면 $X\overset d=Y$다. 유한 방향 일치로 양화를 줄일 수는 없다. 약한 수렴(weak convergence) 부분은 projection 수렴과 Lévy 연속성 정리를 연결한다. [Lemma 3; Appendix B.8, p. 35](https://arxiv.org/pdf/2511.08544v3#page=35)

**Theorem 2의 proof path.** 결합 검정은 방향별 통계의 max와 전체 귀무가설(null hypothesis)의 임계값(threshold)을 사용한다. 전체 위양성(false positive) 제어는 개별 검정의 유의수준(significance level)을 그대로 재사용해서는 안 된다. 예를 들어 Bonferroni를 적용하면 각 방향의 수준을 $\alpha/|\mathcal A|$로 두어 union bound로 전체 $\alpha$를 제어할 수 있다. 이 예시는 리뷰 보충 설명이다.

검정력(statistical power) 쪽은 분리 방향 $\mathbf a^\star$ 주변 $U$에서 검정이 강하고, 방향 집합 mesh가 0으로 가서 결국 $U$ 안에 점을 갖는다는 경로다. 이동하는 방향 $\mathbf a_n$을 쓰려면 $U$에서 충분히 균일한 검정력(uniform power) 조건이 필요하다. B.9의 $\inf_a\lim_n$ 표기만으로 $\lim_n\inf_a$를 바로 얻을 수는 없어 이 교환의 엄밀화는 미확인이다. SIGReg의 학습용 **평균**을 이 max 검정의 정리와 동일시하지 않는다. [Theorem 2; Appendix B.9, p. 36](https://arxiv.org/pdf/2511.08544v3#page=36)

### 4.2 SIGReg: Sketching the Epps–Pulley Test

#### 4.2.1 Moments — Theorem 3와 Appendix B.11

원문 Theorem 3에서 유한 moment 차수 $K$는 여기의 embedding 차원과 다른 역할이다. 서로 다른 실수 $x_0,\ldots,x_{K+1}$에 확률 질량을 두자. $A_{rj}=x_j^r$ ($r=0,\ldots,K$)는 $(K+1)\times(K+2)$ 행렬이므로 kernel에 0이 아닌 $\mathbf v$가 있다. $r=0$ 행에서 $\sum_jv_j=0$다.

양의 확률 벡터 $\mathbf p$와 충분히 작은 $\varepsilon$를 선택하면 $\mathbf p^\pm=\mathbf p\pm\varepsilon\mathbf v$는 서로 다른 유효 확률이지만 $A\mathbf p^+=A\mathbf p^-$다. 따라서 0차부터 $K$차까지 moment가 같아도 분포가 다를 수 있다. 높은 moment는 sample-gradient에 $x_i^{k-1}$ 같은 다항식이 생겨 전역 유계성이 깨질 수 있다. [Theorem 3; Appendix B.11–B.12, p. 37](https://arxiv.org/pdf/2511.08544v3#page=37)

![Figure 5: Gaussian 주변 분포로도 X자 결합 분포가 생길 수 있다.](/assets/reviews/lejepa/paper-figure-5.webp)

**Figure 5 읽기.** 주변 histogram과 identity covariance만으로는 X자 결합 구조가 드러나지 않는다. 사선 투영과 방향별 통계가 이를 검출한다. 곡선은 서로 다른 검정의 반응이며 같은 척도의 정확도는 아니다. [확대](/assets/reviews/lejepa/paper-figure-5.webp) · [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

#### 4.2.2 CDF — 계산 요구와 대안

원문의 선택 기준은 식별성뿐 아니라 gradient 안정성과 DDP 처리다. 누적분포함수(cumulative distribution function, CDF)·순서통계(order statistics)의 정렬 및 분산 표본 수집 부담을 비교한다. Appendix E의 Shapiro–Wilk, F의 multivariate 통계는 대안 배경으로 축약한다. 이들 모든 검정의 독립 정확성·구현은 검증하지 않았다. [§4.2.2; Appendices E–F](https://arxiv.org/pdf/2511.08544v3#page=44)

#### 4.2.3 CF — Theorem 4의 유도와 정규화

Appendix B.12의 Gaussian window 표기를 그대로 사용한다. $x_i=\mathbf a^\top\mathbf z_i$는 투영 스칼라이며,

$$
\widehat\varphi_N(t)=\frac1N\sum_{i=1}^Ne^{itx_i},\quad
\varphi_G(t)=e^{-t^2/2},\quad w_s(t)=e^{-s^2t^2},
\qquad
\widehat D=\int w_s(t)|\widehat\varphi_N(t)-\varphi_G(t)|^2dt.
$$

원문 본문 Epps–Pulley 식의 $EP=N\widehat D$와 다르게 $\widehat D$에는 $N$배가 없다. $|\widehat\varphi_N|,|\varphi_G|\le1$이고 $\partial_{x_i}\widehat\varphi_N=it e^{itx_i}/N$이므로,

$$
\begin{aligned}
|\partial_{x_i}\widehat D|
&\le\frac4N\int w_s(t)|t|dt=\frac4{Ns^2},\\
|\partial_{x_i}^2\widehat D|
&\le\left(\frac2{N^2}+\frac4N\right)\int w_s(t)t^2dt
\le\frac{6\sqrt\pi}{2Ns^3}.
\end{aligned}
$$

**리뷰 보충 유도.** 첫 줄은 오차의 크기가 최대 2임을 사용한다. 두 번째 미분에는 $2|\partial\widehat\varphi_N|^2$와 $2\operatorname{Re}[(\widehat\varphi_N-\varphi_G)\overline{\partial^2\widehat\varphi_N}]$가 나온다. Gaussian 가중에서 $\int|t|w_s=1/s^2$, $\int t^2w_s=\sqrt\pi/(2s^3)$이므로 입력 크기와 무관한 bound다. 지배수렴(dominated convergence)을 위해 이 적분들이 유한해야 한다.

본문 Theorem 4의 $\sigma$는 $s=1/\sigma$로 연결할 수 있지만, $EP$라고 쓴 $O(1/N)$ bound의 유도는 $\widehat D$에 해당한다. 실제 $EP$에는 $N$을 곱해 $|\partial EP|\le4/s^2$다. **boundedness는 유지되지만 $1/N$ 감소의 대상이 다르다.** parameter gradient에는 $\sum_i|\partial_{x_i}D|\|\mathbf a^\top\partial_{\boldsymbol\theta}f_{\boldsymbol\theta}\|$ 형태의 Jacobian이 들어간다. [Theorem 4; Appendix B.12, pp. 37–38](https://arxiv.org/pdf/2511.08544v3#page=37)

Algorithm 1은 17개 주파수 $[-5,5]$, target/window $e^{-t^2/2}$, CF 평균 all-reduce, trapezoidal 적분과 전체 $N$배를 사용한다. equal-sized GPU batch라면 로컬 CF 평균의 평균이 전체 CF 평균이다. sample 쌍을 만드는 대신 matrix projection과 CF 평균을 계산해 dense 비용은 $\mathcal O(NKM+NMT)$다. 분산 autograd 동작과 전체 모델 메모리는 여기서 실행 검증하지 않았다. [Algorithm 1; Table 6](https://arxiv.org/pdf/2511.08544v3#page=10)

**리뷰 보충 예제: 유한 주파수의 식별 가능성(identifiability) 한계.** 17개 grid의 간격은 $\Delta t=0.625$다. $X\sim\mathcal N(0,1)$을 $c=2\pi/\Delta t$만큼 평행이동한 $Y=X+c$는 각 grid에서 $e^{itc}=1$이므로 동일한 CF 값을 낸다. grid 밖에서는 다르다. 이것은 1D 고정 격자(grid)의 예이며 고차원 재표집 학습 전체의 반례나 모든 주파수 Cramér–Wold 정리의 반례는 아니다. 정확 적분의 식별성과 수치 근사를 구분하기 위한 예제다.

### 4.3 How SIGReg Beats the Curse of Dimensionality — Theorem 5

원문의 rate를 보존하면,

$$
\mathbb E_{\mathbf a}\int_{\mathbb R}
|\varphi_{\mathbf a}(t)-\varphi_{\mathcal N}(t)|^2dt
\le C(K,\alpha)M^{-2\alpha/(K-1)}
\int_0^\infty\|\varphi_{\cdot}(r)-\varphi_{\mathcal N}(r)\|_{H^\alpha(\mathcal S^{K-1})}^2dr.
$$

원문은 $p_{\boldsymbol\theta}\in H^\alpha(\mathbb R^K)$, 선택된 방향에서 전체 투영 법칙이 일치함을 가정한다. **Appendix B.10의 논리:** 구면 조화함수(spherical harmonics)에서 차수 $L$까지의 근사 오차가 대략 $L^{-\alpha}$이고, 준균일 방향 $M$개는 $L\sim M^{1/(K-1)}$와 연결된다. 이를 제곱해 $M^{-2\alpha/(K-1)}$를 얻고 주파수에 대해 적분한다. Marcinkiewicz–Zygmund 조건은 유한 점의 값으로 조화함수 공간을 안정적으로 복원하는 역할이다.

**미확인 조건.** Euclidean 밀도의 소볼레프 정규성(Sobolev regularity)에서 필요한 angular CF norm·주파수 적분이 어떻게 유한해지는지, 점 배치와 보간 연산자의 안정성 조건, 인용 원전의 lemma·정리 적용을 독립적으로 모두 검증하지 않았다. 임의 방향 배치·단순한 random 표본 수에 이 rate를 무조건 적용하지 않는다. $C(K,\alpha)$와 norm의 차원 의존성도 있어 $O(K)$ 방향만으로 모든 분포의 오차가 작다는 일반 보장은 아니다. [Theorem 5; Appendix B.10, pp. 36–37](https://arxiv.org/pdf/2511.08544v3#page=36)

![Figure 7: 고정 방향과 step별 방향 재표집의 synthetic discrepancy 비교.](/assets/reviews/lejepa/paper-figure-7.webp)

**Figure 7 읽기.** 가로축은 방향 수 $M$, 세로축은 최종 기대 directional statistic이다. 초록 재표집이 파랑 고정보다 적은 방향으로 낮은 discrepancy를 얻는다. 누적 coverage를 보여주는 합성 실험이며 신경망 SGD의 전역 수렴 증명이 아니다. Figure 6도 두 차원의 X 구조를 둔 표본을 encoder 없이 직접 최적화한 별도 실험이다. [확대](/assets/reviews/lejepa/paper-figure-7.webp) · [Figure 7, p. 10; Figure 6, p. 9](https://arxiv.org/pdf/2511.08544v3#page=10). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

<span id="5-lejepa-실제-손실의-표본정규화-확인" aria-hidden="true"></span>

## 5. LeJEPA: 안정적이고 확장 가능한 구현 (LeJEPA — Stable and Scalable Implementation)

### 5.1 SIGReg + Prediction Loss — Theorem 6, Appendix B.6·B.13

**Theorem 6: 표본 편향(sampling bias)이 어디에서 오는가.** 한 투영·주파수에서 $U_j=e^{itx_j}$라 두자(리뷰 약칭). 독립 표본이면 대각 쌍의 기대값은 1, 비대각 쌍은 $|\varphi_P|^2$다. 따라서,

$$
\begin{aligned}
\mathbb E|\widehat\varphi_N|^2
&=\frac{N+N(N-1)|\varphi_P|^2}{N^2}
=|\varphi_P|^2+\frac{1-|\varphi_P|^2}N,\\
\mathbb E|\widehat\varphi_N-\varphi_G|^2
&=|\varphi_P-\varphi_G|^2+\frac{1-|\varphi_P|^2}N.
\end{aligned}
$$

가중 적분 후 $\mathbb E\widehat D=D+N^{-1}\int w_s(1-|\varphi_P|^2)$다. 미분과 기대값 교환의 지배 조건 아래 parameter-gradient bias는 $-N^{-1}\int w_s\nabla_{\boldsymbol\theta}|\varphi_P|^2$다. 이 $O(1/N)$은 unscaled $\widehat D$와 그 population counterpart를 비교한 절대 bias다.

$EP=N\widehat D$와 $ND$를 비교하면 추가항의 절대 크기는 $O(1)$다. 기준 loss도 $N$배이므로 정규화한 bias가 작아진다는 취지는 유지되지만, Algorithm 1의 그대로인 loss·gradient가 절대적으로 $O(1/N)$ bias라고 쓸 수는 없다. deterministic quadrature에서도 표본 기대값의 대수는 같고, 별도의 적분 근사 오차가 있다. 원문은 U-statistic·sample splitting 대안을 탐구하지 않는다. [Theorem 6; Appendix B.13, pp. 38–39](https://arxiv.org/pdf/2511.08544v3#page=38)

**Eq. (5)→(6): 빠진 항을 전개하기.** 원문은 전역 뷰와 모든 뷰의 pairwise 평균을 중심 평균으로 바꾼다. $\boldsymbol\mu_n=V_g^{-1}\sum_{g=1}^{V_g}\mathbf z_{n,g}$이고 $\boldsymbol\delta_g=\mathbf z_{n,g}-\boldsymbol\mu_n$라 두면 $\sum_g\boldsymbol\delta_g=0$다.

$$
\begin{aligned}
\frac1{V_gV}\sum_{g=1}^{V_g}\sum_{v=1}^V
\|\mathbf z_{n,g}-\mathbf z_{n,v}\|^2
&=\frac1V\sum_{v=1}^V\|\boldsymbol\mu_n-\mathbf z_{n,v}\|^2\\
&\quad+\frac1{V_g}\sum_{g=1}^{V_g}\|\mathbf z_{n,g}-\boldsymbol\mu_n\|^2.
\end{aligned}
$$

**리뷰 보충 유도.** 각 차이를 $\boldsymbol\delta_g+(\boldsymbol\mu_n-\mathbf z_{n,v})$로 나누어 제곱한다. 교차항은 $\sum_g\boldsymbol\delta_g=0$으로 없어지고 두 제곱 항은 남는다. B.6에서 평균 제곱과 제곱 평균을 같다고 한 마지막 단계는 이 분산을 빠뜨린다. 전역 뷰가 모두 같아야 추가항이 0이다. scalar 뷰 $[0,2]$, $V_g=V=2$에서는 pairwise가 2, center가 1, 추가 내부 분산(within-global-view variance)이 1이다. 학습 parameter에 의존하므로 일반적인 상수 생략은 아니다. [Eq. (5)–(7); Appendix B.6, pp. 33–34](https://arxiv.org/pdf/2511.08544v3#page=33)

**Algorithm 2의 구현을 따로 읽기.** 실제 code의 중심 loss는 $(NVK)^{-1}\sum_{n,v}\|\mathbf z_{n,v}-\boldsymbol\mu_n\|^2$다. $1/K$는 feature 좌표까지 평균한 결과다. SIGReg는 뷰마다 $N$개 이미지 표현에 적용하고 뷰 평균을 낸다. Eq. (5)의 pairwise 식과 구현을 동일 목적이라고 단정하지 않되, 보고된 실험은 논문이 제시한 center 기반 pipeline으로 설명한다. 예측기(predictor)·지수이동평균 교사(exponential moving average teacher, EMA teacher)·기울기 차단(stop-gradient)는 필수 구성에서 빠지고 projector·optimizer·schedule은 남는다. [§5.1, Algorithm 2](https://arxiv.org/pdf/2511.08544v3#page=11)

**네 종류의 오차는 따로 확인해야 한다.** Theorem 6의 유한 표본 편향, 유한 방향으로 인한 투영 근사, 17점 격자의 수치 적분 오차, 그리고 유한 encoder의 최적화·표현력 오차는 서로 다른 항목이다. $N$을 늘려 첫 항을 줄여도 방향·주파수의 식별 문제가 자동으로 해결되지는 않는다. 위 alias 예제도 이 구분을 위한 국소 예제다. 이 네 항의 합에 대한 전체 신경망 오차 상한을 이 글에서 새로 증명한 것은 아니다. [§4.1–4.3·5.1, Algorithms 1–2](https://arxiv.org/pdf/2511.08544v3#page=10)

### 5.2 Relation to Prior Work — Appendix B.14

정확한 CF 적분은 kernel discrepancy와 연결되지만 quadrature 구현은 모든 sample pair를 만들지 않는다. moment test로 제한하면 모든 unit $\mathbf a$에서 $\mathbb E[\mathbf a^\top\mathbf z]=0$, $\operatorname{Var}(\mathbf a^\top\mathbf z)=1$을 요구한다.

첫 조건에서 평균을 $\boldsymbol\mu$라 하고 $\mathbf a=\boldsymbol\mu/\|\boldsymbol\mu\|$를 선택하면, 평균이 0이 아니란 가정과 모순되어 $\boldsymbol\mu=0$이다. 둘째 조건은 $\mathbf a^\top\Sigma\mathbf a=1$이다. $\mathbf a=\mathbf e_i$에서 대각 1, $\mathbf a=(\mathbf e_i+\mathbf e_j)/\sqrt2$에서 비대각 0이므로 $\Sigma=I$다.

이는 moment-level constraint의 recovery다. VICReg 원래의 standard-deviation hinge와 비대각 공분산 penalty를 포함하는 동일 목적함수라는 증명은 아니다. VICReg는 공분산 계산에 centering을 사용하지만 평균을 0으로 맞추는 별도 손실을 요구하지 않는다. sliced score matching·sliced Wasserstein·MMD와의 방법 비교 및 원전 읽은 범위는 [전체 해설 §5.2](/reviews/lejepa-complete/#52-relation-to-prior-work)에 있다. [§5.2; Appendix B.14, pp. 39–40](https://arxiv.org/pdf/2511.08544v3#page=39)

## 6–7 실험·결론으로 연결하기

본문 §6.1–6.5의 순서대로 안정성·loss 신호·도메인 내 학습·규모/전이·의미 구조를 [전체 해설](/reviews/lejepa-complete/)에 보존했다. 위 증명과 실험의 역할은 다르다. OLS 계수 이론이 모든 classification 정확도를 증명하지 않으며, CF 안정성은 모든 구조의 SGD 전역 수렴을 증명하지 않는다. Galaxy10·IN-1K 결과는 조건부 실용 근거이고, PCA·attention은 정성 근거다.

## 리뷰 해석과 남은 질문

**리뷰 해석.** 분포 선택의 근거는 유용하지만 등호 조건·정규화·유한 근사를 구분해야 구현의 보장을 정확히 읽을 수 있다.

## 출처와 읽은 범위

- 원문: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics* · Randall Balestriero et al. · 최초 공개 2025-11-11 · [검토 버전](https://arxiv.org/abs/2511.08544v3) · 검토일 2026-10-23.
- 읽은 범위: v3 §3→§4→§5와 Appendix A·B.1–B.14의 관련 정리·증명. 원문에 생략된 중간 계산은 리뷰 보충 유도로 구분한다.
- 그림 출처: 대표 개념도는 자체 제작물(CC BY 4.0). 원문 그림은 Balestriero·LeCun의 v3 자산(CC BY-SA 4.0)으로 전체 패널 crop·무손실 WebP 변환을 적용한다. [원문](https://arxiv.org/abs/2511.08544v3).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
