---
title: "[2/3] LeJEPA 전체 해설: 목표 분포에서 이론·알고리즘·실험까지"
description: "원문 §§1–7과 하위 절을 유지하며 목표 분포의 이론, SIGReg 설계, 관련 연구, 핵심 실험과 반대 결과를 설명한다."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-11T20:35:20+09:00"
topics: ["self-supervised learning", "representation learning", "joint embedding predictive architectures", "distribution matching", "computer vision"]
visibility: "public"
lang: "ko"
translationKey: "lejepa-complete"
paperTitle: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
paperPublishedDate: "2025-11-11"
authors: ["Randall Balestriero", "Yann LeCun"]
year: "2025"
paperUrl: "https://arxiv.org/abs/2511.08544v3"
thumbnail: "/assets/reviews/lejepa/lejepa-core.svg"
thumbnailAlt: "LeJEPA의 두 제약 설명도: 한 이미지의 뷰들은 전역 뷰 평균에 정렬하고, 각 뷰에서 여러 이미지의 표현은 임의 방향으로 투영해 표준 Gaussian과 맞춘다."
---

**LeJEPA의 논증은 ‘후속 과제(downstream task)에 유리한 표현 분포는 무엇인가’에서 시작해, 그 분포를 안정적으로 맞추는 손실과 실제 이미지 학습으로 이어진다.** 저자들은 선형·국소 비선형 프로브(probe)를 분석해 등방성 가우시안(isotropic Gaussian)을 목표로 삼고, SIGReg(Sketched Isotropic Gaussian Regularization)와 이미지 뷰 합의(view agreement)를 결합한다. 이론의 조건과 근사 구현을 구분해야 하지만, 분포 선택·계산·실험을 함께 설명하는 설계가 이 논문의 핵심이다. [§§1–7](https://arxiv.org/pdf/2511.08544v3#page=1)

이 글은 **[2/3] 원문 순서의 전체 해설**이다. [1/3 입문](/reviews/lejepa-overview/) → 이 글 → [3/3 정리·증명 심층](/reviews/lejepa-technical/) 순으로 읽으면 된다. 아래 제목과 하위 절 번호는 원문을 따른다. 주요 증명의 진술·조건은 해당 위치에 남기고, 긴 중간 계산은 심층 편의 같은 §3→§4→§5 순서로 연결했다.

**표기.** 원문의 $f_{\boldsymbol\theta}$, $\mathbf z_{n,v}$, $\mathbf Z$, $K$, $V_g$, $\boldsymbol\mu_n$을 유지한다. $M=|\mathcal A|$는 방향 수, $T$는 주파수 grid 크기의 보조 약칭이다. §3의 ridge 계수 $\lambda$와 §5의 손실 혼합 계수 $\lambda$는 서로 다른 역할이다. 부록에서 같은 차원을 $d$, 표본 수를 $n$으로 쓰는 경우 원문 대응을 설명한다.

대표 그림의 두 손실은 표본을 모으는 축이 다르다. 왼쪽 아래는 한 이미지 안의 뷰 합의, 오른쪽 아래는 각 뷰에서 이미지들 사이의 분포 정규화다. Gaussian 제약은 projector 뒤의 공간에 적용한다. [대표 그림 확대](/assets/reviews/lejepa/lejepa-core.svg)

## 1 Introduction — 저자가 풀려는 문제와 전체 경로

자기지도학습(self-supervised learning)은 정답 라벨 대신 입력들 사이의 관계로 특징을 배운다. 이미지에서는 같은 원본에서 만든 crop·augmentation을 연결한다. 하지만 표현 거리만 줄이면 상수 출력도 해가 된다. 표현 붕괴(representation collapse)를 피하면서 다양한 데이터·구조에 적용하려면 학습 목적 자체를 명확히 해야 한다.

저자들은 경험적으로 유용한 장치들이 늘어나면서 재설정 비용이 커지고, 붕괴 방지 이론만으로는 어떤 표현 기하가 좋은지 결정하기 어렵다고 지적한다. 제안의 경로는 다음과 같다. §2에서 JEPA의 요구를 정하고, §3에서 후속 예측을 통해 목표 분포를 고른다. §4는 고차원 분포 일치를 계산하고, §5는 학습 손실을 구성한다. §6은 안정성·손실 신호·도메인·규모·의미 구조를 평가한다. [§1, pp. 1–3](https://arxiv.org/pdf/2511.08544v3#page=1)

## 2 Background and Notations — 정의와 설계 요구

### 2.1 Notations and Definitions

입력 데이터의 크기는 $(N,V,D)$다. $N$은 독립 표본 수, $V$는 표본마다 만든 뷰(view) 수, $D$는 입력 차원이다. 인코더(encoder) $f_{\boldsymbol\theta}:\mathbb R^D\to\mathbb R^K$가 $K$차원 표현을 만든다. $\mathbf Z\in\mathbb R^{N\times K}$는 한 뷰에서 모은 표현 행렬이다.

JEPA의 두 요구는 관련 입력의 표현을 예측할 수 있는 **예측 가능성**(predictability)과, 표현이 입력을 구별하는 정보를 잃지 않는 **표현의 비퇴화성**(non-degeneracy)이다. 같은 이미지의 대칭적 뷰 합의에서는 별도 예측기(predictor)를 빼는 선택이 가능하다. 정보가 다른 context와 target, 시간·행동 조건부 예측에는 같은 결론을 그대로 적용할 수 없다. 이 논문의 실험은 주로 이미지 표현 학습이다. [§2.1, p. 4](https://arxiv.org/pdf/2511.08544v3#page=4)

### 2.2 The Need for Reliable Pretraining

사전학습법은 특정 ViT나 대규모 자연 이미지뿐 아니라 작은 데이터와 convolution·hybrid 구조에서도 안정적으로 적용할 수 있어야 한다. 설정마다 teacher 갱신·온도·prototype 등의 상호작용을 다시 탐색해야 한다면 전문 도메인에서 활용하기 어렵다. 이 문제의식이 §6.1과 §6.3의 실험 설계로 이어진다. [§2.2](https://arxiv.org/pdf/2511.08544v3#page=4)

### 2.3 The Need for Actionable Theory

붕괴하지 않는다는 조건은 필요한 출발점이지만, 후속 과제에 어떤 표현이 유리한지까지 정하지는 않는다. 저자들은 특정 라벨을 알기 전에 표현의 분포를 선택할 수 있는 이론을 요구한다. 아래 결과는 그 선택의 근거이며, 실제 신경망 전체의 성능을 직접 증명하는 정리는 아니다. [§2.3](https://arxiv.org/pdf/2511.08544v3#page=4)

## 3 Latent Euclidean — Embeddings Should Be Isotropic Gaussian

### 3.1 Linear Probing

표현을 고정하고 라벨 $\mathbf y$를 선형 예측하는 문제부터 시작한다. 원문의 최소제곱법(ordinary least squares, OLS) 표기는 리지 정규화(ridge regularization)까지 포함한다.

$$
\hat\beta=\underset{\beta\in\mathbb R^K}{\arg\min}
\|\mathbf y-\mathbf Z\beta\|_2^2+\lambda\|\beta\|_2^2.
$$

$\mathbf Z_{\rm iso}$와 $\mathbf Z_{\rm aniso}$는 같은 열공간(column span)과 총 에너지를 갖고 공분산 모양만 다르게 비교한다. 고정 설계(fixed design), 열들이 선형 독립인 조건(full column rank), 평균 0·동일 분산의 잡음 등 Appendix B.1–B.2의 조건을 사용한다.

**Lemma 1 — Anisotropy amplifies bias.** $\lambda>0$일 때, 비등방성(anisotropy) 표현에서는 등방성(isotropy)보다 계수 추정 편향(bias)이 커지는 후속 과제가 **존재한다**. 작은 고윳값(eigenvalue) 방향으로 실제 과제 계수를 잡으면 ridge의 수축(shrinkage)이 더 크기 때문이다. 모든 라벨에서 bias가 더 크다는 진술은 아니다.

**Lemma 2 — Anisotropy amplifies variance.** $\lambda=0$에서 계수 분산(variance)의 합은 등방성에서 최소다. $\lambda_k$를 그람 행렬(Gram matrix) $\mathbf Z^\top\mathbf Z$의 고윳값으로 쓰면,

$$
\operatorname{tr}\operatorname{Var}(\hat{\boldsymbol\beta}\mid\mathbf Z)
=\sigma^2\sum_{k=1}^K\frac1{\lambda_k}.
$$

총 에너지가 같을 때 역수 합은 모든 고윳값이 같으면 최소다. 설명용 예제 $(1,1)$은 역수 합 2, $(0.2,1.8)$은 약 5.56이다. **계수 분산**에 관한 결론을 모든 test point의 예측 위험과 동일시하면 안 된다. 증명의 ridge 전개와 Jensen 부등식은 [심층 편 §3.1](/reviews/lejepa-technical/#31-linear-probing-lemmas-12와-appendix-b1b2)에 풀었다. [§3.1, p. 5; Appendix B.1–B.2, pp. 26–27](https://arxiv.org/pdf/2511.08544v3#page=26)

### 3.2 Nonlinear Probing

등방성 공분산(isotropic covariance)을 갖는 분포는 많다. Gaussian이라는 모양을 선택할 근거를 얻기 위해 저자들은 반경 기반 최근접 이웃(k-nearest neighbors, kNN)과 Nadaraya–Watson 커널 회귀(kernel regression)를 분석한다.

$$
\widehat{\mathbf y}(\mathbf q)=\frac1{|\mathcal N_{r_0}(\mathbf q)|}
\sum_{n\in\mathcal N_{r_0}(\mathbf q)}\mathbf y_n,
\qquad
\mathcal N_{r_0}(\mathbf q)=\{n:\|\mathbf z_n-\mathbf q\|\le r_0\}.
$$

$$
\widehat{\mathbf y}(\mathbf q)=
\frac{\sum_{n=1}^N K_h(\mathbf q-\mathbf z_n)\mathbf y_n}
{\sum_{n=1}^N K_h(\mathbf q-\mathbf z_n)}.
$$

$\mathbf q$는 질의 표현, $r_0$는 이웃 반경, $h$는 커널 대역폭(kernel bandwidth)이다. 여기의 kernel 함수 $K$와 표현 차원 $K$는 원문이 재사용하는 기호다. 반경 kNN은 고정 개수의 이웃을 고르는 통상적 kNN과 구분한다.

**Theorem 1 — Isotropic Gaussian Optimality.** 저자들은 적분 제곱 편향(integrated squared bias, ISB)을 다음처럼 연결한다.

$$
\begin{aligned}
\mathrm{ISB}_{k\text{-NN}}&=\frac{r_0^4}{(K+2)^2}\tau_g^2J(p)+O(r_0^4),\\
\mathrm{ISB}_{\rm kernel}&\le
\left(\frac{h^2\mu_2(K)}2\right)^2
\left(2B^2+8L^2J(p)\right)+o(h^4),\\
J(p)&=\int\|\nabla\log p(\mathbf z)\|_2^2p(\mathbf z)\,d\mathbf z.
\end{aligned}
$$

가정은 중요하다. 밀도는 충분히 매끄럽고 피셔 정보(Fisher information)·공분산(covariance)이 유한해야 하며, query는 train과 같은 밀도를 따른다. kNN은 작은 반경과 task-gradient의 등방성 사전 가정, kernel은 작은 $h$·충분한 국소 표본·대칭 정규화 kernel과 $\|\nabla\eta\|\le L$, $|\Delta\eta|\le B$ 조건을 사용한다. $B$는 여기서 곡률 상한이며 §5의 batch 기호와 다르다. [Appendix A, Lemmas 4–5; Theorems 7–8](https://arxiv.org/pdf/2511.08544v3#page=24)

고정 공분산 $\Sigma$에서 $J(p)\ge\operatorname{tr}(\Sigma^{-1})$이고 등호는 Gaussian에서 성립한다. 총 분산까지 고정하면 등방성 $\Sigma$가 선택된다. 이 연결이 Gaussian 목표의 수학적 중심이다. 다만 kNN의 $O(r_0^4)$ 항은 주항과 같은 차수이며 교차항 제거에는 추가 직교 조건이 필요하다. kernel 식은 최악 경우 bias의 **상한**이다. 따라서 이 계산만으로 모든 과제의 정확한 ISB·전체 MSE가 유일하게 최소라고 결론내릴 수는 없다. 저자의 최적성 주장은 그 조건과 논증 범위 안에서 읽는다. 자세한 Taylor 전개와 Fisher 하한은 [심층 편 §3.2](/reviews/lejepa-technical/#32-nonlinear-probing-theorem-1과-appendix-a-b3b7)에 있다. [§3.2; Appendix B.3–B.7](https://arxiv.org/pdf/2511.08544v3#page=28)

### 3.3 Geometric and Practical Insights

저자들은 같은 총 분산의 분포에서 추정 계수와 결정 경계가 변하는 설명용 실험을 배치한다. Figure 3은 두 클래스, 표본 100개에서 logistic regression을 반복해 등방성·비등방성 결정 경계를 비교한다. 이는 OLS 이론과 연결되는 경험적 그림이며 동일한 정리의 가정을 그대로 재현한 것은 아니다. Appendix의 Figure 18도 정규화와 bias를 시각화한다. Figure 17 캡션의 분산 방향은 Lemma 2와 상충하므로 독립적인 확인 근거로 사용하지 않는다. [§3.3, Figure 3; Figure 17, p. 48](https://arxiv.org/pdf/2511.08544v3#page=6)

## 4 SIGReg — Reliable Isotropic Gaussian Regularization in High-Dimension

### 4.1 Hypothesis Testing as a Judge

고차원 밀도를 직접 추정하지 않고 방향별 1D 법칙을 비교한다. 표현 법칙을 $P_{\boldsymbol\theta}$, 목표를 $Q=\mathcal N(0,I)$라 한다. **Lemma 3 — Hyperspherical Cramér–Wold**는 모든 unit 방향에서 두 투영 법칙이 같음과 원래 법칙의 같음을 연결한다. $\mathbf t=r\mathbf a$로 특성함수(characteristic function, CF)를 비교하는 증명이 Appendix B.8에 있다.

**Theorem 2 — Sufficiency of directional tests**는 방향별 검정을 결합한 일관성(consistency)을 논한다. Appendix B.9는 방향 집합의 mesh가 0으로 가는 조건, 전체 귀무가설(null hypothesis)에 맞춘 임계값(threshold), 분포를 구별하는 방향 주변에서의 검정력(statistical power)을 요구한다. 유한한 임의 방향 몇 개가 모든 분포를 인증한다는 정리가 아니다. 검정의 max 통계와 학습용 평균 SIGReg도 구분해야 한다.

$$
\operatorname{SIGReg}(\{\mathbf z_n\}_{n=1}^N)
=\frac1{|\mathcal A|}\sum_{\mathbf a\in\mathcal A}
T(\{\mathbf a^\top\mathbf z_n\}_{n=1}^N).
$$

이 식은 Definition 2의 방향 평균이다. 통계 $T$를 고르는 문제가 다음 §4.2다. [§4.1, Lemma 3, Theorem 2, Definition 2; Appendix B.8–B.9](https://arxiv.org/pdf/2511.08544v3#page=6)

![Figure 2: 입력·표현 공간·방향별 밀도 비교.](/assets/reviews/lejepa/paper-figure-2.webp)

**Figure 2 읽기.** 가운데 색 화살표마다 오른쪽의 투영 밀도를 얻는다. 가로축은 투영 좌표, 검은 곡선은 Gaussian 목표다. 회색 차이는 분포 discrepancy의 개념 표현이며 downstream 정확도가 아니다. [확대](/assets/reviews/lejepa/paper-figure-2.webp) · [Figure 2, p. 3](https://arxiv.org/pdf/2511.08544v3#page=3). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

### 4.2 SIGReg: Sketching the Epps–Pulley Test Is Stable and Scalable

#### 4.2.1 Moments Are Unstable and Insufficient

**Theorem 3**은 유한 차수까지 moment가 같은 서로 다른 분포를 구성한다. 평균·분산·공분산을 맞추는 것만으로 결합 Gaussian을 보장하지 못한다. 차수를 높여도 꼬리 표본에서 다항식 gradient가 커지는 문제가 있다. Appendix B.11의 선형대수 구성은 [심층 편 §4.2](/reviews/lejepa-technical/#42-sigreg-sketching-the-eppspulley-test)에서 설명한다. [§4.2.1; Appendix B.11](https://arxiv.org/pdf/2511.08544v3#page=7)

![Figure 5: 좌표 주변 분포와 공분산이 목표에 맞아도 결합 분포는 X자일 수 있다.](/assets/reviews/lejepa/paper-figure-5.webp)

**Figure 5 읽기.** 좌표별 밀도는 Gaussian이고 공분산은 identity지만 결합 표본은 X자 구조다. 사선 방향으로 투영하면 Gaussian과 다른 밀도가 나타난다. 오른쪽은 방향에 따른 통계의 반응으로, 통계들의 척도가 같다는 뜻은 아니다. ‘좌표별로 좋아 보임’과 ‘결합 법칙 일치’의 차이를 읽어야 한다. [확대](/assets/reviews/lejepa/paper-figure-5.webp) · [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

#### 4.2.2 Cumulative Density Functions Are Impractical

누적분포함수(cumulative distribution function, CDF)는 한 임계값 이하의 확률을 잰다. 분포를 표현할 수 있지만, 원문은 순서통계(order statistics)·정렬과 여러 GPU의 표본을 모으는 비용, gradient 학습에 맞춘 근사를 문제로 든다. 정렬이 모든 점에서 미분 불가능하다는 뜻으로 과장할 필요는 없다. 핵심은 선택한 학습·분산 처리 요구에 비해 실용성이 떨어진다는 평가다. [§4.2.2, p. 9](https://arxiv.org/pdf/2511.08544v3#page=9)

#### 4.2.3 Characteristic Functions Are Stable, Scalable and Identifiable

특성함수(characteristic function, CF)는 복소 지수함수의 기대값이다. 아래 방향 첨자 표기는 본문의 $\hat\phi_X$, $\phi$를 해당 투영과 Gaussian 목표에 연결한 것이다. 표본은 평균으로 계산하고 Gaussian 목표는 닫힌 식을 사용한다.

$$
\hat\varphi_{\mathbf a}(t)=\frac1N\sum_{n=1}^N e^{it\mathbf a^\top\mathbf z_n},
\qquad \varphi_{\mathcal N}(t)=e^{-t^2/2}.
$$

$$
EP(\mathbf a)=N\int_{\mathbb R}
|\hat\varphi_{\mathbf a}(t)-\varphi_{\mathcal N}(t)|^2w(t)\,dt.
$$

원문의 Epps–Pulley 식은 Gaussian 가중 $w(t)$를 사용한다. **Theorem 4 — Stability of Epps–Pulley Test**의 근거는 $|e^{itz}|=1$이라 입력 크기에 관계없이 CF가 유계라는 사실이다. 다만 Appendix B.12의 $O(1/N)$ sample-gradient 유도는 $EP/N$인 **비례배수 없는 거리**에 대한 것이다. $EP=N\widehat D$에는 gradient도 $N$배다. 이 정규화 차이를 포함한 계산은 [심층 편 §4.2.3](/reviews/lejepa-technical/#423-cf-theorem-4의-유도와-정규화)에 남겼다. encoder parameter의 gradient에는 encoder Jacobian도 들어가므로 CF 유계성만으로 전체 신경망 gradient 상한을 무조건 정할 수는 없다. [§4.2.3, Epps–Pulley 식, Theorem 4; Appendix B.12](https://arxiv.org/pdf/2511.08544v3#page=9)

**Algorithm 1을 읽는 순서.** $K\times M$ Gaussian 행렬의 열을 unit 길이로 정규화해 방향을 만든다. $\mathbf Z\mathbf A$를 주파수와 곱해 표본 특성함수(empirical characteristic function, ECF)를 계산한다. step seed를 맞춰 GPU마다 같은 방향을 사용하고 CF 평균을 all-reduce한다. $[-5,5]$의 17개 grid에서 Gaussian CF와의 오차를 Gaussian 가중해 trapezoidal 적분한 뒤 전체 표본 수 $N$을 곱한다. 각 방향 통계는 SIGReg에서 평균한다. 이 설명은 원문 Algorithm 1 기준이며 최신 공식 코드 실행 결과가 아니다. [Algorithm 1, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10)

dense 계산 비용은 보조 기호 $M,T$를 쓰면 $\mathcal O(NKM+NMT)$다. $K,M,T$를 고정할 때 $N$에 선형이며 encoder까지 포함한 모든 비용이 선형이라는 뜻은 아니다. 원문 Table 6은 V100-SXM2-16GB에서 loss forward/backward의 10회 측정(mean ± std)이다. $(N,M,T)=(512,512,16)$은 $0.465236\pm0.011642$ ms, $(8192,512,16)$은 $6.188304\pm0.007226$ ms다. embedding 차원은 표에 미보고이며 전체 학습 속도 비교로 바꾸지 않는다. [Table 6, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

### 4.3 How SIGReg Beats the Curse of Dimensionality

원문은 **매끄러움**(smoothness), **SGD 방향 재표집**(resampling), **합성 데이터 검증**(synthetic validation)의 순서로 설명한다. Theorem 5의 감소율은 방향 수 $M$에 대해 $M^{-2\alpha/(K-1)}$이며, $\alpha$는 소볼레프 매끄러움(Sobolev smoothness)이다. Appendix B.10은 구면 조화함수(spherical harmonics)의 근사와 준균일한 점 배치(quasi-uniform point placement)에서의 보간(interpolation)을 참조한다. 매끄러움이 크다는 사실만으로 임의의 $O(K)$ 방향에서 실용 오차가 항상 작다고 결론낼 수 없다. 차원·매끄러움에 따른 상수와 norm도 남아 있다. [§4.3, Theorem 5; Appendix B.10](https://arxiv.org/pdf/2511.08544v3#page=10)

![Figure 7: 방향 수에 따른 최종 directional statistic, 고정 방향과 재표집 비교.](/assets/reviews/lejepa/paper-figure-7.webp)

**Figure 7 읽기.** 가로축은 step마다 쓰는 방향 수 $M$, 세로축은 학습 후 기대 directional statistic이다. 낮을수록 목표 분포에 가깝다. 파란 고정 방향보다 초록 재표집 방향이 적은 $M$에서도 낮은 discrepancy를 얻는다. 이는 합성 최적화 결과이며 ImageNet 정확도 그래프나 정식 수렴률은 아니다. [확대](/assets/reviews/lejepa/paper-figure-7.webp) · [Figure 7, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

이어 합성 실험은 대부분의 차원은 Gaussian이고 두 차원만 X자 구조인 표본을 **직접 최적화**한다. §4.3 본문은 $D=512$, $M=16$ 사례를 언급하지만, Figure 6 캡션의 설정은 $N=100$, $D=1024$, $M=10$이다. 두 설정을 동일시하지 않는다. encoder 제약 없이 loss 자체를 보는 통제 실험으로, 이미지의 의미 학습은 뒤의 §6에서 따로 확인한다. [§4.3, p. 11](https://arxiv.org/pdf/2511.08544v3#page=11) · [Figure 6, p. 9](https://arxiv.org/pdf/2511.08544v3#page=9)

## 5 LeJEPA — Stable and Scalable Implementation

### 5.1 LeJEPA: SIGReg + Prediction Loss

한 이미지에 전역 뷰(global view) $V_g$개와 지역 뷰(local view) $V_l$개, 총 $V=V_g+V_l$개를 만든다. 각 뷰를 인코더(encoder)·투영 헤드(projector)로 처리한다. $\boldsymbol\mu_n$은 전역 뷰 표현의 평균이다. Algorithm 2의 실제 중심 기반 손실을 설명하면 다음과 같다.

$$
\begin{aligned}
\boldsymbol\mu_n&=\frac1{V_g}\sum_{v=1}^{V_g}\mathbf z_{n,v},\\
\mathcal L_{\rm pred}^{\rm code}&=\frac1{NVK}\sum_{n=1}^N\sum_{v=1}^V
\|\mathbf z_{n,v}-\boldsymbol\mu_n\|_2^2,\\
\mathcal L_{\rm SIGReg}&=\frac1V\sum_{v=1}^V\operatorname{SIGReg}(\{\mathbf z_{n,v}\}_{n=1}^N),\\
\mathcal L_{\rm LeJEPA}&=(1-\lambda)\mathcal L_{\rm pred}^{\rm code}+\lambda\mathcal L_{\rm SIGReg}.
\end{aligned}
$$

$\mathcal L_{\rm pred}^{\rm code}$와 $\mathcal L_{\rm SIGReg}$는 리뷰의 약칭이며 $1/K$는 코드의 feature 평균을 표시한다. 원문 총 손실의 batch 기호 $B$가 여기의 $N$에 해당한다. 중심을 계산하는 뷰 내부 축과 SIGReg가 모으는 이미지 간 축을 혼동하면 안 된다.

**두 가지 수식 확인.** Theorem 6의 표본·gradient bias는 $EP/N$인 비례배수 없는 CF 거리에서 $O(1/N)$이다. Eq. (5)의 전역-전체 pairwise 평균과 Eq. (6)의 중심 평균 사이에는 전역 뷰의 내부 분산이 추가된다. 이 항은 학습 parameter에 의존하므로 임의의 상수로 버릴 수 없다. 원문 Appendix B.6, B.13과 [심층 편 §5.1](/reviews/lejepa-technical/#51-sigreg-prediction-loss-theorem-6-appendix-b6b13)에서 정확한 항등식과 표본 기대값을 나란히 설명한다. [§5.1, Eq. (5)–(7), Theorem 6; Algorithm 2](https://arxiv.org/pdf/2511.08544v3#page=11)

핵심 구조에는 프로토타입(prototype)·기울기 차단(stop-gradient)·지수이동평균 교사(exponential moving average teacher, EMA teacher)·별도 예측기(predictor)가 필수가 아니다. projector는 여전히 사용하며 평가에서는 backbone을 활용한다. 이미지의 대칭 합의에서 predictor를 제거했다는 사실과 모든 JEPA 과제에서 predictor가 불필요하다는 주장은 구분한다. 보통 optimizer·data loader·학습률 schedule을 포함한 설정도 남는다.

### 5.2 Relation to Prior Work

원문은 sliced score matching, sliced Wasserstein, 정확 적분과 MMD, VICReg의 moment 제한 연결을 이 순서로 설명한다. 투영이라는 재료와 분포 discrepancy가 다른 분야에 존재했음을 인정하면서 JEPA 손실로 결합한 위치를 설명한다. Appendix B.14는 모든 방향의 평균 0·분산 1이 전체 평균 0·공분산 $I$를 강제함을 보인다. 이것은 VICReg 원래의 표준편차 하한 힌지(hinge)·공분산 페널티(penalty)와 수치적으로 똑같은 손실이라는 증명은 아니다. [§5.2; Appendix B.14, pp. 39–40](https://arxiv.org/pdf/2511.08544v3#page=12)

다음은 이전 분석에서 직접 확인한 주요 원전 5개의 방법 비교다. 이번에는 이 표를 원문의 관련 연구 위치로 옮겼다. 읽은 범위는 각 방법 절이며 전문 재독으로 표시하지 않는다.

| 원전·읽은 범위 | 방법의 핵심 | LeJEPA와의 차이 |
| --- | --- | --- |
| [VICReg, §§3.1–3.3, pp. 3–4](https://arxiv.org/pdf/2105.04906v3#page=3) | 뷰 합의, 표준편차 하한, 비대각 공분산 penalty | 대칭적 붕괴 방지는 선행한다. SIGReg는 유한 차수 통계보다 전체 투영 법칙을 겨냥한다. |
| [DINO, §3, pp. 2–3](https://arxiv.org/pdf/2104.14294v2#page=2) | EMA teacher, centering·sharpening, 출력 확률 일치 | 명시적 Gaussian 목표와 대칭 합의를 사용한다. 구조 다양성 자체가 처음은 아니다. |
| [I-JEPA, §3, pp. 3–4](https://arxiv.org/pdf/2301.08243v3#page=3) | context에서 target block 표현을 predictor로 예측 | 비대칭 정보 예측과 대칭 crop 합의의 과제가 다르다. |
| [Gaussian Embeddings, pp. 1, 3–4](https://arxiv.org/pdf/2510.05949v1#page=1) | Gaussian 밀도와 표현 변환·Jacobian 부피 | Gaussian이라는 공통점과 후속 추정·sliced CF 논증은 구분한다. |
| [Gretton et al., MMD, PDF pp. 1, 3–4](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) | kernel 평균 임베딩을 통한 분포 비교 | 정확 적분 연결의 배경. MMD에도 선형 시간 추정기가 있으므로 최초의 선형 분포 비교라고 할 수는 없다. |

## 6 LeJEPA: Empirical Validation

실험을 원문의 §6.1→§6.5 순서로 읽는다. 수치는 **논문 보고**, 정확도 단위는 %, 높을수록 좋다. 표현 고정 프로브(frozen probe)는 인코더(encoder)를 고정하고 라벨로 분류기를 학습하는 평가이며 미세조정(finetuning)은 특징 추출기(backbone)도 수정한다. 자기지도 사전학습에 라벨이 없다는 것과 평가에 라벨이 없다는 것은 다르다.

### 6.1 LeJEPA’s Stability Across Hyper-Parameters and Architectures

첫 질문은 설정과 구조를 바꿔도 유용하게 학습되는가다. Table 1의 조건은 **ImageNet-1K, ViT-Large/14, 100 epoch, frozen linear top-1**이다.

| 변경 축 | 원문 대표 결과 | 해석 |
| --- | --- | --- |
| 적분 범위, 17개 grid | $M=512$: $[-1,1]$ 72.13 → $[-5,5]$ 74.21 | 약 2.08%p 차이가 있어 범위 선택이 무의미한 것은 아니다. |
| batch | 128: 72.20; 256: 74.15; 512: 74.72; 1024: 74.07 | 작은 batch도 학습 가능하지만 모든 정확도가 같거나 단조 증가하지 않는다. |
| 전역 뷰, 전체 4개 | $V_g=1$: 53.06; $V_g=2$: 72.26 | 뷰 구성이 성능에 크게 영향을 줄 수 있다. |
| 전역 뷰, 전체 10개 | $V_g=2$: 74.06; $V_g=4$: 75.08 | 권장값 하나가 모든 설정의 최댓값은 아니다. |
| register, $M=4096$ | 0개: 75.61; 8개: 75.84 | 이 비교에서 개선은 작다. |

[Table 1, p. 13](https://arxiv.org/pdf/2511.08544v3#page=13). 서로 다른 구성요소 변경 실험(ablation) 행을 합쳐 하나의 모델 설정처럼 읽지 않는다.

원문의 권장 recipe는 $\lambda=0.05$, 전역 2·지역 8뷰, $N\ge128$, $M=1024$, 17점 $[-5,5]$다. 한편 Experiment Details 1의 공통 실험 설정은 전역 2·지역 6뷰의 **총 8뷰**, AdamW, warmup·cosine schedule이며 learning rate와 weight decay 후보를 사용한다. 권장 recipe와 실험 공통 설정이 다르므로 구분해 기록했다.

Figure 9는 IN-10에서 20M 미만의 약 50개 timm 모델, 8개 구조 계열을 frozen 평가한다. 이는 모든 모델의 IN-1K 결과가 아니다. Figure 8과 Table 7의 $\lambda$·뷰 수 분석도 낮은 민감도를 주장하지만 극단 설정의 성능 하락은 남아 있다. 예를 들어 ResNet50·8뷰에서 $\lambda=0.020$은 83.49, $0.500$은 77.72다. [§6.1, Figures 8·9; Table 7, p. 43](https://arxiv.org/pdf/2511.08544v3#page=14)

추가 ablation인 Table 4(IN-100, 400 epoch)는 projector 깊이, predictor, SWA를 비교한다. 확률적 가중치 평균(stochastic weight averaging, SWA)은 학습 모델들의 가중치 평균이며 여기서 선택적 보완이다. predictor 없이 ViT-S 3층 projector는 81.07, SWA를 쓰면 84.12다. ResNet50의 같은 비교는 83.93→83.50으로 개선되지 않는다. EMA teacher가 붕괴 방지에 필수라는 결과는 아니다. [Table 4, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42)

### 6.2 LeJEPA’s Training Loss Is Informative of Downstream Performance

두 번째 질문은 라벨을 보기 전 학습 손실이 품질에 관한 신호를 주는가다. loss 항을 개별적으로만 최소화하면 부족하다. 합의만 강하면 붕괴할 수 있고, Gaussian만 맞추면 이미지 뷰의 공통 정보를 보장하지 못한다.

![Figure 10: prediction loss와 SIGReg loss 평면에 표시한 downstream 정확도.](/assets/reviews/lejepa/paper-figure-10.webp)

**Figure 10 읽기.** 두 좌표는 prediction과 SIGReg 손실, 색은 downstream 정확도이며 파랑에서 빨강으로 높아진다. 두 손실이 함께 낮은 왼쪽 아래의 frontier를 봐야 한다. 한 축만 낮다고 높은 성능은 아니다. 서로 다른 $\lambda$가 만드는 최종 loss 값을 단순 비교하는 문제도 보인다. [확대](/assets/reviews/lejepa/paper-figure-10.webp) · [Figure 10, p. 15](https://arxiv.org/pdf/2511.08544v3#page=15). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

Figure 11은 $\mathcal L/\lambda^\alpha$ 같은 보정과 스피어만 순위상관(Spearman rank correlation)을 조사한다. 본문은 85%에서 $\alpha\approx0.4$의 99%를 보고한다. 낮은 손실·높은 정확도 관계에 대한 부호 규약은 명료하지 않아 **높은 단조 관련성**으로 표현한다. $\alpha$ 선택에는 관측한 downstream 결과가 쓰인다. 새로운 도메인에서 평가 라벨 없이 같은 보정이 통한다는 전향적 보장은 아니다. [§6.2, Figures 10–11, 추가 Figures 19·21](https://arxiv.org/pdf/2511.08544v3#page=15)

### 6.3 In-Domain LeJEPA Outperforms Frontier Model Transfer Learning

세 번째 질문은 일반 이미지 기반 모델(foundation model)을 가져오는 대신 전문 도메인에서 작은 모델을 직접 사전학습할 가치가 있는가다. Galaxy10은 약 11,000개 학습 이미지와 10개 은하 형태 클래스다. LeJEPA는 기본 pipeline으로 **400 epoch** 도메인 내 사전학습(in-domain pretraining)을 한다. DINOv2/v3는 일반 이미지 사전학습 모델을 가져온 비교다.

![Figure 12: Galaxy10의 라벨 수별 full finetuning과 frozen 평가.](/assets/reviews/lejepa/paper-figure-12.webp)

**Figure 12 읽기.** 가로축은 후속 분류에서 클래스당 사용하는 라벨 수, 세로축은 정확도다. 왼쪽은 full finetuning, 오른쪽은 frozen 평가다. LeJEPA 곡선들은 작은 도메인 모델이며 라벨이 많아지는 구간에서 강점이 뚜렷하다. 3개 seed를 사용했지만 오차막대가 표준편차인지 신뢰구간인지 명시하지 않아 추정하지 않는다. [확대](/assets/reviews/lejepa/paper-figure-12.webp) · [Figure 12, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

| Table 3 모델 | 전체 라벨 frozen | 전체 라벨 finetuning | 1-shot frozen |
| --- | --- | --- | --- |
| LeJEPA ConvNeXtV2-Nano | 76.52 | 82.72 | 28.74 |
| LeJEPA ResNet-34 | 78.17 | 83.28 | 31.08 |
| LeJEPA LeViT-128 | 69.00 | 79.41 | 25.85 |
| DINOv2 Small | 67.62 | 78.34 | 27.68 |
| DINOv3 ViT-S/16 | 71.38 | 81.60 | 30.17 |

[Table 3, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42). 전체 라벨에서의 도메인 내 학습 강점과 1-shot의 예외를 함께 보존했다. 도메인·구조·사전학습 예산이 다르므로 순수한 loss 교체의 통제 비교는 아니다.

Table 5는 flowers102의 1,020개 학습 표본 등 작은 데이터로 넓힌다. flowers102 frozen에서 ResNet18은 74.57, ResNeXt26ts는 82.19, I-JEPA 비교는 85.76이다. Galaxy10에서는 ResNet34 77.29와 I-JEPA 62.93이다. 작은 데이터에서 실용적이어도 모든 데이터에서 최고는 아니다. Table 5의 I-JEPA 행 이름은 IN-22K, 사전학습 칸은 IN-1K로 상충해 출처 확정은 미확인이다. Table 3과 5의 Galaxy 수치도 하나로 섞지 않는다. [Table 5, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

### 6.4 LeJEPA Scales Across Data and Models

큰 데이터와 모델에서는 IN-1K 온라인 선형 프로브(online linear probe)가 ViT-L(0.3B) 77.1, ConvNeXtV2-H(0.6B) 78.5다. 이 수치와 아래 **8개 전이학습(transfer learning) 데이터셋 평균**은 다른 평가다. 초록의 ViT-H 약 79%도 같은 행으로 합치지 않는다. [§6.4, p. 17](https://arxiv.org/pdf/2511.08544v3#page=17)

| Table 2 모델·조건 | 1-shot 평균 | 10-shot 평균 | 전체 라벨 평균 |
| --- | --- | --- | --- |
| LeJEPA ViT-L, 304M, IN-1K, 100 epoch | 29.55 | 60.95 | 79.48 |
| LeJEPA ConvNeXtV2-H, 660M, IN-1K, 100 epoch | 31.58 | 60.70 | 77.56 |
| I-JEPA ViT-H, 632M, IN-1K, 300 epoch | 30.20 | 60.51 | 78.50 |
| I-JEPA ViT-H + STOP, 같은 크기·데이터·epoch | 32.05 | 62.92 | 80.70 |
| I-JEPA ViT-H, IN-22K, 900 epoch | 35.15 | 63.28 | 80.82 |

[Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16). DTD·flowers102·food101 등 일부 과제에서 강하지만 CIFAR·pets 등에서는 열위가 있다. ViT-L의 전체 라벨 평균은 기본 I-JEPA보다 높고, 1-shot 평균은 낮다. +STOP은 위 세 평균 모두 ViT-L보다 높다. IN-22K 행은 별도 예산군이다.

100 대 300 epoch는 짧은 학습 schedule을 보여준다. 모델·해상도·crop·하드웨어가 달라 실제 wall-clock이나 FLOPs가 정확히 3배 절약됐다는 실측은 아니다. Figure 1의 ViT-gigantic(1.8B) loss 곡선은 큰 모델의 최적화 안정성을 보여주는 사례다. 전체 규모별 downstream 우위를 증명하는 표는 아니다. [§6.4; Figure 1](https://arxiv.org/pdf/2511.08544v3#page=1)

### 6.5 Emergent Semantic Structure in LeJEPA Representations

Figure 13은 [CLS] attention을 threshold해 영상 프레임의 전경(foreground)을 드러낸다. Figure 14는 ViT-L, IN-1K, 100 epoch의 패치(patch) 특징을 주성분분석(principal component analysis, PCA)의 상위 세 성분으로 RGB에 표시한다. 이 결과는 라벨 없는 사전학습에서도 공간·의미 구조가 생길 수 있다는 정성적 근거다.

PCA는 **이미지마다 따로** 수행하므로 같은 색이 모든 이미지에서 같은 의미 방향을 뜻하지 않는다. 영상의 시간적 일관성도 예시이며 IoU나 영상 segmentation benchmark 점수는 보고하지 않는다. 그림을 행동·미래 예측 성능으로 확장하지 않는다. 원문 전체 패널과 영상은 [Figures 13–14](https://arxiv.org/pdf/2511.08544v3#page=17), [공식 프로젝트 페이지](https://rbalestr-lab.github.io/lejepa/)에서 확인할 수 있다.

## 7 Conclusion — 저자의 결론과 적용 범위

저자들은 등방성 Gaussian의 이론적 동기, 안정적·확장 가능한 SIGReg, 간결한 LeJEPA, 여러 구조와 도메인 실험을 연결한다. 특정 붕괴 방지 장치들의 조합 대신 표현의 목표 분포를 먼저 정하는 방식이다. 후속 예측 분석이 실제 설계를 안내한다는 것이 원문의 결론이다. [§7, p. 18](https://arxiv.org/pdf/2511.08544v3#page=18)

## 리뷰 해석과 남은 질문

리뷰의 해석으로는 기여가 **분포 선택의 근거와 계산 가능한 제약을 결합한 설계**에 있다. 이론·실험 모두 조건이 있다. 정확한 전체 위험의 최적성, 유한 grid로 분포를 식별하는 능력, 전문 도메인별 손실-성능 관계는 추가 검증이 필요하다. Eq. (5)–(6)의 누락항과 Theorems 4·6의 정규화 차이도 구현·재현에서 먼저 정리해야 한다.

긴 증명을 삭제하는 대신 심층 편으로 옮겼다. [§3의 ridge·국소 bias·Fisher 하한, §4의 식별·안정성·방향 근사, §5의 기대값·누락항 상세 풀이](/reviews/lejepa-technical/)를 원문의 같은 순서로 읽을 수 있다.

## 출처와 읽은 범위

- 원문: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, Randall Balestriero·Yann LeCun, 최초 공개 2025-11-11. 검토본 v3(2025-11-14), 50쪽. [공식 서지·버전 이력](https://arxiv.org/abs/2511.08544). 원문 PDF·공식 HTML·저자 LaTeX를 확보해 비교했다.
- 이번 1차 읽기(2026-10-11): 초록·서론·결론·전체 목차·Figure 1–2 설명으로 전체 지도와 확인 질문을 먼저 기록했다. 2차 읽기: 본문 §§1–7, Appendix A, B.1–B.14의 주요 정리·증명, C의 추가 표·캡션, D–F의 대안 설명을 정독·대조했다. 참고문헌 전체와 구면 보간 원전의 lemma·정리 전문은 재검증하지 않았다.
- 관련 원전 5개의 방법 절은 2026-10-09 분석에서 확인한 범위다. 표에 버전·쪽을 남겼으며 이번에 전문을 다시 읽었다고 표시하지 않는다.
- 대응: 원문 §§1–7·모든 하위 절을 유지했다. 긴 Appendix A·B 계산은 심층 편으로 이동했다. C의 Table 3–7은 해당 실험·비용 위치에 가져왔고 반복 그래프는 요약했다. D–F는 §4의 대안 비교에 필요한 내용만 축약했다. 전체 표의 반복 행·모든 인용 논문의 전문은 포함하지 않는다.
- 직접 검증: 리뷰의 4개 대수·수치 예제. **미실행:** 공식 코드·GPU 사전학습, probe·finetuning·DDP·seed 재현, 전체 모델 속도·메모리 측정, 전체 증명의 형식 검증. **미확인:** Figure 11 부호 규약, Figure 12 오차막대 정의, Figure 17 캡션 상충, Table 5의 I-JEPA 사전학습 데이터 상충, Theorem 5의 참조 원전 세부.
- 대표 그림은 자체 개념도([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), Light 300·강조 600). 본문 원문 그림들은 캡션의 원문 위치와 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)를 따른다. 모든 이미지는 기존 승인 자산의 동일 파일을 재사용했다.
