---
title: "[2/3] LeJEPA 전체 해설: 목표 분포에서 이론·알고리즘·실험까지"
description: "원문 §§1–7과 하위 절을 유지하며 목표 분포의 이론, SIGReg 설계, 관련 연구, 핵심 실험과 반대 결과를 설명한다."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-23T15:26:17+00:00"
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
thumbnailAlt: "LeJEPA의 두 제약 설명도: 한 이미지의 뷰들은 전역 뷰 평균에 정렬하고, 각 뷰에서 여러 이미지의 표현은 임의 방향으로 투영해 표준 가우시안과 맞춘다."
---

LeJEPA는 후속 과제(downstream task)의 추정 오차 분석에서 등방성 가우시안(isotropic Gaussian) 목표를 도출하고, 이를 등방성 가우시안 투영 정규화(Sketched Isotropic Gaussian Regularization, SIGReg)와 뷰 합의(view agreement)로 구현한다. Galaxy10에서 400 epoch 도메인 내 사전학습한 ResNet-34는 전체 라벨 표현 고정(frozen) 선형 평가에서 78.17%로, 자연 이미지로 사전학습한 DINOv3 ViT-S/16의 71.38%보다 높다. [원문 §6.3·Table 3](https://arxiv.org/pdf/2511.08544v3#page=42) **리뷰 해석.** 작은 도메인 모델이 대규모 전이 모델을 앞선다는 주장은 가장 좋은 LeJEPA 구조 기준으로 1-shot부터 전체 라벨까지 성립하지만, frozen 1-shot 차이는 0.91점(31.08 대 30.17)에 그치고 LeViT-128은 전체 라벨에서도 DINOv3보다 낮아 구조에 따라 갈린다.

이 글은 원문 절 순서를 따르는 **[2/3] 전체 해설**이며, [\[1/3\] 입문](/reviews/lejepa-overview/)에서 이어지고 정리의 중간 계산은 [\[3/3\] 정리·증명 심층](/reviews/lejepa-technical/)에 있다.

대표 그림의 두 손실은 표본을 모으는 축이 다르다. 왼쪽 아래는 한 이미지 안의 뷰 합의, 오른쪽 아래는 각 뷰에서 이미지들 사이의 분포 정규화다. 가우시안 제약은 투영 헤드(projector) 뒤의 공간에 적용한다. [대표 그림 확대](/assets/reviews/lejepa/lejepa-core.svg)

<span id="1-introduction-저자가-풀려는-문제와-전체-경로" aria-hidden="true"></span>

## 1. 도입 (Introduction)

자기지도학습(self-supervised learning)은 정답 라벨 대신 입력들 사이의 관계로 특징을 배운다. 이미지에서는 같은 원본에서 만든 crop·augmentation을 연결한다. 하지만 표현 거리만 줄이면 상수 출력도 해가 된다. 표현 붕괴(representation collapse)를 피하면서 다양한 데이터·구조에 적용하려면 학습 목적 자체를 명확히 해야 한다.

저자들은 경험적으로 유용한 장치들이 늘어나면서 재설정 비용이 커지고, 붕괴 방지 이론만으로는 어떤 표현 기하가 좋은지 결정하기 어렵다고 지적한다. 제안의 경로는 다음과 같다. §2에서 결합 임베딩 예측 구조(Joint-Embedding Predictive Architecture, JEPA)의 요구를 정하고, §3에서 후속 예측을 통해 목표 분포를 고른다. §4는 고차원 분포 일치를 계산하고, §5는 학습 손실을 구성한다. §6은 안정성·손실 신호·도메인·규모·의미 구조를 평가한다. [§1, pp. 1–3](https://arxiv.org/pdf/2511.08544v3#page=1)

<span id="2-background-and-notations-정의와-설계-요구" aria-hidden="true"></span>

## 2. 배경과 표기 (Background and Notations)

### 2.1 Notations and Definitions

입력 데이터의 크기는 $(N,V,D)$다. $N$은 독립 표본 수, $V$는 표본마다 만든 뷰(view) 수, $D$는 입력 차원이다. 인코더(encoder) $f_{\boldsymbol\theta}:\mathbb R^D\to\mathbb R^K$가 $K$차원 표현을 만든다. $\mathbf Z\in\mathbb R^{N\times K}$는 한 뷰에서 모은 표현 행렬이다.

JEPA의 두 요구는 관련 입력의 표현을 예측할 수 있는 **예측 가능성**(predictability)과, 표현이 입력을 구별하는 정보를 잃지 않는 **표현의 비퇴화성**(non-degeneracy)이다. 원문은 별도 예측기(predictor)가 뷰 사이 정보가 비대칭일 때, 예컨대 로봇 데이터에서 관측한 행동으로 예측을 조건화할 때만 정당화된다고 본다. 이 논문의 실험은 같은 이미지의 대칭적 뷰 합의를 쓰는 이미지 표현 학습이다. [§2.1, p. 4](https://arxiv.org/pdf/2511.08544v3#page=4)

### 2.2 The Need for Reliable Pretraining

사전학습법은 특정 ViT나 대규모 자연 이미지뿐 아니라 작은 데이터와 convolution·hybrid 구조에서도 안정적으로 적용할 수 있어야 한다. 설정마다 teacher 갱신·온도·prototype 등의 상호작용을 다시 탐색해야 한다면 전문 도메인에서 활용하기 어렵다. 이 문제의식이 §6.1과 §6.3의 실험 설계로 이어진다. [§2.2](https://arxiv.org/pdf/2511.08544v3#page=4)

### 2.3 The Need for Actionable Theory

붕괴하지 않는다는 조건은 필요한 출발점이지만, 후속 과제에 어떤 표현이 유리한지까지 정하지는 않는다. 저자들은 특정 라벨을 알기 전에 표현의 분포를 선택할 수 있는 이론을 요구하고, §§3–5에서 최적성의 필요조건으로부터 설계를 도출하겠다고 밝힌다. [§2.3](https://arxiv.org/pdf/2511.08544v3#page=4)

<span id="3-latent-euclidean-embeddings-should-be-isotropic-gaussian" aria-hidden="true"></span>

## 3. 잠재 유클리드 공간: 표현은 등방성 가우시안이어야 한다 (Latent Euclidean — Embeddings Should Be Isotropic Gaussian)

### 3.1 Linear Probing

표현을 고정하고 라벨 $\mathbf y$를 선형 예측하는 문제부터 시작한다. 원문의 최소제곱법(ordinary least squares, OLS) 표기는 리지 정규화(ridge regularization)까지 포함한다.

$$
\hat\beta=\underset{\beta\in\mathbb R^K}{\arg\min}
\|\mathbf y-\mathbf Z\beta\|_2^2+\lambda\|\beta\|_2^2.
$$

$\mathbf Z_{\rm iso}$와 $\mathbf Z_{\rm aniso}$는 같은 열공간(column span)과 총 에너지를 갖고 공분산 모양만 다르게 비교한다. 고정 설계(fixed design), 열들이 선형 독립인 조건(full column rank), 평균 0·동일 분산의 잡음 등 Appendix B.1–B.2의 조건을 사용한다.

**Lemma 1 — Anisotropy amplifies bias.** $\lambda>0$일 때, 비등방성(anisotropy) 표현에서는 등방성(isotropy)보다 계수 추정 편향(bias)이 커지는 후속 과제가 **존재한다**. 작은 고윳값(eigenvalue) 방향으로 실제 과제 계수를 잡으면 ridge의 수축(shrinkage)이 더 크기 때문이다.

**Lemma 2 — Anisotropy amplifies variance.** $\lambda=0$에서 계수 분산(variance)의 합은 등방성에서 최소다. $\lambda_k$를 그람 행렬(Gram matrix) $\mathbf Z^\top\mathbf Z$의 고윳값으로 쓰면,

$$
\operatorname{tr}\operatorname{Var}(\hat{\boldsymbol\beta}\mid\mathbf Z)
=\sigma^2\sum_{k=1}^K\frac1{\lambda_k}.
$$

총 에너지가 같을 때 역수 합은 모든 고윳값이 같으면 최소다. 설명용 예제 $(1,1)$은 역수 합 2, $(0.2,1.8)$은 약 5.56이다. 증명의 ridge 전개와 Jensen 부등식, 계수 분산과 예측 위험의 차이는 [심층 편 §3.1](/reviews/lejepa-technical/#31-linear-probing-lemmas-12와-appendix-b1b2)에 있다. [§3.1, p. 5; Appendix B.1–B.2, pp. 26–27](https://arxiv.org/pdf/2511.08544v3#page=26)

### 3.2 Nonlinear Probing

등방성 공분산(isotropic covariance)을 갖는 분포는 많다. 가우시안이라는 모양을 선택할 근거를 얻기 위해 저자들은 반경 기반 최근접 이웃(k-nearest neighbors, kNN)과 Nadaraya–Watson 커널 회귀(kernel regression)를 분석한다.

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

$\mathbf q$는 질의 표현, $r_0$는 이웃 반경, $h$는 커널 대역폭(kernel bandwidth)이다. 여기의 kernel 함수 $K$와 표현 차원 $K$는 원문이 재사용하는 기호다. 반경 kNN은 고정 개수가 아니라 반경 $r_0$ 안의 이웃을 평균한다.

**Theorem 1 — Isotropic Gaussian Optimality.** 저자들은 적분 제곱 편향(integrated squared bias, ISB)을 피셔 정보(Fisher information) $J(p)$로 나타내고, 스칼라 공분산 제약 아래 등방성 가우시안이 ISB의 유일한 최소화 분포라고 진술한다.

$$
\begin{aligned}
\mathrm{ISB}_{k\text{-NN}}&=\frac{r_0^4}{(K+2)^2}\tau_g^2J(p)+O(r_0^4),\\
\mathrm{ISB}_{\rm kernel}&\le
\left(\frac{h^2\mu_2(K)}2\right)^2
\left(2B^2+8L^2J(p)\right)+o(h^4),\\
J(p)&=\int\|\nabla\log p(\mathbf z)\|_2^2p(\mathbf z)\,d\mathbf z.
\end{aligned}
$$

$p$는 표현의 밀도, $\eta$는 표현에서 라벨로 가는 과제 함수(target function)다. $\tau_g^2$는 과제 기울기의 등방성 사전 가정 $\mathbb E[\nabla\eta\nabla\eta^\top]=\tau_g^2I$의 크기, $\mu_2(K)$는 kernel의 2차 모멘트 $\int\mathbf u\mathbf u^\top K(\mathbf u)\,d\mathbf u=\mu_2(K)I$다. 밀도는 충분히 매끄럽고 피셔 정보·공분산(covariance)이 유한해야 하며, query는 학습 표현과 같은 밀도를 따른다. kNN은 작은 반경을, kernel은 작은 $h$·충분한 국소 표본·대칭 정규화 kernel과 $\|\nabla\eta\|\le L$, $|\Delta\eta|\le B$를 가정한다. $B$는 곡률 상한이며 §5의 batch 기호와 다르다. [Appendix A, Lemmas 4–5; Theorems 7–8](https://arxiv.org/pdf/2511.08544v3#page=24)

고정 공분산 $\Sigma$에서 $J(p)\ge\operatorname{tr}(\Sigma^{-1})$이고 등호는 가우시안에서 성립한다. 총 분산까지 고정하면 등방성 $\Sigma$가 선택된다. 이 연결이 가우시안 목표의 수학적 중심이다. [§3.2; Appendix B.4](https://arxiv.org/pdf/2511.08544v3#page=28) **리뷰 해석.** kNN 식의 $O(r_0^4)$ 잔차는 주항과 같은 차수이고 kernel 식은 상한이어서, '유일한 최소화'는 정확한 ISB가 아니라 $J(p)$ 항에 대한 결론이다. Taylor 전개와 피셔 정보 하한의 계산은 [심층 편 §3.2](/reviews/lejepa-technical/#32-nonlinear-probing-theorem-1과-appendix-a-b3b7)에 있다.

### 3.3 Geometric and Practical Insights

저자들은 같은 총 분산의 분포에서 추정 계수와 결정 경계가 변하는 설명용 실험을 배치한다. Figure 3은 두 클래스, 표본 100개에서 logistic regression을 반복해 등방성·비등방성 결정 경계를 비교한다. Appendix의 Figure 18도 정규화와 bias를 시각화한다. [§3.3, Figures 3 and 18](https://arxiv.org/pdf/2511.08544v3#page=6)

<span id="4-sigreg-reliable-isotropic-gaussian-regularization-in-high-dimension" aria-hidden="true"></span>

## 4. SIGReg: 고차원의 신뢰할 수 있는 등방성 가우시안 정규화 (SIGReg — Reliable Isotropic Gaussian Regularization in High-Dimension)

### 4.1 Hypothesis Testing as a Judge

고차원 밀도를 직접 추정하지 않고 방향별 1D 법칙을 비교한다. 표현 법칙을 $P_{\boldsymbol\theta}$, 목표를 $Q=\mathcal N(0,I)$라 한다. **Lemma 3 — Hyperspherical Cramér–Wold**는 모든 unit 방향에서 두 투영 법칙이 같음과 원래 법칙의 같음을 연결한다. $\mathbf t=r\mathbf a$로 특성함수(characteristic function, CF)를 비교하는 증명이 Appendix B.8에 있다.

**Theorem 2 — Sufficiency of directional tests**는 방향별 검정을 결합한 일관성(consistency)을 논한다. Appendix B.9는 방향 집합의 mesh가 0으로 가는 조건, 전체 귀무가설(null hypothesis)에 맞춘 임계값(threshold), 분포를 구별하는 방향 주변에서의 검정력(statistical power)을 요구한다. 이 정리의 검정은 방향별 통계의 최댓값을 쓰고, 학습에 쓰는 SIGReg는 아래처럼 방향 평균이다.

$$
\operatorname{SIGReg}(\{\mathbf z_n\}_{n=1}^N)
=\frac1{|\mathcal A|}\sum_{\mathbf a\in\mathcal A}
T(\{\mathbf a^\top\mathbf z_n\}_{n=1}^N).
$$

이 식은 Definition 2의 방향 평균이다. 통계 $T$를 고르는 문제가 다음 §4.2다. [§4.1, Lemma 3, Theorem 2, Definition 2; Appendix B.8–B.9](https://arxiv.org/pdf/2511.08544v3#page=6)

![Figure 2: 입력·표현 공간·방향별 밀도 비교.](/assets/reviews/lejepa/paper-figure-2.webp)

**Figure 2 읽기.** 가운데 색 화살표마다 오른쪽의 투영 밀도를 얻는다. 가로축은 투영 좌표, 검은 곡선은 가우시안 목표다. 회색 빗금은 투영 밀도와 목표의 차이(error)다. [확대](/assets/reviews/lejepa/paper-figure-2.webp) · [Figure 2, p. 3](https://arxiv.org/pdf/2511.08544v3#page=3). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

### 4.2 SIGReg: Sketching the Epps–Pulley Test Is Stable and Scalable

#### 4.2.1 Moments Are Unstable and Insufficient

**Theorem 3**은 유한 차수까지 moment가 같은 서로 다른 분포를 구성한다. 평균·분산·공분산을 맞추는 것만으로 결합 가우시안을 보장하지 못한다. 차수를 높여도 꼬리 표본에서 다항식 gradient가 커지는 문제가 있다. Appendix B.11의 선형대수 구성은 [심층 편 §4.2](/reviews/lejepa-technical/#42-sigreg-sketching-the-eppspulley-test)에서 설명한다. [§4.2.1; Appendix B.11](https://arxiv.org/pdf/2511.08544v3#page=7)

![Figure 5: 좌표 주변 분포와 공분산이 목표에 맞아도 결합 분포는 X자일 수 있다.](/assets/reviews/lejepa/paper-figure-5.webp)

**Figure 5 읽기.** 좌표별 밀도는 가우시안이고 공분산은 identity지만 결합 표본은 X자 구조다. 10개 방향 $\mathbf a_i$로 투영하면 사선 방향에서 가우시안과 다른 밀도가 나타나고, 가장 오른쪽 패널의 검정 통계도 사선 방향(i:2, i:7)에서 크게 반응한다. [확대](/assets/reviews/lejepa/paper-figure-5.webp) · [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

#### 4.2.2 Cumulative Density Functions Are Impractical

누적분포함수(cumulative distribution function, CDF)는 한 임계값 이하의 확률을 잰다. 원문은 정렬이 다중 GPU에서 동기화를 요구해 SGD의 병렬성을 깨고, 정렬·순서통계(order statistics)가 미분 불가능한 연산이라 완화(relaxation) 없이는 경사 학습에 맞지 않으며, sketching 해법은 하이퍼파라미터를 늘린다고 평가한다. [§4.2.2, p. 9](https://arxiv.org/pdf/2511.08544v3#page=9) **리뷰 해석.** 정렬된 값은 동률이 없는 점에서 입력의 조각별 선형 함수라 기울기가 정의되므로, 실제 부담은 미분 가능성보다 동기화와 추가 하이퍼파라미터 쪽이다.

#### 4.2.3 Characteristic Functions Are Stable, Scalable and Identifiable

특성함수는 복소 지수함수의 기대값이다. 아래 방향 첨자 표기는 본문의 $\hat\phi_X$, $\phi$를 해당 투영과 가우시안 목표에 연결한 것이다. 표본은 평균으로 계산하고 가우시안 목표는 닫힌 식을 사용한다.

$$
\hat\varphi_{\mathbf a}(t)=\frac1N\sum_{n=1}^N e^{it\mathbf a^\top\mathbf z_n},
\qquad \varphi_{\mathcal N}(t)=e^{-t^2/2}.
$$

$$
EP(\mathbf a)=N\int_{\mathbb R}
|\hat\varphi_{\mathbf a}(t)-\varphi_{\mathcal N}(t)|^2w(t)\,dt.
$$

원문의 Epps–Pulley 식은 가우시안 가중 $w(t)$를 사용한다. **Theorem 4 — Stability of Epps–Pulley Test**는 $|e^{itz}|=1$이라 입력 크기와 관계없이 CF와 그 미분이 유계라는 사실로, 투영 표본에 대한 $EP$의 1·2차 미분이 $O(1/N)$로 유계라고 진술한다. [§4.2.3, Epps–Pulley 식, Theorem 4; Appendix B.12](https://arxiv.org/pdf/2511.08544v3#page=9) **리뷰 해석.** Appendix B.12의 유도는 $N$배를 하지 않은 거리 $EP/N$에 대한 것이라, $EP$ 자체의 기울기 상한은 $N$배 크고 유계성만 유지된다. 계산은 [심층 편 §4.2.3](/reviews/lejepa-technical/#423-cf-theorem-4의-유도와-정규화)에 있다.

**Algorithm 1을 읽는 순서.** $K\times M$ 가우시안 행렬의 열을 unit 길이로 정규화해 방향을 만든다. $\mathbf Z\mathbf A$를 주파수와 곱해 표본 특성함수(empirical characteristic function, ECF)를 계산한다. step seed를 맞춰 GPU마다 같은 방향을 사용하고 CF 평균을 all-reduce한다. $[-5,5]$의 17개 grid에서 가우시안 CF와의 오차를 가우시안 가중해 trapezoidal 적분한 뒤 전체 표본 수 $N$을 곱한다. 각 방향 통계는 SIGReg에서 평균한다. [Algorithm 1, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10)

방향 수 $M$과 주파수 점 수 $n_t$(리뷰 보조 기호)를 쓰면 dense 계산 비용은 $\mathcal O(NKM+NMn_t)$로, $K,M,n_t$를 고정할 때 $N$에 선형이다. 원문 Table 6은 V100-SXM2-16GB에서 SIGReg 손실 계산 시간을 10회 측정(평균 ± 표준편차)한다. $(N,M,n_t)=(512,512,16)$은 $0.465236\pm0.011642$ ms, $(8192,512,16)$은 $6.188304\pm0.007226$ ms다. embedding 차원은 표에 없다. [Table 6, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

### 4.3 How SIGReg Beats the Curse of Dimensionality

원문은 **매끄러움**(smoothness), **SGD 방향 재표집**(resampling), **합성 데이터 검증**(synthetic validation)의 순서로 설명한다. Theorem 5의 상한은 방향 수 $M$에 대해 $M^{-2\alpha/(K-1)}$로 줄고, $\alpha$는 소볼레프 정규성(Sobolev regularity)이다. 저자들은 이로부터 $\alpha$가 크면 $O(K)$개 방향으로 충분하다고 쓰며, 상수 $C(K,\alpha)$의 차원 의존성을 포함한 검토는 [심층 편 §4.3](/reviews/lejepa-technical/#43-how-sigreg-beats-the-curse-of-dimensionality-theorem-5)에 있다. 저자들은 또 방향을 step마다 다시 뽑으면 누적 방향 수가 학습 시간에 비례해 늘어, $M=16$으로도 수천 개의 고정 방향을 앞선다고 보고한다. [§4.3, Theorem 5; Appendix B.10](https://arxiv.org/pdf/2511.08544v3#page=10)

![Figure 7: 방향 수에 따른 최종 directional statistic, 고정 방향과 재표집 비교.](/assets/reviews/lejepa/paper-figure-7.webp)

**Figure 7 읽기.** 가로축은 step마다 쓰는 방향 수 $M$, 세로축은 합성 최적화를 마친 뒤의 기대 directional statistic이며 낮을수록 목표 분포에 가깝다. 초록 재표집 방향은 파란 고정 방향보다 적은 $M$에서도 낮은 값에 이른다. [확대](/assets/reviews/lejepa/paper-figure-7.webp) · [Figure 7, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

이어 합성 실험은 대부분의 차원은 가우시안이고 두 차원만 X자 구조인 표본을 인코더 없이 **직접 최적화**해 손실 자체의 효과를 본다. §4.3 본문은 이 실험(Figure 6)을 $D=512$, $M=16$ 사례로 소개하지만, Figure 6 캡션의 설정은 $N=100$, $D=1024$, $M=10$이고 부록의 추가 그림에도 $D=512$ 설정은 없다. [§4.3, p. 11](https://arxiv.org/pdf/2511.08544v3#page=11) · [Figure 6, p. 9](https://arxiv.org/pdf/2511.08544v3#page=9)

<span id="5-lejepa-stable-and-scalable-implementation" aria-hidden="true"></span>

## 5. LeJEPA: 안정적이고 확장 가능한 구현 (LeJEPA — Stable and Scalable Implementation)

### 5.1 LeJEPA: SIGReg + Prediction Loss

한 이미지에 전역 뷰(global view) $V_g$개와 지역 뷰(local view) $V_l$개, 총 $V=V_g+V_l$개를 만든다. 각 뷰를 인코더와 투영 헤드로 처리한다. $\boldsymbol\mu_n$은 전역 뷰 표현의 평균이다. Algorithm 2의 실제 중심 기반 손실을 설명하면 다음과 같다.

$$
\begin{aligned}
\boldsymbol\mu_n&=\frac1{V_g}\sum_{v=1}^{V_g}\mathbf z_{n,v},\\
\mathcal L_{\rm pred}^{\rm code}&=\frac1{NVK}\sum_{n=1}^N\sum_{v=1}^V
\|\mathbf z_{n,v}-\boldsymbol\mu_n\|_2^2,\\
\mathcal L_{\rm SIGReg}&=\frac1V\sum_{v=1}^V\operatorname{SIGReg}(\{\mathbf z_{n,v}\}_{n=1}^N),\\
\mathcal L_{\rm LeJEPA}&=(1-\lambda)\mathcal L_{\rm pred}^{\rm code}+\lambda\mathcal L_{\rm SIGReg}.
\end{aligned}
$$

$\mathcal L_{\rm pred}^{\rm code}$와 $\mathcal L_{\rm SIGReg}$는 리뷰의 약칭이며 $1/K$는 코드의 feature 평균을 표시한다. 원문 총 손실의 batch 기호 $B$가 여기의 $N$에 해당한다. Algorithm 2 캡션은 ResNet 같은 비ViT 구조에서는 모든 뷰를 전역 뷰로 두라고 하므로($V_g=V$), 그 경우 $\boldsymbol\mu_n$은 모든 뷰의 평균이다. [§5.1, Algorithm 2](https://arxiv.org/pdf/2511.08544v3#page=11)

원문은 Eq. (5)의 쌍별 평균 손실이 Eq. (6)의 중심 손실과 같다고 쓴다(Appendix B.6). **리뷰 해석.** 이 등식에는 전역 뷰들 사이의 분산 항이 빠져 있어, 실험이 최적화한 것은 Eq. (6)의 중심 손실이다. 전개와 반례는 [심층 편 §5.1](/reviews/lejepa-technical/#51-sigreg-prediction-loss-theorem-6-appendix-b6b13)에 있다.

저자들은 프로토타입(prototype)·기울기 차단(stop-gradient)·교사–학생(teacher–student) 네트워크가 없어 손실의 하이퍼파라미터가 $\lambda$ 하나라고 쓴다. 투영 헤드는 사용하고, 평가는 투영 헤드 앞의 특징 추출기(backbone) 특징으로 한다. 모델 정의·optimizer·data loader를 빼면 구현은 수십 줄이다.

### 5.2 Relation to Prior Work

원문은 JEPA에 슬라이싱과 분포 일치를 쓴 선행 해법은 없지만 생성 모델·최적 수송에는 비슷한 방법이 있다고 보고, sliced score matching과 sliced Wasserstein을 예로 든다. Epps–Pulley 적분을 구적 대신 정확히 계산하면 각 슬라이스 손실은 kernel MMD가 되지만 계산량이 이차다. 통계 $T=\operatorname{mean}^2+(\operatorname{std}-1)^2$를 쓰면 슬라이스가 많은 극한에서 VICReg를 회복하며(Appendix B.14), 저자들은 Theorem 3에 따라 이런 moment 통계가 지름길 해를 허용한다고 경고한다. [§5.2; Appendix B.14, pp. 39–40](https://arxiv.org/pdf/2511.08544v3#page=12) VICReg 손실과의 정확한 대응은 [심층 편 §5.2](/reviews/lejepa-technical/#52-relation-to-prior-work-appendix-b14)에서 다룬다.

### 리뷰 보충: 주요 원전의 방법 비교

| 원전·읽은 범위 | 방법의 핵심 | LeJEPA와의 차이 |
| --- | --- | --- |
| [VICReg, §§3.1–3.3, pp. 3–4](https://arxiv.org/pdf/2105.04906v3#page=3) | 뷰 합의, 표준편차 하한, 비대각 공분산 penalty | 대칭적 붕괴 방지는 선행한다. SIGReg는 유한 차수 통계보다 전체 투영 법칙을 겨냥한다. |
| [DINO, §3, pp. 2–3](https://arxiv.org/pdf/2104.14294v2#page=2) | EMA teacher, centering·sharpening, 출력 확률 일치 | LeJEPA는 교사 없이 명시적 가우시안 목표와 대칭 합의를 쓴다. |
| [I-JEPA, §3, pp. 3–4](https://arxiv.org/pdf/2301.08243v3#page=3) | context에서 target block 표현을 predictor로 예측 | 비대칭 정보 예측과 대칭 crop 합의의 과제가 다르다. |
| [Gaussian Embeddings, pp. 1, 3–4](https://arxiv.org/pdf/2510.05949v1#page=1) | 가우시안 밀도와 표현 변환·Jacobian 부피 | LeJEPA의 가우시안 목표는 후속 추정 오차 분석과 sliced CF 계산에서 나온다. |
| [Gretton et al., MMD, PDF pp. 1, 3–4](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) | kernel 평균 임베딩을 통한 분포 비교 | 정확 적분 연결의 배경. Gretton 등은 선형 시간 MMD 추정기도 제시한다. |

<span id="6-lejepa-empirical-validation" aria-hidden="true"></span>

## 6. LeJEPA의 실험 검증 (LeJEPA: Empirical Validation)

실험을 원문의 §6.1→§6.5 순서로 읽는다. 수치는 **논문 보고**, 정확도 단위는 %, 높을수록 좋다. frozen 평가는 인코더를 고정하고 라벨로 선형 분류기만 학습하며, 미세조정(finetuning)은 backbone도 수정한다.

### 6.1 LeJEPA’s Stability Across Hyper-Parameters and Architectures

첫 질문은 설정과 구조를 바꿔도 유용하게 학습되는가다. Table 1의 조건은 **ImageNet-1K, ViT-Large/14, 100 epoch, frozen linear top-1**이다.

| 변경 축 | 원문 대표 결과 | 읽을 점 |
| --- | --- | --- |
| 적분 범위, 17개 grid, $M=512$ | $[-1,1]$ 72.13 → $[-5,5]$ 74.21 | 구적점 수(5·17·41)를 바꾼 차이는 0.6점 이내다. |
| batch | 128: 72.20; 256: 74.15; 512: 74.72; 1024: 74.07 | 128에서도 학습되며 최고는 512다. |
| 전역 뷰, 전체 4개 | $V_g=1$: 53.06; $V_g=2$: 72.26 | 표에서 가장 큰 변화(19.20점)다. |
| 전역 뷰, 전체 10개 | $V_g=2$: 74.06; $V_g=4$: 75.08 | 권장 구성($V_g=2$, $V_l=8$)보다 $V_g=4$가 1.02점 높다. |
| register, $M=4096$ | 0개: 75.61; 8개: 75.84 | 차이는 0.23점이다. |

[Table 1, p. 13](https://arxiv.org/pdf/2511.08544v3#page=13). 저자들은 특성함수가 0 근처에서 정확해 좁은 범위로도 moment가 잘 특정된다는 이유로, 적분 범위와 구적점 수가 성능에 "negligible impact"를 준다고 쓴다. **리뷰 해석.** 구적점 수에는 맞는 평가지만, 범위를 $[-1,1]$에서 $[-5,5]$로 넓히면 방향 512·2048개 모두에서 2.1–2.5점 올라 batch를 128에서 512로 늘린 효과(2.52점)와 비슷하다.

원문이 권장하는 출발점은 $\lambda=0.05$, 전역 2·지역 8뷰, batch $\ge128$, 방향 1024개, 17점 $[-5,5]$다. 그러나 모든 실험에 적용한다는 공통 설정(Experiment Details 1)은 전역 2뷰를 포함한 **총 8뷰**(지역 6뷰), AdamW, warmup·cosine 학습률 schedule과 learning rate·weight decay 후보다. [§6.1, Experiment Details 1](https://arxiv.org/pdf/2511.08544v3#page=14)

Figure 9는 IN-10에서 20M 미만의 약 50개 timm 모델, 8개 구조 계열을 frozen 평가하며, 저자들은 모두 top-1 91.5–95%에 이른다고 보고한다. Figure 8과 Table 7(IN-100, ResNet50, 400 epoch)에서 저자들은 $\lambda$와 뷰 수에 대해 성능이 안정적이라고 본다. 8뷰 기준 $\lambda=0.010$–$0.150$은 83.07–84.32이고, 범위 끝의 $\lambda=0.001$은 76.67, $0.500$은 77.72다. [§6.1, Figures 8·9; Table 7, p. 43](https://arxiv.org/pdf/2511.08544v3#page=14)

추가 ablation인 Table 4(IN-100, 400 epoch)는 투영 헤드 깊이, 예측기, 확률적 가중치 평균(stochastic weight averaging, SWA)을 비교한다. 원문의 SWA는 중심 $\boldsymbol\mu_n$을 만드는 인코더, 즉 교사 쪽에 적용한 가중치 평균이다. 예측기 없이 3층 투영 헤드에서 ViT-S는 SWA로 81.07→84.12가 되지만 ResNet50은 83.93→83.50으로 오르지 않는다. 저자들은 교사–학생 구성이 ViT에서 작은 이득을 줄 뿐 붕괴 방지에는 필요하지 않다고 보고, 예측기·register 없이 학습하고 ViT에서만 SWA를 선택적으로 쓰라고 권한다. [§6.1; Table 4, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42)

### 6.2 LeJEPA’s Training Loss Is Informative of Downstream Performance

두 번째 질문은 라벨을 보기 전 학습 손실이 품질에 관한 신호를 주는가다. 두 손실 항 중 하나만 최소화하면 부족하다. 합의만 강하면 붕괴할 수 있고, 가우시안만 맞추면 이미지 뷰의 공통 정보를 보장하지 못한다.

![Figure 10: prediction loss와 SIGReg loss 평면에 표시한 downstream 정확도.](/assets/reviews/lejepa/paper-figure-10.webp)

**Figure 10 읽기.** 가로축은 SIGReg 손실, 세로축은 prediction 손실(둘 다 로그 축), 색은 downstream 정확도이며 파랑에서 빨강으로 높아진다. 세 패널은 ResNet50·Galaxy10, ResNet50·IN-10, ViT-B/14·IN-1K다. 두 손실이 함께 낮은 왼쪽 아래로 갈수록 붉어진다. 캡션은 비슷한 정확도를 내는 점들이 서로 다른 $\lambda$에 대응하는 직선 모양의 trade-off 전선을 이루며 그 전선이 왼쪽 아래를 향한다고 설명한다. [확대](/assets/reviews/lejepa/paper-figure-10.webp) · [Figure 10, p. 15](https://arxiv.org/pdf/2511.08544v3#page=15). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

Figure 11은 학습 손실을 $\lambda^\alpha$로 나눈 값과 downstream 정확도의 스피어만 순위상관(Spearman rank correlation) $C^{(\alpha)}$를 $\alpha$별로 그린다. 저자들은 보정 없는 손실($\alpha=0$)에서 약 85%, $\alpha\approx0.4$에서 약 99%를 보고하고, 이를 라벨 없는 모델 선택의 근거로 든다. [§6.2, Figures 10–11, 추가 Figures 19·21](https://arxiv.org/pdf/2511.08544v3#page=15)

**상관 지표를 읽는 기준.** Spearman 계수는 두 관측값 목록 $A,B$(리뷰 보조 기호)를 각각 순위로 바꾼 뒤 계산한 Pearson 상관이다.

$$
\rho_{\mathrm S}=\operatorname{corr}\!\left(\operatorname{rank}(A),\operatorname{rank}(B)\right).
$$

범위는 $[-1,1]$이고 절댓값이 클수록 단조 관계가 강하다. **리뷰 해석.** 손실이 낮을수록 정확도가 높다면 계수는 음수여야 하므로, 원문의 85%·99%는 부호를 뒤집거나 절댓값으로 나타낸 단조 관계의 강도로 보인다.

### 6.3 In-Domain LeJEPA Outperforms Frontier Model Transfer Learning

세 번째 질문은 일반 이미지 기반 모델(foundation model)을 가져오는 대신 전문 도메인에서 작은 모델을 직접 사전학습할 가치가 있는가다. Galaxy10은 약 11,000개 학습 이미지와 10개 은하 형태 클래스다. LeJEPA는 기본 pipeline으로 **400 epoch** 도메인 내 사전학습(in-domain pretraining)을 하고, 비교 대상은 LVD-142M·LVD-1.7B 자연 이미지로 사전학습한 DINOv2/v3 ViT-S다.

![Figure 12: Galaxy10의 라벨 수별 full finetuning과 frozen 평가.](/assets/reviews/lejepa/paper-figure-12.webp)

**Figure 12 읽기.** 가로축은 후속 분류에서 클래스당 사용하는 라벨 수, 세로축은 정확도다. 왼쪽은 full finetuning, 오른쪽은 frozen 평가이며 주황 계열이 LeJEPA, 파랑 계열이 DINOv2/v3다. 미세조정에서는 LeJEPA ConvNeXtV2-Nano와 DINOv3의 차이가 클래스당 5–10개에서 13–15점으로 가장 크고, 전체 라벨에서는 1.12점으로 줄어든다. frozen에서는 1–2개에서 곡선이 겹치다가 5개부터 ResNet-34가 DINOv3보다 6.5–9.1점 높은 상태를 유지한다. 오차막대는 3개 seed에서 나왔지만 표준편차인지 신뢰구간인지 원문에 없다. [확대](/assets/reviews/lejepa/paper-figure-12.webp) · [Figure 12, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

| Table 3 모델 | 전체 라벨 frozen | 전체 라벨 finetuning | 1-shot frozen |
| --- | --- | --- | --- |
| LeJEPA ConvNeXtV2-Nano | 76.52 | 82.72 | 28.74 |
| LeJEPA ResNet-34 | 78.17 | 83.28 | 31.08 |
| LeJEPA LeViT-128 | 69.00 | 79.41 | 25.85 |
| DINOv2 Small | 67.62 | 78.34 | 27.68 |
| DINOv3 ViT-S/16 | 71.38 | 81.60 | 30.17 |

[Table 3, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42). 저자들은 도메인 내 사전학습이 DINOv2·DINOv3를 frozen과 미세조정 모두에서 "substantially" 앞선다고 쓴다. frozen에서 ResNet-34는 1-shot(31.08 대 30.17)부터 전체 라벨까지 DINOv3보다 높지만, LeViT-128은 전체 라벨에서도 69.00으로 DINOv3(71.38)보다 낮다.

Table 5는 flowers102의 1,020개 학습 표본 등 작은 데이터로 넓힌 frozen 평가다. flowers102에서 LeJEPA ResNet18은 74.57, ResNeXt26ts는 82.19로 I-JEPA ViT-H/14의 85.76보다 낮고, Galaxy10에서는 ResNet34 77.29가 I-JEPA 62.93보다 높다. Table 5의 I-JEPA 행 이름은 IN-22K인데 사전학습 칸은 IN-1K이고, Galaxy10 ResNet-34 frozen 값도 Table 3(78.17)과 Table 5(77.29)가 다르다. [Table 5, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

### 6.4 LeJEPA Scales Across Data and Models

IN-1K에서 학습한 큰 모델의 온라인 선형 프로브(online linear probe) 정확도는 ViT-L(0.3B) 77.1%, ConvNeXtV2-H(0.6B) 78.5%다. 전이 평가는 backbone을 고정하고 8개 데이터셋에서 분류기를 학습하며, 비교 기준은 설정이 가장 가까운 I-JEPA ViT-H(0.6B)와 확률적 예측 과제를 추가한 개선판 I-JEPA + STOP이다. [§6.4, p. 17](https://arxiv.org/pdf/2511.08544v3#page=17) **리뷰 해석.** 초록은 frozen 선형 평가에서 ViT-H/14가 79%에 이른다고 쓰지만, 본문과 부록에는 LeJEPA ViT-H 결과를 담은 표나 그림이 없고 보고된 가장 큰 IN-1K 선형 평가 값은 ConvNeXtV2-H의 78.5%다.

| Table 2 모델·조건 | 1-shot 평균 | 10-shot 평균 | 전체 라벨 평균 |
| --- | --- | --- | --- |
| LeJEPA ViT-L, 304M, IN-1K, 100 epoch | 29.55 | 60.95 | 79.48 |
| LeJEPA ConvNeXtV2-H, 660M, IN-1K, 100 epoch | 31.58 | 60.70 | 77.56 |
| I-JEPA ViT-H, 632M, IN-1K, 300 epoch | 30.20 | 60.51 | 78.50 |
| I-JEPA ViT-H + STOP, 같은 크기·데이터·epoch | 32.05 | 62.92 | 80.70 |
| I-JEPA ViT-H, IN-22K, 900 epoch | 35.15 | 63.28 | 80.82 |

[Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16). 저자들은 더 작은 모델과 짧은 학습으로 I-JEPA를 일관되게(consistently) 앞선다고 쓴다. 더 작은 모델이라는 조건은 ViT-L(304M)에만 맞고 ConvNeXtV2-H(660M)는 I-JEPA(632M)보다 크다. 평균으로 보면 LeJEPA 두 모델은 10-shot에서 모두 높지만, ViT-L의 1-shot(29.55 대 30.20)과 ConvNeXtV2-H의 전체 라벨(77.56 대 78.50)은 낮다. 과제별 전체 라벨 결과는 DTD·flowers102·food101에서 높고 CIFAR-10·CIFAR-100·pets에서 낮다. I-JEPA + STOP은 세 평균 모두에서 LeJEPA 두 모델보다 높고, IN-22K·900 epoch 행은 별도 예산군이다.

Table 2 캡션은 100 대 300 epoch를 학습 시간과 계산 자원의 "3× reduction"으로 표현하고, 저자들은 ViT-gigantic(1.8B)의 매끄러운 학습 손실 곡선(Figure 1)을 최적화 안정성의 예로 든다. [§6.4; Figure 1](https://arxiv.org/pdf/2511.08544v3#page=1) **리뷰 해석.** 두 모델은 크기(304M 대 632M)와 구조가 달라 epoch 비가 계산량 비와 같지 않고, 원문은 wall-clock이나 FLOPs를 보고하지 않는다.

### 6.5 Emergent Semantic Structure in LeJEPA Representations

Figure 13은 [CLS] attention을 threshold해 영상 프레임의 전경(foreground)을 드러낸다. Figure 14는 ViT-L, IN-1K, 100 epoch의 패치(patch) 특징을 주성분분석(principal component analysis, PCA)의 상위 세 성분으로 RGB에 표시한다. 저자들은 따뜻한 색이 전경 물체를, 차가운 색이 배경을 일관되게 잡는다고 설명하고, attention 마스크가 프레임 사이에서 물체를 추적한다는 점을 시간적 구조의 증거로 든다. 원문 전체 패널과 영상은 [Figures 13–14](https://arxiv.org/pdf/2511.08544v3#page=17)와 [공식 프로젝트 페이지](https://rbalestr-lab.github.io/lejepa/)에 있다.

**리뷰 해석.** PCA를 **이미지마다 따로** 수행하므로 이미지 사이의 색 일치는 같은 특징 방향을 뜻하지 않고, IoU 같은 분할 지표도 보고되지 않아 이 결과는 정성적 근거에 머문다.

<span id="7-conclusion-저자의-결론과-적용-범위" aria-hidden="true"></span>

## 7. 결론 (Conclusion)

저자들은 등방성 가우시안 표현이 최악 경우의 후속 위험을 유일하게 최소화함을 증명했고, SIGReg가 그 분포를 계산 가능하게 강제하며, 교사–학생 네트워크·기울기 차단·비대칭 구조의 임시 조합 없이 설계 차원에서 붕괴를 없앤다고 결론짓는다. 60개 이상의 구조와 1.8B 모델까지 검증했고, 50줄 미만의 핵심 구현으로 최신 성능에 맞먹는다는 것이 그들의 정리다. [§7, p. 18](https://arxiv.org/pdf/2511.08544v3#page=18)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 이 논문의 강점은 목표 분포의 선택(§3)과 그 분포를 선형 시간에 강제하는 손실(§4)을 하나의 설계로 묶고, 같은 기본 설정을 ImageNet의 여러 구조와 Galaxy10 같은 비자연 이미지 도메인에 그대로 적용해 보인 데 있다. 한계는 근거마다 다르다. 이론의 '유일 최적'은 국소 추정기 편향의 상한·주항에 대한 결론이고, Eq. (5)→(6)과 Theorems 4·6에는 누락항과 정규화 문제가 있다([\[3/3\] 심층 편](/reviews/lejepa-technical/)). 실험에서 전이 비교의 I-JEPA는 모델 크기와 학습 길이가 달라 같은 예산의 대조가 없고, Galaxy10 미세조정의 전체 라벨 우위는 1–2점이며, 손실-정확도 상관의 $\alpha$는 평가 정확도를 보고 고른 값이다. 남은 질문은 평가 라벨 없이 정한 $\alpha$로도 새 도메인에서 상관이 유지되는지, 같은 크기·epoch에서 I-JEPA와의 차이가 남는지, 예측기를 뺀 결론이 행동 조건부 예측 같은 비대칭 JEPA에도 이어지는지다.

## 출처와 읽은 범위

- 원문: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics* · Randall Balestriero et al. · 최초 공개 2025-11-11 · [검토 버전](https://arxiv.org/abs/2511.08544v3) · 검토일 2026-10-23.
- 읽은 범위: v3 본문 §1–7·실험 §6.1–6.5, 부록 C의 주요 표·D–F의 대안 설명. Appendix A·B의 긴 증명은 [3/3] 심층 편에 연결한다.
- 그림 출처: 대표 개념도는 자체 제작물(CC BY 4.0). 원문 그림은 Balestriero·LeCun의 v3 자산(CC BY-SA 4.0)으로 전체 패널 crop·무손실 WebP 변환을 적용한다. [원문](https://arxiv.org/abs/2511.08544v3).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
