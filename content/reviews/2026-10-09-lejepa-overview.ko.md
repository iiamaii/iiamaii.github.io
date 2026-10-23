---
title: "[1/3] LeJEPA 입문: 같은 이미지는 가깝게, 표현 분포는 가우시안으로"
description: "저자의 §§1–7 순서로 문제, 가우시안 목표, SIGReg, 학습과 다섯 실험 축을 짧게 따라가는 입문 지도."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-23T15:26:18+00:00"
topics: ["self-supervised learning", "representation learning", "joint embedding predictive architectures", "distribution matching", "computer vision"]
visibility: "public"
lang: "ko"
translationKey: "lejepa-overview"
paperTitle: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
paperPublishedDate: "2025-11-11"
authors: ["Randall Balestriero", "Yann LeCun"]
year: "2025"
paperUrl: "https://arxiv.org/abs/2511.08544v3"
thumbnail: "/assets/reviews/lejepa/lejepa-core.svg"
thumbnailAlt: "LeJEPA의 두 제약 설명도: 한 이미지의 뷰들은 전역 뷰 평균에 정렬하고, 각 뷰에서 여러 이미지의 표현은 임의 방향으로 투영해 표준 가우시안과 맞춘다."
---

LeJEPA는 같은 이미지의 뷰(view)를 정렬하고 표현 분포를 등방성 가우시안(isotropic Gaussian)에 맞추는 자기지도학습(self-supervised learning) 방법이다. ImageNet-1K에서 100 epoch 학습한 LeJEPA 비전 트랜스포머(vision Transformer, ViT)-L(304M)은 8개 전이 과제의 전체 라벨 표현 고정(frozen) 평가 평균이 79.48%로, 같은 데이터에서 300 epoch 학습한 I-JEPA ViT-H(632M)의 78.50%보다 높다. [원문 Table 2](https://arxiv.org/pdf/2511.08544v3#page=16) **리뷰 해석.** 저자들은 더 작은 모델과 짧은 학습으로 I-JEPA를 일관되게(consistently) 앞선다고 쓰지만, 같은 표의 평균 비교 6개(LeJEPA 두 모델 × 1-shot·10-shot·전체 라벨) 가운데 ViT-L의 1-shot(29.55 대 30.20)과 ConvNeXtV2-H의 전체 라벨(77.56 대 78.50)은 I-JEPA보다 낮아 우위는 4개에서만 성립한다.

이 글은 원문 §§1–7을 따라가는 **[1/3] 입문 지도**이며, §6 실험의 상세는 [\[2/3\] 전체 해설](/reviews/lejepa-complete/), Appendix A·B의 증명은 [\[3/3\] 정리·증명 심층](/reviews/lejepa-technical/)에서 이어진다.

대표 그림에서 왼쪽 아래 손실은 **한 이미지의 여러 뷰**를 모으고, 오른쪽 아래 손실은 **각 뷰에서 여러 이미지**의 표현 분포를 맞춘다. 가우시안 목표는 투영 헤드(projector) 뒤의 공간에 적용한다. 점과 곡선은 설명용이다. [대표 그림 확대](/assets/reviews/lejepa/lejepa-core.svg)

<span id="1-introduction-분포-목표부터-학습-방법까지" aria-hidden="true"></span>

## 1. 도입 (Introduction)

라벨 없이 사진의 특징을 배우려면 같은 사진의 다른 크롭(crop)을 연결해야 한다. 그러나 모든 입력에 같은 벡터를 출력해도 뷰 사이 거리는 작아진다. 이를 **표현 붕괴**(representation collapse)라고 한다. 일부 차원만 사용하는 차원 붕괴(dimensional collapse)도 있다.

저자들은 붕괴를 피하는 장치를 나열하기 전에 질문을 바꾼다. **나중에 어떤 과제를 풀지 모른다면, 학습한 표현은 어떤 분포를 이루는 것이 좋은가?** 답을 통계적 예측기의 분석에서 찾고, 등방성 가우시안 투영 정규화(Sketched Isotropic Gaussian Regularization, SIGReg)와 뷰 합의(view agreement)를 결합한다. [§1, pp. 1–3](https://arxiv.org/pdf/2511.08544v3#page=1)

<span id="2-background-and-notations-jepa가-요구하는-두-조건" aria-hidden="true"></span>

## 2. 배경과 표기 (Background and Notations)

### 2.1 Notations and Definitions

인코더(encoder) $f_{\boldsymbol\theta}:\mathbb R^D\to\mathbb R^K$는 입력을 $K$차원 표현으로 바꾼다. 이미지 $n$의 뷰 $v$에서 나온 표현은 $\mathbf z_{n,v}$다. 한 배치(batch)에는 독립 이미지 $N$개, 이미지마다 뷰 $V$개가 있다.

원문은 결합 임베딩 예측 구조(Joint-Embedding Predictive Architecture, JEPA)를 관련 입력 사이의 **예측 가능성**(predictability)과 표현의 **비퇴화성**(non-degeneracy)으로 정의한다. 이 논문의 이미지 구현은 대칭적 뷰 합의에 집중한다. 저자들은 별도 예측기(predictor)가 관측한 행동으로 예측을 조건화할 때처럼 뷰 사이 정보가 비대칭일 때만 정당화된다고 본다. [§2.1, p. 4](https://arxiv.org/pdf/2511.08544v3#page=4)

### 2.2–2.3 Reliable Pretraining과 Actionable Theory

좋은 표현은 여러 구조와 데이터에서 학습할 수 있어야 한다. 이론도 ‘왜 붕괴하지 않는가’뿐 아니라 실제로 **어떤 분포와 손실을 선택할 것인가**에 답해야 한다. 이 두 요구가 §3의 목표 분포와 §4의 계산 방법으로 이어진다. [§§2.2–2.3](https://arxiv.org/pdf/2511.08544v3#page=4)

<span id="3-latent-euclidean-왜-등방성-gaussian인가" aria-hidden="true"></span>

## 3. 잠재 유클리드 공간: 표현은 등방성 가우시안이어야 한다 (Latent Euclidean — Embeddings Should Be Isotropic Gaussian)

### 3.1 Linear Probing

**선형 프로브**(linear probe)는 인코더를 고정하고 그 특징 위에 작은 선형 예측기를 학습하는 평가다. 같은 총 에너지 아래 특정 방향의 변화가 너무 작으면, 그 방향을 필요로 하는 과제의 계수 추정이 불리해진다. Lemma 1은 리지 정규화(ridge regularization)에서 더 큰 편향(bias)을 겪는 과제가 **존재함**을 보이고, Lemma 2는 최소제곱법(ordinary least squares, OLS)의 계수 분산(variance) 합이 등방성에서 최소임을 보인다. [§3.1; Appendix B.1–B.2](https://arxiv.org/pdf/2511.08544v3#page=5)

### 3.2 Nonlinear Probing

등방성(isotropy)은 모든 방향의 분산이 같다는 뜻이다. 이 조건만으로 분포가 가우시안이 되지는 않는다. 저자들은 가까운 이웃의 정답을 평균하는 반경 기반 최근접 이웃(k-nearest neighbors, kNN)과 커널(kernel) 예측기도 분석한다.

작은 이웃·대역폭(bandwidth)에서의 편향에는 밀도 스코어(density score), 즉 로그 밀도의 변화율인 $\nabla\log p$가 들어간다. 그 크기의 제곱 평균인 **피셔 정보**(Fisher information) $J(p)$는 적절한 조건과 고정 공분산(covariance) 아래 가우시안에서 최소다. 공분산의 총량까지 고정하면 등방성 가우시안이 선택된다. Theorem 1의 가우시안 목표는 이런 국소 추정과 과제 가정에서 나온다. [§3.2; Appendix A, B.3–B.7](https://arxiv.org/pdf/2511.08544v3#page=5)

### 3.3 Geometric and Practical Insights

§3.3의 설명용 실험은 같은 총 분산의 등방성·비등방성(anisotropy) 표현에서 추정 계수와 결정 경계가 어떻게 달라지는지 보여준다. 실제 이미지 인코더의 성능은 §6에서 따로 평가한다. [§3.3, Figure 3](https://arxiv.org/pdf/2511.08544v3#page=6)

<span id="4-sigreg-고차원-분포를-여러-방향에서-확인하기" aria-hidden="true"></span>

## 4. SIGReg: 고차원의 신뢰할 수 있는 등방성 가우시안 정규화 (SIGReg — Reliable Isotropic Gaussian Regularization in High-Dimension)

### 4.1 Hypothesis Testing as a Judge

벡터를 길이 1의 방향 $\mathbf a$로 투영하면 스칼라 $\mathbf a^\top\mathbf z$를 얻는다. 모든 방향의 1차원 분포가 표준 가우시안이라면 원래 벡터도 표준 다변량 가우시안이다. Lemma 3은 이를 Cramér–Wold 정리와 연결한다. 실제 학습에서는 유한한 방향만 보고 반복마다 다시 뽑는다. [§4.1; Appendix B.8–B.9](https://arxiv.org/pdf/2511.08544v3#page=6)

![Figure 2: 표현을 여러 방향으로 투영하여 목표 가우시안과 비교하는 개념도.](/assets/reviews/lejepa/paper-figure-2.webp)

**Figure 2.** 표현을 색깔별 방향으로 투영해 각 투영 밀도를 목표 가우시안(검은 곡선)과 비교하는 개념도다. 읽는 법은 [전체 해설 §4.1](/reviews/lejepa-complete/#41-hypothesis-testing-as-a-judge)에 있다. [확대](/assets/reviews/lejepa/paper-figure-2.webp) · [원문 Figure 2, p. 3](https://arxiv.org/pdf/2511.08544v3#page=3). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

### 4.2 SIGReg: Sketching the Epps–Pulley Test

§4.2.1은 유한 개의 모멘트(moment)가 분포를 유일하게 정하지 못함을, §4.2.2는 누적분포함수(cumulative distribution function, CDF) 기반 비교의 정렬·분산 처리 부담을 설명한다. §4.2.3은 **특성함수**(characteristic function, CF)를 선택한다. 이는 여러 주파수에서 sine·cosine의 평균으로 분포를 요약하는 지문이다. 가우시안의 지문은 닫힌 식으로 알려져 있다.

Epps–Pulley 통계는 표본의 지문과 가우시안 지문의 차이를 가중 적분한다. SIGReg는 여러 방향의 통계를 평균한다. 표본 쌍을 모두 만들 필요가 없고, 방향·주파수 수를 고정하면 batch 크기에 선형으로 계산한다. SIGReg 값이 낮을수록 뽑힌 방향들에서 **표본 투영의 분포가 목표에 가깝다**. Theorem 4의 기울기 상한이 어떤 정규화의 손실에 대한 것인지는 [심층 편 §4.2.3](/reviews/lejepa-technical/#423-cf-theorem-4의-유도와-정규화)에서 다룬다. [§4.2, Theorems 3–4, Algorithm 1](https://arxiv.org/pdf/2511.08544v3#page=7)

### 4.3 How SIGReg Beats the Curse of Dimensionality

저자들의 논리는 두 가지다. 부드러운 밀도에서는 관측한 방향의 영향이 주변 방향으로 퍼질 수 있고, SGD 반복마다 새 방향을 뽑으면 누적 관측 범위가 넓어진다. Theorem 5의 감소율에는 매끄러움(smoothness)과 방향 배치 조건이 있다. 방향 재표집의 효과를 보인 합성 실험은 [전체 해설 §4.3](/reviews/lejepa-complete/#43-how-sigreg-beats-the-curse-of-dimensionality)에서 읽는다. [§4.3; Appendix B.10, Figures 6–7](https://arxiv.org/pdf/2511.08544v3#page=10)

<span id="5-lejepa-sigreg와-prediction-loss의-결합" aria-hidden="true"></span>

## 5. LeJEPA: 안정적이고 확장 가능한 구현 (LeJEPA — Stable and Scalable Implementation)

### 5.1 SIGReg + Prediction Loss

전역 뷰(global view) $V_g$개의 평균 $\boldsymbol\mu_n$을 기준으로 모든 뷰의 표현을 모으고, 각 뷰에서는 서로 다른 이미지들의 분포에 SIGReg를 적용한다.

$$
\boldsymbol\mu_n=\frac1{V_g}\sum_{v=1}^{V_g}\mathbf z_{n,v},\qquad
\mathcal L_{\mathrm{LeJEPA}}=(1-\lambda)\mathcal L_{\mathrm{pred}}+\lambda\mathcal L_{\mathrm{SIGReg}}.
$$

오른쪽 두 손실 기호는 설명용 약칭이며, Algorithm 2의 실제 식과의 대응은 [전체 해설 §5.1](/reviews/lejepa-complete/#51-lejepa-sigreg-prediction-loss)에 있다. 저자들은 별도 예측기, 기울기 차단(stop-gradient), 교사–학생(teacher–student) 네트워크 없이 이 손실만으로 붕괴를 피한다고 보고한다. [§5.1, Algorithms 1–2; Appendix B.6, B.13](https://arxiv.org/pdf/2511.08544v3#page=11)

### 5.2 Relation to Prior Work

투영은 sliced score matching·sliced Wasserstein과 연결되고, 정확히 적분한 CF 거리는 kernel MMD와 연결된다. VICReg와는 뷰 합의와 분산·공분산 제어라는 공통점이 있다. LeJEPA는 투영의 분포 전체를 목표로 한다. 저자들은 JEPA에 슬라이싱·분포 일치를 쓴 선행 해법은 없지만 생성 모델과 최적 수송(optimal transport)에는 비슷한 방법이 있다고 밝힌다. [§5.2; Appendix B.14](https://arxiv.org/pdf/2511.08544v3#page=12)

<span id="6-empirical-validation-다섯-실험-축을-순서대로" aria-hidden="true"></span>

## 6. LeJEPA의 실험 검증 (LeJEPA: Empirical Validation)

수치는 원문 보고값이며 정확도(%)는 높을수록 좋다.

| 원문 절 | 핵심 관측 | 조건과 한계 |
| --- | --- | --- |
| §6.1 안정성 | IN-1K, ViT-L/14, 100 epoch frozen top-1: batch 128은 72.20, 512는 74.72 | 뷰 구성의 영향이 더 크다(전역 뷰 1개·총 4뷰 53.06). [Table 1, p. 13](https://arxiv.org/pdf/2511.08544v3#page=13) |
| §6.2 학습 손실과 성능 | prediction·SIGReg가 함께 작은 영역에서 높은 정확도; 보정한 손실과 강한 단조 관련성 | 보정 지수는 평가 정확도와의 상관을 보고 고른 값이다. [Figures 10–11, p. 15](https://arxiv.org/pdf/2511.08544v3#page=15) |
| §6.3 도메인 내 학습 | Galaxy10 전체 라벨 frozen: ResNet-34 78.17, DINOv3 ViT-S 71.38 | 400 epoch 도메인 내 사전학습. LeViT-128은 69.00으로 DINOv3보다 낮다. [Table 3, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42) |
| §6.4 규모와 전이 | 8개 전이 과제 전체 라벨 평균: ViT-L 79.48, I-JEPA 78.50, 확률적 예측 과제를 더한 I-JEPA + STOP 80.70 | LeJEPA는 304M·100 epoch, I-JEPA 계열은 632M·300 epoch, 모두 IN-1K. [Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16) |
| §6.5 의미 구조 | attention threshold와 patch PCA에서 전경·배경 구조 | 정성적 그림이다. [Figures 13–14, pp. 16–17](https://arxiv.org/pdf/2511.08544v3#page=17) |

![Figure 12: Galaxy10 라벨 수에 따른 finetuning과 frozen 평가.](/assets/reviews/lejepa/paper-figure-12.webp)

**Figure 12.** 가로축은 후속 분류에 쓰는 **클래스당 라벨 수**, 세로축은 정확도다. 왼쪽은 특징 추출기(backbone)도 수정하는 미세조정(finetuning), 오른쪽은 frozen 평가이며, 약 11,000장의 Galaxy10으로 400 epoch 학습한 LeJEPA 소형 모델(주황 계열)과 DINOv2/v3(파랑 계열)를 비교한다. 라벨 수에 따른 차이의 추세는 [전체 해설 §6.3](/reviews/lejepa-complete/#63-in-domain-lejepa-outperforms-frontier-model-transfer-learning)에서 읽는다. [확대](/assets/reviews/lejepa/paper-figure-12.webp) · [Figure 12; Table 3](https://arxiv.org/pdf/2511.08544v3#page=16). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

<span id="7-conclusion-저자가-연결한-설계-논리" aria-hidden="true"></span>

## 7. 결론 (Conclusion)

저자의 결론은 **후속 예측 분석 → 등방성 가우시안 목표 → 계산 가능한 SIGReg → 간결한 학습 구조**가 하나로 이어진다는 것이다. 저자들은 이 설계가 교사–학생 네트워크나 기울기 차단 없이 붕괴를 막으며, 60개 이상의 구조에서 검증했고 50줄 미만의 핵심 구현으로 최신 성능에 맞먹는다고 정리한다. [§7, p. 18](https://arxiv.org/pdf/2511.08544v3#page=18)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 이 논문의 설득력은 '좋은 표현 분포'를 먼저 정하고 그 분포를 방향별 특성함수 비교로 강제하는 연결이 단순하다는 데 있다. 같은 기본 설정으로 Galaxy10에서 직접 학습해 일반 이미지 모델보다 높은 frozen 정확도를 낸 결과가 그 단순성의 실용적 가치를 가장 직접 보여 준다. 남은 질문은 전이 비교의 I-JEPA가 약 2배 크고 3배 오래 학습한 모델이라, 같은 크기·학습 길이에서도 차이가 유지되는지다. 이론의 조건과 실험별 한계는 [\[2/3\] 전체 해설의 해석 절](/reviews/lejepa-complete/#리뷰-해석과-남은-질문)에서 이어진다.

## 출처와 읽은 범위

- 원문: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics* · Randall Balestriero et al. · 최초 공개 2025-11-11 · [검토 버전](https://arxiv.org/abs/2511.08544v3) · 검토일 2026-10-23.
- 읽은 범위: v3 본문 §1–7의 입문 지도. 상세 실험은 [2/3] 전체 해설, Appendix A·B의 증명은 [3/3] 심층 편에 대응한다.
- 그림 출처: 대표 개념도는 자체 제작물(CC BY 4.0). 원문 그림은 Balestriero·LeCun의 v3 자산(CC BY-SA 4.0)으로 전체 패널 crop·무손실 WebP 변환을 적용한다. [원문](https://arxiv.org/abs/2511.08544v3).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
