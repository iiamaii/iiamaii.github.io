---
title: "[1/3] LeJEPA 입문: 같은 이미지는 가깝게, 표현 분포는 Gaussian으로"
description: "저자의 §§1–7 순서로 문제, Gaussian 목표, SIGReg, 학습과 다섯 실험 축을 짧게 따라가는 입문 지도."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-11T20:35:21+09:00"
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
thumbnailAlt: "LeJEPA의 두 제약 설명도: 한 이미지의 뷰들은 전역 뷰 평균에 정렬하고, 각 뷰에서 여러 이미지의 표현은 임의 방향으로 투영해 표준 Gaussian과 맞춘다."
---

**LeJEPA는 후속 과제(downstream task)를 아직 모를 때 유리한 표현 분포를 먼저 분석하고, 그 목표를 실제 학습 손실로 연결하는 자기지도학습(self-supervised learning) 방법이다.** 같은 이미지에서 만든 여러 뷰(view)는 공통 내용을 공유하게 하고, 서로 다른 이미지의 표현 분포는 등방성 가우시안(isotropic Gaussian)에 맞춘다. 이론의 적용에는 조건이 있으며 실제 학습은 유한한 투영과 수치 적분을 사용한다. [원문 §§1–5](https://arxiv.org/pdf/2511.08544v3#page=1)

이 글은 **[1/3] 입문 지도**다. 원문의 §§1–7 순서를 유지하며, 증명과 상세 비교는 [2/3 전체 해설](/reviews/lejepa-complete/) → [3/3 정리·증명 심층](/reviews/lejepa-technical/) 순서로 읽을 수 있다. 전체 해설에는 §6.1–6.5의 핵심 실험이, 심층 편에는 Appendix A·B의 주요 계산이 있다.

대표 그림에서 왼쪽 아래 손실은 **한 이미지의 여러 뷰**를 모으고, 오른쪽 아래 손실은 **각 뷰에서 여러 이미지**의 표현 분포를 맞춘다. Gaussian 목표는 투영 헤드(projector) 뒤의 공간에 적용한다. 점과 곡선은 설명용이다. [대표 그림 확대](/assets/reviews/lejepa/lejepa-core.svg)

## 1 Introduction — 분포 목표부터 학습 방법까지

라벨 없이 사진의 특징을 배우려면 같은 사진의 다른 크롭(crop)을 연결해야 한다. 그러나 모든 입력에 같은 벡터를 출력해도 뷰 사이 거리는 작아진다. 이를 **표현 붕괴**(representation collapse)라고 한다. 일부 차원만 사용하는 차원 붕괴(dimensional collapse)도 있다.

저자들은 붕괴를 피하는 장치를 나열하기 전에 질문을 바꾼다. **나중에 어떤 과제를 풀지 모른다면, 학습한 표현은 어떤 분포를 이루는 것이 좋은가?** 답을 통계적 예측기의 분석에서 찾고, SIGReg(Sketched Isotropic Gaussian Regularization)라는 분포 정규화(distribution regularization)와 뷰 합의(view agreement)를 결합한다. [§1, pp. 1–3](https://arxiv.org/pdf/2511.08544v3#page=1)

## 2 Background and Notations — JEPA가 요구하는 두 조건

### 2.1 Notations and Definitions

인코더(encoder) $f_{\boldsymbol\theta}:\mathbb R^D\to\mathbb R^K$는 입력을 $K$차원 표현으로 바꾼다. 이미지 $n$의 뷰 $v$에서 나온 표현은 $\mathbf z_{n,v}$다. 한 배치(batch)에는 독립 이미지 $N$개, 이미지마다 뷰 $V$개가 있다.

원문의 JEPA 정의는 관련 입력 사이의 **예측 가능성**(predictability)과 표현의 **비퇴화성**(non-degeneracy)을 함께 요구한다. 이번 이미지 구현은 대칭적 뷰 합의에 집중한다. 행동 정보가 있는 미래 예측처럼 입력 정보가 비대칭인 문제에는 예측기(predictor)가 필요할 수 있다. [§2.1, p. 4](https://arxiv.org/pdf/2511.08544v3#page=4)

### 2.2–2.3 Reliable Pretraining과 Actionable Theory

좋은 표현은 여러 구조와 데이터에서 학습할 수 있어야 한다. 이론도 ‘왜 붕괴하지 않는가’뿐 아니라 실제로 **어떤 분포와 손실을 선택할 것인가**에 답해야 한다. 이 두 요구가 §3의 목표 분포와 §4의 계산 방법으로 이어진다. [§§2.2–2.3](https://arxiv.org/pdf/2511.08544v3#page=4)

## 3 Latent Euclidean — 왜 등방성 Gaussian인가

### 3.1 Linear Probing

**선형 프로브**(linear probe)는 encoder를 고정하고 그 특징 위에 작은 선형 예측기를 학습하는 평가다. 같은 총 에너지 아래 특정 방향의 변화가 너무 작으면, 그 방향을 필요로 하는 과제의 계수 추정이 불리해진다. Lemma 1은 리지 정규화(ridge regularization)에서 더 큰 편향(bias)을 겪는 과제가 **존재함**을 보이고, Lemma 2는 최소제곱법(ordinary least squares, OLS)의 계수 분산(variance) 합이 등방성에서 최소임을 보인다. 모든 과제의 분류 정확도가 무조건 높아진다는 뜻은 아니다. [§3.1; Appendix B.1–B.2](https://arxiv.org/pdf/2511.08544v3#page=5)

### 3.2 Nonlinear Probing

등방성(isotropy)은 모든 방향의 분산이 같다는 뜻이다. 이 조건만으로 분포가 Gaussian이 되지는 않는다. 저자들은 가까운 이웃의 정답을 평균하는 반경 기반 최근접 이웃(k-nearest neighbors, kNN)과 커널(kernel) 예측기도 분석한다.

작은 이웃·대역폭(bandwidth)에서의 편향(bias)에는 밀도 스코어(density score), 즉 로그 밀도의 변화율인 $\nabla\log p$가 들어간다. 그 크기의 제곱 평균인 **피셔 정보**(Fisher information) $J(p)$는 적절한 조건과 고정 공분산(covariance) 아래 Gaussian에서 최소다. 공분산의 총량까지 고정하면 등방성 Gaussian이 선택된다. Theorem 1의 Gaussian 목표는 이런 국소 추정과 과제 가정에서 나온다. 전체 위험의 보편적 최적성을 인증한 것으로 읽으면 안 된다. [§3.2; Appendix A, B.3–B.7](https://arxiv.org/pdf/2511.08544v3#page=5)

### 3.3 Geometric and Practical Insights

§3.3의 설명용 실험은 같은 총 분산의 등방성(isotropy)·비등방성(anisotropy) 표현에서 추정 계수와 결정 경계가 어떻게 달라지는지 보여준다. 실제 이미지 backbone의 성능은 뒤의 §6에서 별도로 검증한다. [§3.3, Figure 3](https://arxiv.org/pdf/2511.08544v3#page=6)

## 4 SIGReg — 고차원 분포를 여러 방향에서 확인하기

### 4.1 Hypothesis Testing as a Judge

벡터를 길이 1의 방향 $\mathbf a$로 투영하면 스칼라 $\mathbf a^\top\mathbf z$를 얻는다. 모든 방향의 1차원 분포가 표준 Gaussian이라면 원래 벡터도 표준 다변량 Gaussian이다. Lemma 3은 이를 Cramér–Wold 정리와 연결한다. 실제 학습에서는 유한한 방향만 보고 반복마다 다시 뽑는다. [§4.1; Appendix B.8–B.9](https://arxiv.org/pdf/2511.08544v3#page=6)

![Figure 2: 표현을 여러 방향으로 투영하여 목표 Gaussian과 비교하는 개념도.](/assets/reviews/lejepa/paper-figure-2.webp)

**Figure 2 읽기.** 왼쪽은 입력 분포, 가운데는 표현 공간, 오른쪽은 색깔별 방향의 투영 밀도다. 검은 곡선은 목표 Gaussian이고 가로축은 투영 좌표다. 실제 ImageNet 측정치가 아닌 방법의 개념도다. [확대](/assets/reviews/lejepa/paper-figure-2.webp) · [원문 Figure 2, p. 3](https://arxiv.org/pdf/2511.08544v3#page=3). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

### 4.2 SIGReg: Sketching the Epps–Pulley Test

§4.2.1은 유한 개의 모멘트(moment)가 분포를 유일하게 정하지 못함을, §4.2.2는 누적분포함수(cumulative distribution function, CDF) 기반 비교의 정렬·분산 처리 부담을 설명한다. §4.2.3은 **특성함수**(characteristic function, CF)를 선택한다. 이는 여러 주파수에서 sine·cosine의 평균으로 분포를 요약하는 지문이다. Gaussian의 지문은 닫힌 식으로 알려져 있다.

Epps–Pulley 통계는 표본의 지문과 Gaussian 지문의 차이를 가중 적분한다. SIGReg는 여러 방향의 통계를 평균한다. 표본 쌍을 모두 만들 필요가 없고, 방향·주파수 수를 고정하면 batch 크기에 선형으로 계산한다. 기울기의 정규화 조건은 심층 편에서 구분한다. [§4.2, Theorems 3–4, Algorithm 1](https://arxiv.org/pdf/2511.08544v3#page=7)

### 4.3 How SIGReg Beats the Curse of Dimensionality

저자들의 논리는 두 가지다. 부드러운 밀도에서는 관측한 방향의 영향이 주변 방향으로 퍼질 수 있고, SGD 반복마다 새 방향을 뽑으면 누적 관측 범위가 넓어진다. Theorem 5의 감소율에는 매끄러움(smoothness)과 방향 배치 조건이 있다. 합성 실험의 성공도 전체 학습의 보편적 수렴 증명을 대신하지 않는다. [§4.3; Appendix B.10, Figures 6–7](https://arxiv.org/pdf/2511.08544v3#page=10)

## 5 LeJEPA — SIGReg와 Prediction Loss의 결합

### 5.1 SIGReg + Prediction Loss

전역 뷰(global view) $V_g$개의 평균 $\boldsymbol\mu_n$을 기준으로 모든 뷰의 표현을 모으고, 각 뷰에서는 서로 다른 이미지들의 분포에 SIGReg를 적용한다.

$$
\boldsymbol\mu_n=\frac1{V_g}\sum_{v=1}^{V_g}\mathbf z_{n,v},\qquad
\mathcal L_{\mathrm{LeJEPA}}=(1-\lambda)\mathcal L_{\mathrm{pred}}+\lambda\mathcal L_{\mathrm{SIGReg}}.
$$

뒤의 두 손실 기호는 설명용 평균 손실 약칭이다. Algorithm 2의 중심 손실은 feature 좌표까지 평균한다. 별도 예측기(predictor), 기울기 차단(stop-gradient), 지수이동평균 교사(exponential moving average teacher, EMA teacher)가 핵심 구조의 필수 요소는 아니다. optimizer·학습률·crop 등은 여전히 설정한다. 원문 Eq. (5)→(6)의 등식에는 전역 뷰 내부 분산 항이 빠져 있어, [심층 편](/reviews/lejepa-technical/)에서 계산을 보충했다. [§5.1, Algorithms 1–2; Appendix B.6, B.13](https://arxiv.org/pdf/2511.08544v3#page=11)

### 5.2 Relation to Prior Work

투영은 sliced score matching·sliced Wasserstein과 연결되고, 정확히 적분한 CF 거리는 kernel MMD와 연결된다. VICReg와는 뷰 합의와 분산·공분산 제어라는 공통점이 있다. LeJEPA는 투영의 분포 전체를 목표로 한다. 모든 재료가 이 논문에서 처음 제안된 것은 아니다. [§5.2; Appendix B.14](https://arxiv.org/pdf/2511.08544v3#page=12)

## 6 Empirical Validation — 다섯 실험 축을 순서대로

아래는 **원문 보고 결과**다. 이 리뷰에서 모델을 재학습하지 않았다. 정확도는 %이며 높을수록 좋다.

| 원문 절 | 핵심 관측 | 조건과 한계 |
| --- | --- | --- |
| §6.1 안정성 | IN-1K, ViT-L/14, 100 epoch frozen top-1: batch 128은 72.20, 512는 74.72 | 안정적 학습과 동일 성능은 다르다. 기본 권장값과 개별 ablation을 구분한다. [Table 1, p. 13](https://arxiv.org/pdf/2511.08544v3#page=13) |
| §6.2 학습 손실과 성능 | prediction·SIGReg가 함께 작은 영역에서 높은 정확도; 보정한 손실과 강한 단조 관련성 | 모든 새 과제에서 라벨 없이 모델을 선택한다는 보장은 아니다. [Figures 10–11, p. 15](https://arxiv.org/pdf/2511.08544v3#page=15) |
| §6.3 도메인 내 학습 | Galaxy10 전체 라벨 frozen: ResNet-34 78.17, DINOv3 ViT-S 71.38 | 작은 도메인 모델의 강점. 모든 구조·라벨 수에서 우위는 아니다. [Table 3, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42) |
| §6.4 규모와 전이 | 8개 전이 과제 전체 라벨 평균: ViT-L 79.48, I-JEPA 78.50, +STOP 80.70 | 304M·100 epoch와 632M·300 epoch를 비교한다. epoch 수는 실측 총 비용이 아니다. [Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16) |
| §6.5 의미 구조 | attention threshold와 patch PCA에서 전경·배경 구조 | 정성적 그림이며 segmentation benchmark 점수는 제시하지 않는다. [Figures 13–14, pp. 16–17](https://arxiv.org/pdf/2511.08544v3#page=17) |

![Figure 12: Galaxy10 라벨 수에 따른 finetuning과 frozen 평가.](/assets/reviews/lejepa/paper-figure-12.webp)

**Figure 12 읽기.** 가로축은 사전학습 이미지 수가 아니라 후속 분류에 쓰는 **클래스당 라벨 수**, 세로축은 정확도다. 왼쪽은 특징 추출기(backbone)도 수정하는 미세조정(finetuning), 오른쪽은 표현 고정 평가(frozen evaluation)다. LeJEPA는 약 11,000장의 Galaxy10으로 400 epoch 학습한 작은 모델들이다. 1-shot frozen에서는 LeViT 25.85가 DINOv3 30.17보다 낮다. 원문은 3개 seed를 보고하지만 오차막대 정의는 미확인이다. [확대](/assets/reviews/lejepa/paper-figure-12.webp) · [Figure 12; Table 3](https://arxiv.org/pdf/2511.08544v3#page=16). Balestriero·LeCun, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), 전체 패널 crop·무손실 WebP 변환.

## 7 Conclusion — 저자가 연결한 설계 논리

저자의 결론은 **후속 예측 분석 → 등방성 Gaussian 목표 → 계산 가능한 SIGReg → 간결한 학습 구조**가 하나로 이어진다는 것이다. 여러 이미지 구조와 데이터에서 이를 평가한 점도 중요하다. 이론의 조건, 유한 방향·주파수 근사, 비교 예산을 함께 읽어야 한다. 이 논문의 주 실험은 이미지 표현 학습이며 행동 조건부 미래 예측까지 검증하지 않는다. [§7, p. 18; §2.1](https://arxiv.org/pdf/2511.08544v3#page=18)

## 출처와 읽은 범위

- 원문: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, Randall Balestriero·Yann LeCun. 최초 공개 2025-11-11, 검토본 v3(2025-11-14), 50쪽. [공식 서지·버전 이력](https://arxiv.org/abs/2511.08544).
- 이번 1차 읽기(2026-10-11): 초록·서론·결론, 전체 절 구성, Figure 1–2의 설명을 확인해 논증 지도를 저장했다. 2차 읽기: 본문 §§1–7, Appendix A·B의 주요 정리와 증명, C의 추가 표·캡션, D–F의 대안 설명을 PDF·저자 LaTeX로 대조했다. 모든 참고 원전을 다시 읽은 것은 아니다.
- 이 글은 §§1–7의 짧은 지도다. 2.2–2.3만 함께 요약했다. 상세 관련 연구·모든 핵심 실험은 [전체 해설](/reviews/lejepa-complete/), 주요 증명·보충 유도·간극은 [심층 편](/reviews/lejepa-technical/)에 있다.
- 직접 검증: 리뷰의 중심 손실 항등식, OLS 분산 예제, 1D 고정 주파수 grid의 alias, ECF 표본 bias 수치 예제. **미실행:** 공식 코드 실행, GPU 학습, probe·finetuning·DDP 재현, 속도·메모리 재측정, 전체 증명의 형식 검증. 그림은 기존 승인 자산을 재사용했다.
- 대표 그림은 §§4–5·Algorithms 1–2에 근거한 자체 개념도, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 기본 Light(300), 강조 Semi Bold(600).
