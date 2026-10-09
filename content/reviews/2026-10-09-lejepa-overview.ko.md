---
title: "LeJEPA 입문: 같은 이미지는 가깝게, 표현 분포는 Gaussian으로"
description: "표현 붕괴와 두 손실의 역할을 쉽게 설명하고, Gaussian 목표·임의 투영·주요 실험을 그림과 함께 읽는다."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-09T23:54:43+09:00"
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

정답 라벨 없이 사진의 특징을 학습할 때는 두 문제가 동시에 생긴다. 같은 사진의 다른 부분은 비슷한 표현으로 묶고 싶지만, 모든 사진을 같은 값으로 보내면 아무것도 구분하지 못한다. <strong>LeJEPA는 ‘같은 이미지의 뷰는 합의하게 하고, 여러 이미지의 표현 분포는 표준 Gaussian에 맞추자’는 두 조건으로 이 문제를 풀려 한다.</strong> 복잡한 교사 모델이나 gradient 차단 없이 학습 목적을 구성하고, 그 분포를 선택하는 이유를 통계적 추정 이론과 연결한다. 다만 그 이론이 모든 실제 과제의 성능을 보장하는 것은 아니다. [원문 §§1–5, pp. 1–12](https://arxiv.org/pdf/2511.08544v3#page=1)

이 글은 세 버전 중 입문 편이다. 방법·관련 연구·실험을 빠짐없이 따라가려면 [전체 해설](/reviews/lejepa-complete/), 증명의 조건과 수식 검토가 궁금하면 [기술 심층 리뷰](/reviews/lejepa-technical/)로 이어갈 수 있다.

<strong>대표 그림을 읽는 법.</strong> 위쪽은 이미지의 여러 뷰를 같은 encoder와 projector로 처리하는 흐름이다. 아래 왼쪽은 <strong>한 이미지 안에서</strong> 뷰들의 표현을 전역 뷰의 평균에 모은다. 아래 오른쪽은 <strong>각 뷰에서 여러 이미지에 걸쳐</strong> 표현을 모아 임의 방향으로 투영한 분포를 표준 Gaussian에 맞춘다. 아래 두 가지 손실을 합쳐 학습한다. 점과 곡선은 개념 설명용이며 측정 데이터가 아니다. [대표 그림 확대](/assets/reviews/lejepa/lejepa-core.svg)

## 같은 대상을 알아보면서 서로 다른 대상을 구별하기

고양이 사진을 크게 잘라 만든 두 뷰와 귀 주변을 작게 잘라 만든 뷰를 생각해 보자. 밝기나 배경이 달라도 공통 내용을 포착하는 표현을 얻고 싶다. 이런 식으로 라벨 없이 여러 뷰를 묶는 학습을 자기지도학습이라고 한다. 여기서 <strong>뷰</strong>는 같은 원본 이미지에서 crop과 augmentation으로 만든 입력이다.

문제는 뷰 사이 거리를 줄이라는 조건만으로는 의미 있는 특징을 얻지 못한다는 것이다. encoder가 모든 입력에 `0`을 출력하면 뷰 사이 거리는 완벽하게 줄어든다. 하지만 고양이와 은하 사진도 구별할 수 없다. 이를 <strong>표현 붕괴</strong>라고 부른다. 몇 개 차원만 사용하는 부분 붕괴도 문제다. [§2, pp. 3–4](https://arxiv.org/pdf/2511.08544v3#page=3)

LeJEPA의 첫 번째 조건은 같은 이미지의 뷰들이 공유하는 정보를 유지하는 것이다. 두 번째 조건은 서로 다른 이미지들의 표현이 하나의 점이나 좁은 방향으로 몰리지 않도록 분포를 제어하는 것이다. 두 조건은 서로 다른 축에서 작동한다. 한 사진의 crop들만 Gaussian처럼 흩뜨리는 방법으로 이해하면 안 된다.

## 왜 둥글게 퍼진 Gaussian을 목표로 삼을까

Gaussian은 종 모양의 정규분포다. <strong>등방성</strong>은 특정 방향에 치우치지 않고 모든 방향에 같은 분산을 갖는다는 뜻이다. 표준 다변량 Gaussian `N(0, I)`는 평균이 0이고 각 방향의 분산이 1이다. 서로 다른 이미지의 표현들이 이런 분포를 이루도록 유도한다.

논문의 출발점은 ‘보기 좋은 분포’가 아니라 <strong>나중에 어떤 예측 문제를 풀게 될지 모르는 상황에서 유리한 표현 기하</strong>다. 선형 예측기의 경우, 표현이 어떤 방향으로 거의 변하지 않으면 그 방향에 의존하는 과제를 배우기 어렵다. 동일한 전체 분산을 여러 방향에 균등하게 나누면 이런 약한 방향을 줄일 수 있다. [§3.1, p. 5; Appendix B.1–B.2](https://arxiv.org/pdf/2511.08544v3#page=5)

그런데 공분산만 둥글다고 분포 전체가 Gaussian인 것은 아니다. 논문은 비선형 이웃·kernel 예측기까지 살펴보고, 밀도의 변화율을 측정하는 Fisher information이 작을수록 특정 bias 항이나 그 상한이 줄어드는 조건을 제시한다. 고정된 공분산에서 Gaussian은 이 Fisher information을 최소화한다. <strong>선형 분석은 균등한 방향 분산을, 비선형 분석은 추가 가정 아래 Gaussian 분포를 선택할 이유를 제공한다.</strong> 이는 실제 모든 과제에서 항상 가장 좋다는 보장은 아니다. [§3.2; Appendix A, B.3–B.7](https://arxiv.org/pdf/2511.08544v3#page=5)

## 고차원 분포를 여러 개의 그림자로 확인하기

표현 벡터가 수백 또는 수천 차원이라면 그 전체 밀도를 직접 비교하기 어렵다. LeJEPA는 임의 방향에 벡터를 투영해 1차원 값으로 바꾼다. 물체를 여러 각도에서 비춰 그림자를 보는 것처럼 이해할 수 있다.

모든 방향의 1차원 분포가 표준 Gaussian이면 원래 벡터도 표준 다변량 Gaussian이다. 이것이 Cramér–Wold 정리의 연결 고리다. 실제 학습에서는 모든 방향을 볼 수 없으므로 유한한 방향을 뽑고, 반복 중에 다시 뽑는다. <strong>유한한 관측으로 분포 전체가 정확히 일치했다고 인증하는 것과는 구분해야 한다.</strong> [§4.1, p. 6; Appendix B.8–B.9](https://arxiv.org/pdf/2511.08544v3#page=6)

![LeJEPA Figure 2: 입력 분포를 표현 공간으로 옮기고 여러 방향으로 투영해 표준 Gaussian과 비교한다.](/assets/reviews/lejepa/paper-figure-2.webp)

[그림 확대](/assets/reviews/lejepa/paper-figure-2.webp)

<strong>Figure 2 해설.</strong> 왼쪽은 입력 분포, 가운데는 encoder가 만든 표현 분포를 개념적으로 나타낸다. 가운데의 색깔 화살표는 서로 다른 투영 방향이다. 오른쪽 가로축은 그 방향의 투영 좌표이며, 색 곡선은 투영된 밀도, 검은 곡선은 목표 Gaussian이다. 회색 영역은 두 분포의 차이를 표현한다. 실제 대규모 모델의 표현을 측정한 결과가 아니라 방법의 개념도다. 원문 저자 Randall Balestriero·Yann LeCun의 [Figure 2, p. 3](https://arxiv.org/pdf/2511.08544v3#page=3), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). 그림 부분을 crop하고 무손실 WebP로 변환했다.

투영 후에는 평균과 분산만 재는 대신 <strong>특성함수</strong>를 비교한다. 이는 값들을 여러 주파수의 sine·cosine으로 요약하는 분포의 지문이다. 이 지문을 목표 Gaussian의 지문에 가깝게 만드는 손실을 SIGReg라고 부른다. 정렬이나 모든 이미지 쌍의 비교를 하지 않고, 표본의 평균으로 계산할 수 있다는 장점이 있다. 정확한 식과 근사 범위는 [기술 심층 리뷰](/reviews/lejepa-technical/)에서 설명한다. [§4.2–4.3, pp. 7–11](https://arxiv.org/pdf/2511.08544v3#page=7)

## 실제 학습은 무엇을 하는가

학습 구조에는 이미지 특징을 만드는 encoder와 그 특징을 손실용 공간으로 옮기는 projector가 있다. 따라서 ‘모델의 모든 내부 표현이 Gaussian’이라고 말할 수는 없다. Gaussian 제약은 projector 뒤의 표현에 적용하고, 아래 평가 중 frozen probe는 backbone 특징을 사용한다. [§5, Algorithm 2; §6](https://arxiv.org/pdf/2511.08544v3#page=11)

한 batch에 서로 다른 이미지가 `N`개, 이미지마다 뷰가 `V`개 있다고 하자. 큰 전역 뷰들의 평균 표현을 기준으로 모든 뷰의 거리를 줄인다. 별도로 각 뷰에서 `N`개 이미지 표현에 SIGReg를 적용한다. 최종 손실은 다음과 같다.

```text
전체 손실 = (1 − λ) × 뷰 합의 손실 + λ × SIGReg 손실
```

`λ`는 두 조건의 비중을 정한다. 논문 기본값은 `0.05`다. 핵심 구조는 이동평균 교사(EMA teacher), stop-gradient, 별도 예측 네트워크를 요구하지 않는다. 그렇다고 설정할 것이 모두 사라진 것은 아니다. optimizer, learning rate, weight decay, crop, batch 크기, 투영 수와 수치 적분 설정은 여전히 있다. 논문은 학습률 warmup과 cosine schedule도 사용한다. ‘하이퍼파라미터 하나’는 주로 손실의 혼합 계수에 대한 표현으로 읽는 것이 정확하다. [§5–6.1, pp. 11–14](https://arxiv.org/pdf/2511.08544v3#page=11)

## 실험은 무엇을 보여주었나

LeJEPA의 실험은 ImageNet 계열과 작은 이미지 데이터셋에서 수행됐다. ViT뿐 아니라 convolution·hybrid 모델을 포함하는 여러 구조를 시험했다. 아래 수치는 원문 보고 결과이며 이 리뷰에서 재학습한 결과가 아니다.

| 질문 | 원문 근거 | 읽을 때의 조건 |
| --- | --- | --- |
| 비교적 작은 batch도 학습할 수 있는가? | ImageNet-1K, ViT-L/14, 100 epoch의 frozen linear top-1: batch 128은 72.20%, 512는 74.72% | 안정적으로 학습된다는 것과 같은 정확도라는 것은 다르다. [Table 1, p. 13](https://arxiv.org/pdf/2511.08544v3#page=13) |
| 다른 과제로 전이되는가? | 8개 전이 과제의 전체 라벨 평균: LeJEPA ViT-L 79.48%, I-JEPA 78.50% | 모델 크기와 학습 기간이 다르며, I-JEPA+STOP은 80.70%다. [Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16) |
| 작은 전문 데이터에서도 쓸 만한가? | Galaxy10, 전체 라벨의 frozen probe: LeJEPA ResNet-34 78.17%, DINOv3 ViT-S 71.38% | 도메인에 맞춘 학습과 대규모 일반 사전학습의 비교다. 모든 모델·라벨 수에서의 우위는 아니다. [Table 3, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42) |

<strong>Frozen probe</strong>는 encoder를 고정하고 라벨로 작은 분류기를 학습해 표현의 활용도를 보는 평가다. 자기지도 사전학습에 라벨을 쓰지 않았다는 것과, 평가에서도 라벨을 쓰지 않는다는 것은 다르다. Top-1 정확도는 가장 높은 점수를 준 클래스가 정답인 비율이며 높을수록 좋다.

![LeJEPA Figure 12: Galaxy10에서 클래스당 라벨 수에 따른 full finetuning과 frozen backbone 정확도 비교.](/assets/reviews/lejepa/paper-figure-12.webp)

[그림 확대](/assets/reviews/lejepa/paper-figure-12.webp)

<strong>Figure 12 해설.</strong> 가로축은 사전학습 이미지 수가 아니라 <strong>후속 분류 학습에 사용하는 클래스당 라벨 수</strong>다. 세로축은 정확도(%). 왼쪽은 encoder까지 수정하는 full finetuning, 오른쪽은 encoder를 고정하는 평가다. 주황·갈색 계열은 약 11,000장의 Galaxy10 이미지로 학습한 LeJEPA 모델들, 파란색 계열은 DINOv2/v3의 일반 사전학습 모델들이다. 라벨이 늘어난 구간에서 도메인에 맞춘 작은 모델의 강점이 보인다. 1-shot에서는 LeJEPA의 모든 구조가 앞서는 것은 아니다. 원문은 세 seed를 보고하지만 오차막대의 정확한 요약 방식은 명시하지 않아 신뢰구간이라고 부르지 않는다. 저자·출처: [Figure 12, p. 16; Table 3, p. 42](https://arxiv.org/pdf/2511.08544v3#page=16), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). 전체 그래프 패널을 crop·무손실 WebP 변환했다.

## 이 논문의 핵심을 어떻게 평가할까

이 리뷰의 해석으로는 가장 큰 기여가 <strong>표현 붕괴를 막는 목표를 명시적인 분포 제약으로 만들고, 그 선택을 이론·계산·실험으로 연결한 것</strong>이다. ‘안정적인 학습을 위해 여러 장치를 붙인다’는 접근에서, 어떤 분포가 후속 예측에 유리한지부터 생각하는 접근으로 시선을 옮긴다.

그러나 이론은 smooth density, 후속 과제의 분포와 사전 가정, 작은 이웃이나 bandwidth 같은 조건을 사용한다. 실제 SIGReg도 방향과 주파수가 유한하다. 전이 표에는 우위와 열위가 함께 있고, 일부 비교는 모델·학습량이 다르다. 따라서 ‘증명됐으니 어떤 데이터에서도 최선’ 또는 ‘적은 epoch이니 총 비용도 정확히 그 비율’로 읽지 않는 것이 중요하다. [§3–4; Tables 1–3](https://arxiv.org/pdf/2511.08544v3#page=5)

또한 JEPA라는 이름이 붙어 있지만 이 논문의 주된 실험은 이미지 뷰의 표현 학습이다. 행동을 넣어 미래 상태를 예측하는 world action model이나 로봇 제어까지 검증한 결과는 아니다. 비대칭적 시간·행동 정보가 필요한 문제로 옮길 때는 예측 구조를 다시 검토해야 한다. [§2.1, p. 4](https://arxiv.org/pdf/2511.08544v3#page=4)

## 출처와 읽은 범위

- 원문: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, Randall Balestriero·Yann LeCun, 최초 공개 2025-11-11, 검토 버전 v3(2025-11-14), 50쪽. [공식 서지·버전 이력](https://arxiv.org/abs/2511.08544).
- 첫 번째 읽기: 문제·전체 흐름·컨셉 중심으로 pp. 1–3, 17–18과 전체 절 구성을 확인했다. 두 번째 읽기: 본문 pp. 1–18, 부록 A–F pp. 24–46, 보충 Figures 16–21 pp. 47–50을 검토했다. pp. 18–23의 참고문헌은 관련 비교 원전을 선택 확인했으며, 모든 인용 논문을 전문 읽은 것은 아니다.
- 대표 그림: 이 리뷰에서 직접 제작한 한 장의 개념도, §§4–5·Algorithms 1–2에 근거. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 기본 글꼴 Light(300), 강조 Semi Bold(600). 실제 표본·성능 그래프가 아니다.
- 원문 그림: 위 각 캡션에 그림·쪽·이용 조건·변환 범위를 기록했다. 고밀도 축·범례 가독성을 위해 원래 벡터 그림을 rasterize하고 무손실 WebP로 저장했다.
- <strong>실행한 검증:</strong> 리뷰 작성자가 만든 작은 수식 예제 4개를 계산했다. 범위와 수치는 [기술 심층 리뷰](/reviews/lejepa-technical/)에 있다. 논문 실험 재현으로 간주하지 않는다.
- <strong>미실행:</strong> 공식 코드 실행, GPU 사전학습, probe 재학습, 분산 학습 검사, 논문 속도·메모리 재측정, 전체 증명의 형식 검증.
