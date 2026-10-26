---
title: "Dust: 토큰별 탐색으로 역전파에 다가가는 사전학습"
description: "활성 공간의 가상 집단과 신용 할당을 해설하고, 실제 손실·집단 외삽·계산 비용을 구분해 EGGROLL과 잇는다."
date: "2026-10-26"
publishedAt: "2026-10-26T19:34:40+09:00"
updatedAt: "2026-10-26T19:34:40+09:00"
topics: ["zeroth-order optimization", "evolution strategies", "credit assignment", "language model pretraining", "efficient training"]
visibility: "public"
lang: "ko"
translationKey: "dust-activation-search"
paperTitle: "Dust: Pretraining Transformers Without Backpropagation"
paperPublishedDate: "2026-10-05"
authors: ["Samip Dahal", "Bishwas Mandal", "Serdar Gülbahar", "Akshay Vegesna"]
year: 2026
paperUrl: "https://qlabs.sh/research/dust"
thumbnail: "/assets/reviews/dust/dust-mechanism.webp"
thumbnailAlt: "Dust의 토큰별 활성 섭동, 손실 평가, 신용 할당과 가중치 업데이트."
---

Dust는 역전파(backpropagation) 대신 토큰별 활성값(activation)을 흔들고, 그 결과 생긴 손실 변화로 학습 신호를 추정한다. FineWeb 100만 토큰 사전학습에서 명목 집단 크기 1,024의 시험 교차 엔트로피(cross-entropy)는 5.934로 역전파의 5.959보다 낮지만, 2,000만 토큰에서는 가장 큰 집단도 4.802로 역전파의 4.633에 미치지 못한다. [Table 1](https://qlabs.sh/research/dust/appendix.html#tab-1) **리뷰 해석.** 이 연구의 설득력은 계산량을 더 써서 역전파에 가까운 학습을 실제로 구현한 데 있으며, 오늘의 학습 비용을 줄이는 해법이라는 근거는 아직 약하다.

가중치 공간을 탐색하는 비교 연구의 원리는 [EGGROLL 리뷰](/reviews/eggroll-low-rank-evolution/)에서 설명한다. 두 글은 독립적으로 읽을 수 있다.

## 1. 도입 (Introduction)

언어 모델은 다음 토큰의 오차를 수많은 가중치에 배분해야 한다. 이 신용 할당(credit assignment)을 역전파는 연쇄법칙(chain rule)으로 계산한다. Dust의 출발점은 같은 일을 “여러 방향으로 조금씩 움직여 보고, 손실이 줄어든 방향을 모으는 탐색”으로 할 수 있는가다. 함수값만 평가하는 영차 최적화(zeroth-order optimization)가 학습의 중심이 된다.

저자들은 연산이 풍부해질 때 미분 가능성에 맞춘 모델 설계의 제약을 벗어나고, 탐색으로 새로운 학습 경로를 찾을 수 있다고 제안한다. 기여는 활성 공간의 가상 집단(virtual population), 가중치 공간 진화 전략(evolution strategies, ES)보다 효율적인 사전학습, 큰 모델에서의 집단 효율, 역전파 방향과의 정렬이다. 현재 방법을 역전파보다 계산 효율적으로 만드는 일과 외부 프로그램·길게 반복하는 모델에 적용하는 일은 후속 과제로 남긴다. [§1](https://qlabs.sh/research/dust#introduction)

## 2. 방법 (Method)

### 2.1 활성 공간 섭동 (Activation-Space Perturbation)

가중치에 노이즈를 넣는 ES에서는 한 후보가 하나의 변형된 모델이다. 같은 후보의 가중치 변화가 시퀀스 전체에 작용한다. Dust는 선형층의 출력에 토큰마다 독립적인 가우시안 노이즈(Gaussian noise)를 더한다. 수천 개의 토큰을 처리하는 순전파(forward pass) 안에 서로 다른 탐색 방향을 함께 넣는 셈이다. 잔차 경로를 섞는 $2L$개 스칼라만 일반적인 가중치 공간 ES로 학습하고, 나머지는 활성 섭동으로 업데이트한다. $L$은 블록 수다. [§2.1](https://qlabs.sh/research/dust#activation-space-perturbation)

[![Dust의 토큰별 섭동, 손실 평가, 신용 할당, 가중치 업데이트 흐름.](/assets/reviews/dust/dust-mechanism.webp)](/assets/reviews/dust/dust-mechanism.webp)

리뷰 설명 그림. 한 번의 draw 안에서도 토큰별 신호는 따로 채점한다. 표에 적힌 집단 크기 $K$는 여러 층에 배분하는 draw 수이며, 한 번의 전체 순전파 횟수와 항상 같지는 않다.

### 리뷰 보충 유도: EGGROLL과 연결되는 한 토큰

한 토큰의 입력 $x_t\ne0$에 대해 출력 변화 $\sigma a_t$를 만드는 가중치 변화 하나는 다음과 같다.

$$
\Delta W_t=\sigma\frac{a_t x_t^\top}{\lVert x_t\rVert^2},
\qquad \Delta W_t x_t=\sigma a_t.
$$

이는 리뷰에서 도입한 설명용 식이다. 토큰 하나에서는 랭크 1 변화로 해석할 수 있지만, 다른 토큰마다 독립적으로 고른 변화가 시퀀스 전체에 적용되는 동일한 $\Delta W$가 되는 것은 아니다. **리뷰 해석.** Dust는 EGGROLL의 저랭크 섭동을 단순히 다른 위치에 구현한 방법보다, 토큰별 탐색과 보상을 허용한 다른 신용 할당 설계로 보는 편이 정확하다. 가중치 전체에 일관된 섭동을 적용하는 방식은 [EGGROLL §4](/reviews/eggroll-low-rank-evolution/#4-eggroll)에 연결된다.

### 2.2 신용 할당 (Credit Assignment)

원문의 선형층을 $y_t=W x_t$로 쓴다. $x_t$는 입력, $y_t$는 출력이며 $a_t\sim\mathcal N(0,I)$를 뽑아 $y_t+\sigma a_t$로 평가한다. $\sigma$는 노이즈 크기다. 섭동된 토큰 손실 $\ell_s$를 함께 평가한 draw들의 평균 $\tilde\ell_s$와 비교해 $c_s=\tilde\ell_s-\ell_s$를 만든다. 손실이 작을수록 좋은 방향이므로 감소량이 양의 보상이 된다.

토큰 $t$의 변화가 뒤 토큰에 영향을 줄 수 있어, 원문 Eq.1은 시간에 따른 감쇠 $\gamma$를 둔다.

$$
r_t=\sum_{s\ge t}\gamma^{s-t}c_s.
$$

$K$개의 draw에서 모은 보상 가중 노이즈는 출력 오차의 추정치이고, 여기에 순전파에서 얻은 입력을 외적하면 가중치 업데이트 방향이 된다. 원문 Eq.2–3이다.

$$
\hat g_t=-\frac{1}{K\sigma}\sum_{i=1}^{K}r_t^{(i)}a_t^{(i)},
\qquad
\widehat G_W=\sum_t\hat g_t x_t^\top.
$$

$\hat g_t$는 출력 폭의 벡터이며, $\widehat G_W$의 차원은 $W$와 같다. 역전파도 출력 오차와 입력의 외적을 사용하지만, Dust는 그 오차를 네트워크를 거슬러 미분해서 구하지 않는다. 임베딩에서는 입력이 원핫(one-hot)이므로 해당 토큰 행에 더한다. 평균 손실을 최적화하는 구현은 유효 토큰 수로 나눈다. [§2.2, Eq.1–3](https://qlabs.sh/research/dust#credit-assignment)

### 2.3 간섭과 튜닝 (Interference and Tuning)

여러 위치를 동시에 흔들면 한 위치의 보상에 다른 섭동의 영향도 섞인다. 저자들은 이를 간섭(interference)으로 보고 세 장치를 둔다. 블록·층 종류별로 별도 평가하고 깨끗한 순전파를 캐시해 해당 블록 이후만 다시 계산한다. 어텐션(attention)의 query·key·value·gate·value embedding은 각각 흔든다. 언어 모델 출력층(head)은 캐시한 로짓(logits)의 어휘 구간별 교차 엔트로피만 재계산해 큰 집단을 싸게 평가한다.

어텐션 내부는 토큰 손실에 주는 변화가 너무 약해, 먼저 추정한 어텐션 출력 오차 $\hat g_s$를 목표로 삼는다. 출력 변화 $\Delta o_s$와의 정렬을 원문 Eq.4로 채점한다.

$$
c_s=-\langle\hat g_s,\Delta o_s\rangle.
$$

이 점수에 Eq.1의 시간 보상을 적용하며 head별로 계산한다. 직접 손실과 query에는 $\gamma=0$, 후속 토큰에서 읽히는 key·value·gate·value embedding에는 보통 0.98을 쓴다. 1,000만·2,000만 토큰의 16k 집단에서는 0.99다. [§2.3](https://qlabs.sh/research/dust#interference-and-tuning), [부록 E](https://qlabs.sh/research/dust/appendix.html#population-allocation)

노이즈·배분·감쇠는 소규모 학습 탐색으로 고르거나, 한 배치에서 역전파 기울기와의 코사인 유사도(cosine similarity)가 큰 후보를 고른 뒤 학습으로 확인한다. 실제 최종 설정에는 후자의 진단이 사용된다. 공개 최소 구현은 자동미분 없이 매개변수의 gradient 필드에 추정치를 넣고 경사 하강 최적화기를 호출한다. 따라서 업데이트 과정의 역전파 부재와, 설정을 찾는 과정에서 역전파를 참고하는 것은 별개다. [§2.3, 부록 F](https://qlabs.sh/research/dust/appendix.html#tuning), [공식 구현](https://github.com/qlabs-eng/dust/blob/b20f7c03eac630b6441ba6f254128c1761dc45ad/dust.py#L353)

## 3. 역방향 패스 없는 사전학습 (Pretraining Without a Backward Pass)

### 3.1 설정 (Setup)

기본 모델은 8블록·폭 512·약 3,770만 매개변수의 decoder-only Transformer다. 평균제곱근 정규화(root mean square normalization, RMS normalization), 회전 위치 표현(rotary position embeddings), 제곱 ReLU를 쓰는 다층 퍼셉트론(multilayer perceptron, MLP), 일부 블록의 value embedding을 사용한다. FineWeb을 바이트 쌍 인코딩(byte pair encoding, BPE)의 어휘 4,096개로 토큰화하고, 길이 2,048인 시퀀스 8개를 한 배치로 삼아 한 epoch 학습한다. 주 실험은 일정 학습률의 모멘텀 확률적 경사 하강법(stochastic gradient descent, SGD)이며, 각 조건은 seed 3개다. 검증·시험은 각각 544개 시퀀스이고, 검증 손실이 가장 좋은 체크포인트의 시험 손실을 보고한다. [§3.1](https://qlabs.sh/research/dust#setup), [부록 A](https://qlabs.sh/research/dust/appendix.html#architecture)

지표는 정답 토큰에 준 확률의 음의 자연로그를 평균한 교차 엔트로피이며 단위는 nats/token, 낮을수록 좋다. 원문의 EGGROLL-Transformer는 동일 Transformer를 가중치 공간 ES로 학습한 비교군이다. EGGROLL 원 논문의 정수 RNN EGG와는 모델·토크나이저·데이터가 다르다.

Dust의 $K$는 토큰 손실로 채점한 draw 수다. 어텐션 내부·head·잔차 스칼라 평가는 추가되며, 블록 이후 재계산은 전체 순전파보다 싸다. 1M·SGD·$K=16,384$에서는 writer와 MLP hidden에 각각 7,040, attention output에 1,664, embedding에 640을 배분한다. head에는 별도로 65,536 draw가 있다. Adam의 직접 손실 draw는 같은 명목 $K$에서도 $11K/8$이다. 20M·16k는 어텐션 내부에 더 배분해 직접 손실 draw가 14,336이다. 원문은 256 이상에서 Dust의 계산이 같은 명목 집단의 EGGROLL보다 적다고 보고한다. [부록 E, Tables 3–4](https://qlabs.sh/research/dust/appendix.html#population-allocation)

### 3.2 주요 결과 (Main Results)

아래 표는 원문 Table 1에서 작은 집단, 역전파와의 교차점 부근, 최대 집단을 고른 것이다. 제안법보다 나은 역전파 결과도 유지했다. 값은 검증으로 선택한 체크포인트의 시험 손실 평균이다.

| 방법 / 명목 집단 | 100k 토큰 | 1M 토큰 | 10M 토큰 | 20M 토큰 |
| --- | ---: | ---: | ---: | ---: |
| Dust / 64 | 7.240 | 6.290 | 5.397 | 5.265 |
| Dust / 1,024 | 7.151 | 5.934 | 5.111 | 4.960 |
| Dust / 16,384 | 7.170 | 5.916 | 5.049 | 4.802 |
| EGGROLL-Transformer / 16,384 | 7.263 | 6.708 | 6.033 | 5.865 |
| 역전파 | 7.216 | 5.959 | 4.989 | 4.633 |

[Table 1](https://qlabs.sh/research/dust/appendix.html#tab-1)

[![Dust, EGGROLL-Transformer, 역전파의 실제 시험 손실. 가로축은 명목 집단 크기이며 100k 패널의 세로 범위만 다르다.](/assets/reviews/dust/dust-measured-loss.webp)](/assets/reviews/dust/dust-measured-loss.webp)

원문 Table 1의 모든 유한 집단 행을 다시 그린 그래프. 점은 seed 3개의 평균이며 새 실험이 아니다. 작은 데이터에서는 Dust가 역전파보다 낮지만, 데이터가 늘면 더 큰 집단이 필요하다. 100k에서는 집단을 계속 늘려도 손실이 단조 감소하지 않는다. 원문 Figure 1의 학습 곡선은 큰 집단의 Dust가 학습 중에도 역전파에 가깝고, EGGROLL 비교군은 더 일찍 뒤처지는 경향을 보여준다. [Figures 1–2](https://qlabs.sh/research/dust#main-results)

“1,000–10,000배” 주장의 근거는 실측 시간비가 아니라 **EGGROLL이 Dust의 64 draw 손실에 도달할 집단 크기를 외삽한 비율**이다. 측정 구간에서는 EGGROLL의 16k가 Dust 64의 256배 집단을 쓰고도 따라잡지 못한다. 부록 D는 멱법칙과 로그 집단에 대한 직선으로 외삽하며, 1M에서 전체 멱법칙은 23,000배, 큰 세 점 직선은 8,200배를 예측한다. 한 점을 제외해 다시 맞추면 어떤 경우는 유한 교차점도 사라진다. [부록 D, Table 2](https://qlabs.sh/research/dust/appendix.html#eggrolls-extrapolated-population)

20M에서 Dust의 무한 집단 손실 추정은 4.431, seed bootstrap의 95% 구간은 3.89–4.58이다. 측정된 최대 집단에서도 곡선이 내려가므로 저자들도 이를 측정된 한계값으로 보지 않는다. **리뷰 해석.** 이 외삽은 더 큰 집단을 실험할 이유를 제공하지만, 4.431을 달성한 학습 결과나 확정된 역전파 우위로 취급할 수는 없다.

### 3.3 Adam을 쓴 Dust (Dust Under Adam)

1M 토큰에서 Adam으로 다시 튜닝하면 Dust와 역전파는 개선되고, EGGROLL 비교군은 각 집단에서 SGD와 0.01 이내의 차이다. Figure 3의 역전파 손실은 5.361이며, Dust의 무한 집단 적합값은 5.248, 95% 구간은 5.17–5.30이다. Dust·역전파는 seed 5개, EGGROLL은 3개다. 이 적합값도 관측점과 구분해야 한다. Adam의 embedding draw 증량과 학습률 그룹은 부록 E·F의 Tables 3·7에 제시된다. [§3.3, Figure 3](https://qlabs.sh/research/dust#dust-under-adam)

## 4. 고차원 공간에서의 탐색 (Search in High-Dimensional Space)

### 4.1 과매개변수화 (Overparameterization)

10M 토큰을 고정하고 2.0M·7.3M·38M·243M 모델을 비교한다. 예를 들어 집단 1,024에서 시험 손실은 5.265 → 5.161 → 5.158 → 5.214다. 크기 증가가 노이즈 추정에 불리하더라도 표현력의 이득이 그것을 상쇄할 수 있다. 다만 가장 큰 모델이 항상 최선은 아니며, 16k에서 38M은 5.036, 243M은 5.086이고 각 역전파 기준선은 5.015·5.048이다. [§4.1, Figure 4](https://qlabs.sh/research/dust#overparameterization)

이 연구는 §3의 주 실험보다 이전 설정이다. embedding 학습률과 key·value 학습률이 다르므로, 같은 크기의 모델이라고 §3 표와 직접 행을 합치지 않는다. 저자들은 두 방법 모두 크기별·집단별로 설정을 선택한다. [부록 F, Table 6](https://qlabs.sh/research/dust/appendix.html#tuning)

### 4.2 역전파와 유사한 기울기의 출현 (Emergence of Backprop-Like Gradients)

저자들은 같은 배치의 Dust 추정치와 역전파 기울기 사이 코사인을 층별로 측정한다. 1이면 같은 방향, 0이면 직교, −1이면 반대 방향이며, 크기와 학습 성능은 별도다. 측정 대상은 **역전파로 10M·100M·1B 토큰까지 학습한 체크포인트**다. Dust 자체를 1B 토큰까지 사전학습했다는 실험은 아니다. 집단 범위는 64–131,072이고, 원문 Eq.5의 적합식은 다음과 같다. 부록 B의 Figure 6은 초기화 시점의 Dust와 초기화·100M 시점의 EGGROLL 진단도 보인다.

$$
\cos(K)=\frac{c_{\max}}{\sqrt{1+c/K}}.
$$

$c_{\max}$는 적합된 방향 정렬의 상한, $c$는 그 상한의 $1/\sqrt2$에 도달하는 집단 예산이다. [§4.2, Figure 5](https://qlabs.sh/research/dust#emergence-of-backprop-like-gradients)

[![100M 토큰 역전파 체크포인트에서 측정한 층별 코사인 적합식. head와 key는 필요한 집단 예산이 크게 다르다.](/assets/reviews/dust/dust-cosine-fits.webp)](/assets/reviews/dust/dust-cosine-fits.webp)

원문 Table 8의 층 종류 여섯 개를 선택해 측정 범위 안에서 다시 그린 적합 곡선이다. head의 $c$는 759, key는 616,595로 큰 차이가 있다. value gate의 상한은 약 0.666, token embedding은 약 0.732다. **리뷰 해석.** 집단 평균으로 줄어드는 분산과, 섭동 크기·감쇠·국소 대리 목표에서 생기는 방향 편향을 나눠 보는 것이 후속 연구의 출발점이다. $K$만 늘리면 모든 층이 정확한 역전파 방향으로 간다는 결론은 이 진단에서 나오지 않는다.

## 5. 결론 (Conclusion)

저자들은 역전파에 맞춰 공진화한 구조·최적화기·하드웨어를 넘어, 풍부한 계산으로 신용 할당을 탐색할 수 있다고 결론짓는다. 과매개변수화도 탐색에 활용할 공간과 기하를 제공하는 방향으로 해석한다. 계산 효율 개선, 새로운 구조, 학습·최적화기의 공동 설계가 앞으로의 과제다. [§5](https://qlabs.sh/research/dust#conclusion)

## 6. 관련 연구 (Related Work)

원문의 구분은 세 갈래다. 가중치 섭동 계열에서 MeZO는 큰 모델의 미세조정을 다루며, EGGROLL은 저랭크 섭동으로 평가 비용을 줄인다. 저자들은 초기화에서 학습한 기존 영차 언어 모델 작업의 크기·역전파 대비 결과와, 이미 학습된 모델 주변에서 전문가를 찾는 연구를 구분한다. Dust의 과제는 표현 자체를 처음부터 만드는 Transformer 사전학습이다.

활성 섭동 계보는 node perturbation과 GEMINI에 이어진다. Zoop은 출력 섭동과 국소 미분을 미세조정에 사용한다. forward-gradient 계열은 순방향 자동미분을 사용하고, predictive coding 계열은 국소 미분과 반복적인 국소 동역학으로 신용을 전파한다. Dust는 순전파 섭동과 토큰별 보상·어텐션의 국소 목표를 결합한다. 이 구분이 원문의 “최초로 경쟁력 있는 영차 Transformer 사전학습” 주장이 비교하는 범위다. [§6](https://qlabs.sh/research/dust#related-work)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 가장 중요한 방향 전환은 더 많은 개체를 돌리는 데서, 한 번의 평가에서 어느 위치에 얼마나 유용한 신호를 얻을지 설계하는 쪽으로 이동한 것이다. Dense token loss가 있는 사전학습에서는 이 선택이 효과적이다. 반면 출력 오차의 외적, 인과적 attention 구조, head별 채점과 감쇠 선택이라는 구조적 지식을 사용한다. 풍부한 계산의 성공과 유용한 귀납 편향의 성공을 함께 보여주는 연구다.

**리뷰 해석.** “최초”라는 표현은 해당 시점·과제·비교 설정에서의 저자 주장으로 지지되지만, 모든 역전파 대안에 대한 우선권 조사까지 제시되지는 않는다. 역전파보다 낮은 손실이 나온 작은 데이터 조건과, 아직 뒤처지는 큰 데이터 조건을 함께 남기는 것이 연구의 현재 위치를 가장 잘 설명한다. 최적화 튜닝 비용까지 포함한 동일 GPU 시간의 경쟁력은 별도 검증이 필요하다.

### EGGROLL과 잇는 연구 방향: 정보가 부족한 부분에 예산 배분하기

**리뷰 해석.** [EGGROLL의 하드웨어 경로](/reviews/eggroll-low-rank-evolution/#42-하드웨어-효율적인-구현-hardware-efficient-implementation)는 집단 하나의 비용을 줄이고, Dust는 평가당 신용 정보를 늘린다. 결합의 첫 실험은 같은 Transformer·같은 FineWeb 배치에서 가중치 ES, Dust, 층별 혼합 방식을 **같은 GPU 시간**으로 비교하는 것이다. 시험 손실과 함께 층별 코사인·추정치 분산·노이즈 크기·캐시 메모리·튜닝 시간을 기록하면, 신호가 약한 key나 embedding에 예산을 더 주는 정책을 검증할 수 있다.

**리뷰 해석.** 외부 프로그램이나 정수 모듈처럼 불연속적이거나 토큰별 보상이 없는 부분은 EGGROLL식 전역 섭동으로, dense loss를 얻는 선형 블록은 Dust로 학습하는 혼합은 아직 가설이다. 가장 작은 반증 실험은 같은 모델에서 토큰 손실을 마지막 토큰의 희소 보상으로 바꾸는 것이다. 국소 보상 이득이 사라지면 혼합의 역할이 드러난다. 원문의 RL·집합 단위 목적 사례는 [EGGROLL 부록 L 설명](/reviews/eggroll-low-rank-evolution/#l-추가-transformer-후학습-fine-tuning-pretrained-transformer-llms-with-verifiable-rewards)에서 이어진다.

## 출처와 읽은 범위

- 원문: Samip Dahal, Bishwas Mandal, Serdar Gülbahar, Akshay Vegesna, *Dust: Pretraining Transformers Without Backpropagation*. [공식 본문](https://qlabs.sh/research/dust)·[부록 A–G](https://qlabs.sh/research/dust/appendix.html), 2026-10-26 열람본. 공식 페이지는 October 2026으로 표시하며, 공개일은 홈페이지가 연결한 [저자 발표(2026-10-05)](https://x.com/industriaalist/status/2107194534501433804)를 기준으로 한다.
- 읽은 범위: 본문 §1–6과 부록 A–G. 구현 설명은 [공식 최소 구현의 고정 커밋](https://github.com/qlabs-eng/dust/tree/b20f7c03eac630b6441ba6f254128c1761dc45ad)을 참고한다. EGGROLL의 방법·실험은 [별도 리뷰](/reviews/eggroll-low-rank-evolution/)와 연결한다.
- 그림: 설명도는 자체 제작. 시험 손실은 Table 1, 코사인 곡선은 Table 8의 수치를 바탕으로 재구성했으며, 새로운 측정 결과가 아니다. WebP로 인코딩했다.
- 확인하지 않은 것: 대규모 사전학습과 전체 하이퍼파라미터 탐색의 독립 재현.
