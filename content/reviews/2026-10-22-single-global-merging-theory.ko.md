---
title: "[2/2] Single Global Merging 심화: 부록 실험과 증명의 조건을 풀어 읽기"
description: "부록 A–D를 따라 실험 설정과 추가 비교를 보존하고, 평균 업데이트·Taylor 전개·수렴 증명의 간극을 검토한다."
date: "2026-10-23"
publishedAt: "2026-10-23T12:14:58+09:00"
updatedAt: "2026-10-23T15:26:24+00:00"
topics: ["decentralized learning", "model merging", "distributed optimization", "federated learning", "communication efficiency"]
visibility: "public"
paperTitle: "On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning"
paperPublishedDate: "2025-07-09"
authors: ["Tongtian Zhu", "Tianyu Zhang", "Mingze Wang", "Zhanpeng Zhou", "Can Wang"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2507.06542v4"
thumbnail: "/assets/reviews/single-global-merging/thumbnail.webp"
lang: "ko"
translationKey: "single-global-merging-theory"
thumbnailAlt: "원문 Figure 1(b): sparse 분산 학습의 마지막 가중치 평균에서 전역 테스트 성능이 크게 회복되는 ResNet-18 실험."
---

원문 부록은 본문 관찰을 optimizer·참여자 수·topology·초기화·데이터 이질성으로 넓히는 실험(A–C)과, 분산 SGD(decentralized SGD, DSGD)의 평균 모델이 parallel SGD의 수렴률에 맞출 수 있다는 증명(D)을 담는다. Tiny ImageNet·32명 ResNet-18 설정의 Figure C.6에서는 학습 끝의 완전 평균 대신 exponential graph에서 gossip 혼합을 한 번만 해도 평균 전역 테스트 정확도가 그림상 약 20%에서 43%로 올라, 완전한 최종 전역 병합(global merging)의 약 48%에 가깝다. [원문 C.3.4·Figure C.6](https://arxiv.org/pdf/2507.06542v4#page=27) **리뷰 해석.** 부록 실험은 병합 이득이 여러 설정에서 반복됨을 보이지만, 증명은 공통 분포를 가정한 원전의 상쇄를 비IID(non-IID) 손실에 그대로 옮겨 1차 이질성 항을 놓치고(D.3) 기대 제곱 상한을 실행별 조건에 대입하므로(D.9), 현재 증명으로 수렴률 일치가 확인되는 범위는 동질적 손실과 추가 확률 논증을 전제한 경우다.

[[1/2] 본문·핵심 실험 해설](/reviews/single-global-merging/) 뒤에 읽기를 권하며, 원문 **부록 A→B→C→D**와 D 안의 Lemma D.1–Proposition D.9 순서를 따른다. 중간 계산과 반례에는 **리뷰 보충**이라고 표시한다.

## A. 한계와 예상 질문 (Limitations and Potential Questions)

부록 A는 일곱 질문에 저자가 답하는 형식이다. [A](https://arxiv.org/pdf/2507.06542v4#page=19)

| 원문의 질문 | 저자의 답변 요지 |
| --- | --- |
| 이론은 SGD인데 왜 AdamW를 쓰는가? | 비IID에서 성능이 좋아 일부 실험에 썼고, SGD에서도 모든 관찰이 같다. |
| 이론이 병합 가능성을 어떻게 설명하는가? | 평균 모델이 parallel SGD의 수렴률을 따르고 $m$명 parallel SGD는 단일 로컬 모델보다 빠르므로, 평균 모델이 어느 개별 모델보다 낫다는 것이 전이적으로 정당화된다. |
| 하이퍼파라미터는 어떻게 골랐는가? | 전역 테스트 성능 기준 grid search다. |
| 초기화는 Model soups와 어떻게 다른가? | 실험 대부분이 서로 다른 무작위 초기화를 쓰며, 공통 사전학습 상태에서도 이득이 나타난다. |
| 손실 지형 그림은 어떻게 만들었는가? | 16개 모델을 정16각형 꼭짓점에 두고 Wachspress 좌표로 내부를 보간했다. 격자는 무작위 방향이 아니라 모델들 자체로 정해진다. |
| 실험이 비전 과제에 한정되는가? | 그렇다. 기존 분산 학습 문헌 대부분과 같으며 확장은 향후 과제다. |
| 적은 통신에서도 모델이 비슷한 상태로 수렴한다는 기존 gossip 관찰과 같은가? | 이질적 데이터에서는 한 합의점에 도달하지 않은 채 높은 손실 고리에 놓이면서도 병합 가능하다. |

**리뷰 해석.** 세 답변에는 조건이 남는다. 하이퍼파라미터를 테스트 성능으로 골랐으므로 보고된 정확도에는 선택 편향이 섞일 수 있다. 실험 대부분이 서로 다른 초기화를 쓴다는 답변은, 공통 초기화 $\theta_k^{(0)}=\theta^{(0)}$를 가정하는 Theorem 1·D.7이 그 실험들을 직접 다루지 않는다는 뜻이기도 하며(아래 D.2의 초기 항 참조), AdamW 실험에 대한 수렴 증명도 없다. “전이적” 답변은 평균 정상성(stationarity)의 상한끼리 비교한 것이어서, 마지막 테스트 정확도의 우열이나 Definition 2의 손실 부등식을 증명하지 않는다.

## B. 배경과 관련 연구 보충 (Additional Background and Related Work)

B.1은 그래프·비동기 업데이트·데이터 이질성·프라이버시·대형 모델 훈련으로 이어지는 분산 학습의 흐름을, B.2는 암묵적 편향(implicit bias)을, B.3은 모드 연결성(mode connectivity)과 모델 병합(model merging)을 정리한다. 아래는 이 논문의 논증과 이어지는 부분이다. [B](https://arxiv.org/pdf/2507.06542v4#page=21)

**B.1:** 로컬 계산과 gossip의 결합은 오래된 틀이며, 변화하는 topology와 local steps, 비IID를 다룬 통합 수렴 분석(Koloskova 등, 2020)이 이미 있다.

**B.2:** 분산 SGD와 중앙 SGD의 차이를 다룬 연구 가운데, 이 논문이 직접 의존하는 [Zhu et al. 2023](https://proceedings.mlr.press/v202/zhu23e.html)은 분산 SGD를 무작위 첨예도 인식 최소화(sharpness-aware minimization, SAM)와 연결해 평탄성 편향을 보였다. 저자들은 그 연구가 IID와 매우 큰 batch에 초점을 둔 반면 이 논문은 비IID와 표준 batch를 다룬다고 구분한다. 공통 분포 가정이 증명에서 하는 역할은 아래 D.3에서 확인한다.

**B.3:** 기존 모드 연결성은 공통 초기화, 동질적 데이터, 뉴런 순열 정렬(permutation alignment) 가운데 하나가 있을 때 관찰됐다. 저자들은 서로 다른 초기화·이질적 데이터·순열 정렬 없음의 조합에서도 병합 가능성이 나타난다는 점을 차별점으로 든다.

## C. 추가 실험 (Additional Experiments)

본문 그림은 누르면 큰 이미지로 열어 축과 범례를 확인할 수 있다.

### C.1 실험 설정 (Experimental Setups)

| 항목 | 원문 설정 |
| --- | --- |
| 데이터 | CIFAR-10/100, Tiny ImageNet; 핵심 그림은 CIFAR-100와 Tiny ImageNet |
| 참여자 수 | 주 Figure 1은 32, sliding-window와 landscape는 16; 부록에서 16/32 비교 |
| 이질성 | 주 설정 Dirichlet $\alpha=0.1$; C.5에서 $\alpha=1.0$과 비교 |
| 학습 길이 | 300 통신 라운드 × 라운드당 100 local mini-batch updates |
| 모델 | scratch ResNet-18, pretrained CLIP ViT-B/32 |
| batch size·weight decay | 기본 128, $5\times10^{-4}$ |
| SGD learning rate | ResNet $10^{-2}$, CLIP $10^{-3}$ |
| AdamW learning rate | ResNet $5\times10^{-4}$, CLIP $10^{-5}$, C.1 기재 기준 |
| augmentation | padding 4 RandomCrop, RandomHorizontalFlip, RandAugment: 2 operations, magnitude 9 |
| sparse 통신 | $R=0.2$: 확률 0.2로 한 random peer; 비교에 $R=1$, ring, exponential graph |
| 계산 자원 | 80GB A100 시설. 16모델 시뮬레이션 예: ResNet 약 15GB·8시간, CLIP 약 30GB·12시간 |

Tiny ImageNet의 $64\times64$ 입력을 patch size 32로 처리하므로 CLIP에는 4개 image patches와 CLS, 총 5개 토큰이 들어간다. 저자도 Tiny ImageNet을 full ImageNet보다 가벼운 중간 규모 벤치마크로 소개하며, 실행 시간은 한 GPU에서 분산 학습을 흉내 낸 중앙 시뮬레이션 기준이다.

원문은 참여자당 4096개 표본을 적는다. 이를 서로 겹치지 않는 고정 소유로 읽으면 32명에서 $32\times4096=131{,}072$개가 되어 Tiny ImageNet의 100,000장을 넘는다. [공개 코드의 고정 스냅샷](https://github.com/Raiden-Zhu/ICLR-2026-Grokking-in-Decentralized-Learning/tree/6a5c139c2cdf6d6d7e0b6ba3c00f123e494e6660)은 원문용 resample 모드에서 참여자별 클래스 비율을 유지한 채 iterator를 만들 때마다 표본을 다시 뽑고, sampler는 복원 추출(sampling with replacement)을 쓴다. 고정·상호 배타적인 장치별 데이터셋과는 다른 설정이다.

또한 A의 CLIP AdamW 탐색 집합 $\{10^{-4},5\times10^{-4},10^{-3}\}$에는 C.1이 최적값으로 적은 $10^{-5}$가 없다. 공개 코드는 최종 학습 가능 매개변수를 평균한 뒤 BatchNorm 통계를 다시 추정한다고 설명한다. [C.1](https://arxiv.org/pdf/2507.06542v4#page=24)

### C.2 평가 지표 (Practical Evaluation Metrics)

C.2는 병렬·연합 학습의 표준 지표인 전역 모델 정확도와, 분산 학습에서 쓰는 두 지표(자기 분포에서의 평균 로컬 정확도, 전역 분포에서의 평균 로컬 정확도)를 정의한다. 저자들은 비IID에서 로컬 모델이 크게 갈라지므로 후자를 주 지표로 삼는다. 세 평가와 병합 전후 점프의 관계는 [[1/2] §3.1](/reviews/single-global-merging/)에서 다룬다. 가중치 평균은 출력 앙상블(output ensemble)과 달리 추론 때 하나의 모델만 남긴다. [C.2](https://arxiv.org/pdf/2507.06542v4#page=24)

### C.3 추가 비교와 ablation

| 원문 그림 | 바꾼 조건 | 관찰 |
| --- | --- | --- |
| C.1·C.2 | 16/32 agents, AdamW | AdamW에서도 병합 전후 차이가 나타난다. |
| C.3 | $R=0.2$와 $R=1$, ring·exponential | 통신량과 topology를 바꿔도 병합 이득이 남고, 크기는 설정마다 다르다. |
| C.4(a,b) | batch 64/128, learning rate $5\times10^{-4}/10^{-4}$ | 두 값씩 바꾼 범위에서 차이가 유지된다. |
| C.4(c) | 같은/다른 초기화 | 서로 다른 초기화에서도 비슷하게 회복된다. |
| C.5 | CIFAR-100/Tiny ImageNet, $\alpha=0.1/1$, 두 graph | 이질성이 약할수록 평균과 개별 모델의 차이가 작다. |
| C.6 | 마지막 exponential gossip 1/5회와 완전 전역 평균 | 제약된 연결망의 gossip으로도 대부분 회복된다. |

#### C.3.1 참여자 수와 optimizer

C.1(16명)과 C.2(32명)는 CLIP과 ResNet에서 AdamW를 쓴 경우의 병합 전후 차이를 보여 준다. C.2에서는 최종 전역 병합 뒤 주황점이 FL보다 높다(그림상 CLIP 약 0.61 대 0.59, ResNet 약 0.49 대 0.45). 저자는 Proposition 2의 가속 가능성이 Figure C.1(a)의 AdamW 결과와 맞는다고 적는다(본문 §5.2). **리뷰 해석.** Proposition 2는 SGD에 대한 결과이므로 AdamW 곡선이 정리를 직접 검증하지는 않는다.

#### C.3.2 이웃 수와 topology

C.3(a)는 사전학습 가중치를 쓴 ResNet에서 $R=0.2$와 $R=1$을 비교한다. 더 자주 통신하는 $R=1$은 병합 전 성능이 높고, $R=0.2$는 최종 전역 병합 때의 증가 폭이 더 크다. 병합 이득이 크다는 것과 병합 후 정확도가 높다는 것은 다른 비교다. C.3(b)에서는 exponential graph가 ring보다 개별·평균 모델 모두 조금 높고, 두 graph 모두 평균의 이득을 보인다. [C.3.1–C.3.2·Figures C.1–C.3, pp.25–26](https://arxiv.org/pdf/2507.06542v4#page=26)

#### C.3.3 하이퍼파라미터·초기화·이질성

C.4는 batch size나 learning rate를 바꿔도 평균과 개별 모델 사이의 차이가 남는지를 확인한다. C.4(c)의 같은/다른 초기화 곡선도 비슷하게 회복된다. 이어지는 C.5는 데이터와 label skew가 이 차이를 얼마나 바꾸는지 살핀다.

[![Figure C.5. 두 데이터셋과 Dirichlet 이질성, ring 및 exponential graph에서 개별 모델과 평균 모델 성능 비교.](/assets/reviews/single-global-merging/figure-c5.webp)](/assets/reviews/single-global-merging/figure-c5.webp)

*원문 Figure C.5(a–d), p.27. 가로축은 라운드, 세로축은 전역 테스트 정확도다. 파랑은 exponential graph, 주황은 ring이며, 같은 색의 진한 선은 로컬 모델, 연한 선은 평균 모델이다. (a,c)는 $\alpha=0.1$, (b,d)는 $\alpha=1$이다. 패널마다 세로축 범위가 다르다. CC BY 4.0, PDF 추출·무손실 WebP 변환.*

그림상 $\alpha=0.1$에서는 평균 모델이 개별 모델보다 exponential graph에서 약 4–5%p, ring에서 약 9–13%p 높지만, $\alpha=1$에서는 차이가 2%p 이하로 줄어든다. 저자들은 C.3.1–C.3.3 전체를 하이퍼파라미터·데이터셋·이질성·구조·optimizer·초기화·topology에 걸쳐 병합 이득이 일관되게 나타난다고 요약한다. [C.3.3·Figure C.5, p.27](https://arxiv.org/pdf/2507.06542v4#page=27)

#### C.3.4 마지막 topology-constrained merging

마지막 집계를 완전 평균으로 구현할 수 없을 때도 효과가 남는지 묻는 실험이다. 학습이 끝난 뒤 exponential graph에서 gossip 혼합을 1회 또는 5회 수행하고, 완전 전역 평균과 비교한다. 저자들은 (1) topology 제약 gossip 한 번만으로도 전역 테스트 정확도가 크게 오르고, (2) 그 성능이 무작위 통신 뒤 완전 전역 평균을 한 기준 설정과 비슷하다고 정리한다.

[![Figure C.6. 마지막 exponential gossip 1회, 5회, 완전 전역 평균의 비교.](/assets/reviews/single-global-merging/figure-c6.webp)](/assets/reviews/single-global-merging/figure-c6.webp)

*원문 Figure C.6(a–c), p.27. 가로축은 통신 라운드, 세로축은 평균 전역 테스트 정확도다. (a,b)는 마지막 단계도 exponential graph로 제한하며, (c)는 마지막 완전 평균이다. 주황색 마지막 점프의 높이를 패널끼리 비교한다. 캡션은 AdamW라고 적지만 그림 파일 이름에는 SGD가 들어 있고, (c)는 SGD로 표시된 본문 Figure 1(b)와 같은 곡선이다. CC BY 4.0, PDF 추출·무손실 WebP 변환.*

그림상 마지막 점은 1회 gossip 약 0.43, 5회 약 0.45, 완전 평균 약 0.48이다. [C.3.4·Figure C.6, p.27](https://arxiv.org/pdf/2507.06542v4#page=27) **리뷰 해석.** 저자의 “comparable”은 1회 gossip과 완전 평균 사이의 약 4%p 차이를 가리킨다. gossip 횟수가 늘수록 완전 평균에 다가가는 방향은 일관되지만, 원문이 반복 횟수·seed·오차 막대를 밝히지 않아 1회와 5회의 약 2%p 차이를 구분할 근거는 약하다.

## D. Theory

아래는 원문 부록의 정리 순서다. $\delta_k^{(t)}=\theta_k^{(t)}-\bar\theta^{(t)}$, $g_t=\nabla\mathcal L(\bar\theta^{(t)})$, $H_t=\nabla^2\mathcal L(\bar\theta^{(t)})$, $q_t=\nabla\operatorname{Tr}(H_t\Gamma^{(t)})$는 계산을 줄이기 위한 리뷰 보조 표기다. 원문 기호로 $\Gamma^{(t)}=m^{-1}\sum_k\delta_k^{(t)}\delta_k^{(t)\top}$는 $d\times d$ 모델 공분산, $\Xi_t^2=\operatorname{Tr}(\Gamma^{(t)})$는 합의 거리(consensus distance), $\Theta$는 참여자 모델을 열로 쌓은 $d\times m$ 행렬, $p$는 Assumption 1의 기대 혼합 수축률이다. $q_t$의 미분에서는 공분산을 고정한다.

### 리뷰 보충: 병합 가능성 정의의 비자명성

원문 Definition 2는 비음수 가중치가 존재해 평균 손실 부등식을 만족하는지를 묻고, 실험은 균등 평균을 평가한다. [원문 §3, Definition 2](https://arxiv.org/pdf/2507.06542v4#page=4)

**리뷰 해석.** 인쇄된 정의에는 가중치 합 1이 없고, 그 조건을 추가해도 한 가중치만 1인 선택은 등호를 만족한다. 비자명한 병합 가능성을 정의하려면 균등 평균을 고정하거나 여러 양의 가중치를 요구하는 조건이 더 필요하다. 이는 실험 결과가 아니라 정의의 구별 능력에 관한 문제다.

### Lemma D.1: 합의 거리의 재귀식

혼합 수축과 gradient 상한을 이용하면 원문은 다음을 얻는다. 여기서는 과거 상태를 고정하고 새 gradient·통신 난수에 기댓값을 취하는 조건부 표기 $\mathbb E_t$를 쓴다.

$$
\mathbb E_t\Xi_{t+1}^2
\leq\left(1-\frac p2\right)\Xi_t^2
+\frac{12(1-p)\eta^2}{p}(\phi_t^2+\sigma^2),
\quad
\phi_t^2=\frac1m\sum_k\|\nabla\mathcal L_k(\theta_k^{(t)})\|^2.
$$

증명의 구조는 로컬 업데이트로 모델 차이가 늘고, 혼합으로 줄어드는 두 효과를 분리하는 것이다. $\|a+b\|^2$의 교차항은 Young 부등식으로 나눠 수축률을 $1-p$에서 $1-p/2$로 느슨하게 만든다. local gradient와 stochastic noise를 나눠 두 번째 항을 제어한다. 조건부 기댓값을 취한 뒤 전체 기댓값으로 넘긴다. [Lemma D.1, p.28](https://arxiv.org/pdf/2507.06542v4#page=28)

**리뷰 보충: 재귀식까지의 계산.** 원문 Assumptions 1–3을 전제로, 부록의 오른쪽 행렬곱 표기를 쓰자. $G_t$는 각 열이 참여자별 stochastic gradient인 $d\times m$ 행렬이다. 평균으로 보내는 사영(projection)은

$$
J=\frac{\mathbf1\mathbf1^\top}{m},\qquad \Theta(I-J)=\Theta-\bar\Theta.
$$

다. $WJ=JW=J$이므로

$$
\Theta^{(t+1)}=(\Theta^{(t)}-\eta G_t)W,
\qquad
\Theta^{(t+1)}-\bar\Theta^{(t+1)}
=(\Theta^{(t)}-\eta G_t)(W-J).
$$

가 된다. 새 gradient와 통신 난수에 조건부 혼합 수축을 적용할 수 있는 설정에서, Young 부등식 $\|A+B\|_F^2\leq(1+a)\|A\|_F^2+(1+1/a)\|B\|_F^2$에 $a=p/2$를 넣으면

$$
\begin{aligned}
m\mathbb E_t\Xi_{t+1}^2
&\leq(1-p)\mathbb E_t
\|(\Theta^{(t)}-\bar\Theta^{(t)})-\eta G_t(I-J)\|_F^2\\
&\leq(1-p)(1+p/2)m\Xi_t^2
+(1-p)(1+2/p)\eta^2\mathbb E_t\|G_t\|_F^2.
\end{aligned}
$$

를 얻는다. 여기서는 $I-J$가 직교 사영이라 $\|G_t(I-J)\|_F\leq\|G_t\|_F$임을 썼다. 이어서 $0<p\leq1$과 gradient/noise 분해를 이용한다.

$$
(1-p)(1+p/2)\leq1-p/2,\qquad 1+2/p\leq6/p,
\qquad
\mathbb E_t\|G_t\|_F^2\leq2m(\phi_t^2+\sigma^2).
$$

따라서 오른쪽 계수는 $(1-p/2)m\Xi_t^2+12m(1-p)\eta^2(\phi_t^2+\sigma^2)/p$로 제어된다. 양변을 $m$으로 나누면 앞의 조건부 재귀식이다. 전체 기댓값을 취한 표기로 옮기면 $\Xi_t^2$와 $\phi_t^2$에도 기댓값이 붙는다. 상수 12는 날카로운 최적 상수가 아니라 이 느슨한 분해에서 나오는 상한이다.

### Corollary D.2: 재귀식을 풀어 통신량과 연결하기

**리뷰 보충 유도.** $\rho=1-p/2$이고 $\phi_t^2\leq\phi^2$라면 재귀식을 반복해

$$
\begin{aligned}
\mathbb E\Xi_t^2
&\leq\rho^t\Xi_0^2+
\frac{12(1-p)\eta^2}{p}(\phi^2+\sigma^2)
\sum_{s=0}^{t-1}\rho^s\\
&\leq\rho^t\Xi_0^2+
\frac{24(1-p)\eta^2}{p^2}(\phi^2+\sigma^2)
\end{aligned}
$$

를 얻는다. 마지막은 $\sum_{s\geq0}\rho^s=1/(1-\rho)=2/p$를 사용했다. 원문 Corollary D.2의 진술은 초기화 조건을 두지 않고, 증명은 반복을 풀 때 첫 항 $\rho^t\Xi_0^2$ 없이 둘째 항만 남긴다. [Corollary D.2, p.29](https://arxiv.org/pdf/2507.06542v4#page=29) **리뷰 해석.** 따라서 인쇄된 상한 $24(1-p)\eta^2(\phi^2+\sigma^2)/p^2$는 Theorem 1처럼 모든 참여자가 같은 값에서 시작해 $\Xi_0=0$일 때 성립하고, 서로 다른 초기화에서는 $\rho^t\Xi_0^2$를 더해야 하며 학습 초반에는 이 항이 상한을 지배한다.

원문은 gradient가 천천히 변하는 경우에 최근 gradient로 쓴 국소 상한도 제시하며, 여기서도 $\rho^t\Xi_0^2$ 항은 빠져 있다. 증명은 다음 두 식을 이용한다.

$$
\phi_s^2\leq(1+p/4)\phi_{s+1}^2,\qquad
(1-p/2)(1+p/4)=1-p/4-p^2/8\leq1-p/4
$$

과거 gradient를 현재 근처 값으로 비교하면서 얻는 두 기하 인자를 곱해도 $1-p/4$ 이하로 수축한다. 따라서 기하급수 합의 상한이 $2/p$ 대신 $4/p$가 되어

$$
\mathbb E\Xi_t^2
\leq\frac{48(1-p)\eta^2}{p^2}
(\mathbb E\phi_{t-1}^2+\sigma^2).
$$

라는 국소 상한을 얻는다. 원문은 $\phi_{t-1}^2$에 기댓값을 붙이지 않지만, 여기서는 전체 기댓값 표기에 맞춰 붙였다. 균일 상한 $\phi$ 대신 최근의 $\phi_{t-1}$로 학습 상태를 반영하는 식이다.

### Proposition D.3: 평균 업데이트와 Gaussian smoothing

이중 확률 혼합과 SGD에서는, 과거까지의 상태를 고정한 평균의 실제 기대 업데이트가

$$
\mathbb E_t\bar\theta^{(t+1)}
=\bar\theta^{(t)}-\frac\eta m
\sum_k\nabla\mathcal L_k(\theta_k^{(t)})
$$

다. 원문은 [Zhu et al. 2023](https://proceedings.mlr.press/v202/zhu23e.html)의 결과를 이용해 이를 $\epsilon\sim\mathcal N(0,\Gamma^{(t)})$에 대한 전역 gradient 평균으로 바꾸고 3차 잔차를 남긴다. Gaussian은 구현에서 추가로 주입하는 noise가 아니라 모델 차이의 공분산을 맞춘 분석 도구다. [D.3, p.30](https://arxiv.org/pdf/2507.06542v4#page=30)

원문 Proposition D.3이 주장하는 관계를 식으로 남기면 다음과 같다. 여기서 $\delta^{(t)}$는 개별 편차 $\delta_k^{(t)}$와 다른, 업데이트 단위의 잔차다.

$$
\begin{aligned}
\mathbb E_t\bar\theta^{(t+1)}
&=\bar\theta^{(t)}
-\eta\mathbb E_{\epsilon^{(t)}\sim\mathcal N(0,\Gamma^{(t)})}
\nabla\mathcal L(\bar\theta^{(t)}+\epsilon^{(t)})
+\delta^{(t)},\\
\delta^{(t)}
&=\mathcal O\!\left(\frac{\eta}{m}\sum_k
\|\theta_k^{(t)}-\bar\theta^{(t)}\|^3\right).
\end{aligned}
$$

아래에서는 이 관계가 기대하는 상쇄를 먼저 설명한 뒤, 비IID에서 어떤 항이 남는지 확인한다.

**왜 IID가 중요한가 — 리뷰 보충.** 모든 로컬 손실이 같은 $\mathcal L$이면

$$
\frac1m\sum_k\nabla\mathcal L(\bar\theta+\delta_k)
=g+\frac12\nabla\operatorname{Tr}(H\Gamma)+r,
\quad
\|r\|\leq\frac{L_4}{6m}\sum_k\|\delta_k\|^3.
$$

선형항은 $H(m^{-1}\sum_k\delta_k)=0$으로 지워진다. 2차 gradient 항, 즉 손실의 3차 미분 항이 공분산과 결합한다. 평균과 공분산을 맞춘 Gaussian에도 같은 2차 항이 남지만, 두 분포의 3차 모멘트가 같아지는 것은 아니다.

반면 비IID에서는 Hessian도 참여자마다 다르므로

$$
\frac1m\sum_k H_k(\bar\theta)\delta_k
$$

가 일반적으로 0이 아니다. 원전 Zhu 2023의 C.2는 기대 Hessian을 공통 분포의 Hessian으로 바꾸는 단계를 실제로 사용한다. 이 논문의 Proposition D.3에 적힌 전역 $C^4$ 조건만으로는 그 취소를 보장하지 않는다.

**구체적 반례.** 두 참여자의 1차원 손실을

$$
\mathcal L_1(x)=1-\cos x,\quad
\mathcal L_2(x)=2(1-\cos x),\quad
\theta_1=b+a,\quad\theta_2=b-a
$$

라 두면 전역 손실은 $\mathcal L(x)=\tfrac32(1-\cos x)$다. 모든 1–4차 미분과 gradient heterogeneity가 유계다. 그런데

$$
\begin{aligned}
\frac12\bigl[\mathcal L_1'(b+a)+\mathcal L_2'(b-a)\bigr]
&=\frac32\sin b\cos a-\frac12\cos b\sin a,\\
\frac12\bigl[\mathcal L'(b+a)+\mathcal L'(b-a)\bigr]
&=\frac32\sin b\cos a.
\end{aligned}
$$

차이는 $-\tfrac12\cos b\sin a=\mathcal O(a)$다. $b=0.4$에서 $a=0.1,0.01,0.001$로 줄이면 차이를 $a$로 나눈 값이 약 $-0.4605$로 간다. 이를 $\mathcal O(a^3)$ 잔차로 넣을 수 없다. Gaussian 전역 gradient의 보정은 $a^2$ 차수이므로 이 1차 항을 없애 주지도 않는다. **리뷰 해석.** 이 반례는 D.3을 비IID 손실에 적용하려면 $\frac1m\sum_kH_k(\bar\theta)\delta_k$ 같은 1차 이질성 항을 따로 제어해야 함을 보인다.

### Corollary D.4: stochastic noise의 분산

참여자들의 gradient noise가 조건부 독립이고 평균 0이며 평균 분산 상한이 $\sigma^2$이면, 평균 noise의 분산은 $\sigma^2/m$ 이하가 된다. 실제 평균 gradient를 $\bar g_t=m^{-1}\sum_k\nabla\mathcal L_k(\theta_k^{(t)})$라 쓸 때

$$
\mathbb E_t\|\bar\theta^{(t+1)}-\bar\theta^{(t)}\|^2
\leq\eta^2\|\bar g_t\|^2+\frac{\eta^2\sigma^2}{m}.
$$

이 단계와 $\bar g_t$를 $g_t+q_t/2$로 대체하는 단계는 별개다. [Corollary D.4, p.30](https://arxiv.org/pdf/2507.06542v4#page=30)

**리뷰 보충: 왜 $1/m$인가?** $\nu_k$를 참여자 $k$의 centered gradient noise라 두면

$$
\mathbb E_t\left\|\frac1m\sum_k\nu_k\right\|^2
=\frac1{m^2}\sum_k\mathbb E_t\|\nu_k\|^2
+\frac1{m^2}\sum_{k\ne l}\mathbb E_t[\nu_k^\top\nu_l]
\leq\frac{\sigma^2}{m}.
$$

다. 조건부 독립성과 평균 0 때문에 $k\ne l$인 교차항이 사라진다. 남은 합은 Assumption 3의 평균 noise bound로 제어한다. 그래서 표준편차는 $1/\sqrt m$, 분산은 $1/m$ 비율로 줄어든다. 참여자 noise가 상관되어 있으면 교차항이 남는다.

### Corollary D.5: Taylor 나머지를 확인하기

원문은 gradient를 2차까지 전개한다.

$$
\nabla\mathcal L(\theta+\epsilon)
=g+H\epsilon+\frac12\nabla^3\mathcal L(\theta)[\epsilon,\epsilon]+R_3(\epsilon).
$$

평균 0은 선형항을 없애고, 공분산은 2차항을 $q/2$로 만든다. 여기까지의 구조가 저자 설명의 중심이다.

공분산이 trace로 바뀌는 중간 과정도 확인하자. $i$번째 성분을 쓰고 $\mathbb E[\epsilon_j\epsilon_l]=\Gamma_{jl}$를 대입하면

$$
\left[\mathbb E\,\nabla^3\mathcal L(\theta)
[\epsilon,\epsilon]\right]_i
=\sum_{j,l}\partial_{ijl}\mathcal L(\theta)\,\Gamma_{jl}
=\partial_i\operatorname{Tr}(\nabla^2\mathcal L(\theta)\Gamma).
$$

다. 마지막 단계는 혼합 편미분의 교환과 $\Gamma$의 대칭성을 사용하며, 미분 중 $\Gamma$는 고정한다. 이 2차 모멘트 항등식 자체에는 Gaussian 이외의 centered 분포도 사용할 수 있다. 다만 분포에 따른 나머지의 기댓값은 별도로 제어해야 한다.

**리뷰 보충 검산:** 위 gradient 전개의 올바른 적분 나머지는

$$
R_3(\epsilon)=\frac12\int_0^1(1-s)^2
\nabla^4\mathcal L(\theta+s\epsilon)
[\epsilon,\epsilon,\epsilon]\,ds,
\qquad
\|R_3(\epsilon)\|\leq\frac{L_4}{6}\|\epsilon\|^3.
$$

원문의 kernel $(1-s)^3/6$와 상수 $L_4/24$는 이 전개와 맞지 않는다. $\mathcal L(x)=1-\cos x$, $\theta=0$이면 gradient remainder는 $\sin x-x\sim-x^3/6$이므로 $|x|^3/24$ 경계를 실제로 넘는다. 잔차가 3차라는 점은 유지되지만 이후 충분조건의 명시 상수는 재검토해야 한다. [Corollary D.5, p.31](https://arxiv.org/pdf/2507.06542v4#page=31)

### Theorem D.6: 기존 DSGD 수렴률의 역할

원문이 재진술한 기존 결과는 $\eta\lesssim p/L$에서 평균 제곱 gradient 목표에 필요한 반복 수를 대략

$$
T=\mathcal O\!\left(
\frac{\sigma^2}{m\varepsilon^2}
+\frac{\sqrt p\,\sigma+\zeta}{p\varepsilon^{3/2}}
+\frac1{p\varepsilon}\right)
\cdot\bigl(\mathcal L(\theta^{(0)})-\mathcal L^\star\bigr)
$$

로 표현한다. 여기서는 원문 비교와 같이 매끄러움 상수 등을 생략한 차수만 쓴다. 통신과 이질성 비용이 보인다는 것이 비교의 목적이다. Koloskova 2020의 원래 Theorem 2는 local-step window와 더 일반적인 가정을 포함하며, 위 단순화는 Kong 2021의 재진술과도 연결된다. [D.6, p.31](https://arxiv.org/pdf/2507.06542v4#page=31)

### Theorem D.7: 저자의 A→B→C 증명 순서를 풀어 읽기

D.7과 본문 Theorem 1은 공통 초기화, Assumptions 2–3, $\eta<2/L_2$에서

$$
T=\mathcal O\!\left(
\frac{\sigma^2}{m\varepsilon^2}+\frac1\varepsilon+
\sum_{t=0}^{T-1}U^{(t)}\right)
\cdot\bigl(\mathcal L(\theta^{(0)})-\mathcal L^\star\bigr)
$$

라는 암묵적 표현을 싣는다. 하지만 Table 1과 증명 끝에는 마지막 합 앞에 $1/\varepsilon$이 붙는다. $U$의 교차항 계수도 본문과 부록 사이에서 다르다. 아래는 필요한 추가 전제를 밝힌 뒤 직접 확인할 수 있는 계산만 따로 보인다.

**Step A — 감소 항 분해.** 리뷰 보충으로, D.3의 동질적 전개가 유효하거나 추가 이질성 항을 별도 제어했다고 가정하자. gradient 단위의 나머지를 $r_t$라 두어 $\bar g_t=g_t+q_t/2+r_t$라 쓰면, $L_2$-smoothness와 독립 noise에서

$$
\begin{aligned}
\mathbb E_t\mathcal L(\bar\theta^{(t+1)})
&\leq\mathcal L(\bar\theta^{(t)})
-\left(\eta-\frac{\eta^2L_2}{2}\right)\|g_t\|^2\\
&\quad+\frac{\eta^2L_2-\eta}{2}g_t^\top q_t
+\frac{\eta^2L_2}{8}\|q_t\|^2
+\frac{\eta^2L_2\sigma^2}{2m}+\mathcal R_t,
\end{aligned}
$$

$$
\mathcal R_t=-\eta g_t^\top r_t
+\eta^2L_2(g_t+q_t/2)^\top r_t
+\frac{\eta^2L_2}{2}\|r_t\|^2.
$$

교차항의 $1/2$는 $\|g+q/2\|^2=\|g\|^2+g^\top q+\|q\|^2/4$를 전개하면 확인된다. 원문 D.8에는 이 계수가 있지만 본문 식 (9)와 D.17에는 없다. 또한 $r_t$와 매개변수 업데이트 단위의 $\delta^{(t)}$를 혼용하면 $\eta$ 계수가 달라지므로 단위를 유지해야 한다.

**Step B — 모델 차이 관련 항 제어.** 공분산의 trace와 유계 3차 미분에서

$$
\|q_t\|\leq L_3\Xi_t^2,\qquad
\frac1m\sum_k\|\delta_k^{(t)}\|^3\leq\sqrt m\,\Xi_t^3
$$

가 나온다. 두 번째는 $a_k=\|\delta_k\|$에 대해 $\sum a_k^3\leq(\sum a_k^2)^{3/2}$를 쓰면 된다. 원문의 “$x^{3/2}$의 오목성” 설명은 잘못된 표현이지만 이 부등식 자체는 위 방식으로 성립한다. $m,\eta$를 고정하면 잔차는 3차부터, $\|q_t\|^2$는 4차부터 시작한다. 비IID의 누락 1차 항에는 이 주장을 그대로 적용할 수 없다.

원문 식 (D.8)의 다섯 묶음을 앞의 gradient-unit $r_t$로 대응시키면 차수 관계가 보인다. 아래는 **리뷰 표기로 계수를 구분한 대응표**다.

| 묶음 | 앞의 감소식에서의 항 | $\Xi_t$에 대한 차수 |
| --- | --- | --- |
| $T_1$ | $\eta^2L_2\|q_t\|^2/8$ | 4차 |
| $T_2$ | $(\eta^2L_2-\eta)g_t^\top q_t/2$ | 2차; 부호를 보존하는 중심 항 |
| $T_3$ | $-\eta g_t^\top r_t$ | 3차부터 |
| $T_4$ | $\eta^2L_2(g_t+q_t/2)^\top r_t$ | 3차와 5차 |
| $T_5$ | $\eta^2L_2\|r_t\|^2/2$ | 6차 |

잔차의 노름이 $\|r_t\|\leq L_4\sqrt m\,\Xi_t^3/6$로 제어되는 동질적 전개를 전제로 하면 이 표를 얻는다. $0\leq\Xi_t\leq1$에서는 4·5·6차 항도 각각 상수 곱의 3차 항 아래로 묶을 수 있다. 그래서 원문이 유리한 2차 항을 남기고 나머지를 3차 잔차로 압축하려는 이유가 보인다.

**Step C — 시간에 걸쳐 합하기.** 올바르게 정리한 교차항·잔차를 $\eta\widetilde U_t$라 쓰자. $\widetilde U_t$는 원문의 불일치하는 $U^{(t)}$와 구별한 리뷰 보조 기호다. $c=1-\eta L_2/2>0$, $\Delta=\mathcal L(\bar\theta^{(0)})-\mathcal L^\star$라 두고 위 부등식을 합하면 손실 차이가 망원합(telescoping sum)으로 정리되어

$$
\frac1T\sum_{t=0}^{T-1}\mathbb E\|g_t\|^2
\leq\frac{\Delta}{\eta cT}
+\frac1{cT}\sum_{t=0}^{T-1}\mathbb E\widetilde U_t
+\frac{\eta L_2\sigma^2}{2mc}
$$

를 얻는다. 마지막 항은 고정 학습률의 noise floor다. 원문의 D.12에는 $U$항에 추가 $1/\eta$가 있으나 직전 망원합에서 나누면 위와 같은 계수가 된다.

이 계산은 저자의 **부호를 남겨 둔 감소 분석**이 무엇을 하려는지 보여 주지만, 원문에 인쇄된 수렴률·상수를 그대로 확인하지는 않는다. [D.7, pp.31–34](https://arxiv.org/pdf/2507.06542v4#page=32)

### Proposition D.8: 왜 작은 차이가 유익할 수 있다는가?

먼저 gradient 하한 $\mu_t$가 왜 필요한지 보자. 음의 2차 항을 보장하려면 정렬 부호뿐 아니라 그 크기가 0으로 사라지지 않도록 해야 한다. 본문은 Polyak–Łojasiewicz 조건(PL condition)을 동기로 제시한다. PL 상수를 gradient 하한과 구별해 $\mu_{\mathrm{PL}}$로 쓰면

$$
\frac12\|g_t\|^2\geq\mu_{\mathrm{PL}}
(\mathcal L(\bar\theta^{(t)})-\mathcal L^\star)
\quad\Longrightarrow\quad
\|g_t\|\geq
\sqrt{2\mu_{\mathrm{PL}}
(\mathcal L(\bar\theta^{(t)})-\mathcal L^\star)}.
$$

다. PL 조건이 성립하면 최적점에서 떨어져 있는 동안 양의 gradient 하한을 얻는다. 저자도 이 하한을 수렴 전 구간의 성질로 두고 시점별 $\mu_t$로 쓴다.

원문 Assumption 4(점진적 첨예화, progressive sharpening)는 모든 PSD $\Sigma$와 모든 $\theta$에서

$$
\nabla\mathcal L(\theta)^\top
\nabla\operatorname{Tr}(\nabla^2\mathcal L(\theta)\Sigma)<0
$$

를 요구한다. 문자 그대로는 $\Sigma=0$이나 $\nabla\mathcal L(\theta)=0$에서 성립하지 않는다. $\Sigma$를 0에 가깝게 축소하면 내적도 작아지므로, 정규화 없이 동일한 양의 $\gamma$를 요구할 수도 없다.

원문 증명의 의도를 이해하려면 **고정된 비정상점(nonstationary point)에서, 0이 아닌 방향을 단위 길이로 정규화했을 때 엄격히 음수**라는 지역적 추가 가정으로 읽어야 한다. 이 해석 아래

$$
F_t(v)=\frac{\nabla^3\mathcal L(\bar\theta^{(t)})[v,v,g_t]}{\|g_t\|},
\qquad \|v\|=1
$$

는 단위구면에서 연속이다. 구면은 compact하므로 최대값이 존재하며, 모든 방향에서 음수라는 전제하에 그 최대값도 음수다. 따라서 어떤 $\gamma_t>0$에 대해

$$
g_t^\top q_t\leq-\gamma_t\|g_t\|\Xi_t^2
\leq-\gamma_t\mu_t\Xi_t^2
$$

를 얻는다. 모든 시점에 쓸 균일한 $\gamma^*>0$는 추가로 확보해야 한다.

이제 $\eta>1/L_2$이면 선도항은 음의 2차 항이 되고 잔차는 양의 상한 $C\Xi_t^3$로 제어된다. $-a\Xi_t^2+C\Xi_t^3<0$는 $0<\Xi_t<a/C$에서 성립한다. 이것이 Proposition D.8의 핵심 논리다.

상한 $\eta<2/L_2$와 합쳐 큰 학습률 구간이 필요하고, noise floor도 남는다. Remark D.2는 작은 $\varepsilon$에 대해 $m>3\sigma^2/((2-\ell)\varepsilon)$처럼 참여자 수를 늘리는 조건을 논한다. 고정된 $m$, 양의 noise, 큰 상수 학습률에서 임의 정밀도 수렴을 동시에 결론낼 수는 없다. [D.8·Remarks D.1–D.2, pp.34–36](https://arxiv.org/pdf/2507.06542v4#page=35)

### Proposition D.9: 임계 합의 경계 (Critical Consensus Edge)

저자의 마지막 단계는 작은 차이라는 말을 통신 조건으로 바꾸는 것이다. 원문 기호대로

$$
\begin{aligned}
K&=(\eta L_2-1)\gamma^*\mu_t,\\
C_{\mathrm{lin}}&=(\eta L_2+L_4/24)\sqrt m L_1,\\
\Sigma_{\mathrm{high}}&=\frac{\eta L_2L_3^2}{8}
+\frac{\eta\sqrt mL_2L_3}{2}
+\frac{\eta mL_2L_4^2}{1152}
\end{aligned}
$$

라 두고 $\Xi_t\leq1$에서 고차 다항식을 제어한다. 선형·이차 항에 각각 예산 $K/2$를 주면

$$
\Xi_t<\min\left\{\frac{K}{2C_{\mathrm{lin}}},
\sqrt{\frac{K}{2\Sigma_{\mathrm{high}}}}\right\}
$$

라는 충분조건을 얻는 구조다. $C_{\mathrm{lin}}$과 $\Sigma_{\mathrm{high}}$에는 D.5의 $L_4/24$ 계수 문제가 그대로 들어 있다.

원문은 이 왼쪽을

$$
\sqrt{\frac{24(1-p)\eta^2}{p^2}(\phi^2+\sigma^2)}
$$

로 대체한다. 그러나 Corollary D.2가 주는 것은 **$\mathbb E\Xi_t^2$의 상한**이다. 그 제곱근은 제곱평균제곱근(root mean square, RMS)이며 매 실행의 $\Xi_t$ 상한은 아니다. 예를 들어 $\Xi_t=0.8$일 확률이 0.01이고 나머지는 0이면 RMS는 0.08이다. RMS가 0.1보다 작아도 실제 값이 0.1을 넘는 사건은 남는다. 모든 값이 $\Xi_t\leq1$이라는 조건도 이 예에서 지켜진다.

확률 보장을 원한다면 Markov 부등식 $\Pr(\Xi_t\geq a)\leq\mathbb E\Xi_t^2/a^2$ 같은 단계나 더 강한 집중 경계가 필요하다. 혹은 기대 감소 부등식 전체에서 모멘트를 일관되게 제어해야 한다. 현재 증명은 이 연결을 제공하지 않는다. 부록 D.9의 가정 목록에는 점진적 첨예화(Assumption 4)가 빠져 있지만 증명은 이를 사용한다. [D.9, pp.36–37·본문 Proposition 3](https://arxiv.org/pdf/2507.06542v4#page=36)

**리뷰 해석.** 따라서 Proposition 3은 “후반에 gradient 하한 $\mu_t$가 작아지면 더 강한 혼합이 필요하다”는 설계 직관을 주지만, 실행마다 병합 가능성을 보장하는 조건은 아니다.

<span id="리뷰-해석과-다음-검증" aria-hidden="true"></span>

## 리뷰 해석과 남은 질문

**리뷰 해석.** 부록의 강점은 본문 관찰을 optimizer·topology·초기화·이질성·마지막 gossip의 여러 축에서 반복하고, 예상 질문(A)에 저자가 직접 답해 조건을 드러낸 데 있다. 이론에서는 모델 차이의 2차 항을 부호째 남겨 점진적 첨예화와 결합하는 발상이 새롭고, Step A–C의 감소식은 동질적 전개를 전제하면 위처럼 따라갈 수 있다. 한계는 그 전제에 몰려 있다. 비IID에서 남는 1차 항(D.3), gradient 전개 나머지의 상수(D.5), 문자 그대로는 $\Sigma=0$에서 깨지는 Assumption 4(D.8), 기대 제곱 상한을 실행별 조건에 쓰는 대입(D.9)이 각각 Theorem 1과 Proposition 3의 적용 범위를 좁힌다. 본문·Table 1·부록 사이의 $U^{(t)}$ 계수와 $1/\varepsilon$ 위치 불일치(D.7)도 명시 수렴률을 그대로 쓰기 어렵게 한다.

남은 질문은 세 가지다. 첫째, 별도 validation과 여러 seed에서 같은 통신량으로 초반·후반 동기화와 최종 전역 병합을 비교해도 같은 순서가 나오는가? 둘째, 고정된 상호 배타적 로컬 데이터와 다시 뽑는 label-skew 데이터에서 효과가 같은가? 셋째, 비IID Hessian 차이 항을 보존하고 합의 거리를 확률적으로 제어해도 $U^{(t)}<0$의 충분조건을 세울 수 있는가?

## 출처와 읽은 범위

- 원문: *On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning* · Tongtian Zhu et al. · 최초 공개 2025-07-09 · [검토 버전](https://arxiv.org/abs/2507.06542v4) · 검토일 2026-10-23.
- 읽은 범위: v4 부록 A→B→C→D, D.1–D.9의 정리와 증명. 본문 §1–6은 [1/2] 해설에 연결한다.
- 참고한 외부 원전: [Koloskova 2020](https://proceedings.mlr.press/v119/koloskova20a.html) pp.1·3·5–6, [Kong 2021](https://proceedings.mlr.press/v139/kong21a.html) pp.1·3–4, [Zhu 2023](https://proceedings.mlr.press/v202/zhu23e.html) pp.1·3–4·16–17·22–24, [Wortsman 2022](https://proceedings.mlr.press/v162/wortsman22a.html) pp.1–3, [de Vos 2023](https://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html) pp.1·5–6.
- 참고한 코드: 저자 공개 저장소 [Raiden-Zhu/ICLR-2026-Grokking-in-Decentralized-Learning @ 6a5c139](https://github.com/Raiden-Zhu/ICLR-2026-Grokking-in-Decentralized-Learning/tree/6a5c139c2cdf6d6d7e0b6ba3c00f123e494e6660)의 데이터 sampler와 최종 전역 병합 뒤 BatchNorm 처리 설명(C.1).
- 그림 출처: 대표 Figure 1(b)와 Figures C.5·C.6은 저자 v4 자산(CC BY 4.0). PDF 추출·래스터화·무손실 WebP 변환을 적용하며 데이터·라벨을 유지한다. [원문](https://arxiv.org/abs/2507.06542v4).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증, 공개 코드와 원문 그림을 만든 당시 코드의 일치.
