---
title: "[1/2] Single Global Merging: 드문 통신 뒤 마지막 평균이 성능을 회복하는 이유"
description: "통신 시점과 세 가지 평가 지표를 구분하며 원문 §1–6의 실험·이론·한계를 읽는다."
date: "2026-10-23"
publishedAt: "2026-10-23T12:14:59+09:00"
updatedAt: "2026-10-23T12:14:59+09:00"
topics: ["decentralized learning", "model merging", "distributed optimization", "federated learning", "communication efficiency"]
visibility: "public"
paperTitle: "On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning"
paperPublishedDate: "2025-07-09"
authors: ["Tongtian Zhu", "Tianyu Zhang", "Mingze Wang", "Zhanpeng Zhou", "Can Wang"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2507.06542v4"
thumbnail: "/assets/reviews/single-global-merging/thumbnail.webp"
lang: "ko"
translationKey: "single-global-merging"
thumbnailAlt: "원문 Figure 1(b): sparse 분산 학습의 마지막 가중치 평균에서 전역 테스트 성능이 크게 회복되는 ResNet-18 실험."
---

서로 다른 데이터로 학습한 모델들이 각각은 전체 문제를 잘 풀지 못하는데, 마지막에 가중치를 평균하면 성능이 크게 좋아질 수 있을까? 이 논문은 **학습 중 드문 통신을 이어 온 모델들에서 그런 현상이 나타난다**고 보고한다. 핵심은 평균 연산 자체의 발명이 아니라, **통신을 언제 쓰고 어떤 모델을 평가해야 분산 학습의 잠재력을 제대로 볼 수 있는가**라는 질문이다.

이 글은 ICLR 2026 논문의 arXiv v4를 기준으로 **§1→§2→§3→§4→§5→§6** 순서로 읽는다. [[2/2] 부록·증명 상세 해설](/reviews/single-global-merging-theory/)은 A–D를 따라 실험 설정과 중간 유도, 확인된 증명 간극을 다룬다. 관측된 성능 회복과 그 현상을 설명하려는 이론의 성립 범위는 구분해서 읽어야 한다.

본문 그림은 누르면 큰 이미지로 열어 축과 범례를 확인할 수 있다.

## 1. 도입: 누구와 통신하느냐에서 언제 통신하느냐로 (Introduction)

완전 분산 학습(decentralized learning)에서는 중앙 매개변수 서버(parameter server) 없이 참여자(agent)들이 자기 데이터로 모델을 갱신하고, 이웃과 매개변수를 교환한다. 문제는 두 가지가 겹친다는 데 있다. 통신이 드물면 모델들이 서로 달라지고, 데이터까지 비IID(non-independent and identically distributed, non-IID)라면 각 모델은 일부 클래스에 치우치기 쉽다.

기존 연구는 통신 그래프, 즉 **누가 누구와 연결되는가**를 많이 다뤘다. 저자들은 같은 통신 예산도 **학습의 어느 시점에 배치하는가**에 따라 효과가 달라질 수 있다고 묻는다. 이를 확인하려고 일부 구간에서만 전체 동기화를 활성화하고 그 구간을 앞에서 뒤로 옮긴다. 이어서 그 구간을 마지막 한 번의 전역 병합(global merging)으로 줄인다. [§1, pp.1–3](https://arxiv.org/pdf/2507.06542v4#page=1)

[![Figure 1. 사전학습 CLIP과 처음부터 학습한 ResNet의 마지막 병합 효과, 보간 손실 지형, 세 학습 방식의 통신 구조.](/assets/reviews/single-global-merging/figure1.webp)](/assets/reviews/single-global-merging/figure1.webp)

*원문 Figure 1(a–d). (a)는 사전학습한 CLIP ViT-B/32, (b)는 사전학습 없이 학습한 ResNet-18이다. 파랑은 federated training, 주황은 sparse decentralized training, 초록은 무통신 local training이다. (a,b)의 가로축은 통신 라운드, 세로축은 평균 전역 테스트 정확도다. 마지막 주황색 반등이 핵심 관찰이다. (c)의 색은 정확도가 아니라 보간 모델의 cross-entropy loss이며, (d)는 세 방식의 연결 구조다. 원문 CC BY 4.0 그림을 PDF에서 추출해 무손실 WebP로 변환했다.*

Figure 1(b)를 대표 이미지로 선택한 이유도 이 마지막 변화 때문이다. 학습 중에는 주황과 초록이 비슷해 보이지만, 마지막 평균 이후에는 결과가 크게 갈린다. **“single global merging”은 학습 전체에서 통신을 한 번만 했다는 뜻이 아니다.** 주황색 모델들은 학습 중에도 낮은 빈도로 이웃과 통신해 왔다.

## 2. 관련 연구: 이 논문이 바꾸는 부분 (Related Work)

저자들은 병렬·연합·분산 학습의 시간적 통신 배분을 비교한다. 연합 학습(federated learning, FL)은 보통 중앙의 전역 모델을 유지하지만, 이 논문의 분산 설정에는 학습 내내 모두가 공유하는 전역 모델이 없다. 따라서 초반 정렬이 중요하다는 다른 설정의 결론을 그대로 옮기기보다, 평가 대상과 데이터 분포부터 맞춰야 한다.

직접 확인한 핵심 원전 다섯 편을 정리하면 다음과 같다. 최신 후속 연구를 망라한 목록은 아니다.

| 원전 | 이미 알려진 내용 | 이번 논문과의 연결·차이 |
| --- | --- | --- |
| [Koloskova et al., 2020](https://proceedings.mlr.press/v119/koloskova20a.html), §5–6 | 변하는 그래프·local updates·비IID를 포함한 DSGD 수렴 분석 | 이번 연구는 모델 차이를 오차로만 처리하지 않고 고차 기하의 유익한 항으로 해석하려 한다. 이전 연구가 비IID를 다루지 않았던 것은 아니다. |
| [Kong et al., 2021](https://proceedings.mlr.press/v139/kong21a.html), §3–4 | 임계 합의 거리(critical consensus distance) 아래에서 중앙집중식 수렴률을 회복하는 조건, 학습 단계별 통신 제어 | “완전 합의 없이도 중앙 수렴률 회복” 자체는 선행한다. 이번 논문의 초점은 비IID에서 마지막 병합의 성능과 다른 고차 설명이다. |
| [Zhu et al., 2023](https://proceedings.mlr.press/v202/zhu23e.html), §3–4·C.2 | DSGD와 평균 방향 SAM(average-direction SAM)의 점근적 연결 | 이번 이론의 직접적인 출발점이다. 다만 원전은 참여자들이 같은 분포에서 샘플링하는 IID 설정이므로 비IID로의 이전에는 검토가 필요하다. |
| [Wortsman et al., 2022](https://proceedings.mlr.press/v162/wortsman22a.html), §1–2 | 같은 사전학습 체크포인트에서 독립 미세조정한 모델의 가중치 평균(model soups) | 이번 연구는 서로 다른 초기화와 이질적 데이터에서도 소량의 통신이 평균 가능한 상태를 만드는지 묻는다. 무통신 평균이 언제나 실패한다는 주장은 성립하지 않는다. |
| [de Vos et al., 2023](https://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html), §3 | 무작위 통신의 혼합·수렴 효과 | 무작위 그래프를 쓰는 동기다. 구체적 프로토콜마다 확률적 행렬의 성질이 달라 모든 random-peer 방식이 같은 이론 조건을 만족하지는 않는다. |

또한 원문 §4.2는 최종 평균을 쓰던 D-PSGD, 주기적 전체 평균, gradient sparsification을 사용하는 SCSP와의 차이를 설명한다. 저자들이 강조하는 새 질문은 “최종 평균이라는 연산이 존재하는가”보다 **강한 이질성과 긴 local updates 뒤에도 왜 평균이 효과적인가**다. 이 추가 원전들은 이번에 별도 정독하지 않았으며, 비교는 해당 논문의 자기 설명 범위로 한정한다. [§2·§4.2·B](https://arxiv.org/pdf/2507.06542v4#page=3)

## 3. 표기와 준비: 세 가지 성능을 구분해야 한다 (Notations and Preliminaries)

### 3.1 비IID 분산 학습 (Non-IID Decentralized Learning)

참여자 수는 $m$, 참여자 $k$의 매개변수는 $\theta_k\in\mathbb R^d$, 데이터 분포는 $\mathcal D_k$다. 원문 식 (1)의 전역 모집단 위험(global population risk)은

$$
\mathcal L(\theta)=\frac1m\sum_{k\in\mathcal V}\mathcal L_k(\theta),
\qquad
\mathcal L_k(\theta)=\mathbb E_{\xi_k\sim\mathcal D_k}\mathcal L(\theta;\xi_k).
$$

각자는 자기 데이터로 갱신한 뒤 혼합 행렬(mixing matrix) $W^{(t)}$에 따라 이웃 모델을 섞는다. Algorithm 1은 이 반복을 설명한다. 이론에서는 행·열의 합이 모두 1인 이중 확률 행렬(doubly stochastic matrix)을 가정하여 단순 평균의 보존을 이용한다.

**갱신과 병합을 구분해서 보기.** Algorithm 1의 optimizer를 SGD로 두면, 갱신 후 매개변수를 섞는 의미는 다음과 같다. 송신 줄의 인덱스 불일치는 뒤에서 따로 짚는다.

$$
\begin{aligned}
\theta_k^{(t+1/2)}
&=\theta_k^{(t)}-\eta\nabla\mathcal L_k(\theta_k^{(t)};\xi_k^{(t)}),\\
\theta_k^{(t+1)}
&=\sum_l W_{k,l}^{(t)}\theta_l^{(t+1/2)}.
\end{aligned}
$$

첫 줄은 자기 데이터로 학습하는 단계, 둘째 줄은 이웃 정보를 받아 섞는 단계다. 자기 모델에만 가중치 1을 주면 무통신 학습이며, 모든 모델에 $1/m$을 주면 전체 평균이다. 이 논문의 sparse 학습은 그 사이의 제한된 이웃 교환을 사용하고, 종료 때 전체 평균을 한 번 추가한다. 실제 실험에서는 통신 사이에 100번의 로컬 mini-batch 갱신을 넣으므로, 위 이론의 한 번 갱신과 실험의 한 통신 라운드는 같은 시간 단위가 아니다.

전역 평균은

$$
\bar\theta^{(t)}=\frac1m\sum_{k=1}^m\theta_k^{(t)}
$$

지만, 분석을 위해 정의한 이 값이 매 라운드 실제로 모든 장치에 배포된다는 뜻은 아니다. 마지막 전역 병합에서는 이 평균을 실제 결과 모델로 만든다.

**독해상 구별할 세 지표**는 다음과 같다. 아래 $\operatorname{Acc}_{\mathcal D_k}$와 $\operatorname{Acc}_{\mathcal D}$는 설명을 위한 첨자다.

| 평가 | 수식 | 물어보는 내용 |
| --- | --- | --- |
| 로컬 모델을 자기 분포에서 평가 | $m^{-1}\sum_k\operatorname{Acc}_{\mathcal D_k}(\theta_k)$ | 각 모델이 자기 데이터의 문제를 잘 푸는가? |
| 로컬 모델을 전역 분포에서 평가 | $m^{-1}\sum_k\operatorname{Acc}_{\mathcal D}(\theta_k)$ | 각 모델이 전체 문제를 얼마나 잘 푸는가? 원문 Definition 1의 주 지표다. |
| 평균 모델을 전역 분포에서 평가 | $\operatorname{Acc}_{\mathcal D}(\bar\theta)$ | 지식을 가중치 평균으로 합쳤을 때 얼마나 잘 푸는가? |

매개변수를 평균하는 것과 정확도를 평균하는 것은 다르다. 따라서 병합 직전과 직후 그래프의 변화는, 학습 중인 동일 모델이 갑자기 추가 학습을 많이 했다는 뜻이 아니라 **평가하는 매개변수가 달라진 효과**를 포함한다. [§3.1·Definition 1, pp.3–4·C.2](https://arxiv.org/pdf/2507.06542v4#page=4)

표기상 작은 불일치도 있다. 식 (2)의 로컬 경험 손실 합에는 표본 수 정규화가 명시되지 않고, Algorithm 1의 송신 줄은 갱신 전 $\theta^{(t)}$, 혼합 줄은 갱신 후 $\theta^{(t+1/2)}$를 쓴다. 이 글은 모집단 목적식과 부록의 평균 업데이트를 기준으로 설명하며 이를 조용히 같은 식으로 취급하지 않는다.

### 3.2 병합 가능성 (Mergeability)

원문 Definition 2는 어떤 $w_k\in[0,1]$가 존재하여

$$
\mathcal L\!\left(\sum_k w_k\theta_k\right)
\leq\sum_k w_k\mathcal L(\theta_k)
$$

이면 모델 집합이 병합 가능하다고 정의한다. 의도는 비볼록 손실(non-convex loss)에서도 모델 사이의 평균이 나쁜 지점으로 떨어지지 않는 성질을 표현하는 것이다.

**정의 자체에는 주의가 필요하다.** 원문에는 $\sum_k w_k=1$이 쓰여 있지 않으며, 그 조건을 추가해도 하나의 가중치만 1로 두는 선택은 항상 등호를 만족한다. 그러므로 현재의 “어떤 가중치가 존재한다”는 정의만으로는 비자명한 병합성을 구별하기 어렵다. 이후 실험은 더 구체적인 **균등 평균 $w_k=1/m$의 실제 성능**으로 읽겠다. 이는 리뷰의 점검이며 저자가 별도로 제시한 정리는 아니다. [Definition 2, p.4](https://arxiv.org/pdf/2507.06542v4#page=4)

## 4. 경험적 관찰: 마지막 평균에서 무엇이 드러나는가? (Empirical Observations)

### 4.1 후반부로 옮긴 통신 예산 (Increasing Impact of Communication in the Later Stages of Training)

CIFAR-100, ResNet-18, AdamW, 16명 참여자, Dirichlet $\alpha=0.1$ 조건에서 학습을 여러 창(window)으로 나눈다. 선택한 창 안에서만 AllReduce로 전체 동기화하고, 나머지 라운드에는 확률 $R=0.2$로 한 무작위 이웃과 통신한다. 같은 길이의 창을 이동시켜 **통신량보다 통신 시점**의 효과를 비교하는 설계다.

[![Figure 2. 전체 통신 창의 위치에 따른 최고·최종 정확도와, sparse 또는 무통신 학습에서 평가용 평균 모델의 성능.](/assets/reviews/single-global-merging/figure2.webp)](/assets/reviews/single-global-merging/figure2.webp)

*원문 Figure 2(a–c), PDF p.5. (a,b)의 가로축은 전체 동기화를 넣은 시간 구간이며 세로축은 전역 테스트 정확도다. 밝은 막대는 학습 중 최고 성능, 진한 막대는 최종 성능이다. (c)는 학습 시간에 따른 모델 종류별 정확도다. 그림의 색은 패널마다 다른 비교를 나타내므로 범례를 따로 읽어야 한다. CC BY 4.0, PDF 추출·무손실 WebP 변환.*

(a,b)에서는 통신 창을 뒤로 옮길수록 특히 **최종 성능**이 좋아진다. 중간에 동기화하면 당시 성능이 올라도 이후 이질적인 로컬 학습을 거치며 다시 떨어질 수 있다. 따라서 한 번 도달한 최고 성능과 학습 종료 성능을 구분하는 것이 중요하다.

(a)는 전체 라운드의 $1/10$, (b)는 $1/20$에서 전체 통신한다는 캡션을 따른다. 다만 (b)의 축은 15라운드 구간을 표시하는 반면 §4.2는 창 길이를 10라운드라고 적는다. 이 불일치는 남겨 두고, 확인되지 않은 실행 설정을 추정해 채우지 않는다. 이 실험은 특정 예산·모델·데이터에서의 결과이며, 모든 분산 학습에서 초반 통신을 없애라는 보편 법칙은 아니다.

### 4.2 마지막 한 번의 전역 병합 (A Single Global Merging Significantly Improves Global Test Performance)

Figure 1의 주 실험은 Tiny ImageNet, 32명 참여자, $\alpha=0.1$, SGD 조건이다. 학습은 300 통신 라운드이며 라운드마다 100 local mini-batch updates를 수행한다. 이웃 교환은 라운드당 확률 0.2이므로 참여자당 기대 약 60회다. 마지막에만 전체 모델을 균등 평균한다.

Figure 1(b)의 주황선은 병합 전 약 0.2 수준에서 마지막 평균 후 약 0.48로 뛴다. 파란 FL 기준은 약 0.56이다. 이 수치는 **그림을 읽은 근삿값**이며, 원시 로그나 독립 재현으로 얻은 정확한 평균·신뢰구간이 아니다. 저자들이 보여 주는 것은 큰 성능 회복이지 FL와의 수학적 동등성은 아니다.

반면 통신을 전혀 하지 않은 scratch 모델의 마지막 평균은 거의 실패한다. 사전학습 CLIP의 (a)에서도 sparse 모델의 병합 효과가 보인다. 하지만 독립 미세조정의 평균이 성공하는 Model soups 같은 별도 조건도 있으므로, 초록선의 실패를 모든 독립 모델 평균의 불가능성으로 일반화해서는 안 된다. [§4.2·Remark 2, pp.5–6·C.1](https://arxiv.org/pdf/2507.06542v4#page=6)

**통신 비용도 마지막 평균을 포함해 계산한다.** 모델 크기 $P$, 참여자 $m$, 라운드 $T$에 대해 저자는

$$
C_{\mathrm{dense}}=\mathcal O(2mPT),\qquad
C_{\mathrm{sparse+merge}}=\mathcal O(mRPT+2mP)
$$

를 비교한다. 명시된 항들만 같은 방식으로 세면 비율은 $R/2+1/T$이고 $R=0.2,T=300$에서 약 10.3%다. 이는 이상화된 전송량 계산이다. 실제 WAN에서 9.7배 빨라졌다는 측정이 아니며, 한 번의 AllReduce도 내부적으로 여러 전송을 포함한다. 중앙 서버 없이 Ring-AllReduce로 구현할 수 있다는 점과 전역 집계 비용이 사라진다는 주장은 구분해야 한다.

### 4.3 합의하지 않아도 평균은 좋을 수 있다 (Mergeability Persists under Limited but Nonzero Communication)

Figure 2(c)는 이 논문의 가장 설명력 있는 비교다. 진한 파랑은 sparse 학습의 개별 모델 평균 정확도, 연한 파랑은 **그 시점의 가중치를 평가용으로 평균한 모델**의 정확도다. 연한 파랑이 학습 도중에도 더 높다. 따라서 좋은 평균 모델의 가능성은 마지막 순간에 처음 생긴 것이 아니다.

여기서 반사실적 평가(counterfactual evaluation)란 평균 모델을 계산해 시험하되 **그 모델로 각 참여자를 교체해 학습을 이어 가지는 않는다**는 뜻이다. 진한/연한 주황은 무통신 비교이며, 이 조건에서는 평균의 장점이 나타나지 않는다. 평가용 집계 비용은 실제 배포 학습 통신과 별도로 고려해야 한다.

저자들은 이를 **완전한 합의(consensus) 없이 유지되는 병합 가능성**으로 해석한다. Figure 1(c)는 16개 모델을 정16각형의 꼭짓점에 놓고 Wachspress 무게중심 좌표(barycentric coordinates)로 내부의 매개변수를 보간해 cross-entropy를 그린 것이다. 내부에 낮은 손실이 보이는 것은 평균이 좋을 수 있다는 직관을 준다. 그러나 꼭짓점 배치는 시각화의 선택이다. 고차원 매개변수가 실제 원형으로 배열됐다는 거리 보존 그림이나 모든 두 모델을 잇는 경로가 좋다는 증명은 아니다. [§4.3, p.6·A](https://arxiv.org/pdf/2507.06542v4#page=6)

## 5. 이론: 모델 차이의 일부를 유익한 항으로 읽기 (Theoretical Analysis)

### 5.1 가정과 핵심 양 (Assumptions)

모델 차이는 공분산(covariance)과 합의 거리(consensus distance)로 표현한다.

$$
\Gamma^{(t)}=\frac1m\sum_k
(\theta_k^{(t)}-\bar\theta^{(t)})
(\theta_k^{(t)}-\bar\theta^{(t)})^\top\in\mathbb R^{d\times d},
\qquad
\Xi_t^2=\operatorname{Tr}(\Gamma^{(t)}).
$$

혼합 조건에 등장할 $\Theta$는 참여자 모델을 열로 쌓은 행렬이고, $\bar\Theta$는 각 열을 평균으로 바꾼 행렬이다.

$$
\Theta=[\theta_1,\ldots,\theta_m]\in\mathbb R^{d\times m},
\qquad
\bar\Theta=\Theta\frac{\mathbf1\mathbf1^\top}{m}
=[\bar\theta,\ldots,\bar\theta].
$$

$\|\Theta-\bar\Theta\|_F^2=m\Xi_t^2$이므로, 행렬의 수축을 모델 간 거리의 수축으로 읽을 수 있다.

$\Xi_t=0$이면 모든 매개변수가 같다. 양수면 서로 다르다. $p$는 원문 Assumption 1의 기대 혼합 수축률을 나타낸다.

$$
\mathbb E_W\|\Theta W-\bar\Theta\|_F^2
\leq(1-p)\|\Theta-\bar\Theta\|_F^2.
$$

$p$는 실험의 이웃 선택 확률 $R$과 같은 기호가 아니다. Assumption 2는 $\mathcal L\in C^4$와 1–4차 미분의 전역 상한 $L_q$를, Assumption 3은 gradient noise $\sigma^2$와 참여자 간 gradient diversity $\zeta^2$의 상한을 둔다. 실제 신경망에 자동으로 성립하는 약한 조건으로 읽기 어렵다. 또한 이론의 기본 DSGD는 한 local SGD update와 mixing을 반복하는 분석이다. 실험의 라운드당 100 local updates나 AdamW까지 동일하게 증명한 것은 아니다. [§5.1, p.7](https://arxiv.org/pdf/2507.06542v4#page=7)

뒤의 계산을 따라가려면 각 상수의 역할도 구분해야 한다. Assumption 2는 다음 상한을 둔다.

$$
\|\nabla^q\mathcal L(\theta)\|\leq L_q,\qquad q=1,2,3,4,\quad\forall\theta\in\mathbb R^d.
$$

$L_1$은 gradient 크기, $L_2$는 곡률과 smoothness, $L_3$는 곡률의 변화, $L_4$는 gradient Taylor 전개의 나머지를 제어한다. Assumption 3의 두 조건은 서로 다른 원인의 차이를 다룬다.

$$
\frac1m\sum_k\mathbb E_{\xi_k}
\|\nabla\mathcal L_k(\theta_k;\xi_k)-\nabla\mathcal L_k(\theta_k)\|^2
\leq\sigma^2.
$$

첫 번째는 같은 참여자 안에서 mini-batch 샘플링으로 생기는 확률적 잡음이다.

$$
\frac1m\sum_k
\|\nabla\mathcal L_k(\theta_k)-\nabla\mathcal L(\theta_k)\|^2
\leq\zeta^2.
$$

두 번째는 같은 매개변수에서 평가해도 참여자의 데이터 분포가 달라 생기는 체계적 gradient 차이다. 모델 매개변수 사이의 $\Xi_t$, 샘플링 잡음 $\sigma$, 데이터 이질성 $\zeta$는 서로 다른 양이다.

### 5.2 수렴 분석과 progressive sharpening (Convergence Analysis)

기존의 전형적인 분석은 모델 차이를 제어해야 할 오차로 처리한다. 저자들은 3차 미분을 이용해 그중 일부를 구조적인 힘으로 분리한다. 설명을 위해 리뷰 보조 기호

$$
g_t=\nabla\mathcal L(\bar\theta^{(t)}),\qquad
q_t=\nabla\operatorname{Tr}
\bigl(\nabla^2\mathcal L(\bar\theta^{(t)})\Gamma^{(t)}\bigr)
$$

를 쓰자. 여기서 미분할 때 $\Gamma^{(t)}$는 고정한다. 논문의 의도는 평균 업데이트를 대략 $g_t+q_t/2$ 방향으로 보고, $g_t^\top q_t$의 부호가 유리한 경우를 찾는 것이다. $q_t$는 “현재 모델들의 퍼짐 방향으로 손실 곡률이 어떻게 변하는가”를 나타낸다.

**공분산과 곡률이 만나는 이유 — 리뷰 보충 직관.** $\delta_k=\theta_k-\bar\theta$, $H=\nabla^2\mathcal L(\bar\theta)$라 두고, 같은 전역 손실을 서로 다른 모델 위치에서 평가하는 보조 계산을 해 보자. Taylor 전개를 평균하면 $\sum_k\delta_k=0$으로 선형항이 지워져

$$
\frac1m\sum_k\mathcal L(\bar\theta+\delta_k)
=\mathcal L(\bar\theta)
+\frac12\operatorname{Tr}(H\Gamma)
+\mathcal O\!\left(\frac1m\sum_k\|\delta_k\|^3\right).
$$

가 된다. 이 보조 계산은 실제 비IID 업데이트의 로컬 손실 $\mathcal L_k$들을 이미 같은 것으로 만든 결과가 아니다. 공통 손실 아래에서 왜 이런 곡률 항이 나오는지 설명하기 위한 것이며, 비IID로 옮기는 단계는 2편 D.3에서 검토한다.

$$
\operatorname{Tr}(H\Gamma)=\frac1m\sum_k\delta_k^\top H\delta_k.
$$

따라서 trace는 단순히 모델이 얼마나 멀리 퍼졌는지뿐 아니라 **어떤 곡률 방향으로 퍼졌는지**를 반영한다. 평균 곡률 항을 미분하면 $q_t$가 나타난다. 손실이 정확한 이차함수라 Hessian이 일정하면 $q_t=0$이므로, 저자의 설명은 단순한 이차 근사에 없는 고차 효과에 의존한다.

[![Table 1. 병렬 SGD, 기존 DSGD 분석, 모델 차이의 부호를 남겨 둔 이번 분석의 수렴률 비교.](/assets/reviews/single-global-merging/table1.webp)](/assets/reviews/single-global-merging/table1.webp)

*원문 Table 1, p.8. $\varepsilon$은 평균 제곱 기울기 목표, $m$은 참여자 수다. 이전 분석의 통신·이질성 항과 이번 분석의 $U^{(t)}$를 비교한다. 본문 Theorem 1과 표 사이에는 $\varepsilon$ 계수의 불일치가 있어, 표를 검증된 새 명시 수렴률로 그대로 받아들이면 안 된다. CC BY 4.0, 표 전체 추출·무손실 WebP 변환.*

**Theorem 1의 대상**은 같은 초기화에서 시작한 DSGD의

$$
\frac1T\sum_{t=0}^{T-1}
\mathbb E\|\nabla\mathcal L(\bar\theta^{(t)})\|^2
\leq\varepsilon
$$

라는 평균 정상성(average stationarity)이다. 원문은 parallel SGD의 $\mathcal O(\sigma^2/(m\varepsilon^2)+1/\varepsilon)$에 모델 차이의 보정항 $U^{(t)}$를 더하는 암묵적 경계를 주장한다. $U^{(t)}$ 자체가 학습 경로와 $T$에 의존하므로 바로 대입 가능한 종료 횟수 공식은 아니다.

Assumption 4의 점진적 첨예화(progressive sharpening)는

$$
g_t^\top q_t<0
$$

라는 정렬 관계를 동기로 삼는다. 손실을 줄이는 $-g_t$ 방향으로 가면 해당 방향의 평균 곡률이 커지는 상황이다. Proposition 2는 $\eta>1/L_2$, $\|g_t\|\geq\mu_t>0$에서 충분히 작은 양의 모델 차이가 있으면

$$
U^{(t)}=\frac12(\eta L_2-1)g_t^\top q_t
+\mathcal O(\Xi_t^3)<0
$$

가 가능하다고 설명한다. 유리한 항은 $\Xi_t^2$ 차수, 잔차는 $\Xi_t^3$ 차수라 작은 범위에서 전자가 우세하다는 논리다. 안정 조건과 합치면 요구 학습률은 $1/L_2<\eta<2/L_2$다. 모델들이 조금이라도 다르면 언제나 유익하다는 뜻은 아니다.

**이론의 중요한 경계는 숨기지 않아야 한다.** 원문 Assumption 4는 모든 PSD 행렬과 모든 위치에 엄격한 음의 부호를 요구하지만, 영행렬이나 정상점에서는 내적이 0이다. 비IID 기울기를 하나의 전역 손실로 바꾸는 Proposition D.3에도 일반적으로 사라지지 않는 항이 있다. 계수와 Taylor 잔차 상수의 불일치도 확인됐다. 따라서 위 식은 **저자의 설명 틀과 주장**으로 읽고, 지역적 조건부 유도와 실제 반례는 [2편 D 해설](/reviews/single-global-merging-theory/#d-theory)에서 확인하자. [§5.2, pp.7–9·D, pp.30–37](https://arxiv.org/pdf/2507.06542v4#page=8)

저자의 논증을 원문과 함께 읽는 순서는 다음과 같다. 이 표는 새 가정을 더한 증명이 아니라, 어떤 결과가 다음 결과에 입력되는지 보여 주는 지도다.

| 원문 결과 | 다음 단계에 넘기는 내용 |
| --- | --- |
| Lemma D.1·Corollary D.2 | 통신 혼합률 $p$로 모델 차이 $\Xi_t$를 제어한다. |
| Proposition D.3·Corollaries D.4–D.5 | 평균 업데이트의 곡률 보정과 stochastic noise·잔차를 분리한다. |
| Theorem D.7 | smoothness로 한 단계의 손실 감소를 만들고 시간에 걸쳐 합한다. |
| Proposition D.8 | 정렬·학습률·작은 모델 차이 조건 아래 보정항의 부호를 논한다. |
| Proposition D.9 | 허용 모델 차이를 $p$와 gradient 하한의 조건으로 연결하려 한다. |

한 단계의 간극은 뒤 단계의 적용 범위에도 영향을 준다. 따라서 실험의 성능 반등, 이론의 조건부 감소 설명, 실제 통신 스케줄의 보장을 하나의 결과로 합쳐 읽지 않는다.

### 5.3 후반 통신이 더 중요하다는 설명 (A Theoretical Explanation for Communication Allocation)

Corollary D.2의 공통 초기화·균일 gradient 상한 조건에서는

$$
\mathbb E\Xi_t^2
\leq\frac{24(1-p)\eta^2}{p^2}(\phi^2+\sigma^2).
$$

$p$가 커지면 이 상한은 작아진다. Proposition 3은 이 혼합 효과와 $\mu_t$를 결합해 안전한 모델 차이의 문턱을 제안한다. 다른 상수를 고정하면, $\mu_t$가 작아질수록 허용되는 차이가 줄고 더 강한 혼합이 필요하다는 구조다. 저자는 이를 **큰 gradient의 초반보다 작은 gradient의 후반에 더 많은 통신을 배치하는 근거**로 해석한다.

이것은 측정 가능한 $p$와 $\mu_t$를 추정해 실행하는 완성된 적응형 알고리즘이 아니다. 실제로 gradient가 단조롭게 감소한다는 보장도 없고, 기대 제곱 거리의 상한만으로 매 실행의 거리가 문턱 아래라는 보장이 나오지도 않는다. 또한 $p=0$에서 이 상한을 쓸 수 없다는 사실이 모든 무통신 모델의 병합 실패를 증명하지는 않는다. [§5.3·Proposition 3, p.10·D.9](https://arxiv.org/pdf/2507.06542v4#page=10)

## 6. 함의와 토론: 지금 가져갈 수 있는 설계 원칙 (Implications and Discussions)

저자들의 첫 함의는 모델 병합(model merging)이다. 약한 동기화가 서로 다른 데이터에서 학습한 모델들을 나중에 합칠 수 있는 관계로 유도할 수 있다면, 공통 사전학습 체크포인트나 사후 뉴런 순열 정렬(permutation alignment)에만 의존하지 않는 병합 설계가 가능해진다. 두 번째는 학습 상태에 반응해 통신량을 조절하는 분산 알고리즘이다. [§6, p.10](https://arxiv.org/pdf/2507.06542v4#page=10)

리뷰 관점에서 가장 실용적인 메시지는 **통신이 적은 학습을 평가할 때 개별 모델의 정확도뿐 아니라 마지막 평균 모델도 함께 측정하자**는 것이다. 동시에 평균에 필요한 통신, 데이터 접근과 BatchNorm 처리, 같은 초기화 여부, 테스트셋을 이용한 튜닝 여부를 밝혀야 한다. 이번 논문은 시각 분류 실험에서 흥미로운 현상을 보여 주지만 LLM 학습이나 실제 광역 네트워크의 시간·비용 절감까지 검증한 것은 아니다.

부록에서는 데이터 이질성, optimizer, topology, 초기화, 마지막 부분 gossip을 바꾼 실험을 더 제공한다. 이를 생략하지 않고 [[2/2] 부록·증명 상세 해설](/reviews/single-global-merging-theory/)의 A–C에 이어 정리했다.

## 출처와 읽은 범위

- 원문: Tongtian Zhu, Tianyu Zhang, Mingze Wang, Zhanpeng Zhou, Can Wang, *On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning*. [arXiv:2507.06542v4](https://arxiv.org/abs/2507.06542v4), ICLR 2026. 최초 공개 2025-07-09, 검토판 수정 2026-04-27, PDF 37쪽.
- 첫 번째 읽기: 2026-10-22, 초록·도입/기여 pp.1–3·Figure 1 캡션·절 지도·§6. 독립 개관을 저장한 뒤 정독했다.
- 두 번째 읽기: 본문 §1–6·부록 A–D 전체의 활성 저자 LaTeX, 인쇄 PDF의 주요 본문·실험·증명과 대조. 참고문헌 목록 전체의 인용 원전을 정독한 것은 아니다. 원문 §1–6은 1편, A–D와 증명 보충은 2편에 대응한다.
- 관련 원전 선택 읽기: Koloskova 2020 PDF pp.1·3·5–6, Kong 2021 pp.1·3–4, Zhu 2023 pp.1·3–4·16–17·22–24, Wortsman 2022 pp.1–3, de Vos 2023 pp.1·5–6. 사용한 주장에 해당하는 범위이며 각 논문 전체 리뷰는 아니다.
- 검증: 정의·차원·계수 대조 및 8개 리뷰 수학 검사 수행. 논문 훈련 코드 실행, GPU 실험 재현, 다중 seed 검증, 실제 네트워크 측정은 **미실행**이다. 원시 학습 로그는 확인하지 않았다.
- 그림: 원문 Figure 1, Figure 2, Table 1과 Figure 1(b) 대표 이미지. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)에 따라 출처를 표시하고 PDF·저자 figure PDF를 rasterize해 무손실 WebP로 변환했다. 데이터·선·문구는 재작성하지 않았다.
