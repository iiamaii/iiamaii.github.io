---
title: "[1/2] Single Global Merging: 드문 통신 뒤 최종 전역 병합이 성능을 회복하는 이유"
description: "통신 시점과 세 가지 평가 지표를 구분하며 원문 §1–6의 실험·이론·한계를 읽는다."
date: "2026-10-23"
publishedAt: "2026-10-23T12:14:59+09:00"
updatedAt: "2026-10-23T15:26:25+00:00"
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

이 논문은 이질적인 데이터에서 드물게 이웃과 통신한 모델들을 학습 끝에 한 번 전역 병합(global merging)하면 전역 테스트 성능이 크게 회복된다는 관찰과, 이를 평균 모델의 수렴 분석으로 설명하는 이론을 제시한다. Tiny ImageNet·32명·Dirichlet $\alpha=0.1$에서 확률적 경사하강법(stochastic gradient descent, SGD)으로 처음부터 학습한 ResNet-18(Figure 1(b))은 참여자당 기대 약 60회의 이웃 교환만 거친 뒤 균등 평균하면 평균 전역 테스트 정확도가 그림상 약 20%에서 48%로 올라, 연합 학습(federated learning, FL) 기준 약 56%에 다가간다. [원문 §4.2·Figure 1](https://arxiv.org/pdf/2507.06542v4#page=2) **리뷰 해석.** 무통신 모델의 평균이 0%에 가깝게 머무는 절제(Figure 2(c))가 “제한적이지만 0이 아닌 통신이 병합 가능성을 만든다”는 경험적 주장을 직접 뒷받침하지만, 이를 설명하는 수렴 이론은 공통 초기화와 SGD 한 번마다 혼합하는 분산 SGD(decentralized SGD, DSGD)를 가정해 서로 다른 초기화·라운드당 100회 로컬 갱신·AdamW를 쓴 실험들을 직접 다루지 않는다.

이 글은 ICLR 2026 논문 arXiv v4의 본문 §1–6을 원문 순서대로 읽고, 부록 A–D의 실험 설정과 증명 검토는 [[2/2] 부록·증명 상세 해설](/reviews/single-global-merging-theory/)에서 다룬다.

본문 그림은 누르면 큰 이미지로 열어 축과 범례를 확인할 수 있다.

## 1. 도입: 누구와 통신하느냐에서 언제 통신하느냐로 (Introduction)

완전 분산 학습(decentralized learning)에서는 중앙 매개변수 서버(parameter server) 없이 참여자(agent)들이 자기 데이터로 모델을 갱신하고, 이웃과 매개변수를 교환한다. 문제는 두 가지가 겹친다는 데 있다. 통신이 드물면 모델들이 서로 달라지고, 데이터까지 비IID(non-independent and identically distributed, non-IID)라면 각 모델은 일부 클래스에 치우치기 쉽다.

기존 연구는 통신 그래프, 즉 **누가 누구와 연결되는가**를 많이 다뤘다. 저자들은 같은 통신 예산도 **학습의 어느 시점에 배치하는가**에 따라 효과가 달라질 수 있다고 묻는다. 이를 확인하려고 일부 구간에서만 전체 동기화를 활성화하고 그 구간을 앞에서 뒤로 옮긴다. 이어서 그 구간을 학습 끝의 전역 병합 한 번으로 줄인다. [§1, pp.1–3](https://arxiv.org/pdf/2507.06542v4#page=1)

[![Figure 1. 사전학습 CLIP과 처음부터 학습한 ResNet의 최종 전역 병합 효과, 보간 손실 지형, 세 학습 방식의 통신 구조.](/assets/reviews/single-global-merging/figure1.webp)](/assets/reviews/single-global-merging/figure1.webp)

*원문 Figure 1(a–d). (a)는 사전학습한 CLIP ViT-B/32, (b)는 사전학습 없이 학습한 ResNet-18이다. 파랑은 federated training, 주황은 sparse decentralized training, 초록은 무통신 local training이다. (a,b)의 가로축은 통신 라운드, 세로축은 평균 전역 테스트 정확도다. 마지막 주황색 반등이 핵심 관찰이다. (c)의 색은 정확도가 아니라 보간 모델의 cross-entropy loss이며, (d)는 세 방식의 연결 구조다. 원문 CC BY 4.0 그림을 PDF에서 추출해 무손실 WebP로 변환했다.*

학습 중에는 주황과 초록이 비슷해 보이지만, 최종 전역 병합 이후에는 결과가 크게 갈린다. “single global merging”은 학습 전체에서 통신을 한 번만 했다는 뜻이 아니다. 주황색 모델들은 학습 중에도 낮은 빈도로 이웃과 통신해 왔다.

## 2. 관련 연구: 이 논문이 바꾸는 부분 (Related Work)

원문 §2는 병렬·연합·분산 학습에서 통신을 시간축에 배분하는 방식을 대비한다. 병렬 학습에서는 Gu 등(2024)이 Local SGD의 암묵적 편향(implicit bias)을 분석해 로컬 단계 수의 일정을 제안했다. 연합 학습은 서버 기반 틀을 비IID 데이터로 넓히지만 전역 모델을 유지한다. 저자들은 이 때문에 FL 문헌에 “로컬 모델을 정렬하려면 잦은 초반 통신이 필수”(“frequent, early-stage communication is considered essential”)라는 폭넓은 합의가 생겼다고 본다.

완전 분산 학습에는 중앙 서버가 없으므로, 목표도 하나의 전역 모델이 아니라 각 로컬 모델이 전역 분포에서 일반화하는 것이다. 분산 학습의 통신 배분 연구는 주로 topology 설계 같은 공간적 배분을 다뤘고 시간적 배분은 드물다. 선구적인 IID 연구인 Kong 등(2021)은 학습 초반에 로컬 모델을 전역 평균에 더 강하게 정렬하면 테스트 성능이 소폭(“modestly”) 좋아질 수 있음을 보였다. 저자들은 비IID에서는 로컬 목적과 전역 목적이 같지 않으므로 이 결과가 전역 일반화가 아니라 로컬 일반화에 관한 것이라고 구분한다. 분산 학습의 암묵적 편향과 모델 병합(model merging) 계보는 부록 B.2·B.3으로 보낸다. [원문 §2·Appendix B](https://arxiv.org/pdf/2507.06542v4#page=3)

### 리뷰 보충: 관련 원전의 비교 범위

| 원전 | 이미 알려진 내용 | 이 논문과의 연결·차이 |
| --- | --- | --- |
| [Koloskova et al., 2020](https://proceedings.mlr.press/v119/koloskova20a.html), §5–6 | 변하는 그래프·local updates·비IID를 포함한 DSGD 수렴 분석 | 모델 차이를 제어할 오차로 다룬다. 이 논문은 그중 일부를 고차 기하의 유익한 항으로 남기려 하며, §5.2 Table 1의 비교 대상이다. |
| [Kong et al., 2021](https://proceedings.mlr.press/v139/kong21a.html), §3–4 | 합의 거리가 임계값(critical consensus distance) 아래이면 중앙집중식 수렴률을 회복한다는 분석과, IID에서 초반 합의 제어가 성능을 높인다는 실험 | 원문 §2가 인용하는 것은 실험 쪽이며, 부록 D.1·D.2의 합의 거리 분석도 이 원전에서 가져온다. |
| [Zhu et al., 2023](https://proceedings.mlr.press/v202/zhu23e.html), §3–4·C.2 | DSGD와 평균 방향 첨예도 인식 최소화(average-direction sharpness-aware minimization, SAM)의 점근적 연결 | 이 논문 이론의 직접적인 출발점이며, 원전은 참여자들이 같은 분포에서 샘플링하는 IID 설정이다. |
| [Wortsman et al., 2022](https://proceedings.mlr.press/v162/wortsman22a.html), §1–2 | 같은 사전학습 체크포인트에서 독립 미세조정한 모델의 가중치 평균(model soups) | 이 논문은 서로 다른 초기화와 이질적 데이터에서 소량의 통신이 평균 가능한 상태를 만드는지 묻는다. |
| [de Vos et al., 2023](https://papers.nips.cc/paper_files/paper/2023/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html), §3 | 무작위 통신의 혼합·수렴 효과 | §5.2가 무작위 그래프의 빠른 혼합 근거로 인용한다. |

## 3. 표기와 준비: 세 가지 성능 평가 (Notations and Preliminaries)

### 3.1 비IID 분산 학습 (Non-IID Decentralized Learning)

참여자 수는 $m$, 참여자 $k$의 매개변수는 $\theta_k\in\mathbb R^d$, 데이터 분포는 $\mathcal D_k$다. 원문 식 (1)의 전역 모집단 위험(global population risk)은

$$
\mathcal L(\theta)=\frac1m\sum_{k\in\mathcal V}\mathcal L_k(\theta),
\qquad
\mathcal L_k(\theta)=\mathbb E_{\xi_k\sim\mathcal D_k}\mathcal L(\theta;\xi_k).
$$

각자는 자기 데이터로 갱신한 뒤 혼합 행렬(mixing matrix) $W^{(t)}$에 따라 이웃 모델을 섞는다. Algorithm 1은 이 반복을 설명한다. 이론에서는 행·열의 합이 모두 1인 이중 확률 행렬(doubly stochastic matrix)을 가정하여 단순 평균의 보존을 이용한다.

**갱신과 혼합.** Algorithm 1의 optimizer를 SGD로 둔 것이 이론의 DSGD이며, 한 반복은 다음과 같다.

$$
\begin{aligned}
\theta_k^{(t+1/2)}
&=\theta_k^{(t)}-\eta\nabla\mathcal L_k(\theta_k^{(t)};\xi_k^{(t)}),\\
\theta_k^{(t+1)}
&=\sum_l W_{k,l}^{(t)}\theta_l^{(t+1/2)}.
\end{aligned}
$$

첫 줄은 자기 데이터로 학습하는 단계, 둘째 줄은 이웃 정보를 받아 섞는 단계다. 자기 모델에만 가중치 1을 주면 무통신 학습이며, 모든 모델에 $1/m$을 주면 전체 평균이다. 이 논문의 sparse 학습은 그 사이의 제한된 이웃 교환을 사용하고, 종료 때 전체 평균을 한 번 추가한다. 실험은 통신 한 번 사이에 로컬 mini-batch 갱신을 100번($H=100$) 넣으므로, 실험의 한 통신 라운드는 위 식의 한 반복과 다르다.

전역 평균은 다음과 같다.

$$
\bar\theta^{(t)}=\frac1m\sum_{k=1}^m\theta_k^{(t)}
$$

이 값은 분석을 위한 양이며 학습 중 각 참여자에게 배포되지 않는다. 최종 전역 병합이 이 평균을 실제 결과 모델로 만든다.

원문 Definition 1과 부록 C.2를 합치면 다음 세 평가를 구별할 수 있다. 첨자 $\mathcal D_k$, $\mathcal D$는 평가 분포를 밝히려고 붙인 리뷰 표기다.

| 평가 | 수식 | 물어보는 내용 |
| --- | --- | --- |
| 로컬 모델을 자기 분포에서 평가 | $m^{-1}\sum_k\operatorname{Acc}_{\mathcal D_k}(\theta_k)$ | 각 모델이 자기 데이터의 문제를 잘 푸는가? |
| 로컬 모델을 전역 분포에서 평가 | $m^{-1}\sum_k\operatorname{Acc}_{\mathcal D}(\theta_k)$ | 각 모델이 전체 문제를 얼마나 잘 푸는가? 원문 Definition 1의 주 지표다. |
| 평균 모델을 전역 분포에서 평가 | $\operatorname{Acc}_{\mathcal D}(\bar\theta)$ | 지식을 가중치 평균으로 합쳤을 때 얼마나 잘 푸는가? |

매개변수를 평균하는 것과 정확도를 평균하는 것은 다르다. 병합 직전과 직후 곡선의 점프는 추가 학습이 아니라 평가 대상이 개별 모델(둘째 행)에서 평균 모델(셋째 행)로 바뀐 결과다. [§3.1·Definition 1, pp.3–4·C.2](https://arxiv.org/pdf/2507.06542v4#page=4)

표기상 작은 불일치도 있다. 식 (2)의 로컬 경험 손실 합에는 표본 수 정규화가 없고, Algorithm 1의 송신 줄은 갱신 전 $\theta^{(t)}$를 보내지만 혼합 줄은 갱신 후 $\theta^{(t+1/2)}$를 섞는다. 위 식은 혼합 줄과 부록 D.1의 증명을 따랐다.

### 3.2 병합 가능성 (Mergeability)

원문 Definition 2는 어떤 $w_k\in[0,1]$가 존재하여

$$
\mathcal L\!\left(\sum_k w_k\theta_k\right)
\leq\sum_k w_k\mathcal L(\theta_k)
$$

이면 모델 집합이 병합 가능하다고 정의한다. 의도는 비볼록 손실(non-convex loss)에서도 모델 사이의 평균이 나쁜 지점으로 떨어지지 않는 성질을 표현하는 것이다.

실험에서는 이 정의의 가중치를 균등 평균 $w_k=1/m$으로 구체화한다. 정의의 비자명성은 [[2/2] D](/reviews/single-global-merging-theory/#d-theory)에서 검토한다.

## 4. 경험적 관찰: 최종 전역 병합에서 무엇이 드러나는가? (Empirical Observations)

### 4.1 후반부로 옮긴 통신 예산 (Increasing Impact of Communication in the Later Stages of Training)

CIFAR-100, ResNet-18, AdamW, 16명 참여자, Dirichlet $\alpha=0.1$ 조건에서 학습을 여러 창(window)으로 나눈다. 선택한 창 안에서만 AllReduce로 전체 동기화하고, 나머지 라운드에는 확률 $R=0.2$로 한 무작위 이웃과 통신한다. 같은 길이의 창을 이동시켜 **통신량보다 통신 시점**의 효과를 비교하는 설계다.

[![Figure 2. 전체 통신 창의 위치에 따른 최고·최종 정확도와, sparse 또는 무통신 학습에서 평가용 평균 모델의 성능.](/assets/reviews/single-global-merging/figure2.webp)](/assets/reviews/single-global-merging/figure2.webp)

*원문 Figure 2(a–c), PDF p.5. (a,b)의 가로축은 전체 동기화를 넣은 시간 구간이며 세로축은 전역 테스트 정확도다. 밝은 막대는 학습 중 최고 성능, 진한 막대는 최종 성능이다. (c)는 통신 라운드에 따른 정확도이며, 파랑은 sparse 통신, 주황은 무통신, 진한 색은 개별 모델, 연한 색은 평균 모델이다. CC BY 4.0, PDF 추출·무손실 WebP 변환.*

(a)는 전체 라운드의 $1/10$, (b)는 $1/20$ 동안만 전체 통신한다. 저자들은 통신 예산을 후반으로 옮길수록 성능이 크게 좋아지며 특히 최종 정확도에서 그렇다고 정리한다. 그림상 최고 정확도(연한 막대)는 창이 늦을수록 꾸준히 오르지만, 최종 정확도(진한 막대)는 학습 끝을 포함한 마지막 창에서만 (a) 약 0.55, (b) 약 0.53으로 뛰고 그 이전 창들은 약 0.24–0.28에 머문다. (a)의 삽입 그림(150–180라운드 동기화)처럼 중간에 동기화하면 그 구간의 정확도는 오르지만 sparse 통신으로 돌아가면 다시 떨어진다. [§4.1·Figure 2, p.5](https://arxiv.org/pdf/2507.06542v4#page=5) **리뷰 해석.** 따라서 최종 정확도에서 보이는 후반 배치의 효과는 대부분 학습 끝에서 전체 통신한 창에서 나오며, §4.2의 최종 전역 병합과 같은 현상이다.

(b)의 가로축은 15라운드 구간(285–300 등)을 표시하는데, §4.2는 이 창 길이를 10라운드라고 적는다.

### 4.2 한 번의 전역 병합이 전역 테스트 성능을 크게 높인다 (A Single Global Merging Significantly Improves Global Test Performance)

창을 10라운드로 줄여도 개선이 남자, 저자들은 창을 한 라운드로 줄이면 어떻게 되는지 묻고 전체 통신을 학습 끝의 전역 병합 한 번으로만 적용한다. Figure 1의 주 실험은 Tiny ImageNet, 32명 참여자, $\alpha=0.1$, SGD 조건이며, 300 통신 라운드 동안 라운드마다 100번의 로컬 mini-batch 갱신을 하고 확률 0.2로 무작위 이웃 하나와 통신한다.

Figure 1(b)의 주황선은 병합 전 약 0.2에서 최종 전역 병합 후 약 0.48로 뛰고, 파란 FL 기준은 약 0.56이다. 사전학습한 CLIP의 (a)에서도 약 0.38에서 0.55로 올라 FL 약 0.59에 다가간다(값은 그림에서 읽었다). 반면 처음부터 학습한 ResNet에서 통신하지 않은 초록선은 최종 전역 병합 뒤 0에 가깝게 떨어진다. 저자들은 데이터셋·구조·topology를 바꾼 부록 실험에서도 같은 이득이 나타나며, 분산 학습의 잠재력이 과소평가됐을 수 있다고 해석한다. [§4.2·Figure 1, pp.2·6](https://arxiv.org/pdf/2507.06542v4#page=6)

**선행 방법과의 비교 (Comparisons).** D-PSGD는 IID 설정에서 최종 전역 병합을 도입했지만 병합 전후 성능 차이를 분석하지 않았다. Chen 등(2021)의 주기적 전역 평균은 $H=48$ 단계마다 전역 통신을 요구하는 반면, 이 논문은 병합 한 번으로 성능을 회복한다. 최종 전역 병합 단계를 포함하는 SCSP(Skew-Compensated Sparse Push)는 고정 topology에서 상위 $k$개 gradient만 보내는 gradient 희소화를 쓰고 로컬 단계 하나($H=1$)를 다룬다. 이 논문은 sparse gossip이라는 topology 희소화와 $H=100$의 긴 로컬 갱신, 강한 이질성에서 병합 가능성 자체를 묻는다. 즉 극히 제한된 통신과 강한 이질성에도 로컬 모델이 왜 병합 가능한 상태를 유지하는가가 저자들이 내세우는 새 질문이다.

**성능 이득은 자명한가 (Remark 2).** 라운드마다 확률 $R=0.2$로 이웃 하나와 교환하면 $T=300$ 라운드 동안 참여자당 기대 약 60회를 교환하므로, 이것이 평균적으로 여러 번의 암묵적 전역 집계처럼 쌓인 결과라고 볼 수도 있다. 저자들은 그렇다면 로컬 모델이 이미 정렬되어 병합 후와 비슷한 성능을 내야 하지만, Figure 1(b)의 병합 전 로컬 모델은 무통신 경우에 가까운 낮은 성능에 머물고 한 번의 병합만 큰 이득을 낸다고 반박한다. [§4.2·Remark 2, p.6](https://arxiv.org/pdf/2507.06542v4#page=6)

**통신 비용.** 모델 크기 $P$, 참여자 $m$, 라운드 $T$, 라운드당 기대 이웃 수 $R$에 대해 저자는 Ring-AllReduce 기반 전체 동기화와 이 설정을 다음처럼 비교한다.

$$
C_{\mathrm{dense}}=\mathcal O(2mPT),\qquad
C_{\mathrm{sparse+merge}}=\mathcal O(mRPT+2mP)
$$

둘째 식의 $2mP$는 최종 전역 병합을 Ring-AllReduce로 할 때의 비용이다. 명시된 항만 세면 전송량 비율은 $R/2+1/T$로, $R=0.2$, $T=300$에서 약 10.3%다. 저자는 전역 병합이 어려운 환경에서는 여러 번의 gossip으로 근사할 수 있다고 덧붙인다([[2/2]](/reviews/single-global-merging-theory/)의 C.3.4).

### 4.3 합의하지 않아도 평균은 좋을 수 있다 (Mergeability Persists under Limited but Nonzero Communication)

§4.3은 전역 병합의 효과가 학습 끝에만 있는지 묻는다. Figure 2(c)는 $R=0.2$ sparse 통신(파랑)과 무통신(주황)의 개별 모델(진한 색)과, 매 라운드 가중치를 평균한 모델(연한 색)의 전역 테스트 정확도를 비교한다. 반사실적(counterfactual) 평가란 평균 모델을 계산해 시험만 하고 학습에는 되돌려 넣지 않는다는 뜻이다. 저자들은 연한 파랑이 학습 내내 진한 파랑보다 높으므로 로컬 모델이 모든 단계에서 병합 가능하다고 해석한다.

절제로 통신 없이 학습하면 평균 모델의 정확도는 0에 가깝게 머문다(연한 주황). 저자들은 이를 병합 가능성이 로컬 모델 자체에서 생기지 않는다는 근거로 든다. 병합 전 개별 모델의 정확도는 sparse 통신(진한 파랑, 그림상 끝에서 약 0.24)과 무통신(진한 주황, 약 0.20)이 비슷한데, 평균 모델은 약 0.48과 0 근처로 갈린다. 저자들은 이 대비에서 극히 제한적이지만 0이 아닌 통신이 병합 가능성을 만든다고 결론짓는다. **리뷰 해석.** 그림상 50라운드 무렵까지는 연한 파랑이 진한 파랑보다 낮아, “학습 내내”라는 서술은 초반 구간에는 맞지 않는다.

저자들은 이를 완전한 합의(consensus) 없는 병합 가능성으로도 설명한다. 이전 gossip 연구는 적은 통신에서도 로컬 모델이 비슷한 상태로 수렴할 수 있다고 보았지만, 이질적 데이터에서는 로컬 모델이 한 점으로 합의하지 않으면서도 병합 가능하다는 것이다. Figure 1(c)는 16개 모델을 정16각형 꼭짓점에 놓고 Wachspress 무게중심 좌표(barycentric coordinates)로 내부 매개변수를 보간해 cross-entropy를 그린 것으로, 저자들은 로컬 모델이 중앙의 낮은 손실 영역을 둘러싼 고리 모양의 높은 손실 영역에 놓인다고 읽는다. 공통 사전학습 체크포인트 없이 서로 다른 초기화와 비IID 데이터에서 병합하는 더 어려운 경우라는 점도 강조한다. [§4.3, p.6·A](https://arxiv.org/pdf/2507.06542v4#page=6) **리뷰 해석.** 꼭짓점을 원형으로 놓는 것은 시각화 도구의 선택이므로 고리 모양 자체는 이 배치에서 생기며, 그림이 보여 주는 것은 볼록 결합 내부의 손실이 꼭짓점보다 낮다는 사실이다.

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

$p$는 실험의 이웃 선택 확률 $R$과 다른 양이다. [§5.1, p.7](https://arxiv.org/pdf/2507.06542v4#page=7)

Assumption 2는 $\mathcal L\in C^4$와 1–4차 미분의 전역 상한 $L_q$를 둔다.

$$
\|\nabla^q\mathcal L(\theta)\|\leq L_q,\qquad q=1,2,3,4,\quad\forall\theta\in\mathbb R^d.
$$

$L_1$은 gradient 크기, $L_2$는 곡률과 smoothness, $L_3$는 곡률의 변화, $L_4$는 gradient Taylor 전개의 나머지를 제어한다. Assumption 3은 gradient noise $\sigma^2$와 참여자 간 gradient 다양성(diversity) $\zeta^2$의 상한이며, 두 조건은 서로 다른 원인의 차이를 다룬다.

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

곡률과 모델 공분산의 관계를 전개하는 [Taylor 보충 유도](/reviews/single-global-merging-theory/#d-theory)는 [2/2]의 D.5에 둔다.

[![Table 1. 병렬 SGD, 기존 DSGD 분석, 모델 차이의 부호를 남겨 둔 이 분석의 수렴률 비교.](/assets/reviews/single-global-merging/table1.webp)](/assets/reviews/single-global-merging/table1.webp)

*원문 Table 1, p.8. $\varepsilon$은 평균 제곱 gradient 목표, $m$은 참여자 수다. 이전 분석의 통신·이질성 항과 이 분석의 $U^{(t)}$를 비교한다. Theorem 1과 표의 계수 대응은 [[2/2] D.7](/reviews/single-global-merging-theory/#d-theory)에서 비교한다. CC BY 4.0, 표 전체 추출·무손실 WebP 변환.*

**Theorem 1의 대상**은 같은 초기화에서 시작한 DSGD의

$$
\frac1T\sum_{t=0}^{T-1}
\mathbb E\|\nabla\mathcal L(\bar\theta^{(t)})\|^2
\leq\varepsilon
$$

라는 평균 정상성(average stationarity)이다. 원문은 parallel SGD의 $\mathcal O(\sigma^2/(m\varepsilon^2)+1/\varepsilon)$에 모델 차이의 보정항 $\sum_tU^{(t)}$를 더한 반복 수를 제시한다. 저자도 이것이 닫힌 형태가 아니라 $U^{(t)}$에 의존하는 암묵적 경계이며, 뒤의 Proposition 2·3에서 $U^{(t)}$의 부호 조건을 끌어내는 다리라고 밝힌다. 참여자가 하나이거나 완전 동기화로 모든 모델이 같으면 $U^{(t)}=0$이 되어 parallel SGD 수렴률로 돌아간다.

Assumption 4의 점진적 첨예화(progressive sharpening)는

$$
g_t^\top q_t<0
$$

라는 정렬 관계를 동기로 삼는다. 손실을 줄이는 $-g_t$ 방향으로 가면 해당 방향의 평균 곡률이 커지는 상황이다. Proposition 2는 $\eta>1/L_2$, $\|g_t\|\geq\mu_t>0$에서 충분히 작은 양의 모델 차이가 있으면

$$
U^{(t)}=\frac12(\eta L_2-1)g_t^\top q_t
+\mathcal O(\Xi_t^3)<0
$$

가 가능하다고 설명한다. 유리한 항은 $\Xi_t^2$ 차수, 잔차는 $\Xi_t^3$ 차수라 작은 범위에서 전자가 우세하다는 논리다. 안정 조건과 합치면 요구 학습률은 $1/L_2<\eta<2/L_2$이며, 저자는 이 구간이 2차 목적함수에서 고전 최적화 이론의 진동 수렴(oscillatory convergence) 영역과 일치한다고 짚는다. 비IID 손실에서 이 평균 업데이트 전개가 성립하는지, 계수와 증명의 의존 관계는 [[2/2] D](/reviews/single-global-merging-theory/#d-theory)에서 검토한다.

**제한적이지만 0이 아닌 통신이 병합 가능성을 만드는 이유 (Why Limited but Nonzero Communication Enables Mergeability).** 원문은 Corollary D.2를 인용해 합의 거리의 기대 상한을 $\mathcal O((1-p)/p^2)$로 적는다. 부록 D.2의 구체적인 식은 local gradient 제곱 평균의 균일 상한 $\phi^2$ 아래 다음과 같다.

$$
\mathbb E\Xi_t^2
\leq\frac{24(1-p)\eta^2}{p^2}(\phi^2+\sigma^2).
$$

$p=1$은 완전 연결, $p=0$은 통신 없는 로컬 학습이다. 저자들은 무작위 통신 그래프가 적은 통신량으로도 $p=\Theta(1)$를 달성해 $\Xi_t$를 빠르게 줄인다는 점(de Vos 등, 2023)을 들어, 극히 제한된 통신에서도 Proposition 2의 조건을 만족해 병합 가능성을 보장할 수 있다고 설명한다. 실험에서 무작위 topology를 주 설정으로 쓴 이유도 이것이다. 반대로 $p=0$이면 오른쪽이 무한대로 커져 $\Xi_t$가 발산할 수 있고 Proposition 2의 조건이 성립하지 않으므로, 완전 로컬 학습 모델은 안정적으로 병합되지 않는다(Figure 1(b)의 초록선). [§5.2, p.9](https://arxiv.org/pdf/2507.06542v4#page=9) **리뷰 해석.** 이 논증은 실험의 $R=0.2$ 프로토콜이 Assumption 1의 이중 확률 혼합과 상수 $p$를 만족한다는 전제에 기대지만, 원문은 그 프로토콜의 $p$ 값이나 혼합 행렬 구성을 따로 제시하지 않는다.

### 5.3 후반 통신이 더 중요하다는 설명 (A Theoretical Explanation for Communication Allocation)

Proposition 3(임계 합의 경계, Critical Consensus Edge)은 Assumptions 1–4, $\eta>1/L_2$, $\Xi_t\leq1$ 아래에서, 위 상한의 제곱근이 점진적 첨예화의 정도 $\gamma^*$와 gradient 하한 $\mu_t$로 정해지는 문턱보다 작으면 $U^{(t)}<0$이 성립한다는 충분조건이다. 저자는 왼쪽이 $p$에 대해 감소하고 오른쪽이 $\mu_t$에 대해 증가한다는 점에서 지침을 끌어낸다. 전역 평균 모델이 최솟값에서 먼 초반에는 $\mu_t$가 커서 드문 통신(작은 $p$)으로도 조건이 성립하고, 수렴에 가까운 후반에는 $\mu_t$가 작아져 잦은 통신(큰 $p$)이 필요하다. 저자는 이것이 §4의 후반 통신 관찰과 맞는다고 본다. [§5.3, Proposition 3, p.10](https://arxiv.org/pdf/2507.06542v4#page=10)

**리뷰 해석.** 이 지침을 실제 적응형 스케줄로 쓰려면 $p$와 $\mu_t$를 학습 중에 추정하는 절차가 더 필요하며, 원문은 이를 §6의 향후 알고리즘 방향으로 남긴다. 기대 제곱 상한을 실행별 조건에 대입하는 문제는 [[2/2] D.9](/reviews/single-global-merging-theory/#d-theory)에서 다룬다.

## 6. 함의와 토론: 지금 가져갈 수 있는 설계 원칙 (Implications and Discussions)

저자들의 첫 함의는 모델 병합이다. 약한 동기화가 서로 다른 데이터에서 학습한 모델들을 나중에 합칠 수 있는 관계로 유도할 수 있다면, 공통 사전학습 체크포인트나 사후 뉴런 순열 정렬(permutation alignment)에만 의존하지 않는 병합 설계가 가능해진다. 두 번째는 학습 상태에 반응해 통신량을 조절하는 분산 알고리즘이다. [§6, p.10](https://arxiv.org/pdf/2507.06542v4#page=10)

데이터 이질성·optimizer·topology·초기화·마지막 gossip을 바꾼 부록 실험은 [[2/2] 부록·증명 상세 해설](/reviews/single-global-merging-theory/)의 C에서 다룬다.

## 리뷰 해석과 남은 질문

**리뷰 해석.** 가장 설득력 있는 부분은 측정 설계다. 개별 모델과 평균 모델의 전역 정확도를 따로 재었기 때문에 개별 모델로는 구별되지 않던 sparse 통신과 무통신의 차이가 평균 모델에서 드러났고(§4.3), Remark 2의 반박도 이 대비에 기댄다. 부록은 optimizer·topology·초기화·이질성을 바꿔 같은 방향의 결과를 보탠다. 반면 기여 (1)의 “FL에 가까운 성능”은 그림상 FL보다 ResNet에서 약 8%p, CLIP에서 약 4%p 낮은 수준이다.

근거의 한계는 세 가지다. 첫째, 하이퍼파라미터를 전역 테스트 성능으로 골랐고 반복 횟수·오차 막대를 보고하지 않는다([2/2] A·C). 둘째, 실험은 저해상도 시각 분류이고 통신 비용은 전송량 계산이라 실제 네트워크의 학습 시간은 재지 않았다. 셋째, 이론은 공통 초기화, SGD 한 번마다 혼합하는 DSGD, 모든 PSD 행렬과 모든 매개변수에서의 점진적 첨예화(Assumption 4)를 가정하는데, 저자 스스로 실험 대부분이 서로 다른 초기화를 쓴다고 밝히고 Figure 2는 AdamW와 라운드당 100회 로컬 갱신을 쓴다.

초록의 “전역 병합 모델이 parallel SGD의 수렴률에 맞출 수 있음을 최초로 확립했다”는 주장은 결론보다 증명 경로에서 새롭다. 합의 거리가 임계값 아래로 유지되면 DSGD가 중앙집중식 학습만큼 빠르게 수렴한다는 결과는 Kong 등(2021)이 이미 제시했고, 부록 D.1·D.2의 합의 거리 분석도 이 원전에서 가져온다. 새로운 것은 모델 차이의 2차 항을 부호째 남겨 유익한 항으로 쓰는 분석이며, Theorem 1이 $U^{(t)}$에 의존하는 암묵적 경계라서 “맞춘다”는 결론은 Proposition 2·3의 조건이 모두 성립할 때만 나온다. 그 조건이 비IID 손실에서 성립하는지는 [[2/2] D](/reviews/single-global-merging-theory/#d-theory)에서 검토한다.

남은 질문은 두 가지다. 병합 가능성이 생기는 통신량의 문턱은 $R=0.2$와 $R=1$만 비교한 부록 C.3으로는 알 수 없다. 또 §4.3에서 본 초반 구간, 즉 평균 모델이 개별 모델보다 낮은 약 50라운드까지의 구간은 “초반에는 드문 통신으로 충분하다”는 Proposition 3의 지침과 어떻게 맞는가?

## 출처와 읽은 범위

- 원문: *On the Surprising Effectiveness of a Single Global Merging in Decentralized Learning* · Tongtian Zhu et al. · 최초 공개 2025-07-09 · [검토 버전](https://arxiv.org/abs/2507.06542v4) · 검토일 2026-10-23.
- 읽은 범위: v4 본문 §1–6. 부록 A–D의 실험 조건·추가 결과·증명 상세는 [2/2] 이론 편에 대응한다.
- 참고한 외부 원전: §2 리뷰 보충 표의 다섯 편(표에 적은 절).
- 그림 출처: 대표 Figure 1(b)와 본문 Figures 1·2, Table 1은 저자 v4 자산(CC BY 4.0). PDF 추출·래스터화·무손실 WebP 변환을 적용하며 데이터·라벨을 유지한다. [원문](https://arxiv.org/abs/2507.06542v4).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
