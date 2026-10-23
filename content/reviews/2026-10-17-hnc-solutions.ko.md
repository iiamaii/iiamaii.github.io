---
title: "[1/2] HNC: 같은 출력을 내는 신경망은 내부에서도 같은 계산을 할까?"
description: "한 모델 주변의 함수 보존 영역을 따라가며 RNN의 기억, ViT의 표현, 강화학습 정책을 바꾸는 HNC를 원문 §1–7 순서로 읽는다."
date: "2026-10-17"
publishedAt: "2026-10-17T22:12:00+09:00"
updatedAt: "2026-10-23T08:56:23+00:00"
topics: ["representation learning", "loss landscape", "mechanistic interpretability", "computational neuroscience", "reinforcement learning"]
visibility: "public"
lang: "ko"
translationKey: "hnc-solutions"
paperTitle: "Traversing the solution space of neural networks with Hessian Null Space Continuation"
paperPublishedDate: "2026-09-29"
authors: ["Ann Huang", "Mitchell Ostrow", "Zhouyang Lu", "William T. Redman", "Leo Kozachkov", "Kanaka Rajan"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2609.38081v1"
thumbnail: "/assets/reviews/hnc/overview.webp"
thumbnailAlt: "HNC 원문 Figure 1. 평평한 가중치 방향으로 이동하고 출력을 복원하며, 다른 내부 표현을 가진 대안 모델을 찾는 과정."
---

헤시안 영공간 연속 탐색(Hessian Null Space Continuation, HNC)은 출력 일치를 근사적으로 유지하며 한 신경망 주변의 다른 내부 계산을 찾는다. 비전 트랜스포머(vision Transformer, ViT) 실험에서는 표현을 조향하면서 top-1 결정을 최소 94% 유지하고, 뇌 반응 예측 점수는 한 설정에서 0.618–0.719로 달라진다. [원문 §4·Appendix H](https://arxiv.org/pdf/2609.38081v1#page=46) **리뷰 해석.** 같은 과제 성능만으로 내부 메커니즘을 식별하기 어렵다는 근거이며, 보존 범위는 probe와 별도의 평가 입력으로 확인해야 한다.

55쪽의 [v1](https://arxiv.org/pdf/2609.38081v1)을 두 편으로 나눴다. 이 글은 **본문 §1–7의 순서**를 유지하며 핵심 방법과 네 실험군을 설명한다. [\[2/2\] 수학·알고리즘·검증 심화](/reviews/hnc-mechanics/)는 부록 A–J를 따라 수학 유도, 복원 실패, probe 밖 평가와 추가 실험을 다룬다. 1→2 순서를 권한다.

## 1. 도입: 연결된 해들 안에는 어떤 계산이 있는가? (Introduction)

과매개변수화(overparameterization)된 신경망에는 비슷한 학습 손실을 내는 가중치가 많다. **모드 연결성(mode connectivity)** 연구는 독립적으로 학습한 두 해 사이에도 손실이 낮은 경로가 있을 수 있음을 보여줬다. 한편 **해의 퇴화성(solution degeneracy)** 연구는 같은 과제를 해결하는 모델들의 내부 구조가 다를 수 있다고 지적한다. 여기서 degeneracy는 성능 저하나 표현 붕괴를 뜻하지 않는다. 하나의 과제에 양립하는 해가 여럿이라는 뜻이다.

HNC의 질문은 이 둘을 연결한다. 두 모델을 따로 학습한 뒤 잇는 대신, **한 모델 주변의 연결된 낮은 손실 영역을 직접 돌아다니면 얼마나 다른 계산을 찾을 수 있을까?** 원문의 anchor는 탐색의 기준이 되는 학습 모델이다. 다음 네 연구는 이 질문을 읽는 데 도움을 준다.

| 선행 연구 | 실제로 제공한 관점 | HNC와의 관계 |
| --- | --- | --- |
| [Garipov et al., 2018](https://arxiv.org/pdf/1802.10026v4), §3–4 | 두 학습 해 사이의 낮은 손실 경로와 의미 있는 표현 차이 | HNC는 도착할 두 번째 해 없이 한 anchor 주변을 탐색한다. 표현 다양성 자체가 처음 관찰된 것이라고 소개하지 않는다. |
| [Kornblith et al., 2019](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf), §2–3 | 중심 커널 정렬(centered kernel alignment, CKA)로 표현의 유사 구조 비교 | CKA는 탐색의 평가·조향 지표다. 낮은 CKA 하나가 모든 계산 차이를 증명하지는 않는다. |
| [Ostrow et al., 2023](https://arxiv.org/pdf/2306.10168v2), §2 | 동역학 유사도 분석(dynamical similarity analysis, DSA)으로 시간적 계산 구조 비교 | RNN에서 공간적 모양과 시간적 동역학을 나누어 살핀다. |
| [Huh et al., 2024](https://proceedings.mlr.press/v235/huh24a.html), §2 | 플라톤적 표현 가설(Platonic Representation Hypothesis, PRH), 모델 성능·규모와 표현 정렬의 관찰 | HNC는 과제가 허용하는 다양성과 일반적인 학습이 실제 선택하는 다양성을 구분한다. PRH의 모든 경험적 관찰을 반박하는 실험은 아니다. |

이 원전들은 필요한 절만 추가 확인했으며, 최신 후속 연구 전수 서베이를 한 것은 아니다. HNC의 기여는 곡률을 이용한 구성적 탐색과, 그 경로에서 표현·동역학·행동·기하를 함께 조사한다는 데 있다. [원문 §1, pp. 1–3](https://arxiv.org/pdf/2609.38081v1#page=1)

## 2. 방법 (Method)

### 2.1 손실 곡률에서 함수 보존 방향 찾기

가중치 벡터를 $\boldsymbol\theta$, anchor를 $\boldsymbol\theta_0$, 입력 $x$에 대한 출력을 $f_{\boldsymbol\theta}(x)$라고 하자. 원문 식 (1)은 정답과의 과제 손실이 아니라 <strong>함수 일치 손실(function-matching loss)</strong>을 정의한다.

$$
\mathcal L(\boldsymbol\theta)
=\frac12\mathbb E_{x\sim\mathcal D}
\left[\left\|f_{\boldsymbol\theta}(x)
-f_{\boldsymbol\theta_0}(x)\right\|^2\right].
$$

$\mathcal D$는 입력 분포다. 실제로는 여기서 뽑은 고정 입력 집합 $\mathbf X$, 즉 **probe set**으로 기대값을 추정한다. 분류 모델에서는 정답 라벨 하나가 아니라 anchor의 logit 벡터 전체를 맞춘다. 점수 하나를 비슷하게 유지하는 것보다 더 강한 제약이다.

anchor는 자신의 출력을 정확히 재현하므로 이 손실과 gradient가 0이다. 원문 식 (2)는 작은 가중치 변화 $\boldsymbol\delta$의 영향을 곡률로 근사한다.

$$
\begin{aligned}
\mathcal L(\boldsymbol\theta_0+\boldsymbol\delta)
&=\frac12\boldsymbol\delta^\top\mathbf H\boldsymbol\delta
+O(\|\boldsymbol\delta\|^3),\\
\mathbf H&=\nabla^2_{\boldsymbol\theta}\mathcal L(\boldsymbol\theta_0).
\end{aligned}
$$

Hessian $\mathbf H$의 큰 고유값은 출력 변화에 민감한 **가파른 방향(sharp direction)**, 작은 고유값은 덜 민감한 <strong>평평한 방향(flat direction)</strong>을 뜻한다. 원문은 고유벡터 $\mathbf v_i$와 고유값 $\lambda_i$에 대해 근사 영공간(approximate null space)을 정의한다.

$$
\mathcal V_0=\operatorname{span}
\{\mathbf v_i:\lambda_i\le\epsilon\}.
$$

이는 정확히 0인 고유값만의 공간과 다르다. 임계값 $\epsilon$을 바꾸면 포함되는 방향도 바뀐다. 또 이 근사는 현재 위치 근처에서만 유효하다. 평평한 방향으로 한 번 크게 움직여도 출력이 그대로라는 뜻은 아니다. [§2.1, p. 3](https://arxiv.org/pdf/2609.38081v1#page=3)

### 2.2 이동하고, 복원하고, 다시 방향을 계산한다

![원문 Figure 1: 평평한 가중치 방향으로 이동하고 출력을 복원하는 HNC](/assets/reviews/hnc/overview.webp)

[Figure 1 크게 보기](/assets/reviews/hnc/overview.webp)

**Figure 1의 핵심은 A→D→B다.** A의 높이는 출력 불일치 손실이고 바닥의 두 축은 가중치 방향을 개념적으로 그린 것이다. D의 보라색 화살표는 평평한 방향의 이동(flat step), 청록색은 출력을 다시 맞추는 복원(restore)이다. 이를 반복하면 B처럼 같은 입력의 내부 표현이 달라질 수 있다. B의 PC 축은 내부 활성의 주성분이며, A의 가중치 축과 다른 공간이다. C는 평평함을 고유값 임계값으로 고르는 방법, E는 원하는 속성 쪽으로 이동을 조향하는 방법을 나타낸다. B의 출력 등호는 개념도다. 실제 실험의 보존은 허용 오차와 probe 범위로 해석해야 한다. [Figure 1, p. 2](https://arxiv.org/pdf/2609.38081v1#page=2)

원문 식 (3)은 다음 두 단계를 보여준다.

$$
\begin{aligned}
\tilde{\boldsymbol\theta}_t
&=\boldsymbol\theta_t+\eta\mathbf d_t,\\
\boldsymbol\theta_{t+1}
&=\tilde{\boldsymbol\theta}_t
-\alpha\nabla\mathcal L(\tilde{\boldsymbol\theta}_t).
\end{aligned}
$$

$\eta$는 탐색 이동 크기, $\alpha$는 복원 학습률이다. 실제 Algorithm 1은 $\mathbf d_t$를 단위 길이로 정규화하고 복원을 $m$번 수행한다. 복원 뒤에도 손실 상한 $\tau$를 넘으면 **이동을 취소하고** 보폭을 줄인다. 무방향 탐색에서는 주기적으로 또는 거절 뒤에 평평한 방향을 다시 구한다. 이 수치적 연속 추적(numerical continuation)은 낮은 손실 영역을 따라가는 예측·수정 구조다. 모든 해를 균일하게 표본 추출하는 방법은 아니다.

#### 2.2.1 전체 Hessian을 만들지 않는다

파라미터가 $P$개면 Hessian은 $P\times P$다. HNC는 행렬 전체 대신 행렬–벡터 곱을 사용한다. 무방향 탐색의 LOBPCG는 하위 고유벡터 일부를 구하고, 조향 탐색은 선형 시스템을 푼다. **부록 A–B에서 실제 연산자는 Gauss–Newton $\mathbf G=\mathbf J^\top\mathbf J$다.** anchor에서는 Hessian과 같지만, 출력 잔차가 생긴 다른 위치에서는 일반적으로 같지 않다. 이 구분과 증명은 [2편](/reviews/hnc-mechanics/)에서 전개한다.

상대 임계값은 $\epsilon=\mu_{\mathrm{rel}}\lambda_1$이며 $\lambda_1$은 가장 큰 고유값이다. 본문은 실험별 $10^{-7}$–$10^{-3}$ 범위를 제시한다. 부록의 sweep은 더 넓은 범위도 확인한다. LOBPCG가 찾은 $k$개 방향을 전체 영공간의 차원이라고 세면 안 된다. [§2.2.1·A–B, pp. 4, 18–20](https://arxiv.org/pdf/2609.38081v1#page=18)

#### 2.2.2 원하는 속성으로 조향하기 (Steering)

미분 가능한 목표 $\varphi$를 키우고 싶다면 gradient에서 출력을 크게 바꾸는 성분을 약하게 만든다.

$$
\begin{aligned}
\mathbf d
&=\sum_i
\frac{\mathbf v_i^\top\nabla\varphi}
{1+\lambda_i/\mu}\mathbf v_i,\\
&=(\mathbf I+\mathbf H/\mu)^{-1}\nabla\varphi,\qquad
\mu=\mu_{\mathrm{rel}}\lambda_1.
\end{aligned}
$$

평평한 성분은 거의 통과하고 가파른 성분은 작아진다. 이것이 **soft projection**이다. 0 아니면 1로 자르는 정확한 직교 투영은 아니다. 구현에서는 $\mathbf H$ 대신 $\mathbf G$를 쓰고, 역행렬을 직접 만들지 않고 conjugate gradient(CG)로 방향을 계산한다. CKA 거리를 키우거나, 동역학을 바꾸거나, 가중치 구조를 희소하게 만드는 목표를 같은 틀에 넣을 수 있다. [§2.2.2·A.3](https://arxiv.org/pdf/2609.38081v1#page=4)

<span id="3-rnn-같은-기억을-다른-동역학으로-유지하기" aria-hidden="true"></span>

## 3. 기억 과제에서 RNN의 다른 동역학 찾기 (HNC drives RNNs to learn drastically different dynamics in a memory task)

**3-bit flip-flop(3BFF)** 과제는 세 입력 채널의 마지막 0이 아닌 값을 기억한다. 세 이진 기억이므로 상태는 8개다. 저자들은 64개 tanh 유닛의 RNN을 BPTT와 Adam으로 학습한다. 기본 해는 8개 기억 상태를 안정 고정점(stable fixed point)에 저장하고, 활성 공간에서 이 점들이 큐브처럼 배치된다.

**기억 과제와 고정점을 구체적으로 보면.** 입력의 0은 “기억을 0으로 바꾸라”가 아니라 “새 지시가 없다”는 뜻이다. 한 채널에 $+1,0,0,-1,0$을 주면 첫 pulse 이전 구간을 제외한 목표는 $+1,+1,+1,-1,-1$이다. 세 채널이 독립이므로 마지막 부호의 조합이 8개다. 예시는 과제 정의를 풀어 쓴 것이며 새 실험이 아니다.

$$
\begin{aligned}
\mathbf h_t&=\tanh(\mathbf W_{\mathrm{ih}}\mathbf u_t+\mathbf b_{\mathrm{ih}}
+\mathbf W_{\mathrm{hh}}\mathbf h_{t-1}+\mathbf b_{\mathrm{hh}}),\\
\mathbf y_t&=\mathbf W_{\mathrm{out}}\mathbf h_t+\mathbf b_{\mathrm{out}},\\
\mathbf h^\star&=\tanh(\mathbf b_{\mathrm{ih}}+\mathbf W_{\mathrm{hh}}\mathbf h^\star+\mathbf b_{\mathrm{hh}})
\quad(\mathbf u_t=\mathbf0).
\end{aligned}
$$

이 고정점은 **가중치를 고정하고 입력이 0일 때의 은닉 상태**다. 고정점 근처의 작은 상태 교란이 다시 줄어들면 안정적이라고 한다. HNC가 바꾸는 것은 가중치이고, 비교하는 것은 바뀐 모델의 은닉 동역학과 출력이다. $\mathbf W_{\mathrm{out}}\mathbf h_t$가 일정해도 readout의 영공간 성분은 움직일 수 있다. 따라서 기억 출력 유지와 모든 은닉 상태의 정지는 동치가 아니다. [§3, Appendix F.1](https://arxiv.org/pdf/2609.38081v1#page=29)

HNC는 무방향, CKA 거리 $1-\mathrm{CKA}$를 키우는 조향, DSA 거리를 키우는 조향을 비교한다. CKA는 샘플 사이의 표현 유사 구조를, DSA는 시간에 따른 상태 전이 구조를 측정한다. 한 지표에서 멀어지는 것과 다른 지표에서 멀어지는 것은 같지 않다.

![원문 Figure 2: 3BFF의 고정점·출력·표현 및 동역학 변화](/assets/reviews/hnc/rnn-dynamics.webp)

[Figure 2 크게 보기](/assets/reviews/hnc/rnn-dynamics.webp)

**Figure 2를 읽는 순서.** A는 원래 8개 고정점, B는 task loss와 상대 가중치 거리 $\|\Delta\boldsymbol\theta\|/\|\boldsymbol\theta_0\|$를 보여준다. B 가운데·오른쪽에서는 조향한 방향의 표현·동역학 거리가 독립 학습 seed 쌍보다 커진다. C의 대표 사례에는 안정 고정점이 각각 0·7·15개로 보고되지만 출력은 여전히 과제를 푼다. 이는 고정점 탐색으로 관찰한 개수이며, 공간 전체의 고정점 부재를 수학적으로 인증한 결과는 아니다.

D가 이 논문의 가장 직관적인 장면이다. 출력에 보이는 **readout 부분공간**에서는 상태가 거의 멈추지만, 출력층에 보이지 않는 **readout-null 부분공간**에서는 계속 움직인다. 내부의 모든 변화를 멈추지 않아도 출력 기억을 유지할 수 있다는 뜻이다. 이 활성 공간의 null space는 §2의 가중치 공간 null space와 구분해야 한다.

E는 중요한 반대 결과다. CKA 조향 해는 숨은 상태 교란에 더 약하고, DSA 조향 해는 입력 없는 기억 구간을 길게 늘리면 drift한다. F의 MDS는 131개 모델의 쌍별 거리를 2차원으로 요약한다. 표현 거리와 가중치 거리는 분리되지만, 2차원 배치가 전체 해 공간의 위상이나 완전한 지도를 증명하지는 않는다. [§3·Figure 2, pp. 4–6](https://arxiv.org/pdf/2609.38081v1#page=4)

설정도 함께 보아야 한다. 본 실험은 5개 anchor와 각 방식·anchor당 3회 반복을 사용한다. 대표 결과는 task MSE $0.01$ 아래를 유지하며, Fig2F의 모델들은 정확도 $0.999$ 초과로 보고된다. 100-step trial, 입력 pulse 확률 0.3, 4,611개 파라미터를 사용한다. 원문 F.3의 수용 판정은 고정 128-trial의 task MSE를 적고 있어 일반 Algorithm 1의 함수 일치 손실과 실험별 제어량을 구분해야 한다.

부록 F는 약한 pulse·드문 pulse 등의 분포 밖(out-of-distribution, OOD) 조건에서 일부 대안이 더 빨리 무너짐을 보여준다. 동시에 pruning 조향으로 64→11유닛으로 줄인 한 사례, 8,000-step SGD 재학습에도 대안의 특성이 남는 사례를 제시한다. 이들은 유용한 추가 근거이지만 모든 seed·과제의 압축률이나 최적해 안정성을 보장하지 않는다. [F.1–F.8, pp. 29–35](https://arxiv.org/pdf/2609.38081v1#page=29)

<span id="4-vit-표현-정렬은-함수가-강제한-결과일까" aria-hidden="true"></span>

## 4. ViT에서 미탐색 자유도 드러내기 (HNC reveals underexplored degrees of freedom in ViTs)

PRH는 더 크고 유능한 모델들이 비슷한 표현으로 수렴하는 경향을 논한다. HNC는 다른 질문을 묻는다. **그렇게 학습된 모델 하나의 출력을 거의 유지한 채 표현을 멀리 옮길 여지는 얼마나 남아 있을까?**

anchor는 ImageNet 사전학습 ViT-S/16, 약 22M 파라미터다. Places-365 validation의 512개 이미지를 probe로 사용해 logit을 맞춘다. 13개 깊이의 CLS/pre-logit 특징에서 anchor 층과 현재 모델 층의 **모든 층 쌍 중 최대 CKA**를 낮춘다. 마지막 층만 다르게 만들고 중간 층에는 원래 표현을 남기는 쉬운 우회를 줄이려는 설계다. 실제 조향에는 매끄러운 log-sum-exp를 쓰되 평가는 hard maximum으로 한다.

![원문 Figure 3: ViT의 표현 유사도와 예측 보존을 분리한 평가](/assets/reviews/hnc/vit-representations.webp)

[Figure 3 크게 보기](/assets/reviews/hnc/vit-representations.webp)

**Figure 3에서 A·B·C를 함께 읽자.** A의 가로축은 walk step이 아니라 상대 가중치 이동량이다. CKA와 조향하지 않은 mutual-kNN 정렬이 probe와 held-out 이미지 모두에서 내려간다. B 왼쪽은 anchor와의 예측 일치율이고 오른쪽은 ImageNet 정답 정확도다. 두 지표는 다르다. 원문은 가중치 norm 약 1.3% 이동에서 ImageNet top-1 하락이 1% 미만이라고 서술하고 top-5는 거의 유지된다고 보고한다. 이는 약한 성능 손실을 동반한 근사 보존이다.

C에서는 endpoint의 CKA가 **그림에 비교한 7개 학습 모델과 무작위 초기화 ViT보다도 낮다.** 비교 모델은 ViT-S/in1k, ViT-L/in21k, ViT-B/in21k, DeiT3-S, ViT-S/DINO, ResNet-50, ConvNeXt-S다. 세상의 모든 architecture와 학습 목적보다 다르다는 보편적 순위는 아니다. A–C의 band는 5개 무작위 probe 집합에 대한 평균±표준편차다.

D의 cosine similarity 행렬과 E의 20개 이미지 MDS는 이웃 관계가 어떻게 달라졌는지 보여준다. F는 각 모델에서 가장 비슷한 이미지 쌍 5개를 골라, 의미(DINOv2), 색 분포, 공간 배치의 유사도를 전체 쌍 중 percentile로 비교한다. 50은 무작위 쌍 기준이다. endpoint는 의미·색의 유사성을 상당히 유지하면서 공간 배치에 덜 민감해진다. 5개 쌍에서 관찰한 속성 변화이며 시각 지각 전체에 대한 결론은 아니다. [§4·G.1–G.2, pp. 6–7, 36](https://arxiv.org/pdf/2609.38081v1#page=6)

**probe 밖 검증이 결론의 강도를 결정한다.** G.5의 CKA 조향 해는 ImageNet, V2, Rendition, Sketch에서 작은 정확도 하락과 불완전한 예측 일치를 보인다. 오류 일관성(error consistency)은 약 0.8이며, anchor의 마지막 fine-tuning을 다른 seed로 반복한 비교와 비슷하거나 더 높은 일치를 보고한다. 동일 정확도 손실을 주는 무작위 가중치 노이즈는 CKA를 0.98 위에 남기므로 단순 손상만으로 표현 변화가 설명되지는 않는다.

반면 G.6의 mutual-kNN 조향 설정은 clean ImageNet 예측 일치가 약 0.85, Sketch에서는 약 0.6까지 내려간다. 여기서는 CKA 조향보다 기능 보존이 약하다. 두 설정의 결과를 합쳐 “표현은 언제나 자유롭게 바꾸고 기능은 그대로다”라고 쓰면 안 된다. probe 수, 목적, 보폭의 trade-off가 있다. G.3의 kNN probe는 256개로 기술되지만 Table 8은 512개여서 해당 설정 수는 원문 내부 불일치로 남긴다. [G.3–G.10, pp. 36–46](https://arxiv.org/pdf/2609.38081v1#page=39)

<span id="5-강화학습-같은-행동이-아니라-비슷한-보상" aria-hidden="true"></span>

## 5. 강화학습에서 행동이 다른 고보상 정책 찾기 (HNC finds behaviorally distinct, high-reward policies in reinforcement learning)

여기서 **보존하는 대상이 바뀐다.** 같은 행동 분포를 유지하면 행동 전략을 크게 바꿀 수 없다. 저자들은 anchor $\pi_{\boldsymbol\theta_0}$에서 얻은 상태·행동 buffer와 advantage $A_i^0$를 이용해 다음 surrogate return을 보존한다.

$$
\phi_R(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
\frac{\pi_{\boldsymbol\theta}(a_i\mid s_i)}
{\pi_{\boldsymbol\theta_0}(a_i\mid s_i)}A_i^0.
$$

한편 행동 차이를 키우는 목표는 다음과 같다.

$$
\phi_B(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
D_{\mathrm{KL}}\!\left(
\pi_{\boldsymbol\theta_0}(\cdot\mid s_i)
\,\Vert\,
\pi_{\boldsymbol\theta}(\cdot\mid s_i)\right).
$$

$s_i,a_i$는 관찰 상태와 행동이고 $N$은 buffer 표본 수다. 이 surrogate는 모든 미래 상태 분포에서 실제 return을 정확히 계산한 값이 아니다. 정책이 변하면 방문 상태도 변하므로 buffer를 주기적으로 다시 수집하고 실제 episode return을 따로 평가한다. 연속 행동 실험은 공분산을 고정한 Gaussian policy의 평균 행동 MSE를 조향한다. 차원별 분산이 같을 때에만 KL과 전체 상수배로 같고, 그렇지 않으면 고정 대각 가중치가 다르다. [§5·I.1–I.3, pp. 7–8, 49–50](https://arxiv.org/pdf/2609.38081v1#page=7)

![원문 Figure 4: plume tracking의 다른 전략과 boat race의 reward hacking](/assets/reviews/hnc/rl-strategies.webp)

[Figure 4 크게 보기](/assets/reviews/hnc/rl-strategies.webp)

**왼쪽 A–C: 냄새를 쫓는 다른 방법.** PPO로 학습한 64유닛 RNN은 냄새를 잃으면 바람을 가로질러 반복 탐색(casting)한다. HNC 대안은 plume 가장자리를 더 매끄럽게 따라간다. B는 action KL 증가와 episode return을 함께 보여준다. C는 동일한 초기 조건 240개에서 casting 횟수와 중심선 거리를 짝지어 비교한다. casting 그래프의 점이 대각선 아래에 많다는 것은 대안이 덜 왕복한다는 뜻이다. 궤적 세 개만으로 성공률을 비교한 결과가 아니다.

30-step walk는 5 step마다 buffer를 갱신하고, 정규 평가는 공통 난수와 고정 초기 조건 60개를 사용한다. 추가 OOD 평가는 조건별 240개를 사용한다. 바람 전환·희박한 냄새·noisy wind에서 선택된 대안은 개선되지만, I.5는 step 20에서 여러 조건의 성능이 함께 무너져 보존 영역을 벗어나는 구간도 보고한다. 가장 먼 endpoint가 언제나 가장 좋은 정책은 아니다.

**오른쪽 D–E: 보상을 유지해도 의도는 깨질 수 있다.** Boat race는 화살표 타일을 정방향으로 진입할 때 +3의 proxy reward를 주지만, 의도한 성과는 트랙을 시계방향으로 실제 진행하는 것이다. HNC는 타일 하나를 드나들며 proxy를 받는 **보상 해킹(reward hacking)** 정책을 찾는다. E에서는 proxy return과 true return을 다른 그래프로 봐야 한다. 가장 많이 달라진 3개 정책은 순진행이 0이 된다.

설정의 중요한 조건이 있다. **10개 anchor는 true return으로 PPO 학습했다.** proxy만으로 학습하면 exploit을 스스로 발견하므로, 먼저 정상 순환 정책을 만들고 HNC가 proxy를 보존하며 해킹을 찾는 실험이다. 정상적인 proxy-only 학습의 모든 결과가 HNC 때문에 위험해진다는 주장은 아니다. 추가 Ant-v3 실험도 평균 return을 유지하며 다리 타이밍과 실패 유형이 달라짐을 보여준다. 평균 보상은 초기 상태별 역량까지 동일하게 만들지 않는다. [§5·I.4–I.7, pp. 8, 51–53](https://arxiv.org/pdf/2609.38081v1#page=51)

<span id="6-hnc로-손실-지형의-기하를-측정하기" aria-hidden="true"></span>

## 6. 손실 지형의 기하 측정 (HNC measures loss landscape geometry)

마지막 본문 실험은 “같은 정확도로 학습된 모델도 자유도가 얼마나 남는가?”를 묻는다. CIFAR-100의 클래스 수 $C\in\{7,10,20,50,100\}$와 CNN 폭 $w\in\{16,24,32,48,96\}$를 바꿔 3 seed씩, **75개 anchor**를 만든다. 클래스당 학습 이미지 300개, augmentation 없이 학습하고 각 모델의 학습 이미지 512개로 출력을 제약한다. 클래스 수는 여기서 과제 복잡도의 대리 변수다.

![원문 Figure 5: 폭과 클래스 수에 따른 effective null fraction 및 정규화 곡률](/assets/reviews/hnc/geometry.webp)

[Figure 5 크게 보기](/assets/reviews/hnc/geometry.webp)

**A와 B는 다른 측정이다.** A의 effective null fraction은 가중치 norm의 1%만큼 움직였을 때 함수 손실 증가가 $0.05$를 넘지 않는 방향들의 비율이다. 폭이 커지면 증가하고 클래스가 많아지면 감소한다. “정확한 영공간의 차원을 실험 조건과 무관하게 측정했다”는 뜻은 아니다.

B는 공통 상대 임계값 $\mu=10^{-6}\lambda_{\max}$에서 이동 방향의 정규화 곡률을 본다. 원문 표기는 $\mathbf d^\top\mathbf H\mathbf d/(nC)$이고, 구현·부록 J.3에서는 $\mathbf G$를 사용한다. $\mathbf d$는 단위 방향, $n$은 probe 수다. 클래스가 많으면 민감도가 커지고 폭이 넓으면 낮아지는 경향이다. band는 3 seed의 표준오차다. 기준을 여러 값으로 바꾼 부록에서도 경향은 유지된다.

클래스가 늘면 출력 제약 수 자체도 늘어난다. 이를 모든 의미의 “어려운 과제”에 대한 보편 법칙이나 서로 분리된 basin의 존재 증명으로 확대하면 안 된다. J.1의 “학습 정확도 100%이므로 exact zero-loss”라는 문장은 cross-entropy에서는 일반적으로 성립하지 않는다. 다만 anchor 자신의 출력을 맞추는 손실이 0이라는 HNC 정의는 별개로 성립한다. [§6·J.1–J.4, pp. 8–9, 53–55](https://arxiv.org/pdf/2609.38081v1#page=8)

## 7. 논의: 과제가 허용하는 해와 학습이 고른 해 (Discussion)

이 논문이 보여주는 간극은 **가능한 해의 다양성**과 **일반적인 최적화가 실제로 발견하는 해의 다양성** 사이에 있다. 큰 모델들이 서로 닮아간다는 관찰과, 큰 모델 하나 주변에 다양한 대안이 남는다는 관찰은 동시에 성립할 수 있다. 함수만으로 표현이 유일하게 결정된다고 보기는 어렵다.

주요 한계도 연구 결론의 일부다. 보존은 유한 probe에서 근사적이고, OOD·장기 기억·교란에서 달라질 수 있다. 탐색은 local이며 전체 basin 수를 알려주지 않는다. 조향 지표와 허용 오차가 바뀌면 도달하는 대안도 달라진다. 행렬을 직접 저장하지 않아도 곡률 곱·복원·CG·추가 rollout 비용은 있다. ViT-S는 H200에서 continuation step 약 1분으로 보고되며, 작은 RNN에서는 오히려 dense 계산이 더 빠르다.

부록 H의 Brain-Score 조향은 같은 모델의 내부 특징으로 신경 반응을 예측하는 점수가 달라질 수 있음을 추가한다. 그러나 뇌 자체가 변했다는 실험이 아니며 최적화한 ridge surrogate와 보고한 PLS 기반 공식 점수는 다르다. LLM 확장, model merging·editing·fine-tuning의 일반적 개선은 Discussion의 가능성이지 본문에서 입증한 성능 향상표가 아니다. [§7·A.1·H, pp. 9–10, 19, 46–48](https://arxiv.org/pdf/2609.38081v1#page=9)

## 리뷰 해석과 남은 질문

**리뷰 해석.** HNC의 실용적인 독해는 “같은 기능을 구현하는 진짜 구조 하나를 찾아낸다”보다, **어떤 관찰만으로는 내부 메커니즘을 식별할 수 없는지 시험한다**에 가깝다. 후속 검증에서는 probe 수와 분포, 허용 drift, 실제 계산 예산을 맞춘 뒤 표현 변화·정확도·일치율·OOD·실패를 함께 비교할 필요가 있다. 조향한 지표 하나만 크게 움직이는 것으로 충분하지 않다. 이는 리뷰의 제안이며 실행한 후속 실험은 아니다.

## 출처와 읽은 범위

- 원문: *Traversing the solution space of neural networks with Hessian Null Space Continuation* · Ann Huang et al. · 최초 공개 2026-09-29 · [검토 버전](https://arxiv.org/abs/2609.38081v1) · 검토일 2026-10-23.
- 읽은 범위: v1 본문 §1–7. 설정·지표·반대 결과에 필요한 부록을 연결하며 A–J의 상세는 [2/2] 심층 편에 둔다.
- 그림 출처: 대표 Figure 1과 해당 편의 본문 그림은 저자 v1 자산(CC BY 4.0). PDF 래스터화·WebP 인코딩으로 해상도를 바꾸고 내용·배치·글자·화살표를 유지한다. [원문](https://arxiv.org/abs/2609.38081v1).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
