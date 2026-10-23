---
title: "[1/2] HNC: 같은 출력을 내는 신경망은 내부에서도 같은 계산을 할까?"
description: "한 모델 주변의 함수 보존 영역을 따라가며 RNN의 기억, ViT의 표현, 강화학습 정책을 바꾸는 HNC를 원문 §1–7 순서로 읽는다."
date: "2026-10-17"
publishedAt: "2026-10-17T22:12:00+09:00"
updatedAt: "2026-10-23T15:26:23+00:00"
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

헤시안 영공간 연속 탐색(Hessian Null Space Continuation, HNC)은 학습된 모델 하나(원문의 anchor)의 출력을 고정된 입력 집합(probe set)에서 근사적으로 유지한 채, 손실 곡률이 평평한 가중치 방향으로 조금씩 이동해 내부 계산이 다른 대안 모델을 찾는다. ImageNet 사전학습 비전 트랜스포머(vision Transformer, ViT) ViT-S/16에서는 가중치를 norm 기준 1.3%만 옮기고 ImageNet top-1 하락을 1% 미만으로 유지하면서, 층 간 최대 중심 커널 정렬(centered kernel alignment, CKA)이 비교한 7개 학습 모델과 무작위 초기화 ViT보다도 낮아진다. [원문 §4, pp. 6–7](https://arxiv.org/pdf/2609.38081v1#page=6) **리뷰 해석.** "다른 구조·학습 목적으로 독립 학습한 어떤 모델보다 더 다른 표현"이라는 초록의 주장은 이 7개 모델과 CKA 기준에서 probe 밖 평가와 무작위 가중치 노이즈 대조로 뒷받침되지만, ViT에서 보인 것은 표현 유사도의 변화이고 기억을 구현하는 방식 자체가 바뀌었다는 메커니즘 수준의 증거는 64유닛 순환 신경망(recurrent neural network, RNN) 실험에 있다.

이 글은 본문 §1–7을 따라 방법과 네 실험군을 설명하고, [\[2/2\] 수학·알고리즘·검증 심화](/reviews/hnc-mechanics/)는 부록 A–J의 유도·절제 실험·probe 밖 검증을 다룬다.

## 1. 도입: 연결된 해들 안에는 어떤 계산이 있는가? (Introduction)

과매개변수화(overparameterization)된 신경망에는 손실이 비슷하게 낮은 국소 최소점이 많고, 어느 최소점에 도달하는지는 최적화기·데이터·구조·과제에 달려 있다. 손실 지형 연구는 학습된 해 주변에 거의 평평한 방향이 많다는 것을, **모드 연결성(mode connectivity)** 연구는 독립적으로 학습한 해들이 손실이 낮은 비선형·선형 경로로 이어지곤 한다는 것을 보였다. 저자들에 따르면 이 연구들은 연결 영역을 주로 연결성과 손실로 기술했고, 내부 계산까지 본 Lubana et al.(2023)도 학습으로 얻은 네트워크 쌍을 비교했다. 한 학습 해를 둘러싼 낮은 손실 영역에 정말 다른 표현과 메커니즘이 있는지는 열려 있다.

한편 **해의 퇴화성(solution degeneracy)** 연구는 한 과제가 분포 밖 일반화와 내부 표현의 기하·동역학이 서로 다른 해의 가족을 허용함을 보였다. 여기서 degeneracy는 하나의 과제에 양립하는 해가 여럿이라는 뜻이다. 유사도 패널티나 hard constraint를 건 재학습, saddle에서 헤시안 고유벡터를 따라가는 방법으로 대안 해를 일부러 찾은 연구도 있지만, 이 해들이 가중치 공간에서 어떻게 관계되는지는 잘 알려지지 않았다. 저자들의 질문은 두 흐름이 만나는 곳에 있다. **한 학습 해를 둘러싼, 함수를 보존하는 연결 영역 안에 표현 다양성이 얼마나 있고, 그 영역을 어떻게 구성적으로 가로지를 수 있는가?**

동기는 두 가지다. 학습이 고르는 해는 과제뿐 아니라 최적화기의 편향(예: SGD의 low-rank·low-norm 해 선호)에도 제약되므로, 모델이 허용하는 해 전체를 보면 과제가 요구하는 구조와 최적화기의 영향을 나눠 볼 수 있다. 실용적으로는 함수를 보존하는 국소 이동으로 다른 표현에 닿을 수 있다면 fine-tuning, model merging, 사후 편집에 쓸 자유도가 드러난다. HNC는 anchor에서 출발해 국소 곡률 정보로 함수 보존 대안을 탐색하고, 미분 가능한 목적으로 원하는 성질 쪽에 조향할 수 있다. [원문 §1, pp. 1–3](https://arxiv.org/pdf/2609.38081v1#page=1)

## 2. 방법 (Method)

### 2.1 손실 곡률에서 함수 보존 방향 찾기

가중치 벡터를 $\boldsymbol\theta$, anchor를 $\boldsymbol\theta_0$, 입력 $x$에 대한 출력을 $f_{\boldsymbol\theta}(x)$라고 하자. 원문 식 (1)은 정답과의 과제 손실이 아니라 <strong>함수 일치 손실(function-matching loss)</strong>을 정의한다.

$$
\mathcal L(\boldsymbol\theta)
=\frac12\mathbb E_{x\sim\mathcal D}
\left[\left\|f_{\boldsymbol\theta}(x)
-f_{\boldsymbol\theta_0}(x)\right\|^2\right].
$$

$\mathcal D$는 입력 분포다. 실제로는 여기서 뽑은 고정 입력 집합 $\mathbf X$, 즉 **probe set**으로 기대값을 추정한다. 분류 모델에서는 정답 라벨이 아니라 anchor의 logit 벡터 전체를 맞춘다.

anchor는 자신의 출력을 정확히 재현하므로 손실이 0인 전역 최소점이고 gradient도 0이다. 따라서 작은 가중치 변화 $\boldsymbol\delta$는 원문 식 (2)처럼 손실을 2차 항부터 바꾼다.

$$
\begin{aligned}
\mathcal L(\boldsymbol\theta_0+\boldsymbol\delta)
&=\frac12\boldsymbol\delta^\top\mathbf H\boldsymbol\delta
+O(\|\boldsymbol\delta\|^3),\\
\mathbf H&=\nabla^2_{\boldsymbol\theta}\mathcal L(\boldsymbol\theta_0).
\end{aligned}
$$

헤시안 $\mathbf H$의 큰 고유값은 출력이 빠르게 변하는 **가파른 방향(sharp direction)**, 0에 가까운 고유값은 출력 drift가 작은 <strong>평평한 방향(flat direction)</strong>을 가리킨다. 원문은 고유벡터 $\mathbf v_i$와 고유값 $\lambda_i$에 대해 근사 영공간(approximate null space)을 정의한다.

$$
\mathcal V_0=\operatorname{span}
\{\mathbf v_i:\lambda_i\le\epsilon\}.
$$

저자들은 학습된 네트워크의 헤시안이 큰 고유값 몇 개와 0에 가까운 고유값 다수로 이루어져, 이 근사 영공간이 파라미터 차원의 상당 부분을 차지하는 경우가 많다고 설명한다. [§2.1, p. 3](https://arxiv.org/pdf/2609.38081v1#page=3)

### 2.2 이동하고, 복원하고, 다시 방향을 계산한다

![원문 Figure 1: 평평한 가중치 방향으로 이동하고 출력을 복원하는 HNC](/assets/reviews/hnc/overview.webp)

[Figure 1 크게 보기](/assets/reviews/hnc/overview.webp)

**Figure 1의 핵심은 A→D→B다.** A의 높이는 함수 일치 손실이고 바닥의 두 축은 가중치 방향을 개념적으로 그린 것이다. D의 보라색 화살표는 평평한 방향의 이동(flat step), 청록색은 출력을 다시 맞추는 복원(restore)이다. 이를 반복하면 B처럼 입출력 대응은 같지만 같은 입력의 내부 표현이 다른 네트워크에 닿는다. B의 PC 축은 내부 활성의 주성분으로, A의 가중치 축과 다른 공간이다. C는 고유값 임계값 $\epsilon$으로 평평한 방향을 고르는 단계, E는 원하는 목적 쪽으로 이동을 조향하는 단계다. [Figure 1, p. 2](https://arxiv.org/pdf/2609.38081v1#page=2)

영공간 안에서 한 번 이동하면 anchor 바로 주변만 탐색하므로, 더 멀리 가려면 여러 단계가 필요하다. 저자들은 여기서 두 문제를 짚는다. 헤시안은 2차 근사라서 고유값이 0인 방향으로 움직여도 고차 항 때문에 손실이 변할 수 있고, anchor에서 멀어지면 그 영공간이 실제 평평한 방향과 어긋날 수 있다. HNC는 영공간 이동과 함수 복원을 번갈아 두 문제를 다룬다(원문 식 (3)).

$$
\begin{aligned}
\tilde{\boldsymbol\theta}_t
&=\boldsymbol\theta_t+\eta\mathbf d_t,\\
\boldsymbol\theta_{t+1}
&=\tilde{\boldsymbol\theta}_t
-\alpha\nabla\mathcal L(\tilde{\boldsymbol\theta}_t).
\end{aligned}
$$

$\eta$는 탐색 이동 크기, $\alpha$는 복원 학습률이다. 식은 복원을 한 번만 적지만 실제로는 $m$번 수행하고, Algorithm 1은 $\mathbf d_t$를 단위 길이로 정규화한다. 복원 뒤에도 손실이 상한 $\tau$를 넘으면 헤시안 선형 근사가 맞지 않게 됐다는 신호로 보고, **이동을 취소하고** 보폭을 줄인 뒤 현재 위치에서 영공간을 다시 계산한다(재선형화, relinearization). 저자들은 이 구조를 접선 방향으로 조금 이동한 뒤 곡선 위로 되돌리는 고전적 수치 연속법(numerical continuation)의 예측·수정 구조에 빗댄다.

#### 2.2.1 전체 헤시안을 만들지 않는다

파라미터가 $P$개면 헤시안은 $P\times P$라 저장도 분해도 어렵다. HNC는 자동 미분으로 계산한 행렬–벡터 곱만 쓰는 LOBPCG(locally optimal block preconditioned conjugate gradient)로 가장 작은 고유쌍 $k$개를 구해, 시간·메모리를 $O(P^3)$·$O(P^2)$ 대신 $O(kP)$로 줄인다. anchor에서 헤시안은 양의 준정부호(positive semidefinite)이므로 영공간은 수치적으로 가장 작은 고유값들의 고유벡터가 된다. **부록 A–B에서 실제로 쓰는 연산자는 가우스–뉴턴(Gauss–Newton) $\mathbf G=\mathbf J^\top\mathbf J$다($\mathbf J$는 probe 출력의 야코비안(Jacobian)).** anchor에서는 헤시안과 같고 탐색 내내 양의 준정부호로 남으며, 유도는 [\[2/2\] B](/reviews/hnc-mechanics/)에서 다룬다.

상대 임계값은 $\epsilon=\mu_{\mathrm{rel}}\lambda_1$이며 $\lambda_1$은 현재 선형화 지점의 가장 큰 고유값이다. 상대 임계값이라 재선형화 때마다 영공간을 다시 계산하고, $\mu_{\mathrm{rel}}$은 실험별로 $10^{-7}$–$10^{-3}$에서 고른다. 부록의 sweep은 더 넓은 범위도 확인한다. [§2.2.1·A–B, pp. 4, 18–20](https://arxiv.org/pdf/2609.38081v1#page=18)

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

평평한 성분은 거의 통과하고 가파른 성분은 작아지므로 **soft projection**이라 부른다. 구현에서는 $\mathbf H$ 대신 $\mathbf G$를 쓰고, 역행렬을 직접 만들지 않고 conjugate gradient(CG)로 방향을 계산한다. CKA 거리를 키우거나, 동역학을 바꾸거나, 가중치 구조를 희소하게 만드는 목표를 같은 틀에 넣을 수 있다. [§2.2.2·A.3](https://arxiv.org/pdf/2609.38081v1#page=4)

<span id="3-rnn-같은-기억을-다른-동역학으로-유지하기" aria-hidden="true"></span>

## 3. 기억 과제에서 RNN의 다른 동역학 찾기 (HNC drives RNNs to learn drastically different dynamics in a memory task)

**3-bit flip-flop(3BFF)** 과제에서 네트워크는 세 입력 채널마다 마지막 0이 아닌 입력을 기억해 내보낸다. 세 이진 기억이므로 상태는 $2^3=8$개다. 저자들은 64개 tanh 유닛의 RNN을 시간 역전파(backpropagation through time, BPTT)와 Adam으로 학습한다. 학습된 anchor는 8개 기억 상태를 입력을 꺼도 변하지 않는 안정 고정점(stable fixed point) 8개에 저장하고, 이 점들은 활성 공간에서 큐브의 꼭짓점을 이룬다.

**리뷰 보충: 기억 과제와 고정점.** 입력의 0은 “기억을 0으로 바꾸라”가 아니라 “새 지시가 없다”는 뜻이다. 한 채널에 $+1,0,0,-1,0$을 주면 첫 pulse 이전 구간을 제외한 목표는 $+1,+1,+1,-1,-1$이다. 부록 F.1의 RNN 갱신식과, 입력을 끈 고정점 조건은 다음과 같다.

$$
\begin{aligned}
\mathbf h_t&=\tanh(\mathbf W_{\mathrm{ih}}\mathbf u_t+\mathbf b_{\mathrm{ih}}
+\mathbf W_{\mathrm{hh}}\mathbf h_{t-1}+\mathbf b_{\mathrm{hh}}),\\
\mathbf y_t&=\mathbf W_{\mathrm{out}}\mathbf h_t+\mathbf b_{\mathrm{out}},\\
\mathbf h^\star&=\tanh(\mathbf b_{\mathrm{ih}}+\mathbf W_{\mathrm{hh}}\mathbf h^\star+\mathbf b_{\mathrm{hh}})
\quad(\mathbf u_t=\mathbf0).
\end{aligned}
$$

고정점 근처의 작은 상태 교란이 다시 줄어들면 안정적이라고 한다. 출력 $\mathbf W_{\mathrm{out}}\mathbf h_t$가 일정해도 readout의 영공간 성분은 움직일 수 있는데, Figure 2D가 바로 이 경우다. [§3, Appendix F.1](https://arxiv.org/pdf/2609.38081v1#page=29)

HNC는 무방향 탐색과, 두 거리를 키우는 조향을 비교한다. CKA 거리 $1-\mathrm{CKA}$는 은닉 활성의 표현 기하가, 동역학 유사도 분석(dynamical similarity analysis, DSA) 거리는 가역 변환을 허용한 채 순환 동역학이 얼마나 바뀌었는지를 잰다. 탐색에는 적응적 보폭과 주기적 재선형화를 쓴다.

![원문 Figure 2: 3BFF의 고정점·출력·표현 및 동역학 변화](/assets/reviews/hnc/rnn-dynamics.webp)

[Figure 2 크게 보기](/assets/reviews/hnc/rnn-dynamics.webp)

**Figure 2를 읽는 순서.** A는 과제와 anchor의 8개 고정점이다. B 왼쪽에서 모든 HNC walk의 task loss는 $0.01$ 아래에 머물고, 가운데·오른쪽은 상대 가중치 이동량 $\|\Delta\boldsymbol\theta\|/\|\boldsymbol\theta_0\|$에 따른 표현·동역학 거리다. 무방향 walk는 독립 학습 seed 쌍과 비슷한 거리에 이르고, 조향 walk는 더 적은 가중치 이동으로 그보다 크게 멀어진다. C에서 무방향 endpoint는 8개 고정점이 사라지고 기억 상태마다 영역이 하나씩 있는 연속 manifold가 생기며, CKA·DSA 조향 endpoint는 고정점 수 자체가 달라진다(그림 표기로 안정 고정점 0·7·15개). 세 endpoint 모두 과제를 푼다.

D가 이 논문의 가장 직관적인 장면이다. 무방향 endpoint의 은닉 상태는 입력이 없는 기억 구간에 출력에 보이는 **readout 부분공간**에서는 8개의 작은 점으로 모이지만, 출력층에 보이지 않는 **readout-null 부분공간**(활성 공간의 부분공간으로, §2의 가중치 영공간과 다르다)에서는 8개 군집 안을 계속 움직인다. 저자들은 이 네트워크가 고정점이 아니라 상태 공간의 구분된 영역에 기억을 담아, 내부 동역학이 계속되는데도 출력을 안정적으로 유지한다고 해석한다.

E에서 CKA 조향 endpoint는 은닉 상태 교란 뒤 덜 회복하고, DSA 조향 endpoint는 입력 없는 기억 구간이 길어지면 기억이 drift한다. 저자들은 분포 안의 입출력이 같아도 연장·교란 조건에서는 행동이 달라질 수 있다고 정리한다. F의 다차원 척도법(multidimensional scaling, MDS)은 anchor, 독립 학습 seed 10개, 방식별 walk 3개씩의 모든 checkpoint로 이루어진 131개 모델의 쌍별 거리를 2차원에 배치한다. 표현 거리에서는 독립 seed와 무방향 walk가 anchor 근처에 남고 조향 walk만 멀리 뻗지만, 가중치 거리에서는 조향 walk가 무방향 walk와 독립 seed보다 anchor에 가깝다. 저자들은 이를 표현 거리와 가중치 거리의 분리로 본다. [§3·Figure 2, pp. 4–6](https://arxiv.org/pdf/2609.38081v1#page=4)

본 실험은 5개 anchor와 방식·anchor당 3회 반복을 사용하고, Figure 2F의 131개 모델은 모두 정확도 $0.999$를 넘는다. trial은 100 step, 입력 pulse 확률은 0.3, 모델 파라미터는 4,611개다. 부록 F는 약한·드문 pulse 같은 분포 밖(out-of-distribution, OOD) 입력에서 일부 대안이 anchor보다 먼저 정확도를 잃는 결과, 희소화(pruning) 조향으로 64유닛을 11유닛까지 줄인 사례, 8,000-step SGD 재개 뒤에도 조향한 특성이 남는 결과를 더한다([\[2/2\] F](/reviews/hnc-mechanics/)). [F.1–F.8, pp. 29–35](https://arxiv.org/pdf/2609.38081v1#page=29)

<span id="4-vit-표현-정렬은-함수가-강제한-결과일까" aria-hidden="true"></span>

## 4. ViT에서 미탐색 자유도 드러내기 (HNC reveals underexplored degrees of freedom in ViTs)

플라톤적 표현 가설(Platonic Representation Hypothesis, PRH)은 모델이 커지고 과제가 어려워질수록 표현이 서로 비슷해진다고 보고, 이를 공통된 현실 표현으로의 수렴으로 설명한다. HNC는 다른 질문을 묻는다. **학습된 모델 하나의 함수를 거의 고정했을 때 표현을 바꿀 자유가 얼마나 남아 있을까?**

anchor는 ImageNet 사전학습 ViT-S/16, 약 22M 파라미터다. PRH 평가에 쓰인 Places-365 validation의 512개 이미지를 probe로 사용해 logit을 맞춘다. 블록마다의 CLS 특징과 마지막 pre-logit 특징, 13개 깊이에서 anchor 층과 현재 모델 층의 **모든 층 쌍 중 최대 CKA**를 낮추도록 조향한다. hard maximum은 미분할 수 없어 조향에는 매끄러운 log-sum-exp를 쓰고, 보고는 hard maximum으로 한다.

![원문 Figure 3: ViT의 표현 유사도와 예측 보존을 분리한 평가](/assets/reviews/hnc/vit-representations.webp)

[Figure 3 크게 보기](/assets/reviews/hnc/vit-representations.webp)

**Figure 3에서 A·B·C를 함께 읽자.** A의 가로축은 walk step이 아니라 상대 가중치 이동량이다. CKA는 probe와 HNC 중 한 번도 쓰지 않은 held-out 이미지 모두에서 내려가고, 직접 조향하지 않은 상호 k-최근접 이웃(mutual k-nearest-neighbor, kNN) 정렬도 비슷하게 내려간다. 저자들은 이를 특정 지표나 이미지 집합을 이용한 결과가 아니라 표현 전반이 재조직된 증거로 본다. B 왼쪽은 anchor와의 예측 일치율, 오른쪽은 ImageNet 정답 정확도다. walk 내내 probe 이미지의 top-1 예측은 유지되고, 제약에 쓰지 않은 ImageNet에서 top-1 하락은 1% 미만, top-5는 거의 그대로다.

C에서 endpoint의 가중치는 norm 기준 1.3%만 바뀌었지만 CKA는 **비교한 7개 학습 모델과 무작위 초기화 ViT보다도 낮다.** 비교 모델은 ViT-S/in1k, ViT-L/in21k, ViT-B/in21k, DeiT3-S, ViT-S/DINO, ResNet-50, ConvNeXt-S로 크기·구조·학습 목적·데이터가 다르다. A–C의 band는 5개 무작위 probe 집합에 대한 평균±표준편차다.

D의 cosine similarity 행렬은 probe 이미지 쌍의 유사도가 anchor와 endpoint 사이에 크게 바뀌었음을, E의 4개 클래스 20개 이미지 MDS는 클래스 안팎의 이웃 구조가 재조직됨을 보여 준다. F는 각 모델에서 가장 비슷한 이미지 쌍 5개를 골라 의미(DINOv2), 색 분포, 공간 배치의 유사도를 전체 쌍 중 percentile로 매긴다. 50은 무작위 쌍 기준이다. endpoint의 가까운 쌍은 의미·색은 여전히 비슷하지만 공간 배치 유사도는 무작위 기준에 가까워진다. 저자들은 이를 근거로, 모델 간 표현 수렴을 과제가 완전히 정한 유일한 표현의 증거로 읽기보다 해의 좁은 부분집합만 고르는 최적화 편향의 결과일 수 있다고 본다. [§4·G.1–G.2, pp. 6–7, 36](https://arxiv.org/pdf/2609.38081v1#page=6)

**probe 밖 기능 보존(부록 G.5–G.6).** CKA 조향 endpoint는 ImageNet, V2, Rendition, Sketch에서 작은 정확도 손실을 보이고 오류 일관성(error consistency)은 약 0.8이다. anchor와의 이미지 수준 일치는 anchor의 마지막 fine-tuning을 seed만 바꿔 반복한 모델과 비슷하거나 대체로 높다. 정확도 손실을 맞춘 무작위 가중치 노이즈는 CKA를 0.98 위에 남기므로, 저자들은 표현 변화를 단순한 성능 손상으로 설명할 수 없다고 본다. 반면 mutual-kNN 조향 endpoint는 seed만 바꾼 비교 모델보다 보존이 약하며, 저자들은 두 실험을 합쳐 기능 보존의 정도가 탐색 설정과 조향 목적에 달려 있다고 정리한다. 세부 수치와 반복 단위는 [\[2/2\] G](/reviews/hnc-mechanics/)에 있다. [G.5–G.6, pp. 39–42](https://arxiv.org/pdf/2609.38081v1#page=39)

<span id="5-강화학습-같은-행동이-아니라-비슷한-보상" aria-hidden="true"></span>

## 5. 강화학습에서 행동이 다른 고보상 정책 찾기 (HNC finds behaviorally distinct, high-reward policies in reinforcement learning)

지도학습과 달리 강화학습의 스칼라 보상은 에피소드의 최적 행동 순서를 다 정하지 않는다. 저자들은 이 과소결정(underspecification) 때문에 보상이 비슷하면서 행동이 다른 정책, 나아가 과제 대신 보상 함수를 악용하는 **보상 해킹(reward hacking)** 정책이 있을 수 있다고 보고, 여기서는 **행동 분포가 아니라 return을 보존한다.** anchor 정책 $\pi_{\boldsymbol\theta_0}$의 rollout에서 모은 상태·행동 buffer와 advantage $A_i^0$로 만든 중요도 표본(importance sampling) surrogate return은 다음과 같다.

$$
\phi_R(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
\frac{\pi_{\boldsymbol\theta}(a_i\mid s_i)}
{\pi_{\boldsymbol\theta_0}(a_i\mid s_i)}A_i^0.
$$

행동 차이를 키우는 조향 목표는 buffer 상태에서 두 정책의 행동 분포 KL이다.

$$
\phi_B(\boldsymbol\theta)
=\frac1N\sum_{i=1}^N
D_{\mathrm{KL}}\!\left(
\pi_{\boldsymbol\theta_0}(\cdot\mid s_i)
\,\Vert\,
\pi_{\boldsymbol\theta}(\cdot\mid s_i)\right).
$$

$s_i,a_i$는 관찰 상태와 행동이고 $N$은 buffer 표본 수다. 정책이 바뀌면 방문 상태 분포도 바뀌므로 buffer를 주기적으로 현재 정책에서 다시 모아 surrogate와 그 영공간을 다시 계산하고, 실제 episode return은 따로 평가한다. 연속 행동에서는 공분산을 고정한 Gaussian 정책의 평균 행동 MSE를 조향한다([\[2/2\] I.2](/reviews/hnc-mechanics/)). [§5·I.1–I.3, pp. 7–8, 49–50](https://arxiv.org/pdf/2609.38081v1#page=7)

![원문 Figure 4: plume tracking의 다른 전략과 boat race의 reward hacking](/assets/reviews/hnc/rl-strategies.webp)

[Figure 4 크게 보기](/assets/reviews/hnc/rl-strategies.webp)

**왼쪽 A–C: 냄새를 쫓는 다른 방법.** 근접 정책 최적화(proximal policy optimization, PPO)로 학습한 64유닛 RNN anchor는 실제 파리처럼 냄새를 감지하면 바람을 거슬러 돌진하고, 냄새를 잃으면 바람을 가로질러 왕복 탐색(casting)한다. HNC 대안은 plume 가장자리로 비켜 나가 매끄럽게 따라간다. B에서 action KL은 계속 커지지만 episode return과 성공률은 유지되고 walk 초반에는 오히려 오른다. C는 같은 초기 조건 240개에서 casting 횟수와 중심선 거리를 짝지어 비교하며, 대안은 덜 왕복하고 평균적으로 중심선에서 더 멀리 머문다.

30-step walk는 5 step마다 buffer를 갱신하고, 정규 평가는 공통 난수와 고정 초기 조건 60개를 쓴다. 조건별 240개의 OOD 평가(I.5)에서 anchor는 희박한 냄새·noisy wind의 모든 episode에 실패하지만 분석한 대안은 일부를 성공하고, 바람 전환에서는 anchor의 귀소율을 두 배 넘게 높인다. step 20에서는 모든 조건의 성능이 함께 무너지는데, 저자들은 이 지점을 walk가 보상 level set을 벗어난 곳으로 본다.

**오른쪽 D–E: 보상을 유지해도 의도는 깨질 수 있다.** Boat race는 화살표 타일을 시계방향으로 진입할 때마다 +3의 proxy reward를 주지만, 에이전트가 보지 못하는 true return은 트랙을 시계방향으로 실제 진행한 정도를 잰다. 그래서 타일 하나를 드나들면 진행 없이 proxy를 반복해서 받을 수 있다. E는 proxy return과 true return을 따로 그린다. endpoint는 두 군집으로 갈리고, 가장 많이 달라진 3개 정책은 이 보상 해킹으로 순진행이 0이 된다.

설정의 중요한 조건이 있다. **10개 anchor는 true return으로 PPO 학습했고** 모두 트랙을 따라 돈다. proxy만으로 학습하면 에이전트가 exploit을 스스로 찾아 HNC가 찾을 것이 남지 않으므로, 먼저 정상 정책을 만든 뒤 proxy를 보존하며 탐색한 실험이다. 부록 I.7의 MuJoCo Ant-v3에서도 평균 return을 유지한 채 다리 타이밍과 실패 유형이 달라진다([\[2/2\] I](/reviews/hnc-mechanics/)). [§5·I.4–I.7, pp. 8, 51–53](https://arxiv.org/pdf/2609.38081v1#page=51)

<span id="6-hnc로-손실-지형의-기하를-측정하기" aria-hidden="true"></span>

## 6. 손실 지형의 기하 측정 (HNC measures loss landscape geometry)

마지막 본문 실험은 HNC로 해 집합의 국소 기하를 잰다. 선행 연구가 독립 학습 네트워크 사이의 손실 장벽이나 단순화한 이론 모델로 다룬 질문, 즉 모델 크기와 과제 복잡도가 해 집합을 어떻게 바꾸는지를 학습된 네트워크 하나에서 직접 측정한다. CIFAR-100의 클래스 수 $C\in\{7,10,20,50,100\}$와 CNN 폭 $w\in\{16,24,32,48,96\}$를 바꿔 3 seed씩, **75개 anchor**를 만든다. 클래스당 학습 이미지 300개, augmentation 없이 학습하고 각 모델의 학습 이미지 512개로 출력을 제약한다. 클래스 수는 여기서 과제 복잡도의 대리 변수다.

![원문 Figure 5: 폭과 클래스 수에 따른 effective null fraction 및 정규화 곡률](/assets/reviews/hnc/geometry.webp)

[Figure 5 크게 보기](/assets/reviews/hnc/geometry.webp)

**A와 B는 다른 측정이다.** A의 effective null fraction은 평평한 방향으로 가중치 norm의 1%만큼 움직였을 때 함수 일치 손실 증가가 $\epsilon=0.05$를 넘지 않는 방향들의 비율이다. 출력 drift 허용값에 묶은 기준이라 헤시안 전체 크기가 다른 조건끼리도 비교할 수 있다. 이 비율은 폭이 커지면 늘고 클래스가 많아지면 줄지만, 가장 작은 네트워크·가장 어려운 과제에서도 크게 남는다.

B는 그 방향들을 따라 움직이는 비용을 본다. 출력 민감도 자체를 제약하지 않도록 drift 기준 대신 공통 상대 임계값 $\mu=10^{-6}\lambda_{\max}$를 쓰고, 이동 방향의 정규화 곡률 $\mathbf d^\top\mathbf H\mathbf d/(nC)$를 잰다. $\mathbf d$는 단위 방향, $n$은 probe 수이며, 구현·부록 J.3에서는 $\mathbf H$ 대신 $\mathbf G$를 쓴다. 곡률은 클래스가 많을수록 커지고 폭이 넓을수록 낮아지는 경향이며, band는 3 seed의 표준오차다. 기준을 여러 값으로 바꾼 부록에서도 경향은 유지된다.

저자들은 평평한 방향은 넉넉히 남지만 어려운 과제에서는 HNC가 따라가는 방향이 가중치 변화에 더 민감해진다고 보고, 국소 최소점 주변 지형이 이렇게 뻣뻣해지는 현상을 과제가 어려울수록 손실 지형이 분리된 basin으로 쪼개진다는 선행 연구의 잠재적 메커니즘으로 제시한다. [§6·J.1–J.4, pp. 8–9, 53–55](https://arxiv.org/pdf/2609.38081v1#page=8)

## 7. 논의: 과제가 허용하는 해와 학습이 고른 해 (Discussion)

저자들은 §6의 결과를 Huang et al.(2025)과 함께 읽는다. 그 연구에서는 과제가 어렵고 모델이 클수록 독립 학습한 RNN 사이의 동역학 변이가 줄었는데, HNC로 보면 같은 과제에서 큰 모델의 영공간이 더 크다. 저자들은 이를 큰 모델이 국소 영역 안에 대안을 더 많이 허용하면서도 학습은 그중 더 좁은 부분집합으로 수렴한다는 뜻으로 읽고, PRH가 표현 수렴의 설명으로 든 큰 모델의 단순성 편향과도 맞는다고 본다. 간극은 **과제가 허용하는 해의 다양성**과 **일반적인 최적화가 실제로 고르는 해의 다양성** 사이에 있다.

예상되는 반론은 파라미터 대칭성과 재매개변수화도 함수를 보존하면서 표현을 재조직할 수 있다는 것이다(Theiss et al., 2026). 저자들은 HNC의 해가 재매개변수화를 넘어선다고 답한다. 동역학 체제가 질적으로 바뀌고, 분포 밖 일반화가 달라지며, 행동 전략이 질적으로 다르다. 또 그 연구의 대칭성은 뉴런 복제처럼 네트워크 크기를 바꿀 수 있는 이산 연산이지만, HNC는 학습된 네트워크의 가중치를 연속적으로 조금씩 바꾼다.

한계로는 함수 보존이 유한 probe에서 근사적으로만 강제되어 보지 않은 입력, 특히 분포 이동에서는 출력 일치가 보장되지 않고, 도달 가능한 다양성이 probe 범위와 허용 drift에 달려 있다는 점을 든다. HNC는 국소 탐색이라 basin이 여럿인지 알지 못하지만, 전역 재학습과 번갈아 쓰면 탐색 범위가 넓어진다(부록 E).

보존량은 네트워크의 어떤 미분 가능한 함수여도 된다. 과제 손실을 보존하면 과제와 양립하는 더 넓은 집합을 탐색할 수 있고(부록 D), 조향 목적은 두 번째 과제의 성능이나 신경 데이터와의 정렬이 될 수 있다(부록 H의 Brain-Score 조향). 함수를 바꾸는 것이 목적이면 반대로 가파른 방향이 가장 효과적이다. 저자들은 최적화기의 암묵적 편향 연구, 대형 언어 모델로의 확장, model merging·fine-tuning·편집, 행동이 다르거나 보상을 해킹하는 정책을 찾아 보상 설계를 감사하는 일을 앞으로의 응용으로 제시한다. [§7, pp. 9–10](https://arxiv.org/pdf/2609.38081v1#page=9)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 가장 강한 근거는 3BFF RNN이다. task loss가 상한 아래에 머무는 동안 고정점 수가 바뀌고, 무방향 endpoint는 readout-null 부분공간의 지속적 움직임으로 기억을 유지한다. 저자들이 §7에서 든 근거 중에서는 분포 밖 일반화의 차이가 재매개변수화 반론에 가장 직접적인 답이다. 모든 입력에서 함수를 보존하는 대칭 변환이라면 긴 무입력 구간이나 OOD pulse 같은 입력에서도 행동이 같아야 하는데, 3BFF 대안들은 이런 입력에서 anchor와 다르게 무너진다(Figure 2E, 부록 F). 같은 결과는 probe 밖 보존이 근사라는 한계도 보여 준다.

ViT에서는 대조군이 강점이다. 제약에 쓰지 않은 이미지에서도 CKA와 kNN 정렬이 함께 내려가고, 정확도 손실을 맞춘 무작위 노이즈는 CKA를 거의 바꾸지 못한다. 다만 kNN 조향 endpoint는 무작위 초기화 ViT보다는 정렬이 높게 남고(G.3) 기능 보존도 약해, 결과의 크기가 조향 지표와 보존 수준에 함께 묶여 있다. §6의 effective null fraction 경향은 출력 제약 수와 파라미터 수만으로 정해지는 선형 영공간 하한과 같은 방향이어서([\[2/2\] J](/reviews/hnc-mechanics/)), 과제 난도의 효과는 정규화 곡률과 표현 차원 증가(J.4)가 더 직접 뒷받침한다.

남은 질문은 세 가지다. RNN에서 보인 메커니즘 수준의 차이가 ViT 내부 회로에서도 확인되는가, probe 수와 허용 drift를 맞췄을 때 CKA와 kNN 조향의 보존–변위 trade-off는 어떻게 달라지는가, 그리고 클래스 수를 고정하고 probe 출력 제약 수만 바꿨을 때 §6의 경향이 유지되는가.

## 출처와 읽은 범위

- 원문: *Traversing the solution space of neural networks with Hessian Null Space Continuation* · Ann Huang et al. · 최초 공개 2026-09-29 · [검토 버전](https://arxiv.org/abs/2609.38081v1) · 검토일 2026-10-23.
- 읽은 범위: v1 본문 §1–7. 설정·지표·반대 결과에 필요한 부록을 연결하며 A–J의 상세는 [2/2] 심층 편에 둔다.
- 그림 출처: 대표 Figure 1과 해당 편의 본문 그림은 저자 v1 자산(CC BY 4.0). PDF 래스터화·WebP 인코딩으로 해상도를 바꾸고 내용·배치·글자·화살표를 유지한다. [원문](https://arxiv.org/abs/2609.38081v1).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
