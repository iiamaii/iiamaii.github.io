---
title: "[2/2] HNC 심화: 함수 보존은 왜 가능하고 어디서 깨지는가?"
description: "부록 A–J를 따라 가우스–뉴턴 유도, soft projection, 계산 비용, 절제 실험과 probe 밖 검증을 해설한다."
date: "2026-10-17"
publishedAt: "2026-10-17T22:11:59+09:00"
updatedAt: "2026-10-23T15:26:22+00:00"
topics: ["representation learning", "loss landscape", "mechanistic interpretability", "computational neuroscience", "reinforcement learning"]
visibility: "public"
lang: "ko"
translationKey: "hnc-mechanics"
paperTitle: "Traversing the solution space of neural networks with Hessian Null Space Continuation"
paperPublishedDate: "2026-09-29"
authors: ["Ann Huang", "Mitchell Ostrow", "Zhouyang Lu", "William T. Redman", "Leo Kozachkov", "Kanaka Rajan"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2609.38081v1"
thumbnail: "/assets/reviews/hnc/overview.webp"
thumbnailAlt: "HNC 원문 Figure 1. 곡률이 작은 방향의 탐색과 출력 복원을 반복하는 원리 및 조향 구조."
---

헤시안 영공간 연속 탐색(Hessian Null Space Continuation, HNC)의 계산은 함수 일치 손실의 헤시안이 학습된 기준 모델(원문의 anchor)에서 가우스–뉴턴(Gauss–Newton) 연산자 $\mathbf G=\mathbf J^\top\mathbf J$와 같아진다는 부록 B의 유도에 기댄다($\mathbf J$는 probe 출력의 야코비안(Jacobian)). 이 연산자를 행렬 없이 곱으로만 쓰면, 파라미터 2,200만 개의 비전 트랜스포머(vision Transformer, ViT) ViT-S에서 dense 헤시안에 필요한 약 1.9 PB 대신 약 5.6 GB의 $k=64$ 블록으로 평평한 방향을 구하고 continuation step 하나를 H200에서 약 1분에 마친다. [원문 Appendix A–B, Table 1](https://arxiv.org/pdf/2609.38081v1#page=19) **리뷰 해석.** 유도가 보장하는 것은 anchor 한 점에서의 $\mathbf H=\mathbf G$와 $\mathbf G$의 양의 준정부호성뿐이고 경로 전체의 보존은 작은 보폭·복원·손실 상한이라는 경험적 장치에 기대는데, 절제 실험(C.1)에서 큰 보폭은 복원을 12회 해도 endpoint task loss가 anchor의 3.6배로 남는다.

[\[1/2\] 방법·실험 전체 해설](/reviews/hnc-solutions/)을 먼저 읽기를 권하며, 이 글은 부록 A→J 순서를 따른다.

## A. 알고리즘과 계산 비용 (Hessian Null-Space Continuation Pseudocode)

### Algorithm 1–3: 실제로 무엇을 수용하는가?

입력은 anchor $\boldsymbol\theta_0$, probe $\mathbf X$, 보폭 $\eta$, 복원 학습률 $\alpha$, 복원 횟수 $m$, 손실 상한 $\tau$와 선택적 조향 목표 $\varphi$다. Algorithm 1은 다음 순서다.

1. 현재 위치에서 방향을 계산하고 단위 길이로 정규화한다.
2. 그 방향으로 작은 이동을 제안한다.
3. 함수 일치 손실을 줄이는 복원 gradient step을 $m$번 수행한다.
4. 그래도 손실이 $\tau$를 넘으면 제안을 버리고 이전 가중치를 유지한다. 보폭을 줄이고 방향을 다시 계산한다.
5. 유한 probe에서 손실 상한 검사를 통과한 제안만 다음 checkpoint로 기록한다.

Algorithm 2는 행렬 전체 없이 $\mathbf G\mathbf v=\mathbf J^\top(\mathbf J\mathbf v)$를 계산해 하위 $k$ 고유쌍을 찾는다. Algorithm 3의 무방향 탐색은 그 공간 안의 방향을, 조향 탐색은 soft projection 방향을 쓴다. 3-bit flip-flop(3BFF) 기억 과제의 순환 신경망(recurrent neural network, RNN) 실험에서 무방향 탐색은 영공간에서 균일하게 뽑은 heading 하나를 고정해 두고, 재선형화 때마다 새 기저에 다시 투영한다. [A, p.18·F.3](https://arxiv.org/pdf/2609.38081v1#page=18)

### A.1 계산 비용

| 원문 Table 1 | Dense eigendecomposition | Matrix-free 경로 |
| --- | --- | --- |
| 시간 차수 | $O(P^3)$ | $O(kP)$, block 크기·반복 수 고정 조건 |
| 메모리 차수 | $O(P^2)$ | $O(kP)$ |
| 3BFF RNN, $P=4611$ | 85 MB, 10초 | 1.2 MB, 39초, $k=64$ |
| ViT-S, $P\approx22$M | 헤시안 저장 약 1.9 PB, 조립에만 곱 $2\times10^7$회(GPU 수개월), 분해는 불가능 | $k=64$ block 약 5.6 GB, product 약 1초, continuation step 약 1분 |

RNN 시간은 8-thread CPU node, ViT는 probe 512개와 NVIDIA H200 조건이다. 곱 한 번은 gradient 계산 한 번 정도의 비용이므로, $O(kP)$에는 probe 수·네트워크 연산·반복 수가 상수로 들어 있다. 조향에서는 conjugate gradient(CG) 반복과 복원 step도 더해진다. 4,611개 파라미터의 3BFF RNN에서는 오버헤드가 작은 dense가 오히려 빠르며, 저자들은 이를 반복 solver 검증에 쓴다. [A.1·Table 1, pp.18–19](https://arxiv.org/pdf/2609.38081v1#page=19)

### A.2 LOBPCG: 가장 평평한 방향을 찾는 문제

원문 식 (4)의 Rayleigh quotient는 다음과 같다.

$$
\rho(\mathbf v)=
\frac{\mathbf v^\top\mathbf H\mathbf v}
{\mathbf v^\top\mathbf v}.
$$

대칭 행렬의 정규직교 고유기저에서 $\mathbf v=\sum_i a_i\mathbf v_i$라 쓰면 분자는 $\sum_i\lambda_i a_i^2$, 분모는 $\sum_i a_i^2$다. 즉 고유값의 가중평균이므로 최솟값은 가장 작은 고유값이다. 이 계산은 **리뷰 보충**이며 원문의 최소화 설명을 풀어 쓴 것이다.

LOBPCG(locally optimal block preconditioned conjugate gradient)는 직교한 여러 방향을 block으로 함께 갱신한다. 현재 방향, residual, 이전 방향이 이루는 작은 탐색 공간에서 개선하며(이전 방향 항이 CG의 momentum 역할을 한다), 큰 행렬 대신 곱만 쓴다.

### A.3 Soft projection과 conjugate gradient

구현의 조향 방향은 다음 선형 시스템을 푼 결과다.

$$
(\mathbf I+\mathbf G/\mu)\mathbf d
=\nabla\varphi,\qquad \mu>0.
$$

왜 풀 수 있을까? $\mathbf G$가 양의 준정부호(positive semidefinite, PSD)이므로, 0이 아닌 $\mathbf v$에 대해

$$
\mathbf v^\top(\mathbf I+\mathbf G/\mu)\mathbf v
=\|\mathbf v\|^2+\|\mathbf J\mathbf v\|^2/\mu>0.
$$

따라서 시스템 행렬은 양의 정부호(positive definite)다. CG는 역행렬을 저장하지 않고 이 시스템을 반복적으로 근사한다. 원문은 고정 iteration budget과 relative residual $10^{-8}$ 조기 종료를 쓰고, 일반적으로 직전 방향을 초기값으로 재사용한다. 3BFF 조향은 200회 budget에 매번 0부터 푸는 예외다. ViT는 preconditioned CG 40회다. 조향 탐색은 매 step 현재 가중치에서 연산자를 다시 만들므로 별도 재선형화가 필요 없다.

**리뷰 보충 해석:** $\mathbf g=\nabla\varphi$라 두면 방향은 다음 이차 목적의 유일한 최소해다.

$$
\begin{aligned}
Q(\mathbf d)&=\frac12\|\mathbf d-\mathbf g\|^2
+\frac1{2\mu}\mathbf d^\top\mathbf G\mathbf d,\\
\underset{\mathbf d}{\arg\min}\,Q(\mathbf d)
&=(\mathbf I+\mathbf G/\mu)^{-1}\mathbf g.
\end{aligned}
$$

여기서 $Q$는 리뷰가 붙인 목적함수 이름이다. 미분해 0으로 두면 바로 앞 선형식이 나온다. 원하는 gradient에 가깝게 가되, $\mathbf d^\top\mathbf G\mathbf d=\|\mathbf J\mathbf d\|^2$인 출력의 1차 변화 크기에 비용을 매기는 셈이다. [A.2–A.3, p.19](https://arxiv.org/pdf/2609.38081v1#page=19)

<span id="b-최소점에서-gaussnewton과-hessian이-같아지는-이유" aria-hidden="true"></span>

## B. 손실 최소점의 헤시안과 가우스–뉴턴 근사 (Gauss–Newton approximation of the Hessian at loss minima)

원문에는 HNC 전체의 수렴이나 경로 전체의 함수 동등성을 다루는 번호 붙은 정리가 없고, 중심 수학은 이 절의 미분 유도다. 원문은 probe 출력을 한 벡터로 묶은 잔차(residual)를 정의한다.

$$
\begin{aligned}
\mathbf r(\boldsymbol\theta)
&=f_{\boldsymbol\theta}(\mathbf X)
-f_{\boldsymbol\theta_0}(\mathbf X),\\
\mathcal L(\boldsymbol\theta)&=\frac12\|\mathbf r(\boldsymbol\theta)\|^2,\\
\mathbf J&=\frac{\partial\mathbf r}{\partial\boldsymbol\theta}.
\end{aligned}
$$

여기서는 B의 합 표기를 유지한다. 본문 식 (1)처럼 표본 평균을 쓰면 gradient와 헤시안 양쪽에 같은 $1/N$ 계수가 붙을 뿐이라, 상대 임계값 $\mu_{\mathrm{rel}}\lambda_1$로 고른 영공간은 바뀌지 않는다. $\mathbf r\in\mathbb R^q$, $\boldsymbol\theta\in\mathbb R^P$이면 $\mathbf J\in\mathbb R^{q\times P}$이고 $\mathbf G\in\mathbb R^{P\times P}$다. $q$는 설명용으로 추가한 출력 성분 수다.

**원문의 미분 순서를 풀어 쓰자.** 먼저 각 가중치 성분 $a$에 대해

$$
\frac{\partial\mathcal L}{\partial\theta_a}
=\sum_k r_k\frac{\partial r_k}{\partial\theta_a}.
$$

다시 $b$로 미분할 때 product rule의 두 항을 모두 남긴다.

$$
\begin{aligned}
\frac{\partial^2\mathcal L}{\partial\theta_b\partial\theta_a}
&=\sum_k
\frac{\partial r_k}{\partial\theta_b}
\frac{\partial r_k}{\partial\theta_a}\\
&\quad+\sum_k r_k
\frac{\partial^2r_k}{\partial\theta_b\partial\theta_a}.
\end{aligned}
$$

행렬로 모으면 원문 식 (5)다.

$$
\nabla^2\mathcal L(\boldsymbol\theta)
=\underbrace{\mathbf J^\top\mathbf J}_{\mathbf G}
+\sum_k r_k(\boldsymbol\theta)\nabla^2r_k(\boldsymbol\theta).
$$

anchor에서는 $\mathbf r(\boldsymbol\theta_0)=\mathbf0$이므로 두 번째 항이 사라져 식 (6)을 얻는다.

$$
\nabla^2\mathcal L(\boldsymbol\theta_0)
=\mathbf J^\top\mathbf J=\mathbf G.
$$

anchor 밖에서는 잔차 곡률 항이 남으므로 탐색 중의 $\mathbf G$는 헤시안의 근사다. 그래도 $\mathbf G$는 언제나 PSD라서 고유값이 모두 0 이상이고, 평평한 부분공간이 spectrum의 바닥으로 분명히 정해져 따로 가려낼 음의 곡률 방향이 없다. 곱 $\mathbf G\mathbf v$는 야코비안–벡터 곱과 벡터–야코비안 곱 한 번씩으로 계산된다. [B, 식 (5)–(6), pp.19–20](https://arxiv.org/pdf/2609.38081v1#page=19)

### 영공간은 왜 유한 이동의 보증이 아닌가?

**리뷰 보충:**

$$
\mathbf v^\top\mathbf G\mathbf v=\|\mathbf J\mathbf v\|^2,
\qquad\ker\mathbf G=\ker\mathbf J.
$$

따라서 영방향은 출력의 *1차* 변화가 0인 방향이다. 설명용으로 $f(a,b)=(a+b,b^2)$, anchor $(0,0)$, 단위 방향 $\mathbf v=(1,-1)/\sqrt2$를 쓰면

$$
\begin{aligned}
\mathbf J(0,0)\mathbf v&=\mathbf0,\\
f(t\mathbf v)-f(0,0)&=(0,t^2/2),\\
\mathcal L(t\mathbf v)&=t^4/8.
\end{aligned}
$$

헤시안의 정확한 영방향이어도 유한 $t$에서 손실은 0이 아니고, 이 예제의 정확한 영손실 해는 원점뿐이다. 본문 §2.2가 고차 항을 이유로 복원·작은 보폭·수용 검사를 두는 상황이 이것이다.

<span id="c-구성요소-제거-실험-ablation-study" aria-hidden="true"></span>

## C. HNC 알고리즘 절제 실험 (Ablation study on HNC algorithm)

원문 C는 3BFF에서 복원(C.1), soft projection(C.2), 재선형화(C.3)가 각각 필요한지 묻고, 유사도 penalty를 건 학습과 비교한다(C.4). 따로 적지 않으면 중심 커널 정렬(centered kernel alignment, CKA) 유사도를 낮추는 조향을 쓴다.

| 절 | 바꾼 것 | 결과와 해석 |
| --- | --- | --- |
| C.1 | 복원 횟수 0·3·6·9·12, 세 보폭 | 큰 보폭 $\eta=0.3$에서 12회 복원은 endpoint 손실을 anchor의 6.6배→3.6배로 낮추지만 제거하지 못한다. 작은 이동 자체가 중요하다. |
| C.2 | soft projection damping | $10^{-9}$는 거의 움직이지 않고, 완화하면 다양성과 drift가 함께 커진다. 투영을 빼면 복원이 있어도 수십 step 안에 기능이 무너진다. |
| C.3 | 재선형화(relinearization) 없음/50/20/5 step 간격 | 자주 갱신할수록 손실을 줄이며 큰 보폭에서 특히 중요하다. |
| C.4 | HNC 대 similarity penalty 학습 | 같은 600 step, 각 설정 5 seed에서 HNC가 손실 수준 대비 더 큰 CKA 거리를 보고한다. penalty는 초반 높은 손실 구간을 통과한 뒤 돌아온다. |

저자들은 복원을 drift를 없애는 투영이 아니라 제한하는 안전장치로 보고, 그래서 HNC walk에 작은 보폭·적응적 보폭·손실 상한을 함께 쓴다고 설명한다. [C·Figures 6–9, pp.20–22](https://arxiv.org/pdf/2609.38081v1#page=20)

<span id="d-함수-보존과-과제-손실-보존은-다르다" aria-hidden="true"></span>

## D. HNC의 함수 보존과 과제 손실 보존 (Preserving function versus task loss in HNC)

첫째, 학습은 과제 gradient $\mathbf g$가 작아질 때 멈출 뿐 0이 될 때 멈추지 않는다. 평평한 방향 $\mathbf v$로 $\eta$만큼 움직이면 2차 항은 사라져도 1차 항 $\eta\mathbf g^\top\mathbf v$가 남고, HNC가 쓰는 작은 보폭에서는 보폭에 선형인 이 항이 2차 항보다 크다. 함수 일치 손실은 anchor에서 gradient가 정확히 0이라 평평한 방향이 1차·2차 모두에서 평평하다.

둘째, 두 손실은 헤시안 고유값의 부호가 다르다. 함수 일치 헤시안은 anchor에서 PSD라 대수적으로 가장 작은 고유값이 곧 0에 가장 가까운 평평한 방향이다. 과제 손실 $\mathcal L_{\mathrm{task}}$를 출력 $f$에 대한 손실 $\ell$로 쓰면, 그 헤시안에는 $\partial\ell/\partial f_k$로 가중한 출력 헤시안 항이 더해진다(원문 식 (9)).

$$
\nabla^2\mathcal L_{\mathrm{task}}
=\mathbf J^\top\bigl(\nabla_f^2\ell\bigr)\mathbf J
+\sum_k\frac{\partial\ell}{\partial f_k}\nabla^2 f_k.
$$

$\nabla^2 f_k$에는 부호 제약이 없어 둘째 항이 일부 방향에서 첫째 항을 넘으면 헤시안이 부정부호(indefinite)가 된다. 그러면 가장 작은 고유값을 고르는 절차가 0에 가까운 평평한 방향 대신 음의 곡률 방향을 고른다.

저자들은 3BFF anchor 4개에서 dense 헤시안을 배정밀도로 분해해 두 문제를 확인한다. 과제 손실 헤시안에는 4,611개 고유값 중 수백 개가 음수이고, 함수 일치 헤시안에는 하나도 없다. 과제 손실이 가장 빨리 오르는 평평한 방향(과제 손실 walk의 최악 조건)으로 손실 상한 없이 300 step을 걸으면, 과제 손실 walk는 1차 drift가 2차 항보다 2–3자릿수 커서 과제 손실이 약 10배가 되고, 함수 일치 walk는 과제 손실을 anchor 값의 10% 안에 둔다. 다만 3BFF는 과제 목표가 출력 궤적 전체라 실제로는 두 손실이 가깝다고 저자들은 덧붙인다.

셋째, 출력 벡터 전체와 스칼라 손실은 제약의 강도가 다르다. 다음은 ImageNet probe 512개, 클래스 1,000개에서 **표현 거리 수준을 맞춰** 비교한 원문 Table 2다. 값은 anchor와 top-1 예측이 같은 probe의 비율(probe seed 평균)이다.

| 보존량 | 성분 제약 수 | $\Delta\mathrm{CKA}=0.2$ | 0.4 | 0.6 |
| --- | --- | --- | --- | --- |
| 전체 logits | 512,000 | 0.996 | 0.994 | 0.993 |
| 이미지별 cross-entropy | 512 | 0.916 | 0.894 | 0.875 |
| 평균 cross-entropy | 1 | 0.831 | 0.811 | 0.793 |

같은 cross-entropy를 내는 logit 벡터가 여럿이므로 손실을 고정해도 예측은 바뀔 수 있다. 이미지별 손실을 고정하면 각 이미지 안의 출력 분포가 달라질 수 있고, 평균만 고정하면 일부 probe 이미지의 손해를 다른 이미지의 이득으로 상쇄할 수도 있다.

![원문 Figure 11: 전체 logits, 이미지별 손실, 평균 손실 보존의 비교](/assets/reviews/hnc/constraint-strength.webp)

[Figure 11 크게 보기](/assets/reviews/hnc/constraint-strength.webp)

**Figure 11의 B와 D를 함께 읽는다.** B의 실선은 probe, 점선은 held-out 5,000개이며, 출력 벡터를 맞춘 보라색이 더 높은 일치를 유지한다. A처럼 방식마다 표현이 바뀌는 속도가 달라 같은 step에서의 비교는 의미가 다르므로, D는 가로축을 $\Delta\mathrm{CKA}$로 바꿔 같은 표현 변화에서 비교한다. 선은 probe seed 중앙값, band는 최소–최대 범위다. [D·Figure 11·Table 2, pp.22–25](https://arxiv.org/pdf/2609.38081v1#page=24)

저자들은 함수 일치가 라벨도 과제도 필요 없어 라벨 없는 probe, 이동된 입력, return이 평평하거나 미분 불가능한 정책, 다음 토큰 분포를 맞추는 언어 모델에도 쓸 수 있고, 모든 과제 손실이 출력의 함수이므로 함수 보존 집합은 과제 양립 집합의 부분집합이라고 정리한다. 더 넓은 과제 양립 집합을 원하면 과제 손실을 anchor 값과의 제곱 잔차 $\tfrac12(\mathcal L_{\mathrm{task}}(\boldsymbol\theta)-\mathcal L_{\mathrm{task}}(\boldsymbol\theta_0))^2$로 고정해 출발 gradient를 0으로 만들거나 이동을 과제 gradient에 직교로 투영하고, probe를 늘리거나 입력별 손실·결정을 목표에 더해 제약을 조일 수 있다. 제약을 조일수록 출력 전체에 가까워져 극한에서 함수 일치 손실로 돌아간다.

<span id="e-전역-탐색과-결합하기" aria-hidden="true"></span>

## E. 다른 해를 찾는 전역 탐색과 HNC의 결합 (Combining HNC with global search for alternative solutions)

HNC는 anchor 주변의 국소 영역만 탐색하고, 새로 학습한 모델은 가중치 공간 어디에나 닿을 수 있지만 도착점을 통제할 수 없다. 저자들은 동역학 유사도 분석(dynamical similarity analysis, DSA) 거리로 조향한 local walk와, 지금까지 찾은 모든 네트워크와 동역학이 멀어지도록 반발 regularizer를 건 새 학습을 번갈아 수행한다. 8개 네트워크 체인은 모두 bit accuracy $0.999$를 넘으면서 기억을 고정점, 고정점 없는 지속 drift, 기억 상태보다 많은 attractor로 구현한다. 재학습 한 번은 무관한 seed만큼 멀리 가고 walk는 anchor norm의 일부만큼 더 움직이며, 새 네트워크마다 앞선 모든 네트워크와의 DSA 거리가 독립 seed 두 개 사이보다 크다.

저자들은 이를 전역·국소 탐색을 결합할 수 있다는 원리 증명(proof of principle)으로 제시한다. 손실 상한을 느슨하게 해 얕은 ridge를 넘은 뒤 task loss로 사후 선별하기, endpoint에 작은 random kick을 준 뒤 재시작하기, 다른 네트워크 쪽으로 조향해 두 해를 함수 보존 중간점으로 잇기는 향후 과제로 남긴다. [E·Figures 12–14, pp.26–28](https://arxiv.org/pdf/2609.38081v1#page=26)

<span id="f-rnn-추가-분석-차원압축재학습ood" aria-hidden="true"></span>

## F. 3BFF RNN 실험 상세 (Details for the 3BFF RNN experiments)

### F.1–F.3 설정과 평평한 공간의 크기

64유닛 RNN은 4,611개 파라미터라 전체 헤시안 spectrum을 직접 계산할 수 있다. 5개 anchor에서 곡률은 소수 방향에 몰려 participation ratio(유효 차원)가 $23.2\pm1.6$이고, 나머지 고유값은 가파른 방향과 평평한 방향 사이의 간극 없이 9자릿수에 걸쳐 매끄럽게 줄어든다. 그래서 평평한 부분공간의 크기는 임계값을 어디에 두느냐에 달려 있다.

**Participation ratio의 의미.** $D_{\mathrm{PR}}$을 유효 차원의 리뷰 약칭이라 하면, 0이 아닌 PSD spectrum에 대해

$$
D_{\mathrm{PR}}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
$$

같은 양의 고유값이 $r$개면 $r$이고, 하나에 집중되면 1에 가까워진다. 작은 양의 고유값은 rank에는 포함되지만 이 지표에는 약하게 기여한다. [Appendix F.2](https://arxiv.org/pdf/2609.38081v1#page=29)

원문 Table 3의 상대 임계값별 median flat dimension은 다음과 같다.

| $\mu_{\mathrm{rel}}$ | median dimension | 전체 파라미터 중 비율 |
| --- | --- | --- |
| $10^{-7}$ | 1701 | 36.9% |
| $10^{-6}$ | 2896 | 62.8% |
| $10^{-5}$ | 3684 | 79.9% |
| $10^{-4}$ | 4145 | 89.9% |
| $10^{-3}$ | 4418 | 95.8% |
| $10^{-2}$ | 4565 | 99.0% |

허용할 출력 변화가 커질수록 더 많은 방향이 평평한 방향으로 분류된다. 탐색 설정에서 CKA 조향은 600 step, DSA 조향은 2,000 step이며 각 step 복원 12회를 쓴다. 조향 목적은 32 trial probe에서 계산하고, 수용 검사는 Algorithm 1의 함수 일치 손실 대신 고정 128 trial의 task MSE로 한다(F.3). [F.1–F.3, pp.29–30](https://arxiv.org/pdf/2609.38081v1#page=29)

### F.4–F.8 무엇이 보존되고 무엇이 바뀌는가?

gallery와 10개 anchor의 다차원 척도법(multidimensional scaling, MDS)은 특이한 도착점 하나가 아니라 여러 방향의 대안이 있음을 보여 준다. 가중치 거리에서는 네트워크가 anchor별로 모이지만, 표현 거리에서는 모든 anchor가 한데 모이고 조향 walk가 여러 방향으로 퍼진다. 희소화(pruning) 조향은 유닛별 recurrent·output 출력 가중치 묶음을 0으로 모는 목표로 64유닛 RNN 하나를 11유닛까지 줄이며 bit accuracy 100%를 유지한다. 은닉 크기의 약 17%, 파라미터의 약 5%이고(리뷰 계산: 4,611→212개), MSE는 조금 늘어난다.

8,000-step SGD를 재개하면 task loss가 더 낮은 해에 자리 잡으면서도 조향한 CKA·DSA·희소성의 차이가 그대로 남는다. 재학습은 가중치를 조금만 움직이고 endpoint를 anchor 쪽으로 되돌리지 않는다. 분포 밖(out-of-distribution, OOD) pulse에서는 약한 pulse가 DSA 조향·무방향 endpoint를 anchor보다 먼저 떨어뜨리고, 강한 pulse는 순환 spectral radius가 커진 무방향 endpoint만, 드문 pulse는 기억 유지가 약한 DSA 조향 endpoint를 먼저 떨어뜨린다. [F.4–F.8, pp.30–35](https://arxiv.org/pdf/2609.38081v1#page=31)

<span id="g-vit-검증-지표-최적화와-함수-보존을-따로-평가하기" aria-hidden="true"></span>

## G. ViT 추가 실험 (Additional experiments on ViT)

### G.1–G.4 조향 목적과 설정

**CKA의 계산 대상.** 아래 $X,Y$는 같은 표본 순서의 두 특징 행렬을 열별 중심화한 리뷰 설명 기호다. probe 입력 $\mathbf X$와 구분한다.

$$
\operatorname{CKA}_{\mathrm{linear}}(X,Y)
=\frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F\,\|Y^\top Y\|_F}.
$$

이 선형 CKA는 0–1 범위이며 높을수록 표본 간 기하가 유사하다. 직교 좌표변환과 전체 스케일 변화에는 불변이지만 임의의 가역 변환에는 불변이 아니다. [CKA 원전 §3–5, Eq. (14)](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf#page=5)

$c_{ij}(\boldsymbol\theta)$를 anchor $i$층과 현재 모델 $j$층의 linear CKA라고 하면 원문 식 (12)는

$$
\begin{aligned}
\varphi_{\mathrm{CKA}}(\boldsymbol\theta)
&=1-\frac1\beta\log\sum_{i,j=1}^{L}
\exp\!\bigl(\beta c_{ij}(\boldsymbol\theta)\bigr),\\
L&=13,\qquad\beta=30.
\end{aligned}
$$

이를 키우면 현재 가장 비슷한 층 쌍들에 큰 가중치를 두고 유사도를 낮춘다. 보고는 hard max로 한다. G.2의 의미·색·공간 배치 점수는 각각 DINOv2 ViT-S/14 cosine, HSV $8^3$ histogram cosine, $32\times32$ grayscale correlation이다.

G.3은 플라톤적 표현 가설(Platonic Representation Hypothesis, PRH)이 쓰는 다른 유사도인 상호 k-최근접 이웃(mutual k-nearest-neighbor, kNN) 정렬로 실험을 반복한다. 이미지마다 두 표현이 공유하는 $k=10$ 최근접 이웃의 비율을 층 쌍 최대로 재며, 미분에는 부드러운 이웃 가중치를 쓴다. endpoint의 정렬은 비교한 모든 독립 학습 모델보다 낮지만 무작위 초기화 모델보다는 높다. 원문 G.3 본문은 함수를 Places-365 256개 이미지에서 고정한다고 쓰고($N=256$), G.4 Table 8은 같은 kNN walk의 probe를 512개로 적는다.

### G.5–G.6 held-out 보존과 단순 손상 비교

CKA 조향 endpoint의 핵심 결과(오류 일관성 약 0.8, seed만 바꾼 비교 모델 이상의 일치, 정확도 손실을 맞춘 노이즈가 CKA를 0.98 위에 남김)는 [\[1/2\] §4](/reviews/hnc-solutions/)에 정리했다. 비교의 반복 단위는 서로 다르다. CKA 조향은 5개 probe draw, kNN 조향은 3개 draw이고, diff-seed 비교 모델(anchor의 ImageNet-21k checkpoint에서 마지막 ImageNet-1k fine-tuning만 seed를 바꿔 반복)은 2 seed다. single anchor와 독립 ViT-S/in1k의 band는 이미지 bootstrap이다.

kNN 조향 endpoint는 top-$k$ 정확도 곡선의 모양은 유지하지만 정확도 손실이 더 크다. 정확한 예측 일치는 clean ImageNet 약 0.85에서 Sketch 약 0.6으로 내려가고, 오류 일관성은 약 0.75다. ViT-S/in1k보다는 anchor와 더 일치하지만 네 데이터셋 모두에서 diff-seed보다 낮다. 정확도 손실을 맞춘 노이즈는 walk보다 가중치를 약 5배 더 움직이고도 어떤 독립 학습 모델보다 anchor와 비슷하게 남는다. 저자들은 kNN의 약한 보존을 지표 자체의 한계가 아니라 원문에서 시험한 설정의 보존–변위 trade-off로 보며, 근거로 probe가 클수록 probe 밖 보존이 좋아진다는 G.9 결과를 든다. **리뷰 해석.** 이 설명은 probe 수에 기대는데, kNN 설정의 probe 수가 G.3 본문과 Table 8에서 다르게 적혀 있어 두 조향이 같은 probe 수에서 비교됐는지는 원문으로 확인되지 않는다. [G.5–G.6, pp.39–42](https://arxiv.org/pdf/2609.38081v1#page=39)

오류 일관성(error consistency)은 각 이미지에서 두 모델이 **둘 다 맞거나 둘 다 틀리는지**의 일치를 우연 수준으로 보정한 지표로, 둘 다 틀렸을 때 같은 오답 클래스를 골라야 하는 prediction match와 다르다. 그래서 분포 이동이 큰 Rendition·Sketch에서 prediction match가 내려가도 오류 일관성은 유지될 수 있다. [Appendix G.5](https://arxiv.org/pdf/2609.38081v1#page=39)

### G.7–G.10 층·방향·probe 수를 바꾸면?

초기 block 하나도 조향할 수 있지만, pre-logit보다 같은 표현 변화에 드는 기능 비용이 크다. 저자들은 초기 층의 모델 간 유사성도 함수가 강제하지 않는다고 본다. ViT-S를 frozen ViT-B와 닮게 하는 반대 실험에서는 함수를 고정한 Places-365 256개 이미지에서 depth 평균 CKA가 0.78→0.96으로 오르고, <strong>이 probe의 예측 일치는 96%</strong>다. held-out ImageNet 5,000개에서는 예측의 약 90%를 유지하고 top-1 정확도는 1%p 조금 넘게 내려간다. 저자들은 독립 학습한 두 모델이 공유하는 유사도가 과제가 정한 상한이 아니며, 수렴의 정도 자체가 함수의 요구를 넘어선 설명을 필요로 한다고 결론짓는다.

probe 수를 늘리면 probe 밖 보존이 좋아지면서 표현도 더 멀리 움직이고, 비용은 probe 수에 대략 선형으로 는다. 저자들의 설명은 큰 probe가 함수를 더 단단히 묶어 walk가 손실 상한 아래에 더 오래 머물고, 그만큼 더 많은 step을 조향에 쓴다는 것이다. 마지막 MDS에서는 CKA·kNN 조향이 서로 다른 방향으로 anchor를 떠나고, probe 집합마다 다른 대안으로 퍼지며, 얼마나 멀리 갔는지가 쓰는 거리에 따라 달라진다. [G.7–G.10, pp.43–46](https://arxiv.org/pdf/2609.38081v1#page=43)

<span id="h-brain-score-조향-무엇을-실제로-높였는가" aria-hidden="true"></span>

## H. 함수를 유지하며 Brain-Score 조향하기 (Steering Brain-Score at fixed function)

Brain-Score는 모델의 내부 특징이 기록된 신경 반응을 얼마나 잘 예측하는지 재는 benchmark로, 보통 구조·학습 목적·데이터가 정하는 고정된 속성처럼 쓰인다. 저자들은 그 점수 중 얼마가 학습이 도달한 함수 동등 해의 선택에 달려 있는지 묻는다. MajajHong2015 benchmark는 이미지 3,200개와 macaque 하측두피질(inferior temporal cortex, IT) recording site 168개를 쓴다. 보고 점수는 10-fold로 25성분 부분 최소제곱(partial least squares, PLS) 회귀를 적합해 site별 상관을 구하고, site 중앙값의 fold 평균을 noise ceiling 0.82로 나눈다.

PLS는 반복 적합이라 미분하지 않으므로, 조향은 별도 surrogate를 쓴다. anchor에서 고정한 64차원 PCA 특징에 조향 이미지 절반으로 ridge regression을 적합하고, 나머지 절반에서 site별 상관의 평균을 높이거나 낮춘다. 저자들은 surrogate와 공식 점수가 다른 회귀이고 held-out 평가에서는 다른 이미지·site를 쓰므로, 공식 점수 변화가 최적화의 직접 목표가 아니라고 설명한다.

결과는 두 모델에서 보고된다(top-1 결정 최소 94% 보존, logit KL $10^{-5}$–$10^{-3}$). Brain-Score 신경 leaderboard 1위인 86M 파라미터 상대위치 ViT의 down/up endpoint IT 점수는 0.618–0.719(범위 약 0.10)인데, 이 범위는 조향에 쓴 이미지·site에서 잰 것이고 walk와 surrogate가 보지 않은 site·이미지에서는 0.039다. 비교군 최하위인 plain ViT-S/16은 held-out 이미지에서 0.546–0.627(0.081)로 움직인다. 같은 pipeline으로 잰 7개 architecture의 점수 폭은 본문 0.074, 그림 캡션 0.075이며, 저자들은 한 네트워크 안에서 연 범위가 architecture 간 차이 전체와 "같거나 넘는다(matches or exceeds)"고 결론짓는다. IT를 조향해도 V1·V2·V4 예측도는 거의 그대로다. [H, pp.46–48](https://arxiv.org/pdf/2609.38081v1#page=46)

**리뷰 해석.** "matches or exceeds"는 ViT-S의 held-out 이미지 범위(0.081 대 0.074)에서는 성립한다. 86M 모델에서는 조향에 쓴 이미지·site 기준 0.10에서만 성립하고, held-out 기준 0.039는 모델 간 폭의 절반 정도다. 따라서 신경 예측도가 어느 해에 도달했는지에도 달려 있다는 결론은 두 모델 모두에서 서지만, 그 효과가 architecture 차이만큼 크다는 근거는 ViT-S 한 모델이다.

<span id="i-강화학습-확장의-수학과-실패" aria-hidden="true"></span>

## I. 강화학습으로 HNC 확장하기 (Extending HNC to reinforcement learning)

### I.1 Advantage와 surrogate의 의미

Plume·Ant는 일반화 이점 추정(generalized advantage estimation, GAE)을 쓴다. 원문의 시간별 보상 $r_t$와 critic $V$로

$$
\begin{aligned}
\delta_t&=r_t+\gamma V(s_{t+1})-V(s_t),\\
A_t&=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},\\
\gamma&=0.99,\qquad\lambda=0.95.
\end{aligned}
$$

episode 끝에서는 다음 상태 value를 0으로 둔다. Boat race는 critic 없이 $\gamma=0.9$의 discounted proxy return을 advantage로 쓴다. 세 과제 모두 buffer에서 advantage를 표준화하므로, 기준 정책에서는 importance ratio가 모두 1이라 surrogate가 0이다.

### I.2 Gaussian KL에서 평균 행동 MSE까지

공분산이 같은 Gaussian policy에서는 원문의 일반 KL에서 trace와 log-determinant 항이 상쇄되어

$$
\begin{aligned}
D_{\mathrm{KL}}(\pi_{\boldsymbol\theta_0}\Vert\pi_{\boldsymbol\theta})
&=\frac12(\boldsymbol\mu_{\boldsymbol\theta}-\boldsymbol\mu_{\boldsymbol\theta_0})^\top
\boldsymbol\Sigma^{-1}
(\boldsymbol\mu_{\boldsymbol\theta}-\boldsymbol\mu_{\boldsymbol\theta_0})\\
&=\sum_{j=1}^d
\frac{(\mu_{\boldsymbol\theta,j}-\mu_{\boldsymbol\theta_0,j})^2}{2\sigma_j^2}.
\end{aligned}
$$

상태 인자는 이 블록에서만 생략했다. 공분산은 walk 내내 anchor 값으로 고정하므로 가중치 $1/(2\sigma_j^2)$는 상수이고, 저자들은 비가중 평균 행동 MSE를 최대화한다. 이것은 모든 $\sigma_j$가 같을 때 KL과 상수배로 같고, 그렇지 않으면 고정 대각 가중치만큼 다르다.

### I.3–I.7 세 환경의 조건과 한계

| 환경 | buffer/평가 | 관찰된 결과·한계 |
| --- | --- | --- |
| Plume | current-policy 8 episodes, 5 step마다 갱신; 정규 평가 60 초기 조건 | casting 감소와 보상 유지. OOD는 조건별 240개, 8×30 batch의 표준오차. step 20에서는 보존을 벗어나 성능 붕괴. |
| Boat race | 32 episodes, 3 step마다 갱신, 탐색 $\epsilon=0.25$; 평가 100 episodes | true-return으로 학습한 10 anchor에서 출발. 가장 다른 3 endpoint가 proxy를 받으면서 순진행 0의 exploit으로 이동. |
| Ant-v3 | 4,000 transitions, 5 step마다 갱신; 고정 30 초기 상태 | 19 step에서 action divergence 1.0. 같은 action-KL의 random 방향은 return 절반 이상 손실. HNC 대안은 mean return을 유지하지만 실패 유형은 변한다. |

Ant의 걸음 분석은 걷는 episode만 사용해 anchor 25/30, 대안 27/30이다. 대안은 멈춰 서는 실패가 줄지만 torso 높이가 범위를 벗어나 조기 종료하는 실패가 생긴다. 두 정책의 초기 상태별 return은 서로 상관이 없어, 저자들은 같은 평균 return 아래 서로 다른 역량 분포가 숨어 있다고 본다. Plume OOD 조건은 저마다 초기 조건 grid가 달라, 원문은 같은 조건 안에서 step끼리만 비교할 수 있다고 밝힌다. [I, pp.49–53](https://arxiv.org/pdf/2609.38081v1#page=49)

<span id="j-손실-지형-측정의-해석-범위" aria-hidden="true"></span>

## J. HNC로 손실 지형의 기하 탐사하기 (Probing loss-landscape geometry with HNC)

폭 5개×클래스 수 5개×3 seed의 CNN grid는 75개다. GroupNorm을 써서 각 이미지를 batch 구성과 무관하게 처리하고, 각 모델의 학습 입력 512개에서 함수를 고정한다. J.2는 drift 허용값을, J.3은 상대 spectral cutoff를 여러 값으로 바꾸어도 경향이 유지됨을 보인다. J.4에서 held-out 이미지의 pre-logit 특징 공분산 participation ratio는 클래스 수가 늘면 약 5배로 커지지만 폭에는 거의 둔감하며, 저자들은 이를 클래스 수가 과제 복잡도의 대리 변수라는 확인으로 쓴다. **특징의 유효 차원과 가중치의 flat fraction은 다른 측정**이다.

**리뷰 해석.** J.1은 학습 정확도가 100%에 이르면 학습을 멈추므로 "모든 anchor가 exact zero-loss solution"이라고 쓴다. cross-entropy에서는 정답 확률이 0.6이어도 top-1은 맞지만 손실은 양수이므로, 정확한 표현은 "학습 데이터를 모두 맞게 분류하는 해"다. §6의 측정은 anchor 자신의 출력을 목표로 하는 함수 일치 손실이 anchor에서 정의상 0이라는 사실에만 기대므로 결론은 바뀌지 않는다.

**리뷰 보충:** probe 출력 성분이 $q$개면 $\operatorname{rank}(\mathbf J)\le q$이고 $\dim\ker\mathbf J\ge P-q$다. §6에서는 $q=512C$이므로, 폭 16 모델(원문 Table 12의 $P=75{,}703$–$81{,}748$)의 선형 영공간 비율 하한은 $C=7$에서 약 95%, $C=100$에서 약 37%다. 폭이 커지면 $P$가, 클래스가 늘면 $q$가 커지므로 본문 Figure 5A([\[1/2\] §6](/reviews/hnc-solutions/))의 두 경향은 이 계수 하한과 같은 방향이다. [J, pp.53–55](https://arxiv.org/pdf/2609.38081v1#page=53)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 부록의 강점은 각 장치가 왜 필요한지를 실패로 보여 준다는 데 있다. 투영을 빼면 복원이 있어도 수십 step 안에 기능이 무너지고(C.2), 큰 보폭에서 재선형화가 드물면 손실이 폭주하며(C.3), 과제 손실을 보존량으로 쓰면 1차 drift와 음의 곡률 방향 때문에 탐색이 과제에서 벗어난다(D). 다만 절제(C)는 3BFF RNN에서, 제약 강도 비교(D)는 ViT-S anchor 하나에서만 이루어져 다른 구조에서 같은 설정이 통하는지는 보이지 않는다. C.4는 step 수와 보폭을 맞춘 비교라, 곡률 곱과 CG 반복이 드는 HNC와 곡률 정보가 필요 없는 penalty 학습의 계산량 차이는 비교에 들어가지 않는다.

남은 질문은 세 가지다. 수용된 checkpoint 사이의 경로에서도 probe 밖 함수가 유지되는가, 같은 probe 수에서 kNN 조향이 CKA 조향만큼 함수를 보존하는가, 그리고 RL에서 surrogate return 보존이 실제 return 붕괴로 바뀌는 지점(Plume의 step 20)을 walk 중에 미리 알 수 있는가.

## 출처와 읽은 범위

- 원문: *Traversing the solution space of neural networks with Hessian Null Space Continuation* · Ann Huang et al. · 최초 공개 2026-09-29 · [검토 버전](https://arxiv.org/abs/2609.38081v1) · 검토일 2026-10-23.
- 읽은 범위: v1 부록 A–J의 알고리즘·미분 유도·절제 실험·검증 조건. 반복 gallery·모든 hyperparameter 행은 축약한다.
- 참고 외부 원전: Kornblith et al.(2019)의 CKA 정의(§3–5).
- 그림 출처: 대표 Figure 1과 해당 편의 본문 그림은 저자 v1 자산(CC BY 4.0). PDF 래스터화·WebP 인코딩으로 해상도를 바꾸고 내용·배치·글자·화살표를 유지한다. [원문](https://arxiv.org/abs/2609.38081v1).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
