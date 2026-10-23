---
title: "[2/2] HNC 심화: 함수 보존은 왜 가능하고 어디서 깨지는가?"
description: "부록 A–J를 따라 Gauss–Newton 유도, soft projection, 계산 비용, ablation과 probe 밖 검증을 해설한다."
date: "2026-10-17"
publishedAt: "2026-10-17T22:11:59+09:00"
updatedAt: "2026-10-22T23:22:53+09:00"
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

HNC의 핵심은 “Hessian의 영공간이면 어디로 움직여도 기능이 같다”가 아니다. **anchor에서 출력 변화의 1차 민감도가 0인 방향을 찾고, 유한한 이동에서 생기는 오차를 복원·거절·재계산으로 관리한다.** 이 차이를 이해해야 함수 보존이라는 표현을 정확히 읽을 수 있다.

[\[1/2\] 방법·실험 전체 해설](/reviews/hnc-solutions/)을 먼저 읽기를 권한다. 이 글은 **부록 A→B→C→D→E→F→G→H→I→J** 순서를 따른다. 원문에는 HNC 전체의 수렴이나 전역 함수 동등성을 증명하는 번호 붙은 정리가 없다. 중심 수학은 부록 B의 미분 유도이며, 아래 중간 계산·예제는 필요한 곳에 **리뷰 보충**으로 구분한다.

## A. 알고리즘과 계산 비용 (Hessian Null-Space Continuation Pseudocode)

### Algorithm 1–3: 실제로 무엇을 수용하는가?

입력은 anchor $\boldsymbol\theta_0$, probe $\mathbf X$, 보폭 $\eta$, 복원 학습률 $\alpha$, 복원 횟수 $m$, 손실 상한 $\tau$와 선택적 조향 목표 $\varphi$다. Algorithm 1은 다음 순서다.

1. 현재 위치에서 방향을 계산하고 단위 길이로 정규화한다.
2. 그 방향으로 작은 이동을 제안한다.
3. 함수 일치 손실을 줄이는 복원 gradient step을 $m$번 수행한다.
4. 그래도 손실이 $\tau$를 넘으면 제안을 버리고 이전 가중치를 유지한다. 보폭을 줄이고 방향을 다시 계산한다.
5. 통과한 제안만 다음 checkpoint로 기록한다.

따라서 **제안한 모든 점이 보존된 해인 것은 아니다.** 수용된 checkpoint가 유한 probe의 설정된 손실 검사를 통과한다는 뜻이다. checkpoint 사이의 연속 경로 전체를 별도로 인증한 정리도 아니다.

Algorithm 2는 행렬 전체 없이 $\mathbf G\mathbf v=\mathbf J^\top(\mathbf J\mathbf v)$를 계산해 하위 $k$ 고유쌍을 찾는다. Algorithm 3의 무방향 탐색은 그 공간의 방향을, 조향 탐색은 soft projection 방향을 사용한다. RNN의 무방향 설정은 한 random heading을 유지하고 새 기저에 다시 투영하므로, 매번 독립·균일한 방향을 뽑는 random walk와도 구분해야 한다. [A, p.18·F.3](https://arxiv.org/pdf/2609.38081v1#page=18)

### A.1 비용: matrix-free는 공짜가 아니다

| 원문 Table 1 | Dense eigendecomposition | Matrix-free 경로 |
| --- | --- | --- |
| 시간 차수 | $O(P^3)$ | $O(kP)$, block 크기·반복 수 고정 조건 |
| 메모리 차수 | $O(P^2)$ | $O(kP)$ |
| 3BFF RNN, $P=4611$ | 85 MB, 10초 | 1.2 MB, 39초, $k=64$ |
| ViT-S, $P\approx22$M | Hessian 저장 약 1.9 PB, 분해는 실용적이지 않음 | $k=64$ block 약 5.6 GB, product 약 1초, continuation step 약 1분 |

RNN 시간은 8-thread CPU node, ViT는 probe 512개와 NVIDIA H200 조건이다. 다른 하드웨어 간 속도 우위를 비교한 표가 아니다. $O(kP)$에는 곡률 곱에 필요한 입력 수·네트워크 연산·반복 수가 숨어 있다. 조향에서는 CG 반복과 복원 step도 필요하다. $k$나 iteration을 늘리면 비용도 늘며, 표의 block 저장량을 전체 실행의 peak GPU 메모리로 일반화하지 않는다. 작은 RNN에서는 dense가 오히려 빠르다. [A.1·Table 1, pp.18–19](https://arxiv.org/pdf/2609.38081v1#page=19)

### A.2 LOBPCG: 가장 평평한 방향을 찾는 문제

원문 식 (4)의 Rayleigh quotient는 다음과 같다.

$$
\rho(\mathbf v)=
\frac{\mathbf v^\top\mathbf H\mathbf v}
{\mathbf v^\top\mathbf v}.
$$

대칭 행렬의 정규직교 고유기저에서 $\mathbf v=\sum_i a_i\mathbf v_i$라 쓰면 분자는 $\sum_i\lambda_i a_i^2$, 분모는 $\sum_i a_i^2$다. 즉 고유값의 가중평균이므로 최솟값은 가장 작은 고유값이다. 이 계산은 **리뷰 보충**이며 원문의 최소화 설명을 풀어 쓴 것이다.

LOBPCG는 직교한 여러 방향을 block으로 함께 갱신한다. 현재 방향, residual, 이전 방향의 작은 탐색 공간에서 개선하며, 큰 행렬 대신 곱만 쓴다. $k$개 벡터를 구했다고 전체 평평한 방향이 $k$개라는 결론은 나오지 않는다. LOBPCG 원 논문 전체와 수렴 증명은 이번에 독립 정독하지 않았으며, 여기서는 HNC A.2의 설명과 대칭행렬 항등식을 확인했다.

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

따라서 시스템 행렬은 양의 정부호(positive definite)다. CG는 역행렬을 저장하지 않고 이 시스템을 반복적으로 근사한다. 원문은 고정 iteration budget과 relative residual $10^{-8}$ 조기 종료를 쓰고, 일반적으로 직전 방향을 초기값으로 재사용한다. 3BFF 조향은 200회 budget에 매번 0부터 푸는 예외다. ViT는 preconditioned CG 40회다.

**리뷰 보충 해석:** $\mathbf g=\nabla\varphi$라 두면 방향은 다음 이차 목적의 유일한 최소해다.

$$
\begin{aligned}
Q(\mathbf d)&=\frac12\|\mathbf d-\mathbf g\|^2
+\frac1{2\mu}\mathbf d^\top\mathbf G\mathbf d,\\
\underset{\mathbf d}{\arg\min}\,Q(\mathbf d)
&=(\mathbf I+\mathbf G/\mu)^{-1}\mathbf g.
\end{aligned}
$$

여기서 $Q$는 리뷰가 추가한 목적함수 이름이다. 미분해 0으로 두면 바로 앞 선형식이 나온다. 원하는 gradient에 가깝게 가되 출력 변화의 1차 민감도에는 비용을 매기는 셈이다. 그러나 이후 방향 정규화·유한 이동·복원까지 포함한 전체 HNC가 조향 목적을 매번 증가시킨다는 증명은 아니다. [A.2–A.3, p.19](https://arxiv.org/pdf/2609.38081v1#page=19)

## B. 최소점에서 Gauss–Newton과 Hessian이 같아지는 이유

원문은 probe 출력을 한 벡터로 묶은 잔차(residual)를 정의한다.

$$
\begin{aligned}
\mathbf r(\boldsymbol\theta)
&=f_{\boldsymbol\theta}(\mathbf X)
-f_{\boldsymbol\theta_0}(\mathbf X),\\
\mathcal L(\boldsymbol\theta)&=\frac12\|\mathbf r(\boldsymbol\theta)\|^2,\\
\mathbf J&=\frac{\partial\mathbf r}{\partial\boldsymbol\theta}.
\end{aligned}
$$

여기서는 B의 합 표기를 유지한다. 본문 식 (1)처럼 표본 평균을 쓰면 gradient와 Hessian 양쪽에 같은 $1/N$ 계수가 붙는다. normalization을 혼용해 절대 임계값을 옮기면 안 된다. $\mathbf r\in\mathbb R^q$, $\boldsymbol\theta\in\mathbb R^P$이면 $\mathbf J\in\mathbb R^{q\times P}$이고 $\mathbf G\in\mathbb R^{P\times P}$다. $q$는 설명용으로 추가한 출력 성분 수다.

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

이것은 잔차 함수가 필요한 미분 가능성을 갖는 위치의 항등식이다. 다른 위치에서는 잔차 곡률 항이 남을 수 있으므로 $\mathbf H=\mathbf G$를 전 구간의 사실로 쓰면 안 된다. $\mathbf G$는 여전히 PSD이므로 알고리즘은 음의 곡률을 피하면서 출력 변화의 1차 민감도를 측정할 수 있다. [B, 식 (5)–(6), pp.19–20](https://arxiv.org/pdf/2609.38081v1#page=19)

### 영공간은 왜 유한 이동의 보증이 아닌가?

**리뷰 보충:**

$$
\mathbf v^\top\mathbf G\mathbf v=\|\mathbf J\mathbf v\|^2,
\qquad\ker\mathbf G=\ker\mathbf J.
$$

따라서 영방향은 출력의 *1차* 변화가 0인 방향이다. 다음은 원문 실험이 아닌 설명용 반례다. $f(a,b)=(a+b,b^2)$, anchor $(0,0)$, 단위 방향 $\mathbf v=(1,-1)/\sqrt2$를 쓰면

$$
\begin{aligned}
\mathbf J(0,0)\mathbf v&=\mathbf0,\\
f(t\mathbf v)-f(0,0)&=(0,t^2/2),\\
\mathcal L(t\mathbf v)&=t^4/8.
\end{aligned}
$$

Hessian의 정확한 영방향이어도 유한 $t$에서 손실은 0이 아니다. 실제로 이 예제의 정확한 영손실 해는 원점뿐이다. 국소 영공간이 크다는 것만으로 같은 차원의 전역 해 다양체가 존재한다고 결론내릴 수도 없다. 복원·작은 보폭·수용 검사가 필요한 이유다.

## C. 구성요소 제거 실험 (Ablation Study)

원문 C.1–C.4는 3BFF에서 네 가지 질문을 순서대로 확인한다.

| 절 | 바꾼 것 | 결과와 해석 |
| --- | --- | --- |
| C.1 | 복원 횟수 0·3·6·9·12, 세 보폭 | 큰 보폭 $\eta=0.3$에서 12회 복원은 endpoint 손실을 anchor의 6.6배→3.6배로 낮추지만 제거하지 못한다. 작은 이동 자체가 중요하다. |
| C.2 | soft projection damping | $10^{-9}$는 거의 움직이지 않고, 완화하면 다양성과 drift가 함께 커진다. 투영을 빼면 복원이 있어도 수십 step 안에 기능이 무너진다. |
| C.3 | 재선형화(relinearization) 없음/50/20/5 step 간격 | 자주 갱신할수록 손실을 줄이며 큰 보폭에서 특히 중요하다. |
| C.4 | HNC 대 similarity penalty 학습 | 같은 600 step, 각 설정 5 seed에서 HNC가 손실 수준 대비 더 큰 CKA 거리를 보고한다. penalty는 초반 높은 손실 구간을 통과한 뒤 돌아온다. |

이 비교는 설정된 step 수·보폭을 맞춘 것이며 Hessian/CG 비용을 포함한 동일 wall-clock 예산의 보편적 우위를 입증하지 않는다. 복원은 정확한 제약면 투영이 아니라 drift를 줄이는 보호 장치다. [C·Figures 6–9, pp.20–22](https://arxiv.org/pdf/2609.38081v1#page=20)

## D. 함수 보존과 과제 손실 보존은 다르다

첫째, 학습이 끝난 모델도 과제 gradient가 정확히 0일 필요는 없다. 평평한 방향으로 움직여도 1차 항 $\eta\mathbf g^\top\mathbf v$가 남는다. 원문 3BFF 비교에서는 이 항이 2차 항보다 2–3자릿수 커지고 task loss가 약 10배로 늘었다. anchor 함수 일치는 자신의 출력을 대상으로 하므로 출발점에서 잔차·gradient가 정확히 0이라는 장점이 있다.

둘째, 출력 벡터 전체와 스칼라 손실은 제약의 강도가 다르다. 다음은 ImageNet probe 512개, 클래스 1,000개에서 **표현 거리 수준을 맞춰** 비교한 원문 Table 2다. 값은 anchor와 top-1 예측이 같은 probe의 비율이다.

| 보존량 | 성분 제약 수 | $\Delta\mathrm{CKA}=0.2$ | 0.4 | 0.6 |
| --- | --- | --- | --- | --- |
| 전체 logits | 512,000 | 0.996 | 0.994 | 0.993 |
| 이미지별 cross-entropy | 512 | 0.916 | 0.894 | 0.875 |
| 평균 cross-entropy | 1 | 0.831 | 0.811 | 0.793 |

성분 제약 수가 모두 독립적인 Jacobian rank라는 뜻은 아니다. 같은 cross-entropy를 내는 logit 벡터가 여러 개일 수 있으므로 손실을 맞춰도 예측이 바뀔 수 있다. 이미지별 손실을 맞춘 경우에는 각 이미지 안에서 출력 분포가 달라지는 것이고, 평균만 맞춘 경우에는 이미지 사이 손실의 상쇄도 가능하다.

![원문 Figure 11: 전체 logits, 이미지별 손실, 평균 손실 보존의 비교](/assets/reviews/hnc/constraint-strength.webp)

[Figure 11 크게 보기](/assets/reviews/hnc/constraint-strength.webp)

**Figure 11의 B와 D를 함께 읽는다.** B의 실선은 probe, 점선은 held-out이며, 출력 벡터를 맞춘 보라색이 더 높은 일치를 유지한다. 그러나 A처럼 각 방식의 표현 변화 속도가 다르므로 step을 맞추는 것만으로 충분하지 않다. D는 가로축을 $\Delta\mathrm{CKA}$로 바꾼 비교다. C의 정확도와 B의 anchor 일치도 다른 지표다. band는 probe seed의 범위이며 보편적인 신뢰구간이 아니다. [D·Figure 11·Table 2, pp.22–25](https://arxiv.org/pdf/2609.38081v1#page=24)

과제 수준의 더 넓은 집합을 탐색하고 싶으면 anchor task loss와의 차이를 제곱한 제약을 쓸 수 있다. 그러면 출발 gradient는 0이 되지만 출력 정보가 적다는 두 번째 문제까지 해결되지는 않는다. finite probe의 함수 일치는 labels 없이도 가능하지만 probe 밖 보존은 여전히 별도로 평가해야 한다.

## E. 전역 탐색과 결합하기

저자들은 DSA 조향 local walk와, 이전 해들과 동역학이 다르도록 regularization한 새 학습을 번갈아 수행한다. 8개 네트워크 체인은 기억을 고정점·지속 drift·더 많은 attractor로 구현하며 bit accuracy $0.999$ 초과를 보고한다. local walk는 현재 해의 낮은 손실 영역을 탐색하고 새 학습은 멀리 떨어진 가중치로 갈 수 있다.

그러나 새 모델을 학습하는 구간까지 하나의 함수 보존 경로로 입증한 것은 아니다. 모든 basin을 발견하거나 그 개수를 알려주는 방법도 아니다. ridge를 넘도록 허용 오차를 높이기·작은 random kick으로 재시작하기 등은 이 절의 향후 제안이다. [E·Figures 12–14, pp.26–28](https://arxiv.org/pdf/2609.38081v1#page=26)

## F. RNN 추가 분석: 차원·압축·재학습·OOD

### F.1–F.3 설정과 평평한 공간의 크기

64유닛 RNN은 4,611개 파라미터여서 전체 spectrum을 계산할 수 있다. 5개 anchor의 Hessian participation ratio는 $23.2\pm1.6$이지만, 이것을 영공간 차원으로 빼면 안 된다. 고유값이 큰 몇 방향에 곡률이 집중된 정도를 나타내는 유효 차원이다.

**Participation ratio의 의미.** 이 문단의 $D_{\mathrm{PR}}$은 유효 차원의 리뷰 약칭이다. 0이 아닌 PSD spectrum에 대해

$$
D_{\mathrm{PR}}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
$$

같은 양의 고유값이 $r$개면 $r$이고, 하나에 집중되면 1에 가까워진다. 작은 양의 고유값은 rank에 포함되지만 이 지표에는 약하게 기여한다. 따라서 23.2는 23개 방향만 존재한다는 뜻도, 나머지 4,588개가 정확한 영방향이라는 뜻도 아니다. 모든 고유값이 0이면 위 비율은 정의되지 않는다. [Appendix F.2](https://arxiv.org/pdf/2609.38081v1#page=29)

원문 Table 3의 상대 임계값별 median flat dimension은 다음과 같다.

| $\mu_{\mathrm{rel}}$ | median dimension | 전체 파라미터 중 비율 |
| --- | --- | --- |
| $10^{-7}$ | 1701 | 36.9% |
| $10^{-6}$ | 2896 | 62.8% |
| $10^{-5}$ | 3684 | 79.9% |
| $10^{-4}$ | 4145 | 89.9% |
| $10^{-3}$ | 4418 | 95.8% |
| $10^{-2}$ | 4565 | 99.0% |

명확한 spectral gap 하나가 있어서 유일한 차원을 결정한 결과가 아니다. 허용할 출력 변화가 커질수록 더 많은 방향을 flat하다고 부른다. CKA 조향은 600 step, DSA 조향은 2,000 step, 각 step 복원 12회를 사용한다. steering probe 32 trial과 수용 검사의 128 trial도 구분한다. [F.1–F.3, pp.29–30](https://arxiv.org/pdf/2609.38081v1#page=29)

### F.4–F.8 무엇이 보존되고 무엇이 바뀌는가?

gallery와 10-anchor MDS는 하나의 특이한 도착점만이 아니라 여러 방향의 대안이 있음을 보여준다. 전체 그림의 반복 snapshot을 여기 다시 싣지는 않는다. pruning은 유닛의 recurrent/output outgoing group을 작게 만드는 목표로 한 RNN을 64→11유닛으로 줄여 평가 bit accuracy 100%를 유지한다. 파라미터는 원래의 약 5%지만 MSE는 조금 증가한다. 계산한 원래 구조 기준 파라미터 수는 4,611→212개다.

8,000-step SGD 재개 시에는 낮은 task loss로 내려가도 조향한 CKA·DSA·희소성의 차이가 남는다. 이는 일반 학습이 원래 anchor로 반드시 끌어당기지는 않는다는 관찰이다. OOD pulse 크기·빈도, 긴 무입력 구간과 상태 교란에서는 일부 endpoint가 anchor보다 먼저 실패한다. 표현 다양성과 강건성 향상을 동의어로 쓰지 않는다. [F.4–F.8, pp.30–35](https://arxiv.org/pdf/2609.38081v1#page=31)

## G. ViT 검증: 지표 최적화와 함수 보존을 따로 평가하기

### G.1–G.4 조향 목적과 설정

**CKA의 계산 대상.** 아래 $X,Y$는 같은 표본 순서의 두 특징 행렬을 열별 중심화한 리뷰 설명 기호다. probe 입력 $\mathbf X$와 구분한다.

$$
\operatorname{CKA}_{\mathrm{linear}}(X,Y)
=\frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F\,\|Y^\top Y\|_F}.
$$

일반적인 이 선형 CKA는 0–1 범위이며 높을수록 표본 간 기하가 유사하다. 직교 좌표변환·전체 스케일 변화에는 불변이지만 임의의 가역 변환에는 불변이 아니다. 분모가 0인 상수 특징에서는 정의되지 않는다. CKA 감소는 출력의 변화나 모든 계산의 비동등성을 직접 판정하지 않는다. [CKA 원전 §3–5, Eq. (14)](https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf#page=5)

$c_{ij}(\boldsymbol\theta)$를 anchor $i$층과 현재 모델 $j$층의 linear CKA라고 하면 원문 식 (12)는

$$
\begin{aligned}
\varphi_{\mathrm{CKA}}(\boldsymbol\theta)
&=1-\frac1\beta\log\sum_{i,j=1}^{L}
\exp\!\bigl(\beta c_{ij}(\boldsymbol\theta)\bigr),\\
L&=13,\qquad\beta=30.
\end{aligned}
$$

이를 키우면 가장 비슷하게 남은 층 쌍들에 더 큰 가중치로 낮추는 힘이 작용한다. log-sum-exp는 hard max와 정확히 같지 않으며 최종 보고는 hard max다. G.2의 의미·색·공간 배치는 각각 DINOv2 cosine, HSV $8^3$ histogram cosine, $32\times32$ grayscale correlation이다. 사람 평가와 동일한 지표는 아니다.

G.3의 mutual-kNN은 $k=10$ 이웃 집합 교집합을 측정하고 미분에는 부드러운 이웃 가중치를 사용한다. CKA 목표와 독립적인 metric으로도 정렬이 낮아지는지 확인하는 실험이다. 다만 원문 G.3은 probe 256개, G.4 Table 8은 512개로 달라 특정 kNN run의 정확한 $N$을 확정하지 않는다. 두 조향의 보존 수준 차이를 metric 하나의 본질적 우열로만 설명할 수 없는 이유이기도 하다.

### G.5–G.6 held-out 보존과 단순 손상 비교

CKA 조향은 5개 probe draw, kNN 조향의 off-probe 평가는 3개 draw다. 비교한 diff-seed fine-tuning은 2 seed이며, single anchor·독립 ViT-S/in1k의 band는 이미지 bootstrap이라 반복 단위가 다르다. 하나의 오차 막대 의미로 통일해 읽으면 안 된다.

CKA 조향은 ImageNet/V2/R/Sketch에서 작은 정확도 손실, 오류 일관성 약 0.8을 보인다. kNN 설정은 오류 일관성 약 0.75와 더 낮은 정확한 예측 일치를 보인다. 임의 Gaussian weight noise를 정확도 손실에 맞춘 비교에서도 HNC의 표현 변화가 더 크다. 이는 단순 손상 대안을 약화하지만 지표 변화가 곧 모든 계산의 변경이라는 증명은 아니다. [G.5–G.6, pp.39–42](https://arxiv.org/pdf/2609.38081v1#page=39)

오류 일관성(error consistency)은 각 이미지에서 두 모델이 **둘 다 맞거나 둘 다 틀리는지**의 일치를 우연 수준으로 보정한 지표다. 둘 다 틀렸을 때 같은 오답 클래스를 골라야 하는 prediction match와 다르다. 따라서 약 0.8이라는 값은 80% 정확도나 정확히 80%의 동일 예측을 뜻하지 않는다. [Appendix G.5](https://arxiv.org/pdf/2609.38081v1#page=39)

### G.7–G.10 층·방향·probe 수를 바꾸면?

초기 block 하나도 조향할 수 있지만 pre-logit보다 같은 표현 변화에 기능 비용이 더 든다. ViT-S를 ViT-B와 닮게 하는 반대 실험에서는 평균 depth CKA가 0.78→0.96으로 올라간다. 이때 <strong>probe 예측 일치는 96%</strong>이고, held-out ImageNet 5,000개에서는 약 90%이며 top-1 정확도는 1%p 조금 넘게 내려간다. 완전히 같은 함수로 ViT-B가 된 결과가 아니다.

probe를 늘리는 실험은 보존을 강화하면서 허용된 walk를 더 오래 진행해 더 큰 표현 변화를 얻는 경우를 보여준다. 비용은 probe 수에 따라 대략 선형 증가한다. “제약이 많으면 언제나 도달 영역이 커진다”는 집합론적 결론과 다르다. 마지막 MDS는 metric·probe draw·다시 조향한 방향에 따라 서로 다른 대안이 나타남을 확인한다. [G.7–G.10, pp.43–46](https://arxiv.org/pdf/2609.38081v1#page=43)

## H. Brain-Score 조향: 무엇을 실제로 높였는가?

MajajHong2015 IT benchmark는 원문 기준 이미지 3,200개와 macaque IT 168개 recording site를 사용한다. 보고 점수는 10-fold PLS-25로 특징에서 신경 반응을 예측한 상관을 site median·fold 평균으로 모아 noise ceiling 0.82로 정규화한다.

미분 가능한 조향은 별도 surrogate다. anchor에서 고정한 64차원 PCA 특징에 ridge regression을 적합하고 다른 이미지 절반에서 평균 상관을 높이거나 낮춘다. PLS 공식 점수 자체를 직접 미분한 것은 아니다.

86M relative-position ViT의 up/down IT 점수 범위는 0.618–0.719(약 0.10), 사용하지 않은 site와 image에서는 범위 0.039로 보고된다. plain ViT-S는 held-out image에서 0.546–0.627(0.081)로 변한다. 비교한 7개 모델의 범위는 약 0.074–0.075다. top-1 결정은 최소 94% 보존이고 logit KL은 $10^{-5}$–$10^{-3}$ 범위다. **근사 보존을 exact equality로 바꾸지 않는다.** IT에 대한 조향이 V1·V2·V4의 예측도까지 같은 폭으로 바꾸지는 않는다고 보고한다. 모델의 신경 예측도는 어느 해를 택했는지에도 의존할 수 있다는 결론이다. [H, pp.46–48](https://arxiv.org/pdf/2609.38081v1#page=46)

## I. 강화학습 확장의 수학과 실패

### I.1 Advantage와 surrogate의 의미

Plume·Ant는 GAE를 쓴다. 원문의 시간별 보상 $r_t$와 critic $V$로

$$
\begin{aligned}
\delta_t&=r_t+\gamma V(s_{t+1})-V(s_t),\\
A_t&=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l},\\
\gamma&=0.99,\qquad\lambda=0.95.
\end{aligned}
$$

episode 끝에서는 다음 상태 value를 0으로 둔다. Boat race는 critic 없이 $\gamma=0.9$의 discounted proxy return을 사용한다. buffer에서 advantage를 표준화하므로 기준 정책의 importance ratio가 모두 1이면 surrogate 평균은 0이다. **값이 0이라는 것만으로 gradient도 0이라는 뜻은 아니다.** 지도학습의 제곱 잔차 최소점에서 성립한 PSD 논리를 RL surrogate에 그대로 옮기면 안 된다. buffer 갱신과 실제 episode 평가가 필요한 이유다.

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

상태 인자는 식 길이를 줄이려고 이 블록에서만 생략했다. 공분산은 walk 내내 고정한다. 비가중 평균 행동 MSE와 상수배로 같으려면 모든 $\sigma_j$가 같아야 한다. 그렇지 않으면 fixed reweighting을 생략한 별도의 목적이다. 이 중간 항 상쇄는 원문 I.2의 유도를 풀어 쓴 것이다.

### I.3–I.7 세 환경의 조건과 한계

| 환경 | buffer/평가 | 관찰된 결과·한계 |
| --- | --- | --- |
| Plume | current-policy 8 episodes, 5 step마다 갱신; 정규 평가 60 초기 조건 | casting 감소와 보상 유지. OOD는 조건별 240개, 8×30 batch의 표준오차. step 20에서는 보존을 벗어나 성능 붕괴. |
| Boat race | 32 episodes, 3 step마다 갱신, 탐색 $\epsilon=0.25$; 평가 100 episodes | true-return으로 학습한 10 anchor에서 출발. 가장 다른 3 endpoint가 proxy를 받으면서 순진행 0의 exploit으로 이동. |
| Ant-v3 | 4,000 transitions, 5 step마다 갱신; 고정 30 초기 상태 | 19 step에서 action divergence 1.0. 같은 action-KL의 random 방향은 return 절반 이상 손실. HNC 대안은 mean return을 유지하지만 실패 유형은 변한다. |

Ant의 걸음 분석은 걷는 episode만 사용해 anchor 25/30, 대안 27/30이다. 대안은 멈춰 서는 실패가 줄지만 torso 높이가 범위를 벗어나 조기 종료하는 실패가 생긴다. 두 정책의 초기 상태별 return은 비상관으로 보고된다. 평균 성능의 동등성을 episode별 동일성이나 전반적 안전성으로 바꿔 읽지 않는다. Plume의 서로 다른 OOD 조건은 초기 상태 grid가 달라 조건 안의 paired 비교와 조건 사이 절대 성공률 비교도 구분한다. [I, pp.49–53](https://arxiv.org/pdf/2609.38081v1#page=49)

## J. 손실 지형 측정의 해석 범위

폭 5개×클래스 수 5개×3 seed의 CNN grid는 75개다. GroupNorm을 사용해 batch 구성에 따른 출력을 피하고, 각 모델의 학습 입력 512개에서 함수 변화량을 측정한다. J.2는 절대 drift 허용값을, J.3은 상대 spectral cutoff를 여러 값으로 바꾸어 경향을 확인한다. J.4의 feature covariance participation ratio는 클래스 수가 늘면 약 5배로 커지지만 폭 변화에는 상대적으로 둔감하다. **특징의 유효 차원과 가중치의 flat fraction은 다른 측정**이다.

J.1의 학습 정확도 100%를 cross-entropy 0과 동일시하는 문장은 채택하지 않는다. 예를 들어 정답 확률 0.6도 top-1이 정답일 수 있지만 cross-entropy는 양수다. 함수 일치는 정답 대신 자기 출력을 목표로 하므로 이 구분과 독립적으로 anchor 잔차가 0이다.

**리뷰 보충:** finite probe 출력 성분이 $q$개라면 $\operatorname{rank}(\mathbf J)\le q$이고, $\dim\ker\mathbf J\ge P-q$다. 따라서 많은 파라미터와 적은 probe 제약 자체로 큰 선형 영공간이 생길 수 있다. 이것만으로 실제 입력 분포의 모든 동작에 자유가 그만큼 남는다고 결론낼 수 없다. probe 밖 평가가 실험의 필수 부분이라는 뜻이다. [J, pp.53–55](https://arxiv.org/pdf/2609.38081v1#page=53)

## 출처와 읽은 범위

- 원문: Huang 등 여섯 저자, [HNC v1](https://arxiv.org/abs/2609.38081v1), 최초 공개 2026-09-29, 검토 2026-10-17. 첫 읽기와 전체 정독 범위는 [1편 출처](/reviews/hnc-solutions/#출처와-읽은-범위)에 기록했다.
- 이 글은 부록 A–J의 순서·역할을 유지했다. A–B의 알고리즘·식 (4)–(6), C의 네 ablation, D의 Table 2, F의 Table 3, G–J의 검증 조건과 반대 결과를 보존했다. 반복 gallery와 모든 solver hyperparameter 표·그림의 중복 전사는 줄였다.
- 보충 유도: 성분별 product rule, PSD·양의 정부호, Rayleigh quotient, soft projection의 이차 최소화, 비선형 반례, finite-probe rank bound와 일부 산술. $q,\mathbf g,a,b,t$ 등 설명용 기호는 추가 표기로 밝혔다. 논문 전체의 형식적 정확성이나 전역 수렴을 인증한 것이 아니다.
- 불일치: G.3의 kNN probe 256 대 Table 8의 512, J.1의 정확도/zero-loss 표현, 일반 Algorithm 1과 F.3의 손실 제어량 차이를 구분했다. 저자에게 확인하지 않았다.
- 그림: 공통 Figure 1과 D의 Figure 11, 원문 여섯 저자, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [저자 v1 소스](https://arxiv.org/src/2609.38081v1). PDF rasterization·WebP encoding 외에 내용·배치·화살표·문자를 바꾸지 않았다.
- **미실행:** 저자 코드·GPU/학습/walk/뇌 데이터/RL 재현, raw result 통계 검정, 하드웨어 비용 재측정, 전역·기계 형식 검증, 참조 LOBPCG/CG 이론 전체의 독립 증명, 설정 불일치의 저자 확인. 이 글의 국소 계산 검토와 구분한다.

- 2026-10-22 재점검: 기존 두 읽기와 원문 대응 기록을 재사용해 위에 추가한 지표·가정·예제의 정의를 보완했다. 새 전문 두 읽기로 계산하지 않는다. 새 모델·GPU·로봇 재현과 전체 형식 검증은 미실행이다.

- 추가 지표 대조: CKA 원전의 §3–5, Eq. (14)와 focal Appendix F.2·G.5의 지표 설명을 재확인했다. 뇌·모델 원시 데이터로 점수를 재계산하지 않았다.
