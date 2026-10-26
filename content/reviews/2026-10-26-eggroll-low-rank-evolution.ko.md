---
title: "EGGROLL: 저랭크 탐색으로 넓히는 학습의 범위"
description: "저랭크 섭동의 GPU 효율, 고차원 이론과 정수 RNN·Transformer 후학습을 분석하고 Dust로 이어지는 연구 질문을 정리한다."
date: "2026-10-26"
publishedAt: "2026-10-26T19:34:40+09:00"
updatedAt: "2026-10-26T19:34:40+09:00"
topics: ["zeroth-order optimization", "evolution strategies", "credit assignment", "language model pretraining", "efficient training"]
visibility: "public"
lang: "ko"
translationKey: "eggroll-low-rank-evolution"
paperTitle: "Evolution Strategies at the Hyperscale"
paperPublishedDate: "2025-11-20"
authors: ["Bidipta Sarkar", "Mattie Fellows", "Juan Agustin Duque", "Alistair Letcher", "Antonio León Villares", "Anya Sims", "Clarisse Wibault", "Dmitry Samsonov", "Dylan Cope", "Jarek Liesen", "Kang Li", "Lukas Seier", "Theo Wolf", "Uljad Berdica", "Valentin Mohl", "Alexander David Goldie", "Aaron Courville", "Karin Sevegnani", "Shimon Whiteson", "Jakob Nicolaus Foerster"]
year: 2025
paperUrl: "https://arxiv.org/abs/2511.16652v2"
thumbnail: "/assets/reviews/eggroll/eggroll-fig1.webp"
thumbnailAlt: "EGGROLL 원문 Figure 1: 저랭크 후보의 적합도를 평가해 밀집 가중치를 갱신하는 구조."
---

EGGROLL은 가중치의 무작위 변화인 섭동(perturbation)을 저랭크(low-rank) 행렬로 만들어, GPU에서 큰 진화 전략(evolution strategies, ES) 집단을 빠르게 평가한다. 정수 순환 신경망(recurrent neural network, RNN) 사전학습에서는 시험 손실 3.40 bits/byte로 비교 Transformer의 3.58보다 낮지만, 최대 집단은 약 180배의 GPU 시간을 사용한다. [§6.1, Figure 2](https://arxiv.org/html/2511.16652v2#S6.SS1) **리뷰 해석.** 이 연구는 추론에 적합한 비미분 모델까지 학습 대상으로 넓히는 강한 시스템 기여이며, 역전파와의 계산 효율 경쟁은 과제와 구조를 맞춘 별도의 문제로 남는다.

활성값을 토큰별로 흔들어 학습 신호를 늘리는 후속 접근은 [Dust 리뷰](/reviews/dust-activation-search/)에서 이어진다.

## 1. 도입 (Introduction)

ES는 모델을 조금씩 다르게 만든 여러 후보의 적합도(fitness)를 평가하고, 좋은 후보의 변화 방향을 모아 매개변수를 움직인다. 목적함수나 모델을 미분하지 않아도 되고, 후보 평가는 병렬화하기 쉽다. 그러나 매개변수가 수십억 개이면 후보마다 모든 가중치를 복제·이동하는 비용이 커진다. GPU는 큰 공통 행렬을 여러 입력에 적용할 때 빠르지만, 입력마다 다른 큰 행렬을 읽으면 메모리 이동에 묶인다.

Evolution Guided GeneRal Optimisation via Low-rank Learning(EGGROLL)은 이 섭동을 두 얇은 행렬의 곱으로 표현한다. 후보는 저랭크지만, 후보들의 보상 가중 합을 원래 가중치에 더하므로 업데이트 전체는 고랭크가 될 수 있다. 저자들은 고차원에서의 ES·EGGROLL 분석, 정수 언어 모델 사전학습, 제어와 언어 모델 후학습으로 이를 뒷받침한다. [§1](https://arxiv.org/html/2511.16652v2#S1)

[![EGGROLL 원문 Figure 1. 랭크 1 섭동을 평가한 뒤 가중 합으로 원래 가중치를 갱신하는 구조.](/assets/reviews/eggroll/eggroll-fig1.webp)](/assets/reviews/eggroll/eggroll-fig1.webp)

원문 Figure 1. 같은 초기 가중치에 서로 다른 저랭크 변화를 넣고 평가한다. 오른쪽의 합은 개별 섭동보다 높은 랭크가 될 수 있다.

## 2. 예비 지식 (Preliminaries)

### 2.1 저랭크 행렬 근사 (Low-Rank Matrix Approximations)

원문의 저랭크 적응(low-rank adaptation, LoRA) 비교는 $M_i\approx M_i^0+A_iB_i^\top$이다. $M_i^0$는 고정된 기초 가중치, $A_i\in\mathbb R^{m\times r}$, $B_i\in\mathbb R^{n\times r}$는 얇은 행렬이다. LoRA가 이 인자 자체를 학습하는 것과 달리, EGGROLL은 매 스텝 저랭크 섭동을 새로 뽑고 결과를 원래 밀집 가중치에 반영한다. [§2.1](https://arxiv.org/html/2511.16652v2#S2.SS1)

### 2.2 진화 전략 (Evolution Strategies)

적합도 $f:\mathbb R^d\to\mathbb R$를 최대화한다고 하자. 원문은 평균 $\mu$, 분산 $\sigma^2I_d$의 가우시안 집단을 사용한다. 평활화한 목적함수와 그 업데이트 방향은 다음과 같다.

$$
J(\theta)=\mathbb E_{v\sim\mathcal N(0,I_d)}[f(\mu+\sigma v)],
\qquad
\nabla_\mu J(\theta)=\frac1\sigma\mathbb E[v f(\mu+\sigma v)].
$$

$\theta$는 집단 분포의 매개변수이고, 실제로는 $\mu$를 바꾸며 $\sigma$를 조정 가능한 상수로 둔다. $N$개 후보로 기대값을 평균하고 학습률 $\alpha_t$로 상승한다. 미분하는 대상은 분포이며, $f$ 자체를 통과해 역전파하지 않는다. [§2.2](https://arxiv.org/html/2511.16652v2#S2.SS2)

### 2.3 행렬 매개변수의 ES (Evolution Strategies for Matrix Parameters)

$W,M,E\in\mathbb R^{m\times n}$일 때 표준 가우시안 행렬 $E$를 써서

$$
\widehat\nabla_M J(\theta)=\frac{1}{\sigma N}\sum_{i=1}^N E_i f(W=M+\sigma E_i)
$$

로 추정한다. 열을 쌓는 $\operatorname{vec}$로 행렬을 벡터화하면 차원은 $d=mn$이고, 행렬의 프로베니우스 노름(Frobenius norm)은 벡터의 유클리드 노름과 같다. 뒤의 고차원 증명을 행렬 업데이트에 적용하는 연결이다. [§2.3, 부록 B](https://arxiv.org/html/2511.16652v2#S2.SS3)

## 3. 관련 연구 (Related Work)

저자들은 진화 알고리즘의 분산 병렬화, 정책 학습·메타학습·하이퍼파라미터 탐색을 먼저 검토한다. 저랭크 구조를 사용하는 선행 방법도 목적이 다르다. 작은 집단에 좋은 방향을 주는 부분공간 탐색과, 저랭크로 인자화한 모델 자체를 최적화하는 접근은 EGGROLL처럼 밀집 가중치 전체를 갱신하는 방식과 구분된다.

언어 모델에서는 MeZO 같은 두 점 영차 추정과 저랭크·LoRA 기반 미세조정, 추론 과제의 ES가 비교 대상이다. 기존 연구는 같은 섭동으로 많은 응답을 생성해 GPU를 활용하기도 한다. EGGROLL은 각 생성에 서로 다른 섭동을 배정해 추론 배치 크기에 가까운 집단을 평가하는 방향이다. 저자들은 사전학습에는 큰 집단이 필요하다는 실험을 이 차별점과 연결한다. [§3](https://arxiv.org/html/2511.16652v2#S3)

## 4. EGGROLL

### 4.1 저랭크 진화 전략 (Low-Rank Evolution Strategies)

원문의 Assumption 1은 $A$와 $B$의 원소를 서로 독립인 동일 분포로 뽑으며, 평균 0·분산 1·대칭·절대연속성·유한한 4차 모멘트를 요구한다. 섭동은

$$
E=\frac1{\sqrt r}AB^\top,
\qquad A\in\mathbb R^{m\times r},\quad B\in\mathbb R^{n\times r}
$$

이다. $1/\sqrt r$는 랭크가 늘어도 원소의 분산을 1로 유지한다. 낮은 $r$의 분포는 가우시안이 아니며, 저랭크 다양체 위에 놓여 일반적인 유클리드 공간의 밀도 점수함수를 바로 사용할 수 없다. 저자들은 가우시안의 점수함수 $\hat S(E)=-E$를 **근사**로 사용한다.

각 worker가 적합도 $f_i$를 공유하고, 공통 난수 seed로 섭동을 재생성해 원래 가중치를 갱신한다.

$$
M_{t+1}=M_t+\frac{\alpha_t}{N_{\mathrm{workers}}}
\sum_{i=1}^{N_{\mathrm{workers}}}E_{i,t}f(W=M_t+\sigma E_{i,t}).
$$

여기서는 $1/\sigma$를 학습률에 흡수했다. 원문의 Algorithm 1과 업데이트 식이다. 일반적인 비퇴화 조건에서 합의 랭크는 $\min(Nr,m,n)$까지 커진다. 개별 후보를 저랭크로 제한하는 것과 최종 가중치를 저랭크로 고정하는 것은 다르다. [§4.1](https://arxiv.org/html/2511.16652v2#S4.SS1)

### 4.2 하드웨어 효율적인 구현 (Hardware-Efficient Implementation)

행 벡터 입력 $u_i$에 대해, 후보의 큰 가중치 행렬을 실제로 만들 필요가 없다.

$$
u_i(M+\sigma E_i)^\top
=u_iM^\top+\frac{\sigma}{\sqrt r}(u_iB_i)A_i^\top.
$$

공통 $M$에 대한 큰 행렬곱을 유지하고, 각 후보에 작은 추가 연산만 수행한다. $r=1$이면 입력과 $B_i$의 내적, 그 스칼라와 $A_i$의 곱이다. 업데이트도 $\sum_i E_if_i$를 전부 만들지 않고 $(\operatorname{diag}(f)A)^\top B$라는 행렬곱으로 모은다. [§4.2](https://arxiv.org/html/2511.16652v2#S4.SS2)

산술 집약도(arithmetic intensity)는 이동한 byte당 연산 수다. 부록 F의 정사각 행렬·bfloat16(bf16) 모델에서 밀집 가우시안 ES의 값은 $m/(m+2)<1$이고, EGGROLL은 배치가 커질수록 공통 가중치를 재사용한다. $m=8,192$, 랭크 1, 목표 300 ops/byte일 때 필요한 배치는 일반 추론 324, EGGROLL 352다. [부록 F](https://arxiv.org/html/2511.16652v2#A6)

[![EGGROLL 원문 Figure 2. 정규화 처리량과 정수 EGG 사전학습 손실.](/assets/reviews/eggroll/eggroll-fig2.webp)](/assets/reviews/eggroll/eggroll-fig2.webp)

원문 Figure 2. 왼쪽 100은 순수 배치 추론 처리량이며 EGGROLL은 91, OpenES는 0.41이다. 오른쪽은 집단을 늘린 정수 RNN의 손실이며, 선 색이 집단 크기다. 왼쪽의 처리량 실험과 오른쪽의 학습 성능 실험은 서로 다른 측정이다.

91%의 속도 근거는 GH200 한 GPU, 폭 8,192의 bf16 선형 모델, 최대 배치 1,024에서 **노이즈를 미리 생성한** 실험이다. 부록 E의 Figure 7은 JAX 노이즈 재생성을 포함한 속도도 별도로 보여준다. **리뷰 해석.** 이 수치는 후보 평가의 시스템 효율을 지지하며, 원하는 손실에 도달할 총 학습 비용의 91%나 모든 네트워크의 학습 처리량으로 확대할 수는 없다. [부록 E](https://arxiv.org/html/2511.16652v2#A5)

## 5. 분석 (Analysis)

### 5.1 고차원 가우시안 ES (High-Dimensional Gaussian ES)

표준 가우시안 벡터의 크기는 보통 $\sqrt d$다. 원소별 노이즈가 작아도 전체 섭동은 $\sigma_d\sqrt d$ 크기로 커진다. 저자들은 이 값이 0으로 가는 $\sigma_d=o(d^{-1/2})$에서 ES가 국소 미분 방향으로 접근한다고 분석한다.

Theorem 1의 조건은 초기화 $\mu$ 주위 고정 반지름 안에서 $C^1$이고 gradient가 $\alpha$-Hölder 연속, 전역적으로 함수와 미분이 차원에 독립적인 상수로 다항식에 의해 제한되며, $\lVert\mu\rVert$와 $\lVert\nabla f(\mu)\rVert$가 $O(1)$인 것이다. 원문은 오차율을 $\Theta((\sigma_d\sqrt d)^\alpha)$로 적는다. [§5.1, Theorem 1](https://arxiv.org/html/2511.16652v2#S5.SS1)

증명은 부록 C.1의 다음 순서다. 국소 공 안에서 1이고 바깥에서 0이 되는 cutoff로 $f$를 바꿔, 국소적으로만 매끄러운 함수를 절대연속·적분 가능한 함수로 만든다. 다항식 꼬리 보조정리로 바뀐 부분의 오차를 지수적으로 제한한다. 그 함수에 가우시안 부분적분인 Stein 보조정리를 적용해 보상 가중 노이즈를 평균 미분으로 바꾼다. 마지막으로 국소 Hölder 조건을 적용한다.

$$
\mathbb E\lVert\nabla f(\mu+\sigma_dv)-\nabla f(\mu)\rVert
\le L\sigma_d^\alpha\mathbb E\lVert v\rVert^\alpha
=O((\sigma_d\sqrt d)^\alpha).
$$

이 식은 국소 부분의 상한이며 cutoff 밖의 꼬리 오차는 별도로 더해진다. Stein 보조정리를 불연속 원함수에 곧바로 적용하지 않는 이유가 cutoff다. 저자들은 $f(x)=\frac L2\sum_jx_j\lvert x_j\rvert+a^\top x$, $\mu=0$을 써서 $\alpha=1$일 때 오차 $L\sigma_d\sqrt{2d/\pi}$를 얻는 예를 제시한다. [부록 C.1](https://arxiv.org/html/2511.16652v2#A3.SS1)

Theorem 2는 삼차 다항식

$$
f(x)=a^\top x+\frac12x^\top Bx+\frac16 C(x,x,x)
$$

에서 정확한 잔여 항을 계산한다. $B$는 대칭 행렬, $C$는 대칭 삼중선형 형식이고 각 연산자 노름은 $O(1)$이다. 홀수 가우시안 모멘트가 사라져

$$
\nabla_\mu J-\nabla f(\mu)
=\frac{\sigma_d^2}{2}\mathbb E[C(v,v,\cdot)]
$$

가 남는다. 따라서 국소화되는 구간, $\sigma_d\asymp d^{-1/2}$인 임계 구간, 더 느리게 줄어 일부 목적에서 발산하는 구간을 구분한다. 임계 스케일은 함수 클래스의 최악 경우를 가르는 경계다. [Theorem 2, 부록 C.2](https://arxiv.org/html/2511.16652v2#A3.SS2)

### 5.2 고차원 EGGROLL (High-Dimensional EGGROLL)

랭크 $r$를 고정하고 $d=mn$을 늘리면 EGGROLL의 곱 노이즈는 가우시안보다 무거운 꼬리를 갖는다. Theorem 3은 앞의 다항식·유계 미분 가정에 더해, 국소 $C^2$와 Hessian의 Lipschitz 상수 $L_d$, 인자 분포의 sub-Gaussian 꼬리를 요구한다. 노이즈 조건은 $\sigma_d=o(d^{-1/2})$, $L_d(\sigma_dd)^2=o(1)$이다. 원문이 제시하는 상한은 다음과 같다.

$$
\lVert\hat g_{\mathrm{LR}}-\nabla_W f(M)\rVert_F
=O\!\left(L_d(\sigma_dd)^2\right)
+O\!\left(\frac{\sqrt d}{\sigma_d^2}
\exp\!\left[-K\frac{\rho}{\sqrt d\sigma_d}\right]\right).
$$

$K>0$는 꼬리 상수, $\rho$는 국소 공 반지름이다. 이 $K$는 Dust의 집단 크기와 다른 기호다. 원문은 이 전체가 $o(1)$이고 가우시안 ES와의 차이도 사라진다고 진술한다. [§5.2, Theorem 3](https://arxiv.org/html/2511.16652v2#S5.SS2)

증명의 핵심은 $v=\operatorname{vec}(E)$가 평균 0·공분산 $I_d$라는 점이다. Taylor 전개의 상수 항은 $\mathbb E[v]=0$으로, 이차 항은 대칭성에 따른 삼차 모멘트 0으로 사라진다. 선형 항은 원하는 미분을 남긴다. Hessian 변화로 생기는 나머지를 적분하면

$$
\lVert I_{\mathrm{loc}}\rVert
\le\frac{L_d\sigma_d^2}{6}\mathbb E\lVert v\rVert^4
=O(L_d\sigma_d^2d^2)
$$

가 된다. 인자들의 sub-Gaussian 성질은 곱의 sub-exponential 꼬리와 유한 모멘트를 주며, 이를 국소 공 밖의 적분에 적용한다. 마지막으로 가우시안 ES와의 차이는 두 추정치의 국소 미분 오차를 삼각부등식으로 더한다. [부록 C.3](https://arxiv.org/html/2511.16652v2#A3.SS3)

### 5.3 랭크 분석 (Rank Analysis)

이번 극한은 차원을 고정하고 랭크 $r$를 늘리는 것이다. Theorem 4는 Assumption 1과 **전역 유계 적합도** 아래에서 가우시안 ES 업데이트와의 차이를 $O(r^{-1})$로 진술한다. [§5.3](https://arxiv.org/html/2511.16652v2#S5.SS3)

$\operatorname{vec}(E^r)=r^{-1/2}\sum_{j=1}^r\operatorname{vec}(a_jb_j^\top)$이고 각 항의 공분산은 $I_d$다. 다변량 중심극한정리가 가우시안 극한을 준다. 부록 D는 Edgeworth 전개로 그 속도를 분석한다. 대칭 분포의 3차 누적량(cumulant)이 0이므로 일반적인 $r^{-1/2}$ 항이 사라지고, 4차 누적량의 $r^{-1}$ 항이 앞선다. 원문이 인용한 Edgeworth 근사와 가우시안 사이의 적분 오차를 각각 제한한 뒤 삼각부등식으로 업데이트 차이를 제한한다. 고정 차원·고정 $\sigma$에서의 결과이므로 차원·랭크·노이즈를 동시에 바꾸는 보장으로 사용하지 않는다. 부록 D.1의 다른 mean-field 점수 근사도 제시되지만, 실험에서는 가우시안 근사를 사용한다. [부록 D](https://arxiv.org/html/2511.16652v2#A4)

## 6. 실험 (Experiments)

### 6.1 순수 정수 언어 모델 사전학습 (Pure Integer Language Model Pretraining)

EGG(Evolved Generative GRU)는 6층·폭 256의 비선형 RNN이며 게이트 순환 유닛(gated recurrent unit, GRU)을 변형한 구조다. int8 가중치와 정수 활성값, int32 누산, 비트 시프트와 lookup table을 사용한다. 별도 ReLU·tanh를 없애도 int8로 되돌릴 때의 포화·클리핑이 비선형성을 만든다. 정규화는 제곱근 대신 평균 절댓값을 쓰고, 다음 토큰의 log-likelihood도 지수·로그 테이블로 계산한다. [§6.1, 부록 G](https://arxiv.org/html/2511.16652v2#S6.SS1)

정수 버전은 양·음 섭동 쌍의 적합도 차이 부호를 $\{-1,0,1\}$로 만들고, 보상 가중 행렬곱의 크기가 임계값을 넘으면 가중치를 한 정수 bin씩 움직인다. 이는 실수 ES 식을 그대로 정수에 대입한 것과 구분되는 구현이다. [부록 H](https://arxiv.org/html/2511.16652v2#A8)

MiniPile의 문자/byte 수준 예측을 학습하며, 스텝당 데이터 배치 16개를 고정한다. 각 후보는 100토큰 후 업데이트하고, 문서 경계에서만 은닉 상태를 초기화하는 truncated ES를 쓴다. 집단은 2부터 $2^{20}=1,048,576$까지다. 지표 bits/byte는 정답 byte에 할당한 확률의 음의 밑 2 로그 평균이며 낮을수록 좋다. 최대 집단의 3.40은 같은 깊이·폭·데이터 배치의 fp32 Transformer·역전파 확률적 경사 하강법(stochastic gradient descent, SGD) 3.58보다 낮지만, 약 180배의 GPU-hours를 쓴다. 아키텍처와 수치 형식이 함께 바뀐 비교다. [§6.1, Figure 2b](https://arxiv.org/html/2511.16652v2#S6.SS1)

[![EGGROLL 원문 Figure 8. 데이터 배치 크기와 집단 크기를 함께 바꾼 EGG 시험 손실.](/assets/reviews/eggroll/eggroll-fig8.webp)](/assets/reviews/eggroll/eggroll-fig8.webp)

원문 Figure 8. 가로축은 데이터 배치, 색은 집단 크기이며 낮은 곡선이 좋다. 배치를 늘리면 같은 후보의 적합도 추정이 깨끗해질 수 있지만 새 섭동 방향 수를 늘리는 것과는 다르다. 부록 I는 $b>N/2$일 때 실제 병렬 평가 수가 $\max(2b,N)$로 늘어나는 계산량 효과도 설명한다. 데이터 배치 4에서는 두 최대 집단의 성능이 비슷하나 더 많은 데이터에서 갈라진다. [부록 I](https://arxiv.org/html/2511.16652v2#A9)

### 6.2 강화학습 과제 (Reinforcement Learning Tasks)

처음부터 학습하는 16개 제어 환경에서 3층·폭 256 다층 퍼셉트론(multilayer perceptron, MLP) 정책을 사용한다. 방법·환경별 하이퍼파라미터를 탐색하고 선택한 설정을 seed 10개로 평가한다. 근접 정책 최적화(Proximal Policy Optimization, PPO)로 정규화한 보상과 평균의 표준오차를 보고하며, OpenES와 비교해 7개 환경에서는 비슷하고 7개에서는 높으며 2개에서는 낮다. 환경 길이가 학습 중 달라질 수 있어 per-step 처리량과 총 wall-clock 시간이 다르게 보이는 경우도 있다. [§6.2, Figure 4a](https://arxiv.org/html/2511.16652v2#S6.SS2), [부록 N.4, Figures 14–15](https://arxiv.org/html/2511.16652v2#A14.SS4)

협력형 다중 에이전트 환경 3개에서는 3층·폭 64 정책과 랭크 1, A100 40GB를 사용한다. 원문은 큰 배치에서 OpenES 대비 최대 2.4배 빠르면서 경쟁력 있는 성능을 보고한다. [부록 N.1, Figure 12](https://arxiv.org/html/2511.16652v2#A14.SS1)

### 6.3 기초 모델 후학습 (Foundation Model Fine-tuning)

RWKV-7은 recurrent state를 사용해 Transformer의 길어지는 KV cache보다 많은 후보를 병렬 생성하기 좋다. Countdown의 1.5B 모델에서는 H200 한 장·8시간, seed 3개로 EGGROLL 약 35%, GRPO 약 23%의 검증 정확도를 보고한다. Group Relative Policy Optimization(GRPO)은 응답별 보상으로 정책을 업데이트하는 역전파 기준선이다. 적합도는 질문별 정답 점수를 중심화하고 전체 표준편차로 정규화한 뒤 질문에 걸쳐 평균한다. [§6.3](https://arxiv.org/html/2511.16652v2#S6.SS3), [부록 N.2](https://arxiv.org/html/2511.16652v2#A14.SS2)

[![EGGROLL 원문 Figure 5. RWKV-7 7B의 GSM8K 시간별 검증 점수와 어려운 추론 과제 결과.](/assets/reviews/eggroll/eggroll-fig5.webp)](/assets/reviews/eggroll/eggroll-fig5.webp)

원문 Figure 5. 왼쪽은 8 GPU의 GSM8K 후학습에서 seed별 곡선이다. EGGROLL과 GRPO의 총 병렬 생성은 각각 8,192와 256이고, 프롬프트당 생성 수·온도·optimizer가 다르다. 오른쪽은 별도의 7B DeepScaleR 실험이다. 원문 캡션은 128 GPU·12시간, 본문은 24시간으로 적어 예산 표기가 일치하지 않는다. [Figure 5, §6.3](https://arxiv.org/html/2511.16652v2#S6.F5), [부록 N.3, Tables 12–13](https://arxiv.org/html/2511.16652v2#A14.SS3)

14B 모델은 32 GPU·12시간, 학습·평가 최대 5,000토큰으로 AIME24 13→30%, AIME25 7→33%, HMMT25 11→13%를 보고한다. 이 크기에서는 같은 GRPO 실험을 실행하지 못했고, AIME24로 최고 체크포인트를 선택했다. 별도의 7B Countdown 72.9%와 다른 논문의 Qwen 결과 비교는 모델과 하드웨어 조건이 다르므로 저자도 공정한 시간 비교의 한계를 밝힌다. [Figure 13, 부록 N.2](https://arxiv.org/html/2511.16652v2#A14.F13)

### 6.4 정수 양자화 언어 모델 후학습 (Fine-tuning Integer Quantised LLMs)

RWKV-7 7B를 채널별 absmax int8로 양자화한 뒤, 원래 모델이 준 분포에 맞추는 증류(distillation)를 수행한다. GSM8K의 정답 풀이를 teacher forcing으로 주고, 토큰별 Kullback–Leibler(KL) divergence를 합해 줄인다. Figure 6은 seed 3개의 훈련 perplexity와 미관측 GSM8K 검증 점수가 회복되는 것을 보여준다. [§6.4, Figure 6](https://arxiv.org/html/2511.16652v2#S6.SS4)

이 경로는 순수 정수 EGG와 다르다. 큰 행렬은 int8이나 스케일·일부 작은 매개변수·Adam 모멘트는 bf16이다. Adam이 제안한 실수 업데이트를 정규화·임계화해 $-1,0,+1$의 정수 bin 변화로 바꾼다. 원문의 KL 합은 최소화할 증류 비용이며, ES의 최대화 적합도로 구현할 때에는 음의 비용을 써야 한다. [부록 K](https://arxiv.org/html/2511.16652v2#A11)

## 7. 결론 (Conclusion)

저자들은 저랭크 탐색으로 큰 모델의 블랙박스 최적화를 가능하게 하고, 작은 랭크에서도 밀집 업데이트를 구성할 수 있다고 결론짓는다. 앞으로는 기억·계산 같은 상징 모듈과 연결된 신경 기호 시스템(neurosymbolic systems), 추론 harness와 상호작용하는 모델, 복수 에이전트 전체를 end-to-end로 최적화하는 방향을 제안한다. [§7](https://arxiv.org/html/2511.16652v2#S7)

## L. 추가 Transformer 후학습 (Fine-tuning Pretrained Transformer LLMs with Verifiable Rewards)

v2 부록 L은 Transformer 자체의 후학습도 다룬다. Qwen3-4B-Base를 4만 수학 문제의 DeepScaleR에 학습하고, 랭크 1·집단 2,048로 평가한다. 아래는 Table 1 전체다. RL 값은 저자들이 같은 조건으로 새로 실행한 결과가 아니라 GEM 논문에서 가져왔으며, 최대 응답 길이·프롬프트 등의 공유 설정을 맞췄다고 설명한다. [부록 L.1](https://arxiv.org/html/2511.16652v2#A12.SS1)

| 방법 | MATH500 | OlympiadBench | AIME24 | AMC | MinervaMath | 원문 평균 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B-Base | 50.2 | 24.4 | 10.0 | 33.7 | 21.7 | 28.0 |
| +EGGROLL | 75.8 | 37.3 | 13.3 | 49.4 | 31.3 | 41.4 |
| +RL(인용값) | 67.4 | 33.5 | 16.7 | 49.4 | 40.1 | 41.4 |

단위는 정확도 %. 평균 41.4가 같아도 EGGROLL은 MATH500·OlympiadBench에서 높고 AIME24·MinervaMath에서는 낮다. 반올림된 개별 셀의 평균은 EGGROLL 41.42, RL 41.42다. [Table 1](https://arxiv.org/html/2511.16652v2#A12.T1)

Qwen3-1.7B에서는 집단 256·랭크 1로 pass@1과 pass@4를 최적화한다. 여기서 pass@4는 질문당 네 응답 중 하나라도 정답인 비율이다. 모델을 통해 직접 미분하지 않고 이 집합 보상을 후보 fitness로 쓴다. Figure 10은 pass@4 학습에서 네 응답의 서로 다른 최종 답 수가 늘고, pass@1에서는 줄어드는 경향을 보여준다.

[![EGGROLL 원문 Figure 10. pass@1과 pass@4 학습의 적합도 및 네 응답의 답 다양성.](/assets/reviews/eggroll/eggroll-fig10.webp)](/assets/reviews/eggroll/eggroll-fig10.webp)

원문 Figure 10. 보라색은 pass@1, 분홍색은 pass@4 최적화다. 왼쪽 곡선은 각기 다른 fitness이므로 높이를 같은 지표의 우열로 읽지 않는다. 오른쪽은 네 응답 안의 서로 다른 최종 답 개수다. [부록 L.1](https://arxiv.org/html/2511.16652v2#A12.F10)

vLLM의 다중 LoRA 추론, tensor parallel weight slice를 인식하는 업데이트, 층별 임시 메모리 관리, GPU 간 가중치 동기화가 이 경로를 받친다. [부록 L.2, Tables 2–3](https://arxiv.org/html/2511.16652v2#A12.SS2)

## M. 추가 시계열 모델 사례 (Fine-tuning Time Series Foundation Model: High-Frequency Trading)

부록 M은 주문 흐름에 사전학습한 S5 모델을 주문 실행 에이전트로 조정하는 예제다. 사전학습은 GOOG 2022년 자료 약 25B 토큰, 후학습은 2023년 1월의 고정 구간과 Jax-LOB 시뮬레이터를 쓴다. 10스텝 안에 30주를 파는 과제이며, 남은 수량은 마지막 시장가 주문으로 처리한다. 초기 중간가 대비 매도 대금의 차이인 PnL을 순위 변환해 적합도로 사용한다. 투영 행렬에는 랭크 4 LoRA를 적용하고 SSM·정규화는 고정한다. Figure 11의 훈련 곡선은 이 목적을 직접 최적화할 수 있음을 보이는 보조 사례다. [부록 M, Figure 11·Table 4](https://arxiv.org/html/2511.16652v2#A13)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 시스템 기여와 통계적 효율을 나누면 성과가 선명해진다. 큰 가중치 복제 없이 많은 후보를 평가하는 방법은 정수 연산, 결과만 채점하는 목적, 추론 엔진을 학습에 활용하는 경로를 연다. 정수 EGG의 데이터 재사용과 Transformer 후학습은 이를 서로 다른 방향에서 보여준다. 다음 과제는 평가가 싸졌을 때 원하는 방향을 얻는 데 후보가 얼마나 필요한가다.

### 이론을 사용할 때의 두 범위

**리뷰 해석.** Theorems 1–2의 인쇄된 $\Theta$를 모든 함수의 양방향 오차율로 읽기에는 진술이 강하다. 부록 C는 일반 상한과 그 상한을 달성하는 예를 제시한다. 선형 $f(x)=a^\top x$는 조건을 만족하면서 가우시안 ES 오차가 정확히 0이다. 삼차 $f(x)=x_1^3$, $\mu=0$는 차원이 늘어도 오차가 $3\sigma_d^2$여서 $\Theta(\sigma_d^2d)$가 아니다. 일반 진술은 $O$ 상한과 최악 경우의 날카로움으로 분리하는 편이 정확하다. 원문 Theorems 1–2의 위치와 예는 [§5.1](https://arxiv.org/html/2511.16652v2#S5.SS1)에 있다.

**리뷰 보충 유도.** 반대로 저자들이 경계의 날카로움을 보이려는 예 $f(x)=(u^\top x)\lVert x\rVert^2$, $\lVert u\rVert=1$에서는 가우시안 미분 평균을 직접 계산해

$$
\mathbb E[\nabla f(\sigma_dv)]
=\sigma_d^2\{u\mathbb E\lVert v\rVert^2+2\mathbb E[vv^\top]u\}
=\sigma_d^2(d+2)u
$$

를 얻는다. 따라서 일부 함수에서 임계 스케일이 필요하다는 결론은 유지된다.

**리뷰 해석.** Theorem 3에 표시된 꼬리 상한이 자동으로 $o(1)$이라는 단계에도 추가 확인이 필요하다. $\sigma_d=d^{-1/2}/\log\log d$이면 $\sigma_d\sqrt d\to0$이지만 표시된 꼬리 항은 $d^{3/2}(\log\log d)^2(\log d)^{-K\rho}$ 꼴로 0으로 가지 않는다. 이는 상한에서 결론으로 가는 증명 단계의 부족이며, 실제 업데이트가 발산한다는 반례는 아니다. 예를 들어 $\sigma_d=d^{-1/2-\varepsilon}$, $\varepsilon>0$이면 표시된 꼬리는 사라지고, 국소 항 조건을 함께 확인할 수 있다. 원문 부록 C.3은 이 두 항을 더하는 것이 가장 강한 근거다. 이 지적은 실험 결과나 랭크 증가에 대한 별도 Theorem 4를 뒤집지 않는다.

**리뷰 해석.** 유한 집단의 추정 분산, 적합도 shaping, 정수 격자 위의 동역학은 위 기대 업데이트의 극한과 별도다. 특히 국소 미분이 0인 정수 모델을 학습한 실험의 효용을 “미분으로 선형화한다”는 정리만으로 설명할 수는 없다. 목적에 맞는 유한 노이즈가 어느 탐색을 수행하는지 분석할 여지가 크다.

### Dust와 잇는 연구 방향: 싼 평가에서 유용한 신호로

**리뷰 해석.** [Dust의 방법](/reviews/dust-activation-search/#2-방법-method)은 한 후보에 전역 점수 하나를 주는 대신 토큰별 활성 변화를 각각 채점한다. 그 연구의 EGGROLL-Transformer 기준선이 약한 결과와, 여기의 정수 RNN·RL 후학습 성공은 모순되지 않는다. dense loss를 가진 처음부터의 사전학습, 결과 보상이 있는 이미 학습된 모델, 정수 recurrent 구조라는 조건이 다르기 때문이다.

**리뷰 해석.** 연구 우선순위는 두 축을 함께 재는 비용 곡선이다. 같은 모델·데이터·GPU 시간에서 목표 손실에 도달할 비용을 비교하고, 집단의 수·토큰별 신호·통신·캐시·튜닝 예산을 분해해야 한다. 추천하는 첫 실험은 dense token loss를 쓴 학습에서 층별 EGGROLL/Dust 혼합이 추정 분산과 시험 손실을 줄이는지 보는 것이다. 이어 보상을 마지막 결과 하나로 바꾸면 Dust의 국소 신호 이득이 유지되는지 검증할 수 있다. 구체적인 혼합 가정과 반증 기준은 [Dust의 연결 절](/reviews/dust-activation-search/#eggroll과-잇는-연구-방향-정보가-부족한-부분에-예산-배분하기)에 모았다.

## 출처와 읽은 범위

- 원문: Bidipta Sarkar et al., *Evolution Strategies at the Hyperscale*, [arXiv:2511.16652v2](https://arxiv.org/abs/2511.16652v2), 최초 공개 2025-11-20, 수정 2026-02-16, 76쪽. 검토일 2026-10-26.
- 읽은 범위: 본문 §1–7, 부록 A–D의 중심 정리와 증명 흐름, E–K의 속도·정수 모델·절제·양자화, L–M 추가 실험, N의 주요 설정·학습 곡선. 설정 표의 반복 행과 mean-field 주변 유도는 축약한다. [공식 프로젝트](https://eshyperscale.github.io/)와 [Dust 원문](https://qlabs.sh/research/dust)을 함께 참고한다.
- 그림: v2 Figures 1·2·5·8·10, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 페이지 본문·캡션과 구분해 그림 영역을 추출하고 래스터화·WebP 인코딩했다. 패널·축·데이터·원문 글꼴은 유지했다.
- 확인하지 않은 것: 저자 실험의 독립 GPU 재현과 증명 전체의 형식 검증.
