---
title: "[1/2] Diffusion Forcing: 시간과 노이즈를 따로 조절하면 무엇이 달라질까?"
description: "토큰마다 독립 노이즈를 학습하는 원리부터 장기 영상 생성, Monte Carlo Guidance, 미로 계획과 로봇 기억 실험까지 원문 순서로 읽는다."
date: "2026-10-14"
publishedAt: "2026-10-14T13:58:28+09:00"
updatedAt: "2026-10-23T15:26:21+00:00"
topics: ["diffusion models", "sequence modeling", "world action model", "robot learning", "probabilistic modeling"]
visibility: "public"
lang: "ko"
translationKey: "diffusion-forcing-methods"
paperTitle: "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion"
paperPublishedDate: "2024-07-01"
authors: ["Boyuan Chen", "Diego Marti Monso", "Yilun Du", "Max Simchowitz", "Russ Tedrake", "Vincent Sitzmann"]
year: "2024"
paperUrl: "https://arxiv.org/abs/2407.01392v4"
thumbnail: "/assets/reviews/diffusion-forcing/overview.webp"
thumbnailAlt: "Diffusion Forcing 원문 Figure 2. 시퀀스의 시간축과 노이즈축, 그리고 토큰별 독립 노이즈 학습·샘플링을 teacher forcing·전체 시퀀스 diffusion과 비교한 그림."
---

Diffusion Forcing(DF)은 토큰마다 독립적으로 뽑은 노이즈 수준(independent per-token noise)으로 학습해, 생성 길이·기억·미래 유도(guidance)를 재학습 없이 샘플링 단계에서 고르게 한다. 원문 Table 1의 Maze2D Large(목표 고정)에서 DF의 episode reward는 159.0±2.7로, 같은 모델에서 Monte Carlo 유도(Monte Carlo Guidance, MCG)만 뺀 142.8±5.6보다 높다. [원문 Table 1](https://arxiv.org/pdf/2407.01392v4#page=8) **리뷰 해석.** 샘플링만 바꾼 MCG 절제가 여섯 미로 환경 모두에서 점수를 올려 계획 쪽 주장을 가장 직접 뒷받침하지만, 장기 영상 안정성은 정량 지표 없는 사례로, 로봇의 기억·관찰 오염 결과는 시행 횟수 없는 성공률로만 제시되어 “두 방식의 장점을 결합한다”는 주장의 나머지 근거는 시연 수준이다.

이 글은 v4의 본문 §1–5와 실험·구현 부록 B–F를 원문 순서로 다루고, 부록 A의 ELBO 논증과 인쇄식의 KL 부호·전이 방향·가중치 오류는 [\[2/2\] 이론과 증명 해설](/reviews/diffusion-forcing-theory/)에서 다룬다.

## 1. 도입: 다음 토큰 예측과 전체 시퀀스 확산 (Introduction)

여기서 **토큰**(token)은 단어에 한정되지 않는다. 영상 프레임, 시계열 관측 벡터, 로봇의 행동·관찰 묶음도 토큰이다.

**교사 강요**(teacher forcing)는 학습할 때 정답 과거를 주고 다음 토큰을 예측하게 한다. 저자들은 이렇게 학습한 다음 토큰 모델의 두 한계를 든다. 샘플링을 어떤 목적에 맞게 유도할 수단이 없고, 연속 데이터에서는 추론 때 자기 출력을 다음 입력으로 쓰면서 작은 오차가 누적되어(compounding error) 학습 길이를 넘으면 발산하기 쉽다.

**전체 시퀀스 확산**(full-sequence diffusion)은 정해진 개수의 토큰을 이어 붙여 모든 토큰에 같은 노이즈 수준으로 노이즈를 넣고 제거한다. 원하는 결과 쪽으로 샘플링을 바꾸는 **유도**를 쓸 수 있지만, 비인과적 구조라 가변 길이 생성과 부분 시퀀스 생성이 어렵다. 다음 토큰 모델을 전체 시퀀스 확산으로 학습하는 단순한 결합도 생성 품질이 나쁘다고 저자들은 보고한다.

DF는 “앞 토큰을 완전히 결정한 뒤 다음 토큰을 예측한다”와 “모든 토큰의 노이즈 수준이 같다”는 두 제약을 함께 푼다. 가까운 미래는 먼저 확정하고 먼 미래는 여러 가능성으로 남겨 둘 수 있다. 학습 때 토큰별 노이즈를 독립적으로 뽑았기 때문에, 이 선택은 재학습이 아니라 **샘플링 스케줄**의 변경으로 이루어진다. [§1, Figure 1–2](https://arxiv.org/pdf/2407.01392v4#page=2)

## 2. 관련 연구와 배경 (Related Work and Preliminaries)

이 논문에는 두 축이 있다. 아래 첨자 $t$는 시퀀스의 시간이고, 위 첨자 $k$는 노이즈 수준이다. $\mathbf x_t^{k_t}$는 “시간 $t$의 토큰을 노이즈 수준 $k_t$로 오염시킨 것”이다. $\mathbf z_t$는 과거 정보를 요약한 잠재 상태(latent state)다.

### 베이지안 필터링과 확산의 연결

**베이지안 필터링**(Bayesian filtering)은 사전 모델 $p(\mathbf z_{t+1}\mid\mathbf z_t)$, 관찰 모델 $p(\mathbf x_t\mid\mathbf z_t)$, 새 관찰을 반영하는 사후 모델 $p(\mathbf z_{t+1}\mid\mathbf z_t,\mathbf x_{t+1})$로 잠재 상태를 차례로 갱신한다. 신경망으로 끝단 간 학습하면 잠재 상태는 물리량의 추정치가 아니라 미래 관찰을 예측하는 데 필요한 과거의 요약이 된다.

확산의 전방 과정(forward process)은 원문 식 (2.1)과 같다.

$$
q(\mathbf x^k\mid\mathbf x^{k-1})
=\mathcal N\!\left(
\mathbf x^k;\sqrt{1-\beta_k}\mathbf x^{k-1},
\beta_k\mathbf I\right).
$$

$\alpha_k=1-\beta_k$, $\bar\alpha_k=\prod_{j=1}^k\alpha_j$로 두면, 전방 과정을 한 번에 샘플링할 수 있다.

$$
\mathbf x_t^{k_t}
=\sqrt{\bar\alpha_{k_t}}\mathbf x_t^0
+\sqrt{1-\bar\alpha_{k_t}}\boldsymbol\epsilon_t,
\qquad
\boldsymbol\epsilon_t\sim\mathcal N(0,\mathbf I).
$$

두 번째 식은 [DDPM 식 (4)](https://arxiv.org/pdf/2006.11239v2#page=2)의 닫힌 형태다. $k_t=0$이면 원래 토큰이고, $\mathbf x_t^K$는 근사적으로 순수 노이즈다. 역과정은 신경망으로 평균을 예측하는 Gaussian 마르코프 연쇄이며, 표준 설명대로 노이즈 예측 $\boldsymbol\epsilon_\theta$의 최소제곱 목적으로 학습한다. 유도 쪽에서 원문은 원하는 속성 $y$의 분류기 $c(y\mid\mathbf x^k)$의 로그 기울기로 샘플링 방향을 고치는 분류기 유도(classifier guidance)를 기준으로 삼는다.

원문 §2는 다음 토큰 예측 모델이 미래를 샘플링하려면 이전 프레임을 완전히 확정해야 하므로 여러 단계에 걸친 유도를 받지 못한다고 정리한다. 확산 시퀀스 모델로는 유도로 텍스트를 제어하는 Diffusion-LM, 슬라이딩 윈도로 영상을 늘리는 video diffusion, 오프라인 강화학습의 계획기 Diffuser, 교사 강요로 학습하는 TimeGrad를 들고, 가장 가까운 연구로 노이즈 수준이 시간축 위치에 선형으로 묶인 AR-Diffusion을 꼽는다. 부록 C는 Rolling Diffusion도 위치에 묶인 노이즈 수준을 학습과 샘플링에 그대로 써야 하므로 같은 한계를 공유한다고 덧붙인다.

### 리뷰 보충: 가까운 선행 접근 비교

| 접근 | 무엇을 학습하는가? | Diffusion Forcing과의 차이 |
| --- | --- | --- |
| [DDPM](https://arxiv.org/pdf/2006.11239v2), §2–3 | 한 데이터의 전방·역방향 확산과 노이즈 제거 손실 | DF는 이 구성에 시간별 기억과 토큰별 독립 노이즈를 결합한다. |
| [TimeGrad](https://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf), §3 | 정답 과거를 요약한 순환 신경망(recurrent neural network, RNN) 상태에 조건부로 다음 시계열 벡터를 확산으로 생성 | DF는 과거도 여러 노이즈 수준으로 학습하고 여러 미래 토큰을 함께 유도할 수 있다. |
| [Diffuser](https://proceedings.mlr.press/v162/janner22a/janner22a.pdf), §2–3.1 | 상태·행동 궤적 전체를 함께 생성하고 목표에 맞춰 유도 | DF는 인과적 구조와 토큰별 스케줄로 미래의 불확실성과 계획 길이를 조절한다. |
| [AR-Diffusion](https://arxiv.org/pdf/2305.09515v1), §3.1–3.2 | 문장 단계와 단어 위치로 토큰별 확산 단계를 정함 | 토큰마다 다른 노이즈 수준이라는 선행 아이디어가 있고, DF의 차이는 위치에 묶이지 않은 독립 노이즈 학습이다. |

TimeGrad는 시계열 표, Diffuser는 미로 실험의 비교 대상이다.

## 3. 방법 (Method)

### 3.1 노이즈를 부분 마스킹으로 보기 (Noising as partial masking)

**부분 마스킹**(partial masking)은 정보를 완전히 가리거나 남기는 두 선택 사이에 여러 정도를 두는 관점이다. 저자들은 교사 강요를 시간축을 따라 토큰을 가리는 마스킹으로, 전방 확산을 노이즈축을 따라 조금씩 가리는 부분 마스킹으로 본다. 노이즈 0은 가리지 않은 토큰, 노이즈 $K$는 완전히 가린 토큰이다.

DF는 두 축을 하나로 묶는다. 시퀀스의 토큰마다 다른 노이즈 수준 $k_t$를 줄 수 있고, 이 수준이 각 토큰에 적용한 부분 마스킹의 정도다. [§3.1, p. 4](https://arxiv.org/pdf/2407.01392v4#page=4)

![원문 Figure 2: 시간축과 노이즈축 및 세 가지 학습·샘플링 구성](/assets/reviews/diffusion-forcing/overview.webp)

[Figure 2 크게 보기](/assets/reviews/diffusion-forcing/overview.webp)

**Figure 2를 읽는 법.** 왼쪽의 가로축은 시간, 세로축은 노이즈다. 가운데 DF의 학습 행에서는 같은 시퀀스 안에서도 오염 정도를 따로 뽑으므로 토큰마다 색이 다르다. 오른쪽의 교사 강요는 깨끗한 과거로 다음 토큰을 예측하고, 전체 시퀀스 확산은 모든 토큰을 같은 노이즈 단계에서 처리한다. 아래 샘플링 행에서는 DF가 시간 방향의 연결을 유지하면서 각 열의 노이즈를 서로 다른 속도로 제거하는 모습을 볼 수 있다. 굵은 가로 막대는 전체 시퀀스를 함께 처리하는 비교 구조이고, 색은 노이즈 수준이다. [원문 Figure 2, p. 3](https://arxiv.org/pdf/2407.01392v4#page=3)

### 3.2 토큰마다 다른 노이즈 수준 (Different noise levels for different tokens)

논문의 구현은 인과적 RNN이다. 학습 때 $k_1,\ldots,k_T$를 독립적으로 뽑아 토큰마다 다른 노이즈를 넣는다. 잠재 상태는 다음처럼 갱신된다.

$$
\mathbf z_t\sim p_\theta\!\left(
\mathbf z_t\mid\mathbf z_{t-1},
\mathbf x_t^{k_t},k_t\right).
$$

구현의 $\mathbf z_t$는 결정론적이며, 저자들은 이를 믿음(belief)의 표본이 아니라 믿음의 분포를 나타내는 값으로 설명한다. $k_t=0$은 관찰을 반영한 베이지안 필터링의 사후 갱신(posterior update), $k_t=K$는 관찰 정보가 없는 사전 예측(prior prediction)에 대응한다.

학습 손실은 원문 식 (3.1)이다.

$$
\mathbb E_{\substack{k_{1:T},\,\mathbf x_{1:T},\,\boldsymbol\epsilon_{1:T}\\
\mathbf z_t\sim p_\theta(\mathbf z_t\mid
\mathbf z_{t-1},\mathbf x_t^{k_t},k_t)}}
\left[
\sum_{t=1}^{T}
\left\|
\boldsymbol\epsilon_t-
\boldsymbol\epsilon_\theta(
\mathbf z_{t-1},\mathbf x_t^{k_t},k_t)
\right\|^2
\right].
$$

손실 자체는 표준 확산의 MSE이고, 달라진 점은 **조건으로 주는 과거에도 노이즈가 섞여 있다는 것**이다. 잠재 상태가 그 과거를 요약하며, 기울기는 잠재 상태의 갱신을 거쳐서도 흐른다. Algorithm 1은 시퀀스 샘플링→독립 노이즈 수준 샘플링→토큰 오염→잠재 상태와 예측 계산→시퀀스 손실 역전파의 순서다. 원문은 노이즈 수준 집합을 $[K]$와 $\{0,\ldots,K\}$로 섞어 쓴다. [§3.2·Algorithm 1, pp. 4–5](https://arxiv.org/pdf/2407.01392v4#page=4)

실험이 모두 이 $\boldsymbol\epsilon$ 예측을 쓰지는 않는다. 부록 D.3에 따르면 영상·시계열은 $\mathbf v$ 예측, 계획·모방 학습은 $\mathbf x_0$ 예측을 쓴다. 영상에는 노이즈가 섞인 과거가 담은 신호까지 반영하도록 다시 유도한 **융합 신호 대 잡음비 가중치**(fused SNR reweighting)도 쓴다. 저자들은 이 가중치가 영상 예측의 수렴을 크게 앞당겼고, 비이미지 영역에서는 이득이 없어 쓰지 않았다고 한다(D.1).

**샘플링은 두 차원의 스케줄을 고르는 문제다.** $\mathcal K_{m,t}$는 샘플링 행 $m$에서 시간 $t$ 토큰이 도달할 노이즈 수준이다. Algorithm 2는 행을 따라 내려가고, 각 행 안에서는 과거에서 미래로 잠재 상태를 갱신하며 노이즈를 제거한다. 마지막 행에서는 모든 토큰이 깨끗해진다. 모든 노이즈 수준 조합으로 학습했으므로 $\mathcal K$만 바꿔 재학습 없이 다른 동작을 얻는다는 것이 저자들의 요점이다.

본문 Theorem 3.1은 학습이 증거 하한(evidence lower bound, ELBO)의 재가중을 최적화하고, 적절한 조건에서 모든 노이즈 수준 시퀀스의 우도 하한을 동시에 최대화한다고 주장한다. 저자들은 $k_t\in\{0,K\}$인 특수한 경우를 들어, DF가 앞선 토큰을 임의로 가린 조건부 분포, 곧 학습 집합의 모든 부분 시퀀스 분포를 배운다고 덧붙인다. 이 주장의 조건과 부록 증명의 인쇄식 오류는 [이론 편](/reviews/diffusion-forcing-theory/)에서 다룬다.

### 3.3 새로운 시퀀스 생성 능력 (New Capabilities in Sequence Generation)

**자기회귀 생성 안정화.** 영상 같은 고차원 연속 시퀀스에서 자기회귀 구조는 학습 길이를 넘으면 발산하기 쉽다. DF는 생성한 프레임을 완벽한 정답 과거로 취급하지 않고, 작은 노이즈 수준 $0<k_{\mathrm{small}}\ll K$의 토큰에 대응하는 잠재 상태로 다음 생성을 이어 간다(부록 B.4). 이렇게 약간 오염된 과거는 학습 때 본 분포 안에 있으므로, 추론 오차를 학습 때 경험한 노이즈로 흡수하는 셈이다. 잠재 상태를 다시 초기화하거나 슬라이딩 윈도를 쓰지 않는다.

**먼 미래를 불확실하게 남기기.** 원문은 세 토큰의 예로 이를 설명한다. $K$가 짝수일 때 노이즈 수준 목록이 다음처럼 바뀐다.

$$
\begin{aligned}
(K,K,K)
&\longrightarrow(0,K/2,K)\\
&\longrightarrow(0,0,K/2)\\
&\longrightarrow(0,0,0).
\end{aligned}
$$

각 괄호는 토큰 값이 아니라 **노이즈 수준들의 목록**이다. 저자들은 노이즈 수준을 불확실성으로 해석해, 이 “zig-zag” 샘플링이 가까운 미래를 먼 미래보다 확실하게 둔다고 설명한다. 부록 D.7은 같은 아이디어를 계획용 스케줄 행렬로 일반화해 피라미드 스케줄(pyramid scheduling)이라 부른다.

**장기 유도.** 인과적 생성에서 미래 토큰은 과거에 의존하므로, 미래 목표의 기울기가 이 의존 경로를 따라 과거로 전파된다. 저자들이 꼽는 차이는 과거를 완전히 확정하지 않고도 미래 토큰의 노이즈를 제거할 수 있어, 그 기울기가 아직 확정 중인 과거 토큰의 샘플링을 유도한다는 점이다. [§3.3·B.3–B.5·D.7](https://arxiv.org/pdf/2407.01392v4#page=5)

### 3.4 유연한 순차 의사결정 (Flexible Sequential Decision Making)

행동 $\mathbf a_t$, 보상 $\mathbf r_t$, 다음 관찰 $\mathbf o_{t+1}$을 한 토큰으로 묶는다.

$$
\mathbf x_t=
[\mathbf a_t,\mathbf r_t,\mathbf o_{t+1}]^\top.
$$

과거를 요약한 $\mathbf z_{t-1}$에서 미래 계획 $\hat{\mathbf x}_{t:t+H}$를 생성하고 첫 행동 $\hat{\mathbf a}_t$만 실행한다. 다음에는 **예측한 보상·관찰 대신 환경에서 받은 실제 값**으로 잠재 상태를 갱신한다. 모델 예측 제어(model predictive control, MPC)처럼 반복해서 다시 계획하는 구조다. $H$는 원문의 look-ahead 창이며 구간 $t:t+H$는 양 끝을 포함한다.

**유연한 계획 길이.** 저자들은 DF가 행동을 하나씩 고르므로 길이가 정해지지 않은 과제에 쓸 수 있고, 재학습 없이 $H$를 줄이면 지연(latency)이 낮은 정책(policy)으로, 늘리면 장기 계획기(planner)로 쓸 수 있다고 한다. 전체 궤적 길이로 생성하는 Diffuser는 앞의 것이, 짧은 고정 look-ahead를 쓰는 diffusion policy는 뒤의 것이 불가능하다는 것이 저자들의 대비다. 유도 보상도 궤적 전체의 밀집 보상, look-ahead 안의 보상, 목표 도달의 희소 보상 $-\|\mathbf o_T-\mathbf g\|^2$ 가운데 고를 수 있다. [§3.4, p. 6](https://arxiv.org/pdf/2407.01392v4#page=6) **리뷰 해석.** Diffuser 원전 §3.1도 fully convolutional 구조의 입력 길이를 바꿔 계획 길이를 조절할 수 있다고 밝히므로, DF의 차이는 길이 조절 자체보다 행동을 순차로 고르며 실제 관찰로 기억을 갱신한다는 데 있다. 지연이 얼마나 줄어드는지는 원문이 측정하지 않는다.

**MCG**는 현재 토큰 $\mathbf x_t^k$를 고정한 채 여러 미래를 샘플링하고, 각 미래 보상의 유도 기울기를 평균한다. 저자들은 모델 예측 경로 적분(model predictive path integral, MPPI) 같은 shooting 방법처럼, 한 미래가 아니라 미래 분포 전체의 기대 보상으로 현재 토큰을 유도한다고 설명한다. 아래는 이를 적은 **리뷰 보충 표기**다. $N$은 미래 표본 수, $\widehat G^{(i)}$는 $i$번째 표본의 추정 보상이다.

$$
\widehat{\mathbf g}
=\frac1N\sum_{i=1}^{N}
\nabla_{\mathbf x_t^k}\widehat G^{(i)}.
$$

먼 미래 토큰의 노이즈 수준을 높게 유지하는 zig-zag 스케줄과 함께 쓰면 효과가 커진다고 한다. 부록 B.5는 전체 시퀀스 확산이 모든 토큰의 노이즈를 함께 제거하므로 고정된 노이즈 수준에서 Monte Carlo 추정에 쓸 무작위성의 원천이 없다고 설명하고, 본문 §4.2는 이를 “formally derive”했다고 쓴다. [§3.4·B.3·B.5](https://arxiv.org/pdf/2407.01392v4#page=6) **리뷰 해석.** B.5의 설명은 정식 유도라기보다 구조에 대한 논증이며, 저자들도 B.6의 대체 기법(replacement trick)으로 미래 롤아웃을 흉내 낼 가능성은 열어 둔다.

## 4. 실험 (Experiments)

### 4.1 영상 예측: 일관되고 안정적인 생성과 무한 롤아웃 (Video Prediction)

Minecraft와 DMLab의 1인칭 이동 영상으로 합성곱 RNN 구현을 학습하고, 앞의 안정화를 쓰며 자기회귀로 롤아웃(rollout)한다. 비교 대상인 교사 강요 다음 프레임 확산과 인과적 전체 시퀀스 확산은 **같은 RNN 구조**를 쓴다. 저자들은 DF가 학습 길이를 훨씬 넘어(예: 1000 프레임) 안정적으로 롤아웃하는 반면 두 기준선은 빠르게 발산하고, 학습 길이 안에서도 전체 시퀀스 확산은 프레임 사이가 급격히 바뀐다고 보고한다.

![원문 Figure 3: DMLab과 Minecraft에서 세 학습 방식의 장기 영상 생성 비교](/assets/reviews/diffusion-forcing/video-rollouts.webp)

[Figure 3 크게 보기](/assets/reviews/diffusion-forcing/video-rollouts.webp)

**Figure 3를 읽는 법.** 왼쪽은 DMLab, 오른쪽은 Minecraft다. 빨간 테두리는 조건으로 준 실제 입력이고 이후는 생성 결과다. 가로 방향의 496·500·996·1000은 프레임 위치다. 위 DF 행은 긴 롤아웃에서도 장면 구조를 유지하고, 가운데 인과적 전체 시퀀스 기준선은 장면이 급격히 바뀌며, 아래 교사 강요는 후기 프레임이 패턴이나 잡음으로 무너진다. 두 Seq 행은 서로 다른 생성 사례다. [Figure 3·§4.1, pp. 7–8](https://arxiv.org/pdf/2407.01392v4#page=7)

부록 E.3은 cherry-picking 없이 고른 180 프레임 사례와, 2000 프레임까지 실행해도 붕괴하지 않았다는 관찰을 보고한다. Minecraft 에이전트가 막혀 화면 전체가 흙 블록이 되는 경우가 가끔 있지만, 저자들은 이를 데이터셋 문제로 본다. F.1의 해상도는 Minecraft 128×128, DMLab 64×64이며 전체 데이터의 약 10%만 학습에 쓴다. 학습 클립 길이는 Minecraft가 72 프레임이고, DMLab은 E.3이 36 프레임, F.1이 48 프레임으로 적어 서로 다르다.

영상 모델은 Minecraft 36M·DMLab 24M 매개변수다. $K=1000$이고 추론은 DDIM(denoising diffusion implicit models) 샘플러 100단계, 프레임 묶음(frame stacking)은 각각 8·4다. 학습은 A100 8장, 배치 128, 50K 단계이며 40K 단계 무렵 약 12시간 만에 수렴한다고 보고한다. 영상 실험에는 정량 지표가 없고, 근거는 Figure 3과 E.3의 사례다. [D.2·D.6·D.10·E.3·F.1](https://arxiv.org/pdf/2407.01392v4#page=24)

### 4.2 확산 계획: MCG·인과적 불확실성·유연한 길이 (Diffusion Planning)

오프라인 강화학습 벤치마크 D4RL의 Maze2D U-Maze·Medium·Large에서 무작위 보행 데이터로 미로마다 모델 하나를 학습한다. 관찰은 2D 위치·속도의 4차원 벡터, 행동은 2D 가속도이며, 보상은 목표 반경 0.5 안에 있으면 1, 아니면 0인 희소 보상이다. 목표가 고정된 Maze2D와 목표도 무작위인 Multi2D를 평가한다.

![원문 Table 1 위 도식: DF와 Diffuser가 미로 계획의 노이즈를 제거하는 과정](/assets/reviews/diffusion-forcing/maze-planning.webp)

[계획 도식 크게 보기](/assets/reviews/diffusion-forcing/maze-planning.webp)

**계획 도식을 읽는 법.** 두 미로에서 위 행은 DF, 아래는 Diffuser다. 가로 방향은 환경의 실행 시간이 아니라 **노이즈 제거의 진행**이다. 색 띠의 start→end는 계획 안에서의 순서다. DF는 가까운 구간을 먼저 정하고 먼 구간을 더 불확실하게 남기는 반면, Diffuser는 모든 구간을 같은 노이즈 수준으로 처리한다(원문 캡션). 점이 퍼진 부분은 아직 확정되지 않은 샘플링 상태다.

원문 Table 1의 모든 비교 모델을 두 표로 나눠 옮긴다. 기준선은 MPPI, 오프라인 강화학습의 보수적 Q 학습(conservative Q-learning, CQL)과 암묵적 Q 학습(implicit Q-learning, IQL), 그리고 Diffuser다. 값은 원문이 episode reward라 부르는 점수이며, $\pm$의 실행 횟수와 정의는 보고되지 않는다.

| 환경 | MPPI | CQL | IQL | Diffuser\* | Diffuser 생성 행동 실행 | DF, MCG 제외 | DF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U-Maze | 33.2 | 5.7 | 47.4 | $113.9\pm3.1$ | $6.3\pm2.1$ | $110.1\pm3.9$ | $116.7\pm2.0$ |
| Medium | 10.2 | 5.0 | 34.9 | $121.5\pm2.7$ | $13.5\pm2.3$ | $136.1\pm10.2$ | $149.4\pm7.5$ |
| Large | 5.1 | 12.5 | 58.6 | $123.0\pm6.4$ | $6.3\pm2.1$ | $142.8\pm5.6$ | $159.0\pm2.7$ |
| 원문 평균 | 16.2 | 7.7 | 47.0 | 119.5 | 8.7 | 129.67 | 141.7 |

| Multi2D 환경 | MPPI | CQL | IQL | Diffuser\* | Diffuser 생성 행동 실행 | DF, MCG 제외 | DF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U-Maze | 41.2 | — | 24.8 | $128.9\pm1.8$ | $32.8\pm1.7$ | $107.7\pm4.9$ | $119.1\pm4.0$ |
| Medium | 15.4 | — | 12.1 | $127.2\pm3.4$ | $22.0\pm2.7$ | $145.6\pm6.5$ | $152.3\pm9.9$ |
| Large | 8.0 | — | 13.9 | $132.1\pm5.8$ | $6.9\pm1.7$ | $129.8\pm1.5$ | $167.1\pm2.7$ |
| 원문 평균 | 21.5 | — | 16.9 | 129.4 | 20.6 | 127.7 | 146.2 |

**100을 넘는 점수의 뜻.** 원문은 이 값을 “episode reward”(D.8)·“average reward”(Table 1 캡션)라고만 부르고 정규화 여부를 밝히지 않는다. **리뷰 해석.** 100을 넘는 값의 척도는 D4RL이 정의한 정규화 점수와 맞는다. D4RL 구현([get_normalized_score](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py))은 환경별 기준 return $R_{\min},R_{\max}$로 다음 점수를 계산하며, 기준 최대 return이 이론적 상한이 아니므로 이를 넘는 return은 100을 넘는다.

$$
S(R)=100\,\frac{R-R_{\min}}{R_{\max}-R_{\min}}.
$$

DF가 “highest average reward”를 얻는다는 Table 1 캡션은 표와 맞는다. Single-task 평균은 DF 141.7 대 Diffuser\* 119.5, multi-task는 146.2 대 129.4다. 그러나 본문 §4.2의 “outperforms Diffuser and all baselines across all 6 environments”는 표와 맞지 않는다. Multi2D U-Maze에서는 **DF 119.1이 Diffuser\* 128.9보다 낮다**. MCG를 넣으면 여섯 환경 모두에서 DF의 점수가 오르며(예: Multi2D Large 129.8→167.1), 저자들은 MCG를 빼도 DF가 경쟁력을 유지한다고 쓴다.

<strong>Diffuser*</strong>는 생성한 행동을 버리고, 생성한 상태 경로를 수작업 PD(비례-미분) 제어기로 따라가게 한 결과다. 생성한 행동을 그대로 실행하면 평균이 8.7·20.6으로 떨어진다. 저자들은 이를 Diffuser가 생성한 상태와 행동이 인과적으로 일관되지 않다는 근거로 들고, 생성한 행동을 그대로 실행하는 DF가 PD 제어기를 붙인 Diffuser\*보다도 높다고 강조한다. 두 방법은 유도 목적도 다르다. 둘 다 데이터셋 보상을 버리고 목표로 계획하지만, DF는 도달 시점을 정하지 않도록 모든 시점의 목표 거리 합을 유도 보상으로 쓴다. 저자들은 이 보상을 Diffuser에도 시도했으나 조정을 거듭해도 작동하지 않았다고 한다(D.5). [Table 1·§4.2, p. 8](https://arxiv.org/pdf/2407.01392v4#page=8)

부록 D.8은 episode reward가 좋은 지표가 아니라고 경고한다. 데이터셋에는 목표에 머무는 궤적이 없어 목표에 도달한 에이전트가 다시 떠나기 쉬운데(D.5), Diffuser는 느린 계획을 만들어 목표 근처에 오래 머문 덕에 높은 보상을 받았을 수 있다는 것이다. 저자들은 대안으로 처음 목표에 도달하는 시간(first hitting time)을 권하고 DF가 이 지표에서 뛰어나다고 쓰지만, 그 수치는 제시하지 않는다.

**유연한 길이.** 고정 길이 과제에서는 진행할수록 남은 계획 길이가 줄어야 한다. 저자들은 DF가 이를 설계상 지원하는 반면, Diffuser 같은 전체 시퀀스 모델은 조정을 해도 성능이 낮다고 한다. 부록 B.6은 이미 생성한 토큰을 고정하고 나머지만 갱신하는 “conditioning by replacement”가 한 단계만 남아도 전체 시퀀스의 노이즈를 제거해야 해 비효율적이고, 원리적 근거가 없어 생성 시퀀스가 어긋날 수 있다는 점을 이유로 든다. 이 주장에 대한 별도 수치는 Table 1에 없다.

미로 모델은 4.33M 매개변수, 프레임 묶음 10, DDIM 50단계를 쓰며 11GB 2080Ti 한 장으로 학습할 수 있다. MCG의 미래 표본 수와 추론 비용은 보고되지 않는다. [D.2·D.6·D.8·D.10·F.2](https://arxiv.org/pdf/2407.01392v4#page=25)

### 4.3 조절 가능한 조합적 생성 (Controllable Sequential Compositional Generation)

2D 정사각형의 한 모서리에서 반대 모서리로 가는 궤적들이 X 형태를 이루는 데이터를 학습한다. 기억을 모두 유지하면 학습 분포의 X 형태를 재현하고, 기억 없이 MPC로 짧은 계획을 반복하면 서로 다른 궤적의 일부를 이어 붙인 V 형태를 생성한다.

저자들의 요점은 모델을 다시 학습하지 않고 **샘플링 방식만 바꿔** 조합적 생성(compositional generation)의 정도를 조절한다는 것이다. 부록 E.2의 그림은 이전 상태를 버리면 마르코프 동역학을 복원한다고 설명한다. [§4.3·E.2·Figure 7](https://arxiv.org/pdf/2407.01392v4#page=27) **리뷰 해석.** 긴 문맥을 버리면 새로운 조합과 함께 데이터가 지키던 장기 제약도 사라지므로, 이 조절은 그런 제약을 버려도 되는 과제에서 쓸모가 있다.

### 4.4 로봇: 긴 모방 학습과 관찰 오염 대응 (Robotics)

Franka 로봇이 세 번째 빈 슬롯을 이용해 사과와 오렌지의 자리를 바꾼다. 과일의 초기 배치는 두 가지 중 무작위다. 한 과일이 세 번째 슬롯에 있는 중간 상태에서는 현재 영상만으로 다음에 옮길 과일을 정할 수 없어, 초기 배치를 기억(memory)해야 한다.

![원문 Figure 4: 같은 중간 상태의 서로 다른 목표와 두 카메라의 생성 영상](/assets/reviews/diffusion-forcing/robot-memory.webp)

[Figure 4 크게 보기](/assets/reviews/diffusion-forcing/robot-memory.webp)

**Figure 4를 읽는 법.** 왼쪽 Case A·B의 중간 배치는 같지만 아래 목표 배치는 다르다. 화살표를 현재 영상만으로 결정할 수 없다는 뜻이다. 오른쪽은 손목·외부 카메라의 조건 입력과 생성 영상이며, 120–480은 프레임 위치다. 오른쪽 영상 생성의 그럴듯함과 실제 행동 제어 성공률은 서로 다른 증거다. [Figure 4·§4.4, p. 9](https://arxiv.org/pdf/2407.01392v4#page=9)

학습에는 VR 원격 조작으로 모은 전문가 시연 150개(초기 배치별 절반씩), 두 카메라 영상, 6자유도 행동을 쓴다. 한 시연은 500–600개의 프레임·행동이며, 40개 영상 토큰으로 줄이고 각 토큰에 행동 15개를 묶는다. 원문의 결과는 다음과 같다.

| 조건·비교 | 보고한 성공률 | 조건 |
| --- | --- | --- |
| DF, 기본 관찰 | 80% | 기억이 필요한 과일 교환 작업 |
| DF, 관찰 오염 | 76% | 시각 방해물과 카메라 완전 가림. 방해물 가방은 학습 데이터에 없던 것(zero-shot)이며, 오염 관찰을 $k>0$으로 지정 |
| 다음 프레임 확산 기준선, 관찰 오염 | 48% | 오염 관찰을 정답으로 처리 |
| 기억 없는 Diffusion Policy, 기본 작업 | 실패한다고 보고 | 성공률 수치 없음 |

80→76은 **4%포인트 감소**이며, 원문은 시행 횟수와 신뢰구간을 보고하지 않는다. 오염 여부는 사람이 $k>0$으로 지정하고, 그러면 DF는 사전 모델에 크게 기대어 행동을 예측한다. 부록 E.5에 따르면 학습 데이터는 가방 없이 모았고, 관찰 오염 실험에서는 탁자에 던진 가방을 처음 보는 상태로 시험했다. 전형적인 실패는 로봇이 무작위로 놓인 물체 위치의 시각 단서에 더 이상 반응하지 않는 경우이며, 방해물 때문에 로봇이 이상하게 움직이는 모습은 관찰되지 않았다고 한다.

저자들은 같은 모델이 첫 프레임만으로 작업 영상도 생성할 수 있어, 확산 정책·모방 학습과 영상 생성 모델링을 하나로 묶고 무라벨 영상 사전학습(pre-training on unlabeled video)으로 가는 길을 연다고 쓴다. [§4.4·E.5·F.3](https://arxiv.org/pdf/2407.01392v4#page=33)

### 4.5 시계열 예측: 범용 시퀀스 모델로서의 성능 (Time Series Forecasting)

부록 E.1·F.4는 GluonTS의 여섯 데이터셋을 쓴다. 문맥 길이는 예측 길이와 같고, 예측 길이는 Exchange·Wikipedia 30, 나머지 24다. 차원은 각각 8·137·370·963·1214·2000이며, 모든 분할을 학습 분할의 특징별 평균·표준편차로 정규화한다. 검증 집합은 학습 시계열의 무작위 부분 시퀀스로 만들고, 6 epoch 동안 개선이 없으면 조기 종료한다. epoch당 100배치, 배치 32이며 모든 데이터셋에 같은 구조와 하이퍼파라미터를 쓴다.

지표는 **합산 연속 순위 확률 점수**(summed continuous ranked probability score, $\operatorname{CRPS}_{\mathrm{sum}}$)이며 **낮을수록 좋다**. CRPS는 예측 누적분포함수(CDF) $F$와 실제 관측 $x$에 대해 다음과 같다.

$$
\operatorname{CRPS}(F,x)=\int_{\mathbb R}\left(F(z)-\mathbb I\{x\leq z\}\right)^2\,dz.
$$

$\operatorname{CRPS}_{\mathrm{sum}}$은 100개 표본 궤적과 정답을 각각 특징 차원 방향으로 먼저 합하고, 합의 CDF를 19개 분위수로 추정해 이 점수를 구한 뒤 예측 구간에 걸쳐 평균한다. 차원별 CRPS를 더하는 것과 달리, 합이 같은 서로 다른 특징별 오차는 구별하지 못한다. 점수 계산은 GluonTS 구현을 쓰고, DF는 다섯 seed의 평균·표준편차를 보고한다.

아래 표는 Table 2의 비교 방법 12개 중 흐름·확산 기반 세 방법과, Electricity·Wikipedia에서 DF보다 낮은 NKF만 옮긴 것이다. 순위는 전통 통계·copula 모델을 포함한 전체 표 기준이다.

| 방법 | Exchange | Solar | Electricity | Traffic | Taxi | Wikipedia |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer-MAF | $0.005\pm0.003$ | $0.301\pm0.014$ | $0.021\pm0.000$ | $0.056\pm0.001$ | $0.179\pm0.002$ | $0.063\pm0.003$ |
| TimeGrad | $0.006\pm0.001$ | $0.287\pm0.020$ | $0.021\pm0.001$ | $0.044\pm0.006$ | $0.114\pm0.020$ | $0.049\pm0.002$ |
| ScoreGrad sub-VP SDE | $0.006\pm0.001$ | $0.256\pm0.015$ | $0.019\pm0.001$ | $0.041\pm0.004$ | $0.101\pm0.004$ | $0.043\pm0.002$ |
| NKF | — | $0.320\pm0.020$ | $0.016\pm0.001$ | $0.100\pm0.002$ | — | $0.071\pm0.002$ |
| DF | $0.003\pm0.001$ | $0.289\pm0.002$ | $0.023\pm0.001$ | $0.040\pm0.004$ | $0.075\pm0.002$ | $0.085\pm0.007$ |

부록 E.1은 DF가 ScoreGrad와는 전반적으로 비기고 나머지 기존 방법을 모두 앞서며, Wikipedia에서만 4위라고 쓴다. 표에서 DF가 가장 낮은 곳은 Exchange·Traffic·Taxi다. ScoreGrad 말고도 Solar에서는 TimeGrad(0.287)가 DF(0.289)보다, Electricity에서는 NKF(0.016)·Transformer-MAF·TimeGrad(0.021)가 DF(0.023)보다 낮다. Wikipedia에서는 DF보다 낮은 방법이 넷이라 5위다. [Table 2·E.1, p. 27](https://arxiv.org/pdf/2407.01392v4#page=27) **리뷰 해석.** 이 표는 본문 §4.5의 “경쟁력 있다”와 E.1의 “성능 손실 없이 여러 영역에 적용된다”는 수준의 주장은 뒷받침하지만, 기존 방법을 고르게 앞선다는 E.1의 문장은 지지하지 않는다.

## 5. 논의 (Discussion)

저자들은 결론에서 DF로 학습한 다음 토큰 예측 모델이 다음 토큰 모델과 전체 시퀀스 확산의 장점을 함께 갖고, 새 샘플링·유도 방식이 순차 의사결정 과제에서 큰 성능 향상을 낸다고 정리한다. 한계로는 현재 인과 구현이 RNN이라는 점, 고해상도 영상이나 복잡한 분포에는 부록 B.1을 따른 대형 Transformer가 필요할 것이라는 점, 인터넷 규모 데이터·과제로의 확장을 조사하지 않았다는 점을 든다. [§5, p. 10](https://arxiv.org/pdf/2407.01392v4#page=10)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 가장 강한 근거는 같은 모델 안의 비교다. 영상의 세 방식은 같은 RNN 구조를 쓰고, 미로의 MCG 절제는 샘플링만 바꾼다. 한 번의 학습으로 안정화 롤아웃, zig-zag 계획, 기억 길이 조절, 관찰 오염 지정을 모두 샘플링 선택으로 얻는다는 점이 이 논문의 실질적 기여다. 외부 기준선과의 비교는 조건이 갈린다. Diffuser\*는 PD 제어기와 다른 유도 목적을 쓰고, 지표인 episode reward는 저자 스스로 느린 계획에 유리하다고 지적하며, 시계열에서는 여섯 데이터셋 중 셋에서만 최저다. 노이즈 수준을 불확실성으로 읽는 직관도 보정된 확신도로 측정되지는 않았다.

남은 질문은 다음과 같다. 독립 노이즈·안정화·융합 SNR 가중치의 효과를 같은 예산에서 분리하면 각각 얼마인가? MCG 표본 수에 따라 점수와 지연은 어떻게 바뀌는가? 미로를 첫 도달 시간으로 평가해도 Diffuser\*와의 순서가 유지되는가? 로봇 실험에서 시행 횟수를 밝히고, 오염을 사람이 지정하는 조건과 자동으로 탐지하는 조건을 나누면 76%가 유지되는가?

## 출처와 읽은 범위

- 원문: *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion* · Boyuan Chen et al. · 최초 공개 2024-07-01 · [검토 버전](https://arxiv.org/abs/2407.01392v4) · 검토일 2026-10-23.
- 읽은 범위: v4 본문 §1–5와 실험·구현 부록 B–F. 부록 A의 ELBO·부분 시퀀스 주장은 [2/2] 이론 편에서 다룬다.
- 참고한 외부 원전: [DDPM v2](https://arxiv.org/abs/2006.11239v2) §2–3, [TimeGrad](https://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf) §2–3.1, [Diffuser](https://proceedings.mlr.press/v162/janner22a/janner22a.pdf) §2–3.1, [AR-Diffusion v1](https://arxiv.org/abs/2305.09515v1) §3.1–3.2, [D4RL 공식 구현](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py)의 `get_normalized_score`.
- 그림 출처: 대표 Figure 2와 본문 Figures 2–4·Table 1 상단 도식은 저자 v4 자산(CC BY 4.0). PDF 래스터화·WebP 변환으로 해상도·인코딩을 바꾸고 내용·색·화살표·기호를 유지한다. [원문](https://arxiv.org/abs/2407.01392v4).
- 확인하지 않은 것: 코드 실행·실험 재현, Table 1 점수의 정규화 여부(원문 미기재), Rolling Diffusion 원전.
