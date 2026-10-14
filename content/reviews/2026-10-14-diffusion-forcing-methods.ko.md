---
title: "[1] Diffusion Forcing: 시간과 노이즈를 따로 조절하면 무엇이 달라질까?"
description: "토큰마다 독립 노이즈를 학습하는 원리부터 장기 영상 생성, Monte Carlo Guidance, 미로 계획과 로봇 기억 실험까지 원문 순서로 읽는다."
date: "2026-10-14"
publishedAt: "2026-10-14T13:58:28+09:00"
updatedAt: "2026-10-14T13:58:28+09:00"
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

다음 프레임을 하나씩 예측하는 모델은 원하는 길이만큼 생성하기 쉽다. 하지만 작은 예측 오류가 누적되면 영상이 무너진다. 여러 프레임을 한꺼번에 diffusion으로 생성하면 미래 목표에 맞춰 결과를 유도하기 좋지만, 모든 프레임을 같은 속도로 확정하는 구조는 유연한 길이와 순차 제어에 제약을 준다. **Diffusion Forcing은 토큰마다 서로 다른 노이즈 수준을 학습해 두 방식 사이를 오갈 수 있게 한다.** 이 자유도를 이용해 장기 생성의 안정화와 미래 보상을 고려한 계획을 수행한다. 실험은 가능성을 보여주지만, 모든 환경의 성능 우위나 무한 길이의 정확성을 증명하지는 않는다. [원문 §1, pp. 1–2](https://arxiv.org/pdf/2407.01392v4#page=1)

35쪽의 v4를 두 편으로 나눴다. 이 글은 **본문 §1–5와 실험·구현에 필요한 부록 B–F**를 원문 순서로 설명한다. 이어지는 [[2] 이론과 증명 해설](/reviews/diffusion-forcing-theory/)은 부록 A의 ELBO와 부분 시퀀스 주장을 다룬다. 먼저 이 글을 읽고 이론 편으로 넘어가면 된다. 저자들의 정리에는 표기 불일치가 있어, 이론 편에서 원문 진술과 보충 유도를 구분한다.

## 1. 도입: 다음 토큰 예측과 전체 시퀀스 diffusion (Introduction)

여기서 **토큰**(token)은 단어에 한정되지 않는다. 영상 프레임, 시계열 관측 벡터, 로봇의 행동·관찰 묶음도 토큰이다.

**교사 강요**(teacher forcing)는 학습할 때 정답 과거를 주고 다음 토큰을 예측하게 한다. 추론에서는 정답 대신 모델의 출력을 다음 입력으로 사용한다. 연속 데이터에서는 이 작은 차이가 반복되면서 오차 누적(compounding error)을 일으킬 수 있다.

**전체 시퀀스 확산**(full-sequence diffusion)은 여러 토큰을 묶어 노이즈를 넣고 제거한다. 일반적인 구성에서는 같은 diffusion 단계의 모든 토큰이 동일한 노이즈 수준을 갖는다. 원하는 미래에 가까워지도록 샘플링을 바꾸는 **유도**(guidance)에 유리하다.

저자들은 “시간상 앞에 있는 토큰부터 완전히 결정해야 한다”와 “모든 토큰의 노이즈 수준이 같아야 한다”를 동시에 완화한다. 가까운 미래는 더 먼저 결정하고 먼 미래는 여러 가능성을 남겨둘 수 있다. 학습 때 토큰별 노이즈를 독립적으로 뽑았기 때문에, 이 선택은 모델 재학습보다 **샘플링 계획**의 변경으로 이루어진다. [§1, Figure 1–2](https://arxiv.org/pdf/2407.01392v4#page=2)

## 2. 관련 연구와 배경 (Related Work and Preliminaries)

이 논문에는 두 시간축이 있다. 아래 첨자 $t$는 시퀀스의 시간이고, 위 첨자 $k$는 노이즈 수준이다. $\mathbf x_t^{k_t}$는 “시간 $t$의 토큰을 노이즈 수준 $k_t$로 오염시킨 것”이다. $\mathbf z_t$는 과거 정보를 요약한 잠재 상태(latent state)다.

### 베이지안 필터링과 diffusion의 연결

**베이지안 필터링**(Bayesian filtering)은 새 관찰을 이용해 과거의 믿음을 갱신한다. 여기서 잠재 상태는 반드시 물리적 위치를 뜻하지 않는다. 미래를 예측하는 데 필요한 과거의 요약이다.

Diffusion의 전방 과정(forward process)은 다음과 같다. 원문 식 (2.1)의 표기를 사용한다.

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

두 번째 식은 [DDPM 식 (4)](https://arxiv.org/pdf/2006.11239v2#page=2)에 따라 정규화된 노이즈로 쓴 형태다. $k_t=0$이면 원래 토큰이고, 큰 $K$에서 충분히 오염된 토큰은 거의 순수 노이즈다. 유한 $K$에서 원래 정보가 정확히 0이 된다는 등식과는 구분한다. 원문 배경 절의 혼용된 $t/k$ 첨자는 이 글에서 노이즈축 $k$로 맞췄다.

### 가까운 선행 접근들은 무엇이 다른가?

| 접근 | 무엇을 학습하는가? | Diffusion Forcing과의 차이 |
| --- | --- | --- |
| [DDPM](https://arxiv.org/pdf/2006.11239v2), §2–3 | 한 데이터의 전방·역방향 diffusion과 denoising 손실 | DF는 이 구성에 시간별 기억과 독립적인 토큰별 노이즈를 결합한다. |
| [TimeGrad](https://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf), §3 | 정답 과거를 요약한 RNN 상태에 조건부로 다음 시계열 벡터를 diffusion 생성 | DF는 과거도 여러 노이즈 수준으로 학습하고 여러 미래 토큰을 함께 유도할 수 있다. |
| [Diffuser](https://proceedings.mlr.press/v162/janner22a/janner22a.pdf), §2–3.1 | 상태·행동 궤적 전체를 공동 생성하고 목표에 맞춰 유도 | DF는 인과적 구조와 토큰별 스케줄로 미래의 불확실성과 계획 길이를 조절한다. |
| [AR-Diffusion](https://arxiv.org/pdf/2305.09515v1), §3.1–3.2 | 문장 단계와 단어 위치로 토큰별 diffusion 단계를 정함 | 서로 다른 노이즈 수준이라는 선행 아이디어가 있다. DF의 핵심 차이는 위치에 묶이지 않은 독립 noise 학습이다. |

이 표는 개념 비교다. TimeGrad는 시계열 표, Diffuser는 미로 실험의 비교 대상이지만, AR-Diffusion과의 텍스트 성능 비교는 수행하지 않는다. 원문 부록 C는 Rolling Diffusion과도 비교한다. 해당 연구 전체를 이번 글에서 독립 검토한 것은 아니다. 또 Diffuser 원전 §3.1도 fully convolutional architecture의 입력 길이를 바꾸어 계획 horizon을 조절할 수 있다고 명시한다. 따라서 DF를 가변 길이 계획의 최초 방법으로 소개하지 않는다. DF의 차이는 여러 noise·conditioning pattern과 인과적 기억을 같은 학습 틀에서 다룬다는 데 있다.

## 3. 방법 (Method)

### 3.1 노이즈를 부분 마스킹으로 보기 (Noising as partial masking)

**부분 마스킹**(partial masking)은 정보를 완전히 지우거나 남기는 두 선택 사이에 여러 정도를 두는 관점이다. 영상에 조금 노이즈를 넣으면 장면의 큰 구조는 남지만 작은 정보는 불확실해진다. 더 많이 넣으면 원래 장면을 알아보기 어려워진다.

DF는 토큰을 가리는 정도와 시퀀스의 위치를 각각 조절한다. 노이즈 수준은 관찰을 얼마나 신뢰할지 조절하는 손잡이로 해석할 수 있다. 다만 모델의 노이즈 수준이 자동으로 보정된 불확실성(calibrated uncertainty) 점수라는 뜻은 아니다. [§3.1, p. 4](https://arxiv.org/pdf/2407.01392v4#page=4)

![원문 Figure 2: 시간축과 노이즈축 및 세 가지 학습·샘플링 구성](/assets/reviews/diffusion-forcing/overview.webp)

[Figure 2 크게 보기](/assets/reviews/diffusion-forcing/overview.webp)

**Figure 2를 읽는 법.** 왼쪽의 가로축은 시간, 세로축은 노이즈다. 가운데 DF의 학습 행에서는 각 토큰의 색이 서로 다르다. 같은 시퀀스 안에서도 오염 정도를 다르게 뽑기 때문이다. 오른쪽 teacher forcing은 깨끗한 과거를 기준으로 다음 토큰을 예측하고, full-sequence diffusion은 토큰들을 같은 노이즈 단계에서 처리한다. 아래 sampling 행에서는 DF가 시간 방향의 연결을 유지하면서 각 열을 서로 다른 속도로 denoise하는 모습을 볼 수 있다. 굵은 가로 막대는 전체 시퀀스를 함께 처리하는 비교 구조다. 색은 노이즈 수준을 보여주며, 성공률이나 모델의 confidence가 아니다. [원문 Figure 2, p. 3](https://arxiv.org/pdf/2407.01392v4#page=3)

### 3.2 토큰마다 다른 노이즈 수준 (Different noise levels for different tokens)

논문의 구현은 인과적 순환 신경망(causal recurrent neural network, RNN)이다. 학습 시 $k_1,\ldots,k_T$를 독립적으로 뽑아 토큰마다 다른 노이즈를 넣는다. 잠재 상태는 다음처럼 갱신된다.

$$
\mathbf z_t\sim p_\theta\!\left(
\mathbf z_t\mid\mathbf z_{t-1},
\mathbf x_t^{k_t},k_t\right).
$$

구현의 $\mathbf z_t$는 결정론적이다. 분포 표기는 개념적인 모델을 설명하며, 실제로 매번 잠재 상태를 무작위로 뽑는다는 뜻은 아니다. $k_t=0$은 관찰을 반영한 사후 갱신(posterior update), $k_t=K$는 관찰 정보가 거의 없는 사전 예측(prior prediction)에 대응한다.

학습 손실은 원문 식 (3.1)의 핵심을 유지하면 다음과 같다.

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

중요한 차이는 손실이 MSE라는 점보다 **조건으로 주는 과거도 noisy하다는 점**이다. 노이즈는 현재 토큰만의 일이 아니다. 잠재 상태가 noisy history를 요약하며, 손실의 gradient는 이 기억의 갱신을 통해서도 흐른다. Algorithm 1은 시퀀스 샘플링→독립 noise 샘플링→토큰 오염→latent와 예측 계산→시퀀스 손실 역전파의 순서다. 원문은 noise 집합을 $[K]$와 $\{0,\ldots,K\}$로 혼용한다. 깨끗한 끝점에서는 노이즈를 복원하는 나눗셈 식을 그대로 평가하면 안 된다. 실제 구현의 parameterization과 끝점 처리는 별도로 확인해야 한다. [§3.2·Algorithm 1, pp. 4–5](https://arxiv.org/pdf/2407.01392v4#page=4)

실험은 이 설명용 $\boldsymbol\epsilon$ 예측 하나로 통일되지 않는다. 부록 D.3에서는 영상·시계열에 $\mathbf v$ 예측, 계획·모방 학습에 $\mathbf x_0$ 예측을 사용한다. 영상에는 과거의 정보까지 고려한 **융합 SNR 가중치**(fused SNR reweighting)도 적용한다. 비교를 판단할 때 이 학습 세부를 독립 noise 자체와 구분해야 한다.

**샘플링은 두 차원의 스케줄을 고르는 문제다.** $\mathcal K_{m,t}$는 sampling 행 $m$에서 시간 $t$ 토큰에 요구하는 노이즈 수준이다. Algorithm 2는 행을 따라 진행하고 각 행 안에서는 과거에서 미래로 latent를 갱신하며 denoise한다. 마지막 행에서는 모든 토큰을 깨끗하게 만든다.

설명용으로 $K$가 짝수인 세 토큰을 보면 다음과 같다.

$$
\begin{aligned}
(K,K,K)
&\longrightarrow(0,K/2,K)\\
&\longrightarrow(0,0,K/2)\\
&\longrightarrow(0,0,0).
\end{aligned}
$$

각 괄호는 토큰 값이 아니라 **노이즈 수준들의 목록**이다. 먼 미래를 덜 확정한 상태에서 가까운 미래를 먼저 확정하는 직관을 보여준다. 실제 DDIM 구현의 모든 중간 단계를 적은 알고리즘은 아니다. 부록 D.7은 이를 pyramid scheduling으로 구체화한다. [§3.2–3.3·D.7](https://arxiv.org/pdf/2407.01392v4#page=5)

본문 Theorem 3.1은 학습이 증거 하한(evidence lower bound, ELBO)의 재가중 목적과 연결되고, 적절한 조건에서 여러 noise pattern을 동시에 모델링한다고 주장한다. 충분한 표현력 등의 조건과 부록 수식의 불일치는 [이론 편](/reviews/diffusion-forcing-theory/)에서 자세히 다룬다. 이 주장을 유한 RNN의 모든 부분 시퀀스 정확도 보장으로 읽으면 안 된다.

### 3.3 새로운 시퀀스 생성 능력 (New Capabilities in Sequence Generation)

**장기 생성 안정화.** 생성한 프레임을 완벽한 정답 과거로 취급하지 않고, 약간 noisy한 과거와 연결된 latent를 다음 생성에 사용한다. 부록 B.4의 RNN 구현은 작은 $k_{\mathrm{small}}>0$에 대응하는 latent를 이어간다. 모델이 학습 때 경험한 noisy history로 추론의 오차를 흡수하려는 방식이다. 매번 latent를 초기화하거나 sliding window를 이동하는 방식과 다르다.

**먼 미래를 불확실하게 남기기.** 가까운 토큰의 noise를 먼저 줄이는 zig-zag/pyramid schedule은 먼 미래를 한 가지 경로로 너무 일찍 고정하지 않는다. 가능한 미래를 보면서 지금의 선택을 바꿀 여지를 남긴다.

**장기 guidance.** 인과적 생성에서 미래 토큰은 과거에 의존한다. 따라서 미래 목표에 대한 gradient를 이 의존 경로로 역전파할 수 있다. 아직 결정 중인 과거 토큰을 미래의 목표로 유도한다. 계산의 gradient가 시간상 뒤로 흐르는 것과, 생성 모델이 미래 관측을 몰래 입력으로 받는 것은 다른 개념이다. [§3.3·B.3–B.5](https://arxiv.org/pdf/2407.01392v4#page=6)

### 3.4 유연한 순차 의사결정 (Flexible Sequential Decision Making)

행동 $\mathbf a_t$, 보상 $\mathbf r_t$, 다음 관찰 $\mathbf o_{t+1}$을 한 토큰으로 묶는다.

$$
\mathbf x_t=
[\mathbf a_t,\mathbf r_t,\mathbf o_{t+1}]^\top.
$$

과거를 요약한 $\mathbf z_{t-1}$에서 미래 계획 $\hat{\mathbf x}_{t:t+H}$를 생성하고 첫 행동 $\hat{\mathbf a}_t$만 실행한다. 다음에는 **예측한 보상·관찰 대신 환경에서 받은 실제 값**으로 latent를 갱신한다. 모델 예측 제어(model predictive control, MPC)처럼 반복해서 다시 계획한다.

짧게 내다보면 정책(policy)처럼 빠르게 행동하고, 길게 내다보면 planner처럼 미래 목표를 고려한다. $H$는 원문의 look-ahead 표기이며 구간 $t:t+H$는 양 끝을 포함한다. 실제 latency 절감률을 측정한 표는 없다.

<strong>Monte Carlo Guidance(MCG)</strong>는 현재 토큰을 고정된 조건으로 삼아 여러 미래를 샘플링하고, 각각의 미래 보상에 대한 gradient를 평균한다. 아래는 원문 설명을 적은 **리뷰 보충 표기**다. $N$은 미래 샘플 수, $\widehat G^{(i)}$는 $i$번째 샘플의 추정 보상이다.

$$
\widehat{\mathbf g}
=\frac1N\sum_{i=1}^{N}
\nabla_{\mathbf x_t^k}\widehat G^{(i)}.
$$

잘 나온 미래 하나를 고르는 것과 기대 보상(expected reward)을 여러 미래로 추정하는 것은 다르다. 먼 미래를 noisy하게 유지할수록 다양한 결과를 바탕으로 가까운 선택을 유도하기 쉽다. gradient와 기대값의 교환에는 미분·적분의 정당화가 필요하고, 실험에서 모델·sampling의 오차까지 제거되는 것은 아니다. 부록 B.5는 통상적인 동시 denoising에서는 이런 조건부 미래 rollout이 쉽지 않다고 설명한다. 이를 “어떤 full-sequence 모델도 MCG가 수학적으로 불가능하다”는 보편적 정리로 확대하지 않는다. [§3.4·B.3·B.5](https://arxiv.org/pdf/2407.01392v4#page=6)

## 4. 실험 (Experiments)

### 4.1 영상 예측: 일관성·안정성·긴 rollout (Video Prediction)

Minecraft와 DMLab의 1인칭 이동 영상을 사용한다. DF, teacher-forced next-frame diffusion, causal full-sequence diffusion은 **같은 RNN architecture**를 사용하는 비교다. 일반적인 비인과적 대형 video diffusion 모델 전체와의 비교가 아니다.

![원문 Figure 3: DMLab과 Minecraft에서 세 학습 방식의 장기 영상 생성 비교](/assets/reviews/diffusion-forcing/video-rollouts.webp)

[Figure 3 크게 보기](/assets/reviews/diffusion-forcing/video-rollouts.webp)

**Figure 3를 읽는 법.** 왼쪽은 DMLab, 오른쪽은 Minecraft다. 빨간 테두리는 조건으로 준 실제 입력이고 이후는 생성 결과다. 가로 방향의 496·500·996·1000은 프레임 위치다. 위 DF 행은 긴 rollout에서도 장면 구조를 유지하는 사례를 보여준다. 가운데의 causal full-sequence baseline은 장면이 급격히 바뀌고, 아래 teacher forcing은 후기 프레임이 패턴이나 잡음으로 무너진다. 두 Seq 행은 서로 다른 생성 사례이며 오차 막대나 평균 지표가 아니다. 정지 프레임만으로 모든 시간적 일관성을 평가할 수는 없다. [Figure 3·§4.1, pp. 7–8](https://arxiv.org/pdf/2407.01392v4#page=7)

부록 E.3에서는 180 프레임의 추가 사례와 2000 프레임까지 실행했을 때 붕괴하지 않았다는 관찰을 보고한다. 이는 **관측한 긴 rollout**이지, 무한 길이의 품질 보장이나 미래 상태의 정확도 증명은 아니다. F.1의 해상도는 Minecraft 128×128, DMLab 64×64이며, 전체 데이터의 약 10%만 학습한다. Minecraft의 학습 clip은 72 프레임이다. DMLab은 E.3 캡션에서 36, F.1에서는 48 프레임으로 달라 정확한 길이는 원문 내부 불일치로 남긴다.

영상 모델은 Minecraft 36M·DMLab 24M parameters다. $K=1000$, 추론은 100 DDIM steps, frame stacking은 각각 8·4다. 학습은 8 A100, batch 128, 50K steps이며 약 40K steps에서 12시간 정도의 수렴을 보고한다. FVD 등 전체 영상 품질의 정량 비교는 이 실험의 주 근거로 제시되지 않는다. [D.2·D.6·D.10·E.3·F.1](https://arxiv.org/pdf/2407.01392v4#page=24)

### 4.2 Diffusion 계획: MCG·인과적 불확실성·유연한 길이 (Diffusion Planning)

D4RL Maze2D의 U-Maze·Medium·Large에서 무작위 보행 로그로 미로별 모델을 학습한다. 관찰은 2D 위치·속도의 4차원 벡터, 행동은 2D 가속도다. 고정 목표인 Maze2D와 목표도 무작위인 Multi2D를 평가한다.

![원문 Table 1 위 도식: DF와 Diffuser가 미로 계획을 denoise하는 과정](/assets/reviews/diffusion-forcing/maze-planning.webp)

[계획 도식 크게 보기](/assets/reviews/diffusion-forcing/maze-planning.webp)

**계획 도식을 읽는 법.** 두 미로에서 위 행은 DF, 아래는 Diffuser다. 가로 방향은 환경의 실행 시간이 아니라 **denoising의 진행**이다. 색 띠의 start→end는 계획 안에서의 순서다. DF는 가까운 구간을 먼저 정하고 먼 구간의 후보를 남기면서 경로를 형성한다. 점들이 퍼져 있다는 것은 sampling 중의 미확정 상태이며 실제 로봇이 모든 점을 지나갔다는 뜻이 아니다. 이 도식만으로 성공률을 판단하지 않고 아래 Table 1과 함께 읽어야 한다.

원문 Table 1의 모든 비교 모델을 유지해 두 표로 나눴다. 값은 보고된 benchmark score이며 **성공 확률(%)이 아니다**. 100을 넘는 값도 가능하다. $\pm$는 원문 표기를 유지했지만 해당 표의 반복 수와 오차 정의는 명확히 보고되지 않는다.

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

별표(\*)가 중요하다. <strong>Diffuser*</strong>는 생성한 상태 경로를 따라갈 행동을 수작업 PD controller로 만든 비교다. 생성한 행동을 직접 실행한 열은 성능이 훨씬 낮다. DF는 생성한 행동을 실행한다. 상태 경로와 실행 행동의 관계를 고려하는 설계의 근거지만, 수작업 controller 유무가 다른 비교라는 조건도 유지해야 한다. [Table 1·§4.2, p. 8](https://arxiv.org/pdf/2407.01392v4#page=8)

MCG를 넣으면 여섯 환경 모두에서 **평균 점수**가 올라간다. 다만 이것만으로 각각의 통계적 유의성을 증명한 것은 아니다. Single-task 평균은 DF 141.7 대 Diffuser\* 119.5, multi-task는 146.2 대 129.4다. **Multi2D U-Maze에서는 DF 119.1이 Diffuser\* 128.9보다 낮다.** 따라서 본문의 “여섯 환경 모두에서 우세”라는 문장은 표 전체와 일치하지 않는다.

부록 D.8도 reward가 좋은 planner를 완전히 평가하지 못한다고 경고한다. 목표 근처에 오래 머무는 느린 경로가 높은 점수를 얻을 수 있다. 처음 목표에 도달하는 시간(first hitting time)을 함께 봐야 하지만, 그 지표의 완전한 수치 비교는 제공되지 않는다.

Maze 모델은 4.33M parameters, frame stack 10, 50 DDIM sampling steps를 사용한다. 학습은 2080Ti 11GB에서 가능한 설정으로 보고한다. MCG 미래 샘플 수·동일 추론 예산의 정밀 비교가 부족하므로, 성능 차이를 계산 비용이 같은 상태의 순수 알고리즘 우위로 단정하지 않는다. [D.2·D.6·D.8·D.10·F.2](https://arxiv.org/pdf/2407.01392v4#page=25)

### 4.3 조절 가능한 조합적 생성 (Controllable Sequential Compositional Generation)

2D 평면에서 반대 모서리로 향하는 두 대각선 궤적을 학습한다. 전체 기억을 유지하면 학습 분포의 X 형태를 재현한다. 짧은 계획을 만들고 이전 기억을 버리면 서로 다른 궤적의 부분을 이어 붙인 V 형태도 생성한다.

이 사례의 핵심은 모델을 다시 학습하지 않고 **기억과 sampling 범위**로 조합적 생성(compositional generation)의 정도를 바꾼다는 점이다. 실제 세계에서 아무 부분 궤적이나 안전하게 합칠 수 있다는 보장은 아니다. 긴 문맥의 제약을 지우면 새로운 조합과 함께 중요한 제약도 잃을 수 있다. [§4.3·E.2·Figure 7](https://arxiv.org/pdf/2407.01392v4#page=27)

### 4.4 로봇: 긴 모방 학습과 관찰 오염 대응 (Robotics)

Franka 로봇이 사과와 오렌지의 자리를 바꾸되 세 번째 빈 슬롯을 사용한다. 두 초기 배치를 무작위로 정한다. 중간에는 현재 영상이 같아도 초기 배치에 따라 다음에 옮길 과일이 달라질 수 있다. 그래서 기억(memory)이 필요하다.

![원문 Figure 4: 같은 중간 상태의 서로 다른 목표와 두 카메라의 생성 영상](/assets/reviews/diffusion-forcing/robot-memory.webp)

[Figure 4 크게 보기](/assets/reviews/diffusion-forcing/robot-memory.webp)

**Figure 4를 읽는 법.** 왼쪽 Case A·B의 중간 배치는 같지만 아래 목표 배치는 다르다. 화살표를 현재 영상만으로 결정할 수 없다는 뜻이다. 오른쪽은 손목·외부 카메라의 조건 입력과 생성 영상이며, 120–480은 프레임 위치다. 오른쪽 영상 생성의 그럴듯함과 실제 행동 제어 성공률은 서로 다른 증거다. [Figure 4·§4.4, p. 9](https://arxiv.org/pdf/2407.01392v4#page=9)

150개 expert demonstrations, 두 카메라, 6-DoF 행동을 사용한다. 한 시연은 500–600 frames/actions이며 40개 영상 토큰으로 줄이고 각 토큰에 15개 행동을 묶는다. 원문의 결과는 다음과 같다.

| 조건·비교 | 보고한 성공률 | 해석 범위 |
| --- | --- | --- |
| DF, 기본 관찰 | 80% | 기억이 필요한 특정 fruit-swap 작업 |
| DF, 관찰 오염 | 76% | 외부 방해물·카메라 가림, 오염을 noisy 입력으로 지정 |
| Next-frame diffusion baseline, 관찰 오염 | 48% | 오염 관찰을 정답처럼 처리하는 비교 |
| 기억 없는 Diffusion Policy, 기본 작업 | 실패한다고 보고 | 본문에 정확한 성공률 수치 없음 |

80→76은 **4%포인트 감소**다. 시험 횟수와 신뢰구간은 명확히 보고되지 않아 150개 학습 시연을 평가 시행 수로 사용하면 안 된다. 오염 관찰에 $k>0$을 지정하는 인터페이스가 있으며, 오염을 자동 탐지하는 시스템의 평가로 읽지 않는다. 초기 배치 정보를 놓치면 기억만으로 올바른 목표를 회복하기 어렵다는 실패도 부록 E.5에서 설명한다.

같은 모델의 영상 생성은 unlabeled video pre-training의 가능성을 제시한다. 실제로 대규모 무라벨 영상 사전학습의 이득을 비교한 결과는 아니다. [§4.4·E.5·F.3](https://arxiv.org/pdf/2407.01392v4#page=33)

### 4.5 시계열 예측: 일반적인 시퀀스 모델로서의 성능 (Time Series Forecasting)

부록 E.1·F.4는 GluonTS의 여섯 데이터셋을 사용한다. context 길이는 prediction 길이와 같고, 예측 길이는 Exchange·Wikipedia 30, 나머지 24다. 차원은 각각 8·137·370·963·1214·2000이다.

지표는 **합산 연속 순위 확률 점수**(summed continuous ranked probability score, $\operatorname{CRPS}_{\mathrm{sum}}$)다. 특징 차원별 값을 합친 예측 분포와 실제 합을 비교하며 **낮을수록 좋다**. 모든 개별 feature의 정확도나 의존 구조를 이 값 하나로 완전히 평가하지는 않는다. 100개 sampled trajectories와 19 quantiles를 사용하고, DF는 다섯 seed의 평균·표준편차를 보고한다.

아래는 Table 2의 주요 비교와 Electricity·Wikipedia 해석에 필요한 NKF 행이다. 전통 통계·다른 copula 모델의 행은 원문에 있으며, 전체 값은 별도의 최저값·순위 검산에도 사용했다.

| 방법 | Exchange | Solar | Electricity | Traffic | Taxi | Wikipedia |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer-MAF | $0.005\pm0.003$ | $0.301\pm0.014$ | $0.021\pm0.000$ | $0.056\pm0.001$ | $0.179\pm0.002$ | $0.063\pm0.003$ |
| TimeGrad | $0.006\pm0.001$ | $0.287\pm0.020$ | $0.021\pm0.001$ | $0.044\pm0.006$ | $0.114\pm0.020$ | $0.049\pm0.002$ |
| ScoreGrad sub-VP SDE | $0.006\pm0.001$ | $0.256\pm0.015$ | $0.019\pm0.001$ | $0.041\pm0.004$ | $0.101\pm0.004$ | $0.043\pm0.002$ |
| NKF | — | $0.320\pm0.020$ | $0.016\pm0.001$ | $0.100\pm0.002$ | — | $0.071\pm0.002$ |
| DF | $0.003\pm0.001$ | $0.289\pm0.002$ | $0.023\pm0.001$ | $0.040\pm0.004$ | $0.075\pm0.002$ | $0.085\pm0.007$ |


원문 표와 대조하면 **DF의 평균값은 Exchange·Traffic·Taxi에서 가장 낮지만 Solar·Electricity·Wikipedia에서는 최저가 아니다.** Electricity에서는 NKF 0.016이 원문에서 강조한 ScoreGrad 0.019보다도 낮다. Wikipedia에서는 네 모델의 평균이 DF보다 낮아 전체 표 기준 다섯 번째다. 부록의 “네 번째”라는 설명과 다르다. 오차 범위의 겹침과 비교 모델별 학습 조건을 고려해야 하며, 평균 순위로 통계적 유의성을 대신하지 않는다. [Table 2·E.1, p. 27](https://arxiv.org/pdf/2407.01392v4#page=27)

검증 집합은 학습 시계열의 무작위 subsequences에서 만들고, 6 epochs 동안 개선이 없으면 조기 종료한다. epoch당 100 batches, batch 32다. “새 목적이 다양한 예측 문제에도 작동한다”는 결론은 타당한 해석이지만, 기존 모든 시계열 모델에 대한 우위는 표가 뒷받침하지 않는다.

## 5. 논의: 무엇을 일반화할 수 있을까? (Discussion)

논문이 보여준 것은 독립적인 토큰별 noise 학습이 **추론할 때 선택할 수 있는 conditioning·기억·길이·guidance의 범위**를 넓힌다는 점이다. 장기 영상 생성과 MCG 계획, 기억 기반 로봇 제어는 이 자유도의 서로 다른 활용이다.

저자들은 현재 RNN 구현, 고해상도·복잡한 분포의 확장, internet-scale 데이터에서 미검토한 scaling을 한계로 밝힌다. Transformer 적용은 부록 B.1의 확장 논의이며 본문의 RNN 실험을 대체하는 대규모 검증이 아니다. [§5, p. 10](https://arxiv.org/pdf/2407.01392v4#page=10)

## 리뷰 해석과 남은 질문

이 연구의 장점은 새로운 목표마다 모델을 다시 만들기보다 sampling 구조를 바꿀 여지를 준다는 데 있다. 하지만 noise 수준을 uncertainty로 사용하는 직관, 실제 confidence의 보정, 장기 상태 예측의 정확성은 별개다.

후속 검증에서는 독립 noise·stabilization·fused SNR의 효과를 같은 예산에서 분리하고, MCG 샘플 수에 따른 성공·latency 곡선을 비교하면 좋다. 미로에서는 reward와 첫 도달 시간을 함께 측정하고, 로봇에서는 오염 정보를 직접 지정하는 조건과 자동 탐지 조건을 분리할 필요가 있다. 이는 리뷰의 제안이며 수행한 실험이 아니다.

## 출처와 읽은 범위

- 원문: Chen, Marti Monso, Du, Simchowitz, Tedrake, Sitzmann, *Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion*, [arXiv:2407.01392v4](https://arxiv.org/abs/2407.01392v4). 최초 제출 2024-07-01, v4 수정 2024-12-10, 검토 2026-10-14.
- 첫 읽기: 초록·도입·Discussion, 본문과 부록의 절 지도, Figure 1–4·Table 1 캡션, 부록 A의 개관과 정리 진술. 개관을 독립 저장한 뒤 두 번째 읽기를 시작했다.
- 두 번째 읽기: v4 본문 §1–5와 부록 A–F의 텍스트·식·표·알고리즘을 저자 LaTeX와 PDF로 대조했다. 핵심 방법·이론·계획 표의 페이지를 렌더링했다. 참고문헌 목록을 확인하고 관련 원전은 필요한 절만 추가로 읽었다.
- 원전 추가 읽기: DDPM v2 §2–3.4, TimeGrad §2–3.1, Diffuser §2–3.1, AR-Diffusion v1 §3.1–3.2, Chan tutorial v1 §2.6–2.7. 각 원전 전체 리뷰나 최신 후속 연구 서베이는 수행하지 않았다. B.4가 참조하는 HINT의 정리·적용 조건은 독립 검증하지 않아 안정성 인증의 근거로 사용하지 않았다.
- 구성: 본문 §1–5·§4.1–4.5 순서를 유지했다. B·D·E·F의 필수 구현·실험 조건을 해당 본문 절에 연결했다. 부록 A의 완결된 해설은 [2편](/reviews/diffusion-forcing-theory/)에 있다. 주변 related work·유사한 추가 영상·전체 참고문헌 재게시를 줄였으며 핵심 비교·반대 결과·비용은 유지했다.
- 그림: Figure 2·3·4와 Table 1의 위 도식은 [저자 v4 소스](https://arxiv.org/src/2407.01392v4)의 원본 PDF 그림을 래스터화하고 WebP로 인코딩했다. 내용·색·화살표·기호는 유지했고 해상도·인코딩은 변경했다. 논문과 그림의 원저자는 위 저자들이며 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)에 따라 출처와 변경 사항을 표시한다.
- 검증: 표의 전사·평균·순위·단위와 ELBO 관련 국소 계산을 확인했다. **미실행:** 모델/GPU/로봇 재현, 공개 코드 실행, 영상 전체 정량 품질 측정, 원시 결과 기반 유의성 검정, 실제 latency·메모리 측정, 전체 정리의 형식 검증.
