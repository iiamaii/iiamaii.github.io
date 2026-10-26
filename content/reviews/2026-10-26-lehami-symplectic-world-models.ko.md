---
title: "[1/2] LeHami: 잠재 공간의 부피를 보존하는 월드 모델"
description: "심플렉틱 예측기가 JEPA의 장기 수축을 막는 원리와 진자의 속도 예측·시간 간격 변경·목표 도달 실험을 살펴본다."
date: "2026-10-26"
publishedAt: "2026-10-26T13:56:31+09:00"
updatedAt: "2026-10-26T13:56:31+09:00"
topics: ["world models", "representation learning", "Hamiltonian dynamics", "geometric integration", "model predictive control"]
visibility: "public"
lang: "ko"
translationKey: "lehami-symplectic-world-models"
paperTitle: "LeHami: Symplectic Latent Dynamics for World Models"
paperPublishedDate: "2026-10-11"
authors: ["Romina Yalovetzky", "Randall Balestriero", "Yann LeCun"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.15267v1"
thumbnail: "/assets/reviews/lehami/lehami-fig1-concept.webp"
thumbnailAlt: "LeHami 원문 Figure 1: 행동 조건부 Hamiltonian으로 연결한 잠재 위상 공간과 부피 보존·시간 간격 변경·역방향 예측."
---

LeHami는 공동 임베딩 예측 구조(Joint-Embedding Predictive Architecture, JEPA)의 예측기를 심플렉틱(symplectic) 흐름으로 바꿔, 예측을 반복해도 잠재 위상 공간(latent phase space)의 부피가 보존되게 한다. 구동 진자의 초기 상태를 읽는 25-최근접 이웃(25-nearest neighbors, 25-NN) 프로브에서 각속도 평균 절대 오차(mean absolute error, MAE)는 LeWorldModel(LeWM)의 2.508에서 LeHami의 0.419 rad/s로 낮지만, 같은 표의 각도 오차는 LeWM이 더 작다. [Table 1](https://arxiv.org/html/2610.15267v1#S3.T1) **리뷰 해석.** 이 연구는 잠재 예측에 기하학적 제약을 넣는 설득력 있는 사례이며, 실제 이득은 속도 표현과 일부 장기 예측에서 확인되고 목표 도달 성능은 더 강한 시간 인코더를 쓴 기준선에 미치지 못한다.

이 편은 원문 §1–4의 방법과 주요 실험을 따르며, [\[2/2\] 증명·설계·절제 실험](/reviews/lehami-symplectic-world-models-technical/)은 부록 B–E를 해설한다.

## 1. 도입 (Introduction)

JEPA는 영상을 다시 그리는 대신 미래 영상의 표현을 예측한다. 인코더(encoder)가 모든 영상을 같은 벡터로 보내는 **표현 붕괴**(representation collapse)는 분포 정규화나 사전학습 인코더로 억제할 수 있다. 그러나 인코더가 구별한 상태도 예측기를 여러 번 통과하면서 점점 가까워질 수 있다. 논문은 이 **예측기 수축**(predictor contraction)을 별도의 문제로 다룬다. 매 스텝에서 조금씩 줄어드는 변형도 자기회귀적 롤아웃(autoregressive rollout)에서는 누적된다. [§1, pp.1–3](https://arxiv.org/pdf/2610.15267v1#page=1)

저자들의 해법은 위치와 운동량을 짝지은 해밀턴 역학(Hamiltonian mechanics)에서 온다. 잠재 상태를 위치 유사 블록(position-like block) $\mathbf q$와 운동량 유사 블록(momentum-like block) $\mathbf p$로 나누고, 스칼라 함수 $H_\phi$의 미분으로 상태를 움직인다. 심플렉틱 적분은 이 공간의 방향 있는 면적 구조와 부피를 보존한다. 점 구름을 늘이거나 비틀 수는 있어도 전체 부피를 줄이지 못하게 하는 제약이다.

[![LeHami의 잠재 위상 공간, 행동 입력과 세 구조적 성질.](/assets/reviews/lehami/lehami-fig1-concept.webp)](/assets/reviews/lehami/lehami-fig1-concept.webp)

원문 Figure 1. 왼쪽은 토크가 달라지는 영상과 잠재 궤적, 오른쪽은 부피 보존·관찰 프레임 사이의 적분·역방향 적분이다. 등고선은 행동을 고정했을 때의 $H_\phi$ 값이다. [Fig.1](https://arxiv.org/html/2610.15267v1#S1.F1)

논문이 내세우는 기여는 세 가지다. 첫째, 픽셀로부터 종단간(end-to-end) 학습하면서 매 예측 스텝의 부피 보존을 보장한다. 둘째, 같은 모델에 적분 간격을 인자로 주어 더 촘촘한 시간 예측과 역방향 적분을 수행한다. 셋째, 단일 영상에 없는 운동 정보를 명시적 $\mathbf p$ 채널로 전달해 속도 조건이 있는 목표를 표현한다. 두 진자 환경은 에너지 보존 여부가 달라도 위상 공간 부피는 보존하므로, 에너지 보존과 심플렉틱성을 구별하기 위한 실험 구성이다. [§1 Contributions, §3](https://arxiv.org/pdf/2610.15267v1#page=3)

## 2. LeHami: 심플렉틱 해밀턴 JEPA (LeHami: Symplectic Hamiltonian JEPA)

먼저 저자들은 예측기가 갖는 성질을 진술한다. **Proposition 1**은 분리 가능한 Hamiltonian $H_\phi=K_{\phi_1}(\mathbf p)+V_{\phi_2}(\mathbf q,\mathbf u)$의 leapfrog 스텝에서 행동 $\mathbf u$를 스텝 동안 고정하면, 예측기가 심플렉틱 미분동형사상(symplectic diffeomorphism)이 된다고 한다. 필요한 $C^2$ 매끄러움은 부록 C.1에 명시돼 있다. 이에 따라 모든 방향의 일률적 수축, 점근적으로 안정된 고정점(asymptotically stable fixed point), 서로 다른 초기 잠재 상태의 유한 스텝 병합이 배제된다. 서로 다른 행동으로 만든 스텝을 합성해도 성질이 유지된다. [§2, Proposition 1; C.1](https://arxiv.org/pdf/2610.15267v1#page=3)

저자들이 바로 뒤에 붙이는 범위도 중요하다. 부피는 보존돼도 오차는 커질 수 있다. 분포의 모양을 정하는 정규화는 여전히 필요하다. 이 제약은 잠재 공간의 귀납 편향(inductive bias)이며, 학습된 $H_\phi$를 실제 물리 에너지와 같다고 두지 않는다. 정리의 증명과 이 세 구분의 구체적인 예는 [2편](/reviews/lehami-symplectic-world-models-technical/)에 있다.

### 2.1 구조 (The architecture)

학습 자료는 주석 없는 영상 $\mathbf x_t$와 행동 $\mathbf u_t$의 시퀀스다. 인코더가 $\mathbf q_t$를 만들고, 고정된 사비츠키–골레이(Savitzky–Golay, SG) 미분 필터가 인코더 궤적의 시간 변화로 $\mathbf p_t$를 추정한다. 이 실험의 총 잠재 폭은 64이며 두 블록은 각각 $d=32$차원이다. SG는 학습 파라미터를 추가하지 않는다. 시작 상태에는 현재와 과거의 네 잠재 위치만 사용하고, 예측이 시작된 뒤에는 $\mathbf p$도 모델의 상태로 계속 진화한다. [Eq.1, D.1–D.2](https://arxiv.org/pdf/2610.15267v1#page=4)

[![인코더와 SG로 시작 상태를 만들고 Hamiltonian 흐름을 반복하여 미래 표현을 학습하는 구조.](/assets/reviews/lehami/lehami-fig2-training.webp)](/assets/reviews/lehami/lehami-fig2-training.webp)

원문 Figure 2. $C$는 미래의 실제 영상에서 얻은 표현과 예측 사이의 손실, $R$은 실제 영상의 임베딩에 적용하는 정규화다. 시작점의 SG와 미래 학습 표적의 SG는 서로 다른 시간 창을 사용한다. [Fig.2](https://arxiv.org/html/2610.15267v1#S2.F2)

Hamiltonian은 다음과 같다. $\phi$는 예측기 파라미터, $\boldsymbol\sigma\in\mathbb R^d$는 학습되는 크기 조정 변수다.

$$
H_\phi(\mathbf q,\mathbf p,\mathbf u)
=K_\phi(\mathbf S^{-1}\mathbf p)+V_\phi(\mathbf q,\mathbf u),
\qquad \mathbf S=\operatorname{diag}(e^{\boldsymbol\sigma}).
$$

운동 항 $K_\phi$는 운동량에만, 퍼텐셜 항 $V_\phi$는 위치와 행동에 의존한다. 두 함수는 폭 1024의 3층 $\tanh$ 다층 퍼셉트론(multilayer perceptron, MLP)이다. Hamilton 방정식 $\dot{\mathbf q}=\partial H_\phi/\partial\mathbf p$, $\dot{\mathbf p}=-\partial H_\phi/\partial\mathbf q$를 관찰 간격 $\Delta t$만큼 leapfrog 적분(leapfrog integration)을 적용하면 원문 Eq.4를 얻는다.

$$
\begin{aligned}
\tilde{\mathbf p}_{t+\frac12}
  &=\mathbf p_t-\frac{\Delta t}{2}\nabla_{\mathbf q}V_\phi(\mathbf q_t,\mathbf u_t),\\
\tilde{\mathbf q}_{t+1}
  &=\mathbf q_t+\Delta t\,\mathbf S^{-1}
    \nabla K_\phi(\mathbf S^{-1}\tilde{\mathbf p}_{t+\frac12}),\\
\tilde{\mathbf p}_{t+1}
  &=\tilde{\mathbf p}_{t+\frac12}
    -\frac{\Delta t}{2}\nabla_{\mathbf q}V_\phi(\tilde{\mathbf q}_{t+1},\mathbf u_t).
\end{aligned}
$$

즉 운동량을 반 스텝, 위치를 한 스텝, 운동량을 반 스텝 움직인다. 가운데 $\mathbf S^{-1}$ 인자는 연쇄 법칙에서 나오며, 위치 갱신에만 나타난다. 행동을 퍼텐셜 안에 넣으므로 각 행동에서 힘은 위치에 대한 기울기이고, leapfrog를 명시적으로 계산할 수 있다. [Eqs.2–4, D.2](https://arxiv.org/pdf/2610.15267v1#page=4)

원문 Eq.5의 목적은 시작 프레임 $A$에서 $K$번 자기 예측한 결과를 각 미래 프레임의 임베딩에 맞추는 것이다.

$$
\mathcal L=
\frac1K\sum_{j=1}^{K}\|\tilde{\mathbf q}_{A+j}-\mathbf q_{A+j}\|^2
+\frac{\lambda_p}{K}\sum_{j=1}^{K}\|\tilde{\mathbf p}_{A+j}-\mathbf p_{A+j}\|^2
+\lambda_{\mathcal S}\mathcal S.
$$

여기서는 $A=3$, $K=8$, $\lambda_p=1$, $\lambda_{\mathcal S}=0.25$이며 제곱 오차는 차원 평균이다. 인코더와 흐름 전체로 역전파한다. 스케치 기반 등방 가우시안 정규화(Sketched-Isotropic-Gaussian Regularizer, SIGReg)는 예측 결과가 아니라 실제 영상의 임베딩에 적용하며, 기본 실험에서는 $\mathbf q$만 정규화한다. 결합·분리 정규화의 차이는 2편의 Table 8 해설로 연결된다. [Eq.5, D.1, C.4](https://arxiv.org/pdf/2610.15267v1#page=5)

### 2.2 관련 연구 (Related work)

저자들은 Hamiltonian을 월드 모델에 사용하는 최근 접근들을 차례로 대비한다. *Physically Native World Models*는 미래 관찰을 복원하고 계획하는 관점을 제시한 position paper다. *HaM-World*는 canonical/context 부분 공간과 외부 제어 항을 사용한다. *PH-Dreamer*는 디코더 기반 Dreamer의 순환 전이에 port-Hamiltonian 구조를 더한다. 가장 가까운 *H-JEPA*는 더 넓은 지각 코드의 고정 부분 공간을 제어 상태로 쓰고, 소산 항과 직교 입력 포트(port), 실행 행동을 읽는 port-inverse consistency를 사용한다. LeHami는 이와 달리 행동을 $V_\phi(\mathbf q,\mathbf u)$ 안에 넣고, $(\mathbf q,\mathbf p)$ 전체를 심플렉틱 스텝으로 진화시킨다. [§2.2, pp.5–6](https://arxiv.org/pdf/2610.15267v1#page=5)

### 리뷰 보충: 비교 원전에서 확인되는 차이

[LeWM §3.1](https://arxiv.org/html/2603.19312v1#S3.SS1)은 단일 프레임 인코더·과거 표현을 읽는 transformer·SIGReg를 결합한다. LeHami 논문의 비교 실험은 그 틀에 공통 다중 스텝 학습을 적용한 진자용 구성이다. 원래 LeWM 논문의 규모·학습 방식과 이 비교 모델을 동일시하지 않아야 한다. [H-JEPA §3.4–3.5](https://arxiv.org/html/2609.33497v1#S3.SS4)에서도 속도 차분은 전이의 문맥이고 진화시키는 변수는 제어 상태 $s$이며, 이산화는 명시적 Euler다. Hamiltonian이라는 이름을 공유해도 실제 이산 전이와 행동 경로는 다르다.

## 3. LeHami의 실증 검증 (Empirical validation of LeHami)

### 3.1 실험 구성 (Experimental setup)

두 환경 모두 $64\times64$ RGB 영상이며, 물리 상태는 각도 $\theta$와 각속도 $\omega$다. 보존 진자는 구동 없이 길이를 세 값으로 바꾸고, 구동 진자는 길이·질량을 각각 세 값으로 바꾸면서 토크를 기록한다. 프레임 간격은 0.1 s, 에피소드 길이는 80프레임이다. 보존계의 train/validation/test는 210/45/45, 구동계는 630/135/135개 에피소드이며, 전체 에피소드 단위로 분리한다. [§3.1; Table 3, D.3](https://arxiv.org/pdf/2610.15267v1#page=23)

주 비교군 LeWorldModel(LeWM)은 64차원 임베딩 세 개의 문맥을 읽는 인과적 transformer 예측기다. 인코더·데이터·학습 스케줄을 맞추고 LeHami에서는 예측기와 SG 경로가 달라진다. 기본 픽셀 인코더는 단일 프레임 ConvNet이며, 시간 창 $W=3$ 기준선의 tied/untied 변형도 비교한다. 실험 표의 $W$는 입력 프레임 **개수**다. 보상 신호를 요구하는 Dreamer나 TD-MPC는 이 무보상 목표 조건 실험의 비교 대상에 포함하지 않는다. [§3.1, D.1–D.2](https://arxiv.org/pdf/2610.15267v1#page=6)

학습된 표현을 읽는 프로브(probe)는 ridge, 25-NN, 2층 MLP의 세 수준이다. 각도는 $\sin\theta,\cos\theta$를 예측한 뒤 $\operatorname{atan2}$로 복원하고, 오차의 차이를 $[-\pi,\pi)$로 감싸는 wrapped MAE를 쓴다. 속도 MAE는 rad/s이며 두 MAE 모두 작을수록 좋다. $R^2(\omega)=1-\mathrm{SSE}/\mathrm{SST}$는 클수록 좋고, 음수는 평균 예측보다 나쁜 상태다. SSE는 예측 잔차 제곱합, SST는 평균 주위의 표적 제곱합이다. 롤아웃 비교의 25-NN은 각 모델의 $K=1$ 출력에서 한 번 맞춘 뒤 고정한다. LeWM의 예측기에도 과거 문맥을 처리할 기회를 주기 위해서다. [§3.1, D.4.1](https://arxiv.org/pdf/2610.15267v1#page=22)

### 3.2 완전한 위상 공간 상태를 담는가? (Evaluating Complete Phase-Space Physics Capture)

보존 진자에서는 LeHami가 초기부터 속도를 더 잘 복원하며, 각도는 초기 성능이 비슷하다가 대략 $K=11$ 이후 유리해진다. 저자들은 $K=1$–10의 속도 MAE가 기준선보다 약 2.8배 낮다고 보고한다. 구간별 원표와 잠재 블록 자체의 오차는 [2편 E.2](/reviews/lehami-symplectic-world-models-technical/)에서 함께 다룬다. [§3.2, Fig.4, Table 7](https://arxiv.org/html/2610.15267v1#S3.F4)

[![보존 진자에서 각도와 각속도 MAE, 예측 반복에 따른 Jacobian 진단.](/assets/reviews/lehami/lehami-fig4-conservative.webp)](/assets/reviews/lehami/lehami-fig4-conservative.webp)

원문 Figure 4. 가로축은 롤아웃 스텝, 왼쪽과 가운데는 각각 rad·rad/s 단위의 MAE다. 물리 예측 곡선은 학습 seed 3개의 변동을, 오른쪽 진단은 seed 1개를 사용한다. 오른쪽 LeWM 곡선의 정확한 Jacobian 정의와 전체 문맥 부피의 추가 확인은 [2편 E.2](/reviews/lehami-symplectic-world-models-technical/)에 있다.

구동 진자의 초기 상태 $K=0$에서는 방향이 더욱 뚜렷하다. 다음은 Table 1의 모든 프로브와 두 모델이며, 각도 열의 원문 단위 $10^{-2}$ rad를 **rad로 변환**했다. 평균±표준편차는 학습 seed 3개에 걸친 값이다.

| 모델 | 프로브 | 각도 MAE [rad] ↓ | 각속도 MAE [rad/s] ↓ | $R^2(\omega)$ ↑ |
| --- | --- | --- | --- | --- |
| LeHami | ridge | 0.230±0.020 | 1.168±0.001 | 0.491±0.005 |
| LeWM | ridge | **0.014±0.001** | 2.422±0.001 | −0.014±0.001 |
| LeHami | 25-NN | 0.068±0.018 | **0.419±0.045** | 0.889±0.024 |
| LeWM | 25-NN | **0.011±0.000** | 2.508±0.002 | −0.103±0.001 |
| LeHami | MLP | 0.029±0.004 | **0.184±0.019** | 0.987±0.004 |
| LeWM | MLP | **0.019±0.001** | 2.419±0.001 | −0.011±0.001 |

단일 프레임의 LeWM 목표 표현은 각도를 정밀하게 담지만 운동 방향을 담기 어렵다. LeHami의 SG 경로는 여러 프레임에서 움직임을 읽어 초기 속도 복원을 가능하게 한다. 그러나 구동계에서 이 속도 MAE의 이점은 대략 $K=10$까지이고, 이후에는 문맥을 처리하는 LeWM 예측기와 비슷해진다. LeWM의 초기 각도 이점도 대략 $K=40$에서 줄어든다. [Table 1, §3.2, Fig.15](https://arxiv.org/pdf/2610.15267v1#page=8)

[![구동 진자 롤아웃에서 LeHami와 단일·다중 프레임 LeWM의 각도·각속도 오차.](/assets/reviews/lehami/lehami-fig15-actuated.webp)](/assets/reviews/lehami/lehami-fig15-actuated.webp)

원문 Figure 15. 실선 LeWM은 $W=1$, 점선은 대칭 가중치를 공유하는 $W=3$이다. 모든 곡선의 프로브는 $K=1$에서 맞춘 뒤 고정하며 음영은 학습 seed 3개의 표준편차다. 방향 정보를 버리는 대칭 시간 창의 원인은 2편 E.3에서 설명한다.

### 3.3 시간 간격을 바꾸고 역방향으로 적분하기 (LeHami is a world model with time re-stepping and reversal)

저자들은 보존 진자에서 학습된 $H_\phi$를 고정하고 적분 간격만 바꾼다. 학습 간격은 $\Delta t=0.1$ s다. 프레임당 두 번 또는 네 번, 각각 0.05·0.025 s로 적분했을 때의 각도 MAE 곡선은 기존 간격과 비슷하다. 비교 표적은 같은 초기 조건에서 실제 진자 방정식을 Runge–Kutta로 적분한 궤적이다. 반면 0.2 s로 간격을 늘리면 오차가 크게 증가한다. 심플렉틱 구조와 수치 안정성은 서로 다른 조건이기 때문이다. [§3.3, Fig.5](https://arxiv.org/html/2610.15267v1#S3.F5)

[![학습보다 작은 적분 간격의 예측 오차와 시간 간격 부호를 뒤집는 왕복 적분 오차.](/assets/reviews/lehami/lehami-fig5-time.webp)](/assets/reviews/lehami/lehami-fig5-time.webp)

원문 Figure 5. 왼쪽은 실제 적분 시간에 따른 각도 MAE의 로그 축, 오른쪽은 앞으로 $K$번 이동한 뒤 $\Delta t$ 부호를 바꿔 돌아오는 잠재 상태의 상대 오차(%)다. 제시된 $K\le30$ 범위에서 왕복 오차는 $10^{-4}\%$보다 작다. 이는 학습한 궤적이 물리적 정답과 일치하는지를 재는 값과 구별되는 역적분 진단이다.

### 3.4 속도 조건 계획의 구조와 계산 효율 (LeHami brings computational efficiency and structural priors in velocity planning)

계획은 구동 진자에서만 수행한다. 과거 실험 에피소드의 실제 상태를 시작점 30프레임 뒤의 목표로 삼아, 도달 가능한 50개 테스트 목표를 모든 모델에 동일하게 준다. 목표는 정지한 위쪽 자세로 고정돼 있지 않다. 시작점과 목표는 각 모델의 인코더로 표현하고, 교차 엔트로피 방법(Cross-Entropy Method, CEM)이 다음 잠재 목표 비용을 최소화할 행동 시퀀스를 찾는다.

$$
\mathcal C(\tilde{\mathbf z}_H)=\|\tilde{\mathbf z}_H-\mathbf z_g\|^2.
$$

LeHami에서 $\mathbf z=(\mathbf q,\mathbf p)$다. 비용에는 실제 물리 좌표나 프로브 출력을 넣지 않는다. CEM은 후보 300개·반복 30번·상위 30개를 쓰고, 토크는 $|u|\le4$로 제한해 3프레임씩 유지한다. 모델 예측 제어(Model Predictive Control, MPC)는 $R$스텝만 실행한 뒤 실제 관찰로 다시 계획하며, 목표 시점까지 남은 시간만큼 예측 구간을 줄인다. 성공은 **목표 시점**에서 각도 오차 $<0.1$ rad, 속도 오차 $<0.5$ rad/s인지를 각각, 그리고 동시에 판정한다. [§3.4; D.4.2, pp.23–24](https://arxiv.org/pdf/2610.15267v1#page=23)

Table 2는 $R\in\{30,15,9,6,3\}$ 중 15 TFLOP/episode 이하에서 각 조건의 **최고 성공률**을 택한다. 모델 seed 3개, 각 모델당 planner seed 3개와 50개 목표를 사용한다. 따라서 다음 평균은 단일 50회 실험의 성공 횟수가 아니다. 예를 들어 0.800은 각 모델 seed에서 150회 시행의 비율을 구한 뒤 평균한 값이다.

| 모델 | 위치 성공률 | 속도 성공률 | 위치·속도 동시 성공률 |
| --- | --- | --- | --- |
| LeWM $W=1$ | **0.800±0.019** | 0.184±0.027 | 0.142±0.033 |
| LeWM $W=3$, tied | 0.633±0.020 | 0.473±0.014 | 0.400±0.037 |
| LeWM $W=3$, untied | 0.696±0.016 | **0.800±0.058** | **0.636±0.025** |
| LeHami | 0.480±0.084 | 0.527±0.019 | 0.304±0.088 |
| 토크 0의 수동 운동 | 0.060 | 0.240 | 0.040 |

Table 2의 네 모델과 수동 기준을 모두 남겼다. 각 열의 최고 설정은 다를 수 있으므로 세 열의 확률을 곱하지 않는다. LeHami의 위치 열은 $R=9$, 8.05 TFLOP이고 속도·동시 조건은 $R=6$, 10.98 TFLOP다. untied LeWM은 세 열 모두 $R=9$, 13.65 TFLOP다. LeWM $W=1$의 위치·동시 열은 $R=9$, 12.87 TFLOP, 속도는 $R=15$, 8.78 TFLOP이고 tied 모델의 위치·속도는 $R=9$, 13.65 TFLOP, 동시는 $R=15$, 9.31 TFLOP다. [Table 2](https://arxiv.org/html/2610.15267v1#S3.T2)

속도 열의 원시 시행을 합산하면 LeHami는 450회 중 237회, tied LeWM은 213회, untied LeWM은 360회 성공이다. 동일한 50개 목표를 세 모델 seed와 세 planner seed로 반복한 집계다.

시간 가중치를 공유하는 tied 인코더는 움직임의 크기를 읽어도 방향을 구별하지 못한다. 이를 풀어 준 untied LeWM은 가장 높은 속도·결합 성공률을 얻는다. LeHami는 더 저렴한 흐름을 이용해 속도 조건에서 더 자주 다시 계획할 수 있으나, 위치 목표 도달이 결합 성공률의 병목이다. 속도 부호의 성공률과 목표 분포, 파라미터·계산량의 상세 비교는 2편에 이어진다. [§3.4, E.3](https://arxiv.org/pdf/2610.15267v1#page=9)

## 4. 결론 (Conclusion)

저자들은 자기지도 잠재 공간에 심플렉틱 전이를 넣어 평가 시점의 특정 수축 실패를 구조적으로 배제할 수 있다고 결론짓는다. 보존 진자의 장기 예측, 관찰 간격보다 작은 스텝의 적분, 역적분이 그 활용 예다. 동시에 구동 진자의 위치 목표에서는 LeWM이 강하고, 속도 정보를 담으려면 기준선의 시간 대칭성을 풀어야 한다고 정리한다. [§4, p.9](https://arxiv.org/pdf/2610.15267v1#page=9)

저자가 남기는 확장 방향은 강한 마찰을 포함한 소산계(dissipative system)를 위한 port-Hamiltonian 모델, 정적 자세와 속도 과제 사이의 설계 선택, 고정 SG 대신 학습하는 운동량 표현이다. 이 논문의 실제 실험은 부피를 보존하는 두 진자 환경이다.

## 리뷰 해석과 남은 질문

**리뷰 해석.** 가장 유용한 기여는 학습 분포를 정돈하는 제약과 추론 중 전이 자체가 지키는 제약을 분리한 데 있다. 같은 상태를 매번 예측기에 다시 넣을 때 생기는 문제에, 각 스텝에서 확인할 수 있는 수학적 성질로 답한다. 반면 초기 속도 프로브의 큰 차이에는 입력 정보의 차이도 작용한다. LeHami의 SG는 네 시점의 변화를 읽고 단일 프레임 LeWM의 목표 표현은 한 시점만 읽는다. untied 다중 프레임 기준선과의 계획 비교가 그래서 특히 중요하다.

**리뷰 해석.** 다음 실증 과제는 더 강한 기준선과 같은 시간 정보·목표 비용을 주면서 장기 기하학 제약이 추가하는 몫을 분리하는 것이다. 현재 논문은 predictor와 SG·목표 표현을 묶어 바꾸므로, 어느 이득이 어느 구성요소에서 오는지를 완전히 분해하지 않는다. 부록의 정규화 실패·잠재 오차·계산량은 이 질문에 필요한 근거이며 [2편](/reviews/lehami-symplectic-world-models-technical/)에서 살펴본다.

## 출처와 읽은 범위

- 원문: Romina Yalovetzky, Randall Balestriero, Yann LeCun, *LeHami: Symplectic Latent Dynamics for World Models*, 최초 공개 2026-10-11, arXiv v1, 검토일 2026-10-26. [공식 서지](https://arxiv.org/abs/2610.15267v1).
- 범위: 본문 §1–4와 실험 설정 D.1–D.4, 구동 롤아웃 E.3. 부록 B–E의 상세 해설은 [2편](/reviews/lehami-symplectic-world-models-technical/)에 연결한다.
- 추가 출처: [LeWM v1 §3](https://arxiv.org/html/2603.19312v1#S3), [H-JEPA v1 §3.1–3.5](https://arxiv.org/html/2609.33497v1#S3).
- 그림: LeHami v1 Figures 1·2·4·5·15, Yalovetzky et al., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 대표 이미지는 Figure 1. 저자 PDF를 래스터화하거나 PNG 해상도를 줄인 뒤 WebP로 인코딩했으며 패널·글자·데이터는 바꾸지 않았다.
- 확인하지 않은 것: 모델 재학습과 시뮬레이터에서의 독립 성능 재현.
