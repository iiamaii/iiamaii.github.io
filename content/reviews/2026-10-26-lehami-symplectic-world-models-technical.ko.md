---
title: "[2/2] LeHami 심화: 보장의 증명과 표현이 실패하는 조건"
description: "Leapfrog의 전단 사상 증명, SG와 운동량의 조건, 정규화·시간 대칭성·계산량의 절제 실험으로 LeHami의 보장 범위를 해설한다."
date: "2026-10-26"
publishedAt: "2026-10-26T13:56:31+09:00"
updatedAt: "2026-10-26T13:56:31+09:00"
topics: ["world models", "representation learning", "Hamiltonian dynamics", "geometric integration", "model predictive control"]
visibility: "public"
lang: "ko"
translationKey: "lehami-symplectic-world-models-technical"
paperTitle: "LeHami: Symplectic Latent Dynamics for World Models"
paperPublishedDate: "2026-10-11"
authors: ["Romina Yalovetzky", "Randall Balestriero", "Yann LeCun"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.15267v1"
thumbnail: "/assets/reviews/lehami/lehami-fig1-concept.webp"
thumbnailAlt: "LeHami Figure 1의 위상 공간: 위치와 운동량 블록을 Hamiltonian의 심플렉틱 전이로 연결한다."
---

LeHami의 심플렉틱(symplectic) 보장은 예측 전이의 부피를 지키지만, 학습된 표현에 실제 움직임이 남는지는 정규화와 인코더 설계에도 달려 있다. 구동 진자의 단일 프레임 인코더에서 스케치 기반 등방 가우시안 정규화(Sketched-Isotropic-Gaussian Regularizer, SIGReg)를 두 블록에 결합 적용하면 에피소드 내부 변화가 차지하는 분산 비중은 0.01%이고, 블록별 분리 적용에서는 48.89±16.39%다. [Table 8](https://arxiv.org/html/2610.15267v1#A5.T8) **리뷰 해석.** 부록의 가장 강한 메시지는 부피를 보존하는 흐름과 운동 정보를 담는 표현을 따로 검증해야 한다는 실증적 근거이며, 학습 설계에 따른 실패가 제안 구조 안에서도 발생한다.

[\[1/2\] 방법과 주요 실험](/reviews/lehami-symplectic-world-models/)에 이어, 이 편은 안내용 부록 A를 제외한 B–E의 순서를 따른다.

## B. 해밀턴 역학 배경 (Background in Hamiltonian mechanics)

해밀턴 역학(Hamiltonian mechanics)은 일반화 위치 $\mathbf q\in\mathbb R^d$와 공액 운동량(conjugate momentum) $\mathbf p\in\mathbb R^d$의 짝을 상태로 사용한다. $\mathbf z=(\mathbf q,\mathbf p)$라 두면 흐름은 $\dot{\mathbf z}=\mathbf J\nabla H$로 쓸 수 있다.

$$
\mathbf J=\begin{pmatrix}0&I_d\\-I_d&0\end{pmatrix},
\qquad
\dot{\mathbf q}=\nabla_{\mathbf p}H,\quad
\dot{\mathbf p}=-\nabla_{\mathbf q}H.
$$

심플렉틱 형식(symplectic form) $\Omega=\sum_i dq_i\wedge dp_i$는 방향 있는 면적을 기록한다. 이를 보존하면 부피 형식 $\Omega^d/d!$도 보존한다. 리우빌 방정식(Liouville equation) $\partial_t\rho=-\{\rho,H\}$는 흐름을 따라가는 관찰자에게 확률 밀도 $\rho$가 일정함을 뜻한다. 여기서 푸아송 괄호(Poisson bracket)는 $\{\rho,H\}=\nabla_{\mathbf q}\rho\cdot\nabla_{\mathbf p}H-\nabla_{\mathbf p}\rho\cdot\nabla_{\mathbf q}H$다. [부록 B, p.13](https://arxiv.org/pdf/2610.15267v1#page=13)

### 리뷰 보충 유도: 부피 보존과 에너지 보존의 관계

$H$가 두 번 연속 미분 가능하면 벡터장의 발산은 혼합 편미분의 상쇄로 0이다.

$$
\nabla\cdot\dot{\mathbf z}
=\sum_i\left(
\frac{\partial^2H}{\partial q_i\partial p_i}
-\frac{\partial^2H}{\partial p_i\partial q_i}\right)=0.
$$

반면 궤적을 따라 에너지의 변화율은 $dH/dt=\partial_tH$다. 시간에 명시적으로 의존하지 않는 연속 Hamiltonian의 에너지는 보존되지만, 행동이 바뀌면 $H$도 바뀔 수 있다. 이 구분이 구동 진자를 다루는 출발점이다. 이산 적분의 에너지는 별도의 수치 오차를 가진다.

## C. 이론 (Theory)

### C.1 Leapfrog의 심플렉틱성: 진술과 증명 (Symplecticity of the leapfrog step: formal statement and proof)

부록의 **Proposition 2**는 본문 Proposition 1을 정식화한다. $K_{\phi_1}\in C^2(\mathbb R^d)$, $V_{\phi_2}\in C^2(\mathbb R^d\times\mathcal U)$이고 각 스텝에서 행동 $\mathbf u\in\mathcal U$를 고정한다. 적분 스텝 $h>0$와 파라미터 $\phi$의 값에 관계없이, leapfrog $\Psi_h$는 $\mathbb R^{2d}$의 $C^1$ 미분동형사상(diffeomorphism)이며

$$
D\Psi_h(\mathbf z)^\top\mathbf J D\Psi_h(\mathbf z)=\mathbf J,
\qquad \det D\Psi_h(\mathbf z)=1
$$

을 모든 $\mathbf z$에서 만족한다. $D$는 상태에 대한 Jacobian이다. 증명은 두 전단 사상(shear map)을 합성하는 순서로 진행한다. [C.1, Proposition 2, Eqs.7–8, pp.14–15](https://arxiv.org/pdf/2610.15267v1#page=14)

$$
\begin{aligned}
A_{h/2}(\mathbf q,\mathbf p)
 &=\left(\mathbf q,\mathbf p-\tfrac h2\nabla_{\mathbf q}V_{\phi_2}(\mathbf q,\mathbf u)\right),\\
B_h(\mathbf q,\mathbf p)
 &=\left(\mathbf q+h\nabla_{\mathbf p}K_{\phi_1}(\mathbf p),\mathbf p\right),\\
\Psi_h&=A_{h/2}\circ B_h\circ A_{h/2}.
\end{aligned}
$$

**1. 각 전단이 심플렉틱이다.** Jacobian은 다음 블록 삼각 행렬이다.

$$
DA_{h/2}=\begin{pmatrix}I_d&0\\-\tfrac h2\nabla^2_{\mathbf q\mathbf q}V&I_d\end{pmatrix},
\quad
DB_h=\begin{pmatrix}I_d&h\nabla^2_{\mathbf p\mathbf p}K\\0&I_d\end{pmatrix}.
$$

$C^2$ 조건은 두 Hessian의 대칭성을 보장한다. 원문이 짧게 넘기는 행렬 곱을 펼치면, 대칭 행렬 $C$에 대한 $L=\begin{pmatrix}I&0\\C&I\end{pmatrix}$는 $L^\top\mathbf JL=\begin{pmatrix}C-C^\top&I\\-I&0\end{pmatrix}=\mathbf J$다. 위쪽 전단도 같은 방식으로 성립한다. 이것이 **리뷰 보충 유도**이며, 단순히 행렬식이 1인 것보다 강한 성질이다. 두 전단의 행렬식은 각각 1이고 역함수는 $A_{-h/2}$, $B_{-h}$로 명시된다. 따라서 전역적인 역함수가 존재한다.

**2. 합성이 성질을 보존한다.** 연쇄 법칙을 적용하면

$$
D(\Psi\circ\Phi)^\top\mathbf J D(\Psi\circ\Phi)
=D\Phi^\top(D\Psi^\top\mathbf JD\Psi)D\Phi
=\mathbf J.
$$

행렬식도 곱해져 1이다. 각 단계의 행동 값이 달라도 그 단계에서 고정한 전단을 합성하면 된다. 원문의 $K(\mathbf p)$에는 실제 구현의 $K_\phi(\mathbf S^{-1}\mathbf p)$를 하나의 함수로 포함시킬 수 있다. 학습 뒤 고정된 크기 조정 행렬 $\mathbf S=\operatorname{diag}(e^{\boldsymbol\sigma})$를 거친 Hessian $\mathbf S^{-1}\nabla^2K\mathbf S^{-1}$도 대칭이므로 같은 증명이 적용된다.

**3. 세 결론이 따른다.** $K$번 합성한 Jacobian의 특잇값(singular value)을 $\sigma_i$라 쓰면 $\prod_{i=1}^{2d}\sigma_i=|\det D\Psi_h^K|=1$이다. 따라서 $\sigma_{\max}\ge1\ge\sigma_{\min}$이고 모든 방향이 한꺼번에 줄어들 수 없다. 또한 합성은 단사(injective)이므로 다른 초기 상태는 어느 유한 $K$에서도 같은 점이 되지 않는다.

마지막으로 점근적으로 안정된 고정점은 **Lyapunov 안정성**과 **열린 근방의 모든 점에 대한 끌림**을 동시에 요구한다. 그런 점 $\mathbf z^*$가 있다고 가정하고 끌림 영역 안에 양의 부피를 가진 닫힌 공 $C$를 잡는다. 안정성으로 임의의 작은 $\varepsilon$-공 안에 머무는 근방을 정하고, 끌림과 연속성으로 $C$의 각 점 주변이 그 근방에 들어가는 유한 시점을 얻는다. $C$의 콤팩트성(compactness) 때문에 유한 개 근방으로 덮을 수 있고 진입 시점의 최댓값 이후에는 $C$ 전체가 $\varepsilon$-공 안에 머문다. 그러면

$$
0<\operatorname{vol}(C)
=\operatorname{vol}(\Psi_h^K(C))
\le c_{2d}\varepsilon^{2d}
$$

이며 $\varepsilon\to0$에서 모순이다. 단순히 한 궤적이 어떤 점으로 접근한다는 주장과 달리, 이 증명은 열린 근방 전체가 안정적으로 끌리는 경우를 배제한다. [C.1 증명](https://arxiv.org/pdf/2610.15267v1#page=15)

### C.2 보장의 범위와 한계 (Scope and limitations of the guarantee)

저자의 세 remark는 부피 보존과 정확도, 정규화, 물리 좌표를 구분한다. 첫째, 전단과 늘어남은 무제한일 수 있어 롤아웃 오차에 상한이 생기지 않는다. 둘째, 인코더에 SIGReg가 만든 등방 가우시안 분포가 흐름에서도 유지되려면 $\{\|\mathbf z\|^2,H_\phi\}=0$이라는 추가 조건이 필요하다. 부피 보존만으로 분포의 모양이 고정되지는 않는다. 셋째, 실제 물리량은 학습 신호가 아니라 평가 프로브의 표적이며, $\mathbf q$·$\mathbf p$라는 이름 자체가 관찰 공간의 위치·운동량과 동일함을 뜻하지 않는다. [C.2, pp.15–16](https://arxiv.org/pdf/2610.15267v1#page=15)

### C.3 Hamiltonian을 택한 이유 (Why Hamiltonian and not Lagrangian)

저자들의 선택은 예측기의 계산 구조에 달려 있다. Hamilton 방정식은 $(\mathbf q,\mathbf p)$에서 일계이고, 분리 가능한 Hamiltonian이면 명시적 leapfrog를 사용할 수 있다. 위치와 속도의 접다발(tangent bundle) 위에 쓰는 Lagrange 방정식은 위치에 대해 이계이며 이 구현의 저렴한 위상 공간 보장을 직접 제공하지 않는다. [C.3, p.16](https://arxiv.org/pdf/2610.15267v1#page=16)

### C.4 흐름과 인코더의 설계 (Design choices in the flow and the encoder)

**행동과 에너지.** $\mathbf u_t$를 스텝마다 고정하면 $H_\phi$의 변화는 행동 변경에 의한 항 $W_t$와 고정 행동에서 적분한 잔차 $r_t$로 나뉜다. 원문 Eq.10은 두 항을 더하고 빼는 정확한 항등식이다.

$$
\begin{aligned}
H_\phi(\mathbf z_{t+1},\mathbf u_{t+1})-H_\phi(\mathbf z_t,\mathbf u_t)
&=W_t+r_t,\\
W_t&=H_\phi(\mathbf z_{t+1},\mathbf u_{t+1})-H_\phi(\mathbf z_{t+1},\mathbf u_t),\\
r_t&=H_\phi(\mathbf z_{t+1},\mathbf u_t)-H_\phi(\mathbf z_t,\mathbf u_t).
\end{aligned}
$$

원문은 $W_t$를 일(work)로 해석한다. 소산을 다루는 확장은 양의 준정부호(positive semidefinite) 행렬 $D_{\phi_3}=L_{\phi_3}L_{\phi_3}^\top$를 운동량 방정식에 도입하는 방향이며, 이 논문에서는 실험하지 않았다. [C.4](https://arxiv.org/pdf/2610.15267v1#page=16)

**사비츠키–골레이(Savitzky–Golay, SG) 필터의 두 용도.** 시간마다 인코딩한 $\mathbf q$에 이차 다항식을 최소제곱으로 맞춘 뒤 그 도함수를 현재 시점에서 읽는다. 시작점의 인과적(causal) 필터는 $t-3,\ldots,t$의 네 값을, 학습 표적의 중심형(centered) 필터는 $t-2,\ldots,t+2$의 다섯 값을 사용한다. 후자는 표적일 뿐 모델 입력에 주지 않는다. 예측이 시작된 뒤에는 필터를 다시 적용하지 않는다.

중심형 계수는 $(-2,-1,0,1,2)/10$이며 시간 간격으로 나누어 미분 단위를 맞춘다. 시간 반전에 대한 반대칭성(antisymmetry)이 있어 창을 뒤집으면 추정치의 부호가 정확히 뒤집힌다. 독립적인 프레임 잡음의 분산 증폭 계수 $\sum_mw_m^2$는 중심형 0.10, 인과형 2.45로 24.5배 차이다. 원문은 두 필터의 절단 편향(truncation bias)도 비교한다. 인과형은 시간 반전 시 전방 필터가 되므로 중심형과 같은 정확한 반대칭성을 갖지 않는다. [C.4, D.2](https://arxiv.org/pdf/2610.15267v1#page=17)

**운동량과 질량.** 저자들은 $\mathbf p\approx\dot{\mathbf q}$라는 정의가 학습 좌표에서 단위 질량을 택한다고 설명한다. 이 관계를 정확히 재현하는 연속 흐름이라면 데이터 지지집합에서 $\nabla_{\mathbf p}H_\phi=\mathbf p$이고 운동 항은 $\frac12\|\mathbf p\|^2+C_\phi$에 맞춰져야 한다. 다만 실제로는 미분 추정·이산 스텝의 오차가 있고 데이터 밖까지 정해지지 않으므로 $K_\phi$를 MLP로 유지한다. 상수 질량 행렬은 좌표 변환으로 흡수할 수 있지만 이중 진자처럼 질량 행렬이 위치에 의존하면 같은 전역 선형 변환으로 제거되지 않는다. 이 경우 고정 SG로 읽은 속도를 공액 운동량으로 취급하는 분리형 모델의 가정이 맞지 않을 수 있다. [C.4, Eqs.11–12, p.17](https://arxiv.org/pdf/2610.15267v1#page=17)

**크기 조정과 정규화.** $\mathbf S=\operatorname{diag}(e^{\boldsymbol\sigma})$는 상대적 크기를 학습하며 관찰된 대각 값은 0.95–1.09다. 정규화는 $\mathbf q$만, 결합 $[\mathbf q\Vert\mathbf p]$, 또는 두 블록 각각의 SIGReg를 합하는 세 선택이다. $A=3$, 예측 길이 $K=8$, 중심형 반폭 2이면 14개의 잠재 프레임이 필요하다. 중심형 운동량이 정의되는 10개 시점에 정규화를 적용한다. [C.4, pp.17–18](https://arxiv.org/pdf/2610.15267v1#page=18)

## D. 실험 설정 (Description of the experimental set up)

### D.1 방법론 (Methodology)

모델은 같은 데이터 파이프라인·최적화기·학습률 스케줄을 사용하고 검증 손실로 체크포인트를 선택한다. 기본 LeHami는 $\mathbf q$만 정규화하며 계획 비교는 35 epoch 모델을 쓴다. 잠재 좌표 수가 같아도 LeHami는 32+32, LeWM은 하나의 64차원 코드라는 할당 차이가 있다. [D.1; D.4.2](https://arxiv.org/pdf/2610.15267v1#page=18)

### D.2 구현 (Implementation of LeHami)

시간 ConvNet은 프레임별 초기 특징 추출부(stem) 뒤에 한 번의 시간축 합성곱을 넣고, 기존 공간 trunk와 projector를 연결한다(Fig.6). 주요 LeHami 모델은 시간 혼합 없는 $W=1$을 쓴다. Hamiltonian의 자동미분 경로는 fp32로 계산한다. 여러 관찰 간격을 하나의 $H_\phi$로 학습하는 확장도 설명하지만, **혼합 기록 주기 학습 실험은 하지 않았다**. 주기 변경 실험은 단일 주기로 학습한 모델의 추론 간격을 바꾸는 실험이다. [D.2, Fig.6, pp.19–21](https://arxiv.org/pdf/2610.15267v1#page=19)

### D.3 데이터셋과 구성 (Datasets and their architectures)

두 환경의 방정식은 $\ddot\theta=-g\sin\theta/L$와 $\ddot\theta=-g\sin\theta/L+u/(mL^2)$이며 $g=9.81$ m/s²다. $L\in\{0.7,1.0,1.4\}$ m, 구동계의 $m\in\{0.5,1.0,2.0\}$ kg다. 길이는 화면의 막대 길이에, 질량은 추의 반지름에 드러난다(Fig.7). 토크는 0.3 s, 즉 3프레임 동안 유지한다. [D.3, Eqs.13–14](https://arxiv.org/pdf/2610.15267v1#page=21)

보존계의 초기 각도는 $[-\pi,\pi]$에서 뽑고, 초기 각속도 범위는 $\sqrt{g/L}$에 비례하게 맞춘다. 기준 길이 1 m에서는 $|\omega_0|\le2$ rad/s다. 진동과 회전 궤적을 모두 포함한다. 분할 seed는 42이며 길이 또는 길이×질량의 설계 셀마다 70/15/15 비율로 에피소드를 나눈다. 보존계는 생성된 1,200개 중 0.1 s 간격의 300개만 사용한다. 에피소드 수와 분할 표는 [1편 §3.1](/reviews/lehami-symplectic-world-models/)에 정리했다. [Table 3, D.3](https://arxiv.org/pdf/2610.15267v1#page=23)

### D.4 평가 (Evaluation)

롤아웃 프로브는 지정된 시점에서 학습해 고정한다. 반면 아래 Figure 8의 블록별 프로브는 테스트 에피소드에서 뽑은 특징을 **에피소드별 5-fold 교차검증**으로 나눠 분석한다. 이 그림을 훈련 에피소드에 맞춘 전체 표현 프로브인 Table 1과 같은 프로토콜로 읽으면 안 된다. [D.4.1; Fig.8 caption](https://arxiv.org/pdf/2610.15267v1#page=25)

계획에서는 시작점과 목표의 실제 영상만 잠재 상태를 만드는 데 쓰고, 중간 미래 영상이나 실제 물리 좌표는 후보 순위를 매기는 데 쓰지 않는다. MPC가 다시 관찰할 때는 그때까지 관찰한 프레임만 사용한다. 중심형 시간 인코더가 있는 경우 인코더 지연만큼 문맥·목표 시점을 보정하고 예측 스텝을 더한다. 목표 시점의 각도·속도 허용치와 CEM 예산은 [1편 §3.4](/reviews/lehami-symplectic-world-models/)에 있다. [D.4.2, pp.23–24](https://arxiv.org/pdf/2610.15267v1#page=23)

프로브의 $R^2=1-\mathrm{SSE}/\mathrm{SST}$에서 SSE는 잔차 제곱합, SST는 평균 주위 표적 제곱합이며 1이 최적이고 음수는 평균 예측보다 나쁘다. 각도는 사인·코사인 쌍으로 읽고 오차를 원 위에서 감싼다.

## E. 보충 실험 (Supplementary experimental results)

### E.1 구조적 사전 지식과 계산량 (LeHami replaces manual design choices with structural priors and lower compute)

단일 프레임 표현에는 운동 방향을 직접 담을 정보가 없다. 저자들은 위치·운동량 블록을 나눠 읽어, LeHami의 각도는 주로 $\mathbf q$, 속도는 주로 $\mathbf p$에서 복원됨을 보여 준다. 25-최근접 이웃(25-nearest neighbors, 25-NN) 프로브를 쓴 Figure 8의 각도 $R^2$는 원시 각도가 아니라 $(\sin\theta,\cos\theta)$ 표적에 대한 값이다.

[![LeHami의 두 블록과 LeWM의 전체 표현에서 읽은 각도·각속도 R제곱.](/assets/reviews/lehami/lehami-fig8-blocks.webp)](/assets/reviews/lehami/lehami-fig8-blocks.webp)

원문 Figure 8. 실색은 $\mathbf q$, 빗금은 $\mathbf p$이고 LeWM은 전체 코드 $\mathbf z$다. 25-NN, 에피소드별 5-fold, 학습 seed 3개 평균±표준편차다. 오른쪽에서 운동량 블록의 속도 신호를 읽는다. [Fig.8](https://arxiv.org/html/2610.15267v1#A5.F8)

Table 4는 보존 진자에서 동일한 누적 학습 계산 예산 아래 선택된 체크포인트를 비교한다. 프로브는 이 표에서만 $k=20$인 최근접 이웃이며 모델당 seed 1개다. 아래 각 셀은 각도 평균 절대 오차(mean absolute error, MAE) [rad] / 속도 MAE [rad/s]이며 작을수록 좋다. 예산별 체크포인트의 epoch는 9/19/39/79다.

| 모델 | 0.75 PFLOP | 1.5 PFLOP | 3 PFLOP | 6 PFLOP |
| --- | --- | --- | --- | --- |
| LeHami $W=1$ | 0.283 / 1.002 | 0.207 / 0.929 | 0.252 / 0.934 | 0.222 / 0.902 |
| LeWM $W=1$ | 0.590 / 1.625 | 0.871 / 2.146 | 0.487 / 1.620 | 0.321 / 1.270 |

이 표에서는 LeHami의 0.75 PFLOP 결과가 LeWM의 6 PFLOP 결과보다 두 지표 모두 좋다. 저자의 “최대 1/8의 학습량”은 이 비교다. 100 epoch의 총비용은 두 모델 모두 약 6.0–6.5 PFLOP, 0.7–1.1 GPU-hour로 보고된다. [Table 4, E.1](https://arxiv.org/html/2610.15267v1#A5.T4)

Table 5의 파라미터 수는 다음과 같다. 단위는 백만 개이며 predictor와 predictor projector를 분리했다.

| 모델 | 전체 | 인코더 | 예측기 | 잠재 projector | predictor projector |
| --- | --- | --- | --- | --- | --- |
| LeHami $W=1$ | 3.03 | 0.315 | 2.234 | 0.465 | 없음 |
| LeWM $W=1$ | 4.44 | 0.315 | 3.310 | 0.530 | 0.268 |
| LeWM $W=3$ | 4.45 | 0.318 | 3.310 | 0.530 | 0.268 |

전체로 약 32% 감소다. 본문 §3.4의 3.31M이라는 표현과 달리, Table 5에서는 LeWM의 **예측기만** 3.310M이고 projection head 0.268M은 추가다. [Table 5](https://arxiv.org/html/2610.15267v1#A5.T5)

### E.2 보존 진자 (Conservative pendulum)

Figures 9–11은 주성분 분석(Principal Component Analysis, PCA)으로 잠재 궤적과 에너지 층을 시각화한다. Fig.9는 여러 주성분 쌍, Fig.10은 LeHami와 LeWM의 궤적, Fig.11은 실제 에너지와 학습된 $H_\phi$의 구조를 나란히 놓는다. 저자들은 에너지별 닫힌 곡선과 층화를 관찰한다. [Figs.9–11, pp.27–28](https://arxiv.org/pdf/2610.15267v1#page=27)

Table 6은 물리 좌표가 아니라 **예측한 잠재 블록과 실제 영상의 인코더 출력** 사이의 $R^2$다.

| 예측 스텝 | 0 | 1 | 5 | 10 | 20 | 25 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $R^2_q$ | 1.000 | 0.997 | 0.985 | 0.950 | 0.929 | 0.646 | 0.373 |
| $R^2_p$ | 1.000 | 0.935 | 0.867 | 0.613 | 0.586 | −0.324 | −1.165 |

운동량 블록은 위치 블록보다 빨리 참조 궤적에서 벗어난다. 물리량이 얼마나 읽히는가는 Table 7이 별도로 보여 준다. 25-NN을 $K=1$에 맞춘 뒤 고정한 구간별 평균 MAE이며, 각도는 rad, 속도는 rad/s다. 비율은 LeWM÷LeHami여서 1보다 크면 LeHami가 좋다.

| 구간 | 각도 LeHami | 각도 LeWM | 비율 | 속도 LeHami | 속도 LeWM | 비율 |
| --- | --- | --- | --- | --- | --- | --- |
| $K=1$–10 | 0.107 | **0.100** | 0.93 | **0.364** | 1.020 | 2.80 |
| $K=11$–20 | **0.225** | 0.287 | 1.27 | **0.669** | 1.169 | 1.75 |
| $K=21$–70 | **0.596** | 0.981 | 1.65 | **1.534** | 2.331 | 1.52 |

같은 표의 초기 $K=0$ 속도 $R^2$는 LeHami 0.975, LeWM −0.667이고 LeHami의 ridge는 0.448이다. 속도 신호는 선형으로 완전히 드러나는 형태가 아니다. [Tables 6–7](https://arxiv.org/pdf/2610.15267v1#page=28)

인코더 교체 실험은 별도 조건이다. 길이 1 m, 간격 0.05 s, 500개×100스텝의 단일 막대 데이터에서 ViT/ConvNet의 초기 속도 $R^2$는 LeHami 0.929/0.939, LeWM −0.031/−0.018이다. 고정 DINOv2 위에 projector를 학습하는 Figure 12도 0.05 s, 모델당 seed 1개·epoch 60의 비교이며, LeHami의 속도 이점과 $K>60$의 양쪽 오차 증가를 보고한다. 이 수치들을 세 길이의 주 실험과 합쳐 평균하지 않는다. [E.2, Fig.12](https://arxiv.org/pdf/2610.15267v1#page=29)

**어느 Jacobian의 부피인가?** 상태의 정방형 전이 $F$에서 $J(\mathbf z_i)=\partial F/\partial\mathbf z$라 두면 누적 정규화 log-volume은

$$
\frac1n\sum_{i=0}^{K-1}\log|\det J(\mathbf z_i)|
$$

이다. 0은 부피 보존, 음수는 수축이다. LeHami의 상태 폭은 $n=64$다. LeWM의 예측기는 세 임베딩을 받아 하나를 내므로 전체 상태 전이를 $(\mathbf z_{t-2},\mathbf z_{t-1},\mathbf z_t)\mapsto(\mathbf z_{t-1},\mathbf z_t,\mathbf z_{t+1})$로 구성해야 $n=192$의 정방형 Jacobian을 얻는다. 이 행렬식은 부호를 제외하면 $\det(\partial\mathbf z_{t+1}/\partial\mathbf z_{t-2})$와 같다. 최신 블록 $\partial\mathbf z_{t+1}/\partial\mathbf z_t$만의 행렬식은 전체 문맥의 부피가 아니다. [E.2, pp.29–30](https://arxiv.org/pdf/2610.15267v1#page=29)

[![전체 상태 Jacobian과 최신 블록 Jacobian을 구분한 부피 변화 진단.](/assets/reviews/lehami/lehami-fig13-volume.webp)](/assets/reviews/lehami/lehami-fig13-volume.webp)

원문 Figure 13. 왼쪽 전체 전이는 LeHami 64차원·LeWM 192차원, 오른쪽 최신 블록은 둘 다 64차원이다. 테스트 초기 상태 45개·모델당 학습 seed 1개에서 LeWM은 두 진단 모두 음수이고 LeHami는 수치적으로 0 부근이다. 이 그림은 전체 문맥을 확인하는 근거다. [Fig.13](https://arxiv.org/html/2610.15267v1#A5.F13)

### E.3 구동 진자 (Actuated pendulum)

Figure 14는 행동이 에너지를 더하거나 빼는 동안 궤적이 $H_\phi$의 등위면을 넘나드는 모습을 보여 준다. 이 경우에도 심플렉틱 제약은 각 고정 행동에서의 전이에 적용된다. [Fig.14](https://arxiv.org/html/2610.15267v1#A5.F14)

**정규화 위치의 절제 실험(ablation).** Table 8은 차원별 에피소드 내부 분산 $V_{\rm within}$과 에피소드 평균 사이 분산 $V_{\rm between}$을 구해 $V_{\rm within}/(V_{\rm within}+V_{\rm between})$를 차원 평균한다. 거의 0이면 상태 변화보다 에피소드 구별에만 표현을 쓰는 상황이다. 테스트의 인코더 출력 $K=0,\ldots,74$, 학습 seed 3개를 사용한다.

| 창 | SIGReg 위치 | 내부 분산 비중 [%] | $\mathrm{std}(q)$ | $\mathrm{std}(p)$ | $\mathrm{std}(p)/\mathrm{std}(q)$ |
| --- | --- | --- | --- | --- | --- |
| $W=1$ | 결합 | 0.01±0.00 | 1.356 | 0.006 | 0.0043 |
| $W=1$ | $q$만 | 15.32±0.29 | 0.980 | 0.101 | 0.1031 |
| $W=1$ | 분리 | **48.89±16.39** | 1.174 | 0.236 | 0.2014 |
| $W=3$ | 결합 | 4.90±0.30 | 1.405 | 0.051 | 0.0362 |
| $W=3$ | $q$만 | 8.87±0.54 | 0.979 | 0.051 | 0.0519 |
| $W=3$ | 분리 | **47.06±2.56** | 1.124 | 0.201 | 0.1784 |

결합 정규화에서는 $p$가 거의 0이고 $q$의 표준편차가 약 $\sqrt2$로 커지는 보상 현상을 저자들이 관찰한다. 분리 정규화는 내부 변화 비중을 늘리지만 $p$의 분산을 곧바로 1로 만들지는 않는다. 계획과 본문 Table 1에는 $q$만 정규화한 모델을 사용한다. 이 표의 내부 분산 비중은 물리 상태 복원률이나 계획 성공률과 다른 지표다. [Table 8, E.3](https://arxiv.org/html/2610.15267v1#A5.T8)

**시간 대칭성과 속도 부호.** tied 시간 합성곱 $[a,b,a]$는 입력 세 프레임을 역순으로 놓아도 같은 값을 낸다. 따라서 중심 자세가 같은 정방향·역방향 창을 구별할 수 없다. 원문 Eq.16의 일정 각속도 예제에서 $(\sin\theta,\cos\theta)$의 대칭 평균은

$$
\frac{b+2a\cos(\omega\,dt)}{b+2a}(\sin\theta,\cos\theta)
$$

가 된다. $\cos$가 짝함수여서 속도의 부호가 사라진다. Table 9의 초기 25-NN 결과는 다음과 같다. 두 모델의 예측기 문맥 길이는 모두 3이다.

| LeWM 인코더 | $R^2(\omega)$ | $R^2(\lvert\omega\rvert)$ | 두 값의 차이 |
| --- | --- | --- | --- |
| $W=1$ | −0.048 | 0.032 | 0.080 |
| $W=3$, tied | −0.115 | 0.921 | 1.036 |

이는 **인코더의 단일 목표 표현**에 대한 결과다. LeWM 예측기는 순서가 있는 과거 문맥에서 방향을 추론할 수 있다. 원문은 $W=1$의 한 스텝 뒤 속도 $R^2\approx0.90$도 보고한다. tied 창은 인접 잠재의 변화도 약화시키며, untied 창은 서로 다른 세 가중치로 이 대칭성을 푼다. [Table 9, Eq.16; Fig.15의 롤아웃은 1편](https://arxiv.org/pdf/2610.15267v1#page=32)

**목표 분포와 계산량.** Figure 16의 목표 50개 중 46개(92%)는 $|\omega^*|\ge0.25$ rad/s다. 동일한 속력에서 부호만 바꿔 도착하면 오차 $2|\omega^*|$가 속도 허용치 0.5 rad/s에 도달한다. 목표 각도는 정지한 위쪽 자세에 집중돼 있지 않다. 같은 재계획 설정에서 후보당 한 스텝은 LeHami 13.6 MFLOP, LeWM 21.7 MFLOP여서 비율은 약 0.63이고, 전체 계획 연산의 99% 이상이 예측 반복에 든다. [Fig.16, E.3 Planning compute](https://arxiv.org/pdf/2610.15267v1#page=33)

Table 10은 속도 성공을 속력 오차와 부호 일치로 나눠 각각 예산 내 최고 설정을 택한다. 이 둘을 곱해 본문의 전체 속도 성공률을 복원할 수는 없다.

| 모델 | 속력 성공률 | TFLOP, $R$ | 부호 성공률 | TFLOP, $R$ |
| --- | --- | --- | --- | --- |
| LeWM $W=1$ | 0.251±0.051 | 8.78, 15 | 0.656±0.017 | 5.85, 30 |
| LeWM $W=3$, tied | 0.638±0.008 | 13.65, 9 | 0.649±0.022 | 13.65, 9 |
| LeWM $W=3$, untied | **0.813±0.052** | 13.65, 9 | **0.962±0.017** | 13.65, 9 |
| LeHami | 0.547±0.022 | 10.98, 6 | 0.800±0.034 | 8.05, 9 |
| 수동 운동 | 0.300 | — | 0.680 | — |

LeHami와 untied LeWM만 부호 성공에서 수동 기준을 넘는다. 저자들은 같은 $R=9$에서 LeHami의 결합 성공 병목이 위치임도 보인다. $P(\theta)P(\omega\mid\theta)$는 LeHami 약 $0.48\times0.55=0.26$, tied LeWM 약 $0.63\times0.61=0.39$다. 이는 1편 Table 2의 모델별 **최고 설정**과 구별되는 고정 $R$ 비교다. [Table 10, E.3](https://arxiv.org/pdf/2610.15267v1#page=34)

마지막 Table 11은 길이·질량을 알아내지 못해서 계획이 실패하는지를 점검한다. 630개 훈련 에피소드의 초기 특징으로 프로브를 맞추고 135개 테스트 에피소드에서 고정해 읽는다. 초기에는 모든 값의 $R^2\ge0.999$다. 76스텝 뒤의 값은 다음과 같다.

| 프로브 | 모델 | 길이 $R^2$ | 질량 $R^2$ |
| --- | --- | --- | --- |
| ridge | LeHami | 0.900 [0.871, 0.951] | 0.977 [0.958, 0.993] |
| ridge | LeWM | 0.983 [0.976, 0.990] | 0.993 |
| 25-NN | LeHami | 0.997 [0.992, 1.000] | 1.000 |
| 25-NN | LeWM | 1.000 | 1.000 |

대괄호는 원문에 있는 seed 범위다. 고정 선형 프로브의 보정은 LeHami에서 더 흐트러지지만 국소 비선형 프로브는 두 매개변수를 계속 잘 읽는다. 질량 ridge 출력을 세 실제 질량 중 최근접 값으로 분류하면 두 모델 모두 99.3%이며, 다수 클래스 기준은 34.8%다. 저자들은 두 모델이 질량 정보를 담지 못한 것이 계획 차이의 원인은 아니라고 해석한다. [Table 11, E.3](https://arxiv.org/pdf/2610.15267v1#page=35)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 핵심 정리의 전단 합성 증명은 명확하다. 그러나 본문의 “완전한 물리 상태”나 “부호 있는 속도를 구조적으로 보장”한다는 표현은 그 정리보다 강하다. 가장 강한 지지 근거는 Table 1·Figure 8의 속도 복원, Table 9의 대칭성 분석이다. 부록 C.2는 물리 좌표와의 동일성을 가정하지 않고 C.4는 위치 의존 질량의 한계도 명시한다. 따라서 현재 근거에 맞는 결론은 **잠재 전이의 심플렉틱성은 구조적 보장, 물리 상태의 유용한 복원은 이 실험에서 확인된 성질**이라는 것이다. 실제 Table 8의 결합 정규화 실패는 같은 심플렉틱 구조에서도 움직임 정보가 약해질 수 있음을 보여 준다.

**리뷰 해석.** 부피 보존에서 정확한 에너지 보존을 유도하는 문장은 보정이 필요하다. Figures 9–11과 E.3 마지막 문단은 에너지를 보존한다고 표현하지만, C.4 Eq.10은 이미 적분 잔차 $r_t$를 남긴다. 일반 leapfrog는 심플렉틱이어도 원래 $H$를 매 스텝 정확히 보존하지 않는다. **설명용 반례**로 매끄러운 분리형 $H=(q^2+p^2)/2$, $h=1$, 초기 $(q,p)=(1,0)$를 택하면 한 스텝 뒤 $(1/2,-3/4)$이고 에너지는 $1/2$에서 $13/32$로 바뀐다. 이는 Proposition 2의 가정 안에 있으므로, 시각화는 학습 궤적의 경험적 에너지 구조로 읽어야 한다. 이 보정은 부피 보존 정리를 바꾸지 않는다.

**리뷰 해석.** Figure 4 오른쪽의 “누적 log-volume”에는 측정 정의의 차이가 있다. §3.2는 전체 전이의 행렬식으로 설명하지만, 공개된 그림 notebook은 LeWM의 `logdet_per_dim_newest`를 누적한다. E.2는 이 최신 블록이 전체 부피가 아니라고 구분하며, Figure 13이 전체 192차원 문맥의 수축도 따로 확인한다. 따라서 전체 부피 수축의 근거는 Figure 13에 두는 것이 정확하다. 이는 기준선 수축이라는 관찰을 뒤집는 결함은 아니지만 Figure 4만으로 전체 롤아웃 부피를 측정했다고 설명하면 범위가 넓어진다.

**리뷰 해석.** 정리의 행동 조건도 제어와 연결할 때 남는다. 외부에서 고정한 행동열에 대한 합성은 심플렉틱이지만, 상태에 의존하는 정책을 넣은 닫힌 고리(closed-loop) 전체는 추가 미분 항을 갖는다. $G(\mathbf z)=\Psi(\mathbf z,\pi(\mathbf z))$이면 $DG=D_{\mathbf z}\Psi+D_{\mathbf u}\Psi D\pi$다. 그러므로 다음 검증은 동일한 시간 정보와 목표 표현을 주는 기준선, 다양한 물리계, 그리고 닫힌 고리 계획 성능에서 이 제약의 실제 몫을 분리하는 실험이 적합하다. Table 4의 계산 이점·Figure 13의 진단은 각각 seed 1개이고, 계획 표는 테스트 목표에서 설정별 최고값을 택하므로 독립된 설정 선택과 불확실성 검증도 필요하다.

## 출처와 읽은 범위

- 원문: Yalovetzky, Balestriero, LeCun, *LeHami: Symplectic Latent Dynamics for World Models*, 최초 공개 2026-10-11, arXiv v1, 검토일 2026-10-26. [공식 원문](https://arxiv.org/abs/2610.15267v1).
- 범위: 부록 B–E 전문과 본문 정리·실험의 대응. 원문 그림·표·수식은 PDF와 저자 LaTeX를 기준으로 한다. 본문 §1–4는 [1편](/reviews/lehami-symplectic-world-models/)에서 다룬다.
- 추가 출처: [SRNN v2 §3.1–3.4](https://arxiv.org/html/1909.13334v2#S3)는 separable Hamiltonian·leapfrog·다중 스텝 학습의 배경이다. 공개 [LeHami 코드](https://github.com/rominay/LeHami/tree/f76f65f8b99afb6b7e3063f573645d3e2f290ede)의 `paper_figures/paper_figures.ipynb`와 저장된 계획 CSV는 Jacobian 정의와 집계 조건의 근거다.
- 그림: LeHami v1 Figures 8·13, 대표 썸네일 Figure 1, Yalovetzky et al., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). PNG 해상도 축소 또는 PDF 래스터화 후 WebP로 인코딩했으며 내용·레이블은 유지한다.
- 확인하지 않은 것: 전체 증명의 형식 검증, 모델 재학습, 독립 시뮬레이터 성능 재현.
