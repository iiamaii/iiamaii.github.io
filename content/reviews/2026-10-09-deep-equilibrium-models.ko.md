---
title: "Deep Equilibrium Models: 층을 쌓는 대신 평형 상태를 학습하기"
description: "DEQ의 고정점, Broyden 탐색과 암시적 미분을 예제로 설명하고, 주요 증명과 언어 모델 실험을 통해 메모리 이득·속도 비용·수렴 조건을 살펴본다."
date: "2026-10-09"
publishedAt: "2026-10-09T20:40:07+09:00"
updatedAt: "2026-10-23T15:26:15+00:00"
topics: ["language models", "fixed points", "implicit learning", "memory efficiency", "numerical optimization"]
visibility: "public"
lang: "ko"
translationKey: "deep-equilibrium-models"
paperTitle: "Deep Equilibrium Models"
paperPublishedDate: "2019-09-03"
authors: ["Shaojie Bai", "J. Zico Kolter", "Vladlen Koltun"]
year: "2019"
paperUrl: "https://arxiv.org/abs/1909.01377v2"
thumbnail: "/assets/reviews/deep-equilibrium-models/deq-core.svg"
thumbnailAlt: "공유 블록을 펼친 깊은 네트워크와 DEQ를 비교한 설명 그림. DEQ는 평형 상태를 수치적으로 찾은 뒤 그 상태에서 선형 방정식을 풀어 파라미터 gradient를 계산한다."
---

심층 평형 모델(Deep Equilibrium Model, DEQ)은 가중치를 공유한 블록의 평형 상태를 root-finding으로 직접 풀고 암시적 미분(implicit differentiation)으로 학습해, 깊이에 따른 활성값 저장을 상수로 줄인다. WikiText-103에서 adaptive embedding을 쓴 medium 모델끼리 비교하면 18층 Transformer-XL 대비 테스트 퍼플렉서티(perplexity, PPL)는 23.6에서 23.2로, 메모리(길이 150·batch 15, 단어 임베딩 제외 측정)는 9.0GB에서 3.7GB로 줄지만 학습 시간은 약 2.82배다. [원문 Table 3](https://arxiv.org/pdf/1909.01377v2#page=8) · [Table 4](https://arxiv.org/pdf/1909.01377v2#page=9) **리뷰 해석.** 같은 모델 크기에서 PPL을 유지하며 메모리를 크게 줄인다는 주장은 길이·batch를 통제한 Tables 2–3의 측정으로 성립하지만, 초록의 '기존 모델과 비슷한 계산 요구량'은 학습이 2.4–2.8배 느린 Table 4와 맞지 않는다.

<strong>대표 그림을 읽는 법.</strong> 위 줄은 같은 블록을 여러 층으로 펼치는 방식이다. 가운데 줄은 $f_\theta(z^\star_{1:T};x_{1:T})=z^\star_{1:T}$를 만족하는 끝점을 수치적으로 찾는다. 아래 줄은 끝점에서 선형 방정식을 풀어 gradient를 계산한다. 화살표는 각 줄의 계산 순서를 뜻한다. 반복 계산이 사라지는 것은 아니며, 순전파 탐색 경로 전체를 역전파하지 않는 것이 핵심이다. 이 글에서 직접 제작한 설명 그림이다. [크게 보기](/assets/reviews/deep-equilibrium-models/deq-core.svg)

원문 §1–6 순서로 읽도록 구성했다. Theorem 1의 Appendix A 증명은 §3.1.2에서, Theorem 2의 Appendix B와 표현력 보충인 Appendix C는 §3.2에서, Appendix D의 수렴 관찰은 §3 첫머리에서 함께 설명한다. 저자의 결론 뒤에는 Appendix E의 추가 설명과 리뷰 해석을 둔다.

## 1. 도입 (Introduction)

저자는 가중치를 공유한 깊은 시퀀스 모델이 최신 모델과 경쟁한다는 관찰에서 출발해, 같은 변환을 모든 층에 반복하면 그 극한은 무엇이고 어떻게 모델링할 수 있는지 묻는다. 답으로 제안하는 DEQ는 변환의 고정점을 임의의 root-finding 방법으로 직접 풀고, 암시적 미분으로 중간 활성값을 저장하지 않고 역전파해 한 층 분량의 상수 메모리로 학습한다. 저자는 이를 TrellisNet과 Universal Transformer 계열에 적용해, 상수 메모리로도 층 기반 모델과 대등하거나 약간 나은 정확도를 얻는다고 주장한다. [원문 §1](https://arxiv.org/pdf/1909.01377v2#page=1)

## 2. 배경 (Background)

원문 §2는 네 갈래의 선행연구를 정리한다. 깊은 시퀀스 모델은 인과성(causality), 즉 시점 $t$의 출력이 $x_{1:t}$에만 의존한다는 조건을 만족하는 함수로 정의되며, RNN·시간 합성곱 네트워크(temporal convolutional network, TCN)·self-attention의 세 계열과 이를 잇는 TrellisNet·Universal Transformer가 소개된다. 메모리 효율적 네트워크에서는 체크포인팅이 $L$층 네트워크의 메모리를 $O(\sqrt L)$로 줄이는 대신 순전파를 더 하고, 가역 네트워크(reversible network)는 이전 활성값을 다음 층에서 복원한다. 연속적 관점의 ODE 기반 방법은 adjoint로 잠재 궤적 전체를 역전파하는데, 저자는 DEQ가 quasi-Newton으로 평형 상태를 직접 풀고 그 해에 이른 경로와 무관하게 고정점을 통해 역전파한다는 점에서 다르다고 구분한다. [§2, pp. 2–3](https://arxiv.org/pdf/1909.01377v2#page=2)

마지막 갈래는 암시적 층(implicit layer)이다. 저자는 DEQ를 출력이 비선형 방정식의 해로 정의되는 단일 층으로도 볼 수 있다고 하며, 자신들이 아는 범위에서 가장 큰 규모의 실용적인 암시적 층 응용이라고 주장한다. 순환 역전파(recurrent backpropagation, RBP)와는 (1) 일반 네트워크의 깊이를 평형으로 대체하고 그 보편성을 증명한 점, (2) 고정 입력 RNN 밖의 최신 구조에 적용한 점, (3) 소규모에 머문 RBP와 달리 실제 과제 규모로 확장한 점에서 다르다고 밝힌다. [§2, p. 3](https://arxiv.org/pdf/1909.01377v2#page=3)

### 리뷰 보충: 배경 원전 비교

아래 표는 배경 원전 다섯 편의 해당 절과 DEQ §2–3을 대조한 리뷰 보충이다.

| 연구 | 함께 읽을 이유 | DEQ와의 차이 |
| --- | --- | --- |
| [Trellis Networks for Sequence Modeling](https://arxiv.org/pdf/1810.06682v2#page=3), §3–4 | 입력 주입과 깊이 방향 가중치 공유; convolution과 RNN의 연결 | 유한한 깊이의 네트워크를 펼쳐 계산한다. DEQ는 이 블록의 평형 상태를 출력으로 삼는다. |
| [Universal Transformers](https://arxiv.org/pdf/1807.03819v3#page=2), §2 | 같은 attention·transition을 깊이 방향으로 반복하고 계산량을 조절 | 유한한 반복 또는 위치별 halting을 사용하며 깊이 시간 표현도 있다. DEQ는 일정한 함수의 root와 잔차 중단 기준을 사용한다. |
| [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366v4#page=1), §1–2 | solver가 계산 경로를 정하고 adjoint로 메모리를 절약하는 관점 | ODE는 초기 상태에서 유한한 시각까지의 궤적을 계산한다. DEQ는 고정점 방정식을 풀고 그 주변의 선형 시스템을 미분한다. |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/pdf/1604.06174v2#page=4), §4 | 체크포인팅(checkpointing)으로 저장과 재계산을 교환하는 대안 | 원래 유한 네트워크의 계산을 유지하며 중간 결과를 재계산한다. DEQ는 출력 정의를 평형 상태로 바꾼다. |
| [Reviving and Improving Recurrent Back-Propagation](https://proceedings.mlr.press/v80/liao18c/liao18c.pdf#page=3), §3–4 | 고정점 gradient, 역행렬 조건, Neumann·CG 기반 계산 | 같은 암시적 미분 계보이며, DEQ와의 차이는 위 문단의 원문 §2가 든 세 가지다. |

## 3. Deep Equilibrium 시퀀스 모델 (The Deep Equilibrium Sequence Model)

일반적인 깊은 네트워크에서는 층마다 다른 파라미터를 사용한다. 층들이 가중치를 공유하도록 만들면 파라미터 수의 증가를 막을 수 있지만, 역전파를 위해 각 층의 중간 표현을 저장하는 문제는 남는다. DEQ가 출발하는 구조는 <strong>가중치 공유</strong>(weight tying)와 <strong>매 단계의 입력 주입</strong>(input injection)을 함께 사용한다. 입력 $x$를 고정한 채 은닉 상태(hidden state) $z$를 반복해서 갱신한다. [§3, 식 3](https://arxiv.org/pdf/1909.01377v2#page=3)

$$
\begin{aligned}
z^{[0]}_{1:T}&=0,\\
z^{[i+1]}_{1:T}&=f_\theta(z^{[i]}_{1:T};x_{1:T}).
\end{aligned}
$$

여기서 $i$는 문장의 토큰 순서가 아니라 <strong>계산 깊이</strong>다. $x$와 $z$는 시퀀스 전체의 표현일 수 있다. 자기회귀 언어 모델(autoregressive language model)이라면 시점 $t$의 출력이 미래 입력을 보지 못하도록 블록 안에 causal convolution이나 attention mask를 적용한다. [§2, §4](https://arxiv.org/pdf/1909.01377v2#page=3)

DEQ는 ‘몇 층을 통과한 상태를 쓸까?’ 대신 ‘같은 변환을 다시 적용해도 바뀌지 않는 상태를 쓸 수 있을까?’라고 묻는다. 저자는 특정 종류의 $f_\theta$에서 가중치 공유 모델이 깊이를 늘릴수록 고정점으로 수렴하는 경향이 있다는 가설을 세우고 관찰로 뒷받침한다. Appendix D는 학습된 TrellisNet과 가중치 공유 Transformer에서 길이 100–800의 테스트 시퀀스로 층 사이 활성값 차이가 줄어드는 모습을 보이고, Transformer는 실제 고정점 주변에서 진동하는 경우가 많다고 적는다. [§3; Appendix D, Figure 4](https://arxiv.org/pdf/1909.01377v2#page=14)

### 3.1 Deep Equilibrium 접근 (The Deep Equilibrium Approach)

고정점(fixed point) 또는 평형 상태(equilibrium state) $z^\star$는 다음 식을 만족한다.

$$
\begin{aligned}
z^\star_{1:T}&=f_\theta(z^\star_{1:T};x_{1:T}),\\
g_\theta(z_{1:T};x_{1:T})&=f_\theta(z_{1:T};x_{1:T})-z_{1:T},\\
g_\theta(z^\star_{1:T};x_{1:T})&=0.
\end{aligned}
$$

$f_\theta$를 계속 적용하는 것과 $g=0$이라는 방정식을 푸는 것은 같은 답을 목표로 할 수 있지만, 계산 방식과 수렴 성질은 다르다. DEQ는 두 번째 관점을 택한다. [§3, 식 4; §3.1.1, 식 5](https://arxiv.org/pdf/1909.01377v2#page=4)

설명을 위해 $f(z; x)=0.5z+0.75x$, $x=2$인 스칼라 예제를 생각해 보자. 이는 논문의 실험이 아니라 이 글의 예제다.

$$
\begin{gathered}
0\longrightarrow1.5\longrightarrow2.25\longrightarrow2.625\longrightarrow\cdots\longrightarrow3,\\
z^\star=0.5z^\star+1.5\quad\Longrightarrow\quad z^\star=3.
\end{gathered}
$$

정해진 횟수만큼 반복하면 $3$에 가까운 값을 얻는다. 평형 방정식으로 정의하면 출력은 $3$이다. 실제 DEQ에서는 이런 답을 닫힌 식으로 구할 수 없으므로 허용 오차와 반복 한도를 가진 수치 해법을 사용한다. ‘무한 깊이’는 출력의 정의를 설명하는 표현이지, 무한 번 계산하거나 유한한 비용으로 항상 정확한 답을 얻는다는 뜻은 아니다.

#### 3.1.1 순전파 (Forward Pass)

Newton 방법은 현재 잔차(residual) $g_\theta(z)$와 Jacobian(야코비안)을 이용해 다음 후보를 만든다. Jacobian은 상태의 각 성분을 조금 바꿨을 때 함수의 각 출력이 얼마나 바뀌는지를 담은 미분 행렬이다. 큰 시퀀스에서는 그 행렬과 역행렬을 직접 만드는 것이 비싸다. 논문은 <strong>Broyden 방법</strong>(Broyden’s method)으로 역 Jacobian의 근사를 갱신한다. [§3.1.1, 식 6; §3.1.3, 식 10](https://arxiv.org/pdf/1909.01377v2#page=4)

$$
\begin{aligned}
z^{[i+1]}_{1:T}&=z^{[i]}_{1:T}-\alpha B^{[i]}_{g_\theta}\,g_\theta(z^{[i]}_{1:T};x_{1:T}),\\
B^{[i]}_{g_\theta}&\approx\left(J_{g_\theta}\big|_{z^{[i]}_{1:T}}\right)^{-1}.
\end{aligned}
$$

$\alpha$는 보폭이며, 역 Jacobian 근사는 원문의 $B^{[i]}_{g_\theta}$를 사용한다. 매번 큰 역행렬을 새로 구하는 대신 저차원 갱신으로 관리하며, 논문은 이 근사를 $-I$에서 초기화한다. 잔차가 충분히 작아지거나 반복 한도에 도달하면 멈춘다. 따라서 수치 해법(solver) 설정도 모델을 실제로 실행하는 방법의 일부다.

중요한 차이는 <strong>단순 반복이 불안정해도 방정식의 해는 존재할 수 있다</strong>는 점이다. 예를 들어 $f(z)=1.2z+1$의 고정점은 $-5$지만, $z=0$에서 단순 반복하면 $1, 2.2, 3.64, \ldots$로 커진다. 해의 존재, 단순 반복의 수렴, root-finding solver의 성공은 별개의 문제다. 원문 §3.2도 평형을 안정적·효율적으로 찾으려면 일반적으로 $f_\theta$가 안정적이고 제약되어야 한다고 말한다. [§3.2, p. 5](https://arxiv.org/pdf/1909.01377v2#page=5)

#### 3.1.2 역전파 (Backward Pass)

순전파에서 끝점에 도달했더라도 학습하려면 파라미터 변화가 끝점에 미치는 영향을 알아야 한다. 여기서 <strong>암시적 미분</strong>(implicit differentiation)을 사용한다. 먼저 $z^\star_{1:T}=f_\theta(z^\star_{1:T};x_{1:T})$의 양변을 파라미터 $\theta$에 대해 미분한다. [Theorem 1, 식 8; Appendix A, 식 13–14](https://arxiv.org/pdf/1909.01377v2#page=13)

$$
\begin{aligned}
J_{f_\theta}&=\left.\frac{\partial f_\theta}{\partial z_{1:T}}\right|_{z^\star_{1:T}},\\
\frac{\mathrm dz^\star_{1:T}}{\mathrm d\theta}
&=J_{f_\theta}\frac{\mathrm dz^\star_{1:T}}{\mathrm d\theta}
+\frac{\partial f_\theta(z^\star_{1:T};x_{1:T})}{\partial\theta},\\
(I-J_{f_\theta})\frac{\mathrm dz^\star_{1:T}}{\mathrm d\theta}
&=\frac{\partial f_\theta(z^\star_{1:T};x_{1:T})}{\partial\theta},\\
\frac{\mathrm dz^\star_{1:T}}{\mathrm d\theta}
&=(I-J_{f_\theta})^{-1}\frac{\partial f_\theta(z^\star_{1:T};x_{1:T})}{\partial\theta}.
\end{aligned}
$$

원문 Theorem 1의 row-gradient 표기는 아래와 같다. $J_{g_\theta}=J_{f_\theta}-I$이므로 앞 유도와 같다. 이어지는 adjoint는 열벡터로 전치해 쓴다.

$$
\frac{\partial\ell}{\partial\theta}
=-\frac{\partial\ell}{\partial z^\star_{1:T}}
\left(J_{g_\theta}\big|_{z^\star_{1:T}}\right)^{-1}
\frac{\partial f_\theta(z^\star_{1:T};x_{1:T})}{\partial\theta}.
$$

손실을 $\ell$, 끝점에 대한 손실 기울기(gradient)를 $q=\nabla_{z^\star_{1:T}}\ell$라고 하자. 큰 역행렬을 직접 만들지 않고 다음 선형 방정식을 풀면 된다. 벡터는 열벡터로 표기했다.

$$
\begin{aligned}
q&=\nabla_{z^\star_{1:T}}\ell,\\
(I-J_{f_\theta})^\top v&=q,\\
\nabla_\theta\ell&=\left(\frac{\partial f_\theta}{\partial\theta}\right)^\top v.
\end{aligned}
$$

이것이 증명의 핵심이다. 끝점과 그 주변의 미분만으로 파라미터 민감도(sensitivity)를 구한다. 계산에서는 <strong>벡터–Jacobian 곱</strong>(vector–Jacobian product, VJP), 즉 Jacobian 전체를 만들지 않고 벡터에 대한 역방향 미분을 계산하는 연산을 사용한다. 순전파 solver의 모든 갱신을 자동미분 그래프로 보관할 필요가 없다. 출력 head나 손실이 파라미터에 직접 의존하는 경로는 그 미분을 따로 더해야 한다. [§3.1.2; §3.1.3, 식 11](https://arxiv.org/pdf/1909.01377v2#page=5)

앞의 스칼라 예제에서 목표값을 $1$, 손실을 $\frac12(z^\star-1)^2$로 두면 $q=2$, $J_{f_\theta}=0.5$, $v=4$다. $f(z;x)=az+bx$의 $a=0.5$, $b=0.75$에 대해 $\partial\ell/\partial a=12$, $\partial\ell/\partial b=8$이 된다.

이 식에는 조건이 있다. 함수가 해당 해 주변에서 미분 가능하고 <strong>$I-J_{f_\theta}$가 가역</strong>이어야 한다. 논문의 역행렬 표기도 이 조건을 전제한다. 거의 특이한 행렬(nearly singular matrix)이라면 작은 잔차에도 상태나 gradient 오차가 크게 증폭될 수 있다. 또한 식은 정확한 평형 상태에 대한 것이고 실제 구현은 근사 상태와 근사 선형 해를 쓴다. 원문도 Theorem 1의 해석적 gradient가 정확하려면 허용 오차 $\varepsilon$이 가능한 한 작아야 한다고 적는다. [§5, p. 9](https://arxiv.org/pdf/1909.01377v2#page=9)

#### 3.1.3 역 Jacobian 근사로 가속하기 (Accelerating DEQ by Approximating the Inverse Jacobian)

원문 식 (10)은 역행렬을 매번 계산하는 대신 이전 근사에 rank-one 갱신을 더한다. 두 차분 벡터는 연속한 탐색 상태에서 측정한다. 아래 분모가 0이 아니어야 식이 정의된다.

$$
\begin{aligned}
\Delta z^{[i+1]}&=z^{[i+1]}_{1:T}-z^{[i]}_{1:T},\\
\Delta g_\theta^{[i+1]}&=g_\theta(z^{[i+1]}_{1:T};x_{1:T})-g_\theta(z^{[i]}_{1:T};x_{1:T}),\\
B_{g_\theta}^{[i+1]}&=B_{g_\theta}^{[i]}
+\frac{\bigl(\Delta z^{[i+1]}-B_{g_\theta}^{[i]}\Delta g_\theta^{[i+1]}\bigr)
\Delta z^{[i+1]\top}B_{g_\theta}^{[i]}}
{\Delta z^{[i+1]\top}B_{g_\theta}^{[i]}\Delta g_\theta^{[i+1]}}.
\end{aligned}
$$

**리뷰 보충 유도.** 갱신식 오른쪽에 $\Delta g_\theta^{[i+1]}$를 곱하면 분자·분모의 스칼라가 소거되어 $B_{g_\theta}^{[i+1]}\Delta g_\theta^{[i+1]}=\Delta z^{[i+1]}$가 된다. 이것이 새 두 점의 변화량만 맞추는 할선 조건(secant condition)이며, 분모가 매우 작으면 갱신이 수치적으로 불안정해진다. 역전파의 선형 문제도 VJP와 간접 해법으로 풀어 명시적 Jacobian 저장을 피한다. [§3.1.3, Eqs. (10)–(11)](https://arxiv.org/pdf/1909.01377v2#page=4)

### 3.2 DEQ의 성질 (Properties of Deep Equilibrium Models)

원문 §3.2는 DEQ의 출력이 forward의 RootFind 선택과 무관하다는 점에서 출발해 메모리 비용, $f_\theta$의 선택, DEQ를 쌓는 문제를 차례로 논의한다.

#### 메모리와 함수 선택

DEQ의 메모리 주장은 <strong>깊이를 늘릴 때 필요한 활성값 저장량</strong>(activation storage)에 관한 것이다. 역전파에는 평형 시퀀스 $z^\star_{1:T}$, 입력 관련 변수 $x_{1:T}$, $f_\theta$만 있으면 되고, 역전파 선형 방정식에는 $N\times Td$ 크기의 VJP만 필요하므로 $N\times(Td)^2$ 크기의 Jacobian을 만들지 않는다($N$은 minibatch 크기). 저자는 이 상수 메모리 덕분에 여러 GPU나 half-precision·checkpointing이 필요하던 모델이 GPU 하나에 들어간다고 말한다. 메모리 이득은 $f_\theta$ 종류와 무관하며, 두 구현의 gated activation과 layer normalization은 출력 범위를 제한하는 안정적인 변환의 예로 제시된다. [§3.2, p. 5](https://arxiv.org/pdf/1909.01377v2#page=5)

시퀀스 길이·은닉 폭·batch에 비례하는 비용은 남는다. 저자도 가장 큰 Transformer-XL의 메모리 요구는 넓은 은닉 특징·batch·학습 시퀀스 길이 때문이며 DEQ로 줄지 않는다고 밝힌다. [§5, p. 8](https://arxiv.org/pdf/1909.01377v2#page=8)

#### Theorem 2와 부록 C의 Theorem 3: 표현 가능성과 폭

<strong>Theorem 2: 두 DEQ를 쌓은 결과를 하나의 더 큰 DEQ로 표현할 수 있다.</strong> 첫 평형 상태를 $w^{(1)}_{1:T}$, 둘째를 $w^{(2)}_{1:T}$라 하면, 결합 상태 $[w^{(1)}_{1:T};w^{(2)}_{1:T}]$에 두 갱신을 함께 넣는다. [Appendix B, 식 15–16, p. 13](https://arxiv.org/pdf/1909.01377v2#page=13)

$$
\Gamma_\Theta\!\left(\begin{bmatrix}w^{(1)}_{1:T}\\w^{(2)}_{1:T}\end{bmatrix};x_{1:T}\right)
=\begin{bmatrix}
f_{\theta^{[1]}}(w^{(1)}_{1:T};x_{1:T})\\
v_{\theta^{[2]}}(w^{(2)}_{1:T};w^{(1)}_{1:T})
\end{bmatrix}.
$$

$\Gamma_\Theta$의 고정점에서는 두 원래 방정식이 동시에 성립한다. 결합 상태의 마지막 부분을 읽으면 쌓은 모델과 같은 출력을 얻는다. 다만 은닉 차원은 두 상태 차원의 <strong>합</strong>이 된다. 본문 Theorem 2의 식은 $\mathsf{RootFind}(g^f_{\theta^{[2]}};\mathsf{RootFind}(g^v_{\theta^{[1]}};x_{1:T}))$로 $f$·$v$와 $\theta^{[1]}$·$\theta^{[2]}$의 짝이 정의와 엇갈려 인쇄되어 있고, Appendix B의 재진술(식 15)은 $g^v_{\theta^{[2]}}$, $g^f_{\theta^{[1]}}$로 정의와 맞는다. 이 글은 Appendix B를 따르며, 정리의 결론에는 영향이 없다.

<strong>Theorem 3: 유한한 비공유 네트워크도 더 넓은 공유 네트워크에 넣을 수 있다.</strong> 각 층의 상태를 큰 벡터의 서로 다른 구획에 넣고, 한 구획의 결과를 다음 구획으로 보내는 block-shift 구조를 만든다. 이를 반복하면 각 구획이 원래 층의 출력을 차례로 계산한다. [Appendix C, p. 14](https://arxiv.org/pdf/1909.01377v2#page=14)

이 증명 역시 폭을 늘리는 구성이며, 원문도 실제 DEQ에는 이 희소 구조가 아니라 dense 행렬 $W_z$, $W_x$를 쓴다고 밝힌다. 저자는 이 정리를 가중치 공유로 인한 눈에 띄는 표현력 손실이 없다는 근거로 든다.

**리뷰 보충 유도: Theorem 3의 귀납 단계.** 원문 식 (17)–(21)은 $W_z$의 바로 아래 대각 블록에 $W^{[1]},\ldots,W^{[L-1]}$를 놓고, $W_x$의 첫 블록에 $W^{[0]}$를 놓는다. $\sigma$와 $\tilde b$도 원래 층별 비선형·bias를 각 구획에 저장한다. 아래의 아래첨자 $k$는 그 구획을 가리키는 **리뷰 보조 첨자**다.

$$
\begin{aligned}
\tilde z^{[i+1]}_1&=\sigma^{[0]}(W^{[0]}x+b^{[0]}),\\
\tilde z^{[i+1]}_k&=\sigma^{[k-1]}(W^{[k-1]}\tilde z^{[i]}_{k-1}+b^{[k-1]}),
\quad 2\leq k\leq L,\\
\tilde z^{[i]}_k&=z^{[k]}\quad\text{for }i\geq k.
\end{aligned}
$$

첫 구획은 한 번의 갱신 뒤 원래 첫 층의 출력이 되고, 이후 바뀌지 않는다. $k-1$ 구획이 $i\geq k-1$에서 맞다고 가정하면 다음 갱신의 $k$ 구획은 원래 $k$층 출력을 계산한다. 따라서 $L$번 뒤 모든 구획이 원래 네트워크의 층 출력을 포함한다. 숨은 폭은 각 층 폭의 합으로 늘고 원래 각 층의 파라미터도 큰 공유 행렬 안에 들어간다. [Appendix C, Theorem 3](https://arxiv.org/pdf/1909.01377v2#page=14)

## 4. DEQ의 구현 사례 (Instantiations of DEQ)

원문 §4는 forward·backward 분석이 $f_\theta$의 내부 구조와 무관하다는 점을 전제로 두 구현을 보이고, 이를 통해 DEQ가 Transformer·RNN·TCN 세 계열과 모두 호환된다고 주장한다. [§4, Figure 1, p. 6](https://arxiv.org/pdf/1909.01377v2#page=6)

| 블록 | 상태를 갱신하는 내용 | 입력과 문맥 처리 |
| --- | --- | --- |
| DEQ-TrellisNet | causal 1D convolution과 LSTM 계열 gating | 공유 convolution에 입력의 투영을 매번 주입; 앞 구간의 상태와 padding으로 문맥 처리 |
| DEQ-Transformer | self-attention, feed-forward, residual, layer normalization | 입력을 Q/K/V 계산에 주입; causal mask·상대 위치 표현·이전 구간의 평형 상태 사용 |

Transformer 구현은 Transformer-XL의 문맥 처리와 상대 위치 표현을 활용한다. 상태 갱신은 깊이마다 같은 함수를 사용해야 하므로, 깊이별 시간 임베딩을 쓰는 반복 Transformer와는 갱신 함수가 다르다.

## 5. 실험 (Experiments)

실험은 복사 과제, Penn Treebank(PTB), WikiText-103(WT103)로 구성된다. 두 구현 모두 Broyden 방법을 쓰며, 학습 시 forward·backward의 허용 오차는 각각 $\varepsilon=\sqrt{T}\cdot10^{-5}$, $\sqrt{T}\cdot10^{-8}$이고 추론 시에는 $\sqrt{T}\cdot10^{-2}$로 완화한다. DEQ-Transformer는 WT103에서 학습·추론 모두 길이 150 시퀀스와 상대 위치 임베딩을 쓴다. PPL은 정답 토큰에 할당한 확률의 평균 음의 로그를 지수화한 값이며 낮을수록 좋다. [§5, p. 7; Appendix F](https://arxiv.org/pdf/1909.01377v2#page=7)

<strong>복사 과제(copy task)는 긴 간격을 넘어 정보를 보존하는 작은 스트레스 테스트다.</strong> 처음 10개 기호를 기다림 구간 뒤에 다시 출력한다. $T=400$ 설정의 실제 시퀀스 길이는 $T+20=420$이다. 약 14–16K 파라미터 모델에서 DEQ-Transformer의 보고 loss는 $3.5\times 10^{-6}$, TCN은 $2.7\times 10^{-5}$, LSTM은 $0.0501$, GRU는 $0.0491$이며, 저자는 LSTM·GRU 같은 순환 구조보다 기억 유지가 크게 낫다고 해석한다. [Table 1, p. 7; Appendix F, p. 16](https://arxiv.org/pdf/1909.01377v2#page=16)

다음은 원문의 주요 언어 모델 행을 선택해 다시 적은 표다. <strong>모든 메모리 값은 시퀀스 길이 150, batch size 15, 단어 임베딩 제외라는 공통 측정 조건</strong>이며, 원문 주석대로 실제 학습 하이퍼파라미터와는 다르다. Table 3의 Transformer-XL은 따로 표시한 행을 빼면 가중치를 공유하지 않는다. [Table 2 주석; Table 3, pp. 7–8](https://arxiv.org/pdf/1909.01377v2#page=7)

| 데이터 | 모델 | 전체 / 임베딩 제외 파라미터 | Test PPL ↓ | 메모리 GB ↓ |
| --- | --- | --- | --- | --- |
| PTB | TrellisNet 60층, 보조 손실, MoS 없음 | 24M / 20M | 57.0 | 8.5 |
| PTB | DEQ-TrellisNet | 24M / 20M | 57.1 | 1.2 |
| WT103 | TrellisNet 70층, 보조 손실 | 180M / 45M | 29.2 | 24.7 |
| WT103 | 위 TrellisNet + checkpointing | 180M / 45M | 29.2 | 5.2 |
| WT103 | DEQ-TrellisNet | 180M / 45M | 29.0 | 3.3 |
| WT103 | Transformer-XL 18층, medium, adaptive embedding | 110M / 72M | 23.6 | 9.0 |
| WT103 | DEQ-Transformer, medium, adaptive embedding | 110M / 70M | 23.2 | 3.7 |
| WT103 | Transformer-XL small, 가중치 공유 16층 | 138M / 4.5M | 34.9 | 6.8 |
| WT103 | DEQ-Transformer, small | 138M / 4.5M | 32.4 | 1.1 |

PTB에서는 PPL이 거의 같은 수준이고 메모리는 $8.5\to1.2\,\mathrm{GB}$, 약 85.9% 줄어든다. WT103의 Trellis 비교에서는 $24.7\to 3.3\,\mathrm{GB}$, 약 <strong>86.6%</strong> 절감이지만, checkpointing을 이미 적용한 $5.2\,\mathrm{GB}$와 비교하면 약 <strong>36.5%</strong>다. Transformer에서 같은 크기의 가중치 공유 기준선과 비교할 수 있는 행은 small 쌍뿐이며, PPL은 $34.9\to32.4$로 낮아지고 메모리는 $6.8\to1.1\,\mathrm{GB}$, 약 <strong>83.8%</strong> 줄어든다. 가중치를 공유하지 않는 Transformer-XL과의 세 쌍은 medium 16층 $8.5\to2.7\,\mathrm{GB}$(68.2%), small 4층 $4.8\to1.1\,\mathrm{GB}$(77.1%), adaptive embedding 쌍 $9.0\to 3.7\,\mathrm{GB}$(<strong>58.9%</strong>)다. 모두 표의 반올림된 값으로 계산했다. [Table 3](https://arxiv.org/pdf/1909.01377v2#page=8)

원문 §5는 두 구현 모두에서 메모리가 "over 80% (up to 88%)" 줄고, 가중치를 공유하지 않는 기준선과 비교해도 "a reduction of over 60%"라고 적는다. **리뷰 해석.** 80% 이상은 가중치 공유 기준선 세 쌍(85.9%·86.6%·83.8%)에서 성립하지만, 인쇄된 값으로 계산한 최대 절감률은 86.6%이고 반올림 범위를 최대로 잡아도 약 86.9%여서 초록과 §5의 88%는 표에서 재현되지 않는다. 60% 이상도 adaptive embedding 쌍의 58.9%에서는 성립하지 않는다. 원문 안에서 88%를 만드는 다른 행이나 측정은 찾지 못했으며, 두 차이 모두 큰 폭의 메모리 절감이라는 결론을 바꾸지 않고 최댓값·하한의 표현에만 영향을 준다.

PTB Table 2의 DARTS는 PPL $55.7$, WT103 Table 3의 훨씬 큰 Transformer-XL은 $18.7$로 DEQ보다 낮다. 저자의 비교 단위도 Table 3 캡션대로 같은 모델 크기의 최신 모델이다. [Tables 2–3](https://arxiv.org/pdf/1909.01377v2#page=8)

### Figure 2: 학습 중 탐색 계산량

초기화에 대해 저자는 평형을 quasi-Newton으로 안정적으로 근사하려면 가중치 행렬의 operator norm이 작은 상태에서 시작해야 한다고 말한다. 두 구현은 특정 초기화에 민감하지 않았는데, $\sigma$·$\tanh$ 같은 비선형과 LayerNorm도 $f_\theta$를 수축적으로 만드는 데 도움이 된다고 보며, 실제로는 $\mathcal N(0,0.05)$에서 초기화한다. [§5, p. 8](https://arxiv.org/pdf/1909.01377v2#page=8)

<figure class="review-figure" id="paper-figure-2">
<a href="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" width="1732" height="509" alt="원문 Figure 2. 왼쪽은 학습 epoch에 따른 시간 단계당 Broyden 반복 수, 오른쪽은 함수 평가 수에 따른 평형 잔차의 로그 축 그래프다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 2 · 학습이 진행되면 평형 탐색도 어려워질 수 있다</span>원문 v2 p. 8의 두 패널을 주변 본문에서 분리했다. 축·범례·곡선은 유지했다. <span class="figure-links"><a href="https://arxiv.org/pdf/1909.01377v2#page=8">Shaojie Bai, J. Zico Kolter, Vladlen Koltun, 2019, Figure 2</a> · <a href="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" target="_blank" rel="noopener noreferrer">크게 보기</a> · 검토를 위한 그림 인용; 권리는 원저자에게 있음</span></figcaption>
</figure>

<strong>왼쪽은 계산 부담을 읽는 그래프다.</strong> 가로축은 training epoch이고, 세로축은 원문 정의상 전체 Broyden 반복 수를 시퀀스 길이로 나눈 값이다. 이 그림의 시퀀스 길이는 150이다(그림 제목). 파란 선은 forward, 붉은 선은 backward이며 학습이 진행될수록 두 계산량이 증가한다. 저자는 학습이 $J_{f_\theta}$의 operator norm을 키워 고정점을 풀기 어렵게 만들 가능성을 원인으로 들고, backward가 훨씬 적은 것은 식 (11)의 선형 시스템이 단순하기 때문이라고 설명한다. [Figure 2, §5, p. 8](https://arxiv.org/pdf/1909.01377v2#page=8)

<strong>오른쪽은 상태의 잔차를 읽는 그래프다.</strong> 가로축은 함수 평가 횟수이고, 세로축은 $\|f(z)-z\|_2$의 로그 축이다. 그림에서 읽으면 epoch 1에서는 가중치 공유 Transformer의 단순 반복과 DEQ 모두 약 $10^{-4}$까지 내려가지만, DEQ가 약 절반의 함수 평가로 도달한다. Epoch 12에서는 단순 반복이 약 $2\times10^{1}$ 부근에서 진동하는 반면 DEQ는 약 200회 평가 안에 $10^{-4}$ 근처에 이른다. 저자는 이를 DEQ가 거의 언제나 시퀀스 수준 고정점에 수렴하고 원래의 가중치 공유 Transformer보다 훨씬 효율적이라는 근거로 든다.

### 실행 시간 (Table 4)

DEQ의 실행 시간은 평형에 이르는 Broyden 단계 수에 크게 좌우되며, 저자는 DEQ가 대체로 층 기반 네트워크보다 느리다고 밝힌다. Table 4의 값은 `DEQ 시간 / 비교 모델 시간`이므로 1보다 크면 느리다. [Table 4, p. 9](https://arxiv.org/pdf/1909.01377v2#page=9)

| 비교 모델, WT103 | 학습 시간 비율 | 추론 시간 비율 |
| --- | --- | --- |
| 18층 Transformer | 2.82× | 1.76× |
| 70층 TrellisNet | 2.40× | 1.64× |

### Figure 3: 정밀도와 품질

<figure class="review-figure" id="paper-figure-3">
<a href="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" width="1737" height="469" alt="원문 Figure 3. 왼쪽은 forward 잔차 허용치와 validation PPL, 오른쪽은 forward 반복 한도와 validation PPL의 관계다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 3 · 너무 일찍 멈추면 품질이 낮아진다</span>원문 v2 p. 9의 두 패널을 분리했으며 데이터와 원문 표기를 유지했다. <span class="figure-links"><a href="https://arxiv.org/pdf/1909.01377v2#page=9">Shaojie Bai, J. Zico Kolter, Vladlen Koltun, 2019, Figure 3</a> · <a href="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" target="_blank" rel="noopener noreferrer">크게 보기</a> · 검토를 위한 그림 인용; 권리는 원저자에게 있음</span></figcaption>
</figure>

왼쪽은 forward의 잔차 허용치를 키울수록 어디서 품질이 무너지는지를 보여준다. 가로축은 로그 축이고 값이 작을수록 엄격하다. 작은 허용치 구간에서는 validation PPL이 비슷하지만 너무 크게 완화하면 급격히 나빠진다. 오른쪽은 반복 한도를 늘릴수록 PPL이 개선되고, 추가 계산의 이득이 작아지는 모습을 보여준다. 음영은 원문에 있지만 그 집계 의미가 명시되지 않았다. [Figure 3, §5, p. 9](https://arxiv.org/pdf/1909.01377v2#page=9)

이 그림의 실험은 adaptive embedding 없는 medium DEQ-Transformer이고 지표는 validation PPL이다. 본문은 일반적으로 $\varepsilon <0.1$ 또는 반복 한도 30(<strong>길이 75 시퀀스</strong> 기준)이면 경쟁력 있는 성능에 충분하다고 설명하며, 허용 오차가 너무 크면 정확도가 빠르게 무너지는 것을 평형의 나쁜 추정이 성능을 해친다는 근거로 든다.

## 6. 결론 (Conclusion)

저자는 시퀀스 수준 고정점을 직접 풀고 최적화하는 DEQ가 학습 시 $O(1)$ 메모리만 쓰고, forward의 root solver 선택과 무관하며, 크게 다른 구조를 포괄할 만큼 유연하다고 결론낸다. 실험에서는 시간적 기억 유지가 좋고, 실제 대규모 시퀀스 과제로 확장되며, 최신 방법과 대등하거나 약간 낫다고 요약한다. [원문 §6](https://arxiv.org/pdf/1909.01377v2#page=9)

## 부록 E: DEQ에 대한 추가 설명 (More Remarks on DEQ)

Appendix E는 수학적 식만으로 드러나지 않는 실무 조건을 정리한다. [Appendix E, p. 15](https://arxiv.org/pdf/1909.01377v2#page=15)

- <strong>다른 기법의 적용.</strong> layer normalization과 weight normalization은 그대로 쓸 수 있지만, 층이 없으므로 중간 층 보조 손실을 어떻게 적용할지는 분명하지 않다. 드롭아웃은 TrellisNet을 따라 모든 수준에 같은 마스크를 쓰며, 드롭아웃을 넣으면 quasi-Newton의 수렴이 느려진다. Weight normalization은 더 안정적인 파라미터를 찾고 Figure 2의 Broyden 반복 증가를 늦춘다.
- <strong>Minibatch 안의 불균형.</strong> 시퀀스마다 평형에 이르는 반복 수가 달라 빨리 수렴한 시퀀스가 느린 시퀀스를 기다린다. 저자는 그 규모가 비교적 작지만 때로 GPU 활용이 비효율적일 수 있다고 적는다.
- <strong>얕은 모델로 warm-up.</strong> 2층 정도의 얕은 가중치 공유 스택을 먼저 학습해 그 파라미터로 DEQ를 초기화하면 학습이 빨라진다.
- <strong>부분 시퀀스로 학습.</strong> 길이 1,000을 넘는 시퀀스에서는 특히 학습 초기에 forward 고정점을 정확히 풀기 어려워, 시퀀스를 몇 개의 부분으로 나누어 풀고 backward에도 같은 분할을 쓸 수 있다. 메모리 비용은 전체 시퀀스에 $f_\theta$를 한 번 적용하는 것과 같다.

<span id="리뷰-해석과-후속-질문" aria-hidden="true"></span>

## 리뷰 해석과 남은 질문

**리뷰 해석.** DEQ는 깊이를 ‘저장해야 하는 층의 목록’에서 ‘풀어야 하는 상태 방정식’으로 바꾼다. 메모리 주장의 가장 직접적인 근거는 같은 파라미터의 가중치 공유 small Transformer-XL과의 비교로, PPL이 34.9에서 32.4로 낮아지면서 메모리가 6.8GB에서 1.1GB로 준다. Checkpointing을 적용한 70층 TrellisNet보다도 메모리가 36.5% 적어, 저장과 재계산을 맞바꾸는 기존 기법과의 비교도 측정되어 있다. 대가는 시간이다. 본문은 DEQ가 대체로 층 기반 모델보다 느리다고 인정하고, Figure 2에서는 학습이 진행될수록 Broyden 반복이 늘어난다. 평형 수렴은 저자가 관찰한 설정에서 거의 언제나 성립하지만, 그 안정성은 작은 operator norm의 초기화, 정규화, weight normalization 같은 실무 조건에 기댄다. 초록의 '같은 파라미터 수에서 자주 성능을 개선한다'는 WT103 비교에서 방향은 맞지만, small 쌍을 빼면 PPL 차이가 0.1–0.4이고 반복 실행의 분산은 보고되지 않았다.

남은 질문은 <strong>어떤 함수에 어떤 조건으로 쓸 만한 평형이 생기며, 어느 정밀도로 풀 때 품질·메모리·시간의 교환이 유리한가</strong>다. 후속 연구인 [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656)는 다중 해상도 비전으로 적용을 넓히고, [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342)는 안정성과 계산 효율을 직접 다룬다. 두 후속 논문의 인용 범위는 공식 초록·서지다.

## 출처와 읽은 범위

- 원문: *Deep Equilibrium Models* · Shaojie Bai et al. · 최초 공개 2019-09-03 · [검토 버전](https://arxiv.org/abs/1909.01377v2) · 검토일 2026-10-23.
- 읽은 범위: v2 본문 §1–6·Appendix A–F. A–C의 정리·증명은 §3의 해당 설명에 연결한다.
- 그림 출처: 대표 그림은 자체 제작물(CC BY 4.0). Figures 2·3은 해설을 위해 원문 일부를 인용하며 원저자 권리를 유지한다. arXiv 비독점 배포 라이선스는 일반 재사용 허가가 아니다. [원문](https://arxiv.org/abs/1909.01377v2).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
