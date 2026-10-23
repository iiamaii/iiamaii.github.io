---
title: "Deep Equilibrium Models: 층을 쌓는 대신 평형 상태를 학습하기"
description: "DEQ의 고정점, Broyden 탐색과 암시적 미분을 예제로 설명하고, 주요 증명과 언어 모델 실험을 통해 메모리 이득·속도 비용·수렴 조건을 살펴본다."
date: "2026-10-09"
publishedAt: "2026-10-09T20:40:07+09:00"
updatedAt: "2026-10-23T08:56:14+00:00"
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

심층 평형 모델(Deep Equilibrium Model, DEQ)은 공유 블록의 평형 상태를 직접 풀고 암시적 미분(implicit differentiation)으로 학습해 깊이에 따른 활성값 저장을 줄인다. WikiText-103의 medium attention 비교에서 퍼플렉서티(perplexity, PPL)는 23.6에서 23.2로, 메모리는 9.0GB에서 3.7GB로 바뀌지만 학습 시간은 약 2.82배다. [원문 Table 1·§5](https://arxiv.org/pdf/1909.01377v2#page=6) **리뷰 해석.** 메모리 절감은 분명하며, 실제 선택은 solver의 시간·수렴 정밀도와 함께 판단해야 한다.

<strong>대표 그림을 읽는 법.</strong> 위 줄은 같은 블록을 여러 층으로 펼치는 방식이다. 가운데 줄은 $f_\theta(z^\star_{1:T};x_{1:T})=z^\star_{1:T}$를 만족하는 끝점을 수치적으로 찾는다. 아래 줄은 끝점에서 선형 방정식을 풀어 gradient를 계산한다. 화살표는 각 줄의 계산 순서를 뜻한다. 반복 계산이 사라지는 것은 아니며, 순전파 탐색 경로 전체를 역전파하지 않는 것이 핵심이다. 이 글에서 직접 제작한 설명 그림이다. [크게 보기](/assets/reviews/deep-equilibrium-models/deq-core.svg)

원문 §1–6 순서로 읽도록 구성했다. Theorem 1의 Appendix A 증명은 §3.1.2에서, Theorem 2의 Appendix B와 표현력 보충인 Appendix C는 §3.2에서 함께 설명한다. 저자의 결론 뒤에는 구현 부록과 리뷰 해석을 구분한다.

## 1. 도입 (Introduction)

저자는 시퀀스 전체의 평형 상태를 직접 계산하고 미분하는 대안을 제안한다. 핵심 질문은 깊이 방향 가중치 공유만으로 해결되지 않는 활성값 저장 비용을 끝점 방정식으로 줄일 수 있는가이다. 아래 §2의 배경을 거쳐 §3에서 출력 정의와 계산 방법을 분리한다. [원문 §1](https://arxiv.org/pdf/1909.01377v2#page=1)

## 2. 배경 (Background)

원문 §2는 인과적 시퀀스 모델, 메모리 효율적 네트워크, 연속적 관점의 세 흐름을 설명한다. 인과성(causality)은 시점 $t$의 출력이 $x_{1:t}$에만 의존한다는 조건이다. 체크포인팅은 원래 경로를 재계산하고, 가역 네트워크(reversible network)는 이전 활성값을 복원한다. Neural ODE는 유한 시간의 궤적을 풀며, DEQ는 평형 방정식을 출력으로 정의한다. 아래 표는 이 배경을 원전과 연결한 리뷰의 비교다. [§2, pp. 2–3](https://arxiv.org/pdf/1909.01377v2#page=2)

핵심을 이해하는 데 필요한 선행연구 다섯 편을 서로 다른 역할로 읽었다. 아래 표의 비교는 각 원문의 해당 절과 DEQ의 §2–3을 기준으로 한다.

### 리뷰 보충: 배경 원전 비교

| 연구 | 함께 읽을 이유 | DEQ와의 차이 |
| --- | --- | --- |
| [Trellis Networks for Sequence Modeling](https://arxiv.org/pdf/1810.06682v2#page=3), §3–4 | 입력 주입과 깊이 방향 가중치 공유; convolution과 RNN의 연결 | 유한한 깊이의 네트워크를 펼쳐 계산한다. DEQ는 이 블록의 평형 상태를 출력으로 삼는다. |
| [Universal Transformers](https://arxiv.org/pdf/1807.03819v3#page=2), §2 | 같은 attention·transition을 깊이 방향으로 반복하고 계산량을 조절 | 유한한 반복 또는 위치별 halting을 사용하며 깊이 시간 표현도 있다. DEQ는 일정한 함수의 root와 잔차 중단 기준을 사용한다. |
| [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366v4#page=1), §1–2 | solver가 계산 경로를 정하고 adjoint로 메모리를 절약하는 관점 | ODE는 초기 상태에서 유한한 시각까지의 궤적을 계산한다. DEQ는 고정점 방정식을 풀고 그 주변의 선형 시스템을 미분한다. |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/pdf/1604.06174v2#page=4), §4 | 체크포인팅(checkpointing)으로 저장과 재계산을 교환하는 대안 | 원래 유한 네트워크의 계산을 유지하며 중간 결과를 재계산한다. DEQ는 출력 정의를 평형 상태로 바꾼다. |
| [Reviving and Improving Recurrent Back-Propagation](https://proceedings.mlr.press/v80/liao18c/liao18c.pdf#page=3), §3–4 | 고정점 gradient, 역행렬 조건, Neumann·CG 기반 계산 | 암시적 미분에는 이 논문과 더 오래된 RBP 계보가 있다. DEQ는 이를 현대적인 깊은 시퀀스 블록의 대체 방식으로 구성하고 대규모 언어 모델에서 평가한다. |

따라서 DEQ의 새로움을 ‘고정점 미분을 처음 발명했다’로 설명하면 선행연구를 놓친다. 이 글의 해석으로는 <strong>깊은 가중치 공유 모델의 끝점을 직접 학습하는 설계, 실용적인 root 탐색과 implicit backward의 결합, 언어 모델 규모의 검증</strong>이 중요한 기여다.

## 3. Deep Equilibrium 시퀀스 모델 (The Deep Equilibrium Sequence Model)

일반적인 깊은 네트워크에서는 층마다 다른 파라미터를 사용한다. 층들이 가중치를 공유하도록 만들면 파라미터 수의 증가를 막을 수 있지만, 역전파를 위해 각 층의 중간 표현을 저장하는 문제는 남는다. DEQ가 출발하는 구조는 <strong>가중치 공유</strong>(weight tying)와 <strong>매 단계의 입력 주입</strong>(input injection)을 함께 사용한다. 입력 $x$를 고정한 채 은닉 상태(hidden state) $z$를 반복해서 갱신한다. [§2–3, 식 3–4](https://arxiv.org/pdf/1909.01377v2#page=3)

$$
\begin{aligned}
z^{[0]}_{1:T}&=0,\\
z^{[i+1]}_{1:T}&=f_\theta(z^{[i]}_{1:T};x_{1:T}).
\end{aligned}
$$

여기서 $i$는 문장의 토큰 순서가 아니라 <strong>계산 깊이</strong>다. $x$와 $z$는 시퀀스 전체의 표현일 수 있다. 자기회귀 언어 모델(autoregressive language model)이라면 시점 $t$의 출력이 미래 입력을 보지 못하도록 블록 안에 causal convolution이나 attention mask를 적용한다. 전체 시퀀스의 상태를 계산한다고 해서 미래 토큰의 정답을 이용하는 것은 아니다. [§2.2, §4](https://arxiv.org/pdf/1909.01377v2#page=3)

DEQ는 ‘몇 층을 통과한 상태를 쓸까?’ 대신 ‘같은 변환을 다시 적용해도 바뀌지 않는 상태를 쓸 수 있을까?’라고 묻는다.

### 3.1 Deep Equilibrium 접근 (The Deep Equilibrium Approach)

고정점(fixed point) 또는 평형 상태(equilibrium state) $z^\star$는 다음 식을 만족한다.

$$
\begin{aligned}
z^\star_{1:T}&=f_\theta(z^\star_{1:T};x_{1:T}),\\
g_\theta(z_{1:T};x_{1:T})&=f_\theta(z_{1:T};x_{1:T})-z_{1:T},\\
g_\theta(z^\star_{1:T};x_{1:T})&=0.
\end{aligned}
$$

$f_\theta$를 계속 적용하는 것과 $g=0$이라는 방정식을 푸는 것은 같은 답을 목표로 할 수 있지만, 계산 방식과 수렴 성질은 다르다. DEQ는 두 번째 관점을 택한다. [§3.1, 식 5–7](https://arxiv.org/pdf/1909.01377v2#page=4)

설명을 위해 $f(z; x)=0.5z+0.75x$, $x=2$인 스칼라 예제를 생각해 보자. 이는 논문의 실험이 아니라 이 글의 예제다.

$$
\begin{gathered}
0\longrightarrow1.5\longrightarrow2.25\longrightarrow2.625\longrightarrow\cdots\longrightarrow3,\\
z^\star=0.5z^\star+1.5\quad\Longrightarrow\quad z^\star=3.
\end{gathered}
$$

정해진 횟수만큼 반복하면 $3$에 가까운 값을 얻는다. 평형 방정식으로 정의하면 출력은 $3$이다. 실제 DEQ에서는 이런 답을 닫힌 식으로 구할 수 없으므로 허용 오차와 반복 한도를 가진 수치 해법을 사용한다. ‘무한 깊이’는 출력의 정의를 설명하는 표현이지, 무한 번 계산하거나 유한한 비용으로 항상 정확한 답을 얻는다는 뜻은 아니다.

#### 3.1.1 순전파 (Forward Pass)

Newton 방법은 현재 잔차(residual) $g_\theta(z)$와 야코비안(Jacobian)을 이용해 다음 후보를 만든다. Jacobian은 상태의 각 성분을 조금 바꿨을 때 함수의 각 출력이 얼마나 바뀌는지를 담은 미분 행렬이다. 큰 시퀀스에서는 그 행렬과 역행렬을 직접 만드는 것이 비싸다. 논문은 <strong>Broyden 방법</strong>(Broyden’s method)으로 역 Jacobian의 근사를 갱신한다. [§3.1.1, 식 6–7, 10, p. 4](https://arxiv.org/pdf/1909.01377v2#page=4)

$$
\begin{aligned}
z^{[i+1]}_{1:T}&=z^{[i]}_{1:T}-\alpha B^{[i]}_{g_\theta}\,g_\theta(z^{[i]}_{1:T};x_{1:T}),\\
B^{[i]}_{g_\theta}&\approx\left(J_{g_\theta}\big|_{z^{[i]}_{1:T}}\right)^{-1}.
\end{aligned}
$$

$\alpha$는 보폭이며, 역 Jacobian 근사는 원문의 $B^{[i]}_{g_\theta}$를 사용한다. 매번 큰 역행렬을 새로 구하는 대신 저차원 갱신으로 관리하며, 논문은 이 근사를 $-I$에서 초기화한다. 잔차가 충분히 작아지거나 반복 한도에 도달하면 멈춘다. 따라서 수치 해법(solver) 설정도 모델을 실제로 실행하는 방법의 일부다.

중요한 차이는 <strong>단순 반복이 불안정해도 방정식의 해는 존재할 수 있다</strong>는 점이다. 예를 들어 $f(z)=1.2z+1$의 고정점은 $-5$지만, $z=0$에서 단순 반복하면 $1, 2.2, 3.64, \ldots$로 커진다. 해의 존재, 단순 반복의 수렴, 근 찾기 알고리즘(root solver)의 성공은 별개의 문제다. DEQ도 모든 입력에서 solver가 성공한다는 보장을 이 논문에서 얻지는 않는다. [§3.2, Appendix D, pp. 5, 14–15](https://arxiv.org/pdf/1909.01377v2#page=14)

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

이것이 증명의 핵심이다. 끝점과 그 주변의 미분만으로 파라미터 민감도(sensitivity)를 구한다. 계산에서는 <strong>벡터–야코비안 곱</strong>(vector–Jacobian product, VJP), 즉 Jacobian 전체를 만들지 않고 벡터에 대한 역방향 미분을 계산하는 연산을 사용한다. 순전파 solver의 모든 갱신을 자동미분 그래프로 보관할 필요가 없다. 출력 head나 손실이 파라미터에 직접 의존하는 경로는 그 미분을 따로 더해야 한다. [§3.1.2, 식 11, p. 5](https://arxiv.org/pdf/1909.01377v2#page=5)

앞의 스칼라 예제에서 목표값을 $1$, 손실을 $\frac12(z^\star-1)^2$로 두면 $q=2$, $J_{f_\theta}=0.5$, $v=4$다. $f(z;x)=az+bx$의 $a=0.5$, $b=0.75$에 대해 $\partial\ell/\partial a=12$, $\partial\ell/\partial b=8$이 된다.

이 식에는 조건이 있다. 함수가 해당 해 주변에서 미분 가능하고 <strong>$I-J_{f_\theta}$가 가역</strong>이어야 한다. 논문의 역행렬 표기도 이 조건을 전제한다. 거의 특이한 행렬(nearly singular matrix)이라면 작은 잔차에도 상태나 gradient 오차가 크게 증폭될 수 있다. 또한 식은 정확한 평형 상태에 대한 식이고, 실제 구현은 근사 상태와 근사 선형 해를 사용한다. 잔차가 작다는 사실만으로 gradient가 언제나 정확하다고 말할 수는 없다. [Appendix A; RBP §3.2](https://proceedings.mlr.press/v80/liao18c/liao18c.pdf#page=3)

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

**리뷰 보충 확인.** 갱신식 오른쪽에 $\Delta g_\theta^{[i+1]}$를 곱하면 분자·분모의 스칼라가 소거되어 $B_{g_\theta}^{[i+1]}\Delta g_\theta^{[i+1]}=\Delta z^{[i+1]}$가 된다. 이것이 새 두 점의 변화량을 맞추는 secant condition이다. 전체 역 Jacobian이 정확해졌다는 뜻은 아니다. 분모가 매우 작을 때의 수치 안정성과 solver 중단 조건은 여전히 중요하다. 역전파의 선형 문제도 VJP와 간접 해법으로 풀어 명시적 Jacobian 저장을 피한다. [§3.1.3, Eqs. (10)–(11)](https://arxiv.org/pdf/1909.01377v2#page=4)

### 3.2 DEQ의 성질 (Properties of Deep Equilibrium Models)

출력 정의에서 따라오는 메모리·표현 가능성의 범위를 구분한다.

#### 메모리와 함수 선택

DEQ의 메모리 주장은 <strong>깊이를 늘릴 때 필요한 활성값 저장량</strong>(activation storage)에 관한 것이다. 상태를 찾는 경로 전체 대신 끝점, 입력, 한 번의 국소 계산 그래프 등을 유지한다. 그러나 시퀀스 길이, 은닉 폭, batch, 파라미터, 옵티마이저 상태(optimizer state), attention 행렬과 solver 작업 공간은 여전히 메모리를 사용한다. Broyden 근사의 이력 저장도 구현과 설정에 따라 비용이 있다. [§3.2, p. 5](https://arxiv.org/pdf/1909.01377v2#page=5)

따라서 ‘깊이 방향으로 상수’와 ‘모든 모델에서 GPU 사용량이 일정하다’는 서로 다른 주장이다. 길이가 긴 입력이나 폭이 큰 attention 블록이 공짜가 되지 않는다. 아래 실험의 절감률 역시 임베딩을 제외한 특정 비교 조건에서 읽어야 한다.

#### Theorem 2와 부록 C의 Theorem 3: 표현 가능성과 폭

<strong>Theorem 2: 두 DEQ를 쌓은 결과를 하나의 더 큰 DEQ로 표현할 수 있다.</strong> 첫 평형 상태를 $w^{(1)}_{1:T}$, 둘째를 $w^{(2)}_{1:T}$라 하면, 결합 상태 $[w^{(1)}_{1:T};w^{(2)}_{1:T}]$에 두 갱신을 함께 넣는다. [Appendix B, 식 15–16, p. 13](https://arxiv.org/pdf/1909.01377v2#page=13)

$$
\Gamma_\Theta\!\left(\begin{bmatrix}w^{(1)}_{1:T}\\w^{(2)}_{1:T}\end{bmatrix};x_{1:T}\right)
=\begin{bmatrix}
f_{\theta^{[1]}}(w^{(1)}_{1:T};x_{1:T})\\
v_{\theta^{[2]}}(w^{(2)}_{1:T};w^{(1)}_{1:T})
\end{bmatrix}.
$$

$\Gamma_\Theta$의 고정점에서는 두 원래 방정식이 동시에 성립한다. 결합 상태의 마지막 부분을 읽으면 쌓은 모델과 같은 출력을 얻는다. 다만 은닉 차원은 두 상태 차원의 <strong>합</strong>이 된다. 같은 폭의 블록 하나가 항상 같은 표현력을 갖거나, 합친 solver가 더 빠르다는 증명은 아니다. 본문의 함수 첨자 표기보다 구성이 명확한 Appendix B를 기준으로 읽었다.

<strong>Theorem 3: 유한한 비공유 네트워크도 더 넓은 공유 네트워크에 넣을 수 있다.</strong> 각 층의 상태를 큰 벡터의 서로 다른 구획에 넣고, 한 구획의 결과를 다음 구획으로 보내는 block-shift 구조를 만든다. 이를 반복하면 각 구획이 원래 층의 출력을 차례로 계산한다. [Appendix C, p. 14](https://arxiv.org/pdf/1909.01377v2#page=14)

이 증명 역시 폭을 늘리는 구성이다. 가중치를 공유하면 표현력이 무조건 같다는 주장도, 모든 비용이 줄어든다는 주장도 아니다. 두 정리는 평형·공유 구조의 표현 가능성을 설명하고, 실제 효율은 실험과 solver 성질에서 따로 확인하도록 한다.

**Theorem 3의 생략된 귀납 단계를 보충하면.** 원문 식 (17)–(21)은 $W_z$의 바로 아래 대각 블록에 $W^{[1]},\ldots,W^{[L-1]}$를 놓고, $W_x$의 첫 블록에 $W^{[0]}$를 놓는다. $\sigma$와 $\tilde b$도 원래 층별 비선형·bias를 각 구획에 저장한다. 아래의 아래첨자 $k$는 그 구획을 가리키는 **리뷰 보조 첨자**다.

$$
\begin{aligned}
\tilde z^{[i+1]}_1&=\sigma^{[0]}(W^{[0]}x+b^{[0]}),\\
\tilde z^{[i+1]}_k&=\sigma^{[k-1]}(W^{[k-1]}\tilde z^{[i]}_{k-1}+b^{[k-1]}),
\quad 2\leq k\leq L,\\
\tilde z^{[i]}_k&=z^{[k]}\quad\text{for }i\geq k.
\end{aligned}
$$

첫 구획은 한 번의 갱신 뒤 원래 첫 층의 출력이 되고, 이후 바뀌지 않는다. $k-1$ 구획이 $i\geq k-1$에서 맞다고 가정하면 다음 갱신의 $k$ 구획은 원래 $k$층 출력을 계산한다. 따라서 $L$번 뒤 모든 구획이 원래 네트워크의 층 출력을 포함한다. 숨은 폭은 각 층 폭의 합으로 늘고 원래 각 층의 파라미터도 큰 공유 행렬 안에 들어간다. 학습 가능한 파라미터가 원래보다 반드시 줄어든다는 증명은 아니다. 이 구성은 실제 DEQ 실험에서 사용하는 dense 블록의 수렴 증명으로도 쓰이지 않는다. [Appendix C, Theorem 3](https://arxiv.org/pdf/1909.01377v2#page=14)

## 4. DEQ의 구현 사례 (Instantiations of DEQ)

DEQ는 새로운 attention 연산 하나를 제안하는 논문이 아니다. 평형 방정식과 암시적 학습 방식을 두 가지 시퀀스 블록에 적용한다. [§4, Figure 1, p. 6](https://arxiv.org/pdf/1909.01377v2#page=6)

| 블록 | 상태를 갱신하는 내용 | 입력과 문맥 처리 |
| --- | --- | --- |
| DEQ-TrellisNet | causal 1D convolution과 LSTM 계열 gating | 공유 convolution에 입력의 투영을 매번 주입; 앞 구간의 상태와 padding으로 문맥 처리 |
| DEQ-Transformer | self-attention, feed-forward, residual, layer normalization | 입력을 Q/K/V 계산에 주입; causal mask·상대 위치 표현·이전 구간의 평형 상태 사용 |

Transformer 구현은 Transformer-XL의 문맥 처리와 상대 위치 표현을 활용한다. 상태 갱신은 깊이마다 같은 함수를 사용해야 하므로, 깊이별 시간 임베딩 등을 그대로 포함한 모든 반복 Transformer와 동일하지는 않다. Layer normalization과 gating은 실험적 안정화에 도움을 주지만, 그 존재만으로 전체 함수의 수축성(contractivity)이 증명되지는 않는다.

## 5. 실험 (Experiments)

실험은 복사 과제, Penn Treebank(PTB), WikiText-103(WT103)로 구성된다. 언어 모델 지표인 <strong>퍼플렉서티</strong>(perplexity, PPL)는 정답 토큰에 할당한 확률의 평균 음의 로그를 지수화한 값이며 낮을수록 좋다. 같은 데이터와 토큰화·어휘 등 평가 조건 안에서 비교해야 한다. PPL의 변화율을 정확도 변화율로 읽거나 PTB와 WT103의 값을 직접 비교하면 안 된다. [§5, Tables 1–3; Appendix F](https://arxiv.org/pdf/1909.01377v2#page=7)

<strong>복사 과제(copy task)는 긴 간격을 넘어 정보를 보존하는 작은 스트레스 테스트다.</strong> 처음 10개 기호를 기다림 구간 뒤에 다시 출력한다. $T=400$ 설정의 실제 시퀀스 길이는 $T+20=420$이다. 약 14–16K 파라미터 모델에서 DEQ-Transformer의 보고 loss는 $3.5\times 10^{-6}$, TCN은 $2.7\times 10^{-5}$, LSTM은 $0.0501$, GRU는 $0.0491$이다. Table 1과 부록에 이 loss의 정확한 집계 정의가 명시되어 있지 않아 여기서는 원문의 ‘loss’ 표기를 유지한다. 이 결과는 일반 언어 이해나 추론 능력의 검증과 구분해야 한다. [Table 1, p. 7; Appendix F, p. 16](https://arxiv.org/pdf/1909.01377v2#page=16)

다음은 원문의 주요 언어 모델 행을 선택해 다시 적은 표다. <strong>모든 메모리 값은 시퀀스 길이 150, batch size 15, 단어 임베딩 제외라는 공통 측정 조건</strong>이다. 실제 각 모델의 학습 설정 전체가 같다는 뜻은 아니다. 보조 손실, 전체 파라미터 수, 세부 구조도 모델별로 다르다. [Table 2 주석; Table 3, pp. 7–8](https://arxiv.org/pdf/1909.01377v2#page=7)

| 데이터 | 모델 | 전체 / 임베딩 제외 파라미터 | Test PPL ↓ | 메모리 GB ↓ |
| --- | --- | --- | --- | --- |
| PTB | TrellisNet 60층, 보조 손실, MoS 없음 | 24M / 20M | 57.0 | 8.5 |
| PTB | DEQ-TrellisNet | 24M / 20M | 57.1 | 1.2 |
| WT103 | TrellisNet 70층, 보조 손실 | 180M / 45M | 29.2 | 24.7 |
| WT103 | 위 TrellisNet + checkpointing | 180M / 45M | 29.2 | 5.2 |
| WT103 | DEQ-TrellisNet | 180M / 45M | 29.0 | 3.3 |
| WT103 | Transformer-XL 18층, medium, adaptive embedding | 110M / 72M | 23.6 | 9.0 |
| WT103 | DEQ-Transformer, medium, adaptive embedding | 110M / 70M | 23.2 | 3.7 |

PTB에서는 PPL이 거의 같은 수준이고 메모리는 크게 줄어든다. WT103의 Trellis 비교에서는 $24.7\to 3.3\,\mathrm{GB}$, 약 <strong>86.6%</strong> 절감이다. 하지만 checkpointing을 이미 적용한 $5.2\,\mathrm{GB}$와 비교하면 약 <strong>36.5%</strong>다. Transformer의 adaptive embedding 행에서는 $9.0\to 3.7\,\mathrm{GB}$, 약 <strong>58.9%</strong>다. 이는 표의 반올림된 값을 이용한 계산이다. 초록의 ‘최대 88%’는 저자의 전체 요약 주장으로 구분하며, 모든 구조와 기준 모델에 적용되는 비율로 쓰지 않는다.

품질에서도 DEQ가 모든 행을 이기는 것은 아니다. PTB Table 2의 DARTS는 PPL $55.7$, WT103 Table 3의 훨씬 큰 Transformer-XL은 $18.7$이다. 규모와 학습 조건이 다른 행들이므로, 이 논문의 설득력은 절대적인 최고 PPL보다 <strong>비슷한 규모·품질에서 깊이 방향 저장을 줄일 수 있다는 사례</strong>에 있다. [Tables 2–3](https://arxiv.org/pdf/1909.01377v2#page=8)

실행 시간에는 대가가 있다. Table 4의 값은 `DEQ 시간 / 비교 모델 시간`이므로 1보다 크면 느리다. [Table 4, p. 9](https://arxiv.org/pdf/1909.01377v2#page=9)

| 비교 모델, WT103 | 학습 시간 비율 | 추론 시간 비율 |
| --- | --- | --- |
| 18층 Transformer | 2.82× | 1.76× |
| 70층 TrellisNet | 2.40× | 1.64× |

따라서 이 실험에서 DEQ는 메모리를 절약하지만 더 빠른 모델은 아니다. 서로 다른 하드웨어와 구현까지 포함한 보편적인 지연시간 비율도 아니다. 반복 횟수, 블록 한 번의 비용, batching과 solver 구현을 함께 봐야 한다.

### Figure 2: 학습 중 탐색 계산량

<figure class="review-figure" id="paper-figure-2">
<a href="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" width="1732" height="509" alt="원문 Figure 2. 왼쪽은 학습 epoch에 따른 시간 단계당 Broyden 반복 수, 오른쪽은 함수 평가 수에 따른 평형 잔차의 로그 축 그래프다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 2 · 학습이 진행되면 평형 탐색도 어려워질 수 있다</span>원문 v2 p. 8의 두 패널을 주변 본문에서 분리했다. 축·범례·곡선은 유지했다. <span class="figure-links"><a href="https://arxiv.org/pdf/1909.01377v2#page=8">Shaojie Bai, J. Zico Kolter, Vladlen Koltun, 2019, Figure 2</a> · <a href="/assets/reviews/deep-equilibrium-models/paper-figure-2.webp" target="_blank" rel="noopener noreferrer">크게 보기</a> · 검토를 위한 그림 인용; 권리는 원저자에게 있음</span></figcaption>
</figure>

<strong>왼쪽은 계산 부담을 읽는 그래프다.</strong> 가로축은 training epoch, 세로축은 Broyden 반복 수를 시퀀스의 시간 단계 수로 나눈 값이다. $0.9$를 ‘solver를 한 번도 실행하지 않았다’고 읽으면 안 된다. 길이 150으로 정규화된 값이며, 파란 선은 forward, 붉은 선은 backward다. 이 설정에서는 학습이 진행될수록 두 계산량이 증가한다. [Figure 2, §5, p. 8](https://arxiv.org/pdf/1909.01377v2#page=8)

<strong>오른쪽은 상태의 잔차를 읽는 그래프다.</strong> 가로축은 함수 평가 횟수이고, 세로축은 $\|f(z)-z\|_2$의 로그 축이다. PPL도 정답 상태까지의 거리도 아니다. 가중치 공유 Transformer의 단순 반복은 epoch 1에서는 내려가지만 epoch 12에서는 진동하며 큰 잔차를 유지한다. DEQ의 root 탐색은 두 시점 모두 훨씬 작은 잔차에 도달한다. 이는 단순 반복과 수치적 root 탐색의 차이를 보여준다. 유한층 기준 모델보다 벽시계 시간이 빠르다는 증거로 바꿀 수는 없다.

### Figure 3: 정밀도와 품질

<figure class="review-figure" id="paper-figure-3">
<a href="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" width="1737" height="469" alt="원문 Figure 3. 왼쪽은 forward 잔차 허용치와 validation PPL, 오른쪽은 forward 반복 한도와 validation PPL의 관계다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 3 · 너무 일찍 멈추면 품질이 낮아진다</span>원문 v2 p. 9의 두 패널을 분리했으며 데이터와 원문 표기를 유지했다. <span class="figure-links"><a href="https://arxiv.org/pdf/1909.01377v2#page=9">Shaojie Bai, J. Zico Kolter, Vladlen Koltun, 2019, Figure 3</a> · <a href="/assets/reviews/deep-equilibrium-models/paper-figure-3.webp" target="_blank" rel="noopener noreferrer">크게 보기</a> · 검토를 위한 그림 인용; 권리는 원저자에게 있음</span></figcaption>
</figure>

왼쪽은 forward의 잔차 허용치를 키울수록 어디서 품질이 무너지는지를 보여준다. 가로축은 로그 축이고 값이 작을수록 엄격하다. 작은 허용치 구간에서는 validation PPL이 비슷하지만 너무 크게 완화하면 급격히 나빠진다. 오른쪽은 반복 한도를 늘릴수록 PPL이 개선되고, 추가 계산의 이득이 작아지는 모습을 보여준다. 음영은 원문에 있지만 그 집계 의미가 명시되지 않아 신뢰구간(confidence interval)으로 해석하지 않는다. [Figure 3, §5, p. 9](https://arxiv.org/pdf/1909.01377v2#page=9)

이 그림의 실험은 adaptive embedding 없는 medium DEQ-Transformer다. 본문은 <strong>길이 75인 시퀀스</strong>에서 $\epsilon <0.1$ 또는 반복 한도 30이 경쟁력 있는 결과를 냈다고 설명한다. 이 설정을 앞 표의 메모리 측정 조건인 길이 150이나 모든 모델의 최적 설정으로 일반화하면 안 된다. 그림의 지표는 validation PPL이며 앞 표의 test PPL과도 구분해야 한다.

## 6. 결론 (Conclusion)

저자는 시퀀스 수준 고정점을 직접 풀고 최적화하면, 서로 다른 구조에 적용하면서 깊이 방향 메모리를 줄이고 경쟁력 있는 시퀀스 모델을 만들 수 있다고 결론낸다. 실험에서는 품질·메모리 이득과 더 느린 실행 시간을 함께 보아야 한다. root solver에 독립적인 출력 정의는 모든 solver가 항상 같은 해를 찾아 성공한다는 보증과 다르다. [원문 §6](https://arxiv.org/pdf/1909.01377v2#page=9)

## 부록 D–F: 구현 조건과 추가 한계

논문의 부록은 수학적 식만으로 드러나지 않는 실무 조건을 정리한다. [Appendix D–F, pp. 14–16](https://arxiv.org/pdf/1909.01377v2#page=14)

- <strong>드롭아웃 마스크(dropout mask)를 solver 반복 사이에 공유한다.</strong> 반복마다 함수를 무작위로 바꾸면 고정된 하나의 방정식을 푼다는 전제가 흔들린다.
- <strong>안정화가 필요하다.</strong> 작은 초기 가중치, gating, normalization, 얕은 모델로 시작하는 warm-up 등을 사용한다. 안정적인 예제가 있다는 것과 전체 학습 과정의 전역 수렴 보장은 다르다.
- <strong>Batch 안의 시퀀스마다 필요한 반복 수가 다르다.</strong> 느린 시퀀스를 기다리는 비용 때문에 함수 평가 수만으로 GPU 효율을 알 수 없다.
- <strong>매우 긴 시퀀스는 더 어려울 수 있다.</strong> 부록은 길이 1,000을 넘는 경우 초기 탐색의 어려움과 구간 분할을 논의한다. 깊이 저장을 줄였다고 문맥 길이의 비용까지 사라지지는 않는다.
- <strong>중간 층의 보조 손실을 그대로 옮기기 어렵다.</strong> DEQ는 끝점에 손실을 연결하므로, 유한층 모델의 심층 감독(deep supervision)과 정확히 같은 학습 설정이 아니다.

<span id="리뷰-해석과-후속-질문" aria-hidden="true"></span>

## 리뷰 해석과 남은 질문

**리뷰 해석.** DEQ는 깊이를 ‘저장해야 하는 층의 목록’에서 ‘풀어야 하는 상태 방정식’으로 바꾼다. 출력은 평형 상태, 학습은 그 상태의 민감도다. 이 전환 덕분에 깊이 방향의 activation 저장을 크게 줄일 수 있지만, solver가 얼마나 잘 수렴하는지와 gradient의 수치 안정성(numerical stability)이 새로운 중심 문제가 된다.

이 글의 판단으로는 가장 중요한 독해 질문은 ‘무한 깊이를 구현했는가?’보다 <strong>어떤 함수에 어떤 조건으로 평형 상태가 생기며, 어느 정밀도로 풀었을 때 품질·메모리·시간의 교환이 유리한가?</strong>다. 후속 연구인 [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656)는 다중 해상도 비전으로 적용을 넓히고, [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342)는 안정성과 계산 효율을 직접 다룬다. 두 후속 논문의 인용 범위는 공식 초록·서지다.

## 출처와 읽은 범위

- 원문: *Deep Equilibrium Models* · Shaojie Bai et al. · 최초 공개 2019-09-03 · [검토 버전](https://arxiv.org/abs/1909.01377v2) · 검토일 2026-10-23.
- 읽은 범위: v2 본문 §1–6·Appendix A–F. A–C의 정리·증명은 §3의 해당 설명에 연결한다.
- 그림 출처: 대표 그림은 자체 제작물(CC BY 4.0). Figures 2·3은 해설을 위해 원문 일부를 인용하며 원저자 권리를 유지한다. arXiv 비독점 배포 라이선스는 일반 재사용 허가가 아니다. [원문](https://arxiv.org/abs/1909.01377v2).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
