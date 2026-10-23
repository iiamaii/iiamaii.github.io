---
title: "끝점으로 반복 경로를 대신할 수 있을까: Looped Models와 고정점의 효율"
description: "고정점 관점으로 TBPTT·KV 공유·RL 상태 재사용·distilled prefill을 설명하고, 업데이트 구간 가속과 전체 시간·정확도의 대가를 비교한다."
date: "2026-10-08"
updatedAt: "2026-10-23T08:56:12+00:00"
publishedAt: "2026-10-08T16:55:04+09:00"
topics: ["language models", "looped models", "fixed points", "efficient inference", "reinforcement learning"]
translationKey: "looped-models-fixed-points"
paperTitle: "Towards Looped Models Done Right, Part II: Rethinking at Fixed Points"
paperPublishedDate: "2026-10-05"
authors: ["Benhao Huang", "Chufan Shi", "Junlin Chen", "Shicheng Wen", "Zhengzhong Liu", "Eric Xing", "Xuezhe Ma"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.06833v1"
visibility: "public"
lang: "ko"
thumbnail: "/assets/reviews/looped-models-fixed-points/paper-figure-1.svg"
thumbnailAlt: "깊이 분포와 직교 입력 주입, 고정점 근처의 반복 궤적, 학습 activation·KV 캐시·prefill·RL 채점과 역전파의 비용 절감을 연결한 논문 Figure 1."
---

이 논문은 반복형 언어 모델(looped language model)의 고정점(fixed point) 근처 상태를 재사용해 학습·캐시·강화학습의 비용을 줄인다. 강화학습(reinforcement learning, RL) 상태 재사용과 Neumann-4를 결합하면 업데이트 구간은 2.77초에서 1.39초로 줄지만, 전체 구간은 65.45초에서 57.30초로 줄어든다. [원문 §5, Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=40) **리뷰 해석.** 고정점의 가치는 반복 중단을 넘어 계산 재사용으로 확장되며, 구간별 가속과 품질의 대가를 따로 읽는 것이 핵심이다.

**Figure 1 · 대표 그림을 읽는 법.** 왼쪽은 learned depth prior와 OrthoInj라는 두 학습 개선 방향, 가운데는 반복 경로와 고정점 근처의 끝점, 오른쪽은 그 상태를 활용하는 네 가지 비용 절감을 연결한다. 오른쪽 막대는 서로 다른 실험의 보고값이며, 네 기법을 결합한 누적 가속이 아니다. 특히 RL의 2배는 전체 학습이 아니라 채점·역전파 구간이다. [Benhao Huang et al., 2026, Figure 1](https://arxiv.org/html/2610.06833v1#S0.F1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · 원본 유지 · [크게 보기](/assets/reviews/looped-models-fixed-points/paper-figure-1.svg).

이 글은 원문 §1–7을 따른다. 부록 A.1의 Lemma 1은 §3.2에, A.2의 Lemma 2는 §3.1에 붙여 가정과 중간 계산을 설명한다. RL과 prefill은 §3에서 작동 원리를, §5에서 품질·시간의 대가를 읽으면 된다.

## 1. 도입 (Introduction)

일반적인 Transformer는 서로 다른 가중치를 가진 층들을 차례로 통과한다. **반복형 언어 모델**(looped language model)은 일부 층에서 가중치 공유(weight sharing)를 사용하고 같은 블록을 여러 번 실행한다. 가중치를 늘리지 않고 토큰당 계산량을 늘릴 수 있지만, 반복별 중간 상태를 모두 저장하면 학습 메모리와 생성 중 캐시가 반복 횟수만큼 커진다. 긴 입력을 처음 읽는 **프롬프트 초기 처리**(prefill)도 비싸진다. 이후 토큰을 하나씩 생성하는 단계는 **디코딩**(decoding)이다. [§1–2](https://arxiv.org/html/2610.06833v1#S2)

저자들은 네 가지 질문을 연결한다. 마지막 몇 번만 역전파해도 되는 이유는 무엇인가? 이전 토큰의 마지막 KV만 재사용해도 되는가? 강화학습에서 이미 생성할 때 계산한 상태를 다시 계산해야 하는가? 프롬프트의 끝점 상태를 작은 모델이 대신 예측할 수 있는가? 이를 가능하게 하는 상태를 만들기 위해 학습 중 반복 횟수의 분포와 입력 주입 방식도 바꾼다. [§3–4](https://arxiv.org/html/2610.06833v1#S3)

## 2. 반복형 언어 모델의 고정점 (Fixed Points in Looped Language Models)

실험 모델은 Huginn 구조다. 입력 처리부(prelude)가 입력을 표현으로 바꾸고, 두 Transformer 블록으로 된 반복 핵심 블록(recurrent core)이 그 표현을 매번 주입받으며 반복한다. 출력 처리부(coda)는 최종 상태를 다음 토큰 확률로 바꾼다. 다음 표기의 상태 $z$에는 은닉 상태(hidden state) $\mathbf H$와 attention의 key/value 표현을 저장하는 $C$가 함께 들어간다. [§2](https://arxiv.org/html/2610.06833v1#S2)

$$
\begin{aligned}
e&=P_\theta(x),\\
z^r&=(\mathbf H^r,C^r),\\
z^{r+1}&=F_\theta(z^r;x),\\
z^\star&=F_\theta(z^\star;x).
\end{aligned}
$$

수학적 고정점에서는 같은 연산을 한 번 더 해도 상태가 같다. 실제 모델에서는 유한한 반복 뒤의 근사적인 정체 상태도 고정점이라고 부른다. 두 의미를 구분해야 한다. 상태가 조금만 바뀐다는 관찰은 예측이 정답이라는 뜻도, 모든 입력에서 같은 속도로 수렴한다는 뜻도 아니다. [§2, Appendix A.2](https://arxiv.org/pdf/2610.06833v1#page=22)

| 적용 구간 | 줄이는 것 | 남아 있는 계산 또는 조건 |
| --- | --- | --- |
| 사전학습의 시간축 절단 역전파(truncated backpropagation through time, TBPTT) | 오래된 반복의 활성값(activation) 저장과 역전파 | 순전파는 뽑힌 깊이까지 실행 |
| 최종 반복 KV 공유(terminal KV sharing) | 이전 토큰의 반복별 KV bank | 현재 토큰은 계속 반복하며 자기 반복의 KV 사용 |
| RL rollout-state reuse | 생성한 토큰의 끝점을 복원하는 재계산 | 한 번의 core 순전파와 여러 역방향 VJP, rollout 자체 |
| Distilled prefill | 프롬프트의 여러 teacher 반복 | student, teacher 한 번의 반복·coda, 이후 teacher decoding |

벡터–야코비안 곱(vector–Jacobian product, VJP)은 야코비안(Jacobian) 전체를 만들지 않고 벡터에 대한 역방향 미분을 계산하는 연산이다. 네 방법을 모두 결합한 하나의 배포 시스템에서 누적 가속을 측정한 논문으로 읽으면 안 된다. 특히 RL 실험의 생성·평가는 반복별 KV를 유지한다. [§3, Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=40)

원문 그림 2는 반복을 세로축, 토큰 위치를 가로축으로 두고 인접 반복 사이의 상대 은닉 상태 변화를 색으로 표시한다. 뒤 토큰이 앞 토큰보다 먼저 안정되는 경우도 있다. ‘문장의 앞부분부터 차례로 고정된다’는 단순한 설명이 맞지 않는 이유다. 관측한 8개 시퀀스에서 수렴한 토큰들을 모았을 때 위치와 수렴 깊이의 스피어만 순위 상관(Spearman rank correlation)은 0.14다. 이 수치를 모든 언어·문서에 대한 일반 법칙으로 확대할 수는 없다. [Figure 2, Appendix C.4](https://arxiv.org/pdf/2610.06833v1#page=35)

$$
\rho_{r,t}=\frac{\|h_t^r-h_t^{r-1}\|_2}{\|h_t^{r-1}\|_2+\epsilon}.
$$

그림에서는 관측 구간 끝까지 변화가 2% 아래로 유지되는 첫 깊이를 찾고, 적어도 네 번의 안정된 업데이트를 요구한다. 관측 길이를 64로 늘린 부록 Table 19에서 Small PLN-5 모델의 토큰 99.74%가 1% 기준을 만족한다. 이는 정해진 데이터와 유한한 관측 구간의 **상태 안정성**이다. 반면 훨씬 엄격한 수렴 검사에서는 탈락한 입력도 있고 수백 번의 반복이 필요하다. [Appendix C.4, Table 19; A.1](https://arxiv.org/pdf/2610.06833v1#page=36)

<figure class="review-figure" id="paper-figure-2">
<a href="/assets/reviews/looped-models-fixed-points/paper-figure-2.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/looped-models-fixed-points/paper-figure-2.png" width="836" height="232" alt="토큰 위치와 recurrence에 따른 상대 은닉 상태 변화 heatmap, 국소 안정 깊이 역전, 위치와 안정 깊이 산점도." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 2 · 토큰의 안정화는 순서대로 일어나지 않는다</span>어두운 영역은 작은 상대 상태 변화다. 확대 패널의 계단선은 인접 토큰의 안정 깊이가 뒤바뀔 수 있음을 보여준다. 오른쪽 위치·깊이의 약한 상관은 관측한 시퀀스에 한정된다. <span class="figure-links"><a href="/assets/reviews/looped-models-fixed-points/paper-figure-2.png" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2610.06833v1#S2.F2">Benhao Huang et al., 2026, Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원본 유지, 수정·자르기 없음</span></figcaption>
</figure>

## 3. 고정점 형성이 가능하게 하는 것 (What Fixed-Point Shaping Enables)

<figure class="review-figure" id="paper-figure-3">
<a href="/assets/reviews/looped-models-fixed-points/paper-figure-3.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/looped-models-fixed-points/paper-figure-3.svg" width="508" height="160" alt="좌: 마지막 역전파 창, 중: terminal prefix KV 공유, 우: student prompt prefill과 recurrent teacher decoding의 원문 구조도." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 3 · 학습·캐시 공유·student prefill의 경로</span>(a) 점선 아래 상태를 분리하고 마지막 창만 역전파한다. (b) 과거 토큰의 마지막 KV를 여러 현재-token 반복이 읽는다. (c) 비반복 student가 prefix 끝점을 예측하고 teacher가 이후 생성한다. Student 뒤 한 teacher recurrence와 coda를 거치는 세부는 본문 설명을 함께 읽는다. <span class="figure-links"><a href="/assets/reviews/looped-models-fixed-points/paper-figure-3.svg" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2610.06833v1#S3.F3">Benhao Huang et al., 2026, Figure 3</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원본 유지, 수정·자르기 없음</span></figcaption>
</figure>

### 3.1 학습: 끝점의 기울기 (Training: Endpoint Gradients)

학습 목적과 gradient 추정의 대상을 먼저 구분한다.

#### 사전학습: 절단 역전파와 Lemma 2

시간축 절단 역전파(truncated backpropagation through time, TBPTT)는 마지막 $b$번만 미분하고 그 이전 상태를 분리한다. 정확한 고정점에서 $J_\star$를 상태에 대한 core의 Jacobian, $B_\star$를 상태를 고정했을 때 파라미터에 대한 미분으로 정의하면, 다음 관계가 나온다. $I$는 항등행렬이다. [§3.1, 식 1; Appendix A.2, 식 14–17](https://arxiv.org/pdf/2610.06833v1#page=22)

$$
\begin{aligned}
D_\star&=\frac{\mathrm dz^\star}{\mathrm d\theta}=(I-J_\star)^{-1}B_\star
=\sum_{k=0}^{\infty}J_\star^kB_\star,\\
D_b&=\sum_{k=0}^{b-1}J_\star^kB_\star,\\
D_\star-D_b&=J_\star^bD_\star,\qquad \rho(J_\star)<1.
\end{aligned}
$$

마지막 몇 번의 미분은 고정점 민감도(sensitivity)의 노이만 급수(Neumann series)를 일부만 남기는 것과 연결된다. $\|J_\star\|_2\le\kappa<1$이라면 손실 기울기(gradient)의 생략 오차 상한은 $\|B_\star\|_2\|g\|_2\kappa^b/(1-\kappa)$다. $g$는 끝점 상태에 대한 손실 gradient다. 입력 주입·prelude·출력 head로 직접 이어지는 파라미터 경로도 유지해야 한다. [Appendix A.2, Lemma 2](https://arxiv.org/pdf/2610.06833v1#page=23)

**Lemma 2의 증명을 한 단계씩 읽기.** $J_\star\in\mathbb R^{d_z\times d_z}$, $B_\star\in\mathbb R^{d_z\times d_\theta}$다. 고정점 방정식을 미분하면 $D_\star=J_\star D_\star+B_\star$이므로, $I-J_\star$의 가역성이 implicit sensitivity에 필요하다. 연속 미분 가능성과 이 가역성으로 음함수 정리(implicit function theorem)을 적용한다. 그보다 강한 $\rho(J_\star)<1$이 Neumann 급수의 수렴을 준다. 역행렬이 존재한다는 사실만으로 급수가 수렴하는 것은 아니다.

$c$는 상태를 고정한 손실의 직접 파라미터 미분이고 $g$는 끝점 상태에 대한 열 gradient다. 정확한 끝점을 detach하고 동일 recurrence를 $b$번 적용하면 민감도가 $D_0=0$, $D_{k+1}=J_\star D_k+B_\star$를 만족한다. 따라서 $G_b=c+D_b^\top g$이며, $\|J_\star\|_2\leq\kappa<1$이면 다음과 같이 꼬리를 제한한다. 아래는 부록 A.2 식 (14)–(17)의 중간 연결을 펼친 **리뷰 보충 유도**다.

$$
\begin{aligned}
D_\star&=J_\star D_\star+B_\star,\\
(I-J_\star)\sum_{k=0}^{b-1}J_\star^k&=I-J_\star^b,\\
G_{\mathrm{imp}}&=c+D_\star^\top g,\qquad G_b=c+D_b^\top g,\\
\|G_{\mathrm{imp}}-G_b\|_2
&\leq\|B_\star\|_2\|g\|_2\sum_{k=b}^{\infty}\kappa^k
=\|B_\star\|_2\|g\|_2\frac{\kappa^b}{1-\kappa}.
\end{aligned}
$$

스펙트럼 반지름(spectral radius) 조건과 유클리드 연산자 노름(operator norm) 조건은 다르다. 전자는 수렴을 주지만, 위의 상수 $1/(1-\kappa)$를 그대로 쓰려면 후자의 균일 상한이 필요하다. 유한 경로 TBPTT는 식 (18)의 서로 다른 $J_r,B_r$를 곱하므로 $D_b$와 자동으로 같지 않다. 입력 주입·prelude의 직접 경로는 $B_\star$에, 출력 head는 $c$에 남겨야 한다. [Appendix A.2, Lemma 2](https://arxiv.org/pdf/2610.06833v1#page=23)

하지만 **유한한 반복의 목적함수와 평형 상태의 목적함수는 다르다.** 끝점 값이 같아도 파라미터에 대한 미분은 다를 수 있다. 실제 TBPTT는 경로 위의 서로 다른 Jacobian을 사용하고, 끝점 Neumann 근사는 같은 끝점 Jacobian을 반복해서 사용한다. 끝점 잔차(residual)가 작다는 사실 하나만으로 둘이 가깝다고 보장할 수 없다. [Appendix A.2](https://arxiv.org/pdf/2610.06833v1#page=22)

실험에서도 창을 지나치게 줄이면 품질이 낮아진다. Medium PLN-5에서 $b\ge 4$의 WikiText 퍼플렉서티(perplexity, PPL)는 전체 시간축 역전파(full backpropagation through time, full BPTT)의 1% 안이지만, $b=2$는 약 5.5%, $b=1$은 약 15% 높다. Small 고정 깊이 모델에서는 짧은 창이 깊이 증가와 캐시 공유에 대한 안정성을 높이면서 기본 품질을 떨어뜨린다. 학습 초기부터 끝점 Neumann 근사만 사용한 대조군은 시험한 다섯 학습률에서 모두 붕괴했다. ‘고정점 관점이 있으니 어떤 gradient 근사도 안전하다’는 결론은 성립하지 않는다. [Appendix C.3, Tables 17–18](https://arxiv.org/pdf/2610.06833v1#page=34)

메모리 이득은 별도 측정에서 확인한다. 역전파 창 5/10/15의 peak allocation은 local batch 8에서 48/86/124GB였다. 깊이 30의 full BPTT는 메모리 부족을 일으켰고, 층별 활성값 재계산(activation recomputation)은 28GiB로 낮추지만 창 10 대비 step 시간이 약 1.3배였다. 이 수치를 일반적인 모든 학습 설정의 메모리 절감률로 환산할 수는 없다. [§3.1, Appendix C.3](https://arxiv.org/pdf/2610.06833v1#page=33)

#### 후속 학습: RL rollout 상태 재사용

강화학습은 응답을 생성한 뒤 같은 토큰들을 다시 채점하고 정책을 업데이트한다. 저자들은 rollout 때의 끝점 $z^R$을 저장하고, 파라미터가 바뀌기 전에 그 상태에서 core를 한 번 실행해 미분 그래프를 만든다. 저장 상태가 실제 순전파 값으로 유지되도록 하고, 끝점 Jacobian의 VJP를 반복해 근사 gradient를 얻는다. **Neumann-4는 순전파 네 번이 아니라, 동일 그래프에서 상태 VJP 네 번과 identity 항을 포함한 다섯 항의 합**이다. [Algorithm 1; Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=40)

Algorithm 1에서 중요한 것은 상태의 **값**과 **미분 경로**를 다르게 다루는 부분이다. 다음 식에서 $\operatorname{sg}$는 stop-gradient다. $f-\operatorname{sg}(f)$의 값은 0이므로 forward 값은 저장된 $z^R$이고, backward는 한 번 기록한 $F_\theta$의 그래프로 흐른다.

$$
\bar z=\operatorname{sg}(z^R),\quad f=F_\theta(\bar z;x),\quad
\widetilde z=\bar z+\bigl(f-\operatorname{sg}(f)\bigr),\qquad
v\leftarrow g+J_R^\top v.
$$

$g$는 readout을 통해 얻은 gradient다. $v=g$에서 시작해 $b-1$번 VJP를 반복한 뒤 $B_R^\top v$를 파라미터로 전파한다. 파라미터를 바꾸기 전 같은 on-policy rollout의 상태를 쓰고 업데이트 후 버린다. 오래된 상태를 계속 쓰는 off-policy 보정 기법이 아니다. [§3.1, Algorithm 1](https://arxiv.org/pdf/2610.06833v1#page=4)

### 3.2 추론: 끝점 캐싱과 증류 (Inference: Endpoint Caching and Distillation)

과거 토큰의 문맥과 prompt 처리의 끝점을 재사용한다.

#### 디코딩: 최종 KV 공유와 Lemma 1

이 모델은 prelude 1개, core 2개, coda 1개의 물리적 attention 층을 갖는다. $R=5$면 논리적으로 $1+2\times 5+1=12$번 층을 방문한다. 반복별 캐시는 bank 12개를 저장하지만 terminal sharing은 물리적 층별 마지막 bank 4개만 남긴다. $R=16$이면 34개, $R=32$이면 66개 대신 4개다. 이는 **KV 저장량**의 절감이며 전체 GPU 메모리나 FLOPs가 같은 비율로 줄어든다는 뜻은 아니다. [§3.2, Table 20](https://arxiv.org/pdf/2610.06833v1#page=36)

Lemma 1은 이전 토큰들의 문맥이 수렴하고, 현재 토큰의 업데이트가 해당 영역에서 일관되게 수축(contraction)하며 문맥 변화에 대해 립시츠 연속(Lipschitz continuous)이면, 함께 반복하는 방식과 수렴한 prefix를 고정한 방식이 같은 극한에 도달한다고 보인다. 유한한 prefix 오차 $\eta$에 대해서는 다음 두 항이 남는다. [§3.2, Lemma 1; Appendix A.1, 식 12](https://arxiv.org/pdf/2610.06833v1#page=20)

**Lemma 1의 가정과 증명 연결.** 원문의 집합 $\mathcal H$는 비어 있지 않은 닫힌 유클리드 부분집합이고, 모든 관련 문맥에서 업데이트가 그 집합 안에 머문다. 따라서 완비성(completeness)과 $\kappa<1$의 균일 수축 조건으로 바나흐 수축 사상 정리(Banach contraction theorem)을 적용하면, 고정 문맥 $C^\star_{<t}$에 대한 고정점은 존재하고 유일하다. 문맥이 달라지는 공동 반복에도 같은 결론을 옮기려면 $C^r_{<t}\to C^\star_{<t}$와 문맥에 대한 $\beta$-Lipschitz 조건이 추가로 필요하다.

부록 A.1은 $\delta_r=\|h^r-h^\star\|$로 놓고 삼각부등식으로 아래 첫 줄을 얻는다. 문맥이 수렴하므로 $\eta_{\max}$가 유한하고, 귀납법으로 $\delta_r$를 제한한 뒤 limsup를 취한다. $0\leq\kappa<1$이므로 유한한 $\bar\delta\leq\kappa\bar\delta$는 $\bar\delta=0$을 준다.

$$
\begin{aligned}
\delta_{r+1}&\leq\kappa\delta_r+\beta\|C^r_{<t}-C^\star_{<t}\|,\\
\delta_r&\leq\max\left\{\delta_0,\frac{\beta\eta_{\max}}{1-\kappa}\right\},\qquad
\eta_{\max}=\sup_r\|C^r_{<t}-C^\star_{<t}\|,\\
\bar\delta&=\limsup_{r\to\infty}\delta_r\leq\kappa\bar\delta,\qquad \bar\delta=0.
\end{aligned}
$$

readout이 상태·문맥에 대해 연속이면 두 반복의 극한 next-token 분포도 같다. 토큰을 하나씩 고정하는 확장은 토큰 위치에 대한 귀납법을 쓴다. 앞 토큰의 극한에서 다음 토큰의 문맥이 정해지며, 이는 실제로 토큰이 안정되는 시간 순서와는 다르다. 유한한 부정확 문맥을 고정한 식 (12)는 아래 제시한 두 오차 항으로 남는다. KV 외의 prefix 상태가 업데이트에 들어가면 그 상태도 일치해야 하므로 KV만 같다는 조건으로 보조정리를 확대하면 안 된다. [Appendix A.1, Lemma 1, Eqs. (9)–(12)](https://arxiv.org/pdf/2610.06833v1#page=20)

$$
\|\bar h^R-h^\star\|
\le\kappa^R\|\bar h^0-h^\star\|
+\frac{\beta(1-\kappa^R)}{1-\kappa}\,\eta.
$$

첫 항은 현재 토큰을 덜 계산한 오차다. 둘째 항은 부정확하게 고정한 문맥의 오차다. 반복을 아무리 늘려도 두 번째 원인은 사라지지 않는다. 수축 계수 $\kappa$가 1에 가까우면 문맥 오차가 크게 증폭될 수도 있다.

엄격한 실험에서 304개 prefix 중 수렴 검사를 통과한 것은 고정 깊이 모델 206개, PLN-5 모델 191개다. 통과한 prefix의 수렴 깊이 중앙값은 각각 152.5와 114로, 실제 사용 깊이 5보다 훨씬 크다. 깊이 5에서 고정하면 현재 토큰의 끝점 차이는 엄격하게 수렴한 문맥을 사용했을 때보다 5–6자릿수 크다. 따라서 실제 $R=5$의 공유가 유용하다는 근거는 **정확한 고정점에 도달했다는 주장**보다 유한 오차 분석과 실제 품질 측정에 있다. [Appendix A.1, Figure 6](https://arxiv.org/pdf/2610.06833v1#page=21)

부록 C.6은 prefill과 decoding 깊이를 각각 1–32 사이의 8개 값으로 바꿔 8×8 조합을 비교한다. 충분히 깊게 decode하면 prefill을 5 이상 늘린 이득은 작다. 반대로 얕게 decode하는 모델은 얕은 prefill을 선호하는 경우도 있다. 따라서 ‘입력을 더 오래 읽을수록 항상 좋다’거나 두 깊이를 반드시 같게 해야 한다는 결론은 나오지 않는다. [Appendix C.6, Figure 11](https://arxiv.org/pdf/2610.06833v1#page=37)

#### Prefill: 반복 끝점 증류

비반복 student 두 블록이 teacher의 pre-coda 끝점을 예측한다. Student는 teacher core에서 초기화하고, 정규화한 hidden-state 오차와 teacher 출력 분포에 대한 쿨백–라이블러 발산(Kullback–Leibler divergence, KL divergence)으로 학습한다. Teacher는 고정하며 student에 teacher 사전학습 토큰의 1/4을 추가로 투입한다. **Student 출력 뒤 teacher core 한 번과 coda가 terminal KV를 만들고, 이후 생성은 teacher가 계속 5번 반복한다.** 전체 언어 모델을 student로 교체한 압축 실험이 아니다. [§3.2, 식 4; §5.5, Appendix C.7](https://arxiv.org/pdf/2610.06833v1#page=38)

식 (4)의 두 학습 목표도 구분한다. $N$은 prompt 길이, $d$는 hidden width, $\sigma_{\mathrm T}^2$는 학습 데이터에서 보정한 teacher 상태 에너지다. 첫 항은 teacher 척도로 정규화한 좌표당 상태 오차, 두 번째는 teacher 분포 $p_t$와 student 상태에서 읽은 $\widehat p_t$의 KL이다.

$$
\mathcal L_{\mathrm{distill}}(\psi)=\mathbb E_x\!\left[
\frac{\|\widehat z-z_\theta^R(x)\|_F^2}{Nd\,\sigma_{\mathrm T}^2}
+\frac1N\sum_{t=1}^{N}\mathrm{KL}(p_t\Vert\widehat p_t)
\right].
$$

낮은 상태 MSE가 동일한 생성 품질을 뜻하지 않기 때문에 출력 분포 항을 함께 둔다. 실제 추론에서 student 뒤에 teacher 한 recurrence를 거치는 handoff와, 손실의 출력 분포를 계산하는 coda 경로를 구분해야 한다. [§3.2, Eq. (4)](https://arxiv.org/pdf/2610.06833v1#page=6)

## 4. 고정점 형성 개선 (Improving Fixed-Point Shaping)

깊이 분포와 입력 주입을 별도로 바꾼다.

### 4.1 깊이 분포 학습 (Learning a Depth Prior for Fixed-Point Training)

Huginn의 기준 분포는 shifted Poisson-lognormal(PLN)이다. 평균 반복 횟수를 5로 맞추되 매번 같은 깊이로 학습하지 않는다. 논문은 이 분포에서 시작한 1–64 깊이의 categorical 확률을 학습한다. **입력별 중단 정책이 아니라 전체 학습에 공통인 분포**이며, 반복 전에 microbatch마다 깊이 하나를 뽑는다. [§4.1, Algorithm 2; Appendix B.3](https://arxiv.org/pdf/2610.06833v1#page=25)

$$
\begin{aligned}
\mathcal J_{\mathrm{prior}}(\phi)
&=-\widehat{\mathbb E}_{n}\!\left[\operatorname{sg}(A_n)\log p_\phi(r)\right]\\
&\quad-\lambda_H H(p_\phi)
+\lambda_m\!\left(\mathbb E_{p_\phi}[R]-\bar R\right)^2,\quad\bar R=5.
\end{aligned}
$$

$A_n$는 $\exp(-CE)$, 즉 역 퍼플렉서티(inverse perplexity) 기반 보상(reward)을 이동평균과 분산으로 조정한 어드밴티지(advantage)다. 모델 자체는 기존 교차엔트로피(cross-entropy, CE)로 학습한다. 엔트로피(entropy) $H(p_\phi)$는 분포가 한 깊이에만 몰리는 것을 막고, 마지막 항은 평균 깊이를 예산 근처에 유지한다. 모델 학습을 위해 이미 계산한 손실을 사용하므로 controller를 위해 별도 모델 순전파를 더하지 않는다. 기본 $\lambda _H=0.01$, $\lambda _m=1$, 역전파 창의 최대 길이는 10이다. [§4.1, Appendix B.3](https://arxiv.org/pdf/2610.06833v1#page=25)

그림 4에서는 고정 $R=5$ 학습의 주황 곡선이 테스트 깊이를 벗어나면 불안정해진다. 확률적으로 깊이를 바꾸는 PLN과 학습 분포는 비교적 안정적이고, 학습 분포가 PLN보다 PPL을 낮춘다. 그러나 이것은 모든 문제의 정확도가 반복에 따라 단조롭게 좋아진다는 뜻은 아니다. [Figure 4; Appendix C.5](https://arxiv.org/pdf/2610.06833v1#page=8)

<figure class="review-figure" id="paper-figure-4">
<a href="/assets/reviews/looped-models-fixed-points/paper-figure-4.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/looped-models-fixed-points/paper-figure-4.svg" width="496" height="168" alt="S와 M 모델의 테스트 깊이 1–64에 따른 WikiText PPL; 고정 깊이, PLN, learned prior 비교." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">Figure 4 · 학습 깊이 밖에서의 PPL</span>가로축은 테스트 반복 횟수, 세로축은 WikiText PPL로 낮을수록 좋다. 고정 깊이 학습은 5 근처에서 좋지만 깊이를 벗어나면 불안정하다. 확률적·학습 분포는 더 안정적이다. 확대창은 5 근처의 작은 차이를 보여주며, 생성 정확도 그래프는 아니다. <span class="figure-links"><a href="/assets/reviews/looped-models-fixed-points/paper-figure-4.svg" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2610.06833v1#S4.F4">Benhao Huang et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원본 유지, 수정·자르기 없음</span></figcaption>
</figure>

### 4.2 직교 입력 주입 (Orthogonal Input Injection)

이전 상태를 감쇠해 넘기는 Parcae 방식은 $\Lambda h_t^r+q_t$ 형태다. $q_t$가 현재 반복에 주입하는 입력, $\Lambda$가 학습되는 대각 감쇠 행렬(diagonal decay matrix)이다. 이전 상태에도 $q_t$ 방향 성분이 있으므로 실제 입력 방향의 크기는 달라질 수 있다. 직교 입력 주입(orthogonal input injection, OrthoInj)은 그 성분을 제거한 뒤 $q_t$를 더한다. [§4.2, 식 7–8](https://arxiv.org/pdf/2610.06833v1#page=8)

$$
\begin{aligned}
q_t&=\Delta\odot We_t,\\
Q_{q_t}&=I-\frac{q_tq_t^\top}{\|q_t\|_2^2+\epsilon},\\
\zeta_t^r&=Q_{q_t}\Lambda h_t^r+q_t.
\end{aligned}
$$

$\odot$는 원소별 곱이다. $q_t\ne0$, $\epsilon =0$이면 $q_t$ 방향 성분이 정확히 $q_t$로 유지된다. 구현에서는 $\epsilon =10^{-6}$을 사용한다. 투영이 감쇠 부분의 norm을 늘리지 않는다는 성질이 있어도 **Transformer 전체의 수축성 증명**이 되는 것은 아니다. 또한 이 recipe에는 prelude RMSNorm 제거도 포함된다. 부록의 2×2 비교는 정규화 제거와 투영이 각각 validation PPL에 기여함을 보이지만, 각 설정에서 학습률을 따로 골랐다. [§4.2; Appendix B.4, C.2, Table 14](https://arxiv.org/pdf/2610.06833v1#page=32)

## 5. 실험 (Experiments)

조건·지표를 먼저 정의하고 원문 실험 순서로 읽는다.

### 5.1 설정 (Setup)

논문의 S/M/L 표기는 반올림한 **embedding 제외 파라미터 규모**다. 실제 looped 모델의 총 파라미터는 다음과 같다. [§5.1; Appendix B.1, Table 6](https://arxiv.org/pdf/2610.06833v1#page=24)

| 규모 | Looped non-embedding | Embedding 포함 총합 | 사전학습 토큰 |
| --- | --- | --- | --- |
| S, ‘100M’ | 0.106B | 0.304B | 21.47B |
| M, ‘400M’ | 0.425B | 0.819B | 85.90B |
| L, ‘1.6B’ | 1.699B | 2.488B | 343.60B |

세 규모 모두 context 8,192, global batch 512, BF16을 사용한다. 웹·QA·코드·수학 등을 포함한 혼합 데이터에서 학습하며, 설정별 학습률을 탐색한다. 모든 설정은 학습 seed 하나다. 공개 구현 README는 논문 실험의 장비를 NVIDIA H200으로 명시한다. 아래 latency는 특정 batch·prompt 조건의 측정이며 다른 장비의 보편적인 속도가 아니다. [Appendix B.1](https://arxiv.org/pdf/2610.06833v1#page=24), [공식 구현 README](https://github.com/ifm-ai/xllm-loop#installation)

퍼플렉서티(perplexity, PPL)는 토큰 평균 교차엔트로피(cross-entropy, CE)의 지수로 낮을수록 좋다. **AVG는 LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, OpenBookQA, SciQ의 정확도를 동일 가중치로 평균**하며, GSM8K·DROP·MBPP+를 포함하지 않는다. Entropy 부록 일부는 SciQ 대신 WinoGrande를 쓰므로 AVG를 그대로 섞어 비교할 수 없다. GSM8K 본 실험은 1,319문항·8-shot greedy accuracy, DROP은 고정 500문항·3-shot F1, MBPP+는 고정 100문항·3-shot·온도 0.8에서 8개 샘플로 추정한 pass@1이다. [Appendix D.2, D.5, Table 27](https://arxiv.org/pdf/2610.06833v1#page=44)

더 중요한 함정은 PPL의 채점 범위다. **Table 2·3의 terminal-KV PPL과 Table 8·13의 native PPL은 채점 토큰이 다르다.** 전자는 4,096-token prefix 뒤에서 그 안의 문서를 잇는 토큰을 채점하고, 후자는 전체 시퀀스를 채점한다. 두 표의 PPL을 나누어 공유 손실을 계산하면 틀린다. 공유 효과는 같은 토큰으로 두 캐시를 비교한 Table 20·21이나 CacheEval을 사용해야 한다. [Appendix D.3–4](https://arxiv.org/pdf/2610.06833v1#page=45)

### 5.2 깊이 분포 학습 (Learning the Depth Prior)

다음은 Table 2에서 고정 PLN-5를 학습 분포 $\lambda _H=0.01$로 바꾼 결과다. 모두 테스트 깊이 5, terminal KV 4개이며 입력 주입은 Parcae w/o norm이다. 앞 절의 OrthoInj를 결합한 결과가 아니다. [Table 2](https://arxiv.org/pdf/2610.06833v1#page=10)

| 규모 | Val. PPL ↓ | WikiText PPL ↓ | 7-task AVG, % ↑ | GSM8K, % ↑ |
| --- | --- | --- | --- | --- |
| S | 5.73 → 5.67 | 20.19 → 19.91 | 41.32 → 41.65 | 1.52 → 1.59 |
| M | 4.00 → 3.93 | 12.55 → 12.28 | 49.06 → 49.07 | 13.72 → 15.62 |
| L | 3.10 → 3.07 | 8.76 → 8.68 | 59.30 → 60.00 | 47.61 → 47.92 |

PPL은 세 규모에서 개선되지만 Medium AVG 개선은 0.01 percentage point(pp)다. L FLOPs는 $1253\to1273\quad(\times10^{19})$로 약 1.6% 늘어난다. 평균 깊이 예산과 동일 FLOPs 비교는 구분해야 한다.

L에서 학습 분포+terminal KV의 AVG 60.00은 고정 깊이 모델의 full-cache AVG 59.90과 가깝다. 하지만 서로 다른 가중치의 12층 모델 Untied 12의 AVG 61.47보다는 낮다. Untied 12 대비 약 3배 적다는 것은 non-embedding 파라미터와 KV bank 수다. 동일 물리적 깊이의 Untied 4보다 품질은 높지만, 학습 FLOPs는 $1273/465\approx2.74$배다. **파라미터 효율성과 연산 효율성을 같은 주장으로 합치면 안 된다.** [Tables 2, 6, 8](https://arxiv.org/pdf/2610.06833v1#page=28)

같은 토큰으로 비교한 L 학습 분포의 terminal sharing은 $R=5$에서 Val. PPL을 3.047에서 3.073으로 약 0.85% 높이고, GSM8K를 50.42%에서 47.92%로 낮춘다. $R=16/32$에서는 PPL 차이가 0.11% 이하로 줄어든다. ‘거의 손실 없이 공유’는 이런 설정과 지표의 범위에서 읽어야 한다. [Table 20](https://arxiv.org/pdf/2610.06833v1#page=36)

부록의 중요한 대조군도 있다. Small에서 최종 학습 분포를 **첫 업데이트부터 고정**해 사용하면 validation PPL이 학습 분포의 5.67보다 낮은 5.63이다. 학습 과정의 지속적인 적응보다 좋은 분포 모양을 찾은 것이 이득의 원인일 수 있다. Entropy를 높이면 Medium CacheEval 공유 손실은 감소하지만, WikiText나 task 정확도까지 항상 좋아지는 것은 아니다. 평균 깊이가 같아도 실제 FLOPs는 완전히 같지 않다. [Appendix C.1, Tables 10, 12](https://arxiv.org/pdf/2610.06833v1#page=29)

### 5.3 입력 주입 (Input Injection)

Table 3은 고정 PLN-5 분포에서 입력 주입을 비교한다. Parcae-Decay 대비 OrthoInj의 Val. PPL과 AVG는 개선되지만, WikiText와 생성 지표는 일관되게 개선되지 않는다. [Table 3](https://arxiv.org/pdf/2610.06833v1#page=11)

| 규모 | Val. PPL ↓ | WikiText PPL ↓ | AVG, % ↑ | GSM8K, % ↑ | MBPP+, % ↑ |
| --- | --- | --- | --- | --- | --- |
| S | 5.75 → 5.67 | 19.90 → 20.02 | 41.28 → 41.65 | 1.90 → 1.90 | 0.25 → 1.62 |
| M | 3.96 → 3.94 | 12.45 → 12.32 | 49.03 → 49.16 | 15.01 → 13.95 | 8.25 → 7.88 |
| L | 3.09 → 3.08 | 8.72 → 8.72 | 59.22 → 59.26 | 52.92 → 51.71 | 30.88 → 27.75 |

L의 AVG 차이는 0.04pp이며 코드 점수는 낮다. 본문에는 모든 규모에서 WikiText PPL이 가장 낮다는 표현이 있지만, Table 3의 S에서는 Parcae가 더 낮고 L에서는 동률이다. 이 리뷰는 표의 수치를 따른다. Medium에서 다른 학습률을 사용한 부록 Table 15에서도 낮은 PPL이 높은 task 평균으로 항상 이어지지 않는다. [§5.3; Appendix C.2, Table 15](https://arxiv.org/pdf/2610.06833v1#page=33)

Table 16에서 Parcae의 이전 상태 전달항(carryover)은 관측상 입력을 상쇄하기보다 같은 방향으로 증폭한다. 토큰별 입력 gain 중앙값은 S/M/L에서 약 2.18/2.75/3.43이다. 따라서 여기서 관측한 효과는 ‘입력이 사라지는 것을 복구했다’보다 ‘입력 방향의 크기를 일정하게 만들었다’로 설명하는 것이 정확하다. [Appendix C.2, Table 16](https://arxiv.org/pdf/2610.06833v1#page=33)

### 5.4 RL rollout 상태 재사용 (Rollout-State Reuse in RL)

비교는 L learned-prior checkpoint, 깊이 6, Dr. GRPO, 질문당 16개 응답으로 진행한다. GSM8K 학습은 400질문, 평가는 고정 500질문에서 8개 샘플이다. 여기의 pass@1은 8개 결과의 평균 성공률이며 본 실험의 greedy accuracy와 다르다. pass@8은 8개 중 하나라도 맞힌 질문의 비율이다. [§5.4; Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=41)

| 방법 | GSM8K pass@1 / @8, % | MATH500 pass@1, % | 업데이트, 초 ↓ | GSM 전체 학습, 분 ↓ |
| --- | --- | --- | --- | --- |
| RL 이전 | 40.63 / 77.60 | 13.40 | — | — |
| Full BPTT | 63.20 / 88.00 | 18.27 | 2.77 | 65.45 |
| Recompute + Neumann-4 | 60.05 / 89.60 | 17.32 | 2.09 | 61.71 |
| Reuse + Neumann-4 | 61.65 / 90.20 | 17.82 | 1.39 | 57.30 |
| Reuse + Neumann-3 | 61.30 / 89.80 | 18.77 | 1.30 | 58.98 |

업데이트 열은 **rollout과 optimizer step을 제외한 채점·역전파의 중앙값**이다. 2.77→1.39초는 약 1.99배 가속이다. 전체 시간은 65.45→57.30분, 약 1.14배이며 응답 길이와 실제 업데이트 횟수도 다르다. 따라서 전체 시간 차이를 estimator의 순수한 인과 효과로 해석할 수 없다. [Tables 4, 25](https://arxiv.org/pdf/2610.06833v1#page=42)

MBPP+ 코드 RL에서는 업데이트가 1.03→0.51초로 빨라지지만 전체 시간은 **20.0→21.8분으로 늘어난다**. Rollout과 코드 검증이 지배적이기 때문이다. Pass@1은 40.88→39.38, pass@8은 69→64다. [Table 26](https://arxiv.org/pdf/2610.06833v1#page=43)

GSM8K의 reuse와 full BPTT pass@1 차이는 −1.55pp, 질문 단위 대응 부트스트랩(paired bootstrap) 95% 구간은 $[-3.13,0.05]$다. 이 구간은 학습 seed 하나에 조건부이므로 동등성이나 seed 간 안정성의 증명이 아니다. 재사용과 재계산의 gradient cosine은 0.997–0.999지만 full BPTT와는 0.76–0.87이다. 끝점의 상대 잔차도 평균 0.034로 0이 아니다. 오래된 상태를 여러 업데이트 뒤까지 재사용하는 실험은 하지 않았다. [Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=42)

### 5.5 Prefill 증류 (Distilled Prefill)

| 규모 | Teacher → distilled prefill, ms ↓ | Prefill 가속 | AVG, % ↑ | GSM8K, % ↑ | MBPP+, % ↑ |
| --- | --- | --- | --- | --- | --- |
| S | 195.0 → 125.9 | 1.55배 | 41.65 → 40.63 | 1.59 → 1.74 | 0.62 → 0.62 |
| M | 495.8 → 299.3 | 1.66배 | 49.07 → 46.77 | 15.62 → 8.95 | 10.38 → 6.12 |
| L | 1482.1 → 830.3 | 1.79배 | 60.00 → 55.53 | 47.92 → 32.90 | 27.25 → 16.50 |

8K prompt, batch 8에서 warmed prefill 여섯 번의 평균이다. Teacher를 두 번만 반복하는 단순한 대조군보다 약 1–4% 빠르고 AVG는 0.5–0.9pp 높다. 하지만 전체 teacher 대비 L AVG는 4.47pp, GSM8K는 15.02pp 낮다. 첫 토큰 지연, 전체 생성 시간, 낮은 batch의 interactive serving 속도로 이 가속을 바꿔 말할 수 없다. [Table 5, Appendix C.7.1](https://arxiv.org/pdf/2610.06833v1#page=13)

추가 실험은 상태 인계(handoff)의 중요성을 보여준다. Student 자신의 KV를 teacher에 바로 주면 teacher 한 번의 반복을 거친 경로보다 AVG가 2–7pp 낮다. Student를 단독으로 사용해도 Untied 4보다 좋지 않다. 더 깊은 $R=25$ teacher에서는 prefill 가속이 4.3–5.4배지만 품질 손실이 커지며, 그 student는 $R=5$ student보다 느리고 덜 정확하다. 이 비교는 본문의 hidden+KL recipe와 달리 **hidden loss만 사용**한다. Hidden MSE가 더 낮아도 downstream 품질이 더 나쁠 수 있어 단일 상태 거리만으로 decoder의 민감도를 평가하기 어렵다. [Appendix C.7, Tables 22–23](https://arxiv.org/pdf/2610.06833v1#page=39)

## 6. 관련 연구 (Related Work)

원문은 세 묶음으로 관련 연구를 연결한다. 먼저 반복 모델은 공유 블록으로 테스트 시 계산을 늘리고, DEQ는 극한의 고정점을 직접 푼다. 기존 연구는 고정점 안정화·지도 상태를 attractor로 만들기·수렴 시 중단을 다뤘으며, 이 논문은 학습·추론·후속 학습의 효율을 함께 조사한다. [원문 §6](https://arxiv.org/pdf/2610.06833v1#page=13)

둘째, 효율 연구에서는 기존 TBPTT·반복 간 KV 공유를 고정점으로 설명하고 새로운 RL 상태 재사용·비반복 student prefill을 연결한다. 기존 반복 student 증류나 반복별 prompt KV 예측과 달리, student는 끝점만 예측하고 teacher 한 번의 반복이 모든 KV bank를 만든다. [원문 §6, Appendix E](https://arxiv.org/pdf/2610.06833v1#page=46)

셋째, 고정점 형성에서는 지도하는 깊이 분포와 입력 주입을 구분한다. Ouro·RL-Halting은 상태 기반의 엔트로피 정규화 분포를 학습하지만 최대 깊이까지 펼친다. 저자들은 입력 무관의 학습 prior를 쓰고, 수축성을 강제하는 Parcae와 대비해 OrthoInj로 carryover의 입력 방향 성분을 제거한다. [원문 §6](https://arxiv.org/pdf/2610.06833v1#page=13)

## 7. 논의와 결론 (Discussion and Conclusion)

저자의 결론은 반복 경로가 고정점 근처에서 덜 중요해진다는 성질을 이용해, 학습·캐시·RL 업데이트·prefill의 서로 다른 비용을 줄일 수 있다는 것이다. 학습 깊이 분포와 입력 주입은 그런 끝점을 형성하는 두 축이다. 다음 한계는 원문 §7 및 관련 부록의 범위다.

해석의 범위도 좁혀야 한다. 설정별 학습 seed가 하나이며, 더 큰 모델·다른 구조·긴 reasoning completion·실서비스 serving은 검증되지 않았다. Pretraining decontamination도 확인하지 않았다. Learned prior와 OrthoInj를 결합한 최종 시스템의 추가 이득, RL에서 terminal sharing까지 결합한 누적 이득은 이 표들로 계산할 수 없다. [§7; Appendix C.8, D, E.3](https://arxiv.org/pdf/2610.06833v1#page=14)

<span id="리뷰-해석과-후속-검증" aria-hidden="true"></span>

## 리뷰 해석과 남은 질문

**리뷰 해석.** 이 논문의 강점은 ‘반복을 줄인다’는 하나의 표현 속에 섞이기 쉬운 문제들을 분리한 데 있다. 역전파 경로, 과거 토큰의 문맥, RL의 동일 토큰 재계산, 프롬프트 처리의 끝점 근사를 각각 분석한다. 부록은 정확한 극한과 실제 깊이 5, 상태 유사성과 gradient 유사성, 구간 가속과 전체 가속의 차이를 확인할 대조군을 제공한다. 이는 리뷰 작성자의 해석이다.

내가 가장 유용하게 본 설계 원칙은 **생략한 경로가 하던 일을 무엇이 대신하며, 어떤 오차가 남는지 먼저 정의하는 것**이다. Terminal KV는 문맥 오차를, RL reuse는 gradient 근사와 상태 저장 비용을, distilled prefill은 handoff 품질과 추가 학습 비용을 남긴다. 어느 방법도 낮은 PPL 하나로 추론 품질·학습 안정성·실제 서비스 속도를 모두 보장하지 않는다.

본문을 읽은 뒤 부록에서 확인할 부분을 다음처럼 연결할 수 있다. 이 글은 모든 절의 논지를 다루되, 27개 표와 bibliography를 그대로 재현하지 않고 핵심 비교와 반례를 선별했다.

| 원문 범위 | 읽으면서 확인할 질문 |
| --- | --- |
| §1–2, pp. 1–3 | 왜 고정점인가? 토큰별 안정성은 어떻게 다른가? |
| §3, pp. 4–6 | TBPTT, RL state reuse, KV sharing, prefill의 경로가 어떻게 다른가? |
| §4, pp. 6–8 | 깊이 분포와 입력 주입이 상태를 어떻게 바꾸는가? |
| §5, pp. 9–13 | 세 규모에서 어떤 품질·비용 교환이 관측되는가? |
| §6–7, pp. 13–14 | 기존 연구와의 차이와 저자가 밝힌 한계는 무엇인가? |
| A.1–2, pp. 20–23 | KV 공유의 가정과 유한 오차, gradient 근사의 조건은 무엇인가? |
| B.1–4, pp. 23–27 | 모델·학습률·TBPTT·controller·projection 구현은 무엇인가? |
| C.1–2, pp. 27–33 | 고정 분포 대조군, entropy, 정규화·투영·학습률의 영향을 분리했는가? |
| C.3–6, pp. 33–37 | 창 길이·토큰 안정성·깊은 캐시 공유·prefill/decoding 예산은 어떤가? |
| C.7–8, pp. 38–43 | Distillation 실패와 RL의 전체 시간·불확실성은 무엇인가? |
| D.1–5, pp. 43–46 | 같은 지표 이름 아래 데이터와 채점 토큰이 일치하는가? |
| E.1–5, pp. 46–47 | Learned exit·KV sharing·기존 distillation과 무엇이 다른가? |

후속 검증으로는 같은 checkpoint와 동일 토큰에서 $R$, TBPTT 창, prefix 잔차를 바꾸며 품질·메모리·전체 시간을 함께 측정하는 실험을 권한다. 상태 거리나 PPL이 실제 생성 성공률을 얼마나 예측하는지가 핵심이다. Learned prior의 적응 자체가 필요한지 알아보려면 최종 고정 분포와의 비교를 Medium/Large와 여러 seed에서 반복해야 한다. 두 제안 모두 이 리뷰에서 실행하지 않았다.

## 출처와 읽은 범위

- 원문: *Towards Looped Models Done Right, Part II: Rethinking at Fixed Points* · Benhao Huang et al. · 최초 공개 2026-10-05 · [검토 버전](https://arxiv.org/abs/2610.06833v1) · 검토일 2026-10-23.
- 읽은 범위: 본문 §1–7, Appendix A–E의 설명·수식·알고리즘·실험. A.1·A.2의 증명은 대응 방법 절에서 설명한다.
- 추가 출처: 위 네 배경 논문의 공식 초록·서지 정보, [공식 코드 저장소 README](https://github.com/ifm-ai/xllm-loop). 코드 실행과 checkpoint 검증은 하지 않았다. 관련 연구의 세부 비교 중 원문을 직접 읽지 않은 부분은 중심 논문의 해석으로 표시했다.
- 그림 출처: Figure 1(대표)·2·3·4는 저자 v1 원문 자산으로 CC BY 4.0, 원본을 유지한다. [원문](https://arxiv.org/abs/2610.06833v1).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
