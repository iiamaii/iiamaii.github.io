---
title: "끝점으로 반복 경로를 대신할 수 있을까: Looped Models와 고정점의 효율"
description: "고정점 관점으로 TBPTT·KV 공유·RL 상태 재사용·distilled prefill을 분석한다. 전문과 부록을 읽고, 업데이트 가속과 전체 시간·정확도 손실을 구분한다."
date: "2026-10-08"
publishedAt: "2026-10-08T16:55:04+09:00"
updatedAt: "2026-10-08T16:55:04+09:00"
topics: ["language models", "looped models", "fixed points", "efficient inference", "reinforcement learning"]
translationKey: "looped-models-fixed-points"
paperTitle: "Towards Looped Models Done Right, Part II: Rethinking at Fixed Points"
authors: "Benhao Huang, Chufan Shi, Junlin Chen, Shicheng Wen, Zhengzhong Liu, Eric Xing, Xuezhe Ma"
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.06833v1"
visibility: "public"
lang: "ko"
thumbnail: ""
thumbnailAlt: ""
---

반복형 언어 모델이 같은 계산을 거듭하다가 거의 변하지 않는 상태에 도달한다면, 그곳까지의 **모든 계산 경로를 계속 보관하고 다시 실행해야 할까?** 이 논문은 고정점을 ‘계산을 멈출 위치’뿐 아니라 ‘계산 경로를 대신할 상태’로 바라본다. 학습의 역전파, 추론의 KV 캐시, 강화학습의 재계산, 긴 프롬프트의 처리에 이 관점을 적용한다. 다만 캐시 절감, 업데이트 가속, prefill 가속은 서로 다른 실험이며, 정확도와 전체 실행 시간의 대가는 각각 확인해야 한다. [§1–3, §5, PDF pp. 1–13](https://arxiv.org/pdf/2610.06833v1#page=1)

## 어떤 질문에서 출발했는가

일반적인 Transformer는 서로 다른 가중치를 가진 층들을 차례로 통과한다. **Looped language model**은 일부 층의 가중치를 공유하고 같은 블록을 여러 번 실행한다. 가중치를 늘리지 않고 토큰당 계산량을 늘릴 수 있지만, 반복별 중간 상태를 모두 저장하면 학습 메모리와 생성 중 캐시가 반복 횟수만큼 커진다. 긴 입력을 처음 읽는 **prefill**도 비싸진다. 이후 토큰을 하나씩 생성하는 단계는 **decoding**이다. [§1–2](https://arxiv.org/html/2610.06833v1#S2)

저자들은 네 가지 질문을 연결한다. 마지막 몇 번만 역전파해도 되는 이유는 무엇인가? 이전 토큰의 마지막 KV만 재사용해도 되는가? 강화학습에서 이미 생성할 때 계산한 상태를 다시 계산해야 하는가? 프롬프트의 끝점 상태를 작은 모델이 대신 예측할 수 있는가? 이를 가능하게 하는 상태를 만들기 위해 학습 중 반복 횟수의 분포와 입력 주입 방식도 바꾼다. [§3–4](https://arxiv.org/html/2610.06833v1#S3)

## 핵심 아이디어: 끝점으로 경로를 대신하기

실험 모델은 Huginn 구조다. Prelude가 입력을 표현으로 바꾸고, 두 Transformer 블록으로 된 recurrent core가 그 표현을 매번 주입받으며 반복한다. Coda는 최종 상태를 다음 토큰 확률로 바꾼다. 다음 표기의 상태 `z`에는 은닉 상태 `H`와 attention의 key/value 표현을 저장하는 `C`가 함께 들어간다. [§2](https://arxiv.org/html/2610.06833v1#S2)

```text
e = Pθ(x)                         # prelude가 만든 입력 표현
zʳ = (Hʳ, Cʳ)
zʳ⁺¹ = Fθ(zʳ; x)                  # 공유된 recurrent core
z* = Fθ(z*; x)                    # 고정점
```

수학적 고정점에서는 같은 연산을 한 번 더 해도 상태가 같다. 실제 모델에서는 유한한 반복 뒤의 근사적인 정체 상태도 고정점이라고 부른다. 두 의미를 구분해야 한다. 상태가 조금만 바뀐다는 관찰은 예측이 정답이라는 뜻도, 모든 입력에서 같은 속도로 수렴한다는 뜻도 아니다. [§2, Appendix A.2](https://arxiv.org/pdf/2610.06833v1#page=22)

| 적용 구간 | 줄이는 것 | 남아 있는 계산 또는 조건 |
| --- | --- | --- |
| 사전학습의 TBPTT | 오래된 반복의 activation 저장과 역전파 | 순전파는 뽑힌 깊이까지 실행 |
| Terminal KV sharing | 이전 토큰의 반복별 KV bank | 현재 토큰은 계속 반복하며 자기 반복의 KV 사용 |
| RL rollout-state reuse | 생성한 토큰의 끝점을 복원하는 재계산 | 한 번의 core 순전파와 여러 역방향 VJP, rollout 자체 |
| Distilled prefill | 프롬프트의 여러 teacher 반복 | student, teacher 한 번의 반복·coda, 이후 teacher decoding |

VJP(vector–Jacobian product)는 Jacobian 전체를 만들지 않고 벡터에 대한 역방향 미분을 계산하는 연산이다. 네 방법을 모두 결합한 하나의 배포 시스템에서 누적 가속을 측정한 논문으로 읽으면 안 된다. 특히 RL 실험의 생성·평가는 반복별 KV를 유지한다. [§3, Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=40)

**[Figure 3 · 학습·캐시 공유·student prefill의 경로](https://arxiv.org/html/2610.06833v1#S3.F3)** — (a) 점선 아래 상태를 분리하고 마지막 창만 역전파한다. (b) 과거 토큰의 마지막 KV를 여러 현재-token 반복이 읽는다. (c) 비반복 student가 prefix 끝점을 예측하고 teacher가 이후 생성한다. Student 뒤 한 teacher recurrence와 coda를 거치는 세부는 본문 설명을 함께 읽는다. [원문 그림 크게 보기](https://arxiv.org/html/2610.06833v1/prefix_paths.svg).

## 토큰은 같은 순서와 속도로 수렴하지 않는다

원문 그림 2는 반복을 세로축, 토큰 위치를 가로축으로 두고 인접 반복 사이의 상대 은닉 상태 변화를 색으로 표시한다. 뒤 토큰이 앞 토큰보다 먼저 안정되는 경우도 있다. ‘문장의 앞부분부터 차례로 고정된다’는 단순한 설명이 맞지 않는 이유다. 관측한 8개 시퀀스에서 수렴한 토큰들을 모았을 때 위치와 수렴 깊이의 Spearman 상관은 0.14다. 이 수치를 모든 언어·문서에 대한 일반 법칙으로 확대할 수는 없다. [Figure 2, Appendix C.4](https://arxiv.org/pdf/2610.06833v1#page=35)

```text
ρ(r,t) = ||hₜʳ − hₜʳ⁻¹||₂ / (||hₜʳ⁻¹||₂ + ε)
```

그림에서는 관측 구간 끝까지 변화가 2% 아래로 유지되는 첫 깊이를 찾고, 적어도 네 번의 안정된 업데이트를 요구한다. 관측 길이를 64로 늘린 부록 Table 19에서 Small PLN-5 모델의 토큰 99.74%가 1% 기준을 만족한다. 이는 정해진 데이터와 유한한 관측 구간의 **상태 안정성**이다. 반면 훨씬 엄격한 수렴 검사에서는 탈락한 입력도 있고 수백 번의 반복이 필요하다. [Appendix C.4, Table 19; A.1](https://arxiv.org/pdf/2610.06833v1#page=36)

**[Figure 2 · 토큰의 안정화는 순서대로 일어나지 않는다](https://arxiv.org/html/2610.06833v1#S2.F2)** — 어두운 영역은 작은 상대 상태 변화다. 확대 패널의 계단선은 인접 토큰의 안정 깊이가 뒤바뀔 수 있음을 보여준다. 오른쪽 위치·깊이의 약한 상관은 관측한 시퀀스에 한정된다. [원문 그림 크게 보기](https://arxiv.org/html/2610.06833v1/partial_convergence.png).

## 마지막 몇 번만 역전파해도 되는 이유와 그 한계

TBPTT(truncated backpropagation through time)는 마지막 `b`번만 미분하고 그 이전 상태를 분리한다. 정확한 고정점에서 `J*`를 상태에 대한 core의 Jacobian, `B*`를 상태를 고정했을 때 파라미터에 대한 미분으로 정의하면, 다음 관계가 나온다. `I`는 항등행렬이다. [§3.1, 식 1; Appendix A.2, 식 14–17](https://arxiv.org/pdf/2610.06833v1#page=22)

```text
D* = dz*/dθ = (I − J*)⁻¹ B*
   = B* + J*B* + J*²B* + ...       # spectral radius(J*) < 1

D_b = Σ[k=0..b−1] J*ᵏ B*
D* − D_b = J*ᵇ D*
```

마지막 몇 번의 미분은 고정점 민감도의 Neumann 급수를 일부만 남기는 것과 연결된다. `||J*||₂ ≤ κ < 1`이라면 손실 gradient의 생략 오차 상한은 `||B*||₂ ||g||₂ κᵇ/(1−κ)`다. `g`는 끝점 상태에 대한 손실 gradient다. 입력 주입·prelude·출력 head로 직접 이어지는 파라미터 경로도 유지해야 한다. [Appendix A.2, Lemma 2](https://arxiv.org/pdf/2610.06833v1#page=23)

하지만 **유한한 반복의 목적함수와 평형 상태의 목적함수는 다르다.** 끝점 값이 같아도 파라미터에 대한 미분은 다를 수 있다. 실제 TBPTT는 경로 위의 서로 다른 Jacobian을 사용하고, 끝점 Neumann 근사는 같은 끝점 Jacobian을 반복해서 사용한다. 끝점 잔차가 작다는 사실 하나만으로 둘이 가깝다고 보장할 수 없다. [Appendix A.2](https://arxiv.org/pdf/2610.06833v1#page=22)

실험에서도 창을 지나치게 줄이면 품질이 낮아진다. Medium PLN-5에서 `b≥4`의 WikiText PPL은 full BPTT의 1% 안이지만, `b=2`는 약 5.5%, `b=1`은 약 15% 높다. Small 고정 깊이 모델에서는 짧은 창이 깊이 증가와 캐시 공유에 대한 안정성을 높이면서 기본 품질을 떨어뜨린다. 학습 초기부터 끝점 Neumann 근사만 사용한 대조군은 시험한 다섯 학습률에서 모두 붕괴했다. ‘고정점 관점이 있으니 어떤 gradient 근사도 안전하다’는 결론은 성립하지 않는다. [Appendix C.3, Tables 17–18](https://arxiv.org/pdf/2610.06833v1#page=34)

메모리 이득은 별도 측정에서 확인한다. 역전파 창 5/10/15의 peak allocation은 local batch 8에서 48/86/124GB였다. 깊이 30의 full BPTT는 메모리 부족을 일으켰고, 층별 activation recomputation은 28GiB로 낮추지만 창 10 대비 step 시간이 약 1.3배였다. 이 수치를 일반적인 모든 학습 설정의 메모리 절감률로 환산할 수는 없다. [§3.1, Appendix C.3](https://arxiv.org/pdf/2610.06833v1#page=33)

## 마지막 KV만 저장할 수 있는 조건

이 모델은 prelude 1개, core 2개, coda 1개의 물리적 attention 층을 갖는다. `R=5`면 논리적으로 `1+2×5+1=12`번 층을 방문한다. 반복별 캐시는 bank 12개를 저장하지만 terminal sharing은 물리적 층별 마지막 bank 4개만 남긴다. `R=16`이면 34개, `R=32`이면 66개 대신 4개다. 이는 **KV 저장량**의 절감이며 전체 GPU 메모리나 FLOPs가 같은 비율로 줄어든다는 뜻은 아니다. [§3.2, Table 20](https://arxiv.org/pdf/2610.06833v1#page=36)

Lemma 1은 이전 토큰들의 문맥이 수렴하고, 현재 토큰의 업데이트가 해당 영역에서 일관되게 수축하며 문맥 변화에 Lipschitz 연속이면, 함께 반복하는 방식과 수렴한 prefix를 고정한 방식이 같은 극한에 도달한다고 보인다. 유한한 prefix 오차 `η`에 대해서는 다음 두 항이 남는다. [§3.2, Lemma 1; Appendix A.1, 식 12](https://arxiv.org/pdf/2610.06833v1#page=20)

```text
현재 토큰 오차 ≤ κᴿ × 초기 오차 + β(1−κᴿ)/(1−κ) × η
```

첫 항은 현재 토큰을 덜 계산한 오차다. 둘째 항은 부정확하게 고정한 문맥의 오차다. 반복을 아무리 늘려도 두 번째 원인은 사라지지 않는다. 수축 계수 `κ`가 1에 가까우면 문맥 오차가 크게 증폭될 수도 있다.

엄격한 실험에서 304개 prefix 중 수렴 검사를 통과한 것은 고정 깊이 모델 206개, PLN-5 모델 191개다. 통과한 prefix의 수렴 깊이 중앙값은 각각 152.5와 114로, 실제 사용 깊이 5보다 훨씬 크다. 깊이 5에서 고정하면 현재 토큰의 끝점 차이는 엄격하게 수렴한 문맥을 사용했을 때보다 5–6자릿수 크다. 따라서 실제 `R=5`의 공유가 유용하다는 근거는 **정확한 고정점에 도달했다는 주장**보다 유한 오차 분석과 실제 품질 측정에 있다. [Appendix A.1, Figure 6](https://arxiv.org/pdf/2610.06833v1#page=21)

부록 C.6은 prefill과 decoding 깊이를 각각 1–32 사이의 8개 값으로 바꿔 8×8 조합을 비교한다. 충분히 깊게 decode하면 prefill을 5 이상 늘린 이득은 작다. 반대로 얕게 decode하는 모델은 얕은 prefill을 선호하는 경우도 있다. 따라서 ‘입력을 더 오래 읽을수록 항상 좋다’거나 두 깊이를 반드시 같게 해야 한다는 결론은 나오지 않는다. [Appendix C.6, Figure 11](https://arxiv.org/pdf/2610.06833v1#page=37)

## 학습 중 반복 횟수의 분포도 학습한다

Huginn의 기준 분포는 shifted Poisson-lognormal(PLN)이다. 평균 반복 횟수를 5로 맞추되 매번 같은 깊이로 학습하지 않는다. 논문은 이 분포에서 시작한 1–64 깊이의 categorical 확률을 학습한다. **입력별 중단 정책이 아니라 전체 학습에 공통인 분포**이며, 반복 전에 microbatch마다 깊이 하나를 뽑는다. [§4.1, Algorithm 2; Appendix B.3](https://arxiv.org/pdf/2610.06833v1#page=25)

```text
J_prior = −E[stop_gradient(A) log pφ(R)]
          − λ_H H(pφ) + λ_m(E[R] − 5)²
```

`A`는 `exp(−CE)`, 즉 inverse perplexity 기반 reward를 이동평균과 분산으로 조정한 advantage다. 모델 자체는 기존 cross-entropy(CE)로 학습한다. Entropy `H`는 분포가 한 깊이에만 몰리는 것을 막고, 마지막 항은 평균 깊이를 예산 근처에 유지한다. 모델 학습을 위해 이미 계산한 손실을 사용하므로 controller를 위해 별도 모델 순전파를 더하지 않는다. 기본 `λ_H=0.01`, `λ_m=1`, 역전파 창의 최대 길이는 10이다. [§4.1, Appendix B.3](https://arxiv.org/pdf/2610.06833v1#page=25)

그림 4에서는 고정 `R=5` 학습의 주황 곡선이 테스트 깊이를 벗어나면 불안정해진다. 확률적으로 깊이를 바꾸는 PLN과 학습 분포는 비교적 안정적이고, 학습 분포가 PLN보다 PPL을 낮춘다. 그러나 이것은 모든 문제의 정확도가 반복에 따라 단조롭게 좋아진다는 뜻은 아니다. [Figure 4; Appendix C.5](https://arxiv.org/pdf/2610.06833v1#page=8)

**[Figure 4 · 학습 깊이 밖에서의 PPL](https://arxiv.org/html/2610.06833v1#S4.F4)** — 가로축은 테스트 반복 횟수, 세로축은 WikiText PPL로 낮을수록 좋다. 고정 깊이 학습은 5 근처에서 좋지만 깊이를 벗어나면 불안정하다. 확률적·학습 분포는 더 안정적이다. 확대창은 5 근처의 작은 차이를 보여주며, 생성 정확도 그래프는 아니다. [원문 그림 크게 보기](https://arxiv.org/html/2610.06833v1/fixed_vs_pln_depth_scaling.svg).

부록의 중요한 대조군도 있다. Small에서 최종 학습 분포를 **첫 업데이트부터 고정**해 사용하면 validation PPL이 학습 분포의 5.67보다 낮은 5.63이다. 학습 과정의 지속적인 적응보다 좋은 분포 모양을 찾은 것이 이득의 원인일 수 있다. Entropy를 높이면 Medium CacheEval 공유 손실은 감소하지만, WikiText나 task 정확도까지 항상 좋아지는 것은 아니다. 평균 깊이가 같아도 실제 FLOPs는 완전히 같지 않다. [Appendix C.1, Tables 10, 12](https://arxiv.org/pdf/2610.06833v1#page=29)

## OrthoInj: 매 반복에서 입력 방향을 일정하게 유지하기

이전 상태를 감쇠해 넘기는 Parcae 방식은 `Λh+q` 형태다. `q`가 이번 반복에 주입하는 입력, `Λ`가 학습되는 대각 감쇠 행렬이다. 이전 상태에도 `q` 방향 성분이 있으므로 실제 입력 방향의 크기는 달라질 수 있다. OrthoInj는 그 성분을 제거한 뒤 `q`를 더한다. [§4.2, 식 7–8](https://arxiv.org/pdf/2610.06833v1#page=8)

```text
q = Δ ⊙ W e
Q_q = I − qqᵀ / (||q||₂² + ε)
core 입력 = Q_q Λh + q
```

`⊙`는 원소별 곱이다. `q≠0`, `ε=0`이면 `q` 방향 성분이 정확히 `q`로 유지된다. 구현에서는 `ε=10⁻⁶`을 사용한다. 투영이 감쇠 부분의 norm을 늘리지 않는다는 성질이 있어도 **Transformer 전체의 수축성 증명**이 되는 것은 아니다. 또한 이 recipe에는 prelude RMSNorm 제거도 포함된다. 부록의 2×2 비교는 정규화 제거와 투영이 각각 validation PPL에 기여함을 보이지만, 각 설정에서 학습률을 따로 골랐다. [§4.2; Appendix B.4, C.2, Table 14](https://arxiv.org/pdf/2610.06833v1#page=32)

Table 16에서 Parcae의 carryover는 관측상 입력을 상쇄하기보다 같은 방향으로 증폭한다. 토큰별 입력 gain 중앙값은 S/M/L에서 약 2.18/2.75/3.43이다. 따라서 여기서 관측한 효과는 ‘입력이 사라지는 것을 복구했다’보다 ‘입력 방향의 크기를 일정하게 만들었다’로 설명하는 것이 정확하다. [Appendix C.2, Table 16](https://arxiv.org/pdf/2610.06833v1#page=33)

## 실험을 읽기 전에: 크기와 지표의 기준

논문의 S/M/L 표기는 반올림한 **embedding 제외 파라미터 규모**다. 실제 looped 모델의 총 파라미터는 다음과 같다. [§5.1; Appendix B.1, Table 6](https://arxiv.org/pdf/2610.06833v1#page=24)

| 규모 | Looped non-embedding | Embedding 포함 총합 | 사전학습 토큰 |
| --- | --- | --- | --- |
| S, ‘100M’ | 0.106B | 0.304B | 21.47B |
| M, ‘400M’ | 0.425B | 0.819B | 85.90B |
| L, ‘1.6B’ | 1.699B | 2.488B | 343.60B |

세 규모 모두 context 8,192, global batch 512, BF16을 사용한다. 웹·QA·코드·수학 등을 포함한 혼합 데이터에서 학습하며, 설정별 학습률을 탐색한다. 모든 설정은 학습 seed 하나다. 공개 구현 README는 논문 실험의 장비를 NVIDIA H200으로 명시한다. 아래 latency는 특정 batch·prompt 조건의 측정이며 다른 장비의 보편적인 속도가 아니다. [Appendix B.1](https://arxiv.org/pdf/2610.06833v1#page=24), [공식 구현 README](https://github.com/ifm-ai/xllm-loop#installation)

PPL은 토큰 평균 CE의 지수로 낮을수록 좋다. **AVG는 LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, OpenBookQA, SciQ의 정확도를 동일 가중치로 평균**하며, GSM8K·DROP·MBPP+를 포함하지 않는다. Entropy 부록 일부는 SciQ 대신 WinoGrande를 쓰므로 AVG를 그대로 섞어 비교할 수 없다. GSM8K 본 실험은 1,319문항·8-shot greedy accuracy, DROP은 고정 500문항·3-shot F1, MBPP+는 고정 100문항·3-shot·온도 0.8에서 8개 샘플로 추정한 pass@1이다. [Appendix D.2, D.5, Table 27](https://arxiv.org/pdf/2610.06833v1#page=44)

더 중요한 함정은 PPL의 채점 범위다. **Table 2·3의 terminal-KV PPL과 Table 8·13의 native PPL은 채점 토큰이 다르다.** 전자는 4,096-token prefix 뒤에서 그 안의 문서를 잇는 토큰을 채점하고, 후자는 전체 시퀀스를 채점한다. 두 표의 PPL을 나누어 공유 손실을 계산하면 틀린다. 공유 효과는 같은 토큰으로 두 캐시를 비교한 Table 20·21이나 CacheEval을 사용해야 한다. [Appendix D.3–4](https://arxiv.org/pdf/2610.06833v1#page=45)

## 결과와 근거: 학습 분포는 어떤 이득을 주는가

다음은 Table 2에서 고정 PLN-5를 학습 분포 `λ_H=0.01`로 바꾼 결과다. 모두 테스트 깊이 5, terminal KV 4개이며 입력 주입은 Parcae w/o norm이다. 앞 절의 OrthoInj를 결합한 결과가 아니다. [Table 2](https://arxiv.org/pdf/2610.06833v1#page=10)

| 규모 | Val. PPL ↓ | WikiText PPL ↓ | 7-task AVG, % ↑ | GSM8K, % ↑ |
| --- | --- | --- | --- | --- |
| S | 5.73 → 5.67 | 20.19 → 19.91 | 41.32 → 41.65 | 1.52 → 1.59 |
| M | 4.00 → 3.93 | 12.55 → 12.28 | 49.06 → 49.07 | 13.72 → 15.62 |
| L | 3.10 → 3.07 | 8.76 → 8.68 | 59.30 → 60.00 | 47.61 → 47.92 |

PPL은 세 규모에서 개선되지만 Medium AVG 개선은 0.01 percentage point(pp)다. L FLOPs는 `1253→1273 ×10¹⁹`로 약 1.6% 늘어난다. 평균 깊이 예산과 동일 FLOPs 비교는 구분해야 한다.

L에서 학습 분포+terminal KV의 AVG 60.00은 고정 깊이 모델의 full-cache AVG 59.90과 가깝다. 하지만 서로 다른 가중치의 12층 모델 Untied 12의 AVG 61.47보다는 낮다. Untied 12 대비 약 3배 적다는 것은 non-embedding 파라미터와 KV bank 수다. 동일 물리적 깊이의 Untied 4보다 품질은 높지만, 학습 FLOPs는 `1273/465≈2.74`배다. **파라미터 효율성과 연산 효율성을 같은 주장으로 합치면 안 된다.** [Tables 2, 6, 8](https://arxiv.org/pdf/2610.06833v1#page=28)

같은 토큰으로 비교한 L 학습 분포의 terminal sharing은 `R=5`에서 Val. PPL을 3.047에서 3.073으로 약 0.85% 높이고, GSM8K를 50.42%에서 47.92%로 낮춘다. `R=16/32`에서는 PPL 차이가 0.11% 이하로 줄어든다. ‘거의 손실 없이 공유’는 이런 설정과 지표의 범위에서 읽어야 한다. [Table 20](https://arxiv.org/pdf/2610.06833v1#page=36)

## OrthoInj의 개선은 지표마다 다르다

Table 3은 고정 PLN-5 분포에서 입력 주입을 비교한다. Parcae-Decay 대비 OrthoInj의 Val. PPL과 AVG는 개선되지만, WikiText와 생성 지표는 일관되게 개선되지 않는다. [Table 3](https://arxiv.org/pdf/2610.06833v1#page=11)

| 규모 | Val. PPL ↓ | WikiText PPL ↓ | AVG, % ↑ | GSM8K, % ↑ | MBPP+, % ↑ |
| --- | --- | --- | --- | --- | --- |
| S | 5.75 → 5.67 | 19.90 → 20.02 | 41.28 → 41.65 | 1.90 → 1.90 | 0.25 → 1.62 |
| M | 3.96 → 3.94 | 12.45 → 12.32 | 49.03 → 49.16 | 15.01 → 13.95 | 8.25 → 7.88 |
| L | 3.09 → 3.08 | 8.72 → 8.72 | 59.22 → 59.26 | 52.92 → 51.71 | 30.88 → 27.75 |

L의 AVG 차이는 0.04pp이며 코드 점수는 낮다. 본문에는 모든 규모에서 WikiText PPL이 가장 낮다는 표현이 있지만, Table 3의 S에서는 Parcae가 더 낮고 L에서는 동률이다. 이 리뷰는 표의 수치를 따른다. Medium에서 다른 학습률을 사용한 부록 Table 15에서도 낮은 PPL이 높은 task 평균으로 항상 이어지지 않는다. [§5.3; Appendix C.2, Table 15](https://arxiv.org/pdf/2610.06833v1#page=33)

## RL: 이미 만든 상태를 다시 계산하지 않기

강화학습은 응답을 생성한 뒤 같은 토큰들을 다시 채점하고 정책을 업데이트한다. 저자들은 rollout 때의 끝점 `zᴿ`을 저장하고, 파라미터가 바뀌기 전에 그 상태에서 core를 한 번 실행해 미분 그래프를 만든다. 저장 상태가 실제 순전파 값으로 유지되도록 하고, 끝점 Jacobian의 VJP를 반복해 근사 gradient를 얻는다. **Neumann-4는 순전파 네 번이 아니라, 동일 그래프에서 상태 VJP 네 번과 identity 항을 포함한 다섯 항의 합**이다. [Algorithm 1; Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=40)

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

GSM8K의 reuse와 full BPTT pass@1 차이는 −1.55pp, 질문 단위 paired bootstrap 95% 구간은 `[−3.13, 0.05]`다. 이 구간은 학습 seed 하나에 조건부이므로 동등성이나 seed 간 안정성의 증명이 아니다. 재사용과 재계산의 gradient cosine은 0.997–0.999지만 full BPTT와는 0.76–0.87이다. 끝점의 상대 잔차도 평균 0.034로 0이 아니다. 오래된 상태를 여러 업데이트 뒤까지 재사용하는 실험은 하지 않았다. [Appendix C.8](https://arxiv.org/pdf/2610.06833v1#page=42)

## Distilled prefill: 프롬프트의 끝점만 예측하기

비반복 student 두 블록이 teacher의 pre-coda 끝점을 예측한다. Student는 teacher core에서 초기화하고, 정규화한 hidden-state 오차와 teacher 출력 분포에 대한 KL(Kullback–Leibler) divergence로 학습한다. Teacher는 고정하며 student에 teacher 사전학습 토큰의 1/4을 추가로 투입한다. **Student 출력 뒤 teacher core 한 번과 coda가 terminal KV를 만들고, 이후 생성은 teacher가 계속 5번 반복한다.** 전체 언어 모델을 student로 교체한 압축 실험이 아니다. [§3.2, 식 4; §5.5, Appendix C.7](https://arxiv.org/pdf/2610.06833v1#page=38)

| 규모 | Teacher → distilled prefill, ms ↓ | Prefill 가속 | AVG, % ↑ | GSM8K, % ↑ | MBPP+, % ↑ |
| --- | --- | --- | --- | --- | --- |
| S | 195.0 → 125.9 | 1.55배 | 41.65 → 40.63 | 1.59 → 1.74 | 0.62 → 0.62 |
| M | 495.8 → 299.3 | 1.66배 | 49.07 → 46.77 | 15.62 → 8.95 | 10.38 → 6.12 |
| L | 1482.1 → 830.3 | 1.79배 | 60.00 → 55.53 | 47.92 → 32.90 | 27.25 → 16.50 |

8K prompt, batch 8에서 warmed prefill 여섯 번의 평균이다. Teacher를 두 번만 반복하는 단순한 대조군보다 약 1–4% 빠르고 AVG는 0.5–0.9pp 높다. 하지만 전체 teacher 대비 L AVG는 4.47pp, GSM8K는 15.02pp 낮다. 첫 토큰 지연, 전체 생성 시간, 낮은 batch의 interactive serving 속도로 이 가속을 바꿔 말할 수 없다. [Table 5, Appendix C.7.1](https://arxiv.org/pdf/2610.06833v1#page=13)

추가 실험은 handoff의 중요성을 보여준다. Student 자신의 KV를 teacher에 바로 주면 teacher 한 번의 반복을 거친 경로보다 AVG가 2–7pp 낮다. Student를 단독으로 사용해도 Untied 4보다 좋지 않다. 더 깊은 `R=25` teacher에서는 prefill 가속이 4.3–5.4배지만 품질 손실이 커지며, 그 student는 `R=5` student보다 느리고 덜 정확하다. 이 비교는 본문의 hidden+KL recipe와 달리 **hidden loss만 사용**한다. Hidden MSE가 더 낮아도 downstream 품질이 더 나쁠 수 있어 단일 상태 거리만으로 decoder의 민감도를 평가하기 어렵다. [Appendix C.7, Tables 22–23](https://arxiv.org/pdf/2610.06833v1#page=39)

## 관련 연구 속 위치

이 논문은 고정점이나 반복형 Transformer를 처음 제안한 연구가 아니다. 기존 구성요소를 효율 관점에서 연결하고, 끝점 상태를 재사용하는 두 경로와 학습 recipe를 실험한다. 아래 추가 자료는 공식 초록과 서지 정보를 확인했으며, 세부 비교는 이 논문의 §6·Appendix E에 근거한다. 추가 논문 전체를 재검토한 서베이는 아니다.

| 배경 연구 | 연결되는 질문 | 이번 논문의 차이 |
| --- | --- | --- |
| [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377v2), 2019 | 평형 상태를 직접 풀고 implicit differentiation 사용 | 유한 반복 모델에서 TBPTT·KV 공유·RL 재사용을 연결 |
| [Huginn](https://arxiv.org/abs/2502.05171v2), 2025 | 공유 블록의 반복으로 latent test-time compute 확대 | 실험 backbone으로 사용하며 끝점의 재사용 조건을 분석 |
| [Parcae](https://arxiv.org/abs/2604.12946v1), 2026 | 입력 주입의 안정성과 loop scaling | Carryover의 입력 방향 성분을 투영하는 OrthoInj 비교 |
| [RL-Halting](https://arxiv.org/abs/2606.29983v1), 2026 | 학습한 stochastic stopping으로 길이 외삽 안정화 | 입력 무관의 학습 깊이 분포로 평균 예산·깊은 구간 지도 관리 |

Appendix E는 Ouro·MELT·continuous depth batching의 KV 공유 결과가 모델과 평가 protocol에 따라 달라진다고 설명한다. 이 결과는 공유 캐시를 어떤 looped model에도 그대로 적용할 수 있다는 근거가 아니다. Looped student를 계속 반복시키는 기존 distillation과 달리, 여기서는 비반복 student를 prefill에만 사용한다. 이 세부 비교의 추가 원문 검증은 남아 있다. [§6, Appendix E.1–5](https://arxiv.org/pdf/2610.06833v1#page=46)

## 나의 생각: 좋은 끝점과 좋은 생략은 따로 검증해야 한다

이 논문의 강점은 ‘반복을 줄인다’는 하나의 표현 속에 섞이기 쉬운 문제들을 분리한 데 있다. 역전파 경로, 과거 토큰의 문맥, RL의 동일 토큰 재계산, 프롬프트 처리의 끝점 근사를 각각 분석한다. 부록은 정확한 극한과 실제 깊이 5, 상태 유사성과 gradient 유사성, 구간 가속과 전체 가속의 차이를 확인할 대조군을 제공한다. 이는 리뷰 작성자의 해석이다.

내가 가장 유용하게 본 설계 원칙은 **생략한 경로가 하던 일을 무엇이 대신하며, 어떤 오차가 남는지 먼저 정의하는 것**이다. Terminal KV는 문맥 오차를, RL reuse는 gradient 근사와 상태 저장 비용을, distilled prefill은 handoff 품질과 추가 학습 비용을 남긴다. 어느 방법도 낮은 PPL 하나로 추론 품질·학습 안정성·실제 서비스 속도를 모두 보장하지 않는다.

해석의 범위도 좁혀야 한다. 설정별 학습 seed가 하나이며, 더 큰 모델·다른 구조·긴 reasoning completion·실서비스 serving은 검증되지 않았다. Pretraining decontamination도 확인하지 않았다. Learned prior와 OrthoInj를 결합한 최종 시스템의 추가 이득, RL에서 terminal sharing까지 결합한 누적 이득은 이 표들로 계산할 수 없다. [§7; Appendix C.8, D, E.3](https://arxiv.org/pdf/2610.06833v1#page=14)

## 전체 내용의 지도와 남은 질문

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

후속 검증으로는 같은 checkpoint와 동일 토큰에서 `R`, TBPTT 창, prefix 잔차를 바꾸며 품질·메모리·전체 시간을 함께 측정하는 실험을 권한다. 상태 거리나 PPL이 실제 생성 성공률을 얼마나 예측하는지가 핵심이다. Learned prior의 적응 자체가 필요한지 알아보려면 최종 고정 분포와의 비교를 Medium/Large와 여러 seed에서 반복해야 한다. 두 제안 모두 이 리뷰에서 실행하지 않았다.

## 출처와 읽은 범위

- 원문: Benhao Huang, Chufan Shi, Junlin Chen, Shicheng Wen, Zhengzhong Liu, Eric Xing, Xuezhe Ma, **Towards Looped Models Done Right, Part II: Rethinking at Fixed Points** (2026), [arXiv:2610.06833v1](https://arxiv.org/abs/2610.06833v1). 제출일 2026-10-05. 검토일 2026-10-08.
- 검토 범위: 47쪽 PDF의 본문 §1–7, Appendix A–E의 전체 설명·수식·알고리즘·실험 protocol과 표를 읽고 저자 제출 TeX와 대조했다. 그림은 본문 핵심 그림과 인용한 표·수식의 PDF 화면을 확인했다. 모든 47쪽의 시각적 검수를 했다는 의미는 아니다. [HTML](https://arxiv.org/html/2610.06833v1)에서 빠진 C.2 이후 부록은 PDF·제출 소스로 보완했다.
- 추가 출처: 위 네 배경 논문의 공식 초록·서지 정보, [공식 코드 저장소 README](https://github.com/ifm-ai/xllm-loop). 코드 실행과 checkpoint 검증은 하지 않았다. 관련 연구의 세부 비교 중 원문을 직접 읽지 않은 부분은 focal paper의 해석으로 표시했다.
- 그림 출처: 원문 Figure 2·3·4. 각 그림의 연결 위치에서 원문과 해설을 안내한다. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 표는 원문 수치 중 필요한 비교를 골라 재구성했고, 가속 비율·차이는 그 수치로 계산했다.
- **미실행 검증:** GPU 학습·추론 재현, 공개 코드와 checkpoint의 실행 검증, gradient 수치 검사, 독립적인 증명 전개·가정 감사, 추가 seed 실험, 데이터 오염 검사. 본문 수치는 저자 보고 결과이며 직접 관측한 재현 결과가 아니다.
