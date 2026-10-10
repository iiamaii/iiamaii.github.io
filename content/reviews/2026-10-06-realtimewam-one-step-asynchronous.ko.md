---
title: "반복과 대기를 함께 줄이기: RealtimeWAM의 한 단계 비동기 행동 생성"
description: "Teacher 끝점 지도로 행동 생성을 한 단계로 압축하고 KV 준비 이벤트로 전문가 대기를 줄이는 방법을 읽는다. 평균 성공률, 교란 조건, GPU별 누적 가속과 학습 비용을 구분한다."
date: "2026-10-06"
updatedAt: "2026-10-10T18:16:14+09:00"
topics: ["world action model", "robot learning", "policy distillation", "real-time inference", "gpu optimization"]
visibility: "public"
lang: "ko"
translationKey: "realtimewam-one-step-asynchronous"
paperTitle: "RealtimeWAM: One-Step Asynchronous World Action Models"
paperPublishedDate: "2026-10-05"
authors: ["Chengtao Lv", "Jinyang Du", "Shuyi Feng", "Yang Yong", "Shiqiao Gu", "Shunzi Yang", "Ruihao Gong", "Shen Ren", "Tianwei Zhang", "Wenya Wang"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.06617v1"
thumbnail: "/assets/reviews/realtimewam/paper-figure-2.png"
thumbnailAlt: "TACD의 teacher·EMA·student와 shared video KV를 보여주는 원문 Figure 2"
---

월드 액션 모델이 다음 행동을 잘 예측해도, 그 행동을 너무 늦게 내놓으면 로봇 제어에 쓰기 어렵다. RealtimeWAM은 **행동을 만드는 반복 횟수**와 **영상·행동 전문가 사이의 대기**를 따로 줄인다. H100에서 큰 가속을 보고하지만, 그 수치는 여러 실행 최적화를 합친 결과이며 모든 교란 조건이나 장비의 실시간 제어를 보장하지는 않는다. [원문 §4–5, Figure 5, Table 2](https://arxiv.org/html/2610.06617v1#S4)

**표기 안내.** 식은 원문의 $f_{\theta_{\mathrm S}}$, $v_{\theta_{\mathrm S}}$, $a_0^{\mathrm T}$, $u_{\theta_{\mathrm T}}$, $\mathcal L_{\mathrm{TA}}$를 사용한다. $\operatorname{sg}$는 stop-gradient이며, $a_0^\star$라는 정확한 ODE 끝점과 수치 teacher 끝점 $a_0^{\mathrm T}$를 구분한다.

## 어떤 질문에서 출발했는가

월드 액션 모델(WAM)은 환경의 변화에 대한 학습과 행동 정책을 연결한다. 이 논문의 출발점인 Fast-WAM과 Faster-WAM에서는 영상 전문가가 시각적 조건을 만들고, 행동 전문가가 이를 읽으며 여러 미래 시점의 행동 묶음인 **action chunk**를 생성한다. 두 transformer 전문가를 연결하는 구조를 MoT(mixture of transformers)라고 부른다. [RealtimeWAM §3](https://arxiv.org/html/2610.06617v1#S3)

영상 전문가가 한 번만 실행돼도 행동 전문가는 보통 잡음을 여러 번 제거한다. 그 반복을 줄인 뒤에는 영상 전문가 전체가 끝날 때까지 기다리는 실행 방식이 또 다른 병목으로 남는다. 따라서 “한 단계로 생성하기”와 “필요한 정보가 준비되는 즉시 실행하기”는 서로 다른 질문이다. [§1, Figure 1, §4](https://arxiv.org/html/2610.06617v1#S4)

<figure class="review-figure" id="figure-realtimewam-core">
<a href="/assets/reviews/realtimewam/realtimewam-core.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/realtimewam-core.svg" width="1200" height="1010" alt="위: 같은 잡음 행동과 frozen video KV에서 teacher10단계 끝점을 student1단계의 목표로 사용. 아래: 영상 projection과 전처리가 끝나면 KV 준비 이벤트를 기록하고 action attention 직전에 기다리는 한 conditioning block 의존성 도식." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">핵심 개념 · 반복은 학습으로, 대기는 실행 순서로</span>위쪽 TACD에서 teacher는 학습 중에만 사용한다. local consistency loss도 유지한다. 아래 CEWP에서는 두 stream의 projection을 겹칠 수 있지만, action attention은 필요한 KV 준비를 기다린다. 선 길이는 측정 시간이 아니며 실제 실행 trace도 아니다. Faster의 희소 conditioning에서는 KV fusion 완료가 추가 조건이다. 본 리뷰에서 제작한 설명 그림. <span class="figure-links"><a href="/assets/reviews/realtimewam/realtimewam-core.svg" target="_blank" rel="noopener noreferrer">크게 보기</a></span></figcaption>
</figure>

이전 리뷰의 [SyncWorld](https://iiamaii.github.io/reviews/syncworld-visual-calibration/)는 주어진 행동으로 미래 영상을 예측하는 시뮬레이터였다. 여기서는 world 정보를 이용해 행동을 직접 만드는 정책의 추론 비용을 다룬다. 두 글을 `world action model` 주제로 묶되, 같은 구조나 같은 평가 대상으로 취급하지 않는 이유다.

## 핵심 아이디어: 가까운 예측의 일치와 끝점 접근

TACD(Teacher-Anchored Consistency Distillation)는 학생 행동 모델이 한 번의 계산으로 여러 단계 teacher의 끝점에 가까워지도록 학습하는 방법이다. **Consistency distillation**은 같은 궤적의 가까운 두 잡음 상태에서 얻은 최종 예측을 맞춘다. 하지만 유한한 학습에서 두 예측이 잘 맞는 것과 둘이 올바른 teacher 끝점에 가까운 것은 다를 수 있다. RealtimeWAM은 이 잔차를 local error와 global error로 나눠 설명한다. 이 관측을 consistency model의 일반 이론 전체가 틀렸다는 주장으로 읽어서는 안 된다. [§4.1, 식 6, Figure 3, Appendix C.1](https://arxiv.org/html/2610.06617v1#S4.SS1), [Consistency Models §3–4](https://arxiv.org/html/2303.01469v2#S3)

<figure class="review-figure" id="figure-realtimewam-training">
<a href="/assets/reviews/realtimewam/paper-figure-2.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-2.png" width="1283" height="428" alt="Frozen video expert의 shared KV 아래 teacher rollout, EMA target, student action expert와 두 loss를 보여주는 TACD 원문 그림." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 2 · 일치시키는 두 목표</span>오른쪽 noisy action에서 왼쪽 clean action 방향으로 읽는다. 초록 teacher rollout은 끝점 기준을 만들고, 파란 EMA 예측과 빨간 student 예측의 차이는 local consistency로 줄인다. teacher endpoint에 대한 별도 loss가 남은 전체 편차를 제약한다. 눈송이는 frozen, 불꽃은 학습되는 모듈이다. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-2.png" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2610.06617v1#S4.F2">Chengtao Lv et al., 2026, Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원본 유지, 수정·자르기 없음</span></figcaption>
</figure>

teacher는 학생과 **같은 잡음 행동, 같은 고정 영상 KV**에서 출발해 10단계를 진행한다. KV는 attention이 읽는 key/value 표현이다. teacher가 도달한 끝점까지의 전체 변위를 현재 잡음 시간으로 나눈 평균 속도를 학생의 목표로 삼는다. 현재 시각에서 teacher가 내는 순간 속도만 복사하는 것과는 다르다. local consistency 항도 함께 유지한다. [§4.1, 식 7–8](https://arxiv.org/html/2610.06617v1#S4.SS1)

$$
\begin{aligned}
f_{\theta_{\mathrm S}}(a_t,t)&=a_t-t\,v_{\theta_{\mathrm S}}(a_t,t),\\
a_0^{\mathrm T}&=\texttt{Solver}(a_t,t,0;\theta_{\mathrm T}),\\
u_{\theta_{\mathrm T}}(a_t,t)&=\frac{a_t-a_0^{\mathrm T}}{t},\quad t>0,\\
\mathcal L_{\mathrm{TA}}&=\mathbb E_{a_t,t}\!\left[\left\|v_{\theta_{\mathrm S}}(a_t,t)-\operatorname{sg}\!\left[u_{\theta_{\mathrm T}}(a_t,t)\right]\right\|_2^2\right],\\
\mathcal L&=\mathcal L_{\mathrm{CD}}+\lambda\mathcal L_{\mathrm{TA}},\quad \lambda=0.2.
\end{aligned}
$$

$a_t$는 잡음이 섞인 행동, $t$는 0이 clean이고 1이 noise인 시간, $v_{\theta_{\mathrm S}}$는 학생의 속도 예측이다. $f_{\theta_{\mathrm S}}$는 학생의 clean endpoint 예측, $a_0^{\mathrm T}$는 teacher의 수치 적분 끝점, $u_{\theta_{\mathrm T}}$는 그 끝점까지의 평균 속도다. $\mathcal L_{\mathrm{CD}}$는 local consistency loss이고 $\mathcal L_{\mathrm{TA}}$는 teacher anchor loss다. 영상 전문가는 고정하고 행동 전문가의 LoRA 파라미터만 학습한다. teacher와 EMA target은 배포 시 호출하지 않는다. [§4.1, §5.1](https://arxiv.org/html/2610.06617v1#S5.SS1)

이 감독의 한계도 분명하다. 목표는 teacher가 내는 행동이지 환경의 최적 행동을 확인한 정답이 아니다. Appendix C의 오차 상한에는 teacher 수치 적분 오차도 들어가며, 성공률 보장으로 바뀌지 않는다. teacher 단계 수 5·10·20의 RoboTwin overall은 각각 90.39·90.84·90.77%로, 더 많은 teacher 연산이 항상 더 높은 성공률을 주지도 않는다. [식 9, Appendix C.2–C.3, Table 4](https://arxiv.org/html/2610.06617v1#A3)

## 방법과 가정: 필요한 KV 앞에서만 기다리기

CEWP(Cross-Expert Wavefront Pipelining)는 한 GPU의 두 비차단 CUDA stream에서 영상과 행동 전문가의 계산을 겹친다. 영상 블록이 projection과 필요한 KV 전처리를 마치면 준비 이벤트를 기록한다. 행동 쪽 projection은 자기 입력 hidden state가 준비되면 진행하고, 영상 KV를 실제로 읽는 attention 직전에 이벤트를 기다린다. 영상 쪽은 남은 계산을 계속한다. [§4.2, Figure 4, Appendix D.1](https://arxiv.org/html/2610.06617v1#S4.SS2)

<figure class="review-figure" id="figure-realtimewam-pipeline">
<a href="/assets/reviews/realtimewam/paper-figure-4.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-4.svg" width="1137" height="581" alt="위의 순차 video/action 실행과 아래 CEWP의 두 CUDA stream, KV ready 이벤트, action attention 앞 wait, 오른쪽의 block 내부 의존성을 비교한 원문 도식." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 4 · 전체 cache 대기에서 블록 이벤트로</span>위는 영상 전문가 전체가 끝난 뒤 행동 전문가가 시작한다. 아래는 각 영상 블록의 KV가 준비되면 이벤트를 기록하고 행동 attention이 필요한 시점에 기다린다. 오른쪽 projection→attention→remainder 의존성을 먼저 확인하면, 모든 블록을 독립 실행하는 방법이 아니라는 점을 볼 수 있다. 시간축은 실행 순서를 설명한다. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-4.svg" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2610.06617v1#S4.F4">Chengtao Lv et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원본 유지, 수정·자르기 없음</span></figcaption>
</figure>

이 방식은 원래 연산을 삭제하거나 모든 블록을 독립으로 만드는 병렬화가 아니다. 블록 간 순서와 KV 의존성을 보존한다. Faster-WAM의 희소 상호작용에서는 선택된 conditioning 단계와 여러 KV를 합치는 fusion 완료를 따라야 한다. Appendix D의 동일성 설명은 정확한 산술에서의 의존성 보존에 관한 것이며, 실제 GPU의 bitwise 일치를 이 리뷰에서 검사한 것은 아니다. [§4.2, Appendix D.1–D.2](https://arxiv.org/html/2610.06617v1#A4)

여기서 **Faster-WAM은 arXiv:2608.04404**의 future-conditioning 모델이다. 같은 이름의 2608.02365는 Depth-of-Thought를 사용하는 다른 연구이므로 baseline을 합치지 않는다. `*`는 Fast-WAM 기반, `†`는 이 Faster-WAM 기반 RealtimeWAM을 뜻한다. [RealtimeWAM §3, Table 1](https://arxiv.org/html/2610.06617v1#S3), [Faster-WAM Method](https://arxiv.org/html/2608.04404v1#Sx3)

## 결과와 근거: 평균과 하위 조건을 함께 읽기

**원문 Table 1의 관련 8행.** 성공률 단위는 %, 높을수록 좋다. NFE는 denoising 함수 평가 수이며 영상과 행동을 분리했다. RoboTwin overall은 clean/random 평균, LIBERO는 네 suite 평균이다. 단순히 teacher의 반복 횟수만 1로 줄인 행을 보면, 한 단계 생성에 별도 학습이 필요한 이유가 드러난다. [Table 1, Appendix F.1](https://arxiv.org/html/2610.06617v1#S5.T1)

| Model | Video NFE | Action NFE | RoboTwin Clean ↑ | Random ↑ | Overall ↑ | LIBERO ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Fast-WAM | 1 | 10 | 91.82 | 91.19 | 91.51 | 97.0 |
| Fast-WAM, naive one-step | 1 | 1 | 70.42 | 68.12 | 69.27 | 95.0 |
| CD* | 1 | 1 | 89.88 | 89.38 | 89.63 | 96.9 |
| MeanFlow* | 1 | 1 | 85.74 | 85.72 | 85.73 | 96.9 |
| DMD* | 1 | 1 | 88.64 | 87.80 | 88.22 | 96.1 |
| RealtimeWAM* | 1 | 1 | 91.96 | 89.72 | 90.84 | 97.0 |
| Faster-WAM | 1 | 10 | 93.20 | 92.66 | 92.93 | 98.9 |
| RealtimeWAM† | 1 | 1 | 92.98 | 92.30 | 92.64 | 99.0 |

Fast 대비 RealtimeWAM*의 RoboTwin 평균 손실은 **0.67%포인트(pp)**, Faster 대비 †는 **0.29pp**다. LIBERO는 각각 같은 점수와 0.1pp 증가를 보고한다. 이 작은 평균 차이만으로 통계적 동등성이나 모든 과제의 품질 유지를 결론낼 수는 없다. 일부 baseline은 앞선 연구에서 인용됐으며 전부 같은 새 실행으로 재현한 표도 아니다. [§5.3, Table 1](https://arxiv.org/html/2610.06617v1#S5.SS3)

anchor 자체의 효과를 읽을 때는 학습되는 전문가도 맞춰야 한다. Table 1의 CD*와 달리, Table 3의 **같은 영상 고정·행동 전용 학습**에서는 CD 89.85%→TACD 90.84%로 0.99pp 증가한다. 그러나 random 열은 89.94%→89.72%로 낮아진다. 모든 하위 조건이 좋아졌다는 설명은 이 표와 맞지 않는다. [Table 3](https://arxiv.org/html/2610.06617v1#S5.T4)

**원문 Table 2에서 Faster와 RealtimeWAM†의 모든 교란 열.** LIBERO-Plus는 카메라·로봇·언어·조명 등 학습 밖 조건을 평가한다. 마지막 변화 열은 리뷰에서 뺄셈으로 계산한 pp이며, overall은 평가 trial 수로 가중한 평균이다. 일곱 숫자의 산술평균으로 다시 계산하지 않는다. [Table 2, Appendix F.1](https://arxiv.org/html/2610.06617v1#S5.T2)

| Perturbation | Faster-WAM | RealtimeWAM† | Change (pp) |
| --- | --- | --- | --- |
| Camera | 53.8 | 54.2 | +0.4 |
| Robot | 71.6 | 69.0 | −2.6 |
| Language | 94.7 | 92.5 | −2.2 |
| Lighting | 96.3 | 94.7 | −1.6 |
| Background | 61.3 | 59.4 | −1.9 |
| Noise | 63.6 | 64.6 | +1.0 |
| Layout | 79.1 | 79.3 | +0.2 |
| Overall | 73.6 | 73.0 | −0.6 |

전체 손실은 0.6pp지만 robot 조건은 2.6pp, language는 2.2pp 낮아진다. 따라서 “손실 1% 미만”이라는 전체 평균의 요약을 모든 교란 조건에 그대로 적용할 수 없다. 정확한 자체 평가 rollout 수·반복 seed·신뢰구간은 이번에 읽은 focal 범위에서 확정하지 못했다. baseline 원 논문의 trial 수를 자동으로 대입하지 않았다.

## 속도와 학습 비용: 25배는 어디서 왔는가

<figure class="review-figure" id="figure-realtimewam-speed">
<a href="/assets/reviews/realtimewam/paper-figure-5.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-5.svg" width="1152" height="263" alt="H100에서 Fast-WAM과 Faster-WAM에 TACD, CUDA Graph, CEWP, 효율 커널을 차례로 적용한 지연 막대와 누적 가속 비율." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 5 · 가속을 누적해서 읽기</span>왼쪽 Fast와 오른쪽 Faster를 나눠, 위에서 아래로 하나씩 추가되는 최적화를 읽는다. 가로축은 지연(ms), 낮을수록 빠르다. 파란 speedup 표기는 시작 baseline 대비 누적 값이며, 연결선의 값은 직전 단계 대비 추가 가속이다. 마지막 약25배를 TACD나 CEWP 하나의 효과로 읽으면 안 된다. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-5.svg" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2610.06617v1#S5.F5">Chengtao Lv et al., 2026, Figure 5</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원본 유지, 수정·자르기 없음</span></figcaption>
</figure>

H100에서 Fast-WAM의 299.7ms는 TACD만 적용하면 65.8ms, CUDA Graph를 추가하면 23.3ms, CEWP까지 적용하면 17.4ms, 효율 커널까지 적용하면 12.2ms가 된다. Faster 경로는 218.9→57.1→26.2→22.4→16.1ms다. 논문이 보고한 누적 가속은 24.55배와 13.56배다. CEWP의 추가 기여는 Graph 적용 이후 약 1.34배와 1.17배다. [§5.4, Figure 5](https://arxiv.org/html/2610.06617v1#S5.F5)

**원문 Figure 5(H100)와 Table F.3(추가 GPU)의 baseline→최종 지연(ms).** 영상 VAE 인코딩을 포함하고 에피소드마다 캐시하는 text 인코딩은 제외한다. 센서 취득·통신·모터 구동까지의 전체 지연이 아니다. [Figure 5](https://arxiv.org/html/2610.06617v1#S5.F5), [Appendix F.3](https://arxiv.org/html/2610.06617v1#A6.SS3)

| Backbone | H100 | RTX 5090 | RTX 4090D |
| --- | --- | --- | --- |
| Fast → RealtimeWAM* | 299.7→12.2 | 251.5→17.8 | 480.8→27.2 |
| Faster → RealtimeWAM† | 218.9→16.1 | 214.1→30.9 | 362.4→44.7 |

12.2ms는 H100에서 30Hz의 한 호출 예산인 약 33.3ms 안에 들어간다. 그러나 이것만으로 폐쇄 루프 전체가 30Hz라고 할 수는 없다. †의 RTX 4090D 44.7ms는 그 호출 예산도 넘는다. 소비자 GPU에서 누적 가속 폭이 달라지는 점과 평균 지연 외의 p95/p99, 관측이 행동까지 얼마나 오래됐는지를 확인해야 한다는 판단은 리뷰 해석이다.

**원문 Table A.1의 학습 비용.** 16 H100, 30,000 iteration 조건이다. 영상 고정으로 줄이는 메모리와 teacher anchor로 늘어나는 시간을 구별할 수 있다. [Appendix A](https://arxiv.org/html/2610.06617v1#A1)

| Training setup | Time | Peak memory |
| --- | --- | --- |
| CD, video + action tuned | 8 h 1 m | 65.98 GiB |
| CD, action only | 5 h 21 m | 24.73 GiB |
| TACD, action only | 8 h 3 m | 24.73 GiB |

행동 전용 CD와 TACD의 peak 메모리는 같지만, 학습 시간은 5시간 21분에서 8시간 3분으로 약 50.5% 늘어난다. 배포 비용을 줄이는 대신 teacher rollout 계산을 학습에 지불하는 셈이다. 실제 로봇에서는 AgileX PiPER 양팔과 RTX 4090의 티셔츠 접기 rollout을 제시한다. 이는 질적 시연이며 공통 trial 수·성공률·closed-loop 주파수의 통계 검증을 대신하지 않는다. [Table A.1, Appendix F.6, Figure F.1](https://arxiv.org/html/2610.06617v1#A6.SS6)

## 관련 연구에서 이어지는 질문

핵심 관련 연구는 직접 backbone인 **Fast-WAM·Faster-WAM**, 한 단계 학습 원리인 **Consistency Models·MeanFlow**, 실행 연결을 다루는 **RTC**의 다섯 편이다. Fast/Faster의 미래 정보 사용 차이를 먼저 이해한 뒤, 평균 속도와 끝점 지도, chunk 실행과 forward 내부 대기를 구별하면 비교가 쉬워진다. 각 원문과 실제 읽은 범위는 마지막 절에 적었다.

최근 대표 다섯 편은 다른 비용 축을 보여준다. **DIDO**는 상호작용 중심 영상 토큰과 영상 denoising을 압축하므로 이름의 one-step을 행동 1회로 읽으면 안 된다. **AnyStep-WAM**은 고정 1회 대신 관측별로 예산을 바꾸지만, 그 risk proxy는 물리 실패 확률로 보정된 지표가 아니다. **OpenWAM**의 world/action noise-state 동기화는 CEWP의 CUDA 실행 겹침과 다른 층이고, 양방향 구조에서는 독립 KV 선계산 가정을 다시 확인해야 한다. **ReWAM**은 행동에 유용한 표현을 바꾸며, **WAM-OPD**는 학생이 방문한 이력에 teacher 감독을 붙인다. 서로 다른 GPU·과제 subset의 성공률을 하나의 순위로 만들지는 않았다. [DIDO §3](https://arxiv.org/html/2609.15570v2#S3), [AnyStep §4](https://arxiv.org/html/2609.33748v2#S4), [OpenWAM §3–4](https://arxiv.org/pdf/2609.07398v1), [ReWAM §3](https://arxiv.org/html/2609.38163v1#S3), [WAM-OPD §3–4](https://arxiv.org/html/2608.22364v2#S3)

## 나의 생각과 남은 질문

이 논문의 유용한 관점은 “모델을 작게 만들기”만으로 효율을 설명하지 않고, 반복 횟수와 실행 의존성을 분리한 데 있다. 특히 가속 기여를 단계별로 보여주는 Figure 5와 조건을 맞춘 Table 3이 구현 방향을 판단하는 데 도움이 된다. 반면 teacher의 끝점에 가깝다는 사실은 teacher가 낯선 상태에서도 옳다는 뜻이 아니다. 학생의 방문 분포와 접촉 실패를 보는 WAM-OPD·AnyStep의 질문을 함께 가져갈 필요가 있다. 마지막 판단은 리뷰 해석이며 결합 효과를 검증한 결과가 아니다.

다음 검증은 같은 GPU의 순차/CEWP 실행에서 출력 오차·event trace·p95/p99 지연을 먼저 측정하는 방향을 권한다. 이후 fixed1과 adaptive budget을 같은 평균 연산 예산으로 비교하고, RTC와 결합했을 때 관측 freshness가 개선되는지 확인할 수 있다. 이 검증과 코드·GPU·실물 재현, 통계적 유의성, 보충 동영상 확인은 **미실행**이다.

## 출처와 읽은 범위

- 중심: Chengtao Lv 외, *RealtimeWAM: One-Step Asynchronous World Action Models*, 2026. [arXiv:2610.06617v1](https://arxiv.org/abs/2610.06617v1), 최초 제출 2026-10-05, 확인 2026-10-06. 본문 §1–6, 선택 부록 A·C.1–C.3·D.1–D.2·E·F.1/F.3/F.6. 20쪽 PDF 접근·텍스트 추출, Figure 2·4·5와 PDF p.5·6·9 시각 확인. 모든 부록과 증명을 독립 검산했다는 뜻은 아니다.
- 핵심: [Fast-WAM v2](https://arxiv.org/html/2603.16666v2), §3.1–3.3·§4.1–4.2 설정·§5; [Faster-WAM v1](https://arxiv.org/html/2608.04404v1), Method와 Experiment Setup/Main Results 관련 문단; [Consistency Models v2](https://arxiv.org/html/2303.01469v2), §3–4; [Mean Flows v1](https://arxiv.org/html/2505.13447v1), §3·§4.1·§5 초기 설정; [RTC v2](https://arxiv.org/html/2506.07339v2), §3 관련 방법·§4 설정·§6 한계.
- 최근 대표: [DIDO v2](https://arxiv.org/html/2609.15570v2), §3.1–3.4·§4 초기 설정/Table 1 설명; [AnyStep-WAM v2](https://arxiv.org/html/2609.33748v2), §4.1–4.3·§5.1/Table 1; [OpenWAM v1](https://arxiv.org/pdf/2609.07398v1), PDF p.1·5·8·13–16·33 관련 문단; [ReWAM v1](https://arxiv.org/html/2609.38163v1), 초록·§3.2–3.4; [WAM-OPD v2](https://arxiv.org/html/2608.22364v2), §3.1–3.3·§4.1–4.2/Table 2·§5. 최근 창은 최초 제출 2026-08-01–10-06이며, 관련성으로 고른 대표 다섯 편이지 최신 전체 다섯 편은 아니다.
- 원문 Figure 2·4·5: Chengtao Lv 외, 2026, [동일 v1 원문](https://arxiv.org/html/2610.06617v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 원본 바이트를 유지했으며 수정·자르기 없음. 첫 그림만 이 리뷰의 설명용 제작물이다.
- 모든 수치는 저자 보고이고 독립 재현은 미실행이다. 별도 원어민 영문 교정도 미실행이다.
