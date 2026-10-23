---
title: "반복과 대기를 함께 줄이기: RealtimeWAM의 한 단계 비동기 행동 생성"
description: "Teacher 끝점 지도로 행동 생성을 한 단계로 압축하고 KV 준비 이벤트로 전문가 대기를 줄이는 방법을 읽는다. 평균 성공률, 교란 조건, GPU별 누적 가속과 학습 비용을 구분한다."
date: "2026-10-06"
updatedAt: "2026-10-23T15:26:12+00:00"
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

RealtimeWAM은 교사 기준 일관성 증류(Teacher-Anchored Consistency Distillation, TACD)로 행동 생성을 한 단계로 줄이고, 두 전문가의 계산을 블록 단위로 겹쳐 추론 지연을 낮춘다. H100의 Fast-WAM 기반 설정에서 CUDA Graph·효율 커널까지 더한 지연은 299.7ms에서 12.2ms로 줄고, RoboTwin 평균 성공률은 10단계 Fast-WAM의 91.51%에서 90.84%로 바뀐다. [원문 Table 1·Figure 5](https://arxiv.org/html/2610.06617v1#S5.T1) **리뷰 해석.** 초록의 "1% 미만 하락"은 벤치마크 평균에는 맞지만 LIBERO-Plus의 robot 교란(−2.6%포인트)에는 맞지 않고, 약 25배 가속 가운데 두 기법이 직접 낸 몫은 TACD의 4.56배와 CUDA Graph 뒤 파이프라이닝의 추가 1.34배다.

원문 §1–6 순서로 구성했다. 부록 C의 오차 유도는 §4.1 바로 뒤에서, 부록 A의 한계와 F.6의 실물 시연은 결론 뒤에서 다룬다.

## 1. 도입 (Introduction)

월드 액션 모델(world action model, WAM)은 미래 시각 예측을 학습 목표에 더해, 영상 생성 백본의 시각 표현으로 행동 예측을 이끈다. 이 논문의 출발점인 Fast-WAM과 Faster-WAM에서는 영상 전문가가 시각적 조건을 만들고, 행동 전문가가 이를 읽으며 여러 미래 시점의 행동을 묶은 **행동 묶음**(action chunk)을 생성한다. 두 트랜스포머 전문가를 어텐션으로 연결하는 이 구조를 트랜스포머 혼합(Mixture-of-Transformers, MoT) 구조라고 부른다. [RealtimeWAM §3](https://arxiv.org/html/2610.06617v1#S3)

저자는 남은 병목을 둘로 나눈다. 첫째는 전문가 내부 반복(intra-expert iteration)이다. 영상 전문가는 한 번만 실행되지만 행동 전문가는 잡음을 여러 번 제거하며, Figure 1에서 행동 전문가가 여러 GPU에서 종단 지연의 약 90%를 차지한다. 둘째는 전문가 간 대기(inter-expert waiting)로, 행동 전문가가 영상 전문가의 순전파가 모두 끝난 뒤에야 시작한다. [§1, Figure 1, §4](https://arxiv.org/html/2610.06617v1#S4)

<figure class="review-figure" id="figure-realtimewam-core">
<a href="/assets/reviews/realtimewam/realtimewam-core.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/realtimewam-core.svg" width="1200" height="1010" alt="위: 같은 잡음 행동과 frozen video KV에서 teacher 10단계 끝점을 student 1단계의 목표로 사용. 아래: 영상 projection과 전처리가 끝나면 KV 준비 이벤트를 기록하고 action attention 직전에 기다리는 한 conditioning block 의존성 도식." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">핵심 개념 · 반복은 학습으로, 대기는 실행 순서로</span>위쪽 TACD에서 교사는 학습 중에만 쓰며, 국소 일관성 손실도 유지한다. 아래 CEWP에서는 두 스트림의 투영을 겹칠 수 있지만, 행동 어텐션은 필요한 KV 준비를 기다린다. 선 길이는 측정 시간이 아니며 실제 실행 trace도 아니다. Faster의 희소 conditioning에서는 KV fusion 완료가 추가 조건이다. 본 리뷰에서 제작한 설명 그림. <span class="figure-links"><a href="/assets/reviews/realtimewam/realtimewam-core.svg" target="_blank" rel="noopener noreferrer">크게 보기</a></span></figcaption>
</figure>

이전 리뷰의 [SyncWorld](https://iiamaii.github.io/reviews/syncworld-visual-calibration/)는 주어진 행동으로 미래 영상을 예측하는 시뮬레이터였다. 이 글은 world 정보를 이용해 행동을 직접 만드는 정책의 추론 비용을 다룬다. 두 글은 `world action model` 주제로 묶이지만 구조와 평가 대상이 다르다.

## 2. 관련 연구 (Related Work)

### 2.1 월드 액션 모델 (World Action Models)

저자들은 DreamZero·Cosmos처럼 영상과 행동을 공유 백본에서 생성하는 WAM과, Motus처럼 영상·행동 전문가를 어텐션으로 연결하는 MoT 구조를 구분한다. Fast-WAM은 영상 표현을 한 번 계산해 재사용하고, Light-WAM·Faster-WAM은 효율·분포 밖 강건성을 더 개선한다. 이 계보가 RealtimeWAM의 직접 출발점이다. [원문 §2.1](https://arxiv.org/html/2610.06617v1#S2.SS1)

### 2.2 적은 단계 증류 (Few-Step Distillation)

점진적 증류, 일관성 모델, 분포 일치 증류, MeanFlow, 적대적 증류는 반복 샘플링을 줄이는 선행 방법이다. 가장 가까운 Flash-WAM은 모달리티별 잡음 체계의 차이를 다루지만 추론 때 미래 영상의 잡음 제거를 유지하며, 한 단계 설정에서 성능 하락이 남는다. 저자들은 이 지점에서 MoT 기반 모델의 행동만 한 단계로 증류하는 문제를 설정한다. [원문 §2.2](https://arxiv.org/html/2610.06617v1#S2.SS2)

### 2.3 로봇 정책의 비동기 추론 (Asynchronous Inference for Robot Policies)

원문은 다음 행동 묶음 예측과 현재 묶음 실행을 겹치는 실행 수준, 느린 world/semantic 모듈과 빠른 행동 모듈을 나누는 모듈 수준, 위치·묶음·모달리티별 샘플링 일정을 달리하는 샘플링 수준을 구분한다. RealtimeWAM은 이보다 세밀하게, 한 번의 순전파 안에서 블록 수준으로 비동기 실행을 한다. [원문 §2.3](https://arxiv.org/html/2610.06617v1#S2.SS3)

### 리뷰 보충: 다른 효율 축

DIDO는 영상 토큰·denoising 압축, AnyStep-WAM은 관측별 예산, OpenWAM은 world/action 잡음 상태, ReWAM은 행동에 유용한 표현, WAM-OPD는 학생이 방문하는 이력의 감독을 다룬다. [DIDO §3](https://arxiv.org/html/2609.15570v2#S3), [AnyStep §4](https://arxiv.org/html/2609.33748v2#S4), [OpenWAM §3–4](https://arxiv.org/pdf/2609.07398v1), [ReWAM §3](https://arxiv.org/html/2609.38163v1#S3), [WAM-OPD §3–4](https://arxiv.org/html/2608.22364v2#S3)

## 3. 배경 (Preliminaries)

원문 식 (1)–(2)에서 관찰 $o$와 언어 지시 $l$은 영상 전문가의 블록별 KV 조건 $z(o,l)$로 바뀐다. KV는 어텐션(attention)이 읽는 키·값(key/value) 표현이며, 이를 저장한 것이 KV 캐시(KV cache)다. $N$은 영상 트랜스포머 블록 수, $a$는 행동 묶음, $\psi$와 $\theta$는 영상·행동 전문가의 파라미터다. 영상 조건은 한 번만 계산하지만, 행동은 여전히 여러 단계의 잡음 제거로 생성한다.

### 3.1 효율적인 MoT 기반 WAM (Efficient Mixture-of-Transformers (MoT)-based WAMs)

$$
\begin{aligned}
z(o,l)&=F_{\psi}^{\mathrm{KV}}(o,l)=\{(K_i^v,V_i^v)\}_{i=1}^{N},\\
p_{\theta}(a\mid o,l)&=p_{\theta}(a\mid z(o,l)).
\end{aligned}
$$

### 3.2 행동 생성의 일관성 증류 (Consistency Distillation for Action Generation)

$$
\begin{aligned}
a_t&=(1-t)a_0^{\mathrm{data}}+t\epsilon,\quad\epsilon\sim\mathcal N(0,I),\\
f_\theta(a_t,t)&=a_t-t\,v_\theta(a_t,t),\\
\tilde a_s&=\texttt{Solver}(a_t,t,s;\theta_{\mathrm T}),\quad 0\leq s<t,\\
\mathcal L_{\mathrm{CD}}&=\mathbb E_{a_t,t,s}\!\left[
\left\|f_{\theta_{\mathrm S}}(a_t,t)-\operatorname{sg}\!\left[
f_{\theta_{\mathrm{ema}}}(\tilde a_s,s)\right]\right\|_2^2\right].
\end{aligned}
$$

식 (3)–(5)의 일관성 증류(consistency distillation, CD)는 고정된 교사 $\theta_{\mathrm T}$, 학습되는 학생 $\theta_{\mathrm S}$, 지수 이동 평균(exponential moving average, EMA) 타깃 $\theta_{\mathrm{ema}}$ 세 모델을 쓴다. 데이터의 깨끗한 행동 $a_0^{\mathrm{data}}$와 가우스 잡음 $\epsilon$을 섞어 $a_t$를 만들고, 교사는 $t$에서 가까운 $s<t$로 한 번 적분해 $\tilde a_s$를 만든다. 학습은 학생의 끝점 예측을 EMA 타깃의 끝점 예측에 맞춘다. $v_\theta$는 속도 예측, $f_\theta$는 깨끗한 끝점 예측, $\operatorname{sg}$는 기울기 차단(stop-gradient)이고, EMA 타깃은 학생 파라미터를 갱신마다 지수 이동 평균한 것이다. [§3.1–3.2, Eqs. (1)–(5)](https://arxiv.org/html/2610.06617v1#S3)

## 4. RealtimeWAM

전문가 내부 반복은 TACD(§4.1)로, 전문가 간 대기는 웨이브프런트 파이프라이닝(§4.2)으로 줄인다.

### 4.1 교사 기준 일관성 증류 (Teacher-Anchored Consistency Distillation)

TACD는 학생 행동 모델이 한 번의 계산으로 여러 단계 교사의 끝점(endpoint)에 가까워지도록 학습한다. CD는 같은 궤적의 가까운 두 잡음 상태에서 얻은 끝점 예측을 맞추지만, 저자는 이를 WAM에 바로 적용하면 국소-전역 오차 간극(local-global error gap)이 생긴다고 본다. Figure 3에서 CD는 일관성 손실을 줄이지만 학생 속도와 교사 끝점 평균 속도의 거리는 크게 줄이지 못한다. 저자는 이 잔차를 국소 오차(local error)와 전역 오차(global error)로 나눠 설명한다. [§4.1, 식 6, Figure 3, Appendix C.1](https://arxiv.org/html/2610.06617v1#S4.SS1), [Consistency Models §3–4](https://arxiv.org/html/2303.01469v2#S3)

<figure class="review-figure" id="figure-realtimewam-training">
<a href="/assets/reviews/realtimewam/paper-figure-2.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-2.png" width="1283" height="428" alt="Frozen video expert의 shared KV 아래 teacher rollout, EMA target, student action expert와 두 loss를 보여주는 TACD 원문 그림." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 2 · 일치시키는 두 목표</span>오른쪽 잡음 행동에서 왼쪽 깨끗한 행동 방향으로 읽는다. 초록 교사 rollout이 끝점 기준을 만들고, 파란 EMA 예측과 빨간 학생 예측의 차이는 국소 일관성 손실로 줄인다. 교사 끝점에 대한 별도 손실이 남은 전체 편차를 제약한다. 눈송이는 고정, 불꽃은 학습되는 모듈이다. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-2.png" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2610.06617v1#S4.F2">Chengtao Lv et al., 2026, Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원본 유지, 수정·자르기 없음</span></figcaption>
</figure>

교사는 학생과 **같은 잡음 행동, 같은 고정 영상 KV**에서 출발해 10단계를 진행한다. 교사·학생·EMA 타깃 세 행동 전문가는 고정된 영상 전문가 하나의 KV 캐시를 공유한다. $a_t$에서 시작한 교사 ODE의 정확한 끝점 $a_0^\star$는 실제로 얻을 수 없으므로, 교사가 도달한 끝점까지의 전체 변위를 현재 잡음 시간으로 나눈 구간 평균 속도를 학생의 목표로 삼는다. 현재 시각에서 교사가 내는 순간 속도만 복사하는 것과는 다르다. 국소 일관성 항도 함께 유지한다. [§4.1, 식 7–8](https://arxiv.org/html/2610.06617v1#S4.SS1)

$$
\begin{aligned}
f_{\theta_{\mathrm S}}(a_t,t)&=a_t-t\,v_{\theta_{\mathrm S}}(a_t,t),\\
a_0^{\mathrm T}&=\texttt{Solver}(a_t,t,0;\theta_{\mathrm T}),\\
u_{\theta_{\mathrm T}}(a_t,t)&=\frac{a_t-a_0^{\mathrm T}}{t},\quad t>0,\\
\mathcal L_{\mathrm{TA}}&=\mathbb E_{a_t,t}\!\left[\left\|v_{\theta_{\mathrm S}}(a_t,t)-\operatorname{sg}\!\left[u_{\theta_{\mathrm T}}(a_t,t)\right]\right\|_2^2\right],\\
\mathcal L&=\mathcal L_{\mathrm{CD}}+\lambda\mathcal L_{\mathrm{TA}},\quad \lambda=0.2.
\end{aligned}
$$

$a_t$는 잡음이 섞인 행동, $t$는 0이 깨끗한 행동이고 1이 순수 잡음인 시간, $v_{\theta_{\mathrm S}}$는 학생의 속도 예측이다. $f_{\theta_{\mathrm S}}$는 학생의 끝점 예측, $a_0^{\mathrm T}$는 교사의 수치 적분 끝점, $u_{\theta_{\mathrm T}}$는 그 끝점까지의 평균 속도다. $\mathcal L_{\mathrm{CD}}$는 국소 일관성 손실이고 $\mathcal L_{\mathrm{TA}}$는 교사 앵커 손실(teacher-anchored loss)이다. 영상 전문가는 고정하고 행동 전문가의 저랭크 적응(low-rank adaptation, LoRA) 파라미터만 학습한다. 교사와 EMA 타깃은 배포 시 호출하지 않는다. [§4.1, §5.1](https://arxiv.org/html/2610.06617v1#S5.SS1)

교사 끝점의 정확도는 교사 단계 수 $K$에 달려 있다. $K$=5·10·20의 RoboTwin overall은 각각 90.39·90.84·90.77%이고, 저자는 $K=20$이 추가 개선 없이 교사 연산만 늘린다며 $K=10$을 택한다. [식 9, Appendix C.2–C.3, Table 4](https://arxiv.org/html/2610.06617v1#A3)

#### 부록 C.1–C.3의 유도: 끝점 오차는 무엇을 보장하는가?

아래는 부록 C를 해당 방법 옆으로 옮긴 해설이다. 같은 조건 $z(o,l)$과 학습 표본 분포를 고정하고, 교사 ODE의 해가 유일하며 오차·속도 잔차의 2차 모멘트가 유한하다고 가정한다. 먼저 EMA 예측을 더하고 빼면 식 (6)의 두 오차가 되고, 제곱을 펼치면 교차항이 남는다.

총 오차(total error) $\mathbf e_{\mathrm{total}}=f_{\theta_{\mathrm S}}(a_t,t)-a_0^\star$는 학생 끝점 예측과 $a_0^\star$의 차이다. 국소 오차 $\mathbf e_{\mathrm{local}}=f_{\theta_{\mathrm S}}(a_t,t)-f_{\theta_{\mathrm{ema}}}(\tilde a_s,s)$는 학생과 EMA 예측의 차이, 전역 오차 $\mathbf e_{\mathrm{global}}=f_{\theta_{\mathrm{ema}}}(\tilde a_s,s)-a_0^\star$는 EMA 예측과 $a_0^\star$의 차이다.

$$
\begin{aligned}
\mathbf e_{\mathrm{total}}&=\mathbf e_{\mathrm{local}}+\mathbf e_{\mathrm{global}},\\
\mathbb E\|\mathbf e_{\mathrm{total}}\|_2^2
&=\mathbb E\|\mathbf e_{\mathrm{local}}\|_2^2+\mathbb E\|\mathbf e_{\mathrm{global}}\|_2^2
+2\mathbb E\langle\mathbf e_{\mathrm{local}},\mathbf e_{\mathrm{global}}\rangle.
\end{aligned}
$$

Cauchy–Schwarz 부등식은 교차항의 절댓값을 두 평균제곱오차의 제곱근 곱으로 제한한다. 특히 국소 오차의 평균제곱이 0이면 총 오차의 평균제곱은 전역 오차의 평균제곱과 같다. EMA 타깃과 일치한다는 것만으로는 그 타깃이 정확하다고 할 수 없다는 것이 저자의 결론이다.

다음은 부록 C.2의 정확한 항등식이다. 두 끝점에서 공통 $a_t$가 소거되므로 속도 잔차에 $-t$가 곱해진다. $t>0$에서 제곱하고 $t^2$로 나누면, 속도 손실은 **$1/t^2$로 가중한 끝점 손실**이 된다.

$$
\begin{aligned}
f_{\theta_{\mathrm S}}(a_t,t)-a_0^{\mathrm T}
&=-t\left[v_{\theta_{\mathrm S}}(a_t,t)-u_{\theta_{\mathrm T}}(a_t,t)\right],\\
\mathcal L_{\mathrm{TA}}
&=\mathbb E\!\left[\frac{\|f_{\theta_{\mathrm S}}(a_t,t)-a_0^{\mathrm T}\|_2^2}{t^2}\right].
\end{aligned}
$$

부록 C.3에서는 교사의 수치 적분 오차 $\boldsymbol\delta=a_0^{\mathrm T}-a_0^\star$를 더한다. $\|p+q\|^2\leq2\|p\|^2+2\|q\|^2$는 Cauchy–Schwarz와 $2ab\leq a^2+b^2$에서 나온다. 마지막 줄에는 $0<t\leq1$을 쓰며, norm 안의 속도 인자 $(a_t,t)$는 생략한다. 이 상한은 국소·전역 오차의 합을 직접 제약하며, 두 성분이 따로 줄어들 것을 요구하지 않는다.

$$
\begin{aligned}
\mathbf e_{\mathrm{total}}&=-t\left[v_{\theta_{\mathrm S}}(a_t,t)-u_{\theta_{\mathrm T}}(a_t,t)\right]+\boldsymbol\delta,\\
\mathbb E\|\mathbf e_{\mathrm{total}}\|_2^2
&\leq 2\mathbb E\!\left[t^2\|v_{\theta_{\mathrm S}}-u_{\theta_{\mathrm T}}\|_2^2\right]+2\mathbb E\|\boldsymbol\delta\|_2^2\\
&\leq 2\mathcal L_{\mathrm{TA}}+2\mathbb E\|\boldsymbol\delta\|_2^2.
\end{aligned}
$$

$\boldsymbol\delta$의 크기는 다음과 같이 제한된다. 교사 속도가 상태에 대해 $L$-Lipschitz이고 정확한 궤적의 2차 시간 미분 노름이 $M$ 이하일 때, 균일 Euler 간격 $h=t/K$의 한 단계 오차는 $Mh^2/2$ 이하다. $D_k$는 $k$번째 수치 상태와 정확한 상태의 거리다. 아래 등비급수 전개는 **리뷰 보충 중간 계산**이며 부록 C.3과 같은 순서다.

$$
\begin{aligned}
D_{k+1}&\leq(1+Lh)D_k+\frac{Mh^2}{2},\qquad D_0=0,\quad h=\frac{t}{K},\\
\|\boldsymbol\delta\|_2=D_K
&\leq\frac{Mh^2}{2}\sum_{j=0}^{K-1}(1+Lh)^j\\
&=\frac{Mh}{2L}\left[(1+Lh)^K-1\right]
\leq\frac{Me^{Lt}t^2}{2K}\quad(L>0).
\end{aligned}
$$

마지막 부등식은 $(1+Lh)^K\leq e^{Lt}$와 $e^{Lt}-1\leq Lt e^{Lt}$를 사용한다. $L=0$이면 직접 합해 $D_K\leq Mt^2/(2K)$다. 균일한 $L,M$ 아래 위 상한의 수치 오차 항은 $O(K^{-2})$이며, 저자는 교사 단계 수 $K$를 늘려 이 항을 명시적으로 줄일 수 있다고 정리한다. [Appendix C.1–C.3, Eqs. (C.1)–(C.4)](https://arxiv.org/html/2610.06617v1#A3)

### 4.2 전문가 간 웨이브프런트 파이프라이닝 (Cross-Expert Wavefront Pipelining)

전문가 간 웨이브프런트 파이프라이닝(Cross-Expert Wavefront Pipelining, CEWP)은 한 GPU의 두 비차단 CUDA 실행 스트림(CUDA stream)에서 영상과 행동 전문가의 계산을 겹친다. 저자는 각 확산 트랜스포머(DiT) 블록을 Q/K/V 투영(projection), 어텐션, 나머지 계산으로 나누고, 행동 블록에서 영상 KV를 직접 읽는 연산은 행동 어텐션뿐임을 보인다. 그래서 영상 블록이 투영과 필요한 KV 전처리를 마치면 준비 이벤트(readiness event)를 기록하고, 행동 쪽 투영은 자기 은닉 상태(hidden state)가 준비되는 대로 진행하다가 영상 KV를 읽는 어텐션 직전에만 이벤트를 기다린다. 영상 쪽은 남은 계산을 계속한다. [§4.2, Figure 4](https://arxiv.org/html/2610.06617v1#S4.SS2)

<figure class="review-figure" id="figure-realtimewam-pipeline">
<a href="/assets/reviews/realtimewam/paper-figure-4.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-4.svg" width="1137" height="581" alt="위의 순차 video/action 실행과 아래 CEWP의 두 CUDA stream, KV ready 이벤트, action attention 앞 wait, 오른쪽의 block 내부 의존성을 비교한 원문 도식." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 4 · 전체 cache 대기에서 블록 이벤트로</span>위는 영상 전문가 전체가 끝난 뒤 행동 전문가가 시작한다. 아래는 각 영상 블록의 KV가 준비되면 이벤트를 기록하고 행동 어텐션이 필요한 시점에 기다린다. 오른쪽은 한 블록 안의 투영→어텐션→나머지 계산 의존성이다. 시간축은 실행 순서를 설명한다. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-4.svg" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2610.06617v1#S4.F4">Chengtao Lv et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원본 유지, 수정·자르기 없음</span></figcaption>
</figure>

CEWP는 같은 파라미터·입력·마스크·연산자를 쓰고 실행 순서만 바꾼다. 부록 D는 블록에 같은 상태가 들어가면 같은 Q/K/V와 같은 완성된 영상 KV를 쓰므로 다음 블록 상태도 같다는 귀납으로, 정확한 산술에서 순차 추론과 출력이 같음을 보인다. 저자는 이를 원래 계산 그래프의 또 다른 위상 정렬(topological ordering)이라 부르며, 추가 학습이 필요 없고 TACD와도 독립이라고 정리한다. [Appendix D](https://arxiv.org/html/2610.06617v1#A4)

부록 B의 Algorithm 1은 한 번의 순전파 스케줄을 의사코드로 준다. 영상 KV를 받는 조건화(conditioning) 블록 집합 $\mathcal C$는 Fast-WAM에서 모든 블록 $\{1,\ldots,N\}$, 저자의 Faster-WAM 구현에서 $\{1,5,\ldots,N\}$이다. 각 조건화 블록에서 `Fuse`가 단일 블록 KV(Fast) 또는 누적된 구간 KV의 가중 결합(Faster)을 만든 뒤 이벤트를 기록하고, 그 사이의 행동 블록은 전문가 간 동기화 없이 진행한다. [Appendix B, Algorithm 1](https://arxiv.org/html/2610.06617v1#algorithm1)

부록 E는 두 기법이 줄이는 지연을 단순 모형으로 정리한다. 영상·행동 전문가 한 번의 지연을 $T_v,T_a$, 행동 잡음 제거 단계 수를 $S$라 하면 순차 실행은 $T_v+ST_a$이고, TACD는 $S=1$로 만든다. CEWP의 지연은 겹쳐서 아낀 시간 $T_{\mathrm{overlap}}$과 동기화·스케줄링·자원 경합 비용 $\Delta$로 쓰며, 이상적인 겹침($T_{\mathrm{overlap}}\approx\min(T_v,T_a)$)에서는 $\max(T_v,T_a)+\Delta$가 된다. 저자는 블록 의존성과 GPU 자원 공유 때문에 실제 겹침은 이보다 작다고 덧붙인다. [Appendix E](https://arxiv.org/html/2610.06617v1#A5)

$$
T_{\mathrm{seq}}(S)=T_v+ST_a,\qquad
T_{\mathrm{pipe}}(1)=T_v+T_a-T_{\mathrm{overlap}}+\Delta\approx\max(T_v,T_a)+\Delta.
$$

여기서 **Faster-WAM은 arXiv:2608.04404**의 future-conditioning 모델이다. 같은 이름의 2608.02365는 Depth-of-Thought를 사용하는 다른 연구다. `*`는 Fast-WAM 기반, `†`는 이 Faster-WAM 기반 RealtimeWAM을 뜻한다. [RealtimeWAM §3, Table 1](https://arxiv.org/html/2610.06617v1#S3), [Faster-WAM Method](https://arxiv.org/html/2608.04404v1#Sx3)

## 5. 실험 (Experiments)

성공률과 속도를 별도 축으로 평가한다.

### 5.1 구현·5.2 평가 벤치마크 (Implementation Details; Evaluation Benchmarks)

평가는 RoboTwin 2.0의 clean/random 조건(50개 과제를 clean 2,500개·randomized 25,000개 시연으로 다중 과제 학습), LIBERO의 네 suite, LIBERO-Plus의 일곱 교란 조건에서 한다. 성공률(success rate)은 과제 성공으로 판정된 rollout의 비율이며 높을수록 좋다. 계산 비용은 함수 평가 횟수(number of function evaluations, NFE)와 지연으로 따로 본다. 학습은 영상 전문가를 고정하고 행동 전문가의 rank-128 LoRA만 30,000 iteration 조정하며, TACD의 교사는 10단계를 쓴다. 저자는 공정한 비교를 위해 DMD·CD·MeanFlow 증류 기준선에도 같은 학습 설정을 적용했다고 밝힌다. [§5.1–5.2](https://arxiv.org/html/2610.06617v1#S5.SS1)

### 5.3 주요 결과 (Main Results)

**원문 Table 1의 16행 전부(기반 모델별로 순서를 묶음).** 성공률 단위는 %, 높을수록 좋다. NFE는 영상과 행동의 잡음 제거 함수 평가 수다. RoboTwin overall은 clean/random 평균, LIBERO는 네 suite 평균이다. 부록 F.1에 따르면 "(인용)" 8행은 앞선 논문(Light-WAM·Flash-WAM 등)의 보고값을 그대로 옮긴 것이고, 10단계 Fast/Faster-WAM, 세 증류 기준선, RealtimeWAM은 저자가 평가했다. 교사를 1단계로만 돌린 naive one-step 행은 한 단계 생성에 별도 학습이 필요한 이유를 보여 준다. [Table 1, Appendix F.1](https://arxiv.org/html/2610.06617v1#S5.T1)

| Model | Video NFE | Action NFE | RoboTwin Clean ↑ | Random ↑ | Overall ↑ | LIBERO ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| $\pi_0$ (인용) | — | — | 65.92 | 58.40 | 62.16 | 94.1 |
| $\pi_{0.5}$ (인용) | — | — | 82.74 | 76.76 | 79.75 | 96.9 |
| X-VLA (인용) | — | — | 72.90 | 72.80 | 72.85 | 98.1 |
| Motus (인용) | 10 | 10 | 88.66 | 87.02 | 87.84 | 97.7 |
| LingBot-VA (인용) | 25 | 50 | 92.90 | 91.50 | 92.20 | 98.5 |
| Flash-WAM (인용) | 1 | 2 | 88.42 | 82.66 | 85.54 | 96.1 |
| Flash-WAM (인용) | 1 | 1 | 82.56 | 80.26 | 81.41 | 95.1 |
| Light-WAM (인용) | 1 | 1 | 76.40 | 76.30 | 76.40 | 97.2 |
| Fast-WAM | 1 | 10 | 91.82 | 91.19 | 91.51 | 97.0 |
| Fast-WAM, naive one-step | 1 | 1 | 70.42 | 68.12 | 69.27 | 95.0 |
| CD* | 1 | 1 | 89.88 | 89.38 | 89.63 | 96.9 |
| MeanFlow* | 1 | 1 | 85.74 | 85.72 | 85.73 | 96.9 |
| DMD* | 1 | 1 | 88.64 | 87.80 | 88.22 | 96.1 |
| RealtimeWAM* | 1 | 1 | 91.96 | 89.72 | 90.84 | 97.0 |
| Faster-WAM | 1 | 10 | 93.20 | 92.66 | 92.93 | 98.9 |
| RealtimeWAM† | 1 | 1 | 92.98 | 92.30 | 92.64 | 99.0 |

Fast 대비 RealtimeWAM*의 RoboTwin 평균 손실은 **0.67%포인트**(percentage point, pp), Faster 대비 †는 **0.29pp**다. LIBERO는 각각 같은 점수와 0.1pp 증가를 보고한다. 저자는 이를 한 단계 생성으로 거의 손실 없는 성능을 얻은 결과로 보고, one-step Flash-WAM(81.41%)과 DMD*(88.22%)를 크게 앞선다고 정리한다. [§5.3, Table 1](https://arxiv.org/html/2610.06617v1#S5.SS3)

**원문 Table 2에서 Faster와 RealtimeWAM†의 모든 교란 열.** LIBERO-Plus는 카메라·로봇·언어·조명 등 학습 밖 조건을 평가한다. 마지막 변화 열은 리뷰에서 뺄셈으로 계산한 pp다. overall은 일곱 값의 산술평균이 아니라 교란별 trial 수로 가중한 평균이다. [Table 2, Appendix F.1](https://arxiv.org/html/2610.06617v1#S5.T2)

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

overall은 0.6pp 낮다. 저자는 camera·noise·layout의 소폭 향상을 들어 단계 증류가 분포 밖(out-of-distribution, OOD) 강건성을 대체로 보존한다고 해석한다. 하락 폭은 robot 2.6pp, language 2.2pp가 가장 크다. [원문 Table 2](https://arxiv.org/html/2610.06617v1#S5.T2)

### 5.4 구성요소 제거와 속도 분석 (Ablation Study)

Table 3은 손실 항과 미세조정 대상을 바꾼 RoboTwin 절제다. 영상·행동을 함께 조정한 CD는 89.63%, 영상을 고정한 행동 전용 CD는 89.85%, 여기에 교사 앵커 손실을 더한 TACD는 90.84%로 0.99pp 더 높다. 다만 random 열은 89.94%→89.72%로 낮아진다. 저자는 영상 전문가를 함께 조정하면 성공률이 낮아지고 학습 비용만 늘며, TACD의 향상은 교사 앵커가 국소 목적이 남긴 전역 편차를 줄인다는 분석과 맞는다고 해석한다. [Table 3](https://arxiv.org/html/2610.06617v1#S5.T4)

Table 3 첫 행의 값(89.88/89.38/89.63%)은 Table 1의 CD*와 같다. **리뷰 해석.** 따라서 Table 1의 CD*는 영상까지 조정한 설정으로 보이며, §5.1의 "같은 학습 설정"은 미세조정 대상까지 같다는 뜻은 아닌 것으로 읽힌다. Table 1에서 CD* 대비 1.21pp 차이에는 영상 고정의 몫 0.22pp가 들어 있다.

<figure class="review-figure" id="figure-realtimewam-speed">
<a href="/assets/reviews/realtimewam/paper-figure-5.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/realtimewam/paper-figure-5.svg" width="1152" height="263" alt="H100에서 Fast-WAM과 Faster-WAM에 TACD, CUDA Graph, CEWP, 효율 커널을 차례로 적용한 지연 막대와 누적 가속 비율." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 5 · 가속을 누적해서 읽기</span>왼쪽 Fast와 오른쪽 Faster를 나눠, 위에서 아래로 하나씩 추가되는 최적화를 읽는다. 가로축은 지연(ms), 낮을수록 빠르다. 파란 speedup 표기는 시작 baseline 대비 누적 값이며, 연결선의 값은 직전 단계 대비 추가 가속이다. 마지막 약 25배에는 CUDA Graph와 효율 커널도 들어 있다. <span class="figure-links"><a href="/assets/reviews/realtimewam/paper-figure-5.svg" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2610.06617v1#S5.F5">Chengtao Lv et al., 2026, Figure 5</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원본 유지, 수정·자르기 없음</span></figcaption>
</figure>

H100에서 Fast-WAM의 299.7ms는 TACD만 적용하면 65.8ms, CUDA Graph를 추가하면 23.3ms, CEWP까지 적용하면 17.4ms, 효율 커널까지 적용하면 12.2ms가 된다. Faster 경로는 218.9→57.1→26.2→22.4→16.1ms다. TACD만의 가속은 4.56배와 3.83배, CUDA Graph 뒤 CEWP의 추가 가속은 1.34배와 1.17배이고, 전체 누적 가속은 24.55배와 13.56배다. 효율 커널은 LightX2V 기반의 연산자 수준 최적화 다섯 가지(커널 융합, 모두 참인 어텐션 마스크 제거, 투영 GEMM 묶음)다(부록 F.2). [§5.4, Figure 5](https://arxiv.org/html/2610.06617v1#S5.F5)

**원문 Figure 5(H100)와 Table F.3(추가 GPU)의 baseline→최종 지연(ms).** 영상 변분 오토인코더(variational autoencoder, VAE) 인코딩을 포함하고 에피소드마다 한 번 하는 text 인코딩은 제외한다. 센서 취득·통신·모터 구동은 포함하지 않는다. [Figure 5](https://arxiv.org/html/2610.06617v1#S5.F5), [Appendix F.5](https://arxiv.org/html/2610.06617v1#A6.SS5)

| Backbone | H100 | RTX 5090 | RTX 4090D |
| --- | --- | --- | --- |
| Fast → RealtimeWAM* | 299.7→12.2 | 251.5→17.8 | 480.8→27.2 |
| Faster → RealtimeWAM† | 218.9→16.1 | 214.1→30.9 | 362.4→44.7 |

소비자 GPU에서는 누적 가속이 Fast 기반 14.1·17.7배, Faster 기반 6.9·8.1배(RTX 5090·4090D)로 작아지며, 저자는 현재 커널이 H 계열 GPU에 맞춰져 있기 때문일 수 있다고 설명한다. 저자는 호출당 지연이 30Hz 제어의 실행 시간보다 훨씬 짧아 실시간 행동 생성이 가능하다고 말한다(§1). [Appendix F.5](https://arxiv.org/html/2610.06617v1#A6.SS5) **리뷰 해석.** H100의 12.2ms는 30Hz 한 주기(약 33.3ms) 안에 들어가지만, †의 RTX 4090D 44.7ms는 그 주기를 넘는다.

## 6. 결론 (Conclusion)

저자는 TACD로 행동 잡음 제거를 한 단계로 압축하고 CEWP로 블록별 의존성을 보존하며 실행을 겹치는 두 기여를 연결한다. 결론은 RealtimeWAM이 Fast-WAM·Faster-WAM에서 거의 손실 없는 성능으로 실시간 행동 생성(예: 12.2ms)을 가능하게 하고, H100에서 약 14–25배의 종단 가속을 낸다는 것이다. [원문 §6](https://arxiv.org/html/2610.06617v1#S6)

## 부록 A·F.6: 한계와 실물 시연 (Limitations; Real-World Evaluation of Long-Horizon Garment Folding)

부록 A가 밝히는 첫 한계는 적용 범위다. RealtimeWAM은 MoT 기반 WAM을 대상으로 하며 DreamZero 같은 공유 백본 구조에는 바로 적용할 수 없다. 저자는 공유 백본의 단계 증류가 영상 생성의 단계 증류에 가까워 행동 고유의 특성을 분리하기 어렵다고 설명한다. 이 범위 안에서는 추론 때 미래 영상 예측을 생략하는 Fast-WAM과 유지하는 Faster-WAM을 모두 지원한다. [Appendix A](https://arxiv.org/html/2610.06617v1#A1)

두 번째 한계는 학습 비용이다. **원문 Table A.1**은 16 H100, 30,000 iteration 조건의 학습 시간과 최대 메모리다. [Table A.1](https://arxiv.org/html/2610.06617v1#A1.T1)

| Training setup | Time | Peak memory |
| --- | --- | --- |
| CD, video + action tuned | 8 h 1 m | 65.98 GiB |
| CD, action only | 5 h 21 m | 24.73 GiB |
| TACD, action only | 8 h 3 m | 24.73 GiB |

행동 전용 CD와 TACD의 최대 메모리는 같지만, 학습 시간은 5시간 21분에서 8시간 3분으로 약 50% 늘어난다. 저자는 이 증가가 영상·행동 공동 미세조정(8시간 1분)과 비슷한 수준이고, 영상 고정으로 공동 미세조정보다 메모리를 62.5% 줄인다며 감당할 만한 비용으로 본다. 배포 비용을 줄이는 대신 교사 rollout 계산을 학습에 지불하는 구조다.

부록 F.6은 AgileX PiPER 양팔과 RTX 4090에서 RealtimeWAM*로 티셔츠를 접는 실물 rollout을 질적 평가로 제시한다. 저자는 RTC 같은 실행 수준 비동기 없이도 연속 행동이 눈에 띄는 멈춤 없이 이어졌다고 보고하며, 성공률이나 시도 횟수는 보고하지 않는다. [Appendix F.6, Figure F.1](https://arxiv.org/html/2610.06617v1#A6.SS6)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 이 논문의 강점은 반복 횟수와 실행 의존성(execution dependency)을 서로 다른 병목으로 나누고, 각 기법의 몫을 Figure 5의 단계별 지연으로 따로 보인 데 있다. 행동 전문가가 지연의 약 90%를 차지한다는 Figure 1의 분석과 맞게 TACD가 가장 큰 가속을 낸다. 부록 E의 모형에서 CEWP가 아낄 수 있는 시간은 $\min(T_v,T_a)$ 이하라 가속은 $\Delta$를 무시해도 2배를 넘을 수 없으므로, 1.34배와 1.17배는 이 범위 안의 결과다. Fast 경로의 24.55배 가운데 두 기법을 곱한 몫은 약 6.1배이고, 나머지 약 4배는 TACD 없이도 적용되는 CUDA Graph(Table F.3에서 단독 2.6–4.5배)와 효율 커널에서 나온다.

품질 근거는 반복 실행의 분산 없이 보고된 평균 성공률이다. 교사 앵커의 효과를 직접 보여 주는 Table 3의 증가는 0.99pp이고 random 조건은 오히려 낮아지므로, 1pp 안팎의 차이가 실행 간 변동보다 큰지는 이 표만으로 판단하기 어렵다. 부록 C의 상한은 정확한 교사 ODE 끝점까지의 오차를 다루므로 교사 자체가 낯선 상태에서 내는 행동의 오류는 그 밖에 있고, $K$를 10에서 20으로 늘려 상한을 조여도 성공률은 오르지 않았다(Table 4). 서론의 "MoT 기반 WAM이면 어디에나"라는 표현은 Fast-WAM·Faster-WAM 두 사례에 근거한다.

남은 질문은 다음과 같다. 순차 실행과 CEWP의 실제 GPU 출력 차이와 p95/p99 지연은 얼마인가? 같은 평균 연산 예산에서 고정 1단계와 AnyStep-WAM 같은 관측별 적응 예산은 어떻게 비교되는가? 학생이 실제로 방문하는 상태에서의 오류(WAM-OPD의 질문)는 교사 끝점 감독으로 얼마나 줄어드는가? 실행 수준 비동기인 RTC와 결합하면 관측 최신성(observation freshness)이 개선되는가?

## 출처와 읽은 범위

- 원문: *RealtimeWAM: One-Step Asynchronous World Action Models* · Chengtao Lv et al. · 최초 공개 2026-10-05 · [검토 버전](https://arxiv.org/abs/2610.06617v1) · 검토일 2026-10-23.
- 읽은 범위: v1 본문 §1–6, 부록 A–F.
- 그림 출처: Figures 2·4·5는 저자 v1 원문 자산으로 CC BY 4.0, 원본 바이트를 유지한다. 대표 개념도는 리뷰 자체 제작물이다. [원문](https://arxiv.org/abs/2610.06617v1).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·GPU 실측·보충 동영상.
