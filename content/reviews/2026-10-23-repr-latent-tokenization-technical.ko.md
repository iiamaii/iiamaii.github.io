---
title: "[2/2] RePR 심화: PMI 유도·구현·스케일링 수치를 따라가기"
description: "부록 A.1–A.7의 알고리즘·PMI 조건·원시 실험 수치를 보존하고, 확률 유도와 외삽의 범위를 검토한다."
date: "2026-10-23"
publishedAt: "2026-10-23T15:12:15+09:00"
updatedAt: "2026-10-23T15:12:15+09:00"
topics: ["representation learning", "latent tokenization", "diffusion models", "generative modeling", "scaling laws"]
visibility: "public"
lang: "ko"
translationKey: "repr-latent-tokenization-technical"
paperTitle: "Generative Representation Learning for Scalable Latent Tokenization"
paperPublishedDate: "2026-10-14"
authors: ["Alan Baade", "Kyle Sargent", "Vincent Sitzmann", "Eric Ryan Chan", "Justin Johnson", "Li Fei-Fei", "Gordon Wetzstein"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.20533v1"
thumbnail: "/assets/reviews/repr/figure1.webp"
thumbnailAlt: "원문 Figure 1: 조건부 예측의 정보 차이를 패치 표현의 내적 변화에 맞추는 RePR 구조."
---

RePR(Relative Prediction Representations)의 핵심은 생성 모델이 계산한 **상대적인 예측 이득**을 표현의 내적 차이로 옮기는 것이다. 이 글은 그 설명이 실제 알고리즘·가정·측정에 어떻게 이어지는지 확인한다. [\[1/2\] 본문 해설](/reviews/repr-latent-tokenization/)을 먼저 읽고, 여기서는 원문 **A.1→A.2→A.3→A.4→A.5→A.6→A.7** 순서로 따라가자.

이론을 읽을 때는 세 층을 나눠야 한다. 확률의 연쇄법칙은 정확한 항등식이다. 이상적인 점별 상호정보량(pointwise mutual information, PMI) 인수분해의 성질에는 표현 용량·정보 보존 조건이 붙는다. 실제 RePR은 학습한 예측 모델, 시간 가중치, 마스크와 유한 Monte Carlo 표본을 사용한다. 원문에는 번호가 붙은 Theorem/Lemma와 긴 형식 증명은 없으며, A.4의 재구성 논증과 본문 확률 유도를 중심으로 아래의 **리뷰 보충 유도**를 제공한다.

## A.1 알고리즘 (Algorithm)

### 무엇이 타깃이고 무엇이 학습되는가?

Algorithm 1은 이미지 $X$에서 문맥 $A,B$, 합친 문맥 $AB$, 두 문맥에 들어 있지 않은 타깃 패치 $p$를 뽑는다. 표현 경로에 필요한 입력은 $A,B,Bp$이고, 예측 경로에 필요한 입력은 $B,AB$다.

| 경로 | 계산 | 역할 |
| --- | --- | --- |
| MAR 학습 | $E_\theta,D_\theta,M_\theta$ | 마스크로 선택한 예측 타깃을 복원하는 생성 손실 |
| 정보 차이 추정 | 같은 $p_t,t$를 사용해 $p$를 두 문맥에서 복원 | 다른 문맥 $A$가 추가될 때의 가중 예측 이득 |
| 표현 학습 | $\langle R_\theta(A),R_\theta(Bp)-R_\theta(B)\rangle$ | 정보 차이와 맞아야 할 표현의 변화 |

다음은 원문 Algorithm 1의 핵심 순서를 보여 주는 의사코드다. 외부 코드를 실행한 결과나 바로 실행 가능한 라이브러리 API가 아니다. 원문 의사코드의 $R$은 공유 인코더 출력에 적용되는 branch이며, 본문의 $R_\theta(\mathbf{x})$는 이를 포함한 전체 표현 경로를 뜻한다.

```python
a, b, ab, bp = E(A), E(B), E(AB), E(B, p)
d_ab, d_b = D(ab), D(b)
# MAR: selected targets x, independently sampled t and eps
mar_loss = mean(w(t) * (M(x_t, t, d_ab[m_AB]) - x)**2)
with no_grad():
    # N noise/time samples; the two predictions share them
    p_ab = M(p_t, t, d_ab[p_i])
    p_b = M(p_t, t, d_b[p_i])
    pmi = mean(g(t) * ((p_b - p)**2 - (p_ab - p)**2))
pred = dot(R(a), R(bp) - R(b))
repr_loss = mean((pred - pmi)**2)
loss = mar_loss + loss_weight * repr_loss
```

원문처럼 PMI 타깃 계산은 `no_grad()` 안에 있다. 따라서 표현 손실이 예측 숫자를 바꾸는 방향으로 직접 역전파하지 않는다. 반면 표현·생성 경로가 $E_\theta$를 공유하므로 표현 손실이 **모든 공유 매개변수와 무관한 것**도 아니다. MAR 손실은 예측 모델을 계속 개선한다. 표현 손실은 네 번 중 한 번, 나머지는 MAR만 학습한다.

또한 본문 식 (16)과 의사코드는 $M$의 인자를 제시하는 순서가 다르다. 여기서는 패치의 noisy 값, 시간, 문맥이라는 의미를 추적한다. 실제 구현을 옮길 때는 공개 코드의 함수 signature를 확인해야 하며 이번 리뷰에서는 그 코드를 실행하지 않았다. `mean`의 축과 벡터 norm의 합·평균 정규화도 손실의 숫자 척도를 바꿀 수 있다. [A.1·Algorithm 1, p.16](https://arxiv.org/pdf/2610.20533v1#page=16)

## A.2 추가 실험 (Additional Experiments)

### PMI 적분 표본 수와 공유 노이즈

[![Figure 5. 4096-step PMI 추정값을 기준으로 한 Monte Carlo 표본 수별 오차와 shared/independent noise 비교.](/assets/reviews/repr/figure5.webp)](/assets/reviews/repr/figure5.webp)

*원문 Figure 5, p.16. 가로축은 Monte Carlo 적분 표본 수 $N$, 세로축은 4096표본 추정값과의 평균 제곱 오차(mean squared error, MSE)다. 보라 실선은 공유 노이즈, 주황 점선은 독립 노이즈다. 기준은 참 PMI의 정확한 계산이 아니라 더 많은 표본을 사용한 근사다. 본문 그림은 눌러 확대할 수 있다.*

두 예측은 같은 패치와 같은 noisy realization을 놓고 비교하는 것이 유리하다. 리뷰 보충으로 두 오차 추정치를 $U,V$라 쓰면

$$
\operatorname{Var}(U-V)
=\operatorname{Var}(U)+\operatorname{Var}(V)
-2\operatorname{Cov}(U,V).
$$

공유한 난수가 두 추정치를 양의 방향으로 함께 흔들면 차이의 분산이 줄어든다. 공유가 언제나 유리하다는 일반 정리는 아니며, Figure 5가 이 구성에서의 효과를 보여 준다. 저자들은 $N=32$를 선택하고 diffusion integration이 전체 학습 FLOPs의 1.3%라고 보고한다. 이는 작은 head를 반복한 **적분 부분**의 비용이다. 여러 encoder/decoder 경로를 포함한 모든 표현 학습 오버헤드가 1.3%라는 뜻으로 확대하지 않는다.

### 재구성과 생성의 지표는 다르다

원문 Table 4는 RAE 디코더를 사용한 재구성 비교다.

| 인코더 | 재구성 FID(rFID) ↓ | PSNR ↑ |
| --- | --- | --- |
| DINOv2-B | 0.49 | 18.09 dB |
| RePR-B 반복 3, 600ep 표기 | 0.47 | 21.72 dB |

최대 신호 대 잡음비(peak signal-to-noise ratio, PSNR)는 픽셀 오차를 본다. 아래는 지표를 설명하기 위한 표준 정의이며 원문의 새 정리가 아니다. 픽셀 범위의 최대 기준값을 $L$, 비교할 원소 수를 $n$이라 두면

$$
\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2,
\qquad
\operatorname{PSNR}=10\log_{10}\frac{L^2}{\operatorname{MSE}}.
$$

PSNR은 같은 범위·전처리·집계 방식에서 높을수록 좋으며, 완전 일치에서는 무한대다. 지각적으로 자연스러운 생성과는 다른 질문이다. rFID는 재구성된 이미지 집합과 실제 집합의 특징 분포 차이를 본다. 따라서 rFID가 낮아도 입력별 세부가 모두 정확하다는 뜻은 아니며, 이 표의 0.47을 생성 FID-50K 2.18과 비교해서도 안 된다. Table 4의 인코더 이름·600ep와 A.3의 상세 RAE 비교 비용 기록은 별도로 제시되어 있어 같은 체크포인트 예산으로 조용히 합치지 않는다. [A.2·Table 4, pp.16–17](https://arxiv.org/pdf/2610.20533v1#page=17)

## A.3 구현 상세 (Implementation Details)

### 구조와 실제 예산

추론 시에는 $E_\theta$ 18층과 표현 branch 6층을 쓴다. 뒤의 6층은 추가 self-attention 3층과 cross-attention 3층이며, 학습용 $D_\theta$의 6층은 별도다. 폭 1024에서 표준 ViT-L에 대응한다. 크기는 20M/44M/78M/173M/306M이며 폭은 256/384/512/768/1024다. 출력 차원은 1024로 고정한다.

| 항목 | 원문 Tables 5–6의 핵심 설정 |
| --- | --- |
| 이미지·패치 | $256\times256$, $16\times16$ 패치 |
| 인코더 학습 | batch 2048, AdamW, learning rate $2.4\times10^{-4}$, 폭별 $\mu$P multiplier |
| 증강 | crop scale $[0.8,1.0]$, horizontal flip |
| MAR 시간 분포 | logit-normal $\mu=1.2,\sigma=1.2$, 10% uniform mixture |
| RePR 시간 분포 | 픽셀 $\mu=1.2$, 특징 $\mu=1.6$, 공통 $\sigma=1.2$ |
| 가중치·재사용 | $g(t)=(1-t)/\max(0.33,t)^2$, loss weight 0.005, $A_{\mathrm{Repeat}}=8$, $p_{\mathrm{Repeat}}=16$, $N=32$ |
| 생성 scaling 평가 | ImageNet JiT-B, GPIC MiniT2I; ImageNet 환산 80 epochs |
| GPIC 조건 | FLAN-T5-large 텍스트 인코더, 최대 128 text tokens |

“epoch”는 모든 곳에서 데이터셋을 한 번 돈다는 뜻이 아니다. 저자들은 처리한 image views를 ImageNet 표본 수로 나누어 보고한다. GPIC 1 epoch은 대략 ImageNet 환산 80 epochs다. 체크포인트를 저장한 후 10 epochs의 learning-rate decay도 추가한다. 그래서 반복 2의 $210+210$은 픽셀 학습 200+10, 특징 학습 200+10의 합이다.

Figure 4는 반복 2의 총 420 epochs 모델을 쓴다. RePR-44M/78M 학습은 JiT-L 200ep와 비교한 전체 예산의 약 20%/35%다. Table 3의 RePR-78M은 반복 3 총 830 epochs이며 210+210+410이다. 저자는 JiT-L 600ep에 대응하는 전체 예산의 약 25%를 인코더에 썼다고 설명한다. RAE 비교에는 RePR-B 총 1030 epochs와 $2.19\times10^{20}$ FLOPs를 별도로 적고 I-JEPA-H의 $2.45\times10^{20}$와 비교한다.

계산은 self-attention을 제외한 $6ND$ 근사다. 보고된 FLOPs는 wall-clock 측정과 다르다. 저자는 ImageNet 생성 평가 하나에 약 31 GPU-hours, GPIC에는 56 GPU-hours를 적고 Figure 3 재현에 8-GPU B200 노드 250시간을 **추정**한다. 이번 리뷰에서는 그 시간을 실행·측정하지 않았다. [A.3·Tables 5–6, pp.17–19](https://arxiv.org/pdf/2610.20533v1#page=17)

### 왜 비대칭 역할 정보가 필요한가?

원문 A.3은 가중 PMI가 비대칭이며 PMI 자체도 PSD가 아니라고 명시한다. 동일 벡터를 양쪽에 넣는 Gram 행렬은 항상 PSD다. 리뷰 보충으로 임의 계수 벡터 $c$에 대해

$$
c^\top Kc
=\left\|\sum_i c_iR(\mathbf{x}_i)\right\|_2^2\ge0,
\qquad K_{ij}=\langle R(\mathbf{x}_i),R(\mathbf{x}_j)\rangle.
$$

그러나 주변 분포가 균등하고 공동확률이 대각 0.45·비대각 0.05인 두 상태 예제에서는 PMI 행렬이

$$
\begin{pmatrix}\log1.8&\log0.2\\\log0.2&\log1.8\end{pmatrix}
$$

가 되고, 고유값 하나는 $\log(0.36)<0$이다. 그러므로 무조건 같은 표현끼리의 내적으로 정확히 맞추는 것은 불가능하다. 이 예제는 리뷰의 설명용 계산이다.

역할에 따라 다른 인코딩이 가능하면 특이값 분해(singular value decomposition, SVD)를 쓸 수 있다. 타깃 행렬을 $M=U\Sigma V^\top$라 놓고

$$
F=U\Sigma^{1/2},\qquad G=V\Sigma^{1/2},
\qquad FG^\top=M
$$

로 인수분해하면 된다. 차원이 rank 이상이고 모든 필요한 타깃을 정확히 학습할 수 있다는 조건이 붙는다. 유한 차원·가중 표본 회귀의 모든 해가 자동으로 이런 정확한 해가 되는 것은 아니다. RePR은 추가 정보 $\mathbf{z}$의 위치에 학습 임베딩을 넣어 입력 역할을 구분한다. 이 설계가 본문의 대칭처럼 보이는 $R_\theta$ 표기를 실제 구현에서 보완한다.

## A.4 PMI의 성질 (Properties of PMI)

이 절의 원문은 재구성에 대한 짧은 논증이다. 그 논증을 따라가기 위해 필요한 본문 §3–4의 확률 유도도 이곳에 함께 보충한다. 원문에 없는 새 Theorem 번호를 붙이지 않으며, 아래 계산을 원문의 모든 실제 손실에 대한 완성된 보장으로 제시하지 않는다.

### 대조 점수가 밀도비가 되는 이유 — 리뷰 보충

[CPC 식 (5)](https://arxiv.org/pdf/1807.03748v2#page=4)는 후보 중 하나가 $p(\mathbf{y}\mid\mathbf{x})$, 나머지가 $p(\mathbf{y})$에서 왔다고 가정한다. 어느 후보가 양성인지의 Bayes posterior에서는 공통된 주변확률 곱이 약분되어

$$
p(d=i\mid\mathbf{y}_{1:N},\mathbf{x})
=\frac{p(\mathbf{y}_i\mid\mathbf{x})/p(\mathbf{y}_i)}
{\sum_j p(\mathbf{y}_j\mid\mathbf{x})/p(\mathbf{y}_j)}.
$$

가 된다. categorical cross-entropy의 최적 예측은 이 posterior다. 따라서 log score는 PMI에 $\mathbf{x}$만의 상수를 더한 형태가 된다. negative proposal이 다른 분포이면 비의 분모도 달라진다. 임의의 augmentation·negative 구성을 모두 같은 PMI로 취급할 수 없는 이유다.

### 확산 가중치 $(1-t)/t^3$는 어디서 오는가?

원문이 참조한 [VDM 식 (53–54)](https://arxiv.org/pdf/2107.00630v6#page=18)의 diffusion loss는 신호 대 잡음비(signal-to-noise ratio, SNR)의 미분을 포함한다. RePR의 경로에서는

$$
\operatorname{SNR}(t)=\frac{(1-t)^2}{t^2},
\qquad
-\frac12\frac{\mathrm{d}}{\mathrm{d}t}\operatorname{SNR}(t)
=\frac{1-t}{t^3}.
$$

따라서 이 가중치는 임의의 경험식이 아니라 해당 노이즈 경로의 변분 목적에서 나온다. 다만 $t\to0$에서 특이해진다. 이상적 이산 데이터·최적 denoiser의 극한 논리와 유한 정밀도 학습에서의 안정성은 별개이므로 Table 1(f)의 실패가 중요하다.

끝점 prior·reconstruction 항이 소거되고 denoising bound가 tight하면 true log probability와 연결된다. 이 조건이 충족되지 않은 실제 예측에서는 두 lower bound의 오차가 PMI 차이에도 남는다. 같은 생성 모델 $G$의 정확한 log probability와 그 모델의 VLB 사이의 두 비음수 gap을 $\Delta_{\mathrm{cond}},\Delta_{\mathrm{base}}$라는 **리뷰 보조 기호**로 두면

$$
\log\frac{P_G(\mathbf{y}\mid\mathbf{x})}{P_G(\mathbf{y})}
=\operatorname{VLB}_G(\mathbf{y}\mid\mathbf{x})-\operatorname{VLB}_G(\mathbf{y})
+\Delta_{\mathrm{cond}}-\Delta_{\mathrm{base}}.
$$

오차 차이의 부호는 정해지지 않는다. 두 gap이 모두 0으로 가면 해당 모델의 우도비에 접근한다. 데이터의 PMI까지 맞추려면 모델이 실제 조건부·주변 분포를 맞춰야 한다. 따라서 “두 하한의 차이도 하한”이라는 결론은 나오지 않는다. 실제 $f,g$가 바뀐 목적은 원래 확률량과 같은 것으로 조용히 놓지 않는다.

### bootstrap 연쇄법칙의 중간 계산

본문 식 (13–14)는 다음처럼 펼칠 수 있다.

$$
\begin{aligned}
\operatorname{PMI}(\mathbf{x},y_{\le K})
&=\log\frac{p(y_1,\ldots,y_K\mid\mathbf{x})}{p(y_1,\ldots,y_K)}\\
&=\sum_{k=1}^{K}\log\frac{p(y_k\mid y_{<k},\mathbf{x})}{p(y_k\mid y_{<k})}.
\end{aligned}
$$

두 식에서 마지막 항을 제외한 공통 항을 빼면 한 토큰의 조건부 PMI만 남는다. 이를 모든 $k$에 대해 합하면 중간 prefix의 PMI가 상쇄된다. 빈 문맥의 PMI는 0이다. 필요한 조건은 사용한 조건부 확률의 분모가 정의되는 support이며, 데이터나 모델의 0확률 처리는 따로 필요하다.

그렇다고 유한 표본으로 일부 문맥만 학습한 신경망이 모든 prefix의 식을 정확히 만족하는 것은 아니다. 시간 가중치를 준 패치별 정보 차이를 다시 원래의 무가중 전체 PMI라고 해석할 때도 추가 정합성 조건이 필요하다. 이 구별이 항등식에서 실용적인 회귀 목표로 넘어가는 지점이다.

### PMI로 재구성한다는 주장의 범위

원문은 오염되지 않은 이미지 $\mathbf{x}$와 복원할 픽셀 $\mathbf{y}$를 놓고, 모든 후보 픽셀값에 비영 support를 둔 PMI 인수분해에서 최대 PMI를 찾으면 재구성할 수 있다고 설명한다. 고정된 픽셀 위치에서 **참 픽셀이 이미지로 완전히 결정되고, 유한 후보 집합·양의 주변확률·정확한 인수분해**라면 논리는 명확하다. 참 값의 조건부확률은 1이고 다른 후보는 0이므로 참 후보만 유한한 log ratio를 가지며 나머지는 $-\infty$다.

반면 비영 노이즈를 넣은 상태의 양의 support만으로 모든 경우의 정확한 argmax 복원이 보장되지는 않는다. 리뷰의 설명용 예시로 $p(y=1)=0.0001$, $p(y=1\mid\mathbf{x})=0.01$이면 후보 1의 비는 100이다. 후보 0의 비는 약 0.9901이므로 PMI argmax는 1을 고르지만 조건부 최빈값은 0이다. 이 조건부와 주변확률을 함께 만족하는 양의 공동분포를 구성할 수 있다. **PMI 최대화는 일반적인 조건부 확률 최대화와 다른 연산**이다.

유한하게 분리된 픽셀값과 고정된 양의 prior에서 노이즈를 0으로 보내는 극한은 따로 논증할 수 있다. 그러나 이를 유한 노이즈·마스킹·유한 차원·가중 회귀의 무조건적 재구성 보장으로 옮기지 않는다. 원문도 실제 복원은 마스킹 분포와 학습 수렴에 영향을 받는다고 한정한다.

이것은 특징 예측의 해석에도 중요하다. 가역적인 결정적 $R$이면 확률비에서 대응이 유지되어

$$
\operatorname{PMI}(\mathbf{x},R(\mathbf{y}))
=\operatorname{PMI}(\mathbf{x},\mathbf{y})
$$

이지만 정보를 합쳐 버리는 비가역적인 표현에는 일반적으로 성립하지 않는다. 평균 상호정보량에는 데이터 처리 부등식(data processing inequality)이 적용되며 정보가 줄 수 있다. RePR의 반복 특징 예측은 실험적으로 유익하다는 근거와 이 이상적 설명을 함께 제시한 것이지, 매 반복에서 모든 정보가 보존된다는 완성된 정리가 아니다. [§5.4·A.4, pp.7·17–18](https://arxiv.org/pdf/2610.20533v1#page=17)

## A.5 정성 결과 (Qualitative Results)

[![Figure 6. 계산량 비교에 사용한 JiT-L 조합의 비선별 class-conditional ImageNet 생성 예시.](/assets/reviews/repr/figure6.webp)](/assets/reviews/repr/figure6.webp)

*원문 Figure 6, p.20. 저자는 uncurated 예시로 표기한다. 질감·물체·배경의 모습을 확인할 자료이며, 학습 데이터와의 중복·실패 빈도·다양성 전체를 판정하는 표본은 아니다.*

생성 설정은 latent diffusion에 classifier-free guidance(CFG) 3.0을 구간 $[0.06,1.0]$에 적용하고 AutoGuidance 가중치 1.5를 사용한다. A.5는 AutoGuidance 학습 비용이 전체 학습의 1.6%이며 FLOP 계산에 포함된다고 명시한다. 이는 본문의 unguided FID-10K scaling 곡선과 다른 평가 설정이다.

[![Figure 7. 공간 등변성 제약 유무에 따른 중심 패치와 나머지 패치의 cosine similarity.](/assets/reviews/repr/figure7.webp)](/assets/reviews/repr/figure7.webp)

*원문 Figure 7, p.20. 빨간 중심 패치를 기준으로 다른 위치 특징의 코사인 유사도(cosine similarity)를 시각화한다. 위치 구조가 어떻게 달라지는지 볼 수 있지만, 생성 FID의 개선량이나 모든 이동에 대한 수학적 등변성 보장을 직접 보여 주는 그래프는 아니다.*

## A.6 워터마킹 (Watermarking)

[![Figure 8. clean 이미지와 각 패치에 식별 단서가 반복된 watermarked 이미지.](/assets/reviews/repr/figure8.webp)](/assets/reviews/repr/figure8.webp)

*원문 Figure 8, p.21. 위·아래의 clean/watermarked 예시를 비교한다. 원래 워터마킹 설정의 $32\times32$를 $16\times16$로 줄여 개별 패치도 식별 단서를 얻도록 했다. 이것은 평가를 어렵게 만든 구성된 조건이며 자연 이미지 전체에서 같은 실패 비율이 관측됐다는 뜻은 아니다.*

이 설계에서는 “한 뷰의 ID를 맞추기에 충분한 정보”와 “다른 패치 내용을 더 잘 예측하기 위한 정보”를 분리할 수 있다. RePR은 두 조건부 예측의 차이를 사용하므로 후자의 구조를 학습할 여지가 있다. 다만 워터마크가 생성 목적에도 충분한 shortcut이 되는 새로운 데이터에서는 추가 검증이 필요하다. 본문 Figure 2의 clean 성능과 watermarked 대비 손실도 그대로 남겨야 한다.

## A.7 스케일링 그래프의 원시 수치 (Raw Numbers for Scaling Plots)

### ImageNet: 무엇을 실제로 측정했는가?

Table 7의 핵심 행을 보존하면 다음과 같다. 생성 모델은 JiT-B를 80 epochs 학습하고 FID-10K로 평가했다. epochs 합은 인코더의 이전 반복 학습을 포함한다.

| 인코더 | 픽셀 예측 epochs → FID | 특징 예측 epochs → FID |
| --- | --- | --- |
| 20M | 110→41.61, 210→40.19, 410→37.01, 610→37.21 | 210+110→36.24, 210+210→34.89 |
| 44M | 110→40.93, 210→38.20, 410→36.07, 610→34.82 | 210+110→31.64, 210+210→29.48, 210+410→29.22 |
| 78M | 110→38.26, 210→35.72, 410→34.31, 610→34.00 | 210+110→29.81, 210+210→27.80, 210+410→27.36 |
| 173M | 110→37.22, 210→33.73, 410→31.27, 610→30.68 | 210+210→25.82, 210+410→24.73 |
| 306M | 110→36.14, 211→31.23 | 기재 없음 |

20M의 410→610은 약간 악화한다. 306M의 211은 **원문의 표기 그대로**이며 210으로 임의 수정하지 않았다. 빈 칸을 실패나 0으로 채우지 않는다. 폭만 키우면 모든 지점이 단조 개선되는 표는 아니다.

### GPIC: 다른 특징 공간에서의 측정

Table 8은 FD-DINOv2-10K이며 숫자는 FID와 같은 척도가 아니다. 생성 모델은 MiniT2I를 1 GPIC epoch 학습한다. 아래 epochs는 ImageNet 환산량이다.

| 인코더 | 픽셀 예측 epochs → FD | 특징 예측 epochs → FD |
| --- | --- | --- |
| 44M | 110→514, 210→479, 410→456 | 210+110→416, 210+210→393 |
| 78M | 110→498, 210→463, 410→443 | 210+110→384, 210+210→365 |
| 173M | 110→450, 210→423, 410→409 | 210+110→359, 210+210→354 |
| 306M | 110→451, 210→404, 410→381 | 210+110→347, 210+210→325 |

MiniT2I 기준은 546.3, WebSSL-DINO-1B 기준은 277.1이다. 실측 RePR의 개선과 아직 도달하지 않은 WebSSL 기준을 함께 읽어야 한다.

### 7배·8배는 어떤 나눗셈인가?

원문 power-law fit의 예측값을 나누면 다음과 같다. 이는 **기재 수치의 산술 확인**이며 fit 자체의 독립 재현이 아니다.

$$
\frac{1.1685129\times10^{22}}{1.504\times10^{21}}
\approx7.77,
\qquad
\frac{9.78\times10^{21}}{1.229\times10^{21}}
\approx7.96.
$$

원문은 관측한 약 두 자릿수 FLOP 범위에서 추세를 맞추고, 측정한 로그 범위의 최대 한 배만큼 바깥으로 외삽한다고 설명한다. 이 주장은 fit의 형태, 데이터 크기, 과적합, 후속 생성 모델 학습 예산에 영향을 받는다. 각 점의 전체 세부 FLOPs·회귀 잔차·다중 seed 불확실성이 공개 표만으로 완전히 주어지지 않아, 이 리뷰는 새 곡선을 임의로 적합하거나 신뢰구간을 만들어 넣지 않았다. [§6.2·A.7·Tables 7–8, pp.8·22](https://arxiv.org/pdf/2610.20533v1#page=22)

### FID와 FD의 수치를 읽는 기준

리뷰 보조 표기로 특징 추출기를 $h$, 실제·생성 특징의 평균을 $\mu_r,\mu_g$, 공분산을 $C_r,C_g$라 놓으면 Fréchet 거리의 표준 형태는

$$
D_F(h)=\|\mu_r-\mu_g\|_2^2+
\operatorname{Tr}\!\left(C_r+C_g-
2(C_r^{1/2}C_gC_r^{1/2})^{1/2}\right).
$$

표준 FID 정의의 근거는 [Heusel et al., 식 (6), p.6](https://arxiv.org/pdf/1706.08500v6#page=6)이다. FID는 Inception 특징, 이 논문의 GPIC FD는 DINOv2 특징을 사용한다. 값은 이상적으로 0 이상이며 낮을수록 좋지만, 0도 해당 특징의 1·2차 모멘트가 같다는 뜻이다. 전체 이미지 분포나 텍스트 조건 충실도의 완전 일치를 의미하지 않는다. 유한 표본의 평균·공분산 추정 편향이 있으므로 10K/50K를 섞지 않고, 재구성 입력으로 만든 rFID와 새로 생성한 이미지의 FID도 구분한다. 원문만으로 확정할 수 없는 특징 추출 세부·reference subset·오차 막대는 미확인으로 남긴다.

## 리뷰 해석과 다음 검증

이 연구가 보여 준 강점은 정보량에서 출발한 표현 목적을 실제 패치 학습과 합산 비용 평가까지 연결했다는 것이다. 가장 먼저 독립 검증할 항목으로는 **실측 범위에서 토크나이저 비용을 모두 기록한 작은 예산 재현**을 권한다. 외삽 목표에 바로 도전하기보다 인코더·생성 모델 계산의 경계를 확인할 수 있기 때문이다.

다른 경로는 PMI/가중 PMI 타깃의 편향·분산을 작은 이산 데이터에서 검증하거나, 기존 LeJEPA 표현의 분포 비퇴화성과 생성 정보 보존을 같은 조건에서 비교하는 것이다. 다음 연구 범위는 이 셋 중 계산 재현, 추정량 검증, 표현 비교 중 무엇을 우선할지 결정하면 된다. 이 제안들은 수행된 실험 결과가 아니다.

## 출처와 읽은 범위

- 원문: Alan Baade 외 전체 저자 7명의 [arXiv:2610.20533v1](https://arxiv.org/abs/2610.20533v1), 2026-10-14 최초 공개, preprint, 22쪽. 전체 저자는 상단 서지 정보에서 확인할 수 있다.
- 두 읽기: 2026-10-23 개관 문서를 먼저 저장한 뒤, 본문 §1–7·A.1–A.7 활성 LaTeX를 모두 읽고 PDF의 주요 식·실험·시각자료를 대조했다. 이 글은 A.1–A.7에 대응하며 §3–4의 유도는 A.4의 이해를 위해 이동·보충했다. 참고문헌 전체의 인용 원전은 미독이다.
- 직접 참조 원전: CPC v2 pp.2–4, VDM v6 pp.17–18. Latent Forcing v1 pp.1–3, JiT v2 pp.1–3, RAE v1 pp.1–4, DINOv2 v2 pp.4–6을 비교에 사용했다. LeJEPA v3의 기존 전문 검토 기록을 재사용했다.
- 검증: 확률 연쇄법칙, SNR 미분, 비PSD PMI와 비대칭 SVD, support/argmax 구분, 가역 표현의 확률비, 공유 난수 분산, 외삽 비율 등 리뷰 보충 계산을 검사했다. 이는 부분 수학 검토이며 원문 전체의 형식 검증이 아니다.
- **미실행**: GPU 훈련, 공개 코드 실행·전체 감사, 데이터셋·원시 로그·다중 seed 재현, 외삽 fit의 독립 재적합과 미래 예산 성능 검증.
- Figures 5–8과 대표 Figure 1은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 원문 자산이다. 원래 데이터·레이아웃·글꼴을 유지하고 저자 여백 trim, 해상도 조정·WebP 변환을 적용했다. 사진 예시의 인코딩은 품질 92이며 새 모델 출력을 합성하지 않았다.
