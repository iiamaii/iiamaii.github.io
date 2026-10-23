---
title: "[2/2] RePR 심화: PMI 유도·구현·스케일링 수치를 따라가기"
description: "부록 A.1–A.7의 알고리즘·PMI 조건·원시 실험 수치를 보존하고, 확률 유도와 재구성 주장의 조건을 검토한다."
date: "2026-10-23"
publishedAt: "2026-10-23T15:12:15+09:00"
updatedAt: "2026-10-23T15:26:26+00:00"
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

RePR의 부록은 상대 예측 표현(Relative Prediction Representations)의 의사코드·구현 설정·점별 상호정보량(pointwise mutual information, PMI)의 성질과 스케일링 원시 수치를 담는다. A.7 Table 7에서 78M 인코더는 픽셀 예측 410 epoch의 프레셰 인셉션 거리(Fréchet Inception Distance, FID)-10K가 34.31이지만, 픽셀 210 epoch 뒤 특징 예측 210 epoch를 더한 반복 2(합 420 epoch)에서는 27.80이다(생성 모델은 JiT-B 80 epoch). [원문 Table 7](https://arxiv.org/pdf/2610.20533v1#page=22) **리뷰 해석.** 원문의 이론 중 확산 가중치(§4.1)와 연쇄법칙(§4.3)은 정확한 유도지만, 재구성(A.4)과 특징 예측의 PMI 보존(§5.4)은 최적 인수분해를 전제한 논증이라, 학습된 인코더에 대한 근거는 Table 4의 재구성 최대 신호 대 잡음비(peak signal-to-noise ratio, PSNR) 21.72 dB(DINOv2 18.09 dB)와 Table 7·8의 반복 2 이득이다.

[\[1/2\] 본문 편](/reviews/repr-latent-tokenization/)은 §1–7을 다룬다. 원문에는 번호 붙은 Theorem/Lemma나 긴 형식 증명이 없고, 이론 부록은 A.4의 짧은 재구성 논증이다. 본문 §3–4 확률식의 중간 계산은 A.4 끝의 **리뷰 보충 유도**에 모았다.

## A.1 알고리즘 (Algorithm)

### 무엇이 타깃이고 무엇이 학습되는가?

Algorithm 1은 이미지 $X$에서 문맥 $A,B$, 합친 문맥 $AB$, 두 문맥에 들어 있지 않은 타깃 패치 $p$를 뽑는다. 표현 경로에 필요한 입력은 $A,B,Bp$이고, 예측 경로에 필요한 입력은 $B,AB$다.

| 경로 | 계산 | 역할 |
| --- | --- | --- |
| MAR 학습 | $E_\theta,D_\theta,M_\theta$ | 마스크로 선택한 예측 타깃을 복원하는 생성 손실 |
| 정보 차이 추정 | 같은 $p_t,t$를 사용해 $p$를 두 문맥에서 복원 | 다른 문맥 $A$가 추가될 때의 가중 예측 이득 |
| 표현 학습 | $\langle R_\theta(A),R_\theta(Bp)-R_\theta(B)\rangle$ | 정보 차이와 맞아야 할 표현의 변화 |

다음은 원문 Algorithm 1의 핵심 순서를 옮긴 의사코드다. 원문 의사코드의 $R$은 공유 인코더 출력에 적용되는 branch이며, 본문의 $R_\theta(\mathbf{x})$는 이를 포함한 전체 표현 경로를 뜻한다. 효율용 $A_{\mathrm{Repeat}}$·$p_{\mathrm{Repeat}}$는 원문도 생략한다.

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

원문처럼 PMI 타깃 계산은 `no_grad()` 안에 있어, 표현 손실이 자기 타깃 숫자를 움직여 줄어들 수는 없다. 다만 표현 손실의 기울기는 공유 인코더 $E_\theta$로 흐르고, MAR 손실은 예측 모델을 계속 개선한다. 원문 Table 5에 따르면 MAR+표현 공동 단계와 MAR 단독 단계를 1:3으로 섞는다. 특징 예측 반복에서는 MAR 타깃이 $R_{\theta_{\mathrm{old}}}(E_{\theta_{\mathrm{old}}}(x))$로, $p$가 그 특징의 $p_i$ 위치 값으로 바뀐다. 식 (16)은 $M_\theta(z_p,p_t,t)$, 의사코드는 `M(p_t, t, d)` 순서로 인자를 적는다. [A.1·Algorithm 1, p.16](https://arxiv.org/pdf/2610.20533v1#page=16)

## A.2 추가 실험 (Additional Experiments)

### PMI 적분 표본 수와 공유 노이즈

[![Figure 5. 4096-step PMI 추정값을 기준으로 한 Monte Carlo 표본 수별 오차와 shared/independent noise 비교.](/assets/reviews/repr/figure5.webp)](/assets/reviews/repr/figure5.webp)

*원문 Figure 5, p.16. 가로축은 Monte Carlo 적분 표본 수 $N$, 세로축은 4096표본 추정값과의 평균 제곱 오차(mean squared error, MSE)다. 보라 실선은 공유 노이즈, 주황 점선은 독립 노이즈다. 기준은 참 PMI가 아니라 4096표본 근사다.*

두 예측은 같은 패치와 같은 noisy realization을 놓고 비교하는 것이 유리하다. 리뷰 보충으로 두 오차 추정치를 $U,V$라 쓰면

$$
\operatorname{Var}(U-V)
=\operatorname{Var}(U)+\operatorname{Var}(V)
-2\operatorname{Cov}(U,V).
$$

공유한 난수가 두 추정치를 함께 흔들면(양의 공분산) 차이의 분산이 줄고, 원문 Figure 5 캡션도 노이즈 공유가 분산을 크게 줄인다고 쓴다. 저자들은 $N=32$를 고르며, 이때 diffusion 적분(작은 head를 $N$번 실행하는 부분)이 학습 전체 FLOPs의 1.3%다.

### 재구성 지표

원문 Table 4는 RAE 디코더 설정으로 잰 재구성 성능이다.

| 인코더 | 재구성 FID(rFID) ↓ | PSNR ↑ |
| --- | --- | --- |
| DINOv2-B | 0.49 | 18.09 dB |
| RePR-B 반복 3, 600ep 표기 | 0.47 | 21.72 dB |

PSNR은 픽셀 오차를 본다. 픽셀 범위의 최대값을 $L$, 비교할 원소 수를 $n$이라 두면 표준 정의는

$$
\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2,
\qquad
\operatorname{PSNR}=10\log_{10}\frac{L^2}{\operatorname{MSE}}
$$

이며, 같은 픽셀 범위·전처리에서 높을수록 좋다. rFID는 재구성 이미지 집합과 실제 집합의 특징 분포 차이를 보는 분포 지표이고, 입력별 복원 정확도는 PSNR이 본다. A.4는 이 표의 PSNR 차이를 재구성 주장의 실측 근거로 든다. [A.2·Table 4, pp.16–17](https://arxiv.org/pdf/2610.20533v1#page=17)

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
| 생성 스케일링 평가 | ImageNet JiT-B, GPIC MiniT2I; ImageNet 환산 80 epochs |
| GPIC 조건 | FLAN-T5-large 텍스트 인코더, 최대 128 text tokens |

원문의 epoch은 처리한 이미지 뷰 수를 ImageNet 표본 수로 나눈 값이라 $A_{\mathrm{Repeat}}$와 무관하게 FLOPs에 비례한다. GPIC 1 epoch은 대략 ImageNet 환산 80 epochs다. 저자들은 학습률을 고정한 모델 하나를 100 epoch마다 저장하고, 각 체크포인트에서 10 epoch 동안 학습률을 선형으로 줄인다. 그래서 반복 2의 $210+210$은 픽셀 200+10과 특징 200+10의 합이다. 본문 Table 1의 epoch 표기와의 대응은 [1편 §6.4](/reviews/repr-latent-tokenization/)에 있다.

Figure 4는 반복 2의 총 420 epochs 모델을 쓴다. RePR-44M/78M 학습은 JiT-L 200ep와 비교한 전체 예산의 약 20%/35%다. Table 3의 RePR-78M은 반복 3 총 830 epochs이며 210+210+410이다. 저자는 JiT-L 600ep에 대응하는 전체 예산의 약 25%를 인코더에 썼다고 설명한다. RAE 비교에는 RePR-B 총 1030 epochs와 $2.19\times10^{20}$ FLOPs를 별도로 적고 I-JEPA-H의 $2.45\times10^{20}$와 비교한다.

계산은 self-attention을 제외한 $6ND$ 근사다. 저자는 ImageNet 생성 평가 하나에 약 31 GPU-hours, GPIC에는 56 GPU-hours를 적고 Figure 3 재현에 8-GPU B200 노드 250시간을 **추정**한다. [A.3·Tables 5–6, pp.17–19](https://arxiv.org/pdf/2610.20533v1#page=17)

### 왜 비대칭 역할 정보가 필요한가?

원문 A.3은 가중 PMI가 비대칭이고 PMI 자체도 양의 준정부호(positive semidefinite, PSD)가 아니므로, 추가 정보 $\mathbf{z}$에 해당하는 입력을 $\mathbf{x},\mathbf{y}$와 구별해 표시한다고 쓴다. 같은 벡터를 양쪽에 넣는 Gram 행렬은 항상 PSD다. 리뷰 보충으로 임의 계수 벡터 $c$에 대해

$$
c^\top Kc
=\left\|\sum_i c_iR(\mathbf{x}_i)\right\|_2^2\ge0,
\qquad K_{ij}=\langle R(\mathbf{x}_i),R(\mathbf{x}_j)\rangle.
$$

리뷰 보충 예로, 주변 분포가 균등하고 공동확률이 대각 0.45·비대각 0.05인 두 상태에서는 PMI 행렬이

$$
\begin{pmatrix}\log1.8&\log0.2\\\log0.2&\log1.8\end{pmatrix}
$$

이고 고유값 하나가 $\log(0.36)<0$이라, 같은 표현끼리의 내적으로는 정확히 맞출 수 없다.

원문은 역할 표시가 내적을 비대칭으로 만들어 최적점에서 PMI의 완전한 특이값 분해(singular value decomposition, SVD) 재구성을 허용한다고 쓴다. 타깃 행렬을 $\Pi=U\Sigma V^\top$라 놓으면

$$
F=U\Sigma^{1/2},\qquad G=V\Sigma^{1/2},
\qquad FG^\top=\Pi
$$

로 인수분해되며, 표현 차원이 rank 이상이어야 한다. RePR은 $\mathbf{z}$ 위치의 패치 특징에 학습 임베딩을 더해 이 역할을 구분한다.

## A.4 PMI의 성질 (Properties of PMI)

### PMI로 재구성한다는 주장

원문은 오염되지 않은 이미지 뷰 $\mathbf{x}$와 복원할 픽셀 $\mathbf{y}$를 두고, 학습 중 $\mathbf{y}$에 아주 작은 분산 $\delta>0$의 Gaussian 노이즈를 더해 모든 픽셀값에 비영 support를 준다. 그러면 모든 후보 픽셀값 $v$ 가운데 $\langle R(\mathbf{x}),R(v)\rangle$, 곧 PMI가 가장 큰 값을 골라 $\mathbf{y}$를 복원할 수 있다고 쓴다. 실제 복원 품질은 학습 때 뽑은 마스크와 수렴에 영향을 받는다고 한정하고, 실측 근거로 Table 4의 PSNR 21.72 dB(RAE ViT-XL 디코더에서 DINOv2보다 3.63 dB 높음)를 든다.

**리뷰 보충 유도.** 이 가정에서 참 픽셀값을 $y$, 커널을 $k(d)=e^{-d^2/(2\delta)}$, 픽셀값의 주변 분포를 $\pi$라 두면 후보 $v$의 PMI는 $\log k(v-y)-\log\sum_u\pi(u)k(v-u)$다. $\delta\to0$이면 참값 $y$의 PMI는 $-\log\pi(y)$로 수렴하고 나머지는 $-\infty$로 가므로, 주변 분포를 고정하면 $\delta$가 충분히 작을 때 argmax가 $y$다. 고정된 $\delta$에서는 다를 수 있다. 픽셀이 0, 1, 2만 갖고(간격 1) 참값이 $y=1$이면

$$
\operatorname{PMI}(\mathbf{x},0)>\operatorname{PMI}(\mathbf{x},1)
\iff \pi(0)<\pi(2)\,e^{-1/\delta}
$$

이다. 예를 들어 $\delta=0.5$, $\pi=(0.01,0.49,0.50)$이면 후보 0의 PMI는 약 0.61 nat, 참값 1은 약 0.39 nat라 드문 이웃값 0이 선택된다. **리뷰 해석.** 원문 논증은 주변 분포를 고정하고 $\delta\to0$으로 보내는 극한에서 성립하며, 필요한 $\delta$의 크기는 드문 픽셀값의 빈도에 따라 달라진다(이 예에서는 $\delta\le1/\log(\pi(2)/\pi(0))$).

### 특징 예측과 PMI 보존

본문 §5.4는 이 재구성 성질에서 출발한다. 최적 인수분해가 입력을 재구성하면 결정적 표현 $R$은 단사이고, 단사인 $R$은 확률을 그대로 옮기므로 무가중 PMI에서

$$
\operatorname{PMI}(\mathbf{x},R(\mathbf{y}))
=\operatorname{PMI}(\mathbf{x},\mathbf{y})
$$

이다. **리뷰 해석.** 따라서 이 등식은 앞의 재구성이 성립하는 최적점의 성질이고, 학습된 $R_{\theta_{\mathrm{old}}}$가 서로 다른 입력을 같은 특징으로 보내면 깨진다. 이때 평균 상호정보량은 데이터 처리 부등식(data processing inequality)에 따라 줄어들 수 있다. [§5.4·A.4, pp.7·17–18](https://arxiv.org/pdf/2610.20533v1#page=17)

### 리뷰 보충 유도: 본문 §3–4 확률식의 중간 계산

**대조 점수가 밀도비가 되는 이유(식 (2)).** 대조 예측 부호화(Contrastive Predictive Coding, CPC)의 [식 (5)](https://arxiv.org/pdf/1807.03748v2#page=4)는 후보 중 하나가 $p(\mathbf{y}\mid\mathbf{x})$, 나머지가 $p(\mathbf{y})$에서 왔다고 가정한다. 어느 후보가 양성인지의 Bayes posterior에서는 공통된 주변확률 곱이 약분되어

$$
p(d=i\mid\mathbf{y}_{1:N},\mathbf{x})
=\frac{p(\mathbf{y}_i\mid\mathbf{x})/p(\mathbf{y}_i)}
{\sum_j p(\mathbf{y}_j\mid\mathbf{x})/p(\mathbf{y}_j)}.
$$

가 된다. categorical cross-entropy의 최적 예측은 이 posterior이므로, log score는 PMI에 $\mathbf{x}$만의 상수를 더한 형태, 곧 식 (2)가 된다. negative를 다른 분포에서 뽑으면 비의 분모도 그 분포로 바뀐다.

**확산 가중치 $(1-t)/t^3$(식 (7)).** 원문이 참조한 Variational Diffusion Models(VDM) [식 (53–54)](https://arxiv.org/pdf/2107.00630v6#page=18)의 diffusion loss는 신호 대 잡음비(signal-to-noise ratio, SNR)의 미분을 포함한다. RePR의 경로에서는

$$
\operatorname{SNR}(t)=\frac{(1-t)^2}{t^2},
\qquad
-\frac12\frac{\mathrm{d}}{\mathrm{d}t}\operatorname{SNR}(t)
=\frac{1-t}{t^3}
$$

이므로 이 가중치는 노이즈 경로의 변분 목적에서 나온다. $t\to0$에서는 발산하며, 원문 Table 1(f)에서 이 가중치를 그대로 쓴 설정은 NaN을 냈다.

**VLB 간극.** 원문 §4.1은 끝점 항이 소거되고 denoiser가 최적일 때 VLB가 $\log p$와 같고, 예측이 좋아질수록 근사가 참 PMI에 다가간다고 쓴다. 같은 생성 모델 $G$의 정확한 log probability와 그 VLB 사이의 두 비음수 간극을 리뷰 보조 기호 $\Delta_{\mathrm{cond}},\Delta_{\mathrm{base}}$로 두면

$$
\log\frac{P_G(\mathbf{y}\mid\mathbf{x})}{P_G(\mathbf{y})}
=\operatorname{VLB}_G(\mathbf{y}\mid\mathbf{x})-\operatorname{VLB}_G(\mathbf{y})
+\Delta_{\mathrm{cond}}-\Delta_{\mathrm{base}}.
$$

이다. 두 간극의 차이는 부호가 정해지지 않으므로 학습 중의 추정은 PMI의 하한도 상한도 아니며, 두 간극이 0이 되어도 얻는 것은 모델 $G$의 우도비다. 데이터의 PMI가 되려면 $G$가 실제 조건부·주변 분포를 맞춰야 한다.

**부트스트랩 연쇄법칙(식 (13–14)).** 본문 식은 다음처럼 펼칠 수 있다.

$$
\begin{aligned}
\operatorname{PMI}(\mathbf{x},y_{\le K})
&=\log\frac{p(y_1,\ldots,y_K\mid\mathbf{x})}{p(y_1,\ldots,y_K)}\\
&=\sum_{k=1}^{K}\log\frac{p(y_k\mid y_{<k},\mathbf{x})}{p(y_k\mid y_{<k})}.
\end{aligned}
$$

앞 prefix의 식을 빼면 새 토큰의 조건부 PMI만 남고, 모든 $k$에 대해 합하면 중간 prefix의 PMI가 상쇄된다. 빈 문맥의 PMI는 0이다. 원문의 망원합은 무가중 PMI에 대한 것이며, 시간 가중을 준 증분의 합과 전체 뷰의 가중 PMI($\operatorname{PMI}^*$, 본문 식 (11)) 사이의 관계는 다루지 않는다.

## A.5 정성 결과 (Qualitative Results)

[![Figure 6. 계산량 비교에 사용한 JiT-L 조합의 비선별 class-conditional ImageNet 생성 예시.](/assets/reviews/repr/figure6.webp)](/assets/reviews/repr/figure6.webp)

*원문 Figure 6, p.20. 계산량 비교에 쓴 JiT-L 조합의 class-conditional 생성 예시이며, 저자가 비선별(uncurated)로 표기한다.*

생성은 Latent Forcing의 CFG-Interval+AutoGuidance 설정을 그대로 따라 latent diffusion에 classifier-free guidance(CFG) 3.0을 구간 $[0.06,1.0]$에 적용하고 AutoGuidance 가중치 1.5를 쓴다. A.5는 AutoGuidance 학습 비용이 전체 학습의 1.6%이며 FLOP 계산에 포함된다고 명시한다. 본문의 unguided FID-10K 스케일링 곡선과는 다른 평가 설정이다.

[![Figure 7. 공간 등변성 제약 유무에 따른 중심 패치와 나머지 패치의 cosine similarity.](/assets/reviews/repr/figure7.webp)](/assets/reviews/repr/figure7.webp)

*원문 Figure 7, p.20. 빨간 중심 패치와 다른 위치 특징의 코사인 유사도(cosine similarity)를 공간 등변성 제약 유무로 비교한다.*

## A.6 워터마킹 (Watermarking)

[![Figure 8. clean 이미지와 모든 패치에 식별 워터마크를 넣은 watermarked 이미지.](/assets/reviews/repr/figure8.webp)](/assets/reviews/repr/figure8.webp)

*원문 Figure 8, p.21. 위·아래의 clean/watermarked 예시를 비교한다. 저자들은 기존 워터마킹 방식을 그대로 따르되 워터마크를 $32\times32$에서 $16\times16$ 픽셀로 줄여, 패치 하나만으로도 특징 억제에 충분한 단서가 되게 했다.*

**리뷰 해석.** 대조 목적은 워터마크로 두 뷰를 구별하는 순간 포화되지만, RePR의 회귀 타깃은 워터마크를 포함한 공유 정보 전체의 예측 이득이라 워터마크만 표현해서는 오차가 남는다. 본문 Figure 2의 차이는 이 구조와 맞는다.

## A.7 스케일링 그래프의 원시 수치 (Raw Numbers for Scaling Plots)

### ImageNet

Table 7은 JiT-B를 80 epochs 학습한 생성 모델의 FID-10K다. 반복 2의 epochs 합은 반복 1 학습 210 epoch(200+10 decay)을 포함한다.

| 인코더 | 픽셀 예측 epochs → FID | 특징 예측 epochs → FID |
| --- | --- | --- |
| 20M | 110→41.61, 210→40.19, 410→37.01, 610→37.21 | 210+110→36.24, 210+210→34.89 |
| 44M | 110→40.93, 210→38.20, 410→36.07, 610→34.82 | 210+110→31.64, 210+210→29.48, 210+410→29.22 |
| 78M | 110→38.26, 210→35.72, 410→34.31, 610→34.00 | 210+110→29.81, 210+210→27.80, 210+410→27.36 |
| 173M | 110→37.22, 210→33.73, 410→31.27, 610→30.68 | 210+210→25.82, 210+410→24.73 |
| 306M | 110→36.14, 211→31.23 | 기재 없음 |

20M은 410→610 epoch에서 37.01→37.21로 약간 나빠진다. 306M의 두 번째 행은 다른 행의 210과 달리 원문에 211로 인쇄되어 있다. DINOv2 기준은 $1.1685129\times10^{22}$ FLOPs에서 17.95다.

### GPIC

Table 8은 MiniT2I를 1 GPIC epoch 학습한 생성 모델의 FD-DINOv2-10K이며, epochs는 ImageNet 환산량이다.

| 인코더 | 픽셀 예측 epochs → FD | 특징 예측 epochs → FD |
| --- | --- | --- |
| 44M | 110→514, 210→479, 410→456 | 210+110→416, 210+210→393 |
| 78M | 110→498, 210→463, 410→443 | 210+110→384, 210+210→365 |
| 173M | 110→450, 210→423, 410→409 | 210+110→359, 210+210→354 |
| 306M | 110→451, 210→404, 410→381 | 210+110→347, 210+210→325 |

MiniT2I 기준은 546.3, WebSSL-DINO-1B 기준은 $9.78\times10^{21}$ FLOPs에서 277.1이다. 두 표에 맞춘 거듭제곱 fit의 7.77·7.96배 예측과 연장 범위는 [본문 편 §6.2](/reviews/repr-latent-tokenization/)에서 다룬다. [원문 A.7](https://arxiv.org/pdf/2610.20533v1#page=22)

### 리뷰 보충: FID와 FD의 정의

리뷰 보조 표기로 특징 추출기를 $h$, 실제·생성 특징의 평균을 $\mu_r,\mu_g$, 공분산을 $C_r,C_g$라 놓으면 Fréchet 거리의 표준 형태는

$$
D_F(h)=\|\mu_r-\mu_g\|_2^2+
\operatorname{Tr}\!\left(C_r+C_g-
2(C_r^{1/2}C_gC_r^{1/2})^{1/2}\right).
$$

이다([Heusel et al., 식 (6), p.6](https://arxiv.org/pdf/1706.08500v6#page=6)). FID는 Inception 특징, 이 논문의 GPIC FD는 DINOv2 특징을 쓰므로 두 값의 척도가 다르다. 값이 0이면 해당 특징의 평균·공분산이 같다는 뜻이고, 유한 표본 추정 편향 때문에 10K와 50K 값은 따로 읽는다.

<span id="리뷰-해석과-다음-검증" aria-hidden="true"></span>

## 리뷰 해석과 남은 질문

**리뷰 해석.** 부록의 가장 강한 실측 근거는 Tables 7–8의 반복 2 행이다. 반복 2가 기재된 모든 인코더 크기에서 총 420 epoch의 특징 예측이 픽셀 410 epoch보다 낮은 FID·FD를 낸다(epoch 기준이며, FLOPs 기준 비교는 본문 Figure 3). 구현 쪽에서는 노이즈 공유로 적분 비용을 학습 FLOPs의 1.3%로 줄였고, A.3이 Figure 4·Table 3·Table 2에 쓴 체크포인트와 FLOPs를 따로 적어 합산 비용 비교의 경계가 분명하다.

한계는 이론과 실측 사이의 간격이다. 이론 부록은 최적 인수분해를 전제한 재구성 논증 하나이고, 그 실측 대응인 Table 4의 PSNR은 인코더마다 RAE 디코더를 따로 학습해 얻은 값이라 인코더의 정보량과 디코더 성능이 함께 반영된다. Tables 7–8도 각 설정에 단일 값만 싣는다.

남은 질문은 시간 가중을 준 증분을 회귀하는 실제 목적이 전체 뷰의 $\operatorname{PMI}^*$를 얼마나 근사하는지, 그리고 반복 3 이후에도 이득이 이어지는지다. Table 7에는 반복 3이 없고, 같은 FID-10K 평가의 반복 3 값은 Table 1(e)의 23.81 하나뿐이다.

## 출처와 읽은 범위

- 원문: *Generative Representation Learning for Scalable Latent Tokenization* · Alan Baade et al. · 최초 공개 2026-10-14 · [검토 버전](https://arxiv.org/abs/2610.20533v1) · 검토일 2026-10-23.
- 읽은 범위: v1 A.1–A.7 전체, §3–4와 연결되는 확률 유도의 중간 계산. 참고문헌 전체의 원전 전문은 제외한다.
- 참고한 외부 원전: [CPC 식 (5)](https://arxiv.org/pdf/1807.03748v2#page=4), [VDM 식 (53–54)](https://arxiv.org/pdf/2107.00630v6#page=18), [Heusel et al. 식 (6)](https://arxiv.org/pdf/1706.08500v6#page=6).
- 그림 출처: 대표 이미지(Figure 1)와 본문 Figures 5–8은 저자 v1 자산(CC BY 4.0)이며, 저자 여백 trim·해상도 조정·WebP 변환을 적용하고 사진 예시는 품질 92로 인코딩했다. 내용·배치·원래 글꼴은 유지한다. [원문](https://arxiv.org/abs/2610.20533v1).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
