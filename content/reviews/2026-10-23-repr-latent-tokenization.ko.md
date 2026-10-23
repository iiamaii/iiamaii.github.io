---
title: "[1/2] RePR: 생성하기 좋은 표현은 어떻게 학습할까?"
description: "원문 §1–7을 따라 생성 예측의 정보 차이, 패치 표현 학습, 워터마크 실험과 합산 계산량을 해설한다."
date: "2026-10-23"
publishedAt: "2026-10-23T15:12:16+09:00"
updatedAt: "2026-10-23T15:12:16+09:00"
topics: ["representation learning", "latent tokenization", "diffusion models", "generative modeling", "scaling laws"]
visibility: "public"
lang: "ko"
translationKey: "repr-latent-tokenization"
paperTitle: "Generative Representation Learning for Scalable Latent Tokenization"
paperPublishedDate: "2026-10-14"
authors: ["Alan Baade", "Kyle Sargent", "Vincent Sitzmann", "Eric Ryan Chan", "Justin Johnson", "Li Fei-Fei", "Gordon Wetzstein"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2610.20533v1"
thumbnail: "/assets/reviews/repr/figure1.webp"
thumbnailAlt: "원문 Figure 1: 조건부 예측의 정보 차이를 패치 표현의 내적 변화에 맞추는 RePR 구조."
---

분류를 잘하는 표현이 이미지를 생성하기에도 좋은 표현일까? RePR(Relative Prediction Representations)는 **다른 부분을 보여 주었을 때 새 패치를 얼마나 더 잘 예측할 수 있는지**로 표현을 학습한다. 이 논문의 질문은 생성 품질뿐 아니라 비용까지 향한다. 표현 인코더를 먼저 학습하는 데 쓴 계산을 합쳐도, 처음부터 픽셀만 생성하는 모델보다 유리한가?

이 글은 arXiv v1의 **§1→§2→§3→§4→§5→§6→§7**을 따라간다. [\[2/2\] 부록·구현·이론 심화](/reviews/repr-latent-tokenization-technical/)는 A.1–A.7의 의사코드, 확률식의 중간 유도, 재구성 주장의 조건과 scaling 원시 수치를 풀어 읽는다. 여기서 토크나이저(tokenizer)는 이미지를 연속적인 패치 특징으로 바꾸는 인코더다. 새로운 이산 코드북이나 벡터 양자화(vector quantization)를 제안하는 논문으로 읽으면 출발점부터 달라진다.

## 1. 도입 (Introduction): 생성 모델은 어떤 표현을 필요로 하는가?

자기지도 표현 학습(self-supervised representation learning)은 보통 같은 이미지의 두 증강 뷰를 가깝게 만든다. 이때 색·배경·배치처럼 분류에 불필요하다고 판단한 정보를 버리는 것은 유용할 수 있다. 그러나 생성 모델은 물체 이름뿐 아니라 위치, 질감, 배경과 세부 변화도 만들어야 한다. 저자들은 **특징 억제(feature suppression)**, 직접 지도한 전역 표현과 생성에 사용하는 패치 표현 사이의 차이, 인코더 확장 비용을 문제로 삼는다.

여기서 특징 억제와 차원 붕괴(dimensional collapse)는 구분하자. 벡터의 여러 차원이 사용되더라도 내용은 식별용 워터마크에만 의존할 수 있다. 분포의 비퇴화성(non-degeneracy)만으로 생성에 필요한 모든 변수가 남는 것은 아니다. 이전 [LeJEPA 해설](/reviews/lejepa-complete/)에서 다룬 분포 정규화와 이번 논문의 정보 보존 질문을 연결해 읽을 지점이다.

또 다른 문제는 계산량이다. 이미 만들어진 DINOv2 특징으로 생성 모델을 빨리 학습했다는 결과만으로, 그 특징을 만드는 사전학습까지 효율적이었다고 결론 내릴 수 없다. 논문은 인코더 크기·학습량을 늘릴 때 생성이 얼마나 쉬워지는지와 **인코더 학습 + 생성 모델 학습**의 합산 비용을 분리해 평가한다. [§1, pp.1–2](https://arxiv.org/pdf/2610.20533v1#page=1)

## 2. 관련 연구 (Related Work): RePR이 바꾸는 부분

원문의 세 흐름은 표현 학습, PMI와 표현의 관계, 픽셀·잠재 공간 생성이다. 그중 핵심 비교 대상 다섯 가지를 먼저 구분하면 뒤의 실험이 명확해진다.

| 접근 | 기본 설계 | RePR과의 연결·차이 |
| --- | --- | --- |
| [LeJEPA](https://arxiv.org/abs/2511.08544v3) | 뷰 간 예측 가능성과 Gaussian 표현 분포를 함께 학습 | 이번 논문은 워터마크가 쉬운 단서가 되는 조건에서 어떤 정보가 남는지 시험한다. LeJEPA의 모든 정리를 반박하는 실험은 아니다. |
| [DINOv2](https://arxiv.org/abs/2304.07193v2) | teacher–student 학습과 이미지·패치 목적식 | 생성에 재사용되는 대표 인코더이자 scaling 기준점. 워터마크 실험에 쓰인 모델은 **DINOv3-B/16**으로, DINOv2와 구별한다. |
| [RAE](https://arxiv.org/abs/2510.11690v1) | 동결한 표현 인코더에 디코더를 학습하고 그 잠재 공간에서 생성 | RePR은 생성용 표현 자체의 학습 목적과 비용을 다룬다. RAE 디코더에도 적용하지만, 주 효율 비교는 별도 경로다. |
| [JiT](https://arxiv.org/abs/2511.13720v2) | 깨끗한 이미지를 직접 예측하는 픽셀 확산 Transformer | 토크나이저 없이 학습하는 주 기준선. RePR을 추가할 가치가 있는지 합산 계산으로 비교한다. |
| [Latent Forcing](https://arxiv.org/abs/2602.11401v1) | 잠재 특징과 픽셀을 별도 시간 변수로 함께 생성 | 특징을 먼저 생성하고 이를 조건으로 픽셀을 생성하는 평가 장치. RePR의 인코더 품질과 손실 압축 디코더의 영향을 분리한다. |

주의할 점도 있다. DINOv2에는 이미 iBOT의 **패치 수준 목적식(patch-level objective)**이 있다. 따라서 차이를 “기존 DINO는 패치에 학습 신호가 없다”로 단순화하면 부정확하다. RePR의 특징은 사용할 표현의 내적 구조를 정보량 목표에 직접 맞추는 데 있다. LeJEPA도 명시적인 negative sample 없이 학습하므로, RePR의 차이를 negative sample이 없다는 사실만으로 설명할 수는 없다. [DINOv2 §4, pp.5–6](https://arxiv.org/pdf/2304.07193v2#page=5)

이론의 배경은 [CPC §2.3](https://arxiv.org/pdf/1807.03748v2#page=3)의 밀도비 추정과 [Variational Diffusion Models, E.1–E.3](https://arxiv.org/pdf/2107.00630v6#page=17)의 확산 변분 하한이다. 이 원전들은 필요한 절을 선택해 읽었으며 모든 관련 논문을 전문 검토한 서베이는 아니다. [원문 §2, pp.2–3](https://arxiv.org/pdf/2610.20533v1#page=2)

## 3. 배경 (Background): PMI는 두 관측의 관계를 재는 값이다

점별 상호정보량(pointwise mutual information, PMI)은 특정 두 관측이 독립일 때보다 얼마나 더 자주 함께 나오는지 나타낸다. 원문의 굵은 기호 $\mathbf{x},\mathbf{y}$는 관측된 뷰이며, $R_\theta$는 표현 모델이다.

$$
\operatorname{PMI}(\mathbf{x},\mathbf{y})
=\log\frac{p(\mathbf{y}\mid\mathbf{x})}{p(\mathbf{y})}
=\log\frac{p(\mathbf{x},\mathbf{y})}{p(\mathbf{x})p(\mathbf{y})}.
$$

원문 식 (1)이다. 예를 들어 어떤 패치가 원래 확률 0.01로 나타나는데 다른 뷰를 알면 확률이 0.1이 된다면, PMI는 $\log 10$이다. 이 수치는 설명용 예시다. 양수는 함께 나타나는 경향, 0은 해당 쌍의 독립 수준, 음수는 독립일 때보다 드문 조합을 뜻한다. PMI 자체가 확률이나 유사도의 양수 점수는 아니다. 평균을 취한 상호정보량(mutual information)과도 다르다.

대조학습의 최적 점수는 조건부 확률과 주변 확률의 비에 비례한다. 로그 내적 점수로 쓰면 원문 식 (2)처럼

$$
\langle R_\theta(\mathbf{x}),R_\theta(\mathbf{y})\rangle
\approx\operatorname{PMI}(\mathbf{x},\mathbf{y})+a(\mathbf{x})
$$

가 된다. $a(\mathbf{x})$는 후보들 사이의 softmax가 구별할 수 없는 기준값이다. 이는 충분한 모델 용량·지정한 negative sampling 분포 등 아래의 이상적인 최적 점수 설명이지, 임의의 유한 차원 신경망이 항상 PMI를 정확히 복원한다는 정리는 아니다.

RePR은 이 연결에서 출발해 **PMI를 생성 모델로 추정하고 표현 내적을 그 값에 회귀**시킨다. 식 (3)의 $\log P_{G_\theta}(\mathbf{y}\mid\mathbf{x})-\log P_{G_\theta}(\mathbf{y})$가 그 추정값이다. 학습된 모델의 확률을 실제 데이터 확률로 놓을 수 있는지는 다음 절의 조건에 달려 있다. [§3, p.3](https://arxiv.org/pdf/2610.20533v1#page=3)

## 4. 방법 (Method): 생성의 상대적인 예측 이득을 표현에 옮기기

첫 구상은 표현 내적과 추정 PMI 사이의 제곱 오차다. 원문 식 (4–5)의 핵심은

$$
\mathcal{L}_{\mathrm{PMI}}
=\left(\langle R_\theta(\mathbf{x}),R_\theta(\mathbf{y})\rangle
-\operatorname{stopgrad}\!\left[
\log P_{G_\theta}(\mathbf{y}\mid\mathbf{x})-\log P_{G_\theta}(\mathbf{y})
\right]\right)^2.
$$

정지 기울기(stop-gradient)는 이 회귀 항에서 생성 모델이 정답 쪽 숫자를 움직여 오차를 줄이지 못하게 한다. 생성 모델에는 별도의 예측 손실이 학습 신호를 제공한다. 다만 큰 이미지 전체의 확률을 매번 추정하면 비용이 크고, 실제 PMI는 미세한 저수준 정보에도 민감하다. §4.1–4.4는 이 두 문제를 순서대로 해결한다.

### 4.1 생성 모델에서 PMI 추정 (PMI Estimation from Generative Models)

이산 분포를 출력하는 모델이면 log probability를 읽으면 된다. 확산 모델에는 변분 하한(variational lower bound, VLB)을 사용한다. 원문 식 (6)은 음의 log likelihood를 prior loss, reconstruction loss, diffusion loss로 나눈다. 선형 노이즈 경로는

$$
\mathbf{z}_t=(1-t)\mathbf{x}+t\boldsymbol{\epsilon},
\qquad\boldsymbol{\epsilon}\sim\mathcal{N}(0,I).
$$

이 논문의 식에서는 **$t=0$이 데이터, $t=1$이 순수 노이즈**다. JiT·Latent Forcing 원전의 시간 방향을 무심코 그대로 옮기면 가중치 해석이 뒤집힌다.

최적 denoiser와 소거되는 끝점 항이라는 이상적 조건에서 식 (7–8)은, 다른 뷰를 보았을 때 줄어드는 복원 오차를 적분해 PMI를 얻는다.

$$
\operatorname{PMI}(\mathbf{x},\mathbf{y})
=\mathbb{E}_{t,\boldsymbol{\epsilon}}\!\left[
\frac{1-t}{t^3}
\left(
\|\mathbf{y}-\mathbb{E}(\mathbf{y}\mid\mathbf{y}_t,t)\|_2^2
-\|\mathbf{y}-\mathbb{E}(\mathbf{y}\mid\mathbf{x},\mathbf{y}_t,t)\|_2^2
\right)\right].
$$

같은 noisy target을 두고 조건이 없을 때와 있을 때의 오차를 비교하는 것이다. 조건 덕분에 예측이 쉬워지면 양의 학습 신호가 생긴다. 원문은 유한 정밀도 픽셀·특징을 이산 데이터로 취급하며, 등호는 최적 추정과 끝점 조건에 의존한다. 실제 신경망에서는 **두 우도 근사의 차이**다. 두 하한을 뺐다고 PMI의 하한이 되는 것도 아니다. 가중치의 유도와 근사 간극은 2편에서 전개한다. [§4.1, p.4](https://arxiv.org/pdf/2610.20533v1#page=4)

### 4.2 시간 가중치 (Timestep Weighting)

모든 공유 비트를 같은 비중으로 맞추면 질감·고주파 세부가 중요한 학습 신호를 압도할 수 있다. 저자들은 타깃을 더 강하게 오염시킨 뒤에도 남는 의존성을 강조한다. 노이즈 수준 $s$에서의 PMI를 $q(s)$로 가중한 $\operatorname{PMI}^*$를 정의하고, 실용적으로는 시간 샘플링 $f(t)$와 오차 가중치 $g(t)$를 사용한다.

$$
\operatorname{PMI}^{*}(\mathbf{x},\mathbf{y})
=\mathbb{E}_{s,\boldsymbol{\epsilon}}
\left[q(s)\operatorname{PMI}(\mathbf{x},\mathbf{y}_s)\right].
$$

이것은 어떤 정보를 표현의 유사성 기준으로 삼을지 정하는 귀납적 편향(inductive bias)이다. **가중 PMI와 원래 PMI의 구분이 중요하다.** 실제 설정은 $g(t)=(1-t)/\max(0.33,t)^2$이며 시간 분포도 바꾼다. 원래 식의 $(1-t)/t^3$를 그대로 쓰는 설정은 Table 1(f)에서 NaN을 냈다. 따라서 “정확한 PMI를 계산하기만 하면 된다”는 요약으로는 실험을 설명할 수 없다. [§4.2, pp.4–5](https://arxiv.org/pdf/2610.20533v1#page=4)

### 4.3 부트스트랩 추정 (Bootstrapped PMI Estimation)

큰 뷰 전체를 한 번에 예측하는 대신, 기존 문맥에 작은 토큰 하나를 더할 때의 정보 증가를 사용한다. 원문 식 (13)의 확률 연쇄법칙은

$$
\operatorname{PMI}(\mathbf{x},y_{\le k})
-\operatorname{PMI}(\mathbf{x},y_{<k})
=\log\frac{p(y_k\mid y_{<k},\mathbf{x})}{p(y_k\mid y_{<k})}.
$$

오른쪽은 기존 문맥 $y_{<k}$를 이미 아는 상태에서 $\mathbf{x}$가 새 토큰 예측에 보태는 이득이다. 이 차이를 모든 토큰에 대해 합하면 중간 항이 소거되어 전체 PMI를 되찾는다. 이 망원합(telescoping sum) 관계가 작은 타깃을 학습해 큰 문맥 사이의 관계를 지도하는 근거다. 실제 가중·유한 표본 목적이 자동으로 모든 PMI 관계를 정확히 만족한다는 뜻은 아니다. [§4.3, p.5](https://arxiv.org/pdf/2610.20533v1#page=5)

### 4.4 표현 목적식 (Representation Objective)

일반 문맥 $\mathbf{x},\mathbf{y}$와 추가 정보 $\mathbf{z}$를 쓰면 원문 식 (15)는 다음 형태다.

$$
\mathcal{L}_{\mathrm{RePR}}
=\left(
\left\langle R_\theta(\mathbf{x}),
R_\theta(\mathbf{y},\mathbf{z})-R_\theta(\mathbf{y})\right\rangle
-\widehat{\operatorname{PMI}}^{*}(\mathbf{x},\mathbf{z}\mid\mathbf{y})
\right)^2.
$$

모자 표시는 리뷰에서 **실제 Monte Carlo·학습 모델 추정값**을 구별하려고 붙인 보조 표기다. 원문은 $\operatorname{PMI}^*$로 적는다. 타깃은 $\mathbf{y}$만 있을 때와 $\mathbf{x},\mathbf{y}$가 함께 있을 때의 $\mathbf{z}$ 복원 오차 차이에 $g(t)$를 곱해 평균하고, 전체를 stop-gradient로 둔 값이다.

이 손실은 “두 뷰는 무조건 비슷해져라”보다 구체적이다. 추가 패치를 보았을 때 표현이 얼마나 변해야 하는지, 그 변화가 다른 뷰와 얼마나 관련되어야 하는지를 생성 예측으로 지도한다. 그러나 가중 PMI는 비대칭이고 PMI 행렬은 일반적으로 양의 준정부호(positive semidefinite, PSD)가 아니다. 구현은 추가 패치의 역할을 구별하는 임베딩을 넣어 비대칭 인수분해를 허용한다. 이 중요한 보완은 A.3에 있으며, 같은 벡터를 양쪽에 쓰는 단순 Gram 행렬로 이해하면 놓친다.

## 5. 구조 (Architecture): 두 경로가 같은 예측 이득을 설명한다

[![Figure 1. 아래 예측 경로의 가중 정보 차이를 위 표현 경로의 내적 차이가 맞추고, 오른쪽은 위치 상대적인 패치 출력을 만드는 구조.](/assets/reviews/repr/figure1.webp)](/assets/reviews/repr/figure1.webp)

*원문 Figure 1, p.6. 위쪽은 표현 회귀, 아래쪽은 같은 타깃 패치의 두 조건부 예측, 오른쪽은 상대 위치 query를 이용한 출력이다. 그림의 따옴표가 있는 “log P”는 가중 추정값으로 읽는다. 대표 이미지도 이 그림을 사용한다. 모든 본문 그림은 눌러 확대할 수 있다. 그림은 CC BY 4.0 원문 자산에서 저자의 여백 trim을 적용하고 rasterize·WebP 인코딩했다. 원래 글꼴·화살표·데이터를 유지했다.*

### 5.1 예측 모델 (Prediction Model)

예측 모델은 마스킹 자동인코더(masked autoencoder, MAE)의 회귀 머리를 작은 확산 다층 퍼셉트론(multilayer perceptron, MLP)으로 바꾼 MAR 구조다. $E_\theta$는 보이는 패치를 인코딩하고 $D_\theta$는 마스크 토큰을 포함해 문맥을 만든다. 작은 $M_\theta$가 noisy patch를 복원한다. 원문 식 (16)의 $z_p$는 패치 위치의 **디코더 잠재 문맥**이며, §4의 추가 정보 $\mathbf{z}$와 같은 대상이 아니다.

$$
\mathcal{L}_{\mathrm{MAR}}
=w(t)\|M_\theta(z_p,p_t,t)-p\|_2^2.
$$

중요한 효율 장치는 큰 encoder/decoder의 계산을 공유한 뒤 작은 diffusion head만 여러 노이즈에서 반복하는 것이다. negative sample이 없다는 것과 추가 계산이 없다는 것은 별개다.

### 5.2 패치 특징 표현 (Representing Patch Features)

표현은 하나의 분류 토큰 대신 $H\times W\times D$의 공간 특징이다. $H,W$는 출력 패치 격자이고 $D$는 특징 차원이다. 내적은 대응하는 공간 위치에서 계산한다. 인코더에 절대 위치 단서를 주지 않고, 보이는 입력의 중심에 대한 상대 위치 query가 cross-attention으로 출력 특징을 읽는다. 이 설계는 공간 등변성(spatial equivariance)을 유도한다. 이미지가 이동하면 특징도 공간적으로 따라 이동하게 만드는 의도다.

A.3은 RoPE와 추가 역할 임베딩, 이미지 바깥 padding query를 설명한다. A.5의 특징 유사도 그림은 이 위치 설계의 효과를 보여 준다. 모든 경계·마스크 조건에서 정확한 등변성을 증명한 것으로 확대해서 읽지는 않는다.

### 5.3 마스킹 전략 (Masking Strategy)

같은 이미지에서 서로 겹칠 수 있는 문맥 $A,B$와 두 문맥에 없는 타깃 패치 $p$를 고른다. $AB$는 합친 문맥, $Bp$는 $B$에 타깃을 더한 입력이다. 예측 모델은 $p$를 $B$만 보고 예측한 경우와 $AB$를 보고 예측한 경우를 비교한다. 표현 모델은 $R_\theta(A)$와 $R_\theta(Bp)-R_\theta(B)$의 내적을 맞춘다.

마스크 비율은 고정값이 아니라 $[M_A,100\%]$, $[M_B,100\%]$에서 뽑는다. 기본 최소 마스크 비율은 75%, 93.75%다. $256\times256$ 이미지와 $16\times16$ 패치에서는 각각 최대 64개, 16개가 보인다. $A_{\mathrm{Repeat}}=8$, $p_{\mathrm{Repeat}}=16$은 중간 계산을 재사용하기 위한 반복 설정이다.

### 5.4 반복 특징 예측 (Iterative Feature Prediction)

첫 학습에서는 픽셀을 예측한다. 다음 학습에서는 이전 모델 $R_{\theta_{\mathrm{old}}}$의 전체 이미지 특징을 구하고, 같은 위치의 특징을 타깃으로 삼는다. 이로써 저수준 픽셀만으로 정해지던 학습 신호를 바꾼다. 다만 $\operatorname{PMI}(\mathbf{x},R(\mathbf{y}))=\operatorname{PMI}(\mathbf{x},\mathbf{y})$는 $R$이 필요한 정보를 가역적으로 보존하는 이상적 조건에 의존한다. 실제 유한 차원·가중·마스킹 표현의 자동 보장은 아니다. A.4의 조건과 함께 2편에서 해설한다.

### 5.5 공동 최적화 (Joint Optimization)

예측 손실과 표현 손실을 함께 학습하지만 표현 손실은 네 번 중 한 번만 적용한다. $E_\theta$를 공유하므로 두 경로는 완전히 독립적인 네트워크가 아니다. 표현 회귀의 타깃 경로는 stop-gradient, MAR 경로는 예측 손실로 학습한다. 이 구분은 알고리즘을 이해하거나 구현할 때 핵심이다. [§5, pp.5–7·A.1/A.3](https://arxiv.org/pdf/2610.20533v1#page=6)

## 6. 실험 (Experiments): 정보 보존과 비용을 어떻게 검사했는가?

### 6.1 특징 억제 (Feature Suppression)

[![Figure 2. 학습 뷰 수를 반영한 epoch에 따른 clean·watermarked ImageNet의 검증 top-1 곡선.](/assets/reviews/repr/figure2.webp)](/assets/reviews/repr/figure2.webp)

*원문 Figure 2, p.7. 가로축은 이미지 수와 view 수를 반영한 유효 학습 epoch, 세로축은 ImageNet 검증 top-1 정확도(%)다. 점선은 clean, 실선은 watermarked다. 파랑 DINOv3-B/16, 노랑 LeJEPA-S/8, 검정 RePR-78M을 비교한다.*

모든 이미지 패치에 이미지 ID를 식별할 수 있는 워터마크를 넣으면, 서로 다른 뷰가 같은 이미지인지 구별하는 쉬운 단서가 생긴다. DINOv3와 LeJEPA의 watermarked 곡선은 낮은 정확도로 떨어지지만 RePR은 분류에 유용한 내용을 상당 부분 유지한다.

이 실험은 **해당 워터마크 조건의 강건성(robustness)**을 지지한다. RePR도 clean 대비 손실이 있고, clean 분류 곡선에서는 DINOv3가 RePR보다 높다. 따라서 모든 분류 문제에서 RePR이 우월하다거나 모든 숨은 생성 변수를 보존한다고 결론 내릴 수 없다. 그래프의 probing 결과와 실제 생성 품질도 구분해야 한다. 원문은 이 비교에 다중 seed 신뢰구간을 제시하지 않는다. 워터마크의 실제 모습과 $16\times16$ 설정은 2편 A.6에서 확인한다.

### 6.2 확장 법칙 (Scaling Laws)

[![Figure 3. 인코더 학습 FLOPs와 ImageNet FID-10K·GPIC FD-DINOv2-10K의 관계, 반복 학습과 외삽선.](/assets/reviews/repr/figure3.webp)](/assets/reviews/repr/figure3.webp)

*원문 Figure 3, p.8. 가로축은 인코더 학습 FLOPs의 로그 척도다. 왼쪽은 ImageNet에서 JiT-B를 80 epochs 학습한 뒤의 FID-10K, 오른쪽은 GPIC에서 MiniT2I를 1 GPIC epoch 학습한 뒤의 FD-DINOv2-10K다. 모두 낮을수록 좋다. 원은 픽셀 예측, 마름모는 이전 특징 예측이며 색은 인코더 크기다. 관측 점 바깥으로 뻗은 선과 기준점의 위치를 구분해서 읽어야 한다.*

생성하기 쉬운 정도인 확산 용이성(diffusability)을 평가하려면, 인코더마다 생성 모델의 크기와 학습 기간을 고정해야 한다. 저자들은 Latent Forcing으로 특징을 먼저 생성하고 이어서 픽셀을 생성한다. 별도 손실 압축 디코더의 복원 한계가 결과를 정하지 않도록 하는 선택이다. 여기서 손실 없는 생성 경로라는 표현은 **인코더가 완벽하게 가역적이라는 뜻이 아니라 최종 타깃이 원래 픽셀이라는 뜻**이다.

폭은 256–1024, 추론 인코더는 20M–306M으로 바꾸고 깊이는 고정한다. 더 큰 모델과 더 많은 학습 계산, 두 번째 특징 예측 반복이 대체로 지표를 개선한다. 그러나 ImageNet의 폭 1024 모델은 200 epochs 이후 과적합하기 시작한다고 보고한다. GPIC에서는 더 큰 예산까지 개선이 이어진다.

초록의 **7배·8배는 외삽(extrapolation)**이다. A.7의 원시 수치에 따르면 같은 기준 품질에 필요한 인코더 학습 계산은 다음과 같이 예측된다.

| 평가 | 기준 인코더의 비용·품질 | RePR 외삽 비용 | 비용 비율 |
| --- | --- | --- | --- |
| ImageNet, FID-10K | DINOv2: $1.1685129\times10^{22}$ FLOPs, 17.95 | $1.504\times10^{21}$ FLOPs | 약 1/7.77 |
| GPIC, FD-DINOv2-10K | WebSSL-DINO-1B: $9.78\times10^{21}$ FLOPs, 277.1 | $1.229\times10^{21}$ FLOPs | 약 1/7.96 |

실측 표의 RePR-173M 반복 2는 ImageNet FID 24.73, GPIC의 가장 좋은 기재 반복 2 값은 325다. 각각 기준 17.95·277.1에 아직 미치지 못한다. 따라서 저비용으로 기준점에 이미 도달한 실험이라고 쓰면 안 된다. 원시 표에는 기재된 모든 예산의 세부 FLOPs나 fit 불확실성이 없어, 이 리뷰가 적합 계수와 신뢰구간을 독립 재현한 것도 아니다.

지표 자체도 구분하자. FID(Fréchet Inception Distance)는 Inception 특징 분포의 평균·공분산 차이를, FD-DINOv2는 DINOv2 특징에서 같은 종류의 차이를 본다. 특징 공간이 다르므로 두 숫자를 서로 비교할 수 없다. 10K와 50K 표본 평가도 유한 표본 편향이 달라 직접 섞지 않는다. 이는 시각 품질의 대리 지표이며 의미 충실성·모든 다양성·개별 재구성의 정확성을 보장하지 않는다.

### 6.3 계산량을 고려한 생성 (Compute Optimal Generation)

[![Figure 4. 인코더와 생성 모델의 합산 학습 FLOPs에 따른 조건부·무조건부 unguided FID-10K.](/assets/reviews/repr/figure4.webp)](/assets/reviews/repr/figure4.webp)

*원문 Figure 4, p.9. 위는 조건부 생성, 아래는 무조건부 생성이다. 가로축에는 인코더 사전학습을 포함한 총 학습 FLOPs가 들어간다. 검정은 픽셀 JiT-L, 파랑·노랑은 RePR-44M/78M과 JiT-L의 조합이다. 두 세로축은 unguided FID-10K이며 범위가 다르다.*

$1.19\times10^{20}$ FLOPs에서 저자는 RePR 조합의 unguided FID-10K가 조건부에서 2.61, 무조건부에서 4.18 개선된다고 보고한다. 이 비교에는 지각 손실·GAN·외부 이미지 디코더가 없다. 이어 Table 3은 guidance를 사용한 별도의 50K 평가다.

| 설정 | 총 학습 FLOPs ↓ | FID-50K ↓ | IS ↑ |
| --- | --- | --- | --- |
| Pixel, JiT-L 600ep | $3.57\times10^{20}$ | 2.36 | 298.5 |
| RePR-78M + Pixel, JiT-L 440ep | $3.49\times10^{20}$ | **2.18** | **338.4** |

표의 예산은 약 2.2% 차이로 가깝지만 정확히 같은 수는 아니다. RePR 조합에는 인코더 학습이 포함되고, guidance에 쓰인 보조 학습의 계산도 포함한다고 A.5에서 밝힌다. 인셉션 점수(Inception Score, IS)는 생성물의 분류 확신과 클래스 다양성을 결합한 값이며 높을수록 좋다. 실제 데이터 분포와의 일치나 텍스트 조건 충실도를 직접 재는 지표는 아니다. [IS 원전 §4](https://arxiv.org/pdf/1606.03498v1#page=4)에서 정의와 해석을 확인할 수 있다.

이 결과는 **관측한 모델·예산에서 표현 학습에 일부 계산을 배분할 가치**를 보여 준다. 모든 예산과 모델에서 최적 배분을 증명한 것은 아니다. 또한 계산은 self-attention 항을 제외한 $6ND$ 근사다. 여기서 $N$은 매개변수 수, $D$는 처리 데이터량에 해당하며, PMI 적분 횟수 $N$이나 특징 차원 $D$와 다른 용법이다. GPU 시간이나 메모리·추론 비용이 같은 비율로 줄었다는 뜻도 아니다. GPIC는 FLAN-T5-large 텍스트 인코더를 사용하므로 ImageNet 주 비교의 “외부 모델 없음”을 전체 GPIC 파이프라인에 확대할 수 없다. [§6.3·A.3/A.5](https://arxiv.org/pdf/2610.20533v1#page=8)

### 6.4 절제 실험 (Ablation Studies)

Table 1의 여섯 묶음은 서로 다른 질문을 검사한다. 일부는 생성 지표 대신 분류 프록시(proxy)를 쓰므로 표의 숫자를 한 순위로 합치면 안 된다.

| 원문 묶음 | 설정과 결과 | 해석 |
| --- | --- | --- |
| (a), 78M·100ep | $M_A$: 62.5%→58.62, 75%→58.75, 87.5%→56.21 top-1 | 너무 적은 문맥은 불리하다. 수치는 정확도 %다. |
| (b), 78M·100ep | $M_B$: 75%→56.78, 93.75%→58.75, 98.5%→58.55 top-1 | 짧은 $B$ 문맥이 유리하지만 극단적으로 줄여도 계속 좋아지지는 않는다. |
| (c), 78M·100ep | uniform 57.63, block 58.75 top-1 | block mask가 이 probing 설정에서 더 좋다. |
| (d), 78M·200ep | 표현 loss 가중치 0.005→63.1, 0.01→63.6, 0.4→43.9 top-1 | 기본값 0.005와 표 최고값 0.01을 구별한다. 큰 가중치는 악화한다. |
| (e), 173M·총 600ep 배분 | 픽셀만 30.68 → 반복 2 24.73 → 반복 3 23.81 FID-10K | 같은 총 epochs를 특징 예측에 재배분하면 유리하다. 모든 연산까지 정확히 같은 예산이라는 뜻은 아니다. |
| (f), 78M·200ep | 원래 PMI 가중치 NaN; 바닥값 조정 42.15; 바꾼 분포·가중치 35.09/35.72 FID-10K | 안정성과 실용적인 시간 편향이 핵심이다. 기본 $\mu=1.2$의 35.72보다 $\mu=0.6$의 35.09가 낮다. |

원문 Table 1(e)의 “총 600ep”와 A.3의 체크포인트별 10ep learning-rate decay를 포함한 예산 기록은 구분해서 읽는다. 재현용 비용은 실제 실행 설정과 decay를 포함한 학습량으로 계산해야 한다. 생성 실험이 비싸서 (a–d)에 쓴 분류 정확도가 최종 생성 성능을 대신 증명하지는 않는다. [§6.4·Table 1, p.9](https://arxiv.org/pdf/2610.20533v1#page=9)

### 6.5 GAN+LPIPS 디코딩 (GAN+LPIPS Decoding)

Table 2는 RePR을 기존 RAE 경로에 넣는 전이 시험이다. ImageNet으로만 학습한 인코더와 DiT-XL 20ep라는 범위에서, MoCo-v3-B 13.84, I-JEPA-H 12.46, RePR-173M 반복 3은 **FID-50K 11.13**을 보인다. 이 표는 전체 RAE 계열의 최고 기록을 이긴 결과나 Table 3의 2.18과 직접 비교할 결과가 아니다.

디코더는 학습된 지각 거리(Learned Perceptual Image Patch Similarity, LPIPS)와 적대적 생성 신경망(Generative Adversarial Network, GAN) 손실을 사용한다. 저자들은 이 경로에 외부 사전학습 모델이 들어가고 DINOv1/8 기반 GAN 사전학습만 $4.7\times10^{20}$ FLOPs라고 적는다. 그러므로 §6.3의 통제된 합산 비용 주장과 이 보조 실험의 범위를 구별해야 한다.

## 7. 결론 (Conclusion): 어떤 기여를 가져갈 수 있는가?

저자들의 제안은 **생성에 유용한 관계를 생성 예측으로 측정하고, 그 관계를 직접 지도한 패치 표현을 만든다**는 것이다. 워터마크 시험, 반복 특징 예측, 인코더 scaling, 인코더 학습을 포함한 픽셀 생성 비교가 이를 뒷받침한다.

리뷰 관점에서 가장 유용한 질문은 “분류 잘하는 인코더 중 무엇을 쓸까?”에서 “생성 품질을 위해 인코더 학습에 얼마를 배분할까?”로 옮겨 간다는 점이다. 다만 이상적 PMI 성질, 실용적인 가중·bootstrap 손실, 유한 예산의 실험 결과는 각각 조건이 다르다. 새 토크나이저가 모든 정보를 보존하거나 무한히 확장된다고 읽을 근거는 아직 부족하다. [§7, p.10](https://arxiv.org/pdf/2610.20533v1#page=10)

다음으로 읽을 [2편](/reviews/repr-latent-tokenization-technical/)에서는 A.1–A.7의 순서대로 실제 알고리즘과 계산량 기록을 확인하고, PMI 연쇄법칙·확산 가중치·비대칭 인수분해·재구성 주장의 중간 논리를 전개한다.

## 출처와 읽은 범위

- 원문: Alan Baade, Kyle Sargent, Vincent Sitzmann, Eric Ryan Chan, Justin Johnson, Li Fei-Fei, Gordon Wetzstein, [Generative Representation Learning for Scalable Latent Tokenization, arXiv:2610.20533v1](https://arxiv.org/abs/2610.20533v1). 최초 공개 2026-10-14, 22쪽, preprint. 학회 스타일 파일을 채택 사실로 해석하지 않았다.
- 첫 번째 읽기: 2026-10-23, 공식 초록·도입 pp.1–2, 활성 LaTeX 절 지도·그림/표 캡션·§7. 독립 개관을 저장한 다음 정독했다.
- 두 번째 읽기: 본문 §1–7·A.1–A.7의 활성 저자 LaTeX 전체와 주요 PDF 수식·표·그림 대조. 참고문헌 약 pp.10–15의 모든 인용 원전을 정독하지 않았다. 본문은 이 글, 부록의 완결된 해설은 2편에 대응한다.
- 관련 원전 선택 읽기: Latent Forcing v1 pp.1–3, JiT v2 pp.1–3, RAE v1 pp.1–4, DINOv2 v2 pp.4–6, CPC v2 pp.2–4, VDM v6 pp.17–18. LeJEPA v3는 기존 본문·부록 검토 기록을 재사용했다. 논문 간 절대 FID 순위를 합치지 않았다.
- 검증: 원문 수치·표기·등호 조건 대조와 리뷰 보충 수학 계산을 수행했다. 코드 실행에 의한 GPU 훈련 재현, 원시 로그·다중 seed·독립 scaling fit 검증은 **미실행**이다. 공개 글과 원고의 수식·링크·화면 검수 상태는 인계 기록에 남긴다.
- 그림: 원문 Figures 1–4, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 저자 PDF의 원래 내용과 폰트를 유지하고 여백 trim·rasterization·WebP 변환만 적용했다. 추가 자체 제작 그림은 없다.
