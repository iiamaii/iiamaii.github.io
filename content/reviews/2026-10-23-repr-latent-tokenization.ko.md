---
title: "[1/2] RePR: 생성하기 좋은 표현은 어떻게 학습할까?"
description: "원문 §1–7을 따라 생성 예측의 정보 차이, 패치 표현 학습, 워터마크 실험과 합산 계산량을 해설한다."
date: "2026-10-23"
publishedAt: "2026-10-23T15:12:16+09:00"
updatedAt: "2026-10-23T15:26:27+00:00"
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

상대 예측 표현(Relative Prediction Representations, RePR)은 문맥이 추가될 때의 예측 이득을 패치 표현의 내적 차이로 학습한다. 인코더 학습을 포함한 총 학습량 $3.49\times10^{20}$ FLOPs에서 RePR-78M+JiT-L은 guidance를 쓴 ImageNet 50K 평가의 프레셰 인셉션 거리(Fréchet Inception Distance, FID) 2.18을 기록해, $3.57\times10^{20}$ FLOPs의 픽셀 JiT-L(2.36)보다 낮다. [원문 Table 3](https://arxiv.org/pdf/2610.20533v1#page=9) **리뷰 해석.** 합산 계산량 우위는 RePR-44M/78M 두 크기와 Figure 4·Table 3이 다룬 약 $0.5$–$3.6\times10^{20}$ FLOPs 범위에서 확인된 결과이고, 초록의 7–8배 절감은 반복 2의 실측 최고점(ImageNet FID-10K 24.73)이 DINOv2 기준 17.95에 닿기 전의 추세를 연장한 값이다.

[\[2/2\] 심화 편](/reviews/repr-latent-tokenization-technical/)은 부록 A.1–A.7의 의사코드·확률 유도·원시 수치를 다룬다. 여기서 토크나이저(tokenizer)는 이미지를 연속적인 패치 특징으로 바꾸는 인코더이며 이산 코드북을 쓰지 않는다.

## 1. 도입 (Introduction): 생성 모델은 어떤 표현을 필요로 하는가?

저자들은 표현 학습이 어떤 정보를 남길지 정한다는 데서 출발한다. 증강 불변성(augmentation invariance) 같은 기존 설계는 의미 정보를 남기고 나머지 잠재 변수를 버리는데, 이렇게 만든 DINOv2 같은 인코더가 RAE 등에서 확산 모델의 토크나이저로 쓰인다. 이 인코더들은 생성을 쉽게 하려고 설계되지 않았고, 그 위의 생성 모델보다 몇 자릿수 많은 데이터·계산으로 학습된다. 그래서 논문은 생성에 좋은 표현이 무엇인지, 그런 표현을 학습하는 편이 픽셀을 직접 모델링하는 것보다 계산 효율적인지를 묻는다.

목표는 **토크나이저 인코더 스케일링 법칙**이다. 인코더에 계산과 데이터를 더 줄수록 그 표현 위의 생성 모델이 더 빨리 수렴하고, 인코더와 생성 모델 사이의 계산 최적 배분에서 인코더 몫이 0보다 커야 한다. 저자들은 기존 토크나이저 스케일링 연구가 규모에 따라 좋아지는 디코더와, 오히려 커질수록 나빠진다고 보고된 인코더를 함께 묶어 측정했다고 지적한다.

저자들은 현재 방법이 확장 가능한 인코더가 되지 못하는 원인을 셋으로 든다. 첫째, **특징 억제(feature suppression)**: 대조학습과 LeJEPA의 목적은 소수의 비트만으로 포화될 수 있다. ImageNet probing에는 몇 개의 의미 비트로 충분하지만 생성은 데이터의 모든 잠재 변수를 샘플링해야 한다. 둘째, **인과적 이론 근거(causal theoretical grounding)**의 부재: 대조학습 등은 CLS 토큰을 최적화하지만 RAE가 토큰으로 쓰는 것은 학습 중 '창발'한 내부 패치 특징이며, 이 특징은 규모를 키우면 불안정해진다. 셋째, **구조의 확장성(architectural scalability)**: VAE·VQ-VAE 같은 고전 토크나이저는 재매개변수화 기법(reparameterization trick)이나 straight-through 추정으로 기울기 흐름에 병목을 만들고 의미를 학습하지 못해, DINOv2 같은 표현 학습 기반 토크나이저로 넘어가는 계기가 됐다.

해법은 표현 학습을 판별 과제에서 생성 쪽으로 옮기는 것이다. 단어 임베딩과 대조학습이 데이터의 점별 상호정보량(pointwise mutual information, PMI) 행렬을 암묵적으로 인수분해한다는 관찰에서 출발해, 생성 모델로 가중 PMI를 직접 추정하고 표현이 그것을 명시적으로 인수분해하게 한다. 그러면 음성 표본이 필요 없고 특징 억제에 강하며, 패치 특징을 직접 지도할 수 있다. 실험은 ImageNet과 텍스트-이미지 데이터 GPIC에서 Latent Forcing으로 생성 모델을 학습해 이득을 인코더의 확산 용이성(diffusability)으로 한정하며, 디코더·지각 손실·적대 손실·외부 모델을 쓰지 않는다고 밝힌다. [§1, pp.1–2](https://arxiv.org/pdf/2610.20533v1#page=1)

## 2. 관련 연구 (Related Work): RePR이 바꾸는 부분

**표현 학습.** 대조학습은 같은 장면의 증강 뷰를 가깝게, 다른 장면을 멀게 만들며 DINO v1–v3로 이어졌다. 저자들은 뷰가 강하게 조건화되면 목적을 사소하게 만들지 않을 어려운 음성 표본(hard negative)을 찾기 어려워, 물체 클래스는 잘 probing되지만 장면 구조는 잃는 특징 억제가 생긴다고 본다. 이를 막으려는 수작업 증강 설계, 그리고 마스킹 자동인코더(masked autoencoder, MAE)의 픽셀 예측이나 BYOL·Data2Vec·JEPA·MAGE의 잠재 예측 같은 예측 목적도 함께 정리한다.

**PMI와 표현.** PMI 인수분해와 표현 학습의 연결은 Levy–Goldberg가 Word2Vec에서 처음 보였고 이후 대조 예측 부호화(Contrastive Predictive Coding, CPC) 등 대조학습으로 확장됐다. 플라톤적 표현 가설(Platonic Representation Hypothesis)은 큰 신경망이 비슷한 구조의 표현으로 수렴하며 PMI 인수분해가 그 후보라고 제안했지만, 최근 연구는 대규모에서 신경망 표현이 그런 구조와 맞지 않음을 보였다. 그래서 RePR은 규모에서 그 구조가 창발하기를 기대하는 대신 표현을 PMI에 직접 맞추도록 지도한다.

**픽셀·잠재 공간 생성.** 확산 Transformer는 JiT 같은 픽셀 확산과 RAE·REPA 같은 잠재 확산으로 나뉜다. RAE·REPA는 ImageNet FID에서 최고 수준을 냈지만 표현 학습기의 사전학습을 통제하지 않는다. 이들이 쓰는 DINOv2-B는 ImageNet보다 100배 넘게 많은 데이터로 학습한 11억 매개변수 모델에서 증류됐다. 저자들은 그래서 표현 인코딩의 FLOP 효율이 아직 알려져 있지 않다고 정리한다. [원문 §2, pp.2–3](https://arxiv.org/pdf/2610.20533v1#page=2)

### 리뷰 보충: 실험에 등장하는 모델들의 역할

| 접근 | 기본 설계 | 이 논문에서의 역할 |
| --- | --- | --- |
| [LeJEPA](https://arxiv.org/abs/2511.08544v3) | 뷰 간 예측 가능성과 Gaussian 표현 분포를 함께 학습 | 워터마크 실험(§6.1)의 비교 대상 |
| [DINOv2](https://arxiv.org/abs/2304.07193v2) | teacher–student 학습과 이미지·패치 목적식 | 생성에 재사용되는 대표 인코더이자 스케일링 기준점(§6.2). 워터마크 실험에는 DINOv3-B/16이 쓰인다. |
| [RAE](https://arxiv.org/abs/2510.11690v1) | 동결한 표현 인코더에 디코더를 학습하고 그 잠재 공간에서 생성 | GAN+LPIPS 디코딩 전이 실험(§6.5)의 경로 |
| [JiT](https://arxiv.org/abs/2511.13720v2) | 깨끗한 이미지를 직접 예측하는 픽셀 확산 Transformer | 토크나이저 없이 학습하는 주 기준선(§6.3) |
| [Latent Forcing](https://arxiv.org/abs/2602.11401v1) | 잠재 특징과 픽셀을 별도 시간 변수로 함께 생성 | 특징을 먼저, 이어서 픽셀을 생성해 손실 압축 디코더 없이 인코더를 비교하는 평가 장치(§6.2–6.3) |

## 3. 배경 (Background): PMI는 두 관측의 관계를 재는 값이다

PMI는 특정 두 관측이 독립일 때보다 얼마나 더 자주, 또는 드물게 함께 나오는지 나타낸다. 원문의 굵은 기호 $\mathbf{x},\mathbf{y}$는 관측된 뷰이며, $R_\theta$는 표현 모델이다.

$$
\operatorname{PMI}(\mathbf{x},\mathbf{y})
=\log\frac{p(\mathbf{y}\mid\mathbf{x})}{p(\mathbf{y})}
=\log\frac{p(\mathbf{x},\mathbf{y})}{p(\mathbf{x})p(\mathbf{y})}.
$$

원문 식 (1)이다. 리뷰의 예로, 확률 0.01인 패치가 다른 뷰를 알면 확률 0.1이 된다면 PMI는 $\log 10$이다. 양수는 함께 나타나는 경향, 0은 독립 수준, 음수는 독립일 때보다 드문 조합을 뜻하며, 이를 평균한 값이 상호정보량(mutual information)이다.

원문은 CPC 저자들이 대조 임베딩의 내적이 식별할 수 없는 상수 차이를 남기고 PMI 추정값으로 수렴함을 보였다고 소개한다(식 (2)).

$$
\langle R_\theta(\mathbf{x}),R_\theta(\mathbf{y})\rangle
\approx\operatorname{PMI}(\mathbf{x},\mathbf{y})+a(\mathbf{x})
$$

$a(\mathbf{x})$는 후보들 사이의 softmax가 구별할 수 없는 기준값이다. Levy–Goldberg도 단어·문맥 임베딩 행렬의 내적이 PMI 행렬의 저차원 가중 인수분해임을 보였다. 이 최적 점수의 유도는 [2편 A.4의 보충 유도](/reviews/repr-latent-tokenization-technical/)에 있다.

RePR은 이 연결에서 출발해 **PMI를 생성 모델로 추정하고 표현 내적을 그 값에 회귀**시킨다. 생성 모델 $G_\theta$가 log probability를 준다면 식 (3)의 $\log P_{G_\theta}(\mathbf{y}\mid\mathbf{x})-\log P_{G_\theta}(\mathbf{y})$가 그 추정값이다. 학습된 모델의 확률이 실제 데이터 확률과 얼마나 맞는지는 다음 절의 조건에 달려 있다. [§3, p.3](https://arxiv.org/pdf/2610.20533v1#page=3)

## 4. 방법 (Method): 생성의 상대적인 예측 이득을 표현에 옮기기

첫 구상은 표현 내적과 추정 PMI 사이의 제곱 오차다. 원문 식 (4–5)의 핵심은

$$
\mathcal{L}_{\mathrm{PMI}}
=\left(\langle R_\theta(\mathbf{x}),R_\theta(\mathbf{y})\rangle
-\operatorname{stopgrad}\!\left[
\log P_{G_\theta}(\mathbf{y}\mid\mathbf{x})-\log P_{G_\theta}(\mathbf{y})
\right]\right)^2.
$$

정지 기울기(stop-gradient)는 이 회귀 항에서 생성 모델이 정답 쪽 숫자를 움직여 오차를 줄이지 못하게 하고, 생성 모델은 별도의 예측 손실로 학습된다. 저자들이 대조학습의 InfoNCE 대신 이 회귀를 택한 이유는, InfoNCE가 음성 표본 추출에 의존해 강하게 조건화된 설정에서 확장되지 않기 때문이다. 이 구상은 아직 두 가지 이유로 실용적이지 않다. 이미지의 PMI는 의미보다 고주파 비트에 지배되고(§4.1–4.2), 큰 입력의 우도 추정은 비싸거나 다루기 어렵다(§4.3). §4.4가 둘을 합친 목적식이다.

### 4.1 생성 모델에서 PMI 추정 (PMI Estimation from Generative Models)

이산 분포를 출력하는 모델이면 log probability를 읽으면 된다. 원문은 모든 데이터를 이산으로 가정한다(24비트 픽셀, 유한 정밀도 특징). 확산 모델에는 이산 데이터에서 정확한 확률을 주는 변분 하한(variational lower bound, VLB)을 쓰며, 원문 식 (6)은 음의 log likelihood를 prior loss, reconstruction loss, diffusion loss로 나눈다. 선형 노이즈 경로는

$$
\mathbf{z}_t=(1-t)\mathbf{x}+t\boldsymbol{\epsilon},
\qquad\boldsymbol{\epsilon}\sim\mathcal{N}(0,I).
$$

이 논문의 식에서는 **$t=0$이 데이터, $t=1$이 순수 노이즈**다. $t=1$에서는 prior loss가, 데이터가 이산이라 $t=0$에서는 reconstruction loss가 0이 되어 VLB에는 diffusion loss만 남는다. denoiser가 최적이면 이 bound가 tight해지고, 식 (7–8)은 다른 뷰를 보았을 때 줄어드는 복원 오차를 적분해 PMI를 얻는다.

$$
\operatorname{PMI}(\mathbf{x},\mathbf{y})
=\mathbb{E}_{t,\boldsymbol{\epsilon}}\!\left[
\frac{1-t}{t^3}
\left(
\|\mathbf{y}-\mathbb{E}(\mathbf{y}\mid\mathbf{y}_t,t)\|_2^2
-\|\mathbf{y}-\mathbb{E}(\mathbf{y}\mid\mathbf{x},\mathbf{y}_t,t)\|_2^2
\right)\right].
$$

같은 noisy target을 두고 조건이 없을 때와 있을 때의 오차를 비교하며, 조건 덕분에 예측이 쉬워지면 양의 학습 신호가 생긴다. 바깥 기댓값은 Monte Carlo 표본으로, 안쪽 기댓값은 학습한 denoiser로 구한다. 저자들은 VLB가 최적 denoiser에서만 $\log p$와 같지만 예측이 좋아질수록 근사가 참 PMI에 다가간다고 쓴다. 가중치 $(1-t)/t^3$의 유도와 근사 간극은 [2편](/reviews/repr-latent-tokenization-technical/)에서 다룬다. [§4.1, p.4](https://arxiv.org/pdf/2610.20533v1#page=4)

### 4.2 시간 가중치 (Timestep Weighting)

실제 확산 모델이 VLB를 그대로 최적화하지 않고 시간 가중 목적을 쓰는 것처럼, 저자들은 PMI에도 가중치를 둔다. 우도는 모든 비트를 같게 세지만 낮은 신호 대 잡음비(signal-to-noise ratio, SNR)의 정보가 의미·지각적으로 더 중요한 경우가 많기 때문이다. 타깃을 노이즈 수준 $s$로 오염시킨 $\mathbf{y}_s$의 PMI는 $s$가 커질수록 더 강한 오염에도 남는 의존성만 재며, 이를 $q(s)$로 가중한 것이 $\operatorname{PMI}^*$다.

$$
\operatorname{PMI}^{*}(\mathbf{x},\mathbf{y})
=\mathbb{E}_{s,\boldsymbol{\epsilon}}
\left[q(s)\operatorname{PMI}(\mathbf{x},\mathbf{y}_s)\right].
$$

$s$의 오염과 확산 적분의 시간 $t$가 모두 Gaussian이라 하나의 유효 시간 가중으로 합칠 수 있고, 실용적으로는 시간 샘플링 $f(t)$와 오차 가중치 $g(t)$를 쓴다. 저자들은 이를 어떤 SNR의 정보가 유사성을 정할지에 대한 귀납적 편향(inductive bias)이라 부른다. 실제 설정은 $g(t)=(1-t)/\max(0.33,t)^2$이며, 원래 식의 $(1-t)/t^3$를 그대로 쓰면 Table 1(f)에서 NaN이 난다. [§4.2, pp.4–5](https://arxiv.org/pdf/2610.20533v1#page=4)

### 4.3 부트스트랩 추정 (Bootstrapped PMI Estimation)

큰 뷰 전체의 우도를 구하려면 Monte Carlo 단계가 지나치게 많이 필요하므로, 기존 문맥에 작은 토큰 하나를 더할 때의 정보 증가를 대신 지도한다. 원문 식 (13)의 확률 연쇄법칙은

$$
\operatorname{PMI}(\mathbf{x},y_{\le k})
-\operatorname{PMI}(\mathbf{x},y_{<k})
=\log\frac{p(y_k\mid y_{<k},\mathbf{x})}{p(y_k\mid y_{<k})}.
$$

오른쪽은 기존 문맥 $y_{<k}$를 이미 아는 상태에서 $\mathbf{x}$가 새 토큰 예측에 보태는 이득이다. 이 증분을 모든 토큰에 대해 합하면 중간 항이 소거되어 전체 PMI를 되찾는다. 이 망원합(telescoping sum) 관계가 작은 타깃의 학습으로 큰 문맥 사이의 관계를 지도하는 근거다. [§4.3, p.5](https://arxiv.org/pdf/2610.20533v1#page=5)

### 4.4 표현 목적식 (Representation Objective)

일반 문맥 $\mathbf{x},\mathbf{y}$와 추가 정보 $\mathbf{z}$를 쓰면 원문 식 (15)는 다음 형태다.

$$
\mathcal{L}_{\mathrm{RePR}}
=\left(
\left\langle R_\theta(\mathbf{x}),
R_\theta(\mathbf{y},\mathbf{z})-R_\theta(\mathbf{y})\right\rangle
-\operatorname{PMI}^{*}(\mathbf{x},\mathbf{z}\mid\mathbf{y})
\right)^2.
$$

타깃 $\operatorname{PMI}^*$는 $\mathbf{y}$만 있을 때와 $\mathbf{x},\mathbf{y}$가 함께 있을 때의 $\mathbf{z}$ 복원 오차 차이에 $g(t)$를 곱해 Monte Carlo로 평균한 값이며, 전체가 stop-gradient다.

이 손실은 추가 패치를 보았을 때 표현이 얼마나 변해야 하는지, 그 변화가 다른 뷰와 얼마나 관련되어야 하는지를 생성 예측으로 지도한다. 가중 PMI는 비대칭이고 PMI는 일반적으로 양의 준정부호(positive semidefinite, PSD)가 아니어서, 구현은 $\mathbf{z}$의 위치에 학습 임베딩을 더해 내적을 비대칭으로 만든다(부록 A.3, [2편](/reviews/repr-latent-tokenization-technical/)).

## 5. 구조 (Architecture): 두 경로가 같은 예측 이득을 설명한다

[![Figure 1. 아래 예측 경로의 가중 정보 차이를 위 표현 경로의 내적 차이가 맞추고, 오른쪽은 위치 상대적인 패치 출력을 만드는 구조.](/assets/reviews/repr/figure1.webp)](/assets/reviews/repr/figure1.webp)

*원문 Figure 1, p.6. 위쪽은 표현 회귀, 아래쪽은 같은 타깃 패치의 두 조건부 예측, 오른쪽은 상대 위치 query를 이용한 출력이다. 따옴표가 있는 “log P”는 가중 추정값이다.*

### 5.1 예측 모델 (Prediction Model)

예측 모델은 MAE의 패치 회귀 머리를 작은 확산 다층 퍼셉트론(multilayer perceptron, MLP)으로 바꾼 MAR 구조다. $E_\theta$는 보이는 패치를 인코딩하고 $D_\theta$는 마스크 토큰을 포함해 문맥을 만든다. 작은 $M_\theta$가 noisy patch를 복원한다. 원문 식 (16)의 $z_p$는 패치 위치의 **디코더 잠재 문맥**이며, §4의 추가 정보 $\mathbf{z}$와 같은 대상이 아니다.

$$
\mathcal{L}_{\mathrm{MAR}}
=w(t)\|M_\theta(z_p,p_t,t)-p\|_2^2.
$$

MAR을 고른 이유는 큰 encoder/decoder 계산을 한 번 공유한 뒤 작은 확산 head만 여러 노이즈 수준에서 여러 번 실행해 평균할 수 있기 때문이다.

### 5.2 패치 특징 표현 (Representing Patch Features)

저자들은 대조학습·DINO·LeJEPA의 CLS 토큰처럼 구조를 직접 지도받는 특징과, DINO의 비CLS 토큰이나 정규화 없는 자동인코더 잠재처럼 지도받지 않는 중간 잠재를 구분한다. 후자는 명시적 지도가 없으면 발산하거나 붕괴할 수 있다. DINOv2는 이상치 패치 특징 때문에 register 토큰을 도입했고, DINOv3에서는 특징 품질이 다시 나빠져 열화 전 체크포인트에 맞추는 증류 단계를 추가했다. 저자들은 이런 불안정이 규모가 커질수록 통제하기 어려워질 수 있다며 패치 특징을 직접 지도한다. **리뷰 해석.** DINOv2에는 iBOT의 마스크 패치 토큰 목적식이 있어 비CLS 토큰도 학습 신호를 받으므로([DINOv2 §4, pp.5–6](https://arxiv.org/pdf/2304.07193v2#page=5)), RePR의 차이는 패치 특징 내적의 기하를 정보량 타깃에 직접 맞춘다는 데 있다.

표현은 하나의 분류 토큰 대신 $H\times W\times D$의 공간 특징이다. $H,W$는 출력 패치 격자이고 $D$는 특징 차원이다. 내적은 대응하는 공간 위치에서 계산한다. 인코더에 절대 위치 단서를 주지 않고, 보이는 입력의 중심에 대한 상대 위치 query가 cross-attention으로 출력 특징을 읽는다. 이 설계는 공간 등변성(spatial equivariance), 곧 이미지가 이동하면 특징도 따라 이동하는 성질을 유도한다. A.3은 RoPE와 이미지 바깥 padding query를, A.5의 Figure 7은 이 제약 유무에 따른 특징 유사도를 보여 준다.

### 5.3 마스킹 전략 (Masking Strategy)

같은 이미지에서 서로 겹칠 수 있는 문맥 $A,B$와 두 문맥에 없는 타깃 패치 $p$를 고른다. $AB$는 합친 문맥, $Bp$는 $B$에 타깃을 더한 입력이다. 예측 모델은 $p$를 $B$만 보고 예측한 경우와 $AB$를 보고 예측한 경우를 비교한다. 표현 모델은 $R_\theta(A)$와 $R_\theta(Bp)-R_\theta(B)$의 내적을 맞춘다.

마스크 비율은 고정값이 아니라 $U[M_A,100\%]$, $U[M_B,100\%]$에서 뽑아, §4.3의 망원합에 필요한 여러 문맥 크기를 지도한다. 기본 최소 마스크 비율은 75%, 93.75%로, $256\times256$ 이미지와 $16\times16$ 패치에서는 각각 최대 64개, 16개 패치가 보인다. 효율을 위해 $E(Bp)$ 계산을 여러 $A_i$에 재사용하고, 디코더 한 번으로 여러 타깃 패치 $p_j$를 적분 비용만 더해 예측한다. 그 재사용 횟수가 $A_{\mathrm{Repeat}}=8$, $p_{\mathrm{Repeat}}=16$이다.

### 5.4 반복 특징 예측 (Iterative Feature Prediction)

학습 전체를 한 번 더 반복(iteration)하면서 예측할 증분 패치를 이전 반복의 특징으로 바꾸면 추가 이득이 있다. 저자들은 이것을 MAE 같은 픽셀 예측보다 특징 공간 예측(HuBERT·iBOT·data2vec)이 표현을 크게 개선한 결과와 같은 흐름으로 본다. 반복 1은 픽셀을 예측하고, 반복 2부터는 이전 모델 $R_{\theta_{\mathrm{old}}}$로 전체 이미지를 임베딩해 같은 위치의 패치 특징을 타깃으로 삼는다. 원문은 PMI 인수분해가 최적점에서 입력을 재구성할 수 있으므로(A.4) 무가중 PMI에서는 $\operatorname{PMI}(\mathbf{x},R_\theta(\mathbf{y}))=\operatorname{PMI}(\mathbf{x},\mathbf{y})$이고, 시간 가중과 부트스트랩을 더하면 특징 공간 예측으로 학습 신호를 더 조절할 수 있다고 설명한다. 이 등식의 조건은 [2편 A.4](/reviews/repr-latent-tokenization-technical/)에서 다룬다. $R_{\theta_{\mathrm{old}}}$ 계산은 $A_{\mathrm{Repeat}}$개 뷰가 공유하므로 비용이 작다.

### 5.5 공동 최적화 (Joint Optimization)

예측 손실과 표현 손실을 함께 학습한다. 저자들은 표현 모델이 비교적 쉽게 수렴한다고 보고, 학습 속도를 위해 표현 손실을 네 번 중 한 번만 적용한다. $E_\theta$를 공유하므로 두 경로는 완전히 독립적인 네트워크가 아니며, 표현 회귀의 타깃은 stop-gradient이고 MAR 경로는 예측 손실로 학습된다. [§5, pp.5–7·A.1/A.3](https://arxiv.org/pdf/2610.20533v1#page=6)

## 6. 실험 (Experiments): 정보 보존과 비용을 어떻게 검사했는가?

### 6.1 특징 억제 (Feature Suppression)

[![Figure 2. 학습 뷰 수를 반영한 epoch에 따른 clean·watermarked ImageNet의 검증 top-1 곡선.](/assets/reviews/repr/figure2.webp)](/assets/reviews/repr/figure2.webp)

*원문 Figure 2, p.7. 가로축은 이미지 수와 view 수를 반영한 유효 학습 epoch, 세로축은 ImageNet 검증 top-1 정확도(%)다. 점선은 clean, 실선은 watermarked다. 파랑 DINOv3-B/16, 노랑 LeJEPA-S/8, 검정 RePR-78M을 비교한다.*

워터마크 실험은 모든 이미지 패치에 그 이미지를 식별하는 워터마크를 넣은 ImageNet을 쓴다. 이 데이터에서는 워터마크만 알아봐도 대조 손실이 완벽해진다. 저자들은 DINOv3, LeJEPA, RePR을 원래 ImageNet과 워터마크 버전에서 각각 학습했고, 워터마크 데이터에서 DINOv3와 LeJEPA는 워터마크만 표현하도록 붕괴한 반면 RePR은 분류에 유용한 내용을 상당 부분 유지한다. 저자들은 RePR이 대조학습과 같은 PMI 인수분해 보장을 공유하면서도 강건하다고 정리한다. **리뷰 해석.** 그림에서 RePR도 clean 곡선보다 낮고 clean 조건에서는 DINOv3가 더 높으므로, 이 실험이 보여 주는 것은 분류 성능 우위가 아니라 워터마크 지름길에 대한 강건성이다. 워터마크의 실제 모습은 [2편 A.6](/reviews/repr-latent-tokenization-technical/)에 있다.

### 6.2 스케일링 법칙 (Scaling Laws)

[![Figure 3. 인코더 학습 FLOPs와 ImageNet FID-10K·GPIC FD-DINOv2-10K의 관계, 반복 학습과 외삽선.](/assets/reviews/repr/figure3.webp)](/assets/reviews/repr/figure3.webp)

*원문 Figure 3, p.8. 가로축은 인코더 학습 FLOPs의 로그 척도다. 왼쪽은 ImageNet에서 JiT-B를 80 epochs 학습한 뒤의 FID-10K, 오른쪽은 GPIC에서 MiniT2I를 1 GPIC epoch 학습한 뒤의 FD-DINOv2-10K이며 모두 낮을수록 좋다. 원은 픽셀 예측(반복 1), 마름모는 특징 예측(반복 2)이고 색은 인코더 크기다. 실선은 관측 점의 fit, 그 너머의 파선·점선은 연장이며, 별은 기준 인코더(DINOv2, WebSSL-DINO-1B), 오른쪽의 붉은 파선은 MiniT2I 기준선이다.*

인코더의 확산 용이성을 비교하려면 생성 모델의 크기와 학습 기간을 고정해야 한다. 저자들은 Latent Forcing으로 특징을 먼저 생성하고 이를 조건으로 픽셀을 생성한다. 최종 생성이 픽셀 공간에서 이루어지므로 원문은 이 과정을 손실 없는(lossless) 생성이라 부르며, 손실 압축 토크나이저의 재구성–생성 trade-off를 피해 이득을 압축이 아닌 확산 용이성에서 찾는다. 같은 장치로 JiT 같은 픽셀 확산과 FLOP·매개변수를 맞춘 비교도 가능하다.

폭은 256–1024, 추론 인코더는 20M–306M으로 바꾸고 깊이는 고정한다. 저자들은 두 데이터셋 모두에서 학습 계산 약 두 자릿수에 걸친 거듭제곱 법칙(power law)을 보고하며, 같은 크기의 200 epoch 픽셀 모델 특징을 예측하는 반복 2가 반복 1보다 크게 낫다. ImageNet의 폭 1024 모델은 200 epochs 이후 과적합하기 시작하지만 GPIC에서는 더 큰 예산까지 개선이 이어진다.

초록의 7배·8배는 이 fit을 관측한 로그 FLOP 구간의 폭을 넘지 않는 범위까지 연장해 얻은 예측이다. A.7이 적은 값은 다음과 같다.

| 평가 | 기준 인코더의 비용·품질 | RePR 외삽 비용 | 비용 비율 |
| --- | --- | --- | --- |
| ImageNet, FID-10K | DINOv2: $1.1685129\times10^{22}$ FLOPs, 17.95 | $1.504\times10^{21}$ FLOPs | 약 1/7.77 |
| GPIC, FD-DINOv2-10K | WebSSL-DINO-1B: $9.78\times10^{21}$ FLOPs, 277.1 | $1.229\times10^{21}$ FLOPs | 약 1/7.96 |

측정된 반복 2의 최고점은 ImageNet에서 RePR-173M의 FID-10K 24.73, GPIC에서 RePR-306M의 FD 325로, 각각 기준 17.95·277.1에 아직 미치지 않는다. 원문은 fit 계수와 그 불확실성을 싣지 않는다. FD-DINOv2는 같은 Fréchet 거리를 DINOv2 특징에서 계산한 값이라 FID와 척도가 다르다([2편 A.7](/reviews/repr-latent-tokenization-technical/)).

### 6.3 계산량을 고려한 생성 (Compute Optimal Generation)

[![Figure 4. 인코더와 생성 모델의 합산 학습 FLOPs에 따른 조건부·무조건부 unguided FID-10K.](/assets/reviews/repr/figure4.webp)](/assets/reviews/repr/figure4.webp)

*원문 Figure 4, p.9. 위는 조건부 생성, 아래는 무조건부 생성이다. 가로축에는 인코더 사전학습을 포함한 총 학습 FLOPs가 들어간다. 검정은 픽셀 JiT-L, 파랑·노랑은 RePR-44M/78M과 JiT-L의 조합이다. 두 세로축은 unguided FID-10K이며 범위가 다르다.*

JiT-L 200 epochs 학습량인 $1.19\times10^{20}$ FLOPs에서 저자는 RePR 조합의 unguided FID-10K가 조건부에서 2.61, 무조건부에서 4.18 개선된다고 보고한다. 이 비교에는 지각 손실·GAN·외부 이미지 디코더가 없다. Table 3은 guidance를 쓴 별도의 50K 평가다.

| 설정 | 총 학습 FLOPs ↓ | FID-50K ↓ | IS ↑ |
| --- | --- | --- | --- |
| Pixel, JiT-L 600ep | $3.57\times10^{20}$ | 2.36 | 298.5 |
| RePR-78M + Pixel, JiT-L 440ep | $3.49\times10^{20}$ | **2.18** | **338.4** |

RePR 쪽 예산이 약 2.2% 적다. 이 예산에는 RePR 인코더 학습과 guidance에 쓰인 AutoGuidance 학습이 포함된다(A.3·A.5). 인셉션 점수(Inception Score, IS)는 생성물의 분류 확신과 클래스 다양성을 결합한 값이며 높을수록 좋다([IS 원전 §4](https://arxiv.org/pdf/1606.03498v1#page=4)). FLOPs는 self-attention을 뺀 $6ND$ 근사(매개변수 수 × 처리 데이터량 × 6)다. 저자들은 이 결과와 앞의 스케일링 법칙을 묶어, 표현 학습이 생성 모델만 키우는 것보다 효율적으로 생성 성능을 높이는 추가 스케일링 축이라고 결론짓는다. [§6.3·A.3/A.5](https://arxiv.org/pdf/2610.20533v1#page=8)

### 6.4 절제 실험 (Ablation Studies)

Table 1의 여섯 묶음은 서로 다른 설계 선택을 검사한다. 캡션은 생성 모델 학습 비용 때문에 ImageNet top-1을 프록시로 쓴다고 적지만, 실제로 (a)–(d)는 top-1 정확도(%), (e)·(f)는 FID-10K다.

| 원문 묶음 | 설정과 결과 | 해석 |
| --- | --- | --- |
| (a), 78M·100ep | $M_A$: 62.5%→58.62, 75%→58.75, 87.5%→56.21 top-1 | 문맥 $A$가 너무 적으면 불리하다. |
| (b), 78M·100ep | $M_B$: 75%→56.78, 93.75%→58.75, 98.5%→58.55 top-1 | 짧은 $B$ 문맥이 유리하지만 98.5%에서는 조금 낮아진다. |
| (c), 78M·100ep | uniform 57.63, block 58.75 top-1 | block mask가 더 좋다. |
| (d), 78M·200ep | 표현 손실 가중치 0.005→63.1, 0.01→63.6, 0.025→63.0, 0.1→49.5, 0.4→43.9 top-1 | 기본값 0.005보다 0.01이 약간 높고, 0.1 이상은 크게 나빠진다. |
| (e), 173M·총 600ep 배분 | 픽셀만 $\{600\}$→30.68, 반복 2 $\{200,400\}$→24.73, 반복 3 $\{200,200,200\}$→23.81 FID-10K | 같은 총 epochs를 특징 예측에 재배분하면 FID가 낮아진다. |
| (f), 78M·200ep | 원래 가중치 $(1-t)/t^3$: NaN; 분모 바닥값 0.05: 42.15; logit-normal $\mu=0.6$/1.2와 바닥값 0.33: 35.09/35.72 FID-10K | 안정성과 시간 편향이 핵심이다. 기본 $\mu=1.2$보다 $\mu=0.6$이 낮다. |

저자들은 마스킹 설정이 유용하지만 결정적이지 않고, 표현 손실 가중치가 아주 작을 때 최적이라는 점에서 학습의 어려움이 표현 인수분해보다 생성 쪽에 있다고 해석한다. 반복 학습은 (e)에서 성능을 크게 높이며, MAE에서 Data2Vec으로 가며 얻은 이득과 닮았다고 본다.

**리뷰 해석.** 표의 epoch 표기는 부록과 어긋난다. (e)의 30.68·24.73은 Table 7의 173M 610ep·$210+410$ep 값과 같고, (f)의 기본 행 35.72는 Table 7의 78M 210ep 값과 같다. 따라서 (e)·(f)는 Table 7과 같은 생성 평가(JiT-B 80ep, FID-10K)이고, Table 1의 $\{600\}$·$\{200,400\}$·200ep는 체크포인트마다 붙는 10ep learning-rate decay(A.3)를 뺀 명목값이다. 이렇게 읽으면 (e)의 실제 총량은 610·620ep이고 반복 3은 같은 규칙이면 630ep라, 캡션의 '총 epochs 고정'은 명목 기준이다. 차이는 약 3% 이내여서 (e)의 결론은 바뀌지 않는다. [§6.4·Table 1, p.9](https://arxiv.org/pdf/2610.20533v1#page=9)

### 6.5 GAN+LPIPS 디코딩 (GAN+LPIPS Decoding)

Table 2는 RePR을 RAE 경로에 넣는 전이 시험이다. ImageNet으로만 학습한 인코더와 DiT-XL 20ep 설정에서 MoCo-v3-B 13.84, I-JEPA-H 12.46, RePR-173M 반복 3은 **FID-50K 11.13**이다. MoCo-v3·I-JEPA 값은 RAEv2가 보고한 수치를 옮긴 것이고(캡션), RePR은 RAEv2의 절제 설정을 따라 학습했다. 저자들은 이를 작은 규모에서도 경쟁력 있는 토크나이저라는 근거로 든다.

디코더는 학습된 지각 거리(Learned Perceptual Image Patch Similarity, LPIPS)와 적대적 생성 신경망(Generative Adversarial Network, GAN) 손실을 쓴다. 저자들은 이 경로가 외부 모델 두 개에 기대며, 그중 DINOv1/8 기반 GAN 사전학습만 $4.7\times10^{20}$ FLOPs로 자신들이 학습한 어떤 모델보다 크다고 밝힌다.

## 7. 결론 (Conclusion): 어떤 기여를 가져갈 수 있는가?

저자들은 RePR이 음성 표본 없이 대조학습의 이론적 보장을 공유하면서, 기존 방법이 붕괴하는 조건에서 특징 억제에 강하다고 정리한다. 또 확산 용이성의 경험적 스케일링 법칙과, 같은 학습 FLOPs에서 RePR 기반 생성 모델이 원시 데이터의 직접 확산보다 낫다는 결과를 근거로, 이론적 기반이 강한 표현 학습이 생성 모델의 스케일링과 공존하며 그것을 개선할 수 있다고 결론짓는다. [§7, p.10](https://arxiv.org/pdf/2610.20533v1#page=10)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 이 논문에서 가장 설득력 있는 설계는 인코더 학습 FLOPs를 생성 비교에 넣고, Latent Forcing으로 별도 디코더·지각 손실·GAN을 빼서 차이를 인코더의 확산 용이성으로 돌린 것이다. 그 결과 질문이 '어떤 기성 인코더를 쓸까'에서 '인코더 학습에 계산을 얼마나 배분할까'로 바뀌고, Figure 4에서는 그려진 모든 예산에서 두 RePR 조합의 곡선이 픽셀 JiT-L 아래에 있다.

다만 근거의 정밀도는 제한적이다. Table 3의 FID-50K 차이 0.18과 Figure 4의 곡선은 오차 범위 없이 보고되고, FLOPs는 self-attention을 뺀 근사라 GPU 시간이 같은 비율로 줄어든다는 보장이 없다. GPIC 생성에는 FLAN-T5-large 텍스트 인코더가 조건 입력으로 들어가므로, 도입부의 '외부 모델 없음'이 그대로 들어맞는 것은 ImageNet 실험이다.

초록은 DINO·LeJEPA가 실패하는 경우에 RePR이 차원 붕괴(dimensional collapse)와 특징 억제를 모두 피한다고 쓰지만, 근거는 워터마크 ImageNet의 class probing 곡선(Figure 2)이다. 특징 억제는 여러 차원을 쓰면서 워터마크 같은 일부 정보만 담는 경우도 포함하므로, 표현의 유효 차원을 따로 재지 않은 이 실험은 특징 억제의 근거이고 차원 붕괴는 직접 측정되지 않았다. 분포의 비퇴화성을 다룬 [LeJEPA 해설](/reviews/lejepa-complete/)과 비교하면, 이 실험이 묻는 것은 차원 수가 아니라 어떤 정보가 남는가다.

남은 질문은 ImageNet 폭 1024 모델이 200 epochs 뒤 과적합하는 상황에서도 거듭제곱 법칙이 기준 품질까지 이어지는지, 그리고 Table 3의 우위가 $3.6\times10^{20}$ FLOPs보다 큰 예산과 더 큰 생성 모델에서도 유지되는지다.

## 출처와 읽은 범위

- 원문: *Generative Representation Learning for Scalable Latent Tokenization* · Alan Baade et al. · 최초 공개 2026-10-14 · [검토 버전](https://arxiv.org/abs/2610.20533v1) · 검토일 2026-10-23.
- 읽은 범위: v1 본문 §1–7. A.1–A.7의 알고리즘·상세 유도·원시 수치는 [2/2] 심화 편에서 다룬다.
- 참고한 외부 원전: [CPC §2.3](https://arxiv.org/pdf/1807.03748v2#page=3), [Variational Diffusion Models E.1–E.3](https://arxiv.org/pdf/2107.00630v6#page=17), [DINOv2 §4](https://arxiv.org/pdf/2304.07193v2#page=5), [IS 원전 §4](https://arxiv.org/pdf/1606.03498v1#page=4).
- 그림 출처: 대표 이미지와 본문 Figures 1–4는 저자 v1 자산(CC BY 4.0)이며, 저자 여백 trim·해상도 조정·WebP 변환을 적용했다. 내용·배치·원래 글꼴은 유지한다. [원문](https://arxiv.org/abs/2610.20533v1).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
