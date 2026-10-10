---
title: "[2/3] LeJEPA 전체 해설: 목표 분포에서 이론·알고리즘·실험까지"
description: "50쪽 논문을 문제·관련 연구·목표 분포·SIGReg·학습 구조·실험·한계 순서로 연결해 설명한다."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-10T18:16:19+09:00"
topics: ["self-supervised learning", "representation learning", "joint embedding predictive architectures", "distribution matching", "computer vision"]
visibility: "public"
lang: "ko"
translationKey: "lejepa-complete"
paperTitle: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
paperPublishedDate: "2025-11-11"
authors: ["Randall Balestriero", "Yann LeCun"]
year: "2025"
paperUrl: "https://arxiv.org/abs/2511.08544v3"
thumbnail: "/assets/reviews/lejepa/lejepa-core.svg"
thumbnailAlt: "LeJEPA의 두 제약 설명도: 한 이미지의 뷰들은 전역 뷰 평균에 정렬하고, 각 뷰에서 여러 이미지의 표현은 임의 방향으로 투영해 표준 Gaussian과 맞춘다."
---

LeJEPA는 자기지도 표현 학습에서 두 가지 질문을 함께 다룬다. <strong>후속 과제를 아직 모를 때 어떤 표현 분포를 목표로 해야 하며, 그 분포를 큰 모델에서도 안정적으로 맞출 수 있을까?</strong> 저자들은 등방성 Gaussian을 통계적 예측 문제와 연결하고, SIGReg라는 투영 기반 분포 손실을 이미지 뷰의 합의 손실에 결합한다. 이론·알고리즘·다양한 이미지 실험이 하나의 설계 논리로 이어지는 것이 강점이다. 이 글은 50쪽 논문의 전체 흐름을 설명하면서, 증명의 조건과 실험 비교의 범위를 함께 살펴본다. [§§1–6](https://arxiv.org/pdf/2511.08544v3#page=1)

**표기 안내.** 임베딩 벡터 $\mathbf z_{n,v}$, 전역 평균 $\boldsymbol\mu_n$, 전역 뷰 수 $V_g$와 손실 $\mathcal L$은 원문 표기를 따른다. 본문 계산 비용에서 $M=|\mathcal A|$는 방향 수이고 $T$는 적분 주파수 수를 뜻하는 리뷰 보조 기호다. $\mathcal L_{\mathrm{SIGReg}}$는 뷰별 SIGReg 평균의 약칭이며, 중심 손실의 $1/K$는 Algorithm 2의 feature 평균을 표시한다.

세 버전 중 전체 해설 편이다. [입문 요약](/reviews/lejepa-overview/)은 두 조건의 직관에, [기술 심층 리뷰](/reviews/lejepa-technical/)는 증명·계산 복잡도·원문 수식 검토에 집중한다.

<strong>대표 그림의 핵심.</strong> 모든 뷰가 같은 encoder와 projector를 지나지만, 아래 두 손실이 모으는 표본은 다르다. 왼쪽은 한 이미지의 뷰들, 오른쪽은 각 뷰에서 서로 다른 이미지들의 표현이다. 첫 번째 손실은 공통 내용을 보존하고, 두 번째 손실은 상수 또는 일부 방향으로의 붕괴를 막는 분포 목표를 제공한다. Gaussian은 projector 뒤의 표현 공간에 적용한다. [크게 보기](/assets/reviews/lejepa/lejepa-core.svg)

## 먼저 전체 논리의 지도를 잡기

본문은 문제 정의, 목표 분포의 이론, 분포를 맞추는 알고리즘, 통합 학습, 실험 순서로 진행된다. 긴 부록은 이론의 조건과 세부 실험을 보완한다.

| 논문의 부분 | 답하려는 질문 | 이 글에서 볼 핵심 |
| --- | --- | --- |
| §§1–2, pp. 1–4 | 뷰의 표현을 가깝게 만드는 것만으로 충분한가? | 합의와 non-degeneracy를 함께 요구 |
| §3, pp. 5–6 | 후속 예측에 어떤 분포가 유리한가? | 선형 등방성과 비선형 Fisher information 분석 |
| §4, pp. 6–11 | 고차원 Gaussian 제약을 어떻게 계산하는가? | 임의 1D 투영, 특성함수, SIGReg |
| §5, pp. 11–12 | 실제 encoder 학습 목적은 무엇인가? | 전역 뷰 중심의 합의와 뷰별 분포 손실 |
| §6, pp. 13–18 | 구조·규모·도메인을 바꿔도 쓸 수 있는가? | 민감도, 여러 backbone, 전이와 Galaxy10 |
| Appendix A–F, pp. 24–46; Figures 16–21, pp. 47–50 | 조건·증명·추가 결과는 무엇인가? | 국소 추정 가정, 수치 근사, 추가 표와 시각 결과 |

본문 명칭은 <strong>Latent-Euclidean JEPA</strong>다. 공식 저장소 README는 Lean Joint-Embedding Predictive Architecture라는 풀이도 사용한다. 여기서는 논문의 명칭과 방법을 기준으로 설명한다. JEPA는 픽셀을 직접 재구성하는 대신 표현 공간에서 관련 입력의 정보를 예측·정렬하는 계열의 관점이다. 이 논문에서는 이미지 multi-crop 합의가 주된 구현이며, 행동 조건부 미래 예측이나 제어를 검증하지 않는다. [§2.1, p. 4; 공식 저장소](https://arxiv.org/pdf/2511.08544v3#page=4), [README](https://github.com/galilai-group/lejepa)

## 문제 정의: 불변성만으로는 정보가 남지 않는다

같은 이미지에서 나온 큰 crop과 작은 crop을 같은 의미로 연결하고 싶다. 이를 위해 encoder가 만든 표현의 차이를 줄인다. 그러나 모든 입력을 같은 벡터로 보내는 해도 이 조건을 만족한다. 즉, ‘같은 이미지끼리 가깝다’는 요구에 더해 ‘다른 이미지가 구별될 수 있다’는 조건이 필요하다.

이를 두 축으로 나누면 논문을 이해하기 쉽다.

- <strong>이미지 내부:</strong> 뷰를 바꿔도 공통 내용이 유지되는가?
- <strong>이미지 사이:</strong> 전체 표현이 정보를 담을 수 있는 분포를 이루는가?

기존 방법들은 음성 표본, teacher–student 구조, stop-gradient, 중심화, 분산·공분산 항 등으로 이 균형을 만든다. LeJEPA는 분포 목표를 명시하고, 그 분포가 후속 예측기에 왜 유리할지부터 분석한다. 핵심 구성은 대칭적 뷰 합의와 분포 정규화이며 별도 predictor나 EMA teacher가 필수는 아니다. 비대칭적 예측 문제에서도 predictor가 언제나 불필요하다는 결론은 아니다. [§§2, 5](https://arxiv.org/pdf/2511.08544v3#page=3)

## 관련 연구와의 차이: 무엇을 처음 했다고 말할 수 있나

이번 비교에서는 주요 원전 5개를 실제로 읽었다. 아래는 방법 절을 중심으로 한 비교이며 각 원전 전체를 재검토한 것은 아니다.

| 연구 | 표현이 붕괴하지 않도록 하는 방식 | LeJEPA와 연결되는 점·차이 |
| --- | --- | --- |
| [VICReg](https://arxiv.org/pdf/2105.04906v3#page=3), §§3.1–3.3, pp. 3–4 | 뷰 합의, 각 차원의 표준편차 하한, 공분산의 비대각 항 축소 | 이미 stop-gradient·teacher·음성 표본 없이 작동하는 대칭적 방식이다. LeJEPA는 낮은 차수 통계를 넘어 투영 분포 전체를 목표로 삼는다. VICReg 원래 손실을 정확한 평균 0·공분산 I 제약과 동일시하지 않는다. |
| [DINO](https://arxiv.org/pdf/2104.14294v2#page=2), §3, pp. 2–3 | EMA teacher, 중심화와 sharpening, teacher–student 출력 확률의 cross-entropy | teacher 출력의 학습 규칙 대신 명시적인 Gaussian 목표를 둔다. 원래 DINO도 convolution 모델을 다루므로 구조 다양성 자체가 LeJEPA에서 처음 생긴 것은 아니다. |
| [I-JEPA](https://arxiv.org/pdf/2301.08243v3#page=3), §3, pp. 3–4 | 가려진 context에서 target block의 표현을 predictor로 예측; EMA target encoder | 정보가 다른 두 입력의 예측과, LeJEPA의 대칭적 crop 합의는 과제가 다르다. Table 2의 성능과 함께 학습 구조·계산 조건을 봐야 한다. |
| [Gaussian Embeddings](https://arxiv.org/pdf/2510.05949v1#page=1), pp. 1, 3–4 | Gaussian 밀도와 표현의 변환·Jacobian 부피를 다루는 관점 | Gaussian을 다룬다는 공통점은 있지만, LeJEPA의 후속 추정 bias·variance 분석 및 sliced CF 손실과 동일한 논증은 아니다. |
| [Gretton et al., MMD](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf), PDF pp. 1, 3–4 | kernel 평균 임베딩의 차이로 분포 비교 | Gaussian 가중 특성함수 비교와 kernel discrepancy의 연결을 이해하는 배경이다. 표준 pairwise 계산은 제곱 비용이지만, 기존 MMD에도 선형 시간 추정기가 있으므로 ‘최초의 선형 분포 비교’라고 할 수는 없다. |

이 비교를 바탕으로 보면 기여는 각 재료 하나의 발명보다 <strong>후속 예측 이론 → 목표 분포 → 실제 계산 가능한 손실 → 간결한 학습 구조</strong>를 묶는 데 있다.

## 왜 등방성이고, 왜 Gaussian인가

선형 예측을 먼저 생각해 보자. 표현 행렬을 $\mathbf Z$, 후속 정답을 $\mathbf y=\mathbf Z\boldsymbol\beta+\boldsymbol\varepsilon$라고 하자. $\boldsymbol\beta$는 풀려는 과제의 방향이고 $\boldsymbol\varepsilon$는 잡음이다. 어떤 표현 방향의 변화가 매우 작으면 그 방향에 관한 정보를 잡음과 구별하기 어렵다.

원문은 ridge regression의 shrinkage bias와 OLS의 계수 분산을 분석한다. full-rank·고정 설계·동일 잡음 분산 등의 조건에서 OLS 분산의 trace는 $\sigma^2\sum_{k=1}^K1/\lambda_k$다. $\lambda_k$는 $\mathbf Z^\top\mathbf Z$의 고윳값이다. 전체 에너지가 같다면 한 방향은 작고 다른 방향은 큰 분포보다, 모든 고윳값이 같은 분포가 이 양을 줄인다. 예를 들어 고윳값 $(1,1)$의 역수 합은 2지만 $(0.2,1.8)$은 약 5.56이다. 이 숫자는 리뷰 작성자의 작은 계산 예제다. [§3.1; Appendix B.1–B.2, pp. 26–27](https://arxiv.org/pdf/2511.08544v3#page=26)

이 선형 결과가 정하는 것은 <strong>공분산의 등방성</strong>이다. 정규분포만이 이 공분산을 갖는 것은 아니다. 그래서 저자들은 이웃 평균과 Nadaraya–Watson kernel regression이라는 비선형 후속 예측기를 살펴본다. 이들은 가까운 표현들의 정답을 평균해 새 표현의 정답을 추정한다.

그때 작은 이웃이나 bandwidth에서 발생하는 bias에는 정답 함수의 변화와 표현 밀도의 변화가 함께 들어간다. 밀도 $p$의 score는 $\nabla\log p$이며, 이를 제곱해 평균한 $J(p)=\mathbb E\|\nabla\log p(X)\|_2^2$가 Fisher information이다. 과제 gradient에 대한 등방성 사전 가정 등을 사용하면 특정 bias 성분이 $J(p)$와 연결된다. kernel 분석은 이 값을 포함하는 최악 경우 bias 상한을 제어한다. [§3.2; Appendix A, B.3–B.7](https://arxiv.org/pdf/2511.08544v3#page=24)

적절한 smoothness·꼬리 조건과 고정된 공분산 아래에서 Gaussian은 Fisher information을 최소화한다. 공분산 trace까지 고정하면 등방성 Gaussian이 선택된다. 그러나 국소 근사, 과제 사전 가정, query와 train의 분포 관계를 사용한 결론이다. 모든 비선형 예측기의 정확한 전체 위험이 무조건 유일하게 최소화된다고 바꾸면 논문보다 강한 주장이 된다. 자세한 조건은 [기술 심층 리뷰](/reviews/lejepa-technical/)에 정리했다.

## 분산과 공분산만 보면 놓치는 분포

![LeJEPA Figure 5: X자 모양의 비Gaussian 결합 분포는 좌표별 Gaussian 모양과 달리 일부 투영에서 큰 차이를 보인다.](/assets/reviews/lejepa/paper-figure-5.webp)

[그림 확대](/assets/reviews/lejepa/paper-figure-5.webp)

<strong>Figure 5 해설.</strong> 왼쪽 히스토그램에서 좌표 $x_1$, $x_2$의 주변 분포는 비슷한 종 모양이다. 그런데 두 번째 패널의 결합 분포는 X자 형태라 다변량 Gaussian이 아니다. 세 번째 패널은 가운데 색 화살표 방향으로 투영했을 때 밀도가 달라지는 모습을 보여준다. 네 번째 패널은 방향에 따른 여러 정규화·검정 통계의 반응을 비교한다. 각 곡선은 실험적으로 비교한 통계이며 모두 동일한 척도의 확률이나 실제 downstream 정확도가 아니다. 핵심은 <strong>좋은 주변 통계만으로 결합 분포를 확인할 수 없다는 것</strong>이다. 출처: Balestriero·LeCun, [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). 전체 패널을 crop·무손실 WebP 변환했다.

유한 개의 moment를 맞추는 것에도 한계가 있다. 평균·분산·왜도·첨도 등 몇 개의 통계가 같아도 서로 다른 분포가 존재한다. 높은 차수까지 늘리면 꼬리 표본이 계산과 gradient에 큰 영향을 줄 수 있다. 이 논문은 그 대안으로 특성함수를 사용한다. [§4.2, Theorem 3, pp. 7–9](https://arxiv.org/pdf/2511.08544v3#page=7)

## SIGReg: 분포의 지문을 여러 방향에서 맞추기

벡터 $\mathbf z$를 길이 1인 방향 $\mathbf a$에 투영한 스칼라를 $z=\mathbf a^\top\mathbf z$라 하자. SIGReg는 각 방향에서 empirical characteristic function, 즉 표본 특성함수를 계산한다.

$$
\begin{aligned}
\hat\varphi_{\mathbf a}(t)&=\frac1N\sum_{n=1}^N\exp\!\left(it\mathbf a^\top\mathbf z_n\right),\\
\varphi_{\mathcal N}(t)&=\exp(-t^2/2).
\end{aligned}
$$

$t$는 주파수이며 $i$는 허수 단위다. 구현에서는 복소수 exponential을 cosine 평균과 sine 평균으로 나눌 수 있다. Gaussian의 지문은 닫힌 식으로 알려져 있으므로 목표 표본을 매번 생성할 필요가 없다. 주파수별 차이의 제곱에 가중치를 곱해 적분하고, 여러 방향의 값을 평균한다. 원문의 Epps–Pulley 통계와 Algorithm 1은 batch 크기 $N$을 곱한 스케일을 사용한다. [§4.3, Definition 2; Algorithm 1, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10)

![LeJEPA Figure 2: 고차원 표현의 투영 밀도와 목표 Gaussian의 차이.](/assets/reviews/lejepa/paper-figure-2.webp)

[그림 확대](/assets/reviews/lejepa/paper-figure-2.webp)

<strong>Figure 2 해설.</strong> 왼쪽 입력 분포를 가운데 표현 공간으로 옮긴 뒤, 색깔별 방향의 1D 분포를 오른쪽 검은 목표 곡선과 비교한다. 오른쪽 가로축은 투영 좌표이고 회색은 차이의 개념적 표현이다. 이 그림의 곡선은 설명용 밀도이며 ImageNet에서 측정한 결과가 아니다. 출처: Balestriero·LeCun, [Figure 2, p. 3](https://arxiv.org/pdf/2511.08544v3#page=3), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). 전체 패널 crop·무손실 WebP 변환.

Cramér–Wold 정리는 모든 방향에서의 분포 일치와 원래 벡터 분포의 일치를 연결한다. 실제 학습은 유한 방향과 주파수 grid를 사용한다. 그러므로 SIGReg를 작게 만들었다는 사실만으로 정확한 다변량 정규성을 인증할 수는 없다. 반복 중 방향을 다시 뽑는 것은 고정된 일부 방향에만 맞추는 현상을 줄이는 설계다. [§4.1, §4.4; Appendix B.8–B.10](https://arxiv.org/pdf/2511.08544v3#page=6)

주요 계산은 matrix projection과 sine·cosine 평균이다. $N$개 표본, $K$차원, $M$개 방향, $T$개 주파수라면 dense 구현 비용은 대략 $\mathcal O(NKM + NMT)$다. $K,M,T$를 고정하면 표본 수에 선형이고 모든 표본 쌍을 만들 필요가 없다. 방향 수를 차원과 함께 늘리는 설정이나 encoder 전체의 계산까지 모두 선형이라고 말하는 것은 별개다. [§4.3, p. 9; Algorithm 1](https://arxiv.org/pdf/2511.08544v3#page=9)

## 통합 학습: 두 손실이 묶는 단위를 확인하기

$\mathbf z_{n,v}$를 이미지 $n$, 뷰 $v$의 projected embedding이라 하자. 큰 crop인 전역 뷰가 $V_g$개, 전체 뷰가 $V$개다. 원문의 Algorithm 2는 전역 뷰들의 평균 $\boldsymbol\mu_n$을 중심으로 합의 손실을 계산한다.

$$
\begin{aligned}
\boldsymbol\mu_n&=\frac1{V_g}\sum_{g=1}^{V_g}\mathbf z_{n,g},\\
\mathcal L_{\mathrm{pred}}&=\frac1{NVK}\sum_{n=1}^N\sum_{v=1}^V\|\mathbf z_{n,v}-\boldsymbol\mu_n\|_2^2,\\
\mathcal L_{\mathrm{SIGReg}}&=\frac1V\sum_{v=1}^V\operatorname{SIGReg}\!\left(\{\mathbf z_{n,v}\}_{n=1}^N\right),\\
\mathcal L_{\mathrm{LeJEPA}}&=(1-\lambda)\mathcal L_{\mathrm{pred}}+\lambda\mathcal L_{\mathrm{SIGReg}}.
\end{aligned}
$$

$K$는 projected embedding 차원이다. 위 식은 Algorithm 2의 `.square().mean()`에 맞춰 feature 차원도 평균한다. $\boldsymbol\mu_n$을 만드는 경로에서도 gradient가 흐른다. 별도의 frozen teacher 목표가 아니다. 합의 손실을 볼 때는 같은 $n$의 여러 뷰를 모으고, SIGReg를 볼 때는 같은 $v$의 여러 이미지 $n$을 모은다는 차이가 중요하다. encoder뿐 아니라 projector도 남아 있다. 평가용 backbone 특징과 분포 제약용 projected embedding을 구분한다. Algorithm 2의 non-ViT 설정은 global과 all views를 같게 두므로 그때는 모든 사용 뷰의 평균이 중심이다. [§5, Algorithm 2, pp. 11–12](https://arxiv.org/pdf/2511.08544v3#page=11)

원문 Eq. 5–7은 쌍별 거리 평균을 중심 거리 평균으로 바꾸면서 동치라고 서술한다. 직접 전개하면 <strong>전역 뷰의 중심 주위 분산 항이 추가로 남는다</strong>. 이 리뷰에서는 Algorithm 2가 명시하는 중심 기반 목적을 기준으로 설명한다. 누락 항의 계산 예제와 부록 식 대조는 [기술 심층 리뷰](/reviews/lejepa-technical/)에 있다. 이 대수적 불일치만으로 실험 결과가 잘못됐다고 판정하지 않는다. [§5, p. 12; Appendix B.6, p. 34](https://arxiv.org/pdf/2511.08544v3#page=34)

## 구현 설정: 간결한 목적과 남아 있는 선택들

§6.1의 기본 설정은 $\lambda =0.05$, 전역 뷰 2개와 local 뷰 8개, batch 최소 128, 방향 1,024개, $[-5,5]$의 17개 적분점이다. 한편 p. 14의 Experiment Details 1은 8개 뷰(전역 2, local 6)를 쓴다. 실험별 설정을 읽어야 하며 모든 표가 하나의 구성이라고 가정하면 안 된다. [§6.1; Experiment Details 1, pp. 13–14](https://arxiv.org/pdf/2511.08544v3#page=13)

학습에서는 AdamW, learning-rate warmup과 cosine schedule을 사용한다. 일부 모델 실험은 학습률 $5\times 10^{-3}/5\times 10^{-4}$와 weight decay $0.1/0.01/10^{-5}$를 cross-validation한다. 선택적인 SWA(weight averaging) 결과도 Appendix Table 4에 있다. 따라서 ‘heuristics 없이’라는 제목은 핵심 anti-collapse 구조의 단순화에 초점을 두고 읽어야 한다. 모든 optimizer 설정이나 schedule까지 제거했다는 뜻은 아니다. [p. 14; Table 4, p. 42](https://arxiv.org/pdf/2511.08544v3#page=14)

이 리뷰에서 확인한 [공식 코드 스냅샷](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052)의 MINIMAL 예시는 17점 $[0,3]$ grid와 양의 구간 가중치 doubling, 256개 방향, 모든 뷰의 평균 중심을 사용한다. 논문 Algorithm 1 및 전역 뷰 중심 설정과 그대로 같은 구성이 아니다. 코드 설명을 읽었고 실행은 하지 않았다. 재현할 때는 버전과 설정을 먼저 고정해야 한다.

## 실험 1: 민감도와 구조 다양성

Table 1은 ImageNet-1K의 ViT-L/14를 100 epoch 학습하고 frozen linear top-1 정확도(%)를 비교한다. 아래는 각 ablation의 일부 행이며 서로 다른 설정을 한 실험처럼 합치지 않았다. [Table 1, p. 13](https://arxiv.org/pdf/2511.08544v3#page=13)

| 변경 축 | 설정 | Top-1 정확도 |
| --- | --- | --- |
| batch | 128 / 256 / 512 / 1024 | 72.20 / 74.15 / 74.72 / 74.07 |
| 전역 뷰 수, 전체 뷰는 4 | 1 / 2 | 53.06 / 72.26 |

batch를 줄여도 학습이 가능하다는 것은 의미가 있다. 그러나 성능이 완전히 같지는 않으며 전역 뷰가 하나인 경우 하락도 크다. 적분 범위를 바꾸는 행들에서도 약 2%p 차이가 있다. 이 표의 메시지는 ‘모든 설정에 무감하다’가 아니라 ‘몇 가지 중요한 축에서 큰 모델을 비교적 넓은 설정으로 학습했다’에 가깝다.

별도의 ImageNet-100/ResNet-50 실험에서 Figure 8·Table 7은 뷰 수와 $\lambda$를 함께 바꾼다. 예를 들어 2뷰/λ=0.01은 83.49%, 4뷰/λ=0.02는 84.68%, 8뷰/λ=0.05는 84.32%다. 서로 다른 뷰 수의 최고점이 같은 λ에 있지는 않는다. 기본값은 유용한 출발점이며 모든 설정의 유일한 최적값이라는 뜻은 아니다. [Figure 8, p. 13; Table 7, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

Figure 9는 ImageNet의 10개 클래스 부분집합에서 20M 미만 모델 약 50개, 8개 계열의 결과를 보여준다. 정확도는 약 91.5–95% 범위다. 논문 전체의 60개 이상 모델 실험이나 Figure 1의 1.8B 모델 학습 곡선을, 모든 모델의 전체 ImageNet·전이 과제 검증으로 읽으면 안 된다. 이 결과는 주로 <strong>여러 architecture에서 목적이 작동한다는 범위 확장</strong>을 뒷받침한다. [Figures 1, 9; §6.1, p. 14](https://arxiv.org/pdf/2511.08544v3#page=14)

## 실험 2: 손실이 표현 품질의 단서가 되는가

![LeJEPA Figure 10: SIGReg와 prediction 손실 평면에서 색으로 표시한 분류 정확도.](/assets/reviews/lejepa/paper-figure-10.webp)

[그림 확대](/assets/reviews/lejepa/paper-figure-10.webp)

<strong>Figure 10 해설.</strong> 세 패널은 ResNet-50/Galaxy10, ResNet-50/ImageNet-10, ViT-B/14/ImageNet-1K다. 가로축은 SIGReg, 세로축은 prediction 손실이며 둘 다 log scale이다. 색막대는 정확도이고 빨강이 높고 파랑이 낮다. 대체로 왼쪽 아래의 두 손실이 작은 영역에 높은 정확도가 모인다. 하지만 한 손실만 보면 부족하고, 점과 곡선의 관계는 측정한 설정들의 경향이지 보편 법칙이 아니다. 출처: Balestriero·LeCun, [Figure 10, p. 15](https://arxiv.org/pdf/2511.08544v3#page=15), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). 전체 패널 crop·무손실 WebP 변환.

Figure 11은 손실과 probe 정확도의 순위 관계를 분석한다. 원문이 보고하는 Spearman 값의 크기는 대략 0.60–0.90이고 $\lambda ^{0}·^{4}$로 scaling한 경우 약 0.93–0.99다. 그래프상 손실은 줄고 정확도는 오르는데 값은 양수로 표기돼 있으므로, 이 글에서는 원문의 signed 정의가 명확하지 않은 상태에서 ‘양의 상관’이라고 단정하지 않는다. <strong>측정된 조건에서 강한 역방향 단조 관계를 보였다</strong>는 정도가 안전하다. [§6.2, Figure 11, p. 15](https://arxiv.org/pdf/2511.08544v3#page=15)

이것이 새로운 도메인에서 라벨 없이 최적 모델을 반드시 고를 수 있다는 증명은 아니다. scaling 지수는 라벨을 사용한 관측 결과와 연결해 평가됐다. 후속 검증은 기존 평가에서 지수를 정한 뒤 새 도메인의 라벨을 가린 상태에서 checkpoint를 선택하는 방식이어야 한다.

## 실험 3: 전이 표에서 평균과 개별 과제 함께 읽기

Table 2의 8개 전이 과제는 DTD, Aircraft, Cars, CIFAR-10/100, Flowers102, Food101, Pets다. 다음은 ImageNet-1K 사전학습 블록의 평균 정확도(%)다. [Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16)

| 사전학습 모델 | 모델 규모 | Epoch | 클래스당 1개 라벨 | 10개 라벨 | 전체 라벨 |
| --- | --- | --- | --- | --- | --- |
| I-JEPA | 632M | 300 | 30.20 | 60.51 | 78.50 |
| I-JEPA + STOP | 632M | 300 | 32.05 | 62.92 | 80.70 |
| LeJEPA ViT-L | 304M | 100 | 29.55 | 60.95 | 79.48 |

LeJEPA는 더 작은 모델·더 적은 epoch으로 plain I-JEPA와 비슷하거나 일부 평균에서 높은 결과를 보였다. 그러나 1-shot 평균에서는 조금 낮고, +STOP 비교에는 세 라벨 구간 모두 낮다. ‘I-JEPA 계열을 전부 이겼다’는 요약은 맞지 않는다.

개별 과제를 보면 강점의 위치가 더 선명하다. 10-shot에서 DTD는 64.72% 대 57.68%, Flowers102는 92.53% 대 88.24%, Food101은 50.90% 대 43.97%로 LeJEPA가 plain I-JEPA보다 높다. Pets에서는 77.00% 대 83.23%로 낮다. 전체 라벨에서는 CIFAR-10/100도 LeJEPA 96.50/83.71% 대 I-JEPA 97.54/86.42%로 낮다. 공통 평균 하나만으로 어느 도메인에 유리한지 판단하지 않아야 한다.

100 대 300 epoch는 이미지 반복 횟수의 비교다. backbone 규모, crop 수, 해상도와 연산이 다르므로 총 FLOPs·wall-clock 비용이 정확히 1/3이라는 뜻은 아니다. 같은 표의 ImageNet-22K 900-epoch 블록은 별도 사전학습 조건으로 구분한다.

## 실험 4: 작은 전문 도메인에서 직접 학습하기

![LeJEPA Figure 12: Galaxy10의 라벨 수에 따른 full finetuning과 frozen backbone 비교.](/assets/reviews/lejepa/paper-figure-12.webp)

[그림 확대](/assets/reviews/lejepa/paper-figure-12.webp)

<strong>Figure 12 해설.</strong> 가로축은 후속 과제의 클래스당 라벨 수, 세로축은 정확도(%). 왼쪽 full finetuning은 backbone까지 수정하고, 오른쪽 frozen backbone은 특징 추출기를 고정한다. 주황·갈색 모델들은 Galaxy10에서 400 epoch 자기지도학습한 LeJEPA, 파란색 모델들은 일반 이미지로 대규모 사전학습한 DINOv2/v3다. 작은 전문 데이터로 도메인에 맞게 학습하는 가능성을 보여주지만, 모델 구조·사전학습 분포·학습량을 맞춘 순수 손실 비교는 아니다. 원문은 3개 seed를 보고하며 오차막대가 CI인지 표준편차인지 확정하지 않는다. 출처: Balestriero·LeCun, [Figure 12, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). 전체 패널 crop·무손실 WebP 변환.

Galaxy10은 10종 은하 분류 데이터이고 train 수는 Appendix Table 5 기준 11,008장이다. Table 3에서 전체 라벨 frozen probe는 LeJEPA ResNet-34 78.17%, ConvNeXt Nano 76.52%, DINOv2 Small 67.62%, DINOv3 Small 71.38%다. full finetuning은 각각 83.28%, 82.72%, 78.34%, 81.60%다. <strong>모델을 고정해 읽는 표현 품질과, 모델 전체를 조정한 최종 성능을 구분하면 작은 도메인 학습의 강점이 더 명확해진다.</strong> [Table 3, p. 42](https://arxiv.org/pdf/2511.08544v3#page=42)

1-shot frozen에서는 LeJEPA LeViT가 25.85%, DINOv3 Small이 30.17%여서 모든 LeJEPA 구조가 우위인 것은 아니다. Appendix Table 5의 다른 비교에서도 Flowers102는 LeJEPA ResNeXt 82.19% 대 I-JEPA 85.76%로 낮다. 그 표의 Galaxy10 값과 Table 3 값은 서로 달라 같은 프로토콜처럼 합치지 않았다. Table 5의 I-JEPA 행 이름과 사전학습 데이터 칸도 상충해 그 세부 조건은 미확인으로 남긴다. [Tables 3, 5, pp. 42–43](https://arxiv.org/pdf/2511.08544v3#page=43)

## 추가 결과: 이미지 구조와 계산 비용

§6.4는 ImageNet-1K 온라인 평가로 ViT-L 77.1%, ConvNeXt-H 78.5%를 보고한다. 온라인 probe는 Table 1의 frozen post-training probe, Table 2의 여러 전이 과제 평균과 다른 결과다. 초록의 ViT-H 79%도 같은 상세 설정 표에 직접 대응시켜 비교하지 않는다. [§6.4, p. 17](https://arxiv.org/pdf/2511.08544v3#page=17)

Figure 13의 attention 기반 영역과 Figure 14의 feature PCA 색상은 의미 있는 이미지 구조가 나타나는 정성적 결과다. attention을 threshold한 시각화는 segmentation benchmark의 IoU가 아니고, 이미지마다 별도로 PCA한 RGB 색은 모든 이미지에서 같은 개념을 뜻하는 공통 좌표가 아니다. [§6.5, pp. 17–18](https://arxiv.org/pdf/2511.08544v3#page=17)

보충 Figures 16–21은 분포 검정의 비교, 회귀 계수와 정규화의 영향, 손실·정확도 관계, 적분 근사의 추가 결과다. Appendix D는 sphere 방향의 생성, E는 Shapiro–Wilk 계열의 순서 통계, F는 여러 다변량 정규성 통계를 보완한다. 이들은 SIGReg 설계의 비교 배경이지 모두 학습된 이미지 특징의 시각화는 아니다. Figure 17의 caption은 anisotropic 쪽 분산이 낮다고 적어 본문의 등방성 논지와 상충한다. 해당 caption의 조건·표기 문제는 해결되지 않아 별도 성능 근거로 쓰지 않는다. [pp. 43–50](https://arxiv.org/pdf/2511.08544v3#page=43)

Appendix Table 6은 V100 SXM2 16GB에서 SIGReg 계산 시간을 10회 측정한다. $N=512, M=512, T=16$은 $0.465236\pm 0.011642 \,\mathrm{ms}$, $N=8192$는 $6.188304\pm 0.007226 \,\mathrm{ms}$다. §4.4는 forward–backward 측정이라고 설명한다. 이는 손실 계산의 측정이며 encoder 전체 학습 시간으로 해석하면 안 된다. 표에는 $K$와 각 단계의 시간이 분리돼 있지 않으므로 다른 시스템의 총 비용을 예측하는 근거로 확대하지 않는다. [Table 6, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

## 증명과 결과가 연결되는 범위

이론은 등방성 Gaussian을 목표로 선택하는 이유를 제공한다. SIGReg의 분석은 그 목표를 계산할 때 특성함수의 존재, 표본 gradient의 제어, 표본 크기에 따른 bias 등의 장점을 설명한다. 실험은 유한 방향·유한 grid와 실제 backbone에서도 표현을 학습할 수 있음을 보여준다.

이 세 단계를 하나의 보장으로 합쳐서는 안 된다. smooth density에 대한 population 명제와 유한 batch·유한 projection 학습은 다르다. bounded sample gradient가 encoder gradient 전체나 SGD 수렴을 보장하지 않는다. 분포 matching을 잘하는 것과 모든 downstream 과제에서 최고 정확도를 얻는 것도 다르다. [§§3–4, Theorems 4–6; Appendix B](https://arxiv.org/pdf/2511.08544v3#page=24)

이 리뷰의 판단은 LeJEPA가 <strong>재현할 가치가 있는 간결한 이미지 SSL 기준선</strong>이라는 것이다. 특히 대규모 일반 모델의 전이가 약한 전문 도메인에서, 작은 backbone을 직접 학습하는 실험이 흥미롭다. 다만 데이터 augmentation이 정말 보존해야 하는 의미인지, 계산 예산을 맞췄을 때 어떤 대안보다 유리한지, 실제 projected embedding이 보지 않은 방향·주파수에서도 목표에 가까운지 확인해야 한다.

## 다음 검증을 어떻게 설계할까

첫째, 같은 backbone·crop·입력 해상도·optimizer·총 연산 예산으로 VICReg 또는 I-JEPA 계열과 비교한다. Top-1뿐 아니라 label 수, seed별 변동과 실제 시간·메모리도 기록한다.

둘째, 학습에 쓰지 않은 투영 방향과 더 촘촘한 주파수 grid로 분포 오차를 측정한다. 낮은 training SIGReg가 분포 전체에 대한 좋은 근사인지 별도로 확인한다.

셋째, loss scaling이나 checkpoint 선택 규칙은 기존 도메인에서 고정한 뒤 새 도메인의 라벨을 가린 상태에서 평가한다. 사후 상관을 실제 선택 성능으로 바꾸는 검사다. 이 셋은 리뷰가 제안하는 후속 작업이며 이번에 실행한 실험은 아니다.

## 출처와 읽은 범위

- 원문: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, Randall Balestriero·Yann LeCun, [arXiv 2511.08544](https://arxiv.org/abs/2511.08544), 최초 제출 2025-11-11. 검토한 v3는 2025-11-14, 50쪽이다.
- 첫 번째 읽기: pp. 1–3, 17–18과 전체 절 구성에서 문제·컨셉·주장 흐름을 파악했다. 두 번째 읽기: 본문 pp. 1–18, Appendix A–F pp. 24–46, 보충 Figures 16–21 pp. 47–50을 검토했다. 참고문헌 pp. 18–23은 관련 원전을 선택 확인했다.
- 관련 연구 읽기: VICReg v3 pp. 3–4, DINO v2 pp. 2–3, I-JEPA v3 pp. 3–4, Gaussian Embeddings v1 pp. 1, 3–4, Gretton et al. 2012 PDF pp. 1, 3–4(인쇄 pp. 723, 725–726). 각 방법·정의 절의 비교이며 모든 관련 논문의 전문 분석을 뜻하지 않는다.
- 공식 코드: [commit c293d291](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052)의 README·MINIMAL을 읽었다. 실행은 미실행이다.
- 대표 개념도: 직접 제작, §§4–5·Algorithms 1–2에 근거, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 기본 글꼴 300, 강조 600. 원문 그림은 각 캡션에 출처와 CC BY-SA 4.0을 기록했다. 전체 그림 패널 crop·PDF rasterization·무손실 WebP 변환을 적용했고 데이터는 편집하지 않았다.
- <strong>실행:</strong> 리뷰 작성자의 중심 손실 항등식, 고정 1D grid의 alias 예제, OLS 역고윳값 합, ECF 표본 bias 계산. 논문 재현이 아니며 상세는 [기술 심층 리뷰](/reviews/lejepa-technical/).
- <strong>미실행:</strong> 공식 코드 실행, GPU 학습, frozen probe·finetuning 재현, seed 재현, DDP 검증, 논문 속도·메모리 재측정, 전체 증명의 형식 검증. 원문의 상충하는 조건과 오차막대 정의는 위에 미확인으로 남겼다.
