---
title: "[3/3] LeJEPA 기술 심층 리뷰: 증명 조건, SIGReg와 수식 검토"
description: "Gaussian 최소성, 투영 식별, ECF gradient·표본 bias·계산 비용을 분석하고 원문 거리 손실의 누락 항을 확인한다."
date: "2026-10-09"
publishedAt: "2026-10-09T23:54:43+09:00"
updatedAt: "2026-10-09T23:59:47+09:00"
topics: ["self-supervised learning", "representation learning", "joint embedding predictive architectures", "distribution matching", "computer vision"]
visibility: "public"
lang: "ko"
translationKey: "lejepa-technical"
paperTitle: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
paperPublishedDate: "2025-11-11"
authors: ["Randall Balestriero", "Yann LeCun"]
year: "2025"
paperUrl: "https://arxiv.org/abs/2511.08544v3"
thumbnail: "/assets/reviews/lejepa/lejepa-core.svg"
thumbnailAlt: "LeJEPA의 두 제약 설명도: 한 이미지의 뷰들은 전역 뷰 평균에 정렬하고, 각 뷰에서 여러 이미지의 표현은 임의 방향으로 투영해 표준 Gaussian과 맞춘다."
---

LeJEPA의 ‘provable’이라는 표현을 평가하려면 세 층위를 나눠 읽어야 한다. <strong>어떤 조건에서 Gaussian이 후속 추정에 유리한지, 어떤 조건에서 투영 분포의 일치가 원래 분포의 일치를 뜻하는지, 유한 batch와 수치 grid의 손실이 그 목표를 얼마나 근사하는지</strong>다. 이 글은 그 연결을 직접 전개하고, 원문 수식에서 확인한 불일치와 리뷰 작성자가 실행한 작은 검사를 기록한다. 이론의 가치와 실제 학습에 필요한 검증을 함께 보는 것이 목적이다. [§§3–5; Appendix A–B](https://arxiv.org/pdf/2511.08544v3#page=24)

세 버전 중 기술 심층 편이다. 문제의 직관은 [입문 요약](/reviews/lejepa-overview/), 전체 관련 연구·실험 지도는 [전체 해설](/reviews/lejepa-complete/)에서 설명한다.

<strong>대표 그림의 계산 단위.</strong> 한 이미지의 뷰들은 전역 뷰 평균에 정렬한다. 분포 손실은 각 뷰에서 여러 이미지의 projected embedding에 적용한다. Gaussian 목표를 backbone의 모든 내부 층이나 한 이미지의 crop 집합에 적용한다고 읽으면 안 된다. 두 경로 모두 gradient를 받고, 핵심 구조에는 stop-gradient가 없다. [개념도 확대](/assets/reviews/lejepa/lejepa-core.svg)

## 표기와 분석의 층위를 먼저 고정하기

`N`은 batch의 서로 다른 이미지 수, `V`는 이미지당 뷰 수, `Vg`는 전역 뷰 수, `K`는 projected embedding 차원, `M`은 투영 방향 수, `T`는 적분 주파수 수다. `zn,v`는 encoder와 projector를 거친 벡터다.

| 층위 | 주장의 대상 | 실제 학습으로 옮길 때 추가로 필요한 것 |
| --- | --- | --- |
| 후속 추정 이론 | 고정 설계의 선형 회귀, smooth density의 국소·kernel 예측 | 실제 표현·과제·분포가 가정을 만족하는지 |
| 분포 식별 이론 | 모든 방향과 모든 주파수, 또는 정해진 증가·검정 조건 | 유한 방향·grid가 보지 못하는 차이 |
| 학습 손실 | finite batch의 ECF discrepancy 평균 | optimizer, encoder Jacobian, 수치 오차와 일반화 |
| 실험 | 특정 데이터·모델·평가 protocol | 같은 예산의 재현, 새 도메인과 seed 변동 |

아래 수식은 읽기 편하게 다시 표기한 것이며 기호가 원문과 일부 다르다. 직접 계산한 예제는 원문의 보고 결과와 구분했다.

## 선형 결과: 공분산을 둥글게 만드는 이유

고정된 feature 행렬 `X`와 정답 `y=Xβ+ε`를 생각하자. `X`는 full column rank이고 잡음은 조건부 평균 0, 공분산 `σ²I`라고 가정한다. `G=XᵀX`, ridge 계수를 `η>0`라 하면 다음이 성립한다. [Lemma 1; Appendix B.1, p. 26](https://arxiv.org/pdf/2511.08544v3#page=26)

```text
β̂ridge = (G + ηI)⁻¹Xᵀy
E[β̂ridge | X] − β = −η(G + ηI)⁻¹β
```

`G`의 고윳값이 작은 방향에서는 `η/(λk+η)`만큼 강하게 shrink된다. 같은 전체 에너지를 가진 등방성 설계와 비교할 때, 약한 방향에 놓인 불리한 과제가 존재한다는 논증이다. 어떤 특정 과제에서도 등방성 표현이 더 좋다는 전칭 명제가 아니다.

OLS, 즉 `η=0`에서는 다음 계수 분산을 얻는다. [Lemma 2; Appendix B.2, p. 27](https://arxiv.org/pdf/2511.08544v3#page=27)

```text
Cov(β̂OLS | X) = σ²G⁻¹
tr Cov = σ² Σk 1/λk
Σk λk = c > 0을 고정하면 Σk 1/λk ≥ K²/c
등호: 모든 λk = c/K
```

마지막 부등식은 역수 함수의 볼록성 또는 Cauchy–Schwarz로 확인한다. 리뷰의 2차원 예제에서 고윳값 `(1,1)`은 역수 합 2, `(0.2,1.8)`은 약 5.5556이다. 같은 trace 2라도 작은 고윳값이 비용을 키운다.

이 결과는 feature 공간의 <strong>등방성 공분산</strong>에 관한 것이다. Gaussian joint density의 유일성, 임의 test 분포에서의 prediction risk, 학습된 encoder의 성능까지 자동으로 따라오지는 않는다.

## 비선형 결과: Fisher information은 어느 항을 제어하는가

이웃 평균은 query 주변의 표본 정답을 평균하고, Nadaraya–Watson regression은 거리 kernel로 가중 평균한다. 작은 radius·bandwidth에서 정답 함수 `m(x)`의 곡률과 표현 밀도 `p(x)`의 변화가 bias에 함께 나타난다. 개념적으로 다음 두 종류의 항이 들어간다. [§3.2; Appendix A, B.3–B.7, pp. 24–35](https://arxiv.org/pdf/2511.08544v3#page=24)

```text
밀도와 과제의 상호작용: ∇m(x) · ∇log p(x)
과제의 곡률: Δm(x) / 2
J(p) = ∫ ||∇log p(x)||² p(x) dx
```

`∇log p`는 밀도의 score다. 이 score가 큰 영역에서는 이웃 분포의 기울기가 단순 평균 추정에 영향을 준다. 과제 gradient의 평균이 0이고 second moment가 `τg²I`인 사전 가정을 쓰면 score 항을 제곱한 기대값이 `τg²J(p)`로 정리된다.

단, 원문의 부록 B.4는 gradient와 곡률 항의 decorrelation을 추가로 다룬다. 그 조건이 없으면 cross term도 `O(r⁴)`로 남으며 같은 차수라고 해서 작게 무시할 수 없다. 곡률 항이 `p`와 무관하다는 해석에도 과제 사전 가정이 필요하다. 따라서 ‘leading 항 하나가 작다’와 ‘전체 bias가 무조건 유일하게 최소’는 구분한다. [Appendix B.4, p. 29](https://arxiv.org/pdf/2511.08544v3#page=29)

kernel 분석은 target gradient·Hessian의 bound 등을 이용한 최악 경우 bias 상한에 `2B²+8L²J(p)` 형태를 넣는다. 여기서 `L`, `B`는 smoothness bound이며 SIGReg 손실과 다른 기호다. 이 상한을 줄이는 분포를 선택하는 논증이지, 모든 target의 정확한 risk를 그대로 최소화한 식은 아니다. pointwise variance는 query 밀도에 의존하고, integrated variance에서의 밀도 상쇄도 적분 가능성·동일 train/query 분포 등의 조건을 요구한다. [Appendix B.7, pp. 34–35](https://arxiv.org/pdf/2511.08544v3#page=34)

## Gaussian 최소성의 핵심 증명

평균 0, 공분산 `Σ≻0`, smooth density와 경계항이 사라지는 조건을 둔다. score를 `u(x)=∇log p(x)`라 하면 적분 부분으로 `E[u(X)Xᵀ]=−I`다. 아래 제곱 평균의 non-negativity를 사용한다. [Appendix A; B.5, pp. 32–33](https://arxiv.org/pdf/2511.08544v3#page=32)

```text
0 ≤ E ||u(X) + Σ⁻¹X||²
  = J(p) − tr(Σ⁻¹)

따라서 J(p) ≥ tr(Σ⁻¹)
```

등호는 `u(x)=−Σ⁻¹x`가 거의 모든 곳에서 성립할 때다. 이를 적분하면 `log p(x)=상수−½xᵀΣ⁻¹x`, 즉 Gaussian이다. 더 나아가 `trΣ=c`를 고정하면 `trΣ⁻¹≥K²/c`이고 등호는 `Σ=(c/K)I`다. 목표 공분산을 `I`로 정규화하면 표준 등방성 Gaussian이 된다.

이 증명은 Fisher objective의 최소성을 설명한다. 실제 finite image dataset의 결정적 임베딩이 매끄러운 full-dimensional density를 이루는지, downstream task prior가 가정과 맞는지는 별도 질문이다. 이 조건을 분리해야 ‘provable’이 무엇을 보장하는지 정확해진다.

## 모든 투영이 Gaussian이면 왜 원래 분포도 Gaussian인가

벡터 `Z`의 특성함수는 `φZ(t)=E exp(i tᵀZ)`다. 임의 `t≠0`는 `t=s a`, `‖a‖=1`로 쓸 수 있다. 그러므로 모든 방향 `a`의 모든 스칼라 주파수 `s`에서 투영 특성함수가 같으면 벡터 특성함수도 모든 `t`에서 같다. 특성함수의 유일성이 원래 분포의 일치를 준다. 이것이 Cramér–Wold 연결의 간단한 증명이다. [§4.1; Appendix B.8, p. 35](https://arxiv.org/pdf/2511.08544v3#page=35)

![LeJEPA Figure 5: 좌표별 정규성만으로 결합 정규성을 판정할 수 없는 X자 분포 예.](/assets/reviews/lejepa/paper-figure-5.webp)

[그림 확대](/assets/reviews/lejepa/paper-figure-5.webp)

<strong>Figure 5를 이 증명과 연결하기.</strong> 왼쪽 두 주변 histogram만 보면 Gaussian처럼 보여도, X자 결합 분포는 대각 방향의 투영에서 다른 밀도를 보인다. 세 번째 패널의 각 행은 색 화살표 방향에 대응하고, 마지막 패널은 통계의 방향별 반응이다. ‘모든 방향’이라는 조건을 몇 개 좌표의 평균·분산으로 바꿀 수 없다는 예다. 저자: Balestriero·LeCun, [Figure 5, p. 8](https://arxiv.org/pdf/2511.08544v3#page=8), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). 전체 패널 crop·PDF rasterization·무손실 WebP 변환.

원문 Eq. 4의 최대 검정 통계와 실제 학습에서 사용하는 평균 손실도 구분해야 한다. 부록 B.9의 consistency 논증에는 증가하며 dense해지는 방향 집합, global null threshold의 보정, 대립 분포를 분리하는 근방의 검정력이 필요하다. 유한한 방향의 평균 학습 손실이 곧 level-α 검정은 아니다. 귀무가설을 기각하지 못한 것도 Gaussian이라는 증명이 아니다.

후보에는 CDF 기반 거리, Shapiro–Wilk 같은 순서 통계도 있다. 이들은 정렬·표본 간 순서 관계와 분산 통신을 다뤄야 한다. 정렬이 모든 구간에서 미분 불가능하다는 뜻은 아니다. 순서가 고정된 영역에서는 piecewise 미분이 가능하지만 순서가 바뀌는 경계의 비매끄러움과 계산 구조가 ECF 평균과 다르다. Appendix E–F는 비교 통계의 정의를 보완한다. [§4.2, pp. 7–9; Appendix E–F, pp. 43–46](https://arxiv.org/pdf/2511.08544v3#page=7)

## ECF 손실: 계산과 gradient의 장점

방향 `a`에 대한 표본 투영 `sn=aᵀzn`를 사용한다. 아래 `D`는 batch 크기를 곱하지 않은 discrepancy다.

```text
φ̂a(t) = (1/N) Σn [cos(t sn) + i sin(t sn)]
ψ(t) = exp(−t²/2)
D(a) = ∫ |φ̂a(t) − ψ(t)|² w(t) dt
SIGReg = (1/M) Σm N D(am)     # 원문 Algorithm 1의 스케일
```

복소수 차이의 제곱은 `(cos 평균−ψ)²+(sin 평균)²`로 계산한다. 고차 moment의 존재가 없어도 특성함수는 항상 존재한다. 정렬 없이 표본 평균을 계산할 수 있고, 이 평균은 분산 환경에서도 공통 방향·grid를 사용하면 합칠 수 있다. 다만 local batch 크기가 다르면 sample-count 가중 평균이 필요하고 gradient를 전달하는 collective도 확인해야 한다. 이 분산 구현 조건은 리뷰의 검토 사항이며 이번에 실행한 DDP 결과가 아니다. [§4.3; Algorithm 1](https://arxiv.org/pdf/2511.08544v3#page=10)

원문 Theorem 4의 부록 증명은 `w(t)=exp(−s²t²)`일 때 unscaled `D`의 각 표본에 대한 derivative를 제어한다. `|φ̂|≤1`, `|ψ|≤1`과 `∫|t|w(t)dt=1/s²`를 이용한다.

```text
|∂D/∂sn| ≤ 4/(N s²)
ND를 사용하면 |∂(ND)/∂sn| ≤ 4/s²
```

같은 스케일을 유지해야 ‘N이 커지면 gradient가 작아진다’는 해석의 혼동을 피한다. 네트워크 파라미터로 미분하면 encoder·projector Jacobian이 chain rule에 추가된다. <strong>표본 loss derivative의 bound는 전체 parameter gradient의 bound나 SGD의 전역 수렴 증명이 아니다.</strong> [Theorem 4; Appendix B.12, pp. 37–38](https://arxiv.org/pdf/2511.08544v3#page=37)

## 표본 bias와 유한한 주파수 grid

독립 표본에서 복소수 `exp(it sn)`의 크기는 1이다. 같은 표본 인덱스와 다른 인덱스를 나눠 합하면 다음 기대값을 얻는다. [Theorem 6; Appendix B.13, pp. 38–40](https://arxiv.org/pdf/2511.08544v3#page=38)

```text
E |φ̂(t) − ψ(t)|²
 = |φ(t) − ψ(t)|² + (1 − |φ(t)|²)/N
```

unscaled discrepancy에는 `O(1/N)` 추가 항이 있고, regularity를 두고 미분과 기대값을 교환하면 관련 gradient bias도 이 스케일로 분석한다. 그러나 Algorithm 1처럼 `N`을 곱하면 <strong>추가 항의 절대 크기는 O(1)</strong>이 된다. 상대적 scale이나 정규화를 밝히지 않은 채 ‘scaled 손실의 절대 bias가 0으로 간다’고 말하면 안 된다. 목표 Gaussian에서 finite batch 손실도 정확히 0은 아니다.

리뷰의 Monte Carlo 검사는 `N=32`, 표준 Gaussian, `t=1`, 20,000회 반복에서 unscaled squared discrepancy를 계산했다. 평균 `0.0198677`, 이론값 `(1−e⁻¹)/32=0.0197538`, Monte Carlo standard error `0.0001505`였다. 이론식과 맞는 작은 표본 검사이며 신경망 실험 재현이 아니다.

수치 적분은 별도 근사다. 원문 Algorithm 1은 `[-5,5]`의 등간격 17점을 사용한다. 리뷰에서 그 grid의 간격 `Δ=0.625`를 확인하고, 1D Gaussian의 평균을 `2π/Δ≈10.0531`만큼 옮기는 예제를 계산했다. grid의 모든 `t`가 `Δ`의 정수배여서 phase shift가 1이고 특성함수 값이 그대로다.

```text
grid에서 최대 CF 차이: 약 2.24×10⁻¹⁶
grid 밖 t=0.3에서 CF 차이: 약 1.9082
```

이것은 <strong>고정된 1D 주파수 grid만으로 분포 식별을 인증할 수 없다는 예</strong>다. 모든 주파수를 사용하는 population 정리의 반례가 아니고, 고차원에서 방향을 계속 resample하는 LeJEPA 학습이 이 실패를 겪었다는 실험도 아니다. 표본 수를 늘리는 것으로 quadrature의 관측 빈틈이 자동으로 사라지지는 않는다는 점을 보여준다.

## 방향 수와 계산 복잡도: ‘선형’의 고정 변수를 밝히기

dense projection은 `N×K`와 `K×M`의 곱이고, 각 투영에서 `T`개 주파수를 계산한다. 따라서 손실 계산은 대략 다음 비용이다.

```text
시간: O(NKM + NMT)
단순 구현의 CF 중간 tensor: O(NMT)
```

`K,M,T`를 고정하면 `N`에 선형이다. `M`을 고정하면 projection이 `K`에 선형이지만 `M∝K`로 늘리면 그 항은 `O(NK²)`가 된다. encoder 계산·optimizer 메모리는 이 식 밖에 있다. Gaussian weighted CF와 kernel MMD의 관계는 유용하지만, 기존 MMD에도 선형 추정기가 있었으므로 선형 비용을 최초의 발명이라고 소개하지 않는다. [§4.3, p. 9; Gretton et al., 초록](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf#page=1)

Theorem 5에는 smoothness `α`, quasi-uniform 방향, 정확한 투영 제약 등 아래 `M^(−2α/(K−1))` 형태의 근사율이 등장한다. 상수와 density norm도 차원·smoothness에 의존한다. 고정된 `α`에서 차원이 커지면 지수는 나빠진다. 따라서 차원의 저주를 무조건 없앴다는 결론은 아니다. 실제 random finite slice 평균의 학습 곡선은 이 정리와 다른 종류의 근거다. [Appendix B.10, pp. 36–37](https://arxiv.org/pdf/2511.08544v3#page=36)

![LeJEPA Figure 7: 방향 수에 따른 고정 방향과 resampled 방향의 discrepancy 비교.](/assets/reviews/lejepa/paper-figure-7.webp)

[그림 확대](/assets/reviews/lejepa/paper-figure-7.webp)

<strong>Figure 7 해설.</strong> 가로축 `M`은 slice 수(log scale), 세로축은 투영 discrepancy의 기대값이다. 초록은 방향을 다시 뽑는 조건, 파랑은 고정한 조건이다. 이 실험에서는 resampling이 작은 `M`에서도 낮은 discrepancy에 도달한다. 표시된 `β`와 `R²`는 해당 실험의 적합 요약이며 임의 데이터에서의 수렴 보장이 아니다. downstream 정확도 그래프도 아니다. 출처: Balestriero·LeCun, [Figure 7, p. 10](https://arxiv.org/pdf/2511.08544v3#page=10), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). 해당 그림의 전체 graph와 legend를 crop·무손실 WebP 변환했다.

Table 6의 V100 손실 시간은 `N=512,M=512,T=16`에서 `0.465236±0.011642 ms`, `N=8192`에서 `6.188304±0.007226 ms`다. §4.4는 forward–backward 측정이라고 설명한다. 이는 SIGReg의 측정이고 전체 pretraining 시간은 아니다. 표에 embedding 차원 `K`가 없고 forward/backward별 시간이 나뉘지 않아 동일 조건의 시스템 비교는 추가 기록이 필요하다. [§4.4, p. 10; Table 6, p. 43](https://arxiv.org/pdf/2511.08544v3#page=43)

## 직접 확인한 수식 주의점: 쌍별 거리와 중심 거리

원문 Eq. 5–7 및 Appendix B.6은 모든 뷰와 전역 뷰 사이의 평균 제곱 거리를, 전역 평균에 대한 중심 거리로 바꾼다. 한 이미지에 대해 전역 평균을 `μ=(1/Vg)Σg zg`라고 두면 정확한 전개는 다음과 같다.

```text
A = (1/VVg) Σv,g ||zv − zg||²
B = (1/V) Σv ||zv − μ||²
C = (1/Vg) Σg ||zg − μ||²

A = B + C
```

`zv−zg=(zv−μ)−(zg−μ)`로 전개하면 cross term은 전역 뷰에 대해 합한 뒤 사라진다. 하지만 마지막 제곱 항 `C`는 남는다. `C`는 파라미터에 의존하므로 일반적으로 상수처럼 생략할 수 없다.

리뷰의 scalar 예제에서 전체·전역 뷰를 모두 `[0,2]`로 두면 `μ=1`, `A=2`, `B=1`, `C=1`이다. 원문 PDF를 시각적으로 대조했고 부록 Eq. 23→24에서 빠진 항을 확인했다. OCR 문제로 판단한 것은 아니다. [§5, p. 12; Appendix B.6, p. 34](https://arxiv.org/pdf/2511.08544v3#page=34)

위 항등식은 벡터 제곱거리 기준이다. Algorithm 2의 `.square().mean()`은 feature 차원 `K`도 평균하므로 그 구현의 합의 손실은 `B/K`다. 누락 항도 같은 스케일에서 `C/K`로 남는다. 중심 기반 목적은 명확하게 정의된 학습 목적이다. 따라서 이 글은 그 목적을 설명하며, 쌍별 `A`와 정확히 동치라는 서술에만 주의를 남긴다. <strong>이 불일치가 보고된 정확도를 무효화한다거나 어떤 코드로 모든 실험을 실행했는지까지 밝혀 준다는 뜻은 아니다.</strong> 두 목적의 차이는 같은 예산으로 구현·학습해 검증할 별도 연구 질문이다.

## 논문 알고리즘과 현재 공식 예제를 구분하기

재현을 시작할 때 PDF와 현재 저장소 예제를 동일한 것으로 취급하면 설정이 섞인다. 이 리뷰는 공식 저장소 [commit c293d291](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052)의 README와 MINIMAL을 직접 읽었고 실행하지 않았다.

| 축 | 논문에서 확인한 설정 | 해당 MINIMAL 예제 |
| --- | --- | --- |
| 주파수 grid | Algorithm 1: 17점 `[-5,5]` | 17점 `[0,3]`, 대칭성을 이용해 가중치 doubling |
| 방향 수 | Algorithm 1 기본 256; §6.1 권장 1,024 | 256 |
| 합의 중심 | Algorithm 2: 전역 뷰들의 평균 | 모든 뷰의 평균 |
| teacher·detach | 핵심 구조에 없음 | 핵심 합의에도 없음 |

대칭 적분에서 양의 반구간을 쓰는 것은 타당한 계산 선택일 수 있다. 다만 끝점과 grid 간격도 바뀌므로 숫자상 같은 quadrature는 아니다. 예제 작성일도 PDF v3 이후다. 본문 default의 총 10개 뷰와 Experiment Details 1의 총 8개 뷰 역시 각각 기록해야 한다.

## 실험이 이론을 얼마나 검증하는가

Table 1의 batch ablation은 학습 가능성과 민감도를 확인한다. Figure 9의 여러 backbone은 목적의 구조 범위를 넓힌다. Table 2의 전이는 다양한 후속 과제에서 representation이 쓰인다는 증거다. Figure 12·Table 3은 전문 도메인의 작은 직접 학습이 강할 수 있음을 보여준다. 어떤 결과도 실제 표현 density가 모든 smoothness·task-prior 가정을 만족한다고 인증하지는 않는다.

ImageNet-1K 사전학습 전이 평균에서 LeJEPA ViT-L은 1-shot/10-shot/전체 라벨 `29.55/60.95/79.48%`, plain I-JEPA는 `30.20/60.51/78.50%`, +STOP은 `32.05/62.92/80.70%`다. 모델은 304M/100 epoch 대 632M/300 epoch로 다르다. 이 표는 유망한 결과와 비교의 조건을 함께 보여준다. 조건 없는 우위나 정확히 3배의 총 계산 절감을 증명하지 않는다. [Table 2, p. 16](https://arxiv.org/pdf/2511.08544v3#page=16)

이 리뷰의 후속 제안은 세 가지다. 학습에 쓰지 않은 방향·더 촘촘한 grid의 discrepancy를 측정한다. 중심 손실과 정확한 쌍별 손실을 같은 설정에서 비교한다. 동일 backbone·augmentation·총 예산으로 대안 손실을 비교하고 frozen/finetuning, 여러 seed의 결과를 따로 기록한다. 예상 반례는 낮은 training loss에도 unseen projection이 나쁜 경우, target-relevant한 비등방성이 downstream에 도움이 되는 경우다. 이번에 이 실험들은 실행하지 않았다.

## 출처·읽은 범위·검증 상태

- 원문: *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, Randall Balestriero·Yann LeCun, 최초 공개 2025-11-11, 검토 v3(2025-11-14), [공식 서지](https://arxiv.org/abs/2511.08544).
- 1차 읽기: pp. 1–3, 17–18과 전체 절 구조에서 문제·컨셉·흐름을 파악. 2차 읽기: 본문 pp. 1–18, Appendix A–F pp. 24–46, 보충 Figures 16–21 pp. 47–50. 참고문헌 pp. 18–23은 관련 원전 서지를 선택 확인. 주요 수식·표·축은 원본 PDF 이미지와 대조했다.
- 관련 원전 범위: VICReg v3 pp. 3–4, DINO v2 pp. 2–3, I-JEPA v3 pp. 3–4, Gaussian Embeddings v1 pp. 1, 3–4, Gretton et al. 2012 PDF pp. 1, 3–4. 비교 내용은 [전체 해설](/reviews/lejepa-complete/)에 있다.
- 대표 그림은 직접 제작한 설명도([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)); 기본 Light(300), 강조 Semi Bold(600). 원문 그림의 저자·위치·CC BY-SA 4.0·crop/인코딩 변경은 각 캡션에 남겼다.
- <strong>실행한 검증:</strong> 리뷰 작성자의 4개 검사—중심 거리 항등식, 고정 1D grid alias, OLS 역고윳값 합, ECF sampling bias Monte Carlo. 위 수치는 이 작은 검사에서 나온 것이며 공개 모델의 재현 결과가 아니다.
- <strong>미실행:</strong> 공식 코드 실행, GPU 사전학습, downstream probe·finetuning, DDP, 학습 seed 재현, 논문 시간·메모리 재측정, 전체 증명의 형식 검증. 형식 증명 도구로 검증했다고 주장하지 않는다.
