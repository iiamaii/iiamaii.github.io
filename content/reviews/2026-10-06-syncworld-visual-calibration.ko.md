---
title: "카메라가 바뀌어도 행동을 읽을 수 있을까: SyncWorld와 시각 보정"
description: "짧은 행동·영상 문맥으로 새 로봇 설정에 적응하는 SyncWorld를 살펴본다. 영상 예측의 개선과 정책 성공률을 구분하고, 관련 연구 10편을 통해 전이와 실행 비용의 남은 질문을 정리한다."
date: "2026-10-06"
updatedAt: "2026-10-09T05:04:32Z"
topics: ["world action model", "robot learning", "world simulation", "in-context learning", "visual calibration"]
visibility: "public"
lang: "ko"
translationKey: "syncworld-visual-calibration"
paperTitle: "SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators"
paperPublishedDate: "2026-09-08"
authors: ["Yuncong Yang", "Zhengtao Han", "Furkan Ozyurt", "Zeyuan Yang", "Han Yang", "Junyi Cao", "Haoyu Zhen", "Yilun Du", "Chuang Gan"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2609.09155v1"
thumbnail: "/assets/reviews/syncworld/paper-figure-2.webp"
thumbnailAlt: "SyncWorld 원문 Figure 2의 모델 구조도"
---

로봇의 행동 숫자를 안다고 화면 속 움직임까지 바로 알 수 있을까? SyncWorld는 새 설정에서 짧은 행동·영상 대응을 먼저 관찰하고, 추가 학습 없이 그 문맥을 이용해 미래 영상을 예측한다. 보고된 결과는 이 접근의 가능성을 보여주지만, 모든 몸체의 물리적 정확성이나 모든 과제의 정책 개선까지 입증하지는 않는다. [SyncWorld §3–4, 부록 C.4.3·E](https://arxiv.org/html/2609.09155v1#S3)

## 어떤 질문에서 출발했는가

행동 조건 월드 모델은 현재 관찰과 앞으로 실행할 행동을 받아 미래 관찰을 예측하는 모델이다. 로봇의 이동 명령이 같아도 카메라 방향이 달라지면 영상에서 보이는 이동 방향은 달라질 수 있다. 카메라를 반대편으로 옮기는 상황은 이 문제를 설명하기 위한 예시이며, 별도의 실험 결과가 아니다. SyncWorld는 설정에 따른 이 대응을 **행동–영상 매핑**으로 다룬다. [SyncWorld §3.1](https://arxiv.org/html/2609.09155v1#S3.SS1)

<figure class="review-figure" id="figure-syncworld-core">
<a href="/assets/reviews/syncworld/syncworld-core.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/syncworld-core.svg" width="1200" height="1590" alt="보정 C_s의 행동·영상 쌍, 이력 H_t의 접근·집기·들기, 미래 명령 A_t의 이동·그리퍼 닫기 예시를 넣은 개념도. 입력은 배포 시 가중치가 고정된 SyncWorld를 조건화한다. 샘플은 AI 생성 설명용 이미지이며 파란색 목표 자세는 실행 전 계획이다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">핵심 개념 · 새 문맥, 고정된 모델</span>보정은 새 설정의 행동–영상 대응을 입력 문맥으로 알려준다. 가중치 θ는 배포 시 바꾸지 않는다. C_s는 12방향 보정, H_t는 이미 일어난 관찰·행동, A_t는 예측할 미래 명령이다. 세 입력 카드의 샘플은 역할을 설명하는 AI 생성 예시다. 보정의 행동·영상 쌍, 이미 수행한 접근·집기·들기, 실행 전 명령을 구분한다. 미래 행동의 파란색 목표 자세는 계획을 뜻한다. 논문 데이터·실제 보정 궤적·모델의 예측 영상이나 실험 결과가 아니다. <span class="figure-links"><a href="/assets/reviews/syncworld/syncworld-core.svg" target="_blank" rel="noopener noreferrer">크게 보기</a></span></figcaption>
</figure>

기존 연구는 이미 여러 부분을 해결하고 있었다. IRASim은 행동과 영상 프레임의 대응을 정교하게 모델링했고, WorldGym은 생성된 환경에서 정책을 평가했다. Ctrl-World는 다중 시점과 이력을 활용하며 새 DROID 카메라 설정에서의 제로샷 결과도 보고했다. 따라서 SyncWorld를 “이전에는 설정 전이가 불가능했는데 처음 해결한 모델”로 소개하기보다, 대응을 알려주는 보정 문맥을 명시적으로 설계한 접근으로 읽는 것이 적절하다. 마지막 판단은 리뷰 작성자의 해석이다. [IRASim §3](https://arxiv.org/html/2406.14540v2), [WorldGym §3–4](https://arxiv.org/html/2506.00613v3), [Ctrl-World §4, §5.4](https://arxiv.org/html/2510.10125v3)

## 핵심 아이디어와 필요한 배경

시각 보정은 고정된 카메라에서 로봇이 기본 방향으로 움직이는 모습을 수집하는 과정이다. 위치와 회전의 여섯 자유도에 대해 양·음 방향 구간을 추출해, 총 12개 구간을 정해진 순서로 문맥 앞에 놓는다. 제어 공간은 위치 3개, 회전 3개, 그리퍼 1개의 차원으로 구성된다. 이것은 기계적 관절이 일곱 개라는 뜻이 아니다. 그리퍼는 영상에서 의미를 직접 읽을 수 있다는 이유로 방향 보정에서 제외한다. [SyncWorld §3.1, 식 4, 부록 B.2.2–B.2.3](https://arxiv.org/html/2609.09155v1#S3.SS1)

식 3의 조건 관계를 평문으로 풀면 다음과 같다. 정확한 수식 대신 입력과 출력의 역할을 보여주는 표현이다.

```text
future video ~ W_theta(C_s, H_t, A_t)
```

`C_s`는 설정 `s`의 보정 문맥, `H_t`는 현재까지의 관찰·행동 이력, `A_t`는 앞으로 실행할 행동이다. `W_theta`는 미래 영상의 조건부 분포를 나타내며, `theta`는 배포 시 고정된 모델 파라미터다. 여기서 제로샷은 새 설정에 맞춰 파라미터를 다시 학습하지 않는다는 의미다. 보정 관찰이나 계산이 필요 없다는 의미로 줄여 읽으면 안 된다. [SyncWorld §3.1, 식 3](https://arxiv.org/html/2609.09155v1#S3.SS1)

<figure class="review-figure" id="figure-syncworld-architecture">
<a href="/assets/reviews/syncworld/paper-figure-2.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-2.webp" width="1600" height="704" alt="SyncWorld 구조: 보정 에피소드·상호작용 이력·미래 행동이 자세 임베딩과 영상 잠재 표현으로 변환되어 DiT의 조건이 되고 미래 영상이 생성된다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 2 · 보정에서 미래 영상까지</span>왼쪽의 보정·이력·미래 행동을 입력으로, 가운데 DiT를 생성 본체로, 오른쪽을 예측 결과로 읽는다. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-2.webp" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2609.09155v1#S3.F2">Yuncong Yang et al., 2026, 원문 Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 내용·배치 유지, 해상도 축소·WebP 압축, 자르기 없음</span></figcaption>
</figure>

그림의 세 입력은 역할이 다르다. **보정**은 이 설정에서 행동과 화면 변화의 대응을 알려주고, **이력**은 이미 진행된 상호작용을 담으며, **미래 행동**은 앞으로 실행할 명령이다. 자세는 벡터 임베딩으로, 영상은 압축된 잠재 표현으로 처리된다. 확산 트랜스포머(DiT)는 이 조건을 이용해 잡음 상태의 미래 영상 표현을 정제한다. 그림의 미래 잡음은 생성할 프레임의 출발점이다. [원문 Figure 2, §3.1, 부록 B.1.1](https://arxiv.org/html/2609.09155v1#S3.F2)

## 방법과 가정

이 방법은 보정 영상을 기존 생성 모델에 붙이는 것만으로 완성되지 않는다. 학습 중 보정·이력·미래 행동에 같은 좌표 변환을 적용해, 숫자 자체보다 문맥 안의 대응을 읽도록 유도한다. 또한 보정을 포함한 입력과 보정을 제거한 이력 중심 입력의 예측을 맞추는 **보정–이력 증류**를 사용한다. 추론 시 명시적 보정 없이 나온 결과도 이런 학습을 거친 모델의 결과이므로, 보정 학습 자체를 하지 않은 모델과 구분해야 한다. [SyncWorld §3.2, 부록 B.1.2–B.1.3](https://arxiv.org/html/2609.09155v1#S3.SS2)

학습 자료는 주로 RLBench·RoboCasa·RoboMimic에서 카메라를 바꾸고 행동을 재생해 만든 시뮬레이션 데이터다. 성공 궤적에서 벗어난 행동의 결과도 포함하며, 보정이 없는 DROID 실물 데이터도 사용한다. 새 설정에서 추가 학습이 없다는 설명과, 사전에 다양한 자료로 학습했다는 사실을 함께 보아야 한다. [SyncWorld §3.4, 부록 B.2](https://arxiv.org/html/2609.09155v1#S3.SS4)

문맥을 통한 적응의 이론적 배경으로는 Context and Diversity Matter가 유용하다. 이 논문은 이미 본 환경을 식별하는 것과 문맥에서 전이를 학습하는 것을 구분한다. 다만 이론의 상태·공간 가정과 실험 환경은 로봇 영상과 다르므로, 그 결과를 SyncWorld의 성능에 대한 증명으로 가져오지는 않는다. [Context and Diversity Matter §3–4](https://arxiv.org/html/2509.22353v2)

## 결과와 근거

영상 평가에서는 512×512 해상도로 16개 행동에 대응하는 미래 구간을 예측한다. LIBERO와 ManiSkill은 각각 50개 궤적, 실물은 25개 궤적을 사용하며, 궤적당 두 시점으로 각각 100·100·50개 평가 영상을 구성한다. Table 1은 IRASim·WorldGym·Ctrl-World보다 전반적으로 좋은 영상 품질 지표를 보고한다. 비교 모델들은 같은 downstream 데이터로 미세 조정되지만 백본과 학습 방식까지 같지는 않으므로, 모델 간 차이를 보정 하나의 효과로 해석하기 어렵다. [SyncWorld §4.1–4.2, Table 1, 부록 C.1](https://arxiv.org/html/2609.09155v1#S4.SS1)

<figure class="review-figure" id="figure-syncworld-qualitative">
<a href="/assets/reviews/syncworld/paper-figure-5.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-5.webp" width="2400" height="893" alt="두 카메라 시점의 실제 로봇 장면을 시간 순서로 나열해 실제 관찰, Ctrl-World와 SyncWorld의 미래 예측을 비교한 논문 원본 그림." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 5 · 실제 로봇의 예측 비교</span>왼쪽·오른쪽은 서로 다른 카메라 시점이며, 각 묶음의 행은 위에서부터 실제 관찰(Ground-truth), Ctrl-World, SyncWorld다. 가로 방향으로 시간에 따른 변화를 읽고 팔·그리퍼 위치를 비교한다. 저자가 고른 정성 예시이며 전체 성공률이나 물리 정확성의 증명은 아니다. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-5.webp" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2609.09155v1#S4.F5">Yuncong Yang et al., 2026, 원문 Figure 5</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 내용·배치 유지, 해상도 축소·WebP 압축, 자르기 없음</span></figcaption>
</figure>

다른 시점에서도 예측이 같은 장면을 나타내는지 확인하는 Met3R은 특징 대응에 기반한 일관성 지표다. 단위 없는 점수이며 낮을수록 좋다. 평균은 LIBERO·ManiSkill·실물 세 영역의 보고 평균이다. 먼저 세 영역별 열을 비교한 뒤 평균 열을 읽는 것이 좋다.

**원문 Table 2 — 시점 간 일관성.** 아래는 전체 행을 옮긴 것이다. 낮을수록 좋으며, Oracle은 실제 관찰의 참고 점수다.

| 모델 | LIBERO ↓ | ManiSkill ↓ | 실물 ↓ | 평균 ↓ |
| --- | --- | --- | --- | --- |
| IRASim | 0.579 | 0.633 | 0.468 | 0.560 |
| WorldGym | 0.583 | 0.632 | 0.515 | 0.577 |
| Ctrl-World | 0.559 | 0.642 | 0.494 | 0.565 |
| SyncWorld, 추론 보정 없음 | 0.545 | 0.607 | 0.474 | 0.542 |
| SyncWorld, 추론 보정 사용 | 0.540 | 0.602 | 0.473 | 0.538 |
| Oracle | 0.515 | 0.594 | 0.459 | 0.523 |

추론 보정 사용 여부를 비교하면 평균이 0.542에서 0.538로 낮아진다. 이 두 행 모두 보정·증류 학습을 거친 모델이므로 보정 학습 자체를 제거한 비교와는 다르다. 평균에서는 SyncWorld가 좋지만, 실물 열에서는 IRASim의 값이 더 낮다. 따라서 “모든 영역에서 가장 일관적”이라는 서술은 표와 정확히 일치하지 않는다. 두 값의 차이에 대한 통계적 유의성은 확인하지 않았다. 이 지표는 시점 간 특징 일관성을 평가하며 힘·접촉·물리 법칙 전체의 정확성을 직접 검증하지 않는다. [SyncWorld §4.3, Table 2, 부록 C.3](https://arxiv.org/html/2609.09155v1#S4.SS3)

정책 개선은 별도의 질문이다. SyncWorld는 동결한 시각·언어·행동 정책(VLA) π0에서 8개 행동 후보를 뽑고, 각 후보의 두 시점 미래 영상을 생성한다. GPT-5 시각·언어 모델(VLM)이 과제 진행과 물리적 타당성을 평가해 후보를 선택한다. 이는 생성 예측 제어(GPC)의 후보 순위 방식인 GPC-Rank를 이용한 구성이다. [SyncWorld §3.3, 부록 C.4.1–C.4.2](https://arxiv.org/html/2609.09155v1#S3.SS3), [GPC §III–IV](https://arxiv.org/html/2502.00622v4)

<figure class="review-figure" id="figure-syncworld-policy">
<a href="/assets/reviews/syncworld/paper-figure-4.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-4.webp" width="1800" height="710" alt="원문 Figure 4. VLA가 행동 후보를 만들고 SyncWorld가 후보별 두 시점 영상을 예측한다. VLM이 지시와 예측 결과로 점수를 매기고 최고 점수 후보를 실행한다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 4 · 예측 영상에서 행동 후보 선택까지</span>왼쪽에서 오른쪽으로 후보 생성·미래 영상·점수 평가를 읽는다. 같은 색의 행이 같은 행동 후보다. 오른쪽 숫자는 후보 평가 점수이며 성공률(%)로 읽으면 안 된다. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-4.webp" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2609.09155v1#S3.F4">Yuncong Yang et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 내용·배치 유지, 해상도 축소·WebP 압축, 자르기 없음</span></figcaption>
</figure>

그림은 후보 세 줄을 예시로 보여주지만 실제 평가 구성은 8후보·후보당 16행동·두 시점이다. GPT-5에는 생성 영상에서 뽑은 대표 프레임 3개를 제공한다. SyncWorld는 결과를 예측하고 GPT-5가 점수를 매긴다. 두 단계의 오류가 합쳐지므로 영상 품질 개선을 곧바로 행동 선택 개선으로 읽을 수 없다. [원문 Figure 4, §3.3, 부록 C.4.2·Table 7](https://arxiv.org/html/2609.09155v1#S3.F4)

**원문 Table 3 — 오라클 개선 여지가 있는 세 과제.** 원문의 성공 비율을 %로 변환했고, 보정 없는 행도 함께 옮겼다.

| 방법 | BBQ Sauce | Orange Juice | Black Bowl |
| --- | --- | --- | --- |
| 직접 π0 실행 | 52% | 56% | 48% |
| SyncWorld, 추론 보정 없음 | 54% | 68% | 58% |
| SyncWorld, 추론 보정 사용 | 58% | 72% | 60% |
| 실제 시뮬레이터로 후보 평가 | 60% | 80% | 66% |

성공률은 과제당 50회 실행에서 성공한 비율이다. Orange Juice에서 직접 실행 56%→보정 없음 68%→보정 사용 72%로 읽으면, 전체 후보 평가 방식의 이득과 추론 보정 추가분을 구분할 수 있다. 직접 실행 대비 변화는 16%포인트, 보정 없는 모델 대비 추가 변화는 4%포인트다. 다만 세 과제는 실제 시뮬레이터를 사용한 후보 평가가 직접 정책보다 좋아지는 조건으로 선정됐다. SyncWorld의 성능으로 고른 것은 아니지만 전체 LIBERO 평균도 아니다. [SyncWorld §4.4, Table 3, 부록 C.4.3, Table 8](https://arxiv.org/html/2609.09155v1#A3.SS4.SSS3)

**원문 Table 8 — 개선이 보장되지 않는 추가 과제.** 같은 방식으로 원문의 비율을 %로 옮겼다. 위의 세 과제만으로 전체 효과를 판단하면 이 반례가 사라진다.

| 방법 | Alphabet Soup | Ketchup | Put Cream Cheese |
| --- | --- | --- | --- |
| 직접 π0 실행 | 58% | 80% | 34% |
| SyncWorld | 58% | 72% | 34% |
| 실제 시뮬레이터로 후보 평가 | 62% | 74% | 30% |

Ketchup에서는 실제 시뮬레이터도 직접 실행보다 6%포인트 낮고, Put Cream Cheese에서는 4%포인트 낮다. 정확한 미래 관찰을 제공해도 평가기가 고른 후보가 더 나쁠 수 있다. 저자는 이런 과제에서 VLM/GPC-Rank 신호가 병목이라고 해석한다. 다만 이 표만으로 SyncWorld의 예측 오류와 후보 평가 오류의 기여도를 분리할 수는 없다. [원문 부록 C.4.3, Table 8](https://arxiv.org/html/2609.09155v1#A3.T8)

**원문 Table 6 — 생성 시간과 계산 조건.** 아래는 SyncWorld 행만 발췌한 것이다.

| GPU 구성 | Denoising step | 생성 시간 |
| --- | --- | --- |
| H100 1장 | 50 | 39.4초 |
| H100 1장 | 20 | 15.6초 |
| H100 4장 | 20 | 3.2초 |

실행 비용도 조건을 붙여 읽어야 한다. Table 6의 3.2초는 H100 네 장과 20회 denoising step을 사용한 생성 설정의 값이다. 같은 표에서 H100 한 장, 20회 step은 15.6초다. 어느 값도 정책 추론, 8개 후보와 두 시점의 평가, GPT-5 판단을 모두 포함한 의사결정 지연으로 확인되지 않았다. [SyncWorld 부록 C.2, Table 6](https://arxiv.org/html/2609.09155v1#A3.SS2)

## 한계와 나의 해석

저자가 보여주는 실패에는 극단적인 카메라 시점의 불안정한 움직임, 보지 못한 물체의 흐림과 위치 오류, 여러 물체 중 일부의 환각, 새로운 xArm의 정밀 회전 중 흐림이 있다. 팔이 명령을 따라 움직이는 영상을 만들었다는 사실과 물체 상호작용까지 정확히 예측했다는 사실을 구분할 필요가 있다. 후자는 리뷰 작성자의 평가 기준이다. [SyncWorld 부록 E.1–E.4](https://arxiv.org/html/2609.09155v1#A5)

최근 연구는 이 구분을 더 구체적으로 만든다. XEWorld는 통제된 장면에서 양팔 몸체 전이를 평가하며 외형·운동학·물체 상태를 나눠 본다. 행동 표기 연구는 같은 정보를 담은 절대·상대 명령도 모델의 결과를 바꿀 수 있음을 조사한다. 전자는 SyncWorld와 몸체·행동 공간이 다르고, 후자는 고정 카메라와 오프라인 대리 평가를 사용하므로 SyncWorld를 직접 반박하는 동일 조건 실험은 아니다. 두 논문을 함께 읽고 얻는 제안은 카메라, 행동 표기, 몸체 전이를 분리해서 시험하자는 것이다. [XEWorld §3, §5–6](https://arxiv.org/html/2608.05799v1), [Robot World Models Are Not Invariant to How the Actions Are Written §3–6](https://arxiv.org/html/2609.23252v1)

계산 비용을 줄이는 다른 방향도 있다. Think Like a World Model, Act Like a VLA는 교사의 표현을 작은 정책에 증류하고 배포 시 교사를 제거한다. 기본 교사는 영상 롤아웃 시뮬레이터가 아닌 Cosmos3-Nano의 이해 표현을 제공한다. RoboActualizer는 동결한 V-JEPA 2.1 표현 위에 미래·행동 전문가를 학습하고, 실행할 때 미래 출력을 버린다. 둘 다 플랫폼별 학습을 거치는 접근이어서 SyncWorld의 새 설정 문맥 적응과 같은 조건은 아니다. [표현 증류 §III-B–III-C, §IV](https://arxiv.org/html/2609.24682v2), [RoboActualizer §3–4](https://arxiv.org/html/2609.36413v3)

ReWAM은 표현 설계와 행동 학습의 관계를 다룬다. 여기의 특징 보정은 SyncWorld의 행동–영상 보정 에피소드와 다른 개념이다. 리뷰 작성자로서는 이 흐름을 “영상이 그럴듯한가”, “행동에 맞게 변하는가”, “좋은 행동을 고르는 데 도움이 되는가”라는 세 질문으로 나눠 읽는 것이 유용하다고 본다. 이는 문헌을 연결한 해석이며 통합 실험 결과는 아니다. [ReWAM §3.2–3.4, §4.3](https://arxiv.org/html/2609.38163v1)

## 남은 질문

다음 세 가지는 후속 검증 제안이며 아직 실행하지 않았다.

1. 카메라만 바꾸는 경우, 정보를 보존하는 행동 표기만 바꾸는 경우, 몸체만 바꾸는 경우에 보정의 효과가 각각 유지되는가?
2. 보정 방향이나 순서를 바꾸고, 물체 상호작용 오류를 따로 측정하면 모델이 어떤 문맥을 이용하는지 구분할 수 있는가?
3. 후보·평가기 조건과 전체 계산 예산을 맞추면 영상 시뮬레이터를 사용하는 방식과 표현을 정책에 증류하는 방식의 이득은 어떻게 달라지는가?

## 출처와 읽은 범위

원문은 Yuncong Yang 외 8인의 *SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators* (2026), arXiv:2609.09155v1이다. 최초 제출일은 2026-09-08이며, 아래 원문과 서지 정보를 2026-10-06에 확인했다. 요청된 PDF는 도구의 크기 제한으로 읽지 못해 동일 v1의 공식 HTML을 사용했다. 중심 논문은 본문 §1–5와 아래 부록 범위를 읽었으며, 부록 전체와 보충 동영상을 모두 검토한 것은 아니다. [공식 서지](https://arxiv.org/abs/2609.09155v1), [공식 HTML](https://arxiv.org/html/2609.09155v1)

관련 연구는 핵심 계보 5편과 최근 대표 5편이다. 최근 그룹은 최초 제출일 2026-08-01부터 2026-10-06까지의 자료에서 관련성을 우선해 선정했으며, 시간순으로 가장 최신인 다섯 편이나 SyncWorld의 직접 후속작 목록을 뜻하지 않는다. 표의 버전은 실제 분석한 판본이다.

| 출처·버전 | 실제 읽은 범위 |
| --- | --- |
| [SyncWorld v1](https://arxiv.org/html/2609.09155v1) | 본문 §1–5; 부록 B.1–B.2, C.1–C.4, D–E의 관련 설정·표·실패 설명; 원본 Figure 2·4·5와 캡션; Table 2·3·6·8 행 재대조 |
| [IRASim v2](https://arxiv.org/html/2406.14540v2) | §3.1–3.3, §4.1–4.3, §5의 관련 문단·실험 설명 |
| [WorldGym v3](https://arxiv.org/html/2506.00613v3) | §3.1.1–3.1.3, §4.1–4.2, §6의 관련 문단 |
| [Ctrl-World v3](https://arxiv.org/html/2510.10125v3) | §4.1–4.2, §5.3–5.4, §6의 관련 문단 |
| [GPC v4](https://arxiv.org/html/2502.00622v4) | §III–IV, §V-A–V-C, §VII의 방법·실험·한계 문단 |
| [Context and Diversity Matter v2](https://arxiv.org/html/2509.22353v2) | §3.1–3.4, §4.1–4.2, §5의 관련 문단 |
| [XEWorld v1](https://arxiv.org/html/2608.05799v1) | §3.1–3.2, §5.1–5.6, §6의 관련 문단·표 |
| [행동 표기 불변성 연구 v1](https://arxiv.org/html/2609.23252v1) | §3–4, §5.1, §6 Limitations의 관련 문단 |
| [Think Like a World Model, Act Like a VLA v2](https://arxiv.org/html/2609.24682v2) | §III-B–III-C, §IV-B–IV-E, §V의 관련 문단·표 |
| [RoboActualizer v3](https://arxiv.org/html/2609.36413v3) | §3.1–3.4, §4.1–4.5.2, §5, 부록 C.1–C.2의 관련 문단·표 |
| [ReWAM v1](https://arxiv.org/html/2609.38163v1) | §3.2–3.4, §4.2–4.3, §4.5, §5의 관련 문단·표 |

GPC v4의 공식 제목은 *Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling*이다. RoboActualizer v3의 RoboTwin 평가 범위는 §5와 부록 C.2의 서술이 달라 미확인으로 남겼으며, 이 글은 해당 범위에 대한 결론을 사용하지 않는다.

확인 범위는 원문 텍스트·표·서지 정보와 아래에 명시한 그림이다. 관련 논문 전문 완독, 증명 전체 검산, 코드·가중치 실행, 보충 동영상 확인, 시뮬레이션·실물 재현, 통계적 유의성 검증은 **미실행**이다. 이번 수정에서는 원문 Figure 2·4·5의 이미지와 캡션, Table 2·3·6·8의 행을 직접 검토했다. 원문 그림을 변환한 WebP 세 개는 Yuncong Yang 외 저자(2026)의 [SyncWorld v1](https://arxiv.org/abs/2609.09155v1)에서 가져왔으며, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)에 따라 출처를 표시하고 내용·배치를 유지하고 해상도 축소와 WebP 압축만 적용했다. 직접 제작한 그림은 보정 문맥과 고정된 모델의 관계를 설명하는 개념도 하나이며 실제 실험 결과가 아니다. 표는 원문 값을 Markdown으로 옮겼고, 성공 비율만 %로 변환했다. 표지는 원문 Figure 2를 사용한다. 각 그림의 확대 링크는 최적화된 이미지를 연다. 원본 해상도는 원문에서 확인할 수 있다. 개념도 SVG는 벡터 글자·도형을 유지하고 삽입 이미지의 해상도와 압축만 조정했다.
