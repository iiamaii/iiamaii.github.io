---
title: "카메라가 바뀌어도 행동을 읽을 수 있을까: SyncWorld와 시각 보정"
description: "시각 보정과 이력 증류로 새 로봇 설정에 적응하는 SyncWorld를 살펴본다. 영상 품질·정책 성공률·실행 비용을 각각의 조건과 비교한다."
date: "2026-10-06"
updatedAt: "2026-10-23T15:26:11+00:00"
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

SyncWorld는 짧은 시각 보정(visual calibration) 문맥으로 새 카메라·단일 팔 로봇의 행동–영상 대응을 추론하는 제로샷(zero-shot) 월드 모델이다. 추론 보정 없이 평가할 때 학습에서 보정을 뺀 모델은 LIBERO 영상의 지각 특징 거리(Learned Perceptual Image Patch Similarity, LPIPS)가 0.049에서 0.134로 나빠지고(Table 4), SyncWorld로 정책 후보를 고르면 세 과제 중 이득이 가장 큰 Orange Juice에서 성공이 직접 π0 실행의 50회 중 28회에서 36회로 늘어난다(Table 3). [원문 Table 4](https://arxiv.org/html/2609.09155v1#S4.T4), [Table 3](https://arxiv.org/html/2609.09155v1#S4.T3) **리뷰 해석.** 보정 학습의 효과는 같은 백본 안의 절제(§4.5)가 가장 직접 보여 주지만 그 근거는 영상 지표에 한정되고, 정책 실험에서 추론 보정이 더하는 몫은 Orange Juice에서도 50회 중 34회→36회라 정책 성능의 근거로는 약하다.

원문 §1–5 순서를 따르고, 부록 C.2의 실행 비용과 E의 실패 사례는 결론 뒤에 모았다.

## 1. 도입: 같은 행동이 다른 영상이 되는 이유 (Introduction)

행동 조건 월드 모델(action-conditioned world model)은 현재 관찰과 앞으로 실행할 행동을 받아 미래 관찰을 예측하는 모델이다. 로봇의 이동 명령이 같아도 카메라 위치·로봇 받침 위치·몸체가 달라지면 영상에서 보이는 움직임은 달라진다. SyncWorld는 설정에 따른 이 관계를 **행동–영상 대응**(Action–Visual Mapping)으로 다룬다. 저자는 이 설정 의존성이 두 문제를 낳는다고 본다. 여러 설정을 섞어 학습하면 같은 행동 값에 서로 맞지 않는 대응을 맞춰야 하는 상충 감독(conflicting supervision)이 생기고, 새 설정에서는 대응이 바뀌어 추가 학습 없이는 모델이 깨진다. [SyncWorld §1](https://arxiv.org/html/2609.09155v1#S1)

<figure class="review-figure" id="figure-syncworld-core">
<a href="/assets/reviews/syncworld/syncworld-core.svg" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/syncworld-core.svg" width="1200" height="1590" alt="보정 C_s의 행동·영상 쌍, 이력 H_t의 접근·집기·들기, 미래 명령 A_t의 이동·그리퍼 닫기 예시를 넣은 개념도. 입력은 배포 시 가중치가 고정된 SyncWorld를 조건화한다. 샘플은 AI 생성 설명용 이미지이며 파란색 목표 자세는 실행 전 계획이다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">핵심 개념 · 새 문맥, 고정된 모델</span>보정은 새 설정의 행동–영상 대응을 입력 문맥으로 알려준다. 가중치 θ는 배포 시 바꾸지 않는다. C_s는 12방향 보정, H_t는 이미 일어난 관찰·행동, A_t는 예측할 미래 명령이다. 보정의 행동·영상 쌍, 이미 수행한 접근·집기·들기, 실행 전 명령을 구분하며, 미래 행동의 파란색 목표 자세는 계획을 뜻한다. 세 입력 카드의 샘플은 역할을 설명하는 AI 생성 예시이며 논문 데이터가 아니다. <span class="figure-links"><a href="/assets/reviews/syncworld/syncworld-core.svg" target="_blank" rel="noopener noreferrer">크게 보기</a></span></figcaption>
</figure>

## 2. 관련 연구 (Related Work)

### 2.1 제어 가능한 행동 조건 월드 모델 (Controllable Action-Conditioned World Models)

저자들은 영상으로 로봇 궤적을 만들거나 행동을 추론하는 연구, 영상·행동을 공동 학습하는 WAM, 행동을 조건으로 미래 관찰을 예측하는 모델을 구분한다. Ray map·시각 행동 프롬프트·몸체 마스크처럼 행동을 영상에 연결하는 표현도 선행한다. SyncWorld의 초점은 이 표현들을 대체하는 데 있기보다, 카메라·제어 규약·몸체마다 달라지는 수치 행동의 의미를 보정 문맥으로 지정하는 데 있다. [원문 §2.1](https://arxiv.org/html/2609.09155v1#S2.SS1)

### 2.2 잠재 행동 월드 모델링 (Latent Action World Modeling)

잠재 행동(latent action)은 라벨 없는 영상의 변화를 설명하는 제어 변수를 학습한다. 이 변수로 목표 지향 롤아웃을 만들거나 정책을 증류할 수 있지만 실제 저수준 명령으로의 연결이 필요하다. 저자들은 실제 명령을 직접 입력하고 설정별 대응을 문맥에서 읽는 방식을 이 흐름과 대비한다. [원문 §2.2](https://arxiv.org/html/2609.09155v1#S2.SS2)

### 2.3 월드 모델의 문맥 내 적응 (In-Context Adaptation of World Models)

기존 적응법에는 기하·외부 파라미터 추정, 배포 시의 추가 감독, 테스트 시 최적화·미세조정, 실제 롤아웃을 이용한 폐루프 보정이 있다. SyncWorld는 짧은 행동·영상 쌍을 접두 문맥으로 주고 파라미터 갱신 없이 현재 대응을 추론한다. [원문 §2.3](https://arxiv.org/html/2609.09155v1#S2.SS3)

### 리뷰 보충: 비교 범위를 정하는 선행 모델

IRASim은 프레임별 행동 대응, WorldGym은 생성 환경에서의 정책 평가, Ctrl-World는 다중 시점·이력과 새 카메라의 제로샷 평가를 다룬다. [IRASim §3](https://arxiv.org/html/2406.14540v2), [WorldGym §3–4](https://arxiv.org/html/2506.00613v3), [Ctrl-World §5.4](https://arxiv.org/html/2510.10125v3)

**리뷰 해석.** 새로움은 설정 전이 자체의 최초 달성보다, 설정별 행동 의미를 알려 주는 명시적 보정 문맥과 이력 증류의 결합에서 평가하는 편이 타당하다.

## 3. 방법 (Method)

아래 네 하위 절은 원문의 문제 정의·학습·의사결정·데이터 순서를 따른다.

### 3.1 보정 기반 월드 모델링 (Calibration-based World Modeling)

시각 보정은 고정된 카메라에서 로봇이 기본 방향으로 움직이는 모습을 수집하는 과정이다. 위치와 회전의 여섯 자유도(degrees of freedom, DoF)에 대해 양·음 방향 구간을 추출해, 총 12개 구간을 정해진 순서로 문맥 앞에 놓는다. 제어 공간은 위치 3개, 회전 3개, 그리퍼 1개의 차원으로 구성된다. 그리퍼는 영상에서 의미를 직접 읽을 수 있다는 이유로 방향 보정에서 제외한다. [SyncWorld §3.1, 식 4, 부록 B.2.2–B.2.3](https://arxiv.org/html/2609.09155v1#S3.SS1)

원문 식 3의 조건부 분포를 같은 기호로 쓰면 다음과 같다.

$$
I_{t+1:t+H}\sim W_{\theta}\!\left(\cdot\mid\mathcal{C}^{s},H_t,A_t\right).
$$

$\mathcal C^s$는 설정 $s$의 보정 문맥, $H_t$는 현재까지의 관찰·행동 이력(interaction history), $A_t$는 앞으로 실행할 행동(future actions)이다. $W_\theta$는 미래 영상의 조건부 분포를 나타내며, $\theta$는 배포 시 고정된 모델 파라미터다. 여기서 제로샷은 새 설정에 맞춰 파라미터를 다시 학습하지 않는다는 의미다. [SyncWorld §3.1, 식 3](https://arxiv.org/html/2609.09155v1#S3.SS1)

<figure class="review-figure" id="figure-syncworld-architecture">
<a href="/assets/reviews/syncworld/paper-figure-2.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-2.webp" width="1600" height="704" alt="SyncWorld 구조: 보정 에피소드·상호작용 이력·미래 행동이 자세 임베딩과 영상 잠재 표현으로 변환되어 DiT의 조건이 되고 미래 영상이 생성된다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 2 · 보정에서 미래 영상까지</span>왼쪽의 보정·이력·미래 행동을 입력으로, 가운데 DiT를 생성 본체로, 오른쪽을 예측 결과로 읽는다. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-2.webp" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2609.09155v1#S3.F2">Yuncong Yang et al., 2026, 원문 Figure 2</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 내용·배치 유지, 해상도 축소·WebP 압축, 자르기 없음</span></figcaption>
</figure>

그림의 세 입력은 역할이 다르다. **보정**은 이 설정에서 행동과 화면 변화의 대응을 알려주고, **이력**은 이미 진행된 상호작용을 담으며, **미래 행동**은 앞으로 실행할 명령이다. 자세는 벡터 임베딩(vector embedding)으로, 영상은 압축된 잠재 표현(latent representation)으로 처리된다. 확산 트랜스포머(diffusion transformer, DiT)는 이 조건을 이용해 잡음 상태의 미래 영상 표현을 정제한다. 그림의 미래 잡음은 생성할 프레임의 출발점이다. [원문 Figure 2, §3.1, 부록 B.1.1](https://arxiv.org/html/2609.09155v1#S3.F2)

### 3.2 시각 보정으로 월드 모델 학습하기 (Learning World Model with Visual Calibration)

저자는 보정을 문맥에 넣기만 하면 모델이 보정을 읽지 않고 고정된 좌표 규약을 외우는 지름길을 택할 수 있다고 보고, 두 가지 학습 설계를 더한다. 첫째, 학습 중 보정·이력·미래 행동에 같은 좌표 변환(x·y·z 부호 반전, 축 순서 교환, 이동 축 배율)을 적용해, 숫자 자체보다 문맥 안의 대응을 읽도록 유도한다. 둘째, 보정을 포함한 입력과 보정을 뺀 이력 중심 입력의 예측을 맞추는 **보정–이력 증류**(calibration-to-history distillation)를 쓴다. 따라서 "추론 보정 없음" 결과도 보정으로 학습한 모델의 결과이며, 학습에서 보정을 뺀 모델은 §4.5에서 따로 비교한다. [SyncWorld §3.2, 부록 B.1.2–B.1.3](https://arxiv.org/html/2609.09155v1#S3.SS2)

### 3.3 월드 모델 순위화를 통한 제로샷 정책 개선 (Zero-shot Policy Improvement via World-Model Ranking)

정책 개선은 별도의 질문이다. SyncWorld는 동결한 시각·언어·행동 정책(vision-language-action policy, VLA) π0에서 8개 행동 후보를 뽑고, 각 후보의 두 시점 미래 영상을 생성한다. GPT-5 시각·언어 모델(vision-language model, VLM)이 과제 진행과 물리적 타당성을 평가해 후보를 선택한다. 이는 생성 예측 제어(Generative Predictive Control, GPC)의 후보 순위 방식인 GPC-Rank를 이용한 구성이다. [SyncWorld §3.3, 부록 C.4.1–C.4.2](https://arxiv.org/html/2609.09155v1#S3.SS3), [GPC §III–IV](https://arxiv.org/html/2502.00622v4)

<figure class="review-figure" id="figure-syncworld-policy">
<a href="/assets/reviews/syncworld/paper-figure-4.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-4.webp" width="1800" height="710" alt="원문 Figure 4. VLA가 행동 후보를 만들고 SyncWorld가 후보별 두 시점 영상을 예측한다. VLM이 지시와 예측 결과로 점수를 매기고 최고 점수 후보를 실행한다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 4 · 예측 영상에서 행동 후보 선택까지</span>왼쪽에서 오른쪽으로 후보 생성·미래 영상·점수 평가를 읽는다. 같은 색의 행이 같은 행동 후보다. 오른쪽 숫자는 후보 평가 점수다. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-4.webp" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2609.09155v1#S3.F4">Yuncong Yang et al., 2026, Figure 4</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 내용·배치 유지, 해상도 축소·WebP 압축, 자르기 없음</span></figcaption>
</figure>

그림은 후보 세 줄을 예시로 보여주지만 실제 평가 구성은 8후보·후보당 16행동·두 시점이다. 기본 정책은 LeRobot의 비공식 π0 LIBERO 체크포인트로 과제별 미세조정 없이 쓰며, 고른 16개 행동은 다음 호출 전까지 open-loop로 실행한다. GPT-5에는 생성 영상에서 뽑은 대표 프레임 3개를 제공한다. 저자도 이 실험의 결과가 롤아웃 시뮬레이터의 품질과 순위화 절차의 품질이라는 두 요인에 달려 있다고 밝힌다. [원문 Figure 4, §3.3, 부록 C.4.1–C.4.3·Table 7](https://arxiv.org/html/2609.09155v1#S3.F4)

### 3.4 학습 데이터 (Training Data)

학습 자료는 주로 RLBench·RoboCasa·RoboMimic에서 카메라를 바꾸고 행동을 재생해 만든 시뮬레이션 데이터다. 성공 궤적(successful trajectory)에서 벗어난 행동의 결과도 포함하며, 보정이 없는 DROID 실물 데이터도 사용한다. 전체는 약 166시간(실물 약 54시간)이고 궤적마다 평균 3.6개 시점으로 기록되며, 보정 감독은 주로 시뮬레이션 자료에서 온다. [SyncWorld §3.4, 부록 B.2](https://arxiv.org/html/2609.09155v1#S3.SS4)

## 4. 실험 (Experiments)

저자는 보지 못한 환경·시점·몸체에서의 미래 예측, 시점 간 3D 일관성, 학습 없는 정책 개선의 세 실험을 한다.

### 4.1 실험 설정 (Experiment Setups)

평가 환경은 모두 학습에 없던 것이다. LIBERO와 ManiSkill은 학습 데이터에 있던 Franka Panda 몸체를 쓰고, 실물 평가는 학습에 없던 xArm을 써서 새 몸체 일반화를 시험한다. LIBERO와 ManiSkill은 각각 50개 궤적, 실물은 25개 궤적이며, 궤적당 두 시점으로 각각 100·100·50개 평가 영상을 만든다. 모델은 512×512 해상도로 16개 행동(약 1초)에 대응하는 미래 구간을 예측한다. [SyncWorld §4.1](https://arxiv.org/html/2609.09155v1#S4.SS1)

SyncWorld는 Wan2.2-TI2V-5B 체크포인트의 모든 블록에 자세 조건 모듈을 더하고 잡음 제거 모델 전체를 미세조정(fine-tuning)한다. 추론은 20단계 flow-matching 스케줄, 행동 분류기 없는 안내(classifier-free guidance, CFG) 강도 5.0, 4-GPU 통합 시퀀스 병렬화(Unified Sequence Parallelism, USP)로 한다. 비교 모델 IRASim·WorldGym·Ctrl-World는 같은 후속 과제(downstream task) 데이터로 미세조정하고 같은 held-out 궤적에서 평가한다. [부록 B.1.1, C.1–C.2](https://arxiv.org/html/2609.09155v1#A3.SS1)

### 4.2 월드 모델 품질 (World Model Quality Analysis)

영상 지표는 서로 다른 질문에 답한다. PSNR은 픽셀 평균제곱오차에 기반한 복원 점수(dB, 높을수록 좋음), SSIM은 국소 밝기·대비·구조의 유사도(높을수록 좋음)다. LPIPS는 학습된 시각 특징의 거리(낮을수록 좋음), FID는 실제·생성 이미지 특징 분포의 평균·공분산 차이(낮을수록 좋음)를 측정한다. 앞의 세 지표는 예측 프레임과 같은 시각의 실제 프레임 사이 유사도를, FID는 집합 수준 분포 차이를 본다. [원문 §4.1–4.2, Table 1](https://arxiv.org/html/2609.09155v1#S4.T1)

**Table 1의 LIBERO 열 발췌.** 저자는 SyncWorld가 추론 보정 없이도 세 영역 12개 열 모두에서 기준선보다 좋고, 보정을 쓰면 12개 열 모두 최고라고 보고한다. ManiSkill과 실물(xArm)의 LPIPS도 Ctrl-World의 0.178·0.142에 비해 보정 없이 0.071·0.042, 보정 사용 0.049·0.039다.

| 모델 | PSNR ↑ (dB) | SSIM ↑ | LPIPS ↓ | FID ↓ |
| --- | --- | --- | --- | --- |
| IRASim | 23.5 | 0.876 | 0.149 | 19.9 |
| WorldGym | 24.1 | 0.882 | 0.142 | 17.4 |
| Ctrl-World | 24.8 | 0.892 | 0.137 | 16.5 |
| SyncWorld, 추론 보정 없음 | 27.9 | 0.933 | 0.049 | 8.7 |
| SyncWorld, 추론 보정 사용 | 28.3 | 0.935 | 0.035 | 7.0 |

SyncWorld 두 행은 같은 모델의 두 추론 조건이다. **리뷰 해석.** 기준선과의 격차에는 모델 구조·사전학습의 차이도 섞일 수 있지만, §4.5에서 학습 보정을 뺀 SyncWorld의 LIBERO LPIPS(0.134)가 Ctrl-World(0.137)와 비슷해지므로 큰 격차의 상당 부분은 보정 학습과 함께 나타난다.

<figure class="review-figure" id="figure-syncworld-qualitative">
<a href="/assets/reviews/syncworld/paper-figure-5.webp" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/syncworld/paper-figure-5.webp" width="2400" height="893" alt="두 카메라 시점의 실제 로봇 장면을 시간 순서로 나열해 실제 관찰, Ctrl-World와 SyncWorld의 미래 예측을 비교한 논문 원본 그림." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 Figure 5 · 실제 로봇의 예측 비교</span>왼쪽·오른쪽은 서로 다른 카메라 시점이며, 각 묶음의 행은 위에서부터 실제 관찰(Ground-truth), Ctrl-World, SyncWorld다. 가로 방향으로 시간에 따른 변화를 읽고 팔·그리퍼 위치를 비교한다. 학습에 없던 xArm 몸체와 새 시점에서 저자가 고른 정성 예시다. <span class="figure-links"><a href="/assets/reviews/syncworld/paper-figure-5.webp" target="_blank" rel="noopener noreferrer">크게 보기</a> · <a href="https://arxiv.org/html/2609.09155v1#S4.F5">Yuncong Yang et al., 2026, 원문 Figure 5</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 내용·배치 유지, 해상도 축소·WebP 압축, 자르기 없음</span></figcaption>
</figure>

### 4.3 다중 시점 공간 일관성 (Multi-view Spatial Consistency Analysis)

시점 간 일관성(cross-view consistency)은 한 시점에서 생성한 롤아웃을 같은 궤적의 다른 시점 실제 영상과 비교해 잰다. Met3R은 DUSt3R로 두 영상의 3D 점지도를 예측해 DINO 특징을 3D로 옮긴 뒤, 겹치는 영역의 특징 코사인 유사도로 점수를 매긴다. 단위 없는 점수이며 낮을수록 좋다. 오라클(oracle)은 두 실제 시점 사이의 Met3R로, 저자는 이를 시점 간 일치의 경험적 상한으로 둔다. 평균은 LIBERO·ManiSkill·실물 세 영역의 보고 평균이다. [§4.3, 부록 C.3](https://arxiv.org/html/2609.09155v1#A3.SS3)

**원문 Table 2 전체 행 — 시점 간 일관성(Met3R, 낮을수록 좋음).**

| 모델 | LIBERO ↓ | ManiSkill ↓ | 실물 ↓ | 평균 ↓ |
| --- | --- | --- | --- | --- |
| IRASim | 0.579 | 0.633 | 0.468 | 0.560 |
| WorldGym | 0.583 | 0.632 | 0.515 | 0.577 |
| Ctrl-World | 0.559 | 0.642 | 0.494 | 0.565 |
| SyncWorld, 추론 보정 없음 | 0.545 | 0.607 | 0.474 | 0.542 |
| SyncWorld, 추론 보정 사용 | 0.540 | 0.602 | 0.473 | 0.538 |
| Oracle | 0.515 | 0.594 | 0.459 | 0.523 |

추론 보정을 쓰면 평균이 0.542에서 0.538로 낮아진다. 저자는 SyncWorld가 세 영역 모두에서 모든 기준선보다 낮은 Met3R을 얻고 오라클에 다가간다고 쓰지만(§4.3, Table 2 캡션), 같은 표의 실물 열은 IRASim의 0.468을 굵게 표시하고 SyncWorld는 0.474·0.473이다. [SyncWorld §4.3, Table 2](https://arxiv.org/html/2609.09155v1#S4.T2) **리뷰 해석.** 인쇄된 값으로는 실물 열에서 IRASim이 가장 낮으므로 "세 영역 모두"라는 서술은 LIBERO·ManiSkill에만 맞고, 평균 우위(0.538 대 0.560)라는 결론은 유지된다.

### 4.4 제로샷 정책 개선 (Zero-shot Policy Improvement with SyncWorld)

**원문 Table 3 — 오라클 개선 여지가 있는 세 과제.** 원문의 성공 비율을 %로 변환했고, 보정 없는 행도 함께 옮겼다.

| 방법 | BBQ Sauce | Orange Juice | Black Bowl |
| --- | --- | --- | --- |
| 직접 π0 실행 | 52% | 56% | 48% |
| SyncWorld, 추론 보정 없음 | 54% | 68% | 58% |
| SyncWorld, 추론 보정 사용 | 58% | 72% | 60% |
| 실제 시뮬레이터로 후보 평가 | 60% | 80% | 66% |

성공률은 과제당 50회 실행에서 성공한 비율이다. 성공 횟수로 바꾸면 직접 실행→보정 없음→보정 사용은 BBQ Sauce 26→27→29회, Orange Juice 28→34→36회, Black Bowl 24→29→30회다. 후보 순위화 전체의 이득은 Orange Juice가 8회로 가장 크고, 추론 보정의 추가분은 세 과제에서 2·2·1회다. 저자는 보정이 정책 개선 결과를 일관되게 높이고 SyncWorld가 오라클 상한에 다가간다고 해석한다. 세 과제는 실제 시뮬레이터로 후보를 평가했을 때 직접 정책보다 좋아지는 과제로 골랐으며, 저자는 SyncWorld의 성능으로 고른 것이 아니라고 밝힌다. [SyncWorld §4.4, Table 3, 부록 C.4.3](https://arxiv.org/html/2609.09155v1#A3.SS4.SSS3)

**원문 Table 8 — 오라클 이득이 작거나 없는 추가 과제.** 저자가 투명성을 위해 부록에 둔 과제이며, 같은 방식으로 원문의 비율을 %로 옮겼다.

| 방법 | Alphabet Soup | Ketchup | Put Cream Cheese |
| --- | --- | --- | --- |
| 직접 π0 실행 | 58% | 80% | 34% |
| SyncWorld | 58% | 72% | 34% |
| 실제 시뮬레이터로 후보 평가 | 62% | 74% | 30% |

Ketchup에서는 실제 시뮬레이터도 직접 실행보다 6%포인트 낮고, Put Cream Cheese에서는 4%포인트 낮다. 정확한 미래 관찰을 제공해도 평가기가 고른 후보가 더 나쁠 수 있다. 저자는 이런 과제에서 VLM/GPC-Rank 순위화 신호가 병목이라 시뮬레이터 비교에는 정보가 적다고 해석한다. [원문 부록 C.4.3, Table 8](https://arxiv.org/html/2609.09155v1#A3.T8)

### 4.5 분석: 학습 보정과 증류를 제거하면? (Analysis)

Table 4–5는 앞의 추론 보정 비교와 달리 **학습 단계의 보정·증류**를 각각 제거한다. Table 4는 세 영역의 영상 지표, Table 5는 Met3R이며, 아래는 LIBERO의 두 지표와 Met3R 세 영역 평균을 옮긴 것이다.

| 학습 구성 | LIBERO LPIPS ↓ | LIBERO FID ↓ | Met3R 세 영역 평균 ↓ |
| --- | --- | --- | --- |
| 전체 모델 | 0.049 | 8.7 | 0.542 |
| 보정 제거 | 0.134 | 15.6 | 0.566 |
| 증류 제거 | 0.148 | 17.8 | 0.574 |

전체 모델 행은 Table 1의 **추론 보정 없음** 행과 같다. 저자는 학습에서 보정을 빼면 일반화가 뚜렷이 나빠지고(clear generalization degradation), 보정은 두고 증류만 빼면 주로 추론 때 보정을 쓸 수 없을 때 성능이 떨어진다(mainly hurts performance when test-time calibration is unavailable)고 해석한다. 보정은 영역 간 행동 정렬을 개선하고 증류는 그 능력을 보정 없는 배포로 옮긴다는 것이 저자의 정리다. [§4.5, Tables 4–5](https://arxiv.org/html/2609.09155v1#S4.SS5)

저자는 또 SyncWorld가 세 실험 모두에서 추론 보정 없이도 보정 사용과 가까운 성능을 내며, 좌표 증강과 증류 학습 덕분에 이력이 암묵적 보정 신호로 작동한다고 본다. 다만 부록 F.5는 학습 때 보지 못한 몸체에서 추론 보정이 없으면 행동 차원별 움직임 방향의 대응이 덜 일관되어 정성적 제어성이 떨어진다고 덧붙인다. 부록 D의 증강 확률 비교(추론 보정 사용)에서는 기본값 0.6이 0·0.3·0.9보다 좋다(LIBERO LPIPS 0.035 대 0.055·0.041·0.053). [Appendix F.5](https://arxiv.org/html/2609.09155v1#A6.SS5), [Appendix D, Table 9](https://arxiv.org/html/2609.09155v1#A4)

## 5. 결론 (Conclusion)

저자의 결론은 설정별 행동–영상 대응을 고정된 전역 의미로 강제하는 대신, 짧은 시각 보정과 이력으로 조건화하면 새로운 카메라·단일 팔 몸체에서도 제어 가능한 예측을 만들 수 있다는 것이다. 증류는 명시적 보정이 없는 추론으로 이 능력을 연결하고, 후보 순위 실험은 그 예측이 의사결정에 쓰일 가능성을 보여준다. [원문 §5](https://arxiv.org/html/2609.09155v1#S5)

## 부록 C.2·E: 실행 비용과 실패 사례

**원문 Table 6 — 생성 시간과 계산 조건.** 비교 모델과 SyncWorld의 모든 실행 조건을 함께 옮긴다.

| 모델 | GPU 구성 | Denoising step | 생성 시간 |
| --- | --- | --- | --- |
| WorldGym | A100 1장 | 50 | 37.2초 |
| Ctrl-World | H100 1장 | 50 | 4.0초 |
| SyncWorld | H100 1장 | 50 | 39.4초 |
| SyncWorld | H100 1장 | 20 | 15.6초 |
| SyncWorld | H100 4장 | 20 | 3.2초 |

Table 6은 영상 한 번의 생성 시간이다. 모든 실험의 추론 설정인 H100 네 장·20단계에서는 3.2초, H100 한 장·20단계에서는 15.6초다. H100 한 장·50단계에서는 Ctrl-World가 4.0초, SyncWorld가 39.4초다. 정책 개선 실험은 결정 단계마다 8개 후보의 두 시점 영상, 즉 영상 16개를 생성하고 GPT-5로 평가한다. [SyncWorld 부록 C.2, Table 6](https://arxiv.org/html/2609.09155v1#A3.SS2)

저자가 보여주는 실패에는 극단적인 카메라 시점의 불안정한 움직임, 보지 못한 물체의 흐림과 위치 오류, 여러 물체 중 일부의 환각, 학습에 없던 xArm의 정밀 회전 중 흐림이 있다. [SyncWorld 부록 E.1–E.4](https://arxiv.org/html/2609.09155v1#A5)

## 리뷰 해석과 남은 질문

**리뷰 해석.** 가장 강한 근거는 같은 백본 안에서 학습 보정과 증류를 각각 뺀 §4.5 절제다. 학습 보정을 빼면 세 영역의 영상 지표 12개가 모두 나빠지고 Met3R 평균도 0.542에서 0.566으로 오르며, LIBERO LPIPS는 기준선 Ctrl-World와 비슷한 수준까지 올라간다. 추론 때 보정을 주는 효과(LIBERO LPIPS 0.049→0.035)도 영상 지표에서 분명하지만, 학습 보정을 뺄 때의 변화(0.049→0.134)보다 작다.

이 근거가 닿는 범위에는 제약이 있다. 첫째, 절제는 영상 지표만 보고하며 반복 실행의 분산이 없고, 증류를 뺀 모델을 추론 보정과 함께 평가한 값도 없어 "주로 추론 보정이 없을 때 성능이 떨어진다"는 저자 해석은 표로 직접 확인되지 않는다. 둘째, 새 몸체는 실물 xArm 25개 궤적에서만 시험되며, 부록 F.5가 밝히듯 이 경우 추론 보정이 없으면 제어성이 떨어진다. 셋째, 정책 개선은 오라클 이득이 있는 세 과제만 본문에 보고하고 추론 보정의 추가분이 과제당 1–2회 성공이라, "일관된 향상"을 실행 간 변동과 구분하기 어렵다. Table 8처럼 평가기 신호가 약한 과제에서는 실제 시뮬레이터로도 이득이 없으므로, 정책 이득은 후보를 평가하는 VLM의 품질에도 달려 있다. 부록 E의 실패 네 유형 중 둘이 물체 상호작용(보지 못한 물체, 여러 물체)에서 나온다는 점도 팔 움직임의 제어성과 물체 상호작용의 정확성을 따로 볼 이유다.

최근 연구는 SyncWorld가 함께 다룬 설정 변화를 나눠 본다. XEWorld는 통제된 장면에서 양팔 몸체 전이를 외형·운동학·물체 상태로 나눠 평가하고, 행동 표기 연구는 같은 정보를 담은 절대·상대 명령도 모델의 결과를 바꿀 수 있음을 조사한다. 두 연구는 몸체·행동 공간이나 평가 방식(고정 카메라, 오프라인 대리 평가)이 SyncWorld와 달라 직접 비교는 아니다. 계산 비용 쪽에서는 Think Like a World Model, Act Like a VLA(Cosmos3-Nano의 이해 표현을 작은 정책에 증류)와 RoboActualizer(동결한 V-JEPA 2.1 위에 미래·행동 전문가를 학습하고 실행 때 미래 출력을 버림)처럼 배포 때 영상 롤아웃을 생성하지 않는 접근이 있지만, 둘 다 플랫폼별 학습을 거친다. ReWAM의 특징 보정은 이름은 비슷하지만 SyncWorld의 행동–영상 보정 에피소드와 다른 개념이다. [XEWorld §3, §5–6](https://arxiv.org/html/2608.05799v1), [Robot World Models Are Not Invariant to How the Actions Are Written §3–6](https://arxiv.org/html/2609.23252v1), [표현 증류 §III-B–III-C, §IV](https://arxiv.org/html/2609.24682v2), [RoboActualizer §3–4](https://arxiv.org/html/2609.36413v3), [ReWAM §3.2–3.4, §4.3](https://arxiv.org/html/2609.38163v1)

남은 질문은 다음과 같다.

1. 카메라만 바꾸는 경우, 정보를 보존하는 행동 표기만 바꾸는 경우, 몸체만 바꾸는 경우에 보정의 효과가 각각 유지되는가?
2. 보정 방향이나 순서를 바꾸고, 물체 상호작용 오류를 따로 측정하면 모델이 어떤 문맥을 이용하는지 구분할 수 있는가?
3. 후보·평가기 조건과 전체 계산 예산을 맞추면 영상 시뮬레이터를 사용하는 방식과 표현을 정책에 증류하는 방식의 이득은 어떻게 달라지는가?

## 출처와 읽은 범위

- 원문: *SyncWorld: Visual Calibration Enables World Models as Zero-Shot Simulators* · Yuncong Yang et al. · 최초 공개 2026-09-08 · [검토 버전](https://arxiv.org/abs/2609.09155v1) · 검토일 2026-10-23.
- 읽은 범위: 본문 §1–5와 부록 A–F. 보충 영상은 제외한다.

| 추가 출처 | 읽은 범위 |
| --- | --- |
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

- 그림 출처: Figure 2(대표)·4·5는 저자 v1 원문 자산으로 CC BY 4.0. 내용·배치를 유지하고 해상도 축소·WebP 압축을 적용한다. 자체 개념도는 설명용 AI 생성 샘플을 포함하며 SVG의 삽입 래스터를 최적화한다. [원문](https://arxiv.org/abs/2609.09155v1).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·보충 영상.
