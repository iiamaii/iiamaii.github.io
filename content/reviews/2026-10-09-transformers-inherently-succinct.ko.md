---
title: "Transformer는 같은 규칙을 왜 더 작게 표현할 수 있을까"
description: "Transformers are Inherently Succinct를 계산 이력과 작은 검증기의 관점에서 읽는다. 지수·이중 지수 크기 차이의 증명, UHAT의 가정, 검증 난이도와 부록 식의 문제를 정리한다."
date: "2026-10-09"
updatedAt: "2026-10-23T08:56:13+00:00"
publishedAt: "2026-10-09T00:06:00+09:00"
topics: ["language models", "transformer theory", "formal languages", "computational complexity", "formal verification"]
translationKey: "transformers-inherently-succinct"
paperTitle: "Transformers are Inherently Succinct"
paperPublishedDate: "2025-10-22"
authors: ["Pascal Bergsträßer", "Ryan Cotterell", "Anthony W. Lin"]
year: "2026"
paperUrl: "https://arxiv.org/abs/2510.19315v3"
visibility: "public"
lang: "ko"
thumbnail: "/assets/reviews/transformers-inherently-succinct/transformers-core.svg"
thumbnailAlt: "주소가 붙은 긴 계산 이력에서 수평 이웃과 같은 주소의 이전 행을 attention으로 찾아 작은 UHAT 검증기로 검사하는 구조와 비교 표현의 최악 경우 크기 차이."
---

이 논문은 단일 위치 선택형 어텐션 Transformer(unique-hard attention transformer, UHAT)가 같은 언어를 인식하는 선형 시제 논리(linear temporal logic, LTL)·고정 정밀도 순환 신경망(recurrent neural network, RNN)보다 지수적으로, 유한 오토마타보다 이중 지수적으로 작은 표현을 가질 수 있음을 보인다. 긴 계산 이력을 입력에 두고 작은 attention 검증기로 관계를 검사하는 구성이 하한을 만든다. [원문 Theorems 15·17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9) **리뷰 해석.** 표현력과 기술 길이를 분리한 결과이며, 실용 LLM의 추론 속도를 비교할 때는 모델 가정을 다시 맞춰야 한다.

**대표 그림을 읽는 법.** 왼쪽의 긴 입력에서 필요한 위치를 찾고, 오른쪽의 작은 검증기가 그 관계를 확인하는 구조다. 위 그림은 논문의 증명을 설명하기 위해 직접 제작한 도식이다. [§3–4](https://arxiv.org/pdf/2510.19315v3#page=6) · [크게 보기](/assets/reviews/transformers-inherently-succinct/transformers-core.svg).

1. **칸 안의 숫자는 열 주소다.** `00·01·10·11`은 십진수 0·1·2·3에 해당하는 2비트 주소다. 두 줄은 연속한 두 행이며, 다음 행에서 같은 주소를 다시 사용한다. `tile t`와 `tile u`는 각 행의 타일을 나타내는 간략한 표기다. 같은 행의 타일이 모두 같다는 뜻은 아니다. 실제 증명은 N비트 주소로 $2^N$개 열을 다루며, 그림에는 네 열만 보였다.
2. **두 종류의 화살표는 비교할 정보를 연결한다.** `neighbor`는 같은 행의 `01` 칸에서 `10` 칸으로 이웃 타일의 정보를 전달해 가로 색을 비교한다. `same address`는 현재 행의 `10` 칸이 직전 행의 `10` 칸 정보를 가져와 세로 색을 비교하는 관계다. Attention은 현재 칸에서 필요한 과거 위치를 찾고, 그림의 화살표는 찾은 정보가 현재 검사에 쓰이는 방향으로 그렸다. 선 길이는 실행 시간이나 실제 토큰 거리가 아니다.
3. **오른쪽 상자는 반복해서 쓰는 검증 규칙이다.** 그림에서 왼쪽은 주어진 데이터, 오른쪽은 그 데이터가 유효한지 판단하는 UHAT이다. 각 위치에서 같은 검사 규칙을 공유하므로 칸마다 새로운 규칙이나 가중치를 추가할 필요가 없다. 입력과 위치별 중간 표현을 보관하는 메모리는 여전히 필요하다. 이웃 검사 외에도 입력 형식·주소 순서·경계·종료 조건이 필요하며, 아래에서 다섯 검사 부분으로 설명한다.
4. **아래 두 상자는 같은 문제를 표현하는 비용의 차이다.** 특정 언어족에서는 같은 판정 규칙을 LTL·고정 정밀도 RNN으로 적으면 지수적으로, 유한 오토마타로 적으면 이중 지수적으로 큰 표현이 필요하다는 결과다. 상자의 면적은 그 비율을 재현하지 않으며, 학습 정확도나 추론 속도를 비교한 그래프도 아니다. [Theorem 15·17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9)

그림의 문구 <strong>“Long history. Small verifier.”</strong>는 긴 입력을 짧은 입력으로 바꾼다는 뜻이 아니다. 긴 이력은 그대로 두되, 그 이력을 검사하는 모델의 기술을 작게 유지한다는 뜻이다. 이 구분을 기억하면 뒤의 크기 하한과 검증 난이도를 같은 관점에서 읽을 수 있다.

원문 §1–6 순서로 재구성했다. §3에서 비공집합성의 타일링 구성을 먼저 이해하고, §4에서 같은 구성을 크기 하한에 사용하며, §5에서 동치성 검증으로 확장한다. 부록 식의 문제는 저자의 결과 설명 뒤에 리뷰 점검으로 구분한다.

## 1. 도입 (Introduction)

저자는 표현력 계층만으로 설명되지 않는 **같은 언어의 기술 비용**을 비교한다. 이미 알려진 UHAT–LTL의 언어 클래스 대응이 작은 표현으로의 변환까지 보장하는지 묻고, 긴 계산 이력을 검사하는 작은 모델로 크기 차이를 보인다. 학습이나 실용 벤치마크가 아니라 형식 모델의 존재·하한 정리가 중심이다. [원문 §1](https://arxiv.org/pdf/2510.19315v3#page=1)

## 2. 배경 정의 (Preliminaries)

이론에서 비교할 대상과 크기를 먼저 고정한다.

### 2.1 선형 시제 논리 (Linear Temporal Logic)

선형 시제 논리(linear temporal logic, LTL)는 문자열의 위치에 붙은 명제를 이전·이후 위치와 연결하는 논리다. 논리식이 참인 문자열의 집합이 그 식의 언어다. 유한 문자열에 대한 의미론과 위치·마스킹 규칙을 고정해야 UHAT과 같은 언어를 인식한다는 비교가 성립한다. 이 논문은 그 의미론 아래 표현력과 기술 길이를 구분한다. [§2.1](https://arxiv.org/pdf/2510.19315v3#page=3)

### 2.2 단일 위치 선택형 Transformer (Unique-hard attention transformers)

대상은 **단일 위치 선택형 어텐션 Transformer**(unique-hard attention transformer, UHAT)이다. Attention 점수가 가장 높은 위치 하나의 벡터를 가져오고, 동점이면 정해진 동점 처리(tie-breaking) 규칙으로 하나를 고른다. 여러 위치의 값을 가중 평균하는 softmax attention과는 계산 방식이 다르다. [§2.2](https://arxiv.org/pdf/2510.19315v3#page=4)

| 요소 | 이 논문의 설정 | 읽을 때 주의할 점 |
| --- | --- | --- |
| Attention | 최고 점수의 한 위치 선택 | softmax 모델 전체에 결론을 옮길 수 없다 |
| 위치 정보 | 별도 위치 임베딩(positional embedding) 없음 | RoPE 기반 LLM과 동일한 모델이 아니다 |
| 마스크·동점 | 과거/미래 접근 제한, 왼쪽/오른쪽 우선 | 바로 이전 유효 위치를 찾을 수 있는지가 달라진다 |
| 연산·정밀도 | affine 변환과 ReLU; 상한은 유리수 가중치까지 | 하한은 입력 길이에 무관한 고정 정밀도 정수로도 구성한다고 서술한다 |
| 입력 길이 | 유한하지만 미리 정한 상한 없음 | 고정 context에서의 정확도 실험과 다른 문제다 |

과거를 읽는 엄격 마스크는 $j < i$다. 자기 위치를 제외하므로 일반적인 자기 위치 포함 인과 마스크(causal mask)와 구분해야 한다. 또 논문의 “future masking”은 미래를 **가린다**는 뜻이다. 이때 동점 중 가장 오른쪽을 고르면 조건을 만족하는 가장 최근 과거 위치를 찾을 수 있다. [§2, p.8 Corollary 11·14](https://arxiv.org/pdf/2510.19315v3#page=8)

### 2.3 Boolean RASP

B-RASP는 각 위치의 Boolean 값과 다른 위치를 찾는 attention 선택을 순서대로 정의하는 프로그램이다. 검증 조건을 논리 프로그램으로 먼저 적고 attention 층으로 옮기는 중간 표현 역할을 한다. 선택 가능한 후보가 없을 때의 기본값과 동점 처리까지 의미론에 포함한다. 아래 Lemma 9의 다항 변환은 특정 형태의 검색 조건에 대한 것이므로 모든 Boolean 프로그램의 보편적 다항 compiler로 확대하지 않는다. [§2.3, Lemma 9](https://arxiv.org/pdf/2510.19315v3#page=4)

### 2.4–2.5 RNN, 크기 척도와 간결성

이 글에서 **형식 언어**(formal language)는 문장들의 의미 체계가 아니라, 조건을 만족하는 문자열의 집합이다. 모델은 문자열 하나를 받아 이 집합에 속하는지 판정한다. 예를 들어 “주소가 0부터 순서대로 증가하며, 이웃 타일의 색이 맞는가?”도 하나의 언어 인식 문제다.

비교할 축은 두 가지다.

| 질문 | 비교하는 것 |
| --- | --- |
| 표현력(expressivity) | 어떤 종류의 언어를 인식할 수 있는가? |
| 간결성(succinctness) | **같은 언어**를 인식하는 데 최소한 얼마나 큰 표현이 필요한가? |

더 많은 언어를 인식할 수 있는 모델이 공통 과제에서도 항상 더 작은 것은 아니다. 논문의 UHAT은 특정 설정에서 정규 언어의 일부인 **star-free 언어**(star-free language)를 인식하지만, 더 넓은 정규 언어(regular language)를 다루는 고정 정밀도 순환 신경망(fixed-precision recurrent neural network, RNN)보다 같은 규칙을 훨씬 작게 표현할 수 있다. 여기서 star-free는 합집합·연결·보수 등으로 정의할 수 있는 정규 언어의 부분류다. 세부 분류보다 중요한 것은 **인식 가능한 집합과 표현 크기의 순위가 다를 수 있다**는 점이다. [§1–2, Proposition 1, Definition 2–3](https://arxiv.org/pdf/2510.19315v3#page=3)

크기는 파라미터 개수만 세지 않는다. 구조와 가중치를 이진수로 기술하는 데 필요한 길이를 센다. RNN이 한 좌표에 사용하는 정밀도도 비용에 포함한다. 무한 정밀도의 숫자 하나에 정보를 숨겨 놓고 “작다”고 부르는 비교를 피하기 위해서다. [§2.4–2.5](https://arxiv.org/pdf/2510.19315v3#page=5)

## 3. 비공집합성으로 가장 작은 증거의 크기 이해하기 (The size of smallest witness via non-emptiness problem)

Theorem 4의 질문은 모델이 받아들이는 문자열이 **하나라도 있는가**다. §3의 하한은 $2^N$ 너비 타일링의 EXPSPACE-완전성(Proposition 7)을 Lemma 8의 B-RASP 검증기로, 다시 Lemma 9의 다항 크기 UHAT으로 옮긴다. 유효한 타일링과 수용 문자열의 양방향 대응이 필요하므로 주소·경계·인접 검사 중 하나라도 빠지면 환원이 완성되지 않는다.

상한은 다른 방향이다. Proposition 12가 UHAT 계산값의 비트 길이를 모델 기술 크기의 다항식으로 제한한다. 이때 가능한 값의 종류는 지수적으로 많지만 유한해 열거할 수 있다. Proposition 13은 이를 사용해 동작을 지수 크기의 LTL 식으로 옮긴다. LTL 비공집합성을 식 크기의 다항 공간으로 검사하므로, 원래 UHAT 크기에서는 지수 공간이다. 하한과 이 상한을 합해 EXPSPACE-완전성이 된다. [§3, Theorem 4, Propositions 7·10·12–13; Appendix A.1](https://arxiv.org/pdf/2510.19315v3#page=6)

### 타일링에서 B-RASP와 UHAT으로: Lemmas 8–9

증명은 긴 계산을 한 번에 수행하는 대신, 이미 주어진 계산 이력이 올바른지 검사하는 문제로 바꾼다. 너비가 $2^N$인 타일링(tiling) 표를 만들고 각 칸에 N비트 열 주소를 붙인다. 각 행은 주소 0에서 시작해 마지막 주소까지 진행한 뒤 다음 행으로 넘어간다. 이를 한 줄의 문자열로 읽는다. [§3, Lemma 8; Appendix A.2](https://arxiv.org/pdf/2510.19315v3#page=14)

```text
N = 2인 설명용 예

행 r:      00 t₀ #   01 t₁ #   10 t₂ #   11 t₃ #
행 r+1:    00 u₀ #   01 u₁ #   10 u₂ #   11 u₃ #

가로 검사: u₁의 오른쪽 색 = u₂의 왼쪽 색인가?
세로 검사: t₂의 위쪽 색 = u₂의 아래쪽 색인가?
```

가로 이웃은 직전 칸을 찾으면 된다. 세로 이웃은 **같은 주소를 가진 가장 최근 과거 칸**을 찾으면 된다. 두 행 사이의 거리를 모델 안에 거대한 상수로 적는 대신, 주소의 비트 일치와 최근 위치 선택을 이용한다. 행을 아래에서 위로 쌓는 타일링 표기이므로 위 예의 세로 검사는 이전 행 타일의 위쪽과 현재 행 타일의 아래쪽을 비교한다. [Appendix A.2, 식 (17)](https://arxiv.org/pdf/2510.19315v3#page=16)

검증기는 다섯 부분으로 나뉜다.

| 부분 | 검사할 조건 |
| --- | --- |
| A | 비트·타일·구분자가 정해진 형태로 배치되는가? |
| B | 주소가 0부터 증가하고 마지막에서 0으로 돌아가는가? |
| C | 마지막 주소와 종료 타일이 맞는가? |
| D | 표의 네 경계에서 색 조건이 맞는가? |
| E | 가로·세로 이웃의 색이 맞는가? |

이 조건들을 불리언 프로그램(Boolean program)인 **B-RASP**로 적고 UHAT으로 옮긴다. 다항 크기 변환은 모든 B-RASP 프로그램에 대한 주장이 아니라, 이 구성에서 쓰는 특별한 조건 검색에 대한 결과다. Attention은 유효한 후보가 없어도 최고 점수 위치를 고를 수 있으므로, 선택한 뒤 조건을 다시 확인하고 실패하면 기본값을 사용해야 한다. [Lemma 9; Appendix A.3](https://arxiv.org/pdf/2510.19315v3#page=17)

여기서 절약하는 것은 **검증 규칙의 기술 길이**다. 긴 이력 자체는 여전히 입력에 존재한다. 작은 은닉 벡터 하나에 모든 계산을 저장하거나, 이중 지수 길이의 출력을 빠르게 생성한다는 뜻은 아니다.

### Theorem 4의 상·하한을 연결하기

논문은 크기 차이의 하한만 보이지 않는다. **모든 UHAT을 지수 크기의 LTL로 바꾸는 직접 변환**도 제시한다. 반대 방향인 LTL→UHAT은 다항 크기로 가능하다. 두 방향을 혼동하면 핵심 결론이 뒤집힌다. [Proposition 13·16](https://arxiv.org/pdf/2510.19315v3#page=8)

상한(upper bound)의 직관은 hard attention이 이미 존재하는 벡터 **하나를 가져온다**는 데 있다. 입력의 모든 위치를 더해 값이 길이에 따라 계속 커지는 연산이 아니다. 모델 구조와 유리수 가중치가 정해지면, 각 층에서 등장할 값의 비트 길이를 모델 크기의 다항식으로 제한할 수 있다. 그 값들을 열거하고 “이 위치가 이 벡터를 가지는가?”를 논리식으로 표현한다. [Proposition 12; Appendix A.4–A.5](https://arxiv.org/pdf/2510.19315v3#page=18)

UHAT→LTL 뒤에 표준 변환을 붙이면 **비결정적 유한 오토마타**(nondeterministic finite automaton, NFA)의 이중 지수 상한을 얻는다. 오토마타 하한은 NFA와 결정적 유한 오토마타(deterministic finite automaton, DFA)에 모두 적용되지만, 위 경로만으로 DFA에도 같은 이중 지수 상한이 따라오지는 않는다. NFA를 결정화하는 데 추가 비용이 들 수 있기 때문이다. [§4, Theorem 17](https://arxiv.org/pdf/2510.19315v3#page=9)

마스크와 동점 처리도 결과를 바꾼다. 과거만 읽고 가장 최근 후보를 택하는 설정에는 어려운 타일링 검사를 담을 수 있다. 반면 과거만 읽고 가장 오래된 후보를 택하는 제한에 대해서는 논문이 **NEXP 상한**을 제시한다. 이는 NEXP 완전성까지 보였다는 뜻이 아니다. [Corollary 11·14](https://arxiv.org/pdf/2510.19315v3#page=8)

## 4. 표현 사이의 간결성 (Succinctness across representations)

중심 도구는 **가장 짧은 수용 문자열**(shortest accepted string)이다. 작게 적은 검증기가 아주 긴 문자열만 받아들인다면, 다른 표현도 그 긴 문자열을 요구할 만큼 커야 하는지 물을 수 있다.

논문은 $2^n$비트 카운터를 끝까지 증가시키는 계산을 사용한다. n=3이면 8비트 카운터이고 가능한 값은 256개다. 일반적으로는 $2^{2^n}$개의 값이 생긴다. 이 계산 이력을 검증하는 UHAT은 n과 타일 집합의 기술에 대해 다항 크기로 만들면서, 가장 짧은 수용 문자열조차 이중 지수 길이가 되게 구성한다. [§4, Theorem 15](https://arxiv.org/pdf/2510.19315v3#page=9)

$$
\begin{aligned}
\text{verifier description length}&=\operatorname{poly}(n),\\
\text{shortest accepted string length}&\ge2^{2^n}.
\end{aligned}
$$

비교 대상들은 작은 표현일 때 짧은 수용 문자열이 존재한다는 성질을 가진다. 이 성질과 위 구성을 결합한다.

| 비교 표현 | 작은 표현에서 가능한 짧은 수용 문자열 | 논문이 제시한 최소 크기 차이 |
| --- | --- | --- |
| 선형 시제 논리(linear temporal logic, LTL) | 논리식 크기에 대해 지수 길이 이하 | UHAT보다 지수적으로 커야 하는 언어족 존재 |
| 유한 오토마타 | q개 상태이면 q보다 짧은 수용 경로 존재 | 이중 지수적으로 많은 상태가 필요한 언어족 존재 |
| 고정 정밀도 RNN | k비트 × D차원이면 상태 수 ≤ $2^{kD}$ | 정밀도를 포함한 기술 크기에 지수 차이 |

LTL은 “이전에 어떤 조건이 있었는가”, “어떤 조건이 성립할 때까지 유지되는가”처럼 위치 사이의 시간적 관계를 쓰는 논리다. 유한 오토마타는 현재 상태와 다음 기호로 상태를 갱신하는 유한 상태 기계다. 같은 상태가 반복된 수용 경로에서 중간 순환을 지울 수 있다는 성질이 오토마타의 하한(lower bound)에 쓰인다. [§2; Theorem 15·17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9)

**수량자**(quantifier)가 중요하다. 결과는 특정 언어족이 존재한다는 주장이다. 그 언어족을 인식하는 모든 비교 표현(representation)이 커야 한다는 것이지, 모든 과제에서 Transformer가 작다는 주장은 아니다. 또 지수는 우선 언어족의 인덱스 n을 기준으로 얻는다. UHAT 크기가 n의 다항식이면, 모델 크기에 대한 지수의 형태도 그 다항식에 따라 달라진다. “항상 정확히 모델 크기 s에 대해 $2^s$배 차이 난다”는 식으로 읽으면 안 된다. [Definition 2, Theorem 15](https://arxiv.org/pdf/2510.19315v3#page=5)

### Theorems 15·17, Corollary 18의 크기 하한

**하한 계산의 중간 단계.** 아래는 Theorems 15·17 및 Corollary 18의 논리를 펼친 리뷰 보충이다. $L_n$은 논문이 구성한 언어족, $\varphi_n$은 같은 언어의 LTL 식, $q_n$은 유한 오토마타의 상태 수다. 상수 $c>0$는 LTL의 짧은 증거 상한에 해당한다. RNN은 좌표당 $k$비트, 은닉 차원 $D$라서 상태가 최대 $2^{kD}$개다.

$$
\begin{aligned}
\min_{w\in L_n}|w|&\geq 2^{2^n},\\
2^{2^n}\leq 2^{c|\varphi_n|}
&\Longrightarrow |\varphi_n|\geq 2^n/c,\\
q_n&>2^{2^n},\\
2^{kD}\geq q_n&\Longrightarrow kD>2^n.
\end{aligned}
$$

LTL은 크기 $|\varphi_n|$에 지수적인 길이의 수용 문자열을 갖는다는 small-model 성질 때문에 첫 하한이 나온다. $q_n$개 상태의 오토마타가 문자열을 받아들이면, 최단 수용 경로에는 같은 상태가 반복되지 않아 길이는 $q_n$보다 작다. 이 사실이 이중 지수 상태 하한을 준다. RNN의 유한 상태화를 Proposition 1로 연결하면 그 상태 수를 만들 정밀도·차원의 곱에도 지수 하한이 필요하다. 지수의 기준은 우선 $n$이며 UHAT 크기는 $\operatorname{poly}(n)$라는 점을 유지해야 한다. [§4, Theorems 15·17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9)

### Proposition 16: 반대 방향 변환

LTL에서 UHAT으로는 다항 크기 변환이 가능하다. 이는 앞 절의 UHAT→LTL 지수 변환과 방향이 다르다. [§4, Proposition 16; Appendix A.6](https://arxiv.org/pdf/2510.19315v3#page=20)

## 5. 응용 (Applications)

작은 표현은 거대한 동작 집합을 압축한다. 그렇다면 두 모델이 **모든 길이의 모든 문자열**에서 같은 답을 내는지 확인하는 일은 얼마나 어려울까?

논문은 UHAT의 비공집합성(non-emptiness), 즉 “받아들이는 문자열이 하나라도 있는가?”와 두 모델의 언어 동치성(language equivalence)에 대해 **EXPSPACE 완전성**(EXPSPACE-completeness)을 제시한다. 모델의 이진 기술 길이를 입력 크기로 볼 때, 최악 경우 지수 공간을 요구하는 복잡도 계층과 연결된다는 뜻이다. 문자열 하나의 순전파가 지수 공간을 쓴다는 뜻은 아니다. [Theorem 4·19; Appendix A.1·A.7](https://arxiv.org/pdf/2510.19315v3#page=10)

**Theorem 19의 증명.** 하한은 공언어를 인식하는 고정 UHAT과의 동치성 검사로 옮긴다. 주어진 모델이 그 모델과 동치일 필요충분조건은 자신의 언어가 공집합인 것이다. Theorem 4의 비공집합성 결과와 결정적 공간 클래스의 여집합 닫힘을 이용한다. 상한은 두 UHAT을 Proposition 13으로 각각 LTL 식으로 바꾸고 두 식의 동치성을 다항 공간으로 검사한다. 식 크기가 원래 모델 크기의 지수이므로 전체는 지수 공간이다. 이것은 개별 문자열 forward의 실행 복잡도가 아니라 **모든 문자열에 대한 모델 간 관계**를 결정하는 복잡도다. [§5, Theorem 19; Appendix A.7](https://arxiv.org/pdf/2510.19315v3#page=21)

## 6. 맺음말 (Concluding remarks)

저자는 실제 검증 기법을 적용할 여지와 큰 카운터를 표현하지 못하는 쉬운 부분류를 찾는 방향을 남긴다. 간결한 Transformer가 학습으로 발견되는지, fixed-precision softmax·average-hard attention에서도 같은 크기 차이가 성립하는지는 별도 후속 문제다. 이 논문에는 데이터셋·학습 정확도 비교 실험이 없으므로 그런 표를 이론 결과의 근거로 추가하지 않는다. [원문 §6](https://arxiv.org/pdf/2510.19315v3#page=10)

이 결과로 실용 검증을 포기할 이유는 없다. 길이를 제한하거나, 더 단순한 모델 부분류를 고르면 다른 문제가 된다. 저자들도 큰 카운터를 표현하지 못하는 부분류를 찾는 방향을 논의한다. [§6](https://arxiv.org/pdf/2510.19315v3#page=10)

## 부록 A.2에 대한 리뷰 점검: 식 (14e)의 AND

저자들은 아래 구성에 대해 주소가 올바르게 증가할 때 $B(L)=1$이라고 주장한다. 부록 A.2의 식 (14e)는 과거에 오류가 없을 때, 현재 주소가 정상 증가했는지와 0으로 순환했는지를 **동시에** 요구한다. PDF와 TeX에 같은 연결자가 있다.

<figure class="review-figure" id="paper-equation-14e">
<a href="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png" width="2400" height="108" alt="원문 부록 A.2 식 (14e). 기본값 부분에서 순환 조건 B₁→₀(i)와 증가 조건 B₊₁(i)를 논리곱으로 연결한다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 발췌 · 주소 검사 식 (14e)</span>오른쪽 기본값의 ∧를 본다. 정상적인 한 번의 주소 전이는 증가 또는 순환 중 하나다. 앞쪽의 과거 오류 검색 조건과 구분해서 읽어야 한다. <span class="figure-links"><a href="https://arxiv.org/pdf/2510.19315v3#page=15">Bergsträßer et al., arXiv v3, p.15</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원문 식만 자름, 식·폰트 수정 없음 · <a href="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png">크게 보기</a></span></figcaption>
</figure>

식을 왼쪽부터 읽어 보자. $Q_{\#}(j)$는 j가 구분자 `#`인 위치라는 뜻이고, $j < i$는 현재 위치보다 앞선 곳만 찾는 조건이다. 검은 삼각형의 검색은 그중 **순환도 정상 증가도 아닌 과거 전이**를 찾는다. 그런 오류가 발견되면 0을 반환한다. 오류가 없으면 콜론 오른쪽의 기본값으로 현재 전이를 검사한다. 여기서 $B_{+1}$은 보통의 +1 증가, $B_{1\to0}$는 마지막 주소에서 0으로 돌아가는 순환이며 첫 주소가 0인지 확인하는 역할도 한다. [식 (14c–e)](https://arxiv.org/pdf/2510.19315v3#page=15)

따라서 이 그림에서 살펴볼 연결자는 **맨 오른쪽 기본값의 ∧ 하나**다. 앞쪽 검색 조건의 ∧는 “순환도 아니고 증가도 아니다”라는 오류 조건을 묶는 데 필요하다. 두 위치를 함께 OR로 바꾸는 수정은 이 글에서 검토한 후보가 아니다.

가장 작은 사례는 N=1, 경계 색이 모두 0인 타일 t 하나로 만든 `0t#1t#`다. 주소는 정상적으로 0→1로 증가한다. 마지막 위치에서 조건을 나눠 보면 문제가 드러난다.

| 마지막 위치의 조건 | 값 |
| --- | --- |
| 정상 증가 $B_{+1}$ | 참 |
| 1→0 순환 $B_{1\to0}$ | 거짓 |
| 원문 기본값: 두 조건의 AND | 거짓 → 정상 입력 거부 |
| 수정 후보: 두 조건의 OR | 참 |

이 두 조건은 N≥1에서 동시에 참일 수 없다. 전체 검증 결과에 B가 필수이므로, 인쇄된 구성은 이 정상 타일링을 받아들이지 못한다. [Appendix A.2, 식 (14c–e)](https://arxiv.org/pdf/2510.19315v3#page=15)

주소 순서만을 다루는 [검사 코드](https://github.com/iiamaii/iiamaii.github.io/blob/main/research/checks/transformers-inherently-succinct/appendix-counter-check.py)는 이 국소 논증의 보조 자료다.

**리뷰 해석.** 단순 표기 오류일 가능성은 있지만 저자의 확인을 받지는 않았다. A–E 전체, B-RASP→UHAT 변환, 정밀도 구현을 검증하지 않았으므로 OR가 전체 증명을 복구한다고 단정할 수 없다. 의존 경로는 Lemma 8의 타일링 검증→Lemma 9의 UHAT 변환→Theorem 4의 하한이며, 같은 구성을 사용하는 간결성 하한과 Theorem 19의 하한도 재확인이 필요하다. Proposition 13의 LTL 변환에 기반한 상한은 이 카운터에 의존하지 않는다. 기본값 AND를 OR로 바꾸는 수정의 국소적 필요성은 높게 확신하지만, 전체 구성의 복구와 정리의 진위는 별도 문제다.

<span id="리뷰-해석과-후속-검증" aria-hidden="true"></span>

## 리뷰 해석과 남은 질문

**리뷰 해석.** 이 논문의 가장 유용한 관점은 attention을 **위치 검색을 공유하는 작은 규칙**으로 보는 것이다. 긴 표 전체를 하나의 순환 은닉 상태에 기억하는 대신, 입력의 위치별 표현에서 현재 검사에 필요한 이전 위치를 찾아 관계를 확인한다. 그 결과 표현력 계층만으로는 보이지 않던 표현 비용의 차이를 드러낸다. 이는 증명 구성을 바탕으로 한 리뷰의 해석이다.

동시에 세 가지를 분리해야 한다. 작은 가중치가 **존재한다**는 것, 학습이 그 가중치를 **찾는다**는 것, 실제 기기에서 **빠르게 실행된다**는 것은 별도 주장이다. 이 논문은 첫 번째를 다룬다. 고정 정밀도 softmax와 average-hard attention의 간결성도 §6에서 후속 과제로 남긴다. 따라서 이 결과를 곧바로 “실제 LLM이 RNN보다 항상 효율적”이라는 설명으로 사용하기는 어렵다. [§6](https://arxiv.org/pdf/2510.19315v3#page=10)

가장 먼저 할 후속 검증은 주소 검사 수정 후보를 포함한 **전체 타일링 검증기와 독립 오라클(oracle)의 비교**다. 그다음에 특수 B-RASP→UHAT 변환이 후보 부재와 동점을 올바르게 처리하는지 확인해야 한다. 국소 사례 검사를 전체 정리 인증으로 확대하지 않는 순서다.

배경과 경계를 보려면 다음 세 자료가 도움이 된다. 전수 문헌 조사나 최신성 순위는 아니다.

| 자료 | 연결되는 질문 | 읽은 범위 |
| --- | --- | --- |
| [Yang et al., NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/13d7f172259b11b230cc5da8768abc5f-Abstract-Conference.html) | UHAT·star-free·LTL의 표현력 관계는 무엇인가? | 초록, PDF §5.1–5.2, Theorem 5·6 |
| [Jerad et al., ACL 2025, Unique Hard Attention: A Tale of Two Sides](https://aclanthology.org/2025.acl-short.76/) | 왼쪽/오른쪽 동점 처리가 왜 중요한가? | 초록, §2–3, Table 1 |
| [Yang et al., 2026, Length Generalization Bounds for Transformers](https://arxiv.org/abs/2603.02238v2) | 정밀도를 어디에 적용하느냐가 길이 일반화(length generalization) 경계를 어떻게 바꾸는가? | 초록, §5의 정리·명제 진술과 논의; 전체 증명 검토 아님 |

마지막 연구는 attention 내부의 정밀도, BOS/EOS 등 가정이 중요하다. 그 결과를 이 글의 무위치 임베딩 UHAT에 그대로 대입할 수는 없다. 다음 비교에서는 모델 가정을 먼저 맞춰야 한다.

## 출처와 읽은 범위

- 원문: *Transformers are Inherently Succinct* · Pascal Bergsträßer et al. · 최초 공개 2025-10-22 · [검토 버전](https://arxiv.org/abs/2510.19315v3) · 검토일 2026-10-23.
- 읽은 범위: v3 본문 §1–6·Appendix A의 관련 구성과 증명. 참조 타일링 정리 전체와 전체 compiler의 형식 인증은 제외한다.
- 그림 출처: 대표 개념도는 리뷰 자체 제작물이며 논문 실험 결과를 그린 그래프가 아니다. [원문](https://arxiv.org/abs/2510.19315v3).
- 확인하지 않은 것: 연구 코드 실행·모델 학습/재현·원시 데이터 불확실성 추정·전체 형식 인증.
