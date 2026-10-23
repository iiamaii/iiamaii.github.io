---
title: "Transformer는 같은 규칙을 왜 더 작게 표현할 수 있을까"
description: "Transformers are Inherently Succinct를 계산 이력과 작은 검증기의 관점에서 읽는다. 지수·이중 지수 크기 차이의 증명, UHAT의 가정, 검증 난이도와 부록 식의 문제를 정리한다."
date: "2026-10-09"
updatedAt: "2026-10-23T15:26:14+00:00"
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

이 논문은 단일 위치 선택형 어텐션 Transformer(unique-hard attention transformer, UHAT)가 같은 언어를 인식하는 선형 시제 논리(linear temporal logic, LTL)·고정 정밀도 순환 신경망(recurrent neural network, RNN)보다 지수적으로, 유한 오토마타보다 이중 지수적으로 간결할 수 있음을 보이려 한다. 핵심은 $n$의 다항식 크기 UHAT이 가장 짧은 수용 문자열의 길이가 $2^{2^n}$ 이상인 언어를 인식하게 만드는 타일링 검증기이며, 세 크기 하한과 비공집합성·동치성의 EXPSPACE-hardness가 모두 이 구성을 거친다. [원문 §4, Theorems 15·17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9) **리뷰 해석.** UHAT→LTL 지수 변환에 기반한 상한은 이 구성과 무관하게 성립하지만, 구성의 주소 검사 식 (14e)는 인쇄된 그대로면 어떤 입력도 받아들이지 않아 인쇄본의 증명만으로는 Theorems 4·15·17·19와 Corollary 18의 하한이 성립하지 않으며, 기본값의 AND를 OR로 바꾸면 이 검사는 복구된다([아래 점검](#paper-equation-14e)).

**대표 그림을 읽는 법.** 왼쪽의 긴 입력에서 필요한 위치를 찾고, 오른쪽의 작은 검증기가 그 관계를 확인하는 구조다. 위 그림은 논문의 증명을 설명하기 위해 직접 제작한 도식이다. [§3–4](https://arxiv.org/pdf/2510.19315v3#page=6) · [크게 보기](/assets/reviews/transformers-inherently-succinct/transformers-core.svg).

1. **칸 안의 숫자는 열 주소다.** `00·01·10·11`은 십진수 0·1·2·3에 해당하는 2비트 주소다. 두 줄은 연속한 두 행이며, 다음 행에서 같은 주소를 다시 사용한다. `tile t`와 `tile u`는 각 행의 타일을 줄여 쓴 표기다. 실제 증명은 N비트 주소로 $2^N$개 열을 다루며, 그림에는 네 열만 보였다.
2. **두 종류의 화살표는 비교할 정보를 연결한다.** `neighbor`는 같은 행의 `01` 칸에서 `10` 칸으로 이웃 타일의 정보를 전달해 가로 색을 비교한다. `same address`는 현재 행의 `10` 칸이 직전 행의 `10` 칸 정보를 가져와 세로 색을 비교하는 관계다. Attention은 현재 칸에서 필요한 과거 위치를 찾고, 그림의 화살표는 찾은 정보가 현재 검사에 쓰이는 방향으로 그렸다.
3. **오른쪽 상자는 반복해서 쓰는 검증 규칙이다.** 그림에서 왼쪽은 주어진 데이터, 오른쪽은 그 데이터가 유효한지 판단하는 UHAT이다. 각 위치에서 같은 검사 규칙을 공유하므로 칸마다 새로운 규칙이나 가중치를 추가할 필요가 없다. 이웃 검사 외에도 입력 형식·주소 순서·경계·종료 조건이 필요하며, 아래에서 다섯 검사 부분으로 설명한다.
4. **아래 두 상자는 같은 문제를 표현하는 비용의 차이다.** 특정 언어족에서는 같은 판정 규칙을 LTL·고정 정밀도 RNN으로 적으면 지수적으로, 유한 오토마타로 적으면 이중 지수적으로 큰 표현이 필요하다는 결과다. 상자의 면적은 비율을 반영하지 않는다. [Theorem 15·17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9)

그림의 문구 <strong>“Long history. Small verifier.”</strong>는 긴 이력은 입력에 그대로 두고, 그 이력을 검사하는 모델의 기술만 작게 유지한다는 뜻이다.

원문 §1–6 순서를 따르고, 부록 A.2 식 (14e)의 문제는 §6 뒤의 별도 절에서 다룬다.

## 1. 도입 (Introduction)

고정 정밀도 Transformer는 가정에 따라 정규 언어의 여러 부분류를 인식하는 반면, 고정 정밀도 RNN은 모든 정규 언어를 인식하므로 표현력으로는 RNN이 더 강하다. 저자들은 표현력 대신 **간결성**(succinctness), 즉 같은 언어를 인식하는 가장 작은 표현의 크기를 비교 기준으로 제안한다. LTL이 star-free 언어와 표현력은 같지만 유한 오토마타보다 지수적으로 간결하고, 그 대가로 결정 문제가 더 어렵다는 고전적 예처럼 간결성은 결정 문제의 난이도와 맞물린다. 저자들은 UHAT이 attention으로 $0$부터 $2^{2^N}$까지 세는 카운터를 구현할 수 있다는 점을 핵심 기술로 하한을 세우고, 짝을 이루는 상한과 검증 문제의 복잡도도 함께 제시한다. [원문 §1](https://arxiv.org/pdf/2510.19315v3#page=1)

## 2. 배경 정의 (Preliminaries)

이론에서 비교할 대상과 크기를 먼저 고정한다.

### 2.1 선형 시제 논리 (Linear Temporal Logic)

원문의 LTL 식은 명제 $Q_a$(현재 기호가 $a$인가)와 불리언 연산, 과거를 보는 since·미래를 보는 until 연산자로 만들고, past·future·yesterday 같은 약어도 이 두 연산자로 정의한다. 식이 인식하는 언어는 **마지막 위치**에서 식이 참인 문자열의 집합이다. [§2.1](https://arxiv.org/pdf/2510.19315v3#page=3)

### 2.2 단일 위치 선택형 Transformer (Unique-hard attention transformers)

UHAT의 attention은 점수가 가장 높은 위치 하나의 벡터를 가져오고, 동점이면 정해진 동점 처리(tie-breaking) 규칙으로 하나를 고른다. 여러 위치의 값을 가중 평균하는 softmax attention과는 계산 방식이 다르다. [§2.2](https://arxiv.org/pdf/2510.19315v3#page=4)

| 요소 | 이 논문의 설정 | 결과와의 관계 |
| --- | --- | --- |
| Attention | 최고 점수의 한 위치 선택 | softmax·average-hard attention의 간결성은 §6의 후속 과제다 |
| 위치 정보 | 별도 위치 임베딩(positional embedding) 없이 마스킹만 사용 | 임의의 위치 인코딩을 허용한 선행 연구와 다른 설정이다(§6) |
| 마스크·동점 | 과거/미래 접근 제한, 왼쪽/오른쪽 우선 | 바로 이전 유효 위치를 찾을 수 있는지가 달라진다 |
| 연산·정밀도 | affine 변환과 ReLU; 상한은 유리수 가중치까지 | 하한은 입력 길이에 무관한 고정 정밀도 정수로도 구성한다고 서술한다 |
| 입력 길이 | 유한하지만 미리 정한 상한 없음 | 판정 문제는 모든 길이의 문자열을 대상으로 한다 |

원문의 엄격한 미래 마스킹(strict future masking)은 $j<i$, 즉 자기 위치를 뺀 과거만 보게 하며, 자기 위치를 포함하는 일반적인 인과 마스크(causal mask)와 다르다. 여기에 가장 오른쪽 동점 처리를 더하면 조건을 만족하는 가장 최근 과거 위치를 찾는다. [§2.2, §3](https://arxiv.org/pdf/2510.19315v3#page=4)

### 2.3 Boolean RASP

B-RASP는 각 위치의 Boolean 값과 다른 위치를 찾는 attention 선택을 순서대로 정의하는 프로그램으로, Yang et al.(2024)이 UHAT과 표현력이 같음을 보였다. 이 논문에서는 검증 조건을 논리 프로그램으로 먼저 적고 attention 층으로 옮기는 중간 표현 역할을 한다. 선택 가능한 후보가 없을 때의 기본값과 동점 처리까지 의미론에 포함한다. [§2.3](https://arxiv.org/pdf/2510.19315v3#page=4)

### 2.4–2.5 RNN, 크기 척도와 간결성

**형식 언어**(formal language)는 조건을 만족하는 문자열의 집합이다. 모델은 문자열 하나를 받아 이 집합에 속하는지 판정한다. 예를 들어 “주소가 0부터 순서대로 증가하며, 이웃 타일의 색이 맞는가?”도 하나의 언어 인식 문제다.

비교할 축은 두 가지다.

| 질문 | 비교하는 것 |
| --- | --- |
| 표현력(expressivity) | 어떤 종류의 언어를 인식할 수 있는가? |
| 간결성(succinctness) | **같은 언어**를 인식하는 데 최소한 얼마나 큰 표현이 필요한가? |

더 많은 언어를 인식할 수 있는 모델이 공통 과제에서도 항상 더 작은 것은 아니다. 논문의 UHAT은 특정 설정에서 정규 언어의 일부인 **star-free 언어**(star-free language)를 인식하지만, 더 넓은 정규 언어(regular language)를 다루는 고정 정밀도 RNN보다 같은 규칙을 훨씬 작게 표현할 수 있다. 여기서 star-free는 합집합·연결·보수 등으로 정의할 수 있는 정규 언어의 부분류다. 세부 분류보다 중요한 것은 **인식 가능한 집합과 표현 크기의 순위가 다를 수 있다**는 점이다. [§1–2, Proposition 1, Definition 2–3](https://arxiv.org/pdf/2510.19315v3#page=3)

크기는 파라미터 개수가 아니라 구조와 가중치의 최소 이진 기술 길이다. RNN의 정밀도 $k$는 단항(unary)으로 크기에 넣는데, 저자들은 정밀도 $k$의 Transformer를 정밀도 $2^k$의 RNN과 비교하는 일을 막기 위해서라고 설명한다. Definition 2(f-more succinct)는 증거 언어족 하나에서 비교 대상이 $f$만큼 커야 한다는 존재형 하한이고, Definition 3(g-bounded expansion)은 모든 언어에서 변환 크기가 $g$ 이하라는 보편형 상한이다. “지수적으로 더 간결”은 $f(m)\in\Omega(2^{cm^d})$, “이중 지수적으로”는 $f(m)\in\Omega(2^{2^{cm^d}})$ 꼴의 하한을 뜻하며, 두 정의를 함께 쓰면 크기 차이의 범위가 정해진다. [§2.4–2.5](https://arxiv.org/pdf/2510.19315v3#page=5)

## 3. 비공집합성으로 가장 작은 증거의 크기 이해하기 (The size of smallest witness via non-emptiness problem)

Theorem 4는 UHAT과 B-RASP 프로그램의 비공집합성, 즉 받아들이는 문자열이 **하나라도 있는가**가 EXPSPACE-완전이라는 정리다. 하한은 $2^N$ 너비 타일링 문제의 EXPSPACE-완전성(Proposition 7)을 Lemma 8의 B-RASP 검증기로, 다시 Lemma 9의 다항 크기 UHAT으로 옮긴다(Propositions 5·10). 유효한 타일링과 수용 문자열이 양방향으로 대응해야 하므로, 주소·경계·인접 검사 가운데 하나라도 틀리면 환원이 성립하지 않는다. 상한은 UHAT을 지수 크기의 LTL로 옮겨 얻으며, 두 방향을 합해 EXPSPACE-완전성이 된다. [§3, Theorem 4, Propositions 5·7·10; Appendix A.1](https://arxiv.org/pdf/2510.19315v3#page=6)

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

이 조건들을 불리언 프로그램(Boolean program)인 **B-RASP**로 적고 UHAT으로 옮긴다. 다항 크기 변환은 모든 B-RASP 프로그램에 대한 주장이 아니라, 이 구성에서 쓰는 특별한 조건 검색에 대한 결과다. Attention은 유효한 후보가 없어도 최고 점수 위치를 고를 수 있으므로, 변환은 선택한 뒤 조건을 다시 확인하고 실패하면 기본값을 쓴다. [Lemma 9; Appendix A.3](https://arxiv.org/pdf/2510.19315v3#page=17) 이 구성은 엄격한 미래 마스킹과 가장 오른쪽 동점 처리만 쓰므로, 저자들은 그렇게 제한한 UHAT에서도 비공집합성이 EXPSPACE-hard라고 적는다(Corollary 11).

### 상한: Propositions 12–13과 Corollary 14

상한의 직관은 hard attention이 이미 존재하는 벡터 **하나를 가져온다**는 데 있다. 모델 구조와 유리수 가중치가 정해지면 계산 중에 나타나는 값의 비트 길이는 모델 크기의 다항식으로 제한된다(Proposition 12). 가능한 값의 종류는 지수적으로 많지만 유한해 열거할 수 있으므로, LTL 식은 값의 계산이 아니라 마스킹과 선택 위치만 흉내 내면 된다. 이렇게 **모든 UHAT을 지수 크기의 LTL로 바꾸는 직접 변환**을 만들고(Proposition 13), LTL 비공집합성은 식 크기의 다항 공간에서 결정되므로 원래 UHAT 크기로는 지수 공간이 된다. 저자들은 이 변환이 UHAT을 B-RASP로 먼저 옮기는 Yang et al.(2024)의 이중 지수 구성을 개선한다고 쓴다. [Propositions 12–13; Appendices A.4–A.5](https://arxiv.org/pdf/2510.19315v3#page=18)

마스크와 동점 처리도 결과를 바꾼다. 엄격한 미래 마스킹에 가장 왼쪽 동점 처리(또는 엄격한 과거 마스킹에 가장 오른쪽 동점 처리)를 쓰는 UHAT은 past(또는 future) 연산자만 쓰는 LTL 식으로 옮겨지고, 그 조각의 비공집합성이 NP-완전이므로 **NEXP 상한**을 얻는다(Corollary 14). EXPSPACE-hard인 Corollary 11의 제한과는 동점 처리 방향만 다르다. [Corollaries 11·14](https://arxiv.org/pdf/2510.19315v3#page=8)

## 4. 표현 사이의 간결성 (Succinctness across representations)

저자들은 UHAT 비공집합성이 EXPSPACE-완전인 데 비해 LTL은 PSPACE-완전이라는 차이를 간결성 차이의 단서로 든다. 중심 도구는 **가장 짧은 수용 문자열**(shortest accepted string)이다. 작은 검증기가 아주 긴 문자열만 받아들인다면, 같은 언어를 인식하는 다른 표현도 그 긴 문자열을 요구할 만큼 커야 한다. [§4](https://arxiv.org/pdf/2510.19315v3#page=9)

### Theorem 15: LTL보다 지수적으로 간결하다

Theorem 15의 증거 언어족은 세 단계로 만든다. [§4, Theorem 15](https://arxiv.org/pdf/2510.19315v3#page=9)

1. 결정적 튜링 기계 $\mathcal M_n$이 빈 테이프에 $0$을 $2^n$개 쓰고, 이 $2^n$비트 이진 카운터를 $1^{2^n}$이 될 때까지 증가시킨 뒤 받아들인다. 초기화에는 $n$에 선형인 상태 수가, 증가에는 상수 크기의 기계가 필요하다. 테이프는 $n$에 지수적인 칸을 쓰고 유일한 수용 실행의 길이는 $2^{2^n}$ 이상이다. 예를 들어 $n=3$이면 8비트 카운터로 값이 256개다.
2. van Emde Boas(1997)의 환원으로 $\mathcal M_n$을 너비 $2^{p(n)}$($p$는 다항식)의 타일링 인스턴스 $\mathcal I_n$으로 바꾼다. 각 행이 기계의 한 구성을 담아 올바른 타일링이 올바른 실행에 대응하며, $\mathcal I_n$의 크기는 $n$의 다항식이고 가장 작은 올바른 타일링도 $2^{2^n}$개 이상의 행을 갖는다.
3. Lemmas 8–9로 $\mathcal I_n$의 올바른 타일링 인코딩을 인식하는 UHAT $\mathcal T_n$을 $\mathcal I_n$ 크기의 다항식으로 만든다. $\mathcal T_n$이 인식하는 언어가 $L_n$이다.

$$
\begin{aligned}
\text{verifier description length}&=\operatorname{poly}(n),\\
\text{shortest accepted string length}&\ge2^{2^n}.
\end{aligned}
$$

비교 대상들은 작은 표현일 때 짧은 수용 문자열이 존재한다는 성질을 가진다. 이 성질과 위 구성을 결합한다.

| 비교 표현 | 작은 표현에서 가능한 짧은 수용 문자열 | 논문이 제시한 최소 크기 차이 |
| --- | --- | --- |
| LTL | 논리식 크기에 대해 지수 길이 이하 | UHAT보다 지수적으로 커야 하는 언어족 존재 |
| 유한 오토마타 | q개 상태이면 q보다 짧은 수용 경로 존재 | 이중 지수적으로 많은 상태가 필요한 언어족 존재 |
| 고정 정밀도 RNN | k비트 × D차원이면 상태 수 ≤ $2^{kD}$ | 정밀도를 포함한 기술 크기에 지수 차이 |

LTL 식이 받아들이는 가장 짧은 문자열은 식 크기에 지수적인 길이 이하이므로(Vardi–Wolper식 LTL→오토마타 변환), $L_n$을 인식하는 LTL 식 $\varphi_n$은 $n$에 지수적인 크기여야 한다. 리뷰 보충으로 상수 $c>0$를 두어 펼치면 다음과 같다.

$$
2^{2^n}\leq\min_{w\in L_n}|w|\leq 2^{c|\varphi_n|}
\Longrightarrow |\varphi_n|\geq 2^n/c.
$$

이 하한은 언어족의 인덱스 $n$ 기준이다. 증명은 $|\mathcal T_n|\le c_0n^d$에서 $n\ge(|\mathcal T_n|/c_0)^{1/d}$를 얻어, $|\varphi_n|\ge c_1 2^{c_2'|\mathcal T_n|^{1/d}}$ 꼴의 모델 크기 기준 하한으로 옮긴다. Definition 2대로 결과는 이런 증거 언어족이 **존재한다**는 주장이다.

### Proposition 16: 반대 방향 변환

LTL에서 UHAT으로는 다항 시간·다항 크기 변환이 가능하다. 즉 UHAT은 LTL에 대해 다항 이내로 확장된다(polynomially bounded expansion). 저자들은 Theorem 15와 합하면 UHAT–LTL의 크기 차이가 모든 언어에서 다항 이하, 증거 언어족에서 지수 이상으로 고정된다고 정리한다. [§4, Proposition 16; Appendix A.6](https://arxiv.org/pdf/2510.19315v3#page=20)

### Theorem 17·Corollary 18: 유한 오토마타와 RNN

Theorem 17은 같은 증거 언어족을 다시 쓴다. 공집합이 아닌 언어를 인식하는 오토마타는 크기에 선형인 길이 이하의 문자열을 받아들이므로, $L_n$을 인식하는 가장 작은 오토마타는 $n$에 이중 지수적인 크기를 갖는다. Corollary 18은 정밀도 $k$·차원 $D$의 RNN을 $2^{kD}$개 상태의 오토마타로 보는 Proposition 1을 여기에 결합한다. 리뷰 보충으로 상태 수 $q_n$에 대해 펼치면, 최단 수용 경로는 같은 상태를 반복하지 않아 길이가 $q_n$보다 작으므로 다음이 나온다. [§4, Theorem 17, Corollary 18](https://arxiv.org/pdf/2510.19315v3#page=9)

$$
q_n>2^{2^n},\qquad
2^{kD}\geq q_n\Longrightarrow kD>2^n.
$$

상한도 짝을 이룬다. 저자들은 §1 각주에서 Proposition 13의 지수 크기 LTL 변환에 표준적인 지수 크기 LTL→오토마타 변환을 이어 붙여, 모든 고정 정밀도 Transformer에 이중 지수 크기의 동치 유한 오토마타가 있다고 쓴다. [§1, 각주 1](https://arxiv.org/html/2510.19315v3#footnote1) 반대 방향으로는, LTL과 동치인 counter-free 오토마타를 LTL로 옮기는 최선의 알려진 변환이 지수적이므로, Proposition 16과 합하면 star-free 언어에서 UHAT은 유한 오토마타에 대해 지수 이내로 확장된다. [§4](https://arxiv.org/pdf/2510.19315v3#page=9) **리뷰 해석.** 표준 LTL→오토마타 변환이 주는 것은 비결정적 유한 오토마타(nondeterministic finite automaton, NFA)이므로 이 이중 지수 상한은 NFA에 대한 것이고, 결정적 유한 오토마타(deterministic finite automaton, DFA)는 결정화 비용이 더 들 수 있다. 반면 Theorem 17의 하한 논증은 NFA와 DFA에 모두 적용된다.

## 5. 응용 (Applications)

작은 표현은 거대한 동작 집합을 압축한다. 그렇다면 두 모델이 **모든 길이의 모든 문자열**에서 같은 답을 내는지 확인하는 일은 얼마나 어려울까?

논문은 UHAT의 비공집합성(non-emptiness), 즉 “받아들이는 문자열이 하나라도 있는가?”와 두 모델의 언어 동치성(language equivalence)에 대해 **EXPSPACE 완전성**(EXPSPACE-completeness)을 제시한다. 모델의 이진 기술 길이를 입력 크기로 볼 때 최악 경우 지수 공간이 필요하다는 뜻이며, 저자들은 결정적 유한 오토마타에서 같은 문제가 다항 시간이라는 점과 대비한다. [Theorem 4·19; Appendix A.1·A.7](https://arxiv.org/pdf/2510.19315v3#page=10)

**Theorem 19의 증명.** 하한은 공언어를 인식하는 고정 UHAT과의 동치성 검사로 옮긴다. 주어진 모델이 그 모델과 동치일 필요충분조건은 자신의 언어가 공집합인 것이다. Theorem 4의 비공집합성 결과와 결정적 공간 클래스의 여집합 닫힘을 이용한다. 상한은 두 UHAT을 Proposition 13으로 각각 LTL 식으로 바꾸고 두 식의 동치성을 다항 공간으로 검사한다. 식 크기가 원래 모델 크기의 지수이므로 전체는 지수 공간이다. 이것은 개별 문자열 순전파의 실행 복잡도가 아니라 **모든 문자열에 대한 모델 간 관계**를 결정하는 복잡도다. [§5, Theorem 19; Appendix A.7](https://arxiv.org/pdf/2510.19315v3#page=21)

## 6. 맺음말 (Concluding remarks)

관련 연구 단락에서 저자들은 UHAT과 LTL·star-free 언어의 밀접한 관계를 보인 선행 결과들(Yang et al. 2024 등)이 간결성이나 검증의 계산 복잡도는 다루지 않았다고 정리한다. 복잡도 관점에서 가장 가까운 Sälzer et al.(2025)은 여러 정밀도의 Transformer 검증 문제를 연구해 고정 정밀도 Transformer의 검증이 적어도 NEXP-hard임을 보였다. 그 기법은 Transformer가 유한 오토마타보다 단일 지수적으로 간결할 수 있음을 함의하지만, LTL·RNN과의 비교에는 결론을 주지 않는다. 이 논문은 이를 오토마타 대비 이중 지수, LTL·RNN 대비 지수 차이로 개선한다고 쓴다. 모델도 더 단순하다. soft·hard attention을 섞은 Sälzer et al.과 달리 unique-hard attention만 쓰고, 위치 정보도 임의의 고정 정밀도 위치 인코딩 대신 위치 마스킹으로 제한한다. 위치 인코딩이 없을 때 정밀도가 유한하지만 상한이 없는 average-hard·softmax attention Transformer의 검증은 결정 불가능하다는 최근 결과도 함께 소개한다. [원문 §6](https://arxiv.org/pdf/2510.19315v3#page=10)

형식 검증 단락에서는 최악 경우 복잡도가 EXPSPACE-완전이어도 기호적 방법·시뮬레이션 같은 자동 검증 기법을 실제 Transformer 검증에 가져오는 일을 과제로 제시한다. EXPSPACE-hardness 증명에는 큰 카운터를 표현하는 Transformer가 필요하므로, 그런 카운터를 표현하지 못해 검증이 더 쉬운 부분류를 찾는 방향도 제안한다. 간결한 Transformer의 학습 가능성은 경험적 증거가 엇갈리는 열린 문제로 두고, 고정 정밀도 softmax·average-hard attention Transformer의 간결성은 후속 과제로 남긴다. [§6](https://arxiv.org/pdf/2510.19315v3#page=10)

## 부록 A.2에 대한 리뷰 점검: 식 (14e)의 AND

저자들은 주소 검사 Gadget B에 대해 “$B(L) = 1$ iff the binary blocks count up correctly”라고 쓴다. 부록 A.2의 식 (14e)는 과거에 오류가 없을 때, 현재 주소가 정상 증가했는지와 0으로 순환했는지를 **동시에** 요구한다. PDF와 TeX에 같은 연결자가 있다.

<figure class="review-figure" id="paper-equation-14e">
<a href="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png" target="_blank" rel="noopener noreferrer"><img src="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png" width="2400" height="108" alt="원문 부록 A.2 식 (14e). 기본값 부분에서 순환 조건 B₁→₀(i)와 증가 조건 B₊₁(i)를 논리곱으로 연결한다." loading="lazy" decoding="async"></a>
<figcaption><span class="figure-label">원문 발췌 · 주소 검사 식 (14e)</span>오른쪽 기본값의 ∧를 본다. 정상적인 한 번의 주소 전이는 증가 또는 순환 중 하나다. 앞쪽의 ∧는 과거 오류를 찾는 검색 조건에 속한다. <span class="figure-links"><a href="https://arxiv.org/pdf/2510.19315v3#page=15">Bergsträßer et al., arXiv v3, p.15</a> · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 원문 식만 자름, 식·폰트 수정 없음 · <a href="/assets/reviews/transformers-inherently-succinct/paper-equation-14e.png">크게 보기</a></span></figcaption>
</figure>

식을 왼쪽부터 읽어 보자. $Q_{\#}(j)$는 j가 구분자 `#`인 위치라는 뜻이고, $j < i$는 현재 위치보다 앞선 곳만 찾는 조건이다. 검은 삼각형의 검색은 그중 **순환도 정상 증가도 아닌 과거 전이**를 찾는다. 그런 오류가 발견되면 0을 반환한다. 오류가 없으면 콜론 오른쪽의 기본값으로 현재 전이를 검사한다. 여기서 $B_{+1}$은 보통의 +1 증가, $B_{1\to0}$는 마지막 주소에서 0으로 돌아가는 순환이며 첫 주소가 0인지 확인하는 역할도 한다. [식 (14c–e)](https://arxiv.org/pdf/2510.19315v3#page=15)

따라서 이 그림에서 살펴볼 연결자는 **맨 오른쪽 기본값의 ∧ 하나**다. 앞쪽 검색 조건의 ∧는 “순환도 아니고 증가도 아니다”라는 오류 조건을 묶는 데 필요하다.

기본값의 두 조건은 $N\ge1$에서 동시에 참일 수 없다. $B_{+1}(i)$는 어떤 비트 $B_k(i)$가 1이기를 요구하고, $B_{1\to0}(i)$는 값 쪽이든 기본값 쪽이든 모든 비트 $B_k(i)$가 0이기를 요구하기 때문이다. 따라서 인쇄된 식에서는 과거 오류가 있으면 검색 결과로, 없으면 기본값으로 모든 입력에서 $B(L)=0$이 되고, 출력 $Y(L)=A(L)\wedge B(L)\wedge C(L)\wedge D(L)\wedge E(L)$도 늘 0이다. 인쇄된 Lemma 8의 프로그램은 어떤 타일링 인스턴스에 대해서도 공언어를 인식한다. [Appendix A.2, 식 (14c–e)](https://arxiv.org/pdf/2510.19315v3#page=15)

가장 작은 예는 N=1, 경계 색이 모두 0인 타일 t 하나로 만든 `0t#1t#`다. 주소는 정상적으로 0→1로 증가하지만, 마지막 위치에서 조건을 나눠 보면 다음과 같다.

| 마지막 위치의 조건 | 값 |
| --- | --- |
| 정상 증가 $B_{+1}$ | 참 |
| 1→0 순환 $B_{1\to0}$ | 거짓 |
| 원문 기본값: 두 조건의 AND | 거짓 → 정상 입력 거부 |
| 수정 후보: 두 조건의 OR | 참 |

주소 순서만을 다루는 [검사 코드](https://github.com/iiamaii/iiamaii.github.io/blob/main/research/checks/transformers-inherently-succinct/appendix-counter-check.py)는 이 논증의 보조 자료다.

**리뷰 해석.** 인쇄본에서는 Lemma 8의 프로그램이 늘 공언어를 인식하므로, 이 프로그램을 거치는 Propositions 5·10과 Theorem 4의 하한, 같은 증거 언어족을 쓰는 Theorems 15·17과 Corollary 18, Theorem 4에 기대는 Theorem 19의 하한은 인쇄된 증명으로는 성립하지 않는다. Propositions 12·13·16과 Corollary 14의 상한은 이 구성에 의존하지 않는다. 기본값의 AND를 OR로 바꾸면 Gadget B는 첫 블록이 0이고 이후 블록이 증가 또는 순환하는지를 의도대로 검사하므로 단순 표기 오류일 가능성이 높지만, 저자 확인은 없다. 나머지 gadget과 Lemma 9의 변환, 정밀도 처리는 확인하지 않았으므로 수정한 구성 전체가 정리를 복구하는지는 열려 있다. 수정의 필요성은 높게 확신한다.

<span id="리뷰-해석과-후속-검증" aria-hidden="true"></span>

## 리뷰 해석과 남은 질문

**리뷰 해석.** 이 논문의 강점은 상·하한을 짝으로 제시한 데 있다. LTL 쪽은 Proposition 13의 지수 상한과 Theorem 15의 지수 하한이, 오토마타 쪽은 이중 지수 상한과 Theorem 17의 하한이 맞물려, 표현력 계층으로는 보이지 않던 크기 차이의 규모를 정한다. 증명이 보여 주는 관점도 유용하다. attention을 **위치 검색을 공유하는 작은 규칙**으로 쓰면 긴 표 전체를 순환 은닉 상태 하나에 담지 않고, 입력의 위치별 표현에서 필요한 이전 위치를 찾아 관계를 확인할 수 있다. 상한 쪽은 Proposition 12의 비트 길이 제한과 Proposition 13의 변환만으로 서므로 부록 A.2의 문제와 무관하다. 반면 하한 쪽은 모두 Lemma 8 하나에 기대므로, (14e)를 OR로 고친 뒤 다른 gadget과 Lemma 9의 변환까지 맞는지에 따라 Sälzer et al.(2025)보다 개선했다는 주장이 서는지가 갈린다.

결과의 범위는 작은 가중치가 **존재한다**는 데까지다. 학습이 그런 가중치를 찾는지, 고정 정밀도 softmax·average-hard attention에서도 같은 차이가 나는지는 §6이 남긴 열린 문제다.

남은 질문은 두 가지다. 첫째, 주소 검사를 고친 **전체 타일링 검증기를 독립 오라클(oracle)과 비교**해 A–E 전체가 의도한 언어를 인식하는지 확인할 수 있다. 둘째, 특수 B-RASP→UHAT 변환이 후보 부재와 동점을 올바르게 처리하는지 확인할 수 있다.

다음 세 자료는 배경과 경계를 보여 준다.

| 자료 | 연결되는 질문 | 읽은 범위 |
| --- | --- | --- |
| [Yang et al., NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/13d7f172259b11b230cc5da8768abc5f-Abstract-Conference.html) | UHAT·star-free·LTL의 표현력 관계는 무엇인가? | 초록, PDF §5.1–5.2, Theorem 5·6 |
| [Jerad et al., ACL 2025, Unique Hard Attention: A Tale of Two Sides](https://aclanthology.org/2025.acl-short.76/) | 왼쪽/오른쪽 동점 처리가 왜 중요한가? | 초록, §2–3, Table 1 |
| [Yang et al., 2026, Length Generalization Bounds for Transformers](https://arxiv.org/abs/2603.02238v2) | 정밀도를 어디에 적용하느냐가 길이 일반화(length generalization) 경계를 어떻게 바꾸는가? | 초록, §5의 정리·명제 진술과 논의; 전체 증명 검토 아님 |

마지막 자료는 원문 §6이 softmax·average-hard attention 간결성의 초기 시도로 인용한 연구로, attention 내부의 정밀도와 BOS/EOS 같은 가정에 기대므로 위치 임베딩 없는 UHAT과는 가정이 다르다.

## 출처와 읽은 범위

- 원문: *Transformers are Inherently Succinct* · Pascal Bergsträßer et al. · 최초 공개 2025-10-22 · [검토 버전](https://arxiv.org/abs/2510.19315v3) · 검토일 2026-10-23.
- 읽은 범위: v3 본문 §1–6과 부록 A.
- 그림 출처: 대표 개념도는 리뷰 자체 제작물이다. 식 (14e) 발췌는 arXiv v3 PDF(CC BY 4.0)에서 식 부분만 잘랐다. [원문](https://arxiv.org/abs/2510.19315v3).
- 확인하지 않은 것: Proposition 7이 인용한 타일링 정리의 증명, Gadget B 밖의 gadget과 부록 A.3 변환의 세부.
