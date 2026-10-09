# 논문 통계 데이터

```sh
npm run stats:new -- analysis-id
```

`content/statistics/analysis-id.json`을 비공개로 생성한다. [템플릿](../assets/statistics-analysis.json)의 예시 문구·주소를 실제 분석으로 교체하고, `null`을 확인한 숫자로 채운다. 공개하려면 검증 후 `visibility`를 `public`으로 바꾼다. 비공개 자료는 공개 저장소에 커밋하지 않는다. 기본 템플릿은 코드 검사용 예시이며 연구 데이터가 아니다.

## 공통 필드

| 필드 | 규칙 |
| --- | --- |
| `schemaVersion` | 현재 `1` |
| `id` | 영문 소문자·숫자·하이픈, 공개 분석 사이에서 유일. 발행 후 유지 |
| `visibility` | `private` / `public`. 생략하면 비공개 |
| `updatedAt` | 실제 작성·재분석 시각. 초와 시간대가 있는 ISO 8601. 미래 시각은 해당 시점 이후 재빌드해야 노출 |
| `title`, `summary`, `notes` | 모두 `{ "ko": "한글", "en": "English" }`. notes에 버전·데이터셋·조건·집계 방식·한계를 기록 |
| `paperSlugs` | 관련 공개 리뷰의 `translationKey` 배열. 외부 논문만 분석하면 빈 배열 가능 |
| `sources` | 하나 이상. `{ "url": "https://…", "label": {"ko":"…","en":"…"}, "location": {"ko":"Table 2…","en":"Table 2…"} }` |
| `charts` | 하나 이상의 차트. id는 분석 내에서 유일. title·description은 한영 객체 |

URL은 HTTP(S)만 허용하고 모든 문구는 실행되지 않는 일반 텍스트로 표시한다. 공개 데이터의 잘못된 숫자·존재하지 않는 노드·비공개 리뷰 연결은 빌드를 실패시킨다. 단위가 같아도 평가 조건이 다른 값을 직접 순위로 비교하지 않는다. 평균·변환값은 계산식을 남기며 반올림된 원문 평균은 그대로 옮긴다. 유의성·인과 관계를 분석하지 않았다면 주장하지 않는다.

## 막대그래프 `type: "bar"`

- `unit`: 한영 단위. 무단위 지표도 명시한다.
- `direction`: `higher` / `lower` / `none` (높을수록 좋음 / 낮을수록 좋음 / 단순 빈도·크기).
- `values`: 1–100개 `{ "label": {"ko":"…","en":"…"}, "value": 0.538 }`.
- 유한한 숫자만 허용한다. 누락을 0으로 바꾸지 않는다. 미확인 항목은 차트에서 제외하고 notes에 남긴다. 음수·0도 지원하며 축은 0을 포함한다.

## 관계 그래프 `type: "network"`

```json
{
  "id": "shared-topics",
  "type": "network",
  "title": {"ko": "주제 관계", "en": "Topic links"},
  "description": {"ko": "두 논문에 공통인 주제 수", "en": "Count of shared topics"},
  "relationship": {"ko": "무방향 · 공통 주제 수", "en": "Undirected · shared topic count"},
  "nodes": [
    {"id": "paper-a", "label": {"ko": "논문 A", "en": "Paper A"}},
    {"id": "paper-b", "label": {"ko": "논문 B", "en": "Paper B"}}
  ],
  "edges": [{"source": "paper-a", "target": "paper-b", "weight": 2}]
}
```

1–40개 노드, 연결 없는 노드도 표시한다. label은 짧게 작성한다. 노드에 선택적으로 `paperSlug`를 넣으면 공개 리뷰로 연결한다(`paperSlugs`에도 포함). 현재 지원 범위는 **무방향 가중 관계**이며 `weight`는 양수다. 자기 자신·중복 연결·없는 노드는 거부한다. 인용 방향·시간 흐름이 필요한 분석은 이 형식에 억지로 넣지 않고 새 차트 유형을 구현한다.

## 검증과 갱신

`npm test && npm run build` 후 `/statistics/`, `/en/statistics/`에서 분석 선택·두 테마·모바일·데이터 표·출처를 확인한다. 차트는 정적 HTML/SVG로 생성되므로 JavaScript 없이도 전체 분석이 보인다. 새 차트 유형은 검증기(`scripts/statistics.mjs`)와 렌더러(`scripts/render-statistics.mjs`), 테스트·이 형식을 함께 확장한다.

첫 화면의 리뷰·주제 수는 빌드 시 현재 공개 자료로 계산한다. 개별 JSON의 수치·관계는 **저장된 스냅샷**이다. 예시 `review-topic-snapshot.json`도 새 리뷰·주제를 반영하려면 다시 분석해 저장해야 한다. 버전·집계 시점이 다른 결과를 혼동하지 않도록 제목·notes와 실제 updatedAt을 함께 갱신한다.
