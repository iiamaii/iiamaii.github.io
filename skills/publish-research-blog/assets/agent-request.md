다음 문서를 이번 작업의 포스팅 지침으로 먼저 읽고 적용해.

- 저장소 접근이 있으면: `skills/publish-research-blog/SKILL.md`
- 웹 접근만 있으면: https://raw.githubusercontent.com/iiamaii/iiamaii.github.io/main/skills/publish-research-blog/SKILL.md
- 링크를 읽을 수 없으면: 내가 첨부한 `SKILL.md`를 사용해.

지침에 연결된 `research-handoff.md`, `review-body.md`(영문은 `review-body.en.md`)도 필요한 단계에서 읽어. 파일이 없거나 접근할 수 없으면 읽었다고 가정하지 말고 알려줘.

대상 블로그: `iiamaii/iiamaii.github.io`
연구 자료: {{논문 URL·PDF 또는 기존 조사 결과}}
포스트 키: {{영문 소문자·숫자·하이픈}}
주제: {{공통 topics 목록, 예: ["머신러닝", "컴퓨터 비전"] — 첫 항목은 대표 주제}}
언어: {{한글 / 영어 / 둘 다}}
작업 범위: 비공개 초안 작성

저장소와 실행 도구가 있으면 해당 규칙으로 원고를 작성·검증해. 접근이 없으면 frontmatter를 포함한 언어별 Markdown 원고와 연구 인계 자료를 제공해. 공식 출처에서 논문 최초 공개일과 전체 저자를 원문 순서대로 확인해 `paperPublishedDate`와 `authors` 목록에 기록하고 한영판의 서지 정보를 동일하게 맞춰. 날짜가 미확인이면 추정하지 말고 비공개로 유지해. 원고의 출처 절에는 실제로 확인한 출처와 읽은 범위만 현재형으로 쓰고, 읽기 회차·수행한 검사·실행하지 못한 검증 같은 작업 기록은 연구 인계에 남겨.

기본 요청은 `visibility: "private"` 원고까지야. 공개 게시를 원할 때는 내가 위 작업 범위를 “검증 후 공개 게시”로 바꿔 요청할게. 그 경우 필요한 저장소 쓰기 권한과 실행 도구로 검증·배포하고, 게시 URL과 배포 확인 결과까지 알려줘. 공개 리뷰의 주제·원문 식별자·연재·공개 상태를 바꿀 때는 `npm run stats:update`로 주제 집계를 갱신하고 원고와 함께 검증·인계해. 실험 수치 분석은 관련 원문 근거가 달라질 때 별도로 재검증해.
