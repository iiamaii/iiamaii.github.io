# 연구노트

`iiamaii`의 프로필·철학과 주제별 논문 리뷰를 모으는 GitHub Pages 블로그입니다.

- [메인](https://iiamaii.github.io/): 최신 리뷰를 상단에, 대문 이미지·프로필·철학을 하단에 표시
- [내 프로필](https://iiamaii.github.io/profile/): 소개, 철학, 연구 관심사
- [논문 리뷰](https://iiamaii.github.io/reviews/): 복수 주제·업데이트 기간 필터, 단어 검색, 정렬, 스크롤 추가 표시, 리뷰 본문
- GitHub page 메뉴: 기본적으로 [iiamaii GitHub 프로필](https://github.com/iiamaii)로 연결

기존 예시 글은 게시 목록에서 정리했습니다. 현재 [SyncWorld 리뷰](https://iiamaii.github.io/reviews/syncworld-visual-calibration/), [RealtimeWAM 리뷰](https://iiamaii.github.io/reviews/realtimewam-one-step-asynchronous/), [Looped models 고정점 리뷰](https://iiamaii.github.io/reviews/looped-models-fixed-points/) 세 편을 한글·영문으로 게시합니다. 각 글은 다섯 주제로 분류하며, 앞의 두 편은 `world action model`, 새 글은 `language models`를 첫 주제로 둡니다. 주제는 공개 포스트에서 자동으로 모읍니다. 프로필과 철학은 내용을 채우기 전까지 준비 안내를 보여줍니다.

## 미리보기

Node.js 22 이상이 필요하며 GitHub Actions는 Node.js 24를 사용합니다.

```sh
npm ci
npm run dev
```

`http://localhost:4321/`에서 확인합니다. 파일 수정 후 `npm run build`를 실행하고 브라우저를 새로고침합니다.

## 프로필·철학·대문 이미지

`site.json`에서 다음 항목을 수정합니다.

| 항목 | 용도 |
| --- | --- |
| `profile.name` | 표시 이름 |
| `profile.headline` | 짧은 소개 |
| `profile.bio` | 프로필 본문. 빈 줄로 문단 구분 |
| `profile.interests` | 연구 관심사 문자열 목록 |
| `philosophy.title`, `philosophy.text` | 메인과 프로필에 함께 표시할 철학 |
| `hero.image`, `hero.alt` | 대문 이미지 경로와 설명 |
| `logo.image`, `logo.alt` | 좌측 상단의 홈 링크 로고 |
| `homeReviewLimit` | 메인에 표시할 최신 리뷰 수. 기본 6, 양의 정수 |
| `reviewThumbnail` | 포스트 썸네일이 없을 때 사용하는 기본 이미지 |
| `githubPage` | GitHub page 메뉴의 외부 주소 |
| `title`, `description` | 블로그 이름과 검색 설명 |

이미지는 `public/assets/`에 넣고 `/assets/파일명.webp`처럼 설정합니다. 현재 대문은 이전에 생성한 추상 이미지를 사용하며, 첨부한 로고의 원본은 `public/assets/brand-logo.png`에 보관하고, 배경을 제거한 `public/assets/brand-logo-transparent.png`를 페이지 좌측 상단의 홈 링크로 사용합니다. 생성 이미지 기록은 `research/cover-artwork.json`에 있습니다.

## 한국어·영어 전환

모든 공개 페이지 상단의 **한글 / EN** 버튼으로 언어를 바꿉니다. 선택한 언어로 프로필과 리뷰 목록을 이동하며, 선택한 주제들·검색어·업데이트 기간·정렬·불러올 개수·표시한 묶음도 전환 후 유지됩니다.

| 한국어 | 영어 |
| --- | --- |
| `/` | `/en/` |
| `/profile/` | `/en/profile/` |
| `/reviews/` | `/en/reviews/` |
| `/reviews/논문-id/` | `/en/reviews/논문-id/` |

프로필·철학의 영어 문구는 `site.json`의 `translations.en.profile`, `translations.en.philosophy`에 작성합니다. 주제 표시 이름을 번역하려면 각 언어 포스트의 `topicNames`에 작성하고, 두 원고의 `topics` 목록은 동일하게 유지합니다. 영어 본문이 비어 있고 한글 원문이 있으면 원문 안내와 함께 표시합니다.

## 논문 주제

공개 포스트의 `topics` 목록을 모아 필터의 주제 체크박스 목록을 자동으로 만듭니다. 한 포스트에 **한 개 이상의 주제**를 지정할 수 있습니다. 주제를 `site.json`에 미리 등록할 필요가 없으며 한글과 공백을 사용할 수 있습니다.

```yaml
topics:
  - "머신러닝"
  - "컴퓨터 비전"
visibility: "public"
```

글은 지정한 각 주제로 필터링할 수 있으며, 메인과 리뷰 목록에는 논문당 한 번씩 표시됩니다. 전체 편수·RSS·사이트맵도 중복 계산하지 않습니다. 첫 주제는 기본 썸네일과 본문의 목록 복귀 링크에도 사용합니다. 두 언어가 있으면 한글 원고의 순서를 공통 기준으로 사용합니다.

주제 앞뒤의 공백과 연속된 공백은 정리하고, 같은 주제를 여러 번 적으면 하나로 합칩니다. 빈 목록이나 빈 주제는 오류로 알려줍니다. 주제 목록은 해당 주제의 최신 공개 글부터 표시합니다. 비공개·초안·미래 날짜의 글은 주제 생성에 영향을 주지 않으며, 마지막 공개 글을 비공개로 바꾸면 다음 배포에서 해당 주제도 사라집니다.

`topicNames`는 각 언어에서 표시할 주제 이름을 지정하는 선택 항목입니다. 영어 원고에 다음처럼 작성합니다. 키는 공통 `topics`의 주제이며, 두 언어의 주제 집합은 같아야 합니다. 번역 이름이 없으면 공통 주제 이름을 표시합니다.

```yaml
topics: ["머신러닝", "컴퓨터 비전"]
topicNames:
  "머신러닝": "Machine Learning"
  "컴퓨터 비전": "Computer Vision"
```

기존의 단일 `topic: "머신러닝"`과 `topicName: "Machine Learning"`도 계속 지원합니다. `topic`과 `topics`를 함께 쓰지 않습니다. 여러 주제를 사용하면 `topicName` 대신 `topicNames`를 사용하며, 키는 해당 글의 주제만 포함합니다.

특정 주제에 이름과 기본 썸네일을 지정하고 싶으면 `site.json`의 `topics`에 다음과 같은 선택 설정을 넣을 수 있습니다. 이 설정만으로 필터의 주제 선택 항목이 생성되지는 않습니다.

```json
{
  "id": "machine-learning",
  "name": "머신러닝",
  "description": "학습 방법과 표현에 관한 논문 리뷰.",
  "thumbnail": "/assets/topics/topic-01.svg"
}
```

선택 설정의 `id`는 영문 소문자·숫자·하이픈을 사용하며, 포스트의 `topics`에 같은 값을 입력하면 적용됩니다. `all`은 기존 전체 목록 링크와의 호환을 위해 예약되어 있습니다. 포스트의 `topicNames`가 선택 설정의 표시 이름보다 우선합니다.

## 논문 리뷰 작성

에이전트·작업자는 [포스팅 스킬](skills/publish-research-blog/SKILL.md)을 따른다. 출처 검증부터 한글·영문 작성, 공개 설정, 배포 확인까지 필요한 규칙과 [연구 인계 양식](skills/publish-research-blog/assets/research-handoff.md), [리뷰 본문 템플릿](skills/publish-research-blog/assets/review-body.md)을 포함한다. 요구사항 변경은 이 스킬을 기준으로 반영한다.

다른 AI 서비스에서는 스킬과 두 템플릿을 첨부하거나 저장소에서 읽게 한 뒤 [복사용 작업 요청](skills/publish-research-blog/assets/agent-request.md)의 입력을 채워 전달한다. 저장소·실행 도구가 없으면 원고와 인계 자료를 받고, 접근 가능한 작업자가 검증·게시를 이어간다.

```sh
npm run new -- paper-slug "머신러닝" "컴퓨터 비전"
```

`content/reviews/날짜-paper-slug.ko.md`와 `.en.md` 원고가 함께 생성되며, 둘 다 `visibility: "private"`로 시작합니다. 주제는 공백으로 구분한 인자로 한 개 이상 전달하며, 주제에 공백이 있으면 따옴표로 감쌉니다. 주제를 생략하면 `["미분류"]`를 사용합니다. 각 언어의 제목·설명·본문을 채우고 공개할 파일만 `visibility: "public"`으로 변경합니다. 영어 번역은 별도 원고로 작성하며, 원문을 바꿀 때 번역본도 함께 수정합니다.

```yaml
---
title: "내 리뷰 제목"
description: "카드에 표시할 짧은 설명"
date: "2026-10-06"
publishedAt: "2026-10-06T10:00:00+09:00"
updatedAt: "2026-10-07T14:30:00+09:00"
topics: ["머신러닝", "컴퓨터 비전"]
visibility: "public"
lang: "ko"
translationKey: "paper-slug"
paperTitle: "논문 원제"
authors: "저자"
year: "2026"
paperUrl: "https://논문-원문-주소"
thumbnail: "/assets/논문-썸네일.webp"
thumbnailAlt: "썸네일 설명"
---

## 핵심 아이디어

리뷰 본문을 작성합니다.
```

- `title`, `description`, `date`, 한 개 이상의 `topics`는 공개 시 필수입니다. 기존 단일 `topic`도 지원하며, `visibility`를 정확히 `"public"`으로 지정해야 발행됩니다.
- `visibility: "private"`이거나 공개 여부가 없으면 비공개로 처리합니다. 다른 값은 오류로 알려줍니다.
- `lang`은 `ko` 또는 `en`입니다. 두 언어의 `translationKey`와 주제 집합을 같게 설정하면 같은 글로 묶입니다. 초안 생성 명령이 이를 설정합니다.
- 번역본을 발행하지 않았으면 원문을 표시하고 아직 번역이 없음을 안내합니다. 번역본 발행 후에는 언어 버튼으로 제목·요약·본문까지 전환됩니다.
- 두 언어로 작성해도 카드 수는 논문 한 편으로 계산합니다. 기존의 `.md` 원고는 기본적으로 한글 원고로 처리합니다.
- `thumbnail`을 비우면 첫 주제의 선택 설정 이미지 또는 `reviewThumbnail`을 사용합니다.
- 카드 제목을 누르면 `/reviews/paper-slug/`의 리뷰 본문을 엽니다.
- 본문은 Markdown으로 작성하며, `##` 제목으로 목차를 만듭니다.
- 기존 `draft: true`도 계속 지원하며, 한국 시간 기준 미래 날짜의 글과 함께 게시에서 제외합니다.
- 비공개 글은 메인·리뷰 카드·주제 선택 목록·관련 글·본문 URL·RSS·사이트맵에 포함하지 않습니다. 한글·영어 원고의 공개 여부는 각각 설정하며, 공개되지 않은 번역 대신 공개 원문을 안내와 함께 표시합니다.
- 메인에는 업데이트 시각이 최신인 리뷰를 `homeReviewLimit`개까지 표시하며, 하단 **더보기**로 리뷰 페이지를 엽니다. 관련 글은 주제 중 하나 이상을 공유하는 리뷰를 중복 없이 표시합니다.
- 필터의 **주제 선택**을 열면 공개 포스트에서 모은 주제들이 체크박스로 표시됩니다. 여러 주제를 고르면 그중 하나라도 포함된 리뷰를 중복 없이 표시하며, 선택이 없으면 전체 리뷰를 보여줍니다. 선택한 주제들은 URL에 남고 브라우저 뒤로가기로 복원됩니다.

## 발행·업데이트 시각과 목록 검색

- `date`는 한국 시간 기준 최초 발행일입니다. `publishedAt`·`updatedAt`은 초와 시간대가 있는 ISO 8601 시각이며, `Z` 또는 `+09:00` 같은 시간대를 명시합니다. `publishedAt`의 한국 날짜는 `date`와 같아야 하고, `updatedAt`은 `publishedAt` 이후여야 합니다.
- 초안 생성기는 현재 시각을 두 항목에 넣습니다. 처음 공개할 때 실제 발행일·시각으로 맞추고, 이미 게시한 글을 수정할 때는 `publishedAt`을 유지하며 변경한 언어의 `updatedAt`을 갱신합니다. 두 언어를 같이 수정하면 같은 업데이트 시각을 사용합니다.
- 같은 논문의 공개 언어들 중 최초 발행 시각과 가장 최근 업데이트 시각을 목록에서 공유합니다. 비공개 번역은 순서·검색에 영향을 주지 않습니다. RSS의 발행 시각은 최초 발행을 유지합니다.
- 시각이 없는 기존 글은 `date`의 한국 시간 00:00을 정렬·기간 검색에 사용하고 화면에는 날짜만 표시합니다. 정확한 시각을 모르면 추정해 채우지 않습니다. 미래 발행일 또는 미래 `publishedAt`의 글은 빌드에서 제외됩니다.
- 리뷰 페이지의 **필터**에서 여러 주제와 업데이트 시작·종료 시각을 설정하고, 별도의 **검색**에서 단어를 찾습니다. 목록 위 **정렬**에서 최근·오래된 업데이트순을 바로 선택합니다. 두 조건은 함께 적용되며 **필터 초기화**와 **검색 초기화**는 각각의 조건만 지웁니다. 기간은 한국 시간이며 종료 분 전체를 포함합니다. 검색은 현재 언어의 공개 제목·요약·본문·논문 제목·저자·주제에서 공백으로 나눈 모든 단어를 찾습니다.
- **한 번에** 불러올 개수는 **6 / 12 / 24**, 기본은 12입니다. 목록 하단에 가까워지면 다음 묶음의 카드를 추가합니다. **더 불러오기** 버튼으로도 추가할 수 있습니다. 아직 표시하지 않은 카드는 문서에서 분리해 배치 계산을 줄이고, 썸네일은 지연 로딩합니다. 필터·검색·정렬을 바꾸면 첫 묶음부터 다시 표시합니다. 초기화해도 정렬·불러올 개수는 유지합니다. 조건과 표시한 묶음은 URL·새로고침·언어 전환·뒤로가기에 유지됩니다. JavaScript를 사용할 수 없으면 공개 글 전체를 기본 순서로 표시합니다.


현재 GitHub 저장소는 공개 저장소입니다. `visibility: "private"`는 블로그 게시 여부를 제어하며 저장소의 접근 권한을 바꾸지 않습니다. 비공개 원고를 GitHub에 커밋·푸시하면 원고와 Git 이력을 저장소에서 읽을 수 있으므로, 공개하면 안 되는 원고는 로컬에 보관합니다. `public/`의 파일도 모두 배포되므로 비공개 자료를 넣지 않습니다.

## 게시하기

저장소는 `https://github.com/iiamaii/iiamaii.github.io`이며, GitHub Pages 게시 소스는 **GitHub Actions**입니다.

파일을 수정하고 확인한 다음 `main` 브랜치에 커밋·푸시하면 자동으로 다시 게시됩니다. `npm test`는 임시 폴더의 검증용 원고로 최신 목록 제한, 여러 주제와 중복 제외, 시각 정렬·기간·단어 검색·페이지 이동, 검색 데이터의 비공개 제외, 한글·영문 연결과 피드를 확인합니다.

```sh
npm test
npm run build
git add site.json public/assets scripts README.md
# 공개할 원고만 파일별로 추가합니다.
git add content/reviews/날짜-paper-slug.ko.md
git commit -m "Update profile and paper reviews"
git push
```

[Actions 배포 목록](https://github.com/iiamaii/iiamaii.github.io/actions/workflows/pages.yml)에서 성공을 확인할 수 있습니다. 이전 `/about/` 주소는 `/profile/`로 이동합니다.

## 폰트

제목에는 **함렡(Hahmlet)**, 본문에는 **IBM Plex Sans KR**을 사용합니다. 폰트 파일과 OFL 1.1 라이선스를 사이트에 포함해 외부 CDN 없이 불러옵니다. 제목과 본문의 조합은 `public/assets/style.css`에서 바꿀 수 있습니다.

[임시 폰트 비교 페이지](https://iiamaii.github.io/fonts/)는 직접 주소로만 접근합니다. 블로그의 메뉴·하단·사이트맵에는 연결하지 않고 `noindex`를 유지합니다. 하단에는 별도의 유틸리티 링크를 표시하지 않습니다. 폰트 출처는 `research/font-preview-assets.json`, 조사 자료는 `research/font-candidates*.json`에 정리했습니다.

## 파일 구성

```text
site.json             프로필·철학·이미지·선택 주제 설정
content/reviews/      논문 리뷰 Markdown
public/assets/        이미지·스타일·폰트·목록 필터 스크립트
public/fonts/         폰트 비교 페이지
scripts/build.mjs     HTML·RSS·사이트맵 생성
scripts/i18n.mjs      한국어·영어 화면 문구
scripts/render.mjs    언어별 페이지 생성
scripts/new-post.mjs  논문 리뷰 초안 생성
scripts/post-times.mjs  발행·업데이트 시각 검증
public/assets/review-query.js  검색·필터·페이지 계산
skills/publish-research-blog/  포스팅 스킬·연구 인계·본문 템플릿
AGENTS.md            작업자의 스킬 연결
dist/                 빌드 결과 (Git 제외)
```

`research/blog-design-references.json`은 기존 원본 조사 자료로 보존하며 사이트에는 포함하지 않습니다.
