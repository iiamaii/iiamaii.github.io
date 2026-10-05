# 연구노트

`iiamaii`의 프로필·철학과 주제별 논문 리뷰를 모으는 GitHub Pages 블로그입니다.

- [메인](https://iiamaii.github.io/): 대문 이미지와 철학 / 주제별 논문 카드, 넓은 화면에서 좌우 50:50 구성
- [내 프로필](https://iiamaii.github.io/profile/): 소개, 철학, 연구 관심사
- [논문 리뷰](https://iiamaii.github.io/reviews/): 전체·주제 탭, 썸네일과 짧은 설명, 리뷰 본문
- GitHub page 메뉴: 기본적으로 [iiamaii GitHub 프로필](https://github.com/iiamaii)로 연결

기존 예시 글은 게시 목록에서 정리했습니다. 현재 실제 논문 리뷰는 0편이며, `주제 01–03`과 등록 대기 카드가 표시됩니다. 프로필과 철학은 내용을 채우기 전까지 준비 안내를 보여줍니다.

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
| `githubPage` | GitHub page 메뉴의 외부 주소 |
| `title`, `description` | 블로그 이름과 검색 설명 |

이미지는 `public/assets/`에 넣고 `/assets/파일명.webp`처럼 설정합니다. 현재 대문은 이전에 생성한 추상 이미지를 사용합니다. 생성 기록은 `research/cover-artwork.json`에 있습니다.

## 논문 주제

`site.json`의 `topics` 목록에서 이름, 설명과 기본 썸네일을 수정합니다. 주제를 추가하거나 삭제하면 메인 그룹과 리뷰 탭도 함께 갱신됩니다.

```json
{
  "id": "machine-learning",
  "name": "머신러닝",
  "description": "학습 방법과 표현에 관한 논문 리뷰.",
  "thumbnail": "/assets/topics/topic-01.svg"
}
```

`id`는 중복되지 않는 영문 소문자·숫자·하이픈을 사용합니다. 기존 리뷰가 있으면 해당 리뷰의 `topic`도 함께 바꿉니다. `all`은 전체 탭을 위해 예약되어 있습니다.

## 논문 리뷰 작성

```sh
npm run new -- paper-slug topic-01
```

`content/reviews/날짜-paper-slug.md` 초안이 생성됩니다. 주제 id를 생략하면 첫 번째 주제를 사용합니다. 원문 정보와 본문을 채우고 `draft: false`로 변경합니다.

```yaml
---
title: "내 리뷰 제목"
description: "카드에 표시할 짧은 설명"
date: "2026-10-06"
topic: "topic-01"
paperTitle: "논문 원제"
authors: "저자"
year: "2026"
paperUrl: "https://논문-원문-주소"
thumbnail: "/assets/논문-썸네일.webp"
thumbnailAlt: "썸네일 설명"
draft: false
---

## 핵심 아이디어

리뷰 본문을 작성합니다.
```

- `title`, `description`, `date`, `topic`은 발행 시 필수입니다.
- `thumbnail`을 비우면 해당 주제의 기본 썸네일을 사용합니다.
- 카드 제목을 누르면 `/reviews/paper-slug/`의 리뷰 본문을 엽니다.
- 본문은 Markdown으로 작성하며, `##` 제목으로 목차를 만듭니다.
- 초안과 한국 시간 기준 미래 날짜의 글은 게시되지 않습니다.
- 메인에는 주제별 최신 리뷰 4편까지 표시하며, 리뷰 페이지에는 모두 표시합니다.
- 주제 탭은 클릭과 좌우 방향키·Home·End 키를 지원합니다. 선택한 주제는 URL에 남고 브라우저 뒤로가기로 돌아갈 수 있습니다.

## 게시하기

저장소는 `https://github.com/iiamaii/iiamaii.github.io`이며, GitHub Pages 게시 소스는 **GitHub Actions**입니다.

파일을 수정하고 확인한 다음 `main` 브랜치에 커밋·푸시하면 자동으로 다시 게시됩니다. `npm test`는 임시 폴더의 검증용 원고로 주제별 카드, 초안 제외, 썸네일과 주소 생성을 확인합니다.

```sh
npm test
npm run build
git add site.json content/reviews public/assets scripts README.md
git commit -m "Update profile and paper reviews"
git push
```

[Actions 배포 목록](https://github.com/iiamaii/iiamaii.github.io/actions/workflows/pages.yml)에서 성공을 확인할 수 있습니다. 이전 `/about/` 주소는 `/profile/`로 이동합니다.

## 폰트

제목에는 **함렡(Hahmlet)**, 본문에는 **IBM Plex Sans KR**을 사용합니다. 폰트 파일과 OFL 1.1 라이선스를 사이트에 포함해 외부 CDN 없이 불러옵니다. 제목과 본문의 조합은 `public/assets/style.css`에서 바꿀 수 있습니다.

[폰트 비교 페이지](https://iiamaii.github.io/fonts/)에서는 한글 6종·영문 6종을 문장과 크기를 바꾸며 확인할 수 있습니다. 비교 링크는 하단 메뉴에 있습니다. 폰트 출처는 `research/font-preview-assets.json`, 조사 자료는 `research/font-candidates*.json`에 정리했습니다.

## 파일 구성

```text
site.json             프로필·철학·주제 설정
content/reviews/      논문 리뷰 Markdown
public/assets/        이미지·스타일·폰트·탭 스크립트
public/fonts/         폰트 비교 페이지
scripts/build.mjs     HTML·RSS·사이트맵 생성
scripts/new-post.mjs  논문 리뷰 초안 생성
dist/                 빌드 결과 (Git 제외)
```

`research/blog-design-references.json`은 기존 원본 조사 자료로 보존하며 사이트에는 포함하지 않습니다.
