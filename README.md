# 연구노트

기술·연구 기록을 위한 잡지형 개인 블로그입니다. Markdown 원고를 정적인 HTML로 빌드하고 GitHub Pages에 게시합니다.

게시할 주소: `https://iiamaii.github.io/`

이 주소는 저장소 생성과 GitHub Pages 배포가 끝난 후 사용할 수 있습니다. 이 파일만으로 게시 완료를 의미하지 않습니다.

## 로컬에서 보기

Node.js 22 이상이 필요합니다. GitHub Actions는 Node.js 24를 사용합니다.

```sh
npm ci
npm run dev
```

브라우저에서 `http://localhost:4321/`을 엽니다. 원고나 디자인을 수정한 뒤 `npm run build`를 실행하고 브라우저를 새로고침합니다. 미리보기 서버는 Ctrl+C로 종료합니다.

## 글 작성하기

```sh
npm run new -- my-research-note
```

`content/posts/날짜-my-research-note.md`가 생성됩니다. 파일의 맨 위 설정과 Markdown 본문을 수정하고, 발행할 때 `draft: false`로 바꿉니다.

```yaml
---
title: "글 제목"
description: "목록과 검색 결과에 보여줄 요약"
date: "2026-10-06"
category: "연구 기록"
tags: ["Research", "Code"]
draft: false
featured: false
---

## 질문

본문을 작성합니다.
```

- 파일명은 `YYYY-MM-DD-영문-슬러그.md` 형식입니다. 글 주소는 `/posts/영문-슬러그/`가 됩니다.
- 초안(`draft: true`)과 한국 시간 기준 미래 날짜의 글은 빌드 결과에서 제외됩니다.
- `featured: true`인 가장 최근 글이 홈의 대표 글이 됩니다. 지정한 글이 없으면 가장 최근 글을 사용합니다.
- 표와 코드 블록, 링크, 인용문을 지원합니다. 2단계 제목(`##`)으로 글의 목차를 만듭니다.
- 이미지는 `public/assets/`에 넣고 `cover: "/assets/파일명.webp"`로 지정할 수 있습니다.
- 초기 원고 3개에는 `sample: true`가 설정되어 있습니다. 실제 원고로 바꿀 때 예시 안내 문구를 삭제하고 `sample: false`로 바꿉니다. 필요 없는 예시 파일은 삭제해도 됩니다.

## 블로그 이름과 소개 바꾸기

`site.json`의 `title`, `description`, `author`, `intro`, `github`를 수정합니다. 소개 페이지의 긴 문장은 `scripts/build.mjs`의 `about` 내용을 수정합니다. 스타일은 `public/assets/style.css`에 있습니다.

## GitHub Pages에 게시하기

1. [새 저장소 만들기](https://github.com/new)에서 Owner를 **`iiamaii`**, Repository name을 **`iiamaii.github.io`**, 공개 범위를 **Public**으로 설정합니다. 이 프로젝트를 올릴 때는 README, `.gitignore`, License를 추가하지 않은 빈 저장소로 만듭니다. 무료 GitHub 계정에서는 공개 저장소로 GitHub Pages를 사용할 수 있습니다.
2. 아래 인증과 업로드 절차로 이 프로젝트를 `main` 브랜치에 올립니다. `.github/workflows/pages.yml`이 빌드와 배포를 실행합니다.
3. 저장소의 **Settings → Pages → Build and deployment → Source**를 **GitHub Actions**로 설정합니다.
4. **Actions → Publish research notes → Run workflow → main → Run workflow**로 게시를 실행합니다.
5. **Actions → Publish research notes**에서 성공을 확인하고 Pages 설정의 **Visit site**를 엽니다.

터미널로 처음 올리는 경우, 저장소를 빈 상태로 만든 뒤 아래 명령을 사용할 수 있습니다. 기존 저장소가 있으면 먼저 내용을 확인하고 병합해야 합니다. 강제 푸시는 사용하지 마세요.

```sh
cd /Users/ws/Documents/ChatGPT/blog
git add .gitignore package.json package-lock.json site.json scripts public content .github README.md
git commit -m "Create magazine-style research blog"
git branch -M main
git remote add origin https://github.com/iiamaii/iiamaii.github.io.git
git push -u origin main
```

처음 GitHub에 연결한다면 GitHub CLI의 브라우저 인증을 사용할 수 있습니다. macOS에서 Homebrew를 사용하는 경우 `brew install gh`로 설치한 뒤 `gh auth login`을 실행합니다. **GitHub.com → HTTPS → Git 인증 허용(Yes) → Login with a web browser**를 선택하고 `iiamaii` 계정으로 로그인한 다음 위의 `git push`를 실행합니다. 계정 비밀번호를 Git의 비밀번호로 입력하지 않습니다. 공식 안내: [Git 인증 저장](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git).

첫 업로드 후 Pages 설정을 켰다면 **Actions → Publish research notes → Run workflow → main → Run workflow**로 게시를 다시 시작할 수 있습니다. 워크플로 파일은 이미 있으므로 GitHub가 제안하는 추가 템플릿을 만들 필요는 없습니다. 게시 완료 후 `https://iiamaii.github.io/`에서 확인합니다.

이 폴더가 이미 다른 저장소와 연결되어 있다면 `git remote -v`로 확인하고 원격 주소를 적절히 선택하세요. 초기 디자인 조사 자료인 `research/blog-design-references.json`은 원본 자료이며 사이트에는 포함되지 않습니다.

`main`에 글이나 스타일을 추가로 푸시하면 다시 게시됩니다. 사용자 사이트용으로 `site.json`의 `basePath`는 빈 문자열입니다. 프로젝트 사이트를 사용하는 경우에는 `/저장소이름`으로 바꿀 수 있고, GitHub Actions에서는 Pages가 반환한 경로를 자동으로 사용합니다.

공식 안내: [GitHub Pages 시작하기](https://docs.github.com/en/pages/quickstart), [게시 소스 설정](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site), [사용자 지정 배포 워크플로](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## 폰트 후보

블로그의 **폰트 비교** 메뉴 또는 로컬 미리보기의 `http://localhost:4321/fonts/`에서 추가 후보를 확인합니다. 문장을 직접 입력하고 본문 크기를 조절할 수 있습니다. 폰트 파일은 `public/assets/fonts/`에 공식 라이선스와 함께 포함되어 있어 외부 폰트 CDN 연결 없이 표시됩니다. 다운로드 출처와 파일 해시는 `research/font-preview-assets.json`에 기록했습니다.

기본 후보 9종은 `research/font-candidates.json`, 추가 후보 12종과 공식 라이선스는 `research/font-candidates-distinctive.json`에 정리했습니다. 추가 후보는 함렡, 송명, IBM Plex Sans KR, Wanted Sans, LINE Seed Sans KR, 구기와 Fraunces, Bodoni Moda, Instrument Serif, DM Serif Display, Space Grotesk, Bricolage Grotesque입니다. 모두 OFL 1.1로 무료 웹 사용이 가능합니다. 실제 적용할 폰트는 비교 후 선택합니다.

## 구성

```text
content/posts/        Markdown 원고
public/assets/       스타일과 이미지
scripts/build.mjs    정적 HTML·RSS·사이트맵 생성
scripts/dev.mjs      로컬 미리보기
scripts/new-post.mjs 새 초안 생성
site.json           블로그 기본 정보
dist/               빌드 결과 (Git 제외)
```

페이지는 JavaScript 없이 읽을 수 있습니다. 글별 제목·요약·주소, RSS와 사이트맵을 생성합니다. 글꼴은 Google Fonts에서 불러오며, 연결이 없으면 운영체제 글꼴로 표시합니다.

## 대표 이미지

최종 파일: `public/assets/research-cover.webp` (1536 × 1024). 내장 imagegen 도구로 한 번 생성하고 WebP로 변환했습니다. 실제 과학 구조를 표현하지 않는 추상 이미지입니다.

생성에 사용한 프롬프트와 출처 정보는 [research/cover-artwork.json](research/cover-artwork.json)에 기록했습니다.
