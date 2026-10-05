import fs from 'node:fs/promises';
import path from 'node:path';
import matter from 'gray-matter';
import { Marked, Renderer } from 'marked';

const root = process.cwd();
const out = path.join(root, 'dist');
const config = JSON.parse(await fs.readFile(path.join(root, 'site.json'), 'utf8'));
const base = (process.env.SITE_BASE_PATH ?? config.basePath ?? '').replace(/\/$/, '');
if (base && !/^\/[a-zA-Z0-9_-]+$/.test(base)) throw new Error('basePath must be empty or a single /repository path.');
const url = new URL(config.url);
if (!['http:', 'https:'].includes(url.protocol)) throw new Error('site.url must use http or https.');
const href = (p = '/') => `${base}${p}`;
const absolute = p => new URL(href(p), url.origin).href;
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const dateText = value => value.replaceAll('-', '.');
const label = p => `<span>${esc(p.category)}</span>${p.sample ? '<span class="sample-badge">예시 글</span>' : ''}`;
const postUrl = p => `/posts/${p.slug}/`;

function renderMarkdown(source) {
  const headings = [];
  const used = new Set();
  const parser = new Marked();
  parser.use({ renderer: {
    heading({ tokens, depth }) {
      const text = tokens.map(t => t.text ?? t.raw ?? '').join('');
      const stem = text.toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, '').trim().replace(/\s+/g, '-') || 'section';
      let id = stem;
      let n = 2;
      while (used.has(id)) id = `${stem}-${n++}`;
      used.add(id);
      if (depth === 2) headings.push({ id, text });
      return `<h${depth} id="${esc(id)}">${this.parser.parseInline(tokens)}</h${depth}>\n`;
    },
    table(token) {
      return `<div class="table-scroll" role="region" aria-label="본문 표" tabindex="0">${Renderer.prototype.table.call(this, token)}</div>`;
    }
  } });
  return { html: parser.parse(source), headings };
}

const files = (await fs.readdir(path.join(root, 'content/posts'))).filter(f => f.endsWith('.md'));
const posts = [];
for (const file of files) {
  const { data, content } = matter(await fs.readFile(path.join(root, 'content/posts', file), 'utf8'));
  if (data.draft === true) continue;
  if (typeof data.title !== 'string' || typeof data.description !== 'string') throw new Error(`${file}: title and description are required.`);
  const date = data.date instanceof Date ? data.date.toISOString().slice(0, 10) : String(data.date);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(Date.parse(date))) throw new Error(`${file}: date must be YYYY-MM-DD.`);
  if (date > new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Seoul' })) continue;
  const slug = file.replace(/\.md$/, '').replace(/^\d{4}-\d{2}-\d{2}-/, '');
  if (!/^[a-z0-9-]+$/.test(slug)) throw new Error(`${file}: use lowercase ASCII letters, numbers and hyphens in the filename.`);
  if (posts.some(p => p.slug === slug)) throw new Error(`Duplicate slug: ${slug}`);
  const readingMinutes = Math.max(1, Math.ceil(content.replace(/\s/g, '').length / 500));
  posts.push({ ...data, date, slug, content, readingMinutes, ...renderMarkdown(content) });
}
posts.sort((a, b) => b.date.localeCompare(a.date) || a.slug.localeCompare(b.slug));
const featured = posts.find(p => p.featured) ?? posts[0];

function page({ title, description = config.description, route, content, kind = 'website', published }) {
  return `<!doctype html>
<html lang="${esc(config.language)}">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title === config.title ? title : `${title} · ${config.title}`)}</title>
<meta name="description" content="${esc(description)}">
<meta name="theme-color" content="#ffffff">
<link rel="canonical" href="${esc(absolute(route))}">
<meta property="og:type" content="${kind}"><meta property="og:locale" content="ko_KR">
<meta property="og:title" content="${esc(title)}"><meta property="og:description" content="${esc(description)}">
<meta property="og:site_name" content="${esc(config.title)}"><meta property="og:url" content="${esc(absolute(route))}">
${published ? `<meta property="article:published_time" content="${published}T00:00:00+09:00">` : ''}
<link rel="icon" type="image/svg+xml" href="${href('/favicon.svg')}">
<link rel="stylesheet" href="${href('/assets/style.css')}">
<link rel="alternate" type="application/rss+xml" title="${esc(config.title)}" href="${href('/feed.xml')}">
</head>
<body>
<a class="skip-link" href="#main">본문으로 건너뛰기</a>
<div class="site-shell">
<header class="site-header">
<a class="brand" href="${href('/')}" aria-label="${esc(config.title)} 홈">${esc(config.title)}<span class="brand-square" aria-hidden="true"></span></a>
<span class="brand-caption">A JOURNAL OF EXPLORATION</span>
<nav aria-label="주 메뉴">
<a href="${href('/#notes')}" ${route === '/' ? 'aria-current="page"' : ''}>글</a>
<a href="${href('/about/')}" ${route === '/about/' ? 'aria-current="page"' : ''}>소개</a>
<a href="${esc(config.github)}" target="_blank" rel="noopener noreferrer" aria-label="GitHub 프로필 (새 탭)">GitHub</a>
</nav>
</header>
${content}
<footer class="site-footer"><a class="footer-brand" href="${href('/')}">${esc(config.title)}<span aria-hidden="true">.</span></a><p>읽고, 실험하고, 다시 기록합니다.</p><div><a href="${href('/feed.xml')}">RSS</a><span>© ${new Date().getFullYear()} ${esc(config.author)}</span></div></footer>
</div>
</body></html>`;
}

const entry = (p, i) => `<article class="note-row"><span class="note-number" aria-hidden="true">${String(i + 1).padStart(2, '0')}</span><div><div class="eyebrow">${label(p)}</div><h3><a href="${href(postUrl(p))}">${esc(p.title)}</a></h3><p>${esc(p.description)}</p><div class="post-meta"><time datetime="${p.date}">${dateText(p.date)}</time><span>${p.readingMinutes}분 읽기</span></div></div></article>`;

const home = `<main id="main">
<section class="intro" aria-labelledby="intro-title"><div><p class="eyebrow intro-kicker">기술 · 연구 · 생각</p><h1 id="intro-title">읽고, 실험하고,<br>기록합니다.</h1></div><div class="intro-note"><span class="small-rule" aria-hidden="true"></span><p>궁금한 것을 깊이 들여다보고,<br> 이해한 것을 나의 언어로 남기는 곳.</p><span class="mono">RESEARCH, IN PROGRESS</span></div></section>
${featured ? `<section class="featured" aria-labelledby="featured-title"><a class="cover-link" href="${href(postUrl(featured))}" tabindex="-1" aria-hidden="true">${featured.cover ? `<img src="${href(featured.cover)}" alt="" width="1536" height="1024" fetchpriority="high">` : '<span class="cover-wordmark">연구노트.</span>'}<span class="cover-caption">FIELD NOTES / ${featured.date.slice(0, 4)}</span></a><div class="featured-copy"><div class="eyebrow"><span class="accent">이번에 펼칠 기록</span>${featured.sample ? '<span class="sample-badge">예시 글</span>' : ''}</div><h2 id="featured-title"><a href="${href(postUrl(featured))}">${esc(featured.title)}</a></h2><p>${esc(featured.description)}</p><div class="post-meta"><time datetime="${featured.date}">${dateText(featured.date)}</time><span>${featured.readingMinutes}분 읽기</span></div><a class="read-link" href="${href(postUrl(featured))}">글 읽기<span class="read-line" aria-hidden="true"></span></a></div></section>` : ''}
<div class="journal-grid"><section class="notes-section" id="notes" aria-labelledby="notes-title"><div class="section-title"><h2 id="notes-title">기록들</h2><span class="mono">${String(posts.length).padStart(2, '0')} NOTES</span></div>${posts.length ? posts.map(entry).join('') : '<p class="empty-state">아직 발행한 글이 없습니다. 첫 번째 기록을 준비하고 있습니다.</p>'}</section><aside class="editor-note"><div class="section-title"><h2>이 노트에 관하여</h2><span class="mono">ABOUT</span></div><p class="editor-heading">배운 것을 남기면,<br>다음 질문이 보입니다.</p><p>${esc(config.intro)}</p><p>논문에서 발견한 질문부터 코드로 확인한 작은 실험까지. 기록을 통해 생각을 이어갑니다.</p><a class="text-link" href="${href('/about/')}">소개 읽기</a><div class="sidebar-colophon"><span class="mono">THE NOTEBOOK</span><p>연구 기록<br>논문 읽기<br>실험 노트</p></div></aside></div>
</main>`;

const about = `<main id="main" class="about-page"><div class="page-kicker eyebrow">ABOUT THE NOTEBOOK</div><h1>탐구한 것을<br>기록하는 공간.</h1><div class="about-grid"><div class="prose"><p class="lede">${esc(config.intro)}</p><p>이곳은 기술과 연구를 이해해 가는 과정을 담는 개인 블로그입니다. 읽은 자료의 핵심, 직접 확인한 내용과 아직 남은 질문을 차분히 정리합니다.</p><h2>어떤 기록을 남기는가</h2><p>논문을 읽으며 떠오른 질문, 코드로 실행한 실험, 배운 개념을 자신의 언어로 정리한 글을 모읍니다. 완성된 결론뿐 아니라 생각이 바뀌는 과정도 기록합니다.</p><h2>근거와 생각을 함께</h2><p>자료의 출처를 남기고, 직접 관찰한 것과 해석한 것을 구분하는 글을 지향합니다. 후속 기록이 생기면 이전 글과 연결합니다.</p>${posts.some(p=>p.sample) ? '<div class="sample-notice">현재 보이는 예시 글은 블로그 형식을 보여주기 위한 원고입니다. 작성자의 실제 연구 성과나 실험 결과를 담고 있지 않습니다.</div>' : ''}</div><aside class="about-aside"><span class="mono">ELSEWHERE</span><a href="${esc(config.github)}" target="_blank" rel="noopener noreferrer">GitHub 프로필</a><span class="mono">FOLLOW THE NOTES</span><a href="${href('/feed.xml')}">RSS로 새 글 받기</a></aside></div></main>`;

await fs.rm(out, { recursive: true, force: true });
await fs.mkdir(out, { recursive: true });
await fs.cp(path.join(root, 'public'), out, { recursive: true });
async function write(route, text) {
  const file = path.join(out, route.replace(/^\//, ''), 'index.html');
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.writeFile(file, text);
}
await write('/', page({ title: config.title, route: '/', content: home }));
await write('/about/', page({ title: '소개', route: '/about/', content: about }));

for (const p of posts) {
  const others = posts.filter(q => q.slug !== p.slug).slice(0, 2);
  const body = `<main id="main" class="article-page"><a class="back-link" href="${href('/#notes')}">모든 기록</a><header class="article-header"><div class="eyebrow">${label(p)}</div><h1>${esc(p.title)}</h1><p class="article-deck">${esc(p.description)}</p><div class="post-meta"><time datetime="${p.date}">${dateText(p.date)}</time><span>${p.readingMinutes}분 읽기</span><span>${esc(config.author)}</span></div></header>${p.cover ? `<figure class="article-cover"><img src="${href(p.cover)}" alt="검은색과 파란색 면이 있는 격자 종이 조각의 추상 이미지" width="1536" height="1024"><figcaption>질문과 기록을 주제로 한 추상 이미지 · AI 생성</figcaption></figure>` : ''}<div class="reading-layout"><article class="prose">${p.sample ? '<div class="sample-notice"><strong>예시 글</strong><br>블로그 구성을 보여주는 원고입니다. 자신의 연구 내용으로 바꾸어 사용하세요.</div>' : ''}${p.html}<div class="article-tags" aria-label="태그">${(p.tags ?? []).map(t => `<span>${esc(t)}</span>`).join('')}</div></article>${p.headings.length ? `<aside class="article-toc"><nav aria-label="이 글의 목차"><span class="mono">IN THIS NOTE</span><ol>${p.headings.map(h=>`<li><a href="#${esc(h.id)}">${esc(h.text)}</a></li>`).join('')}</ol></nav></aside>` : ''}</div>${others.length ? `<section class="more-notes" aria-labelledby="more-title"><div class="section-title"><h2 id="more-title">이어 읽을 기록</h2></div><div class="related-grid">${others.map(q=>`<article><div class="eyebrow">${label(q)}</div><h3><a href="${href(postUrl(q))}">${esc(q.title)}</a></h3><p>${esc(q.description)}</p></article>`).join('')}</div></section>` : ''}</main>`;
  await write(postUrl(p), page({ title: p.title, description: p.description, route: postUrl(p), content: body, kind: 'article', published: p.date }));
}

const notFound = page({ title: '페이지를 찾을 수 없습니다', route: '/404.html', content: `<main id="main" class="not-found"><span class="mono">404 / NOT FOUND</span><h1>아직 없는 기록입니다.</h1><p>주소를 확인하거나 글 목록에서 다른 기록을 찾아보세요.</p><a class="read-link" href="${href('/')}">홈으로 돌아가기</a></main>` });
await fs.writeFile(path.join(out, '404.html'), notFound);
const xml = value => esc(value);
await fs.writeFile(path.join(out, 'feed.xml'), `<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel><title>${xml(config.title)}</title><link>${xml(absolute('/'))}</link><description>${xml(config.description)}</description><language>ko</language><atom:link href="${xml(absolute('/feed.xml'))}" rel="self" type="application/rss+xml"/>${posts.map(p=>`<item><title>${xml(p.title)}</title><link>${xml(absolute(postUrl(p)))}</link><guid isPermaLink="true">${xml(absolute(postUrl(p)))}</guid><description>${xml(p.description)}</description><pubDate>${new Date(`${p.date}T00:00:00+09:00`).toUTCString()}</pubDate></item>`).join('')}</channel></rss>`);
await fs.writeFile(path.join(out, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${['/', '/about/', ...posts.map(postUrl)].map(route => `<url><loc>${xml(absolute(route))}</loc></url>`).join('')}</urlset>`);
await fs.writeFile(path.join(out, 'robots.txt'), `User-agent: *\nAllow: /\nSitemap: ${absolute('/sitemap.xml')}\n`);
await fs.writeFile(path.join(out, '.nojekyll'), '');
console.log(`Built ${posts.length} posts to dist/ for ${absolute('/')}`);
