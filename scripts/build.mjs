import fs from 'node:fs/promises';
import path from 'node:path';
import matter from 'gray-matter';
import { Marked, Renderer } from 'marked';

const root = process.cwd();
const out = path.join(root, 'dist');
const config = JSON.parse(await fs.readFile(path.join(root, 'site.json'), 'utf8'));
const base = (process.env.SITE_BASE_PATH ?? config.basePath ?? '').replace(/\/$/, '');
if (base && !/^\/[a-zA-Z0-9_-]+$/.test(base)) throw new Error('basePath must be empty or a single /repository path.');
const origin = new URL(config.url);
if (!['http:', 'https:'].includes(origin.protocol)) throw new Error('site.url must use http or https.');
const href = (route = '/') => `${base}${route}`;
const absolute = route => new URL(href(route), origin.origin).href;
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const paragraphs = value => String(value ?? '').split(/\n\s*\n/).filter(Boolean).map(p => `<p>${esc(p).replaceAll('\n', '<br>')}</p>`).join('');
const reviewUrl = review => `/reviews/${review.slug}/`;
const dateText = value => value.replaceAll('-', '.');
const arrow = '<span aria-hidden="true">↗</span>';
const external = value => {
  const url = new URL(value);
  if (!['https:', 'http:'].includes(url.protocol)) throw new Error(`Use an http(s) URL: ${value}`);
  return url.href;
};
external(config.github);
external(config.githubPage || config.github);

const topics = config.topics ?? [];
if (!topics.length) throw new Error('Define at least one topic in site.json.');
const topicIds = new Set();
for (const topic of topics) {
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(topic.id) || topic.id === 'all' || topicIds.has(topic.id)) throw new Error(`Invalid or duplicate topic id: ${topic.id}`);
  if (!topic.name?.trim()) throw new Error(`Topic ${topic.id} needs a name.`);
  topicIds.add(topic.id);
}
async function checkImage(src) {
  if (!src || !src.startsWith('/assets/') || src.includes('..') || !/\.(svg|webp|png|jpe?g)$/i.test(src)) throw new Error(`Use a local /assets/ image: ${src}`);
  await fs.access(path.join(root, 'public', src));
}
await checkImage(config.hero.image);
await Promise.all(topics.map(topic => checkImage(topic.thumbnail)));

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

const reviews = [];
const today = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Seoul' });
const reviewDirectory = path.join(root, 'content/reviews');
await fs.mkdir(reviewDirectory, { recursive: true });
for (const file of (await fs.readdir(reviewDirectory)).filter(f => f.endsWith('.md'))) {
  const { data, content } = matter(await fs.readFile(path.join(reviewDirectory, file), 'utf8'));
  if (data.draft === true) continue;
  if (typeof data.title !== 'string' || !data.title.trim() || typeof data.description !== 'string' || !data.description.trim()) throw new Error(`${file}: title and description are required.`);
  const topic = topics.find(t => t.id === data.topic);
  if (!topic) throw new Error(`${file}: topic must match an id in site.json.`);
  const date = data.date instanceof Date ? data.date.toISOString().slice(0, 10) : String(data.date);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(Date.parse(date)) || new Date(date).toISOString().slice(0, 10) !== date) throw new Error(`${file}: date must be a valid YYYY-MM-DD.`);
  if (date > today) continue;
  const slug = file.replace(/\.md$/, '').replace(/^\d{4}-\d{2}-\d{2}-/, '');
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) || reviews.some(r => r.slug === slug)) throw new Error(`${file}: use a unique lowercase slug with hyphens.`);
  const thumbnail = data.thumbnail || topic.thumbnail;
  await checkImage(thumbnail);
  if (data.paperUrl) external(data.paperUrl);
  const readingMinutes = Math.max(1, Math.ceil(content.replace(/\s/g, '').length / 500));
  reviews.push({ ...data, topic, date, slug, thumbnail, readingMinutes, ...renderMarkdown(content) });
}
reviews.sort((a, b) => b.date.localeCompare(a.date) || a.slug.localeCompare(b.slug));

function page({ title, description = config.description, route, content, published, scripts = '' }) {
  return `<!doctype html>
<html lang="${esc(config.language)}">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title === config.title ? title : `${title} · ${config.title}`)}</title>
<meta name="description" content="${esc(description)}"><meta name="theme-color" content="#f7f7f3">
<link rel="canonical" href="${esc(absolute(route))}">
<meta property="og:type" content="${published ? 'article' : 'website'}"><meta property="og:locale" content="ko_KR">
<meta property="og:title" content="${esc(title)}"><meta property="og:description" content="${esc(description)}">
<meta property="og:site_name" content="${esc(config.title)}"><meta property="og:url" content="${esc(absolute(route))}">
<meta property="og:image" content="${esc(absolute(config.hero.image))}">
${published ? `<meta property="article:published_time" content="${published}T00:00:00+09:00">` : ''}
<link rel="icon" type="image/svg+xml" href="${href('/favicon.svg')}">
<link rel="stylesheet" href="${href('/assets/style.css')}">
<link rel="alternate" type="application/rss+xml" title="${esc(config.title)}" href="${href('/feed.xml')}">
${scripts}
</head>
<body>
<a class="skip-link" href="#main">본문으로 건너뛰기</a>
<div class="site-shell">
<header class="site-header">
<a class="brand" href="${href('/')}" aria-label="${esc(config.title)} 홈">${esc(config.title)}<span class="brand-dot" aria-hidden="true">.</span></a>
<span class="brand-caption">${esc(config.author)} / RESEARCH JOURNAL</span>
<nav aria-label="주 메뉴">
<a href="${href('/profile/')}" ${route === '/profile/' ? 'aria-current="page"' : ''}>내 프로필</a>
<a href="${href('/reviews/')}" ${route.startsWith('/reviews/') ? 'aria-current="page"' : ''}>논문 리뷰</a>
<a href="${esc(config.githubPage || config.github)}" target="_blank" rel="noopener noreferrer" aria-label="GitHub page (새 탭)">GitHub page ${arrow}</a>
</nav>
</header>
${content}
<footer class="site-footer"><a href="${href('/')}">${esc(config.author)}<span class="brand-dot" aria-hidden="true">.</span></a><p>PROFILE · PHILOSOPHY · PAPER REVIEWS</p><div><a href="${href('/fonts/')}">폰트 비교</a><a href="${href('/feed.xml')}">RSS</a><span>© ${new Date().getFullYear()}</span></div></footer>
</div>
</body></html>`;
}

function card(review, prefix) {
  const titleId = `${prefix}-${review.slug}`;
  return `<article class="review-card"><a class="card-link" href="${href(reviewUrl(review))}" aria-labelledby="${titleId}"><div class="card-image"><img src="${href(review.thumbnail)}" alt="" width="720" height="480" loading="lazy"><span class="card-arrow" aria-hidden="true">↗</span></div><div class="card-copy"><span class="card-topic">${esc(review.topic.name)}</span><h3 id="${titleId}">${esc(review.title)}</h3><p>${esc(review.description)}</p><div class="card-meta"><time datetime="${review.date}">${dateText(review.date)}</time><span>${review.readingMinutes}분 읽기</span></div></div></a></article>`;
}
function emptyCard(topic) {
  return `<article class="review-card empty-card"><div class="card-image"><img src="${href(topic.thumbnail)}" alt="" width="720" height="480" loading="lazy"></div><div class="card-copy"><span class="card-topic">리뷰 준비 중</span><h3>첫 논문을 기다리는 자리.</h3><p>${esc(topic.description || '이 주제의 논문 리뷰를 모읍니다.')} 논문을 등록하면 썸네일과 짧은 소개가 여기에 나타납니다.</p><span class="empty-card-note">아직 등록된 논문이 없습니다.</span></div></article>`;
}
const forTopic = topic => reviews.filter(review => review.topic.id === topic.id);
const topicRoute = topic => `/reviews/?topic=${encodeURIComponent(topic.id)}#review-panel`;
const philosophy = config.philosophy.text?.trim()
  ? `<div class="philosophy-text">${paragraphs(config.philosophy.text)}</div>`
  : '<div class="philosophy-text pending-copy"><p>어떤 질문을 중요하게 생각하고,<br>어떤 태도로 연구하고 싶은가.</p><p class="pending-note">나의 철학을 담을 문장을 준비하고 있습니다.</p></div>';

const home = `<main id="main" class="home-split">
<section class="home-introduction" aria-labelledby="philosophy-title"><div class="introduction-inner">
<div class="section-kicker"><span class="kicker-number">01</span><span>MY PHILOSOPHY</span><span class="kicker-author">${esc(config.profile.name)}</span></div>
<figure class="hero-image"><img src="${href(config.hero.image)}" alt="${esc(config.hero.alt)}" width="1536" height="1024" fetchpriority="high"><figcaption><span>THE OPEN NOTEBOOK</span><span aria-hidden="true">01 / ∞</span></figcaption></figure>
<div class="philosophy-copy"><h1 id="philosophy-title">${esc(config.philosophy.title)}</h1>${philosophy}<a class="text-link" href="${href('/profile/')}">내 프로필 보기 <span aria-hidden="true">→</span></a></div>
</div></section>
<section class="home-reviews" aria-labelledby="home-reviews-title"><div class="section-kicker"><span class="kicker-number">02</span><span>PAPER REVIEWS</span><span class="kicker-author">${String(reviews.length).padStart(2, '0')} REVIEWS</span></div><div class="collection-heading"><h2 id="home-reviews-title">논문으로 이어지는 질문들.</h2><p>주제별로 읽고, 생각하고, 남긴 기록.</p><a class="text-link" href="${href('/reviews/')}">전체 리뷰 <span aria-hidden="true">→</span></a></div>
${topics.map((topic, i) => {
  const list = forTopic(topic);
  return `<section class="topic-group" aria-labelledby="home-topic-${topic.id}"><div class="topic-heading"><h3 id="home-topic-${topic.id}"><span>${String(i + 1).padStart(2, '0')}</span><a href="${href(topicRoute(topic))}">${esc(topic.name)}</a></h3><a class="topic-count" href="${href(topicRoute(topic))}" aria-label="${esc(topic.name)} 리뷰 ${list.length}편 보기">${String(list.length).padStart(2, '0')} <span aria-hidden="true">↗</span></a></div><div class="review-grid">${list.length ? list.slice(0, 4).map(r => card(r, 'home')).join('') : emptyCard(topic)}</div>${list.length > 4 ? `<a class="text-link" href="${href(topicRoute(topic))}">이 주제의 리뷰 더 보기 →</a>` : ''}</section>`;
}).join('')}
</section></main>`;

const profile = `<main id="main" class="profile-page"><div class="page-heading"><p class="section-kicker"><span class="kicker-number">01</span><span>MY PROFILE</span></p><h1>내 프로필<span class="brand-dot" aria-hidden="true">.</span></h1></div><div class="profile-layout"><aside class="profile-identity"><div class="profile-art"><img src="${href(config.hero.image)}" alt="${esc(config.hero.alt)}" width="1536" height="1024"><span class="profile-monogram" aria-hidden="true">ii</span></div><h2>${esc(config.profile.name)}</h2>${config.profile.headline ? `<p>${esc(config.profile.headline)}</p>` : '<p class="muted">소개 준비 중</p>'}<a class="text-link" href="${esc(config.github)}" target="_blank" rel="noopener noreferrer">GitHub ${arrow}</a></aside><div class="profile-copy"><section><p class="section-kicker">ABOUT ME</p><h2>나는 어떤 사람인가.</h2>${config.profile.bio?.trim() ? paragraphs(config.profile.bio) : '<p class="muted">소개와 연구 관심사를 이곳에 차근차근 채워갑니다.</p>'}</section><section><p class="section-kicker">MY PHILOSOPHY</p><h2>${esc(config.philosophy.title)}</h2>${philosophy}</section><section><p class="section-kicker">RESEARCH INTERESTS</p><h2>관심 있는 질문들.</h2>${config.profile.interests?.length ? `<ul class="interest-list">${config.profile.interests.map(value => `<li>${esc(value)}</li>`).join('')}</ul>` : '<p class="muted">연구 관심사를 준비하고 있습니다.</p>'}<a class="text-link" href="${href('/reviews/')}">논문 리뷰 둘러보기 <span aria-hidden="true">→</span></a></section></div></div></main>`;

const reviewIndex = `<main id="main" class="reviews-page"><div class="page-heading"><p class="section-kicker"><span class="kicker-number">02</span><span>PAPER REVIEWS</span></p><h1>읽고, 생각하고,<br>다시 질문합니다<span class="brand-dot" aria-hidden="true">.</span></h1><p class="page-description">논문의 핵심 아이디어와 나의 생각을 주제별로 모읍니다.</p></div><div class="review-toolbar"><div class="topic-tabs" data-topic-tabs role="tablist" aria-label="논문 주제" hidden>${[{ id: 'all', name: '전체', count: reviews.length }, ...topics.map(t => ({ ...t, count: forTopic(t).length }))].map((t, i) => `<button type="button" role="tab" id="tab-${t.id}" aria-controls="review-panel" aria-label="${esc(t.name)}, ${t.count}편" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}" data-topic="${t.id}">${esc(t.name)}<span class="tab-count">${t.count}</span></button>`).join('')}</div><p class="collection-count" data-review-count aria-live="polite">총 ${reviews.length}편의 리뷰</p></div><div id="review-panel" class="review-collection${reviews.length ? '' : ' empty-collection'}" tabindex="0" aria-label="논문 리뷰 목록">${topics.map((topic, i) => {
  const list = forTopic(topic);
  return `<section class="topic-group" data-topic-group="${topic.id}" data-review-total="${list.length}" aria-labelledby="reviews-topic-${topic.id}"><div class="topic-heading"><h2 id="reviews-topic-${topic.id}"><span>${String(i + 1).padStart(2, '0')}</span>${esc(topic.name)}</h2><span class="topic-count">${String(list.length).padStart(2, '0')} REVIEWS</span></div><div class="review-grid">${list.length ? list.map(r => card(r, 'index')).join('') : emptyCard(topic)}</div></section>`;
}).join('')}</div></main>`;

await fs.rm(out, { recursive: true, force: true });
await fs.mkdir(out, { recursive: true });
await fs.cp(path.join(root, 'public'), out, { recursive: true });
async function write(route, text) {
  const file = path.join(out, route.replace(/^\//, ''), 'index.html');
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.writeFile(file, text);
}
await write('/', page({ title: config.title, route: '/', content: home }));
await write('/profile/', page({ title: '내 프로필', route: '/profile/', content: profile }));
await write('/reviews/', page({ title: '논문 리뷰', route: '/reviews/', content: reviewIndex, scripts: `<script src="${href('/assets/reviews.js')}" defer></script>` }));
await write('/about/', `<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=${href('/profile/')}"><link rel="canonical" href="${absolute('/profile/')}"><title>내 프로필로 이동</title></head><body><a href="${href('/profile/')}">내 프로필로 이동</a></body></html>`);
for (const review of reviews) {
  const related = forTopic(review.topic).filter(r => r.slug !== review.slug).slice(0, 3);
  const body = `<main id="main" class="article-page"><a class="text-link back-link" href="${href(topicRoute(review.topic))}"><span aria-hidden="true">←</span> ${esc(review.topic.name)} 리뷰</a><header class="article-header"><p class="section-kicker">${esc(review.topic.name)} / PAPER REVIEW</p><h1>${esc(review.title)}</h1><p class="article-deck">${esc(review.description)}</p><div class="card-meta"><time datetime="${review.date}">${dateText(review.date)}</time><span>${review.readingMinutes}분 읽기</span><span>${esc(config.author)}</span></div>${review.paperTitle || review.paperUrl ? `<div class="paper-source"><span class="section-kicker">ORIGINAL PAPER</span>${review.paperTitle ? `<p>${esc(review.paperTitle)}</p>` : ''}${review.authors || review.year ? `<p class="paper-byline">${esc([review.authors, review.year].filter(Boolean).join(' · '))}</p>` : ''}${review.paperUrl ? `<a class="text-link" href="${esc(review.paperUrl)}" target="_blank" rel="noopener noreferrer">논문 원문 ${arrow}</a>` : ''}</div>` : ''}</header><figure class="article-cover"><img src="${href(review.thumbnail)}" alt="${esc(review.thumbnailAlt || '')}" width="720" height="480"></figure><div class="reading-layout"><article class="prose">${review.html}</article>${review.headings.length ? `<aside class="article-toc"><nav aria-label="이 글의 목차"><span class="section-kicker">IN THIS REVIEW</span><ol>${review.headings.map(h => `<li><a href="#${esc(h.id)}">${esc(h.text)}</a></li>`).join('')}</ol></nav></aside>` : ''}</div>${related.length ? `<section class="related-reviews"><div class="topic-heading"><h2>같은 주제의 리뷰</h2></div><div class="review-grid">${related.map(r => card(r, 'related')).join('')}</div></section>` : ''}</main>`;
  await write(reviewUrl(review), page({ title: review.title, description: review.description, route: reviewUrl(review), content: body, published: review.date }));
}
await fs.writeFile(path.join(out, '404.html'), page({ title: '페이지를 찾을 수 없습니다', route: '/404.html', content: `<main id="main" class="not-found"><p class="section-kicker">404 / NOT FOUND</p><h1>아직 없는 기록입니다.</h1><p>논문 리뷰 목록에서 다른 기록을 찾아보세요.</p><a class="text-link" href="${href('/reviews/')}">논문 리뷰 보기 →</a></main>` }));
await fs.writeFile(path.join(out, 'feed.xml'), `<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel><title>${esc(config.title)}</title><link>${esc(absolute('/'))}</link><description>${esc(config.description)}</description><language>ko</language><atom:link href="${esc(absolute('/feed.xml'))}" rel="self" type="application/rss+xml"/>${reviews.map(r => `<item><title>${esc(r.title)}</title><link>${esc(absolute(reviewUrl(r)))}</link><guid isPermaLink="true">${esc(absolute(reviewUrl(r)))}</guid><description>${esc(r.description)}</description><pubDate>${new Date(`${r.date}T00:00:00+09:00`).toUTCString()}</pubDate></item>`).join('')}</channel></rss>`);
await fs.writeFile(path.join(out, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${['/', '/profile/', '/reviews/', ...reviews.map(reviewUrl)].map(route => `<url><loc>${esc(absolute(route))}</loc></url>`).join('')}</urlset>`);
await fs.writeFile(path.join(out, 'robots.txt'), `User-agent: *\nAllow: /\nSitemap: ${absolute('/sitemap.xml')}\n`);
await fs.writeFile(path.join(out, '.nojekyll'), '');
console.log(`Built ${reviews.length} paper reviews in ${topics.length} topics for ${absolute('/')}`);
