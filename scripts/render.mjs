import { copy, languages, languageRoute, localizedConfig } from './i18n.mjs';

export function createRenderer({ config, lang, reviews, base, assetVersion }) {
  const site = localizedConfig(config, lang);
  const t = copy[lang];
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const href = route => `${base}${languageRoute(route, lang)}`;
  const asset = route => `${base}${route}`;
  const versioned = route => `${asset(route)}?v=${assetVersion}`;
  const absolute = (route, language = lang) => new URL(`${base}${languageRoute(route, language)}`, config.url).href;
  const paragraphs = value => String(value ?? '').split(/\n\s*\n/).filter(Boolean).map(p => `<p>${esc(p).replaceAll('\n', '<br>')}</p>`).join('');
  const lines = value => esc(value).replaceAll('\n', '<br>');
  const arrow = '<span aria-hidden="true">↗</span>';
  const reviewRoute = review => `/reviews/${review.slug}/`;
  const topicRoute = topic => `/reviews/?topic=${encodeURIComponent(topic.id)}#review-panel`;
  const forTopic = topic => reviews.filter(review => review.topics.some(entry => entry.id === topic.id));
  const fallbackBadge = review => review.sourceLang !== lang ? `<span class="original-badge">${esc(t.originalLabel(review.sourceLang))}</span>` : '';

  function page({ title, description = site.description, route, content, published, modified, availableLanguages = languages, canonicalLanguage = lang, scripts = '' }) {
    const languageLinks = languages.map(language => `<a href="${base}${languageRoute(route, language)}" data-language-link hreflang="${language}" lang="${language}" aria-label="${language === 'ko' ? t.koLink : t.enLink}" ${language === lang ? 'aria-current="true"' : ''}>${language === 'ko' ? '한글' : 'EN'}</a>`).join('');
    return `<!doctype html>
<html lang="${lang}">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title === site.title ? title : `${title} · ${site.title}`)}</title>
<meta name="description" content="${esc(description)}"><meta name="theme-color" content="#f7f7f3">
${canonicalLanguage !== lang ? '<meta name="robots" content="noindex,follow">' : ''}
<link rel="canonical" href="${esc(absolute(route, canonicalLanguage))}">
${availableLanguages.map(language => `<link rel="alternate" hreflang="${language}" href="${esc(absolute(route, language))}">`).join('\n')}
<meta property="og:type" content="${published ? 'article' : 'website'}"><meta property="og:locale" content="${t.locale}">
<meta property="og:title" content="${esc(title)}"><meta property="og:description" content="${esc(description)}">
<meta property="og:site_name" content="${esc(site.title)}"><meta property="og:url" content="${esc(absolute(route))}">
<meta property="og:image" content="${esc(new URL(asset(site.hero.image), config.url).href)}">
${published ? `<meta property="article:published_time" content="${esc(published)}"><meta property="article:modified_time" content="${esc(modified || published)}">` : ''}
<link rel="icon" type="image/svg+xml" href="${asset('/favicon.svg')}">
<link rel="stylesheet" href="${versioned('/assets/style.css')}">
<link rel="alternate" type="application/rss+xml" title="${esc(site.title)}" href="${href('/feed.xml')}">
<script src="${versioned('/assets/site.js')}" defer></script>
${scripts}
</head>
<body data-preserve-filters="${route === '/reviews/'}">
<a class="skip-link" href="#main">${t.skip}</a>
<div class="site-shell">
<header class="site-header">
<a class="brand" href="${href('/')}" aria-label="${esc(site.title)} ${t.home}">${site.logo ? `<img class="brand-logo" src="${asset(site.logo.image)}" alt="${esc(site.logo.alt)}" width="72" height="30">` : `${esc(site.title)}<span class="brand-dot" aria-hidden="true">.</span>`}</a>
<span class="brand-caption">${esc(site.author)} / RESEARCH JOURNAL</span>
<nav class="main-nav" aria-label="${t.menu}"><a href="${href('/profile/')}" ${route === '/profile/' ? 'aria-current="page"' : ''}>${t.profile}</a><a href="${href('/reviews/')}" ${route.startsWith('/reviews/') ? 'aria-current="page"' : ''}>${t.reviews}</a><a href="${esc(site.githubPage || site.github)}" target="_blank" rel="noopener noreferrer" aria-label="GitHub page (${t.newTab})">GitHub page ${arrow}</a></nav>
<div class="language-switch" role="group" aria-label="${t.language}">${languageLinks}</div>
</header>
${content}
<footer class="site-footer"><span class="footer-brand">${esc(site.author)}<span class="brand-dot" aria-hidden="true">.</span></span><p>PROFILE · PHILOSOPHY · PAPER REVIEWS</p><span class="copyright">© ${new Date().getFullYear()}</span></footer>
</div>
</body></html>`;
  }

  const formatUpdated = review => review.hasTime
    ? new Intl.DateTimeFormat('sv-SE', { timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(new Date(review.updatedAt)).replaceAll('-', '.')
    : new Intl.DateTimeFormat('sv-SE', { timeZone: 'Asia/Seoul', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(review.updatedAt)).replaceAll('-', '.');
  const reviewTime = review => `<span class="card-updated">${t.updated} <time datetime="${esc(review.updatedAt)}">${formatUpdated(review)}</time></span>`;
  const json = value => JSON.stringify(value).replaceAll('<', '\\u003c').replaceAll('>', '\\u003e').replaceAll('&', '\\u0026');

  function card(review, prefix) {
    const titleId = `${prefix}-${review.slug}`;
    return `<article class="review-card" data-review-slug="${review.slug}" data-content-language="${review.sourceLang}"><a class="card-link" href="${href(reviewRoute(review))}" aria-labelledby="${titleId}"><div class="card-image"><img src="${asset(review.thumbnail)}" alt="" width="720" height="480" loading="lazy"><span class="card-arrow" aria-hidden="true">↗</span></div><div class="card-copy"><span class="card-topic">${esc(review.topics.map(topic => topic.name).join(' · '))}</span>${fallbackBadge(review)}<h3 id="${titleId}" lang="${review.sourceLang}">${esc(review.title)}</h3><p lang="${review.sourceLang}">${esc(review.description)}</p><div class="card-meta">${reviewTime(review)}<span>${t.readingTime(review.readingMinutes)}</span></div></div></a></article>`;
  }

  function emptyReviews() {
    return `<div class="review-empty-state"><div class="review-empty-art"><img src="${asset(site.reviewThumbnail || '/assets/topics/topic-01.svg')}" alt="" width="720" height="480"></div><div class="review-empty-copy"><span class="section-kicker">${t.preparing}</span><h2>${t.emptyCollectionTitle}</h2><p>${t.emptyCollectionIntro}</p></div></div>`;
  }

  function profileContent(section, field) {
    const original = config[section]?.[field];
    const translated = config.translations?.[lang]?.[section]?.[field];
    const has = value => Array.isArray(value) ? value.length > 0 : typeof value === 'string' && value.trim();
    if (lang !== 'ko' && has(translated)) return { value: translated, language: lang };
    return { value: original, language: 'ko' };
  }

  function philosophy() {
    const { value, language } = profileContent('philosophy', 'text');
    if (!value?.trim()) return `<div class="philosophy-text pending-copy"><p>${lines(t.philosophyQuestion)}</p><p class="pending-note">${t.philosophyPending}</p></div>`;
    return `<div class="philosophy-text">${language !== lang ? `<p class="translation-note">${t.profileOriginal}</p>` : ''}<div lang="${language}">${paragraphs(value)}</div></div>`;
  }

  function home() {
    const latest = reviews.slice(0, site.homeReviewLimit);
    const content = `<main id="main" class="home-page"><section class="home-latest" aria-labelledby="home-reviews-title"><div class="section-kicker"><span class="kicker-number">01</span><span>LATEST REVIEWS</span><span class="kicker-author">${String(reviews.length).padStart(2, '0')} REVIEWS</span></div><div class="latest-heading"><h1 id="home-reviews-title">${t.collectionTitle}</h1><p>${t.latestIntro(latest.length)}</p></div>${latest.length ? `<div class="review-grid">${latest.map(review => card(review, 'home')).join('')}</div>` : emptyReviews()}<div class="home-more"><a class="button-link" href="${href('/reviews/')}">${t.moreReviews} <span aria-hidden="true">→</span></a></div></section><section class="home-profile" aria-labelledby="philosophy-title"><div class="section-kicker"><span class="kicker-number">02</span><span>PROFILE & PHILOSOPHY</span><span class="kicker-author">${esc(site.profile.name)}</span></div><div class="home-profile-layout"><figure class="hero-image"><img src="${asset(site.hero.image)}" alt="${esc(site.hero.alt)}" width="1536" height="1024" loading="lazy"><figcaption><span>THE OPEN NOTEBOOK</span><span aria-hidden="true">01 / ∞</span></figcaption></figure><div class="philosophy-copy"><p class="profile-name">${esc(site.profile.name)}</p><h2 id="philosophy-title">${esc(site.philosophy.title)}</h2>${philosophy()}<a class="text-link" href="${href('/profile/')}">${t.profileLink} <span aria-hidden="true">→</span></a></div></div></section></main>`;
    return page({ title: site.title, route: '/', content });
  }

  function profile() {
    const bio = profileContent('profile', 'bio');
    const headline = profileContent('profile', 'headline');
    const interests = profileContent('profile', 'interests');
    const content = `<main id="main" class="profile-page"><div class="page-heading"><p class="section-kicker"><span class="kicker-number">01</span><span>MY PROFILE</span></p><h1>${t.profile}<span class="brand-dot" aria-hidden="true">.</span></h1></div><div class="profile-layout"><aside class="profile-identity"><div class="profile-art"><img src="${asset(site.hero.image)}" alt="${esc(site.hero.alt)}" width="1536" height="1024"></div><h2>${esc(site.profile.name)}</h2>${headline.value?.trim() ? `<p lang="${headline.language}">${esc(headline.value)}</p>` : `<p class="muted">${t.introPending}</p>`}<a class="text-link" href="${esc(site.github)}" target="_blank" rel="noopener noreferrer">GitHub ${arrow}</a></aside><div class="profile-copy"><section><p class="section-kicker">ABOUT ME</p><h2>${t.aboutMe}</h2>${bio.value?.trim() ? `${bio.language !== lang ? `<p class="translation-note">${t.profileOriginal}</p>` : ''}<div class="profile-bio" lang="${bio.language}">${paragraphs(bio.value)}</div>` : `<p class="muted">${t.bioPending}</p>`}</section><section><p class="section-kicker">MY PHILOSOPHY</p><h2>${esc(site.philosophy.title)}</h2>${philosophy()}</section><section><p class="section-kicker">RESEARCH INTERESTS</p><h2>${t.interests}</h2>${interests.value?.length ? `<ul class="interest-list" lang="${interests.language}">${interests.value.map(value => `<li>${esc(value)}</li>`).join('')}</ul>` : `<p class="muted">${t.interestsPending}</p>`}<a class="text-link" href="${href('/reviews/')}">${t.browseReviews} <span aria-hidden="true">→</span></a></section></div></div></main>`;
    return page({ title: t.profile, route: '/profile/', content });
  }

  function reviewIndex() {
    const u = t.reviewUi;
    const topicOptions = site.topics.map(topic => `<label class="topic-option"><input type="checkbox" name="topic" value="${topic.id}"><span data-topic="${topic.id}">${esc(topic.name)}</span><span class="topic-count" aria-hidden="true">${forTopic(topic).length}</span></label>`).join('');
    const records = reviews.map(review => ({ slug: review.slug, topics: review.topics.map(topic => topic.id), publishedAt: review.publishedAt, updatedAt: review.updatedAt, text: `${review.searchText} ${review.topics.map(topic => `${topic.key} ${topic.name}`).join(' ')}` }));
    const content = `<main id="main" class="reviews-page">
<div class="page-heading"><p class="section-kicker"><span class="kicker-number">01</span><span>PAPER REVIEWS</span></p><h1>${lines(t.reviewHeading)}<span class="brand-dot" aria-hidden="true">.</span></h1><p class="page-description">${t.reviewDescription}</p></div>
<div class="review-toolbar" data-review-tools hidden>
  <div class="review-controls">
    <div class="review-tool-buttons">
      <button type="button" class="filter-toggle icon-toggle" data-filter-toggle aria-label="${u.filterToggle}" title="${u.filterToggle}" aria-expanded="false" aria-controls="review-filters"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M4 5h16l-6.5 7.5V19l-3-1.5v-5z"/></svg><span id="filter-status" class="filter-active" data-filter-active hidden><span class="visually-hidden">${u.active}</span></span></button>
      <button type="button" class="filter-toggle icon-toggle" data-search-toggle aria-label="${u.searchToggle}" title="${u.searchToggle}" aria-expanded="false" aria-controls="review-search"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/></svg><span id="search-status" class="filter-active" data-search-active hidden><span class="visually-hidden">${u.active}</span></span></button>
    </div>
    <div class="review-list-options">
      <label class="review-size"><span>${u.sort}</span><select name="sort" data-review-sort aria-label="${u.sort}" title="${u.newest}"><option value="newest">${u.newestShort}</option><option value="oldest">${u.oldestShort}</option></select></label>
      <label class="review-size"><span>${u.size}</span><select name="size" data-review-size aria-label="${u.size}" title="${u.size}"><option value="6">6${u.sizeUnit}</option><option value="12" selected>12${u.sizeUnit}</option><option value="24">24${u.sizeUnit}</option></select></label>
    </div>
  </div>
  <form id="review-filters" class="review-filters" data-review-filters action="${href('/reviews/')}" method="get" hidden>
    <details class="topic-picker" data-topic-picker><summary>${u.topic}<span data-topic-selection>${u.allTopics}</span></summary>
      <fieldset><legend>${u.topic}</legend><p class="topic-hint">${u.topicHint}</p><div class="topic-checklist">${topicOptions || `<p class="topic-hint">${u.noTopics}</p>`}</div><button type="button" class="topic-clear" data-clear-topics>${u.clearTopics}</button></fieldset>
    </details>
    <label>${u.from}<input type="datetime-local" name="from" step="60"></label>
    <label>${u.to}<input type="datetime-local" name="to" step="60"></label>
    <div class="filter-actions"><button type="submit" class="button-link">${u.filterApply}</button><button type="button" class="secondary-button" data-reset-filters>${u.filterReset}</button><p>${u.timeHint}</p></div>
  </form>
  <form id="review-search" class="review-filters review-search" data-review-search action="${href('/reviews/')}" method="get" hidden>
    <label class="filter-search">${u.search}<input type="search" name="q" placeholder="${u.searchPlaceholder}"></label>
    <div class="filter-actions"><button type="submit" class="button-link">${u.apply}</button><button type="button" class="secondary-button" data-reset-search>${u.searchReset}</button></div>
  </form>
</div>
<p class="collection-count review-result-count" data-review-count aria-live="polite">${t.total(reviews.length)}</p>
<div id="review-panel" class="review-collection${reviews.length ? '' : ' is-empty'}" role="region" aria-label="${t.reviewList}">
  <div class="review-grid" data-review-grid>${reviews.map(review => card(review, 'index')).join('')}</div>
  ${reviews.length ? '' : emptyReviews()}
  <div class="review-no-results" data-no-results hidden><h2>${u.noResults}</h2><p>${u.noResultsHint}</p><button type="button" class="secondary-button" data-clear-search>${u.reset}</button></div>
</div>
<div class="review-more" data-review-more hidden><p data-load-progress aria-hidden="true"></p><button type="button" class="secondary-button" data-load-more>${u.loadMore}</button></div>
<script id="review-data" type="application/json">${json({ reviews: records, ui: u })}</script>
</main>`;
    return page({ title: t.reviews, route: '/reviews/', content, scripts: `<script type="module" src="${versioned('/assets/reviews.js')}"></script>` });
  }

  function article(review) {
    const primaryTopic = review.topics[0];
    const related = reviews.filter(other => other.slug !== review.slug && other.topics.some(topic => review.topics.some(entry => entry.id === topic.id))).slice(0, 3);
    const content = `<main id="main" class="article-page"><a class="text-link back-link" href="${href(topicRoute(primaryTopic))}"><span aria-hidden="true">←</span> ${esc(primaryTopic.name)} / ${t.reviews}</a><header class="article-header"><p class="section-kicker">${esc(review.topics.map(topic => topic.name).join(' · '))} / PAPER REVIEW</p><h1 lang="${review.sourceLang}">${esc(review.title)}</h1><p class="article-deck" lang="${review.sourceLang}">${esc(review.description)}</p><div class="card-meta">${reviewTime(review)}<span>${t.readingTime(review.readingMinutes)}</span><span>${esc(site.author)}</span></div>${review.sourceLang !== lang ? `<p class="translation-note" role="status">${t.originalNotice(review.sourceLang)}</p>` : ''}${review.paperTitle || review.paperUrl ? `<div class="paper-source"><span class="section-kicker">${t.originalPaper}</span>${review.paperTitle ? `<p>${esc(review.paperTitle)}</p>` : ''}${review.authors || review.year ? `<p class="paper-byline">${esc([review.authors, review.year].filter(Boolean).join(' · '))}</p>` : ''}${review.paperUrl ? `<a class="text-link" href="${esc(review.paperUrl)}" target="_blank" rel="noopener noreferrer">${t.paperLink} ${arrow}</a>` : ''}</div>` : ''}</header><figure class="article-cover"><img src="${asset(review.thumbnail)}" alt="${esc(review.thumbnailAlt || '')}" width="720" height="480"></figure><div class="reading-layout"><article class="prose" lang="${review.sourceLang}">${review.html}</article>${review.headings.length ? `<aside class="article-toc"><nav aria-label="${t.toc}"><span class="section-kicker">IN THIS REVIEW</span><ol>${review.headings.map(heading => `<li><a href="#${esc(heading.id)}" lang="${review.sourceLang}">${esc(heading.text)}</a></li>`).join('')}</ol></nav></aside>` : ''}</div>${related.length ? `<section class="related-reviews"><div class="topic-heading"><h2>${t.related}</h2></div><div class="review-grid">${related.map(other => card(other, 'related')).join('')}</div></section>` : ''}</main>`;
    return page({ title: review.title, description: review.description, route: reviewRoute(review), content, published: review.publishedAt, modified: review.updatedAt, availableLanguages: review.availableLanguages, canonicalLanguage: review.sourceLang });
  }

  const notFound = () => page({ title: t.notFound, route: '/404.html', content: `<main id="main" class="not-found"><p class="section-kicker">404 / NOT FOUND</p><h1>${t.notFoundTitle}</h1><p>${t.notFoundIntro}</p><a class="text-link" href="${href('/reviews/')}">${t.reviewLink} →</a></main>` });
  const redirect = () => `<!doctype html><html lang="${lang}"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=${href('/profile/')}"><link rel="canonical" href="${absolute('/profile/')}"><title>${t.redirect}</title></head><body><a href="${href('/profile/')}">${t.redirect}</a></body></html>`;
  return { home, profile, reviewIndex, article, notFound, redirect, site, absolute, href, esc };
}
