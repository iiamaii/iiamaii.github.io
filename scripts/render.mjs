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
  const forTopic = topic => reviews.filter(review => review.topic.id === topic.id);
  const fallbackBadge = review => review.sourceLang !== lang ? `<span class="original-badge">${esc(t.originalLabel(review.sourceLang))}</span>` : '';

  function page({ title, description = site.description, route, content, published, availableLanguages = languages, canonicalLanguage = lang, scripts = '' }) {
    const languageLinks = languages.map(language => `<a href="${base}${languageRoute(route, language)}" data-language-link hreflang="${language}" lang="${language}" aria-label="${language === 'ko' ? t.koLink : t.enLink}" ${language === lang ? 'aria-current="true"' : ''}>${language === 'ko' ? '한국어' : 'EN'}</a>`).join('');
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
${published ? `<meta property="article:published_time" content="${published}T00:00:00+09:00">` : ''}
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
<a class="brand" href="${href('/')}" aria-label="${esc(site.title)} ${t.home}">${esc(site.title)}<span class="brand-dot" aria-hidden="true">.</span></a>
<span class="brand-caption">${esc(site.author)} / RESEARCH JOURNAL</span>
<nav class="main-nav" aria-label="${t.menu}"><a href="${href('/profile/')}" ${route === '/profile/' ? 'aria-current="page"' : ''}>${t.profile}</a><a href="${href('/reviews/')}" ${route.startsWith('/reviews/') ? 'aria-current="page"' : ''}>${t.reviews}</a><a href="${esc(site.githubPage || site.github)}" target="_blank" rel="noopener noreferrer" aria-label="GitHub page (${t.newTab})">GitHub page ${arrow}</a></nav>
<div class="language-switch" role="group" aria-label="${t.language}">${languageLinks}</div>
</header>
${content}
<footer class="site-footer"><span class="footer-brand">${esc(site.author)}<span class="brand-dot" aria-hidden="true">.</span></span><p>PROFILE · PHILOSOPHY · PAPER REVIEWS</p><span class="copyright">© ${new Date().getFullYear()}</span></footer>
</div>
</body></html>`;
  }

  function logo(className) {
    return site.logo ? `<div class="${className}"><img src="${asset(site.logo.image)}" alt="${esc(site.logo.alt)}" width="4000" height="4000"></div>` : '';
  }

  function card(review, prefix) {
    const titleId = `${prefix}-${review.slug}`;
    return `<article class="review-card" data-content-language="${review.sourceLang}"><a class="card-link" href="${href(reviewRoute(review))}" aria-labelledby="${titleId}"><div class="card-image"><img src="${asset(review.thumbnail)}" alt="" width="720" height="480" loading="lazy"><span class="card-arrow" aria-hidden="true">↗</span></div><div class="card-copy"><span class="card-topic">${esc(review.topic.name)}</span>${fallbackBadge(review)}<h3 id="${titleId}" lang="${review.sourceLang}">${esc(review.title)}</h3><p lang="${review.sourceLang}">${esc(review.description)}</p><div class="card-meta"><time datetime="${review.date}">${review.date.replaceAll('-', '.')}</time><span>${t.readingTime(review.readingMinutes)}</span></div></div></a></article>`;
  }

  function emptyCard(topic) {
    return `<article class="review-card empty-card"><div class="card-image"><img src="${asset(topic.thumbnail)}" alt="" width="720" height="480" loading="lazy"></div><div class="card-copy"><span class="card-topic">${t.preparing}</span><h3>${t.emptyTitle}</h3><p>${esc(topic.description || t.topicIntro)} ${t.emptyIntro}</p><span class="empty-card-note">${t.noReviews}</span></div></article>`;
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
    const content = `<main id="main" class="home-split"><section class="home-introduction" aria-labelledby="philosophy-title"><div class="introduction-inner"><div class="section-kicker"><span class="kicker-number">01</span><span>MY PHILOSOPHY</span><span class="kicker-author">${esc(site.profile.name)}</span></div><figure class="hero-image"><img src="${asset(site.hero.image)}" alt="${esc(site.hero.alt)}" width="1536" height="1024" fetchpriority="high">${logo('hero-logo')}<figcaption><span>THE OPEN NOTEBOOK</span><span aria-hidden="true">01 / ∞</span></figcaption></figure><div class="philosophy-copy"><h1 id="philosophy-title">${esc(site.philosophy.title)}</h1>${philosophy()}<a class="text-link" href="${href('/profile/')}">${t.profileLink} <span aria-hidden="true">→</span></a></div></div></section><section class="home-reviews" aria-labelledby="home-reviews-title"><div class="section-kicker"><span class="kicker-number">02</span><span>PAPER REVIEWS</span><span class="kicker-author">${String(reviews.length).padStart(2, '0')} REVIEWS</span></div><div class="collection-heading"><h2 id="home-reviews-title">${t.collectionTitle}</h2><p>${t.collectionIntro}</p><a class="text-link" href="${href('/reviews/')}">${t.allReviews} <span aria-hidden="true">→</span></a></div>${site.topics.map((topic, i) => {
      const list = forTopic(topic);
      return `<section class="topic-group" aria-labelledby="home-topic-${topic.id}"><div class="topic-heading"><h3 id="home-topic-${topic.id}"><span>${String(i + 1).padStart(2, '0')}</span><a href="${href(topicRoute(topic))}">${esc(topic.name)}</a></h3><a class="topic-count" href="${href(topicRoute(topic))}" aria-label="${esc(t.topicLink(topic.name, list.length))}">${String(list.length).padStart(2, '0')} ${arrow}</a></div><div class="review-grid">${list.length ? list.slice(0, 4).map(review => card(review, 'home')).join('') : emptyCard(topic)}</div>${list.length > 4 ? `<a class="text-link" href="${href(topicRoute(topic))}">${t.moreTopic} →</a>` : ''}</section>`;
    }).join('')}</section></main>`;
    return page({ title: site.title, route: '/', content });
  }

  function profile() {
    const bio = profileContent('profile', 'bio');
    const headline = profileContent('profile', 'headline');
    const interests = profileContent('profile', 'interests');
    const content = `<main id="main" class="profile-page"><div class="page-heading"><p class="section-kicker"><span class="kicker-number">01</span><span>MY PROFILE</span></p><h1>${t.profile}<span class="brand-dot" aria-hidden="true">.</span></h1></div><div class="profile-layout"><aside class="profile-identity"><div class="profile-art"><img src="${asset(site.hero.image)}" alt="${esc(site.hero.alt)}" width="1536" height="1024">${logo('profile-logo')}</div><h2>${esc(site.profile.name)}</h2>${headline.value?.trim() ? `<p lang="${headline.language}">${esc(headline.value)}</p>` : `<p class="muted">${t.introPending}</p>`}<a class="text-link" href="${esc(site.github)}" target="_blank" rel="noopener noreferrer">GitHub ${arrow}</a></aside><div class="profile-copy"><section><p class="section-kicker">ABOUT ME</p><h2>${t.aboutMe}</h2>${bio.value?.trim() ? `${bio.language !== lang ? `<p class="translation-note">${t.profileOriginal}</p>` : ''}<div class="profile-bio" lang="${bio.language}">${paragraphs(bio.value)}</div>` : `<p class="muted">${t.bioPending}</p>`}</section><section><p class="section-kicker">MY PHILOSOPHY</p><h2>${esc(site.philosophy.title)}</h2>${philosophy()}</section><section><p class="section-kicker">RESEARCH INTERESTS</p><h2>${t.interests}</h2>${interests.value?.length ? `<ul class="interest-list" lang="${interests.language}">${interests.value.map(value => `<li>${esc(value)}</li>`).join('')}</ul>` : `<p class="muted">${t.interestsPending}</p>`}<a class="text-link" href="${href('/reviews/')}">${t.browseReviews} <span aria-hidden="true">→</span></a></section></div></div></main>`;
    return page({ title: t.profile, route: '/profile/', content });
  }

  function reviewIndex() {
    const tabs = [{ id: 'all', name: t.all, count: reviews.length }, ...site.topics.map(topic => ({ ...topic, count: forTopic(topic).length }))];
    const content = `<main id="main" class="reviews-page"><div class="page-heading"><p class="section-kicker"><span class="kicker-number">02</span><span>PAPER REVIEWS</span></p><h1>${lines(t.reviewHeading)}<span class="brand-dot" aria-hidden="true">.</span></h1><p class="page-description">${t.reviewDescription}</p></div><div class="review-toolbar"><div class="topic-tabs" data-topic-tabs role="tablist" aria-label="${t.topicTabs}" hidden>${tabs.map((tab, i) => `<button type="button" role="tab" id="tab-${tab.id}" aria-controls="review-panel" aria-label="${esc(t.tabLabel(tab.name, tab.count))}" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}" data-topic="${tab.id}">${esc(tab.name)}<span class="tab-count">${tab.count}</span></button>`).join('')}</div><p class="collection-count" data-review-count aria-live="polite">${t.total(reviews.length)}</p></div><div id="review-panel" class="review-collection${reviews.length ? '' : ' empty-collection'}" tabindex="0" aria-label="${t.reviewList}">${site.topics.map((topic, i) => {
      const list = forTopic(topic);
      return `<section class="topic-group" data-topic-group="${topic.id}" data-review-total="${list.length}" aria-labelledby="reviews-topic-${topic.id}"><div class="topic-heading"><h2 id="reviews-topic-${topic.id}"><span>${String(i + 1).padStart(2, '0')}</span>${esc(topic.name)}</h2><span class="topic-count">${String(list.length).padStart(2, '0')} REVIEWS</span></div><div class="review-grid">${list.length ? list.map(review => card(review, 'index')).join('') : emptyCard(topic)}</div></section>`;
    }).join('')}</div></main>`;
    return page({ title: t.reviews, route: '/reviews/', content, scripts: `<script src="${versioned('/assets/reviews.js')}" defer></script>` });
  }

  function article(review) {
    const related = forTopic(review.topic).filter(other => other.slug !== review.slug).slice(0, 3);
    const content = `<main id="main" class="article-page"><a class="text-link back-link" href="${href(topicRoute(review.topic))}"><span aria-hidden="true">←</span> ${esc(review.topic.name)} / ${t.reviews}</a><header class="article-header"><p class="section-kicker">${esc(review.topic.name)} / PAPER REVIEW</p><h1 lang="${review.sourceLang}">${esc(review.title)}</h1><p class="article-deck" lang="${review.sourceLang}">${esc(review.description)}</p><div class="card-meta"><time datetime="${review.date}">${review.date.replaceAll('-', '.')}</time><span>${t.readingTime(review.readingMinutes)}</span><span>${esc(site.author)}</span></div>${review.sourceLang !== lang ? `<p class="translation-note" role="status">${t.originalNotice(review.sourceLang)}</p>` : ''}${review.paperTitle || review.paperUrl ? `<div class="paper-source"><span class="section-kicker">${t.originalPaper}</span>${review.paperTitle ? `<p>${esc(review.paperTitle)}</p>` : ''}${review.authors || review.year ? `<p class="paper-byline">${esc([review.authors, review.year].filter(Boolean).join(' · '))}</p>` : ''}${review.paperUrl ? `<a class="text-link" href="${esc(review.paperUrl)}" target="_blank" rel="noopener noreferrer">${t.paperLink} ${arrow}</a>` : ''}</div>` : ''}</header><figure class="article-cover"><img src="${asset(review.thumbnail)}" alt="${esc(review.thumbnailAlt || '')}" width="720" height="480"></figure><div class="reading-layout"><article class="prose" lang="${review.sourceLang}">${review.html}</article>${review.headings.length ? `<aside class="article-toc"><nav aria-label="${t.toc}"><span class="section-kicker">IN THIS REVIEW</span><ol>${review.headings.map(heading => `<li><a href="#${esc(heading.id)}" lang="${review.sourceLang}">${esc(heading.text)}</a></li>`).join('')}</ol></nav></aside>` : ''}</div>${related.length ? `<section class="related-reviews"><div class="topic-heading"><h2>${t.related}</h2></div><div class="review-grid">${related.map(other => card(other, 'related')).join('')}</div></section>` : ''}</main>`;
    return page({ title: review.title, description: review.description, route: reviewRoute(review), content, published: review.date, availableLanguages: review.availableLanguages, canonicalLanguage: review.sourceLang });
  }

  const notFound = () => page({ title: t.notFound, route: '/404.html', content: `<main id="main" class="not-found"><p class="section-kicker">404 / NOT FOUND</p><h1>${t.notFoundTitle}</h1><p>${t.notFoundIntro}</p><a class="text-link" href="${href('/reviews/')}">${t.reviewLink} →</a></main>` });
  const redirect = () => `<!doctype html><html lang="${lang}"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=${href('/profile/')}"><link rel="canonical" href="${absolute('/profile/')}"><title>${t.redirect}</title></head><body><a href="${href('/profile/')}">${t.redirect}</a></body></html>`;
  return { home, profile, reviewIndex, article, notFound, redirect, site, absolute, href, esc };
}
