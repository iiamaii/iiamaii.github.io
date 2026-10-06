import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import matter from 'gray-matter';
import { Marked, Renderer } from 'marked';
import { languages, languageRoute, localizedConfig, copy } from './i18n.mjs';
import { createRenderer } from './render.mjs';

const root = process.cwd();
const out = path.join(root, 'dist');
const config = JSON.parse(await fs.readFile(path.join(root, 'site.json'), 'utf8'));
const base = (process.env.SITE_BASE_PATH ?? config.basePath ?? '').replace(/\/$/, '');
if (base && !/^\/[a-zA-Z0-9_-]+$/.test(base)) throw new Error('basePath must be empty or a single /repository path.');
function external(value) {
  if (!['http:', 'https:'].includes(new URL(value).protocol)) throw new Error(`Use an http(s) URL: ${value}`);
}
external(config.url);
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
if (config.logo) await checkImage(config.logo.image);
await Promise.all(topics.map(topic => checkImage(topic.thumbnail)));

function renderMarkdown(source, lang) {
  const headings = [];
  const used = new Set();
  const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
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
      return `<h${depth} id="${escape(id)}">${this.parser.parseInline(tokens)}</h${depth}>\n`;
    },
    table(token) {
      return `<div class="table-scroll" role="region" aria-label="${copy[lang].table}" tabindex="0">${Renderer.prototype.table.call(this, token)}</div>`;
    }
  } });
  return { html: parser.parse(source), headings };
}

const pairs = new Map();
const today = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Seoul' });
const reviewDirectory = path.join(root, 'content/reviews');
await fs.mkdir(reviewDirectory, { recursive: true });
for (const file of (await fs.readdir(reviewDirectory)).filter(name => name.endsWith('.md'))) {
  const { data, content } = matter(await fs.readFile(path.join(reviewDirectory, file), 'utf8'));
  if (data.draft === true) continue;
  const suffixLanguage = file.match(/\.(ko|en)\.md$/)?.[1];
  const lang = data.lang || suffixLanguage || 'ko';
  if (!languages.includes(lang) || (suffixLanguage && suffixLanguage !== lang)) throw new Error(`${file}: lang must be ko or en and match the filename suffix.`);
  if (typeof data.title !== 'string' || !data.title.trim() || typeof data.description !== 'string' || !data.description.trim()) throw new Error(`${file}: title and description are required.`);
  const topic = topics.find(item => item.id === data.topic);
  if (!topic) throw new Error(`${file}: topic must match an id in site.json.`);
  const date = data.date instanceof Date ? data.date.toISOString().slice(0, 10) : String(data.date);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || Number.isNaN(Date.parse(date)) || new Date(date).toISOString().slice(0, 10) !== date) throw new Error(`${file}: date must be a valid YYYY-MM-DD.`);
  if (date > today) continue;
  const slug = data.translationKey || file.replace(/\.md$/, '').replace(/\.(ko|en)$/, '').replace(/^\d{4}-\d{2}-\d{2}-/, '');
  if (typeof slug !== 'string' || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) throw new Error(`${file}: use a lowercase slug or translationKey with hyphens.`);
  if (data.thumbnail) await checkImage(data.thumbnail);
  if (data.paperUrl) external(data.paperUrl);
  const pair = pairs.get(slug) || {};
  if (pair[lang]) throw new Error(`${file}: duplicate ${lang} review for ${slug}.`);
  const other = pair[lang === 'ko' ? 'en' : 'ko'];
  if (other && other.topic.id !== topic.id) throw new Error(`${file}: translations of the same review must use the same topic.`);
  const readingMinutes = lang === 'en' ? Math.max(1, Math.ceil(content.trim().split(/\s+/).length / 220)) : Math.max(1, Math.ceil(content.replace(/\s/g, '').length / 500));
  pair[lang] = { ...data, date, slug, topic, sourceLang: lang, readingMinutes, ...renderMarkdown(content, lang) };
  pairs.set(slug, pair);
}

const assetBuffers = await Promise.all(['style.css', 'site.js', 'reviews.js'].map(file => fs.readFile(path.join(root, 'public/assets', file))));
const assetVersion = createHash('sha256').update(Buffer.concat(assetBuffers)).digest('hex').slice(0, 12);
await fs.rm(out, { recursive: true, force: true });
await fs.mkdir(out, { recursive: true });
await fs.cp(path.join(root, 'public'), out, { recursive: true });
async function writePage(route, html) {
  const file = path.join(out, route.replace(/^\//, ''), 'index.html');
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.writeFile(file, html);
}
const sitemapRoutes = [];
for (const lang of languages) {
  const site = localizedConfig(config, lang);
  const reviews = [...pairs.values()].map(pair => {
    const source = pair[lang] || pair[lang === 'ko' ? 'en' : 'ko'];
    const other = pair[source.sourceLang === 'ko' ? 'en' : 'ko'];
    return {
      ...source,
      topic: site.topics.find(topic => topic.id === source.topic.id),
      thumbnail: source.thumbnail || other?.thumbnail || source.topic.thumbnail,
      thumbnailAlt: source.thumbnailAlt || other?.thumbnailAlt || '',
      paperTitle: source.paperTitle || other?.paperTitle || '',
      paperUrl: source.paperUrl || other?.paperUrl || '',
      authors: source.authors || other?.authors || '',
      year: source.year || other?.year || '',
      availableLanguages: languages.filter(language => pair[language])
    };
  }).sort((a, b) => b.date.localeCompare(a.date) || a.slug.localeCompare(b.slug));
  const renderer = createRenderer({ config, lang, reviews, base, assetVersion });
  await writePage(languageRoute('/', lang), renderer.home());
  await writePage(languageRoute('/profile/', lang), renderer.profile());
  await writePage(languageRoute('/reviews/', lang), renderer.reviewIndex());
  await writePage(languageRoute('/about/', lang), renderer.redirect());
  for (const review of reviews) await writePage(languageRoute(`/reviews/${review.slug}/`, lang), renderer.article(review));
  await fs.writeFile(path.join(out, languageRoute('/404.html', lang).slice(1)), renderer.notFound());
  const feedReviews = reviews.filter(review => review.sourceLang === lang);
  const { esc, absolute } = renderer;
  await fs.writeFile(path.join(out, languageRoute('/feed.xml', lang).slice(1)), `<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel><title>${esc(site.title)}</title><link>${esc(absolute('/'))}</link><description>${esc(site.description)}</description><language>${lang}</language><atom:link href="${esc(absolute('/feed.xml'))}" rel="self" type="application/rss+xml"/>${feedReviews.map(review => `<item><title>${esc(review.title)}</title><link>${esc(absolute(`/reviews/${review.slug}/`))}</link><guid isPermaLink="true">${esc(absolute(`/reviews/${review.slug}/`))}</guid><description>${esc(review.description)}</description><pubDate>${new Date(`${review.date}T00:00:00+09:00`).toUTCString()}</pubDate></item>`).join('')}</channel></rss>`);
  sitemapRoutes.push(...['/', '/profile/', '/reviews/', ...feedReviews.map(review => `/reviews/${review.slug}/`)].map(route => absolute(route)));
}
const xmlEscape = value => value.replaceAll('&', '&amp;');
await fs.writeFile(path.join(out, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${sitemapRoutes.map(url => `<url><loc>${xmlEscape(url)}</loc></url>`).join('')}</urlset>`);
await fs.writeFile(path.join(out, 'robots.txt'), `User-agent: *\nAllow: /\nSitemap: ${new URL(`${base}/sitemap.xml`, config.url).href}\n`);
await fs.writeFile(path.join(out, '.nojekyll'), '');
console.log(`Built ${pairs.size} paper reviews in Korean and English for ${config.url}${base}/`);
