import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { paperMetadata } from './paper-metadata.mjs';
import { postTimes } from './post-times.mjs';
import { createRenderer } from './render.mjs';

const data = { paperPublishedDate: '2025-10-22', authors: ['Pascal Bergsträßer', 'Ryan Cotterell', 'Anthony W. Lin'] };
test('paper dates and complete ordered names survive legacy and list inputs', () => {
  assert.deepEqual(paperMetadata(data, 'paper'), data);
  assert.deepEqual(paperMetadata({ ...data, authors: data.authors.join(', ') }, 'legacy'), data);
  for (const paperPublishedDate of [undefined, '', '2025-02-29', '2026-13-01', '2026-10-05T00:00:00Z']) assert.throws(() => paperMetadata({ ...data, paperPublishedDate }, 'paper'));
  for (const authors of [undefined, [], [''], [null], ['First Author et al.'], 'First Author, ']) assert.throws(() => paperMetadata({ ...data, authors }, 'paper'));
});

test('unknown original upload time stays date-only even after a timed update', () => {
  const times = postTimes({ updatedAt: '2026-10-09T05:04:32Z' }, '2026-10-06', 'paper');
  assert.equal(times.publishedAt, '2026-10-05T15:00:00.000Z');
  assert.equal(times.hasPublishedTime, false);
  assert.equal(times.hasTime, true);
});

test('bilingual articles distinguish upload, release and expandable escaped authors', () => {
  const config = JSON.parse(fs.readFileSync(new URL('../site.json', import.meta.url), 'utf8'));
  const review = { ...data, authors: ['First <Author>', 'Second & Author'], slug: 'paper', sourceLang: 'ko', title: 'Review', description: 'Summary', topics: [{ id: 'topic', name: 'Topic' }], thumbnail: '/assets/topics/topic-01.svg', publishedAt: '2026-10-06T03:00:00Z', updatedAt: '2026-10-09T05:04:32Z', hasTime: true, hasPublishedTime: true, paperTitle: 'Original paper', readingMinutes: 1, html: '', headings: [] };
  for (const lang of ['ko', 'en']) {
    const renderer = createRenderer({ config, lang, reviews: [], base: '', assetVersion: 'test' });
    const html = renderer.article(review);
    assert.ok(html.includes('2026.10.06 12:00') && html.includes('2025.10.22'));
    assert.ok(html.includes('First &lt;Author&gt; et al.') && html.includes('First &lt;Author&gt; · Second &amp; Author'));
    assert.ok(html.includes('<details class="paper-authors">') && !html.includes('<details class="paper-authors" open'));
    assert.ok(html.includes(lang === 'ko' ? '전체 저자 보기' : 'Show all authors'));
    const single = renderer.article({ ...review, authors: ['Only Author'] });
    assert.ok(single.includes('Only Author') && !single.includes('et al.') && !single.includes('<details class="paper-authors">'));
  }
});
