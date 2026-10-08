import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import matter from 'gray-matter';

const root = process.cwd();
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'blog-review-check-'));
const keepPreview = process.argv.includes('--preview');
let previewReady = false;
const build = (base = '') => spawnSync(process.execPath, ['scripts/build.mjs'], { cwd: temp, env: { ...process.env, SITE_BASE_PATH: base }, encoding: 'utf8' });
const read = route => fs.readFile(path.join(temp, 'dist', route), 'utf8');

try {
  for (const entry of ['scripts', 'public', 'site.json']) await fs.cp(path.join(root, entry), path.join(temp, entry), { recursive: true });
  await fs.symlink(path.join(root, 'node_modules'), path.join(temp, 'node_modules'), 'dir');
  await fs.mkdir(path.join(temp, 'content/reviews'), { recursive: true });
  const config = JSON.parse(await fs.readFile(path.join(temp, 'site.json'), 'utf8'));
  config.topics = [
    { id: 'alpha', name: 'Alpha', thumbnail: '/assets/topics/topic-01.svg' },
    { id: 'beta', name: 'Beta', thumbnail: '/assets/topics/topic-02.svg' },
    { id: 'empty', name: 'Empty', thumbnail: '/assets/topics/topic-03.svg' }
  ];
  config.profile.bio = 'A profile & an idea.';
  config.philosophy.text = 'First paragraph.\n\nSecond paragraph.';
  config.translations.en.profile.bio = 'An English introduction.';
  config.translations.en.philosophy.text = 'English philosophy.\n\nA second English paragraph.';
  await fs.writeFile(path.join(temp, 'site.json'), JSON.stringify(config));
  const fixture = async (slug, fields, body = '## Repeated question\n\nA paragraph.\n\n## Repeated question\n\nAnother paragraph.') => {
    const data = { title: `Fixture ${slug} & "question"`, description: 'A concise test description.', date: '2020-01-01', ...(fields.topics !== undefined ? {} : { topic: 'alpha' }), visibility: 'public', ...fields };
    const metadata = Object.entries(data).filter(([, value]) => value !== undefined).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n');
    await fs.writeFile(path.join(temp, 'content/reviews', `${slug}.md`), `---\n${metadata}\n---\n\n${body}\n`);
  };
  for (let i = 1; i <= 5; i++) await fixture(`alpha-${i}`, { date: `2020-01-0${i}` });
  await fixture('beta-1', { topic: 'beta', date: '2020-01-06', paperTitle: 'Original & paper', paperUrl: 'https://example.org/paper?x=1&y=2', thumbnail: '/assets/topics/topic-03.svg' });
  await fixture('beta-2', { topic: 'beta', date: '2020-01-07' });
  await fixture('draft', { draft: true, topic: 'draft-only-topic' });
  await fixture('future', { date: '9999-01-01', topic: 'future-only-topic' });
  await fixture('private', { visibility: 'private', topic: '비공개 연구', title: 'Private review sentinel' }, 'PRIVATE_REVIEW_BODY_SENTINEL');
  await fixture('private-multi', { visibility: 'private', topics: ['비공개 전용', '비공개 비전'], title: 'Private multi review sentinel' }, 'PRIVATE_MULTI_BODY_SENTINEL');
  await fixture('unmarked', { visibility: undefined, topic: '미지정 연구', title: 'Unmarked review sentinel' }, 'UNMARKED_REVIEW_BODY_SENTINEL');
  await fixture('private-translation.en', { lang: 'en', translationKey: 'alpha-4', visibility: 'private', title: 'Private translation sentinel' }, 'PRIVATE_TRANSLATION_BODY_SENTINEL');
  let result = build();
  assert.equal(result.status, 0, result.stderr);
  const home = await read('index.html');
  const index = await read('reviews/index.html');
  const article = await read('reviews/beta-1/index.html');
  const profile = await read('profile/index.html');
  const sitemap = await read('sitemap.xml');
  const rss = await read('feed.xml');
  assert.equal((home.match(/class="review-card"/g) || []).length, 6, 'Home shows at most six unique reviews across all topics.');
  assert.equal((index.match(/class="review-card"/g) || []).length, 7, 'The review index shows all published reviews.');
  assert.ok(!home.includes('/reviews/alpha-1/'), 'The oldest fifth review stays off the home page.');
  assert.ok(index.includes('/reviews/alpha-1/'));
  assert.ok(home.indexOf('/reviews/alpha-5/') < home.indexOf('/reviews/alpha-4/'), 'Reviews sort newest first.');
  const records = html => JSON.parse(html.match(/<script id="review-data" type="application\/json">([\s\S]*?)<\/script>/)[1]).reviews;
  assert.equal(records(index).filter(review => review.topics.includes('alpha')).length, 5);
  assert.equal(records(index).filter(review => review.topics.includes('beta')).length, 2);
  assert.ok(home.indexOf('class="home-latest"') < home.indexOf('class="home-profile"'), 'Reviews precede the profile in document order.');
  assert.ok(!index.includes('data-topic-group="empty"') && !index.includes('data-topic="empty"'), 'Unused topic presets do not create filter choices.');
  for (const name of ['비공개 연구', '비공개 전용', '비공개 비전', '미지정 연구', 'draft-only-topic', 'future-only-topic']) assert.ok(!index.includes(name), 'Only public, ready reviews contribute topics.');
  assert.ok(index.includes('Fixture beta-1 &amp; &quot;question&quot;'), 'Metadata is escaped in cards.');
  assert.ok(article.includes('https://example.org/paper?x=1&amp;y=2'));
  assert.ok(article.includes('/assets/topics/topic-03.svg'), 'A review can override its topic thumbnail.');
  assert.ok((await read('reviews/alpha-1/index.html')).includes('/assets/topics/topic-01.svg'), 'Missing review thumbnails use the topic image.');
  assert.ok(article.includes('id="repeated-question-2"'), 'Repeated headings have distinct anchors.');
  assert.ok(profile.includes('A profile &amp; an idea.') && profile.includes('<p>Second paragraph.</p>'));
  assert.ok(home.includes('<p>Second paragraph.</p>'), 'The home and profile share the same philosophy.');
  assert.ok(sitemap.includes('/profile/') && sitemap.includes('/reviews/beta-1/'));
  assert.equal((rss.match(/<item>/g) || []).length, 7);
  for (const excluded of ['draft', 'future', 'private', 'private-multi', 'unmarked']) {
    for (const prefix of ['', 'en/']) await assert.rejects(fs.access(path.join(temp, 'dist', `${prefix}reviews`, excluded, 'index.html')));
    assert.ok(!sitemap.includes(`/reviews/${excluded}/`) && !rss.includes(`/reviews/${excluded}/`));
  }
  await assert.rejects(fs.access(path.join(temp, 'dist/posts')), 'The old sample post directory is absent.');
  assert.ok((await read('about/index.html')).includes('url=/profile/'));
  assert.ok(home.includes('class="brand-logo"') && home.includes('/assets/brand-logo-transparent.png'), 'The transparent logo is the header home link.');
  for (const route of ['index.html', 'profile/index.html', 'reviews/index.html', 'en/index.html', 'en/profile/index.html', 'en/reviews/index.html']) {
    const html = await read(route);
    assert.ok(!html.includes('href="/fonts/"'), 'The temporary font page has no public navigation links.');
    assert.ok(!/<footer[^>]*>[\s\S]*?href=/.test(html), 'The footer has no utility links.');
    assert.ok(html.includes('data-language-link'), 'Every public page has a language switch.');
  }
  assert.ok((await read('fonts/index.html')).includes('noindex'), 'The temporary font preview stays available but outside search results.');
  assert.ok((await read('en/index.html')).includes('<html lang="en">'));
  assert.ok((await read('en/profile/index.html')).includes('An English introduction.'));
  assert.ok((await read('en/index.html')).includes('A second English paragraph.'));
  const fallback = await read('en/reviews/beta-1/index.html');
  assert.ok(fallback.includes('An English translation is not available yet.'));
  assert.ok(fallback.includes('<article class="prose" lang="ko">'));
  assert.ok(fallback.includes('name="robots" content="noindex,follow"'));
  assert.ok(fallback.includes(`href="${config.url}/reviews/beta-1/"`), 'Untranslated pages point to the original canonical URL.');

  await fixture('translated.en', { lang: 'en', translationKey: 'alpha-5', title: 'A translated paper review', description: 'An English summary.', topic: 'alpha' }, '## The English idea\n\nThis is the English review body.');
  await fixture('en-only.en', { lang: 'en', translationKey: 'en-only', title: 'English-only paper review', topic: 'beta' }, '## An English original\n\nAn English-only review body.');
  await fixture('pending.en', { lang: 'en', translationKey: 'alpha-4', draft: true, title: 'Unpublished translation' });
  result = build();
  assert.equal(result.status, 0, result.stderr);
  const translated = await read('en/reviews/alpha-5/index.html');
  assert.ok(translated.includes('A translated paper review') && translated.includes('This is the English review body.'));
  assert.ok(translated.includes('<article class="prose" lang="en">'));
  assert.ok(!translated.includes('An English translation is not available yet.'));
  assert.ok(translated.includes(`href="/reviews/alpha-5/" data-language-link`), 'Switching language keeps the same post key.');
  assert.ok((await read('reviews/alpha-5/index.html')).includes('href="/en/reviews/alpha-5/" data-language-link'));
  assert.equal(((await read('reviews/index.html')).match(/class="review-card"/g) || []).length, 8, 'Two translations are one paper, not two cards.');
  assert.equal(((await read('en/reviews/index.html')).match(/class="review-card"/g) || []).length, 8);
  assert.ok((await read('reviews/en-only/index.html')).includes('한국어 번역은 아직 준비되지 않아 영어 원문을 표시합니다.'));
  assert.ok((await read('en/reviews/alpha-4/index.html')).includes('An English translation is not available yet.'));
  assert.ok(!(await read('en/reviews/index.html')).includes('Unpublished translation'));
  assert.equal(((await read('feed.xml')).match(/<item>/g) || []).length, 7);
  assert.equal(((await read('en/feed.xml')).match(/<item>/g) || []).length, 2, 'Each feed contains its published language versions.');

  result = spawnSync(process.execPath, ['scripts/new-post.mjs', 'new-draft', '새로운 주제'], { cwd: temp, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const created = (await fs.readdir(path.join(temp, 'content/reviews'))).filter(name => name.includes('new-draft'));
  assert.equal(created.length, 2, 'The authoring command creates both language drafts.');
  for (const file of created) {
    const { data } = matter(await fs.readFile(path.join(temp, 'content/reviews', file), 'utf8'));
    assert.equal(data.visibility, 'private');
    assert.ok(Number.isFinite(Date.parse(data.publishedAt)) && data.updatedAt === data.publishedAt, 'Drafts include publication and update timestamps.');
    assert.deepEqual(data.topics, ['새로운 주제'], 'New language versions start private and accept unregistered topics.');
  }
  result = spawnSync(process.execPath, ['scripts/new-post.mjs', 'new-multi', '머신러닝', ' 컴퓨터   비전 ', '컴퓨터 비전'], { cwd: temp, encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  for (const file of (await fs.readdir(path.join(temp, 'content/reviews'))).filter(name => name.includes('new-multi'))) {
    const { data } = matter(await fs.readFile(path.join(temp, 'content/reviews', file), 'utf8'));
    assert.equal(data.visibility, 'private');
    assert.deepEqual(data.topics, ['머신러닝', '컴퓨터 비전'], 'The authoring command accepts and deduplicates multiple topics in both languages.');
  }
  result = spawnSync(process.execPath, ['scripts/new-post.mjs', 'new-draft', 'alpha'], { cwd: temp, encoding: 'utf8' });
  assert.notEqual(result.status, 0, 'The authoring command does not overwrite existing drafts.');
  result = build('/project');
  assert.equal(result.status, 0, result.stderr);
  assert.ok((await read('index.html')).includes('href="/project/profile/"'));
  assert.ok((await read('reviews/index.html')).includes('src="/project/assets/reviews.js?v='));
  assert.ok((await read('reviews/beta-1/index.html')).includes('href="/project/reviews/?topic=beta#review-panel"'));
  assert.ok((await read('en/reviews/alpha-5/index.html')).includes('href="/project/reviews/alpha-5/" data-language-link'));
  assert.ok((await read('en/reviews/index.html')).includes('src="/project/assets/site.js?v='));
  await fixture('vision-1', { topic: '컴퓨터 비전', topicName: '컴퓨터 비전', title: 'A public vision review' });
  await fixture('vision-2', { topic: '  컴퓨터   비전  ', title: 'Another public vision review' });
  await fixture('vision.en', { lang: 'en', translationKey: 'vision-1', topic: '컴퓨터 비전', topicName: 'Computer Vision', title: 'An English vision review' });
  result = build();
  assert.equal(result.status, 0, result.stderr);
  const dynamicIndex = await read('reviews/index.html');
  const visionId = dynamicIndex.match(/data-topic="([^"]+)">컴퓨터 비전<\/span>/)?.[1];
  assert.ok(visionId, 'A topic name in a public post automatically creates a checkbox choice without a preset.');
  assert.equal(records(dynamicIndex).filter(review => review.topics.includes(visionId)).length, 2, 'Normalized topics share the same filter id.');
  assert.ok((await read('en/reviews/index.html')).includes(`data-topic="${visionId}">Computer Vision</span>`), 'Translated topic labels share the same filter id.');
  assert.ok((await read('reviews/vision-1/index.html')).includes('/assets/topics/topic-01.svg'), 'New topics have a default thumbnail.');
  const checkNoPrivateContent = async directory => {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) await checkNoPrivateContent(file);
      else if (/\.(html|xml)$/.test(entry.name)) {
        const html = await fs.readFile(file, 'utf8');
        for (const sentinel of ['Private review sentinel', 'Private multi review sentinel', 'Unmarked review sentinel', 'Private translation sentinel', 'PRIVATE_REVIEW_BODY_SENTINEL', 'PRIVATE_MULTI_BODY_SENTINEL', 'UNMARKED_REVIEW_BODY_SENTINEL', 'PRIVATE_TRANSLATION_BODY_SENTINEL']) assert.ok(!html.includes(sentinel), `${entry.name} must not contain private source content.`);
      }
    }
  };
  await checkNoPrivateContent(path.join(temp, 'dist'));
  await fixture('vision-1', { topic: '컴퓨터 비전', visibility: 'private' });
  await fixture('vision-2', { topic: '컴퓨터 비전', visibility: 'private' });
  await fixture('vision.en', { lang: 'en', translationKey: 'vision-1', topic: '컴퓨터 비전', visibility: 'private' });
  result = build();
  assert.equal(result.status, 0, result.stderr);
  assert.ok(!(await read('reviews/index.html')).includes(`data-topic="${visionId}"`), 'The last public review becoming private removes its topic choice.');
  for (const prefix of ['', 'en/']) await assert.rejects(fs.access(path.join(temp, 'dist', `${prefix}reviews/vision-1/index.html`)), 'Republishing removes previously public article files.');
  assert.ok(!(await read('feed.xml')).includes('/reviews/vision-1/') && !(await read('sitemap.xml')).includes('/reviews/vision-1/'));
  const noPresets = { ...config, topics: [] };
  await fs.writeFile(path.join(temp, 'site.json'), JSON.stringify(noPresets));
  result = build();
  assert.equal(result.status, 0, 'Posts do not require a manually registered topic list.');
  assert.ok((await read('reviews/index.html')).includes('data-topic="alpha"'));
  await fs.writeFile(path.join(temp, 'site.json'), JSON.stringify(config));
  await fixture('invalid-topic', { topic: '' });
  result = build();
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes('topic must be a non-empty string'));
  await fs.rm(path.join(temp, 'content/reviews/invalid-topic.md'));
  await fixture('invalid-visibility', { visibility: 'PUBLIC' });
  result = build();
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes('visibility must be public or private'));
  await fs.rm(path.join(temp, 'content/reviews/invalid-visibility.md'));

  const seedMultipleTopics = async () => {
    await fixture('multi-a.ko', { lang: 'ko', translationKey: 'multi-a', date: '2020-02-01', title: '여러 주제로 읽는 논문', topics: ['머신러닝', ' 컴퓨터   비전 ', '컴퓨터 비전', '방법론'] });
    await fixture('multi-a.en', { lang: 'en', translationKey: 'multi-a', date: '2020-02-01', title: 'One paper, multiple topics', topics: ['방법론', '컴퓨터 비전', '머신러닝'], topicNames: { '머신러닝': 'Machine Learning', '컴퓨터 비전': 'Computer Vision', '방법론': 'Methodology' } });
    await fixture('multi-b', { topic: '컴퓨터 비전', title: '비전만 다루는 논문' });
    await fixture('multi-c', { topics: ['컴퓨터 비전', '머신러닝'], title: '두 주제의 후속 논문' });
  };
  const baseTotal = ((await read('feed.xml')).match(/<item>/g) || []).length;
  await seedMultipleTopics();
  result = build();
  assert.equal(result.status, 0, result.stderr);
  const multiIndex = await read('reviews/index.html');
  const multiEnglish = await read('en/reviews/index.html');
  const machineId = multiIndex.match(/data-topic="([^"]+)">머신러닝<\/span>/)?.[1];
  const methodId = multiIndex.match(/data-topic="([^"]+)">방법론<\/span>/)?.[1];
  assert.ok(machineId && methodId);
  assert.equal(records(multiIndex).filter(review => review.topics.includes(machineId)).length, 2);
  assert.equal(records(multiIndex).filter(review => review.topics.includes(visionId)).length, 3);
  assert.equal(records(multiIndex).filter(review => review.topics.includes(methodId)).length, 1);
  const multiRecord = records(multiIndex).find(review => review.slug === 'multi-a');
  assert.ok([machineId, visionId, methodId].every(id => multiRecord.topics.includes(id)), 'A post is searchable through every assigned topic.');
  assert.equal((multiIndex.match(/data-review-slug="multi-a"/g) || []).length, 1, 'A multi-topic post has exactly one card.');
  const visibleCards = [...multiIndex.matchAll(/<article class="review-card"([^>]*)>/g)].filter(match => !/\bhidden\b/.test(match[1]));
  assert.equal(visibleCards.length, 11, 'All shows each of the 11 unique papers once, even when topics overlap.');
  assert.ok(!multiIndex.includes('role="tab"') && !multiIndex.includes('data-topic="all"'), 'Topic tabs and the All tab are removed.');
  const choices = [...multiIndex.matchAll(/type="checkbox" name="topic" value="([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(choices.sort(), [...new Set(records(multiIndex).flatMap(review => review.topics))].sort(), 'Checkbox choices list all and only the public topics once.');
  assert.ok(multiEnglish.includes(`data-topic="${machineId}">Machine Learning</span>`));
  assert.ok(multiEnglish.includes(`data-topic="${visionId}">Computer Vision</span>`));
  assert.ok(multiEnglish.includes(`data-topic="${methodId}">Methodology</span>`));
  for (const html of [multiIndex, await read('index.html')]) {
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
    assert.equal(new Set(ids).size, ids.length, 'Every card has a unique accessible title id.');
  }
  const multiArticle = await read('reviews/multi-a/index.html');
  assert.ok(multiArticle.includes('머신러닝 · 컴퓨터 비전 · 방법론'));
  const related = multiArticle.split('class="related-reviews"')[1];
  assert.equal((related.match(/href="\/reviews\/multi-c\/"/g) || []).length, 1, 'Related reviews sharing more than one topic appear once.');
  assert.ok(related.includes('/reviews/multi-b/'), 'Related reviews can share a secondary topic.');
  assert.ok((await read('en/reviews/multi-a/index.html')).includes(`href="/en/reviews/?topic=${machineId}#review-panel"`), 'The primary topic stays consistent across languages even if the source lists are reordered.');
  assert.equal(((await read('feed.xml')).match(/<item>/g) || []).length, baseTotal + 3);
  assert.equal(((await read('sitemap.xml')).match(/<loc>https:\/\/[^<]+\/reviews\/multi-a\/<\/loc>/g) || []).length, 2, 'Each published language has one sitemap URL regardless of topic count.');
  await checkNoPrivateContent(path.join(temp, 'dist'));
  for (const fields of [
    { topics: [] }, { topics: 'machine-learning' }, { topics: ['valid', ''] }, { topics: ['valid', 7] },
    { topics: ['valid'], topic: 'legacy' }, { topics: ['valid'], topicNames: { unknown: 'Unknown' } },
    { topics: ['valid'], topicNames: { valid: '' } }, { topics: ['valid', 'other'], topicName: 'Ambiguous' }
  ]) {
    await fixture('invalid-multi', fields);
    result = build();
    assert.notEqual(result.status, 0, `Invalid topic metadata should fail: ${JSON.stringify(fields)}`);
    await fs.rm(path.join(temp, 'content/reviews/invalid-multi.md'));
  }
  await fixture('multi-a.en', { lang: 'en', translationKey: 'multi-a', topics: ['머신러닝'] });
  result = build();
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes('must use the same topics'));
  for (const file of ['multi-a.ko', 'multi-a.en', 'multi-b', 'multi-c']) await fixture(file, { visibility: 'private' });
  result = build();
  assert.equal(result.status, 0, result.stderr);
  assert.ok(!(await read('reviews/index.html')).includes(`data-topic="${machineId}"`) && !(await read('reviews/index.html')).includes(`data-topic="${methodId}"`), 'Topics shared only by withdrawn posts disappear.');
  for (const prefix of ['', 'en/']) await assert.rejects(fs.access(path.join(temp, 'dist', `${prefix}reviews/multi-a/index.html`)));
  assert.ok(!(await read('feed.xml')).includes('/reviews/multi-a/') && !(await read('sitemap.xml')).includes('/reviews/multi-a/'));
  await fixture('timed-a', { date: '2020-03-01', publishedAt: '2020-03-01T09:00:00+09:00', updatedAt: '2020-03-03T18:05:42+09:00' }, '## Time-indexed content\n\nBODY_SEARCH_SENTINEL </script><script>UNTRUSTED_LITERAL</script>');
  await fixture('timed-b', { date: '2020-03-02', publishedAt: '2020-03-02T10:00:00+09:00', updatedAt: '2020-03-03T18:04:59+09:00' });
  await fixture('timed-a.en', { date: '2020-03-01', lang: 'en', translationKey: 'timed-a', publishedAt: '2020-03-01T09:00:00+09:00', updatedAt: '2020-03-04T12:00:00Z' });
  await fixture('timed-private.en', { date: '2020-03-01', lang: 'en', translationKey: 'timed-a', visibility: 'private', updatedAt: '2099-01-01T00:00:00Z' });
  await fs.writeFile(path.join(temp, 'site.json'), JSON.stringify({ ...config, homeReviewLimit: 2 }));
  result = build();
  assert.equal(result.status, 0, result.stderr);
  const timedHome = await read('index.html');
  const timedIndex = await read('reviews/index.html');
  const timedRecords = records(timedIndex);
  assert.equal((timedHome.match(/class="review-card"/g) || []).length, 2, 'The configured home limit applies across topics.');
  assert.ok(timedHome.indexOf('/reviews/timed-a/') < timedHome.indexOf('/reviews/timed-b/'), 'Updates outrank a newer initial publication.');
  assert.equal(timedRecords[0].slug, 'timed-a');
  assert.equal(timedRecords[0].updatedAt, '2020-03-04T12:00:00.000Z', 'The latest public translation update is shared across languages.');
  assert.equal(records(await read('en/reviews/index.html'))[0].updatedAt, timedRecords[0].updatedAt);
  assert.ok(timedRecords[0].text.includes('BODY_SEARCH_SENTINEL'), 'The public body is available for word search.');
  assert.ok(!timedIndex.includes('</script><script>UNTRUSTED_LITERAL'), 'Search data cannot close its JSON script element.');
  assert.ok((await read('feed.xml')).includes('Sun, 01 Mar 2020 00:00:00 GMT'), 'RSS retains initial publication time rather than the update time.');
  assert.ok((await read('reviews/timed-a/index.html')).includes('article:modified_time" content="2020-03-04T12:00:00.000Z'));
  await checkNoPrivateContent(path.join(temp, 'dist'));
  await fixture('invalid-time', { publishedAt: '2020-01-01T12:00:00', updatedAt: 'bad' });
  result = build();
  assert.notEqual(result.status, 0, 'Datetime metadata must include a timezone.');
  await fs.rm(path.join(temp, 'content/reviews/invalid-time.md'));
  for (const file of ['timed-a', 'timed-b', 'timed-a.en', 'timed-private.en']) await fs.rm(path.join(temp, `content/reviews/${file}.md`));
  await fs.writeFile(path.join(temp, 'site.json'), JSON.stringify(config));
  await fs.rename(path.join(temp, 'content/reviews'), path.join(temp, 'content/reviews-saved'));
  await fs.mkdir(path.join(temp, 'content/reviews'));
  result = build();
  assert.equal(result.status, 0, result.stderr);
  assert.ok((await read('index.html')).includes('아직 공개된 리뷰가 없습니다.'));
  assert.ok((await read('en/reviews/index.html')).includes('No public reviews yet.'));
  assert.equal(((await read('reviews/index.html')).match(/type="checkbox"/g) || []).length, 0, 'An empty collection has no topic choices.');
  assert.ok(!(await read('reviews/index.html')).includes('data-topic-group='));
  await fs.rm(path.join(temp, 'content/reviews'), { recursive: true });
  await fs.rename(path.join(temp, 'content/reviews-saved'), path.join(temp, 'content/reviews'));
  console.log('Passed: latest reviews, unique multi-topic cards, search index, topic translations, private exclusion, timestamps, bilingual pairing, authoring, RSS, sitemap and project paths.');
  if (keepPreview) {
    await seedMultipleTopics();
    result = build();
    assert.equal(result.status, 0, result.stderr);
    previewReady = true;
    console.log(`PREVIEW_DIRECTORY=${temp}`);
  }
} finally {
  if (!previewReady) await fs.rm(temp, { recursive: true, force: true });
}
