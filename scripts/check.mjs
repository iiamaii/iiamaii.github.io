import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';

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
    const data = { title: `Fixture ${slug} & "question"`, description: 'A concise test description.', date: '2020-01-01', topic: 'alpha', visibility: 'public', ...fields };
    const metadata = Object.entries(data).filter(([, value]) => value !== undefined).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n');
    await fs.writeFile(path.join(temp, 'content/reviews', `${slug}.md`), `---\n${metadata}\n---\n\n${body}\n`);
  };
  for (let i = 1; i <= 5; i++) await fixture(`alpha-${i}`, { date: `2020-01-0${i}` });
  await fixture('beta-1', { topic: 'beta', paperTitle: 'Original & paper', paperUrl: 'https://example.org/paper?x=1&y=2', thumbnail: '/assets/topics/topic-03.svg' });
  await fixture('beta-2', { topic: 'beta' });
  await fixture('draft', { draft: true, topic: 'draft-only-topic' });
  await fixture('future', { date: '9999-01-01', topic: 'future-only-topic' });
  await fixture('private', { visibility: 'private', topic: '비공개 연구', title: 'Private review sentinel' }, 'PRIVATE_REVIEW_BODY_SENTINEL');
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
  assert.equal((home.match(/class="review-card"/g) || []).length, 6, 'Home shows at most four reviews per topic.');
  assert.equal((index.match(/class="review-card"/g) || []).length, 7, 'The review index shows all published reviews.');
  assert.ok(!home.includes('/reviews/alpha-1/'), 'The oldest fifth review stays off the home page.');
  assert.ok(index.includes('/reviews/alpha-1/'));
  assert.ok(home.indexOf('/reviews/alpha-5/') < home.indexOf('/reviews/alpha-4/'), 'Reviews sort newest first.');
  assert.ok(index.includes('data-topic="alpha"') && index.includes('data-review-total="5"'));
  assert.ok(index.includes('data-topic="beta"') && index.includes('data-review-total="2"'));
  assert.ok(!index.includes('data-topic-group="empty"') && !index.includes('data-topic="empty"'), 'Unused topic presets do not create groups or tabs.');
  for (const name of ['비공개 연구', '미지정 연구', 'draft-only-topic', 'future-only-topic']) assert.ok(!index.includes(name), 'Only public, ready reviews contribute topics.');
  assert.ok(index.includes('Fixture beta-1 &amp; &quot;question&quot;'), 'Metadata is escaped in cards.');
  assert.ok(article.includes('https://example.org/paper?x=1&amp;y=2'));
  assert.ok(article.includes('/assets/topics/topic-03.svg'), 'A review can override its topic thumbnail.');
  assert.ok((await read('reviews/alpha-1/index.html')).includes('/assets/topics/topic-01.svg'), 'Missing review thumbnails use the topic image.');
  assert.ok(article.includes('id="repeated-question-2"'), 'Repeated headings have distinct anchors.');
  assert.ok(profile.includes('A profile &amp; an idea.') && profile.includes('<p>Second paragraph.</p>'));
  assert.ok(home.includes('<p>Second paragraph.</p>'), 'The home and profile share the same philosophy.');
  assert.ok(sitemap.includes('/profile/') && sitemap.includes('/reviews/beta-1/'));
  assert.equal((rss.match(/<item>/g) || []).length, 7);
  for (const excluded of ['draft', 'future', 'private', 'unmarked']) {
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
    const source = await fs.readFile(path.join(temp, 'content/reviews', file), 'utf8');
    assert.ok(source.includes('visibility: "private"') && source.includes('topic: "새로운 주제"'), 'New language versions start private and accept unregistered topics.');
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
  const visionId = dynamicIndex.match(/data-topic="([^"]+)">컴퓨터 비전<span/)?.[1];
  assert.ok(visionId, 'A topic name in a public post automatically creates a tab without a preset.');
  assert.ok(dynamicIndex.includes(`data-topic-group="${visionId}" data-review-total="2"`), 'Normalized topic names group reviews together.');
  assert.ok((await read('en/reviews/index.html')).includes(`data-topic="${visionId}">Computer Vision<span`), 'Translated topic labels share the same filter id.');
  assert.ok((await read('reviews/vision-1/index.html')).includes('/assets/topics/topic-01.svg'), 'New topics have a default thumbnail.');
  const checkNoPrivateContent = async directory => {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) await checkNoPrivateContent(file);
      else if (/\.(html|xml)$/.test(entry.name)) {
        const html = await fs.readFile(file, 'utf8');
        for (const sentinel of ['Private review sentinel', 'Unmarked review sentinel', 'Private translation sentinel', 'PRIVATE_REVIEW_BODY_SENTINEL', 'UNMARKED_REVIEW_BODY_SENTINEL', 'PRIVATE_TRANSLATION_BODY_SENTINEL']) assert.ok(!html.includes(sentinel), `${entry.name} must not contain private source content.`);
      }
    }
  };
  await checkNoPrivateContent(path.join(temp, 'dist'));
  await fixture('vision-1', { topic: '컴퓨터 비전', visibility: 'private' });
  await fixture('vision-2', { topic: '컴퓨터 비전', visibility: 'private' });
  await fixture('vision.en', { lang: 'en', translationKey: 'vision-1', topic: '컴퓨터 비전', visibility: 'private' });
  result = build();
  assert.equal(result.status, 0, result.stderr);
  assert.ok(!(await read('reviews/index.html')).includes(`data-topic="${visionId}"`), 'The last public review becoming private removes its topic tab.');
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
  await fs.rename(path.join(temp, 'content/reviews'), path.join(temp, 'content/reviews-saved'));
  await fs.mkdir(path.join(temp, 'content/reviews'));
  result = build();
  assert.equal(result.status, 0, result.stderr);
  assert.ok((await read('index.html')).includes('아직 공개된 리뷰가 없습니다.'));
  assert.ok((await read('en/reviews/index.html')).includes('No public reviews yet.'));
  assert.equal(((await read('reviews/index.html')).match(/role="tab"/g) || []).length, 1, 'An empty collection has only the All tab.');
  assert.ok(!(await read('reviews/index.html')).includes('data-topic-group='));
  await fs.rm(path.join(temp, 'content/reviews'), { recursive: true });
  await fs.rename(path.join(temp, 'content/reviews-saved'), path.join(temp, 'content/reviews'));
  console.log('Passed: automatic public topics, private and unmarked exclusion, removal on visibility change, bilingual pairing, original-language fallback, authoring, metadata, RSS, sitemap and project paths.');
  if (keepPreview) {
    await fixture('vision-1', { topic: '컴퓨터 비전', topicName: '컴퓨터 비전', title: 'A public vision review' });
    await fixture('vision-2', { topic: '컴퓨터 비전', title: 'Another public vision review' });
    await fixture('vision.en', { lang: 'en', translationKey: 'vision-1', topic: '컴퓨터 비전', topicName: 'Computer Vision', title: 'An English vision review' });
    result = build();
    assert.equal(result.status, 0, result.stderr);
    previewReady = true;
    console.log(`PREVIEW_DIRECTORY=${temp}`);
  }
} finally {
  if (!previewReady) await fs.rm(temp, { recursive: true, force: true });
}
