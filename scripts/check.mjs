import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';

const root = process.cwd();
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'blog-review-check-'));
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
  await fs.writeFile(path.join(temp, 'site.json'), JSON.stringify(config));
  const fixture = async (slug, fields) => {
    const data = { title: `Fixture ${slug} & "question"`, description: 'A concise test description.', date: '2020-01-01', topic: 'alpha', ...fields };
    const metadata = Object.entries(data).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n');
    await fs.writeFile(path.join(temp, 'content/reviews', `${slug}.md`), `---\n${metadata}\n---\n\n## Repeated question\n\nA paragraph.\n\n## Repeated question\n\nAnother paragraph.\n`);
  };
  for (let i = 1; i <= 5; i++) await fixture(`alpha-${i}`, { date: `2020-01-0${i}` });
  await fixture('beta-1', { topic: 'beta', paperTitle: 'Original & paper', paperUrl: 'https://example.org/paper?x=1&y=2', thumbnail: '/assets/topics/topic-03.svg' });
  await fixture('beta-2', { topic: 'beta' });
  await fixture('draft', { draft: true });
  await fixture('future', { date: '9999-01-01' });
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
  assert.ok(index.includes('data-topic-group="empty"') && index.includes('아직 등록된 논문이 없습니다.'), 'Topics without reviews have a clear empty state.');
  assert.ok(index.includes('Fixture beta-1 &amp; &quot;question&quot;'), 'Metadata is escaped in cards.');
  assert.ok(article.includes('https://example.org/paper?x=1&amp;y=2'));
  assert.ok(article.includes('/assets/topics/topic-03.svg'), 'A review can override its topic thumbnail.');
  assert.ok((await read('reviews/alpha-1/index.html')).includes('/assets/topics/topic-01.svg'), 'Missing review thumbnails use the topic image.');
  assert.ok(article.includes('id="repeated-question-2"'), 'Repeated headings have distinct anchors.');
  assert.ok(profile.includes('A profile &amp; an idea.') && profile.includes('<p>Second paragraph.</p>'));
  assert.ok(home.includes('<p>Second paragraph.</p>'), 'The home and profile share the same philosophy.');
  assert.ok(sitemap.includes('/profile/') && sitemap.includes('/reviews/beta-1/'));
  assert.equal((rss.match(/<item>/g) || []).length, 7);
  for (const excluded of ['draft', 'future']) {
    await assert.rejects(fs.access(path.join(temp, 'dist/reviews', excluded, 'index.html')));
    assert.ok(!sitemap.includes(`/reviews/${excluded}/`));
  }
  await assert.rejects(fs.access(path.join(temp, 'dist/posts')), 'The old sample post directory is absent.');
  assert.ok((await read('about/index.html')).includes('url=/profile/'));
  result = build('/project');
  assert.equal(result.status, 0, result.stderr);
  assert.ok((await read('index.html')).includes('href="/project/profile/"'));
  assert.ok((await read('reviews/index.html')).includes('src="/project/assets/reviews.js"'));
  assert.ok((await read('reviews/beta-1/index.html')).includes('href="/project/reviews/?topic=beta#review-panel"'));
  await fixture('invalid-topic', { topic: 'unknown' });
  result = build();
  assert.notEqual(result.status, 0);
  assert.ok(result.stderr.includes('topic must match an id'));
  console.log('Passed: review publishing, draft/future exclusion, topic grouping, thumbnails, ordering, shared profile, metadata escaping, anchors, RSS, sitemap and project paths.');
} finally {
  await fs.rm(temp, { recursive: true, force: true });
}
