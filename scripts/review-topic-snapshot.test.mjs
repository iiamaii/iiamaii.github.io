import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { paperIdentity, createReviewTopicSnapshot, sameSnapshot } from './review-topic-snapshot.mjs';

const topic = key => ({ key, name: key });
const review = (title, paperUrl, keys) => ({ title, paperTitle: title, paperUrl, topics: keys.map(topic) });
const args = pairs => ({ pairs, topics: [], url: 'https://example.org', updatedAt: '2020-01-01T00:00:00Z' });
test('paper identities merge source formats and versions, not unrelated papers or titles', () => {
  assert.equal(paperIdentity('https://arxiv.org/abs/2511.08544v3'), paperIdentity('https://arxiv.org/pdf/2511.08544v1.pdf'));
  assert.equal(paperIdentity('https://arxiv.org/html/2511.08544v2'), paperIdentity('https://arxiv.org/abs/2511.08544'));
  assert.equal(paperIdentity('https://doi.org/10.1000/ABC'), paperIdentity('https://dx.doi.org/10.1000/abc'));
  assert.notEqual(paperIdentity('https://example.org/paper-a'), paperIdentity('https://example.org/paper-b'));
  assert.throws(() => paperIdentity(''), /verified paperUrl/);
});
test('series and translations count once, primary topics use the introduction, edges use public topic unions', () => {
  const pairs = new Map([
    ['advanced', { ko: review('[2/2] A', 'https://arxiv.org/abs/2511.08544v3', ['extra', 'shared']) }],
    ['intro', { ko: review('[1/2] A', 'https://arxiv.org/abs/2511.08544v1', ['main']), en: review('[1/2] A English', 'https://arxiv.org/pdf/2511.08544v2', ['main']) }],
    ['other', { en: review('B', 'https://example.org/b', ['shared']) }],
    ['isolated', { ko: review('C', 'https://example.org/c', ['unrelated']) }]
  ]);
  const record = createReviewTopicSnapshot(args(pairs));
  assert.equal(record.paperSlugs.length, 4);
  assert.equal(record.charts[1].nodes.length, 3);
  assert.equal(record.charts[0].values.reduce((sum, row) => sum + row.value, 0), 3);
  assert.equal(record.charts[0].values.some(row => row.label.ko === 'extra'), false);
  assert.ok(record.charts[1].nodes.some(node => node.paperSlug === 'intro'));
  assert.equal(record.charts[1].edges.length, 1);
  assert.equal(record.charts[1].edges[0].weight, 1);
  assert.ok(record.sources.some(source => source.url === 'https://example.org/en/reviews/other/'));
  const reversed = new Map([...pairs].reverse());
  assert.ok(sameSnapshot(record, createReviewTopicSnapshot(args(reversed))));
  assert.ok(sameSnapshot({ b: 2, a: 1 }, { a: 1, b: 2 }));
  pairs.get('intro').en.paperUrl = 'https://example.org/different-paper';
  assert.throws(() => createReviewTopicSnapshot(args(pairs)), /same paperUrl/);
});
test('empty sets have no charts and oversized networks fail without dropping papers', () => {
  assert.equal(createReviewTopicSnapshot(args(new Map())), null);
  const pairs = new Map(Array.from({ length: 41 }, (_, n) => [`paper-${n}`, { ko: review(`Paper ${n}`, `https://example.org/${n}`, ['a']) }]));
  assert.throws(() => createReviewTopicSnapshot(args(pairs)), /exceeds 40/);
});
test('build rejects stale counts; explicit updates preserve dates when unchanged and exclude withdrawn, draft and future reviews', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'blog-statistics-refresh-'));
  const script = path.resolve('scripts/build.mjs');
  try {
    for (const name of ['public', 'site.json']) await fs.cp(name, path.join(directory, name), { recursive: true });
    await fs.symlink(path.resolve('node_modules'), path.join(directory, 'node_modules'), 'dir');
    await fs.mkdir(path.join(directory, 'content/reviews'), { recursive: true });
    const write = async (slug, fields = {}) => {
      const metadata = { title: slug, description: 'Fixture', date: '2020-01-01', visibility: 'public', lang: 'ko', translationKey: slug, topics: ['shared'], paperTitle: slug, paperPublishedDate: '2019-01-01', authors: ['Author'], paperUrl: `https://example.org/${slug}`, ...fields };
      await fs.writeFile(path.join(directory, `content/reviews/${slug}.ko.md`), `---\n${Object.entries(metadata).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n')}\n---\n\nBody.\n`);
    };
    const run = (...flags) => spawnSync(process.execPath, [script, ...flags], { cwd: directory, encoding: 'utf8', env: { ...process.env, SITE_BASE_PATH: '' } });
    await write('public-a'); await write('private', { visibility: 'private' }); await write('draft', { draft: true }); await write('future', { date: '2099-01-01' });
    let result = run('--update-statistics'); assert.equal(result.status, 0, result.stderr);
    const file = path.join(directory, 'content/statistics/review-topic-snapshot.json');
    let record = JSON.parse(await fs.readFile(file, 'utf8'));
    assert.deepEqual(record.paperSlugs, ['public-a']);
    record.updatedAt = '2020-02-01T00:00:00Z';
    const unchanged = JSON.stringify(record, null, 2) + '\n';
    await fs.writeFile(file, unchanged);
    result = run('--update-statistics'); assert.equal(result.status, 0, result.stderr);
    assert.equal(await fs.readFile(file, 'utf8'), unchanged);
    await write('public-b'); result = run();
    assert.notEqual(result.status, 0); assert.match(result.stderr, /stats:update/);
    assert.equal(await fs.readFile(file, 'utf8'), unchanged, 'Normal builds never rewrite the source snapshot.');
    result = run('--update-statistics'); assert.equal(result.status, 0, result.stderr);
    record = JSON.parse(await fs.readFile(file, 'utf8')); assert.equal(record.charts[1].nodes.length, 2);
    await write('public-a', { visibility: 'private' });
    result = run('--update-statistics'); assert.equal(result.status, 0, result.stderr);
    assert.deepEqual(JSON.parse(await fs.readFile(file, 'utf8')).paperSlugs, ['public-b']);
    record = JSON.parse(await fs.readFile(file, 'utf8')); record.visibility = 'private';
    await fs.writeFile(file, JSON.stringify(record));
    result = run('--update-statistics'); assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(await fs.readFile(file, 'utf8')).visibility, 'private');
    record.visibility = 'public'; await fs.writeFile(file, JSON.stringify(record));
    await write('public-b', { visibility: 'private' });
    result = run('--update-statistics'); assert.equal(result.status, 0, result.stderr);
    await assert.rejects(fs.access(file));
  } finally { await fs.rm(directory, { recursive: true, force: true }); }
});
