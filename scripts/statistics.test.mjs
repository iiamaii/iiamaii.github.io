import test from 'node:test';
import { spawnSync } from 'node:child_process';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { validateAnalysis, loadAnalyses } from './statistics.mjs';
import { renderStatistics } from './render-statistics.mjs';
const bi = text => ({ ko: text, en: text });
const fixture = () => ({ schemaVersion: 1, id: 'test-analysis', visibility: 'public', updatedAt: '2020-01-01T00:00:00Z', title: bi('<script>bad()</script>'), summary: bi('Summary'), notes: bi('Conditions'), paperSlugs: ['paper-a'], sources: [{ url: 'https://example.org/paper', label: bi('Source'), location: bi('Table 1') }], charts: [{ id: 'values', type: 'bar', title: bi('Results'), description: bi('Conditions'), unit: bi('units'), direction: 'none', values: [{ label: bi('Negative'), value: -2 }, { label: bi('Zero'), value: 0 }, { label: bi('Positive'), value: 2 }] }] });
const keys = new Set(['paper-a']);
const reviews = [{ slug: 'paper-a', title: 'A', topics: [{ id: 'topic-a' }] }];
const render = (record, lang = 'ko') => renderStatistics({ analyses: [record], reviews, lang, href: route => `/repository${lang === 'en' ? '/en' : ''}${route}` });
test('finite values including zero and negatives retain a shared zero baseline; text is escaped', () => {
  const record = validateAnalysis(fixture(), keys), html = render(record);
  assert.match(html, /&lt;script&gt;bad\(\)&lt;\/script&gt;/); assert.ok(!html.includes('<script>bad'));
  assert.match(html, /left:50%/); assert.match(html, /left:0%;width:50%/); assert.match(html, /left:50%;width:0%/);
  assert.match(html, /\/repository\/reviews\/paper-a\//);
  assert.match(render(record, 'en'), /\/repository\/en\/reviews\/paper-a\//);
});
test('missing, non-finite and string numbers are rejected rather than silently becoming zero', () => {
  for (const value of [null, undefined, NaN, Infinity, '1']) { const raw = fixture(); raw.charts[0].values[0].value = value; assert.throws(() => validateAnalysis(raw, keys), /finite number/); }
});
test('sources, public references, schema and unique chart ids are validated', () => {
  const raw = fixture(); raw.sources[0].url = 'javascript:alert(1)'; assert.throws(() => validateAnalysis(raw, keys), /HTTP\(S\)/);
  assert.throws(() => validateAnalysis(fixture(), new Set()), /not a public review/);
  const version = fixture(); version.schemaVersion = 2; assert.throws(() => validateAnalysis(version, keys), /schemaVersion/);
  const duplicate = fixture(); duplicate.charts.push(structuredClone(duplicate.charts[0])); assert.throws(() => validateAnalysis(duplicate, keys), /duplicate chart/);
});
const network = () => ({ id: 'links', type: 'network', title: bi('Links'), description: bi('Two items'), relationship: bi('Shared topics'), nodes: [{ id: 'a', label: bi('A'), paperSlug: 'paper-a' }, { id: 'b', label: bi('B') }, { id: 'isolated', label: bi('Isolated') }], edges: [{ source: 'a', target: 'b', weight: 2 }] });
test('networks retain isolated nodes and reject dangling, self, duplicate and invalid weighted edges', () => {
  const raw = fixture(); raw.charts = [network()];
  const html = render(validateAnalysis(raw, keys)); assert.match(html, /Isolated/); assert.match(html, /<svg/); assert.match(html, /<td>2<\/td>/);
  for (const edges of [[{ source: 'a', target: 'missing', weight: 1 }], [{ source: 'a', target: 'a', weight: 1 }], [{ source: 'a', target: 'b', weight: 0 }], [{ source: 'a', target: 'b', weight: 1 }, { source: 'b', target: 'a', weight: 2 }]]) { const invalid = structuredClone(raw); invalid.charts[0].edges = edges; assert.throws(() => validateAnalysis(invalid, keys)); }
});
test('missing directory is empty; private and future analyses never enter public output', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'blog-statistics-'));
  try {
    assert.deepEqual(await loadAnalyses(path.join(directory, 'missing'), keys), []);
    await fs.writeFile(path.join(directory, 'private.json'), JSON.stringify({ visibility: 'private', title: 'SECRET' }));
    const future = fixture(); future.id = 'future'; future.updatedAt = '2099-01-01T00:00:00Z';
    await fs.writeFile(path.join(directory, 'future.json'), JSON.stringify(future));
    await fs.writeFile(path.join(directory, 'public.json'), JSON.stringify(fixture()));
    const records = await loadAnalyses(directory, keys, Date.parse('2021-01-01'));
    assert.equal(records.length, 1); assert.equal(records[0].id, 'test-analysis'); assert.ok(!render(records[0]).includes('SECRET'));
    await fs.writeFile(path.join(directory, 'duplicate.json'), JSON.stringify(fixture()));
    await assert.rejects(loadAnalyses(directory, keys), /duplicate analysis id/);
  } finally { await fs.rm(directory, { recursive: true, force: true }); }
});

test('analysis generator starts private with unknown values and never overwrites an existing result', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'blog-statistics-authoring-'));
  try {
    await fs.mkdir(path.join(directory, 'skills/publish-research-blog/assets'), { recursive: true });
    await fs.copyFile('skills/publish-research-blog/assets/statistics-analysis.json', path.join(directory, 'skills/publish-research-blog/assets/statistics-analysis.json'));
    const script = path.resolve('scripts/new-statistics.mjs');
    const run = () => spawnSync(process.execPath, [script, 'test-draft'], { cwd: directory, encoding: 'utf8' });
    const first = run(); assert.equal(first.status, 0, first.stderr);
    const file = path.join(directory, 'content/statistics/test-draft.json');
    const saved = await fs.readFile(file, 'utf8'), raw = JSON.parse(saved);
    assert.equal(raw.visibility, 'private'); assert.equal(raw.id, 'test-draft'); assert.equal(raw.charts[0].values[0].value, null);
    assert.notEqual(run().status, 0); assert.equal(await fs.readFile(file, 'utf8'), saved);
  } finally { await fs.rm(directory, { recursive: true, force: true }); }
});
