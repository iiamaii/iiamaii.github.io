import fs from 'node:fs/promises';
import path from 'node:path';

const slug = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
function fail(field, message) { throw new Error(`${field}: ${message}`); }
function text(value, field) {
  if (typeof value !== 'string' || !value.trim()) fail(field, 'non-empty text required');
  return value.trim();
}
function bilingual(value, field) {
  return Object.fromEntries(['ko', 'en'].map(lang => [lang, text(value?.[lang], `${field}.${lang}`)]));
}
function identifier(value, field) {
  if (typeof value !== 'string' || !slug.test(value)) fail(field, 'use a lowercase hyphenated id');
  return value;
}
function number(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value)) fail(field, 'finite number required; missing is not zero');
  return value;
}
export function validateAnalysis(raw, publicSlugs = new Set()) {
  if (raw.schemaVersion !== 1) fail('schemaVersion', 'must be 1');
  const id = identifier(raw.id, 'id');
  if (!['private', 'public'].includes(raw.visibility ?? 'private')) fail('visibility', 'public or private required');
  if (typeof raw.updatedAt !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})$/.test(raw.updatedAt) || !Number.isFinite(Date.parse(raw.updatedAt))) fail('updatedAt', 'ISO date/time with seconds and timezone required');
  const calendar = raw.updatedAt.slice(0, 10);
  if (new Date(`${calendar}T00:00:00Z`).toISOString().slice(0, 10) !== calendar) fail('updatedAt', 'invalid calendar date');
  const paperSlugs = raw.paperSlugs ?? [];
  if (!Array.isArray(paperSlugs) || new Set(paperSlugs).size !== paperSlugs.length) fail('paperSlugs', 'unique list required');
  for (const key of paperSlugs) if (!publicSlugs.has(key)) fail('paperSlugs', `not a public review: ${key}`);
  if (!Array.isArray(raw.sources) || !raw.sources.length) fail('sources', 'at least one source required');
  const sources = raw.sources.map((source, index) => {
    let url;
    try { url = new URL(source.url); } catch { fail(`sources[${index}].url`, 'HTTP(S) URL required'); }
    if (!['https:', 'http:'].includes(url.protocol)) fail(`sources[${index}].url`, 'HTTP(S) URL required');
    return { url: url.href, label: bilingual(source.label, `sources[${index}].label`), location: bilingual(source.location, `sources[${index}].location`) };
  });
  if (!Array.isArray(raw.charts) || !raw.charts.length) fail('charts', 'at least one chart required');
  const chartIds = new Set();
  const charts = raw.charts.map((chart, index) => {
    const field = `charts[${index}]`;
    const chartId = identifier(chart.id, `${field}.id`);
    if (chartIds.has(chartId)) fail(field, 'duplicate chart id');
    chartIds.add(chartId);
    const common = { id: chartId, type: chart.type, title: bilingual(chart.title, `${field}.title`), description: bilingual(chart.description, `${field}.description`) };
    if (chart.type === 'bar') {
      if (!['higher', 'lower', 'none'].includes(chart.direction)) fail(`${field}.direction`, 'higher, lower or none required');
      if (!Array.isArray(chart.values) || !chart.values.length || chart.values.length > 100) fail(`${field}.values`, '1–100 rows required');
      return { ...common, unit: bilingual(chart.unit, `${field}.unit`), direction: chart.direction,
        values: chart.values.map((row, n) => ({ label: bilingual(row.label, `${field}.values[${n}].label`), value: number(row.value, `${field}.values[${n}].value`) })) };
    }
    if (chart.type === 'network') {
      if (!Array.isArray(chart.nodes) || !chart.nodes.length || chart.nodes.length > 40) fail(`${field}.nodes`, '1–40 nodes required');
      const ids = new Set();
      const nodes = chart.nodes.map(node => {
        const nodeId = identifier(node.id, `${field}.nodes.id`);
        if (ids.has(nodeId)) fail(field, 'duplicate node');
        ids.add(nodeId);
        if (node.paperSlug !== undefined && (!publicSlugs.has(node.paperSlug) || !paperSlugs.includes(node.paperSlug))) fail(field, 'node paperSlug must be a public review in paperSlugs');
        return { id: nodeId, label: bilingual(node.label, `${field}.nodes.label`), ...(node.paperSlug ? { paperSlug: node.paperSlug } : {}) };
      });
      if (!Array.isArray(chart.edges)) fail(`${field}.edges`, 'list required');
      const seen = new Set();
      const edges = chart.edges.map(edge => {
        if (!ids.has(edge.source) || !ids.has(edge.target) || edge.source === edge.target) fail(field, 'edge must connect two existing different nodes');
        const key = [edge.source, edge.target].sort().join(':');
        if (seen.has(key)) fail(field, 'duplicate undirected edge');
        seen.add(key);
        const weight = number(edge.weight, `${field}.edges.weight`);
        if (weight <= 0) fail(field, 'edge weight must be positive');
        return { source: edge.source, target: edge.target, weight };
      });
      return { ...common, relationship: bilingual(chart.relationship, `${field}.relationship`), nodes, edges };
    }
    fail(`${field}.type`, 'supported: bar, network');
  });
  return { schemaVersion: 1, id, visibility: raw.visibility ?? 'private', updatedAt: raw.updatedAt, title: bilingual(raw.title, 'title'), summary: bilingual(raw.summary, 'summary'), notes: bilingual(raw.notes, 'notes'), paperSlugs, sources, charts };
}

export async function loadAnalyses(directory, publicSlugs, now = Date.now()) {
  let files;
  try { files = await fs.readdir(directory); } catch (error) { if (error.code === 'ENOENT') return []; throw error; }
  const records = [], ids = new Set();
  for (const file of files.filter(file => file.endsWith('.json')).sort()) {
    try {
      const raw = JSON.parse(await fs.readFile(path.join(directory, file), 'utf8'));
      if (!['public', 'private'].includes(raw.visibility ?? 'private')) fail('visibility', 'public or private required');
      if (raw.visibility !== 'public') continue;
      const record = validateAnalysis(raw, publicSlugs);
      if (Date.parse(record.updatedAt) > now) continue;
      if (ids.has(record.id)) fail('id', 'duplicate analysis id');
      ids.add(record.id); records.push(record);
    } catch (error) { throw new Error(`${file}: ${error.message}`, { cause: error }); }
  }
  return records.sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt) || a.id.localeCompare(b.id));
}
