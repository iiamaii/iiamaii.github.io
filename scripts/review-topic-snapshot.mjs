import { createHash } from 'node:crypto';

// Identity ignores arXiv versions; unrelated URLs are never merged by title alone.
export function paperIdentity(value) {
  let url;
  try { url = new URL(value); } catch { throw new Error('Topic statistics require a verified paperUrl for every public review.'); }
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Topic statistics require an HTTP(S) paperUrl.');
  if (/(^|\.)arxiv\.org$/i.test(url.hostname)) {
    const id = url.pathname.replace(/^\/(?:abs|pdf|html)\//, '').replace(/\.pdf$/i, '').replace(/v\d+$/, '');
    if (/^(?:\d{4}\.\d{4,5}|[a-z-]+\/\d{7})$/i.test(id)) return `arxiv:${id.toLowerCase()}`;
  }
  if (/^(?:dx\.)?doi\.org$/i.test(url.hostname)) return `doi:${decodeURIComponent(url.pathname.slice(1)).toLowerCase()}`;
  url.hash = '';
  return url.href;
}

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])]));
  return value;
}
export const sameSnapshot = (a, b) => JSON.stringify(stable(a)) === JSON.stringify(stable(b));

export function createReviewTopicSnapshot({ pairs, topics, url, base = '', updatedAt, previous }) {
  if (!pairs.size) return null;
  const entries = [...pairs].map(([slug, pair]) => ({ slug, pair, source: pair.ko || pair.en })).sort((a, b) => a.slug.localeCompare(b.slug));
  const groups = new Map();
  for (const entry of entries) {
    const identity = paperIdentity(entry.source.paperUrl);
    if (Object.values(entry.pair).some(review => paperIdentity(review.paperUrl) !== identity)) throw new Error(`${entry.slug}: public translations must identify the same paperUrl for topic statistics.`);
    if (!groups.has(identity)) groups.set(identity, []);
    groups.get(identity).push(entry);
  }
  if (groups.size > 40) throw new Error('Topic network exceeds 40 unique papers. Split its documented scope before updating; do not silently truncate.');
  const oldNodes = previous?.charts?.find(chart => chart.id === 'shared-topics')?.nodes || [];
  const shortLabel = members => {
    const preserved = oldNodes.find(node => members.some(entry => entry.slug === node.paperSlug));
    if (preserved) return preserved.label;
    const title = members[0].source.paperTitle || members[0].source.title;
    const candidates = [title, ...members.map(entry => entry.source.title.replace(/^\[\d+\/\d+\]\s*/, ''))]
      .map(value => value.split(':')[0].trim()).filter(value => /^[A-Za-z][A-Za-z0-9 .-]{0,39}$/.test(value));
    const label = candidates.sort((a, b) => a.length - b.length || a.localeCompare(b))[0] || title;
    return { ko: label, en: label };
  };
  const papers = [...groups].sort(([a], [b]) => a.localeCompare(b)).map(([identity, members]) => {
    // Prefer the introduction; the Korean public version determines topic order.
    const rank = entry => Number(entry.source.title.match(/^\[(\d+)\/\d+\]/)?.[1] || Number.MAX_SAFE_INTEGER);
    members.sort((a, b) => rank(a) - rank(b) || a.slug.localeCompare(b.slug));
    return {
      id: `paper-${createHash('sha256').update(identity).digest('hex').slice(0, 16)}`,
      slug: members[0].slug, label: shortLabel(members),
      primary: members[0].source.topics[0].key,
      topics: new Set(members.flatMap(entry => entry.source.topics.map(topic => topic.key)))
    };
  });
  const counts = new Map();
  for (const paper of papers) counts.set(paper.primary, (counts.get(paper.primary) || 0) + 1);
  const edges = [];
  for (let i = 0; i < papers.length; i++) for (let j = i + 1; j < papers.length; j++) {
    const weight = [...papers[i].topics].filter(topic => papers[j].topics.has(topic)).length;
    if (weight) edges.push({ source: papers[i].id, target: papers[j].id, weight });
  }
  const topicLabel = key => {
    const topic = topics.find(topic => topic.key === key);
    return { ko: topic?.names?.ko || topic?.name || key, en: topic?.names?.en || topic?.name || key };
  };
  const count = papers.length, reviews = entries.length;
  return {
    schemaVersion: 1, id: 'review-topic-snapshot', visibility: 'public', updatedAt,
    title: { ko: '공개 리뷰의 논문·주제 연결', en: 'Papers and topic links across public reviews' },
    summary: { ko: `공개 리뷰 ${reviews}편을 고유 논문 ${count}개로 묶어 대표 주제 분포와 공통 주제를 집계합니다.`, en: `${reviews} public reviews grouped into ${count} unique papers, with primary-topic counts and shared-topic links.` },
    notes: {
      ko: '공개·발행 시각 조건을 통과한 리뷰만 포함합니다. 한영 번역은 같은 리뷰로, 같은 arXiv ID(버전 제외)·DOI·원문 URL의 연재는 같은 논문으로 묶습니다. 대표 리뷰는 [1/N] 등 가장 앞선 편이며 동순위는 키 순입니다. 막대는 대표 리뷰의 첫 주제별 고유 논문 수, 관계선 가중치는 각 논문의 공개 연재 전체 주제 합집합 사이의 교집합 크기입니다. 선이 없는 논문도 포함합니다. 이 집계는 블로그에 등록한 주제의 연결이며 인용·인과 관계나 연구 분야 전체의 분포를 뜻하지 않습니다.',
      en: 'Only public, already published reviews are included. Translations count as one review; series sharing an arXiv ID (ignoring versions), DOI, or source URL count as one paper. The earliest numbered part represents a paper, with slug order breaking ties. Bars count unique papers by the representative review’s first topic. Edge weights count the intersection of the unions of topics across each paper’s public parts. Isolated papers are included. These are assigned blog-topic links, not citation or causal links or the distribution of the entire research field.'
    },
    paperSlugs: entries.map(entry => entry.slug),
    sources: entries.map(entry => ({
      url: new URL(`${base}${entry.pair.ko ? '' : '/en'}/reviews/${entry.slug}/`, url).href,
      label: { ko: (entry.pair.ko || entry.pair.en).title, en: (entry.pair.en || entry.pair.ko).title },
      location: { ko: '공개 리뷰 메타데이터 · paperUrl / topics / 연재 순서', en: 'Public review metadata · paperUrl / topics / series order' }
    })),
    charts: [{
      id: 'primary-topics', type: 'bar',
      title: { ko: '대표 주제별 고유 논문 수', en: 'Unique papers by primary topic' },
      description: { ko: '논문마다 대표 리뷰의 첫 번째 주제 하나로 분류합니다. 번역·연재 중복을 제외합니다.', en: 'One primary topic per paper, taken from its representative review. Translations and series do not add duplicate papers.' },
      unit: { ko: '고유 논문 수', en: 'unique papers' }, direction: 'none',
      values: [...counts].sort(([a, av], [b, bv]) => bv - av || a.localeCompare(b)).map(([key, value]) => ({ label: topicLabel(key), value }))
    }, {
      id: 'shared-topics', type: 'network',
      title: { ko: '공통 주제로 연결한 논문', en: 'Papers connected by shared topics' },
      description: { ko: '점 하나는 고유 논문 하나이며, 선과 숫자는 두 논문에 공통으로 지정된 주제 수입니다.', en: 'Each node is one unique paper. Lines and numbers show the count of assigned topics shared by each pair.' },
      relationship: { ko: '무방향 관계 · 공통 주제 수', en: 'Undirected links · shared-topic count' },
      nodes: papers.map(paper => ({ id: paper.id, paperSlug: paper.slug, label: paper.label })), edges
    }]
  };
}
