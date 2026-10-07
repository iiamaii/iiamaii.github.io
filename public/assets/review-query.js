export const pageSizes = ['6', '12', '24', 'all'];
export const normalizeSearch = value => String(value).normalize('NFC').toLocaleLowerCase().replace(/\s+/g, ' ').trim();

export function dateTimeValue(value) {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value || '')) return '';
  const time = Date.parse(`${value}:00+09:00`);
  if (!Number.isFinite(time) || new Date(time + 9 * 60 * 60 * 1000).toISOString().slice(0, 16) !== value) return '';
  return value;
}

export function readReviewQuery(params, topicIds) {
  const topic = params.get('topic') || 'all';
  const size = params.get('size') || '12';
  const page = params.get('page') || '1';
  return {
    topic: topicIds.includes(topic) ? topic : 'all',
    q: (params.get('q') || '').trim(),
    from: dateTimeValue(params.get('from')),
    to: dateTimeValue(params.get('to')),
    sort: params.get('sort') === 'oldest' ? 'oldest' : 'newest',
    size: pageSizes.includes(size) ? size : '12',
    page: /^\d+$/.test(page) && Number.isSafeInteger(Number(page)) && Number(page) > 0 ? Number(page) : 1
  };
}

export function queryReviews(reviews, state) {
  const words = normalizeSearch(state.q).split(' ').filter(Boolean);
  const from = state.from ? Date.parse(`${state.from}:00+09:00`) : -Infinity;
  const to = state.to ? Date.parse(`${state.to}:00+09:00`) + 59999 : Infinity;
  const matches = reviews.filter(review => {
    const updated = Date.parse(review.updatedAt);
    return (state.topic === 'all' || review.topics.includes(state.topic)) && updated >= from && updated <= to && words.every(word => normalizeSearch(review.text).includes(word));
  }).sort((a, b) => (state.sort === 'oldest' ? 1 : -1) * (Date.parse(a.updatedAt) - Date.parse(b.updatedAt)) || a.slug.localeCompare(b.slug));
  const size = state.size === 'all' ? Math.max(1, matches.length) : Number(state.size);
  const pageCount = Math.max(1, Math.ceil(matches.length / size));
  const page = Math.min(state.page, pageCount);
  const offset = (page - 1) * size;
  const items = matches.slice(offset, offset + size);
  return { items, total: matches.length, page, pageCount, start: matches.length ? offset + 1 : 0, end: offset + items.length };
}
