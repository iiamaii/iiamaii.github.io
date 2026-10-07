export function postTimes(data, date, file) {
  function timestamp(value, field, fallback) {
    if (value === undefined) return fallback;
    const source = value instanceof Date ? value.toISOString() : value;
    const match = typeof source === 'string' && source.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,3})?(Z|[+-]\d{2}:\d{2})$/);
    if (!match || Number.isNaN(Date.parse(source)) || Number(match[2]) > 23 || Number(match[3]) > 59 || Number(match[4]) > 59 || new Date(`${match[1]}T00:00:00Z`).toISOString().slice(0, 10) !== match[1]) {
      throw new Error(`${file}: ${field} must be an ISO datetime with seconds and a timezone (e.g. 2026-10-07T14:30:00+09:00).`);
    }
    return new Date(source).toISOString();
  }
  const publishedAt = timestamp(data.publishedAt, 'publishedAt', new Date(`${date}T00:00:00+09:00`).toISOString());
  if (new Date(publishedAt).toLocaleDateString('en-CA', { timeZone: 'Asia/Seoul' }) !== date) throw new Error(`${file}: publishedAt must fall on date in Asia/Seoul.`);
  const updatedAt = timestamp(data.updatedAt, 'updatedAt', publishedAt);
  if (Date.parse(updatedAt) < Date.parse(publishedAt)) throw new Error(`${file}: updatedAt cannot be before publishedAt.`);
  return { publishedAt, updatedAt, hasTime: data.publishedAt !== undefined || data.updatedAt !== undefined };
}
