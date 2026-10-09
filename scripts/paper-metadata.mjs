export function paperMetadata(data, file) {
  const value = data.paperPublishedDate instanceof Date ? data.paperPublishedDate.toISOString().slice(0, 10) : data.paperPublishedDate;
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value) || Number.isNaN(Date.parse(value)) || new Date(value).toISOString().slice(0, 10) !== value) {
    throw new Error(`${file}: paperPublishedDate must be a valid YYYY-MM-DD date from the original source.`);
  }
  // Legacy comma-separated names remain readable; new posts use an ordered list.
  const names = typeof data.authors === 'string' ? data.authors.split(',') : data.authors;
  if (!Array.isArray(names) || !names.length || names.some(name => typeof name !== 'string' || !name.trim() || /\bet\s+al\.?\b/i.test(name))) {
    throw new Error(`${file}: authors must contain all author names in source order, without et al.`);
  }
  return { paperPublishedDate: value, authors: names.map(name => name.trim()) };
}
