export function normalizeTopic(value, label = 'topic') {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a non-empty string.`);
  return value.normalize('NFC').trim().replace(/\s+/g, ' ');
}

export function normalizeTopics(values, label = 'topics') {
  if (!Array.isArray(values) || !values.length) throw new Error(`${label} must be a non-empty list of strings.`);
  return [...new Set(values.map(value => normalizeTopic(value, `${label} entry`)))];
}
