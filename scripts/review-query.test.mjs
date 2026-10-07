import assert from 'node:assert/strict';
import test from 'node:test';
import { readReviewQuery, queryReviews } from '../public/assets/review-query.js';
import { postTimes } from './post-times.mjs';

const read = query => readReviewQuery(new URLSearchParams(query), ['all', 'alpha', 'beta']);
const reviews = Array.from({ length: 15 }, (_, index) => ({
  slug: `paper-${String(index).padStart(2, '0')}`,
  topics: index % 2 ? ['alpha', 'beta'] : ['alpha'],
  publishedAt: '2020-03-01T00:00:00+09:00',
  updatedAt: `2020-03-03T18:${String(index).padStart(2, '0')}:42+09:00`,
  text: index % 2 ? 'Teacher consistency GPU 컴퓨터 비전. Evidence in the full body.' : 'Robotics visual calibration.'
}));

test('newest updates sort by time within one date; old-first reverses them', () => {
  assert.equal(queryReviews(reviews, read('')).items[0].slug, 'paper-14');
  assert.equal(queryReviews(reviews, read('sort=oldest')).items[0].slug, 'paper-00');
});

test('topic, every search word, and an inclusive KST minute range combine', () => {
  const result = queryReviews(reviews, read('topic=beta&q=CONSISTENCY+GPU&from=2020-03-03T18:05&to=2020-03-03T18:09'));
  assert.deepEqual(result.items.map(review => review.slug), ['paper-09', 'paper-07', 'paper-05']);
  assert.equal(result.total, 3);
  assert.equal(queryReviews(reviews, read('q=teacher+calibration')).total, 0, 'Every word must match the same review.');
  assert.equal(queryReviews(reviews, read(`q=${encodeURIComponent('컴퓨터 비전'.normalize('NFD'))}`)).total, 7, 'Hangul normalization is consistent.');
  assert.equal(queryReviews(reviews, read('q=full+body')).total, 7);
});

test('page size applies to filtered unique papers, page bounds clamp, All shows every match', () => {
  const first = queryReviews(reviews, read('size=6'));
  const second = queryReviews(reviews, read('size=6&page=2'));
  const last = queryReviews(reviews, read('size=6&page=99'));
  assert.equal(first.items.length, 6);
  assert.deepEqual([second.start, second.end, second.pageCount], [7, 12, 3]);
  assert.deepEqual([last.page, last.start, last.end], [3, 13, 15]);
  assert.equal(new Set([...first.items, ...second.items, ...last.items].map(review => review.slug)).size, 15);
  assert.equal(queryReviews(reviews, read('size=all')).items.length, 15);
  const empty = queryReviews(reviews, read('q=missing&page=4'));
  assert.deepEqual([empty.page, empty.total, empty.start, empty.end], [1, 0, 0, 0]);
});

test('invalid URL controls fall back safely and reversed ranges yield no matches', () => {
  const state = read('topic=unknown&size=0&page=-1&sort=unknown&from=2020-02-31T12:00&to=bad');
  assert.deepEqual([state.topic, state.size, state.page, state.sort, state.from, state.to], ['all', '12', 1, 'newest', '', '']);
  assert.equal(queryReviews(reviews, read('from=2020-03-04T12:00&to=2020-03-03T12:00')).total, 0);
});

test('legacy dates retain midnight KST; explicit timestamps keep seconds and timezone', () => {
  assert.deepEqual(postTimes({}, '2020-03-01', 'legacy.md'), { publishedAt: '2020-02-29T15:00:00.000Z', updatedAt: '2020-02-29T15:00:00.000Z', hasTime: false });
  const timed = postTimes({ publishedAt: '2020-03-01T09:30:45+09:00', updatedAt: '2020-03-03T18:05:42+09:00' }, '2020-03-01', 'timed.md');
  assert.equal(timed.publishedAt, '2020-03-01T00:30:45.000Z');
  assert.equal(timed.updatedAt, '2020-03-03T09:05:42.000Z');
  assert.equal(timed.hasTime, true);
  for (const data of [
    { publishedAt: '2020-03-01T09:00:00' },
    { publishedAt: '2020-02-30T09:00:00+09:00' },
    { publishedAt: '2020-03-02T09:00:00+09:00' },
    { updatedAt: '2020-02-29T12:00:00+09:00' },
    { publishedAt: '2020-03-01T25:00:00+09:00' }
  ]) assert.throws(() => postTimes(data, '2020-03-01', 'bad.md'));
});
