import fs from 'node:fs/promises';
import matter from 'gray-matter';
import { normalizeTopics } from './topics.mjs';

const slug = process.argv[2];
const topicArguments = process.argv.slice(3);
let topics;
try {
  topics = normalizeTopics(topicArguments.length ? topicArguments : ['미분류']);
} catch (error) {
  console.error(error.message);
  process.exit(1);
}
if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
  console.error('Usage: npm run new -- paper-slug ["topic name" ...]');
  process.exit(1);
}
await fs.mkdir('content/reviews', { recursive: true });
for (const file of (await fs.readdir('content/reviews')).filter(name => name.endsWith('.md'))) {
  const { data } = matter(await fs.readFile(`content/reviews/${file}`, 'utf8'));
  const key = data.translationKey || file.replace(/\.md$/, '').replace(/\.(ko|en)$/, '').replace(/^\d{4}-\d{2}-\d{2}-/, '');
  if (key === slug) {
    console.error(`A review with translationKey ${slug} already exists: ${file}`);
    process.exit(1);
  }
}
const date = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Seoul' });
const timestamp = new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
const templates = {
  ko: { title: '논문 리뷰 제목', description: '논문의 핵심 아이디어와 리뷰에서 다루는 질문을 짧게 적어 주세요.', body: '핵심 기여, 조건을 붙인 대표 수치 1–2개, 리뷰어의 한 줄 판정을 2–3문장으로 적습니다.\n\n## 1. 한국어 절 제목 (Original Section Name)\n\n원문 절 순서대로 방법과 결과를 원문 위치와 함께 적습니다.\n\n## 리뷰 해석과 남은 질문\n\n종합 판정, 강점, 한계와 남은 질문을 적습니다.\n\n## 출처와 읽은 범위\n\n- 원문:\n- 읽은 범위:\n- 대표 썸네일 출처:\n- 확인하지 않은 것:' },
  en: { title: 'Paper review title', description: 'A short introduction to the key idea and the question explored in this review.', body: 'State the core contribution, one or two representative results with their conditions, and a one-line verdict in two or three sentences.\n\n## 1. Original Section Name\n\nFollow the paper\'s section order and cite locations for methods and results.\n\n## Review interpretation and open questions\n\nGive the overall assessment, strengths, limitations and open questions.\n\n## Sources and reading scope\n\n- Paper:\n- Scope read:\n- Thumbnail source:\n- Not verified:' }
};
for (const [lang, template] of Object.entries(templates)) {
  const file = `content/reviews/${date}-${slug}.${lang}.md`;
  const metadata = { title: template.title, description: template.description, date, publishedAt: timestamp, updatedAt: timestamp, topics, visibility: 'private', lang, translationKey: slug, paperTitle: '', paperPublishedDate: '', authors: [], year: '', paperUrl: '', thumbnail: '', thumbnailAlt: '' };
  await fs.writeFile(file, `---\n${Object.entries(metadata).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n')}\n---\n\n${template.body}\n`, { flag: 'wx' });
  console.log(`Created ${file}`);
}
console.log('Fill each language version and set visibility: public when ready. New topics are created from public reviews. Unpublished translations display the public original with a notice.');
