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
  ko: { title: '논문 리뷰 제목', description: '논문의 핵심 아이디어와 리뷰에서 다루는 질문을 짧게 적어 주세요.' },
  en: { title: 'Paper review title', description: 'A short introduction to the key idea and the question explored in this review.' }
};
for (const [lang, template] of Object.entries(templates)) {
  const file = `content/reviews/${date}-${slug}.${lang}.md`;
  const bodyFile = lang === 'en' ? 'review-body.en.md' : 'review-body.md';
  const body = await fs.readFile(new URL('../skills/publish-research-blog/assets/' + bodyFile, import.meta.url), 'utf8');
  const metadata = { title: template.title, description: template.description, date, publishedAt: timestamp, updatedAt: timestamp, topics, visibility: 'private', lang, translationKey: slug, paperTitle: '', paperPublishedDate: '', authors: [], year: '', paperUrl: '', thumbnail: '', thumbnailAlt: '' };
  await fs.writeFile(file, `---\n${Object.entries(metadata).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n')}\n---\n\n${body.trimEnd()}\n`, { flag: 'wx' });
  console.log(`Created ${file}`);
}
console.log('Fill each language version and set visibility: public when ready. New topics are created from public reviews. Unpublished translations display the public original with a notice.');
