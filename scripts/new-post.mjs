import fs from 'node:fs/promises';
import matter from 'gray-matter';

const slug = process.argv[2];
const topic = (process.argv[3] || '미분류').normalize('NFC').trim().replace(/\s+/g, ' ');
if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) || !topic) {
  console.error('Usage: npm run new -- paper-slug ["topic name"]');
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
const templates = {
  ko: { title: '논문 리뷰 제목', description: '논문의 핵심 아이디어와 리뷰에서 다루는 질문을 짧게 적어 주세요.', body: '## 어떤 질문에서 출발했는가\n\n논문의 문제와 배경을 정리합니다.\n\n## 핵심 아이디어\n\n방법과 핵심 가정을 나의 언어로 설명합니다.\n\n## 결과와 근거\n\n실험 결과, 비교 기준과 출처를 남깁니다.\n\n## 나의 생각\n\n논문의 강점, 한계와 해석을 적습니다.\n\n## 남은 질문\n\n다음에 읽거나 확인할 것을 적습니다.' },
  en: { title: 'Paper review title', description: 'A short introduction to the key idea and the question explored in this review.', body: '## The question\n\nDescribe the problem and its background.\n\n## Key idea\n\nExplain the method and assumptions in your own words.\n\n## Results and evidence\n\nRecord results, comparisons, and sources.\n\n## My reflections\n\nDiscuss strengths, limitations, and your interpretation.\n\n## Open questions\n\nList what to read or investigate next.' }
};
for (const [lang, template] of Object.entries(templates)) {
  const file = `content/reviews/${date}-${slug}.${lang}.md`;
  const metadata = { title: template.title, description: template.description, date, topic, visibility: 'private', lang, translationKey: slug, paperTitle: '', authors: '', year: '', paperUrl: '', thumbnail: '', thumbnailAlt: '' };
  await fs.writeFile(file, `---\n${Object.entries(metadata).map(([key, value]) => `${key}: ${JSON.stringify(value)}`).join('\n')}\n---\n\n${template.body}\n`, { flag: 'wx' });
  console.log(`Created ${file}`);
}
console.log('Fill each language version and set visibility: public when ready. New topics are created from public reviews. Unpublished translations display the public original with a notice.');
