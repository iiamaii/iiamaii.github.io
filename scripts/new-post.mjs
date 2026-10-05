import fs from 'node:fs/promises';
const slug = process.argv[2];
const config = JSON.parse(await fs.readFile('site.json', 'utf8'));
const topic = process.argv[3] || config.topics[0]?.id;
if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) || !config.topics.some(t => t.id === topic)) {
  console.error('Usage: npm run new -- paper-slug [topic-id from site.json]');
  process.exit(1);
}
const date = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Seoul' });
const file = `content/reviews/${date}-${slug}.md`;
await fs.mkdir('content/reviews', { recursive: true });
await fs.writeFile(file, `---\ntitle: "논문 리뷰 제목"\ndescription: "논문의 핵심 아이디어와 리뷰에서 다루는 질문을 짧게 적어 주세요."\ndate: "${date}"\ntopic: "${topic}"\npaperTitle: ""\nauthors: ""\nyear: ""\npaperUrl: ""\nthumbnail: ""\nthumbnailAlt: ""\ndraft: true\n---\n\n## 어떤 질문에서 출발했는가\n\n논문의 문제와 배경을 정리합니다.\n\n## 핵심 아이디어\n\n방법과 핵심 가정을 나의 언어로 설명합니다.\n\n## 결과와 근거\n\n실험 결과, 비교 기준과 출처를 남깁니다.\n\n## 나의 생각\n\n논문의 강점, 한계와 해석을 적습니다.\n\n## 남은 질문\n\n다음에 읽거나 확인할 것을 적습니다.\n`, { flag: 'wx' });
console.log(`Created ${file}. Add paper details and set draft: false when ready to publish.`);
