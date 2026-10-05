import fs from 'node:fs/promises';
const slug = process.argv[2];
if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
  console.error('Usage: npm run new -- my-research-note');
  process.exit(1);
}
const date = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Seoul' });
const file = `content/posts/${date}-${slug}.md`;
await fs.writeFile(file, `---\ntitle: "새로운 연구 기록"\ndescription: "이 글에서 다루는 질문을 짧게 적어 주세요."\ndate: "${date}"\ncategory: "연구 기록"\ntags: ["Research"]\ndraft: true\n---\n\n## 질문\n\n확인하고 싶은 것을 적습니다.\n\n## 배경과 근거\n\n참고한 자료와 출처를 남깁니다.\n\n## 시도와 관찰\n\n실험 조건, 코드와 관찰한 결과를 기록합니다.\n\n## 남은 질문\n\n다음에 확인할 것을 적습니다.\n`, { flag: 'wx' });
console.log(`Created ${file}. Set draft: false when ready to publish.`);
