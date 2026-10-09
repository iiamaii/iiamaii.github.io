const escape = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export function renderStatistics({ analyses, reviews, lang, href }) {
  const ko = lang === 'ko', e = escape, local = value => e(value[lang]);
  const dataLabel = ko ? '데이터 보기' : 'View data';
  const format = value => new Intl.NumberFormat(ko ? 'ko-KR' : 'en-US', { maximumSignificantDigits: 12 }).format(value);
  function chartHtml(chart, analysisId) {
    const chartId = `${analysisId}-${chart.id}`;
    let visual, rows, headers, caption;
    if (chart.type === 'bar') {
      const min = Math.min(0, ...chart.values.map(row => row.value));
      const max = Math.max(0, ...chart.values.map(row => row.value));
      const extent = Math.max(Math.abs(min), Math.abs(max)) || 1;
      const range = max / extent - min / extent || 1;
      const zero = (-min / extent) / range * 100;
      visual = `<div class="stat-bars" role="img" aria-labelledby="${chartId}-title" aria-describedby="${chartId}-description"><div class="bar-axis"><span>${format(min)}</span><span>${format(max)}</span></div>${chart.values.map(row => {
        const position = (row.value / extent - min / extent) / range * 100;
        return `<div class="stat-bar-row"><span class="stat-bar-label">${local(row.label)}</span><div class="stat-bar-track"><span class="stat-bar-zero" style="left:${zero}%"></span><span class="stat-bar-fill" style="left:${Math.min(position, zero)}%;width:${Math.abs(position - zero)}%"></span></div><strong>${format(row.value)}</strong></div>`;
      }).join('')}</div>`;
      caption = `${local(chart.unit)} · ${chart.direction === 'lower' ? (ko ? '낮을수록 좋음' : 'Lower is better') : chart.direction === 'higher' ? (ko ? '높을수록 좋음' : 'Higher is better') : (ko ? '빈도·크기 표시' : 'Count / magnitude')}`;
      headers = [ko ? '항목' : 'Item', local(chart.unit)];
      rows = chart.values.map(row => [local(row.label), format(row.value)]);
    } else {
      // Put connected components together; isolated nodes remain visible.
      const ordered = [], visited = new Set();
      function visit(id) { if (visited.has(id)) return; visited.add(id); ordered.push(chart.nodes.find(node => node.id === id)); for (const edge of chart.edges) { if (edge.source === id) visit(edge.target); if (edge.target === id) visit(edge.source); } }
      chart.nodes.forEach(node => visit(node.id));
      const positions = new Map(ordered.map((node, i) => { const angle = -Math.PI / 2 + i * Math.PI * 2 / ordered.length; return [node.id, { x: 320 + 205 * Math.cos(angle), y: 195 + 120 * Math.sin(angle) }]; }));
      const maxWeight = Math.max(1, ...chart.edges.map(edge => edge.weight));
      visual = `<svg class="stat-network" viewBox="0 0 640 400" role="img" aria-labelledby="${chartId}-title" aria-describedby="${chartId}-description">${chart.edges.map(edge => {
        const a = positions.get(edge.source), b = positions.get(edge.target);
        return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke-width="${1.5 + edge.weight / maxWeight * 4}"/><text class="edge-weight" x="${(a.x + b.x) / 2}" y="${(a.y + b.y) / 2 - 8}" text-anchor="middle">${format(edge.weight)}</text>`;
      }).join('')}${ordered.map((node, index) => {
        const p = positions.get(node.id), label = node.label[lang];
        return `<g><title>${e(label)}</title><circle cx="${p.x}" cy="${p.y}" r="20"/><text class="node-number" x="${p.x}" y="${p.y + 5}" text-anchor="middle">${index + 1}</text><text class="node-label" x="${p.x}" y="${p.y + 44}" text-anchor="middle">${e(label.length > 24 ? `${label.slice(0, 23)}…` : label)}</text></g>`;
      }).join('')}</svg><ul class="network-legend">${ordered.map((node, index) => `<li><span class="network-node-key">${index + 1}.</span> ${node.paperSlug ? `<a href="${href(`/reviews/${node.paperSlug}/`)}">${local(node.label)} ↗</a>` : local(node.label)}</li>`).join('')}</ul>`;
      caption = local(chart.relationship);
      headers = [ko ? '연결 A' : 'Connection A', ko ? '연결 B' : 'Connection B', ko ? '가중치' : 'Weight'];
      rows = chart.edges.map(edge => [local(chart.nodes.find(node => node.id === edge.source).label), local(chart.nodes.find(node => node.id === edge.target).label), format(edge.weight)]);
    }
    return `<section class="stat-chart"><header><span class="section-kicker">${chart.type === 'bar' ? 'BAR CHART' : 'RELATIONSHIP GRAPH'}</span><h3 id="${chartId}-title">${local(chart.title)}</h3><p id="${chartId}-description">${local(chart.description)}</p></header>${visual}<p class="stat-chart-caption">${caption}</p><details class="stat-data"><summary>${dataLabel}</summary><div class="table-scroll"><table><thead><tr>${headers.map(header => `<th scope="col">${header}</th>`).join('')}</tr></thead><tbody>${rows.length ? rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="3">${ko ? '연결 없음' : 'No connections'}</td></tr>`}</tbody></table></div></details></section>`;
  }
  const countTopics = new Set(reviews.flatMap(review => review.topics.map(topic => topic.id))).size;
  return `<main id="main" class="statistics-page"><div class="page-heading"><p class="section-kicker"><span class="kicker-number">03</span><span>PAPER STATISTICS</span></p><h1>${ko ? '논문 통계' : 'Paper statistics'}<span class="brand-dot" aria-hidden="true">.</span></h1><p>${ko ? '논문에서 읽어낸 수치와 연구 사이의 연결을 살펴봅니다.' : 'Explore reported results and connections across research.'}</p></div><div class="statistics-overview"><div><strong>${reviews.length}</strong><span>${ko ? '공개 리뷰' : 'Public reviews'}</span></div><div><strong>${countTopics}</strong><span>${ko ? '리뷰 주제' : 'Review topics'}</span></div><div><strong>${analyses.length}</strong><span>${ko ? '저장된 분석' : 'Saved analyses'}</span></div></div>${analyses.length ? `<div class="statistics-toolbar" data-statistics-control hidden><label for="analysis-select">${ko ? '분석 선택' : 'Choose analysis'}</label><select id="analysis-select"><option value="all">${ko ? '전체 분석' : 'All analyses'}</option>${analyses.map(record => `<option value="${record.id}">${local(record.title)}</option>`).join('')}</select><span data-analysis-count aria-live="polite"></span></div><div class="statistics-analyses">${analyses.map(record => `<article class="stat-analysis" id="${record.id}" data-analysis="${record.id}"><header class="stat-analysis-heading"><span class="section-kicker">SAVED ANALYSIS · <time datetime="${e(record.updatedAt)}">${new Intl.DateTimeFormat(ko ? 'ko-KR' : 'en-GB', { timeZone: 'Asia/Seoul', dateStyle: 'medium' }).format(new Date(record.updatedAt))}</time></span><h2>${local(record.title)}</h2><p>${local(record.summary)}</p></header><div class="statistics-charts">${record.charts.map(chart => chartHtml(chart, record.id)).join('')}</div><aside class="stat-method"><h3>${ko ? '범위와 해석' : 'Scope and interpretation'}</h3><p>${local(record.notes)}</p>${record.paperSlugs.length ? `<p class="stat-related">${ko ? '관련 리뷰' : 'Related reviews'} · ${record.paperSlugs.map(slug => `<a href="${href(`/reviews/${slug}/`)}">${e(reviews.find(review => review.slug === slug).paperTitle || reviews.find(review => review.slug === slug).title)}</a>`).join(' / ')}</p>` : ''}<ol>${record.sources.map(source => `<li><a href="${e(source.url)}" target="_blank" rel="noopener noreferrer">${local(source.label)} ↗</a><span> · ${local(source.location)}</span></li>`).join('')}</ol></aside></article>`).join('')}</div>` : `<p class="statistics-empty">${ko ? '공개된 분석이 준비되면 이곳에 표시됩니다.' : 'Published analyses will appear here.'}</p>`}</main>`;
}
