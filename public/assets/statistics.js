const control = document.querySelector('[data-statistics-control]');
const select = document.querySelector('#analysis-select');
const analyses = [...document.querySelectorAll('[data-analysis]')];
if (control && select) {
  control.hidden = false;
  function apply() {
    const requested = new URL(location.href).searchParams.get('analysis');
    select.value = analyses.some(item => item.dataset.analysis === requested) ? requested : 'all';
    analyses.forEach(item => { item.hidden = select.value !== 'all' && item.dataset.analysis !== select.value; });
    const count = analyses.filter(item => !item.hidden).length;
    document.querySelector('[data-analysis-count]').textContent = document.documentElement.lang === 'ko' ? `${count}개 분석` : `${count} ${count === 1 ? 'analysis' : 'analyses'}`;
  }
  select.addEventListener('change', () => {
    const url = new URL(location.href);
    if (select.value === 'all') url.searchParams.delete('analysis'); else url.searchParams.set('analysis', select.value);
    history.pushState(null, '', url);
    apply();
    document.dispatchEvent(new Event('site:locationchange'));
  });
  window.addEventListener('popstate', apply);
  apply();
}
