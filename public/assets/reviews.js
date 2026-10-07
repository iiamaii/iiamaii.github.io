const queryModule = new URL('./review-query.js', import.meta.url);
queryModule.search = new URL(import.meta.url).search;
const { readReviewQuery, queryReviews } = await import(queryModule.href);
const data = document.getElementById('review-data');
if (data) {
  const { reviews, ui } = JSON.parse(data.textContent);
  const tools = document.querySelector('[data-review-tools]');
  const tablist = document.querySelector('[data-topic-tabs]');
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  const topicIds = tabs.map(tab => tab.dataset.topic);
  const panel = document.getElementById('review-panel');
  const grid = document.querySelector('[data-review-grid]');
  const cards = new Map([...grid.querySelectorAll('[data-review-slug]')].map(card => [card.dataset.reviewSlug, card]));
  const count = document.querySelector('[data-review-count]');
  const form = document.querySelector('[data-review-filters]');
  const toggle = document.querySelector('[data-filter-toggle]');
  const size = document.querySelector('[data-review-size]');
  const empty = document.querySelector('[data-no-results]');
  const pagination = document.querySelector('[data-review-pagination]');
  const previous = document.querySelector('[data-page-previous]');
  const next = document.querySelector('[data-page-next]');
  let state = readReviewQuery(new URL(location.href).searchParams, topicIds);
  let result;
  const message = (template, values) => template.replace(/\{(\w+)\}/g, (_, key) => values[key]);

  function showFilters(open) {
    form.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    toggle.lastElementChild.textContent = open ? '−' : '+';
  }

  function render(updateHistory = false) {
    result = queryReviews(reviews, state);
    state.page = result.page;
    const visible = new Set(result.items.map(review => review.slug));
    cards.forEach((card, slug) => { card.hidden = !visible.has(slug); });
    result.items.forEach(review => grid.append(cards.get(review.slug)));
    grid.hidden = !result.total;
    empty.hidden = reviews.length === 0 || result.total > 0;
    tabs.forEach(tab => {
      const selected = tab.dataset.topic === state.topic;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected) panel.setAttribute('aria-labelledby', tab.id);
    });
    const selectedName = tabs.find(tab => tab.dataset.topic === state.topic).firstChild.textContent;
    count.textContent = `${selectedName} · ${result.total ? message(ui.range, result) : ui.zero}`;
    for (const name of ['q', 'topic', 'from', 'to', 'sort']) form.elements.namedItem(name).value = state[name];
    form.elements.namedItem('to').setCustomValidity('');
    size.value = state.size;
    pagination.hidden = result.pageCount <= 1;
    previous.disabled = result.page <= 1;
    next.disabled = result.page >= result.pageCount;
    document.querySelector('[data-review-page]').textContent = message(ui.page, { page: result.page, pages: result.pageCount });
    document.querySelector('[data-filter-active]').hidden = !(state.q || state.from || state.to || state.topic !== 'all' || state.sort !== 'newest');
    if (updateHistory) {
      const url = new URL(location.href);
      for (const name of ['topic', 'q', 'from', 'to', 'sort', 'size', 'page']) url.searchParams.delete(name);
      for (const name of ['q', 'from', 'to']) if (state[name]) url.searchParams.set(name, state[name]);
      if (state.topic !== 'all') url.searchParams.set('topic', state.topic);
      if (state.sort !== 'newest') url.searchParams.set('sort', state.sort);
      if (state.size !== '12') url.searchParams.set('size', state.size);
      if (state.page !== 1) url.searchParams.set('page', String(state.page));
      if (url.href !== location.href) history.pushState(null, '', url);
      document.dispatchEvent(new Event('site:locationchange'));
    }
  }

  function change(patch) {
    state = { ...state, page: 1, ...patch };
    render(true);
  }
  const reset = () => change({ q: '', topic: 'all', from: '', to: '', sort: 'newest' });
  toggle.addEventListener('click', () => {
    showFilters(form.hidden);
    if (!form.hidden) form.elements.namedItem('q').focus();
  });
  form.addEventListener('submit', event => {
    event.preventDefault();
    const values = Object.fromEntries(new FormData(form));
    const end = form.elements.namedItem('to');
    end.setCustomValidity(values.from && values.to && values.to < values.from ? ui.rangeError : '');
    if (!form.reportValidity()) return;
    change({ ...values, q: values.q.trim() });
  });
  for (const name of ['from', 'to']) form.elements.namedItem(name).addEventListener('input', () => form.elements.namedItem('to').setCustomValidity(''));
  document.querySelector('[data-reset-filters]').addEventListener('click', reset);
  document.querySelector('[data-clear-search]').addEventListener('click', reset);
  size.addEventListener('change', () => change({ size: size.value }));
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => change({ topic: tab.dataset.topic }));
    tab.addEventListener('keydown', event => {
      let nextIndex;
      if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') nextIndex = 0;
      if (event.key === 'End') nextIndex = tabs.length - 1;
      if (nextIndex === undefined) return;
      event.preventDefault();
      tabs[nextIndex].focus();
      change({ topic: tabs[nextIndex].dataset.topic });
    });
  });
  function goToPage(page) {
    change({ page });
    panel.focus({ preventScroll: true });
    panel.scrollIntoView({ block: 'start' });
  }
  previous.addEventListener('click', () => goToPage(state.page - 1));
  next.addEventListener('click', () => goToPage(state.page + 1));
  window.addEventListener('popstate', () => {
    const hadTabFocus = tablist.contains(document.activeElement);
    state = readReviewQuery(new URL(location.href).searchParams, topicIds);
    render();
    if (state.q || state.from || state.to || state.sort !== 'newest') showFilters(true);
    if (hadTabFocus) tabs.find(tab => tab.getAttribute('aria-selected') === 'true').focus();
  });
  panel.setAttribute('role', 'tabpanel');
  panel.removeAttribute('aria-label');
  render();
  showFilters(Boolean(state.q || state.from || state.to || state.sort !== 'newest'));
  tools.hidden = false;
}
