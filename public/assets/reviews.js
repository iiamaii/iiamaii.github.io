const queryModule = new URL('./review-query.js', import.meta.url);
queryModule.search = new URL(import.meta.url).search;
const { readReviewQuery, queryReviews, reviewBatch } = await import(queryModule.href);
const data = document.getElementById('review-data');
if (data) {
  const { reviews, ui } = JSON.parse(data.textContent);
  const tools = document.querySelector('[data-review-tools]');
  const grid = document.querySelector('[data-review-grid]');
  const cards = new Map([...grid.querySelectorAll('[data-review-slug]')].map(card => [card.dataset.reviewSlug, card]));
  const count = document.querySelector('[data-review-count]');
  const filterForm = document.querySelector('[data-review-filters]');
  const searchForm = document.querySelector('[data-review-search]');
  const filterToggle = document.querySelector('[data-filter-toggle]');
  const searchToggle = document.querySelector('[data-search-toggle]');
  const topicPicker = document.querySelector('[data-topic-picker]');
  const topicInputs = [...filterForm.querySelectorAll('input[name="topic"]')];
  const topicIds = topicInputs.map(input => input.value);
  const size = document.querySelector('[data-review-size]');
  const sort = document.querySelector('[data-review-sort]');
  const empty = document.querySelector('[data-no-results]');
  const more = document.querySelector('[data-review-more]');
  const moreButton = document.querySelector('[data-load-more]');
  let state = readReviewQuery(new URL(location.href).searchParams, topicIds);
  let result;
  const message = (template, values) => template.replace(/\{(\w+)\}/g, (_, key) => values[key]);
  const observer = 'IntersectionObserver' in window ? new IntersectionObserver(entries => {
    if (entries.some(entry => entry.isIntersecting) && result?.hasMore) loadMore();
  }, { rootMargin: '200px 0px' }) : null;

  function showPanel(form, toggle, open) {
    form.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    toggle.lastElementChild.textContent = open ? '−' : '+';
  }

  const hasFilters = () => Boolean(state.from || state.to || state.topics.length);
  function summarizeTopics() {
    const selected = topicInputs.filter(input => input.checked).length;
    document.querySelector('[data-topic-selection]').textContent = selected ? message(ui.selectedTopics, { count: selected }) : ui.allTopics;
  }

  function syncUrl(mode) {
    const url = new URL(location.href);
    for (const name of ['topic', 'q', 'from', 'to', 'sort', 'size', 'page']) url.searchParams.delete(name);
    for (const name of ['q', 'from', 'to']) if (state[name]) url.searchParams.set(name, state[name]);
    state.topics.forEach(topic => url.searchParams.append('topic', topic));
    if (state.sort !== 'newest') url.searchParams.set('sort', state.sort);
    if (state.size !== '12') url.searchParams.set('size', state.size);
    if (state.page !== 1) url.searchParams.set('page', String(state.page));
    if (url.href !== location.href) history[mode === 'push' ? 'pushState' : 'replaceState'](null, '', url);
    document.dispatchEvent(new Event('site:locationchange'));
  }

  function render(historyMode, changedFields = ['q', 'topics', 'from', 'to', 'sort', 'size'], append = false) {
    observer?.disconnect();
    result = append ? reviewBatch(result.matches, state) : queryReviews(reviews, state);
    state.page = result.page;
    // Keep unshown cards detached: only the loaded batches take part in layout.
    if (!append) grid.replaceChildren();
    for (const review of result.items) {
      const card = cards.get(review.slug);
      if (card.parentElement !== grid) grid.append(card);
    }
    grid.hidden = !result.total;
    empty.hidden = reviews.length === 0 || result.total > 0;
    count.textContent = result.total ? message(ui.range, result) : ui.zero;
    for (const name of changedFields) {
      if (name === 'topics') {
        topicInputs.forEach(input => { input.checked = state.topics.includes(input.value); });
        summarizeTopics();
      } else {
        const input = filterForm.elements.namedItem(name) || searchForm.elements.namedItem(name);
        if (input) input.value = state[name];
      }
    }
    if (changedFields.some(name => name === 'from' || name === 'to')) filterForm.elements.namedItem('to').setCustomValidity('');
    size.value = state.size;
    sort.value = state.sort;
    document.querySelector('[data-filter-active]').hidden = !hasFilters();
    document.querySelector('[data-search-active]').hidden = !state.q;
    more.hidden = !result.hasMore;
    document.querySelector('[data-load-progress]').textContent = message(ui.range, result);
    if (historyMode) syncUrl(historyMode);
    if (result.hasMore) observer?.observe(more);
  }

  function change(patch) {
    state = { ...state, page: 1, ...patch };
    if (grid.getBoundingClientRect().top < 0) tools.scrollIntoView({ block: 'start', behavior: 'instant' });
    render('push', Object.keys(patch));
  }
  function loadMore(focusNewCard = false) {
    if (!result.hasMore) return;
    const firstNew = result.end;
    state.page += 1;
    render('replace', [], true);
    if (focusNewCard) cards.get(result.items[firstNew].slug).querySelector('a').focus({ preventScroll: true });
  }
  const reset = () => change({ q: '', topics: [], from: '', to: '' });
  for (const [form, toggle, firstField] of [[filterForm, filterToggle, topicPicker.querySelector('summary')], [searchForm, searchToggle, searchForm.elements.namedItem('q')]]) {
    toggle.addEventListener('click', () => {
      showPanel(form, toggle, form.hidden);
      if (!form.hidden) firstField.focus();
    });
  }
  filterForm.addEventListener('submit', event => {
    event.preventDefault();
    const values = new FormData(filterForm);
    const from = values.get('from');
    const to = values.get('to');
    filterForm.elements.namedItem('to').setCustomValidity(from && to && to < from ? ui.rangeError : '');
    if (!filterForm.reportValidity()) return;
    change({ topics: values.getAll('topic'), from, to });
  });
  searchForm.addEventListener('submit', event => {
    event.preventDefault();
    change({ q: searchForm.elements.namedItem('q').value.trim() });
  });
  topicInputs.forEach(input => input.addEventListener('change', summarizeTopics));
  document.querySelector('[data-clear-topics]').addEventListener('click', () => {
    topicInputs.forEach(input => { input.checked = false; });
    summarizeTopics();
  });
  for (const name of ['from', 'to']) filterForm.elements.namedItem(name).addEventListener('input', () => filterForm.elements.namedItem('to').setCustomValidity(''));
  document.querySelector('[data-reset-filters]').addEventListener('click', () => change({ topics: [], from: '', to: '' }));
  document.querySelector('[data-reset-search]').addEventListener('click', () => change({ q: '' }));
  document.querySelector('[data-clear-search]').addEventListener('click', reset);
  size.addEventListener('change', () => change({ size: size.value }));
  sort.addEventListener('change', () => change({ sort: sort.value }));
  moreButton.addEventListener('click', () => loadMore(true));
  window.addEventListener('popstate', () => {
    state = readReviewQuery(new URL(location.href).searchParams, topicIds);
    render();
    if (hasFilters()) showPanel(filterForm, filterToggle, true);
    if (state.topics.length) topicPicker.open = true;
    if (state.q) showPanel(searchForm, searchToggle, true);
  });
  render();
  showPanel(filterForm, filterToggle, hasFilters());
  if (state.topics.length) topicPicker.open = true;
  showPanel(searchForm, searchToggle, Boolean(state.q));
  tools.hidden = false;
}
