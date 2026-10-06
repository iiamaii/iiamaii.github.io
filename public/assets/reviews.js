const tablist = document.querySelector('[data-topic-tabs]');
if (tablist) {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  const groups = [...document.querySelectorAll('[data-topic-group]')];
  const panel = document.getElementById('review-panel');
  const count = document.querySelector('[data-review-count]');
  const validTopics = new Set(tabs.map(tab => tab.dataset.topic));

  function selectTopic(topic, updateHistory = false) {
    if (!validTopics.has(topic)) topic = 'all';
    let total = 0;
    tabs.forEach(tab => {
      const selected = tab.dataset.topic === topic;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected) panel.setAttribute('aria-labelledby', tab.id);
    });
    groups.forEach(group => {
      const cards = [...group.querySelectorAll('.review-card')];
      cards.forEach(card => {
        card.hidden = topic === 'all' && card.dataset.primaryTopic !== group.dataset.topicGroup;
      });
      const visibleCount = cards.filter(card => !card.hidden).length;
      group.hidden = !visibleCount || (topic !== 'all' && group.dataset.topicGroup !== topic);
      group.querySelector('[data-topic-review-count]').textContent = `${String(visibleCount).padStart(2, '0')} REVIEWS`;
      if (!group.hidden) total += visibleCount;
    });
    const selectedName = tabs.find(tab => tab.dataset.topic === topic).childNodes[0].textContent;
    count.textContent = document.documentElement.lang === 'en'
      ? `${selectedName} · ${total} review${total === 1 ? '' : 's'}`
      : `${selectedName} · ${total}편의 리뷰`;
    if (updateHistory) {
      const url = new URL(window.location.href);
      if (topic === 'all') url.searchParams.delete('topic');
      else url.searchParams.set('topic', topic);
      window.history.pushState({ topic }, '', url);
      document.dispatchEvent(new Event('site:locationchange'));
    }
  }
  tablist.hidden = false;
  panel.setAttribute('role', 'tabpanel');
  panel.removeAttribute('aria-label');
  selectTopic(new URL(window.location.href).searchParams.get('topic') || 'all');
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectTopic(tab.dataset.topic, true));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      tabs[next].focus();
      selectTopic(tabs[next].dataset.topic, true);
    });
  });
  window.addEventListener('popstate', () => {
    const hadTabFocus = tablist.contains(document.activeElement);
    selectTopic(new URL(window.location.href).searchParams.get('topic') || 'all');
    if (hadTabFocus) tabs.find(tab => tab.getAttribute('aria-selected') === 'true').focus();
  });
}
