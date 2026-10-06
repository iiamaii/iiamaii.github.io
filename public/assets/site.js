function syncLanguageLinks() {
  const current = new URL(window.location.href);
  document.querySelectorAll('[data-language-link]').forEach(link => {
    const target = new URL(link.href);
    if (document.body.dataset.preserveFilters === 'true') {
      target.search = current.search;
      target.hash = current.hash;
    }
    link.href = target.href;
  });
}
syncLanguageLinks();
window.addEventListener('popstate', syncLanguageLinks);
window.addEventListener('hashchange', syncLanguageLinks);
document.addEventListener('site:locationchange', syncLanguageLinks);
