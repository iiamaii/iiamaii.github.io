(() => {
  const key = 'iiamaii-theme';
  const root = document.documentElement;
  const system = window.matchMedia('(prefers-color-scheme: dark)');
  const validTheme = value => value === 'dark' || value === 'light' ? value : null;
  const storedTheme = () => {
    try { return validTheme(localStorage.getItem(key)); }
    catch { return null; }
  };
  let preference = storedTheme();

  function apply(theme) {
    root.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]').content = theme === 'dark' ? '#171b19' : '#f7f7f3';
    const button = document.querySelector('[data-theme-toggle]');
    if (button) {
      const label = theme === 'dark' ? button.dataset.lightLabel : button.dataset.darkLabel;
      button.setAttribute('aria-label', label);
      button.title = label;
    }
  }

  // Runs before the stylesheet to apply the saved theme before the first paint.
  apply(preference ?? (system.matches ? 'dark' : 'light'));
  document.addEventListener('DOMContentLoaded', () => {
    const button = document.querySelector('[data-theme-toggle]');
    if (!button) return;
    apply(root.dataset.theme);
    button.hidden = false;
    button.addEventListener('click', () => {
      preference = root.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(key, preference); }
      catch { /* Switching still works when browser storage is unavailable. */ }
      apply(preference);
    });
  });
  system.addEventListener('change', event => {
    if (!preference) apply(event.matches ? 'dark' : 'light');
  });
  window.addEventListener('storage', event => {
    if (event.key !== key && event.key !== null) return;
    preference = storedTheme();
    apply(preference ?? (system.matches ? 'dark' : 'light'));
  });
})();
