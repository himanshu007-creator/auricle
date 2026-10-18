// Theme switch: follows the OS until the visitor picks, then remembers the pick in this browser.
(function () {
  var KEY = 'auricle-theme', root = document.documentElement;
  function saved() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function current() { var q = new URLSearchParams(location.search).get('theme'); if (q === 'light' || q === 'dark') return q; return saved() || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'); }
  function apply(t) {
    root.setAttribute('data-theme', t);
    document.querySelectorAll('img[data-light]').forEach(function (img) {
      if (!img.dataset.dark) img.dataset.dark = img.getAttribute('src');
      img.setAttribute('src', t === 'light' ? img.dataset.light : img.dataset.dark);
    });
    document.querySelectorAll('.theme-toggle').forEach(function (b) {
      b.setAttribute('aria-label', t === 'light' ? 'Switch to dark theme' : 'Switch to light theme');
      b.textContent = t === 'light' ? '☾' : '☀';
    });
  }
  apply(current());
  document.addEventListener('DOMContentLoaded', function () {
    apply(current());
    document.querySelectorAll('.theme-toggle').forEach(function (b) {
      b.addEventListener('click', function () {
        var t = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
        try { localStorage.setItem(KEY, t); } catch (e) {}
        apply(t);
      });
    });
  });
  matchMedia('(prefers-color-scheme: light)').addEventListener('change', function () { if (!saved()) apply(current()); });
})();
