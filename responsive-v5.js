const siteHeader = document.querySelector('#site-header');
const menuToggle = document.querySelector('#menu-toggle');
const primaryNavigation = document.querySelector('#primary-navigation');
const phoneNavigation = window.matchMedia('(max-width: 680px)');
const themeToggle = document.querySelector('#theme-toggle');
const themeToggleIcon = themeToggle?.querySelector('.theme-toggle-icon');
const themeToggleLabel = themeToggle?.querySelector('.theme-toggle-label');
const themeColor = document.querySelector('#theme-color');
const THEME_STORAGE_KEY = 'gate3dgen-theme-v2';

function applyTheme(theme, persist = true) {
  const nextTheme = theme === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = nextTheme;
  if (themeColor) themeColor.content = nextTheme === 'light' ? '#f6f8fb' : '#06070a';

  document.querySelectorAll('[data-dark-src][data-light-src]').forEach(element => {
    const source = nextTheme === 'light' ? element.dataset.lightSrc : element.dataset.darkSrc;
    if (element instanceof HTMLImageElement) element.src = source;
    if (element.matches('[data-figure-expand]')) element.dataset.figureSrc = source;
  });

  const targetTheme = nextTheme === 'light' ? 'dark' : 'light';
  const targetLabel = targetTheme === 'light' ? 'Light' : 'Dark';
  if (themeToggleIcon) themeToggleIcon.textContent = targetTheme === 'light' ? '☀' : '☾';
  if (themeToggleLabel) themeToggleLabel.textContent = targetLabel;
  if (themeToggle) {
    themeToggle.setAttribute('aria-label', `Switch to ${targetTheme} theme`);
    themeToggle.setAttribute('title', `Switch to ${targetTheme} theme`);
    themeToggle.setAttribute('aria-pressed', String(nextTheme === 'light'));
  }

  if (persist) {
    try { localStorage.setItem(THEME_STORAGE_KEY, nextTheme); } catch {}
  }
}

applyTheme(document.documentElement.dataset.theme, false);

themeToggle?.addEventListener('click', () => {
  applyTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
});

window.addEventListener('storage', event => {
  if (event.key === THEME_STORAGE_KEY && (event.newValue === 'dark' || event.newValue === 'light')) {
    applyTheme(event.newValue, false);
  }
});

function setMenuOpen(open) {
  if (!siteHeader || !menuToggle) return;
  siteHeader.classList.toggle('menu-open', open);
  menuToggle.setAttribute('aria-expanded', String(open));
}

menuToggle?.addEventListener('click', () => {
  setMenuOpen(menuToggle.getAttribute('aria-expanded') !== 'true');
});

primaryNavigation?.addEventListener('click', event => {
  if (phoneNavigation.matches && event.target.closest('a')) setMenuOpen(false);
});

document.addEventListener('keydown', event => {
  if (event.key === 'Escape') setMenuOpen(false);
});

document.addEventListener('click', event => {
  if (!phoneNavigation.matches || !siteHeader?.classList.contains('menu-open')) return;
  if (!siteHeader.contains(event.target)) setMenuOpen(false);
});

phoneNavigation.addEventListener?.('change', event => {
  if (!event.matches) setMenuOpen(false);
});

const figureLightbox = document.querySelector('#figure-lightbox');
const figureLightboxImage = document.querySelector('#figure-lightbox-image');
const figureLightboxTitle = document.querySelector('#figure-lightbox-title');
const figureLightboxClose = document.querySelector('#figure-lightbox-close');
let figureReturnFocus = null;

function closeFigureLightbox() {
  if (!figureLightbox?.open) return;
  figureLightbox.close();
  figureReturnFocus?.focus();
}

document.querySelectorAll('[data-figure-expand]').forEach(button => {
  button.addEventListener('click', () => {
    if (!figureLightbox || !figureLightboxImage || !figureLightboxTitle) return;
    figureReturnFocus = button;
    figureLightboxImage.src = button.dataset.figureSrc;
    figureLightboxImage.alt = button.dataset.figureAlt || '';
    figureLightboxTitle.textContent = button.textContent.trim();
    figureLightbox.showModal();
  });
});

figureLightboxClose?.addEventListener('click', closeFigureLightbox);
figureLightbox?.addEventListener('click', event => {
  if (event.target === figureLightbox) closeFigureLightbox();
});

const taskVideo = document.querySelector('#task video');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const saveData = Boolean(navigator.connection?.saveData);

if (taskVideo && (saveData || reducedMotion.matches)) {
  taskVideo.autoplay = false;
  taskVideo.removeAttribute('autoplay');
  taskVideo.pause();
}

if (taskVideo && 'IntersectionObserver' in window && !saveData && !reducedMotion.matches) {
  const taskObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) taskVideo.play().catch(() => {});
      else taskVideo.pause();
    });
  }, { threshold: 0.25 });
  taskObserver.observe(taskVideo);
}
