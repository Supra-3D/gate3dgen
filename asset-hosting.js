// Public assets must be uploaded before deploying the GitHub Pages package.
(() => {
  'use strict';
  const revision = 'main'; // Replace with an uploaded commit SHA to pin a release.
  const baseUrl = `https://huggingface.co/datasets/hllee0615/gate3dgen-web-assets/resolve/${revision}/`;
  const source = new URLSearchParams(location.search).get('assetSource');
  const localPreview = location.protocol === 'file:' ||
    /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
  const useRemote = source === 'remote' || (source !== 'local' && !localPreview);
  const url = path => useRemote ? new URL(path, baseUrl).href : path;
  window.GATE3DGEN_ASSET_HOSTING = Object.freeze({ baseUrl, revision, useRemote, url });
  document.querySelectorAll('source[data-hosted-src]').forEach(sourceElement => {
    sourceElement.src = url(sourceElement.dataset.hostedSrc);
    sourceElement.parentElement.load();
  });
})();
