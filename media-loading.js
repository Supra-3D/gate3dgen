// Both galleries settle before any selected-case full-resolution image starts.
(() => {
  const galleries = [];
  function waitForImage(image, signal) {
    return new Promise(resolve => {
      let finished = false;
      const finish = ok => {
        if (finished) return;
        finished = true;
        image.removeEventListener('load', loaded);
        image.removeEventListener('error', failed);
        signal?.removeEventListener('abort', aborted);
        resolve(ok);
      };
      const loaded = () => {
        if (image.decode) image.decode().then(() => finish(image.naturalWidth > 0), () => finish(false));
        else finish(image.naturalWidth > 0);
      };
      const failed = () => finish(false);
      const aborted = () => finish(false);
      image.addEventListener('load', loaded, { once: true });
      image.addEventListener('error', failed, { once: true });
      signal?.addEventListener('abort', aborted, { once: true });
      if (signal?.aborted) aborted();
      else if (image.complete) image.naturalWidth > 0 ? loaded() : failed();
    });
  }
  window.GATE3DGEN_MEDIA_LOADING = {
    waitForImage,
    registerThumbnails(images) {
      galleries.push(Promise.all([...images].map(image => waitForImage(image))));
    },
    async waitForThumbnails() {
      // Allow both createComparison calls to register their eager thumbnails.
      await Promise.resolve();
      await Promise.all(galleries);
    },
    afterPaint: () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))
  };
})();
