const RESULT_CASES = [
  { id: 'case-7-006-c62ddf3a2acb', title: 'case7-006', note: 'case-7-006-c62ddf3a2acb', beforeImage: 'assets/results/case-7-006-c62ddf3a2acb/before.png', afterImage: 'assets/results/case-7-006-c62ddf3a2acb/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'case-5-inthewild-002-0eb838f3a4d6', title: 'case5-002', note: 'case-5-inthewild-002-0eb838f3a4d6', beforeImage: 'assets/results/case-5-inthewild-002-0eb838f3a4d6/before.png', afterImage: 'assets/results/case-5-inthewild-002-0eb838f3a4d6/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'case-6-026-54f9eb8fa0c9', title: 'case6-026', note: 'case-6-026-54f9eb8fa0c9', beforeImage: 'assets/results/case-6-026-54f9eb8fa0c9/before.png', afterImage: 'assets/results/case-6-026-54f9eb8fa0c9/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'case-5-inthewild-015-90e752ed74ac', title: 'case5-015', note: 'case-5-inthewild-015-90e752ed74ac', beforeImage: 'assets/results/case-5-inthewild-015-90e752ed74ac/before.png', afterImage: 'assets/results/case-5-inthewild-015-90e752ed74ac/after.png', camera: { theta: -96, phi: 82, radius: 105 } },
  { id: 'wild-119-dcfa90255e2f', title: 'wild-119', note: 'wild-119-dcfa90255e2f', beforeImage: 'assets/results/wild-119-dcfa90255e2f/before.png', afterImage: 'assets/results/wild-119-dcfa90255e2f/after.png', camera: { theta: 168, phi: 82, radius: 105 } },
  { id: 'case-5-inthewild-017-b53c552cec87', title: 'case5-017', note: 'case-5-inthewild-017-b53c552cec87', beforeImage: 'assets/results/case-5-inthewild-017-b53c552cec87/before.png', afterImage: 'assets/results/case-5-inthewild-017-b53c552cec87/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'wild-024-edd235c90cb7', title: 'wild-024', note: 'wild-024-edd235c90cb7 · Pixal3D', beforeImage: 'assets/results/wild-024-edd235c90cb7/before.png', afterImage: 'assets/results/wild-024-edd235c90cb7/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'wild-035-aef7dc93b423', title: 'wild-035', note: 'wild-035-aef7dc93b423 · Pixal3D', beforeImage: 'assets/results/wild-035-aef7dc93b423/before.png', afterImage: 'assets/results/wild-035-aef7dc93b423/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'case-6-004-76507ed409e7', title: 'case6-004', note: 'case-6-004-76507ed409e7 · Pixal3D', beforeImage: 'assets/results/case-6-004-76507ed409e7/before.png', afterImage: 'assets/results/case-6-004-76507ed409e7/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'generated-motion-04-formula-fence', title: 'motion_04', note: 'generated-motion-04-formula-fence · Pixal3D', beforeImage: 'assets/results/generated-motion-04-formula-fence/before.png', afterImage: 'assets/results/generated-motion-04-formula-fence/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'case-6-021-a29b3548388f', title: 'case6-021', note: 'case-6-021-a29b3548388f · Pixal3D', beforeImage: 'assets/results/case-6-021-a29b3548388f/before.png', afterImage: 'assets/results/case-6-021-a29b3548388f/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'case-6-037-ffbbda014502', title: 'case6-037', note: 'case-6-037-ffbbda014502 · Pixal3D', modelVersion: 'fragment-cleanup-20261021', beforeImage: 'assets/results/case-6-037-ffbbda014502/before.png', afterImage: 'assets/results/case-6-037-ffbbda014502/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'wild-037-6e48e25237c4', title: 'wild-037', note: 'wild-037-6e48e25237c4 · Pixal3D', beforeImage: 'assets/results/wild-037-6e48e25237c4/before.png', afterImage: 'assets/results/wild-037-6e48e25237c4/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'case-5-inthewild-028-e01ed5b566b6', title: 'case5-028', note: 'case-5-inthewild-028-e01ed5b566b6 · Pixal3D', beforeImage: 'assets/results/case-5-inthewild-028-e01ed5b566b6/before.png', afterImage: 'assets/results/case-5-inthewild-028-e01ed5b566b6/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'case8-001-48cea2466131', title: 'case8-001', note: 'case8-001-48cea2466131 · Pixal3D', beforeImage: 'assets/results/case8-001-48cea2466131/before.jpg', afterImage: 'assets/results/case8-001-48cea2466131/after.png', camera: { theta: 174, phi: 82, radius: 105 } },
  { id: 'wild-065-498d4fb9d01c', title: 'wild-065', note: 'wild-065-498d4fb9d01c', beforeImage: 'assets/results/wild-065-498d4fb9d01c/before.png', afterImage: 'assets/results/wild-065-498d4fb9d01c/after.png', camera: { theta: 174, phi: 82, radius: 105 } }
];

const RESULT_METHODS = [
  { key: 'before', label: 'Original', accent: 'original' },
  { key: 'after', label: 'Gate3DGen', accent: 'ours' }
];

const BASELINE_CASES = [
  {
    id: '05', title: 'Case 4', degradation: 'handheld', note: 'Handheld occlusion · Pixal3D', camera: { theta: 174, phi: 85, radius: 105 },
    imageOverrides: {
      original: 'assets/baselines/05/original-cropped.webp',
      ours: 'assets/baselines/05/ours-cropped.webp'
    }
  },
  {
    id: 'case-6-038-7c37fd2fcc98', title: 'case6-038', degradation: 'uneven lighting / handheld', note: 'case-6-038-7c37fd2fcc98 · Pixal3D', camera: { theta: 174, phi: 82, radius: 105 },
    methodKeys: ['original', 'firered', 'ours'],
    imageOverrides: {
      original: 'assets/baselines/case-6-038-7c37fd2fcc98/original.png',
      firered: 'assets/baselines/case-6-038-7c37fd2fcc98/firered.png',
      ours: 'assets/baselines/case-6-038-7c37fd2fcc98/ours.png'
    }
  },
  {
    id: 'case-5-inthewild-025-2196f7af868c', title: 'case5-025', note: 'case-5-inthewild-025-2196f7af868c · Pixal3D', camera: { theta: 174, phi: 82, radius: 105 },
    imageOverrides: {
      original: 'assets/baselines/case-5-inthewild-025-2196f7af868c/original.png',
      firered: 'assets/baselines/case-5-inthewild-025-2196f7af868c/firered.png',
      ours: 'assets/baselines/case-5-inthewild-025-2196f7af868c/ours.png'
    }
  },
  {
    id: 'case-5-inthewild-009-0ced63a7a6f1', title: 'case5-009', note: 'case-5-inthewild-009-0ced63a7a6f1 · Pixal3D', camera: { theta: 174, phi: 82, radius: 105 },
    imageOverrides: {
      original: 'assets/baselines/case-5-inthewild-009-0ced63a7a6f1/original.png',
      firered: 'assets/baselines/case-5-inthewild-009-0ced63a7a6f1/firered.png',
      ours: 'assets/baselines/case-5-inthewild-009-0ced63a7a6f1/ours.png'
    }
  },
  {
    id: 'case-6-033-b0f2e1f5104f', title: 'case6-033', note: 'case-6-033-b0f2e1f5104f · Pixal3D',
    camera: { theta: 174, phi: 82, radius: 105 },
    imageOverrides: {
      original: 'assets/baselines/case-6-033-b0f2e1f5104f/original.png',
      firered: 'assets/baselines/case-6-033-b0f2e1f5104f/firered.png',
      ours: 'assets/baselines/case-6-033-b0f2e1f5104f/ours.png'
    }
  },
  {
    id: 'case05-aircraft-parallel-runway', title: 'Aircraft runway', degradation: 'motion blur',
    note: 'Parallel runway · Pixal3D; FireRed uses an existing restoration with a background edit',
    camera: { theta: 180, phi: 90, radius: 105 },
    imageOverrides: {
      original: 'assets/baselines/case05-aircraft-parallel-runway/original.png',
      firered: 'assets/baselines/case05-aircraft-parallel-runway/firered.png',
      ours: 'assets/baselines/case05-aircraft-parallel-runway/ours.png'
    }
  }
];

const BASELINE_METHODS = [
  { key: 'original', label: 'Original', accent: 'original' },
  { key: 'firered', label: 'FireRed-Image-Edit', accent: 'baseline' },
  { key: 'ours', label: 'Gate3DGen', accent: 'ours' }
];

// Keep the existing basket and the newly replaced runway aircraft last.
const groupedCases = window.GATE3DGEN_GROUPED_CASES || { results: [], comparison: [] };
RESULT_CASES.splice(RESULT_CASES.length - 1, 0, ...groupedCases.results);
BASELINE_CASES.splice(BASELINE_CASES.length - 1, 0, ...groupedCases.comparison);
RESULT_CASES.splice(RESULT_CASES.length - 1, 0, {
  id: 'case9-012-e5c50ea0a64f', title: 'case9-012', note: 'case9-012-e5c50ea0a64f · Pixal3D · GateB base seed 1234',
  backend: 'pixal3d', gatebSeeds: [1234], actualImageSeed: 1352648451,
  beforeImage: 'assets/results/case9-012-e5c50ea0a64f/before.jpg',
  afterImage: 'assets/results/case9-012-e5c50ea0a64f/after.png',
  camera: { theta: 180, phi: 90, radius: 105 }
});

// The approved order supersedes earlier basket/runway-last defaults.
if (window.GATE3DGEN_APPROVED_ORDER) {
  const order = window.GATE3DGEN_APPROVED_ORDER;
  function orderedCases(cases, ids) {
    const byId = new Map(cases.map(item => [item.id, item]));
    if (!Array.isArray(ids) || ids.length !== cases.length || new Set(ids).size !== ids.length || ids.some(id => !byId.has(id))) throw Error('Invalid approved case order');
    return ids.map(id => byId.get(id));
  }
  const results = orderedCases(RESULT_CASES, order.results);
  const comparison = orderedCases(BASELINE_CASES, order.comparison);
  RESULT_CASES.splice(0, RESULT_CASES.length, ...results);
  BASELINE_CASES.splice(0, BASELINE_CASES.length, ...comparison);
}

const START_THETA = -36;
const START_PHI = 85;
const START_RADIUS = 105;
const COARSE_POINTER = window.matchMedia('(pointer: coarse)').matches;
const DEFER_3D = window.matchMedia('(max-width: 900px)').matches || Boolean(navigator.connection?.saveData);
const mediaLoading = window.GATE3DGEN_MEDIA_LOADING;

function createComparison(config) {
  const state = {
    caseId: config.cases[0].id,
    imageZoom: 1,
    modelZoom: 1,
    loadToken: 0,
    interactive: false,
    hasInteracted: false,
    baseRadii: new WeakMap(),
    ready: new WeakSet(),
    cameraSource: null,
    mirrored: new WeakSet()
  };

  const caseControls = document.querySelector(config.caseControls);
  const caseCarousel = caseControls.closest('[data-case-carousel]');
  const caseScrollButtons = [...caseCarousel.querySelectorAll('[data-case-scroll]')];
  const grid = document.querySelector(config.grid);
  const imageZoomInput = document.querySelector(config.imageZoomInput);
  const modelZoomInput = document.querySelector(config.modelZoomInput);
  const zoomButtons = [...document.querySelectorAll(`[data-scope="${config.scope}"][data-zoom]`)];
  let initialized = false;
  let stageAbort;
  let headerAlignFrame = 0;
  let caseDragStartX = 0;
  let caseDragStartScroll = 0;
  let caseDragMoved = false;
  let caseDragPointerId = null;
  let suppressCaseClick = false;
  let caseResizeObserver;
  let caseHoverScrollFrame = 0;
  let caseHoverScrollLastTime = 0;
  let caseHoverScrollDirection = 0;
  let interactionHintFrame = 0;
  let interactionHintStart = 0;

  function alignResultHeaders() {
    if (config.kind !== 'result') return;
    cancelAnimationFrame(headerAlignFrame);
    headerAlignFrame = requestAnimationFrame(() => {
      const headers = [...grid.querySelectorAll('.result-method-card > header')];
      headers.forEach(header => header.style.removeProperty('min-height'));
      if (window.matchMedia('(max-width: 680px)').matches || !headers.length) return;
      const maxHeight = Math.ceil(Math.max(...headers.map(header => header.getBoundingClientRect().height)));
      headers.forEach(header => { header.style.minHeight = `${maxHeight}px`; });
    });
  }

  function selectedCase() {
    return config.cases.find(item => item.id === state.caseId);
  }

  function selectedMethods(item = selectedCase()) {
    if (!item?.methodKeys) return config.methods;
    return item.methodKeys
      .map(key => config.methods.find(method => method.key === key))
      .filter(Boolean);
  }

  function selectedCamera() {
    const camera = selectedCase()?.camera;
    return {
      theta: camera?.theta ?? START_THETA,
      phi: camera?.phi ?? START_PHI,
      radius: camera?.radius ?? START_RADIUS
    };
  }

  function mediaPath(item, method, mediaType) {
    if (config.kind === 'result') {
      if (mediaType === 'image') return method.key === 'before' ? item.beforeImage : item.afterImage;
      const path = `assets/models/results/${item.id}/${method.key}.glb${item.modelVersion ? `?v=${item.modelVersion}` : ''}`;
      return window.GATE3DGEN_ASSET_HOSTING?.url(path) ?? path;
    }
    if (mediaType === 'image' && item.imageOverrides?.[method.key]) return item.imageOverrides[method.key];
    if (mediaType === 'image') return `assets/baselines/${item.id}/${method.key}.webp`;
    const path = `assets/models/baselines/${item.id}/${method.key}.glb`;
    return window.GATE3DGEN_ASSET_HOSTING?.url(path) ?? path;
  }

  function selectorImagePath(item) {
    if (window.GATE3DGEN_THUMBNAILS?.[item.id]) return window.GATE3DGEN_THUMBNAILS[item.id];
    if (config.kind === 'result') return item.beforeImage;
    return item.imageOverrides?.original ?? `assets/baselines/${item.id}/original.webp`;
  }

  function updateCaseScrollButtons() {
    const maxScroll = Math.max(0, caseControls.scrollWidth - caseControls.clientWidth);
    caseScrollButtons.forEach(button => {
      const direction = Number(button.dataset.caseScroll);
      button.disabled = direction < 0 ? caseControls.scrollLeft <= 2 : caseControls.scrollLeft >= maxScroll - 2;
    });
  }

  function centerCaseButton(button, behavior = 'smooth') {
    if (caseControls.dataset.layout === 'compact') return;
    if (!button) return;
    const left = button.offsetLeft - (caseControls.clientWidth - button.offsetWidth) / 2;
    caseControls.scrollTo({ left: Math.max(0, left), behavior });
  }

  function posterPath(item, method) {
    return config.kind === 'result'
      ? `assets/posters/results/${item.id}/${method.key}.webp`
      : `assets/posters/baselines/${item.id}/${method.key}.webp`;
  }

  function allViewers() {
    return [...grid.querySelectorAll('model-viewer')];
  }

  function viewers() {
    return allViewers().filter(viewer => viewer.hasAttribute('src'));
  }

  function readyViewers() {
    return viewers().filter(viewer => state.ready.has(viewer));
  }

  function modelFrames(loadedOnly = false) {
    const frames = [...grid.querySelectorAll('.model-viewport')];
    return loadedOnly ? frames.filter(frame => state.ready.has(frame.querySelector('model-viewer[src]'))) : frames;
  }

  function clampZoom(value) {
    return Math.max(.65, Math.min(2.5, value));
  }

  function updateZoomReadout(mediaType) {
    const input = mediaType === 'image' ? imageZoomInput : modelZoomInput;
    const value = mediaType === 'image' ? state.imageZoom : state.modelZoom;
    if (!input) return;
    input.value = `${Math.round(value * 100)}%`;
  }

  function applyImageZoom() {
    grid.style.setProperty('--image-media-zoom', state.imageZoom.toFixed(2));
    updateZoomReadout('image');
  }

  function setControlMode(mode) {
    const interactive = mode === 'interactive';
    const imageEnabled = interactive || mode === 'deferred';
    zoomButtons.forEach(button => {
      button.disabled = button.dataset.media === 'image' ? !imageEnabled : !interactive;
    });
    if (imageZoomInput) imageZoomInput.disabled = !imageEnabled;
    if (modelZoomInput) modelZoomInput.disabled = !interactive;
  }

  function setOrbitDegrees({ theta, phi, radius }) {
    readyViewers().forEach(viewer => {
      state.mirrored.add(viewer);
      viewer.cameraOrbit = `${theta}deg ${phi}deg ${radius}%`;
      viewer.jumpCameraToGoal?.();
      requestAnimationFrame(() => requestAnimationFrame(() => state.mirrored.delete(viewer)));
    });
  }

  function stopInteractionHintMotion(settle = false) {
    cancelAnimationFrame(interactionHintFrame);
    interactionHintFrame = 0;
    interactionHintStart = 0;
    if (settle && state.interactive && !state.hasInteracted) setOrbitDegrees(selectedCamera());
  }

  function startInteractionHintMotion() {
    stopInteractionHintMotion(false);
    if (state.hasInteracted || !state.interactive || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    const camera = selectedCamera();
    const duration = 3400;
    interactionHintStart = performance.now();

    const animate = now => {
      if (state.hasInteracted || !state.interactive) {
        stopInteractionHintMotion(false);
        return;
      }

      const progress = Math.min(1, (now - interactionHintStart) / duration);
      const envelope = Math.sin(Math.PI * progress);
      const theta = camera.theta + Math.sin(progress * Math.PI * 4) * 4.5 * envelope;
      readyViewers().forEach(viewer => {
        viewer.cameraOrbit = `${theta}deg ${camera.phi}deg ${camera.radius}%`;
        viewer.jumpCameraToGoal?.();
      });

      if (progress < 1) interactionHintFrame = requestAnimationFrame(animate);
      else stopInteractionHintMotion(true);
    };

    interactionHintFrame = requestAnimationFrame(animate);
  }

  function markInteracted() {
    stopInteractionHintMotion(false);
    state.hasInteracted = true;
    modelFrames().forEach(frame => frame.classList.add('has-interacted'));
    allViewers().forEach(viewer => viewer.setAttribute('interaction-prompt', 'none'));
  }

  function unlockInteraction(viewer) {
    if (!viewer.loaded || state.ready.has(viewer)) return;
    const source = state.cameraSource;
    viewer.updateFraming?.();
    const camera = selectedCamera();
    viewer.cameraOrbit = `${camera.theta}deg ${camera.phi}deg ${camera.radius}%`;
    viewer.jumpCameraToGoal?.();
    const orbit = viewer.getCameraOrbit?.();
    if (orbit?.radius) state.baseRadii.set(viewer, orbit.radius);
    state.ready.add(viewer);
    if (source && source !== viewer && state.ready.has(source)) alignViewerToSource(viewer, source);
    state.interactive = true;
    const frame = viewer.closest('.model-viewport');
    if (frame) {
      frame.classList.remove('is-touring', 'is-loading');
      frame.classList.add('is-interactive');
      frame.classList.toggle('has-interacted', state.hasInteracted);
      const statusNode = frame.querySelector('.model-status');
      if (statusNode) {
        statusNode.textContent = '';
        statusNode.classList.remove('ready', 'error');
      }
      if (COARSE_POINTER) {
        frame.classList.add('needs-touch-activation');
        const touchToggle = frame.querySelector('.model-touch-toggle');
        if (touchToggle) {
          touchToggle.hidden = false;
          touchToggle.textContent = 'Interact with 3D';
        }
      }
    }
      if (COARSE_POINTER) viewer.removeAttribute('camera-controls');
      else viewer.setAttribute('camera-controls', '');
      viewer.setAttribute('interaction-prompt', 'none');
      viewer.setAttribute('disable-pan', '');
    setControlMode('interactive');
    requestAnimationFrame(startInteractionHintMotion);
  }

  function updateModelLoadProgress(viewer, value) {
    const progressNode = viewer.closest('.model-viewport')?.querySelector('.model-load-progress');
    if (!progressNode) return;
    const progress = Math.max(0, Math.min(1, Number(value) || 0));
    progressNode.style.setProperty('--model-load-progress', progress.toFixed(4));
    progressNode.setAttribute('aria-valuenow', String(Math.round(progress * 100)));
  }

  function waitForLoad(viewer, token) {
    if (viewer.loaded) {
      updateModelLoadProgress(viewer, 1);
      viewer.closest('.model-viewport')?.classList.remove('is-loading');
      return Promise.resolve(token === state.loadToken);
    }
    return new Promise(resolve => {
      const cleanup = () => {
        viewer.removeEventListener('load', handleLoad);
        viewer.removeEventListener('error', handleError);
      };
      const handleLoad = () => {
        updateModelLoadProgress(viewer, 1);
        viewer.closest('.model-viewport')?.classList.remove('is-loading');
        cleanup();
        resolve(token === state.loadToken);
      };
      const handleError = () => {
        cleanup();
        const frame = viewer.closest('.model-viewport');
        frame?.classList.remove('is-loading');
        const statusNode = frame?.querySelector('.model-status');
        if (statusNode) {
          statusNode.textContent = '3D load failed';
          statusNode.classList.add('error');
        }
        resolve(token === state.loadToken);
      };
      viewer.addEventListener('load', handleLoad, { once: true });
      viewer.addEventListener('error', handleError, { once: true });
    });
  }

  function setImageZoom(value) {
    if (!state.interactive) return;
    state.imageZoom = clampZoom(value);
    applyImageZoom();
  }

  function setModelZoom(value) {
    if (!state.interactive) return;
    state.modelZoom = clampZoom(value);
    updateZoomReadout('model');
    readyViewers().forEach(viewer => {
      const orbit = viewer.getCameraOrbit?.();
      const base = state.baseRadii.get(viewer);
      if (!orbit || !base) return;
      state.mirrored.add(viewer);
      viewer.cameraOrbit = `${orbit.theta}rad ${orbit.phi}rad ${base / state.modelZoom}m`;
      viewer.jumpCameraToGoal?.();
      requestAnimationFrame(() => requestAnimationFrame(() => state.mirrored.delete(viewer)));
    });
    markInteracted();
  }

  function resetModelView() {
    if (!state.interactive) return;
    state.modelZoom = 1;
    updateZoomReadout('model');
    setOrbitDegrees(selectedCamera());
    requestAnimationFrame(() => requestAnimationFrame(() => {
      readyViewers().forEach(viewer => {
        const orbit = viewer.getCameraOrbit?.();
        if (orbit?.radius) state.baseRadii.set(viewer, orbit.radius);
      });
    }));
  }

  function alignViewerToSource(viewer, source) {
    const orbit = source.getCameraOrbit?.();
    const sourceBase = state.baseRadii.get(source);
    const targetBase = state.baseRadii.get(viewer);
    if (!orbit || !sourceBase || !targetBase) return;
    state.mirrored.add(viewer);
    viewer.cameraOrbit = `${orbit.theta}rad ${orbit.phi}rad ${targetBase * orbit.radius / sourceBase}m`;
    viewer.jumpCameraToGoal?.();
    requestAnimationFrame(() => requestAnimationFrame(() => state.mirrored.delete(viewer)));
  }

  function mirrorCamera(source) {
    if (!state.interactive || !state.ready.has(source) || state.mirrored.has(source)) return;
    state.cameraSource = source;
    const orbit = source.getCameraOrbit?.();
    if (!orbit) return;
    const sourceBase = state.baseRadii.get(source) || orbit.radius;
    const radiusRatio = sourceBase ? orbit.radius / sourceBase : 1;
    state.modelZoom = clampZoom(1 / radiusRatio);
    updateZoomReadout('model');
    markInteracted();

    readyViewers().forEach(viewer => {
      if (viewer === source) return;
      alignViewerToSource(viewer, source);
    });
  }

  function methodTitleMarkup(_item, method) {
    return `<span class="method-title">${method.label}</span>`;
  }

  function modelViewerMarkup(modelPath, modelPoster, method, item) {
    const source = `data-src="${modelPath}"`;
    const loading = DEFER_3D ? 'lazy' : 'eager';
    const camera = item.camera ?? { theta: START_THETA, phi: START_PHI, radius: START_RADIUS };
    return `<model-viewer ${source} loading="${loading}" reveal="auto" interaction-prompt="none" disable-pan shadow-intensity="0.55" shadow-softness="0.9" tone-mapping="aces" camera-orbit="${camera.theta}deg ${camera.phi}deg ${camera.radius}%" min-camera-orbit="auto auto 55%" max-camera-orbit="auto auto 240%" alt="${method.label} interactive 3D reconstruction for ${item.title}"></model-viewer>`;
  }

  function modelActionsMarkup() {
    return `
      <button class="model-load-button" type="button" disabled ${DEFER_3D ? '' : 'hidden'}>Load interactive 3D</button>
      <button class="model-touch-toggle" type="button" hidden>Interact with 3D</button>`;
  }

  function cardMarkup(item, method) {
    const imagePath = mediaPath(item, method, 'image');
    const modelPath = mediaPath(item, method, 'model');
    const modelPoster = posterPath(item, method);
    if (config.kind === 'result') {
      const accentClass = method.accent === 'ours' ? 'ours-card' : '';
      return `
        <article class="media-card result-method-card ${accentClass}">
          <header>${methodTitleMarkup(item, method)}</header>
          <div class="result-media-pair">
            <section class="paired-media">
              <p>2D image</p>
              ${imageMarkup(imagePath, method, item)}
            </section>
            <section class="paired-media">
              <p>Interactive 3D</p>
              <div class="media-viewport model-viewport ${DEFER_3D ? 'has-load-action' : 'is-loading'}" tabindex="0">
                ${modelViewerMarkup(modelPath, modelPoster, method, item)}
                <img class="model-poster-preview" style="visibility:hidden" data-poster-src="${modelPoster}" alt="${method.label} 3D preview for ${item.title}">
                <span class="model-status"></span>
                <span class="model-load-progress" role="progressbar" aria-label="Loading interactive 3D" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i></i></span>
                <span class="model-hand-hint" aria-hidden="true"><b>☝</b><span>Drag to rotate<br>${COARSE_POINTER ? 'Pinch' : 'Scroll'} to zoom</span></span>
                ${modelActionsMarkup()}
              </div>
            </section>
          </div>
        </article>`;
    }
    return `
      <article class="media-card ${method.accent === 'ours' ? 'ours-card' : ''}">
        <header>${methodTitleMarkup(item, method)}</header>
        <div class="media-stack">
          <section class="paired-media">
            <p>2D image</p>
            ${imageMarkup(imagePath, method, item)}
          </section>
          <section class="paired-media">
            <p>Interactive 3D</p>
            <div class="media-viewport model-viewport ${DEFER_3D ? 'has-load-action' : 'is-loading'}" tabindex="0">
              ${modelViewerMarkup(modelPath, modelPoster, method, item)}
              <img class="model-poster-preview" style="visibility:hidden" data-poster-src="${modelPoster}" alt="${method.label} 3D preview for ${item.title}">
              <span class="model-status"></span>
              <span class="model-load-progress" role="progressbar" aria-label="Loading interactive 3D" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i></i></span>
              <span class="model-hand-hint" aria-hidden="true"><b>☝</b><span>Drag to rotate<br>${COARSE_POINTER ? 'Pinch' : 'Scroll'} to zoom</span></span>
              ${modelActionsMarkup()}
            </div>
          </section>
        </div>
      </article>`;
  }

  function imageMarkup(imagePath, method, item) {
    return `<div class="media-viewport image-viewport" tabindex="0">
      <img data-image-src="${imagePath}" data-image-state="waiting" style="visibility:hidden" aria-busy="true" decoding="async" draggable="false" alt="${method.label} 2D image for ${item.title}">
      <div class="image-loading-overlay">
        <span class="image-load-progress" role="progressbar" aria-label="Loading 2D image"><i></i></span>
        <span class="image-load-label" role="status">Loading 2D image…</span>
        <span class="image-load-error" role="alert">2D image could not be loaded.</span>
        <button type="button" class="image-load-retry">Retry</button>
      </div>
    </div>`;
  }

  function wireViewerInteractions() {
    allViewers().forEach(viewer => {
      viewer.addEventListener('progress', event => updateModelLoadProgress(viewer, event.detail?.totalProgress));
      viewer.addEventListener('camera-change', event => {
        if (event.detail?.source === 'user-interaction') mirrorCamera(viewer);
      });
      viewer.addEventListener('pointerdown', markInteracted);
      viewer.addEventListener('wheel', markInteracted, { passive: true });
      viewer.addEventListener('dblclick', resetModelView);
    });
  }

  function resetDeferredViewer(viewer) {
    state.ready.delete(viewer);
    state.baseRadii.delete(viewer);
    if (state.cameraSource === viewer) state.cameraSource = null;
    const frame = viewer.closest('.model-viewport');
    if (!frame) return;
    viewer.removeAttribute('src');
    viewer.removeAttribute('camera-controls');
    frame.classList.remove('is-touring', 'is-loading', 'is-interactive', 'needs-touch-activation', 'touch-active', 'has-interacted');
    frame.classList.add('has-load-action');
    const statusNode = frame.querySelector('.model-status');
    statusNode.textContent = '';
    statusNode.classList.remove('ready', 'error');
    updateModelLoadProgress(viewer, 0);
    const loadButton = frame.querySelector('.model-load-button');
    const touchToggle = frame.querySelector('.model-touch-toggle');
    if (loadButton) {
      loadButton.hidden = false;
      loadButton.textContent = 'Load interactive 3D';
    }
    if (touchToggle) touchToggle.hidden = true;
  }

  async function loadDeferredViewer(button) {
    if (grid.dataset.loadStage !== 'models' || button.disabled) return;
    const frame = button.closest('.model-viewport');
    const viewer = frame?.querySelector('model-viewer');
    if (!viewer?.dataset.src) return;
    state.loadToken += 1;
    const token = state.loadToken;
    if (DEFER_3D) {
      allViewers().forEach(candidate => {
        if (candidate !== viewer && candidate.hasAttribute('src')) resetDeferredViewer(candidate);
      });
    }
    state.interactive = false;
    frame.classList.remove('has-load-action', 'needs-touch-activation', 'touch-active');
    frame.classList.add('is-touring', 'is-loading');
    const statusNode = frame.querySelector('.model-status');
    statusNode.textContent = '';
    statusNode.classList.remove('ready', 'error');
    updateModelLoadProgress(viewer, 0);
    button.hidden = true;
    viewer.setAttribute('src', viewer.dataset.src);
    setControlMode('loading');
    await customElements.whenDefined('model-viewer');
    await waitForLoad(viewer, token);
    if (token !== state.loadToken) return;
    if (!viewer.loaded) {
      frame.classList.remove('is-touring', 'is-loading');
      frame.classList.add('has-load-action');
      button.hidden = false;
      button.textContent = 'Retry interactive 3D';
      state.interactive = true;
      setControlMode('deferred');
      return;
    }
    unlockInteraction(viewer);
  }

  function toggleTouchInteraction(button) {
    const frame = button.closest('.model-viewport');
    const viewer = frame?.querySelector('model-viewer[src]');
    if (!viewer) return;
    const activate = !frame.classList.contains('touch-active');
    modelFrames(true).forEach(candidate => {
      const candidateViewer = candidate.querySelector('model-viewer');
      candidate.classList.remove('touch-active');
      candidate.classList.add('needs-touch-activation');
      candidateViewer?.removeAttribute('camera-controls');
      const candidateToggle = candidate.querySelector('.model-touch-toggle');
      if (candidateToggle) candidateToggle.textContent = 'Interact with 3D';
    });
    if (activate) {
      frame.classList.add('touch-active');
      frame.classList.remove('needs-touch-activation');
      viewer.setAttribute('camera-controls', '');
      button.textContent = 'Done';
    }
  }

  async function render() {
    stageAbort?.abort();
    stageAbort = new AbortController();
    const signal = stageAbort.signal;
    allViewers().forEach(viewer => viewer.removeAttribute('src'));
    grid.querySelectorAll('img[src]').forEach(image => image.removeAttribute('src'));
    stopInteractionHintMotion(false);
    state.loadToken += 1;
    const token = state.loadToken;
    state.interactive = false;
    state.hasInteracted = false;
    state.ready = new WeakSet();
    state.baseRadii = new WeakMap();
    state.cameraSource = null;
    state.imageZoom = 1;
    state.modelZoom = 1;
    const item = selectedCase();
    const methods = selectedMethods(item);
    grid.style.setProperty('--comparison-method-count', String(methods.length));
    grid.innerHTML = methods.map(method => cardMarkup(item, method)).join('');
    alignResultHeaders();
    caseControls.querySelectorAll('button').forEach(button => {
      const active = button.dataset.caseId === state.caseId;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
    });
    centerCaseButton(caseControls.querySelector(`button[data-case-id="${state.caseId}"]`));
    setControlMode('loading');
    applyImageZoom();
    updateZoomReadout('model');
    wireViewerInteractions();

    grid.dataset.loadStage = 'thumbnails';
    await mediaLoading.waitForThumbnails();
    if (signal.aborted || token !== state.loadToken) return;
    grid.dataset.loadStage = 'images';
    const images = [...grid.querySelectorAll('img[data-image-src]')];
    images.forEach(image => { image.src = image.dataset.imageSrc; });
    const results = await Promise.all(images.map(async image => {
      const ok = await mediaLoading.waitForImage(image, signal);
      if (!signal.aborted && token === state.loadToken) {
        image.dataset.imageState = ok ? 'ready' : 'error';
        image.style.visibility = ok ? 'visible' : 'hidden';
        image.setAttribute('aria-busy', 'false');
      }
      return ok;
    }));
    if (signal.aborted || token !== state.loadToken) return;
    if (results.some(ok => !ok)) {
      grid.dataset.loadStage = 'image-error';
      grid.querySelectorAll('.model-status').forEach(node => {
        node.textContent = '2D image failed. Click this case to retry.';
        node.classList.add('error');
      });
      return;
    }
    await mediaLoading.afterPaint();
    if (signal.aborted || token !== state.loadToken) return;
    grid.dataset.loadStage = 'models';
    grid.querySelectorAll('img[data-poster-src]').forEach(async image => {
      image.src = image.dataset.posterSrc;
      const ok = await mediaLoading.waitForImage(image, signal);
      if (ok && !signal.aborted && token === state.loadToken) image.style.visibility = 'visible';
    });
    grid.querySelectorAll('.model-load-button').forEach(button => { button.disabled = false; });

    if (DEFER_3D) {
      state.interactive = true;
      setControlMode('deferred');
      return;
    }

    allViewers().forEach(viewer => viewer.setAttribute('src', viewer.dataset.src));
    await customElements.whenDefined('model-viewer');
    if (signal.aborted || token !== state.loadToken) return;
    const activeViewers = viewers();
    await Promise.all(activeViewers.map(async viewer => {
      const current = await waitForLoad(viewer, token);
      if (current && !signal.aborted && token === state.loadToken && viewer.loaded) unlockInteraction(viewer);
    }));
  }

  const galleryRows = config.kind === 'result' ? 3 : 2;
  const galleryColumns = Math.ceil(config.cases.length / galleryRows);
  caseControls.dataset.layout = 'compact';
  caseControls.style.setProperty('--case-rows', String(galleryRows));
  caseControls.style.setProperty('--case-columns', String(galleryColumns));
  config.cases.forEach(item => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'case-thumbnail';
    button.setAttribute('role', 'option');
    button.dataset.caseId = item.id;
    button.innerHTML = `
      <span class="case-thumbnail-image"><img src="${selectorImagePath(item)}" alt="" loading="eager" fetchpriority="high" decoding="async" draggable="false"></span>`;
    button.setAttribute('aria-label', `Show ${item.title}; ${item.note}`);
    const active = item.id === state.caseId;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
    caseControls.append(button);
  });
  mediaLoading.registerThumbnails(caseControls.querySelectorAll('img'));

  function stopCaseHoverScroll() {
    cancelAnimationFrame(caseHoverScrollFrame);
    caseHoverScrollFrame = 0;
    caseHoverScrollLastTime = 0;
    caseHoverScrollDirection = 0;
    caseControls.classList.remove('is-hover-scrolling');
  }

  function startCaseHoverScroll(button, pointerType = 'mouse') {
    if (pointerType !== 'mouse' || button.disabled) return;
    stopCaseHoverScroll();
    caseHoverScrollDirection = Number(button.dataset.caseScroll);
    caseHoverScrollLastTime = performance.now();
    caseControls.classList.add('is-hover-scrolling');

    const advance = now => {
      const maxScroll = Math.max(0, caseControls.scrollWidth - caseControls.clientWidth);
      if (button.disabled || !caseHoverScrollDirection) {
        stopCaseHoverScroll();
        return;
      }
      const elapsed = Math.min(32, now - caseHoverScrollLastTime);
      caseHoverScrollLastTime = now;
      caseControls.scrollLeft = Math.max(0, Math.min(maxScroll, caseControls.scrollLeft + caseHoverScrollDirection * elapsed * .48));
      updateCaseScrollButtons();
      const reachedEdge = caseHoverScrollDirection < 0 ? caseControls.scrollLeft <= 0 : caseControls.scrollLeft >= maxScroll - 1;
      if (reachedEdge) {
        stopCaseHoverScroll();
        return;
      }
      caseHoverScrollFrame = requestAnimationFrame(advance);
    };

    caseHoverScrollFrame = requestAnimationFrame(advance);
  }

  caseScrollButtons.forEach(button => {
    button.addEventListener('click', () => {
      const direction = Number(button.dataset.caseScroll);
      const maxScroll = Math.max(0, caseControls.scrollWidth - caseControls.clientWidth);
      const distance = Math.max(180, caseControls.clientWidth * .55);
      caseControls.scrollLeft = Math.max(0, Math.min(maxScroll, caseControls.scrollLeft + direction * distance));
      updateCaseScrollButtons();
    });
    button.addEventListener('pointerenter', event => startCaseHoverScroll(button, event.pointerType));
    button.addEventListener('pointerleave', stopCaseHoverScroll);
    button.addEventListener('pointercancel', stopCaseHoverScroll);
  });

  caseControls.addEventListener('scroll', updateCaseScrollButtons, { passive: true });
  if ('ResizeObserver' in window) {
    caseResizeObserver = new ResizeObserver(updateCaseScrollButtons);
    caseResizeObserver.observe(caseControls);
  } else {
    window.addEventListener('resize', updateCaseScrollButtons, { passive: true });
  }
  requestAnimationFrame(updateCaseScrollButtons);

  caseControls.addEventListener('pointerdown', event => {
    if (caseControls.dataset.layout === 'compact') return;
    if (event.pointerType !== 'mouse' || event.button !== 0) return;
    caseDragPointerId = event.pointerId;
    caseDragStartX = event.clientX;
    caseDragStartScroll = caseControls.scrollLeft;
    caseDragMoved = false;
  });

  caseControls.addEventListener('pointermove', event => {
    if (event.pointerId !== caseDragPointerId) return;
    const distance = event.clientX - caseDragStartX;
    if (!caseDragMoved && Math.abs(distance) > 5) {
      caseDragMoved = true;
      caseControls.classList.add('is-dragging');
      caseControls.setPointerCapture(event.pointerId);
    }
    if (!caseDragMoved) return;
    caseControls.scrollLeft = caseDragStartScroll - distance;
    event.preventDefault();
  });

  function finishCaseDrag(event) {
    if (event.pointerId !== caseDragPointerId) return;
    if (caseControls.hasPointerCapture(event.pointerId)) {
      caseControls.releasePointerCapture(event.pointerId);
    }
    caseDragPointerId = null;
    caseControls.classList.remove('is-dragging');
    if (!caseDragMoved) return;
    suppressCaseClick = true;
    requestAnimationFrame(() => { suppressCaseClick = false; });
  }

  caseControls.addEventListener('pointerup', finishCaseDrag);
  caseControls.addEventListener('pointercancel', finishCaseDrag);

  caseControls.addEventListener('click', event => {
    if (suppressCaseClick) {
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    const button = event.target.closest('button[data-case-id]');
    if (!button || (button.dataset.caseId === state.caseId && grid.dataset.loadStage !== 'image-error')) return;
    state.caseId = button.dataset.caseId;
    initialized = true;
    render();
  });

  caseControls.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const currentIndex = config.cases.findIndex(item => item.id === state.caseId);
    const columns = galleryColumns;
    const direction = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: columns, ArrowUp: -columns }[event.key];
    const nextIndex = Math.max(0, Math.min(config.cases.length - 1, currentIndex + direction));
    if (nextIndex === currentIndex) return;
    state.caseId = config.cases[nextIndex].id;
    initialized = true;
    render();
    caseControls.querySelector(`button[data-case-id="${state.caseId}"]`)?.focus({ preventScroll: true });
  });

  grid.addEventListener('click', event => {
    if (event.target.closest('.image-load-retry')) {
      render();
      return;
    }
    const loadButton = event.target.closest('.model-load-button');
    if (loadButton) {
      loadDeferredViewer(loadButton);
      return;
    }
    const touchToggle = event.target.closest('.model-touch-toggle');
    if (touchToggle) toggleTouchInteraction(touchToggle);
  });

  grid.addEventListener('wheel', event => {
    if (!event.target.closest('.image-viewport')) return;
    event.preventDefault();
    setImageZoom(state.imageZoom * (event.deltaY < 0 ? 1.12 : 1 / 1.12));
  }, { passive: false });

  grid.addEventListener('dblclick', event => {
    if (event.target.closest('.image-viewport')) setImageZoom(1);
  });

  zoomButtons.forEach(button => {
    button.addEventListener('click', () => {
      const mediaType = button.dataset.media;
      const current = mediaType === 'image' ? state.imageZoom : state.modelZoom;
      const next = button.dataset.zoom === 'in' ? current * 1.2 : current / 1.2;
      if (mediaType === 'image') setImageZoom(next);
      else setModelZoom(next);
    });
  });

  function commitZoomInput(input, mediaType) {
    const percentage = Number.parseFloat(input.value.replace('%', '').trim());
    if (!Number.isFinite(percentage)) {
      updateZoomReadout(mediaType);
      return;
    }
    if (mediaType === 'image') setImageZoom(percentage / 100);
    else setModelZoom(percentage / 100);
  }

  [[imageZoomInput, 'image'], [modelZoomInput, 'model']].filter(([input]) => input).forEach(([input, mediaType]) => {
    input.addEventListener('focus', () => input.select());
    input.addEventListener('change', () => commitZoomInput(input, mediaType));
    input.addEventListener('keydown', event => {
      if (event.key !== 'Enter') return;
      commitZoomInput(input, mediaType);
      input.blur();
    });
  });

  setControlMode('loading');
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      observer.disconnect();
      if (!initialized) {
        initialized = true;
        render();
      }
    }, { rootMargin: '240px 0px' });
    observer.observe(grid);
  } else {
    initialized = true;
    render();
  }
  if (config.kind === 'result') {
    window.addEventListener('resize', alignResultHeaders, { passive: true });
    document.fonts?.ready.then(alignResultHeaders);
  }
  return {
    stop() {
      stageAbort?.abort();
      cancelAnimationFrame(headerAlignFrame);
      stopInteractionHintMotion(false);
      stopCaseHoverScroll();
      caseResizeObserver?.disconnect();
      if (!caseResizeObserver) window.removeEventListener('resize', updateCaseScrollButtons);
      if (config.kind === 'result') window.removeEventListener('resize', alignResultHeaders);
    }
  };
}

const resultComparison = createComparison({
  kind: 'result',
  scope: 'result',
  cases: RESULT_CASES,
  methods: RESULT_METHODS,
  caseControls: '#result-case-controls',
  grid: '#result-media-grid',
  imageZoomInput: '#result-image-zoom',
  modelZoomInput: '#result-model-zoom'
});

const baselineComparison = createComparison({
  kind: 'baseline',
  scope: 'baseline',
  cases: BASELINE_CASES,
  methods: BASELINE_METHODS,
  caseControls: '#baseline-case-controls',
  grid: '#baseline-media-grid',
  imageZoomInput: '#baseline-image-zoom',
  modelZoomInput: '#baseline-model-zoom'
});

window.addEventListener('pagehide', () => {
  resultComparison.stop();
  baselineComparison.stop();
});

document.querySelector('#copy-citation')?.addEventListener('click', async event => {
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    event.currentTarget.textContent = 'Copied';
    setTimeout(() => { event.currentTarget.textContent = 'Copy BibTeX'; }, 1600);
  } catch {
    event.currentTarget.textContent = 'Select text to copy';
  }
});
