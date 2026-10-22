"""FD-DINOv2 feature and distance helpers from the original wild evaluation implementation."""
from __future__ import annotations
from typing import Any, Sequence
from pathlib import Path
import numpy as np
DEFAULT_DINO_MODEL="facebook/dinov2-large"
_IMAGENET_MEAN=(0.485,0.456,0.406)
_IMAGENET_STD=(0.229,0.224,0.225)

class LazyDinoV2Provider:
    """Lazy DINOv2 CLS-token feature extractor with two backends.

    * ``transformers`` — ``facebook/dinov2-large`` via HF AutoModel.
    * ``hub`` — official ``facebookresearch/dinov2`` loaded from a *local* hub
      directory (``source='local'``), e.g. ``dinov2_vitl14_reg``.  Matches the
      TRELLIS eval protocol.  Local ``weights`` (.pth) are loaded explicitly so
      no network download is required.

    Construction never downloads.  Features are cached by resolved path so
    repeated input photographs are encoded once.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DINO_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
        *,
        backend: str = "transformers",
        hub_dir: str | None = None,
        hub_model: str = "dinov2_vitl14_reg",
        weights: str | None = None,
        image_size: int = 224,
    ):
        self.backend = backend
        self.model_name = model_name if backend == "transformers" else f"hub:{hub_model}"
        self.device = device
        self.batch_size = int(batch_size)
        self.hub_dir = hub_dir
        self.hub_model = hub_model
        self.weights = weights
        self.image_size = int(image_size)
        self.model: Any | None = None
        self.processor: Any | None = None
        self.reason: str | None = None
        self._cache: dict[str, np.ndarray] = {}

    def _load(self) -> None:
        if self.model is not None or self.reason is not None:
            return
        try:
            if self.backend == "hub":
                self._load_hub()
            else:
                from transformers import AutoImageProcessor, AutoModel

                self.processor = AutoImageProcessor.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).eval().to(self.device)
        except Exception as error:  # pragma: no cover - environment dependent
            self.reason = f"dinov2_provider_unavailable:{type(error).__name__}:{error}"

    def _load_hub(self) -> None:  # pragma: no cover - needs weights
        import torch

        if not self.hub_dir or not Path(self.hub_dir).is_dir():
            raise FileNotFoundError(f"dinov2 hub dir not found: {self.hub_dir}")
        if not self.weights or not Path(self.weights).is_file():
            raise FileNotFoundError("Explicit local DINOv2 register weights are required")
        model = torch.hub.load(self.hub_dir, self.hub_model, source="local",
                               pretrained=False)
        if self.weights:
            state = torch.load(self.weights, map_location="cpu")
            if isinstance(state, dict) and "model" in state and not any(k.startswith("blocks") for k in state):
                state = state["model"]
            model.load_state_dict(state, strict=True)
        self.model = model.eval().to(self.device)

    @property
    def available(self) -> bool:
        self._load()
        return self.model is not None

    def _hub_batch(self, images: list[Any]) -> np.ndarray:  # pragma: no cover - needs weights
        import torch

        size = self.image_size
        tensors = []
        for image in images:
            resized = image.resize((size, size))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            array = (array - np.array(_IMAGENET_MEAN, dtype=np.float32)) / np.array(_IMAGENET_STD, dtype=np.float32)
            tensors.append(torch.from_numpy(array.transpose(2, 0, 1)))
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            cls = self.model(batch)
        return cls.detach().cpu().numpy().astype(np.float64)

    def _transformers_batch(self, images: list[Any]) -> np.ndarray:
        import torch

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            cls = self.model(**inputs).last_hidden_state[:, 0]
        return cls.detach().cpu().numpy().astype(np.float64)

    def features(self, paths: Sequence[Path]) -> np.ndarray:
        """Return CLS features ``[N, D]`` for the given image paths."""
        self._load()
        if self.model is None:
            raise RuntimeError(self.reason or "dinov2 provider unavailable")
        from PIL import Image

        ordered = [Path(path) for path in paths]
        pending = [path for path in ordered if str(path.resolve()) not in self._cache]
        for start in range(0, len(pending), self.batch_size):
            batch_paths = pending[start : start + self.batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as handle:
                    images.append(handle.convert("RGB"))
            if self.backend == "hub":
                vectors = self._hub_batch(images)
            else:
                vectors = self._transformers_batch(images)
            for path, vector in zip(batch_paths, vectors):
                vector = np.asarray(vector, dtype=np.float64).reshape(-1)
                norm = float(np.linalg.norm(vector))
                if norm > 1e-12:
                    vector = vector / norm
                self._cache[str(path.resolve())] = vector
        return np.stack([self._cache[str(path.resolve())] for path in ordered], axis=0)

def frechet_distance(generated: np.ndarray, real: np.ndarray, *, eps: float = 1e-6) -> float:
    """Fréchet Distance between two Gaussian-approximated feature sets.

    ``generated`` and ``real`` are ``[N, D]`` / ``[M, D]``.  Uses the
    pytorch-fid / official FID stabilization (diagonal ``eps``, discard tiny
    imaginary components from ``sqrtm``, clamp at zero).
    """

    from scipy import linalg

    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    if generated.ndim != 2 or real.ndim != 2 or generated.shape[1] != real.shape[1]:
        raise ValueError("feature sets must be 2-D with a shared dimension")
    if generated.shape[0] < 2 or real.shape[0] < 2:
        raise ValueError("each feature set needs at least two samples for a covariance")
    if not np.isfinite(generated).all() or not np.isfinite(real).all():
        raise ValueError("Feature sets must be finite")

    mu_gen, mu_real = generated.mean(axis=0), real.mean(axis=0)
    sigma_gen = np.atleast_2d(np.cov(generated, rowvar=False))
    sigma_real = np.atleast_2d(np.cov(real, rowvar=False))
    diff = mu_gen - mu_real

    offset = np.eye(sigma_gen.shape[0], dtype=np.float64) * eps
    covmean = linalg.sqrtm((sigma_gen + offset) @ (sigma_real + offset))
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma_gen + offset) @ (sigma_real + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(covmean.imag, 0, atol=1e-3):
            raise ValueError("Covariance square root has a significant imaginary component")
        covmean = np.real(covmean)
    value = float(diff @ diff + np.trace(sigma_gen + sigma_real - 2.0 * covmean))
    if not np.isfinite(value):
        raise ValueError(f"non-finite frechet distance: {value}")
    return max(0.0, value)
