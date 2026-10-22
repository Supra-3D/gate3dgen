"""Offline-compatible RMBG/BiRefNet adapter used by the 3D wrappers.

The public Pixal3D and TRELLIS.2 repositories ship a very small ``BiRefNet``
wrapper which assumes the RMBG-1.4 output contract.  RMBG-2.0 uses the same
Hugging Face entry point, but returns a list of *logits* at evaluation time;
RMBG-1.4 returns ``([already-sigmoided masks], features)``.  Keeping the
model loading and output normalization here avoids subtly different alpha
matte behavior in the two generators.

This module intentionally never downloads a checkpoint.  ``local_files_only``
is always passed to Transformers and local paths are validated before loading.
The local RMBG-2.0 remote-code snapshot imports only three small utilities
from timm and one from kornia.  Some evaluation images run in a minimal
environment without those optional packages, so a narrowly scoped compatibility
module is installed only when the real package is unavailable.  It is not a
replacement for either package and does not affect environments where they are
installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

from PIL import Image


# Both official Pixal3D and TRELLIS.2 preprocessors define foreground as
# alpha > 0.8 * 255 before reducing the foreground coordinates to a crop.
_OFFICIAL_FOREGROUND_ALPHA_THRESHOLD = int(0.8 * 255)


def _ensure_crop_safe_alpha(alpha: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    """Use a full-frame mask when RMBG leaves no crop-eligible foreground."""

    alpha = alpha.convert("L").resize(image_size, Image.Resampling.BILINEAR)
    extrema = alpha.getextrema()
    if extrema is not None and extrema[1] > _OFFICIAL_FOREGROUND_ALPHA_THRESHOLD:
        return alpha
    warnings.warn(
        "RMBG produced no foreground above the official crop threshold; "
        "falling back to a full-image alpha mask",
        RuntimeWarning,
        stacklevel=2,
    )
    return Image.new("L", image_size, 255)


def _default_rmbg2_model() -> Path:
    configured = os.environ.get("GATE3DGEN_RMBG_MODEL")
    if configured:
        return Path(configured).expanduser()
    project_root = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).expanduser().resolve()
    return project_root / "checkpoints/RMBG-2.0"


DEFAULT_RMBG2_MODEL = _default_rmbg2_model()


def _module_spec(name: str, *, package: bool = False):
    """Return a synthetic import spec for a compatibility module."""

    from importlib.machinery import ModuleSpec

    return ModuleSpec(name, loader=None, is_package=package)


def _install_optional_compat_modules() -> None:
    """Install only the tiny APIs used by the bundled RMBG remote code.

    ``transformers.dynamic_module_utils.check_imports`` checks package
    discoverability before importing the remote source.  Populating
    ``sys.modules`` and giving each module a valid package spec lets the
    completely local RMBG2 source run in the lean evaluation environment.
    The real packages always win and are left untouched.
    """

    if importlib.util.find_spec("timm") is None:
        import torch
        import torch.nn as nn

        timm = ModuleType("timm")
        timm.__path__ = []  # type: ignore[attr-defined]
        timm.__spec__ = _module_spec("timm", package=True)
        models = ModuleType("timm.models")
        models.__path__ = []  # type: ignore[attr-defined]
        models.__spec__ = _module_spec("timm.models", package=True)
        layers = ModuleType("timm.models.layers")
        layers.__spec__ = _module_spec("timm.models.layers")
        registry = ModuleType("timm.models.registry")
        registry.__spec__ = _module_spec("timm.models.registry")

        def to_2tuple(value):
            if isinstance(value, tuple):
                return value
            if isinstance(value, list):
                return tuple(value)
            return (value, value)

        def trunc_normal_(tensor, *args, **kwargs):
            return torch.nn.init.trunc_normal_(tensor, *args, **kwargs)

        class DropPath(nn.Module):
            # The segmenter is always evaluated with ``eval()``.  Keeping the
            # training implementation here also makes the compatibility layer
            # safe if a caller explicitly uses the model for a smoke train step.
            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x):
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
                return x.div(keep) * random_tensor.floor()

        def register_model(fn):
            return fn

        layers.DropPath = DropPath
        layers.to_2tuple = to_2tuple
        layers.trunc_normal_ = trunc_normal_
        registry.register_model = register_model
        models.layers = layers
        models.registry = registry
        timm.models = models
        sys.modules.update(
            {
                "timm": timm,
                "timm.models": models,
                "timm.models.layers": layers,
                "timm.models.registry": registry,
            }
        )

    if importlib.util.find_spec("kornia") is None:
        import torch
        import torch.nn.functional as F

        kornia = ModuleType("kornia")
        kornia.__path__ = []  # type: ignore[attr-defined]
        kornia.__spec__ = _module_spec("kornia", package=True)
        filters = ModuleType("kornia.filters")
        filters.__spec__ = _module_spec("kornia.filters")

        def laplacian(input_tensor, kernel_size: int = 5, border_type: str = "reflect", normalized: bool = False):
            # This branch is only used by the remote model while training;
            # preserve the documented shape and a conventional Laplacian for
            # callers that exercise it in a smoke test.
            size = int(kernel_size)
            if size < 3 or size % 2 == 0:
                raise ValueError("kernel_size must be an odd integer >= 3")
            radius = size // 2
            kernel = torch.zeros((1, 1, size, size), dtype=input_tensor.dtype, device=input_tensor.device)
            kernel[:, :, radius, radius] = float(size * size - 1)
            for index in range(size):
                if index != radius:
                    kernel[:, :, radius, index] = -1.0
                    kernel[:, :, index, radius] = -1.0
            if normalized:
                kernel = kernel / float(size * size - 1)
            channels = input_tensor.shape[1]
            kernel = kernel.expand(channels, 1, size, size)
            padding_mode = "reflect" if border_type == "reflect" else "replicate"
            padded = F.pad(input_tensor, (radius,) * 4, mode=padding_mode)
            return F.conv2d(padded, kernel, groups=channels)

        filters.laplacian = laplacian
        kornia.filters = filters
        sys.modules.update({"kornia": kornia, "kornia.filters": filters})


def ensure_rmbg_compat_modules() -> None:
    """Make the bundled RMBG remote-code imports available offline.

    The 3D generators call :class:`RMBGAdapter`, which installs this
    compatibility layer as part of model loading.  The 2D mask evaluator
    loads the same local remote-code checkpoint directly through
    ``AutoModelForImageSegmentation``; expose the setup as a small public
    helper so that path gets the identical offline behavior before
    Transformers inspects the remote module imports.
    """

    _install_optional_compat_modules()


def _safe_model_path(model_name: str | Path) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    """Return a Transformers-safe path for local remote-code snapshots.

    Transformers 4.35 derives a Python package name from the final path
    component.  ``RMBG-2.0`` therefore becomes the invalid module name
    ``transformers_modules.RMBG-2``.  A temporary, sanitized symlink fixes the
    import without copying the 5 GB checkpoint or modifying user data.
    """

    candidate = Path(model_name).expanduser()
    if not candidate.is_dir():
        return str(model_name), None
    if candidate.name.replace("_", "").replace("-", "").replace(".", "").isalnum() and all(
        character.isalnum() or character == "_" for character in candidate.name
    ):
        return str(candidate), None
    temporary = tempfile.TemporaryDirectory(prefix="texverse_rmbg2_")
    safe = Path(temporary.name) / "rmbg_model"
    safe.symlink_to(candidate, target_is_directory=True)
    return str(safe), temporary


def _first_tensor(value: Any) -> Any:
    """Recursively locate the last tensor in a nested model output."""

    import torch

    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("logits", "predictions", "pred_masks", "output"):
            if key in value:
                result = _first_tensor(value[key])
                if result is not None:
                    return result
        for item in reversed(tuple(value.values())):
            result = _first_tensor(item)
            if result is not None:
                return result
        return None
    if isinstance(value, (tuple, list)):
        for item in reversed(value):
            result = _first_tensor(item)
            if result is not None:
                return result
        return None
    for key in ("logits", "predictions", "pred_masks", "output"):
        result = getattr(value, key, None)
        if result is not None:
            result = _first_tensor(result)
            if result is not None:
                return result
    return None


def _first_tensor_in_order(value: Any) -> Any:
    """Locate the first tensor in a nested sequence.

    RMBG-1.4 labels its first multi-scale prediction ``d1`` and the existing
    repository wrappers intentionally use that entry.  Keep that ordering
    separate from :func:`_first_tensor`, whose reverse traversal selects the
    full-resolution final prediction from RMBG-2.0's output list.
    """

    import torch

    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            result = _first_tensor_in_order(item)
            if result is not None:
                return result
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            result = _first_tensor_in_order(item)
            if result is not None:
                return result
        return None
    for key in ("logits", "predictions", "pred_masks", "output"):
        result = getattr(value, key, None)
        if result is not None:
            result = _first_tensor_in_order(result)
            if result is not None:
                return result
    return None


def _extract_prediction(result: Any):
    """Extract a mask tensor and whether it is already a probability.

    RMBG-1.4: ``([sigmoid(d1), ...], [features])``.  RMBG-2.0 in eval mode:
    ``[m4_logits, m3_logits, m2_logits, p1_logits]``.  A direct tensor or a
    modern ModelOutput is treated as logits, which is the safer contract for
    current BiRefNet releases.
    """

    import torch

    if isinstance(result, (tuple, list)) and result and isinstance(result[0], (tuple, list)):
        # The first branch is the historical RMBG-1.4 multi-scale mask list.
        prediction = _first_tensor_in_order(result[0])
        if prediction is None:
            raise TypeError("RMBG nested output did not contain a tensor")
        # _first_tensor returns the last scale; all scales are probabilities,
        # and selecting the final one gives the highest spatial resolution.
        return prediction, True
    prediction = _first_tensor(result)
    if prediction is None or not torch.is_tensor(prediction):
        raise TypeError(f"unsupported RMBG output type: {type(result)!r}")
    return prediction, False


class RMBGAdapter:
    """Callable alpha-matte predictor with a BiRefNet-compatible ``to/cpu`` API."""

    def __init__(self, model: Any, model_name: str | Path, device: str = "cuda"):
        import torch
        from torchvision import transforms

        self.model = model.eval()
        self.model_name = str(model_name)
        self.device = str(device)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.to(self.device)

    @classmethod
    def from_pretrained(cls, model_name: str | Path, device: str = "cuda") -> "RMBGAdapter":
        import torch
        from transformers import AutoModelForImageSegmentation

        resolved = Path(model_name).expanduser()
        if resolved.exists() and not resolved.is_dir():
            raise ValueError(f"RMBG model path is not a directory: {resolved}")
        if resolved.is_dir():
            required = (resolved / "config.json",)
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"incomplete local RMBG snapshot: {', '.join(missing)}")
        _install_optional_compat_modules()
        safe_path, temporary = _safe_model_path(model_name)
        try:
            # local_files_only is intentional even for model ids: wrappers are
            # used in an offline evaluation and must fail loudly if a snapshot
            # is absent instead of making a hidden network request.
            model = AutoModelForImageSegmentation.from_pretrained(
                safe_path,
                trust_remote_code=True,
                local_files_only=True,
            )
        finally:
            if temporary is not None:
                temporary.cleanup()
        model.eval()
        return cls(model, model_name, device=device)

    def to(self, device: str):
        self.device = str(device)
        self.model.to(self.device)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def __call__(self, image: Image.Image) -> Image.Image:
        import torch
        from torchvision import transforms

        image_size = image.size
        tensor = self.transform_image(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            result = self.model(tensor)
        prediction, already_probability = _extract_prediction(result)
        prediction = prediction.detach().float().cpu()
        if prediction.ndim == 4:
            prediction = prediction[0, 0]
        elif prediction.ndim == 3:
            prediction = prediction[0]
        elif prediction.ndim != 2:
            raise ValueError(f"RMBG prediction has unsupported shape {tuple(prediction.shape)}")
        if not already_probability:
            prediction = prediction.sigmoid()
        prediction = prediction.clamp(0.0, 1.0)
        alpha = _ensure_crop_safe_alpha(
            transforms.ToPILImage()(prediction),
            image_size,
        )
        output = image.convert("RGB")
        output.putalpha(alpha)
        return output


def patch_birefnet_class(birefnet_class: type, model_name: str | Path, device: str = "cuda") -> type:
    """Patch a repository ``BiRefNet`` class to use :class:`RMBGAdapter`.

    The patch is deliberately class-local (the wrapper imports the class from
    its own repository) and replaces the original constructor as well as
    ``__call__``.  Thus an override cannot accidentally instantiate a second
    checkpoint through the repository's old RMBG-1.4 assumptions.
    """

    configured_model_name = model_name

    def _init(self, model_name: str | Path | None = None, *args, **kwargs):
        # The repository pipeline always passes the checkpoint baked into its
        # pipeline.json (often an unavailable absolute cache path).  This
        # class patch represents an explicit CLI override, so the configured
        # model must win even when the old constructor argument is non-empty.
        # Accept and ignore extra repository-specific constructor arguments to
        # keep this adapter usable across Pixal3D and TRELLIS.2 releases.
        self._rmbg_adapter = RMBGAdapter.from_pretrained(configured_model_name, device=device)
        self.model = self._rmbg_adapter.model
        self.transform_image = self._rmbg_adapter.transform_image

    def _to(self, target: str):
        self._rmbg_adapter.to(str(target))
        return self

    def _cuda(self):
        self._rmbg_adapter.cuda()
        return self

    def _cpu(self):
        self._rmbg_adapter.cpu()
        return self

    def _call(self, image: Image.Image):
        return self._rmbg_adapter(image)

    birefnet_class.__init__ = _init  # type: ignore[method-assign]
    birefnet_class.to = _to  # type: ignore[method-assign]
    birefnet_class.cuda = _cuda  # type: ignore[method-assign]
    birefnet_class.cpu = _cpu  # type: ignore[method-assign]
    birefnet_class.__call__ = _call  # type: ignore[method-assign]
    return birefnet_class


def segment_images(
    model_name: str | Path,
    images: Iterable[Image.Image],
    *,
    device: str = "cuda",
) -> list[Image.Image]:
    """Convenience API used by independent single-case mask verification."""

    adapter = RMBGAdapter.from_pretrained(model_name, device=device)
    return [adapter(image) for image in images]
