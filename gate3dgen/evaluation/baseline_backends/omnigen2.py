"""Official OmniGen2 image-edit backend for the TexVerse baseline.

This adapter is intentionally small and model-local.  It passes the supplied
prompt unchanged to the official ``OmniGen2Pipeline`` and fixes the output
contract to 1024x1024, BF16, 50 official edit steps, and a deterministic
per-case seed supplied by the shared evaluator.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()
DEFAULT_CODE_ROOT = WORKSPACE_ROOT / "models/third_party/OmniGen2"
DEFAULT_MODEL_ROOT = WORKSPACE_ROOT / "models/checkpoints/OmniGen2"
TARGET_SIZE = (1024, 1024)


@dataclass(frozen=True)
class OmniGen2Config:
    """Pinned generation settings; no prompt rewriting is performed."""

    model_path: Path = DEFAULT_MODEL_ROOT
    code_root: Path = DEFAULT_CODE_ROOT
    width: int = TARGET_SIZE[0]
    height: int = TARGET_SIZE[1]
    num_inference_steps: int = 50
    text_guidance_scale: float = 5.0
    image_guidance_scale: float = 2.0
    cfg_range_start: float = 0.0
    cfg_range_end: float = 1.0
    max_sequence_length: int = 1024
    negative_prompt: str = ""
    dtype: str = "bfloat16"
    prompt_rewrite: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": "omnigen2",
            "pipeline": "OmniGen2Pipeline",
            "model_path": str(Path(self.model_path).resolve()),
            "code_root": str(Path(self.code_root).resolve()),
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "text_guidance_scale": self.text_guidance_scale,
            "image_guidance_scale": self.image_guidance_scale,
            "cfg_range": [self.cfg_range_start, self.cfg_range_end],
            "max_sequence_length": self.max_sequence_length,
            "negative_prompt": self.negative_prompt,
            "dtype": self.dtype,
            "prompt_rewrite": self.prompt_rewrite,
            "seed_policy": "caller_supplied_stable_case_seed",
            "input_output_size": list(TARGET_SIZE),
            "local_files_only": True,
        }


def _ensure_code_root(code_root: Path) -> None:
    code_root = Path(code_root).resolve()
    if not (code_root / "omnigen2").is_dir():
        raise FileNotFoundError(f"official OmniGen2 code is missing: {code_root}")
    value = str(code_root)
    if value not in sys.path:
        sys.path.insert(0, value)


def _register_official_custom_modules() -> tuple[Any, Any]:
    """Bind official OmniGen2 classes before Diffusers reads ``model_index``.

    The local snapshot names its transformer and scheduler as top-level
    libraries (``transformer_omnigen2``, ``scheduling_flow_match_euler_discrete``).
    ``OmniGen2Pipeline.from_pretrained`` then copies those files into the shared
    Hugging Face ``diffusers_modules.local`` cache.  16/24/32 concurrent GPU
    workers race on that copy and fail with a missing
    ``OmniGen2Transformer2DModel`` or Triton's ``lineno is out of bounds``.
    Import the packaged modules once, alias the model-index names, and pass the
    already-loaded objects into ``from_pretrained`` so the cache is never used.
    """

    import importlib

    transformer_module = importlib.import_module(
        "omnigen2.models.transformers.transformer_omnigen2"
    )
    scheduler_module = importlib.import_module(
        "omnigen2.schedulers.scheduling_flow_match_euler_discrete"
    )
    sys.modules["transformer_omnigen2"] = transformer_module
    sys.modules["scheduling_flow_match_euler_discrete"] = scheduler_module
    transformer_cls = getattr(transformer_module, "OmniGen2Transformer2DModel", None)
    scheduler_cls = getattr(scheduler_module, "FlowMatchEulerDiscreteScheduler", None)
    if transformer_cls is None or scheduler_cls is None:
        raise ImportError("official OmniGen2 transformer/scheduler classes are missing")
    return transformer_cls, scheduler_cls


class OmniGen2Backend:
    """Thin runtime wrapper around the official OmniGen2 pipeline."""

    def __init__(self, config: OmniGen2Config | None = None, *, device: str = "cuda") -> None:
        import torch

        self.config = config or OmniGen2Config()
        self.device = device
        _ensure_code_root(self.config.code_root)
        model_path = Path(self.config.model_path).resolve()
        if not (model_path / "model_index.json").is_file():
            raise FileNotFoundError(
                f"OmniGen2 checkpoint is incomplete (missing model_index.json): {model_path}"
            )
        transformer_cls, scheduler_cls = _register_official_custom_modules()
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

        dtype = torch.bfloat16
        transformer = transformer_cls.from_pretrained(
            str(model_path), subfolder="transformer", torch_dtype=dtype, local_files_only=True
        )
        scheduler = scheduler_cls.from_pretrained(
            str(model_path), subfolder="scheduler", local_files_only=True
        )
        self.pipeline = OmniGen2Pipeline.from_pretrained(
            str(model_path),
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=dtype,
            local_files_only=True,
        )
        self.pipeline = self.pipeline.to(device)
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(disable=True)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.config.as_dict()

    def edit(self, image: Any, prompt: str, seed: int) -> Any:
        """Edit one RGB PIL image using the unchanged prompt and fixed seed."""

        import torch

        if tuple(image.size) != TARGET_SIZE:
            raise ValueError(f"OmniGen2 baseline requires input size {TARGET_SIZE}, got {image.size}")
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.inference_mode():
            output = self.pipeline(
                prompt=str(prompt),
                input_images=[image],
                width=self.config.width,
                height=self.config.height,
                num_inference_steps=self.config.num_inference_steps,
                max_sequence_length=self.config.max_sequence_length,
                text_guidance_scale=self.config.text_guidance_scale,
                image_guidance_scale=self.config.image_guidance_scale,
                cfg_range=(self.config.cfg_range_start, self.config.cfg_range_end),
                negative_prompt=self.config.negative_prompt,
                num_images_per_prompt=1,
                generator=generator,
                output_type="pil",
            )
        images = getattr(output, "images", None)
        if not images:
            raise RuntimeError("OmniGen2Pipeline returned no images")
        restored = images[0].convert("RGB")
        if tuple(restored.size) != TARGET_SIZE:
            raise RuntimeError(f"OmniGen2 returned non-contract size {restored.size}; no resize applied")
        return restored

    def close(self) -> None:
        self.pipeline = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def load_backend(
    model_path: str | Path = DEFAULT_MODEL_ROOT,
    *,
    code_root: str | Path = DEFAULT_CODE_ROOT,
    device: str = "cuda",
    **overrides: Any,
) -> OmniGen2Backend:
    """Public factory used by the common baseline adapter."""

    config = OmniGen2Config(model_path=Path(model_path), code_root=Path(code_root), **overrides)
    return OmniGen2Backend(config, device=device)


def edit_image(backend: OmniGen2Backend, image: Any, prompt: str, seed: int) -> Any:
    """Function-style interface for adapters that do not retain a method."""

    return backend.edit(image, prompt, seed)


def smoke_import(code_root: str | Path = DEFAULT_CODE_ROOT) -> dict[str, Any]:
    """Import-only check that does not load weights or allocate a GPU model."""

    _ensure_code_root(Path(code_root))
    transformer_cls, scheduler_cls = _register_official_custom_modules()
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

    return {
        "status": "import_ok",
        "pipeline": OmniGen2Pipeline.__name__,
        "transformer": transformer_cls.__name__,
        "scheduler": scheduler_cls.__name__,
    }


__all__ = [
    "OmniGen2Backend",
    "OmniGen2Config",
    "edit_image",
    "load_backend",
    "smoke_import",
]
