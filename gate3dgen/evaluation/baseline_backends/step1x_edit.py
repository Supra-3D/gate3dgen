"""Step1X-Edit-v1p2 backend with reasoning/reflection disabled.

The upstream v1p2 pipeline can optionally invoke a separate ``Thinker`` model
to rewrite the prompt and reflectively retry the edit.  Those modes are
explicitly disabled here so the universal-v4 restoration instruction reaches
the editor unchanged.  The benchmark caller supplies one stable seed for each
case, and this backend pins the official v1p2 50-step/bfloat16 configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


MODEL_REPO_ID = "stepfun-ai/Step1X-Edit-v1p2"
MODEL_VARIANT = "v1p2"
DEFAULT_STEPS = 50
DEFAULT_TRUE_CFG_SCALE = 6.0
DEFAULT_RESOLUTION = (1024, 1024)

_PROJECT_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()
_WORKSPACE_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()
_MODEL_CANDIDATES = (
    _WORKSPACE_ROOT / "models/third_party/Step1X-Edit-v1p2",
    _PROJECT_ROOT / "models/third_party/Step1X-Edit-v1p2",
)
DEFAULT_MODEL_ROOT = next((p for p in _MODEL_CANDIDATES if p.is_dir()), _MODEL_CANDIDATES[0])


class Step1XEditContractError(ValueError):
    """Raised when an editor call would violate the baseline contract."""


@dataclass(frozen=True)
class Step1XEditConfig:
    """Immutable runtime settings recorded in baseline provenance."""

    steps: int = DEFAULT_STEPS
    true_cfg_scale: float = DEFAULT_TRUE_CFG_SCALE
    width: int = DEFAULT_RESOLUTION[0]
    height: int = DEFAULT_RESOLUTION[1]
    dtype: str = "bfloat16"
    enable_thinking_mode: bool = False
    enable_reflection_mode: bool = False
    prompt_rewrite: bool = False


def _require_rgb_1024(image: Any) -> Any:
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"Step1X backend expects PIL.Image.Image, got {type(image)!r}")
    if image.size != DEFAULT_RESOLUTION:
        raise Step1XEditContractError(
            f"Step1X baseline requires an exact 1024x1024 pixel grid, got {image.size}"
        )
    return image.convert("RGB")


@dataclass
class Step1XEditV1P2Backend:
    """Loaded v1p2 editor with a common baseline-friendly API."""

    pipeline: Any
    device: str = "cuda"
    config: Step1XEditConfig = Step1XEditConfig()

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        device: str = "cuda",
        config: Step1XEditConfig | None = None,
    ) -> "Step1XEditV1P2Backend":
        """Load the local Step1X-Edit-v1p2 Diffusers checkpoint in BF16."""

        import torch

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Step1X model directory is missing: {model_path}")
        if not (model_path / "model_index.json").is_file():
            raise Step1XEditContractError(
                f"Step1X model directory has no model_index.json: {model_path}"
            )
        try:
            from diffusers import Step1XEditPipelineV1P2
        except ImportError as error:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Step1XEditPipelineV1P2 is unavailable; install the official v1p2 Diffusers branch"
            ) from error
        pipeline = Step1XEditPipelineV1P2.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipeline.to(device)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Step1X v1p2 baseline requires CUDA bfloat16 support")
        return cls(pipeline=pipeline, device=device, config=config or Step1XEditConfig())

    @property
    def metadata(self) -> dict[str, Any]:
        """Return stable settings for a per-run provenance record."""

        return {
            "backend": "step1x_edit",
            "model_repo_id": MODEL_REPO_ID,
            "model_variant": MODEL_VARIANT,
            "steps": self.config.steps,
            "true_cfg_scale": self.config.true_cfg_scale,
            "dtype": self.config.dtype,
            "width": self.config.width,
            "height": self.config.height,
            "enable_thinking_mode": self.config.enable_thinking_mode,
            "enable_reflection_mode": self.config.enable_reflection_mode,
            "prompt_rewrite": self.config.prompt_rewrite,
            "seed_policy": "caller_supplied_case_seed",
            "input_grid_policy": "reject_non_1024x1024",
        }

    def edit(self, image: Any, prompt: str, *, seed: int) -> Any:
        """Edit one image without thinking, reflection, or prompt rewriting."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise Step1XEditContractError("Step1X restoration prompt must be a non-empty string")
        source = _require_rgb_1024(image)
        import torch

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        result = self.pipeline(
            image=source,
            prompt=prompt,
            height=self.config.height,
            width=self.config.width,
            size_level=self.config.height,
            num_inference_steps=self.config.steps,
            true_cfg_scale=self.config.true_cfg_scale,
            generator=generator,
            # Explicitly pass both flags: v1p2 defaults reflection=True.
            enable_thinking_mode=False,
            enable_reflection_mode=False,
            return_dict=True,
        )
        outputs = getattr(result, "final_images", None) or getattr(result, "images", None)
        if not outputs:
            raise Step1XEditContractError("Step1X v1p2 pipeline returned no image")
        return _require_rgb_1024(outputs[0])


def load_backend(model_path: str | Path, *, device: str = "cuda") -> Step1XEditV1P2Backend:
    """Functional loader used by adapters that do not want the class name."""

    return Step1XEditV1P2Backend.load(model_path, device=device)


def edit_image(
    backend: Step1XEditV1P2Backend,
    image: Any,
    prompt: str,
    *,
    seed: int,
) -> Any:
    """Functional one-image boundary for generic baseline adapters."""

    return backend.edit(image, prompt, seed=seed)
