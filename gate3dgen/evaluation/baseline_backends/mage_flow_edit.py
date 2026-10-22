"""Mage-Flow-Edit RL-aligned backend for the restoration baseline.

This adapter is intentionally small and editor-only.  The shared FireRed
repair classifier and the benchmark's atomic output/manifest handling stay in
the caller.  It pins the requested Mage evaluation contract:

* ``microsoft/Mage-Flow-Edit`` (RL-aligned, 4B);
* bfloat16 model execution;
* 30 denoising steps;
* exact 1024x1024 input/output grid;
* the caller's complete restoration prompt is passed through unchanged (the
  Mage ``mage-flow-edit`` chat template only supplies the model's required
  wrapper; it does not rewrite or summarize the instruction);
* one deterministic seed per case.

The module imports Mage and torch lazily so host-only tests do not require the
model environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


MODEL_REPO_ID = "microsoft/Mage-Flow-Edit"
MODEL_VARIANT = "RL-aligned"
DEFAULT_STEPS = 30
DEFAULT_CFG = 5.0
DEFAULT_RESOLUTION = (1024, 1024)
DEFAULT_VL_COND_LONG_EDGE = 384

# The deployment workspace normally stores third-party models at
# ``<workspace>/models/third_party``.  Keep a project-local fallback because
# some shared workers expose the workspace-level ``models`` directory with
# mode 700 while the evaluation checkout remains writable.
_PROJECT_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()
_WORKSPACE_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()
_CODE_CANDIDATES = (
    _WORKSPACE_ROOT / "models/third_party/Mage-official/mage_flow",
    _PROJECT_ROOT / "models/third_party/Mage-official/mage_flow",
)
_MODEL_CANDIDATES = (
    _WORKSPACE_ROOT / "models/third_party/Mage-Flow-Edit-RL-aligned",
    _PROJECT_ROOT / "models/third_party/Mage-Flow-Edit-RL-aligned",
)
DEFAULT_CODE_ROOT = next((p for p in _CODE_CANDIDATES if p.is_dir()), _CODE_CANDIDATES[0])
DEFAULT_MODEL_ROOT = next((p for p in _MODEL_CANDIDATES if p.is_dir()), _MODEL_CANDIDATES[0])


class MageFlowEditContractError(ValueError):
    """Raised when an editor call would violate the baseline contract."""


@dataclass(frozen=True)
class MageFlowEditConfig:
    """Immutable runtime settings recorded in baseline provenance."""

    steps: int = DEFAULT_STEPS
    cfg: float = DEFAULT_CFG
    width: int = DEFAULT_RESOLUTION[0]
    height: int = DEFAULT_RESOLUTION[1]
    dtype: str = "bfloat16"
    prompt_rewrite: bool = False
    prompt_template: str = "mage-flow-edit"
    vl_cond_long_edge: int = DEFAULT_VL_COND_LONG_EDGE


def _require_rgb_1024(image: Any) -> Any:
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"Mage backend expects PIL.Image.Image, got {type(image)!r}")
    if image.size != DEFAULT_RESOLUTION:
        raise MageFlowEditContractError(
            f"Mage baseline requires an exact 1024x1024 pixel grid, got {image.size}"
        )
    return image.convert("RGB")


@dataclass
class MageFlowEditBackend:
    """Loaded Mage-Flow-Edit editor with a common baseline-friendly API."""

    pipeline: Any
    device: str = "cuda"
    config: MageFlowEditConfig = MageFlowEditConfig()
    attention_backend: str = "flash2"

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        device: str = "cuda",
        config: MageFlowEditConfig | None = None,
    ) -> "MageFlowEditBackend":
        """Load a local Mage model directory in bfloat16.

        ``model_path`` must be a local, self-contained Mage checkpoint.  The
        baseline launcher downloads weights before inference and this method
        intentionally does not perform an implicit online fetch.
        """

        import torch

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Mage model directory is missing: {model_path}")
        if not (model_path / "model_index.json").is_file():
            raise MageFlowEditContractError(
                f"Mage model directory has no model_index.json: {model_path}"
            )
        try:
            from mage_flow import MageFlowPipeline
        except ImportError as error:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "MageFlowPipeline is unavailable; install Mage-Flow in its dedicated environment"
            ) from error
        # The upstream checkpoint defaults to FlashAttention-2.  Keep that
        # fast path when the extension imports, but make the dedicated env
        # usable on CUDA installations where the source build is unavailable:
        # Mage ships a numerically equivalent packed SDPA fallback.  The
        # environment variable must be set before the text encoder is built;
        # the project-level backend is selected again after model loading
        # because MageFlowModel initializes it from the checkpoint config.
        attention_backend = "flash2"
        try:
            from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func  # noqa: F401
        except Exception:  # pragma: no cover - depends on local CUDA ABI
            attention_backend = "sdpa"
            os.environ.setdefault("VF_HF_ATTN_IMPL", "sdpa")
        pipeline = MageFlowPipeline.from_pretrained(str(model_path), device=device)
        if attention_backend == "sdpa":
            from mage_flow.models.modules._attn_backend import set_attn_backend

            set_attn_backend("sdpa")
        # Mage's loader owns component placement.  Keep the explicit torch
        # dtype in metadata and fail early on CUDA installations without BF16.
        if str(device).startswith("cuda") and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Mage RL-aligned baseline requires CUDA bfloat16 support")
        return cls(
            pipeline=pipeline,
            device=device,
            config=config or MageFlowEditConfig(),
            attention_backend=attention_backend,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return stable settings for a per-run provenance record."""

        return {
            "backend": "mage_flow_edit",
            "model_repo_id": MODEL_REPO_ID,
            "model_variant": MODEL_VARIANT,
            "steps": self.config.steps,
            "cfg": self.config.cfg,
            "dtype": self.config.dtype,
            "width": self.config.width,
            "height": self.config.height,
            "prompt_rewrite": self.config.prompt_rewrite,
            "prompt_template": self.config.prompt_template,
            "vl_cond_long_edge": self.config.vl_cond_long_edge,
            "attention_backend": self.attention_backend,
            "seed_policy": "caller_supplied_case_seed",
            "input_grid_policy": "reject_non_1024x1024",
        }

    def edit(self, image: Any, prompt: str, *, seed: int) -> Any:
        """Edit one image with the exact fixed Mage restoration settings."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise MageFlowEditContractError("Mage restoration prompt must be a non-empty string")
        source = _require_rgb_1024(image)
        # Mage's ``edit`` API accepts one reference list per prompt.  Supplying
        # the prompt directly and selecting the stock edit template preserves
        # the full caller string while satisfying its Qwen-VL chat format.
        outputs = self.pipeline.edit(
            [prompt],
            [[source]],
            neg_prompts=[" "],
            seeds=[int(seed)],
            heights=[self.config.height],
            widths=[self.config.width],
            steps=self.config.steps,
            cfg=self.config.cfg,
            prompt_template=self.config.prompt_template,
            vl_cond_long_edge=self.config.vl_cond_long_edge,
            device=self.device,
        )
        if not outputs:
            raise MageFlowEditContractError("Mage pipeline returned no image")
        output = _require_rgb_1024(outputs[0])
        return output


def load_backend(model_path: str | Path, *, device: str = "cuda") -> MageFlowEditBackend:
    """Functional loader used by adapters that do not want the class name."""

    return MageFlowEditBackend.load(model_path, device=device)


def edit_image(
    backend: MageFlowEditBackend,
    image: Any,
    prompt: str,
    *,
    seed: int,
) -> Any:
    """Functional one-image boundary for generic baseline adapters."""

    return backend.edit(image, prompt, seed=seed)
