"""Official Black Forest Labs FLUX.1 Kontext [dev] editor backend.

The backend uses the cloned ``black-forest-labs/flux`` reference implementation
and its native 30-step/guidance-2.5 Kontext sampler.  It does not call the
interactive CLI, content-filter prompt rewriting, or any external captioner;
the caller's restoration prompt and per-case seed are passed directly.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()
DEFAULT_CODE_ROOT = WORKSPACE_ROOT / "models/third_party/flux"
DEFAULT_MODEL_ROOT = WORKSPACE_ROOT / "models/checkpoints/FLUX.1-Kontext-dev"
TARGET_SIZE = (1024, 1024)


@dataclass(frozen=True)
class FluxKontextConfig:
    model_path: Path = DEFAULT_MODEL_ROOT
    code_root: Path = DEFAULT_CODE_ROOT
    width: int = TARGET_SIZE[0]
    height: int = TARGET_SIZE[1]
    num_inference_steps: int = 30
    guidance: float = 2.5
    dtype: str = "bfloat16"
    prompt_rewrite: bool = False
    content_filter: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": "flux_kontext",
            "pipeline": "black-forest-labs/flux reference sampler",
            "model_path": str(Path(self.model_path).resolve()),
            "code_root": str(Path(self.code_root).resolve()),
            "width": self.width,
            "height": self.height,
            "num_inference_steps": self.num_inference_steps,
            "guidance": self.guidance,
            "dtype": self.dtype,
            "prompt_rewrite": self.prompt_rewrite,
            "content_filter": self.content_filter,
            "seed_policy": "caller_supplied_stable_case_seed",
            "input_output_size": list(TARGET_SIZE),
            "local_files_only": True,
            "license": "FLUX.1-dev Non-Commercial License",
        }


def _ensure_code_root(code_root: Path) -> None:
    code_root = Path(code_root).resolve()
    source = code_root / "src"
    if not (source / "flux" / "__init__.py").is_file():
        raise FileNotFoundError(f"official FLUX code is missing: {code_root}")
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)


class _LocalHFEmbedder:
    """Local-only T5/CLIP wrapper matching the reference HFEmbedder API."""

    def __init__(self, tokenizer_path: Path, model_path: Path, *, is_clip: bool, device: str) -> None:
        import torch
        from torch import nn
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

        super().__init__()
        self.is_clip = is_clip
        self.max_length = 77 if is_clip else 512
        self.output_key = "pooler_output" if is_clip else "last_hidden_state"
        if is_clip:
            self.tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, max_length=77)
            self.hf_module = CLIPTextModel.from_pretrained(
                str(model_path), local_files_only=True, torch_dtype=torch.bfloat16
            )
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, max_length=512)
            self.hf_module = T5EncoderModel.from_pretrained(
                str(model_path), local_files_only=True, torch_dtype=torch.bfloat16
            )
        self.hf_module = self.hf_module.eval().requires_grad_(False).to(device)

    def __call__(self, text: list[str]) -> Any:
        import torch

        batch = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = self.hf_module(
                input_ids=batch["input_ids"].to(self.hf_module.device),
                attention_mask=None,
                output_hidden_states=False,
            )
        return outputs[self.output_key].bfloat16()


class FluxKontextBackend:
    """Thin deterministic wrapper around the official Kontext sampler."""

    def __init__(self, config: FluxKontextConfig | None = None, *, device: str = "cuda") -> None:
        import torch

        self.config = config or FluxKontextConfig()
        self.device = device
        _ensure_code_root(self.config.code_root)
        model_root = Path(self.config.model_path).resolve()
        flow_path = model_root / "flux1-kontext-dev.safetensors"
        ae_path = model_root / "ae.safetensors"
        required = [flow_path, ae_path, model_root / "text_encoder", model_root / "text_encoder_2"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("FLUX Kontext checkpoint is incomplete: " + ", ".join(missing))

        from flux.sampling import denoise, get_schedule, prepare_kontext, unpack
        from flux.util import load_ae, load_flow_model

        self._denoise = denoise
        self._get_schedule = get_schedule
        self._prepare_kontext = prepare_kontext
        self._unpack = unpack
        # The reference loader honors these explicit local paths and never
        # falls back to a remote checkpoint when they exist.
        os.environ["FLUX_MODEL"] = str(flow_path)
        os.environ["FLUX_AE"] = str(ae_path)
        self.model = load_flow_model("flux-dev-kontext", device=device)
        self.ae = load_ae("flux-dev-kontext", device=device)
        self.t5 = _LocalHFEmbedder(
            model_root / "tokenizer_2", model_root / "text_encoder_2", is_clip=False, device=device
        )
        self.clip = _LocalHFEmbedder(
            model_root / "tokenizer", model_root / "text_encoder", is_clip=True, device=device
        )
        self.model.eval()

    @property
    def metadata(self) -> dict[str, Any]:
        return self.config.as_dict()

    def edit(self, image: Any, prompt: str, seed: int) -> Any:
        import numpy as np
        import torch
        from PIL import Image

        if tuple(image.size) != TARGET_SIZE:
            raise ValueError(f"FLUX Kontext baseline requires input size {TARGET_SIZE}, got {image.size}")
        # The official prepare_kontext API intentionally accepts a path.  Use
        # a private temporary PNG so the exact input pixels reach that API and
        # no run-global temporary state can collide across 32 workers.
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            image.convert("RGB").save(handle.name, format="PNG")
            inp, height, width = self._prepare_kontext(
                t5=self.t5,
                clip=self.clip,
                prompt=str(prompt),
                ae=self.ae,
                img_cond_path=handle.name,
                target_width=self.config.width,
                target_height=self.config.height,
                bs=1,
                seed=int(seed),
                device=torch.device(self.device),
            )
        timesteps = self._get_schedule(
            self.config.num_inference_steps, inp["img"].shape[1], shift=True
        )
        with torch.inference_mode():
            latents = self._denoise(self.model, **{k: v for k, v in inp.items() if k != "img_cond_orig"}, timesteps=timesteps, guidance=self.config.guidance)
            decoded = self._unpack(latents.float(), height, width)
            with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16):
                decoded = self.ae.decode(decoded)
        pixels = decoded.clamp(-1, 1)
        pixels = (127.5 * (pixels[0].float() + 1.0)).byte().permute(1, 2, 0).cpu().numpy()
        restored = Image.fromarray(np.asarray(pixels), mode="RGB")
        if tuple(restored.size) != TARGET_SIZE:
            raise RuntimeError(f"FLUX Kontext returned non-contract size {restored.size}; no resize applied")
        return restored

    def close(self) -> None:
        self.model = self.ae = self.t5 = self.clip = None
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
) -> FluxKontextBackend:
    config = FluxKontextConfig(model_path=Path(model_path), code_root=Path(code_root), **overrides)
    return FluxKontextBackend(config, device=device)


def edit_image(backend: FluxKontextBackend, image: Any, prompt: str, seed: int) -> Any:
    return backend.edit(image, prompt, seed)


def smoke_import(code_root: str | Path = DEFAULT_CODE_ROOT) -> dict[str, Any]:
    _ensure_code_root(Path(code_root))
    from flux.sampling import denoise, prepare_kontext

    return {"status": "import_ok", "prepare": prepare_kontext.__name__, "denoise": denoise.__name__}


__all__ = ["FluxKontextBackend", "FluxKontextConfig", "edit_image", "load_backend", "smoke_import"]
