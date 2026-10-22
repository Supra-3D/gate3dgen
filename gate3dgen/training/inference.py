from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .artifacts import adapter_dir, checkpoint_artifact_kind
from .checkpoint_state import normalize_state_dict_root


EXPECTED_PROMPT_TEMPLATE_START_IDX = 64
EXPECTED_EDIT_SYSTEM_PROMPT = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new "
    "image that meets the user's requirements while maintaining consistency with the original input "
    "where appropriate."
)


def convert_peft_adapter_keys(state: dict[str, Any]) -> dict[str, Any]:
    converted = {}
    for key, value in state.items():
        if "lora_A" not in key and "lora_B" not in key:
            continue
        key = key.replace("base_model.model.", "")
        key = key.replace("base_model.", "")
        key = key.replace(".lora_A.default.weight", ".lora_A.weight")
        key = key.replace(".lora_B.default.weight", ".lora_B.weight")
        if not key.startswith("transformer."):
            key = "transformer." + key
        converted[key] = value
    if not converted:
        raise ValueError("adapter contains no LoRA tensors")
    return converted


def load_lora(pipe: Any, checkpoint_or_adapter: Path) -> Path:
    from safetensors.torch import load_file

    directory = adapter_dir(checkpoint_or_adapter)
    weights = directory / "adapter_model.safetensors"
    state = convert_peft_adapter_keys(load_file(str(weights)))
    config_path = directory / "adapter_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rank = float(config.get("r", 1))
        alpha = float(config.get("lora_alpha", rank))
        if rank > 0 and alpha != rank:
            scale = alpha / rank
            state = {key: (value * scale if ".lora_B." in key else value) for key, value in state.items()}
    pipe.load_lora_weights(state)
    return weights


def _load_full_transformer_state(checkpoint: Path) -> dict[str, Any]:
    safetensors_path = checkpoint / "diffusion_pytorch_model.safetensors"
    fsdp_path = checkpoint / "pytorch_model_fsdp.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))
    if fsdp_path.is_file():
        import torch

        return torch.load(str(fsdp_path), map_location="cpu", weights_only=True, mmap=True)
    raise FileNotFoundError(f"No full transformer weights under {checkpoint}")


def _load_full_transformer(
    model_path: Path,
    checkpoint: Path,
    architecture: str,
    attention_backend: str,
    device: str,
) -> Any:
    import torch
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    from .vendor.train.src.models import QwenImageRestorationTransformer2DModel

    transformer_class = (
        QwenImageRestorationTransformer2DModel
        if architecture == "restoration_custom"
        else QwenImageTransformer2DModel
    )
    diffusers_dir = checkpoint / "transformer"
    if diffusers_dir.is_dir() and any(
        diffusers_dir.glob("diffusion_pytorch_model*.safetensors")
    ):
        transformer = transformer_class.from_pretrained(
            str(checkpoint), subfolder="transformer", torch_dtype=torch.bfloat16
        )
    else:
        from accelerate import init_empty_weights

        config_path = checkpoint / "config.json"
        if not config_path.is_file():
            config_path = model_path / "transformer/config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        # Complete fine-tuned state, not an adapter: never read original base
        # transformer tensors or allocate an FP32 copy just to overwrite it.
        with init_empty_weights():
            transformer = transformer_class.from_config(config)
        state = _load_full_transformer_state(checkpoint)
        # The released purifier has no auxiliary head. Normalize only roots
        # common to every key, then strictly reject unknown or missing tensors.
        state = normalize_state_dict_root(state)
        result = transformer.load_state_dict(state, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"full transformer load mismatch: {result.missing_keys[:3]} / "
                f"{result.unexpected_keys[:3]}"
            )
        del state
        transformer.to(dtype=torch.bfloat16)
    if architecture == "restoration_custom":
        transformer.set_restoration_attention_backend(
            attention_backend, device=device, dtype=torch.bfloat16
        )
    elif attention_backend != "native":
        raise ValueError("non-native attention requires restoration_custom architecture")
    return transformer


def _load_custom_base_transformer(
    model_path: Path, attention_backend: str, device: str
) -> Any:
    import torch
    from .vendor.train.src.models import QwenImageRestorationTransformer2DModel

    transformer = QwenImageRestorationTransformer2DModel.from_pretrained(
        str(model_path), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    transformer.set_restoration_attention_backend(
        attention_backend, device=device, dtype=torch.bfloat16
    )
    return transformer


def load_pipeline(
    model_path: Path,
    checkpoint_or_adapter: Path,
    device: str = "cuda",
    *,
    architecture: str = "stock",
    attention_backend: str = "native",
    cpu_offload: bool = False,
    condition_text_encoder: Any = None,
    condition_processor: Any = None,
) -> Any:
    import torch
    from diffusers import QwenImageEditPlusPipeline

    kind = checkpoint_artifact_kind(checkpoint_or_adapter)
    transformer = None
    if kind == "transformer":
        transformer = _load_full_transformer(
            model_path,
            checkpoint_or_adapter,
            architecture,
            attention_backend,
            device,
        )
    elif architecture == "restoration_custom":
        transformer = _load_custom_base_transformer(
            model_path, attention_backend, device
        )
    elif attention_backend != "native":
        raise ValueError("non-native attention requires restoration_custom architecture")
    conditioning = {}
    if condition_text_encoder is not None:
        if condition_processor is None:
            raise ValueError("Router-conditioned purification requires the matching processor")
        conditioning = {"text_encoder": condition_text_encoder, "processor": condition_processor}
    if transformer is not None and condition_text_encoder is not None:
        from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler

        # The released pipeline is assembled explicitly from our own models
        # and the small frozen runtime. No original Qwen/diffusion weights.
        pipe = QwenImageEditPlusPipeline(
            transformer=transformer, text_encoder=condition_text_encoder,
            processor=condition_processor, tokenizer=condition_processor.tokenizer,
            vae=AutoencoderKLQwenImage.from_pretrained(
                str(model_path / "vae"), torch_dtype=torch.bfloat16, local_files_only=True
            ),
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(model_path / "scheduler"), local_files_only=True
            ),
        )
    else:
        # Unconditional/adapter research paths still use a full initialization
        # pipeline. This does not change the released training recipe.
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, transformer=transformer, **conditioning
        ) if transformer is not None else QwenImageEditPlusPipeline.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, **conditioning
        )
    if condition_text_encoder is not None:
        # Purification uses final hidden states, bypassing the vocabulary LM head.
        pipe.text_encoder = condition_text_encoder.model
    if kind == "adapter":
        load_lora(pipe, checkpoint_or_adapter)
    if cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    validate_conditioning_contract(pipe)
    return pipe


def validate_conditioning_contract(pipe: Any) -> None:
    """Fail before generation if online conditioning differs from training embeddings."""

    template = str(getattr(pipe, "prompt_template_encode", ""))
    start_idx = getattr(pipe, "prompt_template_encode_start_idx", None)
    if EXPECTED_EDIT_SYSTEM_PROMPT not in template:
        raise RuntimeError("pipeline edit system prompt does not match the offline embedding contract")
    if start_idx != EXPECTED_PROMPT_TEMPLATE_START_IDX:
        raise RuntimeError(
            "pipeline prompt template start index does not match offline embeddings: "
            f"{start_idx} != {EXPECTED_PROMPT_TEMPLATE_START_IDX}"
        )


def _scheduler_for_inference_steps(pipe: Any, steps: int) -> tuple[Any, Any]:
    """Install a finite one-step FlowMatch schedule and return (active, original)."""

    if int(steps) != 1:
        return pipe.scheduler, None
    original = pipe.scheduler
    config = dict(original.config)
    if config.get("shift_terminal") is None:
        return original, None

    # FlowMatchEulerDiscreteScheduler cannot stretch the singleton [sigma=1]
    # schedule to shift_terminal: its scale factor is zero and 0/0 yields NaN.
    # Disabling only that stretch preserves the trained one-step path [1 -> 0].
    pipe.scheduler = type(original).from_config(config, shift_terminal=None)
    return pipe.scheduler, original


def _validate_scheduler_state(scheduler: Any, steps: int) -> None:
    import torch

    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas
    if not torch.isfinite(timesteps).all() or not torch.isfinite(sigmas).all():
        raise RuntimeError(
            f"non-finite scheduler state for steps={steps}: "
            f"timesteps={timesteps.detach().cpu().tolist()} "
            f"sigmas={sigmas.detach().cpu().tolist()}"
        )
    if int(steps) == 1:
        expected = torch.tensor([1.0, 0.0], dtype=sigmas.dtype, device=sigmas.device)
        if sigmas.numel() != 2 or not torch.allclose(sigmas, expected, atol=1e-6, rtol=0):
            raise RuntimeError(
                "single-step restoration requires the exact sigma path [1, 0], got "
                f"{sigmas.detach().cpu().tolist()}"
            )


def generate(
    pipe: Any,
    source: Path,
    prompt: str,
    *,
    seed: int,
    steps: int,
    cfg: float,
    negative_prompt: Optional[str] = None,
) -> Image.Image:
    import torch

    with Image.open(source) as image:
        condition = image.convert("RGB")
    inputs = {
        "image": [condition],
        "prompt": prompt,
        "generator": torch.Generator(device=pipe._execution_device).manual_seed(int(seed)),
        "true_cfg_scale": float(cfg),
        "negative_prompt": negative_prompt,
        "num_inference_steps": int(steps),
        "num_images_per_prompt": 1,
    }
    _, original_scheduler = _scheduler_for_inference_steps(pipe, steps)
    try:
        def validate_scheduler(_pipe, _step, _timestep, callback_kwargs):
            if _step == 0:
                _validate_scheduler_state(_pipe.scheduler, steps)
            latents = callback_kwargs.get("latents")
            if latents is not None and not torch.isfinite(latents).all():
                raise RuntimeError(f"non-finite denoising latents at inference step {_step}")
            return callback_kwargs

        inputs["callback_on_step_end"] = validate_scheduler
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        with torch.inference_mode():
            return pipe(**inputs).images[0]
    finally:
        if original_scheduler is not None:
            pipe.scheduler = original_scheduler
