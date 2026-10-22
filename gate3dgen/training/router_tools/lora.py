#!/usr/bin/env python3
"""Small, dependency-light contracts used by the Qwen repair-gate V3 run.

The V3 trainer deliberately keeps the native Qwen-VL LM head and does not add
an MLP classifier.  This module contains the two pieces that are easy to make
silently wrong in a distributed run:

* token-level causal-LM weighting, including the assistant terminator token;
* an explicit LoRA wrapper for the last four Qwen vision blocks.

The LoRA implementation is intentionally local instead of depending on PEFT.
That makes the FSDP checkpoint contain the adapter tensors as ordinary model
parameters and lets the exporter prove that no adapter key was dropped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


IM_END_TOKEN_ID = 151645
V3_SCHEMA_VERSION = "qwen-repair-gate-v3"
V2_SCHEMA_VERSION = "qwen-repair-gate-v2"


@dataclass(frozen=True)
class VisionLoraConfig:
    """The frozen B-variant LoRA contract."""

    enabled: bool = False
    target_blocks: tuple[int, ...] = (28, 29, 30, 31)
    target_suffixes: tuple[str, ...] = (
        "attn.qkv",
        "attn.proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_blocks"] = list(self.target_blocks)
        value["target_suffixes"] = list(self.target_suffixes)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "VisionLoraConfig":
        value = value or {}
        defaults = cls()
        blocks = tuple(int(x) for x in value.get("target_blocks", defaults.target_blocks))
        suffixes = tuple(str(x) for x in value.get("target_suffixes", defaults.target_suffixes))
        result = cls(
            enabled=bool(value.get("enabled", defaults.enabled)),
            target_blocks=blocks,
            target_suffixes=suffixes,
            rank=int(value.get("rank", defaults.rank)),
            alpha=float(value.get("alpha", defaults.alpha)),
            dropout=float(value.get("dropout", defaults.dropout)),
        )
        validate_lora_config(result)
        return result


def validate_lora_config(config: VisionLoraConfig) -> None:
    if config.target_blocks != (28, 29, 30, 31):
        raise ValueError(
            "V3 B requires exactly Qwen vision transformer blocks 28-31; "
            f"got {config.target_blocks}"
        )
    expected = VisionLoraConfig().target_suffixes
    if config.target_suffixes != expected:
        raise ValueError(
            "V3 B LoRA target modules must be exactly "
            f"{list(expected)}, got {list(config.target_suffixes)}"
        )
    if config.rank != 16 or config.alpha != 32.0 or not math.isclose(config.dropout, 0.05):
        raise ValueError(
            "V3 B LoRA requires rank=16, alpha=32, dropout=0.05; "
            f"got rank={config.rank}, alpha={config.alpha}, dropout={config.dropout}"
        )
    if config.rank <= 0 or config.alpha <= 0 or not 0 <= config.dropout < 1:
        raise ValueError(f"invalid LoRA hyperparameters: {config}")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear can wrap only nn.Linear, got {type(base)!r}")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, hidden_states: Tensor) -> Tensor:
        base = self.base(hidden_states)
        residual = F.linear(F.linear(self.dropout(hidden_states), self.lora_A), self.lora_B)
        return base + residual * self.scaling

    def merge_and_unload(self) -> nn.Linear:
        """Return a normal linear layer with the adapter merged into its weight."""

        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.lora_B @ self.lora_A * self.scaling)
            if self.base.bias is not None and merged.bias is not None:
                merged.bias.copy_(self.base.bias)
        merged.requires_grad_(False)
        return merged


def _get_module(root: nn.Module, dotted_name: str) -> nn.Module:
    current: nn.Module = root
    for part in dotted_name.split("."):
        if not hasattr(current, part):
            raise ValueError(f"model has no module {dotted_name!r} (missing {part!r})")
        current = getattr(current, part)
        if not isinstance(current, nn.Module):
            raise ValueError(f"module path {dotted_name!r} does not resolve to nn.Module")
    return current


def _set_module(root: nn.Module, dotted_name: str, value: nn.Module) -> None:
    parent_name, _, child_name = dotted_name.rpartition(".")
    parent = _get_module(root, parent_name) if parent_name else root
    if not hasattr(parent, child_name):
        raise ValueError(f"cannot replace missing module {dotted_name!r}")
    setattr(parent, child_name, value)


def vision_lora_target_names(
    model: nn.Module,
    config: VisionLoraConfig,
) -> list[str]:
    """Resolve and validate exact target names before changing the model."""

    validate_lora_config(config)
    visual = getattr(model, "visual", None)
    if visual is None:
        # A few wrappers expose the Qwen model under ``model``.  Accept that
        # layout, but still return names relative to the supplied model.
        visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None or not hasattr(visual, "blocks"):
        raise ValueError("Qwen model has no visual.blocks module for B LoRA")
    depth = len(visual.blocks)
    if depth < 32:
        raise ValueError(f"Qwen vision depth must be at least 32 for blocks 28-31, got {depth}")
    names: list[str] = []
    # Qwen2.5-VL exposes ``model.visual`` as the registered module and may
    # additionally expose a convenience ``model.visual``/``visual`` alias on
    # the wrapper.  Parameter/state-dict names follow the *registered* path,
    # so derive the prefix from named_modules instead of choosing an alias
    # based on object identity.  This keeps the target names aligned with
    # FSDP's complete state keys and the trainable-parameter contract.
    registered_visual_paths = [
        name for name, module in model.named_modules()
        if name and module is visual and name.split(".")[-1] == "visual"
    ]
    if not registered_visual_paths:
        raise ValueError("could not find registered path for Qwen visual module")
    visual_prefix = min(registered_visual_paths, key=len)
    for block_index in config.target_blocks:
        for suffix in config.target_suffixes:
            name = f"{visual_prefix}.blocks.{block_index}.{suffix}"
            module = _get_module(model, name)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"V3 B target {name} must be nn.Linear, got {type(module)!r}")
            names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("duplicate Qwen vision LoRA target names")
    return names


def inject_vision_lora(model: nn.Module, config: VisionLoraConfig) -> list[str]:
    """Inject adapters into blocks 28-31 and return the exact target names."""

    if not config.enabled:
        return []
    names = vision_lora_target_names(model, config)
    for name in names:
        module = _get_module(model, name)
        if isinstance(module, LoRALinear):
            raise ValueError(f"LoRA target already injected: {name}")
        _set_module(
            model,
            name,
            LoRALinear(module, rank=config.rank, alpha=config.alpha, dropout=config.dropout),
        )
    return names


def iter_lora_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, parameter in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            yield name, parameter


def merge_vision_lora(model: nn.Module) -> list[str]:
    """Merge every injected adapter and replace it with an ordinary Linear."""

    merged: list[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            _set_module(model, name, module.merge_and_unload())
            merged.append(name)
    return merged


def weighted_causal_lm_loss(
    logits: Tensor,
    labels: Tensor,
    token_weights: Tensor,
    sample_weights: Tensor,
) -> Tensor:
    """Compute normalized per-example token CE followed by sample weighting.

    ``labels`` and ``token_weights`` are aligned to the unshifted sequence.
    The final token is therefore never predicted; the assistant answer's
    first ``<|im_end|>`` is included at its own position and is predicted by
    the preceding answer token after the normal causal shift.
    """

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(f"expected logits [B,T,V] and labels [B,T], got {logits.shape}, {labels.shape}")
    if labels.shape != token_weights.shape or labels.shape[0] != sample_weights.shape[0]:
        raise ValueError(
            f"weighted CE shape mismatch: logits={tuple(logits.shape)} labels={tuple(labels.shape)} "
            f"token_weights={tuple(token_weights.shape)} sample_weights={tuple(sample_weights.shape)}"
        )
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_weights = token_weights[:, 1:].to(dtype=logits.dtype)
    valid = shifted_labels != -100
    if not bool(valid.any()):
        raise ValueError("weighted CE received no supervised tokens")
    per_token = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    weights = shifted_weights * valid.to(dtype=logits.dtype)
    per_example_denominator = weights.sum(dim=1).clamp_min(torch.finfo(logits.dtype).eps)
    per_example = (per_token * weights).sum(dim=1) / per_example_denominator
    example_weights = sample_weights.to(dtype=logits.dtype).reshape(-1)
    if bool((example_weights <= 0).any()):
        raise ValueError("sample weights must be positive")
    return (per_example * example_weights).sum() / example_weights.sum().clamp_min(
        torch.finfo(logits.dtype).eps
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total/trainable/LoRA parameter counts for trainer metadata."""

    total = trainable = lora = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
        if ".lora_A" in name or ".lora_B" in name:
            lora += count
    return {"total": total, "trainable": trainable, "lora_trainable": lora}

