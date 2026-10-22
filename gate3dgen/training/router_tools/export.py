#!/usr/bin/env python3
"""Export a complete Qwen repair-gate V3-B model with LoRA merged.

The V3 trainer uses ``SHARDED_STATE_DICT``.  An individual training
checkpoint therefore contains rank-local distributed-checkpoint shards, not a
loadable Qwen model.  This exporter deliberately does *not* scan such a
checkpoint for adapter tensors: doing so would silently discard the trained
language layers 24--27 and final norm.  The input must first be converted to
one complete state with Accelerate's CPU merge utility::

    accelerate merge-weights \
        /path/to/checkpoint-6000/pytorch_model_fsdp_0 \
        /path/to/checkpoint-6000-merged

Then pass ``/path/to/checkpoint-6000-merged`` as ``--merged-fsdp-model``.
The merged directory must contain ``model.safetensors`` (or
``pytorch_model.bin``), or a complete Hugging Face safetensors/bin index and
all of its declared shards.  Raw rank files, optimizer files, and a directory
containing only ``*.distcp``/``.metadata`` are rejected with an actionable
error.  After loading the complete wrapper state with ``strict=True``, this
tool merges every LoRA adapter and writes ordinary Qwen weights using
``save_pretrained``.  No non-strict loading or adapter-only export is allowed.

The small, dependency-light state loading and merge helpers are intentionally
public so CPU tests can exercise the completeness contract without loading
the 8B Qwen model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from safetensors.torch import load_file
except ImportError:  # pragma: no cover - training image supplies safetensors
    load_file = None  # type: ignore[assignment]

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError as exc:  # pragma: no cover - CPU contract tests do not load Qwen
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore[assignment,misc]
    _TRANSFORMERS_IMPORT_ERROR = exc

from gate3dgen.training.router_tools.lora import (
    VisionLoraConfig,
    inject_vision_lora,
    iter_lora_parameters,
    merge_vision_lora,
)


class CompleteStateError(ValueError):
    """Raised when an input is not an unambiguous complete model state."""


_DIRECT_STATE_NAMES = frozenset({
    "model.safetensors",
    "pytorch_model.bin",
    "model.pt",
    "model.pth",
})
_INDEX_NAMES = frozenset({
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
})
_OUTER_PREFIXES = (
    "module.",
    "_fsdp_wrapped_module.",
    "_orig_mod.",
    "model_0.",
)
_RAW_SHARD_MARKERS = (
    ".distcp",
    ".metadata",
    "optimizer",
    "random_states",
    "scheduler",
    "scaler",
)
_RANK_FILE_RE = re.compile(r"(?:^|[_\-.])rank\d+(?:[_\-.]|$)|(?:^|[_\-.])fsdp(?:[_\-.]|$)")


def _error_requires_merge(path: Path, detail: str) -> CompleteStateError:
    return CompleteStateError(
        f"{detail}: {path}. This exporter does not read raw FSDP/Accelerate "
        "rank shards. First run `accelerate merge-weights "
        "<checkpoint>/pytorch_model_fsdp_0 <merged-output>` and pass the "
        "result via --merged-fsdp-model."
    )


def _is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(isinstance(key, str) and isinstance(item, torch.Tensor) for key, item in value.items())


def _unwrap_state_dict(value: Any, source: Path) -> dict[str, torch.Tensor]:
    """Accept only a pure tensor state dict, never an optimizer/checkpoint blob."""

    if _is_tensor_mapping(value):
        return dict(value)
    if not isinstance(value, Mapping):
        raise CompleteStateError(f"complete model state must be a tensor mapping: {source}")
    # A full-state torch save is occasionally wrapped by Accelerate as
    # {"model": state_dict}; accept that one explicit wrapper only.  If any
    # sibling keys exist, rejecting is safer than silently dropping optimizer
    # or RNG state from a supposedly complete input.
    wrapper_keys = [key for key in ("state_dict", "model", "module") if key in value]
    if len(wrapper_keys) == 1 and len(value) == 1:
        return _unwrap_state_dict(value[wrapper_keys[0]], source)
    non_tensor = [key for key, item in value.items() if not isinstance(item, torch.Tensor)]
    if non_tensor:
        raise CompleteStateError(
            f"model state {source} is not a pure tensor mapping; refusing to ignore keys {non_tensor[:8]}"
        )
    raise CompleteStateError(f"could not unwrap complete model state: {source}")


def _load_state_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        if load_file is None:
            raise RuntimeError("safetensors is required to export a V3-B checkpoint")
        return _unwrap_state_dict(load_file(str(path), device="cpu"), path)
    try:
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0 has no weights_only keyword
            loaded = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise CompleteStateError(f"cannot load complete model state {path}: {exc}") from exc
    return _unwrap_state_dict(loaded, path)


def _merge_indexed_state(directory: Path, index_path: Path) -> tuple[dict[str, torch.Tensor], str]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompleteStateError(f"invalid Hugging Face weight index: {index_path}") from exc
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise CompleteStateError(f"weight index has no non-empty weight_map: {index_path}")
    shard_names = sorted({str(name) for name in weight_map.values()})
    result: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = directory / shard_name
        if not shard_path.is_file() or shard_path.name not in _DIRECT_STATE_NAMES and shard_path.suffix not in {".safetensors", ".bin"}:
            raise CompleteStateError(f"weight index references missing/unsupported shard: {shard_path}")
        shard_state = _load_state_file(shard_path)
        for key, value in shard_state.items():
            if key in result:
                raise CompleteStateError(f"duplicate tensor key across indexed shards: {key}")
            result[key] = value
    expected_keys = {str(key) for key in weight_map}
    if set(result) != expected_keys:
        missing = sorted(expected_keys - set(result))
        extra = sorted(set(result) - expected_keys)
        raise CompleteStateError(
            f"indexed state is incomplete/inconsistent: missing={missing[:5]} extra={extra[:5]}"
        )
    return result, "huggingface_indexed_complete"


def _reject_raw_shard_directory(directory: Path) -> None:
    entries = list(directory.iterdir())
    names = [entry.name for entry in entries]
    if any(name == ".metadata" or name.endswith(".metadata") or name.endswith(".distcp") for name in names):
        raise _error_requires_merge(directory, "raw distributed-checkpoint metadata/shards detected")
    if any(_RANK_FILE_RE.search(name) for name in names):
        raise _error_requires_merge(directory, "rank-local/FSDP weight files detected")
    if any(any(marker in name.lower() for marker in _RAW_SHARD_MARKERS) for name in names):
        # This is deliberately conservative.  The only accepted directory
        # formats are a named complete output or an explicit HF index.
        if not any(name in _INDEX_NAMES for name in names):
            raise _error_requires_merge(directory, "checkpoint directory contains non-model training state")


def load_complete_state(source: Path) -> tuple[dict[str, torch.Tensor], str]:
    """Load one complete model state and reject raw FSDP rank shards.

    Returns ``(state_dict, source_format)``.  A directory with a Hugging Face
    index is accepted only when every declared shard is present and all keys
    agree with that index.  A bare file is accepted only for the conventional
    complete names, so ``pytorch_model_fsdp_rank0.bin`` cannot be mistaken for
    a merged model.
    """

    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"merged model state does not exist: {source}")
    if source.is_file():
        if source.name not in _DIRECT_STATE_NAMES:
            if _RANK_FILE_RE.search(source.name) or any(marker in source.name.lower() for marker in _RAW_SHARD_MARKERS):
                raise _error_requires_merge(source, "raw FSDP/Accelerate state file detected")
            raise CompleteStateError(
                f"unsupported complete model state filename {source.name!r}; use model.safetensors "
                "or pytorch_model.bin from accelerate merge-weights"
            )
        return _load_state_file(source), "single_complete_state"
    if not source.is_dir():
        raise CompleteStateError(f"merged model state must be a file or directory: {source}")

    # Inspect sibling files before honoring an index.  A raw distributed
    # checkpoint must not become acceptable merely because an unrelated index
    # file was copied into the same directory.
    _reject_raw_shard_directory(source)
    for index_name in _INDEX_NAMES:
        index_path = source / index_name
        if index_path.is_file():
            return _merge_indexed_state(source, index_path)
    direct = sorted(source / name for name in _DIRECT_STATE_NAMES if (source / name).is_file())
    if len(direct) == 1:
        return _load_state_file(direct[0]), "single_complete_state"
    if not direct:
        raise _error_requires_merge(source, "no complete model.safetensors/pytorch_model.bin found")
    raise CompleteStateError(
        f"ambiguous merged model directory contains multiple complete state files: {[path.name for path in direct]}"
    )


def _strip_outer_prefixes(key: str) -> list[str]:
    candidates = [key]
    current = key
    changed = True
    while changed:
        changed = False
        for prefix in _OUTER_PREFIXES:
            if current.startswith(prefix):
                current = current[len(prefix):]
                candidates.append(current)
                changed = True
                break
    return candidates


def _key_aliases(key: str) -> list[str]:
    """Generate conservative aliases for FSDP wrapper prefixes.

    ``model.`` is a legitimate native Qwen prefix, so it is never removed
    unconditionally.  The alternatives below are matched only against the
    actual expected state keys and an ambiguous match is an error.
    """

    aliases: list[str] = []
    for candidate in _strip_outer_prefixes(key):
        aliases.append(candidate)
        if candidate.startswith("model."):
            aliases.append(candidate[len("model."):])
            aliases.append("model.language_model." + candidate[len("model."):])
        else:
            aliases.append("model." + candidate)
            aliases.append("model.language_model." + candidate)
        if candidate.startswith("language_model."):
            aliases.append("model." + candidate[len("language_model."):])
        if candidate.startswith("model.language_model."):
            aliases.append("model." + candidate[len("model.language_model."):])
    # Preserve insertion order while removing duplicates.
    return list(dict.fromkeys(aliases))


def normalize_complete_state_keys(
    state: Mapping[str, torch.Tensor],
    expected_keys: set[str] | frozenset[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Map known outer FSDP prefixes to model keys without dropping anything."""

    expected = set(expected_keys)
    normalized: dict[str, torch.Tensor] = {}
    raw_to_expected: dict[str, str] = {}
    unresolved: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            raise CompleteStateError(f"complete state contains a non-tensor/non-string item: {raw_key!r}")
        matches = sorted({candidate for candidate in _key_aliases(raw_key) if candidate in expected})
        if not matches:
            unresolved.append(raw_key)
            continue
        if len(matches) > 1:
            ambiguous[raw_key] = matches
            continue
        target = matches[0]
        if target in normalized and not torch.equal(normalized[target], value):
            raise CompleteStateError(f"normalized duplicate key has conflicting tensors: {raw_key} -> {target}")
        normalized[target] = value
        raw_to_expected[raw_key] = target
    if unresolved or ambiguous:
        details = []
        if unresolved:
            details.append(f"unexpected keys={unresolved[:8]}")
        if ambiguous:
            details.append(f"ambiguous keys={dict(list(ambiguous.items())[:3])}")
        raise CompleteStateError("complete model state key normalization failed: " + "; ".join(details))
    missing = sorted(expected - set(normalized))
    if missing:
        raise CompleteStateError(
            f"complete model state is missing {len(missing)} model keys, e.g. {missing[:8]}; "
            "the input must include frozen base, language layers 24-27/final norm, and LoRA wrapper weights"
        )
    if len(normalized) != len(expected):
        raise CompleteStateError(
            f"complete model state key count mismatch after normalization: got {len(normalized)}, expected {len(expected)}"
        )
    normalized_prefixes: set[str] = set()
    for raw_key, _target in raw_to_expected.items():
        candidate = raw_key
        changed = True
        while changed:
            changed = False
            for prefix in _OUTER_PREFIXES:
                if candidate.startswith(prefix):
                    normalized_prefixes.add(prefix)
                    candidate = candidate[len(prefix):]
                    changed = True
                    break
    return normalized, {
        "raw_key_count": len(state),
        "normalized_key_count": len(normalized),
        "outer_prefixes_normalized": sorted(normalized_prefixes),
    }


def _is_trainable_language_key(key: str) -> bool:
    return bool(re.search(r"(?:^|\.)layers\.(?:24|25|26|27)\.", key)) or bool(
        re.search(r"(?:^|\.)norm\.(?:weight|bias)$", key)
    )


def merge_complete_state(
    model: torch.nn.Module,
    complete_state: Mapping[str, torch.Tensor],
    lora_config: VisionLoraConfig,
) -> dict[str, Any]:
    """Inject V3-B, strictly load a *complete* state, then merge all adapters."""

    if not lora_config.enabled:
        raise ValueError("V3 LoRA exporter requires lora_config.enabled=true")
    target_names = inject_vision_lora(model, lora_config)
    expected_keys = set(model.state_dict())
    normalized_state, normalization = normalize_complete_state_keys(complete_state, expected_keys)

    # Completeness has already been checked above; strict=True is intentional
    # and is kept here as a second guard against a model/state topology drift.
    try:
        model.load_state_dict(normalized_state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise CompleteStateError(f"strict complete-state load failed: {exc}") from exc

    language_keys = [key for key in normalized_state if _is_trainable_language_key(key)]
    wrapper_base_keys = [key for key in normalized_state if ".base." in key]
    adapter_keys = {
        name for name, _parameter in iter_lora_parameters(model)
    }
    missing_adapters = sorted(adapter_keys - set(normalized_state))
    if missing_adapters:
        raise CompleteStateError(f"strict state unexpectedly omitted LoRA keys: {missing_adapters[:8]}")
    if not language_keys:
        raise CompleteStateError(
            "complete state has no Language Transformer 24-27/Final Norm keys; refusing adapter-only export"
        )
    if not wrapper_base_keys:
        raise CompleteStateError(
            "complete state has no LoRA wrapper base keys; refusing to merge an adapter without its trained base"
        )
    if len(adapter_keys) != len(target_names) * 2:
        raise CompleteStateError(
            f"injected LoRA topology is incomplete: got {len(adapter_keys)} tensors for {len(target_names)} targets"
        )
    merged_names = merge_vision_lora(model)
    if len(merged_names) != len(target_names):
        raise RuntimeError(f"merged {len(merged_names)} LoRA modules, expected {len(target_names)}")
    post_keys = set(model.state_dict())
    dangling_lora = sorted(key for key in post_keys if ".lora_A" in key or ".lora_B" in key)
    dangling_wrappers = sorted(key for key in post_keys if ".base." in key)
    if dangling_lora or dangling_wrappers:
        raise RuntimeError(
            f"LoRA merge incomplete: dangling adapter keys={dangling_lora[:4]} wrapper base keys={dangling_wrappers[:4]}"
        )
    return {
        "target_names": target_names,
        "loaded_state_key_count": len(normalized_state),
        "loaded_language_trainable_key_count": len(language_keys),
        "loaded_lora_adapter_key_count": len(adapter_keys),
        "loaded_lora_wrapper_base_key_count": len(wrapper_base_keys),
        "merged_module_count": len(merged_names),
        "post_merge_state_key_count": len(post_keys),
        "normalization": normalization,
        "strict_complete_state_load": True,
        "silent_lora_drop": False,
    }


def _read_trainer_state(checkpoint: Path | None) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    trainer_state = checkpoint / "trainer_state.json"
    if not trainer_state.is_file():
        raise CompleteStateError(f"--checkpoint metadata path has no trainer_state.json: {trainer_state}")
    try:
        value = json.loads(trainer_state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompleteStateError(f"cannot read checkpoint trainer_state.json: {trainer_state}") from exc
    if not isinstance(value, dict):
        raise CompleteStateError(f"checkpoint trainer_state.json must contain an object: {trainer_state}")
    return value


def export_merged(
    *,
    base_model: Path,
    merged_fsdp_model: Path,
    output: Path,
    lora_config: VisionLoraConfig,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Load a merged complete wrapper state, merge LoRA, and save standard Qwen."""

    if not lora_config.enabled:
        raise ValueError("V3 LoRA exporter requires lora_config.enabled=true")
    trainer_state = _read_trainer_state(checkpoint)
    recorded = trainer_state.get("lora_config")
    if recorded is not None and recorded != lora_config.as_dict():
        raise ValueError(
            "checkpoint LoRA config differs from exporter config; "
            f"recorded={recorded!r} requested={lora_config.as_dict()!r}"
        )
    # Validate the source format before importing/loading the 8B base model.
    # This keeps raw-shard failures actionable even on a CPU utility host that
    # does not have the training-image transformers build installed.
    complete_state, source_format = load_complete_state(merged_fsdp_model)
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("transformers cannot import Qwen2_5_VLForConditionalGeneration") from _TRANSFORMERS_IMPORT_ERROR
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model / "text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.requires_grad_(False)
    merge_metadata = merge_complete_state(model, complete_state, lora_config)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing export directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=False)
    model.save_pretrained(output, safe_serialization=True)
    metadata = {
        "status": "complete",
        "base_model": str(base_model),
        "merged_fsdp_model": str(merged_fsdp_model),
        "checkpoint_metadata": str(checkpoint) if checkpoint is not None else None,
        "output": str(output),
        "source_format": source_format,
        "lora_config": lora_config.as_dict(),
        **merge_metadata,
        "exported_standard_qwen_weights": True,
        "raw_fsdp_shards_rejected": True,
    }
    (output / "v3_lora_export.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--merged-fsdp-model",
        type=Path,
        required=True,
        help="complete state file/directory produced by `accelerate merge-weights`; raw checkpoint shards are rejected",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional original V3 checkpoint directory, used only to verify trainer_state.json LoRA metadata",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args()
    config = VisionLoraConfig(enabled=True, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    try:
        result = export_merged(
            base_model=args.base_model.expanduser().resolve(),
            merged_fsdp_model=args.merged_fsdp_model.expanduser().resolve(),
            checkpoint=args.checkpoint.expanduser().resolve() if args.checkpoint else None,
            output=args.output.expanduser().resolve(),
            lora_config=config,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

