"""Contracts for offline Qwen-VL hidden-state conditioning.

The restoration trainer consumes tensors produced by a frozen Qwen-VL model;
it must never silently fall back to the base Qwen model when a gate-specific
model is requested.  This module keeps the identity and artifact checks small
enough to run before distributed initialisation.  It intentionally does not
import torch/transformers, so launchers and CPU tests can exercise the whole
contract without loading a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


CONDITION_SCHEMA_VERSION = "qwen-vl-hidden-states-condition-v1"
CONDITION_KIND = "offline_qwen_vl_hidden_states"
_WORKSPACE_PREFIXES = ()
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
_WEIGHT_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


class ConditionContractError(ValueError):
    """Raised when a Qwen condition artifact is missing or ambiguous."""


def _workspace_root(config: Mapping[str, Any] | None = None) -> Path:
    configured = os.environ.get("GATE3DGEN_ROOT") or os.environ.get("QWEN_GATE_WORKSPACE_ROOT") or os.environ.get(
        "FIRERED_WORKSPACE_ROOT"
    )
    if configured:
        return Path(configured).expanduser().resolve()
    # The package is .../scripts/code/<project>/gate3dgen.training.
    return Path.cwd().resolve()


def adapt_workspace_path(value: Any, *, root: Path | None = None) -> str:
    """Remap known host prefixes while preserving arbitrary user paths."""

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    root = (root or _workspace_root()).resolve()
    for prefix in _WORKSPACE_PREFIXES:
        if text == prefix:
            return str(root)
        if text.startswith(prefix + "/"):
            return str(root / text[len(prefix) + 1 :])
    return text


def _resolve_workspace_path(value: Any, *, root: Path) -> Path:
    """Resolve condition paths against the mounted workspace, never the CWD.

    Distributed launchers starts user scripts from an algorithm/job directory.  Calling
    ``Path.resolve()`` directly on a relative config value would therefore
    produce ``a launcher-specific directory`` instead of the data
    mount.  Known absolute host prefixes are still remapped first so the same
    config remains portable between the workstation and training image.
    """

    adapted = Path(adapt_workspace_path(value, root=root)).expanduser()
    if not adapted.is_absolute():
        adapted = root / adapted
    return adapted.resolve()


def portable_path(value: Any) -> str | None:
    """Return a mount-independent path label for identity payloads."""

    if value is None or not str(value).strip():
        return None
    text = str(value).replace("\\", "/")
    for prefix in _WORKSPACE_PREFIXES:
        prefix = prefix.replace("\\", "/")
        if text == prefix:
            return "workspace"
        if text.startswith(prefix + "/"):
            return "workspace/" + text[len(prefix) + 1 :]
    # Configs commonly use /.../scripts/... paths from a third-party mount.
    if "/scripts/" in text:
        return "workspace/scripts/" + text.split("/scripts/", 1)[1]
    return text


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ConditionContractError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ConditionContractError(f"{label} must contain a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ConditionContractError(f"cannot hash condition artifact: {path}") from error
    return digest.hexdigest()


def _sha256_json(path: Path) -> str:
    return _sha256_file(path)


def _weight_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    direct = sorted(
        child
        for child in path.iterdir()
        if child.is_file()
        and (child.name in _WEIGHT_NAMES or child.suffix.lower() in _WEIGHT_SUFFIXES)
        and not child.name.endswith(".index.json")
    )
    # An indexed model may use arbitrary shard names.  Include exactly the
    # declared shards instead of hashing unrelated optimizer/RNG files.
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        index = _read_json(index_path, "condition weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ConditionContractError(f"condition weight index has no weight_map: {index_path}")
        declared = []
        for name in sorted({str(value) for value in weight_map.values()}):
            shard = path / name
            if not shard.is_file():
                raise ConditionContractError(f"condition index references missing shard: {shard}")
            declared.append(shard)
        direct = [index_path, *declared]
        break
    return list(dict.fromkeys(direct))


def _identity_hash(path: Path, declared: Any = None) -> str:
    """Hash the complete condition weight set, honoring an explicit digest.

    Gate A is a 21 GB merged model, so production launchers normally provide a
    recorded digest through ``condition_model_hash`` after the artifact has
    been copied.  ``auto`` remains useful for small CPU fixtures and computes
    a true content digest rather than accepting a path-only identity.
    """

    if declared and str(declared).lower() not in {"auto", "computed"}:
        value = str(declared).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ConditionContractError("condition_model_hash must be a 64-character SHA256")
        return value
    files = _weight_files(path)
    if not files:
        raise ConditionContractError(f"condition model has no complete weight files: {path}")
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _directory_signature(path: Path, *, include_content: bool = True) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return _sha256_file(path) if include_content else f"{path.stat().st_size}:{path.stat().st_mtime_ns}"
    digest = hashlib.sha256()
    files = sorted(child for child in path.rglob("*") if child.is_file())
    for child in files:
        rel = child.relative_to(path).as_posix()
        # Processor directories contain no multi-GB tensors.  Ignore caches
        # and VCS noise so moving a tokenizer does not change its identity.
        if any(part in {"__pycache__", ".git"} for part in child.parts):
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child).encode("ascii") if include_content else str(child.stat().st_size).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _condition_mapping(config: Mapping[str, Any]) -> dict[str, Any] | None:
    value = config.get("condition_model")
    if value is None:
        embedding = config.get("embedding", {})
        if isinstance(embedding, Mapping):
            value = embedding.get("condition_model")
    if value is None:
        value = config.get("condition")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConditionContractError("condition_model must be a mapping")
    return dict(value)


def condition_enabled(config: Mapping[str, Any]) -> bool:
    value = _condition_mapping(config)
    return bool(value and value.get("enabled", True))


def normalize_condition_config(config: dict[str, Any]) -> None:
    """Normalize a config's condition section and explicit env overrides."""

    value = _condition_mapping(config)
    if value is None:
        return
    value = dict(value)
    enabled = bool(value.get("enabled", True))
    value["enabled"] = enabled
    base = config.get("model", {}).get("pretrained_model_path")
    if base:
        value.setdefault("base_model_path", base)
        value.setdefault("processor_path", str(Path(base) / "processor"))
        value.setdefault("config_path", str(Path(base) / "text_encoder"))
    root = _workspace_root(config)
    env_values = {
        "model_path": (
            os.environ.get("GATE3DGEN_ROUTER_EXPORT")
            or os.environ.get("FIRERED_CONDITION_MODEL_PATH")
            or os.environ.get("FIRERED_GATE_B_EXPORT_DIR")
            or os.environ.get("QWEN_GATE_V3_EXPORT_OUTPUT")
        ),
        "base_model_path": os.environ.get("FIRERED_CONDITION_BASE_MODEL_PATH"),
        "processor_path": os.environ.get("FIRERED_CONDITION_PROCESSOR_PATH"),
        "config_path": os.environ.get("FIRERED_CONDITION_CONFIG_PATH"),
        "checkpoint_path": (
            os.environ.get("GATE3DGEN_ROUTER_CHECKPOINT")
            or os.environ.get("FIRERED_CONDITION_CHECKPOINT")
            or os.environ.get("FIRERED_GATE_B_CHECKPOINT")
            or os.environ.get("QWEN_GATE_V3_CHECKPOINT")
        ),
        "variant": os.environ.get("FIRERED_CONDITION_VARIANT"),
        "model_hash": os.environ.get("FIRERED_CONDITION_MODEL_HASH"),
    }
    for key, env_value in env_values.items():
        if env_value is not None and str(env_value).strip():
            value[key] = env_value
    for key in ("model_path", "base_model_path", "processor_path", "config_path", "checkpoint_path"):
        if value.get(key):
            # Downstream command builders consume this normalized mapping
            # directly.  Keep every runtime path absolute here; otherwise an
            # extractor launched from Distributed launchers' algorithm directory sees a
            # relative processor/config value and Transformers mistakes it
            # for a Hugging Face repository id.
            value[key] = str(_resolve_workspace_path(value[key], root=root))
    config["condition_model"] = value


def _checkpoint_info(
    checkpoint: Path,
    *,
    expected_variant: str,
    expected_step: int | None,
    expected_source_world_size: int | None,
) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise ConditionContractError(f"condition checkpoint directory is missing: {checkpoint}")
    match = _CHECKPOINT_RE.match(checkpoint.name)
    if not match:
        raise ConditionContractError(f"condition checkpoint must be named checkpoint-N: {checkpoint}")
    step = int(match.group(1))
    if expected_step is not None and step != int(expected_step):
        raise ConditionContractError(
            f"condition checkpoint step mismatch: expected {expected_step}, got {step}: {checkpoint}"
        )
    marker_path = checkpoint / "checkpoint_complete.json"
    marker = _read_json(marker_path, "condition checkpoint_complete marker")
    if marker.get("status", "complete") != "complete" or marker.get("complete") is False:
        raise ConditionContractError(f"condition checkpoint is not complete: {marker_path}")
    if marker.get("global_step") != step:
        raise ConditionContractError(
            f"condition checkpoint marker step mismatch: directory={step}, marker={marker.get('global_step')}"
        )
    actual_variant = str(marker.get("variant", ""))
    if expected_variant and actual_variant != expected_variant:
        raise ConditionContractError(
            f"condition checkpoint variant mismatch: expected {expected_variant!r}, got {actual_variant!r}"
        )
    marker_world_size = marker.get("world_size")
    state_path = checkpoint / "trainer_state.json"
    state = _read_json(state_path, "condition trainer_state") if state_path.is_file() else {}
    state_variant = str(state.get("variant", actual_variant))
    if state_variant != actual_variant:
        raise ConditionContractError(
            f"condition trainer_state variant mismatch: marker={actual_variant!r}, state={state_variant!r}"
        )
    state_step = state.get("global_step")
    if state_step is not None and int(state_step) != step:
        raise ConditionContractError(
            f"condition trainer_state step mismatch: directory={step}, state={state_step}"
        )
    state_world_size = state.get("world_size", marker_world_size)
    if state_world_size is None or int(state_world_size) <= 0:
        raise ConditionContractError(f"condition checkpoint has no positive world_size: {checkpoint}")
    if marker_world_size is not None and int(marker_world_size) != int(state_world_size):
        raise ConditionContractError(
            f"condition checkpoint world_size mismatch: marker={marker_world_size}, state={state_world_size}"
        )
    if expected_source_world_size is not None and int(state_world_size) != int(expected_source_world_size):
        raise ConditionContractError(
            f"condition source world_size mismatch: expected {expected_source_world_size}, got {state_world_size}"
        )
    return {
        "path": portable_path(checkpoint),
        "step": step,
        "variant": actual_variant,
        "world_size": int(state_world_size),
        "marker_sha256": _sha256_json(marker_path),
    }


def _assert_standard_qwen_export(export_dir: Path) -> dict[str, Any]:
    if not export_dir.is_dir():
        raise ConditionContractError(f"B condition model must be a standard export directory: {export_dir}")
    raw_shards = sorted(
        child.relative_to(export_dir).as_posix()
        for child in export_dir.rglob("*")
        if child.is_file()
        and (child.name.endswith(".distcp") or child.name.endswith(".metadata"))
    )
    if raw_shards:
        raise ConditionContractError(
            f"B standard export contains raw distributed shards: {raw_shards[:5]}"
        )
    config_path = export_dir / "config.json"
    metadata_path = export_dir / "v3_lora_export.json"
    if not config_path.is_file():
        raise ConditionContractError(f"B standard export is missing config.json: {export_dir}")
    if not metadata_path.is_file():
        raise ConditionContractError(f"B standard export is missing v3_lora_export.json: {export_dir}")
    weight_files = _weight_files(export_dir)
    if not weight_files:
        raise ConditionContractError(f"B standard export has no complete model weights: {export_dir}")
    metadata = _read_json(metadata_path, "B standard export metadata")
    if metadata.get("status") != "complete" or metadata.get("exported_standard_qwen_weights") is not True:
        raise ConditionContractError(f"B export metadata does not declare a complete standard Qwen export: {metadata_path}")
    lora = metadata.get("lora_config")
    if not isinstance(lora, Mapping) or lora.get("enabled") is not True:
        raise ConditionContractError(f"B export metadata does not record the expected enabled LoRA recipe: {metadata_path}")
    try:
        merged_module_count = int(metadata.get("merged_module_count", 0))
    except (TypeError, ValueError) as error:
        raise ConditionContractError(
            f"B export metadata has an invalid merged_module_count: {metadata_path}"
        ) from error
    if merged_module_count <= 0:
        raise ConditionContractError(f"B export metadata records no merged LoRA modules: {metadata_path}")
    if metadata.get("raw_fsdp_shards_rejected") is not True:
        raise ConditionContractError(f"B export metadata does not prove raw FSDP shards were rejected: {metadata_path}")
    # Use safetensors metadata when available.  This reads only the header and
    # catches an export that merely copied LoRA wrapper tensors without loading
    # a multi-GB model into CPU RAM.
    residual = []
    try:
        from safetensors import safe_open  # type: ignore
    except ImportError:  # pragma: no cover - training image provides it
        safe_open = None
    for file_path in weight_files:
        if safe_open is not None and file_path.suffix == ".safetensors":
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                residual.extend(
                    key
                    for key in handle.keys()
                    if ".lora_A" in key or ".lora_B" in key or ".base." in key
                )
        elif file_path.name.endswith(".index.json"):
            index = _read_json(file_path, "B export weight index")
            residual.extend(
                str(key)
                for key in index.get("weight_map", {})
                if ".lora_A" in str(key) or ".lora_B" in str(key) or ".base." in str(key)
            )
    if residual:
        raise ConditionContractError(
            f"B standard export still contains LoRA wrapper keys: {residual[:5]}"
        )
    return {
        "config_sha256": _sha256_file(config_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "weight_files": [portable_path(path) for path in weight_files],
        "metadata": metadata,
    }


def resolve_condition_contract(
    config: Mapping[str, Any],
    *,
    require_artifact: bool = True,
    existing_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve and validate the condition identity for a restoration config."""

    if not condition_enabled(config):
        return None
    raw = _condition_mapping(config)
    assert raw is not None
    root = _workspace_root(config)
    value = dict(raw)
    model_path_text = value.get("model_path") or ""
    checkpoint_text = value.get("checkpoint_path") or ""
    variant = str(value.get("variant") or "")
    role = str(value.get("role") or value.get("variant_group") or "").upper()
    if not role:
        role = "B" if "_B_" in variant or variant.upper().endswith("_B") else "A"
    if role not in {"A", "B"}:
        raise ConditionContractError(f"condition role must be A or B, got {role!r}")
    if not variant:
        raise ConditionContractError("condition_model.variant is required")
    if role == "A" and "_A_" not in variant:
        raise ConditionContractError(f"A condition variant must identify A, got {variant!r}")
    if role == "B" and "_B_" not in variant:
        raise ConditionContractError(f"B condition variant must identify B, got {variant!r}")
    if not model_path_text:
        if require_artifact:
            raise ConditionContractError(
                f"condition_model.model_path is required for gate {role}; B never guesses a latest checkpoint/export"
            )
        # Keep CPU dry-runs useful even when the caller intentionally does not
        # expose a model mount.  The placeholder is never accepted by a real
        # extraction/training/preflight path (require_artifact=True).
        model_path_text = "__unverified_condition_model__"
    model_path = _resolve_workspace_path(model_path_text, root=root)
    checkpoint = None
    checkpoint_info = None
    expected_step = value.get("checkpoint_step")
    expected_world_size = value.get("source_world_size")
    if expected_world_size is None:
        # The published A/B variants carry their source topology in the
        # immutable suffix (for example ``..._A_24GPU``).  Treat that suffix
        # as an explicit expectation rather than accepting any positive world
        # size from a checkpoint marker; B_8GPU therefore remains valid only
        # when the caller explicitly selects that exact variant.
        variant_world = re.search(r"_(\d+)GPU$", variant.upper())
        if variant_world:
            expected_world_size = int(variant_world.group(1))
    if checkpoint_text:
        checkpoint = _resolve_workspace_path(checkpoint_text, root=root)
        if require_artifact or checkpoint.exists():
            checkpoint_info = _checkpoint_info(
                checkpoint,
                expected_variant=variant,
                expected_step=int(expected_step) if expected_step is not None else None,
                expected_source_world_size=(
                    int(expected_world_size) if expected_world_size is not None else None
                ),
            )
        else:
            checkpoint_info = {
                "path": portable_path(checkpoint),
                "step": int(expected_step) if expected_step is not None else None,
                "variant": variant,
                "world_size": int(expected_world_size) if expected_world_size is not None else None,
                "marker_sha256": None,
            }
    elif require_artifact:
        raise ConditionContractError("condition_model.checkpoint_path is required for checkpoint identity")
    if require_artifact and not model_path.exists():
        raise ConditionContractError(f"condition model artifact is missing: {model_path}")
    base_model_text = value.get("base_model_path") or config.get("model", {}).get("pretrained_model_path")
    base_model = _resolve_workspace_path(base_model_text, root=root) if base_model_text else None
    processor_text = value.get("processor_path") or (str(base_model) if base_model else "")
    processor_path = _resolve_workspace_path(processor_text, root=root) if processor_text else None
    # FireRed stores its lightweight Qwen processor under ``<model>/processor``.
    # Hash only that subdirectory for condition identity; hashing the complete
    # FireRed root would unnecessarily traverse the multi-GB Qwen/mmDiT/VAE
    # weights during every preflight.  The extractor already accepts both the
    # FireRed root and the standalone processor directory.
    if processor_path is not None and (processor_path / "processor").is_dir():
        processor_path = processor_path / "processor"
    config_text = value.get("config_path")
    config_path = _resolve_workspace_path(config_text, root=root) if config_text else None
    if config_path is None and base_model is not None:
        candidate = base_model / "text_encoder"
        config_path = candidate if (candidate / "config.json").is_file() else base_model
    if require_artifact:
        if processor_path is None or not processor_path.is_dir():
            raise ConditionContractError(
                f"condition processor directory is missing: {processor_path}"
            )
        if config_path is None or not (config_path / "config.json").is_file():
            raise ConditionContractError(
                f"condition Qwen config.json is missing: {config_path}"
            )
    if require_artifact and role == "B":
        export_info = _assert_standard_qwen_export(model_path)
        recorded_checkpoint = export_info["metadata"].get("checkpoint_metadata")
        if not recorded_checkpoint:
            raise ConditionContractError(
                "B standard export metadata must identify the source checkpoint"
            )
        metadata_root = export_info["metadata"].get("checkpoint_metadata_base")
        recorded_root = model_path if metadata_root == "export_directory" else root
        recorded_path = _resolve_workspace_path(recorded_checkpoint, root=recorded_root)
        if checkpoint is None or portable_path(recorded_path) != portable_path(checkpoint):
            raise ConditionContractError(
                "B standard export checkpoint identity differs from condition checkpoint: "
                f"export={portable_path(recorded_path)!r}, checkpoint={portable_path(checkpoint)!r}"
            )
    else:
        export_info = None
        if require_artifact:
            weights = _weight_files(model_path)
            if len(weights) == 0:
                raise ConditionContractError(f"A merged condition model has no model.safetensors: {model_path}")
            # A's known merged_model intentionally has no config/processor;
            # those are supplied separately from the frozen FireRed base.
            if model_path.is_dir() and any(
                child.name.endswith(".distcp") or child.name.endswith(".metadata")
                for child in model_path.rglob("*")
                if child.is_file()
            ):
                raise ConditionContractError(f"A condition model contains raw distributed shards: {model_path}")
    declared_hash = value.get("model_hash")
    model_hash = None
    if require_artifact:
        # Reuse a preflight hash only if it refers to the same model path and
        # size signature.  This avoids hashing A's 21 GB file twice while still
        # invalidating a replaced artifact.
        old_identity = dict(existing_contract or {}).get("identity", {})
        old_model = dict(old_identity.get("model", {}))
        stat_token = _directory_signature(model_path, include_content=False)
        if (
            old_model.get("portable_path") == portable_path(model_path)
            and old_model.get("stat_signature") == stat_token
            and old_model.get("sha256")
        ):
            model_hash = str(old_model["sha256"])
        else:
            model_hash = _identity_hash(model_path, declared_hash)
    prompt = config.get("prompt_library", config.get("prompts", {}))
    if isinstance(prompt, Mapping):
        prompt_id = prompt.get("id") or prompt.get("prompt_id")
        prompt_hash = prompt.get("fingerprint") or prompt.get("sha256")
    else:
        prompt_id = None
        prompt_hash = None
    extraction = {
        "hidden_state_layer": -1,
        "valid_token_prefix_trim": int(value.get("valid_token_prefix_trim", 64)),
        "dtype": str(value.get("dtype", "bfloat16")),
        "disable_inverse": True,
        "t2i_mode": False,
        "condition_prompt_only": True,
        "lm_head_used": False,
        "text_generation_used": False,
        "routing": False,
        "restoration_forward_per_iteration": 1,
    }
    identity: dict[str, Any] = {
        "role": role,
        "variant": variant,
        "checkpoint": checkpoint_info,
        "model": {
            "portable_path": portable_path(model_path),
            "sha256": model_hash,
            "stat_signature": _directory_signature(model_path, include_content=False) if model_path.exists() else None,
        },
        "base_model": portable_path(base_model),
        "processor": {
            "portable_path": portable_path(processor_path),
            "sha256": _directory_signature(processor_path) if require_artifact and processor_path and processor_path.exists() else None,
        },
        "config": {
            "portable_path": portable_path(config_path),
            "sha256": _sha256_file(config_path / "config.json") if require_artifact and config_path and (config_path / "config.json").is_file() else None,
        },
        "export": export_info,
    }
    payload: dict[str, Any] = {
        "schema": CONDITION_SCHEMA_VERSION,
        "kind": CONDITION_KIND,
        "identity": identity,
        "prompt": {"id": prompt_id, "sha256": prompt_hash},
        "extraction": extraction,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    # Keep an explicit cache alias for callers that already use the recipe
    # contract vocabulary.  It is deliberately identical to the full
    # condition fingerprint (model/checkpoint/prompt/extraction included),
    # never to a base-model-only recipe fingerprint.
    payload["cache_fingerprint"] = payload["fingerprint"]
    payload["status"] = "complete" if require_artifact else "unverified"
    # Flat aliases make row/checkpoint validation readable and keep the marker
    # useful to shell tools without requiring callers to understand the full
    # nested identity payload.
    payload.update(
        {
            "condition_variant": variant,
            "condition_role": role,
            "condition_model_hash": model_hash,
            "condition_checkpoint_step": checkpoint_info.get("step") if checkpoint_info else expected_step,
            "condition_checkpoint_world_size": checkpoint_info.get("world_size") if checkpoint_info else expected_world_size,
            "condition_model_path": portable_path(model_path),
            "condition_processor_path": portable_path(processor_path),
        }
    )
    return payload


def assert_condition_contract(actual: Mapping[str, Any], expected: Mapping[str, Any], *, context: str = "condition cache") -> None:
    """Compare the immutable condition identity used by embeddings/resume."""

    keys = (
        "schema",
        "kind",
        "fingerprint",
        "cache_fingerprint",
        "condition_variant",
        "condition_role",
        "condition_model_hash",
        "condition_checkpoint_step",
        "condition_checkpoint_world_size",
        "condition_model_path",
        "prompt",
        "extraction",
    )
    mismatches = {
        key: (actual.get(key), expected.get(key))
        for key in keys
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ConditionContractError(f"{context} identity mismatch: {mismatches}")


def write_condition_contract(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_condition_contract(path: Path) -> dict[str, Any]:
    return _read_json(path, "condition contract marker")


__all__ = [
    "CONDITION_KIND",
    "CONDITION_SCHEMA_VERSION",
    "ConditionContractError",
    "adapt_workspace_path",
    "assert_condition_contract",
    "condition_enabled",
    "normalize_condition_config",
    "portable_path",
    "read_condition_contract",
    "resolve_condition_contract",
    "write_condition_contract",
]
