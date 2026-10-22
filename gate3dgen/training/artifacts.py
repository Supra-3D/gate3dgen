from __future__ import annotations

import json
import re
from pathlib import Path

from .experimental.recipes import CACHE_SCHEMA_VERSION, RECIPE_SCHEMA_VERSION, validate_recipe_name


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def adapter_dir(checkpoint_or_adapter: Path) -> Path:
    path = checkpoint_or_adapter.resolve()
    if path.is_file():
        return path.parent
    nested = path / "adapter"
    if (nested / "adapter_model.safetensors").is_file():
        return nested
    if (path / "adapter_model.safetensors").is_file():
        return path
    raise FileNotFoundError(f"No PEFT adapter weights under {checkpoint_or_adapter}")


def checkpoint_artifact_kind(path: Path) -> str:
    path = path.resolve()
    if path.is_file() and path.name in {"adapter_model.safetensors", "adapter_model.bin"}:
        return "adapter"
    try:
        adapter_dir(path)
        return "adapter"
    except FileNotFoundError:
        pass
    if (
        (path / "pytorch_model_fsdp.bin").is_file()
        or (path / "diffusion_pytorch_model.safetensors").is_file()
        or ((path / "pytorch_model_fsdp_0/.metadata").is_file()
            and bool(list((path / "pytorch_model_fsdp_0").glob("*.distcp"))))
    ):
        return "transformer"
    transformer_dir = path / "transformer"
    if transformer_dir.is_dir() and (
        (transformer_dir / "config.json").is_file()
        or list(transformer_dir.glob("*.safetensors"))
    ):
        return "transformer"
    raise FileNotFoundError(f"No adapter or transformer weights under {path}")


def checkpoint_step(path: Path) -> int:
    match = _CHECKPOINT_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a checkpoint directory: {path}")
    return int(match.group(1))


def checkpoint_is_complete(path: Path) -> bool:
    try:
        step = checkpoint_step(path)
        artifact_kind = checkpoint_artifact_kind(path)
        marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    declared_kind = marker.get("artifact_kind", artifact_kind)
    if marker.get("fsdp_state_dict_type") == "SHARDED_STATE_DICT":
        for dirname in ("pytorch_model_fsdp_0", "optimizer_0"):
            if not (path / dirname / ".metadata").is_file():
                return False
            if any(not (path / dirname / f"__{rank}_0.distcp").is_file()
                   for rank in range(int(marker.get("world_size", 0)))):
                return False
    return (
        marker.get("global_step") == step
        and int(marker.get("world_size", 0)) > 0
        and declared_kind == artifact_kind
    )


def resolve_checkpoint(output_dir: Path, value: str, expected_kind: str | None = None) -> Path:
    if value != "latest":
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = output_dir / value
        kind = checkpoint_artifact_kind(candidate)
        if expected_kind is not None and kind != expected_kind:
            raise ValueError(f"Expected {expected_kind} checkpoint, got {kind}: {candidate}")
        return candidate.resolve()
    candidates = [
        path for path in output_dir.glob("checkpoint-*")
        if checkpoint_is_complete(path)
        and (expected_kind is None or checkpoint_artifact_kind(path) == expected_kind)
    ]
    if not candidates:
        raise FileNotFoundError(f"No complete checkpoint under {output_dir}")
    return max(candidates, key=checkpoint_step).resolve()


def assert_checkpoint_recipe_compatible(
    checkpoint: Path,
    *,
    recipe: str | None = None,
    recipe_fingerprint: str | None = None,
    cache_fingerprint: str | None = None,
    schema_fingerprint: str | None = None,
    recipe_schema: str = RECIPE_SCHEMA_VERSION,
    cache_schema: str = CACHE_SCHEMA_VERSION,
) -> dict:
    """Validate a resume marker before loading optimizer/model state.

    A checkpoint from another albedo variant can have perfectly compatible
    tensor shapes while carrying a different target semantics.  The explicit
    marker therefore fails closed before Accelerate mutates the live model.
    Legacy V1/V2 checkpoints remain compatible when no recipe is requested.
    """

    marker_path = checkpoint / "checkpoint_complete.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint has no valid completion marker: {checkpoint}") from error
    if recipe is None:
        return marker
    recipe = validate_recipe_name(str(recipe))
    expected = {
        "recipe": str(recipe),
        "recipe_schema": str(recipe_schema),
        "cache_schema": str(cache_schema),
    }
    if recipe_fingerprint is not None:
        expected["recipe_fingerprint"] = str(recipe_fingerprint)
    if cache_fingerprint is not None:
        expected["cache_fingerprint"] = str(cache_fingerprint)
    if schema_fingerprint is not None:
        expected["schema_fingerprint"] = str(schema_fingerprint)
    mismatches = {
        key: (marker.get(key), value)
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"checkpoint/resume recipe incompatibility for {checkpoint}: {mismatches}"
        )
    return marker
