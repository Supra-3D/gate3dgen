"""Small, deterministic prompt-library adapter for restoration manifests.

The library is intentionally selected as one named prompt rather than mixed
into a collection of task-specific prompts.  Its file digest and selected ID
are carried into prepared rows/cache fingerprints so an embedding cache cannot
silently outlive a prompt change.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .data_defaults import INSTRUCTION_CN, INSTRUCTION_EN


DEFAULT_PROMPT_LIBRARY = Path(__file__).resolve().parent.parent / "resources/prompts/purifier.json"


def _library_config(config: dict[str, Any] | None) -> tuple[str | None, str | None]:
    config = config or {}
    value = config.get("prompt_library", config.get("prompt", config.get("prompts")))
    if value is None:
        return None, None
    if isinstance(value, str):
        return value, None
    if not isinstance(value, dict):
        raise ValueError("prompt_library must be a path or mapping")
    path = value.get("path") or value.get("library") or value.get("file")
    prompt_id = value.get("id") or value.get("prompt_id")
    return (str(path) if path else None), (str(prompt_id) if prompt_id else None)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        # Project configs use workspace-relative ``scripts/...`` paths.
        workspace = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()
        candidate = workspace / candidate
    return candidate.resolve()


def load_prompt_library(
    config: dict[str, Any] | None = None,
    *,
    path: str | Path | None = None,
    prompt_id: str | None = None,
) -> dict[str, Any] | None:
    """Read and validate one prompt-library JSON object.

    ``None`` means the legacy fixed prompt remains active.  When a library is
    requested, missing/invalid files are fatal so that the user never gets a
    false claim that v3 conditioning was used.
    """

    configured_path, configured_id = _library_config(config)
    path = path or configured_path
    prompt_id = prompt_id or configured_id
    if path is None:
        return None
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"prompt library does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prompt library: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError("prompt library root must be an object")
    prompts = payload.get("restoration_prompts")
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError("prompt library must define non-empty restoration_prompts")
    selected = prompt_id or payload.get("recommended_default_prompt_id")
    if not isinstance(selected, str) or selected not in prompts:
        raise ValueError(
            f"unknown prompt id {selected!r}; available={sorted(str(k) for k in prompts)}"
        )
    entry = prompts[selected]
    if isinstance(entry, str):
        english = entry
    elif isinstance(entry, dict):
        english = entry.get("prompt")
    else:
        english = None
    if not isinstance(english, str) or not english.strip():
        raise ValueError(f"prompt {selected!r} has no non-empty prompt string")
    # v3 declares its actual conditioning contract explicitly as
    # restoration_prompt + output_specification +
    # global_preservation_constraint.  Loading only entry["prompt"] would make
    # the configured library appear active while silently dropping two of its
    # three required instruction components.
    output_specification = payload.get("output_specification")
    preservation_constraint = payload.get("global_preservation_constraint")
    assembly = payload.get("assembly")
    if isinstance(assembly, dict) and assembly.get("formula"):
        for field, value in (
            ("output_specification", output_specification),
            ("global_preservation_constraint", preservation_constraint),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"prompt library assembly requires non-empty {field}"
                )
        english = "\n\n".join(
            part.strip()
            for part in (english, output_specification, preservation_constraint)
        )
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {
        "path": str(resolved),
        "id": selected,
        "prompt": english,
        "fingerprint": digest,
        "version": payload.get("version"),
    }


def restoration_instructions(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return selected EN/CN instructions plus reproducibility metadata."""

    selected = load_prompt_library(config)
    if selected is None:
        return {
            "instruction": INSTRUCTION_EN,
            "instruction_cn": INSTRUCTION_CN,
            "prompt_library": None,
            "prompt_library_id": None,
            "prompt_library_fingerprint": None,
        }
    return {
        "instruction": selected["prompt"],
        # v3 currently provides no translated universal prompt.  Reusing the
        # fully assembled English instruction for both cache slots guarantees
        # every non-droptext sample follows the requested v3 contract instead
        # of sending roughly half of batches through the unrelated legacy CN
        # instruction.
        "instruction_cn": selected["prompt"],
        "prompt_library": selected["path"],
        "prompt_library_id": selected["id"],
        "prompt_id": selected["id"],
        "prompt_library_version": selected["version"],
        "prompt_library_fingerprint": selected["fingerprint"],
    }


def prompt_metadata(config: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = restoration_instructions(config)
    return {
        key: selected[key]
        for key in (
            "prompt_library",
            "prompt_library_id",
            "prompt_library_fingerprint",
        )
    }


__all__ = [
    "DEFAULT_PROMPT_LIBRARY",
    "load_prompt_library",
    "restoration_instructions",
    "prompt_metadata",
]

