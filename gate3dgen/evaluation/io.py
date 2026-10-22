from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


VARIANTS = ("degraded", "restored", "gt", "albedo")


def available_variants(case: dict[str, Any], requested: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return image variants declared by a legacy or official condition row."""
    image_paths = case.get("image_paths") if isinstance(case.get("image_paths"), dict) else {}
    conditions = case.get("conditions") if isinstance(case.get("conditions"), list) else []
    declared = tuple(
        str(item["condition_id"])
        for item in conditions
        if isinstance(item, dict) and item.get("condition_id") and item.get("path")
    )
    candidates = tuple(requested) if requested is not None else (declared or VARIANTS)
    return tuple(name for name in candidates if case.get(name) or image_paths.get(name))


def variant_path(case: dict[str, Any], variant: str) -> str | None:
    """Resolve one image condition without flattening the long-form schema."""
    direct = case.get(variant)
    if direct:
        return str(direct)
    image_paths = case.get("image_paths")
    if isinstance(image_paths, dict) and image_paths.get(variant):
        return str(image_paths[variant])
    for condition in case.get("conditions", ()):
        if isinstance(condition, dict) and condition.get("condition_id") == variant:
            return str(condition["path"]) if condition.get("path") else None
    return None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: JSON value is not an object")
                yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_under(root: Path, relative: str) -> Path:
    # os.path.abspath is lexical: unlike Path.resolve(), it does not issue a
    # remote stat for every path component on network-mounted datasets.
    root_absolute = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(root_absolute / relative))
    try:
        candidate.relative_to(root_absolute)
    except ValueError as error:
        raise ValueError(f"path escapes data root: {relative}") from error
    return candidate


def finite_json(value: Any) -> Any:
    """Convert numpy values and non-finite floats to strict-JSON values."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value
