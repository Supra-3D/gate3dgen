"""Shared contracts for the opt-in albedo training recipes.

The original V1/V2 jobs intentionally do not import this module from their
data path.  Keeping the recipe names and fingerprints here makes it possible
to reject accidental cross-recipe resumes while preserving the old command
line defaults.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


RECIPE_V1 = "v1"
RECIPE_V2 = "v2"
RECIPE_ALBEDO_AWARE = "albedo_aware"
RECIPE_ALBEDO_CONSISTENCY = "albedo_consistency"
RECIPE_RGB_ALBEDO_MULTITASK = "rgb_albedo_multitask"
# Human-facing aliases retained for manifests/tools that label the three
# experiments simply as A/B/C.
RECIPE_ALBEDO_A = RECIPE_ALBEDO_AWARE
RECIPE_ALBEDO_B = RECIPE_ALBEDO_CONSISTENCY
RECIPE_ALBEDO_C = RECIPE_RGB_ALBEDO_MULTITASK
RECIPE_ALIASES = {
    "A": RECIPE_ALBEDO_AWARE,
    "a": RECIPE_ALBEDO_AWARE,
    "albedo_a": RECIPE_ALBEDO_AWARE,
    "B": RECIPE_ALBEDO_CONSISTENCY,
    "b": RECIPE_ALBEDO_CONSISTENCY,
    "albedo_b": RECIPE_ALBEDO_CONSISTENCY,
    "C": RECIPE_RGB_ALBEDO_MULTITASK,
    "c": RECIPE_RGB_ALBEDO_MULTITASK,
    "albedo_c": RECIPE_RGB_ALBEDO_MULTITASK,
}

ALBEDO_RECIPES = (
    RECIPE_ALBEDO_AWARE,
    RECIPE_ALBEDO_CONSISTENCY,
    RECIPE_RGB_ALBEDO_MULTITASK,
)
KNOWN_RECIPES = (RECIPE_V1, RECIPE_V2, *ALBEDO_RECIPES)
RECIPE_SCHEMA_VERSION = "texverse-albedo-recipe-v1"
CACHE_SCHEMA_VERSION = "texverse-albedo-cache-v1"
ALBEDO_AUX_STATE_PREFIX = "albedo_aux_head."


def _normalize_relative(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\\", "/")
    return text or None


def _row_dimensions(row: dict[str, Any]) -> list[int] | None:
    """Return dimensions in a stable representation for cache identity."""

    dimensions = row.get("dimensions")
    if isinstance(dimensions, (list, tuple)) and len(dimensions) >= 2:
        try:
            return [int(dimensions[0]), int(dimensions[1])]
        except (TypeError, ValueError):
            return [str(dimensions[0]), str(dimensions[1])]  # type: ignore[list-item]
    for key in ("edit_image_size", "source_image_size"):
        value = row.get(key)
        if isinstance(value, dict):
            if "width" in value and "height" in value:
                return [int(value["width"]), int(value["height"])]
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, dict) and "width" in first and "height" in first:
                return [int(first["width"]), int(first["height"])]
    return None


def _explicit_dimensions(value: Any) -> list[int] | None:
    if isinstance(value, dict):
        value = [value.get("width"), value.get("height")]
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [int(value[0]), int(value[1])]
    except (TypeError, ValueError):
        return None


def _row_sha_fields(row: dict[str, Any]) -> dict[str, str]:
    """Keep every raw manifest SHA field, including the canonical ``sha256``."""

    result: dict[str, str] = {}
    for key, value in row.items():
        if key == "sha256" or key.endswith("_sha256"):
            result[str(key)] = str(value)
    # Some prepared tools put all source hashes under a map.  Preserve that
    # map too without trusting it as a path or replacing the raw fields.
    raw = row.get("raw_sha256")
    if isinstance(raw, dict):
        for key, value in sorted(raw.items()):
            result[f"raw_sha256.{key}"] = str(value)
    return dict(sorted(result.items()))


def _manifest_sha256(path: Any) -> str | None:
    """Hash the raw source manifest when available; missing dry-run inputs stay optional."""

    if path is None:
        return None
    candidate = Path(os.fspath(path)).expanduser()
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest_sha256(
    config: dict[str, Any], manifest: Any = None
) -> str | None:
    data = config.get("data", {})
    value = manifest or data.get("manifest") or (data.get("manifests", {}) or {}).get("train")
    return _manifest_sha256(value)


def is_albedo_recipe(recipe: str | None) -> bool:
    return str(recipe or "") in ALBEDO_RECIPES


def validate_recipe_name(recipe: str) -> str:
    recipe = RECIPE_ALIASES.get(str(recipe), str(recipe))
    if recipe not in KNOWN_RECIPES:
        raise ValueError(f"unknown training recipe {recipe!r}; expected one of {KNOWN_RECIPES}")
    return recipe


def recipe_from_config(config: dict[str, Any]) -> str:
    """Return the explicit recipe, retaining legacy semantics when omitted."""

    recipe = config.get("recipe")
    if recipe:
        return RECIPE_ALIASES.get(str(recipe), str(recipe))
    # Existing configs are deliberately mapped to names only for provenance;
    # no V1/V2 code path changes behavior based on this value.
    return RECIPE_V2 if config.get("model", {}).get("architecture") == "restoration_custom" else RECIPE_V1


def fingerprint_payload(
    recipe: str,
    config: dict[str, Any],
    rows: Iterable[dict[str, Any]] = (),
    *,
    source_manifest_sha256_value: str | None = None,
) -> dict[str, Any]:
    """Build a stable, JSON-only fingerprint payload.

    Only values which affect data/forward semantics are included.  Absolute
    machine paths are excluded so the same prepared cache can be moved across
    distributed mounts without changing its identity.
    """

    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})
    loss = config.get("loss", {})
    prompt_library = config.get("prompt_library", config.get("prompts", {}))
    if isinstance(prompt_library, str):
        prompt_library = {"path": prompt_library}
    if not isinstance(prompt_library, dict):
        prompt_library = {}
    if source_manifest_sha256_value is None:
        source_manifest_sha256_value = source_manifest_sha256(config)
    row_ids = []
    for row in rows:
        dimensions = _row_dimensions(row)
        row_ids.append(
            {
                "pair_id": str(row.get("pair_id", "")),
                "asset_id": str(row.get("asset_id", row.get("sha256", ""))),
                "task_id": str(row.get("task_id", "")),
                # The three relative paths identify the exact degraded/GT/
                # albedo render used by the recipe.  Absolute historical
                # mounts are intentionally excluded.
                "degraded_relative_path": _normalize_relative(row.get("degraded_relative_path")),
                "gt_relative_path": _normalize_relative(row.get("gt_relative_path")),
                "source_relative_path": _normalize_relative(row.get("source_relative_path")),
                "albedo_relative_path": _normalize_relative(
                    row.get("albedo_relative_path") or row.get("source_relative_path")
                ),
                "subject_mask_relative_path": _normalize_relative(
                    row.get("subject_mask_relative_path")
                ),
                "subject_mask_schema": row.get("subject_mask_schema"),
                "raw_sha256": _row_sha_fields(row),
                "dimensions": dimensions,
                "source_dimensions": _explicit_dimensions(row.get("source_dimensions")),
                "albedo_dimensions": _explicit_dimensions(row.get("albedo_dimensions")),
            }
        )
    row_ids.sort(
        key=lambda value: (
            value["pair_id"],
            value["source_relative_path"] or "",
            value["degraded_relative_path"] or "",
            value["gt_relative_path"] or "",
        )
    )
    return {
        "schema": RECIPE_SCHEMA_VERSION,
        "recipe": str(recipe),
        "source_manifest_sha256": source_manifest_sha256_value,
        # A manifest's absolute mount prefix is not part of its identity.  The
        # relative path and row IDs are enough to detect stale caches while
        # allowing the same cache recipe to be mounted on distributed nodes.
        "data_manifest": str(
            data.get("manifest")
            or (data.get("manifests", {}) or {}).get("train", "")
        ).replace("\\", "/").split("/scripts/", 1)[-1],
        "data_root_name": str(data.get("data_root", "")).rstrip("/").split("/")[-1],
        "albedo_root_name": str(data.get("albedo_root", "")).rstrip("/").split("/")[-1],
        "subject_mask_schema": data.get("subject_mask_schema"),
        "model_architecture": model.get("architecture", "stock"),
        "trainable_block_range": list(training.get("trainable_block_range", [29, 49])),
        "loss": loss,
        # Prompt content changes the offline VLM embeddings.  Keep only the
        # stable selected ID/file digest in the portable fingerprint; absolute
        # mount paths must not make a cache non-portable.
        "prompt_library": {
            "id": prompt_library.get("id") or prompt_library.get("prompt_id"),
            "fingerprint": prompt_library.get("fingerprint")
            or prompt_library.get("sha256"),
        },
        "rows": row_ids,
    }


def recipe_fingerprint(
    recipe: str,
    config: dict[str, Any],
    rows: Iterable[dict[str, Any]] = (),
    *,
    source_manifest: str | None = None,
) -> str:
    payload = fingerprint_payload(
        recipe,
        config,
        rows,
        source_manifest_sha256_value=source_manifest_sha256(config, source_manifest),
    )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_contract(
    config: dict[str, Any],
    rows: Iterable[dict[str, Any]] = (),
    *,
    prepared_manifest: str | None = None,
    source_manifest: str | None = None,
) -> dict[str, Any]:
    """Build a portable cache contract for prepared rows and embeddings."""

    recipe = validate_recipe_name(recipe_from_config(config))
    rows = list(rows)
    manifest_sha256 = source_manifest_sha256(config, source_manifest)
    fingerprint = recipe_fingerprint(
        recipe,
        config,
        rows,
        source_manifest=source_manifest,
    )
    schema_fingerprint = hashlib.sha256(
        f"{CACHE_SCHEMA_VERSION}:{RECIPE_SCHEMA_VERSION}:{recipe}:{fingerprint}".encode()
    ).hexdigest()
    return {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "recipe_schema": RECIPE_SCHEMA_VERSION,
        "recipe": recipe,
        "recipe_fingerprint": fingerprint,
        "schema_fingerprint": schema_fingerprint,
        "cache_fingerprint": fingerprint,
        "row_count": len(rows),
        "prepared_manifest": prepared_manifest,
        "source_manifest_sha256": manifest_sha256,
    }


def write_cache_contract(path: os.PathLike[str] | str, contract: dict[str, Any]) -> None:
    """Atomically write a cache contract marker."""

    destination = os.fspath(path)
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(destination) + ".", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(contract, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_cache_contract(path: os.PathLike[str] | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"cache contract is not an object: {path}")
    return value


def assert_cache_contract(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    context: str = "cache",
) -> None:
    """Reject stale recipe/schema/fingerprint caches before training."""

    keys = (
        "cache_schema",
        "recipe_schema",
        "recipe",
        "recipe_fingerprint",
        "cache_fingerprint",
        "schema_fingerprint",
        "row_count",
        "source_manifest_sha256",
    )
    mismatches = {
        key: (actual.get(key), expected.get(key))
        for key in keys
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(f"{context} recipe/schema incompatibility: {mismatches}")


def checkpoint_recipe_marker(config: dict[str, Any], args: Any | None = None) -> dict[str, Any]:
    """Return compact recipe metadata suitable for checkpoint_complete.json."""

    recipe = recipe_from_config(config) if isinstance(config, dict) else str(getattr(args, "recipe", ""))
    marker = {
        "recipe_schema": RECIPE_SCHEMA_VERSION,
        "recipe": recipe,
        "recipe_fingerprint": getattr(args, "recipe_fingerprint", None),
        "cache_schema": CACHE_SCHEMA_VERSION,
        "cache_fingerprint": getattr(args, "cache_fingerprint", None),
        "schema_fingerprint": getattr(args, "schema_fingerprint", None),
        "aux_state_prefix": ALBEDO_AUX_STATE_PREFIX,
    }
    if args is not None:
        marker["albedo_root"] = getattr(args, "albedo_root", None)
    return marker
