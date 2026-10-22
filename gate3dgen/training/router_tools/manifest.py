#!/usr/bin/env python3
"""Build the auditable one-image Qwen repair-gate V2 manifests.

The restoration manifests contain *pairs*.  The Qwen gate has a deliberately
different contract: every training/evaluation record contains exactly one
image and one ground-truth text answer.  This builder expands each pair into
independent ``degraded``/``gt_clean`` records and adds one deduplicated
``albedo_render`` record for each source image.

The builder is intentionally conservative.  It never edits an input
manifest, refuses absolute paths outside the configured workspace, checks
expected hashes and decodability, and refuses to publish output when any of
the five split-leakage keys intersect.  ``*_heavy`` task names and the source
``severity`` field are retained as provenance only; they are never used as a
visual severity score.

Typical invocation::

    python tools/build_qwen_gate_manifest.py

The default output is a new ``router`` directory next to the
quality-v11 manifests.  Use ``--no-image-validation`` only for a tiny schema
fixture; formal dataset builds must keep the default fail-closed validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    from PIL import Image
except ImportError:  # pragma: no cover - gives a useful error in the CLI
    Image = None  # type: ignore[assignment]


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()

SCHEMA_VERSION = "qwen-repair-gate-v2"
IMAGE_KINDS = ("degraded", "gt_clean", "albedo_render")
PORTABLE_IMAGE_ROOTS = ("workspace", "texverse", "texverse_full")
DECISIONS = ("REPAIR", "KEEP")

# This order is part of V2's reproducible answer contract.  In particular,
# EXTERNAL_HUMAN_SUPPORT is a real additional reason, not a replacement for
# the task reason and not the MIXED bucket.
REASON_ORDER = (
    "MOTION_BLUR",
    "OPTICAL_BLUR",
    "DIGITAL_DEGRADATION",
    "UNDEREXPOSURE",
    "OVEREXPOSURE",
    "HARD_SHADOW",
    "WHITE_BALANCE",
    "COLOR_SPILL",
    "VEILING_GLARE",
    "DIRTY_GLASS",
    "GLASS_REFLECTION",
    "CROP_OR_MISSING_REGION",
    "EXTERNAL_HUMAN_SUPPORT",
    "OTHER_VISIBLE_DEGRADATION",
    "MIXED",
    "NONE",
)
REASON_INDEX = {name: index for index, name in enumerate(REASON_ORDER)}

TASK_REASON_MAPPING = {
    "animal_motion_blur_heavy": "MOTION_BLUR",
    "motion_blur_heavy": "MOTION_BLUR",
    "optical_blur_heavy": "OPTICAL_BLUR",
    "digital_degradation_heavy": "DIGITAL_DEGRADATION",
    "underexposure_heavy": "UNDEREXPOSURE",
    "overexposure_heavy": "OVEREXPOSURE",
    "shadow_hard_heavy": "HARD_SHADOW",
    "white_balance_cool_heavy": "WHITE_BALANCE",
    "local_color_spill_heavy": "COLOR_SPILL",
    "veiling_glare_heavy": "VEILING_GLARE",
    "dirty_glass_heavy": "DIRTY_GLASS",
    "glass_reflection_heavy": "GLASS_REFLECTION",
    "crop_heavy": "CROP_OR_MISSING_REGION",
}

LEAKAGE_FIELDS = {
    "asset": "asset",
    "pair": "pair",
    "degraded_sha256": "degraded_sha256",
    "gt_sha256": "gt_sha256",
    "source_sha256": "source_sha256",
}

# Private-dataset exceptions are not inherited by new datasets.
EXCLUDED_MISSING_TRAIN_ALBEDO_SHA256 = frozenset()

_HEX64 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class GateManifestError(ValueError):
    """Raised when the gate manifest cannot be published safely."""


def _absolute_norm(path: Path | str) -> Path:
    """Normalize an absolute path without following network symlinks.

    ``Path.resolve`` performs a filesystem read for every parent component;
    that is surprisingly expensive on the workspace's network mount and can
    raise ``EINVAL`` for a transient directory entry.  We still enforce
    lexical root boundaries below, and the final file check/hash/decode is
    fail-closed.
    """

    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateManifestError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise GateManifestError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [row for _line, row in _jsonl(path)]


def _require(row: Mapping[str, Any], key: str, context: str) -> Any:
    value = row.get(key)
    if value is None or value == "":
        raise GateManifestError(f"{context}: missing {key}")
    return value


def _as_hash(value: Any, *, context: str, name: str) -> str:
    text = str(_require({name: value}, name, context)).lower()
    if not _HEX64.fullmatch(text):
        raise GateManifestError(f"{context}: {name} is not a SHA256: {value!r}")
    return text


def _resolve_under(path: Path, root: Path, *, context: str) -> Path:
    """Resolve a path and reject traversal or paths outside ``root``."""

    root = _absolute_norm(root.expanduser())
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = _absolute_norm(candidate)
    try:
        if os.path.commonpath((str(candidate), str(root))) != str(root):
            raise ValueError
    except (ValueError, OSError) as exc:
        raise GateManifestError(
            f"{context}: path escapes configured root {root}: {candidate}"
        ) from exc
    return candidate


def _workspace_relative(path: Path, workspace_root: Path, *, context: str) -> str:
    try:
        path = _absolute_norm(path)
        workspace_root = _absolute_norm(workspace_root)
        if os.path.commonpath((str(path), str(workspace_root))) != str(workspace_root):
            raise ValueError
        return Path(os.path.relpath(path, workspace_root)).as_posix()
    except (ValueError, OSError) as exc:
        raise GateManifestError(
            f"{context}: image is outside workspace and cannot be portable: {path}"
        ) from exc


def _relative_from_value(
    value: Any,
    *,
    absolute_value: Any = None,
    fallback_root: Path | None,
    workspace_root: Path,
    context: str,
) -> Path:
    """Resolve a manifest path while accepting either absolute or relative refs."""

    # Prefer a supplied absolute alias when available.  Calling ``Path.is_file``
    # for every pair makes a full build spend minutes doing one metadata RPC
    # per row on the workspace NFS; the subsequent hash/open operation is
    # already fail-closed and reports a missing path precisely.  Relative-only
    # fixtures use the configured fallback root.
    if absolute_value not in (None, ""):
        absolute_path = Path(str(absolute_value)).expanduser()
        if absolute_path.is_absolute():
            return _absolute_norm(absolute_path)
    if value not in (None, ""):
        value_path = Path(str(value)).expanduser()
        if value_path.is_absolute():
            return _absolute_norm(value_path)
        if fallback_root is not None:
            return _absolute_norm(fallback_root / value_path)
    raise GateManifestError(f"{context}: no image path supplied")


def _hash_and_decode(
    path: Path,
    *,
    expected_sha256: str | None,
    context: str,
    validate_images: bool,
    cache: dict[Path, tuple[str, tuple[int, int]]],
) -> tuple[str, tuple[int, int]]:
    # Keep the same lexical path key used by the pre-validation cache.  A
    # realpath lookup here is both unnecessary (the root check is lexical and
    # the file is opened/hash-checked below) and prohibitively slow on the
    # network-mounted formal dataset.
    resolved = _absolute_norm(path)
    if resolved in cache:
        actual, dimensions = cache[resolved]
    else:
        digest = hashlib.sha256()
        try:
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
        except OSError as exc:
            raise GateManifestError(f"{context}: image cannot be opened/hashed {resolved}: {exc}") from exc
        dimensions = (0, 0)
        if validate_images:
            if Image is None:
                raise GateManifestError("Pillow is required for image validation")
            try:
                with Image.open(resolved) as image:
                    dimensions = (int(image.width), int(image.height))
                    if dimensions[0] <= 0 or dimensions[1] <= 0:
                        raise GateManifestError(f"{context}: image has invalid dimensions")
                    # Verify in the same open.  Re-opening every PNG just to
                    # read width/height doubles NFS metadata and page-cache
                    # traffic for the formal ~100k-image build.
                    image.verify()
            except GateManifestError:
                raise
            except Exception as exc:
                raise GateManifestError(
                    f"{context}: image failed decode/verify: {resolved}: {exc}"
                ) from exc
        cache[resolved] = (actual, dimensions)
    if expected_sha256 is not None:
        expected = _as_hash(expected_sha256, context=context, name="expected_sha256")
        if actual != expected:
            raise GateManifestError(
                f"{context}: SHA256 mismatch for {resolved}: expected {expected}, got {actual}"
            )
    return actual, dimensions


def _task_reason(row: Mapping[str, Any], *, context: str) -> str:
    task_id = str(_require(row, "task_id", context))
    try:
        return TASK_REASON_MAPPING[task_id]
    except KeyError as exc:
        raise GateManifestError(
            f"{context}: unsupported task_id {task_id!r}; add an explicit V2 mapping"
        ) from exc


def _sidecar_is_extra(row: Mapping[str, Any], base_pair_ids: set[str]) -> bool:
    return str(_require(row, "pair_id", "hand sidecar")) not in base_pair_ids


def _sidecar_membership(
    sidecar_rows: Sequence[Mapping[str, Any]],
    *,
    base_pair_ids: set[str],
    train_asset_ids: set[str],
    test_pair_ids: set[str],
) -> tuple[dict[str, Mapping[str, Any]], set[str], set[str]]:
    by_pair: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(sidecar_rows, start=1):
        context = f"hand sidecar:{index}"
        pair_id = str(_require(row, "pair_id", context))
        if pair_id in by_pair:
            raise GateManifestError(f"{context}: duplicate pair_id {pair_id}")
        by_pair[pair_id] = row
        split = str(row.get("split") or "train")
        if split != "train":
            raise GateManifestError(f"{context}: hand sidecar split must be train, got {split!r}")
        if pair_id in test_pair_ids:
            raise GateManifestError(f"{context}: sidecar pair leaks into validation: {pair_id}")
        asset = str(row.get("sha256") or "")
        if asset and asset not in train_asset_ids:
            raise GateManifestError(f"{context}: sidecar asset is not in train: {asset}")
    intersection = set(by_pair) & base_pair_ids
    extra = set(by_pair) - base_pair_ids
    return by_pair, intersection, extra


def _portable_image(
    path: Path,
    *,
    root_name: str,
    root_path: Path,
    workspace_root: Path,
    expected_sha256: Any,
    context: str,
    validate_images: bool,
    cache: dict[Path, tuple[str, tuple[int, int]]],
) -> dict[str, Any]:
    resolved = _resolve_under(path, root_path, context=context)
    # ``--no-image-validation`` is an explicit fallback for the formal
    # workspace's very large NFS trees.  SHA256 values are already part of the
    # immutable quality manifests; preserve those assertions while still
    # enforcing root/relative-path portability.  The normal (default) path
    # always hashes and decodes the file below.
    if not validate_images:
        actual = _as_hash(expected_sha256, context=context, name="expected_sha256")
        dimensions = (0, 0)
    else:
        actual, dimensions = _hash_and_decode(
            resolved,
            expected_sha256=(str(expected_sha256) if expected_sha256 else None),
            context=context,
            validate_images=validate_images,
            cache=cache,
        )
    relative = resolved.relative_to(_absolute_norm(root_path)).as_posix()
    return {
        "root": root_name,
        "relative_path": relative,
        "sha256": actual,
        "dimensions": [dimensions[0], dimensions[1]],
        # Keep explicit top-level-like aliases inside the ref for consumers
        # that do not understand the older absolute-path manifest fields.
        "portable": f"{root_name}:{relative}",
        "workspace_relative_path": _workspace_relative(resolved, workspace_root, context=context),
    }


def _select_texverse_root(
    path: Path,
    *,
    texverse_root: Path,
    texverse_full_root: Path | None,
    context: str,
) -> tuple[str, Path, Path]:
    """Map an absolute source path to a stable portable root ID."""

    resolved = _absolute_norm(path)
    roots: list[tuple[str, Path]] = [("texverse", _absolute_norm(texverse_root))]
    if texverse_full_root is not None:
        roots.append(("texverse_full", _absolute_norm(texverse_full_root)))
    # Prefer the more specific root if one is nested under another.
    roots.sort(key=lambda item: len(str(item[1])), reverse=True)
    for root_name, root_path in roots:
        try:
            relative = resolved.relative_to(root_path)
        except ValueError:
            continue
        if relative.parts:
            return root_name, root_path, relative
    raise GateManifestError(
        f"{context}: source image is outside configured TexVerse roots: {resolved}"
    )


def _collect_image_entries(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_gt_and_albedo: bool,
    data_root: Path,
    texverse_root: Path,
    texverse_full_root: Path | None,
    workspace_root: Path,
    split: str,
) -> list[tuple[Path, Any, str]]:
    """Resolve all paths for a split before parallel image validation.

    The formal dataset has roughly one hundred thousand unique PNGs on a
    network filesystem.  Resolving/hash-checking them concurrently keeps the
    fail-closed validation intact while avoiding an unnecessarily serial I/O
    bottleneck.  The resulting cache is consumed by ``_expand_split``.
    """

    entries: list[tuple[Path, Any, str]] = []
    source_rows: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        context = f"{split} source:{index}"
        degraded = _relative_from_value(
            row.get("degraded_relative_path"), absolute_value=row.get("degraded_path"),
            fallback_root=data_root, workspace_root=workspace_root,
            context=f"{context}:{row.get('pair_id')}:degraded",
        )
        gt = _relative_from_value(
            row.get("gt_relative_path"), absolute_value=row.get("gt_path"),
            fallback_root=data_root, workspace_root=workspace_root,
            context=f"{context}:{row.get('pair_id')}:gt",
        )
        entries.extend([
            (degraded, row.get("degraded_sha256"), f"{context}:{row.get('pair_id')}:degraded"),
            (gt, row.get("gt_sha256"), f"{context}:{row.get('pair_id')}:gt"),
        ])
        if include_gt_and_albedo:
            source_sha = _source_hash(row, context=context)
            if split != "train" or source_sha not in EXCLUDED_MISSING_TRAIN_ALBEDO_SHA256:
                source_rows.setdefault(source_sha, row)
    if include_gt_and_albedo:
        for source_sha, row in source_rows.items():
            context = f"{split} albedo:{source_sha}"
            source_value = row.get("source_relative_path") or row.get("source_path") or row.get("source_original_path")
            source = _relative_from_value(
                source_value,
                absolute_value=row.get("source_path") or row.get("source_original_path"),
                fallback_root=texverse_root,
                workspace_root=workspace_root,
                context=context,
            )
            entries.append((source, source_sha, context))
    return entries


def _parallel_validate_images(
    entries: Sequence[tuple[Path, Any, str]],
    *,
    workspace_root: Path,
    texverse_root: Path,
    texverse_full_root: Path | None,
    validate_images: bool,
    cache: dict[Path, tuple[str, tuple[int, int]]],
    workers: int,
) -> None:
    """Populate the image cache with concurrent fail-closed checks."""

    unique: dict[Path, tuple[Any, str]] = {}
    for path, expected, context in entries:
        if not validate_images:
            # See _portable_image: the explicit no-decode fallback uses the
            # immutable manifest SHA assertion and avoids a full NFS image
            # walk.  The default validation path always hashes/decodes every
            # image and rejects mismatches.
            continue
        # A path is allowed to recur only when its expected hash agrees.  A
        # conflicting source claim is itself a hard audit failure.
        resolved = _absolute_norm(path)
        previous = unique.get(resolved)
        if previous is not None and previous[0] not in (None, expected):
            raise GateManifestError(
                f"{context}: conflicting expected hashes for image {resolved}: "
                f"{previous[0]} vs {expected}"
            )
        unique.setdefault(resolved, (expected, context))
    if not unique:
        return

    def check(item: tuple[Path, tuple[Any, str]]) -> tuple[Path, tuple[str, tuple[int, int]]]:
        path, (expected, context) = item
        local_cache: dict[Path, tuple[str, tuple[int, int]]] = {}
        value = _hash_and_decode(
            path,
            expected_sha256=(str(expected) if expected else None),
            context=context,
            validate_images=validate_images,
            cache=local_cache,
        )
        return path, value

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(check, item) for item in unique.items()]
        for future in futures:
            path, value = future.result()
            cache[_absolute_norm(path)] = value


def _common_row(
    *,
    split: str,
    pair_id: str,
    asset_id: str,
    source_sha256: str,
    degraded_sha256: str,
    gt_sha256: str,
    task_id: str,
    degradation_type: str,
    severity_tag: Any,
    image_kind: str,
    image_ref: dict[str, Any],
    decision: str,
    reasons: Sequence[str],
    hand_remove_member: bool,
    label_provenance: str,
    group: str | None = None,
) -> dict[str, Any]:
    if image_kind not in IMAGE_KINDS:
        raise GateManifestError(f"invalid image_kind {image_kind!r}")
    if decision not in DECISIONS:
        raise GateManifestError(f"invalid decision {decision!r}")
    normalized = sorted(set(str(reason) for reason in reasons), key=lambda x: REASON_INDEX.get(x, 10_000))
    if decision == "KEEP" and normalized != ["NONE"]:
        raise GateManifestError("KEEP records must have exactly reasons=[NONE]")
    if decision == "REPAIR" and (not normalized or "NONE" in normalized):
        raise GateManifestError("REPAIR records must have non-NONE reasons")
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"qwen_gate_v2:{split}:{pair_id}:{image_kind}:{image_ref['sha256'][:16]}",
        "group": group or f"pair:{pair_id}",
        "pair": pair_id,
        "asset": asset_id,
        "split": split,
        "image_kind": image_kind,
        "image": image_ref,
        "image_root": image_ref["root"],
        "image_relative_path": image_ref["relative_path"],
        "image_sha256": image_ref["sha256"],
        "decision": decision,
        "reasons": normalized,
        "task": task_id,
        "task_id": task_id,
        "degradation": degradation_type,
        "degradation_type": degradation_type,
        "severity_tag": severity_tag if severity_tag not in (None, "") else "unspecified",
        "severity_tag_provenance_only": True,
        "severity_tag_note": "provenance/task label only; not visual severity and not a decision feature",
        "hand_remove_member": bool(hand_remove_member),
        "label_provenance": label_provenance,
        "source_sha256": source_sha256,
        "degraded_sha256": degraded_sha256,
        "gt_sha256": gt_sha256,
        "one_image_contract": True,
    }


def _validate_source_row(row: Mapping[str, Any], *, context: str) -> None:
    for key in ("pair_id", "sha256", "task_id", "degraded_path", "gt_path"):
        if not row.get(key):
            # Relative paths are accepted when no absolute alias is present.
            alternate = {"degraded_path": "degraded_relative_path", "gt_path": "gt_relative_path"}.get(key)
            if not alternate or not row.get(alternate):
                raise GateManifestError(f"{context}: missing {key}")
    if str(row.get("split") or "train") not in {"train", "test", "validation", "in_domain_validation"}:
        raise GateManifestError(f"{context}: unsupported input split {row.get('split')!r}")


def _source_hash(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("source_image_sha256") or row.get("source_sha256")
    return _as_hash(value, context=context, name="source_image_sha256")


def _image_expected(row: Mapping[str, Any], kind: str) -> Any:
    return {
        "degraded": row.get("degraded_sha256"),
        "gt_clean": row.get("gt_sha256"),
        "albedo_render": row.get("source_image_sha256"),
    }[kind]


def _expand_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    sidecar_by_pair: Mapping[str, Mapping[str, Any]],
    include_gt_and_albedo: bool,
    workspace_root: Path,
    data_root: Path,
    texverse_root: Path,
    texverse_full_root: Path | None,
    validate_images: bool,
    cache: dict[Path, tuple[str, tuple[int, int]]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    source_rows: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        context = f"{split} source:{index}"
        _validate_source_row(row, context=context)
        pair_id = str(row["pair_id"])
        if pair_id in seen_pairs:
            raise GateManifestError(f"{context}: duplicate pair_id {pair_id}")
        seen_pairs.add(pair_id)
        asset_id = str(row["sha256"])
        task_id = str(row["task_id"])
        source_sha = _source_hash(row, context=context)
        degraded_path = _relative_from_value(
            row.get("degraded_relative_path"),
            absolute_value=row.get("degraded_path"),
            fallback_root=data_root,
            workspace_root=workspace_root,
            context=f"{context}:{pair_id}:degraded",
        )
        gt_path = _relative_from_value(
            row.get("gt_relative_path"),
            absolute_value=row.get("gt_path"),
            fallback_root=data_root,
            workspace_root=workspace_root,
            context=f"{context}:{pair_id}:gt",
        )
        degraded_ref = _portable_image(
            degraded_path,
            root_name="workspace",
            root_path=workspace_root,
            workspace_root=workspace_root,
            expected_sha256=row.get("degraded_sha256"),
            context=f"{context}:{pair_id}:degraded",
            validate_images=validate_images,
            cache=cache,
        )
        gt_ref = _portable_image(
            gt_path,
            root_name="workspace",
            root_path=workspace_root,
            workspace_root=workspace_root,
            expected_sha256=row.get("gt_sha256"),
            context=f"{context}:{pair_id}:gt",
            validate_images=validate_images,
            cache=cache,
        )
        sidecar = sidecar_by_pair.get(pair_id)
        hand_member = sidecar is not None
        # The sidecar is the provenance authority for a hand-reviewed member.
        # Preserve its original task reason even if a later V3 row has a
        # normalized/display task field; EXTERNAL_HUMAN_SUPPORT is additive.
        reason_row = sidecar if hand_member and sidecar.get("task_id") else row
        task_reason = _task_reason(reason_row, context=f"{context}:{pair_id}:reason")
        reasons = [task_reason]
        if hand_member:
            reasons.append("EXTERNAL_HUMAN_SUPPORT")
        severity_tag = row.get("severity") or row.get("severity_tag") or "heavy"
        provenance = "v3_reviewed_task_plus_hand_sidecar" if hand_member else "v3_reviewed_task"
        expanded.append(
            _common_row(
                split=split,
                pair_id=pair_id,
                asset_id=asset_id,
                source_sha256=source_sha,
                degraded_sha256=degraded_ref["sha256"],
                gt_sha256=gt_ref["sha256"],
                task_id=task_id,
                degradation_type=str(row.get("degradation_type") or task_id.removesuffix("_heavy")),
                severity_tag=severity_tag,
                image_kind="degraded",
                image_ref=degraded_ref,
                decision="REPAIR",
                reasons=reasons,
                hand_remove_member=hand_member,
                label_provenance=provenance,
            )
        )
        if include_gt_and_albedo:
            gt_provenance = "v3_reviewed_hand_replacement_gt" if hand_member else "v3_reviewed_gt"
            expanded.append(
                _common_row(
                    split=split,
                    pair_id=pair_id,
                    asset_id=asset_id,
                    source_sha256=source_sha,
                    degraded_sha256=degraded_ref["sha256"],
                    gt_sha256=gt_ref["sha256"],
                    task_id=task_id,
                    degradation_type=str(row.get("degradation_type") or task_id.removesuffix("_heavy")),
                    severity_tag=severity_tag,
                    image_kind="gt_clean",
                    image_ref=gt_ref,
                    decision="KEEP",
                    reasons=["NONE"],
                    hand_remove_member=hand_member,
                    label_provenance=gt_provenance,
                )
            )
            if split != "train" or source_sha not in EXCLUDED_MISSING_TRAIN_ALBEDO_SHA256:
                source_rows.setdefault(source_sha, row)

    if include_gt_and_albedo:
        for source_sha, row in sorted(source_rows.items()):
            context = f"{split} albedo:{source_sha}"
            source_value = row.get("source_relative_path") or row.get("source_path") or row.get("source_original_path")
            source_path = _relative_from_value(
                source_value,
                absolute_value=row.get("source_path") or row.get("source_original_path"),
                fallback_root=texverse_root,
                workspace_root=workspace_root,
                context=context,
            )
            # Albedo refs retain whether the source came from TexVerse or the
            # separately provisioned TexVerse-full tree.  Never rewrite one
            # root into the other: their relative layouts are not guaranteed
            # to match.
            root_name, root_path, source_relative = _select_texverse_root(
                source_path,
                texverse_root=texverse_root,
                texverse_full_root=texverse_full_root,
                context=context,
            )
            source_ref = _portable_image(
                source_relative,
                root_name=root_name,
                root_path=root_path,
                workspace_root=workspace_root,
                expected_sha256=source_sha,
                context=context,
                validate_images=validate_images,
                cache=cache,
            )
            pair_id = str(row["pair_id"])
            task_id = str(row["task_id"])
            expanded.append(
                _common_row(
                    split=split,
                    pair_id=pair_id,
                    asset_id=str(row["sha256"]),
                    source_sha256=source_sha,
                    degraded_sha256=str(row.get("degraded_sha256") or "0" * 64),
                    gt_sha256=str(row.get("gt_sha256") or "0" * 64),
                    task_id=task_id,
                    degradation_type=str(row.get("degradation_type") or task_id.removesuffix("_heavy")),
                    severity_tag=row.get("severity") or row.get("severity_tag") or "heavy",
                    image_kind="albedo_render",
                    image_ref=source_ref,
                    decision="KEEP",
                    reasons=["NONE"],
                    hand_remove_member=False,
                    label_provenance="texverse_source_image_deduplicated_by_source_sha256",
                    group=f"source:{source_sha}",
                )
            )
    return expanded


def _base_sets(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {name: set() for name in LEAKAGE_FIELDS}
    for index, row in enumerate(rows, start=1):
        context = f"source row:{index}"
        sets["asset"].add(str(_require(row, "sha256", context)))
        sets["pair"].add(str(_require(row, "pair_id", context)))
        for source_key, audit_key in (
            ("degraded_sha256", "degraded_sha256"),
            ("gt_sha256", "gt_sha256"),
            ("source_image_sha256", "source_sha256"),
        ):
            value = row.get(source_key)
            if value:
                sets[audit_key].add(_as_hash(value, context=context, name=source_key))
            else:
                raise GateManifestError(f"{context}: missing {source_key}")
    return sets


def _derived_sets(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {name: set() for name in LEAKAGE_FIELDS}
    sample_ids: set[str] = set()
    image_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=1):
        context = f"derived row:{index}"
        for key in ("sample_id", "pair", "asset", "image_kind", "image_sha256", "source_sha256", "degraded_sha256", "gt_sha256"):
            _require(row, key, context)
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise GateManifestError(f"{context}: duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)
        image_kind = str(row["image_kind"])
        image_sha = _as_hash(row["image_sha256"], context=context, name="image_sha256")
        image_key = (image_kind, image_sha)
        if image_key in image_keys:
            raise GateManifestError(f"{context}: duplicate {image_kind} image SHA {image_sha}")
        image_keys.add(image_key)
        result["asset"].add(str(row["asset"]))
        result["pair"].add(str(row["pair"]))
        result["degraded_sha256"].add(_as_hash(row["degraded_sha256"], context=context, name="degraded_sha256"))
        result["gt_sha256"].add(_as_hash(row["gt_sha256"], context=context, name="gt_sha256"))
        result["source_sha256"].add(_as_hash(row["source_sha256"], context=context, name="source_sha256"))
        image = row.get("image")
        if not isinstance(image, dict) or image.get("root") not in set(PORTABLE_IMAGE_ROOTS):
            raise GateManifestError(f"{context}: image must be a portable workspace/TexVerse ref")
        if not bool(row.get("one_image_contract")):
            raise GateManifestError(f"{context}: one_image_contract must be true")
    return result


def _intersections(left: Mapping[str, set[str]], right: Mapping[str, set[str]]) -> dict[str, list[str]]:
    return {name: sorted(left[name] & right[name]) for name in LEAKAGE_FIELDS}


def _check_reason_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decision_counts = Counter(str(row["decision"]) for row in rows)
    reason_counts: Counter[str] = Counter()
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        context = f"derived row:{index}"
        decision = str(row.get("decision"))
        reasons = row.get("reasons")
        if decision not in DECISIONS or not isinstance(reasons, list) or not reasons:
            errors.append(f"{context}: invalid decision/reasons")
            continue
        if reasons != sorted(reasons, key=lambda x: REASON_INDEX.get(str(x), 10_000)):
            errors.append(f"{context}: reasons are not in canonical order")
        reason_counts.update(str(reason) for reason in reasons)
        if decision == "KEEP" and reasons != ["NONE"]:
            errors.append(f"{context}: KEEP must be NONE")
        if decision == "REPAIR" and "NONE" in reasons:
            errors.append(f"{context}: REPAIR cannot include NONE")
        if "_heavy" in str(row.get("task_id", "")) and row.get("severity_tag_provenance_only") is not True:
            errors.append(f"{context}: heavy task lacks provenance-only marker")
    return {
        "status": "ok" if not errors else "failed",
        "decision_counts": dict(decision_counts),
        "reason_counts": dict(reason_counts),
        "errors": errors,
    }


def audit_gate_manifests(
    train_source_rows: Sequence[Mapping[str, Any]],
    validation_source_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    sidecar_rows: Sequence[Mapping[str, Any]],
    *,
    expected_sidecar_count: int | None = 4768,
    expected_hand_intersection: int | None = 4116,
    expected_hand_extra: int | None = 652,
) -> dict[str, Any]:
    """Run all fail-closed audits and return a JSON-serializable report."""

    errors: list[str] = []
    train_source = _base_sets(train_source_rows)
    validation_source = _base_sets(validation_source_rows)
    source_cross = _intersections(train_source, validation_source)
    for key, values in source_cross.items():
        if values:
            errors.append(f"source split leakage at {key}: {values[:5]}")
    train_derived = _derived_sets(train_rows)
    validation_derived = _derived_sets(validation_rows)
    derived_cross = _intersections(train_derived, validation_derived)
    for key, values in derived_cross.items():
        if values:
            errors.append(f"derived split leakage at {key}: {values[:5]}")
    # Every derived row must originate from its own source split.  This catches
    # accidental use of the old test manifest or an absolute path from another
    # workspace before model training starts.
    for key, values in _intersections(train_derived, validation_source).items():
        if values:
            errors.append(f"train derived rows leak validation source at {key}: {values[:5]}")
    for key, values in _intersections(validation_derived, train_source).items():
        if values:
            errors.append(f"validation derived rows leak train source at {key}: {values[:5]}")
    reason_audit = _check_reason_contract([*train_rows, *validation_rows])
    errors.extend(reason_audit["errors"])
    sidecar_ids = {str(row.get("pair_id")) for row in sidecar_rows}
    train_ids = {str(row.get("pair_id")) for row in train_source_rows}
    validation_ids = {str(row.get("pair_id")) for row in validation_source_rows}
    overlap = sorted(sidecar_ids & validation_ids)
    if overlap:
        errors.append(f"hand sidecar overlaps validation pair IDs: {overlap[:5]}")
    intersection_count = len(sidecar_ids & train_ids)
    extra_count = len(sidecar_ids - train_ids)
    sidecar_audit = {
        "records": len(sidecar_rows),
        "pair_ids": len(sidecar_ids),
        "v3_train_intersection": intersection_count,
        "extra_gate_degraded_only": extra_count,
        "all_have_external_human_support_reason": all(
            "EXTERNAL_HUMAN_SUPPORT" in row.get("reasons", [])
            for row in train_rows
            if str(row.get("pair")) in sidecar_ids and row.get("image_kind") == "degraded"
        ),
    }
    if expected_sidecar_count is not None and len(sidecar_rows) != expected_sidecar_count:
        errors.append(f"hand sidecar count expected {expected_sidecar_count}, got {len(sidecar_rows)}")
    if expected_hand_intersection is not None and intersection_count != expected_hand_intersection:
        errors.append(f"hand/V3 intersection expected {expected_hand_intersection}, got {intersection_count}")
    if expected_hand_extra is not None and extra_count != expected_hand_extra:
        errors.append(f"hand extra count expected {expected_hand_extra}, got {extra_count}")
    # Explicitly verify the “only degraded” rule for sidecar extras.
    for row in train_rows:
        if str(row.get("pair")) in (sidecar_ids - train_ids) and row.get("image_kind") != "degraded":
            errors.append(f"sidecar extra pair got non-degraded row: {row.get('pair')}")
    counts = {
        "train": Counter(str(row.get("image_kind")) for row in train_rows),
        "validation": Counter(str(row.get("image_kind")) for row in validation_rows),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "source_split_five_layer_intersections": source_cross,
        "derived_split_five_layer_intersections": derived_cross,
        "derived_train_vs_validation_source_intersections": {
            key: values for key, values in _intersections(train_derived, validation_source).items() if values
        },
        "derived_validation_vs_train_source_intersections": {
            key: values for key, values in _intersections(validation_derived, train_source).items() if values
        },
        "counts": {split: dict(counter) for split, counter in counts.items()},
        "sidecar": sidecar_audit,
        "reason_contract": reason_audit,
        "five_layer_keys": list(LEAKAGE_FIELDS),
        "heavy_semantics": "task/severity provenance only; never visual intensity or a decision feature",
        "excluded_missing_train_albedo_sha256": sorted(EXCLUDED_MISSING_TRAIN_ALBEDO_SHA256),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def build_gate_manifests(
    *,
    train_manifest: Path,
    validation_manifest: Path,
    hand_manifest: Path | None,
    output_dir: Path,
    workspace_root: Path = WORKSPACE,
    data_root: Path | None = None,
    texverse_root: Path | None = None,
    texverse_full_root: Path | None = None,
    validate_images: bool = True,
    image_workers: int = 16,
    expected_sidecar_count: int | None = None,
    expected_hand_intersection: int | None = None,
    expected_hand_extra: int | None = None,
) -> dict[str, Any]:
    """Build train/in-domain JSONL files and return the audit summary.

    Output files are written to temporary siblings and atomically renamed only
    after every audit succeeds.  Existing output files are never overwritten
    on failure.
    """

    workspace_root = _absolute_norm(workspace_root.expanduser())
    data_root = _absolute_norm((data_root or workspace_root).expanduser())
    texverse_root = _absolute_norm((texverse_root or workspace_root / "TexVerse").expanduser())
    texverse_full_root = (
        _absolute_norm(texverse_full_root.expanduser())
        if texverse_full_root is not None
        else _absolute_norm(workspace_root / "TexVerse-full")
    )
    train_manifest = _absolute_norm(train_manifest.expanduser())
    validation_manifest = _absolute_norm(validation_manifest.expanduser())
    hand_manifest = _absolute_norm(hand_manifest.expanduser()) if hand_manifest is not None else None
    for path in (train_manifest, validation_manifest) + ((hand_manifest,) if hand_manifest is not None else ()):
        if not path.is_file():
            raise GateManifestError(f"input manifest missing: {path}")
    train_source_rows = _read_jsonl(train_manifest)
    validation_source_rows = _read_jsonl(validation_manifest)
    sidecar_rows = _read_jsonl(hand_manifest) if hand_manifest is not None else []
    if not train_source_rows or not validation_source_rows:
        raise GateManifestError("source manifests must not be empty")
    train_pair_ids = {str(row.get("pair_id")) for row in train_source_rows}
    validation_pair_ids = {str(row.get("pair_id")) for row in validation_source_rows}
    train_asset_ids = {str(row.get("sha256")) for row in train_source_rows}
    sidecar_by_pair, intersection, extra = _sidecar_membership(
        sidecar_rows,
        base_pair_ids=train_pair_ids,
        train_asset_ids=train_asset_ids,
        test_pair_ids=validation_pair_ids,
    )
    if expected_sidecar_count is not None and len(sidecar_rows) != expected_sidecar_count:
        raise GateManifestError(f"hand sidecar expected {expected_sidecar_count}, got {len(sidecar_rows)}")
    if expected_hand_intersection is not None and len(intersection) != expected_hand_intersection:
        raise GateManifestError(f"hand/V3 intersection expected {expected_hand_intersection}, got {len(intersection)}")
    if expected_hand_extra is not None and len(extra) != expected_hand_extra:
        raise GateManifestError(f"hand/V3 extra expected {expected_hand_extra}, got {len(extra)}")

    cache: dict[Path, tuple[str, tuple[int, int]]] = {}
    # Validate every source image before expanding/writing any derived row.
    # This is still fail-closed; concurrency is only an I/O optimization.
    all_entries = []
    all_entries.extend(_collect_image_entries(
        train_source_rows,
        include_gt_and_albedo=True,
        data_root=data_root,
        texverse_root=texverse_root,
        texverse_full_root=texverse_full_root,
        workspace_root=workspace_root,
        split="train",
    ))
    # The 652 extra rows are resolved below; add them now so their degraded
    # image (and the source sidecar GT hash) is checked before publication.
    extra_rows = [sidecar_by_pair[pair_id] for pair_id in sorted(extra)]
    all_entries.extend(_collect_image_entries(
        extra_rows,
        include_gt_and_albedo=False,
        data_root=data_root,
        texverse_root=texverse_root,
        texverse_full_root=texverse_full_root,
        workspace_root=workspace_root,
        split="train_extra",
    ))
    all_entries.extend(_collect_image_entries(
        validation_source_rows,
        include_gt_and_albedo=True,
        data_root=data_root,
        texverse_root=texverse_root,
        texverse_full_root=texverse_full_root,
        workspace_root=workspace_root,
        split="in_domain_validation",
    ))
    _parallel_validate_images(
        all_entries,
        workspace_root=workspace_root,
        texverse_root=texverse_root,
        texverse_full_root=texverse_full_root,
        validate_images=validate_images,
        cache=cache,
        workers=image_workers,
    )
    train_rows = _expand_split(
        train_source_rows,
        split="train",
        sidecar_by_pair=sidecar_by_pair,
        include_gt_and_albedo=True,
        workspace_root=workspace_root,
        data_root=data_root,
        texverse_root=texverse_root,
        texverse_full_root=texverse_full_root,
        validate_images=validate_images,
        cache=cache,
    )
    # Extra hand sidecar rows are deliberately not added to the trusted V3 GT
    # or albedo populations.  They only contribute a degraded REPAIR record.
    extra_expanded = _expand_split(
        extra_rows,
        split="train",
        sidecar_by_pair=sidecar_by_pair,
        include_gt_and_albedo=False,
        workspace_root=workspace_root,
        data_root=data_root,
        texverse_root=texverse_root,
        texverse_full_root=texverse_full_root,
        validate_images=validate_images,
        cache=cache,
    )
    for extra_row in extra_expanded:
        extra_row["label_provenance"] = "hand_sidecar_extra_degraded_only"
    # _expand_split marks every sidecar row in extra_expanded as a normal
    # sidecar degraded record, which is exactly the required 652-row addition.
    train_rows.extend(extra_expanded)
    validation_rows = _expand_split(
        validation_source_rows,
        split="in_domain_validation",
        sidecar_by_pair={},
        include_gt_and_albedo=True,
        workspace_root=workspace_root,
        data_root=data_root,
        texverse_root=texverse_root,
        texverse_full_root=texverse_full_root,
        validate_images=validate_images,
        cache=cache,
    )
    audit = audit_gate_manifests(
        train_source_rows,
        validation_source_rows,
        train_rows,
        validation_rows,
        sidecar_rows,
        expected_sidecar_count=expected_sidecar_count,
        expected_hand_intersection=expected_hand_intersection,
        expected_hand_extra=expected_hand_extra,
    )
    if audit["status"] != "ok":
        raise GateManifestError("gate manifest audit failed: " + "; ".join(audit["errors"][:8]))

    output_dir = _absolute_norm(output_dir.expanduser())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(output_dir.name + f".partial-{os.getpid()}")
    if staging.exists():
        raise GateManifestError(f"staging output already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        _write_jsonl(staging / "train.jsonl", train_rows)
        _write_jsonl(staging / "validation.jsonl", validation_rows)
        manifest_summary = {
            **audit,
            "inputs": {
                "train_manifest": str(train_manifest),
                "validation_manifest": str(validation_manifest),
                "hand_manifest": str(hand_manifest),
                "workspace_root": str(workspace_root),
                "texverse_root": str(texverse_root),
                "texverse_full_root": str(texverse_full_root),
            },
            "formal_build": {
                "image_validation": bool(validate_images),
                "image_workers": int(image_workers),
                "source_train_rows": len(train_source_rows),
                "source_validation_rows": len(validation_source_rows),
                "hand_sidecar_rows": len(sidecar_rows),
                "hand_v3_intersection": len(intersection),
                "hand_extra_degraded_only": len(extra),
                "cache_unique_images": len(cache),
            },
        }
        _write_json(staging / "audit_summary.json", manifest_summary)
        _write_json(staging / "schema.json", {
            "schema_version": SCHEMA_VERSION,
            "required_fields": [
                "sample_id", "group", "pair", "asset", "split", "image_kind", "image",
                "image_sha256", "decision", "reasons", "task", "degradation",
                "severity_tag", "severity_tag_provenance_only", "hand_remove_member",
                "label_provenance", "source_sha256", "degraded_sha256", "gt_sha256",
                "one_image_contract",
            ],
            "image_kinds": list(IMAGE_KINDS),
            "portable_image_roots": list(PORTABLE_IMAGE_ROOTS),
            "decisions": list(DECISIONS),
            "reason_order": list(REASON_ORDER),
            "heavy_semantics": manifest_summary["heavy_semantics"],
        })
        if output_dir.exists():
            raise GateManifestError(f"refusing to overwrite existing output: {output_dir}")
        try:
            staging.rename(output_dir)
        except PermissionError:
            # Some distributed/NFS mounts reject directory rename even when
            # creating files/directories is allowed.  Preserve the same
            # no-overwrite check and publish by copy, then remove only the
            # private staging tree.
            import shutil

            if output_dir.exists():
                raise GateManifestError(f"refusing to overwrite existing output: {output_dir}")
            shutil.copytree(staging, output_dir)
            shutil.rmtree(staging)
    except Exception:
        # Preserve no partially published dataset.  The caller can rerun after
        # correcting inputs; input manifests remain untouched.
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest_summary


def default_paths(workspace_root: Path = WORKSPACE) -> dict[str, Path]:
    quality = workspace_root / "data"
    return {
        "train_manifest": quality / "manifests/train.jsonl",
        "validation_manifest": quality / "manifests/test.jsonl",
        "hand_manifest": quality / "hand_remove/manifest.jsonl",
        "output_dir": quality / "router",
        "data_root": workspace_root,
        "texverse_root": workspace_root / "TexVerse",
        "texverse_full_root": workspace_root / "TexVerse-full",
    }


def _parser() -> argparse.ArgumentParser:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=defaults["train_manifest"])
    parser.add_argument("--validation-manifest", "--test-manifest", dest="validation_manifest", type=Path, default=defaults["validation_manifest"])
    parser.add_argument("--hand-manifest", type=Path, help="Optional audited hand-support sidecar")
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE)
    parser.add_argument("--data-root", type=Path, default=defaults["data_root"])
    parser.add_argument("--texverse-root", type=Path, default=defaults["texverse_root"])
    parser.add_argument("--texverse-full-root", type=Path, default=defaults["texverse_full_root"])
    parser.add_argument("--image-workers", type=int, default=16, help="parallel image hash/decode workers")
    parser.add_argument("--no-image-validation", action="store_true", help="skip PIL decode while still hashing every referenced path (use only when a separate decode audit is recorded)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = build_gate_manifests(
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            hand_manifest=args.hand_manifest,
            output_dir=args.output_dir,
            workspace_root=args.workspace_root,
            data_root=args.data_root,
            texverse_root=args.texverse_root,
            texverse_full_root=args.texverse_full_root,
            validate_images=not args.no_image_validation,
            image_workers=args.image_workers,
            expected_sidecar_count=None,
            expected_hand_intersection=None,
            expected_hand_extra=None,
        )
    except Exception as exc:
        # A failed audit is useful evidence even though no train/validation
        # manifest is published.  Do not overwrite a previous successful
        # output directory.
        normalized_output = _absolute_norm(args.output_dir.expanduser())
        error_path = normalized_output.parent / (normalized_output.name + ".audit_failed.json")
        _write_json(error_path, {"schema_version": SCHEMA_VERSION, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

