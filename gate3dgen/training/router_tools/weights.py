"""Audited mask joins and original family/subtle weighting for router training."""
from __future__ import annotations


import argparse


from concurrent.futures import ThreadPoolExecutor


import hashlib


import json


import math


import os


import shutil


import sys


from collections import Counter, defaultdict


from pathlib import Path


from typing import Any, Iterable, Mapping, Sequence


try:
    from PIL import Image
except ImportError:  # pragma: no cover - formal build environment has Pillow
    Image = None  # type: ignore[assignment]


WORKSPACE = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()


WEIGHT_SCHEMA_VERSION = "qwen-repair-gate-v3-sample-weights"


MASK_SCHEMA_VERSION = "qwen-repair-gate-v3-rmbg2-mask-sidecar"


RMBG2_MASK_SCHEMA = "texverse-clean-gt-rmbg2-soft-mask-v1"


SUBTLE_INELIGIBLE_REASON = "no_trusted_aligned_gt_mask"


FAMILY_WEIGHTS = {
    "animal_motion_blur": 2.0,
    "overexposure": 1.75,
    "shadow": 1.5,
    "white_balance": 1.5,
    "local_color_spill": 1.5,
    "optical_blur": 1.5,
}


REASON_TO_FAMILY = {
    "MOTION_BLUR": "motion_blur",
    "OPTICAL_BLUR": "optical_blur",
    "OVEREXPOSURE": "overexposure",
    "HARD_SHADOW": "shadow",
    "WHITE_BALANCE": "white_balance",
    "COLOR_SPILL": "local_color_spill",
}


class V3ManifestError(ValueError):
    """Raised when a V3 artifact cannot be published safely."""


def _abs(path: Path | str) -> Path:
    return Path(os.path.abspath(os.path.normpath(os.fspath(Path(path).expanduser()))))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3ManifestError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_existing(path: Path | str, workspace_root: Path, *, context: str) -> dict[str, Any]:
    """Describe an existing file with a workspace-portable reference."""

    resolved = _resolve_source(path, workspace_root)
    root = _abs(workspace_root)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise V3ManifestError(f"{context}: path outside workspace: {resolved}") from exc
    if Image is None:
        raise V3ManifestError("Pillow is required for V3 mask validation")
    try:
        with Image.open(resolved) as image:
            dimensions = [int(image.width), int(image.height)]
            mode = str(image.mode)
            image.verify()
    except Exception as exc:
        raise V3ManifestError(f"{context}: image failed decode: {resolved}: {exc}") from exc
    return {
        "root": "workspace",
        "relative_path": relative.as_posix(),
        "workspace_relative_path": relative.as_posix(),
        "portable": f"workspace:{relative.as_posix()}",
        "sha256": _sha(resolved),
        "dimensions": dimensions,
        "mode": mode,
    }


def _portable_declared_existing(
    path: Path | str,
    workspace_root: Path,
    *,
    declared_sha256: str,
    declared_dimensions: Any = None,
    probe: bool = True,
    context: str,
) -> dict[str, Any]:
    """Describe a manifest-audited image without rereading its pixels.

    The gate/reviewed manifests already contain SHA256 and dimensions produced
    by the audited data builder.  Rehashing the same 1024px image once per
    sidecar row is needlessly expensive on the distributed NFS mount; mask files
    are still decoded and hashed below because they are the artifact being
    newly trusted.  We retain a stat/access check here so a missing or empty
    ordinary image cannot pass unnoticed.
    """

    resolved = _resolve_source(path, workspace_root) if probe else _portable_path(path, workspace_root, context=context)
    if probe and (not resolved.is_file() or resolved.stat().st_size <= 0):
        raise V3ManifestError(f"{context}: image is missing or empty: {resolved}")
    root = _abs(workspace_root)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise V3ManifestError(f"{context}: path outside workspace: {resolved}") from exc
    digest = str(declared_sha256 or "")
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise V3ManifestError(f"{context}: invalid declared SHA256: {declared_sha256!r}")
    dimensions = None
    if declared_dimensions is not None:
        try:
            candidate = [int(declared_dimensions[0]), int(declared_dimensions[1])]
        except (TypeError, IndexError, ValueError) as exc:
            raise V3ManifestError(f"{context}: invalid declared dimensions: {declared_dimensions!r}") from exc
        if candidate[0] <= 0 or candidate[1] <= 0:
            raise V3ManifestError(f"{context}: non-positive declared dimensions: {candidate!r}")
        dimensions = candidate
    return {
        "root": "workspace",
        "relative_path": relative.as_posix(),
        "workspace_relative_path": relative.as_posix(),
        "portable": f"workspace:{relative.as_posix()}",
        "sha256": digest.lower(),
        "dimensions": dimensions,
        "mode": "manifest_declared",
    }


def _expanded_image_path(row: Mapping[str, Any], workspace_root: Path, *, context: str) -> Path:
    image = row.get("image")
    if isinstance(image, Mapping):
        root_name = str(image.get("root") or "workspace")
        relative = image.get("relative_path") or image.get("workspace_relative_path")
        if relative:
            if root_name in {"workspace", "TexVerse", "texverse", "TexVerse-full", "texverse_full"}:
                # Gate V2 rows use root=workspace even when the path points at
                # scripts/data/...; the portable reference remains relative to
                # the current workspace and is therefore safe to remap.
                return _portable_path(_abs(workspace_root) / str(relative), workspace_root, context=context)
    for key in ("image_path", "path", "degraded_path", "gt_path", "degraded_relative_path", "gt_relative_path"):
        if row.get(key):
            return _portable_path(row[key], workspace_root, context=context)
    raise V3ManifestError(f"{context}: row has no resolvable image reference")


def _resolve_source(value: Any, workspace_root: Path) -> Path:
    if value in (None, ""):
        raise V3ManifestError("benchmark row is missing an image path")
    path = Path(str(value)).expanduser()
    if path.is_absolute() and path.is_file():
        return path.resolve()
    if not path.is_absolute():
        candidate = _abs(workspace_root / path)
        if candidate.is_file():
            return candidate
    raw = str(path)
    aliases = ()
    for alias in aliases:
        if raw == alias or raw.startswith(alias + "/"):
            candidate = _abs(workspace_root / raw[len(alias):].lstrip("/"))
            if candidate.is_file():
                return candidate
    raise V3ManifestError(f"benchmark image is not accessible: {value}")


def _portable_path(value: Any, workspace_root: Path, *, context: str) -> Path:
    """Resolve a known workspace reference without an NFS existence probe."""

    if value in (None, ""):
        raise V3ManifestError(f"{context}: missing path")
    root = _abs(workspace_root)
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return _abs(root / path)
    raw = str(path)
    for alias in ():
        if raw == alias:
            return root
        if raw.startswith(alias + "/"):
            return _abs(root / raw[len(alias):].lstrip("/"))
    return path.resolve()


def _audit_path(path: Path, workspace_root: Path) -> str:
    """Return a stable workspace-relative path for generated audit JSON."""

    try:
        return path.relative_to(_abs(workspace_root)).as_posix()
    except ValueError:
        return str(path)


def _expanded_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise V3ManifestError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    if not rows:
        raise V3ManifestError(f"empty expanded manifest: {path}")
    return rows


def _unique_manifest_map(
    path: Path,
    *,
    key_name: str,
    expected_count: int | None = None,
    allow_fixture_counts: bool = False,
) -> dict[str, dict[str, Any]]:
    rows = _expanded_rows(_abs(path))
    result: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        key = str(row.get(key_name) or "")
        if not key or key in result:
            raise V3ManifestError(f"{path}:{line_number}: duplicate/empty {key_name}")
        result[key] = row
    if expected_count is not None and not allow_fixture_counts and len(result) != expected_count:
        raise V3ManifestError(
            f"{path}: expected {expected_count} unique {key_name} rows, got {len(result)}"
        )
    return result


def _mask_for_reviewed_gt(cache_root: Path, gt_relative_path: Any) -> Path:
    relative = Path(str(gt_relative_path or ""))
    if relative.is_absolute() or ".." in relative.parts or not str(relative):
        raise V3ManifestError(f"invalid reviewed gt_relative_path for mask lookup: {gt_relative_path!r}")
    return cache_root / (relative.as_posix() + ".rmbg2-alpha.png")


def _resolve_declared_file(value: Any, workspace_root: Path, *, context: str) -> Path:
    try:
        return _resolve_source(value, workspace_root)
    except V3ManifestError as exc:
        raise V3ManifestError(f"{context}: {exc}") from exc


def build_mask_sidecar(
    *,
    gate_manifest: Path,
    reviewed_manifest: Path,
    source_manifest: Path,
    mask_cache_root: Path,
    output_dir: Path,
    workspace_root: Path = WORKSPACE,
    expected_gate_repair_rows: int = 44588,
    expected_reviewed_rows: int = 43936,
    expected_missing_by_design_rows: int = 652,
    allow_fixture_counts: bool = False,
) -> dict[str, Any]:
    """Publish a trusted RMBG2 mask mapping for gate REPAIR samples.

    The reviewed manifest is the trust boundary.  A gate pair outside that
    manifest is allowed only when it is explicitly a hand-remove row; those
    rows are recorded as ``subtle_eligible=false`` rather than receiving a
    fabricated mask or proxy value.  A normal/base pair missing from the
    reviewed mask cache is a hard error.
    """

    workspace_root = _abs(workspace_root)
    gate_manifest = _abs(gate_manifest)
    reviewed_manifest = _abs(reviewed_manifest)
    source_manifest = _abs(source_manifest)
    mask_cache_root = _abs(mask_cache_root)
    marker_path = mask_cache_root / "rmbg2_masks_complete.json"
    manual_review_path = mask_cache_root / "rmbg2_manual_review.json"
    marker = _read_json(marker_path)
    manual_review = _read_json(manual_review_path)
    if not isinstance(marker, dict) or marker.get("status") != "complete":
        raise V3ManifestError(f"RMBG2 cache marker is not complete: {marker_path}")
    if marker.get("schema") != RMBG2_MASK_SCHEMA:
        raise V3ManifestError(
            f"unexpected RMBG2 cache schema in {marker_path}: {marker.get('schema')!r}"
        )
    if not isinstance(manual_review, dict) or manual_review.get("status") != "complete":
        raise V3ManifestError(f"RMBG2 manual-review marker is not complete: {manual_review_path}")
    if manual_review.get("base_complete_marker") != marker_path.name:
        raise V3ManifestError("RMBG2 manual-review marker does not name the complete-cache marker")

    source_marker_path = _resolve_declared_file(marker.get("manifest"), workspace_root, context="cache marker manifest")
    source_manifest_sha = _sha(source_manifest)
    marker_source_sha = _sha(source_marker_path)
    if marker_source_sha != source_manifest_sha:
        raise V3ManifestError(
            "RMBG2 cache was generated from a different source manifest: "
            f"marker={marker_source_sha} requested={source_manifest_sha}"
        )
    marker_declared_sha = str(marker.get("manifest_sha256") or "")
    if marker_declared_sha != source_manifest_sha:
        raise V3ManifestError(
            f"RMBG2 cache marker manifest SHA mismatch: marker={marker_declared_sha} actual={source_manifest_sha}"
        )
    reviewed_declared_path = manual_review.get("reviewed_training_manifest")
    reviewed_marker_path = _resolve_declared_file(
        reviewed_declared_path, workspace_root, context="manual-review reviewed manifest"
    )
    reviewed_sha = _sha(reviewed_manifest)
    if _sha(reviewed_marker_path) != reviewed_sha:
        raise V3ManifestError("manual-review marker points to a different reviewed training manifest")
    if str(manual_review.get("reviewed_training_manifest_sha256") or "") != reviewed_sha:
        raise V3ManifestError("manual-review reviewed manifest SHA does not match the requested file")
    marker_sha = _sha(marker_path)
    manual_review_sha = _sha(manual_review_path)

    reviewed_by_pair = _unique_manifest_map(
        reviewed_manifest,
        key_name="pair_id",
        expected_count=expected_reviewed_rows,
        allow_fixture_counts=allow_fixture_counts,
    )
    gate_rows = _expanded_rows(gate_manifest)
    gate_repairs = [
        row for row in gate_rows
        if str(row.get("decision")) == "REPAIR" and str(row.get("image_kind")) == "degraded"
    ]
    if not allow_fixture_counts and len(gate_repairs) != expected_gate_repair_rows:
        raise V3ManifestError(
            f"gate manifest expected {expected_gate_repair_rows} degraded REPAIR rows, got {len(gate_repairs)}"
        )
    seen_sample_ids: set[str] = set()
    seen_pair_ids: set[str] = set()
    sidecar: list[dict[str, Any]] = []
    mask_jobs: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
    ordinary_missing: list[str] = []
    missing_by_design: list[str] = []
    eligible_count = 0
    mask_count = 0

    for line_number, row in enumerate(gate_repairs, start=1):
        sample_id = str(row.get("sample_id") or "")
        pair_id = str(row.get("pair") or row.get("pair_id") or "")
        if not sample_id or not pair_id:
            raise V3ManifestError(f"gate repair row {line_number} is missing sample_id/pair")
        if sample_id in seen_sample_ids or pair_id in seen_pair_ids:
            raise V3ManifestError(f"gate repair rows contain duplicate sample/pair: {sample_id}/{pair_id}")
        seen_sample_ids.add(sample_id)
        seen_pair_ids.add(pair_id)
        hand_member = bool(row.get("hand_remove_member"))
        degraded = _expanded_image_path(row, workspace_root, context=f"gate:{sample_id}:degraded")
        declared_degraded_sha = str(row.get("degraded_sha256") or row.get("image_sha256") or "")
        image_meta = row.get("image") if isinstance(row.get("image"), Mapping) else {}
        declared_dimensions = image_meta.get("dimensions") if isinstance(image_meta, Mapping) else None
        if declared_dimensions in ([0, 0], (0, 0)):
            declared_dimensions = None
        degraded_ref = _portable_declared_existing(
            degraded,
            workspace_root,
            declared_sha256=declared_degraded_sha,
            declared_dimensions=declared_dimensions,
            probe=False,
            context=f"gate:{sample_id}:degraded",
        )

        common = {
            "schema_version": MASK_SCHEMA_VERSION,
            "sample_id": sample_id,
            "pair_id": pair_id,
            "pair": pair_id,
            "image_kind": "degraded",
            "decision": "REPAIR",
            "task": row.get("task") or row.get("task_id"),
            "task_id": row.get("task_id") or row.get("task"),
            "degradation": row.get("degradation") or row.get("task"),
            "reasons": list(row.get("reasons") or []),
            "hand_remove_member": hand_member,
            "degraded": degraded_ref,
            "degraded_relative_path": degraded_ref["relative_path"],
            "degraded_sha256": degraded_ref["sha256"],
            "cache_root": {
                "root": "workspace",
                "relative_path": mask_cache_root.relative_to(workspace_root).as_posix(),
                "portable": f"workspace:{mask_cache_root.relative_to(workspace_root).as_posix()}",
            },
            "cache_marker": marker_path.relative_to(workspace_root).as_posix(),
            "cache_marker_sha256": marker_sha,
            "cache_manifest_sha256": source_manifest_sha,
            "manual_review_marker": manual_review_path.relative_to(workspace_root).as_posix(),
            "manual_review_sha256": manual_review_sha,
            "reviewed_manifest": reviewed_manifest.relative_to(workspace_root).as_posix(),
            "reviewed_manifest_sha256": reviewed_sha,
        }
        reviewed = reviewed_by_pair.get(pair_id)
        if reviewed is None:
            if not hand_member:
                ordinary_missing.append(sample_id)
                continue
            missing_by_design.append(sample_id)
            common.update(
                {
                    "subtle_eligible": False,
                    "subtle_eligibility_reason": SUBTLE_INELIGIBLE_REASON,
                    "mask_status": "missing_by_design",
                    "mask": None,
                    "gt": None,
                    "gt_relative_path": None,
                    "gt_sha256": row.get("gt_sha256"),
                }
            )
            sidecar.append(common)
            continue

        reviewed_degraded = _portable_path(
            reviewed.get("degraded_path") or reviewed.get("degraded_relative_path"),
            workspace_root,
            context=f"reviewed:{pair_id}:degraded",
        )
        reviewed_degraded_sha = str(reviewed.get("degraded_sha256") or "")
        if not reviewed_degraded_sha or reviewed_degraded_sha != degraded_ref["sha256"]:
            raise V3ManifestError(f"gate/reviewed degraded image is not aligned: {sample_id}")
        reviewed_gt = _portable_path(
            reviewed.get("gt_path") or reviewed.get("gt_relative_path"),
            workspace_root,
            context=f"reviewed:{pair_id}:gt",
        )
        gt_ref = _portable_declared_existing(
            reviewed_gt,
            workspace_root,
            declared_sha256=str(reviewed.get("gt_sha256") or ""),
            declared_dimensions=reviewed.get("dimensions"),
            probe=False,
            context=f"reviewed:{pair_id}:gt",
        )
        if str(row.get("gt_sha256") or "") and str(row.get("gt_sha256")) != gt_ref["sha256"]:
            raise V3ManifestError(f"gate/reviewed GT SHA is not aligned: {sample_id}")
        mask_path = _mask_for_reviewed_gt(mask_cache_root, reviewed.get("gt_relative_path"))
        # Mask decode/hash is the only newly trusted pixel operation.  Submit
        # independent files concurrently: sequential NFS reads make the
        # 43,936-row sidecar unnecessarily long to build.  Preserve all
        # manifest alignment checks above and validate the futures below.
        mask_jobs.append((sample_id, mask_path, common, gt_ref))

    mask_workers = max(1, int(os.environ.get("QWEN_GATE_MASK_WORKERS", "16")))
    with ThreadPoolExecutor(max_workers=mask_workers) as executor:
        futures = [
            executor.submit(
                _portable_existing,
                mask_path,
                workspace_root,
                context=f"mask:{sample_id}",
            )
            for sample_id, mask_path, _common, _gt_ref in mask_jobs
        ]
        for (sample_id, _mask_path, common, gt_ref), future in zip(mask_jobs, futures):
            try:
                mask_ref = future.result()
            except Exception as exc:
                # Every submitted job corresponds to a trusted/base pair;
                # unlike the explicit hand-only exception, a missing or
                # unreadable mask is always a publication blocker.
                raise V3ManifestError(
                    f"ordinary/base mask validation failed for {sample_id}; refusing publication: {exc}"
                ) from exc
            if mask_ref["dimensions"] != gt_ref["dimensions"]:
                raise V3ManifestError(
                    f"mask/GT dimensions mismatch: {sample_id} mask={mask_ref['dimensions']} gt={gt_ref['dimensions']}"
                )
            if mask_ref["mode"] not in {"L", "LA", "I", "I;16", "F"}:
                raise V3ManifestError(f"unexpected RMBG2 mask mode for {sample_id}: {mask_ref['mode']}")
            eligible_count += 1
            mask_count += 1
            common.update(
                {
                    "subtle_eligible": True,
                    "subtle_eligibility_reason": "trusted_reviewed_gt_and_rmbg2_mask",
                    "mask_status": "ok",
                    "mask": mask_ref,
                    "mask_relative_path": mask_ref["relative_path"],
                    "mask_sha256": mask_ref["sha256"],
                    "mask_dimensions": mask_ref["dimensions"],
                    "gt": gt_ref,
                    "gt_relative_path": gt_ref["relative_path"],
                    "gt_sha256": gt_ref["sha256"],
                }
            )
            # Use the pair lookup from the common row when available; the
            # sample id is intentionally opaque and must not be parsed.
            pair_key = str(common["pair_id"])
            common["reviewed_pair_manifest"] = reviewed_by_pair[pair_key].get("pair_id")
            common["reviewed_task_id"] = reviewed_by_pair[pair_key].get("task_id")
            sidecar.append(common)

    if ordinary_missing:
        raise V3ManifestError(
            "ordinary/base gate rows are missing trusted reviewed masks; refusing publication: "
            + ", ".join(ordinary_missing[:12])
        )
    if not allow_fixture_counts and len(missing_by_design) != expected_missing_by_design_rows:
        raise V3ManifestError(
            f"expected {expected_missing_by_design_rows} hand-only missing-by-design rows, got {len(missing_by_design)}"
        )
    if len(sidecar) != len(gate_repairs):
        raise V3ManifestError(
            f"mask sidecar mapping incomplete: emitted {len(sidecar)} of {len(gate_repairs)} gate repair rows"
        )
    sidecar_ids = {str(row["sample_id"]) for row in sidecar}
    if sidecar_ids != seen_sample_ids:
        raise V3ManifestError("mask sidecar sample_id mapping is not one-to-one")

    destination = _abs(output_dir)
    if destination.exists():
        raise V3ManifestError(f"refusing to overwrite existing V3 mask sidecar output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + f".partial-{os.getpid()}")
    staging.mkdir(parents=True)
    audit = {
        "schema_version": MASK_SCHEMA_VERSION,
        "status": "ok",
        "gate_manifest": _audit_path(gate_manifest, workspace_root),
        "gate_manifest_absolute": str(gate_manifest),
        "gate_manifest_sha256": _sha(gate_manifest),
        "gate_repair_rows": len(gate_repairs),
        "sidecar_rows": len(sidecar),
        "reviewed_manifest": _audit_path(reviewed_manifest, workspace_root),
        "reviewed_manifest_absolute": str(reviewed_manifest),
        "reviewed_manifest_sha256": reviewed_sha,
        "reviewed_manifest_rows": len(reviewed_by_pair),
        "trusted_eligible_rows": eligible_count,
        "missing_by_design_rows": len(missing_by_design),
        "missing_by_design_reason": SUBTLE_INELIGIBLE_REASON,
        "missing_by_design_sample_examples": missing_by_design[:12],
        "ordinary_base_missing_rows": len(ordinary_missing),
        "ordinary_base_missing_is_fatal": True,
        "mask_count": mask_count,
        "source_manifest": _audit_path(source_manifest, workspace_root),
        "source_manifest_absolute": str(source_manifest),
        "source_manifest_sha256": source_manifest_sha,
        "cache_root": _audit_path(mask_cache_root, workspace_root),
        "cache_root_absolute": str(mask_cache_root),
        "cache_marker": _audit_path(marker_path, workspace_root),
        "cache_marker_absolute": str(marker_path),
        "cache_marker_sha256": marker_sha,
        "cache_marker_snapshot": marker,
        "manual_review_marker": _audit_path(manual_review_path, workspace_root),
        "manual_review_marker_absolute": str(manual_review_path),
        "manual_review_sha256": manual_review_sha,
        "manual_review_snapshot": manual_review,
        "mask_schema": RMBG2_MASK_SCHEMA,
        "mapping": "one gate REPAIR degraded sample -> one pair_id -> one reviewed GT -> one RMBG2 mask",
        "portable_root": "workspace",
        "dimensions_must_match_gt": True,
    }
    try:
        _write_jsonl(staging / "gate_mask_sidecar.jsonl", sidecar)
        _write_json(staging / "mask_sidecar_audit.json", audit)
        _write_json(
            staging / "schema.json",
            {
                "schema_version": MASK_SCHEMA_VERSION,
                "key": "sample_id",
                "pair_key": "pair_id",
                "mask_field": "mask",
                "required_fields": [
                    "sample_id", "pair_id", "subtle_eligible", "subtle_eligibility_reason",
                    "mask_status", "cache_marker_sha256", "reviewed_manifest_sha256",
                ],
                "missing_by_design_reason": SUBTLE_INELIGIBLE_REASON,
            },
        )
        try:
            staging.rename(destination)
        except PermissionError:
            if destination.exists():
                raise V3ManifestError(f"refusing to overwrite existing V3 mask sidecar output: {destination}")
            shutil.copytree(staging, destination)
            shutil.rmtree(staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit


def _weight_for_row(row: Mapping[str, Any], *, subtle: bool) -> tuple[float, float, str]:
    decision = str(row.get("decision"))
    if decision != "REPAIR":
        return 1.0, 1.0, "keep_default"
    reasons = [str(reason) for reason in row.get("reasons", [])]
    task = str(row.get("task") or row.get("task_id") or "")
    task_family = task.removesuffix("_heavy")
    # Prefer the explicit task family when present.  This preserves the
    # special 2.0 animal_motion_blur weight; its canonical reason is shared
    # with the ordinary motion_blur family.
    families = {task_family} if task_family in FAMILY_WEIGHTS else set()
    families.update(REASON_TO_FAMILY[reason] for reason in reasons if reason in REASON_TO_FAMILY)
    family_weight = max((FAMILY_WEIGHTS.get(family, 1.0) for family in families), default=1.0)
    value = family_weight
    if subtle:
        value = min(2.5, value * 1.25)
    return value, family_weight, "repair_family_weight"


def _read_mask_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise V3ManifestError(f"mask manifest line {line_number} must be an object")
            schema = value.get("schema_version")
            if schema not in (None, MASK_SCHEMA_VERSION):
                raise V3ManifestError(
                    f"mask manifest line {line_number} has unsupported schema_version: {schema!r}"
                )
            sample_id = str(value.get("sample_id") or "")
            pair_id = str(value.get("pair_id") or value.get("pair") or "")
            if not sample_id and not pair_id:
                raise V3ManifestError(f"mask manifest line {line_number} has no sample_id/pair_id")
            mask = value.get("mask")
            mask_path = value.get("mask_path") or value.get("path")
            if mask is not None and not isinstance(mask, Mapping):
                raise V3ManifestError(f"mask manifest {sample_id or pair_id} has a non-object mask field")
            if mask is None and not mask_path and value.get("subtle_eligible", True):
                raise V3ManifestError(f"mask manifest {sample_id or pair_id} has no mask reference")
            keys = {key for key in (sample_id, pair_id) if key}
            for key in keys:
                previous = rows.get(key)
                if previous is not None and str(previous.get("sample_id") or "") != sample_id:
                    raise V3ManifestError(f"mask manifest duplicate/conflicting key at line {line_number}: {key}")
                rows[key] = value
    return rows


def _mask_item_path(item: Mapping[str, Any], workspace_root: Path, *, context: str) -> Path:
    mask = item.get("mask")
    if isinstance(mask, Mapping):
        mask_path = mask.get("relative_path") or mask.get("workspace_relative_path") or mask.get("path")
        root_name = str(mask.get("root") or "workspace")
        if mask_path:
            if root_name == "workspace":
                return _portable_path(_abs(workspace_root) / str(mask_path), workspace_root, context=context)
            return _resolve_source(mask_path, workspace_root)
    mask_path = item.get("mask_path") or item.get("path")
    if mask_path:
        return _resolve_source(mask_path, workspace_root)
    raise V3ManifestError(f"{context}: mask manifest row has no mask path")


def _mask_proxy(
    degraded: Path,
    clean: Path,
    mask: Path,
    *,
    size: int = 128,
) -> float:
    if Image is None:
        raise V3ManifestError("Pillow is required for subtle proxy generation")
    try:
        with Image.open(degraded) as d, Image.open(clean) as g, Image.open(mask) as m:
            d_rgb = d.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            g_rgb = g.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            alpha = m.convert("L").resize((size, size), Image.Resampling.BILINEAR)
            import numpy as np

            d_arr = np.asarray(d_rgb, dtype=np.float32)
            g_arr = np.asarray(g_rgb, dtype=np.float32)
            m_arr = np.asarray(alpha, dtype=np.float32) / 255.0
            valid = m_arr > 0.5
            if not bool(valid.any()):
                raise V3ManifestError(f"RMBG2 mask has no foreground pixels: {mask}")
            return float(np.abs(d_arr - g_arr).mean(axis=2)[valid].mean())
    except V3ManifestError:
        raise
    except Exception as exc:
        raise V3ManifestError(f"subtle proxy failed for {degraded}/{clean}/{mask}: {exc}") from exc


def _mask_proxy_checked(
    sample_id: str,
    degraded: Path,
    clean: Path,
    mask: Path,
    declared_mask_sha: str,
) -> float:
    """Compute one proxy and verify the newly trusted mask artifact."""

    if not declared_mask_sha:
        raise V3ManifestError(f"subtle mask SHA is missing for {sample_id}: {mask}")
    if str(declared_mask_sha).lower() != _sha(mask):
        raise V3ManifestError(f"subtle mask SHA mismatch for {sample_id}: {mask}")
    return _mask_proxy(degraded, clean, mask)


def build_weight_sidecar(
    *,
    expanded_manifest: Path,
    output_dir: Path,
    workspace_root: Path = WORKSPACE,
    mask_manifest: Path | None = None,
    require_subtle_proxy: bool = False,
    allow_fixture_counts: bool = False,
) -> dict[str, Any]:
    rows = _expanded_rows(_abs(expanded_manifest))
    ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise V3ManifestError("expanded manifest has duplicate or empty sample_id")
    # The expanded V2 manifest contains one gt_clean row per pair.  Build the
    # lookup once so subtle-proxy generation is linear in the manifest size;
    # the previous next(...) scan made the 44k-row build O(n^2).
    gt_rows_by_pair: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        if str(row.get("image_kind")) != "gt_clean":
            continue
        pair_id = str(row.get("pair") or row.get("pair_id") or "")
        if not pair_id:
            raise V3ManifestError(f"expanded manifest:{line_number}: gt_clean row has no pair")
        if pair_id in gt_rows_by_pair:
            raise V3ManifestError(f"expanded manifest has duplicate gt_clean rows for pair {pair_id}")
        gt_rows_by_pair[pair_id] = row
    proxy_values: dict[str, float] = {}
    subtle_ids: set[str] = set()
    mask_items_by_sample: dict[str, Mapping[str, Any]] = {}
    proxy_audit: dict[str, Any] = {"status": "unavailable", "reason": "no RMBG2 mask manifest supplied"}
    if mask_manifest is not None:
        mask_manifest = _abs(mask_manifest)
        mask_sidecar_audit_path = mask_manifest.parent / "mask_sidecar_audit.json"
        mask_sidecar_audit: dict[str, Any] | None = None
        if require_subtle_proxy:
            if not mask_sidecar_audit_path.is_file():
                raise V3ManifestError(
                    "subtle proxy is required but the mask-sidecar audit is missing: "
                    f"{mask_sidecar_audit_path}"
                )
            value = _read_json(mask_sidecar_audit_path)
            if not isinstance(value, dict) or value.get("schema_version") != MASK_SCHEMA_VERSION:
                raise V3ManifestError(
                    f"invalid RMBG2 mask-sidecar audit schema: {mask_sidecar_audit_path}"
                )
            if value.get("status") != "ok":
                raise V3ManifestError(
                    f"RMBG2 mask-sidecar audit is not complete: {mask_sidecar_audit_path}"
                )
            mask_sidecar_audit = value
        masks = _read_mask_rows(mask_manifest)
        by_family: dict[str, list[tuple[str, float]]] = defaultdict(list)
        missing_by_design: list[str] = []
        ordinary_missing: list[str] = []
        ordinary_ineligible: list[str] = []
        family_audit: dict[str, dict[str, int]] = defaultdict(lambda: {"eligible": 0, "subtle": 0})
        proxy_jobs: list[tuple[str, str, Path, Path, Path, str]] = []
        for row in rows:
            if row.get("decision") != "REPAIR" or row.get("image_kind") != "degraded":
                continue
            sample_id = str(row["sample_id"])
            pair_id = str(row.get("pair") or row.get("pair_id") or "")
            item = masks.get(sample_id) or masks.get(pair_id)
            if item is None:
                if bool(row.get("hand_remove_member")):
                    missing_by_design.append(sample_id)
                else:
                    ordinary_missing.append(sample_id)
                continue
            mask_items_by_sample[sample_id] = item
            subtle_eligible = bool(item.get("subtle_eligible", True))
            if not subtle_eligible:
                reason = str(item.get("subtle_eligibility_reason") or "")
                if bool(row.get("hand_remove_member")) and reason == SUBTLE_INELIGIBLE_REASON:
                    missing_by_design.append(sample_id)
                    continue
                ordinary_ineligible.append(sample_id)
                continue
            mask = _mask_item_path(item, workspace_root, context=sample_id)
            declared_sha = item.get("mask_sha256") or item.get("sha256")
            nested_mask = item.get("mask")
            if not declared_sha and isinstance(nested_mask, Mapping):
                declared_sha = nested_mask.get("sha256")
            degraded = _expanded_image_path(row, workspace_root, context=f"gate:{sample_id}:degraded")
            clean_row = gt_rows_by_pair.get(pair_id)
            if clean_row is None:
                raise V3ManifestError(f"no aligned gt_clean row for subtle proxy: {sample_id}")
            clean = _expanded_image_path(clean_row, workspace_root, context=f"gate:{sample_id}:gt_clean")
            item_gt_sha = str(item.get("gt_sha256") or "")
            clean_declared_sha = str(clean_row.get("image_sha256") or clean_row.get("gt_sha256") or "")
            if item_gt_sha and clean_declared_sha and item_gt_sha != clean_declared_sha:
                raise V3ManifestError(f"mask sidecar GT SHA is not aligned: {sample_id}")
            family = str(row.get("degradation") or row.get("task") or "unknown")
            proxy_jobs.append((sample_id, family, degraded, clean, mask, str(declared_sha or "")))
            family_audit[family]["eligible"] += 1
        if ordinary_missing or ordinary_ineligible:
            raise V3ManifestError(
                "ordinary/base gate rows are missing or ineligible in the subtle-proxy mask sidecar: "
                + ", ".join((ordinary_missing + ordinary_ineligible)[:8])
            )
        repair_count = sum(
            row.get("decision") == "REPAIR" and row.get("image_kind") == "degraded"
            for row in rows
        )
        if require_subtle_proxy and mask_sidecar_audit is not None:
            if int(mask_sidecar_audit.get("sidecar_rows", -1)) != repair_count:
                raise V3ManifestError(
                    "mask-sidecar audit row count does not cover the expanded gate REPAIR rows: "
                    f"audit={mask_sidecar_audit.get('sidecar_rows')} gate={repair_count}"
                )
            if int(mask_sidecar_audit.get("ordinary_base_missing_rows", -1)) != 0:
                raise V3ManifestError("mask-sidecar audit reports ordinary/base missing rows")
        proxy_workers = max(1, int(os.environ.get("QWEN_GATE_PROXY_WORKERS", "16")))
        with ThreadPoolExecutor(max_workers=proxy_workers) as executor:
            futures = [
                executor.submit(
                    _mask_proxy_checked,
                    sample_id,
                    degraded,
                    clean,
                    mask,
                    declared_sha,
                )
                for sample_id, _family, degraded, clean, mask, declared_sha in proxy_jobs
            ]
            for (sample_id, family, _degraded, _clean, _mask, _declared_sha), future in zip(proxy_jobs, futures):
                try:
                    value = future.result()
                except Exception as exc:
                    raise V3ManifestError(f"subtle proxy failed for {sample_id}: {exc}") from exc
                proxy_values[sample_id] = value
                by_family[family].append((sample_id, value))
        for family, values in by_family.items():
            values.sort(key=lambda item: (item[1], item[0]))
            subtle_count = max(1, int(math.ceil(len(values) * 0.25)))
            subtle_ids.update(sample_id for sample_id, _value in values[:subtle_count])
            family_audit[family]["subtle"] = subtle_count
        proxy_audit = {
            "status": "ok",
            "mask_manifest": _audit_path(mask_manifest, workspace_root),
            "mask_manifest_absolute": str(mask_manifest),
            "mask_manifest_sha256": _sha(mask_manifest),
            "mask_sidecar_audit": _audit_path(mask_sidecar_audit_path, workspace_root)
            if mask_sidecar_audit is not None
            else None,
            "mask_sidecar_audit_sha256": _sha(mask_sidecar_audit_path)
            if mask_sidecar_audit is not None
            else None,
            "proxy_sample_count": len(proxy_values),
            "eligible_sample_count": len(proxy_values),
            "subtle_sample_count": len(subtle_ids),
            "missing_by_design_count": len(missing_by_design),
            "missing_by_design_sample_examples": missing_by_design[:12],
            "ordinary_base_missing_count": len(ordinary_missing),
            "ordinary_base_ineligible_count": len(ordinary_ineligible),
            "family_counts": dict(family_audit),
            "definition": "aligned degraded/GT, RMBG2 foreground, 128x128 masked RGB MAD; family bottom25%",
        }
    elif require_subtle_proxy:
        raise V3ManifestError("subtle proxy is required but no RMBG2 mask manifest was supplied")

    sidecar: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        is_subtle = sample_id in subtle_ids and row.get("decision") == "REPAIR" and row.get("image_kind") == "degraded"
        weight, family_weight, source = _weight_for_row(row, subtle=is_subtle)
        mask_item = mask_items_by_sample.get(sample_id)
        mask_ref = mask_item.get("mask") if isinstance(mask_item, Mapping) else None
        sidecar.append(
            {
                "schema_version": WEIGHT_SCHEMA_VERSION,
                "sample_id": sample_id,
                "pair": row.get("pair"),
                "image_kind": row.get("image_kind"),
                "decision": row.get("decision"),
                "task": row.get("task") or row.get("task_id"),
                "reasons": list(row.get("reasons") or []),
                "severity_tag": row.get("severity_tag"),
                "severity_tag_provenance_only": row.get("severity_tag_provenance_only") is True,
                "mask_status": mask_item.get("mask_status") if isinstance(mask_item, Mapping) else None,
                "subtle_eligible": bool(mask_item.get("subtle_eligible", True)) if isinstance(mask_item, Mapping) else False,
                "subtle_eligibility_reason": (
                    mask_item.get("subtle_eligibility_reason") if isinstance(mask_item, Mapping)
                    else "no_mask_sidecar"
                ),
                "mask": mask_ref,
                "mask_sha256": (
                    mask_item.get("mask_sha256") if isinstance(mask_item, Mapping) else None
                ),
                "mask_dimensions": (
                    mask_item.get("mask_dimensions") if isinstance(mask_item, Mapping) else None
                ),
                "subtle_proxy": bool(is_subtle),
                "subtle_proxy_mad": proxy_values.get(sample_id),
                "family_weight": family_weight,
                "sample_weight": weight,
                "weight_source": source,
                "heavy_semantics": "provenance-only; never activates subtle proxy",
            }
        )
    repair_rows = [row for row in sidecar if row["decision"] == "REPAIR"]
    family_counts = Counter(str(row["task"]) for row in repair_rows)
    audit = {
        "schema_version": WEIGHT_SCHEMA_VERSION,
        "status": "ok",
        "expanded_manifest": str(_abs(expanded_manifest)),
        "rows": len(sidecar),
        "repair_rows": len(repair_rows),
        "keep_rows": len(sidecar) - len(repair_rows),
        "subtle_proxy": proxy_audit,
        "subtle_rows": len(subtle_ids),
        "family_counts": dict(family_counts),
        "family_weights": FAMILY_WEIGHTS,
        "subtle_multiplier": 1.25,
        "max_sample_weight": 2.5,
        "heavy_semantics": "*_heavy is provenance only and never marks a sample subtle",
        "missing_subtle_proxy_is_fail_closed": True,
    }
    destination = _abs(output_dir)
    if destination.exists():
        raise V3ManifestError(f"refusing to overwrite existing V3 weight output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + f".partial-{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        _write_jsonl(staging / "sample_weights.jsonl", sidecar)
        _write_json(staging / "weights_audit.json", audit)
        _write_json(staging / "schema.json", {
            "schema_version": WEIGHT_SCHEMA_VERSION,
            "key": "sample_id",
            "required_fields": ["sample_id", "sample_weight", "subtle_proxy", "severity_tag_provenance_only"],
            "family_weights": FAMILY_WEIGHTS,
            "subtle_proxy": proxy_audit,
        })
        try:
            staging.rename(destination)
        except PermissionError:
            if destination.exists():
                raise V3ManifestError(f"refusing to overwrite existing V3 weight output: {destination}")
            shutil.copytree(staging, destination)
            shutil.rmtree(staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit



def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build audited router mask and sample-weight sidecars.")
    sub = parser.add_subparsers(dest="command", required=True)
    masks = sub.add_parser("masks")
    masks.add_argument("--router-manifest", type=Path, default=Path("data/router/train.jsonl"))
    masks.add_argument("--reviewed-manifest", type=Path, default=Path("data/manifests/train.jsonl"))
    masks.add_argument("--source-manifest", type=Path, default=Path("data/manifests/train.jsonl"))
    masks.add_argument("--mask-cache-root", type=Path, default=Path("outputs/masks"))
    masks.add_argument("--output-dir", type=Path, default=Path("data/router/masks"))
    weights = sub.add_parser("weights")
    weights.add_argument("--router-manifest", type=Path, default=Path("data/router/train.jsonl"))
    weights.add_argument("--mask-manifest", type=Path, default=Path("data/router/masks/gate_mask_sidecar.jsonl"))
    weights.add_argument("--output-dir", type=Path, default=Path("data/router/weights"))
    args = parser.parse_args(argv)
    if args.command == "masks":
        result = build_mask_sidecar(gate_manifest=args.router_manifest, reviewed_manifest=args.reviewed_manifest,
            source_manifest=args.source_manifest, mask_cache_root=args.mask_cache_root,
            output_dir=args.output_dir, workspace_root=WORKSPACE, allow_fixture_counts=True)
    else:
        result = build_weight_sidecar(expanded_manifest=args.router_manifest, mask_manifest=args.mask_manifest,
            output_dir=args.output_dir, workspace_root=WORKSPACE, require_subtle_proxy=True, allow_fixture_counts=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
