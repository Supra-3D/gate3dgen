from __future__ import annotations

import json
import logging
import os
import tempfile
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image

from . import SCHEMA_VERSION
from .experimental.recipes import (
    ALBEDO_RECIPES,
    cache_contract,
    is_albedo_recipe,
    recipe_fingerprint,
    recipe_from_config,
    write_cache_contract,
)
from .pairing import pairing_summary
from .data_defaults import INSTRUCTION_CN, INSTRUCTION_EN
from .prompts import restoration_instructions
from .subject_masks import MASK_SCHEMA, mask_path, mask_relative_path, validate_mask


logger = logging.getLogger("gate3dgen.training.albedo")


def _verify_pair_images(row: dict[str, Any]) -> None:
    expected = (
        int(row["edit_image_size"]["width"]),
        int(row["edit_image_size"]["height"]),
    )
    sizes = []
    for path in (Path(row["source_image"][0]), Path(row["edit_image"])):
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            sizes.append(image.size)
            image.verify()
    if sizes[0] != sizes[1]:
        raise ValueError(
            f"pair {row['pair_id']} source/GT size mismatch: {sizes[0]} vs {sizes[1]}"
        )
    if sizes[0] != expected:
        raise ValueError(
            f"pair {row['pair_id']} actual size {sizes[0]} != manifest size {expected}"
        )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            yield value


def _resolve_pair_path(row: dict[str, Any], data_root: Path, stem: str) -> Path:
    relative = row.get(f"{stem}_relative_path")
    if not relative:
        raise ValueError(f"pair {row.get('pair_id')} has no {stem}_relative_path")
    relative_path = Path(str(relative))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"pair path escapes data root: {relative_path}")
    # Use lexical normalization instead of Path.resolve(): resolving every image
    # path performs network-filesystem readlink calls and is prohibitively slow
    # for the 47k-pair Full dataset. Existence/decoding is checked separately.
    root = Path(os.path.abspath(os.fspath(data_root.expanduser())))
    path = Path(os.path.abspath(os.fspath(root / relative_path)))
    if os.path.commonpath((os.fspath(root), os.fspath(path))) != os.fspath(root):
        raise ValueError(f"pair path escapes data root: {path}")
    return path


def adapt_row(
    row: dict[str, Any],
    data_root: Path,
    require_quality_status: Optional[str] = None,
    *,
    instruction_data: Optional[dict[str, Any]] = None,
    subject_mask_root: Optional[Path] = None,
    require_subject_mask: bool = False,
) -> dict[str, Any]:
    if row.get("basic_audit_status") != "pass":
        raise ValueError(f"pair {row.get('pair_id')} did not pass generation audit")
    if require_quality_status and row.get("quality_status") != require_quality_status:
        raise ValueError(f"pair {row.get('pair_id')} quality_status={row.get('quality_status')!r}")
    degraded = _resolve_pair_path(row, data_root, "degraded")
    gt = _resolve_pair_path(row, data_root, "gt")
    selected_instructions = instruction_data or {
        "instruction": INSTRUCTION_EN,
        "instruction_cn": INSTRUCTION_CN,
        "prompt_library": None,
        "prompt_library_id": None,
        "prompt_library_fingerprint": None,
    }
    prepared = {
        "schema_version": SCHEMA_VERSION,
        "pair_id": str(row["pair_id"]),
        "asset_id": str(row["sha256"]),
        "dataset": "TexVerse",
        "task_id": str(row["task_id"]),
        "degradation_type": str(row.get("degradation_type", row["task_id"])),
        "split": str(row.get("split", "unknown")),
        "source_image": [str(degraded)],
        "edit_image": str(gt),
        "instruction": selected_instructions["instruction"],
        "instruction_cn": selected_instructions["instruction_cn"],
        "inverse_instruction": "",
        "inverse_instruction_cn": "",
        "source_image_size": [{"width": int(row["dimensions"][0]), "height": int(row["dimensions"][1])}],
        "edit_image_size": {"width": int(row["dimensions"][0]), "height": int(row["dimensions"][1])},
    }
    for key in ("prompt_library", "prompt_library_id", "prompt_library_fingerprint"):
        if selected_instructions.get(key) is not None:
            prepared[key] = selected_instructions[key]
    if subject_mask_root is not None:
        relative_mask = mask_relative_path(row.get("gt_relative_path"))
        resolved_mask = mask_path(subject_mask_root, row.get("gt_relative_path"))
        if require_subject_mask and not resolved_mask.is_file():
            raise FileNotFoundError(
                f"required offline GT subject mask is missing: {resolved_mask}"
            )
        prepared.update(
            {
                "subject_mask_image": str(resolved_mask),
                "subject_mask_relative_path": relative_mask.as_posix(),
                "subject_mask_schema": MASK_SCHEMA,
                "subject_mask_source": "clean_gt_rmbg2",
            }
        )
    return prepared


class AlbedoSkip(ValueError):
    """A deterministic, reportable reason to omit one albedo-supervised row."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        message = self.reason if not self.detail else f"{self.reason}: {self.detail}"
        super().__init__(message)


def _resolve_albedo_path(relative: Any, albedo_root: Path, pair_id: Any) -> Path:
    """Resolve manifest.source_relative_path under the authoritative TexVerse root.

    ``Path.resolve`` is intentionally used here (unlike the hot legacy path
    resolver) because a symlink escaping TexVerse is itself an invalid albedo
    input and must be skipped rather than silently read.
    """

    if not isinstance(relative, str) or not relative.strip():
        raise AlbedoSkip("missing_or_invalid_source_relative_path", repr(relative))
    # Treat Windows separators as separators even on the Linux training
    # mount; otherwise ``..\\escape.png`` would be treated as a literal
    # filename during validation and the contract would differ by host OS.
    relative_text = relative.replace("\\", "/")
    relative_path = Path(relative_text)
    # POSIX Path.is_absolute() does not recognize a Windows drive prefix;
    # reject it explicitly so a cross-platform manifest cannot bypass the
    # authoritative-root contract on Linux.
    windows_absolute = len(relative_text) >= 2 and relative_text[1] == ":"
    if relative_path.is_absolute() or windows_absolute or ".." in relative_path.parts:
        raise AlbedoSkip("albedo_path_escape", str(relative_path))
    root = albedo_root.expanduser().resolve(strict=False)
    candidate = (root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise AlbedoSkip("albedo_path_escape", str(candidate)) from error
    return candidate


def _verify_albedo(path: Path, expected: tuple[int, int], pair_id: Any) -> None:
    if not path.is_file():
        raise AlbedoSkip("albedo_missing", str(path))
    try:
        with Image.open(path) as image:
            actual = tuple(image.size)
            image.verify()
    except Exception as error:
        raise AlbedoSkip("albedo_decode_error", f"{path}: {error}") from error
    if actual != expected:
        raise AlbedoSkip(
            "albedo_size_mismatch",
            f"actual={actual} expected={expected}",
        )


def adapt_albedo_row(
    row: dict[str, Any],
    data_root: Path,
    albedo_root: Path,
    require_quality_status: Optional[str] = None,
    *,
    recipe: Optional[str] = None,
    decode: bool = True,
    instruction_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Adapt one row and add validated albedo provenance.

    The legacy :func:`adapt_row` remains unchanged; this opt-in adapter is
    only called by the three albedo recipes.  Pair/GT corruption still raises
    as before, while all albedo-specific failures become ``AlbedoSkip``.
    """

    # The albedo prepare pass is fail-closed per row.  Do not let a malformed
    # source/GT field abort the entire 43k-row filtering job.
    if not isinstance(row, dict):
        raise AlbedoSkip("missing_or_invalid_field", repr(row))
    if not row.get("source_relative_path"):
        raise AlbedoSkip("missing_source_relative_path")
    try:
        prepared = adapt_row(
            row,
            data_root,
            require_quality_status,
            instruction_data=instruction_data,
        )
    except FileNotFoundError as error:
        raise AlbedoSkip("pair_file_missing", str(error)) from error
    except (KeyError, TypeError, ValueError, OSError) as error:
        message = str(error)
        lowered = message.lower()
        if "escape" in lowered or "absolute" in lowered or "relative_path" in lowered:
            reason = "pair_path_invalid"
        elif "size" in lowered or "dimension" in lowered:
            reason = "pair_size_mismatch"
        elif "decode" in lowered or "image" in lowered or "cannot identify" in lowered:
            reason = "pair_decode_error"
        else:
            reason = "missing_or_invalid_field"
        raise AlbedoSkip(reason, message) from error
    expected = (
        int(prepared["edit_image_size"]["width"]),
        int(prepared["edit_image_size"]["height"]),
    )
    # If a producer supplies an explicit source/albedo dimension field, make
    # the same-camera pixel-grid contract observable at prepare time.  Older
    # manifests only have ``dimensions`` and are still accepted; the actual
    # degraded/GT bytes and canonical albedo bytes are checked below.
    for field in ("source_dimensions", "albedo_dimensions"):
        supplied = row.get(field)
        if supplied is None:
            continue
        if isinstance(supplied, dict):
            supplied = [supplied.get("width"), supplied.get("height")]
        if not isinstance(supplied, (list, tuple)) or len(supplied) < 2:
            raise AlbedoSkip("missing_or_invalid_field", f"{field}={supplied!r}")
        try:
            supplied_size = (int(supplied[0]), int(supplied[1]))
        except (TypeError, ValueError) as error:
            raise AlbedoSkip("missing_or_invalid_field", f"{field}={supplied!r}") from error
        if supplied_size != expected:
            raise AlbedoSkip(
                "source_dimension_mismatch",
                f"{field}={supplied_size} expected={expected}",
            )
    # Never fall back to source_path/source_original_path.  Those fields point
    # at stale absolute mounts in historical manifests.
    relative = row.get("source_relative_path")
    try:
        albedo_path = _resolve_albedo_path(relative, albedo_root, row.get("pair_id"))
    except AlbedoSkip:
        raise
    except (OSError, ValueError) as error:
        raise AlbedoSkip("albedo_path_invalid", str(error)) from error
    _verify_albedo(albedo_path, expected, row.get("pair_id"))
    if decode:
        try:
            _verify_pair_images(prepared)
        except FileNotFoundError as error:
            raise AlbedoSkip("pair_file_missing", str(error)) from error
        except (OSError, ValueError) as error:
            message = str(error)
            reason = "pair_size_mismatch" if "size" in message.lower() else "pair_decode_error"
            raise AlbedoSkip(reason, message) from error
    relative_text = str(relative).replace("\\", "/")
    prepared.update(
        {
            # Keep both names while the cache migrates; the training provider
            # accepts ``albedo_image`` and external tools can use ``albedo_path``.
            "albedo_image": str(albedo_path),
            "albedo_path": str(albedo_path),
            "albedo_relative_path": relative_text,
            "albedo_group_id": relative_text,
            "source_relative_path": relative_text,
            "source_original_path": row.get("source_original_path"),
            "source_manifest_path": row.get("source_path"),
            "same_camera_same_pixel": True,
            "recipe": recipe or "",
        }
    )
    # Preserve the raw identity fields needed by cache fingerprints.  Do not
    # copy historical absolute paths except as provenance: runtime loading
    # continues to use only the validated relative fields above.
    for key in (
        "degraded_relative_path",
        "gt_relative_path",
        "dimensions",
        "view_id",
        "source_dimensions",
        "albedo_dimensions",
    ):
        if key in row:
            prepared[key] = row[key]
    for key, value in row.items():
        if key == "sha256" or key.endswith("_sha256"):
            prepared[key] = value
    if isinstance(row.get("raw_sha256"), dict):
        prepared["raw_sha256"] = dict(row["raw_sha256"])
    return prepared


def _validation_config_without_exact_counts(config: dict[str, Any], split: Optional[str]) -> dict[str, Any]:
    """Remove V2's raw-manifest counts after deterministic albedo filtering.

    The filtered size is an observed property (and is recorded in the report),
    not a hard-coded expectation.  Structural checks such as unique IDs and
    split consistency remain enabled.
    """

    filtered = copy.deepcopy(config)
    data = filtered.setdefault("data", {})
    for key in ("expected_pairs", "expected_assets", "expected_tasks", "expected_per_task", "expected_task_counts"):
        data.pop(key, None)
    if split is not None:
        split_expectations = data.get("split_expectations")
        if isinstance(split_expectations, dict):
            split_expectations.pop(split, None)
    return filtered


def _atomic_json_dump(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _albedo_report_path(destination: Path, config: dict[str, Any], split: Optional[str]) -> Path:
    configured = config.get("data", {}).get("albedo_report")
    if configured:
        report = Path(str(configured))
        if not report.is_absolute():
            report = destination.parent / report
        # A full config prepares train and test into the same cache parent.
        # Keep the historical train filename, but avoid letting a later test
        # pass overwrite its per-row report.
        if split and split != "train":
            report = report.with_name(f"{report.stem}.{split}{report.suffix}")
        return report
    suffix = f".{split}" if split else ""
    return destination.parent / f"albedo_prepare_report{suffix}.json"


def _albedo_contract_path(destination: Path, split: Optional[str]) -> Path:
    """Return a split-local marker when train/test share one cache folder."""

    marker = destination.parent / "recipe_contract.json"
    if split and split != "train":
        marker = marker.with_name(f"{marker.stem}.{split}{marker.suffix}")
    return marker


def prepare_albedo_manifest(
    config: dict[str, Any],
    destination: Path,
    decode: bool = True,
    *,
    source_manifest: Optional[Path] = None,
    split: Optional[str] = None,
) -> dict[str, Any]:
    """Prepare one of the three albedo recipes with fail-closed row filtering."""

    recipe = recipe_from_config(config)
    if recipe not in ALBEDO_RECIPES:
        raise ValueError(f"prepare_albedo_manifest requires an albedo recipe, got {recipe!r}")
    configured_source = config["data"].get("manifest") or (
        config["data"].get("manifests", {}) or {}
    ).get("train")
    if source_manifest is None and not configured_source:
        raise ValueError("albedo config must define data.manifest or data.manifests.train")
    source = source_manifest or Path(configured_source)
    if not source.is_file():
        raise FileNotFoundError(source)
    data_root = Path(config["data"]["data_root"])
    albedo_root_value = config["data"].get("albedo_root")
    if not isinstance(albedo_root_value, (str, os.PathLike)) or not str(albedo_root_value).strip():
        raise ValueError("albedo recipe requires data.albedo_root")
    albedo_root = Path(albedo_root_value)
    required = config["data"].get("require_quality_status")
    instruction_data = restoration_instructions(config)
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    input_rows = 0
    def record_skip(
        line_number: int,
        pair_id: str,
        reason: str,
        detail: str = "",
        source_relative_path: Any = None,
    ) -> None:
        event = {
            "event": "ALBEDO_SKIP",
            "pair_id": pair_id,
            "line": line_number,
            "reason": str(reason),
            "detail": str(detail),
            "source_relative_path": source_relative_path,
        }
        skipped.append(event)
        # One structured line per skipped row is intentionally emitted;
        # downstream launch logs can be searched without parsing prose.
        logger.warning("ALBEDO_SKIP %s", json.dumps(event, ensure_ascii=False, sort_keys=True))

    # Parse this manifest locally rather than through iter_jsonl so even a
    # malformed JSON line is a deterministic per-row skip, not a partial run
    # with no summary/report.
    with source.open(encoding="utf-8") as source_handle:
        for line_number, raw_line in enumerate(source_handle, 1):
            if not raw_line.strip():
                continue
            input_rows += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                record_skip(line_number, f"line-{line_number}", "invalid_json", str(error))
                continue
            if not isinstance(row, dict):
                record_skip(line_number, f"line-{line_number}", "missing_or_invalid_field", "row is not an object")
                continue
            pair_id = str(row.get("pair_id", f"line-{line_number}"))
            try:
                prepared = adapt_albedo_row(
                    row,
                    data_root,
                    albedo_root,
                    required,
                    recipe=recipe,
                    decode=decode,
                    instruction_data=instruction_data,
                )
            except AlbedoSkip as error:
                record_skip(
                    line_number,
                    pair_id,
                    error.reason,
                    error.detail,
                    row.get("source_relative_path"),
                )
                continue
            accepted.append(prepared)

    fingerprint = recipe_fingerprint(
        recipe,
        config,
        accepted,
        source_manifest=str(source),
    )
    for prepared in accepted:
        prepared["recipe_fingerprint"] = fingerprint
        prepared["recipe_schema"] = "texverse-albedo-recipe-v1"

    # Compute the complete cache identity before committing rows so every
    # prepared record carries the same schema/cache markers as the sidecar.
    # The fingerprint payload intentionally ignores these marker fields.
    contract = cache_contract(
        config,
        accepted,
        prepared_manifest=str(destination),
        source_manifest=str(source),
    )
    manifest_digest = contract.get("source_manifest_sha256")
    for prepared in accepted:
        prepared["cache_schema"] = contract["cache_schema"]
        prepared["cache_fingerprint"] = contract["cache_fingerprint"]
        prepared["schema_fingerprint"] = contract["schema_fingerprint"]
        prepared["source_manifest_sha256"] = contract["source_manifest_sha256"]

    report_path = _albedo_report_path(destination, config, split)
    groups: dict[str, list[str]] = {}
    for row in accepted:
        groups.setdefault(str(row["albedo_group_id"]), []).append(str(row["pair_id"]))
    group_sizes = Counter(len(pair_ids) for pair_ids in groups.values())
    summary = {
        "event": "ALBEDO_PREP_SUMMARY",
        "recipe": recipe,
        "recipe_fingerprint": fingerprint,
        "source_manifest": str(source),
        "source_manifest_sha256": manifest_digest,
        "prepared_manifest": str(destination),
        "report_path": str(report_path),
        "split": split,
        "input_rows": input_rows,
        "accepted_rows": len(accepted),
        "skipped_rows": len(skipped),
        "skip_reasons": dict(sorted(Counter(item["reason"] for item in skipped).items())),
        "groups": len(groups),
        "singleton_groups": group_sizes.get(1, 0),
        "paired_groups": sum(count for size, count in group_sizes.items() if size > 1),
        "paired_rows": sum(size * count for size, count in group_sizes.items() if size > 1),
        "spatial_alignment": "same_camera_same_pixel_no_warp",
    }
    if recipe == "albedo_consistency":
        # Record the exact eligible-pair population without changing the
        # filtered row set.  Training later builds the same deterministic pairs
        # from these prepared rows.
        summary["pairing"] = pairing_summary(accepted)
    report = {**summary, "skips": skipped, "group_sizes": dict(sorted((str(k), v) for k, v in group_sizes.items()))}
    _atomic_json_dump(report, report_path)
    logger.info("ALBEDO_PREP_SUMMARY %s", json.dumps(summary, ensure_ascii=False, sort_keys=True))

    # Do not bake the raw V2 exact count into an albedo cache: missing albedo
    # rows are expected and their exact count must be discovered, not guessed.
    validation = validate_rows(
        accepted,
        _validation_config_without_exact_counts(config, split),
        # ``adapt_albedo_row`` already verifies pair/GT bytes when ``decode``
        # is true.  Reopening every image here would make the 43k-row pass do
        # the same three decodes twice; structural validation is sufficient at
        # this point, while ``decode=False`` preserves the CLI's opt-out.
        decode=False,
        split=split,
    ) if accepted else {"pairs": 0, "assets": 0, "tasks": {}}
    if not accepted:
        raise ValueError(f"all {input_rows} rows were skipped for albedo; report={report_path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in accepted:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    # Keep the marker next to the prepared split.  It is independent for each
    # recipe/cache namespace and is cheap to compare before embeddings/training.
    contract_path = _albedo_contract_path(destination, split)
    write_cache_contract(contract_path, contract)
    return {
        **summary,
        **validation,
        "source_manifest": str(source),
        "prepared_manifest": str(destination),
        "report_path": str(report_path),
        "cache_contract": str(contract_path),
    }


def validate_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    decode: bool = True,
    split: Optional[str] = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("manifest contains no accepted pairs")
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair_id values are not unique")
    data_config = config["data"]
    expectations = data_config.get("split_expectations", {}).get(split, {})

    def expected_value(key: str) -> Any:
        return expectations.get(key, data_config.get(key))

    if split is not None and "manifests" in data_config:
        wrong_split = Counter(str(row["split"]) for row in rows if str(row["split"]) != split)
        if wrong_split:
            raise ValueError(f"manifest {split!r} contains rows from other splits: {dict(wrong_split)}")
    expected_pairs = expected_value("expected_pairs")
    if expected_pairs is not None and len(rows) != int(expected_pairs):
        raise ValueError(f"expected {expected_pairs} pairs, got {len(rows)}")
    assets = {str(row["asset_id"]) for row in rows}
    expected_assets = expected_value("expected_assets")
    if expected_assets is not None and len(assets) != int(expected_assets):
        raise ValueError(f"expected {expected_assets} assets, got {len(assets)}")
    tasks = Counter(str(row["task_id"]) for row in rows)
    expected_tasks = expected_value("expected_tasks")
    if expected_tasks is not None and len(tasks) != int(expected_tasks):
        raise ValueError(f"expected {expected_tasks} tasks, got {len(tasks)}")
    expected_per_task = expected_value("expected_per_task")
    if expected_per_task is not None:
        wrong = {task: count for task, count in tasks.items() if count != int(expected_per_task)}
        if wrong:
            raise ValueError(f"expected {expected_per_task} pairs/task, got {wrong}")
    expected_task_counts = expected_value("expected_task_counts")
    if expected_task_counts is not None:
        normalized_expected = {
            str(task): int(count) for task, count in expected_task_counts.items()
        }
        actual = dict(sorted(tasks.items()))
        if actual != dict(sorted(normalized_expected.items())):
            raise ValueError(
                f"expected exact task counts {normalized_expected}, got {actual}"
            )
    if decode:
        workers = min(32, max(1, (os.cpu_count() or 1) * 4))
        def verify_row(row: dict[str, Any]) -> None:
            _verify_pair_images(row)
            subject_mask = row.get("subject_mask_image")
            if subject_mask:
                expected = (
                    int(row["edit_image_size"]["width"]),
                    int(row["edit_image_size"]["height"]),
                )
                validate_mask(Path(subject_mask), expected)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(verify_row, rows):
                pass
    return {"pairs": len(rows), "assets": len(assets), "tasks": dict(sorted(tasks.items()))}


def prepare_manifest(
    config: dict[str, Any],
    destination: Path,
    decode: bool = True,
    *,
    source_manifest: Optional[Path] = None,
    split: Optional[str] = None,
) -> dict[str, Any]:
    if is_albedo_recipe(recipe_from_config(config)):
        return prepare_albedo_manifest(
            config,
            destination,
            decode=decode,
            source_manifest=source_manifest,
            split=split,
        )
    source = source_manifest or Path(config["data"]["manifest"])
    if not source.is_file():
        raise FileNotFoundError(source)
    data_root = Path(config["data"]["data_root"])
    required = config["data"].get("require_quality_status")
    instruction_data = restoration_instructions(config)
    subject_mask_root_value = config["data"].get("subject_mask_root")
    # RMBG2 masks supervise restoration losses and are consumed only by the
    # training dataloader.  Evaluation uses degraded/GT RGB directly; attaching
    # a nonexistent train-only mask to test rows makes otherwise valid blind
    # test data impossible to prepare and provides no inference signal.
    training_split = split in (None, "train")
    subject_mask_root = (
        Path(subject_mask_root_value)
        if training_split and subject_mask_root_value
        else None
    )
    require_subject_mask = training_split and bool(
        config["data"].get("subject_mask_required", False)
    )
    rows = [
        adapt_row(
            row,
            data_root,
            required,
            instruction_data=instruction_data,
            subject_mask_root=subject_mask_root,
            require_subject_mask=require_subject_mask,
        )
        for row in iter_jsonl(source)
    ]
    summary = validate_rows(rows, config, decode=decode, split=split)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {**summary, "source_manifest": str(source), "prepared_manifest": str(destination)}
