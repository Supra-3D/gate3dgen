"""Condition manifests for the official-aligned v1 evaluation.

The legacy manifests in this project were intentionally small (``degraded``,
``gt`` and one ``restored`` image).  This module adds an explicit condition
contract without changing that format.  A condition manifest is still JSONL,
but every row has a stable list of image conditions and restoration steps.

The manifest is deliberately only a *description* of files.  It never
silently treats a missing image as a zero score.  Callers can either validate
paths while building the manifest or keep missing conditions with an explicit
``available=false`` field for a resumable run.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import iter_jsonl, write_jsonl


OFFICIAL_ALIGNED_SCHEMA = "official_aligned_v1"
RESTORATION_STEPS: tuple[int, ...] = (1, 3, 5, 10, 20, 30, 40)
RESTORED_VARIANTS = tuple(f"restored_s{step:02d}" for step in RESTORATION_STEPS)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read either JSONL or a JSON array/object manifest.

    Existing benchmark manifests have appeared in both forms over time.  The
    object form is accepted when it has a ``cases``/``rows`` list, which makes
    this helper useful for hand-authored demo manifests as well.
    """

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    # First try the whole document.  This handles pretty-printed JSON arrays
    # and objects; a JSONL file naturally falls through on the decode error.
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows = list(iter_jsonl(path))
    else:
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            rows = value.get("cases", value.get("rows", value))
            if isinstance(rows, dict):
                rows = [rows]
        else:
            raise ValueError(f"manifest must contain objects or a list: {path}")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest rows must be objects: {path}")
    return rows


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _absolute(value: Any, root: Path | None) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute() and root is not None:
        path = root / path
    return str(path.absolute())


def _template_path(
    root: Path | None,
    template: str,
    row: Mapping[str, Any],
    step: int,
) -> str | None:
    if root is None:
        return None
    # Both {step} and {step:02d} are useful in shell templates.  Keep the
    # formatting operation explicit so malformed templates fail early.
    values = dict(row)
    values.update(
        case_id=str(_first(row, "case_id", "pair_id") or ""),
        step=step,
        step_name=f"s{step:02d}",
    )
    try:
        relative = template.format(**values)
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(f"invalid restored template {template!r}: {error}") from error
    return _absolute(relative, root)


def _step_path(row: Mapping[str, Any], step: int, root: Path | None, template: str) -> str | None:
    name = f"restored_s{step:02d}"
    # Accept the canonical key, common short spellings, and a nested mapping.
    value = _first(row, name, f"restored_{step}", f"step_{step}")
    restored = row.get("restored")
    if value is None and isinstance(restored, Mapping):
        value = _first(restored, name, str(step), f"s{step:02d}")
    if value is not None:
        return _absolute(value, root)
    return _template_path(root, template, row, step)


def _dataset_kind(row: Mapping[str, Any], explicit: str | None) -> str:
    if explicit and explicit != "auto":
        return explicit
    value = str(row.get("dataset_kind", ""))
    if value:
        if "wild" in value.lower():
            return "wild_no_gt"
        if "benchmark" in value.lower():
            return "benchmark_with_glb_gt"
    has_mesh = _first(row, "gt_mesh", "glb_gt") not in (None, "")
    has_original = _first(row, "original", "original_path") not in (None, "")
    return "wild_no_gt" if has_original and not has_mesh else "benchmark_with_glb_gt"


def _condition(
    condition_id: str,
    role: str,
    path: str | None,
    *,
    step: int | None = None,
    reference: bool = False,
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "role": role,
        "path": path,
        "available": bool(path and Path(path).is_file()),
        "restoration_step": step,
        "is_reference": reference,
    }


def build_condition_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    restored_root: Path | None = None,
    restored_template: str = "{case_id}/restored_s{step:02d}.png",
    steps: Sequence[int] = RESTORATION_STEPS,
    dataset_kind: str | None = None,
    path_root: Path | None = None,
    validate: bool = False,
) -> list[dict[str, Any]]:
    """Expand legacy rows into the official-aligned condition schema.

    ``clean_gt`` is the canonical name for the clean image reference.  The
    legacy ``gt`` field remains as an alias.  For wild rows, the original
    photograph is retained as ``original`` and there is intentionally no
    image or GLB ground truth.
    """

    selected_steps = tuple(sorted({int(step) for step in steps}))
    if not selected_steps or any(step <= 0 for step in selected_steps):
        raise ValueError("steps must contain positive integers")
    output: list[dict[str, Any]] = []
    for source in rows:
        case_id = str(_first(source, "case_id", "pair_id") or "")
        if not case_id:
            raise ValueError("each condition row needs case_id or pair_id")
        kind = _dataset_kind(source, dataset_kind)
        original = _absolute(_first(source, "original", "original_path"), path_root)
        degraded = _absolute(
            _first(source, "degraded", "degraded_path", "degraded_relative_path"),
            path_root,
        )
        if kind == "wild_no_gt" and original is None:
            original = degraded
        if degraded is None and kind == "wild_no_gt":
            degraded = original
        clean_gt = _absolute(
            _first(source, "clean_gt", "gt", "gt_path", "gt_relative_path"),
            path_root,
        )
        if kind == "wild_no_gt":
            clean_gt = None
        gt_mesh = _absolute(_first(source, "gt_mesh", "glb_gt", "gt_mesh_path"), path_root)

        restored: dict[str, str | None] = {
            f"restored_s{step:02d}": _step_path(source, step, restored_root, restored_template)
            for step in selected_steps
        }
        # Keep the newest requested step available under ``restored`` for
        # older tools.  This alias is not used by the official dashboard.
        latest = restored.get(f"restored_s{selected_steps[-1]:02d}")
        conditions = [
            _condition("degraded", "degraded_input", degraded),
        ]
        if kind == "benchmark_with_glb_gt":
            conditions.append(_condition("clean_gt", "clean_image_gt", clean_gt, reference=True))
        else:
            conditions.append(_condition("original", "wild_original", original, reference=True))
        for step in selected_steps:
            name = f"restored_s{step:02d}"
            conditions.append(_condition(name, "restored", restored[name], step=step))

        image_paths: dict[str, str | None] = {
            "degraded": degraded,
            "clean_gt": clean_gt,
            "gt": clean_gt,
            "original": original,
            **restored,
            "restored": latest,
        }
        result: dict[str, Any] = {
            "schema_version": OFFICIAL_ALIGNED_SCHEMA,
            "case_id": case_id,
            "asset_id": str(_first(source, "asset_id", "asset", "sha256") or case_id),
            "task_id": str(_first(source, "task_id", "degradation_type", "family") or "unknown"),
            "degradation_type": str(_first(source, "degradation_type", "task_id", "family") or "unknown"),
            "dataset_kind": kind,
            "asset_index": source.get("asset_index"),
            "view_id": str(_first(source, "view_id", "view_index") or ""),
            "degraded": degraded,
            "clean_gt": clean_gt,
            "gt": clean_gt,
            "original": original,
            "gt_mesh": gt_mesh if kind == "benchmark_with_glb_gt" else None,
            "restored": latest,
            **restored,
            "image_paths": image_paths,
            "conditions": conditions,
            "restoration_steps": list(selected_steps),
            "has_image_gt": bool(clean_gt),
            "has_glb_gt": bool(gt_mesh and Path(gt_mesh).is_file()),
            "reference_policy": "clean_gt_and_glb_gt" if kind == "benchmark_with_glb_gt" else "original_no_gt",
            "source": dict(source),
        }
        if validate:
            missing = [
                f"{item['condition_id']}={item['path']}"
                for item in conditions
                if item["path"] and not Path(item["path"]).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"{case_id}: missing condition files: {', '.join(missing)}")
            if kind == "benchmark_with_glb_gt" and not clean_gt:
                raise FileNotFoundError(f"{case_id}: clean_gt is required for benchmark rows")
        output.append(result)
    if not output:
        raise ValueError("condition manifest selection is empty")
    return output


def load_condition_manifest(path: Path) -> list[dict[str, Any]]:
    """Read and minimally validate an official-aligned JSONL manifest."""

    rows = _read_rows(path)
    for index, row in enumerate(rows, 1):
        if row.get("schema_version") != OFFICIAL_ALIGNED_SCHEMA:
            raise ValueError(f"{path}:{index}: expected schema_version={OFFICIAL_ALIGNED_SCHEMA}")
        if not row.get("case_id") or not isinstance(row.get("conditions"), list):
            raise ValueError(f"{path}:{index}: missing case_id/conditions")
    return rows


def _parse_steps(values: Sequence[str] | None) -> tuple[int, ...]:
    if not values:
        return RESTORATION_STEPS
    result: list[int] = []
    for value in values:
        for token in re.split(r"[, ]+", value.strip()):
            if token:
                result.append(int(token))
    return tuple(sorted(set(result)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-root", type=Path)
    parser.add_argument("--restored-root", type=Path)
    parser.add_argument(
        "--restored-template",
        default="{case_id}/restored_s{step:02d}.png",
        help="Template below --restored-root; supports {case_id}, {step}, {step_name}",
    )
    parser.add_argument("--steps", action="append", help="Comma/space separated steps (default 1,3,5,10,20,30,40)")
    parser.add_argument("--dataset-kind", choices=("auto", "benchmark_with_glb_gt", "wild_no_gt"), default="auto")
    parser.add_argument("--validate", action="store_true", help="Fail if declared condition paths are missing")
    args = parser.parse_args()
    rows = build_condition_rows(
        _read_rows(args.input_manifest),
        restored_root=args.restored_root,
        restored_template=args.restored_template,
        steps=_parse_steps(args.steps),
        dataset_kind=args.dataset_kind,
        path_root=args.path_root,
        validate=args.validate,
    )
    write_jsonl(args.output, rows)
    print(json.dumps({
        "schema_version": OFFICIAL_ALIGNED_SCHEMA,
        "cases": len(rows),
        "steps": sorted({step for row in rows for step in row["restoration_steps"]}),
        "benchmark": sum(row["dataset_kind"] == "benchmark_with_glb_gt" for row in rows),
        "wild": sum(row["dataset_kind"] == "wild_no_gt" for row in rows),
        "output": str(args.output.absolute()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
