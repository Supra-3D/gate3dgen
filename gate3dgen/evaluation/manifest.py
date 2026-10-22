from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_jsonl, resolve_under, write_jsonl


def _restored_path(restored_root: Path | None, template: str, row: dict[str, Any]) -> str | None:
    if restored_root is None:
        return None
    relative = template.format(**row)
    return str(resolve_under(restored_root, relative))


def build_rows(
    source_manifest: Path,
    data_root: Path,
    albedo_root: Path | None,
    restored_root: Path | None,
    restored_template: str,
    max_cases: int | None,
    one_per_asset: bool,
    one_per_task: bool,
    task_ids: set[str] | None,
    validate_paths: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    seen_tasks: set[str] = set()
    for source in iter_jsonl(source_manifest):
        task_id = str(source["task_id"])
        asset_id = str(source["sha256"])
        if task_ids and task_id not in task_ids:
            continue
        if one_per_asset and asset_id in seen_assets:
            continue
        if one_per_task and task_id in seen_tasks:
            continue
        row = {
            "case_id": str(source["pair_id"]),
            "asset_id": asset_id,
            "task_id": task_id,
            "degradation_type": str(source.get("degradation_type", task_id)),
            "view_id": str(source.get("view_id", "")),
            "degraded": str(resolve_under(data_root, source["degraded_relative_path"])),
            "gt": str(resolve_under(data_root, source["gt_relative_path"])),
            "albedo": str(resolve_under(albedo_root or data_root, source["source_relative_path"])),
            "restored": _restored_path(restored_root, restored_template, source),
        }
        if validate_paths:
            for key in ("degraded", "gt", "albedo"):
                if not Path(row[key]).is_file():
                    raise FileNotFoundError(row[key])
        output.append(row)
        seen_assets.add(asset_id)
        seen_tasks.add(task_id)
        if max_cases is not None and len(output) >= max_cases:
            break
    if not output:
        raise ValueError("case selection is empty")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable TexVerse evaluation manifest")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--albedo-root", type=Path, help="Root containing renders_crossview; defaults to data-root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restored-root", type=Path)
    parser.add_argument(
        "--restored-template",
        default="{pair_id}/restored.png",
        help="Python format string using source-manifest fields",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--one-per-asset", action="store_true")
    parser.add_argument("--one-per-task", action="store_true")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument(
        "--skip-path-check",
        action="store_true",
        help="Only resolve paths while building; generation/metrics/audit still validate inputs before use",
    )
    args = parser.parse_args()
    rows = build_rows(
        args.source_manifest,
        args.data_root,
        args.albedo_root,
        args.restored_root,
        args.restored_template,
        args.max_cases,
        args.one_per_asset,
        args.one_per_task,
        set(args.task_ids) if args.task_ids else None,
        validate_paths=not args.skip_path_check,
    )
    write_jsonl(args.output, rows)
    tasks = Counter(row["task_id"] for row in rows)
    print({"cases": len(rows), "assets": len({r["asset_id"] for r in rows}), "tasks": dict(tasks)})


if __name__ == "__main__":
    main()
