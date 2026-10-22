from __future__ import annotations

import argparse
import json
import os
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any

from .io import write_jsonl


KNOWN_WORKSPACES = tuple(
    Path(value).expanduser()
    for value in os.environ.get("GATE3DGEN_LEGACY_ROOTS", "").split(os.pathsep)
    if value.strip()
)


def rebase_path(value: str, workspace: Path) -> Path:
    """Map explicitly configured legacy roots onto this checkout."""
    path = Path(value)
    for root in KNOWN_WORKSPACES:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        return workspace.resolve() / relative
    return path


def load_asset_map(anchor_manifest: Path, glb_root: Path) -> dict[int, dict[str, str]]:
    anchors = json.loads(anchor_manifest.read_text(encoding="utf-8"))
    mapping: dict[int, dict[str, str]] = {}
    for anchor in anchors:
        index = int(anchor["asset_index"])
        asset = str(anchor["asset"])
        existing = mapping.get(index)
        if existing and existing["asset"] != asset:
            raise ValueError(f"asset_index {index} maps to multiple assets")
        mapping[index] = {"asset": asset, "gt_mesh": str((glb_root / f"{asset}.glb").resolve())}
    return mapping


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    """Read the current JSON-array manifest as well as legacy JSONL files."""
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = document if isinstance(document, list) else document.get("rows", document.get("cases", [document]))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest must contain JSON objects: {path}")
    return [dict(row) for row in rows]


def glb_status(path: Path) -> dict[str, Any]:
    size = path.stat().st_size if path.is_file() else 0
    valid = False
    version = None
    declared_length = None
    if size >= 12:
        with path.open("rb") as handle:
            magic, version, declared_length = struct.unpack("<4sII", handle.read(12))
        valid = magic == b"glTF" and version == 2 and declared_length == size
    return {
        "exists": path.is_file(), "bytes": size, "glb_v2_valid": valid,
        "version": version, "declared_length": declared_length,
    }


def mapping_report(
    source_rows: list[dict[str, Any]], assets: dict[int, dict[str, str]],
    benchmark_manifest: Path, anchor_manifest: Path,
) -> dict[str, Any]:
    counts = Counter(int(row["asset_index"]) for row in source_rows)
    rows = []
    for index, mapped in sorted(assets.items()):
        path = Path(mapped["gt_mesh"])
        rows.append({
            "asset_index": index, "asset": mapped["asset"],
            "gt_mesh": mapped["gt_mesh"], "benchmark_cases": counts[index],
            **glb_status(path),
        })
    return {
        "benchmark_manifest": str(benchmark_manifest.resolve()),
        "anchor_manifest": str(anchor_manifest.resolve()),
        "mapping_method": "asset_index -> rendered anchor asset catalog -> DTC_multiview/<asset>.glb",
        "assets": len(rows),
        "mapped": sum(row["exists"] for row in rows),
        "valid_glb_v2": sum(row["glb_v2_valid"] for row in rows),
        "rows": rows,
    }


def select_demo_rows(rows: list[dict[str, Any]], max_cases: int | None) -> list[dict[str, Any]]:
    """Select one 3D-core case per degradation family, deterministically."""
    core = [row for row in rows if row.get("is_3d_core")]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Round-robin single/mixed and mild/severe so even a short prefix is diverse.
    keys = (("single", "mild"), ("mixed", "severe"), ("single", "severe"), ("mixed", "mild"))
    buckets = {
        key: [row for row in core if (row.get("kind"), row.get("severity")) == key]
        for key in keys
    }
    positions = {key: 0 for key in keys}
    while any(positions[key] < len(buckets[key]) for key in keys):
        progressed = False
        for key in keys:
            bucket = buckets[key]
            while positions[key] < len(bucket):
                row = bucket[positions[key]]
                positions[key] += 1
                family = str(row["family"])
                if family in seen:
                    continue
                selected.append(row)
                seen.add(family)
                progressed = True
                if max_cases is not None and len(selected) >= max_cases:
                    return selected
                break
        if not progressed:
            break
    for row in core:
        family = str(row["family"])
        if family not in seen:
            selected.append(row)
            seen.add(family)
            if max_cases is not None and len(selected) >= max_cases:
                break
    return selected


def build_benchmark_rows(
    benchmark_manifest: Path,
    anchor_manifest: Path,
    glb_root: Path,
    workspace: Path,
    restored_root: Path | None,
    max_cases: int | None,
    all_core: bool,
    selected_case_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    source_rows = _load_manifest_rows(benchmark_manifest)
    if selected_case_ids is not None:
        by_case = {str(row["case_id"]): row for row in source_rows}
        missing = [case_id for case_id in selected_case_ids if case_id not in by_case]
        if missing:
            raise KeyError(f"selected cases absent from benchmark manifest: {missing}")
        chosen = [by_case[case_id] for case_id in selected_case_ids]
    else:
        chosen = [row for row in source_rows if row.get("is_3d_core")] if all_core else select_demo_rows(source_rows, max_cases)
    if max_cases is not None and all_core:
        chosen = chosen[:max_cases]
    assets = load_asset_map(anchor_manifest, glb_root)
    output = []
    for source in chosen:
        index = int(source["asset_index"])
        if index not in assets:
            raise KeyError(f"no asset mapping for asset_index={index}")
        mapped = assets[index]
        row = {
            "case_id": str(source["case_id"]),
            "asset_id": mapped["asset"],
            "asset_index": index,
            "task_id": str(source["family"]),
            "degradation_type": str(source["family"]),
            "degradation_kind": str(source["kind"]),
            "severity": str(source["severity"]),
            "view_id": str(source["view_index"]),
            "dataset_kind": "benchmark_with_glb_gt",
            "degraded": str(rebase_path(source["degraded_path"], workspace)),
            "gt": str(rebase_path(source["gt_path"], workspace)),
            "object_mask": str(rebase_path(source["object_mask_path"], workspace)),
            "gt_mesh": mapped["gt_mesh"],
            "restored": str((restored_root / source["case_id"] / "restored.png").resolve()) if restored_root else None,
            "mapping": {
                "method": "asset_index_to_anchor_asset_catalog_to_DTC_multiview_glb",
                "anchor_manifest": str(anchor_manifest.resolve()),
                "benchmark_manifest": str(benchmark_manifest.resolve()),
            },
        }
        for key in ("degraded", "gt", "object_mask", "gt_mesh"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"{row['case_id']}:{key}: {row[key]}")
        output.append(row)
    if not output:
        raise ValueError("benchmark selection is empty")
    return output


def case_ids_from_brief_report(path: Path) -> list[str]:
    """Read reviewed case order from a legacy brief or degradation showcase."""
    text = path.read_text(encoding="utf-8")
    case_ids = re.findall(r'<div\s+class=["\']case-meta["\']>([^<]+)</div>', text)
    case_ids = [value.strip() for value in case_ids if value.strip()]
    if not case_ids:
        case_ids = re.findall(
            r'<span>[^<]*?\s[·]\s*([^<\s]+)</span>', text, flags=re.IGNORECASE
        )
        case_ids = [value.strip() for value in case_ids if value.strip()]
    if not case_ids:
        raise ValueError(f"no case entries found in selection report: {path}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"duplicate case entries in selection report: {path}")
    return case_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DTC benchmark demo manifest with explicit real-GLB GT mapping")
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--restored-root", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--all-core", action="store_true")
    parser.add_argument(
        "--selection-report", type=Path,
        help="Select cases in the exact order of <div class='case-meta'> entries in this brief HTML report",
    )
    parser.add_argument(
        "--demo-manifest", type=Path,
        help="Authoritative demo_manifest.json; supersedes the legacy --selection-report",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, help="Defaults to asset_glb_mapping.json beside --output")
    args = parser.parse_args()
    if args.selection_report and args.demo_manifest:
        raise ValueError("pass only one of --demo-manifest or legacy --selection-report")
    if args.demo_manifest:
        demo_rows = _load_manifest_rows(args.demo_manifest)
        selected_case_ids = [str(row.get("case_id") or row.get("pair_id") or "") for row in demo_rows]
        if any(not value for value in selected_case_ids) or len(selected_case_ids) != len(set(selected_case_ids)):
            raise ValueError("demo manifest has empty or duplicate case_id values")
    else:
        selected_case_ids = case_ids_from_brief_report(args.selection_report) if args.selection_report else None
    rows = build_benchmark_rows(
        args.benchmark_manifest, args.anchor_manifest, args.glb_root,
        args.workspace, args.restored_root, args.max_cases, args.all_core, selected_case_ids,
    )
    write_jsonl(args.output, rows)
    source_rows = _load_manifest_rows(args.benchmark_manifest)
    assets = load_asset_map(args.anchor_manifest, args.glb_root)
    mapping_path = args.mapping_output or (args.output.parent / "asset_glb_mapping.json")
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(mapping_report(source_rows, assets, args.benchmark_manifest, args.anchor_manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "cases": len(rows),
        "assets": len({row["asset_id"] for row in rows}),
        "families": dict(sorted(Counter(row["task_id"] for row in rows).items())),
        "true_glb_gt": sum(Path(row["gt_mesh"]).is_file() for row in rows),
        "output": str(args.output.resolve()),
        "mapping_output": str(mapping_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
