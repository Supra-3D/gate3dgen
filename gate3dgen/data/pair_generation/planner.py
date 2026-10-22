from __future__ import annotations

import collections
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image

from . import SCHEMA_VERSION
from .common import (
    atomic_write_json, atomic_write_jsonl, extract_relative_source, file_sha256,
    iter_jsonl, stable_digest, stable_int,
)

EXPECTED_TASKS = 12
PILOT_ASSETS = 42


def load_single_tasks(library_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    library = json.loads(library_path.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for degradation_type, group in library["degradation_prompts"].items():
        variants = [item for item in group.get("variants", []) if item.get("severity") != "light"]
        if len(variants) != 1:
            raise ValueError(
                f"Expected one enabled variant for {degradation_type}, found {len(variants)}"
            )
        item = variants[0]
        tasks.append({
            "degradation_type": degradation_type,
            "task_id": item["id"],
            "severity": item.get("severity"),
            "method": item.get("method", group.get("preferred_method", "generative")),
        })
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"Expected {EXPECTED_TASKS} single degradations, found {len(tasks)}")
    if any(task["severity"] != "heavy" for task in tasks):
        raise ValueError("The v1 plan expects all 12 single degradations to be heavy")
    return tasks, library, file_sha256(library_path)


def _verify_image(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def load_assets(input_path: Path, dataset_root: Path, seed: int, *, verify_images: bool) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in iter_jsonl(input_path):
        asset_id = raw["sha256"]
        if asset_id in seen:
            raise ValueError(f"Duplicate asset: {asset_id}")
        seen.add(asset_id)
        category = raw["semantic"]["primary_category"]
        views = sorted(
            raw["vlm"]["view_selection"]["selected_views"],
            key=lambda item: (int(item.get("rank", 999)), item["view_id"]),
        )
        if len(views) != 4:
            raise ValueError(f"Asset {asset_id} has {len(views)} selected views, expected 4")
        compact_views = []
        for slot, view in enumerate(views):
            original = Path(view["path"])
            relative = extract_relative_source(original)
            resolved = dataset_root / relative
            if not resolved.is_file() and original.is_file():
                resolved = original
            dimensions = _verify_image(resolved) if verify_images else None
            compact_views.append({
                "view_slot": slot,
                "view_id": view["view_id"],
                "rank": view.get("rank"),
                "role": view.get("role", []),
                "score": view.get("score"),
                "source_original_path": str(original),
                "source_relative_path": relative,
                "dimensions": list(dimensions) if dimensions else None,
            })
        assets.append({
            "sha256": asset_id,
            "caption": raw.get("caption", ""),
            "primary_coarse": category["coarse"],
            "primary_fine": category["fine"],
            "views": compact_views,
        })
    if not assets:
        raise ValueError("The asset manifest must not be empty")
    return sorted(assets, key=lambda asset: stable_digest(seed, "asset-order", asset["sha256"]))


def _largest_remainder(groups: dict[tuple[str, str], list[dict[str, Any]]], total: int) -> dict[tuple[str, str], int]:
    population = sum(len(items) for items in groups.values())
    quotas = {key: math.floor(len(items) * total / population) for key, items in groups.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        groups,
        key=lambda key: (
            -(len(groups[key]) * total / population - quotas[key]),
            key,
        ),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def assign_splits(assets: list[dict[str, Any]], seed: int) -> None:
    holdout = max(1, int(len(assets) * 0.05)) if len(assets) >= 3 else 0
    split_targets = {"train": len(assets) - 2 * holdout, "validation": holdout, "test": holdout}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for asset in assets:
        groups[(asset["primary_coarse"], asset["primary_fine"])].append(asset)
    holdout_total = split_targets["validation"] + split_targets["test"]
    holdout_quotas = _largest_remainder(groups, holdout_total)
    holdout_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, items in groups.items():
        ordered = sorted(items, key=lambda asset: stable_digest(seed, "holdout", asset["sha256"]))
        holdout_by_group[key] = ordered[:holdout_quotas[key]]
        for asset in ordered[holdout_quotas[key]:]:
            asset["split"] = "train"
    validation_quotas = {key: len(items) // 2 for key, items in holdout_by_group.items()}
    need = split_targets["validation"] - sum(validation_quotas.values())
    odd_groups = [key for key, items in holdout_by_group.items() if len(items) % 2]
    odd_groups.sort(key=lambda key: stable_digest(seed, "validation-group", *key))
    for key in odd_groups[:need]:
        validation_quotas[key] += 1
    for key, items in holdout_by_group.items():
        ordered = sorted(items, key=lambda asset: stable_digest(seed, "validation", asset["sha256"]))
        for asset in ordered[:validation_quotas[key]]:
            asset["split"] = "validation"
        for asset in ordered[validation_quotas[key]:]:
            asset["split"] = "test"
    counts = collections.Counter(asset["split"] for asset in assets)
    if any(counts[key] != value for key, value in split_targets.items()):
        raise AssertionError(f"Split mismatch: {dict(counts)} != {split_targets}")


def select_pilot_assets(assets: list[dict[str, Any]], seed: int) -> set[str]:
    train = [asset for asset in assets if asset["split"] == "train"]
    target = min(PILOT_ASSETS, len(train))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for asset in train:
        groups[(asset["primary_coarse"], asset["primary_fine"])].append(asset)
    for key in groups:
        groups[key].sort(key=lambda asset: stable_digest(seed, "pilot-candidate", asset["sha256"]))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_groups: set[tuple[str, str]] = set()
    coarse_names = sorted({key[0] for key in groups})
    for coarse in coarse_names:
        if len(selected) >= target:
            break
        fine_keys = [key for key in groups if key[0] == coarse]
        fine_keys.sort(key=lambda key: stable_digest(seed, "pilot-coarse", *key))
        candidate = groups[fine_keys[0]][0]
        selected.append(candidate)
        selected_ids.add(candidate["sha256"])
        selected_groups.add(fine_keys[0])
    remaining_groups = sorted(
        (key for key in groups if key not in selected_groups),
        key=lambda key: stable_digest(seed, "pilot-fine", *key),
    )
    for key in remaining_groups:
        if len(selected) >= target:
            break
        candidate = next((item for item in groups[key] if item["sha256"] not in selected_ids), None)
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(candidate["sha256"])
    for candidate in sorted(train, key=lambda item: stable_digest(seed, "pilot-fill", item["sha256"])):
        if len(selected) >= target:
            break
        if candidate["sha256"] not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate["sha256"])
    if len(selected) != target:
        raise AssertionError(f"Pilot selection produced {len(selected)} assets")
    return selected_ids


def build_view_assignments(asset_count: int, task_count: int, seed: int, task_version: str) -> list[list[int]]:
    assignments = [[(task_index + asset_index) % 4 for task_index in range(task_count)] for asset_index in range(asset_count)]
    rng = random.Random(stable_int(seed, task_version, "balanced-switches"))
    for _ in range(asset_count * 64 if asset_count >= 2 else 0):
        first, second = rng.sample(range(asset_count), 2)
        task_a, task_b = rng.sample(range(task_count), 2)
        slot_a, slot_b = assignments[first][task_a], assignments[first][task_b]
        if slot_a == slot_b:
            continue
        if assignments[second][task_a] == slot_b and assignments[second][task_b] == slot_a:
            assignments[first][task_a], assignments[first][task_b] = slot_b, slot_a
            assignments[second][task_a], assignments[second][task_b] = slot_a, slot_b
    return assignments


def validate_plan(rows: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> dict[str, Any]:
    asset_count = len({row["sha256"] for row in rows})
    expected = asset_count * EXPECTED_TASKS
    if len(rows) != expected:
        raise AssertionError(f"Plan has {len(rows)} rows, expected {expected}")
    if len({row["pair_id"] for row in rows}) != expected:
        raise AssertionError("pair_id values are not unique")
    by_asset: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_task_slot: dict[str, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    for row in rows:
        by_asset[row["sha256"]].append(row)
        by_task_slot[row["task_id"]][row["view_slot"]] += 1
    expected_task_ids = {task["task_id"] for task in tasks}
    for asset_id, items in by_asset.items():
        if {item["task_id"] for item in items} != expected_task_ids:
            raise AssertionError(f"Asset {asset_id} does not cover all tasks")
        slots = collections.Counter(item["view_slot"] for item in items)
        if slots != collections.Counter({0: 3, 1: 3, 2: 3, 3: 3}):
            raise AssertionError(f"Asset {asset_id} view allocation is {slots}")
        if len({item["split"] for item in items}) != 1:
            raise AssertionError(f"Asset {asset_id} crosses splits")
    for task_id, counts in by_task_slot.items():
        if sorted(counts.get(slot, 0) for slot in range(4)) != sorted((asset_count + slot) // 4 for slot in range(4)):
            raise AssertionError(f"Task {task_id} view counts are {dict(counts)}")
    return {
        "assets": len(by_asset),
        "pairs": len(rows),
        "task_view_counts": {key: dict(value) for key, value in by_task_slot.items()},
        "split_pairs": dict(collections.Counter(row["split"] for row in rows)),
    }


def create_plan(
    input_path: Path, library_path: Path, dataset_root: Path, output_root: Path,
    *, seed: int = 42, verify_images: bool = True,
) -> dict[str, Any]:
    tasks, library, library_hash = load_single_tasks(library_path)
    task_version = str(library.get("version", "unknown"))
    assets = load_assets(input_path, dataset_root, seed, verify_images=verify_images)
    assign_splits(assets, seed)
    pilot_ids = select_pilot_assets(assets, seed)
    assignments = build_view_assignments(len(assets), len(tasks), seed, task_version)
    rows: list[dict[str, Any]] = []
    plan_index = 0
    for asset_index, asset in enumerate(assets):
        for task_index, task in enumerate(tasks):
            slot = assignments[asset_index][task_index]
            view = asset["views"][slot]
            pair_material = f"{seed}:{asset['sha256']}:{task['task_id']}:{view['view_id']}:{task_version}"
            pair_id = stable_digest("pair", pair_material).hex()[:32]
            pair_seed = stable_int(seed, pair_material)
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "plan_index": plan_index,
                "pair_id": pair_id,
                "global_seed": seed,
                "pair_seed": pair_seed,
                "task_version": task_version,
                "prompt_library_sha256": library_hash,
                "sha256": asset["sha256"],
                "primary_coarse": asset["primary_coarse"],
                "primary_fine": asset["primary_fine"],
                "split": asset["split"],
                "pilot": asset["sha256"] in pilot_ids,
                **task,
                **view,
            })
            plan_index += 1
    audit = validate_plan(rows, tasks)
    pilot_rows = [row for row in rows if row["pilot"]]
    if len(pilot_rows) != len(pilot_ids) * EXPECTED_TASKS:
        raise AssertionError(f"Pilot has {len(pilot_rows)} pairs")
    if {row["split"] for row in pilot_rows} != {"train"}:
        raise AssertionError("Pilot contains non-train assets")
    compact_assets = [{
        "sha256": asset["sha256"],
        "caption": asset["caption"],
        "primary_coarse": asset["primary_coarse"],
        "primary_fine": asset["primary_fine"],
        "split": asset["split"],
        "pilot": asset["sha256"] in pilot_ids,
        "views": asset["views"],
    } for asset in assets]
    plans_dir = output_root / "plans"
    inventory_dir = output_root / "inventory"
    atomic_write_jsonl(inventory_dir / "assets.jsonl", compact_assets)
    atomic_write_jsonl(plans_dir / "master.jsonl", rows)
    atomic_write_jsonl(plans_dir / "pilot.jsonl", pilot_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "input_manifest": str(input_path.resolve()),
        "prompt_library": str(library_path.resolve()),
        "prompt_library_sha256": library_hash,
        "dataset_root": str(dataset_root.absolute()),
        "output_root": str(output_root.resolve()),
        "global_seed": seed,
        "task_version": task_version,
        "tasks": tasks,
        "pilot_assets": len(pilot_ids),
        "pilot_pairs": len(pilot_rows),
        **audit,
    }
    atomic_write_json(plans_dir / "plan_summary.json", summary)
    return summary
