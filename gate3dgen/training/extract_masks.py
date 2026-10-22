#!/usr/bin/env python3
"""Extract resumable RMBG-2.0 soft masks for unique clean GT images."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from gate3dgen.training.subject_masks import (
    MASK_SCHEMA,
    mask_path,
    sha256_file,
    validate_mask,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"manifest row {line_number} is not an object")
            yield value


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_mask(alpha: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        Image.fromarray(np.asarray(alpha, dtype=np.uint8), mode="L").save(temporary, format="PNG")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolve_under(root: Path, relative_value: Any, *, field: str) -> Path:
    from gate3dgen.training.subject_masks import safe_relative_path

    # Match the training adapter's lexical containment rule.  The quality-v11
    # dataset intentionally contains a ``pairs`` symlink to the immutable v1
    # image tree, so resolving symlinks here would reject valid training GTs.
    root_text = os.path.abspath(os.fspath(root.expanduser()))
    candidate_text = os.path.abspath(
        os.fspath(Path(root_text) / safe_relative_path(relative_value, field=field))
    )
    if os.path.commonpath((root_text, candidate_text)) != root_text:
        raise ValueError(f"{field} escapes data root: {candidate_text}")
    return Path(candidate_text)


def unique_items(manifest: Path, data_root: Path, mask_root: Path) -> list[dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(manifest):
        relative = row.get("gt_relative_path")
        key = str(relative).replace("\\", "/")
        gt = resolve_under(data_root, relative, field="gt_relative_path")
        width, height = (int(row["dimensions"][0]), int(row["dimensions"][1]))
        item = {
            "key": key,
            "gt": str(gt),
            "mask": str(mask_path(mask_root, relative)),
            "width": width,
            "height": height,
        }
        previous = items.setdefault(key, item)
        if previous != item:
            raise ValueError(f"inconsistent duplicate GT entry: {key}")
    return [items[key] for key in sorted(items)]


def worker(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    from gate3dgen.evaluation.mask_consistency import build_segmenter

    all_items = unique_items(args.manifest, args.data_root, args.output_root)
    items = [item for index, item in enumerate(all_items) if index % args.num_shards == args.shard_index]
    report_dir = args.output_root / ".rmbg2_reports"
    report_path = report_dir / f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}.json"
    segment = build_segmenter(str(args.model), device=args.device)
    infer_many = getattr(segment, "many", None)
    if infer_many is None:
        raise RuntimeError("RMBG2 segmenter does not expose batch inference")

    processed = cached = 0
    anomalies: list[dict[str, Any]] = []
    started = time.time()
    for start in range(0, len(items), args.batch_size):
        chunk = items[start : start + args.batch_size]
        pending: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        for item in chunk:
            output = Path(item["mask"])
            expected = (int(item["width"]), int(item["height"]))
            try:
                validate_mask(output, expected)
                cached += 1
                continue
            except (FileNotFoundError, OSError, ValueError):
                pass
            gt = Path(item["gt"])
            if not gt.is_file():
                raise FileNotFoundError(gt)
            with Image.open(gt) as opened:
                image = opened.convert("RGB")
                if image.size != expected:
                    raise ValueError(f"GT size {image.size} != manifest size {expected}: {gt}")
                images.append(image.copy())
            pending.append(item)
        if pending:
            alphas = infer_many(images, batch_size=args.batch_size)
            if len(alphas) != len(pending):
                raise RuntimeError("RMBG2 output count differs from input count")
            for item, alpha in zip(pending, alphas):
                output = Path(item["mask"])
                expected = (int(item["width"]), int(item["height"]))
                array = np.asarray(alpha, dtype=np.uint8)
                if array.shape != (expected[1], expected[0]):
                    raise ValueError(f"RMBG2 alpha shape {array.shape} != {(expected[1], expected[0])}")
                atomic_mask(array, output)
                validate_mask(output, expected)
                fraction = float((array >= 204).mean())
                if fraction <= args.min_foreground_fraction or fraction >= args.max_foreground_fraction:
                    anomalies.append({"key": item["key"], "foreground_fraction_at_0.8": fraction})
                processed += 1
        completed = start + len(chunk)
        if completed == len(items) or completed % max(args.batch_size, args.log_every) == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"RMBG2 shard={args.shard_index}/{args.num_shards} "
                f"items={completed}/{len(items)} processed={processed} cached={cached} "
                f"rate={completed / elapsed:.2f}/s",
                flush=True,
            )

    payload = {
        "schema": MASK_SCHEMA,
        "status": "complete",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "unique_gt_total": len(all_items),
        "assigned": len(items),
        "processed": processed,
        "cached": cached,
        "anomalies": anomalies,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "model": str(args.model.resolve()),
        "output_root": str(args.output_root.resolve()),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(payload, report_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def merge(args: argparse.Namespace) -> None:
    items = unique_items(args.manifest, args.data_root, args.output_root)
    report_dir = args.output_root / ".rmbg2_reports"
    reports = []
    for index in range(args.num_shards):
        path = report_dir / f"shard-{index:02d}-of-{args.num_shards:02d}.json"
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        expected = {
            "schema": MASK_SCHEMA,
            "status": "complete",
            "shard_index": index,
            "num_shards": args.num_shards,
            "unique_gt_total": len(items),
            "manifest_sha256": sha256_file(args.manifest),
            "model": str(args.model.resolve()),
        }
        mismatches = {key: (report.get(key), value) for key, value in expected.items() if report.get(key) != value}
        if mismatches:
            raise RuntimeError(f"stale/incompatible shard report {path}: {mismatches}")
        reports.append(report)
    missing = []
    for item in items:
        try:
            validate_mask(
                Path(item["mask"]),
                (int(item["width"]), int(item["height"])),
            )
        except (FileNotFoundError, OSError, ValueError) as error:
            missing.append({"key": item["key"], "error": str(error)})
            if len(missing) >= 20:
                break
    if missing:
        raise RuntimeError(f"subject mask validation failed: {missing}")
    complete = {
        "schema": MASK_SCHEMA,
        "status": "complete",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "model": str(args.model.resolve()),
        "output_root": str(args.output_root.resolve()),
        "num_shards": args.num_shards,
        "unique_gt_total": len(items),
        "processed": sum(int(report["processed"]) for report in reports),
        "cached": sum(int(report["cached"]) for report in reports),
        "anomalies": sum(len(report.get("anomalies", [])) for report in reports),
    }
    atomic_json(complete, args.output_root / "rmbg2_masks_complete.json")
    print(json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--min-foreground-fraction", type=float, default=0.0001)
    parser.add_argument("--max-foreground-fraction", type=float, default=0.9999)
    parser.add_argument("--merge-shards", action="store_true")
    args = parser.parse_args()
    for name in ("batch_size", "num_shards", "log_every"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    parsed = parse_args()
    merge(parsed) if parsed.merge_shards else worker(parsed)


if __name__ == "__main__":
    main()
