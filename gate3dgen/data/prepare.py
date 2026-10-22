"""Join reviewed quality decisions to pairs and record explicit mask review."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .pair_generation.common import atomic_write_json, atomic_write_jsonl, iter_jsonl


def prepare_dataset(manifest: Path, audit: Path, output: Path) -> dict:
    pairs = list(iter_jsonl(manifest))
    reviews = list(iter_jsonl(audit))
    for name, rows in (("pairs", pairs), ("audit", reviews)):
        ids = [str(row["pair_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate pair_id in {name}")
    by_id = {str(row["pair_id"]): row for row in reviews}
    splits = {name: [] for name in ("train", "validation", "test")}
    assets = {}
    excluded = Counter()
    for pair in pairs:
        split, asset = pair["split"], str(pair["sha256"])
        if split not in splits:
            raise ValueError(f"Unknown split: {split}")
        if asset in assets and assets[asset] != split:
            raise ValueError(f"Asset {asset} crosses splits")
        assets[asset] = split
        decision = by_id.get(str(pair["pair_id"]), {}).get("quality_status", "missing")
        if decision != "pass":
            excluded[str(decision)] += 1
            continue
        splits[split].append({**pair, "quality_status": "pass"})
    if not splits["train"]:
        raise ValueError("No reviewed passing training pairs")
    for name, rows in splits.items():
        atomic_write_jsonl(output / f"{name}.jsonl", rows)
    summary = {"accepted": {k: len(v) for k, v in splits.items()}, "excluded": dict(excluded),
               "source_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest()}
    atomic_write_json(output / "preparation.json", summary)
    return summary


def record_mask_review(cache: Path, reviewed: Path, reviewer: str) -> dict:
    marker = cache / "rmbg2_masks_complete.json"
    info = json.loads(marker.read_text())
    if info.get("status") != "complete":
        raise ValueError("Mask extraction must finish before review is recorded")
    if not reviewer.strip() or not reviewed.is_file():
        raise ValueError("A reviewer and reviewed training manifest are required")
    record = {"status": "complete", "reviewer": reviewer,
              "base_complete_marker": marker.name,
              "reviewed_training_manifest": str(reviewed.resolve()),
              "reviewed_training_manifest_sha256": hashlib.sha256(reviewed.read_bytes()).hexdigest()}
    destination = cache / "rmbg2_manual_review.json"
    if destination.exists():
        raise FileExistsError(f"Review already recorded: {destination}")
    atomic_write_json(destination, record)
    return record


def build_router_validation(cases_path: Path, train_manifest: Path, final_cases: Path, output: Path) -> dict:
    from gate3dgen.training.router_tools.manifest import _common_row, _portable_image, REASON_ORDER
    cases = list(iter_jsonl(cases_path))
    train = list(iter_jsonl(train_manifest))
    final = list(iter_jsonl(final_cases))
    if not cases or not train or not final:
        raise ValueError("Training, validation and final evaluation populations must all be declared")
    validation_assets = {str(row["asset_id"]) for row in cases}
    if {str(row["asset"]) for row in train} & {str(row["asset_id"]) for row in final}:
        raise ValueError("Final evaluation assets overlap training")
    if validation_assets & {str(row["asset"]) for row in train}:
        raise ValueError("Router validation assets overlap training")
    if validation_assets & {str(row["asset_id"]) for row in final}:
        raise ValueError("Validation assets must be excluded from final evaluation")
    root = Path.cwd().resolve()
    cache, results, seen, image_labels = {}, [], set(), {}
    train_hashes = {row["image_sha256"] for row in train}
    for case in cases:
        case_id = str(case["case_id"])
        if case_id in seen:
            raise ValueError(f"Duplicate validation case: {case_id}")
        seen.add(case_id)
        reasons = case["reasons"]
        if not reasons or not isinstance(reasons, list) or any(x not in REASON_ORDER or x == "NONE" for x in reasons):
            raise ValueError("Declare canonical degradation reasons for every validation case")
        refs = {}
        for key in ("degraded", "gt"):
            path = Path(case[key])
            if not path.is_absolute():
                path = root / path
            refs[key] = _portable_image(path, root_name="workspace", root_path=root, workspace_root=root,
                                       expected_sha256=None, context=case_id, validate_images=True, cache=cache)
            if refs[key]["sha256"] in train_hashes:
                raise ValueError("Validation image bytes overlap training")
        for key, kind, decision in (("degraded", "degraded", "REPAIR"), ("gt", "gt_clean", "KEEP")):
            digest = refs[key]["sha256"]
            if digest in image_labels and image_labels[digest] != decision:
                raise ValueError("Conflicting labels for identical validation images")
            image_labels[digest] = decision
            results.append(_common_row(split="validation", pair_id=case_id, asset_id=str(case["asset_id"]),
                source_sha256=refs["gt"]["sha256"], degraded_sha256=refs["degraded"]["sha256"],
                gt_sha256=refs["gt"]["sha256"], task_id=case.get("task_id", "benchmark"),
                degradation_type=case.get("task_id", "benchmark"), severity_tag=case.get("severity"),
                image_kind=kind, image_ref=refs[key], decision=decision,
                reasons=reasons if key == "degraded" else ["NONE"], hand_remove_member=False,
                label_provenance="declared_independent_validation"))
    atomic_write_jsonl(output, results)
    audit = {"validation_cases": len(cases), "validation_images": len(results),
             "train_manifest_sha256": hashlib.sha256(train_manifest.read_bytes()).hexdigest(),
             "final_manifest_sha256": hashlib.sha256(final_cases.read_bytes()).hexdigest(),
             "asset_disjoint": True, "training_image_disjoint": True}
    atomic_write_json(output.with_suffix(".audit.json"), audit)
    return audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset", help="Publish only explicitly accepted pairs")
    dataset.add_argument("--manifest", type=Path, required=True)
    dataset.add_argument("--audit", type=Path, required=True)
    dataset.add_argument("--output", type=Path, default=Path("data/manifests"))
    review = commands.add_parser("mask-review", help="Record a completed human mask review, not perform it")
    review.add_argument("--cache", type=Path, default=Path("outputs/masks"))
    review.add_argument("--reviewed-manifest", type=Path, default=Path("data/manifests/train.jsonl"))
    review.add_argument("--reviewer", required=True)
    review.add_argument("--confirm-reviewed", action="store_true", required=True)
    validation = commands.add_parser("router-validation", help="Build independent validation and check held-out assets")
    validation.add_argument("--cases", type=Path, required=True)
    validation.add_argument("--train-manifest", type=Path, default=Path("data/router/train.jsonl"))
    validation.add_argument("--final-cases", type=Path, required=True)
    validation.add_argument("--output", type=Path, default=Path("data/router/demo_validation.jsonl"))
    args = parser.parse_args(argv)
    if args.command == "dataset":
        result = prepare_dataset(args.manifest, args.audit, args.output)
    elif args.command == "mask-review":
        result = record_mask_review(args.cache, args.reviewed_manifest, args.reviewer)
    else:
        result = build_router_validation(args.cases, args.train_manifest, args.final_cases, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
