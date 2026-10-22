"""Publish aggregate counts and split fingerprints without redistributing private asset lists/images."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from gate3dgen.evaluation.io import iter_jsonl


def _statistics(rows):
    counts, tasks, decisions = Counter(), defaultdict(Counter), defaultdict(Counter)
    assets, images = defaultdict(set), defaultdict(set)
    identifiers = set()
    for row in rows:
        split = str(row.get("split", "unspecified"))
        identifier = str(row.get("sample_id") or row.get("pair_id") or row.get("case_id") or "")
        if not identifier or identifier in identifiers:
            raise ValueError("Require unique sample/pair/case identifiers")
        identifiers.add(identifier)
        asset = str(row.get("asset") or row.get("asset_id") or row.get("sha256") or "")
        if not asset:
            raise ValueError("Require stable asset identity")
        counts[split] += 1
        assets[split].add(asset)
        tasks[split][str(row.get("task_id") or row.get("task") or "unspecified")] += 1
        decisions[split][str(row.get("decision", "pair"))] += 1
        for field in ("image_sha256", "gt_sha256", "degraded_sha256", "source_image_sha256"):
            if row.get(field):
                images[split].add(str(row[field]))
    if not identifiers:
        raise ValueError("Empty manifest")
    splits = sorted(counts)
    checks = []
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            shared_assets, shared_images = len(assets[a] & assets[b]), len(images[a] & images[b])
            checks.append({"split_a": a, "split_b": b, "shared_assets": shared_assets,
                           "shared_declared_image_hashes": shared_images})
            if shared_assets or shared_images:
                raise ValueError(f"Asset or image overlap between {a} and {b}")
    fingerprint = lambda values: hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()
    return {"records": sum(counts.values()), "splits": {
        split: {"records": counts[split], "assets": len(assets[split]), "asset_set_sha256": fingerprint(assets[split]),
                "unique_image_hashes": len(images[split]), "image_hash_set_sha256": fingerprint(images[split]),
                "tasks": dict(tasks[split]), "decisions": dict(decisions[split])} for split in splits},
        "asset_ids_distributed": False,
        "image_hash_fields": ["image_sha256", "gt_sha256", "degraded_sha256", "source_image_sha256"],
        "image_hashes_recomputed": False,
        "cross_split_checks": checks,
        "cross_split_asset_isolation_verified": len(splits) > 1,
        "declared_hash_coverage": "only declared SHA fields; not proof of byte isolation when fields are missing",
        "fingerprint_encoding": "sorted identifiers joined by LF, UTF-8, SHA256"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_statistics(path):
    path = Path(path)
    result = _statistics(iter_jsonl(path))
    return {"manifest_sha256": _sha256(path), **result}


def split_statistics(manifests):
    """Audit separately stored populations under explicit split labels, without publishing paths/IDs."""
    if len(manifests) < 2 or any(not str(label).strip() for label in manifests):
        raise ValueError("Declare at least two named populations")
    def rows():
        for split, path in manifests.items():
            count = 0
            for row in iter_jsonl(Path(path)):
                count += 1
                yield {**row, "split": str(split)}
            if not count:
                raise ValueError(f"Empty declared population: {split}")
    result = _statistics(rows())
    result["source_manifests"] = [{"split": str(split), "manifest_sha256": _sha256(path)}
                                  for split, path in manifests.items()]
    result["split_label_source"] = "explicit CLI population label; overrides each source row's internal split"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--split-manifest", action="append", metavar="SPLIT=PATH",
                        help="Repeat for train/development/final to check complete cross-population isolation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.manifest:
        result = manifest_statistics(args.manifest)
    else:
        manifests = {}
        for item in args.split_manifest:
            label, separator, path = item.partition("=")
            if not separator or not label.strip() or not path or label in manifests:
                parser.error("Use distinct --split-manifest SPLIT=PATH assignments")
            manifests[label] = Path(path)
        result = split_statistics(manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
