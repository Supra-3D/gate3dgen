"""Evaluate the router on labeled images at a frozen threshold; never fit on test data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from gate3dgen.router import Router, DEFAULT_THRESHOLD, decision_from_probability
from .io import iter_jsonl, write_jsonl
from .router_metrics import auroc, auprc, expected_calibration_error


def summarize(rows: list[dict], threshold: float) -> dict:
    if not rows:
        raise ValueError("No evaluation records")
    labels = [row["label"] for row in rows]
    scores = [row["purify_probability"] for row in rows]
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Labels must be binary: KEEP=0, PURIFY=1")
    predicted = [int(decision_from_probability(score, threshold) == "PURIFY") for score in scores]
    tp = sum(y == p == 1 for y, p in zip(labels, predicted))
    tn = sum(y == p == 0 for y, p in zip(labels, predicted))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predicted))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predicted))
    divide = lambda a, b: a / b if b else None
    accuracy, recall = (tp + tn) / len(rows), divide(tp, tp + fn)
    harmonic = None if recall is None else (2 * recall * accuracy / (recall + accuracy) if recall + accuracy else 0.0)
    return {"samples": len(rows), "threshold": threshold, "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "accuracy": (tp + tn) / len(rows), "precision": divide(tp, tp + fp),
            "recall": divide(tp, tp + fn), "specificity": divide(tn, tn + fp),
            "f1": divide(2 * tp, 2 * tp + fp + fn),
            "h_recall_accuracy": harmonic,
            "paper_percent": {"recall": None if recall is None else 100 * recall,
                              "accuracy": 100 * accuracy, "H": None if harmonic is None else 100 * harmonic},
            "auroc": auroc(labels, scores) if len(set(labels)) == 2 else None,
            "auprc": auprc(labels, scores) if len(set(labels)) == 2 else None,
            "ece": expected_calibration_error(labels, scores)}


def load_samples(manifest: Path, data_root: Path, texverse: Path, texverse_full: Path) -> list[dict]:
    roots = {"workspace": data_root, "texverse": texverse, "texverse_full": texverse_full,
             "TexVerse": texverse, "TexVerse-full": texverse_full}
    samples = {}
    for row in iter_jsonl(manifest):
        value = row["image"]
        if isinstance(value, dict):
            relative = Path(value["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Portable image references must stay inside their declared root")
            path = roots[value["root"]] / relative
        else:
            path = Path(value)
            if not path.is_absolute():
                path = data_root / path
        label = row.get("label")
        if label is None:
            label = {"KEEP": 0, "REPAIR": 1, "PURIFY": 1}[row["decision"]]
        if label not in (0, 1):
            raise ValueError("Expected binary router label")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get("image_sha256") and row["image_sha256"] != digest:
            raise ValueError(f"Image hash mismatch: {path}")
        if digest in samples and samples[digest]["label"] != label:
            raise ValueError(f"Conflicting labels for identical image bytes: {path}")
        samples.setdefault(digest, {"image": str(path.resolve()), "image_sha256": digest,
                                   "asset_id": row.get("asset_id") or row.get("asset"),
                                   "label": label, "sample_id": row.get("sample_id", digest)})
    if not samples:
        raise ValueError("Empty evaluation manifest")
    return list(samples.values())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path.cwd())
    parser.add_argument("--texverse-root", type=Path, default=Path("data/TexVerse"))
    parser.add_argument("--texverse-full-root", type=Path, default=Path("data/TexVerse-full"))
    parser.add_argument("--runtime", "--base-model", dest="base_model", type=Path,
                        default=Path(os.environ.get("GATE3DGEN_RUNTIME_PATH", "checkpoints/runtime")))
    parser.add_argument("--router-checkpoint", type=Path, default=Path("checkpoints/router/checkpoint-6000"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    rows = load_samples(args.manifest, args.data_root, args.texverse_root, args.texverse_full_root)
    router = Router(args.router_checkpoint, args.base_model, args.device, args.threshold)
    for row in rows:
        row.update(router.predict(Path(row["image"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    report = summarize(rows, args.threshold)
    report.update({"deduplication": "identical file SHA256", "threshold_fitted_on_this_manifest": False,
                   "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()})
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
