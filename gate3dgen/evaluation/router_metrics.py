"""Binary ranking metrics retained from the original evaluation implementation."""
from __future__ import annotations
from typing import Iterable
import numpy as np


def calibrate_threshold(rows):
    """Original demo-only rule: balanced accuracy, then recall, then lower threshold."""
    labels, probabilities = _arrays([r["label"] for r in rows], [r["purify_probability"] for r in rows])
    if len(set(labels)) != 2:
        raise ValueError("Development calibration requires both classes")
    values = sorted(set(probabilities))
    candidates = sorted({0., 1., *((a+b)/2 for a, b in zip(values, values[1:]))})
    from .router import summarize
    sweep = [summarize(rows, t) for t in candidates]
    for item in sweep:
        item["balanced_accuracy"] = (item["recall"] + item["specificity"]) / 2
    selected = max(sweep, key=lambda m: (m["balanced_accuracy"], m["recall"], -m["threshold"]))
    return selected, sweep


def main(argv=None):
    import argparse, hashlib, json
    from pathlib import Path
    from .io import iter_jsonl
    parser = argparse.ArgumentParser(description="Calibrate only on independent development predictions, record the complete sweep")
    parser.add_argument("--development-predictions", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True, help="Used only for population exclusion, never scoring")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = list(iter_jsonl(args.development_predictions))
    final = list(iter_jsonl(args.final_manifest))
    if not final:
        raise ValueError("Declare a nonempty final population")
    asset = lambda row: str(row.get("asset_id") or row.get("asset") or "")
    if any(not asset(r) for r in rows + final):
        raise ValueError("Stable asset identity is required for development/final exclusion")
    if {asset(r) for r in rows} & {asset(r) for r in final}:
        raise ValueError("Development assets overlap final evaluation")
    if any(not r.get("image_sha256") for r in rows):
        raise ValueError("Development predictions must record image SHA256")
    samples = {}
    for row in rows:
        digest = row["image_sha256"]
        if digest in samples and samples[digest]["label"] != row["label"]:
            raise ValueError("Conflicting labels for identical development bytes")
        samples.setdefault(digest, row)
    selected, sweep = calibrate_threshold(list(samples.values()))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    payload = {"schema": "router-development-calibration-v1", "selected_threshold": selected["threshold"],
        "selected_metrics": selected, "sweep": sweep, "development_predictions_sha256": sha(args.development_predictions),
        "final_manifest_sha256": sha(args.final_manifest), "samples": len(samples),
        "candidate_rule": "0, 1, midpoints of unique development probabilities",
        "objective": "maximize balanced accuracy; then recall; then lower threshold",
        "final_scores_used": False, "asset_disjoint": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0

def _arrays(labels: Iterable[int], probabilities: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(list(labels), dtype=np.int64)
    probabilities = np.asarray(list(probabilities), dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.size != probabilities.size:
        raise ValueError("labels and probabilities must be equal-length 1D arrays")
    if labels.size == 0 or set(np.unique(labels)) - {0, 1}:
        raise ValueError("labels must be a non-empty binary array")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    return labels, probabilities


def auroc(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    labels, probabilities = _arrays(labels, probabilities)
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative samples")
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and probabilities[order[end]] == probabilities[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auprc(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    """Average precision, i.e. the standard step-wise area under PR curve."""

    labels, probabilities = _arrays(labels, probabilities)
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("AUPRC requires positive samples")
    order = np.argsort(-probabilities, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = probabilities[order]
    cumulative_tp = np.cumsum(ordered_labels)
    # Evaluate only at the end of each tied-score group so AUPRC is independent
    # of the manifest order within ties.
    group_ends = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    tp = cumulative_tp[group_ends].astype(np.float64)
    predicted = np.arange(1, ordered_labels.size + 1, dtype=np.float64)[group_ends]
    recall = tp / positives
    precision = tp / predicted
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def expected_calibration_error(
    labels: Iterable[int], probabilities: Iterable[float], bins: int = 15
) -> float:
    labels, probabilities = _arrays(labels, probabilities)
    bins = int(bins)
    if bins <= 0:
        raise ValueError("ECE bins must be positive")
    assignments = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return error


if __name__ == "__main__":
    raise SystemExit(main())

