from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np


def clustered_mean_ci(
    items: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    cluster_key: str = "asset_id",
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 20260826,
) -> dict[str, Any]:
    """Mean and percentile CI from a cluster bootstrap.

    Values are averaged within each asset first, then assets are sampled with
    replacement. This prevents the 12 degradations of one TexVerse asset from
    being treated as 12 independent 3D subjects.
    """
    clustered: dict[str, list[float]] = defaultdict(list)
    for item in items:
        number = float(value(item))
        if np.isfinite(number):
            clustered[str(item[cluster_key])].append(number)
    cluster_values = np.asarray(
        [np.mean(numbers) for _, numbers in sorted(clustered.items())], dtype=np.float64
    )
    if cluster_values.size == 0:
        return {"mean": None, "ci95": [None, None], "clusters": 0}
    mean = float(cluster_values.mean())
    if cluster_values.size == 1:
        return {"mean": mean, "ci95": [mean, mean], "clusters": 1}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, cluster_values.size, size=(resamples, cluster_values.size))
    boot = cluster_values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return {"mean": mean, "ci95": [float(low), float(high)], "clusters": int(cluster_values.size)}


def grouped_metric_summary(
    items: list[dict[str, Any]], metric_names: list[str], *, seed: int = 20260826
) -> dict[str, Any]:
    metrics = {
        name: clustered_mean_ci(
            [item for item in items if name in item["metrics"]],
            lambda item, metric=name: item["metrics"][metric],
            seed=seed,
        )
        for name in metric_names
    }
    return {
        "count": len(items),
        "assets": len({str(item["asset_id"]) for item in items}),
        "metrics": metrics,
        # Kept for convenient table generation and backward compatibility.
        "mean": {name: values["mean"] for name, values in metrics.items()},
    }

