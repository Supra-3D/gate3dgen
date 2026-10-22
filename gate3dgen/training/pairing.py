"""Deterministic same-albedo dual-view pairing for recipe B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class AlbedoPair:
    """Two real degradations that share one underlying albedo render."""

    group_id: str
    first: dict[str, Any]
    second: dict[str, Any]

    @property
    def pair_id(self) -> str:
        return f"{self.first.get('pair_id')}__{self.second.get('pair_id')}"


def albedo_group_id(row: dict[str, Any]) -> str:
    value = (
        row.get("albedo_group_id")
        or row.get("albedo_relative_path")
        or row.get("source_relative_path")
        or row.get("albedo_path")
    )
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {row.get('pair_id')} has no albedo group identity")
    return value.replace("\\", "/")


def pair_rows_by_albedo(
    rows: Iterable[dict[str, Any]],
    *,
    require_distinct_degradation: bool = True,
) -> list[AlbedoPair]:
    """Return stable, non-overlapping pairs grouped by albedo path.

    Groups and rows are sorted by their explicit IDs, so pair manifests do not
    depend on filesystem/glob order or worker scheduling.  A group with an odd
    number of views contributes its deterministic prefix; the leftover is
    reported by :func:`pairing_summary` and never silently paired cross-object.
    """

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = albedo_group_id(row)
        groups.setdefault(group, []).append(row)
    pairs: list[AlbedoPair] = []
    for group, group_rows in sorted(groups.items()):
        ordered = sorted(group_rows, key=lambda row: (str(row.get("task_id", "")), str(row.get("pair_id", ""))))
        index = 0
        while index + 1 < len(ordered):
            first, second = ordered[index], ordered[index + 1]
            if require_distinct_degradation and first.get("degradation_type") == second.get("degradation_type"):
                swap = next(
                    (candidate for candidate in range(index + 2, len(ordered))
                     if ordered[candidate].get("degradation_type") != first.get("degradation_type")),
                    None,
                )
                if swap is None:
                    index += 1
                    continue
                ordered[index + 1], ordered[swap] = ordered[swap], ordered[index + 1]
                second = ordered[index + 1]
            pairs.append(AlbedoPair(group, first, second))
            index += 2
    return pairs


def pairing_summary(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    rows = list(rows)
    groups: dict[str, int] = {}
    for row in rows:
        group = albedo_group_id(row)
        groups[group] = groups.get(group, 0) + 1
    # Report the same eligibility rule used by the B sampler.  A group with
    # two rows of one degradation type is not a real dual-view pair, even
    # though its cardinality is even; counting the constructed pairs keeps
    # the report auditable and prevents overstating the B training population.
    pairs = pair_rows_by_albedo(rows, require_distinct_degradation=True)
    paired_groups = {pair.group_id for pair in pairs}
    return {
        "groups": len(groups),
        "rows": sum(groups.values()),
        "paired_groups": len(paired_groups),
        "paired_rows": 2 * len(pairs),
        "singleton_groups": sum(1 for count in groups.values() if count == 1),
        "unpaired_rows": len(rows) - 2 * len(pairs),
        "pairs": len(pairs),
    }


__all__ = [
    "AlbedoPair",
    "albedo_group_id",
    "pair_rows_by_albedo",
    "build_albedo_pairs",
    "pair_same_albedo_views",
    "pairing_summary",
]

# Friendly aliases used by downstream data-contract tools.
build_albedo_pairs = pair_rows_by_albedo
pair_same_albedo_views = pair_rows_by_albedo
