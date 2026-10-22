from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .io import iter_jsonl


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def long_metric_rows(domain: str, path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    for row in iter_jsonl(path):
        for metric, value in sorted(row.get("metrics", {}).items()):
            yield {
                "domain": domain,
                "case_id": row.get("case_id", ""),
                "asset_id": row.get("asset_id", ""),
                "task_id": row.get("task_id", ""),
                "baseline": row.get("baseline", ""),
                "variant": row.get("variant", ""),
                "reference": row.get("reference", ""),
                "metric": metric,
                "value": value,
            }


def improvement_rows(domain: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[tuple[str, dict[str, Any]]] = []
    if domain == "2d" and isinstance(summary.get("restored_vs_degraded"), dict):
        groups.append(("overall", summary["restored_vs_degraded"]))
        groups.extend(
            (f"task:{task}", value)
            for task, value in sorted(summary.get("restored_vs_degraded_by_task", {}).items())
            if isinstance(value, dict)
        )
    groups.extend(
        (key.removeprefix("improvement:"), value)
        for key, value in sorted(summary.items())
        if key.startswith("improvement:") and isinstance(value, dict)
    )
    rows = []
    for group, value in groups:
        for metric, result in sorted(value.get("metrics", {}).items()):
            gain_ci = result.get("gain_ci95", [None, None])
            win_ci = result.get("win_rate_ci95", [None, None])
            rows.append({
                "domain": domain,
                "group": group,
                "metric": metric,
                "pairs": value.get("count"),
                "assets": result.get("assets"),
                "mean_gain": result.get("mean_gain"),
                "gain_ci95_low": gain_ci[0],
                "gain_ci95_high": gain_ci[1],
                "win_rate": result.get("win_rate"),
                "win_rate_ci95_low": win_ci[0],
                "win_rate_ci95_high": win_ci[1],
            })
    return rows


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(output_dir: Path, sources: list[tuple[str, Path]]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    improvements = []
    source_status = []
    for domain, directory in sources:
        per_case = directory / "per_case.jsonl"
        summary_path = directory / "summary.json"
        domain_metrics = list(long_metric_rows(domain, per_case))
        domain_improvements = improvement_rows(domain, _load_json(summary_path))
        metrics.extend(domain_metrics)
        improvements.extend(domain_improvements)
        source_status.append({
            "domain": domain,
            "directory": str(directory.resolve()),
            "metric_rows": len(domain_metrics),
            "improvement_rows": len(domain_improvements),
            "available": per_case.is_file() and summary_path.is_file(),
        })

    metric_fields = ["domain", "case_id", "asset_id", "task_id", "baseline", "variant", "reference", "metric", "value"]
    improvement_fields = [
        "domain", "group", "metric", "pairs", "assets", "mean_gain",
        "gain_ci95_low", "gain_ci95_high", "win_rate", "win_rate_ci95_low", "win_rate_ci95_high",
    ]
    for path, fields, rows in (
        (output_dir / "metrics_long.csv", metric_fields, metrics),
        (output_dir / "paired_improvements.csv", improvement_fields, improvements),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Gate3DGen evaluation report",
        "",
        "Paired gains are oriented so positive values favor purified over degraded inputs. Confidence intervals and win rates are clustered by asset. Interpret 2D and 3D separately; image scores do not establish geometry quality.",
        "",
        "## 2D image restoration",
        "",
        "Higher PSNR, SSIM, edge correlation, DINO cosine and foreground-mask IoU are better; lower MAE, MSE, gradient MAE and LPIPS are better. Foreground object_lpips_alex is the Table 2 LPIPS column; full-image lpips_alex is a separate diagnostic. Gains are oriented uniformly. Reference image metrics require clean GT.",
        "",
        "## 3D geometry and fixed-view rendering",
        "",
        "Always report the reference: glb_gt denotes a reference mesh, *_image_proxy is not geometry GT, and degraded_input only describes changes for wild cases without GT. Lower MD/CD and higher F-score/normal consistency are better. Fixed-view rendering measures appearance projections, not geometry correctness. geometry_visible_shell_v3 uses first-hit depth-visible point clouds for outer-shell CD/F1; opt-in *_proxy_* and historical outer_shell_* fields do not. Check source schema, protocol and per-metric coverage before comparing runs; never mix historical proxy scores with new shell scores.",
        "",
        "## Paired improvements",
        "",
        "| Domain | Group | Metric | Pairs | Assets | Mean gain [95% CI] | Win rate [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    if improvements:
        for row in improvements:
            gain = f"{_format(row['mean_gain'])} [{_format(row['gain_ci95_low'])}, {_format(row['gain_ci95_high'])}]"
            win = f"{_format(row['win_rate'])} [{_format(row['win_rate_ci95_low'])}, {_format(row['win_rate_ci95_high'])}]"
            lines.append(
                f"| {row['domain']} | {row['group']} | {row['metric']} | {_format(row['pairs'], 0)} | "
                f"{_format(row['assets'], 0)} | {gain} | {win} |"
            )
    else:
        lines.append("| — | — | — | 0 | 0 | — | — |")
    lines.extend(["", "## Source coverage", "", "| Domain | Available | Metric values | Improvement rows | Directory |", "|---|---:|---:|---:|---|"])
    for status in source_status:
        lines.append(
            f"| {status['domain']} | {status['available']} | {status['metric_rows']} | "
            f"{status['improvement_rows']} | `{status['directory']}` |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status = {"sources": source_status, "metric_rows": len(metrics), "improvement_rows": len(improvements)}
    (output_dir / "report_metadata.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Gate3DGen evaluation results as Markdown and long-form CSV")
    parser.add_argument("--metrics2d-dir", type=Path)
    parser.add_argument("--metrics3d-dir", type=Path)
    parser.add_argument("--render-shaded-dir", type=Path)
    parser.add_argument("--render-base-color-dir", type=Path)
    parser.add_argument("--render-normal-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = [
        (name, directory)
        for name, directory in (
            ("2d", args.metrics2d_dir),
            ("geometry3d", args.metrics3d_dir),
            ("render_shaded", args.render_shaded_dir),
            ("render_base_color", args.render_base_color_dir),
            ("render_normal", args.render_normal_dir),
        )
        if directory is not None
    ]
    if not sources:
        raise ValueError("at least one metric directory is required")
    print(json.dumps(write_report(args.output_dir, sources), indent=2))


if __name__ == "__main__":
    main()
