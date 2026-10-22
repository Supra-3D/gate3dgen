"""Read-only audit of historical report aggregates and their geometry sources.

This does not reevaluate meshes, change historical results, certify a paper
protocol, or infer PBR view coverage from ambiguous legacy aggregate counts.
Only anonymous aggregate evidence and file hashes are written to the output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from .io import iter_jsonl

GEOMETRY_NAMES = ("mesh_distance", "mesh_distance_fscore", "outer_shell_chamfer_distance", "outer_shell_fscore")
PBR_NAMES = (("normal", "psnr_db"), ("shaded", "lpips_alex"), ("base_color", "lpips_alex"))
LABELS = {
    "Ours": "Gate-B mmDiT", "FireRed": "FireRed 1.1",
    "FLUX.1": "FLUX.1-Kontext-dev", "OmniGen2": "OmniGen2",
    "Step1X": "Step1X-Edit v1.2", "Qwen-Image-Edit": "Qwen-Image-Edit-2511",
    "Mage-Flow": "MAGE Flow Edit RL-Aligned", "GPT Image-2.5 Flare": "gpt-image-2.5-flare",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve reported counts and their units, rather than reinterpret them."""
    rows = []
    for method, scope in report["scopes"].items():
        for backend, model in scope["models"].items():
            for condition, geometry in model["geometry"]["groups"].items():
                if method != "Gate-B mmDiT" and not condition.startswith("restored"):
                    continue
                metrics = {}
                for name in GEOMETRY_NAMES:
                    value = geometry["metrics"][name]
                    metrics[name] = {"mean": value.get("mean"), "valid_cases": value.get("count"),
                                     "expected_cases": scope["expected_cases"]}
                for channel, name in PBR_NAMES:
                    group = model["texture"]["groups"].get(f"{condition}:{channel}", {})
                    value = group.get("metrics", {}).get(name, {})
                    metrics[f"{channel}.{name}"] = {
                        "mean": value.get("mean"), "reported_count": value.get("count"),
                        "reported_records": group.get("records"), "reported_cases": group.get("cases"),
                        "reported_aggregation_unit": group.get("aggregation_unit"),
                        "valid_view_count": None, "complete_four_view_cases": None,
                        "coverage_state": "requires_the_actual_multiview_source_rows",
                    }
                rows.append({"method": method, "backend": backend, "condition": condition, "metrics": metrics})
    return rows


def table_rows(tex: str, label: str) -> dict[str, list[str]]:
    section = tex.split("\\label{" + label + "}", 1)[1].split("\\end{table}", 1)[0]
    result = {}
    for line in section.splitlines():
        if line.lstrip().startswith("%") or "&" not in line:
            continue
        fields = line.split("&")
        numbers = [re.findall(r"\d+\.\d+", field) for field in fields[1:]]
        numbers = [items[0] for items in numbers if len(items) == 1]
        if not numbers:
            continue
        name = re.sub(r"\$[^$]*\$|\\[A-Za-z]+|[{}]", "", fields[0]).strip()
        result[name] = numbers
    return result


def compare_table(rows: list[dict[str, Any]], tex: str, label: str, backend: str) -> dict[str, Any]:
    comparisons = []
    for name, cells in table_rows(tex, label).items():
        method = LABELS.get(name)
        condition = "restored"
        if name == "Clean reference (Oracle)":
            method, condition = "Gate-B mmDiT", "clean_gt"
        elif name == "Degraded (Input)":
            method, condition = "Gate-B mmDiT", "degraded"
        matching = [row for row in rows if row["method"] == method and row["backend"] == backend
                    and (row["condition"].startswith("restored") if condition == "restored" else row["condition"] == condition)]
        if len(matching) != 1 or len(cells) != 7:
            comparisons.append({"row": name, "status": "unresolved"})
            continue
        row = matching[0]
        keys = [*GEOMETRY_NAMES, *[f"{channel}.{metric}" for channel, metric in PBR_NAMES]]
        formatted = []
        for key, scale, cell in zip(keys, (1000, 1000, 1000, 10, 1, 1, 1), cells):
            value = row["metrics"][key]["mean"]
            decimals = len(cell.split(".")[1])
            formatted.append(f"{value * scale:.{decimals}f}" if _finite(value) else None)
        comparisons.append({"row": name, "paper_cells": cells, "report_cells": formatted,
                            "status": "match" if cells == formatted else "mismatch"})
    return {"label": label, "backend": backend, "rows": comparisons,
            "all_reported_cells_match": bool(comparisons) and all(row["status"] == "match" for row in comparisons)}


def compare_ablation_table(report: dict[str, Any], rows: list[dict[str, Any]], tex: str) -> dict[str, Any]:
    """Check candidate numeric matches without claiming a training-contract binding."""
    output = []
    candidates = {"Full": "Gate-B mmDiT", "B": "Gate-B ablate (affected L1 + subject grad = 0)"}
    for name, cells in table_rows(tex, "tab:ablate_editor").items():
        method = candidates.get(name)
        matching = [row for row in rows if row["method"] == method and row["backend"] == "trellis2"
                    and row["condition"].startswith("restored")]
        if len(matching) != 1 or len(cells) != 8:
            output.append({"row": name, "status": "source_not_bound"})
            continue
        row = matching[0]
        quality = report["scopes"][method].get("two_d", {}).get("groups", {}).get(row["condition"], {}).get("metrics", {})
        values = [quality.get(key, {}).get("mean") for key in ("object_lpips_alex", "object_psnr_db")]
        values += [row["metrics"][key]["mean"] for key in (*GEOMETRY_NAMES, "shaded.lpips_alex", "base_color.lpips_alex")]
        formatted = [f"{value * scale:.{len(cell.split('.')[1])}f}" if _finite(value) else None
                     for value, scale, cell in zip(values, (1, 1, 1000, 1000, 1000, 10, 1, 1), cells)]
        output.append({"row": name, "candidate_report_method": method, "paper_cells": cells,
                       "report_cells": formatted, "cell_matches": [a == b for a, b in zip(cells, formatted)],
                       "all_cells_match": cells == formatted, "training_contract_verified": False,
                       "status": "numeric_candidate_only"})
    return {"label": "tab:ablate_editor", "rows": output,
            "complete_table_source_binding": False,
            "note": "Numeric matches alone do not prove the paper's A/B/Full loss and conditioning contracts."}


def audit(comparison: Path, source_code: Path, experiments: Path | None = None, appendix: Path | None = None) -> dict[str, Any]:
    report = json.loads(comparison.read_text())
    rows = report_rows(report)
    evidence = {"comparison_summary": digest(comparison)}
    for relative in ("scripts/run_baseline_dual3d_32gpu.sh", "scripts/run_full_benchmark_dual3d_40gpu.sh", "texverse_eval/metrics3d_official.py"):
        evidence[relative] = digest(source_code / relative)
    audits = []
    for key, root in report["paths"]["source_reports"].items():
        combined_path = Path(root) / "summary.json"
        combined = json.loads(combined_path.read_text())
        evidence[f"{key}.combined_summary"] = digest(combined_path)
        excluded_path = Path(root) / "excluded_dtcplus_cases.json"
        excluded = {row["case_id"] for row in json.loads(excluded_path.read_text())["cases"]}
        evidence[f"{key}.excluded_population"] = digest(excluded_path)
        method = next(label for label, scope in report["scopes"].items() if scope["model_filter_key"] == key)
        counters = defaultdict(Counter)
        scores = defaultdict(list)
        seen = set()
        hash_bindings = []
        for source in combined["runs"]:
            summary_path = Path(source["summary"])
            actual_hash = digest(summary_path)
            evidence[f"{key}.{source['split']}.summary"] = actual_hash
            hash_bindings.append({"split": source["split"], "recorded_summary_hash_matches": actual_hash == source.get("summary_sha256")})
            for backend in ("trellis2", "pixal3d"):
                path = Path(source["run"]) / "metrics" / backend / "full" / "geometry" / "per_case.jsonl"
                evidence[f"{key}.{source['split']}.{backend}.geometry_rows"] = digest(path)
                for row in iter_jsonl(path):
                    if source["split"] == "dtcplus" and row["case_id"] in excluded:
                        continue
                    condition = str(row.get("condition") or row.get("variant"))
                    identity = (backend, condition, row["case_id"])
                    if identity in seen:
                        raise ValueError(f"duplicate historical geometry identity in {key}")
                    seen.add(identity)
                    protocol = row.get("protocol") or {}
                    flags = tuple(str(protocol.get(name)) for name in ("outer_shell_method", "outer_shell_exact", "outer_shell_render_protocol", "samples"))
                    counters[(backend, condition)][flags] += 1
                    for name in GEOMETRY_NAMES:
                        value = row.get("metrics", {}).get(name)
                        if _finite(value):
                            scores[(backend, condition, name)].append(value)
        for item in [row for row in rows if row["method"] == method]:
            backend, condition = item["backend"], item["condition"]
            checks = []
            for name in GEOMETRY_NAMES:
                values = scores[(backend, condition, name)]
                reported = item["metrics"][name]
                mean = statistics.mean(values) if values else None
                match = len(values) == reported["valid_cases"] and mean is not None and math.isclose(mean, reported["mean"], rel_tol=1e-12, abs_tol=1e-15)
                checks.append({"metric": name, "valid_cases": len(values), "mean": mean, "report_matches_source_rows": match})
            audits.append({"method": method, "backend": backend, "condition": condition,
                           "protocol_counts": [{"method": flags[0], "exact": flags[1], "rendered": flags[2], "samples": flags[3], "records": count}
                                               for flags, count in sorted(counters[(backend, condition)].items())],
                           "source_summary_hash_bindings": hash_bindings, "metric_checks": checks})
    tables = []
    if experiments:
        evidence["paper_experiments_tex"] = digest(experiments)
        tables.append(compare_table(rows, experiments.read_text(), "tab:three_d_quality", "trellis2"))
    if appendix:
        evidence["paper_appendix_tex"] = digest(appendix)
        tables.append(compare_table(rows, appendix.read_text(), "tab:pixal3d_quality", "pixal3d"))
    return {"schema_version": "historical_3d_evidence_v1", "verification": "agent_checked_human_approval_pending",
            "population": report["combined_population"], "source_sha256": evidence,
            "reported_rows": rows, "geometry_source_audits": audits, "paper_table_checks": tables,
            "ablation_table_checks": compare_ablation_table(report, rows, experiments.read_text()) if experiments else None,
            "pbr_source_rows_verified": False,
            "notes": ["No images, source paths or case IDs are included.",
                      "Legacy PBR count mixes case and view units; it is preserved, never converted to valid-view coverage.",
                      "Table 5 A is not bound by this comparison report; ablation names do not prove a matching training contract.",
                      "Numerical table agreement does not validate the method definitions or complete/common populations."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-summary", type=Path, required=True)
    parser.add_argument("--source-code-root", type=Path, required=True)
    parser.add_argument("--paper-experiments", type=Path)
    parser.add_argument("--paper-appendix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output already exists; use a fresh evidence file")
    result = audit(args.comparison_summary, args.source_code_root, args.paper_experiments, args.paper_appendix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    checks = [check for group in result["geometry_source_audits"] for check in group["metric_checks"]]
    print(json.dumps({"geometry_groups": len(result["geometry_source_audits"]),
                      "geometry_checks": len(checks), "geometry_checks_passed": all(check["report_matches_source_rows"] for check in checks),
                      "paper_tables": len(result["paper_table_checks"]),
                      "paper_tables_match": all(table["all_reported_cells_match"] for table in result["paper_table_checks"]),
                      "pbr_source_rows_verified": False, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
