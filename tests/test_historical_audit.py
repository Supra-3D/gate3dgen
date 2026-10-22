import json
from pathlib import Path
import subprocess
import sys

import pytest

from gate3dgen.evaluation.historical_audit import GEOMETRY_NAMES, report_rows, table_rows, compare_table, compare_ablation_table


def report():
    geometry = {name: {"mean": value, "count": 2} for name, value in zip(GEOMETRY_NAMES, (0.001, 0.01, 0.0011, 0.1))}
    texture = {
        "restored_s01:" + channel: {"cases": 2, "records": 8, "aggregation_unit": "case_macro_after_view_mean",
            "metrics": {metric: {"mean": value, "count": count}}}
        for channel, metric, value, count in (("normal", "psnr_db", 15.0, 8), ("shaded", "lpips_alex", 0.2, 2), ("base_color", "lpips_alex", 0.3, 2))
    }
    return {"scopes": {"Gate-B mmDiT": {"expected_cases": 2, "models": {"trellis2": {
        "geometry": {"groups": {"restored_s01": {"metrics": geometry}}}, "texture": {"groups": texture}}}}}}


def test_historical_counts_are_not_misrepresented_as_view_coverage():
    rows = report_rows(report())
    value = rows[0]["metrics"]["shaded.lpips_alex"]
    assert value["reported_count"] == 2 and value["reported_records"] == 8
    assert value["valid_view_count"] is None
    assert value["complete_four_view_cases"] is None
    assert value["coverage_state"] == "requires_the_actual_multiview_source_rows"


def test_latex_table_check_ignores_comments_and_preserves_precision():
    tex = r"""\label{tab:test}
    % Ours & 9.999 & 9.99 & 9.999 & 9.9999 & 9.99 & 9.9999 & 9.9999\\
    \textbf{Ours} & \textbf{1.000} & 10.00 & 1.100 & 1.0000 & 15.00 & 0.2000 & 0.3000\\
    \end{table}"""
    assert len(table_rows(tex, "tab:test")) == 1
    result = compare_table(report_rows(report()), tex, "tab:test", "trellis2")
    assert result["all_reported_cells_match"]
    result = compare_table(report_rows(report()), tex.replace("15.00", "16.00"), "tab:test", "trellis2")
    assert not result["all_reported_cells_match"]


def test_ablation_candidate_never_certifies_training_contract():
    data = report()
    data["scopes"]["Gate-B mmDiT"]["two_d"] = {"groups": {"restored_s01": {"metrics": {
        "object_lpips_alex": {"mean": 0.12}, "object_psnr_db": {"mean": 22.0}}}}}
    tex = r"""\label{tab:ablate_editor}
    A & & & 0.1111 & 21.000 & 1.000 & 10.00 & 1.100 & 1.0000 & 0.2000 & 0.3000\\
    Full & & & 0.1200 & 22.000 & 1.000 & 10.00 & 1.100 & 1.0000 & 0.2000 & 0.3000\\
    \end{table}"""
    result = compare_ablation_table(data, report_rows(data), tex)
    assert not result["complete_table_source_binding"]
    assert result["rows"][0]["status"] == "source_not_bound"
    assert result["rows"][1]["all_cells_match"]
    assert not result["rows"][1]["training_contract_verified"]


def test_public_historical_evidence_has_real_geometry_and_table_bindings():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "docs/metadata/historical_3d_audit.json").read_text())
    assert len(data["geometry_source_audits"]) == 22
    checks = [check for group in data["geometry_source_audits"] for check in group["metric_checks"]]
    assert len(checks) == 88 and all(check["report_matches_source_rows"] for check in checks)
    assert all(table["all_reported_cells_match"] for table in data["paper_table_checks"])
    assert sum(len(table["rows"]) for table in data["paper_table_checks"]) == 20
    assert not data["pbr_source_rows_verified"]
    ablations = {row["row"]: row for row in data["ablation_table_checks"]["rows"]}
    assert ablations["A"]["status"] == "source_not_bound"
    assert ablations["Full"]["all_cells_match"]
    assert ablations["B"]["cell_matches"] == [True] * 5 + [False] * 3
    for group in data["geometry_source_audits"]:
        for protocol in group["protocol_counts"]:
            if protocol["method"] != "None":
                assert protocol["method"] == "surface" and protocol["exact"] == protocol["rendered"] == "False"
    text = json.dumps(data)
    assert "/" + "home/" not in text
    assert '"case_id"' not in text and '"asset_id"' not in text


def test_historical_cli_rejects_changed_sample_protocol(tmp_path):
    pytest.importorskip("trimesh")
    result = subprocess.run([
        sys.executable, "-m", "gate3dgen.evaluation.metrics3d_official",
        "--manifest", str(tmp_path / "missing.jsonl"), "--generation-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "output"), "--historical-report-protocol", "--scaled-demo",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "historical-report protocol requires" in result.stderr
    assert not (tmp_path / "output").exists()
