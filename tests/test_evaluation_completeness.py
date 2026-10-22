from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from gate3dgen.evaluation import pbr_metrics as pbr


def paper_rows():
    return [
        {"baseline": backend, "variant": variant, "reference": "glb_gt",
         "reference_is_true_gt": True, "camera_contract": "shared_verified",
         "case_id": "case_a", "channel": channel, "view": view, "status": "ok",
         "metrics": {"psnr_db": 30.0, "lpips_alex": 0.2}}
        for backend in ("pixal3d", "trellis2")
        for variant in ("degraded", "restored_s01")
        for channel in pbr.PAPER_CHANNELS for view in range(4)
    ]


def test_partial_lpips_is_not_complete_coverage():
    rows = paper_rows()[:2]
    rows[1]["metrics"]["lpips_alex"] = None
    group = next(iter(pbr.summarize_render_rows(rows)["groups"].values()))
    assert group["metrics"]["lpips_alex"] == 0.2
    assert group["metric_coverage"]["lpips_alex"] == {
        "valid_count": 1, "expected_count": 2, "coverage": 0.5,
        "positive_infinity_count": 0,
    }
    assert group["coverage_definition"] == "complete_record_status_not_metric_coverage"


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), True])
def test_paper_rejects_invalid_lpips(missing):
    rows = paper_rows()
    next(row for row in rows if row["channel"] == "shaded")["metrics"]["lpips_alex"] = missing
    with pytest.raises(ValueError, match="required paper metric"):
        pbr.validate_paper_render_rows(rows)


def test_complete_paper_population_and_perfect_psnr():
    rows = paper_rows()
    rows[0]["metrics"]["psnr_db"] = float("inf")
    audit = pbr.validate_paper_render_rows(rows)
    assert audit["methods"] == 4 and audit["case_views"] == 4
    group = pbr.summarize_render_rows(rows)["groups"]["pixal3d:degraded:glb_gt:normal"]
    assert group["metrics"]["psnr_db"] == "Infinity"


@pytest.mark.parametrize("mutation", ["duplicate", "missing_view", "different_case", "missing_channel", "camera", "proxy_gt", "partial"])
def test_paper_population_validation(mutation):
    rows = deepcopy(paper_rows())
    if mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "missing_view":
        rows.pop()
    elif mutation == "different_case":
        for row in rows:
            if row["baseline"] == "pixal3d" and row["variant"] == "degraded":
                row["case_id"] = "case_b"
    elif mutation == "missing_channel":
        rows = [row for row in rows if row["channel"] != "base_color"]
    elif mutation == "camera":
        rows[0]["camera_contract"] = "unverified"
    elif mutation == "proxy_gt":
        rows[0]["reference_is_true_gt"] = False
    else:
        rows[0]["status"] = "partial"
    with pytest.raises(ValueError):
        pbr.validate_paper_render_rows(rows)


def test_lpips_failure_marks_image_partial(tmp_path, monkeypatch):
    image = tmp_path / "shaded" / "view_00.png"
    mask = tmp_path / "mask" / image.name
    image.parent.mkdir(); mask.parent.mkdir()
    Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(image)
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(mask)
    monkeypatch.setattr(pbr, "_lpips", lambda *args: (None, "lpips_failed:fixture"))
    rows = pbr.evaluate_paired_render_rows([{
        "channel": "shaded", "camera_contract": "shared_verified",
        "prediction_path": str(image), "target_path": str(image),
    }], enable_lpips=True)
    assert rows[0]["status"] == "partial"
    group = next(iter(pbr.summarize_render_rows(rows)["groups"].values()))
    assert group["coverage"] == 0
    assert group["metric_coverage"]["lpips_alex"]["coverage"] == 0
    rows[0].update(status="unavailable", metrics={})
    group = next(iter(pbr.summarize_render_rows(rows)["groups"].values()))
    assert group["metric_coverage"]["lpips_alex"]["expected_count"] == 1
    assert group["metrics"]["lpips_alex"] is None


@pytest.mark.parametrize("level", [0, 1, 2])
def test_blank_color_management_masks_cannot_produce_perfect_scores(tmp_path, level):
    image = tmp_path / "shaded/view_00.png"
    mask = tmp_path / "mask/view_00.png"
    image.parent.mkdir(); mask.parent.mkdir()
    Image.fromarray(np.full((32, 32, 3), level, dtype=np.uint8)).save(image)
    Image.fromarray(np.full((32, 32), level, dtype=np.uint8)).save(mask)
    rows = pbr.evaluate_paired_render_rows([{
        "channel": "shaded", "camera_contract": "shared_verified",
        "prediction_path": str(image), "target_path": str(image),
    }])
    assert rows[0]["status"] == "unavailable"
    assert "no usable foreground signal" in rows[0]["reason"]


def test_paper_cli_missing_renders_never_writes_summary(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"case_id":"a","restoration_steps":[1]}\n')
    output = tmp_path / "evaluation"
    result = subprocess.run([
        sys.executable, "-m", "gate3dgen.evaluation.pbr_metrics",
        "--manifest", str(manifest), "--render-root", str(tmp_path / "renders"),
        "--output-dir", str(output), "--paper-protocol", "--lpips",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert not (output / "summary.json").exists()
    assert (output / "validation_failure.json").is_file()
    assert (output / "per_case.jsonl").is_file()


def test_source_release_ignores_checkpoint_payloads(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / ".gitignore").write_text((root / ".gitignore").read_text())
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    paths = ["checkpoints/router/model.safetensors", "checkpoints/runtime/vae/config.json",
             "Gate3DGen-hf-upload/README.md", "accidental-model.bin", "._README.md"]
    result = subprocess.run(["git", "check-ignore", *paths], cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == paths
    assert subprocess.run(["git", "check-ignore", "setup/download_weights.py"], cwd=tmp_path,
                          capture_output=True).returncode == 1


def test_all_nonfinite_metrics_report_zero_coverage():
    rows = paper_rows()[:2]
    for row in rows:
        row["metrics"]["lpips_alex"] = float("nan")
    group = next(iter(pbr.summarize_render_rows(rows)["groups"].values()))
    assert group["metrics"]["lpips_alex"] is None
    assert group["metric_coverage"]["lpips_alex"]["valid_count"] == 0


def test_common_subset_cannot_replace_declared_plan():
    planned = paper_rows()
    subset = [row for row in planned if row["variant"] != "degraded"]
    # The remaining methods have identical populations, but an entire declared
    # method was dropped: the full-plan contract must still reject the result.
    with pytest.raises(ValueError, match="entire declared render plan"):
        pbr.validate_paper_render_rows(subset, expected_rows=planned)


def test_geometry_proxy_has_no_shell_metric_aliases():
    trimesh = pytest.importorskip("trimesh")
    from gate3dgen.evaluation.metrics3d_official import evaluate_pair, identity_metrics, METRIC_DIRECTIONS
    mesh = trimesh.creation.box()
    for method in ("raw_surface", "convex_hull"):
        result = evaluate_pair(mesh, mesh, samples=64, surface_proxy_method=method)
        assert result["protocol"]["paper_outer_surface_metrics_available"] is False
        assert not any(name.startswith("outer_shell_") for name in result["metrics"])
        assert f"{method}_proxy_distance" in result["metrics"]
        mapping = result["protocol"]["historical_field_mapping"]
        if method == "raw_surface":
            assert mapping["raw_surface_proxy_distance"] == "outer_shell_chamfer_distance"
        else:
            assert mapping == {}
        assert result["metrics"][f"{method}_proxy_distance"] < 1e-12
        assert set(identity_metrics(method)).issubset(METRIC_DIRECTIONS)
