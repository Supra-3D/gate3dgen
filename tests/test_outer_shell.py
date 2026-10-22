"""Small geometric fixtures distinguish actual visibility from old proxies."""
import json
import subprocess
import sys

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("embreex")
pytest.importorskip("rtree")
from gate3dgen.evaluation.outer_shell import camera_frame, extract_outer_shell, shell_distances
from gate3dgen.evaluation.metrics3d_official import evaluate_pair, _fscore


def extract(mesh, *, samples=500, seed=12):
    center, radius = camera_frame([mesh])
    return extract_outer_shell(mesh, center=center, radius=radius,
                               samples=samples, seed=seed, views=20, resolution=32)


def test_enclosed_internal_surface_is_not_sampled():
    outer = trimesh.creation.box()
    inner = trimesh.creation.box(extents=[0.4] * 3)
    nested = trimesh.util.concatenate([outer, inner])
    p, metadata = extract(nested)
    assert np.allclose(np.max(np.abs(p), axis=1), 0.5, atol=1e-6)
    assert metadata["sample_count"] == 500
    assert metadata["dense_point_count"] > 500
    # Identical external geometry yields identical dense pixels even with hidden faces.
    q, _ = extract(outer)
    assert np.allclose(np.sort(p, axis=0), np.sort(q, axis=0), atol=1e-6)
    # The same hidden surface DOES contribute to all-surface mesh distance.
    inner_points, _ = trimesh.sample.sample_surface(inner, 100, seed=12)
    _, full_surface_distance, _ = trimesh.proximity.closest_point(outer, inner_points)
    assert np.allclose(full_surface_distance, 0.3)


def test_visible_concavity_is_preserved_not_convex_hull():
    # Open-top box: interior bottom and inner side walls are visible from above.
    box = trimesh.creation.box()
    open_box = box.submesh([np.flatnonzero(box.face_normals[:, 2] < 0.9)], append=True)
    p, _ = extract(open_box)
    assert np.any((np.abs(p[:, 0]) < 0.4) & (np.abs(p[:, 1]) < 0.4) & (p[:, 2] < -0.49))
    assert not np.any((np.abs(p[:, 0]) < 0.4) & (np.abs(p[:, 1]) < 0.4) & (p[:, 2] > 0.49))


def test_shell_sampling_deterministic_and_no_duplicate_filling():
    mesh = trimesh.creation.box()
    a, meta = extract(mesh)
    b, _ = extract(mesh)
    assert np.array_equal(a, b)
    assert np.unique(a, axis=0).shape[0] == len(a)
    with pytest.raises(ValueError, match="valid shell depth pixels"):
        extract(mesh, samples=meta["dense_point_count"] + 1)


def test_nearest_point_cd_and_squared_fscore_threshold():
    a, b = np.array([[0., 0., 0.]]), np.array([[0., 0., 0.01]])
    da, db = shell_distances(a, b)
    assert np.mean(np.concatenate([da, db])**2) == pytest.approx(1e-4)
    assert _fscore(da, db, 1e-6)[0] == 0
    assert _fscore(da, db, 1e-3)[0] == 1
    assert _fscore(da, db, float(da[0]**2))[0] == 0
    assert _fscore(da, db, float(da[0]**2), strict=False)[0] == 1


def test_pair_defaults_to_actual_shell_and_exact_md():
    mesh = trimesh.creation.box()
    row = evaluate_pair(mesh, mesh, samples=128, shell_views=20, shell_resolution=32)
    assert row["protocol"]["outer_shell_render_protocol"] is True
    assert row["protocol"]["paper_outer_surface_metrics_available"] is True
    assert row["protocol"]["point_to_triangle_approximation"] is False
    assert row["metrics"]["mesh_distance"] < 1e-12
    assert row["metrics"]["outer_shell_chamfer_distance"] > 0  # independent point samples
    assert not any("proxy" in key for key in row["metrics"])
    assert row["protocol"]["outer_shell_extraction"]["prediction"]["views"] == 20


def test_exact_md_uses_distance_batches_without_changing_values(monkeypatch):
    mesh = trimesh.creation.box()
    closest = trimesh.proximity.closest_point
    sizes = []

    def record_batch(mesh, points):
        sizes.append(len(points))
        return closest(mesh, points)

    monkeypatch.setattr(trimesh.proximity, "closest_point", record_batch)
    small = evaluate_pair(mesh, mesh, samples=33, chunk_size=8,
                          exhaustive_chunk_size=2, shell_views=10, shell_resolution=32)
    assert sizes == [8, 8, 8, 8, 1] * 2
    sizes.clear()
    large = evaluate_pair(mesh, mesh, samples=33, chunk_size=33,
                          exhaustive_chunk_size=2, shell_views=10, shell_resolution=32)
    assert sizes == [33, 33]
    assert small["metrics"] == pytest.approx(large["metrics"], abs=1e-12)


def test_cli_shell_summary_and_resume_configuration_isolation(tmp_path):
    gt = tmp_path / "gt.glb"
    trimesh.creation.box().export(gt)
    pred_dir = tmp_path / "generation" / "trellis2" / "case1"
    pred_dir.mkdir(parents=True)
    trimesh.creation.box().export(pred_dir / "degraded.glb")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"case_id": "case1", "gt_mesh": str(gt)}) + "\n")
    output = tmp_path / "output"
    command = [sys.executable, "-m", "gate3dgen.evaluation.metrics3d_official",
        "--manifest", str(manifest), "--generation-dir", str(tmp_path / "generation"),
        "--output-dir", str(output), "--baseline", "trellis2", "--variant", "degraded",
        "--samples", "128", "--shell-views", "10", "--resume", "--shell-resolution"]
    subprocess.run(command + ["32"], capture_output=True, text=True, check=True)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["paper_outer_surface_metrics_available"] is True
    assert summary["groups"]["trellis2:glb_gt:degraded"]["metric_coverage"]["outer_shell_fscore"]["coverage"] == 1
    before = json.loads((output / "per_case.jsonl").read_text())
    subprocess.run(command + ["32"], capture_output=True, text=True, check=True)
    assert before == json.loads((output / "per_case.jsonl").read_text())
    subprocess.run(command + ["48"], capture_output=True, text=True, check=True)
    after = json.loads((output / "per_case.jsonl").read_text())
    assert before["protocol"]["metric_config_hash"] != after["protocol"]["metric_config_hash"]
    assert len(list(output.glob("per_case.incompatible.*.jsonl"))) == 1


@pytest.mark.parametrize("options", [["--scaled-demo"], ["--shell-views", "12"],
                                   ["--surface-proxy-method", "raw_surface"], ["--allow-missing"]])
def test_paper_cli_rejects_incompatible_protocol(tmp_path, options):
    result = subprocess.run([sys.executable, "-m", "gate3dgen.evaluation.metrics3d_official",
        "--manifest", str(tmp_path / "manifest.jsonl"), "--generation-dir", str(tmp_path),
        "--output-dir", str(tmp_path / "output"), "--paper-protocol", *options],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "paper protocol requires" in result.stderr
    assert not (tmp_path / "output" / "summary.json").exists()
