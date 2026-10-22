import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from gate3dgen.data.pair_generation.planner import load_assets
from gate3dgen.evaluation.metrics3d_official import load_mesh
from gate3dgen.training.data import adapt_row

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_example_source_inventory_is_portable_and_decodable(monkeypatch):
    monkeypatch.chdir(ROOT)
    assets = load_assets(ROOT / "assets/examples/data/assets.jsonl", ROOT / "assets/examples/data", 42, verify_images=True)
    assert len(assets) == 1
    assert assets[0]["sha256"] == "demo_asset_01"
    assert rows(ROOT / "assets/examples/data/assets.jsonl")[0]["source_asset_id"] == "0a648b5b6efa423ab84c13dc77012953"
    assert all(len(asset["views"]) == 4 for asset in assets)
    assert (ROOT / "assets/examples/data/renders_crossview/demo_asset_01/albedo/000.png").is_file()
    assert {view["view_id"] for view in assets[0]["views"]} == {
        "albedo/000", "albedo/003", "albedo/006", "albedo/007"}


def test_real_pairs_are_portable_without_invented_audits_or_holdouts():
    pairs = rows(ROOT / "assets/examples/train/pairs.jsonl")
    assert len(pairs) == 4
    assert len({row["sha256"] for row in pairs}) == 1
    assert {row["split"] for row in pairs} == {"train"}
    assert rows(ROOT / "assets/examples/train/manifests/train.jsonl") == pairs
    for split in ("validation", "test"):
        assert rows(ROOT / f"assets/examples/train/manifests/{split}.jsonl") == []
    for row in pairs:
        assert row["sha256"] == "demo_asset_01"
        assert row["source_asset_id"] == "0a648b5b6efa423ab84c13dc77012953"
        assert row["quality_status"] == "not_audited"
        adapted = adapt_row(row, ROOT / "assets/examples/train", require_subject_mask=False)
        assert Path(adapted["edit_image"]).is_file()
        for stem in ("gt", "degraded"):
            path = ROOT / "assets/examples/train" / row[f"{stem}_relative_path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[f"{stem}_sha256"]
        source = ROOT / "assets/examples/data" / row["source_relative_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_image_sha256"]
        metadata = path.parent / "metadata.json"
        assert metadata.is_file()
        assert json.loads(metadata.read_text())["status"] == "success"
        assert json.loads(metadata.read_text())["sha256"] == row["sha256"]
        assert "/" + "home/" not in metadata.read_text()
        assert "/" + "opt/" not in metadata.read_text()


def test_real_asset_attribution_is_separate_from_render_views():
    license_record = json.loads((ROOT / "assets/examples/asset_license.json").read_text())
    assert license_record["license"] == "CC BY 4.0"
    assert license_record["asset_id"] == "demo_asset_01"
    assert license_record["source_asset_id"] == "0a648b5b6efa423ab84c13dc77012953"
    assert license_record["creator"] == "The Hunt Museum"
    assert license_record["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert set(license_record["rendering"]["selected_views"]) == {"000", "003", "006", "007"}
    assert (ROOT / "assets/examples/ATTRIBUTION.md").is_file()


def test_renamed_historical_pairs_preserve_parameter_sampling_identity():
    from gate3dgen.data.pair_generation.common import DEFAULT_LIBRARY
    from gate3dgen.data.pair_generation.generation import _load_pair_task
    for row in rows(ROOT / "assets/examples/train/pairs.jsonl"):
        metadata = json.loads((ROOT / "assets/examples/train" / row["gt_relative_path"]).with_name("metadata.json").read_text())
        sampling_id = row["parameter_sampling_asset_id"]
        assert sampling_id == row["source_asset_id"]
        assert metadata["parameter_sampling_asset_id"] == sampling_id
        original = {**row, "sha256": sampling_id}
        original.pop("parameter_sampling_asset_id")
        seed = metadata["accepted_attempt_seed"]
        assert _load_pair_task(row, DEFAULT_LIBRARY, seed) == _load_pair_task(original, DEFAULT_LIBRARY, seed)


def test_example_shell_calls_use_real_entry_points_and_explicit_roots(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake_python.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
           "EXAMPLE_OUTPUT_ROOT": "outputs/test-example", "EXAMPLE_GPUS": "2,3"}
    for name in ("EXAMPLE_PAIR_ROOT", "EXAMPLE_PAIR_MANIFEST"):
        env.pop(name, None)
    script = ROOT / "scripts/run_example_data.sh"
    commands = {}
    for action in ("plan", "generate", "audit", "export"):
        result = subprocess.run(["bash", str(script), action], cwd=ROOT, env=env,
                                check=True, text=True, capture_output=True)
        commands[action] = [json.loads(line) for line in result.stdout.splitlines()]
    for action in ("plan", "generate", "audit"):
        for command in commands[action]:
            assert command[command.index("--dataset-root") + 1] == "assets/examples/data"
    assert commands["generate"][-1][commands["generate"][-1].index("--gpus") + 1] == "2,3"
    assert "--vlm-all-pairs" in commands["audit"][0]
    assert "--skip-vision" in commands["audit"][0]
    assert commands["export"][0][:3] == ["-m", "gate3dgen.data.prepare", "dataset"]


def test_evaluation_has_one_real_case_and_original_mesh_reference():
    cases = rows(ROOT / "assets/examples/eval/cases.jsonl")
    assert len(cases) == 1
    case = cases[0]
    assert case["case_id"] == "demo_case_01"
    assert case["asset_id"] == "demo_eval_asset_01"
    assert case["source_case_id"] == "dtcplus_5e236feb89f71086022f__dark"
    assert case["factors"] == ["motion_blur", "uneven_lighting"]
    assert case["historical_outputs"] is True
    assert case["gt"] == "assets/examples/eval/demo_case_01/gt.png"
    plan = json.loads((ROOT / "assets/examples/eval/camera_plan.json").read_text())
    assert {row["case_id"] for row in plan["cases"]} == {row["case_id"] for row in cases}
    for case in cases:
        for field in ("gt", "degraded", "restored", "gt_mesh"):
            path = Path(case[field])
            assert not path.is_absolute() and (ROOT / path).is_file()
        assert len(load_mesh(ROOT / case["gt_mesh"]).faces) > 0
        for variant in ("degraded", "restored_s01"):
            assert (ROOT / "assets/examples/eval/generation/pixal3d" / case["case_id"] / f"{variant}.glb").is_file()
    assert len(plan["cases"][0]["evaluation_views"]) == 4
    assert plan["cases"][0]["source_anchor"]["case_id"] == case["case_id"]
    provenance = json.loads((ROOT / "assets/examples/eval/provenance.json").read_text())
    assert provenance["case_id"] == case["case_id"]
    assert provenance["degradation"]["sir2_used"] is False
    assert provenance["degradation"]["flare7k_used"] is False
    assert len(provenance["files"]) == 6
    for row in provenance["files"]:
        path = ROOT / "assets/examples/eval" / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    gt_record = next(row for row in provenance["files"] if row["path"].endswith("/gt.glb"))
    assert gt_record["sha256"] == "5e236feb89f71086022ff9d8b3e0a38b0b6900d3a6dedbf8b0c7a8e94266da87"
    license_record = json.loads((ROOT / "assets/examples/eval/asset_license.json").read_text())
    assert "additional no-sale" in license_record["license"]
    assert license_record["environment"]["license"] == "CC0 1.0"


