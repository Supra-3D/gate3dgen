"""Public setup and before/after reconstruction entry points."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

from gate3dgen.evaluation import reconstruct

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("external", [False, True])
def test_setup_downloads_both_official_models(tmp_path, monkeypatch, external):
    executable = tmp_path / "conda"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$SETUP_TEST_LOG"\nexit 0\n')
    executable.chmod(0o755)
    log = tmp_path / "commands.txt"
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SETUP_TEST_LOG", str(log))
    destination = tmp_path / "external-models" if external else ROOT / "checkpoints/upstream"
    if external:
        monkeypatch.setenv("GATE3DGEN_UPSTREAM_ROOT", str(destination))
    else:
        monkeypatch.delenv("GATE3DGEN_UPSTREAM_ROOT", raising=False)
    result = subprocess.run(["bash", str(ROOT / "setup/setup.sh")], cwd=ROOT,
                            check=True, capture_output=True, text=True)
    assert f"python setup/download_models.py all --output-root {destination}" in log.read_text()
    assert "Setup complete" in result.stdout
    if external:
        assert not destination.exists()


def load_setup(name):
    spec = importlib.util.spec_from_file_location(f"setup_{name}", ROOT / "setup" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_weights_check_requires_complete_package(tmp_path):
    check = load_setup("check")
    router, purifier, runtime = tmp_path / "router", tmp_path / "purifier", tmp_path / "runtime"
    contents = {
        router / "checkpoint_complete.json": {"status": "complete", "global_step": 6000},
        router / "exported_model/model.safetensors.index.json": {"weight_map": {"w": "model.safetensors"}},
        purifier / "checkpoint_complete.json": {"complete": True, "global_step": 15000,
                                                 "condition_checkpoint_step": 6000, "condition_role": "B"},
        runtime / "runtime_manifest.json": {"complete": True},
    }
    for path, content in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))
    for path in (router / "exported_model/config.json", router / "exported_model/model.safetensors",
                 purifier / "config.json", purifier / "pytorch_model_fsdp.bin",
                 runtime / "processor/preprocessor_config.json", runtime / "processor/tokenizer_config.json",
                 runtime / "processor/tokenizer.json", runtime / "vae/config.json",
                 runtime / "vae/diffusion_pytorch_model.safetensors", runtime / "scheduler/scheduler_config.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    assert check.validate_checkpoints(router, purifier, runtime)["status"] == "ready"
    (runtime / "vae/config.json").unlink()
    with pytest.raises(FileNotFoundError):
        check.validate_checkpoints(router, purifier, runtime)


def test_official_downloads_default_inside_and_allow_external_tests(tmp_path, monkeypatch):
    import huggingface_hub
    download = load_setup("download_models")
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: calls.append(kwargs))
    assert download.main(["all", "--output-root", str(tmp_path)]) == 0
    assert [call["repo_id"] for call in calls] == [
        "FireRedTeam/FireRed-Image-Edit-1.1", "TencentARC/Pixal3D",
    ]
    assert all(set(call) == {"repo_id", "local_dir"} for call in calls)  # no partial-download filters
    assert all(call["local_dir"].parent == tmp_path for call in calls)
    calls.clear()
    assert download.main(["all"]) == 0
    assert all(call["local_dir"].parent == ROOT / "checkpoints/upstream" for call in calls)
    with pytest.raises(SystemExit):
        download.main(["firered", "--output-root", str(ROOT / "checkpoints")])


def test_real_before_after_conditions():
    rows = reconstruct.condition_rows(ROOT / "assets/examples/eval/cases.jsonl", ROOT)
    assert len(rows) == 1 and rows[0]["case_id"] == "demo_case_01"
    assert rows[0]["degraded"].endswith("demo_case_01/degraded.png")
    assert rows[0]["restored_s01"].endswith("demo_case_01/restored.png")
    assert rows[0]["gt_mesh"].endswith("demo_case_01/gt.glb")
    assert all(item["available"] for item in rows[0]["conditions"])


def test_demo_conditions_do_not_require_ground_truth(tmp_path):
    before, after = tmp_path / "input.png", tmp_path / "processed.png"
    before.write_bytes(b"fixture")
    after.write_bytes(b"fixture")
    manifest = tmp_path / "demo.jsonl"
    manifest.write_text(json.dumps({"case_id": "demo_01", "dataset_kind": "wild_no_gt",
                                    "original": str(before), "degraded": str(before),
                                    "restored": str(after)}) + "\n")
    row = reconstruct.condition_rows(manifest, tmp_path)[0]
    assert row["gt_mesh"] is None and row["clean_gt"] is None
    assert row["original"] == str(before) and row["restored_s01"] == str(after)


def test_generation_dry_run_does_not_create_output(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "pixal"
    repo.mkdir()
    (repo / "inference.py").write_text("# fixture")
    monkeypatch.setenv("PIXAL3D_REPO", str(repo))
    monkeypatch.chdir(ROOT)
    output = tmp_path / "output"
    assert reconstruct.main(["--output", str(output), "--dry-run"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["variants"] == ["degraded", "restored_s01"]
    assert record["command"].count("--variant") == 2
    if "PIXAL3D_MODEL" not in reconstruct.os.environ:
        assert record["command"][record["command"].index("--model") + 1] == str(ROOT / "checkpoints/upstream/Pixal3D")
    assert not output.exists()


def test_generation_writes_matched_conditions(tmp_path, monkeypatch):
    repo, model = tmp_path / "pixal", tmp_path / "model"
    repo.mkdir()
    model.mkdir()
    (repo / "inference.py").write_text("# fixture")
    monkeypatch.setenv("PIXAL3D_REPO", str(repo))
    monkeypatch.setenv("PIXAL3D_MODEL", str(model))
    monkeypatch.chdir(ROOT)
    commands = []
    monkeypatch.setattr(reconstruct.subprocess, "run", lambda command, **kwargs: commands.append(command))
    output = tmp_path / "output"
    assert reconstruct.main(["--output", str(output)]) == 0
    row = json.loads((output / "conditions.jsonl").read_text())
    assert row["restored"] == row["restored_s01"]
    assert commands[0][-1] == "--no-renders"
    with pytest.raises(SystemExit):
        reconstruct.main(["--output", str(output)])


def test_eval_defaults_use_retained_pixal3d_meshes():
    script = (ROOT / "scripts/evaluate.sh").read_text()
    assert "assets/examples/eval/generation" in script
    assert "--baseline pixal3d" in script
    assert "--variant degraded --variant restored_s01" in script
    assert "--paper-protocol" in script
    for entry in ("metrics2d", "metrics3d_official", "render_pbr", "pbr_metrics"):
        assert f"gate3dgen.evaluation.{entry}" in script


def test_environment_paths_are_relative_to_setup():
    import yaml
    environment = yaml.safe_load((ROOT / "setup/environment.yml").read_text())
    pip = next(item["pip"] for item in environment["dependencies"] if isinstance(item, dict))
    assert "-c constraints-host.txt" in pip
    assert "-e ..[train,eval,audit,dev]" in pip
    assert (ROOT / "setup/constraints-host.txt").is_file()
