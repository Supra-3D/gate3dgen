from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gate3dgen.data.pair_generation import common as generation_common
from gate3dgen.data.quality_audit import common as audit_common
from gate3dgen.training.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_external_training_output_requires_explicit_allowed_root(monkeypatch, tmp_path):
    output = tmp_path / "external_outputs" / "purifier"
    monkeypatch.setenv("GATE3DGEN_OUTPUT_ROOT", str(output))
    monkeypatch.delenv("GATE3DGEN_ALLOWED_OUTPUT_ROOT", raising=False)
    with pytest.raises(ValueError, match="model output_dir must be inside"):
        load_config("full", str(ROOT / "configs/train/purifier.yaml"))
    monkeypatch.setenv("GATE3DGEN_ALLOWED_OUTPUT_ROOT", str(output.parent))
    config = load_config("full", str(ROOT / "configs/train/purifier.yaml"))
    assert Path(config["paths"]["output_dir"]) == output


def test_external_allowed_root_cannot_be_filesystem_or_repository_root(monkeypatch):
    for allowed in ("/", str(ROOT)):
        monkeypatch.setenv("GATE3DGEN_ALLOWED_OUTPUT_ROOT", allowed)
        with pytest.raises(ValueError, match="dedicated run directory"):
            load_config("full", str(ROOT / "configs/train/purifier.yaml"))


def test_default_paths_are_repository_relative() -> None:
    assert generation_common.PROJECT_ROOT == ROOT
    assert audit_common.PROJECT_ROOT == ROOT
    assert generation_common.DEFAULT_OUTPUT_ROOT == ROOT / "outputs/pairs"
    assert audit_common.OUTPUT_ROOT == ROOT / "outputs/audit"


def test_public_files_do_not_embed_machine_or_service_paths() -> None:
    forbidden = ("/" + "home/", "/" + "opt/", "wise" + "mlops", "model" + "arts")
    source_dirs = ("gate3dgen", "scripts", "configs", "docs", "tests", "assets")
    paths = [path for folder in source_dirs for path in (ROOT / folder).rglob("*")
             if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md", ".sh"}]
    paths += [path for path in ROOT.glob("*") if path.is_file() and path.suffix in {".md", ".yml"}]
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token.lower() not in text.lower(), f"forbidden path/platform token in {path}"

    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)


def test_released_training_config_resolves_inside_repository(monkeypatch) -> None:
    monkeypatch.setenv("GATE3DGEN_MODEL_PATH", str(ROOT.parent / "external-base"))
    config = load_config("full", str(ROOT / "configs/train/purifier.yaml"))
    assert Path(config["model"]["pretrained_model_path"]) == ROOT.parent / "external-base"
    assert Path(config["paths"]["cache_root"]) == ROOT / "outputs/cache/purifier"
    assert Path(config["paths"]["output_dir"]) == ROOT / "outputs/training/purifier"
    assert Path(config["condition_model"]["processor_path"]) == ROOT.parent / "external-base/processor"


def test_large_runtime_artifacts_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in ("data/", "outputs/", "checkpoints/*", "third_party/"):
        assert entry in ignore


def test_prompt_names_and_internal_ids_are_gate3dgen_owned() -> None:
    prompt_root = ROOT / "gate3dgen/resources/prompts"
    expected = {
        "gate3dgen_pair_synthesis.json": "gate3dgen-pair-synthesis-1.0",
        "purifier.json": "gate3dgen-purifier-1.0",
        "router.json": "gate3dgen-router-1.0",
    }
    assert set(expected) <= {path.name for path in prompt_root.glob("*.json")}
    assert not (ROOT / "prompts").exists()
    for name, version in expected.items():
        public = json.loads((prompt_root / name).read_text(encoding="utf-8"))
        assert public["version"] == version
        assert "albedo_edit_prompt_library" not in json.dumps(public)

    restoration = json.loads(
        (prompt_root / "purifier.json").read_text(encoding="utf-8")
    )
    assert restoration["recommended_default_prompt_id"] == "gate3dgen_purify_input_v1"
    assert "gate3dgen_purify_input_v1" in restoration["restoration_prompts"]
