from __future__ import annotations

import sys

import pytest

from gate3dgen.evaluation import blender_runtime as runtime


def test_native_numpy_needs_no_bootstrap(monkeypatch):
    monkeypatch.setattr(runtime, "probe", lambda *args: {"python": [3, 10], "numpy_import_ok": True})
    assert runtime.ensure_runtime("blender") is None


def test_current_python_environment_blender_precedes_system_path(monkeypatch, tmp_path):
    monkeypatch.delenv("GATE3DGEN_BLENDER", raising=False)
    monkeypatch.setattr(runtime.sys, "executable", str(tmp_path / "python"))
    assert runtime.default_blender() == "blender"
    sibling = tmp_path / "blender"
    sibling.touch()
    assert runtime.default_blender() == str(sibling)
    monkeypatch.setenv("GATE3DGEN_BLENDER", "/explicit/blender")
    assert runtime.default_blender() == "/explicit/blender"


def test_color_management_fallback_blocks_rendering(monkeypatch):
    monkeypatch.setattr(runtime, "probe", lambda *args: {
        "python": [3, 10], "numpy_import_ok": True, "color_management_ok": False})
    with pytest.raises(RuntimeError, match="OCIO/color-management"):
        runtime.ensure_runtime("blender")


def test_matching_cached_color_config_is_shared_with_renderer(monkeypatch, tmp_path):
    config = tmp_path / "share/gate3dgen/blender-color-3.0.1/config.ocio"
    config.parent.mkdir(parents=True)
    config.write_text("fixture")
    monkeypatch.setattr(runtime.sys, "prefix", str(tmp_path))
    monkeypatch.delenv("OCIO", raising=False)
    assert runtime.runtime_environment()["OCIO"] == str(config)
    monkeypatch.setenv("OCIO", "/explicit/config.ocio")
    assert runtime.runtime_environment()["OCIO"] == "/explicit/config.ocio"


def test_missing_numpy_gives_setup_instruction(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "probe", lambda *args: {"python": [3, 10], "numpy_import_ok": False})
    with pytest.raises(RuntimeError, match="blender_runtime --install"):
        runtime.ensure_runtime("blender", runtime_path=tmp_path)


def test_bootstrap_downloads_blender_abi_not_host_abi(monkeypatch, tmp_path):
    answers = iter([False, False, True])
    monkeypatch.setattr(runtime, "probe", lambda *args: {"python": [3, 10], "numpy_import_ok": next(answers)})
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    commands = []
    monkeypatch.setattr(runtime.subprocess, "run", lambda command, **kwargs: commands.append(command))
    assert runtime.ensure_runtime("blender", install=True, runtime_path=tmp_path) == tmp_path
    command = commands[0]
    assert command[0] == sys.executable
    assert command[command.index("--python-version") + 1] == "3.10"
    assert command[command.index("--abi") + 1] == "cp310"
    assert command[command.index("--target") + 1] == str(tmp_path)
    assert command[-1] == "numpy==1.26.4"


def test_existing_abi_target_is_reused(monkeypatch, tmp_path):
    answers = iter([False, True])
    monkeypatch.setattr(runtime, "probe", lambda *args: {"python": [3, 10], "numpy_import_ok": next(answers)})
    assert runtime.ensure_runtime("blender", runtime_path=tmp_path) == tmp_path
