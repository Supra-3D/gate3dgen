"""Core release downloads must work anonymously and reject unsafe destinations."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def downloader():
    spec = importlib.util.spec_from_file_location("release_download_weights", ROOT / "setup/download_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_plan_is_read_only(tmp_path, capsys):
    target = tmp_path / "weights"
    assert downloader().main(["--dry-run", "--output-root", str(target)]) == 0
    assert not target.exists()
    assert "anonymous" in capsys.readouterr().out


def test_download_rejects_source_overwrite():
    for target in [ROOT, ROOT / "gate3dgen"]:
        with pytest.raises(SystemExit):
            downloader().main(["--dry-run", "--output-root", str(target)])


def test_partial_remote_release_fails_readiness(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(FileNotFoundError):
        downloader().main(["--output-root", str(tmp_path / "weights")])
    assert calls[0]["token"] is False
