from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import finite_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path | None) -> str | None:
    if path is None or not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-c", f"safe.directory={path.resolve()}", "-C", str(path), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    # Shared repositories may be owned by a service account and
    # rejected by older Git safe.directory handling. Read HEAD without changing
    # global Git configuration so provenance still pins the exact source tree.
    git_dir = path / ".git"
    if git_dir.is_file():
        value = git_dir.read_text(encoding="utf-8").strip()
        if value.startswith("gitdir:"):
            candidate = Path(value.split(":", 1)[1].strip())
            git_dir = candidate if candidate.is_absolute() else (path / candidate)
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref:"):
        return value if len(value) == 40 else None
    reference = value.split(":", 1)[1].strip()
    loose = git_dir / reference
    if loose.is_file():
        commit = loose.read_text(encoding="utf-8").strip()
        return commit if len(commit) == 40 else None
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    return None


def package_versions() -> dict[str, str]:
    output = {}
    for name in ("torch", "numpy", "scipy", "scikit-image", "trimesh", "transformers", "lpips", "huggingface-hub"):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return output


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def write_run_metadata(
    destination: Path,
    *,
    component: str,
    arguments: dict[str, Any],
    manifest: Path,
    repo: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = {
        "schema_version": "texverse-eval-run-v1",
        "component": component,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "arguments": _serializable(arguments),
        "manifest": {"path": str(manifest.resolve()), "sha256": sha256_file(manifest)},
        "repo": {"path": str(repo.resolve()), "commit": git_commit(repo)} if repo else None,
        "packages": package_versions(),
        "extra": _serializable(extra or {}),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(finite_json(metadata), indent=2, sort_keys=True), encoding="utf-8")
