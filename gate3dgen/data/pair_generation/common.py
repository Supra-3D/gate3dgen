from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).expanduser().resolve()
_PACKAGED_PROMPTS = PACKAGE_DIR.parents[1] / "resources/prompts"
DEFAULT_INPUT = Path(os.environ.get("GATE3DGEN_ASSET_MANIFEST", PROJECT_ROOT / "data/assets.jsonl"))
DEFAULT_LIBRARY = Path(
    os.environ.get(
        "GATE3DGEN_PROMPT_LIBRARY",
        _PACKAGED_PROMPTS / "gate3dgen_pair_synthesis.json",
    )
)
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("GATE3DGEN_DATASET_ROOT", PROJECT_ROOT / "data/TexVerse")
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("GATE3DGEN_PAIR_ROOT", PROJECT_ROOT / "outputs/pairs")
)
DEFAULT_MODEL_PATH = Path(
    os.environ.get("GATE3DGEN_MODEL_PATH", PROJECT_ROOT / "models/FireRed-Image-Edit-1.1")
)


def stable_digest(*parts: object) -> bytes:
    return hashlib.sha256(":".join(str(x) for x in parts).encode("utf-8")).digest()


def stable_int(*parts: object, bits: int = 63) -> int:
    value = int.from_bytes(stable_digest(*parts)[:8], "big")
    return value & ((1 << bits) - 1)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2 if pretty else None)
    atomic_write_text(path, text + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_relative_source(path: Path) -> str:
    parts = path.parts
    if "renders_crossview" not in parts:
        raise ValueError(f"Source is not under renders_crossview: {path}")
    index = parts.index("renders_crossview")
    return Path(*parts[index:]).as_posix()


def resolve_source(record: dict[str, Any], dataset_root: Path) -> Path:
    remapped = dataset_root / record["source_relative_path"]
    if remapped.is_file():
        return remapped
    original = Path(record["source_original_path"])
    if original.is_file():
        return original
    return remapped


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
