from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).expanduser().resolve()
_PACKAGED_PROMPTS = PACKAGE_DIR.parents[1] / "resources/prompts"
PAIR_ROOT = Path(os.environ.get("GATE3DGEN_PAIR_ROOT", PROJECT_ROOT / "outputs/pairs"))
DATASET_ROOT = Path(os.environ.get("GATE3DGEN_DATASET_ROOT", PROJECT_ROOT / "data/TexVerse"))
OUTPUT_ROOT = Path(os.environ.get("GATE3DGEN_AUDIT_ROOT", PROJECT_ROOT / "outputs/audit"))
PROMPT_LIBRARY = Path(
    os.environ.get(
        "GATE3DGEN_PROMPT_LIBRARY",
        _PACKAGED_PROMPTS / "gate3dgen_pair_synthesis.json",
    )
)
DINO_REPO = Path(os.environ.get("GATE3DGEN_DINO_REPO", PROJECT_ROOT / "third_party/dinov2"))
DINO_CHECKPOINT = Path(
    os.environ.get("GATE3DGEN_DINO_CHECKPOINT", PROJECT_ROOT / "checkpoints/dinov2_vitb14.pth")
)
SAM_REPO = Path(os.environ.get("GATE3DGEN_SAM_REPO", PROJECT_ROOT / "third_party/sam2"))
SAM_CHECKPOINT = Path(
    os.environ.get("GATE3DGEN_SAM_CHECKPOINT", PROJECT_ROOT / "checkpoints/sam2.1_hiera_small.pt")
)
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
QWEN_MODEL = Path(
    os.environ.get("GATE3DGEN_QWEN_MODEL", PROJECT_ROOT / "checkpoints/Qwen3-VL-32B-Instruct")
)
QWEN_RUNNER = Path(
    os.environ.get("GATE3DGEN_QWEN_RUNNER", PACKAGE_DIR / "qwen_runner.py")
)
THRESHOLDS = PACKAGE_DIR / "thresholds_v2.json"
ADJUDICATION_OVERRIDES = PACKAGE_DIR / "adjudication_overrides_v1.json"


def is_crop_related(row: dict[str, Any]) -> bool:
    """Return whether a single or composed degradation contains a crop factor."""
    fields = (
        "task_id", "degradation_type", "audit_task_id", "audit_degradation_type",
        "factors", "problems", "degradations", "components", "component_tasks",
        "combination_id",
    )

    def contains_crop(value: Any) -> bool:
        if isinstance(value, str):
            return "crop" in {part for part in re.split(r"[^a-z0-9]+", value.lower()) if part}
        if isinstance(value, (list, tuple, set)):
            return any(contains_crop(item) for item in value)
        if isinstance(value, dict):
            return any(contains_crop(item) for item in value.values())
        return False

    return any(contains_crop(row.get(field)) for field in fields)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from exc


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def input_fingerprint(row: dict[str, Any], *, stage: str, extra: Any = None) -> str:
    """Stable resume key for the manifest row and the stage configuration."""
    payload = {"stage": stage, "row": row, "extra": extra}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def ensure_output_within_project(path: Path) -> Path:
    """Keep generated audit artifacts inside the repository by default."""
    resolved = path.expanduser().resolve()
    project = PROJECT_ROOT.resolve()
    try:
        resolved.relative_to(project)
    except ValueError as exc:
        raise ValueError(f"output root must stay under {project}, got {resolved}") from exc
    return resolved


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def atomic_write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(canonical_json(value) + "\n" for value in values))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_thresholds(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def record_path(output_root: Path, stage: str, pair_id: str) -> Path:
    return output_root / "records" / stage / pair_id[:2] / f"{pair_id}.json"


def read_record(output_root: Path, stage: str, pair_id: str) -> dict[str, Any] | None:
    try:
        value = json.loads(record_path(output_root, stage, pair_id).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def resolve_pair_path(row: dict[str, Any], pair_root: Path, stem: str) -> Path:
    relative = row.get(f"{stem}_relative_path")
    if relative:
        candidate = pair_root / str(relative)
        if candidate.is_file():
            return candidate
    absolute = Path(str(row.get(f"{stem}_path", "")))
    if absolute.is_file():
        return absolute
    if relative:
        return pair_root / str(relative)
    raise FileNotFoundError(f"No {stem} path for {row.get('pair_id')}")


def resolve_source_path(row: dict[str, Any], dataset_root: Path) -> Path:
    relative = row.get("source_relative_path")
    if relative:
        candidate = dataset_root / str(relative)
        if candidate.is_file():
            return candidate
    for key in ("source_path", "source_original_path"):
        candidate = Path(str(row.get(key, "")))
        if candidate.is_file():
            return candidate
    return dataset_root / str(relative or "missing")


def resolve_metadata_path(row: dict[str, Any], pair_root: Path) -> Path:
    try:
        return resolve_pair_path(row, pair_root, "gt").parent / "metadata.json"
    except FileNotFoundError:
        candidate = Path(str(row.get("metadata_path", "")))
        return candidate


def stage_rows(manifest: Path, shard_index: int, shard_count: int, limit: int | None = None):
    emitted = 0
    for index, row in enumerate(iter_jsonl(manifest)):
        if index % shard_count != shard_index:
            continue
        yield row
        emitted += 1
        if limit is not None and emitted >= limit:
            break
