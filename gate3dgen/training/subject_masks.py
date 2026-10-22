"""Portable cache contract for offline GT subject masks."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from PIL import Image


MASK_SCHEMA = "texverse-clean-gt-rmbg2-soft-mask-v1"


def safe_relative_path(value: Any, *, field: str) -> Path:
    """Return one normalized relative path without allowing root escapes."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    text = value.replace("\\", "/")
    path = Path(text)
    windows_absolute = len(text) >= 2 and text[1] == ":"
    if path.is_absolute() or windows_absolute or ".." in path.parts:
        raise ValueError(f"{field} is not a safe relative path: {value!r}")
    return path


def mask_relative_path(gt_relative_path: Any) -> Path:
    """Map a GT relative path to a collision-free cached alpha path."""

    relative = safe_relative_path(gt_relative_path, field="gt_relative_path")
    return relative.with_name(relative.name + ".rmbg2-alpha.png")


def mask_path(mask_root: str | os.PathLike[str], gt_relative_path: Any) -> Path:
    root = Path(mask_root).expanduser().resolve(strict=False)
    candidate = (root / mask_relative_path(gt_relative_path)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:  # pragma: no cover - guarded above as defense in depth
        raise ValueError(f"subject mask path escapes root: {candidate}") from error
    return candidate


def validate_mask(path: Path, expected_size: tuple[int, int]) -> None:
    """Validate one cached 8-bit soft alpha without changing it."""

    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"subject mask must be mode L, got {image.mode}: {path}")
        if tuple(image.size) != tuple(expected_size):
            raise ValueError(
                f"subject mask size {tuple(image.size)} != GT size {tuple(expected_size)}: {path}"
            )
        image.verify()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "MASK_SCHEMA",
    "mask_path",
    "mask_relative_path",
    "safe_relative_path",
    "sha256_file",
    "validate_mask",
]
