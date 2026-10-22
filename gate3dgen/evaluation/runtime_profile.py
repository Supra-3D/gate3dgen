"""Small, dependency-optional runtime and memory profiler.

The profiler is intentionally separate from model inference.  A generation
wrapper can call :func:`profile_call` around its already-loaded pipeline and
store the returned record.  This keeps the evaluation runner usable on CPU,
on machines without CUDA, and in unit tests.

``cached`` records are retained for provenance but are excluded by
:func:`aggregate_runtime` from timing means and percentiles: a resumed job is
not a fresh inference.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .io import finite_json, iter_jsonl, write_jsonl


def _resolution(value: Any, dimensions: int = 2) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return [value] * dimensions
    if isinstance(value, str):
        tokens = value.lower().replace("x", " ").replace(",", " ").split()
        if len(tokens) == 1:
            return [int(tokens[0])] * dimensions
        if len(tokens) != dimensions:
            raise ValueError(f"resolution must have {dimensions} dimensions, got {value!r}")
        return [int(token) for token in tokens]
    values = list(value)
    if len(values) == 1:
        values *= dimensions
    if len(values) != dimensions:
        raise ValueError(f"resolution must be scalar or have {dimensions} dimensions, got {value!r}")
    return [int(value) for value in values]


def _torch_cuda(device: str | int | None) -> tuple[Any | None, int | None]:
    """Return torch and a CUDA index without importing torch at module load."""

    try:
        import torch
    except Exception:
        return None, None
    if not torch.cuda.is_available():
        return torch, None
    if device is None:
        index = torch.cuda.current_device()
    elif isinstance(device, int) or str(device).isdigit():
        index = int(device)
    else:
        index = torch.device(str(device)).index
        index = torch.cuda.current_device() if index is None else index
    return torch, index


def _sync(torch: Any | None, index: int | None) -> None:
    if torch is not None and index is not None and torch.cuda.is_available():
        torch.cuda.synchronize(index)


def _memory(torch: Any | None, index: int | None) -> tuple[int | None, int | None]:
    if torch is None or index is None or not torch.cuda.is_available():
        return None, None
    return int(torch.cuda.max_memory_allocated(index)), int(torch.cuda.max_memory_reserved(index))


def _reset_memory(torch: Any | None, index: int | None) -> None:
    if torch is not None and index is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(index)


class RuntimeProfiler:
    """Measure synchronized wall time and CUDA peak memory for one call."""

    def __init__(self, device: str | int | None = None):
        self.device = device
        self._torch, self._cuda_index = _torch_cuda(device)

    @contextmanager
    def measure(
        self,
        *,
        case_id: str | None = None,
        baseline: str | None = None,
        variant: str | None = None,
        steps: int | None = None,
        input_resolution: Any = None,
        requested_resolution: Any = None,
        resolution_dimensions: int = 2,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield a mutable record and fill its timing fields on exit."""

        _sync(self._torch, self._cuda_index)
        _reset_memory(self._torch, self._cuda_index)
        start = time.perf_counter()
        record: dict[str, Any] = {
            "status": "running",
            "cached": False,
            "case_id": case_id,
            "baseline": baseline,
            "variant": variant,
            "steps": steps,
            "input_resolution": _resolution(input_resolution),
            "requested_resolution": _resolution(requested_resolution, dimensions=resolution_dimensions),
            "cuda_device": self._cuda_index,
            **(dict(metadata) if metadata else {}),
        }
        try:
            yield record
        except Exception as error:
            _sync(self._torch, self._cuda_index)
            record.update({
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            })
            allocated, reserved = _memory(self._torch, self._cuda_index)
            record.update({"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved})
            raise
        else:
            _sync(self._torch, self._cuda_index)
            allocated, reserved = _memory(self._torch, self._cuda_index)
            record.update({
                "status": "ok",
                "elapsed_ms": (time.perf_counter() - start) * 1000.0,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
            })


def profile_call(
    function: Callable[[], Any],
    *,
    profiler: RuntimeProfiler | None = None,
    **metadata: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run ``function`` once and return ``(result, runtime_record)``."""

    profiler = profiler or RuntimeProfiler(metadata.pop("device", None))
    with profiler.measure(**metadata) as record:
        result = function()
    return result, record


def mark_cached(
    record: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    reason: str = "existing_output",
) -> dict[str, Any]:
    """Create a provenance record for a skipped/resumed output."""

    value = dict(record)
    value.update({
        "status": "cached",
        "cached": True,
        "elapsed_ms": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "cache_reason": reason,
    })
    if output_path is not None:
        value["output_path"] = str(Path(output_path).absolute())
    return value


def stage_record(
    *,
    name: str,
    elapsed_ms: float | None = None,
    status: str = "ok",
    requested_resolution: Any = None,
    realized_resolution: Any = None,
    peak_allocated_bytes: int | None = None,
    peak_reserved_bytes: int | None = None,
    error: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Normalize a model stage result for inclusion in a 3D runtime record."""

    result: dict[str, Any] = {
        "name": name,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "requested_resolution": _resolution(requested_resolution, dimensions=3),
        "realized_resolution": _resolution(realized_resolution, dimensions=3),
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        **metadata,
    }
    if error:
        result["error"] = error
    return result


def generation_record(
    *,
    case_id: str,
    baseline: str,
    variant: str,
    status: str,
    total_elapsed_ms: float | None = None,
    requested_resolution: Any = None,
    realized_resolution: Any = None,
    peak_allocated_bytes: int | None = None,
    peak_reserved_bytes: int | None = None,
    glb_path: str | Path | None = None,
    mesh_stats: Mapping[str, Any] | None = None,
    stages: Sequence[Mapping[str, Any]] = (),
    cached: bool = False,
    error: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Build the standard Pixal3D/TRELLIS.2 generation provenance record."""

    glb_bytes = None
    if glb_path and Path(glb_path).is_file():
        glb_bytes = int(Path(glb_path).stat().st_size)
    result: dict[str, Any] = {
        "schema_version": "official_aligned_v1",
        "component": "image_to_3d_runtime",
        "case_id": case_id,
        "baseline": baseline,
        "variant": variant,
        "status": "cached" if cached else status,
        "cached": cached,
        "total_elapsed_ms": None if cached else total_elapsed_ms,
        "requested_resolution": _resolution(requested_resolution, dimensions=3),
        "realized_resolution": _resolution(realized_resolution, dimensions=3),
        "peak_allocated_bytes": None if cached else peak_allocated_bytes,
        "peak_reserved_bytes": None if cached else peak_reserved_bytes,
        "glb_path": str(Path(glb_path).absolute()) if glb_path else None,
        "glb_bytes": glb_bytes,
        "mesh_stats": dict(mesh_stats or {}),
        "stages": [dict(stage) for stage in stages],
        **metadata,
    }
    if error:
        result["error"] = error
    return result


def _numeric(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and value == value]


def aggregate_runtime(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate fresh successful measurements; cached rows never affect means."""

    groups: dict[str, list[Mapping[str, Any]]] = {}
    all_rows = [dict(row) for row in rows]
    for row in all_rows:
        key = ":".join(str(row.get(name, "")) for name in ("baseline", "variant", "steps", "requested_resolution"))
        groups.setdefault(key, []).append(row)
    output: dict[str, Any] = {}
    for key, group in sorted(groups.items()):
        fresh = [
            row for row in group
            if not row.get("cached") and row.get("status") == "ok"
        ]
        durations = _numeric(row.get("elapsed_ms", row.get("total_elapsed_ms")) for row in fresh)
        allocated = _numeric(row.get("peak_allocated_bytes") for row in fresh)
        reserved = _numeric(row.get("peak_reserved_bytes") for row in fresh)
        item: dict[str, Any] = {
            "records": len(group),
            "fresh_records": len(fresh),
            "cached_records": sum(bool(row.get("cached")) for row in group),
            "error_records": sum(row.get("status") == "error" for row in group),
            "mean_elapsed_ms": statistics.fmean(durations) if durations else None,
            "median_elapsed_ms": statistics.median(durations) if durations else None,
            "p95_elapsed_ms": _percentile(durations, 0.95),
            "max_peak_allocated_bytes": max(allocated) if allocated else None,
            "max_peak_reserved_bytes": max(reserved) if reserved else None,
        }
        output[key] = item
    return {"schema_version": "official_aligned_v1", "groups": output}


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL runtime records")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate_runtime(iter_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finite_json(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(finite_json(summary), indent=2))


if __name__ == "__main__":
    main()
