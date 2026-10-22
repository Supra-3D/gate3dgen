from __future__ import annotations

import json
import os
import random
import signal
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .common import (
    PACKAGE_DIR, atomic_write_json, file_sha256, iter_jsonl,
    resolve_source, stable_int,
)
from .albedo_pair_backend import (
    _protect_gt_background, _protected_background_max_diff, load_tasks,
)
from .degradation_backend import (
    apply_postprocess, load_pipeline, release_cuda_memory,
)

STOP_REQUESTED = False


def pipeline_code_sha256() -> str:
    """Hash the implementation files recorded by every generated pair."""
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pair_output_dir(output_root: Path, row: dict[str, Any]) -> Path:
    return output_root / "pairs" / row["task_id"] / row["sha256"][:2] / row["pair_id"]


def _atomic_save_image(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    try:
        image.convert("RGB").save(temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            os.fsync(directory_fd)
            os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _call_pipe(pipe, image, prompt: str, negative: str, steps: int, cfg_scale: float):
    return pipe(
        image=[image], prompt=prompt, negative_prompt=negative or " ",
        num_inference_steps=steps, true_cfg_scale=cfg_scale,
        guidance_scale=1.0, num_images_per_prompt=1,
    ).images[0].convert("RGB")


def _load_pair_task(row: dict[str, Any], prompt_library: Path, attempt_seed: int):
    # A display-ID rename must not change a historical pair's sampling key.
    # Ordinary generated plans keep using their own asset ID by default.
    sampling_id = row.get("parameter_sampling_asset_id", row["sha256"])
    tasks, negative = load_tasks(prompt_library, attempt_seed, sampling_id)
    task = tasks.get(row["task_id"])
    if task is None:
        raise KeyError(f"Task {row['task_id']} is not available in {prompt_library}")
    if task["problem_type"] != row["degradation_type"]:
        raise ValueError(f"Task type mismatch for {row['pair_id']}")
    return task, negative


def _validate_rgb(path: Path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        array = np.asarray(rgb)
        if array.size == 0 or rgb.width <= 0 or rgb.height <= 0:
            raise ValueError(f"Empty image: {path}")
        return rgb.copy(), array


def basic_audit(
    row: dict[str, Any], task: dict[str, Any], source_path: Path,
    gt_path: Path, degraded_path: Path, attempt_seed: int,
) -> dict[str, Any]:
    import numpy as np
    source, _ = _validate_rgb(source_path)
    gt, _ = _validate_rgb(gt_path)
    degraded, degraded_array = _validate_rgb(degraded_path)
    if gt.size != degraded.size:
        raise ValueError(f"GT/degraded size mismatch: {gt.size} != {degraded.size}")
    result: dict[str, Any] = {
        "status": "pass",
        "dimensions": list(gt.size),
        "gt_sha256": file_sha256(gt_path),
        "degraded_sha256": file_sha256(degraded_path),
    }
    if task["method"] == "hybrid":
        replay, _, _ = apply_postprocess(gt, task, attempt_seed, source=gt)
        replay_array = np.asarray(replay.convert("RGB"))
        if replay_array.shape != degraded_array.shape or not np.array_equal(replay_array, degraded_array):
            difference = np.abs(replay_array.astype(np.int16) - degraded_array.astype(np.int16))
            raise ValueError(f"Procedural replay mismatch; max_abs_diff={int(difference.max())}")
        result["procedural_replay"] = "pixel_exact"
    if task["affected_scope"] == "object_only":
        difference = _protected_background_max_diff(gt, degraded, source)
        if difference != 0:
            raise ValueError(f"Protected background changed; max_abs_diff={difference}")
        result["protected_background_max_abs_diff"] = difference
    return result


def _read_existing_metadata(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def process_pair(
    row: dict[str, Any], pipe, prompt_library: Path, dataset_root: Path,
    output_root: Path, *, model_path: Path, steps: int, cfg_scale: float,
    max_attempts: int, gpu_label: str, source_hash_cache: dict[str, str],
    code_sha256: str,
) -> tuple[str, Any]:
    directory = pair_output_dir(output_root, row)
    metadata_path = directory / "metadata.json"
    gt_path = directory / "gt.png"
    degraded_path = directory / "degraded.png"
    mask_path = directory / "affected_mask.png"
    existing = _read_existing_metadata(metadata_path)
    recovery_event = None
    if existing and existing.get("pair_id") == row["pair_id"]:
        if existing.get("status") == "success":
            try:
                task, _ = _load_pair_task(row, prompt_library, int(existing["accepted_attempt_seed"]))
                basic_audit(row, task, resolve_source(row, dataset_root), gt_path, degraded_path, int(existing["accepted_attempt_seed"]))
                return "skipped_success", pipe
            except Exception as exc:
                # A completed record whose files no longer pass validation is not
                # terminal. Preserve its history for audit, then grant a fresh set
                # of deterministic attempts so corruption can be repaired.
                recovery_event = {
                    "detected_at": utc_now(),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "previous_attempts": existing.get("attempts", []),
                }
                existing = None
        if existing and existing.get("status") == "reject" and len(existing.get("attempts", [])) >= max_attempts:
            return "skipped_reject", pipe
    if pipe is None:
        pipe = load_pipeline(model_path, "cuda")
    attempts = list((existing or {}).get("attempts", []))
    start_attempt = min(len(attempts), max_attempts)
    source_path = resolve_source(row, dataset_root)
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source image: {source_path}")
    source_key = str(source_path)
    if source_key not in source_hash_cache:
        source_hash_cache[source_key] = file_sha256(source_path)
    from PIL import Image
    source = Image.open(source_path).convert("RGB")
    base_record = {
        **row,
        "model": "firered-image-edit-1.1",
        "model_path": str(model_path.resolve()),
        "prompt_library": str(prompt_library.resolve()),
        "source_path": str(source_path.absolute()),
        "source_image_sha256": source_hash_cache[source_key],
        "gt_output": str(gt_path.resolve()),
        "gt_relative_path": gt_path.relative_to(output_root).as_posix(),
        "degraded_output": str(degraded_path.resolve()),
        "degraded_relative_path": degraded_path.relative_to(output_root).as_posix(),
        "gpu": gpu_label,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "pipeline_code_sha256": code_sha256,
        "recovery_from_invalid_success": recovery_event,
        "attempts": attempts,
    }
    for attempt_index in range(start_attempt, max_attempts):
        attempt_seed = stable_int(row["pair_seed"], attempt_index, "attempt")
        started = time.monotonic()
        try:
            task, negative = _load_pair_task(row, prompt_library, attempt_seed)
            _seed_everything(stable_int(attempt_seed, "gt"))
            gt = _call_pipe(pipe, source, task["gt_prompt"], negative, steps, cfg_scale)
            mask = None
            if task["method"] == "hybrid":
                degraded, mask, degradation_details = apply_postprocess(gt, task, attempt_seed, source=gt)
                strategy = "gt_then_local_postprocess"
            else:
                _seed_everything(stable_int(attempt_seed, "degraded"))
                generated = _call_pipe(pipe, gt, task["degradation_prompt"], negative, steps, cfg_scale)
                degraded = _protect_gt_background(gt, generated, source) if task["affected_scope"] == "object_only" else generated
                degradation_details = {"method": task["method"]}
                strategy = "gt_then_local_api_edit"
                if task["affected_scope"] == "object_only":
                    degradation_details["background_protection"] = "source_object_mask_recomposite"
            _atomic_save_image(gt, gt_path)
            _atomic_save_image(degraded, degraded_path)
            if mask is not None:
                _atomic_save_image(mask.convert("RGB"), mask_path)
            audit = basic_audit(row, task, source_path, gt_path, degraded_path, attempt_seed)
            if mask is not None:
                audit["mask_sha256"] = file_sha256(mask_path)
            attempt = {
                "attempt_index": attempt_index,
                "attempt_seed": attempt_seed,
                "status": "success",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "finished_at": utc_now(),
            }
            attempts.append(attempt)
            record = {
                **base_record,
                "status": "success",
                "accepted_attempt_index": attempt_index,
                "accepted_attempt_seed": attempt_seed,
                "attempts": attempts,
                "gt_prompt": task["gt_prompt"],
                "degradation_prompt": task["degradation_prompt"],
                "parameters": task["parameters"],
                "parameter_sampling": task.get("parameter_sampling"),
                "sampled_placeholders": task["sampled_placeholders"],
                "affected_scope": task["affected_scope"],
                "preserve_scope": task["preserve_scope"],
                "scene_id": task["scene_id"],
                "scene_instance_id": task["scene_instance_id"],
                "scene_spec": task["scene_spec"],
                "degradation_strategy": strategy,
                "degradation": degradation_details,
                "mask_output": str(mask_path.resolve()) if mask is not None else None,
                "mask_relative_path": mask_path.relative_to(output_root).as_posix() if mask is not None else None,
                "basic_audit": audit,
                "completed_at": utc_now(),
            }
            atomic_write_json(metadata_path, record)
            return "success", pipe
        except Exception as exc:
            attempts.append({
                "attempt_index": attempt_index,
                "attempt_seed": attempt_seed,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=12),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "finished_at": utc_now(),
            })
            status = "reject" if len(attempts) >= max_attempts else "retry_pending"
            atomic_write_json(metadata_path, {**base_record, "status": status, "attempts": attempts, "updated_at": utc_now()})
            try:
                del pipe
            except Exception:
                pass
            release_cuda_memory()
            pipe = None
            if status == "reject":
                return "reject", pipe
            pipe = load_pipeline(model_path, "cuda")
    return "reject", pipe


def _signal_handler(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_worker(
    plan_path: Path, output_root: Path, dataset_root: Path, prompt_library: Path,
    model_path: Path, *, shard_index: int, shard_count: int, gpu_label: str,
    steps: int, cfg_scale: float, max_attempts: int,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    state_path = output_root / "state/workers" / f"worker_{shard_index:02d}_{gpu_label}.json"
    counters: dict[str, int] = {
        "success": 0, "reject": 0, "skipped_success": 0,
        "skipped_reject": 0, "fatal": 0,
    }
    rows = [row for row in iter_jsonl(plan_path) if int(row["plan_index"]) % shard_count == shard_index]
    if limit is not None:
        rows = rows[:limit]
    current_library_hash = file_sha256(prompt_library)
    if any(row.get("prompt_library_sha256") != current_library_hash for row in rows):
        raise ValueError("Prompt library hash does not match the frozen plan")
    pending_exists = False
    for row in rows:
        metadata = _read_existing_metadata(pair_output_dir(output_root, row) / "metadata.json")
        if not metadata or metadata.get("status") not in {"success", "reject"}:
            pending_exists = True
            break
    pipe = load_pipeline(model_path, "cuda") if pending_exists else None
    source_hash_cache: dict[str, str] = {}
    code_sha256 = pipeline_code_sha256()
    started = time.monotonic()
    try:
        for local_index, row in enumerate(rows):
            if STOP_REQUESTED:
                break
            if pipe is None:
                metadata = _read_existing_metadata(pair_output_dir(output_root, row) / "metadata.json")
                if not metadata or metadata.get("status") not in {"success", "reject"}:
                    pipe = load_pipeline(model_path, "cuda")
            try:
                outcome, pipe = process_pair(
                    row, pipe, prompt_library, dataset_root, output_root,
                    model_path=model_path, steps=steps, cfg_scale=cfg_scale,
                    max_attempts=max_attempts, gpu_label=gpu_label,
                    source_hash_cache=source_hash_cache,
                    code_sha256=code_sha256,
                )
                counters[outcome] = counters.get(outcome, 0) + 1
            except Exception as exc:
                outcome = "fatal"
                counters["fatal"] += 1
                print(f"FATAL pair={row['pair_id']} {type(exc).__name__}: {exc}", flush=True)
            state = {
                "status": "stopping" if STOP_REQUESTED else "running",
                "gpu": gpu_label,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "local_index": local_index + 1,
                "local_total": len(rows),
                "current_pair_id": row["pair_id"],
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "counters": counters,
                "updated_at": utc_now(),
            }
            atomic_write_json(state_path, state)
            print(f"[{local_index + 1}/{len(rows)}] gpu={gpu_label} pair={row['pair_id']} outcome={outcome}", flush=True)
    finally:
        if pipe is not None:
            del pipe
            release_cuda_memory()
    final = {
        "status": "stopped" if STOP_REQUESTED else "complete",
        "gpu": gpu_label,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "local_total": len(rows),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "counters": counters,
        "updated_at": utc_now(),
    }
    atomic_write_json(state_path, final)
    return final
