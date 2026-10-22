from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from gate3dgen.data.pair_generation.generation import _load_pair_task, basic_audit

from . import DECISION_POLICY_VERSION, SCHEMA_VERSION
from .common import (
    atomic_write_json, file_sha256, input_fingerprint, is_crop_related, read_record, record_path, resolve_metadata_path,
    resolve_pair_path, resolve_source_path, stage_rows, utc_now,
)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _luma(array: np.ndarray) -> np.ndarray:
    value = array.astype(np.float32)
    return 0.2126 * value[..., 0] + 0.7152 * value[..., 1] + 0.0722 * value[..., 2]


def _laplacian_variance(array: np.ndarray) -> float:
    gray = _luma(array)
    padded = np.pad(gray, 1, mode="reflect")
    lap = (
        -4 * padded[1:-1, 1:-1] + padded[:-2, 1:-1] + padded[2:, 1:-1]
        + padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    return float(lap.var())


def _image_stats(array: np.ndarray) -> dict[str, Any]:
    lum = _luma(array)
    return {
        "shape": list(array.shape),
        "mean_luma": round(float(lum.mean()), 6),
        "std": round(float(array.std()), 6),
        "black_ratio": round(float((array.max(axis=2) <= 3).mean()), 8),
        "white_ratio": round(float((array.min(axis=2) >= 252).mean()), 8),
        "channel_means": [round(float(value), 6) for value in array.mean(axis=(0, 1))],
        "laplacian_variance": round(_laplacian_variance(array), 6),
    }


def _effect_metrics(gt: np.ndarray, degraded: np.ndarray, subject_mask: np.ndarray | None = None) -> dict[str, Any]:
    delta = degraded.astype(np.int16) - gt.astype(np.int16)
    gt_luma, degraded_luma = _luma(gt), _luma(degraded)
    gt_lap = max(_laplacian_variance(gt), 1e-9)
    channel_delta = degraded.mean(axis=(0, 1)) - gt.mean(axis=(0, 1))
    result = {
        "mean_abs_diff": round(float(np.abs(delta).mean()), 6),
        "changed_pixel_ratio": round(float(np.any(delta != 0, axis=2).mean()), 8),
        "max_abs_diff": int(np.abs(delta).max()),
        "luma_delta": round(float(degraded_luma.mean() - gt_luma.mean()), 6),
        "black_ratio_delta": round(float((degraded_luma <= 10).mean() - (gt_luma <= 10).mean()), 8),
        "clipped_ratio_delta": round(float((degraded_luma >= 250).mean() - (gt_luma >= 250).mean()), 8),
        "laplacian_variance_ratio": round(_laplacian_variance(degraded) / gt_lap, 6),
        "channel_mean_delta": [round(float(value), 6) for value in channel_delta],
        "channel_delta_spread": round(float(channel_delta.max() - channel_delta.min()), 6),
        "contrast_ratio": round(float(degraded_luma.std() / max(float(gt_luma.std()), 1e-9)), 6),
    }
    if subject_mask is not None and subject_mask.shape == gt_luma.shape and subject_mask.any():
        background = ~subject_mask
        result["subject_luma_delta"] = round(float((degraded_luma - gt_luma)[subject_mask].mean()), 6)
        result["subject_mean_abs_diff"] = round(float(np.abs(delta)[subject_mask].mean()), 6)
        result["background_mean_abs_diff"] = round(float(np.abs(delta)[background].mean()), 6) if background.any() else 0.0
    return result


def _effect_review_reasons(task: str, metrics: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if metrics["mean_abs_diff"] < cfg["minimum_mean_abs_diff_review"]:
        reasons.append("target_effect_too_weak_by_pixel_difference")
    if task == "overexposure" and metrics["luma_delta"] < cfg["overexposure_luma_delta_review"]:
        reasons.append("overexposure_luma_increase_too_small")
    elif task == "underexposure" and metrics["luma_delta"] > cfg["underexposure_luma_delta_review"]:
        reasons.append("underexposure_luma_decrease_too_small")
    elif task in {"optical_blur", "motion_blur"} and metrics["laplacian_variance_ratio"] > cfg["blur_laplacian_ratio_review"]:
        reasons.append("blur_sharpness_drop_too_small")
    elif task == "white_balance" and metrics["channel_delta_spread"] < cfg["white_balance_channel_delta_review"]:
        reasons.append("white_balance_channel_shift_too_small")
    elif task == "shadow" and metrics.get("subject_luma_delta", metrics["luma_delta"]) > cfg["shadow_subject_luma_delta_review"]:
        reasons.append("shadow_subject_darkening_too_small")
    elif task == "local_color_spill" and metrics["channel_delta_spread"] < cfg["local_color_spill_channel_delta_review"]:
        reasons.append("local_color_spill_channel_shift_too_small")
    elif task in {"glass_reflection", "veiling_glare", "dirty_glass"} and metrics["mean_abs_diff"] < cfg["generative_effect_mean_abs_diff_review"]:
        reasons.append(f"{task}_traditional_effect_proxy_too_weak")
    return reasons


def _audit_crop_only(row: dict[str, Any], args, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Fast full-dataset gate: reject crop and only prepare dependencies otherwise."""
    pair_id = str(row["pair_id"])
    crop = is_crop_related(row)
    paths: dict[str, Path] = {}
    metadata: dict[str, Any] = {}
    diagnostics: list[str] = []
    if not crop:
        try:
            paths = {
                "source": resolve_source_path(row, args.dataset_root),
                "gt": resolve_pair_path(row, args.pair_root, "gt"),
                "degraded": resolve_pair_path(row, args.pair_root, "degraded"),
                "metadata": resolve_metadata_path(row, args.pair_root),
            }
            if paths["metadata"].is_file():
                metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            else:
                diagnostics.append("crop_only_metadata_missing")
        except Exception as exc:
            # In crop_only mode only crop is a programmatic rejection. Missing
            # dependencies are left for the image/VLM stages to expose as
            # reviewable evidence instead of expanding this fast hard gate.
            diagnostics.append(f"crop_only_dependency_resolution:{type(exc).__name__}:{exc}")
    method = str(metadata.get("method") or row.get("method") or "")
    reject = ["policy_crop_degradation_disallowed"] if crop else []
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "programmatic",
        "completed_at": utc_now(),
        "pair_id": pair_id,
        "sha256": row.get("sha256"),
        "task_id": row.get("task_id"),
        "degradation_type": row.get("degradation_type"),
        "method": method,
        "affected_scope": metadata.get("affected_scope"),
        "expected_gt_prompt": "",
        "metadata_gt_prompt": metadata.get("gt_prompt", ""),
        "expected_degradation_prompt": metadata.get("degradation_prompt", ""),
        "prompt_library": str(args.prompt_library.resolve()),
        "status": "reject" if crop else "pass",
        "reject_reasons": reject,
        "review_reasons": [],
        "diagnostic_flags": diagnostics,
        "needs_vlm": bool(getattr(args, "vlm_all_pairs", False)) or method != "hybrid",
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "image_statistics": {},
        "degradation_metrics": {},
        "replay": {"status": "skipped", "reason": "programmatic_mode_crop_only"},
        "threshold_version": thresholds["schema_version"],
        "decision_policy_version": DECISION_POLICY_VERSION,
        "decision_source": "hard_rule" if crop else "programmatic_gate",
        "programmatic_mode": "crop_only",
        "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False)),
        "input_fingerprint": input_fingerprint(
            row, stage="programmatic",
            extra={"threshold": thresholds["schema_version"], "policy": DECISION_POLICY_VERSION,
                   "mode": "crop_only", "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False))},
        ),
    }


def audit_pair(row: dict[str, Any], args, thresholds: dict[str, Any]) -> dict[str, Any]:
    if getattr(args, "programmatic_mode", "full") == "crop_only":
        return _audit_crop_only(row, args, thresholds)
    pair_id = str(row["pair_id"])
    errors: list[str] = []
    reviews: list[str] = []
    paths: dict[str, Path] = {}
    metadata: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    effect: dict[str, Any] = {}
    replay: dict[str, Any] = {"status": "not_run"}
    task_definition: dict[str, Any] = {}
    candidate_reviews: list[str] = []
    try:
        paths = {
            "source": resolve_source_path(row, args.dataset_root),
            "gt": resolve_pair_path(row, args.pair_root, "gt"),
            "degraded": resolve_pair_path(row, args.pair_root, "degraded"),
            "metadata": resolve_metadata_path(row, args.pair_root),
        }
        for name, path in paths.items():
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append(f"missing_or_empty_{name}")
        if not errors:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            for key in ("pair_id", "sha256", "task_id"):
                if str(metadata.get(key)) != str(row.get(key)):
                    errors.append(f"metadata_{key}_mismatch")
            if metadata.get("status") != "success":
                errors.append("metadata_status_not_success")
            arrays = {name: _load_rgb(paths[name]) for name in ("source", "gt", "degraded")}
            if arrays["gt"].shape != arrays["degraded"].shape:
                errors.append("gt_degraded_dimension_mismatch")
            declared = row.get("dimensions") or metadata.get("dimensions")
            if isinstance(declared, (list, tuple)) and len(declared) >= 2:
                actual_wh = [int(arrays["gt"].shape[1]), int(arrays["gt"].shape[0])]
                if [int(declared[0]), int(declared[1])] != actual_wh:
                    errors.append("declared_gt_dimensions_mismatch")
            image_cfg = thresholds["image"]
            for name, array in arrays.items():
                stats[name] = _image_stats(array)
                if stats[name]["std"] < image_cfg["minimum_std"]:
                    errors.append(f"{name}_near_empty")
                if stats[name]["black_ratio"] >= image_cfg["all_black_ratio_reject"]:
                    errors.append(f"{name}_almost_all_black")
                if stats[name]["white_ratio"] >= image_cfg["all_white_ratio_reject"]:
                    errors.append(f"{name}_almost_all_white")
            hashes = {name: file_sha256(paths[name]) for name in ("source", "gt", "degraded")}
            expected = {
                "source": metadata.get("source_image_sha256") or row.get("source_image_sha256"),
                "gt": metadata.get("basic_audit", {}).get("gt_sha256") or row.get("gt_sha256"),
                "degraded": metadata.get("basic_audit", {}).get("degraded_sha256") or row.get("degraded_sha256"),
            }
            for name in hashes:
                if expected[name] and hashes[name] != expected[name]:
                    errors.append(f"{name}_sha256_mismatch")
            if hashes["gt"] == hashes["degraded"]:
                errors.append("duplicate_gt_degraded_images")
            if hashes["source"] in {hashes["gt"], hashes["degraded"]}:
                errors.append("duplicate_source_pair_image")
            source_mask = arrays["source"].max(axis=2) > 12
            if source_mask.shape != arrays["gt"].shape[:2]:
                source_mask = np.asarray(
                    Image.fromarray(source_mask.astype(np.uint8) * 255).resize(
                        (arrays["gt"].shape[1], arrays["gt"].shape[0]), Image.Resampling.NEAREST
                    )
                ) > 0
            effect = _effect_metrics(arrays["gt"], arrays["degraded"], source_mask)
            # Effect-strength thresholds are useful diagnostics, but weak
            # values alone do not mean the pair is unusable. Final quality is
            # now governed by subject/background/occlusion semantics.
            effect_reviews = _effect_review_reasons(
                str(row["degradation_type"]), effect, thresholds["programmatic_effect"]
            )
            reviews.extend(effect_reviews)
            # A single global pixel statistic is too weak to gate localized
            # effects.  Two independent weak-shadow signals, however, mean a
            # requested heavy shadow is visibly too small and should be sent
            # to human review rather than silently passed.
            if str(row["degradation_type"]) == "shadow" and {
                "target_effect_too_weak_by_pixel_difference",
                "shadow_subject_darkening_too_small",
            }.issubset(effect_reviews):
                candidate_reviews.append("programmatic_target_effect_too_weak")
            if metadata.get("affected_scope") == "foreground_layer":
                subject_diff = float(effect.get("subject_mean_abs_diff", 0.0))
                background_diff = float(effect.get("background_mean_abs_diff", 0.0))
                effect["foreground_subject_background_diff_ratio"] = round(
                    subject_diff / max(background_diff, 1e-9), 6
                )
                effect_cfg = thresholds["programmatic_effect"]
                if (
                    background_diff >= effect_cfg["foreground_layer_minimum_background_diff"]
                    and effect["foreground_subject_background_diff_ratio"]
                    < effect_cfg["foreground_layer_subject_background_ratio_review"]
                ):
                    candidate_reviews.append("programmatic_foreground_layer_effect_not_over_subject")
            try:
                seed = int(metadata["accepted_attempt_seed"])
                task, _ = _load_pair_task(row, args.prompt_library, seed)
                task_definition = task
                replay_result = basic_audit(row, task, paths["source"], paths["gt"], paths["degraded"], seed)
                replay = {"status": "pass", **replay_result}
            except Exception as exc:
                replay = {"status": "reject", "error": f"{type(exc).__name__}: {exc}"}
                errors.append("parameter_replay_or_background_protection_failed")
    except Exception as exc:
        errors.append(f"programmatic_exception:{type(exc).__name__}:{exc}")
    method = str(metadata.get("method") or row.get("method") or "")
    # Crop is intentionally excluded from this training selection, including
    # both the single crop task and any composed task containing a crop factor.
    if is_crop_related(row) or is_crop_related(metadata) or is_crop_related(task_definition):
        errors.append("policy_crop_degradation_disallowed")
    status = "reject" if errors else "pass"
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "programmatic",
        "completed_at": utc_now(),
        "pair_id": pair_id,
        "sha256": row.get("sha256"),
        "task_id": row.get("task_id"),
        "degradation_type": row.get("degradation_type"),
        "method": method,
        "affected_scope": metadata.get("affected_scope"),
        # Use the task expanded from the configured prompt library and the
        # pair's accepted seed.  Keep the metadata copy for traceability to
        # the prompt that actually generated this pair.
        "expected_gt_prompt": task_definition.get("gt_prompt") or metadata.get("gt_prompt", ""),
        "metadata_gt_prompt": metadata.get("gt_prompt", ""),
        "expected_degradation_prompt": metadata.get("degradation_prompt", ""),
        "prompt_library": str(args.prompt_library.resolve()),
        "status": status,
        "reject_reasons": sorted(set(errors)),
        "review_reasons": sorted(set(candidate_reviews)),
        "diagnostic_flags": sorted(set(reviews + candidate_reviews)),
        "needs_vlm": bool(getattr(args, "vlm_all_pairs", False)) or method != "hybrid",
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "image_statistics": stats,
        "degradation_metrics": effect,
        "replay": replay,
        "threshold_version": thresholds["schema_version"],
        "decision_policy_version": DECISION_POLICY_VERSION,
        "decision_source": "hard_rule" if errors else "programmatic_gate",
        "programmatic_mode": "full",
        "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False)),
        "input_fingerprint": input_fingerprint(
            row, stage="programmatic",
            extra={"threshold": thresholds["schema_version"], "policy": DECISION_POLICY_VERSION,
                   "mode": "full", "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False))},
        ),
    }


def run_programmatic(args, thresholds: dict[str, Any]) -> dict[str, int]:
    rows = list(stage_rows(args.manifest, args.shard_index, args.shard_count, args.limit))
    counters = {"processed": 0, "skipped": 0, "pass": 0, "review": 0, "reject": 0}

    def process(row):
        existing = read_record(args.output_root, "programmatic", row["pair_id"])
        fingerprint = input_fingerprint(
            row, stage="programmatic",
            extra={"threshold": thresholds["schema_version"], "policy": DECISION_POLICY_VERSION,
                   "mode": getattr(args, "programmatic_mode", "full"),
                   "vlm_all_pairs": bool(getattr(args, "vlm_all_pairs", False))},
        )
        if (not getattr(args, "force", False) and existing
                and existing.get("schema_version") == SCHEMA_VERSION
                and existing.get("decision_policy_version") == DECISION_POLICY_VERSION
                and existing.get("input_fingerprint") == fingerprint):
            return "skipped", existing
        record = audit_pair(row, args, thresholds)
        atomic_write_json(record_path(args.output_root, "programmatic", row["pair_id"]), record)
        return "processed", record

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.cpu_workers) as executor:
        for action, record in executor.map(process, rows):
            counters[action] += 1
            counters[record["status"]] += 1
    return counters
