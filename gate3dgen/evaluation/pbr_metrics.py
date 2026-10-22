"""Strict paired-render metrics for geometry and PBR appearance.

The evaluator consumes renders made on the same camera/pixel grid.  It never
resizes, re-crops, independently normalizes, or compares images from two
different render setups.  Missing materials, camera metadata, or optional
learned providers produce an explicit ``unavailable`` status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from .io import finite_json, iter_jsonl, write_jsonl


CHANNELS: tuple[str, ...] = (
    "normal", "mask", "base_color", "metallic", "roughness", "alpha", "shaded"
)
RGB_CHANNELS = {"normal", "base_color", "shaded"}
PAPER_CHANNELS = ("normal", "base_color", "shaded")
SCALAR_CHANNELS = {"mask", "metallic", "roughness", "alpha"}
_LPIPS_MODELS: dict[str, Any] = {}


def _coverage_mask(array: np.ndarray) -> np.ndarray:
    """Decode a rendered silhouette mask without assuming display-white output.

    Blender's color management can encode a fully covered emission/mask pixel
    below 0.5 (66/255 in some of the benchmark renders).  These files are
    coverage masks, not material alpha maps, so every non-zero stored value is
    foreground.  PNG backgrounds are written as exact zero; using half of one
    quantization step also keeps the rule explicit for normalized arrays.
    """

    return array[..., 0] > (0.5 / 255.0)


def _array(path: Path, channel: str) -> np.ndarray:
    with Image.open(path) as image:
        # Keep channel semantics explicit.  Alpha is a scalar map; all other
        # channels use RGB to avoid palette/greyscale surprises.
        mode = "L" if channel in SCALAR_CHANNELS else "RGB"
        result = np.asarray(image.convert(mode), dtype=np.float32) / 255.0
    if result.ndim == 2:
        result = result[..., None]
    return result


def _ssim_fallback(pred: np.ndarray, target: np.ndarray) -> float:
    """Global SSIM fallback when scikit-image is unavailable.

    The bundled dependency normally supplies windowed SSIM.  This fallback is
    deterministic and deliberately labeled in result metadata; it is not
    silently presented as the windowed implementation.
    """

    x, y = pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
    values = []
    c1, c2 = 0.01**2, 0.03**2
    for channel in range(x.shape[1]):
        a, b = x[:, channel], y[:, channel]
        mean_a, mean_b = float(a.mean()), float(b.mean())
        var_a, var_b = float(a.var()), float(b.var())
        cov = float(np.mean((a - mean_a) * (b - mean_b)))
        values.append(((2 * mean_a * mean_b + c1) * (2 * cov + c2)) / ((mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)))
    return float(np.mean(values))


def _ssim(pred: np.ndarray, target: np.ndarray) -> tuple[float, str]:
    try:
        from skimage.metrics import structural_similarity

        height, width = pred.shape[:2]
        win_size = min(7, height, width)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            return _ssim_fallback(pred, target), "global_fallback_small_image"
        return float(structural_similarity(
            target,
            pred,
            channel_axis=2 if pred.shape[-1] > 1 else None,
            data_range=1.0,
            win_size=win_size,
        )), "skimage_windowed"
    except Exception:
        return _ssim_fallback(pred, target), "global_fallback"


def _lpips(pred: np.ndarray, target: np.ndarray, device: str = "cpu") -> tuple[float | None, str]:
    try:
        import torch
        import lpips
    except Exception as error:
        return None, f"lpips_dependency_unavailable:{type(error).__name__}"
    if pred.shape[-1] == 1:
        pred = np.repeat(pred, 3, axis=-1)
        target = np.repeat(target, 3, axis=-1)
    try:
        model = _LPIPS_MODELS.get(device)
        if model is None:
            model = lpips.LPIPS(net="alex").eval().to(device)
            _LPIPS_MODELS[device] = model
        with torch.inference_mode():
            a = torch.from_numpy(pred).permute(2, 0, 1)[None].to(device) * 2 - 1
            b = torch.from_numpy(target).permute(2, 0, 1)[None].to(device) * 2 - 1
            value = float(model(a, b).item())
        return value, "lpips_alex"
    except Exception as error:
        return None, f"lpips_failed:{type(error).__name__}:{error}"


def paired_image_metrics(
    prediction: Path,
    target: Path,
    *,
    channel: str,
    enable_lpips: bool = False,
    device: str = "cpu",
    prediction_mask: Path | None = None,
    target_mask: Path | None = None,
) -> dict[str, Any]:
    """Compare two already aligned renders without modifying their pixels."""

    if channel not in CHANNELS:
        raise ValueError(f"unsupported channel: {channel}")
    pred, ref = _array(prediction, channel), _array(target, channel)
    if pred.shape != ref.shape:
        raise ValueError(f"paired render shape mismatch: prediction={pred.shape}, target={ref.shape}")
    union_mask = None
    intersection_mask = None
    metric_pred, metric_ref = pred, ref
    mask_policy = "full_frame"
    if channel != "mask" and prediction_mask and target_mask:
        pred_coverage = _array(prediction_mask, "mask")
        ref_coverage = _array(target_mask, "mask")
        # Broken OCIO can turn every channel into identical 0/1-valued noise.
        # Such masks are not a low-exposure legitimate silhouette (historical
        # emission masks can be 66/255). Never award perfect scores to them.
        if any(float(np.max(mask)) < 2.5 / 255.0 for mask in (pred_coverage, ref_coverage)):
            raise ValueError("paired render mask has no usable foreground signal")
        pred_mask = _coverage_mask(pred_coverage)
        ref_mask = _coverage_mask(ref_coverage)
        if not np.any(pred_mask) or not np.any(ref_mask):
            raise ValueError("paired render mask is empty")
        if pred_mask.shape != pred.shape[:2] or ref_mask.shape != ref.shape[:2]:
            raise ValueError("paired mask shape does not match render shape")
        union_mask = np.logical_or(pred_mask, ref_mask)
        intersection_mask = np.logical_and(pred_mask, ref_mask)
        if not np.any(union_mask):
            raise ValueError("paired render union mask is empty")
        rows, columns = np.where(union_mask)
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(columns.min()), int(columns.max()) + 1
        crop_mask = union_mask[y0:y1, x0:x1, None]
        metric_pred = np.where(crop_mask, pred[y0:y1, x0:x1], 0.0)
        metric_ref = np.where(crop_mask, ref[y0:y1, x0:x1], 0.0)
        error_values = (pred - ref)[union_mask]
        mask_policy = "union_mask_values_union_tight_bbox_for_ssim_lpips"
    else:
        error_values = (pred - ref).reshape(-1, pred.shape[-1])
    mse = float(np.mean(error_values**2))
    ssim_value, ssim_implementation = _ssim(metric_pred, metric_ref)
    result: dict[str, Any] = {
        "mae": float(np.mean(np.abs(error_values))),
        "mse": mse,
        "psnr_db": "Infinity" if mse == 0 else -10.0 * math.log10(mse),
        "ssim": ssim_value,
        "ssim_implementation": ssim_implementation,
        "normalization": "none_shared_pixel_grid",
        "mask_policy": mask_policy,
        "coverage_mask_threshold": "nonzero_png_value",
    }
    if channel == "mask":
        mask_pred, mask_ref = _coverage_mask(pred), _coverage_mask(ref)
        union = np.logical_or(mask_pred, mask_ref).sum()
        result["iou"] = 1.0 if union == 0 else float(np.logical_and(mask_pred, mask_ref).sum() / union)
    if channel == "alpha":
        alpha_pred, alpha_ref = pred[..., 0] > 0.5, ref[..., 0] > 0.5
        union = np.logical_or(alpha_pred, alpha_ref).sum()
        result["alpha_iou"] = 1.0 if union == 0 else float(np.logical_and(alpha_pred, alpha_ref).sum() / union)
    if channel == "normal":
        # Normal maps are expected to be encoded in [0,1].  The angular
        # metric is supplementary to pixel PSNR/SSIM and uses no renormalizing
        # of the rendered images themselves.
        p = pred * 2 - 1
        t = ref * 2 - 1
        p /= np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)
        t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)
        angle = np.degrees(np.arccos(np.clip(np.sum(p * t, axis=-1), -1.0, 1.0)))
        if intersection_mask is not None:
            if not np.any(intersection_mask):
                raise ValueError("paired normal intersection mask is empty")
            angle = angle[intersection_mask]
        result.update({
            "normal_mean_angular_error_deg": float(np.mean(angle)),
            "normal_median_angular_error_deg": float(np.median(angle)),
            "normal_accuracy_11.25deg": float(np.mean(angle <= 11.25)),
            "normal_accuracy_22.5deg": float(np.mean(angle <= 22.5)),
            "normal_accuracy_30deg": float(np.mean(angle <= 30.0)),
        })
    if enable_lpips and channel in RGB_CHANNELS:
        value, status = _lpips(metric_pred, metric_ref, device)
        result["lpips_alex"] = value
        result["lpips_status"] = status
    return result


def _canonical_camera(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_camera(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_camera(item) for item in value]
    if isinstance(value, float):
        return round(value, 10)
    return value


def _camera_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the render-affecting part of a camera contract.

    ``camera.json`` is a long-lived artifact and its descriptive schema has
    evolved.  In particular, benchmark-camera v1 originally stored only
    ``camera_matrix_world`` while newer renderers also store an equivalent
    one-element ``camera_views`` list and a more precise ``camera_source``.
    Hashing the complete metadata object therefore rejects pixel-aligned
    renders even though their camera and renderer settings are identical.

    Deliberately exclude provenance-only fields such as ``fingerprint``,
    ``camera_source`` and ``render_device``.  Keep every setting that can
    affect the rendered pixel grid or values.
    """

    raw_views = value.get("camera_views")
    matrices: list[Any] = []
    if isinstance(raw_views, list):
        for view in raw_views:
            if isinstance(view, Mapping) and view.get("expected_camera_matrix_world") is not None:
                matrices.append(view["expected_camera_matrix_world"])
    if not matrices and value.get("camera_matrix_world") is not None:
        matrices = [value["camera_matrix_world"]]
    return {
        "protocol": value.get("protocol"),
        "coordinate_frame": value.get("coordinate_frame"),
        "camera_mode": value.get("camera_mode"),
        "camera_matrices_world": matrices,
        "camera_fov_deg": value.get("camera_fov_deg"),
        "views": value.get("views"),
        "resolution": value.get("resolution"),
        "render_samples": value.get("render_samples"),
        "channels": value.get("channels"),
        "renderer": value.get("renderer"),
        "view_transform": value.get("view_transform"),
        "frame_policy": value.get("frame_policy"),
        "shaded_light_mode": value.get("shaded_light_mode"),
        "shaded_light_energy_scale": value.get("shaded_light_energy_scale"),
        "uniform_light_strength": value.get("uniform_light_strength"),
    }


def camera_fingerprint(value: Mapping[str, Any] | None) -> str | None:
    if not value:
        return None
    text = json.dumps(
        _canonical_camera(_camera_identity(value)),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _camera_metadata(render_root: Path, baseline: str, case_id: str, condition: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    candidates = (
        render_root / baseline / case_id / condition / "camera.json",
        render_root / baseline / case_id / condition / "camera_params.json",
        render_root / baseline / case_id / "camera.json",
        render_root / baseline / case_id / "camera_params.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return None, None, f"camera_metadata_not_object:{path}"
            # Some generators wrap the camera in a top-level field.
            camera = value.get("camera", value)
            if not isinstance(camera, dict):
                return None, None, f"camera_metadata_not_object:{path}"
            return camera, camera_fingerprint(camera), None
        except Exception as error:
            return None, None, f"camera_metadata_unreadable:{path}:{error}"
    return None, None, "camera_contract_missing"


def build_paired_render_rows(
    condition_rows: Iterable[Mapping[str, Any]],
    render_root: Path,
    *,
    baseline: str,
    channels: Sequence[str] = CHANNELS,
    views: int = 4,
    variants: Sequence[str] | None = None,
    require_camera_contract: bool = True,
) -> list[dict[str, Any]]:
    """Create strict prediction/reference render pairs from one render root."""

    invalid = set(channels) - set(CHANNELS)
    if invalid:
        raise ValueError(f"unsupported channels: {sorted(invalid)}")
    result: list[dict[str, Any]] = []
    for case in condition_rows:
        wild = case.get("dataset_kind") == "wild_no_gt"
        # Benchmark PBR quality must be measured against renders of the real
        # DTC GLB GT, not the clean-image-input generated 3D proxy.  The
        # required layout is:
        #   predictions: <root>/<baseline>/<case>/<variant>/<channel>/view_XX.png
        #   real GT:     <root>/glb_gt/<case>/<channel>/view_XX.png
        # Both sides must carry the same camera fingerprint.
        reference = "original" if wild else "glb_gt"
        if variants is None:
            if wild:
                selected = ["original", *[f"restored_s{step:02d}" for step in case.get("restoration_steps", (1, 3, 5, 10, 20, 30, 40))]]
            else:
                selected = ["degraded", "clean_gt", *[f"restored_s{step:02d}" for step in case.get("restoration_steps", (1, 3, 5, 10, 20, 30, 40))]]
        else:
            selected = list(variants)
        if wild:
            target_root = render_root / baseline / str(case["case_id"]) / reference
            _, reference_fingerprint, reference_error = _camera_metadata(
                render_root, baseline, str(case["case_id"]), reference
            )
        else:
            target_root = render_root / "glb_gt" / str(case["case_id"])
            _, reference_fingerprint, reference_error = _camera_metadata(
                render_root, "glb_gt", str(case["case_id"]), ""
            )
        for variant in selected:
            _, prediction_fingerprint, prediction_error = _camera_metadata(render_root, baseline, str(case["case_id"]), variant)
            camera_error = prediction_error or reference_error
            camera_ok = bool(reference_fingerprint and prediction_fingerprint and reference_fingerprint == prediction_fingerprint)
            if reference_fingerprint and prediction_fingerprint and not camera_ok:
                camera_error = "camera_fingerprint_mismatch"
            if not reference_fingerprint or not prediction_fingerprint:
                camera_ok = False
            for channel in channels:
                for view in range(views):
                    prediction = render_root / baseline / str(case["case_id"]) / variant / channel / f"view_{view:02d}.png"
                    target = target_root / channel / f"view_{view:02d}.png"
                    result.append({
                        "schema_version": "official_aligned_v1",
                        "case_id": str(case["case_id"]),
                        "asset_id": case.get("asset_id"),
                        "task_id": case.get("task_id"),
                        "dataset_kind": case.get("dataset_kind"),
                        "baseline": baseline,
                        "variant": variant,
                        "reference": reference,
                        "reference_is_true_gt": not wild,
                        "metric_family": "pbr",
                        "step": int(variant.removeprefix("restored_s")) if variant.startswith("restored_s") else None,
                        "channel": channel,
                        "view": view,
                        "prediction_path": str(prediction.absolute()),
                        "target_path": str(target.absolute()),
                        "camera_fingerprint": reference_fingerprint if camera_ok else None,
                        "camera_contract": "shared_verified" if camera_ok else (camera_error or "camera_contract_missing"),
                        "status": "pending",
                        "require_camera_contract": require_camera_contract,
                    })
    return result


def evaluate_paired_render_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    enable_lpips: bool = False,
    device: str = "cpu",
    allow_unverified_camera: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["expected_metrics"] = ["mae", "mse", "psnr_db", "ssim"]
        if enable_lpips and row.get("channel") in RGB_CHANNELS:
            row["expected_metrics"].append("lpips_alex")
        if row.get("camera_contract") != "shared_verified" and not allow_unverified_camera:
            row.update({"status": "unavailable", "reason": row.get("camera_contract", "camera_contract_missing")})
            results.append(row)
            continue
        prediction, target = Path(row["prediction_path"]), Path(row["target_path"])
        if not prediction.is_file() or not target.is_file():
            missing_kind = "missing_material_render" if row.get("channel") in {"base_color", "metallic", "roughness", "alpha"} else None
            row.update({
                "status": "unavailable",
                "reason": missing_kind or ("missing_prediction_render" if not prediction.is_file() else "missing_reference_render"),
            })
            results.append(row)
            continue
        try:
            values = paired_image_metrics(
                prediction,
                target,
                channel=str(row["channel"]),
                enable_lpips=enable_lpips,
                device=device,
                prediction_mask=prediction.parent.parent / "mask" / prediction.name,
                target_mask=target.parent.parent / "mask" / target.name,
            )
            row.update({
                "status": "ok",
                "compliance": "verified_camera" if row.get("camera_contract") == "shared_verified" else "unverified_camera",
                "metrics": values,
            })
            if enable_lpips and row["channel"] in RGB_CHANNELS and not _valid_metric(values.get("lpips_alex"), "lpips_alex"):
                row.update(status="partial", reason=values.get("lpips_status", "lpips_missing"))
        except Exception as error:
            row.update({"status": "unavailable", "reason": f"metric_error:{type(error).__name__}:{error}"})
        results.append(row)
    return results


def _valid_metric(value: Any, name: str) -> bool:
    if name == "psnr_db" and value == "Infinity":
        return True
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    # Perfect pixel equality has mathematically infinite PSNR. Other metrics,
    # particularly LPIPS, must be finite; NaN is never an observation.
    return math.isfinite(value) or (name == "psnr_db" and value == math.inf)


def summarize_render_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[":".join(str(row.get(name, "")) for name in ("baseline", "variant", "reference", "channel"))].append(row)
    output: dict[str, Any] = {}
    for key, group in sorted(groups.items()):
        usable = [row for row in group if row.get("status") in {"ok", "partial"}]
        names = sorted({
            name
            for row in group
            for name, value in row.get("metrics", {}).items()
            if value is None or _valid_metric(value, name) or (isinstance(value, (int, float)) and not isinstance(value, bool))
        } | {"mae", "mse", "psnr_db", "ssim"}
          | {name for row in group for name in row.get("expected_metrics", ())})
        means: dict[str, Any] = {}
        coverage: dict[str, Any] = {}
        for name in names:
            values = [
                math.inf if row.get("metrics", {}).get(name) == "Infinity" else row.get("metrics", {}).get(name)
                for row in usable
                if _valid_metric(row.get("metrics", {}).get(name), name)
            ]
            means[name] = ("Infinity" if any(value == math.inf for value in values)
                           else float(np.mean(values)) if values else None)
            coverage[name] = {
                "valid_count": len(values), "expected_count": len(group),
                "coverage": len(values) / len(group),
                "positive_infinity_count": sum(value == math.inf for value in values),
            }
        complete = sum(row.get("status") == "ok" for row in group)
        output[key] = {
            "records": len(group),
            "coverage": complete / max(len(group), 1),
            "coverage_definition": "complete_record_status_not_metric_coverage",
            "ok": complete,
            "partial": sum(row.get("status") == "partial" for row in group),
            "unavailable": len(group) - len(usable),
            "metrics": means,
            "metric_coverage": coverage,
        }
    return {"schema_version": "official_aligned_v1", "groups": output}


def validate_paper_render_rows(
    rows: Iterable[Mapping[str, Any]], *, views: int = 4,
    expected_rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fail closed before publishing a paper summary; never intersect subsets."""
    values = list(rows)
    if views != 4 or not values:
        raise ValueError("paper PBR evaluation requires a nonempty four-view population")
    populations: dict[tuple[str, str, str], set[tuple[str, int]]] = defaultdict(set)
    seen: set[tuple[str, str, str, str, int]] = set()
    for row in values:
        baseline, variant, channel = (str(row.get(key, "")) for key in ("baseline", "variant", "channel"))
        case, view = str(row.get("case_id", "")), row.get("view")
        identity = (baseline, variant, channel, case, view)
        if not case or not baseline or not variant or isinstance(view, bool) or not isinstance(view, int) or view not in range(views):
            raise ValueError(f"invalid paper case/view identity: {identity}")
        if identity in seen:
            raise ValueError(f"duplicate paper render row: {identity}")
        seen.add(identity)
        if row.get("reference") != "glb_gt" or not row.get("reference_is_true_gt") or row.get("camera_contract") != "shared_verified":
            raise ValueError(f"paper PBR requires verified cameras and real GLB GT: {identity}")
        populations[(baseline, variant, channel)].add((case, view))
        if row.get("status") != "ok":
            raise ValueError(f"incomplete paper render metrics: {identity}: {row.get('reason')}")
        required = ("psnr_db",) if channel == "normal" else (("lpips_alex",) if channel in {"shaded", "base_color"} else ())
        for name in required:
            if not _valid_metric(row.get("metrics", {}).get(name), name):
                raise ValueError(f"missing/nonfinite required paper metric {name}: {identity}")
    if expected_rows is not None:
        planned = {(str(row.get("baseline", "")), str(row.get("variant", "")),
                    str(row.get("channel", "")), str(row.get("case_id", "")), row.get("view"))
                   for row in expected_rows}
        if seen != planned:
            raise ValueError("paper results must cover the entire declared render plan, not a common subset")
    methods = {(baseline, variant) for baseline, variant, _ in populations}
    canonical = None
    for baseline, variant in sorted(methods):
        for channel in PAPER_CHANNELS:
            population = populations.get((baseline, variant, channel))
            if not population:
                raise ValueError(f"missing paper channel: {baseline}:{variant}:{channel}")
            cases = {case for case, _ in population}
            if population != {(case, view) for case in cases for view in range(views)}:
                raise ValueError(f"incomplete paper views: {baseline}:{variant}:{channel}")
            if canonical is None:
                canonical = population
            elif population != canonical:
                raise ValueError("paper methods/channels must use the same case/view population")
    return {"status": "passed", "cases": len({case for case, _ in canonical}),
            "case_views": len(canonical), "methods": len(methods), "views": views,
            "population_sha256": hashlib.sha256(json.dumps(sorted(canonical), separators=(",", ":")).encode()).hexdigest(),
            "required_metrics": {"normal": "psnr_db", "shaded": "lpips_alex", "base_color": "lpips_alex"}}


def audit_render_capability(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit whether a planned render set can support strict paired metrics.

    This is intentionally renderer-agnostic.  It checks the contract that a
    Blender/nvdiffrast job must satisfy (camera identity, image presence, and
    channel coverage), without pretending to render a missing material map.
    """

    values = [dict(row) for row in rows]
    reasons: dict[str, int] = defaultdict(int)
    for row in values:
        reason = row.get("camera_contract")
        if reason != "shared_verified":
            reasons[str(reason or "camera_contract_missing")] += 1
        if not Path(row["prediction_path"]).is_file():
            reasons["missing_prediction_render"] += 1
        if not Path(row["target_path"]).is_file():
            reasons["missing_reference_render"] += 1
    return {
        "schema_version": "official_aligned_v1",
        "renderer_required": "shared-camera Blender/nvdiffrast or equivalent",
        "audit_scope": "render_and_camera_presence_only_not_numeric_metric_completeness",
        "status": "ok" if values and not reasons else ("unavailable" if reasons else "empty"),
        "records": len(values),
        "camera_verified": sum(row.get("camera_contract") == "shared_verified" for row in values),
        "prediction_files": sum(Path(row["prediction_path"]).is_file() for row in values),
        "reference_files": sum(Path(row["target_path"]).is_file() for row in values),
        "unavailable_reasons": dict(sorted(reasons.items())),
        "normalization": "none_shared_pixel_grid",
        "no_independent_render_comparison": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="official-aligned condition JSONL")
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", action="append", choices=("trellis2", "pixal3d", "fixture"))
    parser.add_argument("--channel", action="append", dest="channels", choices=CHANNELS)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--paper-protocol", action="store_true", help="require complete four-view normal/shaded/albedo metrics on identical populations")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-unverified-camera", action="store_true")
    args = parser.parse_args()
    if args.paper_protocol and (not args.lpips or args.allow_unverified_camera or args.views != 4):
        parser.error("--paper-protocol requires --lpips, --views 4 and verified cameras")
    if args.paper_protocol and args.channels and not set(PAPER_CHANNELS).issubset(args.channels):
        parser.error("--paper-protocol requires normal, base_color and shaded channels")
    if args.paper_protocol and any((args.output_dir / name).exists() for name in ("summary.json", "per_case.jsonl", "validation_failure.json")):
        parser.error("paper runs require a fresh output directory to avoid stale formal summaries")
    cases = list(iter_jsonl(args.manifest))
    all_rows: list[dict[str, Any]] = []
    planned_rows: list[dict[str, Any]] = []
    for baseline in args.baseline or ["trellis2", "pixal3d"]:
        planned = build_paired_render_rows(
            cases,
            args.render_root,
            baseline=baseline,
            channels=args.channels or (PAPER_CHANNELS if args.paper_protocol else CHANNELS),
            views=args.views,
            variants=args.variants,
        )
        planned_rows.extend(planned)
        all_rows.extend(evaluate_paired_render_rows(planned, enable_lpips=args.lpips, device=args.device, allow_unverified_camera=args.allow_unverified_camera))
    for row in all_rows:
        if row.get("baseline") == "fixture":
            row["fixture_only"] = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "per_case.jsonl", finite_json(all_rows))
    summary = summarize_render_rows(all_rows)
    # Failure diagnostics remain available, but never overwrite a formal
    # summary with incomplete values. Use a fresh output directory per run.
    if args.paper_protocol:
        try:
            summary["paper_validation"] = validate_paper_render_rows(all_rows, views=args.views, expected_rows=planned_rows)
        except ValueError as error:
            (args.output_dir / "validation_failure.json").write_text(json.dumps({"status": "failed", "reason": str(error), "diagnostics": finite_json(summary)}, indent=2), encoding="utf-8")
            raise SystemExit(f"paper PBR validation failed; no summary generated: {error}") from error
    (args.output_dir / "summary.json").write_text(json.dumps(finite_json(summary), indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "capability_audit.json").write_text(json.dumps(finite_json(audit_render_capability(planned_rows)), indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "run_metadata.json").write_text(json.dumps({
        "schema_version": "official_aligned_v1",
        "component": "pbr_metrics",
        "camera_contract": "shared_verified_required_unless_allow_unverified_camera",
        "normalization": "none",
        "arguments": {key: str(value) for key, value in vars(args).items()},
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "summary": str((args.output_dir / "summary.json").absolute())}, indent=2))


if __name__ == "__main__":
    main()
