from __future__ import annotations

import argparse
import json
import os
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).expanduser().resolve()
DEFAULT_MODEL_CACHE = Path(os.environ.get("GATE3DGEN_MODEL_CACHE", _PROJECT_ROOT / "checkpoints/cache"))
os.environ.setdefault("HF_HOME", str(DEFAULT_MODEL_CACHE / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(DEFAULT_MODEL_CACHE / "torch"))
from PIL import Image
from scipy import ndimage
try:
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity
except ImportError:  # Keep the dependency-light smoke evaluator runnable.
    def rgb2gray(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)
        return image[..., 0] * 0.2125 + image[..., 1] * 0.7154 + image[..., 2] * 0.0721

    def structural_similarity(
        target: np.ndarray,
        prediction: np.ndarray,
        *,
        channel_axis: int | None = None,
        data_range: float = 1.0,
        win_size: int | None = None,
    ) -> float:
        """Global SSIM fallback used only when scikit-image is unavailable."""
        target = np.asarray(target, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        if channel_axis is not None:
            target = np.moveaxis(target, channel_axis, -1)
            prediction = np.moveaxis(prediction, channel_axis, -1)
            values = [structural_similarity(target[..., i], prediction[..., i], data_range=data_range) for i in range(target.shape[-1])]
            return float(np.mean(values))
        mean_target, mean_prediction = target.mean(), prediction.mean()
        var_target, var_prediction = target.var(), prediction.var()
        covariance = np.mean((target - mean_target) * (prediction - mean_prediction))
        c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
        return float(((2 * mean_target * mean_prediction + c1) * (2 * covariance + c2)) / ((mean_target**2 + mean_prediction**2 + c1) * (var_target + var_prediction + c2)))

from .io import finite_json, iter_jsonl, write_jsonl
from .statistics import clustered_mean_ci, grouped_metric_summary
from .provenance import write_run_metadata


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = rgb2gray(image)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    return np.hypot(gx, gy)


def _basic_arrays(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: prediction={pred.shape}, target={gt.shape}")
    error = pred - gt
    mse = float(np.mean(error**2))
    grad_pred, grad_gt = _gradient_magnitude(pred), _gradient_magnitude(gt)
    grad_mae = float(np.mean(np.abs(grad_pred - grad_gt)))
    # Edge correlation guards against blur while remaining deterministic and dependency-light.
    gp, gg = grad_pred.ravel(), grad_gt.ravel()
    edge_corr = float(np.corrcoef(gp, gg)[0, 1]) if gp.std() > 0 and gg.std() > 0 else 0.0
    win_size = min(7, pred.shape[0], pred.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        raise ValueError(f"image too small for SSIM: {pred.shape}")
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "psnr_db": math.inf if mse == 0 else -10.0 * math.log10(mse),
        "ssim": float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0, win_size=win_size)),
        "gradient_mae": grad_mae,
        "edge_correlation": edge_corr,
    }


def _foreground_crop_arrays(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tight, background-zeroed arrays for foreground metrics.

    ``mask`` may be a boolean array, an 8-bit alpha/mask, or a floating-point
    alpha in ``[0, 1]``.  The returned crop keeps the original image-space
    colors inside the binary foreground and zeros pixels outside it.  Keeping
    this operation in array space lets learned metrics (LPIPS/DINO) use the
    exact same foreground protocol as the deterministic metrics.
    """

    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: prediction={pred.shape}, target={gt.shape}")
    if pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError(f"expected RGB arrays, got prediction={pred.shape}")
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., -1]
    if mask_array.shape != pred.shape[:2]:
        raise ValueError(f"mask shape mismatch: mask={mask_array.shape}, image={pred.shape[:2]}")
    mask_float = mask_array.astype(np.float32, copy=False)
    if mask_float.size and float(mask_float.max()) > 1.0:
        mask_float = mask_float / 255.0
    binary = mask_float >= float(threshold)
    ys, xs = np.where(binary)
    if not len(xs):
        raise ValueError("empty foreground mask")
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    crop_mask = binary[y0:y1, x0:x1, None].astype(np.float32)
    return pred[y0:y1, x0:x1] * crop_mask, gt[y0:y1, x0:x1] * crop_mask, binary


def basic_metrics(prediction: Path, target: Path) -> dict[str, float]:
    return _basic_arrays(_rgb(prediction), _rgb(target))


def object_metrics(prediction: Path, target: Path, mask_path: Path) -> dict[str, float]:
    """Metrics on the tight object crop with background zeroed by the GT mask."""
    pred, gt = _rgb(prediction), _rgb(target)
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 127
    pred_crop, gt_crop, _ = _foreground_crop_arrays(pred, gt, mask, threshold=0.5)
    return {f"object_{name}": value for name, value in _basic_arrays(pred_crop, gt_crop).items()}


def object_metrics_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    threshold: float = 0.8,
) -> dict[str, float]:
    """Compute deterministic ``object_*`` metrics from an in-memory mask.

    This is the foreground counterpart to :func:`object_metrics` and is used
    by the RMBG2-aware evaluator so no intermediate PNG quantization is part
    of the numeric calculation.
    """

    pred_crop, gt_crop, _ = _foreground_crop_arrays(prediction, target, mask, threshold=threshold)
    return {f"object_{name}": value for name, value in _basic_arrays(pred_crop, gt_crop).items()}


class LearnedMetrics:
    def __init__(
        self,
        device: str,
        enable_lpips: bool,
        enable_dino: bool,
        dino_model: str | Path | None = None,
    ):
        self.device = device
        self.lpips_model = None
        self.dino_model = None
        self.dino_processor = None
        if enable_lpips:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex").eval().to(device)
        if enable_dino:
            from transformers import AutoImageProcessor, AutoModel

            name = str(dino_model or "facebook/dinov2-large")
            self.dino_processor = AutoImageProcessor.from_pretrained(name)
            self.dino_model = AutoModel.from_pretrained(name).eval().to(device)

    def call_arrays(self, prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
        import torch

        result: dict[str, float] = {}
        prediction = np.asarray(prediction, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if prediction.shape != target.shape:
            raise ValueError(f"shape mismatch: prediction={prediction.shape}, target={target.shape}")
        if prediction.ndim != 3 or prediction.shape[2] != 3:
            raise ValueError(f"expected RGB arrays, got prediction={prediction.shape}")
        images = [
            Image.fromarray(np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8), mode="RGB")
            for array in (prediction, target)
        ]
        try:
            if self.lpips_model is not None:
                tensors = []
                for image in images:
                    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
                    tensors.append(torch.from_numpy(array).permute(2, 0, 1)[None].to(self.device))
                with torch.inference_mode():
                    result["lpips_alex"] = float(self.lpips_model(*tensors).item())
            if self.dino_model is not None:
                inputs = self.dino_processor(images=images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    tokens = self.dino_model(**inputs).last_hidden_state[:, 0]
                    tokens = torch.nn.functional.normalize(tokens, dim=-1)
                result["dino_cosine"] = float((tokens[0] * tokens[1]).sum().item())
        finally:
            for image in images:
                image.close()
        return result

    def __call__(self, prediction: Path, target: Path) -> dict[str, float]:
        images = [_rgb(path) for path in (prediction, target)]
        return self.call_arrays(images[0], images[1])


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[f"task:{row['task_id']}"].append(row)
    output = {}
    for name, items in groups.items():
        metric_names = sorted({key for item in items for key in item["metrics"]})
        output[name] = grouped_metric_summary(items, metric_names)
    return output


def paired_improvement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["variant"]] = row["metrics"]
        by_case[row["case_id"]]["asset_id"] = row["asset_id"]
        by_case[row["case_id"]]["task_id"] = row["task_id"]
    lower_is_better = {
        "mae", "mse", "gradient_mae", "lpips_alex",
        "object_mae", "object_mse", "object_gradient_mae", "object_lpips_alex",
    }
    pairs = [value for value in by_case.values() if "degraded" in value and "restored" in value]
    names = sorted({name for pair in pairs for name in pair["degraded"] if name in pair["restored"]})
    output = {"count": len(pairs), "positive_means_restoration_is_better": True, "metrics": {}}
    for name in names:
        sign = -1.0 if name in lower_is_better else 1.0
        gain_items = [
            {"asset_id": pair["asset_id"], "gain": sign * (pair["restored"][name] - pair["degraded"][name])}
            for pair in pairs
        ]
        gain = clustered_mean_ci(gain_items, lambda item: item["gain"])
        wins = clustered_mean_ci(gain_items, lambda item: float(item["gain"] > 0))
        output["metrics"][name] = {
            "mean_gain": gain["mean"], "gain_ci95": gain["ci95"],
            "win_rate": wins["mean"], "win_rate_ci95": wins["ci95"],
            "assets": gain["clusters"],
        }
    return output


def foreground_metrics(pred, gt, reference_alpha, prediction_alpha, learned, threshold=0.8):
    """Original evaluate_dual2d protocol: GT support for LPIPS; independent alpha IoU."""
    from .mask_consistency import mask_metrics
    pred_crop, gt_crop, _ = _foreground_crop_arrays(pred, gt, reference_alpha, threshold=threshold)
    metrics = object_metrics_arrays(pred, gt, reference_alpha, threshold=threshold)
    if learned.lpips_model is not None or learned.dino_model is not None:
        metrics.update({f"object_{key}": value for key, value in learned.call_arrays(pred_crop, gt_crop).items()})
    metrics["foreground_mask_iou"] = mask_metrics(prediction_alpha, reference_alpha, threshold=threshold)["mask_iou"]
    return metrics


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate restored/degraded images against GT")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--dino", action="store_true")
    parser.add_argument("--allow-missing-restored", action="store_true")
    parser.add_argument("--paper-protocol", action="store_true", help="Require Table 2 foreground LPIPS/IoU and full-image metrics")
    parser.add_argument("--foreground-model", help="External local RMBG-2.0 model; segment each condition independently")
    parser.add_argument("--foreground-mask-model", help="Identity of the RMBG-2.0 model used for cached foreground_masks")
    parser.add_argument("--foreground-threshold", type=float, default=0.8)
    args = parser.parse_args(argv)
    if not 0 < args.foreground_threshold <= 1:
        parser.error("foreground threshold must be in (0, 1]")
    if args.paper_protocol and (not args.lpips or not (args.foreground_model or args.foreground_mask_model)):
        parser.error("Paper protocol requires --lpips and --foreground-model or --foreground-mask-model")
    if args.paper_protocol and args.foreground_threshold != 0.8:
        parser.error("Table 2 requires foreground threshold 0.8; omit --paper-protocol for another experiment")
    if args.paper_protocol and structural_similarity.__module__ == __name__:
        raise RuntimeError("Paper SSIM requires scikit-image")
    segmenter = None
    if args.foreground_model:
        from .mask_consistency import build_segmenter
        segmenter = build_segmenter(args.foreground_model, args.device)
    learned = LearnedMetrics(args.device, args.lpips, args.dino)
    results = []
    def alpha(case, variant, path):
        if segmenter is not None:
            with Image.open(path) as image:
                value = segmenter(image.convert("RGB"))
            destination = args.output_dir / "foreground_masks" / case["case_id"] / f"{variant}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(value, dtype=np.uint8)).save(destination)
            return value, str(destination.resolve())
        mask_path = case.get("foreground_masks", {}).get(variant)
        if not mask_path:
            raise ValueError(f"Missing cached {variant} foreground mask for {case['case_id']}")
        with Image.open(mask_path) as image:
            return np.asarray(image.convert("L")), str(Path(mask_path).resolve())
    for case in iter_jsonl(args.manifest):
        if Path(str(case["case_id"])).name != str(case["case_id"]) or str(case["case_id"]) in {".", ".."}:
            raise ValueError("Unsafe case ID")
        reference = None
        if args.foreground_model or args.foreground_mask_model:
            reference, reference_path = alpha(case, "gt", case["gt"])
        for variant in ("degraded", "restored"):
            path = case.get(variant)
            if not path or not Path(path).is_file():
                if variant == "restored" and args.allow_missing_restored:
                    continue
                raise FileNotFoundError(f"{case['case_id']}:{variant}: {path}")
            metrics = basic_metrics(Path(path), Path(case["gt"]))
            mask_path = Path(case.get("object_mask") or "")
            if mask_path.is_file():
                metrics.update(object_metrics(Path(path), Path(case["gt"]), mask_path))
            metrics.update(learned(Path(path), Path(case["gt"])))
            provenance = {}
            if reference is not None:
                prediction_alpha, prediction_path = alpha(case, variant, path)
                pred, gt = _rgb(Path(path)), _rgb(Path(case["gt"]))
                if np.asarray(prediction_alpha).shape[:2] != pred.shape[:2]:
                    raise ValueError("Prediction alpha dimensions differ from image")
                metrics.update(foreground_metrics(pred, gt, reference, prediction_alpha, learned, args.foreground_threshold))
                provenance = {"foreground_mask_reference": "clean_gt_rmbg2", "foreground_mask_model": args.foreground_model or args.foreground_mask_model,
                              "foreground_mask_threshold": args.foreground_threshold, "foreground_reference_mask_path": reference_path,
                              "foreground_mask_path": prediction_path, "lpips_backbone": "alex", "foreground_background": "black", "foreground_crop": "tight_gt_bbox"}
            results.append({
                "case_id": case["case_id"], "asset_id": case["asset_id"],
                "task_id": case["task_id"], "variant": variant, "metrics": metrics,
                **provenance,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "per_case.jsonl", finite_json(results))
    summary = {variant: aggregate([r for r in results if r["variant"] == variant]) for variant in ("degraded", "restored")}
    summary["restored_vs_degraded"] = paired_improvement(results)
    summary["protocol"] = {"paper_table": 2 if args.paper_protocol else None,
                           "columns": {"LPIPS": "object_lpips_alex", "PSNR": "psnr_db", "SSIM": "ssim",
                                       "Grad": "gradient_mae", "IoU": "foreground_mask_iou"}}
    summary["restored_vs_degraded_by_task"] = {
        task: paired_improvement([row for row in results if str(row["task_id"]) == task])
        for task in sorted({str(row["task_id"]) for row in results})
    }
    (args.output_dir / "summary.json").write_text(json.dumps(finite_json(summary), indent=2, sort_keys=True), encoding="utf-8")
    write_run_metadata(
        args.output_dir / "run_metadata.json", component="metrics2d",
        arguments=vars(args), manifest=args.manifest,
        extra={
            "metric_directions": {
                "lower": ["mae", "mse", "gradient_mae", "lpips_alex", "object_mae", "object_mse", "object_gradient_mae"],
                "higher": ["psnr_db", "ssim", "edge_correlation", "dino_cosine", "object_psnr_db", "object_ssim", "object_edge_correlation"],
            },
            "object_metric_protocol": "tight GT object-mask crop; pixels outside mask zeroed",
        },
    )
    print(json.dumps(finite_json(summary.get("restored") or summary["degraded"]), indent=2))


if __name__ == "__main__":
    main()
