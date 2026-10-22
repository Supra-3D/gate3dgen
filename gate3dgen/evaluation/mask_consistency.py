"""Segmentation-mask diagnostics for restoration and image-to-3D inputs.

The metrics in this module deliberately operate on *uncropped* image-space
alpha mattes.  Measuring after Pixal3D's tight crop would hide translations,
scale changes, and missing foreground regions introduced by segmentation.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageFilter

from .benchmark_manifest import rebase_path
from .io import iter_jsonl, variant_path, write_jsonl
from .rembg_adapter import _safe_model_path, ensure_rmbg_compat_modules

LOCAL_WORKSPACE = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).expanduser().resolve()
# This workspace contains a complete, approved local RMBG2 checkpoint.  Keep
# the path explicit so an audit cannot silently fall back to RMBG-1.4 or make
# an unauthorised network fetch; callers can still override it with --model.
DEFAULT_RMBG_MODEL = str(Path(os.environ.get("GATE3DGEN_RMBG_MODEL", LOCAL_WORKSPACE / "checkpoints/RMBG-2.0")))
DEFAULT_VARIANTS = ("clean_gt", "restored_s03", "restored_s20", "restored_s40")
DEFAULT_VISUAL_BOUNDARY_WIDTH = 3


def _local_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    return rebase_path(str(value), LOCAL_WORKSPACE)


def _binary(mask: np.ndarray, threshold: float) -> np.ndarray:
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 3:
        array = array[..., -1]
    if array.size and float(array.max()) > 1.0:
        array = array / 255.0
    return array >= float(threshold)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_iou(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float:
    if a is None or b is None:
        return 1.0 if a is None and b is None else 0.0
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return 1.0 if union == 0 else float(intersection / union)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary square dilation without scipy/opencv runtime dependencies."""

    radius = max(0, int(radius))
    if radius == 0:
        return mask.copy()
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = mask.copy()
    padded = np.pad(mask, 1, mode="constant")
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return mask & ~eroded


def _visual_boundary(mask: np.ndarray, width: int) -> np.ndarray:
    """Return a display-only, square-thickened foreground boundary.

    The mask itself is never changed by this helper.  ``width`` is the
    requested line width in pixels; the asymmetric padding for even widths
    keeps the resulting square footprint exactly ``width`` pixels wide where
    the boundary is away from the image edge.
    """

    width = int(width)
    if width < 1:
        raise ValueError(f"visual boundary width must be >= 1, got {width}")
    boundary = _boundary(np.asarray(mask, dtype=bool))
    if width == 1 or not boundary.any():
        return boundary
    before = width // 2
    after = width - before - 1
    padded = np.pad(boundary, ((before, after), (before, after)), mode="constant")
    thick = np.zeros_like(boundary, dtype=bool)
    for dy in range(width):
        for dx in range(width):
            thick |= padded[dy : dy + boundary.shape[0], dx : dx + boundary.shape[1]]
    return thick


def mask_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    threshold: float = 0.8,
) -> dict[str, float]:
    """Return the two retained GT-mask metrics in uncropped image space."""

    pred = _binary(prediction, threshold)
    ref = _binary(reference, threshold)
    if pred.shape != ref.shape:
        raise ValueError(f"mask shape mismatch: prediction={pred.shape}, reference={ref.shape}")
    intersection = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    pred_area, ref_area = int(pred.sum()), int(ref.sum())
    iou = 1.0 if union == 0 else intersection / union
    if pred_area:
        py, px = np.argwhere(pred).mean(axis=0)
    else:
        py = px = float("nan")
    if ref_area:
        ry, rx = np.argwhere(ref).mean(axis=0)
    else:
        ry = rx = float("nan")
    diagonal = max(1.0, math.hypot(*pred.shape))
    centroid_shift = (
        float(math.hypot(px - rx, py - ry) / diagonal)
        if pred_area and ref_area
        else (0.0 if pred_area == ref_area == 0 else 1.0)
    )
    return {
        "mask_iou": float(iou),
        "centroid_shift_normalized": centroid_shift,
    }


def _alpha_array(image: Image.Image) -> np.ndarray:
    rgba = image.convert("RGBA")
    return np.asarray(rgba.getchannel("A"), dtype=np.uint8)


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        if image.mode == "RGBA":
            return _alpha_array(image)
        return np.asarray(image.convert("L"), dtype=np.uint8)


def image_diagnostics(image: Image.Image) -> dict[str, float | int | bool]:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {
        "mean_rgb": float(array.mean()),
        "max_rgb": int(array.max()),
        "nonblack_pixel_fraction": float((array.max(axis=2) > 5).mean()),
        "all_black": bool(array.max() <= 5),
    }


def save_mask_visuals(
    image: Image.Image,
    alpha: np.ndarray,
    output_stem: Path,
    threshold: float = 0.8,
    reference_alpha: np.ndarray | None = None,
    visual_boundary_width: int = DEFAULT_VISUAL_BOUNDARY_WIDTH,
) -> None:
    """Write alpha, cutout, and display overlays for one mask.

    ``visual_boundary_width`` only affects the generated ``*_overlay.png``
    and ``*_compare.png`` files.  It is deliberately not passed to
    :func:`mask_metrics`, so changing it cannot change any reported metric.
    """

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    alpha_image = Image.fromarray(np.asarray(alpha, dtype=np.uint8), mode="L")
    alpha_image.save(output_stem.with_name(output_stem.name + "_alpha.png"))
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha_image)
    rgba.save(output_stem.with_name(output_stem.name + "_cutout.png"))

    overlay = image.convert("RGB").copy()
    boundary = _visual_boundary(_binary(alpha, threshold), visual_boundary_width)
    pixels = np.asarray(overlay).copy()
    pixels[boundary] = (255, 32, 32)
    Image.fromarray(pixels).save(output_stem.with_name(output_stem.name + "_overlay.png"))
    if reference_alpha is not None:
        reference_boundary = _visual_boundary(_binary(reference_alpha, threshold), visual_boundary_width)
        comparison = np.asarray(image.convert("RGB")).copy()
        comparison[reference_boundary] = (0, 220, 255)
        comparison[boundary] = (255, 32, 32)
        comparison[np.logical_and(boundary, reference_boundary)] = (255, 220, 0)
        Image.fromarray(comparison).save(output_stem.with_name(output_stem.name + "_compare.png"))


def build_segmenter(model_name: str, device: str = "cuda") -> Callable[[Image.Image], np.ndarray]:
    """Load a BiRefNet/RMBG checkpoint through its documented HF interface."""

    # The local RMBG-2.0 remote-code snapshot imports a tiny subset of timm
    # and kornia.  Install the project's offline compatibility modules before
    # importing/loading the Transformers dynamic module; otherwise
    # ``check_imports`` fails in the lean FireRed evaluation environment even
    # though the 3D generation path already handles the same checkpoint.
    ensure_rmbg_compat_modules()
    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    # Match the 3D RMBG adapter's local-only loading contract.  Apart from
    # preventing an accidental Hub request, the temporary safe basename is
    # required by Transformers 4.35 for this snapshot's ``RMBG-2.0`` folder
    # (the hyphen otherwise becomes part of the dynamic Python package name).
    safe_model_name, temporary = _safe_model_path(model_name)
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            safe_model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    model.eval().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def _prediction(result):
        if (
            isinstance(result, (tuple, list))
            and result
            and isinstance(result[0], (tuple, list))
        ):
            return result[0][0]
        prediction = result
        while isinstance(prediction, (tuple, list)):
            prediction = prediction[-1]
        return prediction.sigmoid()

    def segment_many(images: list[Image.Image], batch_size: int = 4) -> list[np.ndarray]:
        outputs: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensor = torch.stack([transform(image.convert("RGB")) for image in batch]).to(device)
            with torch.inference_mode():
                prediction = _prediction(model(tensor)).detach().float().cpu()
            for item, image in zip(prediction, batch):
                alpha = transforms.ToPILImage()(item.squeeze()).resize(image.size, Image.Resampling.BILINEAR)
                outputs.append(np.asarray(alpha, dtype=np.uint8))
        return outputs

    def segment(image: Image.Image) -> np.ndarray:
        return segment_many([image], batch_size=1)[0]

    # Keep a simple callable interface while exposing batching to the evaluator.
    segment.many = segment_many  # type: ignore[attr-defined]
    return segment


def _mean(rows: list[dict], group: str, key: str) -> float | None:
    values = []
    for row in rows:
        group_values = row.get(group)
        if not isinstance(group_values, dict) or key not in group_values:
            continue
        try:
            value = float(group_values[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else None


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _model_label(rows: list[dict], model_name: str | Path | None) -> str:
    """Resolve the model label shown in HTML without assuming a checkpoint."""

    if model_name:
        return str(model_name)
    for row in rows:
        for key in ("model", "model_name", "segmenter_model"):
            value = row.get(key)
            if value:
                return str(value)
    # This is only reached by callers constructing a report without runtime
    # metadata.  The CLI always supplies the actual --model value.
    return DEFAULT_RMBG_MODEL


def build_html_report(
    rows: list[dict],
    output_dir: Path,
    variants: Iterable[str],
    *,
    model_name: str | Path | None = None,
    visual_boundary_width: int = DEFAULT_VISUAL_BOUNDARY_WIDTH,
) -> dict:
    """Write the two-metric mask-audit report and return summaries."""

    variants = list(variants)
    valid = [row for row in rows if row.get("status") == "ok"]
    by_variant: dict[str, list[dict]] = defaultdict(list)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in valid:
        by_variant[row["variant"]].append(row)
        by_case[row["case_id"]].append(row)

    variant_summary: dict[str, dict] = {}
    summary_rows = []
    for variant in variants:
        items = by_variant.get(variant, [])
        values = {
            "valid": len(items),
            "all_black_inputs": sum(bool(row.get("image_diagnostics", {}).get("all_black")) for row in items),
            "mask_iou_to_reference": _mean(items, "metrics", "mask_iou"),
            "centroid_shift_to_reference": _mean(items, "metrics", "centroid_shift_normalized"),
        }
        variant_summary[variant] = values
        summary_rows.append(
            f"<tr><td>{html.escape(variant)}</td><td>{len(items)}</td><td>{values['all_black_inputs']}</td>"
            f"<td>{_fmt(values['mask_iou_to_reference'])}</td>"
            f"<td>{_fmt(values['centroid_shift_to_reference'])}</td></tr>"
        )

    # Keep case ordering deterministic while avoiding a comparison-based
    # ordering that would imply a baseline absent from the four-variant demo.
    case_order = sorted(by_case)
    cards = []
    for case_id in case_order:
        lookup = {row["variant"]: row for row in by_case[case_id]}
        panels = [
            f"<figure><img loading='lazy' src='{html.escape(case_id)}/reference_object_mask_overlay.png'>"
            "<figcaption>object mask 标注（红线）</figcaption></figure>"
        ]
        for variant in variants:
            row = lookup.get(variant)
            if not row:
                panels.append(f"<figure><div class='missing'>missing</div><figcaption>{html.escape(variant)}</figcaption></figure>")
                continue
            metric = row["metrics"]
            panels.append(
                f"<figure><a href='{html.escape(case_id)}/{html.escape(variant)}_cutout.png'>"
                f"<img loading='lazy' src='{html.escape(case_id)}/{html.escape(variant)}_compare.png'></a>"
                f"<figcaption>{html.escape(variant)}<br>GT IoU {metric['mask_iou']:.3f} · "
                f"质心偏移 {metric['centroid_shift_normalized']:.5f}</figcaption></figure>"
            )
        cards.append(
            f"<section><h2>{html.escape(case_id)}</h2><div class='gallery'>{''.join(panels)}</div></section>"
        )

    model_label = html.escape(_model_label(rows, model_name))
    visual_boundary_width = int(visual_boundary_width)
    if visual_boundary_width < 1:
        raise ValueError(f"visual boundary width must be >= 1, got {visual_boundary_width}")
    output_dir.mkdir(parents=True, exist_ok=True)
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Demo mask 一致性审计 · {model_label}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f5f6f8;color:#17202a}}h1{{margin-bottom:6px}}
.note{{max-width:1100px;line-height:1.65}}table{{border-collapse:collapse;background:white;margin:20px 0;width:100%;font-size:13px}}
th,td{{border:1px solid #d9dde3;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left;position:sticky;left:0;background:white}}
section{{background:white;border-radius:10px;padding:14px;margin:18px 0;box-shadow:0 1px 5px #0001}}h2{{font-size:15px;overflow-wrap:anywhere}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}}figure{{margin:0}}img{{width:100%;aspect-ratio:1;object-fit:contain;background:#ddd}}
figcaption{{font-size:12px;line-height:1.5;padding:5px 2px}}.good{{color:#087f23;font-weight:700}}.bad{{color:#c62828;font-weight:700}}.missing{{aspect-ratio:1;display:grid;place-items:center;background:#eee;color:#888}}
</style></head><body><h1>Demo mask 一致性审计 · {model_label}</h1>
<p class='note'>分割模型：{model_label}。边界颜色：manifest 的 object_mask 标注为青色，当前模型预测为红色，像素级重合为黄色；视觉边界线宽为 {visual_boundary_width}px（仅影响 overlay/compare 图像，不影响指标）。阈值为 0.8，全部在未裁剪的原始图像坐标系中测量。仅保留两个指标：预测 mask 对 manifest object_mask 的 GT IoU（越高越好），以及两者质心距离除以图像对角线的归一化质心偏移（越低越好）。点击图片查看透明抠图。逐例按 case_id 排列。</p>
<table><thead><tr><th>版本</th><th>有效</th><th>全黑输入</th><th>GT IoU↑</th><th>归一化质心偏移↓</th></tr></thead>
<tbody>{''.join(summary_rows)}</tbody></table>{''.join(cards)}</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    return variant_summary


def evaluate_manifest(
    manifest: Path,
    output_dir: Path,
    segment: Callable[[Image.Image], np.ndarray],
    variants: Iterable[str],
    *,
    threshold: float,
    batch_size: int = 4,
    case_ids: set[str] | None = None,
    model_name: str | Path | None = None,
    visual_boundary_width: int = DEFAULT_VISUAL_BOUNDARY_WIDTH,
    num_shards: int = 1,
    shard_index: int = 0,
    rows_filename: str = "per_case.jsonl",
) -> list[dict]:
    variants = list(variants)
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")
    rows: list[dict] = []
    for manifest_index, case in enumerate(iter_jsonl(manifest)):
        if manifest_index % num_shards != shard_index:
            continue
        case_id = str(case["case_id"])
        if case_ids and case_id not in case_ids:
            continue
        reference_key = "original" if case.get("dataset_kind") == "wild_no_gt" else "clean_gt"
        reference_path = _local_path(variant_path(case, reference_key) or "")
        if not reference_path.is_file():
            continue
        with Image.open(reference_path) as source:
            reference_image = source.convert("RGB")
        source = case.get("source") or {}
        declared_mask = (
            case.get("object_mask")
            or case.get("object_mask_path")
            or source.get("object_mask")
            or source.get("object_mask_path")
        )
        if declared_mask and _local_path(str(declared_mask)).is_file():
            reference_alpha = _load_mask(_local_path(str(declared_mask)), reference_image.size)
            reference_kind = "manifest_object_mask"
        else:
            reference_alpha = segment(reference_image)
            reference_kind = f"segmented_{reference_key}"
        save_mask_visuals(
            reference_image,
            reference_alpha,
            output_dir / case_id / "reference_object_mask",
            threshold,
            visual_boundary_width=visual_boundary_width,
        )

        case_rows: list[dict] = []
        loaded: list[tuple[str, Path, Image.Image]] = []
        for variant in variants:
            path = _local_path(variant_path(case, variant) or "")
            if not path.is_file():
                case_rows.append({"case_id": case_id, "variant": variant, "status": "unavailable", "reason": "missing_image"})
                continue
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            loaded.append((variant, path, image))
        images = [item[2] for item in loaded]
        infer_many = getattr(segment, "many", None)
        alphas = infer_many(images, batch_size) if infer_many else [segment(image) for image in images]
        for (variant, path, image), alpha in zip(loaded, alphas):
            if alpha.shape != reference_alpha.shape:
                alpha_image = Image.fromarray(alpha).resize(reference_image.size, Image.Resampling.BILINEAR)
                alpha = np.asarray(alpha_image, dtype=np.uint8)
            save_mask_visuals(
                image.resize(reference_image.size),
                alpha,
                output_dir / case_id / variant,
                threshold,
                reference_alpha,
                visual_boundary_width,
            )
            row = {
                "case_id": case_id,
                "variant": variant,
                "status": "ok",
                "image": str(path),
                "reference": str(reference_path),
                "reference_kind": reference_kind,
                "threshold": threshold,
                "image_diagnostics": image_diagnostics(image),
                "metrics": mask_metrics(alpha, reference_alpha, threshold=threshold),
            }
            if model_name:
                row["model"] = str(model_name)
            case_rows.append(row)

        rows.extend(case_rows)
        print(f"mask audit: {case_id} ({len(loaded)}/{len(variants)} variants)", flush=True)
    write_jsonl(output_dir / rows_filename, rows)
    return rows


def write_report_artifacts(
    rows: list[dict],
    output_dir: Path,
    variants: list[str],
    *,
    model_name: str,
    threshold: float,
    visual_boundary_width: int,
) -> dict:
    valid = [row for row in rows if row.get("status") == "ok"]
    variant_summary = build_html_report(
        rows,
        output_dir,
        variants,
        model_name=model_name,
        visual_boundary_width=visual_boundary_width,
    )
    summary = {
        "rows": len(rows),
        "valid": len(valid),
        "model": model_name,
        "threshold": threshold,
        "visual_boundary_width": visual_boundary_width,
        "variants": variant_summary,
        "mean": {
            key: float(np.mean([row["metrics"][key] for row in valid])) if valid else None
            for key in ("mask_iou", "centroid_shift_normalized")
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def merge_shard_rows(output_dir: Path, num_shards: int, variants: list[str]) -> list[dict]:
    """Fail closed while merging disjoint GPU shard row files."""

    variant_order = {variant: index for index, variant in enumerate(variants)}
    rows: list[dict] = []
    for shard_index in range(num_shards):
        path = output_dir / f"per_case.shard_{shard_index:02d}_of_{num_shards:02d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing mask shard: {path}")
        rows.extend(iter_jsonl(path))
    keys = [(str(row.get("case_id")), str(row.get("variant"))) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (case_id, variant) rows across mask shards")
    rows.sort(key=lambda row: (str(row.get("case_id")), variant_order.get(str(row.get("variant")), 999)))
    write_jsonl(output_dir / "per_case.jsonl", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit image-space foreground-mask consistency across restoration steps")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=DEFAULT_RMBG_MODEL,
        help="Local RMBG2/BiRefNet snapshot or Hugging Face model id (default: %(default)s)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--visual-boundary-width",
        type=int,
        default=DEFAULT_VISUAL_BOUNDARY_WIDTH,
        help="Display-only overlay/compare boundary width in pixels (default: %(default)s)",
    )
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--case-id", action="append", dest="case_ids", help="Restrict the audit to selected cases")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true", help="Merge completed shard JSONLs without loading RMBG2")
    parser.add_argument("--expected-cases", type=int, help="Fail unless merge contains this many complete cases")
    args = parser.parse_args()
    if args.visual_boundary_width < 1:
        parser.error("--visual-boundary-width must be >= 1")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    variants = list(args.variants) if args.variants else list(DEFAULT_VARIANTS)
    if args.merge_shards:
        rows = merge_shard_rows(args.output_dir, args.num_shards, variants)
        expected_rows = args.expected_cases * len(variants) if args.expected_cases is not None else None
        if expected_rows is not None and len(rows) != expected_rows:
            parser.error(f"merged row count {len(rows)} != expected {expected_rows}")
        invalid = [row for row in rows if row.get("status") != "ok"]
        if invalid:
            parser.error(f"merged mask audit contains {len(invalid)} unavailable/error rows")
        summary = write_report_artifacts(
            rows,
            args.output_dir,
            variants,
            model_name=args.model,
            threshold=args.threshold,
            visual_boundary_width=args.visual_boundary_width,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    rows_filename = (
        f"per_case.shard_{args.shard_index:02d}_of_{args.num_shards:02d}.jsonl"
        if args.num_shards > 1
        else "per_case.jsonl"
    )
    rows = evaluate_manifest(
        args.manifest,
        args.output_dir,
        build_segmenter(args.model, args.device),
        variants,
        threshold=args.threshold,
        batch_size=args.batch_size,
        case_ids=set(args.case_ids or ()),
        model_name=args.model,
        visual_boundary_width=args.visual_boundary_width,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        rows_filename=rows_filename,
    )
    if args.num_shards > 1:
        summary = {
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "rows": len(rows),
            "valid": sum(row.get("status") == "ok" for row in rows),
            "rows_file": str(args.output_dir / rows_filename),
        }
    else:
        summary = write_report_artifacts(
            rows,
            args.output_dir,
            variants,
            model_name=args.model,
            threshold=args.threshold,
            visual_boundary_width=args.visual_boundary_width,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
