from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import SCHEMA_VERSION
from .common import atomic_write_json, input_fingerprint, read_record, record_path, stage_rows, utc_now


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("empty_subject_mask")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _pad_box(box: tuple[int, int, int, int], shape: tuple[int, int], fraction: float = 0.12):
    x1, y1, x2, y2 = box
    pad = max(4, round(max(x2 - x1, y2 - y1) * fraction))
    height, width = shape
    return max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def _edge(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    eroded = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return mask & ~eroded


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(mask, radius, constant_values=False)
    result = np.zeros_like(mask)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            result |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return result


def _tolerant_edge_overlap(left: np.ndarray, right: np.ndarray) -> float:
    left_edge, right_edge = _edge(left), _edge(right)
    if not left_edge.any() or not right_edge.any():
        return 0.0
    radius = max(2, round((left.shape[0] ** 2 + left.shape[1] ** 2) ** 0.5 * 0.005))
    left_hit = np.logical_and(left_edge, _dilate(right_edge, radius)).sum() / left_edge.sum()
    right_hit = np.logical_and(right_edge, _dilate(left_edge, radius)).sum() / right_edge.sum()
    return float((left_hit + right_hit) / 2)


def _geometry(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_box, right_box = _bbox(left), _bbox(right)
    lx = (left_box[0] + left_box[2]) / 2
    ly = (left_box[1] + left_box[3]) / 2
    rx = (right_box[0] + right_box[2]) / 2
    ry = (right_box[1] + right_box[3]) / 2
    diagonal = (left.shape[0] ** 2 + left.shape[1] ** 2) ** 0.5
    height, width = right.shape
    border_sides = [
        right_box[0] <= 1, right_box[1] <= 1,
        right_box[2] >= width - 1, right_box[3] >= height - 1,
    ]
    return {
        "mask_iou": round(_mask_iou(left, right), 6),
        "edge_overlap_tolerant": round(_tolerant_edge_overlap(left, right), 6),
        "bbox_center_shift": round(((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5 / diagonal, 6),
        "bbox_area_ratio": round(float(right.sum() / max(int(left.sum()), 1)), 6),
        "right_border_touch_sides": int(sum(border_sides)),
        "right_touches_opposite_borders": bool(
            (border_sides[0] and border_sides[2]) or (border_sides[1] and border_sides[3])
        ),
    }


class VisionModels:
    def __init__(self, args):
        import torch
        from torchvision import transforms
        self.torch = torch
        self.device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        torch.hub.set_dir(str(args.dino_repo.parent))
        self.dino = torch.hub.load(str(args.dino_repo), "dinov2_vitb14", source="local", pretrained=False)
        try:
            state = torch.load(args.dino_checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(args.dino_checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
        missing, _ = self.dino.load_state_dict(state, strict=False)
        if len(missing) > 20:
            raise RuntimeError(f"DINO checkpoint mismatch: {len(missing)} missing keys")
        self.dino = self.dino.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if args.sam_python_packages and str(args.sam_python_packages) not in sys.path:
            # Append (do not prepend): keep torch/torchvision from the active
            # audit environment while filling only SAM2's Hydra/OmegaConf deps.
            sys.path.append(str(args.sam_python_packages))
        if str(args.sam_repo) not in sys.path:
            sys.path.insert(0, str(args.sam_repo))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam = build_sam2(args.sam_config, str(args.sam_checkpoint), device=self.device)
        self.sam = SAM2ImagePredictor(sam)

    def segment(self, image: Image.Image, prompt_box) -> tuple[np.ndarray, float]:
        self.sam.set_image(np.asarray(image.convert("RGB")))
        masks, scores, _ = self.sam.predict(
            point_coords=None, point_labels=None, box=np.asarray(prompt_box, dtype=np.float32),
            multimask_output=True,
        )
        index = int(np.argmax(scores))
        return np.asarray(masks[index], dtype=bool), float(scores[index])

    def embedding(self, image: Image.Image, box) -> Any:
        crop = image.crop(tuple(box)).convert("RGB")
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            autocast = self.torch.autocast("cuda", dtype=self.torch.bfloat16) if str(self.device).startswith("cuda") else contextlib.nullcontext()
            with autocast:
                value = self.dino.forward_features(tensor)["x_norm_clstoken"][0].float()
            return value / value.norm().clamp_min(1e-12)


def _threshold_geometry(metrics, cfg, prefix, scores, reject, review):
    confident = min(scores) >= cfg["score_review"]
    if not confident:
        return
    if metrics["mask_iou"] < cfg[f"{prefix}_iou_reject"]:
        review.append(f"{prefix}_mask_iou_too_low")
    elif metrics["mask_iou"] < cfg[f"{prefix}_iou_review"]:
        review.append(f"{prefix}_mask_iou_gray_zone")
    if metrics["bbox_center_shift"] > cfg["center_shift_reject"] and metrics["mask_iou"] < cfg[f"{prefix}_iou_review"]:
        reject.append(f"{prefix}_bbox_center_shift_too_large")
    elif metrics["bbox_center_shift"] > cfg["center_shift_review"]:
        review.append(f"{prefix}_bbox_center_shift_gray_zone")
    ratio = metrics["bbox_area_ratio"]
    if (ratio < cfg["area_ratio_reject_low"] or ratio > cfg["area_ratio_reject_high"]) and metrics["mask_iou"] < cfg[f"{prefix}_iou_review"]:
        reject.append(f"{prefix}_bbox_area_ratio_out_of_range")
    elif ratio < cfg["area_ratio_review_low"] or ratio > cfg["area_ratio_review_high"]:
        review.append(f"{prefix}_bbox_area_ratio_gray_zone")
    if metrics["edge_overlap_tolerant"] < cfg["edge_iou_review"] and metrics["mask_iou"] < cfg[f"{prefix}_iou_review"]:
        review.append(f"{prefix}_edge_overlap_low")


def audit_vision_pair(row: dict[str, Any], program: dict[str, Any], models: VisionModels, thresholds) -> dict[str, Any]:
    reject: list[str] = []
    review: list[str] = []
    result: dict[str, Any] = {}
    try:
        source = Image.open(program["paths"]["source"]).convert("RGB")
        gt = Image.open(program["paths"]["gt"]).convert("RGB")
        degraded = Image.open(program["paths"]["degraded"]).convert("RGB")
        source_array = np.asarray(source)
        source_mask_native = source_array.max(axis=2) > 12
        source_box_native = _bbox(source_mask_native)
        if source.size == gt.size:
            source_mask = source_mask_native
            source_box_for_gt = source_box_native
        else:
            source_mask = np.asarray(
                Image.fromarray(source_mask_native.astype(np.uint8) * 255).resize(gt.size, Image.Resampling.NEAREST)
            ) > 0
            sx, sy = gt.width / source.width, gt.height / source.height
            source_box_for_gt = (
                round(source_box_native[0] * sx), round(source_box_native[1] * sy),
                round(source_box_native[2] * sx), round(source_box_native[3] * sy),
            )
        gt_mask, gt_score = models.segment(gt, _pad_box(source_box_for_gt, (gt.height, gt.width)))
        gt_box = _bbox(gt_mask)
        degraded_mask, degraded_score = models.segment(degraded, _pad_box(gt_box, (degraded.height, degraded.width), 0.20))
        sg = _geometry(source_mask, gt_mask)
        gd = _geometry(gt_mask, degraded_mask)
        source_embedding = models.embedding(source, source_box_native)
        gt_embedding = models.embedding(gt, gt_box)
        degraded_embedding = models.embedding(degraded, _bbox(degraded_mask))
        dino = {
            "source_gt_similarity": round(float((source_embedding * gt_embedding).sum().item()), 6),
            "gt_degraded_similarity": round(float((gt_embedding * degraded_embedding).sum().item()), 6),
        }
        sam_cfg, dino_cfg = thresholds["sam"], thresholds["dino"]
        if min(gt_score, degraded_score) < sam_cfg["score_review"]:
            review.append("sam_prediction_low_confidence")
        _threshold_geometry(sg, sam_cfg, "source_gt", (1.0, gt_score), reject, review)
        if str(row.get("degradation_type")) != "crop":
            _threshold_geometry(gd, sam_cfg, "gt_degraded", (gt_score, degraded_score), reject, review)
        if dino["source_gt_similarity"] < dino_cfg["source_gt_reject"] and gt_score >= sam_cfg["score_review"] and sg["mask_iou"] < sam_cfg["source_gt_iou_reject"]:
            reject.append("source_gt_dino_identity_too_low")
        elif dino["source_gt_similarity"] < dino_cfg["source_gt_review"]:
            review.append("source_gt_dino_identity_gray_zone")
        if dino["gt_degraded_similarity"] < dino_cfg["gt_degraded_reject"] and min(gt_score, degraded_score) >= sam_cfg["score_review"] and gd["mask_iou"] < sam_cfg["gt_degraded_iou_reject"]:
            reject.append("gt_degraded_dino_identity_too_low")
        elif dino["gt_degraded_similarity"] < dino_cfg["gt_degraded_review"]:
            review.append("gt_degraded_dino_identity_gray_zone")
        hard_reject: list[str] = []
        if (sg["right_touches_opposite_borders"]
                and sg["bbox_area_ratio"] >= sam_cfg["severe_out_of_frame_area_ratio"]):
            hard_reject.append("vision_gt_subject_severely_out_of_frame")
        result = {
            "sam": {"gt_score": round(gt_score, 6), "degraded_score": round(degraded_score, 6)},
            "source_to_gt": sg,
            "gt_to_degraded": gd,
            "dino": dino,
        }
    except Exception as exc:
        review.append(f"vision_exception:{type(exc).__name__}:{exc}")
    # DINO/SAM/bbox/edge values are retained as diagnostic evidence. Their
    # Pilot gray zones generated too many false Reviews, so they no longer
    # determine acceptance without a semantic subject/background failure.
    diagnostics = sorted(set(reject + review))
    runtime_reviews = [reason for reason in diagnostics if reason.startswith("vision_exception:")]
    hard_reject = locals().get("hard_reject", [])
    status = "reject" if hard_reject else ("review" if runtime_reviews else "pass")
    return {
        "schema_version": SCHEMA_VERSION, "stage": "vision", "completed_at": utc_now(),
        "pair_id": row["pair_id"], "sha256": row.get("sha256"), "task_id": row.get("task_id"),
        "status": status, "reject_reasons": hard_reject, "hard_reject_reasons": hard_reject,
        "review_reasons": runtime_reviews,
        "diagnostic_flags": diagnostics,
        **result, "threshold_version": thresholds["schema_version"],
        "input_fingerprint": input_fingerprint(
            row, stage="vision", extra={"threshold": thresholds["schema_version"]}
        ),
    }


def run_vision(args, thresholds) -> dict[str, int]:
    rows = list(stage_rows(args.manifest, args.shard_index, args.shard_count, args.limit))
    pending = []
    counters = {"processed": 0, "skipped": 0, "pass": 0, "review": 0, "reject": 0, "dependency_reject": 0}
    for row in rows:
        existing = read_record(args.output_root, "vision", row["pair_id"])
        fingerprint = input_fingerprint(
            row, stage="vision", extra={"threshold": thresholds["schema_version"]}
        )
        if (not getattr(args, "force", False) and existing
                and existing.get("schema_version") == SCHEMA_VERSION
                and existing.get("input_fingerprint") == fingerprint):
            counters["skipped"] += 1
            counters[existing["status"]] += 1
            continue
        program = read_record(args.output_root, "programmatic", row["pair_id"])
        if not program or program.get("status") == "reject":
            counters["dependency_reject"] += 1
            continue
        pending.append((row, program))
    if not pending:
        return counters
    models = VisionModels(args)
    for row, program in pending:
        record = audit_vision_pair(row, program, models, thresholds)
        atomic_write_json(record_path(args.output_root, "vision", row["pair_id"]), record)
        counters["processed"] += 1
        counters[record["status"]] += 1
    return counters
