#!/usr/bin/env python3
"""Run controlled albedo degradations with local Qwen and FireRed pipelines."""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).expanduser().resolve()
from . import occlusion_selector as vlm_occlusion_selector

DEFAULT_IMAGE = PROJECT_ROOT / "data/example.png"
DEFAULT_LIBRARY = Path(__file__).resolve().parents[2] / "resources/prompts/gate3dgen_pair_synthesis.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/local_drill_prompt_matrix"
DEFAULT_QWEN = PROJECT_ROOT / "checkpoints/Qwen-Image-Edit-2511"
DEFAULT_FIRERED = PROJECT_ROOT / "checkpoints/FireRed-Image-Edit-1.0"
DEFAULT_FIRERED_11 = PROJECT_ROOT / "checkpoints/FireRed-Image-Edit-1.1"
MODEL_NAMES = ("qwen-image-edit", "firered-image-edit", "firered-image-edit-1.1")
DEFAULT_MODELS = ("qwen-image-edit", "firered-image-edit-1.1")
PROCEDURAL_METHODS = {
    "identity", "procedural_crop", "procedural_blur", "procedural_digital",
    "procedural_white_balance", "procedural_overexposure", "procedural_underexposure",
    "procedural_motion_blur", "procedural_recompression", "procedural_optical_aberration",
}
POSTPROCESS_METHODS = {
    "crop": "procedural_crop",
    "blur": "procedural_blur",
    "optical_blur": "procedural_blur",
    "digital": "procedural_digital",
    "white_balance": "procedural_white_balance",
    "exposure": "procedural_overexposure",
    "underexposure": "procedural_underexposure",
    "motion_blur": "procedural_motion_blur",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def stable_rng(seed: int, task_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def load_tasks(path: Path, seed: int = 42) -> tuple[dict[str, dict[str, Any]], str]:
    library = json.loads(path.read_text(encoding="utf-8"))
    prefix = library["global_real_photo_prefix"].strip()
    suffix = library["global_preservation_suffix"].strip()
    negative_prompt = library["global_negative_prompt"]
    tasks: dict[str, dict[str, Any]] = {}

    def add_task(raw: dict[str, Any], problem_type: str, severity: str) -> None:
        task_id = raw["id"]
        if task_id in tasks:
            raise ValueError(f"Duplicate task id: {task_id}")
        rng = stable_rng(seed, task_id)
        placeholders: dict[str, list[str]] = raw.get("placeholders", {})
        sampled = {name: rng.choice(values) for name, values in placeholders.items()}
        prompt = raw["prompt"].format(**sampled)
        method = raw.get("method", "generative")
        tasks[task_id] = {
            "id": task_id,
            "problem_type": problem_type,
            "severity": severity,
            "method": method,
            "factors": raw.get("factors", problem_type.split("+")),
            "factor_severity": {factor: severity for factor in raw.get("factors", problem_type.split("+"))},
            "parameters": raw.get("parameters", {}),
            "sampled_placeholders": sampled,
            "postprocess": raw.get("postprocess"),
            "scene_rationale": raw.get("scene_rationale"),
            "sampled_reflection_content": sampled.get("reflection_content"),
            "allows_camera_change": raw.get("allows_camera_change", False),
            "enabled_by_default": raw.get("enabled_by_default", True),
            "prompt_prefix": prefix,
            "prompt_body": prompt.strip(),
            "prompt_suffix": suffix,
            "prompt": " ".join((prefix, prompt.strip(), suffix)),
        }

    for problem_type, group in library["degradation_prompts"].items():
        for variant in group.get("variants", []):
            merged = dict(variant)
            merged.setdefault("method", group.get("preferred_method", "generative"))
            merged.setdefault("placeholders", group.get("placeholders", {}))
            add_task(merged, problem_type, variant["severity"])
    for combination in library.get("combination_prompts", []):
        add_task(combination, "+".join(combination["problems"]), combination.get("severity", "combination"))
    return tasks, negative_prompt


def validate_tasks(tasks: dict[str, dict[str, Any]]) -> None:
    """Fail early when a hybrid task cannot describe its deterministic tail."""
    for task in tasks.values():
        if task["method"] == "hybrid":
            postprocess = task.get("postprocess")
            if not isinstance(postprocess, dict) or postprocess.get("type") not in POSTPROCESS_METHODS:
                raise ValueError(f"Hybrid task {task['id']} requires a valid postprocess type")
            if not isinstance(postprocess.get("parameters", {}), dict):
                raise ValueError(f"Hybrid task {task['id']} postprocess parameters must be an object")


def load_pipeline(model_path: Path, device_mode: str):
    import torch
    # Some older PyTorch images predate a few optional namespaces
    # referenced during diffusers 0.36 import.  They are only used for
    # backend discovery here; provide harmless CPU-only shims so the local
    # Qwen/FireRed pipeline can load without changing CUDA execution.
    import types
    if hasattr(torch.utils, "_pytree") and not hasattr(torch.utils._pytree, "register_pytree_node"):
        old_register_pytree_node = torch.utils._pytree._register_pytree_node
        def _register_pytree_node_compat(*args, **kwargs):
            # serialized_type_name was added after the bundled torch 2.1 API.
            kwargs.pop("serialized_type_name", None)
            return old_register_pytree_node(*args, **kwargs)
        torch.utils._pytree.register_pytree_node = _register_pytree_node_compat
    if not hasattr(torch, "xpu"):
        torch.xpu = types.SimpleNamespace(
            empty_cache=lambda: None,
            is_available=lambda: False,
            device_count=lambda: 0,
            manual_seed=lambda seed: None,
        )
    if hasattr(torch, "distributed") and not hasattr(torch.distributed, "device_mesh"):
        torch.distributed.device_mesh = types.SimpleNamespace(DeviceMesh=object)
    if not hasattr(torch, "compiler"):
        torch.compiler = types.SimpleNamespace(is_compiling=lambda: False)
    elif not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False
    # torch 2.1 lacks the enable_gqa keyword used by newer diffusers.  Match
    # the reference behavior by explicitly repeating K/V heads when needed.
    import torch.nn.functional as F
    if not getattr(F.scaled_dot_product_attention, "_qwen_gqa_compat", False):
        _sdpa = F.scaled_dot_product_attention
        def _scaled_dot_product_attention_compat(query, key, value, *args, **kwargs):
            enable_gqa = kwargs.pop("enable_gqa", False)
            if enable_gqa and query.shape[-3] != key.shape[-3]:
                groups = query.shape[-3] // key.shape[-3]
                key = key.repeat_interleave(groups, dim=-3)
                value = value.repeat_interleave(groups, dim=-3)
            return _sdpa(query, key, value, *args, **kwargs)
        _scaled_dot_product_attention_compat._qwen_gqa_compat = True
        F.scaled_dot_product_attention = _scaled_dot_product_attention_compat
    from diffusers import QwenImageEditPlusPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for local image-edit inference")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, local_files_only=True
    )
    if device_mode == "cpu-offload":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=None)
    return pipe


def release_cuda_memory() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_non_vlm_task_prompt(task: dict[str, Any]) -> str:
    """Embed non-VLM task metadata into fluent edit-model instructions."""
    task_id = str(task.get("id", ""))
    parameters = task.get("parameters", {}) or {}
    clauses: list[str] = []

    if task_id.startswith("occlusion_") and "coverage_fraction" in parameters:
        clauses.append(
            f"Target approximately {float(parameters['coverage_fraction']):.0%} coverage of the object."
        )
    elif task_id.startswith("crop_") and "fraction" in parameters:
        clauses.append(
            f"Let approximately {float(parameters['fraction']):.0%} of the object extend beyond the image boundary."
        )
    elif task_id.startswith("overexposure_"):
        gain = parameters.get("gain")
        threshold = parameters.get("clip_threshold")
        if gain is not None and threshold is not None:
            clauses.append(f"Use an exposure increase of about {gain} with clipping beginning near {threshold}.")
    elif task_id.startswith("optical_blur_") and "gaussian_radius" in parameters:
        clauses.append(f"Use an optical blur strength of approximately {parameters['gaussian_radius']}.")
    elif task_id.startswith("digital_degradation_"):
        parts = []
        if "short_edge" in parameters:
            parts.append(f"downsampling to about {parameters['short_edge']} pixels on the short edge")
        if "gaussian_radius" in parameters:
            parts.append(f"softness around {parameters['gaussian_radius']}")
        if "noise_std" in parameters:
            parts.append(f"sensor noise around {parameters['noise_std']}")
        if "jpeg_quality" in parameters:
            parts.append(f"JPEG quality around {parameters['jpeg_quality']}")
        if parts:
            clauses.append("Use " + ", ".join(parts) + ".")
    elif task_id.startswith("underexposure_"):
        gain = parameters.get("gain")
        threshold = parameters.get("black_threshold")
        noise = parameters.get("dark_noise_std")
        if gain is not None:
            text = f"Use an exposure gain around {gain}"
            if threshold is not None:
                text += f", with shadow clipping near {threshold}"
            if noise is not None:
                text += f" and dark-region noise around {noise}"
            clauses.append(text + ".")
    elif task_id.startswith("motion_blur_") and "length" in parameters:
        clauses.append(f"Use a directional blur length of approximately {parameters['length']} pixels.")
    elif task_id.startswith("white_balance_") and "channel_gains" in parameters:
        gains = parameters["channel_gains"]
        if isinstance(gains, (list, tuple)) and len(gains) == 3:
            clauses.append(
                f"Apply the specified global color balance with red, green and blue gains of "
                f"{gains[0]}, {gains[1]} and {gains[2]}."
            )
    elif parameters:
        readable = []
        for name, value in parameters.items():
            label = name.replace("_", " ")
            if isinstance(value, (list, tuple)):
                rendered = ", ".join(str(item) for item in value)
            else:
                rendered = str(value)
            readable.append(f"{label} of {rendered}")
        clauses.append("Use the requested settings, including " + ", ".join(readable) + ".")

    if not task.get("allows_camera_change", False):
        clauses.append("Keep the camera viewpoint unchanged.")
    elif task.get("method") == "generative":
        clauses.append("Allow only a minor natural camera viewpoint change if needed.")

    prompt_parts = [
        task.get("prompt_prefix", "").strip(),
        task.get("prompt_body", task["prompt"]).strip(),
        " ".join(clauses).strip(),
        task.get("prompt_suffix", "").strip(),
    ]
    return " ".join(part for part in prompt_parts if part).strip()


def run_generative(pipe: Any, image: Any, task: dict[str, Any], negative_prompt: str, args: argparse.Namespace):
    import torch
    generator_device = "cpu" if hasattr(pipe, "hf_device_map") else "cuda"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    with torch.inference_mode():
        return pipe(
            image=[image], prompt=task["prompt"], negative_prompt=negative_prompt or " ",
            generator=generator, true_cfg_scale=args.cfg_scale, guidance_scale=1.0,
            num_inference_steps=args.steps, num_images_per_prompt=1,
        ).images[0]


def object_bbox(image: Any) -> tuple[int, int, int, int]:
    import numpy as np
    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    if alpha.min() < 250:
        mask = alpha > 10
    else:
        rgb = rgba[:, :, :3].astype(int)
        corners = np.stack((rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]))
        background = np.median(corners, axis=0)
        mask = np.max(np.abs(rgb - background), axis=2) > 12
    ys, xs = np.where(mask)
    if not len(xs):
        raise RuntimeError("Could not detect object bbox for crop")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def object_pixel_mask(image: Any) -> Any:
    """Boolean mask of foreground object pixels."""
    import numpy as np

    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    if alpha.min() < 250:
        return alpha > 10
    rgb = rgba[:, :, :3].astype(int)
    corners = np.stack((rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]))
    background = np.median(corners, axis=0)
    return np.max(np.abs(rgb - background), axis=2) > 12


def _region_center_in_bbox(bbox: tuple[int, int, int, int], region: str) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    if region.startswith("lower-left"):
        return x0 + 0.25 * (x1 - x0), y0 + 0.75 * (y1 - y0)
    if region.startswith("lower-right"):
        return x0 + 0.75 * (x1 - x0), y0 + 0.75 * (y1 - y0)
    if region.startswith("upper"):
        return cx, y0 + 0.2 * (y1 - y0)
    return cx, cy


def build_occlusion_mask_on_object(
    obj_mask: Any,
    bbox: tuple[int, int, int, int],
    target_fraction: float,
    region: str,
    seed: int,
) -> Any:
    """Build a binary mask covering ~target_fraction of object pixels, anchored in region."""
    import math
    import numpy as np

    height, width = obj_mask.shape
    target_pixels = max(1, int(round(float(target_fraction) * obj_mask.sum())))
    cx, cy = _region_center_in_bbox(bbox, region)

    best_mask = np.zeros_like(obj_mask, dtype=bool)
    for axis_ratio in (1.6, 1.2, 1.0, 0.75, 0.55):
        rx = max(8.0, math.sqrt(target_pixels / math.pi) * axis_ratio)
        ry = max(8.0, math.sqrt(target_pixels / math.pi) / axis_ratio)
        yy, xx = np.ogrid[:height, :width]
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        candidate = obj_mask & ellipse
        if candidate.sum() >= target_pixels * 0.85:
            best_mask = candidate
            break
        if candidate.sum() > best_mask.sum():
            best_mask = candidate

    if best_mask.sum() < target_pixels:
        ys, xs = np.where(obj_mask)
        if len(xs):
            dist = (xs - cx) ** 2 + (ys - cy) ** 2
            order = np.argsort(dist)
            flat = np.zeros(obj_mask.size, dtype=bool)
            idx = order[:target_pixels]
            flat[ys[idx] * width + xs[idx]] = True
            best_mask = flat.reshape(height, width)
    return best_mask


def _shift_bool_mask(mask, dx: int, dy: int):
    import numpy as np
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    x_src0 = max(0, -dx)
    x_src1 = min(w, w - dx) if dx >= 0 else w
    y_src0 = max(0, -dy)
    y_src1 = min(h, h - dy) if dy >= 0 else h
    x_dst0 = max(0, dx)
    x_dst1 = x_dst0 + (x_src1 - x_src0)
    y_dst0 = max(0, dy)
    y_dst1 = y_dst0 + (y_src1 - y_src0)
    if x_src1 > x_src0 and y_src1 > y_src0:
        out[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
    return out


def _dilate_bool_mask(mask, radius: int):
    from PIL import Image, ImageFilter
    if radius <= 0:
        return mask
    size = radius * 2 + 1
    img = Image.fromarray((mask.astype('uint8') * 255))
    return (img.filter(ImageFilter.MaxFilter(size)).convert('L').point(lambda v: 255 if v > 0 else 0).getdata())


def _dilate_mask(mask, radius: int):
    import numpy as np
    from PIL import Image, ImageFilter
    if radius <= 0:
        return mask
    size = radius * 2 + 1
    img = Image.fromarray((mask.astype('uint8') * 255))
    return (np.asarray(img.filter(ImageFilter.MaxFilter(size)).convert('L')) > 0)


def _foreground_expansion_mask(base_mask, region: str, seed: int, strength: float):
    import numpy as np
    rng = np.random.default_rng(seed)
    radius = max(10, int(round(16 + 48 * strength)))
    expanded = _dilate_mask(base_mask, radius)
    h, w = base_mask.shape
    shift_scale = max(8, int(round(radius * 0.45)))
    if region.startswith('lower-left'):
        expanded |= _shift_bool_mask(expanded, -shift_scale, shift_scale)
    elif region.startswith('lower-right'):
        expanded |= _shift_bool_mask(expanded, shift_scale, shift_scale)
    elif region.startswith('upper'):
        expanded |= _shift_bool_mask(expanded, 0, -shift_scale)
    else:
        expanded |= _shift_bool_mask(expanded, int(rng.integers(-shift_scale, shift_scale + 1)), int(rng.integers(-shift_scale, shift_scale + 1)))
    return expanded


def _synthesize_occluder_layer(image_arr, fg_mask, seed: int, severity: str, region: str):
    import numpy as np
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(seed ^ 0x5A17)
    h, w = fg_mask.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    mask = fg_mask.astype(np.float32)
    if mask.max() <= 0:
        return image_arr, mask
    if severity == 'heavy':
        base_palette = np.array([32, 34, 40], dtype=np.float32) + rng.integers(-8, 9, size=3)
    else:
        base_palette = np.array([48, 50, 58], dtype=np.float32) + rng.integers(-12, 13, size=3)
    noise_small = rng.normal(0, 1.0, (max(12, h // 48), max(12, w // 48))).astype(np.float32)
    noise_img = Image.fromarray(((noise_small - noise_small.min()) / (np.ptp(noise_small) + 1e-6) * 255).astype(np.uint8), mode='L')
    noise = np.asarray(noise_img.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gradient = (0.65 + 0.35 * (1.0 - np.clip((xx / max(w - 1, 1)) if region.startswith('left') or region.startswith('lower-left') else (1.0 - xx / max(w - 1, 1)), 0.0, 1.0)))
    if region.startswith('upper'):
        gradient = 0.82 + 0.18 * (yy / max(h - 1, 1))
    elif region.startswith('lower'):
        gradient = 0.82 + 0.18 * (1.0 - yy / max(h - 1, 1))
    stripe = 0.04 * np.sin((xx * (0.06 if severity == 'heavy' else 0.08)) + (yy * 0.03) + rng.uniform(0, 6.28))
    texture = base_palette[None, None, :] * (0.72 + 0.28 * noise[..., None]) * gradient[..., None]
    texture += stripe[..., None] * 255.0
    if severity == 'heavy':
        texture = texture * 0.88
    else:
        texture = texture * 0.96
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4 if severity == 'heavy' else 3))
    alpha_arr = np.asarray(alpha, dtype=np.float32) / 255.0
    result = image_arr * (1.0 - alpha_arr[..., None]) + np.clip(texture, 0, 255) * alpha_arr[..., None]
    return result, alpha_arr


def apply_occlusion_composite(source: Any, generated: Any, task: dict[str, Any], seed: int) -> tuple[Any, Any, dict[str, Any]]:
    """Enforce bbox-anchored foreground occlusion using generated scene pixels."""
    import numpy as np
    from PIL import Image

    params = dict(task.get("postprocess", {}).get("parameters", task.get("parameters", {})))
    fraction = float(params["coverage_fraction"])
    region = task.get("sampled_placeholders", {}).get("region", "central region")
    severity = task.get("severity", "light")

    gen = generated.convert("RGB")
    src = source.convert("RGB")
    if gen.size != src.size:
        src = src.resize(gen.size, Image.Resampling.LANCZOS)

    bbox = object_bbox(src)
    obj_mask = object_pixel_mask(src)
    occ_mask = build_occlusion_mask_on_object(obj_mask, bbox, fraction, region, seed)
    foreground_mask = _foreground_expansion_mask(occ_mask, region, seed, fraction)

    gen_arr = np.asarray(gen, dtype=np.float32)
    result_arr, alpha_arr = _synthesize_occluder_layer(gen_arr, foreground_mask, seed, severity, region)
    result = Image.fromarray(np.clip(result_arr, 0, 255).astype(np.uint8))
    mask_img = Image.fromarray((foreground_mask.astype(np.uint8) * 255))
    obj_total = int(obj_mask.sum())
    details = {
        "method": "procedural_occlusion",
        "parameters": params,
        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
        "region": region,
        "target_coverage_fraction": fraction,
        "actual_object_coverage_fraction": round(float((foreground_mask & obj_mask).sum()) / max(obj_total, 1), 6),
        "foreground_mask_fraction": round(float(foreground_mask.mean()), 6),
    }
    return result, mask_img, details

def apply_procedural(image: Any, task: dict[str, Any], seed: int) -> tuple[Any, Any | None, dict[str, Any]]:
    import numpy as np
    from PIL import Image, ImageFilter
    method = task["method"]
    params = dict(task["parameters"])
    rgb = image.convert("RGB")
    mask = None
    details: dict[str, Any] = {"method": method, "parameters": params}

    if method == "identity":
        return rgb.copy(), None, details
    if method == "procedural_crop":
        side = task["sampled_placeholders"].get("side", "right")
        fraction = float(params["fraction"])
        x0, y0, x1, y1 = object_bbox(image)
        obj_w, obj_h = x1 - x0 + 1, y1 - y0 + 1
        shifts = {
            "right": ((rgb.width - 1) + round(obj_w * fraction) - x1, 0),
            "left": (-round(obj_w * fraction) - x0, 0),
            "top": (0, -round(obj_h * fraction) - y0),
            "bottom": (0, (rgb.height - 1) + round(obj_h * fraction) - y1),
        }
        dx, dy = shifts[side]
        canvas = Image.new("RGB", rgb.size)
        # Extend the nearest source edge instead of introducing artificial white pixels.
        edge = rgb.crop((0, 0, 1, rgb.height)).resize((rgb.width, rgb.height)) if dx > 0 else rgb.crop((rgb.width - 1, 0, rgb.width, rgb.height)).resize((rgb.width, rgb.height)) if dx < 0 else rgb.crop((0, 0, rgb.width, 1)).resize((rgb.width, rgb.height)) if dy > 0 else rgb.crop((0, rgb.height - 1, rgb.width, rgb.height)).resize((rgb.width, rgb.height))
        canvas.paste(edge)
        canvas.paste(rgb, (dx, dy))
        mask = Image.new("L", rgb.size, 0)
        if dx > 0: mask.paste(255, (0, 0, min(dx, rgb.width), rgb.height))
        elif dx < 0: mask.paste(255, (max(0, rgb.width + dx), 0, rgb.width, rgb.height))
        if dy > 0: mask.paste(255, (0, 0, rgb.width, min(dy, rgb.height)))
        elif dy < 0: mask.paste(255, (0, max(0, rgb.height + dy), rgb.width, rgb.height))
        details.update({"side": side, "fraction": fraction, "bbox": [x0, y0, x1, y1], "shift": [dx, dy]})
        return canvas, mask, details
    if method == "procedural_blur":
        return rgb.filter(ImageFilter.GaussianBlur(float(params["gaussian_radius"]))), None, details
    if method == "procedural_white_balance":
        array = np.asarray(rgb).astype(np.float32)
        array *= np.asarray(params["channel_gains"], dtype=np.float32)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)), None, details
    if method == "procedural_overexposure":
        array = np.asarray(rgb).astype(np.float32) * float(params["gain"])
        threshold = float(params["clip_threshold"])
        clipped = np.max(array, axis=2) >= threshold
        array[array >= threshold] = 255
        mask = Image.fromarray((clipped * 255).astype(np.uint8))
        details["clipped_pixel_ratio"] = round(float(clipped.mean()), 6)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)), mask, details
    if method == "procedural_underexposure":
        array = np.asarray(rgb).astype(np.float32) * float(params["gain"])
        threshold = float(params["black_threshold"])
        dark_weight = 1.0 - np.clip(array.mean(axis=2, keepdims=True) / 128.0, 0.0, 1.0)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, float(params["dark_noise_std"]), array.shape) * dark_weight
        array += noise
        clipped = np.max(array, axis=2) <= threshold
        array[array <= threshold] = 0
        mask = Image.fromarray((clipped * 255).astype(np.uint8))
        details["black_clipped_pixel_ratio"] = round(float(clipped.mean()), 6)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)), mask, details
    if method == "procedural_motion_blur":
        import math
        length = int(params["length"])
        angle = float(task["sampled_placeholders"].get("angle_degrees", 0))
        source = np.asarray(rgb, dtype=np.float32)
        pad = length
        padded = np.pad(source, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        accumulator = np.zeros_like(source)
        for index in range(length):
            offset = index - (length - 1) / 2
            dx = round(offset * math.cos(math.radians(angle)))
            dy = round(offset * math.sin(math.radians(angle)))
            accumulator += padded[pad + dy:pad + dy + rgb.height, pad + dx:pad + dx + rgb.width]
        details.update({"length": length, "angle_degrees": angle})
        return Image.fromarray(np.clip(accumulator / length, 0, 255).astype(np.uint8)), None, details
    if method == "procedural_recompression":
        codec = str(params["codec"]).upper()
        current = rgb
        for _ in range(int(params.get("passes", 1))):
            buffer = io.BytesIO()
            current.save(buffer, codec, quality=int(params["quality"]))
            current = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
        return current, None, details
    if method == "procedural_optical_aberration":
        kind = params["kind"]
        array = np.asarray(rgb)
        if kind == "chromatic_aberration":
            shift = int(params["channel_shift_pixels"])
            output = array.copy()
            output[:, :, 0] = np.roll(array[:, :, 0], shift, axis=1)
            output[:, :, 2] = np.roll(array[:, :, 2], -shift, axis=1)
            return Image.fromarray(output), None, details
        if kind == "barrel_distortion":
            coefficient = float(params["coefficient"])
            height, width = array.shape[:2]
            yy, xx = np.indices((height, width), dtype=np.float32)
            nx = (xx - (width - 1) / 2) / max((width - 1) / 2, 1)
            ny = (yy - (height - 1) / 2) / max((height - 1) / 2, 1)
            radius2 = nx * nx + ny * ny
            factor = 1 + coefficient * radius2
            source_x = np.clip(np.rint(nx * factor * (width - 1) / 2 + (width - 1) / 2), 0, width - 1).astype(int)
            source_y = np.clip(np.rint(ny * factor * (height - 1) / 2 + (height - 1) / 2), 0, height - 1).astype(int)
            return Image.fromarray(array[source_y, source_x]), None, details
        raise ValueError(f"Unsupported optical aberration kind: {kind}")
    if method == "procedural_digital":
        short_edge = int(params["short_edge"])
        scale = short_edge / min(rgb.size)
        small = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))), Image.Resampling.LANCZOS)
        small = small.filter(ImageFilter.GaussianBlur(float(params["gaussian_radius"])))
        array = np.asarray(small).astype(np.float32)
        rng = np.random.default_rng(seed)
        array += rng.normal(0, float(params["noise_std"]), array.shape)
        noisy = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        buffer = io.BytesIO()
        noisy.save(buffer, "JPEG", quality=int(params["jpeg_quality"]))
        degraded = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB").resize(rgb.size, Image.Resampling.NEAREST)
        details["intermediate_size"] = list(small.size)
        return degraded, None, details
    raise ValueError(f"Unsupported procedural method: {method}")


def apply_postprocess(image: Any, task: dict[str, Any], seed: int, source: Any | None = None) -> tuple[Any, Any | None, dict[str, Any]]:
    """Apply the task-configured light tail after generative scene construction."""
    postprocess = task.get("postprocess") or {}
    post_type = postprocess.get("type")
    if post_type == "occlusion":
        if source is None:
            raise ValueError("Occlusion postprocess requires the source albedo image")
        result, mask, details = apply_occlusion_composite(source, image, task, seed)
        return result, mask, {"postprocess": post_type, "parameters": dict(postprocess.get("parameters", {})), **details}
    method = POSTPROCESS_METHODS.get(post_type)
    if method is None:
        raise ValueError(f"Unsupported hybrid postprocess type: {post_type}")
    post_task = {
        **task,
        "method": method,
        "parameters": dict(postprocess.get("parameters", {})),
    }
    result, mask, details = apply_procedural(image, post_task, seed)
    details = {"postprocess": post_type, "parameters": post_task["parameters"], **details}
    return result, mask, details


def base_record(model: str, model_path: Path, task: dict[str, Any], negative_prompt: str, args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    return {
        "model": model, "model_path": str(model_path.resolve()), "task": task["id"],
        "problem_type": task["problem_type"], "severity": task["severity"],
        "method": task["method"], "factors": task["factors"],
        "factor_severity": task["factor_severity"], "parameters": task["parameters"],
        "sampled_placeholders": task["sampled_placeholders"],
        "postprocess": task.get("postprocess"),
        "scene_rationale": task.get("scene_rationale"),
        "sampled_reflection_content": task.get("sampled_reflection_content"),
        "base_prompt": task.get("base_prompt", task["prompt"]),
        "semantic_clause": task.get("semantic_clause"),
        "allows_camera_change": task["allows_camera_change"], "prompt": task["prompt"],
        "negative_prompt": negative_prompt, "negative_prompt_mode": "separate_parameter",
        "vlm_enabled": args.vlm_semantic_mode == "enabled",
        "vlm_selection_manifest": str(args.vlm_selection_manifest.resolve()) if args.vlm_selection_manifest else None,
        "seed": args.seed, "steps": args.steps, "cfg_scale": args.cfg_scale,
        "output": str(output_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled local albedo degradation matrix")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(DEFAULT_MODELS))
    parser.add_argument("--qwen-model-path", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--firered-model-path", type=Path, default=DEFAULT_FIRERED)
    parser.add_argument("--firered11-model-path", type=Path, default=DEFAULT_FIRERED_11)
    parser.add_argument("--task", action="append", dest="tasks", help="Task ID; repeatable")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--device-mode", choices=("cpu-offload", "cuda"), default="cpu-offload")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vlm-semantic-mode", choices=("off", "enabled"), default="off",
                        help="Use the VLM occlusion selector for occlusion tasks")
    parser.add_argument("--vlm-config", type=Path, default=vlm_occlusion_selector.DEFAULT_CONFIG)
    parser.add_argument("--vlm-memory-path", type=Path,
                        help="Shared JSON memory for the last three occluder selections")
    parser.add_argument("--vlm-selection-manifest", type=Path,
                        help="Read precomputed VLM occlusion selections from this manifest")
    parser.add_argument("--vlm-gpu", default=None,
                        help="Pin VLM subprocess to this physical GPU id; defaults to config/auto")
    args = parser.parse_args()
    args.model_paths = {"qwen-image-edit": args.qwen_model_path, "firered-image-edit": args.firered_model_path, "firered-image-edit-1.1": args.firered11_model_path}
    return args


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tasks, negative_prompt = load_tasks(args.prompt_library, args.seed)
    validate_tasks(tasks)
    selection_manifest = None
    if args.vlm_semantic_mode == "enabled" and args.vlm_selection_manifest:
        selection_manifest = json.loads(args.vlm_selection_manifest.read_text(encoding="utf-8"))
        expected_key = vlm_occlusion_selector.source_memory_key(args.image)
        if selection_manifest.get("source_image_key") != expected_key:
            raise ValueError("VLM selection manifest source image hash does not match input image")
        if int(selection_manifest.get("seed", -1)) != args.seed:
            raise ValueError("VLM selection manifest seed does not match run seed")
        expected_library_key = hashlib.sha256(args.prompt_library.read_bytes()).hexdigest()[:24]
        if selection_manifest.get("prompt_library_key") != expected_library_key:
            raise ValueError("VLM selection manifest prompt library does not match run library")
        expected_selector_key = hashlib.sha256(vlm_occlusion_selector.PROMPT_PATH.read_bytes()).hexdigest()[:24]
        if selection_manifest.get("selector_prompt_key") != expected_selector_key:
            raise ValueError("VLM selection manifest was generated with a different selector prompt")
    semantic_meta = {"status": "occlusion_selector_enabled" if args.vlm_semantic_mode == "enabled" else "disabled"}
    source_key = vlm_occlusion_selector.source_memory_key(args.image)
    occlusion_memory: list[dict[str, Any]] = []
    memory_path = args.vlm_memory_path or args.output_dir / "occlusion_selection_memory.json"
    if args.vlm_semantic_mode == "enabled" and memory_path.is_file():
        try:
            loaded = json.loads(memory_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get(source_key), list):
                occlusion_memory = loaded[source_key][-3:]
            elif isinstance(loaded, list):
                # Accept the pre-keyed format only for backward-compatible local runs.
                occlusion_memory = loaded[-3:]
        except Exception:
            occlusion_memory = []
    if args.list_tasks:
        for index, task in enumerate(tasks.values(), 1):
            default = "default" if task["enabled_by_default"] else "control"
            print(f"{index:02d} {task['id']:32s} {task['method']:24s} {task['severity']:11s} {default}")
        print(f"total={len(tasks)} default={sum(t['enabled_by_default'] for t in tasks.values())}")
        return
    selected = args.tasks or [task_id for task_id, task in tasks.items() if task["enabled_by_default"]]
    unknown = [task_id for task_id in selected if task_id not in tasks]
    if unknown:
        raise ValueError(f"Unknown task IDs: {unknown}. Use --list-tasks.")
    needs_model = any(tasks[task_id]["method"] not in PROCEDURAL_METHODS for task_id in selected)
    required = [args.image, args.prompt_library]
    if needs_model:
        required += [args.model_paths[name] for name in args.models]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))
    run_id = args.run_id or datetime.now().strftime("matrix_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / safe_name(run_id)
    total = len(args.models) * len(selected)
    print(f"Input: {args.image.resolve()}\nModels: {', '.join(args.models)}\nTasks: {len(selected)} | Jobs: {total}\nOutput: {run_dir.resolve()}")
    if args.dry_run:
        for model in args.models:
            for task_id in selected:
                print(f"DRY-RUN {model} :: {task_id} [{tasks[task_id]['method']}] -> {run_dir / model / task_id / 'output.png'}")
        return

    from PIL import Image
    run_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.image)
    records: list[dict[str, Any]] = []
    job_index = 0
    for model in args.models:
        model_path = args.model_paths[model]
        pipe = None
        try:
            model_tasks = [tasks[task_id] for task_id in selected if tasks[task_id]["method"] not in PROCEDURAL_METHODS]
            for task_id in selected:
                job_index += 1
                task = tasks[task_id]
                task_dir = run_dir / model / task_id
                output_path = task_dir / "output.png"
                metadata_path = task_dir / "metadata.json"
                print(f"[{job_index}/{total}] {model} :: {task_id} [{task['method']}]", flush=True)
                record = base_record(model, model_path, task, negative_prompt, args, output_path)
                if args.skip_existing and output_path.is_file():
                    if metadata_path.is_file():
                        record.update(json.loads(metadata_path.read_text(encoding="utf-8")))
                    record["status"] = "success"
                    record["reused_existing"] = True
                    records.append(record)
                    continue
                task_dir.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                selection = None
                try:
                    if task["method"] in PROCEDURAL_METHODS:
                        result, mask, details = apply_procedural(source, task, args.seed)
                    else:
                        if pipe is None:
                            print(f"Loading {model}: {model_path}", flush=True)
                            pipe = load_pipeline(model_path, args.device_mode)
                        effective_task = task
                        selection = None
                        selection_meta = None
                        if args.vlm_semantic_mode == "enabled" and memory_path.is_file():
                            try:
                                latest_memory = json.loads(memory_path.read_text(encoding="utf-8"))
                                if isinstance(latest_memory, dict) and isinstance(latest_memory.get(source_key), list):
                                    occlusion_memory = latest_memory[source_key][-3:]
                            except Exception:
                                pass
                        history_before = list(occlusion_memory[-3:])
                        if (args.vlm_semantic_mode == "enabled" and
                                "occlusion" in task.get("factors", [])):
                            if selection_manifest is not None:
                                selection = selection_manifest.get("selections", {}).get(task["id"])
                                if selection is None:
                                    raise ValueError(f"VLM selection manifest has no selection for {task['id']}")
                                selection_meta = {"status": "manifest", "manifest": str(args.vlm_selection_manifest.resolve())}
                            else:
                                selection, selection_meta = vlm_occlusion_selector.select_occluder(
                                    args.image, task["severity"], task["id"], history_before, args.vlm_config
                                )
                            effective_task = dict(task)
                            effective_task["base_prompt"] = task["prompt"]
                            effective_task["prompt"] = vlm_occlusion_selector.build_occlusion_edit_prompt(
                                task["prompt"], selection, task["severity"]
                            )
                        elif args.vlm_semantic_mode == "off":
                            effective_task = dict(task)
                            effective_task["base_prompt"] = task["prompt"]
                            effective_task["prompt"] = build_non_vlm_task_prompt(task)
                            record["non_vlm_final_prompt"] = effective_task["prompt"]
                        result = run_generative(pipe, source.convert("RGB"), effective_task, negative_prompt, args)
                        mask, details = None, {"method": task["method"]}
                        if selection is not None:
                            details["vlm_selection"] = selection
                            details["vlm_selection_meta"] = selection_meta
                            details["selection_history_before"] = [item.get("occluder_type") for item in history_before]
                            record["vlm_final_prompt"] = effective_task["prompt"]
                        if task["method"] == "hybrid":
                            result, mask, post = apply_postprocess(result, task, args.seed, source=source)
                            details["postprocess"] = post
                    result.save(output_path)
                    if selection is not None:
                        committed = {
                            **selection,
                            "task_id": task["id"],
                            "model": model,
                        }
                        occlusion_memory.append(committed)
                        occlusion_memory = occlusion_memory[-3:]
                        memory_path.parent.mkdir(parents=True, exist_ok=True)
                        existing_memory: dict[str, Any] = {}
                        if memory_path.is_file():
                            try:
                                loaded_memory = json.loads(memory_path.read_text(encoding="utf-8"))
                                if isinstance(loaded_memory, dict):
                                    existing_memory = loaded_memory
                            except Exception:
                                existing_memory = {}
                        existing_memory[source_key] = occlusion_memory
                        memory_path.write_text(json.dumps(existing_memory, ensure_ascii=False, indent=2), encoding="utf-8")
                        details["selection_history_after"] = [item.get("occluder_type") for item in occlusion_memory]
                        details["occlusion_postprocess"] = None
                    if mask is not None:
                        mask_path = task_dir / "affected_mask.png"
                        mask.save(mask_path)
                        record["affected_mask"] = str(mask_path.resolve())
                    record["degradation"] = details
                    record["status"] = "success"
                except Exception as exc:
                    record["status"] = "failed"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["traceback"] = traceback.format_exc()
                    print(f"  Failed: {record['error']}", file=sys.stderr)
                    # A failed offloaded call can leave accelerate hooks or
                    # partially allocated tensors in an inconsistent state.
                    # Reload before the next generative task so one failure
                    # cannot poison the rest of the matrix.
                    if pipe is not None and task["method"] not in PROCEDURAL_METHODS:
                        del pipe
                        pipe = None
                        release_cuda_memory()
                record["elapsed_seconds"] = round(time.monotonic() - started, 2)
                metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                records.append(record)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"Failed to load {model}: {error}", file=sys.stderr)
            completed = {r["task"] for r in records if r["model"] == model}
            records.extend({**base_record(model, model_path, tasks[task_id], negative_prompt, args, run_dir / model / task_id / "output.png"), "status": "failed", "error": error} for task_id in selected if task_id not in completed)
        finally:
            if pipe is not None:
                del pipe
                pipe = None
                release_cuda_memory()
    summary = {
        "schema_version": "2.0", "run_id": run_id, "source_image": str(args.image.resolve()),
        "prompt_library": str(args.prompt_library.resolve()), "models": args.models,
        "task_count": len(selected), "job_count": total, "seed": args.seed,
        "steps": args.steps, "cfg_scale": args.cfg_scale, "device_mode": args.device_mode,
        "vlm_semantic_mode": args.vlm_semantic_mode, "vlm_semantic": semantic_meta,
        "records": records,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = sum(record["status"] == "failed" for record in records)
    print(f"Summary: {summary_path.resolve()}\nCompleted: {len(records) - failed}/{len(records)} | Failed: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
