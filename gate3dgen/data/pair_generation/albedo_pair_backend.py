#!/usr/bin/env python3
"""Generate clean-scene/degraded albedo pairs with a local image editor."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).expanduser().resolve()
from .degradation_backend import (
    DEFAULT_FIRERED_11, DEFAULT_QWEN, MODEL_NAMES, apply_postprocess,
    apply_procedural, build_non_vlm_task_prompt, load_pipeline, object_pixel_mask,
    release_cuda_memory, safe_name, stable_rng,
)

DEFAULT_LIBRARY = Path(__file__).resolve().parents[2] / "resources/prompts/gate3dgen_pair_synthesis.json"
DEFAULT_IMAGE = PROJECT_ROOT / "data/example.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/albedo_pairs/local"
DEGRADATION_LOCK = (
    "Use the input image as the exact scene reference. Preserve the same camera viewpoint, "
    "framing, object silhouette, object position, support, background layout, background "
    "details and lighting context. Add only the requested degradation. Do not reconstruct, "
    "redesign, recompose or replace the scene."
)

SCENE_LOCK = (
    "The input GT already contains the complete scene. Keep its background, support, "
    "camera, framing and all unaffected pixels unchanged. Do not add, remove, replace "
    "or redesign any background element. Apply only the named degradation."
)

DEFAULT_DEGRADED_SUFFIX = (
    "Use the input GT image as the complete and immutable scene. Preserve the exact object geometry, "
    "silhouette, camera viewpoint, framing, support, perspective, background layout and every unaffected "
    "region. Do not add, remove, replace, simplify or redesign any background element. Apply only the "
    "named degradation at the requested severity; do not introduce any other degradation."
)

SCENE_BY_TYPE = {
    "crop": "scene_handheld_detail", "shadow": "scene_shadow_window",
    "glass_reflection": "scene_glass_display", "veiling_glare": "scene_glare_source",
    "overexposure": "scene_bright_window", "optical_blur": "scene_depth_corridor",
    "digital_degradation": "scene_handheld_detail", "white_balance": "scene_warm_indoor",
    "underexposure": "scene_dim_room", "motion_blur": "scene_motion_street",
    "local_color_spill": "scene_color_environment", "dirty_glass": "scene_rainy_glass",
}
OBJECT_ONLY_TYPES = {"shadow", "local_color_spill"}
FOREGROUND_TYPES = {"glass_reflection", "dirty_glass"}


def _scene_prompt(scene_id: str, scene: dict) -> str:
    elements = ", ".join(scene.get("background_elements", []))
    foreground = ", ".join(scene.get("foreground_elements", []))
    midground = ", ".join(scene.get("midground_elements", []))
    negative = ", ".join(scene.get("negative", []))
    return (
        f"Scene specification ({scene_id}): {scene.get('description', '')} "
        f"Foreground elements: {foreground}. Support: {scene.get('support', '')}. "
        f"Midground elements: {midground}. Background elements: {elements}. "
        f"Light source: {scene.get('light_source', '')}. Camera context: {scene.get('camera_context', '')}. "
        f"GT lighting: {scene.get('gt_lighting', '')}. GT exclusions: {negative}."
    ).strip()


def _existing_scene_degradation_prompt(text: str) -> str:
    """Rewrite legacy background-creation prefixes so stage two reuses GT's scene."""
    text = re.sub(r"^Add a simple .*?background[^.:]*(?:\.|:)\s*",
                  "Use the existing GT background and support; ", text, count=1, flags=re.IGNORECASE)
    return text


def _protect_gt_background(gt, generated, source_clean):
    """Restore GT pixels outside the original albedo foreground for object-only edits."""
    import numpy as np
    from PIL import Image
    src_mask = object_pixel_mask(source_clean)
    if src_mask.shape != (gt.height, gt.width):
        src_mask = np.asarray(Image.fromarray((src_mask * 255).astype("uint8")).resize(gt.size, Image.Resampling.NEAREST)) > 0
    base = np.asarray(gt.convert("RGB"))
    edited = np.asarray(generated.convert("RGB"))
    return Image.fromarray(np.where(src_mask[..., None], edited, base).astype("uint8"))


def _protected_background_max_diff(gt, degraded, source_clean) -> int:
    import numpy as np
    from PIL import Image
    mask = object_pixel_mask(source_clean)
    if mask.shape != (gt.height, gt.width):
        mask = np.asarray(Image.fromarray((mask * 255).astype("uint8")).resize(gt.size, Image.Resampling.NEAREST)) > 0
    base = np.asarray(gt.convert("RGB"), dtype=np.int16)
    edited = np.asarray(degraded.convert("RGB"), dtype=np.int16)
    outside = ~mask
    if not outside.any():
        return 0
    return int(np.abs(base - edited)[outside].max())


def _sample_parameter_spec(spec, rng):
    if isinstance(spec, list):
        return [_sample_parameter_spec(item, rng) for item in spec]
    if not isinstance(spec, dict):
        return spec
    if "choices" in spec:
        return rng.choice(spec["choices"])
    if "min" in spec and "max" in spec:
        if spec.get("type") == "integer":
            return rng.randint(int(spec["min"]), int(spec["max"]))
        return round(rng.uniform(float(spec["min"]), float(spec["max"])), 6)
    return {key: _sample_parameter_spec(value, rng) for key, value in spec.items()}


def _resolve_parameters(raw, seed, task_id, scene_instance_id, object_id):
    defaults = dict(raw.get("parameters", {}))
    ranges = raw.get("parameter_ranges", {}) or {}
    if not ranges:
        return defaults, None
    material = f"{seed}:{object_id}:{task_id}:{scene_instance_id}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    parameter_seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(parameter_seed)
    resolved = dict(defaults)
    sampled = {}
    for name, spec in ranges.items():
        resolved[name] = _sample_parameter_spec(spec, rng)
        sampled[name] = {"spec": spec, "value": resolved[name]}
    return resolved, {"seed": parameter_seed, "ranges": ranges, "sampled": sampled}


def load_tasks(path: Path, seed: int, object_id: str = "default") -> tuple[dict[str, dict], str]:
    library = json.loads(path.read_text(encoding="utf-8"))
    tasks = {}
    prefix = library.get("global_real_photo_prefix", "").strip()
    suffix = library.get("global_preservation_suffix", "").strip()
    negative = library.get("global_negative_prompt", "")
    degraded_suffix = library.get("global_degradation_suffix", DEFAULT_DEGRADED_SUFFIX).strip()
    scene_library = library.get("scene_library", {})
    scene_assignments = library.get("scene_assignments", SCENE_BY_TYPE)

    def add(raw, problem_type, severity):
        if "gt_prompt" not in raw or "degradation_prompt" not in raw:
            raise ValueError(f"Pair task {raw.get('id')} must define gt_prompt and degradation_prompt")
        task_id = raw["id"]
        if task_id in tasks:
            raise ValueError(f"Duplicate task id: {task_id}")
        sampled = {k: stable_rng(seed, task_id).choice(v) for k, v in raw.get("placeholders", {}).items()}
        def fmt(text): return text.format(**sampled).strip()
        scene_id = raw.get("scene_id") or scene_assignments.get(problem_type, SCENE_BY_TYPE.get(problem_type, "scene_handheld_detail"))
        scene = scene_library.get(scene_id, {})
        scene_instance_id = f"{problem_type}__seed{seed}__{scene_id}"
        parameters, parameter_sampling = _resolve_parameters(raw, seed, task_id, scene_instance_id, object_id)
        if parameter_sampling:
            parameter_sampling["object_id"] = object_id
        postprocess = raw.get("postprocess")
        if postprocess and parameter_sampling:
            postprocess = dict(postprocess)
            postprocess["parameters"] = dict(parameters)
        default_scope = "object_only" if problem_type in OBJECT_ONLY_TYPES else (
            "foreground_layer" if problem_type in FOREGROUND_TYPES else "global_capture"
        )
        scope = raw.get("affected_scope", default_scope)
        tasks[task_id] = {
            "id": task_id, "problem_type": problem_type, "severity": severity,
            "method": raw.get("method", "generative"), "factors": raw.get("factors", problem_type.split("+")),
            "parameters": parameters, "parameter_sampling": parameter_sampling, "postprocess": postprocess,
            "sampled_placeholders": sampled, "scene_rationale": raw.get("scene_rationale"),
            "allows_camera_change": raw.get("allows_camera_change", False),
            "affected_scope": scope, "preserve_scope": raw.get("preserve_scope", "background_and_geometry"),
            "scene_id": scene_id, "scene_instance_id": scene_instance_id, "scene_spec": scene,
            "gt_prompt": " ".join(x for x in (prefix, fmt(raw["gt_prompt"]), _scene_prompt(scene_id, scene), suffix) if x),
            "degradation_prompt": " ".join(x for x in (_existing_scene_degradation_prompt(fmt(raw["degradation_prompt"])), SCENE_LOCK, DEGRADATION_LOCK, degraded_suffix) if x),
        }
    for problem_type, group in library["degradation_prompts"].items():
        for raw in group.get("variants", []):
            if raw.get("severity") == "light":
                continue
            merged = dict(raw)
            merged.setdefault("method", group.get("preferred_method", "generative"))
            merged.setdefault("placeholders", group.get("placeholders", {}))
            add(merged, problem_type, raw["severity"])
    for raw in library.get("combination_prompts", []):
        add(raw, "+".join(raw["problems"]), raw.get("severity", "combination"))
    return tasks, negative


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    object_id = args.object_id or args.image.stem
    tasks, negative = load_tasks(args.prompt_library, args.seed, object_id)
    selected = args.tasks or list(tasks)
    unknown = [x for x in selected if x not in tasks]
    if unknown:
        raise ValueError(f"Unknown task IDs: {unknown}")
    if args.list_tasks:
        for task_id, task in tasks.items():
            print(f"{task_id:32s} {task['method']:12s} {task['severity']}")
        return
    missing = [str(x) for x in (args.image, args.prompt_library) if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))
    run_id = args.run_id or datetime.now().strftime("pair_%Y%m%d_%H%M%S")
    run_dir = args.output_dir / safe_name(run_id)
    if args.dry_run:
        for model in args.models:
            for task_id in selected:
                print(f"DRY-RUN {model} :: {task_id} :: clean -> gt.png -> degraded.png")
        return
    from PIL import Image
    source = Image.open(args.image).convert("RGB")
    records = []
    for model in args.models:
        model_path = args.model_paths[model]
        pipe = None
        try:
            for task_id in selected:
                task = tasks[task_id]
                task_dir = run_dir / model / task_id
                gt_path, degraded_path, metadata_path = (task_dir / x for x in ("gt.png", "degraded.png", "metadata.json"))
                source_path = task_dir / "source_clean.png"
                pair_id = f"{model}__{task_id}__seed{args.seed}"
                record = {"pair_id": pair_id, "model": model, "gt_model": model, "degradation_model": model,
                          "task": task_id, "pair_mode": "two-stage", "input_kind": "clean_albedo",
                          "output_kind": "degraded_scene", "gt_prompt": task["gt_prompt"],
                          "degradation_prompt": task["degradation_prompt"], "parameters": task["parameters"],
                          "parameter_sampling": task.get("parameter_sampling"),
                          "sampled_placeholders": task["sampled_placeholders"], "affected_scope": task["affected_scope"],
                          "preserve_scope": task["preserve_scope"], "scene_id": task["scene_id"],
                          "scene_instance_id": task["scene_instance_id"], "scene_spec": task["scene_spec"],
                          "source_clean": str(source_path.resolve()),
                          "gt_output": str(gt_path.resolve()), "degraded_input": str(gt_path.resolve()),
                          "degraded_output": str(degraded_path.resolve()),
                          "status": "failed"}
                if args.skip_existing and gt_path.is_file() and degraded_path.is_file() and metadata_path.is_file():
                    record.update(json.loads(metadata_path.read_text(encoding="utf-8")))
                    record["status"] = "success"
                    record["reused_existing"] = True
                    records.append(record)
                    continue
                task_dir.mkdir(parents=True, exist_ok=True)
                source.save(source_path)
                started = time.monotonic()
                try:
                    if args.degraded_only and not gt_path.is_file():
                        raise FileNotFoundError(f"--degraded-only requires existing GT: {gt_path}")
                    if not gt_path.is_file():
                        if pipe is None:
                            pipe = load_pipeline(model_path, args.device_mode)
                        gt_task = {"prompt": task["gt_prompt"]}
                        gt = pipe(image=[source], prompt=task["gt_prompt"], negative_prompt=negative or " ",
                                   num_inference_steps=args.steps, true_cfg_scale=args.cfg_scale,
                                   guidance_scale=1.0, num_images_per_prompt=1).images[0]
                        gt.convert("RGB").save(gt_path)
                    record["gt_generation_status"] = "success"
                    if args.gt_only:
                        record["status"] = "gt_only"
                    else:
                        gt = Image.open(gt_path).convert("RGB")
                        if task["method"] in {"hybrid", "procedural_crop", "procedural_blur", "procedural_digital", "procedural_white_balance", "procedural_overexposure", "procedural_underexposure", "procedural_motion_blur", "procedural_recompression", "procedural_optical_aberration"}:
                            # Most hybrid tasks are deterministic tails. The stress stack
                            # retains its generative glass/reflection stage before JPEG damage.
                            if task["id"] == "real_capture_stack":
                                if pipe is None: pipe = load_pipeline(model_path, args.device_mode)
                                generated = pipe(image=[gt], prompt=task["degradation_prompt"], negative_prompt=negative or " ", num_inference_steps=args.steps, true_cfg_scale=args.cfg_scale, guidance_scale=1.0, num_images_per_prompt=1).images[0]
                                degraded, mask, details = apply_postprocess(generated.convert("RGB"), task, args.seed, source=gt)
                                record["degradation_strategy"] = "gt_then_local_api_edit_then_local_postprocess"
                            else:
                                degraded, mask, details = apply_postprocess(gt, task, args.seed, source=gt) if task.get("postprocess") else apply_procedural(gt, {**task, "method": task["method"]}, args.seed)
                                record["degradation_strategy"] = "gt_then_local_postprocess"
                        else:
                            if pipe is None:
                                pipe = load_pipeline(model_path, args.device_mode)
                            generated = pipe(image=[gt], prompt=task["degradation_prompt"], negative_prompt=negative or " ",
                                              num_inference_steps=args.steps, true_cfg_scale=args.cfg_scale,
                                              guidance_scale=1.0, num_images_per_prompt=1).images[0]
                            degraded = _protect_gt_background(gt, generated, source) if task["affected_scope"] == "object_only" else generated
                            mask, details = None, {"method": task["method"]}
                            if task["affected_scope"] == "object_only":
                                details["background_protection"] = "source_object_mask_recomposite"
                            record["degradation_strategy"] = "gt_then_local_api_edit"
                        degraded.convert("RGB").save(degraded_path)
                        if mask is not None:
                            mask.save(task_dir / "affected_mask.png")
                        if task["affected_scope"] == "object_only":
                            record["protected_background_max_abs_diff"] = _protected_background_max_diff(gt, degraded, source)
                        record["degradation"] = details
                        record["degradation_status"] = "success"
                        record["status"] = "success"
                    record["gt_sha256"] = sha256(gt_path)
                    if degraded_path.is_file(): record["degraded_sha256"] = sha256(degraded_path)
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    if pipe is not None:
                        del pipe; pipe = None; release_cuda_memory()
                record["elapsed_seconds"] = round(time.monotonic() - started, 2)
                metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                records.append(record)
        finally:
            if pipe is not None:
                del pipe; release_cuda_memory()
    summary = {"schema_version": "pair-1.0", "run_id": run_id, "source_image": str(args.image.resolve()),
               "prompt_library": str(args.prompt_library.resolve()), "models": args.models,
               "pair_mode": "two-stage", "records": records}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(r["status"] == "failed" for r in records): raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=["qwen-image-edit", "firered-image-edit-1.1"])
    parser.add_argument("--qwen-model-path", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--firered-model-path", type=Path, default=DEFAULT_FIRERED_11)
    parser.add_argument("--firered11-model-path", type=Path, default=DEFAULT_FIRERED_11)
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--object-id", help="Stable object identifier used for parameter sampling; defaults to the input filename stem.")
    parser.add_argument("--run-id"); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=40); parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--gpu", default="0"); parser.add_argument("--device-mode", choices=("cpu-offload", "cuda"), default="cpu-offload")
    parser.add_argument("--skip-existing", action="store_true"); parser.add_argument("--gt-only", action="store_true"); parser.add_argument("--degraded-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.model_paths = {"qwen-image-edit": args.qwen_model_path, "firered-image-edit": args.firered_model_path, "firered-image-edit-1.1": args.firered11_model_path}
    run(args)

if __name__ == "__main__": main()
