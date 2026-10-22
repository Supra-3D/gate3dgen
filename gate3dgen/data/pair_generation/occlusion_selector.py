#!/usr/bin/env python3
"""VLM selector for diverse, pure-generative foreground occlusion edits."""
from __future__ import annotations
import base64, hashlib, json, os, re
from pathlib import Path
from typing import Any

WORKSPACE = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()
DEFAULT_CONFIG = Path(os.environ.get("GATE3DGEN_VLM_CONFIG", WORKSPACE / "data/occlusion_vlm.yaml"))
PROMPT_PATH = Path(os.environ.get("GATE3DGEN_OCCLUSION_PROMPT", WORKSPACE / "data/occlusion_selector_prompt.json"))
REGIONS = ("lower-left region", "lower-right region", "central region", "upper region")
FALLBACK_OCCLUDERS = ("pedestrian", "passing car", "tree branch", "vehicle side", "shopping cart", "door frame")
SCHEMA_VERSION = "occlusion-selector-1.0"

def _image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"

def _parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    candidate = text
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        candidate = candidate[start:end + 1] if start >= 0 and end > start else candidate
    candidate = re.sub(r"[\x00-\x1f]", " ", candidate)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("VLM response must be a JSON object")
    if isinstance(value.get("selection"), dict):
        value = value["selection"]
    return value

def _clip_text(value: Any, limit: int) -> str:
    return str(value or "").strip().replace("\n", " ")[:limit]

def _range_for(severity: str) -> tuple[float, float]:
    return (0.08, 0.35) if severity == "light" else (0.35, 0.65)

def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

def _validate(value: dict[str, Any], severity: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    aliases = {
        "object_category": ("object_category", "object", "object_type", "object_identity", "object_name"),
        "scene_context": ("scene_context", "scene", "environment", "capture_context"),
        "occluder_type": ("occluder_type", "occluder", "foreground_subject", "occluder_category"),
        "occluder_description": ("occluder_description", "description", "foreground_description"),
        "region": ("region", "placement", "occlusion_region"),
        "coverage_fraction": ("coverage_fraction", "coverage", "occlusion_ratio", "ratio"),
        "confidence": ("confidence", "score"),
        "reason": ("reason", "rationale", "explanation"),
    }
    value = {key: next((value[name] for name in names if name in value), None) for key, names in aliases.items()}
    if value["confidence"] is None:
        value["confidence"] = 0.5
    if value["reason"] is None:
        value["reason"] = "normalized VLM selection"
    required = ("object_category", "scene_context", "occluder_type", "occluder_description", "region", "coverage_fraction", "confidence", "reason")
    for key in required:
        if key not in value:
            raise ValueError(f"missing VLM field: {key}")
    low, high = _range_for(severity)
    occluder_type = _clip_text(value["occluder_type"], 80)
    used = {_canonical(item.get("occluder_type", "")) for item in history}
    if not occluder_type or _canonical(occluder_type) in used:
        raise ValueError(f"occluder_type is empty or recently used: {occluder_type}")
    region = _clip_text(value["region"], 40) or "lower-right region"
    if region not in REGIONS:
        raise ValueError(f"invalid region: {region}")
    raw_coverage = value["coverage_fraction"]
    coverage = max(low, min(high, float(raw_coverage))) if raw_coverage is not None else round((low + high) / 2, 3)
    description = _clip_text(value["occluder_description"], 240)
    if not description:
        description = f"one real {occluder_type} passing close to the camera with visible surface texture"
    return {
        "object_category": _clip_text(value["object_category"], 120) or "the visible object",
        "scene_context": _clip_text(value["scene_context"], 180) or "a compatible ordinary real-world capture",
        "occluder_type": occluder_type,
        "occluder_description": description,
        "region": region,
        "coverage_fraction": round(coverage, 3),
        "confidence": max(0.0, min(1.0, float(value["confidence"]))),
        "reason": _clip_text(value["reason"], 300),
    }

def _load_client(config_path: Path):
    import yaml
    import requests
    from types import SimpleNamespace
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = cfg.get("qwen_vlm", cfg)
    if not cfg.get("base_url") or not cfg.get("model"):
        raise ValueError("Occlusion VLM config requires base_url and model for your authorized local server")
    def create(**payload):
        payload["model"] = cfg["model"]
        payload["max_tokens"] = int(cfg.get("max_new_tokens", 512))
        headers = {"Content-Type": "application/json"}
        if cfg.get("api_key_env"):
            headers["Authorization"] = "Bearer " + os.environ[cfg["api_key_env"]]
        response = requests.post(cfg["base_url"].rstrip("/") + "/chat/completions", json=payload,
                                 headers=headers, timeout=float(cfg.get("timeout", 120)))
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), cfg

def _fallback(severity: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    used = {_canonical(item.get("occluder_type", "")) for item in history}
    choice = next((x for x in FALLBACK_OCCLUDERS if _canonical(x) not in used), FALLBACK_OCCLUDERS[0])
    low, high = _range_for(severity)
    return {
        "object_category": "the visible foreground object",
        "scene_context": "a compatible ordinary real-world capture",
        "occluder_type": choice,
        "occluder_description": f"one real {choice} passing close to the camera with visible surface texture",
        "region": "lower-right region",
        "coverage_fraction": round((low + high) / 2, 3),
        "confidence": 0.0,
        "reason": "fallback selection after VLM failure",
    }

def select_occluder(image_path: Path, severity: str, task_id: str, history: list[dict[str, Any]], config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    history = list(history[-3:])
    if not PROMPT_PATH.is_file():
        raise FileNotFoundError("Supply your actual selector prompt JSON via GATE3DGEN_OCCLUSION_PROMPT; no historical prompt is fabricated")
    base = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    low, high = _range_for(severity)
    excluded = [item.get("occluder_type", "") for item in history]
    prompt = f"""{base['prompt']}
Task id: {task_id}
Severity: {severity}
Allowed coverage range: {low:.2f} to {high:.2f}
Allowed regions: {', '.join(REGIONS)}
Recently used occluder types to exclude: {json.dumps(excluded, ensure_ascii=False)}
Do not return a bbox, mask, postprocess instruction, or edit-model prompt."""
    metadata = {"schema_version": SCHEMA_VERSION, "status": "fallback", "history_before": history, "excluded_occluders": excluded}
    try:
        client, cfg = _load_client(config_path)
        content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}}]
        response = client.chat.completions.create(model="local-qwen-vlm", messages=[{"role": "user", "content": content}], response_format={"type": "json_object"})
        selection = _validate(_parse_json(response.choices[0].message.content), severity, history)
        metadata.update(status="generated", vlm_config={k: cfg.get(k) for k in ("env_name", "routing_mode", "vl_model_path", "max_new_tokens")})
        return selection, metadata
    except Exception as first_error:
        metadata["first_error"] = f"{type(first_error).__name__}: {first_error}"
        retry_prompt = prompt + "\nYour previous choice was invalid. Select a different occluder type and return valid JSON only."
        try:
            client, _ = _load_client(config_path)
            content = [{"type": "text", "text": retry_prompt}, {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}}]
            response = client.chat.completions.create(model="local-qwen-vlm", messages=[{"role": "user", "content": content}], response_format={"type": "json_object"})
            selection = _validate(_parse_json(response.choices[0].message.content), severity, history)
            metadata.update(status="generated_retry")
            return selection, metadata
        except Exception as second_error:
            metadata["second_error"] = f"{type(second_error).__name__}: {second_error}"
            return _fallback(severity, history), metadata

def build_occlusion_edit_prompt(base_prompt: str, selection: dict[str, Any], severity: str) -> str:
    percent = f"{selection['coverage_fraction']:.0%}"
    extra = " The remaining visible part of the object must still clearly identify the same object." if severity == "heavy" else " Keep most of the object visible and make the occlusion localized."
    full_scene = (
        "Build a continuous full-frame ordinary real-world photograph around the object using the selected scene context. "
        "Fill the frame with a physically appropriate support and contact, a ground plane, foreground, middle-ground and "
        "background depth layers, coherent perspective, natural illumination and context-compatible environmental details. "
        "New background pixels may be generated outside the protected object to complete the scene. Do not leave large "
        "empty black regions, a void background, a studio-like cutout, or an unrelated empty backdrop."
    )
    clause = (
        f"The visible object appears to be {selection['object_category']} in {selection['scene_context']}. {full_scene} "
        f"Add exactly one {selection['occluder_description']} between the camera and the object. "
        f"Place it in the {selection['region']}. It must be a real foreground subject with a continuous physical silhouette, "
        f"natural perspective, correct depth ordering and visible surface texture. It should approximately occlude {percent} of the object. "
        f"Keep all uncovered object pixels, geometry, material, intrinsic albedo, texture, labels and markings unchanged. "
        f"Do not paint the occluder onto the object.{extra}"
    )
    return f"{base_prompt} {clause}"

def source_memory_key(image_path: Path) -> str:
    return hashlib.sha256(image_path.read_bytes()).hexdigest()[:24]
