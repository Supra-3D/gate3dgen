from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .experimental.recipes import is_albedo_recipe, recipe_from_config, validate_recipe_name
from .prompts import load_prompt_library
from .condition_contract import condition_enabled, normalize_condition_config


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()
SUPPORTED_SPLITS = ("train", "validation", "test")
DEFAULT_ALBEDO_ROOT = str(WORKSPACE / "data/TexVerse")


def _resolve_workspace_path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE / path


def configured_manifests(config: dict[str, Any]) -> dict[str, Path]:
    data = config["data"]
    if "manifests" in data:
        manifests = data["manifests"]
        if not isinstance(manifests, dict) or "train" not in manifests:
            raise ValueError("data.manifests must be a mapping containing train")
        unknown = set(manifests) - set(SUPPORTED_SPLITS)
        if unknown:
            raise ValueError(f"unsupported manifest splits: {sorted(unknown)}")
        return {split: Path(value) for split, value in manifests.items()}
    if "manifest" not in data:
        raise ValueError("data must define manifest or manifests")
    return {"train": Path(data["manifest"])}


def load_config(scope: str, config_path: Optional[str] = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else WORKSPACE / "configs/train/purifier.yaml"
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("scope") != scope:
        raise ValueError(f"Invalid config for scope={scope}: {path}")
    config["_config_path"] = str(path.resolve())
    data = config["data"]
    if "manifests" in data:
        data["manifests"] = {
            split: str(_resolve_workspace_path(value))
            for split, value in data["manifests"].items()
        }
        # Backward-compatible alias used by the training-only path.
        data["manifest"] = data["manifests"]["train"]
    else:
        data["manifest"] = str(_resolve_workspace_path(data["manifest"]))
    config["data"]["data_root"] = str(_resolve_workspace_path(config["data"]["data_root"]))
    subject_mask_root = config["data"].get("subject_mask_root")
    if subject_mask_root:
        config["data"]["subject_mask_root"] = str(
            _resolve_workspace_path(subject_mask_root).resolve()
        )
    recipe = validate_recipe_name(recipe_from_config(config))
    config["recipe"] = recipe
    if is_albedo_recipe(recipe):
        # source_relative_path is the only accepted albedo path input.  An
        # old manifest's source_path/source_original_path is provenance only;
        # never use it as a fallback.  Env override is explicit and wins over
        # YAML so a mounted TexVerse can be selected without editing configs.
        albedo_root = os.environ.get(
            "FIRERED_ALBEDO_ROOT",
            config["data"].get("albedo_root", DEFAULT_ALBEDO_ROOT),
        )
        if not str(albedo_root).strip():
            raise ValueError("albedo recipe requires a non-empty data.albedo_root")
        config["data"]["albedo_root"] = str(Path(albedo_root).expanduser().resolve())
    model_override = os.environ.get("GATE3DGEN_MODEL_PATH")
    model_path = model_override or config["model"]["pretrained_model_path"]
    config["model"]["pretrained_model_path"] = str(_resolve_workspace_path(model_path))
    cache_root = os.environ.get("GATE3DGEN_TRAIN_CACHE_ROOT") or config["paths"]["cache_root"]
    config["paths"]["cache_root"] = str(_resolve_workspace_path(cache_root))
    output_root = os.environ.get("GATE3DGEN_OUTPUT_ROOT") or config["paths"]["output_dir"]
    output_dir = _resolve_workspace_path(output_root).resolve()
    explicit_allowed_root = os.environ.get("GATE3DGEN_ALLOWED_OUTPUT_ROOT")
    allowed_output_root = (
        Path(explicit_allowed_root).expanduser().resolve()
        if explicit_allowed_root else (WORKSPACE / "outputs").resolve()
    )
    if allowed_output_root in {Path("/"), WORKSPACE.resolve()}:
        raise ValueError("GATE3DGEN_ALLOWED_OUTPUT_ROOT must be a dedicated run directory")
    try:
        output_dir.relative_to(allowed_output_root)
    except ValueError as error:
        raise ValueError(
            f"model output_dir must be inside {allowed_output_root}, got {output_dir}"
        ) from error
    config["paths"]["output_dir"] = str(output_dir)
    disable_init = os.environ.get("FIRERED_DISABLE_INIT_FROM_CHECKPOINT", "0") == "1"
    init_checkpoint = None if disable_init else (
        os.environ.get("FIRERED_INIT_FROM_CHECKPOINT")
        or config.get("training", {}).get("init_from_checkpoint")
    )
    if init_checkpoint:
        config["training"]["init_from_checkpoint"] = str(
            _resolve_workspace_path(init_checkpoint).resolve()
        )
    else:
        config["training"].pop("init_from_checkpoint", None)
    # Gate A/B runs use a separately trained Qwen-VL checkpoint to produce
    # offline hidden states.  Normalize known distributed/home prefixes and
    # explicit environment overrides after the base FireRed model path is
    # resolved; artifact existence is checked by the launcher preflight, not
    # here, so ``--dry-run`` stays CPU-only.
    normalize_condition_config(config)
    # The classifier is opt-in for hand-written/legacy configs so a restore-only
    # job remains numerically compatible.  Production classify YAMLs set this
    # explicitly to true; environment overrides are useful for launcher smoke
    # tests without editing a mounted config.
    classification = config.get("classification", config.get("repair_classifier", {}))
    if classification is None:
        classification = {}
    if not isinstance(classification, dict):
        raise ValueError("classification must be a mapping")
    config["classification"] = {
        "enabled": bool(classification.get("enabled", False)),
        # ``vae_latent`` preserves the optional V1/V2 compatibility path.
        "feature_source": str(classification.get("feature_source", "vae_latent")),
        "mode": str(classification.get("mode", "vae_latent_mlp")),
        "loss_weight": float(classification.get("loss_weight", 0.1)),
        "threshold": float(classification.get("threshold", 0.5)),
        "temperature": float(classification.get("temperature", 1.0)),
        "hidden_dim": int(classification.get("hidden_dim", 128)),
        "dropout": float(classification.get("dropout", 0.0)),
        "label_source": str(classification.get("label_source", "paired_degraded_gt")),
        "threshold_selection": dict(classification.get("threshold_selection", {})),
    }
    prompt_override = os.environ.get("FIRERED_PROMPT_LIBRARY")
    if prompt_override:
        prompt_value = config.get("prompt_library", config.get("prompts", {}))
        prompt_value = dict(prompt_value) if isinstance(prompt_value, dict) else {}
        prompt_value["path"] = prompt_override
        config["prompt_library"] = prompt_value
    elif "prompts" in config and "prompt_library" not in config:
        config["prompt_library"] = config.pop("prompts")
    if "prompt_library" in config:
        selected_prompt = load_prompt_library(config)
        config["prompt_library"] = {
            "path": selected_prompt["path"],
            "id": selected_prompt["id"],
            "fingerprint": selected_prompt["fingerprint"],
            "version": selected_prompt.get("version"),
        }
    validate_training_config(config)
    return config


def validate_training_config(config: dict[str, Any]) -> None:
    model = config["model"]
    training = config["training"]
    loss = config.get("loss", {"objective": "flow"})
    data = config.get("data", {})
    recipe = validate_recipe_name(recipe_from_config(config))
    architecture = model.get("architecture", "stock")
    backend = model.get("attention_backend", "native")
    if architecture not in {"stock", "restoration_custom"}:
        raise ValueError(f"unsupported model architecture: {architecture}")
    if backend not in {"native", "fa3", "auto"}:
        raise ValueError(f"unsupported attention backend: {backend}")
    if architecture == "stock" and backend != "native":
        raise ValueError("stock architecture only supports native attention")

    mode = training.get("mode", "lora")
    if mode not in {"lora", "partial_unfreeze"}:
        raise ValueError(f"unsupported training mode: {mode}")
    if mode == "partial_unfreeze":
        first, last = training.get("trainable_block_range", [29, 49])
        if not 0 <= int(first) <= int(last) < 60:
            raise ValueError(f"invalid trainable_block_range: {(first, last)}")
        if not training.get("use_fsdp", False):
            raise ValueError("partial_unfreeze requires use_fsdp=true for the 20B transformer")

    num_processes = int(training.get("num_processes", 1))
    num_machines = int(training.get("num_machines", 1))
    processes_per_machine = int(
        training.get("num_processes_per_machine", num_processes // num_machines)
    )
    if min(num_processes, num_machines, processes_per_machine) <= 0:
        raise ValueError("distributed process counts must all be positive")
    if num_processes != num_machines * processes_per_machine:
        raise ValueError(
            "distributed topology must satisfy num_processes == "
            "num_machines * num_processes_per_machine, got "
            f"{num_processes} != {num_machines} * {processes_per_machine}"
        )
    if is_albedo_recipe(recipe) and (num_processes, num_machines, processes_per_machine) != (32, 4, 8):
        raise ValueError(
            "albedo recipes require exactly 4 nodes x 8 GPUs = 32 ranks; got "
            f"{num_machines} x {processes_per_machine} ({num_processes})"
        )
    vae_encode_mode = training.get("vae_encode_mode", "sample")
    if vae_encode_mode not in {"sample", "mode"}:
        raise ValueError(f"training.vae_encode_mode must be sample or mode, got {vae_encode_mode!r}")

    objective = loss.get("objective", "flow")
    if objective not in {"flow", "hybrid_x0_pixel"}:
        raise ValueError(f"unsupported loss objective: {objective}")
    for key in (
        "flow_weight", "latent_x0_weight", "pixel_l1_weight", "pixel_lpips_weight",
        "pixel_background_weight",
        "pixel_low_frequency_weight",
    ):
        if float(loss.get(key, 0.0)) < 0:
            raise ValueError(f"loss weight must be non-negative: {key}")
    if objective == "hybrid_x0_pixel":
        if not isinstance(loss.get("pixel_loss_fg_only", False), bool):
            raise ValueError("pixel_loss_fg_only must be a boolean")
        background_weight = float(loss.get("pixel_background_weight", 0.0))
        if background_weight > 0 and not loss.get("pixel_loss_fg_only", False):
            raise ValueError("pixel_background_weight requires pixel_loss_fg_only=true")
        sigma_band = loss.get("x0_sigma_band", [0.5, 1.0])
        if not isinstance(sigma_band, (list, tuple)) or len(sigma_band) != 2:
            raise ValueError("x0_sigma_band must contain exactly [lo, hi]")
        lo, hi = (float(value) for value in sigma_band)
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError(
                f"x0_sigma_band must satisfy 0 < lo <= hi <= 1, got {(lo, hi)}"
            )
        active_weight = sum(
            float(loss.get(key, 0.0))
            for key in (
                "flow_weight",
                "latent_x0_weight",
                "pixel_l1_weight",
                "pixel_lpips_weight",
            )
        )
        if active_weight <= 0:
            raise ValueError("hybrid_x0_pixel requires at least one positive loss weight")
    if bool(loss.get("restoration_v3", False)):
        if is_albedo_recipe(recipe):
            raise ValueError("restoration V3 uses cached clean-GT subject masks, not an albedo recipe")
        if not data.get("subject_mask_root") or not data.get("subject_mask_required", False):
            raise ValueError("restoration V3 requires data.subject_mask_root and subject_mask_required=true")
        if data.get("subject_mask_schema") != "texverse-clean-gt-rmbg2-soft-mask-v1":
            raise ValueError("restoration V3 requires the RMBG2 clean-GT subject-mask schema")
    scales = loss.get("pixel_low_frequency_scales", [8, 16, 32])
    if not isinstance(scales, (list, tuple)) or not scales or any(int(v) <= 0 for v in scales):
        raise ValueError("pixel_low_frequency_scales must contain positive integers")
    multipliers = loss.get("task_multipliers", {})
    if not isinstance(multipliers, dict):
        raise ValueError("loss.task_multipliers must be a mapping")
    if any(not str(task).strip() or float(value) <= 0 for task, value in multipliers.items()):
        raise ValueError("loss.task_multipliers requires non-empty tasks and positive values")
    if is_albedo_recipe(recipe):
        for key in (
            "albedo_color_weight",
            "albedo_gradient_weight",
            "albedo_perceptual_weight",
            "albedo_consistency_weight",
            "albedo_aux_weight",
        ):
            if float(loss.get(key, 0.0)) < 0:
                raise ValueError(f"loss weight must be non-negative: {key}")
        if recipe == "rgb_albedo_multitask" and float(loss.get("albedo_aux_weight", 0.0)) > 0:
            if not bool(model.get("albedo_aux_head", False)):
                raise ValueError("recipe C albedo_aux_weight > 0 requires model.albedo_aux_head=true")
            if mode != "partial_unfreeze":
                raise ValueError("recipe C auxiliary head requires partial_unfreeze so it remains a separately trainable head")
    classification = config.get("classification", {})
    if not isinstance(classification, dict):
        raise ValueError("classification must be a mapping")
    if not isinstance(classification.get("enabled", False), bool):
        raise ValueError("classification.enabled must be a boolean")
    for key in ("loss_weight", "dropout"):
        if float(classification.get(key, 0.0)) < 0:
            raise ValueError(f"classification.{key} must be non-negative")
    threshold = float(classification.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("classification.threshold must be in [0, 1]")
    if float(classification.get("temperature", 1.0)) <= 0.0:
        raise ValueError("classification.temperature must be positive")
    if int(classification.get("hidden_dim", 128)) <= 0:
        raise ValueError("classification.hidden_dim must be positive")
    if float(classification.get("dropout", 0.0)) >= 1.0:
        raise ValueError("classification.dropout must be < 1")
    if str(classification.get("label_source", "paired_degraded_gt")) != "paired_degraded_gt":
        raise ValueError("classification.label_source currently supports paired_degraded_gt only")
    if not isinstance(classification.get("threshold_selection", {}), dict):
        raise ValueError("classification.threshold_selection must be a mapping")
    if str(classification.get("feature_source", "vae_latent")) != "vae_latent":
        raise ValueError("classification.feature_source only supports legacy vae_latent")
    if str(classification.get("mode", "vae_latent_mlp")) not in {
        "vae_latent_mlp",
        "legacy_repair_classifier",
    }:
        raise ValueError(
            "legacy classification requires mode=vae_latent_mlp or "
            f"legacy_repair_classifier, got {classification.get('mode')!r}"
        )
    prompt_library = config.get("prompt_library")
    if prompt_library is not None:
        if not isinstance(prompt_library, dict):
            raise ValueError("prompt_library must be a mapping after config normalization")
        if not prompt_library.get("path") or not prompt_library.get("id"):
            raise ValueError("prompt_library requires path and id")
    condition = config.get("condition_model")
    if condition is not None:
        if not isinstance(condition, dict):
            raise ValueError("condition_model must be a mapping")
        if condition_enabled(config):
            role = str(condition.get("role") or condition.get("variant_group") or "").upper()
            variant = str(condition.get("variant") or "")
            if role not in {"A", "B"}:
                role = "B" if "_B_" in variant else "A"
            if not variant:
                raise ValueError("condition_model.variant is required when condition_model is enabled")
            if role == "A" and "_A_" not in variant:
                raise ValueError(f"A condition variant must identify A, got {variant!r}")
            if role == "B" and "_B_" not in variant:
                raise ValueError(f"B condition variant must identify B, got {variant!r}")
            trim = int(condition.get("valid_token_prefix_trim", 64))
            if trim < 0:
                raise ValueError("condition_model.valid_token_prefix_trim must be non-negative")
            if condition.get("checkpoint_step") is not None and int(condition["checkpoint_step"]) < 0:
                raise ValueError("condition_model.checkpoint_step must be non-negative")


def runtime_paths(config: dict[str, Any]) -> dict[str, Path]:
    scope = config["scope"]
    cache_root = Path(config["paths"]["cache_root"])
    manifests = configured_manifests(config)
    if len(manifests) == 1:
        prepared_by_split = {"train": cache_root / "prepared" / scope / "source.jsonl"}
    else:
        prepared_by_split = {
            split: cache_root / "prepared" / scope / f"{split}.jsonl"
            for split in manifests
        }
    return {
        "cache_root": cache_root,
        "prepared": prepared_by_split["train"],
        "prepared_by_split": prepared_by_split,
        "meta_root": cache_root / "meta" / scope,
        "task_meta": cache_root / "meta" / scope / "texverse_restore",
        "embeddings": cache_root / "embeddings" / scope,
        "output_dir": Path(config["paths"]["output_dir"]),
    }
