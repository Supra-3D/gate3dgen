#!/usr/bin/env python3
"""Train the Qwen-VL V3 repair gate (A: language, B: language + vision LoRA).

This is a new namespace and intentionally does not mutate the V2 trainer.  It
keeps the native Qwen LM head and computes weighted causal-LM CE directly from
the returned logits.  The assistant answer's first ``<|im_end|>`` is part of
the labels, so the model is explicitly trained to stop after its canonical
reason instead of receiving an unbounded repeated-reason target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader
try:
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLProcessor,
        get_cosine_schedule_with_warmup,
    )
except ImportError as _TRANSFORMERS_IMPORT_ERROR:  # pragma: no cover - CPU schema tooling
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore[assignment,misc]
    Qwen2VLProcessor = None  # type: ignore[assignment,misc]

    def get_cosine_schedule_with_warmup(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError("transformers is unavailable for Qwen training") from _TRANSFORMERS_IMPORT_ERROR

from gate3dgen.training.router_tools.image_contract import inspect_image_token_contract
from gate3dgen.training.router_tools.lora import (
    IM_END_TOKEN_ID,
    V2_SCHEMA_VERSION,
    V3_SCHEMA_VERSION,
    VisionLoraConfig,
    count_parameters,
    inject_vision_lora,
    iter_lora_parameters,
    weighted_causal_lm_loss,
)
from gate3dgen.training.router_tools.common import (
    PROJECT_DIR,
    WORKSPACE,
    cycle,
    find_unique_subsequence,
    load_json,
    next_batch_or_raise_synchronized,
    prune_checkpoints,
    reset_scheduler_to_optimizer_step,
    sequence_accuracy,
    synchronize_image_contract,
    write_inflight_diagnostic,
)
from gate3dgen.training.router_tools.common import resolve_workspace_path


CANONICAL_REASON_ORDER = (
    "MOTION_BLUR", "OPTICAL_BLUR", "DIGITAL_DEGRADATION", "UNDEREXPOSURE",
    "OVEREXPOSURE", "HARD_SHADOW", "WHITE_BALANCE", "COLOR_SPILL",
    "VEILING_GLARE", "DIRTY_GLASS", "GLASS_REFLECTION", "CROP_OR_MISSING_REGION",
    "EXTERNAL_HUMAN_SUPPORT", "OTHER_VISIBLE_DEGRADATION", "MIXED", "NONE",
)
CANONICAL_REASON_INDEX = {reason: index for index, reason in enumerate(CANONICAL_REASON_ORDER)}


def validate_canonical_reasons(reasons: Sequence[str], sample_id: str = "") -> list[str]:
    """Reject repeated/out-of-order reason targets before tokenization.

    The gate is trained against one canonical comma-separated reason list.  A
    repeated reason is a malformed target, not a hard negative; accepting it
    would teach the LM to continue generating duplicate explanations.  This
    helper is kept separate from the dataset so the contract has a small CPU
    fixture test as well as the runtime check.
    """

    normalized = [str(reason) for reason in reasons]
    prefix = f"{sample_id}: " if sample_id else ""
    if not normalized:
        raise ValueError(f"{prefix}reason target must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{prefix}duplicate canonical reason target rejected: {normalized}")
    if normalized != sorted(normalized, key=lambda value: CANONICAL_REASON_INDEX.get(value, 10_000)):
        raise ValueError(f"{prefix}non-canonical reason target rejected: {normalized}")
    return normalized


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("scope") != "qwen_repair_gate":
        raise ValueError(f"invalid Qwen repair-gate V3 config: {path}")
    for key in ("data", "model", "prompt", "training", "paths", "loss", "sample_weights"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"V3 config section {key!r} must be a mapping")
    variant = str(config.get("variant", ""))
    if not variant.startswith("QWEN_VL_LM_GATE_V3_"):
        raise ValueError(f"V3 config variant must start with QWEN_VL_LM_GATE_V3_, got {variant!r}")
    return config


def validate_config(config: Mapping[str, Any], *, smoke_test: bool = False) -> None:
    model = config["model"]
    training = config["training"]
    data = config["data"]
    loss = config["loss"]
    sample_weights = config["sample_weights"]
    if tuple(int(x) for x in model.get("trainable_language_layers", [])) != (24, 27):
        raise ValueError("V3 gate requires exactly language layers 24-27")
    if model.get("train_final_norm") is not True:
        raise ValueError("V3 gate requires train_final_norm=true")
    if model.get("train_lm_head") is not False:
        raise ValueError("V3 gate keeps the native LM head frozen")
    if model.get("train_vision_encoder") is not False:
        raise ValueError("V3 gate does not full-finetune the vision encoder")
    variant = str(config["variant"])
    is_b = "_B_" in variant or variant.endswith("_B")
    lora_config = VisionLoraConfig.from_mapping(model.get("vision_lora"))
    if is_b != lora_config.enabled:
        raise ValueError(f"variant B/A and vision_lora.enabled disagree: variant={variant!r}")
    if not is_b and lora_config.enabled:
        raise ValueError("A variant cannot enable vision LoRA")
    if data.get("manifest_schema") != V2_SCHEMA_VERSION:
        raise ValueError("V3 trainer consumes V2-compatible pre-expanded manifests")
    if not bool(data.get("expanded_manifest", False)):
        raise ValueError("V3 trainer requires expanded_manifest=true")
    for name in ("structure", "decision", "reason", "eos"):
        if float(loss.get(f"{name}_token_weight", 0)) <= 0:
            raise ValueError(f"loss.{name}_token_weight must be positive")
    if float(loss.get("structure_token_weight")) != 0.5:
        raise ValueError("V3 structure token weight must be 0.5")
    if float(loss.get("decision_token_weight")) != 3.0:
        raise ValueError("V3 decision token weight must be 3.0")
    if float(loss.get("reason_token_weight")) != 1.0:
        raise ValueError("V3 reason token weight must be 1.0")
    if float(loss.get("eos_token_weight")) != 5.0:
        raise ValueError("V3 EOS token weight must be 5.0")
    if float(sample_weights.get("subtle_multiplier", 0)) != 1.25:
        raise ValueError("V3 subtle multiplier must be 1.25")
    if float(sample_weights.get("max_sample_weight", 0)) != 2.5:
        raise ValueError("V3 maximum sample weight must be 2.5")
    if is_b:
        if float(training.get("vision_lora_learning_rate_multiplier", 0)) != 10.0:
            raise ValueError("V3 B vision LoRA learning-rate multiplier must be 10")
        if float(training.get("vision_lora_weight_decay", -1)) != 0.0:
            raise ValueError("V3 B vision LoRA weight decay must be 0")
    topology = (
        int(training.get("num_processes", 0)),
        int(training.get("num_machines", 0)),
        int(training.get("num_processes_per_machine", 0)),
    )
    if not smoke_test and topology not in {(32, 4, 8), (24, 3, 8), (16, 2, 8), (8, 1, 8)}:
        raise ValueError(
            "V3 gate requires 4x8=32, 3x8=24, 2x8=16, or 1x8=8 ranks, "
            f"got {topology}"
        )
    if int(training.get("per_device_batch_size", 0)) != 1:
        raise ValueError("V3 recipes require per_device_batch_size=1")
    if int(training.get("max_train_steps", 0)) <= 0:
        raise ValueError("max_train_steps must be positive")
    if not bool(data.get("demo_validation", True)):
        raise ValueError("V3 training must validate on independent benchmark demo data")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_weight_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"V3 sample-weight sidecar missing: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema_version") != "qwen-repair-gate-v3-sample-weights":
                raise ValueError(f"{path}:{line_number}: invalid V3 weight schema")
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate/empty sample_id")
            weight = float(row.get("sample_weight", 0))
            if weight <= 0 or weight > 2.5:
                raise ValueError(f"{path}:{line_number}: sample_weight outside (0,2.5]: {weight}")
            if row.get("severity_tag_provenance_only") is not True:
                raise ValueError(f"{path}:{line_number}: severity tag is not provenance-only")
            if bool(row.get("subtle_proxy")) and str(row.get("decision")) != "REPAIR":
                raise ValueError(f"{path}:{line_number}: only REPAIR samples may be subtle")
            rows[sample_id] = row
    if not rows:
        raise ValueError(f"empty V3 sample-weight sidecar: {path}")
    return rows


class GateV3Dataset(torch.utils.data.Dataset):
    """Strict V2-compatible expanded rows plus V3 sample weight metadata."""

    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        prompt_spec: Mapping[str, Any],
        expected_examples: int | None,
        image_sample_size: int,
        condition_image_size: int,
        *,
        weights: Mapping[str, Mapping[str, Any]] | None = None,
        require_weights: bool = False,
        limit_examples: int | None = None,
        texverse_root: Path | None = None,
        texverse_full_root: Path | None = None,
    ) -> None:
        from gate3dgen.training.router_tools.common import RepairGateDataset

        self._base = RepairGateDataset(
            manifest,
            data_root,
            dict(prompt_spec),
            expected_examples,
            image_sample_size,
            condition_image_size,
            limit_pairs=None,
            expanded_manifest=True,
            texverse_root=texverse_root,
            texverse_full_root=texverse_full_root,
        )
        self.rows = self._base.rows
        if limit_examples is not None:
            self.rows = self.rows[: int(limit_examples)]
        self._base.rows = self.rows
        self.weights = dict(weights or {})
        self.require_weights = bool(require_weights)
        if self.require_weights:
            missing = [str(row["sample_id"]) for row in self.rows if str(row["sample_id"]) not in self.weights]
            if missing:
                raise ValueError(
                    f"V3 required sample-weight sidecar is missing {len(missing)} rows, e.g. {missing[:3]}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        reasons = validate_canonical_reasons(row.get("reasons", []), str(row["sample_id"]))
        example = self._base[index]
        example["answer"] = f"DECISION: {example['decision']}\nREASONS: {','.join(reasons)}"
        weight_row = self.weights.get(str(row["sample_id"]))
        if weight_row is None:
            weight = 1.0
            subtle = False
        else:
            weight = float(weight_row["sample_weight"])
            subtle = bool(weight_row.get("subtle_proxy"))
        example["sample_weight"] = weight
        example["subtle_proxy"] = subtle
        example["sample_id"] = str(row["sample_id"])
        example["task_id"] = str(row.get("task") or row.get("task_id") or "")
        return example


def _span_weight(
    answer: str,
    answer_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]] | None,
    *,
    decision: str,
    reasons: str,
    structure_weight: float,
    decision_weight: float,
    reason_weight: float,
) -> list[float]:
    """Assign structural/decision/reason weights without subword heuristics."""

    decision_start = answer.index(decision)
    reason_start = answer.index(reasons)
    result: list[float] = []
    for index, _token_id in enumerate(answer_ids):
        if offsets is None or index >= len(offsets):
            # The Qwen fast tokenizer normally provides offsets.  A fallback
            # is kept for minimal CPU fixtures, but it is still deterministic
            # and does not use a token substring (which could split a reason).
            result.append(reason_weight)
            continue
        start, end = offsets[index]
        if end <= start:
            result.append(structure_weight)
        elif start < decision_start + len(decision) and end > decision_start:
            result.append(decision_weight)
        elif start < reason_start + len(reasons) and end > reason_start:
            result.append(reason_weight)
        else:
            result.append(structure_weight)
    return result


class GateV3Collator:
    def __init__(self, processor: Any, loss_config: Mapping[str, Any]) -> None:
        self.processor = processor
        tokenizer = processor.tokenizer
        self.image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.spatial_merge_size = int(processor.image_processor.merge_size)
        eos_id = int(getattr(tokenizer, "eos_token_id", IM_END_TOKEN_ID) or IM_END_TOKEN_ID)
        if eos_id != IM_END_TOKEN_ID:
            raise ValueError(f"V3 requires Qwen assistant EOS id {IM_END_TOKEN_ID}, got {eos_id}")
        self.eos_token_id = IM_END_TOKEN_ID
        self.structure_weight = float(loss_config["structure_token_weight"])
        self.decision_weight = float(loss_config["decision_token_weight"])
        self.reason_weight = float(loss_config["reason_token_weight"])
        self.eos_weight = float(loss_config["eos_token_weight"])

    def _image_contract(self, model_inputs: dict[str, torch.Tensor], examples: list[dict[str, Any]]) -> dict[str, Any]:
        return inspect_image_token_contract(
            model_inputs,
            examples,
            image_token_id=self.image_token_id,
            spatial_merge_size=self.spatial_merge_size,
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        conversations = []
        for example in examples:
            conversations.append([
                {"role": "system", "content": [{"type": "text", "text": example["system_prompt"]}]},
                {"role": "user", "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["user_prompt"]},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
            ])
        model_inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            padding=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        labels = torch.full_like(model_inputs["input_ids"], -100)
        token_weights = torch.zeros_like(model_inputs["input_ids"], dtype=torch.float32)
        attention_mask = model_inputs["attention_mask"]
        for batch_index, example in enumerate(examples):
            valid_positions = attention_mask[batch_index].nonzero(as_tuple=True)[0]
            valid_ids = model_inputs["input_ids"][batch_index, valid_positions].tolist()
            encoded = self.processor.tokenizer(
                example["answer"],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            answer_ids = list(encoded.get("input_ids", getattr(encoded, "input_ids", [])))
            if not answer_ids:
                raise ValueError(f"tokenizer produced no answer tokens: {example['sample_id']}")
            if self.eos_token_id in answer_ids:
                raise ValueError(f"answer must not contain EOS before the explicit terminator: {example['sample_id']}")
            answer_start = find_unique_subsequence(valid_ids, answer_ids)
            answer_end = answer_start + len(answer_ids)
            if answer_end >= len(valid_ids) or int(valid_ids[answer_end]) != self.eos_token_id:
                raise ValueError(
                    f"answer is not immediately followed by Qwen EOS {self.eos_token_id}: "
                    f"sample={example['sample_id']} next={valid_ids[answer_end:answer_end + 2]}"
                )
            supervised_positions = valid_positions[answer_start:answer_end]
            labels[batch_index, supervised_positions] = model_inputs["input_ids"][batch_index, supervised_positions]
            answer_text = str(example["answer"])
            decision = str(example["decision"])
            reasons_text = ",".join(str(reason) for reason in example["reasons"])
            offsets = encoded.get("offset_mapping")
            local_weights = _span_weight(
                answer_text,
                answer_ids,
                offsets,
                decision=decision,
                reasons=reasons_text,
                structure_weight=self.structure_weight,
                decision_weight=self.decision_weight,
                reason_weight=self.reason_weight,
            )
            token_weights[batch_index, supervised_positions] = torch.tensor(local_weights, dtype=torch.float32)
            eos_position = valid_positions[answer_end]
            labels[batch_index, eos_position] = self.eos_token_id
            token_weights[batch_index, eos_position] = self.eos_weight
        if not bool((labels != -100).any()):
            raise RuntimeError("V3 collator produced no supervised answer/EOS tokens")
        output = dict(model_inputs)
        output["labels"] = labels
        output["token_weights"] = token_weights
        output["sample_weights"] = torch.tensor(
            [float(example.get("sample_weight", 1.0)) for example in examples], dtype=torch.float32
        )
        output["pair_ids"] = [example["pair_id"] for example in examples]
        output["sample_ids"] = [example["sample_id"] for example in examples]
        output["sample_kinds"] = [example["sample_kind"] for example in examples]
        output["gate_image_contract"] = self._image_contract(model_inputs, examples)
        return output


def pop_gate_metadata(batch: dict[str, Any]) -> dict[str, Any]:
    diagnostics = batch.pop("gate_image_contract")
    batch.pop("pair_ids")
    batch.pop("sample_ids")
    batch.pop("sample_kinds")
    return diagnostics


def _pop_loss_metadata(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    token_weights = batch.pop("token_weights")
    sample_weights = batch.pop("sample_weights")
    return token_weights, sample_weights


def validate_v3_resume(config: Mapping[str, Any], resume: Path | None, expected: Mapping[str, Any]) -> None:
    if resume is None:
        return
    state_path = resume / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError(f"V3 resume requires trainer_state.json: {state_path}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read V3 resume trainer_state: {state_path}") from exc
    for key in (
        "variant", "world_size", "config_sha256", "prompt_sha256", "train_manifest_sha256",
        "validation_manifest_sha256", "secondary_validation_manifest_sha256",
        "sample_weights_sha256", "loss_weights", "lora_config",
    ):
        if key not in state:
            raise ValueError(f"V3 resume state missing immutable field {key!r}")
        if key in expected and state[key] != expected[key]:
            raise ValueError(
                f"V3 resume mismatch for {key}: checkpoint={state[key]!r}, current={expected[key]!r}"
            )
    if int(state.get("world_size")) != int(config["training"]["num_processes"]):
        raise ValueError("V3 resume world-size mismatch; FSDP shard topology must match exactly")


@torch.no_grad()
def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    loader: DataLoader,
) -> dict[str, float]:
    model.eval()
    loss_sum = torch.zeros((), device=accelerator.device)
    correct_sum = torch.zeros((), device=accelerator.device, dtype=torch.long)
    example_sum = torch.zeros((), device=accelerator.device, dtype=torch.long)
    batch_sum = torch.zeros((), device=accelerator.device, dtype=torch.long)
    for validation_batch_index, batch in enumerate(loader):
        diagnostics = pop_gate_metadata(batch)
        synchronize_image_contract(accelerator, diagnostics, global_step=-(validation_batch_index + 1))
        token_weights, sample_weights = _pop_loss_metadata(batch)
        outputs = model(**batch, use_cache=False)
        loss = weighted_causal_lm_loss(outputs.logits, batch["labels"], token_weights, sample_weights)
        correct, count = sequence_accuracy(outputs.logits, batch["labels"])
        loss_sum += loss.detach().float()
        correct_sum += correct
        example_sum += count
        batch_sum += 1
    stats = torch.stack([loss_sum, correct_sum.float(), example_sum.float(), batch_sum.float()])
    stats = accelerator.reduce(stats, reduction="sum")
    model.train()
    return {
        "validation_loss": float((stats[0] / stats[3].clamp_min(1)).item()),
        "validation_exact_answer_accuracy": float((stats[1] / stats[2].clamp_min(1)).item()),
    }


def save_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    step: int,
    checkpoints_total_limit: int,
    metadata: Mapping[str, Any],
) -> None:
    checkpoint = output_dir / f"checkpoint-{step}"
    accelerator.save_state(str(checkpoint), safe_serialization=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state = dict(metadata)
        state["global_step"] = step
        (checkpoint / "trainer_state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (checkpoint / "checkpoint_complete.json").write_text(
            json.dumps({"global_step": step, "status": "complete", "variant": metadata["variant"]}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (checkpoint / "v3_lora_config.json").write_text(
            json.dumps(
                {
                    "variant": metadata["variant"],
                    "lora_config": metadata["lora_config"],
                    "lora_target_names": metadata["lora_target_names"],
                    "status": "complete",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        prune_checkpoints(output_dir, checkpoints_total_limit)
    accelerator.wait_for_everyone()


def _model_trainable_contract(model: torch.nn.Module, *, first_layer: int, last_layer: int, is_b: bool, lora_names: Sequence[str]) -> list[str]:
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    allowed = tuple(
        [f"model.language_model.layers.{index}." for index in range(first_layer, last_layer + 1)]
        + ["model.language_model.norm."]
    )
    unexpected: list[str] = []
    for name in trainable_names:
        if name.startswith(allowed):
            continue
        if is_b and any(name.startswith(f"{target}.") and ("lora_A" in name or "lora_B" in name) for target in lora_names):
            continue
        unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"unexpected V3 trainable Qwen parameters: {unexpected[:20]}")
    if is_b:
        actual_lora = [name for name, _parameter in iter_lora_parameters(model)]
        if not actual_lora or len(actual_lora) != len(lora_names) * 2:
            raise RuntimeError(
                f"V3 B expected {len(lora_names) * 2} LoRA tensors, got {len(actual_lora)}"
            )
    elif any("lora_" in name for name in trainable_names):
        raise RuntimeError("V3 A unexpectedly contains LoRA parameters")
    if not trainable_names:
        raise RuntimeError("no trainable Qwen parameters")
    return trainable_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    validate_config(config, smoke_test=args.smoke_test)
    if Qwen2_5_VLForConditionalGeneration is None or Qwen2VLProcessor is None:
        raise RuntimeError("Qwen V3 training requires a compatible transformers installation")
    data_config = config["data"]
    model_config = config["model"]
    train_config = dict(config["training"])
    if args.smoke_test:
        train_config.update(
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            num_workers=0,
            max_train_steps=1,
            warmup_steps=0,
            logging_steps=1,
            validation_steps=1,
            checkpointing_steps=2,
        )
    variant = str(config["variant"])
    is_b = "_B_" in variant or variant.endswith("_B")
    lora_config = VisionLoraConfig.from_mapping(model_config.get("vision_lora"))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )
    if not args.smoke_test and accelerator.num_processes != int(train_config["num_processes"]):
        raise RuntimeError(
            f"V3 config/runtime world size mismatch: config={train_config['num_processes']} runtime={accelerator.num_processes}"
        )
    seed = int(train_config["seed"])
    random.seed(seed + accelerator.process_index)
    torch.manual_seed(seed + accelerator.process_index)

    model_root = resolve_workspace_path(os.environ.get("GATE3DGEN_MODEL_PATH", model_config["pretrained_model_path"]))
    prompt_path = resolve_workspace_path(os.environ.get("QWEN_GATE_V3_PROMPT", config["prompt"]["path"]))
    train_manifest_path = resolve_workspace_path(os.environ.get("QWEN_GATE_V3_TRAIN_MANIFEST", data_config["train_manifest"]))
    validation_manifest_path = resolve_workspace_path(os.environ.get("QWEN_GATE_V3_VALIDATION_MANIFEST", data_config["validation_manifest"]))
    secondary_validation_path = resolve_workspace_path(os.environ.get("QWEN_GATE_V3_SECONDARY_VALIDATION_MANIFEST", data_config.get("secondary_validation_manifest", data_config["validation_manifest"])))
    weights_path = resolve_workspace_path(os.environ.get("QWEN_GATE_V3_WEIGHTS_MANIFEST", config["sample_weights"]["path"]))
    for required in (model_root / "text_encoder/config.json", model_root / "processor", prompt_path, train_manifest_path, validation_manifest_path, weights_path):
        if not required.exists():
            raise FileNotFoundError(f"required V3 input missing: {required}")
    prompt_spec = load_json(prompt_path)
    weight_rows = _read_weight_sidecar(weights_path)
    if bool(config["sample_weights"].get("require_subtle_proxy", False)):
        audit_path = weights_path.parent / "weights_audit.json"
        if not audit_path.is_file():
            raise ValueError(f"subtle proxy is required but weight audit is missing: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        proxy = audit.get("subtle_proxy") if isinstance(audit, dict) else None
        if not isinstance(proxy, dict) or proxy.get("status") != "ok" or int(audit.get("subtle_rows", 0)) <= 0:
            raise ValueError(
                "subtle proxy is required but the sidecar audit is not complete; "
                f"status={proxy!r} subtle_rows={audit.get('subtle_rows') if isinstance(audit, dict) else None}"
            )
    train_manifest_sha = _sha256_file(train_manifest_path)
    validation_manifest_sha = _sha256_file(validation_manifest_path)
    secondary_validation_sha = _sha256_file(secondary_validation_path)
    weights_sha = _sha256_file(weights_path)
    prompt_sha = _sha256_file(prompt_path)
    expected_resume = {
        "variant": variant,
        "world_size": int(train_config["num_processes"]),
        "config_sha256": _json_sha(config),
        "prompt_sha256": prompt_sha,
        "train_manifest_sha256": train_manifest_sha,
        "validation_manifest_sha256": validation_manifest_sha,
        "secondary_validation_manifest_sha256": secondary_validation_sha,
        "sample_weights_sha256": weights_sha,
        "loss_weights": dict(config["loss"]),
        "lora_config": lora_config.as_dict(),
    }
    validate_v3_resume(config, args.resume, expected_resume)

    processor = Qwen2VLProcessor.from_pretrained(model_root, subfolder="processor")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_root / "text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.requires_grad_(False)
    lora_names = inject_vision_lora(model, lora_config) if is_b else []
    first_layer, last_layer = (int(value) for value in model_config["trainable_language_layers"])
    # Re-enable language layers after LoRA injection by design: this prevents
    # a broad model freeze/injection helper from accidentally freezing A/B LM
    # parameters after the adapter has been installed.
    for layer_index in range(first_layer, last_layer + 1):
        model.language_model.layers[layer_index].requires_grad_(True)
    model.language_model.norm.requires_grad_(True)
    model.lm_head.requires_grad_(False)
    model.visual.requires_grad_(False)
    for _name, parameter in iter_lora_parameters(model):
        parameter.requires_grad_(True)
    model.config.use_cache = False
    if bool(model_config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable_names = _model_trainable_contract(
        model, first_layer=first_layer, last_layer=last_layer, is_b=is_b, lora_names=lora_names
    )
    counts = count_parameters(model)

    data_root = resolve_workspace_path(data_config["data_root"])
    texverse_root = resolve_workspace_path(os.environ.get("QWEN_GATE_TEXVERSE_ROOT", data_config.get("texverse_root", "TexVerse")))
    texverse_full_root = resolve_workspace_path(os.environ.get("QWEN_GATE_TEXVERSE_FULL_ROOT", data_config.get("texverse_full_root", "TexVerse-full")))
    train_dataset = GateV3Dataset(
        train_manifest_path,
        data_root,
        prompt_spec,
        data_config.get("expected_train_examples"),
        int(data_config["image_sample_size"]),
        int(data_config["condition_image_size"]),
        weights=weight_rows,
        require_weights=True,
        limit_examples=1 if args.smoke_test else None,
        texverse_root=texverse_root,
        texverse_full_root=texverse_full_root,
    )
    validation_dataset = GateV3Dataset(
        validation_manifest_path,
        data_root,
        prompt_spec,
        data_config.get("expected_validation_examples"),
        int(data_config["image_sample_size"]),
        int(data_config["condition_image_size"]),
        weights=None,
        require_weights=False,
        limit_examples=1 if args.smoke_test else None,
        texverse_root=texverse_root,
        texverse_full_root=texverse_full_root,
    )
    secondary_dataset = GateV3Dataset(
        secondary_validation_path,
        data_root,
        prompt_spec,
        data_config.get("expected_secondary_validation_examples"),
        int(data_config["image_sample_size"]),
        int(data_config["condition_image_size"]),
        weights=None,
        require_weights=False,
        limit_examples=1 if args.smoke_test else None,
        texverse_root=texverse_root,
        texverse_full_root=texverse_full_root,
    )
    collator = GateV3Collator(processor, config["loss"])
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=int(train_config["num_workers"]),
        pin_memory=True, persistent_workers=int(train_config["num_workers"]) > 0, collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=int(train_config["num_workers"]),
        pin_memory=True, persistent_workers=int(train_config["num_workers"]) > 0, collate_fn=collator,
    )
    secondary_loader = DataLoader(
        secondary_dataset, batch_size=1, shuffle=False, num_workers=int(train_config["num_workers"]),
        pin_memory=True, persistent_workers=int(train_config["num_workers"]) > 0, collate_fn=collator,
    )
    language_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not (".lora_A" in name or ".lora_B" in name)
    ]
    lora_parameters = [parameter for _name, parameter in iter_lora_parameters(model) if parameter.requires_grad]
    parameter_groups = [{"params": language_parameters, "lr": float(train_config["learning_rate"]), "weight_decay": float(train_config["weight_decay"])}]
    if is_b:
        parameter_groups.append(
            {
                "params": lora_parameters,
                "lr": float(train_config["learning_rate"]) * float(train_config["vision_lora_learning_rate_multiplier"]),
                "weight_decay": float(train_config["vision_lora_weight_decay"]),
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(train_config["warmup_steps"]), num_training_steps=int(train_config["max_train_steps"])
    )
    model, optimizer, train_loader, validation_loader, secondary_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, secondary_loader, scheduler
    )
    start_step = int(args.resume.name.split("-")[-1]) if args.resume else 0
    if args.resume:
        accelerator.load_state(str(args.resume.resolve()))
        repaired_lrs = reset_scheduler_to_optimizer_step(scheduler, start_step)
        accelerator.print(f"V3 resume scheduler repaired: optimizer_step={start_step} lr={repaired_lrs[0]:.3e}")

    output_dir = resolve_workspace_path(os.environ.get("QWEN_GATE_V3_OUTPUT_ROOT", config["paths"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **expected_resume,
        "variant": variant,
        "base_model": str(model_root),
        "config_sha256": expected_resume["config_sha256"],
        "prompt_path": str(prompt_path),
        "trainable_language_layers": [first_layer, last_layer],
        "train_final_norm": True,
        "train_lm_head": False,
        "vision_encoder_frozen": True,
        "restoration_modules_loaded": False,
        "manifest_schema": V2_SCHEMA_VERSION,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "secondary_validation_examples": len(secondary_dataset),
        "benchmark_demo_validation": str(validation_manifest_path),
        "secondary_in_domain_validation": str(secondary_validation_path),
        "sample_weights_path": str(weights_path),
        "sample_weights_policy": dict(config["sample_weights"]),
        "loss_weights": dict(config["loss"]),
        "lora_config": lora_config.as_dict(),
        "lora_implementation_version": "qwen_gate_v3_local_lora_v1",
        "lora_learning_rate_multiplier": float(train_config.get("vision_lora_learning_rate_multiplier", 10.0)),
        "lora_weight_decay": float(train_config.get("vision_lora_weight_decay", 0.0)),
        "lora_target_names": lora_names,
        "checkpoint_state_format": "complete_qwen_wrapper_sharded_state_dict",
        "export_requires_accelerate_merge_weights": bool(is_b),
        "parameter_counts": counts,
        "trainable_parameter_names_count": len(trainable_names),
        "trainable_parameter_names": trainable_names,
        "layer_indexing": "zero_based_module_indices",
        "smoke_test": args.smoke_test,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scheduler_unit": "global_optimizer_step",
        "image_token_contract_validation": True,
        "eos_token_id": IM_END_TOKEN_ID,
        "eos_supervision": "first assistant EOS immediately after canonical answer; fail closed otherwise",
        "repeat_reason_policy": "canonical duplicate targets rejected; no unlikelihood loss",
    }
    if accelerator.is_main_process:
        (output_dir / "run_config.json").write_text(json.dumps({"config": config, "metadata": metadata}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        accelerator.print(
            f"Qwen gate V3 {variant}: trainable={counts['trainable']} params, "
            f"layers={first_layer}-{last_layer}, final_norm=True, lm_head=False, "
            f"vision_lora={is_b}, mmDiT=False"
        )

    max_steps = int(train_config["max_train_steps"])
    logging_steps = int(train_config["logging_steps"])
    validation_steps = int(train_config["validation_steps"])
    checkpointing_steps = int(train_config["checkpointing_steps"])
    model.train()
    train_iterator = iter(cycle(train_loader))
    global_step = start_step
    micro_step = start_step * int(train_config["gradient_accumulation_steps"])
    while global_step < max_steps:
        micro_step += 1
        batch = next_batch_or_raise_synchronized(accelerator, train_iterator, micro_step)
        diagnostics = pop_gate_metadata(batch)
        write_inflight_diagnostic(diagnostics_dir, accelerator.process_index, micro_step, diagnostics)
        synchronize_image_contract(accelerator, diagnostics, micro_step)
        with accelerator.accumulate(model):
            token_weights, sample_weights = _pop_loss_metadata(batch)
            outputs = model(**batch, use_cache=False)
            loss = weighted_causal_lm_loss(outputs.logits, batch["labels"], token_weights, sample_weights)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), float(train_config["max_grad_norm"]))
            optimizer.step()
            if accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if not accelerator.sync_gradients:
            continue
        global_step += 1
        if global_step % logging_steps == 0:
            mean_loss = accelerator.reduce(loss.detach().float(), reduction="mean")
            accelerator.print(f"step={global_step}/{max_steps} weighted_ce_loss={mean_loss.item():.6f} lr={scheduler.get_last_lr()[0]:.3e}")
        if global_step % validation_steps == 0:
            demo_metrics = evaluate(accelerator, model, validation_loader)
            domain_metrics = evaluate(accelerator, model, secondary_loader)
            accelerator.print(
                f"step={global_step} demo_validation_loss={demo_metrics['validation_loss']:.6f} "
                f"demo_exact={demo_metrics['validation_exact_answer_accuracy']:.6f} "
                f"in_domain_loss={domain_metrics['validation_loss']:.6f} "
                f"in_domain_exact={domain_metrics['validation_exact_answer_accuracy']:.6f}"
            )
        if not args.smoke_test and global_step % checkpointing_steps == 0:
            save_checkpoint(accelerator, output_dir, global_step, int(train_config["checkpoints_total_limit"]), metadata)
    if not args.smoke_test and max_steps % checkpointing_steps != 0:
        save_checkpoint(accelerator, output_dir, max_steps, int(train_config["checkpoints_total_limit"]), metadata)
    accelerator.wait_for_everyone()
    if args.smoke_test and accelerator.is_main_process:
        smoke_result = {
            "status": "complete",
            "variant": variant,
            "global_step": max_steps,
            "parameter_counts": counts,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
        }
        (output_dir / "smoke_complete.json").write_text(json.dumps(smoke_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        accelerator.print(json.dumps(smoke_result, sort_keys=True))
    accelerator.end_training()


if __name__ == "__main__":
    main()

