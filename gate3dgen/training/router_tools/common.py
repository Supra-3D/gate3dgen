"""Shared dataset and distributed helpers extracted from the original router trainer."""
from __future__ import annotations


import argparse


import hashlib


import json


import math


import os


import random


import shutil


import traceback


from pathlib import Path


from typing import Any, Iterable


import torch


import torch.distributed as dist


import yaml


from accelerate import Accelerator


from PIL import Image


from torch.utils.data import DataLoader, Dataset


try:
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLProcessor,
        get_cosine_schedule_with_warmup,
    )
except ImportError as _TRANSFORMERS_IMPORT_ERROR:  # pragma: no cover - CPU schema tooling
    # Keep manifest/prompt contract readers importable in lightweight audit
    # environments.  The training entry point below raises the original
    # dependency error before touching model code.
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore[assignment,misc]
    Qwen2VLProcessor = None  # type: ignore[assignment,misc]

    def get_cosine_schedule_with_warmup(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError("transformers is unavailable for Qwen training") from _TRANSFORMERS_IMPORT_ERROR


from gate3dgen.training.router_tools.image_contract import inspect_image_token_contract


PROJECT_DIR = Path(__file__).resolve().parents[1]


WORKSPACE = Path(os.environ.get("GATE3DGEN_ROOT", Path.cwd())).resolve()


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_manifest(path: Path, expected_pairs: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("pair_id", "task_id", "degraded_relative_path", "gt_relative_path"):
                if not row.get(key):
                    raise ValueError(f"{path}:{line_number} missing {key}")
            rows.append(row)
    if expected_pairs is not None and len(rows) != int(expected_pairs):
        raise ValueError(
            f"manifest pair count mismatch for {path}: expected {expected_pairs}, got {len(rows)}"
        )
    return rows


def read_expanded_manifest(path: Path, expected_examples: int | None) -> list[dict[str, Any]]:
    """Read the V2 one-image manifest without inferring labels from task names.

    V1 expanded a pair inside the dataloader and consequently had to infer
    ``REPAIR``/``KEEP`` from the position in the pair.  V2 records already
    carry the audited image kind, decision, and (possibly multiple) reasons.
    Keeping this reader strict makes it impossible for a stale pair manifest
    or an unknown ``*_heavy`` task name to silently alter labels.
    """

    rows: list[dict[str, Any]] = []
    required = (
        "sample_id",
        "pair",
        "asset",
        "split",
        "image_kind",
        "image",
        "image_sha256",
        "decision",
        "reasons",
        "task",
        "degradation",
        "severity_tag",
        "severity_tag_provenance_only",
        "hand_remove_member",
        "label_provenance",
        "one_image_contract",
    )
    image_kinds = {"degraded", "gt_clean", "albedo_render"}
    decisions = {"REPAIR", "KEEP"}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            if row.get("schema_version") != "qwen-repair-gate-v2":
                raise ValueError(
                    f"{path}:{line_number} requires schema_version=qwen-repair-gate-v2, "
                    f"got {row.get('schema_version')!r}"
                )
            missing = [key for key in required if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_number} missing V2 fields {missing}")
            if row.get("image_kind") not in image_kinds:
                raise ValueError(f"{path}:{line_number} invalid image_kind {row.get('image_kind')!r}")
            if row.get("decision") not in decisions:
                raise ValueError(f"{path}:{line_number} invalid decision {row.get('decision')!r}")
            if not isinstance(row.get("reasons"), list) or not row["reasons"]:
                raise ValueError(f"{path}:{line_number} reasons must be a non-empty list")
            if any(not isinstance(reason, str) for reason in row["reasons"]):
                raise ValueError(f"{path}:{line_number} reasons must contain strings")
            if row["decision"] == "KEEP" and row["reasons"] != ["NONE"]:
                raise ValueError(f"{path}:{line_number} KEEP requires reasons=[NONE]")
            if row["decision"] == "REPAIR" and "NONE" in row["reasons"]:
                raise ValueError(f"{path}:{line_number} REPAIR cannot contain NONE")
            if row.get("severity_tag_provenance_only") is not True:
                raise ValueError(f"{path}:{line_number} severity_tag must be provenance-only")
            image = row["image"]
            if not isinstance(image, dict) or image.get("root") not in {
                "workspace", "texverse", "texverse_full", "TexVerse", "TexVerse-full"
            }:
                raise ValueError(f"{path}:{line_number} image is not a portable ref")
            if not image.get("relative_path") or Path(str(image["relative_path"])).is_absolute() or ".." in Path(str(image["relative_path"])).parts:
                raise ValueError(f"{path}:{line_number} image relative_path is not portable")
            if str(image.get("sha256", "")).lower() != str(row.get("image_sha256", "")).lower():
                raise ValueError(f"{path}:{line_number} image SHA aliases disagree")
            if row.get("one_image_contract") is not True:
                raise ValueError(f"{path}:{line_number} one_image_contract must be true")
            rows.append(row)
    if expected_examples is not None and len(rows) != int(expected_examples):
        raise ValueError(
            f"expanded manifest count mismatch for {path}: expected {expected_examples}, got {len(rows)}"
        )
    return rows


def format_answer(template: str, decision: str, reason: str) -> str:
    answer = template.format(decision=decision, reason=reason)
    expected = f"DECISION: {decision}\nREASON: {reason}"
    if answer != expected:
        raise ValueError(f"answer_template must render exactly {expected!r}, got {answer!r}")
    return answer


def format_v2_answer(template: str, decision: str, reasons: list[str]) -> str:
    """Render the fixed V2 two-line multi-reason answer contract."""

    reason_text = ",".join(str(reason) for reason in reasons)
    answer = template.format(decision=decision, reasons=reason_text)
    expected = f"DECISION: {decision}\nREASONS: {reason_text}"
    if answer != expected:
        raise ValueError(f"V2 answer_template must render exactly {expected!r}, got {answer!r}")
    return answer


class RepairGateDataset(Dataset):
    """Expose each pair as one degraded REPAIR and one clean KEEP example."""

    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        prompt_spec: dict[str, Any],
        expected_pairs: int | None,
        image_sample_size: int,
        condition_image_size: int,
        limit_pairs: int | None = None,
        expanded_manifest: bool = False,
        texverse_root: Path | None = None,
        texverse_full_root: Path | None = None,
    ) -> None:
        self.expanded_manifest = bool(expanded_manifest)
        self.rows = (
            read_expanded_manifest(manifest, expected_pairs)
            if self.expanded_manifest
            else read_manifest(manifest, expected_pairs)
        )
        if limit_pairs is not None:
            self.rows = self.rows[: int(limit_pairs)]
        self.data_root = data_root
        self.texverse_root = texverse_root or (data_root / "TexVerse")
        self.texverse_full_root = texverse_full_root or (data_root / "TexVerse-full")
        self.system_prompt = str(prompt_spec["system_prompt"])
        self.user_prompt = str(prompt_spec["user_prompt"])
        self.answer_template = str(prompt_spec["answer_template"])
        self.reason_mapping = dict(prompt_spec["task_reason_mapping"])
        self.reason_categories = set(prompt_spec["reason_categories"])
        self.image_sample_size = int(image_sample_size)
        self.condition_image_size = int(condition_image_size)
        self._expanded_gt_rows_by_pair = (
            {
                str(row["pair"]): row
                for row in self.rows
                if str(row.get("image_kind")) == "gt_clean"
            }
            if self.expanded_manifest
            else {}
        )
        if not self.expanded_manifest:
            unknown_tasks = sorted({str(row["task_id"]) for row in self.rows} - set(self.reason_mapping))
            if unknown_tasks:
                raise ValueError(f"prompt has no fixed reason mapping for tasks: {unknown_tasks}")
        mapped_reasons = set(self.reason_mapping.values())
        if not mapped_reasons <= self.reason_categories:
            raise ValueError(
                f"task_reason_mapping contains undeclared categories: "
                f"{sorted(mapped_reasons - self.reason_categories)}"
            )
        if "NONE" not in self.reason_categories:
            raise ValueError("reason_categories must contain NONE")

    def __len__(self) -> int:
        return len(self.rows) if self.expanded_manifest else 2 * len(self.rows)

    def _image_path(self, row: dict[str, Any], is_degraded: bool) -> Path:
        key = "degraded_relative_path" if is_degraded else "gt_relative_path"
        path = (self.data_root / str(row[key])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing gate image for pair {row['pair_id']}: {path}")
        return path

    def _expanded_declared_path(self, row: dict[str, Any]) -> Path:
        image = row["image"]
        root = str(image["root"])
        if root in {"texverse", "TexVerse"}:
            root_path = Path(os.path.abspath(os.fspath(self.texverse_root)))
        elif root in {"texverse_full", "TexVerse-full"}:
            root_path = Path(os.path.abspath(os.fspath(self.texverse_full_root)))
        elif root == "workspace":
            root_path = Path(os.path.abspath(os.fspath(self.data_root)))
        else:
            raise ValueError(f"V2 image root is not portable: {root!r}")
        relative = Path(str(image["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"V2 image path is not portable: {relative}")
        # Lexical normalization avoids a realpath/NFS metadata lookup for
        # every one-image sample; opening the image below remains fail-closed.
        path = Path(os.path.abspath(os.path.join(os.fspath(root_path), os.fspath(relative))))
        try:
            path.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(f"V2 image path escapes {root}: {relative}") from exc
        return path

    def _expanded_image_path(self, row: dict[str, Any]) -> tuple[Path, str | None]:
        path = self._expanded_declared_path(row)
        if path.is_file():
            return path, None

        # The albedo sidecar can lose files after manifest creation. Its paired
        # reviewed GT is also a valid KEEP/NONE sample, so substitute that GT
        # deterministically. Missing degraded/GT images remain hard failures.
        if str(row.get("image_kind")) == "albedo_render":
            gt_row = self._expanded_gt_rows_by_pair.get(str(row.get("pair")))
            if gt_row is not None:
                gt_path = self._expanded_declared_path(gt_row)
                if gt_path.is_file():
                    return gt_path, "missing_albedo_render_to_paired_gt_clean"
        raise FileNotFoundError(
            f"missing V2 gate image for sample {row['sample_id']}: {path}; "
            "no valid paired GT fallback"
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.expanded_manifest:
            row = self.rows[index]
            image_path, image_fallback = self._expanded_image_path(row)
            decision = str(row["decision"])
            reasons = [str(reason) for reason in row["reasons"]]
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                from gate3dgen.training.vendor.train.src.utils.image_utils import apply_condition_transform_to_images

                image = apply_condition_transform_to_images(
                    [image],
                    ref_height=image.height,
                    ref_width=image.width,
                    image_sample_size=self.image_sample_size,
                    condition_image_size=self.condition_image_size,
                )[0]
            return {
                "dataset_index": int(index),
                "sample_id": str(row["sample_id"]),
                "pair_id": str(row["pair"]),
                "task_id": str(row["task"]),
                "sample_kind": str(row["image_kind"]),
                "decision": decision,
                "reason": reasons[0] if len(reasons) == 1 else ",".join(reasons),
                "reasons": reasons,
                "answer": format_v2_answer(self.answer_template, decision, reasons),
                "system_prompt": self.system_prompt,
                "user_prompt": self.user_prompt,
                "image_path": str(image_path),
                "image_fallback": image_fallback,
                "image": image,
            }
        row = self.rows[index // 2]
        is_degraded = index % 2 == 0
        decision = "REPAIR" if is_degraded else "KEEP"
        reason = self.reason_mapping[str(row["task_id"])] if is_degraded else "NONE"
        image_path = self._image_path(row, is_degraded)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            # Match FireRed's Qwen condition preprocessing: deterministic
            # center crop at image_sample_size, then resize to the configured
            # condition-token area. Import lazily so this standalone task does
            # not pull in any restoration model.
            from gate3dgen.training.vendor.train.src.utils.image_utils import apply_condition_transform_to_images

            image = apply_condition_transform_to_images(
                [image],
                ref_height=image.height,
                ref_width=image.width,
                image_sample_size=self.image_sample_size,
                condition_image_size=self.condition_image_size,
            )[0]
        return {
            "dataset_index": int(index),
            "pair_id": str(row["pair_id"]),
            "task_id": str(row["task_id"]),
            "sample_kind": "degraded" if is_degraded else "clean",
            "decision": decision,
            "reason": reason,
            "answer": format_answer(self.answer_template, decision, reason),
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "image_path": str(image_path),
            "image": image,
        }


def find_unique_subsequence(sequence: list[int], needle: list[int]) -> int:
    positions = [
        index
        for index in range(len(sequence) - len(needle) + 1)
        if sequence[index : index + len(needle)] == needle
    ]
    if len(positions) != 1:
        raise ValueError(
            f"expected exactly one answer-token span, found {len(positions)}; "
            "the fixed answer must occur once in the rendered chat"
        )
    return positions[0]


def synchronize_image_contract(
    accelerator: Accelerator,
    diagnostics: dict[str, Any],
    global_step: int,
) -> None:
    """Make a CPU collator contract failure terminate every rank together."""

    local_failed = diagnostics.get("status") != "ok"
    failure_flag = torch.tensor(
        [int(local_failed)], dtype=torch.int32, device=accelerator.device
    )
    failure_flags = accelerator.gather(failure_flag)
    if not bool(failure_flags.any().item()):
        return
    local_payload = {
        "rank": accelerator.process_index,
        "global_step": int(global_step),
        **diagnostics,
    }
    if dist.is_available() and dist.is_initialized():
        gathered: list[Any] = [None] * accelerator.num_processes
        dist.all_gather_object(gathered, local_payload if local_failed else None)
        failures = [item for item in gathered if item is not None]
    else:
        failures = [local_payload]
    raise RuntimeError(
        "Qwen-VL image-token contract failed before model forward: "
        + json.dumps(failures, ensure_ascii=False, sort_keys=True)
    )


def write_inflight_diagnostic(
    diagnostics_dir: Path,
    process_index: int,
    global_step: int,
    diagnostics: dict[str, Any],
) -> None:
    """Atomically retain the exact batch active on every rank at a crash."""

    payload = {
        "rank": int(process_index),
        "global_step": int(global_step),
        **diagnostics,
    }
    destination = diagnostics_dir / f"rank{process_index:02d}_inflight.json"
    temporary = diagnostics_dir / f".{destination.name}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def cycle(loader: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    while True:
        yield from loader


def next_batch_or_raise_synchronized(
    accelerator: Accelerator,
    iterator: Iterable[dict[str, Any]],
    micro_step: int,
) -> dict[str, Any]:
    """Turn rank-local DataLoader exceptions into one synchronized failure."""

    batch = None
    local_error = None
    try:
        batch = next(iterator)  # type: ignore[arg-type]
    except Exception:
        local_error = traceback.format_exc()

    if dist.is_available() and dist.is_initialized():
        failed = torch.tensor(
            [1 if local_error is not None else 0],
            device=accelerator.device,
            dtype=torch.int32,
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if int(failed.item()) != 0:
            gathered: list[Any] = [None] * accelerator.num_processes
            dist.all_gather_object(
                gathered,
                {"rank": accelerator.process_index, "error": local_error},
            )
            failures = [item for item in gathered if item and item.get("error")]
            raise RuntimeError(
                f"synchronized DataLoader failure at micro_step={micro_step}: "
                + json.dumps(failures, ensure_ascii=False)
            )
    elif local_error is not None:
        raise RuntimeError(
            f"DataLoader failure at micro_step={micro_step}\n{local_error}"
        )
    if batch is None:  # pragma: no cover - guarded by synchronized failure
        raise RuntimeError(f"DataLoader returned no batch at micro_step={micro_step}")
    return batch


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    for old in checkpoints[:-keep]:
        shutil.rmtree(old)


def sequence_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    predictions = shifted_logits.argmax(dim=-1)
    token_correct = ((predictions == shifted_labels) | ~mask).all(dim=1)
    has_labels = mask.any(dim=1)
    return token_correct[has_labels].sum(), has_labels.sum()


def reset_scheduler_to_optimizer_step(scheduler: Any, optimizer_step: int) -> list[float]:
    """Repair an Accelerate scheduler saved with world-size-multiplied steps.

    AcceleratedScheduler normally advances once per process when batches are
    not split. This trainer defines its cosine horizon in optimizer steps, so
    it must advance exactly once per global optimizer update. Older V1
    checkpoints contain the multiplied ``last_epoch``; recompute both the
    scheduler counters and optimizer learning rates from the checkpoint name.
    """

    inner = getattr(scheduler, "scheduler", scheduler)
    if not hasattr(inner, "lr_lambdas") or not hasattr(inner, "base_lrs"):
        raise TypeError(f"unsupported scheduler type for repair: {type(inner)!r}")
    step = int(optimizer_step)
    learning_rates = [
        float(base_lr) * float(lr_lambda(step))
        for base_lr, lr_lambda in zip(inner.base_lrs, inner.lr_lambdas)
    ]
    inner.last_epoch = step
    inner._step_count = step + 1
    inner._last_lr = learning_rates
    for parameter_group, learning_rate in zip(inner.optimizer.param_groups, learning_rates):
        parameter_group["lr"] = learning_rate
    return learning_rates



