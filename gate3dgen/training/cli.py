from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from .artifacts import (
    assert_checkpoint_recipe_compatible,
    checkpoint_is_complete,
    resolve_checkpoint,
)
from .config import PROJECT_DIR, configured_manifests, load_config, runtime_paths
from .condition_contract import (
    CONDITION_KIND,
    CONDITION_SCHEMA_VERSION,
    ConditionContractError,
    assert_condition_contract,
    condition_enabled,
    read_condition_contract,
    resolve_condition_contract,
    write_condition_contract,
)
from .data import (
    INSTRUCTION_EN,
    adapt_albedo_row,
    adapt_row,
    iter_jsonl,
    prepare_manifest,
    validate_rows,
)
from .prompts import prompt_metadata, restoration_instructions
from .experimental.recipes import (
    assert_cache_contract,
    cache_contract,
    is_albedo_recipe,
    recipe_from_config,
    recipe_fingerprint,
    read_cache_contract,
    write_cache_contract,
)


LORA_TARGETS = (
    "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,"
    "img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
)


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _adapted_rows(config: dict[str, Any], decode: bool, split: str = "train") -> list[dict[str, Any]]:
    manifests = configured_manifests(config)
    if split not in manifests:
        raise ValueError(f"split {split!r} is not configured; available={sorted(manifests)}")
    manifest = manifests[split]
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    root = Path(config["data"]["data_root"])
    required = config["data"].get("require_quality_status")
    recipe = recipe_from_config(config)
    instruction_data = restoration_instructions(config)
    if is_albedo_recipe(recipe):
        albedo_root = Path(config["data"]["albedo_root"])
        rows = []
        for row in iter_jsonl(manifest):
            try:
                rows.append(
                    adapt_albedo_row(
                        row,
                        root,
                        albedo_root,
                        required,
                        recipe=recipe,
                        decode=decode,
                        instruction_data=instruction_data,
                    )
                )
            except ValueError:
                # `check` is an inexpensive gate.  `prepare-data` emits the
                # complete per-row ALBEDO_SKIP report for rejected rows.
                continue
    else:
        subject_mask_root_value = config["data"].get("subject_mask_root")
        subject_mask_root = Path(subject_mask_root_value) if subject_mask_root_value else None
        require_subject_mask = bool(config["data"].get("subject_mask_required", False))
        rows = [
            adapt_row(
                row,
                root,
                required,
                instruction_data=instruction_data,
                subject_mask_root=subject_mask_root,
                require_subject_mask=require_subject_mask,
            )
            for row in iter_jsonl(manifest)
        ]
    validate_rows(rows, config, decode=decode, split=split)
    return rows


def _selected_splits(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    manifests = configured_manifests(config)
    selected = getattr(args, "split", None)
    if selected is not None:
        if selected not in manifests:
            raise ValueError(f"split {selected!r} is not configured; available={sorted(manifests)}")
        return [selected]
    return list(manifests)


def _assert_disjoint_splits(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    seen_pairs: dict[str, str] = {}
    seen_assets: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            pair_id = str(row["pair_id"])
            asset_id = str(row["asset_id"])
            if pair_id in seen_pairs and seen_pairs[pair_id] != split:
                raise ValueError(f"pair_id {pair_id} appears in {seen_pairs[pair_id]} and {split}")
            if asset_id in seen_assets and seen_assets[asset_id] != split:
                raise ValueError(f"asset_id {asset_id} appears in {seen_assets[asset_id]} and {split}")
            seen_pairs[pair_id] = split
            seen_assets[asset_id] = split


def _check_model(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(path)
    required = ["model_index.json", "transformer", "vae", "scheduler", "processor", "text_encoder"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise ValueError(f"FireRed model is missing: {missing}")
    if not list(path.rglob("*.safetensors")):
        raise ValueError(f"FireRed model has no safetensors weights: {path}")


def command_check(args: argparse.Namespace, config: dict[str, Any]) -> None:
    rows_by_split = {
        split: _adapted_rows(config, decode=not args.no_decode, split=split)
        for split in _selected_splits(args, config)
    }
    if len(rows_by_split) > 1:
        _assert_disjoint_splits(rows_by_split)
    if not args.skip_model:
        _check_model(Path(config["model"]["pretrained_model_path"]))
    summaries = {
        split: validate_rows(rows, config, decode=False, split=split)
        for split, rows in rows_by_split.items()
    }
    _json({"status": "ok", "scope": config["scope"], "model_checked": not args.skip_model, "splits": summaries})


def command_prepare(args: argparse.Namespace, config: dict[str, Any]) -> None:
    paths = runtime_paths(config)
    manifests = configured_manifests(config)
    summaries = {}
    for split in _selected_splits(args, config):
        summaries[split] = prepare_manifest(
            config,
            paths["prepared_by_split"][split],
            decode=not args.no_decode,
            source_manifest=manifests[split],
            split=split,
        )
    _json({"status": "ok", "splits": summaries})


def command_condition_preflight(
    args: argparse.Namespace, config: dict[str, Any]
) -> None:
    """Validate a gate condition before any distributed workers start.

    This command is intentionally separate from ``check``: it reads only the
    Qwen checkpoint/export metadata and writes one cache-local identity marker;
    it does not decode the training manifest or launch embedding extraction.
    """

    paths = runtime_paths(config)
    payload = _condition_contract_for_run(
        config, paths, require_artifact=True, write_marker=True
    )
    if payload is None:
        raise ConditionContractError(
            "condition-preflight requires config.condition_model.enabled=true"
        )
    _json(
        {
            "status": "ok",
            "schema": payload["schema"],
            "kind": payload["kind"],
            "fingerprint": payload["fingerprint"],
            "variant": payload["condition_variant"],
            "checkpoint_step": payload["condition_checkpoint_step"],
            "checkpoint_world_size": payload["condition_checkpoint_world_size"],
            "marker": str(_condition_marker_path(paths)),
        }
    )


def _run(command: list[str], env: Optional[dict[str, str]] = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    roots = [str(PROJECT_DIR.parents[1])]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env.setdefault("REDEDIT_LOG_LEVEL", "INFO")
    return env


def _embedding_rows(task_meta: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(task_meta.glob("*.jsonl")):
        rows.extend(iter_jsonl(path))
    return rows


def _condition_marker_path(paths: dict[str, Path]) -> Path:
    """Return the immutable condition identity marker for one cache root."""

    return paths["meta_root"] / "condition_contract.json"


def _condition_contract_for_run(
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    require_artifact: bool,
    write_marker: bool = False,
) -> dict[str, Any] | None:
    """Resolve a gate condition and compare it with a cache marker.

    ``require_artifact=False`` is used only by CPU dry-runs.  Real extraction,
    training and the distributed launcher preflight always require the exact
    checkpoint/export and model weights to be present.
    """

    if not condition_enabled(config):
        return None
    marker_path = _condition_marker_path(paths)
    existing = None
    if marker_path.is_file():
        existing = read_condition_contract(marker_path)
    payload = resolve_condition_contract(
        config,
        require_artifact=require_artifact,
        existing_contract=existing,
    )
    if payload is None:
        raise ConditionContractError("condition_model is enabled but no identity was resolved")
    if existing is not None and require_artifact:
        assert_condition_contract(existing, payload, context=str(marker_path))
    if write_marker:
        write_condition_contract(marker_path, payload)
    return payload


def _embedding_expected_contract(
    recipe_contract: dict[str, Any] | None,
    condition_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Combine legacy recipe and Qwen condition metadata checks."""

    if recipe_contract is None and condition_payload is None:
        return None
    result = dict(recipe_contract or {})
    if condition_payload is not None:
        result.update(
            {
                "condition_schema": condition_payload["schema"],
                "condition_kind": condition_payload["kind"],
                "condition_fingerprint": condition_payload["fingerprint"],
                "cache_fingerprint": condition_payload["cache_fingerprint"],
                "condition_variant": condition_payload["condition_variant"],
                "condition_role": condition_payload["condition_role"],
                "condition_model_hash": condition_payload["condition_model_hash"],
                "condition_checkpoint_step": condition_payload["condition_checkpoint_step"],
                "condition_checkpoint_world_size": condition_payload[
                    "condition_checkpoint_world_size"
                ],
                "condition_model_path": condition_payload["condition_model_path"],
                "condition_prompt": condition_payload["prompt"],
                "condition_extraction": condition_payload["extraction"],
            }
        )
    return result


# These fields determine the Qwen-VL condition that was encoded.  They must be
# compared for every recipe, including legacy/non-albedo recipes that do not
# carry the albedo recipe contract marker.  ``prompt_library`` itself is
# deliberately absent: it is an absolute path in prepared rows and therefore
# is not portable across machines.  The selected ID and file fingerprint below
# identify its content without tying a cache to a mount prefix.
_EMBEDDING_CONDITION_FIELDS = (
    "instruction",
    "instruction_cn",
    "prompt_library_id",
    "prompt_library_fingerprint",
    "source_image",
)


def _portable_source_image_identity(value: Any) -> Any:
    """Normalize condition-image paths without binding a cache to one mount."""

    if not isinstance(value, list):
        return value
    normalized = []
    for item in value:
        text = str(item).replace("\\", "/")
        if "/scripts/" in text:
            text = "scripts/" + text.split("/scripts/", 1)[1]
        normalized.append(text)
    return normalized


def validate_embeddings(
    prepared: Path,
    task_meta: Path,
    expected_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = list(iter_jsonl(prepared))
    rows = _embedding_rows(task_meta)
    if len(rows) != len(expected):
        raise ValueError(f"embedding row count {len(rows)} != prepared row count {len(expected)}")
    expected_by_pair: dict[str, dict[str, Any]] = {}
    for row in expected:
        pair_id = str(row["pair_id"])
        if pair_id in expected_by_pair:
            raise ValueError("prepared manifest contains duplicate pair_id values")
        expected_by_pair[pair_id] = row
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("embedding metadata contains duplicate pair_id values")
    if set(pair_ids) != set(expected_by_pair):
        raise ValueError("embedding metadata pair IDs differ from prepared manifest")
    required = ("embeddings_tensor_en", "embeddings_tensor_cn", "embeddings_tensor_droptext")
    for row in rows:
        if any(key not in row or not Path(row[key]).is_file() for key in required):
            raise ValueError(f"pair {row.get('pair_id')} has incomplete embeddings")
        if any(key.endswith("_inv") for key in row):
            raise ValueError(f"pair {row.get('pair_id')} unexpectedly contains inverse embeddings")
        pair_id = str(row["pair_id"])
        prepared_row = expected_by_pair[pair_id]
        for key in _EMBEDDING_CONDITION_FIELDS:
            embedded_value = row.get(key)
            prepared_value = prepared_row.get(key)
            if key == "source_image":
                embedded_value = _portable_source_image_identity(embedded_value)
                prepared_value = _portable_source_image_identity(prepared_value)
            if embedded_value != prepared_value:
                raise ValueError(
                    f"pair {pair_id} embedding metadata {key} differs from prepared manifest"
                )
        if expected_contract is not None:
            for key in (
                "recipe",
                "recipe_schema",
                "recipe_fingerprint",
                "cache_schema",
                "cache_fingerprint",
                "schema_fingerprint",
                "source_manifest_sha256",
            ):
                if key in expected_contract and row.get(key) != expected_contract[key]:
                    raise ValueError(
                        f"pair {row.get('pair_id')} embedding {key} differs from prepared cache"
                    )
            # These fields are emitted by extract_vlm_embeds for gate A/B and
            # tie every tensor to the exact frozen Qwen state/prompt.  They
            # deliberately compare flat values; the full nested payload lives
            # in meta/<scope>/condition_contract.json.
            condition_fields = (
                "condition_schema",
                "condition_kind",
                "condition_fingerprint",
                "cache_fingerprint",
                "condition_variant",
                "condition_role",
                "condition_model_hash",
                "condition_checkpoint_step",
                "condition_checkpoint_world_size",
                "condition_model_path",
                "condition_prompt",
                "condition_extraction",
            )
            for key in condition_fields:
                if key in expected_contract and row.get(key) != expected_contract[key]:
                    raise ValueError(
                        f"pair {row.get('pair_id')} embedding {key} differs from condition contract"
                    )
    summary = {
        "rows": len(rows),
        "jsonl_files": len(list(task_meta.glob("*.jsonl"))),
    }
    return summary


def _albedo_cache_contract(config: dict[str, Any], prepared: Path) -> dict[str, Any] | None:
    """Validate the recipe marker before embeddings or training are reused."""

    recipe = recipe_from_config(config)
    if not is_albedo_recipe(recipe):
        return None
    rows = list(iter_jsonl(prepared))
    expected = cache_contract(config, rows, prepared_manifest=str(prepared))
    marker_path = prepared.parent / "recipe_contract.json"
    if not marker_path.is_file():
        raise ValueError(f"albedo cache contract is missing: {marker_path}")
    actual = read_cache_contract(marker_path)
    assert_cache_contract(actual, expected, context=str(prepared))
    return expected


def command_extract(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if getattr(args, "split", None) not in (None, "train"):
        raise ValueError("extract-embeddings only supports the train split")
    paths = runtime_paths(config)
    if not paths["prepared"].is_file():
        prepare_manifest(
            config,
            paths["prepared"],
            decode=True,
            source_manifest=configured_manifests(config)["train"],
            split="train",
        )
    prepared = paths["prepared"]
    prepared_contract = _albedo_cache_contract(config, prepared)
    condition_payload = _condition_contract_for_run(
        config, paths, require_artifact=True, write_marker=False
    )
    embedding_contract = _embedding_expected_contract(
        prepared_contract, condition_payload
    )
    task_meta = paths["task_meta"]
    embeddings = paths["embeddings"]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        smoke_root = paths["cache_root"] / "smoke" / config["scope"] / f"limit-{args.limit}"
        prepared = smoke_root / "source.jsonl"
        rows = list(iter_jsonl(paths["prepared"]))[:args.limit]
        prepared.parent.mkdir(parents=True, exist_ok=True)
        temporary = prepared.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, prepared)
        task_meta = smoke_root / "meta" / "texverse_restore"
        embeddings = smoke_root / "embeddings"
    if task_meta.is_dir():
        try:
            summary = validate_embeddings(
                prepared,
                task_meta,
                embedding_contract,
            )
            if prepared_contract is not None:
                marker_path = paths["meta_root"] / "recipe_contract.json"
                if not marker_path.is_file():
                    write_cache_contract(marker_path, prepared_contract)
            if condition_payload is not None:
                marker_path = _condition_marker_path(paths)
                if not marker_path.is_file():
                    raise ValueError(
                        f"condition embedding marker is missing: {marker_path}"
                    )
                assert_condition_contract(
                    read_condition_contract(marker_path),
                    condition_payload,
                    context=str(marker_path),
                )
            _json({"status": "already_complete", **summary})
            return
        except (OSError, ValueError):
            pass
    staging_root = paths["cache_root"] / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"embed-{config['scope']}-", dir=staging_root))
    # Embedding extraction is intentionally a single-node preprocessing job.
    # A multi-node training world size (for example 32) must not be interpreted
    # as 32 local torchrun workers on every node.
    embedding = config.get("embedding", {})
    local_default = embedding.get(
        "num_processes",
        config["training"].get(
            "num_processes_per_machine", config["training"]["num_processes"]
        ),
    )
    processes = int(args.num_processes or local_default)
    command = [
        sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={processes}",
        "-m", "gate3dgen.training.vendor.train.src.extract_vlm_embeds", str(prepared),
        "--output_jsonl_dir", str(stage),
        "--embeddings_save_dir", str(embeddings),
        "--model_path", config["model"]["pretrained_model_path"],
        "--batch_size", str(4 if args.batch_size is None else args.batch_size),
        "--image_sample_size", str(config["model"]["resolution"]),
        "--num_workers", str(
            embedding.get("num_workers", 4)
            if args.num_workers is None else args.num_workers
        ),
        "--disable_inverse",
    ]
    if condition_payload is not None:
        condition = config["condition_model"]
        # Gate A/B condition weights and processor are intentionally separate
        # from FireRed's mmDiT/VAE model.  The extractor loads Qwen hidden
        # states from this path while the training process remains offline.
        command[command.index("--model_path") + 1] = str(condition["model_path"])
        for flag, key in (
            ("--processor_path", "processor_path"),
            ("--config_path", "config_path"),
        ):
            if condition.get(key):
                command.extend([flag, str(condition[key])])
        command.extend(
            [
                "--condition_schema", str(condition_payload["schema"]),
                "--condition_kind", str(condition_payload["kind"]),
                "--condition_fingerprint", str(condition_payload["fingerprint"]),
                "--condition_variant", str(condition_payload["condition_variant"]),
                "--condition_role", str(condition_payload["condition_role"]),
                "--condition_model_hash", str(condition_payload["condition_model_hash"]),
                "--condition_checkpoint_step", str(condition_payload["condition_checkpoint_step"]),
                "--condition_checkpoint_world_size", str(
                    condition_payload["condition_checkpoint_world_size"]
                ),
                "--condition_model_identity", str(
                    condition_payload["condition_model_path"]
                ),
                "--condition_prompt_json", json.dumps(
                    condition_payload["prompt"], ensure_ascii=False, sort_keys=True
                ),
                "--condition_extraction_json", json.dumps(
                    condition_payload["extraction"], ensure_ascii=False, sort_keys=True
                ),
            ]
        )
    _run(command, _python_env())
    summary = validate_embeddings(
        prepared,
        stage,
        embedding_contract,
    )
    # Some OBS/FUSE mounts reject directory-level rename(2) with EPERM even
    # though atomic file replacement works. Commit each completed rank shard
    # separately so interrupted runs remain detectable by validate_embeddings.
    task_meta.mkdir(parents=True, exist_ok=True)
    for stale in task_meta.glob("*.jsonl"):
        stale.unlink()
    for shard in sorted(stage.glob("*.jsonl")):
        os.replace(shard, task_meta / shard.name)
    stage.rmdir()
    validate_embeddings(
        prepared,
        task_meta,
        embedding_contract,
    )
    if prepared_contract is not None:
        contract = dict(prepared_contract)
        contract["prepared_manifest"] = str(prepared)
        write_cache_contract(paths["meta_root"] / "recipe_contract.json", contract)
    if condition_payload is not None:
        marker = dict(condition_payload)
        marker.update(
            {
                "prepared_manifest": str(prepared),
                "task_meta": str(task_meta),
                "embedding_root": str(embeddings),
                "embedding_rows": int(summary["rows"]),
                "status": "complete",
            }
        )
        write_condition_contract(_condition_marker_path(paths), marker)
    _json({"status": "ok", "task_meta": str(task_meta), **summary})


def _train_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    resume: Optional[Path],
    condition_payload: dict[str, Any] | None = None,
) -> list[str]:
    paths = runtime_paths(config)
    training = config["training"]
    model = config["model"]
    processes = int(getattr(args, "num_processes", None) or training["num_processes"])
    max_steps = int(getattr(args, "max_train_steps", None) or training["max_train_steps"])
    recipe = recipe_from_config(config)
    # The production shared 32-GPU launcher supplies 100000 steps explicitly.
    # Make direct CLI training and --dry-run report that same effective value;
    # an explicit user --max-train-steps remains the highest-priority override.
    if is_albedo_recipe(recipe) and getattr(args, "max_train_steps", None) is None:
        max_steps = 100000
    mode = getattr(args, "training_mode", None) or training.get("mode", "lora")
    architecture = model.get("architecture", "stock")
    attention_backend = (
        getattr(args, "attention_backend", None)
        or model.get("attention_backend", "native")
    )
    launch = [
        sys.executable, "-m", "accelerate.commands.launch", "--mixed_precision=bf16",
        "--num_processes", str(processes),
    ]
    num_machines = int(
        getattr(args, "num_machines", None)
        or training.get("num_machines", 1)
    )
    if num_machines > 1:
        processes_per_machine = int(
            training.get("num_processes_per_machine", processes // num_machines)
        )
        if processes != num_machines * processes_per_machine:
            raise ValueError(
                "multi-node process topology mismatch: "
                f"num_processes={processes}, num_machines={num_machines}, "
                f"num_processes_per_machine={processes_per_machine}"
            )
        machine_rank_value = getattr(args, "machine_rank", None)
        if machine_rank_value is None:
            machine_rank_text = (
                os.environ.get("NODE_RANK")
                or os.environ.get("VC_TASK_INDEX")
                or os.environ.get("MA_TASK_INDEX")
                or os.environ.get("MACHINE_RANK")
                or os.environ.get("SLURM_NODEID")
                or os.environ.get("RANK")
            )
            if machine_rank_text is None:
                if getattr(args, "dry_run", False):
                    machine_rank_text = "0"
                else:
                    raise ValueError(
                        "multi-node training requires --machine-rank or a distributed "
                        "rank variable (NODE_RANK/VC_TASK_INDEX/MA_TASK_INDEX/RANK)"
                    )
            machine_rank_value = int(machine_rank_text)
        machine_rank = int(machine_rank_value)
        if not 0 <= machine_rank < num_machines:
            raise ValueError(
                f"machine_rank must be in [0, {num_machines - 1}], got {machine_rank}"
            )
        main_process_ip = (
            getattr(args, "main_process_ip", None)
            or os.environ.get("MASTER_ADDR")
        )
        if not main_process_ip:
            if getattr(args, "dry_run", False):
                main_process_ip = "127.0.0.1"
            else:
                raise ValueError(
                    "multi-node training requires --main-process-ip or MASTER_ADDR"
                )
        main_process_port = int(
            getattr(args, "main_process_port", None)
            or os.environ.get("MASTER_PORT")
            or training.get("main_process_port", 29500)
        )
        launch.extend([
            "--num_machines", str(num_machines),
            "--machine_rank", str(machine_rank),
            "--main_process_ip", str(main_process_ip),
            "--main_process_port", str(main_process_port),
            # Match the known-good distributed torchrun launcher: static
            # rendezvous consumes main_process_ip/main_process_port directly.
            "--rdzv_backend", "static",
            "--same_network",
        ])
    configured_mode = training.get("mode", "lora")
    use_fsdp = (
        mode == "partial_unfreeze"
        if mode != configured_mode
        else bool(training.get("use_fsdp", mode == "partial_unfreeze"))
    )
    if use_fsdp:
        fsdp_state_dict_type = str(training.get("fsdp_state_dict_type", "FULL_STATE_DICT")).upper()
        if fsdp_state_dict_type not in {"FULL_STATE_DICT", "SHARDED_STATE_DICT"}:
            raise ValueError("training.fsdp_state_dict_type must be FULL_STATE_DICT or SHARDED_STATE_DICT")
        fsdp_backward_prefetch = str(
            training.get("fsdp_backward_prefetch", "BACKWARD_PRE")
        ).upper()
        if fsdp_backward_prefetch not in {
            "BACKWARD_PRE",
            "BACKWARD_POST",
            "NO_PREFETCH",
        }:
            raise ValueError(
                "training.fsdp_backward_prefetch must be BACKWARD_PRE, "
                "BACKWARD_POST, or NO_PREFETCH"
            )
        launch.extend([
            "--use_fsdp",
            "--fsdp_version=1",
            "--fsdp_sharding_strategy=FULL_SHARD",
            "--fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP",
            "--fsdp_transformer_layer_cls_to_wrap=QwenImageTransformerBlock",
            f"--fsdp_state_dict_type={fsdp_state_dict_type}",
            "--fsdp_use_orig_params=true",
            "--fsdp_cpu_ram_efficient_loading=false",
            "--fsdp_sync_module_states=false",
            "--fsdp_offload_params=" + str(bool(training.get("fsdp_offload_params", False))).lower(),
            "--fsdp_activation_checkpointing=false",
            f"--fsdp_backward_prefetch={fsdp_backward_prefetch}",
        ])
    command = launch + [
        "-m", "gate3dgen.training.vendor.train.src.sft",
        "--pretrained_model_name_or_path", model["pretrained_model_path"],
        "--transformer_architecture", architecture,
        "--attention_backend", attention_backend,
        "--train_data_meta_dir", str(paths["meta_root"]),
        "--train_data_weights", "texverse_restore=1",
        "--train_src_img_num_weights", "0=0,1=1,2=0,3=0",
        "--train_batch_size", str(training["batch_size"]),
        "--image_sample_size", str(model["resolution"]),
        "--gradient_accumulation_steps", str(training["gradient_accumulation_steps"]),
        "--max_train_steps", str(max_steps),
        "--learning_rate", str(training["learning_rate"]),
        "--lr_scheduler", training.get("lr_scheduler", "cosine"),
        "--lr_warmup_steps", str(training["warmup_steps"]),
        "--checkpointing_steps", str(training["checkpointing_steps"]),
        "--checkpoints_total_limit", str(training["checkpoints_total_limit"]),
        "--output_dir", str(paths["output_dir"]),
        "--gradient_checkpointing", "--mixed_precision", "bf16", "--allow_tf32",
        "--adam_weight_decay", "0.03", "--adam_epsilon", "1e-10", "--max_grad_norm", "0.05",
        "--condition_encoder_mode", "offline", "--uniform_sampling", "--vae_mini_batch", "1",
        "--vae_encode_mode", str(training.get("vae_encode_mode", "sample")),
        "--dataloader_num_workers", str(
            training.get("num_workers", 4)
            if getattr(args, "num_workers", None) is None else args.num_workers
        ), "--report_to", "tensorboard",
        "--seed", str(training["seed"]),
    ]
    if condition_enabled(config):
        payload = condition_payload
        if payload is None:
            # A dry-run does not touch model files.  Resolve the structural
            # identity so the generated command still advertises that it is
            # tied to a gate condition; real runs pass the verified payload
            # from command_train below.
            payload = resolve_condition_contract(config, require_artifact=False)
        if payload is None:
            raise ConditionContractError("condition_model is enabled but no condition payload was resolved")
        command.extend(
            [
                "--condition_schema", str(payload["schema"]),
                "--condition_kind", str(payload["kind"]),
                "--condition_fingerprint", str(payload["fingerprint"]),
                "--condition_cache_fingerprint", str(payload["cache_fingerprint"]),
                "--condition_variant", str(payload["condition_variant"]),
                "--condition_role", str(payload["condition_role"]),
                "--condition_model_hash", str(payload.get("condition_model_hash") or "unverified"),
                "--condition_checkpoint_step", str(payload.get("condition_checkpoint_step") or "unverified"),
                "--condition_checkpoint_world_size", str(
                    payload.get("condition_checkpoint_world_size") or "unverified"
                ),
                "--condition_model_identity", str(payload["condition_model_path"]),
            ]
        )
    if not bool(training.get("save_final_checkpoint", True)):
        command.append("--skip_final_checkpoint")
    if use_fsdp:
        command.append("--use_fsdp")
    if mode == "lora":
        command.extend([
            "--use_peft_lora",
            "--lora_r", str(model["lora_rank"]),
            "--lora_alpha", str(model["lora_alpha"]),
            "--lora_dropout", str(model["lora_dropout"]),
            "--lora_target_modules", LORA_TARGETS,
        ])
    elif mode == "partial_unfreeze":
        block_range = (
            getattr(args, "trainable_block_range", None)
            or training.get("trainable_block_range", [29, 49])
        )
        first, last = (int(value) for value in block_range)
        if not 0 <= first <= last < 60:
            raise ValueError(f"invalid trainable_block_range: {(first, last)}")
        command.append("--trainable_modules")
        command.extend(f"transformer_blocks.{index}." for index in range(first, last + 1))
        command.extend([
            "--trainable_modules_low_lr_ratio",
            str(training.get("low_lr_ratio", 0.5)),
        ])
    else:
        raise ValueError(f"unsupported training mode: {mode}")

    loss = config.get("loss", {"objective": "flow"})
    objective = getattr(args, "loss_objective", None) or loss.get("objective", "flow")
    if objective == "hybrid_x0_pixel":
        sigma_band = loss.get("x0_sigma_band", [0.5, 1.0])
        command.extend([
            "--x0_pixel_loss",
            "--x0_sigma_band", f"{float(sigma_band[0])},{float(sigma_band[1])}",
            "--flow_loss_weight", str(loss.get("flow_weight", 1.0)),
            "--latent_x0_weight", str(loss.get("latent_x0_weight", 0.1)),
            "--pixel_l1_weight", str(loss.get("pixel_l1_weight", 0.5)),
            "--pixel_lpips_weight", str(loss.get("pixel_lpips_weight", 0.0)),
            "--pixel_background_weight", str(loss.get("pixel_background_weight", 0.0)),
        ])
        if loss.get("restoration_v3", False):
            command.append("--restoration_v3_loss")
            for key, flag in (
                ("pixel_full_l1_weight", "--pixel_full_l1_weight"),
                ("pixel_subject_l1_weight", "--pixel_subject_l1_weight"),
                ("pixel_affected_l1_weight", "--pixel_affected_l1_weight"),
                (
                    "pixel_subject_gradient_weight",
                    "--pixel_subject_gradient_weight",
                ),
            ):
                command.extend([flag, str(loss.get(key, 0.0))])
        command.extend([
            "--pixel_low_frequency_weight",
            str(loss.get("pixel_low_frequency_weight", 0.0)),
            "--pixel_low_frequency_scales",
            ",".join(str(int(value)) for value in loss.get(
                "pixel_low_frequency_scales", [8, 16, 32]
            )),
            "--restoration_task_multipliers",
            ",".join(
                f"{task}={float(multiplier)}"
                for task, multiplier in sorted(loss.get("task_multipliers", {}).items())
            ),
        ])
        if loss.get("pixel_loss_fg_only", False):
            command.append("--pixel_loss_fg_only")
    elif objective != "flow":
        raise ValueError(f"unsupported loss objective: {objective}")
    if is_albedo_recipe(recipe):
        # Prepared rows carry the exact post-filter fingerprint.  Do not
        # recompute it from the stale/raw manifest when constructing a train
        # command: this is the value checkpoints and cache markers validate.
        prepared_rows = list(iter_jsonl(paths["prepared"])) if paths["prepared"].is_file() else []
        # The dry-run path intentionally does not require a prepared cache,
        # but still emits a complete deterministic recipe/schema contract.  A
        # real run has rows here and therefore gets the post-filter identity.
        prepared_contract = cache_contract(
            config, prepared_rows, prepared_manifest=str(paths["prepared"])
        )
        fingerprint = (
            str(prepared_contract["recipe_fingerprint"])
            if prepared_contract.get("recipe_fingerprint")
            else recipe_fingerprint(recipe, config, prepared_rows)
        )
        command.extend([
            "--recipe", recipe,
            "--recipe_schema", "texverse-albedo-recipe-v1",
            "--recipe_fingerprint", fingerprint,
            "--albedo_root", str(config["data"]["albedo_root"]),
            "--cache_schema", "texverse-albedo-cache-v1",
            "--cache_fingerprint", fingerprint,
            "--schema_fingerprint",
            str(prepared_contract["schema_fingerprint"]),
        ])
        for key, flag in (
            ("albedo_color_weight", "--albedo_color_weight"),
            ("albedo_gradient_weight", "--albedo_gradient_weight"),
            ("albedo_perceptual_weight", "--albedo_perceptual_weight"),
            ("albedo_consistency_weight", "--albedo_consistency_weight"),
            ("albedo_aux_weight", "--albedo_aux_weight"),
        ):
            command.extend([flag, str(loss.get(key, 0.0))])
        if bool(loss.get("albedo_foreground_only", False)):
            command.append("--albedo_foreground_only")
        if recipe == "albedo_consistency":
            command.append("--albedo_dual_view")
        if recipe == "rgb_albedo_multitask" and bool(model.get("albedo_aux_head", False)):
            command.extend([
                "--albedo_aux_head",
                "--albedo_aux_hidden_dim", str(model.get("albedo_aux_hidden_dim", 256)),
                "--albedo_aux_output_dim", str(model.get("albedo_aux_output_dim", 3)),
            ])
    if resume is not None:
        command.extend(["--resume_from_checkpoint", resume.name])
    init_checkpoint = training.get("init_from_checkpoint")
    if init_checkpoint:
        command.extend(["--init_from_checkpoint", str(init_checkpoint)])
    return command


def command_train(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if getattr(args, "split", None) not in (None, "train"):
        raise ValueError("train only supports the train split")
    if getattr(args, "dry_run", False):
        condition_payload = None
        if condition_enabled(config):
            condition_payload = resolve_condition_contract(config, require_artifact=False)
        command = _train_command(args, config, None, condition_payload)
        _json({
            "status": "dry_run",
            "recipe": recipe_from_config(config),
            "world_size": int(
                getattr(args, "num_processes", None)
                or config["training"]["num_processes"]
            ),
            "command": command,
        })
        return
    paths = runtime_paths(config)
    expected_contract = _albedo_cache_contract(config, paths["prepared"])
    condition_payload = _condition_contract_for_run(
        config, paths, require_artifact=True, write_marker=False
    )
    embedding_contract = _embedding_expected_contract(
        expected_contract, condition_payload
    )
    validate_embeddings(
        paths["prepared"],
        paths["task_meta"],
        embedding_contract,
    )
    if expected_contract is not None:
        metadata_marker = paths["meta_root"] / "recipe_contract.json"
        if not metadata_marker.is_file():
            raise ValueError(f"albedo embedding cache contract is missing: {metadata_marker}")
        assert_cache_contract(
            read_cache_contract(metadata_marker),
            expected_contract,
            context=str(paths["task_meta"]),
        )
    if condition_payload is not None:
        marker_path = _condition_marker_path(paths)
        if not marker_path.is_file():
            raise ValueError(f"condition embedding marker is missing: {marker_path}")
        assert_condition_contract(
            read_condition_contract(marker_path),
            condition_payload,
            context=str(marker_path),
        )
    _check_model(Path(config["model"]["pretrained_model_path"]))
    mode = getattr(args, "training_mode", None) or config["training"].get("mode", "lora")
    expected_kind = "adapter" if mode == "lora" else "transformer"
    init_checkpoint_value = config["training"].get("init_from_checkpoint")
    if init_checkpoint_value and getattr(args, "resume", None):
        raise ValueError("init_from_checkpoint and --resume are mutually exclusive")
    init_checkpoint = None
    if init_checkpoint_value:
        init_checkpoint = resolve_checkpoint(
            paths["output_dir"], str(init_checkpoint_value), expected_kind=expected_kind
        )
        if not checkpoint_is_complete(init_checkpoint):
            raise ValueError(f"init checkpoint is incomplete: {init_checkpoint}")
        if init_checkpoint.parent.resolve() == paths["output_dir"].resolve():
            raise ValueError(
                "init_from_checkpoint must come from a different output namespace; "
                "use --resume for an in-place continuation"
            )
    resume = (
        resolve_checkpoint(paths["output_dir"], args.resume, expected_kind=expected_kind)
        if getattr(args, "resume", None) else None
    )
    if resume is not None and is_albedo_recipe(recipe_from_config(config)):
        assert_checkpoint_recipe_compatible(
            resume,
            recipe=recipe_from_config(config),
            recipe_fingerprint=(expected_contract or {}).get("recipe_fingerprint"),
            cache_fingerprint=(expected_contract or {}).get("cache_fingerprint"),
            schema_fingerprint=(expected_contract or {}).get("schema_fingerprint"),
        )
    condition_checkpoint = resume or init_checkpoint
    if condition_checkpoint is not None and condition_payload is not None:
        marker_path = condition_checkpoint / "checkpoint_complete.json"
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"condition-gated checkpoint has no valid completion marker: {marker_path}"
            ) from error
        expected_marker = {
            "condition_schema": condition_payload["schema"],
            "condition_kind": condition_payload["kind"],
            "condition_fingerprint": condition_payload["fingerprint"],
            "condition_cache_fingerprint": condition_payload["cache_fingerprint"],
            "condition_variant": condition_payload["condition_variant"],
            "condition_role": condition_payload["condition_role"],
            "condition_model_hash": condition_payload["condition_model_hash"],
            "condition_checkpoint_step": condition_payload["condition_checkpoint_step"],
            "condition_checkpoint_world_size": condition_payload[
                "condition_checkpoint_world_size"
            ],
            "condition_model_identity": condition_payload["condition_model_path"],
        }
        mismatches = {
            key: (marker.get(key), value)
            for key, value in expected_marker.items()
            if marker.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"condition-gated checkpoint identity mismatch: {mismatches}"
            )
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    _run(
        _train_command(args, config, resume, condition_payload),
        _python_env(),
    )



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare, cache and train the released router-conditioned purifier.")
    parser.add_argument("command", choices=("check", "prepare-data", "condition-preflight", "extract-embeddings", "train"))
    parser.add_argument("scope", nargs="?", default="full", choices=("full",))
    parser.add_argument("--config", default="configs/train/purifier.yaml")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--no-decode", action="store_true")
    for name in ("num-processes", "num-machines", "machine-rank", "main-process-port", "num-workers", "batch-size", "max-train-steps", "limit"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--main-process-ip")
    parser.add_argument("--attention-backend", choices=("native", "fa3", "auto"))
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.scope, args.config)
    paper_a = config.get("paper_ablation") == "purifier-A"
    if (not condition_enabled(config) and not paper_a) or config.get("classification", {}).get("enabled"):
        raise ValueError("This entry point trains the released frozen-router-conditioned purifier only.")
    commands = {
        "check": command_check, "prepare-data": command_prepare,
        "condition-preflight": command_condition_preflight,
        "extract-embeddings": command_extract, "train": command_train,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
