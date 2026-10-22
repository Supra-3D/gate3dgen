# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
import argparse
import gc
import logging
import math
import os
import pickle
import random
import json
import shutil
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import datasets
from datasets import IterableDataset as HFIterableDataset
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.parallelism_config import ParallelismConfig
from accelerate.logging import get_logger as accelerate_get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import (
    ProjectConfiguration,
    set_seed,
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    load_fsdp_model,
)
from diffusers.optimization import get_scheduler
from einops import rearrange
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from glob import glob

from .utils.other import linear_decay
from .utils.image_utils import save_image
from .utils.log_utils import get_logger, DistributedColoredFormatter, get_dist_prefix, get_default_log_level, log_once
from gate3dgen.training.experimental.repair_classifier import (
    get_repair_classifier,
    load_repair_classifier_payload,
)


def _find_fsdp_wrapper(module):
    """Return the outermost FSDP wrapper containing ``module``, if any.

    ``get_repair_classifier`` intentionally walks through Accelerate/DDP/FSDP
    containers to find the registered head.  Saving ``head.state_dict()``
    directly after a FULL_SHARD forward is unsafe, however: the root has
    already resharded the child and the resulting tensors can have zero
    length.  We need the wrapper itself in order to enter
    ``summon_full_params`` before reading the sidecar state.
    """

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except (ImportError, AttributeError):  # pragma: no cover - old torch builds
        return None

    queue = [module]
    seen = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if isinstance(candidate, FSDP):
            return candidate
        queue.extend(getattr(candidate, "children", lambda: ())())
    return None


def _repair_classifier_payload(model, *, required=False):
    """Collect a complete, CPU-owned classifier payload for checkpoint save.

    FSDP's ``FULL_SHARD`` model save is already collective.  This helper is
    therefore called by *every* rank in the FSDP save hook, while only rank 0
    writes the returned payload.  ``rank0_only=False`` is deliberate: making
    only rank 0 enter ``summon_full_params`` would deadlock the other ranks.
    """

    classifier = get_repair_classifier(model)
    if classifier is None:
        if required:
            raise RuntimeError("repair classifier was enabled but no registered head was found")
        return None

    fsdp = _find_fsdp_wrapper(model)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if fsdp is not None and distributed:
        context = fsdp.summon_full_params(
            fsdp, recurse=True, writeback=False, rank0_only=False
        )
    else:
        context = nullcontext()
    with context:
        # Clone while the full parameter views are alive.  Returning the
        # views themselves would leave the sidecar tied to FSDP's flat
        # storage and could be resharded before ``torch.save`` runs.
        state_dict = {
            str(key): value.detach().cpu().clone()
            for key, value in classifier.state_dict().items()
        }
    return {
        "state_dict": state_dict,
        "input_dim": int(classifier.input_dim),
        "hidden_dim": int(classifier.hidden_dim),
        "dropout": float(classifier.dropout),
    }


def _atomic_json_dump(payload, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _marker_scalar(value):
    """Normalize JSON scalar values used by resume compatibility markers.

    Older launchers sometimes serialized step/world-size values as strings.
    Condition provenance is part of the checkpoint identity, so accepting an
    equivalent numeric representation is safe while still rejecting absent or
    otherwise different values.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return value


def _save_repair_classifier_state(model, output_dir, *, required=False):
    """Persist the tiny gate in a small sidecar alongside every checkpoint.

    Full transformer/FSDP checkpoints already include the registered child in
    their model state.  PEFT adapter export intentionally omits frozen/base
    tensors, so writing this small sidecar keeps the classifier resumable and
    available to inference without serializing the 20B backbone again.
    """

    payload = _repair_classifier_payload(model, required=required)
    if payload is None:
        return None
    path = os.path.join(output_dir, "repair_classifier.pt")
    torch.save(payload, path)
    return path


def _load_repair_classifier_state(model, input_dir, *, required=False):
    """Restore the tiny gate sidecar when a checkpoint is resumed.

    PEFT adapter exports do not include parameters belonging to the base model,
    and a plain diffusers ``from_pretrained`` reconstruction may likewise not
    know about this dynamically registered child.  Keep the sidecar load
    explicit so both paths restore the head without weakening backbone
    checkpoint validation.
    """

    classifier = get_repair_classifier(model)
    path = os.path.join(input_dir, "repair_classifier.pt")
    if classifier is None:
        if required:
            raise RuntimeError("repair classifier was enabled but no registered head was found")
        return None
    # ``load_repair_classifier_payload`` can recover even when the sidecar was
    # not written, provided the Accelerate FSDP full-state file is present.
    # Keep the fast missing-check only for checkpoints that have neither form.
    full_state_path = os.path.join(input_dir, "pytorch_model_fsdp.bin")
    if not os.path.isfile(path) and not os.path.isfile(full_state_path):
        if required:
            raise RuntimeError(f"repair classifier checkpoint is missing: {path}")
        return None
    try:
        payload = load_repair_classifier_payload(
            Path(input_dir), recover_from_full_state=True
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        if required:
            raise RuntimeError(
                f"invalid repair classifier checkpoint or full-state recovery failed: {path}"
            ) from error
        return None
    for field in ("input_dim", "hidden_dim"):
        expected = int(getattr(classifier, field))
        if field in payload and int(payload[field]) != expected:
            raise RuntimeError(
                f"repair classifier {field} mismatch: checkpoint={payload[field]} model={expected}"
            )
    result = classifier.load_state_dict(payload["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"repair classifier checkpoint mismatch: {result.missing_keys} / "
            f"{result.unexpected_keys} ({path})"
        )
    return path


def sft(
    data_provider_func,
    model_provider_func,
    forward_step,
    args,
):
    # ===================== Accelerator 初始化 =====================
    logger = get_logger("REDEdit.sft")
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    use_peft_lora = getattr(args, "use_peft_lora", False)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=use_peft_lora)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        kwargs_handlers=[ddp_kwargs],
    )

    # 配置根 logger 为带分布式前缀的彩色格式，便于多进程/多机阅读
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(DistributedColoredFormatter(dist_prefix=get_dist_prefix()))
    root.addHandler(handler)
    root.setLevel(get_default_log_level())

    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None

    if fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD or fsdp_plugin.sharding_strategy is ShardingStrategy.HYBRID_SHARD or fsdp_plugin.reshard_after_forward == True:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None:  # FSDP 2 中为 None
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        log_once(logger, logging.INFO, "Using FSDP stage: %s", fsdp_stage)

        args.use_fsdp = True
        if fsdp_stage == 3:
            log_once(logger, logging.INFO, "Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        fsdp_stage = 0
        log_once(logger, logging.INFO, "FSDP/HSDP is not enabled.")

    if use_peft_lora and fsdp_plugin is not None:
        log_once(logger, logging.WARNING, "LoRA 训练建议不使用 FSDP；当前已启用 FSDP，保存时将仅保存 adapter 权重。")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # 各进程打印当前配置（带 rank 前缀）
    _accel_logger = accelerate_get_logger(__name__, log_level="INFO")
    _accel_logger.info(str(accelerator.state), main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 混合精度对应的权重 dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    
    # ===================== 设置随机种子 =====================
    if args.seed is not None:
        set_seed(args.seed)

    # ===================== 模型初始化 =====================
    log_once(logger, logging.INFO, "Loading model via model_provider...")
    transformer3d, text_encoder, vae, extra_modules = model_provider_func(args, weight_dtype, accelerator.device)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        log_once(logger, logging.INFO, "Gradient checkpointing enabled.")

    if use_peft_lora and getattr(args, "lora_r", 0) <= 0:
        raise ValueError("LoRA 训练请设置 --lora_r > 0")

    # ===================== 数据加载 =====================
    log_once(logger, logging.INFO, "Building train dataloader (process_index=%s)...", accelerator.process_index)
    train_dataloader = data_provider_func(args, accelerator.process_index, accelerator.num_processes)

    # ===================== 优化器初始化 =====================
    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))

    if use_peft_lora:
        trainable_params = [p for p in transformer3d.parameters() if p.requires_grad]
        if args.use_came:
            optimizer = optimizer_cls(
                trainable_params,
                lr=args.learning_rate,
                betas=(0.9, 0.999, 0.9999),
                eps=(1e-30, 1e-16),
            )
        else:
            optimizer = optimizer_cls(
                trainable_params,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )
    else:
        low_lr_ratio = float(args.trainable_modules_low_lr_ratio)
        if not 0.0 < low_lr_ratio <= 1.0:
            raise ValueError("trainable_modules_low_lr_ratio must be in (0, 1]")
        trainable_params_optim = [
            {'params': [], 'lr': args.learning_rate},
            {'params': [], 'lr': args.learning_rate * low_lr_ratio},
        ]
        in_already = []
        for name, param in transformer3d.named_parameters():
            high_lr_flag = False
            if name in in_already:
                continue
            # C registers its auxiliary head before Accelerate/FSDP.  The
            # head is intentionally outside the legacy transformer block
            # range, so include its exact prefix in the normal learning-rate
            # group; otherwise the grad-enabled/optimizer parameter contract
            # would reject a valid C run.
            if (
                getattr(args, "albedo_aux_head", False)
                and name.startswith("albedo_aux_head.")
            ) or (
                getattr(args, "repair_classifier_enabled", False)
                and not getattr(args, "disable_repair_classifier", False)
                and name.startswith("repair_classifier.")
            ):
                in_already.append(name)
                trainable_params_optim[0]['params'].append(param)
                continue
            for trainable_module_name in (args.trainable_modules or []):
                if trainable_module_name in name:
                    in_already.append(name)
                    high_lr_flag = True
                    trainable_params_optim[0]['params'].append(param)
                    break
            if high_lr_flag:
                continue
            for trainable_module_name in (args.trainable_modules_low_learning_rate or []):
                if trainable_module_name in name:
                    in_already.append(name)
                    trainable_params_optim[1]['params'].append(param)
                    break
        trainable_params_optim = [
            group for group in trainable_params_optim if group["params"]
        ]
        optimizer_ids = {
            id(parameter)
            for group in trainable_params_optim
            for parameter in group["params"]
        }
        grad_ids = {
            id(parameter)
            for parameter in transformer3d.parameters()
            if parameter.requires_grad
        }
        if optimizer_ids != grad_ids:
            grad_only = [
                name for name, parameter in transformer3d.named_parameters()
                if parameter.requires_grad and id(parameter) not in optimizer_ids
            ]
            raise RuntimeError(
                "optimizer/grad-enabled parameter sets differ: "
                f"grad_only={grad_only[:3]} optimizer_only={len(optimizer_ids - grad_ids)}"
            )
        if args.use_came:
            optimizer = optimizer_cls(
                trainable_params_optim,
                lr=args.learning_rate,
                # weight_decay=args.adam_weight_decay,
                betas=(0.9, 0.999, 0.9999),
                eps=(1e-30, 1e-16)
            )
        else:
            optimizer = optimizer_cls(
                trainable_params_optim,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )


    if not args.streaming:
        train_dataloader_len = len(train_dataloader)
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
    
    # 学习率调度器初始化 ==============================================================================
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator` ==============================================================================
    transformer3d, optimizer, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if args.streaming:
        args.num_train_epochs = 100 # set to a large number to avoid early stopping
    else:
        train_dataloader_len = len(train_dataloader)
        num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ===================== 保存/加载 hook（LoRA / FSDP / 普通模式） =====================
    if use_peft_lora:
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                model = accelerator.unwrap_model(models[0]) if hasattr(accelerator, "unwrap_model") else models[0]
                if hasattr(model, "save_pretrained"):
                    adapter_path = os.path.join(output_dir, "adapter")
                    os.makedirs(adapter_path, exist_ok=True)
                    model.save_pretrained(adapter_path)
                _save_repair_classifier_state(
                    model,
                    output_dir,
                    required=bool(
                        getattr(args, "repair_classifier_enabled", False)
                        and not getattr(args, "disable_repair_classifier", False)
                    ),
                )
            if hasattr(train_dataloader, "state_dict"):
                with open(os.path.join(output_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"), "wb") as f:
                    pickle.dump([train_dataloader.state_dict(), epoch], f)
            if weights:
                weights.pop()

        def load_model_hook(models, input_dir):
            adapter_path = os.path.join(input_dir, "adapter")
            if os.path.isdir(adapter_path):
                from safetensors.torch import load_file

                model = accelerator.unwrap_model(models[0]) if hasattr(accelerator, "unwrap_model") else models[0]
                state = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
                state = {
                    key.replace(".lora_A.weight", ".lora_A.default.weight")
                       .replace(".lora_B.weight", ".lora_B.default.weight"): value
                    for key, value in state.items()
                }
                current = model.state_dict()
                mismatched = [
                    key for key, value in state.items()
                    if key not in current or current[key].shape != value.shape
                ]
                if mismatched:
                    raise RuntimeError(f"LoRA checkpoint has {len(mismatched)} mismatched tensors")
                load_result = model.load_state_dict(state, strict=False)
                if load_result.unexpected_keys:
                    raise RuntimeError(f"Unexpected LoRA keys while resuming: {load_result.unexpected_keys}")
                _load_repair_classifier_state(
                    model,
                    input_dir,
                    required=bool(
                        getattr(args, "repair_classifier_enabled", False)
                        and not getattr(args, "disable_repair_classifier", False)
                    ),
                )
            while models:
                models.pop()
            pkl_path = os.path.join(input_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    state_dict, first_epoch = pickle.load(f)
                    train_dataloader.load_state_dict(state_dict)
                    log_once(logger, logging.INFO, "Load dataloader state and first_epoch=%s from %s", first_epoch, pkl_path)
    elif fsdp_stage != 0:
        def save_model_hook(models, weights, output_dir):
            # Accelerate saves the FSDP FULL_STATE_DICT as
            # pytorch_model_fsdp.bin before invoking this hook. Do not gather and
            # serialize the 20B transformer a second time here.  The tiny
            # classifier is different: it is nested in the root FSDP flat
            # parameter, so every rank must temporarily summon full params
            # before rank 0 writes its sidecar.  Calling the collector only on
            # rank 0 would deadlock ``summon_full_params``.
            classifier_payload = _repair_classifier_payload(
                models[0],
                required=bool(
                    getattr(args, "repair_classifier_enabled", False)
                    and not getattr(args, "disable_repair_classifier", False)
                ),
            )
            if accelerator.is_main_process:
                if classifier_payload is not None:
                    torch.save(classifier_payload, os.path.join(output_dir, "repair_classifier.pt"))
            with open(os.path.join(output_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"), "wb") as file:
                pickle.dump([train_dataloader.state_dict(), epoch], file)

        def load_model_hook(models, input_dir):
            # FSDP FULL_STATE_DICT: Accelerate already loaded
            # pytorch_model_fsdp.bin and may invoke this pre-hook with an
            # empty models list. Only touch models[0] when present.
            if models:
                _load_repair_classifier_state(
                    models[0],
                    input_dir,
                    required=bool(
                        getattr(args, "repair_classifier_enabled", False)
                        and not getattr(args, "disable_repair_classifier", False)
                    ),
                )
                while models:
                    models.pop()
            pkl_path = os.path.join(
                input_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"
            )
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as file:
                    state_dict, first_epoch = pickle.load(file)
                    train_dataloader.load_state_dict(state_dict)
                    log_once(
                        logger,
                        logging.INFO,
                        "Load dataloader state dict and first_epoch=%s from %s",
                        first_epoch,
                        pkl_path,
                    )
    else:
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                _save_repair_classifier_state(
                    models[0],
                    output_dir,
                    required=bool(
                        getattr(args, "repair_classifier_enabled", False)
                        and not getattr(args, "disable_repair_classifier", False)
                    ),
                )
                weights.pop()

            with open(os.path.join(output_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"), "wb") as file:
                pickle.dump([train_dataloader.state_dict(), epoch], file)

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = extra_modules["dit_class"].from_pretrained(
                    input_dir, subfolder="transformer"
                )
                model.register_to_config(**load_model.config)

                load_result = model.load_state_dict(load_model.state_dict(), strict=False)
                allowed_missing = {
                    key
                    for key in load_result.missing_keys
                    if key.startswith("repair_classifier.")
                    or key.startswith("albedo_aux_head.")
                }
                unexpected = list(load_result.unexpected_keys)
                missing = [
                    key for key in load_result.missing_keys if key not in allowed_missing
                ]
                if missing or unexpected:
                    raise RuntimeError(
                        "transformer checkpoint mismatch while resuming: "
                        f"missing={missing[:3]} unexpected={unexpected[:3]}"
                    )
                _load_repair_classifier_state(
                    model,
                    input_dir,
                    required=bool(
                        getattr(args, "repair_classifier_enabled", False)
                        and not getattr(args, "disable_repair_classifier", False)
                    ),
                )
                del load_model

            with open(os.path.join(input_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"), "rb") as file:
                state_dict, first_epoch = pickle.load(file)
                train_dataloader.load_state_dict(state_dict)
                log_once(logger, logging.INFO,
                    "Load dataloader state dict and first_epoch=%s from %s",
                    first_epoch,
                    os.path.join(input_dir, f"dataloader_{accelerator.process_index}_state_dict.pkl"),
                )
                
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.init_from_checkpoint:
        if args.resume_from_checkpoint:
            raise ValueError(
                "--init_from_checkpoint and --resume_from_checkpoint are mutually exclusive"
            )
        if fsdp_stage == 0 or accelerator.state.fsdp_plugin is None:
            raise ValueError("--init_from_checkpoint currently requires FSDP")
        init_dir = os.path.realpath(os.path.expanduser(args.init_from_checkpoint))
        marker_path = os.path.join(init_dir, "checkpoint_complete.json")
        try:
            with open(marker_path, encoding="utf-8") as marker_file:
                init_marker = json.load(marker_file)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid init checkpoint marker: {marker_path}") from error
        expected_init = {
            "complete": True,
            "artifact_kind": "transformer",
            "world_size": accelerator.num_processes,
            "transformer_architecture": args.transformer_architecture,
            "condition_fingerprint": getattr(args, "condition_fingerprint", None),
            "condition_variant": getattr(args, "condition_variant", None),
            "condition_role": getattr(args, "condition_role", None),
            "condition_model_hash": getattr(args, "condition_model_hash", None),
            "condition_checkpoint_step": _marker_scalar(
                getattr(args, "condition_checkpoint_step", None)
            ),
        }
        init_mismatches = {
            key: (init_marker.get(key), value)
            for key, value in expected_init.items()
            if value is not None and _marker_scalar(init_marker.get(key)) != value
        }
        if init_mismatches:
            raise RuntimeError(
                f"init checkpoint identity mismatch: {init_mismatches} ({marker_path})"
            )
        expected_modules = list(args.trainable_modules or [])
        if list(init_marker.get("trainable_modules") or []) != expected_modules:
            raise RuntimeError(
                "init checkpoint trainable_modules mismatch: "
                f"checkpoint={init_marker.get('trainable_modules')} current={expected_modules}"
            )
        accelerator.print(
            f"Initializing FSDP transformer weights only from {init_dir}; "
            "optimizer/scheduler/dataloader/global-step remain fresh"
        )
        load_result = load_fsdp_model(
            accelerator.state.fsdp_plugin,
            accelerator,
            transformer3d,
            init_dir,
            0,
        )
        if load_result is not None:
            missing = list(getattr(load_result, "missing_keys", []))
            unexpected = list(getattr(load_result, "unexpected_keys", []))
            if missing or unexpected:
                raise RuntimeError(
                    "init checkpoint transformer mismatch: "
                    f"missing={missing[:3]} unexpected={unexpected[:3]}"
                )
        accelerator.wait_for_everyone()
        set_seed(args.seed, device_specific=True)

    def save_checkpoint(global_step, epoch):
        """Save adapter, optimizer/scheduler, RNG and rank-local dataloader state."""
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

        if accelerator.is_main_process and args.checkpoints_total_limit is not None:
            checkpoints = []
            for name in os.listdir(args.output_dir):
                if not name.startswith("checkpoint-"):
                    continue
                try:
                    checkpoints.append((int(name.rsplit("-", 1)[1]), name))
                except ValueError:
                    continue
            checkpoints.sort()
            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                for _, removing_checkpoint in checkpoints[:num_to_remove]:
                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

        accelerator.wait_for_everyone()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        # Every rank must participate so Accelerate writes rank-local RNG state;
        # our save hook additionally writes rank-local dataloader state.
        accelerator.save_state(save_path)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            required_rank_files = [
                os.path.join(save_path, "scheduler.bin"),
            ]
            sharded_state = bool(args.use_fsdp and
                accelerator.state.fsdp_plugin.state_dict_type.name == "SHARDED_STATE_DICT")
            if sharded_state:
                for dirname in ("pytorch_model_fsdp_0", "optimizer_0"):
                    required_rank_files.append(os.path.join(save_path, dirname, ".metadata"))
                    for rank in range(accelerator.num_processes):
                        required_rank_files.append(os.path.join(save_path, dirname, f"__{rank}_0.distcp"))
            else:
                required_rank_files.append(os.path.join(save_path, "optimizer.bin"))
            for rank in range(accelerator.num_processes):
                required_rank_files.extend([
                    os.path.join(save_path, f"dataloader_{rank}_state_dict.pkl"),
                    os.path.join(save_path, f"random_states_{rank}.pkl"),
                ])
            missing = [path for path in required_rank_files if not os.path.isfile(path)]
            artifact_kind = "adapter" if use_peft_lora else "transformer"
            if use_peft_lora:
                artifact_path = os.path.join(save_path, "adapter", "adapter_model.safetensors")
                if not os.path.isfile(artifact_path):
                    missing.append(artifact_path)
                if bool(
                    getattr(args, "repair_classifier_enabled", False)
                    and not getattr(args, "disable_repair_classifier", False)
                ):
                    classifier_path = os.path.join(save_path, "repair_classifier.pt")
                    if not os.path.isfile(classifier_path):
                        missing.append(classifier_path)
            else:
                full_state = os.path.join(save_path, "pytorch_model_fsdp.bin")
                legacy_full_state = os.path.join(save_path, "diffusion_pytorch_model.safetensors")
                diffusers_dir = os.path.join(save_path, "transformer")
                if (
                    not sharded_state
                    and
                    not os.path.isfile(full_state)
                    and not os.path.isfile(legacy_full_state)
                    and not os.path.isdir(diffusers_dir)
                ):
                    missing.append(f"{full_state}, {legacy_full_state}, or {diffusers_dir}")
            if missing:
                raise RuntimeError(f"Checkpoint is missing {len(missing)} rank-local state files: {missing}")
            _atomic_json_dump(
                {
                    "complete": True,
                    "global_step": global_step,
                    "epoch": epoch,
                    "world_size": accelerator.num_processes,
                    "seed": args.seed,
                    "artifact_kind": artifact_kind,
                    "fsdp_state_dict_type": (accelerator.state.fsdp_plugin.state_dict_type.name
                                             if args.use_fsdp else None),
                    "transformer_architecture": args.transformer_architecture,
                    "attention_backend": extra_modules.get("attention_backend", args.attention_backend),
                    "objective": "hybrid_x0_pixel" if args.x0_pixel_loss else "flow",
                    "x0_sigma_band": args.x0_sigma_band if args.x0_pixel_loss else None,
                    "flow_loss_weight": args.flow_loss_weight if args.x0_pixel_loss else None,
                    "latent_x0_weight": args.latent_x0_weight if args.x0_pixel_loss else None,
                    "pixel_l1_weight": args.pixel_l1_weight if args.x0_pixel_loss else None,
                    "pixel_lpips_weight": args.pixel_lpips_weight if args.x0_pixel_loss else None,
                    "pixel_loss_fg_only": args.pixel_loss_fg_only if args.x0_pixel_loss else False,
                    "pixel_background_weight": args.pixel_background_weight if args.x0_pixel_loss else 0.0,
                    "restoration_v3_loss": bool(
                        args.x0_pixel_loss and getattr(args, "restoration_v3_loss", False)
                    ),
                    "pixel_full_l1_weight": getattr(args, "pixel_full_l1_weight", 0.0),
                    "pixel_subject_l1_weight": getattr(args, "pixel_subject_l1_weight", 0.0),
                    "pixel_affected_l1_weight": getattr(args, "pixel_affected_l1_weight", 0.0),
                    "pixel_subject_gradient_weight": getattr(
                        args, "pixel_subject_gradient_weight", 0.0
                    ),
                    "pixel_low_frequency_weight": getattr(
                        args, "pixel_low_frequency_weight", 0.0
                    ),
                    "pixel_low_frequency_scales": getattr(
                        args, "pixel_low_frequency_scales", ""
                    ),
                    "restoration_task_multipliers": getattr(
                        args, "restoration_task_multipliers", ""
                    ),
                    "init_from_checkpoint": os.path.realpath(
                        args.init_from_checkpoint
                    ) if args.init_from_checkpoint else None,
                    "vae_encode_mode": args.vae_encode_mode,
                    "trainable_modules": list(args.trainable_modules or []),
                    # Recipe/cache fields are intentionally absent from the
                    # legacy launcher command unless an albedo recipe opts in.
                    # When present they make same-shape cross-recipe resumes
                    # fail closed before any optimizer state is loaded.
                    "recipe": getattr(args, "recipe", "v1"),
                    "recipe_schema": getattr(args, "recipe_schema", None),
                    "recipe_fingerprint": getattr(args, "recipe_fingerprint", None),
                    "cache_schema": getattr(args, "cache_schema", None),
                    "cache_fingerprint": getattr(args, "cache_fingerprint", None),
                    "schema_fingerprint": getattr(args, "schema_fingerprint", None),
                    "albedo_root": getattr(args, "albedo_root", None),
                    "albedo_aux_head": bool(getattr(args, "albedo_aux_head", False)),
                    "albedo_aux_prefix": "albedo_aux_head.",
                    "repair_classifier_enabled": bool(
                        getattr(args, "repair_classifier_enabled", False)
                        and not getattr(args, "disable_repair_classifier", False)
                    ),
                    "repair_classifier_loss_weight": float(
                        getattr(args, "repair_classifier_loss_weight", 0.1)
                    ),
                    "repair_classifier_threshold": float(
                        getattr(args, "repair_classifier_threshold", 0.5)
                    ),
                    "repair_classifier_label_source": "paired_degraded_gt",
                    "repair_classifier_prefix": "repair_classifier.",
                    # Offline Qwen hidden states are an input contract, not a
                    # train-time model.  Persist their identity in every
                    # checkpoint so a resume cannot silently mix A/B/base
                    # condition caches or a different gate step.
                    "condition_schema": getattr(args, "condition_schema", None),
                    "condition_kind": getattr(args, "condition_kind", None),
                    "condition_fingerprint": getattr(args, "condition_fingerprint", None),
                    "condition_cache_fingerprint": getattr(
                        args, "condition_cache_fingerprint", None
                    ),
                    "condition_variant": getattr(args, "condition_variant", None),
                    "condition_role": getattr(args, "condition_role", None),
                    "condition_model_hash": getattr(args, "condition_model_hash", None),
                    "condition_checkpoint_step": _marker_scalar(
                        getattr(args, "condition_checkpoint_step", None)
                    ),
                    "condition_checkpoint_world_size": _marker_scalar(
                        getattr(args, "condition_checkpoint_world_size", None)
                    ),
                    "condition_model_identity": getattr(args, "condition_model_identity", None),
                    "condition_model_path": getattr(args, "condition_model_identity", None),
                },
                os.path.join(save_path, "checkpoint_complete.json"),
            )
        accelerator.wait_for_everyone()
        log_once(logger, logging.INFO, "Saved state%s to %s", " (adapter)" if use_peft_lora else "", save_path)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts", None)
        tracker_config.pop("trainable_modules", None)
        tracker_config.pop("trainable_modules_low_learning_rate", None)
        tracker_config.pop("fix_sample_size", None)
        tracker_config.pop("lora_target_modules", None)
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # ===================== 训练循环 =====================
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    log_once(logger, logging.INFO, "***** Running %s *****", "LoRA training" if use_peft_lora else "training")
    log_once(logger, logging.INFO, "  Num Epochs = %s", args.num_train_epochs)
    log_once(logger, logging.INFO, "  Instantaneous batch size per device = %s", args.train_batch_size)
    log_once(logger, logging.INFO, "  Total train batch size (w. parallel, distributed & accumulation) = %s", total_batch_size)
    log_once(logger, logging.INFO, "  Gradient Accumulation steps = %s", args.gradient_accumulation_steps)
    log_once(logger, logging.INFO, "  Total optimization steps = %s", args.max_train_steps)
    # Potentially load in the weights and states from a previous save
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [
                d for d in dirs
                if d.startswith("checkpoint-")
                and os.path.isfile(os.path.join(args.output_dir, d, "checkpoint_complete.json"))
            ]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            
            pkl_path = os.path.join(os.path.join(args.output_dir, path), f"dataloader_{accelerator.process_index}_state_dict.pkl")

            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, first_epoch = pickle.load(file)
                log_once(logger, logging.INFO, "Load pkl from %s, first_epoch=%s.", pkl_path, first_epoch)
            else:
                first_epoch = global_step // num_update_steps_per_epoch if not args.streaming else 0

            accelerator.print(f"Resuming from checkpoint {path}")
            # Gate A/B training consumes offline Qwen hidden-state tensors.
            # Their model/checkpoint identity is therefore part of the
            # optimizer state contract and must be checked before Accelerate
            # restores any rank-local state.
            condition_fingerprint = getattr(args, "condition_fingerprint", None)
            if condition_fingerprint:
                marker_path = os.path.join(args.output_dir, path, "checkpoint_complete.json")
                try:
                    with open(marker_path, encoding="utf-8") as marker_file:
                        marker = json.load(marker_file)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"invalid condition checkpoint marker: {marker_path}"
                    ) from error
                expected = {
                    "condition_schema": getattr(args, "condition_schema", None),
                    "condition_kind": getattr(args, "condition_kind", None),
                    "condition_fingerprint": condition_fingerprint,
                    "condition_cache_fingerprint": getattr(
                        args, "condition_cache_fingerprint", None
                    ),
                    "condition_variant": getattr(args, "condition_variant", None),
                    "condition_role": getattr(args, "condition_role", None),
                    "condition_model_hash": getattr(args, "condition_model_hash", None),
                    "condition_checkpoint_step": _marker_scalar(
                        getattr(args, "condition_checkpoint_step", None)
                    ),
                    "condition_checkpoint_world_size": _marker_scalar(
                        getattr(args, "condition_checkpoint_world_size", None)
                    ),
                    "condition_model_identity": getattr(args, "condition_model_identity", None),
                }
                mismatches = {
                    key: (marker.get(key), value)
                    for key, value in expected.items()
                    if value is not None and _marker_scalar(marker.get(key)) != value
                }
                if mismatches:
                    raise RuntimeError(
                        "checkpoint/resume condition incompatibility: "
                        f"{mismatches} ({marker_path})"
                    )
            # A checkpoint marker is deliberately checked before Accelerate
            # restores optimizer/model state.  V1/V2 have no opt-in recipe
            # marker requirement; albedo runs must match recipe and cache
            # fingerprints exactly.
            if str(getattr(args, "recipe", "v1")) in {
                "albedo_aware", "albedo_consistency", "rgb_albedo_multitask"
            }:
                marker_path = os.path.join(args.output_dir, path, "checkpoint_complete.json")
                try:
                    with open(marker_path, encoding="utf-8") as marker_file:
                        marker = json.load(marker_file)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                    raise RuntimeError(f"invalid albedo checkpoint marker: {marker_path}") from error
                expected = {
                    "recipe": getattr(args, "recipe", None),
                    "recipe_schema": getattr(args, "recipe_schema", None),
                    "recipe_fingerprint": getattr(args, "recipe_fingerprint", None),
                    "cache_schema": getattr(args, "cache_schema", None),
                    "cache_fingerprint": getattr(args, "cache_fingerprint", None),
                    "schema_fingerprint": getattr(args, "schema_fingerprint", None),
                }
                mismatches = {
                    key: (marker.get(key), value)
                    for key, value in expected.items()
                    if value is not None and marker.get(key) != value
                }
                if mismatches:
                    raise RuntimeError(
                        "checkpoint/resume recipe incompatibility: "
                        f"{mismatches} ({marker_path})"
                    )
            classifier_enabled = bool(
                getattr(args, "repair_classifier_enabled", False)
                and not getattr(args, "disable_repair_classifier", False)
            )
            if classifier_enabled:
                marker_path = os.path.join(args.output_dir, path, "checkpoint_complete.json")
                try:
                    with open(marker_path, encoding="utf-8") as marker_file:
                        marker = json.load(marker_file)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"invalid repair-classifier checkpoint marker: {marker_path}"
                    ) from error
                expected = {
                    "repair_classifier_enabled": True,
                    "repair_classifier_loss_weight": float(
                        getattr(args, "repair_classifier_loss_weight", 0.1)
                    ),
                    "repair_classifier_threshold": float(
                        getattr(args, "repair_classifier_threshold", 0.5)
                    ),
                    "repair_classifier_label_source": "paired_degraded_gt",
                }
                mismatches = {
                    key: (marker.get(key), value)
                    for key, value in expected.items()
                    if marker.get(key) != value
                }
                if mismatches:
                    raise RuntimeError(
                        "checkpoint/resume classifier incompatibility: "
                        f"{mismatches} ({marker_path})"
                    )
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        initial_global_step = 0

    rng_micro_step = global_step * args.gradient_accumulation_steps

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        train_loss_components = {}
        if args.streaming:
            if not (args.resume_from_checkpoint and epoch == first_epoch):
                # Dont set epoch if resuming from checkpoint and epoch is the first epoch
                # because the state of the streaming dataset has already loaded from checkpoint
                train_dataloader.dataset.set_epoch(epoch)
            
        for step, batch in enumerate(train_dataloader):
            if isinstance(batch, dict) and batch == {}:
                log_once(logger, logging.WARNING, "Empty batch encountered; skipping.")
                continue
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(accelerator.device)

            # Data batch sanity check
            if accelerator.is_main_process and epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                os.makedirs(os.path.join(args.output_dir, "sanity_check"), exist_ok=True)
                if batch.get('source_images_transposed', None) is not None:
                    source_images_transposed = batch['source_images_transposed']
                    source_images =  list(map(list, zip(*source_images_transposed)))
                
                    for idx, (pixel_value, source_im, text) in enumerate(zip(pixel_values, source_images, texts)):
                        pixel_value = pixel_value[None, ...]
                        sanity_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                        save_image(pixel_value, f"{args.output_dir}/sanity_check/{sanity_name[:10]}.jpg", rescale=True)
                        for local_index, _source_im in enumerate(source_im):
                            _source_im = Image.fromarray(np.uint8(_source_im))
                            _source_im.save(f"{args.output_dir}/sanity_check/source_{local_index}_{sanity_name[:10]}.jpg")
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        sanity_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                        save_image(pixel_value, f"{args.output_dir}/sanity_check/{sanity_name[:10]}.jpg", rescale=True)
            
            with accelerator.accumulate(transformer3d):
                loss = forward_step(
                    args, 
                    accelerator.process_index,
                    transformer3d, 
                    vae, 
                    text_encoder, 
                    extra_modules, 
                    batch, 
                    weight_dtype, 
                    accelerator.device,
                    rng_micro_step,
                )
                rng_micro_step += 1

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                for name, value in extra_modules.get("loss_components", {}).items():
                    component = torch.as_tensor(
                        value, dtype=torch.float32, device=accelerator.device
                    ).reshape(1)
                    avg_component = accelerator.gather(component).mean().item()
                    train_loss_components[name] = (
                        train_loss_components.get(name, 0.0)
                        + avg_component / args.gradient_accumulation_steps
                    )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_fsdp:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm
                    
                    if not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer3d.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    clip_parameters = (
                        transformer3d.parameters() if args.use_fsdp else trainable_params
                    )
                    norm_sum = accelerator.clip_grad_norm_(clip_parameters, actual_max_grad_norm)
                    if not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                # Record step loss to TensorBoard
                if accelerator.is_main_process:
                    step_loss_value = train_loss
                    writer.add_scalar("loss/step_loss", step_loss_value, global_step=global_step)
                    writer.add_scalar("loss/train_loss", train_loss, global_step=global_step)
                    for name, value in train_loss_components.items():
                        writer.add_scalar(f"loss/{name}", value, global_step=global_step)
                train_loss = 0.0
                train_loss_components = {}

                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(global_step, epoch)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    # Avoid rewriting a very large FSDP checkpoint when the final step was
    # already persisted by the periodic checkpoint branch above.
    if args.skip_final_checkpoint:
        log_once(
            logger,
            logging.INFO,
            "Skipping final checkpoint because --skip_final_checkpoint is enabled.",
        )
    elif global_step % args.checkpointing_steps != 0:
        save_checkpoint(global_step, epoch)

    if accelerator.is_main_process:
        writer.flush()
        writer.close()
    accelerator.end_training()


    return


if __name__ == "__main__":
    from .forward_step import forward_step_impl
    from .model_provider import model_provider_impl
    from .data_provider import data_provider_impl
    from .arguments import parse_args

    args = parse_args()
    if getattr(args, "use_peft_lora", False) and getattr(args, "lora_r", 0) <= 0:
        raise ValueError("LoRA 训练请设置 --lora_r > 0")
    sft(
        data_provider_impl,
        model_provider_impl,
        forward_step_impl,
        args,
    )
