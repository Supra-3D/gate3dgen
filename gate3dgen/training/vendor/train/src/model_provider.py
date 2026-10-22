# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
"""
模型构建：加载 VAE、Transformer、可选 Text Encoder 与 Scheduler，并返回供 SFT 使用的组件。
"""
import logging
import torch
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from .utils.discrete_sampler import DiscreteSampling
from .utils.log_utils import get_logger, log_once
from .models import QwenImageRestorationTransformer2DModel
from gate3dgen.training.lora_contract import assert_only_lora_trainable
from gate3dgen.training.training_contract import freeze_first_by_name, write_trainable_parameters

logger = get_logger(__name__)


def _assert_uniform_parameter_dtype(module, expected_dtype):
    """Fail before FSDP if any module parameter has a different dtype.

    FSDP's flat-parameter construction requires one dtype for every parameter
    it manages.  The base Transformer is explicitly loaded in ``weight_dtype``
    and auxiliary heads are registered after that load, so a newly-created
    ``nn.Module`` can otherwise silently remain in the PyTorch default
    ``float32`` dtype and only fail much later inside ``accelerator.prepare``.
    """

    mismatched = [
        (name, str(parameter.dtype), bool(parameter.requires_grad))
        for name, parameter in module.named_parameters()
        if parameter.dtype != expected_dtype
    ]
    if mismatched:
        preview = ", ".join(
            f"{name}={dtype}{'*' if trainable else ''}"
            for name, dtype, trainable in mismatched[:8]
        )
        raise RuntimeError(
            "FSDP requires uniform Transformer parameter dtype "
            f"{expected_dtype}; mismatched parameters: {preview}"
        )


def apply_partial_unfreeze(transformer3d, trainable_patterns):
    """Freeze the complete transformer, then enable only exact pattern matches."""

    trainable, frozen = freeze_first_by_name(
        transformer3d, trainable_patterns or []
    )
    log_once(
        logger,
        logging.INFO,
        "Partial unfreeze: %s trainable / %s frozen parameters (%.2f%% trainable)",
        f"{trainable:,}",
        f"{frozen:,}",
        100.0 * trainable / (trainable + frozen),
    )
    return trainable, frozen

def create_peft_lora_model(transformer3d, args, weight_dtype=None):
    """
    使用 PEFT 库为 transformer 创建 LoRA 模型
    参考: https://huggingface.co/docs/peft/accelerate/fsdp
    参考: https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh
    """
    from peft import LoraConfig, get_peft_model
    
    # 解析 target_modules
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    
    print(f"Creating LoRA with config:")
    print(f"  rank (r): {args.lora_r}")
    print(f"  alpha: {args.lora_alpha}")
    print(f"  dropout: {args.lora_dropout}")
    print(f"  target_modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        init_lora_weights="gaussian",
    )
    
    # 先冻结所有参数
    transformer3d.requires_grad_(False)
    
    # 用 PEFT 包装模型
    transformer3d = get_peft_model(transformer3d, lora_config)
    trainable_lora_tensors = assert_only_lora_trainable(transformer3d.named_parameters())
    print(f"Validated {trainable_lora_tensors} trainable LoRA tensors; base model is frozen")
    
    # 关键修复: 将 LoRA 参数转换为与基础模型相同的 dtype
    # FSDP 要求所有参数必须具有统一的 dtype
    if weight_dtype is not None:
        print(f"Converting LoRA parameters to {weight_dtype} for FSDP compatibility")
        for name, param in transformer3d.named_parameters():
            if param.requires_grad:  # 只转换 LoRA 参数
                param.data = param.data.to(weight_dtype)
    
    # 打印可训练参数信息
    transformer3d.print_trainable_parameters()
    
    # 验证所有参数的 dtype
    dtypes = set()
    for name, param in transformer3d.named_parameters():
        dtypes.add(param.dtype)
    print(f"Model parameter dtypes: {dtypes}")
    if len(dtypes) > 1:
        print("WARNING: Model has mixed dtypes, this may cause issues with FSDP!")
    
    # 如果有预训练的 LoRA 权重，加载它
    if args.lora_path is not None:
        print(f"Loading pretrained LoRA weights from: {args.lora_path}")
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        import os
        
        if os.path.isdir(args.lora_path):
            # 如果是目录，尝试加载 adapter_model.safetensors
            lora_file = os.path.join(args.lora_path, "adapter_model.safetensors")
            if not os.path.exists(lora_file):
                lora_file = os.path.join(args.lora_path, "adapter_model.bin")
        else:
            lora_file = args.lora_path
            
        if lora_file.endswith(".safetensors"):
            state_dict = load_file(lora_file)
        else:
            state_dict = torch.load(lora_file, map_location="cpu")
        
        # 将加载的权重也转换为正确的 dtype
        if weight_dtype is not None:
            state_dict = {k: v.to(weight_dtype) for k, v in state_dict.items()}
        
        set_peft_model_state_dict(transformer3d, state_dict)
        print(f"LoRA weights loaded successfully!")
    
    return transformer3d


def model_provider_impl(
    args, 
    weight_dtype, 
    device
):
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    ).to(weight_dtype)
    vae.eval()

    architecture = getattr(args, "transformer_architecture", "stock")
    transformer_class = (
        QwenImageRestorationTransformer2DModel
        if architecture == "restoration_custom"
        else QwenImageTransformer2DModel
    )
    transformer3d = transformer_class.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    ).to(weight_dtype)
    attention_backend = getattr(args, "attention_backend", "native")
    if architecture == "restoration_custom":
        attention_backend = transformer3d.set_restoration_attention_backend(
            attention_backend, device=device, dtype=weight_dtype
        )
    elif attention_backend != "native":
        raise ValueError(
            "non-native attention requires --transformer_architecture restoration_custom"
        )

    # Register C's auxiliary module immediately after constructing the
    # backbone, before optional transformer checkpoint loading and before
    # Accelerate/FSDP can wrap it.  This also lets a direct --transformer_path
    # containing ``albedo_aux_head.*`` load without treating those keys as
    # unexpected.
    albedo_aux_requested = bool(getattr(args, "albedo_aux_head", False))
    if albedo_aux_requested:
        from gate3dgen.training.experimental.albedo_head import install_albedo_aux_prehook, register_albedo_aux_head
        proj_out = getattr(transformer3d, "proj_out", None)
        input_dim = getattr(proj_out, "in_features", None)
        if input_dim is None or int(input_dim) <= 0:
            raise ValueError(
                "--albedo_aux_head requires transformer3d.proj_out.in_features "
                "so the head consumes shared pre-projection features"
            )
        input_dim = int(input_dim)
        register_albedo_aux_head(
            transformer3d,
            input_dim=input_dim,
            hidden_dim=int(getattr(args, "albedo_aux_hidden_dim", 256)),
            output_dim=int(getattr(args, "albedo_aux_output_dim", 3)),
        )
        transformer3d.albedo_aux_head.to(dtype=weight_dtype)
        install_albedo_aux_prehook(transformer3d)
        log_once(
            logger,
            logging.INFO,
            "Registered C albedo_aux_head before checkpoint/FSDP on proj_out pre-hook (input_dim=%s)",
            input_dim,
        )

    # Register the gate before any FSDP/Accelerate wrapping.  The classifier
    # consumes the packed latent width (VAE z_dim * 2 * 2), not prompt/VLM
    # features and not the transformer's velocity output.
    repair_classifier_requested = bool(
        getattr(args, "repair_classifier_enabled", False)
        and not getattr(args, "disable_repair_classifier", False)
    )
    if repair_classifier_requested:
        from gate3dgen.training.experimental.repair_classifier import (
            get_repair_classifier, install_repair_classifier_prehook, register_repair_classifier_head,
        )
        latent_z_dim = getattr(vae.config, "z_dim", None)
        if latent_z_dim is None:
            latent_z_dim = getattr(getattr(transformer3d, "config", None), "in_channels", None)
            # Qwen's transformer in_channels already describes packed width.
        latent_width = int(latent_z_dim) * 4 if int(latent_z_dim or 0) < 32 else int(latent_z_dim)
        if latent_width <= 0:
            raise ValueError(
                "--repair_classifier_enabled requires VAE config.z_dim or "
                "transformer config.in_channels"
            )
        register_repair_classifier_head(
            transformer3d,
            input_dim=latent_width,
            hidden_dim=int(getattr(args, "repair_classifier_hidden_dim", 128)),
            dropout=float(getattr(args, "repair_classifier_dropout", 0.0)),
        )
        # The Transformer is loaded in ``weight_dtype`` before this head is
        # registered.  New PyTorch modules default to fp32, which creates a
        # mixed-dtype parameter set and makes FSDP fail while flattening the
        # model.  Keep the head numerically aligned with the backbone before
        # any optimizer/FSDP construction; its forward still performs the
        # latent pooling in fp32 for stability.
        classifier = get_repair_classifier(transformer3d)
        if classifier is None:  # pragma: no cover - registration is above
            raise RuntimeError("registered repair classifier is missing")
        classifier.to(dtype=weight_dtype)
        install_repair_classifier_prehook(transformer3d)
        log_once(
            logger,
            logging.INFO,
            "Registered frozen-VAE latent repair classifier (input_dim=%s, hidden_dim=%s, dtype=%s, root_prehook=on)",
            latent_width,
            getattr(args, "repair_classifier_hidden_dim", 128),
            weight_dtype,
        )
    # 冻结 VAE，仅训练 Transformer
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    if args.transformer_path is not None:
        log_once(logger, logging.INFO, "Loading transformer from checkpoint: %s", args.transformer_path)
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        log_once(logger, logging.INFO, "Transformer load_state_dict: missing_keys=%s, unexpected_keys=%s", len(m), len(u))
        assert len(u) == 0

    if args.vae_path is not None:
        log_once(logger, logging.INFO, "Loading VAE from checkpoint: %s", args.vae_path)
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        log_once(logger, logging.INFO, "VAE load_state_dict: missing_keys=%s, unexpected_keys=%s", len(m), len(u))
        assert len(u) == 0

    # 根据是否使用 LoRA 来设置训练模式
    if getattr(args, 'use_peft_lora', False):
        # 使用 PEFT LoRA 微调
        log_once(logger, logging.INFO, "=== Using PEFT LoRA for training ===")
        # 传入 weight_dtype 以确保 LoRA 参数与基础模型具有相同的 dtype
        transformer3d = create_peft_lora_model(transformer3d, args, weight_dtype=weight_dtype)
        transformer3d.train()
        if repair_classifier_requested:
            classifier = get_repair_classifier(transformer3d)
            if classifier is None:
                raise RuntimeError("registered repair classifier is missing after PEFT wrapping")
            classifier.requires_grad_(True)
    else:
        transformer3d.train()
        apply_partial_unfreeze(
            transformer3d,
            list(args.trainable_modules or [])
            + list(args.trainable_modules_low_learning_rate or []),
        )
        if albedo_aux_requested:
            transformer3d.albedo_aux_head.requires_grad_(True)
        if repair_classifier_requested:
            classifier = get_repair_classifier(transformer3d)
            if classifier is None:
                raise RuntimeError("registered repair classifier is missing after freeze-first setup")
            classifier.requires_grad_(True)

    # Keep a local, actionable error if a future auxiliary module introduces a
    # dtype mismatch.  Without this check the same problem is reported deep in
    # FSDP's flattening internals after every rank has loaded the 20B model.
    _assert_uniform_parameter_dtype(transformer3d, weight_dtype)
    write_trainable_parameters(transformer3d, args.output_dir)


    # 可选：同步加载 Text Encoder（用于在线编码），否则使用预提取的 embedding
    if args.condition_encoder_mode == "sync":
        # Get Tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="tokenizer"
        )

        # Get processor
        processor = Qwen2VLProcessor.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="processor"
        )

        # Get text encoder
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            torch_dtype=weight_dtype
        )
        text_encoder = text_encoder.eval()
        text_encoder.requires_grad_(False)
        text_encoder.to(device)
        log_once(logger, logging.INFO, "Text encoder loaded (condition_encoder_mode=sync).")
    else:
        tokenizer = None
        processor = None
        text_encoder = None

    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device)

    # 噪声调度器与时间步采样
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    lpips_model = None
    needs_lpips = (
        getattr(args, "pixel_lpips_weight", 0.0) > 0
        or getattr(args, "albedo_perceptual_weight", 0.0) > 0
    )
    if needs_lpips:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "pixel/albedo perceptual loss requires `pip install lpips`; "
                "set both LPIPS weights to 0 to use L1/latent supervision without LPIPS"
            ) from error
        lpips_model = lpips.LPIPS(net="vgg").to(device=device, dtype=torch.float32)
        lpips_model.eval()
        lpips_model.requires_grad_(False)
        log_once(logger, logging.INFO, "Loaded frozen LPIPS-VGG for restoration loss")

    extra_modules = {
        "noise_scheduler": noise_scheduler,
        "idx_sampling": idx_sampling,
        "latents_mean": latents_mean,
        "latents_std": latents_std,
        "dit_class": transformer_class,
        "tokenizer": tokenizer,
        "processor": processor,
        "lpips_model": lpips_model,
        "transformer_architecture": architecture,
        "attention_backend": attention_backend,
    }
    
    return (transformer3d, text_encoder, vae, extra_modules)
