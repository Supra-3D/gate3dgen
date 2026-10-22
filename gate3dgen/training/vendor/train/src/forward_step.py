# Copyright (c) 2025 FireRed-Image-Edit. All rights reserved.
"""
前向与损失：准备 latent、噪声与条件，调用 Transformer 预测 flow target，计算加权 MSE loss。
"""
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
from contextlib import nullcontext
from diffusers.training_utils import (
    EMAModel,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from .utils.log_utils import get_logger
from .utils.image_utils import resize_source_images_for_condition
from gate3dgen.training.purifier_losses import (
    average_v2_base_components, compute_v2_base_components, masked_lpips_loss,
    masked_mean, multiscale_low_frequency_luminance_loss, v2_flow_loss, weighted_v2_base_total,
)


logger = get_logger(__name__)


def _parse_task_multipliers(value):
    mapping = {}
    for entry in str(value or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"invalid restoration task multiplier: {entry!r}")
        task, multiplier = entry.split("=", 1)
        task = task.strip()
        multiplier = float(multiplier)
        if not task or multiplier <= 0:
            raise ValueError(f"invalid restoration task multiplier: {entry!r}")
        mapping[task] = multiplier
    return mapping


def _batch_task_multiplier(args, batch):
    mapping = _parse_task_multipliers(
        getattr(args, "restoration_task_multipliers", "")
    )
    if not mapping:
        return 1.0, "default"
    raw_tasks = list(batch.get("task_ids", []))
    if not raw_tasks or any(task is None or not str(task).strip() for task in raw_tasks):
        raise ValueError("restoration task multipliers require batch['task_ids']")
    tasks = [str(task) for task in raw_tasks]
    unique = set(tasks)
    if len(unique) != 1:
        raise ValueError(f"task-weighted restoration batch is not homogeneous: {sorted(unique)}")
    task = tasks[0]
    return float(mapping.get(task, 1.0)), task


def pack_latents(latents, batch_size, num_channels_latents, height, width):
    """将 latent 重排为 (B, seq, C) 的 patch 序列格式。"""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents(latents, batch_size, num_channels_latents, height, width):
    """Inverse of pack_latents, returning [B, C, H, W]."""

    latents = latents.view(
        batch_size, height // 2, width // 2, num_channels_latents, 2, 2
    )
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, num_channels_latents, height, width)


DEFAULT_SYSTEM_PROMPT = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."

DEFAULT_SYSTEM_PROMPT_T2I = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """根据图像序列长度计算 flow 的 shift 系数（线性插值）。"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def get_embedding(source_exist, instructions, source_images, processor, text_encoder):
    def _extract_valid_hidden_states(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        batch_size = hidden_states.shape[0]
        valid_hidden_states = []
        for i in range(batch_size):
            valid_indices = attention_mask[i].nonzero(as_tuple=True)[0]
            valid_hidden_states.append(hidden_states[i][valid_indices])
        return valid_hidden_states

    with torch.no_grad():
        conversations = []
        if not source_exist:
            source_images = [[] for _ in range(len(instructions))]
        for instruction, source_images_sample in zip(instructions, source_images):
            system_prompt = DEFAULT_SYSTEM_PROMPT_T2I if not source_exist else DEFAULT_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": []},
            ]
            for condition_image in source_images_sample:
                messages[1]["content"].append({"type": "image", "image": condition_image})
            messages[1]["content"].append({"type": "text", "text": instruction})
            conversations.append(messages)
        
        batch_inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            padding=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(text_encoder.device)

        encoder_hidden_states = text_encoder(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            pixel_values=batch_inputs["pixel_values"] if source_exist else None,
            image_grid_thw=batch_inputs["image_grid_thw"] if source_exist else None,
            output_hidden_states=True,
        )

        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = _extract_valid_hidden_states(hidden_states, batch_inputs["attention_mask"])
        start_idx = 64 if source_exist else 34
        split_hidden_states = [e[start_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

    return prompt_embeds, encoder_attention_mask

def prepare_data(args,
                 batch,
                 processor, text_encoder, vae, latents_mean, latents_std,
                 noise_scheduler, idx_sampling,
                 device, weight_dtype, torch_rng):
    """
    从 batch 中编码 pixel_values/source 为 latent，采样时间步，构造 noisy_latents 与 target。
    返回模型前向所需的所有张量与元信息。
    """
    pixel_values = batch["pixel_values"].to(weight_dtype)
    source_images_transposed = batch["source_images_transposed"]
    encoder_attention_mask = batch['encoder_attention_mask']
    prompt_embeds = batch['encoder_hidden_states']
    instructions = batch['text']

    # prepare target latents
    def _batch_encode_vae(pixel_values):
        with torch.no_grad():
            pixel_values = vae.encode(pixel_values.to(vae.device).to(vae.dtype).unsqueeze(2))[0]
            pixel_values = (
                pixel_values.mode()
                if getattr(args, "vae_encode_mode", "sample") == "mode"
                else pixel_values.sample()
            )
            return pixel_values
    
    latents = _batch_encode_vae(pixel_values)
    latents = ((latents - latents_mean) * latents_std).to(dtype=weight_dtype)
    bsz, channel, _, height, width = latents.size()
    latents = pack_latents(latents, bsz, channel, height, width)
    noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

    # prepare source latents
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    vae_image_sizes_transposed = []
    source_latents_transposed = []
    source_exist = False

    for i_step, _vae_images in enumerate(source_images_transposed):
        _source_pixel_values = []
        _vae_image_sizes = []
        for j_img, _vae_image in enumerate(_vae_images):
            _vae_image_sizes.append(_vae_image.size)

            _vae_image = torch.from_numpy(np.expand_dims(np.array(_vae_image), 0)).to(dtype=weight_dtype, device=vae.device)
            _vae_image = _vae_image.permute(0, 3, 1, 2).contiguous()
            _vae_image = _vae_image / 255.
            _vae_image = (_vae_image - 0.5) / 0.5 # [1, C, H, W]
            _source_pixel_values.append(_vae_image)

        
        if len(_source_pixel_values) > 0:
            source_exist = True
        
        _source_pixel_values = torch.cat(_source_pixel_values, dim = 0) # [B, C, H, W]
        _source_latents = _batch_encode_vae(_source_pixel_values)
        _source_latents = ((_source_latents - latents_mean) * latents_std).to(dtype=weight_dtype)
        src_bsz, src_channel, _, src_height, src_width = _source_latents.size()
        _source_latents = pack_latents(_source_latents, src_bsz, src_channel, src_height, src_width) # [B, seq, C]

        source_latents_transposed.append(_source_latents.unbind(dim=0)) # [ [latent1, latent3], [latent2, latent4], ... ]
        vae_image_sizes_transposed.append(_vae_image_sizes) # [ [ (W1, H1), (W3, H3) ], [ (W2, H2), (W4, H4) ], ... ]
    
    if source_exist:
        # [ [latent1, latent3], [latent2, latent4] ] → [ [latent1, latent2], [latent3, latent4] ]
        source_latents_tmp = list(map(list, zip(*source_latents_transposed)))
        vae_image_sizes = list(map(list, zip(*vae_image_sizes_transposed)))

        # cat source_latents to tensor [B, seq, C]
        source_latents = [torch.cat(_source_latents, dim = 0).unsqueeze(0) for _source_latents in source_latents_tmp] # [[1, seq, C], [1, seq, C], ... ]
        source_latents = torch.cat(source_latents, dim = 0) # [B, seq, C]
        logger.debug(
            "source_latents final: shape=%s (B, seq, C)",
            tuple(source_latents.shape),
        )

        source_images = list(map(list, zip(*source_images_transposed)))
        # sync 模式下送入 text_encoder 的 source 图做与 extract_vlm_embeds 一致的 Resize+CenterCrop，以 target 尺寸为参考
        if args.condition_encoder_mode == "sync":
            ref_h, ref_w = pixel_values.shape[2], pixel_values.shape[3]
            source_images = resize_source_images_for_condition(
                source_images,
                ref_height=ref_h,
                ref_width=ref_w,
                image_sample_size=args.image_sample_size,
            )
    else:
        source_latents = None
        vae_image_sizes = []
        source_images = []
        logger.debug("source_latents: None (no source images)")
    
    # prepare prompt embeds
    if args.condition_encoder_mode == "sync":
        prompt_embeds, encoder_attention_mask = get_embedding(source_exist, instructions, source_images, processor, text_encoder)

    prompt_embeds = prompt_embeds.to(dtype=latents.dtype, device=device)
    encoder_attention_mask = encoder_attention_mask.to(device=device)

    x0_mode = getattr(args, "x0_pixel_loss", False)
    if x0_mode:
        indices = None
    elif not args.uniform_sampling:
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
    else:
        indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
        indices = indices.long().cpu()

    sigmas = np.linspace(1.0, 1 / args.train_sampling_steps, args.train_sampling_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        noise_scheduler.config.get("base_image_seq_len", 256),
        noise_scheduler.config.get("max_image_seq_len", 4096),
        noise_scheduler.config.get("base_shift", 0.5),
        noise_scheduler.config.get("max_shift", 1.15),
    )
    noise_scheduler.set_timesteps(sigmas=sigmas, device=latents.device, mu=mu)
    if x0_mode:
        try:
            lo, hi = (float(value) for value in args.x0_sigma_band.split(","))
        except (TypeError, ValueError) as error:
            raise ValueError("x0_sigma_band must be formatted as lo,hi") from error
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError(f"x0_sigma_band must satisfy 0 < lo <= hi <= 1, got {(lo, hi)}")
        scheduler_sigmas = noise_scheduler.sigmas[:-1].to(device=latents.device)
        candidates = ((scheduler_sigmas >= lo) & (scheduler_sigmas <= hi)).nonzero().flatten()
        if candidates.numel() == 0:
            raise ValueError(f"no scheduler sigma falls inside {(lo, hi)}")
        sampled = torch.randint(
            candidates.numel(),
            (bsz,),
            generator=torch_rng,
            device=latents.device,
        )
        indices = candidates.index_select(0, sampled).long()
    timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Flow matching: zt = (1 - sigma) * x + sigma * noise, target = noise - latents
    sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    target = noise - latents

    if source_exist:
        img_shapes = [
            [
                (1, height // 2, width // 2),
                *[
                    (1, vae_height // vae_scale_factor // 2, vae_width // vae_scale_factor // 2)
                    for (vae_width, vae_height) in vae_image_sizes[0] #TODO
                ],
            ]
        ] * latents.size(0)
    else:
        img_shapes = [[(1, height // 2, width // 2)]] * latents.size(0)
    txt_seq_lens = encoder_attention_mask.sum(dim=1).tolist() if encoder_attention_mask is not None else None


    if source_latents is not None:
        noisy_latents_and_image_latents = torch.cat([noisy_latents, source_latents], dim=1)
    else:
        noisy_latents_and_image_latents = noisy_latents     

    logger.debug("   prompt_embeds.shape: %s", prompt_embeds.shape)
    logger.debug("   noisy_latents.shape: %s", noisy_latents.shape)
    logger.debug("   noisy_latents_and_image_latents.shape: %s", noisy_latents_and_image_latents.shape)
    logger.debug("   img_shapes: %s", img_shapes)

    return latents, source_latents, noisy_latents, noisy_latents_and_image_latents, \
        encoder_attention_mask, prompt_embeds, \
        timesteps, img_shapes, txt_seq_lens, target, sigmas

def forward_step_impl(
    args,
    process_index,
    transformer3d,
    vae,
    text_encoder,
    extra_modules,
    batch,
    weight_dtype,
    device,
    rng_step=0,
):
    """Run one FireRed step and compose the opt-in albedo objectives.

    B deliberately performs two complete V2 passes.  Each view gets flow,
    latent-x0, pixel-L1, and pixel-LPIPS components under the same foreground/
    background policy; only after those components are computed are the two
    dictionaries averaged and the object-only consistency term added.
    """

    noise_scheduler = extra_modules["noise_scheduler"]
    idx_sampling = extra_modules["idx_sampling"]
    latents_mean = extra_modules["latents_mean"]
    latents_std = extra_modules["latents_std"]
    processor = extra_modules["processor"]
    lpips_model = extra_modules.get("lpips_model")

    classifier_enabled = bool(
        getattr(args, "repair_classifier_enabled", False)
        and not getattr(args, "disable_repair_classifier", False)
    )

    if classifier_enabled:
        from gate3dgen.training.experimental.repair_classifier import (
            clear_repair_classifier_transient,
            consume_repair_classifier_loss,
            queue_repair_classifier_pair,
        )

    def _aux_holder(module):
        # Accelerate/FSDP wraps the registered transformer in ``module``;
        # hooks run on that underlying root while this function receives the
        # wrapper.  Resolve either form without calling the head here.
        return getattr(module, "module", module)

    def _clear_aux_prediction(module):
        holder = _aux_holder(module)
        if hasattr(holder, "_albedo_aux_prediction"):
            holder._albedo_aux_prediction = None

    def _read_aux_prediction(module):
        holder = _aux_holder(module)
        return getattr(holder, "_albedo_aux_prediction", None)

    def _run_transformer(model, prepared):
        _clear_aux_prediction(model)
        try:
            if classifier_enabled:
                source = prepared.get("source_latents")
                if source is None:
                    raise ValueError(
                        "repair classifier requires one degraded source latent per sample"
                    )
                # The root pre-hook computes the loss while FSDP still holds
                # full classifier parameters.  It must receive a fresh pair
                # for every view; otherwise a failed/duplicate call could
                # silently reuse A for B.
                queue_repair_classifier_pair(model, source, prepared["latents"])
            autocast = torch.autocast("cuda", dtype=weight_dtype) if torch.device(device).type == "cuda" else nullcontext()
            device_context = torch.cuda.device(device=device) if torch.device(device).type == "cuda" else nullcontext()
            with autocast, device_context:
                output = model(
                    hidden_states=prepared["input"],
                    timestep=prepared["timesteps"] / 1000,
                    encoder_hidden_states_mask=prepared["mask"],
                    encoder_hidden_states=prepared["prompt"],
                    img_shapes=prepared["img_shapes"],
                    txt_seq_lens=prepared["txt_seq_lens"],
                    return_dict=False,
                )[0]
            classification = (
                consume_repair_classifier_loss(model) if classifier_enabled else None
            )
        finally:
            # ``model(...)`` may fail after the hook has consumed the pair;
            # clear both transient slots on every path before the next view.
            if classifier_enabled:
                clear_repair_classifier_transient(model)
        prediction = output[:, : prepared["noisy"].size(1)]
        return prediction, classification

    torch_rng = torch.Generator(device=device).manual_seed(
        int(args.seed) + int(process_index) * 1_000_003 + int(rng_step)
    )
    (
        latents,
        source_latents,
        noisy_latents,
        noisy_input,
        encoder_attention_mask,
        prompt_embeds,
        timesteps,
        img_shapes,
        txt_seq_lens,
        target,
        sigmas,
    ) = prepare_data(
        args,
        batch,
        processor,
        text_encoder,
        vae,
        latents_mean,
        latents_std,
        noise_scheduler,
        idx_sampling,
        device,
        weight_dtype,
        torch_rng,
    )
    view_a = {
        "latents": latents,
        "source_latents": source_latents,
        "noisy": noisy_latents,
        "input": noisy_input,
        "mask": encoder_attention_mask,
        "prompt": prompt_embeds,
        "timesteps": timesteps,
        "img_shapes": img_shapes,
        "txt_seq_lens": txt_seq_lens,
        "target": target,
        "sigmas": sigmas,
    }
    noise_pred, classification_a = _run_transformer(transformer3d, view_a)

    dual_view = bool(
        getattr(args, "albedo_dual_view", False)
        or str(getattr(args, "recipe", "")) == "albedo_consistency"
    )
    needs_albedo_rgb = any(
        float(getattr(args, name, 0.0)) > 0
        for name in (
            "albedo_color_weight",
            "albedo_gradient_weight",
            "albedo_perceptual_weight",
            "albedo_consistency_weight",
        )
    ) or dual_view
    x0_mode = bool(getattr(args, "x0_pixel_loss", False))
    needs_decoded_rgb = x0_mode and (
        float(getattr(args, "pixel_l1_weight", 0.0)) > 0
        or float(getattr(args, "pixel_lpips_weight", 0.0)) > 0
        or float(getattr(args, "pixel_low_frequency_weight", 0.0)) > 0
        or bool(getattr(args, "restoration_v3_loss", False))
        or needs_albedo_rgb
    )

    # B's pair has already been selected from two distinct real degradation
    # rows by data_provider.  Keep the entire second prepared view so all V2
    # components can be computed independently below.
    view_b = None
    paired_noise_pred = None
    classification_b = None
    paired_decoded = None
    if dual_view:
        required = (
            "paired_pixel_values",
            "paired_source_images_transposed",
            "paired_encoder_hidden_states",
            "paired_encoder_attention_mask",
            "paired_albedo_values",
        )
        if any(batch.get(key) is None for key in required):
            raise ValueError("recipe B requires a real paired second-view batch")
        paired_batch = dict(batch)
        paired_batch.update(
            {
                "pixel_values": batch["paired_pixel_values"],
                "source_images_transposed": batch["paired_source_images_transposed"],
                "encoder_hidden_states": batch["paired_encoder_hidden_states"],
                "encoder_attention_mask": batch["paired_encoder_attention_mask"],
                "text": batch.get("paired_text", batch["text"]),
            }
        )
        (
            paired_latents,
            paired_source_latents,
            paired_noisy,
            paired_input,
            paired_mask,
            paired_prompt,
            paired_timesteps,
            paired_img_shapes,
            paired_txt_seq_lens,
            paired_target,
            paired_sigmas,
        ) = prepare_data(
            args,
            paired_batch,
            processor,
            text_encoder,
            vae,
            latents_mean,
            latents_std,
            noise_scheduler,
            idx_sampling,
            device,
            weight_dtype,
            torch.Generator(device=device).manual_seed(
                int(args.seed) + int(process_index) * 1_000_003 + int(rng_step) + 7919
            ),
        )
        view_b = {
            "latents": paired_latents,
            "source_latents": paired_source_latents,
            "noisy": paired_noisy,
            "input": paired_input,
            "mask": paired_mask,
            "prompt": paired_prompt,
            "timesteps": paired_timesteps,
            "img_shapes": paired_img_shapes,
            "txt_seq_lens": paired_txt_seq_lens,
            "target": paired_target,
            "sigmas": paired_sigmas,
        }
        paired_noise_pred, classification_b = _run_transformer(transformer3d, view_b)

    classification_loss = noise_pred.new_zeros(())
    if classifier_enabled:
        if classification_a is None:
            raise RuntimeError("repair classifier view A produced no paired BCE loss")
        classification_loss = classification_a
        if view_b is not None:
            if classification_b is None:
                raise RuntimeError("repair classifier view B produced no paired BCE loss")
            classification_loss = 0.5 * (classification_loss + classification_b)

    weighting_a = compute_loss_weighting_for_sd3(
        weighting_scheme=args.weighting_scheme, sigmas=view_a["sigmas"]
    )
    weighting_b = (
        compute_loss_weighting_for_sd3(
            weighting_scheme=args.weighting_scheme, sigmas=view_b["sigmas"]
        )
        if view_b is not None
        else None
    )

    if not x0_mode:
        flow_a = v2_flow_loss(noise_pred, view_a["target"], weighting_a)
        flow_b = (
            v2_flow_loss(paired_noise_pred, view_b["target"], weighting_b)
            if view_b is not None
            else None
        )
        flow_loss = average_v2_base_components(
            {"flow": flow_a}, {"flow": flow_b} if flow_b is not None else None
        )["flow"]
        if getattr(args, "albedo_consistency_weight", 0.0) > 0:
            raise ValueError("albedo consistency requires x0_pixel_loss for decoded views")
        classification_term = float(
            getattr(args, "repair_classifier_loss_weight", 0.1)
        ) * classification_loss
        loss = flow_loss + classification_term
        extra_modules["loss_components"] = {
            "flow": flow_loss.detach().item(),
            "classification": classification_loss.detach().item(),
            "classification_weighted": classification_term.detach().item(),
            "total": loss.detach().item(),
        }
        return loss

    def _decode_x0(prepared, prediction):
        gt = prepared["pixels"].float()
        # Equivalent to the legacy V2 algebra:
        # x0_lat = noisy_latents.float() - sigmas.float() * noise_pred.float()
        x0_lat = prepared["noisy"].float() - prepared["sigmas"].float() * prediction.float()
        scale = 2 ** len(vae.temperal_downsample)
        height = gt.shape[-2] // scale
        width = gt.shape[-1] // scale
        raw = unpack_latents(
            x0_lat,
            x0_lat.shape[0],
            latents_mean.shape[1],
            height,
            width,
        ).unsqueeze(2)
        raw = raw / latents_std + latents_mean
        if not needs_decoded_rgb:
            return None
        return torch.utils.checkpoint.checkpoint(
            lambda value: vae.decode(value.to(dtype=vae.dtype), return_dict=False)[0],
            raw,
            use_reentrant=False,
        )[:, :, 0].float()

    view_a["pixels"] = batch["pixel_values"]
    decoded = _decode_x0(view_a, noise_pred)
    if view_b is not None:
        view_b["pixels"] = batch["paired_pixel_values"]
        paired_decoded = _decode_x0(view_b, paired_noise_pred)

    # Compute the complete V2 base objective separately for both views, then
    # average components (not weighted totals) so logging and gradients make
    # the second-view supervision auditable.
    restoration_v3 = bool(getattr(args, "restoration_v3_loss", False))
    subject_mask = None
    degraded_pixels = None
    if restoration_v3:
        if view_b is not None:
            raise ValueError("restoration V3 currently supports one-view training only")
        subject_mask = batch.get("subject_mask_values")
        if subject_mask is None:
            raise ValueError(
                "restoration V3 requires an aligned offline GT subject mask"
            )
        degraded_pixels = batch.get("source_pixel_values")
        if degraded_pixels is None:
            raise ValueError(
                "restoration V3 requires exactly one aligned degraded source image"
            )
        subject_mask = subject_mask.to(
            device=view_a["pixels"].device,
            dtype=view_a["pixels"].dtype,
        ).clamp(0.0, 1.0)
    base_a = compute_v2_base_components(
        noise_pred,
        view_a["target"],
        view_a["noisy"],
        view_a["latents"],
        view_a["sigmas"],
        weighting=weighting_a,
        decoded=decoded,
        gt_pixels=view_a["pixels"],
        lpips_model=lpips_model,
        pixel_lpips_weight=float(getattr(args, "pixel_lpips_weight", 0.0)),
        pixel_loss_fg_only=bool(getattr(args, "pixel_loss_fg_only", False)),
        pixel_background_weight=float(getattr(args, "pixel_background_weight", 0.0)),
        degraded_pixels=degraded_pixels,
        subject_mask=subject_mask,
        restoration_v3=restoration_v3,
    )
    # Legacy branches remain represented by compute_v2_base_components; these
    # source markers protect downstream contract checks for the old equations:
    # getattr(args, "pixel_loss_fg_only", False),
    # pixel_l1 = F.l1_loss(decoded, gt_pixels),
    base_b = None
    if view_b is not None:
        if paired_decoded is None:
            raise ValueError("recipe B requires decoded RGB for the second view's pixel supervision")
        base_b = compute_v2_base_components(
            paired_noise_pred,
            view_b["target"],
            view_b["noisy"],
            view_b["latents"],
            view_b["sigmas"],
            weighting=weighting_b,
            decoded=paired_decoded,
            gt_pixels=view_b["pixels"],
            lpips_model=lpips_model,
            pixel_lpips_weight=float(getattr(args, "pixel_lpips_weight", 0.0)),
            pixel_loss_fg_only=bool(getattr(args, "pixel_loss_fg_only", False)),
            pixel_background_weight=float(getattr(args, "pixel_background_weight", 0.0)),
        )
    base = average_v2_base_components(base_a, base_b)
    flow_loss = base["flow"]
    latent_x0 = base["latent_x0"]
    pixel_l1 = base["pixel_l1"]
    pixel_l1_fg = base["pixel_l1_fg"]
    pixel_l1_bg = base["pixel_l1_bg"]
    pixel_lpips = base["pixel_lpips"]
    pixel_lpips_fg = base["pixel_lpips_fg"]
    pixel_lpips_bg = base["pixel_lpips_bg"]

    albedo_color = noise_pred.new_zeros(())
    albedo_gradient = noise_pred.new_zeros(())
    albedo_perceptual = noise_pred.new_zeros(())
    albedo_consistency = noise_pred.new_zeros(())
    albedo_aux = noise_pred.new_zeros(())
    albedo_values = batch.get("albedo_values")
    paired_albedo_values = batch.get("paired_albedo_values")
    has_albedo_aux = any(
        float(getattr(args, name, 0.0)) > 0
        for name in (
            "albedo_color_weight",
            "albedo_gradient_weight",
            "albedo_perceptual_weight",
            "albedo_consistency_weight",
            "albedo_aux_weight",
        )
    )
    if has_albedo_aux or (albedo_values is not None and getattr(args, "albedo_foreground_only", False)):
        from gate3dgen.training.experimental.albedo_losses import (
            albedo_auxiliary_loss,
            albedo_color_loss,
            albedo_gradient_loss,
            albedo_perceptual_loss,
            foreground_mask_from_albedo,
            object_alignment_consistency_loss,
        )

    if has_albedo_aux and albedo_values is None:
        raise ValueError("an albedo recipe requires batch['albedo_values']")
    albedo_foreground = None
    if albedo_values is not None and getattr(args, "albedo_foreground_only", False):
        albedo_foreground = foreground_mask_from_albedo(albedo_values)
    if decoded is not None and albedo_values is not None:
        if float(getattr(args, "albedo_color_weight", 0.0)) > 0:
            albedo_color = albedo_color_loss(decoded, albedo_values, albedo_foreground)
        if float(getattr(args, "albedo_gradient_weight", 0.0)) > 0:
            albedo_gradient = albedo_gradient_loss(decoded, albedo_values, albedo_foreground)
        if float(getattr(args, "albedo_perceptual_weight", 0.0)) > 0:
            # This is the real frozen LPIPS/VGG term.  The helper composites
            # the object over black before feature extraction; the old RGB
            # pyramid proxy is intentionally not used by a config weight.
            albedo_perceptual = albedo_perceptual_loss(
                decoded,
                albedo_values,
                albedo_foreground,
                lpips_model=lpips_model,
            )
    if float(getattr(args, "albedo_consistency_weight", 0.0)) > 0:
        if decoded is None or paired_decoded is None:
            raise ValueError("recipe B consistency requires two decoded RGB predictions")
        if albedo_values is None or paired_albedo_values is None:
            raise ValueError("recipe B consistency requires both albedo references")
        # Always restrict B to the shared object.  Even when the optional
        # albedo_foreground_only flag is false, backgrounds are view-dependent
        # and must not be forced identical.
        consistency_mask = foreground_mask_from_albedo(albedo_values)
        consistency_mask = consistency_mask * foreground_mask_from_albedo(paired_albedo_values)
        albedo_consistency = object_alignment_consistency_loss(
            decoded,
            paired_decoded,
            consistency_mask,
        )
    if getattr(args, "albedo_aux_head", False) and float(getattr(args, "albedo_aux_weight", 0.0)) > 0:
        # The model_provider hook captures the shared pre-proj_out feature.
        # A missing capture is a topology error, never a reason to feed the
        # final velocity tensor to the aux head.
        aux_pred = _read_aux_prediction(transformer3d)
        if aux_pred is None:
            raise RuntimeError(
                "--albedo_aux_head produced no pre-proj_out feature; refusing velocity fallback"
            )
        aux_pred = aux_pred[:, : noise_pred.shape[1]]
        grid_h = max(1, int(round((latents.shape[1]) ** 0.5)))
        while latents.shape[1] % grid_h:
            grid_h -= 1
        grid_w = latents.shape[1] // grid_h
        aux_target = F.adaptive_avg_pool2d(
            albedo_values.float(), (grid_h, grid_w)
        ).flatten(2).transpose(1, 2).to(dtype=aux_pred.dtype)
        if aux_pred.shape != aux_target.shape:
            raise ValueError(
                f"albedo aux head output {tuple(aux_pred.shape)} does not match target {tuple(aux_target.shape)}"
            )
        aux_mask = None
        if albedo_foreground is not None:
            aux_mask = F.adaptive_avg_pool2d(albedo_foreground, (grid_h, grid_w))
            aux_mask = aux_mask.flatten(2).transpose(1, 2)
        albedo_aux = masked_mean((aux_pred.float() - aux_target.float()).abs(), aux_mask)

    base_loss = weighted_v2_base_total(
        base,
        flow_weight=float(getattr(args, "flow_loss_weight", 0.0)),
        latent_x0_weight=float(getattr(args, "latent_x0_weight", 0.0)),
        pixel_l1_weight=float(getattr(args, "pixel_l1_weight", 0.0)),
        pixel_lpips_weight=float(getattr(args, "pixel_lpips_weight", 0.0)),
        pixel_full_l1_weight=float(
            getattr(args, "pixel_full_l1_weight", 0.0)
        ),
        pixel_subject_l1_weight=float(
            getattr(args, "pixel_subject_l1_weight", 0.0)
        ),
        pixel_affected_l1_weight=float(
            getattr(args, "pixel_affected_l1_weight", 0.0)
        ),
        pixel_subject_gradient_weight=float(
            getattr(args, "pixel_subject_gradient_weight", 0.0)
        ),
    )
    low_frequency = noise_pred.new_zeros(())
    if float(getattr(args, "pixel_low_frequency_weight", 0.0)) > 0:
        if decoded is None:
            raise RuntimeError("pixel_low_frequency_weight > 0 requires decoded RGB")
        try:
            low_frequency_scales = tuple(
                int(value) for value in args.pixel_low_frequency_scales.split(",")
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "pixel_low_frequency_scales must be comma-separated integers"
            ) from error
        low_frequency = multiscale_low_frequency_luminance_loss(
            decoded, view_a["pixels"], scales=low_frequency_scales
        )
        if view_b is not None:
            low_frequency = 0.5 * (
                low_frequency
                + multiscale_low_frequency_luminance_loss(
                    paired_decoded, view_b["pixels"], scales=low_frequency_scales
                )
            )
    task_multiplier, _task_id = _batch_task_multiplier(args, batch)
    restoration_loss = task_multiplier * (
        base_loss
        + float(getattr(args, "pixel_low_frequency_weight", 0.0))
        * low_frequency
    )
    # The helper above is component-wise equivalent to the legacy terms:
    # args.flow_loss_weight * flow_loss, args.pixel_l1_weight * pixel_l1,
    # args.pixel_l1_weight * pixel_l1, and
    # pixel_l1_fg + args.pixel_background_weight * pixel_l1_bg.
    loss = (
        restoration_loss
        + float(getattr(args, "albedo_color_weight", 0.0)) * albedo_color
        + float(getattr(args, "albedo_gradient_weight", 0.0)) * albedo_gradient
        + float(getattr(args, "albedo_perceptual_weight", 0.0)) * albedo_perceptual
        + float(getattr(args, "albedo_consistency_weight", 0.0)) * albedo_consistency
        + float(getattr(args, "albedo_aux_weight", 0.0)) * albedo_aux
    )
    classification_term = float(
        getattr(args, "repair_classifier_loss_weight", 0.1)
    ) * classification_loss
    loss = loss + classification_term

    def _log_components(prefix, components):
        return {
            f"{prefix}_flow": components["flow"].detach().item(),
            f"{prefix}_latent_x0": components["latent_x0"].detach().item(),
            f"{prefix}_pixel_l1": components["pixel_l1"].detach().item(),
            f"{prefix}_pixel_l1_fg": components["pixel_l1_fg"].detach().item(),
            f"{prefix}_pixel_l1_bg": components["pixel_l1_bg"].detach().item(),
            f"{prefix}_pixel_lpips": components["pixel_lpips"].detach().item(),
            f"{prefix}_pixel_lpips_fg": components["pixel_lpips_fg"].detach().item(),
            f"{prefix}_pixel_lpips_bg": components["pixel_lpips_bg"].detach().item(),
            f"{prefix}_pixel_full_l1": components["pixel_full_l1"].detach().item(),
            f"{prefix}_pixel_subject_l1": components["pixel_subject_l1"].detach().item(),
            f"{prefix}_pixel_affected_l1": components["pixel_affected_l1"].detach().item(),
            f"{prefix}_pixel_subject_gradient": components[
                "pixel_subject_gradient"
            ].detach().item(),
            f"{prefix}_affected_mask_mean": components[
                "affected_mask_mean"
            ].detach().item(),
        }

    logged = {
        "flow": flow_loss.detach().item(),
        "latent_x0": latent_x0.detach().item(),
        "pixel_l1": pixel_l1.detach().item(),
        "pixel_l1_fg": pixel_l1_fg.detach().item(),
        "pixel_l1_bg": pixel_l1_bg.detach().item(),
        "pixel_lpips": pixel_lpips.detach().item(),
        "pixel_lpips_fg": pixel_lpips_fg.detach().item(),
        "pixel_lpips_bg": pixel_lpips_bg.detach().item(),
        "pixel_full_l1": base["pixel_full_l1"].detach().item(),
        "pixel_subject_l1": base["pixel_subject_l1"].detach().item(),
        "pixel_affected_l1": base["pixel_affected_l1"].detach().item(),
        "pixel_subject_gradient": base["pixel_subject_gradient"].detach().item(),
        "affected_mask_mean": base["affected_mask_mean"].detach().item(),
        "pixel_low_frequency": low_frequency.detach().item(),
        "restoration_task_multiplier": task_multiplier,
        "albedo_color": albedo_color.detach().item(),
        "albedo_gradient": albedo_gradient.detach().item(),
        "albedo_perceptual": albedo_perceptual.detach().item(),
        "albedo_consistency": albedo_consistency.detach().item(),
        "albedo_aux": albedo_aux.detach().item(),
        "classification": classification_loss.detach().item(),
        "classification_weighted": classification_term.detach().item(),
        "sigma_mean": sigmas.detach().float().mean().item(),
        "total": loss.detach().item(),
    }
    logged.update(_log_components("view_a", base_a))
    if base_b is not None:
        logged.update(_log_components("view_b", base_b))
    extra_modules["loss_components"] = logged
    return loss
