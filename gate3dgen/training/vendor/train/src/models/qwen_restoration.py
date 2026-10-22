"""Restoration-specific Qwen Image transformer with an optional FA3 backend.

This is the single-output migration of the custom PBR transformer.  It keeps
the stock Qwen Image parameter topology and forward contract: no modality
adapters, triptych canvas, or Layer3D position encoding is registered.  The
only runtime extension is a mask-correct Hopper FlashAttention-3 processor.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenImageTransformer2DModel,
    apply_rotary_emb_qwen,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers


logger = logging.getLogger(__name__)

ATTENTION_NATIVE = "native"
ATTENTION_FA3 = "fa3"
ATTENTION_AUTO = "auto"
ATTENTION_BACKENDS = (ATTENTION_NATIVE, ATTENTION_FA3, ATTENTION_AUTO)


@lru_cache(maxsize=1)
def _load_flash_attention_3_ops():
    try:
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "FlashAttention-3 requires the t2h flash_attn_interface extension"
        ) from error
    return flash_attn_func, flash_attn_varlen_func


@lru_cache(maxsize=16)
def validate_fa3_runtime(device: torch.device | str, dtype: torch.dtype) -> None:
    """Fail closed for explicit FA3; auto mode catches this and falls back."""

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FA3 image attention requires CUDA")
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"FA3 requires bfloat16 or float16, got {dtype}")
    capability = torch.cuda.get_device_capability(device)
    if capability < (9, 0):
        raise RuntimeError(
            "FA3 requires Hopper compute capability >= 9.0, "
            f"got {capability[0]}.{capability[1]}"
        )
    _load_flash_attention_3_ops()


def _mask_plan(attention_mask: torch.Tensor, query_seq_len: int) -> dict:
    if attention_mask.dtype != torch.bool or attention_mask.ndim != 2:
        raise ValueError("FA3 attention mask must be a 2D bool tensor")
    batch_size = attention_mask.shape[0]
    key_lengths = attention_mask.sum(dim=1, dtype=torch.int32)
    if not bool((key_lengths > 0).all().item()):
        raise ValueError("every FA3 sample must retain at least one key token")
    cu_seqlens_k = torch.zeros(
        batch_size + 1, dtype=torch.int32, device=attention_mask.device
    )
    cu_seqlens_k[1:] = key_lengths.cumsum(dim=0)
    cu_seqlens_q = torch.arange(
        0,
        (batch_size + 1) * query_seq_len,
        query_seq_len,
        dtype=torch.int32,
        device=attention_mask.device,
    )
    return {
        "key_indices": attention_mask.reshape(-1).nonzero(as_tuple=False).flatten(),
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "max_seqlen_q": int(query_seq_len),
        "max_seqlen_k": int(key_lengths.max().item()),
    }


def _flash_attention_3(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    validate_fa3_runtime(query.device, query.dtype)
    flash_attn_func, flash_attn_varlen_func = _load_flash_attention_3_ops()
    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
    if attention_mask is None:
        return flash_attn_func(
            query, key, value, softmax_scale=None, causal=False
        )

    batch_size, seq_len, heads, head_dim = query.shape
    if attention_mask.shape != (batch_size, seq_len):
        raise ValueError(
            f"FA3 mask {tuple(attention_mask.shape)} does not match "
            f"Q/K/V sequence {(batch_size, seq_len)}"
        )
    plan = _mask_plan(attention_mask, seq_len)
    flat_key = key.reshape(batch_size * seq_len, heads, head_dim)
    flat_value = value.reshape(batch_size * seq_len, heads, head_dim)
    packed_query = query.reshape(batch_size * seq_len, heads, head_dim)
    output = flash_attn_varlen_func(
        packed_query,
        flat_key.index_select(0, plan["key_indices"]),
        flat_value.index_select(0, plan["key_indices"]),
        plan["cu_seqlens_q"],
        plan["cu_seqlens_k"],
        plan["max_seqlen_q"],
        plan["max_seqlen_k"],
        softmax_scale=None,
        causal=False,
    )
    return output.reshape(batch_size, seq_len, heads, head_dim)


class QwenDoubleStreamAttnProcessorFA3:
    """Qwen text/image joint attention using mask-correct Hopper FA3."""

    _attention_backend = "_flash_3"
    _parallel_config = None

    def __init__(self) -> None:
        _load_flash_attention_3_ops()

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_hidden_states is None:
            raise ValueError("FA3 Qwen attention requires encoder_hidden_states")

        seq_txt = encoder_hidden_states.shape[1]
        img_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        img_key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        img_value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        txt_query = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        if attention_mask is None and encoder_hidden_states_mask is not None:
            text_mask = encoder_hidden_states_mask.to(
                device=joint_query.device, dtype=torch.bool
            )
            image_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=joint_query.device
            )
            attention_mask = torch.cat([text_mask, image_mask], dim=1)

        joint_output = _flash_attention_3(
            joint_query, joint_key, joint_value, attention_mask
        ).flatten(2, 3).to(joint_query.dtype)
        txt_output = attn.to_add_out(joint_output[:, :seq_txt].contiguous())
        img_output = attn.to_out[0](joint_output[:, seq_txt:].contiguous())
        if len(attn.to_out) > 1:
            img_output = attn.to_out[1](img_output)
        return img_output, txt_output


class QwenImageRestorationTransformer2DModel(QwenImageTransformer2DModel):
    """Stock-compatible Qwen transformer with optional restoration FA3."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes=None,
        txt_seq_lens=None,
        guidance: torch.Tensor = None,
        attention_kwargs=None,
        controlnet_block_samples=None,
        return_dict: bool = True,
    ):
        """Run the stock-compatible single-output restoration mmDiT."""

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )
        image_rotary_emb = self.pos_embed(
            img_shapes, txt_seq_lens, device=hidden_states.device
        )
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
            if controlnet_block_samples is not None:
                interval = max(
                    1,
                    (len(self.transformer_blocks) + len(controlnet_block_samples) - 1)
                    // len(controlnet_block_samples),
                )
                hidden_states = hidden_states + controlnet_block_samples[
                    index_block // interval
                ]

        hidden_states = self.norm_out(hidden_states, temb)
        sample = self.proj_out(hidden_states)
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if return_dict:
            return Transformer2DModelOutput(sample=sample)
        return (sample,)

    def set_restoration_attention_backend(
        self,
        backend: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> str:
        backend = str(backend)
        if backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"attention backend must be one of {ATTENTION_BACKENDS}, got {backend!r}"
            )
        selected = backend
        if backend == ATTENTION_AUTO:
            try:
                validate_fa3_runtime(device, dtype)
                selected = ATTENTION_FA3
            except RuntimeError as error:
                logger.warning("FA3 unavailable; falling back to native attention: %s", error)
                selected = ATTENTION_NATIVE
        if selected == ATTENTION_FA3:
            validate_fa3_runtime(device, dtype)
            factory = QwenDoubleStreamAttnProcessorFA3
        else:
            factory = QwenDoubleStreamAttnProcessor2_0
        for block in self.transformer_blocks:
            block.attn.set_processor(factory())
        self._restoration_attention_backend = selected
        return selected
