"""Independent pooled-latent router MLP, with no purifier/FSDP integration."""
from __future__ import annotations
import torch
from torch import nn


def _latent_tokens(latents: torch.Tensor) -> torch.Tensor:
    """Normalize packed or spatial latent inputs to ``[batch, tokens, width]``.

    Training uses FireRed's packed ``[B, tokens, channels*4]`` representation.
    Accepting ``[B, C, H, W]`` as well keeps the head useful for CPU tests and
    inference utilities that naturally hold a spatial latent tensor.
    """

    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"latent input must be a torch.Tensor, got {type(latents)!r}")
    if latents.ndim == 3:
        return latents
    if latents.ndim == 4:
        # VAE-style spatial latent: [B, C, H, W] -> [B, H*W, C].
        return latents.flatten(2).transpose(1, 2)
    if latents.ndim == 5 and latents.shape[2] == 1:
        # Qwen VAE keeps a singleton temporal axis: [B, C, 1, H, W].
        return latents[:, :, 0].flatten(2).transpose(1, 2)
    raise ValueError(
        "repair classifier expects packed [B,T,C], spatial [B,C,H,W], or "
        f"singleton-temporal [B,C,1,H,W] latents; got {tuple(latents.shape)}"
    )


class LatentRouterMLP(nn.Module):
    """Mean-pooled latent MLP producing one degradation logit per sample."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        input_dim, hidden_dim = int(input_dim), int(hidden_dim)
        dropout = float(dropout)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("repair classifier dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("repair classifier dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        tokens = _latent_tokens(latents)
        if tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"repair classifier latent width {tokens.shape[-1]} does not match "
                f"input_dim={self.input_dim}"
            )
        pooled = tokens.float().mean(dim=1)
        # Keep the output in the head's parameter dtype for mixed precision,
        # while doing the reduction in fp32 for numerical stability.
        logits = self.net(pooled.to(dtype=next(self.parameters()).dtype))
        return logits.squeeze(-1)
