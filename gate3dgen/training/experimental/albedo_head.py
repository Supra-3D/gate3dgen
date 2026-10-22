"""Opt-in C RGB+albedo dual-head and safe checkpoint export helpers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

import torch
from torch import nn
from ..checkpoint_state import _normalize_state_dict_root


ALBEDO_AUX_PREFIX = "albedo_aux_head."

class AlbedoAuxHead(nn.Module):
    """A small RGB albedo regression head over transformer token features.

    The head is deliberately separate from the stock FireRed transformer.  It
    can be registered before FSDP wrapping and removed from exported RGB
    checkpoints without renaming or weakening backbone keys.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 3):
        super().__init__()
        input_dim, hidden_dim, output_dim = int(input_dim), int(hidden_dim), int(output_dim)
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("aux head dimensions must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        if token_features.ndim != 3:
            raise ValueError("AlbedoAuxHead expects [B, tokens, channels]")
        return self.net(token_features)


def register_albedo_aux_head(
    backbone: nn.Module,
    *,
    input_dim: int,
    hidden_dim: int = 256,
    output_dim: int = 3,
) -> nn.Module:
    """Register C's head before FSDP/Accelerate sees the model."""

    if hasattr(backbone, "albedo_aux_head"):
        raise ValueError("albedo_aux_head is already registered")
    head = AlbedoAuxHead(input_dim, hidden_dim, output_dim)
    backbone.add_module("albedo_aux_head", head)
    return backbone


def install_albedo_aux_prehook(backbone: nn.Module) -> nn.Module:
    """Attach C's head to the shared feature immediately before ``proj_out``.

    The RGB transformer output is a velocity/noise prediction and is therefore
    *not* a valid feature for an albedo head.  ``proj_out`` is the final RGB
    projection; its input is the shared hidden representation used by both
    objectives.  A ``forward_pre_hook`` lets the head run while the root
    transformer is still inside its FSDP forward, before parameters are
    resharded.  The prediction is kept as a transient tensor and does not
    change the transformer's RGB return value or state-dict topology.

    ``with_kwargs=True`` is used because transformer implementations differ in
    whether they invoke ``proj_out`` positionally or with ``hidden_states=``.
    """

    if getattr(backbone, "_albedo_aux_prehook_installed", False):
        return backbone
    aux_head = getattr(backbone, "albedo_aux_head", None)
    if aux_head is None:
        raise ValueError("albedo_aux_head must be registered before installing its proj_out hook")
    proj_out = getattr(backbone, "proj_out", None)
    if not isinstance(proj_out, nn.Module) or not hasattr(proj_out, "in_features"):
        raise ValueError("C auxiliary head requires transformer3d.proj_out.in_features")

    def _capture_shared_hidden(
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("proj_out pre-hook did not receive a hidden-state tensor")
        if hidden.ndim == 4:
            # A few transformer variants keep a spatial feature map until the
            # final projection.  Convert it to [B,t,C] without changing the
            # feature values seen by the RGB projection.
            hidden = hidden.flatten(2).transpose(1, 2)
        if hidden.ndim != 3:
            raise ValueError(
                f"C auxiliary head requires [B,t,C] shared features, got {tuple(hidden.shape)}"
            )
        expected = int(getattr(proj_out, "in_features"))
        if hidden.shape[-1] != expected:
            raise ValueError(
                f"proj_out hidden width {hidden.shape[-1]} differs from in_features={expected}"
            )
        # The hook is attached to proj_out, but use the root module to find the
        # registered head.  ``backbone`` is deliberately captured rather than
        # ``module`` because proj_out itself has no aux child.
        backbone._albedo_aux_prediction = aux_head(hidden)

    # ``with_kwargs`` is supported by the PyTorch versions used by the V2
    # launcher and avoids a fragile wrapper around the FSDP root forward.
    handle = proj_out.register_forward_pre_hook(_capture_shared_hidden, with_kwargs=True)
    backbone._albedo_aux_prehook_handle = handle
    backbone._albedo_aux_prehook_installed = True
    backbone._albedo_aux_prediction = None
    backbone._albedo_aux_proj_out = proj_out
    return backbone


def install_albedo_aux_forward(backbone: nn.Module) -> nn.Module:
    """Compatibility alias for callers using the old helper name.

    The implementation is intentionally the pre-``proj_out`` hook above; no
    caller should interpret this alias as running the head on final velocity.
    """

    return install_albedo_aux_prehook(backbone)





def split_aux_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    prefix: str = ALBEDO_AUX_PREFIX,
) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    """Split only the exact aux prefix from a combined state dictionary."""

    normalized, prefix = _normalize_state_dict_root(state_dict, prefix)
    backbone: OrderedDict[str, torch.Tensor] = OrderedDict()
    aux: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in normalized.items():
        if key.startswith(prefix):
            aux[key[len(prefix):]] = value
        else:
            backbone[key] = value
    return backbone, aux


def export_rgb_backbone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    prefix: str = ALBEDO_AUX_PREFIX,
) -> OrderedDict[str, torch.Tensor]:
    """Export C as a strict-loadable RGB backbone by removing aux keys only."""

    backbone, _ = split_aux_state_dict(state_dict, prefix=prefix)
    return backbone


def load_rgb_backbone_strict(
    backbone: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    prefix: str = ALBEDO_AUX_PREFIX,
) -> Any:
    """Load RGB weights strictly while tolerating C's exact aux prefix.

    Any unexpected key outside ``prefix`` remains an error.  This prevents a
    typo or an unrelated adapter from being silently discarded.
    """

    filtered = export_rgb_backbone_state_dict(state_dict, prefix=prefix)
    return backbone.load_state_dict(filtered, strict=True)


__all__ = [
    "ALBEDO_AUX_PREFIX",
    "AlbedoAuxHead",
    "register_albedo_aux_head",
    "install_albedo_aux_prehook",
    "install_albedo_aux_forward",
    "split_aux_state_dict",
    "export_rgb_backbone_state_dict",
    "load_rgb_backbone_strict",
]
