"""Strict common-root normalization shared by released and legacy checkpoint readers."""
from __future__ import annotations
from collections import OrderedDict
from typing import Any, Mapping


_STATE_DICT_ROOT_PREFIXES = (
    "_fsdp_wrapped_module.",
    "module.",
    "model.",
    "transformer.",
)


def _normalize_state_dict_root(
    state_dict: Mapping[str, Any],
    prefix: str,
) -> tuple[OrderedDict[str, Any], str]:
    """Normalize a common FSDP/container root without weakening strictness.

    A full state dict obtained from a wrapped module can contain keys such as
    ``_fsdp_wrapped_module.transformer_blocks.29.attn.to_q.weight``. The model
    itself has no such container, so strict loading would reject all keys
    unless the common root is removed.  We only strip a known root when *all*
    keys carry it, and reject collisions after normalization.  Consequently a
    typo or unrelated adapter key still reaches ``load_state_dict(...,
    strict=True)`` and fails closed.
    """

    normalized = OrderedDict((str(key), value) for key, value in state_dict.items())
    normalized_prefix = str(prefix)
    while normalized:
        candidates = [
            candidate
            for candidate in _STATE_DICT_ROOT_PREFIXES
            if all(key.startswith(candidate) for key in normalized)
        ]
        if not candidates:
            break
        root = max(candidates, key=len)
        next_state: OrderedDict[str, Any] = OrderedDict()
        for key, value in normalized.items():
            stripped = key[len(root):]
            if not stripped or stripped in next_state:
                raise ValueError(
                    f"state-dict root normalization produced a key collision for {key!r}"
                )
            next_state[stripped] = value
        normalized = next_state
        if normalized_prefix.startswith(root):
            normalized_prefix = normalized_prefix[len(root):]
    return normalized, normalized_prefix


def normalize_state_dict_root(state_dict: Mapping[str, Any]) -> OrderedDict[str, Any]:
    """Strip only a root common to every key; never discard auxiliary or unknown tensors."""
    normalized, _ = _normalize_state_dict_root(state_dict, "")
    return normalized
