"""Lightweight degraded-vs-clean gate over frozen VAE latents.

The restoration model is intentionally not asked to infer a degradation label
from its prompt.  Instead, this module provides a small, independently
registered head that consumes the *source* (degraded) and *target* (GT) VAE
latents produced by the existing frozen VAE.  A training pair contributes two
examples to one BCE objective: ``source -> 1`` and ``target -> 0``.

Keeping the head as a child of the transformer makes it visible to Accelerate,
FSDP, optimizer construction, and regular transformer checkpoint state while
the head itself remains only a few thousand parameters.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


from ..latent_mlp import LatentRouterMLP as LatentRepairClassifier, _latent_tokens


REPAIR_CLASSIFIER_PREFIX = "repair_classifier."

# Keep the suffix list independent of the default 64/128 dimensions.  The
# current trained head uses those dimensions, but recovery should also work
# for a small test head or a future architecture revision whose malformed
# sidecar still records different ``input_dim``/``hidden_dim`` values.
_REPAIR_CLASSIFIER_STATE_KEYS = frozenset(
    {
        "net.0.weight",
        "net.0.bias",
        "net.1.weight",
        "net.1.bias",
        "net.4.weight",
        "net.4.bias",
    }
)

# These attributes intentionally live outside ``state_dict``.  The classifier
# is a child of the Transformer so FSDP can shard and checkpoint it, but its
# paired latent inputs/loss are per-forward ephemeral values.  Keeping them on
# the unwrapped root lets the root pre-hook consume them while FSDP has full
# parameter views, before the root reshares the child parameters.
_REPAIR_CLASSIFIER_PENDING_PAIR = "_repair_classifier_pending_pair"
_REPAIR_CLASSIFIER_LOSS = "_repair_classifier_loss"
_REPAIR_CLASSIFIER_PREHOOK_HANDLE = "_repair_classifier_prehook_handle"








# Friendly name for callers/tests that use the shorter terminology.
RepairClassifier = LatentRepairClassifier


def register_repair_classifier_head(
    backbone: nn.Module,
    *,
    input_dim: int,
    hidden_dim: int = 128,
    dropout: float = 0.0,
) -> nn.Module:
    """Attach a classifier to a backbone before distributed wrapping."""

    if hasattr(backbone, "repair_classifier"):
        raise ValueError("repair_classifier is already registered")
    backbone.add_module(
        "repair_classifier",
        LatentRepairClassifier(input_dim, hidden_dim=hidden_dim, dropout=dropout),
    )
    return backbone


def _find_repair_classifier_backbone(module: nn.Module) -> nn.Module | None:
    """Find the module that owns the registered head through wrappers.

    Accelerate/FSDP/PEFT may expose the original Transformer through one or
    more ``module``/``base_model``/``model`` containers.  Returning the owner
    (rather than just the head) is necessary because the transient forward
    state and the pre-hook must be attached to the same unwrapped root that
    FSDP invokes internally.
    """

    candidates: list[Any] = [module]
    seen: set[int] = set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        # Read the direct child table instead of ``getattr``.  FSDP forwards
        # unknown attributes from its ``_fsdp_wrapped_module``, so looking up
        # ``candidate.repair_classifier`` first would incorrectly identify the
        # FSDP wrapper as the owner and place the transient pair on the wrong
        # object.
        head = getattr(candidate, "_modules", {}).get("repair_classifier")
        if isinstance(head, LatentRepairClassifier):
            return candidate
        for name in ("_fsdp_wrapped_module", "module", "base_model", "model"):
            child = getattr(candidate, "_modules", {}).get(name)
            if child is None:
                # A few container implementations expose the wrapped model
                # as a property rather than a registered child module.
                child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                candidates.append(child)
    return None


def get_repair_classifier(module: nn.Module) -> LatentRepairClassifier | None:
    """Find the registered head through common DDP/FSDP/PEFT wrappers."""
    backbone = _find_repair_classifier_backbone(module)
    return getattr(backbone, "repair_classifier", None) if backbone is not None else None


def install_repair_classifier_prehook(backbone: nn.Module) -> nn.Module:
    """Install the FSDP-safe paired-classification root pre-hook.

    The root Transformer is unsharded by FSDP immediately before it invokes
    the underlying module's forward hooks.  Computing the paired BCE here
    therefore observes complete classifier parameters.  Calling the head
    after ``transformer3d(...)`` is unsafe: the root has already resharded its
    child parameters and a LayerNorm/Linear can see a zero-length shard.

    A caller must first inject one pair with
    :func:`queue_repair_classifier_pair`.  The hook consumes that pair exactly
    once, stores the loss as a transient value, and fails closed when a root
    forward is attempted without a pair.  No transient attribute enters the
    model state dict.
    """

    if getattr(backbone, _REPAIR_CLASSIFIER_PREHOOK_HANDLE, None) is not None:
        return backbone
    classifier = getattr(backbone, "repair_classifier", None)
    if not isinstance(classifier, LatentRepairClassifier):
        raise ValueError(
            "repair classifier must be registered before installing its pre-hook"
        )

    def _compute_paired_loss(
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        pair = getattr(backbone, _REPAIR_CLASSIFIER_PENDING_PAIR, None)
        # Consume the pending pair before doing any work.  If classification
        # or the wrapped forward raises, no input may be reused by a later
        # view/step.
        setattr(backbone, _REPAIR_CLASSIFIER_PENDING_PAIR, None)
        setattr(backbone, _REPAIR_CLASSIFIER_LOSS, None)
        if pair is None:
            raise RuntimeError(
                "repair classifier root pre-hook received no transient "
                "degraded/clean latent pair; refusing to run without labels"
            )
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise RuntimeError(
                "repair classifier transient pair must be "
                "(degraded_latents, clean_latents)"
            )
        degraded_latents, clean_latents = pair
        if not isinstance(degraded_latents, torch.Tensor) or not isinstance(
            clean_latents, torch.Tensor
        ):
            raise TypeError(
                "repair classifier transient pair must contain torch.Tensor latents"
            )
        try:
            loss = paired_bce_loss(classifier, degraded_latents, clean_latents)
        except BaseException:
            # Do not leave a graph or stale inputs attached to a long-lived
            # model if classifier evaluation itself fails.
            setattr(backbone, _REPAIR_CLASSIFIER_PENDING_PAIR, None)
            setattr(backbone, _REPAIR_CLASSIFIER_LOSS, None)
            raise
        setattr(backbone, _REPAIR_CLASSIFIER_LOSS, loss)

    handle = backbone.register_forward_pre_hook(_compute_paired_loss, with_kwargs=True)
    setattr(backbone, _REPAIR_CLASSIFIER_PREHOOK_HANDLE, handle)
    setattr(backbone, _REPAIR_CLASSIFIER_PENDING_PAIR, None)
    setattr(backbone, _REPAIR_CLASSIFIER_LOSS, None)
    return backbone


def queue_repair_classifier_pair(
    module: nn.Module,
    degraded_latents: torch.Tensor,
    clean_latents: torch.Tensor,
) -> nn.Module:
    """Queue one source/GT latent pair for the next root Transformer call."""

    if not isinstance(degraded_latents, torch.Tensor) or not isinstance(
        clean_latents, torch.Tensor
    ):
        raise TypeError("repair classifier pair inputs must be torch.Tensor values")
    backbone = _find_repair_classifier_backbone(module)
    if backbone is None:
        raise RuntimeError("repair classifier is enabled but its root is not registered")
    if getattr(backbone, _REPAIR_CLASSIFIER_PREHOOK_HANDLE, None) is None:
        raise RuntimeError("repair classifier root pre-hook is not installed")
    if getattr(backbone, _REPAIR_CLASSIFIER_PENDING_PAIR, None) is not None:
        raise RuntimeError("repair classifier already has an unconsumed transient pair")
    setattr(backbone, _REPAIR_CLASSIFIER_PENDING_PAIR, (degraded_latents, clean_latents))
    # A previous loss should never survive into a new view.  The normal path
    # consumes it immediately; this assignment also makes recovery from a
    # caller-side exception deterministic.
    setattr(backbone, _REPAIR_CLASSIFIER_LOSS, None)
    return backbone


def consume_repair_classifier_loss(module: nn.Module) -> torch.Tensor:
    """Return and clear the loss produced by the most recent root pre-hook."""

    backbone = _find_repair_classifier_backbone(module)
    if backbone is None:
        raise RuntimeError("repair classifier is enabled but its root is not registered")
    loss = getattr(backbone, _REPAIR_CLASSIFIER_LOSS, None)
    setattr(backbone, _REPAIR_CLASSIFIER_LOSS, None)
    setattr(backbone, _REPAIR_CLASSIFIER_PENDING_PAIR, None)
    if not isinstance(loss, torch.Tensor):
        raise RuntimeError(
            "repair classifier root forward produced no transient paired BCE loss"
        )
    return loss


def clear_repair_classifier_transient(module: nn.Module) -> None:
    """Clear pending pair/loss state, including exceptional forward paths."""

    backbone = _find_repair_classifier_backbone(module)
    if backbone is None:
        return
    setattr(backbone, _REPAIR_CLASSIFIER_PENDING_PAIR, None)
    setattr(backbone, _REPAIR_CLASSIFIER_LOSS, None)


def paired_labels(batch_size: int, *, device: torch.device | str, dtype=torch.float32) -> torch.Tensor:
    """Return labels in the fixed ``[degraded=1, GT=0]`` pair order."""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("paired classifier batch size must be positive")
    return torch.cat(
        [
            torch.ones(batch_size, device=device, dtype=dtype),
            torch.zeros(batch_size, device=device, dtype=dtype),
        ]
    )


def paired_bce_loss(
    classifier: nn.Module,
    degraded_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    *,
    return_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute paired BCE with source/degraded positives and GT negatives."""

    degraded_logits = classifier(degraded_latents).reshape(-1)
    clean_logits = classifier(clean_latents).reshape(-1)
    if degraded_logits.shape != clean_logits.shape:
        raise ValueError(
            "paired classifier inputs must have equal batch sizes: "
            f"{tuple(degraded_logits.shape)} vs {tuple(clean_logits.shape)}"
        )
    logits = torch.cat([degraded_logits, clean_logits], dim=0)
    labels = paired_labels(
        degraded_logits.numel(), device=logits.device, dtype=logits.dtype
    )
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    if return_logits:
        return loss, degraded_logits, clean_logits
    return loss


# Explicit aliases make the contract discoverable without coupling callers to
# one spelling of "paired BCE".
compute_paired_bce_loss = paired_bce_loss
paired_classification_loss = paired_bce_loss


def classifier_probability(logits: torch.Tensor) -> torch.Tensor:
    """Convert one-logit output to a repair probability."""

    return torch.sigmoid(logits)


def classifier_decision(
    logits: torch.Tensor, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(probability, repair_boolean)`` at a validated threshold."""

    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"classifier threshold must be in [0, 1], got {threshold}")
    probability = classifier_probability(logits)
    return probability, probability >= threshold


def _strip_common_roots(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Normalize roots emitted by FSDP/DDP/container state dictionaries."""

    normalized: OrderedDict[str, torch.Tensor] = OrderedDict(
        (str(key), value) for key, value in state_dict.items()
    )
    prefixes = ("_fsdp_wrapped_module.", "module.", "model.", "transformer.")
    while normalized:
        common = [prefix for prefix in prefixes if all(key.startswith(prefix) for key in normalized)]
        if not common:
            break
        prefix = max(common, key=len)
        stripped: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, value in normalized.items():
            key = key[len(prefix):]
            if not key or key in stripped:
                raise ValueError("repair classifier state-dict root normalization collision")
            stripped[key] = value
        normalized = stripped
    return normalized


def split_repair_classifier_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    prefix: str = REPAIR_CLASSIFIER_PREFIX,
) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    """Split exact classifier keys from a combined transformer state dict."""

    normalized = _strip_common_roots(state_dict)
    backbone: OrderedDict[str, torch.Tensor] = OrderedDict()
    classifier: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in normalized.items():
        if key.startswith(prefix):
            classifier[key[len(prefix):]] = value
        else:
            backbone[key] = value
    return backbone, classifier


def export_repair_backbone_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    prefix: str = REPAIR_CLASSIFIER_PREFIX,
) -> OrderedDict[str, torch.Tensor]:
    """Remove only classifier keys before loading a plain RGB transformer."""

    return split_repair_classifier_state_dict(state_dict, prefix=prefix)[0]


def _repair_classifier_shapes(input_dim: int, hidden_dim: int) -> dict[str, tuple[int, ...]]:
    """Return the exact sidecar tensor shapes for one classifier head."""

    input_dim, hidden_dim = int(input_dim), int(hidden_dim)
    return {
        "net.0.weight": (input_dim,),
        "net.0.bias": (input_dim,),
        "net.1.weight": (hidden_dim, input_dim),
        "net.1.bias": (hidden_dim,),
        "net.4.weight": (1, hidden_dim),
        "net.4.bias": (1,),
    }


def _validate_repair_classifier_payload(payload: Any, *, context: str) -> None:
    """Fail with an actionable message when a sidecar is a local FSDP shard.

    With ``use_orig_params=True`` a tiny auxiliary head can be entirely absent
    from rank 0's local shard.  PyTorch represents those parameters as
    one-dimensional zero-length tensors.  Treating that file as a complete
    sidecar produces the misleading ``size mismatch ... torch.Size([0])``
    error seen by inference.  Validate both keys and shapes before constructing
    the inference module so we can recover from the full FSDP state dict.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a mapping")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"{context} has no state_dict mapping")
    try:
        input_dim = int(payload["input_dim"])
        hidden_dim = int(payload["hidden_dim"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context} has invalid input_dim/hidden_dim") from error
    if input_dim <= 0 or hidden_dim <= 0:
        raise ValueError(f"{context} has non-positive input_dim/hidden_dim")
    expected = _repair_classifier_shapes(input_dim, hidden_dim)
    actual_keys = {str(key) for key in state}
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"{context} keys mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    for key, shape in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{context} {key} is not a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"{context} {key} has shape {tuple(value.shape)}, expected {shape}"
            )
        if value.numel() == 0:
            raise ValueError(f"{context} {key} is an empty FSDP shard")
        if value.is_floating_point() and not torch.isfinite(value.detach().float()).all():
            raise ValueError(f"{context} {key} contains non-finite values")


def _load_torch_object(path: Path) -> Any:
    """Load a PyTorch zip object, memory-mapping tensor payloads when possible.

    ``mmap=True`` is important for a 55 GB FSDP full state: the classifier
    recovery needs six small tensors, not a second in-memory copy of the 20B
    transformer.  The fallback keeps this helper usable on older torch builds.
    """

    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch
            state = torch.load(str(path), map_location="cpu")
    return state


def _load_torch_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    """Load a PyTorch zip checkpoint without materializing unrelated tensors.

    ``mmap=True`` is important for a 55 GB FSDP full state: the classifier
    recovery needs six small tensors, not a second in-memory copy of the 20B
    transformer.  The fallback keeps this helper usable on older torch builds.
    """

    state = _load_torch_object(path)
    if isinstance(state, Mapping) and isinstance(state.get("state_dict"), Mapping):
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError(f"full FSDP checkpoint is not a state-dict mapping: {path}")
    return state


def _recover_repair_classifier_payload(
    checkpoint_root: Path,
    *,
    sidecar_payload: Mapping[str, Any] | None,
    sidecar_error: Exception,
) -> dict[str, Any]:
    """Recover a complete head from ``pytorch_model_fsdp.bin``.

    Older 32-GPU runs wrote the sidecar after FSDP had re-sharded the model;
    the full model file, saved by Accelerate with ``FULL_STATE_DICT``, still
    contains the six complete ``repair_classifier.*`` tensors.  Extraction is
    prefix-tolerant for DDP/FSDP/container wrappers and clones only the tiny
    classifier tensors off the memory-mapped full state.
    """

    full_path = checkpoint_root / "pytorch_model_fsdp.bin"
    if not full_path.is_file():
        raise ValueError(
            f"invalid repair classifier sidecar ({sidecar_error}); full FSDP state is missing: {full_path}"
        )
    try:
        full_state = _load_torch_state_dict(full_path)
        recovered_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for full_key, value in full_state.items():
            key = str(full_key)
            marker = REPAIR_CLASSIFIER_PREFIX
            if marker not in key:
                continue
            suffix = key.rsplit(marker, 1)[-1]
            if suffix in _REPAIR_CLASSIFIER_STATE_KEYS:
                recovered_state[suffix] = value.detach().cpu().clone()
        if not recovered_state:
            raise ValueError("no repair_classifier.* tensors found in the full FSDP state")

        # Prefer metadata from the malformed sidecar when it is still useful;
        # otherwise infer dimensions from LayerNorm/Linear tensor shapes.
        metadata = sidecar_payload if isinstance(sidecar_payload, Mapping) else {}
        try:
            input_dim = int(metadata.get("input_dim", recovered_state["net.0.weight"].shape[0]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("cannot infer classifier input_dim from recovered state") from error
        try:
            hidden_dim = int(metadata.get("hidden_dim", recovered_state["net.1.weight"].shape[0]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("cannot infer classifier hidden_dim from recovered state") from error
        try:
            dropout = float(metadata.get("dropout", 0.0))
        except (TypeError, ValueError):
            dropout = 0.0
        payload = {
            "state_dict": recovered_state,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }
        _validate_repair_classifier_payload(payload, context=f"recovered classifier from {full_path}")
        return payload
    except Exception as error:
        raise ValueError(
            f"invalid repair classifier sidecar ({sidecar_error}); full-state recovery failed: {error}"
        ) from error


def load_repair_classifier_payload(
    checkpoint_root: Path,
    *,
    recover_from_full_state: bool = True,
) -> dict[str, Any]:
    """Load a complete classifier sidecar, recovering legacy FSDP shards.

    The normal path reads only ``repair_classifier.pt``.  If that sidecar was
    produced by the pre-fix FSDP save hook and contains zero-length tensors,
    the helper memory-maps the full model state and extracts the valid head.
    This keeps old checkpoints usable without loading the 20B RGB transformer
    for the classifier-only evaluation stage.
    """

    checkpoint_root = Path(checkpoint_root)
    sidecar = checkpoint_root / "repair_classifier.pt"
    sidecar_payload: Mapping[str, Any] | None = None
    sidecar_error: Exception | None = None
    if sidecar.is_file():
        try:
            sidecar_payload = _load_torch_object(sidecar)
            _validate_repair_classifier_payload(sidecar_payload, context=str(sidecar))
            return dict(sidecar_payload)
        except Exception as error:
            sidecar_error = error
    else:
        sidecar_error = FileNotFoundError(f"repair classifier checkpoint is missing: {sidecar}")

    if not recover_from_full_state:
        raise ValueError(str(sidecar_error))
    return _recover_repair_classifier_payload(
        checkpoint_root,
        sidecar_payload=sidecar_payload,
        sidecar_error=sidecar_error or ValueError("unknown sidecar error"),
    )


__all__ = [
    "REPAIR_CLASSIFIER_PREFIX",
    "LatentRepairClassifier",
    "RepairClassifier",
    "register_repair_classifier_head",
    "get_repair_classifier",
    "install_repair_classifier_prehook",
    "queue_repair_classifier_pair",
    "consume_repair_classifier_loss",
    "clear_repair_classifier_transient",
    "paired_labels",
    "paired_bce_loss",
    "compute_paired_bce_loss",
    "paired_classification_loss",
    "classifier_probability",
    "classifier_decision",
    "split_repair_classifier_state_dict",
    "export_repair_backbone_state_dict",
    "load_repair_classifier_payload",
]
