"""Small, dependency-light losses used by the opt-in albedo recipes.

The training job is intentionally still a normal FireRed RGB restoration job.
These helpers only add supervision when a recipe explicitly opts in.  Keeping
the operations here independent from Diffusers makes the data/loss contract
testable on CPU and, importantly, makes it hard to accidentally turn an
albedo experiment into a new model topology.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _rgb01(value: torch.Tensor) -> torch.Tensor:
    """Convert either RGB ``[0, 1]`` or FireRed ``[-1, 1]`` to ``[0, 1]``."""

    value = _as_nchw(value, "image")
    # Targets loaded from the albedo PNG path are [0, 1], while decoded VAE
    # outputs are [-1, 1].  The range test is deliberately detached: it is a
    # representation conversion, not a learnable branch.
    if bool(value.detach().amin() < 0):
        value = value * 0.5 + 0.5
    return value.clamp(0.0, 1.0)


def foreground_mask_from_albedo(
    albedo: torch.Tensor,
    *,
    threshold: float = 2.0 / 255.0,
    dilation: int = 3,
) -> torch.Tensor:
    """Infer a conservative object mask from an albedo image.

    Albedo supervision is about object appearance; applying a consistency
    penalty to every background pixel would make B learn camera/background
    identity instead of view-invariant object appearance.  The mask is based
    on luminance and a small max-pool dilation, matching the established
    FireRed foreground convention.
    """

    value = _rgb01(albedo)
    luminance = (
        0.299 * value[:, :1, ...]
        + 0.587 * value[:, 1:2, ...]
        + 0.114 * value[:, 2:3, ...]
        if value.shape[1] >= 3
        else value[:, :1, ...]
    )
    mask = (luminance > float(threshold)).to(value.dtype)
    dilation = int(dilation)
    if dilation < 0:
        raise ValueError("dilation must be non-negative")
    if dilation:
        kernel = 2 * dilation + 1
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
    return mask.clamp(0.0, 1.0)


def albedo_color_loss(
    prediction: torch.Tensor,
    albedo: torch.Tensor,
    object_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RGB color/reflectance L1 loss, optionally restricted to the object."""

    prediction = _rgb01(prediction)
    albedo = _rgb01(albedo)
    if albedo.shape[-2:] != prediction.shape[-2:]:
        albedo = F.interpolate(albedo, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
    return masked_mean((prediction - albedo).abs(), object_mask)


def albedo_gradient_loss(
    prediction: torch.Tensor,
    albedo: torch.Tensor,
    object_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Match horizontal and vertical albedo edges (a finite-difference loss)."""

    prediction = _rgb01(prediction)
    albedo = _rgb01(albedo)
    if albedo.shape[-2:] != prediction.shape[-2:]:
        albedo = F.interpolate(albedo, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
    pred_x, pred_y = _gradient(prediction)
    target_x, target_y = _gradient(albedo)
    mask_x = object_mask[..., :, 1:] if object_mask is not None else None
    mask_y = object_mask[..., 1:, :] if object_mask is not None else None
    return 0.5 * (
        masked_mean((pred_x - target_x).abs(), mask_x)
        + masked_mean((pred_y - target_y).abs(), mask_y)
    )


def albedo_multiscale_rgb_proxy(
    prediction: torch.Tensor,
    albedo: torch.Tensor,
    object_mask: Optional[torch.Tensor] = None,
    *,
    levels: int = 3,
) -> torch.Tensor:
    """Explicitly named legacy RGB proxy kept for non-production experiments."""

    levels = int(levels)
    if levels <= 0:
        raise ValueError("levels must be positive")
    prediction = _rgb01(prediction)
    albedo = _rgb01(albedo)
    if albedo.shape[-2:] != prediction.shape[-2:]:
        albedo = F.interpolate(albedo, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
    total = prediction.new_zeros(())
    current_pred, current_target = prediction, albedo
    current_mask = object_mask
    for _ in range(levels):
        total = total + albedo_color_loss(current_pred, current_target, current_mask)
        if min(current_pred.shape[-2:]) <= 1:
            break
        current_pred = F.avg_pool2d(current_pred, kernel_size=2, stride=2, ceil_mode=True)
        current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2, ceil_mode=True)
        if current_mask is not None:
            current_mask = F.avg_pool2d(current_mask, kernel_size=2, stride=2, ceil_mode=True)
    return total / float(levels)


def albedo_perceptual_loss(
    prediction: torch.Tensor,
    albedo: torch.Tensor,
    object_mask: Optional[torch.Tensor] = None,
    *,
    lpips_model: object = None,
    background_weight: float = 0.0,
) -> torch.Tensor:
    """True frozen LPIPS/VGG albedo perceptual loss.

    The production-facing name intentionally cannot silently fall back to the
    old multi-scale RGB proxy.  Callers that explicitly need that deterministic
    proxy must use :func:`albedo_multiscale_rgb_proxy` by name.
    """

    return masked_lpips_loss(
        lpips_model,
        prediction,
        albedo,
        object_mask,
        background_weight=background_weight,
    )


def object_alignment_consistency_loss(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    object_mask: torch.Tensor,
    *,
    reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Align two degradation views only on the shared foreground/object.

    ``object_mask`` is required by design.  There is no implicit full-image
    fallback, which prevents B from penalizing legitimate view-dependent
    backgrounds.  If a reference image is provided, it is resized to the
    prediction shape before deriving the mask intersection.
    """

    prediction_a = _rgb01(prediction_a)
    prediction_b = _rgb01(prediction_b)
    if prediction_a.shape != prediction_b.shape:
        raise ValueError("dual-view predictions must have the same shape")
    if object_mask is None:
        raise ValueError("object_mask is required; full-image consistency is not allowed")
    mask = _resize_mask(object_mask, prediction_a)
    assert mask is not None
    if reference is not None:
        reference_mask = foreground_mask_from_albedo(reference)
        reference_mask = _resize_mask(reference_mask, prediction_a)
        assert reference_mask is not None
        mask = mask * reference_mask
    return masked_mean((prediction_a - prediction_b).abs(), mask)


def albedo_auxiliary_loss(
    prediction: torch.Tensor,
    albedo: torch.Tensor,
    object_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Default C auxiliary-head target loss (RGB albedo regression)."""

    return albedo_color_loss(prediction, albedo, object_mask)


def compute_albedo_losses(
    prediction: torch.Tensor,
    albedo: torch.Tensor,
    object_mask: Optional[torch.Tensor] = None,
    *,
    color_weight: float = 0.0,
    gradient_weight: float = 0.0,
    perceptual_weight: float = 0.0,
    lpips_model: object = None,
) -> dict[str, torch.Tensor]:
    """Return A's named components and weighted total for tests/integration."""

    if float(perceptual_weight) > 0 and lpips_model is None:
        raise ValueError("perceptual_weight > 0 requires a frozen LPIPS model")
    components = {
        "color": albedo_color_loss(prediction, albedo, object_mask),
        "gradient": albedo_gradient_loss(prediction, albedo, object_mask),
        "perceptual": (
            albedo_perceptual_loss(
                prediction, albedo, object_mask, lpips_model=lpips_model
            )
            if lpips_model is not None
            else prediction.new_zeros(())
        ),
    }
    components["total"] = (
        float(color_weight) * components["color"]
        + float(gradient_weight) * components["gradient"]
        + float(perceptual_weight) * components["perceptual"]
    )
    return components


__all__ = [
    "foreground_mask_from_albedo",
    "masked_mean",
    "albedo_color_loss",
    "albedo_gradient_loss",
    "albedo_perceptual_loss",
    "albedo_multiscale_rgb_proxy",
    "masked_lpips_loss",
    "v2_flow_loss",
    "compute_v2_base_components",
    "average_v2_base_components",
    "weighted_v2_base_total",
    "object_alignment_consistency_loss",
    "albedo_auxiliary_loss",
    "compute_albedo_losses",
]

from ..purifier_losses import (
    _as_nchw,
    _gradient,
    _resize_mask,
    average_v2_base_components,
    compute_v2_base_components,
    masked_lpips_loss,
    masked_mean,
    multiscale_low_frequency_luminance_loss,
    v2_flow_loss,
    weighted_v2_base_total,
)
