"""Purifier losses matching the released original trainer and paper Eq. 7–9."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

def _as_nchw(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must be a [B,C,H,W] tensor")
    if value.shape[1] not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels, got {value.shape[1]}")
    return value

def _resize_mask(mask: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.ndim not in (2, 3, 4):
        raise ValueError("mask must be [B,1,H,W], [B,H,W], [B,t], or [B,t,1]")
    mask = mask.to(device=reference.device, dtype=reference.dtype)
    if mask.shape[0] != reference.shape[0]:
        raise ValueError("mask batch size differs from image batch size")
    if reference.ndim == 4 and mask.ndim != 4:
        raise ValueError("spatial image masks must be [B,1,H,W]")
    if reference.ndim == 3:
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        if mask.ndim != 3 or mask.shape[1] != reference.shape[1]:
            raise ValueError("token masks must have shape [B,t] or [B,t,1]")
        if mask.shape[2] not in (1, reference.shape[2]):
            raise ValueError("token mask channel dimension differs from token values")
    if reference.ndim == 4 and mask.shape[-2:] != reference.shape[-2:]:
        mask = F.interpolate(mask, size=reference.shape[-2:], mode="nearest")
    return mask.clamp(0.0, 1.0)

def masked_mean(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean over channels/pixels, optionally normalized by selected pixels."""

    if mask is None:
        return value.mean()
    mask = _resize_mask(mask, value)
    assert mask is not None
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    # The mask is normalized by selected spatial pixels and channels equally;
    # this keeps the scale stable when object area differs between views.
    weighted = value * mask
    denominator = mask.expand_as(value).sum().clamp_min(1.0)
    return weighted.sum() / denominator

def _gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value[..., :, 1:] - value[..., :, :-1], value[..., 1:, :] - value[..., :-1, :]

def multiscale_low_frequency_luminance_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    scales: tuple[int, ...] = (8, 16, 32),
) -> torch.Tensor:
    """Match coarse illumination/color structure without a spatial mask.

    Average pooling suppresses texture and edges, so this term primarily
    penalizes broad under/over-exposure, hard-shadow and color-cast errors.
    It is intentionally full-image: task selection is handled by an explicit
    scalar loss multiplier rather than a learned or hand-authored mask.
    """

    prediction = _as_nchw(prediction, "prediction").float()
    target = _as_nchw(target, "target").float()
    if prediction.shape != target.shape:
        if target.shape[-2:] != prediction.shape[-2:]:
            target = F.interpolate(
                target, size=prediction.shape[-2:], mode="bilinear", align_corners=False
            )
        if target.shape[1] != prediction.shape[1]:
            raise ValueError("low-frequency prediction and target channel counts differ")
    if prediction.shape[1] < 3:
        raise ValueError("low-frequency luminance loss requires RGB inputs")

    scales = tuple(int(scale) for scale in scales)
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("low-frequency scales must contain positive integers")

    # Preserve the native FireRed [-1, 1] range.  The constant RGB-to-luma
    # projection keeps gradients dense while avoiding an unnecessary clamp.
    coefficients = prediction.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    prediction_luma = (prediction[:, :3] * coefficients).sum(dim=1, keepdim=True)
    target_luma = (target[:, :3] * coefficients).sum(dim=1, keepdim=True)
    terms = []
    height, width = prediction_luma.shape[-2:]
    for scale in scales:
        kernel = min(scale, height, width)
        terms.append(
            F.l1_loss(
                F.avg_pool2d(prediction_luma, kernel_size=kernel, stride=kernel),
                F.avg_pool2d(target_luma, kernel_size=kernel, stride=kernel),
            )
        )
    return torch.stack(terms).mean()

def masked_lpips_loss(
    lpips_model: object,
    prediction: torch.Tensor,
    target: torch.Tensor,
    object_mask: Optional[torch.Tensor] = None,
    *,
    background_weight: float = 0.0,
) -> torch.Tensor:
    """Run a frozen LPIPS/VGG model on object/background black composites.

    LPIPS expects the usual ``[-1, 1]`` image range.  The mask is applied by
    replacing the outside region with ``-1`` (black in LPIPS space), rather
    than multiplying the feature loss after the fact.  This keeps A's
    perceptual target and B's foreground/background pixel contract aligned,
    while allowing view-dependent backgrounds to remain unconstrained.
    """

    if lpips_model is None:
        raise ValueError("masked_lpips_loss requires a frozen LPIPS model")
    prediction = _as_nchw(prediction, "prediction").float()
    target = _as_nchw(target, "target").float()
    if prediction.shape != target.shape:
        if target.shape[-2:] != prediction.shape[-2:]:
            target = F.interpolate(target, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
        if target.shape[1] != prediction.shape[1]:
            raise ValueError("LPIPS prediction and target channel counts differ")
    if object_mask is None:
        return lpips_model(prediction.clamp(-1.0, 1.0), target.clamp(-1.0, 1.0)).mean()

    mask = _resize_mask(object_mask, prediction)
    assert mask is not None
    prediction = prediction.clamp(-1.0, 1.0)
    target = target.clamp(-1.0, 1.0)
    prediction_fg = (prediction + 1.0) * mask - 1.0
    target_fg = (target + 1.0) * mask - 1.0
    foreground = lpips_model(prediction_fg, target_fg).mean()
    background_weight = float(background_weight)
    if background_weight <= 0.0:
        return foreground
    background = lpips_model(
        (prediction + 1.0) * (1.0 - mask) - 1.0,
        (target + 1.0) * (1.0 - mask) - 1.0,
    ).mean()
    return foreground + background_weight * background

def v2_flow_loss(
    noise_pred: torch.Tensor,
    target: torch.Tensor,
    weighting: Optional[torch.Tensor] = None,
    *,
    threshold: float = 50.0,
) -> torch.Tensor:
    """The existing V2 clipped flow-MSE component, factored for both views."""

    noise_pred = noise_pred.float()
    target = target.float()
    diff = noise_pred - target
    loss = F.mse_loss(noise_pred, target, reduction="none")
    loss = loss * (diff.abs() <= float(threshold)).to(loss.dtype)
    if weighting is not None:
        loss = loss * weighting.float()
    return loss.mean()

def compute_v2_base_components(
    noise_pred: torch.Tensor,
    target: torch.Tensor,
    noisy_latents: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    weighting: Optional[torch.Tensor] = None,
    decoded: Optional[torch.Tensor] = None,
    gt_pixels: Optional[torch.Tensor] = None,
    lpips_model: object = None,
    pixel_lpips_weight: float = 0.0,
    pixel_loss_fg_only: bool = False,
    pixel_background_weight: float = 0.0,
    degraded_pixels: Optional[torch.Tensor] = None,
    subject_mask: Optional[torch.Tensor] = None,
    restoration_v3: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute every V2 base component for one view.

    B calls this once for each real degradation view and averages the returned
    components before adding its consistency term.  Keeping the component
    calculation in a dependency-light helper makes it CPU-testable and
    guarantees that both views receive flow, latent-x0, pixel-L1, and LPIPS
    gradients under identical foreground/background rules.
    """

    x0_lat = noisy_latents.float() - sigmas.float() * noise_pred.float()
    zero = x0_lat.new_zeros(())
    components = {
        "flow": v2_flow_loss(noise_pred, target, weighting),
        "latent_x0": F.mse_loss(x0_lat, latents.float()),
        "pixel_l1": zero,
        "pixel_l1_fg": zero,
        "pixel_l1_bg": zero,
        "pixel_lpips": zero,
        "pixel_lpips_fg": zero,
        "pixel_lpips_bg": zero,
        "pixel_full_l1": zero,
        "pixel_subject_l1": zero,
        "pixel_affected_l1": zero,
        "pixel_subject_gradient": zero,
        "affected_mask_mean": zero,
    }
    if decoded is None or gt_pixels is None:
        return components
    decoded = decoded.float()
    gt_pixels = gt_pixels.float()
    if decoded.shape != gt_pixels.shape:
        if gt_pixels.shape[-2:] != decoded.shape[-2:]:
            gt_pixels = F.interpolate(gt_pixels, size=decoded.shape[-2:], mode="bilinear", align_corners=False)
        if gt_pixels.shape[1] != decoded.shape[1]:
            raise ValueError("decoded and GT channel counts differ")
    if restoration_v3:
        components["pixel_full_l1"] = F.l1_loss(decoded, gt_pixels)
        if subject_mask is None:
            raise ValueError("restoration V3 requires an explicit subject mask")
        subject_mask = _resize_mask(subject_mask, decoded)
        assert subject_mask is not None
        l1_map = (decoded - gt_pixels).abs()
        components["pixel_subject_l1"] = masked_mean(l1_map, subject_mask)

        pred_dx, pred_dy = _gradient(decoded)
        gt_dx, gt_dy = _gradient(gt_pixels)
        mask_x = subject_mask[..., :, 1:] * subject_mask[..., :, :-1]
        mask_y = subject_mask[..., 1:, :] * subject_mask[..., :-1, :]
        components["pixel_subject_gradient"] = 0.5 * (
            masked_mean((pred_dx - gt_dx).abs(), mask_x)
            + masked_mean((pred_dy - gt_dy).abs(), mask_y)
        )

        if degraded_pixels is None:
            raise ValueError("restoration V3 requires degraded RGB pixels")
        degraded_pixels = degraded_pixels.float()
        if degraded_pixels.shape != gt_pixels.shape:
            if degraded_pixels.shape[-2:] != gt_pixels.shape[-2:]:
                degraded_pixels = F.interpolate(
                    degraded_pixels,
                    size=gt_pixels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if degraded_pixels.shape[1] != gt_pixels.shape[1]:
                raise ValueError("degraded and GT channel counts differ")
        # RGB tensors use [-1, 1].  Dividing the channel-mean absolute
        # difference by two gives a soft [0, 1] degradation magnitude.  A
        # small local average makes generated reflection/glass boundaries
        # less brittle while retaining a fully differentiable weighted loss.
        affected_mask = (
            (degraded_pixels - gt_pixels).abs().mean(dim=1, keepdim=True) * 0.5
        ).clamp(0.0, 1.0)
        affected_mask = F.avg_pool2d(
            affected_mask, kernel_size=7, stride=1, padding=3
        )
        components["affected_mask_mean"] = affected_mask.mean()
        components["pixel_affected_l1"] = masked_mean(l1_map, affected_mask)
    foreground_mask = None
    if pixel_loss_fg_only:
        gt01 = (gt_pixels * 0.5 + 0.5).clamp(0.0, 1.0)
        luminance = (
            0.299 * gt01[:, 0]
            + 0.587 * gt01[:, 1]
            + 0.114 * gt01[:, 2]
        ).unsqueeze(1)
        foreground_mask = (luminance > (2.0 / 255.0)).to(decoded.dtype)
        foreground_mask = F.max_pool2d(foreground_mask, kernel_size=7, stride=1, padding=3)
        background_mask = 1.0 - foreground_mask
        l1_map = (decoded - gt_pixels).abs()
        channels = l1_map.shape[1]
        components["pixel_l1_fg"] = l1_map.mul(foreground_mask).sum() / (
            foreground_mask.sum() * channels
        ).clamp_min(1.0)
        components["pixel_l1_bg"] = l1_map.mul(background_mask).sum() / (
            background_mask.sum() * channels
        ).clamp_min(1.0)
        components["pixel_l1"] = (
            components["pixel_l1_fg"]
            + float(pixel_background_weight) * components["pixel_l1_bg"]
        )
    else:
        components["pixel_l1"] = F.l1_loss(decoded, gt_pixels)
    if float(pixel_lpips_weight) > 0.0:
        if lpips_model is None:
            raise RuntimeError("pixel_lpips_weight > 0 but LPIPS was not initialized")
        if foreground_mask is None:
            components["pixel_lpips"] = lpips_model(
                decoded.clamp(-1.0, 1.0), gt_pixels.clamp(-1.0, 1.0)
            ).mean()
        else:
            components["pixel_lpips_fg"] = masked_lpips_loss(
                lpips_model, decoded, gt_pixels, foreground_mask
            )
            if float(pixel_background_weight) > 0.0:
                components["pixel_lpips_bg"] = masked_lpips_loss(
                    lpips_model,
                    decoded,
                    gt_pixels,
                    1.0 - foreground_mask,
                )
            components["pixel_lpips"] = (
                components["pixel_lpips_fg"]
                + float(pixel_background_weight) * components["pixel_lpips_bg"]
            )
    return components

def average_v2_base_components(
    view_a: dict[str, torch.Tensor],
    view_b: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, torch.Tensor]:
    """Average V2 component tensors across views without detaching gradients."""

    if view_b is None:
        return dict(view_a)
    keys = tuple(view_a.keys())
    if tuple(view_b.keys()) != keys:
        raise ValueError("V2 view component keys differ")
    return {key: 0.5 * (view_a[key] + view_b[key]) for key in keys}

def weighted_v2_base_total(
    components: dict[str, torch.Tensor],
    *,
    flow_weight: float,
    latent_x0_weight: float,
    pixel_l1_weight: float,
    pixel_lpips_weight: float,
    pixel_full_l1_weight: float = 0.0,
    pixel_subject_l1_weight: float = 0.0,
    pixel_affected_l1_weight: float = 0.0,
    pixel_subject_gradient_weight: float = 0.0,
) -> torch.Tensor:
    """Apply the existing V2 weights after optional per-view averaging."""

    # Keep older callers/tests that construct only the V2 component mapping
    # compatible.  V3 components are additive and their default weights are
    # zero, so an absent component is equivalent to a scalar zero here.
    zero = components["flow"].new_zeros(())
    return (
        float(flow_weight) * components["flow"]
        + float(latent_x0_weight) * components["latent_x0"]
        + float(pixel_l1_weight) * components["pixel_l1"]
        + float(pixel_lpips_weight) * components["pixel_lpips"]
        + float(pixel_full_l1_weight) * components.get("pixel_full_l1", zero)
        + float(pixel_subject_l1_weight)
        * components.get("pixel_subject_l1", zero)
        + float(pixel_affected_l1_weight)
        * components.get("pixel_affected_l1", zero)
        + float(pixel_subject_gradient_weight)
        * components.get("pixel_subject_gradient", zero)
    )
