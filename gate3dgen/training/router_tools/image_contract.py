"""Pure tensor checks for Qwen-VL one-image repair-gate batches."""

from __future__ import annotations

from typing import Any

import torch


def inspect_image_token_contract(
    model_inputs: dict[str, torch.Tensor],
    examples: list[dict[str, Any]],
    *,
    image_token_id: int,
    spatial_merge_size: int,
) -> dict[str, Any]:
    """Compare image placeholders, vision grids, and flattened pixel patches."""

    errors: list[str] = []
    input_ids = model_inputs.get("input_ids")
    image_grid_thw = model_inputs.get("image_grid_thw")
    pixel_values = model_inputs.get("pixel_values")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        errors.append("input_ids must be a rank-2 tensor")
    if not isinstance(image_grid_thw, torch.Tensor) or image_grid_thw.ndim != 2:
        errors.append("image_grid_thw must be a rank-2 tensor")
    elif image_grid_thw.shape[1] != 3:
        errors.append(f"image_grid_thw must have shape [N,3], got {list(image_grid_thw.shape)}")
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 2:
        errors.append("pixel_values must be a rank-2 tensor")

    actual_counts: list[int] = []
    expected_counts: list[int] = []
    grid_values: list[list[int]] = []
    patch_rows = None
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
        actual_counts = [int((row == image_token_id).sum().item()) for row in input_ids]
    if (
        isinstance(image_grid_thw, torch.Tensor)
        and image_grid_thw.ndim == 2
        and image_grid_thw.shape[1] == 3
    ):
        grid_values = image_grid_thw.tolist()
        grid_products = image_grid_thw.to(torch.int64).prod(dim=-1)
        divisor = spatial_merge_size**2
        if bool((image_grid_thw <= 0).any()):
            errors.append(f"image_grid_thw contains non-positive values: {grid_values}")
        if bool((grid_products % divisor != 0).any()):
            errors.append(
                f"image grids are not divisible by merge_size^2={divisor}: {grid_values}"
            )
        expected_counts = (grid_products // divisor).tolist()
        if len(grid_values) != len(examples):
            errors.append(
                f"one-image gate batch expected {len(examples)} grids, got {len(grid_values)}"
            )
        if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 2:
            patch_rows = int(pixel_values.shape[0])
            expected_patch_rows = int(grid_products.sum().item())
            if patch_rows != expected_patch_rows:
                errors.append(
                    f"pixel patch rows mismatch: pixel_values={patch_rows}, "
                    f"grid_product={expected_patch_rows}"
                )
    if actual_counts and expected_counts and actual_counts != expected_counts:
        errors.append(
            f"image placeholder/feature mismatch: tokens={actual_counts}, "
            f"grid_features={expected_counts}"
        )

    samples = []
    for batch_index, example in enumerate(examples):
        samples.append(
            {
                "batch_index": batch_index,
                "dataset_index": int(example["dataset_index"]),
                "pair_id": str(example["pair_id"]),
                "sample_kind": str(example["sample_kind"]),
                "task_id": str(example["task_id"]),
                "image_path": str(example["image_path"]),
                "image_fallback": example.get("image_fallback"),
                "image_size": list(example["image"].size),
                "image_token_count": (
                    actual_counts[batch_index] if batch_index < len(actual_counts) else None
                ),
                "expected_image_feature_count": (
                    expected_counts[batch_index] if batch_index < len(expected_counts) else None
                ),
                "image_grid_thw": (
                    grid_values[batch_index] if batch_index < len(grid_values) else None
                ),
            }
        )
    return {
        "status": "error" if errors else "ok",
        "errors": errors,
        "image_token_id": int(image_token_id),
        "spatial_merge_size": int(spatial_merge_size),
        "pixel_patch_rows": patch_rows,
        "samples": samples,
    }

