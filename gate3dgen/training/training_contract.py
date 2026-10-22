from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def write_trainable_parameters(model, output_dir):
    """Record the exact pre-FSDP parameter selection, including shapes and zero-based names."""
    import json, os, re
    from pathlib import Path
    if int(os.environ.get("RANK", "0")) != 0:
        return
    rows = [{"name": n, "shape": list(p.shape), "numel": p.numel(), "dtype": str(p.dtype)}
            for n, p in model.named_parameters() if p.requires_grad]
    blocks = getattr(model, "transformer_blocks", None)
    block_count = len(blocks) if blocks is not None else None
    selected = sorted({int(match.group(1)) for row in rows
                       if (match := re.search(r"(?:^|\.)transformer_blocks\.(\d+)\.", row["name"]))})
    destination = Path(output_dir) / "trainable_parameters.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"layer_indexing": "zero_based_module_indices", "parameters": rows,
                                      "transformer_block_count": block_count,
                                      "blocks_with_trainable_parameters": selected,
                                      "blocks_with_no_trainable_parameters": None if block_count is None else
                                          [i for i in range(block_count) if i not in selected],
                                      "trainable_numel": sum(r["numel"] for r in rows)}, indent=2) + "\n")


def freeze_first_by_name(model: Any, trainable_patterns: Iterable[str]) -> tuple[int, int]:
    """Freeze a model first, then enable parameters matching explicit names."""

    patterns = [str(pattern) for pattern in trainable_patterns]
    if not patterns:
        raise RuntimeError("partial-unfreeze requires at least one trainable module pattern")
    model.requires_grad_(False)
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            parameter.requires_grad_(True)
            trainable += int(parameter.numel())
        else:
            frozen += int(parameter.numel())
    if trainable == 0:
        raise RuntimeError(f"trainable module patterns matched no parameters: {patterns}")
    return trainable, frozen
