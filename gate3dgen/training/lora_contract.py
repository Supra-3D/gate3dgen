from typing import Any, Iterable, Tuple


def assert_only_lora_trainable(named_parameters: Iterable[Tuple[str, Any]]) -> int:
    """Fail before training if PEFT left base-model parameters trainable."""
    trainable = [name for name, parameter in named_parameters if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA model has no trainable parameters")
    unexpected = [name for name in trainable if "lora_" not in name.lower()]
    if unexpected:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {unexpected[:10]}")
    return len(trainable)

