"""Degradation-aware router from the paper (Qwen2.5-VL checkpoint 6000)."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

RESOURCE_PROMPTS = Path(__file__).parent / "resources/prompts"
DEFAULT_THRESHOLD = 0.006440580124035478


def decision_from_probability(probability: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    if not 0 <= probability <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Probability and threshold must be finite and between zero and one")
    return "PURIFY" if probability >= threshold else "KEEP"


def decision_token_branch(tokenizer):
    # These strings are the trained vocabulary contract, not the public API names.
    purify_ids = tokenizer("DECISION: REPAIR", add_special_tokens=False).input_ids
    keep_ids = tokenizer("DECISION: KEEP", add_special_tokens=False).input_ids
    common = 0
    for left, right in zip(purify_ids, keep_ids):
        if left != right:
            break
        common += 1
    if not 0 < common < min(len(purify_ids), len(keep_ids)):
        raise ValueError("Cannot isolate the trained binary decision token branch")
    return purify_ids[:common], purify_ids[common], keep_ids[common]


class Router:
    def __init__(self, checkpoint: Path, base_model: Path, device: str = "cuda:0",
                 threshold: float = DEFAULT_THRESHOLD):
        import torch
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

        marker = json.loads((checkpoint / "checkpoint_complete.json").read_text())
        if marker.get("status") != "complete" or marker.get("global_step") != 6000:
            raise ValueError("This release requires the completed router checkpoint-6000")
        decision_from_probability(0.0, threshold)
        self.threshold = threshold
        self.device = device
        self.processor = Qwen2VLProcessor.from_pretrained(
            str(base_model), subfolder="processor", local_files_only=True
        )
        self.processor.tokenizer.padding_side = "left"
        config = Qwen2_5_VLConfig.from_pretrained(
            str(checkpoint / "exported_model"), local_files_only=True
        )
        for name, token in {
            "image_token_id": "<|image_pad|>", "video_token_id": "<|video_pad|>",
            "vision_start_token_id": "<|vision_start|>", "vision_end_token_id": "<|vision_end|>",
        }.items():
            if getattr(config, name, None) is None:
                setattr(config, name, self.processor.tokenizer.convert_tokens_to_ids(token))
        self.model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(checkpoint / "exported_model"), config=config, torch_dtype=torch.bfloat16,
            local_files_only=True, low_cpu_mem_usage=True, output_loading_info=True,
        )
        errors = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                  if info.get(key)}
        if errors:
            raise RuntimeError(f"Router weights do not match the expected architecture: {errors}")
        self.model.requires_grad_(False).eval().to(device)
        self.prompt = json.loads((RESOURCE_PROMPTS / "router.json").read_text())

    def predict(self, source: Path) -> dict:
        import torch

        from .training.vendor.train.src.utils.image_utils import apply_condition_transform_to_images

        with Image.open(source) as opened:
            image = opened.convert("RGB")
        image = apply_condition_transform_to_images(
            [image], ref_height=image.height, ref_width=image.width,
            image_sample_size=512, condition_image_size=147456,
        )[0]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.prompt["system_prompt"]}]},
            {"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": self.prompt["user_prompt"]}
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            [messages], add_generation_prompt=True, padding=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        common, positive, negative = decision_token_branch(self.processor.tokenizer)
        prefix = torch.tensor([common], dtype=inputs["input_ids"].dtype, device=self.device)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(prefix)], dim=1)
        with torch.inference_mode():
            result = self.model(**inputs, use_cache=False, logits_to_keep=1)
            probability = result.logits[:, -1, [positive, negative]].float().softmax(-1)[0, 0].item()
        return {"decision": decision_from_probability(probability, self.threshold),
                "purify_probability": probability, "threshold": self.threshold}
