"""One-step purifier conditioned by the frozen router's hidden states."""
from __future__ import annotations

import json
from pathlib import Path

from .training.inference import generate, load_pipeline


class Purifier:
    def __init__(self, checkpoint: Path, base_model: Path, router, device: str = "cuda:0",
                 cpu_offload: bool = False):
        marker = json.loads((checkpoint / "checkpoint_complete.json").read_text())
        if marker.get("global_step") != 15000 or marker.get("condition_checkpoint_step") != 6000:
            raise ValueError("This release requires purifier step 15000 conditioned by router step 6000")
        self.pipe = load_pipeline(
            base_model, checkpoint, device=device, architecture="restoration_custom",
            attention_backend="native", cpu_offload=cpu_offload,
            condition_text_encoder=router.model, condition_processor=router.processor,
        )
        library = json.loads((Path(__file__).parent / "resources/prompts/purifier.json").read_text())
        prompt = library["restoration_prompts"][library["recommended_default_prompt_id"]]["prompt"]
        self.prompt = "\n\n".join([prompt, library["output_specification"], library["global_preservation_constraint"]])

    def purify(self, source: Path, seed: int = 42):
        return generate(self.pipe, source, self.prompt, seed=seed, steps=1, cfg=1.0)
