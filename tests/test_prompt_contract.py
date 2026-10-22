from __future__ import annotations

import hashlib
from pathlib import Path

from gate3dgen.training.prompts import restoration_instructions


ROOT = Path(__file__).resolve().parents[1]


def test_restoration_prompt_contract_uses_public_gate3dgen_identity() -> None:
    path = ROOT / "gate3dgen/resources/prompts/purifier.json"
    config = {
        "prompt": {
            "library": str(path),
            "id": "gate3dgen_purify_input_v1",
        }
    }
    contract = restoration_instructions(config)
    assert contract["prompt_id"] == "gate3dgen_purify_input_v1"
    assert contract["prompt_library_version"] == "gate3dgen-purifier-1.0"
    assert contract["prompt_library_fingerprint"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "Restore this image for single-image 3D generation" in contract["instruction"]
    assert "Preserve object identity" in contract["instruction"]

