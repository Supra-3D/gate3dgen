from __future__ import annotations

import json
import copy
import hashlib
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
pytest.importorskip("transformers")
import yaml
from PIL import Image
from torch import nn

from gate3dgen.training.router_tools.export import (
    CompleteStateError,
    load_complete_state,
    merge_complete_state,
    normalize_complete_state_keys,
)
from gate3dgen.training.router_tools.lora import (
    LoRALinear,
    VisionLoraConfig,
    count_parameters,
    inject_vision_lora,
    weighted_causal_lm_loss,
)
from gate3dgen.training.router_tools.weights import (
    MASK_SCHEMA_VERSION,
    RMBG2_MASK_SCHEMA,
    SUBTLE_INELIGIBLE_REASON,
    build_mask_sidecar,
    build_weight_sidecar,
)
from gate3dgen.training.router_tools.train import (
    GateV3Collator,
    validate_canonical_reasons,
    validate_config,
)




def test_weighted_causal_loss_is_finite_and_has_gradient():
    logits = torch.randn(2, 5, 11, requires_grad=True)
    labels = torch.tensor([
        [-100, 1, 2, 3, -100],
        [-100, 4, 5, 6, -100],
    ])
    token_weights = torch.tensor([
        [0.0, 0.5, 3.0, 5.0, 0.0],
        [0.0, 0.5, 1.0, 5.0, 0.0],
    ])
    sample_weights = torch.tensor([2.0, 1.0])
    loss = weighted_causal_lm_loss(logits, labels, token_weights, sample_weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


class _VisionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 8)
        self.attn.proj = nn.Linear(8, 8)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 16)
        self.mlp.up_proj = nn.Linear(8, 16)
        self.mlp.down_proj = nn.Linear(16, 8)


class _TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([_VisionBlock() for _ in range(32)])


class _TinyCompleteQwen(_TinyQwen):
    """Small wrapper with language layers/norm/lm head for exporter fixtures."""

    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(32)])
        self.language_model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 7)


class _TinyRegisteredVisualAliasQwen(nn.Module):
    """Mimic Qwen's registered ``model.visual`` plus convenience alias."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([_VisionBlock() for _ in range(32)])

    @property
    def visual(self) -> nn.Module:
        return self.model.visual


def _complete_lora_fixture_state(model: _TinyCompleteQwen, config: VisionLoraConfig) -> dict[str, torch.Tensor]:
    injected = copy.deepcopy(model)
    inject_vision_lora(injected, config)
    state = {key: value.detach().clone() for key, value in injected.state_dict().items()}
    # Simulate actual language-layer and final-norm updates.  These keys must
    # survive export just like the LoRA delta.
    state["language_model.layers.24.weight"] += 0.125
    state["language_model.norm.weight"] += 0.075
    target = "visual.blocks.28.attn.qkv"
    state[f"{target}.lora_A"] = torch.full_like(state[f"{target}.lora_A"], 0.2)
    state[f"{target}.lora_B"] = torch.full_like(state[f"{target}.lora_B"], 0.1)
    return state


def test_b_lora_injects_only_last_four_blocks_and_merges():
    model = _TinyQwen()
    config = VisionLoraConfig(enabled=True)
    names = inject_vision_lora(model, config)
    assert len(names) == 20
    assert all("blocks.2." not in name for name in names)
    assert all(any(f"blocks.{index}." in name for index in (28, 29, 30, 31)) for name in names)
    counts = count_parameters(model)
    assert counts["lora_trainable"] == 4 * (2 * (16 * 8 + 8 * 16) + 3 * (16 * 8 + 16 * 16))
    assert isinstance(model.visual.blocks[28].attn.qkv, LoRALinear)
    assert isinstance(model.visual.blocks[27].attn.qkv, nn.Linear)


def test_b_lora_uses_registered_model_visual_prefix_when_wrapper_has_alias():
    model = _TinyRegisteredVisualAliasQwen()
    names = inject_vision_lora(model, VisionLoraConfig(enabled=True))
    assert names[0] == "model.visual.blocks.28.attn.qkv"
    assert all(key.startswith("model.visual.blocks.") for key in model.state_dict() if ".lora_" in key)


def test_exporter_strictly_preserves_language_update_and_merges_lora():
    config = VisionLoraConfig(enabled=True)
    base = _TinyCompleteQwen()
    state = _complete_lora_fixture_state(base, config)
    expected_language = state["language_model.layers.24.weight"].clone()
    expected_norm = state["language_model.norm.weight"].clone()
    expected_a = state["visual.blocks.28.attn.qkv.lora_A"].clone()
    expected_b = state["visual.blocks.28.attn.qkv.lora_B"].clone()
    exported = copy.deepcopy(base)
    metadata = merge_complete_state(exported, state, config)

    assert torch.equal(exported.language_model.layers[24].weight, expected_language)
    assert torch.equal(exported.language_model.norm.weight, expected_norm)
    merged = exported.visual.blocks[28].attn.qkv
    assert isinstance(merged, nn.Linear)
    probe = torch.randn(2, 8)
    base_output = base.visual.blocks[28].attn.qkv(probe)
    delta = torch.nn.functional.linear(
        torch.nn.functional.linear(probe, expected_a), expected_b
    ) * (config.alpha / config.rank)
    assert torch.allclose(merged(probe), base_output + delta, atol=1e-5, rtol=1e-5)
    assert not any(".lora_A" in key or ".lora_B" in key for key in exported.state_dict())
    assert not any(".base." in key for key in exported.state_dict())
    assert metadata["strict_complete_state_load"] is True
    assert metadata["loaded_language_trainable_key_count"] > 0
    assert metadata["loaded_lora_adapter_key_count"] == 40
    assert metadata["loaded_lora_wrapper_base_key_count"] == 40
    assert metadata["merged_module_count"] == 20


def test_exporter_normalizes_realistic_fsdp_prefixes_without_dropping_keys():
    expected = {
        "model.language_model.layers.24.self_attn.q_proj.weight",
        "model.language_model.layers.24.self_attn.q_proj.bias",
        "model.language_model.norm.weight",
        "model.visual.blocks.28.attn.qkv.lora_A",
    }
    state = {
        "module._fsdp_wrapped_module.model.language_model.layers.24.self_attn.q_proj.weight": torch.ones(2, 2),
        "module._fsdp_wrapped_module.model.language_model.layers.24.self_attn.q_proj.bias": torch.ones(2),
        "module._fsdp_wrapped_module.model.language_model.norm.weight": torch.ones(2),
        "module._fsdp_wrapped_module.model.visual.blocks.28.attn.qkv.lora_A": torch.ones(3, 2),
    }
    normalized, audit = normalize_complete_state_keys(state, expected)
    assert set(normalized) == expected
    assert torch.equal(normalized["model.language_model.layers.24.self_attn.q_proj.weight"], torch.ones(2, 2))
    assert "module." in audit["outer_prefixes_normalized"]


def test_exporter_rejects_raw_fsdp_directory_and_requires_merge_weights(tmp_path: Path):
    raw = tmp_path / "checkpoint-6000"
    raw.mkdir()
    (raw / ".metadata").write_text("raw distributed checkpoint", encoding="utf-8")
    with pytest.raises(CompleteStateError, match="accelerate merge-weights"):
        load_complete_state(raw)
    (raw / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CompleteStateError, match="accelerate merge-weights"):
        load_complete_state(raw)


def test_exporter_accepts_only_a_complete_merged_model_file(tmp_path: Path):
    config = VisionLoraConfig(enabled=True)
    state = _complete_lora_fixture_state(_TinyCompleteQwen(), config)
    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    torch.save(state, merged_dir / "pytorch_model.bin")
    loaded, source_format = load_complete_state(merged_dir)
    assert source_format == "single_complete_state"
    assert set(loaded) == set(state)
    assert torch.equal(loaded["language_model.layers.24.weight"], state["language_model.layers.24.weight"])


def test_exporter_rejects_incomplete_state_instead_of_adapter_only_load():
    config = VisionLoraConfig(enabled=True)
    model = _TinyCompleteQwen()
    injected = copy.deepcopy(model)
    inject_vision_lora(injected, config)
    incomplete = {
        key: value.detach().clone()
        for key, value in injected.state_dict().items()
        if ".lora_A" in key or ".lora_B" in key
    }
    with pytest.raises(CompleteStateError, match="missing"):
        merge_complete_state(model, incomplete, config)


class _FakeImageProcessor:
    merge_size = 2


class _FakeTokenizer:
    eos_token_id = 151645

    def convert_tokens_to_ids(self, value: str) -> int:
        assert value == "<|image_pad|>"
        return 99

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert add_special_tokens is False
        result = {
            "input_ids": [1000 + ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }
        return result


class _FakeProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = _FakeImageProcessor()

    def __init__(self, *, adjacent_eos: bool = True) -> None:
        self.adjacent_eos = adjacent_eos

    def apply_chat_template(self, conversations, **_kwargs):
        answer = conversations[0][2]["content"][0]["text"]
        answer_ids = [1000 + ord(char) for char in answer]
        terminator = 151645 if self.adjacent_eos else 0
        ids = [10, 99, 99, 99, 99, 11, *answer_ids, terminator]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "pixel_values": torch.zeros((16, 3), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 2, 8]], dtype=torch.long),
        }


def _collator_example() -> dict[str, object]:
    answer = "DECISION: REPAIR\nREASONS: MOTION_BLUR"
    return {
        "sample_id": "fixture-repair",
        "pair_id": "fixture-pair",
        "sample_kind": "degraded",
        "dataset_index": 0,
        "task_id": "motion_blur",
        "decision": "REPAIR",
        "reasons": ["MOTION_BLUR"],
        "answer": answer,
        "system_prompt": "system",
        "user_prompt": "user",
        "image": Image.new("RGB", (8, 8)),
        "image_path": "/tmp/fixture.png",
    }


def _fixture_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_fixture_image(path: Path, value: int, *, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "L":
        image = Image.new(mode, (8, 8), color=value)
    else:
        image = Image.new(mode, (8, 8), color=(value, value, value))
    image.save(path)


def _make_rmbg2_sidecar_fixture(tmp_path: Path, *, trusted_count: int = 4) -> dict[str, object]:
    """Create a tiny complete-cache/reviewed/gate fixture for proxy tests."""

    workspace = tmp_path / "workspace"
    cache_root = workspace / "scripts/texverse_subject_mask_cache/fixture"
    source_manifest = workspace / "scripts/data/manifests/source.jsonl"
    reviewed_manifest = workspace / "scripts/data/manifests/reviewed.jsonl"
    gate_manifest = workspace / "scripts/data/manifests/gate.jsonl"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text("{\"pair_id\": \"source-fixture\"}\n", encoding="utf-8")

    gate_rows: list[dict[str, object]] = []
    reviewed_rows: list[dict[str, object]] = []
    expanded_rows: list[dict[str, object]] = []
    trusted_pairs: list[str] = []
    for index in range(trusted_count):
        pair_id = f"pair-trusted-{index}"
        sample_id = f"sample-trusted-{index}"
        degraded_path = workspace / f"data/degraded-{index}.png"
        clean_path = workspace / f"data/gt-{index}.png"
        mask_path = cache_root / f"trusted/gt-{index}.png.rmbg2-alpha.png"
        _write_fixture_image(clean_path, 100)
        # The foreground MAD values are 1, 2, ..., trusted_count.  This lets
        # the test prove that exactly the bottom 25% is selected.
        _write_fixture_image(degraded_path, 100 + index + 1)
        _write_fixture_image(mask_path, 255, mode="L")
        degraded_sha = _fixture_sha(degraded_path)
        gt_sha = _fixture_sha(clean_path)
        gate_row = {
            "sample_id": sample_id,
            "pair": pair_id,
            "decision": "REPAIR",
            "image_kind": "degraded",
            "image": {
                "root": "workspace",
                "relative_path": f"data/degraded-{index}.png",
                "sha256": degraded_sha,
            },
            "image_sha256": degraded_sha,
            "degraded_sha256": degraded_sha,
            "gt_sha256": gt_sha,
            "task": "optical_blur_heavy",
            "task_id": "optical_blur_heavy",
            "degradation": "optical_blur",
            "reasons": ["OPTICAL_BLUR"],
            "hand_remove_member": False,
            "severity_tag": "heavy",
            "severity_tag_provenance_only": True,
        }
        gate_rows.append(gate_row)
        expanded_rows.append(gate_row)
        clean_row = {
            "sample_id": f"{sample_id}:gt",
            "pair": pair_id,
            "decision": "KEEP",
            "image_kind": "gt_clean",
            "image": {
                "root": "workspace",
                "relative_path": f"data/gt-{index}.png",
                "sha256": gt_sha,
            },
            "image_sha256": gt_sha,
            "gt_sha256": gt_sha,
            "task": "optical_blur_heavy",
            "task_id": "optical_blur_heavy",
            "degradation": "optical_blur",
            "reasons": ["NONE"],
            "hand_remove_member": False,
            "severity_tag": "heavy",
            "severity_tag_provenance_only": True,
        }
        expanded_rows.append(clean_row)
        reviewed_rows.append(
            {
                "pair_id": pair_id,
                "degraded_path": str(degraded_path),
                "degraded_relative_path": f"data/degraded-{index}.png",
                "degraded_sha256": degraded_sha,
                "gt_path": str(clean_path),
                "gt_relative_path": f"trusted/gt-{index}.png",
                "gt_sha256": gt_sha,
                "dimensions": [8, 8],
                "task_id": "optical_blur_heavy",
            }
        )
        trusted_pairs.append(pair_id)

    hand_pair = "pair-hand-only"
    hand_sample = "sample-hand-only"
    hand_degraded = workspace / "data/hand-degraded.png"
    hand_gt = workspace / "data/hand-gt.png"
    _write_fixture_image(hand_degraded, 130)
    _write_fixture_image(hand_gt, 100)
    hand_degraded_sha = _fixture_sha(hand_degraded)
    hand_gt_sha = _fixture_sha(hand_gt)
    hand_row = {
        "sample_id": hand_sample,
        "pair": hand_pair,
        "decision": "REPAIR",
        "image_kind": "degraded",
        "image": {
            "root": "workspace",
            "relative_path": "data/hand-degraded.png",
            "sha256": hand_degraded_sha,
        },
        "image_sha256": hand_degraded_sha,
        "degraded_sha256": hand_degraded_sha,
        "gt_sha256": hand_gt_sha,
        "task": "optical_blur_heavy",
        "task_id": "optical_blur_heavy",
        "degradation": "optical_blur",
        "reasons": ["OPTICAL_BLUR", "EXTERNAL_HUMAN_SUPPORT"],
        "hand_remove_member": True,
        "severity_tag": "heavy",
        "severity_tag_provenance_only": True,
    }
    gate_rows.append(hand_row)
    expanded_rows.append(hand_row)
    expanded_rows.append(
        {
            "sample_id": f"{hand_sample}:gt",
            "pair": hand_pair,
            "decision": "KEEP",
            "image_kind": "gt_clean",
            "image": {
                "root": "workspace",
                "relative_path": "data/hand-gt.png",
                "sha256": hand_gt_sha,
            },
            "image_sha256": hand_gt_sha,
            "gt_sha256": hand_gt_sha,
            "task": "optical_blur_heavy",
            "task_id": "optical_blur_heavy",
            "degradation": "optical_blur",
            "reasons": ["NONE"],
            "hand_remove_member": True,
            "severity_tag": "heavy",
            "severity_tag_provenance_only": True,
        }
    )

    def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_jsonl(reviewed_manifest, reviewed_rows)
    write_jsonl(gate_manifest, gate_rows)
    expanded_manifest = workspace / "scripts/data/manifests/expanded.jsonl"
    write_jsonl(expanded_manifest, expanded_rows)
    cache_root.mkdir(parents=True, exist_ok=True)
    marker = {
        "status": "complete",
        "schema": RMBG2_MASK_SCHEMA,
        "manifest": str(source_manifest),
        "manifest_sha256": _fixture_sha(source_manifest),
        "processed": trusted_count,
        "unique_gt_total": trusted_count,
        "num_shards": 1,
        "anomalies": [],
    }
    (cache_root / "rmbg2_masks_complete.json").write_text(
        json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
    )
    reviewed_sha = _fixture_sha(reviewed_manifest)
    (cache_root / "rmbg2_manual_review.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "schema": "texverse-clean-gt-rmbg2-manual-review-v1",
                "base_complete_marker": "rmbg2_masks_complete.json",
                "reviewed_training_manifest": str(reviewed_manifest),
                "reviewed_training_manifest_sha256": reviewed_sha,
                "excluded_rows_manifest": None,
                "excluded_rows_manifest_sha256": None,
                "accepted_mask_overrides": [],
                "excluded_training_cases": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "workspace": workspace,
        "cache_root": cache_root,
        "source_manifest": source_manifest,
        "reviewed_manifest": reviewed_manifest,
        "gate_manifest": gate_manifest,
        "expanded_manifest": expanded_manifest,
        "trusted_pairs": trusted_pairs,
        "hand_sample": hand_sample,
        "hand_pair": hand_pair,
    }


def test_collator_supervises_adjacent_eos_with_distinct_weight():
    collator = GateV3Collator(
        _FakeProcessor(),
        {
            "structure_token_weight": 0.5,
            "decision_token_weight": 3.0,
            "reason_token_weight": 1.0,
            "eos_token_weight": 5.0,
        },
    )
    batch = collator([_collator_example()])
    labels = batch["labels"][0]
    weights = batch["token_weights"][0]
    supervised = labels != -100
    assert int(supervised.sum()) == len(_collator_example()["answer"]) + 1
    assert float(weights[supervised][-1]) == 5.0
    assert 3.0 in weights[supervised].tolist()
    assert 1.0 in weights[supervised].tolist()
    assert 0.5 in weights[supervised].tolist()
    assert int(labels[supervised][-1]) == 151645


def test_collator_fails_closed_when_eos_is_not_immediately_adjacent():
    collator = GateV3Collator(
        _FakeProcessor(adjacent_eos=False),
        {
            "structure_token_weight": 0.5,
            "decision_token_weight": 3.0,
            "reason_token_weight": 1.0,
            "eos_token_weight": 5.0,
        },
    )
    with pytest.raises(ValueError, match="immediately followed"):
        collator([_collator_example()])


def test_v3_reason_target_rejects_duplicates_and_bad_order():
    with pytest.raises(ValueError, match="duplicate"):
        validate_canonical_reasons(["MOTION_BLUR", "MOTION_BLUR"], "duplicate-fixture")
    with pytest.raises(ValueError, match="non-canonical"):
        validate_canonical_reasons(["OPTICAL_BLUR", "MOTION_BLUR"], "order-fixture")


def test_rmbg2_mask_sidecar_records_trusted_and_hand_only_rows(tmp_path: Path):
    fixture = _make_rmbg2_sidecar_fixture(tmp_path, trusted_count=4)
    output_dir = tmp_path / "mask_sidecar"
    audit = build_mask_sidecar(
        gate_manifest=fixture["gate_manifest"],
        reviewed_manifest=fixture["reviewed_manifest"],
        source_manifest=fixture["source_manifest"],
        mask_cache_root=fixture["cache_root"],
        output_dir=output_dir,
        workspace_root=fixture["workspace"],
        expected_gate_repair_rows=5,
        expected_reviewed_rows=4,
        expected_missing_by_design_rows=1,
        allow_fixture_counts=True,
    )
    assert audit["status"] == "ok"
    assert audit["trusted_eligible_rows"] == 4
    assert audit["missing_by_design_rows"] == 1
    assert audit["ordinary_base_missing_rows"] == 0
    rows = [
        json.loads(line)
        for line in (output_dir / "gate_mask_sidecar.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 5
    trusted = next(row for row in rows if row["sample_id"] == "sample-trusted-0")
    assert trusted["subtle_eligible"] is True
    assert trusted["mask_status"] == "ok"
    assert trusted["mask"]["root"] == "workspace"
    assert trusted["mask"]["relative_path"] == "scripts/texverse_subject_mask_cache/fixture/trusted/gt-0.png.rmbg2-alpha.png"
    assert len(trusted["mask"]["sha256"]) == 64
    assert trusted["mask"]["dimensions"] == [8, 8]
    assert trusted["cache_marker_sha256"] == audit["cache_marker_sha256"]
    assert trusted["reviewed_manifest_sha256"] == audit["reviewed_manifest_sha256"]
    hand = next(row for row in rows if row["sample_id"] == fixture["hand_sample"])
    assert hand["subtle_eligible"] is False
    assert hand["subtle_eligibility_reason"] == SUBTLE_INELIGIBLE_REASON
    assert hand["mask_status"] == "missing_by_design"
    assert hand["mask"] is None


def test_rmbg2_mask_sidecar_rejects_missing_ordinary_base_mask(tmp_path: Path):
    fixture = _make_rmbg2_sidecar_fixture(tmp_path, trusted_count=4)
    missing_mask = fixture["cache_root"] / "trusted/gt-0.png.rmbg2-alpha.png"
    missing_mask.unlink()
    with pytest.raises(ValueError, match="ordinary/base mask validation failed"):
        build_mask_sidecar(
            gate_manifest=fixture["gate_manifest"],
            reviewed_manifest=fixture["reviewed_manifest"],
            source_manifest=fixture["source_manifest"],
            mask_cache_root=fixture["cache_root"],
            output_dir=tmp_path / "mask_sidecar",
            workspace_root=fixture["workspace"],
            expected_gate_repair_rows=5,
            expected_reviewed_rows=4,
            expected_missing_by_design_rows=1,
            allow_fixture_counts=True,
        )


def test_rmbg2_weight_builder_selects_exact_bottom25_and_keeps_hand_heavy_family_only(
    tmp_path: Path,
):
    fixture = _make_rmbg2_sidecar_fixture(tmp_path, trusted_count=4)
    mask_dir = tmp_path / "mask_sidecar"
    build_mask_sidecar(
        gate_manifest=fixture["gate_manifest"],
        reviewed_manifest=fixture["reviewed_manifest"],
        source_manifest=fixture["source_manifest"],
        mask_cache_root=fixture["cache_root"],
        output_dir=mask_dir,
        workspace_root=fixture["workspace"],
        expected_gate_repair_rows=5,
        expected_reviewed_rows=4,
        expected_missing_by_design_rows=1,
        allow_fixture_counts=True,
    )
    weight_dir = tmp_path / "weights"
    audit = build_weight_sidecar(
        expanded_manifest=fixture["expanded_manifest"],
        output_dir=weight_dir,
        workspace_root=fixture["workspace"],
        mask_manifest=mask_dir / "gate_mask_sidecar.jsonl",
        require_subtle_proxy=True,
        allow_fixture_counts=True,
    )
    assert audit["subtle_proxy"]["status"] == "ok"
    assert audit["subtle_proxy"]["eligible_sample_count"] == 4
    assert audit["subtle_proxy"]["missing_by_design_count"] == 1
    assert audit["subtle_proxy"]["ordinary_base_missing_count"] == 0
    assert audit["subtle_proxy"]["family_counts"]["optical_blur"]["subtle"] == 1
    weights = [
        json.loads(line)
        for line in (weight_dir / "sample_weights.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    trusted_subtle = [row for row in weights if row["sample_id"].startswith("sample-trusted-") and row["subtle_proxy"]]
    assert [row["sample_id"] for row in trusted_subtle] == ["sample-trusted-0"]
    assert trusted_subtle[0]["subtle_proxy_mad"] == pytest.approx(1.0)
    hand = next(row for row in weights if row["sample_id"] == fixture["hand_sample"])
    assert hand["severity_tag"] == "heavy"
    assert hand["severity_tag_provenance_only"] is True
    assert hand["subtle_proxy"] is False
    assert hand["subtle_eligible"] is False
    assert hand["sample_weight"] == pytest.approx(1.5)
    assert hand["heavy_semantics"] == "provenance-only; never activates subtle proxy"

