from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gate3dgen.data.pair_generation.planner import create_plan
from gate3dgen.data.prepare import prepare_dataset
from gate3dgen.data.prepare import build_router_validation
from gate3dgen.evaluation.router import load_samples, summarize
from gate3dgen.training.router_tools.launch import command_for
from gate3dgen.training.router_tools.manifest import build_gate_manifests

ROOT = Path(__file__).resolve().parents[1]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("count", [1, 5])
def test_generation_plan_supports_custom_asset_counts(tmp_path, count):
    rows = []
    for index in range(count):
        views = []
        for slot in range(4):
            path = tmp_path / f"renders_crossview/{index}/{slot:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (index * 20, slot * 20, 30)).save(path)
            views.append({"view_id": f"{slot:03d}", "rank": slot, "path": str(path)})
        rows.append({"sha256": f"{index:032x}", "semantic": {"primary_category": {"coarse": "object", "fine": "fixture"}},
                     "vlm": {"view_selection": {"selected_views": views}}})
    manifest = tmp_path / "assets.jsonl"
    write_rows(manifest, rows)
    out = tmp_path / "plan"
    summary = create_plan(manifest, ROOT / "gate3dgen/resources/prompts/gate3dgen_pair_synthesis.json", tmp_path, out)
    assert summary["pairs"] == count * 12
    planned = [json.loads(line) for line in (out / "plans/master.jsonl").read_text().splitlines()]
    for asset in rows:
        assert len({r["split"] for r in planned if r["sha256"] == asset["sha256"]}) == 1
    second = create_plan(manifest, ROOT / "gate3dgen/resources/prompts/gate3dgen_pair_synthesis.json", tmp_path, tmp_path / "repeat")
    assert second["split_pairs"] == summary["split_pairs"]
    assert (out / "plans/master.jsonl").read_bytes() == (tmp_path / "repeat/plans/master.jsonl").read_bytes()


def test_preparation_filters_unreviewed_rows_and_rejects_asset_leakage(tmp_path):
    pairs = [{"pair_id": "a", "sha256": "one", "split": "train"},
             {"pair_id": "b", "sha256": "two", "split": "test"}]
    source, audit = tmp_path / "pairs.jsonl", tmp_path / "audit.jsonl"
    write_rows(source, pairs)
    write_rows(audit, [{"pair_id": "a", "quality_status": "pass"}])
    result = prepare_dataset(source, audit, tmp_path / "out")
    assert result["accepted"]["train"] == 1 and result["excluded"] == {"missing": 1}
    pairs[1]["sha256"] = "one"
    write_rows(source, pairs)
    with pytest.raises(ValueError, match="crosses splits"):
        prepare_dataset(source, audit, tmp_path / "bad")


def test_router_manifest_builds_from_owned_pairs_without_private_hand_sidecar(tmp_path):
    rng = np.random.default_rng(42)
    pairs = []
    for index, split in enumerate(("train", "validation")):
        images = {}
        for kind in ("degraded", "gt", "source"):
            rel = (f"renders_crossview/{index}/000.png" if kind == "source" else f"pairs/{index}/{kind}.png")
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(path)
            images[kind] = (path, rel, hashlib.sha256(path.read_bytes()).hexdigest())
        row = {"pair_id": str(index), "sha256": f"{index:032x}", "split": split,
               "task_id": "optical_blur_heavy", "degradation_type": "optical_blur"}
        for kind, (path, rel, digest) in images.items():
            row[kind + "_path"] = str(path)
            row[kind + "_relative_path"] = rel
            row["source_image_sha256" if kind == "source" else kind + "_sha256"] = digest
        pairs.append(row)
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    write_rows(train, pairs[:1]); write_rows(validation, pairs[1:])
    build_gate_manifests(train_manifest=train, validation_manifest=validation, hand_manifest=None,
                         output_dir=tmp_path / "router", workspace_root=tmp_path,
                         data_root=tmp_path, texverse_root=tmp_path, texverse_full_root=tmp_path,
                         image_workers=1)
    output = [json.loads(line) for line in (tmp_path / "router/train.jsonl").read_text().splitlines()]
    assert {row["image_kind"] for row in output} == {"degraded", "gt_clean", "albedo_render"}
    assert [row["decision"] for row in output].count("REPAIR") == 1


def test_router_metrics_and_duplicate_labels(tmp_path):
    rows = [{"label": 0, "purify_probability": 0.1}, {"label": 1, "purify_probability": 0.9}]
    result = summarize(rows, 0.5)
    assert result["accuracy"] == result["auroc"] == 1
    Image.new("RGB", (8, 8), "red").save(tmp_path / "a.png")
    manifest = tmp_path / "eval.jsonl"
    write_rows(manifest, [{"image": "a.png", "label": 0}] * 2)
    assert len(load_samples(manifest, tmp_path, tmp_path, tmp_path)) == 1
    write_rows(manifest, [{"image": "a.png", "label": 0}, {"image": "a.png", "label": 1}])
    with pytest.raises(ValueError, match="Conflicting labels"):
        load_samples(manifest, tmp_path, tmp_path, tmp_path)


def test_router_launcher_matches_released_fsdp_contract():
    command = command_for(ROOT / "configs/train/router.yaml")
    assert command[command.index("--num_processes") + 1] == "8"
    assert command[command.index("--fsdp_sync_module_states") + 1] == "true"
    assert command[command.index("--fsdp_use_orig_params") + 1] == "true"
    assert "gate3dgen.training.router_tools.train" in command


def test_router_validation_keeps_three_populations_disjoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Image.new("RGB", (16, 16), "red").save("degraded.png")
    Image.new("RGB", (16, 16), "blue").save("gt.png")
    cases, train, final = [tmp_path / name for name in ("cases.jsonl", "train.jsonl", "final.jsonl")]
    row = {"case_id": "validation", "asset_id": "validation-asset", "degraded": "degraded.png",
           "gt": "gt.png", "task_id": "optical_blur", "reasons": ["OPTICAL_BLUR"]}
    write_rows(cases, [row])
    write_rows(train, [{"asset": "train-asset", "image_sha256": "0" * 64}])
    write_rows(final, [{"asset_id": "final-asset"}])
    output = tmp_path / "validation.jsonl"
    assert build_router_validation(cases, train, final, output)["validation_images"] == 2
    labels = [json.loads(line)["decision"] for line in output.read_text().splitlines()]
    assert labels == ["REPAIR", "KEEP"]
    write_rows(final, [{"asset_id": "train-asset"}])
    with pytest.raises(ValueError, match="overlap training"):
        build_router_validation(cases, train, final, output)
    write_rows(final, [{"asset_id": "validation-asset"}])
    with pytest.raises(ValueError, match="excluded from final"):
        build_router_validation(cases, train, final, output)
    write_rows(final, [{"asset_id": "final-asset"}])
    write_rows(train, [{"asset": "train-asset", "image_sha256": hashlib.sha256(Path("gt.png").read_bytes()).hexdigest()}])
    with pytest.raises(ValueError, match="bytes overlap training"):
        build_router_validation(cases, train, final, output)


@pytest.mark.parametrize("protocol,invoked", [("routed", False), ("force-purify", True)])
def test_evaluation_protocol_does_not_confuse_router_decision_with_invocation(tmp_path, monkeypatch, protocol, invoked):
    from gate3dgen.evaluation import infer
    from gate3dgen import purifier
    source = tmp_path / "input.jpg"
    Image.new("RGB", (16, 16), "blue").save(source)
    manifest = tmp_path / "manifest.jsonl"
    write_rows(manifest, [{"case_id": "case", "asset_id": "asset", "degraded": str(source)}])

    class FakeRouter:
        def __init__(self, *args): pass
        def predict(self, image): return {"decision": "KEEP", "purify_probability": 0.001, "threshold": 0.00644}

    class FakePurifier:
        def __init__(self, *args): assert invoked
        def purify(self, image, seed): return Image.new("RGB", (16, 16), "green")

    monkeypatch.setattr(infer, "Router", FakeRouter)
    monkeypatch.setattr(purifier, "Purifier", FakePurifier)
    out = tmp_path / "result"
    assert infer.main(["--manifest", str(manifest), "--output-dir", str(out), "--base-model", "unused", "--protocol", protocol,
                       "--profile", "--device", "cpu"]) == 0
    row = json.loads((out / "manifest.jsonl").read_text())
    assert row["router"]["decision"] == "KEEP"
    assert row["purifier_invoked"] == invoked
    profile = json.loads((out / "profile.json").read_text())
    assert profile["cases"] == [{"case_id": "case", "input_resolution": [16,16], "output_resolution": [16,16],
                                 "decision": "KEEP", "purifier_invoked": invoked}]
    assert profile["python"] and profile["packages"]["torch"]
    assert len(profile["purifier_inference_seconds"]) == int(invoked)
    if not invoked:
        assert Path(row["restored"]).read_bytes() == source.read_bytes()
