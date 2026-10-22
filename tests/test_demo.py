from __future__ import annotations

import json

import pytest
from PIL import Image

from gate3dgen import demo, purifier
from gate3dgen.router import DEFAULT_THRESHOLD, decision_from_probability


@pytest.mark.parametrize("option", ["--cpu-offload", "--device", "--check"])
def test_demo_keeps_advanced_settings_out_of_cli(option):
    with pytest.raises(SystemExit) as exc:
        demo.main(["input.png", option])
    assert exc.value.code == 2


@pytest.fixture(autouse=True)
def fake_generation(monkeypatch):
    original = demo.generate_3d
    def generate(rows, output):
        for row in rows:
            folder = output / "generation/pixal3d" / row["case_id"]
            folder.mkdir(parents=True)
            for variant in ("degraded", "restored_s01"):
                (folder / f"{variant}.glb").write_bytes(b"fixture")
    monkeypatch.setattr(demo, "generate_3d", generate)
    return original


def test_3d_call_preserves_matched_image_paths(tmp_path, monkeypatch, fake_generation):
    from gate3dgen.evaluation import reconstruct
    source, processed = tmp_path / "source.png", tmp_path / "processed.png"
    rows = [{"case_id": "demo_01", "dataset_kind": "wild_no_gt",
             "original": str(source), "degraded": str(source), "restored": str(processed)}]
    observed = {}
    def generate(argv):
        observed["argv"] = argv
        observed["row"] = json.loads(demo.Path(argv[1]).read_text())
        return 0
    monkeypatch.setattr(reconstruct, "main", generate)
    output = tmp_path / "output"
    fake_generation(rows, output)
    assert observed["row"] == rows[0]
    assert observed["argv"] == ["--manifest", str(output / "manifest.jsonl"),
                                "--output", str(output / "generation")]


def test_keep_is_byte_identical_and_does_not_load_purifier(tmp_path, monkeypatch):
    source = tmp_path / "input.jpg"
    Image.new("RGB", (16, 16), "blue").save(source)
    class FakeRouter:
        def __init__(self, checkpoint, runtime):
            assert runtime == demo.Path("checkpoints/runtime")
        def predict(self, path):
            return {"decision": "KEEP", "purify_probability": 0.001, "threshold": DEFAULT_THRESHOLD}
    monkeypatch.delenv("GATE3DGEN_RUNTIME_PATH", raising=False)
    monkeypatch.setenv("GATE3DGEN_MODEL_PATH", "ABSENT_FIRERED")
    monkeypatch.setattr(demo, "Router", FakeRouter)
    def forbidden(*args, **kwargs):
        raise AssertionError("KEEP must not initialize a purifier")
    monkeypatch.setattr(purifier, "Purifier", forbidden)
    output = tmp_path / "output"
    assert demo.main([str(source), "--output", str(output)]) == 0
    assert (output / source.name).read_bytes() == source.read_bytes()
    record = json.loads((output / "input.jpg.json").read_text())
    assert record["decision"] == "KEEP" and not record["purifier_invoked"]
    assert record["generation_status"] == "complete"
    assert (output / record["glb"]["restored_s01"]).is_file()


def test_routed_purification_default_settings(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    Image.new("RGB", (16, 16), "blue").save(source)
    class FakeRouter:
        def __init__(self, *args): pass
        def predict(self, source):
            return {"decision": "PURIFY", "purify_probability": 0.9}
    class FakePurifier:
        def __init__(self, checkpoint, runtime, router, *, cpu_offload):
            assert isinstance(router, FakeRouter) and cpu_offload
        def purify(self, source, seed):
            assert seed == 42
            return Image.new("RGB", (16, 16), "green")
    monkeypatch.setattr(demo, "Router", FakeRouter)
    monkeypatch.setattr(purifier, "Purifier", FakePurifier)
    output = tmp_path / "output"
    assert demo.main([str(source), "--output", str(output)]) == 0
    record = json.loads((output / "input.png.json").read_text())
    assert record["mode"] == "routed" and record["purifier_invoked"]
    assert record["generation_status"] == "complete"


def test_only_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        demo.main(["input.png", "--router-only", "--purify-only"])


def test_router_only_never_purifies_or_generates_glb(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    Image.new("RGB", (16, 16)).save(source)
    class FakeRouter:
        def __init__(self, *args): pass
        def predict(self, path):
            return {"decision": "PURIFY", "purify_probability": 0.9}
    def forbidden(*args, **kwargs):
        raise AssertionError("Router-only must not load purifier or 3D backend")
    monkeypatch.setattr(demo, "Router", FakeRouter)
    monkeypatch.setattr(purifier, "Purifier", forbidden)
    monkeypatch.setattr(demo, "generate_3d", forbidden)
    output = tmp_path / "output"
    assert demo.main([str(source), "--output", str(output), "--router-only"]) == 0
    record = json.loads((output / "input.png.json").read_text())
    assert record["mode"] == "router-only" and not record["purifier_invoked"]
    assert not (output / "input.png").exists()


def test_purify_only_skips_decision_but_keeps_conditioning(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    Image.new("RGB", (16, 16)).save(source)
    class FakeRouter:
        def __init__(self, *args): pass
        def predict(self, path):
            raise AssertionError("Purify-only must skip classification")
    class FakePurifier:
        def __init__(self, checkpoint, runtime, router, **kwargs):
            assert isinstance(router, FakeRouter)
        def purify(self, path, seed):
            return Image.new("RGB", (16, 16), "green")
    monkeypatch.setattr(demo, "Router", FakeRouter)
    monkeypatch.setattr(purifier, "Purifier", FakePurifier)
    output = tmp_path / "output"
    assert demo.main([str(source), "--output", str(output), "--purify-only"]) == 0
    record = json.loads((output / "input.png.json").read_text())
    assert record["mode"] == "purify-only" and record["purifier_invoked"]
    assert record["decision"] is None and not record["routing_evaluated"]
    assert record["generation_status"] == "complete"


def test_3d_failure_retains_processed_image(tmp_path, monkeypatch):
    source = tmp_path / "input.png"
    Image.new("RGB", (16, 16)).save(source)
    class FakeRouter:
        def __init__(self, *args): pass
        def predict(self, path):
            return {"decision": "KEEP", "purify_probability": 0.001}
    def failure(*args):
        raise RuntimeError("backend unavailable")
    monkeypatch.setattr(demo, "Router", FakeRouter)
    monkeypatch.setattr(demo, "generate_3d", failure)
    output = tmp_path / "output"
    assert demo.main([str(source), "--output", str(output)]) == 1
    assert (output / "input.png").read_bytes() == source.read_bytes()
    record = json.loads((output / "input.png.json").read_text())
    assert record["generation_status"] == "failed" and "backend unavailable" in record["generation_error"]


def test_demo_refuses_overwriting_input(tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (16, 16)).save(source)
    with pytest.raises(SystemExit):
        demo.main([str(source), "--output", str(tmp_path)])


def test_frozen_router_threshold_and_invalid_scores():
    assert decision_from_probability(DEFAULT_THRESHOLD) == "PURIFY"
    assert decision_from_probability(DEFAULT_THRESHOLD / 2) == "KEEP"
    for invalid in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError):
            decision_from_probability(invalid)


def test_purifier_receives_router_conditioning(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "checkpoint_complete.json").write_text(json.dumps({
        "global_step": 15000, "condition_checkpoint_step": 6000,
    }))
    class FakeRouter:
        model = object()
        processor = object()
    observed = {}
    def load(*args, **kwargs):
        observed.update(kwargs)
        return object()
    monkeypatch.setattr(purifier, "load_pipeline", load)
    purifier.Purifier(checkpoint, tmp_path / "base", FakeRouter())
    assert observed["condition_text_encoder"] is FakeRouter.model
    assert observed["condition_processor"] is FakeRouter.processor
    assert observed["architecture"] == "restoration_custom"
