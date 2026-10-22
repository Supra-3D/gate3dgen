from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys

import pytest
from PIL import Image

from gate3dgen.evaluation.glb_compat import blender_compatible, convert, read_glb, write_glb


def webp_fixture(path):
    image = Image.new("RGBA", (4, 4), (40, 80, 120, 90))
    data = io.BytesIO()
    image.save(data, format="WEBP", lossless=True)
    binary = b"geometry" + data.getvalue()
    document = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 8},
                        {"buffer": 0, "byteOffset": 8, "byteLength": len(data.getvalue())}],
        "images": [{"bufferView": 1, "mimeType": "image/webp"}],
        "textures": [{"extensions": {"EXT_texture_webp": {"source": 0}}}],
        "extensionsUsed": ["EXT_texture_webp"], "extensionsRequired": ["EXT_texture_webp"]}
    write_glb(path, document, binary)


def test_lossless_webp_conversion_preserves_source_and_geometry(tmp_path):
    source, target = tmp_path / "source.glb", tmp_path / "target.glb"
    webp_fixture(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    old_doc, old_binary = read_glb(source)
    assert not blender_compatible(source)
    result = convert(source, target)
    assert result["converted_images"] == 1 and blender_compatible(target)
    assert before == hashlib.sha256(source.read_bytes()).hexdigest()
    doc, binary = read_glb(target)
    assert binary[:len(old_binary)] == old_binary
    assert doc["bufferViews"][0] == old_doc["bufferViews"][0]
    assert doc["textures"][0] == {"source": 0}
    image = doc["images"][0]
    view = doc["bufferViews"][image["bufferView"]]
    with Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])) as png:
        assert png.convert("RGBA").getpixel((0, 0)) == (40, 80, 120, 90)
    assert not list(tmp_path.glob("*.partial"))


def test_source_overwrite_and_invalid_glb_rejected(tmp_path):
    source = tmp_path / "source.glb"
    webp_fixture(source)
    with pytest.raises(ValueError, match="read-only"):
        convert(source, source)
    source.write_bytes(b"truncated")
    assert not blender_compatible(source)
    with pytest.raises(ValueError, match="truncated"):
        convert(source, tmp_path / "target.glb")


def test_packaged_standalone_converter_runs_in_isolated_host(tmp_path):
    from gate3dgen.evaluation import glb_compat
    source, target = tmp_path / "source.glb", tmp_path / "target.glb"
    webp_fixture(source)
    result = subprocess.run([sys.executable, "-I", glb_compat.__file__,
                            "--input", str(source), "--output", str(target)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["converted_images"] == 1
    assert blender_compatible(target)


def test_render_wrapper_passes_host_python_and_python_exit_code(tmp_path, capsys):
    from gate3dgen.evaluation.render_pbr import main
    main(["--metrics", str(tmp_path / "metrics.jsonl"), "--camera-plan", str(tmp_path / "camera.json"),
          "--render-root", str(tmp_path / "renders"), "--mode", "prediction", "--dry-run"])
    command = capsys.readouterr().out
    assert "--python-exit-code 1" in command
    assert "--compatibility-python" in command and sys.executable in command
    assert "--compatibility-cache-root" in command


@pytest.mark.skipif(not os.environ.get("GATE3DGEN_TEST_BLENDER"),
                    reason="set GATE3DGEN_TEST_BLENDER for the external Blender integration test")
def test_webp_glb_renders_all_seven_channels_in_blender(tmp_path):
    import numpy as np
    import trimesh
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)), image=Image.new("RGBA", (4, 4), (40, 80, 120, 90)))
    source = tmp_path / "webp.glb"
    mesh.export(source, extension_webp=True)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    identity = np.eye(4).tolist()
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({"case_id": "fixture", "baseline": "trellis2", "variant": "degraded",
        "prediction_path": str(source), "reference_path": str(source),
        "alignment": {"matrix_normalized_prediction_to_reference": identity}}) + "\n")
    camera = tmp_path / "camera.json"
    camera.write_text(json.dumps({"cases": [{"case_id": "fixture", "source_anchor": {
        "expected_camera_matrix_world": [[1, 0, 0, .5], [0, 1, 0, .5], [0, 0, 1, 3], [0, 0, 0, 1]]}}]}))
    command = [sys.executable, "-m", "gate3dgen.evaluation.render_pbr",
        "--blender", os.environ["GATE3DGEN_TEST_BLENDER"], "--metrics", str(metrics),
        "--camera-plan", str(camera), "--render-root", str(tmp_path / "renders"),
        "--mode", "prediction", "--views", "1", "--resolution", "32", "--samples", "1"]
    if os.environ.get("GATE3DGEN_BLENDER_PYTHONPATH"):
        command += ["--blender-python-path", os.environ["GATE3DGEN_BLENDER_PYTHONPATH"]]
    environment = os.environ.copy()
    environment["GATE3DGEN_PBR_RENDER_DEVICE"] = "CPU"
    rendered = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=120)
    assert rendered.returncode == 0, rendered.stdout + rendered.stderr
    assert "[compat converted]" in rendered.stdout
    assert before == hashlib.sha256(source.read_bytes()).hexdigest()
    for channel in ("normal", "mask", "base_color", "metallic", "roughness", "alpha", "shaded"):
        image = tmp_path / "renders" / "trellis2" / "fixture" / "degraded" / channel / "view_00.png"
        with Image.open(image) as png:
            assert png.size == (32, 32)
            if channel == "mask":
                array = np.asarray(png.convert("RGB"))
                assert array.max() > 2, "blank/invalid-color-management render"
                assert np.any(array == 0), "fixture must include the black canvas"
    # A differently shaped prediction must not silently refit its own camera.
    stretched = mesh.copy()
    stretched.apply_scale([1.0, 0.3, 0.3])
    prediction = tmp_path / "stretched.glb"
    stretched.export(prediction, extension_webp=True)
    metrics.write_text(json.dumps({"case_id": "fixture", "baseline": "trellis2", "variant": "degraded",
        "prediction_path": str(prediction), "reference_path": str(source),
        "alignment": {"matrix_normalized_prediction_to_reference": identity}}) + "\n")
    for mode in ("gt", "prediction"):
        paired_command = list(command)
        paired_command[paired_command.index("--mode") + 1] = mode
        rendered = subprocess.run(paired_command + ["--overwrite"], capture_output=True, text=True,
                                  env=environment, timeout=120)
        assert rendered.returncode == 0, rendered.stdout + rendered.stderr
    gt_camera = json.loads((tmp_path / "renders/glb_gt/fixture/camera.json").read_text())
    pred_camera = json.loads((tmp_path / "renders/trellis2/fixture/degraded/camera.json").read_text())
    assert gt_camera["frame_policy"] == pred_camera["frame_policy"] == "shared_reference_fit_v1"
    assert gt_camera["camera_views"] == pred_camera["camera_views"]
    assert gt_camera["camera_matrix_world"] == pred_camera["camera_matrix_world"]
    assert gt_camera["renderer"].startswith("Blender_") and gt_camera["renderer"].endswith("_PBR_v3")
    # An old-image cache cannot be recertified by rewriting its metadata.
    camera.unlink()
    camera.write_text(json.dumps({"cases": [{"case_id": "fixture", "source_anchor": {
        "expected_camera_matrix_world": [[1, 0, 0, .5], [0, 1, 0, .5], [0, 0, 1, 4], [0, 0, 0, 1]]}}]}))
    # Move to a different orientation, not just a radius discarded by fitting.
    plan = json.loads(camera.read_text())
    plan["cases"][0]["source_anchor"]["expected_camera_matrix_world"] = [
        [0, 0, 1, 3], [1, 0, 0, .5], [0, 1, 0, .5], [0, 0, 0, 1]]
    camera.write_text(json.dumps(plan))
    changed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=120)
    assert changed.returncode != 0
    assert "Cached render camera/source changed" in changed.stdout + changed.stderr
