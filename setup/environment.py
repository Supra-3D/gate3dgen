"""Validate installed data/train/eval dependencies without loading checkpoints."""
from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distribution as installed_distribution
from importlib.metadata import version
from pathlib import Path

MODULES = {
    "numpy": "numpy", "PIL": "Pillow", "yaml": "PyYAML", "scipy": "scipy",
    "skimage": "scikit-image", "trimesh": "trimesh", "embreex": "embreex", "rtree": "rtree",
    "torch": "torch", "torchvision": "torchvision", "accelerate": "accelerate",
    "datasets": "datasets", "diffusers": "diffusers", "transformers": "transformers",
    "peft": "peft", "lpips": "lpips", "cv2": "opencv-python", "orjson": "orjson",
    "safetensors": "safetensors", "tensorboard": "tensorboard", "torchdata": "torchdata",
    "einops": "einops", "imageio": "imageio", "loguru": "loguru", "requests": "requests",
    "timm": "timm", "kornia": "kornia",
}


def check(blender=None):
    checks = {}
    def run(name, callback):
        try:
            checks[name] = {"ok": True, "details": callback()}
        except Exception as error:  # noqa: BLE001 -- collect every dependency failure in one report
            checks[name] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    for module, distribution in MODULES.items():
        run(module, lambda module=module, distribution=distribution:
            (importlib.import_module(module), version(distribution))[1])
    for module in ("gate3dgen.training.router_tools.train",
                   "gate3dgen.training.vendor.train.src.models",
                   "gate3dgen.training.vendor.train.src.sft",
                   "gate3dgen.evaluation.pbr_metrics"):
        run(module, lambda module=module: importlib.import_module(module).__name__)
    def webp():
        from PIL import Image, features
        if not features.check("webp"):
            raise RuntimeError("Pillow lacks WebP codec")
        data = io.BytesIO()
        Image.new("RGBA", (2, 2), (20, 40, 60, 100)).save(data, format="WEBP", lossless=True)
        data.seek(0)
        with Image.open(data) as image:
            if image.convert("RGBA").getpixel((0, 0)) != (20, 40, 60, 100):
                raise RuntimeError("WebP RGBA roundtrip failed")
        return "lossless RGBA encode/decode OK"
    run("webp_codec", webp)
    def geometry():
        import trimesh

        from gate3dgen.evaluation.metrics3d_official import evaluate_pair
        mesh = trimesh.creation.box()
        row = evaluate_pair(mesh, mesh, samples=64, shell_views=10, shell_resolution=16)
        if not row["protocol"]["outer_shell_render_protocol"]:
            raise RuntimeError("visible-shell backend did not run")
        return {name: row["metrics"][name] for name in ("mesh_distance", "outer_shell_chamfer_distance")}
    run("geometry_smoke", geometry)
    def lpips():
        import lpips as library
        import torch
        # Exercise architecture and bundled calibration weights without downloading
        # pretrained AlexNet. This is dependency validation, NOT a paper LPIPS score.
        model = library.LPIPS(net="alex", pnet_rand=True, verbose=False)
        with torch.no_grad():
            score = model(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64))
        if not torch.isfinite(score).all():
            raise RuntimeError("LPIPS smoke result nonfinite")
        return "forward OK; random backbone, no benchmark claim"
    run("lpips_smoke", lpips)
    def pip_check():
        result = subprocess.run([sys.executable, "-I", "-m", "pip", "check"],
                                check=False, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout.strip()
    run("pip_check", pip_check)
    def entrypoints():
        results = []
        for entry in installed_distribution("gate3dgen").entry_points:
            if entry.group != "console_scripts":
                continue
            command = Path(sys.executable).parent / entry.name
            result = subprocess.run([str(command), "--help"], check=False, capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(f"{entry.name} --help failed: {result.stdout}\n{result.stderr}")
            results.append(entry.name)
        if not results:
            raise RuntimeError("installed package has no console entry points")
        return results
    run("entrypoints_help", entrypoints)
    if blender:
        from gate3dgen.evaluation.blender_runtime import ensure_runtime, probe
        run("blender_numpy", lambda: str(ensure_runtime(blender)))
        def color_management():
            details = probe(blender)
            if not details["color_management_ok"]:
                raise RuntimeError("Blender OCIO/color-management resources are invalid")
            return {key: details[key] for key in ("color_management_ok", "ocio_config", "ocio_config_sha256")}
        run("blender_color_management", color_management)
    cuda = {}
    try:
        import torch
        cuda = {"available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
        if cuda["available"]:
            cuda["device"] = torch.cuda.get_device_name(0)
            run("cuda_smoke", lambda: float((torch.ones(4, device="cuda") * 2).sum().cpu()))
    except Exception as error:  # noqa: BLE001 -- diagnostic, not a successful runtime assertion
        cuda["error"] = str(error)
    return {"schema_version": "environment_check_v1", "python": sys.version,
            "tested_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {"prefix": sys.prefix, "base_prefix": sys.base_prefix,
                            "executable": sys.executable},
            "isolated": {"user_site_disabled": not __import__("site").ENABLE_USER_SITE,
                         "pythonpath_set": bool(os.environ.get("PYTHONPATH"))},
            "checks": checks, "cuda": cuda, "ok": all(value["ok"] for value in checks.values()),
            "scope": "dependency and synthetic smoke checks; no checkpoint inference or full benchmark"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = check(args.blender)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
