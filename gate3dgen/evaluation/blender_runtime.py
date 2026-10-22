"""Check/install NumPy for Blender's own Python, separately from host conda.

This is not a Blender installer. Blender and its glTF/Cycles resources must
already be installed. Pillow/WebP decoding runs in the host interpreter.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

MARKER = "GATE3DGEN_BLENDER_RUNTIME="


def default_blender() -> str:
    configured = os.environ.get("GATE3DGEN_BLENDER")
    if configured:
        return configured
    sibling = Path(sys.executable).parent / "blender"
    return str(sibling) if sibling.is_file() else "blender"


def runtime_environment() -> dict:
    """Keep Blender's ABI isolated; reuse an explicitly installed OCIO config."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    cached = Path(sys.prefix) / "share/gate3dgen/blender-color-3.0.1/config.ocio"
    if "OCIO" not in environment and cached.is_file():
        environment["OCIO"] = str(cached)
    return environment


def probe(blender: str, runtime_path: Path | None = None) -> dict:
    expression = (
        "import sys,json,importlib.util,bpy; "
        + (f"sys.path.insert(0, {str(runtime_path)!r}); " if runtime_path else "")
        + "print('" + MARKER + "'+json.dumps({'python':list(sys.version_info[:2]),"
          "'numpy':bool(importlib.util.find_spec('numpy')),"
          "'blender_version':bpy.app.version_string,"
          "'opencolorio_enabled':bool(bpy.app.build_options.opencolorio)})); "
          "import numpy; print('GATE3DGEN_NUMPY_VERSION='+numpy.__version__)"
    )
    environment = runtime_environment()
    result = subprocess.run([blender, "--background", "--factory-startup",
        "--python-exit-code", "1", "--python-expr", expression],
        capture_output=True, text=True, env=environment, timeout=120)
    markers = [line[len(MARKER):] for line in result.stdout.splitlines() if line.startswith(MARKER)]
    if not markers:
        raise RuntimeError(f"Blender runtime probe failed:\n{result.stdout}\n{result.stderr}")
    data = json.loads(markers[-1])
    data["numpy_import_ok"] = result.returncode == 0
    messages = (result.stdout + result.stderr).lower()
    data["color_management_ok"] = data["opencolorio_enabled"] and not any(message in messages for message in (
        "color management: using fallback mode", "could not find role", "failed to load ocio",
        "error: could not load ocio", "color management: error",
    ))
    ocio = environment.get("OCIO")
    data["ocio_config"] = ocio
    data["ocio_config_sha256"] = (hashlib.sha256(Path(ocio).read_bytes()).hexdigest()
                                  if ocio and Path(ocio).is_file() else None)
    return data


def ensure_runtime(blender: str, *, install: bool = False, runtime_path: Path | None = None) -> Path | None:
    native = probe(blender)
    if not native.get("color_management_ok", True):
        raise RuntimeError("Blender's OCIO/color-management resources are missing or invalid. "
                           "Use an OpenColorIO-enabled Blender build with complete resources, or set OCIO to its matching "
                           "colormanagement/config.ocio with all referenced LUTs. "
                           "Fallback color management is not valid for PBR evaluation.")
    if native["numpy_import_ok"]:
        return None
    major, minor = native["python"]
    target = runtime_path or Path(os.environ.get("GATE3DGEN_BLENDER_PYTHONPATH",
        str(Path(sys.prefix) / "share" / "gate3dgen" / f"blender-py{major}{minor}")))
    if probe(blender, target)["numpy_import_ok"]:
        return target
    if not install:
        raise RuntimeError("Blender's Python cannot import NumPy. Run: "
                           "python -m gate3dgen.evaluation.blender_runtime --install "
                           f"--blender {blender}")
    machine = platform.machine()
    if platform.system() != "Linux" or machine not in {"x86_64", "aarch64"}:
        raise RuntimeError("Automatic Blender NumPy bootstrap supports Linux x86_64/aarch64 only")
    # NumPy 1.x supports the older Blender 3 glTF importer. Never inject host
    # Python 3.12 binary modules into Blender's Python 3.10.
    requirement = "numpy==1.26.4" if (major, minor) < (3, 13) else "numpy>=2.1,<3"
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-I", "-m", "pip", "install", "--no-deps", "--upgrade",
        "--only-binary=:all:", "--implementation", "cp", "--python-version", f"{major}.{minor}",
        "--abi", f"cp{major}{minor}", "--platform", f"manylinux2014_{machine}",
        "--target", str(target), requirement], check=True)
    if not probe(blender, target)["numpy_import_ok"]:
        raise RuntimeError(f"Blender still cannot import ABI-matched NumPy from {target}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default=default_blender())
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--runtime-path", type=Path)
    args = parser.parse_args(argv)
    path = ensure_runtime(args.blender, install=args.install, runtime_path=args.runtime_path)
    print(json.dumps({"blender": args.blender, "runtime_path": str(path) if path else None,
                      **probe(args.blender, path)}, indent=2))


if __name__ == "__main__":
    main()
