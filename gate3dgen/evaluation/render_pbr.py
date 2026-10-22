"""Launch the original seven-channel benchmark-camera renderer inside Blender."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from .blender_runtime import default_blender


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default=default_blender())
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--camera-plan", type=Path, required=True)
    parser.add_argument("--mode", choices=("prediction", "gt"), required=True)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--threads", type=int, default=0, help="Blender CPU threads; 0 uses Blender's default")
    parser.add_argument("--frame-mode", choices=("fit", "plan"), default="fit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compatibility-cache-root", type=Path)
    parser.add_argument("--blender-python-path", type=Path, help="ABI-matched Blender NumPy target, not host site-packages")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 0:
        parser.error("--threads must be nonnegative")
    script = Path(__file__).with_name("blender_render_pbr.py")
    command = [args.blender, "--background", "--threads", str(args.threads),
               "--python-exit-code", "1", "--python", str(script), "--",
               "--metrics", str(args.metrics.resolve()), "--render-root", str(args.render_root.resolve()),
               "--camera-mode", "benchmark", "--camera-plan", str(args.camera_plan.resolve()),
               "--mode", args.mode, "--views", str(args.views), "--resolution", str(args.resolution),
               "--samples", str(args.samples), "--frame-mode", args.frame_mode,
               "--compatibility-python", sys.executable,
               "--compatibility-cache-root", str((args.compatibility_cache_root or
                   args.render_root / ".compatibility_cache").resolve())]
    if args.overwrite:
        command.append("--overwrite")
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        for path in (args.metrics, args.camera_plan):
            if not path.is_file():
                raise FileNotFoundError(path)
        from .blender_runtime import ensure_runtime, runtime_environment
        runtime = ensure_runtime(args.blender, runtime_path=args.blender_python_path)
        environment = runtime_environment()
        if runtime:
            environment["GATE3DGEN_BLENDER_PYTHONPATH"] = str(runtime)
        else:
            environment.pop("GATE3DGEN_BLENDER_PYTHONPATH", None)
        subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
