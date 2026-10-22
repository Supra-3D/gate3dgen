"""Selectively purify an image or directory for image-to-3D generation."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .router import Router


def generate_3d(rows: list[dict], output: Path) -> None:
    from .evaluation.io import write_jsonl
    from .evaluation.reconstruct import main as reconstruct
    manifest = output / "manifest.jsonl"
    write_jsonl(manifest, rows)
    reconstruct(["--manifest", str(manifest), "--output", str(output / "generation")])


def process_images(images, args) -> list[dict]:
    runtime = Path(os.environ.get("GATE3DGEN_RUNTIME_PATH", "checkpoints/runtime"))
    router_checkpoint = Path(os.environ.get("GATE3DGEN_ROUTER_CHECKPOINT", "checkpoints/router/checkpoint-6000"))
    purifier_checkpoint = Path(os.environ.get("GATE3DGEN_PURIFIER_CHECKPOINT", "checkpoints/purifier/checkpoint-15000"))
    router = Router(router_checkpoint, runtime)
    # Route first: the purifier installs offload hooks on the shared encoder.
    decisions = [(source, {"decision": None, "purify_probability": None}) if args.purify_only
                 else (source, router.predict(source)) for source in images]
    args.output.mkdir(parents=True, exist_ok=True)
    purifier = None
    rows = []
    for source, decision in decisions:
        invoke = not args.router_only and (args.purify_only or decision["decision"] == "PURIFY")
        record = {"input": source.name, **decision,
                  "mode": "router-only" if args.router_only else "purify-only" if args.purify_only else "routed",
                  "routing_evaluated": not args.purify_only, "purifier_invoked": invoke,
                  "router_step": 6000, "purifier_step": 15000}
        if not args.router_only:
            target = args.output / (f"{source.stem}.png" if invoke else source.name)
            if invoke:
                from .purifier import Purifier
                if purifier is None:
                    purifier = Purifier(purifier_checkpoint, runtime, router, cpu_offload=True)
                purifier.purify(source, seed=42).save(target)
            else:
                shutil.copyfile(source, target)
            record.update(output=target.name, generation_status="pending")
            rows.append({"case_id": source.stem, "asset_id": source.stem, "dataset_kind": "wild_no_gt",
                         "original": str(source.resolve()), "degraded": str(source.resolve()),
                         "restored": str(target.resolve()), "router": decision,
                         "purifier_invoked": invoke, "evaluation_protocol": record["mode"], "seed": 42})
        (args.output / f"{source.name}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="An image or a directory of images")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--router-only", action="store_true", help="Classify only; do not purify or generate GLBs")
    mode.add_argument("--purify-only", action="store_true", help="Skip classification; purify and generate GLBs")
    args = parser.parse_args(argv)
    images = sorted(path for path in args.input.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}) if args.input.is_dir() else [args.input]
    if not images or any(not path.is_file() for path in images):
        parser.error("No input images found")
    targets = [args.output / path.name for path in images]
    targets += [args.output / f"{path.stem}.png" for path in images]
    sources = {path.resolve() for path in images}
    if any(path.resolve() in sources for path in targets):
        parser.error("Output must not overwrite input images")
    if len({path.stem for path in images}) != len(images):
        parser.error("Input filenames must have distinct stems")
    rows = process_images(images, args)
    if args.router_only:
        return 0
    # Release the shared router/purifier before starting the separate 3D backend.
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    error = None
    try:
        generate_3d(rows, args.output)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, SystemExit) as exception:
        error = f"Pixal3D generation failed: {exception}"
    for source in images:
        path = args.output / f"{source.name}.json"
        record = json.loads(path.read_text())
        meshes = {variant: args.output / "generation/pixal3d" / source.stem / f"{variant}.glb"
                  for variant in ("degraded", "restored_s01")}
        complete = all(mesh.is_file() and mesh.stat().st_size > 0 for mesh in meshes.values())
        record["generation_status"] = "complete" if complete and error is None else "failed"
        record["glb"] = {key: str(mesh.relative_to(args.output)) for key, mesh in meshes.items() if mesh.is_file()}
        if record["generation_status"] == "failed":
            record["generation_error"] = error or "Pixal3D did not produce both GLB files"
        path.write_text(json.dumps(record, indent=2) + "\n")
        if not complete and error is None:
            error = "Pixal3D did not produce both GLB files"
    if error:
        print(error + "; processed images and decision records have been retained.", file=sys.stderr)
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
