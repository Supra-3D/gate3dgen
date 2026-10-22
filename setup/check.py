"""Check dependencies and the released inference package."""
from __future__ import annotations

import argparse
import json
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def validate_checkpoints(router: Path, purifier: Path, runtime: Path) -> dict:
    marker = json.loads((router / "checkpoint_complete.json").read_text())
    if marker.get("status") != "complete" or marker.get("global_step") != 6000:
        raise ValueError("Router checkpoint must be completed step 6000")
    export = router / "exported_model"
    index = json.loads((export / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
        raise ValueError("Router shard path escapes the checkpoint")
    files = [export / name for name in names]
    files += [export / "config.json", runtime / "processor/preprocessor_config.json",
              runtime / "processor/tokenizer_config.json", runtime / "processor/tokenizer.json"]
    marker = json.loads((purifier / "checkpoint_complete.json").read_text())
    if not marker.get("complete") or marker.get("global_step") != 15000:
        raise ValueError("Purifier checkpoint must be completed step 15000")
    if marker.get("condition_checkpoint_step") != 6000 or marker.get("condition_role") != "B":
        raise ValueError("Purifier conditioning does not match the released router")
    files += [purifier / "pytorch_model_fsdp.bin", purifier / "config.json",
              runtime / "vae/config.json", runtime / "vae/diffusion_pytorch_model.safetensors",
              runtime / "scheduler/scheduler_config.json", runtime / "runtime_manifest.json"]
    missing = [str(path) for path in files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))
    if not json.loads((runtime / "runtime_manifest.json").read_text()).get("complete"):
        raise ValueError("Runtime export is incomplete")
    return {"router_step": 6000, "purifier_step": 15000, "status": "ready"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--environment-only", action="store_true")
    group.add_argument("--weights-only", action="store_true")
    parser.add_argument("--blender", default=os.environ.get("GATE3DGEN_BLENDER"))
    args = parser.parse_args(argv)
    results = {}
    if not args.weights_only:
        from gate3dgen.evaluation.blender_runtime import default_blender
        # Load beside this entry point even when Python -I removes the script directory.
        spec = spec_from_file_location("gate3dgen_setup_environment", Path(__file__).with_name("environment.py"))
        environment = module_from_spec(spec)
        spec.loader.exec_module(environment)
        results["environment"] = environment.check(args.blender or default_blender())
    if not args.environment_only:
        try:
            results["weights"] = validate_checkpoints(
                Path(os.environ.get("GATE3DGEN_ROUTER_CHECKPOINT", "checkpoints/router/checkpoint-6000")),
                Path(os.environ.get("GATE3DGEN_PURIFIER_CHECKPOINT", "checkpoints/purifier/checkpoint-15000")),
                Path(os.environ.get("GATE3DGEN_RUNTIME_PATH", "checkpoints/runtime")),
            )
        except (OSError, ValueError, KeyError) as error:
            results["weights"] = {"status": "error", "error": str(error)}
    print(json.dumps(results, indent=2))
    return 0 if all(value.get("ok", value.get("status") == "ready") for value in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
