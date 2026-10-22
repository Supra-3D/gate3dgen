"""Generate benchmark outputs with explicit routed or force-purify protocols."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import resource
from pathlib import Path

from gate3dgen.router import Router, DEFAULT_THRESHOLD
from .io import iter_jsonl, write_jsonl


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime", "--base-model", dest="base_model", type=Path,
                        default=Path(os.environ.get("GATE3DGEN_RUNTIME_PATH", "checkpoints/runtime")))
    parser.add_argument("--router-checkpoint", type=Path, default=Path("checkpoints/router/checkpoint-6000"))
    parser.add_argument("--purifier-checkpoint", type=Path, default=Path("checkpoints/purifier/checkpoint-15000"))
    parser.add_argument("--protocol", choices=("routed", "force-purify"), default="routed")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Record synchronized runtime, GPU peak memory and process peak RSS")
    args = parser.parse_args(argv)
    cases = list(iter_jsonl(args.manifest))
    ids = [str(row["case_id"]) for row in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Manifest must have unique, nonempty case IDs")
    for case in cases:
        name = str(case["case_id"])
        if name in {"", ".", ".."} or Path(name).name != name or "\\" in name:
            raise ValueError(f"Unsafe case ID: {name}")
        if not Path(case["degraded"]).is_file():
            raise FileNotFoundError(case["degraded"])
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new output directory to keep evaluation protocols distinct")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def timer():
        if args.profile and args.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize(args.device)
        return time.perf_counter()
    timings = {"router_inference_seconds": [], "purifier_inference_seconds": []}
    if args.profile:
        import platform
        from importlib.metadata import version, PackageNotFoundError
        from PIL import Image
        versions = {}
        for package in ("torch", "diffusers", "transformers", "accelerate", "safetensors"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        timings.update(python=platform.python_version(), packages=versions, device=args.device,
                       protocol=args.protocol, seed=args.seed, cases=[], timing_scope="cold single-process run; loading separate; purification includes output save")
        if args.device.startswith("cuda"):
            import torch
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(args.device)
            timings["cuda_runtime"] = torch.version.cuda
        for case in cases:
            with Image.open(case["degraded"]) as image:
                timings["cases"].append({"case_id": case["case_id"], "input_resolution": list(image.size)})
    started = timer()
    router = Router(args.router_checkpoint, args.base_model, args.device, args.threshold)
    timings["router_load_seconds"] = timer() - started
    decisions = []
    for row in cases:
        started = timer()
        decisions.append(router.predict(Path(row["degraded"])))
        timings["router_inference_seconds"].append(timer() - started)
    purifier = None
    results = []
    for case, decision in zip(cases, decisions):
        source = Path(case["degraded"])
        invoke = args.protocol == "force-purify" or decision["decision"] == "PURIFY"
        folder = args.output_dir / str(case["case_id"])
        folder.mkdir()
        target = folder / ("restored.png" if invoke else "restored" + source.suffix.lower())
        if invoke:
            from gate3dgen.purifier import Purifier
            if purifier is None:
                started = timer()
                purifier = Purifier(args.purifier_checkpoint, args.base_model, router, args.device, args.cpu_offload)
                timings["purifier_load_seconds"] = timer() - started
            started = timer()
            purifier.purify(source, args.seed).save(target)
            timings["purifier_inference_seconds"].append(timer() - started)
        else:
            shutil.copyfile(source, target)
        if args.profile:
            with Image.open(target) as image:
                timings["cases"][len(results)].update(output_resolution=list(image.size),
                                                     decision=decision["decision"], purifier_invoked=invoke)
        results.append({**case, "restored": str(target.resolve()),
                        "router": decision, "purifier_invoked": invoke,
                        "evaluation_protocol": args.protocol, "seed": args.seed})
        write_jsonl(args.output_dir / "manifest.jsonl", results)
    if args.profile:
        timings["peak_process_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        timings["cpu_offload"] = args.cpu_offload
        if args.device.startswith("cuda"):
            import torch
            timings.update(gpu_name=torch.cuda.get_device_name(args.device), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                           peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        (args.output_dir / "profile.json").write_text(json.dumps(timings, indent=2) + "\n")
    (args.output_dir / "run.json").write_text(json.dumps({
        "protocol": args.protocol, "threshold": args.threshold, "seed": args.seed,
        "router_checkpoint": str(args.router_checkpoint), "purifier_checkpoint": str(args.purifier_checkpoint),
        "cases": len(results), "status": "complete",
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
