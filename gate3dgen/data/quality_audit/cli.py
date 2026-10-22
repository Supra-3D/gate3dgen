from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .common import (
    DATASET_ROOT, DINO_CHECKPOINT, DINO_REPO, OUTPUT_ROOT, PAIR_ROOT, PROJECT_ROOT,
    PROMPT_LIBRARY, QWEN_MODEL, QWEN_RUNNER, SAM_CHECKPOINT, SAM_CONFIG, SAM_REPO,
    THRESHOLDS, ADJUDICATION_OVERRIDES, ensure_output_within_project, iter_jsonl, load_thresholds,
)
from .finalize import finalize
from .programmatic import run_programmatic
from .vision import run_vision
from .vlm import run_vlm


def _default_qwen_python() -> Path:
    candidates = [Path(sys.executable)]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return Path(sys.executable)


def _default_sam_python_packages() -> Path | None:
    configured = os.environ.get("GATE3DGEN_SAM_PYTHON_PACKAGES")
    return Path(configured).expanduser() if configured else None


def _discover_gpus(value: str) -> list[str]:
    if value != "auto":
        result = [part.strip() for part in value.split(",") if part.strip()]
        if not result:
            raise ValueError("--gpus is empty")
        available_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], text=True
        )
        available = {line.strip() for line in available_output.splitlines() if line.strip()}
        missing = [gpu for gpu in result if gpu not in available]
        if missing:
            raise ValueError(
                f"requested GPU indices are not visible: {missing}; available={sorted(available)}"
            )
        return result
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        return [part.strip() for part in visible.split(",") if part.strip()]
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _manifest(args) -> Path:
    if args.manifest:
        return args.manifest
    if args.scope == "pilot":
        return args.pair_root / "manifests/pilot_pass.jsonl"
    return args.pair_root / "audit/audit_input.jsonl"


def command_check(args) -> int:
    paths = {
        "manifest": args.manifest, "pair_root": args.pair_root, "dataset_root": args.dataset_root,
        "thresholds": args.thresholds, "adjudication_overrides": args.adjudication_overrides,
        "prompt_library": args.prompt_library,
    }
    if args.vlm_backend == "local":
        paths.update({"qwen_model": args.qwen_model, "qwen_runner": args.qwen_runner, "qwen_python": args.qwen_python})
    if not args.skip_vision:
        paths.update({
            "dino_repo": args.dino_repo, "dino_checkpoint": args.dino_checkpoint,
            "sam_repo": args.sam_repo, "sam_checkpoint": args.sam_checkpoint,
        })
    problems = [f"missing {name}: {path}" for name, path in paths.items() if not path.exists()]
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
        import torch
        import torchvision  # noqa: F401
        if not torch.cuda.is_available():
            problems.append("torch.cuda.is_available() is false")
    except Exception as exc:
        problems.append(f"dependency import failed: {type(exc).__name__}: {exc}")
    try:
        gpus = _discover_gpus(args.gpus)
    except Exception as exc:
        problems.append(f"GPU discovery failed: {exc}")
        gpus = []
    result = {"status": "failed" if problems else "ok", "problems": problems, "gpus": gpus, "paths": {key: str(value) for key, value in paths.items()}}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if problems else 0


def _worker_command(args, command: str, shard_index: int, shard_count: int, vlm_base_url: str = "") -> list[str]:
    values = [
        sys.executable, "-m", "gate3dgen.data.quality_audit.cli", command,
        "--scope", args.scope, "--manifest", str(args.manifest), "--pair-root", str(args.pair_root),
        "--dataset-root", str(args.dataset_root), "--output-root", str(args.output_root),
        "--thresholds", str(args.thresholds), "--prompt-library", str(args.prompt_library),
        "--shard-index", str(shard_index), "--shard-count", str(shard_count),
        "--dino-repo", str(args.dino_repo), "--dino-checkpoint", str(args.dino_checkpoint),
        "--sam-repo", str(args.sam_repo), "--sam-checkpoint", str(args.sam_checkpoint),
        "--sam-config", args.sam_config, "--qwen-model", str(args.qwen_model),
        "--qwen-runner", str(args.qwen_runner), "--qwen-python", str(args.qwen_python),
        "--vlm-max-new-tokens", str(args.vlm_max_new_tokens), "--vlm-timeout", str(args.vlm_timeout),
        "--vlm-temperature", str(args.vlm_temperature),
        "--vlm-backend", str(args.vlm_backend),
        "--vlm-served-model", str(args.vlm_served_model),
        "--vlm-concurrency", str(args.vlm_concurrency),
        "--programmatic-mode", str(args.programmatic_mode),
    ]
    if vlm_base_url:
        values.extend(["--vlm-base-url", vlm_base_url])
    if args.sam_python_packages:
        values.extend(["--sam-python-packages", str(args.sam_python_packages)])
    if args.limit is not None:
        values.extend(["--limit", str(args.limit)])
    if args.force:
        values.append("--force")
    if args.skip_vision:
        values.append("--skip-vision")
    if args.vlm_all_pairs:
        values.append("--vlm-all-pairs")
    return [str(value) for value in values]


def _run_gpu_stage(args, command: str, gpus: list[str]) -> None:
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    try:
        endpoints = [value.strip() for value in str(args.vlm_base_urls).split(",") if value.strip()]
        if command == "vlm-worker" and args.vlm_backend == "vllm" and len(endpoints) != len(gpus):
            raise ValueError(f"vLLM endpoint count ({len(endpoints)}) must equal GPU worker count ({len(gpus)})")
        for index, gpu in enumerate(gpus):
            log = logs / f"{args.scope}_{command}_{index:02d}_gpu_{gpu}.log"
            handle = log.open("a", encoding="utf-8")
            handles.append(handle)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            endpoint = endpoints[index] if command == "vlm-worker" and args.vlm_backend == "vllm" else ""
            processes.append(subprocess.Popen(_worker_command(args, command, index, len(gpus), endpoint), env=environment, stdout=handle, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
        if any(code != 0 for code in codes):
            raise RuntimeError(f"{command} workers failed: {codes}; see {logs}")
    finally:
        for handle in handles:
            handle.close()


def command_all(args, thresholds) -> int:
    print(json.dumps({"stage": "programmatic", "result": run_programmatic(args, thresholds)}, ensure_ascii=False), flush=True)
    gpus = _discover_gpus(args.gpus)
    if not args.skip_vision:
        _run_gpu_stage(args, "vision-worker", gpus)
    else:
        print(json.dumps({"stage": "vision", "result": "skipped_by_policy"}, ensure_ascii=False), flush=True)
    _run_gpu_stage(args, "vlm-worker", gpus)
    summary, _ = finalize(args, thresholds)
    print(json.dumps({"stage": "finalize", "summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return 0


def command_status(args) -> int:
    # Status must stay responsive for the 98MB/47k-row full manifest. Prefer
    # the validated preparation summary, otherwise count newlines without
    # decoding every large JSON object.
    total = None
    prepare_summary = args.output_root / "inputs/prepare_summary.json"
    if prepare_summary.is_file():
        try:
            prepared = json.loads(prepare_summary.read_text(encoding="utf-8"))
            if Path(str(prepared.get("combined_manifest", ""))).resolve() == args.manifest.resolve():
                total = int(prepared["total"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            total = None
    if total is None:
        with args.manifest.open("rb") as handle:
            total = sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""))
    counts = {}
    for stage in ("programmatic", "vision", "vlm"):
        root = args.output_root / "records" / stage
        counts[stage] = sum(1 for _ in root.glob("*/*.json")) if root.exists() else 0
    summary_path = args.output_root / "reports/pair_quality_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    print(json.dumps({"manifest": str(args.manifest), "total": total, "stage_records": counts, "latest_summary": summary}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TexVerse GT/degraded quality audit")
    parser.add_argument("command", choices=("check", "programmatic", "vision-worker", "vlm-worker", "finalize", "report", "all", "status"))
    parser.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pair-root", type=Path, default=PAIR_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS)
    parser.add_argument("--adjudication-overrides", type=Path, default=ADJUDICATION_OVERRIDES)
    parser.add_argument("--prompt-library", type=Path, default=PROMPT_LIBRARY)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--cpu-workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--programmatic-mode", choices=("full", "crop_only"), default="full",
                        help="full audit, or fast full-dataset gate that only rejects crop tasks")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Do not run or require DINOv2/SAM vision records")
    parser.add_argument("--vlm-all-pairs", action="store_true",
                        help="Run VLM for every non-rejected pair, including hybrid tasks")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Recompute selected stage records even when versions match")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dino-repo", type=Path, default=DINO_REPO)
    parser.add_argument("--dino-checkpoint", type=Path, default=DINO_CHECKPOINT)
    parser.add_argument("--sam-repo", type=Path, default=SAM_REPO)
    parser.add_argument("--sam-checkpoint", type=Path, default=SAM_CHECKPOINT)
    parser.add_argument("--sam-config", default=SAM_CONFIG)
    parser.add_argument("--sam-python-packages", type=Path, default=_default_sam_python_packages())
    parser.add_argument("--qwen-model", type=Path, default=QWEN_MODEL)
    parser.add_argument("--qwen-runner", type=Path, default=QWEN_RUNNER)
    parser.add_argument("--qwen-python", type=Path, default=_default_qwen_python())
    parser.add_argument("--vlm-max-new-tokens", type=int, default=2400)
    parser.add_argument("--vlm-timeout", type=int, default=1800)
    parser.add_argument("--vlm-temperature", type=float, default=0.0,
                        help="Recorded generation temperature; current local runner is deterministic at 0")
    parser.add_argument("--vlm-backend", choices=("local", "vllm"), default="local")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:8100/v1",
                        help="OpenAI-compatible endpoint used by one VLM worker")
    parser.add_argument("--vlm-base-urls", default="",
                        help="Comma-separated endpoints, one for each GPU worker")
    parser.add_argument("--vlm-served-model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--vlm-concurrency", type=int, default=4,
                        help="Concurrent requests per vLLM endpoint")
    parser.add_argument("--run-token", default="", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_root = ensure_output_within_project(args.output_root)
    if not 0.0 <= args.vlm_temperature <= 2.0:
        raise SystemExit("--vlm-temperature must be in [0, 2]")
    if args.vlm_concurrency < 1:
        raise SystemExit("--vlm-concurrency must be >= 1")
    args.manifest = _manifest(args)
    thresholds = load_thresholds(args.thresholds)
    if args.command == "check":
        return command_check(args)
    if args.command == "programmatic":
        print(json.dumps(run_programmatic(args, thresholds), ensure_ascii=False, indent=2)); return 0
    if args.command == "vision-worker":
        print(json.dumps(run_vision(args, thresholds), ensure_ascii=False, indent=2)); return 0
    if args.command == "vlm-worker":
        print(json.dumps(run_vlm(args, thresholds), ensure_ascii=False, indent=2)); return 0
    if args.command in {"finalize", "report"}:
        summary, _ = finalize(args, thresholds); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0
    if args.command == "status":
        return command_status(args)
    return command_all(args, thresholds)


if __name__ == "__main__":
    raise SystemExit(main())
