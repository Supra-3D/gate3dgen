from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .common import (
    DEFAULT_DATASET_ROOT, DEFAULT_INPUT, DEFAULT_LIBRARY, DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_ROOT,
    atomic_write_json, file_sha256,
)
from .generation import run_worker, utc_now
from .planner import create_plan, load_single_tasks
from .reporting import collect_report, write_report


def discover_gpus(value: str) -> list[str]:
    if value != "auto":
        result = [item.strip() for item in value.split(",") if item.strip()]
        if not result:
            raise ValueError("--gpus is empty")
        return result
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible not in {"-1", "none", "None"}:
        return [item.strip() for item in visible.split(",") if item.strip()]
    command = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True)
    result = [line.strip() for line in output.splitlines() if line.strip()]
    if not result:
        raise RuntimeError("No visible NVIDIA GPUs")
    return result


def command_check(args) -> int:
    problems = []
    for path, label in ((args.input, "input manifest"), (args.prompt_library, "prompt library"), (args.dataset_root, "dataset root"), (args.model_path, "FireRed model")):
        if path is None or not path.exists():
            problems.append(f"missing {label}: {path}")
    if args.model_path and args.model_path.exists():
        if not ((args.model_path / "model_index.json").is_file() or (args.model_path / "config.json").is_file()):
            problems.append(f"model has no model_index.json/config.json: {args.model_path}")
        weights = list(args.model_path.rglob("*.safetensors")) + list(args.model_path.rglob("*.bin"))
        if not weights or any(path.stat().st_size == 0 for path in weights):
            problems.append(f"model weights are missing or empty: {args.model_path}")
        for index_path in args.model_path.rglob("*.index.json"):
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                referenced = set(index.get("weight_map", {}).values())
                missing = [name for name in referenced if not (index_path.parent / name).is_file()]
                if missing:
                    problems.append(f"model index {index_path} references {len(missing)} missing shards")
            except Exception as exc:
                problems.append(f"invalid model weight index {index_path}: {exc}")
    try:
        import torch
        import diffusers  # noqa: F401
        import numpy  # noqa: F401
        import PIL  # noqa: F401
        if not torch.cuda.is_available():
            problems.append("torch.cuda.is_available() is false")
    except Exception as exc:
        problems.append(f"dependency import failed: {type(exc).__name__}: {exc}")
    try:
        tasks, _, _ = load_single_tasks(args.prompt_library)
        if len(tasks) != 12:
            problems.append(f"expected 12 single tasks, got {len(tasks)}")
    except Exception as exc:
        problems.append(f"prompt library validation failed: {exc}")
    try:
        gpus = discover_gpus(args.gpus)
    except Exception as exc:
        problems.append(f"GPU discovery failed: {exc}")
        gpus = []
    result = {"status": "failed" if problems else "ok", "problems": problems, "gpus": gpus, "model_path": str(args.model_path) if args.model_path else None}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if problems else 0


def _runtime_paths(output_root: Path):
    return output_root / "state/pipeline.lock", output_root / "state/pipeline.pid.json"


def _process_matches(pid: int, token: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "gate3dgen.data.pair_generation.pipeline" in command and token in command


def _acquire_runtime(output_root: Path, token: str, scope: str) -> tuple[Path, Path]:
    lock_path, pid_path = _runtime_paths(output_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        existing = None
        try:
            existing = json.loads(pid_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        if existing and _process_matches(int(existing.get("pid", -1)), str(existing.get("token", ""))):
            raise RuntimeError(f"Pipeline is already running: pid={existing['pid']}")
        lock_path.unlink(missing_ok=True)
        pid_path.unlink(missing_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(descriptor)
    runtime = {"pid": os.getpid(), "pgid": os.getpgrp(), "token": token, "scope": scope, "started_at": utc_now()}
    atomic_write_json(pid_path, runtime)
    return lock_path, pid_path


def command_run(args) -> int:
    plan_path = args.output_root / "plans" / ("pilot.jsonl" if args.scope == "pilot" else "master.jsonl")
    if not plan_path.is_file():
        raise FileNotFoundError(f"Run plan first; missing {plan_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)
    gpus = discover_gpus(args.gpus)
    lock_path, pid_path = _acquire_runtime(args.output_root, args.run_token, args.scope)
    children: list[subprocess.Popen] = []
    log_handles = []
    stopping = False

    def stop_children(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True
        for child in children:
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)
    try:
        logs = args.output_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for shard_index, gpu in enumerate(gpus):
            log_path = logs / f"{args.scope}_worker_{shard_index:02d}_gpu_{gpu}.log"
            handle = log_path.open("a", encoding="utf-8")
            log_handles.append(handle)
            command = [
                sys.executable, "-m", "gate3dgen.data.pair_generation.pipeline", "worker",
                "--plan", str(plan_path), "--output-root", str(args.output_root),
                "--dataset-root", str(args.dataset_root), "--prompt-library", str(args.prompt_library),
                "--model-path", str(args.model_path), "--shard-index", str(shard_index),
                "--shard-count", str(len(gpus)), "--gpu-label", gpu,
                "--steps", str(args.steps), "--cfg-scale", str(args.cfg_scale),
                "--max-attempts", str(args.max_attempts),
            ]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            children.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment))
        exit_codes = []
        while children:
            exit_codes = [child.poll() for child in children]
            if all(code is not None for code in exit_codes):
                break
            time.sleep(2)
        summary = write_report(plan_path, args.output_root)
        print(json.dumps({"workers": exit_codes, "report": summary}, ensure_ascii=False, indent=2), flush=True)
        return 130 if stopping else (1 if any(code != 0 for code in exit_codes) else 0)
    finally:
        stop_children()
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
        for handle in log_handles:
            handle.close()
        lock_path.unlink(missing_ok=True)
        pid_path.unlink(missing_ok=True)


def command_status(args) -> int:
    _, pid_path = _runtime_paths(args.output_root)
    runtime = None
    try:
        runtime = json.loads(pid_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    running = bool(runtime and _process_matches(int(runtime.get("pid", -1)), str(runtime.get("token", ""))))
    active_scope = runtime.get("scope") if running else "full"
    plan_name = "pilot.jsonl" if active_scope == "pilot" else "master.jsonl"
    plan_path = args.output_root / "plans" / plan_name
    progress = None
    if plan_path.is_file():
        progress, _ = collect_report(plan_path, args.output_root)
    print(json.dumps({"running": running, "active_scope": active_scope, "runtime": runtime, "progress": progress}, ensure_ascii=False, indent=2))
    return 0


def command_stop(args) -> int:
    lock_path, pid_path = _runtime_paths(args.output_root)
    try:
        runtime = json.loads(pid_path.read_text(encoding="utf-8"))
    except Exception:
        print("Pipeline is not running")
        lock_path.unlink(missing_ok=True)
        return 0
    pid, pgid, token = int(runtime["pid"]), int(runtime["pgid"]), str(runtime["token"])
    if not _process_matches(pid, token):
        print("Removed stale pipeline state; recorded process does not match")
        lock_path.unlink(missing_ok=True)
        pid_path.unlink(missing_ok=True)
        return 0
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + args.stop_timeout
    while time.monotonic() < deadline and _process_matches(pid, token):
        time.sleep(0.5)
    if _process_matches(pid, token):
        os.killpg(pgid, signal.SIGKILL)
    lock_path.unlink(missing_ok=True)
    pid_path.unlink(missing_ok=True)
    print(f"Stopped pipeline process group {pgid}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TexVerse FireRed GT/degradation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    common.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    common.add_argument("--prompt-library", type=Path, default=DEFAULT_LIBRARY)
    check = sub.add_parser("check", parents=[common])
    check.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    check.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    check.add_argument("--gpus", default="auto")
    plan = sub.add_parser("plan", parents=[common])
    plan.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--skip-image-verify", action="store_true")
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--scope", choices=("pilot", "full"), required=True)
    run.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    run.add_argument("--gpus", default="auto")
    run.add_argument("--steps", type=int, default=40)
    run.add_argument("--cfg-scale", type=float, default=4.0)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--limit", type=int, help="Process at most this many plan rows per worker; intended for smoke tests")
    run.add_argument("--run-token", default=lambda: uuid.uuid4().hex)
    worker = sub.add_parser("worker", parents=[common])
    worker.add_argument("--plan", type=Path, required=True)
    worker.add_argument("--model-path", type=Path, required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--shard-count", type=int, required=True)
    worker.add_argument("--gpu-label", required=True)
    worker.add_argument("--steps", type=int, default=40)
    worker.add_argument("--cfg-scale", type=float, default=4.0)
    worker.add_argument("--max-attempts", type=int, default=3)
    worker.add_argument("--limit", type=int)
    sub.add_parser("status", parents=[common])
    stop = sub.add_parser("stop", parents=[common])
    stop.add_argument("--stop-timeout", type=float, default=30.0)
    report = sub.add_parser("report", parents=[common])
    report.add_argument("--scope", choices=("pilot", "full"), default="full")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "check":
        return command_check(args)
    if args.command == "plan":
        summary = create_plan(args.input, args.prompt_library, args.dataset_root, args.output_root, seed=args.seed, verify_images=not args.skip_image_verify)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        if callable(args.run_token):
            args.run_token = args.run_token()
        return command_run(args)
    if args.command == "worker":
        run_worker(
            args.plan, args.output_root, args.dataset_root, args.prompt_library, args.model_path,
            shard_index=args.shard_index, shard_count=args.shard_count, gpu_label=args.gpu_label,
            steps=args.steps, cfg_scale=args.cfg_scale, max_attempts=args.max_attempts,
            limit=args.limit,
        )
        return 0
    if args.command == "status":
        return command_status(args)
    if args.command == "stop":
        return command_stop(args)
    if args.command == "report":
        plan_path = args.output_root / "plans" / ("pilot.jsonl" if args.scope == "pilot" else "master.jsonl")
        print(json.dumps(write_report(plan_path, args.output_root), ensure_ascii=False, indent=2))
        return 0
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
