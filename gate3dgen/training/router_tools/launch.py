"""Launch the released router recipe with the original Accelerate FSDP settings."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def command_for(config: Path, resume: Path | None = None) -> list[str]:
    cfg = yaml.safe_load(config.read_text())
    train = cfg["training"]
    if cfg.get("variant") not in ("QWEN_VL_LM_GATE_V3_B_8GPU", "QWEN_VL_LM_GATE_V3_A_8GPU"):
        raise ValueError("This launcher is for the released 8-GPU router recipe")
    if (train["num_processes"], train["num_machines"]) != (8, 1):
        raise ValueError("The released router recipe uses one node and eight ranks")
    command = [sys.executable, "-m", "accelerate.commands.launch", "--mixed_precision", "bf16",
        "--num_processes", "8", "--num_machines", "1", "--machine_rank", "0",
        "--main_process_ip", "127.0.0.1", "--main_process_port", str(train["main_process_port"]),
        "--use_fsdp", "--fsdp_version", "1", "--fsdp_sharding_strategy", "SHARD_GRAD_OP",
        "--fsdp_auto_wrap_policy", "TRANSFORMER_BASED_WRAP",
        "--fsdp_transformer_layer_cls_to_wrap", "Qwen2_5_VLDecoderLayer",
        "--fsdp_backward_prefetch", "NO_PREFETCH", "--fsdp_forward_prefetch", "false",
        "--fsdp_state_dict_type", "SHARDED_STATE_DICT", "--fsdp_use_orig_params", "true",
        "--fsdp_cpu_ram_efficient_loading", "true", "--fsdp_sync_module_states", "true",
        "--module", "gate3dgen.training.router_tools.train", "--config", str(config.resolve())]
    if resume is not None:
        command.extend(["--resume", str(resume.resolve())])
    return command


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/router.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    command = command_for(args.config, args.resume)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        subprocess.run(command, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
