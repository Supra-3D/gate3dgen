import json
import argparse
from pathlib import Path

from gate3dgen.training.artifacts import checkpoint_is_complete, resolve_checkpoint


def _checkpoint(root: Path, step: int, complete: bool = True) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").touch()
    if complete:
        (checkpoint / "checkpoint_complete.json").write_text(
            json.dumps({"global_step": step, "world_size": 8}), encoding="utf-8"
        )
    return checkpoint


def test_latest_ignores_incomplete_checkpoint(tmp_path):
    expected = _checkpoint(tmp_path, 100)
    _checkpoint(tmp_path, 200, complete=False)
    assert resolve_checkpoint(tmp_path, "latest") == expected.resolve()


def test_explicit_adapter_file_is_supported(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.touch()
    assert resolve_checkpoint(tmp_path, str(weights)) == weights.resolve()


def test_full_transformer_checkpoint_is_complete(tmp_path):
    checkpoint = tmp_path / "checkpoint-30"
    checkpoint.mkdir()
    (checkpoint / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    (checkpoint / "checkpoint_complete.json").write_text(
        json.dumps({
            "global_step": 30, "world_size": 8,
            "artifact_kind": "transformer",
        }),
        encoding="utf-8",
    )
    assert checkpoint_is_complete(checkpoint)
    assert resolve_checkpoint(tmp_path, "latest", expected_kind="transformer") == checkpoint.resolve()


def test_fsdp_transformer_checkpoint_is_complete(tmp_path):
    checkpoint = tmp_path / "checkpoint-31"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model_fsdp.bin").write_bytes(b"weights")
    (checkpoint / "checkpoint_complete.json").write_text(
        json.dumps({
            "global_step": 31, "world_size": 4,
            "artifact_kind": "transformer",
        }),
        encoding="utf-8",
    )
    assert checkpoint_is_complete(checkpoint)
    assert resolve_checkpoint(tmp_path, "latest", expected_kind="transformer") == checkpoint.resolve()


def test_purifier_cpu_offload_is_explicit_and_default_recipe_unchanged():
    from gate3dgen.training.config import load_config
    from gate3dgen.training.cli import _train_command

    config = load_config("full", str(Path(__file__).resolve().parents[1] / "configs/train/purifier.yaml"))
    args = argparse.Namespace(num_processes=2, num_machines=1, dry_run=True)
    command = _train_command(args, config, None)
    assert "--fsdp_offload_params=false" in command
    config["training"]["fsdp_offload_params"] = True
    command = _train_command(args, config, None)
    assert "--fsdp_offload_params=true" in command
    assert "transformer_blocks.29." in command
    assert "transformer_blocks.49." in command
    assert command[command.index("--pixel_lpips_weight") + 1] == "0.5"
    assert "--fsdp_state_dict_type=FULL_STATE_DICT" in command
    config["training"]["fsdp_state_dict_type"] = "SHARDED_STATE_DICT"
    assert "--fsdp_state_dict_type=SHARDED_STATE_DICT" in _train_command(args, config, None)


def test_sharded_checkpoint_requires_every_model_and_optimizer_shard(tmp_path):
    checkpoint = tmp_path / "checkpoint-1"
    for dirname in ("pytorch_model_fsdp_0", "optimizer_0"):
        folder = checkpoint / dirname
        folder.mkdir(parents=True)
        (folder / ".metadata").write_bytes(b"metadata")
        for rank in range(2):
            (folder / f"__{rank}_0.distcp").write_bytes(b"weights")
    (checkpoint / "checkpoint_complete.json").write_text(json.dumps({
        "global_step": 1, "world_size": 2, "artifact_kind": "transformer",
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
    }))
    assert checkpoint_is_complete(checkpoint)
    assert resolve_checkpoint(tmp_path, "latest", expected_kind="transformer") == checkpoint.resolve()
    (checkpoint / "optimizer_0/__1_0.distcp").unlink()
    assert not checkpoint_is_complete(checkpoint)
