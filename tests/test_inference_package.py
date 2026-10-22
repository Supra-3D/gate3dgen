"""Self-contained release must never load original diffusion/Qwen tensor files."""
import pytest
import torch

from gate3dgen.training.inference import _load_full_transformer


def test_full_transformer_loads_owned_state_without_pretrained_weights(tmp_path, monkeypatch):
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    config = {"num_layers": 1, "num_attention_heads": 2, "attention_head_dim": 8,
              "in_channels": 4, "out_channels": 4, "joint_attention_dim": 16,
              "axes_dims_rope": (2, 2, 4)}
    original = QwenImageTransformer2DModel.from_config(config).to(dtype=torch.bfloat16)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    original.save_config(checkpoint)
    torch.save(original.state_dict(), checkpoint / "pytorch_model_fsdp.bin")
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not load a base transformer")
    monkeypatch.setattr(QwenImageTransformer2DModel, "from_pretrained", forbidden)
    loaded = _load_full_transformer(tmp_path / "ABSENT_BASE", checkpoint, "stock", "native", "cpu")
    assert not any(parameter.is_meta for parameter in loaded.parameters())
    for name, tensor in loaded.state_dict().items():
        assert torch.equal(tensor, original.state_dict()[name])
    state = original.state_dict()
    state.pop(next(iter(state)))
    torch.save(state, checkpoint / "pytorch_model_fsdp.bin")
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_full_transformer(tmp_path / "ABSENT_BASE", checkpoint, "stock", "native", "cpu")


