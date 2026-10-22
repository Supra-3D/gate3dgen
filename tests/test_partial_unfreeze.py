from gate3dgen.training.training_contract import freeze_first_by_name


class Parameter:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def numel(self):
        return 4


class TinyTransformer:
    def __init__(self):
        self.parameters = {
            **{f"transformer_blocks.{index}.weight": Parameter() for index in range(4)},
            "proj_out.weight": Parameter(),
        }

    def requires_grad_(self, value):
        for parameter in self.parameters.values():
            parameter.requires_grad_(value)

    def named_parameters(self):
        return self.parameters.items()


def test_partial_unfreeze_freezes_everything_before_matching():
    model = TinyTransformer()
    trainable, frozen = freeze_first_by_name(
        model, ["transformer_blocks.1.", "transformer_blocks.2."]
    )
    assert trainable > 0 and frozen > 0
    for name, parameter in model.named_parameters():
        expected = "transformer_blocks.1." in name or "transformer_blocks.2." in name
        assert parameter.requires_grad is expected


def test_released_zero_based_range_and_projection_scope():
    model = TinyTransformer()
    model.parameters = {
        **{f"transformer_blocks.{index}.attn.{projection}.weight": Parameter()
           for index in range(60) for projection in ("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")},
        **{f"{name}.weight": Parameter() for name in ("txt_norm", "txt_in", "img_in", "time_text_embed", "norm_out", "proj_out")},
    }
    freeze_first_by_name(model, [f"transformer_blocks.{index}." for index in range(29, 50)])
    for name, parameter in model.named_parameters():
        expected = name.startswith("transformer_blocks.") and 29 <= int(name.split(".")[1]) <= 49
        assert parameter.requires_grad is expected


def test_parameter_export_records_all_sixty_blocks(tmp_path, monkeypatch):
    import json
    import pytest
    torch = pytest.importorskip("torch")
    from gate3dgen.training.training_contract import write_trainable_parameters
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(60)])
    model.txt_in = torch.nn.Linear(2, 2)
    freeze_first_by_name(model, [f"transformer_blocks.{index}." for index in range(29, 50)])
    monkeypatch.setenv("RANK", "0")
    write_trainable_parameters(model, tmp_path)
    result = json.loads((tmp_path / "trainable_parameters.json").read_text())
    assert result["transformer_block_count"] == 60
    assert result["blocks_with_trainable_parameters"] == list(range(29, 50))
    assert result["blocks_with_no_trainable_parameters"] == list(range(29)) + list(range(50, 60))
    assert not any(row["name"].startswith("txt_in.") for row in result["parameters"])
