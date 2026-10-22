import pytest

from gate3dgen.training.lora_contract import assert_only_lora_trainable


class Parameter:
    def __init__(self, requires_grad):
        self.requires_grad = requires_grad


def test_mock_model_only_allows_lora_parameters_to_train():
    parameters = [
        ("base_model.weight", Parameter(False)),
        ("base_model.layer.lora_A.default.weight", Parameter(True)),
        ("base_model.layer.lora_B.default.weight", Parameter(True)),
    ]
    assert assert_only_lora_trainable(parameters) == 2


def test_mock_model_rejects_trainable_base_parameter():
    parameters = [("base_model.weight", Parameter(True))]
    with pytest.raises(RuntimeError, match="Non-LoRA"):
        assert_only_lora_trainable(parameters)

