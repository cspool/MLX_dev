from mlxsim.fig21_layer_contract import elementwise_signature


def test_elementwise_signature_is_shape_linear() -> None:
    first = elementwise_signature(
        sequence_length=128,
        batch=8,
        hidden_dimension=4096,
        ffn_dimension=11008,
    )
    second = elementwise_signature(
        sequence_length=256,
        batch=8,
        hidden_dimension=4096,
        ffn_dimension=11008,
    )
    for name, value in first["fu_instruction_instances"].items():
        assert second["fu_instruction_instances"][name] == 2 * value


def test_elementwise_contract_has_all_fu_classes() -> None:
    signature = elementwise_signature(
        sequence_length=128,
        batch=8,
        hidden_dimension=4096,
        ffn_dimension=11008,
    )
    assert set(signature["fu_instruction_instances"]) == {
        "mul",
        "add",
        "frsqrt",
        "fexp",
        "fdiv",
        "shuffle",
    }
