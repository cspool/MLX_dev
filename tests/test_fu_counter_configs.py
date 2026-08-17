from scripts.compile_fu_counter_configs import transform


def test_backend_transform_is_exact() -> None:
    parent = {"memory_backend": "dsagen_dma", "blocks": [{"tag": 1}]}
    output = transform(parent, "fixed")
    assert output["memory_backend"] == "fixed"
    output["memory_backend"] = parent["memory_backend"]
    assert output == parent
