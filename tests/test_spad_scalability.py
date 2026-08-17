from scripts.compile_spad_scalability import transform


def test_transform_changes_only_memory_backend() -> None:
    parent = {"memory_backend": "fixed", "blocks": [{"trip_count": 4}]}
    output = transform(parent)
    assert output["memory_backend"] == "dsagen_spad"
    output["memory_backend"] = "fixed"
    assert output == parent
