from scripts.compile_multiport_spad import transform


def test_transform_changes_only_backend() -> None:
    parent = {"memory_backend": "fixed", "routing": {"mesh_width": 4}}
    output = transform(parent)
    assert output["memory_backend"] == "dsagen_spad"
    output["memory_backend"] = "fixed"
    assert output == parent
