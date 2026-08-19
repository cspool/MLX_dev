import yaml

from mlxsim.model_frontend import (
    build_onnx_model,
    canonical_signature,
    import_fx_graph,
    import_onnx_graph,
    plan_graph,
)
from scripts.audit_automatic_model_frontend import DEFAULT_CONFIG, build_audit


def test_fx_and_onnx_import_to_same_canonical_graph() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    model = config["model_contract"]
    shape = tuple(model["input_shape"])
    fx_graph, _ = import_fx_graph(
        input_shape=shape, hidden_dimension=int(model["hidden_dimension"]), seed=190
    )
    onnx_graph = import_onnx_graph(
        build_onnx_model(
            input_shape=shape, hidden_dimension=int(model["hidden_dimension"]), seed=190
        )
    )
    assert canonical_signature(fx_graph) == canonical_signature(onnx_graph)
    assert [node["kind"] for node in fx_graph["nodes"]] == model["canonical_kinds"]
    plan = plan_graph(fx_graph, config["planning"])
    assert len(plan["nodes"]) == 6
    assert plan["peak_spm_bytes"] <= config["planning"]["spm_bytes"]


def test_automatic_model_frontend_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["frontends"] == 2
    assert report["summary"]["nodes_per_frontend"] == 6
    assert report["summary"]["total_source_nodes"] == 12
    assert report["summary"]["canonical_matches"] == 6
    assert report["summary"]["lineage_entries"] == 12
    assert report["summary"]["profiles"] == 12
    assert report["summary"]["executions"] == 24
    assert report["summary"]["execution_replays"] == 12
    assert report["summary"]["automatic_model_frontend_complete"]
