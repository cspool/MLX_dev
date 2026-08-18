import yaml

from mlxsim.pipelined_block_contexts import scenarios
from scripts.audit_pipelined_block_contexts import DEFAULT_CONFIG, build_audit


def test_pipelined_block_context_documents() -> None:
    documents = scenarios()
    assert len(documents) == 10
    assert documents["fma_ii1_ctx4"]["pe_dependency_model"] == "dpu_pipelined"
    assert (
        documents["fma_ii1_ctx4"]["dpu"]["iteration_contexts_per_block"] == 4
    )
    assert (
        documents["fma_ii1_ctx2"]["dpu"]["iteration_contexts_per_block"] == 2
    )
    assert (
        documents["fma_ii2_ctx4"]["functional_units"]["fma"][
            "initiation_interval"
        ]
        == 2
    )


def test_pipelined_block_context_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["executions"] == 60
    assert report["summary"]["sanitizer_executions"] == 20
    assert report["summary"]["acceptance_gates_passed"] == 12
    assert report["summary"]["fma_ii1_issue_cycles"] == list(range(8))
    assert report["summary"]["fma_ii1_complete_cycles"] == list(range(4, 12))
    assert report["summary"]["fma_ii1_total_cycles"] == 12
    assert report["summary"]["limited_context_total_cycles"] == 18
    assert report["summary"]["legacy_regressions_exact"]
    assert report["patch_checks"]["h108_source_exact"]
    assert report["paper_reproduction_claim"].startswith("none_")

