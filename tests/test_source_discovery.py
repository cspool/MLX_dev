from mlxsim.source_discovery import (
    critical_domain_matches,
    crossref_artifact_links,
    evaluate_artifact_candidate,
    exact_paper_identity,
    normalize_text,
)

TITLE = (
    "MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures"
)
AUTHORS = ["Haibin Wu", "Wenming Li", "Jian Weng"]


def test_identity_normalizes_markup_and_punctuation() -> None:
    text = (
        "<b>MLX — Multi-Layer Execution for Structured LLM Workload "
        "Acceleration on Spatial Architectures</b>; Haibin Wu"
    )
    report = exact_paper_identity(text, title=TITLE, authors=AUTHORS)
    assert report["pass"] is True
    assert report["matched_authors"] == ["Haibin Wu"]
    assert normalize_text("FFT-CMP") == "fft cmp"


def test_unrelated_apple_mlx_does_not_match_paper_identity() -> None:
    report = exact_paper_identity(
        "Apple MLX is an array framework for Apple silicon by Awni Hannun.",
        title=TITLE,
        authors=AUTHORS,
    )
    assert report["pass"] is False
    assert report["checks"]["title"] is False


def test_critical_domains_are_exposed_as_term_matches() -> None:
    matches = critical_domain_matches(
        text="Cycle-accurate simulator and LoRA checkpoint",
        paths=["rtl/top.sv", "results/nsight_trace.json"],
    )
    assert "simulator" in matches["architecture_simulator_rtl_mapping"]
    assert "lora" in matches["structured_operator_model_training"]
    assert "checkpoint" in matches["dataset_evaluator_checkpoint_manifest"]
    assert "nsight" in matches["native_trace_raw_measurement"]


def test_candidate_requires_every_registered_gate() -> None:
    candidate = {
        "exact_paper_identity": True,
        "anonymous_retrieval": True,
        "stable_identifier": True,
        "critical_domains": ["architecture_simulator_rtl_mapping"],
        "license_and_dependencies_recorded": True,
        "excluded_noise": False,
    }
    evaluated = evaluate_artifact_candidate(candidate)
    assert evaluated["pass"] is True
    assert evaluate_artifact_candidate(evaluated) == evaluated
    candidate["critical_domains"] = []
    assert evaluate_artifact_candidate(candidate)["pass"] is False


def test_crossref_pdf_is_not_mislabeled_as_artifact() -> None:
    record = {
        "link": [
            {"URL": "https://example.test/paper.pdf", "content-type": "application/pdf"},
            {"URL": "https://example.test/code", "content-type": "text/html"},
        ],
        "relation": {"is-supplemented-by": [{"id": "10.1/archive"}]},
    }
    links = crossref_artifact_links(record)
    assert links == [
        {"URL": "https://example.test/code", "content-type": "text/html"},
        {"relation": "is-supplemented-by", "id": "10.1/archive"},
    ]
