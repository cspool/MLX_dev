from __future__ import annotations

from mlxsim.lineage import (
    deduplicate_doi_records,
    evaluate_candidate_features,
    evaluate_exact_parent_gate,
    evaluate_family_gate,
    normalize_doi,
    scan_feature,
    title_identity,
)

FEATURES = {
    "chronology_ownership": [
        "predates_mlx",
        "ict_or_ricore_ownership",
        "overlapping_hardware_authors",
    ],
    "parent_hardware_fingerprints": [
        "taped_out_verilog",
        "process_12nm",
        "frequency_1ghz",
        "simd_width_32",
        "mesh_dimensions",
        "peak_about_1tops",
        "pe_array_area_7_712_mm2",
        "pe_array_power_5_846_w",
    ],
    "software_interface": [
        "risc_v_host",
        "per_pe_or_dataflow_assembly",
        "llvm_compiler",
        "spatial_assembler",
        "binary_header_configuration",
    ],
    "execution_substrate": [
        "programmable_spatial_dataflow_pes",
        "heterogeneous_functional_units",
        "decoupled_load_compute_transfer",
        "explicit_operand_routing",
        "data_reuse",
        "instruction_reuse_or_hierarchy",
    ],
    "mlx_specific_delta": [
        "closed_dependency_components",
        "tagged_instruction_blocks",
        "bounded_active_layer_window",
        "skip_hop_links",
        "semantic_fft_compression",
        "hierarchical_bsmm",
    ],
    "simulation": ["explicit_simict_use", "explicit_simict_derivation"],
}


def source(text: str) -> dict[str, object]:
    return {
        "source_id": "primary",
        "candidate_id": "candidate",
        "source_class": "publisher_full_text",
        "feature_eligible": True,
        "text": text,
    }


def test_doi_normalization_and_source_preserving_deduplication() -> None:
    assert normalize_doi("https://doi.org/10.1109/TPDS.2025.3555329.") == (
        "10.1109/tpds.2025.3555329"
    )
    records = deduplicate_doi_records(
        [
            {
                "candidate_id": "dfu_e",
                "source_id": "crossref",
                "doi": "10.1109/TPDS.2025.3555329",
            },
            {
                "candidate_id": "dfu_e",
                "source_id": "openalex",
                "doi": "https://doi.org/10.1109/tpds.2025.3555329",
            },
        ]
    )
    assert len(records) == 1
    assert records[0]["candidate_ids"] == ["dfu_e"]
    assert [item["source_id"] for item in records[0]["source_records"]] == [
        "crossref",
        "openalex",
    ]


def test_registered_title_alias_handles_institutional_typo() -> None:
    prefix = "M2-DFU: Multi-Mode Dataflow Architecture for Adaptive and"
    alias = f"{prefix} High-Efficency Data Processing"
    assert title_identity(
        "M2-DFU: Multi-Mode Dataflow Architecture for Adaptive and High-Efficency Data Processing",
        title=(
            "M2-DFU: Multi-Mode Dataflow Architecture for Adaptive and "
            "High-Efficiency Data Processing"
        ),
        aliases=(alias,),
    )


def test_generic_terms_do_not_match_high_specificity_features() -> None:
    generic = "RISC-V dataflow mesh reuse with a compiler and processing elements."
    for feature in (
        "risc_v_host",
        "programmable_spatial_dataflow_pes",
        "data_reuse",
        "llvm_compiler",
    ):
        assert scan_feature(generic, feature) is None


def test_family_gate_requires_both_software_and_substrate_classes() -> None:
    text = (
        "A RISC-V core acts as the host controller. The programmable spatial dataflow "
        "processing elements use data reuse in an on-chip buffer."
    )
    matrix = evaluate_candidate_features(
        candidate_id="candidate",
        source_texts=[source(text)],
        frozen_feature_classes=FEATURES,
    )
    gate = evaluate_family_gate(
        matrix,
        explicit_family_link=False,
        prior_tapeout_or_ownership=True,
        material_conflict=False,
        minimum_high_specificity_cross_class_matches=3,
    )
    assert gate["software_interface_matches"] == ["risc_v_host"]
    assert gate["execution_substrate_matches"] == [
        "data_reuse",
        "programmable_spatial_dataflow_pes",
    ]
    assert gate["pass"] is True

    substrate_only = evaluate_candidate_features(
        candidate_id="candidate",
        source_texts=[source("Data reuse in an on-chip buffer with instruction reuse mapping.")],
        frozen_feature_classes=FEATURES,
    )
    failed = evaluate_family_gate(
        substrate_only,
        explicit_family_link=False,
        prior_tapeout_or_ownership=True,
        material_conflict=False,
        minimum_high_specificity_cross_class_matches=2,
    )
    assert failed["checks"]["software_and_substrate_both_represented"] is False
    assert failed["pass"] is False


def test_exact_parent_gate_uses_target_specific_hardware_fields() -> None:
    text = (
        "Our taped-out chip is implemented in Verilog RTL at 12 nm and 1 GHz. "
        "Its SIMD32 array provides approximately 1 TOP/s."
    )
    matrix = evaluate_candidate_features(
        candidate_id="candidate",
        source_texts=[source(text)],
        frozen_feature_classes=FEATURES,
    )
    gate = evaluate_exact_parent_gate(
        matrix,
        explicit_primary_link=False,
        material_conflict=False,
        minimum_hardware_fingerprints=4,
        minimum_exact_numeric_fingerprints=2,
    )
    assert gate["pass"] is True
    assert set(gate["reported_exact_numeric_fingerprints"]) >= {
        "process_12nm",
        "frequency_1ghz",
        "simd_width_32",
    }


def test_unbound_mesh_dimensions_never_count_as_exact_numeric_match() -> None:
    matrix = evaluate_candidate_features(
        candidate_id="candidate",
        source_texts=[source("The architecture is arranged as an 8x8 mesh.")],
        frozen_feature_classes=FEATURES,
    )
    gate = evaluate_exact_parent_gate(
        matrix,
        explicit_primary_link=False,
        material_conflict=False,
        minimum_hardware_fingerprints=1,
        minimum_exact_numeric_fingerprints=1,
    )
    assert gate["reported_noncomparable_hardware_fields"] == ["mesh_dimensions"]
    assert gate["reported_exact_numeric_fingerprints"] == []
    assert gate["pass"] is False
