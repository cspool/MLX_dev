from pathlib import Path

from mlxsim.identifiability import (
    fft_ambiguity_witness,
    layer_plan_combinatorics,
    line_segment_bytes,
    missing_field_audit,
)


def test_line_segment_bytes_uses_inclusive_one_indexed_lines(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    assert line_segment_bytes(source, 2, 3) == b"beta\ngamma\n"


def test_fft_witness_distinguishes_interpretations_and_future_input() -> None:
    report = fft_ambiguity_witness(
        chunk_length=32,
        compression_ratio=0.75,
        perturbation_index=31,
        perturbation_delta=1.0,
    )
    assert report["compressed_length"] == 24
    assert report["changed_earlier_positions"] > 0
    assert report["maximum_earlier_absolute_change"] > 1e-12
    assert report["literal_prefix_maximum_imaginary"] > 1e-12
    assert report["interpretation_maximum_absolute_difference"] > 1e-12


def test_layer_plan_combinatorics_matches_registered_canaries() -> None:
    report = layer_plan_combinatorics(
        total_layers=32,
        minimum_modified_layers=20,
        chunk_lengths=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    assert report["admissible_layer_subsets"] == 462_411_533
    assert report["minimum_chunk_assignments_at_minimum_layers"] == 10**20


def test_missing_field_audit_requires_each_registered_domain() -> None:
    fields = [
        {"id": "first", "domain": "fft", "disclosed": False},
        {"id": "second", "domain": "training", "disclosed": False},
    ]
    assert missing_field_audit(fields, ["fft", "training"])["pass"] is True
    assert missing_field_audit(fields, ["fft", "bsmm"])["pass"] is False
