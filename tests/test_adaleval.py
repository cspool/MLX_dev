import json
import math
from pathlib import Path

from mlxsim.adaleval import (
    aggregate_records,
    aggregate_replicate_records,
    build_stackselect_prompt,
    extract_stackselect_answer,
    fixed_replication_seed,
    load_stackselect,
    replication_seed_stream_sha256,
    select_length_quantiles,
    utf8_stream_sha256,
    validate_generation_result,
    wrap_internlm2_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_registered_prompt_streams_match_official_data() -> None:
    expected = {
        "1k": "87d6c9f4fdb1896fd4dbf1109a2730d4d48b0a041719466b56bb718f87b277fd",
        "2k": "361c7cdb52b0e64f3303a58fcbe7a140d25272018592dc5af3ac3eaa8db1c944",
        "4k": "571d27559a8cffa18d8061abc1df92795470a862dbe21892e192aaf551ae7906",
    }
    for setting, digest in expected.items():
        items = load_stackselect(
            PROJECT_ROOT / f"third_party/ada-leval/data/stackselect_{setting}.json"
        )
        prompts = [build_stackselect_prompt(item) for item in items]
        assert len(prompts) == 1000
        assert utf8_stream_sha256(prompts) == digest


def test_internlm2_wrapper_preserves_historical_trailing_newline() -> None:
    wrapped = wrap_internlm2_prompt("question")
    assert "English and 中文.\n<|im_end|>" in wrapped
    assert wrapped.endswith("<|im_start|>assistant\n")


def test_extractor_matches_upstream_highest_candidate_rule() -> None:
    assert extract_stackselect_answer("Answer: A2", 4) == "A2"
    assert extract_stackselect_answer("A2 or A4", 4) == "A4"
    assert extract_stackselect_answer("I choose 3", 4) == "A3"
    assert extract_stackselect_answer("nothing", 4) == "???"


def test_aggregate_records_checks_both_target_families() -> None:
    records = []
    for setting, correct_count in (("1k", 2), ("2k", 1), ("4k", 1)):
        for position in range(2):
            correct = position < correct_count
            records.append(
                {
                    "setting": setting,
                    "dataset_position": position,
                    "correct": correct,
                    "extracted": "A1" if correct else "???",
                }
            )
    report = aggregate_records(
        records,
        settings=["1k", "2k", "4k"],
        rows_per_setting=2,
        paper_targets={"1k": 100.0, "2k": 50.0, "4k": 50.0},
        official_targets={"1k": 100.0, "2k": 50.0, "4k": 50.0},
        tolerance=0.10,
    )
    assert report["paper_pass"] is True
    assert report["official_pass"] is True
    assert report["total_records"] == 6


def test_load_stackselect_adds_stable_index(tmp_path: Path) -> None:
    source = tmp_path / "stack.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question_id": 7,
                    "answer": "A2",
                    "question": "Q",
                    "all_answers": ["x", "y"],
                    "tags": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    assert load_stackselect(source)[0]["index"] == "7_A2"


def test_generation_result_validation_rejects_engine_garbage() -> None:
    validate_generation_result(
        text="A2",
        input_token_len=1126,
        generate_token_len=3,
        finish_reason="stop",
        expected_input_token_len=1126,
        max_new_tokens=512,
    )

    try:
        validate_generation_result(
            text="",
            input_token_len=1126,
            generate_token_len=525_663_136,
            finish_reason="length",
            expected_input_token_len=1126,
            max_new_tokens=512,
        )
    except ValueError as error:
        assert "outside valid bounds" in str(error)
    else:
        raise AssertionError("invalid TurboMind response was accepted")


def test_registered_replication_seed_stream_is_stable() -> None:
    namespace = "H31|internlm2-adaleval-4k"
    assert fixed_replication_seed(namespace, 0, 0) == 16_401_114_298_343_750_778
    assert fixed_replication_seed(namespace, 2, 999) == 16_069_441_306_084_588_667
    assert (
        replication_seed_stream_sha256(
            namespace, replicate_count=3, rows_per_replicate=1000
        )
        == "daa62375cb5a52782118a93beebf45ba21eb2544fb803e2a7484544a4bb945b0"
    )


def test_replicate_aggregate_uses_all_registered_draws() -> None:
    records = []
    for replicate, correct_positions in enumerate(({0, 1}, {0}, {1})):
        for position in range(2):
            correct = position in correct_positions
            records.append(
                {
                    "replicate": replicate,
                    "dataset_position": position,
                    "correct": correct,
                    "extracted": "A1" if correct else "A2",
                }
            )
    report = aggregate_replicate_records(
        records,
        replicate_count=3,
        rows_per_replicate=2,
        paper_target=100.0 * 4 / 6,
        official_target=100.0 * 4 / 6,
        tolerance=0.10,
    )
    assert report["total_correct"] == 4
    assert report["mean_accuracy_pct"] == 100.0 * 4 / 6
    assert math.isclose(
        report["sample_mean_accuracy_pct"],
        100.0 * 4 / 6,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert report["primary_pass"] is True
    assert report["unanimous_prediction_positions"] == 0
    assert report["unanimous_correctness_positions"] == 0


def test_length_quantile_selection_is_inclusive_and_stable() -> None:
    records = [
        {"input_token_len": length, "dataset_position": position}
        for position, length in enumerate((20, 10, 30, 20, 40))
    ]
    selected = select_length_quantiles(records, count=3)
    assert [record["dataset_position"] for record in selected] == [1, 3, 4]
    assert [record["input_token_len"] for record in selected] == [10, 20, 40]
