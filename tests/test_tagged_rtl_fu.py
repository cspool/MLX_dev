import json
import subprocess

import pytest

from scripts.run_mlx_tagged_fu import build_fu, build_scalar, verify_constants


def test_exp_range_reduction_table_is_mathematical_not_workload_specific():
    verify_constants()


def test_scalar_fp16_all_exp_inputs_and_binary_anchor_operands(tmp_path):
    result = subprocess.run(
        [str(build_scalar()), str(tmp_path / "numeric.json"), "5000"],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    assert "MLX_TAGGED_FP16_PASS" in result.stdout
    data = json.loads((tmp_path / "numeric.json").read_text())
    assert data["exp_input_encodings"] == 65536
    assert data["mismatches"] == 0
    assert all(data["checked"][op] >= 65536 * 17 for op in ("add", "mul", "div", "max", "fma"))


@pytest.mark.parametrize("lanes", [4, 32])
def test_functional_fu_vector_lanes_latency_ii_backpressure_and_reset(tmp_path, lanes):
    result = subprocess.run(
        [str(build_fu(lanes)), str(tmp_path / "vector.json")],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "MLX_TAGGED_FU_PASS" in result.stdout
    data = json.loads((tmp_path / "vector.json").read_text())
    assert data["lanes"] == lanes
    assert data["mismatches"] == 0
    assert data["held_response_cycles"] > 0
    assert data["initiation_intervals"] == [1, 17]
