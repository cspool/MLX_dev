import json

import yaml

from mlxsim.fig23_complete_block import compile_complete_block_scaling
from scripts.audit_fig23_complete_block import build_audit
from scripts.compile_fig23_complete_block import DEFAULT_CONFIG, PROJECT_ROOT


def test_complete_block_scaling_conserves_work_and_events() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    base = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h48_fixed"]["path"]).read_text())
    documents = {}
    metadata = {}
    for name, hardware in config["robustness_grid"]["configurations"].items():
        document, item = compile_complete_block_scaling(
            base,
            sequence_length=512,
            hidden_dimension=512,
            batch=8,
            active_window=2,
            baseline_repeat=16,
            hardware_name=name,
            simd_width=hardware["simd_width"],
            mesh=tuple(hardware["mesh"]),
        )
        documents[name] = document
        metadata[name] = item
    reference = metadata["baseline"]["work"]
    for name, item in metadata.items():
        assert item["stage_count"] == 28
        assert item["event_checks"] == {
            "unique_emitters": True,
            "all_waits_resolved": True,
            "adjacent_tags": True,
        }
        assert set(item["operation_classes"]) == {
            "add",
            "mul",
            "fma",
            "fmax",
            "fexp",
            "fdiv",
            "frsqrt",
            "shuffle",
        }
        for key, value in item["work"].items():
            if key.startswith("scalarized_"):
                assert value == reference[key], name
        assert documents[name]["active_window"] == 2


def test_complete_block_scaling_is_deterministic() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    base = json.loads((PROJECT_ROOT / config["frozen_inputs"]["h48_fixed"]["path"]).read_text())
    kwargs = {
        "sequence_length": 8192,
        "hidden_dimension": 512,
        "batch": 8,
        "active_window": 4,
        "baseline_repeat": 256,
        "hardware_name": "simd32_8x8",
        "simd_width": 32,
        "mesh": (8, 8),
    }
    assert compile_complete_block_scaling(base, **kwargs) == compile_complete_block_scaling(
        base, **kwargs
    )


def test_complete_block_scaling_audit() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    report = build_audit(config)
    assert report["audit_integrity"]
    assert report["hypothesis_status"] == "supported"
    assert report["summary"]["compiled_configs"] == 40
    assert report["summary"]["executions"] == 120
    assert report["summary"]["individual_speedup_passes"] == 20
    assert report["summary"]["joint_speedup_passes"] == 10
    assert report["summary"]["minimum_simd_speedup"] > 1.2
    assert report["summary"]["minimum_mesh_speedup"] > 1.2
    assert report["summary"]["minimum_joint_speedup"] > 1.2
    assert report["summary"]["all_work_conserved"]
    assert report["summary"]["all_builds_identical_and_clean"]
    assert report["summary"]["figure23_target_join_eligible"]
    assert report["summary"]["acceptance_gates_passed"] == 10
