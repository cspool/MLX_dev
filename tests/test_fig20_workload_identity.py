from scripts.audit_fig20_workload_identity import logical_profiles


def test_logical_profiles_capture_rectangular_ffn() -> None:
    config = {
        "shape": {
            "sequence_lengths": [256, 8192],
            "batch": 1,
            "kernels": ["qkv", "attention", "ffn1", "ffn2"],
        }
    }
    profiles = logical_profiles(config)
    assert profiles["ffn1-N256"]["output_elements"] != profiles["ffn2-N256"][
        "output_elements"
    ]
    assert profiles["attention-N256"]["component_count"] == 2
