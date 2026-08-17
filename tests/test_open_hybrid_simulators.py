from __future__ import annotations

from scripts import audit_open_hybrid_simulators as audit


def test_accelsim_parser_uses_final_cumulative_values() -> None:
    text = """
Processing kernel /tmp/kernel-1.traceg
gpu_tot_sim_cycle = 7706
gpu_tot_sim_insn = 6471680
gpu_tot_issued_cta = 256
Processing kernel /tmp/kernel-2.traceg
gpu_tot_sim_cycle = 14903
gpu_tot_sim_insn = 9290080
gpu_tot_issued_cta = 512
wall_seconds=10.932
"""
    metrics = audit.parse_accelsim_metrics(text)
    assert metrics["cumulative_cycles"] == 14903
    assert metrics["cumulative_instructions"] == 9290080
    assert metrics["cumulative_ctas"] == 512
    assert metrics["kernel_count"] == 2
    assert metrics["wall_seconds"] == 10.932


def test_dsagen_parser_requires_real_cgra_and_dma_work() -> None:
    text = """
Cycles: 569
CGRA Instances: 256 -- Activity Ratio: 0.001382
CGRA Insts / Cycle: 1024 / 569 = 1.8
Read DMA:\t16384 B (28.79 B/c, 64 B/r) 256
Write DMA:\t8192 B (14.4 B/c, 61.59 B/r) 133
[single-core] sanity check passed successfully!
Exiting @ tick 114277000 because exiting with last active thread context
"""
    metrics = audit.parse_dsagen_metrics(text)
    assert metrics == {
        "cycles": 569,
        "cgra_instances": 256,
        "cgra_instructions": 1024,
        "dma_read_bytes": 16384,
        "dma_write_bytes": 8192,
        "exit_tick": 114277000,
        "sanity_check_passed": True,
        "simulated_exit_code_nonzero": False,
    }


def test_marker_audit_reports_each_missing_marker() -> None:
    assert audit.required_markers("alpha beta", ["alpha", "gamma"]) == {
        "alpha": True,
        "gamma": False,
    }


def test_line_number_is_one_based() -> None:
    assert audit.line_number("first\nsecond token\n", "token") == 2
    assert audit.line_number("first\n", "absent") is None
