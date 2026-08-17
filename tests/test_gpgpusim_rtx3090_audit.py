from __future__ import annotations

from scripts.audit_gpgpusim_rtx3090_proxy import last_int


def test_last_int_uses_cumulative_final_metric() -> None:
    text = "gpu_tot_sim_cycle = 10\ngpu_tot_sim_cycle = 42\n"
    assert last_int(text, "gpu_tot_sim_cycle") == 42
