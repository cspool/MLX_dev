"""High-offset CPU/device data flow, not full-model or SoC memory validation."""
import json
import subprocess

import pytest

from test_spike_graph_runtime import ROOT, graph, native
from system_sim.physical_host.graph_lowering import compile_graph
from scripts.run_mlx_spike_graph import check_embedded_payload_limit


@pytest.mark.parametrize("options",[
    {"device_bytes":1048577}, {"device_bytes":4096},
    {"device_base":2**40}, {"device_bytes":2**40},
])
def test_mapping_limits_match_sparse_plugin(graph,options):
    _,program,life=graph
    with pytest.raises(ValueError,match="mapping invalid"):
        compile_graph(program,life,**options)


def test_sparse_capacity_does_not_authorize_large_embedded_asset_loading():
    program={"assets":{"weight":{"dtype":"f16","shape":[6738415616],"kind":"mapped_file"}}}
    with pytest.raises(RuntimeError,match="not a full-model asset loader"):
        check_embedded_payload_limit(program,33286912)
    assert check_embedded_payload_limit({"assets":{"a":{"dtype":"bool","shape":[1]},"b":{"dtype":"f16","shape":[2]}}},0)==12


@pytest.mark.parametrize("high",[False,True])
def test_actual_rv64_graph_in_sparse_16gib_aperture(graph,tmp_path,high):
    original,_,_=graph
    out=tmp_path/"run";offset=2**32+65536 if high else 65536
    command=[str(ROOT/".venv/bin/python"),"-m","scripts.run_mlx_spike_graph",
        "--program",str(original/"program.json"),"--lifetimes",str(original/"life.json"),
        "--reference",str(original/"out/result.json"),"--output",str(out),
        "--device-bytes",str(16*2**30),"--data-offset",str(offset)]
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=300)
    assert process.returncode==0,process.stdout+process.stderr
    report=json.loads((out/"report.json").read_text());plan=json.loads((out/"plan.json").read_text())
    assert report["source_calls"]==45 and report["device_windows"]==21
    assert report["actual_cpu_dispatch"] and report["output_bytes_checked_in_elf"]
    assert report["device_capacity_bytes"]==16*2**30 and report["data_offset"]==offset
    assert all(b["base"]>=2**32+offset for b in plan["assets"].values())
    backing=report["host_memory_backing"]
    assert 0<backing["resident_pages"]<32
    assert backing["writer_pages"]<backing["resident_pages"]
    assert not report["full_model_execution_verified"] and not report["mlx_system_verified"]
