import copy
import json
import subprocess

import pytest

from test_spike_graph_runtime import ROOT,graph,native,test_generated_rv64_dispatch_runs_complete_generation_graph as embedded_graph
from system_sim.physical_host.asset_source import source_manifest,SOURCE_BASE
from system_sim.physical_host.graph_lowering import literal_bytes
from scripts.run_mlx_spike_graph import sha,SPIKE


@pytest.fixture(scope="module")
def source_binary():
    build=ROOT/"build/mlx-spike-matrix"
    for command in (["cmake","-S",str(ROOT/"system_sim/physical_device"),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake","--build",str(build),"--target","asset-source-contract","-j4"]):
        process=subprocess.run(command,capture_output=True,text=True,timeout=180)
        assert process.returncode==0,process.stdout+process.stderr
    return build/"asset-source-contract"


def probe(binary,path,config,actions=(),error=False):
    file=path/"job.json";file.write_text(json.dumps({"config":config,"actions":actions}))
    process=subprocess.run([str(binary),str(file)],capture_output=True,text=True,timeout=30)
    if error:
        assert process.returncode!=0,process.stdout
        return
    assert process.returncode==0,process.stdout+process.stderr
    return json.loads(process.stdout)


@pytest.fixture
def source_case(tmp_path):
    data=bytes(i%251 for i in range(140000));file=tmp_path/"input.bin";file.write_bytes(data)
    config={"version":1,"bytes":2**34,"regions":[{"name":"a","offset":2**32+4096,"bytes":100000,"path":str(file),"file_offset":4093}]}
    return file,data,config


def test_native_source_cross_cache_file_offset_and_read_only(source_binary,tmp_path,source_case):
    file,data,config=source_case;start=config["regions"][0]["offset"]
    actions=[{"kind":"read","offset":start+i,"bytes":8} for i in (0,65528,65536,99992)]
    actions.extend([{"kind":"read","offset":start-8,"bytes":8},{"kind":"read","offset":start+100000,"bytes":8},{"kind":"read","offset":start+1,"bytes":8},{"kind":"store","offset":start,"bytes":8}])
    result=probe(source_binary,tmp_path,config,actions)
    for event,at in zip(result["events"][:4],(0,65528,65536,99992)):
        assert event["ok"] and event["data"]==list(data[4093+at:4101+at])
    assert all(not event["ok"] and event["data"]==[0xa5]*8 for event in result["events"][4:])
    assert result["source"]["failed_reads"]==3 and result["source"]["rejected_writes"]==1
    assert result["source"]["host_cache_bytes"]==65536 and file.read_bytes()==data


@pytest.mark.parametrize("damage",["version","overlap","duplicate","negative","extent","capacity","directory"])
def test_native_source_rejects_invalid_mapping(source_binary,tmp_path,source_case,damage):
    file,_,config=source_case;config=copy.deepcopy(config);region=config["regions"][0]
    if damage=="version":config["version"]=2
    elif damage=="overlap":config["regions"].append({**region,"name":"b","offset":region["offset"]+64})
    elif damage=="duplicate":config["regions"].append({**region,"offset":region["offset"]+200000})
    elif damage=="negative":region["file_offset"]=-1
    elif damage=="extent":region["bytes"]=file.stat().st_size
    elif damage=="capacity":config["bytes"]=4096
    else:region["path"]=str(tmp_path)
    probe(source_binary,tmp_path,config,error=True)


def test_native_source_detects_truncation_without_partial_output(source_binary,tmp_path,source_case):
    file,_,config=source_case;at=config["regions"][0]["offset"]
    result=probe(source_binary,tmp_path,config,[{"kind":"read","offset":at,"bytes":8},
        {"kind":"truncate","path":str(file),"bytes":16},{"kind":"read","offset":at+65536,"bytes":8}])
    assert not result["events"][-1]["ok"] and result["events"][-1]["data"]==[0xa5]*8
    assert not result["source"]["files_unchanged"]


def test_native_source_preserves_file_offsets_above_4gib(source_binary,tmp_path):
    file=tmp_path/"sparse.bin";offset=2**32+37;raw=bytes(range(19))
    with file.open("wb") as output:output.seek(offset);output.write(raw)
    config={"version":1,"bytes":4096,"regions":[{"name":"high","offset":64,"bytes":19,"path":str(file),"file_offset":offset}]}
    result=probe(source_binary,tmp_path,config,[{"kind":"read","offset":64,"bytes":8},{"kind":"read","offset":80,"bytes":2},{"kind":"read","offset":82,"bytes":1}])
    assert [e["data"] for e in result["events"]]==[list(raw[:8]),list(raw[16:18]),[raw[18]]]


def test_manifest_keeps_source_files_and_packs_only_literals(tmp_path,source_case):
    file,_,_=source_case;program={"assets":{"file":{"kind":"mapped_file","path":str(file),"byte_offset":4093,"bytes":18,"dtype":"f16","shape":[9]},
        "literal":{"kind":"literal","dtype":"i64","shape":[1],"values":[2**60+1]}}}
    entries,config=source_manifest(program,tmp_path,source_offset=2**32+4096)
    assert (tmp_path/"literal_assets.bin").stat().st_size==8
    assert entries[0]["source_address"]==SOURCE_BASE+2**32+4096
    assert config["regions"][0]["path"]==str(file) and config["regions"][0]["file_offset"]==4093


@pytest.mark.parametrize("perturb",[False,True])
def test_file_loaded_graph_matches_embedded_assets(graph,tmp_path,perturb):
    embedded_graph(graph,tmp_path,perturb)
    original,_,_=graph;out=tmp_path/"files"
    program=json.loads((tmp_path/"program.json").read_text());name=next(n["args"][0]["value"] for n in program["nodes"] if n["kind"]=="embedding")
    spec=program["assets"][name];raw=literal_bytes(spec);file=tmp_path/"weights.bin";file.write_bytes(bytes(37)+raw+b"tail")
    program["assets"][name]={"kind":"mapped_file","dtype":spec["dtype"],"shape":spec["shape"],"path":str(file),"byte_offset":37,"bytes":len(raw),"file_sha256":sha(file)}
    (tmp_path/"mapped-program.json").write_text(json.dumps(program))
    command=[str(ROOT/".venv/bin/python"),"-m","scripts.run_mlx_spike_graph","--program",str(tmp_path/"mapped-program.json"),
        "--lifetimes",str(original/"life.json"),"--reference",str(tmp_path/"reference.json"),"--output",str(out),"--asset-source","files","--source-offset",str(2**32+4096)]
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=300)
    assert process.returncode==0,process.stdout+process.stderr
    report=json.loads((out/"report.json").read_text());device=json.loads((out/"device.json").read_text())
    assert report["source_calls"]==45 and report["all_assets_loaded_and_digest_checked"]
    assert report["output_bytes_checked_in_elf"] and not report["full_model_execution_verified"]
    assert device.pop("asset_load_audit")["attempted"]
    assert device==json.loads((tmp_path/"run/device.json").read_text())

    # The audit checks destination bytes, not just source read counts. A bad
    # digest or unwritten destination must stop before any compute data access.
    for damage in ("digest","unwritten"):
        directory=tmp_path/damage;directory.mkdir()
        config=json.loads((out/"plugin.json").read_text());config["report"]=str(directory/"device.json")
        if damage=="digest":config["loaded_assets"][0]["sha256"]="0"*64
        else:config["loaded_assets"][0]["base"]+=8192
        source=json.loads((out/"asset-source-config.json").read_text());source["report"]=str(directory/"source.json")
        (directory/"plugin.json").write_text(json.dumps(config));(directory/"source.json.config").write_text(json.dumps(source))
        plugin=out/"libmlx_spike_matrix.so"
        if not plugin.exists():plugin=ROOT/"build/mlx-spike-matrix/libmlx_spike_matrix.so"
        process=subprocess.run([str(SPIKE),"--isa=RV64IMAFD","-m64",f"--extlib={plugin}",f"--device=mlx_matrix,0x100000000,{directory/'plugin.json'}",
            f"--device=mlx_asset_source,{hex(SOURCE_BASE)},{directory/'source.json.config'}",str(out/"test.elf")],capture_output=True,text=True,timeout=60)
        assert process.returncode!=0
        failed=json.loads((directory/"device.json").read_text())
        assert failed["asset_load_audit"]["error"] and not failed["device_reads"] and not failed["device_writes"]


def test_actual_cpu_load_only_does_not_execute_model(graph,tmp_path):
    original,_,_=graph;out=tmp_path/"load"
    command=[str(ROOT/".venv/bin/python"),"-m","scripts.run_mlx_spike_graph","--program",str(original/"program.json"),
        "--lifetimes",str(original/"life.json"),"--output",str(out),"--asset-source","files","--load-only"]
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=300)
    assert process.returncode==0,process.stdout+process.stderr
    report=json.loads((out/"report.json").read_text());device=json.loads((out/"device.json").read_text())
    assert report["source_calls"]==report["device_windows"]==0 and report["compiled_source_calls"]==45
    assert report["all_assets_loaded_and_digest_checked"] and report["load_only"] and not report["output_bytes_checked_in_elf"]
    assert device["cpu_payload_write_bytes"]==report["loaded_asset_bytes"] and not device["device_reads"] and not device["device_writes"]
