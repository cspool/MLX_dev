import copy
import json
import struct
import subprocess
from pathlib import Path

import pytest

from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_vector_program import vector_program
from mlxsim.model_memory_program import Planner,make_layout
from system_sim.physical_device.lowering import lower_matrix_window
from system_sim.physical_device.vector_lowering import lower_vector
from system_sim.physical_device.memory_lowering import lower_memory

ROOT=Path(__file__).resolve().parents[1]
BASE=2**32


@pytest.fixture(scope="module")
def binary():
    build=ROOT/"build/mlx-clocked-device"
    for command in (["cmake","-S",str(ROOT/"system_sim/clocked_device"),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake","--build",str(build),"--target","clocked-device-driver","-j4"]):
        p=subprocess.run(command,capture_output=True,text=True,timeout=180)
        assert p.returncode==0,p.stdout+p.stderr
    return build/"clocked-device-driver"


def chain(perturb=False):
    layouts={name:make_layout("f16",shape,name) for name,shape in (("a",[1,4]),("b",[4,4]),("m",[1,4]),("v",[1,4]))}
    bindings={name:{"base":BASE+65536+i*128,"bytes":8 if name!="b" else 32,"writable":name in {"m","v"}} for i,name in enumerate(layouts)}
    node={"id":"m","kind":"linear","args":[{"value":"a"},{"value":"b"}],"output":{"shape":[1,4],"dtype":"f16"},"matrix_program":matrix_program("f16","f16"),"source_operator_id":1}
    matrix,_=lower_matrix_window(node,layouts,bindings)
    node={"id":"v","kind":"mul","args":[{"value":"m"},0.5],"kwargs":{},"output":{"shape":[1,4],"dtype":"f16"},"vector_program":vector_program("mul",["f16","f32"],"f16"),"source_operator_id":2}
    vector_bindings=copy.deepcopy(bindings);vector_bindings["m"]["writable"]=False
    vector,_=lower_vector(node,layouts,vector_bindings)
    planner=Planner();planner.add_asset("v",{"dtype":"f16","shape":[1,4]})
    node={"id":"out","kind":"cast","args":[{"value":"v"},"torch.float32",False,True],"kwargs":{},"output":{"shape":[1,4],"dtype":"f32"},"source_operator_id":3}
    planner.register(node,planned=True)
    mbindings={"v":{**bindings["v"],"writable":False},"out":{"base":BASE+66048,"bytes":16,"writable":True}}
    memory,_=lower_memory(node,planner.layouts,mbindings)
    weights=[0.,0.,0.,1.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.]
    if perturb:weights[3]=2.
    def initial(name,values):return {"address":bindings[name]["base"],"data":list(struct.pack('<'+'e'*len(values),*values))}
    return {"base":BASE,"bytes":1048576,"matrix_options":{"rows":1,"columns":1,"trace":True},"vector_options":{"rows":1,"columns":1,"trace":True},"memory_options":{"trace":True},
        "initial":[initial("a",[1.,2.,3.,4.]),initial("b",weights)],"commands":[{"address":BASE+4096,"data":list(wire),"source_id":i+1} for i,wire in enumerate((matrix,vector,memory))],
        "outputs":[{"address":BASE+66048,"bytes":16}]}


def run(binary,directory,job,error=None):
    directory.mkdir();(directory/"job.json").write_text(json.dumps(job));p=subprocess.run([str(binary),str(directory/"job.json"),str(directory/"report.json")],capture_output=True,text=True,timeout=120)
    if error:
        assert p.returncode!=0 and error in p.stderr,p.stdout+p.stderr
        return
    assert p.returncode==0,p.stdout+p.stderr
    return json.loads((directory/"report.json").read_text())


@pytest.mark.parametrize("perturb",[False,True])
def test_external_descriptor_and_tensor_responses_execute_chain(binary,tmp_path,perturb):
    job=chain(perturb);report=run(binary,tmp_path/"chain",job)
    assert report["outputs"]==[list(struct.pack('<4f',4. if perturb else 2.,0.5,1.,1.5))]
    assert [r["backend"] for r in report["launches"]]==["matrix","vector","memory"]
    assert all(r["done"] and r["transport"]["idle"] and not r["busy"] for r in report["launches"])
    for item,command in zip(report["launches"],job["commands"]):
        assert item["descriptor_bytes_fetched"]==len(command["data"])
        assert item["cycle"]-item["launch_start_cycle"]==item["fetch_cycles"]+item["run_cycles"]
    assert report["reset_while_busy_rejected"] and not report["mlx_system_verified"]


def test_status_polling_never_advances_target_time(binary,tmp_path):
    job=chain();ordinary=run(binary,tmp_path/"ordinary",job);job["status_polls"]=7
    assert ordinary==run(binary,tmp_path/"polls",job)


def test_backpressure_and_nacks_preserve_tokens_and_data(binary,tmp_path):
    job=chain();job.update(accept_period=3,latency=5,nack_every=3)
    report=run(binary,tmp_path/"retry",job)
    assert report["outputs"]==[list(struct.pack('<4f',2.,0.5,1.,1.5))]
    port=report["launches"][-1]["transport"]
    assert port["nacks"]>0 and port["accepted"]==port["submitted"]+port["nacks"]
    assert port["submitted"]==port["responses"]==port["consumed"]
    identities=[c["id"] for c in report["commits"]]
    assert len(identities)==len(set(identities))
    attempts={}
    for a in report["attempts"]:
        comparable={k:v for k,v in a.items() if k!="cycle"}
        assert attempts.setdefault(a["id"],comparable)==comparable


@pytest.mark.parametrize("fault,message",[("descriptor_error","descriptor memory response"),("write_error","reported error")])
def test_error_responses_drain_without_false_success(binary,tmp_path,fault,message):
    job=chain();job["fault"]=fault;report=run(binary,tmp_path/"fault",job)
    last=report["launches"][-1]
    assert not last["done"] and last["complete"] and message in last["error"]
    assert last["transport"]["idle"] and not report["outputs"]
    assert not report["after_reset"]["busy"]


def test_wrong_response_identity_is_a_protocol_failure(binary,tmp_path):
    job=chain();job["fault"]="wrong_token"
    run(binary,tmp_path/"token",job,error="matching in-flight")


@pytest.mark.parametrize("damage",["magic","reserved","alignment","size","overlap"])
def test_descriptor_contract_is_checked_before_compute(binary,tmp_path,damage):
    job=chain();command=job["commands"][0]
    if damage=="magic":command["data"][0]^=1
    elif damage=="reserved":command["data"][15*8]=1
    elif damage=="alignment":command["address"]+=1
    elif damage=="size":command["data"]=command["data"][:-8]
    else:command["address"]=BASE+65536
    if damage in {"alignment","size"}:
        run(binary,tmp_path/"invalid",job,error="alignment/range" if damage=="alignment" else "descriptor size")
    else:
        report=run(binary,tmp_path/"invalid",job);last=report["launches"][-1]
        assert not last["done"] and last["error"] and last["run_cycles"]==0
        assert all(not c["write"] for c in report["commits"])


def test_cycle_limit_keeps_inflight_ownership_until_response(binary,tmp_path):
    job=chain();job.update(max_busy_cycles=2,latency=12)
    report=run(binary,tmp_path/"limit",job);last=report["launches"][-1]
    assert "cycle limit" in last["error"] and last["drain_cycles"]>0
    assert last["transport"]["idle"] and len(report["commits"])==1
    assert last["cycle"]-last["launch_start_cycle"]==last["fetch_cycles"]+last["run_cycles"]+last["drain_cycles"]


def test_rejected_descriptor_can_be_followed_by_fresh_work(binary,tmp_path):
    job=chain();bad=copy.deepcopy(job["commands"][0]);bad["data"][0]^=1;bad["source_id"]=0
    job["commands"].insert(0,bad);job["recover"]=True
    report=run(binary,tmp_path/"recover",job)
    assert [r["done"] for r in report["launches"]]==[False,True,True,True]
    assert report["outputs"]==[list(struct.pack('<4f',2.,0.5,1.,1.5))]
    assert report["launches"][-1]["error"]==""


def test_quiescent_reset_never_reuses_request_tokens(binary,tmp_path):
    job=chain();job["reset_between"]=True
    report=run(binary,tmp_path/"reset",job)
    identities=[c["id"] for c in report["commits"]]
    assert identities==sorted(set(identities)) and all(r["done"] for r in report["launches"])
    assert report["outputs"]==[list(struct.pack('<4f',2.,0.5,1.,1.5))]
