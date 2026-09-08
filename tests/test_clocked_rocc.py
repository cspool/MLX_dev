import json
import struct
import subprocess

import pytest

from test_clocked_device import ROOT,chain
from mlxsim.model_memory_program import Planner
from system_sim.physical_device.memory_lowering import lower_memory


@pytest.fixture(scope="module")
def binary():
    build=ROOT/"build/mlx-clocked-rocc"
    for command in (["cmake","-S",str(ROOT/"system_sim/clocked_rocc"),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake","--build",str(build),"--target","clocked-rocc-driver","-j4"]):
        p=subprocess.run(command,capture_output=True,text=True,timeout=180);assert p.returncode==0,p.stdout+p.stderr
    return build/"clocked-rocc-driver"


@pytest.mark.parametrize("tags,perturb,reject",[(1,False,False),(2,True,False),(6,False,True)])
def test_rocc_handshakes_tag_reuse_and_controlled_launch(binary,tmp_path,tags,perturb,reject):
    job=chain(perturb);job.update(tag_bits=tags,reject_frontend=reject);(tmp_path/"job.json").write_text(json.dumps(job))
    p=subprocess.run([str(binary),str(tmp_path/"job.json"),str(tmp_path/"report.json")],capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
    report=json.loads((tmp_path/"report.json").read_text())
    assert report["outputs"]==[list(struct.pack('<4f',4. if perturb else 2.,0.5,1.,1.5))]
    assert report["reply_hold_checked"] and report["busy_reset_checked"] and not report["frontend_error"]
    assert report["requests"]==report["responses"]==2696
    assert sum(w.get("done",False) for w in report["windows"])==3
    assert not report["mlx_system_verified"]


def test_rocc_rejects_unknown_cache_tag(binary,tmp_path):
    job=chain();job["wrong_tag"]=True;(tmp_path/"job.json").write_text(json.dumps(job))
    p=subprocess.run([str(binary),str(tmp_path/"job.json"),str(tmp_path/"report.json")],capture_output=True,text=True,timeout=120)
    assert p.returncode!=0 and "matching accepted tag" in p.stderr


def test_byte_store_bus_expansion_reaches_high_word_lanes(binary,tmp_path):
    job=chain();base=job["base"];planner=Planner();planner.add_asset("v",{"dtype":"f16","shape":[1,4]})
    node={"id":"out","kind":"cast","args":[{"value":"v"},"torch.bool",False,True],"kwargs":{},"output":{"dtype":"bool","shape":[1,4]},"source_operator_id":3}
    planner.register(node,planned=True)
    bindings={"v":{"base":base+65536+3*128,"bytes":8,"writable":False},"out":{"base":base+66052,"bytes":4,"writable":True}}
    wire,_=lower_memory(node,planner.layouts,bindings);job["commands"][-1]["data"]=list(wire);job["outputs"]=[{"address":base+66052,"bytes":4}]
    (tmp_path/"job.json").write_text(json.dumps(job));p=subprocess.run([str(binary),str(tmp_path/"job.json"),str(tmp_path/"report.json")],capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
    assert json.loads((tmp_path/"report.json").read_text())["outputs"]==[[1,1,1,1]]
