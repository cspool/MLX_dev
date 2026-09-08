import copy
import json
import subprocess
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def binary():
    build=ROOT/"build/mlx-system-profile"
    for command in (["cmake","-S",str(ROOT/"system_sim/model_image"),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",str(build),"--target","system-profile-dump","-j4"]):
        result=subprocess.run(command,capture_output=True,text=True,timeout=180);assert result.returncode==0,result.stdout+result.stderr
    return build/"system-profile-dump"


def profile():
    return {"version":1,"name":"full-4x4","max_busy_cycles":10**13,"matrix_options":{"max_cycles":10**12,"cache_control":True,"trace":False},"vector_options":{"max_cycles":10**12,"trace":False},"memory_options":{"max_cycles":10**12,"trace":False}}


def run(binary,tmp_path,value=None):
    command=[str(binary)]
    if value is not None:
        path=tmp_path/"profile.json";path.write_text(json.dumps(value));command.append(str(path))
    return subprocess.run(command,capture_output=True,text=True,timeout=30)


def test_explicit_profile_preserves_full_geometry_and_cycle_limits(binary,tmp_path):
    result=run(binary,tmp_path,profile());assert result.returncode==0,result.stderr
    parsed=json.loads(result.stdout)
    for kind in ("matrix","vector"):
        assert parsed[kind+"_options"]["rows"]==parsed[kind+"_options"]["columns"]==4
        assert parsed[kind+"_options"]["contexts"]==2 and parsed[kind+"_options"]["max_cycles"]==10**12
    assert parsed["matrix_options"]["multiply_latency"]==4 and parsed["vector_options"]["exp_latency"]==8
    assert run(binary,tmp_path,parsed).stdout==result.stdout


def test_absent_profile_retains_demo_geometry(binary,tmp_path):
    result=run(binary,tmp_path);assert result.returncode==0,result.stderr
    parsed=json.loads(result.stdout);assert parsed["name"]=="demo-1pe" and parsed["matrix_options"]["rows"]==parsed["vector_options"]["columns"]==1


@pytest.mark.parametrize("damage",["extra","missing","limit","geometry","backend_field","stale"])
def test_invalid_or_inconsistent_profiles_are_rejected(binary,tmp_path,damage):
    value=copy.deepcopy(profile())
    if damage=="extra":value["skip_compute"]=True
    elif damage=="missing":del value["memory_options"]
    elif damage=="limit":value["max_busy_cycles"]=0
    elif damage=="geometry":value["vector_options"]["rows"]=1
    elif damage=="backend_field":value["matrix_options"]["faster"]=True
    else:value["vector_options"]["inject_stale_response"]=True
    assert run(binary,tmp_path,value).returncode!=0
