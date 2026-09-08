import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_matrix_reference import kasc_reference
from test_matrix_window_scheduler import ROOT,BUILD,make_job
from test_model_tensor_semantics import literal


@pytest.fixture(scope="module")
def binary():
    for command in (["cmake","-S",str(ROOT/"simulator_ext/matrix_schedule"),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"],["cmake","--build",str(BUILD),"--target","matrix-external-memory","-j4"]):
        result=subprocess.run(command,capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stdout+result.stderr
    return BUILD/"matrix-external-memory"


def run(binary,path,job,error=None):
    path.mkdir();(path/"job.json").write_text(json.dumps(job))
    result=subprocess.run([str(binary),str(path/"job.json"),str(path/"out")],capture_output=True,text=True,timeout=120)
    if error:
        assert result.returncode!=0 and error in result.stderr,result.stdout+result.stderr
        return
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((path/"out/result.json").read_text())
    dtype=np.float16 if job["program"]["output_dtype"]=="f16" else np.float32
    actual=np.fromfile(path/"out/output.bin",dtype=dtype).reshape(job["m"],job["n"])
    assert report["done"] and report["external_memory_port"] and report["virtual_tensor_backing_used"]
    retries=report.get("transport",{}).get("nacks",0)
    assert report["dma_requests"]==report["dma_responses"]==len(report["physical_requests"])-retries
    assert report["physical_commits"]["reads"]+report["physical_commits"]["writes"]==report["dma_requests"]
    assert report["physical_commits"]["read_bytes"]==report["dma_read_bytes"]
    assert report["physical_commits"]["write_bytes"]==report["dma_write_bytes"]
    assert all(item["address"]>=2**32 for item in report["physical_requests"])
    assert not report["mlx_system_verified"] and not report["inference_performance_eligible"]
    return report,actual


def external_job(**kwargs):
    job,expected=make_job(**kwargs)
    job["physical_values"]={name:copy.deepcopy(job[name]) for name in ("a","b","bias") if name in job}
    # These tensor values are deliberately not supplied to the compute engine.
    # The C++ driver constructs metadata with null data/writable pointers.
    job["a"]["values"]=[[999.0]]
    job["b"]["values"]=[[999.0]]
    return job,expected


@pytest.mark.parametrize("precision",["f16","f32"])
@pytest.mark.parametrize("k,bias",[(0,True),(5,True),(65,False)])
def test_matrix_values_come_from_physical_responses(binary,tmp_path,precision,k,bias):
    job,expected=external_job(m=3,n=19,k=k,precision=precision,bias=bias,dma_response_period=5,spm_period=2,writeback_period=3)
    report,actual=run(binary,tmp_path/"external",job)
    raw=np.uint16 if precision=="f16" else np.uint32
    np.testing.assert_array_equal(actual.view(raw),expected.view(raw))
    assert report["numeric_instructions"]["mul_active_lanes"]==3*19*k
    assert report["active_contexts"]==report["allocated_spm_vectors"]==0


def test_external_memory_retains_batch_offset_and_noncontiguous_addressing(binary,tmp_path):
    a=np.arange(30,dtype=np.float32).reshape(2,3,5)/8
    w=np.arange(95,dtype=np.float32).reshape(19,5)/32
    job={"schema":"mlx_matrix_window_job_v1","m":3,"n":19,"k":5,"a_batch":1,"transposed_b":False,
         "program":matrix_program("f32","f32"),"options":{"rows":1,"columns":2},
         "a":{"dtype":"f32","shape":[2,3,5]},"b":{"dtype":"f32","shape":[5,19],"strides":[1,5]},
         "physical_values":{"a":literal(torch.from_numpy(a)),"b":literal(torch.from_numpy(w))}}
    _,actual=run(binary,tmp_path/"strided",job)
    expected=kasc_reference(a[1],w,transposed_b=True)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def test_readonly_region_alias_is_authoritative(binary,tmp_path):
    job,_=external_job(m=3,n=3,k=3,precision="f32")
    a=np.asarray(job["physical_values"]["a"]["values"],dtype=np.float32)
    job["region_overrides"]={"1":{"base":2**32}}
    _,actual=run(binary,tmp_path/"alias",job)
    np.testing.assert_array_equal(actual.view(np.uint32),kasc_reference(a,a,transposed_b=True).view(np.uint32))


@pytest.mark.parametrize("fault,error",[("token","owner mismatch"),("error","reported error"),("unsolicited","owner mismatch"),("unstable","backpressure"),("withdraw","backpressure")])
def test_external_response_faults_cannot_complete_a_matrix(binary,tmp_path,fault,error):
    job,_=external_job(m=2,n=3,k=2,precision="f32",dma_response_period=17)
    job["fault"]=fault
    run(binary,tmp_path/"fault",job,error)


@pytest.mark.parametrize("overrides,error",[
    ({"0":{"readable":False}},"permission"),
    ({"3":{"writable":False}},"permission"),
    ({"0":{"bytes":2}},"out of bounds"),
    ({"0":{"base":2**64-2,"bytes":16}},"address overflow"),
    ({"3":{"base":2**32}},"overlap"),
    ({"0":{"base":2**32+1}},"misaligned"),
])
def test_physical_binding_checks_permissions_bounds_and_aliases(binary,tmp_path,overrides,error):
    job,_=external_job(m=2,n=3,k=2,precision="f32")
    job["region_overrides"]=overrides
    run(binary,tmp_path/"invalid",job,error)


def test_shared_memory_port_token_and_backpressure_contract(binary):
    result=subprocess.run(["cmake","--build",str(BUILD),"--target","model-io-contract","-j4"],capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(BUILD/"model-io-contract")],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert "MODEL_IO_CONTRACT_PASS" in result.stdout


def test_registered_transport_retries_without_duplicate_matrix_effects(binary,tmp_path):
    job,expected=external_job(m=3,n=19,k=5,precision="f32",bias=True,dma_response_period=3)
    job["buffered_bridge"]=True
    report,actual=run(binary,tmp_path/"buffered",job)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    transport=report["transport"]
    assert transport["nacks"]>0 and transport["idle"]
    assert transport["submitted"]==transport["responses"]==transport["consumed"]==report["dma_requests"]
    assert transport["accepted"]==transport["submitted"]+transport["nacks"]
