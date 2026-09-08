"""Host-side control caching must preserve every simulated event and counter."""
import copy

import numpy as np
import pytest

from test_matrix_window_scheduler import binary, make_job, run, assert_trace_contract
from test_matrix_external_memory import binary as external_binary, external_job, run as run_external


def without_host_counters(report):
    result = copy.deepcopy(report); result.pop("host_optimization")
    return result


@pytest.mark.parametrize("precision", ["f16", "f32"])
@pytest.mark.parametrize("shape,bias", [((3,19,0),True), ((5,37,5),False), ((3,19,65),True)])
@pytest.mark.parametrize("overlap", [False,True])
def test_cached_admission_has_identical_trace_and_cycle_accounting(binary,tmp_path,precision,shape,bias,overlap):
    m,n,k=shape
    job,expected=make_job(m=m,n=n,k=k,precision=precision,bias=bias,rows=1,columns=2,overlap=overlap,
        dma_request_period=3,dma_response_period=5,spm_period=2,writeback_period=3,trace_limit=200000)
    job["options"]["cache_control"]=False
    original,old_values=run(binary,tmp_path/"scan",job)
    fast_job=copy.deepcopy(job);fast_job["options"]["cache_control"]=True
    cached,new_values=run(binary,tmp_path/"cached",fast_job)
    np.testing.assert_array_equal(old_values.view(np.uint8),new_values.view(np.uint8))
    np.testing.assert_array_equal(new_values.view(np.uint8),expected.view(np.uint8))
    assert without_host_counters(original)==without_host_counters(cached)
    assert_trace_contract(cached,fast_job,new_values)
    assert original["host_optimization"]["invariant_checks"]==original["cycles"]
    assert cached["host_optimization"]["control_recomputations"] <= 2*cached["admitted"]
    assert cached["host_optimization"]["invariant_checks"] <= 2*cached["admitted"]


def test_cached_control_handles_four_by_four_admission_pressure(binary,tmp_path):
    job,_=make_job(m=3,n=145,k=5,rows=4,columns=4,trace_limit=200000,cache_control=False)
    scan,old=run(binary,tmp_path/"scan",job);job["options"]["cache_control"]=True
    cached,new=run(binary,tmp_path/"cached",job)
    np.testing.assert_array_equal(old.view(np.uint8),new.view(np.uint8))
    assert without_host_counters(scan)==without_host_counters(cached)
    assert cached["admission_stall_cycles"]>0 and cached["admitted"]>16


def test_cached_control_preserves_external_retry_and_byte_commits(external_binary,tmp_path):
    job,_=external_job(m=3,n=19,k=5,precision="f32",bias=True,dma_response_period=3,cache_control=False)
    job["buffered_bridge"]=True
    scan,old=run_external(external_binary,tmp_path/"scan",job);job["options"]["cache_control"]=True
    cached,new=run_external(external_binary,tmp_path/"cached",job)
    np.testing.assert_array_equal(old.view(np.uint8),new.view(np.uint8))
    assert without_host_counters(scan)==without_host_counters(cached)
    assert cached["transport"]["nacks"]>0


@pytest.mark.parametrize("cached", [False,True])
def test_cached_control_never_suppresses_stale_owner_checks(binary,tmp_path,cached):
    job,_=make_job(cache_control=cached,inject_stale_dma_epoch=True)
    run(binary,tmp_path/"stale",job,error="stale or mismatched")
