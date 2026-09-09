import copy

import numpy as np
import pytest

from mlxsim.model_matrix_events import matrix_events
from test_event_schedule import binary,event,program
from test_event_loops import run
from test_matrix_window_scheduler import binary as matrix_binary,make_job,run as run_matrix


def ports(count=2,same_pe=True):
    p=program(count=count,same_pe=same_pe);p["schema"]="mlx_event_schedule_v3"
    p["hardware"]["latencies"]["predicate_skip"]=0
    p["hardware"].update(spm_port_period=1,writeback_period=1,dma_request_period=1,dma_response_period=1,compute_ii=1)
    return p


def at(r,name,kind):
    return next(e["cycle"] for e in r["trace"] if e.get("operation")==name and e["event"]==kind)


def test_dma_does_not_consume_same_pe_instruction_issue(binary,tmp_path):
    p=ports();p["blocks"][0]["events"]=[event("dma","dma_read")];p["blocks"][1]["events"]=[event("compute")]
    r=run(binary,tmp_path/"independent",p)
    assert at(r,"dma","issue")==at(r,"compute","issue")==1
    assert r["integrated_usage"]["compute_dma_inflight_overlap_pe_cycles"]==4
    assert r["counts"]["writeback_port_claims"]==1 and r["counts"]["spm_port_claims"]==1


def test_writeback_conflict_holds_compute_until_actual_completion(binary,tmp_path):
    p=ports();p["hardware"]["latencies"]["mul"]=2
    p["blocks"][0]["events"]=[event("load","spm_read")];p["blocks"][1]["events"]=[event("compute")]
    r=run(binary,tmp_path/"writeback",p)
    assert at(r,"load","complete")==4 and at(r,"compute","complete")==5
    assert r["counts"]["writeback_wait_cycles"]==1 and r["cycles"]==6


def test_spm_write_completion_precedes_dma_read_response(binary,tmp_path):
    p=ports();p["hardware"]["latencies"]["dma_read"]=3
    p["blocks"][0]["events"]=[event("dma","dma_read")];p["blocks"][1]["events"]=[event("store","spm_write")]
    r=run(binary,tmp_path/"spm",p)
    assert at(r,"store","issue")==at(r,"dma","issue")==1
    assert at(r,"store","complete")==4 and at(r,"dma","complete")==5
    assert r["counts"]["spm_completion_wait_cycles"]==1


def test_predicated_row_uses_issue_but_no_fu_or_writeback(binary,tmp_path):
    p=ports(count=1);p["blocks"][0]["events"]=[dict(id="skip",op="predicate_skip",dependencies=[]),event("compute")]
    r=run(binary,tmp_path/"predicated",p)
    assert at(r,"skip","issue")==at(r,"skip","complete")==1
    assert at(r,"compute","issue")==2 and r["counts"]["writeback_port_claims"]==1
    assert r["integrated_usage"]["compute_inflight_pe_cycles"]==4


def test_native_periods_and_single_source_admission_edges(binary,tmp_path):
    p=ports(count=1)
    p["hardware"].update(spm_port_period=2,writeback_period=4,dma_request_period=3,dma_response_period=5)
    p["hardware"]["latencies"].update(spm_read=1,dma_write=2)
    p["blocks"][0]["events"]=[event("load","spm_read"),event("drain","dma_write")]
    r=run(binary,tmp_path/"periods",p)
    assert [at(r,"load","issue"),at(r,"load","complete"),at(r,"drain","issue"),at(r,"drain","complete")]==[2,4,6,10]
    assert r["cycles"]==11
    p=ports(count=3,same_pe=False)
    for b in p["blocks"]:b["source_operator_id"]=0
    r=run(binary,tmp_path/"admission",p)
    assert [b["admit_cycle"] for b in r["block_intervals"]]==[0,1,2]


def test_instruction_spm_access_precedes_dma_store_request(binary,tmp_path):
    p=ports();p["blocks"][0]["events"]=[event("dma","dma_write")];p["blocks"][1]["events"]=[event("load","spm_read")]
    r=run(binary,tmp_path/"request",p)
    assert at(r,"load","issue")==1 and at(r,"dma","issue")==2


def test_source_head_of_line_waits_for_missing_admission_dependency(binary,tmp_path):
    p=ports(count=3,same_pe=False)
    p["blocks"][0]["admission_dependencies"]=["b2"];p["blocks"][1]["source_operator_id"]=0;p["blocks"][2]["source_operator_id"]=1
    r=run(binary,tmp_path/"head",p);intervals=r["block_intervals"]
    assert intervals[0]["admit_cycle"]==intervals[2]["retire_cycle"]
    assert intervals[1]["admit_cycle"]==intervals[0]["admit_cycle"]+1


def test_port_calendar_does_not_hide_residency_deadlock_with_periodic_wakes(binary,tmp_path):
    p=ports(count=2,same_pe=True);p["hardware"]["contexts"]=1
    p["blocks"][0]["events"]=[event("wait","dma_read",["later"])]
    p["blocks"][1]["events"]=[event("later")]
    run(binary,tmp_path/"deadlock",p,"deadlock")


@pytest.mark.parametrize("precision",["f16","f32"])
@pytest.mark.parametrize("geometry",[(1,1,0),(1,17,3),(3,17,5),(2,16,65)])
@pytest.mark.parametrize("bias",[False,True])
def test_complete_matrix_event_stream_matches_native_window(binary,matrix_binary,tmp_path,precision,geometry,bias):
    m,n,k=geometry;job,expected=make_job(m,n,k,precision=precision,bias=bias,rows=1,columns=2,trace=True,max_cycles=10000000)
    native,actual=run_matrix(matrix_binary,tmp_path/"native",job)
    np.testing.assert_array_equal(actual,expected)
    result=run(binary,tmp_path/"event",matrix_events(job))
    assert result["cycles"]==native["cycles"], (result["cycles"],native["cycles"])
    work=result["declared_work"];areas=result["integrated_usage"]
    assert work.get("mul_declared_active_lanes",0)==native["numeric_instructions"]["mul_active_lanes"]==m*n*k
    assert work.get("dma_read_bytes",0)==native["dma_read_bytes"]
    assert work["dma_write_bytes"]==native["dma_write_bytes"]
    assert work.get("predicate_skip_events",0)==native["numeric_instructions"].get("inactive_row_instructions",0)
    for event_counter,native_counter in (("compute_inflight_pe_cycles","compute_busy_pe_cycles"),("spm_inflight_cycles","spm_busy_cycles"),("dma_inflight_cycles","dma_busy_cycles"),("compute_dma_inflight_overlap_pe_cycles","compute_dma_overlap_pe_cycles")):
        assert areas.get(event_counter,0)==native[native_counter],event_counter


@pytest.mark.parametrize("precision",["f16","f32"])
@pytest.mark.parametrize("periods",[(2,3,2,3,7),(3,2,1,5,1),(1,4,3,1,9)])
def test_matrix_window_port_period_contention_matches_native(binary,matrix_binary,tmp_path,precision,periods):
    spm,writeback,request,response,ii=periods
    job,expected=make_job(3,19,5,precision=precision,bias=True,rows=2,columns=2,spm_period=spm,writeback_period=writeback,
                          dma_request_period=request,dma_response_period=response,compute_ii=ii,trace=True)
    native,actual=run_matrix(matrix_binary,tmp_path/"native",job);np.testing.assert_array_equal(actual,expected)
    result=run(binary,tmp_path/"event",matrix_events(job))
    assert result["cycles"]==native["cycles"]
    assert result["counts"].get("writeback_wait_cycles",0)==native["writeback_stall_cycles"]
    assert result["integrated_usage"]["dma_inflight_cycles"]==native["dma_busy_cycles"]


@pytest.mark.parametrize("damage",["missing_period","zero_period","predicate_latency","predicate_work","old_schema"])
def test_new_port_contract_is_explicit(binary,tmp_path,damage):
    p=ports(count=1);expected="period"
    if damage=="missing_period":del p["hardware"]["writeback_period"]
    elif damage=="zero_period":p["hardware"]["spm_port_period"]=0
    elif damage=="predicate_latency":p["hardware"]["latencies"]["predicate_skip"]=1;expected="predicate"
    elif damage=="predicate_work":p["blocks"][0]["events"]=[dict(id="bad",op="predicate_skip",dependencies=[],active_lanes=1)];expected="predicate"
    else:p["schema"]="mlx_event_schedule_v2";expected="unknown"
    run(binary,tmp_path/damage,p,expected)


@pytest.mark.parametrize("damage",["serial","program","geometry","dtype","period"])
def test_matrix_event_lowering_rejects_changed_execution_contract(damage):
    job,_=make_job()
    if damage=="serial":job["options"]["overlap"]=False
    elif damage=="program":job["program"]["body"].pop()
    elif damage=="geometry":job["m"]+=1
    elif damage=="dtype":job["a"]["dtype"]="bool"
    else:job["options"]["spm_period"]=0
    with pytest.raises(ValueError):matrix_events(job)
