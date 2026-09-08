"""Actual local instruction writes, visibility, sharing and issue contention."""
import copy
import subprocess
from collections import Counter

import numpy as np
import pytest

from test_ready_graph import binary,execute,BUILD
from test_block_pipeline import matrix_pair
from test_physical_model import outputs


def audit_loading(result):
    array=result["array"]
    assert array["template_configuration_cycles_modeled"] and array["pending_template_words"]==0
    assert array["template_words_loaded"]==array["template_words_requested"]==array["template_config_pe_cycles"]
    trace=array["template_trace"]
    assert not array["template_trace_truncated"] and len(trace)==array["template_words_loaded"]
    writes=Counter((e["cycle"],e["pe"]) for e in trace)
    assert max(writes.values(),default=0)<=1
    assert len({e["cycle"] for e in trace})==array["template_config_wall_cycles"]
    assert all(e["cycle"]%array["template_word_period"]==0 for e in trace)
    templates={}
    for event in trace:
        templates.setdefault((event["pe"],event["template_id"]),[]).append(event)
    for entries in templates.values():
        assert [e["word_index"] for e in entries]==list(range(len(entries)))
        assert entries[-1]["last"] and entries[-1]["visible_cycle"]==entries[-1]["cycle"]+1
    for family in ("matrix","vector"):
        for window in result["windows"][family]:
            ready={};admitted={}
            for event in window["trace"]:
                key=event["lease_id"]
                if event["event"]=="admit":admitted[key]=event["array_cycle"]
                elif event["event"]=="template_ready":ready[key]=event["array_cycle"]
                elif event["event"] in {"issue","predicated_issue"}:
                    assert not writes[event["array_cycle"],event["pe"]]
                    assert event["array_cycle"]>admitted[key]
                    if key in ready:assert event["array_cycle"]>=ready[key]


def test_registered_template_words_visibility_sharing_and_reconfiguration(binary):
    build=subprocess.run(["cmake","--build",str(BUILD),"--target","template-loader-contract","-j4"],capture_output=True,text=True,timeout=180)
    assert build.returncode==0,build.stdout+build.stderr
    result=subprocess.run([str(BUILD/"template-loader-contract")],capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and "TEMPLATE_LOADER_CONTRACT_PASS" in result.stdout,result.stdout+result.stderr


@pytest.mark.parametrize("period",[1,3,7])
@pytest.mark.parametrize("precision",["f16","f32"])
def test_local_template_programming_does_not_change_math_or_block_events(binary,tmp_path,period,precision):
    program,expected,tokens=matrix_pair(m=5,n=19,precision=precision,slots=2)
    base={"tile_pipeline":True,"memory":{"latency":2,"nack_every":5}}
    untimed=execute(binary,tmp_path/"untimed",program,base)
    timed=execute(binary,tmp_path/"timed",program,{**base,"template_load_timing":True,"template_word_period":period,"template_trace_limit":100000})
    audit_loading(timed)
    actual,ids=outputs(timed)[0];old,old_ids=outputs(untimed)[0]
    np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(actual.view(np.uint8),old.view(np.uint8));assert ids==old_ids==tokens
    assert timed["pipeline_groups"][0]["finished"] and not timed["inference_performance_eligible"]
    for family in ("matrix","vector"):
        assert [w["numeric_instructions"] for w in timed["windows"][family]]==[w["numeric_instructions"] for w in untimed["windows"][family]]
        assert any(w["template_wait_context_cycles"]>0 for w in timed["windows"][family])


def test_template_logging_is_optional_and_does_not_advance_the_target(binary,tmp_path):
    program,_,_=matrix_pair()
    options={"tile_pipeline":True,"template_load_timing":True,"template_word_period":3,"template_trace_limit":100000}
    observed=execute(binary,tmp_path/"observed",program,options)
    options["template_trace_limit"]=0;plain=execute(binary,tmp_path/"plain",program,options)
    for result in (observed,plain):
        result["outputs"][0]["logits_file"]="$OUTPUT"
        result["array"].pop("template_trace");result["array"].pop("template_trace_truncated")
    assert observed==plain


def test_each_pe_has_a_bounded_parallel_template_write_port(binary,tmp_path):
    program,expected,tokens=matrix_pair(m=8,n=16,pes=4)
    result=execute(binary,tmp_path/"multiple-pe",program,{"tile_pipeline":True,"template_load_timing":True,"template_trace_limit":100000})
    audit_loading(result)
    actual,ids=outputs(result)[0];np.testing.assert_array_equal(actual,expected);assert ids==tokens
    array=result["array"]
    assert array["template_config_pe_cycles"]>array["template_config_wall_cycles"]
    assert len({e["pe"] for e in array["template_trace"]})>1


@pytest.mark.parametrize("change",[{"template_word_period":0},{"template_word_period":1025},{"template_trace_limit":1000001}])
def test_invalid_template_port_configuration_is_rejected(binary,tmp_path,change):
    program,_,_=matrix_pair()
    execute(binary,tmp_path/"bad",program,{"tile_pipeline":True,"template_load_timing":True,**change},"invalid template programming")
