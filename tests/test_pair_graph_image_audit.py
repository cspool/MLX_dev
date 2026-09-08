"""Audit checks against completed real Rocket fixtures, not synthetic PASS flags."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from scripts.audit_mlx_pair_graph_system import device_work,feedback_paths

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"artifacts/tagged/pair-graph-rocket-003"
INPUTS=ROOT/"artifacts/tagged/pair-graph-003"


def load(path):return json.loads(path.read_text())


@pytest.fixture(params=["separated-pair","batch-pair"])
def completed(request):
    name=request.param
    return (load(INPUTS/name/"program.json"),load(RUN/name/"plan.json"),load(RUN/name/"device.json"),load(RUN/"profile.json"))


def test_real_completed_task_work_matches_every_source_batch(completed):
    work=device_work(*completed)
    assert work["pair_launches"]==1 and work["source_calls"] in (4,5)
    assert work["work"]["matrix_mac_lanes"] in (16,840)


def test_optional_observer_does_not_change_execution_audit(completed):
    expected=device_work(*completed);program,plan,device,profile=copy.deepcopy(completed)
    device.pop("progress_observer")
    assert device_work(program,plan,device,profile)==expected


@pytest.mark.parametrize("damage",["source_basis","missing","identity","descriptor","cycle","busy","read_responses","macs","consumer","batch","epoch","context_capacity","context_peak","templates","drain"])
def test_altered_real_system_evidence_is_rejected(completed,damage):
    program,plan,device,profile=copy.deepcopy(completed);window=next(w for w in device["windows"] if w["backend"]=="pair");kernel=window["kernel"]
    if damage=="source_basis":device["source_id_basis"]="model_operator_id"
    elif damage=="missing":device["windows"].pop()
    elif damage=="identity":window["source_id"]+=1
    elif damage=="descriptor":window["descriptor_bytes_fetched"]-=8
    elif damage=="cycle":window["run_cycles"]-=1
    elif damage=="busy":window["busy"]=True
    elif damage=="read_responses":window["transport"]["consumed"]-=1
    elif damage=="macs":kernel["producer_windows"][0]["numeric_instructions"]["mul_active_lanes"]-=1
    elif damage=="consumer":kernel["consumer_source"]+=1
    elif damage=="batch":kernel["producer_windows"].pop()
    elif damage=="epoch":kernel["block_events"]["epoch"]+=1
    elif damage=="context_capacity":kernel["array"]["context_slots_per_pe"]+=1
    elif damage=="context_peak":kernel["array"]["peak_contexts"]=100000
    elif damage=="templates":kernel["array"]["pending_template_words"]=1
    else:window["transport"]["inflight"]=1
    with pytest.raises(RuntimeError):device_work(program,plan,device,profile)


@pytest.mark.parametrize("mode",["running","failed","duplicate_marker"])
def test_audit_refuses_nonterminal_or_bad_cpu_marker_before_rebuild(tmp_path,mode):
    run=tmp_path/"run";case=run/"case";case.mkdir(parents=True)
    state={"status":"running" if mode=="running" else "exited","exit_code":1 if mode=="failed" else 0,"validation":"registered_graph_checks_passed"}
    (case/"execution.json").write_text(json.dumps(state));(case/"chipyard.log").write_text("MLX_CLOCKED_CHAIN_PASS\n"*(2 if mode=="duplicate_marker" else 1))
    cases=tmp_path/"cases.json";cases.write_text(json.dumps([{"name":"case"}]))
    output=tmp_path/"audit";process=subprocess.run([str(ROOT/".venv/bin/python"),"-m","scripts.audit_mlx_pair_graph_system","--run",str(run),"--cases",str(cases),"--output",str(output)],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert process.returncode!=0 and not output.exists()
    assert "successful terminal result" in process.stderr if mode!="duplicate_marker" else "unique terminal marker" in process.stderr


def generation():return load(INPUTS/"pytest/test_generated_rv64_dispatch_r0/program.json")


def test_generation_proof_follows_computed_tokens_and_growing_cache():
    paths=feedback_paths(generation())
    assert paths["forwards"]==3 and [r["cache_length"] for r in paths["links"]]==[1,2,3]
    assert [r["index_source"] for r in paths["links"]][1:]==["v13","v28"]


@pytest.mark.parametrize("damage",["token_literal","cache_reset","argmax_input","cache_length","missing_forward"])
def test_generation_proof_rejects_reference_feedback_or_lost_state(damage):
    program=generation();nodes={n["id"]:n for n in program["nodes"]}
    if damage=="token_literal":nodes["v15"]["args"][1]["value"]="asset:t2"
    elif damage=="cache_reset":nodes["v16"]["args"][0][0]["value"]="asset:t3"
    elif damage=="argmax_input":nodes["v13"]["args"][0]["value"]="v9"
    elif damage=="cache_length":nodes["v16"]["output"]["shape"][1]=1
    else:program["outputs"].pop()
    with pytest.raises(RuntimeError):feedback_paths(program)
