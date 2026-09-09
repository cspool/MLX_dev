"""Modern source groups, tuple lifetimes and actual CPU QA graph execution."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from system_sim.physical_host.graph_lowering import compile_graph
from system_sim.physical_host.qa_host import audit_output
from mlxsim.model_system_evidence import task_coverage
from test_pair_graph import storage_binary,lifetime
from test_qa_result_contract import binaries,capture,run as native_run

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module",params=[False,True])
def modern_case(request,tmp_path_factory,binaries,storage_binary):
    path=tmp_path_factory.mktemp("modern-qa")
    program,_,_,_=capture(path,length=8,unicode=True,composite=request.param)
    reference=native_run(binaries,"tensor",path/"reference",program)
    life=lifetime(storage_binary,path/"input",program)
    return path,program,life,reference


@pytest.mark.parametrize("paired",[False,True])
def test_modern_graph_preserves_groups_tuple_aliases_and_all_result_roles(modern_case,paired):
    _,program,life,_=modern_case
    blob,plan=compile_graph(program,life,block_pairs=paired)
    assert plan["host_abi_version"]==3 and plan["lowered_calls"]==len(program["nodes"])
    assert plan["original_source_calls"]==len(program.get("source_groups",program["nodes"]))
    assert plan["source_count_basis"]=="lowered_stage_slots"
    assert {r["role"] for r in plan["outputs"]}=={"start_logits","end_logits","context_mask","offsets_utf8"}
    assert len(blob)==plan["command_bytes"]
    task_coverage(program,plan)
    for node,event in zip(program["nodes"],life["events"],strict=True):
        if node["kind"]=="split":
            assert set(event["value_allocations"])=={row["id"] for row in node["split_outputs"]}
            assert all(value==event["allocation"] for value in event["value_allocations"].values())
            task=next(t for t in plan["tasks"] if t["source_id"]==node["source_operator_id"])
            assert task["kind"]==0 and task["bytes"]==0


@pytest.fixture(scope="module")
def cpu_qa_run(modern_case):
    path,program,life,reference=modern_case
    out=path/"spike"
    process=subprocess.run([str(ROOT/".venv/bin/python"),"-m","scripts.run_mlx_spike_graph",
                            "--program",str(path/"input/program.json"),"--lifetimes",str(path/"input/life.json"),
                            "--reference",str(path/"reference/out/result.json"),"--output",str(out)],
                           cwd=ROOT,capture_output=True,text=True,timeout=300)
    assert process.returncode==0,process.stdout+process.stderr
    report=json.loads((out/"report.json").read_text())
    return out,program,reference,report


def test_rv64_executes_qa_readback_and_span_from_actual_logits(cpu_qa_run):
    out,program,reference,report=cpu_qa_run
    assert report["actual_cpu_dispatch"] and report["original_source_calls"]==len(program.get("source_groups",program["nodes"]))
    qa=report["qa_cpu_output"]
    assert qa["actual_cpu_readback_and_span_execution"] and qa["independent_span_recomputation"]
    assert qa["cpu_postprocess_cycle_span"]>0 and qa["reference_checks_and_output_logging_excluded"]
    for actual,expected in zip(qa["outputs"],reference["outputs"],strict=True):
        assert actual["span"]["text"]==expected["span"]["text"]
        for role in ("start_logits","end_logits"):
            assert Path(actual["outputs"][role]["file"]).read_bytes()==Path(expected["outputs"][role]["file"]).read_bytes()
    assert not report["rocket_execution_verified"] and not report["full_model_execution_verified"]
    assert "GOLDEN MUST NOT ENTER PROGRAM" not in (out/"test.c").read_text()


@pytest.mark.parametrize("damage",["missing_logits","duplicate_logits","missing_span","score","clock"])
def test_qa_cpu_evidence_cannot_use_missing_or_modified_records(cpu_qa_run,tmp_path,damage):
    out,_,reference,_=cpu_qa_run
    log=(out/"spike.log").read_text();plan=json.loads((out/"plan.json").read_text());lines=log.splitlines()
    if damage=="missing_logits":lines=[l for l in lines if "role=start_logits" not in l]
    elif damage=="duplicate_logits":lines.append(next(l for l in lines if "role=start_logits" in l))
    elif damage=="missing_span":lines=[l for l in lines if not l.startswith("MLX_QA_RESULT")]
    elif damage=="score":lines=[l.replace("score_bits=","score_bits=0") if l.startswith("MLX_QA_RESULT") else l for l in lines]
    else:lines=["MLX_QA_CPU_CYCLES begin=10 graph_end=9 post_end=12" if l.startswith("MLX_QA_CPU_CYCLES") else l for l in lines]
    with pytest.raises(ValueError):audit_output('\n'.join(lines),plan,reference,tmp_path)


@pytest.mark.parametrize("damage",["group_count","stage_membership","group_metadata","result_role","tuple_output"])
def test_modern_plan_audit_rejects_missing_identity_or_work(modern_case,damage):
    _,program,life,_=modern_case
    _,plan=compile_graph(program,life)
    if damage=="group_count":plan["original_source_calls"]-=1
    elif damage=="stage_membership":plan["stage_to_source_group"][0]+=1
    elif damage=="group_metadata":plan["source_group_table"][0]["stage_count"]+=1
    elif damage=="result_role":plan["outputs"].pop()
    else:
        life=copy.deepcopy(life);event=next(e for e in life["events"] if e["kind"]=="split")
        event["value_allocations"].pop(next(iter(event["value_allocations"])))
        with pytest.raises(ValueError,match="split lifetime"):compile_graph(program,life)
        return
    with pytest.raises(RuntimeError):task_coverage(program,plan)


def test_rv64_runtime_counts_source_group_only_after_every_stage(cpu_qa_run,tmp_path):
    # Reuse the actually built matching plugin; the firmware executes only
    # view tasks so these fourteen group-state checks perform no device work.
    out,_,_,_=cpu_qa_run
    host=ROOT/"system_sim/physical_host";elf=tmp_path/"groups.elf"
    command=["riscv64-unknown-elf-gcc","-std=c11","-O2","-march=rv64imafd","-mabi=lp64d","-mcmodel=medany",
             "-ffreestanding","-fno-builtin","-fno-tree-loop-distribute-patterns","-ffp-contract=off","-Wall","-Wextra","-Werror",
             "-nostdlib","-static","-Wl,--no-relax","-I",str(host),"-T",str(host/"link.ld"),str(host/"start.S"),
             str(host/"control_runtime.c"),str(host/"graph_runtime.c"),str(ROOT/"tests/fixtures/modern_graph_runtime_contract.c"),"-o",str(elf)]
    build=subprocess.run(command,capture_output=True,text=True,timeout=120)
    assert build.returncode==0,build.stdout+build.stderr
    config=tmp_path/"plugin.json";config.write_text(json.dumps(dict(base=2**32,bytes=1048576,report=str(tmp_path/"device.json"))))
    run=subprocess.run([str(ROOT/"build/riscv-fesvr-build/spike"),"--isa=RV64IMAFD","-m64",f"--extlib={out/'libmlx_spike_matrix.so'}",
                        f"--device=mlx_matrix,0x100000000,{config}",str(elf)],capture_output=True,text=True,timeout=120)
    assert run.returncode==0,run.stdout+run.stderr
    device=json.loads((tmp_path/"device.json").read_text())
    assert device["launches"]==0 and device["memory_idle"]
