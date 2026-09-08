"""Audit pair task compilation, lifetimes, device execution and RV64 contracts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from scripts.run_mlx_clocked_chipyard import ROOT,source_identity
from scripts.run_mlx_spike_graph import sha
from scripts.verify_mlx_ready_graph import run
from system_sim.physical_host.graph_lowering import compile_graph
from mlxsim.model_system_evidence import task_coverage


TESTS=["tests/test_pair_graph.py","tests/test_pair_wire.py","tests/test_spike_graph_runtime.py","tests/test_system_model_evidence.py","tests/test_system_image.py",
       "tests/test_block_pipeline.py","tests/test_ready_graph.py","tests/test_shared_array_scheduler.py","tests/test_template_loading.py",
       "tests/test_clocked_device.py","tests/test_clocked_rocc.py","tests/test_system_progress.py"]


def sources():
    result=source_identity()
    for file in [Path(__file__).resolve(),ROOT/"scripts/verify_mlx_graph_runtime.py",*sorted((ROOT/"tests").glob("*.py")),ROOT/"tests/fixtures/graph_runtime_contract.c"]:
        result[str(file.relative_to(ROOT))]=sha(file)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--asan-binary",type=Path,required=True);parser.add_argument("--asan-rocc-binary",type=Path,required=True);parser.add_argument("--full-program",type=Path);parser.add_argument("--full-lifetimes",type=Path)
    args=parser.parse_args();out=args.output.resolve();asan=args.asan_binary.resolve();rocc_asan=args.asan_rocc_binary.resolve()
    if out.exists() or bool(args.full_program)!=bool(args.full_lifetimes):raise RuntimeError("fresh output and both full compilation inputs required")
    out.mkdir(parents=True);before=sources();asan_hash=sha(asan);rocc_hash=sha(rocc_asan)
    run([sys.executable,"-m","pytest","-q",*TESTS,f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],out/"pytest.log",timeout=900)
    run([sys.executable,"-m","scripts.verify_mlx_graph_runtime","--output",out/"rv64-contract"],out/"rv64.log",timeout=300)
    env=os.environ.copy();env["ASAN_OPTIONS"]="detect_leaks=1:halt_on_error=1";env["UBSAN_OPTIONS"]="halt_on_error=1:print_stacktrace=1"
    checks=[]
    for file in sorted((out/"pytest").rglob("job.json")):
        if any(p.is_symlink() for p in file.parents) or not (file.parent/"report.json").exists():continue
        job=json.loads(file.read_text())
        if not any(len(c.get("data",[]))==8832 for c in job.get("commands",[])):continue
        expected=file.parent/"report.json";report=json.loads(expected.read_text())
        adapter=report.get("classification")=="clocked_cpp_rocc_requestor_adapter_execution_scope_requires_external_evidence"
        if not adapter and not isinstance(report.get("launches"),list):raise RuntimeError("unknown sanitizer driver report schema")
        replay=file
        if "progress" in job:
            job["progress"]=str(out/f"asan-{len(checks):02d}-progress.json")
            replay=out/f"asan-{len(checks):02d}-job.json";replay.write_text(json.dumps(job)+"\n")
        target=out/f"asan-{len(checks):02d}.json";run([rocc_asan if adapter else asan,replay,target],out/f"asan-{len(checks):02d}.log",environment=env)
        if json.loads(target.read_text())!=report:raise RuntimeError("sanitizer pair graph output/event mismatch")
        checks.append({"driver":"rocc_adapter" if adapter else "clocked_device","job":str(file.relative_to(out)),"job_sha256":sha(file),"replay_job":str(replay.relative_to(out)),"replay_job_sha256":sha(replay),"report_sha256":sha(expected),"sanitizer_sha256":sha(target)})
    if sum(c["driver"]=="clocked_device" for c in checks)<21 or sum(c["driver"]=="rocc_adapter" for c in checks)<2:raise RuntimeError("missing actual pair device/observer/replanned graph checks")
    full=None
    if args.full_program:
        program_path=args.full_program.resolve();life_path=args.full_lifetimes.resolve();fingerprints={str(p):sha(p) for p in (program_path,life_path)}
        program=json.loads(program_path.read_text());life=json.loads(life_path.read_text());blob,plan=compile_graph(program,life,block_pairs=True,event_slots=32,device_base=0x88000000,device_bytes=16*2**30-128*2**20)
        task_coverage(program,plan);(out/"full-plan.json").write_text(json.dumps(plan,indent=2)+"\n")
        if any(sha(Path(p))!=h for p,h in fingerprints.items()):raise RuntimeError("full compiler inputs changed")
        full={"classification":"complete_program_compilation_not_inference_execution","inputs":fingerprints,"source_calls":plan["source_calls"],"pair_count":plan["pair_count"],
              "nonadjacent_pairs":sum(len(g)==2 and g[1]>g[0]+1 for g in plan["execution_groups"]),"task_count":plan["task_count"],"command_bytes":len(blob),
              "commands_sha256":hashlib.sha256(blob).hexdigest(),"plan_sha256":sha(out/"full-plan.json"),"required_mapped_bytes":plan["required_mapped_bytes"],"model_data_executed":False}
    # Prepare reproducible complete small graphs for the next actual Rocket run.
    sys.path.insert(0,str(ROOT/"tests"));import test_pair_graph as fixture
    cases=[]
    for name,maker in (("separated-pair",fixture.separated_pair),("batch-pair",fixture.batch_pair)):
        program,expected=maker();directory=out/name;fixture.lifetime(ROOT/"build/mlx-model-storage/model-storage-contract",directory,program)
        (directory/"logits.bin").write_bytes(expected.tobytes());reference={"outputs":[{"forward_id":0,"dtype":"f16" if expected.dtype.itemsize==2 else "f32","shape":list(expected.shape),"tokens":expected.argmax(-1).reshape(-1).tolist(),"logits_file":str(directory/"logits.bin")}]}
        (directory/"reference.json").write_text(json.dumps(reference)+"\n")
        cases.append({"name":name,"program":str(directory/"program.json"),"lifetimes":str(directory/"life.json"),"reference":str(directory/"reference.json"),"block_pairs":True})
    for directory in sorted((out/"pytest").glob("test_generated_rv64_dispatch_r[01]")):
        state=json.loads((directory/"run/execution.json").read_text());life=next(p for p in state["inputs"] if p.endswith('/life.json'))
        cases.append({"name":"generation-"+directory.name[-1],"program":str(directory/"program.json"),"lifetimes":life,"reference":str(directory/"reference.json"),"block_pairs":True})
    if len(cases)!=4:raise RuntimeError("missing normal/perturbed actual-feedback generation inputs")
    (out/"system-cases.json").write_text(json.dumps(cases,indent=2)+"\n")
    if sources()!=before or sha(asan)!=asan_hash or sha(rocc_asan)!=rocc_hash:raise RuntimeError("pair graph verification sources/binary changed")
    report={"classification":"pair_graph_compiler_device_and_rv64_contract_not_actual_rocket_graph_or_full_model_acceptance","sources":before,"regression_xml_sha256":sha(out/"regression.xml"),
            "sanitizer_cases":checks,"asan_binary_sha256":asan_hash,"asan_rocc_binary_sha256":rocc_hash,"rv64_contract_sha256":sha(out/"rv64-contract/report.json"),"full_compilation":full,
            "system_cases_sha256":sha(out/"system-cases.json"),"actual_rocket_graph_execution":False,"full_model_execution_verified":False,"inference_performance_eligible":False,"rtl_verified":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"PAIR_GRAPH_COMPONENT_CHECKS_PASS {out/'report.json'}",flush=True)


if __name__=="__main__":main()
