"""Run a full compiled model through shared native physical memory (not Chipyard)."""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from mlxsim.model_tensor_compiler import compile_inventory
from mlxsim.model_physical_evidence import EXECUTION_CLASSIFICATION, verify_physical_execution
from scripts.run_mlx_tensor_semantics import ROOT,sha,source_identity as tensor_sources,compare_logits

BUILD = ROOT / "build/mlx-physical-model"


def source_identity():
    result=tensor_sources()
    files=[Path(__file__).resolve(),ROOT/"src/mlxsim/model_physical_evidence.py"]
    for name in ("model_system","model_storage"):
        files.extend(p for p in (ROOT/"simulator_ext"/name).iterdir() if p.suffix in {".cc",".h"} or p.name=="CMakeLists.txt")
    for file in files:result[str(file.relative_to(ROOT))]=sha(file)
    return result


def build(output):
    with (output/"build.log").open("w") as log:
        for command in (["cmake","-S",str(ROOT/"simulator_ext/model_system"),"-B",str(BUILD),"-DCMAKE_BUILD_TYPE=Release"],
                        ["cmake","--build",str(BUILD),"--target","mlx-physical-model","-j4"]):
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    return BUILD/"mlx-physical-model"


def runtime_libraries(binary):
    listing=subprocess.run(["ldd",str(binary)],capture_output=True,text=True,check=True).stdout
    result={}
    for line in listing.splitlines():
        part=line.split("=>",1)[-1].strip().split()
        if part and part[0].startswith("/"):
            path=Path(part[0]).resolve();result[str(path)]=sha(path)
    if not result:raise RuntimeError("cannot bind physical executable runtime libraries")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--system-options",type=Path,required=True)
    for name in ("matrix","vector","memory","control"):
        parser.add_argument(f"--{name}-options",type=Path)
    parser.add_argument("--timeout",type=int,default=86400,help="host watchdog seconds; never a target performance result")
    parser.add_argument("--compile-only",action="store_true")
    args=parser.parse_args()
    if args.timeout<=0:raise RuntimeError("host watchdog must be positive")
    output=args.output.resolve()
    if output.exists():raise RuntimeError("choose a fresh physical execution directory")
    output.mkdir(parents=True);sources=source_identity();inventory_hash=sha(args.inventory)
    inventory=json.loads(args.inventory.read_text());system_options=json.loads(args.system_options.read_text())
    options={name:json.loads(getattr(args,name+"_options").read_text()) if getattr(args,name+"_options") else {} for name in ("matrix","vector","memory","control")}
    program,compilation=compile_inventory(inventory,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled",
        schedule_options=options["matrix"],vector_schedule_options=options["vector"],memory_schedule_options=options["memory"],control_schedule_options=options["control"])
    (output/"program.json").write_text(json.dumps(program,indent=2)+"\n")
    (output/"compilation.json").write_text(json.dumps(compilation,indent=2)+"\n")
    (output/"system-options.json").write_text(json.dumps(system_options,indent=2)+"\n")
    program_hash,system_hash=sha(output/"program.json"),sha(output/"system-options.json")
    if args.compile_only:
        print(f"PHYSICAL_MODEL_COMPILED_ONLY calls={len(program['nodes'])}")
        return
    for name,digest in sources.items():
        source=ROOT/name;target=output/"sources"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha(target)!=digest:raise RuntimeError("source changed while snapshotting")
    built=build(output);binary=output/"mlx-physical-model";shutil.copy2(built,binary);binary_hash=sha(binary);libraries=runtime_libraries(binary)
    print("Checking complete weight/tokenizer identities before physical execution",flush=True)
    for name,info in inventory["model_identity"]["files"].items():
        if sha(Path(name))!=info["sha256"]:raise RuntimeError(f"model asset changed: {name}")
    started=time.monotonic();status="running";returncode=None
    manifest={"classification":"native_physical_execution_attempt_not_system_certificate","status":status,"sources":sources,
        "inventory_sha256":inventory_hash,"program_sha256":program_hash,"system_options_sha256":system_hash,"binary_sha256":binary_hash,
        "compiled_source_calls":len(program["nodes"]),"runtime_libraries":libraries,"host_watchdog_seconds":args.timeout,"mlx_system_verified":False,"inference_performance_eligible":False}
    (output/"execution.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Running full shared-physical C++ model: {output/'native.log'}",flush=True)
    with (output/"native.log").open("w") as log:
        try:
            process=subprocess.run([str(binary),str(output/"program.json"),str(output/"system-options.json"),str(output/"native")],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            returncode=process.returncode;status="completed" if returncode==0 else "failed"
        except subprocess.TimeoutExpired:status="host_watchdog_expired"
    manifest.update(status=status,returncode=returncode,host_elapsed_seconds=time.monotonic()-started)
    unchanged=sources==source_identity() and sha(binary)==binary_hash and sha(args.inventory)==inventory_hash and sha(output/"program.json")==program_hash and sha(output/"system-options.json")==system_hash and libraries==runtime_libraries(binary)
    manifest["source_and_program_identity_unchanged"]=unchanged
    (output/"execution.json").write_text(json.dumps(manifest,indent=2)+"\n")
    if not unchanged:raise RuntimeError("physical execution sources/program/binary changed")
    if status!="completed":raise RuntimeError(f"physical execution {status}; no correctness claim: {output/'native.log'}")
    print("Checking complete asset identities after physical execution",flush=True)
    for name,info in inventory["model_identity"]["files"].items():
        if sha(Path(name))!=info["sha256"]:raise RuntimeError(f"model asset changed during execution: {name}")
    native_file=output/"native/result.json";native=json.loads(native_file.read_text());coverage=verify_physical_execution(program,native,system_options)
    reference={row["forward_id"]:row for row in inventory["reference_checks"]}
    if {o["forward_id"] for o in native["outputs"]}!=set(reference):raise RuntimeError("physical reference outputs missing/duplicated")
    comparisons=[compare_logits(actual,reference[actual["forward_id"]]) for actual in native["outputs"]]
    report={"classification":EXECUTION_CLASSIFICATION,"inventory_sha256":inventory_hash,"program_sha256":program_hash,
        "system_options_sha256":system_hash,"binary_sha256":binary_hash,"runtime_libraries":libraries,"sources":sources,"native_report_sha256":sha(native_file),
        "executed_source_calls":native["executed_source_calls"],"coverage":coverage,"comparison":comparisons,
        "all_comparisons_passed":all(row["within_tolerance"] and row["tokens_equal"] for row in comparisons),
        "mlx_system_verified":False,"inference_performance_eligible":False}
    (output/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    if not report["all_comparisons_passed"]:raise RuntimeError(f"physical framework comparison failed: {output/'comparison.json'}")
    print(f"PHYSICAL_FULL_MODEL_NUMERIC_OUTPUTS_PASS (not Chipyard) {output/'comparison.json'}")


if __name__=="__main__":main()
