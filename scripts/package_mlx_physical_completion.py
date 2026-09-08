"""Package a completed native run, retaining failed GPU and closed system gates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from scripts.run_mlx_tensor_semantics import sha


def require(value,message):
    if not value:raise RuntimeError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("executed-root","run","source-inventory","numeric-reference","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();root=args.executed_root.resolve();run=args.run.resolve();out=args.output.resolve()
    require(not out.exists(),"choose a fresh completion package")
    require(run.is_relative_to(root),"execution directory is outside its source repository")
    execution=json.loads((run/"execution.json").read_text());comparison=json.loads((run/"comparison.json").read_text())
    numeric=json.loads((run/"numeric-conformance-001.json").read_text());native=json.loads((run/"native/result.json").read_text())
    require(execution["status"]=="completed" and execution["returncode"]==0 and execution["source_and_program_identity_unchanged"],"native execution was not completed with fixed sources")
    require(numeric["numeric_conformance_passed"] and numeric["executed_source_calls"]==native["executed_source_calls"]==6181
            and numeric["used_parameter_tensors"]==291 and numeric["remaining_functional_source_calls"]==0,"complete numeric contract was not verified")
    require(numeric["historical_framework_comparison_passed"]==comparison["all_comparisons_passed"],"GPU comparison was relabelled")
    require(all(not report["mlx_system_verified"] and not report["inference_performance_eligible"] for report in (native,numeric,comparison)),"native evidence cannot promote system or inference performance")
    require(sha(run/"native/result.json")==comparison["native_report_sha256"]==numeric["native_report_sha256"],"actual native report changed")
    require(sha(run/"program.json")==execution["program_sha256"]==comparison["program_sha256"]==numeric["program_sha256"],"executed program changed")
    require(sha(run/"mlx-physical-model")==execution["binary_sha256"]==numeric["binary_sha256"],"attempt-owned binary changed")
    require(numeric["execution_source_snapshot"]==execution["sources"],"numeric audit used different executed sources")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    copies=[]
    for name in ("program.json","compilation.json","execution.json","comparison.json","numeric-conformance-001.json",
                 "system-options.json","build.log","native.log","native/result.json","native/observations.json"):
        copies.append((run/name,Path(name)))
    for name,digest in execution["sources"].items():
        rel=Path(name);require(not rel.is_absolute() and ".." not in rel.parts,"invalid source snapshot path")
        snapshot=run/"sources"/rel
        require(sha(snapshot)==digest,"executed source snapshot changed")
        committed=subprocess.check_output(["git","show",f"{commit}:{name}"],cwd=root)
        require(hashlib.sha256(committed).hexdigest()==digest,"executed snapshot differs from the claimed repository commit")
        copies.append((snapshot,Path("sources")/rel))
    require(sha(args.source_inventory)==numeric["source_inventory_sha256"] and sha(args.numeric_reference)==numeric["numeric_reference_inventory_sha256"],"reference inventory changed")
    for label,path in (("framework",args.source_inventory.resolve()),("numeric",args.numeric_reference.resolve())):
        copies.append((path,Path("references")/label/"inventory.json"))
        inventory=json.loads(path.read_text())
        require(inventory["model_identity"]["parameters"]==6738415616 and inventory["model_identity"]["config"]["num_hidden_layers"]==32,"reference model identity changed")
        for row in inventory["reference_checks"]:
            file=Path(row["logits_file"]);require(sha(file)==row["logits_sha256"],"reference logits changed")
            copies.append((file,Path("references")/label/f"logits-{row['forward_id']}.f16.bin"))
    for row,check in zip(native["outputs"],numeric["comparisons"],strict=True):
        file=Path(row["logits_file"])
        require(file.resolve().is_relative_to(run) and row["forward_id"]==check["forward_id"] and sha(file)==check["logits_sha256"],"actual output identity changed")
        copies.append((file,Path("native")/file.name))
    before={str(source):sha(source) for source,_ in copies}
    require(len({str(target) for _,target in copies})==len(copies),"duplicate package destination")
    out.mkdir(parents=True);entries=[]
    for source,target in copies:
        destination=out/target;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        digest=sha(destination);require(digest==before[str(source)],"package copy differs from bound source")
        entries.append({"path":str(target),"source_path":str(source),"bytes":destination.stat().st_size,"sha256":digest})
    require(all(sha(Path(p))==h for p,h in before.items()),"completion inputs changed while packaging")
    report={"classification":"completed_full_public_dense_native_numeric_contract_with_original_gpu_gate_preserved",
            "executed_repository_commit":commit,"executed_snapshot_matches_commit":True,"original_run":str(run),
            "cpp_process_returncode":0,"numeric_conformance_passed":True,"historical_framework_comparison_passed":comparison["all_comparisons_passed"],
            "model_parameters":6738415616,"model_layers":32,"used_parameter_tensors":291,"executed_source_calls":6181,
            "matrix_mac_lanes":numeric["matrix_mac_lanes"],"physical_execution_coverage":numeric["physical_execution_coverage"],
            "actual_tokens":[row["tokens"] for row in native["outputs"]],"bitwise_equal_numeric_logits":sum(row["elements"] for row in numeric["comparisons"]),
            "framework_comparison":comparison["comparison"],"atomic_primitives_shared_with_cpp_fu":True,
            "model_variant":numeric["model_variant"],"cross_operator_execution":native["cross_operator_execution"],
            "mlx_system_verified":False,"inference_performance_eligible":False,
            "file_count":len(entries),"total_bytes":sum(e["bytes"] for e in entries),"files":entries,
            "packager_sha256":sha(Path(__file__))}
    (out/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"NATIVE_COMPLETION_PACKAGED numeric_contract=true gpu_gate={comparison['all_comparisons_passed']} {out/'manifest.json'}")


if __name__=="__main__":main()
