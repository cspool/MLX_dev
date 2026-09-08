"""Validate physical allocation lifetimes, explicitly without loading model data."""
import argparse
import json
import subprocess
from pathlib import Path

from scripts.run_mlx_tensor_semantics import ROOT, sha, source_identity


def identity():
    result = source_identity()
    files = [ROOT / "scripts/verify_mlx_model_storage.py", ROOT / "tests/model_storage_contract.cc", ROOT / "tests/test_model_storage.py"]
    files += list((ROOT / "simulator_ext/model_storage").iterdir())
    for path in files:
        if path.is_file(): result[str(path.relative_to(ROOT))] = sha(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists(): raise RuntimeError("choose a fresh storage verification directory")
    out.mkdir(parents=True); before = identity(); program_hash = sha(args.program)
    with (out / "pytest.log").open("w") as log:
        process = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "tests/test_model_storage.py",
            f"--basetemp={out / 'pytest'}", f"--junitxml={out / 'regression.xml'}"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if process.returncode: raise RuntimeError(f"storage regression failed: {out / 'pytest.log'}")
    binary = ROOT / "build/mlx-model-storage/model-storage-contract"
    with (out / "model-lifetimes.log").open("w") as log:
        process = subprocess.run([str(binary), str(args.program.resolve()), str(out / "model-lifetimes.json")], stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if process.returncode: raise RuntimeError(f"model lifetime replay failed: {out / 'model-lifetimes.log'}")
    program = json.loads(args.program.read_text()); lifetime = json.loads((out / "model-lifetimes.json").read_text())
    views = sum(n.get("memory_program", {}).get("mode") == "view" for n in program["nodes"])
    if lifetime["source_nodes"] != len(program["nodes"]) or lifetime["checked_views"] != views or lifetime["allocating_nodes"] != len(program["nodes"])-views:
        raise RuntimeError("physical lifetime replay omitted model nodes")
    initial_bytes = sum(__import__("math").prod(a["shape"]) * {"f16":2,"f32":4,"i64":8,"bool":1}[a["dtype"]] for a in program["assets"].values())
    if lifetime["initial"]["live_bytes"] != initial_bytes: raise RuntimeError("model asset byte count changed")
    drained = lifetime["drained"]
    if drained["reserved_bytes"] or drained["allocations"] or drained["total_allocations"] != drained["total_frees"]:
        raise RuntimeError("model buffers leaked after final result release")
    if [e["source_operator_id"] for e in lifetime["events"]] != [n["source_operator_id"] for n in program["nodes"]]:
        raise RuntimeError("model lifetime replay reordered or omitted nodes")
    if lifetime["model_inference_executed"] or lifetime["model_data_loaded"]:
        raise RuntimeError("metadata replay improperly claims data execution")
    if before != identity() or sha(args.program) != program_hash: raise RuntimeError("storage validation sources/program changed")
    report = {"classification":"physical_buffer_ownership_and_model_lifetimes_not_inference_execution",
        "sources":before,"program_sha256":program_hash,"binary_sha256":sha(binary),
        "regression_xml_sha256":sha(out / "regression.xml"),"lifetime_report_sha256":sha(out / "model-lifetimes.json"),
        "source_nodes":lifetime["source_nodes"],"checked_views":views,"allocating_nodes":lifetime["allocating_nodes"],
        "initial_live_bytes":initial_bytes,"peak_reserved_bytes":drained["peak_reserved_bytes"],"final_reserved_bytes":0,
        "model_data_loaded":False,"model_inference_executed":False,"mlx_system_verified":False,"inference_performance_eligible":False}
    (out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"MODEL_STORAGE_OWNERSHIP_VERIFIED {out / 'report.json'}")


if __name__ == "__main__": main()
