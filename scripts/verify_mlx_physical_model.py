"""Verify shared physical graph execution; full-model compilation is separate."""
import argparse
from collections import Counter
import json
import subprocess
from pathlib import Path

from mlxsim.model_tensor_compiler import compile_inventory
from scripts.run_mlx_tensor_semantics import ROOT, sha, source_identity


def identity():
    result = source_identity()
    files = [ROOT / "scripts/verify_mlx_physical_model.py", ROOT / "scripts/run_mlx_physical_model.py", ROOT / "src/mlxsim/model_physical_evidence.py", ROOT / "tests/test_physical_model.py", ROOT / "tests/physical_memory_contract.cc",
        ROOT / "tests/test_model_storage.py", ROOT / "tests/model_storage_contract.cc",
        ROOT / "tests/test_model_memory_program.py", ROOT / "tests/test_model_tensor_semantics.py",
        ROOT / "src/mlxsim/model_execution_inventory.py"]
    for name in ("model_system", "model_storage"): files += list((ROOT / "simulator_ext" / name).iterdir())
    for path in files:
        if path.is_file(): result[str(path.relative_to(ROOT))] = sha(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prior-program", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists(): raise RuntimeError("choose a fresh physical-model evidence directory")
    out.mkdir(parents=True); before = identity()
    inputs = {str(p.resolve()):sha(p) for p in (args.inventory,args.prior_program)}
    inventory = json.loads(args.inventory.read_text()); prior = json.loads(args.prior_program.read_text())
    program, routing = compile_inventory(inventory,matrix_backend="scheduled",vector_backend="scheduled",memory_backend="scheduled",control_backend="scheduled")
    if any(program[name] != prior[name] for name in ("nodes","assets","outputs")):
        raise RuntimeError("physical scheduling selection changed model data bindings or computation")
    counts = Counter()
    for node in program["nodes"]:
        paths = [name for name in ("matrix_program","vector_program","memory_program","control_program") if name in node]
        if len(paths) != 1: raise RuntimeError("model node lacks a unique compiled backend")
        counts[paths[0]] += 1
    (out / "program.json").write_text(json.dumps(program,indent=2)+"\n")
    (out / "compilation.json").write_text(json.dumps(routing,indent=2)+"\n")
    with (out / "pytest.log").open("w") as log:
        process = subprocess.run([str(ROOT / ".venv/bin/python"),"-m","pytest","-q","tests/test_physical_model.py","tests/test_model_storage.py",
            f"--basetemp={out / 'pytest'}",f"--junitxml={out / 'regression.xml'}"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if process.returncode: raise RuntimeError(f"physical model tests failed: {out / 'pytest.log'}")
    cases = []
    for file in sorted((out / "pytest").rglob("out/result.json")):
        if any(p.is_symlink() for p in file.parents): continue
        report = json.loads(file.read_text())
        if report.get("classification") != "shared_native_physical_model_execution_not_chipyard_system_validation": continue
        if report["arena_drained"]["reserved_bytes"] or not report["memory"]["idle"] or report["memory"]["submitted"] != report["memory"]["consumed"]:
            raise RuntimeError("physical graph did not drain storage and transactions")
        cases.append({"case":str(file.parent.parent.relative_to(out)),"source_calls":report["executed_source_calls"],
            "backend_windows":{name:len(w) for name,w in report["windows"].items()},"requests":report["memory"]["submitted"],
            "nacks":report["memory"]["nacks"],"physical_reads":report["memory"]["reads"],"physical_writes":report["memory"]["writes"],
            "program_sha256":sha(file.parent.parent / "program.json"),"options_sha256":sha(file.parent.parent / "system-options.json"),
            "result_sha256":sha(file),"logits_sha256":[sha(Path(o["logits_file"])) for o in report["outputs"]]})
    if len(cases)<8 or not any(c["nacks"] and all(c["backend_windows"].values()) for c in cases):
        raise RuntimeError("missing physical full-path/perturbation/retry cases")
    if before != identity() or any(sha(Path(name))!=digest for name,digest in inputs.items()):
        raise RuntimeError("physical model verification inputs changed")
    binary = ROOT / "build/mlx-physical-model/mlx-physical-model"
    result = {"classification":"four_backend_shared_physical_graph_validation_not_full_model_or_chipyard_execution",
        "sources":before,"inputs":inputs,"cases":cases,"compiled_model_route_counts":dict(counts),
        "compiled_model_program_sha256":sha(out / "program.json"),"binary_sha256":sha(binary),"regression_xml_sha256":sha(out / "regression.xml"),
        "full_model_execution_verified":False,"real_weight_loader_verified":False,"cross_operator_overlap_verified":False,
        "mlx_system_verified":False,"inference_performance_eligible":False}
    (out / "report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(f"PHYSICAL_GRAPH_COMPONENT_VERIFIED {out / 'report.json'}")


if __name__ == "__main__": main()
