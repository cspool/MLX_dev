"""Bind memory cycle/port tests and full-model compilation coverage to sources."""
import argparse
from collections import Counter
import json
import subprocess
from pathlib import Path

from mlxsim.model_tensor_compiler import compile_inventory
from scripts.run_mlx_tensor_semantics import ROOT, sha, source_identity


def identity():
    result = source_identity()
    for name in ("scripts/verify_mlx_memory_window.py", "tests/memory_external_memory.cc",
                 "tests/test_memory_window_scheduler.py", "tests/test_model_memory_program.py",
                 "tests/test_model_tensor_semantics.py", "scripts/verify_mlx_model_numeric.py"):
        result[name] = sha(ROOT / name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prior-program", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists(): raise RuntimeError("choose a fresh memory-window evidence directory")
    out.mkdir(parents=True); before = identity()
    files = {str(p.resolve()): sha(p) for p in (args.inventory, args.prior_program)}
    inventory = json.loads(args.inventory.read_text())
    prior = json.loads(args.prior_program.read_text())
    legacy, _ = compile_inventory(inventory, matrix_backend="microcode", vector_backend="microcode",
                                  memory_backend="planned", control_backend="rv64_leaf")
    if legacy != prior: raise RuntimeError("compiler changed the prior full-model functional program")
    program, routing = compile_inventory(inventory, matrix_backend="microcode", vector_backend="microcode",
                                         memory_backend="scheduled", control_backend="rv64_leaf")
    if program["assets"] != prior["assets"] or program["nodes"] != prior["nodes"] or program["outputs"] != prior["outputs"]:
        raise RuntimeError("scheduled memory changed model bindings, computation, or token dependencies")
    counts = Counter()
    for node, route in zip(program["nodes"], routing["routes"], strict=True):
        paths = [name for name in ("matrix_program", "vector_program", "memory_program", "control_program") if name in node]
        if len(paths) != 1: raise RuntimeError("source operator lacks exactly one executable route")
        counts[paths[0]] += 1
        if paths[0] == "memory_program" and route["memory_plan_entry"] != "mlx::memory_model::Simulator":
            raise RuntimeError("memory route did not select the cycle engine")
    (out / "program.json").write_text(json.dumps(program, indent=2) + "\n")
    (out / "compilation.json").write_text(json.dumps(routing, indent=2) + "\n")
    with (out / "pytest.log").open("w") as log:
        process = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q",
            "tests/test_memory_window_scheduler.py", "tests/test_model_memory_program.py", "tests/test_model_numeric_gate.py",
            f"--basetemp={out / 'pytest'}", f"--junitxml={out / 'regression.xml'}"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if process.returncode: raise RuntimeError(f"memory component tests failed: {out / 'pytest.log'}")
    cases = []
    for file in sorted((out / "pytest").rglob("out/result.json")):
        if any(p.is_symlink() for p in file.parents): continue
        report = json.loads(file.read_text())
        if not report.get("functional_oracle_equal"): continue
        windows = report["windows"]
        if not all(w["done"] and w["dma_requests"] == w["dma_responses"] for w in windows):
            raise RuntimeError("memory cycle case did not drain")
        case_job = file.parent.parent / "job.json"
        cases.append({"case": str(file.parent.parent.relative_to(out)),
            "virtual_tensor_backing_used": report["virtual_tensor_backing_used"],
            "cycles": sum(w["cycles"] for w in windows), "dma_requests": sum(w["dma_requests"] for w in windows),
            "physical_commits": report["physical_commits"], "transport": report.get("transport"),
            "output_sha256": sha(file.parent / "output.bin"), "report_sha256": sha(file), "job_sha256": sha(case_job)})
    if len(cases) < 16 or not any(c["transport"] and c["transport"]["nacks"] for c in cases):
        raise RuntimeError("missing positive memory/transport cases")
    if before != identity() or any(sha(Path(p)) != digest for p, digest in files.items()):
        raise RuntimeError("memory evidence inputs changed during validation")
    binary = ROOT / "build/mlx-memory-window/memory-external-memory"
    report = {"classification": "memory_transfer_component_and_full_model_compilation_not_full_model_execution",
        "sources": before, "inputs": files, "cases": cases, "compiled_route_counts": dict(counts),
        "prior_functional_program_unchanged": True, "compiled_program_sha256": sha(out / "program.json"),
        "binary_sha256": sha(binary), "regression_xml_sha256": sha(out / "regression.xml"),
        "full_model_executed_by_this_report": False, "mlx_system_verified": False, "inference_performance_eligible": False}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"MEMORY_COMPONENT_VERIFIED {out / 'report.json'}")


if __name__ == "__main__": main()
