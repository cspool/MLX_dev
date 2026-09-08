"""Verify real ready-graph executions without promoting startup checks to inference."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_mlx_tensor_semantics import ROOT, sha
from scripts.verify_mlx_physical_model import identity as model_sources


def sources():
    result = model_sources()
    for file in (Path(__file__).resolve(), ROOT / "tests/test_ready_graph.py", ROOT / "tests/physical_mux_contract.cc",
                 ROOT / "tests/test_shared_array_scheduler.py", ROOT / "tests/shared_array_driver.cc"):
        result[str(file.relative_to(ROOT))] = sha(file)
    return result


def run(command, log, *, environment=None, timeout=300, expected=0):
    print(f"READY_GRAPH_CHECK {log}", flush=True)
    with log.open("w") as output:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, env=environment, stdout=output,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode != expected:
        raise RuntimeError(f"ready graph check failed: {log}")
    return result.returncode


def normalized(report):
    result = copy.deepcopy(report)
    for output in result["outputs"]:
        output["logits_file"] = "$OUTPUT/" + Path(output["logits_file"]).name
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asan-build", type=Path, required=True)
    parser.add_argument("--full-program", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists():
        raise RuntimeError("choose a fresh ready graph validation directory")
    out.mkdir(parents=True); before = sources(); full_hash = sha(args.full_program)
    tests = ["tests/test_ready_graph.py", "tests/test_shared_array_scheduler.py", "tests/test_physical_model.py",
             "tests/test_model_storage.py", "tests/test_model_tensor_semantics.py", "tests/test_model_numeric_gate.py",
             "tests/test_matrix_window_scheduler.py", "tests/test_matrix_external_memory.py",
             "tests/test_matrix_control_cache.py", "tests/test_vector_window_scheduler.py"]
    run([sys.executable, "-m", "pytest", "-q", *tests, f"--basetemp={out / 'pytest'}",
         f"--junitxml={out / 'regression.xml'}"], out / "pytest.log", timeout=900)
    release = ROOT / "build/mlx-ready-graph/mlx-ready-graph"
    asan = args.asan_build.resolve() / "mlx-ready-graph"
    contract = args.asan_build.resolve() / "physical-mux-contract"
    binaries = {str(p): sha(p) for p in (release, asan, contract)}
    environment = os.environ.copy(); environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    run([contract], out / "asan-mux.log", environment=environment)
    cases = []
    for file in sorted((out / "pytest").rglob("out/result.json")):
        if any(p.is_symlink() for p in file.parents):
            continue
        report = json.loads(file.read_text())
        if report.get("classification") != "ready_graph_shared_array_execution_not_cdc_cpu_or_system_acceptance":
            continue
        directory = file.parent.parent
        destination = out / f"asan-{len(cases):02d}"
        run([asan, directory / "program.json", directory / "options.json", destination], out / f"asan-{len(cases):02d}.log", environment=environment)
        compared = json.loads((destination / "result.json").read_text())
        if normalized(compared) != normalized(report):
            raise RuntimeError("sanitizer changed graph events, cycles, ownership or resources")
        for a, b in zip(report["outputs"], compared["outputs"], strict=True):
            if Path(a["logits_file"]).read_bytes() != Path(b["logits_file"]).read_bytes():
                raise RuntimeError("sanitizer changed graph numerical results")
        cases.append({"case": str(directory.relative_to(out)), "source_calls": report["executed_source_calls"],
                      "windows": {k: len(v) for k, v in report["windows"].items()},
                      "peak_active_nodes": report["peak_active_nodes"], "requests": report["memory"]["submitted"],
                      "program_sha256": sha(directory / "program.json"), "options_sha256": sha(directory / "options.json"),
                      "report_sha256": sha(file), "sanitizer_report_sha256": sha(destination / "result.json")})
    if len(cases) < 8:
        raise RuntimeError("missing generation, perturbation, join, batch or lifetime executions")
    options = {"base": 2**32, "bytes": 16 * 2**30, "max_cycles": 1, "max_active_nodes": 32,
               "operator_progress": True, "memory": {"trace_limit": 0}}
    (out / "startup-options.json").write_text(json.dumps(options) + "\n")
    run([release, args.full_program.resolve(), out / "startup-options.json", out / "startup"], out / "startup.log", expected=1)
    log = (out / "startup.log").read_text()
    if "ready graph exceeded global cycle budget" not in log or (out / "startup/result.json").exists():
        raise RuntimeError("full-program startup did not stop at the explicit one-edge limit")
    if before != sources() or sha(args.full_program) != full_hash or any(sha(Path(p)) != h for p, h in binaries.items()):
        raise RuntimeError("ready graph validation sources or binaries changed")
    report = {"classification": "ready_graph_four_backend_component_validation_not_full_model_or_cdc_acceptance",
              "sources": before, "binaries": binaries, "cases": cases, "asan_ubsan_lsan_cases": len(cases),
              "regression_xml_sha256": sha(out / "regression.xml"),
              "full_program_startup": {"program_sha256": full_hash, "log_sha256": sha(out / "startup.log"),
                                       "cycle_limit": 1, "exit_code": 1, "full_inference_result": False},
              "full_model_execution_verified": False, "complete_cdc_verified": False,
              "mlx_system_verified": False, "inference_performance_eligible": False, "rtl_verified": False}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"READY_GRAPH_COMPONENT_CHECKS_PASS {out / 'report.json'}")


if __name__ == "__main__":
    main()
