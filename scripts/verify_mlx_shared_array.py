"""Validate shared matrix/vector resources and compare recorded legacy binaries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_mlx_tensor_semantics import ROOT, sha
from scripts.verify_mlx_physical_model import identity as model_sources


def sources():
    result = model_sources()
    for file in (Path(__file__).resolve(), ROOT / "simulator_ext/shared_schedule/CMakeLists.txt",
                 ROOT / "tests/shared_array_driver.cc", ROOT / "tests/test_shared_array_scheduler.py"):
        result[str(file.relative_to(ROOT))] = sha(file)
    return result


def execute(command, log, *, env=None, timeout=300):
    print(f"SHARED_ARRAY_CHECK {log}", flush=True)
    with log.open("w") as output:
        result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"shared array check failed: {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--asan-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists():
        raise RuntimeError("choose a fresh shared-array verification directory")
    out.mkdir(parents=True); before = sources(); baseline = args.baseline_root.resolve()
    tests = ["tests/test_shared_array_scheduler.py", "tests/test_matrix_window_scheduler.py",
             "tests/test_matrix_external_memory.py", "tests/test_matrix_control_cache.py",
             "tests/test_vector_window_scheduler.py", "tests/test_physical_model.py",
             "tests/test_model_tensor_semantics.py", "tests/test_model_numeric_gate.py"]
    execute([sys.executable, "-m", "pytest", "-q", *tests, f"--basetemp={out / 'pytest'}",
             f"--junitxml={out / 'regression.xml'}"], out / "pytest.log", timeout=900)
    sys.path.insert(0, str(ROOT / "tests"))
    from test_matrix_window_scheduler import make_job as matrix_job
    from test_vector_window_scheduler import make_job as vector_job
    from test_matrix_external_memory import external_job
    jobs = []
    for precision in ("f16", "f32"):
        for m, n, k, bias in ((3, 19, 0, True), (5, 37, 5, False), (3, 19, 65, True)):
            job, _ = matrix_job(m=m, n=n, k=k, precision=precision, bias=bias, rows=1, columns=2,
                                dma_request_period=3, dma_response_period=5, spm_period=2,
                                writeback_period=3, trace_limit=200000)
            jobs.append((f"matrix-{precision}-{k}", "mlx-matrix-window/mlx-matrix-window", job))
        for kind in ("neg", "rsqrt", "softmax"):
            jobs.append((f"vector-{precision}-{kind}", "mlx-vector-window/mlx-vector-window",
                         vector_job(kind, width=33, rows=2, precision=precision, columns=2)))
        job, _ = external_job(m=3, n=19, k=5, precision=precision, bias=True, dma_response_period=3)
        job["buffered_bridge"] = True
        jobs.append((f"external-{precision}", "mlx-matrix-window/matrix-external-memory", job))
    binaries, comparisons = {}, []
    for name, relative, job in jobs:
        directory = out / name; directory.mkdir()
        (directory / "job.json").write_text(json.dumps(job) + "\n")
        for label, root in (("legacy", baseline), ("shared", ROOT)):
            binary = root / "build" / relative
            binaries[str(binary)] = sha(binary)
            execute([binary, directory / "job.json", directory / label], directory / f"{label}.log")
        old, new = directory / "legacy", directory / "shared"
        if json.loads((old / "result.json").read_text()) != json.loads((new / "result.json").read_text()):
            raise RuntimeError(f"legacy event/cycle/resource behavior changed: {name}")
        if (old / "output.bin").read_bytes() != (new / "output.bin").read_bytes():
            raise RuntimeError(f"legacy numerical output changed: {name}")
        comparisons.append({"case": name, "job_sha256": sha(directory / "job.json"),
                            "result_sha256": sha(new / "result.json"), "output_sha256": sha(new / "output.bin")})
    asan = args.asan_binary.resolve(); binaries[str(asan)] = sha(asan)
    environment = os.environ.copy(); environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    selected = []
    for file in sorted((out / "pytest").rglob("out/result.json")):
        if any(p.is_symlink() for p in file.parents):
            continue
        if file.parent.parent.name not in {"concurrent", "periods", "rom-pressure", "retry"}:
            continue
        report = json.loads(file.read_text())
        if report.get("classification") != "shared_array_two_operator_execution_not_complete_graph_or_system_validation":
            continue
        selected.append(file)
    for index, file in enumerate(selected):
        directory = out / f"asan-{index:02d}"
        execute([asan, file.parent.parent / "job.json", directory], out / f"asan-{index:02d}.log", env=environment)
        if json.loads((directory / "result.json").read_text()) != json.loads(file.read_text()):
            raise RuntimeError("sanitizer changed target events, cycles or resources")
        for name in ("matrix.bin", "vector.bin"):
            if (directory / name).read_bytes() != (file.parent / name).read_bytes():
                raise RuntimeError("sanitizer changed shared-array values")
    if len(selected) < 14:
        raise RuntimeError("missing mixed, retry, timing or ROM-pressure executions")
    if before != sources() or any(sha(Path(p)) != checksum for p, checksum in binaries.items()):
        raise RuntimeError("shared array verification sources or binaries changed")
    result = {"classification": "shared_array_frontend_and_storage_component_validation_not_full_model_or_system",
              "sources": before, "executed_binaries": binaries, "legacy_binary_comparisons": comparisons,
              "legacy_scope": "exact behavior of the recorded existing executables; baseline was not rebuilt here",
              "regression_xml_sha256": sha(out / "regression.xml"), "asan_ubsan_lsan_cases": len(selected),
              "shared_rf_spm_rom_and_service_slots_executed": True,
              "complete_graph_cdc_verified": False, "full_model_execution_verified": False,
              "mlx_system_verified": False, "inference_performance_eligible": False, "rtl_verified": False}
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"SHARED_ARRAY_COMPONENT_CHECKS_PASS {out / 'report.json'}")


if __name__ == "__main__":
    main()
