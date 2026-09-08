"""Verify port-driven RV64 leaves; explicitly not Rocket or full-model timing."""
import argparse
import json
import subprocess
from pathlib import Path

from scripts.run_mlx_tensor_semantics import ROOT, sha, source_identity
from scripts.verify_mlx_controller import cases, assembly


def identity():
    result = source_identity()
    files = [ROOT / "scripts/verify_mlx_control_window.py", ROOT / "scripts/verify_mlx_controller.py",
             ROOT / "tests/control_external_memory.cc", ROOT / "tests/test_control_window_scheduler.py",
             ROOT / "tests/test_model_control_program.py", ROOT / "tests/fixtures/rv64_control_words.S",
             ROOT / "tests/test_model_memory_program.py", ROOT / "tests/test_model_tensor_semantics.py",
             ROOT / "src/mlxsim/model_execution_inventory.py"]
    files += list((ROOT / "simulator_ext/control_schedule").iterdir())
    for path in files:
        if path.is_file(): result[str(path.relative_to(ROOT))] = sha(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spike", type=Path, default=ROOT / "build/riscv-fesvr-build/spike")
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists(): raise RuntimeError("choose a fresh controller-window evidence directory")
    out.mkdir(parents=True); before = identity()
    with (out / "pytest.log").open("w") as log:
        process = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q",
            "tests/test_control_window_scheduler.py", "tests/test_model_control_program.py", "tests/test_model_memory_program.py",
            f"--basetemp={out / 'pytest'}", f"--junitxml={out / 'regression.xml'}"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if process.returncode: raise RuntimeError(f"controller-window tests failed: {out / 'pytest.log'}")
    completed, integrated = [], []
    for file in sorted((out / "pytest").rglob("out/result.json")):
        if any(p.is_symlink() for p in file.parents): continue
        report = json.loads(file.read_text())
        if report.get("control_windows"):
            program_file = file.parent.parent / "program.json"; program = json.loads(program_file.read_text())
            if not all(program.get(name+"_backend") == "scheduled" for name in ("matrix", "vector", "memory", "control")):
                raise RuntimeError("integrated controller case did not use four cycle backends")
            if report["functional_entry_calls"] or report["blas_calls"] or any(not w["done"] or w["dma_requests"] != w["dma_responses"] for w in report["control_windows"]):
                raise RuntimeError("integrated controller case used fallback or failed to drain")
            integrated.append({"case": str(file.parent.parent.relative_to(out)), "program_sha256": sha(program_file),
                "native_report_sha256": sha(file), "source_calls": report["executed_source_calls"],
                "control_calls": len(report["control_windows"]), "logits_sha256": [sha(Path(o["logits_file"])) for o in report["outputs"]]})
        if not report.get("functional_oracle_equal"): continue
        if not report["done"] or report["dma_requests"] != report["dma_responses"]:
            raise RuntimeError("controller cycle case did not drain")
        completed.append({"case": str(file.parent.parent.relative_to(out)), "cycles": report["cycles"],
            "virtual_tensor_backing_used": report["virtual_tensor_backing_used"],
            "instructions": report["instructions"], "branches_taken": report["branches_taken"],
            "dma_requests": report["dma_requests"], "output_sha256": sha(file.parent / "output.bin"),
            "report_sha256": sha(file), "job_sha256": sha(file.parent.parent / "job.json")})
    if len(completed) < 20: raise RuntimeError("missing controller cycle conformance cases")
    if not integrated: raise RuntimeError("missing four-backend model-runner integration evidence")
    # Re-execute the exact registered expectations in an independently built
    # RISC-V ISA simulator. Tensor-port loop behavior is NOT certified by this.
    items = cases(); source, linker, elf = out / "test.S", out / "test.ld", out / "test.elf"
    source.write_text(assembly(items))
    linker.write_text("OUTPUT_ARCH(riscv)\nENTRY(_start)\nSECTIONS { . = 0x80000000; .text : { *(.text.init) *(.text*) } .rodata : { *(.rodata*) } . = ALIGN(64); .tohost : { *(.tohost) } }\n")
    subprocess.run(["riscv64-unknown-elf-gcc", "-march=rv64imafdc", "-mabi=lp64d", "-nostdlib", "-static", "-mcmodel=medany", "-Wl,--no-relax", "-T", str(linker), str(source), "-o", str(elf)], check=True, capture_output=True, timeout=60)
    with (out / "spike.log").open("w") as log:
        process = subprocess.run([str(args.spike.resolve()), "--isa=RV64IMAFDC", "-m64", str(elf)], stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if process.returncode: raise RuntimeError(f"independent Spike expectations failed: {out / 'spike.log'}")
    if before != identity(): raise RuntimeError("controller sources changed during validation")
    binary = ROOT / "build/mlx-control-window/control-external-memory"
    report = {"classification": "controller_port_and_stepped_leaf_component_not_full_model_or_rocket_execution",
        "sources": before, "cases": completed, "integration_cases": integrated, "independent_spike_cases": len(items),
        "binary_sha256": sha(binary), "spike_sha256": sha(args.spike.resolve()),
        "assembly_sha256": sha(source), "elf_sha256": sha(elf), "regression_xml_sha256": sha(out / "regression.xml"),
        "integrated_into_model_runner": True, "full_model_verified": False,
        "rocket_execution_verified": False, "mlx_system_verified": False, "inference_performance_eligible": False}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"CONTROLLER_WINDOW_COMPONENT_VERIFIED {out / 'report.json'}")


if __name__ == "__main__": main()
