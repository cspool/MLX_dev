"""Recompile unchanged generated C++ in an isolated directory (host diagnostic).

The original simulator, RTL, static backend libraries and running attempts are
never rewritten. This is not a new system/model acceptance certificate.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from scripts.mlx_system_attempt import digest, record


def prepare(generated, output, optimization):
    generated, output = Path(generated).resolve(), Path(output).resolve()
    if optimization not in ("-Os", "-O3"):
        raise ValueError("only explicit, non-fast-math host optimizations are supported")
    if output.exists():
        raise ValueError("host rebuild needs a fresh output directory")
    makefile = generated / "VTestHarness.mk"
    if not makefile.is_file() or not (generated / "VTestHarness_classes.mk").is_file():
        raise ValueError("expected pinned VTestHarness generated C++ build")
    # GNU make / compiler argument syntax is deliberately narrow for this tool.
    if any(re.search(r"[^a-zA-Z0-9_/.-]", str(p)) for p in (generated, output)):
        raise ValueError("unsafe path for generated make syntax")
    output.mkdir(parents=True)
    objects = output / "objects"
    objects.mkdir()
    copied, external = {}, {}
    for source in sorted(generated.iterdir()):
        if source.is_file() and source.suffix in (".cpp", ".h", ".mk"):
            copied[str(source)] = digest(source)
            shutil.copy2(source, objects / source.name)
    # Original user sources/libraries and forced headers remain read-only. Bind
    # them too; do not mistake a generated-code-only snapshot for full provenance.
    for path in re.findall(r"/[a-zA-Z0-9_/.-]+", makefile.read_text()):
        source = Path(path)
        if source.is_file():
            external[str(source.resolve())] = digest(source)
    verilator = re.search(r"^VERILATOR_ROOT = (.+)$", makefile.read_text(), re.M)
    if not verilator:
        raise ValueError("missing Verilator runtime path")
    include = Path(verilator.group(1)) / "include"
    for source in sorted(include.rglob("*")):
        if source.is_file() and source.suffix in (".cpp", ".c", ".h", ".mk"):
            external[str(source.resolve())] = digest(source)
    wrapper = objects / "host.mk"
    wrapper.write_text(
        "include VTestHarness.mk\n"
        ".PHONY: mlx-host-diagnostic\n"
        "mlx-host-diagnostic: $(VK_USER_OBJS) $(VK_GLOBAL_OBJS) $(VM_PREFIX)__ALL.a\n"
        "\t$(LINK) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) $(LIBS) $(SC_LIBS) "
        f"-o {output / 'simulator'}\n"
    )
    command = ["make", "-C", str(objects), "-f", "host.mk", "mlx-host-diagnostic",
               "CFG_CXXFLAGS_STD_NEWEST=-std=gnu++17", f"OPT_FAST={optimization}",
               f"OPT_GLOBAL={optimization}", f"OPT_SLOW={optimization}", "-j4"]
    report = {"classification": "host_compiler_flag_diagnostic_not_system_acceptance",
              "status": "prepared", "generated_sources": copied,
              "external_build_inputs": external, "wrapper_sha256": digest(wrapper),
              "command": command, "optimization": optimization,
              "hardware_or_simulator_source_changed": False,
              "full_model_execution_verified": False, "inference_performance_eligible": False}
    record(output / "build.json", report)
    return command, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimization", choices=("Os", "O3"), default="O3")
    args = parser.parse_args()
    out = args.output.resolve()
    command, report = prepare(args.generated, out, "-" + args.optimization)
    # Retain a dry-run showing every output destination before starting any build.
    with (out / "dry-run.log").open("w") as log:
        subprocess.run(command + ["-n"], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=60)
    start = time.monotonic()
    report["status"] = "building"
    record(out / "build.json", report)
    with (out / "build.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report.update(exit_code=result.returncode, host_build_seconds=time.monotonic() - start)
    for group in ("generated_sources", "external_build_inputs"):
        if any(digest(path) != value for path, value in report[group].items()):
            report["status"] = "inputs_changed"
            record(out / "build.json", report)
            raise RuntimeError("original build inputs changed during isolated rebuild")
    for path, value in report["generated_sources"].items():
        if digest(out / "objects" / Path(path).name) != value:
            raise RuntimeError("copied generated sources changed during build")
    report["status"] = "built" if result.returncode == 0 else "build_failed"
    if result.returncode == 0:
        report["simulator_sha256"] = digest(out / "simulator")
    record(out / "build.json", report)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(f"ISOLATED_HOST_REBUILD_COMPLETE {out / 'build.json'}")


if __name__ == "__main__":
    main()
