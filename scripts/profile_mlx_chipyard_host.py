"""Replay a pinned Chipyard input in a new, explicitly diagnostic process."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from scripts.mlx_system_attempt import digest, linked_libraries, record, run_process


def flat_profile(text):
    rows = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(.+)", line)
        if match:
            name = match[4].strip()
            if re.match(r"\d", name):
                raise ValueError("call-arc profiles are outside this PC-only diagnostic")
            rows.append({"function": name, "sampled_self_seconds": float(match[3])})
    total = sum(row["sampled_self_seconds"] for row in rows)
    if total <= 0:
        raise ValueError("no executable PC samples recorded")
    groups = {"generated_hardware": 0.0, "mlx_namespace": 0.0, "other_executable": 0.0}
    for row in rows:
        group = ("generated_hardware" if row["function"].startswith("VTestHarness::") else
                 "mlx_namespace" if row["function"].startswith("mlx::") else "other_executable")
        groups[group] += row["sampled_self_seconds"]
    return {"executable_sample_seconds": total, "groups_sample_seconds": groups,
            "top_functions": rows[:30], "shared_library_time_included": False,
            "denominator": "sampled_executable_PC_time_not_process_wall_or_target_time"}


def diagnostic_command(original, owned, maximum):
    if type(maximum) is not int or not 0 < maximum <= 100_000_000:
        raise ValueError("diagnostic cycle limit must be positive and bounded")
    command = list(original)
    positions = [i for i, value in enumerate(command) if value.startswith("+max-cycles=")]
    if len(positions) != 1:
        raise ValueError("expected exactly one explicit original cycle limit")
    command[0] = str(owned)
    command[positions[0]] = f"+max-cycles={maximum}"
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=1_000_000)
    parser.add_argument("--host-build", type=Path)
    parser.add_argument("--sampler", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError("choose a fresh diagnostic output directory")
    original = json.loads(args.execution.read_text())
    frozen_binary = Path(original["command"][0])
    if digest(frozen_binary) != original["simulator_sha256"]:
        raise ValueError("original executed binary identity changed")
    binary = frozen_binary
    host_build = None
    if args.host_build:
        host_build = json.loads((args.host_build / "build.json").read_text())
        binary = args.host_build.resolve() / "simulator"
        if host_build["status"] != "built" or digest(binary) != host_build["simulator_sha256"]:
            raise ValueError("host rebuild is incomplete or changed")
        for group in ("generated_sources", "external_build_inputs"):
            if any(digest(path) != value for path, value in host_build[group].items()):
                raise ValueError("host rebuild inputs changed")
    libraries = linked_libraries(binary)
    if libraries != original["runtime_libraries"]:
        raise ValueError("runtime library identities differ from the pinned execution")
    profile = args.execution.resolve().parent.parent / "profile.json"
    pinned = {**original["inputs"], str(frozen_binary): original["simulator_sha256"],
              str(args.execution.resolve().parent / "test.elf"): original["elf_sha256"],
              str(profile): original["profile_identity"]["effective_sha256"], **libraries}
    if any(digest(path) != value for path, value in pinned.items()):
        raise ValueError("pinned input identity changed")
    out.mkdir(parents=True)
    owned = out / "simulator"
    shutil.copy2(binary, owned)
    command = diagnostic_command(original["command"], owned, args.max_cycles)
    record(out / "original-execution-snapshot.json", original)
    if host_build:
        record(out / "host-build-snapshot.json", host_build)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MLX_") and key not in ("LD_PRELOAD", "GMON_OUT_PREFIX")}
    env.update(MLX_CLOCKED_PROFILE=str(profile), MLX_CLOCKED_REPORT=str(out / "device.json"),
               MLX_WIDE_MEMORY_REPORT=str(out / "memory.json"),
               MLX_CLOCKED_PROGRESS=str(out / "progress.json"),
               MLX_CLOCKED_PROGRESS_PERIOD="100000",
               MLX_CLOCKED_LAUNCH_MAP=str(args.execution.resolve().parent / "launch-map.json"))
    sampler_hash = None
    if args.sampler:
        sampler = out / "sampler.so"
        shutil.copy2(args.sampler, sampler)
        sampler_hash = digest(sampler)
        env.update(LD_PRELOAD=str(sampler), MLX_HOST_PROFILE_META=str(out / "sampler.json"),
                   GMON_OUT_PREFIX=str(out / "gmon"))
    record(out / "protocol.json", {"classification": "host_diagnostic_not_model_acceptance",
           "original_binary_sha256": original["simulator_sha256"], "binary_sha256": digest(owned),
           "sampler_sha256": sampler_hash, "inputs": pinned, "command": command,
           "full_model_execution_verified": False, "inference_performance_eligible": False})
    try:
        state = run_process(command, out / "run.log", out / "execution.json", cwd=out,
                            timeout=args.timeout, env=env,
                            metadata={"classification": "owned_host_diagnostic_not_acceptance"})
    except subprocess.CalledProcessError:
        state = json.loads((out / "execution.json").read_text())
    if any(digest(path) != value for path, value in pinned.items()):
        raise ValueError("diagnostic changed original input identities")
    if digest(owned) != digest(binary) or (sampler_hash and digest(out / "sampler.so") != sampler_hash):
        raise ValueError("owned diagnostic binaries changed")
    result = {"classification": "host_replay_diagnostic_not_model_acceptance",
              "exit_code": state["exit_code"], "host_elapsed_seconds": state["host_elapsed_seconds"],
              "source_inputs_unchanged": True, "full_model_execution_verified": False,
              "inference_performance_eligible": False, "original_case_reports_equal": {}}
    for name in ("device.json", "memory.json"):
        path = out / name
        reference = args.execution.resolve().parent / name
        if path.exists():
            report = json.loads(path.read_text())
            if name == "device.json":
                report.pop("progress_observer", None)
                result["device_cycles"] = report["device"]["cycle"]
            if reference.exists():
                expected = json.loads(reference.read_text())
                expected.pop("progress_observer", None)
                result["original_case_reports_equal"][name] = expected == report
    if sampler_hash:
        gmon = list(out.glob("gmon.*"))
        if len(gmon) != 1:
            raise ValueError("expected one completed process PC histogram")
        with (out / "gprof.txt").open("w") as log:
            subprocess.run(["gprof", "-b", "-p", str(owned), str(gmon[0])], stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=60)
        result["pc_profile"] = flat_profile((out / "gprof.txt").read_text())
    record(out / "report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
