import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.profile_mlx_chipyard_host import diagnostic_command, flat_profile
from scripts.rebuild_mlx_chipyard_host import prepare

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    out = tmp_path_factory.mktemp("host-profile")
    sampler, binary = out / "sampler.so", out / "probe"
    subprocess.run(["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(ROOT / "system_sim/host_diagnostics/gmon_preload.c"), "-ldl", "-o", str(sampler)], check=True)
    subprocess.run(["gcc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(ROOT / "tests/host_profile_probe.c"), "-o", str(binary)], check=True)
    return out, sampler, binary


def test_sampler_preserves_execution_and_records_real_pcs(probe):
    out, sampler, binary = probe
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "GMON_OUT_PREFIX", "MLX_HOST_PROFILE_META")}
    plain = subprocess.run([str(binary)], capture_output=True, check=True, env=env)
    env.update(LD_PRELOAD=str(sampler), GMON_OUT_PREFIX=str(out / "gmon"), MLX_HOST_PROFILE_META=str(out / "meta.json"))
    profiled = subprocess.run([str(binary)], capture_output=True, check=True, env=env)
    assert plain.stdout == profiled.stdout
    metadata = json.loads((out / "meta.json").read_text())
    assert metadata["low_pc"] < metadata["high_pc"]
    assert not metadata["shared_library_samples_included"] and not metadata["call_arcs_included"]
    files = list(out.glob("gmon.*"))
    assert len(files) == 1
    text = subprocess.run(["gprof", "-b", "-p", str(binary), str(files[0])], capture_output=True, text=True, check=True).stdout
    result = flat_profile(text)
    assert result["executable_sample_seconds"] > 0
    assert "mlx_host_diagnostic_hot_loop" in [row["function"] for row in result["top_functions"]]


@pytest.mark.parametrize("mode", ["missing", "relative", "exists"])
def test_sampler_rejects_ambiguous_or_existing_output(probe, tmp_path, mode):
    _, sampler, binary = probe
    metadata = tmp_path / "meta.json"
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "GMON_OUT_PREFIX", "MLX_HOST_PROFILE_META")}
    env["LD_PRELOAD"] = str(sampler)
    if mode != "missing":
        env.update(GMON_OUT_PREFIX=str(tmp_path / "gmon"), MLX_HOST_PROFILE_META="relative" if mode == "relative" else str(metadata))
    if mode == "exists":
        metadata.write_text("preserve\n")
    result = subprocess.run([str(binary)], capture_output=True, text=True, env=env)
    assert result.returncode == 125 and "MLX_HOST_PROFILE_ERROR" in result.stderr
    if mode == "exists":
        assert metadata.read_text() == "preserve\n"


def test_flat_profile_partition_has_explicit_denominator():
    result = flat_profile(" 60.00 0.60 0.60 VTestHarness::_eval(foo)\n 30.00 0.90 0.30 mlx::tick()\n 10.00 1.00 0.10 helper()\n")
    assert result["groups_sample_seconds"] == {"generated_hardware": .6, "mlx_namespace": .3, "other_executable": .1}
    assert result["executable_sample_seconds"] == pytest.approx(1)
    assert not result["shared_library_time_included"]


@pytest.mark.parametrize("text", ["no time accumulated", " 10.00 1.00 1.00 12 0.01 0.01 foo()\n"])
def test_empty_or_call_arc_profile_rejected(text):
    with pytest.raises(ValueError):
        flat_profile(text)


def test_diagnostic_command_changes_only_owned_binary_and_limit():
    before = ["/frozen/simulator", "+max-cycles=10000000000000", "-s", "73129", "+loadmem=/frozen/test.elf", "/frozen/test.elf"]
    after = diagnostic_command(before, "/owned/simulator", 4000000)
    assert after[2:] == before[2:]
    assert after[:2] == ["/owned/simulator", "+max-cycles=4000000"]
    assert before[0] == "/frozen/simulator" and before[1] == "+max-cycles=10000000000000"


@pytest.mark.parametrize("limit", [0, -1, True, 100000001])
def test_unbounded_diagnostic_rejected(limit):
    with pytest.raises(ValueError):
        diagnostic_command(["sim", "+max-cycles=10"], "owned", limit)


@pytest.mark.parametrize("original", [["sim"], ["sim", "+max-cycles=1", "+max-cycles=2"]])
def test_missing_or_duplicate_cycle_limit_rejected(original):
    with pytest.raises(ValueError):
        diagnostic_command(original, "owned", 10)


def test_rebuild_uses_explicit_owned_target_and_copies_no_objects(tmp_path):
    source = tmp_path / "generated"
    source.mkdir()
    (source / "VTestHarness.mk").write_text(f"default: /frozen/simulator\nVERILATOR_ROOT = {tmp_path / 'verilator'}\n")
    (source / "VTestHarness_classes.mk").write_text("# classes\n")
    (source / "VTestHarness.cpp").write_text("// unchanged generated source\n")
    (source / "VTestHarness.o").write_bytes(b"must not reuse object")
    out = tmp_path / "owned"
    command, report = prepare(source, out, "-O3")
    assert "mlx-host-diagnostic" in command and "OPT_FAST=-O3" in command
    assert not (out / "objects/VTestHarness.o").exists()
    assert (out / "objects/VTestHarness.cpp").read_bytes() == (source / "VTestHarness.cpp").read_bytes()
    assert f"-o {out / 'simulator'}" in (out / "objects/host.mk").read_text()
    assert not report["hardware_or_simulator_source_changed"]
    assert not report["inference_performance_eligible"]
    with pytest.raises(ValueError, match="fresh"):
        prepare(source, out, "-O3")


@pytest.mark.parametrize("flags", ["-Ofast", "-O3 -ffast-math", "-O0"])
def test_unregistered_host_flags_rejected(tmp_path, flags):
    with pytest.raises(ValueError, match="optimizations"):
        prepare(tmp_path / "generated", tmp_path / "new", flags)
