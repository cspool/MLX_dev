"""Replay QA output/ownership components with C/C++ memory sanitizers."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from mlxsim.model_result_contract import verify_qa_result
from mlxsim.model_physical_evidence import verify_physical_execution
from mlxsim.model_ready_evidence import verify_ready_execution
from scripts.run_mlx_qa_model import sources, ROOT, require
from scripts.run_mlx_tensor_semantics import sha
from scripts.mlx_system_attempt import record, linked_libraries


def normalized(result):
    result = copy.deepcopy(result)
    for out in result["outputs"]:
        for row in out["outputs"].values(): row["file"] = "$OUTPUT/" + Path(row["file"]).name
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("tests", "build", "output"): parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); tests = args.tests.resolve()
    require(not out.exists(), "choose a fresh QA safety directory")
    suite = ET.parse(tests / "regression.xml").getroot().find("testsuite")
    require(suite is not None and all(suite.get(k) == "0" for k in ("failures", "errors", "skipped")), "QA regression suite incomplete")
    before = sources()
    for file in (Path(__file__), ROOT / "tests/test_qa_result_contract.py", ROOT / "tests/test_native_performance_comparison.py"):
        before[str(file.resolve().relative_to(ROOT))] = sha(file)
    paths = [p for p in (tests / "pytest").rglob("program.json") if p.relative_to(tests / "pytest").parts[0].startswith("test_qa_")
             and p.parent.name in {"physical", "paired", "perturb", "bad"} and not any(a.is_symlink() for a in p.parents)]
    require(len(paths) == 19, "QA replay set differs")
    binaries = {mode: args.build.resolve() / name for mode, name in (("physical", "mlx-physical-model"), ("paired", "mlx-ready-graph"))}
    files = {str(tests / "regression.xml"): sha(tests / "regression.xml")}
    for b in binaries.values(): files[str(b)] = sha(b); files.update(linked_libraries(b))
    for p in paths:
        files[str(p)] = sha(p); files[str(p.parent / "options.json")] = sha(p.parent / "options.json")
        for f in (p.parent / "out").glob("*"):
            if f.is_file(): files[str(f)] = sha(f)
    out.mkdir(parents=True); replays = []
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    for i, p in enumerate(sorted(paths)):
        mode = "paired" if p.parent.name in {"paired", "bad"} else "physical"
        destination = out / f"case-{i:02d}"; log = out / f"case-{i:02d}.log"
        expected = 0 if (p.parent / "out/result.json").exists() else 1
        with log.open("w") as stream:
            r = subprocess.run([str(binaries[mode]), str(p), str(p.parent / "options.json"), str(destination)], stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=180)
        require(r.returncode == expected and not any(s in log.read_text() for s in ("ERROR: AddressSanitizer", "runtime error:", "LeakSanitizer", "DEADLYSIGNAL")), "QA sanitizer replay failed")
        if expected:
            require(not (destination / "result.json").exists(), "rejected QA produced success")
        else:
            program = json.loads(p.read_text()); options = json.loads((p.parent / "options.json").read_text())
            old = json.loads((p.parent / "out/result.json").read_text()); actual = json.loads((destination / "result.json").read_text())
            require(normalized(old) == normalized(actual), "QA sanitizer changed values/events/cycles/resource reports")
            for spec, a, b in zip(program["outputs"], old["outputs"], actual["outputs"], strict=True):
                verify_qa_result(program, spec, b)
                for role in ("start_logits", "end_logits"):
                    require(Path(a["outputs"][role]["file"]).read_bytes() == Path(b["outputs"][role]["file"]).read_bytes(), "QA sanitizer changed actual logits")
            if mode == "physical": verify_physical_execution(program, actual, options)
            else: verify_ready_execution(program, actual, options)
        replays.append(dict(program=str(p), mode=mode, output=str(destination), expected_exit=expected, log_sha256=sha(log)))
    require(sum(r["expected_exit"] == 0 for r in replays) == 7, "QA successful replay coverage differs")
    require(all(sha(ROOT / p) == h for p, h in before.items()) and all(sha(Path(p)) == h for p, h in files.items()), "QA replay provenance changed")
    record(out / "report.json", dict(classification="qa_runtime_component_sanitizer_validation_not_full_model", sources=before, files=files,
                                     regression_tests=int(suite.get("tests")), replays=replays, full_model_execution_verified=False, mlx_system_verified=False))
    print(f"QA_RUNTIME_SAFETY_PASS replays={len(replays)} {out}")


if __name__ == "__main__": main()
