"""Prove host-side control caching preserves simulated matrix behavior."""
import argparse
import copy
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.run_mlx_tensor_semantics import ROOT,sha,source_identity
from mlxsim.model_matrix_program import matrix_program
from mlxsim.model_matrix_reference import kasc_reference


def identity():
    result=source_identity()
    for name in ("scripts/verify_mlx_matrix_control_cache.py","tests/test_matrix_control_cache.py","tests/test_matrix_window_scheduler.py","tests/test_matrix_external_memory.py","tests/matrix_external_memory.cc","src/mlxsim/model_matrix_reference.py"):
        result[name]=sha(ROOT/name)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise RuntimeError("choose a fresh matrix control-cache evidence directory")
    out.mkdir(parents=True);before=identity()
    with (out/"pytest.log").open("w") as log:
        result=subprocess.run([str(ROOT/".venv/bin/python"),"-m","pytest","-q","tests/test_matrix_control_cache.py","tests/test_matrix_window_scheduler.py","tests/test_matrix_external_memory.py",f"--basetemp={out/'pytest'}",f"--junitxml={out/'regression.xml'}"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode:raise RuntimeError("matrix control-cache conformance tests failed")
    a=np.arange(8*65,dtype=np.float32).reshape(8,65)/1024;b=np.arange(129*65,dtype=np.float32).reshape(129,65)/32768
    expected=kasc_reference(a.astype(np.float16),b.astype(np.float16),transposed_b=True,output_dtype=np.float16)
    literal=lambda v:{"kind":"literal","dtype":"f16","shape":list(v.shape),"values":v.astype(np.float16).tolist()}
    job={"schema":"mlx_matrix_window_job_v1","m":8,"n":129,"k":65,"a":literal(a),"b":literal(b),"program":matrix_program("f16","f16"),"options":{"rows":4,"columns":4,"trace":False}}
    binary=ROOT/"build/mlx-matrix-window/mlx-matrix-window";timings={"cached":[],"scan":[]};reports={};attempts=[]
    # Alternate ordering; host timings are descriptive, never pass/fail gates
    # or an estimate of full-model/MLX inference performance.
    for index,mode in enumerate(("cached","scan","scan","cached","cached","scan")):
        directory=out/f"host-{index}-{mode}";directory.mkdir();spec=copy.deepcopy(job);spec["options"]["cache_control"]=mode=="cached"
        source=directory/"job.json";source.write_text(json.dumps(spec));started=time.perf_counter()
        process=subprocess.run([str(binary),str(source),str(directory/"out")],capture_output=True,text=True,timeout=120)
        seconds=time.perf_counter()-started
        if process.returncode:raise RuntimeError(process.stdout+process.stderr)
        report=json.loads((directory/"out/result.json").read_text());data=np.fromfile(directory/"out/output.bin",dtype=np.float16).reshape(8,129)
        if not np.array_equal(data.view(np.uint16),expected.view(np.uint16)):raise RuntimeError("control-cache output differs from reference")
        timings[mode].append(seconds);reports[mode]=report
        attempts.append({"mode":mode,"host_seconds":seconds,"job_sha256":sha(source),"report_sha256":sha(directory/"out/result.json"),"output_sha256":sha(directory/"out/output.bin")})
    strip=lambda r:{k:v for k,v in r.items() if k!="host_optimization"}
    if strip(reports["cached"])!=strip(reports["scan"]):raise RuntimeError("host optimization changed target cycles/counters")
    if before!=identity():raise RuntimeError("control-cache source changed during validation")
    report={"classification":"host_simulator_optimization_not_inference_performance","sources":before,"regression_xml_sha256":sha(out/"regression.xml"),
        "target_cycles_unchanged":reports["cached"]["cycles"],"host_optimization":{mode:r["host_optimization"] for mode,r in reports.items()},
        "host_seconds_median":{mode:statistics.median(v) for mode,v in timings.items()},"attempts":attempts,"binary_sha256":sha(binary),
        "mlx_system_verified":False,"inference_performance_eligible":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(f"MATRIX_CONTROL_CACHE_EQUIVALENCE_PASS {out/'report.json'}")


if __name__=="__main__":main()
