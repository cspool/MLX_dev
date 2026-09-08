import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_real_cpu_drives_matrix_and_uses_computed_tokens(tmp_path):
    out=tmp_path/"chain"
    result=subprocess.run([str(ROOT/".venv/bin/python"),"-m","scripts.verify_mlx_spike_matrix_chain","--output",str(out)],cwd=ROOT,capture_output=True,text=True,timeout=300)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads((out/"report.json").read_text());cases={c["case"]:c for c in report["cases"]}
    assert len(cases)==29 and cases["normal"]["tokens"]==[1,2,3] and cases["perturbed"]["tokens"]==[2,3,0]
    assert cases["normal-f16"]["tokens"]==[1,2,3] and cases["perturbed-f16"]["tokens"]==[2,3,0]
    for name in ("reject-magic","reject-reserved","reject-word"):
        assert cases[name]["successful_windows"]==cases[name]["device_reads"]==cases[name]["device_writes"]==0
    assert cases["reject-uninitialized"]["successful_windows"]==cases["reject-uninitialized"]["device_writes"]==0
    assert cases["recover"]["launches"]==4 and cases["recover"]["successful_windows"]==3
    for kind in ("vector","softmax"):
        for precision in ("f16","f32"):
            for suffix,tokens in (("",[1,2,3]),("-perturbed",[2,3,0])):
                case=cases[f"{kind}-{precision}{suffix}"]
                assert case["tokens"]==tokens and case["successful_windows"]==6
                if kind=="softmax":assert case["reference_primitives"]["atomic_primitives_shared_with_cpp_fu"]
    for precision in ("f16","f32"):
        for suffix,tokens in (("",[1,2,3]),("-perturbed",[2,3,0])):
            case=cases[f"memory-{precision}{suffix}"]
            assert case["tokens"]==tokens and case["memory_stage"] and case["successful_windows"]==9
        for kind in ("where","cat"):
            for suffix in ("","-perturbed"):
                case=cases[f"{kind}-{precision}{suffix}"]
                expected=([2,0,2] if suffix else [1,2,0]) if kind=="where" else ([2,3,0] if suffix else [1,2,3])
                assert case["tokens"]==expected and case["memory_stage"]==kind and case["successful_windows"]==9
    assert report["host_model_data_flow_executed"] and report["poll_driven_accelerator_clock"]
    assert not report["full_model_verified"] and not report["rocket_execution_verified"] and not report["inference_performance_eligible"]
