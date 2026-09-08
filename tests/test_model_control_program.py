import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mlxsim.model_control_program import control_program, i, r, f, beq
from test_model_tensor_semantics import literal, node, ref

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/mlx-control-leaf"
MASK = 2**64 - 1


@pytest.fixture(scope="module")
def control_binary():
    for command in (["cmake", "-S", str(ROOT / "simulator_ext/control_model"), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release"], ["cmake", "--build", str(BUILD), "--target", "mlx-control-leaf", "-j4"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
    return BUILD / "mlx-control-leaf"


def execute(binary, directory, job, *, error=None):
    directory.mkdir()
    (directory / "job.json").write_text(json.dumps(job))
    result = subprocess.run([str(binary), str(directory / "job.json"), str(directory / "result.json")], capture_output=True, text=True, timeout=30)
    if error:
        assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((directory / "result.json").read_text())
    assert not report["rocket_execution_verified"]
    return report


def probe(binary, directory, words, x=None, floats=None, *, error=None):
    return execute(binary, directory, {"schema": "rv64_leaf_probe_v1", "words": words, "x": {str(reg): value & MASK for reg, value in (x or {}).items()}, "f": {str(reg): value for reg, value in (floats or {}).items()}}, error=error)


def test_instruction_encodings_match_the_real_riscv_assembler(tmp_path):
    elf, raw = tmp_path / "words.elf", tmp_path / "words.bin"
    subprocess.run(["riscv64-unknown-elf-gcc", "-march=rv64imafd", "-mabi=lp64d", "-nostdlib", "-Wl,--no-relax,-Ttext=0x10000,-e,controller_words", str(ROOT / "tests/fixtures/rv64_control_words.S"), "-o", str(elf)], check=True, capture_output=True)
    subprocess.run(["riscv64-unknown-elf-objcopy", "-O", "binary", "--only-section=.text", str(elf), str(raw)], check=True, capture_output=True)
    expected = [i(0,12,10,0), i(0,10,10,1), r(0,0,12,10,11), r(1,0,12,10,11), r(0,2,12,11,10), i(4,12,12,1), f(0x50,0,12,10,11),
                f(0x70,1,13,10), i(7,13,13,768), r(0,3,13,0,13), f(0x70,1,14,11), i(7,14,14,768), r(0,3,14,0,14),
                i(4,14,14,1), r(0,7,13,13,14), f(0x50,1,12,11,10), r(0,6,12,12,13), beq(12,0,12), i(0,20,21,0), f(0x10,0,11,10,10), f(0x68,0,10,10,2)]
    assert list(struct.unpack("<" + "I" * len(expected), raw.read_bytes())) == expected


def test_rv64_wrap_signed_comparison_and_x0(control_binary, tmp_path):
    report = probe(control_binary, tmp_path / "alu", [r(0,0,12,10,11), r(0,2,13,10,11), r(0,3,14,10,11), i(0,0,0,12)], {0: 99, 10: -1, 11: 1})
    assert report["x"][12] == 0 and report["x"][13] == 1 and report["x"][14] == 0 and report["x"][0] == 0
    report = probe(control_binary, tmp_path / "multiply", [r(1,0,12,10,11)], {10: 2**63 + 1, 11: 3})
    assert report["x"][12] == ((2**63 + 1) * 3) & MASK


@pytest.mark.parametrize("bits,classification", [(0xFF800000,1), (0xBF800000,2), (0x80000001,4), (0x80000000,8), (0,16), (1,32), (0x3F800000,64), (0x7F800000,128), (0x7F800001,256), (0x7FC00000,512)])
def test_fclass_and_nan_boxing(control_binary, tmp_path, bits, classification):
    result = probe(control_binary, tmp_path / "class", [f(0x70,1,12,10)], floats={10: 0xFFFFFFFF00000000 | bits})
    assert result["x"][12] == classification and result["fflags"] == 0
    result = probe(control_binary, tmp_path / "unboxed", [f(0x70,1,12,10)], floats={10: bits})
    assert result["x"][12] == 512


def test_float_compare_flags_and_fcvt_signed_long(control_binary, tmp_path):
    result = probe(control_binary, tmp_path / "compare", [f(0x50,0,12,10,11)], floats={10: 0xFFFFFFFF7FC00000, 11: 0xFFFFFFFF3F800000})
    assert result["x"][12] == 0 and result["fflags"] == 16
    result = probe(control_binary, tmp_path / "convert", [f(0x68,0,10,10,2)], {10: 2**60 + 1})
    expected = struct.unpack("<I", struct.pack("<f", float(2**60 + 1)))[0]
    assert result["f"][10] == 0xFFFFFFFF00000000 | expected and result["fflags"] == 1


@pytest.mark.parametrize("kind", ["arange", "add", "mul", "le", "argmax"])
def test_tensor_controller_routes_use_rv64_results(control_binary, tmp_path, kind):
    a = torch.tensor([[2**60 + 1, 2**60 + 3, -(2**60)]])
    b = torch.tensor([[2, -1, 3]])
    if kind == "arange":
        args, expected = [9], torch.arange(9)
    elif kind == "argmax":
        args, expected = [ref("a"), -1], a.argmax(-1)
    else:
        args = [ref("a"), ref("b")]
        expected = {"add": lambda: a+b, "mul": lambda: a*b, "le": lambda: a<=b}[kind]()
    item = node(0, kind, args, expected)
    item["control_program"] = control_program(kind)
    report = execute(control_binary, tmp_path / "tensor", {"schema": "mlx_control_job_v1", "assets": {"a": literal(a), "b": literal(b)}, "node": item})
    assert report["values"] == expected.flatten().tolist()
    assert report["instructions"] > 0


def test_argmax_float_ties_and_first_nan(control_binary, tmp_path):
    raw = np.array([[0x3F800000,0x40000000,0x40000000], [0x7FC00123,0x7FC00000,0x40400000]], dtype=np.uint32)
    data = tmp_path / "values.bin"; data.write_bytes(raw.tobytes())
    assets = {"x": {"kind": "mapped_file", "dtype": "f32", "shape": [2,3], "path": str(data), "byte_offset": 0, "bytes": raw.nbytes}}
    item = node(0, "argmax", [ref("x"), -1], torch.tensor([1,0]))
    item["control_program"] = control_program("argmax", "f32")
    report = execute(control_binary, tmp_path / "float", {"schema": "mlx_control_job_v1", "assets": assets, "node": item})
    assert report["values"] == [1,0] and report["fflags_observed"] == 16


def test_decoder_rejects_unsupported_words_and_runaway_branch(control_binary, tmp_path):
    probe(control_binary, tmp_path / "unknown", [0xFFFFFFFF], error="unsupported RV64")
    probe(control_binary, tmp_path / "loop", [beq(0,0,0)], error="step limit")
    probe(control_binary, tmp_path / "misaligned", [beq(0,0,2)], error="bounded leaf")
