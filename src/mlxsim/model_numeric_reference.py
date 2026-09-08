"""Explicit target-numeric reference with independent graph/reduction control.

Atomic transcendental functions deliberately bind the same system libm as the
C++ functional FU model. This verifies lowering/dataflow, NOT independent libm
accuracy or fidelity to the author's unpublished floating-point circuit.
"""

import ctypes
import ctypes.util
import hashlib
from pathlib import Path

import numpy as np
import torch

from mlxsim.model_matrix_reference import MatrixReferenceMode

FLOAT_OPERATORS = {
    "aten.sub.Tensor": "sub",
    "aten.add.Tensor": "add", "aten.mul.Tensor": "mul", "aten.pow.Tensor_Scalar": "pow",
    "aten.rsqrt.default": "rsqrt", "aten.silu.default": "silu", "aten.cos.default": "cos",
    "aten.sin.default": "sin", "aten.neg.default": "neg", "aten.mean.dim": "mean",
    "aten.softmax.int": "softmax",
}


def pairwise_last(values):
    values = np.asarray(values, dtype=np.float32)
    if not values.shape[-1]:
        raise ValueError("explicit numeric reference rejects empty reductions")
    while values.shape[-1] > 1:
        left, right = values[..., ::2], values[..., 1::2]
        if left.shape[-1] != right.shape[-1]:
            right = np.pad(right, [(0, 0)] * (right.ndim - 1) + [(0, 1)])
        values = np.add(left, right, dtype=np.float32)
    return values


class NumericReferenceMode(MatrixReferenceMode):
    def __init__(self):
        super().__init__()
        self.float_calls = {kind: 0 for kind in FLOAT_OPERATORS.values()}
        self.library = ctypes.CDLL(ctypes.util.find_library("m"))
        self.functions = {}
        for name in ("expf", "sqrtf", "cosf", "sinf"):
            function = getattr(self.library, name)
            function.argtypes, function.restype = [ctypes.c_float], ctypes.c_float
            self.functions[name] = function
        paths = {Path(line.split()[-1]).resolve() for line in Path("/proc/self/maps").read_text().splitlines() if "/libm.so" in line}
        if len(paths) != 1:
            raise RuntimeError("cannot uniquely identify the numeric reference libm")
        self.library_path = paths.pop()
        self.library_sha256 = hashlib.sha256(self.library_path.read_bytes()).hexdigest()

    def provenance(self):
        if hashlib.sha256(self.library_path.read_bytes()).hexdigest() != self.library_sha256:
            raise RuntimeError("numeric reference primitive library changed during execution")
        return {"profile": "mlx-vector-fp32-v1", "atomic_primitives_shared_with_cpp_fu": True,
                "independent_matrix_and_reduction_control": True, "libm_path": str(self.library_path),
                "libm_sha256": self.library_sha256, "hardware_transcendental_verified": False}

    def unary(self, name, value):
        value = np.asarray(value, dtype=np.float32)
        function = self.functions[name]
        return np.fromiter((function(float(x)) for x in value.flat), dtype=np.float32, count=value.size).reshape(value.shape)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        kind = FLOAT_OPERATORS.get(str(func))
        if kind is None or not isinstance(args[0], torch.Tensor) or args[0].dtype not in {torch.float16, torch.float32}:
            return super().__torch_dispatch__(func, types, args, kwargs)
        with torch._C._DisableTorchDispatch(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dtype = args[0].dtype
            a = args[0].detach().cpu().numpy().astype(np.float32)
            if kind in {"add", "sub", "mul"}:
                other = args[1]
                if isinstance(other, torch.Tensor):
                    if other.dtype not in {torch.float16, torch.float32}:
                        raise RuntimeError("explicit floating profile does not narrow integer tensor operands")
                    dtype = torch.promote_types(dtype, other.dtype)
                    other = other.detach().cpu().numpy()
                b = np.asarray(other, dtype=np.float32)
                if kind == "add":
                    result = np.add(a, np.multiply(b, np.float32(kwargs.get("alpha", 1)), dtype=np.float32), dtype=np.float32)
                elif kind == "sub":
                    result = np.subtract(a, np.multiply(b, np.float32(kwargs.get("alpha", 1)), dtype=np.float32), dtype=np.float32)
                else:
                    result = np.multiply(a, b, dtype=np.float32)
            elif kind == "pow":
                if args[1] != 2:
                    raise RuntimeError("explicit vector profile only lowers square")
                result = np.multiply(a, a, dtype=np.float32)
            elif kind == "neg":
                result = np.negative(a, dtype=np.float32)
            elif kind in {"sin", "cos"}:
                result = self.unary(kind + "f", a)
            elif kind == "rsqrt":
                result = np.divide(np.float32(1), self.unary("sqrtf", a), dtype=np.float32)
            elif kind == "silu":
                denominator = np.add(np.float32(1), self.unary("expf", np.negative(a, dtype=np.float32)), dtype=np.float32)
                result = np.divide(a, denominator, dtype=np.float32)
            elif kind == "mean":
                if args[0].dtype == torch.float32 and kwargs.get("dtype") == torch.float16:
                    raise RuntimeError("narrowing mean.dtype needs a separately registered precision contract")
                if args[1] not in ([-1], [a.ndim - 1]):
                    raise RuntimeError("explicit mean profile requires the last axis")
                result = np.divide(pairwise_last(a), np.float32(a.shape[-1]), dtype=np.float32)
                if len(args) < 3 or not args[2]:
                    result = result[..., 0]
                dtype = kwargs.get("dtype") or dtype
            else:
                if args[1] not in (-1, a.ndim - 1):
                    raise RuntimeError("explicit softmax profile requires the last axis")
                dtype = args[2] if len(args) > 2 and args[2] is not None else dtype
                if args[0].dtype == torch.float32 and dtype == torch.float16:
                    a = a.astype(np.float16).astype(np.float32)
                maximum = np.max(a, axis=-1, keepdims=True)
                exponents = self.unary("expf", np.subtract(a, maximum, dtype=np.float32))
                result = np.divide(exponents, pairwise_last(exponents), dtype=np.float32)
            if dtype not in {torch.float16, torch.float32}:
                raise RuntimeError("explicit floating reference output dtype is unsupported")
            output = torch.from_numpy(np.asarray(result).astype(np.float16 if dtype == torch.float16 else np.float32)).to(args[0].device)
        self.float_calls[kind] += 1
        return output
