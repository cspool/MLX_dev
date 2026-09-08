"""Independent validation-only reference for the declared matrix numeric mode.

Never imported by the C++ execution path and never supplies model activations.
"""

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def kasc_reference(a, b, *, transposed_b=False, bias=None, output_dtype=np.float32):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if transposed_b:
        b = np.swapaxes(b, -1, -2)
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-2]:
        raise ValueError("invalid reference matrix dimensions")
    out = np.zeros(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1]), dtype=np.float32)
    for ki in range(a.shape[-1]):
        product = np.multiply(a[..., ki, None], b[..., None, ki, :], dtype=np.float32)
        np.add(out, product, out=out)
    if bias is not None:
        np.add(out, np.asarray(bias, dtype=np.float32), out=out)
    return out.astype(output_dtype)


class MatrixReferenceMode(TorchDispatchMode):
    """Explicit numeric-contract reference, never an implicit GPU replacement.

    Only matrix primitive order changes. All other operations retain the
    selected framework/device implementation and are still compared normally.
    """

    def __init__(self):
        super().__init__()
        self.calls = {"linear": 0, "matmul": 0}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in {torch.ops.aten.linear.default, torch.ops.aten.matmul.default}:
            return func(*args, **kwargs)
        if kwargs:
            raise RuntimeError("matrix contract reference requires positional operands")
        a, b = args[:2]
        linear = func == torch.ops.aten.linear.default
        bias = args[2] if linear and len(args) > 2 else None
        if a.dtype not in {torch.float16, torch.float32} or b.dtype != a.dtype:
            raise RuntimeError("matrix contract reference requires homogeneous FP16/FP32 inputs")
        with torch._C._DisableTorchDispatch():
            result = kasc_reference(a.detach().cpu().numpy(), b.detach().cpu().numpy(),
                                    transposed_b=linear,
                                    bias=bias.detach().cpu().numpy() if bias is not None else None,
                                    output_dtype=np.float16 if a.dtype == torch.float16 else np.float32)
            output = torch.from_numpy(result).to(a.device)
        self.calls["linear" if linear else "matmul"] += 1
        return output
