"""Explicit, bounded matrix microcode for the reconstructed full-precision mode.

This does not reuse the old FP16 PE opcode space or certify timing/system DMA.
"""

PROFILE = "mlx-matrix-f32-kasc-v1"
OP = {"zero32": 1, "load_a16": 2, "load_b16": 3, "cvt16_32": 4,
      "mul32": 5, "add32": 6, "cvt32_16": 7, "store16": 8,
      "load_a32": 9, "load_b32": 10, "store32": 11,
      "load_bias16": 12, "load_bias32": 13}


def word(name, dst=0, a=0, b=0, row=2):
    if any(type(x) is not int or not 0 <= x < 16 for x in (dst, a, b)) or row not in (0, 1, 2):
        raise ValueError("invalid matrix instruction field")
    return OP[name] | dst << 8 | a << 12 | b << 16 | row << 20


def matrix_program(input_dtype, output_dtype, bias=False):
    if input_dtype not in {"f16", "f32"} or output_dtype not in {"f16", "f32"}:
        raise ValueError("matrix microcode requires FP16/FP32 tensors")
    half = input_dtype == "f16"
    prologue = [word("zero32", dst=4 + row, row=row) for row in range(2)]
    body = [word("load_b16" if half else "load_b32", dst=1)]
    if half:
        body.append(word("cvt16_32", dst=1, a=1))
    for row in range(2):
        body.append(word("load_a16" if half else "load_a32", dst=0, row=row))
        if half:
            body.append(word("cvt16_32", dst=0, a=0, row=row))
        body += [word("mul32", dst=2, a=0, b=1, row=row), word("add32", dst=4 + row, a=4 + row, b=2, row=row)]
    epilogue = []
    if bias:
        epilogue.append(word("load_bias16" if half else "load_bias32", dst=3))
        if half:
            epilogue.append(word("cvt16_32", dst=3, a=3))
    for row in range(2):
        if bias:
            epilogue.append(word("add32", dst=4 + row, a=4 + row, b=3, row=row))
        if output_dtype == "f16":
            epilogue.append(word("cvt32_16", dst=4 + row, a=4 + row, row=row))
        epilogue.append(word("store16" if output_dtype == "f16" else "store32", a=4 + row, row=row))
    words = prologue + body + epilogue
    if len(words) > 32:
        raise ValueError("matrix template exceeds physical instruction capacity")
    return {
        "profile": PROFILE, "input_dtype": input_dtype, "output_dtype": output_dtype,
        "has_bias": bool(bias), "tile_m": 2, "tile_n": 16, "tile_k": 64,
        "rf_vectors": 16, "rf_vector_bytes": 64, "spm_bytes": 8192, "rom_words": 32,
        "rf_vectors_used": 6, "spm_bytes_used": (64 * 16 + 2 * 64 + 16) * (2 if half else 4),
        "prologue": prologue, "body": body, "epilogue": epilogue,
        "numeric_contract": "K ascending; MUL.F32.RNE then ADD.F32.RNE; no contraction; final output cast RNE",
        "target_status": "reconstructed_functional_microcode_timing_and_system_pending",
    }
