"""Bounded floating-point vector/normalization microprograms (functional stage)."""

FLOAT_KINDS = {"add", "sub", "mul", "pow", "rsqrt", "silu", "cos", "sin", "neg", "mean", "softmax"}
OP = {"load_a16": 2, "load_b16": 3, "cvt_up": 4, "mul": 5, "add": 6,
      "cvt_down": 7, "store16": 8, "load_a32": 9, "load_b32": 10, "store32": 11,
      "constant": 14, "neg": 15, "sub": 16, "exp": 17, "div": 18, "sqrt": 19,
      "cos": 20, "sin": 21, "shuffle": 22, "max": 23, "move_scalar": 24,
      "broadcast": 25, "load_stack": 26, "store_stack": 27}


def vector_program(kind, input_dtypes, output_dtype, *, width=None, alpha=1):
    if kind not in FLOAT_KINDS or output_dtype not in {"f16", "f32"} or any(d not in {"f16", "f32"} for d in input_dtypes):
        raise ValueError("unsupported floating-point vector program contract")
    if len(input_dtypes) != (2 if kind in {"add", "sub", "mul"} else 1):
        raise ValueError("vector program operand count mismatch")
    if kind in {"mean", "softmax"} and (type(width) is not int or not 1 <= width <= 2**31):
        raise ValueError("reduction width must be in [1, 2**31]")
    rom, phases, constants = [], {}, []

    def emit(name, dst=0, a=0, b=0, imm=0):
        word = OP[name] | dst << 8 | a << 12 | b << 16 | imm << 20
        if word not in rom:
            rom.append(word)
        return rom.index(word)

    def constant(value, dst=1):
        if value not in constants:
            constants.append(value)
        return emit("constant", dst=dst, imm=constants.index(value))

    def load(operand=0, dst=0, negative_padding=False):
        precision = input_dtypes[operand]
        result = [emit(f"load_{'a' if operand == 0 else 'b'}{'16' if precision == 'f16' else '32'}", dst=dst, imm=int(negative_padding))]
        if precision == "f16":
            result.append(emit("cvt_up", dst=dst, a=dst))
        return result

    def trans(name, dst=0, a=0, b=0):
        return [emit(name, dst=dst, a=a, b=b, imm=group) for group in range(4)]

    def store():
        return ([emit("cvt_down", a=0)] if output_dtype == "f16" else []) + [emit("store16" if output_dtype == "f16" else "store32", a=0)]

    if kind not in {"mean", "softmax"}:
        body = load()
        if kind in {"add", "sub", "mul"}:
            body += load(1, 1)
            if kind in {"add", "sub"} and alpha != 1:
                body += [constant(alpha, 7), emit("mul", dst=1, a=1, b=7)]
            body.append(emit(kind, a=0, b=1))
        elif kind == "pow":
            body.append(emit("mul", a=0, b=0))
        elif kind == "neg":
            body.append(emit("neg", a=0))
        elif kind in {"cos", "sin"}:
            body += trans(kind)
        elif kind == "rsqrt":
            body += trans("sqrt") + [constant(1.0)] + trans("div", a=1, b=0)
        elif kind == "silu":
            body += [emit("neg", dst=1, a=0)] + trans("exp", dst=1, a=1)
            body += [constant(1.0, 7), emit("add", dst=1, a=1, b=7)] + trans("div", a=0, b=1)
        phases["body"] = body + store()
    else:
        lanes = min(16, 1 << (width - 1).bit_length())

        def tree(op):
            result = []
            for distance in (1, 2, 4, 8):
                if distance < lanes:
                    result += [emit("shuffle", dst=1, a=0, imm=distance), emit(op, a=0, b=1)]
            return result

        phases["to_carry"] = [emit("move_scalar", dst=4, a=0)]
        phases["save_carry"] = [emit("store_stack", a=4)]
        phases["root"] = [emit("load_stack", dst=0)]
        phases["merge_sum"] = [emit("load_stack", dst=1), emit("add", dst=4, a=1, b=4)]
        if kind == "mean":
            phases["sum_tile"] = load() + tree("add")
            phases["final"] = [constant(width)] + trans("div", a=0, b=1) + store()
        else:
            phases["max_tile"] = load(negative_padding=True) + tree("max")
            phases["merge_max"] = [emit("load_stack", dst=1), emit("max", dst=4, a=1, b=4)]
            phases["save_max"] = [emit("move_scalar", dst=5, a=0)]
            exp_tile = load(negative_padding=True) + [emit("broadcast", dst=1, a=5), emit("sub", a=0, b=1)] + trans("exp")
            phases["sum_tile"] = exp_tile + tree("add")
            phases["save_sum"] = [emit("move_scalar", dst=6, a=0)]
            phases["output_tile"] = exp_tile + [emit("broadcast", dst=1, a=6)] + trans("div", a=0, b=1) + store()
    if len(rom) > 32:
        raise ValueError(f"{kind} vector program exceeds 32-word ROM: {len(rom)}")
    return {"profile": "mlx-vector-fp32-v1", "kind": kind, "input_dtypes": input_dtypes,
            "output_dtype": output_dtype, "width": width, "lanes": 16, "trans_lanes": 4,
            "rf_vectors": 16, "rf_vector_bytes": 64, "rf_vectors_used": 8,
            "spm_bytes": 8192, "spm_bytes_used": 320, "rom_words": 32,
            "rom": rom, "phases": phases, "constants": constants,
            "reduction": "adjacent_pairwise_zero_padding_binary_carry" if width else None,
            "timing_verified": False, "system_verified": False}
