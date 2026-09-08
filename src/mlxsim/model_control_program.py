"""Descriptor-bound RV64I/M/F ALU leaves for runtime/control tensor operations.

Words use real RISC-V encodings; tensor I/O and loops are still a C++ binding,
not a fetched Rocket ELF or a complete RISC-V CPU simulator.
"""

def r(funct7, funct3, rd, a, b):
    return funct7 << 25 | b << 20 | a << 15 | funct3 << 12 | rd << 7 | 0x33


def i(funct3, rd, a, imm):
    if not -2048 <= imm <= 2047:
        raise ValueError("RV64 immediate out of range")
    return (imm & 4095) << 20 | a << 15 | funct3 << 12 | rd << 7 | 0x13


def f(funct7, funct3, rd, a, b=0):
    return funct7 << 25 | b << 20 | a << 15 | funct3 << 12 | rd << 7 | 0x53


def beq(a, b, offset):
    if offset & 1 or not -4096 <= offset <= 4094:
        raise ValueError("RV64 branch offset out of range")
    raw = offset & 8191
    return ((raw >> 12) & 1) << 31 | ((raw >> 5) & 63) << 25 | b << 20 | a << 15 | ((raw >> 1) & 15) << 8 | ((raw >> 11) & 1) << 7 | 0x63


def control_program(kind, input_dtype="i64"):
    extended = kind in {"ge", "bitwise_and", "all", "guard"}
    if kind not in {"arange", "add", "mul", "le", "argmax", "ge", "bitwise_and", "all", "guard"} or input_dtype not in {"i64", "f16", "f32"}:
        raise ValueError("unsupported controller operation/type")
    integer = input_dtype == "i64"
    if kind in {"arange", "add", "mul", "bitwise_and", "all", "guard"} and not integer:
        raise ValueError("controller arithmetic entry is integer only")
    phases = {}
    if kind == "arange":
        phases = {"init": [i(0, 10, 0, 0)], "body": [i(0, 12, 10, 0)], "advance": [i(0, 10, 10, 1)]}
    elif kind in {"add", "mul"}:
        phases["body"] = [r(1 if kind == "mul" else 0, 0, 12, 10, 11)]
    elif kind == "le":
        phases["body"] = [r(0, 2, 12, 11, 10), i(4, 12, 12, 1)] if integer else [f(0x50, 0, 12, 10, 11)]
    elif kind == "ge":
        phases["body"] = [r(0, 2, 12, 10, 11), i(4, 12, 12, 1)] if integer else [f(0x50, 0, 12, 11, 10)]
    elif kind == "bitwise_and":
        phases["body"] = [r(0, 7, 12, 10, 11)]
    elif kind == "all":
        phases = {"init": [i(0, 12, 0, 1)], "body": [r(0, 3, 10, 0, 10), r(0, 7, 12, 12, 10)]}
    elif kind == "guard":
        phases["body"] = [r(0, 3, 12, 0, 10), r(0, 4, 13, 12, 11)]
    else:
        move_best = i(0, 11, 10, 0) if integer else f(0x10, 0, 11, 10, 10)
        phases["init"] = [move_best, i(0, 20, 0, 0), i(0, 21, 0, 0)]
        phases["advance"] = [i(0, 21, 21, 1)]
        phases["compare"] = [r(0, 2, 12, 11, 10)] if integer else [
            f(0x70, 1, 13, 10), i(7, 13, 13, 0x300), r(0, 3, 13, 0, 13),
            f(0x70, 1, 14, 11), i(7, 14, 14, 0x300), r(0, 3, 14, 0, 14),
            i(4, 14, 14, 1), r(0, 7, 13, 13, 14), f(0x50, 1, 12, 11, 10), r(0, 6, 12, 12, 13)]
        phases["select"] = [beq(12, 0, 12), i(0, 20, 21, 0), move_best]
    if sum(map(len, phases.values())) > 32:
        raise ValueError("controller leaf template exceeds 32 words")
    return {"profile": "mlx-controller-rv64-leaf-v2" if extended else "mlx-controller-rv64-leaf-v1", "kind": kind, "input_dtype": input_dtype,
            "xlen": 64, "flen": 64, "gpr_count": 32, "fpr_count": 32, "phases": phases,
            "memory_binding": "descriptor_bound_not_riscv_load_store_execution",
            "rocket_execution_verified": False, "timing_verified": False}
