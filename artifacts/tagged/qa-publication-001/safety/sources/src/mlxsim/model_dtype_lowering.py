"""Explicit dtype-sensitive lowering passes, separate from numeric oracle code."""

from mlxsim.model_vector_program import OP


def lower_softmax_input_cast(node):
    """Honor softmax(dtype=FP16) BEFORE max/exp/reduction, not just at output.

    Operates on compiler-owned microcode. The cast is repeated in RF for each
    streaming pass, avoiding an unbounded temporary tensor in SPM.
    """
    if node["kind"] != "softmax" or "vector_program" not in node:
        return False
    program = node["vector_program"]
    if program["input_dtypes"] != ["f32"] or program["output_dtype"] != "f16":
        return False
    if len(node["args"]) < 3 or node["args"][2] != "torch.float16":
        raise ValueError("narrowing softmax requires an explicit source dtype contract")
    if program.get("softmax_input_cast") == "f32_to_f16_before_reduction":
        return False
    # CVT_DOWN r0,r0 and CVT_UP r0,r0 have no other fields set.
    casts = []
    for word in (OP["cvt_down"], OP["cvt_up"]):
        if word not in program["rom"]:
            program["rom"].append(word)
        casts.append(program["rom"].index(word))
    if len(program["rom"]) > program["rom_words"]:
        raise ValueError("dtype-correct softmax exceeds the available ROM")
    for phase in ("max_tile", "sum_tile", "output_tile"):
        rewritten = []
        loads = 0
        for index in program["phases"][phase]:
            word = program["rom"][index]
            rewritten.append(index)
            if word & 255 == OP["load_a32"]:
                if (word >> 8) & 15:
                    raise ValueError("softmax input must load r0 for this cast lowering")
                rewritten.extend(casts)
                loads += 1
        if loads != 1:
            raise ValueError("dtype-correct softmax requires one input load per pass")
        program["phases"][phase] = rewritten
    program["softmax_input_cast"] = "f32_to_f16_before_reduction"
    return True
