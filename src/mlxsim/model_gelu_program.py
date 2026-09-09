"""Fixed erf-form GELU recipe, evaluated by existing C++ FP32 primitives.

Coefficient source: Abramowitz/Stegun, p.299, equation 7.1.26.
The printed real-arithmetic erf bound is NOT a bound for this FP32 sequence.
"""
import math

GELU = "aten.gelu.default"
GELU_PROFILE = "mlx-gelu-erf-as7126-fp32-v1"
COEFFICIENTS = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
P = 0.3275911
CLIP = 8.0
INV_SQRT2 = 0.7071067811865476
TINY = 2.0**-126

# name, target kind, operator, operand names/literals. x is the original input
# promoted to FP32; clipping affects only erf evaluation, not the outer x/2.
STAGES = [
    ("negative_input","neg","aten.neg.default",("x",)),
    ("clip_negative","maximum","mlx.primitive.maximum",("negative_input",-CLIP)),
    ("clip_upper","neg","aten.neg.default",("clip_negative",)),
    ("clip_lower","maximum","mlx.primitive.maximum",("clip_upper",-CLIP)),
    ("scaled","mul","aten.mul.Tensor",("clip_lower",INV_SQRT2)),
    ("negative_scaled","neg","aten.neg.default",("scaled",)),
    ("magnitude","maximum","mlx.primitive.maximum",("scaled","negative_scaled")),
    ("sign_denominator","maximum","mlx.primitive.maximum",("magnitude",TINY)),
    ("sign","div","mlx.primitive.div",("scaled","sign_denominator")),
    ("t_scale","mul","aten.mul.Tensor",("magnitude",P)),
    ("t_denominator","add","aten.add.Tensor",("t_scale",1.0)),
    ("t","div","mlx.primitive.div",(1.0,"t_denominator")),
    ("poly_a5","mul","aten.mul.Tensor",("t",COEFFICIENTS[4])),
    ("poly_add4","add","aten.add.Tensor",("poly_a5",COEFFICIENTS[3])),
    ("poly_mul3","mul","aten.mul.Tensor",("poly_add4","t")),
    ("poly_add3","add","aten.add.Tensor",("poly_mul3",COEFFICIENTS[2])),
    ("poly_mul2","mul","aten.mul.Tensor",("poly_add3","t")),
    ("poly_add2","add","aten.add.Tensor",("poly_mul2",COEFFICIENTS[1])),
    ("poly_mul1","mul","aten.mul.Tensor",("poly_add2","t")),
    ("poly_add1","add","aten.add.Tensor",("poly_mul1",COEFFICIENTS[0])),
    ("polynomial","mul","aten.mul.Tensor",("poly_add1","t")),
    ("square","mul","aten.mul.Tensor",("scaled","scaled")),
    ("negative_square","neg","aten.neg.default",("square",)),
    ("exponential","exp","aten.exp.default",("negative_square",)),
    ("correction","mul","aten.mul.Tensor",("polynomial","exponential")),
    ("negative_correction","neg","aten.neg.default",("correction",)),
    ("erf_magnitude","add","aten.add.Tensor",("negative_correction",1.0)),
    ("signed_erf","mul","aten.mul.Tensor",("erf_magnitude","sign")),
    ("cdf_factor","add","aten.add.Tensor",("signed_erf",1.0)),
    ("half_input","mul","aten.mul.Tensor",("x",0.5)),
    ("gelu","mul","aten.mul.Tensor",("half_input","cdf_factor")),
]


def gelu_contract(event):
    args,kwargs=event["inputs"],event.get("kwargs",{})
    if event.get("mutable") or not isinstance(args,list) or len(args)!=1 or set(kwargs)-{"approximate"} or kwargs.get("approximate","none")!="none":
        raise ValueError("GELU only registers immutable erf-form approximate=none")
    x=args[0];out=event["outputs"]
    if not isinstance(x,dict) or x.get("dtype") not in {"torch.float16","torch.float32"} or not isinstance(x.get("shape"),list):
        raise ValueError("GELU requires an FP16/FP32 tensor")
    if len(x["shape"])>8 or any(type(n) is not int or n<0 for n in x["shape"]):raise ValueError("GELU shape is invalid")
    if not isinstance(out,dict) or out.get("shape")!=x["shape"] or out.get("dtype")!=x["dtype"] or out.get("tensor_id")==x.get("tensor_id"):
        raise ValueError("GELU output shape/precision/identity differs")
    return {"input_dtype":x["dtype"],"output_dtype":out["dtype"],"shape":x["shape"],"approximate":"none",
            "coefficient_profile":"AS7.1.26","clip":CLIP,"inverse_sqrt2":INV_SQRT2,"sign_floor":TINY,
            "p":P,"coefficients":list(COEFFICIENTS)}


def emit_gelu(event,emit):
    cfg=gelu_contract(event);shape=cfg["shape"];x=event["inputs"][0]
    if x["dtype"]!="torch.float32":x=emit("input_f32","aten.to.dtype",[x,"torch.float32"],shape)
    values={"x":x}
    for name,_,operator,operands in STAGES:
        args=[values[a] if isinstance(a,str) else a for a in operands]
        values[name]=emit(name,operator,args,shape,final=name=="gelu" and cfg["output_dtype"]=="torch.float32")
    if cfg["output_dtype"]!="torch.float32":emit("output_cast","aten.to.dtype",[values["gelu"],cfg["output_dtype"]],shape,cfg["output_dtype"],final=True)
    return cfg


def validate_gelu_group(group,nodes):
    cfg=group["gelu_config"];ids=group["lowered_ids"]
    if not ids or any(type(i) is not int or not 0<=i<len(nodes) for i in ids):raise ValueError("GELU missing stage")
    if cfg.get("input_dtype") not in {"torch.float16","torch.float32"} or cfg.get("output_dtype")!=cfg["input_dtype"]:
        raise ValueError("GELU source precision contract differs")
    if cfg.get("approximate")!="none" or cfg.get("coefficient_profile")!="AS7.1.26" or cfg.get("clip")!=CLIP or cfg.get("inverse_sqrt2")!=INV_SQRT2 or cfg.get("sign_floor")!=TINY or cfg.get("p")!=P or cfg.get("coefficients")!=list(COEFFICIENTS):
        raise ValueError("GELU fixed coefficient contract differs")
    shape=cfg.get("shape")
    if not isinstance(shape,list) or len(shape)>8 or any(type(n) is not int or n<0 for n in shape):raise ValueError("GELU source shape contract differs")
    half=cfg["input_dtype"]=="torch.float16"
    expected=(["input_f32"] if half else [])+[s[0] for s in STAGES]+(["output_cast"] if half else [])
    if group["stage_names"]!=expected or len(ids)!=len(expected):raise ValueError("GELU recipe stages incomplete")
    offset=1 if half else 0
    xnode=nodes[ids[0]]
    if half:
        if xnode["kind"]!="cast" or len(xnode["args"])!=2 or xnode["args"][1]!="torch.float32" or xnode["kwargs"] or xnode["output"]!={"shape":shape,"dtype":"f32"}:raise ValueError("GELU input conversion differs")
        x={"value":xnode["id"]}
    else:x=xnode["args"][0]
    if not isinstance(x,dict) or set(x)!={"value"}:raise ValueError("GELU original input binding missing")
    values={"x":x}
    for index,(name,kind,_,operands) in enumerate(STAGES,offset):
        node=nodes[ids[index]];args=[values[a] if isinstance(a,str) else a for a in operands]
        if node["kind"]!=kind or node["args"]!=args or node["kwargs"] or node["output"]!={"shape":shape,"dtype":"f32"}:
            raise ValueError("GELU primitive operands/kind/shape differ from fixed recipe")
        values[name]={"value":node["id"]}
    if half:
        node=nodes[ids[-1]]
        if node["kind"]!="cast" or node["args"]!=[values["gelu"],"torch.float16"] or node["output"]!={"shape":shape,"dtype":"f16"}:raise ValueError("GELU output conversion differs")
    return expected
