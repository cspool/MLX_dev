"""Explicit source-to-primitive recipes; no Python tensor execution or timing."""
import copy
import math
from .model_gelu_program import GELU,GELU_PROFILE,emit_gelu

LAYER_NORM = "aten.layer_norm.default"
LAYER_NORM_PROFILE = "mlx-layernorm-shifted-fp32-v1"


def layer_norm_contract(event):
    if event.get("mutable"):raise ValueError("LayerNorm cannot be mutable")
    args, kwargs = event["inputs"], event.get("kwargs", {})
    if not isinstance(args,list) or not 2 <= len(args) <= 6 or set(kwargs)-{"weight","bias","eps","cudnn_enable"}:
        raise ValueError("unregistered LayerNorm arguments")
    defaults = [("weight",None),("bias",None),("eps",1e-5),("cudnn_enable",True)]
    options={}
    for index,(name,default) in enumerate(defaults,2):
        if index<len(args) and name in kwargs:raise ValueError("duplicate LayerNorm argument")
        options[name]=args[index] if index<len(args) else kwargs.get(name,default)
    x=args[0];out=event["outputs"];normalized=args[1]
    if not isinstance(x,dict) or x.get("dtype") not in {"torch.float16","torch.float32"} or not x.get("shape"):
        raise ValueError("LayerNorm requires a nonscalar FP16/FP32 input")
    if not isinstance(normalized,list) or len(normalized)!=1 or type(normalized[0]) is not int or not 1<=normalized[0]<=2**31 or normalized!=x["shape"][-1:]:
        raise ValueError("LayerNorm currently registers one nonempty final normalized axis")
    if not isinstance(out,dict) or out.get("shape")!=x["shape"] or out.get("dtype")!=x["dtype"]:
        raise ValueError("LayerNorm output shape/precision changed")
    if out.get("tensor_id")==x.get("tensor_id"):
        raise ValueError("LayerNorm must produce a distinct value")
    for name in ("weight","bias"):
        value=options[name]
        if value is not None and (not isinstance(value,dict) or value.get("shape")!=normalized or value.get("dtype") not in {x["dtype"],"torch.float32"}):
            raise ValueError("LayerNorm affine parameter shape/precision mismatch")
    eps=options["eps"]
    if type(eps) not in (int,float) or not math.isfinite(eps) or eps<0 or type(options["cudnn_enable"]) is not bool:
        raise ValueError("LayerNorm epsilon/backend flag is invalid")
    return options


def expand_composites(inventory):
    result=copy.deepcopy(inventory);result["operations"]=[];groups=[]
    result["classification"]="mlx_compiler_expanded_inventory_v1"
    def dense_strides(shape):
        stride=1;out=[]
        for n in reversed(shape):out.append(stride);stride*=max(1,n)
        return out[::-1]
    for source,event in enumerate(inventory["operations"]):
        if event["operator_id"]!=source:raise ValueError("source operators must have sequential identities")
        start=len(result["operations"]);stage_names=[]
        def emit(name,operator,inputs,shape,dtype="torch.float32",kwargs=None,final=False):
            index=len(result["operations"]);identifier=event["outputs"]["tensor_id"] if final else f"__mlx_composite_{source}_{len(stage_names)}"
            if not final and identifier in result["tensors"]:raise ValueError("composite tensor identity collision")
            device=inventory["tensors"][event["inputs"][0]["tensor_id"]]["device"]
            meta={"shape":list(shape),"stride":dense_strides(shape),"dtype":dtype,"device":device,"parameter_or_buffer_names":[]}
            result["tensors"][identifier]=meta
            output={"tensor_id":identifier,"shape":list(shape),"stride":meta["stride"],"dtype":dtype}
            low={key:copy.deepcopy(event[key]) for key in ("request_id","forward_id","phase","module_path","layer_idx") if key in event}
            low.update(operator_id=index,operator=operator,schema="project_explicit_composite_primitive",mutable=False,
                       inputs=copy.deepcopy(inputs),kwargs=kwargs or {},outputs=output,
                       origin_source_operator_id=source,lowering_stage=len(stage_names),lowering_stage_name=name)
            stage_names.append(name);result["operations"].append(low);return output
        if event["operator"]==GELU:
            cfg=emit_gelu(event,emit);profile=GELU_PROFILE
        elif event["operator"]!=LAYER_NORM:
            low=copy.deepcopy(event);low.update(operator_id=start,origin_source_operator_id=source,lowering_stage=0,lowering_stage_name="direct")
            result["operations"].append(low);stage_names=["direct"];profile="direct"
        else:
            cfg=layer_norm_contract(event);x=event["inputs"][0];shape=x["shape"];row=shape[:-1]+[1]
            if x["dtype"]!="torch.float32":x=emit("input_f32","aten.to.dtype",[x,"torch.float32"],shape)
            anchor=emit("anchor_select","aten.select.int",[x,-1,0],shape[:-1])
            anchor=emit("anchor_expand","aten.unsqueeze.default",[anchor,-1],row)
            shifted=emit("shift","aten.sub.Tensor",[x,anchor],shape)
            mean=emit("mean_shift","aten.mean.dim",[shifted,[-1],True],row)
            centered=emit("center","aten.sub.Tensor",[shifted,mean],shape)
            square=emit("square","aten.pow.Tensor_Scalar",[centered,2],shape)
            variance=emit("variance","aten.mean.dim",[square,[-1],True],row)
            variance=emit("epsilon","aten.add.Tensor",[variance,cfg["eps"]],row)
            inverse=emit("rsqrt","aten.rsqrt.default",[variance],row)
            tail=[]
            for name in ("weight","bias"):
                if cfg[name] is not None:tail.append(name)
            narrow=event["outputs"]["dtype"]!="torch.float32"
            value=emit("normalize","aten.mul.Tensor",[centered,inverse],shape,final=not tail and not narrow)
            for index,name in enumerate(tail):
                parameter=cfg[name]
                if parameter["dtype"]!="torch.float32":parameter=emit(name+"_f32","aten.to.dtype",[parameter,"torch.float32"],parameter["shape"])
                value=emit("affine_"+name,"aten.mul.Tensor" if name=="weight" else "aten.add.Tensor",[value,parameter],shape,final=index==len(tail)-1 and not narrow)
            if narrow:emit("output_cast","aten.to.dtype",[value,event["outputs"]["dtype"]],shape,event["outputs"]["dtype"],final=True)
            profile=LAYER_NORM_PROFILE
        groups.append({"source_operator_id":source,"source_operator":event["operator"],"forward_id":event["forward_id"],"layer_idx":event["layer_idx"],
                       "lowering_profile":profile,"lowered_ids":list(range(start,len(result["operations"]))),"stage_names":stage_names})
        if profile==LAYER_NORM_PROFILE:
            groups[-1]["layer_norm_config"]={"input_dtype":event["inputs"][0]["dtype"],"output_dtype":event["outputs"]["dtype"],
                "shape":event["inputs"][0]["shape"],"epsilon":cfg["eps"],"cudnn_enable":cfg["cudnn_enable"],
                "weight_dtype":cfg["weight"]["dtype"] if cfg["weight"] is not None else None,
                "bias_dtype":cfg["bias"]["dtype"] if cfg["bias"] is not None else None}
        elif profile==GELU_PROFILE:groups[-1]["gelu_config"]=cfg
    return result,groups
