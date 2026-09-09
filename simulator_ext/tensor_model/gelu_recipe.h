#pragma once
#include "tensor.h"
#include <map>

namespace mlx::tensor_model {
struct GeluArg {const char *name=nullptr;double scalar=0;GeluArg(const char *n):name(n){} GeluArg(double v):scalar(v){}};
struct GeluStage {const char *name,*kind;std::vector<GeluArg> args;};
inline std::vector<std::string> validate_gelu_group(const Json::Value &g,const Json::Value &nodes){
  const auto &cfg=g["gelu_config"],&ids=g["lowered_ids"];
  require((cfg["input_dtype"]=="torch.float16"||cfg["input_dtype"]=="torch.float32")&&cfg["output_dtype"]==cfg["input_dtype"],"GELU source precision contract differs");
  const double coefficients[]={0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429};
  require(cfg["approximate"]=="none"&&cfg["coefficient_profile"]=="AS7.1.26"&&cfg["clip"].asDouble()==8.0&&cfg["inverse_sqrt2"].asDouble()==0.7071067811865476&&cfg["sign_floor"].asDouble()==0x1p-126&&cfg["p"].asDouble()==0.3275911&&cfg["coefficients"].isArray()&&cfg["coefficients"].size()==5,"GELU fixed coefficient contract differs");
  for(unsigned i=0;i<5;++i)require(cfg["coefficients"][i].isNumeric()&&cfg["coefficients"][i].asDouble()==coefficients[i],"GELU coefficient differs");
  auto sizes=shape(cfg["shape"]);for(auto size:sizes)require(size>=0,"GELU shape is invalid");
  const std::vector<GeluStage> recipe={
    {"negative_input","neg",{"x"}}, {"clip_negative","maximum",{"negative_input",-8.0}},
    {"clip_upper","neg",{"clip_negative"}}, {"clip_lower","maximum",{"clip_upper",-8.0}},
    {"scaled","mul",{"clip_lower",0.7071067811865476}}, {"negative_scaled","neg",{"scaled"}},
    {"magnitude","maximum",{"scaled","negative_scaled"}}, {"sign_denominator","maximum",{"magnitude",0x1p-126}},
    {"sign","div",{"scaled","sign_denominator"}}, {"t_scale","mul",{"magnitude",0.3275911}},
    {"t_denominator","add",{"t_scale",1.0}}, {"t","div",{1.0,"t_denominator"}},
    {"poly_a5","mul",{"t",1.061405429}}, {"poly_add4","add",{"poly_a5",-1.453152027}},
    {"poly_mul3","mul",{"poly_add4","t"}}, {"poly_add3","add",{"poly_mul3",1.421413741}},
    {"poly_mul2","mul",{"poly_add3","t"}}, {"poly_add2","add",{"poly_mul2",-0.284496736}},
    {"poly_mul1","mul",{"poly_add2","t"}}, {"poly_add1","add",{"poly_mul1",0.254829592}},
    {"polynomial","mul",{"poly_add1","t"}}, {"square","mul",{"scaled","scaled"}},
    {"negative_square","neg",{"square"}}, {"exponential","exp",{"negative_square"}},
    {"correction","mul",{"polynomial","exponential"}}, {"negative_correction","neg",{"correction"}},
    {"erf_magnitude","add",{"negative_correction",1.0}}, {"signed_erf","mul",{"erf_magnitude","sign"}},
    {"cdf_factor","add",{"signed_erf",1.0}}, {"half_input","mul",{"x",0.5}},
    {"gelu","mul",{"half_input","cdf_factor"}}
  };
  bool half=cfg["input_dtype"]=="torch.float16";unsigned offset=half?1:0;
  require(ids.size()==recipe.size()+(half?2:0),"GELU recipe stages incomplete");
  for(const auto &id:ids)require(id.isUInt()&&id.asUInt()<nodes.size(),"GELU missing stage");
  std::vector<std::string> names;std::map<std::string,std::string> values;const auto &first=nodes[ids[0].asUInt()];
  if(half){require(first["kind"]=="cast"&&first["args"].size()==2&&first["args"][1]=="torch.float32"&&first["kwargs"].isObject()&&first["kwargs"].empty()&&first["output"]["dtype"]=="f32"&&first["output"]["shape"]==cfg["shape"],"GELU input conversion differs");values["x"]=first["id"].asString();names.push_back("input_f32");}
  else{require(first["args"][0].isObject()&&first["args"][0].size()==1&&first["args"][0]["value"].isString(),"GELU original input binding missing");values["x"]=first["args"][0]["value"].asString();}
  for(unsigned i=0;i<recipe.size();++i){const auto &stage=recipe[i];const auto &node=nodes[ids[i+offset].asUInt()];
    require(node["kind"]==stage.kind&&node["args"].isArray()&&node["args"].size()==stage.args.size()&&node["kwargs"].isObject()&&node["kwargs"].empty()&&node["output"]["dtype"]=="f32"&&node["output"]["shape"]==cfg["shape"],"GELU primitive kind/shape differs from recipe");
    for(unsigned j=0;j<stage.args.size();++j){const auto &want=stage.args[j];const auto &arg=node["args"][j];
      if(want.name)require(arg.isObject()&&arg.size()==1&&arg["value"]==values.at(want.name),"GELU primitive operand wiring differs");
      else require(arg.isNumeric()&&arg.asDouble()==want.scalar,"GELU primitive literal differs");
    }
    values[stage.name]=node["id"].asString();names.push_back(stage.name);
  }
  if(half){const auto &last=nodes[ids[ids.size()-1].asUInt()];require(last["kind"]=="cast"&&last["args"].size()==2&&last["args"][0]["value"]==values.at("gelu")&&last["args"][1]=="torch.float16"&&last["output"]["dtype"]=="f16"&&last["output"]["shape"]==cfg["shape"],"GELU output conversion differs");names.push_back("output_cast");}
  return names;
}
}
