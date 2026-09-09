#pragma once
#include "tensor.h"
#include <stdexcept>

namespace mlx::control_model {
inline bool extended_kind(const std::string &kind){return kind=="ge"||kind=="bitwise_and"||kind=="all"||kind=="guard";}
inline const char *profile(const std::string &kind){return extended_kind(kind)?"mlx-controller-rv64-leaf-v2":"mlx-controller-rv64-leaf-v1";}
inline unsigned input_count(const std::string &kind){return kind=="arange"?0:kind=="argmax"||kind=="all"?1:2;}
inline void validate_extended_program(const Json::Value &program,const std::string &kind){
  if(!extended_kind(kind))return;
  auto r=[](unsigned f3,unsigned d,unsigned a,unsigned b){return b<<20|a<<15|f3<<12|d<<7|0x33u;};
  auto i=[](unsigned f3,unsigned d,unsigned a,unsigned immediate){return immediate<<20|a<<15|f3<<12|d<<7|0x13u;};
  auto words=[](std::initializer_list<unsigned> code){Json::Value out(Json::arrayValue);for(auto word:code)out.append(word);return out;};
  Json::Value phases(Json::objectValue);
  bool integer=program["input_dtype"]=="i64";
  if(kind=="ge")phases["body"]=integer?words({r(2,12,10,11),i(4,12,12,1)}):words({0x50u<<25|10u<<20|11u<<15|12u<<7|0x53u});
  else{
    if(!integer)throw std::runtime_error("Boolean controller requires integer register domain");
    if(kind=="bitwise_and")phases["body"]=words({r(7,12,10,11)});
    else if(kind=="all"){phases["init"]=words({i(0,12,0,1)});phases["body"]=words({r(3,10,0,10),r(7,12,12,10)});}
    else phases["body"]=words({r(3,12,0,10),r(4,13,12,11)});
  }
  const auto &actual=program["phases"];
  if(!actual.isObject()||actual.getMemberNames()!=phases.getMemberNames())throw std::runtime_error("noncanonical Boolean/guard RV64 program");
  for(const auto &name:phases.getMemberNames()){
    if(!actual[name].isArray()||actual[name].size()!=phases[name].size())throw std::runtime_error("noncanonical Boolean/guard RV64 program");
    for(unsigned j=0;j<phases[name].size();++j)
      if(!actual[name][j].isUInt()||actual[name][j].asUInt()!=phases[name][j].asUInt())throw std::runtime_error("noncanonical Boolean/guard RV64 program");
  }
}
struct Stats {
  uint64_t calls=0,instructions=0,branches=0,read_bytes=0,write_bytes=0;
  unsigned fflags=0;
  bool v2=false;
  Json::Value json()const;
};
tensor_model::Tensor execute(const Json::Value &node,const tensor_model::Values &values,Stats &stats);
} // namespace mlx::control_model
