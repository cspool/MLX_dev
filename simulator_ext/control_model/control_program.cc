#include "control_program.h"
#include "rv64_leaf.h"
#include <cstring>
#include <cfenv>
#include <stdexcept>

namespace mlx::control_model {
using namespace tensor_model;
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
int64_t signed_bits(uint64_t raw){int64_t value;std::memcpy(&value,&raw,8);return value;}
const Tensor &ref(const Json::Value &arg,const Values &values){check(arg.isObject()&&arg["value"].isString(),"expected controller tensor reference");auto i=values.find(arg["value"].asString());check(i!=values.end(),"unbound controller input");return i->second;}
uint64_t broadcast(uint64_t flat,const Shape &out,const Shape &in){uint64_t result=0,step=1;check(in.size()<=out.size(),"controller broadcast rank mismatch");
  for(size_t d=out.size();d-->0;){auto index=out[d]?flat%out[d]:0;if(out[d])flat/=out[d];if(d+in.size()>=out.size()){auto n=in[d+in.size()-out.size()];check(n==1||n==out[d],"controller broadcast extent mismatch");if(n!=1)result+=index*step;step*=n;}}return result;}
}
Json::Value Stats::json()const{Json::Value r(Json::objectValue);r["profile"]="mlx-controller-rv64-leaf-v1";r["classification"]="rv64_alu_leaf_execution_not_rocket_or_system_validation";r["calls"]=Json::UInt64(calls);r["instructions"]=Json::UInt64(instructions);r["branches_taken"]=Json::UInt64(branches);r["read_bytes"]=Json::UInt64(read_bytes);r["write_bytes"]=Json::UInt64(write_bytes);r["fflags_observed"]=fflags;r["register_bytes"]=512;r["rocket_execution_verified"]=false;r["timing_verified"]=false;return r;}
Tensor execute(const Json::Value &node,const Values &values,Stats &stats){
  const auto &p=node["control_program"],&args=node["args"];auto kind=node["kind"].asString();
  check(p["profile"]=="mlx-controller-rv64-leaf-v1"&&p["kind"]==node["kind"]&&p["xlen"]==64&&p["flen"]==64&&p["gpr_count"]==32&&p["fpr_count"]==32,"controller profile/resource mismatch");
  check(p["input_dtype"]=="i64"||p["input_dtype"]=="f16"||p["input_dtype"]=="f32","unregistered controller input precision");
  check(std::fegetround()==FE_TONEAREST,"controller input conversions require RNE");
  check(kind=="arange"||kind=="add"||kind=="mul"||kind=="le"||kind=="argmax","unsupported controller operation");
  unsigned words=0;for(const auto &name:p["phases"].getMemberNames()){check(p["phases"][name].isArray(),"controller phase is not code");words+=p["phases"][name].size();}check(words>0&&words<=32,"controller template capacity violation");
  auto output=Tensor::allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));
  const bool integer=p["input_dtype"]=="i64";
  auto phase=[&](RV64 &state,const char *name){check(p["phases"].isMember(name),"missing controller phase");state.run(p["phases"][name]);};
  auto accumulate=[&](const RV64 &state){stats.instructions+=state.retired;stats.branches+=state.branches_taken;stats.fflags|=state.fflags;};
  auto load=[&](RV64 &state,const Json::Value &arg,unsigned reg,uint64_t flat){
    if(arg.isObject()&&arg.isMember("value")){const auto &input=ref(arg,values);auto index=broadcast(flat,output.sizes,input.sizes);stats.read_bytes+=element_bytes(input.type);
      if(integer){check(input.type==DType::I64||input.type==DType::Bool,"controller integer operand has floating dtype");state.x[reg]=uint64_t(input.integer(index));}
      else{check(input.type==DType::F16||input.type==DType::F32,"controller FP comparison requires floating tensors");state.set_float(reg,input.number(index));}}
    else if(integer){check(arg.isInt64()||arg.isBool(),"controller integer scalar is not exact int64");state.x[reg]=arg.isBool()?arg.asBool():uint64_t(arg.asInt64());}
    else state.set_float(reg,float(scalar(arg)));
  };
  ++stats.calls;
  if(kind=="arange"){
    check(integer&&output.type==DType::I64&&args.size()==1&&args[0].isUInt64()&&output.sizes==Shape{int64_t(args[0].asUInt64())},"controller arange contract mismatch");
    RV64 state;phase(state,"init");for(uint64_t index=0;index<output.numel();++index){phase(state,"body");output.set_integer(index,signed_bits(state.x[12]));phase(state,"advance");}accumulate(state);
  }else if(kind=="argmax"){
    const auto &input=ref(args[0],values);check(!input.sizes.empty()&&input.sizes.back()>0&&output.type==DType::I64,"controller argmax requires nonempty final dimension");
    check(dtype_name(input.type)==p["input_dtype"].asString(),"argmax precision binding mismatch");auto dim=args[1].asInt64();check(dim==-1||dim==int64_t(input.sizes.size()-1),"controller argmax axis unsupported");
    auto expected=input.sizes;if(args.size()>2&&args[2].asBool())expected.back()=1;else expected.pop_back();check(expected==output.sizes,"argmax output shape mismatch");uint64_t width=input.sizes.back();
    for(uint64_t row=0;row<input.numel()/width;++row){RV64 state;
      auto value=[&](uint64_t column){stats.read_bytes+=element_bytes(input.type);if(integer)state.x[10]=uint64_t(input.integer(row*width+column));else state.set_float(10,input.number(row*width+column));};
      value(0);phase(state,"init");for(uint64_t column=1;column<width;++column){value(column);phase(state,"advance");phase(state,"compare");phase(state,"select");}
      check(state.x[20]<width,"controller argmax produced an invalid index");output.set_integer(row,int64_t(state.x[20]));accumulate(state);
    }
  }else{
    check(args.size()==2&&(kind=="le"?output.type==DType::Bool:integer&&output.type==DType::I64),"controller elementwise type/arity mismatch");
    check(!node["kwargs"].isMember("alpha")||scalar(node["kwargs"]["alpha"])==1,"controller integer add requires unit alpha");
    Shape expected;
    for(const auto &arg:args)if(arg.isObject()&&arg.isMember("value")){const auto &input=ref(arg,values);if(expected.size()<input.sizes.size())expected.insert(expected.begin(),input.sizes.size()-expected.size(),1);auto shift=expected.size()-input.sizes.size();
      for(size_t d=0;d<input.sizes.size();++d){auto size=input.sizes[d];auto &current=expected[shift+d];if(current==1)current=size;else check(size==1||size==current,"controller broadcast shape mismatch");}}
    check(expected==output.sizes,"controller output shape is not the broadcast result");
    for(uint64_t index=0;index<output.numel();++index){RV64 state;load(state,args[0],10,index);load(state,args[1],11,index);phase(state,"body");output.set_integer(index,signed_bits(state.x[12]));accumulate(state);}
  }
  stats.write_bytes+=output.numel()*element_bytes(output.type);return output;
}
} // namespace mlx::control_model
