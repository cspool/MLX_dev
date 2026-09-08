#include "memory_program.h"
#include "memory_schedule.h"
#include "../model_io/tensor_memory_port.h"
#include "mlx_tagged_simulator.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

namespace mlx::memory_model {
using namespace tensor_model;
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
unsigned axis(int64_t value,size_t rank,bool insert=false){int64_t count=rank+unsigned(insert);if(value<0)value+=count;check(value>=0&&value<count,"memory axis out of bounds");return unsigned(value);}
bool dense(const Tensor &t){
  std::vector<std::pair<int64_t,int64_t>> dimensions;for(size_t i=0;i<t.sizes.size();++i)if(t.sizes[i]>1)dimensions.emplace_back(t.steps[i],t.sizes[i]);
  std::sort(dimensions.begin(),dimensions.end());uint64_t step=1;
  for(auto [stride,size]:dimensions){if(stride<0||uint64_t(stride)!=step)return false;step*=size;}return true;
}
Shape resolve_shape(const Json::Value &raw,uint64_t count){
  Shape out=shape(raw);int inferred=-1;uint64_t known=1;
  for(size_t i=0;i<out.size();++i){if(out[i]==-1){check(inferred<0,"multiple inferred reshape axes");inferred=int(i);}
    else{check(out[i]>=0&&(!out[i]||known<=UINT64_MAX/uint64_t(out[i])),"invalid reshape extent/product");known*=out[i];}}
  if(inferred>=0){check(known&&count%known==0,"ambiguous inferred reshape");out[inferred]=count/known;}
  check(elements(out)==count,"reshape changes element count");return out;
}
std::optional<Shape> reshape_steps(const Tensor &t,const Shape &target){
  if(t.sizes.empty())return strides(target);
  if(!t.numel())return t.sizes==target?t.steps:strides(target);
  Shape out(target.size());int next=int(target.size())-1;uint64_t old_count=1,new_count=1;int64_t base=t.steps.back();
  for(int d=int(t.sizes.size())-1;d>=0;--d){old_count*=t.sizes[d];
    bool boundary=d==0||(t.sizes[d-1]!=1&&(base<0||uint64_t(t.steps[d-1])!=old_count*uint64_t(base)));
    if(boundary){while(next>=0&&(new_count<old_count||target[next]==1)){check(base>=0&&(!base||new_count<=INT64_MAX/uint64_t(base)),"reshape stride overflow");out[next]=int64_t(new_count)*base;new_count*=target[next];--next;}
      if(new_count!=old_count)return std::nullopt;
      if(d){base=t.steps[d-1];old_count=new_count=1;}}}
  return next==-1?std::optional<Shape>(out):std::nullopt;
}
void span(const Tensor &t){
  check(bool(t.storage),"memory tensor has no storage metadata");
  check(t.sizes.size()==t.steps.size()&&t.offset>=0,"invalid memory view rank/base");
  if(!t.numel())return;
  uint64_t limit=t.storage->bytes/element_bytes(t.type);check(uint64_t(t.offset)<limit,"memory view base exceeds storage");uint64_t at=t.offset;
  for(size_t d=0;d<t.sizes.size();++d){check(t.steps[d]>=0,"negative memory view stride");uint64_t count=t.sizes[d]-1,step=t.steps[d];check(!step||count<=(limit-1-at)/step,"memory view exceeds storage");at+=count*step;}
}
void layout_matches(const Tensor &t,const Json::Value &p){
  check(bool(t.storage),"memory tensor has no storage metadata");
  check(dtype_name(t.type)==p["dtype"].asString()&&t.sizes==shape(p["shape"])&&t.steps==shape(p["strides"])&&t.offset==p["offset"].asInt64(),"runtime layout disagrees with compiled memory plan");
  check(t.storage->bytes/element_bytes(t.type)==p["storage_elements"].asUInt64(),"memory storage extent disagrees with plan");span(t);
}
const Tensor &ref(const Json::Value &arg,const Values &values){check(arg.isObject()&&arg["value"].isString(),"expected memory SSA reference");auto it=values.find(arg["value"].asString());check(it!=values.end(),"unbound memory SSA value");return it->second;}
uint64_t broadcast(uint64_t index,const Shape &out,const Shape &in){
  check(in.size()<=out.size(),"memory broadcast rank mismatch");uint64_t result=0,step=1;
  for(size_t d=out.size();d-->0;){uint64_t pos=out[d]?index%out[d]:0;if(out[d])index/=out[d];if(d+in.size()>=out.size()){auto n=in[d+in.size()-out.size()];check(n==1||n==out[d],"memory broadcast extent mismatch");if(n!=1)result+=pos*step;step*=n;}}
  return result;
}
struct Value {uint64_t bits=0;DType type=DType::F32;bool valid=false;};
int64_t integer_bits(uint64_t raw){int64_t value;std::memcpy(&value,&raw,8);return value;}
void append_index(Value &address,const Tensor &source,unsigned dim,int64_t coordinate){
  check(dim<source.sizes.size(),"advanced index axis out of range");
  auto extent=source.sizes[dim];if(coordinate<0)coordinate+=extent;
  check(coordinate>=0&&coordinate<extent,"advanced index out of range");
  if(dim==0){address.bits=0;address.type=DType::I64;address.valid=true;}
  check(address.valid&&address.type==DType::I64,"advanced index has an unready address accumulator");
  check(uint64_t(extent)&&address.bits<=(UINT64_MAX-uint64_t(coordinate))/uint64_t(extent),"advanced index address overflow");
  address.bits=address.bits*uint64_t(extent)+uint64_t(coordinate);
}
float number(const Value &v){
  if(v.type==DType::F16)return tagged::half_to_float(uint16_t(v.bits));
  if(v.type==DType::F32){uint32_t raw=uint32_t(v.bits);float out;std::memcpy(&out,&raw,4);return out;}
  return v.type==DType::I64?float(integer_bits(v.bits)):float(v.bits!=0);
}
Value convert(Value v,DType target){
  check(v.valid,"conversion reads an unready transfer register");if(v.type==target)return v;Value out;out.type=target;out.valid=true;
  if(target==DType::Bool){out.bits=v.type==DType::I64?integer_bits(v.bits)!=0:number(v)!=0.0f;return out;}
  if(target==DType::I64){int64_t value;
    if(v.type==DType::Bool)value=v.bits!=0;
    else{double f=number(v);check(std::isfinite(f)&&f>=-9223372036854775808.0&&f<9223372036854775808.0,"float-to-int64 conversion outside registered finite range");value=int64_t(f);}
    std::memcpy(&out.bits,&value,8);return out;}
  float f=number(v);if(target==DType::F16)out.bits=tagged::float_to_half(f);else{uint32_t raw;std::memcpy(&raw,&f,4);out.bits=raw;}return out;
}
Value literal(const Json::Value &arg){
  Value v;v.valid=true;if(arg.isInt64()){v.type=DType::I64;auto i=arg.asInt64();std::memcpy(&v.bits,&i,8);}
  else if(arg.isBool()){v.type=DType::Bool;v.bits=arg.asBool();}
  else{v.type=DType::F32;float f=float(scalar(arg));uint32_t raw;std::memcpy(&raw,&f,4);v.bits=raw;}return v;
}
Tensor derive_view(const Json::Value &node,Tensor t,const Json::Value &p){
  const auto kind=node["kind"].asString();const auto &args=node["args"];
  if(kind=="dropout_inference")check(args.size()==3&&args[2].isBool()&&!args[2].asBool(),"training dropout cannot be a view");
  if(kind=="transpose"){auto a=axis(args[1].asInt64(),t.sizes.size()),b=axis(args[2].asInt64(),t.sizes.size());std::swap(t.sizes[a],t.sizes[b]);std::swap(t.steps[a],t.steps[b]);}
  else if(kind=="unsqueeze"){auto d=axis(args[1].asInt64(),t.sizes.size(),true);int64_t step=d<t.sizes.size()?t.sizes[d]*t.steps[d]:1;t.sizes.insert(t.sizes.begin()+d,1);t.steps.insert(t.steps.begin()+d,step);}
  else if(kind=="select"){auto d=axis(args[1].asInt64(),t.sizes.size());auto i=args[2].asInt64();if(i<0)i+=t.sizes[d];check(i>=0&&i<t.sizes[d],"select index out of bounds");t.offset+=i*t.steps[d];t.sizes.erase(t.sizes.begin()+d);t.steps.erase(t.steps.begin()+d);}
  else if(kind=="slice"){auto d=axis(args.size()>1?args[1].asInt64():0,t.sizes.size());auto n=t.sizes[d];int64_t start=args.size()>2&&!args[2].isNull()?args[2].asInt64():0,end=args.size()>3&&!args[3].isNull()?args[3].asInt64():n,step=args.size()>4?args[4].asInt64():1;
    check(step>0,"nonpositive memory slice step");if(start<0)start+=n;if(end<0)end+=n;start=std::clamp<int64_t>(start,0,n);end=std::clamp<int64_t>(end,0,n);t.offset+=start*t.steps[d];t.sizes[d]=end>start?(end-start+step-1)/step:0;t.steps[d]*=step;}
  else if(kind=="expand"){auto wanted=shape(args[1]);check(wanted.size()>=t.sizes.size(),"expand reduced rank");auto extra=wanted.size()-t.sizes.size();t.sizes.insert(t.sizes.begin(),extra,1);t.steps.insert(t.steps.begin(),extra,0);
    for(size_t d=0;d<wanted.size();++d){if(wanted[d]==-1){check(d>=extra,"cannot infer leading expand axis");wanted[d]=t.sizes[d];}check(wanted[d]>=0&&(t.sizes[d]==1||t.sizes[d]==wanted[d]),"invalid expand extent");if(wanted[d]!=t.sizes[d])t.steps[d]=0;}t.sizes=wanted;}
  else if(kind=="squeeze"){check(args.size()==2&&args[1].isInt64(),"squeeze requires an integer axis");if(t.sizes.empty())check(args[1].asInt64()==-1||args[1].asInt64()==0,"scalar squeeze axis out of bounds");else{auto d=axis(args[1].asInt64(),t.sizes.size());if(t.sizes[d]==1){t.sizes.erase(t.sizes.begin()+d);t.steps.erase(t.steps.begin()+d);}}}
  else if(kind=="reshape"){auto wanted=resolve_shape(args[1],t.numel());auto steps=reshape_steps(t,wanted);check(bool(steps),"compiled view requires a data copy");t.sizes=wanted;t.steps=*steps;}
  else if(kind=="cast"||kind=="cast_device"||kind=="contiguous"){
    check(dtype_name(t.type)==node["output"]["dtype"].asString(),"dtype-changing conversion cannot alias");unsigned copy_index=kind=="cast"?3:4;
    check(kind=="contiguous"||args.size()<=copy_index||!args[copy_index].asBool(),"forced copy cannot alias");
    check(p["same_reference_device"].asBool(),"device transfer cannot alias");
    if(kind=="contiguous"||node["kwargs"]["memory_format"]=="torch.contiguous_format")check(t.contiguous(),"noncontiguous conversion cannot alias");
  }else check(kind=="alias"||kind=="dropout_inference","operation cannot be an affine view");
  return t;
}
} // namespace

bool supports(const std::string &k){return extended_kind(k)||k=="embedding"||k=="where"||k=="cat"||k=="cast"||k=="cast_device"||k=="contiguous"||k=="reshape"||k=="transpose"||k=="slice"||k=="select"||k=="unsqueeze"||k=="expand"||k=="alias"||k=="dropout_inference";}
Json::Value Stats::json()const{Json::Value r(Json::objectValue);r["classification"]="bounded_memory_plans_not_system_dma_validation";r["profile"]=v2?"mlx-memory-plan-v2":"mlx-memory-plan-v1";r["calls"]=Json::UInt64(calls);r["view_elisions"]=Json::UInt64(views);r["allocations"]=Json::UInt64(allocations);r["instructions"]=Json::UInt64(instructions);r["read_bytes"]=Json::UInt64(read_bytes);r["write_bytes"]=Json::UInt64(write_bytes);r["index_reads"]=Json::UInt64(index_reads);r["predicate_reads"]=Json::UInt64(predicate_reads);for(unsigned op=1;op<6;++op)r["opcode_counts"][std::to_string(op)]=Json::UInt64(opcode_counts[op]);r["staging_bytes"]=128;r["register_bytes_total"]=32;r["physical_dma_verified"]=false;r["timing_verified"]=false;return r;}

namespace {
struct Prepared {
  Tensor output;
  std::string kind,selector;
  std::vector<std::string> names;
  unsigned concat_dim=0,slot_bytes=0;
  uint64_t inner=1;
  std::vector<const Tensor*> concatenated;
  bool view=false,empty_embedding=false;
};
Prepared prepare(const Json::Value &node,const Values &values,const Tensor *supplied=nullptr){
  Prepared result;
  const auto &p=node["memory_program"],&args=node["args"],&out_spec=p["output_layout"];auto kind=node["kind"].asString();
  check(supports(kind)&&p["profile"]==profile(kind)&&p["kind"]==node["kind"],"unsupported/mismatched memory program");
  check(p["register_count"]==4&&p["register_bytes"]==8&&p["staging_bytes"]==128&&p["chunk_bytes"]==64,"memory transfer resource contract mismatch");
  std::set<std::string> referenced;
  auto collect=[&](auto &&self,const Json::Value &v)->void{if(v.isObject()&&v.isMember("value"))referenced.insert(v["value"].asString());else if(v.isArray())for(const auto &child:v)self(self,child);};
  collect(collect,args);
  auto names=p["input_layouts"].getMemberNames();check(referenced==std::set<std::string>(names.begin(),names.end()),"memory plan does not bind exactly its source operands");
  std::sort(names.begin(),names.end());result.names=names;result.kind=kind;
  for(const auto &id:p["input_layouts"].getMemberNames()){auto it=values.find(id);check(it!=values.end(),"missing planned memory operand");layout_matches(it->second,p["input_layouts"][id]);}
  if(p["mode"]=="view"){
    check(p["words"].isArray()&&p["words"].empty(),"view plan contains transfer instructions");const auto &input=ref(args[0],values);
    auto output=derive_view(node,input,p);layout_matches(output,out_spec);
    check(output.storage==input.storage&&out_spec["root"]==p["input_layouts"][args[0]["value"].asString()]["root"],"view lost its storage owner");
    check(output.sizes==shape(node["output"]["shape"])&&dtype_name(output.type)==node["output"]["dtype"].asString(),"view output contract mismatch");
    if(supplied){layout_matches(*supplied,out_spec);check(supplied->storage==output.storage,"supplied view lost its storage owner");}
    result.output=output;result.view=true;return result;
  }
  check(p["mode"]=="transfer"&&p["words"].isArray()&&!p["words"].empty()&&p["words"].size()<=(kind=="advanced_index"?11u:4u),"invalid memory transfer program");
  for(const auto &raw:p["words"]){check(raw.isUInt(),"invalid memory instruction word");unsigned word=raw.asUInt(),op=word&255;check(op>=1&&op<=5&&(word>>10)==0&&(op==2||(word>>8)==0),"memory opcode/reserved field violation");}
  const auto selector=p["selector"].asString();check((selector=="linear"&&(kind=="cast"||kind=="cast_device"||kind=="contiguous"||kind=="reshape"))||(selector=="concat"&&kind=="cat")||(selector=="indexed_rows"&&kind=="embedding")||(selector=="predicate_select"&&kind=="where")||(selector=="indexed_nd"&&kind=="advanced_index")||(selector=="constant_one"&&kind=="new_ones"),"memory selector/operation mismatch");
  auto target=dtype(out_spec["dtype"].asString());auto output=supplied?*supplied:Tensor::allocate(target,shape(out_spec["shape"]));
  if(!supplied)output.steps=shape(out_spec["strides"]);
  for(const auto &name:names)check(output.storage!=values.at(name).storage,"transfer output aliases an input storage owner");
  check(out_spec["offset"].asInt64()==0&&out_spec["root"]==node["id"],"allocated output has an invalid root/base");layout_matches(output,out_spec);
  check(output.sizes==shape(node["output"]["shape"])&&dtype_name(target)==node["output"]["dtype"].asString(),"memory output shape/type mismatch");
  auto expected_steps=strides(output.sizes);
  if(kind=="cast"||kind=="cast_device"){const auto &input=ref(args[0],values);check(input.sizes==output.sizes,"cast changes shape");if(node["kwargs"]["memory_format"]!="torch.contiguous_format"&&dense(input))expected_steps=input.steps;}
  if(kind=="reshape"){const auto &input=ref(args[0],values);check(node["source_operator"]!="aten.view.default","view cannot silently materialize");check(resolve_shape(args[1],input.numel())==output.sizes,"copy reshape has incorrect shape");}
  if(kind=="contiguous")check(ref(args[0],values).sizes==output.sizes,"contiguous copy changes shape");
  check(output.steps==expected_steps,"output allocation has unregistered layout");
  if(kind=="embedding"){const auto &weight=ref(args[0],values),&ids=ref(args[1],values);check(weight.sizes.size()==2&&ids.type==DType::I64&&target==weight.type,"embedding input contract mismatch");Shape expected=ids.sizes;expected.push_back(weight.sizes[1]);check(expected==output.sizes,"embedding output shape mismatch");}
  if(kind=="advanced_index"){
    const auto &input=ref(args[0],values);check(args.size()==2&&args[1].isArray()&&input.sizes.size()>=1&&input.sizes.size()<=8&&args[1].size()==input.sizes.size()&&target==input.type,"advanced index requires one I64 tensor per source axis");
    Shape expected;
    for(const auto &arg:args[1]){const auto &indices=ref(arg,values);check(indices.type==DType::I64,"advanced index requires I64 index tensors");
      if(expected.size()<indices.sizes.size())expected.insert(expected.begin(),indices.sizes.size()-expected.size(),1);
      auto shift=expected.size()-indices.sizes.size();for(size_t d=0;d<indices.sizes.size();++d){auto size=indices.sizes[d];auto &current=expected[shift+d];if(current==1)current=size;else check(size==1||size==current,"advanced index broadcast mismatch");}}
    check(expected==output.sizes,"advanced index output shape mismatch");
  }
  if(kind=="new_ones"){
    const auto &input=ref(args[0],values);check(args.size()==2&&args[1].isArray()&&shape(args[1])==output.sizes,"new_ones output shape mismatch");
    const char *types[]={"torch.float16","torch.float32","torch.int64","torch.bool"};const auto &kw=node["kwargs"];
    check(kw["dtype"].isNull()?target==input.type:kw["dtype"]==types[unsigned(target)],"new_ones dtype contract mismatch");
    check(kw["layout"].isNull()||kw["layout"]=="torch.strided","new_ones layout not registered");
    check(kw["pin_memory"].isNull()||(kw["pin_memory"].isBool()&&!kw["pin_memory"].asBool()),"new_ones pinned memory not registered");
  }
  if(kind=="where"){
    check(ref(args[0],values).type==DType::Bool,"where requires a Boolean predicate");Shape expected;
    for(const auto &arg:args)if(arg.isObject()&&arg.isMember("value")){const auto &input=ref(arg,values);if(expected.size()<input.sizes.size())expected.insert(expected.begin(),input.sizes.size()-expected.size(),1);size_t shift=expected.size()-input.sizes.size();
      for(size_t d=0;d<input.sizes.size();++d){auto extent=input.sizes[d];auto &current=expected[shift+d];if(current==1)current=extent;else check(extent==1||current==extent,"where operand broadcast mismatch");}}
    check(expected==output.sizes,"where output shape mismatch");
  }
  unsigned concat_dim=0;uint64_t inner=1;std::vector<const Tensor*> concatenated;
  if(kind=="cat"){
    for(const auto &arg:args[0]){const auto &input=ref(arg,values);if(input.sizes!=Shape{0})concatenated.push_back(&input);}
    if(!concatenated.empty()){concat_dim=axis(args.size()>1?args[1].asInt64():0,concatenated[0]->sizes.size());auto expected=concatenated[0]->sizes;expected[concat_dim]=0;
      for(const auto *input:concatenated){check(input->type==target&&input->sizes.size()==expected.size(),"concat input dtype/rank mismatch");for(size_t d=0;d<expected.size();++d)if(d!=concat_dim)check(input->sizes[d]==concatenated[0]->sizes[d],"concat non-axis extent mismatch");expected[concat_dim]+=input->sizes[concat_dim];}
      check(expected==output.sizes,"concat output extent mismatch");for(size_t d=concat_dim+1;d<expected.size();++d)inner*=expected[d];}
    else check(!output.numel(),"empty concat cannot synthesize values");
  }
  // Validate the complete template even for empty outputs. This also prevents
  // a missing index/predicate load from reusing a previous element's register.
  std::vector<unsigned> expected;
  if(selector=="indexed_rows")expected.push_back(4);
  if(selector=="indexed_nd")expected.insert(expected.end(),ref(args[0],values).sizes.size(),4);
  if(selector=="predicate_select")expected.push_back(5);
  expected.insert(expected.end(),{1,2|(unsigned(target)<<8),3});
  check(p["words"].size()==expected.size(),"memory transfer template mismatch");
  for(unsigned i=0;i<expected.size();++i)check(p["words"][i].asUInt()==expected[i],"memory transfer template mismatch");
  result.output=output;result.selector=selector;result.concat_dim=concat_dim;result.inner=inner;result.concatenated=concatenated;
  result.empty_embedding=kind=="embedding"&&ref(args[0],values).sizes[1]==0;
  unsigned slot_bytes=element_bytes(target);
  for(const auto &id:p["input_layouts"].getMemberNames())slot_bytes=std::max(slot_bytes,element_bytes(values.at(id).type));
  if(kind=="where")slot_bytes=8; // selected literals can have integer width
  result.slot_bytes=slot_bytes;
  check(std::fegetround()==FE_TONEAREST,"memory conversions require round-to-nearest-even");
  return result;
}
} // namespace

Tensor execute(const Json::Value &node,const Values &values,Stats &stats){
  auto prepared=prepare(node,values);++stats.calls;stats.v2|=extended_kind(prepared.kind);
  auto output=prepared.output;if(prepared.view){++stats.views;return output;}
  const auto &p=node["memory_program"],&args=node["args"];
  const auto &kind=prepared.kind,&selector=prepared.selector;
  const auto target=output.type;
  const auto slot_bytes=prepared.slot_bytes,concat_dim=prepared.concat_dim;
  const auto inner=prepared.inner;const auto &concatenated=prepared.concatenated;
  std::array<Value,4> registers{};std::array<uint8_t,128> staging{};
  const unsigned chunk_elements=64/slot_bytes;check(chunk_elements>0,"invalid memory transfer slot width");
  auto read=[&](const Tensor &input,uint64_t index,unsigned slot){Value value;value.type=input.type;value.valid=true;auto count=element_bytes(input.type);std::memcpy(staging.data()+slot*slot_bytes,input.storage->data+input.position(index)*count,count);std::memcpy(&value.bits,staging.data()+slot*slot_bytes,count);stats.read_bytes+=count;return value;};
  auto operand=[&](const Json::Value &arg,uint64_t flat,unsigned slot){if(arg.isObject()&&arg.isMember("value")){const auto &input=ref(arg,values);return read(input,broadcast(flat,output.sizes,input.sizes),slot);}return literal(arg);};
  check(std::fegetround()==FE_TONEAREST,"memory conversions require round-to-nearest-even");++stats.allocations;
  if(kind=="embedding"&&ref(args[0],values).sizes[1]==0){
    bool has_index_load=false;for(const auto &word:p["words"])has_index_load|=word.asUInt()==4;
    check(has_index_load,"empty embedding still requires index validation");const auto &weight=ref(args[0],values),&ids=ref(args[1],values);
    for(uint64_t index=0;index<ids.numel();++index){auto value=read(ids,index,unsigned(index%chunk_elements));auto row=integer_bits(value.bits);check(row>=0&&row<weight.sizes[0],"embedding index out of range");++stats.instructions;++stats.opcode_counts[4];++stats.index_reads;}
  }
  for(uint64_t flat=0;flat<output.numel();++flat){unsigned slot=flat%chunk_elements,index_axis=0;registers[0].valid=registers[1].valid=false;bool stored=false;
    for(const auto &raw:p["words"]){check(raw.isUInt(),"invalid memory instruction word");unsigned word=raw.asUInt(),op=word&255;check(op>=1&&op<=5&&(word>>10)==0&&(op==2||(word>>8)==0),"memory opcode/reserved field violation");++stats.instructions;++stats.opcode_counts[op];
      if(op==4){
        if(selector=="indexed_nd"){check(index_axis<args[1].size(),"advanced index instruction axis mismatch");const auto &ids=ref(args[1][index_axis],values);registers[2]=read(ids,broadcast(flat,output.sizes,ids.sizes),slot);append_index(registers[3],ref(args[0],values),index_axis,integer_bits(registers[2].bits));++index_axis;}
        else{check(selector=="indexed_rows","index load used outside embedding");const auto &weight=ref(args[0],values),&ids=ref(args[1],values);check(weight.sizes[1]>0,"empty embedding row issued a load");uint64_t index=flat/uint64_t(weight.sizes[1]);registers[2]=read(ids,index,slot);auto row=integer_bits(registers[2].bits);check(row>=0&&row<weight.sizes[0],"embedding index out of range");}
        ++stats.index_reads;
      }
      else if(op==5){check(selector=="predicate_select","predicate load used outside selection");registers[2]=operand(args[0],flat,slot);check(registers[2].type==DType::Bool,"predicate has wrong type");++stats.predicate_reads;}
      else if(op==1){
        if(selector=="constant_one")registers[0]=literal(Json::Value(1));
        else if(selector=="indexed_nd"){check(registers[3].valid&&index_axis==ref(args[0],values).sizes.size(),"advanced index lacks a complete address");registers[0]=read(ref(args[0],values),registers[3].bits,slot);}
        else if(selector=="linear")registers[0]=read(ref(args[0],values),flat,slot);
        else if(selector=="indexed_rows"){check(registers[2].valid&&registers[2].type==DType::I64,"embedding address lacks a loaded index");const auto &weight=ref(args[0],values);registers[0]=read(weight,uint64_t(integer_bits(registers[2].bits))*weight.sizes[1]+flat%uint64_t(weight.sizes[1]),slot);}
        else if(selector=="predicate_select"){check(registers[2].valid&&registers[2].type==DType::Bool,"selection lacks a loaded predicate");registers[0]=operand(args[registers[2].bits?1:2],flat,slot);}
        else{uint64_t width=uint64_t(output.sizes[concat_dim])*inner,outer=flat/width,index=flat%width;bool found=false;for(const auto *input:concatenated){uint64_t count=uint64_t(input->sizes[concat_dim])*inner;if(index<count){registers[0]=read(*input,outer*count+index,slot);found=true;break;}index-=count;}check(found,"concat transfer did not cover an output");}
      }else if(op==2){check(((word>>8)&3)==unsigned(target),"conversion word target mismatch");registers[1]=convert(registers[0],target);}
      else{check(registers[1].valid&&registers[1].type==target&&!stored,"store reads unready register or duplicates an output");unsigned count=element_bytes(target);std::memcpy(staging.data()+64+slot*slot_bytes,&registers[1].bits,count);std::memcpy(output.storage->writable+output.position(flat)*count,staging.data()+64+slot*slot_bytes,count);stats.write_bytes+=count;stored=true;}
    }check(stored,"memory transfer omitted output store");
  }return output;
}

void ScheduleOptions::validate()const{
  check(dma_latency&&convert_latency&&request_period&&response_period&&max_cycles,"memory timing values must be positive");
  check(max_cycles<UINT64_MAX-std::max(dma_latency,convert_latency),"memory cycle bound overflows latency");
  check(trace_limit<=1000000,"memory trace limit exceeds bound");
}
ScheduleOptions ScheduleOptions::parse(const Json::Value &value){
  check(value.isObject(),"memory schedule options must be an object");ScheduleOptions out;
  const std::set<std::string> allowed={"dma_latency","convert_latency","request_period","response_period","trace_limit","max_cycles","trace"};
  for(const auto &name:value.getMemberNames())check(allowed.count(name),"unknown memory schedule option");
  auto read=[&](const char *name,unsigned &target){if(value.isMember(name)){check(value[name].isUInt(),"memory timing option must be unsigned");target=value[name].asUInt();}};
  read("dma_latency",out.dma_latency);read("convert_latency",out.convert_latency);read("request_period",out.request_period);read("response_period",out.response_period);read("trace_limit",out.trace_limit);
  if(value.isMember("max_cycles")){check(value["max_cycles"].isUInt64(),"invalid memory cycle bound");out.max_cycles=value["max_cycles"].asUInt64();}
  if(value.isMember("trace")){check(value["trace"].isBool(),"memory trace option must be Boolean");out.trace=value["trace"].asBool();}
  out.validate();return out;
}

struct Simulator::Impl {
  Json::Value node;
  Values values;
  ScheduleOptions options;
  Prepared plan;
  std::unique_ptr<model_io::TensorMemoryPort> local_port;
  model_io::MemoryPort *port;
  bool external=false,complete=false,failed=false;
  Stats stats;
  uint64_t cycle=0,flat=0,total=0,next_id=0,requests=0,responses=0;
  uint64_t request_stalls=0,response_stalls=0,conversion_stalls=0,trace_events=0;
  unsigned pc=0;
  std::array<Value,4> registers{};
  std::array<uint8_t,128> staging{};
  struct Pending {model_io::Request request;unsigned op;DType type;uint64_t issue_cycle;};
  std::optional<Pending> pending;
  std::optional<model_io::Response> held;
  struct Conversion {Value value;uint64_t due;};
  std::optional<Conversion> conversion;
  Json::Value events{Json::arrayValue};

  Impl(const Json::Value &n,const Values &v,ScheduleOptions o,model_io::MemoryPort *p,const Tensor *output)
      :node(n),options(o),port(p),external(p!=nullptr){
    options.validate();
    // Hold only this operator's inputs, including their storage ownership.
    for(const auto &name:n["memory_program"]["input_layouts"].getMemberNames()){
      auto it=v.find(name);check(it!=v.end(),"missing planned memory operand");values.emplace(name,it->second);
    }
    check(!external||n["memory_program"]["mode"]=="view"||output,"external memory transfer requires output metadata");
    plan=prepare(node,values,output);
    check(plan.names.size()<64,"memory region table exceeds 64 entries");
    stats.calls=1;stats.views=plan.view;stats.allocations=!plan.view;stats.v2=extended_kind(plan.kind);
    total=plan.empty_embedding?ref(node["args"][1],values).numel():plan.output.numel();
    complete=plan.view||total==0;
    if(!port&&!plan.view){std::vector<Tensor> regions;for(const auto &name:plan.names)regions.push_back(values.at(name));regions.push_back(plan.output);local_port=std::make_unique<model_io::TensorMemoryPort>(std::move(regions),options.dma_latency);port=local_port.get();}
  }
  unsigned slot()const{return unsigned(flat%(64/plan.slot_bytes))*plan.slot_bytes;}
  void event(const char *kind,const model_io::Request *request=nullptr,uint64_t data=0){
    ++trace_events;if(!options.trace||events.size()>=options.trace_limit)return;
    Json::Value e;e["event"]=kind;e["cycle"]=Json::UInt64(cycle);e["element"]=Json::UInt64(flat);e["pc"]=pc;
    if(request){e["request_id"]=Json::UInt64(request->id);e["region"]=request->region;e["offset"]=Json::UInt64(request->offset);e["bytes"]=request->bytes;e["write"]=request->write;e["data"]=Json::UInt64(data);}
    events.append(e);
  }
  void issued(unsigned op){++stats.instructions;++stats.opcode_counts[op];}
  void retire(){
    ++flat;pc=0;registers={};complete=flat==total;
  }
  void next(){
    ++pc;if(plan.empty_embedding||pc==node["memory_program"]["words"].size())retire();
  }
  unsigned region(const Tensor &tensor)const{
    for(unsigned i=0;i<plan.names.size();++i)if(&values.at(plan.names[i])==&tensor)return i;
    throw std::runtime_error("unbound memory transfer region");
  }
  bool can_request(){
    if(cycle%options.request_period||!port->request_ready()){++request_stalls;return false;}return true;
  }
  void submit(const Tensor &tensor,uint64_t index,unsigned op,bool write=false){
    check(next_id!=UINT64_MAX,"memory request token exhausted");
    model_io::Request request;request.id=next_id++;request.bytes=element_bytes(tensor.type);request.offset=tensor.position(index)*request.bytes;
    request.region=write?unsigned(plan.names.size()):region(tensor);request.write=write;
    if(write){std::memcpy(staging.data()+64+slot(),&registers[1].bits,request.bytes);std::memcpy(&request.data,staging.data()+64+slot(),request.bytes);}
    port->submit(request);pending=Pending{request,op,tensor.type,cycle};++requests;issued(op);event("request",&request,request.data);
  }
  void load_operand(const Json::Value &arg,unsigned op){
    if(arg.isObject()&&arg.isMember("value")){
      if(!can_request())return;
      const auto &input=ref(arg,values);submit(input,broadcast(flat,plan.output.sizes,input.sizes),op);
    }else{check(op==1,"predicate cannot be a descriptor literal");registers[0]=literal(arg);issued(op);event("literal");next();}
  }
  void consume(const model_io::Response &response){
    check(pending&&response.id==pending->request.id,"memory response owner mismatch");
    check(!response.error,"memory port reported error");
    check(cycle>pending->issue_cycle,"memory response consumed on request issue cycle");
    const auto &request=pending->request;unsigned op=pending->op;
    if(request.write)stats.write_bytes+=request.bytes;
    else{
      Value value;value.type=pending->type;value.valid=true;
      std::memcpy(staging.data()+slot(),&response.data,request.bytes);std::memcpy(&value.bits,staging.data()+slot(),request.bytes);
      registers[op==1?0:2]=value;stats.read_bytes+=request.bytes;
      if(op==4){auto row=integer_bits(value.bits);if(plan.selector=="indexed_nd")append_index(registers[3],ref(node["args"][0],values),pc,row);else check(row>=0&&row<ref(node["args"][0],values).sizes[0],"embedding index out of range");++stats.index_reads;}
      if(op==5){check(value.type==DType::Bool,"predicate has wrong type");++stats.predicate_reads;}
    }
    event("response",&request,request.write?request.data:response.data);
    port->consume_response();pending.reset();held.reset();++responses;next();
  }
  void step(){
    check(!failed,"memory simulator is faulted");check(cycle<options.max_cycles,"memory schedule exceeded max_cycles");
    port->advance(cycle);auto response=port->response();
    if(held)check(response&&response->id==held->id&&response->data==held->data&&response->error==held->error,"memory response changed or withdrew under backpressure");
    if(response){
      check(pending&&response->id==pending->request.id,"memory response owner mismatch");check(!response->error,"memory port reported error");held=response;
    }
    if(pending){
      if(response&&cycle%options.response_period==0)consume(*response);
      else ++response_stalls;
    }else if(conversion){
      if(cycle>=conversion->due){registers[1]=conversion->value;conversion.reset();event("convert_complete");next();}
      else ++conversion_stalls;
    }else{
      const auto &args=node["args"];unsigned word=plan.empty_embedding?4:node["memory_program"]["words"][pc].asUInt(),op=word&255;
      if(op==4){if(can_request()){
        if(plan.selector=="indexed_nd"){check(pc<args[1].size(),"advanced index instruction axis mismatch");const auto &ids=ref(args[1][pc],values);submit(ids,broadcast(flat,plan.output.sizes,ids.sizes),op);}
        else{const auto &ids=ref(args[1],values);auto width=ref(args[0],values).sizes[1];submit(ids,plan.empty_embedding?flat:flat/uint64_t(width),op);}}}
      else if(op==5)load_operand(args[0],op);
      else if(op==1){
        if(plan.selector=="constant_one")load_operand(Json::Value(1),op);
        else if(plan.selector=="predicate_select"){check(registers[2].valid,"selection lacks a loaded predicate");load_operand(args[registers[2].bits?1:2],op);}
        else if(can_request()){
          if(plan.selector=="indexed_nd"){check(registers[3].valid,"advanced index lacks a complete address");submit(ref(args[0],values),registers[3].bits,op);}
          else if(plan.selector=="linear")submit(ref(args[0],values),flat,op);
          else if(plan.selector=="indexed_rows"){
            check(registers[2].valid,"embedding address lacks a loaded index");const auto &weight=ref(args[0],values);
            submit(weight,uint64_t(integer_bits(registers[2].bits))*weight.sizes[1]+flat%uint64_t(weight.sizes[1]),op);
          }else{
            auto width=uint64_t(plan.output.sizes[plan.concat_dim])*plan.inner,outer=flat/width,index=flat%width;bool found=false;
            for(const auto *input:plan.concatenated){auto count=uint64_t(input->sizes[plan.concat_dim])*plan.inner;if(index<count){submit(*input,outer*count+index,op);found=true;break;}index-=count;}
            check(found,"concat transfer did not cover an output");
          }
        }
      }else if(op==2){
        conversion=Conversion{convert(registers[0],plan.output.type),cycle+options.convert_latency};issued(op);event("convert_issue");
      }else{
        check(op==3&&registers[1].valid&&registers[1].type==plan.output.type,"store reads unready transfer register");
        if(can_request())submit(plan.output,flat,op,true);
      }
    }
    ++cycle;
  }
  Json::Value result()const{
    check(complete&&!failed,"memory schedule result requested before completion");
    check(!pending&&!conversion&&!held&&requests==responses,"memory schedule completed without draining");
    Json::Value r;r["classification"]="memory_transfer_cycle_component_not_system_validation";r["done"]=true;r["cycles"]=Json::UInt64(cycle);
    r["external_memory_port"]=external;r["dma_requests"]=Json::UInt64(requests);r["dma_responses"]=Json::UInt64(responses);
    r["dma_read_bytes"]=Json::UInt64(stats.read_bytes);r["dma_write_bytes"]=Json::UInt64(stats.write_bytes);
    r["request_stalls"]=Json::UInt64(request_stalls);r["response_stalls"]=Json::UInt64(response_stalls);r["conversion_stalls"]=Json::UInt64(conversion_stalls);
    r["numeric_instructions"]=stats.json();r["trace"]=events;r["trace_events"]=Json::UInt64(trace_events);r["trace_truncated"]=options.trace&&trace_events>events.size();
    r["resources"]["data_register_bytes"]=32;r["resources"]["staging_bytes"]=128;
    r["resources"]["conversion_result_latch_bytes"]=8;r["resources"]["request_data_latch_bytes"]=8;
    r["resources"]["max_inflight_transactions"]=1;r["resources"]["max_inflight_conversions"]=1;
    r["options"]["dma_latency"]=options.dma_latency;r["options"]["convert_latency"]=options.convert_latency;
    r["options"]["request_period"]=options.request_period;r["options"]["response_period"]=options.response_period;
    r["options"]["max_cycles"]=Json::UInt64(options.max_cycles);r["options"]["trace"]=options.trace;r["options"]["trace_limit"]=options.trace_limit;
    r["dma_latency_source"]=external?"external_port_responses":"standalone_tensor_endpoint";
    r["region_names"]=Json::Value(Json::arrayValue);for(const auto &name:plan.names)r["region_names"].append(name);
    if(!plan.view)r["region_names"].append(node["id"]);
    r["view_elided"]=plan.view;r["inflight_transactions"]=0;r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;
    return r;
  }
};

Simulator::Simulator(const Json::Value &node,const Values &values,ScheduleOptions options,model_io::MemoryPort *port,const Tensor *output)
    :impl(std::make_unique<Impl>(node,values,options,port,output)){}
Simulator::~Simulator()=default;
bool Simulator::tick(){if(impl->complete)return false;try{impl->step();}catch(...){impl->failed=true;throw;}return !impl->complete;}
bool Simulator::done()const{return impl->complete&&!impl->failed;}
Tensor Simulator::output()const{check(done(),"memory output requested before completion");return impl->plan.output;}
Stats Simulator::instruction_stats()const{check(done(),"memory stats requested before completion");return impl->stats;}
Json::Value Simulator::result()const{return impl->result();}
} // namespace mlx::memory_model
