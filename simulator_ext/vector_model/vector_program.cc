#include "vector_program.h"
#include "mlx_tagged_simulator.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace mlx::vector_model {
using namespace tensor_model;
namespace {
void check(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
float floating(uint32_t bits){float value;std::memcpy(&value,&bits,4);return value;}
uint32_t bits(float value){uint32_t out;std::memcpy(&out,&value,4);return out;}
enum Op : unsigned {LoadA16=2,LoadB16=3,CvtUp=4,Mul=5,Add=6,CvtDown=7,Store16=8,
                    LoadA32=9,LoadB32=10,Store32=11,Constant=14,Neg=15,Sub=16,Exp=17,
                    Div=18,Sqrt=19,Cos=20,Sin=21,Shuffle=22,Max=23,Move=24,Broadcast=25,
                    LoadStack=26,StoreStack=27};
struct Instruction{unsigned op,dst,a,b,imm;};
struct Register{std::array<uint32_t,16> data{};unsigned type=0,valid=0;};
struct Operand{const Tensor *tensor=nullptr;float immediate=0;DType type=DType::F32;};
uint64_t broadcast_index(uint64_t index,const Shape &out,const Shape &in){
  check(in.size()<=out.size(),"vector operand broadcast rank mismatch");uint64_t result=0,step=1;
  for(size_t d=out.size();d-->0;){uint64_t at=out[d]?index%out[d]:0;if(out[d])index/=out[d];
    if(d+in.size()>=out.size()){auto size=in[d+in.size()-out.size()];check(size==1||size==out[d],"vector operand broadcast extent mismatch");if(size!=1)result+=at*step;step*=size;}}
  return result;
}
struct Machine {
  const Json::Value &node,&program;
  Stats &stats;
  std::vector<Operand> operands;
  std::vector<Instruction> rom;
  std::array<Register,16> rf{};
  std::array<uint8_t,8192> spm{};
  Tensor output;
  Shape source_shape;
  uint64_t base=0,valid_count=0,output_base=0,width=0;
  unsigned lanes=0,level=0,stack_valid=0;
  bool reduction=false;

  Machine(const Json::Value &n,const Values &values,Stats &s):node(n),program(n["vector_program"]),stats(s) {
    check(program["profile"]=="mlx-vector-fp32-v1"&&program["kind"]==node["kind"],"unsupported/mismatched vector microcode profile");
    check(supports(node["kind"].asString()),"vector microcode attached to unsupported tensor operation");
    check(program["lanes"]==16&&program["trans_lanes"]==4&&program["rf_vectors"]==16&&program["rf_vector_bytes"]==64&&program["rf_vectors_used"]==8&&program["spm_bytes"]==8192&&program["spm_bytes_used"]==320&&program["rom_words"]==32,"vector resource contract mismatch");
    check(program["rom"].isArray()&&!program["rom"].empty()&&program["rom"].size()<=32,"vector ROM capacity violation");
    for(const auto &value:program["rom"]){check(value.isUInt(),"invalid vector instruction word");unsigned word=value.asUInt();Instruction i{word&255,(word>>8)&15,(word>>12)&15,(word>>16)&15,(word>>20)&15};
      check(!(word>>24)&&i.op>=2&&i.op<=27&&i.op!=12&&i.op!=13&&i.dst<8&&i.a<8&&i.b<8,"vector opcode/field violation");
      if(i.op==Mul||i.op==Add||i.op==Sub||i.op==Div||i.op==Max){}
      else if(i.op==LoadA16||i.op==LoadB16||i.op==LoadA32||i.op==LoadB32||i.op==Constant||i.op==LoadStack)check(i.a==0&&i.b==0,"noncanonical vector load fields");
      else if(i.op==Store16||i.op==Store32||i.op==StoreStack)check(i.dst==0&&i.b==0,"noncanonical vector store fields");
      else check(i.b==0,"noncanonical vector unary fields");
      if(i.op==Shuffle)check(i.imm==1||i.imm==2||i.imm==4||i.imm==8,"invalid shuffle distance");
      else if(i.op>=Exp&&i.op<=Sin)check(i.imm<4,"transcendental lane group exceeds quarter width");
      else if(i.op==LoadA16||i.op==LoadA32||i.op==LoadB16||i.op==LoadB32)check(i.imm<=1,"unknown masked-load padding mode");
      else if(i.op!=Constant)check(i.imm==0,"noncanonical vector immediate");
      rom.push_back(i);
    }
    check(program["phases"].isObject(),"missing vector phases");
    for(const auto &name:program["phases"].getMemberNames())for(const auto &index:program["phases"][name])check(index.isUInt()&&index.asUInt()<rom.size(),"vector phase points outside ROM");
    check(program["input_dtypes"].isArray()&&program["input_dtypes"].size()==(node["kind"]=="add"||node["kind"]=="sub"||node["kind"]=="mul"?2u:1u),"vector operand arity mismatch");
    if(node["kind"]=="pow")check(node["args"].size()==2&&scalar(node["args"][1])==2,"vector pow requires exponent 2");
    for(Json::ArrayIndex i=0;i<program["input_dtypes"].size();++i){
      check(i<node["args"].size(),"missing vector operand");const auto &arg=node["args"][i];Operand operand;
      if(arg.isObject()&&arg.isMember("value")){auto found=values.find(arg["value"].asString());check(found!=values.end(),"unbound vector operand");operand.tensor=&found->second;operand.type=found->second.type;}
      else operand.immediate=float(scalar(arg));
      check((operand.type==DType::F16||operand.type==DType::F32)&&program["input_dtypes"][i]==dtype_name(operand.type),"vector operand dtype contract mismatch");operands.push_back(operand);
    }
    output=Tensor::allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));
    check(program["output_dtype"]==dtype_name(output.type)&&(output.type==DType::F16||output.type==DType::F32),"vector output precision mismatch");
    reduction=node["kind"]=="mean"||node["kind"]=="softmax";
    if(reduction){check(operands.size()==1&&operands[0].tensor&&!operands[0].tensor->sizes.empty(),"reduction requires a nonscalar tensor");
      source_shape=operands[0].tensor->sizes;width=source_shape.back();check(width>0&&width<=(1ULL<<31)&&program["width"].asUInt64()==width,"vector reduction width mismatch");
      auto expected=source_shape;
      if(node["kind"]=="mean"){check(node["args"][1].isArray()&&node["args"][1].size()==1,"mean requires one axis");auto axis=node["args"][1][0].asInt64();check(axis==-1||axis==int64_t(source_shape.size()-1),"mean requires the last axis");if(node["args"].size()>2&&node["args"][2].asBool())expected.back()=1;else expected.pop_back();}
      else{auto axis=node["args"][1].asInt64();check(axis==-1||axis==int64_t(source_shape.size()-1),"softmax requires the last axis");}
      check(output.sizes==expected,"vector reduction output shape mismatch");
    }else source_shape=output.sizes;
    check(std::numeric_limits<float>::is_iec559&&sizeof(float)==4&&std::fegetround()==FE_TONEAREST,"vector arithmetic requires IEEE FP32 RNE");
#if defined(__SSE__)
    check((_mm_getcsr()&((1u<<15)|(1u<<6)))==0,"vector arithmetic forbids FTZ/DAZ");
#endif
    ++stats.calls;stats.max_rom=std::max(stats.max_rom,unsigned(rom.size()));
  }
  void need(unsigned r,unsigned type,unsigned mask)const{check(rf[r].type==type&&(rf[r].valid&mask)==mask,"invalid vector register type/readiness");}
  void write(unsigned r,unsigned type,unsigned mask){if(rf[r].type!=type)rf[r].valid=0;rf[r].type=type;rf[r].valid|=mask;}
  void phase(const char *name){check(program["phases"].isMember(name),"missing required vector phase");for(const auto &index:program["phases"][name])issue(rom[index.asUInt()]);}
  void issue(const Instruction &i){
    ++stats.instructions;++stats.opcode_counts[i.op];
    unsigned first=0,last=lanes;
    if(i.op>=Exp&&i.op<=Sin){first=i.imm*4;last=std::min(lanes,first+4);if(first>=last)return;stats.trans_lanes+=last-first;}
    const unsigned mask=last>first?((1u<<(last-first))-1)<<first:0;
    if(i.op==LoadA16||i.op==LoadB16||i.op==LoadA32||i.op==LoadB32){
      unsigned operand=(i.op==LoadB16||i.op==LoadB32)?1:0;check(operand<operands.size(),"vector load operand out of bounds");const auto &a=operands[operand];
      unsigned type=(i.op==LoadA16||i.op==LoadB16)?1:2;check(type==(a.type==DType::F16?1u:2u),"vector load precision mismatch");
      for(unsigned lane=0;lane<lanes;++lane){float value=i.imm?-std::numeric_limits<float>::infinity():0.0f;
        if(lane<valid_count){if(a.tensor){uint64_t index=reduction?base+lane:broadcast_index(base+lane,source_shape,a.tensor->sizes);value=a.tensor->number(index);stats.read_bytes+=element_bytes(a.type);}else value=a.immediate;}
        unsigned bytes=type==1?2:4;uint32_t raw=type==1?uint32_t(tagged::float_to_half(value)):bits(value);
        std::memcpy(spm.data()+operand*64+lane*bytes,&raw,bytes);rf[i.dst].data[lane]=0;std::memcpy(&rf[i.dst].data[lane],spm.data()+operand*64+lane*bytes,bytes);
      }write(i.dst,type,mask);return;
    }
    if(i.op==Constant){check(i.imm<program["constants"].size(),"constant pool index out of range");uint32_t raw=bits(float(scalar(program["constants"][i.imm])));for(unsigned lane=0;lane<lanes;++lane)rf[i.dst].data[lane]=raw;write(i.dst,2,mask);return;}
    if(i.op==LoadStack||i.op==StoreStack){check(level<32,"bounded reduction stack overflow");stats.max_stack_level=std::max(stats.max_stack_level,level);
      if(i.op==StoreStack){need(i.a,2,1);std::memcpy(spm.data()+128+level*4,&rf[i.a].data[0],4);stack_valid|=1u<<level;}
      else{check(stack_valid&(1u<<level),"read of uninitialized reduction partial");std::memcpy(&rf[i.dst].data[0],spm.data()+128+level*4,4);write(i.dst,2,1);}return;
    }
    if(i.op==Move||i.op==Broadcast){need(i.a,2,1);uint32_t raw=rf[i.a].data[0];if(i.op==Move){rf[i.dst].data[0]=raw;write(i.dst,2,1);}else{for(unsigned lane=0;lane<lanes;++lane)rf[i.dst].data[lane]=raw;write(i.dst,2,mask);}return;}
    if(i.op==Store16||i.op==Store32){unsigned type=i.op==Store16?1:2,bytes=type==1?2:4;check(output.type==(type==1?DType::F16:DType::F32),"vector store precision mismatch");need(i.a,type,mask);
      for(unsigned lane=0;lane<valid_count;++lane){check(output_base+lane<output.numel(),"vector store outside output");std::memcpy(spm.data()+256+lane*bytes,&rf[i.a].data[lane],bytes);std::memcpy(output.storage->writable+(output_base+lane)*bytes,spm.data()+256+lane*bytes,bytes);stats.write_bytes+=bytes;}return;
    }
    if(i.op==CvtUp||i.op==CvtDown){need(i.a,i.op==CvtUp?1:2,mask);for(unsigned lane=first;lane<last;++lane)rf[i.dst].data[lane]=i.op==CvtUp?bits(tagged::half_to_float(uint16_t(rf[i.a].data[lane]))):tagged::float_to_half(floating(rf[i.a].data[lane]));write(i.dst,i.op==CvtUp?2:1,mask);return;}
    need(i.a,2,mask);
    if(i.op==Shuffle){check(lanes&&(lanes&(lanes-1))==0&&i.imm<lanes,"shuffle crosses inactive reduction lanes");auto source=rf[i.a].data;for(unsigned lane=0;lane<lanes;++lane)rf[i.dst].data[lane]=source[lane^i.imm];write(i.dst,2,mask);return;}
    if(i.op==Add||i.op==Mul||i.op==Sub||i.op==Div||i.op==Max)need(i.b,2,mask);
    for(unsigned lane=first;lane<last;++lane){float a=floating(rf[i.a].data[lane]),b=floating(rf[i.b].data[lane]),result=0;
      switch(i.op){case Add:result=a+b;break;case Mul:result=a*b;break;case Sub:result=a-b;break;case Div:result=a/b;break;case Max:result=a<b?b:a;break;case Neg:result=-a;break;case Exp:result=std::exp(a);break;case Sqrt:result=std::sqrt(a);break;case Cos:result=std::cos(a);break;case Sin:result=std::sin(a);break;default:throw std::runtime_error("unimplemented vector arithmetic opcode");}
      rf[i.dst].data[lane]=bits(result);
    }write(i.dst,2,mask);stats.arithmetic_lanes+=last-first;
  }
  void reduce(uint64_t row,bool maximum){
    uint64_t chunks=(width+15)/16,padded=1;while(padded<chunks)padded*=2;
    unsigned vector_lanes=1;while(vector_lanes<std::min<uint64_t>(16,width))vector_lanes*=2;
    stack_valid=0;
    for(uint64_t tile=0;tile<padded;++tile){base=row*width+tile*16;valid_count=tile*16<width?std::min<uint64_t>(16,width-tile*16):0;lanes=vector_lanes;
      phase(maximum?"max_tile":"sum_tile");lanes=1;phase("to_carry");level=0;
      while(stack_valid&(1u<<level)){phase(maximum?"merge_max":"merge_sum");stack_valid&=~(1u<<level);++level;check(level<32,"reduction carry exceeds bounded stack");}
      phase("save_carry");
    }
    lanes=1;phase("root");
  }
  Tensor run(){
    if(!reduction){for(base=0;base<output.numel();base+=16){rf={};lanes=unsigned(std::min<uint64_t>(16,output.numel()-base));valid_count=lanes;output_base=base;phase("body");}}
    else{uint64_t rows=operands[0].tensor->numel()/width;for(uint64_t row=0;row<rows;++row){rf={};
      if(node["kind"]=="mean"){reduce(row,false);lanes=1;valid_count=1;output_base=row;phase("final");}
      else{reduce(row,true);phase("save_max");reduce(row,false);phase("save_sum");
        for(uint64_t start=0;start<width;start+=16){base=row*width+start;lanes=unsigned(std::min<uint64_t>(16,width-start));valid_count=lanes;output_base=base;phase("output_tile");}}
    }}return output;
  }
};
} // namespace
bool supports(const std::string &kind){return kind=="sub"||kind=="add"||kind=="mul"||kind=="pow"||kind=="rsqrt"||kind=="silu"||kind=="cos"||kind=="sin"||kind=="neg"||kind=="mean"||kind=="softmax";}
Json::Value Stats::json()const{
  Json::Value r(Json::objectValue);r["profile"]="mlx-vector-fp32-v1";r["classification"]="bounded_vector_microcode_functional_not_cycle_or_system_validation";
  r["calls"]=Json::UInt64(calls);r["instructions"]=Json::UInt64(instructions);r["transcendental_lanes"]=Json::UInt64(trans_lanes);r["arithmetic_lanes"]=Json::UInt64(arithmetic_lanes);
  r["global_read_bytes"]=Json::UInt64(read_bytes);r["global_write_bytes"]=Json::UInt64(write_bytes);r["max_rom_words"]=max_rom;r["max_stack_level"]=max_stack_level;r["spm_bytes_used"]=320;r["rf_vectors_used"]=8;
  for(unsigned op=0;op<28;++op)if(opcode_counts[op])r["opcode_counts"][std::to_string(op)]=Json::UInt64(opcode_counts[op]);
  r["timing_verified"]=false;r["system_verified"]=false;return r;
}
Tensor execute(const Json::Value &node,const Values &values,Stats &stats){return Machine(node,values,stats).run();}
} // namespace mlx::vector_model
