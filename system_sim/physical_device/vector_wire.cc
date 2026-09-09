#include "vector_wire.hh"
#include <cmath>
#include <cstring>
#include <set>

namespace mlx::physical_device {
using namespace tensor_model;
namespace {
const char *kinds[]={"","add","mul","pow","rsqrt","silu","cos","sin","neg","mean","softmax","sub","div","maximum","exp"};
const char *phases[]={"body","to_carry","save_carry","root","merge_sum","sum_tile","final","max_tile","merge_max","save_max","save_sum","output_tile"};
bool zero(const void *value,size_t bytes){const auto *p=static_cast<const uint8_t*>(value);for(size_t i=0;i<bytes;++i)if(p[i])return false;return true;}
Json::Value number(uint64_t bits){
  double value;std::memcpy(&value,&bits,8);if(std::isfinite(value))return Json::Value(value);
  Json::Value out;out["float_literal"]=std::isnan(value)?"nan":std::signbit(value)?"-inf":"inf";return out;
}
}
DecodedVector decode_vector(const mlx_vector_wire &wire){
  require(wire.magic==MLX_VECTOR_WIRE_MAGIC&&((wire.version==1&&wire.kind>=1&&wire.kind<=10)||(wire.version==2&&wire.kind>=11&&wire.kind<=14))&&!(wire.flags&~UINT64_C(3))&&zero(wire.reserved,sizeof(wire.reserved))&&zero(wire.tail_reserved,sizeof(wire.tail_reserved)),"vector wire header/reserved fields invalid");
  bool reduction=wire.kind==MLX_VECTOR_MEAN||wire.kind==MLX_VECTOR_SOFTMAX;bool mean=wire.kind==MLX_VECTOR_MEAN;
  require(mean||!(wire.flags&MLX_VECTOR_KEEP_DIM),"vector wire keepdim outside mean");
  require(reduction?(wire.width>0&&wire.width<=UINT64_C(0x80000000)):wire.width==0,"vector wire reduction width invalid");
  require(wire.rom_count>0&&wire.rom_count<=32&&wire.constant_count<=16&&wire.phase_count<=12,"vector wire table capacity invalid");
  DecodedVector result;auto &node=result.node;auto &p=node["vector_program"];
  node["kind"]=kinds[wire.kind];node["id"]="output";node["kwargs"]=Json::Value(Json::objectValue);node["args"]=Json::Value(Json::arrayValue);
  p["profile"]="mlx-vector-fp32-v1";p["kind"]=node["kind"];p["lanes"]=16;p["trans_lanes"]=4;p["rf_vectors"]=16;p["rf_vector_bytes"]=64;p["rf_vectors_used"]=8;p["spm_bytes"]=8192;p["spm_bytes_used"]=320;p["rom_words"]=32;
  p["width"]=reduction?Json::Value(Json::UInt64(wire.width)):Json::Value();p["reduction"]=reduction?Json::Value("adjacent_pairwise_zero_padding_binary_carry"):Json::Value();p["timing_verified"]=false;p["system_verified"]=false;
  result.output=decode_float_tensor(wire.output,true);node["output"]["dtype"]=dtype_name(result.output.type);node["output"]["shape"]=Json::Value(Json::arrayValue);
  for(auto n:result.output.sizes)node["output"]["shape"].append(Json::Int64(n));
  p["output_dtype"]=node["output"]["dtype"];
  p["input_dtypes"]=Json::Value(Json::arrayValue);unsigned count=wire.kind==MLX_VECTOR_ADD||wire.kind==MLX_VECTOR_MUL||wire.kind==MLX_VECTOR_SUB||wire.kind==MLX_VECTOR_DIV||wire.kind==MLX_VECTOR_MAXIMUM?2:1;
  const mlx_host_operand *operands[]={&wire.a,&wire.b};
  for(unsigned i=0;i<2;++i){
    const auto &arg=*operands[i];const char *name=i?"b":"a";
    if(i>=count){require(zero(&arg,sizeof(arg)),"unused vector operand is nonzero");result.regions.push_back({0,0,false,false});continue;}
    require(!arg.reserved,"vector operand reserved field is nonzero");
    if(arg.kind==MLX_HOST_TENSOR){
      require(!arg.scalar_dtype&&!arg.scalar_bits,"vector tensor has scalar fields");auto tensor=decode_float_tensor(arg.tensor,false);p["input_dtypes"].append(dtype_name(tensor.type));result.values[name]=std::move(tensor);
      Json::Value reference;reference["value"]=name;node["args"].append(reference);result.regions.push_back({arg.tensor.base,arg.tensor.bytes,true,false});
    }else{
      require(arg.kind==MLX_HOST_SCALAR&&arg.scalar_dtype==MLX_VECTOR_LITERAL_F64&&zero(&arg.tensor,sizeof(arg.tensor)),"vector immediate encoding invalid");
      p["input_dtypes"].append("f32");node["args"].append(number(arg.scalar_bits));result.regions.push_back({0,0,false,false});
    }
  }
  result.regions.push_back({wire.output.base,wire.output.bytes,false,true});
  if(!reduction){
    Shape expected;
    for(const auto &[name,t]:result.values){(void)name;if(expected.size()<t.sizes.size())expected.insert(expected.begin(),t.sizes.size()-expected.size(),1);auto shift=expected.size()-t.sizes.size();
      for(size_t d=0;d<t.sizes.size();++d){auto n=t.sizes[d];auto &current=expected[shift+d];if(current==1)current=n;else require(n==1||n==current,"vector wire operand broadcast mismatch");}
    }
    require(expected==result.output.sizes,"vector wire broadcast output shape mismatch");
  }
  if(wire.kind==MLX_VECTOR_POW2)node["args"].append(2);
  if(mean){Json::Value axis(Json::arrayValue);axis.append(-1);node["args"].append(axis);node["args"].append(bool(wire.flags&MLX_VECTOR_KEEP_DIM));}
  if(wire.kind==MLX_VECTOR_SOFTMAX){node["args"].append(-1);node["args"].append(result.output.type==DType::F16?"torch.float16":"torch.float32");}
  require(bool(wire.flags&MLX_VECTOR_NARROW_INPUT)==(wire.kind==MLX_VECTOR_SOFTMAX&&p["input_dtypes"][0]=="f32"&&result.output.type==DType::F16),"vector wire narrowing flag mismatch");
  if(wire.flags&MLX_VECTOR_NARROW_INPUT){
    require(wire.kind==MLX_VECTOR_SOFTMAX&&p["input_dtypes"][0]=="f32"&&result.output.type==DType::F16,"vector wire narrowing flag has incompatible dtype/kind");p["softmax_input_cast"]="f32_to_f16_before_reduction";
  }
  p["rom"]=Json::Value(Json::arrayValue);p["constants"]=Json::Value(Json::arrayValue);p["phases"]=Json::Value(Json::objectValue);
  for(unsigned i=0;i<32;++i){require(wire.rom[i]<=UINT32_MAX,"vector wire ROM word is not 32-bit");if(i<wire.rom_count)p["rom"].append(Json::UInt(wire.rom[i]));else require(!wire.rom[i],"unused vector ROM slot is nonzero");}
  for(unsigned i=0;i<16;++i){if(i<wire.constant_count)p["constants"].append(number(wire.constants[i]));else require(!wire.constants[i],"unused vector constant slot is nonzero");}
  std::set<std::string> expected;
  if(!reduction)expected={"body"};else if(mean)expected={"to_carry","save_carry","root","merge_sum","sum_tile","final"};
  else expected={"to_carry","save_carry","root","merge_sum","sum_tile","max_tile","merge_max","save_max","save_sum","output_tile"};
  unsigned present=0;
  for(unsigned i=0;i<12;++i){
    auto length=wire.phase_lengths[i];require(length<=32&&bool(length)==bool(expected.count(phases[i])),"vector wire phase set/length invalid");if(length){++present;p["phases"][phases[i]]=Json::Value(Json::arrayValue);}
    for(unsigned j=0;j<32;++j){auto index=wire.phase_indices[i][j];if(j<length){require(index<wire.rom_count,"vector wire phase index outside ROM");p["phases"][phases[i]].append(Json::UInt(index));}else require(index==0,"unused vector phase index is nonzero");}
  }
  require(present==wire.phase_count,"vector wire phase count mismatch");return result;
}
} // namespace mlx::physical_device
