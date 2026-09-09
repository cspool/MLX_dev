#include "memory_wire.hh"
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

namespace mlx::physical_device {
using namespace tensor_model;
namespace {
const char *kinds[]={"","embedding","where","cat","cast","cast_device","contiguous","reshape","transpose","slice","select","unsqueeze","expand","alias","dropout_inference","advanced_index","new_ones","squeeze"};
const char *selectors[]={"","linear","concat","indexed_rows","predicate_select","indexed_nd","constant_one"};
bool zero(const void *value,size_t bytes){const auto *p=static_cast<const uint8_t*>(value);for(size_t i=0;i<bytes;++i)if(p[i])return false;return true;}
int64_t signed_bits(uint64_t raw){int64_t value;std::memcpy(&value,&raw,8);return value;}
Json::Value real(uint64_t raw){double value;std::memcpy(&value,&raw,8);if(std::isfinite(value))return value;Json::Value out;out["float_literal"]=std::isnan(value)?"nan":std::signbit(value)?"-inf":"inf";return out;}
std::string name(unsigned index){std::ostringstream out;out<<'a'<<std::setfill('0')<<std::setw(2)<<index;return out.str();}
std::string root(uint64_t id){return "r"+std::to_string(id);}
Json::Value layout(const Tensor &t,const std::string &owner){
  Json::Value p;p["dtype"]=dtype_name(t.type);p["shape"]=Json::Value(Json::arrayValue);p["strides"]=Json::Value(Json::arrayValue);
  for(auto n:t.sizes)p["shape"].append(Json::Int64(n));
  for(auto n:t.steps)p["strides"].append(Json::Int64(n));
  p["offset"]=Json::Int64(t.offset);p["root"]=owner;p["storage_elements"]=Json::UInt64(t.storage->bytes/element_bytes(t.type));return p;
}
const char *torch_dtype(DType type){return type==DType::F16?"torch.float16":type==DType::F32?"torch.float32":type==DType::I64?"torch.int64":"torch.bool";}
}
template<class Wire> DecodedMemory decode_wire(const Wire &wire){
  constexpr bool v2=sizeof(Wire)==sizeof(mlx_memory_wire_v2);
  constexpr unsigned capacity=v2?12:4;
  require(wire.magic==MLX_MEMORY_WIRE_MAGIC&&wire.version==(v2?2:1)&&wire.kind>=(v2?15:1)&&wire.kind<=(v2?17:14)&&wire.mode<=1&&wire.selector>=1&&wire.selector<=(v2?6:4),"memory wire header invalid");
  require(!(wire.flags&~uint64_t(v2?127:63))&&zero(wire.reserved,sizeof(wire.reserved))&&zero(wire.tail_reserved,sizeof(wire.tail_reserved)),"memory wire reserved fields invalid");
  require((wire.flags&(MLX_MEMORY_CONTIGUOUS|MLX_MEMORY_PRESERVE_FORMAT))!=(MLX_MEMORY_CONTIGUOUS|MLX_MEMORY_PRESERVE_FORMAT),"memory wire formats conflict");
  require(wire.operand_count<=64&&wire.argument_count>0&&wire.argument_count<=256&&wire.word_count<=capacity&&wire.param_count<=16,"memory wire control capacity exceeded");
  require(wire.kind==MLX_MEMORY_CAST||wire.kind==MLX_MEMORY_CAST_DEVICE||!(wire.flags&(MLX_MEMORY_FORCE_COPY|MLX_MEMORY_NONBLOCKING)),"memory wire copy flags outside cast");
  require(wire.kind==MLX_MEMORY_RESHAPE||!(wire.flags&MLX_MEMORY_STRICT_VIEW),"memory wire strict-view flag outside reshape");
  require(wire.kind==MLX_MEMORY_NEW_ONES||!(wire.flags&MLX_MEMORY_EXPLICIT_DTYPE),"memory wire explicit dtype outside new_ones");
  DecodedMemory result;result.view=wire.mode==0;auto &node=result.node;auto &p=node["memory_program"];
  node["id"]="output";node["kind"]=kinds[wire.kind];node["source_operator"]=(wire.flags&MLX_MEMORY_STRICT_VIEW)?"aten.view.default":"wire.memory";node["kwargs"]=Json::Value(Json::objectValue);
  if(wire.flags&MLX_MEMORY_CONTIGUOUS)node["kwargs"]["memory_format"]="torch.contiguous_format";
  if(wire.flags&MLX_MEMORY_PRESERVE_FORMAT)node["kwargs"]["memory_format"]="torch.preserve_format";
  p["profile"]=v2?"mlx-memory-plan-v2":"mlx-memory-plan-v1";p["kind"]=node["kind"];p["mode"]=result.view?"view":"transfer";p["selector"]=selectors[wire.selector];p["reason"]="wire_checked_layout";
  p["register_count"]=4;p["register_bytes"]=8;p["staging_bytes"]=128;p["chunk_bytes"]=64;p["same_reference_device"]=bool(wire.flags&MLX_MEMORY_SAME_DEVICE);p["physical_dma_verified"]=false;
  struct Root {uint64_t base,bytes;DType type;std::shared_ptr<Storage> storage;};std::map<uint64_t,Root> roots;
  std::vector<Json::Value> operands;unsigned tensor_count=0;p["input_layouts"]=Json::Value(Json::objectValue);
  for(unsigned i=0;i<64;++i){
    const auto &arg=wire.operands[i];if(i>=wire.operand_count){require(zero(&arg,sizeof(arg)),"unused memory operand slot is nonzero");continue;}
    if(arg.kind==MLX_HOST_TENSOR){
      require(!arg.scalar_dtype&&!arg.scalar_bits&&arg.root_id>0&&arg.root_id<=64,"memory tensor root/scalar fields invalid");auto t=decode_tensor(arg.tensor,false);
      auto found=roots.find(arg.root_id);
      if(found==roots.end())roots.emplace(arg.root_id,Root{arg.tensor.base,arg.tensor.bytes,t.type,t.storage});
      else{require(found->second.base==arg.tensor.base&&found->second.bytes==arg.tensor.bytes&&found->second.type==t.type,"memory alias root changed physical storage");t.storage=found->second.storage;}
      auto id=name(i);result.values[id]=t;Json::Value reference;reference["value"]=id;operands.push_back(reference);p["input_layouts"][id]=layout(t,root(arg.root_id));result.regions.push_back({arg.tensor.base,arg.tensor.bytes,true,false});++tensor_count;
    }else{
      require(wire.kind==MLX_MEMORY_WHERE&&arg.kind==MLX_HOST_SCALAR&&arg.root_id==0&&zero(&arg.tensor,sizeof(arg.tensor)),"memory scalar operand invalid");
      if(arg.scalar_dtype==MLX_HOST_I64)operands.push_back(Json::Value(Json::Int64(signed_bits(arg.scalar_bits))));
      else if(arg.scalar_dtype==MLX_HOST_BOOL){require(arg.scalar_bits<=1,"memory Boolean scalar is not canonical");operands.push_back(Json::Value(bool(arg.scalar_bits)));}
      else{require(arg.scalar_dtype==MLX_MEMORY_LITERAL_F64,"memory scalar dtype invalid");operands.push_back(real(arg.scalar_bits));}
    }
  }
  require(tensor_count<=63,"memory wire region capacity exceeded");std::set<unsigned> referenced;Json::Value args(Json::arrayValue);
  for(unsigned i=0;i<256;++i){auto index=wire.argument_indices[i];if(i<wire.argument_count){require(index<operands.size(),"memory argument index out of range");args.append(operands[index]);referenced.insert(unsigned(index));}else require(index==0,"unused memory argument index is nonzero");}
  require(referenced.size()==wire.operand_count,"memory operand table contains an unused source");
  result.output=decode_tensor(wire.output,!result.view);
  if(result.view){
    require(wire.output_root&&roots.count(wire.output_root),"memory view output root missing");const auto &r=roots.at(wire.output_root);
    require(r.base==wire.output.base&&r.bytes==wire.output.bytes&&r.type==result.output.type,"memory view output changed its backing");result.output.storage=r.storage;
  }else{require(wire.output_root==0,"memory transfer output is not fresh");result.regions.push_back({wire.output.base,wire.output.bytes,false,true});}
  p["output_layout"]=layout(result.output,result.view?root(wire.output_root):"output");node["output"]["shape"]=p["output_layout"]["shape"];node["output"]["dtype"]=dtype_name(result.output.type);
  p["words"]=Json::Value(Json::arrayValue);for(unsigned i=0;i<capacity;++i){if(i<wire.word_count){require(wire.words[i]<=UINT32_MAX,"memory instruction word is not 32-bit");p["words"].append(Json::UInt(wire.words[i]));}else require(!wire.words[i],"unused memory instruction is nonzero");}
  for(unsigned i=unsigned(wire.param_count);i<16;++i)require(!wire.params[i],"unused memory parameter is nonzero");
  auto param=[&](unsigned i){return Json::Value(Json::Int64(signed_bits(wire.params[i])));};auto count=[&](unsigned n){require(wire.param_count==n,"memory wire parameter count mismatch");};
  if(wire.kind==MLX_MEMORY_EMBEDDING){require(args.size()==2,"memory embedding source count mismatch");count(3);require(wire.params[1]<=1&&wire.params[2]<=1,"memory embedding flags invalid");args.append(param(0));args.append(bool(wire.params[1]));args.append(bool(wire.params[2]));}
  else if(wire.kind==MLX_MEMORY_WHERE){require(args.size()==3,"memory where source count mismatch");count(0);}
  else if(wire.kind==MLX_MEMORY_CAT){count(1);Json::Value list=args;args=Json::Value(Json::arrayValue);args.append(list);args.append(param(0));}
  else if(wire.kind==MLX_MEMORY_ADVANCED_INDEX){
    count(0);require(args.size()>=2&&args.size()<=9,"memory advanced index source count mismatch");
    Json::Value source=args[0],indices(Json::arrayValue);for(unsigned i=1;i<args.size();++i)indices.append(args[i]);
    args=Json::Value(Json::arrayValue);args.append(source);args.append(indices);
  }
  else{
    require(args.size()==1&&args[0].isObject()&&args[0].isMember("value"),"memory unary source must be one tensor");
    switch(wire.kind){
      case MLX_MEMORY_CAST:case MLX_MEMORY_CAST_DEVICE:
        count(0);if(wire.kind==MLX_MEMORY_CAST_DEVICE)args.append("$BOUND_DEVICE");args.append(torch_dtype(result.output.type));args.append(bool(wire.flags&MLX_MEMORY_NONBLOCKING));args.append(bool(wire.flags&MLX_MEMORY_FORCE_COPY));break;
      case MLX_MEMORY_CONTIG:case MLX_MEMORY_ALIAS:count(0);break;
      case MLX_MEMORY_RESHAPE:case MLX_MEMORY_EXPAND:case MLX_MEMORY_NEW_ONES:{require(wire.param_count<=8,"memory requested shape rank exceeds bound");Json::Value shape(Json::arrayValue);for(unsigned i=0;i<wire.param_count;++i)shape.append(param(i));args.append(shape);
        if(wire.kind==MLX_MEMORY_NEW_ONES&&(wire.flags&MLX_MEMORY_EXPLICIT_DTYPE))node["kwargs"]["dtype"]=torch_dtype(result.output.type);
        break;}
      case MLX_MEMORY_TRANSPOSE:case MLX_MEMORY_SELECT:count(2);args.append(param(0));args.append(param(1));break;
      case MLX_MEMORY_UNSQUEEZE:case MLX_MEMORY_SQUEEZE:count(1);args.append(param(0));break;
      case MLX_MEMORY_SLICE:count(5);require(wire.params[4]<=3,"memory slice null mask invalid");args.append(param(0));args.append(wire.params[4]&1?Json::Value():param(1));args.append(wire.params[4]&2?Json::Value():param(2));args.append(param(3));break;
      case MLX_MEMORY_DROPOUT:count(1);args.append(real(wire.params[0]));args.append(false);break;
      default:throw std::runtime_error("memory wire kind not implemented");
    }
  }
  node["args"]=args;return result;
}
DecodedMemory decode_memory(const mlx_memory_wire &wire){return decode_wire(wire);}
DecodedMemory decode_memory(const mlx_memory_wire_v2 &wire){return decode_wire(wire);}
DecodedMemory decode_memory_bytes(const void *data,size_t bytes){
  require(data,"memory wire data is missing");
  if(bytes==sizeof(mlx_memory_wire)){auto wire=std::make_unique<mlx_memory_wire>();std::memcpy(wire.get(),data,bytes);return decode_memory(*wire);}
  require(bytes==sizeof(mlx_memory_wire_v2),"memory wire byte size invalid");
  auto wire=std::make_unique<mlx_memory_wire_v2>();std::memcpy(wire.get(),data,bytes);return decode_memory(*wire);
}
} // namespace mlx::physical_device
