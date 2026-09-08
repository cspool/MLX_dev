#include "matrix_wire.hh"
#include <cstring>

namespace mlx::physical_device {
using namespace tensor_model;
namespace {
bool zero(const void *value,size_t bytes){const auto *p=static_cast<const uint8_t*>(value);for(size_t i=0;i<bytes;++i)if(p[i])return false;return true;}
}
Tensor decode_tensor(const mlx_host_tensor &wire,bool output){
  require(wire.rank<=8&&wire.dtype<=MLX_HOST_BOOL,"wire tensor rank/dtype invalid");
  require(!(wire.access&~UINT64_C(3))&&(wire.access&(output?MLX_HOST_WRITE:MLX_HOST_READ)),"matrix wire permissions invalid");
  Tensor t;t.type=static_cast<DType>(wire.dtype);
  auto bytes=element_bytes(t.type);require(wire.base%bytes==0&&wire.bytes%bytes==0&&wire.bytes<=UINT64_MAX-wire.base&&wire.offset<=INT64_MAX,"matrix wire address/offset overflow");
  t.offset=wire.offset;t.storage=std::make_shared<Storage>();t.storage->bytes=wire.bytes;
  for(unsigned d=0;d<8;++d){
    if(d<wire.rank){require(wire.shape[d]<=INT64_MAX&&wire.stride[d]<=INT64_MAX,"matrix wire shape/stride exceeds range");t.sizes.push_back(wire.shape[d]);t.steps.push_back(wire.stride[d]);}
    else require(!wire.shape[d]&&!wire.stride[d],"matrix wire unused dimensions are nonzero");
  }
  if(t.numel())t.position(t.numel()-1);
  return t;
}
Tensor decode_float_tensor(const mlx_host_tensor &wire,bool output){
  require(wire.dtype<=MLX_HOST_F32,"matrix/vector wire dtype invalid");return decode_tensor(wire,output);
}
DecodedMatrix decode_matrix(const mlx_matrix_wire &wire){
  require(wire.magic==MLX_MATRIX_WIRE_MAGIC&&wire.version==MLX_MATRIX_WIRE_VERSION&&!(wire.flags&~UINT64_C(3))&&zero(wire.reserved,sizeof(wire.reserved)),"matrix wire header/reserved fields invalid");
  require(wire.prologue_count&&wire.body_count&&wire.epilogue_count&&wire.prologue_count<=32&&wire.body_count<=32&&wire.epilogue_count<=32,"matrix wire phase counts invalid");
  auto total=wire.prologue_count+wire.body_count+wire.epilogue_count;require(total<=32,"matrix wire exceeds ROM capacity");
  DecodedMatrix result;result.a=decode_float_tensor(wire.a,false);result.b=decode_float_tensor(wire.b,false);result.output=decode_float_tensor(wire.output,true);result.has_bias=wire.flags&MLX_MATRIX_HAS_BIAS;
  if(result.has_bias)result.bias=decode_float_tensor(wire.bias,false);else require(zero(&wire.bias,sizeof(wire.bias)),"unused matrix bias descriptor is nonzero");
  result.transpose_b=wire.flags&MLX_MATRIX_TRANSPOSE_B;result.m=wire.m;result.n=wire.n;result.k=wire.k;result.a_batch=wire.a_batch;result.b_batch=wire.b_batch;result.output_batch=wire.output_batch;
  auto &p=result.program;p["profile"]="mlx-matrix-f32-kasc-v1";p["tile_m"]=2;p["tile_n"]=16;p["tile_k"]=64;p["rf_vectors"]=16;p["rf_vector_bytes"]=64;p["rf_vectors_used"]=6;p["spm_bytes"]=8192;p["rom_words"]=32;
  p["spm_bytes_used"]=(64*16+2*64+16)*element_bytes(result.a.type);p["input_dtype"]=dtype_name(result.a.type);p["output_dtype"]=dtype_name(result.output.type);p["has_bias"]=result.has_bias;
  for(unsigned i=0;i<32;++i){
    require(wire.words[i]<=UINT32_MAX,"matrix wire instruction is not 32-bit");
    if(i>=total){require(wire.words[i]==0,"matrix wire unused ROM words are nonzero");continue;}
    const char *phase=i<wire.prologue_count?"prologue":i<wire.prologue_count+wire.body_count?"body":"epilogue";p[phase].append(Json::UInt(wire.words[i]));
  }
  result.regions={{wire.a.base,wire.a.bytes,true,false},{wire.b.base,wire.b.bytes,true,false},
    result.has_bias?model_io::Region{wire.bias.base,wire.bias.bytes,true,false}:model_io::Region{0,0,false,false},
    {wire.output.base,wire.output.bytes,false,true}};
  return result;
}
} // namespace mlx::physical_device
