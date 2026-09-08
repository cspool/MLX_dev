#include "matrix_program.h"
#include "mlx_tagged_simulator.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace mlx::tensor_model {
namespace {
inline void require(bool ok,const char *message) { if (!ok) throw std::runtime_error(message); }
enum Op : unsigned { Zero32=1, LoadA16=2, LoadB16=3, Cvt16_32=4, Mul32=5, Add32=6,
                     Cvt32_16=7, Store16=8, LoadA32=9, LoadB32=10, Store32=11,
                     LoadBias16=12, LoadBias32=13 };
struct Inst { unsigned op, dst, a, b, row; };
struct Reg { std::array<uint32_t,16> bits{}; unsigned type=0; }; // 1=F16, 2=F32
float as_float(uint32_t bits) { float value; std::memcpy(&value,&bits,4); return value; }
uint32_t as_bits(float value) { uint32_t bits; std::memcpy(&bits,&value,4); return bits; }
uint32_t half_as_float_bits(uint16_t value) {
  unsigned exponent=(value>>10)&31;
  if (exponent && exponent!=31) return uint32_t(value&0x8000)<<16 | (exponent+112)<<23 | uint32_t(value&1023)<<13;
  return as_bits(tagged::half_to_float(value));
}

// Validate view bounds once. Subsequent affine matrix accesses are bounded by
// the checked M/N/K extents, avoiding tensor shape parsing on every PE load.
struct Reader {
  const Tensor &tensor;
  bool dense;
  explicit Reader(const Tensor &t):tensor(t),dense(t.contiguous()) {
    require(t.type==DType::F16 || t.type==DType::F32,"matrix memory dtype is not floating point");
    if (t.numel()) {
      require(t.offset>=0,"negative matrix view base");
      uint64_t last=t.offset;
      for (size_t i=0;i<t.sizes.size();++i) {
        require(t.steps[i]>=0,"negative matrix stride");
        require(!t.steps[i] || uint64_t(t.sizes[i]-1)<=(UINT64_MAX-last)/uint64_t(t.steps[i]),"matrix address overflow");
        last+=uint64_t(t.sizes[i]-1)*uint64_t(t.steps[i]);
      }
      require(last<t.storage->bytes/element_bytes(t.type),"matrix view exceeds global storage");
    }
  }
  uint32_t read(uint64_t flat) const {
    uint64_t index=tensor.offset;
    if (dense) index+=flat;
    else for (size_t d=tensor.sizes.size();d-->0;) {
      index+=(flat%uint64_t(tensor.sizes[d]))*uint64_t(tensor.steps[d]); flat/=tensor.sizes[d];
    }
    uint32_t bits=0;
    std::memcpy(&bits,tensor.storage->data+index*element_bytes(tensor.type),element_bytes(tensor.type));
    return bits;
  }
};
std::vector<Inst> decode(const Json::Value &words) {
  require(words.isArray(),"matrix phase must contain instruction words");
  std::vector<Inst> out;
  for (const auto &value:words) {
    require(value.isUInt(),"matrix instruction is not a 32-bit word");
    uint32_t word=value.asUInt();
    Inst i{word&255,(word>>8)&15,(word>>12)&15,(word>>16)&15,(word>>20)&3};
    require(!(word>>22) && i.op>=1 && i.op<=13 && i.row<=2,"invalid matrix opcode or reserved bits");
    require(i.dst<6 && i.a<6 && i.b<6,"matrix instruction exceeds allocated RF frame");
    if (i.op==Mul32 || i.op==Add32) { /* all register fields used */ }
    else if (i.op==Cvt16_32 || i.op==Cvt32_16) require(i.b==0,"noncanonical conversion word");
    else if (i.op==Store16 || i.op==Store32) require(i.dst==0 && i.b==0 && i.row<2,"noncanonical store word");
    else require(i.a==0 && i.b==0,"noncanonical load/zero word");
    if (i.op==LoadA16 || i.op==LoadA32) require(i.row<2,"A load requires a row selector");
    if (i.op==LoadB16 || i.op==LoadB32 || i.op==LoadBias16 || i.op==LoadBias32) require(i.row==2,"shared load has noncanonical row");
    out.push_back(i);
  }
  return out;
}
} // namespace

Json::Value MatrixInstructionStats::json() const {
  Json::Value out(Json::objectValue);
  out["profile"]="mlx-matrix-f32-kasc-v1";
  out["classification"]="executed_pe_matrix_microcode_not_cycle_or_system_validation";
  out["calls"]=Json::UInt64(calls); out["output_tiles"]=Json::UInt64(tiles);
  out["instructions"]=Json::UInt64(instructions); out["inactive_row_instructions"]=Json::UInt64(inactive_row_instructions);
  out["mul_active_lanes"]=Json::UInt64(mul_lanes); out["add_active_lanes"]=Json::UInt64(add_lanes);
  out["global_read_bytes"]=Json::UInt64(memory_read_bytes); out["global_write_bytes"]=Json::UInt64(memory_write_bytes);
  out["max_rom_words"]=max_rom_words; out["max_spm_bytes"]=max_spm_bytes;
  for (unsigned op=1;op<14;++op) out["opcode_counts"][std::to_string(op)]=Json::UInt64(opcode_counts[op]);
  out["timing_verified"]=false; out["system_dma_verified"]=false;
  return out;
}

void execute_matrix_program(const Json::Value &program,const Tensor &a,const Tensor &b,
                            const Tensor *bias,bool transposed_b,uint64_t a_batch,
                            uint64_t b_batch,uint64_t m,uint64_t n,uint64_t k,
                            Tensor &output,uint64_t output_batch,MatrixInstructionStats &stats) {
  require(program["profile"]=="mlx-matrix-f32-kasc-v1","unsupported matrix program profile");
  require(program["tile_m"]==2 && program["tile_n"]==16 && program["tile_k"]==64 &&
          program["rf_vectors"]==16 && program["rf_vector_bytes"]==64 &&
          program["spm_bytes"]==8192 && program["rom_words"]==32,"unsupported matrix resource contract");
  require(program["input_dtype"]==dtype_name(a.type) && a.type==b.type &&
          program["output_dtype"]==dtype_name(output.type) && program["has_bias"].asBool()==bool(bias),"matrix program dtype/bias mismatch");
  require(!bias || (bias->type==a.type && bias->sizes==Shape{int64_t(n)}),"matrix bias contract mismatch");
  require(std::numeric_limits<float>::is_iec559 && sizeof(float)==4 && std::fegetround()==FE_TONEAREST,
          "matrix instructions require IEEE binary32 round-to-nearest-even");
#if defined(__SSE__)
  require((_mm_getcsr()&((1u<<15)|(1u<<6)))==0,"matrix numeric contract forbids FTZ/DAZ");
#endif
  auto prologue=decode(program["prologue"]), body=decode(program["body"]), epilogue=decode(program["epilogue"]);
  const size_t word_count=prologue.size()+body.size()+epilogue.size();
  require(word_count>0 && word_count<=32 && !body.empty() && !epilogue.empty(),"matrix ROM capacity/phase violation");
  require(program["rf_vectors_used"]==6,"matrix RF allocation metadata mismatch");
  const unsigned bytes=element_bytes(a.type), spm_used=(64*16+2*64+16)*bytes;
  require(program["spm_bytes_used"].isUInt() && program["spm_bytes_used"].asUInt()==spm_used && spm_used<=8192,"matrix SPM capacity violation");
  require(output.contiguous() && output.offset==0 && output.storage->writable,"matrix output must have owned contiguous storage");
  require(a_batch*m*k+m*k<=a.numel() && b_batch*k*n+k*n<=b.numel() &&
          output_batch*m*n+m*n<=output.numel(),"matrix batch window exceeds tensor extent");
  Reader ar(a),br(b); std::unique_ptr<Reader> bias_reader=bias?std::make_unique<Reader>(*bias):nullptr;
  ++stats.calls; stats.max_rom_words=std::max(stats.max_rom_words,unsigned(word_count));
  stats.max_spm_bytes=std::max(stats.max_spm_bytes,spm_used);
  std::array<uint8_t,8192> spm{};
  const unsigned a_base=64*16*bytes, bias_base=(64*16+2*64)*bytes;
  auto spm_write=[&](unsigned offset,uint32_t bits){std::memcpy(spm.data()+offset,&bits,bytes);};
  auto spm_read=[&](unsigned offset){uint32_t bits=0; std::memcpy(&bits,spm.data()+offset,bytes); return bits;};
  for (uint64_t row_base=0;row_base<m;row_base+=2) for (uint64_t col_base=0;col_base<n;col_base+=16) {
    ++stats.tiles;
    const unsigned rows=unsigned(std::min<uint64_t>(2,m-row_base)), lanes=unsigned(std::min<uint64_t>(16,n-col_base));
    std::array<Reg,16> rf{}; std::array<bool,2> stored{};
    auto execute=[&](const std::vector<Inst> &code,unsigned ki) {
      for (const auto &i:code) {
        ++stats.instructions; ++stats.opcode_counts[i.op];
        if (i.row<2 && i.row>=rows) { ++stats.inactive_row_instructions; continue; }
        auto &dst=rf[i.dst];
        if (i.op==Zero32) { dst.bits.fill(0); dst.type=2; }
        else if (i.op==LoadA16 || i.op==LoadA32 || i.op==LoadB16 || i.op==LoadB32 || i.op==LoadBias16 || i.op==LoadBias32) {
          unsigned type=(i.op==LoadA16 || i.op==LoadB16 || i.op==LoadBias16)?1:2;
          require(type==(a.type==DType::F16?1u:2u),"SPM load precision mismatch");
          for (unsigned lane=0;lane<lanes;++lane) {
            unsigned address=(i.op==LoadA16 || i.op==LoadA32)?a_base+(i.row*64+ki)*bytes:
                (i.op==LoadB16 || i.op==LoadB32)?(ki*16+lane)*bytes:bias_base+lane*bytes;
            if (i.op==LoadBias16 || i.op==LoadBias32) require(bool(bias),"bias load without bias binding");
            dst.bits[lane]=spm_read(address);
          }
          dst.type=type;
        } else if (i.op==Cvt16_32 || i.op==Cvt32_16) {
          require(rf[i.a].type==(i.op==Cvt16_32?1u:2u),"conversion reads invalid/wrong-type register");
          for (unsigned lane=0;lane<lanes;++lane) dst.bits[lane]=i.op==Cvt16_32?
              half_as_float_bits(uint16_t(rf[i.a].bits[lane])):tagged::float_to_half(as_float(rf[i.a].bits[lane]));
          dst.type=i.op==Cvt16_32?2:1;
        } else if (i.op==Mul32 || i.op==Add32) {
          require(rf[i.a].type==2 && rf[i.b].type==2,"arithmetic reads invalid/wrong-type register");
          for (unsigned lane=0;lane<lanes;++lane) {
            float x=as_float(rf[i.a].bits[lane]),y=as_float(rf[i.b].bits[lane]);
            dst.bits[lane]=as_bits(i.op==Mul32?x*y:x+y);
          }
          dst.type=2;
          if (i.op==Mul32) stats.mul_lanes+=lanes; else stats.add_lanes+=lanes;
        } else {
          require(rf[i.a].type==(i.op==Store16?1u:2u) && output.type==(i.op==Store16?DType::F16:DType::F32),"store reads invalid/wrong-type register");
          require(!stored[i.row],"duplicate matrix output store"); stored[i.row]=true;
          const uint64_t first=output_batch*m*n+(row_base+i.row)*n+col_base;
          for (unsigned lane=0;lane<lanes;++lane) std::memcpy(output.storage->writable+(first+lane)*element_bytes(output.type),&rf[i.a].bits[lane],element_bytes(output.type));
          stats.memory_write_bytes+=lanes*element_bytes(output.type);
        }
      }
    };
    execute(prologue,0);
    for (uint64_t k_base=0;k_base<k;k_base+=64) {
      const unsigned count=unsigned(std::min<uint64_t>(64,k-k_base));
      // Actual bounded transfers into SPM. No full-weight F32 materialization,
      // no golden activation reads, and no pretend DMA completion/cycle count.
      for (unsigned row=0;row<rows;++row) for (unsigned ki=0;ki<count;++ki)
        spm_write(a_base+(row*64+ki)*bytes,ar.read(a_batch*m*k+(row_base+row)*k+k_base+ki));
      for (unsigned lane=0;lane<lanes;++lane) for (unsigned ki=0;ki<count;++ki) {
        uint64_t flat=b_batch*k*n+(transposed_b?(col_base+lane)*k+k_base+ki:(k_base+ki)*n+col_base+lane);
        spm_write((ki*16+lane)*bytes,br.read(flat));
      }
      stats.memory_read_bytes+=(rows+lanes)*count*bytes;
      for (unsigned ki=0;ki<count;++ki) execute(body,ki);
    }
    if (bias) {
      for (unsigned lane=0;lane<lanes;++lane) spm_write(bias_base+lane*bytes,bias_reader->read(col_base+lane));
      stats.memory_read_bytes+=lanes*bytes;
    }
    execute(epilogue,0);
    require(stored[0] && (rows==1 || stored[1]),"matrix program did not store every output row");
  }
}
} // namespace mlx::tensor_model
