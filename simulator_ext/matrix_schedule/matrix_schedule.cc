#include "matrix_schedule.h"
#include "../model_io/tensor_memory_port.h"
#include "mlx_tagged_simulator.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace mlx::matrix_schedule {
using tensor_model::Tensor;
using tensor_model::DType;
using tensor_model::element_bytes;
using tensor_model::dtype_name;
namespace {
void check(bool ok,const char *message) { if (!ok) throw std::runtime_error(message); }
enum Op : unsigned { Zero=1, LoadA16=2, LoadB16=3, CvtUp=4, Mul=5, Add=6, CvtDown=7,
                     Store16=8, LoadA32=9, LoadB32=10, Store32=11, Bias16=12, Bias32=13 };
enum Phase : unsigned { Prologue, Fill, Body, BiasFill, Epilogue, Drain };
const char *phase_name(Phase phase) {
  static const char *names[]={"prologue","fill","body","bias_fill","epilogue","drain"};
  return names[phase];
}
struct Instruction { unsigned op,dst,a,b,row; };
using Register=shared_array::Register;
struct Context {
  bool active=false, busy=false, memory_busy=false;
  uint64_t block=0,epoch=0,row_base=0,col_base=0,k_base=0;
  unsigned pe=0,slot=0,arena=0,rows=0,lanes=0,count=0,ki=0,pc=0,move=0;
  Phase phase=Prologue;
  std::array<bool,2> stored{};
  shared_array::Lease lease;
};
struct Owner {
  unsigned context=0,pc=0,ki=0,move=0;
  uint64_t block=0,epoch=0,k_base=0;
  Phase phase=Prologue;
};
struct PendingInstruction { Owner owner; Instruction instruction; Register value; uint64_t due=0; };
struct PendingDMA {
  Owner owner;
  unsigned region=0,bytes=0,spm=0;
  uint64_t id=0,offset=0,data=0;
  bool write=false;
};
std::vector<Instruction> decode(const Json::Value &words) {
  check(words.isArray(),"matrix phase is not an instruction array");
  std::vector<Instruction> result;
  for (const auto &v:words) {
    check(v.isUInt(),"matrix word is not unsigned 32-bit");
    unsigned w=v.asUInt();
    Instruction i{w&255,(w>>8)&15,(w>>12)&15,(w>>16)&15,(w>>20)&3};
    check(!(w>>22) && i.op>=1 && i.op<=13 && i.dst<6 && i.a<6 && i.b<6 && i.row<=2,"invalid matrix opcode/fields");
    if (i.op==Mul || i.op==Add) {}
    else if (i.op==CvtUp || i.op==CvtDown) check(i.b==0,"noncanonical conversion fields");
    else if (i.op==Store16 || i.op==Store32) check(i.dst==0 && i.b==0 && i.row<2,"noncanonical store fields");
    else check(i.a==0 && i.b==0,"noncanonical load/zero fields");
    if (i.op==LoadA16 || i.op==LoadA32) check(i.row<2,"A load has no row selector");
    if (i.op==LoadB16 || i.op==LoadB32 || i.op==Bias16 || i.op==Bias32) check(i.row==2,"shared load has a row selector");
    result.push_back(i);
  }
  return result;
}
bool memory_op(unsigned op) { return op==LoadA16 || op==LoadA32 || op==LoadB16 || op==LoadB32 || op==Bias16 || op==Bias32 || op==Store16 || op==Store32; }
bool store_op(unsigned op) { return op==Store16 || op==Store32; }
float floating(uint32_t bits) { float f; std::memcpy(&f,&bits,4); return f; }
uint32_t bits(float f) { uint32_t b; std::memcpy(&b,&f,4); return b; }
uint32_t convert_half(uint16_t h) {
  unsigned e=(h>>10)&31;
  if (e && e!=31) return uint32_t(h&0x8000)<<16 | (e+112)<<23 | uint32_t(h&1023)<<13;
  return bits(tagged::half_to_float(h));
}
} // namespace

Options Options::parse(const Json::Value &v) {
  Options o;
  check(v.isNull()||v.isObject(),"schedule options must be an object");
  const std::set<std::string> allowed={"rows","columns","contexts","dma_latency","spm_latency","multiply_latency","add_latency","convert_latency","dma_request_period","dma_response_period","spm_period","writeback_period","compute_ii","trace_limit","max_cycles","overlap","trace","inject_stale_dma_epoch","cache_control"};
  if(v.isObject())for(const auto &key:v.getMemberNames())check(allowed.count(key),"unknown schedule option");
  auto set=[&](const char *key,unsigned &field){if(v.isMember(key)){check(v[key].isUInt(),"noninteger schedule option");field=v[key].asUInt();}};
  set("rows",o.rows);set("columns",o.columns);set("contexts",o.contexts);
  set("dma_latency",o.dma_latency);set("spm_latency",o.spm_latency);
  set("multiply_latency",o.multiply_latency);set("add_latency",o.add_latency);set("convert_latency",o.convert_latency);
  set("dma_request_period",o.dma_request_period);set("dma_response_period",o.dma_response_period);
  set("spm_period",o.spm_period);set("writeback_period",o.writeback_period);set("compute_ii",o.compute_ii);set("trace_limit",o.trace_limit);
  if(v.isMember("max_cycles")){check(v["max_cycles"].isUInt64(),"invalid cycle limit");o.max_cycles=v["max_cycles"].asUInt64();}
  for(const char *key:{"overlap","trace","inject_stale_dma_epoch","cache_control"})if(v.isMember(key))check(v[key].isBool(),"schedule flags must be Boolean");
  if(v.isMember("overlap"))o.overlap=v["overlap"].asBool();
  if(v.isMember("trace"))o.trace=v["trace"].asBool();
  if(v.isMember("inject_stale_dma_epoch"))o.inject_stale_dma_epoch=v["inject_stale_dma_epoch"].asBool();
  if(v.isMember("cache_control"))o.cache_control=v["cache_control"].asBool();
  o.validate();return o;
}
void Options::validate() const {
  check(rows>=1&&rows<=4&&columns>=1&&columns<=4,"matrix array exceeds bounded geometry");
  check(contexts>=1&&contexts<=2,"six-vector frames limit each PE to two contexts");
  for(unsigned v:{dma_latency,spm_latency,multiply_latency,add_latency,convert_latency,dma_request_period,dma_response_period,spm_period,writeback_period,compute_ii})
    check(v>=1&&v<=1024,"matrix timing/ready period outside [1,1024]");
  check(max_cycles>0&&max_cycles<=1000000000000ULL,"invalid matrix cycle limit");
  check(!trace||(trace_limit>0&&trace_limit<=10000000),"invalid matrix trace limit");
}

struct Simulator::Impl {
  Options options;
  std::array<Tensor,4> tensors; // A, B, bias, output; no golden output tensor
  bool bias=false,transpose=false,stale_injected=false;
  uint64_t a_batch,b_batch,m,n,k,out_batch,cycles=0,next_block=0,retired=0,total_blocks=0;
  unsigned bytes,out_bytes,spm_bytes,arena_vectors,a_base,bias_base,pes,rom_words;
  std::array<std::vector<Instruction>,3> code;
  std::array<Context,32> contexts{};
  std::unique_ptr<shared_array::Resources> owned_array;
  shared_array::Resources *array=nullptr;
  shared_array::Client client=0;
  uint64_t source_id=UINT64_MAX,array_version=0;
  std::array<std::optional<PendingInstruction>,16> fu;
  std::array<uint64_t,16> next_compute{};
  std::optional<PendingInstruction> spm_pending;
  std::optional<PendingDMA> dma;
  std::unique_ptr<model_io::MemoryPort> internal_memory;
  model_io::MemoryPort *memory_port=nullptr;
  std::optional<model_io::Response> held_response;
  uint64_t next_request_id=1;
  bool external_memory=false;
  tensor_model::MatrixInstructionStats numeric;
  uint64_t admitted=0,dma_requests=0,dma_responses=0,dma_read_bytes=0,dma_write_bytes=0;
  uint64_t same_pe_overlap=0,compute_dma_overlap=0,writeback_stalls=0,admission_stalls=0;
  uint64_t dma_busy_cycles=0,compute_busy_pe_cycles=0,spm_busy_cycles=0;
  unsigned peak_contexts=0,peak_spm_vectors=0;
  bool resources_dirty=true;
  std::vector<unsigned> active_order;
  std::optional<shared_array::Offer> admission_offer;
  uint64_t control_recomputations=0,invariant_checks=0;
  Json::Value trace{Json::arrayValue};

  Impl(const Json::Value &p,Tensor a,Tensor b,const Tensor *bias_arg,bool tb,uint64_t ab,uint64_t bb,
       uint64_t mm,uint64_t nn,uint64_t kk,Tensor out,uint64_t ob,Options o,model_io::MemoryPort *port,shared_array::Resources *shared,uint64_t source)
      :options(o),bias(bool(bias_arg)),transpose(tb),a_batch(ab),b_batch(bb),m(mm),n(nn),k(kk),out_batch(ob),memory_port(port),external_memory(port!=nullptr) {
    options.validate();pes=options.rows*options.columns;
    tensors[0]=std::move(a);tensors[1]=std::move(b);tensors[3]=std::move(out);if(bias)tensors[2]=*bias_arg;
    bytes=element_bytes(tensors[0].type);out_bytes=element_bytes(tensors[3].type);
    check(p["profile"]=="mlx-matrix-f32-kasc-v1"&&p["tile_m"]==2&&p["tile_n"]==16&&p["tile_k"]==64,"unsupported scheduled matrix profile");
    check(p["rf_vectors"]==16&&p["rf_vector_bytes"]==64&&p["spm_bytes"]==8192&&p["rom_words"]==32&&p["rf_vectors_used"]==6,"matrix resource descriptor mismatch");
    check((tensors[0].type==DType::F16||tensors[0].type==DType::F32)&&tensors[1].type==tensors[0].type,"scheduled matrix input dtype mismatch");
    check((tensors[3].type==DType::F16||tensors[3].type==DType::F32)&&p["input_dtype"]==dtype_name(tensors[0].type)&&p["output_dtype"]==dtype_name(tensors[3].type),"scheduled matrix numeric descriptor mismatch");
    check(p["has_bias"].asBool()==bias&&(!bias||(tensors[2].type==tensors[0].type&&tensors[2].numel()==n)),"scheduled matrix bias mismatch");
    check(m<=INT32_MAX&&n<=INT32_MAX&&k<=INT32_MAX,"matrix dimensions exceed bounded interface");
    check((!m||!k||a_batch<tensors[0].numel()/(m*k))&&(!k||!n||b_batch<tensors[1].numel()/(k*n))&&(!m||!n||out_batch<tensors[3].numel()/(m*n)),"matrix batch exceeds bound storage");
    check(tensors[0].storage&&tensors[1].storage&&tensors[3].storage&&(!bias||tensors[2].storage),"matrix memory descriptor is missing");
    check(tensors[3].contiguous()&&(external_memory||tensors[3].storage->writable),"output requires contiguous memory binding");
    check(std::numeric_limits<float>::is_iec559&&sizeof(float)==4&&std::fegetround()==FE_TONEAREST,"matrix requires IEEE FP32 RNE");
#if defined(__SSE__)
    check((_mm_getcsr()&((1u<<15)|(1u<<6)))==0,"matrix requires gradual underflow, not FTZ/DAZ");
#endif
    code[0]=decode(p["prologue"]);code[1]=decode(p["body"]);code[2]=decode(p["epilogue"]);
    for(const auto &i:code[0])check(!memory_op(i.op),"prologue cannot access unfilled SPM");
    for(const auto &i:code[1])check(!store_op(i.op)&&i.op!=Bias16&&i.op!=Bias32,"body cannot store outputs or read unfilled bias");
    for(const auto &i:code[2])check(i.op!=LoadA16&&i.op!=LoadA32&&i.op!=LoadB16&&i.op!=LoadB32,"epilogue cannot read expired input tiles");
    rom_words=code[0].size()+code[1].size()+code[2].size();
    check(rom_words>0&&rom_words<=32&&!code[0].empty()&&!code[1].empty()&&!code[2].empty(),"matrix ROM/phase capacity violation");
    spm_bytes=(64*16+2*64+16)*bytes;arena_vectors=(spm_bytes+63)/64;
    check(p["spm_bytes_used"].isUInt()&&p["spm_bytes_used"].asUInt()==spm_bytes&&arena_vectors<=128,"SPM allocation descriptor mismatch");
    a_base=64*16*bytes;bias_base=(64*16+2*64)*bytes;
    total_blocks=((m+1)/2)*((n+15)/16);
    numeric.calls=1;numeric.max_rom_words=rom_words;numeric.max_spm_bytes=spm_bytes;
    if(!memory_port){internal_memory=std::make_unique<model_io::TensorMemoryPort>(std::vector<Tensor>(tensors.begin(),tensors.end()),options.dma_latency);memory_port=internal_memory.get();}
    shared_array::Hardware hardware=shared?shared->hardware():shared_array::Hardware{};
    hardware.rows=options.rows;hardware.columns=options.columns;hardware.contexts=options.contexts;
    hardware.spm_period=options.spm_period;hardware.writeback_period=options.writeback_period;hardware.compute_ii=options.compute_ii;
    hardware.dma_request_period=options.dma_request_period;hardware.dma_response_period=options.dma_response_period;
    hardware.multiply_latency=options.multiply_latency;hardware.add_latency=options.add_latency;hardware.convert_latency=options.convert_latency;hardware.spm_latency=options.spm_latency;
    if(shared)check(hardware==shared->hardware(),"matrix frontend differs from shared physical hardware");
    else owned_array=std::make_unique<shared_array::Resources>(hardware);
    array=shared?shared:owned_array.get();source_id=source;std::vector<uint32_t> words;
    for(const char *part:{"prologue","body","epilogue"})for(const auto &word:p[part])words.push_back(word.asUInt());
    const auto key="matrix:"+std::to_string(code[0].size())+":"+std::to_string(code[1].size())+":"+std::to_string(code[2].size());
    client=array->attach(key,words,6,arena_vectors,source_id);
  }
  ~Impl(){if(client)array->detach(client,done());}
  Owner owner(unsigned id) const {
    const auto &c=contexts[id];return Owner{id,c.pc,c.ki,c.move,c.block,c.epoch,c.k_base,c.phase};
  }
  Context &validate(const Owner &o) {
    auto &c=contexts[o.context];
    check(c.active&&c.epoch==o.epoch&&c.block==o.block&&c.phase==o.phase&&c.pc==o.pc&&c.ki==o.ki&&c.move==o.move&&c.k_base==o.k_base,"stale or mismatched matrix completion owner");
    return c;
  }
  void event(const char *name,unsigned id,const char *pipeline="control",unsigned op=0) {
    if(!options.trace)return;
    check(trace.size()<options.trace_limit,"matrix trace limit exceeded; disable trace for larger runs");
    const auto &c=contexts[id];Json::Value e(Json::objectValue);
    e["cycle"]=Json::UInt64(cycles);e["event"]=name;e["pe"]=c.pe;e["context_slot"]=c.slot;
    e["block_id"]=Json::UInt64(c.block);e["epoch"]=Json::UInt64(c.epoch);e["phase"]=phase_name(c.phase);
    e["iteration"]=Json::UInt64(c.k_base+c.ki);e["pc"]=c.pc;e["pipeline"]=pipeline;e["opcode"]=op;trace.append(e);
    if(!owned_array){auto &row=trace[trace.size()-1];row["array_cycle"]=Json::UInt64(array->cycle());row["shared_client"]=Json::UInt64(client);row["source_operator_id"]=Json::UInt64(source_id);row["lease_id"]=Json::UInt64(c.lease.id);
      if(std::strcmp(name,"admit")==0){row["rf_base"]=c.lease.rf_base;row["rf_vectors"]=c.lease.rf_count;row["spm_base"]=c.lease.spm_base;row["spm_vectors"]=c.lease.spm_count;row["rom_base"]=c.lease.rom_base;row["rom_words"]=c.lease.rom_count;}}
  }
  void dma_event(const char *name,const PendingDMA &p) {
    event(name,p.owner.context,"dma");if(!options.trace)return;
    auto &e=trace[trace.size()-1];e["region"]=p.region;e["byte_offset"]=Json::UInt64(p.offset);
    e["request_id"]=Json::UInt64(p.id);
    e["bytes"]=p.bytes;e["write"]=p.write;e["data"]=Json::UInt64(p.data);e["memory_index"]=p.owner.move;e["spm_byte_offset"]=p.spm;
  }
  Register &reg(Context &c,unsigned index) {return array->reg(c.lease,index);}
  uint8_t *spm_data(const Context &c,unsigned absolute,unsigned count){check(absolute>=c.arena*64,"matrix SPM address precedes owned arena");return array->spm(c.lease,absolute-c.arena*64,count);}
  const std::vector<Instruction> &instructions(const Context &c) const {
    check(c.phase==Prologue||c.phase==Body||c.phase==Epilogue,"instruction requested in memory phase");
    return code[c.phase==Prologue?0:c.phase==Body?1:2];
  }
  bool instruction_phase(const Context &c) const {return c.phase==Prologue||c.phase==Body||c.phase==Epilogue;}
  Instruction front(const Context &c)const{
    const auto offset=c.phase==Prologue?0:c.phase==Body?code[0].size():code[0].size()+code[1].size();
    auto word=array->instruction(c.lease,unsigned(offset+c.pc));return Instruction{word&255,(word>>8)&15,(word>>12)&15,(word>>16)&15,(word>>20)&3};
  }
  bool eligible(unsigned id) const {
    const auto &c=contexts[id];
    if(!c.active)return false;
    if(!options.overlap)for(unsigned s=0;s<options.contexts;++s){const auto &other=contexts[c.pe*2+s];if(other.active&&other.block<c.block)return false;}
    return true;
  }
  std::vector<unsigned> priority() const {
    std::vector<unsigned> ids;
    for(unsigned p=0;p<pes;++p)for(unsigned s=0;s<options.contexts;++s)if(contexts[p*2+s].active)ids.push_back(p*2+s);
    std::sort(ids.begin(),ids.end(),[&](unsigned a,unsigned b){return contexts[a].block<contexts[b].block;});return ids;
  }
  void recompute_control(){
    // Admission and block priority depend ONLY on active slots, next_block,
    // and SPM ownership. PC/FU/DMA changes still arbitrate every actual cycle.
    active_order=priority();admission_offer.reset();++control_recomputations;
    if(next_block<total_blocks)admission_offer=array->offer(client,unsigned(next_block%pes));
    resources_dirty=false;array_version=array->version();
  }
  void enter_memory(Context &c,Phase phase) {c.phase=phase;c.pc=0;c.move=0;c.ki=0;c.count=unsigned(std::min<uint64_t>(64,k-c.k_base));}
  void finish_instruction(unsigned id) {
    auto &c=contexts[id];c.busy=false;
    if(++c.pc<instructions(c).size())return;
    c.pc=0;
    if(c.phase==Prologue){if(k)enter_memory(c,Fill);else if(bias)enter_memory(c,BiasFill);else c.phase=Epilogue;}
    else if(c.phase==Body){
      if(++c.ki<c.count)return;
      c.k_base+=c.count;c.ki=0;
      if(c.k_base<k)enter_memory(c,Fill);else if(bias)enter_memory(c,BiasFill);else c.phase=Epilogue;
    }else{
      check(c.stored[0]&&(c.rows==1||c.stored[1]),"matrix epilogue omitted an output store");enter_memory(c,Drain);
    }
  }
  unsigned memory_count(const Context &c) const {return c.phase==Fill?(c.rows+c.lanes)*c.count:c.phase==BiasFill?c.lanes:c.rows*c.lanes;}
  PendingDMA request(unsigned id) {
    auto &c=contexts[id];PendingDMA r;r.owner=owner(id);r.id=next_request_id;r.bytes=c.phase==Drain?out_bytes:bytes;
    uint64_t flat;
    if(c.phase==Fill&&c.move<c.rows*c.count){unsigned row=c.move/c.count,ki=c.move%c.count;r.region=0;flat=a_batch*m*k+(c.row_base+row)*k+c.k_base+ki;r.spm=c.arena*64+a_base+(row*64+ki)*bytes;}
    else if(c.phase==Fill){unsigned index=c.move-c.rows*c.count,lane=index/c.count,ki=index%c.count;r.region=1;
      flat=b_batch*k*n+(transpose?(c.col_base+lane)*k+c.k_base+ki:(c.k_base+ki)*n+c.col_base+lane);r.spm=c.arena*64+(ki*16+lane)*bytes;}
    else if(c.phase==BiasFill){r.region=2;flat=c.col_base+c.move;r.spm=c.arena*64+bias_base+c.move*bytes;}
    else{unsigned row=c.move/c.lanes,lane=c.move%c.lanes;r.region=3;r.write=true;flat=out_batch*m*n+(c.row_base+row)*n+c.col_base+lane;r.spm=c.arena*64+a_base+(row*16+lane)*out_bytes;}
    r.offset=tensors[r.region].position(flat)*r.bytes;
    check(r.spm+r.bytes<=(c.arena+arena_vectors)*64&&r.spm+r.bytes<=8192,"DMA SPM address exceeds owned arena");
    check(r.offset+r.bytes<=tensors[r.region].storage->bytes,"DMA global address exceeds bound storage");
    if(r.write)std::memcpy(&r.data,spm_data(c,r.spm,r.bytes),r.bytes);
    return r;
  }
  void finish_memory(unsigned id) {
    auto &c=contexts[id];c.memory_busy=false;
    if(++c.move<memory_count(c))return;
    c.move=0;
    if(c.phase==Fill){c.phase=Body;c.pc=0;c.ki=0;event("wake",id);}
    else if(c.phase==BiasFill){c.phase=Epilogue;c.pc=0;event("wake",id);}
    else{
      check(!c.busy,"retiring context has an in-flight instruction");event("retire",id);
      for(unsigned r=0;r<6;++r)reg(c,r).type=0;
      array->retire(c.lease);
      c.active=false;++retired;resources_dirty=true;
    }
  }
  unsigned latency(unsigned op) const {return memory_op(op)?options.spm_latency:op==Mul?options.multiply_latency:op==Add?options.add_latency:op==Zero?1:options.convert_latency;}
  Register evaluate(Context &c,const Instruction &i) {
    Register result;
    if(i.op==Zero){result.type=2;return result;}
    if(store_op(i.op)){check(reg(c,i.a).type==(i.op==Store16?1u:2u)&&tensors[3].type==(i.op==Store16?DType::F16:DType::F32),"store reads invalid/wrong-type register");return reg(c,i.a);}
    if(memory_op(i.op)){
      result.type=(i.op==LoadA16||i.op==LoadB16||i.op==Bias16)?1:2;
      check(result.type==(tensors[0].type==DType::F16?1u:2u),"SPM load precision mismatch");
      check((i.op!=Bias16&&i.op!=Bias32)||bias,"bias instruction has no bound bias");
      for(unsigned lane=0;lane<c.lanes;++lane){unsigned address=(i.op==LoadA16||i.op==LoadA32)?a_base+(i.row*64+c.ki)*bytes:(i.op==LoadB16||i.op==LoadB32)?(c.ki*16+lane)*bytes:bias_base+lane*bytes;std::memcpy(&result.data[lane],spm_data(c,c.arena*64+address,bytes),bytes);}return result;
    }
    if(i.op==CvtUp||i.op==CvtDown){check(reg(c,i.a).type==(i.op==CvtUp?1u:2u),"conversion reads invalid/wrong-type register");result.type=i.op==CvtUp?2:1;
      for(unsigned lane=0;lane<c.lanes;++lane)result.data[lane]=i.op==CvtUp?convert_half(uint16_t(reg(c,i.a).data[lane])):tagged::float_to_half(floating(reg(c,i.a).data[lane]));
      return result;}
    check(reg(c,i.a).type==2&&reg(c,i.b).type==2,"arithmetic reads invalid/wrong-type register");result.type=2;
    for(unsigned lane=0;lane<c.lanes;++lane){float a=floating(reg(c,i.a).data[lane]),b=floating(reg(c,i.b).data[lane]);result.data[lane]=bits(i.op==Mul?a*b:a+b);}
    return result;
  }
  void complete_instruction(PendingInstruction &p,const char *pipeline) {
    auto &c=validate(p.owner);check(c.busy,"completion has no in-flight owner");
    if(store_op(p.instruction.op)){
      unsigned row=p.instruction.row;check(!c.stored[row],"duplicate matrix output store");c.stored[row]=true;
      for(unsigned lane=0;lane<c.lanes;++lane)std::memcpy(spm_data(c,c.arena*64+a_base+(row*16+lane)*out_bytes,out_bytes),&p.value.data[lane],out_bytes);
    }else reg(c,p.instruction.dst)=p.value;
    event("complete",p.owner.context,pipeline,p.instruction.op);finish_instruction(p.owner.context);
  }
  void invariant() {
    ++invariant_checks;
    unsigned live=0,used=0;
    for(unsigned p=0;p<pes;++p){unsigned count=0;for(unsigned s=0;s<options.contexts;++s){auto &c=contexts[p*2+s];if(!c.active)continue;++count;++live;
      array->validate_lease(c.lease);}
      check(count*6<=16,"live RF allocations exceed PE capacity");}
    used=array->live_spm_vectors(client);check(live==array->live_contexts(client),"matrix shared context count differs");
    check(used==live*arena_vectors&&used<=128,"global SPM resource conservation failure");
    peak_contexts=std::max(peak_contexts,live);peak_spm_vectors=std::max(peak_spm_vectors,used);
  }
  bool done() const {return retired==total_blocks;}
  bool tick() {
    if(done()){check(!dma&&!spm_pending,"done with pending shared response");for(unsigned p=0;p<pes;++p)check(!fu[p],"done with pending compute");return false;}
    check(cycles<options.max_cycles,"matrix scheduler exceeded cycle bound");
    if(owned_array)array->begin_cycle(cycles);
    array->enter(client);const auto edge=array->cycle();
    memory_port->advance(cycles);
    if(array_version!=array->version())resources_dirty=true;
    if(resources_dirty||!options.cache_control)recompute_control();
    const auto &ids=active_order;
    bool port=array->spm_ready();
    std::array<bool,16> write_port{};for(unsigned p=0;p<pes;++p)write_port[p]=array->writeback_ready(p);
    bool finish_spm=false,finish_dma=false;std::array<bool,16> finish_fu{};
    // Decisions use beginning-of-cycle state; completion/admission commits are
    // applied only after issue selection. New data is issuable next cycle.
    if(spm_pending&&spm_pending->due<=cycles){auto &c=validate(spm_pending->owner);
      if(store_op(spm_pending->instruction.op)){if(port){array->claim_spm(c.lease);port=false;finish_spm=true;}}
      else if(write_port[c.pe]){array->claim_writeback(c.lease);write_port[c.pe]=false;finish_spm=true;}else ++writeback_stalls;}
    check(!dma||array->unit_owned_by(shared_array::Unit::Dma,client),"matrix lost shared DMA service ownership");
    const auto incoming=(array->unit_owned_by(shared_array::Unit::Dma,client)||array->unit_ready(shared_array::Unit::Dma))?memory_port->response():std::nullopt;
    if(held_response)check(incoming&&incoming->id==held_response->id&&incoming->data==held_response->data&&incoming->error==held_response->error,"matrix memory response changed or withdrew under backpressure");
    if(incoming){check(bool(dma),"unsolicited matrix memory response");check(incoming->id==dma->id,"stale matrix memory response token");check(!incoming->error,"matrix memory response reported error");validate(dma->owner);held_response=incoming;
      if(edge%options.dma_response_period==0){if(dma->write)finish_dma=true;else if(port){array->claim_spm(contexts[dma->owner.context].lease);port=false;finish_dma=true;}}}
    for(unsigned p=0;p<pes;++p)if(fu[p]&&fu[p]->due<=cycles){validate(fu[p]->owner);if(write_port[p]){array->claim_writeback(contexts[fu[p]->owner.context].lease);write_port[p]=false;finish_fu[p]=true;}else ++writeback_stalls;}
    for(unsigned p=0;p<pes;++p)if(fu[p]){++compute_busy_pe_cycles;
      if(dma)++compute_dma_overlap;
      const bool other_dma=dma&&contexts[dma->owner.context].pe==p&&dma->owner.context!=fu[p]->owner.context;
      const bool other_spm=spm_pending&&contexts[spm_pending->owner.context].pe==p&&spm_pending->owner.context!=fu[p]->owner.context;
      if(other_dma||other_spm)++same_pe_overlap;}
    if(dma)++dma_busy_cycles;
    if(spm_pending)++spm_busy_cycles;
    std::vector<PendingInstruction> issues;std::vector<unsigned> predicated;
    std::array<bool,16> selected{};bool new_spm=false;
    for(unsigned id:ids){auto &c=contexts[id];if(!eligible(id)||!instruction_phase(c)||c.busy||selected[c.pe]||!array->issue_ready(c.pe))continue;
      const auto i=front(c);
      if(i.row<2&&i.row>=c.rows){array->claim_issue(c.lease);selected[c.pe]=true;predicated.push_back(id);continue;}
      if(memory_op(i.op)){if(spm_pending||new_spm||!port||!array->unit_ready(shared_array::Unit::Spm))continue;array->claim_spm(c.lease);array->claim_unit(shared_array::Unit::Spm,c.lease);port=false;new_spm=true;}
      else{if(fu[c.pe]||cycles<next_compute[c.pe]||!array->unit_ready(shared_array::Unit::Compute,c.pe))continue;array->claim_unit(shared_array::Unit::Compute,c.lease);}
      array->claim_issue(c.lease);selected[c.pe]=true;issues.push_back(PendingInstruction{owner(id),i,evaluate(c,i),cycles+latency(i.op)});
    }
    std::optional<PendingDMA> new_dma;
    if(!dma&&array->unit_ready(shared_array::Unit::Dma)&&edge%options.dma_request_period==0&&memory_port->request_ready()){for(unsigned id:ids){auto &c=contexts[id];if(!eligible(id)||instruction_phase(c)||c.memory_busy)continue;
      if(c.phase==Drain&&!port)continue;
      if(c.phase==Drain){array->claim_spm(c.lease);port=false;}
      array->claim_unit(shared_array::Unit::Dma,c.lease);new_dma=request(id);break;}}
    // Copy the offer before completion commits, preserving the old rule that
    // resources freed this cycle are not available to same-cycle admission.
    auto admission=admission_offer;
    if(next_block<total_blocks&&!admission)++admission_stalls;
    if(finish_spm){array->complete_unit(shared_array::Unit::Spm,contexts[spm_pending->owner.context].lease);complete_instruction(*spm_pending,"spm");spm_pending.reset();}
    if(finish_dma){auto pending=*dma;auto &c=validate(pending.owner);check(c.memory_busy,"DMA response has no request owner");
      if(pending.write){dma_write_bytes+=pending.bytes;numeric.memory_write_bytes+=pending.bytes;}
      else{pending.data=incoming->data;std::memcpy(spm_data(c,pending.spm,pending.bytes),&pending.data,pending.bytes);dma_read_bytes+=pending.bytes;numeric.memory_read_bytes+=pending.bytes;}
      ++dma_responses;dma_event("dma_response",pending);dma.reset();memory_port->consume_response();held_response.reset();array->complete_unit(shared_array::Unit::Dma,c.lease);finish_memory(pending.owner.context);
    }
    for(unsigned p=0;p<pes;++p)if(finish_fu[p]){array->complete_unit(shared_array::Unit::Compute,contexts[fu[p]->owner.context].lease);complete_instruction(*fu[p],"compute");fu[p].reset();}
    for(unsigned id:predicated){auto &c=contexts[id];auto i=front(c);++numeric.instructions;++numeric.opcode_counts[i.op];++numeric.inactive_row_instructions;event("predicated_issue",id,"control",i.op);finish_instruction(id);}
    for(auto &pending:issues){auto &c=validate(pending.owner);c.busy=true;unsigned op=pending.instruction.op;++numeric.instructions;++numeric.opcode_counts[op];
      if(op==Mul)numeric.mul_lanes+=c.lanes;
      if(op==Add)numeric.add_lanes+=c.lanes;
      event("issue",pending.owner.context,memory_op(op)?"spm":"compute",op);
      if(memory_op(op))spm_pending=pending;else{fu[c.pe]=pending;next_compute[c.pe]=cycles+options.compute_ii;}}
    if(new_dma){auto &c=validate(new_dma->owner);c.memory_busy=true;++dma_requests;dma_event("dma_request",*new_dma);
      check(next_request_id!=UINT64_MAX,"matrix memory request token exhausted");
      memory_port->submit(model_io::Request{new_dma->id,new_dma->offset,new_dma->data,new_dma->region,new_dma->bytes,new_dma->write});++next_request_id;
      if(options.inject_stale_dma_epoch&&!stale_injected){++new_dma->owner.epoch;stale_injected=true;}dma=new_dma;}
    if(admission){auto admitted_lease=array->admit(*admission,next_block);check(bool(admitted_lease),"matrix shared admission offer changed before commit");const auto lease=*admitted_lease;unsigned id=lease.pe*2+lease.slot;auto &c=contexts[id];uint64_t epoch=c.epoch+1;c=Context{};c.active=true;c.pe=lease.pe;c.slot=lease.slot;c.lease=lease;c.epoch=epoch;c.block=next_block++;
      c.row_base=(c.block/((n+15)/16))*2;c.col_base=(c.block%((n+15)/16))*16;c.rows=unsigned(std::min<uint64_t>(2,m-c.row_base));c.lanes=unsigned(std::min<uint64_t>(16,n-c.col_base));c.arena=lease.spm_base;
      for(unsigned r=0;r<6;++r)reg(c,r).type=0;
      ++admitted;++numeric.tiles;resources_dirty=true;event("admit",id);
    }
    if(resources_dirty||!options.cache_control)invariant();
    if(owned_array)array->end_cycle();
    ++cycles;return true;
  }
  Json::Value result() const {
    Json::Value r(Json::objectValue);r["classification"]="cpp_matrix_window_cycle_component_not_chipyard_model_validation";
    r["done"]=done();r["cycles"]=Json::UInt64(cycles);r["admitted"]=Json::UInt64(admitted);r["retired"]=Json::UInt64(retired);
    r["peak_contexts"]=peak_contexts;r["peak_spm_vectors"]=peak_spm_vectors;r["spm_capacity_vectors"]=128;r["rf_frame_vectors"]=6;r["rom_words"]=rom_words;
    r["same_pe_context_overlap_cycles"]=Json::UInt64(same_pe_overlap);r["compute_dma_overlap_pe_cycles"]=Json::UInt64(compute_dma_overlap);
    r["writeback_stall_cycles"]=Json::UInt64(writeback_stalls);r["admission_stall_cycles"]=Json::UInt64(admission_stalls);
    r["dma_requests"]=Json::UInt64(dma_requests);r["dma_responses"]=Json::UInt64(dma_responses);
    r["dma_read_bytes"]=Json::UInt64(dma_read_bytes);r["dma_write_bytes"]=Json::UInt64(dma_write_bytes);
    r["dma_busy_cycles"]=Json::UInt64(dma_busy_cycles);r["compute_busy_pe_cycles"]=Json::UInt64(compute_busy_pe_cycles);r["spm_busy_cycles"]=Json::UInt64(spm_busy_cycles);
    r["pending_dma"]=bool(dma);r["pending_spm"]=bool(spm_pending);unsigned pending_fu=0;for(unsigned p=0;p<pes;++p)pending_fu+=bool(fu[p]);r["pending_compute"]=pending_fu;
    unsigned active=0,allocated=array->live_spm_vectors(client);for(const auto &c:contexts)active+=c.active;
    r["active_contexts"]=active;r["allocated_spm_vectors"]=allocated;
    r["numeric_instructions"]=numeric.json();r["trace"]=trace;r["blas_calls"]=0;r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;
    r["external_memory_port"]=external_memory;
    r["host_optimization"]["cache_control"]=options.cache_control;
    r["host_optimization"]["control_recomputations"]=Json::UInt64(control_recomputations);
    r["host_optimization"]["invariant_checks"]=Json::UInt64(invariant_checks);
    r["counter_units"]["cycles"]="wall_cycle";
    r["counter_units"]["same_pe_context_overlap_cycles"]="pe_cycle";
    r["counter_units"]["compute_dma_overlap_pe_cycles"]="pe_cycle";
    r["counter_units"]["writeback_stall_cycles"]="blocked_response_cycle";
    r["counter_units"]["admission_stall_cycles"]="wall_cycle";
    r["counter_units"]["dma_busy_cycles"]="wall_cycle";
    r["counter_units"]["spm_busy_cycles"]="wall_cycle";
    r["counter_units"]["compute_busy_pe_cycles"]="pe_cycle";
    if(!owned_array){r["external_shared_array"]=true;r["shared_client"]=Json::UInt64(client);r["source_operator_id"]=Json::UInt64(source_id);}
    return r;
  }
};

Simulator::Simulator(const Json::Value &p,Tensor a,Tensor b,const Tensor *bias,bool tb,uint64_t ab,uint64_t bb,
                     uint64_t m,uint64_t n,uint64_t k,Tensor out,uint64_t ob,Options o,model_io::MemoryPort *port,shared_array::Resources *array,uint64_t source)
    :impl(std::make_unique<Impl>(p,std::move(a),std::move(b),bias,tb,ab,bb,m,n,k,std::move(out),ob,o,port,array,source)){}
Simulator::~Simulator()=default;
bool Simulator::tick(){return impl->tick();}
bool Simulator::done()const{return impl->done();}
Json::Value Simulator::result()const{return impl->result();}
} // namespace mlx::matrix_schedule
