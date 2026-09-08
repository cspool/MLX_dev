#include "vector_schedule.h"
#include "../model_io/tensor_memory_port.h"
#include "mlx_tagged_simulator.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace mlx::vector_schedule {
using namespace tensor_model;
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
float floating(uint32_t raw){float value;std::memcpy(&value,&raw,4);return value;}
uint32_t bits(float value){uint32_t raw;std::memcpy(&raw,&value,4);return raw;}
enum Op:unsigned{LoadA16=2,LoadB16=3,CvtUp=4,Mul=5,Add=6,CvtDown=7,Store16=8,LoadA32=9,LoadB32=10,Store32=11,
                 Constant=14,Neg=15,Sub=16,Exp=17,Div=18,Sqrt=19,Cos=20,Sin=21,Shuffle=22,Max=23,Move=24,Broadcast=25,LoadStack=26,StoreStack=27};
struct Instruction{unsigned op,dst,a,b,imm;};
using Register=shared_array::Register;
struct Operand{Tensor tensor;bool literal=false;float value=0;DType type=DType::F32;};
struct Context{
  bool active=false,busy=false,memory_busy=false,stored=false;
  bool flow_ready=true;
  unsigned pe=0,slot=0,arena=0,pc=0,lanes=0,valid_count=0,level=0,stack_valid=0,move=0;
  unsigned fill=0,operand=0; // 0=none, 1=initialize, 2=read, 3=ready
  uint64_t block=0,epoch=0,row=0,tile=0,base=0,output_base=0,out_tile=0;
  bool maximum=false;
  std::string phase;
  shared_array::Lease lease;
};
struct Owner{unsigned id,pc,level,move;uint64_t block,epoch,base;std::string phase;};
struct Pending{Owner owner;Instruction instruction;Register value;unsigned mask=0;uint64_t due=0;};
struct PendingMemory{Owner owner;Request request;unsigned spm=0;};
bool load_operand(unsigned op){return op==LoadA16||op==LoadB16||op==LoadA32||op==LoadB32;}
bool store(unsigned op){return op==Store16||op==Store32||op==StoreStack;}
bool memory(unsigned op){return load_operand(op)||store(op)||op==LoadStack;}
bool trans(unsigned op){return op>=Exp&&op<=Sin;}
uint64_t broadcast_index(uint64_t flat,const Shape &out,const Shape &in){
  check(in.size()<=out.size(),"vector schedule broadcast rank mismatch");uint64_t result=0,step=1;
  for(size_t d=out.size();d-->0;){uint64_t at=out[d]?flat%out[d]:0;if(out[d])flat/=out[d];if(d+in.size()>=out.size()){auto size=in[d+in.size()-out.size()];check(size==1||size==out[d],"vector schedule broadcast extent mismatch");if(size!=1)result+=at*step;step*=size;}}return result;
}
std::vector<Instruction> decode(const Json::Value &words){
  check(words.isArray()&&!words.empty()&&words.size()<=32,"vector schedule ROM capacity violation");std::vector<Instruction> result;
  for(const auto &raw:words){check(raw.isUInt(),"vector schedule word is not unsigned 32-bit");unsigned w=raw.asUInt();Instruction i{w&255,(w>>8)&15,(w>>12)&15,(w>>16)&15,(w>>20)&15};
    check(!(w>>24)&&i.op>=2&&i.op<=27&&i.op!=12&&i.op!=13&&i.dst<8&&i.a<8&&i.b<8,"vector schedule opcode/register field violation");
    if(i.op==Mul||i.op==Add||i.op==Sub||i.op==Div||i.op==Max){}
    else if(load_operand(i.op)||i.op==Constant||i.op==LoadStack)check(i.a==0&&i.b==0,"noncanonical vector load word");
    else if(store(i.op))check(i.dst==0&&i.b==0,"noncanonical vector store word");
    else check(i.b==0,"noncanonical vector unary word");
    if(i.op==Shuffle)check(i.imm==1||i.imm==2||i.imm==4||i.imm==8,"invalid vector shuffle distance");
    else if(trans(i.op))check(i.imm<4,"vector SFU exceeds quarter width");
    else if(load_operand(i.op))check(i.imm<=1,"invalid vector load padding");
    else if(i.op!=Constant)check(i.imm==0,"noncanonical vector immediate");
    result.push_back(i);
  }return result;
}
} // namespace

Options Options::parse(const Json::Value &v){
  Options o;check(v.isNull()||v.isObject(),"vector timing options must be an object");
  const std::set<std::string> allowed={"rows","columns","contexts","dma_latency","spm_latency","multiply_latency","add_latency","convert_latency","exp_latency","div_latency","sqrt_latency","dma_request_period","dma_response_period","spm_period","writeback_period","vector_ii","trans_ii","trace_limit","max_cycles","overlap","trace","inject_stale_response"};
  if(v.isObject())for(const auto &key:v.getMemberNames())check(allowed.count(key),"unknown vector timing option");
  auto set=[&](const char *key,unsigned &field){if(v.isMember(key)){check(v[key].isUInt(),"noninteger vector timing option");field=v[key].asUInt();}};
  set("rows",o.rows);set("columns",o.columns);set("contexts",o.contexts);set("dma_latency",o.dma_latency);set("spm_latency",o.spm_latency);
  set("multiply_latency",o.multiply_latency);set("add_latency",o.add_latency);set("convert_latency",o.convert_latency);set("exp_latency",o.exp_latency);set("div_latency",o.div_latency);set("sqrt_latency",o.sqrt_latency);
  set("dma_request_period",o.dma_request_period);set("dma_response_period",o.dma_response_period);set("spm_period",o.spm_period);set("writeback_period",o.writeback_period);set("vector_ii",o.vector_ii);set("trans_ii",o.trans_ii);set("trace_limit",o.trace_limit);
  if(v.isMember("max_cycles")){check(v["max_cycles"].isUInt64(),"invalid vector cycle bound");o.max_cycles=v["max_cycles"].asUInt64();}
  for(const char *key:{"overlap","trace","inject_stale_response"})if(v.isMember(key))check(v[key].isBool(),"vector timing flags must be Boolean");
  if(v.isMember("overlap"))o.overlap=v["overlap"].asBool();
  if(v.isMember("trace"))o.trace=v["trace"].asBool();
  if(v.isMember("inject_stale_response"))o.inject_stale_response=v["inject_stale_response"].asBool();
  o.validate();return o;
}
void Options::validate()const{
  check(rows>=1&&rows<=4&&columns>=1&&columns<=4&&contexts>=1&&contexts<=2,"vector geometry/RF context capacity exceeded");
  for(unsigned n:{dma_latency,spm_latency,multiply_latency,add_latency,convert_latency,exp_latency,div_latency,sqrt_latency,dma_request_period,dma_response_period,spm_period,writeback_period,vector_ii,trans_ii})check(n>=1&&n<=1024,"vector latency/ready/II outside [1,1024]");
  check(max_cycles>0&&max_cycles<=1000000000000ULL&&(!trace||(trace_limit>0&&trace_limit<=10000000)),"invalid vector cycle/trace bound");
}

struct Simulator::Impl{
  Json::Value node,program;
  Options options;
  Tensor output;
  std::vector<Operand> operands;
  std::vector<Instruction> rom;
  std::array<Context,32> contexts{};
  std::unique_ptr<shared_array::Resources> owned_array;
  shared_array::Resources *array=nullptr;
  shared_array::Client client=0;
  uint64_t source_id=UINT64_MAX;
  model_events::BlockFlow *flow=nullptr;
  uint64_t dependency_wait_cycles=0;
  std::array<std::optional<Pending>,16> vector_fu,trans_fu;
  std::array<uint64_t,16> next_vector{},next_trans{};
  std::optional<Pending> spm_pending;
  std::optional<PendingMemory> dma;
  std::optional<Response> held_response;
  std::unique_ptr<MemoryPort> internal;
  MemoryPort *port;
  bool external=false,reduction=false,softmax=false,stale_injected=false;
  uint64_t cycles=0,next_block=0,retired=0,total_blocks=0,width=0,chunks=0,padded_chunks=0,next_request=1;
  unsigned pes=0,reduce_lanes=0,peak_contexts=0,peak_spm=0;
  uint64_t dma_requests=0,dma_responses=0,writeback_stalls=0,admission_stalls=0,same_pe_overlap=0,vec_sfu_overlap=0;
  uint64_t vector_busy=0,trans_busy=0,dma_busy=0,spm_busy=0;
  vector_model::Stats numeric;
  Json::Value trace{Json::arrayValue};

  Impl(const Json::Value &n,const Values &values,Tensor out,Options o,MemoryPort *p,shared_array::Resources *shared,uint64_t source,model_events::BlockFlow *block_flow)
      :node(n),program(n["vector_program"]),options(o),output(std::move(out)),port(p),external(p!=nullptr){
    options.validate();pes=options.rows*options.columns;
    check(program["profile"]=="mlx-vector-fp32-v1"&&program["kind"]==node["kind"]&&vector_model::supports(node["kind"].asString()),"invalid scheduled vector profile/kind");
    check(program["lanes"]==16&&program["trans_lanes"]==4&&program["rf_vectors"]==16&&program["rf_vector_bytes"]==64&&program["rf_vectors_used"]==8&&program["spm_bytes"]==8192&&program["spm_bytes_used"]==320&&program["rom_words"]==32,"scheduled vector resource descriptor mismatch");
    rom=decode(program["rom"]);check(program["phases"].isObject(),"vector phases are missing");
    for(const auto &key:program["phases"].getMemberNames()){check(program["phases"][key].isArray()&&!program["phases"][key].empty(),"empty/invalid vector phase");for(const auto &index:program["phases"][key])check(index.isUInt()&&index.asUInt()<rom.size(),"vector phase points outside ROM");}
    unsigned expected_inputs=node["kind"]=="add"||node["kind"]=="mul"?2:1;
    check(program["input_dtypes"].isArray()&&program["input_dtypes"].size()==expected_inputs,"vector operand count mismatch");
    if(node["kind"]=="pow")check(node["args"].size()==2&&scalar(node["args"][1])==2,"scheduled vector pow requires exponent 2");
    std::array<Tensor,3> regions;
    for(unsigned index=0;index<expected_inputs;++index){const auto &arg=node["args"][index];Operand input;
      if(arg.isObject()&&arg.isMember("value")){auto found=values.find(arg["value"].asString());check(found!=values.end(),"unbound vector schedule operand");input.tensor=found->second;input.type=input.tensor.type;regions[index]=input.tensor;}
      else{input.literal=true;input.value=float(scalar(arg));}
      check((input.type==DType::F16||input.type==DType::F32)&&program["input_dtypes"][index]==dtype_name(input.type),"vector schedule input precision mismatch");operands.push_back(input);
    }
    check(output.sizes==shape(node["output"]["shape"])&&dtype_name(output.type)==program["output_dtype"].asString()&&output.contiguous()&&output.offset==0&&(external||output.storage->writable),"vector schedule output binding mismatch");
    check(output.type==DType::F16||output.type==DType::F32,"vector schedule output dtype unsupported");regions[2]=output;
    reduction=node["kind"]=="mean"||node["kind"]=="softmax";softmax=node["kind"]=="softmax";
    if(reduction){check(!operands[0].literal&&!operands[0].tensor.sizes.empty(),"vector reduction requires a tensor");const auto &input=operands[0].tensor;width=input.sizes.back();
      check(width>0&&width<=(1ULL<<31)&&program["width"].asUInt64()==width,"vector reduction width mismatch");auto expected=input.sizes;
      if(!softmax){check(node["args"][1].isArray()&&node["args"][1].size()==1,"mean requires one axis");auto axis=node["args"][1][0].asInt64();check(axis==-1||axis==int64_t(input.sizes.size()-1),"mean requires the last axis");if(node["args"].size()>2&&node["args"][2].asBool())expected.back()=1;else expected.pop_back();}
      else{auto axis=node["args"][1].asInt64();check(axis==-1||axis==int64_t(input.sizes.size()-1),"softmax requires the last axis");}
      check(expected==output.sizes,"vector reduction output shape mismatch");total_blocks=input.numel()/width;chunks=(width+15)/16;padded_chunks=1;while(padded_chunks<chunks)padded_chunks*=2;reduce_lanes=1;while(reduce_lanes<std::min<uint64_t>(16,width))reduce_lanes*=2;
    }else total_blocks=(output.numel()+15)/16;
    check(std::numeric_limits<float>::is_iec559&&sizeof(float)==4&&std::fegetround()==FE_TONEAREST,"vector schedule requires IEEE FP32 RNE");
#if defined(__SSE__)
    check((_mm_getcsr()&((1u<<15)|(1u<<6)))==0,"vector schedule forbids FTZ/DAZ");
#endif
    if(!port){internal=std::make_unique<model_io::TensorMemoryPort>(std::vector<Tensor>(regions.begin(),regions.end()),options.dma_latency);port=internal.get();}
    numeric.calls=1;numeric.max_rom=rom.size();
    shared_array::Hardware hardware=shared?shared->hardware():shared_array::Hardware{};
    hardware.rows=options.rows;hardware.columns=options.columns;hardware.contexts=options.contexts;
    hardware.spm_period=options.spm_period;hardware.writeback_period=options.writeback_period;hardware.compute_ii=options.vector_ii;hardware.sfu_ii=options.trans_ii;
    hardware.dma_request_period=options.dma_request_period;hardware.dma_response_period=options.dma_response_period;
    hardware.multiply_latency=options.multiply_latency;hardware.add_latency=options.add_latency;hardware.convert_latency=options.convert_latency;hardware.spm_latency=options.spm_latency;
    hardware.exp_latency=options.exp_latency;hardware.div_latency=options.div_latency;hardware.sqrt_latency=options.sqrt_latency;
    if(shared)check(hardware==shared->hardware(),"vector frontend differs from shared physical hardware");
    else owned_array=std::make_unique<shared_array::Resources>(hardware);
    array=shared?shared:owned_array.get();source_id=source;
    if(source_id==UINT64_MAX&&node["source_operator_id"].isUInt64())source_id=node["source_operator_id"].asUInt64();
    std::vector<uint32_t> words;for(const auto &word:program["rom"])words.push_back(word.asUInt());
    client=array->attach("vector:"+program["phases"].toStyledString(),words,8,5,source_id);
    flow=block_flow;if(flow){check(shared&&p,"vector block events require shared external execution");array->residency_limit(client,flow->per_pe_limit(),flow->total_limit());}
  }
  ~Impl(){if(client)array->detach(client,done());}
  Owner owner(unsigned id)const{const auto &c=contexts[id];return Owner{id,c.pc,c.level,c.move,c.block,c.epoch,c.base,c.phase};}
  Context &validate(const Owner &o){auto &c=contexts[o.id];check(c.active&&c.block==o.block&&c.epoch==o.epoch&&c.pc==o.pc&&c.level==o.level&&c.move==o.move&&c.base==o.base&&c.phase==o.phase,"stale or mismatched vector completion owner");return c;}
  Register &reg(Context &c,unsigned r){return array->reg(c.lease,r);}
  uint8_t *spm_data(const Context &c,unsigned absolute,unsigned bytes){check(absolute>=c.arena*64,"vector SPM address precedes owned arena");return array->spm(c.lease,absolute-c.arena*64,bytes);}
  void need(Context &c,unsigned r,unsigned type,unsigned mask){auto &v=reg(c,r);check(v.type==type&&(v.valid&mask)==mask,"vector issue reads invalid register type/readiness");}
  Instruction front(const Context &c)const{check(program["phases"].isMember(c.phase),"unknown vector execution phase");auto w=array->instruction(c.lease,program["phases"][c.phase][c.pc].asUInt());return Instruction{w&255,(w>>8)&15,(w>>12)&15,(w>>16)&15,(w>>20)&15};}
  void event(const char *name,unsigned id,const char *pipeline="control",unsigned opcode=0,unsigned mask=0){if(!options.trace)return;check(trace.size()<options.trace_limit,"vector trace bound exceeded");const auto &c=contexts[id];Json::Value e(Json::objectValue);
    e["cycle"]=Json::UInt64(cycles);e["event"]=name;e["pe"]=c.pe;e["context_slot"]=c.slot;e["block_id"]=Json::UInt64(c.block);e["epoch"]=Json::UInt64(c.epoch);e["phase"]=c.phase;e["pc"]=c.pc;e["base"]=Json::UInt64(c.base);e["level"]=c.level;e["move"]=c.move;e["pipeline"]=pipeline;e["opcode"]=opcode;e["lane_mask"]=mask;
    if(!owned_array){e["array_cycle"]=Json::UInt64(array->cycle());e["shared_client"]=Json::UInt64(client);e["source_operator_id"]=Json::UInt64(source_id);e["lease_id"]=Json::UInt64(c.lease.id);
      if(std::strcmp(name,"admit")==0){e["rf_base"]=c.lease.rf_base;e["rf_vectors"]=c.lease.rf_count;e["spm_base"]=c.lease.spm_base;e["spm_vectors"]=c.lease.spm_count;e["rom_base"]=c.lease.rom_base;e["rom_words"]=c.lease.rom_count;}}
    trace.append(e);
  }
  void memory_event(const char *name,const PendingMemory &p,uint64_t data){event(name,p.owner.id,"dma");if(!options.trace)return;auto &e=trace[trace.size()-1];e["request_id"]=Json::UInt64(p.request.id);e["region"]=p.request.region;e["byte_offset"]=Json::UInt64(p.request.offset);e["bytes"]=p.request.bytes;e["write"]=p.request.write;e["data"]=Json::UInt64(data);e["spm_byte_offset"]=p.spm;}
  bool eligible(unsigned id)const{const auto &c=contexts[id];if(!c.active||!c.flow_ready)return false;if(!options.overlap)for(unsigned slot=0;slot<options.contexts;++slot){const auto &other=contexts[c.pe*2+slot];if(other.active&&other.block<c.block)return false;}return true;}
  std::vector<unsigned> priority()const{std::vector<unsigned> ids;for(unsigned p=0;p<pes;++p)for(unsigned s=0;s<options.contexts;++s)if(contexts[p*2+s].active)ids.push_back(p*2+s);std::sort(ids.begin(),ids.end(),[&](unsigned a,unsigned b){return contexts[a].block<contexts[b].block;});return ids;}
  void phase(Context &c,const char *name){check(program["phases"].isMember(name),"required scheduled vector phase is missing");c.phase=name;c.pc=0;}
  void reduction_tile(Context &c){c.base=c.row*width+c.tile*16;c.valid_count=c.tile*16<width?unsigned(std::min<uint64_t>(16,width-c.tile*16)):0;c.lanes=reduce_lanes;c.fill=0;c.move=0;phase(c,c.maximum?"max_tile":"sum_tile");}
  void output_tile(Context &c){c.base=c.row*width+c.out_tile*16;c.output_base=c.base;c.valid_count=unsigned(std::min<uint64_t>(16,width-c.out_tile*16));c.lanes=c.valid_count;c.stored=false;c.fill=0;c.move=0;phase(c,"output_tile");}
  void advance(unsigned id){auto &c=contexts[id];c.busy=false;if(++c.pc<program["phases"][c.phase].size())return;c.pc=0;
    if(c.phase=="body"||c.phase=="final"||c.phase=="output_tile"){check(c.stored,"vector program omitted its output store");c.phase="drain";c.move=0;}
    else if(c.phase=="max_tile"||c.phase=="sum_tile"){c.lanes=1;phase(c,"to_carry");}
    else if(c.phase=="to_carry"){c.level=0;phase(c,(c.stack_valid&1)?(c.maximum?"merge_max":"merge_sum"):"save_carry");}
    else if(c.phase=="merge_max"||c.phase=="merge_sum"){c.stack_valid&=~(1u<<c.level);check(++c.level<32,"vector reduction stack overflow");phase(c,(c.stack_valid&(1u<<c.level))?(c.maximum?"merge_max":"merge_sum"):"save_carry");}
    else if(c.phase=="save_carry"){if(++c.tile<padded_chunks)reduction_tile(c);else{c.lanes=1;phase(c,"root");}}
    else if(c.phase=="root"){if(c.maximum)phase(c,"save_max");else if(softmax)phase(c,"save_sum");else{c.lanes=c.valid_count=1;c.output_base=c.row;c.stored=false;phase(c,"final");}}
    else if(c.phase=="save_max"){c.maximum=false;c.stack_valid=0;c.tile=0;reduction_tile(c);}
    else if(c.phase=="save_sum"){c.out_tile=0;output_tile(c);}
    else throw std::runtime_error("unregistered vector phase transition");
  }
  unsigned mask(const Context &c,const Instruction &i)const{unsigned first=trans(i.op)?i.imm*4:0,last=trans(i.op)?std::min(c.lanes,first+4):c.lanes;if(i.op==Move||i.op==LoadStack||i.op==StoreStack)return 1;return first<last?((1u<<(last-first))-1)<<first:0;}
  unsigned latency(unsigned op)const{if(memory(op))return options.spm_latency;if(op==Mul)return options.multiply_latency;if(op==Add||op==Sub||op==Max)return options.add_latency;if(op==CvtUp||op==CvtDown)return options.convert_latency;if(op==Exp||op==Cos||op==Sin)return options.exp_latency;if(op==Div)return options.div_latency;if(op==Sqrt)return options.sqrt_latency;return 1;}
  Register evaluate(Context &c,const Instruction &i,unsigned active){Register result;unsigned bytes=0;
    if(load_operand(i.op)){check(c.fill==3,"vector load issued before operand fill completed");auto &input=operands[c.operand];unsigned type=(i.op==LoadA16||i.op==LoadB16)?1:2;check(type==(input.type==DType::F16?1u:2u),"vector SPM load precision mismatch");result.type=type;bytes=element_bytes(input.type);for(unsigned lane=0;lane<c.lanes;++lane)std::memcpy(&result.data[lane],spm_data(c,c.arena*64+c.operand*64+lane*bytes,bytes),bytes);}
    else if(i.op==LoadStack){check(c.level<32&&(c.stack_valid&(1u<<c.level)),"vector load reads an unready reduction partial");result.type=2;std::memcpy(&result.data[0],spm_data(c,c.arena*64+128+c.level*4,4),4);}
    else if(store(i.op)){unsigned type=i.op==Store16?1:2;need(c,i.a,type,active);if(i.op!=StoreStack)check(output.type==(type==1?DType::F16:DType::F32),"vector output store precision mismatch");result=reg(c,i.a);}
    else if(i.op==Constant){check(i.imm<program["constants"].size(),"vector constant index out of bounds");result.type=2;result.data.fill(bits(float(scalar(program["constants"][i.imm]))));}
    else if(i.op==Move||i.op==Broadcast){need(c,i.a,2,1);result.type=2;result.data.fill(reg(c,i.a).data[0]);}
    else if(i.op==CvtUp||i.op==CvtDown){need(c,i.a,i.op==CvtUp?1:2,active);result.type=i.op==CvtUp?2:1;for(unsigned lane=0;lane<16;++lane)if(active&(1u<<lane))result.data[lane]=i.op==CvtUp?bits(tagged::half_to_float(uint16_t(reg(c,i.a).data[lane]))):tagged::float_to_half(floating(reg(c,i.a).data[lane]));}
    else{need(c,i.a,2,active);result.type=2;if(i.op==Shuffle){check(c.lanes&&(c.lanes&(c.lanes-1))==0&&i.imm<c.lanes,"vector shuffle reads an inactive lane");for(unsigned lane=0;lane<c.lanes;++lane)result.data[lane]=reg(c,i.a).data[lane^i.imm];}
      else{if(i.op==Mul||i.op==Add||i.op==Sub||i.op==Div||i.op==Max)need(c,i.b,2,active);for(unsigned lane=0;lane<16;++lane)if(active&(1u<<lane)){float a=floating(reg(c,i.a).data[lane]),b=floating(reg(c,i.b).data[lane]),value=0;switch(i.op){case Mul:value=a*b;break;case Add:value=a+b;break;case Sub:value=a-b;break;case Div:value=a/b;break;case Max:value=a<b?b:a;break;case Neg:value=-a;break;case Exp:value=std::exp(a);break;case Sqrt:value=std::sqrt(a);break;case Cos:value=std::cos(a);break;case Sin:value=std::sin(a);break;default:throw std::runtime_error("unknown scheduled vector arithmetic");}result.data[lane]=bits(value);}}}
    result.valid=active;return result;
  }
  void complete(Pending &p,const char *pipeline){auto &c=validate(p.owner);check(c.busy,"vector completion has no busy owner");auto op=p.instruction.op;
    if(op==StoreStack){check(c.level<32,"vector partial store outside stack");std::memcpy(spm_data(c,c.arena*64+128+c.level*4,4),&p.value.data[0],4);c.stack_valid|=1u<<c.level;numeric.max_stack_level=std::max(numeric.max_stack_level,c.level);}
    else if(op==Store16||op==Store32){check(!c.stored,"duplicate vector output store");auto bytes=element_bytes(output.type);for(unsigned lane=0;lane<c.valid_count;++lane)std::memcpy(spm_data(c,c.arena*64+256+lane*bytes,bytes),&p.value.data[lane],bytes);c.stored=true;}
    else{auto &destination=reg(c,p.instruction.dst);if(destination.type!=p.value.type)destination.valid=0;destination.type=p.value.type;for(unsigned lane=0;lane<16;++lane)if(p.mask&(1u<<lane))destination.data[lane]=p.value.data[lane];destination.valid|=p.mask;if(load_operand(op))c.fill=0;}
    event("complete",p.owner.id,pipeline,op,p.mask);advance(p.owner.id);
  }
  void initialize(unsigned id){auto &c=contexts[id];const auto &i=front(c);auto &input=operands[c.operand];auto bytes=element_bytes(input.type);for(unsigned lane=0;lane<c.lanes;++lane){float value=i.imm?-std::numeric_limits<float>::infinity():0;if(input.literal&&lane<c.valid_count)value=input.value;uint32_t raw=input.type==DType::F16?uint32_t(tagged::float_to_half(value)):bits(value);std::memcpy(spm_data(c,c.arena*64+c.operand*64+lane*bytes,bytes),&raw,bytes);}c.move=0;c.fill=input.literal||!c.valid_count?3:2;event(c.fill==3?"operand_ready":"operand_initialized",id,"spm_init");}
  PendingMemory request(unsigned id){auto &c=contexts[id];PendingMemory p;p.owner=owner(id);p.request.id=next_request;p.request.write=c.phase=="drain";
    if(p.request.write){p.request.region=2;p.request.bytes=element_bytes(output.type);p.request.offset=output.position(c.output_base+c.move)*p.request.bytes;p.spm=c.arena*64+256+c.move*p.request.bytes;std::memcpy(&p.request.data,spm_data(c,p.spm,p.request.bytes),p.request.bytes);}
    else{check(c.fill==2&&!operands[c.operand].literal,"invalid vector operand DMA state");const auto &input=operands[c.operand].tensor;uint64_t flat=reduction?c.base+c.move:broadcast_index(c.base+c.move,output.sizes,input.sizes);p.request.region=c.operand;p.request.bytes=element_bytes(input.type);p.request.offset=input.position(flat)*p.request.bytes;p.spm=c.arena*64+c.operand*64+c.move*p.request.bytes;}
    check(p.spm+p.request.bytes<=(c.arena+5)*64,"vector DMA exceeds owned SPM arena");return p;
  }
  void response(const PendingMemory &p,uint64_t data){auto &c=validate(p.owner);check(c.memory_busy,"vector DMA response has no request owner");c.memory_busy=false;++dma_responses;
    if(p.request.write)numeric.write_bytes+=p.request.bytes;else{std::memcpy(spm_data(c,p.spm,p.request.bytes),&data,p.request.bytes);numeric.read_bytes+=p.request.bytes;}
    memory_event("dma_response",p,p.request.write?p.request.data:data);
    if(++c.move<c.valid_count)return;
    c.move=0;
    if(!p.request.write){c.fill=3;event("operand_ready",p.owner.id);}
    else if(reduction&&softmax&&++c.out_tile<chunks)output_tile(c);
    else{event("retire",p.owner.id);check(!c.busy,"retiring vector context has in-flight instruction");for(unsigned r=0;r<8;++r)reg(c,r).valid=0;if(flow)flow->completed(c.block,c.lease.id,array->cycle());array->retire(c.lease);c.active=false;++retired;}
  }
  void invariant(){unsigned active=0,allocated=array->live_spm_vectors(client);for(unsigned p=0;p<pes;++p){unsigned used=0;for(unsigned s=0;s<options.contexts;++s){const auto &c=contexts[p*2+s];if(!c.active)continue;++active;++used;array->validate_lease(c.lease);}check(used*8<=16,"vector RF allocations exceed capacity");}
    check(active==array->live_contexts(client),"vector shared context count differs");
    check(allocated==active*5&&allocated<=128,"vector SPM resource conservation failure");peak_contexts=std::max(peak_contexts,active);peak_spm=std::max(peak_spm,allocated);
  }
  bool done()const{return retired==total_blocks;}
  bool tick(){
    if(done()){check(!dma&&!spm_pending,"vector done with pending shared response");for(unsigned p=0;p<pes;++p)check(!vector_fu[p]&&!trans_fu[p],"vector done with pending FU");return false;}
    check(cycles<options.max_cycles,"vector schedule exceeded cycle bound");if(owned_array)array->begin_cycle(cycles);array->enter(client);const auto edge=array->cycle();
    if(flow)for(unsigned id=0;id<contexts.size();++id){auto &c=contexts[id];if(c.active&&!c.flow_ready){if(flow->inputs_ready(c.block)){c.flow_ready=true;event("wake",id,"dependency");}else ++dependency_wait_cycles;}}
    port->advance(cycles);auto ids=priority();bool spm_port=array->spm_ready();std::array<bool,16> write_port{};for(unsigned p=0;p<pes;++p)write_port[p]=array->writeback_ready(p);
    bool finish_spm=false;std::array<bool,16> finish_vector{},finish_trans{};std::optional<Response> answer;
    if(spm_pending&&spm_pending->due<=cycles){auto &c=validate(spm_pending->owner);if(store(spm_pending->instruction.op)){if(spm_port){array->claim_spm(c.lease);spm_port=false;finish_spm=true;}}else if(write_port[c.pe]){array->claim_writeback(c.lease);write_port[c.pe]=false;finish_spm=true;}else ++writeback_stalls;}
    check(!dma||array->unit_owned_by(shared_array::Unit::Dma,client),"vector lost shared DMA service ownership");
    auto incoming=(array->unit_owned_by(shared_array::Unit::Dma,client)||array->unit_ready(shared_array::Unit::Dma))?port->response():std::nullopt;
    if(held_response)check(incoming&&incoming->id==held_response->id&&incoming->data==held_response->data&&incoming->error==held_response->error,"vector memory response changed or withdrew under backpressure");
    if(incoming){check(bool(dma),"unsolicited vector memory response");auto tested=*incoming;if(options.inject_stale_response&&!stale_injected){++tested.id;stale_injected=true;}check(tested.id==dma->request.id,"stale vector memory response token");check(!tested.error,"vector memory response reported error");validate(dma->owner);
      held_response=*incoming;
      if(edge%options.dma_response_period==0&&(dma->request.write||spm_port)){answer=tested;if(!dma->request.write){array->claim_spm(contexts[dma->owner.id].lease);spm_port=false;}}}
    for(unsigned p=0;p<pes;++p){if(vector_fu[p]&&vector_fu[p]->due<=cycles){validate(vector_fu[p]->owner);if(write_port[p]){array->claim_writeback(contexts[vector_fu[p]->owner.id].lease);write_port[p]=false;finish_vector[p]=true;}else ++writeback_stalls;}
      if(trans_fu[p]&&trans_fu[p]->due<=cycles){validate(trans_fu[p]->owner);if(write_port[p]){array->claim_writeback(contexts[trans_fu[p]->owner.id].lease);write_port[p]=false;finish_trans[p]=true;}else ++writeback_stalls;}
      if(vector_fu[p])++vector_busy;
      if(trans_fu[p])++trans_busy;
      if(vector_fu[p]&&trans_fu[p])++vec_sfu_overlap;
      std::set<unsigned> owners;if(vector_fu[p])owners.insert(vector_fu[p]->owner.id);if(trans_fu[p])owners.insert(trans_fu[p]->owner.id);if(spm_pending&&contexts[spm_pending->owner.id].pe==p)owners.insert(spm_pending->owner.id);if(dma&&contexts[dma->owner.id].pe==p)owners.insert(dma->owner.id);if(owners.size()>1)++same_pe_overlap;
    }
    if(dma)++dma_busy;
    if(spm_pending)++spm_busy;
    std::vector<unsigned> start_fill,predicated;std::vector<Pending> issues;std::array<bool,16> selected{};bool new_spm=false;
    for(unsigned id:ids){auto &c=contexts[id];if(!eligible(id)||c.busy||c.phase=="drain"||selected[c.pe]||!array->issue_ready(c.pe))continue;const auto &i=front(c);unsigned active=mask(c,i);
      if(!active){array->claim_issue(c.lease);selected[c.pe]=true;predicated.push_back(id);continue;}
      if(load_operand(i.op)&&c.fill!=3){if(!c.fill)start_fill.push_back(id);continue;}
      if(memory(i.op)){if(spm_pending||new_spm||!spm_port||!array->unit_ready(shared_array::Unit::Spm))continue;array->claim_spm(c.lease);array->claim_unit(shared_array::Unit::Spm,c.lease);new_spm=true;spm_port=false;}
      else if(trans(i.op)){if(trans_fu[c.pe]||cycles<next_trans[c.pe]||!array->unit_ready(shared_array::Unit::Sfu,c.pe))continue;array->claim_unit(shared_array::Unit::Sfu,c.lease);}
      else{if(vector_fu[c.pe]||cycles<next_vector[c.pe]||!array->unit_ready(shared_array::Unit::Compute,c.pe))continue;array->claim_unit(shared_array::Unit::Compute,c.lease);}
      array->claim_issue(c.lease);selected[c.pe]=true;issues.push_back(Pending{owner(id),i,evaluate(c,i,active),active,cycles+latency(i.op)});
    }
    std::optional<unsigned> initialize_id;std::optional<PendingMemory> new_dma;
    for(unsigned id:ids){auto &c=contexts[id];if(!eligible(id)||c.busy||c.memory_busy)continue;
      if(c.fill==1&&spm_port&&!initialize_id){array->claim_spm(c.lease);initialize_id=id;spm_port=false;}
      if(!dma&&!new_dma&&array->unit_ready(shared_array::Unit::Dma)&&edge%options.dma_request_period==0&&port->request_ready()&&(c.fill==2||(c.phase=="drain"&&spm_port))){if(c.phase=="drain"){array->claim_spm(c.lease);spm_port=false;}array->claim_unit(shared_array::Unit::Dma,c.lease);new_dma=request(id);}
    }
    std::optional<shared_array::Offer> admission;
    if(next_block<total_blocks){if(!flow||flow->admission_ready(next_block))admission=array->offer(client,unsigned(next_block%pes));if(!admission)++admission_stalls;}
    if(finish_spm){array->complete_unit(shared_array::Unit::Spm,contexts[spm_pending->owner.id].lease);complete(*spm_pending,"spm");spm_pending.reset();}
    if(answer){auto pending=*dma;array->complete_unit(shared_array::Unit::Dma,contexts[pending.owner.id].lease);response(pending,answer->data);dma.reset();port->consume_response();held_response.reset();}
    for(unsigned p=0;p<pes;++p){if(finish_vector[p]){array->complete_unit(shared_array::Unit::Compute,contexts[vector_fu[p]->owner.id].lease);complete(*vector_fu[p],"vector");vector_fu[p].reset();}if(finish_trans[p]){array->complete_unit(shared_array::Unit::Sfu,contexts[trans_fu[p]->owner.id].lease);complete(*trans_fu[p],"trans");trans_fu[p].reset();}}
    if(initialize_id)initialize(*initialize_id);
    for(unsigned id:start_fill){auto &c=contexts[id];const auto &i=front(c);c.operand=(i.op==LoadB16||i.op==LoadB32)?1:0;check(c.operand<operands.size(),"vector operand selector out of bounds");c.fill=1;c.move=0;event("operand_wait",id);}
    for(unsigned id:predicated){auto &c=contexts[id];const auto &i=front(c);++numeric.instructions;++numeric.opcode_counts[i.op];event("predicated_issue",id,"control",i.op,0);advance(id);}
    for(auto &pending:issues){auto &c=validate(pending.owner);c.busy=true;auto op=pending.instruction.op;++numeric.instructions;++numeric.opcode_counts[op];unsigned count=unsigned(__builtin_popcount(pending.mask));if(trans(op))numeric.trans_lanes+=count;if(op==Mul||op==Add||op==Sub||op==Div||op==Max||op==Neg||trans(op))numeric.arithmetic_lanes+=count;
      const char *pipeline=memory(op)?"spm":trans(op)?"trans":"vector";event("issue",pending.owner.id,pipeline,op,pending.mask);
      if(memory(op))spm_pending=pending;else if(trans(op)){trans_fu[c.pe]=pending;next_trans[c.pe]=cycles+options.trans_ii;}else{vector_fu[c.pe]=pending;next_vector[c.pe]=cycles+options.vector_ii;}}
    if(new_dma){auto &c=validate(new_dma->owner);c.memory_busy=true;memory_event("dma_request",*new_dma,new_dma->request.data);port->submit(new_dma->request);++next_request;++dma_requests;dma=*new_dma;}
    if(admission){auto admitted_lease=array->admit(*admission,next_block);check(bool(admitted_lease),"vector shared admission offer changed before commit");const auto lease=*admitted_lease;unsigned id=lease.pe*2+lease.slot;auto &c=contexts[id];auto epoch=c.epoch+1;c=Context{};c.active=true;c.epoch=epoch;c.pe=lease.pe;c.slot=lease.slot;c.lease=lease;c.arena=lease.spm_base;c.block=next_block++;for(unsigned r=0;r<8;++r)reg(c,r)=Register{};
      if(reduction){c.row=c.block;c.maximum=softmax;reduction_tile(c);}else{c.base=c.block*16;c.output_base=c.base;c.valid_count=c.lanes=unsigned(std::min<uint64_t>(16,output.numel()-c.base));phase(c,"body");}
      if(flow){flow->admitted(c.block,c.lease.id);c.flow_ready=flow->inputs_ready(c.block);}event("admit",id);if(!c.flow_ready)event("wait_event",id,"dependency");
    }
    invariant();if(owned_array)array->end_cycle();++cycles;return true;
  }
  Json::Value result()const{Json::Value r(Json::objectValue);r["classification"]="cpp_vector_window_cycle_component_not_full_system_validation";r["done"]=done();r["cycles"]=Json::UInt64(cycles);r["admitted"]=Json::UInt64(next_block);r["retired"]=Json::UInt64(retired);r["peak_contexts"]=peak_contexts;r["peak_spm_vectors"]=peak_spm;r["spm_capacity_vectors"]=128;r["rf_frame_vectors"]=8;r["spm_frame_vectors"]=5;
    r["dma_requests"]=Json::UInt64(dma_requests);r["dma_responses"]=Json::UInt64(dma_responses);r["writeback_stall_response_cycles"]=Json::UInt64(writeback_stalls);r["admission_stall_cycles"]=Json::UInt64(admission_stalls);r["same_pe_context_overlap_cycles"]=Json::UInt64(same_pe_overlap);r["vector_sfu_overlap_pe_cycles"]=Json::UInt64(vec_sfu_overlap);
    r["vector_busy_pe_cycles"]=Json::UInt64(vector_busy);r["trans_busy_pe_cycles"]=Json::UInt64(trans_busy);r["dma_busy_cycles"]=Json::UInt64(dma_busy);r["spm_busy_cycles"]=Json::UInt64(spm_busy);r["numeric_instructions"]=numeric.json();r["external_memory_port"]=external;r["pending_dma"]=bool(dma);r["pending_spm"]=bool(spm_pending);unsigned pending=0,active=0,allocated=array->live_spm_vectors(client);for(unsigned p=0;p<pes;++p)pending+=bool(vector_fu[p])+bool(trans_fu[p]);for(const auto &c:contexts)active+=c.active;r["pending_fu"]=pending;r["active_contexts"]=active;r["allocated_spm_vectors"]=allocated;r["trace"]=trace;r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;
    if(!owned_array){r["external_shared_array"]=true;r["shared_client"]=Json::UInt64(client);r["source_operator_id"]=Json::UInt64(source_id);}
    if(flow){r["block_flow"]=flow->description();r["dependency_wait_context_cycles"]=Json::UInt64(dependency_wait_cycles);}
    r["counter_units"]["cycles"]="wall_cycle";r["counter_units"]["same_pe_context_overlap_cycles"]="pe_cycle";r["counter_units"]["vector_sfu_overlap_pe_cycles"]="pe_cycle";r["counter_units"]["writeback_stall_response_cycles"]="blocked_response_cycle";return r;}
};
Simulator::Simulator(const Json::Value &n,const Values &v,Tensor out,Options o,MemoryPort *p,shared_array::Resources *array,uint64_t source,model_events::BlockFlow *flow):impl(std::make_unique<Impl>(n,v,std::move(out),o,p,array,source,flow)){}
Simulator::~Simulator()=default;
bool Simulator::tick(){return impl->tick();}
bool Simulator::done()const{return impl->done();}
Json::Value Simulator::result()const{return impl->result();}
} // namespace mlx::vector_schedule
