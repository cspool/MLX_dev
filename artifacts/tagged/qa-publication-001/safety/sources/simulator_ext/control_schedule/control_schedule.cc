#include "control_schedule.h"
#include "../model_io/tensor_memory_port.h"
#include "mlx_tagged_simulator.h"
#include <algorithm>
#include <cfenv>
#include <cstring>
#include <optional>
#include <set>
#include <stdexcept>

namespace mlx::control_schedule {
using namespace tensor_model;
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
int branch_offset(unsigned word){
  unsigned raw=((word>>31)&1)<<12|((word>>7)&1)<<11|((word>>25)&63)<<5|((word>>8)&15)<<1;
  return (raw&4096)?int(raw)-8192:int(raw);
}
Json::Value single(unsigned word){Json::Value words(Json::arrayValue);words.append(word);return words;}
void metadata(const Tensor &t){
  check(bool(t.storage)&&t.sizes.size()==t.steps.size()&&t.offset>=0,"controller tensor metadata missing/invalid");
  for(auto stride:t.steps)check(stride>=0,"negative controller tensor stride");
  if(t.numel())t.position(t.numel()-1);
}
const Tensor &ref(const Json::Value &v,const Values &values){
  check(v.isObject()&&v["value"].isString(),"expected controller SSA reference");auto it=values.find(v["value"].asString());check(it!=values.end(),"missing controller tensor input");return it->second;
}
bool reference(const Json::Value &value){return value.isObject()&&value.isMember("value");}
uint64_t broadcast(uint64_t flat,const Shape &out,const Shape &in){
  check(in.size()<=out.size(),"controller broadcast rank mismatch");uint64_t result=0,stride=1;
  for(size_t d=out.size();d-->0;){auto index=out[d]?flat%out[d]:0;if(out[d])flat/=out[d];
    if(d+in.size()>=out.size()){auto n=in[d+in.size()-out.size()];check(n==1||n==out[d],"controller broadcast extent mismatch");if(n!=1)result+=index*stride;stride*=n;}}
  return result;
}
float float_bits(uint32_t raw){float value;std::memcpy(&value,&raw,4);return value;}
} // namespace

void Leaf::begin(const Json::Value &words){
  check(words.isArray()&&!words.empty()&&words.size()<=32,"invalid stepped RV64 phase extent");
  for(unsigned i=0;i<words.size();++i){
    check(words[i].isUInt(),"RV64 phase word must be 32-bit");unsigned word=words[i].asUInt();
    if((word&127)==0x63){
      unsigned f=(word>>12)&7;check(f==0||f==1,"unsupported stepped RV64 branch");
      int target=int(i)*4+branch_offset(word);check(target>=0&&target<=int(words.size()*4)&&target%4==0,"RV64 branch leaves the bounded phase");
    }else{control_model::RV64 decoder;decoder.run(single(word));}
  }
  code=words;pc=0;steps=0;
}
bool Leaf::done()const{return pc==code.size();}
unsigned Leaf::word()const{check(!done(),"read past stepped RV64 phase");return code[pc].asUInt();}
void Leaf::step(){
  check(!done()&&++steps<=256,"stepped RV64 phase step limit");unsigned instruction=word();state.x[0]=0;
  if((instruction&127)==0x63){
    unsigned a=(instruction>>15)&31,b=(instruction>>20)&31,f=(instruction>>12)&7;
    bool taken=(state.x[a]==state.x[b])==(f==0);if(taken){pc=unsigned(int(pc)+branch_offset(instruction)/4);++state.branches_taken;}else ++pc;
    ++state.retired;
  }else{state.run(single(instruction));++pc;}
  state.x[0]=0;
}

void Options::validate()const{
  check(dma_latency&&alu_latency&&multiply_latency&&float_latency&&branch_latency&&request_period&&response_period&&max_cycles,"controller timing must be positive");
  check(max_cycles<UINT64_MAX-std::max({dma_latency,alu_latency,multiply_latency,float_latency,branch_latency}),"controller cycle bound overflows latency");
  check(trace_limit<=1000000,"controller trace bound exceeded");
}
Options Options::parse(const Json::Value &value){
  check(value.isObject(),"controller timing options must be an object");Options out;
  std::map<std::string,unsigned*> fields={{"dma_latency",&out.dma_latency},{"alu_latency",&out.alu_latency},{"multiply_latency",&out.multiply_latency},{"float_latency",&out.float_latency},{"branch_latency",&out.branch_latency},{"request_period",&out.request_period},{"response_period",&out.response_period},{"trace_limit",&out.trace_limit}};
  for(const auto &name:value.getMemberNames()){
    if(fields.count(name)){check(value[name].isUInt(),"controller timing must be unsigned");*fields[name]=value[name].asUInt();}
    else if(name=="max_cycles"){check(value[name].isUInt64(),"controller max_cycles must be unsigned");out.max_cycles=value[name].asUInt64();}
    else if(name=="trace"){check(value[name].isBool(),"controller trace must be Boolean");out.trace=value[name].asBool();}
    else throw std::runtime_error("unknown controller timing option");
  }
  out.validate();return out;
}

struct Simulator::Impl {
  Json::Value node,phases;
  std::string kind,phase;
  std::array<Tensor,3> tensors{};
  Options options;
  std::unique_ptr<model_io::TensorMemoryPort> local;
  model_io::MemoryPort *port;
  bool external=false,integer=false,complete=false,failed=false;
  uint64_t cycle=0,row=0,column=0,width=0,next_id=0,requests=0,responses=0,read_bytes=0,write_bytes=0;
  uint64_t instructions=0,branches=0,request_stalls=0,response_stalls=0,execution_stalls=0,trace_events=0;
  unsigned fflags=0;
  Leaf leaf;
  enum class Stage {Load0,Load1,Run,Store};
  Stage stage=Stage::Load0;
  struct Pending {model_io::Request request;uint64_t issue;};
  std::optional<Pending> pending;
  std::optional<model_io::Response> held;
  std::optional<uint64_t> execution_due;
  Json::Value events{Json::arrayValue};

  Impl(const Json::Value &n,const Values &values,Tensor output,Options o,model_io::MemoryPort *p)
      :node(n),kind(n["kind"].asString()),options(o),port(p),external(p!=nullptr){
    options.validate();const auto &program=node["control_program"],&args=node["args"];
    check(program["profile"]==control_model::profile(kind)&&program["kind"]==kind&&program["xlen"]==64&&program["flen"]==64&&program["gpr_count"]==32&&program["fpr_count"]==32,"controller profile/resource mismatch");
    check(program["input_dtype"]=="i64"||program["input_dtype"]=="f16"||program["input_dtype"]=="f32","controller input dtype not registered");
    check(std::fegetround()==FE_TONEAREST,"controller requires RNE");integer=program["input_dtype"]=="i64";
    check(kind=="arange"||kind=="add"||kind=="mul"||kind=="le"||kind=="argmax"||control_model::extended_kind(kind),"controller kind not supported");
    control_model::validate_extended_program(program,kind);
    tensors[2]=output;metadata(output);check(output.sizes==shape(node["output"]["shape"])&&dtype_name(output.type)==node["output"]["dtype"].asString(),"controller output contract mismatch");
    check(output.offset==0&&output.contiguous(),"controller output requires contiguous new storage");
    unsigned input_count=control_model::input_count(kind);
    for(unsigned i=0;i<input_count;++i){
      if(reference(args[i])){
        tensors[i]=ref(args[i],values);metadata(tensors[i]);auto type=tensors[i].type;
        check(integer?(type==DType::I64||type==DType::Bool):(type==DType::F16||type==DType::F32),"controller operand dtype mismatch");
        check(tensors[i].storage!=output.storage,"controller output aliases an input");
      }else if(integer)check(args[i].isInt64()||args[i].isBool(),"controller integer literal is not exact int64");
      else scalar(args[i]);
    }
    if(kind=="arange")check(integer&&output.type==DType::I64&&args.size()==1&&args[0].isUInt64()&&args[0].asUInt64()<=INT64_MAX&&output.sizes==Shape{args[0].asInt64()},"controller arange contract mismatch");
    else if(kind=="all"){
      check(args.size()==1&&reference(args[0])&&integer&&tensors[0].type==DType::Bool&&output.type==DType::Bool&&output.sizes.empty(),"controller all requires Boolean input and scalar output");
      width=tensors[0].numel();
    }else if(kind=="guard"){
      check(args.size()==2&&reference(args[0])&&integer&&tensors[0].type==DType::Bool&&tensors[0].numel()==1&&args[1].isBool()&&output.type==DType::Bool&&output.sizes.empty(),"controller guard requires one Boolean element and expected Boolean");
    }else if(kind=="argmax"){
      const auto &input=tensors[0];check(reference(args[0])&&args.size()>=2&&args.size()<=3&&!input.sizes.empty()&&input.sizes.back()>0&&output.type==DType::I64,"controller argmax contract mismatch");
      check(dtype_name(input.type)==program["input_dtype"].asString(),"argmax precision binding mismatch");auto axis=args[1].asInt64();check(axis==-1||axis==int64_t(input.sizes.size()-1),"controller argmax axis unsupported");
      auto expected=input.sizes;if(args.size()==3&&args[2].asBool())expected.back()=1;else expected.pop_back();check(expected==output.sizes,"argmax output shape mismatch");width=input.sizes.back();
    }else{
      const bool predicate=kind=="le"||kind=="ge"||kind=="bitwise_and";
      check(args.size()==2&&(predicate?output.type==DType::Bool:integer&&output.type==DType::I64),"controller elementwise type/arity mismatch");
      if(kind=="bitwise_and")check(integer&&reference(args[0])&&reference(args[1])&&tensors[0].type==DType::Bool&&tensors[1].type==DType::Bool,"controller bitwise_and requires Boolean tensors");
      check(!node["kwargs"].isMember("alpha")||scalar(node["kwargs"]["alpha"])==1,"controller integer add requires unit alpha");Shape expected;
      for(unsigned i=0;i<2;++i)if(reference(args[i])){
        const auto &input=tensors[i];if(expected.size()<input.sizes.size())expected.insert(expected.begin(),input.sizes.size()-expected.size(),1);auto shift=expected.size()-input.sizes.size();
        for(size_t d=0;d<input.sizes.size();++d){auto n=input.sizes[d];auto &e=expected[shift+d];if(e==1)e=n;else check(n==1||e==n,"controller broadcast shape mismatch");}
      }
      check(expected==output.sizes,"controller broadcast output mismatch");
    }
    phases=program["phases"];std::set<std::string> expected=kind=="arange"?std::set<std::string>{"init","body","advance"}:kind=="all"?std::set<std::string>{"init","body"}:kind=="argmax"?std::set<std::string>{"init","advance","compare","select"}:std::set<std::string>{"body"};
    auto names=phases.getMemberNames();check(std::set<std::string>(names.begin(),names.end())==expected,"controller phase set mismatch");
    unsigned count=0;for(const auto &name:names){leaf.begin(phases[name]);count+=phases[name].size();}check(count>0&&count<=32,"controller template capacity violation");
    leaf=Leaf{};
    if(!port){std::vector<Tensor> regions(tensors.begin(),tensors.end());local=std::make_unique<model_io::TensorMemoryPort>(std::move(regions),options.dma_latency);port=local.get();}
    // Even arange(0) executes the registered init leaf, as the functional path
    // does. Other empty outputs still validate code but perform no tensor I/O.
    if(kind=="arange"||kind=="all")begin("init");else complete=output.numel()==0;
  }
  void event(const char *name,const model_io::Request *request=nullptr){
    ++trace_events;if(!options.trace||events.size()>=options.trace_limit)return;
    Json::Value e;e["event"]=name;e["cycle"]=Json::UInt64(cycle);e["row"]=Json::UInt64(row);e["column"]=Json::UInt64(column);e["phase"]=phase;e["pc"]=leaf.position();
    if(request){e["request_id"]=Json::UInt64(request->id);e["region"]=request->region;e["offset"]=Json::UInt64(request->offset);e["bytes"]=request->bytes;e["write"]=request->write;e["data"]=Json::UInt64(request->data);}
    if(stage==Stage::Run&&!leaf.done())e["word"]=leaf.word();
    events.append(e);
  }
  void begin(const char *name){phase=name;leaf.begin(phases[name]);stage=Stage::Run;}
  void bind(unsigned operand,uint64_t bits){
    auto reg=10+operand;auto type=tensors[operand].type;
    if(integer)leaf.state.x[reg]=type==DType::Bool?uint64_t(uint8_t(bits)!=0):bits;
    else leaf.state.set_float(reg,type==DType::F16?tagged::half_to_float(uint16_t(bits)):float_bits(uint32_t(bits)));
  }
  void loaded(unsigned operand){
    if(kind=="argmax")begin(column?"advance":"init");
    else if(kind=="all")begin("body");
    else if(operand==0)stage=Stage::Load1;else begin("body");
  }
  void account(){instructions+=leaf.state.retired;branches+=leaf.state.branches_taken;fflags|=leaf.state.fflags;leaf.state.retired=leaf.state.branches_taken=0;}
  void phase_done(){
    account();
    if(kind=="arange"){
      if(phase=="init"){if(!tensors[2].numel())complete=true;else begin("body");}
      else if(phase=="body")stage=Stage::Store;
      else if(row==tensors[2].numel())complete=true;else begin("body");
    }else if(kind=="all"){
      if(phase=="body")++column;
      stage=column==width?Stage::Store:Stage::Load0;
    }else if(kind=="argmax"){
      if(phase=="advance")begin("compare");
      else if(phase=="compare")begin("select");
      else{++column;if(column==width)stage=Stage::Store;else stage=Stage::Load0;}
    }else{if(kind=="guard")check(leaf.state.x[13]==0,"control-flow guard mismatch");stage=Stage::Store;}
  }
  void store_done(){
    ++row;
    if(kind=="arange")begin("advance");
    else{
      complete=row==tensors[2].numel();leaf.state=control_model::RV64{};column=0;stage=Stage::Load0;
    }
  }
  bool can_request(){if(cycle%options.request_period||!port->request_ready()){++request_stalls;return false;}return true;}
  void request(unsigned region,uint64_t index,bool write){
    check(next_id!=UINT64_MAX,"controller request token exhausted");const auto &tensor=tensors[region];model_io::Request r;r.id=next_id++;r.region=region;r.bytes=element_bytes(tensor.type);r.offset=tensor.position(index)*r.bytes;r.write=write;
    if(write){r.data=leaf.state.x[kind=="argmax"?20:12];if(kind=="argmax")check(r.data<width,"controller argmax produced invalid index");if(tensor.type==DType::Bool)r.data=r.data!=0;}
    port->submit(r);pending=Pending{r,cycle};++requests;event("request",&r);
  }
  void load(unsigned operand){
    const auto &arg=node["args"][operand];
    if(reference(arg)){
      if(!can_request())return;
      request(operand,kind=="argmax"?row*width+column:kind=="all"?column:kind=="guard"?0:broadcast(row,tensors[2].sizes,tensors[operand].sizes),false);
    }else{
      if(integer)leaf.state.x[10+operand]=arg.isBool()?arg.asBool():uint64_t(arg.asInt64());else leaf.state.set_float(10+operand,float(scalar(arg)));
      event("literal");loaded(operand);
    }
  }
  void step(){
    check(!failed&&cycle<options.max_cycles,"controller faulted or exceeded max_cycles");port->advance(cycle);auto response=port->response();
    if(held)check(response&&response->id==held->id&&response->data==held->data&&response->error==held->error,"controller response changed or withdrew under backpressure");
    if(response){check(pending&&response->id==pending->request.id,"controller response owner mismatch");check(!response->error,"controller memory port reported error");held=response;}
    if(pending){
      if(response&&cycle%options.response_period==0){
        check(cycle>pending->issue,"controller consumed response on issue cycle");auto r=pending->request;
        if(r.write)write_bytes+=r.bytes;else{bind(r.region,response->data);read_bytes+=r.bytes;r.data=response->data;}
        event("response",&r);port->consume_response();pending.reset();held.reset();++responses;
        if(r.write)store_done();else loaded(r.region);
      }else ++response_stalls;
    }else if(execution_due){
      if(cycle>=*execution_due){event("instruction_complete");leaf.step();execution_due.reset();if(leaf.done())phase_done();}
      else ++execution_stalls;
    }else if(stage==Stage::Run){
      auto word=leaf.word();unsigned opcode=word&127,latency=opcode==0x63?options.branch_latency:opcode==0x53?options.float_latency:opcode==0x33&&(word>>25)==1?options.multiply_latency:options.alu_latency;
      execution_due=cycle+latency;event("instruction_issue");
    }else if(stage==Stage::Store){if(can_request())request(2,row,true);}
    else load(stage==Stage::Load0?0:1);
    ++cycle;
  }
  Json::Value report()const{
    check(complete&&!failed&&!pending&&!held&&!execution_due&&requests==responses,"controller result before drained completion");
    Json::Value r;r["classification"]="response_driven_controller_leaf_component_not_riscv_cpu";r["done"]=true;r["cycles"]=Json::UInt64(cycle);
    r["external_memory_port"]=external;r["dma_requests"]=Json::UInt64(requests);r["dma_responses"]=Json::UInt64(responses);r["read_bytes"]=Json::UInt64(read_bytes);r["write_bytes"]=Json::UInt64(write_bytes);
    r["instructions"]=Json::UInt64(instructions);r["branches_taken"]=Json::UInt64(branches);r["fflags_observed"]=fflags;
    r["request_stalls"]=Json::UInt64(request_stalls);r["response_stalls"]=Json::UInt64(response_stalls);r["execution_stalls"]=Json::UInt64(execution_stalls);
    r["trace"]=events;r["trace_events"]=Json::UInt64(trace_events);r["trace_truncated"]=options.trace&&trace_events>events.size();
    r["register_bytes"]=512;r["max_inflight_instructions"]=1;r["max_inflight_transactions"]=1;
    r["instruction_fetch"]="descriptor_rom_not_system_fetch";r["memory_binding"]="descriptor_port_not_riscv_load_store";
    if(control_model::extended_kind(kind))r["program_profile"]=node["control_program"]["profile"];
    r["rocket_execution_verified"]=false;r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;
    return r;
  }
};

Simulator::Simulator(const Json::Value &node,const Values &values,Tensor output,Options options,model_io::MemoryPort *port)
    :impl(std::make_unique<Impl>(node,values,output,options,port)){}
Simulator::~Simulator()=default;
bool Simulator::tick(){if(impl->complete)return false;try{impl->step();}catch(...){impl->failed=true;throw;}return !impl->complete;}
bool Simulator::done()const{return impl->complete&&!impl->failed;}
Json::Value Simulator::result()const{return impl->report();}
} // namespace mlx::control_schedule
