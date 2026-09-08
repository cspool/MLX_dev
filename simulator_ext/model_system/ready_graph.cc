#include "ready_graph.h"
#include "../model_io/physical_mux.h"
#include "../model_events/completion_window.h"
#include "matrix_schedule.h"
#include "vector_schedule.h"
#include "memory_schedule.h"
#include "control_schedule.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>

namespace mlx::model_system {
using namespace tensor_model;
namespace {
void references(const Json::Value &value,std::set<std::string> &result){
  if(value.isObject()){
    if(value.isMember("value")){require(value["value"].isString(),"invalid graph SSA reference");result.insert(value["value"].asString());}
    else for(const auto &key:value.getMemberNames())references(value[key],result);
  }else if(value.isArray())for(const auto &child:value)references(child,result);
}
bool is_ref(const Json::Value &arg){return arg.isObject()&&arg.isMember("value");}
const Tensor &ref(const Json::Value &arg,const Values &values){require(is_ref(arg),"expected graph tensor reference");return values.at(arg["value"].asString());}
uint64_t broadcast(uint64_t index,const Shape &out,const Shape &in){
  require(in.size()<=out.size(),"graph matrix broadcast rank mismatch");uint64_t result=0,step=1;
  for(size_t d=out.size();d-->0;){auto at=out[d]?index%out[d]:0;if(out[d])index/=out[d];
    if(d+in.size()>=out.size()){auto n=in[d+in.size()-out.size()];require(n==1||n==out[d],"graph matrix batch broadcast mismatch");if(n!=1)result+=at*step;step*=n;}}
  return result;
}
struct Task {
  const Json::Value *node=nullptr;
  std::string family;
  std::vector<std::string> inputs;
  std::vector<unsigned> consumers;
  unsigned missing=0;
  bool started=false,complete=false,view=false,window_pending=false;
  uint64_t begin=0,window_begin=0,batch=0,batches=1,m=0,n=0,k=0;
  bool linear=false;
  Shape a_batch,b_batch,batch_shape;
  Values values;
  Tensor output;
  std::vector<model_storage::Arena::Pin> pins;
  std::vector<model_io::Region> regions;
  std::unique_ptr<model_io::PhysicalMemoryPort> channel;
  std::unique_ptr<model_io::AddressSpacePort> port;
  int pipeline=-1;
  bool pipeline_producer=false;
  std::string streaming_input;
  std::unique_ptr<model_events::PairFlow> flow;
  std::unique_ptr<matrix_schedule::Simulator> matrix;
  std::unique_ptr<vector_schedule::Simulator> vector;
  std::unique_ptr<memory_model::Simulator> memory;
  std::unique_ptr<control_schedule::Simulator> control;
  void tick(){if(matrix)matrix->tick();else if(vector)vector->tick();else if(memory)memory->tick();else control->tick();}
  bool done()const{return matrix?matrix->done():vector?vector->done():memory?memory->done():control->done();}
  Json::Value result()const{return matrix?matrix->result():vector?vector->result():memory?memory->result():control->result();}
  void drop_model(){matrix.reset();vector.reset();memory.reset();control.reset();port.reset();}
};
struct Runner {
  struct Pair {unsigned producer=0,consumer=0,producer_limit=0,consumer_limit=0;model_events::Mapping mapping;std::shared_ptr<model_events::CompletionWindow> window;uint64_t begin=0;};
  const Json::Value &program;
  matrix_schedule::Options matrix_options;
  vector_schedule::Options vector_options;
  memory_model::ScheduleOptions memory_options;
  control_schedule::Options control_options;
  model_storage::Arena arena;
  PhysicalMemory memory;
  model_io::PhysicalMux mux;
  model_io::RequestTokens tokens;
  std::unique_ptr<shared_array::Resources> array;
  Values values;
  std::vector<Task> tasks;
  std::map<std::string,unsigned> uses;
  std::set<std::string> keep;
  std::set<std::pair<uint64_t,unsigned>> ready,active;
  uint64_t cycle=0,max_cycles,preloaded_assets=0,preloaded_bytes=0,completed=0,readback_requests=0;
  unsigned limit,memory_active=0,control_active=0,peak_active=0;
  bool overlap=true,progress=false;
  bool tile_pipeline=false,whole_pipeline_barrier=false;
  unsigned event_slots=32;
  uint64_t next_pipeline_epoch=1;
  std::vector<Pair> pairs;
  std::optional<unsigned> current_pair;
  Json::Value pipeline_reports{Json::arrayValue};
  Json::Value windows{Json::objectValue},events{Json::arrayValue};

  Runner(const Json::Value &p,const Json::Value &options)
      :program(p),matrix_options(matrix_schedule::Options::parse(p["matrix_schedule_options"])),
       vector_options(vector_schedule::Options::parse(p["vector_schedule_options"])),
       memory_options(memory_model::ScheduleOptions::parse(p["memory_schedule_options"])),
       control_options(control_schedule::Options::parse(p["control_schedule_options"])),
       arena(options.get("base",Json::UInt64(0x100000000ULL)).asUInt64(),options.get("bytes",Json::UInt64(16ULL<<30)).asUInt64()),
       memory(arena,MemoryOptions::parse(options.get("memory",Json::Value(Json::objectValue)))),mux(memory,65),
       max_cycles(options.get("max_cycles",Json::UInt64(1000000000000ULL)).asUInt64()),limit(options.get("max_active_nodes",32).asUInt()){
    require(options.isObject(),"ready graph options must be an object");
    for(const auto &key:options.getMemberNames())require(key=="base"||key=="bytes"||key=="memory"||key=="max_cycles"||key=="max_active_nodes"||key=="overlap"||key=="operator_progress"||key=="tile_pipeline"||key=="pipeline_whole_source_barrier","unsupported ready graph option");
    for(const char *key:{"overlap","operator_progress","tile_pipeline","pipeline_whole_source_barrier"})if(options.isMember(key))require(options[key].isBool(),"ready graph flags must be Boolean");
    overlap=options.get("overlap",true).asBool();progress=options.get("operator_progress",false).asBool();
    tile_pipeline=options.get("tile_pipeline",false).asBool();whole_pipeline_barrier=options.get("pipeline_whole_source_barrier",false).asBool();
    require(!whole_pipeline_barrier||tile_pipeline,"whole-source pipeline barrier requires tile mode");
    require(!tile_pipeline||(overlap&&limit>=2),"tile pipeline requires two active source descriptors");
    require(limit>=1&&limit<=64&&max_cycles>0&&max_cycles<UINT64_MAX-1000000,"invalid ready graph window/cycle budget");
    require(p["schema"]=="mlx_tensor_semantics_v1"&&p["timing_mode"]=="unmodeled","unsupported ready graph program");
    for(const char *kind:{"matrix","vector","memory","control"}){require(p[std::string(kind)+"_backend"]=="scheduled","ready graph requires all four scheduled routes");windows[kind]=Json::Value(Json::arrayValue);}
    shared_array::Hardware h;h.rows=matrix_options.rows;h.columns=matrix_options.columns;h.contexts=matrix_options.contexts;
    h.spm_period=matrix_options.spm_period;h.writeback_period=matrix_options.writeback_period;h.compute_ii=matrix_options.compute_ii;h.sfu_ii=vector_options.trans_ii;
    h.dma_request_period=matrix_options.dma_request_period;h.dma_response_period=matrix_options.dma_response_period;
    h.multiply_latency=matrix_options.multiply_latency;h.add_latency=matrix_options.add_latency;h.convert_latency=matrix_options.convert_latency;h.spm_latency=matrix_options.spm_latency;
    h.exp_latency=vector_options.exp_latency;h.div_latency=vector_options.div_latency;h.sqrt_latency=vector_options.sqrt_latency;
    require(h.rows==vector_options.rows&&h.columns==vector_options.columns&&h.contexts==vector_options.contexts&&h.spm_period==vector_options.spm_period&&h.writeback_period==vector_options.writeback_period&&h.compute_ii==vector_options.vector_ii&&h.multiply_latency==vector_options.multiply_latency&&h.add_latency==vector_options.add_latency&&h.convert_latency==vector_options.convert_latency&&h.spm_latency==vector_options.spm_latency&&h.dma_request_period==vector_options.dma_request_period&&h.dma_response_period==vector_options.dma_response_period,"matrix/vector hardware profiles disagree");
    array=std::make_unique<shared_array::Resources>(h);
    Assets loader;
    for(const auto &name:p["assets"].getMemberNames()){auto data=loader.load(p["assets"][name]);auto meta=arena.allocate(data.type,data.sizes,false);memory.bind(meta,data,true);values.emplace(name,std::move(meta));preloaded_bytes+=data.storage->bytes;++preloaded_assets;}
    require(p["nodes"].isArray()&&!p["nodes"].empty(),"ready graph is empty");tasks.resize(p["nodes"].size());
    std::map<std::string,unsigned> producer;std::set<uint64_t> sources;
    for(unsigned i=0;i<tasks.size();++i){auto &task=tasks[i];task.node=&p["nodes"][i];const auto &node=*task.node;auto name=node["id"].asString();
      require(!name.empty()&&!values.count(name)&&!producer.count(name)&&node["source_operator_id"].isUInt64()&&sources.insert(node["source_operator_id"].asUInt64()).second,"duplicate/invalid source or value identity");
      unsigned routes=0;for(const char *family:{"matrix","vector","memory","control"})if(node.isMember(std::string(family)+"_program")){task.family=family;++routes;}
      require(routes==1,"graph source lacks one executable backend");task.view=task.family=="memory"&&node["memory_program"]["mode"]=="view";
      std::set<std::string> deps;references(node["args"],deps);references(node["kwargs"],deps);
      for(const auto &dep:deps){require(values.count(dep)||producer.count(dep),"graph has a missing, cyclic or forward SSA reference");task.inputs.push_back(dep);++uses[dep];if(producer.count(dep)){++task.missing;tasks[producer.at(dep)].consumers.push_back(i);}}
      producer[name]=i;if(!task.missing)ready.emplace(node["source_operator_id"].asUInt64(),i);
    }
    require(p["outputs"].isArray()&&!p["outputs"].empty(),"graph has no result contract");std::set<int> forwards;
    for(const auto &out:p["outputs"]){require(out["forward_id"].isInt()&&forwards.insert(out["forward_id"].asInt()).second,"duplicate/invalid graph output forward");for(const char *role:{"logits","token"}){auto id=out[role].asString();require(values.count(id)||producer.count(id),"graph output is unbound");keep.insert(id);}}
    if(tile_pipeline)load_pipelines(p["block_pipeline_plan"]);
  }
  void load_pipelines(const Json::Value &plan){
    require(plan["profile"]=="mlx-bounded-pair-events-v1"&&plan["pairs"].isArray()&&plan["event_slots"].isUInt(),"missing or unsupported block pipeline plan");
    event_slots=plan["event_slots"].asUInt();require(event_slots&&event_slots<=32,"pipeline event capacity exceeds bounded bank");
    auto demand=[](const Task &task){const auto &p=(*task.node)[task.family+"_program"];unsigned words=task.family=="matrix"?p["prologue"].size()+p["body"].size()+p["epilogue"].size():p["rom"].size();
      require(p["rf_vectors_used"].isUInt()&&p["spm_bytes_used"].isUInt(),"invalid pipeline storage requirement");auto rf=p["rf_vectors_used"].asUInt(),bytes=p["spm_bytes_used"].asUInt();
      require(rf==(task.family=="matrix"?6u:8u)&&bytes>0&&bytes<=8192&&words>0&&words<=32,"invalid pipeline storage requirement");
      return std::array<unsigned,3>{rf,(bytes+63)/64,words};};
    for(const auto &spec:plan["pairs"]){
      require(spec["producer"].isUInt()&&spec["consumer"].isUInt(),"invalid pipeline source index");Pair pair;pair.producer=spec["producer"].asUInt();pair.consumer=spec["consumer"].asUInt();
      require(pair.producer<pair.consumer&&pair.consumer<tasks.size(),"pipeline pair is not a forward graph edge");auto &p=tasks[pair.producer],&c=tasks[pair.consumer];const auto &pn=*p.node,&cn=*c.node;
      require(p.pipeline<0&&c.pipeline<0&&p.consumers.size()==1&&p.consumers[0]==pair.consumer,"pipeline is not a disjoint closed pair");
      require((p.family=="matrix"||p.family=="vector")&&c.family=="vector"&&cn["kind"]!="mean"&&cn["kind"]!="softmax","unsupported producer/pointwise pipeline");
      require(pn["source_operator_id"].asUInt64()<cn["source_operator_id"].asUInt64()&&spec["producer_source"].asUInt64()==pn["source_operator_id"].asUInt64()&&spec["consumer_source"].asUInt64()==cn["source_operator_id"].asUInt64(),"pipeline logical source priority differs");
      auto name=pn["id"].asString();require(spec["value"]==pn["id"]&&pn["output"]["shape"]==cn["output"]["shape"],"pipeline pending input shape/value mismatch");
      bool direct=false;for(const auto &arg:cn["args"])direct|=is_ref(arg)&&arg["value"].asString()==name;require(direct,"pipeline input is not a direct tensor operand");
      model_events::Mapping expected;auto sizes=shape(pn["output"]["shape"]);expected.elements=elements(sizes);require(expected.elements>0,"empty output cannot establish a pipeline");
      if(p.family=="matrix"){require(sizes.size()>=2&&(pn["kind"]=="linear"||pn["kind"]=="matmul"),"invalid pipeline matrix mapping");expected.kind=model_events::Mapping::Kind::Matrix;expected.n=sizes.back();expected.m=pn["kind"]=="linear"?elements(Shape(sizes.begin(),sizes.end()-1)):sizes[sizes.size()-2];expected.batches=pn["kind"]=="linear"?1:elements(Shape(sizes.begin(),sizes.end()-2));}
      else if(pn["kind"]=="mean"||pn["kind"]=="softmax"){expected.kind=model_events::Mapping::Kind::Reduction;expected.row_width=pn["kind"]=="mean"?1:sizes.back();}
      pair.mapping=model_events::Mapping::parse(spec["mapping"]);const auto &actual=pair.mapping;
      require(actual.kind==expected.kind&&actual.elements==expected.elements&&actual.m==expected.m&&actual.n==expected.n&&actual.batches==expected.batches&&actual.row_width==expected.row_width,"pipeline block mapping differs from source geometry");
      require(spec["producer_blocks"].asUInt64()==actual.producer_blocks()&&spec["consumer_blocks"].asUInt64()==(actual.elements+15)/16,"pipeline block count differs");
      auto pd=demand(p),cd=demand(c);require(array->hardware().contexts>=2&&pd[0]+cd[0]<=16&&pd[1]+cd[1]<=128&&pd[2]+cd[2]<=32,"pipeline pair exceeds reserved context/RF/SPM/ROM capacity");
      for(unsigned i=0;i<3;++i){const char *key=i==0?"rf":i==1?"spm":"rom";require(spec["producer_resources"][key].asUInt()==pd[i]&&spec["consumer_resources"][key].asUInt()==cd[i],"pipeline resource descriptor differs");}
      unsigned pes=array->hardware().rows*array->hardware().columns;pair.producer_limit=pes;pair.consumer_limit=std::min(pes,(128-pd[1])/cd[1]);
      require(pair.consumer_limit&&spec["producer_context_limit"].asUInt()==pair.producer_limit&&spec["consumer_context_limit"].asUInt()==pair.consumer_limit,"pipeline producer reservation differs");
      p.pipeline=c.pipeline=int(pairs.size());p.pipeline_producer=true;c.streaming_input=name;pairs.push_back(pair);
    }
  }
  void region(Task &task,const Tensor *value,bool write=false){
    if(value){task.pins.push_back(arena.pin(*value,write));task.regions.push_back(task.pins.back().region());}
    else task.regions.push_back({0,0,false,false});
  }
  void output(Task &task){
    const auto &node=*task.node;task.output=arena.allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));auto data=Tensor::allocate(task.output.type,task.output.sizes);
    if(node.isMember("memory_program"))task.output.steps=shape(node["memory_program"]["output_layout"]["strides"]);
    memory.bind(task.output,data,false);
  }
  void open_window(Task &task){
    const auto &node=*task.node;const auto &args=node["args"];task.port=std::make_unique<model_io::AddressSpacePort>(*task.channel,tokens,task.regions,cycle);task.window_begin=cycle;
    if(task.pipeline>=0){auto &pair=pairs[unsigned(task.pipeline)];require(bool(pair.window),"pipeline group has no event bank");auto offset=task.family=="matrix"?task.batch*((task.m+1)/2)*((task.n+15)/16):0;
      task.flow=std::make_unique<model_events::PairFlow>(pair.window,pair.mapping,task.pipeline_producer,task.pipeline_producer?pair.producer_limit:pair.consumer_limit,offset,whole_pipeline_barrier);}
    if(task.family=="matrix"){
      const auto &a=ref(args[0],task.values),&b=ref(args[1],task.values);const Tensor *bias=task.linear&&args.size()>2&&!args[2].isNull()?&ref(args[2],task.values):nullptr;
      task.matrix=std::make_unique<matrix_schedule::Simulator>(node["matrix_program"],a,b,bias,task.linear,task.linear?0:broadcast(task.batch,task.batch_shape,task.a_batch),task.linear?0:broadcast(task.batch,task.batch_shape,task.b_batch),task.m,task.n,task.k,task.output,task.batch,matrix_options,task.port.get(),array.get(),node["source_operator_id"].asUInt64(),task.flow.get());
    }else if(task.family=="vector")task.vector=std::make_unique<vector_schedule::Simulator>(node,task.values,task.output,vector_options,task.port.get(),array.get(),node["source_operator_id"].asUInt64(),task.flow.get());
    else if(task.family=="control")task.control=std::make_unique<control_schedule::Simulator>(node,task.values,task.output,control_options,task.port.get());
    else{task.memory=std::make_unique<memory_model::Simulator>(node,task.values,memory_options,task.port.get(),task.view?nullptr:&task.output);if(task.view)task.output=task.memory->output();}
    task.window_pending=false;
  }
  void start(unsigned index){
    auto &task=tasks[index];const auto &node=*task.node;const auto &args=node["args"];
    for(const auto &name:task.inputs){if(name==task.streaming_input){auto &parent=tasks[pairs[unsigned(task.pipeline)].producer];require(parent.started&&!parent.complete&&parent.output.storage,"streaming parent buffer is unavailable");task.values.emplace(name,parent.output);}else task.values.emplace(name,values.at(name));}
    if(!task.view)output(task);
    if(task.family=="matrix"){
      require(node["kind"]=="linear"||node["kind"]=="matmul","matrix route has the wrong source kind");task.linear=node["kind"]=="linear";
      const auto &a=ref(args[0],task.values),&b=ref(args[1],task.values);require(a.sizes.size()>=2&&b.sizes.size()>=2,"graph matrix rank below two");
      task.m=task.linear?elements(Shape(a.sizes.begin(),a.sizes.end()-1)):a.sizes[a.sizes.size()-2];task.n=task.linear?b.sizes[0]:b.sizes.back();task.k=a.sizes.back();
      require(task.k==uint64_t(task.linear?b.sizes[1]:b.sizes[b.sizes.size()-2]),"graph matrix contraction mismatch");
      task.a_batch=Shape(a.sizes.begin(),a.sizes.end()-2);task.b_batch=Shape(b.sizes.begin(),b.sizes.end()-2);Shape expected;
      if(task.linear){require(b.sizes.size()==2,"graph linear weight rank mismatch");expected=a.sizes;expected.back()=task.n;}
      else{task.batch_shape.resize(std::max(task.a_batch.size(),task.b_batch.size()),1);for(size_t d=0;d<task.batch_shape.size();++d){auto sa=d+task.a_batch.size()>=task.batch_shape.size()?task.a_batch[d+task.a_batch.size()-task.batch_shape.size()]:1,sb=d+task.b_batch.size()>=task.batch_shape.size()?task.b_batch[d+task.b_batch.size()-task.batch_shape.size()]:1;require(sa==sb||sa==1||sb==1,"graph matrix batch mismatch");task.batch_shape[d]=sa==1?sb:sa;}task.batches=elements(task.batch_shape);expected=task.batch_shape;expected.push_back(task.m);expected.push_back(task.n);}
      require(task.batches&&expected==task.output.sizes,"graph matrix output or batch scope mismatch");
      region(task,&a);region(task,&b);region(task,task.linear&&args.size()>2&&!args[2].isNull()?&ref(args[2],task.values):nullptr);region(task,&task.output,true);
    }else if(task.family=="vector"||task.family=="control"){
      unsigned count=task.family=="vector"?node["vector_program"]["input_dtypes"].size():node["kind"]=="arange"?0:node["kind"]=="argmax"?1:2;
      require(count<=2,"graph backend input region capacity exceeded");for(unsigned i=0;i<2;++i)region(task,i<count&&is_ref(args[i])?&ref(args[i],task.values):nullptr);region(task,&task.output,true);
    }else{
      auto names=node["memory_program"]["input_layouts"].getMemberNames();std::sort(names.begin(),names.end());for(const auto &name:names)region(task,&task.values.at(name));if(!task.view)region(task,&task.output,true);
    }
    task.channel=mux.channel("source:"+std::to_string(node["source_operator_id"].asUInt64()));task.begin=cycle;task.started=true;open_window(task);
    memory_active+=task.family=="memory"&&!task.view;control_active+=task.family=="control";active.emplace(node["source_operator_id"].asUInt64(),index);
    if(progress)std::cout<<"READY_GRAPH begin source="<<node["source_operator_id"].asUInt64()<<" cycle="<<cycle<<std::endl;
  }
  void launch_ready(){
    // Binding/recycling physical storage remains a quiescent operation. This
    // gate does not stop already active frontends while they wait for data.
    if(!memory.idle()||!mux.idle())return;
    memory.collect();
    for(auto it=ready.begin();it!=ready.end()&&active.size()<(overlap?limit:1u);){
      auto index=it->second;auto &task=tasks[index];
      if((task.family=="control"&&control_active)||(task.family=="memory"&&!task.view&&memory_active)){++it;continue;}
      const bool pe=task.family=="matrix"||task.family=="vector";
      if(current_pair&&pe){++it;continue;}
      if(task.pipeline>=0){
        require(task.pipeline_producer,"pipeline consumer entered the ordinary ready queue");auto &pair=pairs[unsigned(task.pipeline)];auto &consumer=tasks[pair.consumer];bool others=true;
        for(const auto &name:consumer.inputs)if(name!=consumer.streaming_input)others&=values.count(name)!=0;
        bool pe_active=false;for(const auto &[source,at]:active){(void)source;pe_active|=tasks[at].family=="matrix"||tasks[at].family=="vector";}
        if(!others||pe_active||!array->idle()||active.size()+2>limit){++it;continue;}
        require(next_pipeline_epoch!=UINT64_MAX,"pipeline event epochs exhausted");pair.window=std::make_shared<model_events::CompletionWindow>(next_pipeline_epoch++,pair.mapping.producer_blocks(),event_slots);pair.begin=cycle;current_pair=unsigned(task.pipeline);
        it=ready.erase(it);start(index);start(pair.consumer);continue;
      }
      it=ready.erase(it);start(index);
    }
    peak_active=std::max(peak_active,unsigned(active.size()));
  }
  void finish(unsigned index){
    auto &task=tasks[index];const auto &node=*task.node;auto result=task.result();
    require(result["done"].asBool()&&result["dma_requests"]==result["dma_responses"],"graph backend has undrained requests");
    result["source_operator_id"]=node["source_operator_id"];result["forward_id"]=node["forward_id"];result["layer_idx"]=node["layer_idx"];result["batch_index"]=Json::UInt64(task.batch);
    result["shared_start_cycle"]=Json::UInt64(task.window_begin);result["shared_end_cycle"]=Json::UInt64(cycle+1);windows[task.family].append(result);task.drop_model();
    if(task.family=="matrix"&&++task.batch<task.batches){task.window_pending=true;return;}
    require(task.output.sizes==shape(node["output"]["shape"])&&dtype_name(task.output.type)==node["output"]["dtype"].asString(),"graph output type/shape changed");memory.require_initialized(task.output);
    require(!task.output.storage->data&&!task.output.storage->writable,"graph backend gained local numerical storage");
    Json::Value event;event["source_operator_id"]=node["source_operator_id"];event["kind"]=node["kind"];event["family"]=task.family;event["forward_id"]=node["forward_id"];event["layer_idx"]=node["layer_idx"];
    event["start_cycle"]=Json::UInt64(task.begin);event["publish_cycle"]=Json::UInt64(cycle+1);event["batches"]=Json::UInt64(task.batches);auto allocation=arena.allocation(task.output);event["allocation_id"]=Json::UInt64(allocation.id);event["physical_base"]=Json::UInt64(allocation.base);events.append(event);
    auto name=node["id"].asString();require(values.emplace(name,task.output).second,"graph redefined a published value");
    for(const auto &dep:task.inputs){require(uses.at(dep)>0,"graph use count underflow");if(!--uses[dep]&&!keep.count(dep))require(values.erase(dep)==1,"graph prematurely released an input");}
    if(!uses[name]&&!keep.count(name))values.erase(name);
    task.values.clear();task.output=Tensor{};task.pins.clear();task.regions.clear();task.channel.reset();task.flow.reset();task.complete=true;++completed;
    memory_active-=task.family=="memory"&&!task.view;control_active-=task.family=="control";active.erase({node["source_operator_id"].asUInt64(),index});
    for(auto child:task.consumers){require(tasks[child].missing>0,"graph dependency count underflow");if(!--tasks[child].missing&&!tasks[child].started)ready.emplace((*tasks[child].node)["source_operator_id"].asUInt64(),child);}
    if(progress)std::cout<<"READY_GRAPH complete source="<<node["source_operator_id"].asUInt64()<<" cycle="<<cycle+1<<std::endl;
  }
  Tensor readback(const Tensor &value){
    auto output=Tensor::allocate(value.type,value.sizes);auto pin=arena.pin(value);auto channel=mux.channel("result-readback");model_io::AddressSpacePort port(*channel,tokens,{pin.region()},cycle);uint64_t local=0;
    for(uint64_t index=0;index<value.numel();++index){bool sent=false,received=false;model_io::Request request;request.id=index;request.bytes=element_bytes(value.type);request.offset=value.position(index)*request.bytes;
      while(!received){require(cycle<max_cycles,"graph result readback exceeded cycle budget");port.advance(local);
        if(sent){if(auto response=port.response()){require(response->id==request.id&&!response->error,"graph result readback owner/error mismatch");std::memcpy(output.storage->writable+index*request.bytes,&response->data,request.bytes);port.consume_response();received=true;}}
        else if(port.request_ready()){port.submit(request);sent=true;}
        ++cycle;++local;
      }
      ++readback_requests;
    }
    require(memory.idle()&&mux.idle(),"graph readback did not drain");return output;
  }
  Json::Value run(const std::filesystem::path &directory){
    while(completed<tasks.size()){
      require(cycle<max_cycles,"ready graph exceeded global cycle budget");
      for(const auto &[source,index]:active){(void)source;if(tasks[index].window_pending)open_window(tasks[index]);}
      launch_ready();require(!active.empty(),"ready graph cannot make progress");if(current_pair)pairs[*current_pair].window->advance(cycle);mux.advance(cycle);array->begin_cycle(cycle);std::vector<unsigned> finished;
      for(const auto &[source,index]:active){(void)source;auto &task=tasks[index];if(!task.done())task.tick();if(task.done())finished.push_back(index);}
      array->end_cycle();for(auto index:finished)finish(index);
      if(current_pair){auto &pair=pairs[*current_pair];if(tasks[pair.producer].complete&&tasks[pair.consumer].complete){require(pair.window->finished()&&array->idle(),"closed pipeline did not drain its events/resources");auto report=pair.window->snapshot();report["producer_source"]=(*tasks[pair.producer].node)["source_operator_id"];report["consumer_source"]=(*tasks[pair.consumer].node)["source_operator_id"];report["begin_cycle"]=Json::UInt64(pair.begin);report["end_cycle"]=Json::UInt64(cycle+1);pipeline_reports.append(report);pair.window.reset();current_pair.reset();}}
      ++cycle;
    }
    require(active.empty()&&ready.empty()&&array->idle()&&memory.idle()&&mux.idle(),"ready graph did not drain");uint64_t graph_cycles=cycle;Json::Value outputs(Json::arrayValue);
    for(const auto &spec:program["outputs"]){auto logits=readback(values.at(spec["logits"].asString())),token=readback(values.at(spec["token"].asString()));
      for(uint64_t i=0;i<logits.numel();++i)require(std::isfinite(logits.number(i)),"nonfinite ready graph result");
      auto file=directory/("logits-"+std::to_string(spec["forward_id"].asInt())+"."+dtype_name(logits.type)+".bin");std::ofstream bytes(file,std::ios::binary);if(logits.storage->bytes)bytes.write(reinterpret_cast<const char*>(logits.storage->data),logits.storage->bytes);require(bool(bytes),"cannot write ready graph logits");
      Json::Value row;row["forward_id"]=spec["forward_id"];row["logits_file"]=file.string();row["dtype"]=dtype_name(logits.type);row["shape"]=Json::Value(Json::arrayValue);for(auto n:logits.sizes)row["shape"].append(Json::Int64(n));row["tokens"]=Json::Value(Json::arrayValue);for(uint64_t i=0;i<token.numel();++i)row["tokens"].append(Json::Int64(token.integer(i)));outputs.append(row);
    }
    values.clear();memory.collect();auto drained=arena.snapshot();require(drained["reserved_bytes"].asUInt64()==0&&memory.idle()&&mux.idle(),"ready graph retains output/input storage");
    Json::Value r;r["classification"]="ready_graph_shared_array_execution_not_cdc_cpu_or_system_acceptance";r["executed_source_calls"]=Json::UInt64(completed);r["events"]=events;r["windows"]=windows;r["outputs"]=outputs;
    r["array"]=array->snapshot();r["memory"]=memory.snapshot();r["physical_mux"]=mux.snapshot();r["arena_drained"]=drained;r["peak_active_nodes"]=peak_active;r["max_active_nodes"]=limit;r["overlap"]=overlap;
    r["graph_cycles"]=Json::UInt64(graph_cycles);r["host_readback_cycles"]=Json::UInt64(cycle-graph_cycles);r["host_readback_requests"]=Json::UInt64(readback_requests);r["shared_elapsed_cycles"]=Json::UInt64(cycle);
    r["preloaded_assets"]=Json::UInt64(preloaded_assets);r["preloaded_bytes"]=Json::UInt64(preloaded_bytes);r["virtual_tensor_backing_used"]=true;r["functional_entry_calls"]=0;r["blas_calls"]=0;r["python_or_gpu_execution_fallbacks"]=0;
    r["dependency_visibility"]="whole_source_completion_next_edge_not_partial_tile_cdc";r["control_execution"]="scheduled_rv64_leaf_not_actual_cpu";r["weight_loading"]="preloaded_not_cpu_or_dma_loader";
    if(tile_pipeline){require(!current_pair&&pipeline_reports.size()==pairs.size(),"not all compiled pipeline pairs executed");r["classification"]="ready_graph_bounded_pair_events_not_general_cdc_or_system_acceptance";r["dependency_visibility"]="compiled_closed_pairs_use_next_edge_block_events_other_edges_whole_source";r["pipeline_groups"]=pipeline_reports;r["pipeline_whole_source_barrier"]=whole_pipeline_barrier;}
    r["full_model_execution_verified"]=false;r["complete_cdc_verified"]=false;r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;return r;
  }
};
}
Json::Value execute_ready_graph(const Json::Value &program,const Json::Value &options,const std::filesystem::path &directory){Runner runner(program,options);return runner.run(directory);}
}
