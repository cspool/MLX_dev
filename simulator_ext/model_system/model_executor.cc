#include "model_executor.h"
#include "matrix_schedule.h"
#include "vector_schedule.h"
#include "memory_schedule.h"
#include "control_schedule.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <openssl/evp.h>
#include <set>
#include <sstream>

namespace mlx::model_system {
using namespace tensor_model;
namespace {
const Tensor &ref(const Json::Value &arg,const Values &values){
  require(arg.isObject()&&arg["value"].isString(),"physical model expected SSA reference");auto it=values.find(arg["value"].asString());require(it!=values.end(),"physical model references an unbound/released SSA value");return it->second;
}
bool is_ref(const Json::Value &arg){return arg.isObject()&&arg.isMember("value");}
uint64_t broadcast(uint64_t flat,const Shape &out,const Shape &in){
  require(in.size()<=out.size(),"physical matrix batch rank mismatch");uint64_t index=0,step=1;
  for(size_t d=out.size();d-->0;){auto at=out[d]?flat%out[d]:0;if(out[d])flat/=out[d];
    if(d+in.size()>=out.size()){auto n=in[d+in.size()-out.size()];require(n==1||n==out[d],"physical matrix batch broadcast mismatch");if(n!=1)index+=at*step;step*=n;}}
  return index;
}
// Preserve all per-window numeric accounting without merging resource
// capacities as if they were work. Resource bounds remain in window reports.
void accumulate(Json::Value &total,const Json::Value &source,const std::vector<std::string> &counts){
  for(const auto &name:counts)total[name]=Json::UInt64(total.get(name,Json::UInt64(0)).asUInt64()+source[name].asUInt64());
  for(const auto &op:source["opcode_counts"].getMemberNames())total["opcode_counts"][op]=Json::UInt64(total["opcode_counts"].get(op,Json::UInt64(0)).asUInt64()+source["opcode_counts"][op].asUInt64());
  for(const auto &name:source.getMemberNames())if(name.rfind("max_",0)==0)total[name]=Json::UInt64(std::max(total.get(name,Json::UInt64(0)).asUInt64(),source[name].asUInt64()));
}
struct Runner {
  const Json::Value &program;
  model_storage::Arena arena;
  PhysicalMemory memory;
  model_io::RequestTokens tokens;
  Values values;
  uint64_t cycle=0,max_cycles,preloaded_bytes=0,preloaded_assets=0,readback_cycles=0,readback_requests=0;
  uint64_t diagnostic_cycles=0,diagnostic_requests=0,diagnostic_bytes=0;
  bool operator_progress=false;
  std::set<unsigned> observe_ids;
  Json::Value observations{Json::arrayValue};
  std::chrono::steady_clock::time_point host_start;
  Json::Value reports{Json::objectValue},events{Json::arrayValue},matrix_stats,vector_stats,memory_stats,control_stats;

  Runner(const Json::Value &p,const Json::Value &options)
      :program(p),arena(options.get("base",Json::UInt64(uint64_t(1)<<32)).asUInt64(),options.get("bytes",Json::UInt64(uint64_t(16)<<30)).asUInt64()),
       memory(arena,MemoryOptions::parse(options.get("memory",Json::Value(Json::objectValue)))),max_cycles(options.get("max_cycles",Json::UInt64(1000000000000ULL)).asUInt64()){
    require(options.isObject(),"physical system options must be an object");
    for(const auto &name:options.getMemberNames())require(name=="base"||name=="bytes"||name=="max_cycles"||name=="memory"||name=="operator_progress"||name=="observe_operators","unknown physical system option");
    if(options.isMember("operator_progress")){require(options["operator_progress"].isBool(),"operator progress flag must be Boolean");operator_progress=options["operator_progress"].asBool();}
    if(options.isMember("observe_operators")){
      require(options["observe_operators"].isArray()&&options["observe_operators"].size()<=48,"invalid physical observation ID list");
      for(const auto &id:options["observe_operators"]){require(id.isUInt(),"invalid physical observation ID");require(observe_ids.insert(id.asUInt()).second,"duplicate physical observation ID");}
      auto missing=observe_ids;uint64_t bytes=0;
      for(const auto &node:program["nodes"])if(missing.erase(node["source_operator_id"].asUInt())){
        auto count=elements(shape(node["output"]["shape"]));auto width=element_bytes(dtype(node["output"]["dtype"].asString()));
        require(count<=1048576/width,"physical observation exceeds per-output byte limit");auto size=count*width;
        require(size<=33554432-bytes,"physical observations exceed total byte limit");bytes+=size;
      }
      require(missing.empty(),"physical observation ID is not in the program");
    }
    require(max_cycles>0&&max_cycles<UINT64_MAX-1000000,"invalid shared cycle limit");
    require(program["schema"]=="mlx_tensor_semantics_v1"&&program["timing_mode"]=="unmodeled","invalid physical model program");
    for(const char *name:{"matrix","vector","memory","control"})require(program[std::string(name)+"_backend"]=="scheduled","physical model requires all four scheduled backends");
    matrix_schedule::Options::parse(program["matrix_schedule_options"]);vector_schedule::Options::parse(program["vector_schedule_options"]);
    memory_model::ScheduleOptions::parse(program["memory_schedule_options"]);control_schedule::Options::parse(program["control_schedule_options"]);
    Assets loader;
    for(const auto &name:program["assets"].getMemberNames()){
      const auto &spec=program["assets"][name];auto data=loader.load(spec);auto metadata=arena.allocate(data.type,data.sizes,false);
      memory.bind(metadata,data,true);values.emplace(name,std::move(metadata));preloaded_bytes+=data.storage->bytes;++preloaded_assets;
    }
    for(const char *name:{"matrix","vector","memory","control"})reports[name]=Json::Value(Json::arrayValue);
    host_start=std::chrono::steady_clock::now();
  }
  struct Binding {
    std::vector<model_storage::Arena::Pin> pins;
    std::vector<model_io::Region> regions;
    Binding(Runner &r,const std::vector<const Tensor*> &inputs,const Tensor *output){
      for(const auto *input:inputs){
        if(input){pins.push_back(r.arena.pin(*input));regions.push_back(pins.back().region());}
        else regions.push_back({0,0,false,false});
      }
      if(output){pins.push_back(r.arena.pin(*output,true));regions.push_back(pins.back().region());}
    }
  };
  template<class Simulator> Json::Value run_window(Simulator &model,const char *kind,const Json::Value &node,uint64_t batch=0){
    uint64_t start=cycle;
    while(!model.done()){require(cycle<max_cycles,"physical model exceeded shared max_cycles");model.tick();++cycle;}
    auto report=model.result();require(report["done"].asBool()&&report["cycles"].asUInt64()==cycle-start&&report["dma_requests"]==report["dma_responses"]&&memory.idle(),"physical model window did not drain on shared clock");
    report["source_operator_id"]=node["source_operator_id"];report["forward_id"]=node["forward_id"];report["layer_idx"]=node["layer_idx"];
    report["shared_start_cycle"]=Json::UInt64(start);report["shared_end_cycle"]=Json::UInt64(cycle);report["batch_index"]=Json::UInt64(batch);reports[kind].append(report);
    return report;
  }
  Tensor allocate_output(const Json::Value &node){
    auto metadata=arena.allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));
    auto backing=Tensor::allocate(metadata.type,metadata.sizes);
    if(node.isMember("memory_program"))metadata.steps=shape(node["memory_program"]["output_layout"]["strides"]);
    memory.bind(metadata,backing,false);return metadata;
  }
  Tensor execute_node(const Json::Value &node){
    unsigned paths=0;for(const char *field:{"matrix_program","vector_program","memory_program","control_program"})paths+=node.isMember(field);
    require(paths==1,"physical model operator lacks exactly one compiled route");
    const auto &args=node["args"];bool view=node["memory_program"]["mode"]=="view";Tensor output;
    if(!view)output=allocate_output(node);
    if(node.isMember("memory_program")){
      auto names=node["memory_program"]["input_layouts"].getMemberNames();std::sort(names.begin(),names.end());std::vector<const Tensor*> inputs;
      for(const auto &name:names){auto it=values.find(name);require(it!=values.end(),"missing physical memory operand");inputs.push_back(&it->second);}
      Binding binding(*this,inputs,view?nullptr:&output);model_io::AddressSpacePort port(memory,tokens,binding.regions,cycle);
      memory_model::Simulator model(node,values,memory_model::ScheduleOptions::parse(program["memory_schedule_options"]),&port,view?nullptr:&output);
      auto report=run_window(model,"memory",node);output=model.output();
      accumulate(memory_stats,report["numeric_instructions"],{"calls","view_elisions","allocations","instructions","read_bytes","write_bytes","index_reads","predicate_reads"});
    }else if(node.isMember("control_program")){
      std::vector<const Tensor*> inputs(2,nullptr);unsigned count=node["kind"]=="arange"?0:node["kind"]=="argmax"?1:2;
      for(unsigned i=0;i<count;++i)if(is_ref(args[i]))inputs[i]=&ref(args[i],values);
      Binding binding(*this,inputs,&output);model_io::AddressSpacePort port(memory,tokens,binding.regions,cycle);
      control_schedule::Simulator model(node,values,output,control_schedule::Options::parse(program["control_schedule_options"]),&port);
      auto report=run_window(model,"control",node);report["calls"]=1;accumulate(control_stats,report,{"calls","instructions","branches_taken","read_bytes","write_bytes"});
      control_stats["fflags_observed"]=control_stats.get("fflags_observed",0).asUInt()|report["fflags_observed"].asUInt();
    }else if(node.isMember("vector_program")){
      std::vector<const Tensor*> inputs(2,nullptr);for(unsigned i=0;i<node["vector_program"]["input_dtypes"].size();++i)if(is_ref(args[i])){require(i<2,"physical vector operand count exceeds port contract");inputs[i]=&ref(args[i],values);}
      Binding binding(*this,inputs,&output);model_io::AddressSpacePort port(memory,tokens,binding.regions,cycle);
      vector_schedule::Simulator model(node,values,output,vector_schedule::Options::parse(program["vector_schedule_options"]),&port);
      auto report=run_window(model,"vector",node);accumulate(vector_stats,report["numeric_instructions"],{"calls","instructions","transcendental_lanes","arithmetic_lanes","global_read_bytes","global_write_bytes"});
    }else{
      require(node["kind"]=="linear"||node["kind"]=="matmul","physical matrix code attached to wrong kind");
      bool linear=node["kind"]=="linear";const auto &a=ref(args[0],values),&b=ref(args[1],values);
      require(a.sizes.size()>=2&&b.sizes.size()>=2,"physical matrix rank below two");
      uint64_t m=linear?elements(Shape(a.sizes.begin(),a.sizes.end()-1)):a.sizes[a.sizes.size()-2],k=a.sizes.back(),n=linear?b.sizes[0]:b.sizes.back();
      require(k==uint64_t(linear?b.sizes[1]:b.sizes[b.sizes.size()-2]),"physical matrix contracted dimension mismatch");
      Shape ba(a.sizes.begin(),a.sizes.end()-2),bb(b.sizes.begin(),b.sizes.end()-2),batch,expected;
      if(linear){require(b.sizes.size()==2,"physical linear weight is not rank two");expected=a.sizes;expected.back()=n;}
      else{
        batch.resize(std::max(ba.size(),bb.size()),1);
        for(size_t d=0;d<batch.size();++d){auto sa=d+ba.size()>=batch.size()?ba[d+ba.size()-batch.size()]:1,sb=d+bb.size()>=batch.size()?bb[d+bb.size()-batch.size()]:1;require(sa==sb||sa==1||sb==1,"physical matrix broadcast mismatch");batch[d]=sa==1?sb:sa;}
        expected=batch;expected.push_back(m);expected.push_back(n);
      }
      require(expected==output.sizes,"physical matrix output shape mismatch");const Tensor *bias=linear&&args.size()>2&&!args[2].isNull()?&ref(args[2],values):nullptr;
      uint64_t batches=linear?1:elements(batch);
      require(batches>0,"empty matrix batch is not registered by the physical executor");
      for(uint64_t index=0;index<batches;++index){
        Binding binding(*this,{&a,&b,bias},&output);model_io::AddressSpacePort port(memory,tokens,binding.regions,cycle);
        matrix_schedule::Simulator model(node["matrix_program"],a,b,bias,linear,linear?0:broadcast(index,batch,ba),linear?0:broadcast(index,batch,bb),m,n,k,output,index,matrix_schedule::Options::parse(program["matrix_schedule_options"]),&port);
        auto report=run_window(model,"matrix",node,index);accumulate(matrix_stats,report["numeric_instructions"],{"calls","output_tiles","instructions","inactive_row_instructions","mul_active_lanes","add_active_lanes","global_read_bytes","global_write_bytes"});
      }
    }
    require(output.sizes==shape(node["output"]["shape"])&&dtype_name(output.type)==node["output"]["dtype"].asString(),"physical model output contract mismatch");
    require(!output.storage->data&&!output.storage->writable,"physical backend escaped into host tensor data");memory.require_initialized(output);return output;
  }
  template<class Consumer> void read_values(const Tensor &value,Consumer consume){
    auto pin=arena.pin(value);model_io::AddressSpacePort port(memory,tokens,{pin.region()},cycle);
    uint64_t local_cycle=0;
    for(uint64_t index=0;index<value.numel();++index){
      bool submitted=false,received=false;model_io::Request request;request.id=index;request.bytes=element_bytes(value.type);request.offset=value.position(index)*request.bytes;
      while(!received){
        require(cycle<max_cycles,"host readback exceeded shared max_cycles");port.advance(local_cycle);
        if(submitted){
          if(auto response=port.response()){
            require(response->id==request.id&&!response->error,"physical host readback response mismatch");consume(index,response->data,request.bytes);port.consume_response();received=true;
          }
        }else if(port.request_ready()){port.submit(request);submitted=true;}
        ++cycle;++local_cycle;
      }
    }
    require(memory.idle(),"host readback did not drain");
  }
  Tensor readback(const Tensor &value){
    auto result=Tensor::allocate(value.type,value.sizes);auto start=cycle;
    read_values(value,[&](uint64_t index,uint64_t data,unsigned bytes){std::memcpy(result.storage->writable+index*bytes,&data,bytes);});
    readback_cycles+=cycle-start;readback_requests+=value.numel();return result;
  }
  void observe(const Json::Value &node,const Tensor &value,const std::filesystem::path &directory){
    auto start=cycle;std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> hash(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    require(hash&&EVP_DigestInit_ex(hash.get(),EVP_sha256(),nullptr)==1,"cannot initialize physical output digest");
    read_values(value,[&](uint64_t,uint64_t data,unsigned bytes){require(EVP_DigestUpdate(hash.get(),&data,bytes)==1,"cannot update physical output digest");});
    unsigned char digest[32];unsigned bytes=0;require(EVP_DigestFinal_ex(hash.get(),digest,&bytes)==1&&bytes==32,"cannot finalize physical output digest");
    std::ostringstream hex;hex<<std::hex<<std::setfill('0');for(auto byte:digest)hex<<std::setw(2)<<unsigned(byte);
    Json::Value row;for(const char *field:{"source_operator_id","forward_id","layer_idx","phase"})row[field]=node[field];
    row["dtype"]=dtype_name(value.type);row["shape"]=node["output"]["shape"];row["bytes"]=Json::UInt64(value.numel()*element_bytes(value.type));row["sha256"]=hex.str();
    row["shared_start_cycle"]=Json::UInt64(start);row["shared_end_cycle"]=Json::UInt64(cycle);row["producer_completed"]=true;observations.append(row);
    diagnostic_cycles+=cycle-start;diagnostic_requests+=value.numel();diagnostic_bytes+=value.numel()*element_bytes(value.type);
    Json::Value report;report["classification"]="physical_output_digests_not_model_completion";report["observations"]=observations;report["certifies_full_model"]=false;
    auto temporary=directory/"observations.json.tmp";{std::ofstream file(temporary);file<<report<<'\n';require(bool(file),"cannot save physical observations");}
    std::filesystem::rename(temporary,directory/"observations.json");
  }
  void progress(const char *state,const Json::Value &node){
    if(!operator_progress)return;
    auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-host_start).count();
    std::cout<<"PHYSICAL_OPERATOR "<<state<<" id="<<node["source_operator_id"].asUInt()<<" forward="<<node["forward_id"].asInt()<<" layer="<<(node["layer_idx"].isNull()?-1:node["layer_idx"].asInt())<<" phase="<<node["phase"].asString()<<" cycle="<<cycle<<" host_seconds="<<std::setprecision(9)<<seconds<<std::endl;
  }
  Json::Value run(const std::filesystem::path &directory){
    std::filesystem::create_directories(directory);std::string boundary;
    for(const auto &node:program["nodes"]){
      auto position=std::to_string(node["forward_id"].asInt())+":"+(node["layer_idx"].isNull()?"outside":std::to_string(node["layer_idx"].asInt()));
      if(position!=boundary){std::cout<<"PHYSICAL_MODEL_BOUNDARY "<<position<<" operator="<<node["source_operator_id"].asUInt()<<std::endl;boundary=position;}
      auto name=node["id"].asString();require(!values.count(name),"physical model redefines live SSA value");auto start=cycle;
      progress("begin",node);
      try{values.emplace(name,execute_node(node));}catch(const std::exception &error){progress("failed",node);throw std::runtime_error("physical operator "+std::to_string(node["source_operator_id"].asUInt())+": "+error.what());}
      progress("complete",node);
      Json::Value event;event["source_operator_id"]=node["source_operator_id"];event["kind"]=node["kind"];event["entry"]="tensor_model::"+node["kind"].asString();event["forward_id"]=node["forward_id"];event["layer_idx"]=node["layer_idx"];
      event["shared_start_cycle"]=Json::UInt64(start);event["shared_end_cycle"]=Json::UInt64(cycle);auto allocation=arena.allocation(values.at(name));event["allocation_id"]=Json::UInt64(allocation.id);event["physical_base"]=Json::UInt64(allocation.base);events.append(event);
      if(observe_ids.erase(node["source_operator_id"].asUInt()))observe(node,values.at(name),directory);
      for(const auto &released:node["release"])require(values.erase(released.asString())==1,"physical model released missing SSA value");
      memory.collect();
    }
    require(observe_ids.empty(),"requested physical observation did not execute");auto device_end=cycle-diagnostic_cycles;Json::Value outputs(Json::arrayValue);
    for(const auto &spec:program["outputs"]){
      auto logits=readback(values.at(spec["logits"].asString())),token=readback(values.at(spec["token"].asString()));
      for(uint64_t i=0;i<logits.numel();++i)require(std::isfinite(logits.number(i)),"nonfinite physical model logits");
      auto path=directory/("logits-"+std::to_string(spec["forward_id"].asInt())+"."+dtype_name(logits.type)+".bin");std::ofstream file(path,std::ios::binary);
      if(logits.storage->bytes)file.write(reinterpret_cast<const char*>(logits.storage->data),logits.storage->bytes);
      require(bool(file),"cannot write physical model logits");Json::Value result;result["forward_id"]=spec["forward_id"];result["logits_file"]=path.string();result["dtype"]=dtype_name(logits.type);result["shape"]=Json::Value(Json::arrayValue);for(auto n:logits.sizes)result["shape"].append(Json::Int64(n));
      result["tokens"]=Json::Value(Json::arrayValue);for(uint64_t i=0;i<token.numel();++i)result["tokens"].append(Json::Int64(token.integer(i)));outputs.append(result);
    }
    auto before_release=arena.snapshot();values.clear();memory.collect();auto drained=arena.snapshot();
    require(drained["reserved_bytes"].asUInt64()==0&&memory.idle(),"physical model ended with owned buffers or outstanding traffic");
    Json::Value r;r["classification"]="shared_native_physical_model_execution_not_chipyard_system_validation";
    r["executed_source_calls"]=program["nodes"].size();r["events"]=events;r["outputs"]=outputs;r["windows"]=reports;
    r["matrix_microcode"]=matrix_stats;r["vector_microcode"]=vector_stats;r["memory_programs"]=memory_stats;r["control_programs"]=control_stats;
    r["memory"]=memory.snapshot();r["arena_before_result_release"]=before_release;r["arena_drained"]=drained;
    r["preloaded_assets"]=Json::UInt64(preloaded_assets);r["preloaded_bytes"]=Json::UInt64(preloaded_bytes);r["device_component_cycles"]=Json::UInt64(device_end);r["host_readback_cycles"]=Json::UInt64(readback_cycles);r["host_readback_requests"]=Json::UInt64(readback_requests);r["shared_elapsed_cycles"]=Json::UInt64(cycle);
    r["observations"]=observations;r["diagnostic_readback_cycles"]=Json::UInt64(diagnostic_cycles);r["diagnostic_readback_requests"]=Json::UInt64(diagnostic_requests);r["diagnostic_readback_bytes"]=Json::UInt64(diagnostic_bytes);
    r["virtual_tensor_backing_used"]=true;r["functional_entry_calls"]=0;r["blas_calls"]=0;r["python_or_gpu_execution_fallbacks"]=0;
    r["timing_mode"]="unmodeled_host_and_system";r["cross_operator_execution"]="serial_with_shared_address_space";r["weight_loading"]="preloaded_not_cpu_or_dma_loader";r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;return r;
  }
};
} // namespace
Json::Value execute(const Json::Value &program,const Json::Value &options,const std::filesystem::path &directory){Runner runner(program,options);return runner.run(directory);}
} // namespace mlx::model_system
