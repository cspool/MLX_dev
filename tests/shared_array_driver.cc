#include "physical_memory.h"
#include "matrix_schedule.h"
#include "vector_schedule.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

using namespace mlx;
using namespace mlx::tensor_model;
namespace {
struct Buffer {Tensor metadata,backing;model_storage::Arena::Pin pin;};
struct Kernel {
  std::unique_ptr<model_io::AddressSpacePort> port;
  std::unique_ptr<matrix_schedule::Simulator> matrix;
  std::unique_ptr<vector_schedule::Simulator> vector;
  bool started=false;
  bool done()const{return started&&(matrix?matrix->done():vector->done());}
  void tick(){if(matrix)matrix->tick();else vector->tick();}
  Json::Value report()const{return matrix?matrix->result():vector->result();}
};
}
int main(int argc,char **argv){
  try{
    require(argc==3,"usage: shared-array-driver job.json output-directory");Json::Value job;std::ifstream input(argv[1]);input>>job;
    const bool overlap=job.get("overlap",true).asBool(),vector_first=job.get("vector_first",false).asBool();
    const auto &matrix_job=job["matrix"],&vector_job=job["vector"];
    auto mo=matrix_schedule::Options::parse(matrix_job["options"]);
    auto vo=vector_schedule::Options::parse(vector_job["options"]);
    shared_array::Hardware h;h.rows=mo.rows;h.columns=mo.columns;h.contexts=mo.contexts;
    h.spm_period=mo.spm_period;h.writeback_period=mo.writeback_period;h.compute_ii=mo.compute_ii;h.sfu_ii=vo.trans_ii;
    h.dma_request_period=mo.dma_request_period;h.dma_response_period=mo.dma_response_period;
    h.multiply_latency=mo.multiply_latency;h.add_latency=mo.add_latency;h.convert_latency=mo.convert_latency;h.spm_latency=mo.spm_latency;
    h.exp_latency=vo.exp_latency;h.div_latency=vo.div_latency;h.sqrt_latency=vo.sqrt_latency;
    shared_array::Resources array(h);model_storage::Arena arena(0x100000000ULL,4*1024*1024);
    model_system::PhysicalMemory memory(arena,model_system::MemoryOptions::parse(job["memory"]));model_io::RequestTokens tokens;Assets loader;
    auto bind=[&](const Json::Value &spec,bool output){
      Buffer b;b.backing=output?Tensor::allocate(dtype(spec["dtype"].asString()),shape(spec["shape"])):loader.load(spec);
      require(b.backing.contiguous()&&b.backing.offset==0,"shared driver requires contiguous fixture inputs");
      b.metadata=arena.allocate(b.backing.type,b.backing.sizes,output);memory.bind(b.metadata,b.backing,!output);b.pin=arena.pin(b.metadata,output);
      require(!b.metadata.storage->data&&!b.metadata.storage->writable,"shared driver gave a backend local numerical data");return b;
    };
    auto a=bind(matrix_job["a"],false),b=bind(matrix_job["b"],false);Buffer bias;bool with_bias=matrix_job.isMember("bias");if(with_bias)bias=bind(matrix_job["bias"],false);
    auto m=matrix_job["m"].asUInt64(),n=matrix_job["n"].asUInt64(),k=matrix_job["k"].asUInt64();
    Json::Value matrix_output;matrix_output["dtype"]=matrix_job["program"]["output_dtype"];matrix_output["shape"].append(Json::UInt64(m));matrix_output["shape"].append(Json::UInt64(n));
    auto output_m=bind(matrix_output,true),output_v=bind(vector_job["node"]["output"],true);
    Values values;std::map<std::string,Buffer> vector_inputs;
    for(const auto &name:vector_job["assets"].getMemberNames()){auto value=bind(vector_job["assets"][name],false);values.emplace(name,value.metadata);vector_inputs.emplace(name,std::move(value));}
    std::vector<model_io::Region> mr{a.pin.region(),b.pin.region(),with_bias?bias.pin.region():model_io::Region{},output_m.pin.region()};
    std::vector<model_io::Region> vr(3);vr[2]=output_v.pin.region();
    const auto &node=vector_job["node"];
    for(Json::ArrayIndex i=0;i<node["vector_program"]["input_dtypes"].size();++i)if(node["args"][i].isObject())vr[i]=vector_inputs.at(node["args"][i]["value"].asString()).pin.region();
    Kernel mk,vk;uint64_t cycle=0;
    auto start_matrix=[&]{mk.port=std::make_unique<model_io::AddressSpacePort>(memory,tokens,mr,cycle);mk.matrix=std::make_unique<matrix_schedule::Simulator>(matrix_job["program"],a.metadata,b.metadata,with_bias?&bias.metadata:nullptr,true,0,0,m,n,k,output_m.metadata,0,mo,mk.port.get(),&array,vector_first?2:1);mk.started=true;};
    auto start_vector=[&]{vk.port=std::make_unique<model_io::AddressSpacePort>(memory,tokens,vr,cycle);vk.vector=std::make_unique<vector_schedule::Simulator>(node,values,output_v.metadata,vo,vk.port.get(),&array,vector_first?1:2);vk.started=true;};
    Kernel *first=vector_first?&vk:&mk,*second=vector_first?&mk:&vk;
    std::function<void()> start_first=vector_first?std::function<void()>(start_vector):std::function<void()>(start_matrix);
    std::function<void()> start_second=vector_first?std::function<void()>(start_matrix):std::function<void()>(start_vector);
    start_first();if(overlap)start_second();
    while(!first->done()||!second->done()){
      require(cycle<job.get("max_cycles",Json::UInt64(2000000)).asUInt64(),"shared operator group exceeded cycle limit");
      if(!second->started&&first->done())start_second();
      array.begin_cycle(cycle);
      if(!first->done())first->tick();
      if(second->started&&!second->done())second->tick();
      array.end_cycle();++cycle;
    }
    require(array.idle()&&memory.idle(),"shared operators did not drain array and physical memory");
    memory.require_initialized(output_m.metadata);memory.require_initialized(output_v.metadata);
    Json::Value report;report["classification"]="shared_array_two_operator_execution_not_complete_graph_or_system_validation";
    report["cycles"]=Json::UInt64(cycle);report["overlap"]=overlap;report["vector_first"]=vector_first;report["array"]=array.snapshot();report["memory"]=memory.snapshot();
    report["matrix"]=mk.report();report["vector"]=vk.report();report["virtual_tensor_backing_used"]=true;report["executed_source_calls"]=2;
    report["full_model_execution_verified"]=false;report["mlx_system_verified"]=false;report["inference_performance_eligible"]=false;
    auto directory=std::filesystem::path(argv[2]);std::filesystem::create_directories(directory);
    for(auto pair:{std::make_pair("matrix.bin",&output_m),std::make_pair("vector.bin",&output_v)}){
      std::ofstream output(directory/pair.first,std::ios::binary);auto &tensor=pair.second->backing;
      if(tensor.storage->bytes)output.write(reinterpret_cast<const char*>(tensor.storage->data),tensor.storage->bytes);
      require(bool(output),"cannot write shared operator output");
    }
    std::ofstream json(directory/"result.json");json<<report<<'\n';require(bool(json),"cannot write shared operator report");
    std::cout<<"SHARED_ARRAY_OPERATOR_GROUP_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"SHARED_ARRAY_OPERATOR_GROUP_FAIL: "<<error.what()<<std::endl;return 1;}
}
