#include "vector_schedule.h"
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace mlx::tensor_model;
using namespace mlx::vector_schedule;
namespace {
Tensor virtual_tensor(const Json::Value &spec){
  Tensor t;t.type=dtype(spec["dtype"].asString());t.sizes=shape(spec["shape"]);t.steps=strides(t.sizes);t.storage=std::make_shared<Storage>();t.storage->bytes=t.numel()*element_bytes(t.type);return t;
}
class ForeignMemory final:public MemoryPort{
public:
  std::array<Tensor,3> regions;
  std::optional<Request> pending;
  std::optional<Response> answer;
  uint64_t cycle=0,due=0,requests=0;
  std::string fault;
  void advance(uint64_t now)override{
    cycle=now;if(fault=="unsolicited"&&!requests){answer=Response{99,0,false};return;}
    if(answer&&fault=="unstable"){answer->data^=1;return;}
    if(answer&&fault=="withdraw"){answer.reset();pending.reset();return;}
    if(!pending||answer||now<due)return;
    Response r{pending->id,0,fault=="error"};if(fault=="token")++r.id;
    auto &t=regions[pending->region];require(pending->offset+pending->bytes<=t.storage->bytes,"foreign memory request out of bounds");
    if(pending->write)std::memcpy(t.storage->writable+pending->offset,&pending->data,pending->bytes);
    else std::memcpy(&r.data,t.storage->data+pending->offset,pending->bytes);
    answer=r;
  }
  bool request_ready()const override{return !pending&&cycle%3==0;}
  void submit(const Request &r)override{require(request_ready(),"external request was not accepted");pending=r;due=cycle+2+r.id%5;++requests;}
  std::optional<Response> response()const override{return answer;}
  void consume_response()override{require(bool(answer),"consumed missing external response");answer.reset();pending.reset();}
};
}
int main(int argc,char **argv){
  try{require(argc==3,"usage: vector-external-memory job.json output-directory");std::ifstream stream(argv[1]);Json::Value job;stream>>job;const auto &node=job["node"];
    Values virtual_values;for(const auto &name:job["assets"].getMemberNames())virtual_values[name]=virtual_tensor(job["assets"][name]);
    Tensor metadata=virtual_tensor(node["output"]);ForeignMemory port;Assets loader;port.fault=job.get("fault","").asString();
    for(Json::ArrayIndex i=0;i<node["vector_program"]["input_dtypes"].size();++i){const auto &arg=node["args"][i];if(arg.isObject()&&arg.isMember("value"))port.regions[i]=loader.load(job["external_values"][arg["value"].asString()]);}
    port.regions[2]=Tensor::allocate(metadata.type,metadata.sizes);
    Simulator simulator(node,virtual_values,metadata,Options::parse(job["options"]),&port);while(simulator.tick()){}
    auto report=simulator.result();require(report["done"].asBool()&&report["external_memory_port"].asBool(),"external vector run incomplete");require(!metadata.storage->data&&!metadata.storage->writable,"virtual output unexpectedly has local data");
    report["virtual_tensor_backing_used"]=true;
    auto directory=std::filesystem::path(argv[2]);std::filesystem::create_directories(directory);std::ofstream output(directory/"output.bin",std::ios::binary);if(port.regions[2].storage->bytes)output.write(reinterpret_cast<const char*>(port.regions[2].storage->data),port.regions[2].storage->bytes);require(bool(output),"cannot write external result");
    std::ofstream json(directory/"result.json");json<<report<<'\n';require(bool(json),"cannot write external report");std::cout<<"VECTOR_EXTERNAL_MEMORY_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"VECTOR_EXTERNAL_MEMORY_FAIL: "<<error.what()<<std::endl;return 1;}
}
