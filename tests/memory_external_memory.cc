#include "memory_schedule.h"
#include "../simulator_ext/model_io/address_space_port.h"
#include "../simulator_ext/model_io/queued_physical_port.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace mlx::tensor_model;
using namespace mlx::model_io;
using namespace mlx::memory_model;
namespace {
struct Allocation {uint64_t base;std::shared_ptr<Storage> metadata,backing;};
class ForeignMemory final:public PhysicalMemoryPort {
public:
  std::vector<Allocation> allocations;
  QueuedPhysicalPort queue;
  std::optional<PhysicalRequest> pending;
  std::optional<Response> answer;
  Json::Value traffic{Json::arrayValue};
  uint64_t cycle=0,due=0,reads=0,writes=0,read_bytes=0,write_bytes=0,last_retry=UINT64_MAX;
  bool buffered=false;
  std::string fault;
  const Allocation &allocation(const std::shared_ptr<Storage> &metadata)const{
    for(const auto &a:allocations)if(a.metadata==metadata)return a;
    throw std::runtime_error("unbound foreign allocation");
  }
  Tensor bind(Tensor real){
    Tensor virtual_value=real;virtual_value.storage=std::make_shared<Storage>();virtual_value.storage->bytes=real.storage->bytes;
    require(real.storage->bytes<0x100000,"test allocation exceeds slot");
    allocations.push_back({(uint64_t(1)<<32)+allocations.size()*0x100000,virtual_value.storage,real.storage});return virtual_value;
  }
  void accept(const PhysicalRequest &r){
    require(!pending,"test memory accepted a second transaction");pending=r;due=cycle+2+r.id%5;
    Json::Value t;t["id"]=Json::UInt64(r.id);t["address"]=Json::UInt64(r.address);t["bytes"]=r.bytes;t["write"]=r.write;t["cycle"]=Json::UInt64(cycle);traffic.append(t);
  }
  void advance(uint64_t now)override{
    require(now>=cycle,"foreign cycle regressed");cycle=now;
    if(buffered){queue.advance(now);if(queue.request_valid()&&now%3==0){auto r=*queue.presented_request();queue.accept_request();accept(r);}}
    if(!buffered&&fault=="unsolicited"&&traffic.empty()){answer=Response{99,0,false};return;}
    if(answer&&fault=="unstable"){answer->data^=1;return;}
    if(answer&&fault=="withdraw"){answer.reset();pending.reset();return;}
    if(!pending||answer||now<due)return;
    const auto request=*pending;
    if(buffered&&request.id%5==0&&last_retry!=request.id){last_retry=request.id;pending.reset();queue.receive_nack(request.id);return;}
    Response response{request.id,0,fault=="error"};if(fault=="token")++response.id;
    bool found=false;
    if(!response.error)for(const auto &a:allocations){
      if(request.address<a.base)continue;
      auto offset=request.address-a.base;
      if(offset>a.backing->bytes||request.bytes>a.backing->bytes-offset)continue;
      if(request.write){std::memcpy(a.backing->writable+offset,&request.data,request.bytes);++writes;write_bytes+=request.bytes;}
      else{std::memcpy(&response.data,a.backing->data+offset,request.bytes);++reads;read_bytes+=request.bytes;}
      found=true;break;
    }
    require(found||response.error,"foreign physical request outside allocations");
    pending.reset();if(buffered)queue.receive_response(response);else answer=response;
  }
  bool request_ready()const override{return buffered?queue.request_ready():!pending&&!answer&&cycle%3==0;}
  void submit(const PhysicalRequest &r)override{require(request_ready(),"foreign request not accepted");if(buffered)queue.submit(r);else accept(r);}
  std::optional<Response> response()const override{return buffered?queue.response():answer;}
  void consume_response()override{if(buffered)queue.consume_response();else{require(bool(answer),"missing foreign response");answer.reset();}}
};
// Deliberately does not police response identity/stability: fault tests must
// also exercise the transfer engine's own checks, not only AddressSpacePort.
class RawLogicalPort final:public MemoryPort {
  ForeignMemory &physical;std::vector<Region> regions;uint64_t origin;
public:
  RawLogicalPort(ForeignMemory &p,std::vector<Region> r,uint64_t o):physical(p),regions(std::move(r)),origin(o){}
  void advance(uint64_t cycle)override{physical.advance(origin+cycle);}
  bool request_ready()const override{return physical.request_ready();}
  void submit(const Request &r)override{require(r.region<regions.size(),"raw region out of bounds");physical.submit({r.id,regions[r.region].base+r.offset,r.data,r.bytes,r.write});}
  std::optional<Response> response()const override{return physical.response();}
  void consume_response()override{physical.consume_response();}
};
}

int main(int argc,char **argv){
  try{
    require(argc==3,"usage: memory-external-memory job.json output-directory");std::ifstream input(argv[1]);Json::Value job;input>>job;
    bool external=job.get("external",true).asBool();Assets loader;Values values,oracle_values;ForeignMemory physical;
    physical.fault=job.get("fault","").asString();physical.buffered=job.get("buffered_bridge",false).asBool();
    for(const auto &name:job["assets"].getMemberNames()){
      const auto &spec=job["physical_values"].isMember(name)?job["physical_values"][name]:job["assets"][name];auto real=loader.load(spec);
      require(dtype_name(real.type)==job["assets"][name]["dtype"].asString()&&real.sizes==shape(job["assets"][name]["shape"]),"physical asset metadata mismatch");
      oracle_values[name]=real;values[name]=external?physical.bind(real):real;
    }
    RequestTokens tokens;uint64_t origin=11;Json::Value reports{Json::arrayValue};Stats oracle_stats;
    Tensor last,oracle_last;
    for(const auto &node:job["nodes"]){
      const auto &p=node["memory_program"];bool view=p["mode"]=="view";Tensor output;
      if(!view){output=Tensor::allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));output.steps=shape(p["output_layout"]["strides"]);if(external)output=physical.bind(output);}
      std::vector<Region> regions;
      if(external){
        auto names=p["input_layouts"].getMemberNames();std::sort(names.begin(),names.end());
        for(const auto &name:names){const auto &a=physical.allocation(values.at(name).storage);regions.push_back({a.base,a.metadata->bytes,true,false});}
        if(!view){const auto &a=physical.allocation(output.storage);regions.push_back({a.base,a.metadata->bytes,false,true});}
        for(const auto &key:job["region_overrides"].getMemberNames()){
          unsigned i=std::stoul(key);require(i<regions.size(),"invalid region override");const auto &v=job["region_overrides"][key];
          if(v.isMember("base"))regions[i].base=v["base"].asUInt64();
          if(v.isMember("bytes"))regions[i].bytes=v["bytes"].asUInt64();
          if(v.isMember("readable"))regions[i].readable=v["readable"].asBool();
          if(v.isMember("writable"))regions[i].writable=v["writable"].asBool();
        }
      }
      std::unique_ptr<MemoryPort> address;
      if(external){
        if(job.get("raw_logical_port",false).asBool())address=std::make_unique<RawLogicalPort>(physical,regions,origin);
        else address=std::make_unique<AddressSpacePort>(physical,tokens,regions,origin);
      }
      Simulator simulator(node,values,ScheduleOptions::parse(job.get("options",Json::Value(Json::objectValue))),address.get(),view?nullptr:&output);
      if(!simulator.done()){
        bool rejected=false;try{simulator.output();}catch(const std::exception&){rejected=true;}require(rejected,"unfinished output escaped");
      }
      while(simulator.tick()){}
      auto report=simulator.result();require(simulator.done()&&report["dma_requests"]==report["dma_responses"],"memory execution did not drain");
      last=simulator.output();values[node["id"].asString()]=last;reports.append(report);origin+=report["cycles"].asUInt64()+1;
      // Compare with the synchronous plan only after target execution completes.
      // It never provides target inputs, predicates, indices, or intermediates.
      oracle_last=execute(node,oracle_values,oracle_stats);oracle_values[node["id"].asString()]=oracle_last;
      if(external)require(!last.storage->data&&!last.storage->writable,"external output had host data");
      for(const auto &name:node["release"]){values.erase(name.asString());oracle_values.erase(name.asString());}
    }
    require(!reports.empty(),"missing memory test nodes");
    auto target=last;if(external)target.storage=physical.allocation(last.storage).backing;
    target=target.materialize(target.type);auto expected=oracle_last.materialize(oracle_last.type);
    require(target.storage->bytes==expected.storage->bytes&&(!target.storage->bytes||std::memcmp(target.storage->data,expected.storage->data,target.storage->bytes)==0),"scheduled memory differs from functional plan");
    Json::Value report;report["windows"]=reports;report["physical_requests"]=physical.traffic;
    report["physical_commits"]["reads"]=Json::UInt64(physical.reads);report["physical_commits"]["writes"]=Json::UInt64(physical.writes);
    report["physical_commits"]["read_bytes"]=Json::UInt64(physical.read_bytes);report["physical_commits"]["write_bytes"]=Json::UInt64(physical.write_bytes);
    report["functional_oracle_equal"]=true;report["virtual_tensor_backing_used"]=external;
    report["mlx_system_verified"]=false;report["inference_performance_eligible"]=false;
    if(physical.buffered)report["transport"]=physical.queue.snapshot();
    auto directory=std::filesystem::path(argv[2]);std::filesystem::create_directories(directory);
    std::ofstream data(directory/"output.bin",std::ios::binary);if(target.storage->bytes)data.write(reinterpret_cast<const char*>(target.storage->data),target.storage->bytes);require(bool(data),"cannot write memory output");
    std::ofstream json(directory/"result.json");json<<report<<'\n';require(bool(json),"cannot write memory report");std::cout<<"MEMORY_EXTERNAL_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"MEMORY_EXTERNAL_FAIL: "<<error.what()<<std::endl;return 1;}
}
