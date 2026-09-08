#include "matrix_schedule.h"
#include "../simulator_ext/model_io/address_space_port.h"
#include "../simulator_ext/model_io/queued_physical_port.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

using namespace mlx::tensor_model;
using namespace mlx::model_io;
namespace {
Tensor metadata(const Json::Value &p){Tensor t;t.type=dtype(p["dtype"].asString());t.sizes=shape(p["shape"]);t.steps=p.isMember("strides")?shape(p["strides"]):strides(t.sizes);t.offset=p.get("offset",0).asInt64();t.storage=std::make_shared<Storage>();t.storage->bytes=p.isMember("storage_elements")?p["storage_elements"].asUInt64()*element_bytes(t.type):t.numel()*element_bytes(t.type);return t;}
class ForeignPhysical final:public PhysicalMemoryPort {
public:
  struct Buffer {uint64_t base;Tensor value;};
  std::vector<Buffer> buffers;
  std::optional<PhysicalRequest> pending;
  std::optional<Response> answer;
  uint64_t cycle=0,due=0,requests=0;
  uint64_t committed_reads=0,committed_writes=0,read_bytes=0,write_bytes=0;
  bool generate_nack=false;
  std::string fault;
  Json::Value events{Json::arrayValue};
  void advance(uint64_t now)override{
    cycle=now;if(fault=="unsolicited"&&!requests){answer=Response{999,0,false};return;}
    if(answer&&fault=="unstable"){answer->data^=1;return;}
    if(answer&&fault=="withdraw"){answer.reset();pending.reset();return;}
    if(!pending||answer||now<due)return;
    auto q=*pending;Response result{q.id,0,fault=="error"};if(fault=="token")++result.id;bool found=false;
    if(generate_nack){answer=result;return;}
    for(auto &buffer:buffers)if(q.address>=buffer.base&&q.address-buffer.base<=buffer.value.storage->bytes&&q.bytes<=buffer.value.storage->bytes-(q.address-buffer.base)){
      auto offset=q.address-buffer.base;
      if(q.write){std::memcpy(buffer.value.storage->writable+offset,&q.data,q.bytes);++committed_writes;write_bytes+=q.bytes;}
      else{std::memcpy(&result.data,buffer.value.storage->data+offset,q.bytes);++committed_reads;read_bytes+=q.bytes;}
      found=true;break;
    }
    require(found,"physical request escaped supplied memory");answer=result;
  }
  bool request_ready()const override{return !pending&&cycle%3==0;}
  void submit(const PhysicalRequest &q)override{
    require(request_ready(),"physical request was not accepted");pending=q;due=cycle+2+q.id%5;++requests;
    Json::Value e(Json::objectValue);e["id"]=Json::UInt64(q.id);e["address"]=Json::UInt64(q.address);e["bytes"]=q.bytes;e["write"]=q.write;e["data"]=Json::UInt64(q.data);events.append(e);
  }
  std::optional<Response> response()const override{return answer;}
  void consume_response()override{require(bool(answer),"physical response missing");answer.reset();pending.reset();generate_nack=false;}
};
class BufferedForeign final:public PhysicalMemoryPort {
public:
  QueuedPhysicalPort queue;
  ForeignPhysical &memory;
  std::set<uint64_t> retried;
  explicit BufferedForeign(ForeignPhysical &m):memory(m){}
  void advance(uint64_t cycle)override{
    queue.advance(cycle);memory.advance(cycle);
    if(auto response=memory.response()){if(memory.generate_nack)queue.receive_nack(response->id);else queue.receive_response(*response);memory.consume_response();}
    if(queue.request_valid()&&memory.request_ready()){auto request=*queue.presented_request();queue.accept_request();memory.generate_nack=request.id%5==0&&retried.insert(request.id).second;memory.submit(request);}
  }
  bool request_ready()const override{return queue.request_ready();}
  void submit(const PhysicalRequest &q)override{queue.submit(q);}
  std::optional<Response> response()const override{return queue.response();}
  void consume_response()override{queue.consume_response();}
};
}
int main(int argc,char **argv){
  try{
    require(argc==3,"usage: matrix-external-memory job.json output-directory");std::ifstream input(argv[1]);Json::Value job;input>>job;
    auto a=metadata(job["a"]),b=metadata(job["b"]);Tensor bias;bool has_bias=job.isMember("bias");if(has_bias)bias=metadata(job["bias"]);
    auto m=job["m"].asUInt64(),n=job["n"].asUInt64(),k=job["k"].asUInt64();
    Json::Value out_spec(Json::objectValue);out_spec["dtype"]=job["program"]["output_dtype"];out_spec["shape"].append(Json::UInt64(m));out_spec["shape"].append(Json::UInt64(n));
    auto output=metadata(out_spec);ForeignPhysical physical;physical.fault=job.get("fault","").asString();Assets loader;
    std::vector<Region> regions;
    for(unsigned region=0;region<4;++region){uint64_t base=0x100000000ULL*(region+1);Tensor value;
      if(region==0)value=loader.load(job["physical_values"]["a"]);
      else if(region==1)value=loader.load(job["physical_values"]["b"]);
      else if(region==2&&has_bias)value=loader.load(job["physical_values"]["bias"]);
      else if(region==3)value=Tensor::allocate(output.type,output.sizes);
      auto bytes=value.storage?value.storage->bytes:0;regions.push_back(Region{base,bytes,region!=3,region==3});if(bytes||region==3)physical.buffers.push_back({base,value});
    }
    for(const auto &key:job["region_overrides"].getMemberNames()){unsigned index=std::stoul(key);require(index<regions.size(),"invalid region override");const auto &spec=job["region_overrides"][key];if(spec.isMember("base"))regions[index].base=spec["base"].asUInt64();if(spec.isMember("bytes"))regions[index].bytes=spec["bytes"].asUInt64();if(spec.isMember("readable"))regions[index].readable=spec["readable"].asBool();if(spec.isMember("writable"))regions[index].writable=spec["writable"].asBool();}
    BufferedForeign buffered(physical);bool use_buffer=job.get("buffered_bridge",false).asBool();
    RequestTokens tokens;AddressSpacePort port(use_buffer?static_cast<PhysicalMemoryPort&>(buffered):static_cast<PhysicalMemoryPort&>(physical),tokens,regions);
    mlx::matrix_schedule::Simulator simulator(job["program"],a,b,has_bias?&bias:nullptr,job.get("transposed_b",true).asBool(),job.get("a_batch",0).asUInt64(),job.get("b_batch",0).asUInt64(),m,n,k,output,0,mlx::matrix_schedule::Options::parse(job["options"]),&port);
    while(simulator.tick()){}auto report=simulator.result();require(report["done"].asBool()&&report["external_memory_port"].asBool(),"external matrix did not complete");require(!output.storage->data&&!output.storage->writable,"virtual output gained local backing");
    report["virtual_tensor_backing_used"]=true;report["physical_requests"]=physical.events;
    report["physical_commits"]["reads"]=Json::UInt64(physical.committed_reads);report["physical_commits"]["writes"]=Json::UInt64(physical.committed_writes);report["physical_commits"]["read_bytes"]=Json::UInt64(physical.read_bytes);report["physical_commits"]["write_bytes"]=Json::UInt64(physical.write_bytes);
    if(use_buffer)report["transport"]=buffered.queue.snapshot();
    const auto &result=physical.buffers.back().value;auto directory=std::filesystem::path(argv[2]);std::filesystem::create_directories(directory);std::ofstream data(directory/"output.bin",std::ios::binary);if(result.storage->bytes)data.write(reinterpret_cast<const char*>(result.storage->data),result.storage->bytes);require(bool(data),"cannot write external matrix output");std::ofstream json(directory/"result.json");json<<report<<'\n';require(bool(json),"cannot write external matrix report");
    std::cout<<"MATRIX_EXTERNAL_MEMORY_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"MATRIX_EXTERNAL_MEMORY_FAIL: "<<error.what()<<std::endl;return 1;}
}
