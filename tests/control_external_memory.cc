#include "control_schedule.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace mlx::tensor_model;
using namespace mlx::control_schedule;
using namespace mlx::model_io;
namespace {
class ForeignMemory final:public MemoryPort {
public:
  std::array<Tensor,3> regions{};
  uint64_t cycle=0,due=0,requests=0,reads=0,writes=0;
  std::optional<Request> pending;
  std::optional<Response> answer;
  std::string fault;
  bool dirty_high_bits=false;
  void advance(uint64_t now)override{
    require(now>=cycle,"foreign clock regressed");cycle=now;
    if(fault=="unsolicited"&&!requests){answer=Response{99,0,false};return;}
    if(answer&&fault=="unstable"){answer->data^=1;return;}
    if(answer&&fault=="withdraw"){answer.reset();pending.reset();return;}
    if(!pending||answer||now<due)return;
    const auto &r=*pending;const auto &t=regions[r.region];require(t.storage&&r.offset<=t.storage->bytes&&r.bytes<=t.storage->bytes-r.offset,"foreign controller access out of bounds");
    Response response{r.id,0,fault=="error"};if(fault=="token")++response.id;
    if(!response.error){
      if(r.write){std::memcpy(t.storage->writable+r.offset,&r.data,r.bytes);++writes;}
      else{std::memcpy(&response.data,t.storage->data+r.offset,r.bytes);if(dirty_high_bits&&r.bytes<8)response.data|=UINT64_MAX<<(r.bytes*8);++reads;}
    }
    answer=response;
  }
  bool request_ready()const override{return !pending&&!answer&&cycle%3==0;}
  void submit(const Request &r)override{require(request_ready(),"foreign controller request not ready");pending=r;due=cycle+2+r.id%5;++requests;}
  std::optional<Response> response()const override{return answer;}
  void consume_response()override{require(bool(answer),"missing foreign controller response");answer.reset();pending.reset();}
};
Tensor virtual_tensor(Tensor real){real.storage=std::make_shared<Storage>(Storage{nullptr,nullptr,nullptr,real.storage->bytes});return real;}
Json::Value probe(const Json::Value &job){
  Leaf leaf;
  for(const auto &name:job["x"].getMemberNames())leaf.state.x.at(std::stoul(name))=job["x"][name].asUInt64();
  for(const auto &name:job["f"].getMemberNames())leaf.state.f.at(std::stoul(name))=job["f"][name].asUInt64();
  auto reference=leaf.state;leaf.begin(job["words"]);Json::Value trace(Json::arrayValue);
  while(!leaf.done()){Json::Value event;event["pc"]=leaf.position();event["word"]=leaf.word();leaf.step();event["next_pc"]=leaf.position();trace.append(event);}
  reference.run(job["words"]);require(leaf.state.x==reference.x&&leaf.state.f==reference.f&&leaf.state.retired==reference.retired&&leaf.state.branches_taken==reference.branches_taken&&leaf.state.fflags==reference.fflags,"stepped RV64 differs from synchronous leaf");
  Json::Value result;for(auto x:leaf.state.x)result["x"].append(Json::UInt64(x));for(auto f:leaf.state.f)result["f"].append(Json::UInt64(f));
  result["retired"]=Json::UInt64(leaf.state.retired);result["branches_taken"]=Json::UInt64(leaf.state.branches_taken);result["fflags"]=leaf.state.fflags;result["trace"]=trace;return result;
}
}

int main(int argc,char **argv){
  try{
    require(argc==3,"usage: control-external-memory job.json output-directory");std::ifstream input(argv[1]);Json::Value job;input>>job;Json::Value report;
    auto directory=std::filesystem::path(argv[2]);
    if(job["schema"]=="rv64_leaf_probe_v1")report=probe(job);
    else{
      const auto &node=job["node"];bool external=job.get("external",true).asBool();Assets loader;Values real_values,values;ForeignMemory port;
      port.fault=job.get("fault","").asString();port.dirty_high_bits=job.get("dirty_high_bits",false).asBool();
      for(const auto &name:job["assets"].getMemberNames()){
        const auto &spec=job["physical_values"].isMember(name)?job["physical_values"][name]:job["assets"][name];auto real=loader.load(spec);
        const auto &layout=job["layouts"][name];if(layout.isMember("shape"))real.sizes=shape(layout["shape"]);
        if(layout.isMember("strides"))real.steps=shape(layout["strides"]);
        if(layout.isMember("offset"))real.offset=layout["offset"].asInt64();
        real_values[name]=real;values[name]=external?virtual_tensor(real):real;
      }
      for(unsigned i=0;i<2&&i<node["args"].size();++i){const auto &arg=node["args"][i];if(arg.isObject()&&arg.isMember("value"))port.regions[i]=real_values.at(arg["value"].asString());}
      auto output=Tensor::allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));port.regions[2]=output;
      auto metadata=external?virtual_tensor(output):output;Simulator simulator(node,values,metadata,Options::parse(job.get("options",Json::Value(Json::objectValue))),external?&port:nullptr);
      if(!simulator.done()){bool rejected=false;try{simulator.result();}catch(const std::exception&){rejected=true;}require(rejected,"unfinished controller report escaped");}
      try{while(simulator.tick()){}}
      catch(const std::exception &error){
        if(job.get("expect_guard_mismatch",false).asBool()){
          require(node["kind"]=="guard"&&external&&std::string(error.what())=="control-flow guard mismatch","unexpected guard test failure");
          require(port.reads==1&&port.writes==0&&!simulator.done(),"guard mismatch wrote a result or failed to read the actual input");
          bool rejected=false;try{simulator.result();}catch(const std::exception&){rejected=true;}require(rejected,"failed guard exposed a success report");
          Json::Value failure;failure["classification"]="expected_guard_mismatch_not_success";
          failure["physical_reads"]=Json::UInt64(port.reads);failure["physical_writes"]=Json::UInt64(port.writes);
          failure["result_report_rejected"]=true;
          std::filesystem::create_directories(directory);std::ofstream file(directory/"failure-evidence.json");file<<failure<<'\n';require(bool(file),"cannot save guard failure evidence");
        }
        throw;
      }
      report=simulator.result();report["virtual_tensor_backing_used"]=external;
      if(external){require(!metadata.storage->data&&!metadata.storage->writable,"external controller used host backing");require(port.requests==report["dma_responses"].asUInt64()&&port.reads+port.writes==port.requests,"foreign controller did not drain");}
      report["physical_reads"]=Json::UInt64(port.reads);report["physical_writes"]=Json::UInt64(port.writes);
      mlx::control_model::Stats stats;auto oracle=mlx::control_model::execute(node,real_values,stats);
      require(output.storage->bytes==oracle.storage->bytes&&(!output.storage->bytes||std::memcmp(output.storage->data,oracle.storage->data,output.storage->bytes)==0),"controller output differs from synchronous binding");
      require(stats.instructions==report["instructions"].asUInt64()&&stats.branches==report["branches_taken"].asUInt64()&&stats.read_bytes==report["read_bytes"].asUInt64()&&stats.write_bytes==report["write_bytes"].asUInt64()&&stats.fflags==report["fflags_observed"].asUInt(),"controller instruction/traffic accounting differs from synchronous binding");
      report["functional_oracle_equal"]=true;
      std::filesystem::create_directories(directory);std::ofstream data(directory/"output.bin",std::ios::binary);if(output.storage->bytes)data.write(reinterpret_cast<const char*>(output.storage->data),output.storage->bytes);require(bool(data),"cannot write controller output");
    }
    report["rocket_execution_verified"]=false;report["mlx_system_verified"]=false;report["inference_performance_eligible"]=false;
    std::filesystem::create_directories(directory);std::ofstream json(directory/"result.json");json<<report<<'\n';require(bool(json),"cannot write controller report");std::cout<<"CONTROLLER_EXTERNAL_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"CONTROLLER_EXTERNAL_FAIL: "<<error.what()<<std::endl;return 1;}
}
