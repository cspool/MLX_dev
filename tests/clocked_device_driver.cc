#include "device.hh"
#include "mapped_memory.hh"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

using namespace mlx;
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
std::vector<uint8_t> bytes(const Json::Value &value){std::vector<uint8_t> out;for(const auto &v:value){check(v.isUInt()&&v.asUInt()<256,"invalid byte input");out.push_back(uint8_t(v.asUInt()));}return out;}
template<class F> void rejects(F action){bool rejected=false;try{action();}catch(const std::exception &){rejected=true;}check(rejected,"busy reset/launch unexpectedly accepted");}
}
int main(int argc,char **argv){
  try{
    check(argc==3,"usage: clocked-device-driver job.json report.json");std::ifstream input(argv[1]);Json::Value job;input>>job;
    clocked_device::Options options;options.matrix=matrix_schedule::Options::parse(job["matrix_options"]);options.vector=vector_schedule::Options::parse(job["vector_options"]);options.memory=memory_model::ScheduleOptions::parse(job["memory_options"]);
    options.max_busy_cycles=job.get("max_busy_cycles",1000000).asUInt64();clocked_device::Device device(options);
    uint64_t base=job["base"].asUInt64(),capacity=job["bytes"].asUInt64();physical_device::MappedMemory memory(capacity);
    auto at=[&](uint64_t address){check(address>=base,"test bus address below memory");return address-base;};
    for(const auto &segment:job["initial"]){auto raw=bytes(segment["data"]);memory.write(at(segment["address"].asUInt64()),raw.data(),raw.size());}
    auto latency=job.get("latency",2).asUInt64(),accept_period=job.get("accept_period",1).asUInt64(),nack_every=job.get("nack_every",0).asUInt64();
    check(latency&&accept_period,"test bus timings must be positive");
    std::optional<model_io::PhysicalRequest> pending;uint64_t due=0,ticks=0;std::set<uint64_t> nacked;
    Json::Value attempts{Json::arrayValue},commits{Json::arrayValue},launches{Json::arrayValue};bool guarded=false;
    for(const auto &command:job["commands"]){
      if(device.complete()&&job.get("reset_between",false).asBool())device.reset();
      auto raw=bytes(command["data"]);auto address=command["address"].asUInt64();
      memory.write(at(address),raw.data(),raw.size());device.launch(address,raw.size(),command["source_id"].asUInt64());
      while(!device.complete()){
        check(ticks<2000000,"test driver watchdog");
        auto before=device.report();for(unsigned p=0;p<job.get("status_polls",0).asUInt();++p)check(device.report()==before,"status polling changed device state");
        clocked_device::Inputs edge;auto request=device.request();edge.memory_ready=ticks%accept_period==0;
        if(pending&&ticks>=due){
          auto q=*pending;
          if(nack_every&&q.id%nack_every==0&&nacked.insert(q.id).second)edge.nack=q.id;
          else{
            model_io::Response response{q.id,0,false};
            auto fault=job.get("fault","").asString();
            try{
              if((fault=="descriptor_error"&&q.address<base+65536)||(fault=="write_error"&&q.write))throw std::runtime_error("injected bus fault");
              if(q.write)memory.write(at(q.address),&q.data,q.bytes,1);else memory.read(at(q.address),&response.data,q.bytes);
            }catch(const std::exception &){response.error=true;}
            Json::Value commit;commit["id"]=Json::UInt64(q.id);commit["cycle"]=Json::UInt64(ticks);commit["write"]=q.write;commit["address"]=Json::UInt64(q.address);commit["bytes"]=q.bytes;commit["error"]=response.error;commits.append(commit);
            if(fault=="wrong_token")++response.id;
            edge.response=response;
          }
          pending.reset();
        }
        if(request&&edge.memory_ready){
          check(!pending,"test bus accepted two outstanding requests");pending=request;due=ticks+latency;
          Json::Value attempt;attempt["cycle"]=Json::UInt64(ticks);attempt["id"]=Json::UInt64(request->id);attempt["address"]=Json::UInt64(request->address);attempt["write"]=request->write;attempt["bytes"]=request->bytes;attempt["data"]=Json::UInt64(request->data);attempts.append(attempt);
        }
        device.tick(edge);++ticks;
        if(pending&&!guarded){rejects([&]{device.reset();});rejects([&]{device.launch(address,raw.size(),999);});guarded=true;}
      }
      check(!pending&&!device.busy(),"terminal device still has memory ownership");launches.append(device.report());
      if(!device.report()["done"].asBool()&&!job.get("recover",false).asBool())break;
    }
    Json::Value outputs{Json::arrayValue};
    if(device.report()["done"].asBool())for(const auto &segment:job["outputs"]){
      std::vector<uint8_t> raw(size_t(segment["bytes"].asUInt64()));memory.read(at(segment["address"].asUInt64()),raw.data(),raw.size());Json::Value data{Json::arrayValue};for(auto v:raw)data.append(unsigned(v));outputs.append(data);
    }
    Json::Value report;report["launches"]=launches;report["attempts"]=attempts;report["commits"]=commits;report["outputs"]=outputs;report["reset_while_busy_rejected"]=guarded;report["clock_ticks"]=Json::UInt64(ticks);
    device.reset();report["after_reset"]=device.report();report["mlx_system_verified"]=false;report["inference_performance_eligible"]=false;
    std::ofstream output(argv[2]);output<<report<<'\n';
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
