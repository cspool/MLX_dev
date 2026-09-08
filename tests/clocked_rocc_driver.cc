#include "adapter.hh"
#include "progress.hh"
#include "mapped_memory.hh"
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace mlx;
namespace {void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}}
int main(int argc,char **argv){
  try{
    check(argc==3,"usage: clocked-rocc-driver job.json report.json");std::ifstream input(argv[1]);Json::Value job;input>>job;
    clocked_device::Options options;options.matrix=matrix_schedule::Options::parse(job["matrix_options"]);options.vector=vector_schedule::Options::parse(job["vector_options"]);options.memory=memory_model::ScheduleOptions::parse(job["memory_options"]);
    clocked_rocc::Adapter device(40,job.get("tag_bits",2).asUInt(),options);physical_device::MappedMemory memory(job["bytes"].asUInt64());uint64_t base=job["base"].asUInt64(),cycle=0;
    std::unique_ptr<clocked_rocc::Progress> observer;
    if(job.isMember("progress"))observer=std::make_unique<clocked_rocc::Progress>(job["progress"].asString(),job.get("progress_map",Json::Value(Json::arrayValue)),job.get("progress_period",10).asUInt64(),job.get("progress_limit",50000).asUInt64());
    auto write=[&](const Json::Value &segment){std::vector<uint8_t> raw;for(const auto &v:segment["data"])raw.push_back(uint8_t(v.asUInt()));memory.write(segment["address"].asUInt64()-base,raw.data(),raw.size());};
    for(const auto &segment:job["initial"])write(segment);
    struct Pending {clocked_rocc::Outputs request;uint64_t due;};std::optional<Pending> pending;
    bool busy_guard=false,held_guard=false;Json::Value replies{Json::arrayValue};
    auto edge=[&](clocked_rocc::Inputs in){
      check(cycle<2000000,"RoCC test watchdog");auto out=device.eval(in);
      in.memory_ready=cycle%3==0;
      if(pending&&cycle>=pending->due){
        const auto &q=pending->request;unsigned width=1u<<q.memory_size;uint64_t data=0;
        // HellaCache selects addressed byte lanes from the already expanded
        // 64-bit store bus. It does not shift a low-lane logical payload here.
        if(q.memory_write){auto selected=q.memory_data>>(8*(q.memory_address&7));memory.write(q.memory_address-base,&selected,width,1);}
        else memory.read(q.memory_address-base,&data,width);
        in.memory_response_valid=true;in.memory_response_tag=q.memory_tag;in.memory_response_data=data;
        if(job.get("wrong_tag",false).asBool())in.memory_response_tag^=1;
        pending.reset();
      }
      if(out.memory_valid&&in.memory_ready){check(!pending,"two test cache requests accepted");pending=Pending{out,cycle+4};}
      auto launches=device.launch_count(),terminals=device.terminal_count();device.tick(in);++cycle;
      if(observer)observer->sample(device,launches!=device.launch_count(),terminals!=device.terminal_count());
      if(pending&&!busy_guard){bool rejected=false;try{device.reset();}catch(const std::exception &){rejected=true;}check(rejected,"reset dropped in-flight cache transaction");busy_guard=true;}
      return out;
    };
    auto command=[&](unsigned funct,uint64_t a,uint64_t b,unsigned privilege=3){
      clocked_rocc::Inputs in;in.command_valid=true;in.funct=funct;in.xd=true;in.rd=11;in.rs1=a;in.rs2=b;in.privilege=privilege;
      while(!edge(in).command_ready){}
      in.command_valid=false;
      while(!device.eval(in).response_valid)edge(in);
      auto captured=device.eval(in);
      for(unsigned hold=0;hold<7;++hold){auto held=edge(in);check(held.response_valid&&held.rd==11&&held.data==captured.data,"CPU response changed under backpressure");}
      held_guard=true;in.response_ready=true;auto accepted=edge(in);check(accepted.response_valid,"CPU response disappeared");replies.append(Json::UInt64(accepted.data));return accepted.data;
    };
    check(command(3,14,0)==clocked_rocc::ABI_MAGIC,"RoCC ABI query mismatch");
    check(command(3,999,0)==UINT64_MAX,"unregistered status query accepted");
    if(job.get("reject_frontend",false).asBool()){
      check(command(0,0,0)==4,"invalid function not rejected");
      check(command(1,base+4096,1088,0)==4,"non-machine launch not rejected");
      check(command(1,base+4097,1088)==4,"unaligned launch not rejected");
      check(device.report()["requests"].asUInt64()==0,"frontend rejection touched memory");
    }
    for(const auto &item:job["commands"]){
      write(item);check(command(1,item["address"].asUInt64(),item["data"].size())==1,"valid launch did not report busy");
      check(command(2,0,0)==2,"WAIT did not return completed status");
    }
    auto report=device.report();report["outputs"]=Json::arrayValue;
    if(observer){observer->sample(device,false,false,true);report["progress_observer"]=observer->status();}
    for(const auto &segment:job["outputs"]){std::vector<uint8_t> raw(size_t(segment["bytes"].asUInt64()));memory.read(segment["address"].asUInt64()-base,raw.data(),raw.size());Json::Value values{Json::arrayValue};for(auto value:raw)values.append(unsigned(value));report["outputs"].append(values);}
    report["reply_hold_checked"]=held_guard;report["busy_reset_checked"]=busy_guard;report["replies"]=replies;
    check(!pending&&!report["cache_request_owned"].asBool()&&!report["cpu_response_pending"].asBool(),"RoCC ended with owned handshake");
    device.reset();std::ofstream output(argv[2]);output<<report<<'\n';
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
