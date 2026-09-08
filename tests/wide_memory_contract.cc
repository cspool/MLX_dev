#include "memory.hh"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace mlx::wide_memory;
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
template<class F> void rejects(F call){bool rejected=false;try{call();}catch(const std::exception &){rejected=true;}check(rejected,"invalid wide-memory action accepted");}
void store(Memory &memory,uint64_t address,uint64_t data,unsigned size=3){
  Inputs in;in.aw_valid=true;in.aw_address=address;in.aw_id=7;in.aw_size=size;memory.tick(in);
  in={};in.w_valid=true;in.w_last=true;in.w_data=data;in.w_strobe=((1u<<(1u<<size))-1u)<<unsigned(address&7);memory.tick(in);
  auto held=memory.eval();check(held.b_valid&&held.b_id==7,"write response missing");memory.tick({});check(memory.eval().b_valid,"write response not held");in={};in.b_ready=true;memory.tick(in);
}
uint64_t load(Memory &memory,uint64_t address){Inputs in;in.ar_valid=true;in.ar_address=address;in.ar_id=11;memory.tick(in);auto result=memory.eval();check(result.r_valid&&result.r_id==11&&result.r_last,"read response missing");memory.tick({});check(memory.eval().r_data==result.r_data,"read changed under backpressure");in={};in.r_ready=true;memory.tick(in);return result.r_data;}
}
int main(int argc,char **argv){
  try{
    constexpr uint64_t base=UINT64_C(0x80000000),size=UINT64_C(16)<<30,low=base+4096,high=low+(UINT64_C(1)<<32);
    if(argc==1){
      Memory memory(base,size);check(load(memory,low)==0,"initial RAM must match zero-initialized mm_magic");
      store(memory,low,UINT64_C(0x1122334455667788));store(memory,high,UINT64_C(0xa1b2c3d4e5f60718));
      check(load(memory,low)==UINT64_C(0x1122334455667788)&&load(memory,high)==UINT64_C(0xa1b2c3d4e5f60718),"addresses separated by 4GiB aliased");
      store(memory,high+2,UINT64_C(0x3c003c003c003c00),1);
      check(load(memory,high)==UINT64_C(0xa1b2c3d43c000718),"narrow write lanes corrupted");
      store(memory,base+size-8,UINT64_MAX);check(load(memory,base+size-8)==UINT64_MAX,"upper capacity boundary truncated");
      Inputs in;in.ar_valid=true;in.ar_address=base-8;rejects([&]{memory.tick(in);});in.ar_address=base+size;rejects([&]{memory.tick(in);});in.ar_address=UINT64_MAX-7;rejects([&]{memory.tick(in);});
      in.ar_address=high+1;rejects([&]{memory.tick(in);});in.ar_address=high;in.ar_size=1;in.ar_len=2;rejects([&]{memory.tick(in);});in.ar_size=3;in.ar_len=1;in.ar_burst=2;rejects([&]{memory.tick(in);});
      in={};in.aw_valid=true;in.aw_address=high;memory.tick(in);Inputs reset;reset.reset=true;rejects([&]{memory.tick(reset);});
      in={};in.w_valid=true;in.w_last=false;in.w_strobe=255;rejects([&]{memory.tick(in);});in.w_last=true;memory.tick(in);in={};in.b_ready=true;memory.tick(in);
      in={};in.ar_valid=true;in.ar_address=high;in.ar_len=1;memory.tick(in);check(!memory.eval().r_last,"burst last asserted early");in={};in.r_ready=true;memory.tick(in);check(memory.eval().r_last,"burst last missing");memory.tick(in);
      check(memory.idle(),"wide memory did not drain");std::cout<<memory.report()<<'\n';return 0;
    }
    check(argc==2,"usage: wide-memory-contract [job.json]");std::ifstream input(argv[1]);Json::Value job;input>>job;Memory memory(job["base"].asUInt64(),job["bytes"].asUInt64());
    if(job.isMember("elf"))memory.preload_elf(job["elf"].asString());
    if(job.isMember("segments"))memory.preload_segments(job["segments"]);
    Json::Value observed{Json::arrayValue};
    for(const auto &region:job["inspect"]){std::vector<uint8_t> data(size_t(region["bytes"].asUInt64()));memory.inspect(region["address"].asUInt64(),data.data(),data.size());Json::Value bytes{Json::arrayValue};for(auto b:data)bytes.append(unsigned(b));observed.append(bytes);}
    auto report=memory.report();report["observed"]=observed;std::cout<<report<<'\n';
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
