#include "VMLXWideMemory.h"
#include <verilated.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>

double simulation_time=0;
double sc_time_stamp(){return simulation_time;}
namespace {void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}}
int main(int argc,char **argv){
  Verilated::commandArgs(argc,argv);
  try{
    VMLXWideMemory memory;
    memory.axi_ar_valid=memory.axi_aw_valid=memory.axi_w_valid=0;
    memory.axi_r_ready=memory.axi_b_ready=0;
    auto edge=[&]{memory.clock=0;memory.eval();++simulation_time;memory.clock=1;memory.eval();++simulation_time;};
    memory.reset=1;for(unsigned i=0;i<4;++i)edge();memory.reset=0;edge();
    auto store=[&](uint64_t address,uint64_t value,unsigned size,unsigned mask){
      memory.axi_aw_valid=1;memory.axi_aw_bits_addr=address;memory.axi_aw_bits_id=3;memory.axi_aw_bits_len=0;memory.axi_aw_bits_size=size;memory.axi_aw_bits_burst=1;
      memory.clock=0;memory.eval();while(!memory.axi_aw_ready){edge();memory.clock=0;memory.eval();}edge();memory.axi_aw_valid=0;
      memory.axi_w_valid=1;memory.axi_w_bits_data=value;memory.axi_w_bits_strb=mask;memory.axi_w_bits_last=1;
      memory.clock=0;memory.eval();check(memory.axi_w_ready,"DPI write not ready after AW");edge();memory.axi_w_valid=0;
      while(!memory.axi_b_valid)edge();check(memory.axi_b_bits_id==3&&memory.axi_b_bits_resp==0,"bad DPI write response");
      edge();check(memory.axi_b_valid,"DPI B response did not hold");memory.axi_b_ready=1;edge();memory.axi_b_ready=0;
    };
    auto load=[&](uint64_t address){
      memory.axi_ar_valid=1;memory.axi_ar_bits_addr=address;memory.axi_ar_bits_id=5;memory.axi_ar_bits_size=3;memory.axi_ar_bits_len=0;memory.axi_ar_bits_burst=1;
      memory.clock=0;memory.eval();check(memory.axi_ar_ready,"DPI read not ready");edge();memory.axi_ar_valid=0;
      while(!memory.axi_r_valid)edge();auto value=memory.axi_r_bits_data;check(memory.axi_r_bits_id==5&&memory.axi_r_bits_last&&!memory.axi_r_bits_resp,"bad DPI read response");
      edge();check(memory.axi_r_valid&&memory.axi_r_bits_data==value,"DPI read changed on stall");memory.axi_r_ready=1;edge();memory.axi_r_ready=0;return value;
    };
    constexpr uint64_t low=UINT64_C(0x80001000),high=low+(UINT64_C(1)<<32),tail=UINT64_C(0x80000000)+(UINT64_C(16)<<30)-8;
    store(low,UINT64_C(0x123456789abcdef0),3,255);store(high,UINT64_C(0xfedcba9876543210),3,255);
    check(load(low)==UINT64_C(0x123456789abcdef0)&&load(high)==UINT64_C(0xfedcba9876543210),"SV/DPI narrowed addresses or aliased memory");
    store(high+6,UINT64_C(0x3c003c003c003c00),1,192);check(load(high)==UINT64_C(0x3c00ba9876543210),"SV/DPI corrupted subword mask/data");
    store(tail,UINT64_MAX,3,255);check(load(tail)==UINT64_MAX,"SV/DPI upper memory bound failed");
    memory.final();std::cout<<"WIDE_MEMORY_DPI_PASS\n";
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
