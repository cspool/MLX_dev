#include "memory.hh"
#include <vpi_user.h>
#include <svdpi.h>
#include <cstdlib>
#include <fstream>
#include <iostream>

extern "C" void *mlx_wide_memory_init(unsigned long long base,unsigned long long bytes,unsigned long long word,unsigned long long line){
  if(word!=8||line!=64)throw std::runtime_error("wide memory requires a 64-bit data bus and 64-byte cache line");
  auto memory=std::make_unique<mlx::wide_memory::Memory>(base,bytes);s_vpi_vlog_info info;
  if(!vpi_get_vlog_info(&info))throw std::runtime_error("wide memory cannot read simulator arguments");
  std::string elf,segments;
  for(int i=1;i<info.argc;++i){std::string arg(info.argv[i]);
    if(arg.rfind("+loadmem=",0)==0){if(!elf.empty()||arg.size()==9)throw std::runtime_error("duplicate/empty wide-memory ELF preload");elf=arg.substr(9);}
    if(arg.rfind("+mlx_memory_segments=",0)==0){if(!segments.empty()||arg.size()==21)throw std::runtime_error("duplicate/empty memory segment preload");segments=arg.substr(21);}
    if(arg=="+dramsim"||arg.rfind("+loadmem_addr=",0)==0)throw std::runtime_error("wide memory profile does not accept legacy DRAMSim/hex relocation options");
  }
  if(!elf.empty())memory->preload_elf(elf);
  if(!segments.empty()){std::ifstream input(segments);Json::Value rows;input>>rows;memory->preload_segments(rows);}
  if(const char *path=std::getenv("MLX_WIDE_MEMORY_INIT_REPORT")){
    std::ofstream output(path);output<<memory->report()<<'\n';
    if(!output)std::cerr<<"MLX_MEMORY_INIT_OBSERVER_ERROR cannot write initialization report\n";
  }
  return memory.release();
}
extern "C" void mlx_wide_memory_destroy(void *channel){
  if(!channel)return;
  auto *memory=static_cast<mlx::wide_memory::Memory*>(channel);auto report=memory->report();
  if(const char *path=std::getenv("MLX_WIDE_MEMORY_REPORT")){std::ofstream output(path);output<<report<<'\n';}
  std::cout<<"MLX_WIDE_MEMORY_FINAL bytes="<<report["bytes"].asUInt64()<<" ar="<<report["ar_requests"].asUInt64()<<" aw="<<report["aw_requests"].asUInt64()<<'\n';delete memory;
}
extern "C" void mlx_wide_memory_reset(void *channel){if(channel){mlx::wide_memory::Inputs in;in.reset=true;static_cast<mlx::wide_memory::Memory*>(channel)->tick(in);}}
extern "C" void mlx_wide_memory_tick(void *channel,svBit reset,svBit ar_valid,svBit *ar_ready,unsigned long long ar_address,unsigned int ar_id,unsigned int ar_size,unsigned int ar_len,unsigned int ar_burst,
    svBit aw_valid,svBit *aw_ready,unsigned long long aw_address,unsigned int aw_id,unsigned int aw_size,unsigned int aw_len,unsigned int aw_burst,
    svBit w_valid,svBit *w_ready,unsigned int w_strobe,unsigned long long w_data,svBit w_last,
    svBit *r_valid,svBit r_ready,unsigned int *r_id,unsigned int *r_response,unsigned long long *r_data,svBit *r_last,
    svBit *b_valid,svBit b_ready,unsigned int *b_id,unsigned int *b_response){
  mlx::wide_memory::Inputs in;in.reset=reset;in.ar_valid=ar_valid;in.ar_address=ar_address;in.ar_id=ar_id;in.ar_size=ar_size;in.ar_len=ar_len;in.ar_burst=ar_burst;
  in.aw_valid=aw_valid;in.aw_address=aw_address;in.aw_id=aw_id;in.aw_size=aw_size;in.aw_len=aw_len;in.aw_burst=aw_burst;in.w_valid=w_valid;in.w_strobe=w_strobe;in.w_data=w_data;in.w_last=w_last;in.r_ready=r_ready;in.b_ready=b_ready;
  auto *memory=static_cast<mlx::wide_memory::Memory*>(channel);memory->tick(in);auto out=memory->eval();
  *ar_ready=out.ar_ready;*aw_ready=out.aw_ready;*w_ready=out.w_ready;*r_valid=out.r_valid;*r_id=out.r_id;*r_response=out.r_response;*r_data=out.r_data;*r_last=out.r_last;*b_valid=out.b_valid;*b_id=out.b_id;*b_response=out.b_response;
}
