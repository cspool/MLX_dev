#include "adapter.hh"
#include "profile.hh"
#include "progress.hh"
#include <svdpi.h>
#include <cstdlib>
#include <fstream>
#include <iostream>

#ifndef MLX_CLOCKED_BUILD_ID
#define MLX_CLOCKED_BUILD_ID "unbound"
#endif

namespace {
struct Instance {
  mlx::model_image::Profile profile;
  mlx::clocked_rocc::Adapter adapter;
  std::unique_ptr<mlx::clocked_rocc::Progress> progress;
  Instance(unsigned bits,unsigned tags,mlx::model_image::Profile value):profile(std::move(value)),adapter(bits,tags,profile.options){}
};
}

extern "C" void *mlx_clocked_rocc_create(unsigned int bits,unsigned int tags){
  try{
    Json::Value requested;
    if(const char *path=std::getenv("MLX_CLOCKED_PROFILE")){std::ifstream input(path);if(!input)throw std::runtime_error("cannot open system profile");input>>requested;}
    auto instance=std::make_unique<Instance>(bits,tags,mlx::model_image::parse_profile(requested));
    if(const char *path=std::getenv("MLX_CLOCKED_PROGRESS")){
      Json::Value mapping(Json::arrayValue);
      if(const char *map=std::getenv("MLX_CLOCKED_LAUNCH_MAP")){std::ifstream input(map);if(!input)throw std::runtime_error("cannot open launch progress map");input>>mapping;}
      uint64_t period=1000000;if(const char *value=std::getenv("MLX_CLOCKED_PROGRESS_PERIOD")){size_t consumed=0;period=std::stoull(value,&consumed);if(consumed!=std::string(value).size())throw std::runtime_error("invalid progress interval");}
      instance->progress=std::make_unique<mlx::clocked_rocc::Progress>(path,mapping,period);
    }
    return instance.release();
  }catch(const std::exception &e){std::cerr<<"MLX_CLOCKED_CREATE_ERROR "<<e.what()<<'\n';return nullptr;}
}
extern "C" void mlx_clocked_rocc_reset(void *model){static_cast<Instance*>(model)->adapter.reset();}
extern "C" void mlx_clocked_rocc_destroy(void *model){
  if(!model)return;
  auto *instance=static_cast<Instance*>(model);auto report=instance->adapter.report();report["build_identity"]=MLX_CLOCKED_BUILD_ID;report["effective_profile"]=instance->profile.effective;
  if(instance->progress){instance->progress->sample(instance->adapter,false,false,true);report["progress_observer"]=instance->progress->status();}
  if(const char *path=std::getenv("MLX_CLOCKED_REPORT")){std::ofstream output(path);output<<report<<'\n';}
  std::cout<<"MLX_CLOCKED_DEVICE_FINAL launches="<<report["launches"].asUInt64()<<" requests="<<report["requests"].asUInt64()<<" responses="<<report["responses"].asUInt64()<<'\n';delete instance;
}
extern "C" void mlx_clocked_rocc_tick(void *model,svBit valid,unsigned int funct,svBit xd,unsigned int rd,unsigned long long rs1,unsigned long long rs2,
    unsigned int privilege,svBit response_ready,svBit memory_ready,svBit memory_response_valid,unsigned int memory_response_tag,unsigned long long memory_response_data){
  mlx::clocked_rocc::Inputs input;input.command_valid=valid;input.funct=funct;input.xd=xd;input.rd=rd;input.rs1=rs1;input.rs2=rs2;input.privilege=privilege;
  input.response_ready=response_ready;input.memory_ready=memory_ready;input.memory_response_valid=memory_response_valid;input.memory_response_tag=memory_response_tag;input.memory_response_data=memory_response_data;
  auto *instance=static_cast<Instance*>(model);auto launches=instance->adapter.launch_count(),terminals=instance->adapter.terminal_count();
  instance->adapter.tick(input);
  if(instance->progress)instance->progress->sample(instance->adapter,launches!=instance->adapter.launch_count(),terminals!=instance->adapter.terminal_count());
}
extern "C" void mlx_clocked_rocc_eval(void *model,unsigned long long,svBit reset,unsigned int funct,svBit response_ready,
    svBit *ready,svBit *response_valid,unsigned int *rd,unsigned long long *data,svBit *busy,svBit *memory_valid,svBit *memory_write,
    unsigned long long *address,unsigned long long *write_data,unsigned int *tag,unsigned int *size,unsigned int *mask,unsigned int *privilege){
  mlx::clocked_rocc::Outputs out;
  if(model&&!reset){mlx::clocked_rocc::Inputs input;input.funct=funct;input.response_ready=response_ready;out=static_cast<Instance*>(model)->adapter.eval(input);}
  *ready=out.command_ready;*response_valid=out.response_valid;*rd=out.rd;*data=out.data;*busy=out.busy;*memory_valid=out.memory_valid;*memory_write=out.memory_write;
  *address=out.memory_address;*write_data=out.memory_data;*tag=out.memory_tag;*size=out.memory_size;*mask=out.memory_mask;*privilege=out.privilege;
}
