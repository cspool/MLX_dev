#include "physical_memory.h"
#include <functional>
#include <iostream>

using namespace mlx::tensor_model;
using namespace mlx::model_storage;
using namespace mlx::model_system;
using namespace mlx::model_io;
namespace {
void rejects(const std::function<void()> &operation,const char *message){
  bool caught=false;try{operation();}catch(const std::exception &error){caught=true;require(std::string(error.what()).find(message)!=std::string::npos,"wrong physical memory rejection: "+std::string(error.what()));}
  require(caught,"physical memory accepted a prohibited operation");
}
Response transact(PhysicalMemory &memory,PhysicalRequest request,uint64_t &cycle){
  memory.advance(cycle++);require(memory.request_ready(),"physical contract port not ready");memory.submit(request);
  for(unsigned step=0;step<100;++step){memory.advance(cycle++);if(auto response=memory.response()){memory.consume_response();return *response;}}
  throw std::runtime_error("physical contract transaction did not complete");
}
}
int main(){
  try{
    constexpr uint64_t base=uint64_t(1)<<32;
    {
      Arena arena(base,128);PhysicalMemory memory(arena,{1,1,0,1000});auto metadata=arena.allocate(DType::I64,{2});auto data=Tensor::allocate(DType::I64,{2});memory.bind(metadata,data,false);uint64_t cycle=0;
      rejects([&]{memory.require_initialized(metadata);},"never written");
      transact(memory,{1,base,42,8,true},cycle);require(transact(memory,{2,base,0,8,false},cycle).data==42,"partial initialized read wrong");
      rejects([&]{memory.require_initialized(metadata);},"never written");
      // A zero-valued result still needs a committed store; zero-filled host
      // allocation is not evidence that the target produced those bytes.
      transact(memory,{3,base+8,0,8,true},cycle);memory.require_initialized(metadata);
      require(transact(memory,{4,base+8,0,8,false},cycle).data==0,"zero store result wrong");
      require(memory.snapshot()["writes"].asUInt64()==2&&memory.snapshot()["reads"].asUInt64()==2,"physical commit accounting wrong");
    }
    {
      Arena arena(base,128);PhysicalMemory memory(arena,{1,1,0,1000});auto metadata=arena.allocate(DType::I64,{2});auto data=Tensor::allocate(DType::I64,{2});memory.bind(metadata,data,false);uint64_t cycle=0;
      transact(memory,{1,base,0,8,true},cycle);
      rejects([&]{transact(memory,{2,base+8,0,8,false},cycle);},"before initialization");
      require(memory.snapshot()["reads"].asUInt64()==0,"uninitialized read committed");
    }
    {
      Arena arena(base,128);PhysicalMemory memory(arena,{1,1,0,1000});auto metadata=arena.allocate(DType::I64,{1});auto data=Tensor::allocate(DType::I64,{1});data.set_integer(0,7);memory.bind(metadata,data,true);arena.seal_read_only(metadata);uint64_t cycle=0;
      rejects([&]{transact(memory,{1,base,42,8,true},cycle);},"read-only");require(data.integer(0)==7&&memory.snapshot()["writes"].asUInt64()==0,"sealed physical buffer changed");
    }
    {
      Arena arena(base,128);PhysicalMemory memory(arena,{1,1,0,1000});auto metadata=arena.allocate(DType::I64,{1});auto data=Tensor::allocate(DType::I64,{1});memory.bind(metadata,data,true);metadata={};uint64_t cycle=0;
      rejects([&]{transact(memory,{1,base,0,8,false},cycle);},"retired allocation");
    }
    {
      Arena arena(base,128),foreign(base,128);PhysicalMemory memory(arena);auto metadata=foreign.allocate(DType::I64,{1});auto data=Tensor::allocate(DType::I64,{1});
      rejects([&]{memory.bind(metadata,data,true);},"does not belong");
    }
    std::cout<<"PHYSICAL_MEMORY_CONTRACT_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"PHYSICAL_MEMORY_CONTRACT_FAIL: "<<error.what()<<std::endl;return 1;}
}
