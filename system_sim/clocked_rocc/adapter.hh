#pragma once
#include "device.hh"

namespace mlx::clocked_rocc {
constexpr uint64_t ABI_MAGIC=UINT64_C(0x4d4c58434c4b0001);
struct Inputs {
  bool command_valid=false,xd=false,response_ready=false,memory_ready=false,memory_response_valid=false;
  unsigned funct=0,rd=0,privilege=3,memory_response_tag=0;
  uint64_t rs1=0,rs2=0,memory_response_data=0;
};
struct Outputs {
  bool command_ready=false,response_valid=false,busy=false,memory_valid=false,memory_write=false;
  unsigned rd=0,memory_tag=0,memory_size=0,memory_mask=0,privilege=3;
  uint64_t data=0,memory_address=0,memory_data=0;
};
// Connect at SimpleHellaCacheIF's requestor side. That existing Chipyard layer
// owns cache NACK/replay and stage-one store data; do not duplicate its replay.
class Adapter {
  clocked_device::Device device;
  unsigned tag_bits,privilege=3;
  struct Owner {uint64_t logical;unsigned wire;};
  std::optional<Owner> owner;
  bool response_valid=false,active_launch=false;
  unsigned response_rd=0;
  uint64_t response_data=0,commands=0,launches=0,requests=0,responses=0;
  std::string frontend_error;
  Json::Value windows{Json::arrayValue};
  uint64_t status(uint64_t index)const;
public:
  Adapter(unsigned address_bits,unsigned tag_bits,clocked_device::Options options={});
  Outputs eval(const Inputs &inputs)const;
  void tick(const Inputs &inputs);
  void reset();
  Json::Value report()const;
  Json::Value progress()const;
  uint64_t launch_count()const{return launches;}
  uint64_t terminal_count()const{return windows.size();}
};
}
