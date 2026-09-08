#pragma once
#include "mm.h"
#include <jsoncpp/json/json.h>
#include <memory>
#include <string>

namespace mlx::wide_memory {
struct Inputs {
  bool reset=false,ar_valid=false,aw_valid=false,w_valid=false,w_last=false,r_ready=false,b_ready=false;
  uint64_t ar_address=0,aw_address=0,w_data=0;
  unsigned ar_id=0,ar_size=3,ar_len=0,ar_burst=1,aw_id=0,aw_size=3,aw_len=0,aw_burst=1,w_strobe=0;
};
struct Outputs {
  bool ar_ready=false,aw_ready=false,w_ready=false,r_valid=false,r_last=false,b_valid=false;
  uint64_t r_data=0;
  unsigned r_id=0,r_response=0,b_id=0,b_response=0;
};
// Preserve the existing mm_magic AXI timing, but explicitly translate a
// configured physical range. No 32-bit narrowing or modulo address aliases.
class Memory {
  uint64_t base,capacity,cycle=0,ar_count=0,aw_count=0,r_count=0,w_count=0,b_count=0,read_bytes=0,write_bytes=0;
  uint64_t outstanding_reads=0,outstanding_writes=0;
  uint64_t max_read_address=0,max_write_address=0;
  std::unique_ptr<mm_magic_t> backend;
  bool writing=false;
  uint64_t write_address=0;
  unsigned write_size=0,write_remaining=0;
  Json::Value initialized{Json::arrayValue};
  uint64_t offset(uint64_t address,uint64_t bytes)const;
  uint64_t burst(uint64_t address,unsigned size,unsigned len)const;
public:
  Memory(uint64_t base,uint64_t capacity);
  Outputs eval()const;
  void tick(const Inputs &inputs);
  bool idle()const;
  void preload_elf(const std::string &path); // Initialization, not CPU/DMA time.
  void preload_segments(const Json::Value &segments); // File bytes + optional zero tail.
  void inspect(uint64_t address,void *data,size_t bytes)const;
  Json::Value report()const;
};
}
