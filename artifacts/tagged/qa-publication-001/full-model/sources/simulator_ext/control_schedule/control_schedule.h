#pragma once
#include "../control_model/control_program.h"
#include "../control_model/rv64_leaf.h"
#include "../model_io/memory_port.h"
#include <memory>

namespace mlx::control_schedule {
struct Options {
  unsigned dma_latency=8,alu_latency=1,multiply_latency=3,float_latency=3,branch_latency=1;
  unsigned request_period=1,response_period=1,trace_limit=200000;
  uint64_t max_cycles=10000000;
  bool trace=true;
  static Options parse(const Json::Value &value);
  void validate()const;
};

// Descriptor-bound tensor I/O and loop control, with actual RV64 ALU leaves.
// This is NOT a complete RISC-V processor: instructions are descriptor ROM,
// not fetched ELF, and tensor accesses are MemoryPort requests, not LD/ST.
// Regions 0/1 bind argument 0/1 backing storage; region 2 binds output. Unused
// argument regions may be unbound. External tensors may have null data pointers.
class Simulator {
public:
  Simulator(const Json::Value &node,const tensor_model::Values &values,
            tensor_model::Tensor output,Options options={},model_io::MemoryPort *port=nullptr);
  ~Simulator();
  bool tick();
  bool done()const;
  Json::Value result()const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

// Single-instruction stepping of the already-registered bounded leaf subset.
// ALU/FP semantics delegate to RV64; branches keep an explicit phase-local PC.
class Leaf {
  Json::Value code;
  unsigned pc=0,steps=0;
public:
  control_model::RV64 state;
  void begin(const Json::Value &words);
  bool done()const;
  unsigned word()const;
  unsigned position()const{return pc;}
  void step();
};
} // namespace mlx::control_schedule
