#pragma once
#include "memory_program.h"
#include "../model_io/memory_port.h"
#include <memory>

namespace mlx::memory_model {
struct ScheduleOptions {
  unsigned dma_latency=8,convert_latency=2;
  unsigned request_period=1,response_period=1,trace_limit=200000;
  uint64_t max_cycles=10000000;
  bool trace=true;
  static ScheduleOptions parse(const Json::Value &value);
  void validate() const;
};

// Regions 0..N-1 are input_layouts names in lexicographic order; region N is
// the transfer output. Views retain the input storage owner, issue no traffic,
// and need no output region. External transfer execution requires caller-owned
// output metadata; its storage may have null data/writable pointers.
// Only metadata and descriptor literals are accessed by this state machine.
// TensorMemoryPort is the default standalone endpoint, not system memory.
class Simulator {
public:
  Simulator(const Json::Value &node,const tensor_model::Values &values,
            ScheduleOptions options={},model_io::MemoryPort *port=nullptr,
            const tensor_model::Tensor *output=nullptr);
  ~Simulator();
  bool tick();
  bool done() const;
  tensor_model::Tensor output() const;
  Stats instruction_stats() const;
  Json::Value result() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlx::memory_model
