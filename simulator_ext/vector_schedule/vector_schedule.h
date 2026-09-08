#pragma once
#include "vector_program.h"
#include "../model_io/memory_port.h"
#include "../shared_array/resources.h"
#include <memory>
#include <optional>

namespace mlx::vector_schedule {
using model_io::Request;
using model_io::Response;
using model_io::MemoryPort;
struct Options {
  unsigned rows=4,columns=4,contexts=2,dma_latency=8,spm_latency=3;
  unsigned multiply_latency=4,add_latency=2,convert_latency=2,exp_latency=8,div_latency=12,sqrt_latency=12;
  unsigned dma_request_period=1,dma_response_period=1,spm_period=1,writeback_period=1;
  unsigned vector_ii=1,trans_ii=1,trace_limit=200000;
  uint64_t max_cycles=10000000;
  bool overlap=true,trace=true,inject_stale_response=false;
  static Options parse(const Json::Value &value);
  void validate() const;
};

// External ports bind region 0/1 to operand backing storage and region 2 to
// output backing storage. Tensor values are NEVER read directly when an
// external port is provided. Literals are descriptor constants, not regions.
class Simulator {
public:
  Simulator(const Json::Value &node,const tensor_model::Values &values,
            tensor_model::Tensor output,Options options={},MemoryPort *port=nullptr,
            shared_array::Resources *array=nullptr,uint64_t source_id=UINT64_MAX);
  ~Simulator();
  bool tick();
  bool done() const;
  Json::Value result() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlx::vector_schedule
