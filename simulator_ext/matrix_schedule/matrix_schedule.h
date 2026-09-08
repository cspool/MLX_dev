#pragma once
#include "matrix_program.h"
#include "../model_io/memory_port.h"
#include <memory>

namespace mlx::matrix_schedule {
struct Options {
  unsigned rows=4, columns=4, contexts=2;
  unsigned dma_latency=8, spm_latency=3, multiply_latency=4, add_latency=2, convert_latency=2;
  unsigned dma_request_period=1, dma_response_period=1, spm_period=1, writeback_period=1;
  unsigned compute_ii=1, trace_limit=100000;
  uint64_t max_cycles=10000000;
  bool overlap=true, trace=true, inject_stale_dma_epoch=false;
  bool cache_control=true; // Host optimization only; identical simulated cycles.
  static Options parse(const Json::Value &value);
  void validate() const;
};

// One matrix batch window. Actual data is consumed through bounded DMA/SPM
// requests, then compiled microinstructions; no BLAS/functional result replay.
class Simulator {
public:
  Simulator(const Json::Value &program, tensor_model::Tensor a, tensor_model::Tensor b,
            const tensor_model::Tensor *bias, bool transposed_b,
            uint64_t a_batch,uint64_t b_batch,uint64_t m,uint64_t n,uint64_t k,
            tensor_model::Tensor output,uint64_t output_batch,Options options={},model_io::MemoryPort *memory_port=nullptr);
  ~Simulator();
  bool tick();
  bool done() const;
  Json::Value result() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlx::matrix_schedule
