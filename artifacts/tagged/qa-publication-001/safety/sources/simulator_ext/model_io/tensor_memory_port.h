#pragma once
#include "memory_port.h"
#include "tensor.h"
#include <vector>

namespace mlx::model_io {
// Standalone behavioral endpoint only. System execution supplies a physical
// memory adapter instead; no tensor value is read by the compute engine.
class TensorMemoryPort final:public MemoryPort {
  std::vector<tensor_model::Tensor> regions;
  unsigned latency;
  uint64_t cycle=0,due=0;
  std::optional<Request> pending;
  std::optional<Response> answer;
public:
  TensorMemoryPort(std::vector<tensor_model::Tensor> regions,unsigned latency);
  void advance(uint64_t cycle) override;
  bool request_ready() const override;
  void submit(const Request &request) override;
  std::optional<Response> response() const override;
  void consume_response() override;
};
} // namespace mlx::model_io
