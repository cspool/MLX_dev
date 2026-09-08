#pragma once
#include "tensor.h"

namespace mlx::vector_model {
bool supports(const std::string &kind);
struct Stats {
  uint64_t calls=0,instructions=0,trans_lanes=0,arithmetic_lanes=0,read_bytes=0,write_bytes=0;
  uint64_t opcode_counts[28]{};
  unsigned max_rom=0,max_stack_level=0;
  Json::Value json() const;
};
tensor_model::Tensor execute(const Json::Value &node,const tensor_model::Values &values,Stats &stats);
} // namespace mlx::vector_model
