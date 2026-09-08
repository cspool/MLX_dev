#pragma once
#include "tensor.h"

namespace mlx::memory_model {
bool supports(const std::string &kind);
struct Stats {
  uint64_t calls=0,views=0,allocations=0,instructions=0,read_bytes=0,write_bytes=0,index_reads=0,predicate_reads=0;
  uint64_t opcode_counts[6]{};
  Json::Value json() const;
};
tensor_model::Tensor execute(const Json::Value &node,const tensor_model::Values &values,Stats &stats);
} // namespace mlx::memory_model
