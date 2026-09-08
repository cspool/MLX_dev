#pragma once
#include "tensor.h"

namespace mlx::memory_model {
inline bool extended_kind(const std::string &kind){return kind=="advanced_index"||kind=="new_ones"||kind=="squeeze";}
inline const char *profile(const std::string &kind){return extended_kind(kind)?"mlx-memory-plan-v2":"mlx-memory-plan-v1";}
bool supports(const std::string &kind);
struct Stats {
  uint64_t calls=0,views=0,allocations=0,instructions=0,read_bytes=0,write_bytes=0,index_reads=0,predicate_reads=0;
  uint64_t opcode_counts[6]{};
  bool v2=false;
  Json::Value json() const;
};
tensor_model::Tensor execute(const Json::Value &node,const tensor_model::Values &values,Stats &stats);
} // namespace mlx::memory_model
