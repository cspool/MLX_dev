#pragma once
#include "tensor.h"

namespace mlx::control_model {
struct Stats {
  uint64_t calls=0,instructions=0,branches=0,read_bytes=0,write_bytes=0;
  unsigned fflags=0;
  Json::Value json()const;
};
tensor_model::Tensor execute(const Json::Value &node,const tensor_model::Values &values,Stats &stats);
} // namespace mlx::control_model
