#pragma once
#include "memory_wire.h"
#include "matrix_wire.hh"
#include "memory_schedule.h"

namespace mlx::physical_device {
struct DecodedMemory {
  Json::Value node;
  tensor_model::Values values;
  tensor_model::Tensor output;
  std::vector<model_io::Region> regions;
  bool view=false;
};
DecodedMemory decode_memory(const mlx_memory_wire &wire);
} // namespace mlx::physical_device
