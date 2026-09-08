#pragma once
#include "vector_wire.h"
#include "matrix_wire.hh"
#include "vector_schedule.h"

namespace mlx::physical_device {
struct DecodedVector {
  Json::Value node;
  tensor_model::Values values;
  tensor_model::Tensor output;
  std::vector<model_io::Region> regions;
};
DecodedVector decode_vector(const mlx_vector_wire &wire);
} // namespace mlx::physical_device
