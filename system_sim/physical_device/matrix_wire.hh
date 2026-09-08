#pragma once
#include "matrix_wire.h"
#include "matrix_schedule.h"
#include "../../simulator_ext/model_io/address_space_port.h"

namespace mlx::physical_device {
struct DecodedMatrix {
  Json::Value program;
  tensor_model::Tensor a,b,bias,output;
  bool transpose_b=false,has_bias=false;
  uint64_t m=0,n=0,k=0,a_batch=0,b_batch=0,output_batch=0;
  std::vector<model_io::Region> regions;
};
DecodedMatrix decode_matrix(const mlx_matrix_wire &wire);
tensor_model::Tensor decode_float_tensor(const mlx_host_tensor &wire,bool output);
tensor_model::Tensor decode_tensor(const mlx_host_tensor &wire,bool output);
} // namespace mlx::physical_device
