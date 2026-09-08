#pragma once
#include "pair_wire.h"
#include "matrix_wire.hh"
#include "vector_wire.hh"
#include "../../simulator_ext/model_events/completion_window.h"
#include <optional>

namespace mlx::physical_device {
struct DecodedPair {
  std::optional<DecodedMatrix> matrix;
  std::optional<DecodedVector> vector;
  DecodedVector consumer;
  model_events::Mapping mapping;
  uint64_t producer_source=0,consumer_source=0,batches=1;
  unsigned event_slots=32;
  std::vector<model_io::Region> regions;
  const tensor_model::Tensor &producer_output()const{return matrix?matrix->output:vector->output;}
  const std::vector<model_io::Region> &producer_regions()const{return matrix?matrix->regions:vector->regions;}
};
DecodedPair decode_pair(const mlx_pair_wire &wire);
}
