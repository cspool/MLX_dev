#pragma once
#include "pair_wire.hh"
#include "../../simulator_ext/model_io/physical_mux.h"
#include <memory>

namespace mlx::physical_device {
// One externally clocked producer/consumer command. Both operands and both
// outputs use the caller's actual physical bus; descriptor data is not output.
class PairRuntime {
  struct Impl;
  std::unique_ptr<Impl> impl;
public:
  PairRuntime(const DecodedPair &pair,matrix_schedule::Options matrix,vector_schedule::Options vector,
              model_io::PhysicalMemoryPort &physical,model_io::RequestTokens &tokens,
              uint64_t clock_origin,uint64_t epoch);
  ~PairRuntime();
  void tick();
  bool done()const;
  Json::Value result()const;
};
}
