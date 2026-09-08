#pragma once
#include <array>
#include <cstdint>
#include <jsoncpp/json/json.h>

namespace mlx::control_model {
struct RV64 {
  std::array<uint64_t,32> x{},f{};
  uint64_t retired=0,branches_taken=0;
  unsigned fflags=0;
  void set_float(unsigned reg,float value);
  float get_float(unsigned reg) const;
  void run(const Json::Value &words);
};
} // namespace mlx::control_model
