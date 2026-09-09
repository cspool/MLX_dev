#include "mlx_tagged_simulator.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  using namespace mlx::tagged;
  try {
    for (unsigned bits = 0; bits <= 65535; ++bits) {
      // NaN payload preservation is not part of float conversion; finite
      // values, signed zeros and infinities must roundtrip bit-for-bit.
      if ((bits & 0x7c00) == 0x7c00 && (bits & 0x3ff)) continue;
      if (float_to_half(half_to_float(uint16_t(bits))) != bits)
        throw std::runtime_error("half conversion roundtrip failed");
    }
    if (float_to_half(std::ldexp(1.0f, -25)) != 0 ||
        float_to_half(std::ldexp(3.0f, -25)) != 2 ||
        float_to_half(65520.0f) != 0x7c00)
      throw std::runtime_error("round-to-nearest-even boundary failed");
    std::array<Vector, 3> operands{};
    operands[0].fill(0x3c01); operands[1].fill(0x3c01); operands[2].fill(0xbc02);
    if (arithmetic(2, operands, 32)[0] != 0)
      throw std::runtime_error("FMA incorrectly contracted instead of step rounding");
    operands[0].fill(0x8000); operands[1].fill(0);
    if (arithmetic(4, operands, 32)[0] != 0x8000)
      throw std::runtime_error("MAX signed-zero tie contract failed");
    std::cout << "MLX_NUMERICS_PASS finite_half_roundtrip_and_rounding_boundaries\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
  return 0;
}
