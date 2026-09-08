#include "mlx_tagged_simulator.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mlx::tagged {
float half_to_float(uint16_t value) {
  const unsigned exponent = (value >> 10) & 31, fraction = value & 1023;
  const float sign = value & 0x8000 ? -1.0f : 1.0f;
  if (exponent == 31) return sign * (fraction ? std::numeric_limits<float>::quiet_NaN()
                                             : std::numeric_limits<float>::infinity());
  return std::copysign(exponent ? std::ldexp(float(1024 + fraction), int(exponent) - 25)
                               : std::ldexp(float(fraction), -24), sign);
}
uint16_t float_to_half(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  const unsigned sign = (bits >> 16) & 0x8000, exponent = (bits >> 23) & 255;
  unsigned fraction = bits & 0x7fffff;
  if (exponent == 255) return sign | (fraction ? (0x7e00 | (fraction >> 13)) : 0x7c00);
  const int adjusted = int(exponent) - 112;
  if (adjusted >= 31) return sign | 0x7c00;
  if (adjusted < -10) return sign;
  unsigned result, shift;
  if (adjusted <= 0) { fraction |= 0x800000; shift = unsigned(14 - adjusted); result = fraction >> shift; }
  else { shift = 13; result = unsigned(adjusted) << 10 | fraction >> 13; }
  const unsigned remainder = fraction & ((1u << shift) - 1), midpoint = 1u << (shift - 1);
  if (remainder > midpoint || (remainder == midpoint && (result & 1))) ++result;
  return uint16_t(sign | result);
}
Vector arithmetic(unsigned op, const std::array<Vector, 3> &operands, unsigned lanes) {
  Vector output{};
  for (unsigned lane = 0; lane < lanes; ++lane) {
    const float a = half_to_float(operands[0][lane]);
    const float b = half_to_float(operands[1][lane]);
    const float c = half_to_float(operands[2][lane]);
    if (op == 7) { output[lane] = operands[0][lane ^ 1]; continue; }
    if ((op == 5 || op == 6) && lane >= lanes / 4) { output[lane] = operands[0][lane]; continue; }
    float value;
    switch (op) {
      case 2: value = half_to_float(float_to_half(a * b)) + c; break;
      case 3: value = a + b; break;
      case 4:
        if (std::isnan(a)) { output[lane] = operands[0][lane]; continue; }
        if (std::isnan(b)) { output[lane] = operands[1][lane]; continue; }
        value = a >= b ? a : b; break;
      case 5: value = std::exp(a); break;
      case 6: value = a / b; break;
      case 9: value = a * b; break;
      default: throw std::invalid_argument("unsupported arithmetic opcode");
    }
    output[lane] = float_to_half(value);
  }
  return output;
}
} // namespace mlx::tagged
