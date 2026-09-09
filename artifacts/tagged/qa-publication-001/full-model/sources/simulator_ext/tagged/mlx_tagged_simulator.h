#pragma once

#include <array>
#include <cstdint>
#include <jsoncpp/json/json.h>
#include <memory>
#include <string>
#include <vector>

namespace mlx::tagged {

using Vector = std::array<uint16_t, 32>;
enum class Pipeline : unsigned { Load, Store, Compute, Transfer };
struct Hardware {
  unsigned rows = 4, columns = 4, lanes = 32, contexts = 4;
  unsigned registers = 16, instruction_words = 32, spm_vectors = 128;
  unsigned pes() const { return rows * columns; }
};
struct Instruction {
  unsigned op = 0, dst = 0, arity = 0, spm = 0;
  std::array<unsigned, 3> src{};
  int stride = 0;
  unsigned target = 0; // descriptor index; logical block IDs remain separate
  Pipeline pipeline() const;
};
struct Block {
  unsigned id = 0, layer = 0, pe = 0, wave = 0, registers = 0, trips = 1;
  std::vector<Instruction> instructions;
};
struct Program {
  std::string name;
  Hardware hardware;
  std::vector<Block> blocks;
  std::array<Vector, 128> inputs{};
  std::array<bool, 128> input_valid{};
  std::vector<unsigned> outputs;
  unsigned waves = 0;
  static Program parse(const Json::Value &json);
  void validate() const;
};
struct Timing {
  // XFER is timed by the bounded packet network, not a fixed delay.
  std::array<unsigned, 10> latency{3, 1, 4, 3, 1, 8, 12, 1, 0, 3};
  unsigned compute_ii = 1, memory_period = 1, link_period = 1, receive_period = 1;
  uint64_t max_cycles = 100000;
  bool overlap = true, trace = true;
  void validate() const;
};

const char *opcode_name(unsigned op);
const char *pipeline_name(Pipeline pipeline);
float half_to_float(uint16_t value);
uint16_t float_to_half(float value);
Vector arithmetic(unsigned op, const std::array<Vector, 3> &operands, unsigned lanes);

// Owns its program and state. tick() advances exactly one model cycle; polling
// a completed instance is idempotent. Results never execute hidden extra work.
class Simulator {
public:
  explicit Simulator(Program program, Timing timing = {});
  ~Simulator();
  Simulator(Simulator &&) noexcept;
  Simulator &operator=(Simulator &&) noexcept;
  Simulator(const Simulator &) = delete;
  Simulator &operator=(const Simulator &) = delete;
  bool tick();
  bool done() const;
  uint64_t cycles() const;
  Json::Value snapshot() const;
  Json::Value result() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
Json::Value simulate(const Program &program, const Timing &timing);

} // namespace mlx::tagged
