#pragma once

#include "mlx_tagged_simulator.h"

#include <map>
#include <memory>
#include <string>

namespace mlx::system {

// Program.image() words plus four system-only I/O slot masks. Packed host
// buffers contain the selected SPM vectors in ascending slot order.
constexpr uint32_t INPUT_MASK_LO = 0x1e00;
constexpr uint32_t INPUT_MASK_HI = 0x1e01;
constexpr uint32_t OUTPUT_MASK_LO = 0x1e02;
constexpr uint32_t OUTPUT_MASK_HI = 0x1e03;
constexpr uint32_t ABI_MAGIC = 0x4d4c5802;

struct Inputs {
  bool command_valid = false, command_xd = false, response_ready = false;
  uint8_t funct = 0, rd = 0, privilege = 3;
  uint64_t rs1 = 0, rs2 = 0;
  bool memory_ready = false, memory_response_valid = false;
  uint64_t memory_response_data = 0;
  uint32_t memory_response_tag = 0;
};
struct Outputs {
  bool command_ready = false, response_valid = false, busy = false;
  uint8_t response_rd = 0;
  uint64_t response_data = 0;
  bool memory_valid = false, memory_write = false;
  uint64_t memory_address = 0, memory_data = 0;
  uint32_t memory_tag = 0;
  uint8_t memory_size = 3, memory_mask = 255, privilege = 3;
};

tagged::Program decode_image(const std::map<uint32_t, uint64_t> &image,
                             const std::array<uint64_t, 4> &masks);

class Device {
public:
  explicit Device(unsigned address_bits = 40, tagged::Timing timing = {});
  Outputs eval(const Inputs &inputs) const;
  void tick(const Inputs &inputs);
  void reset();
  uint64_t status(unsigned index) const;
  Json::Value report() const;
  bool complete() const;
private:
  enum class Phase { Idle, Read, Run, Write, Complete, Error };
  Phase phase = Phase::Idle;
  unsigned address_bits;
  tagged::Timing timing;
  std::map<uint32_t, uint64_t> image;
  std::array<uint64_t, 4> masks{};
  tagged::Program program;
  std::unique_ptr<tagged::Simulator> model;
  Json::Value kernel_result;
  std::vector<unsigned> input_slots, output_slots;
  std::vector<tagged::Vector> output_data;
  uint64_t device_cycles = 0, system_cycles = 0, dma_cycles = 0, kernel_cycles = 0;
  uint64_t config_commands = 0, memory_requests = 0, memory_responses = 0, dma_bytes = 0;
  uint64_t input_base = 0, output_base = 0;
  unsigned vector_index = 0, beat_index = 0;
  uint8_t privilege = 3, error_code = 0;
  std::string error_message;
  bool waiting_memory = false, response_valid = false;
  uint8_t response_rd = 0;
  uint64_t response_data = 0;
  Json::Value events{Json::arrayValue};
  void configure(uint64_t address, uint64_t word);
  void launch(uint64_t input, uint64_t output, uint8_t dprv);
  void fail(uint8_t code, const std::string &message);
  void event(const char *kind, uint64_t address = 0, uint64_t data = 0);
  void begin_kernel();
};
} // namespace mlx::system
