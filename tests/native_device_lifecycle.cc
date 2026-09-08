#include "device.h"
#include <iostream>
#include <optional>
#include <stdexcept>

using mlx::system::Inputs;
void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
struct Bus {
  mlx::system::Device device;
  std::map<uint64_t, uint64_t> memory;
  std::optional<std::pair<unsigned, uint64_t>> pending;
  std::optional<uint64_t> response;
  unsigned cycles = 0;
  bool step(Inputs in = {}) {
    check(cycles < 10000, "lifecycle watchdog");
    in.response_ready = true;
    in.memory_ready = cycles % 3 == 0;
    if (pending && pending->first == cycles) {
      in.memory_response_valid = true;
      in.memory_response_data = pending->second;
      pending.reset();
    }
    const auto out = device.eval(in);
    if (out.response_valid) response = out.response_data;
    if (out.memory_valid && in.memory_ready) {
      check(!pending, "multiple outstanding memory requests");
      if (out.memory_write) memory[out.memory_address] = out.memory_data;
      pending = {cycles + 4, out.memory_write ? 0 : memory.at(out.memory_address)};
    }
    device.tick(in);
    ++cycles;
    return in.command_valid && out.command_ready;
  }
  uint64_t command(unsigned funct, uint64_t a = 0, uint64_t b = 0, bool xd = false) {
    Inputs in;
    in.command_valid = true; in.funct = funct; in.rs1 = a; in.rs2 = b; in.command_xd = xd;
    response.reset();
    while (!step(in)) {}
    if (xd) while (!response) step();
    return response.value_or(0);
  }
  void configure() {
    const std::map<unsigned, uint64_t> words = {
      {0, 0x4d4c580200000001ULL}, {1, 1},
      {2, 1ULL | 1ULL<<8 | 4ULL<<16 | 2ULL<<24 | 4ULL<<32 | 32ULL<<40 | 4ULL<<48},
      {0x100, 7}, {0x101, 3ULL<<8 | 2ULL<<16}, {0x102, 1}, {0x103, 0},
      {0x1000, 0}, {0x1001, 9ULL<<60 | 2ULL<<54 | 1ULL<<50},
      {0x1002, 1ULL<<60 | 1ULL<<54 | 1ULL<<46 | 1ULL<<20},
      {0x1e00, 1}, {0x1e01, 0}, {0x1e02, 2}, {0x1e03, 0},
    };
    for (auto [address, word] : words) command(0, word, address);
  }
  void run(uint16_t input, uint16_t output) {
    const uint64_t spread = 0x0001000100010001ULL;
    memory[0x1000] = uint64_t(input) * spread;
    memory[0x2000] = 0xdeadbeefdeadbeefULL;
    command(1, 0x1000, 0x2000);
    Inputs blocked;
    blocked.funct = 0; blocked.command_valid = true; blocked.rs2 = 0;
    check(!step(blocked), "configuration was accepted while busy");
    const auto wait = command(2, 0, 0, true);
    check((wait & 20) == 4, "valid launch did not complete successfully");
    check(memory[0x2000] == uint64_t(output) * spread, "relaunch used stale input/output");
    check(device.status(13) == 16 && device.status(5) == 3, "per-launch counters accumulated");
    check(!pending, "completed device has outstanding DMA");
  }
};

int main() {
  try {
    Bus bus;
    bus.configure();
    bus.run(0x4000, 0x4400); // 2 * 2 = 4
    bus.run(0x4200, 0x4880); // 3 * 3 = 9, without reconfiguration
    bus.command(1, 0x1001, 0x2000);
    check(bus.command(2, 0, 0, true) & 16, "misaligned DMA pointer was accepted");
    check(bus.device.status(1) == 0 && bus.device.status(13) == 0 && bus.device.status(5) == 0,
          "bad-pointer launch exposed previous counters");
    bus.run(0x4000, 0x4400); // recover without changing the valid image
    bus.command(0, 0, 0); // invalidate the image after completed launches
    bus.command(1, 0x1000, 0x2000);
    check(bus.command(2, 0, 0, true) & 16, "invalid image did not return an error");
    check(bus.device.status(1) == 0 && bus.device.status(13) == 0 && bus.device.status(5) == 0,
          "failed launch exposed counters from the previous execution");
    bus.configure();
    bus.run(0x4000, 0x4400);
    bus.command(1, 0x1000, 0x2000);
    while (!bus.pending) bus.step();
    Inputs stale;
    stale.memory_response_valid = true; stale.memory_response_tag = 1;
    bus.step(stale);
    check(bus.device.status(15) && (bus.device.status(0) & 2),
          "wrong-tag response did not retain the outstanding transaction");
    Inputs reconfigure;
    reconfigure.command_valid = true; reconfigure.funct = 0; reconfigure.rs2 = 0;
    check(!bus.step(reconfigure), "reconfiguration discarded an outstanding response");
    while (bus.pending) bus.step();
    check(!(bus.device.status(0) & 2), "error did not drain the original transaction");
    bus.configure();
    bus.run(0x4000, 0x4400);
    bus.device.reset();
    check(bus.device.status(0) == 33 && bus.device.status(1) == 0 && bus.device.status(2) == 0,
          "reset retained prior launch state");
    bus.configure();
    bus.run(0x4200, 0x4880);
    std::cout << "MLX_DEVICE_LIFECYCLE_PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "MLX_DEVICE_LIFECYCLE_FAIL: " << error.what() << '\n';
    return 1;
  }
}
