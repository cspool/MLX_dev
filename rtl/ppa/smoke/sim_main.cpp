#include "Vppa_smoke.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <iostream>

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Vppa_smoke dut;
  auto tick = [&dut]() {
    dut.clk = 0;
    dut.eval();
    dut.clk = 1;
    dut.eval();
  };
  dut.rst_n = 0;
  dut.valid_i = 0;
  dut.lhs_i = 0;
  dut.rhs_i = 0;
  dut.addend_i = 0;
  tick();
  tick();
  tick();
  dut.rst_n = 1;

  const std::array<uint16_t, 3> lhs = {6, 12, 65535};
  const std::array<uint16_t, 3> rhs = {7, 100, 2};
  const std::array<uint32_t, 3> addend = {1, 35, 131055};
  const std::array<uint32_t, 3> expected = {43, 1235, 262125};
  int seen = 0;
  for (int index = 0; index < 3; ++index) {
    dut.valid_i = 1;
    dut.lhs_i = lhs[index];
    dut.rhs_i = rhs[index];
    dut.addend_i = addend[index];
    tick();
    if (dut.valid_o) {
      if (seen >= 3 || dut.result_o != expected[seen]) {
        std::cerr << "result mismatch at " << seen << '\n';
        return 1;
      }
      ++seen;
    }
  }
  dut.valid_i = 0;
  tick();
  if (dut.valid_o) {
    if (seen >= 3 || dut.result_o != expected[seen]) {
      std::cerr << "result mismatch at " << seen << '\n';
      return 1;
    }
    ++seen;
  }
  tick();
  const uint32_t expected_checksum = expected[0] ^ expected[1] ^ expected[2];
  if (seen != 3 || dut.checksum_o != expected_checksum) {
    std::cerr << "final state mismatch\n";
    return 1;
  }
  std::cout << "PPA_SMOKE_PASS checksum=" << dut.checksum_o << '\n';
  dut.final();
  return 0;
}
