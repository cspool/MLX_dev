#include "Vmlx_pe_top.h"
#include "verilated.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template <std::size_t N>
void clear_wide(WData (&value)[N]) {
  for (std::size_t index = 0; index < N; ++index) value[index] = 0;
}

template <std::size_t N>
void set_lane(WData (&value)[N], int lane, uint16_t bits) {
  const int word = lane / 2;
  const int shift = (lane % 2) * 16;
  value[word] &= ~(0xffffU << shift);
  value[word] |= static_cast<uint32_t>(bits) << shift;
}

std::vector<uint64_t> read_program(const std::string &path) {
  std::ifstream stream(path);
  std::vector<uint64_t> words;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    std::stringstream parser;
    parser << std::hex << line;
    uint64_t word = 0;
    parser >> word;
    words.push_back(word);
  }
  return words;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  if (argc != 5) {
    std::cerr << "usage: sim program.hex workload simd full_features\n";
    return 2;
  }
  const std::vector<uint64_t> program = read_program(argv[1]);
  const std::string workload = argv[2];
  const int simd_width = std::stoi(argv[3]);
  const bool full_features = std::stoi(argv[4]) != 0;
  Vmlx_pe_top dut;
  auto tick = [&dut]() {
    dut.clk = 0;
    dut.eval();
    dut.clk = 1;
    dut.eval();
  };

  dut.rst_n = 0;
  dut.cfg_valid_i = 0;
  dut.cfg_addr_i = 0;
  dut.cfg_word_i = 0;
  dut.fetch_addr_i = 0;
  dut.tag_configure_i = 0;
  dut.tag_configure_id_i = 0;
  dut.tag_trip_count_i = 0;
  dut.tag_frontier_i = 0;
  dut.tag_ready_i = 0;
  dut.tag_issue_i = 0;
  dut.tag_issue_id_i = 0;
  dut.tag_complete_i = 0;
  dut.tag_complete_id_i = 0;
  dut.tag_query_id_i = 0;
  dut.pipeline_class_i = 0;
  dut.pipeline_ready_i = 0xf;
  dut.rf_read_addr_a_i = 0;
  dut.rf_read_addr_b_i = 0;
  dut.rf_read_addr_c_i = 0;
  dut.rf_write_enable_i = 0;
  dut.rf_write_addr_i = 0;
  clear_wide(dut.rf_write_data_i);
  dut.network_valid_i = 0;
  dut.network_dx_i = 0;
  dut.network_dy_i = 0;
  dut.network_destination_register_i = 0;
  dut.network_tag_i = 0;
  dut.network_payload_i = 0;
  dut.fu_valid_i = 0;
  dut.fu_op_i = 0;
  clear_wide(dut.fu_vector_a_i);
  clear_wide(dut.fu_vector_b_i);
  clear_wide(dut.fu_vector_c_i);
  for (int lane = 0; lane < simd_width; ++lane) {
    set_lane(dut.fu_vector_a_i, lane, lane == 1 ? 0x4000 : 0x3c00);
    set_lane(dut.fu_vector_b_i, lane, 0x4000);
    set_lane(dut.fu_vector_c_i, lane, 0x4200);
  }
  tick();
  tick();
  tick();
  dut.rst_n = 1;

  for (std::size_t index = 0; index < program.size(); ++index) {
    dut.cfg_valid_i = 1;
    dut.cfg_addr_i = index;
    dut.cfg_word_i = program[index];
    tick();
  }
  dut.cfg_valid_i = 0;

  uint32_t checksum = 0;
  int operation_count = 0;
  for (const uint64_t word : program) {
    const uint8_t opcode = (word >> 60) & 0xf;
    ++operation_count;
    if ((opcode >= 2 && opcode <= 7) || opcode == 9) {
      dut.fu_op_i = opcode;
      dut.fu_valid_i = 1;
      tick();
      dut.fu_valid_i = 0;
      const bool removed = !full_features && (opcode == 6 || opcode == 7);
      if (removed) {
        if (!dut.fu_illegal_o) {
          std::cerr << "removed operation accepted\n";
          return 1;
        }
      } else {
        uint16_t expected = 0x3c00;
        if (opcode == 2) expected = 0x4500;
        if (opcode == 3) expected = 0x4200;
        if (opcode == 4) expected = 0x4000;
        if (opcode == 5) expected = 0x4170;
        if (opcode == 6) expected = 0x3800;
        if (opcode == 7) expected = 0x4000;
        if (opcode == 9) expected = 0x4000;
        const uint16_t observed = dut.fu_vector_result_o[0] & 0xffff;
        if (!dut.fu_valid_o || dut.fu_illegal_o || observed != expected) {
          std::cerr << "FU mismatch opcode=" << static_cast<int>(opcode)
                    << " observed=" << std::hex << observed
                    << " expected=" << expected << '\n';
          return 1;
        }
        checksum ^= observed;
      }
    } else if (opcode == 8) {
      dut.network_valid_i = 1;
      dut.network_dx_i = (word >> 33) & 0x1f;
      dut.network_dy_i = (word >> 28) & 0x1f;
      tick();
      dut.network_valid_i = 0;
      if (!dut.network_valid_o || dut.network_consumed_hops_o == 0) {
        std::cerr << "network xfer mismatch\n";
        return 1;
      }
    }
  }
  if (!full_features) {
    dut.fu_op_i = 6;
    dut.fu_valid_i = 1;
    tick();
    dut.fu_valid_i = 0;
    if (!dut.fu_illegal_o) {
      std::cerr << "reduced divide was not rejected\n";
      return 1;
    }
  }
  std::cout << "MLX_RTL_PASS workload=" << workload << " simd=" << simd_width
            << " operations=" << operation_count << " checksum=" << std::hex
            << checksum << '\n';
  dut.final();
  return 0;
}
