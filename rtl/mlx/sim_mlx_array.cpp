#include "Vmlx_array_4x4.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Vector = std::array<uint32_t, 16>;

struct ProgramEntry {
  uint32_t pe;
  uint32_t index;
  uint64_t word;
};

Vector parse_vector(std::string text) {
  Vector result{};
  if (!text.empty() && text.back() == '\r') text.pop_back();
  if (text.size() > 128) throw std::runtime_error("vector wider than 512 bits");
  text.insert(0, 128 - text.size(), '0');
  for (std::size_t word = 0; word < result.size(); ++word) {
    const std::size_t offset = text.size() - 8 * (word + 1);
    result[word] = static_cast<uint32_t>(std::stoul(text.substr(offset, 8), nullptr, 16));
  }
  return result;
}

std::vector<Vector> read_vectors(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open vector file " + path);
  std::vector<Vector> vectors;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty()) vectors.push_back(parse_vector(line));
  }
  return vectors;
}

std::vector<ProgramEntry> read_program(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open program " + path);
  std::vector<ProgramEntry> entries;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    std::istringstream parser(line);
    std::string pe;
    std::string index;
    std::string word;
    parser >> pe >> index >> word;
    entries.push_back({static_cast<uint32_t>(std::stoul(pe, nullptr, 16)),
                       static_cast<uint32_t>(std::stoul(index, nullptr, 16)),
                       std::stoull(word, nullptr, 16)});
  }
  return entries;
}

template <typename Wide>
void copy_to_wide(Wide &destination, const Vector &source) {
  for (std::size_t word = 0; word < source.size(); ++word) destination[word] = source[word];
}

template <typename Wide>
Vector copy_from_wide(const Wide &source) {
  Vector result{};
  for (std::size_t word = 0; word < result.size(); ++word) result[word] = source[word];
  return result;
}

std::string vector_string(const Vector &value) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (auto word = value.rbegin(); word != value.rend(); ++word) stream << std::setw(8) << *word;
  return stream.str();
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  if (argc != 8 && argc != 9) {
    std::cerr << "usage: sim program input golden workload input_vectors output_vectors "
                 "output_base [vcd]\n";
    return 2;
  }
  const std::vector<ProgramEntry> program = read_program(argv[1]);
  const std::vector<Vector> inputs = read_vectors(argv[2]);
  const std::vector<Vector> golden = read_vectors(argv[3]);
  const std::string workload = argv[4];
  const unsigned input_vectors = std::stoul(argv[5]);
  const unsigned output_vectors = std::stoul(argv[6]);
  const unsigned output_base = std::stoul(argv[7]);
  if (inputs.size() != input_vectors || golden.size() != output_vectors) {
    std::cerr << "vector count mismatch\n";
    return 2;
  }

  std::array<Vector, 128> spm{};
  for (std::size_t index = 0; index < inputs.size(); ++index) spm[index] = inputs[index];
  std::array<unsigned, 16> instruction_counts{};
  for (const auto &entry : program) {
    if (entry.pe >= 16 || entry.index >= 32) {
      std::cerr << "program coordinate out of range\n";
      return 2;
    }
    instruction_counts[entry.pe] =
        std::max(instruction_counts[entry.pe], entry.index + 1);
  }

  Vmlx_array_4x4 dut;
  VerilatedVcdC trace;
  VerilatedVcdC *trace_ptr = nullptr;
  if (argc == 9) {
    Verilated::traceEverOn(true);
    dut.trace(&trace, 99);
    trace.open(argv[8]);
    trace_ptr = &trace;
  }
  uint64_t sim_time = 0;
  bool response_pending = false;
  Vector response_data{};
  auto half = [&](uint8_t clock) {
    dut.clk = clock;
    dut.eval();
    if (trace_ptr) trace_ptr->dump(sim_time);
    ++sim_time;
  };
  auto tick = [&]() {
    dut.spm_rsp_valid_i = response_pending;
    copy_to_wide(dut.spm_rsp_rdata_i, response_data);
    half(0);
    const bool request = dut.spm_req_valid_o && dut.spm_req_ready_i;
    const bool write = dut.spm_req_write_o;
    const unsigned address = dut.spm_req_addr_o;
    const Vector write_data = copy_from_wide(dut.spm_req_wdata_o);
    half(1);
    response_pending = false;
    if (request) {
      if (address >= spm.size()) throw std::runtime_error("SPM address outside range");
      if (write) {
        spm[address] = write_data;
      } else {
        response_data = spm[address];
        response_pending = true;
      }
    }
  };

  dut.rst_n = 0;
  dut.cfg_valid_i = 0;
  dut.cfg_pe_i = 0;
  dut.cfg_index_i = 0;
  dut.cfg_word_i = 0;
  dut.launch_i = 0;
  dut.input_vectors_i = input_vectors;
  dut.spm_req_ready_i = 1;
  dut.spm_rsp_valid_i = 0;
  copy_to_wide(dut.spm_rsp_rdata_i, response_data);
  for (int cycle = 0; cycle < 4; ++cycle) tick();
  dut.rst_n = 1;
  tick();

  for (const auto &entry : program) {
    dut.cfg_valid_i = 1;
    dut.cfg_pe_i = entry.pe;
    dut.cfg_index_i = entry.index;
    dut.cfg_word_i = entry.word;
    tick();
  }
  for (unsigned pe = 0; pe < instruction_counts.size(); ++pe) {
    dut.cfg_valid_i = 1;
    dut.cfg_pe_i = 16;
    dut.cfg_index_i = pe;
    dut.cfg_word_i = instruction_counts[pe];
    tick();
  }
  dut.cfg_valid_i = 0;
  dut.launch_i = 1;
  tick();
  dut.launch_i = 0;

  unsigned timeout = 0;
  while (!dut.done_o && timeout < 20000) {
    tick();
    ++timeout;
  }
  if (!dut.done_o) {
    std::cerr << "array timeout workload=" << workload << '\n';
    return 1;
  }
  unsigned mismatches = 0;
  for (unsigned index = 0; index < output_vectors; ++index) {
    if (spm[output_base + index] != golden[index]) {
      std::cerr << "MLX_RTL_MISMATCH workload=" << workload << " vector=" << index
                << " got=" << vector_string(spm[output_base + index])
                << " expected=" << vector_string(golden[index]) << '\n';
      ++mismatches;
    }
  }
  if (trace_ptr) {
    trace.flush();
    trace.close();
  }
  if (mismatches) return 1;
  std::cout << "MLX_ARRAY_PASS workload=" << workload << " cycles=" << dut.stat_cycles_o
            << " instructions=" << dut.stat_instructions_o << " load=" << dut.stat_load_o
            << " store=" << dut.stat_store_o << " compute=" << dut.stat_compute_o
            << " xfer=" << dut.stat_xfer_o << " stalls=" << dut.stat_stall_o
            << " hops=" << dut.stat_hops_o << " conflicts=" << dut.stat_conflicts_o << '\n';
  dut.final();
  return 0;
}
