#include "Vmlx_tagged_pe_control.h"
#ifdef MLX_RTL_FU
#include "Vmlx_tagged_fu.h"
#endif
#include "mlx_tagged_simulator.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#ifndef MLX_CONTROL_LANES
#define MLX_CONTROL_LANES 32
#endif
#ifndef MLX_CONTROL_BUILD_ID
#define MLX_CONTROL_BUILD_ID "manual-unbound"
#endif

double sc_time_stamp() { return 0; }
using mlx::tagged::Program;
using mlx::tagged::Timing;
using mlx::tagged::Vector;
void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}
unsigned field(uint64_t packed, unsigned slot, unsigned bits) {
  return (packed >> (slot * bits)) & ((1ULL << bits) - 1);
}
template <class Port> Vector read_vector(const Port &port) {
  Vector value{};
  for (unsigned lane = 0; lane < MLX_CONTROL_LANES; ++lane) {
    if constexpr (std::is_integral_v<Port>) value[lane] = port >> (lane * 16);
    else value[lane] = port[lane / 2] >> ((lane % 2) * 16);
  }
  return value;
}
template <class Port> void write_vector(Port &port, const Vector &value) {
  if constexpr (std::is_integral_v<Port>) {
    port = 0;
    for (unsigned lane = 0; lane < MLX_CONTROL_LANES; ++lane) port |= uint64_t(value[lane]) << (lane * 16);
  } else {
    for (unsigned lane = 0; lane < MLX_CONTROL_LANES; lane += 2)
      port[lane / 2] = uint32_t(value[lane]) | uint32_t(value[lane + 1]) << 16;
  }
}
struct Control {
  Vmlx_tagged_pe_control dut;
#ifdef MLX_RTL_FU
  Vmlx_tagged_fu fu;
#endif
  uint64_t edges = 0;
  Control() {
    dut.clk = 0; dut.rst_n = 0;
#ifdef MLX_RTL_FU
    fu.clk = 0; fu.rst_n = 0; fu.operation_i = 3; fu.request_tag_i = 0; fu.compute_ii_i = 1;
    write_vector(fu.operand_a_i, Vector{}); write_vector(fu.operand_b_i, Vector{}); write_vector(fu.operand_c_i, Vector{});
#endif
    idle();
    dut.context_slot_i = dut.context_block_i = dut.context_layer_i = 0;
    dut.context_epoch_i = dut.context_start_i = dut.context_length_i = 0;
    dut.context_trips_i = dut.context_rf_base_i = dut.context_registers_i = 0;
    dut.program_address_i = dut.program_word_i = 0;
    dut.writeback_slot_i = dut.writeback_block_i = dut.writeback_epoch_i = 0;
    dut.writeback_iteration_i = dut.writeback_register_i = 0;
    write_vector(dut.writeback_data_i, Vector{});
    dut.overlap_i = 1;
    edge(true); dut.rst_n = 1; dut.eval();
#ifdef MLX_RTL_FU
    fu.rst_n = 1; fu.eval();
#endif
  }
  void idle() {
    dut.clear_config_i = dut.program_write_i = dut.context_write_i = dut.admit_i = 0;
    dut.complete_i = dut.complete_block_i = dut.complete_epoch_i = 0;
    dut.complete_iteration_i = dut.complete_pc_i = 0;
    dut.writeback_valid_i = dut.writeback_input_i = 0;
    dut.resource_ready_i = 0; dut.issue_ready_i = 1;
#ifdef MLX_RTL_FU
    fu.request_valid_i = 0; fu.response_ready_i = 0;
#endif
  }
  void edge(bool execution = false) {
#ifdef MLX_RTL_FU
    // Fixture configuration is outside the reference's execution clock.
    // The FU clock is held during setup, including inter-wave setup; no
    // physical system-clock or configuration-latency claim is made here.
    fu.clk = 0; fu.eval();
#else
    (void)execution;
#endif
    dut.clk = 0; dut.eval(); dut.clk = 1; dut.eval(); dut.clk = 0; dut.eval(); ++edges;
#ifdef MLX_RTL_FU
    if (execution) { fu.clk = 1; fu.eval(); fu.clk = 0; fu.eval(); }
#endif
  }
  void code(unsigned address, uint64_t word) {
    idle(); dut.program_write_i = 1; dut.program_address_i = address; dut.program_word_i = word;
    edge(); idle();
  }
  void descriptor(unsigned slot, unsigned id, unsigned layer, unsigned epoch,
                  unsigned start, unsigned length, unsigned trips, unsigned base, unsigned registers) {
    idle(); dut.context_write_i = 1; dut.context_slot_i = slot; dut.context_block_i = id;
    dut.context_layer_i = layer; dut.context_epoch_i = epoch; dut.context_start_i = start;
    dut.context_length_i = length; dut.context_trips_i = trips; dut.context_rf_base_i = base;
    dut.context_registers_i = registers; edge(); idle();
  }
  void completion(unsigned slot, unsigned block, unsigned epoch, unsigned iteration, unsigned pc) {
    dut.complete_i |= 1u << slot;
    dut.complete_block_i |= uint64_t(block) << (slot * 16);
    dut.complete_epoch_i |= uint32_t(epoch) << (slot * 8);
    dut.complete_iteration_i |= uint64_t(iteration) << (slot * 16);
    dut.complete_pc_i |= uint32_t(pc) << (slot * 5);
  }
  void writeback(unsigned slot, unsigned block, unsigned epoch, unsigned iteration, unsigned reg, bool input) {
    require(!dut.writeback_valid_i, "test service exceeded the single RF write port");
    dut.writeback_valid_i = 1; dut.writeback_input_i = input; dut.writeback_slot_i = slot;
    dut.writeback_block_i = block; dut.writeback_epoch_i = epoch;
    dut.writeback_iteration_i = iteration; dut.writeback_register_i = reg;
  }
  uint64_t word(unsigned slot) const {
    return uint64_t(dut.frontier_word_o[slot * 2]) | uint64_t(dut.frontier_word_o[slot * 2 + 1]) << 32;
  }
};

void protocol_test(const std::string &kind) {
  Control c; auto &d = c.dut;
  // The physical slots are intentionally not in tag priority order.
  const uint64_t code = kind == "reserved" ? 1ULL << 56 : kind == "bad-pipeline" ? 1ULL << 54 :
      kind == "bad-opcode" ? 15ULL << 60 : kind == "register-range" ? 15ULL << 50 : 0;
  if (kind != "missing-code") c.code(0, code);
  if (kind == "rf-data") c.code(1, 1ULL << 60 | 1ULL << 54);
  const unsigned length = kind == "rf-data" ? 2 : 1;
  c.descriptor(0, 37, 10, 9, 0, kind == "template-range" ? 33 : length, kind == "zero-trips" ? 0 : 2, 0, 2);
  c.descriptor(1, kind == "duplicate-id" ? 37 : 9001, 2, kind == "epoch-mismatch" ? 8 : 9, 0, length, 2,
               kind == "rf-overlap" ? 1 : kind == "rf-capacity" ? 15 : 2, 2);
  d.admit_i = 1; c.edge(); c.idle();
  const bool admission_fault = kind == "missing-code" || kind == "reserved" || kind == "zero-trips"
      || kind == "duplicate-id" || kind == "rf-overlap" || kind == "bad-pipeline" || kind == "bad-opcode"
      || kind == "register-range" || kind == "template-range" || kind == "epoch-mismatch" || kind == "rf-capacity";
  if (admission_fault) {
    require(d.error_o && !d.active_o && !d.inflight_o, "invalid admission changed execution state");
    return;
  }
  require(!d.error_o && d.active_o == 3, "protocol fixture admission failed");
  d.resource_ready_i = 3; d.issue_ready_i = 0; d.eval();
  require(d.issue_valid_o && d.issue_slot_o == 1 && d.issue_word_o == 0, "tag priority did not select its instruction");
  for (unsigned n = 0; n < 5; ++n) c.edge();
  require(!d.inflight_o && !d.pc_o && !d.iteration_o, "unaccepted offer advanced a context");
  d.resource_ready_i = 1; d.eval();
  require(d.issue_valid_o && d.issue_slot_o == 0, "resource-blocked priority tag prevented bypass");
  d.resource_ready_i = 3; d.issue_ready_i = 1; c.edge(); c.idle();
  require(d.inflight_o == 2 && !d.iteration_o, "issue did not preserve independent frontier state");
  if (kind == "rf-data") {
    Vector values{};
    for (unsigned lane = 0; lane < MLX_CONTROL_LANES; ++lane) values[lane] = 0x3c00 + lane;
    c.completion(1, 9001, 9, 0, 0); c.writeback(1, 9001, 9, 0, 0, false);
    write_vector(d.writeback_data_i, values); c.edge(); c.idle();
    d.resource_ready_i = 3; d.eval();
    require(d.issue_valid_o && d.issue_slot_o == 1 && (d.issue_word_o >> 60) == 1
        && read_vector(d.issue_operand_a_o) == values, "selected instruction did not read its physical RF partition");
    c.edge(); c.idle(); c.completion(1, 9001, 9, 0, 1); c.edge(); c.idle();
    require(field(d.iteration_o, 1, 16) == 1 && !field(d.rf_valid_o, 1, 16), "iteration did not invalidate the RF partition");
    return;
  }
  if (kind == "arbiter") {
    d.resource_ready_i = 3; d.overlap_i = 0; d.eval();
    require(!d.issue_valid_o, "serial policy ignored an in-flight context");
    d.overlap_i = 1; d.eval();
    require(d.issue_valid_o && d.issue_slot_o == 0, "waiting context froze another ready context");
    d.issue_ready_i = 0;
    c.completion(1, 9001, 9, 0, 0); c.writeback(1, 9001, 9, 0, 0, false);
    c.edge(); c.idle();
    require(field(d.iteration_o, 1, 16) == 1 && field(d.iteration_o, 0, 16) == 0
        && !d.inflight_o && !field(d.rf_valid_o, 1, 16), "trip count/valid bits advanced incorrectly");
    // A second complete cannot advance the new iteration without a new issue.
    c.completion(1, 9001, 9, 0, 0); c.writeback(1, 9001, 9, 0, 0, false);
  } else if (kind == "config-busy") {
    d.program_write_i = 1; d.program_address_i = 0; d.program_word_i = 0;
  } else if (kind == "stale-writeback" || kind == "overwrite-mailbox") {
    c.writeback(0, 37, 9, kind == "stale-writeback" ? 1 : 0, 1, true);
    if (kind == "overwrite-mailbox") {
      c.edge(); c.idle(); require(!d.error_o, "first mailbox arrival was rejected");
      c.writeback(0, 37, 9, 0, 1, true);
    }
  } else {
    unsigned slot = kind == "unsolicited-complete" ? 0 : 1;
    c.completion(slot, kind == "stale-block" ? 123 : slot == 0 ? 37 : 9001,
                 kind == "stale-epoch" ? 8 : 9, kind == "stale-iteration" ? 1 : 0,
                 kind == "stale-pc" ? 1 : 0);
    if (kind != "missing-writeback") c.writeback(slot, slot == 0 ? 37 : 9001, 9, 0, 0, false);
  }
  const auto active = d.active_o, inflight = d.inflight_o;
  const auto pc = d.pc_o; const auto iteration = d.iteration_o;
  c.edge();
  require(d.error_o && d.active_o == active && d.inflight_o == inflight
      && d.pc_o == pc && d.iteration_o == iteration, "invalid event advanced or retired a context");
}

struct Operation {
  unsigned slot, index, id, layer, epoch, iteration, pc, op, dst, target, address;
  uint64_t due;
  Vector data{};
};
#ifdef MLX_RTL_FU
uint64_t compute_tag(const Operation &o) {
  return uint64_t(o.slot) | uint64_t(o.pc)<<2 | uint64_t(o.iteration)<<7 | uint64_t(o.epoch)<<23
      | uint64_t(o.id)<<31 | uint64_t(o.dst)<<47 | uint64_t(o.op)<<51;
}
#endif

Json::Value run_control(const Program &p, const std::map<unsigned, uint64_t> &image, const Timing &timing) {
  require(p.hardware.pes() == 1, "this component driver covers one PE; full array is not implemented here");
  require(p.hardware.lanes == MLX_CONTROL_LANES, "RTL SIMD width differs from the program resource contract");
  require(p.hardware.contexts == 4 && p.hardware.registers == 16 && p.hardware.instruction_words == 32,
          "component driver requires the compiled four-slot/16-register/32-word geometry");
  Control control; auto &d = control.dut; d.overlap_i = timing.overlap;
#ifdef MLX_RTL_FU
  control.fu.compute_ii_i = timing.compute_ii;
#endif
  // Memory/loopback remain TEST services, driven only by actual RTL requests.
  // With MLX_RTL_FU, compute holds tracing metadata, never computes a result
  // or decides availability/latency; those come from the RTL FU handshakes.
  auto spm = p.inputs;
  std::optional<Operation> memory, compute, packet;
  uint64_t cycle = 0, overlap_cycles = 0, config_cycles = 0;
#ifndef MLX_RTL_FU
  uint64_t next_compute = 0;
#endif
  unsigned issued = 0, completed = 0, retired = 0;
  std::vector<unsigned> group;
  std::map<unsigned, unsigned> slots;
  Json::Value events(Json::arrayValue);
  mlx::tagged::Simulator reference(p, timing);
  const std::vector<std::string> keys = {"event", "cycle", "block_id", "logical_layer_id", "context_slot", "epoch", "iteration", "pc", "op"};
  auto state_check = [&] {
    reference.tick();
    require(reference.cycles() == cycle, "reference clock mismatch");
    const auto snapshot = reference.snapshot();
    for (const auto &s : snapshot["contexts"]) {
      unsigned slot = s["slot"].asUInt();
      auto equal = [&](const char *key, unsigned actual) {
        require(s[key].asUInt() == actual, std::string("context mismatch at cycle ") + std::to_string(cycle)
            + " block=" + std::to_string(s["block_id"].asUInt()) + " field=" + key);
      };
      equal("block_id", field(d.block_o, slot, 16)); equal("layer", field(d.layer_o, slot, 16));
      equal("epoch", field(d.epoch_o, slot, 8)); equal("iteration", field(d.iteration_o, slot, 16));
      equal("pc", field(d.pc_o, slot, 5)); equal("valid_registers", field(d.rf_valid_o, slot, 16));
      require(s["inflight"].asBool() == bool(d.inflight_o & (1u << slot)), "inflight state mismatch");
      require(s["retired"].asBool() == !(d.active_o & (1u << slot)), "retirement state mismatch");
    }
  };
  auto metadata = [&](unsigned slot) {
    Operation o{};
    o.slot = slot; o.index = group.at(slot); o.id = field(d.block_o, slot, 16);
    o.layer = field(d.layer_o, slot, 16); o.epoch = field(d.epoch_o, slot, 8);
    o.iteration = field(d.iteration_o, slot, 16); o.pc = field(d.pc_o, slot, 5);
    o.op = control.word(slot) >> 60;
    return o;
  };
  auto emit = [&](const char *kind, const Operation &o) {
    Json::Value event(Json::objectValue);
    event["event"] = kind; event["cycle"] = Json::UInt64(cycle); event["block_id"] = o.id;
    event["logical_layer_id"] = o.layer; event["context_slot"] = o.slot; event["epoch"] = o.epoch;
    event["iteration"] = o.iteration; event["pc"] = o.pc; event["op"] = mlx::tagged::opcode_name(o.op);
    events.append(std::move(event));
  };
  for (unsigned wave = 0; wave < p.waves; ++wave) {
    require(!memory && !compute && !packet, "wave reuse before draining services");
    const auto configuration_start = control.edges;
    control.idle(); d.clear_config_i = 1; control.edge(); control.idle();
    for (const auto &[address, word] : image)
      if (address >= 0x1000 && address < 0x1020) control.code(address - 0x1000, word);
    group.clear(); slots.clear();
    for (unsigned i = 0; i < p.blocks.size(); ++i) if (p.blocks[i].wave == wave) group.push_back(i);
    std::sort(group.begin(), group.end(), [&](unsigned a, unsigned b) {
      return std::tie(p.blocks[a].layer, p.blocks[a].id) < std::tie(p.blocks[b].layer, p.blocks[b].id);
    });
    unsigned base = 0;
    for (unsigned slot = 0; slot < group.size(); ++slot) {
      const auto index = group[slot]; const auto &block = p.blocks[index];
      const auto code = image.at(0x101 + index * 4);
      control.descriptor(slot, block.id, block.layer, wave, code & 255, (code >> 8) & 255,
                         block.trips, base, block.registers);
      slots[index] = slot; base += block.registers;
    }
    config_cycles += control.edges - configuration_start;
    d.admit_i = 1; control.edge(true); control.idle();
    require(!d.error_o, "RTL rejected valid context/program admission: " + std::to_string(d.error_code_o));
    for (unsigned slot = 0; slot < group.size(); ++slot) emit("admit", metadata(slot));
    ++cycle; state_check();
    while (!d.done_o) {
      require(cycle < timing.max_cycles, "control component watchdog");
      control.idle();
      if (__builtin_popcount(unsigned(d.inflight_o)) >= 2) ++overlap_cycles;
      std::vector<Operation> completions;
      auto finish = [&](const Operation &o, bool write, bool incoming = false, unsigned target = 0) {
        completions.push_back(o);
        control.completion(o.slot, o.id, o.epoch, o.iteration, o.pc);
        if (write) {
          unsigned slot = incoming ? target : o.slot;
          control.writeback(slot, field(d.block_o, slot, 16), o.epoch, o.iteration, o.dst, incoming);
          write_vector(d.writeback_data_i, o.data);
        }
      };
      if (memory && memory->due <= cycle) finish(*memory, memory->op == 0);
#ifdef MLX_RTL_FU
      control.fu.eval();
      if (control.fu.response_valid_o && !d.writeback_valid_i) {
        require(bool(compute), "unowned RTL FU response");
        require(!control.fu.response_error_o && control.fu.response_tag_o == compute_tag(*compute),
                "RTL FU error or changed operation ownership");
        auto returned = *compute;
        const uint64_t tag = control.fu.response_tag_o;
        returned.slot = tag & 3; returned.pc = (tag >> 2) & 31; returned.iteration = (tag >> 7) & 65535;
        returned.epoch = (tag >> 23) & 255; returned.id = (tag >> 31) & 65535;
        returned.dst = (tag >> 47) & 15; returned.op = (tag >> 51) & 15;
        returned.data = read_vector(control.fu.response_data_o);
        finish(returned, true); control.fu.response_ready_i = 1;
      }
      const bool compute_ready = control.fu.request_ready_o;
#else
      if (compute && compute->due <= cycle && !d.writeback_valid_i) finish(*compute, true);
      const bool compute_ready = !compute && cycle >= next_compute;
#endif
      if (packet) {
        const auto target = slots.at(packet->target);
        if ((d.active_o & (1u << target)) && field(d.iteration_o, target, 16) == packet->iteration
            && !(field(d.rf_valid_o, target, 16) & (1u << packet->dst))
            && !d.writeback_valid_i && cycle % timing.receive_period == 0) finish(*packet, true, true, target);
      }
      for (unsigned slot = 0; slot < group.size(); ++slot) {
        const auto word = control.word(slot); const auto op = word >> 60;
        bool available = op <= 1 ? !memory && cycle % timing.memory_period == 0 :
            op == 8 ? !packet : compute_ready;
        if (op == 8) {
          const auto target = slots.at((word >> 4) & 65535);
          available = available && (d.active_o & (1u << target))
              && field(d.iteration_o, target, 16) == field(d.iteration_o, slot, 16)
              && !(field(d.rf_valid_o, target, 16) & (1u << ((word >> 50) & 15)));
        }
        if (available) d.resource_ready_i |= 1u << slot;
      }
      d.eval();
      require(!d.error_o && d.complete_accept_o == d.complete_i
          && bool(d.writeback_accept_o) == bool(d.writeback_valid_i), "valid service completion was rejected");
      std::optional<Operation> issue;
      if (d.issue_fire_o) {
        const unsigned slot = d.issue_slot_o; auto o = metadata(slot); const auto word = d.issue_word_o;
        require(word == control.word(slot), "selected tag did not select its instruction word");
        o.dst = (word >> 50) & 15; o.target = (word >> 4) & 65535;
        const int stride = int(int8_t((word >> 12) & 255));
        o.address = int((word >> 20) & 255) + stride * int(o.iteration);
        o.due = cycle + timing.latency.at(o.op);
        const std::array<Vector, 3> operands = {read_vector(d.issue_operand_a_o),
            read_vector(d.issue_operand_b_o), read_vector(d.issue_operand_c_o)};
        if (o.op == 0) o.data = spm.at(o.address);
        else if (o.op == 1 || o.op == 8) o.data = operands[0];
        else {
#ifdef MLX_RTL_FU
          control.fu.request_valid_i = 1; control.fu.operation_i = o.op;
          control.fu.request_tag_i = compute_tag(o);
          write_vector(control.fu.operand_a_i, operands[0]); write_vector(control.fu.operand_b_i, operands[1]);
          write_vector(control.fu.operand_c_i, operands[2]); control.fu.eval();
          require(control.fu.request_ready_o, "compute issue lacked a real FU resource");
#else
          o.data = mlx::tagged::arithmetic(o.op, operands, p.hardware.lanes);
#endif
        }
        issue = o;
      }
      const auto retiring = d.retire_o;
      for (const auto &o : completions) {
        if (o.op == 1) spm.at(o.address) = o.data;
        emit("complete", o); ++completed;
        if (retiring & (1u << o.slot)) { emit("retire", o); ++retired; }
        if (o.op <= 1) memory.reset(); else if (o.op == 8) packet.reset(); else compute.reset();
      }
      if (issue) {
        emit("issue", *issue); ++issued;
        if (issue->op <= 1) memory = *issue;
        else if (issue->op == 8) packet = *issue;
        else {
          compute = *issue;
#ifndef MLX_RTL_FU
          next_compute = cycle + timing.compute_ii;
#endif
        }
      }
      control.edge(true); control.idle();
      require(!d.error_o, "RTL protocol error " + std::to_string(d.error_code_o));
      ++cycle; state_check();
    }
  }
  require(reference.done(), "RTL finished before the independent C++ model");
  const auto golden = reference.result();
  Json::Value expected(Json::arrayValue);
  for (const auto &event : golden["events"]) {
    const auto kind = event["event"].asString();
    if (kind != "admit" && kind != "issue" && kind != "complete" && kind != "retire") continue;
    Json::Value selected(Json::objectValue);
    for (const auto &key : keys) selected[key] = event[key];
    expected.append(std::move(selected));
  }
  require(events == expected, "RTL control event trace differs from independent native model");
  require(issued == completed && retired == p.blocks.size(), "RTL instruction/context conservation failed");
  require(overlap_cycles == golden["counters"]["overlap_pe_cycles"].asUInt64(), "overlap accounting differs");
  Json::Value result(Json::objectValue);
  result["build_identity"] = MLX_CONTROL_BUILD_ID;
#ifdef MLX_RTL_FU
  result["classification"] = "rtl_pe_frontend_and_fu_with_cpp_memory_services_not_full_rtl";
  result["fu_execution_clock_only"] = true;
#else
  result["classification"] = "rtl_pe_frontend_with_cpp_test_services_not_full_rtl";
#endif
  result["cycles"] = Json::UInt64(cycle); result["configuration_cycles"] = Json::UInt64(config_cycles);
  result["issue"] = issued; result["complete"] = completed; result["retired"] = retired;
  result["overlap_pe_cycles"] = Json::UInt64(overlap_cycles); result["events"] = events;
  result["state_comparisons"] = Json::UInt64(cycle);
  for (auto address : p.outputs) {
    auto &out = result["outputs"][std::to_string(address)]; out = Json::Value(Json::arrayValue);
    for (unsigned lane = 0; lane < p.hardware.lanes; ++lane) out.append(spm[address][lane]);
  }
  require(result["outputs"] == golden["outputs"], "test service data differs from independent model");
  return result;
}

int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--protocol") {
      protocol_test(argv[2]); std::cout << "MLX_TAGGED_CONTROL_PROTOCOL_PASS " << argv[2] << '\n'; return 0;
    }
    require(argc == 9, "usage: control-test program.json image.hex output.json overlap load-latency compute-II memory-period receive-period");
    std::ifstream source(argv[1]); Json::Value raw; source >> raw; const auto program = Program::parse(raw);
    std::ifstream input(argv[2]); std::map<unsigned, uint64_t> image; unsigned address; uint64_t word;
    while (input >> std::hex >> address >> word) image[address] = word;
    require(input.eof(), "invalid image file");
    Timing timing; timing.overlap = std::stoul(argv[4]) != 0; timing.latency[0] = std::stoul(argv[5]);
    timing.compute_ii = std::stoul(argv[6]); timing.memory_period = std::stoul(argv[7]);
    timing.receive_period = std::stoul(argv[8]); timing.validate();
    const auto result = run_control(program, image, timing);
    std::ofstream output(argv[3]); output << result << '\n'; require(bool(output), "cannot write control result");
    std::cout << "MLX_TAGGED_CONTROL_PASS cycles=" << result["cycles"].asUInt64() << '\n';
  } catch (const std::exception &error) {
    std::cerr << "MLX_TAGGED_CONTROL_FAIL: " << error.what() << '\n'; return 1;
  }
}
