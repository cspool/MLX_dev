#include "mlx_tagged_simulator.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace mlx::tagged {
namespace {
enum Count : unsigned {
  Admitted, Retired, Reserved, Released, Waves, Issue, Complete, Sent, Delivered, Links,
  Overlap, ResidentCycles, InflightCycles, ReadyCycles, MaxResident, WritebackStall,
  DeliveryStall, RouteStall, IssueLoad, IssueStore, IssueCompute, IssueXfer,
  CompleteLoad, CompleteStore, CompleteCompute, CompleteXfer,
  BusyLoad, BusyStore, BusyCompute, BusyXfer,
  StallDependency, StallSerial, StallCompute, StallSpm, StallNetwork, StallIssue, StallCredit, CountSize
};
const char *counter_names[CountSize] = {
  "admitted", "retired", "registers_reserved", "registers_released", "waves_admitted",
  "issue", "complete", "xfer_sent", "xfer_delivered", "routed_links", "overlap_pe_cycles",
  "resident_context_cycles", "inflight_context_cycles", "ready_context_cycles",
  "max_resident_contexts_per_pe", "writeback_stall", "network_delivery_stall", "network_route_stall",
  "issue_load", "issue_store", "issue_compute", "issue_xfer",
  "complete_load", "complete_store", "complete_compute", "complete_xfer",
  "busy_load_pe_cycles", "busy_store_pe_cycles", "busy_compute_pe_cycles", "busy_xfer_pe_cycles",
  "stall_dependency", "stall_serial_policy", "stall_compute_capacity_or_ii", "stall_spm",
  "stall_network", "stall_issue_bandwidth", "stall_destination_credit"
};
const char *reasons[] = {"dependency", "serial_policy", "compute_capacity_or_ii", "spm", "network", "issue_bandwidth", "destination_credit"};
struct Context {
  unsigned slot = 0, base = 0, wave = 0, pc = 0, iteration = 0;
  uint16_t valid = 0;
  bool inflight = false, retired = false;
  int last_wait = -1;
};
struct Operation {
  unsigned block = 0, wave = 0, iteration = 0, pc = 0, address = 0;
  uint64_t due = 0;
  Vector data{};
};
struct Packet { Operation operation; unsigned target = 0, reg = 0, pe = 0; };
bool ready(const Context &context, const Instruction &instruction) {
  for (unsigned i = 0; i < instruction.arity; ++i)
    if (!(context.valid & (1u << instruction.src[i]))) return false;
  return true;
}

class Machine {
public:
  Machine(const Program &program, const Timing &timing)
      : program(program), timing(timing), contexts(program.blocks.size()), memory(program.inputs) {}

  bool tick() {
    if (finished) return true;
    if (cycle >= timing.max_cycles) {
      std::ostringstream message;
      message << "cycle budget exhausted; pending contexts:";
      for (const auto &group : pe_contexts) for (auto index : group) {
        const auto &c = contexts[index];
        if (!c.retired) message << " (block=" << program.blocks[index].id << ",pc=" << c.pc
                                << ",iteration=" << c.iteration << ",inflight=" << c.inflight << ")";
      }
      throw std::runtime_error(message.str());
    }
    step();
    return finished;
  }
  bool done() const { return finished; }
  uint64_t cycles() const { return cycle; }
  Json::Value snapshot() const {
    Json::Value result(Json::objectValue);
    result["cycles"] = Json::UInt64(cycle);
    result["done"] = finished;
    result["wave"] = wave;
    result["contexts"] = Json::Value(Json::arrayValue);
    for (const auto &group : pe_contexts) for (auto index : group) {
      const auto &c = contexts[index]; const auto &b = program.blocks[index];
      Json::Value state(Json::objectValue);
      state["block_id"] = b.id; state["layer"] = b.layer; state["pe"] = b.pe;
      state["slot"] = c.slot; state["epoch"] = c.wave; state["pc"] = c.pc;
      state["iteration"] = c.iteration; state["inflight"] = c.inflight;
      state["retired"] = c.retired; state["valid_registers"] = c.valid;
      result["contexts"].append(std::move(state));
    }
    result["outstanding_memory"] = bool(memory_job);
    result["outstanding_compute"] = unsigned(std::count_if(compute.begin(), compute.end(),
        [](const auto &value) { return bool(value); }));
    result["network_packets"] = unsigned(std::count_if(routers.begin(), routers.end(),
        [](const auto &value) { return bool(value); }));
    return result;
  }
  Json::Value result() const {
    if (!finished) throw std::logic_error("result requested before model completion");
    for (auto pair : {std::pair{Issue, Complete}, {Admitted, Retired}, {Reserved, Released}, {Sent, Delivered}})
      if (counts[pair.first] != counts[pair.second]) throw std::runtime_error("resource/instruction conservation failure");
    Json::Value result(Json::objectValue);
    result["backend"] = "tagged_cpp_v2";
    result["cycles"] = Json::UInt64(cycle);
    result["program"] = program.name;
    result["abi_version"] = 2;
    result["tracing"] = timing.trace;
    for (unsigned i = 0; i < CountSize; ++i) result["counters"][counter_names[i]] = Json::UInt64(counts[i]);
    for (auto address : program.outputs) {
      auto &vector = result["outputs"][std::to_string(address)];
      vector = Json::Value(Json::arrayValue);
      for (unsigned lane = 0; lane < program.hardware.lanes; ++lane) vector.append(memory[address][lane]);
    }
    result["events"] = events;
    return result;
  }

private:
  const Program &program;
  const Timing &timing;
  std::vector<Context> contexts;
  std::array<std::vector<unsigned>, 16> pe_contexts;
  std::array<Vector, 128> memory;
  std::array<std::array<Vector, 16>, 16> rf{};
  std::array<std::optional<Operation>, 16> compute;
  std::optional<Operation> memory_job;
  std::array<std::optional<Packet>, 16> routers;
  std::array<uint64_t, 16> next_compute{};
  std::array<uint64_t, CountSize> counts{};
  Json::Value events{Json::arrayValue};
  uint64_t cycle = 0;
  unsigned wave = 0, live = 0;
  bool admitted = false, finished = false;

  std::string uid(unsigned index) const {
    const auto &c = contexts[index];
    return std::to_string(c.wave) + ":" + std::to_string(program.blocks[index].id) + ":"
           + std::to_string(c.iteration) + ":" + std::to_string(c.pc);
  }
  Json::Value base_event(const char *event, unsigned index) const {
    const auto &c = contexts[index];
    const auto &b = program.blocks[index];
    const auto &i = b.instructions[c.pc];
    Json::Value e(Json::objectValue);
    e["backend"] = "tagged_cpp_v2"; e["cycle"] = Json::UInt64(cycle); e["phase"] = "edge_commit";
    e["event"] = event; e["pe"] = b.pe; e["logical_layer_id"] = b.layer; e["block_id"] = b.id;
    e["context_slot"] = c.slot; e["epoch"] = c.wave; e["iteration"] = c.iteration; e["pc"] = c.pc;
    e["op"] = opcode_name(i.op); e["pipeline"] = pipeline_name(i.pipeline()); e["uid"] = uid(index);
    return e;
  }
  void emit(const char *event, unsigned index) {
    if (timing.trace) events.append(base_event(event, index));
  }
  void address_event(const char *name, unsigned index, unsigned address) {
    if (!timing.trace) return;
    auto e = base_event(name, index); e["address"] = address; events.append(std::move(e));
  }
  void register_event(unsigned index, unsigned reg, const std::string &producer = "") {
    if (!timing.trace) return;
    auto e = base_event("rf_write", index); e["register"] = reg; e["physical_register"] = contexts[index].base + reg;
    if (!producer.empty()) e["producer_uid"] = producer;
    events.append(std::move(e));
  }
  void reason_event(const char *event, unsigned index, int reason) {
    if (!timing.trace) return;
    auto e = base_event(event, index); e["reason"] = reasons[reason]; events.append(std::move(e));
  }
  void admit() {
    live = 0;
    for (unsigned pe = 0; pe < program.hardware.pes(); ++pe) {
      auto &group = pe_contexts[pe]; group.clear();
      for (unsigned i = 0; i < program.blocks.size(); ++i)
        if (program.blocks[i].pe == pe && program.blocks[i].wave == wave) group.push_back(i);
      std::sort(group.begin(), group.end(), [&](unsigned a, unsigned b) {
        return std::tie(program.blocks[a].layer, program.blocks[a].id) <
               std::tie(program.blocks[b].layer, program.blocks[b].id);
      });
      unsigned base = 0, slot = 0;
      for (auto index : group) {
        const auto &b = program.blocks[index];
        contexts[index] = Context{};
        auto &c = contexts[index]; c.base = base; c.slot = slot++; c.wave = wave;
        if (timing.trace) {
          auto e = base_event("admit", index); e["rf_base"] = base; e["registers"] = b.registers;
          events.append(std::move(e));
        }
        base += b.registers; ++counts[Admitted]; counts[Reserved] += b.registers; ++live;
      }
    }
    admitted = true; ++counts[Waves];
  }
  void complete(const Operation &operation) {
    auto &c = contexts[operation.block]; const auto &b = program.blocks[operation.block];
    if (c.retired || !c.inflight || c.wave != operation.wave || c.pc != operation.pc || c.iteration != operation.iteration)
      throw std::runtime_error("stale or duplicate completion");
    emit("complete", operation.block);
    ++counts[Complete]; ++counts[CompleteLoad + unsigned(b.instructions[c.pc].pipeline())];
    c.inflight = false;
    if (c.pc + 1 < b.instructions.size()) ++c.pc;
    else if (c.iteration + 1 < b.trips) {
      emit("iteration_complete", operation.block); ++c.iteration; c.pc = 0; c.valid = 0;
    } else {
      emit("retire", operation.block); c.retired = true; --live; ++counts[Retired]; counts[Released] += b.registers;
    }
  }
  unsigned next_hop(unsigned source, unsigned target) const {
    const unsigned columns = program.hardware.columns;
    const int dx = int(target % columns) - int(source % columns);
    const int dy = int(target / columns) - int(source / columns);
    return dx ? int(source) + (dx > 0 ? 1 : -1) * std::min(2, std::abs(dx)) :
                int(source) + (dy > 0 ? 1 : -1) * std::min(2, std::abs(dy)) * int(columns);
  }
  void step() {
    if (!admitted) { admit(); ++cycle; return; }
    const auto pes = program.hardware.pes();
    uint16_t writers = 0, reserved_routers = 0;
    std::array<Operation, 17> completions;
    unsigned ncomplete = 0;
    if (memory_job && memory_job->due <= cycle) {
      completions[ncomplete++] = *memory_job;
      if (program.blocks[memory_job->block].instructions[memory_job->pc].op == 0)
        writers |= 1u << program.blocks[memory_job->block].pe;
    }
    for (unsigned pe = 0; pe < pes; ++pe) if (compute[pe] && compute[pe]->due <= cycle) {
      if (!(writers & (1u << pe))) { writers |= 1u << pe; completions[ncomplete++] = *compute[pe]; }
      else ++counts[WritebackStall];
    }
    struct Delivery { unsigned pe; Packet packet; };
    struct Move { unsigned source, destination; Packet packet; };
    std::array<Delivery, 16> deliveries;
    std::array<Move, 16> moves;
    std::array<int, 16> proposal;
    proposal.fill(-1);
    uint16_t delivering = 0;
    unsigned ndelivery = 0, nmoves = 0;
    for (unsigned pe = 0; pe < pes; ++pe) if (routers[pe]) {
      const auto &packet = *routers[pe];
      if (pe == packet.pe) {
        const auto &target = contexts[packet.target];
        if (target.retired || packet.operation.wave != target.wave || packet.operation.iteration < target.iteration)
          throw std::runtime_error("late packet for retired context/iteration");
        if (target.iteration == packet.operation.iteration && !(target.valid & (1u << packet.reg)) &&
            !(writers & (1u << pe)) && cycle % timing.receive_period == 0) {
          writers |= 1u << pe; deliveries[ndelivery++] = {pe, packet}; delivering |= 1u << pe;
        } else ++counts[DeliveryStall];
      } else {
        unsigned next = next_hop(pe, packet.pe);
        if (!(reserved_routers & (1u << next)) && cycle % timing.link_period == 0) {
          reserved_routers |= 1u << next; proposal[pe] = int(next);
        } else ++counts[RouteStall];
      }
    }
    // Simultaneous dequeue/enqueue is legal even for a full registered buffer.
    // Remove blocked paths to a fixed point, retaining complete exchange cycles.
    bool changed;
    do {
      changed = false;
      uint16_t remove = 0;
      for (unsigned pe = 0; pe < pes; ++pe) if (proposal[pe] >= 0) {
        unsigned target = unsigned(proposal[pe]);
        if (routers[target] && !(delivering & (1u << target)) && proposal[target] < 0)
          remove |= 1u << pe;
      }
      for (unsigned pe = 0; pe < pes; ++pe) if (remove & (1u << pe)) {
        proposal[pe] = -1; ++counts[RouteStall]; changed = true;
      }
    } while (changed);
    reserved_routers = 0;
    for (unsigned pe = 0; pe < pes; ++pe) if (proposal[pe] >= 0) {
      unsigned target = unsigned(proposal[pe]);
      moves[nmoves++] = {pe, target, *routers[pe]}; reserved_routers |= 1u << target;
    }
    // Pick candidates using only start-of-cycle state. Resource availability
    // and instruction readiness are separate; a blocked context is bypassed.
    bool memory_available = !memory_job && cycle % timing.memory_period == 0;
    std::array<Operation, 16> issues;
    unsigned nissue = 0;
    for (unsigned pe = 0; pe < pes; ++pe) {
      unsigned resident = 0, inflight = 0, ready_count = 0, busy_mask = 0;
      for (auto index : pe_contexts[pe]) {
        const auto &c = contexts[index]; const auto &i = program.blocks[index].instructions[c.pc];
        if (c.retired) continue;
        ++resident;
        if (c.inflight) { ++inflight; busy_mask |= 1u << unsigned(i.pipeline()); }
        else if (ready(c, i)) ++ready_count;
      }
      counts[ResidentCycles] += resident; counts[InflightCycles] += inflight; counts[ReadyCycles] += ready_count;
      counts[MaxResident] = std::max(counts[MaxResident], uint64_t(resident));
      if (inflight >= 2) ++counts[Overlap];
      for (unsigned p = 0; p < 4; ++p) if (busy_mask & (1u << p)) ++counts[BusyLoad + p];
      bool issued = false;
      for (auto index : pe_contexts[pe]) {
        auto &c = contexts[index]; const auto &b = program.blocks[index]; const auto &i = b.instructions[c.pc];
        if (c.retired || c.inflight) continue;
        int reason = -1;
        if (!ready(c, i)) reason = 0;
        else if (!timing.overlap && inflight) reason = 1;
        else if (i.pipeline() == Pipeline::Compute && (compute[pe] || cycle < next_compute[pe])) reason = 2;
        else if (i.op <= 1 && !memory_available) reason = 3;
        else if (i.op == 8 && (routers[pe] || (reserved_routers & (1u << pe)))) reason = 4;
        else if (i.op == 8 && (contexts[i.target].retired || contexts[i.target].iteration != c.iteration
                              || (contexts[i.target].valid & (1u << i.dst)))) reason = 6;
        else if (issued) reason = 5;
        if (reason >= 0) {
          ++counts[StallDependency + unsigned(reason)];
          if (c.last_wait != reason) reason_event("wait_event", index, reason);
          c.last_wait = reason; continue;
        }
        if (c.last_wait >= 0) { reason_event("wake", index, c.last_wait); c.last_wait = -1; }
        Operation operation;
        operation.block = index; operation.wave = c.wave; operation.iteration = c.iteration; operation.pc = c.pc;
        operation.address = int64_t(i.spm) + int64_t(i.stride) * c.iteration;
        operation.due = cycle + (i.op == 8 ? 1 : timing.latency[i.op]);
        if (i.pipeline() == Pipeline::Compute) {
          std::array<Vector, 3> operands{};
          for (unsigned r = 0; r < i.arity; ++r) operands[r] = rf[pe][c.base + i.src[r]];
          operation.data = arithmetic(i.op, operands, program.hardware.lanes);
        } else if (i.op == 0) operation.data = memory[operation.address];
        else operation.data = rf[pe][c.base + i.src[0]];
        issues[nissue++] = operation;
        if (i.op <= 1) memory_available = false;
        issued = true;
      }
    }
    // Commit selected effects. No newly produced value can issue in this edge.
    for (unsigned n = 0; n < ncomplete; ++n) {
      const auto &o = completions[n]; auto &c = contexts[o.block];
      const auto &b = program.blocks[o.block]; const auto &i = b.instructions[o.pc];
      if (i.op == 1) memory[o.address] = o.data;
      else { rf[b.pe][c.base + i.dst] = o.data; c.valid |= 1u << i.dst; register_event(o.block, i.dst); }
      if (i.op <= 1) { address_event("memory_response", o.block, o.address); memory_job.reset(); }
      else compute[b.pe].reset();
      complete(o);
    }
    for (unsigned n = 0; n < ndelivery; ++n) {
      const auto &d = deliveries[n]; const auto &p = d.packet; auto &target = contexts[p.target];
      rf[d.pe][target.base + p.reg] = p.operation.data; target.valid |= 1u << p.reg;
      if (timing.trace) {
        const auto producer = uid(p.operation.block);
        register_event(p.target, p.reg, producer);
        auto e = base_event("xfer_receive", p.target); e["producer_uid"] = producer; e["register"] = p.reg;
        e["event_id"] = std::to_string(wave) + ":" + std::to_string(program.blocks[p.target].id) + ":" +
                        std::to_string(p.operation.iteration) + ":" + std::to_string(p.reg);
        events.append(std::move(e));
      }
      complete(p.operation); routers[d.pe].reset(); ++counts[Delivered];
    }
    for (unsigned n = 0; n < nmoves; ++n) routers[moves[n].source].reset();
    for (unsigned n = 0; n < nmoves; ++n) {
      const auto &m = moves[n]; routers[m.destination] = m.packet; ++counts[Links];
      if (timing.trace) {
        auto e = base_event("route", m.packet.operation.block); e["source_pe"] = m.source; e["target_pe"] = m.destination;
        events.append(std::move(e));
      }
    }
    for (unsigned n = 0; n < nissue; ++n) {
      const auto &o = issues[n]; auto &c = contexts[o.block];
      const auto &b = program.blocks[o.block]; const auto &i = b.instructions[o.pc];
      if (c.inflight || c.retired || c.pc != o.pc || c.iteration != o.iteration || c.wave != o.wave)
        throw std::runtime_error("issue state changed during edge planning");
      c.inflight = true; emit("issue", o.block); ++counts[Issue]; ++counts[IssueLoad + unsigned(i.pipeline())];
      if (i.pipeline() == Pipeline::Compute) { compute[b.pe] = o; next_compute[b.pe] = cycle + timing.compute_ii; }
      else if (i.op <= 1) { memory_job = o; address_event("memory_request", o.block, o.address); }
      else {
        routers[b.pe] = Packet{o, i.target, i.dst, program.blocks[i.target].pe}; ++counts[Sent];
        if (timing.trace) {
          auto e = base_event("xfer_send", o.block); e["target_block"] = program.blocks[i.target].id; e["register"] = i.dst;
          events.append(std::move(e));
        }
      }
    }
    if (!live) {
      if (memory_job || std::any_of(compute.begin(), compute.end(), [](const auto &x) { return bool(x); }) ||
          std::any_of(routers.begin(), routers.end(), [](const auto &x) { return bool(x); }))
        throw std::runtime_error("retired wave still has outstanding work");
      admitted = false; finished = ++wave == program.waves;
    }
    ++cycle;
  }
};
} // namespace

struct Simulator::Impl {
  Program program;
  Timing timing;
  Machine machine;
  Impl(Program p, Timing t) : program(std::move(p)), timing(std::move(t)), machine(program, timing) {
    program.validate();
    timing.validate();
  }
};
Simulator::Simulator(Program program, Timing timing)
    : impl(std::make_unique<Impl>(std::move(program), std::move(timing))) {}
Simulator::~Simulator() = default;
Simulator::Simulator(Simulator &&) noexcept = default;
Simulator &Simulator::operator=(Simulator &&) noexcept = default;
bool Simulator::tick() { return impl->machine.tick(); }
bool Simulator::done() const { return impl->machine.done(); }
uint64_t Simulator::cycles() const { return impl->machine.cycles(); }
Json::Value Simulator::snapshot() const { return impl->machine.snapshot(); }
Json::Value Simulator::result() const { return impl->machine.result(); }

Json::Value simulate(const Program &program, const Timing &timing) {
  Simulator model(program, timing);
  while (!model.tick()) {}
  return model.result();
}
} // namespace mlx::tagged
