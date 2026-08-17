#include "mlx_overlay.hh"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using dsa::sim::mlx::Config;
using dsa::sim::mlx::Coord;
using dsa::sim::mlx::Event;
using dsa::sim::mlx::Instruction;
using dsa::sim::mlx::MemoryAdapter;
using dsa::sim::mlx::MemoryBackend;
using dsa::sim::mlx::MemoryRequest;
using dsa::sim::mlx::Overlay;
using dsa::sim::mlx::PipelineKind;
using dsa::sim::mlx::TaggedBlock;

struct Check {
  std::string name;
  bool pass{false};
  std::string observation;
};

struct Scenario {
  std::string id;
  std::unique_ptr<Overlay> overlay;
  std::vector<Check> checks;
};

class FakeMemoryAdapter : public MemoryAdapter {
 public:
  FakeMemoryAdapter(unsigned capacity, uint64_t delay)
      : capacity_(capacity), delay_(delay) {}

  void cycle(uint64_t cycle) { cycle_ = cycle; }

  bool available(const MemoryRequest &) const override {
    return outstanding_.size() < capacity_;
  }

  uint64_t issue(const MemoryRequest &) override {
    if (outstanding_.size() >= capacity_) {
      throw std::runtime_error("fake adapter capacity exceeded");
    }
    uint64_t token = next_token_++;
    outstanding_[token] = cycle_ + delay_;
    ++issued_;
    return token;
  }

  bool takeCompletion(uint64_t token) override {
    auto iterator = outstanding_.find(token);
    if (iterator == outstanding_.end() || iterator->second > cycle_) {
      return false;
    }
    outstanding_.erase(iterator);
    ++completed_;
    return true;
  }

  uint64_t issued() const { return issued_; }
  uint64_t completed() const { return completed_; }
  std::size_t outstanding() const { return outstanding_.size(); }

 private:
  unsigned capacity_;
  uint64_t delay_;
  uint64_t cycle_{0};
  uint64_t next_token_{1};
  uint64_t issued_{0};
  uint64_t completed_{0};
  std::map<uint64_t, uint64_t> outstanding_;
};

std::string
Escape(const std::string &value)
{
  std::ostringstream output;
  for (char character : value) {
    if (character == '\\' || character == '\"') output << '\\';
    if (character == '\n') {
      output << "\\n";
    } else {
      output << character;
    }
  }
  return output.str();
}

std::string
Quote(const std::string &value)
{
  return "\"" + Escape(value) + "\"";
}

Config
BaseConfig(unsigned window)
{
  Config config;
  config.active_window = window;
  config.register_file = {4, 2, 1};
  config.pipeline_timing = {
      {PipelineKind::Load, {1, 1}},
      {PipelineKind::Store, {1, 1}},
      {PipelineKind::Compute, {1, 1}},
      {PipelineKind::Xfer, {1, 1}},
  };
  config.functional_units = {
      {"add", {"alu", {1, 1}}},
      {"mul", {"mul", {3, 1}}},
      {"fma", {"fma", {4, 1}}},
      {"fexp", {"transcendental", {8, 4}}},
  };
  config.routing = {4, 4, {2, 1}, 1, 1};
  return config;
}

Instruction
Compute(const std::string &id, const std::string &operation, const std::string &event = "")
{
  Instruction instruction;
  instruction.id = id;
  instruction.pipeline = PipelineKind::Compute;
  instruction.operation = operation;
  instruction.emit_event = event;
  return instruction;
}

Instruction
Memory(const std::string &id, PipelineKind pipeline, uint64_t address)
{
  Instruction instruction;
  instruction.id = id;
  instruction.pipeline = pipeline;
  instruction.operation = pipeline == PipelineKind::Load ? "load" : "store";
  instruction.memory_address = address;
  instruction.memory_bytes = 8;
  if (pipeline == PipelineKind::Load) instruction.writes = {0};
  if (pipeline == PipelineKind::Store) instruction.reads = {0};
  return instruction;
}

TaggedBlock
Block(
    const std::string &id, uint64_t tag, Coord pe,
    std::vector<Instruction> instructions, uint64_t trips = 1,
    std::vector<std::string> wait_events = {})
{
  TaggedBlock block;
  block.id = id;
  block.tag = tag;
  block.pe = pe;
  block.trip_count = trips;
  block.wait_events = std::move(wait_events);
  block.instructions = std::move(instructions);
  return block;
}

std::vector<const Event *>
Select(
    const Overlay &overlay, const std::string &kind,
    const std::string &block = "")
{
  std::vector<const Event *> selected;
  for (const auto &event : overlay.events()) {
    if (event.kind == kind && (block.empty() || event.block == block)) {
      selected.push_back(&event);
    }
  }
  return selected;
}

uint64_t
TagEventCycle(const Overlay &overlay, const std::string &kind, uint64_t tag)
{
  for (const Event *event : Select(overlay, kind)) {
    if (event->tag == tag) return event->cycle;
  }
  throw std::runtime_error("tag event not found");
}

uint64_t
Stalls(const Overlay &overlay, const std::string &reason)
{
  auto iterator = overlay.stats().stalls_by_reason.find(reason);
  return iterator == overlay.stats().stalls_by_reason.end() ? 0 : iterator->second;
}

Scenario
EventOverlap()
{
  Config config = BaseConfig(2);
  std::vector<TaggedBlock> blocks = {
      Block("producer_fast", 1, {0, 0}, {Compute("emit", "add", "edge")}, 2),
      Block("producer_slow", 1, {1, 0}, {Compute("slow", "fexp")}, 2),
      Block("successor", 2, {2, 0}, {Compute("consume", "add")}, 2, {"edge"}),
  };
  auto overlay = std::make_unique<Overlay>(config, std::move(blocks));
  overlay->run(128);
  auto emissions = Select(*overlay, "event_emit", "producer_fast");
  auto successor_issues = Select(*overlay, "issue", "successor");
  uint64_t predecessor_complete = TagEventCycle(*overlay, "tag_complete", 1);
  uint64_t successor_complete = TagEventCycle(*overlay, "tag_complete", 2);
  bool event_counts = emissions.size() == 2 && emissions[0]->cycle == 1 &&
                      emissions[1]->cycle == 2;
  bool iteration_order = successor_issues.size() == 2 &&
                         successor_issues[0]->cycle >= emissions[0]->cycle &&
                         successor_issues[1]->cycle >= emissions[1]->cycle;
  bool overlap = successor_issues.front()->cycle < predecessor_complete &&
                 successor_complete < predecessor_complete;
  return {
      "event_counted_cross_layer_overlap", std::move(overlay),
      {{"two_events_emitted", event_counts, "edge counts expected at cycles one/two"},
       {"iteration_event_order", iteration_order,
        "consumer iteration i cannot precede event count i+1"},
       {"successor_overlaps_predecessor", overlap,
        "tag2 must issue and retire before slow tag1 retirement"},
       {"dependency_stall_observed", false, "filled after overlay move"}}};
}

Scenario
AdapterBackpressure()
{
  Config config = BaseConfig(2);
  config.memory_backend = MemoryBackend::Adapter;
  std::vector<TaggedBlock> blocks = {
      Block(
          "memory_a", 1, {0, 0},
          {Memory("load_a", PipelineKind::Load, 0),
           Memory("store_a", PipelineKind::Store, 8)}),
      Block("memory_b", 2, {1, 0}, {Memory("load_b", PipelineKind::Load, 16)}),
  };
  auto overlay = std::make_unique<Overlay>(config, std::move(blocks));
  FakeMemoryAdapter adapter(1, 3);
  overlay->setMemoryAdapter(&adapter);
  while (!overlay->done() && overlay->now() < 128) {
    adapter.cycle(overlay->now());
    overlay->step();
  }
  bool complete = overlay->done() && adapter.issued() == 3 &&
                  adapter.completed() == 3 && adapter.outstanding() == 0;
  bool overlay_counts = overlay->stats().external_memory_requests == 3 &&
                        overlay->stats().external_memory_completions == 3;
  bool delayed = overlay->stats().external_memory_wait_cycles >= 6;
  bool backpressure = Stalls(*overlay, "memory_queue_full") >= 1;
  return {
      "memory_adapter_backpressure", std::move(overlay),
      {{"adapter_tokens_complete_once", complete,
        "three issued tokens must produce three consumed completions"},
       {"overlay_request_completion_counts", overlay_counts,
        "overlay external request/completion counts must both be three"},
       {"callback_delay_observed", delayed,
        "three-cycle callbacks must accumulate memory wait cycles"},
       {"queue_full_stall_observed", backpressure,
        "capacity-one adapter must reject a concurrent issue"}}};
}

void
Finalize(Scenario *scenario)
{
  if (scenario->id == "event_counted_cross_layer_overlap") {
    scenario->checks[3].pass = Stalls(*scenario->overlay, "event_dependency") >= 1;
  }
}

bool
Pass(const Scenario &scenario)
{
  return std::all_of(
      scenario.checks.begin(), scenario.checks.end(),
      [](const Check &check) { return check.pass; });
}

std::string
Report(const std::vector<Scenario> &scenarios, const std::vector<bool> &deterministic)
{
  bool integrity = true;
  uint64_t assertions = 0;
  for (std::size_t index = 0; index < scenarios.size(); ++index) {
    integrity = integrity && Pass(scenarios[index]) && deterministic[index];
    assertions += scenarios[index].checks.size() + 1;
  }
  std::ostringstream output;
  output << "{\"schema_version\":1,\"audit_integrity\":"
         << (integrity ? "true" : "false")
         << ",\"scenario_count\":" << scenarios.size()
         << ",\"assertion_count\":" << assertions
         << ",\"paper_performance_targets_consumed\":false,\"scenarios\":[";
  for (std::size_t index = 0; index < scenarios.size(); ++index) {
    if (index) output << ",";
    const auto &scenario = scenarios[index];
    output << "{\"id\":" << Quote(scenario.id)
           << ",\"pass\":" << (Pass(scenario) ? "true" : "false")
           << ",\"deterministic_replay\":"
           << (deterministic[index] ? "true" : "false")
           << ",\"summary\":" << scenario.overlay->summaryJson(scenario.id)
           << ",\"assertions\":[";
    for (std::size_t check_index = 0; check_index < scenario.checks.size(); ++check_index) {
      if (check_index) output << ",";
      const auto &check = scenario.checks[check_index];
      output << "{\"name\":" << Quote(check.name)
             << ",\"pass\":" << (check.pass ? "true" : "false")
             << ",\"observation\":" << Quote(check.observation) << "}";
    }
    output << "]}";
  }
  output << "]}";
  return output.str();
}

std::vector<Scenario>
RunScenarios()
{
  std::vector<Scenario> scenarios;
  scenarios.push_back(EventOverlap());
  scenarios.push_back(AdapterBackpressure());
  for (auto &scenario : scenarios) Finalize(&scenario);
  return scenarios;
}

}  // namespace

int
main(int argc, char **argv)
{
  std::string trace_path;
  std::string report_path;
  for (int index = 1; index < argc; ++index) {
    std::string argument = argv[index];
    if ((argument == "--trace" || argument == "--report") && index + 1 < argc) {
      std::string value = argv[++index];
      if (argument == "--trace") trace_path = value;
      if (argument == "--report") report_path = value;
    } else {
      std::cerr << "usage: " << argv[0] << " [--trace PATH] [--report PATH]\n";
      return 2;
    }
  }
  try {
    auto first = RunScenarios();
    auto second = RunScenarios();
    std::vector<bool> deterministic;
    std::ostringstream trace;
    for (std::size_t index = 0; index < first.size(); ++index) {
      std::string first_trace = first[index].overlay->eventsJsonLines(first[index].id);
      std::string second_trace = second[index].overlay->eventsJsonLines(second[index].id);
      deterministic.push_back(
          first_trace == second_trace &&
          first[index].overlay->summaryJson(first[index].id) ==
              second[index].overlay->summaryJson(second[index].id));
      trace << first_trace;
    }
    std::string report = Report(first, deterministic);
    if (!trace_path.empty()) {
      std::ofstream output(trace_path);
      if (!output) throw std::runtime_error("cannot open trace output");
      output << trace.str();
    }
    if (!report_path.empty()) {
      std::ofstream output(report_path);
      if (!output) throw std::runtime_error("cannot open report output");
      output << report << "\n";
    }
    std::cout << report << std::endl;
    return report.find("\"audit_integrity\":true") != std::string::npos ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "MLX CDC/memory driver failed: " << error.what() << std::endl;
    return 1;
  }
}
