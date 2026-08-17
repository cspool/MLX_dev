#include "mlx_overlay.hh"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
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
using dsa::sim::mlx::FunctionalUnit;
using dsa::sim::mlx::Instruction;
using dsa::sim::mlx::Overlay;
using dsa::sim::mlx::PipelineKind;
using dsa::sim::mlx::TaggedBlock;
using dsa::sim::mlx::Timing;

struct Check {
  std::string name;
  bool pass{false};
  std::string observation;
};

struct ScenarioRun {
  std::string id;
  std::unique_ptr<Overlay> overlay;
  std::vector<Check> checks;
};

std::string
Escape(const std::string &value)
{
  std::ostringstream out;
  for (char character : value) {
    if (character == '\\' || character == '\"') {
      out << '\\';
    }
    if (character == '\n') {
      out << "\\n";
    } else {
      out << character;
    }
  }
  return out.str();
}

std::string
Quote(const std::string &value)
{
  return "\"" + Escape(value) + "\"";
}

Config
BaseConfig(unsigned active_window)
{
  Config config;
  config.active_window = active_window;
  config.register_file = {4, 2, 1};
  config.pipeline_timing = {
      {PipelineKind::Load, {3, 1}},
      {PipelineKind::Store, {2, 1}},
      {PipelineKind::Compute, {1, 1}},
      {PipelineKind::Xfer, {1, 1}},
  };
  config.functional_units = {
      {"add", {"alu", {2, 1}}},
      {"mul", {"mul", {3, 1}}},
      {"fma", {"fma", {4, 1}}},
      {"fexp", {"transcendental", {8, 4}}},
  };
  config.routing = {4, 4, {2, 1}, 1, 1};
  return config;
}

Instruction
Inst(
    const std::string &id, PipelineKind pipeline, const std::string &operation,
    std::vector<unsigned> reads = {}, std::vector<unsigned> writes = {},
    Coord destination = {})
{
  Instruction instruction;
  instruction.id = id;
  instruction.pipeline = pipeline;
  instruction.operation = operation;
  instruction.reads = std::move(reads);
  instruction.writes = std::move(writes);
  instruction.destination = destination;
  if (pipeline == PipelineKind::Xfer && !instruction.writes.empty()) {
    instruction.destination_register = instruction.writes.front();
  }
  return instruction;
}

TaggedBlock
Block(
    const std::string &id, uint64_t tag, Coord pe,
    std::vector<Instruction> instructions,
    std::vector<uint64_t> predecessors = {}, uint64_t trip_count = 1)
{
  TaggedBlock block;
  block.id = id;
  block.tag = tag;
  block.pe = pe;
  block.trip_count = trip_count;
  block.predecessors = std::move(predecessors);
  block.instructions = std::move(instructions);
  return block;
}

std::vector<const Event *>
Events(
    const Overlay &overlay, const std::string &kind,
    const std::string &block = "", const std::string &instruction = "")
{
  std::vector<const Event *> result;
  for (const auto &event : overlay.events()) {
    if (event.kind == kind && (block.empty() || event.block == block) &&
        (instruction.empty() || event.instruction == instruction)) {
      result.push_back(&event);
    }
  }
  return result;
}

uint64_t
Stalls(const Overlay &overlay, const std::string &reason)
{
  auto iterator = overlay.stats().stalls_by_reason.find(reason);
  return iterator == overlay.stats().stalls_by_reason.end() ? 0 : iterator->second;
}

ScenarioRun
LowerTagContention()
{
  Config config = BaseConfig(2);
  config.functional_units["add"].timing.latency = 1;
  std::vector<TaggedBlock> blocks;
  blocks.push_back(Block(
      "tag2_mul", 2, {0, 0},
      {Inst("mul", PipelineKind::Compute, "mul")}));
  blocks.push_back(Block(
      "tag1_a", 1, {0, 0},
      {Inst("add", PipelineKind::Compute, "add")}, {}, 2));
  blocks.push_back(Block(
      "tag1_b", 1, {0, 0},
      {Inst("add", PipelineKind::Compute, "add")}, {}, 2));
  auto overlay = std::make_unique<Overlay>(config, std::move(blocks));
  overlay->run(64);
  auto issues = Events(*overlay, "issue");
  std::vector<uint64_t> tags;
  std::vector<std::string> issue_blocks;
  for (const Event *event : issues) {
    tags.push_back(event->tag);
    issue_blocks.push_back(event->block);
  }
  bool order = tags == std::vector<uint64_t>({1, 1, 1, 1, 2});
  bool round_robin = issue_blocks == std::vector<std::string>(
      {"tag1_a", "tag1_b", "tag1_a", "tag1_b", "tag2_mul"});
  return {
      "lower_tag_compute_contention", std::move(overlay),
      {{"lower_tag_issues_first", order,
        "four tag1 iterations must issue before tag2"},
       {"equal_tag_round_robin", round_robin,
        "equal-tag blocks must alternate A/B/A/B"},
       {"contention_is_observed", false, "filled after overlay move"}}};
}

ScenarioRun
FourPipelineOverlap()
{
  std::vector<TaggedBlock> blocks;
  blocks.push_back(Block(
      "load_block", 1, {0, 0},
      {Inst("load", PipelineKind::Load, "load")}));
  blocks.push_back(Block(
      "store_block", 1, {0, 0},
      {Inst("store", PipelineKind::Store, "store")}));
  blocks.push_back(Block(
      "compute_block", 2, {0, 0},
      {Inst("fma", PipelineKind::Compute, "fma")}));
  blocks.push_back(Block(
      "xfer_block", 3, {0, 0},
      {Inst("xfer", PipelineKind::Xfer, "xfer", {}, {0}, {1, 0})}));
  auto overlay = std::make_unique<Overlay>(BaseConfig(3), std::move(blocks));
  overlay->run(64);
  unsigned cycle_zero_issues = 0;
  for (const Event *event : Events(*overlay, "issue")) {
    cycle_zero_issues += event->cycle == 0;
  }
  uint64_t serialized_bound = 3 + 2 + 4 + 2;
  return {
      "four_pipeline_overlap", std::move(overlay),
      {{"all_four_issue_together", cycle_zero_issues == 4,
        "four distinct pipeline issues expected at cycle zero"},
       {"overlap_beats_serial", false,
        "filled after overlay move; serialized bound=" +
            std::to_string(serialized_bound)}}};
}

ScenarioRun
ActiveWindowBound()
{
  std::vector<TaggedBlock> blocks;
  const std::vector<Coord> coordinates =
      {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}};
  for (uint64_t tag = 1; tag <= 5; ++tag) {
    blocks.push_back(Block(
        "tag" + std::to_string(tag), tag, coordinates[tag - 1],
        {Inst("add", PipelineKind::Compute, "add")}));
  }
  auto overlay = std::make_unique<Overlay>(BaseConfig(3), std::move(blocks));
  overlay->run(64);
  bool five_admissions = Events(*overlay, "admit").size() == 5;
  bool bounded = overlay->stats().max_active_tags == 3;
  return {
      "active_window_bound", std::move(overlay),
      {{"all_tags_admitted", five_admissions, "five admission events expected"},
       {"window_never_exceeded", bounded, "maximum active tags expected three"}}};
}

ScenarioRun
RegisterHazards()
{
  std::vector<TaggedBlock> blocks;
  blocks.push_back(Block(
      "bank_compute", 1, {0, 0},
      {Inst("read_r0", PipelineKind::Compute, "add", {0})}));
  blocks.push_back(Block(
      "bank_store", 2, {0, 0},
      {Inst("read_r4", PipelineKind::Store, "store", {4})}));
  blocks.push_back(Block(
      "raw_block", 3, {1, 0},
      {Inst("produce_r1", PipelineKind::Load, "load", {}, {1}),
       Inst("consume_r1", PipelineKind::Compute, "add", {1})}));
  auto overlay = std::make_unique<Overlay>(BaseConfig(3), std::move(blocks));
  overlay->run(64);
  auto producer_complete = Events(*overlay, "complete", "raw_block", "produce_r1");
  auto consumer_issue = Events(*overlay, "issue", "raw_block", "consume_r1");
  auto bank_store_issue = Events(*overlay, "issue", "bank_store");
  bool raw_order = producer_complete.size() == 1 && consumer_issue.size() == 1 &&
                   consumer_issue[0]->cycle >= producer_complete[0]->cycle;
  bool bank_delay = bank_store_issue.size() == 1 && bank_store_issue[0]->cycle == 1 &&
                    Stalls(*overlay, "rf_read_bank") >= 1;
  return {
      "register_raw_and_bank_pressure", std::move(overlay),
      {{"raw_waits_for_writeback", raw_order,
        "consumer issue must not precede producer completion"},
       {"same_bank_cross_pipeline_stalls", bank_delay,
        "tag2 store should issue one cycle later after RF bank conflict"}}};
}

ScenarioRun
FuInitiationInterval()
{
  std::vector<TaggedBlock> blocks;
  blocks.push_back(Block(
      "fexp1", 1, {0, 0},
      {Inst("fexp", PipelineKind::Compute, "fexp")}));
  blocks.push_back(Block(
      "fexp2", 2, {0, 0},
      {Inst("fexp", PipelineKind::Compute, "fexp")}));
  auto overlay = std::make_unique<Overlay>(BaseConfig(2), std::move(blocks));
  overlay->run(64);
  auto first = Events(*overlay, "issue", "fexp1");
  auto second = Events(*overlay, "issue", "fexp2");
  bool interval = first.size() == 1 && second.size() == 1 &&
                  second[0]->cycle - first[0]->cycle >= 4;
  return {
      "fu_initiation_interval", std::move(overlay),
      {{"fexp_ii_is_four", interval, "fexp issue gap must be at least four cycles"},
       {"fu_stall_observed", false, "filled after overlay move"}}};
}

std::vector<int>
RouteSteps(const std::vector<Coord> &route)
{
  std::vector<int> steps;
  for (std::size_t index = 1; index < route.size(); ++index) {
    steps.push_back(
        std::abs(route[index].x - route[index - 1].x) +
        std::abs(route[index].y - route[index - 1].y));
  }
  return steps;
}

ScenarioRun
GreedySkipHop()
{
  Config config = BaseConfig(1);
  bool route1 = RouteSteps(Overlay::GreedyRoute({0, 0}, {1, 0}, config.routing)) ==
                std::vector<int>{1};
  bool route2 = RouteSteps(Overlay::GreedyRoute({0, 0}, {2, 0}, config.routing)) ==
                std::vector<int>{2};
  bool route3 = RouteSteps(Overlay::GreedyRoute({0, 0}, {3, 0}, config.routing)) ==
                (std::vector<int>{2, 1});
  bool signed_xy =
      RouteSteps(Overlay::GreedyRoute({3, 3}, {0, 0}, config.routing)) ==
      (std::vector<int>{2, 1, 2, 1});
  std::vector<TaggedBlock> blocks;
  blocks.push_back(Block(
      "route3", 1, {0, 0},
      {Inst("xfer3", PipelineKind::Xfer, "xfer", {}, {0}, {3, 0})}));
  config.pipeline_timing[PipelineKind::Load] = {1, 1};
  blocks.push_back(Block(
      "route_collision", 2, {2, 0},
      {Inst("delay", PipelineKind::Load, "load"),
       Inst("xfer1", PipelineKind::Xfer, "xfer", {}, {1}, {3, 0})}));
  config.active_window = 2;
  auto overlay = std::make_unique<Overlay>(config, std::move(blocks));
  overlay->run(64);
  bool observed = overlay->stats().route_hops == 3 &&
                  overlay->stats().skip_hops == 1 &&
                  overlay->stats().unit_hops == 2;
  bool contention = overlay->stats().link_stalls == 1 &&
                    Stalls(*overlay, "link_capacity") == 1;
  return {
      "greedy_skip_hop", std::move(overlay),
      {{"distance_one", route1, "distance one route must use [1]"},
       {"distance_two", route2, "distance two route must use [2]"},
       {"distance_three", route3, "distance three route must use [2,1]"},
       {"signed_dimension_order", signed_xy,
        "negative XY route must use [-2,-1,-2,-1] magnitudes"},
       {"runtime_route_matches", observed,
        "runtime packets must record one skip and two unit hops"},
       {"directed_link_contention", contention,
        "two packets sharing link (2,0)->(3,0) must serialize"}}};
}

ScenarioRun
AdjacentLayerDependency()
{
  std::vector<TaggedBlock> blocks;
  blocks.push_back(Block(
      "producer", 1, {0, 0},
      {Inst("add", PipelineKind::Compute, "add")}));
  blocks.push_back(Block(
      "successor", 2, {1, 0},
      {Inst("mul", PipelineKind::Compute, "mul")}, {1}));
  auto overlay = std::make_unique<Overlay>(BaseConfig(2), std::move(blocks));
  overlay->run(64);
  auto producer_done = Events(*overlay, "tag_complete");
  auto successor_admit = Events(*overlay, "admit");
  uint64_t done_cycle = 0;
  uint64_t admit_cycle = 0;
  bool found_done = false;
  bool found_admit = false;
  for (const Event *event : producer_done) {
    if (event->tag == 1) {
      found_done = true;
      done_cycle = event->cycle;
    }
  }
  for (const Event *event : successor_admit) {
    if (event->tag == 2) {
      found_admit = true;
      admit_cycle = event->cycle;
    }
  }
  bool ordered = found_done && found_admit && admit_cycle >= done_cycle;
  return {
      "adjacent_layer_dependency", std::move(overlay),
      {{"successor_waits_for_predecessor", ordered,
        "tag2 admission cannot precede tag1 completion"}}};
}

using ScenarioFactory = ScenarioRun (*)();

std::vector<std::pair<std::string, ScenarioFactory>>
Factories()
{
  return {
      {"lower_tag_compute_contention", LowerTagContention},
      {"four_pipeline_overlap", FourPipelineOverlap},
      {"active_window_bound", ActiveWindowBound},
      {"register_raw_and_bank_pressure", RegisterHazards},
      {"fu_initiation_interval", FuInitiationInterval},
      {"greedy_skip_hop", GreedySkipHop},
      {"adjacent_layer_dependency", AdjacentLayerDependency},
  };
}

void
FinalizeChecks(ScenarioRun *run)
{
  if (run->id == "lower_tag_compute_contention") {
    run->checks[2].pass = Stalls(*run->overlay, "pipeline_contention") >= 1;
  } else if (run->id == "four_pipeline_overlap") {
    run->checks[1].pass = run->overlay->stats().cycles < 11;
  } else if (run->id == "fu_initiation_interval") {
    run->checks[1].pass = Stalls(*run->overlay, "fu_initiation") >= 1;
  }
}

bool
ScenarioPass(const ScenarioRun &run)
{
  return std::all_of(run.checks.begin(), run.checks.end(),
                     [](const Check &check) { return check.pass; });
}

std::string
ReportJson(
    const std::vector<ScenarioRun> &runs,
    const std::vector<bool> &deterministic)
{
  bool integrity = true;
  uint64_t assertion_count = 0;
  for (std::size_t index = 0; index < runs.size(); ++index) {
    integrity = integrity && ScenarioPass(runs[index]) && deterministic[index];
    assertion_count += runs[index].checks.size() + 1;
  }
  std::ostringstream out;
  out << "{\"schema_version\":1,\"audit_integrity\":"
      << (integrity ? "true" : "false")
      << ",\"scenario_count\":" << runs.size()
      << ",\"assertion_count\":" << assertion_count
      << ",\"paper_target_values_consumed\":false,\"scenarios\":[";
  for (std::size_t index = 0; index < runs.size(); ++index) {
    if (index) out << ",";
    const auto &run = runs[index];
    out << "{\"id\":" << Quote(run.id)
        << ",\"pass\":" << (ScenarioPass(run) ? "true" : "false")
        << ",\"deterministic_replay\":"
        << (deterministic[index] ? "true" : "false")
        << ",\"summary\":" << run.overlay->summaryJson(run.id)
        << ",\"assertions\":[";
    for (std::size_t check_index = 0; check_index < run.checks.size(); ++check_index) {
      if (check_index) out << ",";
      const auto &check = run.checks[check_index];
      out << "{\"name\":" << Quote(check.name)
          << ",\"pass\":" << (check.pass ? "true" : "false")
          << ",\"observation\":" << Quote(check.observation) << "}";
    }
    out << "]}";
  }
  out << "]}";
  return out.str();
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
      if (argument == "--trace") {
        trace_path = value;
      } else {
        report_path = value;
      }
    } else {
      std::cerr << "usage: " << argv[0]
                << " [--trace PATH] [--report PATH]" << std::endl;
      return 2;
    }
  }

  try {
    std::vector<ScenarioRun> runs;
    std::vector<bool> deterministic;
    std::ostringstream traces;
    for (const auto &factory : Factories()) {
      ScenarioRun first = factory.second();
      ScenarioRun second = factory.second();
      FinalizeChecks(&first);
      FinalizeChecks(&second);
      std::string first_trace = first.overlay->eventsJsonLines(first.id);
      std::string second_trace = second.overlay->eventsJsonLines(second.id);
      bool same = first_trace == second_trace &&
                  first.overlay->summaryJson(first.id) ==
                      second.overlay->summaryJson(second.id);
      deterministic.push_back(same);
      traces << first_trace;
      runs.push_back(std::move(first));
    }
    std::string report = ReportJson(runs, deterministic);
    if (!trace_path.empty()) {
      std::ofstream trace_output(trace_path);
      if (!trace_output) throw std::runtime_error("cannot open trace output");
      trace_output << traces.str();
    }
    if (!report_path.empty()) {
      std::ofstream report_output(report_path);
      if (!report_output) throw std::runtime_error("cannot open report output");
      report_output << report << "\n";
    }
    std::cout << report << std::endl;
    return report.find("\"audit_integrity\":true") != std::string::npos ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "MLX overlay driver failed: " << error.what() << std::endl;
    return 1;
  }
}
