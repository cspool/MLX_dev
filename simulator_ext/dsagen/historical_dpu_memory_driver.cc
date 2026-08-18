#include "historical_dpu_memory.hh"
#include "mlx_overlay.hh"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void
Write(const std::string &path, const std::string &content)
{
  if (path.empty()) return;
  std::ofstream output(path);
  if (!output) throw std::runtime_error("cannot open output: " + path);
  output << content;
}

}  // namespace

int
main(int argc, char **argv)
{
  std::string overlay_config;
  std::string memory_config;
  std::string summary_path;
  std::string overlay_trace_path;
  std::string memory_trace_path;
  uint64_t max_cycles = 100000;
  for (int index = 1; index < argc; ++index) {
    std::string argument = argv[index];
    if ((argument == "--overlay-config" || argument == "--memory-config" ||
         argument == "--summary" || argument == "--overlay-trace" ||
         argument == "--memory-trace" || argument == "--max-cycles") &&
        index + 1 < argc) {
      std::string value = argv[++index];
      if (argument == "--overlay-config") overlay_config = value;
      if (argument == "--memory-config") memory_config = value;
      if (argument == "--summary") summary_path = value;
      if (argument == "--overlay-trace") overlay_trace_path = value;
      if (argument == "--memory-trace") memory_trace_path = value;
      if (argument == "--max-cycles") max_cycles = std::stoull(value);
    } else {
      std::cerr << "usage: " << argv[0]
                << " --overlay-config PATH --memory-config PATH"
                << " [--summary PATH] [--overlay-trace PATH]"
                << " [--memory-trace PATH] [--max-cycles N]\n";
      return 2;
    }
  }
  if (overlay_config.empty() || memory_config.empty()) {
    std::cerr << "--overlay-config and --memory-config are required\n";
    return 2;
  }
  try {
    auto memory_settings =
        dsa::sim::mlx::LoadHistoricalDpuMemoryConfig(memory_config);
    dsa::sim::mlx::HistoricalDpuMemoryAdapter memory(memory_settings);
    std::unique_ptr<dsa::sim::mlx::Overlay> overlay =
        dsa::sim::mlx::Overlay::FromJsonFile(overlay_config);
    overlay->setMemoryAdapter(&memory);
    while (!overlay->done() && overlay->now() < max_cycles) {
      overlay->step();
    }
    if (!overlay->done()) {
      throw std::runtime_error("DPU overlay exceeded max cycles");
    }
    uint64_t end_to_end_cycles = overlay->now();
    while (!memory.idle() && end_to_end_cycles < max_cycles) {
      memory.advance(end_to_end_cycles);
      ++end_to_end_cycles;
    }
    if (!memory.idle()) {
      throw std::runtime_error("DPU memory drain exceeded max cycles");
    }
    std::string overlay_trace = overlay->eventsJsonLines("historical_dpu_memory");
    std::string memory_trace = memory.eventsJsonLines();
    std::string summary =
        "{\"paper_performance_targets_consumed\":false,"
        "\"overlay_cycles\":" +
        std::to_string(overlay->now()) +
        ",\"end_to_end_cycles\":" + std::to_string(end_to_end_cycles) +
        ",\"overlay\":" +
        overlay->summaryJson("historical_dpu_memory") +
        ",\"memory\":" + memory.summaryJson() + "}";
    Write(summary_path, summary + "\n");
    Write(overlay_trace_path, overlay_trace);
    Write(memory_trace_path, memory_trace);
    std::cout << summary << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Historical DPU memory driver failed: " << error.what()
              << std::endl;
    return 1;
  }
}

