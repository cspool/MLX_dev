#include "historical_dpu_memory.hh"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int
main(int argc, char **argv)
{
  std::string memory_config;
  std::string summary_path;
  uint64_t max_transfers = 1000000;
  for (int index = 1; index < argc; ++index) {
    std::string argument = argv[index];
    if ((argument == "--memory-config" || argument == "--summary" ||
         argument == "--max-transfers") &&
        index + 1 < argc) {
      std::string value = argv[++index];
      if (argument == "--memory-config") memory_config = value;
      if (argument == "--summary") summary_path = value;
      if (argument == "--max-transfers") max_transfers = std::stoull(value);
    } else {
      std::cerr << "usage: " << argv[0]
                << " --memory-config PATH [--summary PATH]"
                << " [--max-transfers N]\n";
      return 2;
    }
  }
  if (memory_config.empty()) {
    std::cerr << "--memory-config is required\n";
    return 2;
  }
  try {
    auto settings = dsa::sim::mlx::LoadHistoricalDpuMemoryConfig(memory_config);
    dsa::sim::mlx::HistoricalDpuMemoryAdapter memory(settings);
    uint64_t next_tile = 0;
    uint64_t transfers = 0;
    while (!memory.idle()) {
      while (next_tile < settings.tile_count && memory.tileReady(next_tile)) {
        memory.completeReadyTile(next_tile);
        ++next_tile;
      }
      if (memory.idle()) break;
      if (transfers++ >= max_transfers) {
        throw std::runtime_error("DPU full schedule exceeded transfer bound");
      }
      memory.advanceToNextDmaCompletion();
    }
    if (next_tile != settings.tile_count) {
      throw std::runtime_error("DPU full schedule did not release every tile");
    }
    std::string summary =
        "{\"paper_performance_targets_consumed\":false,"
        "\"controller_transfers\":" +
        std::to_string(transfers) +
        ",\"end_to_end_cycles\":" + std::to_string(memory.now() + 1) +
        ",\"memory\":" + memory.summaryJson() + "}";
    if (!summary_path.empty()) {
      std::ofstream output(summary_path);
      if (!output) throw std::runtime_error("cannot open schedule summary");
      output << summary << "\n";
    }
    std::cout << summary << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Historical DPU schedule driver failed: " << error.what()
              << std::endl;
    return 1;
  }
}

