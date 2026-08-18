#include "mlx_overlay.hh"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int
main(int argc, char **argv)
{
  std::string config_path;
  std::string trace_path;
  std::string summary_path;
  uint64_t max_cycles = 100000;
  for (int index = 1; index < argc; ++index) {
    std::string argument = argv[index];
    if ((argument == "--config" || argument == "--trace" ||
         argument == "--summary" || argument == "--max-cycles") &&
        index + 1 < argc) {
      std::string value = argv[++index];
      if (argument == "--config") config_path = value;
      if (argument == "--trace") trace_path = value;
      if (argument == "--summary") summary_path = value;
      if (argument == "--max-cycles") max_cycles = std::stoull(value);
    } else {
      std::cerr << "usage: " << argv[0]
                << " --config PATH [--trace PATH] [--summary PATH]"
                << " [--max-cycles N]\n";
      return 2;
    }
  }
  if (config_path.empty()) {
    std::cerr << "--config is required\n";
    return 2;
  }
  try {
    std::unique_ptr<dsa::sim::mlx::Overlay> overlay =
        dsa::sim::mlx::Overlay::FromJsonFile(config_path);
    overlay->run(max_cycles);
    std::string trace = overlay->eventsJsonLines("dpu_contract");
    std::string summary = overlay->summaryJson("dpu_contract");
    if (!trace_path.empty()) {
      std::ofstream output(trace_path);
      if (!output) throw std::runtime_error("cannot open DPU trace output");
      output << trace;
    }
    if (!summary_path.empty()) {
      std::ofstream output(summary_path);
      if (!output) throw std::runtime_error("cannot open DPU summary output");
      output << summary << "\n";
    }
    std::cout << summary << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "DPU contract driver failed: " << error.what() << std::endl;
    return 1;
  }
}
