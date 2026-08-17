#include "mlx_overlay.hh"
#include "standalone_spad_adapter.hh"

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
  std::string adapter_summary_path;
  bool standalone_spad = false;
  uint64_t max_cycles = 1000000;
  for (int index = 1; index < argc; ++index) {
    std::string argument = argv[index];
    if (argument == "--standalone-spad") {
      standalone_spad = true;
    } else if ((argument == "--config" || argument == "--trace" ||
         argument == "--summary" || argument == "--adapter-summary" ||
         argument == "--max-cycles") &&
        index + 1 < argc) {
      std::string value = argv[++index];
      if (argument == "--config") config_path = value;
      if (argument == "--trace") trace_path = value;
      if (argument == "--summary") summary_path = value;
      if (argument == "--adapter-summary") adapter_summary_path = value;
      if (argument == "--max-cycles") max_cycles = std::stoull(value);
    } else {
      std::cerr << "usage: " << argv[0]
                << " --config PATH [--trace PATH] [--summary PATH]"
                << " [--standalone-spad] [--adapter-summary PATH]"
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
    std::unique_ptr<dsa::sim::mlx::StandaloneSpadAdapter> adapter;
    if (overlay->requiresMemoryAdapter()) {
      if (!standalone_spad) {
        throw std::runtime_error(
            "standalone JSON driver requires --standalone-spad for adapter configs");
      }
      adapter.reset(new dsa::sim::mlx::StandaloneSpadAdapter());
      overlay->setMemoryAdapter(adapter.get());
    }
    overlay->run(max_cycles);
    std::string trace = overlay->eventsJsonLines("json_driver");
    std::string summary = overlay->summaryJson("json_driver");
    if (!trace_path.empty()) {
      std::ofstream output(trace_path);
      if (!output) throw std::runtime_error("cannot open trace output");
      output << trace;
    }
    if (!summary_path.empty()) {
      std::ofstream output(summary_path);
      if (!output) throw std::runtime_error("cannot open summary output");
      output << summary << "\n";
    }
    if (!adapter_summary_path.empty()) {
      if (!adapter) {
        throw std::runtime_error("--adapter-summary requires --standalone-spad");
      }
      std::ofstream output(adapter_summary_path);
      if (!output) throw std::runtime_error("cannot open adapter summary output");
      output << adapter->summaryJson() << "\n";
    }
    std::cout << summary << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MLX JSON driver failed: " << error.what() << std::endl;
    return 1;
  }
}
