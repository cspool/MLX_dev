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
  unsigned standalone_spad_ports = 1;
  std::string standalone_spad_axis = "x";
  uint64_t max_cycles = 1000000;
  for (int index = 1; index < argc; ++index) {
    std::string argument = argv[index];
    if (argument == "--standalone-spad") {
      standalone_spad = true;
    } else if ((argument == "--config" || argument == "--trace" ||
         argument == "--summary" || argument == "--adapter-summary" ||
         argument == "--standalone-spad-ports" ||
         argument == "--standalone-spad-axis" ||
         argument == "--max-cycles") &&
        index + 1 < argc) {
      std::string value = argv[++index];
      if (argument == "--config") config_path = value;
      if (argument == "--trace") trace_path = value;
      if (argument == "--summary") summary_path = value;
      if (argument == "--adapter-summary") adapter_summary_path = value;
      if (argument == "--standalone-spad-ports") {
        standalone_spad_ports = static_cast<unsigned>(std::stoul(value));
        standalone_spad = true;
      }
      if (argument == "--standalone-spad-axis") {
        standalone_spad_axis = value;
      }
      if (argument == "--max-cycles") max_cycles = std::stoull(value);
    } else {
      std::cerr << "usage: " << argv[0]
                << " --config PATH [--trace PATH] [--summary PATH]"
                << " [--standalone-spad] [--adapter-summary PATH]"
                << " [--standalone-spad-ports N] [--standalone-spad-axis x|y]"
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
    std::unique_ptr<dsa::sim::mlx::MultiPortSpadAdapter> multiport_adapter;
    if (overlay->requiresMemoryAdapter()) {
      if (!standalone_spad) {
        throw std::runtime_error(
            "standalone JSON driver requires --standalone-spad for adapter configs");
      }
      if (standalone_spad_ports == 1) {
        adapter.reset(new dsa::sim::mlx::StandaloneSpadAdapter());
        overlay->setMemoryAdapter(adapter.get());
      } else {
        dsa::sim::mlx::MultiPortSpadAdapter::Axis axis;
        if (standalone_spad_axis == "x") {
          axis = dsa::sim::mlx::MultiPortSpadAdapter::Axis::X;
        } else if (standalone_spad_axis == "y") {
          axis = dsa::sim::mlx::MultiPortSpadAdapter::Axis::Y;
        } else {
          throw std::runtime_error("standalone scratchpad axis must be x or y");
        }
        multiport_adapter.reset(
            new dsa::sim::mlx::MultiPortSpadAdapter(standalone_spad_ports, axis));
        overlay->setMemoryAdapter(multiport_adapter.get());
      }
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
      if (!adapter && !multiport_adapter) {
        throw std::runtime_error("--adapter-summary requires --standalone-spad");
      }
      std::ofstream output(adapter_summary_path);
      if (!output) throw std::runtime_error("cannot open adapter summary output");
      output << (adapter ? adapter->summaryJson() : multiport_adapter->summaryJson())
             << "\n";
    }
    std::cout << summary << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MLX JSON driver failed: " << error.what() << std::endl;
    return 1;
  }
}
