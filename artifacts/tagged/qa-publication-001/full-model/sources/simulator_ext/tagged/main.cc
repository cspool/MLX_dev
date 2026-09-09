#include "mlx_tagged_simulator.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  try {
    std::string input, output;
    mlx::tagged::Timing timing;
    unsigned repeat = 1;
    for (int arg = 1; arg < argc; ++arg) {
      std::string key = argv[arg];
      if (key == "--serial") { timing.overlap = false; continue; }
      if (key == "--no-trace") { timing.trace = false; continue; }
      if (++arg == argc) throw std::invalid_argument("missing argument for " + key);
      if (key == "--program") input = argv[arg];
      else if (key == "--output") output = argv[arg];
      else {
        size_t end;
        const auto value = std::stoull(argv[arg], &end);
        if (end != std::string(argv[arg]).size() || value == 0 || value > 0xffffffffu)
          throw std::invalid_argument("invalid positive parameter " + key);
        if (key == "--compute-ii") timing.compute_ii = value;
        else if (key == "--load-latency") timing.latency[0] = value;
        else if (key == "--memory-period") timing.memory_period = value;
        else if (key == "--link-period") timing.link_period = value;
        else if (key == "--receive-period") timing.receive_period = value;
        else if (key == "--max-cycles") timing.max_cycles = value;
        else if (key == "--repeat") repeat = value;
        else throw std::invalid_argument("unknown option " + key);
      }
    }
    if (input.empty() || output.empty()) throw std::invalid_argument(
        "usage: mlx-tagged-sim --program FILE --output FILE [--serial] [--no-trace] [--repeat N]");
    std::ifstream stream(input);
    if (!stream) throw std::runtime_error("cannot open program " + input);
    Json::Value json;
    Json::CharReaderBuilder reader;
    std::string errors;
    if (!Json::parseFromStream(reader, stream, &json, &errors)) throw std::invalid_argument(errors);
    const auto program = mlx::tagged::Program::parse(json);
    Json::Value result;
    const auto start = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < repeat; ++i) result = mlx::tagged::simulate(program, timing);
    result["native_elapsed_ns"] = Json::UInt64(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
    result["repeat"] = repeat;
    std::ofstream destination(output);
    if (!destination) throw std::runtime_error("cannot open output " + output);
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    destination << Json::writeString(writer, result) << '\n';
    destination.close();
    if (!destination) throw std::runtime_error("failed to write simulation result");
    std::cout << "MLX_TAGGED_CPP_PASS cycles=" << result["cycles"].asUInt64()
              << " instructions=" << result["counters"]["issue"].asUInt64()
              << " overlap=" << result["counters"]["overlap_pe_cycles"].asUInt64() << '\n';
  } catch (const std::exception &error) {
    std::cerr << "MLX_TAGGED_CPP_ERROR: " << error.what() << '\n';
    return 2;
  }
  return 0;
}
