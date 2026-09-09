#include "c_api.h"
#include "mlx_tagged_simulator.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

struct mlx_tagged_model {
  mlx::tagged::Simulator simulator;
  std::string json, error;
  bool failed = false;
  mlx_tagged_model(mlx::tagged::Program p, mlx::tagged::Timing t)
      : simulator(std::move(p), std::move(t)) {}
};
namespace {
Json::Value parse(const char *text) {
  if (!text) throw std::invalid_argument("missing JSON input");
  Json::CharReaderBuilder reader;
  reader["rejectDupKeys"] = true;
  Json::Value value;
  std::string errors;
  std::istringstream input(text);
  if (!Json::parseFromStream(reader, input, &value, &errors)) throw std::invalid_argument(errors);
  return value;
}
mlx::tagged::Timing timing(const char *text) {
  mlx::tagged::Timing t;
  if (!text) return t;
  const auto values = parse(text);
  if (!values.isObject()) throw std::invalid_argument("timing must be an object");
  for (const auto &key : values.getMemberNames()) {
    const auto &value = values[key];
    if (key == "overlap" || key == "trace") {
      if (!value.isBool()) throw std::invalid_argument("invalid Boolean timing option");
      if (key == "overlap") t.overlap = value.asBool(); else t.trace = value.asBool();
      continue;
    }
    if (!value.isUInt() || !value.asUInt()) throw std::invalid_argument("invalid positive timing option");
    const auto n = value.asUInt();
    if (key == "load_latency") t.latency[0] = n;
    else if (key == "compute_ii") t.compute_ii = n;
    else if (key == "memory_period") t.memory_period = n;
    else if (key == "link_period") t.link_period = n;
    else if (key == "receive_period") t.receive_period = n;
    else if (key == "max_cycles") t.max_cycles = n;
    else throw std::invalid_argument("unknown timing option: " + key);
  }
  t.validate();
  return t;
}
const char *serialize(mlx_tagged_model *model, const Json::Value &value) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  model->json = Json::writeString(writer, value);
  model->error.clear();
  return model->json.c_str();
}
} // namespace

extern "C" mlx_tagged_model *mlx_tagged_create(const char *program_json, const char *timing_json,
                                               char *error_buffer, size_t capacity) {
  if (error_buffer && capacity) error_buffer[0] = '\0';
  try {
    return new mlx_tagged_model(mlx::tagged::Program::parse(parse(program_json)), timing(timing_json));
  } catch (const std::exception &error) {
    if (error_buffer && capacity) {
      const auto length = std::min(capacity - 1, std::strlen(error.what()));
      std::memcpy(error_buffer, error.what(), length); error_buffer[length] = '\0';
    }
    return nullptr;
  }
}
extern "C" int mlx_tagged_tick(mlx_tagged_model *model) {
  if (!model || model->failed) return -1;
  try { const bool done = model->simulator.tick(); model->error.clear(); return done ? 1 : 0; }
  catch (const std::exception &error) { model->error = error.what(); model->failed = true; return -1; }
}
extern "C" uint64_t mlx_tagged_cycles(const mlx_tagged_model *model) {
  return model ? model->simulator.cycles() : 0;
}
extern "C" const char *mlx_tagged_snapshot(mlx_tagged_model *model) {
  if (!model) return nullptr;
  try { return serialize(model, model->simulator.snapshot()); }
  catch (const std::exception &error) { model->error = error.what(); return nullptr; }
}
extern "C" const char *mlx_tagged_result(mlx_tagged_model *model) {
  if (!model || model->failed) return nullptr;
  try { return serialize(model, model->simulator.result()); }
  catch (const std::exception &error) { model->error = error.what(); return nullptr; }
}
extern "C" const char *mlx_tagged_error(const mlx_tagged_model *model) {
  return model ? model->error.c_str() : "null model handle";
}
extern "C" void mlx_tagged_destroy(mlx_tagged_model *model) { delete model; }
