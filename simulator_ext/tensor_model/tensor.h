#pragma once
#include <jsoncpp/json/json.h>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlx::vector_model { struct Stats; }
namespace mlx::memory_model { struct Stats; }
namespace mlx::control_model { struct Stats; }

namespace mlx::tensor_model {
enum class DType { F16, F32, I64, Bool };
using Shape = std::vector<int64_t>;
DType dtype(const std::string &name);
std::string dtype_name(DType type);
unsigned element_bytes(DType type);
Shape shape(const Json::Value &value);
uint64_t elements(const Shape &shape);
Shape strides(const Shape &shape);
double scalar(const Json::Value &value);
void require(bool condition, const std::string &message);

struct Storage {
  std::shared_ptr<void> owner;
  const uint8_t *data = nullptr;
  uint8_t *writable = nullptr;
  uint64_t bytes = 0;
};
struct Tensor {
  DType type = DType::F32;
  Shape sizes, steps;
  int64_t offset = 0;
  std::shared_ptr<Storage> storage;
  uint64_t numel() const { return elements(sizes); }
  bool contiguous() const;
  uint64_t position(uint64_t flat) const;
  float number(uint64_t flat) const;
  int64_t integer(uint64_t flat) const;
  void set_number(uint64_t flat, float value);
  void set_integer(uint64_t flat, int64_t value);
  std::vector<float> floats() const;
  Tensor materialize(DType target) const;
  static Tensor allocate(DType type, Shape sizes);
};

class Assets {
  std::map<std::string, std::shared_ptr<Storage>> files;
public:
  Tensor load(const Json::Value &spec);
};
using Values = std::map<std::string, Tensor>;
struct MatrixInstructionStats;
class Kernels {
  void *blas_handle = nullptr;
  using Gemm = void (*)(int,int,int,int,int,int,float,const float*,int,const float*,int,float,float*,int);
  Gemm gemm = nullptr;
  std::unique_ptr<MatrixInstructionStats> instruction_stats;
  std::unique_ptr<vector_model::Stats> vector_stats;
  std::unique_ptr<memory_model::Stats> memory_stats;
  std::unique_ptr<control_model::Stats> control_stats;
  Json::Value scheduled_options;
  Json::Value vector_scheduled_options;
  Json::Value memory_scheduled_options;
  Json::Value control_scheduled_options;
  Json::Value window_reports{Json::arrayValue};
  Json::Value vector_window_reports{Json::arrayValue};
  Json::Value memory_window_reports{Json::arrayValue};
  Json::Value control_window_reports{Json::arrayValue};
  Tensor matrix(const Json::Value &node, const Values &values, bool linear);
  Tensor vector(const Json::Value &node, const Values &values);
  Tensor memory(const Json::Value &node, const Values &values);
  Tensor control(const Json::Value &node, const Values &values);
public:
  explicit Kernels(const std::string &blas_path, unsigned threads, const Json::Value &schedule_options=Json::Value(), const Json::Value &vector_options=Json::Value(), const Json::Value &memory_options=Json::Value(), const Json::Value &control_options=Json::Value());
  ~Kernels();
  Tensor execute(const Json::Value &node, const Values &values);
  Json::Value matrix_instruction_report() const;
  Json::Value vector_instruction_report() const;
  Json::Value memory_instruction_report() const;
  Json::Value control_instruction_report() const;
  Json::Value matrix_window_report() const { return window_reports; }
  Json::Value vector_window_report() const { return vector_window_reports; }
  Json::Value memory_window_report() const { return memory_window_reports; }
  Json::Value control_window_report() const { return control_window_reports; }
  uint64_t blas_calls = 0;
  uint64_t functional_calls = 0;
  uint64_t matrix_macs = 0;
  uint64_t materialized_bytes = 0;
};
} // namespace mlx::tensor_model
