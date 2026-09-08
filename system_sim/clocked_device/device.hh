#pragma once
#include "matrix_wire.hh"
#include "vector_wire.hh"
#include "memory_wire.hh"
#include "pair_runtime.hh"
#include <memory>
#include <optional>

namespace mlx::clocked_device {
struct Options {
  unsigned address_bits=40;
  uint64_t max_busy_cycles=1000000000000ULL;
  matrix_schedule::Options matrix;
  vector_schedule::Options vector;
  memory_model::ScheduleOptions memory;
};
struct Inputs {
  bool memory_ready=false;
  std::optional<uint64_t> nack;
  std::optional<model_io::Response> response;
};
// One caller-supplied system edge per tick, independent of CPU status polling.
// Descriptor bytes and all tensor data arrive exclusively as bus responses.
// This controller is not yet a Rocket/HellaCache signal adapter.
class Device {
  struct Impl;
  std::unique_ptr<Impl> impl;
public:
  explicit Device(Options options={});
  ~Device();
  void launch(uint64_t descriptor_address,uint64_t descriptor_bytes,uint64_t source_id);
  void tick(const Inputs &inputs={});
  std::optional<model_io::PhysicalRequest> request()const;
  bool busy()const;
  bool complete()const;
  void reset(); // Only after all requests and responses have drained.
  Json::Value report()const; // Pure observation; never advances a clock.
  Json::Value progress()const; // Small scalar state, no kernel/result copying.
};
}
