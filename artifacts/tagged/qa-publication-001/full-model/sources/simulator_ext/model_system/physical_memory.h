#pragma once
#include "buffer_arena.h"
#include "../model_io/queued_physical_port.h"
#include <map>

namespace mlx::model_system {
struct MemoryOptions {
  unsigned latency=8,accept_period=1,nack_every=0,trace_limit=100000;
  static MemoryOptions parse(const Json::Value &value);
};

// One preloaded native address space. Backend tensors are metadata-only;
// numerical accesses occur exclusively here, after accepted physical requests.
// Payload preloading is simulator initialization, NOT modeled CPU/DMA loading.
class PhysicalMemory final:public model_io::PhysicalMemoryPort {
  struct Region {
    uint64_t id,bytes;
    std::weak_ptr<tensor_model::Storage> owner;
    std::shared_ptr<tensor_model::Storage> backing;
    std::vector<uint8_t> written;
    uint64_t unwritten_bytes=0;
  };
  model_storage::Arena arena;
  std::map<uint64_t,Region> regions;
  model_io::QueuedPhysicalPort queue;
  MemoryOptions options;
  std::optional<model_io::PhysicalRequest> pending;
  uint64_t cycle=0,due=0,last_nack=0,reads=0,writes=0,read_bytes=0,write_bytes=0,trace_events=0;
  Json::Value events{Json::arrayValue};
  void event(const char *kind,const model_io::PhysicalRequest &request,uint64_t data=0);
public:
  explicit PhysicalMemory(model_storage::Arena arena,MemoryOptions options={});
  void bind(const tensor_model::Tensor &metadata,const tensor_model::Tensor &backing,bool initialized);
  void require_initialized(const tensor_model::Tensor &metadata)const;
  void collect();
  bool idle()const;
  void advance(uint64_t cycle)override;
  bool request_ready()const override;
  void submit(const model_io::PhysicalRequest &request)override;
  std::optional<model_io::Response> response()const override;
  void consume_response()override;
  Json::Value snapshot()const;
};
} // namespace mlx::model_system
