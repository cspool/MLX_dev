#pragma once
#include "memory_port.h"
#include <vector>

namespace mlx::model_io {
struct PhysicalRequest {
  uint64_t id=0,address=0,data=0;
  unsigned bytes=0;
  bool write=false;
};
class PhysicalMemoryPort {
public:
  virtual ~PhysicalMemoryPort()=default;
  virtual void advance(uint64_t cycle){(void)cycle;}
  virtual bool request_ready()const=0;
  virtual void submit(const PhysicalRequest &request)=0;
  virtual std::optional<Response> response()const=0;
  virtual void consume_response()=0;
};
struct Region {
  uint64_t base=0,bytes=0;
  bool readable=false,writable=false;
};
class RequestTokens {
  uint64_t next=1;
public:
  uint64_t take();
};
// One outstanding transaction per adapter. Share RequestTokens across kernel
// adapters using the same physical transport, so a reused local id cannot
// accept an old kernel's delayed response. Not a cache/CPU timing model.
class AddressSpacePort final:public MemoryPort {
  PhysicalMemoryPort &physical;
  RequestTokens &tokens;
  std::vector<Region> regions;
  uint64_t origin=0,last_cycle=0;
  struct Pending {uint64_t local,physical;};
  std::optional<Pending> pending;
  mutable std::optional<Response> held;
public:
  AddressSpacePort(PhysicalMemoryPort &physical,RequestTokens &tokens,std::vector<Region> regions,uint64_t cycle_origin=0);
  void advance(uint64_t cycle)override;
  bool request_ready()const override;
  void submit(const Request &request)override;
  std::optional<Response> response()const override;
  void consume_response()override;
};
} // namespace mlx::model_io
