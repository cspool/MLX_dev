#pragma once
#include <cstdint>
#include <optional>

namespace mlx::model_io {
// Logical region + byte offset is deliberately separate from a host pointer.
// A system adapter owns region->physical-address binding and access checks.
struct Request {
  uint64_t id=0,offset=0,data=0;
  unsigned region=0,bytes=0;
  bool write=false;
};
struct Response {uint64_t id=0,data=0;bool error=false;};
class MemoryPort {
public:
  virtual ~MemoryPort()=default;
  virtual void advance(uint64_t cycle) { (void)cycle; }
  // submit is an accepted transaction. An RTL/cache adapter should expose
  // its bounded request-buffer capacity here, then drive downstream valid
  // independently of downstream ready; no request may disappear on stall.
  virtual bool request_ready() const=0;
  virtual void submit(const Request &request)=0;
  // Once present, a response must remain stable until consume_response.
  virtual std::optional<Response> response() const=0;
  virtual void consume_response()=0;
};
} // namespace mlx::model_io
