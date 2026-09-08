#pragma once
#include "address_space_port.h"
#include <jsoncpp/json/json.h>

namespace mlx::model_io {
// Registered, one-transaction transport bridge. A system/cache driver consumes
// presented_request and calls accept_request, receive_nack, or receive_response.
// This class never reads or writes memory and does not implement a cache.
class QueuedPhysicalPort final:public PhysicalMemoryPort {
  uint64_t cycle=0,queued_at=0,accepted_at=0;
  std::optional<PhysicalRequest> queued,inflight;
  std::optional<Response> answer;
  uint64_t submitted=0,accepted=0,nacks=0,responses=0,consumed=0;
public:
  void advance(uint64_t cycle)override;
  bool request_ready()const override;
  void submit(const PhysicalRequest &request)override;
  std::optional<Response> response()const override;
  void consume_response()override;
  bool request_valid()const;
  std::optional<PhysicalRequest> presented_request()const;
  void accept_request();
  void receive_nack(uint64_t request_id);
  void receive_response(Response response);
  bool idle()const;
  void reset();
  Json::Value snapshot()const;
};
} // namespace mlx::model_io
