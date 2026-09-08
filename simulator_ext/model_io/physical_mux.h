#pragma once
#include "address_space_port.h"
#include <json/json.h>
#include <map>
#include <memory>
#include <string>

namespace mlx::model_io {
// One shared physical transaction, independently named response channels.
// It does not weaken AddressSpacePort's token/address/permission checks.
class PhysicalMux {
  class Channel;
  struct Entry {std::string name;uint64_t submitted=0,arrived=0,consumed=0;bool open=true;};
  PhysicalMemoryPort &physical;
  unsigned limit,active=0;
  uint64_t next_id=1,cycle=0,advances=0,submitted_at=0;
  bool started=false,poisoned=false;
  std::optional<uint64_t> released_at;
  struct Pending {uint64_t channel,token;};
  std::optional<Pending> owner;
  std::optional<Response> held;
  std::map<uint64_t,Entry> entries;
  void check_channel(uint64_t id)const;
  std::optional<Response> checked_response()const;
  bool ready(uint64_t id)const;
  void submit(uint64_t id,const PhysicalRequest &request);
  std::optional<Response> response(uint64_t id)const;
  void consume(uint64_t id);
  void close(uint64_t id)noexcept;
public:
  explicit PhysicalMux(PhysicalMemoryPort &physical,unsigned max_channels=64);
  PhysicalMux(const PhysicalMux &)=delete;
  PhysicalMux &operator=(const PhysicalMux &)=delete;
  std::unique_ptr<PhysicalMemoryPort> channel(const std::string &name);
  void advance(uint64_t cycle);
  bool idle()const;
  Json::Value snapshot()const;
};
} // namespace mlx::model_io
