#pragma once
#include <cstdint>
#include <json/json.h>

namespace mlx::model_events {
// Optional control-plane readiness; it never supplies tensor data/results.
class BlockFlow {
public:
  virtual ~BlockFlow()=default;
  virtual bool admission_ready(uint64_t block)const=0;
  virtual bool inputs_ready(uint64_t block)const=0;
  virtual void admitted(uint64_t block,uint64_t lease)=0;
  virtual void completed(uint64_t block,uint64_t lease,uint64_t cycle)=0;
  virtual uint64_t revision()const=0;
  virtual unsigned per_pe_limit()const=0;
  virtual unsigned total_limit()const=0;
  virtual Json::Value description()const=0;
};
}
