#pragma once
#include "adapter.hh"
#include <chrono>
#include <filesystem>

namespace mlx::clocked_rocc {
class Progress {
  std::filesystem::path path;
  Json::Value map;
  uint64_t period,limit,ticks=0,next=0,events=0,emitted=0;
  bool failed=false;
  std::string failure;
  std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
public:
  Progress(std::filesystem::path path,Json::Value map,uint64_t period=1000000,uint64_t limit=50000);
  void sample(const Adapter &adapter,bool launched=false,bool terminal=false,bool final=false);
  Json::Value status()const;
};
}
