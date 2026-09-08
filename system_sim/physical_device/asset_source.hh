#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <jsoncpp/json/json.h>

namespace mlx::physical_device {
// Read-only host-file backed input aperture. The cache below only reduces
// host syscalls: it is not target memory or a simulated hardware cache.
class AssetSource {
  struct File;
  struct Region {
    uint64_t offset,bytes,file_offset,reads=0,read_bytes=0,sequential_bytes=0;
    bool sequential=true;
    std::string name;
    File *file;
  };
  std::map<std::string,std::unique_ptr<File>> files;
  std::vector<Region> regions;
  uint64_t capacity=0,failed_reads=0,rejected_writes=0,cache_fills=0;
  std::array<uint8_t,65536> cache{};
  File *cached_file=nullptr;
  uint64_t cached_offset=0,cached_bytes=0;
  std::string error;
  void read(uint64_t offset,size_t bytes,uint8_t *data);
public:
  explicit AssetSource(const Json::Value &config);
  ~AssetSource();
  bool load(uint64_t offset,size_t bytes,uint8_t *data);
  bool store(uint64_t offset,size_t bytes,const uint8_t *data);
  Json::Value snapshot()const;
};
}
