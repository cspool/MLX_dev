#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <jsoncpp/json/json.h>

namespace mlx::physical_device {
/* Sparse host representation of a configured device address space. This is
 * not a hardware cache, capacity increase, paging policy or DRAM timing model. */
class MappedMemory {
  static constexpr uint64_t page_bytes=4096;
  struct Page {
    std::array<uint8_t,page_bytes> data{};
    std::array<uint8_t,page_bytes/8> valid{};
    std::unique_ptr<std::array<uint64_t,page_bytes>> writer;
  };
  uint64_t capacity;
  std::map<uint64_t,std::unique_ptr<Page>> pages;
  void bounds(uint64_t offset,uint64_t bytes)const;
public:
  explicit MappedMemory(uint64_t bytes);
  uint64_t size()const{return capacity;}
  void read(uint64_t offset,void *destination,uint64_t bytes)const;
  void write(uint64_t offset,const void *source,uint64_t bytes,uint64_t generation=0);
  void require_written(uint64_t offset,uint64_t bytes,uint64_t generation)const;
  Json::Value snapshot()const;
};
} // namespace mlx::physical_device
