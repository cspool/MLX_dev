#include "mapped_memory.hh"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mlx::physical_device {
namespace {void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}}
MappedMemory::MappedMemory(uint64_t bytes):capacity(bytes){check(bytes&&bytes%page_bytes==0,"mapped memory capacity must be page aligned");}
void MappedMemory::bounds(uint64_t offset,uint64_t bytes)const{check(offset<=capacity&&bytes<=capacity-offset,"mapped memory access out of bounds");}
void MappedMemory::read(uint64_t offset,void *destination,uint64_t bytes)const{
  bounds(offset,bytes);check(!bytes||destination,"mapped memory read has no destination");
  // Validate the entire read first: a later uninitialized page must not leave
  // a partially filled descriptor/result in the caller's destination.
  for(uint64_t at=offset,left=bytes;left;){
    auto number=at/page_bytes,index=at%page_bytes,count=std::min(left,page_bytes-index);auto found=pages.find(number);check(found!=pages.end()&&found->second,"uninitialized mapped memory page");
    for(uint64_t i=index;i<index+count;++i)check(found->second->valid[i/8]&(1u<<(i%8)),"uninitialized mapped memory byte");
    at+=count;left-=count;
  }
  auto *out=static_cast<uint8_t*>(destination);
  for(uint64_t at=offset,left=bytes;left;){
    auto index=at%page_bytes,count=std::min(left,page_bytes-index);std::memcpy(out,pages.at(at/page_bytes)->data.data()+index,size_t(count));out+=count;at+=count;left-=count;
  }
}
void MappedMemory::write(uint64_t offset,const void *source,uint64_t bytes,uint64_t generation){
  bounds(offset,bytes);check(!bytes||source,"mapped memory write has no source");
  // Allocate all required backing before mutating any data. Empty pages may
  // remain if the host allocator fails, but no unwritten byte becomes valid.
  for(uint64_t at=offset,left=bytes;left;){
    auto number=at/page_bytes,index=at%page_bytes,count=std::min(left,page_bytes-index);
    auto found=pages.find(number);
    if(found==pages.end())found=pages.emplace(number,std::make_unique<Page>()).first;
    auto &page=found->second;
    if(generation&&!page->writer)page->writer=std::make_unique<std::array<uint64_t,page_bytes>>();
    at+=count;left-=count;
  }
  auto *input=static_cast<const uint8_t*>(source);
  for(uint64_t at=offset,left=bytes;left;){
    auto index=at%page_bytes,count=std::min(left,page_bytes-index);auto &page=*pages.at(at/page_bytes);std::memcpy(page.data.data()+index,input,size_t(count));
    for(uint64_t i=index;i<index+count;++i){page.valid[i/8]|=uint8_t(1u<<(i%8));if(generation)(*page.writer)[i]=generation;}
    at+=count;left-=count;input+=count;
  }
}
void MappedMemory::require_written(uint64_t offset,uint64_t bytes,uint64_t generation)const{
  bounds(offset,bytes);check(generation,"device write generation must be nonzero");
  for(uint64_t at=offset,left=bytes;left;){
    auto index=at%page_bytes,count=std::min(left,page_bytes-index);auto found=pages.find(at/page_bytes);
    check(found!=pages.end()&&found->second&&found->second->writer,"output was not produced by this device launch");
    for(uint64_t i=index;i<index+count;++i)check((*found->second->writer)[i]==generation,"output was not produced by this device launch");
    at+=count;left-=count;
  }
}
Json::Value MappedMemory::snapshot()const{
  uint64_t allocated=0,stamped=0;
  for(const auto &[number,page]:pages){(void)number;if(page){++allocated;stamped+=bool(page->writer);}}
  Json::Value r;r["classification"]="sparse_host_backing_not_hardware_paging";r["capacity_bytes"]=Json::UInt64(capacity);r["page_bytes"]=Json::UInt64(page_bytes);
  r["resident_pages"]=Json::UInt64(allocated);r["writer_pages"]=Json::UInt64(stamped);r["resident_data_bytes"]=Json::UInt64(allocated*page_bytes);r["validity_bytes"]=Json::UInt64(allocated*(page_bytes/8));r["writer_bytes"]=Json::UInt64(stamped*page_bytes*8);return r;
}
} // namespace mlx::physical_device
