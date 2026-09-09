#include "address_space_port.h"
#include <stdexcept>

namespace mlx::model_io {
namespace {void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}}
uint64_t RequestTokens::take(){check(next!=0,"physical request token space exhausted");return next++;}
AddressSpacePort::AddressSpacePort(PhysicalMemoryPort &p,RequestTokens &t,std::vector<Region> r,uint64_t base_cycle)
    :physical(p),tokens(t),regions(std::move(r)),origin(base_cycle){
  check(!regions.empty()&&regions.size()<=64,"region table exceeds registered capacity");
  for(size_t i=0;i<regions.size();++i){const auto &a=regions[i];check(a.bytes<=UINT64_MAX-a.base,"physical region address overflow");
    for(size_t j=0;j<i;++j){const auto &b=regions[j];bool overlap=a.bytes&&b.bytes&&a.base<b.base+b.bytes&&b.base<a.base+a.bytes;
      check(!overlap||(!a.writable&&!b.writable),"writable physical regions overlap");}}
}
void AddressSpacePort::advance(uint64_t cycle){check(cycle>=last_cycle&&cycle<=UINT64_MAX-origin,"invalid address-space clock/origin");last_cycle=cycle;physical.advance(origin+cycle);}
bool AddressSpacePort::request_ready()const{return !pending&&physical.request_ready();}
void AddressSpacePort::submit(const Request &request){
  check(request_ready(),"address-space port has no request capacity");
  check(request.region<regions.size(),"unknown logical memory region");
  check(request.bytes==1||request.bytes==2||request.bytes==4||request.bytes==8,"unregistered physical access width");
  const auto &region=regions[request.region];check(request.write?region.writable:region.readable,"memory region permission violation");
  check(request.offset<=region.bytes&&request.bytes<=region.bytes-request.offset,"logical memory access out of bounds");
  check(request.offset<=UINT64_MAX-region.base,"physical address addition overflow");uint64_t address=region.base+request.offset;
  check(address%request.bytes==0,"misaligned physical memory request");
  const uint64_t id=tokens.take();physical.submit(PhysicalRequest{id,address,request.data,request.bytes,request.write});pending=Pending{request.id,id};
}
std::optional<Response> AddressSpacePort::response()const{
  auto value=physical.response();
  if(held)check(value&&value->id==held->id&&value->data==held->data&&value->error==held->error,"physical response changed or withdrew under backpressure");
  if(!value)return std::nullopt;
  check(pending&&value->id==pending->physical,"physical response owner mismatch");held=value;
  return Response{pending->local,value->data,value->error};
}
void AddressSpacePort::consume_response(){check(bool(response()),"no address-space response to consume");physical.consume_response();pending.reset();held.reset();}
} // namespace mlx::model_io
