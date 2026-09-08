#include "tensor_memory_port.h"
#include <cstring>
#include <stdexcept>

namespace mlx::model_io {
namespace {void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}}
TensorMemoryPort::TensorMemoryPort(std::vector<tensor_model::Tensor> r,unsigned delay):regions(std::move(r)),latency(delay){check(latency>0,"memory endpoint requires positive latency");}
void TensorMemoryPort::advance(uint64_t now){
  check(now>=cycle,"memory endpoint clock moved backwards");cycle=now;
  if(!pending||answer||cycle<due)return;
  const auto &q=*pending;auto &t=regions[q.region];Response r{q.id,0,false};
  // Data is sampled/committed by the endpoint when its response becomes
  // available, and held stable until the engine accepts that response.
  if(q.write)std::memcpy(t.storage->writable+q.offset,&q.data,q.bytes);
  else std::memcpy(&r.data,t.storage->data+q.offset,q.bytes);
  answer=r;
}
bool TensorMemoryPort::request_ready()const{return !pending;}
void TensorMemoryPort::submit(const Request &q){
  check(!pending&&!answer&&q.region<regions.size(),"invalid/overcommitted tensor memory request");
  check(q.bytes==1||q.bytes==2||q.bytes==4||q.bytes==8,"unsupported tensor memory access size");
  const auto &t=regions[q.region];check(bool(t.storage),"memory region is not bound");
  check(q.offset<=t.storage->bytes&&q.bytes<=t.storage->bytes-q.offset,"tensor memory access exceeds backing storage");
  check(q.write?t.storage->writable!=nullptr:t.storage->data!=nullptr,"tensor memory region lacks requested access");
  check(cycle<=UINT64_MAX-latency,"memory endpoint deadline overflow");pending=q;due=cycle+latency;
}
std::optional<Response> TensorMemoryPort::response()const{return answer;}
void TensorMemoryPort::consume_response(){check(bool(answer),"no tensor memory response to consume");answer.reset();pending.reset();}
} // namespace mlx::model_io
