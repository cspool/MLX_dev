#include "buffer_arena.h"
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

namespace mlx::model_storage {
using namespace tensor_model;
namespace {void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}}

struct Arena::State {
  uint64_t base,bytes,alignment,next_id=1,live_bytes=0,reserved_bytes=0,peak_reserved=0,total_allocations=0,total_frees=0;
  struct Record {Allocation allocation;uint64_t pins=0;const Storage *storage=nullptr;};
  std::map<uint64_t,uint64_t> free;
  std::map<uint64_t,Record> live;
  std::map<const Storage*,uint64_t> identities;

  State(uint64_t b,uint64_t size,uint64_t a):base(b),bytes(size),alignment(a){free.emplace(base,bytes);}
  uint64_t reserve(uint64_t logical){
    check(logical<=UINT64_MAX-(alignment-1),"buffer alignment size overflow");
    // Empty tensors still get a distinct minimum address slot until freed.
    uint64_t needed=std::max(alignment,(logical+alignment-1)&~(alignment-1));
    check(needed<=bytes,"buffer exceeds physical arena capacity");return needed;
  }
  Record &find_storage(const std::shared_ptr<Storage> &storage){
    check(bool(storage)&&storage->data==nullptr&&storage->writable==nullptr,"arena expects virtual tensor metadata");
    auto index=identities.find(storage.get());check(index!=identities.end(),"tensor storage does not belong to this arena");
    auto entry=live.find(index->second);check(entry!=live.end(),"stale physical allocation identity");
    check(storage->bytes==entry->second.allocation.bytes,"physical allocation extent was changed");
    return entry->second;
  }
  Record &find(const Tensor &tensor){
    auto &entry=find_storage(tensor.storage);
    check(tensor.offset>=0&&tensor.sizes.size()==tensor.steps.size(),"invalid physical view metadata");
    for(auto stride:tensor.steps)check(stride>=0,"negative physical view stride");
    if(tensor.numel())tensor.position(tensor.numel()-1);
    return entry;
  }
  void release(uint64_t id)noexcept{
    // The only caller is the final shared Storage owner. Pins retain that
    // Storage, so a nonzero pin count here indicates an internal lifetime bug.
    auto entry=live.find(id);if(entry==live.end()||entry->second.pins)std::terminate();
    auto a=entry->second.allocation;identities.erase(entry->second.storage);live.erase(entry);
    live_bytes-=a.bytes;reserved_bytes-=a.reserved_bytes;++total_frees;
    auto next=free.lower_bound(a.base);uint64_t start=a.base,size=a.reserved_bytes;
    if(next!=free.begin()){
      auto previous=std::prev(next);
      if(previous->first+previous->second==start){start=previous->first;size+=previous->second;free.erase(previous);}
    }
    next=free.lower_bound(start);
    if(next!=free.end()&&start+size==next->first){size+=next->second;free.erase(next);}
    free.emplace(start,size);
  }
  struct Owner {
    std::shared_ptr<State> state;
    uint64_t id=0;
    explicit Owner(std::shared_ptr<State> s):state(std::move(s)){}
    ~Owner(){if(id)state->release(id);}
  };
};

struct Arena::Pin::Held {
  std::shared_ptr<State> state;
  Tensor value;
  Allocation info;
  bool write=false;
  Held(std::shared_ptr<State> s,Tensor t,Allocation a,bool w):state(std::move(s)),value(std::move(t)),info(a),write(w){}
  ~Held(){
    auto entry=state->live.find(info.id);if(entry==state->live.end()||!entry->second.pins)std::terminate();
    --entry->second.pins;
    // value is destroyed after the body; only then may the final owner free.
  }
};

Arena::Pin::Pin(std::shared_ptr<Held> h):held(std::move(h)){}
model_io::Region Arena::Pin::region()const{check(bool(held),"empty physical buffer pin");return {held->info.base,held->info.bytes,!held->write,held->write};}
const Tensor &Arena::Pin::tensor()const{check(bool(held),"empty physical buffer pin");return held->value;}
Allocation Arena::Pin::allocation()const{check(bool(held),"empty physical buffer pin");return held->info;}

Arena::Arena(uint64_t base,uint64_t bytes,uint64_t alignment){
  check(alignment>=8&&!(alignment&(alignment-1)),"physical arena alignment must be a power of two of at least 8 bytes");
  check(base%alignment==0&&bytes>=alignment&&bytes%alignment==0,"physical arena extent/base is not aligned");
  check(bytes<=UINT64_MAX-base,"physical arena address overflow");state=std::make_shared<State>(base,bytes,alignment);
}
Tensor Arena::allocate(DType type,Shape shape,bool writable){
  const auto count=elements(shape);const auto width=element_bytes(type);
  check(count<=UINT64_MAX/width,"tensor byte extent overflow");uint64_t bytes=count*width,reserved=state->reserve(bytes);
  check(state->next_id!=0,"physical allocation identity exhausted");
  auto selected=state->free.end();for(auto it=state->free.begin();it!=state->free.end();++it)if(it->second>=reserved){selected=it;break;}
  check(selected!=state->free.end(),"physical arena exhausted or fragmented");
  auto base=selected->first,available=selected->second;
  Tensor result;result.type=type;result.sizes=std::move(shape);result.steps=strides(result.sizes);result.storage=std::make_shared<Storage>();result.storage->bytes=bytes;
  const uint64_t id=state->next_id;
  auto owner=std::make_shared<State::Owner>(state);
  bool tail_added=false,record_added=false,identity_added=false;
  try{
    if(available>reserved){tail_added=state->free.emplace(base+reserved,available-reserved).second;check(tail_added,"physical free-range collision");}
    record_added=state->live.emplace(id,State::Record{Allocation{id,base,bytes,reserved,writable},0,result.storage.get()}).second;check(record_added,"physical allocation identity collision");
    identity_added=state->identities.emplace(result.storage.get(),id).second;check(identity_added,"physical storage identity collision");
  }catch(...){
    if(identity_added)state->identities.erase(result.storage.get());
    if(record_added)state->live.erase(id);
    if(tail_added)state->free.erase(base+reserved);
    throw;
  }
  state->free.erase(selected);++state->next_id;
  state->live_bytes+=bytes;state->reserved_bytes+=reserved;state->peak_reserved=std::max(state->peak_reserved,state->reserved_bytes);++state->total_allocations;
  owner->id=id;result.storage->owner=std::move(owner);
  return result;
}
Arena::Pin Arena::pin(const Tensor &tensor,bool write)const{
  auto &record=state->find(tensor);check(!write||record.allocation.writable,"physical buffer is read-only");
  check(record.pins!=UINT64_MAX,"physical pin count overflow");
  auto held=std::make_shared<Pin::Held>(state,tensor,record.allocation,write);++record.pins;return Pin(std::move(held));
}
Allocation Arena::allocation(const Tensor &tensor)const{return state->find(tensor).allocation;}
Allocation Arena::allocation(const std::shared_ptr<Storage> &storage)const{return state->find_storage(storage).allocation;}
void Arena::seal_read_only(const Tensor &tensor){
  auto &record=state->find(tensor);check(!record.pins,"cannot change physical permissions with outstanding pins");record.allocation.writable=false;
}
Json::Value Arena::snapshot()const{
  Json::Value r;r["classification"]="physical_address_ownership_not_memory_initialization_or_system_validation";
  r["base"]=Json::UInt64(state->base);r["bytes"]=Json::UInt64(state->bytes);r["alignment"]=Json::UInt64(state->alignment);
  r["live_bytes"]=Json::UInt64(state->live_bytes);r["reserved_bytes"]=Json::UInt64(state->reserved_bytes);r["peak_reserved_bytes"]=Json::UInt64(state->peak_reserved);
  r["total_allocations"]=Json::UInt64(state->total_allocations);r["total_frees"]=Json::UInt64(state->total_frees);r["allocations"]=Json::Value(Json::arrayValue);r["free_ranges"]=Json::Value(Json::arrayValue);
  for(const auto &[id,record]:state->live){const auto &a=record.allocation;Json::Value row;row["id"]=Json::UInt64(id);row["base"]=Json::UInt64(a.base);row["bytes"]=Json::UInt64(a.bytes);row["reserved_bytes"]=Json::UInt64(a.reserved_bytes);row["pins"]=Json::UInt64(record.pins);row["writable"]=a.writable;r["allocations"].append(row);}
  uint64_t free_bytes=0;for(const auto &[base,bytes]:state->free){Json::Value row;row["base"]=Json::UInt64(base);row["bytes"]=Json::UInt64(bytes);r["free_ranges"].append(row);free_bytes+=bytes;}
  check(free_bytes+state->reserved_bytes==state->bytes,"physical arena byte conservation failed");r["free_bytes"]=Json::UInt64(free_bytes);
  r["inference_performance_eligible"]=false;r["mlx_system_verified"]=false;return r;
}
} // namespace mlx::model_storage
