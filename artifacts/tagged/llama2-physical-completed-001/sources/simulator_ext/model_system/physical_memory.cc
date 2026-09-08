#include "physical_memory.h"
#include <cstring>
#include <set>

namespace mlx::model_system {
using namespace tensor_model;
MemoryOptions MemoryOptions::parse(const Json::Value &value){
  require(value.isObject(),"physical memory options must be an object");MemoryOptions out;
  std::map<std::string,unsigned*> fields={{"latency",&out.latency},{"accept_period",&out.accept_period},{"nack_every",&out.nack_every},{"trace_limit",&out.trace_limit}};
  for(const auto &name:value.getMemberNames()){require(fields.count(name)&&value[name].isUInt(),"unknown/invalid physical memory option");*fields[name]=value[name].asUInt();}
  require(out.latency&&out.accept_period&&out.trace_limit<=1000000,"physical memory timing/trace bounds invalid");return out;
}
PhysicalMemory::PhysicalMemory(model_storage::Arena a,MemoryOptions o):arena(std::move(a)),options(o){require(options.latency&&options.accept_period&&options.trace_limit<=1000000,"invalid physical memory options");}
bool PhysicalMemory::idle()const{return !pending&&queue.idle();}
void PhysicalMemory::collect(){
  require(idle(),"cannot collect physical buffers while a transaction is outstanding");
  for(auto it=regions.begin();it!=regions.end();)if(it->second.owner.expired())it=regions.erase(it);else ++it;
}
void PhysicalMemory::bind(const Tensor &metadata,const Tensor &backing,bool initialized){
  collect();auto a=arena.allocation(metadata);
  require(backing.storage&&backing.storage->bytes==a.bytes,"physical payload capacity mismatch");
  require(!a.bytes||backing.storage->data,"physical payload has no data");
  require(!a.writable||!a.bytes||backing.storage->writable,"writable physical payload is read-only");
  auto after=regions.lower_bound(a.base);require(after==regions.end()||a.base+a.bytes<=after->first,"physical payload overlaps another live buffer");
  if(after!=regions.begin()){auto before=std::prev(after);require(before->first+before->second.bytes<=a.base,"physical payload overlaps previous buffer");}
  std::vector<uint8_t> written;if(!initialized){require(a.bytes<=UINT64_MAX-7,"physical validity extent overflow");written.resize((a.bytes+7)/8,0);}
  require(regions.emplace(a.base,Region{a.id,a.bytes,metadata.storage,backing.storage,std::move(written),initialized?0:a.bytes}).second,"duplicate physical payload binding");
}
void PhysicalMemory::require_initialized(const Tensor &metadata)const{
  auto a=arena.allocation(metadata);auto it=regions.find(a.base);
  require(it!=regions.end()&&it->second.id==a.id&&it->second.unwritten_bytes==0,"physical output contains bytes never written by a backend");
}
void PhysicalMemory::event(const char *kind,const model_io::PhysicalRequest &r,uint64_t data){
  ++trace_events;if(events.size()>=options.trace_limit)return;
  Json::Value e;e["event"]=kind;e["cycle"]=Json::UInt64(cycle);e["id"]=Json::UInt64(r.id);e["address"]=Json::UInt64(r.address);e["bytes"]=r.bytes;e["write"]=r.write;e["data"]=Json::UInt64(data);events.append(e);
}
void PhysicalMemory::advance(uint64_t now){
  require(now>=cycle,"shared physical clock regressed");cycle=now;queue.advance(now);
  if(pending&&now>=due){
    auto request=*pending;
    if(options.nack_every&&request.id%options.nack_every==0&&request.id!=last_nack){
      last_nack=request.id;queue.receive_nack(request.id);event("nack",request);pending.reset();
    }else{
      auto it=regions.upper_bound(request.address);require(it!=regions.begin(),"physical request has no backing allocation");--it;
      auto &region=it->second;auto owner=region.owner.lock();require(bool(owner),"physical request refers to a retired allocation");
      auto allocation=arena.allocation(owner);require(allocation.id==region.id,"physical payload allocation identity changed");
      auto offset=request.address-it->first;require(offset<=region.bytes&&request.bytes<=region.bytes-offset,"physical request exceeds live allocation");
      model_io::Response response{request.id,0,false};
      if(request.write){
        require(allocation.writable&&region.backing->writable,"physical store to read-only allocation");std::memcpy(region.backing->writable+offset,&request.data,request.bytes);
        if(region.unwritten_bytes)for(uint64_t at=offset;at<offset+request.bytes;++at){auto mask=uint8_t(1u<<(at%8));if(!(region.written[at/8]&mask)){region.written[at/8]|=mask;--region.unwritten_bytes;}}
        ++writes;write_bytes+=request.bytes;
      }else{
        if(region.unwritten_bytes)for(uint64_t at=offset;at<offset+request.bytes;++at)require(region.written[at/8]&(1u<<(at%8)),"physical read before initialization");
        std::memcpy(&response.data,region.backing->data+offset,request.bytes);++reads;read_bytes+=request.bytes;
      }
      event("commit",request,request.write?request.data:response.data);queue.receive_response(response);pending.reset();
    }
  }
  if(queue.request_valid()&&cycle%options.accept_period==0){
    require(!pending,"physical endpoint double acceptance");auto request=*queue.presented_request();queue.accept_request();pending=request;
    require(cycle<=UINT64_MAX-options.latency,"physical memory deadline overflow");due=cycle+options.latency;event("accept",request,request.data);
  }
}
bool PhysicalMemory::request_ready()const{return queue.request_ready();}
void PhysicalMemory::submit(const model_io::PhysicalRequest &r){queue.submit(r);event("submit",r,r.data);}
std::optional<model_io::Response> PhysicalMemory::response()const{return queue.response();}
void PhysicalMemory::consume_response(){queue.consume_response();}
Json::Value PhysicalMemory::snapshot()const{
  auto r=queue.snapshot();r["classification"]="shared_preloaded_native_memory_not_hellacache";r["reads"]=Json::UInt64(reads);r["writes"]=Json::UInt64(writes);r["read_bytes"]=Json::UInt64(read_bytes);r["write_bytes"]=Json::UInt64(write_bytes);
  r["clock"]=Json::UInt64(cycle);r["live_payload_bindings"]=Json::UInt64(regions.size());r["trace"]=events;r["trace_events"]=Json::UInt64(trace_events);r["trace_truncated"]=trace_events>events.size();
  r["options"]["latency"]=options.latency;r["options"]["accept_period"]=options.accept_period;r["options"]["nack_every"]=options.nack_every;r["options"]["trace_limit"]=options.trace_limit;
  r["initialization_mode"]="preloaded_payload_not_timed_loader";return r;
}
} // namespace mlx::model_system
