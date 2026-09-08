#include "physical_mux.h"
#include <stdexcept>

namespace mlx::model_io {
namespace {void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}}
class PhysicalMux::Channel final:public PhysicalMemoryPort {
  PhysicalMux &parent;
  uint64_t id;
public:
  Channel(PhysicalMux &p,uint64_t value):parent(p),id(value){}
  ~Channel(){parent.close(id);}
  void advance(uint64_t cycle)override{parent.advance(cycle);}
  bool request_ready()const override{return parent.ready(id);}
  void submit(const PhysicalRequest &r)override{parent.submit(id,r);}
  std::optional<Response> response()const override{return parent.response(id);}
  void consume_response()override{parent.consume(id);}
};
PhysicalMux::PhysicalMux(PhysicalMemoryPort &p,unsigned count):physical(p),limit(count){check(count>0&&count<=256,"invalid physical mux channel capacity");}
void PhysicalMux::check_channel(uint64_t id)const{
  auto found=entries.find(id);check(!poisoned&&found!=entries.end()&&found->second.open,"unknown, closed or poisoned physical mux channel");
}
std::unique_ptr<PhysicalMemoryPort> PhysicalMux::channel(const std::string &name){
  check(!poisoned&&!name.empty()&&active<limit&&next_id!=UINT64_MAX,"physical mux channel capacity/identity exhausted");
  auto id=next_id++;entries.emplace(id,Entry{name,0,0,0,true});++active;return std::make_unique<Channel>(*this,id);
}
std::optional<Response> PhysicalMux::checked_response()const{
  check(!poisoned,"physical mux is poisoned");auto value=physical.response();
  if(held)check(value&&value->id==held->id&&value->data==held->data&&value->error==held->error,"mux response changed or withdrew under backpressure");
  if(value)check(owner&&value->id==owner->token&&started&&cycle>submitted_at,"mux response has no matching registered owner");
  return value;
}
void PhysicalMux::advance(uint64_t now){
  check(!poisoned&&(!started||now>=cycle),"physical mux clock regressed or failed");
  // External bus responses may be presented before the owner advances this
  // edge. Validate an already-held response first, then use the new clock to
  // validate a newly arrived response (without accepting a same-edge reply).
  if(held)checked_response();
  if(!started||now!=cycle){cycle=now;started=true;physical.advance(now);++advances;}
  auto value=checked_response();
  if(value&&!held){held=value;++entries.at(owner->channel).arrived;}
}
bool PhysicalMux::ready(uint64_t id)const{
  check_channel(id);checked_response();return started&&!owner&&(!released_at||*released_at!=cycle)&&physical.request_ready();
}
void PhysicalMux::submit(uint64_t id,const PhysicalRequest &request){
  check(ready(id),"physical mux has no shared request capacity");
  physical.submit(request);owner=Pending{id,request.id};submitted_at=cycle;++entries.at(id).submitted;
}
std::optional<Response> PhysicalMux::response(uint64_t id)const{
  check_channel(id);auto value=checked_response();
  if(!value||!owner||owner->channel!=id)return std::nullopt;
  check(bool(held),"physical response appeared outside the mux clock step");return value;
}
void PhysicalMux::consume(uint64_t id){
  check(bool(response(id)),"channel cannot consume another or missing mux response");
  physical.consume_response();++entries.at(id).consumed;owner.reset();held.reset();released_at=cycle;
}
void PhysicalMux::close(uint64_t id)noexcept{
  auto found=entries.find(id);
  if(found==entries.end()||!found->second.open){poisoned=true;return;}
  if(owner&&owner->channel==id)poisoned=true;
  found->second.open=false;--active;
}
bool PhysicalMux::idle()const{return !poisoned&&!owner&&!held&&!physical.response();}
Json::Value PhysicalMux::snapshot()const{
  Json::Value result;result["classification"]="single_transaction_physical_response_mux_not_cpu_or_cache_validation";
  result["clock"]=Json::UInt64(cycle);result["advance_calls"]=Json::UInt64(advances);result["idle"]=idle();result["poisoned"]=poisoned;result["active_channels"]=active;
  result["owner_channel"]=owner?Json::Value(Json::UInt64(owner->channel)):Json::Value();result["channels"]=Json::Value(Json::arrayValue);
  for(const auto &[id,e]:entries){Json::Value row;row["id"]=Json::UInt64(id);row["name"]=e.name;row["open"]=e.open;row["submitted"]=Json::UInt64(e.submitted);row["arrived"]=Json::UInt64(e.arrived);row["consumed"]=Json::UInt64(e.consumed);result["channels"].append(row);}
  result["inference_performance_eligible"]=false;return result;
}
} // namespace mlx::model_io
