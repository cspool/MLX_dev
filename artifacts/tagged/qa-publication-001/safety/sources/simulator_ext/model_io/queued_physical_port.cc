#include "queued_physical_port.h"
#include <stdexcept>

namespace mlx::model_io {
namespace {void check(bool value,const char *message){if(!value)throw std::runtime_error(message);}}
void QueuedPhysicalPort::advance(uint64_t now){check(now>=cycle,"transport clock moved backwards");cycle=now;}
bool QueuedPhysicalPort::request_ready()const{return idle();}
bool QueuedPhysicalPort::idle()const{return !queued&&!inflight&&!answer;}
void QueuedPhysicalPort::submit(const PhysicalRequest &q){
  check(idle(),"physical transport capacity exceeded");check(q.bytes==1||q.bytes==2||q.bytes==4||q.bytes==8,"transport access width unsupported");
  check(q.address%q.bytes==0,"transport request is misaligned");queued=q;queued_at=cycle;++submitted;
}
bool QueuedPhysicalPort::request_valid()const{return queued&&cycle>queued_at;}
std::optional<PhysicalRequest> QueuedPhysicalPort::presented_request()const{return request_valid()?queued:std::nullopt;}
void QueuedPhysicalPort::accept_request(){check(request_valid()&&!inflight&&!answer,"transport accepted a missing/not-yet-visible request");inflight=queued;queued.reset();accepted_at=cycle;++accepted;}
void QueuedPhysicalPort::receive_nack(uint64_t id){
  check(inflight&&inflight->id==id&&!queued&&!answer&&cycle>accepted_at,"transport nack has no matching in-flight request");
  queued=inflight;inflight.reset();queued_at=cycle;++nacks;
}
void QueuedPhysicalPort::receive_response(Response r){
  check(inflight&&inflight->id==r.id&&!queued&&!answer&&cycle>accepted_at,"transport response has no matching in-flight request");
  answer=r;inflight.reset();++responses;
}
std::optional<Response> QueuedPhysicalPort::response()const{return answer;}
void QueuedPhysicalPort::consume_response(){check(bool(answer),"transport consumed a missing response");answer.reset();++consumed;}
void QueuedPhysicalPort::reset(){check(idle(),"transport reset requires drained requests/responses");cycle=queued_at=accepted_at=0;submitted=accepted=nacks=responses=consumed=0;}
Json::Value QueuedPhysicalPort::snapshot()const{
  Json::Value r(Json::objectValue);r["classification"]="registered_memory_transport_not_cache_or_chipyard_validation";
  r["submitted"]=Json::UInt64(submitted);r["accepted"]=Json::UInt64(accepted);r["nacks"]=Json::UInt64(nacks);r["responses"]=Json::UInt64(responses);r["consumed"]=Json::UInt64(consumed);
  r["queued"]=bool(queued);r["inflight"]=bool(inflight);r["response_pending"]=bool(answer);r["idle"]=idle();r["physical_cache_verified"]=false;return r;
}
} // namespace mlx::model_io
