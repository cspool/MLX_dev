#include "../simulator_ext/model_io/physical_mux.h"
#include <iostream>
#include <stdexcept>

using namespace mlx::model_io;
namespace {
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F fn,const char *message){bool rejected=false;try{fn();}catch(const std::runtime_error &e){rejected=std::string(e.what()).find(message)!=std::string::npos;}require(rejected,"expected mux rejection did not occur");}
struct Memory final:PhysicalMemoryPort {
  uint64_t cycle=0,due=0,advances=0;
  std::optional<PhysicalRequest> request;
  std::optional<Response> answer;
  bool error=false;
  void advance(uint64_t now)override{cycle=now;++advances;if(request&&!answer&&now>=due)answer=Response{request->id,0x1122334455667788ULL,error};}
  bool request_ready()const override{return !request;}
  void submit(const PhysicalRequest &q)override{require(request_ready(),"memory busy");request=q;due=cycle+1;}
  std::optional<Response> response()const override{return answer;}
  void consume_response()override{require(bool(answer),"no answer");answer.reset();request.reset();}
};
}
int main(){
  try{
    Memory memory;PhysicalMux mux(memory,2);auto ca=mux.channel("cpu"),cb=mux.channel("pe");RequestTokens tokens;
    std::vector<Region> regions{{0x100000000ULL,16,true,true}};
    AddressSpacePort a(*ca,tokens,regions),b(*cb,tokens,regions);a.advance(0);b.advance(0);require(memory.advances==1,"mux advanced physical memory twice in one edge");
    a.submit(Request{7,0,0,0,8,false});auto first=memory.request->id;require(!b.request_ready(),"mux overbooked shared memory");
    a.advance(1);b.advance(1);require(!b.response(),"non-owner received the CPU response");
    require(a.response()->id==7&&a.response()->data==0x1122334455667788ULL,"CPU response identity/data lost");
    auto before=mux.snapshot();for(unsigned i=0;i<20;++i){a.response();b.response();a.advance(1);b.advance(1);}require(mux.snapshot()==before&&memory.advances==2,"polling changed mux counters or clock");
    rejects([&]{b.consume_response();},"no address-space response");
    memory.answer->data^=1;rejects([&]{b.response();},"backpressure");memory.answer->data^=1;
    auto held=memory.answer;memory.answer.reset();rejects([&]{a.response();},"backpressure");memory.answer=held;
    a.consume_response();require(!b.request_ready(),"mux reused a response slot in its release edge");
    b.advance(2);b.submit(Request{7,8,0,0,8,false});require(memory.request->id!=first,"local token reuse aliased a previous source");
    memory.error=true;b.advance(3);require(!a.response()&&b.response()->error,"error response was dropped or sent to wrong owner");
    b.consume_response();require(mux.idle(),"consumed mux did not drain");
    rejects([&]{a.advance(1);},"clock regressed");rejects([&]{mux.channel("excess");},"capacity");
    auto old_id=mux.snapshot()["channels"][1]["id"];cb.reset();auto cc=mux.channel("later");require(mux.snapshot()["channels"][2]["id"]!=old_id,"closed channel ID reused");
    cc.reset();ca.reset();require(mux.snapshot()["active_channels"].asUInt()==0,"channel count did not drain");

    Memory bad;PhysicalMux bad_mux(bad);auto bad_channel=bad_mux.channel("owner");bad_mux.advance(0);
    bad_channel->submit(PhysicalRequest{91,0x100000000ULL,0,8,false});bad.answer=Response{92,0,false};
    rejects([&]{bad_channel->response();},"matching registered owner");
    bad_channel.reset();require(bad_mux.snapshot()["poisoned"].asBool()&&!bad_mux.idle()&&bad_mux.snapshot()["owner_channel"].isUInt64(),"pending close silently recycled ownership");
    rejects([&]{bad_mux.channel("reuse");},"capacity");
    Memory stray;PhysicalMux stray_mux(stray);auto idle=stray_mux.channel("idle");stray_mux.advance(0);stray.answer=Response{1,0,false};
    rejects([&]{idle->response();},"matching registered owner");stray.answer.reset();
    std::cout<<"PHYSICAL_MUX_CONTRACT_PASS"<<std::endl;
  }catch(const std::exception &e){std::cerr<<"PHYSICAL_MUX_CONTRACT_FAIL: "<<e.what()<<std::endl;return 1;}
}
