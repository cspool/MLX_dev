#include "../simulator_ext/model_io/address_space_port.h"
#include "../simulator_ext/model_io/tensor_memory_port.h"
#include "../simulator_ext/model_io/queued_physical_port.h"
#include <iostream>
#include <stdexcept>

using namespace mlx::model_io;
using namespace mlx::tensor_model;
namespace {
template<class F>void rejects(F function,const char *message){bool failed=false;try{function();}catch(const std::runtime_error &error){failed=std::string(error.what()).find(message)!=std::string::npos;}require(failed,"expected memory contract rejection did not occur");}
struct Physical final:PhysicalMemoryPort{
  bool ready=true;uint64_t cycle=0;std::optional<PhysicalRequest> request;std::optional<Response> answer;
  void advance(uint64_t now)override{cycle=now;}
  bool request_ready()const override{return ready&&!request;}
  void submit(const PhysicalRequest &q)override{require(request_ready(),"physical queue full");request=q;}
  std::optional<Response> response()const override{return answer;}
  void consume_response()override{require(bool(answer),"missing physical response");answer.reset();request.reset();}
};
}
int main(){
  try{
    Physical physical;RequestTokens tokens;
    std::vector<Region> regions{{0x100000000ULL,16,true,false},{0x200000000ULL,16,false,true}};
    AddressSpacePort first(physical,tokens,regions,100);
    first.advance(5);require(physical.cycle==105,"cycle origin not applied");
    physical.ready=false;require(!first.request_ready(),"physical request backpressure was ignored");physical.ready=true;
    first.submit(Request{7,4,0,0,4,false});auto old=physical.request->id;
    require(physical.request->address==0x100000004ULL&&physical.request->bytes==4,"physical address was truncated or changed");
    rejects([&]{first.submit(Request{8,0,0,0,4,false});},"capacity");
    physical.answer=Response{old,0x12345678,false};require(first.response()->id==7&&first.response()->data==0x12345678,"response token was not translated");
    physical.answer->data=5;rejects([&]{first.response();},"backpressure");physical.answer->data=0x12345678;first.consume_response();
    AddressSpacePort second(physical,tokens,regions,200);second.advance(0);
    second.submit(Request{7,0,0xABCD,1,2,true});auto fresh=physical.request->id;
    require(fresh!=old&&physical.request->address==0x200000000ULL&&physical.request->data==0xABCD,"kernel token reuse or write payload failure");
    physical.answer=Response{old,0,false};rejects([&]{second.response();},"owner mismatch");
    physical.answer=Response{fresh,0,true};require(second.response()->id==7&&second.response()->error,"physical error was hidden");second.consume_response();
    rejects([&]{second.submit(Request{9,0,0,1,4,false});},"permission");
    rejects([&]{second.submit(Request{9,0,0,0,4,true});},"permission");
    rejects([&]{second.submit(Request{9,UINT64_MAX,0,0,4,false});},"out of bounds");
    rejects([&]{second.submit(Request{9,1,0,0,4,false});},"misaligned");
    rejects([&]{second.submit(Request{9,0,0,0,3,false});},"access width");
    rejects([&]{AddressSpacePort invalid(physical,tokens,{{UINT64_MAX-2,8,true,false}});},"overflow");
    rejects([&]{AddressSpacePort invalid(physical,tokens,{{100,16,true,false},{104,16,false,true}});},"overlap");
    AddressSpacePort readonly_alias(physical,tokens,{{100,16,true,false},{104,16,true,false}});
    AddressSpacePort bad_clock(physical,tokens,regions,UINT64_MAX);rejects([&]{bad_clock.advance(1);},"clock/origin");

    auto data=Tensor::allocate(DType::I64,{2});data.set_integer(0,4);data.set_integer(1,2);
    TensorMemoryPort memory({data},5);memory.advance(0);memory.submit(Request{1,0,0,0,8,false});memory.advance(4);require(!memory.response(),"memory replied before latency");
    data.set_integer(0,9);memory.advance(5);require(memory.response()->data==9,"memory did not sample the actual backing data");data.set_integer(0,13);memory.advance(6);require(memory.response()->data==9,"held response follows changing backing data");memory.consume_response();
    memory.submit(Request{2,8,37,0,8,true});memory.advance(10);require(data.integer(1)==2,"write committed early");memory.advance(11);require(data.integer(1)==37&&memory.response()->id==2,"write/ack endpoint semantics failed");memory.consume_response();
    rejects([&]{memory.advance(9);},"backwards");rejects([&]{memory.submit(Request{3,UINT64_MAX,0,0,8,false});},"exceeds");
    QueuedPhysicalPort queue;PhysicalRequest q{71,0x80000000,0,4,false};queue.advance(0);queue.submit(q);
    require(!queue.request_valid()&&!queue.request_ready(),"transport skipped its registered request stage");
    rejects([&]{queue.accept_request();},"not-yet-visible");rejects([&]{queue.reset();},"drained");
    queue.advance(1);auto held=queue.presented_request();require(held&&held->id==71,"queued request missing");queue.advance(4);require(queue.presented_request()->address==held->address,"request changed under backpressure");
    queue.accept_request();rejects([&]{queue.receive_response(Response{71,9,false});},"matching");
    queue.advance(5);queue.receive_nack(71);require(!queue.request_valid(),"retry became visible in the nack cycle");
    queue.advance(6);require(queue.presented_request()->id==71,"retry changed its request identity");queue.accept_request();queue.advance(7);
    rejects([&]{queue.receive_response(Response{72,9,false});},"matching");queue.receive_response(Response{71,9,false});
    queue.advance(9);require(queue.response()->data==9&&!queue.request_ready(),"response was not held while stalled");
    rejects([&]{queue.reset();},"drained");queue.consume_response();auto counters=queue.snapshot();
    require(counters["submitted"].asUInt64()==1&&counters["accepted"].asUInt64()==2&&counters["nacks"].asUInt64()==1&&counters["responses"].asUInt64()==1&&counters["consumed"].asUInt64()==1,"transport request/retry/response accounting mismatch");queue.reset();require(queue.idle(),"drained reset failed");
    std::cout<<"MODEL_IO_CONTRACT_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"MODEL_IO_CONTRACT_FAIL: "<<error.what()<<std::endl;return 1;}
}
