#include "../simulator_ext/model_events/completion_window.h"
#include <iostream>
#include <stdexcept>

using namespace mlx::model_events;
namespace {
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F fn,const char *text){bool failed=false;try{fn();}catch(const std::runtime_error &e){failed=std::string(e.what()).find(text)!=std::string::npos;}require(failed,"expected event rejection did not occur");}
}
int main(){
  try{
    CompletionWindow ordered(7,300,1);ordered.advance(0);
    for(uint64_t block=0;block<300;++block){require(ordered.can_admit(7,block),"single-record window lost forward progress");ordered.admit(7,block,block+1);ordered.complete(7,block,block+1,block);
      require(!ordered.ready(7,block),"completion woke a consumer on the same edge");ordered.advance(block+1);require(ordered.ready(7,block),"completed prefix forgot a producer");}
    require(ordered.finished()&&ordered.snapshot()["peak_event_slots"].asUInt()==1,"event history required unbounded storage");
    rejects([&]{ordered.complete(6,0,1,300);},"identity mismatch");
    CompletionWindow out_of_order(8,5,3);out_of_order.advance(0);
    for(uint64_t block=0;block<3;++block)out_of_order.admit(8,block,100+block);
    require(!out_of_order.can_admit(8,3),"event admission exceeded finite window");
    out_of_order.complete(8,2,102,0);rejects([&]{out_of_order.complete(8,2,102,0);},"duplicate");out_of_order.advance(1);
    require(out_of_order.ready(8,2)&&!out_of_order.ready(8,0),"out-of-order event matched the wrong block");
    out_of_order.complete(8,0,100,1);out_of_order.advance(2);out_of_order.admit(8,3,103);
    rejects([&]{out_of_order.complete(8,0,100,2);},"stale");
    rejects([&]{out_of_order.complete(8,3,999,2);},"stale");
    rejects([&]{out_of_order.complete(8,4,104,2);},"unadmitted");
    out_of_order.complete(8,1,101,2);out_of_order.complete(8,3,103,2);out_of_order.advance(3);out_of_order.admit(8,4,104);out_of_order.complete(8,4,104,3);out_of_order.advance(4);
    require(out_of_order.finished(),"event prefix did not drain");
    Json::Value spec;spec["kind"]="matrix";spec["elements"]=95;spec["m"]=5;spec["n"]=19;spec["batches"]=1;auto mapping=Mapping::parse(spec);
    require(mapping.producer_blocks()==6&&mapping.dependencies(1)==std::vector<uint64_t>({0,1}),"tail-straddling consumer dependencies are wrong");
    require(mapping.producer_of(19)==0&&mapping.producer_of(38)==2,"matrix row/tile mapping is wrong");
    std::cout<<"COMPLETION_WINDOW_CONTRACT_PASS"<<std::endl;
  }catch(const std::exception &e){std::cerr<<"COMPLETION_WINDOW_CONTRACT_FAIL: "<<e.what()<<std::endl;return 1;}
}
