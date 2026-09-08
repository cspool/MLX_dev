#include "../simulator_ext/shared_array/resources.h"
#include <iostream>
#include <stdexcept>

using namespace mlx::shared_array;
namespace {
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F fn,const char *word){bool failed=false;try{fn();}catch(const std::runtime_error &e){failed=std::string(e.what()).find(word)!=std::string::npos;}require(failed,"expected template contract rejection did not occur");}
}
int main(){
  try{
    Hardware h;h.rows=h.columns=1;h.template_load_timing=true;h.template_word_period=2;h.template_trace_limit=100;
    Resources array(h);std::vector<uint32_t> aw{0x11223344,0x55667788,0xCAFEBABE,0x13572468},bw{91,92,93};
    auto a=array.attach("a",aw,6,37,1),b=array.attach("b",bw,8,5,2);array.begin_cycle(0);array.enter(a);auto al=*array.admit(*array.offer(a,0),0);
    rejects([&]{array.instruction(al,0);},"before template");rejects([&]{array.claim_issue(al);},"before template");rejects([&]{array.claim_unit(Unit::Compute,al);},"before template");rejects([&]{array.retire(al);},"before template");
    array.enter(b);auto bl=*array.admit(*array.offer(b,0),0);array.end_cycle();
    for(uint64_t cycle=1;cycle<=15;++cycle){array.begin_cycle(cycle);array.enter(a);
      require(array.template_ready(al)==(cycle>=9),"first template visibility edge changed");
      if(cycle>=9){for(unsigned i=0;i<aw.size();++i)require(array.instruction(al,i)==aw[i],"actual programmed instruction word differs");
        if(cycle%2==0&&cycle<=14)rejects([&]{array.claim_issue(al);},"bandwidth");else array.claim_issue(al);
      }
      array.enter(b);require(array.template_ready(bl)==(cycle>=15),"second template bypassed single-word port");
      if(cycle==15){for(unsigned i=0;i<bw.size();++i)require(array.instruction(bl,i)==bw[i],"second template data differs");array.retire(bl);}
      array.end_cycle();
    }
    array.detach(b,true);auto c=array.attach("a",aw,6,37,3);array.begin_cycle(16);array.enter(a);array.enter(c);auto cl=*array.admit(*array.offer(c,0),1);
    require(array.template_ready(cl),"an existing loaded template was needlessly reset");array.end_cycle();
    require(array.snapshot()["template_words_loaded"].asUInt64()==7,"shared template was loaded twice");
    array.begin_cycle(17);array.enter(a);array.retire(al);array.enter(c);array.retire(cl);array.end_cycle();array.detach(a,true);array.detach(c,true);
    auto d=array.attach("different",{201,202},6,37,4);array.begin_cycle(18);array.enter(d);auto dl=*array.admit(*array.offer(d,0),2);
    rejects([&]{array.validate_lease(al);},"stale");rejects([&]{array.instruction(dl,0);},"before template");array.end_cycle();
    for(uint64_t cycle=19;cycle<=23;++cycle){array.begin_cycle(cycle);array.enter(d);require(array.template_ready(dl)==(cycle==23),"reused ROM exposed stale contents");if(cycle==23){require(array.instruction(dl,0)==201&&array.instruction(dl,1)==202,"ROM reconfiguration changed words");array.retire(dl);}array.end_cycle();}
    array.detach(d,true);auto report=array.snapshot();require(array.idle()&&report["template_words_loaded"].asUInt64()==9&&report["template_words_requested"].asUInt64()==9&&report["pending_template_words"].asUInt()==0,"template load accounting did not drain");
    require(report["template_config_pe_cycles"].asUInt64()==9&&report["template_config_wall_cycles"].asUInt64()==9,"configuration clock units disagree");
    std::cout<<"TEMPLATE_LOADER_CONTRACT_PASS"<<std::endl;
  }catch(const std::exception &e){std::cerr<<"TEMPLATE_LOADER_CONTRACT_FAIL: "<<e.what()<<std::endl;return 1;}
}
