#include "../simulator_ext/shared_array/resources.h"
#include <iostream>
#include <stdexcept>

using namespace mlx::shared_array;
namespace {
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F fn){bool failed=false;try{fn();}catch(const std::runtime_error &){failed=true;}require(failed,"illegal caller/lease was accepted through cached failure");}
void capacity(unsigned rf,unsigned spm,unsigned words,unsigned slots,bool globally_full){
  Hardware h;h.rows=1;h.columns=2;h.contexts=slots;Resources array(h);
  auto a=array.attach("a",std::vector<uint32_t>(words,1),rf,spm,1),b=array.attach("b",{2},8,40,2);
  array.begin_cycle(0);array.enter(a);auto before=*array.offer(a,0);auto lease=*array.admit(before,0);
  require(!array.admit(before,99),"stale successful proposal was accepted after allocation");
  array.enter(b);require(!array.offer(b,0),"fixture did not exhaust the target PE resource");
  for(unsigned i=0;i<100;++i)require(!array.offer(b,0),"stable failed offer changed");
  require(bool(array.offer(b,1))!=globally_full,"cached PE zero failure contaminated another PE");
  rejects([&]{array.offer(a,0);});rejects([&]{array.offer(b,2);});array.end_cycle();
  array.begin_cycle(1);array.enter(a);array.retire(lease);array.enter(b);
  require(!array.offer(b,0),"retirement freed resources before the shared edge ended");array.end_cycle();array.detach(a,true);
  array.begin_cycle(2);array.enter(b);auto renewed=array.offer(b,globally_full?1:0);require(bool(renewed),"cached failure survived real resource release");
  auto next=*array.admit(*renewed,1);array.retire(next);array.end_cycle();array.detach(b,true);require(array.idle(),"capacity test leaked resources");
}
void limit_and_detach(){
  Hardware h;h.rows=1;h.columns=2;Resources array(h);auto a=array.attach("a",{1},6,1,1),b=array.attach("b",{2},8,1,2);
  array.residency_limit(a,1,1);array.begin_cycle(0);array.enter(a);auto lease=*array.admit(*array.offer(a,0),0);
  require(!array.offer(a,1)&&!array.offer(a,1),"total residency reservation was ignored");rejects([&]{array.residency_limit(a,2,4);});
  array.retire(lease);array.enter(b);array.end_cycle();array.residency_limit(a,2,4);
  array.begin_cycle(1);array.enter(a);auto next=*array.admit(*array.offer(a,1),1);array.retire(next);array.enter(b);array.end_cycle();array.detach(a,true);
  array.begin_cycle(2);rejects([&]{array.enter(a);});array.enter(b);array.end_cycle();array.detach(b,true);require(array.idle(),"limit test leaked resources");
}
}
int main(){try{
  capacity(16,1,1,2,false); // RF capacity, independent PE remains eligible.
  capacity(1,100,1,2,true); // Global SPM capacity blocks both PEs.
  capacity(1,1,32,2,false); // ROM capacity.
  capacity(1,1,1,1,false); // Physical context slot capacity.
  limit_and_detach();std::cout<<"SHARED_OFFER_CACHE_CONTRACT_PASS\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
