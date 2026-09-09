#include "completion_window.h"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace mlx::model_events {
namespace {void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}}
CompletionWindow::CompletionWindow(uint64_t epoch,uint64_t blocks,unsigned slots)
    :generation(epoch),total(blocks),capacity(slots){check(epoch&&blocks&&slots&&slots<=32,"invalid bounded completion window");}
void CompletionWindow::key(uint64_t epoch,uint64_t block)const{check(epoch==generation&&block<total,"completion epoch/block identity mismatch");}
bool CompletionWindow::can_admit(uint64_t epoch,uint64_t block)const{
  key(epoch,block);return started&&block>=frontier&&block-frontier<capacity&&entries[block%capacity].state==State::Empty;
}
bool CompletionWindow::ready(uint64_t epoch,uint64_t block)const{
  key(epoch,block);if(block<frontier)return true;if(block-frontier>=capacity)return false;
  const auto &e=entries[block%capacity];return e.block==block&&e.state==State::Done;
}
void CompletionWindow::admit(uint64_t epoch,uint64_t block,uint64_t lease){
  check(lease&&can_admit(epoch,block),"completion window has no admission credit");entries[block%capacity]=Entry{block,lease,0,State::Active};++admissions;++changes;
  unsigned used=0;for(const auto &e:entries)used+=e.state!=State::Empty;peak=std::max(peak,used);
}
void CompletionWindow::complete(uint64_t epoch,uint64_t block,uint64_t lease,uint64_t now){
  key(epoch,block);check(started&&now==cycle&&now<UINT64_MAX,"completion arrived on an invalid edge");auto &e=entries[block%capacity];
  check(e.state==State::Active&&e.block==block&&e.lease==lease,"completion has a stale/duplicate/unadmitted owner");
  e.state=State::Pending;e.visible=now+1;++completions;++changes;
}
void CompletionWindow::advance(uint64_t now){
  check(!started||now>=cycle,"completion clock regressed");cycle=now;started=true;
  for(auto &e:entries)if(e.state==State::Pending&&e.visible<=now){e.state=State::Done;++changes;}
  while(frontier<total){auto &e=entries[frontier%capacity];if(e.state!=State::Done||e.block!=frontier)break;e=Entry{};++frontier;++changes;}
}
Json::Value CompletionWindow::snapshot()const{
  Json::Value r;r["classification"]="bounded_completed_block_events_not_tensor_data";r["epoch"]=Json::UInt64(generation);r["blocks"]=Json::UInt64(total);r["frontier"]=Json::UInt64(frontier);
  r["admitted"]=Json::UInt64(admissions);r["completed"]=Json::UInt64(completions);r["event_slots"]=capacity;r["peak_event_slots"]=peak;r["finished"]=finished();
  unsigned active=0,pending=0,done=0;for(const auto &e:entries){active+=e.state==State::Active;pending+=e.state==State::Pending;done+=e.state==State::Done;}
  r["active"]=active;r["pending_visibility"]=pending;r["out_of_order_done"]=done;return r;
}
uint64_t Mapping::producer_blocks()const{
  if(kind==Kind::Matrix)return batches*((m+1)/2)*((n+15)/16);
  if(kind==Kind::Reduction)return elements/row_width;
  return (elements+15)/16;
}
uint64_t Mapping::producer_of(uint64_t flat)const{
  check(flat<elements,"consumer element outside producer output");
  if(kind==Kind::Matrix){auto batch=flat/(m*n),row=(flat/n)%m,col=flat%n;return (batch*((m+1)/2)+row/2)*((n+15)/16)+col/16;}
  if(kind==Kind::Reduction)return flat/row_width;
  return flat/16;
}
std::vector<uint64_t> Mapping::dependencies(uint64_t block)const{
  check(block<(elements+15)/16,"consumer block outside mapped output");std::set<uint64_t> needed;
  for(uint64_t flat=block*16;flat<std::min(elements,block*16+16);++flat)needed.insert(producer_of(flat));
  return {needed.begin(),needed.end()};
}
Mapping Mapping::parse(const Json::Value &value){
  Mapping m;check(value.isObject()&&value["elements"].isUInt64(),"invalid pipeline output mapping");m.elements=value["elements"].asUInt64();
  check(m.elements>0&&m.elements<=(UINT64_MAX-16)/2,"pipeline mapping extent overflow");auto kind=value["kind"].asString();
  if(kind=="matrix"){
    m.kind=Kind::Matrix;for(const char *key:{"m","n","batches"})check(value[key].isUInt64()&&value[key].asUInt64()>0,"invalid matrix block mapping");
    m.m=value["m"].asUInt64();m.n=value["n"].asUInt64();m.batches=value["batches"].asUInt64();
    check(m.m<=UINT64_MAX/m.n&&m.m*m.n<=UINT64_MAX/m.batches&&m.m*m.n*m.batches==m.elements,"matrix pipeline extent mismatch");
  }else if(kind=="reduction"){
    m.kind=Kind::Reduction;check(value["row_width"].isUInt64(),"invalid reduction block mapping");m.row_width=value["row_width"].asUInt64();check(m.row_width&&m.elements%m.row_width==0,"reduction pipeline extent mismatch");
  }else check(kind=="vector","unknown pipeline block mapping");
  return m;
}
PairFlow::PairFlow(std::shared_ptr<CompletionWindow> w,Mapping m,bool output,unsigned limit,uint64_t start,bool barrier)
    :window(std::move(w)),mapping(m),generation(window->epoch()),offset(start),resident_limit(limit),producer(output),whole_barrier(barrier){
  check(limit&&limit<=32&&window->blocks()==mapping.producer_blocks()&&offset<window->blocks()&&(producer||offset==0),"pipeline flow/credit geometry mismatch");
}
bool PairFlow::admission_ready(uint64_t block)const{if(producer){check(block<window->blocks()-offset,"producer block offset overflow");return window->can_admit(generation,offset+block);}return true;}
bool PairFlow::inputs_ready(uint64_t block)const{
  if(producer)return true;
  if(whole_barrier)return window->finished();
  for(auto required:mapping.dependencies(block))if(!window->ready(generation,required))return false;
  return true;
}
void PairFlow::admitted(uint64_t block,uint64_t lease){if(producer){check(block<window->blocks()-offset,"producer block offset overflow");window->admit(generation,offset+block,lease);}}
void PairFlow::completed(uint64_t block,uint64_t lease,uint64_t cycle){if(producer){check(block<window->blocks()-offset,"producer block offset overflow");window->complete(generation,offset+block,lease,cycle);}}
uint64_t PairFlow::revision()const{return window->revision();}
Json::Value PairFlow::description()const{auto r=window->snapshot();r["role"]=producer?"producer":"consumer";r["whole_source_barrier"]=whole_barrier;r["block_offset"]=Json::UInt64(offset);return r;}
}
