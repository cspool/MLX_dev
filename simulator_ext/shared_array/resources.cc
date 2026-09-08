#include "resources.h"
#include <algorithm>
#include <stdexcept>

namespace mlx::shared_array {
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
template<size_t N> std::optional<unsigned> space(const std::array<uint64_t,N> &owners,unsigned count){
  for(unsigned base=0;base+count<=N;++base){bool free=true;for(unsigned i=0;i<count;++i)free&=owners[base+i]==0;if(free)return base;}
  return std::nullopt;
}
}
void Hardware::validate()const{
  check(rows>=1&&rows<=4&&columns>=1&&columns<=4&&contexts>=1&&contexts<=2,"shared array geometry exceeds bounded storage");
  for(unsigned n:{spm_period,writeback_period,compute_ii,sfu_ii,dma_request_period,dma_response_period,multiply_latency,add_latency,convert_latency,spm_latency,exp_latency,div_latency,sqrt_latency})
    check(n>=1&&n<=1024,"invalid shared array service parameter");
  check(template_word_period>=1&&template_word_period<=1024&&template_trace_limit<=1000000,"invalid template programming limits");
}
bool Hardware::operator==(const Hardware &o)const{
  return std::tie(rows,columns,contexts,spm_period,writeback_period,compute_ii,sfu_ii,dma_request_period,dma_response_period,multiply_latency,add_latency,convert_latency,spm_latency,exp_latency,div_latency,sqrt_latency,template_load_timing,template_word_period,template_trace_limit)==
         std::tie(o.rows,o.columns,o.contexts,o.spm_period,o.writeback_period,o.compute_ii,o.sfu_ii,o.dma_request_period,o.dma_response_period,o.multiply_latency,o.add_latency,o.convert_latency,o.spm_latency,o.exp_latency,o.div_latency,o.sqrt_latency,o.template_load_timing,o.template_word_period,o.template_trace_limit);
}
bool Lease::operator==(const Lease &o)const{
  return std::tie(id,client,block,pe,slot,rf_base,rf_count,spm_base,spm_count,rom_base,rom_count)==
         std::tie(o.id,o.client,o.block,o.pe,o.slot,o.rf_base,o.rf_count,o.spm_base,o.spm_count,o.rom_base,o.rom_count);
}
Resources::Resources(Hardware h):config(h),pes(h.rows*h.columns){config.validate();finishing.reserve(34);retiring.reserve(32);}
const Resources::ClientInfo &Resources::client_info(Client id)const{
  auto it=clients.find(id);check(it!=clients.end()&&it->second.attached,"unknown or detached shared array client");return it->second;
}
Client Resources::attach(const std::string &key,const std::vector<uint32_t> &words,unsigned rf,unsigned spm,uint64_t source){
  check(!open&&!poisoned,"cannot attach during an edge or after shared array failure");
  check(!key.empty()&&!words.empty()&&words.size()<=32&&rf>0&&rf<=16&&spm>0&&spm<=128,"client exceeds RF/SPM/ROM capacity");
  check(next_client!=UINT64_MAX,"shared client identities exhausted");auto id=next_client++;
  clients.emplace(id,ClientInfo{key,words,rf,spm,source,UINT64_MAX,true,config.contexts,pes*config.contexts});return id;
}
void Resources::residency_limit(Client id,unsigned per_pe,unsigned total){
  check(!open&&!poisoned&&live_contexts(id)==0,"cannot change active residency reservations");client_info(id);
  check(per_pe&&per_pe<=config.contexts&&total&&total<=pes*per_pe,"invalid pipeline residency reservation");
  clients.at(id).per_pe_limit=per_pe;clients.at(id).total_limit=total;++allocation_version;
}
void Resources::detach(Client id,bool completed)noexcept{
  auto found=clients.find(id);if(found==clients.end()||!found->second.attached){poisoned=true;return;}
  bool held=false;for(const auto &r:residents)held|=r&&r->lease.client==id;
  if(open||held||!completed)poisoned=true;
  found->second.attached=false;
}
void Resources::begin_cycle(uint64_t cycle){
  check(!open&&!poisoned&&cycle==next_cycle&&cycle<UINT64_MAX,"invalid or poisoned shared array edge");
  open=true;current_cycle=cycle;current_client=0;last_priority.reset();issued.fill(false);written.fill(false);config_port.fill(false);spm_claimed=false;
  bool work=false;
  if(config.template_load_timing&&cycle%config.template_word_period==0)for(unsigned pe=0;pe<pes;++pe){
    for(auto &t:templates[pe])if(t.loaded<t.words.size()){
      check(t.refs&&cycle>t.requested_at,"template programming has no registered request");unsigned index=t.loaded,address=t.base+index;
      check(rom_owner[pe][address]==t.id,"template programming lost ROM ownership");rom[pe][address]=t.words[index];++t.loaded;++template_words_loaded;++template_pe_cycles;config_port[pe]=true;work=true;
      if(t.loaded==t.words.size())t.ready_at=cycle+1;
      ++template_trace_events;
      if(template_trace.size()<config.template_trace_limit){Json::Value e;e["event"]="template_word_write";e["cycle"]=Json::UInt64(cycle);e["pe"]=pe;e["template_id"]=Json::UInt64(t.id);e["word_index"]=index;e["rom_address"]=address;e["word"]=Json::UInt64(t.words[index]);e["last"]=t.loaded==t.words.size();if(e["last"].asBool())e["visible_cycle"]=Json::UInt64(t.ready_at);template_trace.append(e);}
      break;
    }
  }
  template_wall_cycles+=work;
}
void Resources::enter(Client id){
  check(open&&!poisoned,"client tick outside an active shared edge");const auto &info=client_info(id);
  auto priority=std::make_pair(info.source,id);check(!last_priority||*last_priority<priority,"clients must tick once in logical-source priority order");
  check(info.last_cycle==UINT64_MAX||info.last_cycle+1==current_cycle,"active client skipped a shared clock edge");
  clients.at(id).last_cycle=current_cycle;last_priority=priority;current_client=id;
}
void Resources::caller(Client id)const{check(open&&!poisoned&&current_client==id,"shared resource access has the wrong current client");client_info(id);}
const Resources::Resident &Resources::resident(const Lease &lease,bool allow_retiring)const{
  check(lease.pe<pes&&lease.slot<config.contexts&&lease.id!=0,"invalid shared context lease");const auto &r=residents[lease.pe*2+lease.slot];
  check(r&&r->lease==lease&&(allow_retiring||!r->retiring),"stale, foreign or retired shared context lease");return *r;
}
void Resources::validate_lease(const Lease &lease)const{resident(lease);}
std::optional<Offer> Resources::offer(Client id,unsigned pe)const{
  caller(id);check(pe<pes,"shared block PE outside mapped array");const auto &info=client_info(id);std::optional<unsigned> slot;
  unsigned local=0,total=0;for(const auto &r:residents)if(r&&r->lease.client==id){++total;local+=r->lease.pe==pe;}
  if(local>=info.per_pe_limit||total>=info.total_limit)return std::nullopt;
  for(unsigned s=0;s<config.contexts&&!slot;++s)if(!residents[pe*2+s])slot=s;
  if(!slot)return std::nullopt;
  auto rf=space(rf_owner[pe],info.rf),spm=space(spm_owner,info.spm);if(!rf||!spm)return std::nullopt;
  std::optional<unsigned> code;
  for(const auto &t:templates[pe])if(t.key==info.key&&t.words==info.words){code=t.base;break;}
  if(!code)code=space(rom_owner[pe],unsigned(info.words.size()));
  if(!code)return std::nullopt;
  return Offer{Lease{0,id,0,pe,*slot,*rf,info.rf,*spm,info.spm,*code,unsigned(info.words.size())},allocation_version};
}
std::optional<Lease> Resources::admit(const Offer &proposal,uint64_t block){
  auto id=proposal.placement.client;caller(id);auto fresh=offer(id,proposal.placement.pe);
  if(!fresh||fresh->allocation_version!=proposal.allocation_version||!(fresh->placement==proposal.placement))return std::nullopt;
  check(next_lease!=UINT64_MAX&&next_template!=UINT64_MAX,"shared allocation identities exhausted");auto lease=proposal.placement;lease.id=next_lease++;lease.block=block;
  const auto &info=client_info(id);Template *code=nullptr;
  for(auto &t:templates[lease.pe])if(t.key==info.key&&t.words==info.words){code=&t;break;}
  if(!code){templates[lease.pe].push_back(Template{next_template++,info.key,info.words,lease.rom_base,0,0,current_cycle,0});code=&templates[lease.pe].back();
    for(unsigned i=0;i<lease.rom_count;++i){check(!rom_owner[lease.pe][lease.rom_base+i],"non-atomic ROM allocation");rom_owner[lease.pe][lease.rom_base+i]=code->id;if(!config.template_load_timing)rom[lease.pe][lease.rom_base+i]=info.words[i];}
    template_words_requested+=lease.rom_count;
    if(!config.template_load_timing){code->loaded=lease.rom_count;code->ready_at=current_cycle;template_words_loaded+=lease.rom_count;}
  }
  ++code->refs;
  for(unsigned i=0;i<lease.rf_count;++i){auto at=lease.rf_base+i;check(!rf_owner[lease.pe][at],"non-atomic RF allocation");rf_owner[lease.pe][at]=lease.id;registers[lease.pe][at]=Register{};}
  for(unsigned i=0;i<lease.spm_count;++i){auto at=lease.spm_base+i;check(!spm_owner[at],"non-atomic SPM allocation");spm_owner[at]=lease.id;}
  residents[lease.pe*2+lease.slot]=Resident{lease,code->id,false};++allocation_version;++admissions;invariant();return lease;
}
Register &Resources::reg(const Lease &lease,unsigned at){caller(lease.client);resident(lease);check(at<lease.rf_count,"register access outside owned frame");return registers[lease.pe][lease.rf_base+at];}
uint8_t *Resources::spm(const Lease &lease,unsigned at,unsigned bytes){caller(lease.client);resident(lease);check(at<=lease.spm_count*64&&bytes<=lease.spm_count*64-at,"SPM access outside owned arena");return scratchpad.data()+lease.spm_base*64+at;}
bool Resources::template_ready(const Lease &lease)const{
  if(!config.template_load_timing)return true;
  const auto &r=resident(lease);for(const auto &t:templates[lease.pe])if(t.id==r.template_id)return t.loaded==t.words.size()&&current_cycle>=t.ready_at;
  throw std::runtime_error("resident template is missing");
}
uint32_t Resources::instruction(const Lease &lease,unsigned at)const{caller(lease.client);const auto &r=resident(lease);check(template_ready(lease),"instruction read before template programming completed");check(at<lease.rom_count&&rom_owner[lease.pe][lease.rom_base+at]==r.template_id,"instruction access outside owned template");return rom[lease.pe][lease.rom_base+at];}
Resources::Service &Resources::service(Unit unit,unsigned pe){
  check(pe<pes,"service PE outside array");switch(unit){case Unit::Compute:return compute[pe];case Unit::Sfu:return sfu[pe];case Unit::Spm:return spm_service;case Unit::Dma:return dma_service;}throw std::runtime_error("invalid shared service");
}
const Resources::Service &Resources::service(Unit unit,unsigned pe)const{return const_cast<Resources*>(this)->service(unit,pe);}
bool Resources::issue_ready(unsigned pe)const{check(open&&pe<pes,"issue queried outside array edge");return !issued[pe]&&!config_port[pe];}
bool Resources::writeback_ready(unsigned pe)const{check(open&&pe<pes,"writeback queried outside array edge");return current_cycle%config.writeback_period==0&&!written[pe];}
bool Resources::spm_ready()const{check(open,"SPM queried outside array edge");return current_cycle%config.spm_period==0&&!spm_claimed;}
void Resources::claim_issue(const Lease &lease){caller(lease.client);resident(lease);check(template_ready(lease),"issue before template programming completed");check(issue_ready(lease.pe),"shared PE issue bandwidth exceeded");issued[lease.pe]=true;}
void Resources::claim_writeback(const Lease &lease){caller(lease.client);resident(lease);check(writeback_ready(lease.pe),"shared RF writeback bandwidth exceeded");written[lease.pe]=true;}
void Resources::claim_spm(const Lease &lease){caller(lease.client);resident(lease);check(spm_ready(),"shared SPM port bandwidth exceeded");spm_claimed=true;}
bool Resources::unit_ready(Unit unit,unsigned pe)const{check(open,"service queried outside array edge");const auto &s=service(unit,pe);return !s.lease&&current_cycle>=s.next;}
bool Resources::unit_owned_by(Unit unit,Client id,unsigned pe)const{const auto &s=service(unit,pe);return s.lease&&s.client==id;}
void Resources::claim_unit(Unit unit,const Lease &lease){
  caller(lease.client);resident(lease);check(template_ready(lease),"service before template programming completed");check(unit_ready(unit,lease.pe),"shared service slot/II exceeded");auto &s=service(unit,lease.pe);
  auto ii=unit==Unit::Compute?config.compute_ii:unit==Unit::Sfu?config.sfu_ii:1;
  check(current_cycle<=UINT64_MAX-ii,"shared service clock overflow");s={lease.id,lease.client,current_cycle+ii,false};
}
void Resources::complete_unit(Unit unit,const Lease &lease){caller(lease.client);resident(lease);auto &s=service(unit,lease.pe);check(s.lease==lease.id&&s.client==lease.client&&!s.completing,"shared completion owner mismatch or duplicate");s.completing=true;finishing.push_back(&s);}
void Resources::retire(const Lease &lease){
  caller(lease.client);resident(lease);
  check(template_ready(lease),"retiring before template programming completed");
  for(Unit u:{Unit::Compute,Unit::Sfu,Unit::Spm,Unit::Dma}){const auto &s=service(u,lease.pe);check(s.lease!=lease.id||s.completing,"retiring context still owns an in-flight service");}
  residents[lease.pe*2+lease.slot]->retiring=true;
  retiring.push_back(lease.pe*2+lease.slot);
}
unsigned Resources::live_contexts(Client id)const{unsigned n=0;for(const auto &r:residents)n+=r&&r->lease.client==id&&!r->retiring;return n;}
unsigned Resources::live_spm_vectors(Client id)const{unsigned n=0;for(const auto &r:residents)if(r&&r->lease.client==id&&!r->retiring)n+=r->lease.spm_count;return n;}
void Resources::end_cycle(){
  check(open&&!poisoned,"ending inactive or poisoned shared edge");
  for(auto *s:finishing){s->lease=s->client=0;s->completing=false;}
  finishing.clear();
  for(auto slot:retiring){auto &entry=residents[slot];check(entry&&entry->retiring,"missing retired shared context");const auto lease=entry->lease;
    for(unsigned i=0;i<lease.rf_count;++i){check(rf_owner[lease.pe][lease.rf_base+i]==lease.id,"RF owner changed before release");rf_owner[lease.pe][lease.rf_base+i]=0;registers[lease.pe][lease.rf_base+i]=Register{};}
    for(unsigned i=0;i<lease.spm_count;++i){check(spm_owner[lease.spm_base+i]==lease.id,"SPM owner changed before release");spm_owner[lease.spm_base+i]=0;}
    auto &pool=templates[lease.pe];auto code=std::find_if(pool.begin(),pool.end(),[&](const Template &t){return t.id==entry->template_id;});check(code!=pool.end()&&code->refs,"missing live instruction template");
    if(!--code->refs){for(unsigned i=0;i<code->words.size();++i)rom_owner[lease.pe][code->base+i]=0;pool.erase(code);}
    entry.reset();++retirements;++allocation_version;
  }
  if(!retiring.empty())invariant();
  retiring.clear();open=false;current_client=0;next_cycle=current_cycle+1;
}
void Resources::invariant(){
  unsigned contexts=0,spm=0;std::array<unsigned,16> rf{},code{};
  for(const auto &entry:residents)if(entry){const auto &l=entry->lease;++contexts;
    for(unsigned i=0;i<l.rf_count;++i)check(rf_owner[l.pe][l.rf_base+i]==l.id,"resident RF ownership lost");
    rf[l.pe]+=l.rf_count;
    for(unsigned i=0;i<l.spm_count;++i)check(spm_owner[l.spm_base+i]==l.id,"resident SPM ownership lost");
    spm+=l.spm_count;
  }
  check(contexts<=pes*config.contexts&&spm<=128,"shared context/SPM conservation failed");
  check(admissions>=retirements&&admissions-retirements==contexts,"shared context allocation count mismatch");
  unsigned spm_slots=0;for(auto owner:spm_owner)spm_slots+=owner!=0;check(spm_slots==spm,"orphan shared SPM allocation");
  for(unsigned p=0;p<pes;++p){unsigned actual=0;for(auto owner:rf_owner[p])actual+=owner!=0;check(actual==rf[p]&&rf[p]<=16,"orphan/excess shared RF allocation");
    for(auto owner:rom_owner[p])code[p]+=owner!=0;
    check(code[p]<=32,"shared ROM capacity exceeded");
    for(const auto &t:templates[p]){unsigned refs=0;for(const auto &r:residents)refs+=r&&r->lease.pe==p&&r->template_id==t.id;check(refs==t.refs&&refs>0,"template reference accounting failed");}
    peak_rf=std::max(peak_rf,rf[p]);peak_rom=std::max(peak_rom,code[p]);
  }
  peak_contexts=std::max(peak_contexts,contexts);peak_spm=std::max(peak_spm,spm);
}
bool Resources::idle()const{
  for(const auto &r:residents)if(r)return false;
  for(const auto &s:compute)if(s.lease)return false;
  for(const auto &s:sfu)if(s.lease)return false;
  return !spm_service.lease&&!dma_service.lease&&!poisoned;
}
Json::Value Resources::snapshot()const{
  Json::Value r;r["classification"]="shared_tensor_array_resources_not_model_or_system_validation";
  r["cycle"]=Json::UInt64(current_cycle);r["in_cycle"]=open;r["poisoned"]=poisoned;r["idle"]=idle();r["admitted"]=Json::UInt64(admissions);r["retired"]=Json::UInt64(retirements);
  r["peak_contexts"]=peak_contexts;r["peak_spm_vectors"]=peak_spm;r["peak_rf_vectors_per_pe"]=peak_rf;r["peak_rom_words_per_pe"]=peak_rom;
  r["rf_vectors_per_pe"]=16;r["spm_vectors_total"]=128;r["rom_words_per_pe"]=32;r["context_slots_per_pe"]=config.contexts;
  r["template_words_loaded"]=Json::UInt64(template_words_loaded);r["template_configuration_cycles_modeled"]=config.template_load_timing;
  if(config.template_load_timing){unsigned pending=0;for(const auto &pool:templates)for(const auto &t:pool)pending+=unsigned(t.words.size())-t.loaded;
    r["template_words_requested"]=Json::UInt64(template_words_requested);r["template_config_wall_cycles"]=Json::UInt64(template_wall_cycles);r["template_config_pe_cycles"]=Json::UInt64(template_pe_cycles);r["template_word_period"]=config.template_word_period;r["pending_template_words"]=pending;
    r["template_trace"]=template_trace;r["template_trace_events"]=Json::UInt64(template_trace_events);r["template_trace_truncated"]=template_trace_events>template_trace.size();r["template_transport_scope"]="local_per_pe_programming_from_decoded_descriptor_not_host_or_cache_loading";
  }
  r["inference_performance_eligible"]=false;r["full_model_execution_verified"]=false;return r;
}
} // namespace mlx::shared_array
