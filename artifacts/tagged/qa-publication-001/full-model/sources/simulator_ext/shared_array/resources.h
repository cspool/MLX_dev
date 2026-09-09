#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <json/json.h>

namespace mlx::shared_array {

// These are the same 64-byte values used by both tensor microcode profiles.
// Readiness/type metadata is not a second copy of the register data.
struct Register { std::array<uint32_t,16> data{}; unsigned type=0,valid=0; };
enum class Unit { Compute, Sfu, Spm, Dma };
using Client = uint64_t;

struct Hardware {
  unsigned rows=4,columns=4,contexts=2;
  unsigned spm_period=1,writeback_period=1,compute_ii=1,sfu_ii=1;
  unsigned dma_request_period=1,dma_response_period=1;
  unsigned multiply_latency=4,add_latency=2,convert_latency=2,spm_latency=3;
  unsigned exp_latency=8,div_latency=12,sqrt_latency=12;
  bool template_load_timing=false;
  unsigned template_word_period=1,template_trace_limit=0;
  void validate()const;
  bool operator==(const Hardware &other)const;
};

struct Lease {
  uint64_t id=0,client=0,block=0;
  unsigned pe=0,slot=0,rf_base=0,rf_count=0,spm_base=0,spm_count=0,rom_base=0,rom_count=0;
  bool operator==(const Lease &other)const;
};
struct Offer { Lease placement; uint64_t allocation_version=0; };

// One physical bank and arbitration domain, shared by all attached frontends.
// Clients execute in declared logical-source order within an edge. Resource
// releases are committed at end_cycle, never exposed to a later caller in the
// same cycle. Optional local template writes share the PE issue opportunity;
// descriptor/host transport is a separate, still unmodeled operation.
class Resources {
public:
  explicit Resources(Hardware hardware={});
  Resources(const Resources &)=delete;
  Resources &operator=(const Resources &)=delete;
  Resources(Resources &&)=delete;
  Resources &operator=(Resources &&)=delete;
  const Hardware &hardware()const{return config;}
  Client attach(const std::string &decode_key,const std::vector<uint32_t> &words,
                unsigned rf_vectors,unsigned spm_vectors,uint64_t source_id=UINT64_MAX);
  void detach(Client client,bool completed)noexcept;
  void residency_limit(Client client,unsigned per_pe,unsigned total);
  void begin_cycle(uint64_t cycle);
  void enter(Client client);
  void end_cycle();
  uint64_t cycle()const{return current_cycle;}
  uint64_t version()const{return allocation_version;}
  bool in_cycle()const{return open;}
  std::optional<Offer> offer(Client client,unsigned pe)const;
  std::optional<Lease> admit(const Offer &offer,uint64_t block);
  void retire(const Lease &lease);
  Register &reg(const Lease &lease,unsigned relative);
  uint8_t *spm(const Lease &lease,unsigned relative,unsigned bytes);
  uint32_t instruction(const Lease &lease,unsigned relative)const;
  bool template_ready(const Lease &lease)const;
  void validate_lease(const Lease &lease)const;
  unsigned live_contexts(Client client)const;
  unsigned live_spm_vectors(Client client)const;
  bool issue_ready(unsigned pe)const;
  bool writeback_ready(unsigned pe)const;
  bool spm_ready()const;
  void claim_issue(const Lease &lease);
  void claim_writeback(const Lease &lease);
  void claim_spm(const Lease &lease);
  bool unit_ready(Unit kind,unsigned pe=0)const;
  bool unit_owned_by(Unit kind,Client client,unsigned pe=0)const;
  void claim_unit(Unit kind,const Lease &lease);
  void complete_unit(Unit kind,const Lease &lease);
  bool idle()const;
  Json::Value snapshot()const;
private:
  struct ClientInfo {
    std::string key;
    std::vector<uint32_t> words;
    unsigned rf=0,spm=0;
    uint64_t source=UINT64_MAX,last_cycle=UINT64_MAX;
    bool attached=true;
    unsigned per_pe_limit=2,total_limit=32;
    // Host-only memoization of a failed physical-capacity query. Register/SPM
    // values, template loading and issue readiness are not cached here.
    mutable std::optional<std::pair<uint64_t,unsigned>> failed_offer=std::nullopt;
  };
  struct Resident {Lease lease;uint64_t template_id=0;bool retiring=false;};
  struct Template {uint64_t id=0;std::string key;std::vector<uint32_t> words;unsigned base=0,refs=0,loaded=0;uint64_t requested_at=0,ready_at=0;};
  struct Service {uint64_t lease=0,client=0,next=0;bool completing=false;};
  Hardware config;
  unsigned pes=0;
  bool open=false,poisoned=false,spm_claimed=false;
  uint64_t current_cycle=0,next_cycle=0,next_client=1,next_lease=1,next_template=1;
  uint64_t allocation_version=0,current_client=0,admissions=0,retirements=0,template_words_loaded=0;
  uint64_t template_words_requested=0,template_wall_cycles=0,template_pe_cycles=0,template_trace_events=0;
  Json::Value template_trace{Json::arrayValue};
  unsigned peak_contexts=0,peak_spm=0,peak_rf=0,peak_rom=0;
  std::optional<std::pair<uint64_t,uint64_t>> last_priority;
  std::map<Client,ClientInfo> clients;
  std::array<std::optional<Resident>,32> residents;
  std::array<std::array<Register,16>,16> registers{};
  std::array<uint8_t,8192> scratchpad{};
  std::array<std::array<uint64_t,16>,16> rf_owner{};
  std::array<uint64_t,128> spm_owner{};
  std::array<std::array<uint64_t,32>,16> rom_owner{};
  std::array<std::array<uint32_t,32>,16> rom{};
  std::array<std::vector<Template>,16> templates;
  std::array<bool,16> issued{},written{},config_port{};
  std::array<Service,16> compute{},sfu{};
  Service spm_service,dma_service;
  std::vector<Service*> finishing;
  std::vector<unsigned> retiring;
  const ClientInfo &client_info(Client client)const;
  const Resident &resident(const Lease &lease,bool permit_retiring=false)const;
  void caller(Client client)const;
  Service &service(Unit kind,unsigned pe);
  const Service &service(Unit kind,unsigned pe)const;
  void invariant();
};
} // namespace mlx::shared_array
