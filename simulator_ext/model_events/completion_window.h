#pragma once
#include "block_flow.h"
#include <array>
#include <memory>
#include <vector>

namespace mlx::model_events {
// One reusable event bank: 32 physical records plus a completed-prefix count.
// Prefix compression does not discard data: the tensor backing remains pinned
// until the actual consumer finishes. Every bank instance has an epoch.
class CompletionWindow {
  enum class State { Empty, Active, Pending, Done };
  struct Entry {uint64_t block=0,lease=0,visible=0;State state=State::Empty;};
  std::array<Entry,32> entries{};
  uint64_t generation,total,frontier=0,cycle=0,changes=0,admissions=0,completions=0;
  unsigned capacity,peak=0;
  bool started=false;
  void key(uint64_t epoch,uint64_t block)const;
public:
  CompletionWindow(uint64_t epoch,uint64_t blocks,unsigned slots=32);
  void advance(uint64_t cycle);
  bool can_admit(uint64_t epoch,uint64_t block)const;
  bool ready(uint64_t epoch,uint64_t block)const;
  void admit(uint64_t epoch,uint64_t block,uint64_t lease);
  void complete(uint64_t epoch,uint64_t block,uint64_t lease,uint64_t cycle);
  bool finished()const{return frontier==total;}
  uint64_t revision()const{return changes;}
  uint64_t epoch()const{return generation;}
  uint64_t blocks()const{return total;}
  Json::Value snapshot()const;
};

// A closed, single-producer/single-consumer pair. Consumer lanes must cover
// the same logical output shape; no unbounded per-element readiness bitmap.
struct Mapping {
  enum class Kind { Matrix, Vector, Reduction };
  Kind kind=Kind::Vector;
  uint64_t elements=0,m=0,n=0,batches=1,row_width=0;
  uint64_t producer_blocks()const;
  uint64_t producer_of(uint64_t flat)const;
  std::vector<uint64_t> dependencies(uint64_t consumer_block)const;
  static Mapping parse(const Json::Value &value);
};

class PairFlow final:public BlockFlow {
  std::shared_ptr<CompletionWindow> window;
  Mapping mapping;
  uint64_t generation,offset=0;
  unsigned resident_limit;
  bool producer,whole_barrier;
public:
  PairFlow(std::shared_ptr<CompletionWindow> window,Mapping mapping,bool producer,
           unsigned total_limit,uint64_t block_offset=0,bool whole_source_barrier=false);
  bool admission_ready(uint64_t block)const override;
  bool inputs_ready(uint64_t block)const override;
  void admitted(uint64_t block,uint64_t lease)override;
  void completed(uint64_t block,uint64_t lease,uint64_t cycle)override;
  uint64_t revision()const override;
  unsigned per_pe_limit()const override{return 1;}
  unsigned total_limit()const override{return resident_limit;}
  Json::Value description()const override;
};
}
