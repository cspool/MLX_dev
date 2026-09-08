#include "pair_runtime.hh"
#include <algorithm>

namespace mlx::physical_device {
using namespace tensor_model;
namespace {
shared_array::Hardware hardware(const matrix_schedule::Options &m,const vector_schedule::Options &v){
  m.validate();v.validate();require(m.rows==v.rows&&m.columns==v.columns&&m.contexts==v.contexts&&m.contexts>=2,"pair requires matching two-context array profiles");
  require(m.multiply_latency==v.multiply_latency&&m.add_latency==v.add_latency&&m.convert_latency==v.convert_latency&&m.spm_latency==v.spm_latency&&m.spm_period==v.spm_period&&m.writeback_period==v.writeback_period&&m.compute_ii==v.vector_ii&&m.dma_request_period==v.dma_request_period&&m.dma_response_period==v.dma_response_period,"pair shared service profiles disagree");
  shared_array::Hardware h;h.rows=m.rows;h.columns=m.columns;h.contexts=m.contexts;h.spm_period=m.spm_period;h.writeback_period=m.writeback_period;h.compute_ii=m.compute_ii;h.sfu_ii=v.trans_ii;
  h.dma_request_period=m.dma_request_period;h.dma_response_period=m.dma_response_period;h.multiply_latency=m.multiply_latency;h.add_latency=m.add_latency;h.convert_latency=m.convert_latency;h.spm_latency=m.spm_latency;
  h.exp_latency=v.exp_latency;h.div_latency=v.div_latency;h.sqrt_latency=v.sqrt_latency;h.template_load_timing=true;h.template_word_period=1;h.template_trace_limit=m.trace||v.trace?100000:0;return h;
}
uint64_t broadcast(uint64_t index,const Shape &out,const Shape &input){
  require(input.size()<=out.size(),"pair matrix broadcast rank mismatch");uint64_t result=0,step=1;
  for(size_t d=out.size();d-->0;){auto at=index%out[d];index/=out[d];if(d+input.size()>=out.size()){auto n=input[d+input.size()-out.size()];require(n==1||n==out[d],"pair matrix broadcast extent mismatch");if(n!=1)result+=at*step;step*=n;}}
  return result;
}
}
struct PairRuntime::Impl {
  const DecodedPair &pair;
  matrix_schedule::Options mo;
  vector_schedule::Options vo;
  shared_array::Resources array;
  std::shared_ptr<model_events::CompletionWindow> events;
  std::unique_ptr<model_events::PairFlow> producer_flow,consumer_flow;
  model_io::PhysicalMux mux;
  model_io::RequestTokens &tokens;
  std::unique_ptr<model_io::PhysicalMemoryPort> producer_channel,consumer_channel;
  std::unique_ptr<model_io::AddressSpacePort> producer_port,consumer_port;
  std::unique_ptr<matrix_schedule::Simulator> matrix;
  std::unique_ptr<vector_schedule::Simulator> vector,consumer;
  uint64_t origin=0,cycles=0,batch=0;
  unsigned producer_limit=0,consumer_limit=0;
  bool producer_finished=false,next_batch=false;
  Json::Value windows{Json::arrayValue};

  Impl(const DecodedPair &p,matrix_schedule::Options m,vector_schedule::Options v,model_io::PhysicalMemoryPort &physical,model_io::RequestTokens &t,uint64_t start,uint64_t epoch)
      :pair(p),mo(m),vo(v),array(hardware(m,v)),events(std::make_shared<model_events::CompletionWindow>(epoch,p.mapping.producer_blocks(),p.event_slots)),mux(physical,2),tokens(t),origin(start){
    const auto &pp=p.matrix?p.matrix->program:p.vector->node["vector_program"],&cp=p.consumer.node["vector_program"];
    auto prf=p.matrix?6u:8u,pspm=(pp["spm_bytes_used"].asUInt()+63)/64;
    auto pwords=p.matrix?pp["prologue"].size()+pp["body"].size()+pp["epilogue"].size():pp["rom"].size();auto cwords=cp["rom"].size();
    require(prf+8<=16&&pspm>0&&pspm+5<=128&&pwords+cwords<=32,"pair exceeds joint RF/SPM/ROM capacity");
    producer_limit=mo.rows*mo.columns;consumer_limit=std::min(producer_limit,(128-pspm)/5);require(consumer_limit>0,"pair cannot reserve producer progress");
    producer_channel=mux.channel("producer:"+std::to_string(p.producer_source));consumer_channel=mux.channel("consumer:"+std::to_string(p.consumer_source));
    start_producer();consumer_flow=std::make_unique<model_events::PairFlow>(events,p.mapping,false,consumer_limit);
    consumer_port=std::make_unique<model_io::AddressSpacePort>(*consumer_channel,tokens,p.consumer.regions,origin);
    consumer=std::make_unique<vector_schedule::Simulator>(p.consumer.node,p.consumer.values,p.consumer.output,vo,consumer_port.get(),&array,p.consumer_source,consumer_flow.get());
  }
  void start_producer(){
    require(cycles<=UINT64_MAX-origin,"pair physical clock origin overflow");producer_port=std::make_unique<model_io::AddressSpacePort>(*producer_channel,tokens,pair.producer_regions(),origin+cycles);
    auto offset=pair.matrix?batch*((pair.matrix->m+1)/2)*((pair.matrix->n+15)/16):0;producer_flow=std::make_unique<model_events::PairFlow>(events,pair.mapping,true,producer_limit,offset);
    if(pair.matrix){const auto &p=*pair.matrix;uint64_t ab=0,bb=0;
      if(!p.transpose_b){Shape output(p.output.sizes.begin(),p.output.sizes.end()-2),a(p.a.sizes.begin(),p.a.sizes.end()-2),b(p.b.sizes.begin(),p.b.sizes.end()-2);ab=broadcast(batch,output,a);bb=broadcast(batch,output,b);}
      matrix=std::make_unique<matrix_schedule::Simulator>(p.program,p.a,p.b,p.has_bias?&p.bias:nullptr,p.transpose_b,ab,bb,p.m,p.n,p.k,p.output,batch,mo,producer_port.get(),&array,pair.producer_source,producer_flow.get());
    }else{const auto &p=*pair.vector;vector=std::make_unique<vector_schedule::Simulator>(p.node,p.values,p.output,vo,producer_port.get(),&array,pair.producer_source,producer_flow.get());}
    next_batch=false;
  }
  bool producer_done()const{return matrix?matrix->done():vector->done();}
  bool done()const{return producer_finished&&consumer->done()&&events->finished()&&array.idle()&&mux.idle();}
  void tick(){
    if(done())return;
    require(cycles<UINT64_MAX-origin,"pair physical clock overflow");if(next_batch)start_producer();
    events->advance(cycles);mux.advance(origin+cycles);array.begin_cycle(cycles);
    if(!producer_finished){if(matrix)matrix->tick();else vector->tick();}
    if(!consumer->done())consumer->tick();
    array.end_cycle();
    if(!producer_finished&&producer_done()){
      auto result=matrix?matrix->result():vector->result();require(result["done"].asBool()&&result["dma_requests"]==result["dma_responses"],"pair producer batch did not drain");result["batch_index"]=Json::UInt64(batch);windows.append(result);
      matrix.reset();vector.reset();producer_port.reset();producer_flow.reset();
      if(++batch<pair.batches)next_batch=true;else producer_finished=true;
    }
    ++cycles;
  }
  Json::Value result()const{
    Json::Value r;r["classification"]="clocked_full_source_pair_kernel_not_complete_model_system_validation";r["done"]=done();r["cycles"]=Json::UInt64(cycles);
    r["producer_source"]=Json::UInt64(pair.producer_source);r["consumer_source"]=Json::UInt64(pair.consumer_source);r["producer_batches"]=Json::UInt64(pair.batches);r["producer_windows"]=windows;r["consumer_window"]=consumer->result();
    r["array"]=array.snapshot();r["block_events"]=events->snapshot();r["physical_mux"]=mux.snapshot();r["external_memory_port"]=true;r["functional_entry_calls"]=0;r["blas_calls"]=0;
    r["full_model_execution_verified"]=false;r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;return r;
  }
};
PairRuntime::PairRuntime(const DecodedPair &p,matrix_schedule::Options m,vector_schedule::Options v,model_io::PhysicalMemoryPort &port,model_io::RequestTokens &tokens,uint64_t origin,uint64_t epoch)
    :impl(std::make_unique<Impl>(p,m,v,port,tokens,origin,epoch)){}
PairRuntime::~PairRuntime()=default;
void PairRuntime::tick(){impl->tick();}
bool PairRuntime::done()const{return impl->done();}
Json::Value PairRuntime::result()const{return impl->result();}
}
