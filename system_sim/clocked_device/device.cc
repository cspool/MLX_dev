#include "device.hh"
#include "queued_physical_port.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

namespace mlx::clocked_device {
using namespace model_io;
using namespace physical_device;
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
struct Transport final:PhysicalMemoryPort {
  QueuedPhysicalPort queue;
  std::optional<PhysicalRequest> owner;
  std::map<uint64_t,uint64_t> written; // Acknowledged byte intervals, not data.
  uint64_t address_limit;
  explicit Transport(unsigned bits):address_limit((UINT64_C(1)<<bits)-1){}
  void advance(uint64_t cycle)override{queue.advance(cycle);}
  bool request_ready()const override{return queue.request_ready();}
  void submit(const PhysicalRequest &q)override{
    check(q.address<=address_limit&&q.bytes&&q.bytes-1<=address_limit-q.address,"request exceeds system address width");
    queue.submit(q);owner=q;
  }
  std::optional<Response> response()const override{return queue.response();}
  void consume_response()override{queue.consume_response();owner.reset();}
  void answer(const Response &r){
    queue.receive_response(r);
    check(owner&&owner->id==r.id,"transport response lost its request identity");
    if(owner->write&&!r.error){
      uint64_t begin=owner->address,end=begin+owner->bytes;auto it=written.lower_bound(begin);
      if(it!=written.begin()){auto prev=std::prev(it);if(prev->second>=begin){begin=prev->first;end=std::max(end,prev->second);it=written.erase(prev);}}
      while(it!=written.end()&&it->first<=end){end=std::max(end,it->second);it=written.erase(it);}
      written.emplace(begin,end);
    }
  }
  bool produced(uint64_t address,unsigned bytes)const{
    auto it=written.upper_bound(address);if(it==written.begin())return false;--it;
    return address<it->second&&bytes<=it->second-address;
  }
};
}

struct Device::Impl {
  enum class Phase {Idle,Fetch,Run,Drain,Done,Failed};
  Options options;
  Transport transport;
  RequestTokens tokens;
  Phase phase=Phase::Idle;
  uint64_t cycle=0,source=0,launches=0,start=0,fetch_cycles=0,run_cycles=0,drain_cycles=0;
  uint64_t descriptor_address=0,descriptor_bytes=0,fetched=0;
  std::vector<uint8_t> descriptor;
  std::string error,backend;
  std::optional<DecodedMatrix> matrix;
  std::optional<DecodedVector> vector;
  std::optional<DecodedMemory> memory;
  std::optional<DecodedPair> pair;
  std::unique_ptr<AddressSpacePort> addresses;
  std::unique_ptr<matrix_schedule::Simulator> matrix_model;
  std::unique_ptr<vector_schedule::Simulator> vector_model;
  std::unique_ptr<memory_model::Simulator> memory_model;
  std::unique_ptr<PairRuntime> pair_model;
  Json::Value kernel;

  explicit Impl(Options value):options(std::move(value)),transport(valid_bits(options.address_bits)){
    check(options.max_busy_cycles>0,"clocked device cycle limit must be positive");
  }
  static unsigned valid_bits(unsigned bits){check(bits>=32&&bits<=63,"unsupported clocked device address width");return bits;}
  bool busy()const{return phase==Phase::Fetch||phase==Phase::Run||phase==Phase::Drain;}
  bool complete()const{return phase==Phase::Done||phase==Phase::Failed;}
  void release_models(){matrix_model.reset();vector_model.reset();memory_model.reset();pair_model.reset();}
  void release_decoded(){addresses.reset();matrix.reset();vector.reset();memory.reset();pair.reset();}
  void fail(const std::string &message){
    if(error.empty())error=message;
    release_models();phase=transport.queue.idle()?Phase::Failed:Phase::Drain;
    if(phase==Phase::Failed)release_decoded();
  }
  void launch(uint64_t address,uint64_t bytes,uint64_t id){
    check(!busy()&&transport.queue.idle(),"launch requires drained device");
    check(bytes==sizeof(mlx_matrix_wire)||bytes==sizeof(mlx_vector_wire)||bytes==sizeof(mlx_memory_wire)||bytes==sizeof(mlx_pair_wire),"unregistered descriptor size");
    check(address%8==0&&address<=transport.address_limit&&bytes-1<=transport.address_limit-address,"descriptor address alignment/range invalid");
    release_models();release_decoded();kernel=Json::Value();transport.written.clear();error.clear();backend.clear();
    descriptor_address=address;descriptor_bytes=bytes;descriptor.assign(size_t(bytes),0);fetched=0;
    source=id;start=cycle;fetch_cycles=run_cycles=drain_cycles=0;++launches;phase=Phase::Fetch;
  }
  uint64_t expected_magic()const{
    return descriptor_bytes==sizeof(mlx_matrix_wire)?MLX_MATRIX_WIRE_MAGIC:descriptor_bytes==sizeof(mlx_vector_wire)?MLX_VECTOR_WIRE_MAGIC:descriptor_bytes==sizeof(mlx_pair_wire)?MLX_PAIR_WIRE_MAGIC:MLX_MEMORY_WIRE_MAGIC;
  }
  void decode(){
    if(descriptor_bytes==sizeof(mlx_matrix_wire)){auto wire=std::make_unique<mlx_matrix_wire>();std::memcpy(wire.get(),descriptor.data(),sizeof(*wire));matrix=decode_matrix(*wire);backend="matrix";}
    else if(descriptor_bytes==sizeof(mlx_vector_wire)){auto wire=std::make_unique<mlx_vector_wire>();std::memcpy(wire.get(),descriptor.data(),sizeof(*wire));vector=decode_vector(*wire);backend="vector";}
    else if(descriptor_bytes==sizeof(mlx_pair_wire)){auto wire=std::make_unique<mlx_pair_wire>();std::memcpy(wire.get(),descriptor.data(),sizeof(*wire));pair=decode_pair(*wire);backend="pair";}
    else{auto wire=std::make_unique<mlx_memory_wire>();std::memcpy(wire.get(),descriptor.data(),sizeof(*wire));memory=decode_memory(*wire);backend="memory";}
    const auto &regions=matrix?matrix->regions:vector?vector->regions:pair?pair->regions:memory->regions;
    for(const auto &r:regions)if(r.bytes){
      check(r.base<=transport.address_limit&&r.bytes-1<=transport.address_limit-r.base,"tensor region exceeds system address width");
      check(r.base+r.bytes<=descriptor_address||r.base>=descriptor_address+descriptor_bytes,"tensor region overlaps live descriptor");
    }
    // The last descriptor beat commits on this edge. Backend local cycle zero
    // starts on the next edge, sharing the same monotonic transport clock.
    if(pair)pair_model=std::make_unique<PairRuntime>(*pair,options.matrix,options.vector,transport,tokens,cycle+1,launches);
    else addresses=std::make_unique<AddressSpacePort>(transport,tokens,regions,cycle+1);
    if(matrix)matrix_model=std::make_unique<matrix_schedule::Simulator>(matrix->program,matrix->a,matrix->b,matrix->has_bias?&matrix->bias:nullptr,matrix->transpose_b,
      matrix->a_batch,matrix->b_batch,matrix->m,matrix->n,matrix->k,matrix->output,matrix->output_batch,options.matrix,addresses.get());
    else if(vector)vector_model=std::make_unique<vector_schedule::Simulator>(vector->node,vector->values,vector->output,options.vector,addresses.get());
    else if(memory)memory_model=std::make_unique<memory_model::Simulator>(memory->node,memory->values,options.memory,addresses.get(),&memory->output);
    phase=Phase::Run;
  }
  bool engine_done()const{return matrix_model?matrix_model->done():vector_model?vector_model->done():pair_model?pair_model->done():memory_model->done();}
  void fetch(){
    if(auto response=transport.response()){
      check(!response->error,"descriptor memory response reported error");
      std::memcpy(descriptor.data()+fetched,&response->data,8);transport.consume_response();fetched+=8;
      if(fetched==8){uint64_t magic;std::memcpy(&magic,descriptor.data(),8);check(magic==expected_magic(),"descriptor magic/size mismatch");}
      if(fetched==descriptor_bytes){decode();return;}
    }
    if(transport.request_ready())transport.submit({tokens.take(),descriptor_address+fetched,0,8,false});
  }
  void check_output()const{
    if(pair){
      const auto &producer=pair->producer_output(),&consumer=pair->consumer.output;const auto &regions=pair->producer_regions();
      for(const auto &item:{std::make_pair(&producer,regions.back().base),std::make_pair(&consumer,pair->consumer.regions.back().base)}){
        auto width=tensor_model::element_bytes(item.first->type);for(uint64_t index=0;index<item.first->numel();++index)check(transport.produced(item.second+item.first->position(index)*width,unsigned(width)),"pair output lacks acknowledged stores from this launch");
      }
      return;
    }
    if(memory&&memory->view)return;
    const auto &out=matrix?matrix->output:vector?vector->output:memory->output;
    const auto &regions=matrix?matrix->regions:vector?vector->regions:memory->regions;
    auto base=matrix?regions[3].base:vector?regions[2].base:regions.back().base;
    auto width=tensor_model::element_bytes(out.type);
    auto count=matrix?matrix->m*matrix->n:out.numel(),first=matrix?matrix->output_batch*count:0;
    for(uint64_t index=0;index<count;++index)check(transport.produced(base+out.position(first+index)*width,unsigned(width)),"output lacks acknowledged stores from this launch");
  }
  void tick(const Inputs &input){
    check(cycle<UINT64_MAX,"clocked device system clock overflow");
    check(!(input.nack&&input.response),"simultaneous nack and response forbidden");
    transport.advance(cycle);
    // These are transport protocol violations, not recoverable kernel faults.
    // A caller must not turn a wrong identity into an arbitrary valid reply.
    if(input.nack)transport.queue.receive_nack(*input.nack);
    if(input.response)transport.answer(*input.response);
    if(input.memory_ready&&transport.queue.request_valid())transport.queue.accept_request();
    try{
      if(phase==Phase::Drain){
        ++drain_cycles;
        if(transport.response())transport.consume_response();
        if(transport.queue.idle()){release_decoded();phase=Phase::Failed;}
      }else if(busy()){
        // Count the fault-detection edge as well, even if the execution
        // budget prevents another descriptor or backend action on that edge.
        if(phase==Phase::Fetch)++fetch_cycles;else ++run_cycles;
        check(cycle-start<options.max_busy_cycles,"clocked device exceeded busy cycle limit");
        if(phase==Phase::Fetch){fetch();}
        else{
          if(!engine_done()){if(matrix_model)matrix_model->tick();else if(vector_model)vector_model->tick();else if(pair_model)pair_model->tick();else memory_model->tick();}
          if(engine_done()){
            check(transport.queue.idle(),"backend completed before transport drained");check_output();
            kernel=matrix_model?matrix_model->result():vector_model?vector_model->result():pair_model?pair_model->result():memory_model->result();
            release_models();release_decoded();phase=Phase::Done;
          }
        }
      }
    }catch(const std::exception &e){fail(e.what());}
    ++cycle;transport.advance(cycle);
  }
  Json::Value report()const{
    Json::Value r;r["classification"]="externally_clocked_descriptor_and_tensor_bus_controller_not_chipyard_validation";
    r["cycle"]=Json::UInt64(cycle);r["source_id"]=Json::UInt64(source);r["launches"]=Json::UInt64(launches);r["launch_start_cycle"]=Json::UInt64(start);
    r["fetch_cycles"]=Json::UInt64(fetch_cycles);r["run_cycles"]=Json::UInt64(run_cycles);r["drain_cycles"]=Json::UInt64(drain_cycles);
    r["descriptor_bytes_fetched"]=Json::UInt64(fetched);r["busy"]=busy();r["complete"]=complete();r["done"]=phase==Phase::Done;r["error"]=error;r["backend"]=backend;
    r["kernel"]=kernel;r["transport"]=transport.queue.snapshot();r["clock_driver"]="one_tick_per_caller_system_edge_status_reads_are_pure";
    r["full_model_execution_verified"]=false;r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;return r;
  }
  Json::Value progress()const{
    Json::Value r;static const char *names[]={"idle","fetch","run","drain","done","failed"};r["phase"]=names[unsigned(phase)];
    r["cycle"]=Json::UInt64(cycle);r["source_id"]=Json::UInt64(source);r["descriptor_bytes_fetched"]=Json::UInt64(fetched);r["descriptor_bytes"]=Json::UInt64(descriptor_bytes);
    r["busy"]=busy();r["done"]=phase==Phase::Done;r["error"]=error;r["backend"]=backend;r["memory_idle"]=transport.queue.idle();return r;
  }
};
Device::Device(Options options):impl(std::make_unique<Impl>(std::move(options))){}
Device::~Device()=default;
void Device::launch(uint64_t address,uint64_t bytes,uint64_t id){impl->launch(address,bytes,id);}
void Device::tick(const Inputs &inputs){impl->tick(inputs);}
std::optional<PhysicalRequest> Device::request()const{return impl->transport.queue.presented_request();}
bool Device::busy()const{return impl->busy();}
bool Device::complete()const{return impl->complete();}
Json::Value Device::report()const{return impl->report();}
Json::Value Device::progress()const{return impl->progress();}
void Device::reset(){
  check(!busy()&&impl->transport.queue.idle(),"reset requires drained device");
  impl->release_models();impl->release_decoded();impl->phase=Impl::Phase::Idle;impl->kernel=Json::Value();impl->error.clear();impl->backend.clear();
  impl->transport.written.clear(); // Request tokens intentionally never rewind.
}
}
