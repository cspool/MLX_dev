#include "matrix_wire.hh"
#include "vector_wire.hh"
#include "memory_wire.hh"
#include "mapped_memory.hh"
#include "mmio_plugin.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <openssl/evp.h>
#include <string>

namespace mlx::physical_device {
using namespace tensor_model;
using namespace model_io;
class MatrixDevice {
  uint64_t base=0,cycles=0,descriptor=0,launches=0,error=0,generation=0;
  uint64_t cpu_reads=0,cpu_writes=0,cpu_read_bytes=0,cpu_write_bytes=0;
  uint64_t device_reads=0,device_writes=0;
  bool busy=false,done=false,draining=false;
  std::string error_message;
  std::filesystem::path report_file;
  std::unique_ptr<MappedMemory> memory;
  Json::Value windows{Json::arrayValue};
  Json::Value expected_assets,observed_assets{Json::arrayValue};
  bool assets_audited=false;
  std::string asset_audit_error;
  matrix_schedule::Options timing;
  vector_schedule::Options vector_timing;
  memory_model::ScheduleOptions memory_timing;
  RequestTokens tokens;
  std::optional<DecodedMatrix> decoded;
  std::optional<DecodedVector> decoded_vector;
  std::optional<DecodedMemory> decoded_memory;
  struct Port final:PhysicalMemoryPort {
    MatrixDevice &device;unsigned latency;uint64_t now=0,due=0;
    std::optional<PhysicalRequest> pending;
    std::optional<Response> answer;
    Port(MatrixDevice &d,unsigned delay):device(d),latency(delay){}
    void advance(uint64_t cycle)override{
      require(cycle>=now,"plugin accelerator clock regressed");now=cycle;
      if(!pending||answer||cycle<due)return;
      auto &q=*pending;Response r{q.id,0,false};
      try{auto offset=device.offset(q.address,q.bytes);
        if(q.write){
          device.memory->write(offset,&q.data,q.bytes,device.generation);
          ++device.device_writes;
        }else{
          device.memory->read(offset,&r.data,q.bytes);++device.device_reads;
        }
      }catch(const std::exception &e){r.error=true;device.error_message=e.what();}
      answer=r;
    }
    bool request_ready()const override{return !pending&&!answer;}
    void submit(const PhysicalRequest &q)override{require(request_ready()&&now<=UINT64_MAX-latency,"plugin overcommitted request or deadline overflow");pending=q;due=now+latency;}
    std::optional<Response> response()const override{return answer;}
    void consume_response()override{require(bool(answer),"plugin consumed missing response");answer.reset();pending.reset();}
    bool idle()const{return !pending&&!answer;}
  };
  std::unique_ptr<Port> port;
  std::unique_ptr<AddressSpacePort> address_space;
  std::unique_ptr<matrix_schedule::Simulator> model;
  std::unique_ptr<vector_schedule::Simulator> vector_model;
  std::unique_ptr<memory_model::Simulator> transfer_model;

  void clear_models(){model.reset();vector_model.reset();transfer_model.reset();}
  void clear_decoded(){decoded.reset();decoded_vector.reset();decoded_memory.reset();}
  bool engine_done()const{return model?model->done():vector_model?vector_model->done():transfer_model->done();}
  void engine_tick(){if(model)model->tick();else if(vector_model)vector_model->tick();else transfer_model->tick();}

  uint64_t offset(uint64_t address,size_t bytes)const{
    require(address>=base+MLX_MATRIX_DATA_OFFSET,"plugin access before payload memory");auto at=address-base;
    require(at<=memory->size()&&bytes<=memory->size()-at,"plugin access outside mapped test memory");return at;
  }
  void output_written()const{
    if(decoded_memory&&decoded_memory->view)return; // checked alias, no newly produced bytes
    const auto &output=decoded?decoded->output:decoded_vector?decoded_vector->output:decoded_memory->output;auto bytes=element_bytes(output.type);
    auto count=decoded?decoded->m*decoded->n:output.numel();auto first=decoded?decoded->output_batch*count:0;
    auto base=decoded?decoded->regions[3].base:decoded_vector?decoded_vector->regions[2].base:decoded_memory->regions.back().base;
    for(uint64_t i=0;i<count;++i){
      auto at=offset(base+output.position(first+i)*bytes,bytes);
      memory->require_written(at,bytes,generation);
    }
  }
  void audit_assets(){
    if(expected_assets.isNull()||assets_audited)return;
    assets_audited=true;
    try{
      // Spike runs MMIO callbacks on a bounded coroutine stack. Keep the
      // diagnostic buffer on the heap even when this method is inlined.
      std::vector<uint8_t> chunk(65536);
      for(const auto &asset:expected_assets){
        auto at=offset(asset["base"].asUInt64(),asset["bytes"].asUInt64());auto left=asset["bytes"].asUInt64();
        auto digest=std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)>(EVP_MD_CTX_new(),EVP_MD_CTX_free);
        require(bool(digest)&&EVP_DigestInit_ex(digest.get(),EVP_sha256(),nullptr)==1,"asset audit digest initialization failed");
        while(left){auto count=std::min(left,uint64_t(chunk.size()));memory->read(at,chunk.data(),count);require(EVP_DigestUpdate(digest.get(),chunk.data(),size_t(count))==1,"asset audit digest failed");at+=count;left-=count;}
        unsigned char raw[EVP_MAX_MD_SIZE];unsigned size=0;require(EVP_DigestFinal_ex(digest.get(),raw,&size)==1&&size==32,"asset audit digest final failed");
        std::string hex;const char *digits="0123456789abcdef";for(unsigned i=0;i<size;++i){hex+=digits[raw[i]>>4];hex+=digits[raw[i]&15];}
        Json::Value row;row["value"]=asset["value"];row["base"]=asset["base"];row["bytes"]=asset["bytes"];row["sha256"]=hex;observed_assets.append(row);
        require(hex==asset["sha256"].asString(),"CPU-loaded asset bytes do not match input digest");
      }
    }catch(const std::exception &e){asset_audit_error=e.what();throw;}
  }
  void launch(){
    error=0;error_message.clear();done=false;++launches;
    try{
      audit_assets();require(asset_audit_error.empty(),"previous asset load audit failed");
      require(port->idle(),"plugin launch while memory remains active");require(descriptor%8==0,"plugin descriptor is misaligned");
      auto at=offset(descriptor,8);uint64_t magic;memory->read(at,&magic,8);
      require(magic==MLX_MATRIX_WIRE_MAGIC||magic==MLX_VECTOR_WIRE_MAGIC||magic==MLX_MEMORY_WIRE_MAGIC,"unknown plugin descriptor magic");
      auto size=magic==MLX_MATRIX_WIRE_MAGIC?sizeof(mlx_matrix_wire):magic==MLX_VECTOR_WIRE_MAGIC?sizeof(mlx_vector_wire):sizeof(mlx_memory_wire);offset(descriptor,size);
      if(magic==MLX_MATRIX_WIRE_MAGIC){mlx_matrix_wire wire;memory->read(at,&wire,sizeof(wire));decoded=decode_matrix(wire);}
      else if(magic==MLX_VECTOR_WIRE_MAGIC){mlx_vector_wire wire;memory->read(at,&wire,sizeof(wire));decoded_vector=decode_vector(wire);}
      else{mlx_memory_wire wire;memory->read(at,&wire,sizeof(wire));decoded_memory=decode_memory(wire);}
      const auto &regions=decoded?decoded->regions:decoded_vector?decoded_vector->regions:decoded_memory->regions;
      for(const auto &region:regions)if(region.bytes){offset(region.base,region.bytes);require(region.base+region.bytes<=descriptor||region.base>=descriptor+size,"device data overlaps its descriptor");}
      address_space=std::make_unique<AddressSpacePort>(*port,tokens,regions,cycles);
      if(decoded)model=std::make_unique<matrix_schedule::Simulator>(decoded->program,decoded->a,decoded->b,decoded->has_bias?&decoded->bias:nullptr,decoded->transpose_b,
        decoded->a_batch,decoded->b_batch,decoded->m,decoded->n,decoded->k,decoded->output,decoded->output_batch,timing,address_space.get());
      else if(decoded_vector)vector_model=std::make_unique<vector_schedule::Simulator>(decoded_vector->node,decoded_vector->values,decoded_vector->output,vector_timing,address_space.get());
      else transfer_model=std::make_unique<memory_model::Simulator>(decoded_memory->node,decoded_memory->values,memory_timing,address_space.get(),&decoded_memory->output);
      require(generation!=UINT64_MAX,"plugin output generation exhausted");++generation;busy=true;
    }catch(const std::exception &e){error=1;error_message=e.what();clear_models();address_space.reset();clear_decoded();busy=false;done=false;}
  }
  void step(){
    if(!busy)return;
    if(draining){
      port->advance(cycles++);if(port->response())port->consume_response();
      if(port->idle()){draining=false;busy=false;address_space.reset();clear_decoded();}return;
    }
    try{
      if(!engine_done()){engine_tick();++cycles;}
      if(engine_done()){
        require(port->idle(),"plugin kernel completed with a memory request in flight");output_written();auto report=model?model->result():vector_model?vector_model->result():transfer_model->result();report["launch"]=Json::UInt64(launches);report["backend"]=model?"matrix":vector_model?"vector":"memory";windows.append(report);
        clear_models();address_space.reset();clear_decoded();busy=false;done=true;
      }
    }catch(const std::exception &e){
      ++cycles;error=2;error_message=e.what();clear_models();draining=!port->idle();busy=draining;
      if(!draining){address_space.reset();clear_decoded();}
    }
  }
  Json::Value report()const{
    Json::Value r;r["classification"]="spike_cpu_and_compute_backends_mapped_test_memory_not_chipyard_timing";r["windows"]=windows;
    r["launches"]=Json::UInt64(launches);r["device_cycles"]=Json::UInt64(cycles);r["cpu_payload_reads"]=Json::UInt64(cpu_reads);r["cpu_payload_writes"]=Json::UInt64(cpu_writes);r["cpu_payload_read_bytes"]=Json::UInt64(cpu_read_bytes);r["cpu_payload_write_bytes"]=Json::UInt64(cpu_write_bytes);
    r["device_reads"]=Json::UInt64(device_reads);r["device_writes"]=Json::UInt64(device_writes);r["busy"]=busy;r["done"]=done;r["error"]=Json::UInt64(error);r["error_message"]=error_message;r["memory_idle"]=port->idle();
    r["clock_driver"]="one_backend_tick_per_cpu_status_load_not_cpu_cycle_coupling";r["memory_domain"]="plugin_mapped_test_memory_not_hellacache";
    r["host_memory_backing"]=memory->snapshot();
    if(!expected_assets.isNull()){
      r["asset_load_audit"]["classification"]="host_diagnostic_readback_of_cpu_stores_not_timed_device_dma";
      r["asset_load_audit"]["attempted"]=assets_audited;r["asset_load_audit"]["error"]=asset_audit_error;r["asset_load_audit"]["assets"]=observed_assets;
    }
    r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;return r;
  }
public:
  explicit MatrixDevice(const std::string &path){
    std::ifstream input(path);Json::Value config;input>>config;base=config["base"].asUInt64();report_file=config["report"].asString();
    auto size=config.get("bytes",65536).asUInt64();require(base%4096==0&&base<(UINT64_C(1)<<40)&&size>=8192&&size%4096==0&&size<=(UINT64_C(1)<<40)-base,"invalid plugin memory map");
    memory=std::make_unique<MappedMemory>(size);
    expected_assets=config["loaded_assets"];
    if(!expected_assets.isNull()){
      require(expected_assets.isArray(),"loaded_assets must be an array");
      for(const auto &asset:expected_assets){require(asset["base"].isUInt64()&&asset["bytes"].isUInt64()&&asset["value"].isString()&&asset["sha256"].isString()&&asset["sha256"].asString().size()==64,"invalid asset audit descriptor");offset(asset["base"].asUInt64(),asset["bytes"].asUInt64());}
    }
    timing=matrix_schedule::Options::parse(config["matrix_options"]);unsigned latency=config.get("memory_latency",2).asUInt();require(latency&&latency<=1024,"invalid plugin memory latency");port=std::make_unique<Port>(*this,latency);
    vector_timing=vector_schedule::Options::parse(config["vector_options"]);
    memory_timing=memory_model::ScheduleOptions::parse(config.get("memory_options",Json::Value(Json::objectValue)));
  }
  ~MatrixDevice(){
    try{audit_assets();}catch(const std::exception &e){error=3;error_message=e.what();}
    std::filesystem::create_directories(report_file.parent_path());std::ofstream output(report_file);output<<report()<<'\n';
  }
  bool load(reg_t at,size_t bytes,uint8_t *data){
    try{
      if(bytes!=1&&bytes!=2&&bytes!=4&&bytes!=8)return false;
      if(at%bytes)return false;
      if(at>=MLX_MATRIX_DATA_OFFSET){
        if(busy)return false;
        auto position=offset(base+at,bytes);memory->read(position,data,bytes);++cpu_reads;cpu_read_bytes+=bytes;return true;
      }
      if(bytes!=8)return false;
      uint64_t value=0;
      if(at==MLX_MATRIX_REG_ID)value=MLX_MATRIX_WIRE_MAGIC;
      else if(at==MLX_MATRIX_REG_DESCRIPTOR)value=descriptor;
      else if(at==MLX_MATRIX_REG_STATUS){step();value=(busy?MLX_MATRIX_STATUS_BUSY:0)|(done?MLX_MATRIX_STATUS_DONE:0)|(error?MLX_MATRIX_STATUS_ERROR:0);}
      else if(at==MLX_MATRIX_REG_CYCLES)value=cycles;
      else if(at==MLX_MATRIX_REG_ERROR)value=error;
      else return false;
      std::memcpy(data,&value,8);return true;
    }catch(const std::exception &e){error_message=e.what();return false;}
  }
  bool store(reg_t at,size_t bytes,const uint8_t *data){
    try{
      if(bytes!=1&&bytes!=2&&bytes!=4&&bytes!=8)return false;
      if(at%bytes||busy)return false;
      if(at>=MLX_MATRIX_DATA_OFFSET){auto position=offset(base+at,bytes);memory->write(position,data,bytes);++cpu_writes;cpu_write_bytes+=bytes;return true;}
      if(bytes!=8)return false;
      uint64_t value;std::memcpy(&value,data,8);
      if(at==MLX_MATRIX_REG_DESCRIPTOR){descriptor=value;return true;}
      if(at==MLX_MATRIX_REG_LAUNCH&&value==1){launch();return true;}
      return false;
    }catch(const std::exception &e){error_message=e.what();return false;}
  }
};
static mmio_plugin_registration_t<MatrixDevice> registration("mlx_matrix");
} // namespace mlx::physical_device
