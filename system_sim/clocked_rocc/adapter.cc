#include "adapter.hh"
#include <stdexcept>

namespace mlx::clocked_rocc {
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
clocked_device::Options configured(unsigned bits,clocked_device::Options options){options.address_bits=bits;return options;}
}
Adapter::Adapter(unsigned bits,unsigned tags,clocked_device::Options options):device(configured(bits,std::move(options))),tag_bits(tags){check(tags>=1&&tags<=16,"unsupported RoCC request tag width");}
uint64_t Adapter::status(uint64_t index)const{
  if(index==14)return ABI_MAGIC;
  const auto state=device.report();
  if(index==0)return uint64_t(device.busy())|uint64_t(state["done"].asBool()&&frontend_error.empty())<<1|uint64_t(!frontend_error.empty()||!state["error"].asString().empty())<<2;
  if(index==1)return state["cycle"].asUInt64();
  if(index==2)return state["run_cycles"].asUInt64();
  if(index==3)return state["fetch_cycles"].asUInt64();
  if(index==4)return state["drain_cycles"].asUInt64();
  if(index==5)return state["descriptor_bytes_fetched"].asUInt64();
  if(index==6)return requests;
  if(index==7)return responses;
  if(index==15)return !frontend_error.empty()||!state["error"].asString().empty();
  if(index==16)return launches;
  return UINT64_MAX; // Unregistered query, no mutation of an active kernel.
}
Outputs Adapter::eval(const Inputs &in)const{
  Outputs out;out.response_valid=response_valid;out.rd=response_rd;out.data=response_data;
  out.busy=device.busy()||response_valid||bool(owner);out.privilege=privilege;
  if(!response_valid||in.response_ready){
    if(in.funct==3)out.command_ready=true;
    else if(in.funct==2)out.command_ready=device.complete()||!frontend_error.empty();
    else out.command_ready=!device.busy()&&!owner;
  }
  if(auto q=device.request()){
    out.memory_valid=true;out.memory_write=q->write;out.memory_address=q->address;out.memory_data=q->data;
    // The HellaCache requestor supplies StoreGen-style replicated data.
    // SimpleHellaCacheIF only registers it; DCache then selects byte lanes
    // with the address/size mask. A low-lane logical F16 value is insufficient
    // for stores at byte offsets 2, 4 or 6 of a cache word.
    if(q->write&&q->bytes<8){
      auto mask=(UINT64_C(1)<<(q->bytes*8))-1;auto value=q->data&mask;out.memory_data=0;
      for(unsigned byte=0;byte<8;byte+=q->bytes)out.memory_data|=value<<(byte*8);
    }
    out.memory_tag=unsigned(q->id&((UINT64_C(1)<<tag_bits)-1));
    out.memory_size=q->bytes==8?3:q->bytes==4?2:q->bytes==2?1:0;
    out.memory_mask=((1u<<q->bytes)-1u)<<unsigned(q->address&7);
  }
  return out;
}
void Adapter::tick(const Inputs &in){
  const auto out=eval(in);clocked_device::Inputs edge;edge.memory_ready=in.memory_ready;
  if(in.memory_response_valid){
    check(owner&&owner->wire==in.memory_response_tag,"RoCC response has no matching accepted tag");
    edge.response=model_io::Response{owner->logical,in.memory_response_data,false};owner.reset();++responses;
  }
  if(out.memory_valid&&in.memory_ready){
    check(!owner,"RoCC overcommitted its request identity slot");auto q=device.request();check(bool(q),"RoCC request disappeared");
    owner=Owner{q->id,out.memory_tag};++requests;
  }
  if(response_valid&&in.response_ready)response_valid=false;
  if(in.command_valid&&out.command_ready){
    ++commands;uint64_t value=0;
    if(in.funct==1){
      frontend_error.clear();++launches;
      try{
        check(in.privilege==3,"clocked RoCC profile currently requires machine mode");
        device.launch(in.rs1,in.rs2,launches-1);active_launch=true;privilege=in.privilege;
      }catch(const std::exception &e){
        frontend_error=e.what();Json::Value failed;failed["frontend_error"]=frontend_error;failed["source_id"]=Json::UInt64(launches-1);windows.append(failed);
      }
      value=status(0);
    }else if(in.funct==2)value=status(0);
    else if(in.funct==3)value=status(in.rs1);
    else{frontend_error="unregistered clocked RoCC function";value=4;}
    if(in.xd){response_valid=true;response_rd=in.rd;response_data=value;}
  }
  // CPU polling has no special clock action. Every real external edge calls
  // the device once, including edges when the host is stalled on WAIT.
  device.tick(edge);
  if(active_launch&&device.complete()){
    check(!owner,"RoCC completion still owns a cache transaction");windows.append(device.report());active_launch=false;
  }
}
void Adapter::reset(){
  check(!owner&&!device.busy(),"clocked RoCC reset requires a drained cache/device");device.reset();
  response_valid=false;frontend_error.clear();active_launch=false;
}
Json::Value Adapter::report()const{
  Json::Value r;r["classification"]="clocked_cpp_rocc_requestor_adapter_execution_scope_requires_external_evidence";
  r["device"]=device.report();r["windows"]=windows;r["frontend_error"]=frontend_error;r["commands"]=Json::UInt64(commands);r["launches"]=Json::UInt64(launches);
  r["requests"]=Json::UInt64(requests);r["responses"]=Json::UInt64(responses);r["cache_request_owned"]=bool(owner);r["cpu_response_pending"]=response_valid;r["tag_bits"]=tag_bits;
  r["cache_replay_owner"]="Chipyard_SimpleHellaCacheIF_not_a_second_cpp_retry_loop";
  r["source_id_basis"]="transport_launch_ordinal_not_model_operator_identity";
  r["full_model_execution_verified"]=false;r["mlx_system_verified"]=false;r["inference_performance_eligible"]=false;return r;
}
Json::Value Adapter::progress()const{
  auto r=device.progress();r["launches"]=Json::UInt64(launches);r["terminal_windows"]=Json::UInt64(windows.size());r["commands"]=Json::UInt64(commands);
  r["requests"]=Json::UInt64(requests);r["responses"]=Json::UInt64(responses);r["cache_request_owned"]=bool(owner);r["cpu_response_pending"]=response_valid;r["frontend_error"]=frontend_error;return r;
}
}
