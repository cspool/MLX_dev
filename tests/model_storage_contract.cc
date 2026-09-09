#include "buffer_arena.h"
#include "memory_program.h"
#include "source_groups.h"
#include "result_contract.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>

using namespace mlx::tensor_model;
using namespace mlx::model_storage;
namespace {
Json::Value info(Allocation a){Json::Value r;r["id"]=Json::UInt64(a.id);r["base"]=Json::UInt64(a.base);r["bytes"]=Json::UInt64(a.bytes);r["reserved_bytes"]=Json::UInt64(a.reserved_bytes);r["writable"]=a.writable;return r;}
void builtin(){
  Arena arena(uint64_t(1)<<32,256);auto original=arena.allocate(DType::F32,{17});auto initial=arena.allocation(original);
  require(!original.storage->data&&!original.storage->writable,"physical allocation acquired host data");
  Tensor alias=original;alias.sizes={4};alias.steps={2};alias.offset=1;original={};
  require(arena.allocation(alias).id==initial.id,"view lost allocation owner");
  auto pin=arena.pin(alias);auto copy=pin;alias={};pin={};
  auto other=arena.allocate(DType::F32,{32});require(arena.allocation(other).base!=initial.base,"pending read address reused");
  require(arena.snapshot()["allocations"].size()==2,"pin did not preserve allocation");
  copy={};auto reused=arena.allocate(DType::F32,{17});auto next=arena.allocation(reused);
  require(next.base==initial.base&&next.id!=initial.id,"drained allocation did not safely reuse address with fresh identity");
  reused={};other={};require(arena.snapshot()["free_ranges"].size()==1&&arena.snapshot()["free_bytes"].asUInt64()==256,"free ranges failed to coalesce");
  Arena::Pin survivor;uint64_t owned_id=0;
  {Arena temporary(uint64_t(2)<<32,128);auto tensor=temporary.allocate(DType::I64,{8});owned_id=temporary.allocation(tensor).id;survivor=temporary.pin(tensor,true);}
  require(survivor.allocation().id==owned_id&&survivor.region().writable,"pin did not outlive Arena wrapper");survivor={};
}
Json::Value model_lifetimes(const Json::Value &program){
  validate_value_contract(program);validate_source_groups(program);validate_result_contract(program);
  Arena arena(uint64_t(1)<<32,uint64_t(16)<<30);Values values;std::set<std::string> used_assets;
  for(const auto &name:program["assets"].getMemberNames()){
    const auto &asset=program["assets"][name];values.emplace(name,arena.allocate(dtype(asset["dtype"].asString()),shape(asset["shape"]),false));
  }
  auto initial=arena.snapshot();Json::Value events(Json::arrayValue),guards(Json::arrayValue);uint64_t views=0,materializations=0;
  for(const auto &node:program["nodes"]){
    require(node.get("control_dependencies",Json::Value(Json::arrayValue))==guards,"missing or foreign lifetime guard dependency");
    Json::Value event;event["source_operator_id"]=node["source_operator_id"];event["kind"]=node["kind"];
    {
      std::set<std::string> inputs;
      auto collect=[&](auto &&self,const Json::Value &v)->void{
        if(v.isObject()){if(v.isMember("value"))inputs.insert(v["value"].asString());else for(const auto &key:v.getMemberNames())self(self,v[key]);}
        else if(v.isArray())for(const auto &child:v)self(self,child);
      };
      collect(collect,node["args"]);collect(collect,node["kwargs"]);collect(collect,node["control_dependencies"]);
      std::vector<Arena::Pin> pins;
      for(const auto &name:inputs){auto found=values.find(name);require(found!=values.end(),"model uses an unbound/released physical SSA value");pins.push_back(arena.pin(found->second));if(program["assets"].isMember(name))used_assets.insert(name);}
      auto name=node["id"].asString();require(!values.count(name),"model redefines a live physical SSA value");unsigned routes=0;
      for(const char *field:{"matrix_program","vector_program","memory_program","control_program"})routes+=node.isMember(field);
      require(routes==1,"model physical plan requires exactly one executable route");
      Tensor output;
      if(node["memory_program"]["mode"]=="view"){
        mlx::memory_model::Stats stats;output=mlx::memory_model::execute(node,values,stats);require(stats.views==1&&stats.allocations==0,"physical view unexpectedly materialized data");++views;
      }else{
        output=arena.allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));
        if(node.isMember("memory_program"))output.steps=shape(node["memory_program"]["output_layout"]["strides"]);
        pins.push_back(arena.pin(output,true));++materializations;
      }
      event["allocation"]=info(arena.allocation(output));
      if(node["kind"]=="split"){
        for(auto &[id,value]:split_views(node,output)){
          event["value_allocations"][id]=info(arena.allocation(value));
          require(values.emplace(id,std::move(value)).second,"lifetime split output was redefined");
        }
      }else values.emplace(name,output);
      // Deliberately erase SSA owners while pins are retained, as could happen
      // during asynchronous completion. No physical address may recycle yet.
      bool owner_released=false;
      for(const auto &released:node["release"]){
        if(node["kind"]=="split"&&released==name){require(!owner_released,"split container released twice");owner_released=true;continue;}
        require(values.erase(released.asString())==1,"model releases a missing physical SSA value "+released.asString()+" after "+name);
      }
      for(const auto &pin:pins)require(pin.allocation().id==arena.allocation(pin.tensor()).id,"physical allocation recycled before pin drain");
    }
    auto state=arena.snapshot();event["live_bytes_after_drain"]=state["live_bytes"];event["reserved_bytes_after_drain"]=state["reserved_bytes"];event["live_allocations_after_drain"]=state["allocations"].size();events.append(event);
    if(node["kind"]=="guard"){Json::Value dep;dep["value"]=node["id"];guards.append(dep);}
  }
  for(const auto &output:program["outputs"])for(const auto &field:result_roles(program))require(values.count(output[field].asString()),"model result storage was released before host observation");
  auto final=arena.snapshot();values.clear();auto drained=arena.snapshot();require(drained["reserved_bytes"].asUInt64()==0&&drained["allocations"].empty()&&drained["free_ranges"].size()==1,"model storage owners leaked");
  Json::Value report;report["classification"]="full_model_address_lifetime_replay_not_inference_execution";
  report["initial"]=initial;report["before_result_release"]=final;report["drained"]=drained;report["events"]=events;
  report["source_nodes"]=program["nodes"].size();report["checked_views"]=Json::UInt64(views);report["allocating_nodes"]=Json::UInt64(materializations);report["referenced_assets"]=Json::Value(Json::arrayValue);
  for(const auto &name:used_assets)report["referenced_assets"].append(name);
  report["model_data_loaded"]=false;report["model_inference_executed"]=false;report["mlx_system_verified"]=false;report["inference_performance_eligible"]=false;return report;
}
}
int main(int argc,char **argv){
  try{
    builtin();if(argc==1){std::cout<<"MODEL_STORAGE_CONTRACT_PASS"<<std::endl;return 0;}
    require(argc==3,"usage: model-storage-contract [job.json output.json]");std::ifstream input(argv[1]);Json::Value job;input>>job;
    if(job.isMember("schema")){
      auto report=model_lifetimes(job);auto output=std::filesystem::path(argv[2]);std::filesystem::create_directories(output.parent_path());std::ofstream stream(output);stream<<report<<'\n';require(bool(stream),"cannot write model lifetime report");std::cout<<"MODEL_LIFETIME_REPLAY_PASS"<<std::endl;return 0;
    }
    Arena arena(job["base"].asUInt64(),job["bytes"].asUInt64(),job.get("alignment",64).asUInt64());
    std::map<std::string,Tensor> values;std::map<std::string,Arena::Pin> pins;Json::Value events(Json::arrayValue);
    for(const auto &action:job["actions"]){
      auto before=arena.snapshot();Json::Value event;const auto op=action["op"].asString(),name=action["name"].asString();event["op"]=op;event["name"]=name;
      bool failed=false;
      try{
        if(op=="allocate"){
          require(!values.count(name),"duplicate tensor name");auto t=arena.allocate(dtype(action.get("dtype","f32").asString()),shape(action["shape"]),action.get("writable",true).asBool());
          event["allocation"]=info(arena.allocation(t));values.emplace(name,std::move(t));
        }else if(op=="view"){
          require(!values.count(name),"duplicate tensor name");auto t=values.at(action["source"].asString());
          if(action.isMember("shape"))t.sizes=shape(action["shape"]);
          if(action.isMember("strides"))t.steps=shape(action["strides"]);
          if(action.isMember("offset"))t.offset=action["offset"].asInt64();
          event["allocation"]=info(arena.allocation(t));values.emplace(name,std::move(t));
        }else if(op=="release")require(values.erase(name)==1,"missing tensor release");
        else if(op=="pin"){
          require(!pins.count(name),"duplicate pin name");auto p=arena.pin(values.at(action["source"].asString()),action.get("write",false).asBool());event["allocation"]=info(p.allocation());pins.emplace(name,std::move(p));
        }else if(op=="unpin")require(pins.erase(name)==1,"missing pin release");
        else if(op=="seal")arena.seal_read_only(values.at(name));
        else if(op=="foreign"){
          auto t=Tensor::allocate(DType::F32,{1});arena.pin(t);
        }else if(op=="clone_storage"){
          auto t=values.at(action["source"].asString());t.storage=std::make_shared<Storage>(*t.storage);arena.pin(t);
        }else throw std::runtime_error("unknown storage test operation");
      }catch(const std::exception &error){failed=true;event["error"]=error.what();require(action.isMember("error")&&std::string(error.what()).find(action["error"].asString())!=std::string::npos,"unexpected storage failure: "+std::string(error.what()));require(arena.snapshot()==before,"failed storage operation changed arena state");}
      require(failed==action.isMember("error"),"expected storage failure did not occur");event["state"]=arena.snapshot();events.append(event);
    }
    Json::Value result;result["events"]=events;result["final"]=arena.snapshot();result["mlx_system_verified"]=false;result["inference_performance_eligible"]=false;
    auto output=std::filesystem::path(argv[2]);std::filesystem::create_directories(output.parent_path());std::ofstream stream(output);stream<<result<<'\n';require(bool(stream),"cannot write storage contract result");std::cout<<"MODEL_STORAGE_CONTRACT_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"MODEL_STORAGE_CONTRACT_FAIL: "<<error.what()<<std::endl;return 1;}
}
