#include "profile.hh"
#include <set>
#include <stdexcept>

namespace mlx::model_image {
namespace {
void check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
Json::Value encode(const matrix_schedule::Options &o){
  Json::Value r;
#define ITEM(name) r[#name]=o.name
  ITEM(rows);ITEM(columns);ITEM(contexts);ITEM(dma_latency);ITEM(spm_latency);ITEM(multiply_latency);ITEM(add_latency);ITEM(convert_latency);
  ITEM(dma_request_period);ITEM(dma_response_period);ITEM(spm_period);ITEM(writeback_period);ITEM(compute_ii);ITEM(trace_limit);ITEM(overlap);ITEM(trace);ITEM(inject_stale_dma_epoch);ITEM(cache_control);
#undef ITEM
  r["max_cycles"]=Json::UInt64(o.max_cycles);return r;
}
Json::Value encode(const vector_schedule::Options &o){
  Json::Value r;
#define ITEM(name) r[#name]=o.name
  ITEM(rows);ITEM(columns);ITEM(contexts);ITEM(dma_latency);ITEM(spm_latency);ITEM(multiply_latency);ITEM(add_latency);ITEM(convert_latency);ITEM(exp_latency);ITEM(div_latency);ITEM(sqrt_latency);
  ITEM(dma_request_period);ITEM(dma_response_period);ITEM(spm_period);ITEM(writeback_period);ITEM(vector_ii);ITEM(trans_ii);ITEM(trace_limit);ITEM(overlap);ITEM(trace);ITEM(inject_stale_response);
#undef ITEM
  r["max_cycles"]=Json::UInt64(o.max_cycles);return r;
}
Json::Value encode(const memory_model::ScheduleOptions &o){
  Json::Value r;r["dma_latency"]=o.dma_latency;r["convert_latency"]=o.convert_latency;r["request_period"]=o.request_period;r["response_period"]=o.response_period;r["trace_limit"]=o.trace_limit;r["trace"]=o.trace;r["max_cycles"]=Json::UInt64(o.max_cycles);return r;
}
}
Profile parse_profile(const Json::Value &value){
  Profile profile;
  if(value.isNull()){
    profile.options.matrix.rows=profile.options.matrix.columns=profile.options.vector.rows=profile.options.vector.columns=1;
    profile.options.matrix.trace=profile.options.vector.trace=profile.options.memory.trace=false;
  }else{
    check(value.isObject(),"system profile must be an object");
    const std::set<std::string> keys={"version","name","max_busy_cycles","matrix_options","vector_options","memory_options"};
    for(const auto &key:value.getMemberNames())check(keys.count(key),"unknown system profile field");
    check(value["version"].isUInt()&&value["version"].asUInt()==1&&value["name"].isString()&&!value["name"].asString().empty(),"invalid system profile identity");
    check(value["max_busy_cycles"].isUInt64()&&value["max_busy_cycles"].asUInt64()>0,"invalid system busy cycle limit");profile.options.max_busy_cycles=value["max_busy_cycles"].asUInt64();
    for(const auto *key:{"matrix_options","vector_options","memory_options"})check(value[key].isObject(),"system profile backend options missing");
    profile.options.matrix=matrix_schedule::Options::parse(value["matrix_options"]);profile.options.vector=vector_schedule::Options::parse(value["vector_options"]);profile.options.memory=memory_model::ScheduleOptions::parse(value["memory_options"]);
  }
  const auto &m=profile.options.matrix;const auto &v=profile.options.vector;
  check(m.rows==v.rows&&m.columns==v.columns&&m.contexts==v.contexts,"registered system profile requires one common PE/context geometry");
  check(!m.inject_stale_dma_epoch&&!v.inject_stale_response,"system execution profile cannot inject stale test responses");
  auto &out=profile.effective;out["version"]=1;out["name"]=value.isNull()?"demo-1pe":value["name"];out["max_busy_cycles"]=Json::UInt64(profile.options.max_busy_cycles);
  out["matrix_options"]=encode(m);out["vector_options"]=encode(v);out["memory_options"]=encode(profile.options.memory);return profile;
}
}
