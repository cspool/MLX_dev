#include "progress.hh"
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace mlx::clocked_rocc {
Progress::Progress(std::filesystem::path output,Json::Value mapping,uint64_t interval,uint64_t maximum):path(std::move(output)),map(std::move(mapping)),period(interval),limit(maximum){
  if(path.empty()||!map.isArray()||map.size()>1000000||!period||period>1000000000000ULL||!limit||limit>1000000)throw std::runtime_error("invalid bounded system progress configuration");
  const std::set<std::string> allowed={"launch_ordinal","source_operator_id","source_ordinal","forward_id","layer_idx","phase","kind","family","batch_index","batch_count","shape","dtype","consumer"};
  const std::set<std::string> consumer_fields={"source_operator_id","source_ordinal","forward_id","layer_idx","phase","kind","shape","dtype"};
  for(const auto &row:map){
    if(!row.isObject()||row.toStyledString().size()>2048)throw std::runtime_error("system progress map row exceeds metadata contract");
    for(const auto &key:row.getMemberNames())if(!allowed.count(key))throw std::runtime_error("unexpected system progress map field");
    if(row["family"]=="pair"||row.isMember("consumer")){
      const auto &c=row["consumer"];
      if(row["family"]!="pair"||!c.isObject()||!row["source_operator_id"].isUInt64()||!row["source_ordinal"].isUInt64()
          ||!c["source_operator_id"].isUInt64()||!c["source_ordinal"].isUInt64()
          ||c["source_operator_id"].asUInt64()<=row["source_operator_id"].asUInt64()||c["source_ordinal"].asUInt64()<=row["source_ordinal"].asUInt64()
          ||!c["shape"].isArray()||c["shape"].size()>8||!c["kind"].isString()||!c["dtype"].isString())throw std::runtime_error("invalid pair system progress metadata");
      for(const auto &key:c.getMemberNames())if(!consumer_fields.count(key))throw std::runtime_error("unexpected pair system progress field");
      for(const auto &dimension:c["shape"])if(!dimension.isUInt64())throw std::runtime_error("invalid pair system progress shape");
    }
  }
}
void Progress::sample(const Adapter &adapter,bool launched,bool terminal,bool final){
  ++ticks;if(failed||(!launched&&!terminal&&!final&&ticks<next))return;next=ticks+period;
  try{
    auto r=adapter.progress();r["classification"]="read_only_system_progress_not_completion_certificate";r["event_index"]=Json::UInt64(events++);r["launch_event"]=launched;r["terminal_event"]=terminal;
    r["final_snapshot"]=final;
    r["host_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    auto count=adapter.launch_count();if(count&&count<=map.size())r["source"]=map[Json::ArrayIndex(count-1)];else r["source"]=Json::Value();
    r["event_log_truncated"]=emitted>=limit;
    auto temporary=path;temporary+=".tmp";
    {std::ofstream output(temporary);output<<r<<'\n';if(!output)throw std::runtime_error("cannot write system progress snapshot");}
    std::filesystem::rename(temporary,path);
    if(emitted<limit){auto log=path;log+="l";Json::StreamWriterBuilder writer;writer["indentation"]="";std::ofstream output(log,std::ios::app);output<<Json::writeString(writer,r)<<'\n';if(!output)throw std::runtime_error("cannot append system progress event");++emitted;}
  }catch(const std::exception &error){failed=true;failure=error.what();std::cerr<<"MLX_PROGRESS_OBSERVER_ERROR "<<failure<<'\n';}
}
Json::Value Progress::status()const{
  Json::Value r;r["events_observed"]=Json::UInt64(events);r["events_logged"]=Json::UInt64(emitted);r["event_limit"]=Json::UInt64(limit);r["period_cycles"]=Json::UInt64(period);r["failed"]=failed;r["error"]=failure;return r;
}
}
