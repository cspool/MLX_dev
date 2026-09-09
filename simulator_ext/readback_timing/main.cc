// Exact event recurrence for ready_graph.cc's post-drain scalar readback loop.
// No numerical data, CPU postprocessing, initialization, cache or retry model.
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
using U=uint64_t;
void need(bool p,const char *message){if(!p)throw std::runtime_error(message);}
U number(const Json::Value &v){need(v.isUInt64()&&!v.isBool(),"nonnegative integer required");return v.asUInt64();}
U add(U a,U b){need(b<=std::numeric_limits<U>::max()-a,"cycle/count overflow");return a+b;}
int main(int argc,char **argv){try{
  need(argc==3,"usage: mlx-readback-timing input.json output.json");
  Json::Value p;std::ifstream input(argv[1]);need(bool(input),"missing input");input>>p;
  need(p["schema"]=="mlx_post_graph_readback_v1","unknown readback contract");
  U start=number(p["graph_cycles"]),cycle=start,latency=number(p["memory"]["latency"]),period=number(p["memory"]["accept_period"]);
  need(period>0&&latency>0,"positive memory latency/accept period required");
  need(number(p["memory"]["nack_every"])==0,"retry readback not implemented");
  need(p["reads"].isArray(),"read descriptors required");
  U requests=0,bytes=0;Json::Value intervals(Json::arrayValue);
  for(const auto &read:p["reads"]){
    U n=number(read["elements"]),width=number(read["element_bytes"]),begin=cycle;
    need(width==1||width==2||width==4||width==8,"invalid scalar read width");
    for(U i=0;i<n;++i){
      // Submit at S; registered queue first visible at S+1. Physical accept
      // is globally period-aligned. Response consumption increments the clock.
      U visible=add(cycle,1),remainder=visible%period;
      U accept=add(visible,remainder?period-remainder:0);
      cycle=add(add(accept,latency),1);requests=add(requests,1);bytes=add(bytes,width);
    }
    Json::Value row=read;row["begin_cycle"]=Json::UInt64(begin);row["end_cycle"]=Json::UInt64(cycle);intervals.append(row);
  }
  Json::Value result;result["classification"]="post_graph_registered_scalar_readback_timing_not_tensor_execution";
  result["graph_cycles"]=Json::UInt64(start);result["host_readback_cycles"]=Json::UInt64(cycle-start);
  result["shared_elapsed_cycles"]=Json::UInt64(cycle);result["host_readback_requests"]=Json::UInt64(requests);
  result["readback_bytes"]=Json::UInt64(bytes);result["read_intervals"]=intervals;
  result["tensor_values_executed"]=false;result["cpu_postprocessing_timed"]=false;result["mlx_system_verified"]=false;
  std::ofstream output(argv[2]);output<<result;need(bool(output),"cannot write result");return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
