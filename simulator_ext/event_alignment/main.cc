#include "matrix_schedule.h"
#include "vector_schedule.h"
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace mlx;
using namespace mlx::tensor_model;
struct Window {
  Tensor a,b,bias,output;Values values;
  std::unique_ptr<matrix_schedule::Simulator> matrix;
  std::unique_ptr<vector_schedule::Simulator> vector;
  bool done()const{return matrix?matrix->done():vector->done();}
  void tick(){if(matrix)matrix->tick();else vector->tick();}
  Json::Value result()const{return matrix?matrix->result():vector->result();}
};
struct Source {uint64_t id=0,begin=UINT64_MAX,end=UINT64_MAX;unsigned current=0;bool active=false,complete=false;std::vector<unsigned> windows,parents;Json::Value intervals{Json::arrayValue};uint64_t window_begin=0;};
int main(int argc,char **argv){
  try{
    require(argc==3,"usage: mlx-event-native-group job.json output-directory");Json::Value job;std::ifstream input(argv[1]);input>>job;
    require(job["schema"]=="mlx_native_array_group_v1"&&job["windows"].isArray()&&job["windows"].size()>=2&&job["windows"].size()<=32,"invalid native concurrent group");
    shared_array::Hardware h;const auto &config=job["hardware"];
    std::map<std::string,unsigned*> fields={{"rows",&h.rows},{"columns",&h.columns},{"contexts",&h.contexts},{"spm_period",&h.spm_period},{"writeback_period",&h.writeback_period},
      {"compute_ii",&h.compute_ii},{"sfu_ii",&h.sfu_ii},{"dma_request_period",&h.dma_request_period},{"dma_response_period",&h.dma_response_period},
      {"multiply_latency",&h.multiply_latency},{"add_latency",&h.add_latency},{"convert_latency",&h.convert_latency},{"spm_latency",&h.spm_latency},
      {"exp_latency",&h.exp_latency},{"div_latency",&h.div_latency},{"sqrt_latency",&h.sqrt_latency}};
    require(config.isObject(),"missing native group hardware");for(const auto &name:config.getMemberNames()){require(fields.count(name)&&config[name].isUInt(),"invalid native hardware field");*fields.at(name)=config[name].asUInt();}
    shared_array::Resources array(h);Assets loader;std::vector<std::unique_ptr<Window>> windows;std::vector<Source> sources;std::vector<uint64_t> identities;
    const bool graph=job.isMember("sources");unsigned limit=job.get("max_active_sources",32).asUInt();require(limit&&limit<=64,"invalid graph frontend capacity");
    if(graph){std::map<uint64_t,unsigned> known;unsigned next_window=0;for(const auto &item:job["sources"]){Source s;s.id=item["source_operator_id"].asUInt64();require(!known.count(s.id),"duplicate native graph source");
        for(const auto &p:item["parents"]){require(known.count(p.asUInt64()),"native graph parent missing or forward");s.parents.push_back(known.at(p.asUInt64()));}
        for(const auto &index:item["windows"]){require(index.isUInt()&&index.asUInt()==next_window++&&index.asUInt()<job["windows"].size(),"native graph windows must partition in order");s.windows.push_back(index.asUInt());identities.push_back(s.id);}
        require(!s.windows.empty(),"native graph source has no windows");known.emplace(s.id,unsigned(sources.size()));sources.push_back(std::move(s));}
      require(next_window==job["windows"].size(),"native graph omitted windows");
    }else for(unsigned i=0;i<job["windows"].size();++i)identities.push_back(i);
    auto load=[&](const Json::Value &spec){if(spec["kind"]!="group_output")return loader.load(spec);require(graph&&spec["window"].isUInt()&&spec["window"].asUInt()<windows.size(),"invalid native graph input");auto t=windows[spec["window"].asUInt()]->output;require(t.sizes==shape(spec["shape"])&&dtype_name(t.type)==spec["dtype"].asString(),"native producer layout mismatch");return t;};
    for(unsigned source=0;source<job["windows"].size();++source){const auto &spec=job["windows"][source];auto w=std::make_unique<Window>();
      if(spec["schema"]=="mlx_matrix_window_job_v1"){
        w->a=load(spec["a"]);w->b=load(spec["b"]);bool bias=spec.isMember("bias");if(bias)w->bias=load(spec["bias"]);
        auto m=spec["m"].asUInt64(),n=spec["n"].asUInt64(),k=spec["k"].asUInt64();w->output=Tensor::allocate(dtype(spec["program"]["output_dtype"].asString()),{int64_t(m),int64_t(n)});
        auto o=matrix_schedule::Options::parse(spec["options"]);require(o.overlap,"native group must preserve concurrency");
        w->matrix=std::make_unique<matrix_schedule::Simulator>(spec["program"],w->a,w->b,bias?&w->bias:nullptr,spec.get("transposed_b",true).asBool(),0,0,m,n,k,w->output,0,o,nullptr,&array,identities[source]);
      }else{
        require(spec["schema"]=="mlx_vector_window_job_v1","unsupported native group family");for(const auto &name:spec["assets"].getMemberNames())w->values.emplace(name,load(spec["assets"][name]));
        const auto &node=spec["node"];w->output=Tensor::allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));
        auto o=vector_schedule::Options::parse(spec["options"]);require(o.overlap,"native group must preserve concurrency");w->vector=std::make_unique<vector_schedule::Simulator>(node,w->values,w->output,o,nullptr,&array,identities[source]);
      }
      windows.push_back(std::move(w));
    }
    uint64_t cycles=0;unsigned peak_sources=0;for(;;){bool work=false;for(const auto &w:windows)work|=!w->done();if(!work)break;
      require(cycles<job.get("max_cycles",Json::UInt64(10000000)).asUInt64(),"native group cycle budget exceeded");
      array.begin_cycle(cycles);
      if(graph){unsigned active=0;for(const auto &s:sources)active+=s.active;
        if(array.unit_ready(shared_array::Unit::Dma)){std::map<uint64_t,unsigned> ready;for(unsigned i=0;i<sources.size();++i){const auto &s=sources[i];bool parents=true;for(auto p:s.parents)parents&=sources[p].complete;if(!s.active&&!s.complete&&parents)ready.emplace(s.id,i);}
          for(auto [id,i]:ready){(void)id;if(active>=limit)break;auto &s=sources[i];s.active=true;s.begin=s.window_begin=cycles;++active;}}
        peak_sources=std::max(peak_sources,active);
      }
      if(graph){std::map<uint64_t,unsigned> order;for(unsigned i=0;i<sources.size();++i)if(sources[i].active)order.emplace(sources[i].id,i);
        for(auto [id,i]:order){(void)id;auto &s=sources[i];auto &w=windows[s.windows[s.current]];if(!w->done())w->tick();}
      }else for(auto &w:windows)if(!w->done())w->tick();
      array.end_cycle();++cycles;
      if(graph)for(auto &s:sources)if(s.active&&windows[s.windows[s.current]]->done()){Json::Value interval;interval["index"]=s.current;interval["begin_cycle"]=Json::UInt64(s.window_begin);interval["end_cycle"]=Json::UInt64(cycles);s.intervals.append(interval);
        if(++s.current==s.windows.size()){s.active=false;s.complete=true;s.end=cycles;}else s.window_begin=cycles;}
    }
    require(array.idle(),"native group failed to drain");auto directory=std::filesystem::path(argv[2]);std::filesystem::create_directories(directory);
    Json::Value report;report["cycles"]=Json::UInt64(cycles);report["classification"]="concurrent_array_windows_fixed_memory_not_full_graph_or_system";report["array"]=array.snapshot();report["windows"]=Json::Value(Json::arrayValue);
    for(unsigned index=0;index<windows.size();++index){auto &w=*windows[index];auto r=w.result();require(r["done"].asBool()&&r["dma_requests"]==r["dma_responses"],"native window failed to drain");report["windows"].append(r);
      std::ofstream data(directory/("output-"+std::to_string(index)+".bin"),std::ios::binary);if(w.output.storage->bytes)data.write(reinterpret_cast<const char*>(w.output.storage->data),w.output.storage->bytes);require(bool(data),"cannot save group output");}
    report["tensor_values_executed"]=true;report["full_model_verified"]=false;report["mlx_system_verified"]=false;
    if(graph){report["source_frontend_peak"]=peak_sources;for(const auto &s:sources){require(s.complete,"native graph did not publish all sources");Json::Value row;row["source_operator_id"]=Json::UInt64(s.id);row["begin_cycle"]=Json::UInt64(s.begin);row["publish_cycle"]=Json::UInt64(s.end);row["windows"]=s.intervals;report["source_intervals"].append(row);}}
    std::ofstream output(directory/"result.json");output<<report<<'\n';require(bool(output),"cannot save group report");std::cout<<"NATIVE_ARRAY_GROUP_PASS cycles="<<cycles<<'\n';
  }catch(const std::exception &error){std::cerr<<"NATIVE_ARRAY_GROUP_FAIL: "<<error.what()<<'\n';return 1;}
}
