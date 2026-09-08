#include "asset_source.hh"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc,char **argv){
  try{
    if(argc!=2)throw std::runtime_error("usage: asset-source-contract job.json");
    std::ifstream input(argv[1]);Json::Value job;input>>job;
    mlx::physical_device::AssetSource source(job["config"]);Json::Value events{Json::arrayValue};
    for(const auto &action:job["actions"]){
      if(action["kind"].asString()=="truncate"){std::filesystem::resize_file(action["path"].asString(),action["bytes"].asUInt64());continue;}
      auto count=action["bytes"].asUInt64();if(count>64)throw std::runtime_error("probe buffer limit");std::vector<uint8_t> bytes(size_t(count),0xa5);
      bool ok=action["kind"].asString()=="store"?source.store(action["offset"].asUInt64(),size_t(count),bytes.data()):source.load(action["offset"].asUInt64(),size_t(count),bytes.data());
      Json::Value event;event["ok"]=ok;event["data"]=Json::arrayValue;for(auto byte:bytes)event["data"].append(unsigned(byte));events.append(event);
    }
    Json::Value report;report["events"]=events;report["source"]=source.snapshot();std::cout<<report<<'\n';
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
