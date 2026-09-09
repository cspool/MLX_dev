#include "memory_wire.hh"
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace mlx::tensor_model;
using namespace mlx::physical_device;
namespace {
struct UnusedMemory final:mlx::model_io::PhysicalMemoryPort {
  bool request_ready()const override{return false;}
  void submit(const mlx::model_io::PhysicalRequest&)override{throw std::runtime_error("memory decode unexpectedly requested data");}
  std::optional<mlx::model_io::Response> response()const override{return std::nullopt;}
  void consume_response()override{throw std::runtime_error("memory decode unexpectedly consumed data");}
};
Json::Value decode(const std::filesystem::path &path){
  auto size=std::filesystem::file_size(path);require(size==sizeof(mlx_memory_wire)||size==sizeof(mlx_memory_wire_v2),"memory wire file size mismatch");
  std::vector<uint8_t> wire(size);std::ifstream input(path,std::ios::binary);input.read(reinterpret_cast<char*>(wire.data()),size);require(bool(input),"cannot read memory wire");auto d=decode_memory_bytes(wire.data(),wire.size());
  UnusedMemory memory;mlx::model_io::RequestTokens tokens;mlx::model_io::AddressSpacePort port(memory,tokens,d.regions);mlx::memory_model::Simulator model(d.node,d.values,{},&port,&d.output);
  Json::Value row;row["file"]=path.string();row["node"]=d.node;row["backend_constructor_validated"]=true;row["view_elided"]=d.view;
  if(d.view)require(model.done()&&model.result()["dma_requests"].asUInt64()==0,"memory wire view performed data transfer");
  for(const auto &r:d.regions){Json::Value region;region["base"]=Json::UInt64(r.base);region["bytes"]=Json::UInt64(r.bytes);region["readable"]=r.readable;region["writable"]=r.writable;row["regions"].append(region);}return row;
}
}
int main(int argc,char **argv){
  try{require(argc==3,"usage: memory-wire-dump wire.bin-or-files.json output.json");std::filesystem::path path(argv[1]);Json::Value rows(Json::arrayValue);
    if(path.extension()==".json"){std::ifstream input(path);Json::Value files;input>>files;require(files.isArray(),"memory wire list is not an array");for(const auto &file:files)rows.append(decode(file.asString()));}else rows.append(decode(path));
    Json::Value result;result["classification"]="memory_wire_decode_layout_validation_not_transfer_execution";result["windows"]=rows;result["mlx_system_verified"]=false;
    std::filesystem::create_directories(std::filesystem::path(argv[2]).parent_path());std::ofstream output(argv[2]);output<<result<<'\n';require(bool(output),"cannot write memory wire report");std::cout<<"MEMORY_WIRE_DECODE_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"MEMORY_WIRE_DECODE_FAIL: "<<error.what()<<std::endl;return 1;}
}
