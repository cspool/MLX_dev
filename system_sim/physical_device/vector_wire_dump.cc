#include "vector_wire.hh"
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace mlx::tensor_model;
using namespace mlx::physical_device;
namespace {
struct UnusedMemory final:mlx::model_io::PhysicalMemoryPort {
  bool request_ready()const override{return false;}
  void submit(const mlx::model_io::PhysicalRequest&)override{throw std::runtime_error("vector decode unexpectedly accessed data");}
  std::optional<mlx::model_io::Response> response()const override{return std::nullopt;}
  void consume_response()override{throw std::runtime_error("vector decode unexpectedly consumed data");}
};
Json::Value decode(const std::filesystem::path &path){
  require(std::filesystem::file_size(path)==sizeof(mlx_vector_wire),"vector wire file size mismatch");mlx_vector_wire wire;std::ifstream input(path,std::ios::binary);input.read(reinterpret_cast<char*>(&wire),sizeof(wire));require(bool(input),"cannot read vector wire");auto d=decode_vector(wire);
  UnusedMemory memory;mlx::model_io::RequestTokens tokens;mlx::model_io::AddressSpacePort port(memory,tokens,d.regions);mlx::vector_schedule::Simulator model(d.node,d.values,d.output,{},&port);
  Json::Value row;row["file"]=path.string();row["node"]=d.node;
  for(const auto &[name,t]:d.values){auto &v=row["values"][name];v["dtype"]=dtype_name(t.type);v["shape"]=Json::Value(Json::arrayValue);v["strides"]=Json::Value(Json::arrayValue);for(auto n:t.sizes)v["shape"].append(Json::Int64(n));for(auto n:t.steps)v["strides"].append(Json::Int64(n));v["offset"]=Json::Int64(t.offset);v["bytes"]=Json::UInt64(t.storage->bytes);}
  for(const auto &r:d.regions){Json::Value region;region["base"]=Json::UInt64(r.base);region["bytes"]=Json::UInt64(r.bytes);region["readable"]=r.readable;region["writable"]=r.writable;row["regions"].append(region);}
  row["backend_constructor_validated"]=true;return row;
}
}
int main(int argc,char **argv){
  try{require(argc==3,"usage: vector-wire-dump wire.bin-or-files.json output.json");Json::Value rows(Json::arrayValue);std::filesystem::path path(argv[1]);
    if(path.extension()==".json"){std::ifstream input(path);Json::Value files;input>>files;require(files.isArray(),"vector wire list must be an array");for(const auto &file:files)rows.append(decode(file.asString()));}
    else rows.append(decode(path));
    Json::Value report;report["classification"]="vector_wire_decode_constructor_validation_not_execution";report["windows"]=rows;report["mlx_system_verified"]=false;
    std::filesystem::create_directories(std::filesystem::path(argv[2]).parent_path());std::ofstream output(argv[2]);output<<report<<'\n';require(bool(output),"cannot write vector wire report");std::cout<<"VECTOR_WIRE_DECODE_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"VECTOR_WIRE_DECODE_FAIL: "<<error.what()<<std::endl;return 1;}
}
