#include "matrix_wire.hh"
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace mlx::tensor_model;
using namespace mlx::physical_device;
namespace {
struct UnusedMemory final:mlx::model_io::PhysicalMemoryPort {
  bool request_ready()const override{return false;}
  void submit(const mlx::model_io::PhysicalRequest&)override{throw std::runtime_error("wire validation unexpectedly requested data");}
  std::optional<mlx::model_io::Response> response()const override{return std::nullopt;}
  void consume_response()override{throw std::runtime_error("wire validation unexpectedly consumed data");}
};
Json::Value tensor(const Tensor &t,const mlx::model_io::Region &r){
  Json::Value row;row["dtype"]=dtype_name(t.type);row["shape"]=Json::Value(Json::arrayValue);row["strides"]=Json::Value(Json::arrayValue);
  for(auto n:t.sizes)row["shape"].append(Json::Int64(n));
  for(auto n:t.steps)row["strides"].append(Json::Int64(n));
  row["offset"]=Json::Int64(t.offset);row["base"]=Json::UInt64(r.base);row["bytes"]=Json::UInt64(r.bytes);return row;
}
Json::Value decode(const std::filesystem::path &file){
  require(std::filesystem::file_size(file)==sizeof(mlx_matrix_wire),"matrix wire file size mismatch");mlx_matrix_wire wire;
  std::ifstream input(file,std::ios::binary);input.read(reinterpret_cast<char*>(&wire),sizeof(wire));require(bool(input),"cannot read matrix wire file");auto d=decode_matrix(wire);
  UnusedMemory memory;mlx::model_io::RequestTokens tokens;mlx::model_io::AddressSpacePort port(memory,tokens,d.regions);
  mlx::matrix_schedule::Simulator model(d.program,d.a,d.b,d.has_bias?&d.bias:nullptr,d.transpose_b,d.a_batch,d.b_batch,d.m,d.n,d.k,d.output,d.output_batch,{},&port);
  Json::Value row;row["file"]=file.string();row["program"]=d.program;row["m"]=Json::UInt64(d.m);row["n"]=Json::UInt64(d.n);row["k"]=Json::UInt64(d.k);
  row["a_batch"]=Json::UInt64(d.a_batch);row["b_batch"]=Json::UInt64(d.b_batch);row["output_batch"]=Json::UInt64(d.output_batch);row["transpose_b"]=d.transpose_b;row["has_bias"]=d.has_bias;
  row["a"]=tensor(d.a,d.regions[0]);row["b"]=tensor(d.b,d.regions[1]);if(d.has_bias)row["bias"]=tensor(d.bias,d.regions[2]);row["output"]=tensor(d.output,d.regions[3]);row["backend_constructor_validated"]=true;return row;
}
}
int main(int argc,char **argv){
  try{
    require(argc==3,"usage: matrix-wire-dump wire.bin-or-files.json output.json");Json::Value rows(Json::arrayValue);auto path=std::filesystem::path(argv[1]);
    if(path.extension()==".json"){std::ifstream input(path);Json::Value files;input>>files;require(files.isArray(),"wire file list must be an array");for(const auto &file:files)rows.append(decode(file.asString()));}
    else rows.append(decode(path));
    Json::Value result;result["classification"]="matrix_wire_decode_constructor_validation_not_execution";result["windows"]=rows;result["mlx_system_verified"]=false;
    std::filesystem::create_directories(std::filesystem::path(argv[2]).parent_path());std::ofstream output(argv[2]);output<<result<<'\n';require(bool(output),"cannot write wire validation report");std::cout<<"MATRIX_WIRE_DECODE_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"MATRIX_WIRE_DECODE_FAIL: "<<error.what()<<std::endl;return 1;}
}
