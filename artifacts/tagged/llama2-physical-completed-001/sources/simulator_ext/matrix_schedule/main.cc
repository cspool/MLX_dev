#include "matrix_schedule.h"
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc,char **argv) {
  try {
    using namespace mlx::tensor_model;
    require(argc==3,"usage: mlx-matrix-window job.json output-directory");
    std::ifstream input(argv[1]);Json::Value job;input>>job;
    require(job["schema"]=="mlx_matrix_window_job_v1","unsupported matrix window job");
    Assets assets;auto a=assets.load(job["a"]),b=assets.load(job["b"]);
    Tensor bias;bool has_bias=job.isMember("bias");if(has_bias)bias=assets.load(job["bias"]);
    auto m=job["m"].asUInt64(),n=job["n"].asUInt64(),k=job["k"].asUInt64();
    auto output=Tensor::allocate(dtype(job["program"]["output_dtype"].asString()),{int64_t(m),int64_t(n)});
    auto options=mlx::matrix_schedule::Options::parse(job["options"]);
    mlx::matrix_schedule::Simulator simulator(job["program"],a,b,has_bias?&bias:nullptr,
        job.get("transposed_b",true).asBool(),0,0,m,n,k,output,0,options);
    while(simulator.tick()){}
    auto report=simulator.result();
    require(simulator.done()&&report["dma_requests"]==report["dma_responses"],"matrix window did not drain");
    require(report["numeric_instructions"]["mul_active_lanes"].asUInt64()==m*n*k,"matrix instruction coverage mismatch");
    require(report["dma_write_bytes"].asUInt64()==m*n*element_bytes(output.type),"matrix output DMA byte count mismatch");
    require(!simulator.tick()&&report==simulator.result(),"completed tick is not idempotent");
    auto directory=std::filesystem::path(argv[2]);std::filesystem::create_directories(directory);
    std::ofstream bytes(directory/"output.bin",std::ios::binary);
    if(output.storage->bytes)bytes.write(reinterpret_cast<const char*>(output.storage->data),output.storage->bytes);
    require(bool(bytes),"cannot write scheduled matrix output");
    std::ofstream json(directory/"result.json");json<<report<<'\n';require(bool(json),"cannot save matrix window report");
    std::cout<<"MATRIX_WINDOW_COMPONENT_PASS cycles="<<report["cycles"]<<std::endl;
  }catch(const std::exception &error){std::cerr<<"MATRIX_WINDOW_COMPONENT_FAIL: "<<error.what()<<std::endl;return 1;}
}
