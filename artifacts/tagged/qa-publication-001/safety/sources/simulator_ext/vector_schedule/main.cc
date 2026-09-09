#include "vector_schedule.h"
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc,char **argv){
  try{using namespace mlx::tensor_model;require(argc==3,"usage: mlx-vector-window job.json output-directory");std::ifstream input(argv[1]);Json::Value job;input>>job;require(job["schema"]=="mlx_vector_window_job_v1","unsupported vector window job");Assets loader;Values values;for(const auto &name:job["assets"].getMemberNames())values[name]=loader.load(job["assets"][name]);const auto &node=job["node"];auto output=Tensor::allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));
    mlx::vector_schedule::Simulator simulator(node,values,output,mlx::vector_schedule::Options::parse(job["options"]));while(simulator.tick()){}auto report=simulator.result();require(simulator.done()&&!simulator.tick()&&report==simulator.result(),"vector completion is not idempotent");require(report["dma_requests"]==report["dma_responses"],"vector DMA did not drain");require(report["numeric_instructions"]["global_write_bytes"].asUInt64()==output.numel()*element_bytes(output.type),"vector output DMA byte count mismatch");
    auto directory=std::filesystem::path(argv[2]);std::filesystem::create_directories(directory);std::ofstream bytes(directory/"output.bin",std::ios::binary);if(output.storage->bytes)bytes.write(reinterpret_cast<const char*>(output.storage->data),output.storage->bytes);require(bool(bytes),"cannot save vector output");std::ofstream json(directory/"result.json");json<<report<<'\n';require(bool(json),"cannot save vector report");std::cout<<"VECTOR_WINDOW_COMPONENT_PASS cycles="<<report["cycles"]<<std::endl;
  }catch(const std::exception &error){std::cerr<<"VECTOR_WINDOW_COMPONENT_FAIL: "<<error.what()<<std::endl;return 1;}
}
