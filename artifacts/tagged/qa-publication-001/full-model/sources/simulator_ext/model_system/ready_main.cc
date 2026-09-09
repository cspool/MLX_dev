#include "ready_graph.h"
#include <fstream>
#include <iostream>

int main(int argc,char **argv){
  try{
    using mlx::tensor_model::require;require(argc==4,"usage: mlx-ready-graph program.json options.json output-directory");
    uint16_t endian=1;require(*reinterpret_cast<uint8_t*>(&endian)==1,"ready graph requires a little-endian host");
    std::filesystem::path out(argv[3]);require(!std::filesystem::exists(out),"choose a fresh ready graph output directory");
    Json::Value program,options;std::ifstream input(argv[1]),config(argv[2]);require(bool(input)&&bool(config),"cannot open ready graph inputs");input>>program;config>>options;
    std::filesystem::create_directories(out);auto result=mlx::model_system::execute_ready_graph(program,options,out);std::ofstream report(out/"result.json");report<<result<<'\n';require(bool(report),"cannot save ready graph result");
    std::cout<<"READY_GRAPH_EXECUTION_PASS (not CDC/system/model acceptance)"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"READY_GRAPH_EXECUTION_FAIL: "<<error.what()<<std::endl;return 1;}
}
