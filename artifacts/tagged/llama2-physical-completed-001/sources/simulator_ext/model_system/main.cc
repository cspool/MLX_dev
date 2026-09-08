#include "model_executor.h"
#include <fstream>
#include <iostream>

int main(int argc,char **argv){
  try{
    mlx::tensor_model::require(argc==4,"usage: mlx-physical-model program.json system-options.json output-directory");
    uint16_t endian=1;mlx::tensor_model::require(*reinterpret_cast<uint8_t*>(&endian)==1,"physical model requires little-endian host");
    Json::Value program,options;std::ifstream input(argv[1]),config(argv[2]);mlx::tensor_model::require(bool(input)&&bool(config),"cannot open physical model inputs");input>>program;config>>options;
    auto result=mlx::model_system::execute(program,options,argv[3]);
    std::ofstream output(std::filesystem::path(argv[3])/"result.json");output<<result<<'\n';mlx::tensor_model::require(bool(output),"cannot write physical model report");
    std::cout<<"PHYSICAL_MODEL_EXECUTION_PASS (not Chipyard/system performance)"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"PHYSICAL_MODEL_EXECUTION_FAIL: "<<error.what()<<std::endl;return 1;}
}
