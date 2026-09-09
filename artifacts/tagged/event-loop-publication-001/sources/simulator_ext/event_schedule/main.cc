#include "event_schedule.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){
  try{
    if(argc!=3)throw std::runtime_error("usage: mlx-event-schedule program.json result.json");
    std::ifstream input(argv[1]);Json::Value program;input>>program;
    auto result=mlx::event_schedule::simulate(program);
    std::ofstream output(argv[2]);output<<result<<'\n';
    if(!output)throw std::runtime_error("cannot save event result");
    std::cout<<"MLX_EVENT_COMPONENT_COMPLETE cycles="<<result["cycles"].asUInt64()<<'\n';
  }catch(const std::exception &e){std::cerr<<"MLX_EVENT_FAIL: "<<e.what()<<'\n';return 1;}
}
