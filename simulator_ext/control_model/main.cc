#include "control_program.h"
#include "rv64_leaf.h"
#include <fstream>
#include <iostream>

int main(int argc,char **argv){
  try{using namespace mlx::tensor_model;require(argc==3,"usage: mlx-control-leaf input.json output.json");std::ifstream input(argv[1]);Json::Value job;input>>job;Json::Value report(Json::objectValue);
    if(job["schema"]=="rv64_leaf_probe_v1"){
      mlx::control_model::RV64 machine;for(const auto &key:job["x"].getMemberNames()){auto reg=std::stoul(key);require(reg<32,"integer register out of bounds");machine.x[reg]=job["x"][key].asUInt64();}
      for(const auto &key:job["f"].getMemberNames()){auto reg=std::stoul(key);require(reg<32,"floating register out of bounds");machine.f[reg]=job["f"][key].asUInt64();}
      machine.run(job["words"]);for(unsigned reg=0;reg<32;++reg){report["x"].append(Json::UInt64(machine.x[reg]));report["f"].append(Json::UInt64(machine.f[reg]));}report["fflags"]=machine.fflags;report["retired"]=Json::UInt64(machine.retired);report["branches_taken"]=Json::UInt64(machine.branches_taken);
    }else{require(job["schema"]=="mlx_control_job_v1","unsupported controller job");Assets loader;Values values;for(const auto &key:job["assets"].getMemberNames())values[key]=loader.load(job["assets"][key]);mlx::control_model::Stats stats;auto value=mlx::control_model::execute(job["node"],values,stats);report=stats.json();for(uint64_t index=0;index<value.numel();++index)report["values"].append(Json::Int64(value.integer(index)));}
    report["rocket_execution_verified"]=false;std::ofstream output(argv[2]);output<<report<<'\n';require(bool(output),"cannot save controller result");std::cout<<"CONTROL_LEAF_COMPONENT_PASS"<<std::endl;
  }catch(const std::exception &error){std::cerr<<"CONTROL_LEAF_COMPONENT_FAIL: "<<error.what()<<std::endl;return 1;}
}
