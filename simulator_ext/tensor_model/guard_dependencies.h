#pragma once
#include <json/json.h>
#include <set>
#include <stdexcept>
#include <string>

namespace mlx::tensor_model {
// Every operation after a specialization guard retains its control edge.
// A serialized DAG must not drop the barrier and issue branch work early.
inline void validate_guard_dependencies(const Json::Value &program){
  std::set<std::string> guards;
  for(const auto &node:program["nodes"]){
    std::set<std::string> actual;
    const auto &deps=node["control_dependencies"];
    if(!deps.isNull()){
      if(!deps.isArray())throw std::runtime_error("invalid control dependency list");
      for(const auto &dep:deps){
        if(!dep.isObject()||dep.size()!=1||!dep["value"].isString()||!actual.insert(dep["value"].asString()).second)
          throw std::runtime_error("invalid/duplicate control dependency");
      }
    }
    if(actual!=guards)throw std::runtime_error("missing or foreign control-flow guard dependency");
    if(node["kind"]=="guard"){
      if(!node["id"].isString()||!guards.insert(node["id"].asString()).second)
        throw std::runtime_error("invalid/duplicate control-flow guard identity");
    }
  }
}
}
