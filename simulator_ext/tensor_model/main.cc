#include "tensor.h"
#include "guard_dependencies.h"
#include "value_outputs.h"
#include "control_program.h"
#include "vector_program.h"
#include "memory_program.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

using namespace mlx::tensor_model;
int main(int argc,char **argv) {
  try {
    require(argc==5 || argc==6,"usage: mlx-tensor-semantics program.json output-directory cpu-blas.so threads [observation-ids.json]");
    uint16_t endian=1; require(*reinterpret_cast<uint8_t*>(&endian)==1,"little-endian host required");
    std::ifstream input(argv[1]); Json::Value program; input>>program;
    require((program["schema"]=="mlx_tensor_semantics_v1"||program["schema"]=="mlx_tensor_semantics_v2") && program["timing_mode"]=="unmodeled",
            "unsupported tensor semantics program");
    validate_guard_dependencies(program);
    validate_value_contract(program);
    const bool scheduled_matrix=program["matrix_backend"]=="scheduled";
    const bool strict_matrix=program["matrix_backend"]=="microcode" || scheduled_matrix;
    const bool scheduled_vector=program["vector_backend"]=="scheduled";
    const bool strict_vector=program["vector_backend"]=="microcode" || scheduled_vector;
    const bool scheduled_memory=program["memory_backend"]=="scheduled";
    const bool strict_memory=program["memory_backend"]=="planned" || scheduled_memory;
    const bool scheduled_control=program["control_backend"]=="scheduled";
    const bool strict_control=program["control_backend"]=="rv64_leaf" || scheduled_control;
    require(!strict_matrix || std::string(argv[3])=="none","microcode program must disable BLAS");
    const std::filesystem::path output(argv[2]); std::filesystem::create_directories(output);
    std::set<unsigned> requested_observations;
    if (argc==6) {
      std::ifstream spec(argv[5]); Json::Value ids; spec>>ids;
      require(ids.isArray(),"observation IDs must be a JSON array");
      for (const auto &id:ids) {
        require(id.isUInt(),"invalid observation operator ID");
        requested_observations.insert(id.asUInt());
      }
    }
    for(const auto &node:program["nodes"])require(node["kind"]!="split"||!requested_observations.count(node["source_operator_id"].asUInt()),"tuple-view observations require explicit result selection");
    Json::Value observations(Json::arrayValue); uint64_t observation_bytes=0;
    Assets assets; Values values;
    for (const auto &name:program["assets"].getMemberNames()) values[name]=assets.load(program["assets"][name]);
    require(!scheduled_vector || program["vector_schedule_options"].isObject(),"scheduled vectors require explicit timing options");
    require(!scheduled_memory || program["memory_schedule_options"].isObject(),"scheduled memory requires explicit timing options");
    require(!scheduled_control || program["control_schedule_options"].isObject(),"scheduled controller requires explicit timing options");
    Kernels kernels(argv[3],std::stoul(argv[4]),scheduled_matrix?program["matrix_schedule_options"]:Json::Value(),scheduled_vector?program["vector_schedule_options"]:Json::Value(),scheduled_memory?program["memory_schedule_options"]:Json::Value(),scheduled_control?program["control_schedule_options"]:Json::Value());
    require(!scheduled_matrix || program["matrix_schedule_options"].isObject(),"scheduled program requires explicit timing options");
    Json::Value events(Json::arrayValue); std::string boundary;
    uint64_t executed=0;
    for (const auto &node:program["nodes"]) {
      const bool matrix=node["kind"]=="linear" || node["kind"]=="matmul";
      require(!node.isMember("matrix_program") || matrix,"matrix microcode attached to a nonmatrix operator");
      require(!strict_matrix || !matrix || node.isMember("matrix_program"),"strict matrix program has a missing lowering");
      const bool floating_output=node["output"]["dtype"]=="f16"||node["output"]["dtype"]=="f32";
      require(!strict_vector || !floating_output || !mlx::vector_model::supports(node["kind"].asString()) || node.isMember("vector_program"),"strict floating vector program has a missing lowering");
      require(!strict_memory || !mlx::memory_model::supports(node["kind"].asString()) || node.isMember("memory_program"),"strict memory program has a missing lowering");
      const bool control=mlx::control_model::extended_kind(node["kind"].asString())||node["kind"]=="arange"||node["kind"]=="le"||node["kind"]=="argmax"||((node["kind"]=="add"||node["kind"]=="mul")&&node["output"]["dtype"]=="i64");
      require(!strict_control || !control || node.isMember("control_program"),"strict controller program has a missing lowering");
      unsigned paths=0;for(const char *field:{"matrix_program","vector_program","memory_program","control_program"})paths+=node.isMember(field);
      require(paths<=1,"operator has ambiguous executable lowering paths");
      const auto position=std::to_string(node["forward_id"].asInt())+":"+
          (node["layer_idx"].isNull()?"outside":std::to_string(node["layer_idx"].asInt()));
      if (position!=boundary) { std::cout<<"TENSOR_BOUNDARY "<<position<<" operator="<<executed<<std::endl; boundary=position; }
      try {
        values[node["id"].asString()]=kernels.execute(node,values);
      } catch (const std::exception &error) {
        throw std::runtime_error("operator "+std::to_string(node["source_operator_id"].asUInt())+
            " "+node["source_operator"].asString()+" "+node["module_path"].asString()+": "+error.what());
      }
      publish_split_views(node,values.at(node["id"].asString()),values);
      Json::Value event(Json::objectValue);
      event["source_operator_id"]=node["source_operator_id"]; event["entry"]="tensor_model::"+node["kind"].asString();
      event["forward_id"]=node["forward_id"]; event["layer_idx"]=node["layer_idx"];
      event["value_id"]=node["id"]; event["status"]="native_semantics_completed";
      if(node["kind"]=="split"){event.removeMember("value_id");event["produced_values"]=Json::Value(Json::arrayValue);for(const auto &id:output_ids(node))event["produced_values"].append(id);}
      if (node.isMember("matrix_program")) event["matrix_entry"]=scheduled_matrix?"mlx::matrix_schedule::Simulator":"mlx::tensor_model::execute_matrix_program";
      if (node.isMember("vector_program")) event["vector_entry"]=scheduled_vector?"mlx::vector_schedule::Simulator":"mlx::vector_model::execute";
      if (node.isMember("memory_program")) event["memory_entry"]=scheduled_memory?"mlx::memory_model::Simulator":"mlx::memory_model::execute";
      if (node.isMember("control_program")) event["control_entry"]=scheduled_control?"mlx::control_schedule::Simulator":"mlx::control_model::execute";
      events.append(event); ++executed;
      if (requested_observations.erase(node["source_operator_id"].asUInt())) {
        const auto &value=values.at(node["id"].asString());
        const uint64_t bytes=value.numel()*element_bytes(value.type);
        require(bytes<=1024*1024 && observation_bytes+bytes<=32*1024*1024,"numerical observation byte budget exceeded");
        auto dense=value.materialize(value.type);
        auto file=output/(node["id"].asString()+".bin");
        std::ofstream stream(file,std::ios::binary);
        if (bytes) stream.write(reinterpret_cast<const char*>(dense.storage->data),bytes);
        require(bool(stream),"cannot write diagnostic output");
        Json::Value item=event; item["shape"]=node["output"]["shape"];
        item["dtype"]=dtype_name(value.type); item["file"]=file.string();
        observations.append(item); observation_bytes+=bytes;
      }
      for (const auto &name:node["release"]) values.erase(name.asString());
    }
    require(requested_observations.empty(),"requested observation IDs did not execute");
    Json::Value results(Json::arrayValue);
    for (const auto &spec:program["outputs"]) {
      const auto &logits=values.at(spec["logits"].asString());
      const auto &tokens=values.at(spec["token"].asString());
      auto dense=logits.materialize(logits.type);
      for (uint64_t i=0;i<logits.numel();++i) require(std::isfinite(logits.number(i)),"nonfinite final model logits");
      auto file=output/("logits-"+std::to_string(spec["forward_id"].asInt())+"."+dtype_name(logits.type)+".bin");
      std::ofstream stream(file,std::ios::binary); stream.write(reinterpret_cast<const char*>(dense.storage->data),dense.storage->bytes);
      require(bool(stream),"cannot write tensor model output");
      Json::Value result(Json::objectValue); result["forward_id"]=spec["forward_id"];
      result["logits_file"]=file.string(); result["dtype"]=dtype_name(logits.type);
      for (auto d:logits.sizes) result["shape"].append(Json::Int64(d));
      for (uint64_t i=0;i<tokens.numel();++i) result["tokens"].append(Json::Int64(tokens.integer(i)));
      results.append(result);
    }
    Json::Value report(Json::objectValue);
    report["classification"]="native_cpp_tensor_semantics_not_mlx_system";
    report["executed_source_calls"]=Json::UInt64(executed); report["events"]=events; report["outputs"]=results;
    report["observations"]=observations; report["observation_bytes"]=Json::UInt64(observation_bytes);
    report["matrix_macs"]=Json::UInt64(kernels.matrix_macs);
    report["matrix_microcode"]=kernels.matrix_instruction_report();
    report["vector_microcode"]=kernels.vector_instruction_report();
    report["vector_windows"]=kernels.vector_window_report();
    report["memory_programs"]=kernels.memory_instruction_report();
    report["memory_windows"]=kernels.memory_window_report();
    report["control_programs"]=kernels.control_instruction_report();
    report["control_windows"]=kernels.control_window_report();
    report["functional_entry_calls"]=Json::UInt64(kernels.functional_calls);
    report["matrix_windows"]=kernels.matrix_window_report();
    report["blas_calls"]=Json::UInt64(kernels.blas_calls);
    report["materialized_value_bytes"]=Json::UInt64(kernels.materialized_bytes);
    report["python_or_gpu_execution_fallbacks"]=0;
    report["timing_mode"]="unmodeled"; report["mlx_system_verified"]=false; report["performance_eligible"]=false;
    std::ofstream summary(output/"result.json"); summary<<report<<'\n'; require(bool(summary),"cannot save execution report");
    std::cout<<"NATIVE_TENSOR_SEMANTICS_PASS calls="<<executed<<std::endl;
  } catch (const std::exception &error) {
    std::cerr<<"NATIVE_TENSOR_SEMANTICS_FAIL: "<<error.what()<<std::endl; return 1;
  }
}
