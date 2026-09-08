#pragma once
#include "value_outputs.h"
#include <map>
#include <cmath>

namespace mlx::tensor_model {
inline void validate_source_groups(const Json::Value &program){
  if(!program.isMember("source_groups")){require(program["schema"]!="mlx_tensor_semantics_v3","v3 source groups missing");return;}
  const auto &groups=program["source_groups"],&nodes=program["nodes"];
  require(program["schema"]=="mlx_tensor_semantics_v3"&&program["value_contract"]=="source_groups_v1"&&groups.isArray()&&!groups.empty(),"invalid source-group contract");
  unsigned cursor=0;
  for(unsigned source=0;source<groups.size();++source){const auto &g=groups[source],&ids=g["lowered_ids"],&names=g["stage_names"],&kinds=g["lowered_kinds"];
    require(g["source_operator_id"].isUInt()&&g["source_operator_id"].asUInt()==source&&ids.isArray()&&!ids.empty()
            &&names.isArray()&&names.size()==ids.size()&&kinds.isArray()&&kinds.size()==ids.size(),"invalid original source group");
    std::vector<std::string> expected;
    if(g["lowering_profile"]=="direct"){
      require(g["source_operator"]!="aten.layer_norm.default","LayerNorm cannot bypass its recipe");expected={"direct"};
    }else{
      require(g["lowering_profile"]=="mlx-layernorm-shifted-fp32-v1"&&g["source_operator"]=="aten.layer_norm.default","unknown source lowering profile");
      const auto &cfg=g["layer_norm_config"];
      require((cfg["input_dtype"]=="torch.float16"||cfg["input_dtype"]=="torch.float32")&&cfg["output_dtype"]==cfg["input_dtype"],"LayerNorm source precision contract differs");
      auto sizes=shape(cfg["shape"]);require(!sizes.empty()&&sizes.back()>0&&sizes.back()<=(1LL<<31),"LayerNorm source shape contract differs");
      for(auto size:sizes)require(size>=0,"LayerNorm source shape contract differs");
      require(cfg["epsilon"].isNumeric()&&std::isfinite(cfg["epsilon"].asDouble())&&cfg["epsilon"].asDouble()>=0&&cfg["cudnn_enable"].isBool(),"LayerNorm source epsilon/flag contract differs");
      if(cfg["input_dtype"]!="torch.float32")expected.push_back("input_f32");
      for(const char *name:{"anchor_select","anchor_expand","shift","mean_shift","center","square","variance","epsilon","rsqrt","normalize"})expected.push_back(name);
      for(const char *parameter:{"weight","bias"}){
        const auto &type=cfg[std::string(parameter)+"_dtype"];
        require(type.isNull()||type==cfg["input_dtype"]||type=="torch.float32","LayerNorm source affine precision contract differs");
        if(!type.isNull()){if(type!="torch.float32")expected.push_back(std::string(parameter)+"_f32");expected.push_back(std::string("affine_")+parameter);}
      }
      if(cfg["output_dtype"]!="torch.float32")expected.push_back("output_cast");
    }
    require(expected.size()==ids.size(),"source recipe stages incomplete");
    for(unsigned stage=0;stage<ids.size();++stage){
      require(ids[stage].isUInt()&&ids[stage].asUInt()==cursor&&cursor<nodes.size(),"source groups must partition all lowered stages");
      const auto &node=nodes[cursor++];
      if(g["lowering_profile"]=="direct")require(node["source_operator"]==g["source_operator"],"direct source operator label differs");
      require(names[stage]==expected[stage]&&node["lowering_stage_name"]==names[stage]&&node["kind"]==kinds[stage]
              &&node["source_operator_id"].asUInt()==ids[stage].asUInt()&&node["origin_source_operator_id"].isUInt()
              &&node["origin_source_operator_id"].asUInt()==source&&node["lowering_stage"].isUInt()&&node["lowering_stage"].asUInt()==stage
              &&node["forward_id"]==g["forward_id"]&&node["layer_idx"]==g["layer_idx"],"lowered stage lost its original source identity");
      if(g["lowering_profile"]!="direct"){
        const std::map<std::string,std::string> required={{"input_f32","cast"},{"anchor_select","select"},{"anchor_expand","unsqueeze"},{"shift","sub"},{"mean_shift","mean"},{"center","sub"},{"square","pow"},{"variance","mean"},{"epsilon","add"},{"rsqrt","rsqrt"},{"normalize","mul"},{"weight_f32","cast"},{"affine_weight","mul"},{"bias_f32","cast"},{"affine_bias","add"},{"output_cast","cast"}};
        require(node["kind"]==required.at(expected[stage]),"LayerNorm primitive kind differs from recipe");
        if(expected[stage]=="epsilon")require(node["args"][1].isNumeric()&&node["args"][1].asDouble()==g["layer_norm_config"]["epsilon"].asDouble(),"LayerNorm source epsilon differs from its primitive");
      }
    }
    auto outputs=output_ids(nodes[cursor-1]);require(g["output_values"].isArray()&&g["output_values"].size()==outputs.size(),"source output binding mismatch");
    if(g["lowering_profile"]!="direct")require(nodes[cursor-1]["output"]["shape"]==g["layer_norm_config"]["shape"]&&nodes[cursor-1]["output"]["dtype"]==(g["layer_norm_config"]["output_dtype"]=="torch.float16"?"f16":"f32"),"LayerNorm source output contract differs");
    for(unsigned i=0;i<outputs.size();++i)require(g["output_values"][i]==outputs[i],"source output binding mismatch");
  }
  require(cursor==nodes.size(),"ungrouped lowered stages");
}

inline void report_source_groups(const Json::Value &program,Json::Value &report){
  if(!program.isMember("source_groups"))return;
  validate_source_groups(program);auto &events=report["events"];
  require(events.size()==program["nodes"].size(),"not every lowered stage produced a completion event");
  std::map<unsigned,Json::ArrayIndex> by_id;
  auto annotate=[&](Json::Value &event){auto id=event["source_operator_id"].asUInt();require(id<program["nodes"].size(),"unknown lowered event identity");
    const auto &node=program["nodes"][id];event["lowered_operator_id"]=id;event["origin_source_operator_id"]=node["origin_source_operator_id"];
    event["lowering_stage"]=node["lowering_stage"];event["lowering_stage_name"]=node["lowering_stage_name"];};
  for(unsigned i=0;i<events.size();++i){annotate(events[i]);require(by_id.emplace(events[i]["source_operator_id"].asUInt(),i).second,"duplicate lowered completion");}
  if(report.isMember("windows"))for(const auto &family:report["windows"].getMemberNames())for(auto &window:report["windows"][family])annotate(window);
  for(const char *field:{"matrix_windows","vector_windows","memory_windows","control_windows"})if(report.isMember(field)&&report[field].isArray())for(auto &window:report[field])annotate(window);
  Json::Value completed(Json::arrayValue);
  for(const auto &group:program["source_groups"]){Json::Value record;
    record["source_operator_id"]=group["source_operator_id"];record["source_operator"]=group["source_operator"];record["lowered_ids"]=group["lowered_ids"];record["complete"]=true;
    uint64_t begin=UINT64_MAX,end=0;bool timed=true;
    for(const auto &id:group["lowered_ids"]){const auto &event=events[by_id.at(id.asUInt())];
      if(event.isMember("start_cycle")){begin=std::min(begin,event["start_cycle"].asUInt64());end=std::max(end,event["publish_cycle"].asUInt64());}
      else if(event.isMember("shared_start_cycle")){begin=std::min(begin,event["shared_start_cycle"].asUInt64());end=std::max(end,event["shared_end_cycle"].asUInt64());}
      else timed=false;
    }
    record["timing_observed"]=timed;if(timed){record["start_cycle"]=Json::UInt64(begin);record["complete_cycle"]=Json::UInt64(end);}
    completed.append(record);
  }
  report["executed_lowered_calls"]=report["executed_source_calls"];report["executed_source_calls"]=program["source_groups"].size();
  report["event_identity_domain"]="lowered_ids_with_explicit_origin_source_ids";report["source_group_events"]=completed;
}
}
