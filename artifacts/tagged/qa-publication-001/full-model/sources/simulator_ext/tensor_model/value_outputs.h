#pragma once
#include "tensor.h"
#include <algorithm>
#include <limits>
#include <set>

namespace mlx::tensor_model {
inline std::vector<std::string> output_ids(const Json::Value &node){
  if(node["kind"]!="split"){
    require(!node.isMember("split_outputs"),"tuple outputs attached to a non-split source");
    return {node["id"].asString()};
  }
  const auto &outputs=node["split_outputs"];
  require(outputs.isArray()&&!outputs.empty(),"split has no output descriptors");
  std::vector<std::string> ids;
  for(unsigned i=0;i<outputs.size();++i){
    auto expected=node["id"].asString()+":"+std::to_string(i);
    require(outputs[i].isObject()&&outputs[i].size()==2&&outputs[i]["id"]==expected
            &&outputs[i]["output"].isObject()&&outputs[i]["output"].size()==2,"split output identity/descriptor is not canonical");
    ids.push_back(expected);
  }
  return ids;
}

inline Values split_views(const Json::Value &node,const Tensor &owner){
  require(node["kind"]=="split","split views requested for another operation");
  const auto &args=node["args"],&p=node["memory_program"];
  require(args.isArray()&&(args.size()==2||args.size()==3)&&args[1].isInt64()&&args[1].asInt64()>=0
          &&(args.size()==2||args[2].isInt64()),"split size/axis contract mismatch");
  require(owner.storage&&owner.sizes.size()==owner.steps.size()&&!owner.sizes.empty()&&owner.offset>=0,"split owner metadata invalid");
  for(auto stride:owner.steps)require(stride>=0,"negative split owner stride");
  if(owner.numel())owner.position(owner.numel()-1);
  int64_t axis=args.size()==3?args[2].asInt64():0;
  if(axis<0)axis+=int64_t(owner.sizes.size());
  require(axis>=0&&uint64_t(axis)<owner.sizes.size(),"split axis out of bounds");
  auto width=uint64_t(owner.sizes[unsigned(axis)]),step=uint64_t(args[1].asInt64());
  require(step||!width,"split size zero requires an empty axis");
  uint64_t count=step?std::max<uint64_t>(1,width/step+unsigned(width%step!=0)):1;
  auto ids=output_ids(node);require(count==ids.size(),"split omits or adds outputs");
  const auto &layouts=p["split_layouts"];
  require(layouts.isObject()&&layouts.size()==ids.size(),"split layout set is incomplete");
  Values result;
  for(unsigned index=0;index<ids.size();++index){
    require(!step||uint64_t(index)<=UINT64_MAX/step,"split position overflow");
    uint64_t begin=uint64_t(index)*step;require(begin<=width,"split begins outside its source");
    Tensor view=owner;view.sizes[unsigned(axis)]=int64_t(std::min(step,width-begin));
    auto stride=owner.steps[unsigned(axis)];require(stride>=0&&(!stride||begin<=(uint64_t(INT64_MAX)-uint64_t(owner.offset))/uint64_t(stride)),"split offset overflow");
    view.offset+=int64_t(begin*uint64_t(stride));
    const auto &spec=node["split_outputs"][index]["output"],&layout=layouts[ids[index]];
    require(spec["dtype"]==dtype_name(view.type)&&shape(spec["shape"])==view.sizes,"split output shape/dtype mismatch");
    require(layout.isObject()&&layout.size()==6&&layout["dtype"]==dtype_name(view.type)&&shape(layout["shape"])==view.sizes
            &&shape(layout["strides"])==view.steps&&layout["offset"].isInt64()&&layout["offset"].asInt64()==view.offset
            &&layout["storage_elements"].isUInt64()&&layout["storage_elements"].asUInt64()==view.storage->bytes/element_bytes(view.type)
            &&layout["root"]==p["output_layout"]["root"],"split view layout/ownership mismatch");
    if(view.numel())view.position(view.numel()-1);
    result.emplace(ids[index],std::move(view));
  }
  return result;
}

inline void publish_split_views(const Json::Value &node,const Tensor &owner,Values &values){
  if(node["kind"]!="split")return;
  for(auto &[id,value]:split_views(node,owner))require(values.emplace(id,std::move(value)).second,"split redefined a live output");
}

inline void validate_value_contract(const Json::Value &program){
  bool split=false;std::set<std::string> owners;
  for(const auto &node:program["nodes"]){output_ids(node);if(node["kind"]=="split"){split=true;owners.insert(node["id"].asString());}}
  bool grouped=program.isMember("source_groups");
  require(program["schema"]==(program.isMember("output_contract")?"mlx_tensor_semantics_v4":grouped?"mlx_tensor_semantics_v3":split?"mlx_tensor_semantics_v2":"mlx_tensor_semantics_v1")
          &&(grouped?program["value_contract"]=="source_groups_v1":!split||program["value_contract"]=="tuple_view_outputs_v1"),"unsupported/missing tuple-view program contract");
  if(!split)return;
  std::set<std::string> declared;
  for(const auto &id:program["assets"].getMemberNames())declared.insert(id);
  auto check_refs=[&](auto &&self,const Json::Value &value)->void{
    if(value.isObject()){
      if(value.isMember("value"))require(!owners.count(value["value"].asString()),"internal split owner cannot be a tensor operand");
      else for(const auto &key:value.getMemberNames())self(self,value[key]);
    }else if(value.isArray())for(const auto &item:value)self(self,item);
  };
  for(const auto &node:program["nodes"]){
    require(declared.insert(node["id"].asString()).second,"duplicate tuple-program definition");
    if(node["kind"]=="split")for(const auto &id:output_ids(node))require(declared.insert(id).second,"duplicate tuple output definition");
    check_refs(check_refs,node["args"]);check_refs(check_refs,node["kwargs"]);check_refs(check_refs,node["control_dependencies"]);
  }
  std::vector<std::string> roles=program.isMember("output_contract")?std::vector<std::string>{"start_logits","end_logits","context_mask","offsets_utf8"}:std::vector<std::string>{"logits","token"};
  for(const auto &output:program["outputs"])for(const auto &role:roles)
    require(!owners.count(output[role].asString()),"internal split owner cannot be exported as a tensor");
}
}
