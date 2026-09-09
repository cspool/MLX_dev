#pragma once
#include "value_outputs.h"
#include "qa_span.h"
#include <filesystem>
#include <fstream>
#include <set>

namespace mlx::tensor_model {
inline bool qa_results(const Json::Value &p){return p["output_contract"]=="mlx-qa-result-v1";}
inline std::vector<std::string> result_roles(const Json::Value &p){
  return qa_results(p)?std::vector<std::string>{"start_logits","end_logits","context_mask","offsets_utf8"}:std::vector<std::string>{"logits","token"};
}
inline std::set<uint64_t> utf8_boundaries(const std::string &text){
  std::set<uint64_t> boundaries{0};
  for(size_t i=0;i<text.size();){
    auto lead=uint8_t(text[i]);unsigned width=lead<0x80?1:lead>=0xc2&&lead<=0xdf?2:lead>=0xe0&&lead<=0xef?3:lead>=0xf0&&lead<=0xf4?4:0;
    require(width&&width<=text.size()-i,"invalid QA UTF8 context");
    uint32_t code=lead&(width==1?0x7f:width==2?0x1f:width==3?0x0f:0x07);
    for(unsigned j=1;j<width;++j){auto c=uint8_t(text[i+j]);require((c&0xc0)==0x80,"invalid QA UTF8 continuation");code=(code<<6)|(c&0x3f);}
    require((width==1||code>=(width==2?0x80U:width==3?0x800U:0x10000U))&&code<=0x10ffff&&!(code>=0xd800&&code<=0xdfff),"invalid QA UTF8 codepoint");
    i+=width;boundaries.insert(i);
  }
  return boundaries;
}
inline void validate_result_contract(const Json::Value &p){
  bool qa=qa_results(p);
  require((!p.isMember("output_contract")||qa)&&((p["schema"]=="mlx_tensor_semantics_v4")==qa),"unsupported/missing QA result contract");
  if(!qa)return;
  std::map<std::string,std::pair<Json::Value,int>> computed;
  std::set<int> node_forwards,forwards;
  for(const auto &node:p["nodes"]){
    require(node["forward_id"].isInt()&&node["forward_id"].asInt()>=0,"invalid QA source forward");
    int forward=node["forward_id"].asInt();node_forwards.insert(forward);
    for(const auto &id:output_ids(node)){
      Json::Value spec=node["output"];
      if(node["kind"]=="split")for(const auto &out:node["split_outputs"])if(out["id"]==id)spec=out["output"];
      require(computed.emplace(id,std::make_pair(spec,forward)).second,"duplicate QA value definition");
    }
  }
  require(p["outputs"].isArray()&&!p["outputs"].empty(),"QA outputs are missing");
  for(const auto &out:p["outputs"]){
    require(out.isObject()&&out.size()==7&&out["forward_id"].isInt()&&out["forward_id"].asInt()>=0
            &&forwards.insert(out["forward_id"].asInt()).second&&out["context_utf8"].isString()
            &&out["max_answer_tokens"].isUInt()&&out["max_answer_tokens"].asUInt()==30,"invalid QA result descriptor/policy");
    auto boundaries=utf8_boundaries(out["context_utf8"].asString());uint64_t n=0;
    for(const auto &role:result_roles(p))require(out[role].isString(),"invalid QA result value identity");
    for(const char *role:{"start_logits","end_logits"}){
      auto it=computed.find(out[role].asString());require(it!=computed.end(),"QA output must be computed, not an asset/internal owner");
      const auto &spec=it->second.first;auto dims=shape(spec["shape"]);
      require(spec["dtype"]=="f32"&&dims.size()==2&&dims[0]==1&&dims[1]>0&&it->second.second==out["forward_id"].asInt(),"QA output shape/dtype/forward mismatch");
      require(!n||n==uint64_t(dims[1]),"QA output lengths differ");n=uint64_t(dims[1]);
    }
    require(out["start_logits"]!=out["end_logits"],"QA needs distinct named output bindings");
    for(const char *role:{"context_mask","offsets_utf8"}){
      bool mask=std::string(role)=="context_mask";const auto &a=p["assets"][out[role].asString()];
      require(a.isObject()&&a.size()==5&&a["kind"]=="literal"&&a["origin"]=="qa_tokenizer_input"
              &&a["dtype"]==(mask?"bool":"i64")&&shape(a["shape"])==(mask?Shape{int64_t(n)}:Shape{int64_t(n),2})
              &&a["values"].isArray()&&a["values"].size()==n,"QA metadata must be explicit tokenizer input assets");
    }
    const auto &mask=p["assets"][out["context_mask"].asString()]["values"],&offsets=p["assets"][out["offsets_utf8"].asString()]["values"];
    uint64_t previous_a=0,previous_b=0;bool any=false;
    for(unsigned i=0;i<n;++i){
      const auto &pair=offsets[i];require(mask[i].isBool()&&pair.isArray()&&pair.size()==2&&pair[0].isUInt64()&&pair[1].isUInt64(),"invalid QA mask/offset input values");
      uint64_t a=pair[0].asUInt64(),b=pair[1].asUInt64();
      if(!mask[i].asBool()){require(a==0&&b==0,"masked QA offsets must be zero");continue;}
      require(a<b&&boundaries.count(a)&&boundaries.count(b)&&a>=previous_a&&b>=previous_b,"invalid QA UTF8 offsets/order");
      previous_a=a;previous_b=b;any=true;
    }
    require(any,"QA has no context tokens");
    for(const auto &node:p["nodes"])for(const auto &released:node["release"])for(const auto &role:result_roles(p))
      require(released!=out[role],"QA result released before readback");
  }
  require(forwards==node_forwards,"QA does not cover every forward");
}

/* read_value is either native materialization or the owning physical port's
 * response-driven readback. The span computation itself has no CPU timing model. */
template<class Read> Json::Value execute_qa_result(const Json::Value &p,const Json::Value &spec,
                                                const std::filesystem::path &directory,Read read_value){
  Values data;
  for(const auto &role:result_roles(p))data.emplace(role,read_value(spec[role].asString()));
  const auto &start=data.at("start_logits"),&end=data.at("end_logits"),&mask=data.at("context_mask"),&offsets=data.at("offsets_utf8");
  require(start.type==DType::F32&&end.type==DType::F32&&start.sizes==end.sizes&&start.sizes.size()==2&&start.sizes[0]==1
          &&mask.type==DType::Bool&&mask.sizes==Shape{start.sizes[1]}&&offsets.type==DType::I64&&offsets.sizes==Shape{start.sizes[1],2},"actual QA output/metadata shape or type differs");
  auto n=start.numel();std::vector<float> first=start.floats(),last=end.floats();std::vector<uint8_t> eligible(n);
  for(uint64_t i=0;i<n;++i){
    auto flag=mask.integer(i);require(flag==0||flag==1,"noncanonical actual QA mask");eligible[i]=uint8_t(flag);
    require(bool(flag)==p["assets"][spec["context_mask"].asString()]["values"][Json::ArrayIndex(i)].asBool(),"actual QA mask differs from bound tokenizer input");
    for(unsigned j=0;j<2;++j)require(offsets.integer(2*i+j)==p["assets"][spec["offsets_utf8"].asString()]["values"][Json::ArrayIndex(i)][j].asInt64(),"actual QA offsets differ from bound tokenizer input");
  }
  mlx_qa_span selected;require(mlx_qa_select_span(first.data(),last.data(),eligible.data(),n,30,&selected)==0,"QA span selection rejected nonfinite/invalid inputs");
  Json::Value row;row["forward_id"]=spec["forward_id"];row["output_contract"]="mlx-qa-result-v1";
  for(const char *role:{"start_logits","end_logits"}){
    auto dense=data.at(role).materialize(DType::F32);auto path=directory/(std::string(role)+"-"+std::to_string(spec["forward_id"].asInt())+".f32.bin");
    std::ofstream bytes(path,std::ios::binary);bytes.write(reinterpret_cast<const char*>(dense.storage->data),dense.storage->bytes);require(bool(bytes),"cannot write QA computed output");
    auto &out=row["outputs"][role];out["value_id"]=spec[role];out["file"]=path.string();out["dtype"]="f32";out["shape"]=Json::Value(Json::arrayValue);for(auto d:dense.sizes)out["shape"].append(Json::Int64(d));out["bytes"]=Json::UInt64(dense.storage->bytes);
  }
  auto first_byte=offsets.integer(2*selected.start),last_byte=offsets.integer(2*selected.end+1);
  auto text=spec["context_utf8"].asString();require(first_byte>=0&&last_byte>=first_byte&&uint64_t(last_byte)<=text.size(),"actual QA span byte range invalid");
  row["span"]["start"]=Json::UInt64(selected.start);row["span"]["end"]=Json::UInt64(selected.end);row["span"]["score_f64"]=selected.score;
  row["span"]["text"]=text.substr(size_t(first_byte),size_t(last_byte-first_byte));row["span"]["start_byte"]=Json::Int64(first_byte);row["span"]["end_byte"]=Json::Int64(last_byte);
  auto &post=row["postprocessing"];post["entry"]="mlx_qa_select_span";post["backend"]="portable_c_host_execution_cpu_timing_unmodeled";
  post["candidate_spans"]=Json::UInt64(selected.candidates);post["max_answer_tokens"]=30;post["score_dtype"]="f64";post["tie_policy"]="first_lexicographic";
  post["metadata_readback_verified"]=true;post["actual_cpu_execution"]=false;post["target_cpu_cycles"]=Json::Value();
  return row;
}
}
