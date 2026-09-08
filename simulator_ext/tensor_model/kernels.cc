#include "tensor.h"
#include "matrix_program.h"
#include "matrix_schedule.h"
#include "vector_program.h"
#include "vector_schedule.h"
#include "memory_program.h"
#include "memory_schedule.h"
#include "control_program.h"
#include "control_schedule.h"
#include "mlx_tagged_simulator.h"
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <limits>
#include <numeric>

namespace mlx::tensor_model {
namespace {
const Tensor &ref(const Json::Value &value,const Values &values) {
  require(value.isObject() && value["value"].isString(),"expected tensor SSA reference");
  auto found=values.find(value["value"].asString());
  require(found!=values.end(),"unbound/released SSA value: "+value["value"].asString());
  return found->second;
}
int axis(int64_t value,size_t rank,bool insertion=false) {
  int limit=int(rank)+(insertion?1:0);
  if (value<0) value+=limit;
  require(value>=0 && value<limit,"tensor axis out of bounds"); return int(value);
}
uint64_t broadcast_index(uint64_t flat,const Shape &output,const Shape &input) {
  require(input.size()<=output.size(),"broadcast rank mismatch");
  uint64_t index=0, step=1;
  for (size_t dim=output.size();dim-->0;) {
    uint64_t coordinate=output[dim]?flat%uint64_t(output[dim]):0;
    if (output[dim]) flat/=output[dim];
    if (dim+input.size()>=output.size()) {
      int64_t size=input[dim+input.size()-output.size()];
      require(size==1 || size==output[dim],"broadcast extent mismatch");
      if (size!=1) index+=coordinate*step;
      step*=size;
    }
  }
  return index;
}
float get(const Json::Value &arg,const Values &values,uint64_t index,const Shape &s) {
  if (arg.isObject() && arg.isMember("value")) {
    const auto &t=ref(arg,values); return t.number(broadcast_index(index,s,t.sizes));
  }
  return float(scalar(arg));
}
int64_t get_int(const Json::Value &arg,const Values &values,uint64_t index,const Shape &s) {
  if (arg.isObject() && arg.isMember("value")) {
    const auto &t=ref(arg,values); return t.integer(broadcast_index(index,s,t.sizes));
  }
  require(arg.isInt64() || arg.isBool(),"integer arithmetic received noninteger scalar");
  return arg.isBool()?arg.asBool():arg.asInt64();
}
float pairwise_sum(std::vector<float> values) {
  while (values.size()>1) {
    size_t next=0;
    for (size_t i=0;i<values.size();i+=2) values[next++]=values[i]+(i+1<values.size()?values[i+1]:0.0f);
    values.resize(next);
  }
  return values.empty()?0.0f:values[0];
}
void check_output(const Tensor &value,const Json::Value &node) {
  require(value.sizes==shape(node["output"]["shape"]),"native output shape differs from compiled contract");
  require(dtype_name(value.type)==node["output"]["dtype"].asString(),"native output dtype differs from compiled contract");
}
} // namespace

Kernels::Kernels(const std::string &path,unsigned threads,const Json::Value &schedule_options,const Json::Value &vector_options,const Json::Value &memory_options,const Json::Value &control_options)
    :scheduled_options(schedule_options),vector_scheduled_options(vector_options),memory_scheduled_options(memory_options),control_scheduled_options(control_options) {
  if (!scheduled_options.isNull()) {
    require(path=="none","scheduled matrix execution cannot enable BLAS");
    matrix_schedule::Options::parse(scheduled_options);
  }
  if (!vector_scheduled_options.isNull()) vector_schedule::Options::parse(vector_scheduled_options);
  if (!memory_scheduled_options.isNull()) memory_model::ScheduleOptions::parse(memory_scheduled_options);
  if (!control_scheduled_options.isNull()) control_schedule::Options::parse(control_scheduled_options);
  instruction_stats=std::make_unique<MatrixInstructionStats>();
  vector_stats=std::make_unique<vector_model::Stats>();
  memory_stats=std::make_unique<memory_model::Stats>();
  control_stats=std::make_unique<control_model::Stats>();
  if (path=="none") return; // Strict microcode route may never open BLAS.
  blas_handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
  require(blas_handle!=nullptr,"cannot load explicitly selected CPU BLAS");
  gemm=reinterpret_cast<Gemm>(dlsym(blas_handle,"scipy_cblas_sgemm"));
  if (!gemm) gemm=reinterpret_cast<Gemm>(dlsym(blas_handle,"cblas_sgemm"));
  require(gemm!=nullptr,"CPU BLAS has no 32-bit CBLAS SGEMM entry");
  using SetThreads=void (*)(int);
  auto set=reinterpret_cast<SetThreads>(dlsym(blas_handle,"scipy_openblas_set_num_threads"));
  if (!set) set=reinterpret_cast<SetThreads>(dlsym(blas_handle,"openblas_set_num_threads"));
  if (set) set(int(threads));
}
Kernels::~Kernels() { if (blas_handle) dlclose(blas_handle); }
Json::Value Kernels::matrix_instruction_report() const { return instruction_stats->json(); }
Json::Value Kernels::vector_instruction_report() const { return vector_stats->json(); }
Json::Value Kernels::memory_instruction_report() const { return memory_stats->json(); }
Json::Value Kernels::control_instruction_report() const { return control_stats->json(); }

Tensor Kernels::control(const Json::Value &node,const Values &values) {
  if(control_scheduled_options.isNull())return control_model::execute(node,values,*control_stats);
  auto output=Tensor::allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));
  control_schedule::Simulator simulator(node,values,output,control_schedule::Options::parse(control_scheduled_options));
  while(simulator.tick()){}
  auto report=simulator.result();require(report["done"].asBool()&&report["dma_requests"]==report["dma_responses"],"controller cycle window did not drain");
  ++control_stats->calls;control_stats->instructions+=report["instructions"].asUInt64();control_stats->branches+=report["branches_taken"].asUInt64();
  control_stats->v2|=control_model::extended_kind(node["kind"].asString());
  control_stats->read_bytes+=report["read_bytes"].asUInt64();control_stats->write_bytes+=report["write_bytes"].asUInt64();control_stats->fflags|=report["fflags_observed"].asUInt();
  report["source_operator_id"]=node["source_operator_id"];report["forward_id"]=node["forward_id"];report["layer_idx"]=node["layer_idx"];
  control_window_reports.append(report);return output;
}

Tensor Kernels::memory(const Json::Value &node,const Values &values) {
  if(memory_scheduled_options.isNull())return memory_model::execute(node,values,*memory_stats);
  memory_model::Simulator simulator(node,values,memory_model::ScheduleOptions::parse(memory_scheduled_options));
  while(simulator.tick()){}
  auto report=simulator.result();const auto count=simulator.instruction_stats();
  memory_stats->calls+=count.calls;memory_stats->views+=count.views;memory_stats->allocations+=count.allocations;
  memory_stats->instructions+=count.instructions;memory_stats->read_bytes+=count.read_bytes;memory_stats->write_bytes+=count.write_bytes;
  memory_stats->index_reads+=count.index_reads;memory_stats->predicate_reads+=count.predicate_reads;
  for(unsigned op=1;op<6;++op)memory_stats->opcode_counts[op]+=count.opcode_counts[op];
  report["source_operator_id"]=node["source_operator_id"];report["forward_id"]=node["forward_id"];report["layer_idx"]=node["layer_idx"];
  memory_window_reports.append(report);return simulator.output();
}

Tensor Kernels::vector(const Json::Value &node,const Values &values) {
  if (vector_scheduled_options.isNull()) return vector_model::execute(node,values,*vector_stats);
  auto output=Tensor::allocate(dtype(node["output"]["dtype"].asString()),shape(node["output"]["shape"]));
  vector_schedule::Simulator simulator(node,values,output,vector_schedule::Options::parse(vector_scheduled_options));
  while(simulator.tick()){}
  auto report=simulator.result();require(report["done"].asBool()&&report["dma_requests"]==report["dma_responses"],"vector window did not drain");
  report["source_operator_id"]=node["source_operator_id"];report["forward_id"]=node["forward_id"];report["layer_idx"]=node["layer_idx"];
  const auto &count=report["numeric_instructions"];
  vector_stats->calls+=count["calls"].asUInt64();vector_stats->instructions+=count["instructions"].asUInt64();
  vector_stats->trans_lanes+=count["transcendental_lanes"].asUInt64();vector_stats->arithmetic_lanes+=count["arithmetic_lanes"].asUInt64();
  vector_stats->read_bytes+=count["global_read_bytes"].asUInt64();vector_stats->write_bytes+=count["global_write_bytes"].asUInt64();
  vector_stats->max_rom=std::max(vector_stats->max_rom,count["max_rom_words"].asUInt());vector_stats->max_stack_level=std::max(vector_stats->max_stack_level,count["max_stack_level"].asUInt());
  for(const auto &op:count["opcode_counts"].getMemberNames())vector_stats->opcode_counts[std::stoul(op)]+=count["opcode_counts"][op].asUInt64();
  vector_window_reports.append(report);return output;
}

Tensor Kernels::matrix(const Json::Value &node,const Values &values,bool linear) {
  const auto &args=node["args"];
  const auto &a=ref(args[0],values), &b=ref(args[1],values);
  require((a.type==DType::F16 || a.type==DType::F32) && a.type==b.type,"GEMM input dtype contract mismatch");
  require(a.sizes.size()>=2 && b.sizes.size()>=2,"matrix rank below two is not implemented");
  const int64_t m=linear?int64_t(elements(Shape(a.sizes.begin(),a.sizes.end()-1))):a.sizes[a.sizes.size()-2];
  const int64_t k=a.sizes.back(), n=linear?b.sizes[0]:b.sizes.back();
  require(k==(linear?b.sizes[1]:b.sizes[b.sizes.size()-2]),"GEMM contracted dimension mismatch");
  require(m<=INT32_MAX && n<=INT32_MAX && k<=INT32_MAX,"GEMM tile dimensions exceed CPU kernel interface");
  Shape batch_a(a.sizes.begin(),a.sizes.end()-2), batch_b(b.sizes.begin(),b.sizes.end()-2), batch;
  Shape out_shape;
  if (linear) {
    require(b.sizes.size()==2,"linear weight must be a matrix");
    out_shape=a.sizes; out_shape.back()=n;
  } else {
    batch.resize(std::max(batch_a.size(),batch_b.size()),1);
    for (size_t d=0;d<batch.size();++d) {
      auto sa=d+batch_a.size()>=batch.size()?batch_a[d+batch_a.size()-batch.size()]:1;
      auto sb=d+batch_b.size()>=batch.size()?batch_b[d+batch_b.size()-batch.size()]:1;
      require(sa==sb || sa==1 || sb==1,"batched GEMM broadcast mismatch"); batch[d]=std::max(sa,sb);
    }
    out_shape=batch; out_shape.push_back(m); out_shape.push_back(n);
  }
  auto out=Tensor::allocate(dtype(node["output"]["dtype"].asString()),out_shape);
  const uint64_t batches=linear?1:elements(batch);
  if (node.isMember("matrix_program")) {
    const Tensor *bias=linear && args.size()>2 && !args[2].isNull()?&ref(args[2],values):nullptr;
    for (uint64_t batch_index=0;batch_index<batches;++batch_index) {
      uint64_t ai=linear?0:broadcast_index(batch_index,batch,batch_a);
      uint64_t bi=linear?0:broadcast_index(batch_index,batch,batch_b);
      if (scheduled_options.isNull()) {
        execute_matrix_program(node["matrix_program"],a,b,bias,linear,ai,bi,m,n,k,out,batch_index,*instruction_stats);
      } else {
        matrix_schedule::Simulator simulator(node["matrix_program"],a,b,bias,linear,ai,bi,m,n,k,out,batch_index,
                                              matrix_schedule::Options::parse(scheduled_options));
        while (simulator.tick()) {}
        auto report=simulator.result();
        require(report["done"].asBool() && report["dma_requests"]==report["dma_responses"],"matrix schedule did not drain");
        report["source_operator_id"]=node["source_operator_id"];report["forward_id"]=node["forward_id"];
        report["layer_idx"]=node["layer_idx"];report["batch_index"]=Json::UInt64(batch_index);
        const auto &count=report["numeric_instructions"];
        instruction_stats->calls+=count["calls"].asUInt64();instruction_stats->tiles+=count["output_tiles"].asUInt64();
        instruction_stats->instructions+=count["instructions"].asUInt64();
        instruction_stats->inactive_row_instructions+=count["inactive_row_instructions"].asUInt64();
        instruction_stats->mul_lanes+=count["mul_active_lanes"].asUInt64();instruction_stats->add_lanes+=count["add_active_lanes"].asUInt64();
        instruction_stats->memory_read_bytes+=count["global_read_bytes"].asUInt64();instruction_stats->memory_write_bytes+=count["global_write_bytes"].asUInt64();
        instruction_stats->max_rom_words=std::max(instruction_stats->max_rom_words,count["max_rom_words"].asUInt());
        instruction_stats->max_spm_bytes=std::max(instruction_stats->max_spm_bytes,count["max_spm_bytes"].asUInt());
        for(unsigned op=1;op<14;++op)instruction_stats->opcode_counts[op]+=count["opcode_counts"][std::to_string(op)].asUInt64();
        window_reports.append(report);
      }
    }
    matrix_macs+=batches*uint64_t(m)*n*k;
    return out;
  }
  require(gemm!=nullptr,"matrix operation has no compiled instruction route and BLAS is disabled");
  auto left=a.floats(), right=b.floats();
  std::vector<float> product(uint64_t(m)*n);
  for (uint64_t batch_index=0;batch_index<batches;++batch_index) {
    uint64_t ai=linear?0:broadcast_index(batch_index,batch,batch_a);
    uint64_t bi=linear?0:broadcast_index(batch_index,batch,batch_b);
    if (m && n && k) { ++blas_calls; gemm(101,111,linear?112:111,int(m),int(n),int(k),1.0f,
        left.data()+ai*uint64_t(m)*k,int(k),right.data()+bi*uint64_t(k)*n,
        int(linear?k:n),0.0f,product.data(),int(n)); }
    else std::fill(product.begin(),product.end(),0);
    for (uint64_t i=0;i<product.size();++i) {
      float value=product[i];
      if (linear && args.size()>2 && !args[2].isNull()) value+=ref(args[2],values).number(i%uint64_t(n));
      out.set_number(batch_index*uint64_t(m)*n+i,value);
    }
  }
  matrix_macs+=batches*uint64_t(m)*n*k;
  return out;
}

Tensor Kernels::execute(const Json::Value &node,const Values &values) {
  if (node.isMember("control_program")) return control(node,values);
  if (node.isMember("memory_program")) return memory(node,values);
  if (node.isMember("vector_program")) return vector(node,values);
  if (!node.isMember("matrix_program")) ++functional_calls;
  const auto kind=node["kind"].asString(); const auto &args=node["args"];
  const DType target=dtype(node["output"]["dtype"].asString());
  const Shape declared=shape(node["output"]["shape"]);
  Tensor out;
  if (kind=="linear" || kind=="matmul") out=matrix(node,values,kind=="linear");
  else if (kind=="alias" || kind=="dropout_inference") {
    require(kind!="dropout_inference" || (args.size()==3 && args[2].isBool() && !args[2].asBool()),
            "training dropout is not an inference alias");
    out=ref(args[0],values);
  } else if (kind=="cast" || kind=="cast_device" || kind=="contiguous") {
    const auto &input=ref(args[0],values);
    const size_t copy_index=kind=="cast"?3:4;
    const bool copy=kind!="contiguous" && args.size()>copy_index && args[Json::ArrayIndex(copy_index)].asBool();
    if (!copy && input.type==target && (kind!="contiguous" || input.contiguous())) out=input;
    else { out=input.materialize(target); materialized_bytes+=out.numel()*element_bytes(target); }
  } else if (kind=="unsqueeze") {
    out=ref(args[0],values); int dim=axis(args[1].asInt64(),out.sizes.size(),true);
    int64_t stride=size_t(dim)<out.sizes.size()?out.sizes[dim]*out.steps[dim]:1;
    out.sizes.insert(out.sizes.begin()+dim,1); out.steps.insert(out.steps.begin()+dim,stride);
  } else if (kind=="transpose") {
    out=ref(args[0],values); int a=axis(args[1].asInt64(),out.sizes.size()), b=axis(args[2].asInt64(),out.sizes.size());
    std::swap(out.sizes[a],out.sizes[b]); std::swap(out.steps[a],out.steps[b]);
  } else if (kind=="reshape") {
    out=ref(args[0],values); Shape requested=shape(args[1]); int inferred=-1; uint64_t known=1;
    for (size_t i=0;i<requested.size();++i) {
      if (requested[i]==-1) { require(inferred<0,"multiple inferred reshape axes"); inferred=int(i); }
      else { require(requested[i]>=0,"invalid reshape extent"); known*=requested[i]; }
    }
    if (inferred>=0) { require(known && out.numel()%known==0,"cannot infer reshape extent"); requested[inferred]=out.numel()/known; }
    require(elements(requested)==out.numel(),"reshape changes element count");
    if (!out.contiguous()) { out=out.materialize(out.type); materialized_bytes+=out.numel()*element_bytes(out.type); }
    out.sizes=requested; out.steps=strides(requested);
  } else if (kind=="expand") {
    out=ref(args[0],values); auto requested=shape(args[1]);
    require(requested.size()>=out.sizes.size(),"expand reduced rank");
    const size_t extra=requested.size()-out.sizes.size();
    out.sizes.insert(out.sizes.begin(),extra,1); out.steps.insert(out.steps.begin(),extra,0);
    for (size_t i=0;i<requested.size();++i) {
      if (requested[i]==-1) { require(i>=extra,"new expand axis cannot be inferred"); requested[i]=out.sizes[i]; }
      require(requested[i]>=0 && (out.sizes[i]==1 || out.sizes[i]==requested[i]),"expand extent mismatch");
      if (out.sizes[i]!=requested[i]) out.steps[i]=0;
    }
    out.sizes=requested;
  } else if (kind=="slice") {
    out=ref(args[0],values); int dim=axis(args.size()>1?args[1].asInt64():0,out.sizes.size());
    int64_t n=out.sizes[dim], begin=args.size()>2&&!args[2].isNull()?args[2].asInt64():0;
    int64_t end=args.size()>3&&!args[3].isNull()?args[3].asInt64():n;
    int64_t step=args.size()>4?args[4].asInt64():1; require(step>0,"nonpositive slice step");
    if (begin<0) begin+=n;
    if (end<0) end+=n;
    begin=std::clamp<int64_t>(begin,0,n); end=std::clamp<int64_t>(end,0,n);
    out.offset+=begin*out.steps[dim]; out.sizes[dim]=end>begin?(end-begin+step-1)/step:0; out.steps[dim]*=step;
  } else if (kind=="select") {
    out=ref(args[0],values); int dim=axis(args[1].asInt64(),out.sizes.size()); int64_t index=args[2].asInt64();
    if (index<0) index+=out.sizes[dim];
    require(index>=0 && index<out.sizes[dim],"select index out of bounds");
    out.offset+=index*out.steps[dim]; out.sizes.erase(out.sizes.begin()+dim); out.steps.erase(out.steps.begin()+dim);
  } else if (kind=="embedding") {
    const auto &weight=ref(args[0],values), &ids=ref(args[1],values);
    require(weight.sizes.size()==2 && ids.type==DType::I64,"embedding shape/index type mismatch");
    Shape s=ids.sizes; s.push_back(weight.sizes[1]); out=Tensor::allocate(weight.type,s);
    for (uint64_t i=0;i<ids.numel();++i) {
      auto row=ids.integer(i); require(row>=0 && row<weight.sizes[0],"embedding index out of range");
      for (int64_t column=0;column<weight.sizes[1];++column)
        out.set_number(i*weight.sizes[1]+column,weight.number(row*weight.sizes[1]+column));
    }
  } else if (kind=="arange") {
    require(args.size()==1,"only zero-start unit-step arange is implemented");
    int64_t end=args[0].asInt64(); require(end>=0,"negative arange end"); out=Tensor::allocate(target,{end});
    for (int64_t i=0;i<end;++i) out.set_integer(i,i);
  } else if (kind=="cat") {
    std::vector<Tensor> inputs;
    for (const auto &arg:args[0]) if (ref(arg,values).numel()) inputs.push_back(ref(arg,values));
    if (inputs.empty()) out=Tensor::allocate(target,declared);
    else {
      int dim=axis(args.size()>1?args[1].asInt64():0,inputs[0].sizes.size()); Shape s=inputs[0].sizes; s[dim]=0;
      for (const auto &input:inputs) {
        require(input.type==target && input.sizes.size()==s.size(),"cat dtype/rank mismatch");
        for (size_t d=0;d<s.size();++d) if (int(d)!=dim) require(s[d]==input.sizes[d],"cat extent mismatch");
        s[dim]+=input.sizes[dim];
      }
      out=Tensor::allocate(target,s); uint64_t inner=1, outer=1;
      for (size_t d=0;d<s.size();++d) { if (int(d)<dim) outer*=s[d]; else if (int(d)>dim) inner*=s[d]; }
      for (uint64_t o=0;o<outer;++o) {
        uint64_t start=0;
        for (const auto &input:inputs) {
          uint64_t count=input.sizes[dim]*inner;
          for (uint64_t i=0;i<count;++i) {
            uint64_t dest=o*uint64_t(s[dim])*inner+start+i;
            if (target==DType::I64 || target==DType::Bool) out.set_integer(dest,input.integer(o*count+i));
            else out.set_number(dest,input.number(o*count+i));
          }
          start+=count;
        }
      }
      materialized_bytes+=out.numel()*element_bytes(target);
    }
  } else if (kind=="mean" || kind=="softmax" || kind=="argmax") {
    const auto &input=ref(args[0],values);
    int dim;
    if (kind=="mean") { require(args[1].isArray() && args[1].size()==1,"multi-axis reduction not implemented"); dim=axis(args[1][0].asInt64(),input.sizes.size()); }
    else dim=axis(args[1].asInt64(),input.sizes.size());
    require(size_t(dim)+1==input.sizes.size(),"this reduction entry requires the last axis");
    uint64_t width=input.sizes.back(); require(width>0,"empty reduction");
    Shape s=input.sizes;
    if (kind!="softmax") { bool keep=kind=="mean"?args.size()>2&&args[2].asBool():args.size()>2&&args[2].asBool(); if (keep) s.back()=1; else s.pop_back(); }
    out=Tensor::allocate(target,s);
    std::vector<float> row(width);
    for (uint64_t outer=0;outer<input.numel()/width;++outer) {
      for (uint64_t i=0;i<width;++i) {
        float value=input.number(outer*width+i);
        // softmax(dtype=FP16) narrows its input BEFORE exponentiation, not
        // merely the final probabilities. The vector lowering emits CVTs.
        if (kind=="softmax" && input.type==DType::F32 && target==DType::F16)
          value=tagged::half_to_float(tagged::float_to_half(value));
        row[i]=value;
      }
      if (kind=="mean") out.set_number(outer,pairwise_sum(row)/float(width));
      else if (kind=="argmax") {
        uint64_t best=0;
        for (uint64_t i=1;i<width;++i) {
          if (input.type==DType::I64) {
            if (input.integer(outer*width+i)>input.integer(outer*width+best)) best=i;
          } else if ((std::isnan(row[i]) && !std::isnan(row[best])) || row[i]>row[best]) best=i;
        }
        out.set_integer(outer,best);
      } else {
        float maximum=*std::max_element(row.begin(),row.end());
        for (auto &value:row) value=std::exp(value-maximum);
        float sum=pairwise_sum(row);
        for (uint64_t i=0;i<width;++i) out.set_number(outer*width+i,row[i]/sum);
      }
    }
  } else {
    const std::vector<std::string> legal={"add","mul","le","where","pow","rsqrt","silu","cos","sin","neg"};
    require(std::find(legal.begin(),legal.end(),kind)!=legal.end(),"unimplemented native tensor kernel: "+kind);
    out=Tensor::allocate(target,declared);
    if (kind=="where") require(ref(args[0],values).type==DType::Bool,"where predicate is not Boolean");
    const bool integer_compare=kind=="le" && ref(args[0],values).type==DType::I64;
    for (uint64_t i=0;i<out.numel();++i) {
      if (kind=="where" && target==DType::I64) {
        out.set_integer(i,get_int(args[get_int(args[0],values,i,declared)?1:2],values,i,declared)); continue;
      }
      if ((target==DType::I64 || integer_compare) && kind!="where") {
        int64_t a=get_int(args[0],values,i,declared), b=get_int(args[1],values,i,declared);
        require(!node["kwargs"].isMember("alpha") || scalar(node["kwargs"]["alpha"])==1,"integer alpha is not unit");
        if (kind=="le") out.set_integer(i,a<=b);
        else if (kind=="add") out.set_integer(i,int64_t(uint64_t(a)+uint64_t(b)));
        else if (kind=="mul") out.set_integer(i,int64_t(uint64_t(a)*uint64_t(b)));
        else throw std::runtime_error("unsupported integer operation");
        continue;
      }
      float a=get(args[0],values,i,declared), value=0;
      if (kind=="add") value=a+get(args[1],values,i,declared)*(node["kwargs"].isMember("alpha")?float(scalar(node["kwargs"]["alpha"])):1.0f);
      else if (kind=="mul") value=a*get(args[1],values,i,declared);
      else if (kind=="le") value=a<=get(args[1],values,i,declared);
      else if (kind=="where") value=get(args[a!=0?1:2],values,i,declared);
      else if (kind=="pow") value=scalar(args[1])==2?a*a:std::pow(a,float(scalar(args[1])));
      else if (kind=="rsqrt") value=1.0f/std::sqrt(a);
      else if (kind=="silu") value=a/(1.0f+std::exp(-a));
      else if (kind=="cos") value=std::cos(a);
      else if (kind=="sin") value=std::sin(a);
      else if (kind=="neg") value=-a;
      out.set_number(i,value);
    }
  }
  check_output(out,node); return out;
}
} // namespace mlx::tensor_model
